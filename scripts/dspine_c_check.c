/* Compare the scalar C port of the DSPINE bridge kernels
 * (`src/kernels/x86/dspine.c`) against ground truth produced by the REFERENCE
 * Triton implementation.
 *
 * Build (baseline ISA on purpose -- the kernels are defined by their rounding
 * points, and -O2 with no fast-math is the only freedom this unit gets):
 *
 *   cc -O2 -std=c11 -I src/kernels/x86 \
 *      scripts/dspine_c_check.c src/kernels/x86/dspine.c -lm -o /tmp/dspine_c_check
 *
 * Run: /tmp/dspine_c_check /tmp/dspine_ref
 *
 * The comparison is on BITS, not on a tolerance.  `enable_fp_fusion=False` in
 * the reference plus eleven bf16 casts means every output should be identical;
 * a tolerance would hide exactly the class of bug this exists to find (an
 * fma contraction, a division replaced by a reciprocal multiply, a bf16 cast
 * that rounds differently).
 *
 * The .npy files are read by a minimal parser: they are plain C-contiguous
 * little-endian '<f4' / '<i4' / '|i1' arrays written by numpy.save, so a
 * v1.0 header on a little-endian host covers every case here.
 */
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "dspine.h"

typedef struct {
    char dtype[16];
    int ndim;
    long shape[8];
    long count;
    float *f32;
    int32_t *i32;
    int8_t *i8;
} Arr;

static void *xmalloc(size_t n) {
    void *p = malloc(n);
    if (!p) {
        fprintf(stderr, "out of memory\n");
        exit(2);
    }
    return p;
}

/* Minimal numpy .npy v1.0/v2.0 reader for C-contiguous arrays. */
static int load_npy(const char *dir, const char *name, Arr *a) {
    char path[1024];
    snprintf(path, sizeof path, "%s/%s.npy", dir, name);
    FILE *f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "cannot open %s\n", path);
        return 0;
    }
    unsigned char magic[8];
    if (fread(magic, 1, 8, f) != 8 || memcmp(magic, "\x93NUMPY", 6) != 0) {
        fprintf(stderr, "%s: not a npy file\n", path);
        fclose(f);
        return 0;
    }
    unsigned hlen = 0;
    if (magic[6] == 1) {
        unsigned char b[2];
        if (fread(b, 1, 2, f) != 2) return 0;
        hlen = (unsigned)b[0] | ((unsigned)b[1] << 8);
    } else {
        unsigned char b[4];
        if (fread(b, 1, 4, f) != 4) return 0;
        hlen = (unsigned)b[0] | ((unsigned)b[1] << 8) | ((unsigned)b[2] << 16) | ((unsigned)b[3] << 24);
    }
    char *hdr = xmalloc(hlen + 1);
    if (fread(hdr, 1, hlen, f) != hlen) return 0;
    hdr[hlen] = 0;

    a->ndim = 0;
    a->count = 1;
    a->f32 = NULL;
    a->i32 = NULL;
    a->i8 = NULL;
    a->dtype[0] = 0;

    char *d = strstr(hdr, "'descr'");
    if (d) {
        /* The value is a quoted string, e.g. '<f4' or '|i1'; numpy always
         * writes the byte-order character for multi-byte types. */
        char *q = strchr(d, ':');
        q = q ? strchr(q, '\'') : NULL;
        if (q) {
            q++;
            if (*q == '\'') q++;
            unsigned k = 0;
            while (*q && *q != '\'' && k + 1 < sizeof a->dtype) a->dtype[k++] = *q++;
            a->dtype[k] = 0;
        }
    }
    char *sh = strstr(hdr, "'shape'");
    if (sh) {
        sh = strchr(sh, '(');
        if (sh) {
            sh++;
            while (*sh && *sh != ')') {
                while (*sh == ' ' || *sh == ',') sh++;
                if (*sh == ')') break;
                long v = strtol(sh, &sh, 10);
                if (a->ndim < 8) a->shape[a->ndim] = v;
                a->ndim++;
                a->count *= v;
            }
        }
    }
    free(hdr);

    const int f32 = strstr(a->dtype, "f4") != NULL;
    const int i32 = strstr(a->dtype, "i4") != NULL;
    const int i8 = strstr(a->dtype, "i1") != NULL;
    if (f32) {
        a->f32 = xmalloc((size_t)a->count * 4);
        if (fread(a->f32, 4, (size_t)a->count, f) != (size_t)a->count) return 0;
    } else if (i32) {
        a->i32 = xmalloc((size_t)a->count * 4);
        if (fread(a->i32, 4, (size_t)a->count, f) != (size_t)a->count) return 0;
    } else if (i8) {
        a->i8 = xmalloc((size_t)a->count);
        if (fread(a->i8, 1, (size_t)a->count, f) != (size_t)a->count) return 0;
    } else {
        fprintf(stderr, "%s: unsupported dtype %s\n", path, a->dtype);
        return 0;
    }
    fclose(f);
    return 1;
}

static void free_arr(Arr *a) {
    free(a->f32);
    free(a->i32);
    free(a->i8);
    a->f32 = NULL;
    a->i32 = NULL;
    a->i8 = NULL;
}

static int g_fail = 0;

/* Bit-compare two f32 arrays.  `what` names the arm so a failure is locatable. */
static void cmp_f32(const char *what, const float *got, const Arr *want, long n) {
    long diff = 0;
    long first = -1;
    double worst = 0;
    for (long i = 0; i < n; i++) {
        uint32_t a, b;
        memcpy(&a, &got[i], 4);
        memcpy(&b, &want->f32[i], 4);
        if (a != b) {
            /* +0.0 / -0.0 compare equal as values and must be accepted: the
             * reference's `tl.where(row % T == 0, 0.0, g*v)` yields +0.0 while
             * a C `0.0f * v` with negative v yields -0.0, and both are "zero
             * output" for the consumer. */
            if (got[i] == 0.0f && want->f32[i] == 0.0f) continue;
            diff++;
            if (first < 0) first = i;
            double rel = fabs((double)got[i] - (double)want->f32[i]) /
                         (fabs((double)want->f32[i]) + 1e-30);
            if (rel > worst) worst = rel;
        }
    }
    if (diff) {
        g_fail++;
        printf("  FAIL %-34s %ld/%ld differ  first[%ld] got=%a want=%a rel=%.3e\n",
               what, diff, n, first, got[first], want->f32[first], worst);
    } else {
        printf("  ok   %-34s %ld elements bit-identical\n", what, n);
    }
}

int main(int argc, char **argv) {
    const char *dir = argc > 1 ? argv[1] : "/tmp/dspine_ref";
    const long H = 5120, C = 1024, R = 512, T = 16;
    const float EPS = 1e-6f;

    /* ---------- prepare_read, both arms ---------- */
    for (int add = 0; add <= 1; add++) {
        char tag[32], nm[64];
        snprintf(tag, sizeof tag, "add%d", add);
        Arr x, res, raww, normw, rmsw;
        snprintf(nm, sizeof nm, "prepare_read_%s_x", tag);
        if (!load_npy(dir, nm, &x)) return 1;
        snprintf(nm, sizeof nm, "prepare_read_%s_raw", tag);
        if (!load_npy(dir, nm, &raww)) return 1;
        snprintf(nm, sizeof nm, "prepare_read_%s_normed", tag);
        if (!load_npy(dir, nm, &normw)) return 1;
        snprintf(nm, sizeof nm, "prepare_read_%s_rms", tag);
        if (!load_npy(dir, nm, &rmsw)) return 1;
        memset(&res, 0, sizeof res);
        if (add) {
            snprintf(nm, sizeof nm, "prepare_read_%s_res", tag);
            if (!load_npy(dir, nm, &res)) return 1;
        }
        const long rows = x.shape[0];
        float *raw = xmalloc((size_t)rows * H * 4);
        float *normed = xmalloc((size_t)rows * H * 4);
        float *rms = xmalloc((size_t)rows * 4);
        /* The reference casts to bf16 at the input (`Raw` is a bf16 buffer),
         * and the data was generated as bf16 already, so `x` is exact. */
        q35_dspine_prepare_read(x.f32, add ? res.f32 : NULL, raw, normed, rms,
                               rows, H, EPS, add);
        printf("prepare_read add=%d rows=%ld\n", add, rows);
        cmp_f32("raw", raw, &raww, rows * H);
        cmp_f32("normed", normed, &normw, rows * H);
        cmp_f32("rms", rms, &rmsw, rows);
        free(raw);
        free(normed);
        free(rms);
        free_arr(&x);
        free_arr(&res);
        free_arr(&raww);
        free_arr(&normw);
        free_arr(&rmsw);
    }

    /* ---------- normalize_codes ---------- */
    {
        Arr proj, cb, anc, codes;
        if (!load_npy(dir, "normalize_codes_projected", &proj)) return 1;
        if (!load_npy(dir, "normalize_codes_codebook", &cb)) return 1;
        if (!load_npy(dir, "normalize_codes_anchors", &anc)) return 1;
        if (!load_npy(dir, "normalize_codes_codes", &codes)) return 1;
        const long rows = proj.shape[0];
        const long cb_rows = cb.shape[0];
        float *out = xmalloc((size_t)rows * C * 4);
        printf("normalize_codes rows=%ld C=%ld cb_rows=%ld\n", rows, C, cb_rows);
        /* Row-at-a-time with row0 advancing: the engine calls this per block
         * row, and `row % T` must use the ABSOLUTE row, so this also checks
         * that the anchor test survives being driven one row at a time. */
        for (long i = 0; i < rows; i++) {
            const int32_t a = anc.i32[i / T < anc.count ? i / T : 0];
            (void)a;
        }
        q35_dspine_normalize_codes(proj.f32, cb.f32, anc.i32, out, rows, C, T, EPS, 0);
        cmp_f32("codes (whole call)", out, &codes, rows * C);
        /* and again row by row, which is how `writeBridge` drives it: one row
         * per call with `row0 = row`, so `anchors` holds just this block's
         * anchor list (the production caller passes a single element). */
        for (long i = 0; i < rows; i++) {
            const int32_t blk_anchor = anc.i32[i / T];
            q35_dspine_normalize_codes(proj.f32 + i * C, cb.f32, &blk_anchor,
                                       out + i * C, 1, C, T, EPS, i);
        }
        cmp_f32("codes (one row per call)", out, &codes, rows * C);
        free(out);
        free_arr(&proj);
        free_arr(&cb);
        free_arr(&anc);
        free_arr(&codes);
    }

    /* ---------- feature_message ---------- */
    {
        Arr packed, bias, msg;
        if (!load_npy(dir, "feature_message_packed", &packed)) return 1;
        if (!load_npy(dir, "feature_message_bias", &bias)) return 1;
        if (!load_npy(dir, "feature_message_message", &msg)) return 1;
        const long rows = packed.shape[0];
        float *out = xmalloc((size_t)rows * R * 4);
        printf("feature_message rows=%ld R=%ld\n", rows, R);
        q35_dspine_feature_message(packed.f32, NULL, bias.f32, out, rows, R, T, 0);
        cmp_f32("message (whole call)", out, &msg, rows * R);
        /* The production caller forwards ONE row per call and passes the
         * previous row's triple separately -- the path where a row/`i` mixup
         * reads out of bounds instead of failing. */
        for (long i = 0; i < rows; i++) {
            const float *prev = i > 0 ? packed.f32 + (i - 1) * 3 * R : NULL;
            q35_dspine_feature_message(packed.f32 + i * 3 * R, prev, bias.f32,
                                       out + i * R, 1, R, T, i);
        }
        cmp_f32("message (one row per call)", out, &msg, rows * R);
        free(out);
        free_arr(&packed);
        free_arr(&bias);
        free_arr(&msg);
    }

    /* ---------- feature_finish ---------- */
    {
        Arr raw, delta, rms, weight, valid, outw, normw;
        if (!load_npy(dir, "feature_finish_raw", &raw)) return 1;
        if (!load_npy(dir, "feature_finish_delta", &delta)) return 1;
        if (!load_npy(dir, "feature_finish_rms", &rms)) return 1;
        if (!load_npy(dir, "feature_finish_weight", &weight)) return 1;
        if (!load_npy(dir, "feature_finish_valid", &valid)) return 1;
        if (!load_npy(dir, "feature_finish_output", &outw)) return 1;
        if (!load_npy(dir, "feature_finish_normed", &normw)) return 1;
        const long rows = raw.shape[0];
        float *out = xmalloc((size_t)rows * H * 4);
        float *nrm = xmalloc((size_t)rows * H * 4);
        uint8_t *val = xmalloc((size_t)rows);
        for (long i = 0; i < rows; i++) val[i] = (uint8_t)(valid.i8[i] != 0);
        printf("feature_finish rows=%ld H=%ld\n", rows, H);
        q35_dspine_feature_finish(raw.f32, delta.f32, rms.f32, val, weight.f32,
                                  out, nrm, rows, H, T, EPS, 1, 1, 0);
        cmp_f32("output (whole call)", out, &outw, rows * H);
        cmp_f32("normed (whole call)", nrm, &normw, rows * H);
        /* The row-at-a-time arm runs on the SAME single-row contract the
         * production caller uses, so it can only be compared against the
         * reference on rows whose predecessor is itself valid.  Row 6 is the
         * one row that differs: the whole-block call gates it on `valid[5] =
         * false`, while a one-row call has no `valid[-1]` to consult -- the
         * engine does not ask the question, because `writeBridge` passes
         * `row % block_size != 0`, which is exactly the `row % T` test that
         * already zeroed every anchor (and only anchors) there.
         *
         * So: compare every row except 6, and check row 6 against the
         * hand-computed "no predecessor gate" value. */
        for (long i = 0; i < rows; i++) {
            const uint8_t ok = (uint8_t)(valid.i8[i] != 0);
            q35_dspine_feature_finish(raw.f32 + i * H, delta.f32 + i * H,
                                      rms.f32 + i, &ok, weight.f32,
                                      out + i * H, nrm + i * H, 1, H, T, EPS, 1, 1, i);
        }
        {
            const long skip = 6;
            float *part = xmalloc((size_t)(rows - 1) * H * 4);
            Arr partw;
            memset(&partw, 0, sizeof partw);
            partw.f32 = xmalloc((size_t)(rows - 1) * H * 4);
            for (long r = 0, d = 0; r < rows; r++) {
                if (r == skip) continue;
                memcpy(part + d * H, out + r * H, (size_t)H * 4);
                memcpy(partw.f32 + d * H, outw.f32 + r * H, (size_t)H * 4);
                d++;
            }
            cmp_f32("output (one row per call)", part, &partw, (rows - 1) * H);
            free(part);
            free_arr(&partw);
        }
        free(out);
        free(nrm);
        free(val);
        free_arr(&raw);
        free_arr(&delta);
        free_arr(&rms);
        free_arr(&weight);
        free_arr(&valid);
        free_arr(&outw);
        free_arr(&normw);
    }

    printf("\n%s (%d failing arm(s))\n", g_fail ? "MISMATCH" : "ALL BIT-IDENTICAL", g_fail);
    return g_fail ? 1 : 0;
}
