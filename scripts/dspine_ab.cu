/* A/B the CUDA bridge kernels (`src/kernels/cuda/dspine.cu`) against the
 * scalar C port (`src/kernels/x86/dspine.c`).
 *
 * WHY THIS IS THE ACCEPTANCE TEST
 * Both sides were transcribed from the reference independently, and each keeps
 * the reference's bf16 cast points and expression order.  Neither matches
 * Triton's reduction order (see the header of dspine.cu), so the reference is
 * NOT the right oracle for the reduction: it is a moving target across Triton
 * releases.  What must hold is that OUR two backends agree -- otherwise
 * "run the draft on the GPU" would silently change the model's behaviour
 * depending on which backend happened to run.
 *
 * Build:
 *   nvcc -arch=sm_89 -O3 -std=c++17 -I src/kernels/cuda \
 *     scripts/dspine_ab.cu src/kernels/cuda/dspine.cu \
 *     src/kernels/x86/dspine.c -o /tmp/dspine_ab
 * (dspine.c is plain C11 but nvcc compiles it as C++ happily: it has no
 *  designated initialisers or C-only constructs.)
 *
 * Run: /tmp/dspine_ab
 */
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

extern "C" {
void q35_dspine_prepare_read(const float *x, const float *residual, float *raw,
                             float *normed, float *rms_out, long rows, long h,
                             float eps, int add);
void q35_dspine_normalize_codes(const float *projected, const float *codebook,
                                const int32_t *anchors, float *codes, long rows,
                                long c, long t, float eps, long row0);
void q35_dspine_feature_message(const float *packed, const float *packed_prev,
                                const float *bias, float *message, long rows,
                                long r, long t, long row0);
void q35_dspine_feature_finish(const float *raw, const float *delta,
                               const float *rms, const uint8_t *valid,
                               const float *weight, float *output,
                               float *normed, long rows, long h, long t,
                               float norm_eps, int write, int norm, long row0);

int q35_launch_dspine_prepare_read(const float *d_x, const float *d_residual,
                                   float *d_raw, float *d_normed, float *d_rms,
                                   int rows, int h, float eps, int add, void *stream);
int q35_launch_dspine_normalize_codes(const float *d_projected, const float *d_codebook,
                                      const int32_t *d_anchors, float *d_codes, int rows,
                                      int c, int t, float eps, int row0, void *stream);
int q35_launch_dspine_feature_message(const float *d_packed, const float *d_packed_prev,
                                      const float *d_bias, float *d_message, int rows, int r,
                                      int t, int row0, void *stream);
int q35_launch_dspine_feature_finish(const float *d_raw, const float *d_delta,
                                     const float *d_rms, const uint8_t *d_valid,
                                     const float *d_weight, float *d_output, float *d_normed,
                                     int rows, int h, int t, float norm_eps, int write,
                                     int norm, int row0, void *stream);
}

#define CK(x)                                                                        \
    do {                                                                             \
        cudaError_t e_ = (x);                                                        \
        if (e_ != cudaSuccess) {                                                      \
            printf("CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); \
            exit(1);                                                                  \
        }                                                                            \
    } while (0)

static int g_fail = 0;

static void cmp(const char *what, const float *got, const float *want, size_t n) {
    size_t diff = 0, first = (size_t)-1;
    double worst = 0;
    for (size_t i = 0; i < n; i++) {
        uint32_t a, b;
        memcpy(&a, &got[i], 4);
        memcpy(&b, &want[i], 4);
        if (a == b) continue;
        if (got[i] == 0.0f && want[i] == 0.0f) continue;  // +0 vs -0
        diff++;
        if (first == (size_t)-1) first = i;
        double rel = fabs((double)got[i] - (double)want[i]) / (fabs((double)want[i]) + 1e-30);
        if (rel > worst) worst = rel;
    }
    if (diff) {
        g_fail++;
        printf("  FAIL %-32s %zu/%zu differ first[%zu] cuda=%a cpu=%a rel=%.3e\n", what, diff, n,
               first, got[first], want[first], worst);
    } else {
        printf("  ok   %-32s %zu elements bit-identical (CUDA == C)\n", what, n);
    }
}

/* Deterministic, bf16-representable inputs: the kernels narrow to bf16 anyway,
 * and starting from bf16 removes the "two different roundings of the same f32"
 * question from the comparison. */
static void fill(float *p, size_t n, unsigned seed, float scale) {
    for (size_t i = 0; i < n; i++) {
        seed = seed * 1664525u + 1013904223u;
        const int32_t v = (int32_t)(seed >> 8) - (1 << 23);
        float f = (float)v / (float)(1 << 23) * scale;
        p[i] = __bfloat162float(__float2bfloat16_rn(f));
    }
}

int main(void) {
    const long H = 5120, C = 1024, R = 512, T = 16;
    const float EPS = 1e-6f;
    const int rows = 18;

    /* ---------- prepare_read, both ADD arms ---------- */
    for (int add = 0; add <= 1; add++) {
        const size_t n = (size_t)rows * H;
        float *x = (float *)malloc(n * 4), *res = (float *)malloc(n * 4);
        float *craw = (float *)malloc(n * 4), *cnrm = (float *)malloc(n * 4);
        float *crms = (float *)malloc((size_t)rows * 4);
        fill(x, n, 1u, 1.7f);
        fill(res, n, 2u, 0.9f);

        q35_dspine_prepare_read(x, add ? res : NULL, craw, cnrm, crms, rows, H, EPS, add);

        float *dx, *dres, *draw, *dnrm, *drms;
        CK(cudaMalloc(&dx, n * 4));
        CK(cudaMalloc(&dres, n * 4));
        CK(cudaMalloc(&draw, n * 4));
        CK(cudaMalloc(&dnrm, n * 4));
        CK(cudaMalloc(&drms, (size_t)rows * 4));
        CK(cudaMemcpy(dx, x, n * 4, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dres, res, n * 4, cudaMemcpyHostToDevice));
        int rc = q35_launch_dspine_prepare_read(dx, add ? dres : NULL, draw, dnrm, drms, rows,
                                                (int)H, EPS, add, 0);
        if (rc) { printf("launch prepare_read failed %d\n", rc); return 1; }
        CK(cudaDeviceSynchronize());
        float *graw = (float *)malloc(n * 4), *gnrm = (float *)malloc(n * 4);
        float *grms = (float *)malloc((size_t)rows * 4);
        CK(cudaMemcpy(graw, draw, n * 4, cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(gnrm, dnrm, n * 4, cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(grms, drms, (size_t)rows * 4, cudaMemcpyDeviceToHost));

        printf("prepare_read add=%d\n", add);
        cmp("raw", graw, craw, n);
        cmp("rms", grms, crms, (size_t)rows);
        cmp("normed", gnrm, cnrm, n);
        cudaFree(dx); cudaFree(dres); cudaFree(draw); cudaFree(dnrm); cudaFree(drms);
        free(x); free(res); free(craw); free(cnrm); free(crms);
        free(graw); free(gnrm); free(grms);
    }

    /* ---------- normalize_codes ---------- */
    {
        const size_t np = (size_t)rows * C, nc = (size_t)4096 * C;
        float *proj = (float *)malloc(np * 4), *cb = (float *)malloc(nc * 4);
        int32_t *anch = (int32_t *)malloc((size_t)rows * 4);
        float *ccodes = (float *)malloc(np * 4);
        fill(proj, np, 3u, 1.0f);
        fill(cb, nc, 4u, 0.5f);
        for (long i = 0; i < rows; i++) anch[i] = (int32_t)((i * 977) % 4096);

        q35_dspine_normalize_codes(proj, cb, anch, ccodes, rows, C, T, EPS, 0);

        float *dp, *dcb, *dco;
        int32_t *da;
        CK(cudaMalloc(&dp, np * 4));
        CK(cudaMalloc(&dcb, nc * 4));
        CK(cudaMalloc(&da, (size_t)rows * 4));
        CK(cudaMalloc(&dco, np * 4));
        CK(cudaMemcpy(dp, proj, np * 4, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dcb, cb, nc * 4, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(da, anch, (size_t)rows * 4, cudaMemcpyHostToDevice));
        int rc = q35_launch_dspine_normalize_codes(dp, dcb, da, dco, rows, (int)C, (int)T, EPS,
                                                   0, 0);
        if (rc) { printf("launch normalize_codes failed %d\n", rc); return 1; }
        CK(cudaDeviceSynchronize());
        float *g = (float *)malloc(np * 4);
        CK(cudaMemcpy(g, dco, np * 4, cudaMemcpyDeviceToHost));
        printf("normalize_codes rows=%d (2 anchor rows)\n", rows);
        cmp("codes", g, ccodes, np);
        cudaFree(dp); cudaFree(dcb); cudaFree(da); cudaFree(dco);
        free(proj); free(cb); free(anch); free(ccodes); free(g);
    }

    /* ---------- feature_message ---------- */
    {
        const size_t nk = (size_t)rows * 3 * R, nm = (size_t)rows * R;
        float *packed = (float *)malloc(nk * 4), *bias = (float *)malloc((size_t)R * 4);
        float *cmsg = (float *)malloc(nm * 4);
        fill(packed, nk, 5u, 1.3f);
        fill(bias, (size_t)R, 6u, 0.7f);
        q35_dspine_feature_message(packed, NULL, bias, cmsg, rows, R, T, 0);

        float *dpk, *db, *dmsg;
        CK(cudaMalloc(&dpk, nk * 4));
        CK(cudaMalloc(&db, (size_t)R * 4));
        CK(cudaMalloc(&dmsg, nm * 4));
        CK(cudaMemcpy(dpk, packed, nk * 4, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(db, bias, (size_t)R * 4, cudaMemcpyHostToDevice));
        /* The whole block in one call: no separate predecessor (the kernel takes
         * the previous row out of `packed`, exactly like the C port). */
        int rc = q35_launch_dspine_feature_message(dpk, NULL, db, dmsg, rows, (int)R, (int)T, 0, 0);
        if (rc) { printf("launch feature_message failed %d\n", rc); return 1; }
        CK(cudaDeviceSynchronize());
        float *g = (float *)malloc(nm * 4);
        CK(cudaMemcpy(g, dmsg, nm * 4, cudaMemcpyDeviceToHost));
        printf("feature_message rows=%d R=%ld\n", rows, R);
        cmp("message", g, cmsg, nm);
        /* One row per call, carrying the predecessor explicitly -- the shape the
         * draft actually uses.  A port that reads the sender out of `packed`
         * instead scores its own row here and produces all-zero messages. */
        for (long i = 0; i < rows; i++) {
            const float *prev = i > 0 ? packed + (i - 1) * 3 * R : NULL;
            q35_dspine_feature_message(packed + i * 3 * R, prev, bias, cmsg + i * R,
                                       1, R, T, i);
            const float *dprev = NULL;
            float *dprev_buf = NULL;
            if (i > 0) {
                CK(cudaMalloc(&dprev_buf, (size_t)3 * R * 4));
                CK(cudaMemcpy(dprev_buf, packed + (i - 1) * 3 * R, (size_t)3 * R * 4,
                              cudaMemcpyHostToDevice));
                dprev = dprev_buf;
            }
            float *dcur = NULL;
            CK(cudaMalloc(&dcur, (size_t)3 * R * 4));
            CK(cudaMemcpy(dcur, packed + i * 3 * R, (size_t)3 * R * 4, cudaMemcpyHostToDevice));
            rc = q35_launch_dspine_feature_message(dcur, dprev, db, dmsg, 1, (int)R, (int)T,
                                                   (int)i, 0);
            if (rc) { printf("launch feature_message(1 row) failed %d\n", rc); return 1; }
            CK(cudaDeviceSynchronize());
            CK(cudaMemcpy(g + i * R, dmsg, (size_t)R * 4, cudaMemcpyDeviceToHost));
            cudaFree(dcur);
            if (dprev_buf) cudaFree(dprev_buf);
        }
        cmp("message (one row per call)", g, cmsg, nm);
        cudaFree(dpk); cudaFree(db); cudaFree(dmsg);
        free(packed); free(bias); free(cmsg); free(g);
    }

    /* ---------- feature_finish ---------- */
    {
        const size_t n = (size_t)rows * H;
        float *raw = (float *)malloc(n * 4), *del = (float *)malloc(n * 4);
        float *rms = (float *)malloc((size_t)rows * 4), *w = (float *)malloc((size_t)H * 4);
        uint8_t *val = (uint8_t *)malloc((size_t)rows);
        float *cout = (float *)malloc(n * 4), *cnrm = (float *)malloc(n * 4);
        fill(raw, n, 7u, 1.1f);
        fill(del, n, 8u, 0.8f);
        fill(rms, (size_t)rows, 9u, 1.0f);
        fill(w, (size_t)H, 10u, 0.6f);
        for (long i = 0; i < rows; i++) rms[i] = fabsf(rms[i]) + 0.01f;
        for (long i = 0; i < rows; i++) val[i] = 1;
        val[5] = 0;  // exercise the valid[] gate

        q35_dspine_feature_finish(raw, del, rms, val, w, cout, cnrm, rows, H, T, EPS, 1, 1, 0);

        float *dr, *dd, *drm, *dw, *dout, *dno;
        uint8_t *dv;
        CK(cudaMalloc(&dr, n * 4));
        CK(cudaMalloc(&dd, n * 4));
        CK(cudaMalloc(&drm, (size_t)rows * 4));
        CK(cudaMalloc(&dw, (size_t)H * 4));
        CK(cudaMalloc(&dv, (size_t)rows));
        CK(cudaMalloc(&dout, n * 4));
        CK(cudaMalloc(&dno, n * 4));
        CK(cudaMemcpy(dr, raw, n * 4, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dd, del, n * 4, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(drm, rms, (size_t)rows * 4, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dw, w, (size_t)H * 4, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dv, val, (size_t)rows, cudaMemcpyHostToDevice));
        int rc = q35_launch_dspine_feature_finish(dr, dd, drm, dv, dw, dout, dno, rows, (int)H,
                                                  (int)T, EPS, 1, 1, 0, 0);
        if (rc) { printf("launch feature_finish failed %d\n", rc); return 1; }
        CK(cudaDeviceSynchronize());
        float *gout = (float *)malloc(n * 4), *gno = (float *)malloc(n * 4);
        CK(cudaMemcpy(gout, dout, n * 4, cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(gno, dno, n * 4, cudaMemcpyDeviceToHost));
        printf("feature_finish rows=%d H=%ld T=%ld\n", rows, H, T);
        cmp("output", gout, cout, n);
        cmp("normed", gno, cnrm, n);
        cudaFree(dr); cudaFree(dd); cudaFree(drm); cudaFree(dw); cudaFree(dv);
        cudaFree(dout); cudaFree(dno);
        free(raw); free(del); free(rms); free(w); free(val);
        free(cout); free(cnrm); free(gout); free(gno);
    }

    printf("\n%s (%d failing arm(s))\n", g_fail ? "MISMATCH" : "CUDA == C, BIT-IDENTICAL",
           g_fail);
    return g_fail ? 1 : 0;
}
