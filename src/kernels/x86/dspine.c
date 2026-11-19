/* DSPINE code bridge: the five Triton kernels of
 * `sglang_dspine/kernels/dspine.py`, transcribed expression by expression.
 *
 * WHY THIS IS SCALAR C AND NOT SIMD: every one of these kernels is defined by
 * its ROUNDING POINTS, not by its arithmetic.  The reference runs them with
 * `enable_fp_fusion=False` and casts to bf16 at eleven specific places; any
 * re-association or FMA contraction changes the result.  A vectorized rewrite
 * with `-ffast-math`-style freedom would be subtly wrong, so this unit is
 * compiled at baseline ISA with no vectorizer-friendly structure.  The
 * workloads are tiny anyway (one 16-row block per spec round: 16x1024 codes +
 * 16x512 messages), so there is nothing to win.
 *
 * ROUND-TO-NEAREST-EVEN: `f32_to_bf16` below must match torch's
 * `.to(torch.bfloat16)`.  It keeps the upper 16 bits after adding a rounding
 * bias built from the bits being discarded, so an exact tie goes to even.
 *
 * Sources (all line numbers are the reference at /mnt/llm-dev/dspine):
 *   prepare_read     kernels/dspine.py:11
 *   normalize_codes  kernels/dspine.py:41
 *   feature_message  kernels/dspine.py:63
 *   feature_finish   kernels/dspine.py:85
 */

#include <math.h>
#include <stdint.h>
#include <string.h>

/* Smallest positive normal f32: Triton's `tl.maximum(..., 1.1754943508222875e-38)`
 * in prepare_read (:30).  Spelled as a bit pattern so no compiler can fold or
 * round it differently. */
#define Q35_FLT_MIN_NORMAL 0x00800000u
#define Q35_EPS_BIAS 1.1754943508222875e-38f

/* f32 -> bf16 bits, round-to-nearest-even; matches torch `.to(torch.bfloat16)`
 * for finite inputs (NaN payloads are not preserved, the reference does not
 * depend on them). */
uint16_t q35_f32_to_bf16_bits(float x) {
    uint32_t u;
    memcpy(&u, &x, sizeof u);
    /* 0x7FFF + ((u >> 16) & 1) is the standard RNE bias: round up when the
     * discarded part is above half, or exactly half with an odd LSB. */
    uint32_t bias = 0x7FFFu + ((u >> 16) & 1u);
    u += bias;
    return (uint16_t)(u >> 16);
}

float q35_f32_to_bf16(float x) {
    uint16_t h = q35_f32_to_bf16_bits(x);
    uint32_t u = (uint32_t)h << 16;
    float r;
    memcpy(&r, &u, sizeof r);
    return r;
}

/* ---------------------------------------------------------------------------
 * q35_dspine_sum_tree -- the reduction order shared with the CUDA backend.
 *
 * The reference sums with `tl.sum`, whose lowering depends on Triton's thread
 * layout for the target (`sizePerThread=4`, 128 threads on this one) -- that
 * order CHANGES between Triton releases and between GPU architectures, so
 * matching it bit-for-bit would pin this port to one Triton version for a
 * difference of ~1e-7 relative on a value that then feeds bf16 (resolution
 * 3.9e-3).  Four orders of magnitude below the destination.
 *
 * What is NOT negotiable is that our own two backends agree: `dspine.cu`
 * `block_reduce` folds each lane's strided partial sums with
 * `__shfl_down_sync` 16,8,4,2,1 and then folds the warp results through shared
 * memory in warp order.  This function reproduces that shape exactly on the
 * host, so CPU and CUDA differ only where the reference itself is not
 * well-defined.  `scripts/dspine_c_check.c` verifies the whole elementwise
 * chain, and a CUDA-vs-C check is what a disagreement would show up in.
 *
 * The tree, spelled out (kThreads=256, 8 warps):
 *   lane partial = sum of v[i] for i = tid, tid+256, tid+512, ... (strided)
 *   warp partial = fold the 32 lanes by XOR-with-offset 16, 8, 4, 2, 1
 *   block        = fold the 8 warp partials in the same butterfly
 * `shfl_down_sync(v, 16)` computes `v + v_from_lane(lane+16)`, which for the
 * low half is the same value as the XOR form used here; both round identically
 * because addition is commutative in IEEE-754 (not associative, but the
 * operand PAIR is what matters, and the pair is the same).
 * ------------------------------------------------------------------------- */
#define Q35_DSPINE_THREADS 256
#define Q35_DSPINE_WARPS (Q35_DSPINE_THREADS / 32)

/* Fold `n` values in place with the butterfly: offsets 16,8,4,2,1 are lane
 * steps, so only the low 32 entries survive as the reduced value at [0]. */
static float q35_dspine_fold32(const float *v) {
    float w[32];
    memcpy(w, v, sizeof w);
    for (int off = 16; off > 0; off >>= 1) {
        for (int lane = 0; lane < off; lane++) w[lane] += w[lane + off];
    }
    return w[0];
}

float q35_dspine_sum_tree(const float *v, long n) {
    float warp[Q35_DSPINE_WARPS];
    for (int wid = 0; wid < Q35_DSPINE_WARPS; wid++) {
        /* Each of the 32 lanes of this warp starts from its own strided sum. */
        float lane_vals[32];
        for (int lane = 0; lane < 32; lane++) {
            const long tid = (long)wid * 32 + lane;
            float acc = 0.0f;
            for (long i = tid; i < n; i += Q35_DSPINE_THREADS) acc += v[i];
            lane_vals[lane] = acc;
        }
        warp[wid] = q35_dspine_fold32(lane_vals);
    }
    /* Across warps: the device loads `(tid < nwarps) ? warp_sums[tid] : 0` into
     * a 32-lane array and folds that, so unused lanes contribute +0.0f. */
    float wide[32];
    for (int i = 0; i < 32; i++) wide[i] = (i < Q35_DSPINE_WARPS) ? warp[i] : 0.0f;
    return q35_dspine_fold32(wide);
}

/* `sum(v[i]^2)` in the SAME order: square while accumulating the strided
 * partial.  Squaring on the way in, rather than materialising a squared array
 * first, gives the identical result (each element is squared exactly once, and
 * `a*a` is exact-ish the same either way) and needs no buffer. */
float q35_dspine_sum_sq(const float *v, long n) {
    float warp[Q35_DSPINE_WARPS];
    for (int wid = 0; wid < Q35_DSPINE_WARPS; wid++) {
        float lane_vals[32];
        for (int lane = 0; lane < 32; lane++) {
            const long tid = (long)wid * 32 + lane;
            float acc = 0.0f;
            for (long i = tid; i < n; i += Q35_DSPINE_THREADS) acc += v[i] * v[i];
            lane_vals[lane] = acc;
        }
        warp[wid] = q35_dspine_fold32(lane_vals);
    }
    float wide[32];
    for (int i = 0; i < 32; i++) wide[i] = (i < Q35_DSPINE_WARPS) ? warp[i] : 0.0f;
    return q35_dspine_fold32(wide);
}

/* ---------------------------------------------------------------------------
 * prepare_read  (kernels/dspine.py:11)
 *
 *   x = X[row]                             (ADD off)
 *   x = (x + Residual[row]).to(bf16).to(f32)   (ADD on)   <- rounding point #1
 *   rms = sqrt(max(sum(x*x)/H, FLT_MIN_NORMAL))
 *   Raw[row]  = x
 *   Normed[row] = x / (rms + EPS)          <- NOT rsqrt, and NOT x*rms^-1
 *   RMS[row] = rms
 *
 * The division (rather than a reciprocal multiply) is the reason this kernel
 * differs from `rmsnorm`: `a / (b + eps)` and `a * (1/(b+eps))` differ in the
 * last ulp for roughly a third of inputs, and the reference's `tl` division is
 * the former.  `enable_fp_fusion=False` there keeps the compiler from turning
 * it into the latter.
 * ------------------------------------------------------------------------- */
void q35_dspine_prepare_read(const float *x, const float *residual, float *raw,
                             float *normed, float *rms_out, long rows, long h,
                             float eps, int add) {
    for (long row = 0; row < rows; row++) {
        const float *xr = x + row * h;
        float *rawr = raw + row * h;
        float *nrmr = normed + row * h;
        const float *resr = residual ? residual + row * h : 0;

        for (long i = 0; i < h; i++) {
            float v = xr[i];
            if (add) v += resr[i];
            /* Rounding point #1.  The MLX reference is the authority for this
             * checkpoint (`dspine_mlx/bridge.py:112`):
             *     raw = hidden if residual is None else (hidden + residual)
             *     raw = raw.astype(mx.bfloat16)     <- ALWAYS, not just when
             *     f = raw.astype(mx.float32)           a residual was added
             * The Triton port (`kernels/dspine.py:20-25`) only narrows inside
             * `if ADD`, which is the same thing at every call site that adds a
             * residual -- but `raw` is fed to `feature_finish` unchanged, so on
             * the ADD=False path the two references would disagree in the low
             * mantissa bits.  MLX wins because it is what the GGUF was trained
             * and exported from. */
            v = q35_f32_to_bf16(v);
            rawr[i] = v;
        }

        const float sum_ss = q35_dspine_sum_sq(rawr, h);
        float mean = sum_ss / (float)h;
        float clipped;
        memcpy(&clipped, &mean, sizeof clipped);
        /* max(mean, FLT_MIN_NORMAL): Triton compares as floats, so this is a
         * >= test on the numeric value, not on bits. */
        if (!(mean >= Q35_EPS_BIAS)) clipped = Q35_EPS_BIAS;
        float rms = sqrtf(clipped);
        if (rms_out) rms_out[row] = rms;

        for (long i = 0; i < h; i++) nrmr[i] = rawr[i] / (rms + eps);
    }
}

/* ---------------------------------------------------------------------------
 * normalize_codes  (kernels/dspine.py:41)
 *
 *   if ANCHOR_ROW and row % T == 0:
 *       z = Codebook[Anchors[row // T]] * sqrt(C)     (no normalisation)
 *   else:
 *       z = Projected[row]
 *       z = sqrt(C) * z / (sqrt(sum(z*z)) + EPS)
 *
 * The anchor rows are NOT renormalised -- the codebook entries are already
 * unit-ish and the reference scales them by sqrt(C) only.  `sqrt(float(C))` is
 * computed in f32 from a float cast, so it is `sqrtf((float)C)`.
 * ------------------------------------------------------------------------- */
void q35_dspine_normalize_codes(const float *projected, const float *codebook,
                                const int32_t *anchors, float *codes, long rows,
                                long c, long t, float eps, long row0) {
    const float scale = sqrtf((float)c);
    for (long i = 0; i < rows; i++) {
        const long row = row0 + i;
        float *z = codes + i * c;
        if (t > 0 && row % t == 0 && anchors) {
            /* `Anchors[row // T]` (kernels/dspine.py:53): the anchor id table is
             * indexed by the row's position INSIDE THE BLOCK divided by T, i.e.
             * "which anchor of this block is this".  With the reference's
             * one-call-per-block shape that is exactly `row // T`.  A caller
             * that splits the block into single rows passes a one-element
             * `anchors` and `row0 = row`, and `row // T` is then the OUTER
             * index (0 for every row of the first block, 1 for the second,
             * ...), which is out of range for that one-element array.
             *
             * So the block-local index is `(row0 + i) / t - row0 / t`: zero for
             * every row of the call's own block, whatever shape the caller
             * used.  This is the ONLY reading that is correct for both call
             * shapes, and it was the source of a silent wrong-anchor bug: the
             * engine drives one row per call, so the old `anchors[i]` pinned
             * every block to `anchors[0]`. */
            const long anchor_ix = (row0 + i) / t - row0 / t;
            const int32_t tok = anchors[anchor_ix];
            const float *src = codebook + (long)tok * c;
            for (long k = 0; k < c; k++) z[k] = q35_f32_to_bf16(src[k] * scale);
            continue;
        }
        const float *src = projected + i * c;
        /* Same tree order the device uses; see `q35_dspine_sum_sq`. */
        const float norm = sqrtf(q35_dspine_sum_sq(src, c));
        /* `sqrt(C) * z / (norm + EPS)` -- multiply first, divide second, in
         * that order.  `z * (sqrt(C)/(norm+eps))` is a different rounding. */
        /* rounding point: `codes` is bf16 in both references
         * (mx: `.astype(mx.bfloat16)`, triton: stored into a bf16 buffer). */
        for (long k = 0; k < c; k++) z[k] = q35_f32_to_bf16(scale * src[k] / (norm + eps));
    }
}

/* ---------------------------------------------------------------------------
 * feature_message  (kernels/dspine.py:63)
 *
 *   packed[row]    = [gate.query(R) | predecessor.sender(R) | prev.value(R)]
 *   query  = packed[row][0:R]
 *   sender = packed[prev][R:2R]        prev = max(row-1, 0)
 *   value  = packed[prev][2R:3R]
 *   gate   = sigmoid(query + sender + bias)
 *   msg    = (row % T == 0) ? 0 : gate * value
 *
 * Row 0 is an anchor: its own `prev` is itself, and the `%T` mask zeroes it
 * anyway, so `prev = max(row-1,0)` never reads out of bounds.
 * ------------------------------------------------------------------------- */
void q35_dspine_feature_message(const float *packed, const float *packed_prev,
                                const float *bias, float *message, long rows,
                                long r, long t, long row0) {
    for (long i = 0; i < rows; i++) {
        /* `row` is the row's index INSIDE THE BLOCK and is what the anchor /
         * predecessor logic needs; the buffers only ever hold this call's
         * `rows` rows, so every pointer walks with `i`.  Mixing them up reads
         * and writes out of bounds -- silently, since these are raw pointers. */
        const long row = row0 + i;
        /* The predecessor value comes from the PREVIOUS row's packed triple.
         * A caller that feeds one row at a time cannot supply it from `packed`,
         * so it passes that row separately as `packed_prev` (null on row 0,
         * where `prev` clamps to the row itself). */
        const float *prow = i > 0 ? packed + (i - 1) * 3 * r : (packed_prev ? packed_prev : packed);
        const float *q = packed + i * 3 * r;
        const float *s = prow + r;
        const float *v = prow + 2 * r;
        float *m = message + i * r;
        const int anchor = (t > 0 && row % t == 0);
        for (long k = 0; k < r; k++) {
            /* tl.sigmoid is 1/(1+exp(-x)); expf is the closest libm call.
             * `gate * value` is then multiplied by a mask that is exactly 0 or
             * 1, so the reference's `tl.where` and this multiplication agree
             * bit for bit (0*v is +-0, and the consumer only adds it). */
            const float g = 1.0f / (1.0f + expf(-(q[k] + s[k] + bias[k])));
            /* rounding point: `message.astype(mx.bfloat16)` (bridge.py:139) */
            m[k] = anchor ? 0.0f : q35_f32_to_bf16(g * v[k]);
        }
    }
}

/* ---------------------------------------------------------------------------
 * feature_finish  (kernels/dspine.py:85)
 *
 *   edge   = (row % T != 0) && valid[row] && valid[prev]
 *   update = rms * delta * edge          (WRITE)
 *   result = (raw + update).to(bf16)     <- rounding point
 *   if NORM:
 *       variance = sum(result^2)/H
 *       y = (result * rsqrt(variance + NORM_EPS)).to(bf16)   <- rounding
 *       y = (y * weight).to(bf16)                            <- rounding
 *
 * THREE consecutive bf16 roundings on the NORM path.  Measured in the
 * reference: collapsing them into one changes 25.97% of the elements, so this
 * is not cosmetic.
 *
 * Note `rsqrt` here and a plain division in prepare_read: the two kernels
 * genuinely differ, which is why they are separate functions rather than one
 * parameterised normaliser.
 * ------------------------------------------------------------------------- */
void q35_dspine_feature_finish(const float *raw, const float *delta,
                               const float *rms, const uint8_t *valid,
                               const float *weight, float *output,
                               float *normed, long rows, long h, long t,
                               float norm_eps, int write, int norm, long row0) {
    for (long i = 0; i < rows; i++) {
        const long row = row0 + i;
        const float *rawr = raw + i * h;
        const float *delr = delta + i * h;
        float *outr = output + i * h;
        const float rmsv = rms ? rms[i] : 0.0f;

        /* `row % t != 0` is the block-local anchor test; it must use the
         * ABSOLUTE row inside the block (`row0 + i`), not `i`, or a caller that
         * processes one row at a time sees row 0 every time and treats every
         * non-anchor row as an anchor -- silently skipping every write. */
        int edge = (t > 0 && row % t != 0);
        /* `valid` and `rms` are indexed by THIS CALL's row, because that is how
         * the production caller drives this kernel: one row per call, with a
         * one-element `rms` and a one-element `valid`.  The predecessor gate
         * therefore has to look at `i - 1` INSIDE the call, and `i == 0` has no
         * in-call predecessor -- the reference's whole-block call always has
         * one, so the per-row caller supplies the answer in `valid[0]` itself
         * (see `writeBridge`, which passes `row % block_size != 0`).
         *
         * Reading `valid[row - 1]` here instead (the global row) is what an
         * earlier version did: on a one-element array that is an
         * out-of-bounds byte, and it silently let anchor rows through.
         */
        if (valid) {
            edge = edge && (valid[i] != 0);
            if (i > 0) edge = edge && (valid[i - 1] != 0);
        }
        const float update_scale = write ? (edge ? rmsv : 0.0f) : 0.0f;

        for (long i = 0; i < h; i++) {
            /* rounding point: result is bf16 in the reference, and the NORM
             * branch below re-reads it as such. */
            outr[i] = q35_f32_to_bf16(rawr[i] + update_scale * delr[i]);
        }

        if (!norm) continue;

        const float variance = q35_dspine_sum_sq(outr, h) / (float)h;
        const float inv = 1.0f / sqrtf(variance + norm_eps);
        float *nrmr = normed + i * h;
        for (long i = 0; i < h; i++) {
            /* two more bf16 roundings, in this order */
            const float y = q35_f32_to_bf16(outr[i] * inv);
            nrmr[i] = q35_f32_to_bf16(y * (weight ? weight[i] : 1.0f));
        }
    }
}
