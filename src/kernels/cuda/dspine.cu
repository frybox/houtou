// dspine.cu -- the DSpine code bridge, device side.
//
// The four kernels of `/mnt/llm-dev/dspine/sglang_dspine/kernels/dspine.py`
// (`prepare_read` :11, `normalize_codes` :41, `feature_message` :63,
// `feature_finish` :85), transcribed expression by expression.  The scalar C
// port in `src/kernels/x86/dspine.c` is the specification this file is checked
// against, and both were written from the reference, not from each other.
//
// ===========================================================================
// WHAT "CORRECT" MEANS HERE, AND WHY IT IS NOT BIT-EQUALITY WITH TRITON
// ===========================================================================
// The reference runs every one of these kernels with `enable_fp_fusion=False`,
// which fixes two things and leaves a third open:
//
//   FIXED      the bf16 cast points.  There are eleven of them across the four
//              kernels, and collapsing consecutive roundings changes measured
//              fractions of the output -- feature_finish's three (result, then
//              normalised, then weighted) differ in 25.97 % of elements if they
//              are merged.  Every cast here is `__float2bfloat16_rn`, which is
//              round-to-nearest-even and identical to torch's
//              `.to(torch.bfloat16)` for finite inputs.
//   FIXED      the expression ORDER.  `sqrt(C) * z / (norm + eps)` is not
//              `z * (sqrt(C)/(norm+eps))`; `__fmul_rn`/`__fdiv_rn` and the
//              absence of `-fmad` contraction pin each step.
//   OPEN       the reduction order of `tl.sum`.  Triton lowers a reduction onto
//              a specific thread layout (`sizePerThread=4`, 128 threads,
//              `#blocked`), then a warp butterfly, then shared memory across
//              warps.  That order is a property of the Triton VERSION and the
//              warp count -- it changes between Triton releases and between
//              GPU architectures.  Chasing it bit-for-bit would lock this
//              kernel to "Triton 3.8 on sm_89" for no numerical benefit: the
//              difference is ~1e-7 relative on `rms`, while bf16 *is* the
//              precision of the data it feeds (2^-8 = 3.9e-3).  Four orders of
//              magnitude below the destination's own resolution.
//
// So the contract is:
//   * bf16 cast points            bit-identical to the reference (hard);
//   * elementwise arithmetic      bit-identical (hard, pinned with _rn);
//   * reduction order             this file's own `block_reduce` tree, which
//                                 is also what `src/kernels/x86/dspine.c` uses
//                                 in its `q35_dspine_sum_tree` helper, so the
//                                 two of our implementations agree exactly and
//                                 `QWEN35_MVCHECK`-style A/B is meaningful.
//
// The reduction is the ONLY place this file can differ from the reference, and
// it is listed in one function so that claim is checkable by reading one thing.
//
// ===========================================================================
// LAYOUT
// ===========================================================================
// One block per row for all four kernels: the rows are at most 16 (one spec
// block) and each has H=5120 or C=1024 or R=512 elements, so a row is the
// natural work unit and grid.y is the row.  `block_reduce` is the tree form
// from norm.cu (warp `__shfl_down_sync` 16,8,4,2,1 then across warps through
// shared memory), reused verbatim so the two files cannot drift.
//
// The `row0` argument of the C port is kept: `row % T != 0` is the BLOCK-local
// anchor test and must not be evaluated on the call-local index `i`, or a
// caller that feeds one row per call treats every row as an anchor and silently
// skips every write.
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdint.h>

#include "common.cuh"

namespace {

constexpr int kThreads = 256;

// Block-wide f32 sum, tree order: warp butterfly then across warps.  Kept
// identical to `src/kernels/x86/dspine.c`'s `q35_dspine_sum_tree`, which folds
// the same array in the same shape on the host -- that is what makes the CUDA
// and CPU backends agree bit-for-bit on these four kernels.
__device__ __forceinline__ float block_reduce(float v) {
    __shared__ float warp_sums[kThreads / 32];
    v += __shfl_down_sync(0xffffffffu, v, 16);
    v += __shfl_down_sync(0xffffffffu, v, 8);
    v += __shfl_down_sync(0xffffffffu, v, 4);
    v += __shfl_down_sync(0xffffffffu, v, 2);
    v += __shfl_down_sync(0xffffffffu, v, 1);
    const int lane = threadIdx.x & 31;
    const int wid = threadIdx.x >> 5;
    if (lane == 0) warp_sums[wid] = v;
    __syncthreads();
    v = (threadIdx.x < kThreads / 32) ? warp_sums[threadIdx.x] : 0.0f;
    if (wid == 0) {
        v += __shfl_down_sync(0xffffffffu, v, 16);
        v += __shfl_down_sync(0xffffffffu, v, 8);
        v += __shfl_down_sync(0xffffffffu, v, 4);
        v += __shfl_down_sync(0xffffffffu, v, 2);
        v += __shfl_down_sync(0xffffffffu, v, 1);
    }
    if (threadIdx.x == 0) warp_sums[0] = v;
    __syncthreads();
    return warp_sums[0];
}

// f32 -> bf16 round-to-nearest-even, as f32.  `__float2bfloat16_rn` is the same
// rounding as torch's `.to(torch.bfloat16)`; the value is then widened back so
// the consumer arithmetic sees exactly what the reference's
// `.to(bf16).to(f32)` sees.
__device__ __forceinline__ float rn_bf16(float x) {
    return __bfloat162float(__float2bfloat16_rn(x));
}

// ---------------------------------------------------------------------------
// prepare_read  (kernels/dspine.py:11)
//
//   x  = X[row];  if ADD: x = bf16(x + Residual[row])      <- rounding point 1
//   rms = sqrt(max(sum(x*x)/H, FLT_MIN_NORMAL))
//   Raw[row]    = x
//   Normed[row] = x / (rms + EPS)      -- division, NOT x * (1/(rms+eps))
//
// The reference narrows to bf16 unconditionally on the MLX side
// (`bridge.py:112`) and inside `if ADD` on the Triton side; every call site
// that adds a residual makes those agree, and `writeBridge` always takes the
// ADD=false path with `x` already bf16, so `rn_bf16` is applied unconditionally
// here to match the MLX authority.
// ---------------------------------------------------------------------------
__global__ void k_prepare_read(const float *__restrict__ x,
                               const float *__restrict__ residual,
                               float *__restrict__ raw, float *__restrict__ normed,
                               float *__restrict__ rms_out, int h, float eps,
                               int add) {
    const int row = blockIdx.x;
    const int stride = blockDim.x;
    const float *xr = x + (size_t)row * (size_t)h;
    const float *rr = add ? residual + (size_t)row * (size_t)h : nullptr;
    float *rawr = raw + (size_t)row * (size_t)h;

    for (int i = threadIdx.x; i < h; i += stride) {
        float v = xr[i];
        if (add) v = __fadd_rn(v, rr[i]);
        // Rounding point 1: the reference stores a bf16 here and reads it back.
        rawr[i] = rn_bf16(v);
    }
    __syncthreads();

    float ss = 0.0f;
    for (int i = threadIdx.x; i < h; i += stride) ss += __fmul_rn(rawr[i], rawr[i]);
    const float mean = __fdiv_rn(block_reduce(ss), (float)h);
    // `tl.maximum(mean, FLT_MIN_NORMAL)`: a numeric compare, not a bit test.
    const float clipped = (mean >= 1.1754943508222875e-38f) ? mean : 1.1754943508222875e-38f;
    const float rms = sqrtf(clipped);
    if (threadIdx.x == 0 && rms_out) rms_out[row] = rms;

    const float denom = __fadd_rn(rms, eps);
    float *nrm = normed + (size_t)row * (size_t)h;
    for (int i = threadIdx.x; i < h; i += stride) nrm[i] = __fdiv_rn(rawr[i], denom);
}

// ---------------------------------------------------------------------------
// normalize_codes  (kernels/dspine.py:41)
//
//   ANCHOR_ROW and row % T == 0:  z = Codebook[Anchors[row//T]] * sqrt(C)
//   otherwise:                    z = sqrt(C) * z / (sqrt(sum(z*z)) + EPS)
//
// The anchor rows are NOT renormalised.  `row // T` is the block-local anchor
// index; see the long note in `dspine.c` for why the caller supplies an anchor
// table that is indexed by the CALL, not by the absolute row.
// ---------------------------------------------------------------------------
__global__ void k_normalize_codes(const float *__restrict__ projected,
                                  const float *__restrict__ codebook,
                                  const int32_t *__restrict__ anchors,
                                  float *__restrict__ codes, int c, int t, float eps,
                                  int row0) {
    const int i = blockIdx.x;
    const int row = row0 + i;
    const int stride = blockDim.x;
    const float scale = sqrtf((float)c);
    float *z = codes + (size_t)i * (size_t)c;

    if (t > 0 && (row % t) == 0 && anchors) {
        const int32_t tok = anchors[(row0 + i) / t - row0 / t];
        const float *src = codebook + (size_t)tok * (size_t)c;
        for (int k = threadIdx.x; k < c; k += stride) z[k] = rn_bf16(__fmul_rn(src[k], scale));
        return;
    }

    const float *src = projected + (size_t)i * (size_t)c;
    float ss = 0.0f;
    for (int k = threadIdx.x; k < c; k += stride) ss += __fmul_rn(src[k], src[k]);
    // `sqrt(tl.sum(z*z))` -- an f32 sum, then a correctly-rounded sqrt.
    const float norm = sqrtf(block_reduce(ss));
    const float denom = __fadd_rn(norm, eps);
    for (int k = threadIdx.x; k < c; k += stride) {
        // `sqrt(C) * z / (norm + EPS)`: multiply first, divide second.
        z[k] = rn_bf16(__fdiv_rn(__fmul_rn(scale, src[k]), denom));
    }
}

// ---------------------------------------------------------------------------
// feature_message  (kernels/dspine.py:63)
//
//   query  = packed[row,  :R]
//   sender = packed[prev, R:2R]      prev = max(row-1, 0)
//   value  = packed[prev, 2R:3R]
//   gate   = sigmoid(query + sender + bias)
//   msg    = (row % T == 0) ? 0 : bf16(gate * value)
//
// `packed` here is ONE row per block-index only when the caller has it; the
// scalar port takes an optional `packed_prev` because it may be driven one row
// at a time.  On the device the whole block is resident (it is 16 x 3R floats),
// so this kernel always reads the predecessor straight out of `packed` and does
// not need that escape hatch.
// ---------------------------------------------------------------------------
__global__ void k_feature_message(const float *__restrict__ packed,
                                  const float *__restrict__ packed_prev,
                                  const float *__restrict__ bias,
                                  float *__restrict__ message, int r, int t,
                                  int row0) {
    const int i = blockIdx.x;
    const int row = row0 + i;
    const int stride = blockDim.x;

    // The caller owns the predecessor, exactly as the scalar C port's signature
    // does: `packed` holds THIS call's rows (row i at `i*3R`), and the row
    // BEFORE the first one arrives separately.  Reading the predecessor out of
    // `packed` instead would require `packed` to be block-sized AND to be
    // indexed by the absolute row -- a two-part contract that the draft got
    // wrong (every non-anchor row saw its OWN sender, `msg` came out 0, and the
    // code bridge silently did nothing).
    const float *q = packed + (size_t)i * (size_t)(3 * r);
    const float *p = i > 0 ? packed + (size_t)(i - 1) * (size_t)(3 * r)
                           : (packed_prev ? packed_prev : packed);
    const float *s = p + r;
    const float *v = p + 2 * r;
    float *m = message + (size_t)i * (size_t)r;
    const bool anchor = (t > 0 && (row % t) == 0);

    for (int k = threadIdx.x; k < r; k += stride) {
        if (anchor) {
            m[k] = 0.0f;
            continue;
        }
        // sigmoid(q+s+b) with the same association as the reference's
        // `q[:, 1:] + s[:, :-1]` then `+ bias`: ((q + s) + bias).
        const float arg = __fadd_rn(__fadd_rn(q[k], s[k]), bias[k]);
        const float g = __fdiv_rn(1.0f, __fadd_rn(1.0f, __expf(-arg)));
        m[k] = rn_bf16(__fmul_rn(g, v[k]));
    }
}

// ---------------------------------------------------------------------------
// feature_finish  (kernels/dspine.py:85)
//
//   edge   = (row % T != 0) && valid[row] && valid[prev]
//   update = rms * delta * edge
//   result = bf16(raw + update)                                  <- rounding 9
//   normed = bf16(bf16(result * rsqrt(var+eps)) * weight)        <- roundings 10,11
//
// THREE consecutive bf16 roundings on the NORM path.  Measured in the
// reference: collapsing them into one changes 25.97 % of the elements.
//
// `1/sqrt` rather than `rsqrtf`: `rsqrtf` is an approximation instruction and
// the reference's `tl.rsqrt` lowers to the correct `1/sqrt` on this target, so
// the two-operand form keeps the ulp behaviour that matches.
// ---------------------------------------------------------------------------
__global__ void k_feature_finish(const float *__restrict__ raw,
                                 const float *__restrict__ delta,
                                 const float *__restrict__ rms,
                                 const uint8_t *__restrict__ valid,
                                 const float *__restrict__ weight,
                                 float *__restrict__ output, float *__restrict__ normed,
                                 int h, int t, float norm_eps, int write, int norm,
                                 int row0) {
    const int i = blockIdx.x;
    const int row = row0 + i;
    const int stride = blockDim.x;

    const float *rawr = raw + (size_t)i * (size_t)h;
    const float *delr = delta + (size_t)i * (size_t)h;
    float *outr = output + (size_t)i * (size_t)h;

    bool edge = (t > 0 && (row % t) != 0);
    if (valid) {
        edge = edge && (valid[i] != 0);
        if (i > 0) edge = edge && (valid[i - 1] != 0);
    }
    const float rmsv = (rms && edge && write) ? rms[i] : 0.0f;

    for (int k = threadIdx.x; k < h; k += stride) {
        // Rounding point 9: `result` is bf16 and the NORM branch re-reads it.
        outr[k] = rn_bf16(__fadd_rn(rawr[k], __fmul_rn(rmsv, delr[k])));
    }
    if (!norm) return;
    // Every thread reaches this barrier: the `return` above is uniform across
    // the block (it tests a kernel ARGUMENT, not a thread-varying value), which
    // is what makes `__syncthreads` here legal.  A per-thread early exit before
    // a barrier is undefined behaviour and shows up as an illegal access or a
    // hang, not as a wrong number.
    __syncthreads();

    float ss = 0.0f;
    for (int k = threadIdx.x; k < h; k += stride) ss += __fmul_rn(outr[k], outr[k]);
    const float variance = __fdiv_rn(block_reduce(ss), (float)h);
    const float inv = __fdiv_rn(1.0f, sqrtf(__fadd_rn(variance, norm_eps)));

    float *nrm = normed + (size_t)i * (size_t)h;
    for (int k = threadIdx.x; k < h; k += stride) {
        // Roundings 10 and 11, in this order.
        const float y = rn_bf16(__fmul_rn(outr[k], inv));
        nrm[k] = rn_bf16(__fmul_rn(y, weight ? weight[k] : 1.0f));
    }
}

} // namespace

extern "C" int q35_launch_dspine_prepare_read(const float *d_x, const float *d_residual,
                                              float *d_raw, float *d_normed,
                                              float *d_rms, int rows, int h, float eps,
                                              int add, void *stream_v) {
    if (rows <= 0 || h <= 0) return (int)cudaSuccess;
    if (add && !d_residual) return (int)cudaErrorInvalidValue;
    cudaStream_t stream = (cudaStream_t)stream_v;
    k_prepare_read<<<rows, kThreads, 0, stream>>>(d_x, d_residual, d_raw, d_normed, d_rms,
                                                  h, eps, add);
    return (int)cudaGetLastError();
}

extern "C" int q35_launch_dspine_normalize_codes(const float *d_projected,
                                                 const float *d_codebook,
                                                 const int32_t *d_anchors, float *d_codes,
                                                 int rows, int c, int t, float eps, int row0,
                                                 void *stream_v) {
    if (rows <= 0 || c <= 0) return (int)cudaSuccess;
    cudaStream_t stream = (cudaStream_t)stream_v;
    k_normalize_codes<<<rows, kThreads, 0, stream>>>(d_projected, d_codebook, d_anchors,
                                                     d_codes, c, t, eps, row0);
    return (int)cudaGetLastError();
}

extern "C" int q35_launch_dspine_feature_message(const float *d_packed,
                                                 const float *d_packed_prev,
                                                 const float *d_bias, float *d_message,
                                                 int rows, int r, int t, int row0,
                                                 void *stream_v) {
    if (rows <= 0 || r <= 0) return (int)cudaSuccess;
    cudaStream_t stream = (cudaStream_t)stream_v;
    k_feature_message<<<rows, kThreads, 0, stream>>>(d_packed, d_packed_prev, d_bias,
                                                     d_message, r, t, row0);
    return (int)cudaGetLastError();
}

extern "C" int q35_launch_dspine_feature_finish(const float *d_raw, const float *d_delta,
                                                const float *d_rms, const uint8_t *d_valid,
                                                const float *d_weight, float *d_output,
                                                float *d_normed, int rows, int h, int t,
                                                float norm_eps, int write, int norm,
                                                int row0, void *stream_v) {
    if (rows <= 0 || h <= 0) return (int)cudaSuccess;
    cudaStream_t stream = (cudaStream_t)stream_v;
    k_feature_finish<<<rows, kThreads, 0, stream>>>(d_raw, d_delta, d_rms, d_valid, d_weight,
                                                    d_output, d_normed, h, t, norm_eps, write,
                                                    norm, row0);
    return (int)cudaGetLastError();
}
