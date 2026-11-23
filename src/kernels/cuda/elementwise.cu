// elementwise.cu -- the four pure elementwise (per-element, no reduction)
// activation operations, moved off the CPU in P2b:
//
//   scaleVec   x[i] *= s
//   addResid   dst[i] += src[i]
//   copyRange  dst[i]  = src[i]
//   zeroBuf    buf[i]  = 0
//
// ---------------------------------------------------------------------------
// WHY THESE ARE TRIVIAL AND WHY THAT IS THE POINT
// ---------------------------------------------------------------------------
// None of the four has a reduction, a lookup table or a transcendental, so
// there is no summation order to preserve and no table to share.  Each element
// is one IEEE-754 single operation on the same input the CPU kernel sees,
// therefore GPU and CPU results must be **bit-identical**, not "within a
// tolerance" (the acceptance probe compares token ids, which requires exactly
// this).  Keep it that way: no fused multiply-add, no fast-math reassociation,
// no `-use_fast_math`.
//
// To make that guarantee independent of nvcc's default contraction (`-fmad`,
// which is ON and is allowed to weld a `*` and a neighbouring `+` into one
// fma), the two arithmetic kernels use the explicit round-to-nearest
// intrinsics `__fmul_rn` / `__fadd_rn`.  Those are the *uncontractable*
// spelling of exactly the operation the CPU performs, so a later edit that
// happens to add an add next to the multiply cannot silently change the
// numerics, and the file needs no special nvcc flag (the shared stage-1 flags
// in build.zig stay uniform across .cu files).
//
// Copies and zero-fill are bit moves / exact zero, so they need no intrinsic.
//
// ---------------------------------------------------------------------------
// LAYOUT / CONTRACT
// ---------------------------------------------------------------------------
// All four take raw device pointers plus an ELEMENT count (`n` of f32), never a
// byte count, and never a stride: every buffer the engine hands over is a
// contiguous span of f32 (backend.Buf).  `n <= 0` returns cudaSuccess without
// launching, which is what the Zig side does for an empty Buf -- the alternative
// (launching a zero-thread grid) is a cudaErrorInvalidConfiguration.
//
// There is one thread per element from a single flat index (no grid-stride
// loop), so the mapping is exact: element i is handled by exactly one thread
// and no element is read or written twice.  That is what makes the in-place
// `addResid` / `scaleVec` safe, and it is also why the result does not depend
// on the block size.
#include "common.cuh"
#include "xd.h"  // XD_CTL_*: the per-step control block indices
#include <math_constants.h>

namespace {

extern "C" __global__ void k_scale_vec(float *__restrict__ x, float s, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] = __fmul_rn(x[i], s);
}

// `tokens` runs of `n` elements starting `stride` floats apart (see
// `q35_launch_scale_vec_strided`).  Same `__fmul_rn` per element.
__global__ void k_scale_vec_strided(float *__restrict__ x, float s, int n, int stride) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float *xp = x + (size_t)blockIdx.y * (size_t)stride + (size_t)i;
    *xp = __fmul_rn(*xp, s);
}

extern "C" __global__ void k_add_resid(float *__restrict__ dst, const float *__restrict__ src, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = __fadd_rn(dst[i], src[i]);
}

extern "C" __global__ void k_copy_range(float *__restrict__ dst, const float *__restrict__ src, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = src[i];
}

// `k_copy_range` with the destination row chosen on the DEVICE: `dst_base` is
// the region base, `stride` the byte offset from an f32 base to the next row,
// and the row index comes from the per-step control block (XD_CTL_KV_SLOT).
//
// This is the f32-KV append inside a captured graph.  The captured node holds
// only `dst_base` (a constant) and the control-block pointer, so a replay needs
// no `SetParams` even though the destination moves one row per token -- the same
// argument `q35_k_kv_quant_dslot` is built on for the quantized formats.
extern "C" __global__ void k_copy_range_dslot(float *__restrict__ dst_base,
                                              int dst_stride_floats,
                                              const float *__restrict__ src,
                                              int rows, int n,
                                              const int *__restrict__ ctl) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const int r = i / dst_stride_floats;
    const int j = i - r * dst_stride_floats;
    if (r >= rows) return;
    const int slot = ctl[XD_CTL_KV_SLOT] + r;
    dst_base[(size_t)slot * (size_t)dst_stride_floats + (size_t)j] = src[i];
}

// Split `[h][q_head | gate_head]` into two dense head-major arrays.
//
// One block per head, `hd` elements in, `2*hd` out: the source block for head
// `h` is contiguous (two `hd` runs), so a block copy is the natural shape and
// the whole thing is one launch for the entire projection instead of `2*heads`
// launches of one head each.  That is the entire point -- the bytes are the
// same, the launch count is not (`PROF-COPY` put the per-head form at 768
// calls/token and 99% of the `copy_range` slot).
//
// EXACTNESS: a pure move, no arithmetic, so it is bit-identical to
// `elementwise.bufDeinterleaveQG` by construction.
extern "C" __global__ void k_deinterleave_qg(float *__restrict__ q, float *__restrict__ gate,
                                 const float *__restrict__ src, int hd) {
    const int h = blockIdx.x;
    const float *s = src + (size_t)h * 2 * hd;
    for (int i = threadIdx.x; i < hd; i += blockDim.x) {
        q[(size_t)h * hd + i] = s[i];
        gate[(size_t)h * hd + i] = s[hd + i];
    }
}

// Chunk form: blockIdx.y selects the token, and each row starts `row_stride`
// (= heads*hd) floats past the previous one on every buffer.  Same per-element
// move, so a prefill chunk gets one launch instead of one per token.
__global__ void k_deinterleave_qg_strided(float *__restrict__ q, float *__restrict__ gate,
                                         const float *__restrict__ src, int hd,
                                         int row_stride) {
    const int h = blockIdx.x;
    const int t = blockIdx.y;
    const float *s = src + (size_t)t * 2 * row_stride + (size_t)h * 2 * hd;
    float *qt = q + (size_t)t * row_stride + (size_t)h * hd;
    float *gt = gate + (size_t)t * row_stride + (size_t)h * hd;
    for (int i = threadIdx.x; i < hd; i += blockDim.x) {
        qt[i] = s[i];
        gt[i] = s[hd + i];
    }
}

extern "C" __global__ void k_zero_buf(float *__restrict__ buf, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) buf[i] = 0.0f;
}

// 256 threads: same choice as mmvq.cu / convert.cu (an integer multiple of the
// warp size that keeps the launch config identical across the kernel set, so a
// profiler comparison between kernels is not confounded by occupancy).
constexpr int kThreads = 256;

inline int gridFor(int n) { return (n + kThreads - 1) / kThreads; }

} // namespace

extern "C" int q35_launch_scale_vec(float *d_x, float s, int n, void *stream_v) {
    if (n <= 0) return (int)cudaSuccess;
    cudaStream_t stream = (cudaStream_t)stream_v;
    k_scale_vec<<<gridFor(n), kThreads, 0, stream>>>(d_x, s, n);
    return (int)cudaGetLastError();
}

// `tokens * n` elements scaled in ONE launch: token t's run starts `stride`
// floats past token t-1's.  Same per-element multiply as `q35_launch_scale_vec`,
// so a prefill chunk gets one launch where it used to get one per token.
extern "C" int q35_launch_scale_vec_strided(float *d_x, int offset, float s, int n,
                                          int tokens, int stride, void *stream_v) {
    if (n <= 0 || tokens <= 0) return (int)cudaSuccess;
    if (stride < offset + n) return (int)cudaErrorInvalidValue;
    cudaStream_t stream = (cudaStream_t)stream_v;
    k_scale_vec_strided<<<dim3((unsigned)gridFor(n), (unsigned)tokens), kThreads, 0,
                          stream>>>(d_x + (size_t)offset, s, n, stride);
    return (int)cudaGetLastError();
}

extern "C" int q35_launch_add_resid(float *d_dst, const float *d_src, int n, void *stream_v) {
    if (n <= 0) return (int)cudaSuccess;
    cudaStream_t stream = (cudaStream_t)stream_v;
    k_add_resid<<<gridFor(n), kThreads, 0, stream>>>(d_dst, d_src, n);
    return (int)cudaGetLastError();
}

// Host-stub address of the copy kernel, so a replaying decode graph can find
// this node's captured arguments (see `q35_func_rope64` for the mechanism).
// The KV append is why it matters: its destination is `cache + slot * kv_dim`,
// a pointer that moves once per token.
extern "C" void *q35_func_copy_range(void) { return (void *)&k_copy_range; }

extern "C" int q35_launch_copy_range(float *d_dst, const float *d_src, int n, void *stream_v) {
    if (n <= 0) return (int)cudaSuccess;
    cudaStream_t stream = (cudaStream_t)stream_v;
    k_copy_range<<<gridFor(n), kThreads, 0, stream>>>(d_dst, d_src, n);
    return (int)cudaGetLastError();
}

extern "C" int q35_launch_deinterleave_qg(float *d_q, float *d_gate, const float *d_src,
                                          int heads, int hd, void *stream_v) {
    if (heads <= 0 || hd <= 0) return (int)cudaSuccess;
    cudaStream_t stream = (cudaStream_t)stream_v;
    // `hd` threads per block is 256 for every model in this family (head_dim is
    // 128 or 256), i.e. the same as `kThreads`; the loop in the kernel handles
    // a larger `hd` without changing the launch config.
    k_deinterleave_qg<<<heads, hd < kThreads ? hd : kThreads, 0, stream>>>(d_q, d_gate, d_src, hd);
    return (int)cudaGetLastError();
}

// `tokens` rows split in ONE launch: row t's [q|gate] block starts at
// `t * 2 * heads * hd` on both sides, exactly where the per-row call pointed.
// Pure move, no arithmetic, so it is bit-identical to `tokens` per-row calls
// (that is what `forwardFullChunk` used to make -- 940 launches per 525-token
// prefill, 1.2% of device time at 6 us each).
extern "C" int q35_launch_deinterleave_qg_strided(float *d_q, float *d_gate, const float *d_src,
                                                  int heads, int hd, int tokens,
                                                  void *stream_v) {
    if (heads <= 0 || hd <= 0 || tokens <= 0) return (int)cudaSuccess;
    cudaStream_t stream = (cudaStream_t)stream_v;
    k_deinterleave_qg_strided<<<dim3((unsigned)heads, (unsigned)tokens),
                                hd < kThreads ? hd : kThreads, 0, stream>>>(
        d_q, d_gate, d_src, hd, heads * hd);
    return (int)cudaGetLastError();
}

// Publish one int32 into device memory from a 1-thread kernel.
//
// This is the CUDA-graph escape hatch for "a scalar launch argument that changes
// every token": if the value lives in DEVICE memory, the captured node's
// argument block and shared-memory size stay constant, so the node never needs
// `cudaGraphExecKernelNodeSetParams`.  Same trick as `k_rope64_setenv`
// (rope.cu:182), generalised so any kernel can take a per-token int this way.
// Measured motivation: the per-layer, per-token SetParams on the attn_decode
// node was costing 14.3 ms/token of the 27B's 33.8 ms decode.
extern "C" __global__ void k_set_int1(int *d, int v) { *d = v; }

extern "C" int q35_launch_set_int1(int *d, int v, void *stream_v) {
    if (d == nullptr) return (int)cudaErrorInvalidValue;
    k_set_int1<<<1, 1, 0, (cudaStream_t)stream_v>>>(d, v);
    return (int)cudaGetLastError();
}

extern "C" int q35_launch_copy_range_dslot(float *d_dst_base, int dst_stride_floats,
                                          const float *d_src, int rows, int n,
                                          const int *d_ctl, void *stream_v) {
    if (rows <= 0 || n <= 0 || dst_stride_floats <= 0) return (int)cudaSuccess;
    if (d_ctl == nullptr) return (int)cudaErrorInvalidValue;
    if (n != rows * dst_stride_floats) return (int)cudaErrorInvalidValue;
    k_copy_range_dslot<<<gridFor(n), kThreads, 0, (cudaStream_t)stream_v>>>(
        d_dst_base, dst_stride_floats, d_src, rows, n, d_ctl);
    return (int)cudaGetLastError();
}

extern "C" int q35_launch_zero_buf(float *d_buf, int n, void *stream_v) {
    if (n <= 0) return (int)cudaSuccess;
    cudaStream_t stream = (cudaStream_t)stream_v;
    k_zero_buf<<<gridFor(n), kThreads, 0, stream>>>(d_buf, n);
    return (int)cudaGetLastError();
}

// ---------------------------------------------------------------------------
// argmax -- the one REDUCTION here, because it cannot be avoided: `sample`
// needs the index of the largest logit and the engine holds logits in device
// memory (P2c).  Doing it on the host would mean a D2H of the whole vocabulary
// per token, which is exactly the kind of transfer P2c removes.
//
// EXACTNESS: a max over floats is order-independent (no rounding), so the
// requirement is only that the TIE-BREAK matches `elementwise.argmaxKernel`:
// it uses `v > best` scanning ascending, so the FIRST index of the maximum
// wins.  This kernel therefore reduces (value, index) pairs with
// "greater value wins; equal value keeps the SMALLER index".
// ---------------------------------------------------------------------------

// Block reduction over (value, index).  Each thread scans a strided slice, then
// the block folds 256 candidates down in shared memory.
extern "C" __global__ void k_argmax(const float *__restrict__ d_x, int n, int *d_out) {
    __shared__ float s_val[256];
    __shared__ int s_idx[256];
    const int tid = threadIdx.x;
    float best = -CUDART_INF_F;
    int best_i = 0;
    for (int i = tid; i < n; i += blockDim.x) {
        const float v = d_x[i];
        // Strict > keeps the first index on ties within a thread's slice; the
        // (value, index) combine below extends that across threads.
        if (v > best) { best = v; best_i = i; }
    }
    s_val[tid] = best;
    s_idx[tid] = best_i;
    __syncthreads();
    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            if (s_val[tid + stride] > s_val[tid] ||
                (s_val[tid + stride] == s_val[tid] && s_idx[tid + stride] < s_idx[tid])) {
                s_val[tid] = s_val[tid + stride];
                s_idx[tid] = s_idx[tid + stride];
            }
        }
        __syncthreads();
    }
    if (tid == 0) *d_out = s_idx[0];
}

// ---------------------------------------------------------------------------
// argmax: TWO-STAGE, MANY BLOCKS.
//
// The single-block version above (`k_argmax<<<1,256>>>`) scans all `n` logits
// from ONE SM.  On the 27B the lm_head is 248320 rows, i.e. ~1 MB of f32, and a
// lone SM streams that at roughly 1/114 of the chip's bandwidth: measured
// 13.4 ms of DEVICE time per token for `argmax` alone, which was 34% of the
// whole decode step (35.4 ms total, of which the matvecs were only 22.4 ms).
// The wait showed up as "sample"/"argmax" in PROF-SLOT and looked like host
// overhead; it was one block reading 1 MB.
//
// Stage 1 runs `blocks` blocks of 256 threads over strided slices and writes
// one (value, index) pair each.  Stage 2 folds those `blocks` pairs with a
// single block.  Ties still resolve to the SMALLER index -- the same
// "value, then index" combine, so the answer is identical to the serial scan --
// which is why this can replace the old kernel without changing the token
// stream.
// ---------------------------------------------------------------------------
constexpr int kArgmaxThreads = 256;
constexpr int kArgmaxMaxBlocks = 512;

__device__ __forceinline__ void argmax_combine(float &bv, int &bi, float v, int i) {
    if (v > bv || (v == bv && i < bi)) { bv = v; bi = i; }
}

// Reduce (value, index) across one block into lane 0 of `s_val`/`s_idx`.
__device__ __forceinline__ void argmax_block_reduce(float *s_val, int *s_idx, int tid) {
    __syncthreads();
    for (int stride = kArgmaxThreads >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            float v = s_val[tid + stride];
            int i = s_idx[tid + stride];
            argmax_combine(s_val[tid], s_idx[tid], v, i);
        }
        __syncthreads();
    }
}

__global__ void k_argmax_partial(const float *__restrict__ d_x, int n,
                                 float *__restrict__ d_pv, int *__restrict__ d_pi,
                                 int nblocks) {
    __shared__ float s_val[kArgmaxThreads];
    __shared__ int s_idx[kArgmaxThreads];
    const int tid = threadIdx.x;
    // Interleave block and thread so consecutive warps read consecutive memory
    // and the whole grid strides the array exactly once.
    const int stride = blockDim.x * nblocks;
    float best = -CUDART_INF_F;
    int best_i = 0;
    for (int i = blockIdx.x * blockDim.x + tid; i < n; i += stride) {
        const float v = d_x[i];
        if (v > best) { best = v; best_i = i; }
    }
    s_val[tid] = best;
    s_idx[tid] = best_i;
    argmax_block_reduce(s_val, s_idx, tid);
    if (tid == 0) { d_pv[blockIdx.x] = s_val[0]; d_pi[blockIdx.x] = s_idx[0]; }
}

__global__ void k_argmax_final(const float *__restrict__ d_pv, const int *__restrict__ d_pi,
                               int m, int *__restrict__ d_out) {
    __shared__ float s_val[kArgmaxThreads];
    __shared__ int s_idx[kArgmaxThreads];
    const int tid = threadIdx.x;
    float best = -CUDART_INF_F;
    int best_i = 0;
    for (int i = tid; i < m; i += blockDim.x) argmax_combine(best, best_i, d_pv[i], d_pi[i]);
    s_val[tid] = best;
    s_idx[tid] = best_i;
    argmax_block_reduce(s_val, s_idx, tid);
    if (tid == 0) *d_out = s_idx[0];
}

// One block per row: blockIdx.y picks the row, the block strides that row's `n`
// floats and reduces in shared memory.  A single block per row is enough here
// because the caller is the spec VERIFY (a handful of vocab scans) rather than
// per-token decode, and it keeps the whole call to one launch + one D2H.
extern "C" __global__ void k_argmax_batch(const float *__restrict__ d_x, int n, int *__restrict__ d_out) {
    __shared__ float s_val[kArgmaxThreads];
    __shared__ int s_idx[kArgmaxThreads];
    const int tid = threadIdx.x;
    const float *row = d_x + (size_t)blockIdx.y * (size_t)n;
    const int stride = blockDim.x;
    float best = -CUDART_INF_F;
    int best_i = 0;
    for (int i = tid; i < n; i += stride) {
        const float v = row[i];
        if (v > best) { best = v; best_i = i; }
    }
    s_val[tid] = best;
    s_idx[tid] = best_i;
    argmax_block_reduce(s_val, s_idx, tid);
    if (tid == 0) d_out[blockIdx.y] = s_idx[0];
}

extern "C" int q35_launch_argmax_batch(const float *d_x, int n, int rows,
                                       int *d_out, void *stream_v) {
    // `rows` independent argmaxes over `n` contiguous floats each, written to
    // `d_out[0..rows)`.  One launch, so a speculative verify's per-row picks
    // cost ONE blocking D2H instead of `rows` of them: measured on 27B MTP, the
    // five per-row calls each waited 6.5 ms on a 9 us kernel (36 calls, 235 ms
    // of pure host wait per run).
    if (n <= 0 || rows <= 0) return (int)cudaSuccess;
    cudaStream_t stream = (cudaStream_t)stream_v;
    k_argmax_batch<<<dim3(1, rows), 256, 0, stream>>>(d_x, n, d_out);
    return (int)cudaGetLastError();
}

// `q35_launch_argmax` plus the top-1 softmax probability.  One block, two
// reductions: the (value, index) argmax exactly as `k_argmax` does it, then a
// sum of `exp(x - best)` over the same row.  The shift keeps the sum finite for
// any logit range, and it is accumulated in `double` so a 248k-element row does
// not lose the tail to f32 rounding.
__global__ void k_argmax_conf(const float *__restrict__ d_x, int n, int *d_out) {
    __shared__ float s_val[256];
    __shared__ int s_idx[256];
    __shared__ double s_sum[256];
    const int tid = threadIdx.x;
    float best = -CUDART_INF_F;
    int best_i = 0;
    for (int i = tid; i < n; i += blockDim.x) {
        const float v = d_x[i];
        if (v > best) { best = v; best_i = i; }
    }
    s_val[tid] = best;
    s_idx[tid] = best_i;
    __syncthreads();
    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            if (s_val[tid + stride] > s_val[tid] ||
                (s_val[tid + stride] == s_val[tid] && s_idx[tid + stride] < s_idx[tid])) {
                s_val[tid] = s_val[tid + stride];
                s_idx[tid] = s_idx[tid + stride];
            }
        }
        __syncthreads();
    }
    const float top = s_val[0];
    // Second pass: sum of exp(x - top).  Same striding, so every element is
    // visited exactly once.
    double acc = 0.0;
    for (int i = tid; i < n; i += blockDim.x) {
        acc += (double)__expf(d_x[i] - top);
    }
    s_sum[tid] = acc;
    __syncthreads();
    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) s_sum[tid] += s_sum[tid + stride];
        __syncthreads();
    }
    if (tid == 0) {
        d_out[0] = s_idx[0];
        d_out[1] = __float_as_int(top);
        const float sum = (float)s_sum[0];
        d_out[2] = __float_as_int(sum);
    }
}

extern "C" int q35_launch_argmax_conf(const float *d_x, int n, int *d_out, void *stream_v) {
    if (n <= 0) return (int)cudaSuccess;
    cudaStream_t stream = (cudaStream_t)stream_v;
    k_argmax_conf<<<1, 256, 0, stream>>>(d_x, n, d_out);
    return (int)cudaGetLastError();
}

extern "C" int q35_launch_argmax(const float *d_x, int n, int *d_out, void *stream_v) {
    if (n <= 0) return (int)cudaSuccess;
    cudaStream_t stream = (cudaStream_t)stream_v;
    if (n <= kArgmaxThreads * 8) {
        k_argmax<<<1, 256, 0, stream>>>(d_x, n, d_out);
        return (int)cudaGetLastError();
    }
    // ONE scratch allocation, made on the first call and never resized: this
    // launcher also runs INSIDE a CUDA graph capture (decode replays `argmax`
    // as a graph node), and `cudaMalloc` during a capture is illegal -- it
    // aborts the graph, exactly as `ensureRopeDyn`'s comment records for the
    // rope channel.  Fixed size for `kArgmaxMaxBlocks`, allocated once.
    static float *s_pv = nullptr;
    static int *s_pi = nullptr;
    if (s_pv == nullptr) {
        if (cudaMalloc((void **)&s_pv, sizeof(float) * kArgmaxMaxBlocks) != cudaSuccess) {
            s_pv = nullptr;
            return (int)cudaGetLastError();
        }
        if (cudaMalloc((void **)&s_pi, sizeof(int) * kArgmaxMaxBlocks) != cudaSuccess) {
            cudaFree(s_pv);
            s_pv = nullptr;
            return (int)cudaGetLastError();
        }
    }
    int blocks = (n + kArgmaxThreads * 4 - 1) / (kArgmaxThreads * 4);
    if (blocks > kArgmaxMaxBlocks) blocks = kArgmaxMaxBlocks;
    if (blocks < 1) blocks = 1;
    k_argmax_partial<<<blocks, kArgmaxThreads, 0, stream>>>(d_x, n, s_pv, s_pi, blocks);
    k_argmax_final<<<1, kArgmaxThreads, 0, stream>>>(s_pv, s_pi, blocks, d_out);
    return (int)cudaGetLastError();
}

// ---------------------------------------------------------------------------
// The per-step control block (xd.h XD_CTL_*).
//
// One 1-thread kernel per step writes the whole block on the LAUNCH STREAM, so
// stream order alone makes it visible to every node of a replayed graph -- no
// event, no `cudaMemcpyAsync` whose address would be baked into a node.
//
// Why a kernel and not an async H2D: an H2D launched *during* a capture would be
// recorded into the graph and replayed with the capture token's value.  A
// `cudaMemcpyAsync` issued outside the capture is fine, but it costs a driver
// round trip; a 1-thread kernel costs ~3 us and composes with the graph's own
// stream.  (This is the same argument `k_rope64_setenv` records in rope.cu.)
__global__ void k_xd_ctl_publish(int *__restrict__ ctl, int pos, int base_bits,
                                 int kv_slot, int attn_slot, int rows) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    ctl[XD_CTL_POS] = pos;
    ctl[XD_CTL_BASE_BITS] = base_bits;
    ctl[XD_CTL_KV_SLOT] = kv_slot;
    ctl[XD_CTL_ATTN_SLOT] = attn_slot;
    ctl[XD_CTL_ROWS] = rows;
}

extern "C" int q35_launch_xd_ctl_publish(void *d_ctl, int pos, int base_bits,
                                         int kv_slot, int attn_slot, int rows,
                                         void *stream_v) {
    if (d_ctl == nullptr) return (int)cudaErrorInvalidValue;
    k_xd_ctl_publish<<<1, 1, 0, (cudaStream_t)stream_v>>>(
        (int *)d_ctl, pos, base_bits, kv_slot, attn_slot, rows);
    return (int)cudaGetLastError();
}
