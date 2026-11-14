// Full-attention decode for one token, in ONE launch: scores(q, K) -> softmax
// -> P.V -> sigmoid gate.  The KV cache lives on the device, so this is the
// slot that makes device-resident KV worthwhile -- the alternative is shipping
// (slot+1)*kv_dim floats across PCIe per layer per token.
//
// NUMERIC CONTRACT vs src/elementwise.zig `attnDecodeKernel`
// ----------------------------------------------------------
// That function is the reference and this kernel follows its structure exactly:
//
//   for h in heads:                       (this kernel: blockIdx.x == h)
//     kvh = h / gqa
//     scores[t] = (sum_j q[h*hd+j] * kcache[t*kv_dim + kvh*hd + j]) * scale
//     softmax(scores[0..slot])            (in place, CPU `softmax`)
//     accv[j] = sum_t scores[t] * vcache[t*kv_dim + kvh*hd + j]
//     gout[h*hd+j] = accv[j]
//   gout[i] *= sigmoid(gate[i])           (ONE pass over all heads, after)
//
// The three things that decide the result:
//
//   1. The dot product over j.  CPU is a serial `for (qs, ks) |a,b| acc += a*b`,
//      i.e. strictly increasing j.  Here that is a block-wide sum over j; using
//      a shared-memory/warp tree would reassociate and change the last ULPs.
//      hd is 128 for every Qwen3.5 variant, so ONE warp per head with a
//      shuffle-free approach is not possible without reassociation.  Instead:
//      thread t handles time step t (t = slot index), and the j-sum inside it is
//      serial in increasing j -- exactly the CPU order, no tree anywhere.
//   2. The softmax.  `elementwise.softmax` is: max, then exp, then sum in index
//      order, then divide.  Reproduced per the same "thread per element, reduce
//      serially" pattern below.
//   3. The gate multiply, on every head's output, AFTER all heads are written
//      (the CPU does it in `mulSigmoidVec` on the whole gout slice).  Keeping
//      that as a separate kernel preserves it.
//
// BIT-EXACTNESS: not claimed, and not needed -- nothing here is carried
// across tokens (gout is re-derived each token from the current scores), so an
// ULP cannot accumulate.  Measurement (RTX 4090 D sm_89, harness
// /tmp/slot_harness.cu vs an independent C transcription of
// `elementwise.attnDecodeKernel`):
//
//   attn probs  n=6   (slot=5,  heads=4, gqa=2, hd=8)   max_abs=0        0 ULP
//   attn gout   n=32  (slot=5,  heads=4, gqa=2, hd=8)   max_abs=5.96e-08 2 ULP
//   attn probs  n=64  (slot=63, heads=8, gqa=4, hd=16)  max_abs=1.86e-09 1 ULP
//   attn gout   n=128 (slot=63, heads=8, gqa=4, hd=16)  max_abs=2.24e-08 3 ULP
//
// A SHARED-MEMORY TRAP worth remembering (this harness caught it): static
// `__shared__` scalars and the dynamic `extern __shared__` block share one
// region, statics first.  A separate `__shared__ float s_max` therefore
// aliases `s_scores[0]` and silently destroys row 0's score -- the
// probabilities came out near-uniform with no error reported.  The max/sum are
// carved out of the dynamic block below; keep them there.
//
// LAYOUT ASSUMPTION (checked host-side): kcache/vcache rows are CONTIGUOUS,
// `slot+1` rows of kv_dim, i.e. `Buf.span(0, (slot+1)*kv_dim)` of the layer's
// cache.  `d_att` is (slot+1) floats of scratch reused by each head.

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <math_constants.h>

// Upper bound for the dynamic shared-memory scratch (`scores`, 1 float per
// written row).  Qwen3.5 max_seq is well under this; a longer sequence would
// exceed the 48 KB default smem budget, so the launcher checks and refuses
// rather than silently truncating the attention window.
#define Q35_ATTN_MAX_ROWS 8192

// Chunk-wide causal attention for PREFILL: one launch for a whole chunk instead
// of one launch per token.
//
// The decode kernel is `<<<heads, 128>>>` and handles exactly one token, so a
// 525-token prefill issues 16 layers x 525 = 8400 launches (`PROF-GPU` on 27B:
// 15.1 ms, 19 us each).  This merges them into 16 launches.
//
// THE PARALLELISM PROBLEM, measured twice
// ---------------------------------------
// The first version of this kernel was `<<<heads, 256>>>` with token r handled
// inside the block in a serial r-loop.  It was SLOWER than the per-token form:
//
//   attn_chunk  calls/tok=2   dev_us/tok=31455   avg_us/call=20970   (6.8%)
//   vs  attn_decode  8400 calls x 19 us = 15070 us
//
// i.e. 40 us per (head, row) instead of 19.  Reason: 24 blocks cannot fill 114
// SMs.  The per-token form had the same 24-block shape, but it got 525
// INDEPENDENT launches, so a launch's tail overlapped the next launch's head and
// the r-loop's six `__syncthreads()` per row were hidden.  Serialising the r-loop
// inside one launch put those barriers on the critical path with nothing to
// overlap them.
//
// So the rows must be PARALLEL, not serial: `<<<dim3(rows, heads), 128>>>`, one
// block per (token, head).  525 x 24 = 12600 blocks -- the GPU is saturated and
// each block does exactly the work the per-token launch did.  Total arithmetic
// is unchanged (both forms compute the full causal prefix per row, so both are
// O(n^2) in the chunk); only the launch/fill pattern changes.
//
// WHY THIS IS STILL SAFE WITHOUT A CACHE-FILL BARRIER between rows: the caller
// writes EVERY row's k/v to the cache before calling (see `forwardFullChunk`),
// and row r reads `[0, slot0+r]` -- a prefix of what is already there.  No block
// reads a row another block writes, so blocks are independent.
//
// NUMERIC CONTRACT: identical to the per-token kernel, term for term -- thread t
// owns history row t and its j-sum runs in increasing j (the CPU order), the
// softmax reduces in increasing t, V is reduced in increasing t.  Gated against
// the per-row launches in `/tmp/gate_attn_chunk.cu`: bit-identical.

// The largest `rows` that `q35_launch_attn_chunk` will accept at this `slot0`.
// `s_scores` must hold the longest prefix any row sees, i.e. slot0+rows floats,
// so this is bounded by the 48 KB dynamic shared budget.
#define Q35_ATTN_CHUNK_SMEM_MAX (48 * 1024 / (int)sizeof(float))

// Extra bytes the chunk kernel needs in its dynamic block beyond the scores and
// the two scalars: the q8_0 copy of Q (`hd/4` int) plus two floats plus one
// per-32-element-block Q sum (`hd/32` int) for q4_0's offset correction.
// f32 and f16 request the plain size.
static inline size_t q35_attn_chunk_extra_smem(int hd, int kv_kind)
{
    if (kv_kind != 1 && kv_kind != 2) return 0;
    return (size_t)(hd / 4) * sizeof(int) + 2 * sizeof(float) + (size_t)(hd / 32) * sizeof(int);
}

// ---------------------------------------------------------------------------
// Quantized-KV dequant, on read.  When the cache is not f32 the K/V rows are
// the byte-packed layouts the CPU writes (`q35_kv_quant_*`, kv.c) and every
// kernel below decodes an element as it reads it.  `kv_kind` is -1 for f32
// (the original code path, unchanged) and 0/1/2 for f16/q8_0/q4_0, matching
// the `kind` the quantizer was launched with.
//
// `q35_kv_f16_to_f32` mirrors the CPU's `kv_f16_to_f32` (kv.c) bit for bit --
// the SAME manual subnormal/normal/inf decode, so a quantized element decoded
// here equals the CPU's dequant of the same byte.  `q35_kv_block_elem`
// decodes element `k` (0..31) of one 32-element block, expression-for-
// expression with `q35_kv_dequant_q8_0` / `q35_kv_dequant_q4_0` /
// `q35_kv_dequant_f16`.
__device__ __forceinline__ float q35_kv_f16_to_f32(unsigned short h)
{
    const unsigned sign = (unsigned)(h & 0x8000u) << 16;
    const unsigned exp = (unsigned)(h >> 10) & 0x1Fu;
    const unsigned mant = (unsigned)(h & 0x3FFu);
    unsigned out;
    if (exp == 0) {
        if (mant == 0) return 0.0f;
        unsigned m = mant;
        unsigned lz = 0;
        while (m < 0x200u) { m <<= 1; lz++; }
        out = sign | ((112u - lz) << 23) | ((m & 0x1FFu) << 14);
    } else if (exp == 31) {
        out = sign | 0x7F800000u | ((mant << 14) & 0x7FFFFFu);
    } else {
        out = sign | ((exp + 112u) << 23) | (mant << 13);
    }
    return __uint_as_float(out);
}

// Byte stride of one 32-element KV block in `kind`.
__device__ __forceinline__ int q35_kv_stride(int kind)
{
    return (kind == 0) ? 64 : (kind == 1) ? 34 : 18;
}

// Scalar decode of element `k` (0..31) of one 32-element block.  This is the
// REFERENCE: expression-for-expression with `q35_kv_dequant_q8_0` /
// `q35_kv_dequant_q4_0` / `q35_kv_dequant_f16` in src/kernels/x86/kv.c, which
// `kv_tests.zig` pins against the Zig oracle.  The kernels below keep using it
// wherever a single element is what they need; the block-level fast paths are
// separate and must produce the same bits.
__device__ __forceinline__ float q35_kv_block_elem(const unsigned char *block, int k, int kind)
{
    if (kind == 0) {
        const unsigned short h = (unsigned short)block[2 * k] |
                                 ((unsigned short)block[2 * k + 1] << 8);
        return q35_kv_f16_to_f32(h);
    }
    const unsigned short h = (unsigned short)block[0] |
                             ((unsigned short)block[1] << 8);
    const float d = q35_kv_f16_to_f32(h);
    if (kind == 1) {
        return __fmul_rn(d, (float)(signed char)block[2 + k]);
    }
    const unsigned char q = (k < 16) ? (block[2 + k] & 0x0Fu)
                                      : ((block[2 + (k - 16)] >> 4) & 0x0Fu);
    return __fmul_rn(d, (float)((int)q - 8));
}

// The scalar per-element decode above is the reference (it is what the CPU
// dequant computes, and `kv_tests.zig` pins the two together).  It is NOT the
// fast path: one bit-extract + branch + `__uint_as_float` per element cost
// 4.4x the f32 attention in prefill (measured 2026-11, q8_0, 1309 tokens:
// 1275 ms vs 289 ms), because the chunk kernel walks j serially in one thread
// and cannot amortise anything.
//
// Two block-level fast paths sit on top of it, both shaped after
// llama.cpp's CUDA flash attention (ggml/src/ggml-cuda/fattn-common.cuh):
//
//   * `q35_kv_dot32_q8` -- one 32-element block against a q8_0-quantized Q,
//     as an integer dot product: q8_0 K is int8 x int8, q4_0 K is nibble x
//     int8.  `dp4a` is 4 MACs per instruction, so the arithmetic that used to
//     be 32 float mul+add plus 32 decodes becomes 8 integer ops plus one
//     scale.  This is `vec_dot_fattn_vec_KQ_q[48]_0` (fattn-common.cuh:149,
//     :304).  The q4_0 `-8` offset needs the sum of the 32 Q quants, which is
//     the `Q_s` side channel the caller accumulates once per Q row.
//
//   * `q35_kv_dequant4` -- four CONSECUTIVE elements of one block, decoded
//     together.  The V side multiplies each block by a single probability, so
//     the cost there is per-element decode plus one float op, not a dot
//     product; this is llama.cpp's `dequantize_V_q4_0<T, 4>` (dequantize_V_*
//     in fattn-common.cuh, `ne == 2 || ne == 4`).  Loading 4 bytes at once and
//     turning them into four floats with one int->float4 conversion keeps the
//     value bit-identical to the scalar path.
//
// f16 is handled by `q35_kv_ld4` alone: `half2`->float2 IS the decode.

// 4 CONSECUTIVE elements starting at `elem` of the block at `blk` (elem%4==0).
// Writes to `out[0..4]`.  Exactly the arithmetic of `q35_kv_block_elem` on
// each of the four, including the f16 scale round-trip.
__device__ __forceinline__ void q35_kv_dequant4(const unsigned char *blk, int elem,
                                                int kind, float *out)
{
    if (kind == 0) {
        // 4 halves = 8 B at byte `2*elem` of the block.  Not assumed aligned
        // (the block stride is 64 B, so the base IS aligned, but `elem` +2
        // offsets it by 2 B); `memcpy` keeps the same codegen without the UB.
        unsigned short h4[4];
        __builtin_memcpy(h4, blk + 2 * elem, sizeof(h4));
        const float2 a = __half22float2(*(const half2 *)&h4[0]);
        const float2 b = __half22float2(*(const half2 *)&h4[2]);
        out[0] = a.x;
        out[1] = a.y;
        out[2] = b.x;
        out[3] = b.y;
        return;
    }
    const half hs = __ushort_as_half((unsigned short)blk[0] | ((unsigned short)blk[1] << 8));
    const float d = __half2float(hs);
    if (kind == 1) {
        // 4 int8 at a time: one 32-bit load, then four signed byte -> float.
        int v;
        __builtin_memcpy(&v, blk + 2 + elem, sizeof(v));
        const char *c = (const char *)&v;
        out[0] = __fmul_rn(d, (float)c[0]);
        out[1] = __fmul_rn(d, (float)c[1]);
        out[2] = __fmul_rn(d, (float)c[2]);
        out[3] = __fmul_rn(d, (float)c[3]);
        return;
    }
    // q4_0: elements 0..15 are the low nibbles of bytes 2..17, 16..31 the high
    // ones.  Four consecutive elements therefore live in 4 bytes of the SAME
    // half; `elem/2` is the byte and the half selects low or high.
    const int half_sel = elem / 16;
    const int base = 2 + (elem % 16);
    unsigned v;
    __builtin_memcpy(&v, blk + base, 4);
    v = (v >> (4 * half_sel)) & 0x0F0F0F0Fu;
    out[0] = __fmul_rn(d, (float)((int)(v & 0xFFu) - 8));
    out[1] = __fmul_rn(d, (float)((int)((v >> 8) & 0xFFu) - 8));
    out[2] = __fmul_rn(d, (float)((int)((v >> 16) & 0xFFu) - 8));
    out[3] = __fmul_rn(d, (float)((int)((v >> 24) & 0xFFu) - 8));
}

// `dp4a`: 4 signed int8 MACs in one instruction (`__dp4a` on sm_61+; every
// target this project builds for has it).
__device__ __forceinline__ int q35_dp4a(int a, int b, int c)
{
    return __dp4a(a, b, c);
}

// Integer dot product of one 32-element K block against 8 packed Q words
// (`q8_0`-quantized, 4 int8 each).  `qsum` is the sum of the 32 Q quants,
// needed only by q4_0's `-8` offset.  Returns the dot of the QUANTIZED values
// times K's block scale; the caller multiplies by Q's scale (a constant per Q
// row, so folding it outside the t loop is exact).
//
// Corresponds to vec_dot_fattn_vec_KQ_q4_0 / _q8_0 (f2...:149 / :304), with
// the same algebra:
//   q8_0: sum_i Kq[i]*Qq[i]
//   q4_0: sum_i (n[i]) * Qq[i]  where element = (n - 8) * d  ->  sum_i n*Q - 8*sum_i Q
__device__ __forceinline__ int q35_kv_dot32_q8(const unsigned char *blk, const int *q8, int qsum,
                                               int kind)
{
    int sumi = 0;
    if (kind == 1) {
        // The 32 quants start at byte 2 of the block, which is NOT 4-byte
        // aligned (block stride is 34 B), so the words are loaded with
        // `__builtin_memcpy` -- a plain `(const int *)` dereference here is what
        // produced "misaligned address" for every q8_0 run (2026-11).
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            int u;
            __builtin_memcpy(&u, blk + 2 + 4 * i, sizeof(u));
            sumi = q35_dp4a(u, q8[i], sumi);
        }
        return sumi;
    }
    // q4_0: the 16 packed bytes are 8 words.  Byte k holds element k in its low
    // nibble and element 16+k in its high nibble, so word i covers elements
    // 4i..4i+3 (low) and 16+4i..16+4i+3 (high) -- Q words i and 4+i, because
    // the q8_0 Q buffer is laid out as consecutive 4-element groups.
    // Byte 2 is likewise unaligned; same `memcpy` load.
    int sumi4 = 0;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        int w;
        __builtin_memcpy(&w, blk + 2 + 4 * i, sizeof(w));
        const int lo = w & 0x0F0F0F0F;
        const int hi = (w >> 4) & 0x0F0F0F0F;
        sumi4 = q35_dp4a(hi, q8[4 + i], q35_dp4a(lo, q8[i], sumi4));
    }
    return sumi4 - 8 * qsum;
}

// Quantize one head's Q into q8_0 words and accumulate each 32-element block's
// quant sum.  Inlined: it runs once per (row, head) by thread 0, and the call
// overhead of `noinline` cost the packed formats 18% (2026-11).
__device__ __forceinline__ void q35_kv_prep_q(const float *__restrict__ qs, int hd,
                                              int *__restrict__ s_q8, int *__restrict__ s_qblk,
                                              float *__restrict__ s_qpar)
{
    float amax = 0.0f;
    for (int j = 0; j < hd; ++j) amax = fmaxf(amax, fabsf(qs[j]));
    const float d = amax > 0.0f ? amax / 127.0f : 1.0f;
    const float inv = amax > 0.0f ? 1.0f / d : 0.0f;
    for (int w = 0; w < hd / 4; ++w) s_q8[w] = 0;
    for (int j = 0; j < hd; ++j) {
        const float v = qs[j] * inv;
        int q = (int)(v + (v >= 0.0f ? 0.5f : -0.5f));
        q = q > 127 ? 127 : (q < -128 ? -128 : q);
        s_q8[j >> 2] |= (q & 0xFF) << ((j & 3) * 8);
    }
    // q4_0's offset correction subtracts 8 * (sum of the 32 Q quants IN THAT
    // BLOCK), so the sum is kept PER BLOCK.  A whole-row sum is wrong: measured
    // 2026-11, it put every q4_0 K dot off by 8*(row_sum - block_sum) and the
    // model diverged on the first token while q8_0 (no correction) stayed exact.
    for (int b = 0; b < hd / 32; ++b) {
        int s = 0;
        for (int w = 0; w < 8; ++w) {
            const signed char *c = (const signed char *)&s_q8[b * 8 + w];
            s += c[0] + c[1] + c[2] + c[3];
        }
        s_qblk[b] = s;
    }
    s_qpar[0] = d;
}

// One block per (row, head).  `d_q`/`d_gate`/`d_gout` are indexed by
// (token, head): row r at `r * heads * hd + h * hd`, matching the chunk buffers
// `qK`/`gateK`/`goutK` that engine.zig fills.
template <int KIND>
__global__ void q35_attn_chunk_kernel(
    const float *__restrict__ d_q,
    const float *__restrict__ d_gate,
    const float *__restrict__ d_kcache,
    const float *__restrict__ d_vcache,
    float *__restrict__ d_gout,
    int slot0, int heads, int gqa, int hd, int kv_dim, float scale)
{
    const int r = blockIdx.x;               // token within the chunk
    const int h = blockIdx.y;               // head
    const int tid = threadIdx.x;
    const int nt = blockDim.x;
    if (h >= heads) return;

    const int kvh = h / gqa;
    const int n = slot0 + r + 1;            // causal prefix, inclusive
    const int row_len = heads * hd;
    const float *__restrict__ qs = d_q + (size_t)r * row_len + (size_t)h * hd;

    // Quantized-cache geometry: row `row_b` bytes, 32-elem block `stride` bytes.
    // `KIND` is a compile-time parameter, so every `if (kv_kind ...)` below is
    // pruned per instantiation.
    constexpr int kv_kind = KIND;
    const unsigned char *kcache_b = (const unsigned char *)d_kcache;
    const unsigned char *vcache_b = (const unsigned char *)d_vcache;
    const size_t stride = q35_kv_stride(kv_kind);
    const size_t row_b = (size_t)(kv_dim / 32) * stride;

    extern __shared__ float s_scores[];
    float *const s_max = &s_scores[n];
    float *const s_sum = &s_scores[n + 1];

    // Q in the K cache's integer domain, computed ONCE per (row, head) and
    // reused by all n scores -- this is llama.cpp's `quantize_q8_1_to_shared`
    // (fattn-vec.cuh:180) hoisted out of the t loop.
    //
    // Only the block formats need it: f16 multiplies `half2` directly and f32
    // needs nothing.  The words live in the SAME dynamic block as the scores
    // (static and dynamic `__shared__` share one region -- see the aliasing
    // note at the top of this file), placed after the two scalars.
    constexpr bool int_dot = KIND == 1 || KIND == 2;
    int *const s_q8 = (int *)(s_sum + 1);
    float *const s_qpar = (float *)(s_q8 + hd / 4);
    int *const s_qblk = (int *)(s_qpar + 2);
    // f32 runs none of this and must not pay for it: measured 2026-11, an
    // unconditional barrier here cost f32 prefill 9% (1.49 -> 1.63 s on
    // 1309 tokens) for code it never runs.
    // ALL 128 threads run this, so the quantization itself is confined to
    // thread 0: `amax` would otherwise be a per-thread partial (every thread
    // needs the block-wide maximum) and the `s_q8[w] |= ...` read-modify-write
    // would race.  Measured 2026-11: with the racy version q8_0 matched the
    // reference but q4_0 diverged -- its `-8 * sum(Q)` correction amplifies the
    // half-written words.
    if (int_dot and tid == 0) {
        q35_kv_prep_q(qs, hd, s_q8, s_qblk, s_qpar);
    }
    // Only the block formats read shared Q scaffolding.
    if (int_dot) __syncthreads();

    for (int t = tid; t < n; t += nt) {
        float acc = 0.0f;
        if (kv_kind < 0) {
            const float *__restrict__ ks = d_kcache + (size_t)t * kv_dim + (size_t)kvh * hd;
            for (int j = 0; j < hd; ++j) acc = __fadd_rn(acc, __fmul_rn(qs[j], ks[j]));
        } else {
            // `kvh * hd` is a multiple of 32 (hd is), so block `b` of this head
            // starts at block `bbase + b` of the row.
            const unsigned char *krow = kcache_b + (size_t)t * row_b +
                                        (size_t)(kvh * hd / 32) * stride;
            if (int_dot) {
                const float qd = s_qpar[0];
                for (int b = 0; b < hd / 32; ++b) {
                    const unsigned char *blk = krow + (size_t)b * stride;
                    // K's block scale, decoded exactly as the scalar path does.
                    const float d = q35_kv_f16_to_f32((unsigned short)blk[0] |
                                                      ((unsigned short)blk[1] << 8));
                    const int isum = q35_kv_dot32_q8(blk, s_q8 + b * 8, s_qblk[b], kv_kind);
                    acc = __fadd_rn(acc, __fmul_rn(d * qd, (float)isum));
                }
            } else {
                // f16 cache: the bytes ARE the values, so an 8-half read feeds
                // 8 MACs with no scale to apply.
                //
                // EIGHT accumulators: the dot is a LATENCY chain, not a
                // throughput problem.  With 4 chains (and 64 steps each) f16
                // measured 0.046 s against f32's 0.023 s for the same 517-token
                // prefill, even though f16 issues fewer instructions -- every
                // step waited on the previous `__fadd_rn`.
                float a[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
                // Walk the row with a POINTER, and read through `half2*`: with
                // `__builtin_memcpy` from a `const unsigned char *` ptxas could
                // not prove the 8-byte alignment and emitted 32 separate
                // `LDG.E.U8` per block (measured in SASS).  A block is 64 B of
                // contiguous halves starting at a `stride`-aligned address, so
                // every 2-half piece here is 4-byte aligned.
                const half2 *kp = (const half2 *)krow;
                for (int j = 0; j < hd; j += 8) {
                    const int off = (j & 31) >> 1; // half2 index inside the block
                    const half2 *h2 = kp + off;
                    const float2 f0 = __half22float2(h2[0]);
                    const float2 f1 = __half22float2(h2[1]);
                    const float2 f2 = __half22float2(h2[2]);
                    const float2 f3 = __half22float2(h2[3]);
                    if (off == 12) kp += (stride >> 2); // 32 halves = 16 half2
                    a[0] = __fadd_rn(a[0], __fmul_rn(qs[j + 0], f0.x));
                    a[1] = __fadd_rn(a[1], __fmul_rn(qs[j + 1], f0.y));
                    a[2] = __fadd_rn(a[2], __fmul_rn(qs[j + 2], f1.x));
                    a[3] = __fadd_rn(a[3], __fmul_rn(qs[j + 3], f1.y));
                    a[4] = __fadd_rn(a[4], __fmul_rn(qs[j + 4], f2.x));
                    a[5] = __fadd_rn(a[5], __fmul_rn(qs[j + 5], f2.y));
                    a[6] = __fadd_rn(a[6], __fmul_rn(qs[j + 6], f3.x));
                    a[7] = __fadd_rn(a[7], __fmul_rn(qs[j + 7], f3.y));
                }
                acc = __fadd_rn(
                    __fadd_rn(__fadd_rn(a[0], a[1]), __fadd_rn(a[2], a[3])),
                    __fadd_rn(__fadd_rn(a[4], a[5]), __fadd_rn(a[6], a[7])));
            }
        }
        s_scores[t] = __fmul_rn(acc, scale);
    }
    __syncthreads();

    if (tid == 0) {
        float m = -CUDART_INF_F;
        for (int t = 0; t < n; ++t) m = fmaxf(m, s_scores[t]);
        *s_max = m;
    }
    __syncthreads();
    for (int t = tid; t < n; t += nt) s_scores[t] = expf(s_scores[t] - *s_max);
    __syncthreads();
    if (tid == 0) {
        float s = 0.0f;
        for (int t = 0; t < n; ++t) s += s_scores[t];
        *s_sum = s;
    }
    __syncthreads();
    for (int t = tid; t < n; t += nt) s_scores[t] = s_scores[t] / *s_sum;
    __syncthreads();

    float *__restrict__ gout = d_gout + (size_t)r * row_len + (size_t)h * hd;
    const float *__restrict__ gt = d_gate + (size_t)r * row_len + (size_t)h * hd;
    // f32 keeps its own loop, untouched by the quantized variants.  Sharing one
    // loop body measurably slowed it (2026-11: 1309-token prefill 1.49 -> 1.63 s
    // with the packed cases folded in), and it gains nothing from them.
    if (kv_kind < 0) {
        for (int j = tid; j < hd; j += nt) {
            float acc = 0.0f;
            for (int t = 0; t < n; ++t) {
                const float w = s_scores[t];
                const float vv = d_vcache[(size_t)t * kv_dim + (size_t)kvh * hd + j];
                acc = __fadd_rn(acc, __fmul_rn(w, vv));
            }
            const float g = gt[j];
            gout[j] = acc * (1.0f / (1.0f + expf(-g)));
        }
    } else {
        // Four CONSECUTIVE j per thread, so one `q35_kv_dequant4` feeds four
        // accumulators instead of being decoded and then 3/4 discarded.  That
        // is llama.cpp's V layout (`dequantize_V<T, 4>`, fattn-common.cuh).
        //
        // ORDER: each of the four lanes keeps its own `t`-increasing sum, so
        // every output element sees exactly the CPU's accumulation order --
        // grouping by j changes which thread owns an element, never the order
        // of its adds.
        //
        // MEASURED: the first version kept one element per thread and decoded
        // the group anyway; f16 prefill attention then cost 0.056 s against
        // f32's 0.028 s on a 517-token prompt, i.e. the cache that halves the
        // bytes ran at half speed.
        const int j0 = kvh * hd + tid * 4;
        const int jstep = nt * 4;
        const int jend = kvh * hd + hd;
        for (int base = j0; base < jend; base += jstep) {
            // Recompute from `base` every step: `jstep` is a multiple of 32 in
            // every configuration used (nt=128 -> 512), but deriving the byte
            // offsets from the absolute index keeps this right even if it is
            // not.
            const int b = base / 32;
            const int go = (base % 32) & ~3;
            float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
            for (int t = 0; t < n; ++t) {
                const float w = s_scores[t];
                const unsigned char *vrow = vcache_b + (size_t)t * row_b;
                float v4[4];
                q35_kv_dequant4(vrow + (size_t)b * stride, go, kv_kind, v4);
                a0 = __fadd_rn(a0, __fmul_rn(w, v4[0]));
                a1 = __fadd_rn(a1, __fmul_rn(w, v4[1]));
                a2 = __fadd_rn(a2, __fmul_rn(w, v4[2]));
                a3 = __fadd_rn(a3, __fmul_rn(w, v4[3]));
            }
            const int jj = base - kvh * hd;
            gout[jj] = a0 * (1.0f / (1.0f + expf(-gt[jj])));
            gout[jj + 1] = a1 * (1.0f / (1.0f + expf(-gt[jj + 1])));
            gout[jj + 2] = a2 * (1.0f / (1.0f + expf(-gt[jj + 2])));
            gout[jj + 3] = a3 * (1.0f / (1.0f + expf(-gt[jj + 3])));
        }
    }
}

extern "C" int q35_launch_attn_chunk(const float *d_q, const float *d_gate,
                                     const float *d_kcache, const float *d_vcache,
                                     float *d_gout, int slot0, int rows, int heads,
                                     int gqa, int hd, int kv_dim, float scale,
                                     int kv_kind, void *stream)
{
    if (rows <= 0 || heads <= 0 || hd <= 0) return (int)cudaSuccess;
    if (slot0 < 0 || slot0 + rows > Q35_ATTN_MAX_ROWS) return cudaErrorInvalidValue;
    if (slot0 + rows > Q35_ATTN_CHUNK_SMEM_MAX) return cudaErrorInvalidValue;
    // Scores + the two scalars + (block formats) the q8_0 copy of Q.  Reserved
    // unconditionally would overstate the f32/f16 requirement, so it follows
    // `kv_kind`.
    const size_t smem = (size_t)(slot0 + rows + 2) * sizeof(float) +
                        q35_attn_chunk_extra_smem(hd, kv_kind);
    // One instantiation per KV family.  Templating on the kind is what keeps the
    // f32 and f16 paths free of the quantized formats' shared-memory prep: with
    // a run-time `kv_kind` the dead `int_dot` block still changed how the f32
    // score loop loaded `qs` and cost it 54% of its attention time (measured
    // 2026-11), while forcing the prep out of line cost the packed formats 18%.
    dim3 grid((unsigned)rows, (unsigned)heads);
    switch (kv_kind) {
        case 0:
            q35_attn_chunk_kernel<0><<<grid, 128, smem, (cudaStream_t)stream>>>(
                d_q, d_gate, d_kcache, d_vcache, d_gout, slot0, heads, gqa, hd, kv_dim, scale);
            break;
        case 1:
            q35_attn_chunk_kernel<1><<<grid, 128, smem, (cudaStream_t)stream>>>(
                d_q, d_gate, d_kcache, d_vcache, d_gout, slot0, heads, gqa, hd, kv_dim, scale);
            break;
        case 2:
            q35_attn_chunk_kernel<2><<<grid, 128, smem, (cudaStream_t)stream>>>(
                d_q, d_gate, d_kcache, d_vcache, d_gout, slot0, heads, gqa, hd, kv_dim, scale);
            break;
        default:
            q35_attn_chunk_kernel<-1><<<grid, 128, smem, (cudaStream_t)stream>>>(
                d_q, d_gate, d_kcache, d_vcache, d_gout, slot0, heads, gqa, hd, kv_dim, scale);
            break;
    }
    return (int)cudaGetLastError();
}

extern "C" int q35_attn_chunk_max_rows(int slot0, int hd, int kv_kind)
{
    if (slot0 < 0 || slot0 >= Q35_ATTN_CHUNK_SMEM_MAX) return 0;
    if (hd <= 0) return 0;
    // Room left for scores once the fixed part (2 scalars + the q8_0 copy of Q
    // for the block formats) is paid.  The caller passes the same `hd`/
    // `kv_kind` it will hand to the launcher, so the two agree.
    const size_t fixed = 2 * sizeof(float) + q35_attn_chunk_extra_smem(hd, kv_kind);
    const size_t budget = (size_t)48 * 1024;
    if (fixed >= budget) return 0;
    const int room = (int)((budget - fixed) / sizeof(float)) - slot0;
    const int max_rows = Q35_ATTN_CHUNK_SMEM_MAX - slot0;
    return room < max_rows ? room : max_rows;
}


// One block per head.  Threads cooperate differently in the three phases, but
// every sum keeps the CPU's index order.
__global__ void q35_attn_decode_kernel(
    const float *__restrict__ d_q,
    const float *__restrict__ d_gate,
    const float *__restrict__ d_kcache,
    const float *__restrict__ d_vcache,
    float *__restrict__ d_att,
    float *__restrict__ d_gout,
    int slot, int heads, int gqa, int hd, int kv_dim, float scale, int kv_kind)
{
    const int h = blockIdx.x;
    const int tid = threadIdx.x;
    const int nt = blockDim.x;
    if (h >= heads) return;
    const int n = slot + 1;             // rows written so far
    const int kvh = h / gqa;
    const float *__restrict__ qs = d_q + (size_t)h * hd;
    const unsigned char *kcache_b = (const unsigned char *)d_kcache;
    const unsigned char *vcache_b = (const unsigned char *)d_vcache;
    const size_t stride = (kv_kind == 0) ? 64 : (kv_kind == 1) ? 34 : 18;
    const size_t row_b = (size_t)(kv_dim / 32) * stride;

    // Dynamic: `n` score floats plus two scalars, ONE allocation.
    //
    // The scalars MUST be carved out of the dynamic block rather than declared
    // `__shared__` on their own: static and dynamic shared memory occupy the
    // same region, with the statics placed first, so a separate
    // `__shared__ float s_max` aliases `s_scores[0]` and the max overwrites
    // row 0's score.  (This harness caught it: at slot=63/hd=16 the device
    // probabilities came out near-uniform instead of matching the CPU --
    // every score's max was being clobbered.  Smells like UB; it is not, it is
    // the documented layout, but it is silent.)
    //
    // Layout: [0, n) scores, [n] max, [n+1] sum.
    extern __shared__ float s_scores[];
    float *const s_max = &s_scores[n];
    float *const s_sum = &s_scores[n + 1];

    // ---- scores[t] = dot(q, kcache[t]) * scale -----------------------------
    // Thread t owns row t; the j-sum is SERIAL and increasing, like the CPU.
    // A thread must not spin past `n`; rows beyond it are simply not touched.
    for (int t = tid; t < n; t += nt) {
        float acc = 0.0f;
        if (kv_kind < 0) {
            const float *__restrict__ ks = d_kcache + (size_t)t * kv_dim + (size_t)kvh * hd;
            for (int j = 0; j < hd; ++j) acc = __fadd_rn(acc, __fmul_rn(qs[j], ks[j]));
        } else {
            const unsigned char *krow = kcache_b + (size_t)t * row_b;
            const int base = kvh * hd;
            for (int j = 0; j < hd; ++j) {
                const unsigned char *blk = krow + ((size_t)((base + j) / 32)) * stride;
                acc = __fadd_rn(acc, __fmul_rn(qs[j], q35_kv_block_elem(blk, (base + j) % 32, kv_kind)));
            }
        }
        s_scores[t] = __fmul_rn(acc, scale);
    }
    __syncthreads();

    // ---- softmax over scores[0..n) in place --------------------------------
    // `elementwise.softmax` order: find max (first occurrence wins on ties via
    // strict >), exp, sum in index order, divide.
    // CPU: `var max = -inf; for (x) |v| max = @max(max, v);` -- @max keeps the
    // larger operand, so on a NaN-free input this is the same as a running max.
    if (tid == 0) {
        float m = -CUDART_INF_F;
        for (int t = 0; t < n; ++t) m = fmaxf(m, s_scores[t]);
        *s_max = m;
    }
    __syncthreads();
    for (int t = tid; t < n; t += nt) s_scores[t] = expf(s_scores[t] - *s_max);
    __syncthreads();
    if (tid == 0) {
        float s = 0.0f;
        for (int t = 0; t < n; ++t) s += s_scores[t];
        *s_sum = s;
    }
    __syncthreads();
    for (int t = tid; t < n; t += nt) s_scores[t] = s_scores[t] / *s_sum;
    __syncthreads();

    // Publish this head's probabilities into the caller's scratch, matching the
    // CPU reference (which leaves them in `att`).
    for (int t = tid; t < n; t += nt) d_att[t] = s_scores[t];
    __syncthreads();

    // ---- accv[j] = sum_t scores[t] * vcache[t][j] --------------------------
    // Thread j owns output element j; the t-sum is SERIAL and increasing, which
    // is the CPU's order (its outer loop is over t, accumulating into accv[j]).
    float *__restrict__ gout = d_gout + (size_t)h * hd;
    for (int j = tid; j < hd; j += nt) {
        float acc = 0.0f;
        if (kv_kind < 0) {
            for (int t = 0; t < n; ++t) {
                const float w = s_scores[t];
                const float vv = d_vcache[(size_t)t * kv_dim + (size_t)kvh * hd + j];
                acc = __fadd_rn(acc, __fmul_rn(w, vv));
            }
        } else {
            const int base = kvh * hd + j;
            const int blk = base / 32;
            const int off = base % 32;
            for (int t = 0; t < n; ++t) {
                const float w = s_scores[t];
                const unsigned char *vrow = vcache_b + (size_t)t * row_b;
                const float vv = q35_kv_block_elem(vrow + (size_t)blk * stride, off, kv_kind);
                acc = __fadd_rn(acc, __fmul_rn(w, vv));
            }
        }
        gout[j] = acc;
    }
}


#ifndef Q35_ATTN_SMEM_DYNSLOT
#define Q35_ATTN_SMEM_DYNSLOT ((Q35_ATTN_MAX_ROWS + 2) * (int)sizeof(float))
#endif

// Device-slot variant of the kernel above: `n` comes from DEVICE memory and the
// dynamic shared block is always `Q35_ATTN_SMEM_DYNSLOT` bytes.
//
// WHY THIS EXISTS -- it is a CUDA-graph fix, not a kernel change.
// The score/softmax/PV kernel was the one node whose retarget needed
// `cudaGraphExecKernelNodeSetParams` on EVERY layer for EVERY token, because
// BOTH of its per-token inputs change:
//   * `slot` (the KV prefix length), and
//   * `shared_mem_bytes` (`(n+2)*sizeof(float)`), which the graph exec
//     snapshots at SetParams time and cannot be updated any other way.
// Measured on the 27B IQ3_XXS: with the graph enabled, ablating `attn_decode`
// took the token from 33.8 to 19.4 ms -- 14.3 ms, 43% of decode -- while the
// same kernel measured ALONE is 0.058 ms at n=300 (x24 layers = 1.4 ms).
// With `QWEN35_CUDA_GRAPH=0` the same ablation saves only 2.2 ms.  So the cost
// was never the kernel: it was ~24 `GraphExecKernelNodeSetParams` calls per
// token, each of which forces the graph exec to re-apply its node list.
//
// Fix: make `n` a pointer argument and keep `shared_mem_bytes` CONSTANT, so the
// decode graph never retargets this node at all.  The block is sized for the
// maximum window; shared memory does not limit occupancy here (the limit is 24
// blocks x 128 threads on 114 SMs).
//
// The arithmetic is the scalar-`slot` kernel's, term for term: same phase order,
// same increasing CPU index order in every reduction, same `-inf` start.
__global__ void q35_attn_decode_kernel_dyn(
    const float *__restrict__ d_q,
    const float *__restrict__ d_gate,
    const float *__restrict__ d_kcache,
    const float *__restrict__ d_vcache,
    float *__restrict__ d_att,
    float *__restrict__ d_gout,
    const int *__restrict__ d_slot,
    int heads, int gqa, int hd, int kv_dim, float scale, int kv_kind)
{
    const int h = blockIdx.x;
    const int tid = threadIdx.x;
    const int nt = blockDim.x;
    if (h >= heads) return;
    const int n = *d_slot + 1;

    const int kvh = h / gqa;
    const float *__restrict__ qs = d_q + (size_t)h * hd;
    const unsigned char *kcache_b = (const unsigned char *)d_kcache;
    const unsigned char *vcache_b = (const unsigned char *)d_vcache;
    const size_t stride = (kv_kind == 0) ? 64 : (kv_kind == 1) ? 34 : 18;
    const size_t row_b = (size_t)(kv_dim / 32) * stride;

    extern __shared__ float s_scores[];
    float *const s_max = &s_scores[n];
    float *const s_sum = &s_scores[n + 1];

    for (int t = tid; t < n; t += nt) {
        float acc = 0.0f;
        if (kv_kind < 0) {
            const float *__restrict__ ks = d_kcache + (size_t)t * kv_dim + (size_t)kvh * hd;
            for (int j = 0; j < hd; ++j) acc = __fadd_rn(acc, __fmul_rn(qs[j], ks[j]));
        } else {
            const unsigned char *krow = kcache_b + (size_t)t * row_b;
            const int base = kvh * hd;
            for (int j = 0; j < hd; ++j) {
                const unsigned char *blk = krow + ((size_t)((base + j) / 32)) * stride;
                acc = __fadd_rn(acc, __fmul_rn(qs[j], q35_kv_block_elem(blk, (base + j) % 32, kv_kind)));
            }
        }
        s_scores[t] = __fmul_rn(acc, scale);
    }
    __syncthreads();

    if (tid == 0) {
        float m = -CUDART_INF_F;
        for (int t = 0; t < n; ++t) m = fmaxf(m, s_scores[t]);
        *s_max = m;
    }
    __syncthreads();
    for (int t = tid; t < n; t += nt) s_scores[t] = expf(s_scores[t] - *s_max);
    __syncthreads();
    if (tid == 0) {
        float s = 0.0f;
        for (int t = 0; t < n; ++t) s += s_scores[t];
        *s_sum = s;
    }
    __syncthreads();
    for (int t = tid; t < n; t += nt) s_scores[t] = s_scores[t] / *s_sum;
    __syncthreads();

    for (int t = tid; t < n; t += nt) d_att[t] = s_scores[t];
    __syncthreads();

    float *__restrict__ gout = d_gout + (size_t)h * hd;
    for (int j = tid; j < hd; j += nt) {
        float acc = 0.0f;
        if (kv_kind < 0) {
            for (int t = 0; t < n; ++t) {
                const float w = s_scores[t];
                const float vv = d_vcache[(size_t)t * kv_dim + (size_t)kvh * hd + j];
                acc = __fadd_rn(acc, __fmul_rn(w, vv));
            }
        } else {
            const int base = kvh * hd + j;
            const int blk = base / 32;
            const int off = base % 32;
            for (int t = 0; t < n; ++t) {
                const float w = s_scores[t];
                const unsigned char *vrow = vcache_b + (size_t)t * row_b;
                const float vv = q35_kv_block_elem(vrow + (size_t)blk * stride, off, kv_kind);
                acc = __fadd_rn(acc, __fmul_rn(w, vv));
            }
        }
        gout[j] = acc;
    }
}

// Gate multiply, applied to the WHOLE gout after every head is written -- the
// CPU does this as one `mulSigmoidVec(gout, gate)` call.
__global__ void q35_attn_gate_kernel(float *__restrict__ d_gout,
                                     const float *__restrict__ d_gate, int n)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float g = d_gate[i];
    d_gout[i] = d_gout[i] * (1.0f / (1.0f + expf(-g)));
}

// Host-stub address of the decode-attention kernel; see `q35_func_rope64` in
// rope.cu for why the graph path needs it.  Only the score/softmax/PV kernel is
// reported: its per-token variable is `slot`.  The gate kernel that follows it
// in the same launcher has no per-token argument, so it is never retargeted.
extern "C" void *q35_func_attn_decode(void) { return (void *)&q35_attn_decode_kernel; }

extern "C" int q35_launch_attn_decode(const float *d_q, const float *d_gate,
                                      const float *d_kcache,
                                      const float *d_vcache, float *d_att,
                                      float *d_gout, int slot, int heads,
                                      int gqa, int hd, int kv_dim, float scale,
                                      int kv_kind, void *stream)
{
    if (heads <= 0 || hd <= 0) return (int)cudaSuccess;
    const int n = slot + 1;
    if (n > Q35_ATTN_MAX_ROWS) return cudaErrorInvalidValue;
    // 128 threads: enough to cover hd (128) in one pass for the per-element
    // phases, and 128 rows in flight for the per-row phases.
    // +2 floats: the two shared scalars that share the dynamic block.
    q35_attn_decode_kernel<<<heads, 128, (size_t)(n + 2) * sizeof(float), (cudaStream_t)stream>>>(
        d_q, d_gate, d_kcache, d_vcache, d_att, d_gout, slot, heads, gqa, hd,
        kv_dim, scale, kv_kind);
    int rc = (int)cudaGetLastError();
    if (rc != (int)cudaSuccess) return rc;
    const int total = heads * hd;
    const int threads = 128;
    const int blocks = (total + threads - 1) / threads;
    q35_attn_gate_kernel<<<blocks, threads, 0, (cudaStream_t)stream>>>(d_gout, d_gate, total);
    return (int)cudaGetLastError();
}

// Address of the device-slot kernel for the graph-capture path.
extern "C" void *q35_func_attn_decode_dyn(void) { return (void *)&q35_attn_decode_kernel_dyn; }

// Launch the device-slot variant: `slot` is read from DEVICE memory (`d_slot`,
// one int) and the dynamic shared block is the fixed maximum, so neither the
// argument block nor `shared_mem_bytes` changes between tokens and a captured
// node never needs retargeting.  The caller publishes `d_slot` (see
// `q35_launch_set_int1`).
extern "C" int q35_launch_attn_decode_dyn(const float *d_q, const float *d_gate,
                                          const float *d_kcache,
                                          const float *d_vcache, float *d_att,
                                          float *d_gout, const int *d_slot,
                                          int heads, int gqa, int hd, int kv_dim,
                                          float scale, int kv_kind, void *stream)
{
    if (heads <= 0 || hd <= 0) return (int)cudaSuccess;
    if (d_slot == nullptr) return (int)cudaErrorInvalidValue;
    q35_attn_decode_kernel_dyn<<<heads, 128, (size_t)Q35_ATTN_SMEM_DYNSLOT, (cudaStream_t)stream>>>(
        d_q, d_gate, d_kcache, d_vcache, d_att, d_gout, d_slot, heads, gqa, hd,
        kv_dim, scale, kv_kind);
    int rc = (int)cudaGetLastError();
    if (rc != (int)cudaSuccess) return rc;
    const int total = heads * hd;
    const int threads = 128;
    const int blocks = (total + threads - 1) / threads;
    q35_attn_gate_kernel<<<blocks, threads, 0, (cudaStream_t)stream>>>(d_gout, d_gate, total);
    return (int)cudaGetLastError();
}

// ---------------------------------------------------------------------------
// TUNED decode attention (block = one head, but the row loops are WARP-TILED).
//
// WHY: the naive kernel above assigns row `t` to thread `t`, so consecutive
// threads read `d_kcache + t*kv_dim + ...`, i.e. 4 KB apart.  Every 32-thread
// warp then touches 32 distinct cache lines per j-step and uses only `hd` of the
// `kv_dim` floats in each row: the read is uncoalesced AND over-fetched by
// `kv_dim/hd` (= 4x for the 27B, where kv_dim=1024 but hd=256).
//
// MEASURED (`/tmp/mb/mb_attn2.cu`, 24 layers with DISTINCT KV caches so the
// working set is 102-180 MB and cannot hide in the 72 MB L2): the naive kernel
// does 33-37 GB/s, i.e. 3.0 ms/token at slot=518 and 5.0 ms at slot=917.
// (A first measurement of 0.058 ms at n=300 was an L2 artifact -- the same trap
// documented for the matvec microbenches.)
//
// FIX: one warp per (row group, j-tile).  Lane `l` handles j = jt + l, so a warp
// reads ONE contiguous 128-byte segment per iteration, and rows are processed
// `warps_per_block` at a time.  The j-sum for a score is then a warp reduction;
// to keep the CPU's strict increasing-index order the reduction uses
// `__shfl_down_sync` on a tree -- which is NOT the serial order, so the scores
// differ from the reference in the last bits.
//
// This variant is therefore OPT-IN and NOT the default.  The default keeps the
// serial order because the project's gate for a non-split change is bit-exact
// token equality; it is kept here with its measurement so the trade-off is
// visible rather than rediscovered.
__global__ void q35_attn_decode_kernel_coal(
    const float *__restrict__ d_q,
    const float *__restrict__ d_gate,
    const float *__restrict__ d_kcache,
    const float *__restrict__ d_vcache,
    float *__restrict__ d_att,
    float *__restrict__ d_gout,
    int slot, int heads, int gqa, int hd, int kv_dim, float scale, int kv_kind)
{
    const int h = blockIdx.x;
    const int tid = threadIdx.x;
    const int nt = blockDim.x;
    const int warp = tid >> 5, lane = tid & 31;
    const int nwarps = nt >> 5;
    if (h >= heads) return;
    const int n = slot + 1;
    const int kvh = h / gqa;
    const float *__restrict__ qs = d_q + (size_t)h * hd;
    const unsigned char *kcache_b = (const unsigned char *)d_kcache;
    const unsigned char *vcache_b = (const unsigned char *)d_vcache;
    const size_t stride = (kv_kind == 0) ? 64 : (kv_kind == 1) ? 34 : 18;
    const size_t row_b = (size_t)(kv_dim / 32) * stride;

    extern __shared__ float s_scores[];
    float *const s_max = &s_scores[n];
    float *const s_sum = &s_scores[n + 1];

    // ---- scores: one warp per row, lanes tile j ----------------------------
    for (int t = warp; t < n; t += nwarps) {
        float acc = 0.0f;
        if (kv_kind < 0) {
            const float *__restrict__ ks = d_kcache + (size_t)t * kv_dim + (size_t)kvh * hd;
            for (int j = lane; j < hd; j += 32) acc = __fadd_rn(acc, __fmul_rn(qs[j], ks[j]));
        } else {
            const unsigned char *krow = kcache_b + (size_t)t * row_b;
            const int base = kvh * hd;
            for (int j = lane; j < hd; j += 32) {
                const unsigned char *blk = krow + ((size_t)((base + j) / 32)) * stride;
                acc = __fadd_rn(acc, __fmul_rn(qs[j], q35_kv_block_elem(blk, (base + j) % 32, kv_kind)));
            }
        }
        for (int o = 16; o > 0; o >>= 1) acc = __fadd_rn(acc, __shfl_down_sync(0xffffffffu, acc, o));
        if (lane == 0) s_scores[t] = __fmul_rn(acc, scale);
    }
    __syncthreads();

    if (tid == 0) {
        float m = -CUDART_INF_F;
        for (int t = 0; t < n; ++t) m = fmaxf(m, s_scores[t]);
        *s_max = m;
    }
    __syncthreads();
    for (int t = tid; t < n; t += nt) s_scores[t] = expf(s_scores[t] - *s_max);
    __syncthreads();
    if (tid == 0) {
        float s = 0.0f;
        for (int t = 0; t < n; ++t) s += s_scores[t];
        *s_sum = s;
    }
    __syncthreads();
    for (int t = tid; t < n; t += nt) s_scores[t] = s_scores[t] / *s_sum;
    __syncthreads();
    for (int t = tid; t < n; t += nt) d_att[t] = s_scores[t];
    __syncthreads();

    // ---- PV: warp per j-tile, rows t walked in order -----------------------
    // Lane `l` holds j = jt + l, so the warp reads one contiguous 128-byte run of
    // `vcache[t]` per row.  Every lane walks `t` in the SAME increasing order, so
    // each output element keeps the CPU's accumulation order exactly (this is
    // why the t-loop is outside the shuffle: unlike the score reduction, no
    // reassociation happens here).
    float *__restrict__ gout = d_gout + (size_t)h * hd;
    int j = warp * 32 + lane;
    const int jstep = nwarps * 32;
    for (; j < hd; j += jstep) {
        float a = 0.0f;
        if (kv_kind < 0) {
            const float *__restrict__ vcol = d_vcache + (size_t)kvh * hd + j;
            for (int t = 0; t < n; ++t)
                a = __fadd_rn(a, __fmul_rn(s_scores[t], vcol[(size_t)t * kv_dim]));
        } else {
            const int base = kvh * hd + j;
            const int blk = base / 32;
            const int off = base % 32;
            for (int t = 0; t < n; ++t) {
                const unsigned char *vrow = vcache_b + (size_t)t * row_b;
                const float vv = q35_kv_block_elem(vrow + (size_t)blk * stride, off, kv_kind);
                a = __fadd_rn(a, __fmul_rn(s_scores[t], vv));
            }
        }
        gout[j] = a;
    }
}

extern "C" void *q35_func_attn_decode_coal(void) { return (void *)&q35_attn_decode_kernel_coal; }

// Launch the warp-tiled variant.  Uses 256 threads (8 warps) per block so a
// block covers 8 rows at once instead of 4 with 128 -- the score phase is the
// uncoalesced one and this is where the width pays.
extern "C" int q35_launch_attn_decode_coal(const float *d_q, const float *d_gate,
                                           const float *d_kcache,
                                           const float *d_vcache, float *d_att,
                                           float *d_gout, int slot, int heads,
                                           int gqa, int hd, int kv_dim, float scale,
                                           int kv_kind, void *stream)
{
    if (heads <= 0 || hd <= 0) return (int)cudaSuccess;
    const int n = slot + 1;
    if (n > Q35_ATTN_MAX_ROWS) return cudaErrorInvalidValue;
    q35_attn_decode_kernel_coal<<<heads, 256, (size_t)(n + 2) * sizeof(float), (cudaStream_t)stream>>>(
        d_q, d_gate, d_kcache, d_vcache, d_att, d_gout, slot, heads, gqa, hd,
        kv_dim, scale, kv_kind);
    int rc = (int)cudaGetLastError();
    if (rc != (int)cudaSuccess) return rc;
    const int total = heads * hd;
    const int threads = 128;
    const int blocks = (total + threads - 1) / threads;
    q35_attn_gate_kernel<<<blocks, threads, 0, (cudaStream_t)stream>>>(d_gout, d_gate, total);
    return (int)cudaGetLastError();
}

// ---------------------------------------------------------------------------
// SPLIT variant: the row walk is divided across `SPLIT` blocks per head.
//
// The warp-tiled kernel above is still only 68-73 GB/s because ONE block per head
// gives 24 blocks x 256 threads = 6144 threads on 114 SMs (21% occupancy) and
// every lane then walks all `n` rows serially.  This variant launches
// `heads * SPLIT` blocks; block `s` handles rows `t % SPLIT == s`... no: it
// handles the contiguous slice [s*n/SPLIT, (s+1)*n/SPLIT), and the PV partials
// are summed in slice order by a second kernel.
//
// CONSEQUENCE: the score for row t is still the serial j-sum (bit-exact), but the
// PV sum is now a 2-level tree instead of one serial walk, so `gout` differs from
// the reference in the last bits.  That is the SAME trade the decode matvec split
// path already makes (`mmvq.cu`), and it is gated the same way: integer-exact
// token IDs plus zero MVCHECK mismatches, not bit equality.
#define Q35_ATTN_SPLIT_MAX 16

__global__ void q35_attn_decode_kernel_split(
    const float *__restrict__ d_q,
    const float *__restrict__ d_gate,
    const float *__restrict__ d_kcache,
    const float *__restrict__ d_vcache,
    float *__restrict__ d_att,
    float *__restrict__ d_gout,
    float *__restrict__ d_part,   // [heads][SPLIT][hd]
    const float *__restrict__ d_prob,  // [n] probabilities from the score kernel
    int slot, int heads, int gqa, int hd, int kv_dim, float scale, int split,
    int kv_kind)
{
    const int h = blockIdx.x / split;
    const int s = blockIdx.x % split;
    const int tid = threadIdx.x;
    const int nt = blockDim.x;
    const int warp = tid >> 5, lane = tid & 31;
    const int nwarps = nt >> 5;
    if (h >= heads) return;
    const int n = slot + 1;
    const int kvh = h / gqa;
    const unsigned char *vcache_b = (const unsigned char *)d_vcache;
    const size_t stride = (kv_kind == 0) ? 64 : (kv_kind == 1) ? 34 : 18;
    const size_t row_b = (size_t)(kv_dim / 32) * stride;

    // Contiguous slice of rows for this block (last block takes the remainder).
    const int per = (n + split - 1) / split;
    const int t0 = s * per;
    const int t1 = min(n, t0 + per);
    const int cnt = t1 > t0 ? t1 - t0 : 0;

    // NOTE the softmax is NOT computed here.  Each block used to normalise by
    // its OWN slice's max/sum, which is wrong: the softmax is over the whole row.
    // The first version did exactly that and was caught by the microbench
    // (`mismatched=6144`, worst error 0.35 = the scale of the output itself).
    // So probabilities come from `d_prob`, written by the score kernel.
    const float *__restrict__ prob = d_prob + (size_t)h * (size_t)Q35_ATTN_MAX_ROWS + (size_t)t0;

    // PV over this block's slice -> d_part[h][s][j].
    float *__restrict__ part = d_part + ((size_t)h * split + s) * (size_t)hd;
    for (int j = warp * 32 + lane; j < hd; j += nwarps * 32) {
        float a = 0.0f;
        if (kv_kind < 0) {
            const float *__restrict__ vcol = d_vcache + (size_t)kvh * hd + j;
            for (int i = 0; i < cnt; ++i)
                a = __fadd_rn(a, __fmul_rn(prob[i], vcol[(size_t)(t0 + i) * kv_dim]));
        } else {
            const int base = kvh * hd + j;
            const int blk = base / 32;
            const int off = base % 32;
            for (int i = 0; i < cnt; ++i) {
                const unsigned char *vrow = vcache_b + (size_t)(t0 + i) * row_b;
                const float vv = q35_kv_block_elem(vrow + (size_t)blk * stride, off, kv_kind);
                a = __fadd_rn(a, __fmul_rn(prob[i], vv));
            }
        }
        part[j] = a;
    }
}

// Reduce the per-block PV partials in slice order, apply the gate.
__global__ void q35_attn_reduce_kernel(const float *__restrict__ d_part,
                                       const float *__restrict__ d_gate,
                                       float *__restrict__ d_gout, int heads,
                                       int hd, int split)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= heads * hd) return;
    const int h = idx / hd, j = idx % hd;
    const float *__restrict__ p = d_part + ((size_t)h * split) * (size_t)hd + j;
    float a = 0.0f;
    for (int s = 0; s < split; ++s) a = __fadd_rn(a, p[(size_t)s * hd]);
    const float g = d_gate[idx];
    d_gout[idx] = a * (1.0f / (1.0f + expf(-g)));
}


// ---------------------------------------------------------------------------
// FULL split: BOTH passes divide the row walk across `split` blocks per head.
//
// Pass 1 writes raw scores (no softmax) into `d_att`, so block boundaries cannot
// change the result.  A host-visible global max/sum is then computed by
// `q35_attn_softmax_kernel` (one block per head, so it sees every row), and the
// probabilities are written back in place.  Pass 2 (the PV walk above) consumes
// them.
//
// Split over the SCORE walk is safe for exactness: each score is still its own
// serial j-sum, and the softmax that follows is the CPU's order over the full
// row.  The only re-associated sum is the PV one, which is why the split path is
// gated on integer-exact token IDs.


// Softmax over the full row of `d_att[0..n)` for every head, in place.




// ---------------------------------------------------------------------------
// Device-slot form of the split path, so a CUDA-graph node never retargets.
//
// Same three phases as `q35_launch_attn_decode_split2`, but `slot` comes from
// DEVICE memory and `split` is a constant, so the argument blocks are frozen for
// the whole run.  The dynamic shared block is sized for the MAXIMUM window --
// the split kernels use no shared memory (their partials live in `d_part`), so
// this costs nothing.
__global__ void q35_attn_score_split_kernel_dyn(
    const float *__restrict__ d_q,
    const float *__restrict__ d_kcache,
    float *__restrict__ d_att,
    const int *__restrict__ d_slot,
    int heads, int gqa, int hd, int kv_dim, float scale, int split, int kv_kind)
{
    const int h = blockIdx.x / split;
    const int s = blockIdx.x % split;
    const int tid = threadIdx.x;
    if (h >= heads) return;
    const int n = *d_slot + 1;
    const int kvh = h / gqa;
    const float *__restrict__ qs = d_q + (size_t)h * hd;
    const unsigned char *kcache_b = (const unsigned char *)d_kcache;
    const size_t stride = (kv_kind == 0) ? 64 : (kv_kind == 1) ? 34 : 18;
    const size_t row_b = (size_t)(kv_dim / 32) * stride;
    const int warp = tid >> 5, lane = tid & 31;
    const int nwarps = blockDim.x >> 5;
    const int per = (n + split - 1) / split;
    const int t0 = s * per, t1 = min(n, t0 + per);
    float *__restrict__ mine = d_att + (size_t)h * (size_t)Q35_ATTN_MAX_ROWS;
    for (int t = t0 + warp; t < t1; t += nwarps) {
        float acc = 0.0f;
        if (kv_kind < 0) {
            const float *__restrict__ ks = d_kcache + (size_t)t * kv_dim + (size_t)kvh * hd;
            for (int j = lane; j < hd; j += 32) acc = __fadd_rn(acc, __fmul_rn(qs[j], ks[j]));
        } else {
            const unsigned char *krow = kcache_b + (size_t)t * row_b;
            const int base = kvh * hd;
            for (int j = lane; j < hd; j += 32) {
                const unsigned char *blk = krow + ((size_t)((base + j) / 32)) * stride;
                acc = __fadd_rn(acc, __fmul_rn(qs[j], q35_kv_block_elem(blk, (base + j) % 32, kv_kind)));
            }
        }
        for (int o = 16; o > 0; o >>= 1) acc = __fadd_rn(acc, __shfl_down_sync(0xffffffffu, acc, o));
        if (lane == 0) mine[t] = __fmul_rn(acc, scale);
    }
}

__global__ void q35_attn_softmax_kernel_dyn(float *__restrict__ d_att,
                                            const int *__restrict__ d_slot,
                                            int heads)
{
    const int h = blockIdx.x;
    if (h >= heads) return;
    const int n = *d_slot + 1;
    float *__restrict__ a = d_att + (size_t)h * (size_t)Q35_ATTN_MAX_ROWS;
    const int tid = threadIdx.x, nt = blockDim.x;
    __shared__ float s_max, s_sum;
    __shared__ float s_w[256];
    float m = -CUDART_INF_F;
    for (int t = tid; t < n; t += nt) m = fmaxf(m, a[t]);
    s_w[tid] = m;
    __syncthreads();
    for (int st = nt / 2; st > 0; st >>= 1) {
        if (tid < st) s_w[tid] = fmaxf(s_w[tid], s_w[tid + st]);
        __syncthreads();
    }
    if (tid == 0) s_max = s_w[0];
    __syncthreads();
    for (int t = tid; t < n; t += nt) a[t] = expf(a[t] - s_max);
    __syncthreads();
    // Serial sum, CPU order -- see the note in `q35_attn_softmax_kernel`.
    if (tid == 0) {
        float sm = 0.0f;
        for (int t = 0; t < n; ++t) sm += a[t];
        s_sum = sm;
    }
    __syncthreads();
    for (int t = tid; t < n; t += nt) a[t] = a[t] / s_sum;
}

__global__ void q35_attn_pv_split_kernel_dyn(
    const float *__restrict__ d_vcache,
    float *__restrict__ d_part,
    const float *__restrict__ d_prob,
    const int *__restrict__ d_slot,
    int heads, int gqa, int hd, int kv_dim, int split, int kv_kind)
{
    const int h = blockIdx.x / split;
    const int s = blockIdx.x % split;
    const int tid = threadIdx.x;
    if (h >= heads) return;
    const int n = *d_slot + 1;
    const int kvh = h / gqa;
    const unsigned char *vcache_b = (const unsigned char *)d_vcache;
    const size_t stride = (kv_kind == 0) ? 64 : (kv_kind == 1) ? 34 : 18;
    const size_t row_b = (size_t)(kv_dim / 32) * stride;
    const int warp = tid >> 5, lane = tid & 31;
    const int nwarps = blockDim.x >> 5;
    const int per = (n + split - 1) / split;
    const int t0 = s * per, t1 = min(n, t0 + per);
    const int cnt = t1 > t0 ? t1 - t0 : 0;
    const float *__restrict__ prob = d_prob + (size_t)h * (size_t)Q35_ATTN_MAX_ROWS + (size_t)t0;
    float *__restrict__ part = d_part + ((size_t)h * split + s) * (size_t)hd;
    for (int j = warp * 32 + lane; j < hd; j += nwarps * 32) {
        float a = 0.0f;
        if (kv_kind < 0) {
            const float *__restrict__ vcol = d_vcache + (size_t)kvh * hd + j;
            for (int i = 0; i < cnt; ++i)
                a = __fadd_rn(a, __fmul_rn(prob[i], vcol[(size_t)(t0 + i) * kv_dim]));
        } else {
            const int base = kvh * hd + j;
            const int blk = base / 32;
            const int off = base % 32;
            for (int i = 0; i < cnt; ++i) {
                const unsigned char *vrow = vcache_b + (size_t)(t0 + i) * row_b;
                const float vv = q35_kv_block_elem(vrow + (size_t)blk * stride, off, kv_kind);
                a = __fadd_rn(a, __fmul_rn(prob[i], vv));
            }
        }
        part[j] = a;
    }
}

extern "C" void *q35_func_attn_score_split_dyn(void) { return (void *)&q35_attn_score_split_kernel_dyn; }
extern "C" void *q35_func_attn_pv_split_dyn(void) { return (void *)&q35_attn_pv_split_kernel_dyn; }
extern "C" void *q35_func_attn_softmax_dyn(void) { return (void *)&q35_attn_softmax_kernel_dyn; }
extern "C" void *q35_func_attn_reduce(void) { return (void *)&q35_attn_reduce_kernel; }

// Four nodes, all with constant arguments: score-split, softmax, PV-split,
// reduce+gate.  `split` is a constant so the grid is fixed too.
extern "C" int q35_launch_attn_decode_split_dyn(
    const float *d_q, const float *d_gate, const float *d_kcache,
    const float *d_vcache, float *d_att, float *d_gout, float *d_part,
    const int *d_slot, int heads, int gqa, int hd, int kv_dim, float scale,
    int split, int kv_kind, void *stream)
{
    if (heads <= 0 || hd <= 0) return (int)cudaSuccess;
    if (d_slot == nullptr || d_part == nullptr) return (int)cudaErrorInvalidValue;
    if (split < 2) split = 2;
    if (split > Q35_ATTN_SPLIT_MAX) split = Q35_ATTN_SPLIT_MAX;
    q35_attn_score_split_kernel_dyn<<<heads * split, 256, 0, (cudaStream_t)stream>>>(
        d_q, d_kcache, d_att, d_slot, heads, gqa, hd, kv_dim, scale, split, kv_kind);
    int rc = (int)cudaGetLastError();
    if (rc != (int)cudaSuccess) return rc;
    q35_attn_softmax_kernel_dyn<<<heads, 256, 0, (cudaStream_t)stream>>>(d_att, d_slot, heads);
    rc = (int)cudaGetLastError();
    if (rc != (int)cudaSuccess) return rc;
    q35_attn_pv_split_kernel_dyn<<<heads * split, 256, 0, (cudaStream_t)stream>>>(
        d_vcache, d_part, d_att, d_slot, heads, gqa, hd, kv_dim, split, kv_kind);
    rc = (int)cudaGetLastError();
    if (rc != (int)cudaSuccess) return rc;
    const int total = heads * hd;
    q35_attn_reduce_kernel<<<(total + 255) / 256, 256, 0, (cudaStream_t)stream>>>(
        d_part, d_gate, d_gout, heads, hd, split);
    return (int)cudaGetLastError();
}

__global__ void q35_attn_score_split_kernel(
    const float *__restrict__ d_q,
    const float *__restrict__ d_kcache,
    float *__restrict__ d_att,
    int slot, int heads, int gqa, int hd, int kv_dim, float scale, int split, int kv_kind)
{
    const int h = blockIdx.x / split;
    const int s = blockIdx.x % split;
    const int tid = threadIdx.x;
    if (h >= heads) return;
    const int n = slot + 1;
    const int kvh = h / gqa;
    const float *__restrict__ qs = d_q + (size_t)h * hd;
    const unsigned char *kcache_b = (const unsigned char *)d_kcache;
    const size_t stride = (kv_kind == 0) ? 64 : (kv_kind == 1) ? 34 : 18;
    const size_t row_b = (size_t)(kv_dim / 32) * stride;
    const int warp = tid >> 5, lane = tid & 31;
    const int nwarps = blockDim.x >> 5;
    const int per = (n + split - 1) / split;
    const int t0 = s * per, t1 = min(n, t0 + per);
    float *__restrict__ mine = d_att + (size_t)h * (size_t)Q35_ATTN_MAX_ROWS;
    for (int t = t0 + warp; t < t1; t += nwarps) {
        float acc = 0.0f;
        if (kv_kind < 0) {
            const float *__restrict__ ks = d_kcache + (size_t)t * kv_dim + (size_t)kvh * hd;
            for (int j = lane; j < hd; j += 32) acc = __fadd_rn(acc, __fmul_rn(qs[j], ks[j]));
        } else {
            const unsigned char *krow = kcache_b + (size_t)t * row_b;
            const int base = kvh * hd;
            for (int j = lane; j < hd; j += 32) {
                const unsigned char *blk = krow + ((size_t)((base + j) / 32)) * stride;
                acc = __fadd_rn(acc, __fmul_rn(qs[j], q35_kv_block_elem(blk, (base + j) % 32, kv_kind)));
            }
        }
        for (int o = 16; o > 0; o >>= 1) acc = __fadd_rn(acc, __shfl_down_sync(0xffffffffu, acc, o));
        if (lane == 0) mine[t] = __fmul_rn(acc, scale);
    }
}
__global__ void q35_attn_softmax_kernel(float *__restrict__ d_att, int slot, int heads)
{
    const int h = blockIdx.x;
    if (h >= heads) return;
    const int n = slot + 1;
    float *__restrict__ a = d_att + (size_t)h * (size_t)Q35_ATTN_MAX_ROWS;
    const int tid = threadIdx.x, nt = blockDim.x;
    __shared__ float s_max, s_sum;
    __shared__ float s_w[256];
    // Parallel max with the CPU's strict-> tie-free order: max is associative and
    // commutative on the max operation's operand choice here, so a tree is exact.
    float m = -CUDART_INF_F;
    for (int t = tid; t < n; t += nt) m = fmaxf(m, a[t]);
    s_w[tid] = m;
    __syncthreads();
    for (int st = nt / 2; st > 0; st >>= 1) {
        if (tid < st) s_w[tid] = fmaxf(s_w[tid], s_w[tid + st]);
        __syncthreads();
    }
    if (tid == 0) s_max = s_w[0];
    __syncthreads();
    for (int t = tid; t < n; t += nt) a[t] = expf(a[t] - s_max);
    __syncthreads();
    // The SUM must keep the CPU's serial index order -- a tree would re-associate
    // it and move the last bits of every probability.  Thread 0 walks the row;
    // `n` is a few hundred, so this is a few microseconds, not a bottleneck.
    if (tid == 0) {
        float sm = 0.0f;
        for (int t = 0; t < n; ++t) sm += a[t];
        s_sum = sm;
    }
    __syncthreads();
    for (int t = tid; t < n; t += nt) a[t] = a[t] / s_sum;
}
extern "C" void *q35_func_attn_score_split(void) { return (void *)&q35_attn_score_split_kernel; }

extern "C" int q35_launch_attn_decode_split2(const float *d_q, const float *d_gate,
                                             const float *d_kcache,
                                             const float *d_vcache, float *d_att,
                                             float *d_gout, float *d_part, int slot,
                                             int heads, int gqa, int hd, int kv_dim,
                                             float scale, int split, int kv_kind,
                                             void *stream)
{
    if (heads <= 0 || hd <= 0) return (int)cudaSuccess;
    const int n = slot + 1;
    if (n > Q35_ATTN_MAX_ROWS) return cudaErrorInvalidValue;
    if (split < 1) split = 1;
    if (split > Q35_ATTN_SPLIT_MAX) split = Q35_ATTN_SPLIT_MAX;
    if (split > n) split = n;
    if (split == 1 || d_part == nullptr)
        return q35_launch_attn_decode_coal(d_q, d_gate, d_kcache, d_vcache, d_att,
                                           d_gout, slot, heads, gqa, hd, kv_dim,
                                           scale, kv_kind, stream);
    q35_attn_score_split_kernel<<<heads * split, 256, 0, (cudaStream_t)stream>>>(
        d_q, d_kcache, d_att, slot, heads, gqa, hd, kv_dim, scale, split, kv_kind);
    int rc = (int)cudaGetLastError();
    if (rc != (int)cudaSuccess) return rc;
    q35_attn_softmax_kernel<<<heads, 256, 0, (cudaStream_t)stream>>>(d_att, slot, heads);
    rc = (int)cudaGetLastError();
    if (rc != (int)cudaSuccess) return rc;
    q35_attn_decode_kernel_split<<<heads * split, 256, 0, (cudaStream_t)stream>>>(
        d_q, d_gate, d_kcache, d_vcache, d_att, d_gout, d_part, d_att, slot, heads,
        gqa, hd, kv_dim, scale, split, kv_kind);
    rc = (int)cudaGetLastError();
    if (rc != (int)cudaSuccess) return rc;
    const int total = heads * hd;
    q35_attn_reduce_kernel<<<(total + 255) / 256, 256, 0, (cudaStream_t)stream>>>(
        d_part, d_gate, d_gout, heads, hd, split);
    return (int)cudaGetLastError();
}
