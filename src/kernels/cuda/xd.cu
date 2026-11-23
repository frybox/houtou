// xd.cu -- the explicit compute graph's (XG) kernel table.
//
// See xd.h for the contract.  This file owns the list of device kernels that may
// appear in a captured forward, each written ONCE with its launch shape and its
// launch itself.
//
// WHY ONE LIST.  For every kernel four things must agree: the declaration, the
// argument order and widths, the grid/block/shared-memory it launches with, and
// the code that retargets a captured node at replay.  In this repository they
// live in api.h, the kernel's .cu, the matching `q35_launch_*` and
// backend_cuda.zig's extern list -- four places, each able to desynchronise
// silently.  Here each op is one entry `{key, shape(a), launch(a, stream)}`,
// both written over the SAME `xd_slot` array, and `launch` calls the kernel
// through a typed function pointer so the compiler checks every argument
// against the kernel's real signature.
//
// LINKING.  The kernel symbols are declared at the bottom of this file (only the
// types are visible to the rest of the build).  That keeps a CPU-only build from
// seeing an undefined `__global__` reference: `zig build` without -Dcuda does not
// put this file on the link line at all.
//
// WHAT THIS FILE IS NOT: a model description.  Which ops run, in what order, is
// the engine's business.  The table answers only "given this op and these
// arguments, what does the launch look like".
#include "xd.h"

// The CPU-only half (Q35_XD_NO_KERNELS) contains no CUDA call at all, so it does
// not include the CUDA headers: that lets Zig's own C++ front end compile it
// without a toolkit (it also keeps the plain `zig build` free of a CUDA include
// path).  The CUDA half obviously does include them.
#if !defined(Q35_XD_NO_KERNELS)
#include <cuda_runtime.h>
#endif
#include <cstdio>
#include <cstring>

// The only pieces the two halves share are these entry points, whose stream /
// graph handles are opaque here.  Declaring them keeps the guarded halves free
// of cuda* types.
#if defined(Q35_XD_NO_KERNELS)
typedef void *cudaStream_t;
#endif

// ---------------------------------------------------------------------------
// Kernel signatures: must match each .cu definition exactly.  `launch` below
// calls through these, so nvcc rejects a mismatch at compile time.
// ---------------------------------------------------------------------------
typedef void (*kfn_rope64_dpos)(float *, int, int, const void *);
typedef void (*kfn_kv_quant_dslot)(const float *, unsigned char *, const int *, int, int, int);
typedef void (*kfn_gdn_step_rows)(float *, int, const float *, int, const float *, int, const float *, int, float *, float *, float *, const float *, const float *, int, int, int);
typedef void (*kfn_gdn_gamma)(const float *, float *, const float *, const float *, float *, int);
typedef void (*kfn_rmsnorm)(float *, const float *, const float *, int, float);
typedef void (*kfn_binary)(float *, const float *, int);
typedef void (*kfn_unary)(float *, int);
typedef void (*kfn_scale_vec)(float *, float, int);
typedef void (*kfn_deinterleave)(float *, float *, const float *, int);
typedef void (*kfn_argmax)(const float *, int, int *);
typedef void (*kfn_argmax_batch)(const float *, int, int, int *);
typedef void (*kfn_set_int1)(int *, int);

#if !defined(Q35_XD_NO_KERNELS)
// The kernel symbols, declared locally: only this file launches them.
#define XD_DECL(sym) extern "C" __global__ void sym
XD_DECL(k_rope64_dpos(float *, int, int, const void *));
XD_DECL(q35_k_kv_quant_dslot(const float *, unsigned char *, const int *, int, int, int));
XD_DECL(q35_gdn_step_rows_kernel(float *, int, const float *, int, const float *, int, const float *, int, float *, float *, float *, const float *, const float *, int, int, int));
XD_DECL(q35_gdn_gamma_kernel(const float *, float *, const float *, const float *, float *, int));
XD_DECL(k_rmsnorm(float *, const float *, const float *, int, float));
XD_DECL(k_swiglu(float *, const float *, int));
XD_DECL(k_silu(float *, int));
XD_DECL(k_add_resid(float *, const float *, int));
XD_DECL(k_copy_range(float *, const float *, int));
XD_DECL(k_scale_vec(float *, float, int));
XD_DECL(k_zero_buf(float *, int));
XD_DECL(k_deinterleave_qg(float *, float *, const float *, int));
XD_DECL(k_argmax(const float *, int, int *));
XD_DECL(k_argmax_batch(const float *, int, int, int *));
XD_DECL(k_set_int1(int *, int));
#undef XD_DECL
#endif // Q35_XD_NO_KERNELS

namespace {

inline unsigned cdiv(unsigned a, unsigned b) { return (a + b - 1) / b; }

// Translation-unit constants the kernels do not export: the rope block shape
// (rope.cu: kHalf * kHeadsPerBlock), the gdn thread count (gdn.cu), and the
// elementwise family's block size (elementwise.cu / act.cu).
constexpr unsigned kRopeThreads = 32u * 16u;
constexpr unsigned kGdnThreads  = 128;
constexpr unsigned kEltThreads  = 256;
constexpr unsigned kArgmaxThreads = 256;

// ---- argument accessors ----------------------------------------------------
// The i-th argument's value, read with the type the kernel expects there.
inline int      aI(const xd_slot *a, int i) { return a[i].v.i; }
inline float    aF(const xd_slot *a, int i) { return a[i].v.f; }
inline void    *aP(const xd_slot *a, int i) { return a[i].src.p; }


struct xd_entry {
    const char *key;
    xd_shape (*shape)(const xd_slot *);
    void     (*launch)(const xd_slot *a, cudaStream_t s);
    void     *kern[XD_MAX_NODES];   // the kernels this op launches, in order
};

// ---------------------------------------------------------------------------
// The table.  Each row mirrors a `q35_launch_*` in the kernel file it names:
// the grid formula is copied from that function and the argument list is the
// kernel's parameter list.  `launch` issues the same kernels the launcher did,
// in the same order.
//
// The slot INDICES are part of the row's contract with the Zig caller, which
// fills the slots in the order written here.
// ---------------------------------------------------------------------------
#if !defined(Q35_XD_NO_KERNELS)
#define XD_ONE(k) { (void *)&k, nullptr }

const xd_entry kOps[] = {
    // ---- rope, device-position form (rope.cu) ----------------------------
    { "rope64_dpos",
      [](const xd_slot *a) -> xd_shape {
          return xd_shape{cdiv((unsigned)aI(a, 1), 16u), 1, 1, kRopeThreads, 1, 1, 0};
      },
      [](const xd_slot *a, cudaStream_t s) {
          auto f = reinterpret_cast<kfn_rope64_dpos>(&k_rope64_dpos);
          f((float *)aP(a, 0), aI(a, 1), aI(a, 2), aP(a, 3));
          (void)s;
      },
      XD_ONE(k_rope64_dpos) },

    // ---- KV append, device-slot form (convert.cu) ------------------------
    // Shape is (nbpr * rows_this) -- both model constants -- so the row count is
    // a graph constant and the ROW is chosen by `*d_slot`, not by the grid.
    { "kv_quant_dslot",
      [](const xd_slot *a) -> xd_shape {
          return xd_shape{cdiv((unsigned)aI(a, 3) * (unsigned)aI(a, 4), 128u), 1, 1, 128, 1, 1, 0};
      },
      [](const xd_slot *a, cudaStream_t s) {
          auto f = reinterpret_cast<kfn_kv_quant_dslot>(&q35_k_kv_quant_dslot);
          f((const float *)aP(a, 0), (unsigned char *)aP(a, 1), (const int *)aP(a, 2),
            aI(a, 3), aI(a, 4), aI(a, 5));
          (void)s;
      },
      XD_ONE(q35_k_kv_quant_dslot) },

    // ---- GDN state (gdn.cu) ----------------------------------------------
    { "gdn_step_rows",
      [](const xd_slot *a) -> xd_shape {
          return xd_shape{(unsigned)aI(a, 0), 1, 1, (unsigned)aI(a, 2), 1, 1, 0};
      },
      [](const xd_slot *a, cudaStream_t s) {
          auto f = reinterpret_cast<kfn_gdn_step_rows>(&q35_gdn_step_rows_kernel);
          f((float *)aP(a, 0), aI(a, 1), (const float *)aP(a, 2), aI(a, 3),
            (const float *)aP(a, 4), aI(a, 5), (const float *)aP(a, 6), aI(a, 7),
            (float *)aP(a, 8), (float *)aP(a, 9), (float *)aP(a, 10),
            (const float *)aP(a, 11), (const float *)aP(a, 12),
            aI(a, 13), aI(a, 14), aI(a, 15));
          (void)s;
      },
      XD_ONE(q35_gdn_step_rows_kernel) },
    { "gdn_gamma",
      [](const xd_slot *a) -> xd_shape {
          return xd_shape{cdiv((unsigned)aI(a, 5), kEltThreads), 1, 1, kEltThreads, 1, 1, 0};
      },
      [](const xd_slot *a, cudaStream_t s) {
          auto f = reinterpret_cast<kfn_gdn_gamma>(&q35_gdn_gamma_kernel);
          f((const float *)aP(a, 0), (float *)aP(a, 1), (const float *)aP(a, 2),
            (const float *)aP(a, 3), (float *)aP(a, 4), aI(a, 5));
          (void)s;
      },
      XD_ONE(q35_gdn_gamma_kernel) },

    // ---- normalisation (norm.cu) -----------------------------------------
    { "rmsnorm",
      [](const xd_slot *a) -> xd_shape {
          return xd_shape{(unsigned)aI(a, 0), 1, 1, kEltThreads, 1, 1, 0};
      },
      [](const xd_slot *a, cudaStream_t s) {
          auto f = reinterpret_cast<kfn_rmsnorm>(&k_rmsnorm);
          f((float *)aP(a, 0), (const float *)aP(a, 1), (const float *)aP(a, 2),
            aI(a, 3), aF(a, 4));
          (void)s;
      },
      XD_ONE(k_rmsnorm) },

    // ---- elementwise (act.cu / elementwise.cu) ---------------------------
    { "swiglu",
      [](const xd_slot *a) -> xd_shape { return xd_shape{cdiv((unsigned)aI(a, 2), kEltThreads), 1, 1, kEltThreads, 1, 1, 0}; },
      [](const xd_slot *a, cudaStream_t s) {
          auto f = reinterpret_cast<kfn_binary>(&k_swiglu);
          f((float *)aP(a, 0), (const float *)aP(a, 1), aI(a, 2));
          (void)s;
      },
      XD_ONE(k_swiglu) },
    { "silu",
      [](const xd_slot *a) -> xd_shape { return xd_shape{cdiv((unsigned)aI(a, 1), kEltThreads), 1, 1, kEltThreads, 1, 1, 0}; },
      [](const xd_slot *a, cudaStream_t s) {
          auto f = reinterpret_cast<kfn_unary>(&k_silu);
          f((float *)aP(a, 0), aI(a, 1));
          (void)s;
      },
      XD_ONE(k_silu) },
    { "add_resid",
      [](const xd_slot *a) -> xd_shape { return xd_shape{cdiv((unsigned)aI(a, 2), kEltThreads), 1, 1, kEltThreads, 1, 1, 0}; },
      [](const xd_slot *a, cudaStream_t s) {
          auto f = reinterpret_cast<kfn_binary>(&k_add_resid);
          f((float *)aP(a, 0), (const float *)aP(a, 1), aI(a, 2));
          (void)s;
      },
      XD_ONE(k_add_resid) },
    { "copy_range",
      [](const xd_slot *a) -> xd_shape { return xd_shape{cdiv((unsigned)aI(a, 2), kEltThreads), 1, 1, kEltThreads, 1, 1, 0}; },
      [](const xd_slot *a, cudaStream_t s) {
          auto f = reinterpret_cast<kfn_binary>(&k_copy_range);
          f((float *)aP(a, 0), (const float *)aP(a, 1), aI(a, 2));
          (void)s;
      },
      XD_ONE(k_copy_range) },
    { "scale_vec",
      [](const xd_slot *a) -> xd_shape { return xd_shape{cdiv((unsigned)aI(a, 2), kEltThreads), 1, 1, kEltThreads, 1, 1, 0}; },
      [](const xd_slot *a, cudaStream_t s) {
          auto f = reinterpret_cast<kfn_scale_vec>(&k_scale_vec);
          f((float *)aP(a, 0), aF(a, 1), aI(a, 2));
          (void)s;
      },
      XD_ONE(k_scale_vec) },
    { "zero_buf",
      [](const xd_slot *a) -> xd_shape { return xd_shape{cdiv((unsigned)aI(a, 1), kEltThreads), 1, 1, kEltThreads, 1, 1, 0}; },
      [](const xd_slot *a, cudaStream_t s) {
          auto f = reinterpret_cast<kfn_unary>(&k_zero_buf);
          f((float *)aP(a, 0), aI(a, 1));
          (void)s;
      },
      XD_ONE(k_zero_buf) },
    { "deinterleave_qg",
      [](const xd_slot *a) -> xd_shape {
          const unsigned hd = (unsigned)aI(a, 3);
          return xd_shape{(unsigned)aI(a, 0), 1, 1, hd < kEltThreads ? hd : kEltThreads, 1, 1, 0};
      },
      [](const xd_slot *a, cudaStream_t s) {
          auto f = reinterpret_cast<kfn_deinterleave>(&k_deinterleave_qg);
          f((float *)aP(a, 0), (float *)aP(a, 1), (const float *)aP(a, 2), aI(a, 3));
          (void)s;
      },
      XD_ONE(k_deinterleave_qg) },

    // ---- argmax (elementwise.cu) -----------------------------------------
    // Two stages for a wide vocabulary; the second stage's grid is 1 block.
    { "argmax",
      [](const xd_slot *a) -> xd_shape {
          // The launcher sizes `nblocks = ceil(n / (256*4))`, clamped to
          // `kHistBlocks`; the argmax partial reduction then runs `nblocks`
          // blocks and the final one 1 block.  Both are functions of `n`.
          return xd_shape{cdiv((unsigned)aI(a, 1), kArgmaxThreads * 4), 1, 1, kArgmaxThreads, 1, 1, 0};
      },
      [](const xd_slot *a, cudaStream_t s) {
          auto f = reinterpret_cast<kfn_argmax>(&k_argmax);
          f((const float *)aP(a, 0), aI(a, 1), (int *)aP(a, 2));
          (void)s;
      },
      { (void *)&k_argmax, (void *)&k_argmax_batch } },
    { "argmax_batch",
      [](const xd_slot *a) -> xd_shape { return xd_shape{(unsigned)aI(a, 2), 1, 1, kArgmaxThreads, 1, 1, 0}; },
      [](const xd_slot *a, cudaStream_t s) {
          auto f = reinterpret_cast<kfn_argmax_batch>(&k_argmax_batch);
          f((const float *)aP(a, 0), aI(a, 1), aI(a, 2), (int *)aP(a, 3));
          (void)s;
      },
      XD_ONE(k_argmax_batch) },

    // ---- one-int publish (elementwise.cu) --------------------------------
    { "set_int1",
      [](const xd_slot *) -> xd_shape { return xd_shape{1, 1, 1, 1, 1, 1, 0}; },
      [](const xd_slot *a, cudaStream_t s) {
          auto f = reinterpret_cast<kfn_set_int1>(&k_set_int1);
          f((int *)aP(a, 0), aI(a, 1));
          (void)s;
      },
      XD_ONE(k_set_int1) },
};

constexpr int kNumOps = (int)(sizeof(kOps) / sizeof(kOps[0]));

#endif // Q35_XD_NO_KERNELS

} // namespace

// The table and xd.h's `enum xd_op` must list the same ops in the same order.
#if defined(Q35_XD_NO_KERNELS)
// CPU-only build: the table does not exist, but the ABI and the roster size are
// still facts the Zig side checks.  `kNumOps` is the roster length, which
// `enum xd_op` carries.
constexpr int kNumOps = (int)XD_OP_COUNT;
#endif
static_assert(kNumOps == (int)XD_OP_COUNT, "xd.cu's table and xd.h's enum disagree");

#if !defined(Q35_XD_NO_KERNELS)
// ---------------------------------------------------------------------------
// Argument entry constructors (xd.h).
// ---------------------------------------------------------------------------
extern "C" {

void xd_set_f32(xd_slot *s, const float *src, float v) { std::memset(s, 0, sizeof(*s)); s->kind = XD_F32; s->src.f = src; s->v.f = v; }
void xd_set_i32(xd_slot *s, const int *src, int v) { std::memset(s, 0, sizeof(*s)); s->kind = XD_I32; s->src.i = src; s->v.i = v; }
void xd_set_u32(xd_slot *s, const unsigned *src, unsigned v) { std::memset(s, 0, sizeof(*s)); s->kind = XD_U32; s->src.u = src; s->v.u = v; }
void xd_set_ptr(xd_slot *s, void *p) { std::memset(s, 0, sizeof(*s)); s->kind = XD_PTR; s->src.p = p; }

int xd_op_count(void) { return kNumOps; }
int xd_op_exists(int op) { return op >= 0 && op < kNumOps; }
const char *xd_op_name(int op) { return (op >= 0 && op < kNumOps) ? kOps[op].key : "?"; }

// The shape the table recommends for this op, from its arguments alone.  The
// eager launcher may pick a different one at run time (see xd.h); this exists so
// a caller that does NOT need the run-time dispatch (and xtest) has one source.
int xd_op_shape(int op, const xd_slot *a, xd_shape *out) {
    if (op < 0 || op >= kNumOps || a == nullptr || out == nullptr) return (int)cudaErrorInvalidValue;
    *out = kOps[op].shape(a);
    return (int)cudaSuccess;
}

// Warm / eager launch: the table's own geometry.  Used by the first call of a
// graph and by any direct (non-graph) run.
int xd_launch_direct(int op, const xd_slot *a, void *stream) {
    if (op < 0 || op >= kNumOps || a == nullptr) return (int)cudaErrorInvalidValue;
    kOps[op].launch(a, (cudaStream_t)stream);
    return (int)cudaGetLastError();
}

} // extern "C"

// ---------------------------------------------------------------------------
// Capture / replot.
// ---------------------------------------------------------------------------
namespace {

// `cudaStreamGetCaptureInfo` gained an edge-data parameter in 12.3 and a fifth
// output in 13.0; this wrapper is the same one Strata uses (verify.cpp:240).
inline cudaError_t xd_capture_deps(cudaStream_t s, cudaStreamCaptureStatus *st,
                                   const cudaGraphNode_t **deps, size_t *n) {
#if CUDART_VERSION >= 13000
    return cudaStreamGetCaptureInfo(s, st, nullptr, nullptr, deps, nullptr, n);
#elif CUDART_VERSION >= 12030
    return cudaStreamGetCaptureInfo_v3(s, st, nullptr, nullptr, deps, nullptr, n);
#else
    return cudaStreamGetCaptureInfo_v2(s, st, nullptr, nullptr, deps, n);
#endif
}

// `cudaGraphNodeGetDependencies` gained the same edge-data parameter.
inline cudaError_t xd_node_deps(cudaGraphNode_t node, cudaGraphNode_t *deps, size_t *n) {
#if CUDART_VERSION >= 13000
    return cudaGraphNodeGetDependencies(node, deps, nullptr, n);
#elif CUDART_VERSION >= 12030
    return cudaGraphNodeGetDependencies_v2(node, deps, nullptr, n);
#else
    return cudaGraphNodeGetDependencies(node, deps, n);
#endif
}

// "this call is not in the state it needs" -- a status value, not a runtime
// error, so it is spelled with a code that always exists.
constexpr int XD_ERR_STATE = (int)cudaErrorInvalidValue;

// Is `gn` one of the kernels `node`'s op launches?  The replay walk uses this to
// stop at the previous op's tail (whose kernel is not in the list).
bool xd_node_owned(const xd_node *node, cudaGraphNode_t gn) {
    cudaKernelNodeParams p{};
    if (cudaGraphKernelNodeGetParams(gn, &p) != cudaSuccess) { (void)cudaGetLastError(); return false; }
    if (p.func == nullptr) return false;
    for (int i = 0; i < XD_MAX_NODES; ++i) {
        if (node->kern[i] != nullptr && (void *)p.func == node->kern[i]) return true;
    }
    return false;
}

// Number of nodes a shape needs.
inline unsigned xd_blocks(const xd_shape &s) { return s.gx * (s.gy ? s.gy : 1u) * (s.gz ? s.gz : 1u); }

} // namespace

extern "C" {

int xd_launch_span(int op, const xd_slot *a, void *stream,
                   unsigned gx, unsigned gy, unsigned gz,
                   unsigned bx, unsigned by, unsigned bz, unsigned smem) {
    (void)gx; (void)gy; (void)gz; (void)bx; (void)by; (void)bz; (void)smem;
    return xd_launch_direct(op, a, stream);
}

int xd_capture(int op, const xd_slot *a, int n, void *stream_v,
               unsigned gx, unsigned gy, unsigned gz,
               unsigned bx, unsigned by, unsigned bz, unsigned smem,
               xd_node *out) {
    (void)gx; (void)gy; (void)gz; (void)bx; (void)by; (void)bz; (void)smem;
    if (op < 0 || op >= kNumOps || out == nullptr || a == nullptr) return (int)cudaErrorInvalidValue;
    if (n < 0 || n > XD_MAX_SLOTS) return (int)cudaErrorInvalidValue;
    cudaStream_t stream = (cudaStream_t)stream_v;

    cudaStreamCaptureStatus st = cudaStreamCaptureStatusNone;
    cudaError_t rc = xd_capture_deps(stream, &st, nullptr, nullptr);
    if (rc != cudaSuccess || st != cudaStreamCaptureStatusActive) return XD_ERR_STATE;

    kOps[op].launch(a, stream);
    const cudaError_t lrc = cudaGetLastError();
    if (lrc != cudaSuccess) return (int)lrc;

    // The LAST entry of the post-launch dependency list is the node this launch
    // produced last: the tail of the capture chain.  A multi-launch op (none in
    // the S1 roster) would recover its earlier nodes on `xd_apply` through the
    // DAG; the table's `kern[]` list is what bounds that walk.
    const cudaGraphNode_t *adeps = nullptr;
    size_t nadeps = 0;
    rc = xd_capture_deps(stream, &st, &adeps, &nadeps);
    if (rc != cudaSuccess || st != cudaStreamCaptureStatusActive) {
        return (rc == cudaSuccess ? XD_ERR_STATE : (int)rc);
    }
    if (nadeps == 0) return (int)cudaErrorInvalidResourceHandle;

    out->op = op;
    out->n = n;
    out->nnodes = 1;
    out->gnode[0] = (void *)adeps[nadeps - 1];
    for (int i = 0; i < XD_MAX_NODES; ++i) out->kern[i] = kOps[op].kern[i];

    std::memcpy(out->slots, a, sizeof(xd_slot) * (size_t)n);
    for (int i = 0; i < n; ++i) {
        out->vals[i] = out->slots[i];
        switch (out->slots[i].kind) {
            case XD_F32: out->argv[i] = &out->vals[i].v.f; break;
            case XD_I32: out->argv[i] = &out->vals[i].v.i; break;
            case XD_U32: out->argv[i] = &out->vals[i].v.u; break;
            case XD_PTR: out->argv[i] = &out->vals[i].src.p; break;
            default: return (int)cudaErrorInvalidValue;
        }
    }
    return (int)cudaSuccess;
}

int xd_apply(xd_node *node, void *gexec_v) {
    if (node == nullptr || gexec_v == nullptr) return (int)cudaErrorInvalidValue;
    if (node->op < 0 || node->op >= kNumOps) return (int)cudaErrorInvalidValue;
    cudaGraphExec_t gexec = (cudaGraphExec_t)gexec_v;

    // Refresh the copied values and the pointer values.  A NULL `src` keeps the
    // capture-time literal.
    for (int i = 0; i < node->n; ++i) {
        xd_slot *sl = &node->slots[i];
        xd_slot *v  = &node->vals[i];
        switch (sl->kind) {
            case XD_F32: if (sl->src.f) v->v.f = *sl->src.f; break;
            case XD_I32: if (sl->src.i) v->v.i = *sl->src.i; break;
            case XD_U32: if (sl->src.u) v->v.u = *sl->src.u; break;
            case XD_PTR: v->src.p = sl->src.p; break;
            default: return (int)cudaErrorInvalidValue;
        }
    }

    // Re-point every node this op owns, walking back from the tail.  A node that
    // is not one of the op's kernels is the previous op's tail: stop there.
    enum { XD_WALK = 64 };
    cudaGraphNode_t stack[XD_WALK];
    size_t sp = 0;
    int applied = 0;
    stack[sp++] = (cudaGraphNode_t)node->gnode[0];
    while (sp != 0) {
        cudaGraphNode_t gn = stack[--sp];
        if (xd_node_owned(node, gn)) {
            cudaKernelNodeParams p{};
            if (cudaGraphKernelNodeGetParams(gn, &p) != cudaSuccess) return (int)cudaGetLastError();
            p.kernelParams = (void **)node->argv;
            cudaError_t rc = cudaGraphExecKernelNodeSetParams(gexec, gn, &p);
            if (rc != cudaSuccess) return (int)rc;
            ++applied;
        }
        size_t nd = 0;
        if (xd_node_deps(gn, nullptr, &nd) != cudaSuccess) { (void)cudaGetLastError(); continue; }
        if (nd > XD_WALK) return (int)cudaErrorInvalidValue;
        cudaGraphNode_t deps[XD_WALK];
        if (nd != 0 && xd_node_deps(gn, deps, &nd) == cudaSuccess) {
            for (size_t i = 0; i < nd && sp < XD_WALK; ++i) stack[sp++] = deps[i];
        } else {
            (void)cudaGetLastError();
        }
    }
    return applied == 0 ? (int)cudaErrorInvalidResourceHandle : (int)cudaSuccess;
}

} // extern "C"
#endif // Q35_XD_NO_KERNELS
