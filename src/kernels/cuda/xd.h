// xd.h -- the explicit compute graph's (XG) narrow C surface.
//
// WHAT THIS IS.  `xd.cu` owns the ONE table of device kernels that may appear
// in a captured forward, and the three primitives that turn a launch into graph
// nodes and back:
//
//   xd_launch_direct  the eager/warm launch (no capture open)
//   xd_capture        called INSIDE an open stream capture: launch once (CUDA
//                     records it), then recover the nodes that launch produced
//                     from `cudaStreamGetCaptureInfo` and store them in a node
//   xd_apply          before a replay: refresh every copied argument value from
//                     its live cell and re-point every tracked node at this
//                     node's own argument array
//
// WHY A TABLE.  The previous decode-graph code identified the nodes it had to
// retarget AFTER a capture, from the CUDA graph object, by classifying host-stub
// addresses and matching argument VALUES (the removed `graphBuild` /
// `classifyNode` / `copyArgs` in src/backend_cuda.zig).  That cannot describe an
// MTP draft or a verify: their per-round arguments change, some launches are not
// in the graph at all, and nothing in the CUDA object says which op is which.
// Here the identity is the table index, the order is the call order, and the
// argument layout is the kernel's own signature (xd.cu), so the compiler -- not
// a hand-written offset table -- checks every argument.
//
// THE TWO WAYS AN ARGUMENT CAN CHANGE
//   1. COPIED (re-copied by xd_apply).  Every slot the caller can write between
//      two replays into its own cell.  Costs one xd_apply over that op's slots.
//   2. DEVICE POINTER TO A CELL.  The kernel takes `const int *p` and reads *p
//      at execution time, so the value changes with NO xd_apply at all.  That
//      is how `rope64`'s pos/base already work (`k_rope64_dpos`) and how the
//      sample chain's `d_ctl` works.  This is the only free path, and it is why
//      `xd_ctl` exists at all.
//
// A MULTI-LAUNCH OP (`attn_decode_dyn` issues the decode kernel and then the
// gate kernel) records SEVERAL nodes under one `xd_node`.  They are kept in
// launch order and re-pointed in launch order, so the replay reproduces the
// eager sequence exactly.
#ifndef Q35_CUDA_XD_H
#define Q35_CUDA_XD_H

#include <stddef.h>

enum xd_kind { XD_F32 = 0, XD_I32 = 1, XD_U32 = 2, XD_PTR = 3 };

// Launch geometry of one op, as the table computes it from the op's immediate
// arguments (never from a device value -- that is what keeps a captured node's
// shape constant, and it is why the table can be checked without a GPU).
struct xd_shape {
    unsigned gx, gy, gz;
    unsigned bx, by, bz;
    unsigned smem;
};

// Widest recipe this table has.  `gdn_step_rows` needs 17 (13 pointer/stride
// arguments plus `heads` for the grid and `nk`/`kd`/`vd` for the kernel); the
// S1 comment said 16 because that roster never carried `kd`.  20 leaves room for
// the ops S3 adds without another ABI change.
#define XD_MAX_SLOTS 20
#define XD_MAX_NODES 4

// One argument entry.  `src` may be NULL for a literal (then `v` is the value
// every replay uses).  For a pointer argument `src.p` IS the value, so it is
// read (never dereferenced) by xd_apply.
struct xd_slot {
    int kind;
    union {
        const float    *f;
        const int      *i;
        const unsigned *u;
        void           *p;
    } src;
    union {
        float    f;
        int      i;
        unsigned u;
    } v;
};

// One XG op as recorded for a graph: the nodes it produced, and the slot array
// it is launched from (the SAME array `xd_capture` launched with).  `vals` is
// the private copy xd_apply refreshes; `argv` holds the addresses handed to
// `cudaGraphExecKernelNodeSetParams`, one per slot, each pointing into `vals`.
struct xd_node {
    int    op;
    int    n;                          // number of slots in use
    int    nnodes;                     // graph nodes this op produced
    void  *gnode[XD_MAX_NODES];        // cudaGraphNode_t, in launch order
    xd_slot slots[XD_MAX_SLOTS];       // caller view: `src` + inline `v`
    xd_slot vals[XD_MAX_SLOTS];        // private values (immediates copied)
    void  *argv[XD_MAX_SLOTS];         // kernelParams[] handed to SetParams
    void  *gate_argv[4];               // the trailing gate launch's params
    int    derived[XD_MAX_SLOTS];      // cells for recomputed values (`head*hd`)
    void  *kern[XD_MAX_NODES];         // the CUDA kernels this op launches
};

// Op ids, in table order (xd.cu).  Spelled here because the Zig side needs to
// name two of them by value (attn's second launch and the sample chain); the
// initialiser in xd.cu is checked against this list by `xd_op_count`.
//
// S1 ROSTER.  These are the decode-path ops whose launch shape is fully
// determined by model constants.  The remaining decode ops (the quantized matvec
// dispatch, the coop/split variants, convStep, attnChunk, gdnStepRowsChunk,
// argmaxConf, the sampler chain) are added in S3, when the ops that read a
// per-step scalar from a host cell are switched to device cells.  Listing them
// now would mean a table row without a verified launch.
enum xd_op {
    XD_OP_rope64_dpos = 0,
    XD_OP_kv_quant_dslot,
    XD_OP_gdn_step_rows,
    XD_OP_gdn_gamma,
    XD_OP_rmsnorm,
    XD_OP_swiglu,
    XD_OP_silu,
    XD_OP_add_resid,
    XD_OP_copy_range,
    XD_OP_scale_vec,
    XD_OP_zero_buf,
    XD_OP_deinterleave_qg,
    XD_OP_argmax,
    XD_OP_argmax_batch,
    XD_OP_set_int1,
    XD_OP_COUNT
};

#ifdef __cplusplus
extern "C" {
#endif

// ABI guard.  `struct xd_slot` / `struct xd_node` are passed by pointer across
// the C++/Zig boundary, so their layout is part of the contract and must match
// src/xd.zig field for field.  These asserts run in every .cu compile.
static_assert(sizeof(struct xd_slot) == 24, "xd_slot layout changed; update src/xd.zig");
static_assert(offsetof(struct xd_slot, kind) == 0, "xd_slot.kind moved");
static_assert(offsetof(struct xd_slot, src) == 8, "xd_slot.src moved");
static_assert(offsetof(struct xd_slot, v) == 16, "xd_slot.v moved");
static_assert(sizeof(struct xd_node) == 1312, "xd_node layout changed; update src/xd.zig");
static_assert(offsetof(struct xd_node, gnode) == 16, "xd_node.gnode moved");
static_assert(offsetof(struct xd_node, slots) == 48, "xd_node.slots moved");
static_assert(offsetof(struct xd_node, vals) == 528, "xd_node.vals moved");
static_assert(offsetof(struct xd_node, argv) == 1008, "xd_node.argv moved");
static_assert(offsetof(struct xd_node, gate_argv) == 1168, "xd_node.gate_argv moved");
static_assert(offsetof(struct xd_node, derived) == 1200, "xd_node.derived moved");

// ---------------------------------------------------------------------------
// The per-step control block (`cell` in plan §1.5).
//
// One device buffer per graph, refreshed once per step.  Every kernel that needs
// a per-token scalar takes the block's address and reads the field at execution
// time, so a captured node's ARGUMENT LIST never changes and a replay needs no
// `cudaGraphExecKernelNodeSetParams` for any of them.
//
// Field meanings mirror `backend.GraphCtl` (src/backend.zig) field for field;
// the static_asserts below are the cross-language check.
// ---------------------------------------------------------------------------
// 嵌入行号数组的长度。一次块前向最多读多少行 embedding 由引擎的
// `spec_verify_rows` 决定（现在 16）；`backend.zig` 的 `GraphCtl` 用同一个数字，
// 两边的 comptime 断言把它钉住。
#define XD_CTL_TOKENS    16

#define XD_CTL_POS       0   // rope phase base (row 0 of this step)
#define XD_CTL_BASE_BITS 1   // bit pattern of the rope theta
#define XD_CTL_KV_SLOT   2   // first KV row written by this step
#define XD_CTL_ATTN_SLOT 3   // last KV row full attention must cover
#define XD_CTL_ROWS      4   // rows this step produces
#define XD_CTL_TOK       5   // 本步各行的嵌入 token id（`XD_CTL_TOKENS` 个）
// 采样器的 16 个字段接在嵌入行号之后（见 `sample.cu` 的 `SCTL_BASE`）。
// 两者是**同一块**常驻缓冲：采样 kernel 是图的最后一个节点，它读的状态和
// 前 65 层是同一时刻的，由同一次 `publishCtl` 整块写入。
#define XD_SCTL_BASE     (XD_CTL_TOK + XD_CTL_TOKENS)   // 采样字段的起始下标 = 21
#define XD_SCTL_INTS     16
#define XD_CTL_INTS      (XD_SCTL_BASE + XD_SCTL_INTS)  // 整块 = 37

static_assert(sizeof(int) == 4, "the control block is an int32 array");

// A kernel that needs one per-step scalar takes `const int *ctl` and reads
// `ctl[XD_CTL_<field>]`.  The named offsets above keep call sites from repeating
// the arithmetic (and from getting it wrong).

// ---- argument entry constructors ------------------------------------------
void xd_set_f32(xd_slot *s, const float *src, float v);
void xd_set_i32(xd_slot *s, const int *src, int v);
void xd_set_u32(xd_slot *s, const unsigned *src, unsigned v);
void xd_set_ptr(xd_slot *s, void *p);

// ---- introspection ---------------------------------------------------------
int         xd_op_count(void);
int         xd_op_exists(int op);
const char *xd_op_name(int op);

// ---- warm / eager launch ---------------------------------------------------
// Launch one op with its arguments, on `stream`.  Returns 0 or a cudaError_t
// value.  Never touches a graph.
int xd_launch_direct(int op, const xd_slot *a, void *stream);

// ---- record ----------------------------------------------------------------
// Call with a stream capture OPEN.  Launches the op once (so CUDA records it)
// and stores the nodes that launch produced into `out`.
//
// THE GEOMETRY COMES FROM THE CALLER.  `pgrid` / `pblock` / `psmem` are the same
// shapes the EAGER launcher would use (the backend computes them; several
// kernels' shapes depend on run-time dispatch, e.g. the coop/split matvec), so
// the recorded node is byte-identical to the eager one.  The `shape()` columns
// in xd.cu are the pure-function reference for the shapes that ARE decidable
// from the arguments, kept so a mismatch is visible in xtest / XD_DUMP.
//
//   * `out->op`, `out->n` are set from `op` and `n` (`n` <= XD_MAX_SLOTS);
//   * `out->slots[i]` are copied from `a[i]`, so a later `xd_apply` can refresh
//     them without the caller holding the array;
//   * `out->gnode[0..nnodes)` are the recorded nodes;
//   * `out->argv` / `out->gate_argv` point at `out->vals`, ready for
//     `xd_apply`.
// Returns 0 on success.  A non-zero return is a cudaError_t value AND means the
// caller must abandon the capture (the launch is already part of it).
int xd_capture(int op, const xd_slot *a, int n, void *stream,
               unsigned gx, unsigned gy, unsigned gz,
               unsigned bx, unsigned by, unsigned bz, unsigned smem,
               xd_node *out);

// ---- replay preparation ----------------------------------------------------
// Copy every immediate value from its `src` cell (a NULL src keeps the capture
// value), take every pointer's current value, then hand every node of this op
// its argument array with `cudaGraphExecKernelNodeSetParams`.  `gexec` is the
// instantiated graph the nodes belong to.
int xd_apply(xd_node *node, void *gexec);

// The table's own geometry for `op` given its arguments.  A caller may use it
// when it does not need the eager launcher's run-time dispatch; `xd_capture`
// takes its geometry from the caller regardless.
int xd_op_shape(int op, const xd_slot *a, xd_shape *out);

// Same as `xd_launch_direct`, but with an explicit geometry.  S1 ignores the
// geometry (the table launches with its own); kept in the interface so S3 can
// switch the callers over without another ABI change.
int xd_launch_span(int op, const xd_slot *a, void *stream,
                   unsigned gx, unsigned gy, unsigned gz,
                   unsigned bx, unsigned by, unsigned bz, unsigned smem);

// ---- per-op record entry points (S3c) --------------------------------------
//
// WHY THESE EXIST.  A captured op must be launched from an `xd_slot` array:
// that is what `xd_apply` refreshes at replay time.  The backend's forwarders
// hold the kernel's OWN parameters (d_out, d_x, rows, ...), in the kernel's
// order, not the table's slot order, and the two differ on purpose -- the table
// also carries values that only size the grid and are not kernel arguments
// (e.g. rmsnorm's `rows` is slot 0 and is NOT passed to `k_rmsnorm`).
//
// Writing the mapping in the backend would put a hand-copied slot table in 26
// op bodies; here each mapping sits next to the table row that consumes it, and
// the compiler checks it against the row's `launch` lambda.  One narrow function
// per op, expanding to `xd_slot[]` + `xd_capture`.
//
// Each returns what `xd_capture` returns: 0 on success; anything else is a
// cudaError_t AND the launch is already part of the capture, so the caller must
// abandon it (it does: `graphAbandon`).
//
// `out` is the node the recorder stores; `stream` must be the capture stream.
int xd_rec_rope64_dpos(void *stream, xd_node *out,
                       float *d_x, int heads, int hd, const void *d_ctl);
int xd_rec_kv_quant_dslot(void *stream, xd_node *out,
                          const float *d_x, void *d_out, const int *d_slot,
                          int nblocks_per_row, int rows_this, int row_bytes);
int xd_rec_gdn_step_rows(void *stream, xd_node *out,
                         float *d_state, int state_stride,
                         const float *d_k, int k_stride,
                         const float *d_v, int v_stride,
                         const float *d_q, int q_stride,
                         float *d_km, float *d_delta, float *d_out,
                         const float *d_gamma, const float *d_beta,
                         int heads, int nk, int kd, int vd);
int xd_rec_gdn_gamma(void *stream, xd_node *out,
                     const float *d_a_raw, float *d_b_raw,
                     const float *d_ssm_a, const float *d_dt_bias,
                     float *d_gamma, int nv);
int xd_rec_rmsnorm(void *stream, xd_node *out,
                   float *d_out, const float *d_x, const float *d_w,
                   int rows, int n, float eps);
int xd_rec_swiglu(void *stream, xd_node *out, float *d_gate, const float *d_up, int n);
int xd_rec_silu(void *stream, xd_node *out, float *d_x, int n);
int xd_rec_add_resid(void *stream, xd_node *out, float *d_dst, const float *d_src, int n);
int xd_rec_copy_range(void *stream, xd_node *out, float *d_dst, const float *d_src, int n);
int xd_rec_scale_vec(void *stream, xd_node *out, float *d_x, float s, int n);
int xd_rec_zero_buf(void *stream, xd_node *out, float *d_buf, int n);
int xd_rec_deinterleave_qg(void *stream, xd_node *out,
                           float *d_q, float *d_gate, const float *d_src,
                           int heads, int hd);
int xd_rec_argmax(void *stream, xd_node *out, const float *d_x, int n, int *d_out);
int xd_rec_argmax_batch(void *stream, xd_node *out, const float *d_x, int n, int rows, int *d_out);
int xd_rec_set_int1(void *stream, xd_node *out, int *d, int v);

#ifdef __cplusplus
}
#endif

#endif // Q35_CUDA_XD_H
