//! Hardware abstraction boundary.
//!
//! The engine models *one* architecture (Qwen3.5) and must not know how a
//! matvec is executed.  Everything that depends on the hardware -- which ISA
//! tier is available, whether weights are streamed from host memory or from
//! device memory, how a quantization block is unpacked -- lives behind `Exec`.
//!
//! Layout:
//!
//!   gguf.QuantRow   packed tensor + its layout geometry (no maths, no hardware)
//!   backend.zig     this file: the `Exec` interface + `LinearW` + dispatch
//!   backend_cpu.zig CPU implementation (C kernels under src/kernels/x86)
//!   backend_cuda.zig CUDA implementation (currently a CPU trampoline; see file)
//!
//! Adding a backend means implementing `Exec` and registering it in
//! `selectBackend`; the engine does not change.
//!
//! Activation and model state (residual stream, projections, KV cache, GDN
//! conv history and 128x128 state, logits) are **backend-owned regions**
//! (`Buf`), not host slices the engine dereferences.  Every per-op call takes
//! `Buf`s, so the arithmetic runs *inside* the backend: on a device backend the
//! activations stay on the device and nothing crosses PCIe between two ops; the
//! only host-visible values are the token ids the engine feeds in and the token
//! id it gets back.
//!
//! The CPU backend implements each elementwise slot with the same Zig loops the
//! engine used to contain (src/elementwise.zig), byte-for-byte, so the CPU path
//! is bit-identical with the historical engine.

const std = @import("std");
const gguf = @import("gguf.zig");
const sampler = @import("sampler.zig");

pub const QuantRow = gguf.QuantRow;

/// A 2-D linear weight as the engine sees it: either still packed in its
/// quantization format (the normal case, zero-copy -- this includes f16 and
/// bf16, which stay 2 B/elem and are widened inside the dot kernels) or, for
/// `.f32` tensors, the native f32 slice.
///
/// Both forms are zero-copy views into the GGUF mapping: `.f32` is the tensor
/// bytes reinterpreted as `[]const f32` (the mapping pads every tensor to a
/// 32-byte alignment, so the 4-byte f32 alignment holds).  Nothing in either
/// arm is allocated, so a `LinearW` carries no destructor of its own -- the
/// view is valid only while the mapping is, and a device backend uploads from
/// it without copying host-side first.
pub const LinearW = union(enum) {
    quant: QuantRow,
    f32: []const f32,
};

/// A region of activation / state memory owned by the backend.
///
/// This is what makes "skeleton on the host, arithmetic on the device" possible.
/// The engine no longer holds `[]f32` scratch it can only touch in host memory:
/// it holds opaque regions the backend may have made device-resident, and every
/// per-op call takes `Buf`s, so elementwise kernels (`rmsnorm`, `rope64`,
/// `convStep`, `gdnStep`, ...) read and write activations *inside* the backend
/// — nothing crosses PCIe between two ops.
///
/// `ptr` is a host pointer for the CPU backend and a device pointer for a
/// device backend; the engine must never dereference it.  `slice()` exists for
/// the CPU backend (and for the matvec/dequant signatures, which take plain
/// slices: on a device backend that slice is a device pointer, which the
/// backend's own kernel treats as such).
pub const Buf = struct {
    ptr: [*]f32,
    len: usize,

    pub fn slice(self: Buf) []f32 {
        return self.ptr[0..self.len];
    }
    /// Sub-window `[off, off+len)`.  Valid on both sides: pure pointer
    /// arithmetic on a `Buf`, the backend resolves its own base.
    pub fn span(self: Buf, off: usize, len: usize) Buf {
        std.debug.assert(off + len <= self.len);
        return .{ .ptr = self.ptr + off, .len = len };
    }
    /// Wrap an existing host slice (the CPU case for GGUF-side tensors: norm
    /// weights, conv taps, rope tables).  A device backend must instead upload
    /// the tensor at load time and hand back a device-backed `Buf` — see
    /// `loadVecF32`.
    pub fn wrap(s: []const f32) Buf {
        return .{ .ptr = @constCast(s.ptr), .len = s.len };
    }

    /// The "nothing allocated" sentinel: `len == 0`, pointer never
    /// dereferenced.  Use it for optional/unallocated scratch fields instead
    /// of `&.{}`, which is a host-slice value and says nothing about which
    /// address space the region lives in.
    pub const empty: Buf = .{ .ptr = undefined, .len = 0 };
};

/// Storage format of one full-attention layer's KV cache rows.
///
/// `f32` is the historical layout (one float per element).  `f16` is the
/// default: 2 B/elem, 10-bit mantissa, near-lossless for post-norm K/V
/// values.  `q8_0`/`q4_0` reuse the project's OWN block layouts (the same
/// ones the weight tensors use): 34 B per 32 elements (f16 `d` + 32 x i8,
/// dequant `d*q`) and 18 B per 32 elements (f16 `d` + 16 nibbles, symmetric
/// `(q-8)*d`) -- see `quant_dequant.zig` for the canonical formulas.  The
/// block types need `kv_dim % 32 == 0` (`kvTypeFits`); every model has it,
/// because `kv_dim` is a multiple of `head_dim` and `head_dim` of 64.
pub const KvType = enum { f32, f16, q8_0, q4_0 };

/// Bytes of ONE KV row (one token, `kv_dim` elements) in format `t`.
pub fn kvRowBytes(kv_dim: usize, t: KvType) usize {
    return switch (t) {
        .f32 => kv_dim * @sizeOf(f32),
        .f16 => kv_dim * 2,
        .q8_0 => (kv_dim / 32) * 34,
        .q4_0 => (kv_dim / 32) * 18,
    };
}

/// Whether a `kv_dim`-wide row is representable in format `t`.
pub fn kvTypeFits(kv_dim: usize, t: KvType) bool {
    return switch (t) {
        .f32, .f16 => true,
        .q8_0, .q4_0 => kv_dim % 32 == 0,
    };
}

/// The two halves of one layer's KV cache, as the engine sees them.
pub const KvRegion = struct {
    k: Buf,
    v: Buf,
};

/// Errors every backend can report while loading weights.
pub const LoadError = error{
    MissingTensor,
    UnsupportedTensorType,
    OutOfMemory,
    /// The selected backend (e.g. `--backend cuda`) cannot run on this host:
    /// no libcudart, no device, or a device too old for the compiled kernels.
    /// The reason is printed by the backend itself, because this error set only
    /// carries names.
    BackendUnavailable,
};

/// What shape of forward a graph captures.  **Decode** is the one-row forward
/// (`forwardGdn`/`forwardFull`/`forwardMlp`); **chunk** is the K-row forward
/// (`forwardChunkRun`), shared by prefill and by the speculative verify.
pub const GraphKind = enum { decode, chunk };

/// Per-step scalars of one graph.  This is the HOST mirror of the device control
/// block (`GraphCtlDev` in `src/kernels/cuda/xd.h`): the backend copies it to
/// the device once per step, so a per-token scalar never has to be retargeted on
/// a graph node (see plan §1.5, `cell`).
///
/// EVERY field here is read by a kernel through `const int *`/`const float *`
/// pointing at the device block, never passed by value.  That is the whole point:
/// a captured node's argument list stays constant, so a replay needs no
/// `cudaGraphExecKernelNodeSetParams` for any of them.
///
/// The CPU backend stores one of these as a plain value and never reads it.
pub const GraphCtl = extern struct {
    /// Position of this step's first row (`rope64`'s phase base).
    pos: u32 = 0,
    /// Bit pattern of the RoPE `base` (theta), as an `i32` so the block is one
    /// uniform int array the publish kernel copies verbatim.  The engine fills it
    /// with `@bitCast(base)`; kernels read it with `__int_as_float`.
    rope_base_bits: i32 = 0,
    /// First KV row this step writes (`pos` for decode, `slot0` for a chunk).
    kv_slot: u32 = 0,
    /// KV row the full-attention kernels must cover (`slot` = `kv_slot` for
    /// decode; for a chunk the kernels derive it from `kv_slot + rows`).
    attn_slot: u32 = 0,
    /// Rows this step produces (1 for decode, k for a verify chunk).
    rows: u32 = 1,
};

comptime {
    // The device block is a `int32_t[GRAPH_CTL_INTS]`; this must stay 5 ints
    // wide.  Widening it needs the .cu side's array size changed in step.
    std.debug.assert(@sizeOf(GraphCtl) == 5 * @sizeOf(i32));
    std.debug.assert(@alignOf(GraphCtl) == 4);
}

/// A forward pass being recorded, replayed, or run straight through.
///
/// `ptr == null` is the **pass-through** graph: every op behaves exactly as it
/// did before this type existed.  The CPU backend always returns that, and so
/// does a device backend when graphs are disabled or have not been built yet.
/// A non-null `ptr` is a backend-owned handle (the CUDA backend stores the
/// registry entry there), and `ctl` is this step's scalars for it.
pub const Graph = struct {
    ptr: ?*anyopaque = null,
    kind: GraphKind = .decode,
    ctl: GraphCtl = .{},

    /// The graph a caller passes when it wants the historical direct path.
    pub const direct = Graph{};
};

/// The hardware boundary.  One instance per loaded model.
///
/// Every method the engine calls during a forward pass is here; the engine
/// holds `*Exec` and never names a concrete backend.
pub const Exec = struct {
    ptr: *anyopaque,
    vtable: *const VTable,

    pub const VTable = struct {
        // ---- weight loading -------------------------------------------------
        /// Load one 2-D weight, keeping quantized tensors packed.
        loadLinear: *const fn (ctx: *anyopaque, name: []const u8) LoadError!LinearW,
        /// Release what `loadLinear`/`allocScratch` allocated.
        freeLinear: *const fn (ctx: *anyopaque, w: LinearW) void,
        /// Load one 1-D f32 weight vector (a norm's gamma, a conv tap row,
        /// `ssm_a`, `dt_bias`, ...) by tensor name.  The returned `Buf` is a
        /// backend region -- a host pointer on a CPU backend, a device pointer
        /// on a device backend -- that the elementwise slots read directly; the
        /// engine keeps only this handle and never dereferences it.  The tensor
        /// is widened (f16/bf16/quant -> f32) and, on a device backend, uploaded
        /// ONCE here at load time, not per token.
        loadVecF32: *const fn (ctx: *anyopaque, tensor_name: []const u8) LoadError!Buf,
        /// Release what `loadVecF32` allocated.  A `Buf.empty` argument is a
        /// no-op (the sentinel the engine uses for an unallocated optional
        /// vector), so callers can free unconditionally.
        freeVec: *const fn (ctx: *anyopaque, v: Buf) void,

        // ---- activation scratch --------------------------------------------
        /// Allocate the per-token / per-chunk scratch the kernels need.
        /// `max_cols` is the widest row any weight in the model can have.
        allocScratch: *const fn (ctx: *anyopaque, max_cols: usize) anyerror!void,
        /// Grow scratch for a K-token prefill chunk.
        allocScratchK: *const fn (ctx: *anyopaque, max_cols: usize, k: usize) anyerror!void,
        freeScratch: *const fn (ctx: *anyopaque) void,

        /// A region of activation/state memory the backend owns.  The engine
        /// asks for regions by length and never owns the memory itself: a CPU
        /// backend allocates from its allocator and wraps the slice, a device
        /// backend allocates on the device.  Regions live as long as the model
        /// run, released by `freeScratch` (all of them at once).
        allocBuf: *const fn (ctx: *anyopaque, len: usize) anyerror!Buf,

        // ---- KV cache -------------------------------------------------------
        //
        // The full-attention KV cache is the one backend-owned region whose
        // LAYOUT depends on a model-level choice (`KvType`), so the engine
        // never does pointer arithmetic on it: it allocates through `allocKv`,
        // writes rows through `kvAppend`, and hands the WHOLE regions to
        // `attnDecode`/`attnChunk` (which already carry `slot`/`slot0` and
        // `kv_dim`, and so know how many rows are live).  `Buf.len` on the
        // returned regions is in f32 units (`ceil(bytes/4)`); only the
        // backend interprets the bytes.

        /// Storage format for this model's KV cache.  Must be set before the
        /// first `allocKv`/`kvAppend`/attention call; the backend keeps it for
        /// the whole run (one model load has one format).
        setKvType: *const fn (ctx: *anyopaque, t: KvType) void,
        /// Allocate one layer's KV cache: `max_seq` rows of `kv_dim` elements,
        /// in the format `setKvType` chose.  Replaces the historical two
        /// `allocBuf(max_seq * kv_dim)` calls; only the backend knows the
        /// format-dependent byte size.
        allocKv: *const fn (ctx: *anyopaque, kv_dim: usize, max_seq: usize) anyerror!KvRegion,
        /// Quantize `rows` f32 rows (in `src`, `kv_dim` wide, row-major) into
        /// the cache `dst` starting at row `slot`.  `dst` is the layer's WHOLE
        /// k or v region -- the backend strides it by its own row bytes, so the
        /// caller does no format-dependent arithmetic.
        ///
        /// Decode calls this with `rows = 1` per token; on a graph-replaying
        /// backend the destination advances one row per token exactly like the
        /// historical `copyRangeSlot` KV append (which this replaces).  Prefill
        /// calls it with `rows = k` for the chunk, outside any capture.
        kvAppend: *const fn (g: Graph, ctx: *anyopaque, dst: Buf, src: Buf, slot: usize, kv_dim: usize, rows: usize) void,

        // ---- dequantization -------------------------------------------------
        /// Expand one row of a packed weight into `dst` (f32, `cols` wide).
        /// Used by the embedding lookup, which needs values rather than dots.
        dequantRow: *const fn (ctx: *anyopaque, dst: Buf, w: QuantRow, row: usize) void,

        // ---- matvec ---------------------------------------------------------
        /// `out = W * x` with `W` packed, `x` `w.cols()` f32 values.  Both
        /// activation sides are `Buf`s, so on a device backend the input is the
        /// region the previous op wrote and the output is the region the next
        /// op reads: no PCIe between them.
        matvec: *const fn (g: Graph, ctx: *anyopaque, out: Buf, w: LinearW, x: Buf) void,
        /// `matvec(out[0..r], a, x)` then `matvec(out[r..2r], b, x)` as ONE
        /// device launch, when `a` and `b` are the *same* shape and type and
        /// their device ranges are adjacent (`a.dev + a_bytes == b.dev`), so the
        /// two are one `2r x cols` matrix in memory.
        ///
        /// Why this is worth an interface slot.  On the 27B's 48 recurrent
        /// layers `ssm_alpha.weight` and `ssm_beta.weight` are exactly that pair
        /// (verified for all 48 in the GGUF: `alpha.off + alpha.size ==
        /// beta.off`), 48x5120 q8_0 each.  Two launches of 255 KB apiece is
        /// 96 launches per token, and 255 KB does not even reach the launch
        /// floor: measured 3.1-4.4 us at 60-85 GB/s for EVERY block shape tried
        /// (/tmp/shp/bw2.cu, 24..192 blocks x 32..128 threads -- the 3 us is
        /// fixed latency, not traffic).  So 96 x 3.1 us, about 0.3 ms or 1.6% of
        /// the token, buys 25 MB/s of actual work.
        ///
        /// The fallback is two plain `matvec` calls, so a backend (or a GGUF)
        /// that cannot prove adjacency pays exactly what it paid before.
        matvecPair: *const fn (g: Graph, ctx: *anyopaque, out: Buf, a: LinearW, b: LinearW, x: Buf) void,
        /// Same, for `k` tokens at once: `out[j*rows + r] = W[r] . x_j`.
        matvecBatchK: *const fn (g: Graph, ctx: *anyopaque, out: Buf, w: LinearW, xK: Buf, k: usize) void,

        /// Create the intra-op thread pool (CPU); a device backend ignores it.
        /// `nt` is the worker count, 1 = serial.
        setThreads: *const fn (ctx: *anyopaque, io: std.Io, nt: usize) void,

        // ---- elementwise / state kernels -------------------------------------
        //
        // These used to be plain Zig loops inside the engine, which required the
        // activations to live in host memory.  They are backend calls now: the
        // engine hands `Buf`s (regions the backend owns), so a device backend
        // runs the arithmetic on the device and activation never leaves it.
        // The CPU backend implements each slot with the same Zig loops the
        // engine used to contain, byte-for-byte (src/elementwise.zig), so the
        // CPU path is bit-identical to the historical engine.

        /// out = x * rsqrt(mean(x^2) + eps) * w, all `n` wide.
        rmsnorm: *const fn (g: Graph, ctx: *anyopaque, out: Buf, x: Buf, w: Buf, eps: f32) void,
        /// `rows` consecutive `rmsnorm` calls on `out.span(r*h, h)` / `x.span(r*h, h)`.
        ///
        /// The ROW-BATCHED form: `out` and `x` are `rows * w.len` long and pack
        /// `rows` consecutive rows of `w.len` floats; `w` stays ONE row and is
        /// reused for every row.  Every row is normalised independently (its own
        /// sum of squares), so this is exactly `rows` consecutive `rmsnorm`
        /// calls on `out.span(r*h, h)` -- one host launch instead of `rows`.
        ///
        /// It is a SEPARATE slot rather than "infer rows from the lengths" (which
        /// `out.len == rows * w.len` would allow) because `copyRange` feeds host
        /// embedding rows through this vtable: a host-only `Buf` of length
        /// `rows * n` must NOT silently become a device launch.  A batch caller
        /// has to ask for the batch reading explicitly.
        rmsnormRows: *const fn (g: Graph, ctx: *anyopaque, out: Buf, x: Buf, w: Buf, rows: usize, eps: f32) void,
        /// `rows` consecutive `l2norm` calls on `x.span(r*n, n)`.  Same
        /// "explicit, never inferred" rule as `rmsnormRows`; each row's eps is
        /// inside its own sum, so the rows are fully independent.
        l2normRows: *const fn (g: Graph, ctx: *anyopaque, x: Buf, n: usize) void,
        /// Rotate the 64-dim rotary block of a row in place (cross-half pairing).
        rope64: *const fn (g: Graph, ctx: *anyopaque, x: Buf, pos: u32, base: f32) void,
        /// `rows` consecutive `rope64` calls on `x.span(r * heads * hd, heads * hd)`
        /// with `pos + r`, i.e. row r is prefill token `pos + r`.  `heads` is the
        /// head count PER ROW, so `x.len == rows * heads * hd`.  Same
        /// "explicit, never inferred" rule as `rmsnormRows`.
        ///
        /// PREFILL ONLY in practice: the decode path must keep issuing single-row
        /// `rope64` calls because the decode CUDA graph identifies and retargets
        /// those nodes by host-stub address (see `q35_func_rope64`).  Prefill
        /// never captures a graph, so the batch reading is free there.
        rope64Rows: *const fn (g: Graph, ctx: *anyopaque, x: Buf, pos: u32, base: f32, rows: usize, heads: usize, hd: usize) void,
        /// RoPE over `rotary_dim` of each head, for architectures that are not
        /// Qwen3.5/3.8.  Qwen3 rotates the WHOLE head (`rotary_dim == hd`),
        /// so `rope64Rows` cannot express it.
        ///
        /// Separate slot on purpose: `rope64Rows` is what the decode CUDA graph
        /// identifies and retargets by host-stub address for Qwen3.5, and adding
        /// a parameter there would invalidate every captured node.  Backends
        /// that do not implement this may fall back to their CPU twin.
        ropeRows: *const fn (g: Graph, ctx: *anyopaque, x: Buf, pos: u32, base: f32, rows: usize, heads: usize, hd: usize, rotary_dim: usize) void,
        /// x *= rsqrt(sum(x^2) + 1e-6)
        l2norm: *const fn (g: Graph, ctx: *anyopaque, x: Buf) void,
        /// x *= s
        scaleVec: *const fn (g: Graph, ctx: *anyopaque, x: Buf, s: f32) void,
        /// The GDN q/k normalisation as ONE launch: `nk` q rows at `x[0..]` and
        /// `nk` k rows at `x[k_offset..]`, each an independent l2norm of `kd`
        /// floats (eps inside the sum), then the q rows scaled by `qscale`.
        /// Bit-identical to `l2normRows(q)` + `l2normRows(k)` + `scaleVec(q)`,
        /// which is 3 launches where this is 1 -- and at 3.7 us per launch
        /// (measured, /tmp/shp/lch.cu) that matters: a decode token issues 1220
        /// kernels, so the 48 recurrent layers' 96 saved launches are 1.8% of
        /// the token.  Added after decode RoPE batching and the GDN l2norm fix
        /// had both landed; see docs in norm.cu.
        l2normQkScale: *const fn (g: Graph, ctx: *anyopaque, x: Buf, nk: usize, kd: usize, k_offset: usize, qscale: f32) void,
        /// x = silu(x) in place
        siluInPlace: *const fn (g: Graph, ctx: *anyopaque, x: Buf) void,
        /// out[i] *= silu(gate[i])
        mulSilu: *const fn (g: Graph, ctx: *anyopaque, out: Buf, gate: Buf) void,
        /// out[i] *= sigmoid(gate[i])
        mulSigmoid: *const fn (g: Graph, ctx: *anyopaque, out: Buf, gate: Buf) void,
        /// softmax over the whole slice
        softmax: *const fn (g: Graph, ctx: *anyopaque, x: Buf) void,
        /// dst[i] += src[i]
        addResid: *const fn (g: Graph, ctx: *anyopaque, dst: Buf, src: Buf) void,
        /// SwiGLU: gate[i] = silu(gate[i]) * up[i], in place
        swiglu: *const fn (g: Graph, ctx: *anyopaque, gate: Buf, up: Buf) void,
        /// Gated Delta Net: one token of the causal conv (history `conv` is
        /// `y.len * 3` floats, oldest..newest) + silu, in place on `y`.
        convStep: *const fn (g: Graph, ctx: *anyopaque, conv: Buf, y: Buf, wconv: Buf, k: usize) void,
        /// Gated Delta Net: one token of the delta-rule state update.
        /// state is [nv, kd, vd] (kd == vd == 128), `k`/`q` are kd wide,
        /// `v`/`km`/`delta`/`out` are vd wide, per head.
        gdnStep: *const fn (g: Graph, ctx: *anyopaque, state: Buf, k: Buf, v: Buf, q: Buf, km: Buf, delta: Buf, out: Buf, gamma: f32, beta: f32, vd: usize) void,
        /// Batched `gdnStep`: the SAME per-head call, `heads` times, in one
        /// launch instead of `heads` launches (48 of them per token per layer on
        /// this model).  Semantics are defined as "for h in 0..heads-1: the
        /// `gdnStep` call that `h` would have made", so the CPU implementation
        /// here is literally that loop.
        ///
        /// Rows are *strided bases*, not a packed array: the engine's regions
        /// interleave (a v-head run of `vd` floats inside `qkv`, a
        /// `kd*vd` state block per head), so each operand gets an explicit
        /// element stride.  `heads`, `nk` and `kd`/`vd` are passed for the same
        /// "explicit, never inferred" reason as `rmsnormRows`.
        ///
        ///   state[*h*st_state .. +kd*vd]   the [kd, vd] matrix for head h
        ///   k     [kh*kd .. +kd],  kh = h % nk      (kv-head mapping: a v head
        ///   q     [kh*kd .. +kd]    reuses k head h%nk, and `q` is implicitly
        ///                                            scaled by 1/sqrt(kd))
        ///   v     [h*vd  .. +vd]
        ///   out   [h*vd  .. +vd]
        ///   km    [h*vd  .. +vd]   PER-HEAD scratch, `heads*vd` long
        ///   delta [h*vd  .. +vd]   PER-HEAD scratch, `heads*vd` long
        ///   gamma [h], beta [h]    per-head scalars, in backend regions
        ///
        /// `km`/`delta` must be per-head (stride vd): with a single shared `vd`
        /// buffer the `heads` blocks would race, and the wrong answer would be
        /// silent.  The caller sizes them.
        ///
        /// `gamma`/`beta` are `Buf`s, i.e. backend regions (device memory on a
        /// device backend) of at least `heads` floats -- `gdnGamma`'s output
        /// and its in-place sigmoid(b) write-back respectively.  Passing them
        /// as regions is what keeps a token from making `2 * heads` floats of
        /// PCIe round trip per GDN layer (read back for an `f32` argument,
        /// then re-uploaded): a device backend just reads its own memory.
        ///
        /// Their CONTENTS change every token (per-layer, per-position), so they
        /// must NEVER be routed through a `devConst`-style cache: those key on
        /// pointer identity and never re-upload, which would silently serve the
        /// first layer's coefficients forever.
        gdnStepRows: *const fn (g: Graph, ctx: *anyopaque, state: Buf, k: Buf, v: Buf, q: Buf, km: Buf, delta: Buf, out: Buf,
            gamma: Buf, beta: Buf, heads: usize, nk: usize, kd: usize, vd: usize) void,

        /// Chunk-wide twin of `gdnStepRows` for prefill: `rows` successive state
        /// steps in ONE call.  Row r is exactly the `gdnStepRows` call on the
        /// r-th slice of every strided operand, and the steps are inherently
        /// SEQUENTIAL (`state` is carried from row to row), so this is not a
        /// parallelism win -- it is a LAUNCH-count win.
        ///
        /// Rationale, measured: the per-token loop issued `rows` launches of a
        /// ~4 us kernel per GDN layer.  On a 525-token prefill that is 10,500
        /// launches whose device time totalled 42 ms while the wall clock they
        /// occupied was 641 ms -- the host could not submit them fast enough.
        /// Deleting the loop entirely (wrong results, measurement only) took
        /// prefill from 299 to 453 tok/s, which puts the whole launch overhead
        /// at ~594 ms.  Fusing the loop into the kernel recovers it without
        /// touching the arithmetic: same statements, same __fmul_rn/__fadd_rn
        /// pinning, same i-order, only the row index becomes a loop variable.
        ///
        /// `gamma`/`beta` are per-head coefficient rows (`heads` floats each
        /// per token), so they stride the same way `out` does.
        ///
        /// Explicit strides: `k` and `q` live inside one `qkv` row of
        /// `2*nk*kd + heads*vd` floats, so their per-token stride is the whole
        /// row, NOT `nk*kd`.  Inferring it from the buffer length would read the
        /// gate half as if it were the next token's k.
        gdnStepRowsChunk: *const fn (g: Graph, ctx: *anyopaque, state: Buf, k: Buf, v: Buf, q: Buf, km: Buf, delta: Buf, out: Buf,
            gamma: Buf, beta: Buf, heads: usize, nk: usize, kd: usize, vd: usize, rows: usize,
            k_stride: usize, v_stride: usize, q_stride: usize) void,

        /// `gdnStepRowsChunk` 加"每行结束后写一份状态快照"。
        ///
        /// 目的（推测解码）：verify 块把 `k` 行推进状态，但只有被接受的
        /// `r <= k` 行该留下。没有快照时回滚要在每一层重跑 `r` 行递推
        /// （实测占 MTP 总时间 73.9%）；有快照时回滚只是把 state 视图指到
        /// 第 `r` 个槽——零计算、零拷贝。与 llama.cpp 的 `n_rs_seq` 同一交易。
        ///
        ///   * `snap` 是 `nsnap * heads * kd * vd`，槽 `t` = 跑完第 t+1 行之后
        ///     的状态。第 0 行**之前**的状态不写（调用方自己有）。
        ///   * 空 `snap` 表示不要快照；被拒绝的形状同样退化为普通 chunk 递推。
        gdnStepRowsSnap: *const fn (g: Graph, ctx: *anyopaque, state: Buf, k: Buf, v: Buf, q: Buf, km: Buf, delta: Buf, out: Buf,
            gamma: Buf, beta: Buf, snap: Buf, heads: usize, nk: usize, kd: usize, vd: usize, rows: usize,
            k_stride: usize, v_stride: usize, q_stride: usize) void,

        /// Full-attention decode for one token: scores against the KV rows,
        /// softmax, PV accumulation into `gout`, then the sigmoid gate.  `kcache`
        /// / `vcache` are this layer's WHOLE KV regions (`kv_dim` elements per
        /// token row, `slot+1` rows written so far); the backend strides them in
        /// its own `KvType` layout.
        attnDecode: *const fn (g: Graph, ctx: *anyopaque, q: Buf, gate: Buf, kcache: Buf, vcache: Buf,
            att: Buf, gout: Buf, slot: usize, heads: usize, gqa: usize, hd: usize,
            kv_dim: usize, scale: f32) void,
        /// Chunk-wide twin of `attnDecode` for prefill: `rows` tokens in ONE call
        /// instead of `rows` calls.
        ///
        /// `q`/`gate`/`gout` are `rows * heads * hd` (row r at
        /// `r*heads*hd + h*hd`).  Row r attends to cache rows `[0, slot0+r]`
        /// inclusive, so the rows are still ordered inside the backend -- what
        /// this removes is the per-token KERNEL BOUNDARY, not the dependency
        /// (8400 launches -> 16 for a 525-token 27B prefill).
        ///
        /// PRECONDITIONS the caller owns (this does the score/softmax/PV/gate
        /// only): q/k RoPE, q_norm/k_norm and the [q|gate] deinterleave are done
        /// chunk-wide already, and `kcache`/`vcache` are the layer's WHOLE
        /// regions, holding rows `[0, slot0+rows)` (the chunk's own k/v
        /// included).  `att` is scratch, unused by the CUDA
        /// path; a backend may need it and may grow it.
        ///
        /// Equivalent to calling `attnDecode` once per row with
        /// `slot = slot0 + r`, ULP for ULP -- the CUDA kernel was gated against
        /// exactly that (`/tmp/gate_attn_chunk.cu`: 23 rows, bit-identical).
        attnChunk: *const fn (g: Graph, ctx: *anyopaque, q: Buf, gate: Buf, kcache: Buf, vcache: Buf,
            att: Buf, gout: Buf, slot0: usize, rows: usize, heads: usize, gqa: usize,
            hd: usize, kv_dim: usize, scale: f32) void,
        /// GDN per-token beta/gamma from the raw a/b projections.
        gdnGamma: *const fn (g: Graph, ctx: *anyopaque, a_raw: Buf, b_raw: Buf, ssm_a: Buf, dt_bias: Buf, gamma: Buf, nv: usize) void,

        /// GDN chunk prelude: the three per-token GDN steps that do NOT depend on
        /// the running state, done for a whole prefill chunk in a handful of
        /// launches instead of per token.
        ///
        /// `forwardGdnCore` needs `tokens` x {convStep, gdnGamma, l2norm(x 2
        /// runs x nk_reps), scaleVec(x nk_reps)} before the first `gdnStepRows`.
        /// None of those read or write `state`, and `gdnStepRows` only *reads*
        /// qkv -- it writes state/km/delta/out -- so they can all run up front.
        /// The one genuine coupling is the conv history: token t's taps read what
        /// token t-1 shifted in, so a backend MUST process the tokens in ascending
        /// order (the CUDA kernel does, inside one thread per channel).
        ///
        /// Semantics are defined as exactly the CPU bodies below, so a backend
        /// is free to fuse/launch as it likes as long as the numbers match.
        /// Every region is `tokens` rows of `row_stride` floats:
        ///   qkv   [t*row_stride .. +conv_dim]   conv + silu applied in place
        ///   ab    [t*a_stride .. +nv]           a, and [t*b_stride .. +nv] b
        ///   gamma [t*gamma_stride .. +nv]       written by gamma, shared by tokens
        ///   l2q   [t*row_stride + row_offset .. ] the q run, normalised nk_reps
        ///   l2k   [t*row_stride + kq_offset ..]  the k run, same
        /// `nk_reps` is `nv / nk` and is applied to each run in exactly the order
        /// the per-head loop did (q then k, then the q scale, repeated).
        gdnPrefillCore: *const fn (g: Graph, ctx: *anyopaque, qkv: Buf, conv: Buf, wconv: Buf,
            ab: Buf, ssm_a: Buf, dt_bias: Buf, gamma: Buf,
            tokens: usize, conv_dim: usize, ab_a_offset: usize, ab_b_offset: usize,
            ab_stride: usize, gamma_stride: usize,
            kq_dim: usize, nv: usize, nk: usize, kd: usize, conv_k: usize, eps: f32) void,
        /// Region-to-region copy (KV writes, GDN state snapshots): a device
        /// backend maps this to cudaMemcpy, so a snapshot never touches host.
        copyRange: *const fn (g: Graph, ctx: *anyopaque, dst: Buf, src: Buf) void,
        /// `copyRange` whose destination advances once per token: the KV cache
        /// append, `dst = cache.k/v + slot * kv_dim`.
        ///
        /// Identical to `copyRange` for every backend except one that replays a
        /// captured schedule.  Such a backend snapshots a launch's arguments,
        /// and a destination that moves per token would then be re-recorded to
        /// the capture token's row forever -- the cache never grows and each
        /// token attends to the prefix it saw the first time.  Wrong numbers, no
        /// error, so the slot has to be declared rather than inferred: a bare
        /// device pointer carries no trace of which row of which cache it is.
        copyRangeSlot: *const fn (g: Graph, ctx: *anyopaque, dst: Buf, src: Buf, slot: usize) void,
        /// Split the interleaved `attn_q` projection into `q` and `gate`.
        ///
        /// The GGUF `attn_q` rows are ordered `[h][q_head | gate_head]` -- one
        /// `hd`-wide q block then one `hd`-wide gate block per head -- because
        /// upstream packs them that way; nothing in the engine wants that
        /// layout, both `q` and `gate` are consumed per head.  Doing it as
        /// `2*heads` `copyRange` calls costs one launch per 1 KB on a device
        /// backend (`PROF-COPY`: 768 calls/token, 99% of the `copy_range`
        /// slot), so it gets its own slot and moves the whole `2*q_dim` in one
        /// go.
        ///
        /// `dst_q.len == dst_gate.len == heads*hd`, `src.len == 2*heads*hd`.
        deinterleaveQG: *const fn (g: Graph, ctx: *anyopaque, dst_q: Buf, dst_gate: Buf, src: Buf, heads: usize, hd: usize) void,
        /// Chunk-wide twin of `deinterleaveQG`: `tokens` rows in ONE call.
        /// Row t's `[q|gate]` block starts `t * 2 * heads * hd` floats into
        /// `src`, and row t's q/gate start `t * heads * hd` into each output.
        /// Pure move, no arithmetic -- exactly `tokens` per-row calls.
        deinterleaveQGStrided: *const fn (g: Graph, ctx: *anyopaque, dst_q: Buf, dst_gate: Buf, src: Buf, heads: usize, hd: usize, tokens: usize) void,
        /// Copy a host tensor row into a backend region (the f32 embedding row
        /// when token_embd is unquantized).
        copyFromHost: *const fn (g: Graph, ctx: *anyopaque, dst: Buf, src: []const f32) void,
        /// Copy a small backend region back into a host slice.
        ///
        /// NOT on the per-token path any more: its one historical user was the
        /// per-layer `nv` scalars (`beta`/`gamma`) that `gdnGamma` produces on
        /// the backend, which `gdnStepRows` now takes as `Buf`s and reads on
        /// the device.  It survives for diagnostics and for callers that
        /// genuinely need a host copy of a small region (tests, `QWEN35_DUMP_*`
        /// style tooling, debug paths).  Never use it to bring activations
        /// back -- that is what P2c removed, and a device->host copy also
        /// serialises the stream.
        copyOut: *const fn (g: Graph, ctx: *anyopaque, dst: []f32, src: Buf) void,
        /// Zero a region (reset of the conv history / GDN state).
        zeroBuf: *const fn (g: Graph, ctx: *anyopaque, buf: Buf) void,
        /// Greedy argmax over logits.  On a device backend this runs on the
        /// device and the token id is the only thing that comes back.
        argmax: *const fn (g: Graph, ctx: *anyopaque, logits: Buf) u32,
        /// `argmax` plus the top-1 softmax probability, for the speculative
        /// drafter's confidence early stop.  `p` receives a value in (0, 1];
        /// a backend that cannot produce it cheaply reports 1.0, which makes
        /// the stop a no-op rather than changing any output.
        argmaxConf: *const fn (g: Graph, ctx: *anyopaque, logits: Buf, p: *f32) u32,
        /// `rows` independent argmaxes over `rows * vocab` contiguous logits
        /// (the batched speculative verify's output), written to `out[0..rows)`.
        ///
        /// A separate slot rather than `rows` calls to `argmax` because on a
        /// device each `argmax` ends in a BLOCKING read-back: measured on 27B
        /// MTP, five per-row calls cost 6.5 ms of host wait each against a 9 us
        /// kernel.  One launch + one copy removes that.
        argmaxRows: *const fn (g: Graph, ctx: *anyopaque, logits: Buf, rows: usize, out: []u32) void,
        /// Sampling.  The policy/PRNG object stays in the engine (set from the
        /// CLI); the backend runs the algorithm wherever the logits live, so a
        /// device backend samples on the device and returns a 4-byte token id.
        /// Passing the sampler by pointer keeps its PRNG state advancing
        /// identically on both paths.
        sample: *const fn (g: Graph, ctx: *anyopaque, logits: Buf, scratch: Buf, sp: *sampler.Sampler) u32,


        // ---- explicit compute graph (XG) ------------------------------------
        //
        // A forward pass is handed to the backend as a `Graph`.  The FIRST time
        // a given (kind, rows) is seen the backend runs it straight through and
        // warms its lazy allocations; the second time it RECORDS the calls and
        // instantiates the graph, then launches it once (so the recorded step
        // really executes); every later time it replays.  See
        // `src/graph_registry.zig` for the state machine and plan §1.2.
        //
        // `graphBegin` returns the handle the engine threads through every op of
        // the pass; `graphEnd` closes it.  A CPU backend returns the
        // pass-through graph and does nothing on `graphEnd`, so the engine's
        // code is identical on both.
        graphBegin: *const fn (ctx: *anyopaque, kind: GraphKind, ctl: GraphCtl) Graph,
        graphEnd: *const fn (ctx: *anyopaque, g: *Graph) void,

        /// Mark a phase boundary in the pool's timing counters (no-op for a
        /// device backend, which reports its own timings).
        profPhase: *const fn (ctx: *anyopaque, on: bool) void,
        profReset: *const fn (ctx: *anyopaque) void,
        /// End of the prefill loop.  A CPU backend ignores it (its pool
        /// profiler already separates the phases); the CUDA backend uses it to
        /// close its GPU-side prefill window, because `profPhase(false)` only
        /// fires after the decode loop and would fold decode into prefill.
        profPrefillEnd: *const fn (ctx: *anyopaque) void,


        // ---- introspection --------------------------------------------------
        name: *const fn (ctx: *anyopaque) []const u8,
        /// Human-readable one-liner for the tier actually in use.
        describe: *const fn (ctx: *anyopaque) []const u8,
        /// Restrict implementation, e.g. "baseline"/"avx2"/"avx512".
        setTier: *const fn (ctx: *anyopaque, tier: []const u8) void,

        // ---- profiling ------------------------------------------------------
        /// Matvec-domain counters, printed at shutdown when profiling is on.
        reportProfile: *const fn (ctx: *anyopaque) void,
        /// One-shot *complete* profile dump, called by `Engine.profBackendReport`
        /// at a caller-chosen point (the CLI calls it right after `generate`
        /// returns, gated on `$QWEN35_PROF`).
        ///
        /// Separate from `reportProfile` because that one is the host pool's
        /// table and is also what a device backend's `deinit` would print; the
        /// engine is arena-allocated, so `deinit` never runs on the CLI path and
        /// a device backend's per-type / `PROF-SLOT` / `PROF-COPY` tables would
        /// otherwise only appear at an internal trigger point (`% 500` matvecs,
        /// or the closing of the prefill window) -- which a long EOS-less decode
        /// never reaches, leaving the run with a single header line.
        profReport: *const fn (ctx: *anyopaque) void,
    };

    // Convenience forwarding.  Kept trivial on purpose: these are called on
    // the hot path, so each one is a single indirect call.

    pub fn loadLinear(self: Exec, tensor_name: []const u8) LoadError!LinearW {
        return self.vtable.loadLinear(self.ptr, tensor_name);
    }
    pub fn freeLinear(self: Exec, w: LinearW) void {
        self.vtable.freeLinear(self.ptr, w);
    }
    pub fn loadVecF32(self: Exec, tensor_name: []const u8) LoadError!Buf {
        return self.vtable.loadVecF32(self.ptr, tensor_name);
    }
    pub fn freeVec(self: Exec, v: Buf) void {
        self.vtable.freeVec(self.ptr, v);
    }
    pub fn allocScratch(self: Exec, max_cols: usize) anyerror!void {
        return self.vtable.allocScratch(self.ptr, max_cols);
    }
    pub fn allocScratchK(self: Exec, max_cols: usize, k: usize) anyerror!void {
        return self.vtable.allocScratchK(self.ptr, max_cols, k);
    }
    pub fn freeScratch(self: Exec) void {
        self.vtable.freeScratch(self.ptr);
    }
    pub fn allocBuf(self: Exec, len: usize) anyerror!Buf {
        return self.vtable.allocBuf(self.ptr, len);
    }
    pub fn setKvType(self: Exec, t: KvType) void {
        self.vtable.setKvType(self.ptr, t);
    }
    pub fn allocKv(self: Exec, kv_dim: usize, max_seq: usize) anyerror!KvRegion {
        return self.vtable.allocKv(self.ptr, kv_dim, max_seq);
    }
    /// Quantize `rows` f32 rows into the KV cache `dst` at row `slot`; see
    /// `VTable.kvAppend`.
    pub fn kvAppend(self: Exec, g: Graph, dst: Buf, src: Buf, slot: usize, kv_dim: usize, rows: usize) void {
        self.vtable.kvAppend(g, self.ptr, dst, src, slot, kv_dim, rows);
    }
    pub fn dequantRow(self: Exec, dst: Buf, w: QuantRow, row: usize) void {
        self.vtable.dequantRow(self.ptr, dst, w, row);
    }
    pub fn matvec(self: Exec, g: Graph, out: Buf, w: LinearW, x: Buf) void {
        self.vtable.matvec(g, self.ptr, out, w, x);
    }
    /// Two same-shape matvecs over adjacent weights, as one launch when the
    /// backend can prove the layout.  `out` is `2*rows` long, `a`'s result
    /// first.  Semantically identical to two `matvec` calls.
    pub fn matvecPair(self: Exec, g: Graph, out: Buf, a: LinearW, b: LinearW, x: Buf) void {
        self.vtable.matvecPair(g, self.ptr, out, a, b, x);
    }
    pub fn matvecBatchK(self: Exec, g: Graph, out: Buf, w: LinearW, xK: Buf, k: usize) void {
        self.vtable.matvecBatchK(g, self.ptr, out, w, xK, k);
    }
    pub fn rmsnorm(self: Exec, g: Graph, out: Buf, x: Buf, w: Buf, eps: f32) void {
        self.vtable.rmsnorm(g, self.ptr, out, x, w, eps);
    }
    /// See `VTable.rmsnormRows`: `out` / `x` are `rows * w.len` long, `w` is one
    /// row.  Equivalent to `rows` calls of `rmsnorm` on consecutive sub-rows.
    pub fn rmsnormRows(self: Exec, g: Graph, out: Buf, x: Buf, w: Buf, rows: usize, eps: f32) void {
        self.vtable.rmsnormRows(g, self.ptr, out, x, w, rows, eps);
    }
    pub fn rope64(self: Exec, g: Graph, x: Buf, pos: u32, base: f32) void {
        self.vtable.rope64(g, self.ptr, x, pos, base);
    }
    /// See `VTable.rope64Rows`: `x` is `rows * heads * hd` long, row r (prefill
    /// token `pos + r`) is `heads * hd` floats.
    pub fn rope64Rows(self: Exec, g: Graph, x: Buf, pos: u32, base: f32, rows: usize, heads: usize, hd: usize) void {
        self.vtable.rope64Rows(g, self.ptr, x, pos, base, rows, heads, hd);
    }
    /// RoPE over `rotary_dim` of each head (`rotary_dim == 64` reproduces
    /// `rope64Rows` exactly).
    pub fn ropeRows(self: Exec, g: Graph, x: Buf, pos: u32, base: f32, rows: usize, heads: usize, hd: usize, rotary_dim: usize) void {
        self.vtable.ropeRows(g, self.ptr, x, pos, base, rows, heads, hd, rotary_dim);
    }
    pub fn l2norm(self: Exec, g: Graph, x: Buf) void {
        self.vtable.l2norm(g, self.ptr, x);
    }
    /// See `VTable.l2normRows`: `x` is `rows * n` long.  Equivalent to `rows`
    /// calls of `l2norm` on `x.span(r*n, n)`.
    pub fn l2normRows(self: Exec, g: Graph, x: Buf, n: usize) void {
        self.vtable.l2normRows(g, self.ptr, x, n);
    }
    pub fn scaleVec(self: Exec, g: Graph, x: Buf, s: f32) void {
        self.vtable.scaleVec(g, self.ptr, x, s);
    }
    /// One-launch form of `l2normRows(x[0..nk*kd], kd)`,
    /// `l2normRows(x[k_offset..k_offset+nk*kd], kd)` and
    /// `scaleVec(x[0..nk*kd], qscale)`.  The eps is the GDN one (1e-6), which
    /// the three-call form also uses via `l2normRows`.
    pub fn l2normQkScale(self: Exec, g: Graph, x: Buf, nk: usize, kd: usize, k_offset: usize, qscale: f32) void {
        self.vtable.l2normQkScale(g, self.ptr, x, nk, kd, k_offset, qscale);
    }
    pub fn siluInPlace(self: Exec, g: Graph, x: Buf) void {
        self.vtable.siluInPlace(g, self.ptr, x);
    }
    pub fn mulSilu(self: Exec, g: Graph, out: Buf, gate: Buf) void {
        self.vtable.mulSilu(g, self.ptr, out, gate);
    }
    pub fn mulSigmoid(self: Exec, g: Graph, out: Buf, gate: Buf) void {
        self.vtable.mulSigmoid(g, self.ptr, out, gate);
    }
    pub fn softmax(self: Exec, g: Graph, x: Buf) void {
        self.vtable.softmax(g, self.ptr, x);
    }
    pub fn addResid(self: Exec, g: Graph, dst: Buf, src: Buf) void {
        self.vtable.addResid(g, self.ptr, dst, src);
    }
    pub fn swiglu(self: Exec, g: Graph, gate: Buf, up: Buf) void {
        self.vtable.swiglu(g, self.ptr, gate, up);
    }
    pub fn convStep(self: Exec, g: Graph, conv: Buf, y: Buf, wconv: Buf, k: usize) void {
        self.vtable.convStep(g, self.ptr, conv, y, wconv, k);
    }
    pub fn gdnStep(self: Exec, g: Graph, state: Buf, k: Buf, v: Buf, q: Buf, km: Buf, delta: Buf, out: Buf, gamma: f32, beta: f32, vd: usize) void {
        self.vtable.gdnStep(g, self.ptr, state, k, v, q, km, delta, out, gamma, beta, vd);
    }
    /// Batched `gdnStep` -- see `VTable.gdnStepRows` for the per-head strides.
    /// Byte-for-byte the `heads` single calls it replaces; the engine uses this
    /// one (the single-row slot is kept for fallback and for A/B).
    pub fn gdnStepRows(self: Exec, g: Graph, state: Buf, k: Buf, v: Buf, q: Buf, km: Buf, delta: Buf, out: Buf,
                       gamma: Buf, beta: Buf, heads: usize, nk: usize, kd: usize, vd: usize) void {
        self.vtable.gdnStepRows(g, self.ptr, state, k, v, q, km, delta, out, gamma, beta, heads, nk, kd, vd);
    }
    /// See `VTable.gdnStepRowsChunk`: `rows` sequential `gdnStepRows` steps in
    /// one call, on the successive slices of every strided operand.
    pub fn gdnStepRowsChunk(self: Exec, g: Graph, state: Buf, k: Buf, v: Buf, q: Buf, km: Buf, delta: Buf, out: Buf,
                            gamma: Buf, beta: Buf, heads: usize, nk: usize, kd: usize, vd: usize,
                            rows: usize, k_stride: usize, v_stride: usize, q_stride: usize) void {
        self.vtable.gdnStepRowsChunk(g, self.ptr, state, k, v, q, km, delta, out, gamma, beta, heads, nk, kd, vd, rows, k_stride, v_stride, q_stride);
    }
    /// `gdnStepRowsChunk` 加每行的状态快照，见 `VTable.gdnStepRowsSnap`。
    pub fn gdnStepRowsSnap(self: Exec, g: Graph, state: Buf, k: Buf, v: Buf, q: Buf, km: Buf, delta: Buf, out: Buf,
                           gamma: Buf, beta: Buf, snap: Buf,
                           heads: usize, nk: usize, kd: usize, vd: usize,
                           rows: usize, k_stride: usize, v_stride: usize, q_stride: usize) void {
        self.vtable.gdnStepRowsSnap(g, self.ptr, state, k, v, q, km, delta, out, gamma, beta, snap, heads, nk, kd, vd, rows, k_stride, v_stride, q_stride);
    }
    pub fn attnDecode(self: Exec, g: Graph, q: Buf, gate: Buf, kcache: Buf, vcache: Buf, att: Buf, gout: Buf,
                      slot: usize, heads: usize, gqa: usize, hd: usize, kv_dim: usize, scale: f32) void {
        self.vtable.attnDecode(g, self.ptr, q, gate, kcache, vcache, att, gout, slot, heads, gqa, hd, kv_dim, scale);
    }
    /// `rows` tokens of prefill attention in one call; see `VTable.attnChunk`
    /// for the preconditions and the equivalence to the per-row form.
    pub fn attnChunk(self: Exec, g: Graph, q: Buf, gate: Buf, kcache: Buf, vcache: Buf, att: Buf, gout: Buf,
                     slot0: usize, rows: usize, heads: usize, gqa: usize, hd: usize,
                     kv_dim: usize, scale: f32) void {
        self.vtable.attnChunk(g, self.ptr, q, gate, kcache, vcache, att, gout, slot0, rows, heads, gqa, hd, kv_dim, scale);
    }
    /// beta = sigmoid(b_raw) written IN PLACE into `b_raw`, and
    /// gamma[h] = exp(ssm_a[h] * softplus(a_raw[h] + dt_bias[h])).
    /// `a_raw` and `b_raw` are separate buffers because the batched prefill
    /// writes alpha as `abK[0..k*nv]` and beta as `abK[k*nv..]` -- two
    /// non-contiguous regions -- while the serial path passes the two halves
    /// of one row.
    pub fn gdnGamma(self: Exec, g: Graph, a_raw: Buf, b_raw: Buf, ssm_a: Buf, dt_bias: Buf, gamma: Buf, nv: usize) void {
        self.vtable.gdnGamma(g, self.ptr, a_raw, b_raw, ssm_a, dt_bias, gamma, nv);
    }

    /// GDN chunk prelude -- see `VTable.gdnPrefillCore` for the contract.
    pub fn gdnPrefillCore(self: Exec, g: Graph, qkv: Buf, conv: Buf, wconv: Buf, ab: Buf,
        ssm_a: Buf, dt_bias: Buf, gamma: Buf, tokens: usize, conv_dim: usize,
        ab_a_offset: usize, ab_b_offset: usize, ab_stride: usize,
        gamma_stride: usize, kq_dim: usize, nv: usize, nk: usize, kd: usize,
        conv_k: usize, eps: f32) void {
        self.vtable.gdnPrefillCore(g, self.ptr, qkv, conv, wconv, ab, ssm_a, dt_bias, gamma, tokens, conv_dim, ab_a_offset, ab_b_offset, ab_stride, gamma_stride, kq_dim, nv, nk, kd, conv_k, eps);
    }
    pub fn copyRange(self: Exec, g: Graph, dst: Buf, src: Buf) void {
        self.vtable.copyRange(g, self.ptr, dst, src);
    }

    /// Device -> host copy into a plain host slice.
    ///
    /// `copyRange` cannot serve this: it decides the direction from whether the
    /// SOURCE is a backend region, so a host destination (wrapped as a `Buf`)
    /// is written through as if it were device memory, which aborts the context
    /// on CUDA.  Any caller holding a host buffer uses this instead.
    pub fn copyOut(self: Exec, g: Graph, dst: []f32, src: Buf) void {
        std.debug.assert(dst.len >= src.len);
        self.vtable.copyOut(g, self.ptr, dst[0..src.len], src);
    }


    /// KV-cache append: `copyRange` whose destination is `base + slot * stride`.
    /// See `VTable.copyRangeSlot` for why a replaying backend needs the slot
    /// spelled out.
    pub fn copyRangeSlot(self: Exec, g: Graph, dst: Buf, src: Buf, slot: usize) void {
        self.vtable.copyRangeSlot(g, self.ptr, dst, src, slot);
    }
    pub fn deinterleaveQG(self: Exec, g: Graph, dst_q: Buf, dst_gate: Buf, src: Buf, heads: usize, hd: usize) void {
        self.vtable.deinterleaveQG(g, self.ptr, dst_q, dst_gate, src, heads, hd);
    }
    pub fn deinterleaveQGStrided(self: Exec, g: Graph, dst_q: Buf, dst_gate: Buf, src: Buf, heads: usize, hd: usize, tokens: usize) void {
        self.vtable.deinterleaveQGStrided(g, self.ptr, dst_q, dst_gate, src, heads, hd, tokens);
    }
    pub fn copyFromHost(self: Exec, g: Graph, dst: Buf, src: []const f32) void {
        self.vtable.copyFromHost(g, self.ptr, dst, src);
    }
    pub fn copyToHost(self: Exec, dst: []f32, src: Buf) void {
        self.vtable.copyToHost(self.ptr, dst, src);
    }
    pub fn zeroBuf(self: Exec, g: Graph, buf: Buf) void {
        self.vtable.zeroBuf(g, self.ptr, buf);
    }
    pub fn argmax(self: Exec, g: Graph, logits: Buf) u32 {
        return self.vtable.argmax(g, self.ptr, logits);
    }
    /// See `VTable.argmaxConf`.
    pub fn argmaxConf(self: Exec, g: Graph, logits: Buf, p: *f32) u32 {
        return self.vtable.argmaxConf(g, self.ptr, logits, p);
    }
    /// `rows` argmaxes over `rows * vocab` logits, one launch and one
    /// read-back on a device backend.  See `VTable.argmaxRows`.
    pub fn argmaxRows(self: Exec, g: Graph, logits: Buf, rows: usize, out: []u32) void {
        self.vtable.argmaxRows(g, self.ptr, logits, rows, out);
    }
    pub fn sample(self: Exec, g: Graph, logits: Buf, scratch: Buf, sp: *sampler.Sampler) u32 {
        return self.vtable.sample(g, self.ptr, logits, scratch, sp);
    }
    pub fn profPhase(self: Exec, on: bool) void {
        self.vtable.profPhase(self.ptr, on);
    }
    pub fn profReset(self: Exec) void {
        self.vtable.profReset(self.ptr);
    }
    pub fn profPrefillEnd(self: Exec) void {
        self.vtable.profPrefillEnd(self.ptr);
    }
    /// Open / close one decode step.  See the vtable comments: this is a
    /// boundary hint the backend may ignore, and the CPU backend does.
    pub fn graphBegin(self: Exec, kind: GraphKind, ctl: GraphCtl) Graph {
        return self.vtable.graphBegin(self.ptr, kind, ctl);
    }
    pub fn graphEnd(self: Exec, g: *Graph) void {
        self.vtable.graphEnd(self.ptr, g);
    }
    pub fn setThreads(self: Exec, io: std.Io, nt: usize) void {
        self.vtable.setThreads(self.ptr, io, nt);
    }
    pub fn setName(self: Exec, tier: []const u8) void {
        self.vtable.setTier(self.ptr, tier);
    }
    pub fn name(self: Exec) []const u8 {
        return self.vtable.name(self.ptr);
    }
    pub fn describe(self: Exec) []const u8 {
        return self.vtable.describe(self.ptr);
    }
    pub fn reportProfile(self: Exec) void {
        self.vtable.reportProfile(self.ptr);
    }
    /// Print the complete profile report once.  Callers gate this on their own
    /// debug switch; the vtable slot itself is unconditional.
    pub fn profReport(self: Exec) void {
        self.vtable.profReport(self.ptr);
    }
};

/// Which backend a model run uses.
///
/// `.cuda` is wired up but currently a *trampoline*: `backend_cuda.CudaBackend`
/// embeds a CPU backend and hands out its `Exec`, so `--backend cuda` exercises
/// the selection path end-to-end with byte-identical numbers while the device
/// implementation does not exist yet.  See the header of `backend_cuda.zig`.
pub const Kind = enum { cpu, cuda };

pub fn selectBackend(kind: Kind) type {
    return switch (kind) {
        .cpu => @import("backend_cpu.zig").CpuBackend,
        .cuda => @import("backend_cuda.zig").CudaBackend,
    };
}
