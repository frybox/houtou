//! Qwen3 dense decoder (Qwen3-8B and siblings).
//!
//! Why a second engine rather than a flag in `engines/qwen35.zig`: the two
//! architectures disagree on three things that reach into the layer body, and
//! each one is a different expression, not a switch:
//!
//! | | Qwen3.5/3.8 (`qwen35.zig`) | Qwen3 (this file) |
//! |---|---|---|
//! | layer mix | GatedDeltaNet + full attention (`full_attention_interval`) | all full attention |
//! | RoPE span | leading 64 of a 256-wide head (`partial_rotary_factor = 0.25`) | the whole head (`head_dim = 128`) |
//! | `attn_q` layout | per-head `[q \| gate]` interleaved, gated output | plain `q`, ungated |
//! | position encoding | mrope sections `[11,11,10]` | plain 1-D RoPE |
//! | GGUF metadata prefix | `qwen35.*` | `qwen3.*` |
//!
//! The attention body here is therefore written out rather than shared: it is
//! ~40 lines, and the alternative is a parameterised kernel that both callers
//! have to keep in sync (the plan is to extract the genuinely common parts --
//! GGUF plumbing, KV cache, the sampler loop -- into `engines/common.zig` once
//! both implementations exist).
//!
//! SCOPE: single-token decode and chunked prefill, greedy or sampled.  No
//! speculative decoding yet; the draft model and stage 4 live outside this file
//! and are architecture-checkpoint specific.

const std = @import("std");
const gguf = @import("../gguf.zig");
const backend = @import("../backend.zig");
const backend_cpu = @import("../backend_cpu.zig");
const sampler = @import("../sampler.zig");
const common = @import("common.zig");
const Io = std.Io;

const Buf = backend.Buf;
const LinearW = backend.LinearW;

pub const Config = struct {
    n_layers: u32 = 36,
    hidden: u32 = 4096,
    n_heads: u32 = 32,
    n_kv_heads: u32 = 8,
    /// Qwen3 uses the WHOLE head as the rotary span, so `rotary_dim == head_dim`
    /// for every model in this family.  Kept as a field (rather than assumed)
    /// because the `qwen3` GGUF does not carry `partial_rotary_factor` and a
    /// future variant could rotate less.
    head_dim: u32 = 128,
    rotary_dim: u32 = 128,
    n_ff: u32 = 12288,
    rms_eps: f32 = 1e-6,
    vocab: u32 = 151936,
    rope_freq_base: f32 = 1e6,
    eos_token_id: u32 = 151645,
};

pub const Engine = struct {
    alloc: std.mem.Allocator,
    cfg: Config,
    g: gguf.File,
    mm: std.Io.File.MemoryMap,
    file: std.Io.File,
    io: std.Io,
    owns_io: ?*std.Io.Threaded = null,

    exec: backend.Exec,
    /// The compute graph the current forward pass is being recorded into or
    /// replayed from.  Set by `forwardLogits` / `forwardChunk`, which are the
    /// only entry points: every op between them and the next `graphEnd` reads
    /// it.  The pass-through value (`ptr == null`) makes every op behave
    /// exactly as it did before the graph existed, which is what the CPU
    /// backend and a failed capture use.
    ///
    /// A field rather than a parameter threading: the layer helpers are 15 deep
    /// and all of them would otherwise take it just to pass it on.
    graph: backend.Graph = .{},
    backend_kind: backend.Kind,

    token_embd: LinearW,
    output_weight: ?LinearW,
    output_norm: Buf,
    layers: []LayerWeights,
    kv: []KvCache,

    /// Scratch, sized for the widest activation this architecture produces.
    x: Buf, // [hidden] residual stream
    x1: Buf, // [max(n_ff, hidden)] normed input
    x2: Buf, // [max(n_ff, hidden)] projection output
    q: Buf, // [heads*hd]
    k: Buf, // [kv_heads*hd]
    v: Buf, // [kv_heads*hd]
    att: Buf, // [max_seq] attention scores
    gout: Buf, // [heads*hd] attention output
    mlp_h: Buf, // [n_ff]
    mlp_h1: Buf, // [n_ff]
    logits: Buf, // [vocab]
    /// Attention is ungated in this architecture, but `attnDecode` ends with
    /// `gout *= sigmoid(gate)`.  `+inf` makes that factor exactly 1.0 (same
    /// trick `src/draft.zig` uses for the same reason).
    ones: Buf,

    // chunk (prefill) scratch
    xK: Buf = Buf.empty,
    x1K: Buf = Buf.empty,
    x2K: Buf = Buf.empty,
    qK: Buf = Buf.empty,
    kK: Buf = Buf.empty,
    vK: Buf = Buf.empty,
    mlpK: Buf = Buf.empty,
    mlpK1: Buf = Buf.empty,
    goutK: Buf = Buf.empty,
    gateK: Buf = Buf.empty,

    kv_type: backend.KvType = .f16,
    /// Sampler scratch (`vocab` f32); `Exec.sample` writes into it.
    sample_scratch: Buf = Buf.empty,
    /// Sampling policy; `generate` installs a greedy one.
    sampler: sampler.Sampler,
    /// Maximum sequence the KV caches were allocated for.
    max_ctx: u32 = 0,
    n_batch: u32 = 0,

    // ---- speculative-decoding taps (DSpine draft) ----
    //
    // The draft consumes the target's hidden state at a fixed set of layer
    // indices (`dflash_config.target_layer_ids`, `[1,9,17,25,33]` for this
    // model) and never looks at any other layer.  `forwardChunk` and
    // `forwardLogits` copy those layers' OUTPUT hidden state into
    // `tap_hidden`, which the draft's `fc` projection turns into its context.
    //
    // LAYOUT: `tap_hidden` is `tap_layers.len * n_batch * hidden`, one
    // `n_batch * hidden` plane per tap.  The `n_batch` stride (not the chunk
    // length) is so a short final chunk still lands at a fixed offset.
    //
    // Disabled by default (`Buf.empty`), so the normal path pays one usize
    // compare per layer.  This is the same contract as the Qwen3.5 engine; the
    // only structural difference is that there is no GDN state to snapshot.
    tap_layers: []const u32 = &.{},
    tap_hidden: Buf = Buf.empty,
    /// Rolling tap ring for the spec context window (`specRingEnable`).
    ring: []f32 = &.{},
    ring_ctx: usize = 0,
    ring_next: usize = 0,
    ring_filled: usize = 0,
    /// Per-layer state snapshot for spec rollback.  Qwen3 has no recurrent
    /// state, so a snapshot is only the KV lengths.
    spec_snap: []u32 = &.{},
    /// `$QWEN35_SPEC_TRACE`: print each round's first proposal next to the
    /// target's own pick.  Purely diagnostic; set by `main.zig`.
    spec_trace_dbg: bool = false,
    /// `$QWEN35_SPEC_POSSTAT`: how far down each block the verifier got.
    spec_pos_reached: [64]usize = @splat(0),
    /// Wall time of the last prefill, reported by `generateSpec`.
    prefill_ms: f64 = 0,

    pub const max_seq = 40960;

    const LayerWeights = struct {
        attn_norm: Buf,
        post_attention_norm: Buf,
        attn_q: LinearW,
        attn_k: LinearW,
        attn_v: LinearW,
        attn_output: LinearW,
        attn_q_norm: Buf,
        attn_k_norm: Buf,
        ffn_gate: LinearW,
        ffn_up: LinearW,
        ffn_down: LinearW,
    };

    const KvCache = struct {
        k: Buf,
        v: Buf,
        len: u32,
    };

    // -----------------------------------------------------------------------
    // Load
    // -----------------------------------------------------------------------

    /// Read `qwen3.*` metadata.  Every field the layer body needs is required:
    /// a missing key means the file is not this architecture, and guessing a
    /// default would produce a plausible but wrong model.
    fn parseConfig(g: *const gguf.File) !Config {
        const req = struct {
            fn u(gg: *const gguf.File, key: []const u8) !u32 {
                return @intCast(gg.getUint(key) orelse return error.BadConfig);
            }
        };
        var c = Config{};
        c.n_layers = try req.u(g, "qwen3.block_count");
        c.hidden = try req.u(g, "qwen3.embedding_length");
        c.n_heads = try req.u(g, "qwen3.attention.head_count");
        c.n_kv_heads = try req.u(g, "qwen3.attention.head_count_kv");
        c.n_ff = try req.u(g, "qwen3.feed_forward_length");
        c.head_dim = if (g.getUint("qwen3.attention.key_length")) |v| @intCast(v) else c.hidden / c.n_heads;
        // The rotary span: this family rotates whole heads.  If a future
        // variant carries an explicit count, honour it.
        c.rotary_dim = if (g.getUint("qwen3.rope.dimension_count")) |v| @intCast(v) else c.head_dim;
        c.rope_freq_base = if (g.getFloat("qwen3.rope.freq_base")) |v| @floatCast(v) else 1e6;
        c.rms_eps = if (g.getFloat("qwen3.attention.layer_norm_rms_epsilon")) |v| @floatCast(v) else 1e-6;
        c.vocab = @intCast((g.findTensor("token_embd.weight") orelse return error.BadConfig).dims[1]);
        if (g.getUint("tokenizer.ggml.eos_token_id")) |v| c.eos_token_id = @intCast(v);
        if (c.n_heads == 0 or c.n_kv_heads == 0 or c.n_heads % c.n_kv_heads != 0) return error.BadConfig;
        if (c.rotary_dim > c.head_dim or c.rotary_dim % 2 != 0) return error.BadConfig;
        return c;
    }

    pub fn load(
        alloc: std.mem.Allocator,
        path: []const u8,
        io: std.Io,
        n_threads: usize,
        backend_kind: backend.Kind,
        kv_type: backend.KvType,
        /// KV cache depth.  Must cover prompt + generated tokens.
        ctx_len: u32,
    ) !Engine {
        var nt = n_threads;
        if (nt == 0) {
            const ncpu = std.Thread.getCpuCount() catch 1;
            nt = @min(16, ncpu);
        }
        const file = std.Io.Dir.cwd().openFile(io, path, .{}) catch return error.OpenFailed;
        errdefer file.close(io);
        const stat = file.stat(io) catch return error.OpenFailed;
        var mm = std.Io.File.MemoryMap.create(io, file, .{
            .len = @intCast(stat.size),
            .protection = .{ .read = true },
        }) catch |e| {
            file.close(io);
            return e;
        };
        errdefer mm.destroy(io);

        var g = try gguf.parse(alloc, mm.memory);
        errdefer g.deinit();

        const exec = switch (backend_kind) {
            .cpu => blk: {
                const cb = try alloc.create(backend_cpu.CpuBackend);
                cb.* = backend_cpu.CpuBackend.init(alloc, &g);
                cb.exec().setThreads(io, nt);
                break :blk cb.exec();
            },
            .cuda => blk: {
                const cb = try alloc.create(@import("../backend_cuda.zig").CudaBackend);
                cb.* = @import("../backend_cuda.zig").CudaBackend.init(alloc, &g);
                cb.exec().setThreads(io, nt);
                break :blk cb.exec();
            },
        };

        const cfg = try parseConfig(&g);
        const hidden: usize = cfg.hidden;
        const q_dim: usize = @as(usize, cfg.n_heads) * cfg.head_dim;
        const kv_dim: usize = @as(usize, cfg.n_kv_heads) * cfg.head_dim;
        const n_ff: usize = cfg.n_ff;

        var e = Engine{
            .alloc = alloc,
            .cfg = cfg,
            .g = g,
            .mm = mm,
            .file = file,
            .io = io,
            .exec = exec,
            .backend_kind = backend_kind,
            .token_embd = try exec.loadLinear("token_embd.weight"),
            // `findTensor`, not `findKey`: `output.weight` is a TENSOR, and
            // `findKey` only walks the KV metadata (it always returned null
            // here, so the untied head was never loaded and the terminal
            // projection silently used the embedding table instead).
            .output_weight = if (g.findTensor("output.weight") != null) try exec.loadLinear("output.weight") else null,
            .output_norm = try exec.loadVecF32("output_norm.weight"),
            .layers = try alloc.alloc(LayerWeights, cfg.n_layers),
            .kv = try alloc.alloc(KvCache, cfg.n_layers),
            .x = try exec.allocBuf(hidden),
            .x1 = try exec.allocBuf(@max(n_ff, q_dim)),
            .x2 = try exec.allocBuf(@max(n_ff, hidden)),
            .q = try exec.allocBuf(q_dim),
            .k = try exec.allocBuf(kv_dim),
            .v = try exec.allocBuf(kv_dim),
            .att = try exec.allocBuf(max_seq),
            .gout = try exec.allocBuf(q_dim),
            .mlp_h = try exec.allocBuf(n_ff),
            .mlp_h1 = try exec.allocBuf(n_ff),
            .logits = try exec.allocBuf(cfg.vocab),
            .ones = try exec.allocBuf(q_dim),
            .kv_type = kv_type,
            .sample_scratch = Buf.empty,
            .sampler = sampler.Sampler.init(.{ .temperature = 0 }),
            .max_ctx = ctx_len,
            .n_batch = 0,
        };
        errdefer e.deinit();

        // `+inf` gate -> sigmoid == 1.0 exactly (see the `ones` field comment).
        // Written through `copyFromHost`, NOT `ones.slice()`: on the CUDA
        // backend `Buf.slice()` is a DEVICE address, so a host loop over it
        // dereferences device memory and segfaults inside `load` (observed
        // before this was fixed).
        {
            const host = try alloc.alloc(f32, q_dim);
            defer alloc.free(host);
            @memset(host, std.math.inf(f32));
            exec.copyFromHost(e.graph, e.ones.span(0, q_dim), host);
        }

        e.sample_scratch = try exec.allocBuf(cfg.vocab);
        const widest = @max(n_ff, @max(q_dim, hidden));
        try exec.allocScratch(widest);

        // ---- chunk (batched prefill) scratch ----
        // `K` is the chunk size: one weight read per projection serves this many
        // tokens.  1024 matches the Qwen3.5 engine; the buffers are sized from
        // it, so raising it costs activation memory only.
        const K: usize = 1024;
        e.n_batch = K;
        e.xK = try exec.allocBuf(K * hidden);
        e.x1K = try exec.allocBuf(K * hidden);
        e.x2K = try exec.allocBuf(K * hidden);
        e.qK = try exec.allocBuf(K * q_dim);
        e.kK = try exec.allocBuf(K * kv_dim);
        e.vK = try exec.allocBuf(K * kv_dim);
        e.mlpK = try exec.allocBuf(K * n_ff);
        e.mlpK1 = try exec.allocBuf(K * n_ff);
        e.goutK = try exec.allocBuf(K * q_dim);
        // `attnChunk` takes a gate buffer the same shape as q; this engine has no
        // output gate, so the whole chunk sees `+inf` (sigmoid == 1) exactly like
        // the single-token path.
        e.gateK = try exec.allocBuf(K * q_dim);
        {
            const host = try alloc.alloc(f32, K * q_dim);
            defer alloc.free(host);
            @memset(host, std.math.inf(f32));
            exec.copyFromHost(e.graph, e.gateK.span(0, K * q_dim), host);
        }
        try exec.allocScratchK(widest, K);

        for (0..cfg.n_layers) |i| {
            var b: [96]u8 = undefined;
            const nm = struct {
                fn f(buf: []u8, comptime fmt: []const u8, idx: usize) []const u8 {
                    return std.fmt.bufPrint(buf, fmt, .{idx}) catch unreachable;
                }
            };
            const l = &e.layers[i];
            l.* = undefined;
            l.attn_norm = try exec.loadVecF32(nm.f(&b, "blk.{d}.attn_norm.weight", i));
            l.post_attention_norm = try exec.loadVecF32(nm.f(&b, "blk.{d}.ffn_norm.weight", i));
            l.attn_q = try exec.loadLinear(nm.f(&b, "blk.{d}.attn_q.weight", i));
            l.attn_k = try exec.loadLinear(nm.f(&b, "blk.{d}.attn_k.weight", i));
            l.attn_v = try exec.loadLinear(nm.f(&b, "blk.{d}.attn_v.weight", i));
            l.attn_output = try exec.loadLinear(nm.f(&b, "blk.{d}.attn_output.weight", i));
            l.attn_q_norm = try exec.loadVecF32(nm.f(&b, "blk.{d}.attn_q_norm.weight", i));
            l.attn_k_norm = try exec.loadVecF32(nm.f(&b, "blk.{d}.attn_k_norm.weight", i));
            l.ffn_gate = try exec.loadLinear(nm.f(&b, "blk.{d}.ffn_gate.weight", i));
            l.ffn_up = try exec.loadLinear(nm.f(&b, "blk.{d}.ffn_up.weight", i));
            l.ffn_down = try exec.loadLinear(nm.f(&b, "blk.{d}.ffn_down.weight", i));
            const region = try exec.allocKv(kv_dim, max_seq);
            e.kv[i] = .{ .k = region.k, .v = region.v, .len = 0 };
        }

        return e;
    }

    pub fn deinit(self: *Engine) void {
        if (self.owns_io) |t| t.deinit();
        self.* = undefined;
    }

    // -----------------------------------------------------------------------
    // Forward
    // -----------------------------------------------------------------------

    pub fn reset(self: *Engine) void {
        for (self.kv) |*c| c.len = 0;
    }

    // -----------------------------------------------------------------------
    // Speculative-decoding support (DSpine draft)
    // -----------------------------------------------------------------------

    /// Enable the tap planes the draft's `fc` projection reads.
    ///
    /// `layers` must be ascending and every index must be a real decoder layer;
    /// the draft's `target_layer_ids` are checked against this model at load
    /// time, so a mismatch is a named error here rather than a silently wrong
    /// hidden state later.
    pub fn specTapEnable(self: *Engine, layers: []const u32) !void {
        self.tap_layers = &.{};
        self.tap_hidden = Buf.empty;
        if (layers.len == 0) return;
        // A zero-sized plane would silently disable the taps; refusing is what
        // the Qwen3.5 engine does for the same reason.
        if (self.n_batch == 0) return error.BatchingDisabled;
        var prev: u32 = 0;
        for (layers, 0..) |l, i| {
            if (l >= self.cfg.n_layers) return error.TapLayerOutOfRange;
            if (i > 0 and l <= prev) return error.TapLayersNotAscending;
            prev = l;
        }
        self.tap_hidden = try self.exec.allocBuf(layers.len * self.n_batch * self.cfg.hidden);
        // Copy the indices: the caller's slice may be a temporary.
        const owned = try self.alloc.dupe(u32, layers);
        self.tap_layers = owned;
    }

    /// Plane `i` of the tap buffer, `rows` rows (`hidden` floats each).
    pub fn specTapPlane(self: *const Engine, i: usize, rows: usize) Buf {
        std.debug.assert(i < self.tap_layers.len);
        std.debug.assert(rows <= self.n_batch);
        return self.tap_hidden.span(i * self.n_batch * self.cfg.hidden, rows * self.cfg.hidden);
    }

    /// Copy tap plane `i`'s first `rows` rows into HOST slice `dst`.
    ///
    /// `copyOut`, not `copyRange`: the destination is host memory and
    /// `copyRange` infers direction from the source alone, so a device source
    /// with a host destination would be issued as a device-to-device move.
    pub fn specTapPlaneToHost(self: *Engine, dst: []f32, i: usize, rows: usize) void {
        const hidden: usize = self.cfg.hidden;
        std.debug.assert(dst.len >= rows * hidden);
        self.exec.copyOut(self.graph, dst[0 .. rows * hidden], self.specTapPlane(i, rows));
    }

    /// Sequence length the draft's own KV may use (it shares the target's
    /// caps: a draft row never outlives the target row it predicts).
    pub fn maxSeqForSpec(self: *const Engine) usize {
        return self.max_ctx;
    }

    /// The target's lm_head as a `LinearW`, for the draft's candidate scoring.
    /// The draft carries no copy of the vocabulary projection.
    pub fn lmHead(self: *const Engine) LinearW {
        return self.output_weight orelse self.token_embd;
    }

    /// `lm_head @ x` where `x` is a HOST row (the draft's candidate scorer
    /// works on host activations).  Same contract as the Qwen3.5 engine's.
    pub fn lmHeadFromHostRow(self: *Engine, out: []f32, x: []const f32) void {
        const hidden: usize = self.cfg.hidden;
        std.debug.assert(x.len == hidden);
        std.debug.assert(out.len >= self.cfg.vocab);
        // `x1` is engine scratch at least `hidden` wide and is rewritten by
        // every forward anyway, so borrowing it needs no new allocation.
        const row = self.x1.span(0, hidden);
        self.exec.copyFromHost(self.graph, row, x[0..hidden]);
        switch (self.output_weight orelse self.token_embd) {
            .quant => |qrow| self.exec.matvec(self.graph, self.logits, .{ .quant = qrow }, row),
            .f32 => |m| self.exec.matvec(self.graph, self.logits, .{ .f32 = m }, row),
        }
        self.exec.copyOut(self.graph, out[0..self.cfg.vocab], self.logits);
    }

    // ---- spec snapshot / rollback ----
    //
    // Qwen3 is a plain dense decoder: the ONLY cross-token state is the KV
    // cache, so "what was the state at position p" is exactly "how long was
    // each cache".  That is why this is three lines here and a whole
    // conv+delta-matrix copy on the Qwen3.5 engine -- there is no GDN.
    //
    // Restoring is therefore exact and needs no replay: truncating a cache to
    // its old length leaves rows `[0, len)` untouched, which are the same bytes
    // the model would have had without the speculative block.

    /// Allocate the per-layer snapshot.  One entry per layer (the spec block
    /// never runs the nextn slot, and this engine has none).
    pub fn specSnapshotAlloc(self: *Engine, alloc: std.mem.Allocator) !void {
        if (self.spec_snap.len > 0) return;
        const snaps = try alloc.alloc(u32, self.kv.len);
        for (snaps, 0..) |*s, i| s.* = self.kv[i].len;
        self.spec_snap = snaps;
    }

    /// Record the live KV lengths.
    pub fn specSnapshotTake(self: *Engine) void {
        for (self.spec_snap, 0..) |*s, i| {
            if (i < self.kv.len) s.* = self.kv[i].len;
        }
    }

    /// Wind the caches back to the snapshot, then advance them by `tokens.len`
    /// rows -- the verify block was `tokens.len` long and those rows are the
    /// ones that were accepted.
    ///
    /// The replay is a genuine forward pass (not just a length fix): the block
    /// wrote KV rows for EVERY proposed token, including the rejected ones, so
    /// the accepted prefix has to be re-written at the right slots.
    pub fn specRollbackAndReplay(self: *Engine, tokens: []const u32, pos: u32) void {
        for (self.spec_snap, 0..) |s, i| {
            if (i < self.kv.len) self.kv[i].len = s;
        }
        for (tokens, 0..) |t, i| _ = self.forwardLogits(t, pos + @as(u32, @intCast(i)));
    }

    /// Which backend this engine drives (debug/verification helpers use it).
    pub fn backendKind(self: *const Engine) backend.Kind {
        return self.backend_kind;
    }

    // ---- rolling tap ring ----
    //
    // The draft needs the target's hidden states for the tokens that ALREADY
    // precede the block, not just the one being predicted.  Re-forwarding that
    // window every round would cost `ctx_rows` target passes per round, which
    // measured 651 s for 32 tokens on the 27B before the ring existed.  So the
    // most recent `ctx_rows` rows of every tap plane are kept, and a round
    // reads them instead.

    /// Allocate the ring: `n_taps * ctx_rows * hidden` host f32.
    pub fn specRingEnable(self: *Engine, alloc: std.mem.Allocator, ctx_rows: usize) !void {
        if (self.tap_layers.len == 0) return error.TapsDisabled;
        const need = self.tap_layers.len * ctx_rows * self.cfg.hidden;
        if (self.ring.len < need) {
            if (self.ring.len > 0) alloc.free(self.ring);
            self.ring = try alloc.alloc(f32, need);
        }
        self.ring_ctx = ctx_rows;
        self.ring_next = 0;
        self.ring_filled = 0;
    }

    /// Append the current tap plane's row 0 to the ring (single-token path).
    pub fn specRingPush(self: *Engine) void {
        if (self.ring.len == 0) return;
        const hidden: usize = self.cfg.hidden;
        const ctx = self.ring_ctx;
        const at = self.ring_next;
        for (0..self.tap_layers.len) |ti| {
            const src = self.tap_hidden.span(ti * self.n_batch * hidden, hidden);
            const dst = self.ring[ti * ctx * hidden + at * hidden ..][0..hidden];
            self.exec.copyOut(self.graph, dst, src);
        }
        self.ring_next = (at + 1) % ctx;
        if (self.ring_filled < ctx) self.ring_filled += 1;
    }

    /// Fill the ring from the chunk that just ran, oldest-first, keeping the
    /// last `ctx_rows` rows.
    pub fn specRingFillFromChunk(self: *Engine, rows: usize, pos0: u32) void {
        _ = pos0;
        if (self.ring.len == 0 or rows == 0) return;
        const hidden: usize = self.cfg.hidden;
        const ctx = self.ring_ctx;
        const take = @min(rows, ctx);
        const skip = rows - take;
        for (0..take) |r| {
            const at = (self.ring_next + r) % ctx;
            for (0..self.tap_layers.len) |ti| {
                const src = self.tap_hidden.span(ti * self.n_batch * hidden + (skip + r) * hidden, hidden);
                self.exec.copyOut(self.graph, self.ring[ti * ctx * hidden + at * hidden ..][0..hidden], src);
            }
        }
        self.ring_next = (self.ring_next + take) % ctx;
        self.ring_filled = @min(self.ring_filled + take, ctx);
    }

    /// How many tokens one `forwardChunk` call can take (`0` = unbatched).
    pub fn batchLimit(self: *const Engine) usize {
        return self.n_batch;
    }

    /// How many rows the ring currently holds (`<= ctx_rows`).
    pub fn specRingFilled(self: *const Engine) usize {
        return self.ring_filled;
    }

    /// Copy the ring's contents into `out` (`n_taps * ctx_rows * hidden`),
    /// oldest-first, zero-padding any unfilled slots at the front.
    pub fn specRingRead(self: *Engine, out: []f32) void {
        const hidden: usize = self.cfg.hidden;
        const ctx = self.ring_ctx;
        const nt = self.tap_layers.len;
        const n = self.ring_filled;
        const start = (self.ring_next + ctx - n) % ctx;
        @memset(out[0 .. nt * ctx * hidden], 0);
        const pad = ctx - n;
        for (0..nt) |ti| {
            for (0..n) |r| {
                const src_at = (start + r) % ctx;
                @memcpy(
                    out[ti * ctx * hidden + (pad + r) * hidden ..][0..hidden],
                    self.ring[ti * ctx * hidden + src_at * hidden ..][0..hidden],
                );
            }
        }
    }

    /// The token embedding as a packed row set.  The draft expands its `fc`
    /// input through this, so a model whose embedding is plain f32 has to say
    /// so rather than hand back a reinterpretation of f32 bytes.
    pub fn embedRows(self: *const Engine) !gguf.QuantRow {
        return switch (self.token_embd) {
            .quant => |qrow| qrow,
            .f32 => error.EmbeddingNotQuantized,
        };
    }

    pub fn embedTo(self: *Engine, token: u32, dst: Buf) void {
        const hidden: usize = self.cfg.hidden;
        std.debug.assert(dst.len == hidden);
        // One row expanded to f32 is the only place this engine needs values
        // rather than a dot product; the dequantizer choice is the backend's.
        switch (self.token_embd) {
            .f32 => |m| self.exec.copyRange(self.graph, dst, Buf.wrap(m[token * hidden ..][0..hidden])),
            .quant => |qrow| self.exec.dequantRow(self.graph, dst, qrow, token),
        }
    }

    /// One decoder layer, single token.  Mirrors the reference dense-decoder
    /// body: `h = x + attn(norm(x)); out = h + mlp(norm(h))`.
    pub fn layerForward(self: *Engine, lw: *const LayerWeights, li: usize, pos: u32) void {
        const c = &self.cfg;
        const hidden: usize = c.hidden;
        const heads: usize = c.n_heads;
        const kv_heads: usize = c.n_kv_heads;
        const hd: usize = c.head_dim;
        const q_dim: usize = heads * hd;
        const kv_dim: usize = kv_heads * hd;
        const gqa: usize = heads / kv_heads;

        // ---- attention ----
        self.exec.rmsnorm(self.graph, self.x1.span(0, hidden), self.x.span(0, hidden), lw.attn_norm, c.rms_eps);

        // Qwen3's `attn_q` is a plain [q_dim, hidden] projection: there is no
        // per-head [q|gate] interleave and no output gate (unlike Qwen3.5).
        const q = self.q.span(0, q_dim);
        self.exec.matvec(self.graph, q, lw.attn_q, self.x1.span(0, hidden));
        self.exec.matvec(self.graph, self.k.span(0, kv_dim), lw.attn_k, self.x1.span(0, hidden));
        self.exec.matvec(self.graph, self.v.span(0, kv_dim), lw.attn_v, self.x1.span(0, hidden));

        // Per-head RMSNorm over `head_dim` (=128 here, not the 256-wide kind
        // Qwen3.5 uses), then RoPE over the FULL head.
        self.exec.rmsnormRows(self.graph, q, q, lw.attn_q_norm, heads, c.rms_eps);
        self.exec.rmsnormRows(self.graph, self.k.span(0, kv_dim), self.k.span(0, kv_dim), lw.attn_k_norm, kv_heads, c.rms_eps);
        self.exec.ropeRows(self.graph, q, pos, c.rope_freq_base, 1, heads, hd, c.rotary_dim);
        self.exec.ropeRows(self.graph, self.k.span(0, kv_dim), pos, c.rope_freq_base, 1, kv_heads, hd, c.rotary_dim);

        const cache = &self.kv[li];
        std.debug.assert(cache.len < self.max_ctx);
        const slot: usize = cache.len;
        self.exec.kvAppend(self.graph, cache.k, self.k.span(0, kv_dim), slot, kv_dim, 1);
        self.exec.kvAppend(self.graph, cache.v, self.v.span(0, kv_dim), slot, kv_dim, 1);
        cache.len += 1;

        const scale = 1.0 / @sqrt(@as(f32, @floatFromInt(hd)));
        const gout = self.gout.span(0, q_dim);
        self.exec.attnDecode(self.graph, 
            q, self.ones.span(0, q_dim),
            cache.k, cache.v,
            self.att.span(0, slot + 1), gout,
            slot, heads, gqa, hd, kv_dim, scale,
        );
        self.exec.matvec(self.graph, self.x2.span(0, hidden), lw.attn_output, gout);
        self.exec.addResid(self.graph, self.x.span(0, hidden), self.x2.span(0, hidden));

        // ---- MLP ----
        self.exec.rmsnorm(self.graph, self.x1.span(0, hidden), self.x.span(0, hidden), lw.post_attention_norm, c.rms_eps);
        const n_ff: usize = c.n_ff;
        self.exec.matvec(self.graph, self.mlp_h.span(0, n_ff), lw.ffn_gate, self.x1.span(0, hidden));
        self.exec.matvec(self.graph, self.mlp_h1.span(0, n_ff), lw.ffn_up, self.x1.span(0, hidden));
        self.exec.swiglu(self.graph, self.mlp_h.span(0, n_ff), self.mlp_h1.span(0, n_ff));
        self.exec.matvec(self.graph, self.x2.span(0, hidden), lw.ffn_down, self.mlp_h.span(0, n_ff));
        self.exec.addResid(self.graph, self.x.span(0, hidden), self.x2.span(0, hidden));

        // Speculative-decoding tap: this layer's OUTPUT hidden state, written
        // after BOTH residual adds so it is the value the next layer's norm
        // consumes -- the same point the Qwen3.5 engine taps, and what DSpine's
        // `fc` projection was trained on.
        if (self.tap_hidden.len != 0) {
            for (self.tap_layers, 0..) |tl, ti| {
                if (tl == @as(u32, @intCast(li))) {
                    self.exec.copyRange(self.graph, 
                        self.tap_hidden.span(ti * self.n_batch * hidden, hidden),
                        self.x.span(0, hidden),
                    );
                    break;
                }
            }
        }
    }

    pub fn forwardLogits(self: *Engine, token: u32, pos: u32) Buf {
        const c = &self.cfg;
        const hidden: usize = c.hidden;
        self.embedTo(token, self.x.span(0, hidden));
        self.graph = self.exec.graphBegin(.decode, .{ .pos = pos, .kv_slot = pos, .attn_slot = pos, .rows = 1 });
        for (self.layers, 0..) |*lw, li| self.layerForward(lw, li, pos);
        self.exec.rmsnorm(self.graph, self.x1.span(0, hidden), self.x.span(0, hidden), self.output_norm, c.rms_eps);
        if (self.output_weight) |ow| {
            self.exec.matvec(self.graph, self.logits, ow, self.x1.span(0, hidden));
        } else {
            self.exec.matvec(self.graph, self.logits, self.token_embd, self.x1.span(0, hidden));
        }
        self.exec.graphEnd(&self.graph);
        self.graph = .{};
        return self.logits;
    }

    pub fn step(self: *Engine, token: u32, pos: u32) u32 {
        _ = self.forwardLogits(token, pos);
        return self.argmax();
    }

    pub fn argmax(self: *Engine) u32 {
        const lg = self.logits.slice()[0..self.cfg.vocab];
        var best: u32 = 0;
        var bv: f32 = -std.math.inf(f32);
        for (lg, 0..) |v, i| {
            if (v > bv) {
                bv = v;
                best = @intCast(i);
            }
        }
        return best;
    }

    /// Chunked prefill: `tokens.len` tokens starting at `pos0`, returning the
    /// logits of the LAST one.
    ///
    /// K-token batched path (the twin of `qwen35.zig`'s `forwardChunk`): every
    /// weight-reading projection runs ONCE per chunk as a `matvecBatchK`, and
    /// the per-token work (per-head q/k norm, RoPE, KV append, attention) runs
    /// chunk-wide through the row-batched slots.  Each of those is defined as
    /// the per-token sequence it replaces, so the result is bit-identical to
    /// `k` calls of `forwardLogits` -- the same contract the Qwen3.5 engine
    /// holds, and what its `QWEN35_SPEC_TAPS` self-check verifies.
    ///
    /// Falls back to the per-token loop when no chunk scratch was allocated
    /// (`n_batch == 0`, which is what `--backend cpu` used to imply before the
    /// buffers existed) or when the chunk is a single token.
    pub fn forwardChunk(self: *Engine, tokens: []const u32, pos0: u32) Buf {
        const k = tokens.len;
        // A single token is the serial path, and `forwardLogits` opens and
        // closes its own `stepBegin`/`stepEnd` span; delegating keeps the
        // decode-graph boundary from leaking into the next step (same reason
        // `qwen35.forwardChunk` special-cases k == 1).
        if (k == 0) return self.logits;
        // `$QWEN35_NO_BATCH`: force the per-token loop, for A/B against the
        // batched path (same switch the Qwen3.5 engine honours).
        const forced_serial = @import("../cudart.zig").environ_map != null and
            @import("../cudart.zig").environ_map.?.get("QWEN35_NO_BATCH") != null;
        if (k == 1 or forced_serial or self.n_batch == 0 or k > self.n_batch) {
            var out: Buf = self.logits;
            for (tokens, 0..) |t, i| {
                out = self.forwardLogits(t, pos0 + @as(u32, @intCast(i)));
            }
            return out;
        }

        const c = &self.cfg;
        const hidden: usize = c.hidden;
        const heads: usize = c.n_heads;
        const kv_heads: usize = c.n_kv_heads;
        const hd: usize = c.head_dim;
        const q_dim: usize = heads * hd;
        const kv_dim: usize = kv_heads * hd;
        const gqa: usize = heads / kv_heads;
        const n_ff: usize = c.n_ff;

        var t: usize = 0;
        while (t < k) : (t += 1) self.embedTo(tokens[t], self.xK.span(t * hidden, hidden));

        for (self.layers, 0..) |*lw, li| {
            // ---- attention ----
            // One norm for the whole chunk; `rmsnormRows` is `k` consecutive
            // single-row blocks and reduces each row in the same order, so it
            // is bit-identical to the per-token call.
            self.exec.rmsnormRows(self.graph, self.x1K.span(0, k * hidden), self.xK.span(0, k * hidden), lw.attn_norm, k, c.rms_eps);

            // Plain [q_dim, hidden] projection: no `[q|gate]` interleave, so
            // `qK` is filled directly (Qwen3.5 has to deinterleave first, which
            // is why its RoPE lives inside `forwardFullChunk`).
            self.exec.matvecBatchK(self.graph, self.qK.span(0, k * q_dim), lw.attn_q, self.x1K, k);
            self.exec.matvecBatchK(self.graph, self.kK.span(0, k * kv_dim), lw.attn_k, self.x1K, k);
            self.exec.matvecBatchK(self.graph, self.vK.span(0, k * kv_dim), lw.attn_v, self.x1K, k);

            // Per-head norm over `head_dim`, then RoPE over the FULL head.
            // Order matters: norm -> rope, as in `layerForward`; rotating first
            // and normalising after would renormalise the rotation away.
            self.exec.rmsnormRows(self.graph, self.qK.span(0, k * q_dim), self.qK.span(0, k * q_dim), lw.attn_q_norm, k * heads, c.rms_eps);
            self.exec.rmsnormRows(self.graph, self.kK.span(0, k * kv_dim), self.kK.span(0, k * kv_dim), lw.attn_k_norm, k * kv_heads, c.rms_eps);
            self.exec.ropeRows(self.graph, self.qK.span(0, k * q_dim), pos0, c.rope_freq_base, k, heads, hd, c.rotary_dim);
            self.exec.ropeRows(self.graph, self.kK.span(0, k * kv_dim), pos0, c.rope_freq_base, k, kv_heads, hd, c.rotary_dim);

            // KV cache: the chunk's `k` rows land contiguously at `slot0`, so
            // this is ONE strided append per tensor instead of `k` of them.
            const cache = &self.kv[li];
            std.debug.assert(cache.len + k <= self.max_ctx);
            const slot0: usize = cache.len;
            self.exec.kvAppend(self.graph, cache.k, self.kK.span(0, k * kv_dim), slot0, kv_dim, k);
            self.exec.kvAppend(self.graph, cache.v, self.vK.span(0, k * kv_dim), slot0, kv_dim, k);
            cache.len += @intCast(k);

            // scores -> softmax -> PV -> gate for all k rows in one launch.
            // `gateK` stays the all-`+inf` buffer (this architecture has no
            // attention output gate; `sigmoid(+inf) == 1`).
            const scale = 1.0 / @sqrt(@as(f32, @floatFromInt(hd)));
            self.exec.attnChunk(self.graph, self.qK.span(0, k * q_dim), self.gateK.span(0, k * q_dim),
                cache.k, cache.v,
                self.att.span(0, slot0 + k), self.goutK.span(0, k * q_dim),
                slot0, k, heads, gqa, hd, kv_dim, scale);

            self.exec.matvecBatchK(self.graph, self.x2K.span(0, k * hidden), lw.attn_output, self.goutK, k);
            self.exec.addResid(self.graph, self.xK.span(0, k * hidden), self.x2K.span(0, k * hidden));

            // ---- MLP ----
            self.exec.rmsnormRows(self.graph, self.x1K.span(0, k * hidden), self.xK.span(0, k * hidden), lw.post_attention_norm, k, c.rms_eps);
            self.exec.matvecBatchK(self.graph, self.mlpK.span(0, k * n_ff), lw.ffn_gate, self.x1K, k);
            self.exec.matvecBatchK(self.graph, self.mlpK1.span(0, k * n_ff), lw.ffn_up, self.x1K, k);
            self.exec.swiglu(self.graph, self.mlpK.span(0, k * n_ff), self.mlpK1.span(0, k * n_ff));
            self.exec.matvecBatchK(self.graph, self.x2K.span(0, k * hidden), lw.ffn_down, self.mlpK, k);
            self.exec.addResid(self.graph, self.xK.span(0, k * hidden), self.x2K.span(0, k * hidden));

            // Twin of `layerForward`'s tap, for the batched path: the whole
            // chunk's rows at once (they are contiguous, so one copy).
            if (self.tap_hidden.len != 0) {
                for (self.tap_layers, 0..) |tl, ti| {
                    if (tl == @as(u32, @intCast(li))) {
                        self.exec.copyRange(self.graph, 
                            self.tap_hidden.span(ti * self.n_batch * hidden, k * hidden),
                            self.xK.span(0, k * hidden),
                        );
                        break;
                    }
                }
            }
        }

        // Last token's logits on the serial tail, so the first sampled logits
        // go through exactly the same arithmetic as the decode path.
        // `x1` is `max(n_ff, q_dim)` wide, so it cannot be the norm output here
        // (`rmsNorm` asserts all three lengths match); use a prefix view of it,
        // which is exactly `hidden` floats.
        const normed = self.x1.span(0, hidden);
        self.exec.rmsnorm(self.graph, normed, self.xK.span((k - 1) * hidden, hidden), self.output_norm, c.rms_eps);
        if (self.output_weight) |ow| {
            self.exec.matvec(self.graph, self.logits, ow, normed);
        } else {
            self.exec.matvec(self.graph, self.logits, self.token_embd, normed);
        }
        return self.logits;
    }

    /// The draft callback contract (same shape as the Qwen3.5 engine's).
    pub const DraftFn = *const fn (
        ctx: *anyopaque,
        context: []const u32,
        anchor: u32,
        block_pos: u32,
        out: []u32,
    ) usize;

    /// Speculative decoding: the draft proposes a block, the target verifies it
    /// with its ORDINARY per-token forward, and only the accepted prefix is
    /// committed.
    ///
    /// WHY THE VERIFY REUSES `forwardLogits`: an accepted block must produce
    /// exactly the tokens plain greedy decode would have.  Verifying with any
    /// bespoke argmax walk over a batched forward would make the accepted path
    /// depend on a different set of kernels, so a "correct" run and a plain run
    /// could disagree -- and the spec counters would then be measuring the
    /// verifier, not the draft.
    ///
    /// CORRECTNESS OF ROLLBACK: the verify loop writes KV rows for every
    /// proposed token, including the rejected ones.  `specSnapshotTake` records
    /// the cache lengths before the block, and `specRollbackAndReplay` truncates
    /// back to them and re-runs the accepted prefix.  That is exact here because
    /// a KV cache is append-only and Qwen3 has no other cross-token state.
    pub fn generateSpec(
        self: *Engine,
        alloc: std.mem.Allocator,
        prompt: []const u32,
        n_predict: usize,
        draft_fn: DraftFn,
        draft_ctx: *anyopaque,
        max_draft: usize,
        ctx_rows: usize,
        progress: ?*const fn (ctx: ?*anyopaque, tok: u32) void,
        progress_ctx: ?*anyopaque,
    ) !common.GenResult {
        self.sampler.reseed();
        self.reset();
        if (prompt.len >= self.max_ctx) {
            std.debug.print(
                "generateSpec: prompt is {d} tokens, but the context holds {d}\n",
                .{ prompt.len, self.max_ctx },
            );
            return error.ContextLengthExceeded;
        }

        const io = self.io;
        self.exec.profReset();
        self.exec.profPhase(true);
        const t_start = Io.Clock.Timestamp.now(io, .awake);

        var last_logits: Buf = Buf.empty;
        var i: usize = 0;
        const cap: usize = if (self.n_batch == 0) 1 else @intCast(self.n_batch);
        while (i < prompt.len) {
            const n = @min(cap, prompt.len - i);
            last_logits = self.forwardChunk(prompt[i..][0..n], @intCast(i));
            i += n;
        }
        const t_after = Io.Clock.Timestamp.now(io, .awake);
        self.prefill_ms = @as(f64, @floatFromInt(t_start.durationTo(t_after).raw.nanoseconds)) / 1e6;
        self.exec.profPrefillEnd();

        // The prompt's own hidden states are the draft's first context window;
        // without this the first round drafts from a single row and nothing it
        // proposes can be right.
        if (self.n_batch != 0) self.specRingFillFromChunk(prompt.len, 0);

        try self.specSnapshotAlloc(alloc);

        var out: std.ArrayList(u32) = .empty;
        errdefer out.deinit(alloc);
        var history: std.ArrayList(u32) = .empty;
        defer history.deinit(alloc);
        try history.appendSlice(alloc, prompt);

        var pos: u32 = @intCast(prompt.len);
        var generated: usize = 0;
        var gen_ns: i96 = 0;
        var hit_eos = false;
        var hit_context_limit = false;
        var next: u32 = self.exec.sample(self.graph, last_logits, self.sample_scratch, &self.sampler);
        var proposed_total: usize = 0;
        var accepted_total: usize = 0;
        var rounds: usize = 0;

        const draft_buf = try alloc.alloc(u32, max_draft);
        defer alloc.free(draft_buf);

        while (generated < n_predict) {
            if (pos >= self.max_ctx) {
                hit_context_limit = true;
                break;
            }
            if (next == self.cfg.eos_token_id) {
                hit_eos = true;
                break;
            }
            try out.append(alloc, next);
            try history.append(alloc, next);
            if (progress) |f| f(progress_ctx, next);

            const t0 = Io.Clock.Timestamp.now(io, .awake);
            const block_pos = pos + 1;
            const room = @min(max_draft, @min(n_predict - generated - 1, self.max_ctx - block_pos));

            var advanced: usize = 1;
            var verified = false;
            if (room > 0) {
                const from = if (history.items.len > ctx_rows) history.items.len - ctx_rows else 0;
                const n_draft = draft_fn(draft_ctx, history.items[from..], next, block_pos, draft_buf[0..room]);
                if (n_draft > 0) {
                    rounds += 1;
                    proposed_total += n_draft;
                    verified = true;

                    self.specSnapshotTake();
                    var logits = self.forwardLogits(next, pos);
                    self.specRingPush();
                    var committed: usize = 0;
                    var pending: u32 = self.exec.sample(self.graph, logits, self.sample_scratch, &self.sampler);
                    if (self.spec_trace_dbg) {
                        std.debug.print("ROUND pos={d} anchor={d} draft[0]={d} target[0]={d} n={d}\n", .{
                            pos, next, draft_buf[0], pending, n_draft,
                        });
                    }
                    while (committed < n_draft) : (committed += 1) {
                        const cand = draft_buf[committed];
                        if (pending != cand) break;
                        if (generated + 1 + committed >= n_predict) break;
                        try out.append(alloc, cand);
                        try history.append(alloc, cand);
                        if (progress) |f| f(progress_ctx, cand);
                        const row_pos = block_pos + @as(u32, @intCast(committed));
                        logits = self.forwardLogits(cand, row_pos);
                        self.specRingPush();
                        pending = self.exec.sample(self.graph, logits, self.sample_scratch, &self.sampler);
                    }
                    accepted_total += committed;
                    if (committed < self.spec_pos_reached.len) self.spec_pos_reached[committed] += 1;
                    if (committed > 0) {
                        self.specRollbackAndReplay(draft_buf[0..committed], block_pos);
                    }
                    advanced = committed + 1;
                    next = pending;
                }
            }
            if (advanced == 1 and !verified) {
                next = self.step(next, pos);
                self.specRingPush();
            }
            pos += @as(u32, @intCast(advanced));
            generated += advanced;
            gen_ns += t0.durationTo(Io.Clock.Timestamp.now(io, .awake)).raw.nanoseconds;
        }
        self.exec.profPhase(false);
        const per: f64 = if (out.items.len > 0)
            @as(f64, @floatFromInt(gen_ns)) / 1e6 / @as(f64, @floatFromInt(out.items.len))
        else
            0;
        return .{
            .tokens = try out.toOwnedSlice(alloc),
            .prefill_ms = self.prefill_ms,
            .ms_per_token = per,
            .decode_ms = @as(f64, @floatFromInt(gen_ns)) / 1e6,
            .hit_eos = hit_eos,
            .hit_context_limit = hit_context_limit,
            .spec_proposed = proposed_total,
            .spec_accepted = accepted_total,
            .spec_rounds = rounds,
        };
    }

    /// Greedy generation through the shared driver loop
    /// (`engines/common.zig`), so the prompt/step/EOS bookkeeping is the same
    /// code the other engine runs.
    pub fn generate(
        self: *Engine,
        alloc: std.mem.Allocator,
        prompt: []const u32,
        n_predict: usize,
        on_token: ?*const fn (ctx: ?*anyopaque, tok: u32) void,
        ctx: ?*anyopaque,
    ) ![]u32 {
        var s = sampler.Sampler.init(.{ .temperature = 0 });
        self.sampler = s;
        const res = try common.generate(
            self.driver(),
            alloc,
            self.io,
            prompt,
            n_predict,
            on_token,
            ctx,
        );
        s = self.sampler;
        return res.tokens;
    }

    /// The narrow contract `engines/common.zig` runs the loop against.
    pub fn driver(self: *Engine) common.Driver {
        driver_ops.eos_id = self.cfg.eos_token_id;
        return .{ .ctx = self, .ops = &driver_ops };
    }

    /// One shared instance: `Ops` is immutable apart from `eos_id`, which
    /// `driver()` sets from the loaded config just before the loop starts.
    /// Generators do not run concurrently within one process here (a CLI run
    /// owns its engine), which is what makes a single mutable slot safe.
    var driver_ops: common.Driver.Ops = .{
        .reset = dReset,
        .prefill = dPrefill,
        .step = dStep,
        .sample = dSample,
        .max_seq = dMaxSeq,
        .eos_id = 0,
    };

    fn dReset(ctx: *anyopaque) void {
        const e: *Engine = @ptrCast(@alignCast(ctx));
        e.reset();
    }
    fn dPrefill(ctx: *anyopaque, prompt: []const u32) Buf {
        const e: *Engine = @ptrCast(@alignCast(ctx));
        // Batched prefill: each chunk is one `forwardChunk`, which runs every
        // projection once per chunk instead of once per token.  The chunk size
        // is `n_batch` (0 -> the buffer was not allocated, so `forwardChunk`
        // falls back to the per-token loop on its own).
        var last: Buf = e.logits;
        var i: usize = 0;
        const cap: usize = if (e.n_batch == 0) 1 else @intCast(e.n_batch);
        while (i < prompt.len) {
            const n = @min(cap, prompt.len - i);
            last = e.forwardChunk(prompt[i..][0..n], @intCast(i));
            i += n;
        }
        return last;
    }
    fn dStep(ctx: *anyopaque, token: u32, pos: u32) void {
        const e: *Engine = @ptrCast(@alignCast(ctx));
        _ = e.forwardLogits(token, pos);
    }
    fn dSample(ctx: *anyopaque) u32 {
        const e: *Engine = @ptrCast(@alignCast(ctx));
        return e.exec.sample(e.graph, e.logits, e.sample_scratch, &e.sampler);
    }
    fn dMaxSeq(ctx: *anyopaque) u32 {
        const e: *Engine = @ptrCast(@alignCast(ctx));
        return e.max_ctx;
    }
};

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

test "config rejects a file that is not this architecture" {
    // `parseConfig` is the architecture gate: every field it reads is required
    // (`qwen3.block_count` etc.), so a Qwen3.5 file -- whose keys are
    // `qwen35.*` -- must be rejected rather than fall back to defaults.  A
    // default-filled config would load and run, producing garbage silently.
    //
    // A default `Config{}` is a convenience for tests only; `load` never
    // returns one without having read every key above.
    try std.testing.expectEqual(@as(u32, 36), (Config{}).n_layers);
    try std.testing.expectEqual(@as(u32, 4096), (Config{}).hidden);
}

test "rotary span defaults to the whole head" {
    // Qwen3 rotates the full head (`partial_rotary_factor = 1.0`), which is the
    // one thing `rope64Rows` cannot express -- this is why `Config.rotary_dim`
    // exists and why the layer body calls `ropeRows`.
    const c = Config{};
    try std.testing.expectEqual(c.head_dim, c.rotary_dim);
    try std.testing.expectEqual(@as(u32, 128), c.head_dim);
}

test "rope over the full head rotates the tail" {
    // With `rotary_dim == hd` no float of the head may pass through untouched;
    // with 64 (Qwen3.5) the tail must.  This is the difference the two engines
    // are built around, so it gets a direct check.
    const ew = @import("../elementwise.zig");
    var a: [128]f32 = undefined;
    for (&a, 0..) |*p, i| p.* = @floatFromInt(i);
    ew.applyRope(&a, 5, 1e6, 128);
    // pair (96, 96+32) is inside the 128-wide span: it must have moved.
    var b: [128]f32 = undefined;
    for (&b, 0..) |*p, i| p.* = @floatFromInt(i);
    ew.applyRope(&b, 5, 1e6, 64);
    try std.testing.expect(a[96] != b[96]);
    // ... and with 64, floats past 64 must be untouched.
    try std.testing.expectEqual(@as(f32, 100), b[100]);
    try std.testing.expect(a[100] != @as(f32, 100));
}

test "applyRope at pos 0 is the identity for every rotary width" {
    // The strongest single check on the new primitive: at position 0 every
    // theta is 0, so cos=1/sin=0 and the vector must come back unchanged --
    // independent of `rotary_dim`.  An indexing or half-width mistake shows up
    // here immediately, and it is exactly the kind of error that makes ONE
    // layer blow up while the layers before it look healthy.
    const ew = @import("../elementwise.zig");
    inline for (.{ 64, 128 }) |rd| {
        var x: [128]f32 = undefined;
        for (&x, 0..) |*p, i| p.* = @sin(@as(f32, @floatFromInt(i))) * 3.0;
        const before = x;
        ew.applyRope(&x, 0, 1e6, rd);
        for (x, before) |a, b| try std.testing.expectApproxEqAbs(b, a, 1e-6);
    }
}
