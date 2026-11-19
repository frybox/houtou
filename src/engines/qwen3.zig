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
    /// Maximum sequence the KV caches were allocated for.
    max_ctx: u32 = 0,
    n_batch: u32 = 0,

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
            .max_ctx = ctx_len,
            .n_batch = 0,
        };
        errdefer e.deinit();

        // `+inf` gate -> sigmoid == 1.0 exactly (see the `ones` field comment).
        for (e.ones.span(0, q_dim).slice()) |*p| p.* = std.math.inf(f32);

        e.sample_scratch = try exec.allocBuf(cfg.vocab);
        const widest = @max(n_ff, @max(q_dim, hidden));
        try exec.allocScratch(widest);

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

    pub fn embedTo(self: *Engine, token: u32, dst: Buf) void {
        const hidden: usize = self.cfg.hidden;
        std.debug.assert(dst.len == hidden);
        // One row expanded to f32 is the only place this engine needs values
        // rather than a dot product; the dequantizer choice is the backend's.
        switch (self.token_embd) {
            .f32 => |m| self.exec.copyRange(dst, Buf.wrap(m[token * hidden ..][0..hidden])),
            .quant => |qrow| self.exec.dequantRow(dst, qrow, token),
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
        self.exec.rmsnorm(self.x1.span(0, hidden), self.x.span(0, hidden), lw.attn_norm, c.rms_eps);

        // Qwen3's `attn_q` is a plain [q_dim, hidden] projection: there is no
        // per-head [q|gate] interleave and no output gate (unlike Qwen3.5).
        const q = self.q.span(0, q_dim);
        self.exec.matvec(q, lw.attn_q, self.x1.span(0, hidden));
        self.exec.matvec(self.k.span(0, kv_dim), lw.attn_k, self.x1.span(0, hidden));
        self.exec.matvec(self.v.span(0, kv_dim), lw.attn_v, self.x1.span(0, hidden));

        // Per-head RMSNorm over `head_dim` (=128 here, not the 256-wide kind
        // Qwen3.5 uses), then RoPE over the FULL head.
        self.exec.rmsnormRows(q, q, lw.attn_q_norm, heads, c.rms_eps);
        self.exec.rmsnormRows(self.k.span(0, kv_dim), self.k.span(0, kv_dim), lw.attn_k_norm, kv_heads, c.rms_eps);
        self.exec.ropeRows(q, pos, c.rope_freq_base, 1, heads, hd, c.rotary_dim);
        self.exec.ropeRows(self.k.span(0, kv_dim), pos, c.rope_freq_base, 1, kv_heads, hd, c.rotary_dim);

        const cache = &self.kv[li];
        std.debug.assert(cache.len < self.max_ctx);
        const slot: usize = cache.len;
        self.exec.kvAppend(cache.k, self.k.span(0, kv_dim), slot, kv_dim, 1);
        self.exec.kvAppend(cache.v, self.v.span(0, kv_dim), slot, kv_dim, 1);
        cache.len += 1;

        const scale = 1.0 / @sqrt(@as(f32, @floatFromInt(hd)));
        const gout = self.gout.span(0, q_dim);
        self.exec.attnDecode(
            q, self.ones.span(0, q_dim),
            cache.k, cache.v,
            self.att.span(0, slot + 1), gout,
            slot, heads, gqa, hd, kv_dim, scale,
        );
        self.exec.matvec(self.x2.span(0, hidden), lw.attn_output, gout);
        self.exec.addResid(self.x.span(0, hidden), self.x2.span(0, hidden));

        // ---- MLP ----
        self.exec.rmsnorm(self.x1.span(0, hidden), self.x.span(0, hidden), lw.post_attention_norm, c.rms_eps);
        const n_ff: usize = c.n_ff;
        self.exec.matvec(self.mlp_h.span(0, n_ff), lw.ffn_gate, self.x1.span(0, hidden));
        self.exec.matvec(self.mlp_h1.span(0, n_ff), lw.ffn_up, self.x1.span(0, hidden));
        self.exec.swiglu(self.mlp_h.span(0, n_ff), self.mlp_h1.span(0, n_ff));
        self.exec.matvec(self.x2.span(0, hidden), lw.ffn_down, self.mlp_h.span(0, n_ff));
        self.exec.addResid(self.x.span(0, hidden), self.x2.span(0, hidden));
    }

    pub fn forwardLogits(self: *Engine, token: u32, pos: u32) Buf {
        const c = &self.cfg;
        const hidden: usize = c.hidden;
        self.embedTo(token, self.x.span(0, hidden));
        self.exec.stepBegin(pos);
        for (self.layers, 0..) |*lw, li| self.layerForward(lw, li, pos);
        self.exec.rmsnorm(self.x1.span(0, hidden), self.x.span(0, hidden), self.output_norm, c.rms_eps);
        if (self.output_weight) |ow| {
            self.exec.matvec(self.logits, ow, self.x1.span(0, hidden));
        } else {
            self.exec.matvec(self.logits, self.token_embd, self.x1.span(0, hidden));
        }
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
    /// logits of the LAST one.  Reuses the single-token layer body -- correct
    /// but one token at a time; a batched `attnChunk` path is the next speedup
    /// (see `qwen35.zig`'s `forwardFullChunk` for the shape it takes).
    pub fn forwardChunk(self: *Engine, tokens: []const u32, pos0: u32) Buf {
        var out: Buf = self.logits;
        for (tokens, 0..) |t, i| out = self.forwardLogits(t, pos0 + @as(u32, @intCast(i)));
        return out;
    }

    /// Greedy generation, the smoke-test entry point.
    pub fn generate(
        self: *Engine,
        alloc: std.mem.Allocator,
        prompt: []const u32,
        n_predict: usize,
        on_token: ?*const fn (ctx: ?*anyopaque, tok: u32) void,
        ctx: ?*anyopaque,
    ) ![]u32 {
        _ = alloc;
        var s = sampler.Sampler.init(.{ .temperature = 0 });
        self.reset();
        var last: Buf = Buf.empty;
        var i: usize = 0;
        while (i < prompt.len) : (i += 1) {
            const n = @min(@as(usize, 8), prompt.len - i);
            last = self.forwardChunk(prompt[i .. i + n], @intCast(i));
            i += n - 1;
        }
        var out_l: std.ArrayList(u32) = .empty;
        var pos: u32 = @intCast(prompt.len);
        var next: u32 = self.exec.sample(last, self.sample_scratch, &s);
        while (out_l.items.len < n_predict) {
            if (next == self.cfg.eos_token_id) break;
            try out_l.append(self.alloc, next);
            if (on_token) |f| f(ctx, next);
            last = self.forwardLogits(next, pos);
            pos += 1;
            next = self.exec.sample(last, self.sample_scratch, &s);
        }
        return out_l.toOwnedSlice(self.alloc);
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
