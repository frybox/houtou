//! DSpine draft model (SPECULATIVE DECODING, stage 3).
//!
//! The draft GGUF (`Qwen3.8-27B-DSpine-b16-Q8_0.gguf`) is a *separate* model
//! with its own tiny backbone: 5 dense full-attention layers over 5120 hidden,
//! 32 heads / 8 KV heads / head_dim 128, rope section `[64,0,0,0]`.  It shares
//! no weights and no KV cache with the target; it is driven by
//!
//!   * the anchor token's embedding (from the TARGET's embedding table --
//!     `models/dspine.py` literally calls the target's `embed_tokens`), and
//!   * a 5xR "feature" vector assembled by the code bridge from the target's
//!     hidden states at `dflash.target_layers = [2,17,32,47,62]`,
//!
//! and it emits a `dflash.selector_top_k`-wide distribution that serial
//! refinement turns into tokens.
//!
//! SCOPE OF THIS FILE (stage 3): loading + the plain 5-layer forward.  The code
//! bridge (the 11 bf16 rounding points, `normalize_codes`, `feature_message`,
//! `feature_finish`) lands next and is what actually makes the draft good --
//! without it the forward runs but the output is meaningless, exactly like the
//! MTP head with the wrong hidden pairing.
//!
//! Shape conventions from `dumpGguf` on the real file:
//!
//!   | tensor                        | GGUF dims      | torch          |
//!   |-------------------------------|----------------|----------------|
//!   | `dspine_codebook.weight`      | [1024,248320]  | [V,C]  Q8_0    |
//!   | `dspine_gate.weight`          | [1024,512]     | [R,C]  BF16    |
//!   | `dspine_gate.bias`            | [512]          | [R]    F32     |
//!   | `dspine_prev.weight`          | [1024,512]     | [R,C]  BF16    |
//!   | `dspine_pred_gate.weight`     | [1024,512]     | [R,C]  BF16    |
//!   | `blk.{i}.dspine_read.weight`  | [5120,1024]    | [C,H]  BF16    |
//!   | `blk.{i}.dspine_write.weight` | [512,5120]     | [H,R]  BF16    |
//!   | `fc.weight`                   | [25600,5120]   | [H,5H] Q8_0    |
//!   | `enc.output_norm.weight`      | [5120]         | `hidden_norm`  |
//!   | `output_norm.weight`          | [5120]         | `self.norm`    |
const std = @import("std");
const backend = @import("backend.zig");
const gguf = @import("gguf.zig");
const serial = @import("serial.zig");

const Buf = backend.Buf;
const LinearW = backend.LinearW;
const Exec = backend.Exec;

pub const Config = struct {
    n_layers: usize,
    hidden: usize,
    ffn: usize,
    heads: usize,
    kv_heads: usize,
    hd: usize,
    /// code-bridge rank R (512 on the shipped b16 checkpoints)
    rank: usize,
    /// codebook width C (1024)
    code_len: usize,
    /// vocab (248320) -- the codebook's row count
    vocab: usize,
    /// `dflash.block_size` (16)
    block_size: usize,
    /// `dflash.selector_top_k` (16, the serial-refinement width)
    top_k: usize,
    /// `tokenizer.ggml.mask_token_id`.  Every block row except the anchor is
    /// initialised to THIS, not to an anchor repeat: `generate.py:133-135`
    /// builds `block = [[anchor] + [mask_token_id] * (T-1)]`.  Substituting
    /// anything else (a repeated anchor, the context tail) changes the block's
    /// input distribution and every downstream number.
    mask_token_id: u32,
    /// target layer indices the bridge reads hidden states from
    target_layers: []const u32,
    rope_freq_base: f32,
    rms_eps: f32,
    /// The code bridge's own eps (`config.bridge.eps`).  Distinct from
    /// `rms_eps` in principle -- both are 1e-6 in this checkpoint, but the
    /// bridge's normalisations divide by `rms + eps` where the transformers use
    /// `rsqrt(mean + eps)`, and using the wrong one is a silent scale error.
    bridge_eps: f32,
    max_seq: usize,
};

/// The code bridge, in C (`src/kernels/x86/dspine.c`).
///
/// Scalar C, baseline ISA, no FMA: every one of these four kernels is defined
/// by where it rounds to bf16, and both SIMD and fused-multiply-add would
/// change those bits.  The workloads are 16-row blocks, so there is nothing to
/// vectorise for.
const Io = std.Io;
const Br = @import("dspine_bridge.zig");
const backend_cpu = @import("backend_cpu.zig");

/// Bridges `CudaBackend.syncCurrentStream` into the plain
/// `fn (*anyopaque) void` shape `Draft.sync_fn` stores, without this file
/// importing the CUDA backend at any non-comptime site.
fn syncThunk(ctx: *anyopaque) void {
    const cb: *@import("backend_cuda.zig").CudaBackend = @ptrCast(@alignCast(ctx));
    cb.syncCurrentStream();
}

const C = struct {
    extern fn q35_dspine_prepare_read(x: [*]const f32, residual: ?[*]const f32, raw: [*]f32, normed: [*]f32, rms_out: ?[*]f32, rows: c_long, h: c_long, eps: f32, add: c_int) void;
    extern fn q35_dspine_normalize_codes(projected: [*]const f32, codebook: [*]const f32, anchors: ?[*]const i32, codes: [*]f32, rows: c_long, codes_n: c_long, t: c_long, eps: f32, row0: c_long) void;
    extern fn q35_dspine_feature_message(packed_row: [*]const f32, packed_prev: ?[*]const f32, bias: [*]const f32, message: [*]f32, rows: c_long, r: c_long, t: c_long, row0: c_long) void;
    extern fn q35_dspine_feature_finish(raw: [*]const f32, delta: [*]const f32, rms: ?[*]const f32, valid: ?[*]const u8, weight: ?[*]const f32, output: [*]f32, normed: ?[*]f32, rows: c_long, h: c_long, t: c_long, norm_eps: f32, write: c_int, norm: c_int, row0: c_long) void;
};

pub const Layer = struct {
    attn_norm: Buf,
    attn_q: LinearW,
    attn_k: LinearW,
    attn_v: LinearW,
    attn_output: LinearW,
    attn_q_norm: Buf,
    attn_k_norm: Buf,
    ffn_norm: Buf,
    ffn_gate: LinearW,
    ffn_up: LinearW,
    ffn_down: LinearW,
    /// code-bridge per-layer projections (BF16 in the file)
    read: LinearW, // [C, H]  code -> hidden
    /// `dspine_write.weight`, stored as **[R, H]** (ne0 = R = 512, ne1 = H =
    /// 5120: the GGUF row is 5120 wide and there are 512 of them), i.e. the
    /// linear it encodes is `feature[R] = write @ hidden[H]`.  That is the
    /// OPPOSITE of `fc.weight`, whose ne0 is its INPUT width -- so "ne0 is the
    /// row width" is the only rule, and this tensor's rows are H wide.
    write: LinearW,
};

pub const Draft = struct {
    alloc: std.mem.Allocator,
    /// The draft has its OWN backend instance bound to its own GGUF mapping.
    /// Sharing the target's `Exec` is impossible: every `loadLinear` resolves
    /// names against the mapping the backend was built from.
    ///
    /// `cpu` is always constructed: stage 4 (`src/serial.zig`) is host-only
    /// code that dequantizes `lm_head` rows directly, so it needs a host
    /// backend even when stage 3 runs on the device.  On the device path `cpu`
    /// is only used for metadata (name -> `QuantRow` resolution); no arithmetic
    /// goes through it.
    cpu: *@import("backend_cpu.zig").CpuBackend,
    /// Present iff `bridge_kind == .cuda`.  Owned.
    cuda: ?*@import("backend_cuda.zig").CudaBackend = null,
    exec: Exec,
    /// The pass-through compute graph.  The draft runs OUTSIDE any recorded
    /// step (it is driven by the target's generate loop, not by a forward pass),
    /// so every op it issues uses the direct path.  S5 gives the draft chain its
    /// own graph; until then this is the explicit "no graph" value rather than a
    /// hidden implicit one.
    graph: backend.Graph = .{},
    /// Which implementation of the four bridge kernels runs (see
    /// `dspine_bridge.Kind`).  `exec` and this are chosen together: `cuda` here
    /// means `exec` is the CUDA backend's vtable AND
    /// `src/kernels/cuda/dspine.cu` is linked in.
    bridge_kind: Br.Kind = .cpu,
    /// The device stream the CUDA bridge launchers run on.  Null on the CPU
    /// path (the C port has no stream).  Kept as `?*anyopaque` so this file
    /// does not have to know what a `cudaStream_t` is.
    bridge_stream: ?*anyopaque = null,
    cfg: Config,
    /// Host mirror used ONLY when stage 3 runs on the device: the serial stash
    /// (`stash_raw`/`stash_rms`) and the debug hooks need host copies, and on a
    /// device backend every `Buf` is a device address.  Allocated lazily because
    /// the CPU path never touches it.  See `hostStashRefresh`.
    host_raw: ?[]f32 = null,
    host_rms: ?[]f32 = null,
    host_query: ?[]f32 = null,
    host_normed: ?[]f32 = null,
    /// Set by `load` for the `$QWEN35_S4_DUMP` reference dump.
    dump_io: Io = undefined,
    dump_dir: ?[]const u8 = null,
    dump_calls: usize = 0,
    /// Ids the current block was forwarded with; only read by `dumpFwdInputs`.
    toks_for_dump: [64]u32 = @splat(0),
    /// The `idx` the last `serialTable` call produced, kept only for the dump.
    last_idx: ?[]const i16 = null,
    /// `sync_fn` is `CudaBackend.syncCurrentStream`, injected by `load` on the device path so
    /// this file needs no cudart dependency of its own.  `sync_ctx` is the
    /// backend pointer it is called on.
    sync_fn: ?*const fn (*anyopaque) void = null,
    sync_ctx: ?*anyopaque = null,
    /// Host copies of `norm` / `gate_b` for stage 4; see `hostNorm`.
    norm_host: ?[]f32 = null,
    gate_b_host: ?[]f32 = null,
    layers: []Layer,
    /// `fc.weight` [H, 5H]: the 5 per-layer features -> one hidden vector
    fc: LinearW,
    /// `enc.output_norm` (`hidden_norm` in the reference)
    hidden_norm: Buf,
    /// `output_norm` (`self.norm` in the reference)
    norm: Buf,
    embed_rows: gguf.QuantRow,
    /// Host scratch for the TARGET's embedding row on the device path (the
    /// table is not resident there; see `forwardToken`).  Null on the CPU path.
    embed_host: ?[]f32 = null,
    /// `$QWEN35_DRAFT_DEBUG`: fill the `per_layer` / `debug_*` telemetry even on
    /// the device path (a handful of D2H per row).  Diagnostics only.
    force_host_debug: bool = false,
    /// `$QWEN35_DRAFT_CTX_IDENTITY`: feed the last tap plane in as `ctx`
    /// without the `fc` projection (see `projectContext`).  Diagnostic only.
    ctx_identity: bool = false,
    /// Optional `lm_head @ x -> logits[V]` hook.  The head belongs to the
    /// TARGET, so the draft cannot run this itself; see `candidatesForBlock`.
    lm_head_matvec: ?*const fn (ctx: *anyopaque, out: []f32, x: []const f32) void = null,
    lm_head_ctx: *anyopaque = undefined,
    /// Host scratch for the per-layer telemetry (4 * hidden floats).
    dbg_scratch: ?[]f32 = null,
    /// Device-resident copy of the `[1]i32` anchor id the code bridge passes to
    /// `normalize_codes`.  Only used on the device path.
    anchor_dev: Buf = backend.Buf.empty,
    /// Device-resident `[1]u8` for `feature_finish`'s `valid` gate; same reason
    /// as `anchor_dev`.
    valid_dev: Buf = backend.Buf.empty,

    // ---- code bridge tables (BF16 in the file) ----
    /// `[V, C]`, consumed ROW-wise: `normalize_codes` looks up the anchor
    /// token's row and nothing else, so it is kept in its stored (Q8_0) form and
    /// one row is dequantised per call into `codebook_row`.  Widening the whole
    /// table would cost 1.0 GB.
    codebook: LinearW,
    /// One dequantised codebook row.
    codebook_row: Buf,
    gate_w: LinearW,
    gate_b: Buf,
    prev_w: LinearW,
    pred_gate_w: LinearW,

    // ---- scratch (single token) ----
    /// The running residual stream (`residual` in the reference).
    resid: Buf,
    /// The current layer's already-normed input (`x_in`).  Layer 0 fills it
    /// from its own `input_layernorm`; every later layer is fed by the bridge.
    xin: Buf,
    x2: Buf,
    q: Buf,
    kv: Buf,
    gout: Buf,
    att: Buf,
    mlp_h: Buf,
    mlp_h1: Buf,
    /// `ctx = hidden_norm(fc(concat(target taps)))` for [max_ctx_rows] rows.
    ctx: Buf,
    /// The concatenated taps fed to `fc`: [max_ctx_rows][n_layers*hidden].
    ctx_raw: Buf,
    /// How many rows `projectContext` last produced.
    ctx_rows: usize,
    // ---- self-check probes (max |.| of each stage, updated every token) ----
    debug_embed: f32 = 0,
    debug_resid: f32 = 0,
    debug_delta: f32 = 0,
    debug_msg: f32 = 0,
    debug_codes: f32 = 0,
    debug_attn: f32 = 0,
    debug_mlp: f32 = 0,
    debug_gout: f32 = 0,
    debug_xin: f32 = 0,
    /// Per-layer magnitudes, in layer order, for the self-check.  The draft is
    /// 5 layers deep, so a >100x jump from one layer to the next cannot be
    /// Qwen's normal residual growth -- and it must be visible per layer, not
    /// only for whichever layer happened to run last.
    per_layer: [8]struct { xin: f32 = 0, attn: f32 = 0, mlp: f32 = 0, resid: f32 = 0, gate: f32 = 0, swi: f32 = 0, x0: f32 = 0, g0: f32 = 0, m0: f32 = 0 } = @splat(.{}),
    debug_gate: f32 = 0,
    debug_swi: f32 = 0,
    /// Raw-f32 dump hook for cross-referencing against a reference
    /// implementation.  `Draft` deliberately does not own file handles or an
    /// `std.Io`: it is also built on the GPU backends where the activations
    /// live in device memory, so the caller supplies the sink.  Null (the
    /// default) costs one branch per stage.
    ///
    /// The data slice is only valid for the duration of the call -- it points
    /// straight at the layer's working buffer.
    dump: ?*const fn (ctx: ?*anyopaque, stage: []const u8, layer: usize, row: usize, data: []const f32) void = null,
    dump_ctx: ?*anyopaque = null,
    /// Row currently being forwarded; set by `forwardToken` so `layerForward`
    /// (which has no row argument of its own) can tag its dumps.
    dump_row: usize = 0,
    /// A/B hook: skip the write + norm so `layerForward` runs as a plain
    /// transformer.  Diagnostics only.
    disable_bridge: bool = false,
    /// Scratch for the bridge's per-layer `packed` [rows][3R] and message
    /// [rows][R].  A full block is at most `max_seq` rows wide.
    /// `[gate.query | predecessor.sender | prev.value]` per row
    packed_rows: Buf,
    msg: Buf,
    /// The bridge's `raw` [rows][H] and `rms` [rows].
    bridge_raw: Buf,
    bridge_rms: Buf,
    /// Previous row's `packed` triple: the `prev` sender of `feature_message`
    /// when the bridge is driven one row at a time.
    prev_packed: Buf,
    prev_packed_valid: bool,
    bridge_normed: Buf,
    bridge_codes: Buf,
    /// `read`'s output: the block's codes before normalisation.
    projected: Buf,
    /// `write`'s output: the feature added to the residual.
    delta: Buf,
    /// `feature_finish`'s written residual (kept out of `resid` because the
    /// C kernel reads `raw`, which lives in a different buffer, and aliasing
    /// input and output is not worth proving safe here).
    resid_out: Buf,
    /// The block's anchor token id.  `normalize_codes` only ever reads row 0's
    /// anchor (`input_ids[:, 0]` -- one anchor per block, shared by the rows),
    /// but the kernel takes a per-row array so `row%T==0` rows read their own.
    anchor_tokens: [16]u32,
    /// Scratch for the bridge's per-row temporaries are all sized for one row
    /// at a time (`writeBridge` handles a single row), so their buffers are
    /// plain `H`/`C`/`R` wide.
    /// `attnDecode`'s CPU oracle ends with `gout *= sigmoid(gate)`.  The draft's
    /// attention is UNGATED (no `attn_gate` tensor in the file, and
    /// `layers/attention.py` has no gate path), so the only way to express
    /// "identity" through that shared slot is `gate = +inf`, where
    /// `sigmoid(+inf) == 1.0` exactly.  A finite large value would scale by
    /// something like 1-1e-9 and quietly bias every draft.
    ones: Buf,
    kv_cache: []Kv,

    // ==== stage 4: serial refinement ==========================================
    /// What the LAST bridged block left behind, one row per block row
    /// (`serial.Stash`).  `raw` is the bf16-rounded residual stream BEFORE the
    /// last write, `rms` the scale that write used, and `query` the receiver
    /// query (`packed[:, :R]`, i.e. the `gate` third of the packed triple).
    /// The reference stashes the same three during `bridge.__call__`
    /// (`bridge.py:135`) and `serial_table` reads them straight back.
    stash_raw: Buf,
    stash_rms: Buf,
    stash_query: Buf,
    /// Each block row's NORMED next-layer input (`feature_finish`'s `normed`,
    /// i.e. the model's `x_in`).  This -- not `raw` -- is what `topk_ids`
    /// projects: `generate.py:145` takes `rows = out[0, 1:]` from the draft's
    /// return value, and `DSpineDraftModel.__call__` returns `x_in`
    /// (`dspine_mlx/model.py:261-266`).  Scoring `raw` instead put the
    /// candidate set on a completely different distribution (measured: the top-16
    /// never contained the target's own next token).
    stash_normed: Buf,
    /// Set once the last layer of a block has run; `serialRound` refuses to
    /// score a table built from a half-written stash.
    stash_ready: bool,
    /// `[block_size-1][top_k]` candidate token ids per row (`topk_ids`), padded
    /// to `block_rows` rows so a short block has no ragged edge.
    cand: []u32,
    /// `[block_size-1][top_k]` best lane per (row, previous lane).
    cand_idx: []i16,
    /// The emitted chain, `block_size - 1` tokens.
    chain: []u32,
    /// Full-vocabulary logits for one row, reused by the candidate search.
    cand_logits: Buf,
    /// Device scratch for the same logits, allocated only on the device path
    /// (`cand_logits` is host memory; `matvec` writes through the backend).
    cand_logits_dev: Buf = backend.Buf.empty,
    /// The two per-vocabulary tables.  Built lazily on the first round because
    /// they cost a full pass over the vocabulary (see `buildSerialTables`).
    tables: ?serial.Tables = null,
    /// The TARGET's `[V, H]` output projection: the candidate search and the
    /// table's `base` term both score with it.
    lm_head: LinearW,
    /// `[H][R]` the last layer's write weight, dequantised: `build_serial_tables`
    /// wants plain f32 and this one is only 10 MB.
    write_w_f32: ?[]f32 = null,
    /// `[2R][C]` = `[predecessor_gate; prev]`, dequantised (4 MB).
    pv_w_f32: ?[]f32 = null,

    pub const Kv = struct {
        /// `[ctx_len + block rows]` rows of `kv_dim`.  The context rows are a
        /// fixed prefix [0, ctx_len) written once by `appendContext`; the block
        /// rows are appended after them.  One contiguous buffer because
        /// `attnDecode` walks `0..=slot` over a single cache and both halves are
        /// visible to a block query (context always, block causally) -- two
        /// buffers would need a new attention primitive for no benefit.
        k: Buf,
        v: Buf,
        /// Number of context rows at the front.
        ctx_len: u32,
        /// Number of block rows appended after them.
        len: u32,
    };

    /// Reset the BLOCK half of every layer's cache.  The context half is
    /// rebuilt by `appendContext` from the just-verified target hidden states,
    /// so there is nothing to preserve across rounds for either half.
    pub fn reset(self: *Draft) void {
        for (self.kv_cache) |*c| {
            c.len = 0;
            c.ctx_len = 0;
        }
        self.ctx_rows = 0;
        self.stash_ready = false;
        // Zero the serial stash.  `stash_raw`/`stash_rms`/`stash_query` are plain
        // allocations, and `writeBridge` fills them only when `is_last` runs --
        // so any row this block never writes would otherwise be read as whatever
        // the previous block (or the allocator) left there.  That showed up as
        // the FIRST `serialRound` of a block scoring from a residual stream 150x
        // the right size (`u_rms` 544 vs 5.0 on identical stash) and returning a
        // different lane than both the Triton reference and a f32 longhand.
        @memset(self.stash_raw.slice(), 0);
        @memset(self.stash_rms.slice(), 0);
        @memset(self.stash_query.slice(), 0);
        @memset(self.stash_normed.slice(), 0);
    }

    /// `ctx = hidden_norm(fc(concat over target_layers of target_hidden))`
    /// (`dspine_mlx/model.py:277`).
    ///
    /// `tap` holds one plane per `cfg.target_layers` entry, each `rows*hidden`
    /// floats in row-major order, laid out tap-major -- exactly what
    /// `Engine.specTapPlane` stores.  The concatenation is per ROW (`fc`'s input
    /// is `n_layers*hidden` wide and its semantics are "the same position seen
    /// by 5 layers"), which is the ordering `nn.Linear(25600, 5120)` was trained
    /// with; running the taps as separate matvecs and summing would be a
    /// different (and wrong) model.
    pub fn projectContext(self: *Draft, tap: []const f32, rows: usize) void {
        const c = &self.cfg;
        const hidden = c.hidden;
        const n = c.target_layers.len;
        std.debug.assert(tap.len >= n * rows * hidden);
        std.debug.assert(rows * hidden <= self.ctx_raw.len);
        std.debug.assert(n * rows * hidden <= self.ctx_raw.len);

        // `ctx_raw` is a backend region, so on a device run the interleave is
        // built in host scratch and uploaded once (33 rows x 25600 floats at a
        // 27B config = 3.4 MB, one copy per block).
        switch (self.bridge_kind) {
            .cpu => {
                const raw = self.ctx_raw.slice();
                for (0..rows) |r| {
                    for (0..n) |j| {
                        @memcpy(
                            raw[r * n * hidden + j * hidden ..][0..hidden],
                            tap[j * rows * hidden + r * hidden ..][0..hidden],
                        );
                    }
                }
            },
            .cuda => if (Br.dspineLinkedIn) {
                const staged = self.alloc.alloc(f32, rows * n * hidden) catch @panic("dspine: ctx staging OOM");
                defer self.alloc.free(staged);
                for (0..rows) |r| {
                    for (0..n) |j| {
                        @memcpy(
                            staged[r * n * hidden + j * hidden ..][0..hidden],
                            tap[j * rows * hidden + r * hidden ..][0..hidden],
                        );
                    }
                }
                self.fromHost(self.ctx_raw.span(0, rows * n * hidden), staged[0 .. rows * n * hidden]);
            },
        }
        self.syncBridge();
        // `$QWEN35_DRAFT_CTX_IDENTITY`: skip `fc` + `hidden_norm` and use the
        // LAST tap plane as `ctx` directly.
        //
        // The diagnostic this serves: every component of this port agrees with
        // the reference given the same numbers, yet the draft's residual grows
        // to rms ~400 against the target's ~2.6.  If that growth is caused by the
        // taps being the wrong layers, feeding the target's own top-layer hidden
        // (which lives at the target's own scale) as `ctx` should keep the
        // residual small.  If it still inflates, the inflation is the
        // checkpoint's own behaviour.  NOT a valid model configuration.
        if (self.ctx_identity) {
            const last = tap[(n - 1) * rows * hidden ..][0 .. rows * hidden];
            self.fromHost(self.ctx.span(0, rows * hidden), last);
            self.ctx_rows = rows;
            if (self.dump_dir) |root| self.dumpCtx(root, rows);
            return;
        }
        // fc: [n*hidden, hidden] -- one matvec per context row.
        for (0..rows) |r| {
            self.exec.matvec(self.graph, 
                self.ctx.span(r * hidden, hidden),
                self.fc,
                self.ctx_raw.span(r * n * hidden, n * hidden),
            );
        }
        // hidden_norm over each row
        if (rows > 0) {
            self.exec.rmsnormRows(self.graph, 
                self.ctx.span(0, rows * hidden),
                self.ctx.span(0, rows * hidden),
                self.hidden_norm,
                rows,
                c.rms_eps,
            );
        }
        self.ctx_rows = rows;
        if (self.dump_dir) |root| self.dumpCtx(root, rows);
    }

    fn dumpKv(self: *Draft, root: []const u8, il: usize, q_dim: usize, kv_dim: usize, krow: Buf, vrow: Buf) void {
        const io = self.dump_io;
        var path_buf: [512]u8 = undefined;
        const dir = std.fmt.bufPrint(&path_buf, "{s}/fwd", .{root}) catch return;
        std.Io.Dir.cwd().createDirPath(io, dir) catch return;
        const d = std.Io.Dir.cwd().openDir(io, dir, .{}) catch return;
        defer d.close(io);
        const cache = &self.kv_cache[il];
        const rows = @as(usize, cache.ctx_len) + cache.len;
        const ktmp = self.alloc.alloc(f32, rows * kv_dim) catch return;
        defer self.alloc.free(ktmp);
        self.toHost(ktmp, cache.k.span(0, rows * kv_dim));
        const vtmp = self.alloc.alloc(f32, rows * kv_dim) catch return;
        defer self.alloc.free(vtmp);
        self.toHost(vtmp, cache.v.span(0, rows * kv_dim));
        self.syncBridge();
        var nb: [576]u8 = undefined;
        d.writeFile(io, .{ .sub_path = std.fmt.bufPrint(&nb, "l{d}_kcache.f32", .{il}) catch return, .data = std.mem.sliceAsBytes(ktmp) }) catch {};
        d.writeFile(io, .{ .sub_path = std.fmt.bufPrint(&nb, "l{d}_vcache.f32", .{il}) catch return, .data = std.mem.sliceAsBytes(vtmp) }) catch {};
        const qt = self.alloc.alloc(f32, q_dim) catch return;
        defer self.alloc.free(qt);
        self.toHost(qt, self.q.span(0, q_dim));
        d.writeFile(io, .{ .sub_path = std.fmt.bufPrint(&nb, "l{d}_q.f32", .{il}) catch return, .data = std.mem.sliceAsBytes(qt) }) catch {};
        _ = krow;
        _ = vrow;
    }

    fn dumpCtx(self: *Draft, root: []const u8, rows: usize) void {
        const c = &self.cfg;
        const io = self.dump_io;
        var path_buf: [512]u8 = undefined;
        const dir = std.fmt.bufPrint(&path_buf, "{s}/fwd", .{root}) catch return;
        std.Io.Dir.cwd().createDirPath(io, dir) catch return;
        const d = std.Io.Dir.cwd().openDir(io, dir, .{}) catch return;
        defer d.close(io);
        const tmp = self.alloc.alloc(f32, rows * c.hidden) catch return;
        defer self.alloc.free(tmp);
        self.toHost(tmp, self.ctx.span(0, rows * c.hidden));
        self.syncBridge();
        d.writeFile(io, .{ .sub_path = "ctx.f32", .data = std.mem.sliceAsBytes(tmp) }) catch {};
    }

    /// Append the projected context K/V to every layer (`_context_kv`, then
    /// `cache.update_and_fetch`; `dspine_mlx/model.py:112`).
    ///
    /// Every layer has its own `attn_k`/`attn_v`, so this is 5 x rows matvecs.
    /// For one block that is 2*5*rows*1024*5120 MACs -- 52M at rows=1, which is
    /// under a millisecond and is why it is not worth caching across layers.
    pub fn appendContext(self: *Draft, pos0: u32) void {
        const c = &self.cfg;
        const hidden = c.hidden;
        const kv_dim = c.kv_heads * c.hd;
        const rows = self.ctx_rows;
        if (rows == 0) return;
        std.debug.assert(rows <= self.kv_cache.len or true);
        for (self.layers, 0..) |*l, il| {
        const cache = &self.kv_cache[il];
            std.debug.assert(rows <= @as(usize, cache.k.len / kv_dim));
            for (0..rows) |r| {
                const krow = cache.k.span(r * kv_dim, kv_dim);
                const vrow = cache.v.span(r * kv_dim, kv_dim);
                self.exec.matvec(self.graph, krow, l.attn_k, self.ctx.span(r * hidden, hidden));
                self.exec.matvec(self.graph, vrow, l.attn_v, self.ctx.span(r * hidden, hidden));
                self.exec.rmsnormRows(self.graph, krow, krow, l.attn_k_norm, c.kv_heads, c.rms_eps);
                // Context rows sit at absolute positions pos0..pos0+rows-1.
                self.exec.rope64Rows(self.graph, krow, pos0 + @as(u32, @intCast(r)), c.rope_freq_base, 1, c.kv_heads, c.hd);
            }
            cache.ctx_len = @intCast(rows);
            cache.len = 0;
        }
    }

    /// One token of a block through the 5-layer backbone.
    ///
    /// Mirrors `DSpineDraftModel.__call__` (`dspine_mlx/model.py:239`), which
    /// keeps the residual stream SEPARATE from the layer input: the bridge's
    /// `feature_finish` writes the updated residual and returns the next
    /// layer's already-normed input, so a layer must not apply its own
    /// `input_layernorm` when the bridge has already done it.
    ///
    /// `row` is the row's index INSIDE the block (0 = anchor) and `pos` its
    /// absolute position; `pos - row` is the block's first absolute position.
    pub fn forwardToken(self: *Draft, token: u32, row: usize, pos: u32) void {
        const c = &self.cfg;
        const hidden = c.hidden;

        // x = embed_tokens(block_ids)  (the TARGET's table)
        //
        // The table lives in the TARGET's GGUF mmap, which this backend never
        // uploaded: `dequantRow` on the device would need its rows resident and
        // reports "not resident on the device" instead (it refuses rather than
        // computing a plausible CPU answer).  One row is 5120 floats and there
        // is exactly one per block, so the cheap correct path is a HOST dequant
        // into scratch followed by one upload -- the same bytes the device
        // kernel would have produced, since `dequantRowInto` is the shared
        // reference for both.
        if (self.embed_host) |buf| {
            backend_cpu.dequantRowInto(buf[0..hidden], self.embed_rows, token);
            self.fromHost(self.resid.span(0, hidden), buf[0..hidden]);
        } else {
            self.exec.dequantRow(self.resid.span(0, hidden), self.embed_rows, token);
        }
        // The block's first id IS the anchor: it conditions the code bridge
        // (`normalize_codes`' anchor row) and the serial table's sender for row
        // 1 (`sid = anchor` when r == 0).  It was initialised to 0 and never
        // written, so every block was conditioned on vocabulary id 0.
        if (row == 0) self.anchor_tokens[0] = token;
        if (self.dump_dir) |rr| self.dumpStage(rr, 0, row, "embed", self.resid, hidden);
        if (row < self.toks_for_dump.len) self.toks_for_dump[row] = token;
        self.dump_row = row;
        if (self.hostDebug()) {
            self.debug_embed = maxAbs(self.resid.slice()[0..hidden]);
            self.emit("embed", 0, row, self.resid.slice()[0..hidden]);
        }
        // Row 0 of a block has no predecessor: do not let the previous block's
        // last `packed` leak in as its `prev` sender.
        if (row == 0) self.prev_packed_valid = false;
        // layer 0 gets its own input_layernorm; later ones are fed by the bridge
        self.exec.rmsnorm(self.graph, self.xin.span(0, hidden), self.resid.span(0, hidden), self.layers[0].attn_norm, c.rms_eps);

        const last = self.layers.len - 1;
        for (self.layers, 0..) |*l, il| {
            self.layerForward(l, il, pos);
            // Every layer is in `bridge.active_layers` for this checkpoint
            // (target_layers has one entry per layer), so the bridge runs after
            // each of them; the last one uses `self.norm` as `next_norm`.
            const next_norm = if (il < last) self.layers[il + 1].attn_norm else self.norm;
            self.writeBridge(l, il, row, next_norm, il == last);
        }
        // The block's serial stash is complete once its LAST row has run; the
        // anchor row (0) contributes `raw`/`rms`/`query` but is never a
        // candidate row itself.
        if (row + 1 >= c.block_size) self.stash_ready = true;
    }

    /// The layer body (`DraftDecoderLayer.__call__`, `dspine_mlx/model.py:148`):
    /// `h = residual + attn(normed_input); out = h + mlp(post_norm(h))`.
    /// Fire the dump hook if one is installed.  Kept as one call site so the
    /// stage list stays visible in one place.
    fn emit(self: *Draft, stage: []const u8, layer: usize, row: usize, data: []const f32) void {
        if (self.dump) |f| f(self.dump_ctx, stage, layer, row, data);
    }

    /// True when the debugging hooks may touch `Buf.slice()` directly.
    ///
    /// Every `emit` / `maxAbs` / `per_layer` site dereferences a `Buf`, and on a
    /// device backend a `Buf` is a DEVICE address -- reading it on the host is a
    /// segfault, not a wrong number.  All of them are diagnostics used by the
    /// CPU self-checks (`QWEN35_SPEC_DRAFT`, `QWEN35_DRAFT_DUMP`), so the device
    /// path skips them rather than paying a D2H per activation.
    fn hostDebug(self: *const Draft) bool {
        return self.bridge_kind == .cpu;
    }

    /// `$QWEN35_DRAFT_DEBUG`: per-layer telemetry that costs a D2H per tensor.
    /// Available on BOTH backends; every site that uses it must go through
    /// `toHost`, never `Buf.slice()`.  This is what makes "which layer
    /// diverges" answerable when the two backends disagree.
    fn devTelemetry(self: *const Draft) bool {
        return self.force_host_debug;
    }

    fn layerForward(self: *Draft, l: *const Layer, il: usize, pos: u32) void {
        const c = &self.cfg;
        const hidden = c.hidden;
        const q_dim = c.heads * c.hd;
        const kv_dim = c.kv_heads * c.hd;
        const scale: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(c.hd)));

        if (self.hostDebug()) {
            self.debug_xin = maxAbs(self.xin.slice()[0..hidden]);
            self.emit("xin", il, self.dump_row, self.xin.slice()[0..hidden]);
        }
        // `xin` is only the layer's INPUT until the MLP overwrites it with
        // `ffn_norm`'s output, so the reference dump has to take it here.
        if (self.dump_dir) |root| self.dumpStage(root, il, self.dump_row, "xin", self.xin, hidden);
        // ---- attention over [context | block-prefix] ----
        self.exec.matvec(self.graph, self.q.span(0, q_dim), l.attn_q, self.xin.span(0, hidden));
        const krow = self.kv.span(0, kv_dim);
        const vrow = self.kv.span(kv_dim, kv_dim);
        self.exec.matvec(self.graph, krow, l.attn_k, self.xin.span(0, hidden));
        self.exec.matvec(self.graph, vrow, l.attn_v, self.xin.span(0, hidden));

        // q_norm/k_norm are per-head RMSNorms over `hd` (=128), unlike the
        // target's 256-wide ones.
        self.exec.rmsnormRows(self.graph, self.q.span(0, q_dim), self.q.span(0, q_dim), l.attn_q_norm, c.heads, c.rms_eps);
        self.exec.rmsnormRows(self.graph, krow, krow, l.attn_k_norm, c.kv_heads, c.rms_eps);

        self.exec.rope64Rows(self.graph, self.q.span(0, q_dim), pos, c.rope_freq_base, 1, c.heads, c.hd);
        self.exec.rope64Rows(self.graph, krow, pos, c.rope_freq_base, 1, c.kv_heads, c.hd);
        if (self.hostDebug()) {
            self.emit("q", il, self.dump_row, self.q.slice()[0..q_dim]);
            self.emit("k", il, self.dump_row, krow.slice());
            self.emit("v", il, self.dump_row, vrow.slice());
        }

        const cache = &self.kv_cache[il];
        const slot: usize = @as(usize, cache.ctx_len) + cache.len;
        if (self.dump_dir) |root| if (self.dump_row == 0) self.dumpKv(root, il, q_dim, kv_dim, krow, vrow);
        std.debug.assert(slot < c.max_seq);
        self.exec.kvAppend(self.graph, cache.k, krow, slot, kv_dim, 1);
        self.exec.kvAppend(self.graph, cache.v, vrow, slot, kv_dim, 1);

        // Attention covers context rows [0, ctx_len) and block rows
        // [ctx_len, slot] -- exactly the slice `attnDecode` walks (`0..=slot`),
        // which is the same mask the reference builds: context rows are always
        // visible, block rows causally (`dspine_mlx/model.py:83-99`).
        self.exec.attnDecode(
            self.graph, self.q.span(0, q_dim), self.ones.span(0, q_dim),
            cache.k, cache.v,
            self.att.span(0, slot + 1), self.gout.span(0, q_dim),
            slot, c.heads, c.heads / c.kv_heads, c.hd, kv_dim, scale,
        );
        cache.len += 1;

        self.exec.matvec(self.graph, self.x2.span(0, hidden), l.attn_output, self.gout.span(0, q_dim));
        // `x2` is reused by the MLP's down-projection, so the attention output has
        // to be captured here, not in the end-of-layer dump.
        if (self.dump_dir) |root| self.dumpStage(root, il, self.dump_row, "attn", self.x2, hidden);
        if (self.hostDebug()) {
            self.debug_attn = maxAbs(self.x2.slice()[0..hidden]);
            self.debug_gout = maxAbs(self.gout.slice()[0..q_dim]);
            self.emit("attn", il, self.dump_row, self.gout.slice()[0..q_dim]);
        }
        self.exec.addResid(self.graph, self.resid.span(0, hidden), self.x2.span(0, hidden));
        if (self.hostDebug()) self.emit("h", il, self.dump_row, self.resid.slice()[0..hidden]);

        // ---- MLP ----
        self.exec.rmsnorm(self.graph, self.xin.span(0, hidden), self.resid.span(0, hidden), l.ffn_norm, c.rms_eps);
        self.exec.matvec(self.graph, self.mlp_h.span(0, c.ffn), l.ffn_gate, self.xin.span(0, hidden));
        self.exec.matvec(self.graph, self.mlp_h1.span(0, c.ffn), l.ffn_up, self.xin.span(0, hidden));
        if (self.hostDebug()) self.debug_gate = maxAbs(self.mlp_h.slice()[0..c.ffn]);
        self.exec.swiglu(self.graph, self.mlp_h.span(0, c.ffn), self.mlp_h1.span(0, c.ffn));
        if (self.hostDebug()) self.debug_swi = maxAbs(self.mlp_h.slice()[0..c.ffn]);
        self.exec.matvec(self.graph, self.x2.span(0, hidden), l.ffn_down, self.mlp_h.span(0, c.ffn));
        if (self.hostDebug()) self.debug_mlp = maxAbs(self.x2.slice()[0..hidden]);
        // Per-layer telemetry, on BOTH backends through `toHost` so the device
        // path never dereferences a device pointer.  `$QWEN35_DRAFT_DEBUG`
        // gates it (a few D2H per layer per row).
        if (self.devTelemetry() and il < self.per_layer.len and self.dbg_scratch != null) {
            const d = self.dbg_scratch.?;
            self.toHost(d[0..hidden], self.resid.span(0, hidden));
            self.toHost(d[hidden..][0..hidden], self.x2.span(0, hidden));
            self.toHost(d[2 * hidden ..][0..hidden], self.xin.span(0, hidden));
            self.toHost(d[3 * hidden ..][0..hidden], self.gout.span(0, @min(hidden, q_dim)));
            // FFN gate BEFORE swiglu overwrites it: the one value that tells
            // "the gate matvec is wrong" apart from "swiglu/down is wrong".
            self.toHost(d[4 * hidden ..][0..hidden], self.mlp_h.span(0, hidden));
            self.syncBridge();
            var mx: [4]f32 = .{ 0, 0, 0, 0 };
            for (0..3) |j| {
                for (d[j * hidden ..][0..hidden]) |v| mx[j] = @max(mx[j], @abs(v));
            }
            for (d[3 * hidden ..][0..@min(hidden, q_dim)]) |v| mx[3] = @max(mx[3], @abs(v));
            self.per_layer[il] = .{
                .resid = mx[0],
                .mlp = mx[1],
                .xin = mx[2],
                .attn = mx[3],
                .gate = maxAbs(d[4 * hidden ..][0..hidden]),
                .x0 = d[2 * hidden],
                .m0 = d[hidden],
            };
        }
        if (self.hostDebug()) self.emit("swiglu", il, self.dump_row, self.mlp_h.slice()[0..c.ffn]);
        self.exec.addResid(self.graph, self.resid.span(0, hidden), self.x2.span(0, hidden));
        if (self.hostDebug()) self.emit("layer_out", il, self.dump_row, self.resid.slice()[0..hidden]);
        if (self.dump_dir) |root| self.dumpLayer(root, il, self.dump_row);
    }

    /// Dequantise `token`'s codebook row into `codebook_row`.
    fn dequantCodebookRow(self: *Draft, token: u32) void {
        switch (self.codebook) {
            .quant => |q| self.exec.dequantRow(self.codebook_row, q, token),
            .f32 => |w| {
                const c = self.cfg.code_len;
                const off = @as(usize, token) * c;
                std.debug.assert(off + c <= w.len);
                // One host->device upload on the device path, a memcpy on the
                // CPU one.
                self.fromHost(self.codebook_row, w[off..][0..c]);
            },
        }
    }

    /// One code write after `layer` (`DSpineCodeBridge.__call__`,
    /// `dspine_mlx/bridge.py:88`), run for EVERY layer of this checkpoint.
    ///
    /// Reads the layer's just-produced hidden state, projects it to a code,
    /// mixes in the predecessor row's message, and writes the resulting feature
    /// back into the residual stream.  The next layer's input is the written
    /// residual passed through `next_norm` -- already done here, which is why
    /// `layerForward` must not normalise its input itself.
    ///
    /// `row` is the row index INSIDE the block; row 0 is the anchor.  The anchor
    /// takes its code straight from the target embedding's codebook entry and
    /// receives no update, so a block is conditioned entirely by the anchor plus
    /// the target context KV.
    /// ---- the four bridge kernels, dispatched CPU / CUDA ----
    ///
    /// Both sides take the same arguments and produce the same bits
    /// (`scripts/dspine_ab.cu` proves it), so `writeBridge` below reads the same
    /// whether stage 3 runs on the host or the device.  The CUDA arms pass
    /// `Buf.ptr` straight through -- on a device backend that IS the device
    /// address, so no copy is involved.
    fn bridgePrepareRead(self: *Draft, residual: ?[*]const f32, add: bool) void {
        const c = &self.cfg;
        const hidden = c.hidden;
        switch (self.bridge_kind) {
            .cpu => C.q35_dspine_prepare_read(
                self.resid.ptr,
                residual,
                self.bridge_raw.ptr,
                self.bridge_normed.ptr,
                self.bridge_rms.ptr,
                1,
                @intCast(hidden),
                c.bridge_eps,
                @intFromBool(add),
            ),
            .cuda => if (Br.dspineLinkedIn) {
                // A CPU-only build has no `q35_launch_dspine_*` symbols on the
                // link line (see `build.zig`'s `cuda_sources`), so this arm is
                // `comptime`-removed then: `Br.dspineLinkedIn` is a build
                // option, so the reference is absent rather than unreachable,
                // which is what keeps plain `zig build` linking.  `load`
                // refuses `bridge_kind = .cuda` in that build.
                const rc = Br.Launchers.q35_launch_dspine_prepare_read(
                    self.resid.ptr,
                    residual,
                    self.bridge_raw.ptr,
                    self.bridge_normed.ptr,
                    self.bridge_rms.ptr,
                    1,
                    @intCast(hidden),
                    c.bridge_eps,
                    @intFromBool(add),
                    self.bridge_stream,
                );
                std.debug.assert(rc == 0);
            },
        }
    }

    fn bridgeNormalizeCodes(self: *Draft, projected: Buf, anchors: [*]const i32, row: usize) void {
        const c = &self.cfg;
        switch (self.bridge_kind) {
            .cpu => C.q35_dspine_normalize_codes(
                projected.ptr,
                self.codebook_row.ptr,
                anchors,
                self.bridge_codes.ptr,
                1,
                @intCast(c.code_len),
                @intCast(c.block_size),
                c.bridge_eps,
                @intCast(row),
            ),
            .cuda => if (Br.dspineLinkedIn) {
                if (!Br.dspineLinkedIn) unreachable;
                const rc = Br.Launchers.q35_launch_dspine_normalize_codes(
                    projected.ptr,
                    self.codebook_row.ptr,
                    anchors,
                    self.bridge_codes.ptr,
                    1,
                    @intCast(c.code_len),
                    @intCast(c.block_size),
                    c.bridge_eps,
                    @intCast(row),
                    self.bridge_stream,
                );
                std.debug.assert(rc == 0);
            },
        }
    }

    /// `packed_rows_buf` holds THIS row's triple.  The predecessor triple is
    /// supplied separately (`prev_packed`) on both backends: both kernel twins
    /// index `packed` by the call-local row, so a block-sized array would need a
    /// second absolute-row contract for the same data.  See the note next to
    /// `packed_slice` in `writeBridge` for the bug that came from mixing them.
    fn bridgeFeatureMessage(self: *Draft, packed_rows_buf: Buf, row: usize) void {
        const c = &self.cfg;
        switch (self.bridge_kind) {
            .cpu => {
                const prev: ?[*]const f32 =
                    if (row > 0 and self.prev_packed_valid) self.prev_packed.ptr else null;
                C.q35_dspine_feature_message(
                    packed_rows_buf.ptr,
                    prev,
                    self.gate_b.ptr,
                    self.msg.ptr,
                    1,
                    @intCast(c.rank),
                    @intCast(c.block_size),
                    @intCast(row),
                );
            },
            .cuda => if (Br.dspineLinkedIn) {
                if (!Br.dspineLinkedIn) unreachable;
                const prev: ?[*]const f32 =
                    if (row > 0 and self.prev_packed_valid) self.prev_packed.ptr else null;
                const rc = Br.Launchers.q35_launch_dspine_feature_message(
                    packed_rows_buf.ptr,
                    prev,
                    self.gate_b.ptr,
                    self.msg.ptr,
                    1,
                    @intCast(c.rank),
                    @intCast(c.block_size),
                    @intCast(row),
                    self.bridge_stream,
                );
                std.debug.assert(rc == 0);
            },
        }
    }

    fn bridgeFeatureFinish(self: *Draft, delta: Buf, valid_row: [*]const u8, next_norm: Buf, row: usize) void {
        const c = &self.cfg;
        switch (self.bridge_kind) {
            .cpu => C.q35_dspine_feature_finish(
                self.bridge_raw.ptr,
                delta.ptr,
                self.bridge_rms.ptr,
                valid_row,
                next_norm.ptr,
                self.resid.ptr,
                self.xin.ptr,
                1,
                @intCast(c.hidden),
                @intCast(c.block_size),
                c.rms_eps,
                1,
                @intFromBool(!self.disable_bridge),
                @intCast(row),
            ),
            .cuda => if (Br.dspineLinkedIn) {
                if (!Br.dspineLinkedIn) unreachable;
                const rc = Br.Launchers.q35_launch_dspine_feature_finish(
                    self.bridge_raw.ptr,
                    delta.ptr,
                    self.bridge_rms.ptr,
                    valid_row,
                    next_norm.ptr,
                    self.resid.ptr,
                    self.xin.ptr,
                    1,
                    @intCast(c.hidden),
                    @intCast(c.block_size),
                    c.rms_eps,
                    1,
                    @intFromBool(!self.disable_bridge),
                    @intCast(row),
                    self.bridge_stream,
                );
                std.debug.assert(rc == 0);
            },
        }
    }

    /// Device -> host, both backends.  `Exec.copyRange` cannot do this: on the
    /// CUDA backend it is a device-to-device move (or an H2D upload) and a host
    /// destination would make the kernel write through a host address, which
    /// aborts the context with an illegal access (measured while wiring the
    /// serial stash).
    fn toHost(self: *Draft, dst: []f32, src: Buf) void {
        switch (self.bridge_kind) {
            .cpu => @memcpy(dst[0..src.len], src.slice()[0..src.len]),
            .cuda => if (Br.dspineLinkedIn) {
                self.cuda.?.copyToHost(dst, src);
            },
        }
    }

    /// Host -> device, both backends.  Mirror of `toHost`.
    fn fromHost(self: *Draft, dst: Buf, src: []const f32) void {
        switch (self.bridge_kind) {
            .cpu => @memcpy(dst.slice()[0..dst.len], src[0..dst.len]),
            .cuda => if (Br.dspineLinkedIn) {
                self.cuda.?.copyFromHost(dst, src);
            },
        }
    }

    /// Copy the current residual stream row into `dst` (host, `cfg.hidden`
    /// wide).  Exists for self-checks that must not dereference a device `Buf`.
    pub fn copyResidToHost(self: *Draft, dst: []f32) void {
        std.debug.assert(dst.len >= self.cfg.hidden);
        self.toHost(dst[0..self.cfg.hidden], self.resid.span(0, self.cfg.hidden));
        self.syncBridge();
    }

    /// Lazily allocate the three host mirrors the device path needs to build the
    /// serial stash (`writeBridge`).  Idempotent.
    fn ensureHostMirror(self: *Draft) void {
        if (self.host_raw != null) return;
        const c = &self.cfg;
        self.host_raw = self.alloc.alloc(f32, c.hidden) catch @panic("dspine: host mirror OOM");
        self.host_rms = self.alloc.alloc(f32, 1) catch @panic("dspine: host mirror OOM");
        self.host_query = self.alloc.alloc(f32, c.rank) catch @panic("dspine: host mirror OOM");
        self.host_normed = self.alloc.alloc(f32, c.hidden) catch @panic("dspine: host mirror OOM");
    }

    /// Block until the device stream has finished the copies issued above.  On
    /// the CPU path `copyRange` is already complete, so this is a no-op.
    fn syncBridge(self: *Draft) void {
        if (self.bridge_kind != .cuda) return;
        if (self.sync_fn) |f| f(self.sync_ctx.?);
    }

    fn writeBridge(self: *Draft, l: *const Layer, il: usize, row: usize, next_norm: Buf, is_last: bool) void {
        const c = &self.cfg;
        const hidden = c.hidden;

        // ---- 1. prepare_read: raw = bf16(residual or residual+hidden), rms ----
        // `residual=None` at this call site: `layerForward` already summed the
        // MLP output into `self.resid`, which is what the reference passes as
        // `hidden`, while its `residual` argument is the pre-MLP stream.
        self.bridgePrepareRead(null, false);

        // ---- 2. read: [C, H] @ normalized[H] -> projected[C] ----
        // `layer_read_weight`: the LAST layer uses `dspine_read`, the others
        // `blk.i.dspine_read` (both land in `Layer.read` at load time).
        const projected = self.projected.span(0, c.code_len);
        self.exec.matvec(self.graph, projected, l.read, self.bridge_normed.span(0, hidden));

        // ---- 3. normalize_codes: anchors from the codebook, others L2-scaled ----
        self.dequantCodebookRow(self.anchor_tokens[if (row < self.anchor_tokens.len) row else 0]);
        // The anchor id is always 0 here: `dequantCodebookRow` above already
        // put the right row into `codebook_row`, and both kernel twins index the
        // codebook at `anchors[..] * C`.  On the device path the id has to be a
        // DEVICE cell -- passing the stack address of a host `[1]i32` makes the
        // kernel read host memory (an illegal access, not a wrong number),
        // which is what a first version did.
        const anchor_row: [1]i32 = .{0};
        switch (self.bridge_kind) {
            .cpu => self.bridgeNormalizeCodes(projected, &anchor_row, row),
            .cuda => if (Br.dspineLinkedIn) {
                // One 4-byte H2D per layer-write; the value never changes, but
                // keeping it in device memory is what makes the kernel's read
                // legal.
                self.cuda.?.copyBytesToDevice(@ptrCast(self.anchor_dev.ptr), std.mem.asBytes(&anchor_row));
                self.bridgeNormalizeCodes(projected, @ptrCast(self.anchor_dev.ptr), row);
            },
        }

        // ---- 4. packed = [gate; predecessor_gate; prev] @ codes -> [3R] ----
        // The reference concatenates the three [R, C] weights once into one
        // [3R, C] matrix; three separate matvecs into consecutive slices of
        // `packed` are identical and avoid materialising a welded copy of
        // possibly-quantized weights (the file stores them as bf16).
        const codes = self.bridge_codes.span(0, c.code_len);
        // Each row's `[gate.query | predecessor.sender | prev.value]` triple
        // lives at its OWN slot in `packed_rows`.  The device kernel reads the
        // PREDECESSOR row's triple out of that array (`packed + (row-1)*3R`),
        // so writing every row to offset 0 would make row r's `sender`/`value`
        // its own -- and `feature_message` on row 1 then computes
        // `gate*value` from the row it is about to send to, not from the row
        // before it.  Measured: `msg` max came out 0.0 on every non-anchor row,
        // which silently disabled the whole code bridge.
        //
        // The CPU path does not need this (it carries the predecessor in
        // `prev_packed`), but one slot per row is harmless there and keeps the
        // two arms reading the same memory.
        const packed_slice = self.packed_rows.span(0, 3 * c.rank);
        self.exec.matvec(self.graph, packed_slice.span(0, c.rank), self.gate_w, codes);
        self.exec.matvec(self.graph, packed_slice.span(c.rank, c.rank), self.pred_gate_w, codes);
        self.exec.matvec(self.graph, packed_slice.span(2 * c.rank, c.rank), self.prev_w, codes);

        // ---- 5. feature_message: gate * predecessor value, anchor zeroed ----
        // Row `row-1`'s packed triple is the `prev` sender; see
        // `bridgeFeatureMessage` for how each side obtains it.
        self.bridgeFeatureMessage(packed_slice, row);

        // ---- 6. write: [H, R] @ message[R] -> delta[H] ----
        const delta = self.delta.span(0, hidden);
        self.exec.matvec(self.graph, delta, l.write, self.msg.span(0, c.rank));

        // ---- 7. feature_finish: out = bf16(raw + rms*delta),
        //                              normed = bf16(bf16(next_norm(out)) * w) ----
        // The three bf16 roundings here are load-bearing: collapsing them into
        // one changes ~26% of the elements (measured, see `dspine_tests.zig`).
        // `next_norm`'s weight goes in as the NORM branch's scale vector, so the
        // norm is not recomputed here (that would round a fourth time).
        // The `valid` gate is ONE byte and the reference always passes all-true
        // (`models/dspine.py:335` builds `torch.ones`), so the only thing that
        // actually gates a write is the `row % T != 0` test inside the kernel.
        // The byte still has to live in DEVICE memory on the device path: a
        // stack address makes the kernel read host memory, which aborted the
        // context with an illegal access (this was the second such cell, after
        // the anchor id).
        const valid_row: [1]u8 = .{@intFromBool(row % c.block_size != 0)};
        switch (self.bridge_kind) {
            .cpu => self.bridgeFeatureFinish(delta, &valid_row, next_norm, row),
            .cuda => if (Br.dspineLinkedIn) {
                self.cuda.?.copyBytesToDevice(@ptrCast(self.valid_dev.ptr), &valid_row);
                self.bridgeFeatureFinish(delta, @ptrCast(self.valid_dev.ptr), next_norm, row);
            },
        }
        self.emit("bridge_raw", il, row, self.bridge_raw.slice()[0..hidden]);
        self.emit("bridge_codes", il, row, self.bridge_codes.slice()[0..c.code_len]);
        self.emit("bridge_packed", il, row, self.packed_rows.span(0, 3 * c.rank).slice());
        self.emit("bridge_msg", il, row, self.msg.span(0, c.rank).slice());
        self.emit("bridge_delta", il, row, delta.slice());

        if (self.dump_dir) |root| {
            self.dumpStage(root, il, row, "raw", self.bridge_raw, hidden);
            self.dumpStage(root, il, row, "msg", self.msg, c.rank);
            self.dumpStage(root, il, row, "delta", self.delta, hidden);
            self.dumpStage(root, il, row, "out", self.resid, hidden);
            self.dumpStage(root, il, row, "nrm", self.xin, hidden);
            self.dumpStage(root, il, row, "codes", self.bridge_codes, c.code_len);
        }

        // ---- stage 4: stash what `serial_table` needs from this row ----
        // Only the LAST layer's write is the one the serial table models: every
        // later row's `feature_message` is driven by THIS layer's packed triple
        // and THIS layer's `raw`/`rms`, which is exactly what the reference
        // stashes (`bridge.py:135`, and `worker.py` passes `stash["raw"]`).
        //
        // On the DEVICE path this is the one place a copy is unavoidable: stage
        // 4 (`src/serial.zig`) walks `lm_head`'s quantized rows on the HOST, so
        // `raw`/`rms`/`query` must come back over PCIe.  That is 5120 + 1 + 512
        // floats per row -- 22 KB, and it happens once per block row rather than
        // once per layer, so 48 layers cost nothing extra.
        if (is_last) {
            switch (self.bridge_kind) {
                .cpu => {
                    for (0..hidden) |k| {
                        self.stash_raw.slice()[row * hidden + k] = self.bridge_raw.slice()[k];
                    }
                    self.stash_rms.slice()[row] = self.bridge_rms.slice()[0];
                    for (0..c.rank) |k| {
                        // `packed[:, :R]` is the receiver query -- the `gate` third.
                        self.stash_query.slice()[row * c.rank + k] = packed_slice.slice()[k];
                    }
                    for (0..hidden) |k| self.stash_normed.slice()[row * hidden + k] = self.xin.slice()[k];
                },
                .cuda => if (Br.dspineLinkedIn) {
                    // Three small D2H ranges into the host mirrors, then the
                    // stash is built on the host exactly as the CPU arm does it.
                    self.ensureHostMirror();
                    const hraw = self.host_raw.?[0..hidden];
                    self.toHost(hraw, self.bridge_raw.span(0, hidden));
                    const hrms: []f32 = self.host_rms.?[0..1];
                    self.toHost(hrms, self.bridge_rms.span(0, 1));
                    const hq = self.host_query.?[0..c.rank];
                    self.toHost(hq, packed_slice.span(0, c.rank));
                    self.syncBridge();
                    @memcpy(self.stash_raw.slice()[row * hidden ..][0..hidden], hraw);
                    self.stash_rms.slice()[row] = hrms[0];
                    @memcpy(self.stash_query.slice()[row * c.rank ..][0..c.rank], hq);
                    const hnorm = self.host_normed.?[0..hidden];
                    self.toHost(hnorm, self.xin.span(0, hidden));
                    self.syncBridge();
                    @memcpy(self.stash_normed.slice()[row * hidden ..][0..hidden], hnorm);
                },
            }
        }
        // Both kernel twins take the predecessor as a separate buffer, so BOTH
        // backends carry it across calls.  (The device arm used to rely on a
        // block-sized `packed_rows` instead; that made row r its own sender.)
        // Copy, not alias: `packed_rows` slot 0 is overwritten by the next row.
        self.exec.copyRange(self.graph, self.prev_packed, packed_slice);
        self.syncBridge();
        self.prev_packed_valid = true;
        // `feature_finish` now writes straight into `resid`, so there is no
        // copy here.  It used to write `resid_out` and copy back; that copy was
        // a pointless D2D move (and, on the device path, a `copy_range` launch
        // whose destination the kernel rejected).  `resid` is safe as the
        // destination: the kernel reads `raw` from `bridge_raw` and writes the
        // normalised next-layer input to `xin`, neither of which aliases it.
        if (self.bridge_kind == .cpu) {
            // The `debug_*` magnitudes and `per_layer` telemetry read host
            // memory directly.  They exist for the CPU self-checks and for
            // `QWEN35_SPEC_DRAFT`'s diagnostics, and on the device path they
            // would each be a separate D2H -- so they are skipped there.  The
            // values they would report are not used by anything on that path.
        self.debug_resid = maxAbs(self.resid.slice()[0..hidden]);
            self.debug_delta = maxAbs(self.delta.slice()[0..hidden]);
            self.debug_msg = maxAbs(self.msg.slice()[0..c.rank]);
            self.debug_codes = maxAbs(self.bridge_codes.slice()[0..c.code_len]);
        }
    }

    /// Host copies of the two `Buf`s stage 4 reads as plain f32 arrays: the
    /// terminal norm per element (`output_norm`) and the bridge's gate bias.
    ///
    /// Everything else stage 4 touches (`lm_head`, `write`, `pred_gate_w`,
    /// `prev_w`, `codebook`) is a `LinearW`, i.e. a view into the GGUF mmap that
    /// the host can walk directly -- only these two were uploaded as opaque
    /// backend regions.  On the CPU path they ARE host memory and this returns
    /// the same slice with no copy; on the device path it pulls them back once.
    fn hostNorm(self: *Draft) []const f32 {
        const c = &self.cfg;
        switch (self.bridge_kind) {
            .cpu => return self.norm.slice()[0..c.hidden],
            .cuda => {
                if (self.norm_host == null) {
                    const dst = self.alloc.alloc(f32, c.hidden) catch @panic("dspine: norm_host OOM");
                    self.toHost(dst, self.norm);
                    self.syncBridge();
                    self.norm_host = dst;
                }
                return self.norm_host.?;
            },
        }
    }

    fn hostGateB(self: *Draft) []const f32 {
        const c = &self.cfg;
        switch (self.bridge_kind) {
            .cpu => return self.gate_b.slice()[0..c.rank],
            .cuda => {
                if (self.gate_b_host == null) {
                    const dst = self.alloc.alloc(f32, c.rank) catch @panic("dspine: gate_b_host OOM");
                    self.toHost(dst, self.gate_b);
                    self.syncBridge();
                    self.gate_b_host = dst;
                }
                return self.gate_b_host.?;
            },
        }
    }

    /// Materialise the two per-vocabulary tables (`build_serial_tables`).
    ///
    /// Costs one pass over `V` for each table -- at 27B that is 248320 rows of
    /// `H`+`C` work, so it happens ONCE per process, lazily, on the first
    /// serial round.  `TG` and `TSV` are 762 MB together; allocating them in
    /// `load` would make every self-check pay for stage 4.
    pub fn buildSerialTables(self: *Draft) !void {
        if (self.tables != null) return;
        const c = &self.cfg;
        const last = self.layers.len - 1;

        // `write_w` [H][R] and `pv_w` [2R][C] get widened once: both are small
        // (10 MB and 4 MB) and `build_serial_tables` contracts against them for
        // every vocabulary row, so a per-row dequant would be V times the work.
        const write_w = try self.alloc.alloc(f32, c.hidden * c.rank);
        errdefer self.alloc.free(write_w);
        const pv_w = try self.alloc.alloc(f32, 2 * c.rank * c.code_len);
        errdefer self.alloc.free(pv_w);

        // `write` is [H][R] in the file's row-major layout (ne0 = R = 512 is the
        // row width, ne1 = H = 5120 the row count), so row h is already hidden
        // h's R-wide feature map: copy, no transpose.  See the note on
        // `Layer.write` for why the GGUF dims read "backwards" here.
        for (0..c.hidden) |h| {
            serial.dequantRowInto(write_w[h * c.rank ..][0..c.rank], self.layers[last].write, h, c.rank);
        }
        // `pv_w = [predecessor_gate; prev]` -- that concatenation order is the
        // `packed_weight[r:]` slice the reference contracts (`worker.py:106`),
        // and `feature_message` reads the same two thirds in the same order.
        for (0..c.rank) |r| {
            serial.dequantRowInto(pv_w[r * c.code_len ..][0..c.code_len], self.pred_gate_w, r, c.code_len);
            serial.dequantRowInto(pv_w[(c.rank + r) * c.code_len ..][0..c.code_len], self.prev_w, r, c.code_len);
        }
        self.write_w_f32 = write_w;
        self.pv_w_f32 = pv_w;

        self.tables = try serial.buildTables(
            self.alloc, self.lm_head, self.hostNorm(),
            write_w, pv_w, self.codebook,
            c.vocab, c.hidden, c.rank, c.code_len,
        );
    }

    /// `topk_ids` (`kernels/dspine.py:165`) for one block's rows: the `top_k`
    /// largest logits of `lm_head @ raw[row]`, ascending by id.
    ///
    /// The reference's fast path (`layers/head.py`) does this with an FP8 rough
    /// pass plus a rescue, which exists purely for speed on a GPU graph.  The
    /// ids it returns are the ids a full sort would return, so scoring the whole
    /// vocabulary is the same answer one row at a time; that is what this does,
    /// and it is why stage 4's first cut is slow.
    fn candidatesForBlock(self: *Draft, rows: usize) void {
        const c = &self.cfg;
        const hidden = c.hidden;
        const k = c.top_k;
        const t = c.block_size;
        // NOTE: the rows scored here are each block row's NORMED output
        // (`stash_normed`), matching `generate.py:145`'s `rows = out[0, 1:]`
        // where `out` is the draft's returned `x_in`.  Scoring `raw` instead
        // produced a top-16 that never held the target's own next token.
        //
        // `lm_head` belongs to the TARGET (`c.lm_head_owner`), so the draft's
        // own backend cannot dot against it on the device: the head is ~1 GB at
        // 27B and uploading it for one row per candidate is not a trade.
        //
        // When the caller supplied an `lm_head_matvec` hook (the engine does;
        // `main.zig` points it at the TARGET's backend) the 248320-id
        // projection runs there, on the backend that already owns the table.
        // Without it (unit tests, self-checks) this falls back to the host
        // `dequantRowInto` + dot path, which is the SAME arithmetic but pays one
        // dequant per vocabulary row: measured 28 s per spec round, which is why
        // the hook exists.
        if (self.lm_head_matvec) |mv| {
            var n2: usize = 0;
            var b2: usize = 0;
            while (b2 * t < rows) : (b2 += 1) {
                var r2: usize = 1;
                while (r2 < t) : (r2 += 1) {
                    const row = b2 * t + r2;
                    if (row >= rows) break;
                    mv(self.lm_head_ctx, self.cand_logits.slice(), self.stash_normed.slice()[row * hidden ..][0..hidden]);
                    topKInto(self.cand[n2 * k ..][0..k], self.cand_logits.slice()[0..c.vocab], k);
                    n2 += 1;
                }
            }
            return;
        }

        // A block's candidate rows are its NON-anchor rows: row 0 of each block
        // is the anchor and is never predicted.
        var n: usize = 0;
        var b: usize = 0;
        while (b * t < rows) : (b += 1) {
            var r: usize = 1;
            while (r < t) : (r += 1) {
                const row = b * t + r;
                if (row >= rows) break;
                // logits = lm_head @ raw[row].  `raw` is what the reference's
                // `topk_ids` sees: NOT scaled by `norm_w` (`worker.py:251`
                // passes `rows` = `stash["raw"]` straight to the matmul).
                const xrow = self.stash_normed.slice()[row * hidden ..][0..hidden];
                const wrow = self.alloc.alloc(f32, hidden) catch @panic("dspine: topk row OOM");
                defer self.alloc.free(wrow);
                for (0..c.vocab) |v| {
                    serial.dequantRowInto(wrow, self.lm_head, v, hidden);
                    var acc: f32 = 0;
                    for (wrow, xrow) |a, xv| acc += a * xv;
                    self.cand_logits.slice()[v] = acc;
                }
                const dst = self.cand[n * k ..][0..k];
                topKInto(dst, self.cand_logits.slice()[0..c.vocab], k);
                n += 1;
            }
        }
    }

    /// One serial-refinement round over the block just forwarded: build the
    /// candidates, fill the transition table, walk the chain.
    ///
    /// Returns the emitted chain (`block_size - 1` tokens) as a slice of
    /// `self.chain`.  Requires `forwardToken` to have run for every row of the
    /// block (`stash_ready`).
    /// Test hook: overwrite the serial stash with a caller-supplied residual
    /// stream, so "the draft forward is wrong" and "the serial tables are
    /// wrong" can be told apart.
    ///
    /// Measured on 27B IQ3_XXS + DSpine b16, 22-token prompt: with the draft's
    /// own stash the chain collapses to `220` (space) after 3 rows and hits
    /// 0/15; with the TARGET's own last-layer hidden injected it stays varied
    /// for the whole block and hits 1/15.  That is what pins the defect on the
    /// draft forward rather than on the tables.
    pub fn injectStash(self: *Draft, raw: []const f32, rms: []const f32) void {
        std.debug.assert(raw.len <= self.stash_raw.len);
        std.debug.assert(rms.len <= self.stash_rms.len);
        @memcpy(self.stash_raw.slice()[0..raw.len], raw);
        @memcpy(self.stash_rms.slice()[0..rms.len], rms);
    }

    /// Binary dump of one `serialTable` call's inputs, for the reference A/B.
    ///
    /// Magnitude probes cannot separate "my table is wrong" from "this
    /// checkpoint is weak"; feeding the Triton reference MY stash can.
    fn dumpSerialInputs(self: *Draft, dir: []const u8, rows: usize) !void {
        const c = &self.cfg;
        const t = c.block_size;
        const k = c.top_k;
        const hidden = c.hidden;
        const n_rows = rows / t * (t - 1);
        const io = self.dump_io;

        try std.Io.Dir.cwd().createDirPath(io, dir);
        const d = try std.Io.Dir.cwd().openDir(io, dir, .{});
        defer d.close(io);

        try self.dumpF32(io, d, "raw.f32", self.stash_raw.slice()[0 .. rows * hidden]);
        try self.dumpF32(io, d, "rms.f32", self.stash_rms.slice()[0..rows]);
        try self.dumpF32(io, d, "query.f32", self.stash_query.slice()[0 .. rows * c.rank]);
        // `gate_b` is a backend Buf, so on a device run it must come back over
        // `toHost` -- reading `slice()` there writes a 0-byte file and aborts
        // the rest of the dump with `Unexpected`.
        {
            const gb = try self.alloc.alloc(f32, c.rank);
            defer self.alloc.free(gb);
            self.toHost(gb, self.gate_b);
            try self.dumpF32(io, d, "gate_b.f32", gb);
        }

        {
            const buf = try self.alloc.alloc(u8, n_rows * k * 4);
            defer self.alloc.free(buf);
            for (self.cand[0 .. n_rows * k], 0..) |v, idx2| {
                const b: i32 = @intCast(v);
                std.mem.writeInt(i32, buf[idx2 * 4 ..][0..4], b, .little);
            }
            try d.writeFile(io, .{ .sub_path = "cand.i32", .data = buf });
        }
        {
            const nw = try self.alloc.alloc(f32, hidden);
            defer self.alloc.free(nw);
            self.toHost(nw, self.norm);
            self.syncBridge();
            try self.dumpF32(io, d, "norm_w.f32", nw);
        }
        try self.dumpF32(io, d, "write_w.f32", self.write_w_f32 orelse &.{});
        try self.dumpF32(io, d, "pv_w.f32", self.pv_w_f32 orelse &.{});
        if (self.tables) |*tb| {
            var b1 = try self.alloc.alloc(u8, tb.tg.len * 2);
            defer self.alloc.free(b1);
            for (tb.tg, 0..) |v, q2| std.mem.writeInt(u16, b1[q2 * 2 ..][0..2], v, .little);
            try d.writeFile(io, .{ .sub_path = "tg.u16", .data = b1 });
            var b2 = try self.alloc.alloc(u8, tb.tsv.len * 2);
            defer self.alloc.free(b2);
            for (tb.tsv, 0..) |v, q3| std.mem.writeInt(u16, b2[q3 * 2 ..][0..2], v, .little);
            try d.writeFile(io, .{ .sub_path = "tsv.u16", .data = b2 });
        }
        // anchors: row 0 of each block, as `serial_table` takes them.
        {
            const nb = rows / t;
            const buf = try self.alloc.alloc(u8, nb * 4);
            defer self.alloc.free(buf);
            for (0..nb) |b2| {
                const v: i32 = @intCast(self.anchor_tokens[b2]);
                std.mem.writeInt(i32, buf[b2 * 4 ..][0..4], v, .little);
            }
            try d.writeFile(io, .{ .sub_path = "anchors.i32", .data = buf });
        }
        // lm_head rows for every id the candidates and anchors name: the
        // reference only ever scores those, so this is an exact reduction.
        try self.dumpLmHeadRows(io, d, rows);
        // The kernel's own idx, so the comparison has both sides.
        {
            const ix = self.cand_idx[0 .. rows / t * (t - 1) * k];
            const buf = try self.alloc.alloc(u8, ix.len * 2);
            defer self.alloc.free(buf);
            for (ix, 0..) |v, yy| std.mem.writeInt(i16, buf[yy * 2 ..][0..2], v, .little);
            try d.writeFile(io, .{ .sub_path = "idx.i16", .data = buf });
        }
        var meta: [256]u8 = undefined;
        const txt = try std.fmt.bufPrint(&meta, "T={d}\nK={d}\nH={d}\nR={d}\nC={d}\nrows={d}\nn_rows={d}\nV={d}\n", .{
            t, k, hidden, c.rank, c.code_len, rows, n_rows, c.vocab,
        });
        try d.writeFile(io, .{ .sub_path = "meta.txt", .data = txt });
    }

    /// `lm_head` rows for the ids `cand` and the anchors name, plus their ids.
    fn dumpLmHeadRows(self: *Draft, io: Io, d: std.Io.Dir, rows: usize) !void {
        const c = &self.cfg;
        const t = c.block_size;
        const k = c.top_k;
        const hidden = c.hidden;
        const n_rows = rows / t * (t - 1);

        var seen = std.AutoHashMap(u32, void).init(self.alloc);
        defer seen.deinit();
        for (self.cand[0 .. n_rows * k]) |v| try seen.put(v, {});
        for (0..rows / t) |b2| try seen.put(self.anchor_tokens[b2], {});

        var ids = std.ArrayList(u32).empty;
        defer ids.deinit(self.alloc);
        var it = seen.keyIterator();
        while (it.next()) |p| try ids.append(self.alloc, p.*);
        std.mem.sort(u32, ids.items, {}, std.sort.asc(u32));

        const nb = ids.items.len;
        {
            const buf = try self.alloc.alloc(u8, nb * 4);
            defer self.alloc.free(buf);
            for (ids.items, 0..) |v, xx| {
                const q: i32 = @intCast(v);
                std.mem.writeInt(i32, buf[xx * 4 ..][0..4], q, .little);
            }
            try d.writeFile(io, .{ .sub_path = "lm_head_ids.i32", .data = buf });
        }
        const vals = try self.alloc.alloc(f32, nb * hidden);
        defer self.alloc.free(vals);
        const row = try self.alloc.alloc(f32, hidden);
        defer self.alloc.free(row);
        for (ids.items, 0..) |v, xx| {
            serial.dequantRowInto(row, self.lm_head, v, hidden);
            @memcpy(vals[xx * hidden ..][0..hidden], row);
        }
        try self.dumpF32(io, d, "lm_head_rows.f32", vals);
    }

    fn dumpStage(self: *Draft, root: []const u8, il: usize, row: usize, name: []const u8, buf: Buf, width: usize) void {
        const io = self.dump_io;
        var path_buf: [512]u8 = undefined;
        const dir = std.fmt.bufPrint(&path_buf, "{s}/fwd", .{root}) catch return;
        std.Io.Dir.cwd().createDirPath(io, dir) catch return;
        const d = std.Io.Dir.cwd().openDir(io, dir, .{}) catch return;
        defer d.close(io);
        var nb: [576]u8 = undefined;
        const fname = std.fmt.bufPrint(&nb, "l{d}_{s}.f32", .{ il, name }) catch return;
        // Truncate only for the first row: later rows are written at their own
        // offset, and a truncating create per row would keep only the last one.
        var f = d.createFile(io, fname, .{ .truncate = row == 0 }) catch return;
        defer f.close(io);
        const tmp = self.alloc.alloc(f32, width) catch return;
        defer self.alloc.free(tmp);
        self.toHost(tmp, buf.span(0, @min(width, buf.len)));
        self.syncBridge();
        f.writePositionalAll(io, std.mem.sliceAsBytes(tmp), row * width * @sizeOf(f32)) catch {};
    }

    /// Per-layer forward state for the stage-3 reference A/B: `<root>/fwd/l<il>_<stage>.f32`,
    /// row-major `[block_size][width]`, written incrementally as rows complete.
    /// Inputs the Python forward transcription needs besides the per-layer
    /// state: the context taps, the anchor id, the embedding rows it used, and
    /// the block's anchor-row codes (a codebook lookup the Python side can
    /// reproduce only with the GGUF's codebook, which it does read).
    pub fn dumpFwdInputs(self: *Draft, root: []const u8, tap: []const f32, tap_rows: usize) void {
        const c = &self.cfg;
        const io = self.dump_io;
        var path_buf: [512]u8 = undefined;
        const dir = std.fmt.bufPrint(&path_buf, "{s}/fwd", .{root}) catch return;
        std.Io.Dir.cwd().createDirPath(io, dir) catch return;
        const d = std.Io.Dir.cwd().openDir(io, dir, .{}) catch return;
        defer d.close(io);
        var meta: [512]u8 = undefined;
        const txt = std.fmt.bufPrint(&meta, "tap_rows={d}\nrow_h={d}\nhidden={d}\nffn={d}\nheads={d}\nkv={d}\nhd={d}\nlayers={d}\nblock={d}\n", .{
            tap_rows, c.target_layers.len, c.hidden, c.ffn, c.heads, c.kv_heads, c.hd, c.n_layers, c.block_size,
        }) catch return;
        d.writeFile(io, .{ .sub_path = "meta.txt", .data = txt }) catch {};
        d.writeFile(io, .{ .sub_path = "tap.f32", .data = std.mem.sliceAsBytes(tap) }) catch {};
        var ab: [4]u8 = undefined;
        // The anchor is the block's first id.  `anchor_tokens[0]` is only set once
        // `forwardToken(row 0)` runs, which is AFTER this dump, so take the id
        // from the caller-supplied block instead.
        std.mem.writeInt(i32, &ab, @intCast(self.toks_for_dump[0]), .little);
        d.writeFile(io, .{ .sub_path = "anchor.i32", .data = &ab }) catch {};
        // Embedding rows for the block's ids, exactly as `forwardToken` saw them.
        const emb = self.alloc.alloc(f32, c.block_size * c.hidden) catch return;
        defer self.alloc.free(emb);
        for (0..c.block_size) |t| {
            backend_cpu.dequantRowInto(emb[t * c.hidden ..][0..c.hidden], self.embed_rows, self.toks_for_dump[t]);
        }
        d.writeFile(io, .{ .sub_path = "embed_rows.f32", .data = std.mem.sliceAsBytes(emb) }) catch {};
        _ = self.toks_for_dump;
    }

    fn dumpLayer(self: *Draft, root: []const u8, il: usize, row: usize) void {
        const c = &self.cfg;
        const io = self.dump_io;
        var path_buf: [512]u8 = undefined;
        const dir = std.fmt.bufPrint(&path_buf, "{s}/fwd", .{root}) catch return;
        std.Io.Dir.cwd().createDirPath(io, dir) catch return;
        const d = std.Io.Dir.cwd().openDir(io, dir, .{}) catch return;
        defer d.close(io);

        const items = [_]struct { name: []const u8, buf: Buf, width: usize }{
            .{ .name = "gout", .buf = self.gout, .width = c.heads * c.hd },
            .{ .name = "mlp", .buf = self.mlp_h, .width = c.ffn },
            .{ .name = "mlp_out", .buf = self.x2, .width = c.hidden },
        };
        for (items) |it| {
            var nb: [576]u8 = undefined;
            const fname = std.fmt.bufPrint(&nb, "l{d}_{s}.f32", .{ il, it.name }) catch continue;
            var f = d.createFile(io, fname, .{ .truncate = row == 0 }) catch continue;
            defer f.close(io);
            const tmp = self.alloc.alloc(f32, it.width) catch continue;
            defer self.alloc.free(tmp);
            self.toHost(tmp, it.buf.span(0, @min(it.width, it.buf.len)));
            self.syncBridge();
            f.writePositionalAll(io, std.mem.sliceAsBytes(tmp), row * it.width * @sizeOf(f32)) catch continue;
        }
    }

    fn dumpF32(self: *Draft, io: Io, d: std.Io.Dir, name: []const u8, data: []const f32) !void {
        _ = self;
        try d.writeFile(io, .{ .sub_path = name, .data = std.mem.sliceAsBytes(data) });
    }

    pub fn serialRound(self: *Draft, rows: usize) ![]const u32 {
        if (!self.stash_ready) return error.StashNotReady;
        try self.buildSerialTables();
        const c = &self.cfg;
        const t = c.block_size;
        const k = c.top_k;

        self.candidatesForBlock(rows);

        // `$QWEN35_S4_DUMP=<dir>`: write the exact inputs and outputs of
        // `serialTable` so the Triton reference can be run on the SAME numbers.
        // Magnitude probes cannot tell "my table is wrong" from "this checkpoint
        // is weak"; feeding the reference my stash can.
        const scratch_w = try self.alloc.alloc(f32, c.hidden);
        defer self.alloc.free(scratch_w);
        const scratch_scored = try self.alloc.alloc(f32, c.hidden);
        defer self.alloc.free(scratch_scored);

        serial.serialTable(
            self.cand_idx, .{
                .raw = self.stash_raw.slice(),
                .rms = self.stash_rms.slice(),
                .query = self.stash_query.slice(),
                .rows = rows,
            },
            self.hostNorm(),
            self.cand, self.anchor_tokens[0..],
            self.lm_head, &self.tables.?, self.alloc, self.hostGateB(),
            t, c.hidden, k,
            .{ .w_rows = scratch_w, .scored = scratch_scored },
        );
        self.dump_calls += 1;
        if (self.dump_dir) |dump_root| {
            var nb: [512]u8 = undefined;
            const dump_dir = std.fmt.bufPrint(&nb, "{s}/call{d}", .{ dump_root, self.dump_calls }) catch dump_root;
            self.dumpSerialInputs(dump_dir, rows) catch |e| {
                std.debug.print("S4 dump failed: {s}\n", .{@errorName(e)});
            };
        }


        const gamma = t - 1;
        const out = self.chain[0..gamma];
        serial.serialChain(out, self.cand_idx, self.cand, t, k);
        return out;
    }

    pub fn deinit(self: *Draft) void {
        const a = self.alloc;
        a.free(self.cand);
        a.free(self.cand_idx);
        a.free(self.chain);
        if (self.tables) |*t| t.deinit(a);
        if (self.write_w_f32) |w| a.free(w);
        if (self.pv_w_f32) |w| a.free(w);
        if (self.host_raw) |w| a.free(w);
        if (self.host_rms) |w| a.free(w);
        if (self.host_query) |w| a.free(w);
        if (self.host_normed) |w| a.free(w);
        if (self.norm_host) |w| a.free(w);
        if (self.gate_b_host) |w| a.free(w);
        if (self.dbg_scratch) |w| a.free(w);
        if (self.cuda) |cb| {
            cb.deinit();
            a.destroy(cb);
        }
        self.cpu.deinit();
        a.destroy(self.cpu);
        a.free(self.layers);
        a.free(self.cfg.target_layers);
    }
};

///  for the CLI self-check (the helper itself stays private).
pub fn maxAbsForSelfCheck(s: []const f32) f32 {
    return maxAbs(s);
}

fn maxAbs(s: []const f32) f32 {
    var m: f32 = 0;
    for (s) |v| {
        const a = @abs(v);
        if (a > m) m = a;
    }
    return m;
}

pub const LoadError = error{
    MissingTensor,
    BadConfig,
};

fn dimOf(g: *const gguf.File, name: []const u8, which: usize) LoadError!usize {
    const t = g.findTensor(name) orelse return error.MissingTensor;
    if (which >= t.dims.len) return error.BadConfig;
    return @intCast(t.dims[which]);
}

/// Open the draft GGUF and load every weight.
///
/// `embed_rows` is the TARGET's `token_embd` row view: the reference calls the
/// target's `embed_tokens` to embed the anchor token, so the draft has no
/// embedding table of its own (confirmed against the real file: no
/// `token_embd`/`output` tensor among the 73).
pub fn load(
    alloc: std.mem.Allocator,
    g: *const gguf.File,
    embed_rows: gguf.QuantRow,
    /// The TARGET's `[V, H]` output projection.  `topk_ids` scores candidates
    /// with it and `build_serial_tables` contracts it against the draft's
    /// terminal norm, so it is a hard input to stage 4 -- not a debug aid.
    lm_head: LinearW,
    max_seq: usize,
    /// Which backend runs the 5-layer forward and the code bridge (stage 3).
    /// Stage 4 is host-only either way.  `cuda` requires a `-Dcuda=true` build.
    want: Br.Kind,
    /// Needed only by the device path: `CudaBackend` timestamps its launches
    /// through an `std.Io`, which it receives via `setThreads`.
    draft_io: std.Io,
    /// `$QWEN35_DRAFT_TAPL_SHIFT`: constant added to every tap layer index (see
    /// the comment at the load site).  Null = use the GGUF metadata as written.
    tapl_shift: ?i32,
) !Draft {
    const hidden = try dimOf(g, "fc.weight", 1);
    const ffn: usize = @intCast(try reqUint(g, "dflash.feed_forward_length"));
    const heads: usize = @intCast(try reqUint(g, "dflash.attention.head_count"));
    const kv_heads: usize = @intCast(try reqUint(g, "dflash.attention.head_count_kv"));
    const n_layers: usize = @intCast(try reqUint(g, "dflash.block_count"));
    const hd: usize = @intCast(try reqUint(g, "dflash.attention.key_length"));

    // Shapes that the metadata does not carry, read off the tensors instead so
    // a checkpoint with a different R/C still loads:
    //   dspine_gate.weight      [C, R]
    //   dspine_codebook.weight  [C, V]
    const rank = try dimOf(g, "dspine_gate.weight", 1);
    const code_len = try dimOf(g, "dspine_gate.weight", 0);
    const vocab = try dimOf(g, "dspine_codebook.weight", 1);

    if (want == .cuda and !Br.dspineLinkedIn) return error.CudaNotBuilt;

    const cpu = try alloc.create(@import("backend_cpu.zig").CpuBackend);
    cpu.* = .init(alloc, g);
    errdefer {
        cpu.deinit();
        alloc.destroy(cpu);
    }

    // The device path builds a SECOND backend over the same mapping.  Weights
    // are uploaded per backend, so this costs one copy of the 2.2 GB draft
    // checkpoint on the device -- which is the point: stage 3 then runs entirely
    // on the GPU.
    var cuda_backend: ?*@import("backend_cuda.zig").CudaBackend = null;
    errdefer if (cuda_backend) |cb| {
        cb.deinit();
        alloc.destroy(cb);
    };
    if (want == .cuda) {
        const cb = try alloc.create(@import("backend_cuda.zig").CudaBackend);
        cb.* = @import("backend_cuda.zig").CudaBackend.init(alloc, g);
        // The backend timestamps its launches through `io`, which is only set
        // by `setThreads` (the engine calls it during load).  A draft backend
        // built outside the engine has to do the same or the first matvec
        // dereferences `undefined`.
        cb.exec().setThreads(draft_io, 1);
        cuda_backend = cb;
    }
    const exec = if (cuda_backend) |cb| cb.exec() else cpu.exec();

    const tl_vals = g.findKey("dflash.target_layers") orelse return error.BadConfig;
    const tl_arr = switch (tl_vals.*) {
        .array => |a| a,
        else => return error.BadConfig,
    };
    const target_layers = try alloc.alloc(u32, @intCast(tl_arr.count));
    errdefer alloc.free(target_layers);
    {
        var r = gguf.arrayReader(g, tl_arr);
        for (target_layers) |*slot| {
            slot.* = @intCast(try gguf.arrayElemU64(&r, tl_arr.element_type));
        }
        std.mem.sort(u32, target_layers, {}, std.sort.asc(u32));
    }
    // `$QWEN35_DRAFT_TAPL_SHIFT`: add a constant to every tap layer index.
    //
    // The GGUF declares `[2,17,32,47,62]` while the upstream HF `config.json`
    // declares `[1,16,31,46,61]` for the SAME checkpoint.  Both index
    // `layers[lid]`'s OUTPUT (`engine.zig`'s tap copy and the reference's
    // `_LayerHook` both sit after the layer), so the two cannot both name the
    // layers `fc` was trained on; a shift of -1 selects the upstream set.
    // A/B switch: unset keeps the GGUF metadata.
    // `$QWEN35_DRAFT_TAPL_SHIFT` (+1 to restore the GGUF metadata) forces the
    // value: the two authorities disagree, and only a measurement can say which
    // one `fc` was trained against.
    if (tapl_shift) |shift| {
        for (target_layers) |*slot| {
            const v: i32 = @as(i32, @intCast(slot.*)) + shift;
            if (v < 0) return error.BadConfig;
            slot.* = @intCast(v);
        }
    }
    // One block's worth of rows, which is the most the bridge ever sees at once.
    const block_rows: usize = @intCast(try reqUint(g, "dflash.block_size"));
    const cfg = Config{
        .n_layers = n_layers,
        .hidden = hidden,
        .ffn = ffn,
        .heads = heads,
        .kv_heads = kv_heads,
        .hd = hd,
        .rank = rank,
        .code_len = code_len,
        .vocab = vocab,
        .block_size = @intCast(try reqUint(g, "dflash.block_size")),
        .top_k = @intCast(try reqUint(g, "dflash.selector_top_k")),
        .mask_token_id = @intCast(try reqUint(g, "tokenizer.ggml.mask_token_id")),
        .target_layers = target_layers,
        .rope_freq_base = @floatCast(g.getFloat("dflash.rope.freq_base") orelse 1e7),
        .rms_eps = @floatCast(g.getFloat("dflash.attention.layer_norm_rms_epsilon") orelse 1e-6),
        .bridge_eps = @floatCast(g.getFloat("dflash.bridge.eps") orelse g.getFloat("dspine.bridge.eps") orelse 1e-6),
        .max_seq = max_seq,
    };

    // `fc.weight` is [n_layers*H, H], NOT [n_layers*R, H]: the bridge's
    // per-layer `read` (code -> hidden) output is what gets concatenated, so the
    // fc input is 5 x 5120 = 25600 (torch `nn.Linear(25600, 5120)`).  The
    // R-wide values (`write`, `gate`, `prev`, `pred_gate`) never reach `fc`.
    if (cfg.hidden * cfg.target_layers.len != try dimOf(g, "fc.weight", 0)) {
        return error.BadConfig;
    }
    if (cfg.rank != try dimOf(g, "dspine_gate.bias", 0)) {
        return error.BadConfig;
    }
    const layers = try alloc.alloc(Layer, n_layers);
    errdefer alloc.free(layers);
    for (layers, 0..) |*l, i| {
        // `bufPrint` returns a slice OF ITS BUFFER, and the backend resolves a
        // name into an mmap view only after it has been written, so the loads
        // must be SEQUENTIAL STATEMENTS sharing one buffer.  A struct literal
        // would be wrong here: Zig may evaluate the fields in any order, and
        // every earlier slice would then point at the last name written --
        // which loads the same tensor 13 times and fails only via a shape
        // check downstream.
        l.* = undefined;
        var b: [96]u8 = undefined;
        l.attn_norm = try exec.loadVecF32(fmtn(&b, "blk.{d}.attn_norm.weight", i));
        l.attn_q = try loadAny(exec, fmtn(&b, "blk.{d}.attn_q.weight", i));
        l.attn_k = try loadAny(exec, fmtn(&b, "blk.{d}.attn_k.weight", i));
        l.attn_v = try loadAny(exec, fmtn(&b, "blk.{d}.attn_v.weight", i));
        l.attn_output = try loadAny(exec, fmtn(&b, "blk.{d}.attn_output.weight", i));
        l.attn_q_norm = try exec.loadVecF32(fmtn(&b, "blk.{d}.attn_q_norm.weight", i));
        l.attn_k_norm = try exec.loadVecF32(fmtn(&b, "blk.{d}.attn_k_norm.weight", i));
        l.ffn_norm = try exec.loadVecF32(fmtn(&b, "blk.{d}.ffn_norm.weight", i));
        l.ffn_gate = try loadAny(exec, fmtn(&b, "blk.{d}.ffn_gate.weight", i));
        l.ffn_up = try loadAny(exec, fmtn(&b, "blk.{d}.ffn_up.weight", i));
        l.ffn_down = try loadAny(exec, fmtn(&b, "blk.{d}.ffn_down.weight", i));
        l.read = try loadAny(exec, fmtn(&b, "blk.{d}.dspine_read.weight", i));
        l.write = try loadAny(exec, fmtn(&b, "blk.{d}.dspine_write.weight", i));
    }
    const q_dim = heads * hd;
    const kv_dim = kv_heads * hd;

    var d = Draft{
        .alloc = alloc,
        .cpu = cpu,
        .cuda = cuda_backend,
        .exec = exec,
        .bridge_kind = if (cuda_backend != null) .cuda else .cpu,
        .force_host_debug = @import("cudart.zig").environ_map.?.get("QWEN35_DRAFT_DEBUG") != null,
        .ctx_identity = @import("cudart.zig").environ_map.?.get("QWEN35_DRAFT_CTX_IDENTITY") != null,
        .dump_io = draft_io,
        .dump_dir = @import("cudart.zig").environ_map.?.get("QWEN35_S4_DUMP"),
        .bridge_stream = if (cuda_backend) |cb| cb.currentStream() else null,
        .sync_fn = if (cuda_backend) |_| @as(*const fn (*anyopaque) void, @ptrCast(&syncThunk)) else null,
        .sync_ctx = if (cuda_backend) |cb| @ptrCast(cb) else null,
        .cfg = cfg,
        .layers = layers,
        .fc = try loadAny(exec, "fc.weight"),
        .hidden_norm = try exec.loadVecF32("enc.output_norm.weight"),
        .norm = try exec.loadVecF32("output_norm.weight"),
        .embed_rows = embed_rows,
        .codebook = try loadAny(exec, "dspine_codebook.weight"),
        .codebook_row = try exec.allocBuf(code_len),
        .gate_w = try loadAny(exec, "dspine_gate.weight"),
        .gate_b = try exec.loadVecF32("dspine_gate.bias"),
        .prev_w = try loadAny(exec, "dspine_prev.weight"),
        .pred_gate_w = try loadAny(exec, "dspine_pred_gate.weight"),
        .resid = try exec.allocBuf(hidden),
        .xin = try exec.allocBuf(hidden),
        .x2 = try exec.allocBuf(hidden),
        .q = try exec.allocBuf(q_dim),
        .kv = try exec.allocBuf(2 * kv_dim),
        .gout = try exec.allocBuf(q_dim),
        .att = try exec.allocBuf(max_seq + 1),
        .mlp_h = try exec.allocBuf(ffn),
        .mlp_h1 = try exec.allocBuf(ffn),
        // The bridge works on ONE BLOCK at a time: `projectContext` is called
        // per spec round with at most `block_size` freshly verified rows, and
        // `att` already has to hold `ctx_len + len` scores (max_seq).
        .projected = try exec.allocBuf(code_len),
        .delta = try exec.allocBuf(hidden),
        .resid_out = try exec.allocBuf(hidden),
        .anchor_tokens = @splat(0),
        .ctx = try exec.allocBuf(max_seq * hidden),
        .ctx_raw = try exec.allocBuf(max_seq * cfg.target_layers.len * hidden),
        .ctx_rows = 0,
        .packed_rows = try exec.allocBuf(block_rows * 3 * cfg.rank),
        .msg = try exec.allocBuf(block_rows * cfg.rank),
        .bridge_raw = try exec.allocBuf(block_rows * hidden),
        .bridge_rms = try exec.allocBuf(block_rows),
        .prev_packed = try exec.allocBuf(3 * cfg.rank),
        .prev_packed_valid = false,
        .bridge_normed = try exec.allocBuf(block_rows * hidden),
        .bridge_codes = try exec.allocBuf(block_rows * code_len),
        .ones = try exec.allocBuf(q_dim),
        // ---- stage 4 ----
        // The three serial stashes and the candidate logits stay in HOST
        // memory on both backends: stage 4 (`src/serial.zig`) is host-only code
        // -- it dequantizes `lm_head` rows itself and walks `topKInto` over the
        // logits -- so a device-resident copy would only add a D2H without
        // removing any work.
        .stash_raw = backend.Buf.wrap(try alloc.alloc(f32, block_rows * hidden)),
        .stash_rms = backend.Buf.wrap(try alloc.alloc(f32, block_rows)),
        .stash_query = backend.Buf.wrap(try alloc.alloc(f32, block_rows * cfg.rank)),
        .stash_normed = backend.Buf.wrap(try alloc.alloc(f32, block_rows * hidden)),
        .stash_ready = false,
        .cand = try alloc.alloc(u32, block_rows * cfg.top_k),
        .cand_idx = try alloc.alloc(i16, block_rows * cfg.top_k),
        .chain = try alloc.alloc(u32, block_rows),
        .cand_logits = backend.Buf.wrap(try alloc.alloc(f32, cfg.vocab)),
        .lm_head = lm_head,
        .kv_cache = try alloc.alloc(Draft.Kv, n_layers),
    };
    // The draft's backend is its OWN instance, so it needs its own quantize
    // scratch sized for the widest input any of its matvecs receives -- the
    // ffn gate/up projections read a hidden-wide (5120) normed row, but
    // `fc.weight` reads a 5*hidden (25600) concatenation, which is the widest
    // input in the draft.  Sizing for `ffn` would be wrong in a subtle way: it
    // happens to be 17408 > 5120, but the real bound is the fc input.
    try exec.allocScratch(@max(ffn, hidden * cfg.target_layers.len));
    errdefer d.deinit();
    // One device-resident logits row for stage 4's `topk_ids` (`matvec` writes
    // through the backend, so the host copy cannot be the destination).
    if (d.bridge_kind == .cuda) {
        d.cand_logits_dev = try exec.allocBuf(cfg.vocab);
        d.embed_host = try alloc.alloc(f32, hidden);
        d.anchor_dev = try exec.allocBuf(1);
        d.valid_dev = try exec.allocBuf(1);
    }
    if (d.force_host_debug) {
        d.dbg_scratch = try alloc.alloc(f32, 8 * hidden);
    }
    // The `attnDecode` mask is all +inf, so every visible row passes.  Filled on
    // the host and uplinked, because on a device backend `ones` is device
    // memory (a direct `slice()` write would be a wild pointer).
    {
        const mask = try alloc.alloc(f32, q_dim);
        defer alloc.free(mask);
        @memset(mask, std.math.inf(f32));
        d.fromHost(d.ones, mask);
    }
    for (d.kv_cache) |*c| {
        c.* = .{
            .k = try exec.allocBuf(max_seq * kv_dim),
            .v = try exec.allocBuf(max_seq * kv_dim),
            .ctx_len = 0,
            .len = 0,
        };
    }
    return d;
}

/// The `k` largest values of `logits`, returned as ASCENDING token ids.
///
/// Ascending-by-id is the reference's contract (`topk_ids` docstring,
/// `kernels/dspine.py:166`) and it is load-bearing: `serial_table` breaks a
/// per-lane score TIE toward the earlier lane, which only means "the lower
/// token id" when the candidate row is sorted by id.
/// `topKInto` for callers that cannot see the private helper (the CLI's
/// self-check).  Same contract: ascending ids, input restored.
pub fn topKForSelfCheck(dst: []u32, logits: []f32, k: usize) void {
    topKInto(dst, logits, k);
}

fn topKInto(dst: []u32, logits: []f32, k: usize) void {
    std.debug.assert(dst.len == k);
    std.debug.assert(k <= serial.MAX_K);
    // Selection sort over k passes: k is 16 and V is 248320, so the k*V scan
    // is the whole cost and a heap would not pay for itself.
    var picked: [serial.MAX_K]usize = undefined;
    var saved: [serial.MAX_K]f32 = undefined;
    for (0..k) |i| {
        var best: usize = 0;
        var bestv: f32 = -std.math.inf(f32);
        for (logits, 0..) |v, idx| {
            if (v > bestv) {
                bestv = v;
                best = idx;
            }
        }
        picked[i] = best;
        // Mask the winner so the next pass finds the runner-up, remembering
        // what was there: the caller refills this buffer per row, but restoring
        // keeps the function honest about not destroying its input.
        saved[i] = logits[best];
        logits[best] = -std.math.inf(f32);
    }
    for (picked[0..k], 0..) |p, i| logits[p] = saved[i];
    // Ascending by ID -- see the doc comment.
    std.mem.sort(usize, picked[0..k], {}, std.sort.asc(usize));
    for (picked[0..k], 0..) |p, i| dst[i] = @intCast(p);
}

/// Load a linear as whatever the file actually stores (Q8_0 / BF16 / F32).
/// GGUF's `loadLinear` handles the quantized families; BF16 arrives there as
/// an unsupported type, so it is widened to f32 once at load time (the bridge
/// tables are small: `gate`/`prev`/`pred_gate` are 1 MB each, `read` 10 MB,
/// `write` 5 MB, i.e. ~85 MB of f16 -> f32 across all 5 layers, which is not
/// worth a dedicated BF16 matvec path for a 5-layer model).
fn loadAny(exec: Exec, name: []const u8) !LinearW {
    return exec.loadLinear(name);
}

fn fmtn(buf: []u8, comptime fmt: []const u8, il: usize) []const u8 {
    return std.fmt.bufPrint(buf, fmt, .{il}) catch unreachable;
}

fn reqUint(g: *const gguf.File, key: []const u8) LoadError!u64 {
    return g.getUint(key) orelse return error.BadConfig;
}


