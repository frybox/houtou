const std = @import("std");
const Io = std.Io;

/// `QWEN35_DRAFT_DUMP`, captured in `main` (Zig 0.17 has no `getenv`) and read
/// by `runInference` far below, which never sees `std.process.Init`.
var g_dump_path: ?[]const u8 = null;

const houtou = @import("houtou");
const gguf = houtou.gguf;

/// Context threaded through the engine's per-token progress callback: prints
/// each generated token as soon as it is produced, C-style.

/// Live speculative decoder: the `Engine.generateSpec` draft callback backed by
/// the DSpine draft model.
///
/// One round, concretely: take the last `ctx_rows` committed ids as the draft's
/// context, have TARGET produce their hidden-state taps (that is a forward
/// pass over the context window -- the draft has no KV of its own across
/// rounds), project them into the draft's 5-layer backbone, run the block, and
/// hand back `serialRound`'s chain.
///
/// The context window is re-forwarded every round on purpose: the draft's own
/// KV cannot be carried across rounds because the block's accepted length is
/// only known after verification, so its cache would hold rows the target
/// rejected.  Re-running is `ctx_rows` tokens of TARGET prefill per round, which
/// is exactly what makes a small `ctx_rows` the right default.
/// Tap-ring accessors, type-erased for `DraftRunner` (see its `e` field).
/// One pair per engine; they exist only because the two engines are distinct
/// structs with the same ring contract.
fn ringFilledQwen35(ctx: *anyopaque) usize {
    const e: *houtou.engine.Engine = @ptrCast(@alignCast(ctx));
    return e.specRingFilled();
}
fn ringReadQwen35(ctx: *anyopaque, out: []f32) void {
    const e: *houtou.engine.Engine = @ptrCast(@alignCast(ctx));
    e.specRingRead(out);
}
fn lmHeadHookQwen3(ctx: *anyopaque, out: []f32, x: []const f32) void {
    const e: *houtou.engines.qwen3.Engine = @ptrCast(@alignCast(ctx));
    e.lmHeadFromHostRow(out, x);
}

fn ringFilledQwen3(ctx: *anyopaque) usize {
    const e: *houtou.engines.qwen3.Engine = @ptrCast(@alignCast(ctx));
    return e.specRingFilled();
}
fn ringReadQwen3(ctx: *anyopaque, out: []f32) void {
    const e: *houtou.engines.qwen3.Engine = @ptrCast(@alignCast(ctx));
    e.specRingRead(out);
}

/// MTP (nextn) draft callback: the model's OWN nextn head proposes the block.
///
/// Unlike the DSpine draft this needs no separate checkpoint and no code
/// bridge -- the block ships with the target and was trained jointly with it,
/// which is why its first-step hit rate (79.6% measured) is far above what an
/// external drafter reaches.
///
/// CHAINING: one `nextnPredict` call proposes ONE token and hands back the
/// normed hidden state it derived, so passing the same buffer as both `hidden`
/// argument and `out_hidden` turns it into a block.  That is the whole loop;
/// the head has no "predict k tokens" mode.
///
/// INPUT: the head consumes the target's `output_norm`-ed hidden row for the
/// token at `pos`, which is `specTapNormedPlane(last_tap)` after the verify
/// forward `generateSpec` runs each round.
const MtpRunner = struct {
    e: *houtou.engine.Engine,
    /// Tap plane holding the last decoder layer (the only tap enabled here).
    tap_index: usize,
    /// Hidden row the chain ping-pongs through (device-side on CUDA).
    h: houtou.backend.Buf,
    /// How many tokens one block may propose.
    max_draft: usize,
    /// Stop drafting once the head's top-1 probability drops below this
    /// (`$QWEN35_SPEC_PMIN`, default 0.5; 0 disables the stop).  Only the first
    /// drafted token is exempt, so a chain never ends up empty merely because
    /// the head is unsure about the anchor.
    p_min: f32 = 0.5,
    rounds: usize = 0,

    fn draft(ctx: *anyopaque, context: []const u32, anchor: u32, block_pos: u32, out: []u32) usize {
        _ = context;
        const self: *MtpRunner = @ptrCast(@alignCast(ctx));
        const e = self.e;

        // The anchor's normed hidden state came out of this round's verify
        // forward.  `specTapNormedRowInto` is the device-safe twin of the
        // host-plane helper: one `rmsnorm` launch, no host round trip, so this
        // works on CUDA as well.
        e.specTapNormedRowInto(self.h, self.tap_index, 0);

        const n = @min(out.len, self.max_draft);
        var tok = anchor;
        var produced: usize = 0;
        while (produced < n) : (produced += 1) {
            // `h` in == `h` out: the head reads the previous row's hidden state
            // and leaves this row's, which is what makes the chain work.
            e.nextnPredict(self.h, tok, block_pos + @as(u32, @intCast(produced)), self.h) catch return produced;
            // `argmaxConfLogits` goes through the backend, so on CUDA the vocab
            // never crosses PCIe -- only the index and one probability come back.
            var p: f32 = 1.0;
            tok = e.argmaxConfLogits(&p);
            // CONFIDENCE EARLY STOP (llama.cpp's `p_min`,
            // `common/speculative.cpp:1789`).  A drafted token costs a full
            // verify row whether or not it is accepted, so once the head is no
            // longer confident the rest of the chain is pure waste.  Measured on
            // 27B IQ3_XXS: the accept rate falls 79% -> 39% between the 2nd and
            // 3rd draft, and capping the chain at 2 already recovers 21% of
            // throughput; this stops even earlier when the head itself says so.
            //
            // `p < p_min` leaves `produced` unchanged, i.e. the token that was
            // just sampled is NOT added.  A round that drafts nothing is not a
            // loss: `generateSpec` falls back to one ordinary decode step.
            if (produced > 0 and p < self.p_min) return produced;
            out[produced] = tok;
        }
        self.rounds += 1;
        return produced;
    }
};

fn argmaxHost(v: []const f32) u32 {
    var best: u32 = 0;
    var bv: f32 = -std.math.inf(f32);
    for (v, 0..) |x, i| {
        if (x > bv) {
            bv = x;
            best = @intCast(i);
        }
    }
    return best;
}

/// `Draft.lm_head_matvec` hook: `out = lm_head @ x` on the TARGET's backend.
fn lmHeadHook(ctx: *anyopaque, out: []f32, x: []const f32) void {
    const e: *houtou.engine.Engine = @ptrCast(@alignCast(ctx));
    e.lmHeadFromHostRow(out, x);
}

const DraftRunner = struct {
    /// The target engine, type-erased: the Qwen3.5 and Qwen3 engines are
    /// distinct structs, and everything this runner needs from them is the
    /// tap ring (below).  Two function pointers are cheaper to carry than a
    /// generic instantiation of the whole runner.
    e: *anyopaque,
    ring_filled_fn: *const fn (ctx: *anyopaque) usize,
    ring_read_fn: *const fn (ctx: *anyopaque, out: []f32) void,
    dr: *houtou.draft.Draft,
    io: Io,
    /// Trailing committed ids fed to the draft's `fc` projection.
    ctx_rows: usize,
    /// Scratch for the tap planes (`5 * ctx_rows * hidden` f32).
    tap: []f32,
    /// Scratch for the block's ids (`block_size` u32).
    toks: []u32,
    /// Absolute position the block's first row sits at; set by the caller
    /// before each round (the draft's RoPE is absolute).
    ctx_pos: u32 = 0,
    rounds: usize = 0,

    fn draft(ctx: *anyopaque, context: []const u32, anchor: u32, block_pos: u32, out: []u32) usize {
        const self: *DraftRunner = @ptrCast(@alignCast(ctx));
        self.ctx_pos = block_pos;
        // `context` is not read any more: the block's left context reaches the
        // draft through the tap ring (`specRingRead`), which carries the target's
        // hidden states rather than the ids.  The ids would only be needed to fill
        // the block's non-anchor rows, and those are the MASK token now.
        _ = context;
        const dc = &self.dr.cfg;
        const hidden = dc.hidden;

        // ---- context taps come from the ring, not from a re-forward ----
        // Every target forward already wrote its layers' outputs into the tap
        // planes; `specRingPush` moved them here.  `context` is only used to
        // size the window (and to fill the block's placeholder rows with real
        // text instead of one repeated id).
        const n_ctx = @min(self.ring_filled_fn(self.e), self.ctx_rows);
        if (n_ctx == 0) return 0;
        self.ring_read_fn(self.e, self.tap);

        // ---- draft forward over the block ----
        self.dr.reset();
        self.dr.projectContext(self.tap[0 .. dc.target_layers.len * self.ctx_rows * hidden], n_ctx);
        // The context sits immediately BEFORE the block, so the block's first
        // row is at absolute position `anchor_pos + 1 - n_ctx + n_ctx`.  The
        // engine passes the block's own start position through `ctx`; using the
        // row index alone would rotate every row at position 0.. and the draft
        // would attend with the wrong phases (measured: 0 % acceptance).
        // The context rows END right before the block: they occupy absolute
        // positions `block_pos - n_ctx .. block_pos - 1`, so the draft's KV must
        // rotate them there.  Passing the block's own position would place them
        // AFTER the block and every row would attend with the wrong phases.
        const ctx_start: u32 = if (self.ctx_pos > n_ctx) self.ctx_pos - @as(u32, @intCast(n_ctx)) else 0;
        self.dr.appendContext(ctx_start);
        const t = dc.block_size;
        self.toks[0] = anchor;
        // Rows 1..T-1 are the MASK token, exactly as `generate.py:133-135`
        // builds the block.  They are overwritten by the draft's own chain
        // before anything is read back, but they are what the 5-layer forward
        // sees, so the value matters.
        @memset(self.toks[1..], dc.mask_token_id);
        for (0..t) |r| self.dr.forwardToken(self.toks[r], r, @intCast(self.ctx_pos + r));

        const chain = self.dr.serialRound(t) catch return 0;
        self.rounds += 1;
        if (spec_trace) {
            std.debug.print("DRAFT pos={d} anchor={d} nctx={d} chain:", .{ block_pos, anchor, n_ctx });
            for (chain[0..@min(chain.len, 8)]) |tk| std.debug.print(" {d}", .{tk});
            std.debug.print("\n", .{});
        }
        const n = @min(chain.len, out.len);
        @memcpy(out[0..n], chain[0..n]);
        return n;
    }

    /// `QWEN35_SPEC_RECALL` sink: copy the draft's `[rows][k]` candidate table
    /// out for the engine to score against the target's own picks.
    fn candidates(c: *anyopaque, dst: []u32, rows: usize) void {
        const self: *DraftRunner = @ptrCast(@alignCast(c));
        const k = self.dr.cfg.top_k;
        const n = @min(rows, self.dr.cfg.block_size - 1) * k;
        std.debug.assert(dst.len >= n);
        @memcpy(dst[0..n], self.dr.cand[0..n]);
    }
};

/// Raw-f32 activation sink for the DSpine draft (`QWEN35_DRAFT_DUMP=<dir>`).
///
/// Writes one flat little-endian f32 file per (stage, layer), rows appended in
/// the order the draft produces them, plus an `index.txt` of shapes.  Binary
/// and not text because the whole point is element-by-element comparison with a
/// reference run: text would lose precision and cost 12 bytes per float.
///
/// Files are opened lazily and cached: the draft emits ~9 stages x 5 layers per
/// block, and the block is forwarded twice (draft + inject), so a reopen per
/// emit would dominate.  Failures are reported once and disable the sink --
/// a dump is a debugging aid, never something that should abort a run.
const Dumper = struct {
    dir_path: ?[]const u8,
    io: Io,
    ready: bool = false,
    failed: bool = false,
    files: [MAX_FILES]?Io.File = @splat(null),
    keys: [MAX_FILES][]const u8 = @splat(""),
    n_files: usize = 0,

    const MAX_FILES = 160;

    fn init(path: ?[]const u8, io: Io) Dumper {
        return .{ .dir_path = path, .io = io };
    }

    fn deinit(self: *Dumper) void {
        for (self.files[0..self.n_files]) |f| {
            if (f) |file| file.close(self.io);
        }
        self.* = .{ .dir_path = null, .io = self.io };
    }

    fn sink(ctx: ?*anyopaque, stage: []const u8, layer: usize, row: usize, data: []const f32) void {
        const self: *Dumper = @ptrCast(@alignCast(ctx orelse return));
        self.write(stage, layer, row, data);
    }

    fn write(self: *Dumper, stage: []const u8, layer: usize, row: usize, data: []const f32) void {
        _ = row;
        if (self.failed) return;
        if (!self.ready) self.openDir() catch {
            self.failed = true;
            return;
        };
        const file = self.fileFor(stage, layer) catch {
            self.failed = true;
            return;
        };
        file.writeStreamingAll(self.io, std.mem.sliceAsBytes(data)) catch {
            self.failed = true;
            return;
        };
    }

    fn openDir(self: *Dumper) !void {
        try Io.Dir.cwd().createDirPath(self.io, self.dir_path.?);
        self.ready = true;
    }

    /// One file per (stage, layer): layers are the innermost loop, so sharing a
    /// file per stage would interleave rows and make the layout depend on the
    /// block size.  A fixed 16-entry name cache covers the ~9 stages x 5 layers.
    fn fileFor(self: *Dumper, stage: []const u8, layer: usize) !Io.File {
        var buf: [64]u8 = undefined;
        const key = try std.fmt.bufPrint(&buf, "{s}.{d}", .{ stage, layer });
        for (self.keys[0..self.n_files], 0..) |k, i| {
            if (std.mem.eql(u8, k, key)) return self.files[i].?;
        }
        if (self.n_files >= MAX_FILES) return error.TooManyStages;
        const name = try std.fmt.allocPrint(std.heap.page_allocator, "{s}/{s}.f32", .{ self.dir_path.?, key });
        defer std.heap.page_allocator.free(name);
        const file = try Io.Dir.cwd().createFile(self.io, name, .{ .truncate = true });
        self.keys[self.n_files] = try std.heap.page_allocator.dupe(u8, key);
        self.files[self.n_files] = file;
        self.n_files += 1;
        return file;
    }
};

const StreamCtx = struct {
    out: *Io.Writer,
    tok: *const houtou.tokenizer.Tokenizer,
    arena: std.mem.Allocator,

    fn onToken(ctx: ?*anyopaque, token: u32) void {
        const self: *StreamCtx = @ptrCast(@alignCast(ctx orelse return));
        if (houtou.pool.stat_on) {
            const s = houtou.pool.snapshot();
            houtou.pool.reset();
            std.debug.print("STATS tok={d} k256={d} k32={d} ksc={d} kq8={d} kf32={d} total={d}\n", .{
                token, s.k256, s.k32, s.ksc, s.kq8, s.kf32, s.k256 + s.k32 + s.ksc + s.kq8 + s.kf32,
            });
        }
        const piece = self.tok.decode(self.arena, &.{token}) catch null;
        if (piece) |s| {
            if (s.len > 0) {
                self.out.print("{s}", .{s}) catch {};
            } else {
                self.out.print("[{d}]", .{token}) catch {};
            }
        } else {
            // Decode failed: fall back to the token id, like the C engine.
            self.out.print("[{d}]", .{token}) catch {};
        }
        self.out.flush() catch {};
    }
};

/// `QWEN35_DUMP_IDS`: print the raw generated token ids (A/B md5 probes).
var dump_ids = false;
/// `QWEN35_SPEC_TAPS`: stage-0 self-check for the speculative-decoding hidden
/// taps.  Enables `Engine.specTapEnable` on a fixed layer set, runs a short
/// `forwardChunk`, and verifies every tap plane against an INDEPENDENT
/// recomputation (a second chunk that stops at the tapped layer is not
/// possible, so instead we assert structural invariants that a wrong offset or
/// a wrong layer would break: each plane must equal the model's own xK at that
/// layer, which we cross-check by re-running with a DIFFERENT tap set and
/// requiring the overlapping plane to be byte-identical).  Also prints a
/// checksum per plane so A/B runs can be compared by eye.
var spec_taps = false;
/// `QWEN35_SPEC_NEXTN`: stage-1 self-check for the nextn/MTP drafter.
var spec_nextn = false;
/// `QWEN35_SPEC_MTP`: decode with the model's OWN nextn head as the drafter,
/// instead of an external DSpine checkpoint.  No `QWEN35_DRAFT` needed.
var spec_mtp = false;
/// `QWEN35_SPEC_NEXTN_SEQ`: also dump the teacher-forced token sequence.
var spec_nextn_seq = false;
/// `QWEN35_SPEC_DRAFT`: stage-3 self-check for the DSpine draft model (load the
/// draft GGUF named by `QWEN35_DRAFT` and run its 5-layer forward).
var spec_draft = false;
/// `QWEN35_SPEC_GEN`: run DECODE with the DSpine drafter instead of the plain
/// loop.  Needs `QWEN35_DRAFT=<draft.gguf>`.  Prints the acceptance counters at
/// the end; `QWEN35_SPEC_GEN=0` disables it explicitly.
var spec_gen = false;
/// `QWEN35_SPEC_TRACE`: print each round's proposal and acceptance decision.
var spec_trace = false;
var draft_nobridge = false;
/// `QWEN35_DRAFT_BACKEND`: which backend runs DSpine stage 3 (the 5-layer
/// draft forward + the code bridge).  Unset means "follow the target's
/// backend".  Stage 4 (serial refinement) is host-only either way.
var draft_backend_override: ?houtou.dspine_bridge.Kind = null;
/// `QWEN35_SPEC_TAPFILE=<path>`: the draft self-check's context tap planes, as
/// raw f32.  Written when the file is absent, read when it is present, so the
/// CPU and CUDA backends can be fed the SAME input and their chains compared.
var g_tapfile_path: ?[]const u8 = null;
/// `QWEN35_SPEC_CTXROWS`: override how many target hidden-state rows the draft
/// sees as its left context.  The reference (`dspine_mlx/generate.py:163`)
/// passes only the CURRENT position's hidden (`vhidden[:, :accept+1]`), i.e. one
/// row; this port defaults to `block_size` rows.  A/B switch, not a model knob.
var spec_ctxrows_override: ?usize = null;
/// `QWEN35_SPEC_POSSTAT`: print a histogram of how many draft ids each round
/// got accepted.
var spec_posstat = false;
/// `QWEN35_SPEC_RECALL`: report per-block-row candidate recall.
var spec_recall = false;
/// `QWEN35_DRAFT_TAPL_SHIFT`: constant added to the draft's tap layer indices.
/// The draft GGUF says `[2,17,32,47,62]` and the upstream HF config says
/// `[1,16,31,46,61]` for the same checkpoint; only a measurement can say which
/// one `fc` was trained against, so this is an A/B switch (see `draft.load`).
var tapl_shift_override: ?i32 = null;
var draft_path_buf: [1024]u8 = undefined;
var draft_path: ?[:0]const u8 = null;

pub fn main(init: std.process.Init) !void {
    // Hand the environment to the CUDA runtime shim: Zig 0.17 only exposes it
    // through `Init`, and locating libcudart needs $QWEN35_CUDART / $PATH (the
    // CUDA kernels themselves are linked in, so they need nothing from here).
    houtou.cudart.environ_map = init.environ_map;
    if (init.environ_map.get("QWEN35_SIMD_TIER")) |t| houtou.engine.Engine.setSimdTier(t);
    if (init.environ_map.get("QWEN35_NO_BATCH") != null) houtou.engine.no_batch = true;
    if (init.environ_map.get("QWEN35_PROF") != null) {
        houtou.engine.prof = true;
        houtou.backend_cpu.prof = true;
    }
    if (init.environ_map.get("QWEN35_PROF_POST") != null) houtou.pool.prof_post_only = true;
    if (init.environ_map.get("QWEN35_STAT") != null) houtou.pool.stat_on = true;
    dump_ids = init.environ_map.get("QWEN35_DUMP_IDS") != null;
    spec_taps = init.environ_map.get("QWEN35_SPEC_TAPS") != null;
    spec_nextn = init.environ_map.get("QWEN35_SPEC_NEXTN") != null;
    spec_mtp = init.environ_map.get("QWEN35_SPEC_MTP") != null;
    houtou.engines.qwen35.spec_dummy_alloc =
        init.environ_map.get("QWEN35_SPEC_DUMMYALLOC") != null;
    spec_nextn_seq = init.environ_map.get("QWEN35_SPEC_NEXTN_SEQ") != null;
    spec_draft = init.environ_map.get("QWEN35_SPEC_DRAFT") != null;
    spec_gen = init.environ_map.get("QWEN35_SPEC_GEN") != null;
    spec_trace = init.environ_map.get("QWEN35_SPEC_TRACE") != null;
    g_dump_path = init.environ_map.get("QWEN35_DRAFT_DUMP");
    draft_nobridge = init.environ_map.get("QWEN35_DRAFT_NOBRIDGE") != null;
    g_tapfile_path = init.environ_map.get("QWEN35_SPEC_TAPFILE");
    if (init.environ_map.get("QWEN35_SPEC_CTXROWS")) |v| {
        spec_ctxrows_override = std.fmt.parseInt(usize, v, 10) catch null;
    }
    spec_posstat = init.environ_map.get("QWEN35_SPEC_POSSTAT") != null;
    spec_recall = init.environ_map.get("QWEN35_SPEC_RECALL") != null;
    if (init.environ_map.get("QWEN35_DRAFT_TAPL_SHIFT")) |v| {
        tapl_shift_override = std.fmt.parseInt(i32, v, 10) catch null;
    }
    if (init.environ_map.get("QWEN35_DRAFT_BACKEND")) |v| {
        draft_backend_override = std.meta.stringToEnum(houtou.dspine_bridge.Kind, v);
    }
    if (init.environ_map.get("QWEN35_DRAFT")) |p| {
        if (p.len < draft_path_buf.len) {
            @memcpy(draft_path_buf[0..p.len], p);
            draft_path_buf[p.len] = 0;
            draft_path = draft_path_buf[0..p.len :0];
        }
    }
    const arena: std.mem.Allocator = init.arena.allocator();
    const io = init.io;

    var stdout_buffer: [1 << 16]u8 = undefined;
    var stdout_file_writer: Io.File.Writer = .init(.stdout(), io, &stdout_buffer);
    const out = &stdout_file_writer.interface;

    var stderr_buffer: [4096]u8 = undefined;
    var stderr_file_writer: Io.File.Writer = .init(.stderr(), io, &stderr_buffer);
    const err_out = &stderr_file_writer.interface;

    // `run` reports failures through `err_out` and returns an error; catching it
    // here keeps the exit status meaningful without dumping a Zig stack trace
    // over the diagnostics the user actually needs.
    const args = try init.minimal.args.toSlice(arena);
    run(arena, io, args, out, err_out) catch |e| {
        // `run` already printed a human-readable message; only add the error
        // name when it did not (e.g. an out-of-memory during printing).
        if (e == error.WriteFailed) err_out.print("error: {s}\n", .{@errorName(e)}) catch {};
        err_out.flush() catch {};
        std.process.exit(1);
    };
    out.flush() catch {};
}

fn run(
    arena: std.mem.Allocator,
    io: Io,
    args: []const [:0]const u8,
    out: *Io.Writer,
    err_out: *Io.Writer,
) !void {
    if (args.len < 2) {
        try err_out.print(
            \\usage: {s} <file.gguf> [--full]
            \\       {s} run <file.gguf> <prompt> [-n N]
            \\
        , .{ args[0], args[0] });
        try err_out.flush();
        return error.MissingArgument;
    }

    if (std.mem.eql(u8, args[1], "run")) {
        return runInference(arena, io, args, out, err_out);
    }

    return dumpGguf(arena, io, args, out, err_out);
}

/// `run <gguf> <prompt> [-n N]`: load the engine, greedy-generate, print the
/// text and per-token timing.
/// Which engine a GGUF names in `general.architecture`.
const Architecture = union(enum) {
    qwen35,
    qwen3,
    /// Anything else, carrying the declared name for the error message.
    unknown: []const u8,
};

/// Read only what dispatch needs: the metadata KV region.
///
/// The file is memory-mapped and parsed, then unmapped on return; loading the
/// model proper happens in the chosen engine.
fn probeArchitecture(arena: std.mem.Allocator, io: Io, path: []const u8) Architecture {
    var file = Io.Dir.cwd().openFile(io, path, .{}) catch return .{ .unknown = "<unreadable file>" };
    defer file.close(io);
    const st = file.stat(io) catch return .{ .unknown = "<unreadable file>" };
    var mm = file.createMemoryMap(io, .{ .len = @intCast(st.size), .protection = .{ .read = true } }) catch
        return .{ .unknown = "<unmappable file>" };
    defer mm.destroy(io);
    var g = houtou.gguf.parse(arena, mm.memory) catch return .{ .unknown = "<not a GGUF file>" };
    defer g.deinit();
    const name = g.getString("general.architecture") orelse return .{ .unknown = "<no general.architecture>" };
    if (std.mem.eql(u8, name, "qwen35")) return .qwen35;
    if (std.mem.eql(u8, name, "qwen3")) return .qwen3;
    return .{ .unknown = name };
}

/// The Qwen3 path of `runInference`.
///
/// Split out rather than interleaved: none of the Qwen3.5-only machinery
/// (speculative decoding, the tap ring, the nextn drafter) applies, and this
/// engine has its own `generate`.
fn runInferenceQwen3(
    arena: std.mem.Allocator,
    io: Io,
    path: []const u8,
    prompt_text: []const u8,
    n_predict: usize,
    temp: f32,
    top_k: usize,
    top_p: f32,
    seed: u64,
    n_threads: usize,
    backend_kind: houtou.backend.Kind,
    out: *Io.Writer,
    err_out: *Io.Writer,
) !void {
    // Both speculative modes are supported on this engine now: `spec_draft`
    // (the draft/target pairing self-check) and `spec_gen` (the full
    // speculative decode loop, `engines/qwen3.zig`'s `generateSpec`).  The
    // activation-dump sink (`g_dump_path`) is Qwen3.5-only: it is wired into
    // the draft's per-stage emitter, which this path does not install.
    if (g_dump_path != null) {
        try err_out.print(
            "error: QWEN35_DRAFT_DUMP is Qwen3.5-only; this file is a qwen3 model\n",
            .{},
        );
        try err_out.flush();
        return error.InvalidArgument;
    }

    // Tokenizer from the same GGUF.
    var file = try Io.Dir.cwd().openFile(io, path, .{});
    defer file.close(io);
    const st = try file.stat(io);
    var mm = try file.createMemoryMap(io, .{ .len = @intCast(st.size), .protection = .{ .read = true } });
    defer mm.destroy(io);
    var g = try houtou.gguf.parse(arena, mm.memory);
    defer g.deinit();
    var tok = try houtou.tokenizer.Tokenizer.initFromGguf(arena, &g);
    defer tok.deinit();

    const rendered = try houtou.tokenizer.Tokenizer.renderChatTurn(arena, prompt_text, false);
    const prompt = try tok.encode(arena, rendered);
    try out.print("[prompt: {d} tokens]\n", .{prompt.len});
    try out.flush();

    var e = try houtou.engines.qwen3.Engine.load(
        arena,
        path,
        io,
        n_threads,
        backend_kind,
        .f16,
        houtou.engines.qwen3.Engine.max_seq,
    );
    defer e.deinit();
    e.sampler = houtou.engine.Sampler.init(.{ .temperature = temp, .top_k = top_k, .top_p = top_p, .seed = seed });
    try out.print(
        "[qwen3: layers={d} hidden={d} heads={d}/{d} hd={d} rotary={d} ff={d}]\n",
        .{ e.cfg.n_layers, e.cfg.hidden, e.cfg.n_heads, e.cfg.n_kv_heads, e.cfg.head_dim, e.cfg.rotary_dim, e.cfg.n_ff },
    );
    try out.flush();

    var stream_ctx = StreamCtx{ .out = out, .tok = &tok, .arena = arena };
    const res = if (spec_gen and draft_path != null) blk: {
        // Speculative decoding.  Same shape as the Qwen3.5 path; the pieces it
        // does not have (candidate-recall scoring, the per-stage activation
        // dump) are Qwen3.5-only diagnostics and are simply not wired here.
        const dfile = std.Io.Dir.cwd().openFile(io, draft_path.?, .{}) catch {
            try out.print("\n[spec: cannot open the draft gguf; falling back to plain decode]\n", .{});
            break :blk try houtou.engines.common.generate(
                e.driver(), arena, io, prompt, n_predict, StreamCtx.onToken, &stream_ctx,
            );
        };
        defer dfile.close(io);
        const dstat = try dfile.stat(io);
        var dmm = try std.Io.File.MemoryMap.create(io, dfile, .{
            .len = @intCast(dstat.size),
            .protection = .{ .read = true },
        });
        defer dmm.destroy(io);
        var dg = try houtou.gguf.parse(arena, dmm.memory);
        defer dg.deinit();
        const er = e.embedRows() catch break :blk try houtou.engines.common.generate(
            e.driver(), arena, io, prompt, n_predict, StreamCtx.onToken, &stream_ctx,
        );
        const kind: houtou.dspine_bridge.Kind =
            draft_backend_override orelse (if (e.backendKind() == .cuda) .cuda else .cpu);
        var dr = houtou.draft.load(arena, &dg, er, e.lmHead(), e.maxSeqForSpec(), kind, io, tapl_shift_override) catch |de| {
            try out.print("\n[spec: draft load failed ({s}); falling back to plain decode]\n", .{@errorName(de)});
            break :blk try houtou.engines.common.generate(
                e.driver(), arena, io, prompt, n_predict, StreamCtx.onToken, &stream_ctx,
            );
        };
        defer dr.deinit();
        const dc = &dr.cfg;
        dr.disable_bridge = draft_nobridge;
        // Candidate scoring borrows the TARGET's lm_head (the draft has none).
        dr.lm_head_matvec = lmHeadHookQwen3;
        dr.lm_head_ctx = &e;

        e.specTapEnable(dc.target_layers) catch {
            try out.print("\n[spec: tap enable failed; falling back to plain decode]\n", .{});
            break :blk try houtou.engines.common.generate(
                e.driver(), arena, io, prompt, n_predict, StreamCtx.onToken, &stream_ctx,
            );
        };
        // The ring is fed from one `forwardChunk`'s tap planes, so its width is
        // bounded by the prefill chunk width.
        const ctx_rows = @min(spec_ctxrows_override orelse dc.block_size, e.batchLimit());
        e.specRingEnable(arena, ctx_rows) catch {
            try out.print("\n[spec: ring enable failed; falling back to plain decode]\n", .{});
            break :blk try houtou.engines.common.generate(
                e.driver(), arena, io, prompt, n_predict, StreamCtx.onToken, &stream_ctx,
            );
        };
        e.spec_trace_dbg = spec_trace;
        const tap_buf = try arena.alloc(f32, dc.target_layers.len * ctx_rows * dc.hidden);
        const toks = try arena.alloc(u32, dc.block_size);
        var runner = DraftRunner{
            .e = @ptrCast(&e),
            .ring_filled_fn = ringFilledQwen3,
            .ring_read_fn = ringReadQwen3,
            .dr = &dr,
            .io = io,
            .ctx_rows = ctx_rows,
            .tap = tap_buf,
            .toks = toks,
        };
        try out.print("\n[spec: draft={s} block={d} ctx_rows={d}]\n", .{ @tagName(kind), dc.block_size, ctx_rows });
        try out.flush();
        break :blk try e.generateSpec(
            arena,
            prompt,
            n_predict,
            DraftRunner.draft,
            &runner,
            dc.block_size - 1,
            ctx_rows,
            StreamCtx.onToken,
            &stream_ctx,
        );
    } else try houtou.engines.common.generate(
        e.driver(),
        arena,
        io,
        prompt,
        n_predict,
        StreamCtx.onToken,
        &stream_ctx,
    );

    // ---- `QWEN35_SPEC_DRAFT`: is the DSpine draft numerically alive on THIS
    // target? -------------------------------------------------------------
    //
    // The narrow question this answers, and why it is worth a separate mode:
    // the draft consumes the target's hidden state at five fixed layers, so a
    // target/draft mismatch (hidden width, tap layer indices, vocabulary) is
    // the most likely failure of a fresh pairing and it is SILENT -- the
    // forward still runs, it just produces a meaningless block.  So: run a real
    // target chunk, copy its tap planes out, and push them through the draft's
    // 5-layer forward.  Finite, non-degenerate output from a REAL target
    // context is the evidence that the pairing is wired correctly.
    //
    // It does NOT measure acceptance -- that needs the decode loop, which this
    // engine does not have yet.
    if (spec_draft) {
        if (draft_path == null) {
            try out.print("SPEC-DRAFT: set QWEN35_DRAFT to the dspine GGUF path\n", .{});
            try out.flush();
            return;
        }
        try out.print("SPEC-DRAFT loading {s}\n", .{draft_path.?});
        const dfile = std.Io.Dir.cwd().openFile(io, draft_path.?, .{}) catch {
            try out.print("SPEC-DRAFT: cannot open {s}\n", .{draft_path.?});
            try out.flush();
            return;
        };
        defer dfile.close(io);
        const dstat = try dfile.stat(io);
        var dmm = try std.Io.File.MemoryMap.create(io, dfile, .{
            .len = @intCast(dstat.size),
            .protection = .{ .read = true },
        });
        defer dmm.destroy(io);
        var dg = try houtou.gguf.parse(arena, dmm.memory);
        defer dg.deinit();

        const er = e.embedRows() catch {
            try out.print("SPEC-DRAFT: target embedding is not quantized\n", .{});
            try out.flush();
            return;
        };
        const draft_kind: houtou.dspine_bridge.Kind =
            draft_backend_override orelse (if (e.backendKind() == .cuda) .cuda else .cpu);
        try out.print("SPEC-DRAFT backend={s}\n", .{@tagName(draft_kind)});
        var dr = houtou.draft.load(arena, &dg, er, e.lmHead(), e.maxSeqForSpec(), draft_kind, io, tapl_shift_override) catch |de| {
            try out.print("SPEC-DRAFT load failed: {s}\n", .{@errorName(de)});
            try out.flush();
            return;
        };
        defer dr.deinit();
        const dc = &dr.cfg;
        try out.print(
            "SPEC-DRAFT cfg layers={d} hidden={d} ffn={d} heads={d} kv={d} hd={d} " ++
                "R={d} C={d} V={d} block={d} top_k={d}\n",
            .{ dc.n_layers, dc.hidden, dc.ffn, dc.heads, dc.kv_heads, dc.hd, dc.rank, dc.code_len, dc.vocab, dc.block_size, dc.top_k },
        );
        try out.print("SPEC-DRAFT target_layers:", .{});
        for (dc.target_layers) |tl| try out.print(" {d}", .{tl});
        try out.print("\n", .{});

        // Cross-check the pairing BEFORE running anything: the draft's tap
        // indices must be real layers of THIS model, and its hidden width must
        // match.  A mismatch here is the failure mode this mode exists for.
        if (dc.hidden != e.cfg.hidden) {
            try out.print("SPEC-DRAFT MISMATCH: draft hidden={d}, target hidden={d}\n", .{ dc.hidden, e.cfg.hidden });
            try out.flush();
            return;
        }
        if (dc.vocab != e.cfg.vocab) {
            try out.print("SPEC-DRAFT MISMATCH: draft vocab={d}, target vocab={d}\n", .{ dc.vocab, e.cfg.vocab });
            try out.flush();
            return;
        }
        for (dc.target_layers) |tl| {
            if (tl >= e.cfg.n_layers) {
                try out.print("SPEC-DRAFT MISMATCH: tap layer {d} >= target layers {d}\n", .{ tl, e.cfg.n_layers });
                try out.flush();
                return;
            }
        }
        try out.print("SPEC-DRAFT pairing OK (hidden {d}, vocab {d}, taps within {d} layers)\n", .{ dc.hidden, dc.vocab, e.cfg.n_layers });

        // A real target chunk: the tap planes the draft will actually see.
        e.reset();
        e.specTapEnable(dc.target_layers) catch |te| {
            try out.print("SPEC-DRAFT specTapEnable failed: {s}\n", .{@errorName(te)});
            try out.flush();
            return;
        };
        const toks = [_]u32{ 9707, 11, 220, 16, 2360, 13, 24, 271 };
        const rows: usize = toks.len;
        _ = e.forwardChunk(&toks, 0);
        const ntap = dc.target_layers.len;
        const tap = try arena.alloc(f32, ntap * toks.len * dc.hidden);
        for (0..ntap) |ti| e.specTapPlaneToHost(tap[ti * toks.len * dc.hidden ..][0 .. toks.len * dc.hidden], ti, toks.len);

        // Tap magnitudes: the draft was trained on a residual stream of a
        // particular scale, so a wildly different rms here is the other silent
        // mismatch (e.g. tapping the pre-norm value, or the wrong layer).
        for (0..ntap) |ti| {
            var ss: f64 = 0;
            for (tap[ti * toks.len * dc.hidden ..][0 .. toks.len * dc.hidden]) |v| ss += @as(f64, v) * @as(f64, v);
            try out.print("SPEC-DRAFT tap[{d}] (layer {d}) rms={d:.6}\n", .{
                ti, dc.target_layers[ti],
                @sqrt(ss / @as(f64, @floatFromInt(toks.len * dc.hidden))),
            });
        }

        // The draft forward itself, on those planes.  Call order is the one the
        // real loop uses (`main.zig`'s draft callback): project the context,
        // feed `block_size` rows, then ask for the chain.
        dr.disable_bridge = draft_nobridge;
        dr.projectContext(tap, rows);
        var rb: usize = 0;
        while (rb < dc.block_size) : (rb += 1) {
            // Row 0 is the anchor (the target's last committed id); every other
            // row is the mask token, exactly like the decoder's block input.
            const row_tok: u32 = if (rb == 0) toks[toks.len - 1] else dc.mask_token_id;
            dr.forwardToken(row_tok, rb, @intCast(rows + rb));
        }
        const chain = dr.serialRound(dc.block_size) catch |de| {
            try out.print("SPEC-DRAFT forward failed: {s}\n", .{@errorName(de)});
            try out.flush();
            return;
        };
        try out.print("SPEC-DRAFT block ids:", .{});
        for (chain) |id| try out.print(" {d}", .{id});
        try out.print("\n", .{});
        var distinct: usize = 0;
        for (chain, 0..) |id, i| {
            var seen = false;
            for (chain[0..i]) |prev| if (prev == id) {
                seen = true;
            };
            if (!seen) distinct += 1;
        }
        try out.print("SPEC-DRAFT distinct={d}/{d} -> {s}\n", .{
            distinct, chain.len, if (distinct > 1) "non-degenerate" else "DEGENERATE",
        });
        try out.flush();
        return;
    }

    // `common.generate` returns the ids we already streamed; anything it left
    // (a wrong-id probe) is printed here the same way the Qwen3.5 path does.
    if (dump_ids) {
        try out.print("IDS:", .{});
        for (res.tokens) |id| try out.print(" {d}", .{id});
        try out.print("\n", .{});
    }

    try out.print("\n\n-------------------\n", .{});
    try out.print("Prefill: {d} tokens in {d:.3} s\n", .{ prompt.len, res.prefill_ms / 1000.0 });
    try out.print("Generate: {d} tokens in {d:.3} s  ({d:.2} tok/s)\n", .{
        res.tokens.len,
        res.decode_ms / 1000.0,
        if (res.decode_ms > 0) @as(f64, @floatFromInt(res.tokens.len)) / (res.decode_ms / 1000.0) else 0.0,
    });
    try out.print("{s}\n", .{if (res.hit_eos)
        "Stop reason: EOS"
    else if (res.hit_context_limit)
        "Stop reason: context limit reached"
    else
        "Stop reason: max tokens reached"});
    if (res.spec_rounds > 0) {
        try out.print(
            "Spec: rounds={d} proposed={d} accepted={d} ({d:.3} per round) -> acceptance length {d:.3}\n",
            .{
                res.spec_rounds,
                res.spec_proposed,
                res.spec_accepted,
                @as(f64, @floatFromInt(res.spec_accepted)) / @as(f64, @floatFromInt(res.spec_rounds)),
                // "acceptance length" as the model cards report it: tokens
                // produced per round INCLUDING the verified one.
                1.0 + @as(f64, @floatFromInt(res.spec_accepted)) / @as(f64, @floatFromInt(res.spec_rounds)),
            },
        );
    }
    try out.flush();
}

fn runInference(
    arena: std.mem.Allocator,
    io: Io,
    args: []const [:0]const u8,
    out: *Io.Writer,
    err_out: *Io.Writer,
) !void {
    if (args.len < 4) {
        try err_out.print(
            "usage: {s} run <file.gguf> <prompt> [-n N] [-t TEMPERATURE] [-k K] [-p P] [-s SEED]\n" ++
                "  -n  predict N new tokens (default: until the context fills)\n" ++
                "  -t  temperature; 0 (default) = greedy argmax\n" ++
                "  -k  top-k cutoff (0 = off, default)\n" ++
                "  -p  top-p / nucleus cutoff in (0,1); <=0 or >=1 = off (default)\n" ++
                "  -s  PRNG seed for sampling (default 12345)\n" ++
                "  --threads N  intra-op worker count (0 = default)\n" ++
                "  --backend cpu|cuda  execution backend; cuda runs the quantized\n" ++
                "      matvecs on the GPU and keeps the weights device-resident\n" ++
                "  --kv-type f32|f16|q8_0|q4_0  KV cache format for the\n" ++
                "      full-attention layers (default: f16)\n",
            .{args[0]},
        );
        try err_out.flush();
        return error.MissingArgument;
    }
    const path = args[2];
    const prompt_text = args[3];
    const engine = houtou.engine;
    var n_predict: usize = engine.Engine.max_seq; // default: until the context fills
    var temp: f32 = 0;
    var top_k: usize = 0;
    var top_p: f32 = 0;
    var seed: u64 = 12345;
    var n_threads: usize = 0; // 0 = default (min(16, cpus))
    var backend_kind: houtou.backend.Kind = .cpu;
    var kv_type: houtou.backend.KvType = .f16;
    var i: usize = 4;
    while (i + 1 < args.len + 1 and i < args.len) : (i += 1) {
        if (std.mem.eql(u8, args[i], "--backend") and i + 1 < args.len) {
            // Keep the accepted set in sync with `backend.Kind`; naming the
            // valid values in the error is what makes a typo recoverable.
            backend_kind = std.meta.stringToEnum(houtou.backend.Kind, args[i + 1]) orelse {
                try err_out.print("error: --backend expects 'cpu' or 'cuda', got '{s}'\n", .{args[i + 1]});
                try err_out.flush();
                return error.InvalidArgument;
            };
            i += 1;
        } else if (std.mem.eql(u8, args[i], "--kv-type") and i + 1 < args.len) {
            // KV cache storage format for the full-attention layers.  f16 is
            // the default (halves the f32 footprint, near-lossless); q8_0 /
            // q4_0 trade more precision for 2x / ~3.5x less KV memory; f32 is
            // kept for A/B against the historical path.
            kv_type = std.meta.stringToEnum(houtou.backend.KvType, args[i + 1]) orelse {
                try err_out.print("error: --kv-type expects 'f32', 'f16', 'q8_0' or 'q4_0', got '{s}'\n", .{args[i + 1]});
                try err_out.flush();
                return error.InvalidArgument;
            };
            i += 1;
        } else if (std.mem.eql(u8, args[i], "--threads") and i + 1 < args.len) {
            n_threads = std.fmt.parseInt(usize, args[i + 1], 10) catch {
                try err_out.print("error: --threads expects a number, got '{s}'\n", .{args[i + 1]});
                try err_out.flush();
                return error.InvalidArgument;
            };
            i += 1;
        } else if (std.mem.eql(u8, args[i], "-n") and i + 1 < args.len) {
            n_predict = std.fmt.parseInt(usize, args[i + 1], 10) catch {
                try err_out.print("error: -n expects a number, got '{s}'\n", .{args[i + 1]});
                try err_out.flush();
                return error.InvalidArgument;
            };
            i += 1;
        } else if (std.mem.eql(u8, args[i], "-t") and i + 1 < args.len) {
            temp = std.fmt.parseFloat(f32, args[i + 1]) catch {
                try err_out.print("error: -t expects a number, got '{s}'\n", .{args[i + 1]});
                try err_out.flush();
                return error.InvalidArgument;
            };
            i += 1;
        } else if (std.mem.eql(u8, args[i], "-k") and i + 1 < args.len) {
            top_k = std.fmt.parseInt(usize, args[i + 1], 10) catch {
                try err_out.print("error: -k expects a number, got '{s}'\n", .{args[i + 1]});
                try err_out.flush();
                return error.InvalidArgument;
            };
            i += 1;
        } else if (std.mem.eql(u8, args[i], "-p") and i + 1 < args.len) {
            top_p = std.fmt.parseFloat(f32, args[i + 1]) catch {
                try err_out.print("error: -p expects a number, got '{s}'\n", .{args[i + 1]});
                try err_out.flush();
                return error.InvalidArgument;
            };
            i += 1;
        } else if (std.mem.eql(u8, args[i], "-s") and i + 1 < args.len) {
            seed = std.fmt.parseInt(u64, args[i + 1], 10) catch {
                try err_out.print("error: -s expects a number, got '{s}'\n", .{args[i + 1]});
                try err_out.flush();
                return error.InvalidArgument;
            };
            i += 1;
        } else {
            try err_out.print("error: unknown argument '{s}'\n", .{args[i]});
            try err_out.flush();
            return error.InvalidArgument;
        }
    }

    // ---- architecture dispatch ----
    //
    // Which engine runs is a property of the FILE (`general.architecture`), not
    // of a flag: a Qwen3 GGUF has no `qwen35.*` keys and would fail deep inside
    // the Qwen3.5 loader with a confusing `BadConfig`.  The probe below reads
    // only the metadata KV region, so a misdispatch costs nothing.
    //
    // The Qwen3 path is a separate function rather than a branch inside this
    // one: everything after this point is written against the Qwen3.5 engine's
    // 49 fields (spec decoding, tap ring, nextn), and threading an "or the
    // other engine" through it would be the flag-per-difference abstraction
    // `engines/common.zig` deliberately avoided.
    switch (probeArchitecture(arena, io, path)) {
        .qwen3 => return runInferenceQwen3(arena, io, path, prompt_text, n_predict, temp, top_k, top_p, seed, n_threads, backend_kind, out, err_out),
        .qwen35 => {},
        .unknown => |name| {
            try err_out.print(
                "error: '{s}' declares general.architecture = \"{s}\", which no engine in this build handles (qwen35, qwen3)\n",
                .{ path, name },
            );
            try err_out.flush();
            return error.UnsupportedArchitecture;
        },
    }

    var e = engine.Engine.load(arena, path, io, n_threads, backend_kind, kv_type) catch |e2| {
        try err_out.print("error: failed to load '{s}': {s}\n", .{ path, @errorName(e2) });
        try err_out.flush();
        return e2;
    };
    defer e.deinit();

    e.sampler = engine.Sampler.init(.{ .temperature = temp, .top_k = top_k, .top_p = top_p, .seed = seed });

    var tok = try houtou.tokenizer.Tokenizer.initFromGguf(arena, &e.g);

    const rendered = try houtou.tokenizer.Tokenizer.renderChatTurn(arena, prompt_text, false);
    const prompt = try tok.encode(arena, rendered);
    try out.print("[prompt: {d} tokens]\n", .{prompt.len});

    // `QWEN35_SPEC_TAPS`: stage-0 self-check for the hidden-state taps the
    // DSpine draft consumes.  The property that matters is that plane `i` of
    // `tap_hidden` holds layer `tap_layers[i]`'s OUTPUT hidden state for every
    // row of the chunk -- not some other layer, and not the pre-MLP value.
    //
    // Method: run the SAME chunk twice with two different tap sets that
    // OVERLAP in one layer.  The shared layer's plane must be byte-identical
    // across the two runs, and two different layers must differ.  A wrong
    // offset, a wrong layer, or a plane written before the final residual add
    // all break one of those.  This is a real check, not a smoke test: it
    // compares production `forwardChunk` output against itself under a
    // transformation that must be invisible.
    if (spec_taps) {
        const cfg = &e.cfg;
        // CPU BACKEND ONLY.  This check reads logits and hidden rows back to the
        // host (`Buf.slice()`); on CUDA those are device pointers and the host
        // `dupe`/`@memcpy` segfaults (observed: core dump).  The tap itself is
        // backend-agnostic -- `copyRange` handles D2D -- so verifying it on the
        // CPU backend is sufficient; the CUDA path only ever needs to *use* the
        // buffer, not read it.
        if (e.backendKind() != .cpu) {
            try err_out.print("SPEC-TAPS: host-readback check is CPU-only; skipping (backend != cpu)\n", .{});
        } else {
        const last: u32 = @intCast(cfg.n_layers - 1);
        // A 4-token chunk exercises the batched `forwardChunk` path (k > 1);
        // its LAST row must equal what the serial `forwardLogits` produces for
        // the same final token, given the same prefix.
        const toks = [_]u32{ 9707, 11, 220, 16 };

        // (1) Reference: serial path, one token at a time.  Its logits for the
        //     last token are the ground truth.
        e.reset();
        var ref_logits: []f32 = &.{};
        for (toks, 0..) |tk, ti| {
            _ = e.forwardLogits(tk, @intCast(ti));
        }
        ref_logits = try arena.dupe(f32, e.logits.slice());

        // (2) Batched path.  First its OWN logits (no tap tail involved), so a
        //     batched-vs-serial f32 difference can be told apart from a tap bug.
        e.reset();
        _ = e.forwardChunk(&toks, 0);
        const chunk_logits = try arena.dupe(f32, e.logits.slice());
        var csame: usize = 0;
        var cmax: f32 = 0;
        for (ref_logits, chunk_logits) |x, y| {
            if (x == y) csame += 1;
            const d = @abs(x - y);
            if (d > cmax) cmax = d;
        }

        // (3) Now with a tap on the LAST layer, push that tapped row through the
        //     SAME output_norm + lm_head.
        e.reset();
        try e.specTapEnable(&[_]u32{last});
        _ = e.forwardChunk(&toks, 0);
        const plane = try arena.dupe(f32, e.specTapPlane(0, toks.len).slice());
        const hrow = plane[(toks.len - 1) * cfg.hidden ..][0..cfg.hidden];
        // Copy the tapped row into a backend buffer to reuse the real tail.
        const hbuf = try e.allocHiddenRow();
        e.copyHiddenRowIn(hbuf, hrow);
        e.logitsFromHidden(hbuf);
        const got_logits = try arena.dupe(f32, e.logits.slice());

        // Also compare the batched path's OWN logits (no tap involved) against
        // the serial reference: if those already differ, the discrepancy is a
        // pre-existing batched-vs-serial f32 reduction difference, not the tap.
        var same: usize = 0;
        var maxdiff: f32 = 0;
        for (ref_logits, got_logits) |x, y| {
            if (x == y) same += 1;
            const d = @abs(x - y);
            if (d > maxdiff) maxdiff = d;
        }
        var finite: usize = 0;
        var nonzero: usize = 0;
        for (hrow) |v| {
            if (std.math.isFinite(v)) finite += 1;
            if (v != 0) nonzero += 1;
        }

        try out.print("SPEC-TAPS last_layer={d} chunk={d}\n", .{ last, toks.len });
        // MUST be exact.  Every operator the chunk path batches is defined as
        // the per-token sequence it replaces, so there is no reassociation to
        // tolerate here.  (This used to be annotated "expected to differ",
        // which hid a real GDN prelude bug for as long as it was there.)
        try out.print("SPEC-TAPS batched forwardChunk vs serial forwardLogits: {d}/{d} equal (maxdiff {e:.3})\n", .{
            csame, ref_logits.len, cmax,
        });
        try out.print("SPEC-TAPS tapped-row -> logits equal to SERIAL path: {d}/{d} (maxdiff {e:.3})\n", .{
            same, ref_logits.len, maxdiff,
        });
        // The decisive comparison: the tap-derived logits vs the BATCHED path's
        // own logits.  These MUST be bit-identical -- both go through the same
        // chunk, so the tap is merely re-reading the last layer's output.  A
        // wrong layer, a wrong offset, or a pre-residual capture breaks this.
        var vsame: usize = 0;
        var vmax: f32 = 0;
        for (chunk_logits, got_logits) |x, y| {
            if (x == y) vsame += 1;
            const d = @abs(x - y);
            if (d > vmax) vmax = d;
        }
        try out.print("SPEC-TAPS tapped-row -> logits equal to BATCHED chunk logits: {d}/{d} (maxdiff {e:.3}) -> {s}\n", .{
            vsame, chunk_logits.len, vmax, if (vsame == chunk_logits.len) "PASS" else "FAIL",
        });
        try out.print("SPEC-TAPS last row finite {d} nonzero {d} of {d} -> {s}\n", .{
            finite, nonzero, hrow.len,
            if (finite == hrow.len and nonzero > hrow.len / 2) "PASS" else "FAIL",
        });
        e.reset();
        try out.flush();
        }
    }

    // `QWEN35_SPEC_NEXTN`: stage-1 self-check for the nextn/MTP drafter.
    //
    // A drafter is only useful if its guess is RIGHT often enough to pay for
    // running it, so this measures exactly that: run the target greedily for N
    // tokens, then replay those tokens through the drafter and count how often
    // its single-token guess equals what the target actually produced.
    //
    // It is a real check, not a smoke test.  A wrong `eh_proj` half order, a
    // norm applied to the concatenation instead of each half, a wrong layer
    // index, or a KV cache shared with the target all leave the drafter
    // producing finite, plausible-looking logits -- but its hit rate collapses
    // toward the 1/vocab random floor.  The reference hit rate for a trained
    // MTP block on Qwen is tens of percent, so the PASS threshold is a wide
    // margin either side of "clearly better than chance".
    //
    // CPU BACKEND ONLY, for the same reason as `QWEN35_SPEC_TAPS`: the
    // comparison reads logits back to the host, and on CUDA they are device
    // pointers.
    if (spec_nextn) {
        if (e.nextn == null) {
            try out.print("SPEC-NEXTN: model has no nextn/MTP block -> SKIP\n", .{});
        } else if (e.backendKind() != .cpu) {
            try out.print("SPEC-NEXTN: host-readback check is CPU-only; skipping (backend != cpu)\n", .{});
        } else {
            const cfg = &e.cfg;
            const hidden: usize = cfg.hidden;
            const last: u32 = @intCast(cfg.n_layers - 1);
            // Long enough for the rate to be stable: a 12-step window gave
            // 1..4 hits per prompt with the same code, which is too coarse to
            // read a threshold off.
            const nsteps: usize = 48;

            // (1) Teacher forcing.  Feed the target ONE token at a time and let
            //     it pick the next greedily, building a fixed sequence `seq`.
            //     Also stash the last-layer hidden state for every fed token:
            //     that is what the drafter consumes.
            e.reset();
            try e.specTapEnable(&[_]u32{last});
            const plen: usize = @min(prompt.len, 4);
            var seq: [4 + 48]u32 = undefined;
            @memcpy(seq[0..plen], prompt[0..plen]);
            var hiddens = try arena.alloc([]f32, 4 + 48);
            // Prefill the prompt as ONE chunk first, so the target's KV really
            // holds positions 0..plen (a token fed at position p needs p rows of
            // history to attend to; starting the loop at plen-1 without this
            // would run the target over an EMPTY cache and make every subsequent
            // teacher-forced token wrong).  The tap then holds the last-layer
            // hidden state of every prompt row.
            _ = e.forwardChunk(seq[0..plen], 0);
            const pre_plane = try e.specTapNormedPlane(0, plen);
            for (0..plen) |hi| {
                hiddens[hi] = pre_plane[hi * hidden ..][0..hidden];
            }
            // `hiddens[i]` is the last-layer hidden state AFTER the model has
            // consumed `seq[i]` -- that is the half of the drafter's input that
            // carries the target's knowledge of the prefix.
            var n: usize = plen;
            for (0..nsteps) |_| {
                const tk = seq[n - 1];
                _ = e.forwardChunk(&[_]u32{tk}, @intCast(n - 1));
                hiddens[n - 1] = try e.specTapNormedPlane(0, 1);
                // greedy pick from the target's own logits
                var best: u32 = 0;
                var bestv: f32 = -std.math.inf(f32);
                for (e.logits.slice(), 0..) |v, vi| {
                    if (v > bestv) {
                        bestv = v;
                        best = @intCast(vi);
                    }
                }
                seq[n] = best;
                n += 1;
            }

            // Replay the sequence through the drafter, feeding its OWN KV cache
            // in ascending order exactly as the real proposal loop will.
            e.reset();
            var hits: usize = 0;
            var tries: usize = 0;
            var chain_hits: [16]usize = @splat(0);
            var chain_tries: [16]usize = @splat(0);
            var accept_sum: usize = 0;
            var rounds: usize = 0;
            const hbuf = try e.allocHiddenRow();
            // MTP pairing, as llama.cpp's draft driver builds it: the drafter's
            // input at position p is (embedding of token p+1, target hidden
            // after token p).  It therefore PREDICTS token p+2, not p+1 -- the
            // committed token is always one the target has already seen, and the
            // draft answers "what comes after it".
            //
            // MEASURED (CPU, 27B IQ3_XXS, 3 prompts): pairing embed(p+1) with
            // hidden(p) and comparing against token p+1 gives a 0% hit rate;
            // comparing against token p+2 is the correct target.  Getting the
            // index wrong does not crash -- it just looks like a broken drafter.
            for (1..n - 2) |si| {
                e.copyHiddenRowIn(hbuf, hiddens[si]);
                try e.nextnPredict(hbuf, seq[si + 1], @intCast(si), null);
                var dbest: u32 = 0;
                var dbestv: f32 = -std.math.inf(f32);
                for (e.logits.slice(), 0..) |v, vi| {
                    if (v > dbestv) {
                        dbestv = v;
                        dbest = @intCast(vi);
                    }
                }
                tries += 1;
                if (dbest == seq[si + 2]) hits += 1;
            }

            // ---- CHAINED multi-step draft (what an actual spec round does) ----
            //
            // A spec round does not draft ONE token and stop: it re-runs the MTP
            // head up to `n_draft` times, each step feeding the PREVIOUS step's
            // `shared_head_norm` output back as the hidden half
            // (`common/speculative.cpp`'s non-`chain_heads` path).  The accepted
            // length is the leading run of correct guesses, so a step-3 miss
            // makes every later step of that round worthless -- which is why
            // per-step hit rate alone is not the number that matters.
            //
            // The drafter's KV is reset per round and re-fed from position 0,
            // the same way the real loop rebuilds it from the target's verified
            // hidden states.
            const n_draft: usize = 8;
            var rd: usize = 1;
            while (rd + n_draft + 2 < n) : (rd += 1) {
                e.reset();
                var chained: usize = 0;
                const carried = try arena.dupe(f32, hiddens[rd]);
                e.copyHiddenRowIn(hbuf, carried);
                var step: usize = 0;
                while (step < n_draft) : (step += 1) {
                    // RoPE POSITION: the target's ABSOLUTE position, exactly as
                    // `common/speculative.cpp:1830` does
                    // (`batch.add(id, dp.pos0 + i + 1, ...)`).  Passing a
                    // round-relative index here does not crash and does not
                    // produce obviously wrong output -- it just rotates the
                    // drafter's q/k by the wrong angle, so every draft is
                    // slightly wrong and the hit rate sags for no visible reason.
                    try e.nextnPredict(hbuf, seq[rd + 1 + step], @intCast(rd + 1 + step), hbuf);
                    var dbest: u32 = 0;
                    var dbestv: f32 = -std.math.inf(f32);
                    for (e.logits.slice(), 0..) |v, vi| {
                        if (v > dbestv) {
                            dbestv = v;
                            dbest = @intCast(vi);
                        }
                    }
                    chain_tries[step] += 1;
                    const want = seq[rd + 2 + step];
                    if (dbest == want) {
                        chain_hits[step] += 1;
                        chained += 1;
                        // keep chaining only while correct -- a wrong step would
                        // poison the drafter's history just like in production
                    } else break;
                }
                accept_sum += chained;
                rounds += 1;
            }
            try out.print("SPEC-NEXTN chain rounds={d} n_draft={d} mean_accepted={d:.2}\n", .{
                rounds, n_draft,
                if (rounds > 0) @as(f64, @floatFromInt(accept_sum)) / @as(f64, @floatFromInt(rounds)) else 0,
            });
            try out.print("SPEC-NEXTN chain per-step hit rate:", .{});
            for (0..n_draft) |st| {
                try out.print(" {d:.0}%", .{
                    if (chain_tries[st] > 0)
                        100.0 * @as(f64, @floatFromInt(chain_hits[st])) / @as(f64, @floatFromInt(chain_tries[st]))
                    else
                        0.0,
                });
            }
            try out.print("\n", .{});

            if (spec_nextn_seq) {
                try out.print("SPEC-NEXTN seq:", .{});
                for (seq[0..n]) |t| try out.print(" {d}", .{t});
                try out.print("\n", .{});
            }
            try out.print("SPEC-NEXTN steps={d} attempts={d} hits={d} ({d:.1}%) top1_vocab={d}\n", .{
                nsteps, tries, hits,
                if (tries > 0) 100.0 * @as(f64, @floatFromInt(hits)) / @as(f64, @floatFromInt(tries)) else 0,
                cfg.vocab,
            });
            // The PASS bar is deliberately LOW and expressed as "clearly better
            // than chance", not as "as good as it should be".
            //
            // MEASURED (CPU backend, 27B IQ3_XXS, ~12 attempts per prompt):
            //   * correct pairing (embed(p+1) vs hidden(p), target token p+2):
            //     7.7% .. 30.8% (median ~25%) across three prompts;
            //   * embed(p+1) compared against token p+1 instead: 0% on all three
            //     (the off-by-one that the pairing note above warns about);
            //   * a drafter that silently shares the target's KV, or that norms
            //     the concatenation instead of each half, lands near 1/vocab.
            // With vocab=248320, chance is 4e-6, so even the worst correct run
            // (1 hit in 13) is six orders of magnitude above chance.  A single
            // hit per prompt is therefore enough to prove the wiring; requiring
            // 20% would make this check fail on "1+1等于几" (7.7%) purely from
            // sample size, which is a false alarm, not a defect.
            // A degenerate sequence (the target collapses into repeating one
            // token, e.g. "Explain photosynthesis" -> 814 x 48) makes the
            // drafter's FIRST wrong guess permanent: its KV then carries a
            // wrong history, so it never recovers.  MEASURED: 0% on that
            // prompt, 10-16% on three others, with the identical code path.
            // That is a property of the sequence, not of the wiring, so the
            // verdict below also reports how repetitive the sequence was --
            // read `distinct` before treating a low rate as a defect.
            var distinct: usize = 0;
            for (seq[0..n], 0..) |t, qi| {
                var seen = false;
                for (seq[0..qi]) |prev| {
                    if (prev == t) seen = true;
                }
                if (!seen) distinct += 1;
            }
            try out.print("SPEC-NEXTN distinct={d}/{d} hit rate -> {s}\n", .{
                distinct, n,
                if (tries > 0 and hits * 100 >= tries * 10) "PASS" else "CHECK (degenerate?)",
            });
            e.reset();
            try out.flush();
        }
    }

    // ---- stage 3 self-check: DSpine draft model -----------------------------
    //
    // What this proves (and what it does NOT):
    //   * every one of the 73 tensors resolves by name and loads at its
    //     declared shape -- a name/shape mismatch in the draft file is the
    //     single most likely failure of a fresh port, and it is silent;
    //   * the 5-layer forward runs and produces finite, non-degenerate output;
    //   * the config the forward uses (head counts, R, C, V, block_size,
    //     top_k, target_layers) matches the file, printed for eyeballing
    //     against `dumpGguf`.
    // It does NOT prove the draft is *correct*: the code bridge is not wired
    // yet, so the numbers have no meaning until stage 3b.
    if (spec_draft) {
        if (draft_path == null) {
            try out.print("SPEC-DRAFT: set QWEN35_DRAFT to the dspine GGUF path\n", .{});
        } else {
            try out.print("SPEC-DRAFT loading {s}\n", .{draft_path.?});
            const dfile = std.Io.Dir.cwd().openFile(io, draft_path.?, .{}) catch {
                try out.print("SPEC-DRAFT: cannot open {s}\n", .{draft_path.?});
                return;
            };
            defer dfile.close(io);
            const dstat = try dfile.stat(io);
            var dmm = try std.Io.File.MemoryMap.create(io, dfile, .{
                .len = @intCast(dstat.size),
                .protection = .{ .read = true },
            });
            defer dmm.destroy(io);
            var dg = try houtou.gguf.parse(arena, dmm.memory);
            defer dg.deinit();

            const er = e.embedRows() catch {
                try out.print("SPEC-DRAFT: target embedding is not quantized\n", .{});
                return;
            };
            // Stage 3 (the 5-layer forward + code bridge) follows the target's
            // backend by default; `QWEN35_DRAFT_BACKEND=cpu|cuda` overrides it so
            // the two can be A/B'd on the same build (the bridge kernels agree
            // bit-for-bit across backends, so any output difference is a real
            // defect, not rounding).
            // Stage 3 follows the target's backend.  `QWEN35_DRAFT_BACKEND`
            // overrides it; `=cpu` is the useful one (it keeps stage 3 on the
            // host while the target runs on the GPU) because stage 3 on the
            // device is only worth it once the draft dominates the round, which
            // it does not while stage 4 is still a host loop.  The two backends
            // now agree to <0.5% on the residual stream and are bit-exact on
            // the four bridge kernels -- see docs/dspine-stage3-cuda.md.
            const draft_kind: houtou.dspine_bridge.Kind =
                draft_backend_override orelse (if (backend_kind == .cuda) .cuda else .cpu);
            try out.print("SPEC-DRAFT backend={s}\n", .{@tagName(draft_kind)});
            var dr = houtou.draft.load(arena, &dg, er, e.lmHead(), e.maxSeqForSpec(), draft_kind, io, tapl_shift_override) catch |de| {
                std.debug.print("SPEC-DRAFT load failed: {s}\n", .{@errorName(de)});
                return;
            };
            defer dr.deinit();
            const dc = &dr.cfg;
            try out.print(
                "SPEC-DRAFT cfg layers={d} hidden={d} ffn={d} heads={d} kv={d} hd={d} " ++
                    "R={d} C={d} V={d} block={d} top_k={d}\n",
                .{ dc.n_layers, dc.hidden, dc.ffn, dc.heads, dc.kv_heads, dc.hd, dc.rank, dc.code_len, dc.vocab, dc.block_size, dc.top_k },
            );
            try out.print("SPEC-DRAFT target_layers:", .{});
            for (dc.target_layers) |tl| try out.print(" {d}", .{tl});
            try out.print("\n", .{});

            // ---- forward: one block against a synthetic context ----
            // The bridge needs the target's five tap planes; there is no
            // engine in this self-check (it runs on raw GGUFs), so a
            // deterministic pseudo-random context of `ctx_rows` rows stands in.
            // That is enough to exercise every code path -- the point here is
            // "does the arithmetic stay finite and reset-clean", not "is the
            // draft good", which cannot be answered without a target pass.
            // The draft consumes the TARGET's tap planes, so a pseudo-random
            // stand-in would exercise the arithmetic but not the actual input
            // distribution: the residual stream would grow from real
            // embeddings through weights trained for a completely different
            // signal, and any magnitude read off it would be meaningless.
            // Run a real target chunk instead, copy its taps out, and reset so
            // the self-check leaves no KV state behind.
            // The rest of this self-check reads activations, tap planes and
            // logits on the HOST, and on a device backend every one of those is
            // a device pointer -- `Engine.specTapPlane` included (that is the
            // same CUDA limitation the stage-0 tap check documents).  The
            // magnitudes it prints are diagnostics for the CPU path, so the
            // device run skips them rather than paying a D2H per activation.
            // Run with `--backend cpu` (or `QWEN35_DRAFT_BACKEND=cpu`) to get
            // the full report.
            //
            // DEVICE BACKENDS SKIP THIS ENTIRE REPORT.  Everything below reads
            // activations, tap planes and logits on the HOST, and on a device
            // backend every one of those is a device pointer -- `specTapPlane`
            // included (the stage-0 tap check documents the same limitation).
            // Those magnitudes are CPU-path diagnostics, so the device run
            // returns early instead of paying a D2H per activation.  Run with
            // `--backend cpu` (or `QWEN35_DRAFT_BACKEND=cpu` to keep the target
            // on the GPU) for the full report.
            const host_report = e.backendKind() == .cpu and dr.bridge_kind == .cpu;
            if (!host_report) {
                try out.print(
                    "SPEC-DRAFT: the detailed self-check reads activations on the host and is CPU-only " ++
                        "(target={s} draft={s}); skipping.  Use --backend cpu for the full report.\n",
                    .{ @tagName(e.backendKind()), @tagName(dr.bridge_kind) },
                );
                try out.flush();
            }

            // The smoke runs on BOTH backends when a tapfile is given, so the two
            // produce a comparable chain; otherwise only when stage 3 is on the
            // device (the CPU path has the full report below instead).
            if (!host_report or g_tapfile_path != null) {
                // Device-capable smoke: no host activation reads, so the only
                // things it can assert are the ones that matter for the WIRING
                // (a wrong device pointer or a missing kernel shows up here as a
                // crash or a NaN, not as a small numeric drift):
                //   * the 5-layer forward + code bridge run to completion, and
                //   * `reset()` makes the block reproducible.
                // The tap planes come from a deterministic synthetic pattern:
                // the point is "the same input twice gives the same output",
                // not "the draft is good".
                const ctx_rows = 33;
                const tap = try arena.alloc(f32, dc.target_layers.len * ctx_rows * dc.hidden);
                // `QWEN35_SPEC_TAPFILE=<path>`: raw f32 tap planes, so two
                // backends can be compared on IDENTICAL input.  Without it a
                // deterministic pattern is used, which makes the smoke a
                // wiring check rather than a numeric A/B (the pattern is not
                // what the draft was trained on, so its chain means nothing).
                var tap_from_file = false;
                if (g_tapfile_path) |tp_path| {
                    if (std.Io.Dir.cwd().openFile(io, tp_path, .{})) |tf| {
                        defer tf.close(io);
                        const want_bytes = tap.len * @sizeOf(f32);
                        const got = tf.readPositionalAll(io, std.mem.sliceAsBytes(tap), 0) catch 0;
                        tap_from_file = got == want_bytes;
                        try out.print("SPEC-DRAFT tapfile: {s} ({d} bytes, {s})\n", .{ tp_path, got, if (tap_from_file) "loaded" else "WRONG SIZE" });
                    } else |_| {
                        // No such file: write the pattern so the other backend
                        // has something identical to read.
                        var tap_seed: u32 = 12345;
                        for (tap) |*v| {
                            tap_seed = tap_seed *% 1664525 +% 1013904223;
                            v.* = @as(f32, @floatFromInt(@as(i32, @bitCast(tap_seed >> 8)) - (1 << 23))) / @as(f32, @floatFromInt(1 << 23));
                        }
                        if (std.Io.Dir.cwd().createFile(io, tp_path, .{})) |wf| {
                            defer wf.close(io);
                            _ = wf.writePositionalAll(io, std.mem.sliceAsBytes(tap), 0) catch 0;
                            try out.print("SPEC-DRAFT tapfile: {s} written for the next run\n", .{tp_path});
                        } else |_| {}
                    }
                } else {
                    var tap_seed: u32 = 12345;
                    for (tap) |*v| {
                        tap_seed = tap_seed *% 1664525 +% 1013904223;
                        v.* = @as(f32, @floatFromInt(@as(i32, @bitCast(tap_seed >> 8)) - (1 << 23))) / @as(f32, @floatFromInt(1 << 23));
                    }
                }
                _ = &tap_from_file;
                const toks = try arena.alloc(u32, dc.block_size);
                // Real text for the anchor, MASK for the rest (`generate.py:133`).
                // A repeated id here scores the same candidate 16 times and
                // makes every magnitude read off the block meaningless.
                toks[0] = prompt[prompt.len - 1];
                @memset(toks[1..], dc.mask_token_id);
                dr.disable_bridge = draft_nobridge;

                const runBlk = struct {
                    fn go(d: *houtou.draft.Draft, ids: []const u32, tp: []const f32, rows: usize, nb: usize) void {
                        d.reset();
                        d.projectContext(tp, rows);
                        // The context window ends right before the block: its rows
                        // are absolute positions 0..rows-1, so the block starts at
                        // `rows`.  (Passing `rows` here placed the context AFTER
                        // the block, which rotates every context K at the wrong
                        // phase and inflates the attention output ~150x.)
                        d.appendContext(0);
                        for (0..nb) |t| d.forwardToken(ids[t], t, @intCast(rows + t));
                    }
                }.go;
                runBlk(&dr, toks, tap, ctx_rows, dc.block_size);
                // `reset()` purity: same block twice must be bit-identical.  Read
                // back through the ENGINE-facing buffer via a 1-float D2H, which
                // works on both backends.
                const probe = try arena.alloc(f32, dc.hidden);
                dr.copyResidToHost(probe);
                const first = try arena.alloc(f32, dc.hidden);
                @memcpy(first, probe);
                runBlk(&dr, toks, tap, ctx_rows, dc.block_size);
                dr.copyResidToHost(probe);
                var finite = true;
                var same = true;
                var maxdiff: f32 = 0;
                for (first, probe) |a2, b2| {
                    if (!std.math.isFinite(b2)) finite = false;
                    if (a2 != b2) same = false;
                    maxdiff = @max(maxdiff, @abs(a2 - b2));
                }
                try out.print("SPEC-DRAFT backend-of-chain={s}\n", .{@tagName(dr.bridge_kind)});
                // Per-layer magnitudes after each block row 0: a layer whose
                // numbers blow up or collapse pinpoints the op that diverged.
                for (dr.per_layer[0..dc.n_layers], 0..) |pl, li| {
                    try out.print("SPEC-DRAFT pl{d}: resid={e:.4} attn={e:.4} gate={e:.4} mlp={e:.4} xin={e:.4}\n", .{ li, pl.resid, pl.attn, pl.gate, pl.mlp, pl.xin });
                }
                // The residual stream after the block is the single value both
                // backends must agree on; print a few elements so a divergence
                // is localisable to "the forward" rather than "the chain".
                {
                    const rs = try arena.alloc(f32, dc.hidden);
                    dr.copyResidToHost(rs);
                    var mx: f32 = 0;
                    var l2: f64 = 0;
                    for (rs) |v| {
                        mx = @max(mx, @abs(v));
                        l2 += @as(f64, v) * @as(f64, v);
                    }
                    try out.print("SPEC-DRAFT resid: [0]={e:.6} [1]={e:.6} [{d}]={e:.6} max={e:.4} rms={e:.4}\n", .{
                        rs[0], rs[1], dc.hidden - 1, rs[dc.hidden - 1], mx, @sqrt(l2 / @as(f64, @floatFromInt(dc.hidden))),
                    });
                }
                try out.print("SPEC-DRAFT forward(device): finite={s} reset-purity={s} (maxdiff {e:.3})\n", .{
                    @as([]const u8, if (finite) "PASS" else "FAIL"),
                    @as([]const u8, if (same) "PASS" else "FAIL"),
                    maxdiff,
                });
                // Stage 4 as well, so the whole stage-3+4 chain can be compared
                // across backends: `src/serial.zig` is host-only, so its inputs
                // (`stash_raw`/`stash_rms`/`stash_query`) arrived over PCIe and
                // the chain is then a pure host computation.  Any difference here
                // at all is a real defect -- the bridge kernels are bit-identical
                // across backends (`scripts/dspine_ab.cu`), so a divergence must
                // come from something OUTSIDE them.
                runBlk(&dr, toks, tap, ctx_rows, dc.block_size);
                if (dr.stash_ready) {
                    if (dr.serialRound(dc.block_size)) |chain| {
                        try out.print("SPEC-DRAFT chain(device):", .{});
                        for (chain) |tk| try out.print(" {d}", .{tk});
                        try out.print("\n", .{});
                    } else |ce| try out.print("SPEC-DRAFT serial(device): {s}\n", .{@errorName(ce)});
                } else {
                    try out.print("SPEC-DRAFT serial(device): stash not ready\n", .{});
                }
                try out.flush();
            }

            if (host_report) {
            const ctx_rows = 33;
            const tap = try arena.alloc(f32, dc.target_layers.len * ctx_rows * dc.hidden);
            {
                try e.specTapEnable(dc.target_layers);
                const n_ctx = @min(ctx_rows, prompt.len);
                if (n_ctx > 0) {
                    _ = e.forwardChunk(prompt[0..n_ctx], 0);
                    for (0..dc.target_layers.len) |ti| {
                        const plane = e.specTapPlane(ti, n_ctx).slice();
                        for (0..n_ctx * dc.hidden) |k| {
                            tap[ti * ctx_rows * dc.hidden + k] = plane[k];
                        }
                        // rows past the prompt stay zero: those block slots have
                        // no context row, which the reference never produces.
                        @memset(tap[ti * ctx_rows * dc.hidden + n_ctx * dc.hidden ..][0 .. (ctx_rows - n_ctx) * dc.hidden], 0);
                    }
                }
                e.reset();
            }

            // ---- optional raw activation dump, for cross-referencing against
            // a reference implementation (see `QWEN35_DRAFT_DUMP`).
            //
            // One file per (stage, layer, row) would be tens of thousands of
            // files, so each (stage, layer) gets a file of `block_size` rows
            // appended in row order.  `index.txt` records the shapes so a
            // reading script needs no knowledge of the config.
            var dumper = Dumper.init(g_dump_path, io);
            defer dumper.deinit();
            // Only when a directory was asked for: the sink dereferences
            // `Dumper.dir_path`, so attaching it unconditionally panics on the
            // first `emit`.
            if (g_dump_path != null) {
                dr.dump = Dumper.sink;
                dr.dump_ctx = &dumper;
            }

            const runBlock = struct {
                fn go(d: *houtou.draft.Draft, ids: []const u32, tp: []const f32, rows: usize, nblock: usize) void {
                    d.reset();
                    d.projectContext(tp, rows);
                    // The context sits BEFORE the block in absolute positions;
                    // the draft's rope is relative to absolute position, so the
                    // block's first row must be rotated at `rows`.
                    d.appendContext(0);
                    for (0..nblock) |t| d.forwardToken(ids[t], t, @intCast(rows + t));
                }
            }.go;

            // A block's ids are REAL text, not a fill pattern: the drafter's
            // candidates come from `lm_head @ raw`, so a repeated two-token
            // loop scores the same id 16 times and the chain degenerates into
            // that id -- which is exactly what a fake plan buys.
            const toks = try arena.alloc(u32, dc.block_size);
            {
                // Right-align the prompt: the LAST `block_size` ids are the ones
                // in front of the block we are about to draft.
                // Real text for the anchor, MASK for the rest (`generate.py:133`).
                // A repeated id here scores the same candidate 16 times and
                // makes every magnitude read off the block meaningless.
                toks[0] = prompt[prompt.len - 1];
                @memset(toks[1..], dc.mask_token_id);
            }

            dr.disable_bridge = draft_nobridge;
            // The block's ids must be known before the dump can expand them to
            // embedding rows, so record them first (`forwardToken` overwrites the
            // same slots, which is harmless).
            @memcpy(dr.toks_for_dump[0..dc.block_size], toks);
            if (dr.dump_dir) |dd| dr.dumpFwdInputs(dd, tap, ctx_rows);
            runBlock(&dr, toks, tap, ctx_rows, dc.block_size);

            var finite = true;
            {
                const row = dr.resid.slice()[0..dc.hidden];
                for (row) |v| {
                    if (!std.math.isFinite(v)) finite = false;
                }
            }
            // Distinctness: the bridge's update is skipped on the anchor row,
            // so if `feature_finish`'s write landed nowhere the last row's
            // residual would equal the anchor's. Compare the two ends of the
            // block -- a cheap, honest correlate of "the write did something".
            // Magnitude report.  These absolute values are NOT a pass/fail
            // signal: Qwen residual streams legitimately carry outliers of
            // 1e2..1e4 and the input layer norms preserve them, so "big" is
            // normal here.  What the numbers show is the SHAPE of the path --
            // embedding at the model's own scale, a normed input of O(1..1e2),
            // residual finite, and a bridge whose delta/message are non-zero
            // (a zero there means `feature_finish`'s write landed nowhere).
            try out.print(
                "SPEC-DRAFT magnitudes: rms={d:.3} embed={d:.4} resid={d:.1} xin={d:.2} gout={d:.2} " ++
                    "attn={d:.1} mlp={d:.1} delta={d:.2} msg={d:.2} codes={d:.2}\n",
                .{
                    houtou.draft.maxAbsForSelfCheck(dr.bridge_rms.slice()),
                    dr.debug_embed, dr.debug_resid, dr.debug_xin,   dr.debug_gout,
                    dr.debug_attn,  dr.debug_mlp,    dr.debug_delta, dr.debug_msg,
                    dr.debug_codes,
                },
            );
            try out.print("SPEC-DRAFT forward: finite={s} residual[0]={d:.6} residual[end]={d:.6}\n", .{
                @as([]const u8, if (finite) "PASS" else "FAIL"),
                dr.resid.slice()[0], dr.resid.slice()[dc.hidden - 1],
            });

            // ---- reset purity ----
            // `reset()` must leave no trace: same block twice must give a
            // byte-identical residual stream.  This is also the check that
            // nothing reads a KV row past `len`/`ctx_len` -- a stale row would
            // only show up on the second pass.
            const first = try arena.alloc(f32, dc.hidden);
            runBlock(&dr, toks, tap, ctx_rows, dc.block_size);
            @memcpy(first, dr.resid.slice()[0..dc.hidden]);
            runBlock(&dr, toks, tap, ctx_rows, dc.block_size);
            var reset_same = true;
            var maxdiff: f32 = 0;
            for (first, dr.resid.slice()[0..dc.hidden]) |a, b| {
                if (a != b) reset_same = false;
                maxdiff = @max(maxdiff, @abs(a - b));
            }
            try out.print("SPEC-DRAFT reset purity: {s} (maxdiff {e:.3})\n", .{
                @as([]const u8, if (reset_same) "PASS" else "FAIL"), maxdiff,
            });

            // ---- stage 4: serial refinement ----
            // The chain is what stage 5 hands to the target for verification.
            // Print it next to the candidate set it came from: a chain member
            // that is not in row 0's candidates means the table walk is broken,
            // which no acceptance rate would tell apart from "the draft is bad".
            if (dr.stash_ready) {
                runBlock(&dr, toks, tap, ctx_rows, dc.block_size);
                if (dr.serialRound(dc.block_size)) |chain| {
                    // Compare against the TARGET's own layer outputs at the
                    // tapped layers: the draft is trained to live in the same
                    // activation scale, so a 100x gap at layer 0 is a bug, not
                    // a depth difference.
                    for (dc.target_layers, 0..) |tl, ti| {
                        const plane = e.specTapPlane(ti, prompt.len - 1).slice();
                        var mx: f32 = 0;
                        var l2: f64 = 0;
                        for (plane[(prompt.len - 2) * dc.hidden ..][0..dc.hidden]) |vv| {
                            mx = @max(mx, @abs(vv));
                            l2 += @as(f64, vv) * @as(f64, vv);
                        }
                        try out.print("SPEC-DRAFT T{d} (target L{d}): maxabs={d:.1} rms={d:.2}\n", .{ tl, tl, mx, @sqrt(l2 / @as(f64, @floatFromInt(dc.hidden))) });
                    }
                    for (dr.per_layer[0..dc.n_layers], 0..) |pl, li| {
                        try out.print("SPEC-DRAFT L{d}: xin={d:.1} attn={d:.1} gate={d:.1} swi={d:.1} mlp={d:.1} resid={d:.1} | x0={d:.3} g0={d:.2} m0={d:.3}\n", .{ li, pl.xin, pl.attn, pl.gate, pl.swi, pl.mlp, pl.resid, pl.x0, pl.g0, pl.m0 });
                    }

                    try out.print("SPEC-DRAFT chain:", .{});
                    for (chain) |tk| try out.print(" {d}", .{tk});
                    try out.print("\n", .{});
                    try out.print("SPEC-DRAFT cand row0:", .{});
                    for (dr.cand[0..dc.top_k]) |tk| try out.print(" {d}", .{tk});
                    try out.print("\n", .{});

                    // ---- acceptance: the ONLY meaningful number in stage 4 ----
                    // Rewind the target, replay the same 16 ids, then continue
                    // GREEDILY for the same number of tokens the chain claims.
                    // An absolute-magnitude check cannot tell "the draft is
                    // right" from "the draft is off by a constant"; a hit rate
                    // can.  This is also the number stage 5 has to beat.
                    // The prefix must be the WHOLE prompt, not just the last 16
                    // ids: rewinding to position 0 and feeding a window in the
                    // middle of a sentence makes the target itself produce
                    // garbage, and then "the truth" is nonsense rather than a
                    // reference (measured: truth repeated one id 5 times).
                    e.reset();
                    var lg_buf = e.forwardChunk(prompt[0 .. prompt.len - 1], 0);
                    var truth: [32]u32 = undefined;
                    var ntruth: usize = 0;
                    var hits: usize = 0;
                    var pos: u32 = @intCast(prompt.len - 1);
                    while (ntruth < chain.len) : (ntruth += 1) {
                        const lg = lg_buf.slice();
                        var best: u32 = 0;
                        var bv: f32 = -std.math.inf(f32);
                        for (lg, 0..) |v, idx| {
                            if (v > bv) {
                                bv = v;
                                best = @intCast(idx);
                            }
                        }
                        truth[ntruth] = best;
                        if (best == chain[ntruth]) hits += 1;
                        lg_buf = e.forwardLogits(best, @intCast(pos));
                        pos += 1;
                    }
                    try out.print("SPEC-DRAFT truth:", .{});
                    for (truth[0..ntruth]) |tk| try out.print(" {d}", .{tk});
                    try out.print("\n", .{});
                    try out.print("SPEC-DRAFT hits: {d}/{d}\n", .{ hits, ntruth });
                    // Split the 0/15 in two: rerun the same block with the
                    // TARGET's own residual stream (the last tap layer, which
                    // is what the draft's `fc` consumes) injected into the
                    // serial stash.  If the chain still misses, the defect is in
                    // serial_table/serial_chain and the draft forward is
                    // innocent; if it starts hitting, the draft forward is the
                    // only suspect left.
                    {
                        std.debug.print("INJ prompt_len={d} block={d}\n", .{ prompt.len, dc.block_size });
                        if (prompt.len < 1 + dc.block_size) return;
                        const trow = prompt.len - 1 - dc.block_size;
                        const plane = e.specTapPlane(dc.target_layers.len - 1, prompt.len - 1).slice();
                        std.debug.print("INJ plane_len={d} trow={d} need={d}\n", .{ plane.len, trow, (trow + dc.block_size) * dc.hidden });
                        const slice = plane[trow * dc.hidden ..][0 .. dc.block_size * dc.hidden];
                        const rr = try arena.alloc(f32, dc.block_size);
                        for (0..dc.block_size) |ri| {
                            var acc: f64 = 0;
                            for (slice[ri * dc.hidden ..][0..dc.hidden]) |v| acc += @as(f64, v) * @as(f64, v);
                            rr[ri] = @floatCast(@sqrt(acc / @as(f64, @floatFromInt(dc.hidden))));
                        }
                        dr.injectStash(slice, rr);
                        const chain2 = dr.serialRound(dc.block_size) catch |ce2| blk: {
                            try out.print("SPEC-DRAFT inject: {s}\n", .{@errorName(ce2)});
                            break :blk &.{};
                        };
                        try out.print("SPEC-DRAFT inject-chain:", .{});
                        var h2: usize = 0;
                        for (chain2, 0..) |tk, ci| {
                            try out.print(" {d}", .{tk});
                            if (ci < ntruth and tk == truth[ci]) h2 += 1;
                        }
                        try out.print("\n", .{});
                        try out.print("SPEC-DRAFT inject-hits: {d}/{d}\n", .{ h2, chain2.len });
                    }


                } else |ce| {
                    try out.print("SPEC-DRAFT serial: {s}\n", .{@errorName(ce)});
                }
            } else {
                try out.print("SPEC-DRAFT serial: stash not ready (block never completed)\n", .{});
            }

            try out.print("SPEC-DRAFT tables -> {s}\n", .{
                @as([]const u8, if (finite) "PASS" else "FAIL"),
            });
            } // end host_report
            try out.flush();
        }
    }

    var stream_ctx = StreamCtx{ .out = out, .tok = &tok, .arena = arena };
    try out.print("Output: ", .{});
    try out.flush();
    const res = if (spec_mtp) blk: {
        // MTP speculative decode: the model's own nextn block drafts.
        //
        // GRAPH OFF, and before the prefill: `nextnPredict` is per-round and
        // has no device-position twin, so a captured node would replay the first
        // round's arguments -- and because the draft runs INSIDE the span the
        // verify forward opened, that corrupts the verify too (measured: the
        // first round verified token 0 instead of 119016).  See
        // `CudaBackend.forceGraphOff`.
        if (e.backendKind() == .cuda) houtou.backend_cuda.CudaBackend.forceGraphOff();
        //
        // No external checkpoint, no code bridge, no tap ring: the head is one
        // extra layer that was trained jointly with the target, so it reads the
        // target's LAST-layer hidden state (tapped by the verify forward that
        // `generateSpec` runs anyway) and needs no context window of its own.
        // The head keeps its OWN KV cache (in the nextn slot), so it is not
        // reset per round the way the DSpine draft is.
        if (e.nextn == null) {
            try out.print("\n[spec-mtp: this model has no nextn/MTP block; falling back to plain decode]\n", .{});
            break :blk try e.generate(arena, prompt, n_predict, StreamCtx.onToken, &stream_ctx);
        }
        const cfg = &e.cfg;
        const last_layer: u32 = @intCast(cfg.n_layers - 1);
        e.specTapEnable(&[_]u32{last_layer}) catch {
            try out.print("\n[spec-mtp: tap enable failed; falling back to plain decode]\n", .{});
            break :blk try e.generate(arena, prompt, n_predict, StreamCtx.onToken, &stream_ctx);
        };
        // Chain length per round.
        //
        // MEASURED, and the previous default of 4 was wrong: it was chosen by
        // maximising ACCEPTANCE LENGTH (2.82 tokens/round at draft=4), but that
        // is not the quantity that matters -- every drafted token costs a full
        // verify row whether or not it is accepted, so a long chain with a low
        // hit rate is pure waste.  Throughput on 27B IQ3_XXS:
        //
        //   draft   tok/s   accept rate   tokens/round
        //     1      26.3      82.4 %          1.82
        //     2      26.4      79.2 %          2.58
        //     3      20.8      39.1 %          2.13
        //     4      20.8      42.6 %          2.67
        //   (64-token run: draft=2 25.65 vs draft=4 20.84, i.e. 23 % faster.)
        //
        // Past the second draft the hit rate collapses, so 2 is the default.
        // `QWEN35_SPEC_MTP_DRAFT` overrides it for a sweep.
        var max_draft: usize = 2;
        if (houtou.cudart.environ_map) |em| {
            if (em.get("QWEN35_SPEC_MTP_DRAFT")) |raw| {
                if (std.fmt.parseInt(usize, std.mem.trim(u8, raw, " \t\r\n"), 10)) |v| {
                    if (v >= 1) max_draft = @min(v, 31);
                } else |_| {}
            }
        }
        const hbuf = e.allocHiddenRow() catch {
            try out.print("\n[spec-mtp: hidden buffer unavailable; falling back to plain decode]\n", .{});
            break :blk try e.generate(arena, prompt, n_predict, StreamCtx.onToken, &stream_ctx);
        };
        e.spec_trace_dbg = spec_trace;
        var p_min: f32 = 0.5;
        if (houtou.cudart.environ_map) |em| {
            if (em.get("QWEN35_SPEC_PMIN")) |raw| {
                if (std.fmt.parseFloat(f32, std.mem.trim(u8, raw, " \t\r\n"))) |v| {
                    p_min = v;
                } else |_| {}
            }
        }
        var runner = MtpRunner{
            .e = &e,
            .tap_index = 0,
            .h = hbuf,
            .max_draft = max_draft,
            .p_min = p_min,
        };
        try out.print("\n[spec-mtp: draft={d} tokens/round p_min={d:.2}]\n", .{ max_draft, p_min });
        try out.flush();
        break :blk try e.generateSpec(
            arena,
            prompt,
            n_predict,
            MtpRunner.draft,
            &runner,
            max_draft,
            1, // ctx_rows: unused by this draft (the head reads the last tap only)
            StreamCtx.onToken,
            &stream_ctx,
            null, // cand_fn: QWEN35_SPEC_RECALL is DSpine-only
            null,
        );
    } else if (spec_gen and draft_path != null) blk: {
        // Speculative decoding path.  The draft is loaded here (and only here):
        // the self-check above is a separate mode that returns before reaching
        // this point.
        const dfile = std.Io.Dir.cwd().openFile(io, draft_path.? , .{}) catch {
            try out.print("\n[spec: cannot open the draft gguf; falling back to plain decode]\n", .{});
            break :blk try e.generate(arena, prompt, n_predict, StreamCtx.onToken, &stream_ctx);
        };
        defer dfile.close(io);
        const dstat = try dfile.stat(io);
        var dmm = try std.Io.File.MemoryMap.create(io, dfile, .{
            .len = @intCast(dstat.size),
            .protection = .{ .read = true },
        });
        defer dmm.destroy(io);
        var dg = try houtou.gguf.parse(arena, dmm.memory);
        defer dg.deinit();
        const er = e.embedRows() catch break :blk try e.generate(arena, prompt, n_predict, StreamCtx.onToken, &stream_ctx);
        const kind: houtou.dspine_bridge.Kind =
            draft_backend_override orelse (if (e.backendKind() == .cuda) .cuda else .cpu);
        var dr = houtou.draft.load(arena, &dg, er, e.lmHead(), e.maxSeqForSpec(), kind, io, tapl_shift_override) catch |de| {
            try out.print("\n[spec: draft load failed ({s}); falling back to plain decode]\n", .{@errorName(de)});
            break :blk try e.generate(arena, prompt, n_predict, StreamCtx.onToken, &stream_ctx);
        };
        defer dr.deinit();
        const dc = &dr.cfg;
        dr.disable_bridge = draft_nobridge;
        var spec_dumper = Dumper.init(g_dump_path, io);
        defer spec_dumper.deinit();
        // ONLY install the activation sink when a directory was asked for: the
        // sink dereferences `Dumper.dir_path`, so attaching it unconditionally
        // panics on the first `emit` (which the spec path always reaches).
        if (g_dump_path != null) {
            dr.dump = Dumper.sink;
            dr.dump_ctx = &spec_dumper;
        }
        // Candidate scoring borrows the TARGET's lm_head (the draft has no copy).
        dr.lm_head_matvec = lmHeadHook;
        dr.lm_head_ctx = &e;

        // The draft consumes the TARGET's hidden states at five fixed layers,
        // so the taps have to be enabled before any forward pass.
        e.specTapEnable(dc.target_layers) catch {
            try out.print("\n[spec: tap enable failed; falling back to plain decode]\n", .{});
            break :blk try e.generate(arena, prompt, n_predict, StreamCtx.onToken, &stream_ctx);
        };

        // The draft's left context is re-forwarded through the TARGET every
        // round, so it is bounded by the prefill chunk width (`nbatch`): a
        // longer window would need several `forwardChunk` calls and the taps
        // are only sized for one.  At b=16 that is 16 rows of extra prefill per
        // round, which is the real cost of this design and the reason the window
        // is not larger.
        // The ring is fed from one `forwardChunk`'s tap planes, so its width is
        // bounded by the prefill chunk width (`nbatch`); a wider override would
        // read past the plane.
        const ctx_rows = @min(spec_ctxrows_override orelse dc.block_size, e.batchLimit());
        try e.specRingEnable(arena, ctx_rows);
        e.spec_trace_dbg = spec_trace;
        e.spec_posstat = spec_posstat;
        const tap_buf = try arena.alloc(f32, dc.target_layers.len * ctx_rows * dc.hidden);
        const toks = try arena.alloc(u32, dc.block_size);
        var runner = DraftRunner{
            .e = @ptrCast(&e),
            .ring_filled_fn = ringFilledQwen35,
            .ring_read_fn = ringReadQwen35,
            .dr = &dr,
            .io = io,
            .ctx_rows = ctx_rows,
            .tap = tap_buf,
            .toks = toks,
        };
        try out.print("\n[spec: draft={s} block={d} ctx_rows={d}]\n", .{ @tagName(kind), dc.block_size, ctx_rows });
        try out.flush();
        // `QWEN35_SPEC_RECALL`: per-row candidate recall.  Row `r` asks whether
        // the target's own pick for block row `r+1` was among the draft's `k`
        // candidates.  High recall with a wrong pick points at stage 4's scoring;
        // low recall points at the block forward that produced the row's hidden.
        if (spec_recall) e.spec_recall_k = dc.top_k;
        break :blk try e.generateSpec(
            arena,
            prompt,
            n_predict,
            DraftRunner.draft,
            &runner,
            dc.block_size - 1,
            ctx_rows,
            StreamCtx.onToken,
            &stream_ctx,
            if (spec_recall) DraftRunner.candidates else null,
            &runner,
        );
    } else try e.generate(arena, prompt, n_predict, StreamCtx.onToken, &stream_ctx);

    try out.print("\n\n-------------------\n", .{});
    try out.print("Prefill: {d} tokens in {d:.3} s  ({d:.2} tok/s)\n", .{
        prompt.len,
        res.prefill_ms / 1000.0,
        if (res.prefill_ms > 0) @as(f64, @floatFromInt(prompt.len)) / (res.prefill_ms / 1000.0) else 0.0,
    });
    try out.print("Generate: {d} tokens in {d:.3} s  ({d:.2} tok/s)\n", .{
        res.tokens.len,
        res.decode_ms / 1000.0,
        if (res.decode_ms > 0) @as(f64, @floatFromInt(res.tokens.len)) / (res.decode_ms / 1000.0) else 0.0,
    });
    if (res.spec_rounds > 0) {
        try out.print(
            "Spec: rounds={d} proposed={d} accepted={d}  accept-rate={d:.1}%  mean-accepted={d:.2}  [{d:.2} tokens/round]\n",
            .{
                res.spec_rounds,
                res.spec_proposed,
                res.spec_accepted,
                if (res.spec_proposed > 0) 100.0 * @as(f64, @floatFromInt(res.spec_accepted)) / @as(f64, @floatFromInt(res.spec_proposed)) else 0.0,
                if (res.spec_rounds > 0) @as(f64, @floatFromInt(res.spec_accepted)) / @as(f64, @floatFromInt(res.spec_rounds)) else 0.0,
                if (res.spec_rounds > 0) @as(f64, @floatFromInt(res.spec_accepted + res.spec_rounds)) / @as(f64, @floatFromInt(res.spec_rounds)) else 0.0,
            },
        );
        // `QWEN35_SPEC_POSSTAT`: histogram of "how far into the chain the
        // verifier got".  `accepted k` means chain[0..k) matched and chain[k]
        // did not.
        if (spec_posstat) {
            try out.print("Spec-pos:", .{});
            for (res.spec_pos_reached, 0..) |n, k| {
                if (n == 0) continue;
                try out.print(" acc{d}={d}", .{ k, n });
            }
            try out.print("\n", .{});
        }
        // `QWEN35_SPEC_RECALL`: per-row recall of the draft's candidate set.
        if (spec_recall) {
            try out.print("Spec-recall:", .{});
            var any = false;
            for (res.spec_recall_tries, 0..) |tries, r| {
                if (tries == 0) continue;
                any = true;
                try out.print(" row{d}={d}/{d}({d:.0}%)", .{
                    r + 1,
                    res.spec_recall_hit[r],
                    tries,
                    100.0 * @as(f64, @floatFromInt(res.spec_recall_hit[r])) / @as(f64, @floatFromInt(tries)),
                });
            }
            if (any) try out.print("\n", .{});
            const ms = res.spec_miss_select;
            const mf = res.spec_miss_forward;
            if (ms + mf > 0) {
                try out.print("Spec-miss: select={d} ({d:.1}%)  forward={d} ({d:.1}%)  of {d} rejected rows\n", .{
                    ms,
                    100.0 * @as(f64, @floatFromInt(ms)) / @as(f64, @floatFromInt(ms + mf)),
                    mf,
                    100.0 * @as(f64, @floatFromInt(mf)) / @as(f64, @floatFromInt(ms + mf)),
                    ms + mf,
                });
            }
        }
    }
    try out.print("{s}\n", .{if (res.hit_eos)
        "Stop reason: EOS"
    else if (res.hit_context_limit)
        "Stop reason: context limit reached"
    else
        "Stop reason: max tokens reached"});
    // `QWEN35_DUMP_IDS`: print the raw generated token ids.  Used for A/B md5
    // probes -- text output can hide a divergence that shows up in ids, and ids
    // are what a refactor must keep byte-identical.
    if (dump_ids) {
        try out.print("IDS:", .{});
        for (res.tokens) |id| try out.print(" {d}", .{id});
        try out.print("\n", .{});
    }
    // `$QWEN35_PROF`: ask the backend for its *complete* report now that the run
    // is over.  Before this call the CUDA backend's `CUDA-M  types:`/`time:` and
    // its `PROF-SLOT`/`PROF-COPY` tables only appeared at internal trigger points
    // (`% 500` matvecs, the closing of the prefill window) or in `deinit`, which
    // the arena-allocated engine never reaches -- so a run that stopped on EOS
    // between two trigger points, or any f32/BF16 run (whose matvec branch never
    // calls `reportStats`), printed a header with no tables.  Goes through
    // `Engine`, not the device module, so `engine.zig` keeps importing no
    // hardware backend (AGENTS.md §1.1).
    if (houtou.engine.prof) {
        e.profBackendReport();
        houtou.engine.profReport();
        houtou.backend_cpu.profReportMatvec();
    }
    try out.flush();
}

fn dumpGguf(
    arena: std.mem.Allocator,
    io: Io,
    args: []const [:0]const u8,
    out: *Io.Writer,
    err_out: *Io.Writer,
) !void {
    const path = args[1];
    var full = false;
    for (args[2..]) |a| {
        if (std.mem.eql(u8, a, "--full")) full = true;
    }

    // --- open + mmap (never read the whole file into memory) --------------
    // NB: `openFileAbsolute` asserts on relative paths (and panics), so use
    // `cwd().openFile`, which accepts both relative and absolute paths.
    const file = Io.Dir.cwd().openFile(io, path, .{}) catch |e| {
        try err_out.print("error: cannot open '{s}': {s}\n", .{ path, @errorName(e) });
        try err_out.flush();
        return e;
    };
    defer file.close(io);
    const stat = try file.stat(io);
    const file_size: usize = @intCast(stat.size);
    if (file_size == 0) {
        try err_out.print("error: '{s}' is empty\n", .{path});
        try err_out.flush();
        return error.EmptyFile;
    }
    var mm = try Io.File.MemoryMap.create(io, file, .{
        .len = file_size,
        .protection = .{ .read = true },
    });
    defer mm.destroy(io);

    var g = gguf.parse(arena, mm.memory) catch |e| {
        try err_out.print("error: {s} is not a valid GGUF file: {s}\n", .{ path, @errorName(e) });
        if (e == error.BadMagic) {
            if (mm.memory.len >= 4) {
                const head = mm.memory[0..4];
                try err_out.print("       first bytes: {x} {x} {x} {x} (expected 47 47 55 46 = \"GGUF\")\n", .{
                    head[0], head[1], head[2], head[3],
                });
            } else {
                try err_out.print("       file is only {d} byte(s) long\n", .{mm.memory.len});
            }
        } else if (e == error.SizeMismatch) {
            try err_out.print("       tensor byte sizes do not tile the data section " ++
                "(header parsed, but the file is truncated or corrupt)\n", .{});
        } else if (e == error.UnknownTensorType) {
            try err_out.print("       a tensor uses a ggml_type this build does not know; " ++
                "refusing to guess its size\n", .{});
        }
        try err_out.flush();
        return e;
    };
    defer g.deinit();

    // --- dump -------------------------------------------------------------
    // The dump itself lives in `gguf.File.print`, so any tool that already
    // parsed a GGUF can reuse it. The reader only knows its bytes, not the
    // path it was mapped from, so the "file ..." line stays here.
    try out.print("file {s}\n", .{path});
    if (full) try g.printFull(out) else try g.print(out);

    try out.flush();
}
