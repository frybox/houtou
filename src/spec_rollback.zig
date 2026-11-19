//! GDN / KV rollback for speculative decoding (DSpine port, stage 2).
//!
//! WHY THIS EXISTS
//! ---------------
//! Speculative decoding runs the target model over `block` rows (anchor +
//! `block-1` drafts) in ONE chunk, then keeps only the accepted prefix.  The
//! target's per-token state has already advanced by `block` rows by the time
//! we know how many were accepted, so it must be wound back to `accepted + 1`.
//!
//! WHAT MUST BE ROLLED BACK (the complete list -- nothing else carries
//! per-token state; see `Engine.step` and `Engine.forwardChunk`):
//!
//!   * full-attention layers: `KvCache.len` (the rows themselves can stay --
//!     they are overwritten by the next verify, and nothing reads past `len`)
//!   * GDN layers: `GdnState.conv`  (the last K-1=3 raw samples per channel)
//!                 `GdnState.state` (the [nv, kd, vd] delta-rule matrix)
//!
//! APPROACH: SNAPSHOT + REPLAY, not incremental undo.
//! -------------------------------------------------
//! A conv history and a delta-rule matrix are both *non-linearly* consumed by
//! each step (the conv shifts, the delta rule divides by a running norm), so
//! there is no cheap inverse.  Instead -- following `dspine_mlx/hybrid.py`
//! `rollback()` -- we keep ONE pre-verify copy of the state per GDN layer,
//! and when fewer than `block` rows are accepted we restore that copy and
//! replay only the accepted prefix.
//!
//! The replay target already exists: `gdnStepRowsChunk` is the exact per-row
//! recurrence the verify chunk used (`Engine.forwardGdnCoreChunk`), so
//! replaying `accepted` rows through it reproduces the state the model would
//! have had if the verify chunk had been `accepted` rows long.  No new kernel.
//!
//! COST: the snapshot is `2 * n_gdn_layers * (conv_dim*3 + nv*kd*vd) * 4 B`
//! = 2 * 48 * (10240*3 + 16*128*128) * 4 B = 31.4 MB for the 27B.  It is
//! taken once per round (not per layer), and the replay is ~1.26 us per
//! token per layer, i.e. 0.06-0.97 ms for 1-15 accepted tokens -- against a
//! ~11 ms verify, so it does not move the needle.
//!
//! NOT IN SCOPE HERE: the *decision* of how many rows were accepted.  That
//! belongs to the sampler (stage 5); this module takes `accepted` as an
//! already-decided count and makes the state consistent with it.

const std = @import("std");

/// Snapshot of one GDN layer's carried state, plus its full-attention
/// counterpart's row count.
///
/// The buffers are owned by the caller (the engine allocates them once at
/// load: they are the same size every round, so re-allocating per round would
/// be pure churn).  This struct only records *where* the pre-verify bytes live.
pub const LayerSnapshot = struct {
    /// Pre-verify `GdnState.conv` bytes (`conv_dim * (conv_kernel-1)` floats).
    conv: []f32,
    /// Pre-verify `GdnState.state` bytes (`nv * kd * vd` floats).
    state: []f32,
    /// Pre-verify `GdnState.conv_len`.
    conv_len: u32,
};

/// How many rows of a verify block were accepted, and therefore how many
/// rows the state must reflect.
///
/// `accepted` counts the DRAFTS that matched, so the committed row count is
/// `accepted + 1` (the anchor, or the bonus token that replaces the first
/// mismatch).  Keeping the two separate is what stops the classic off-by-one:
/// `dspine` computes `gamma = block - 1` drafts and commits `accepted + 1`.
pub const Rollback = struct {
    /// Number of draft tokens that matched the target's prediction.
    accepted: u32,
    /// Rows the verify chunk actually ran (`= block`, i.e. `1 + gamma`).
    block: u32,
    /// Rows that remain committed after trimming (`= accepted + 1`).
    committed: u32,

    /// `true` when the whole block was accepted: nothing to undo, and the
    /// caller must NOT replay (the state already reflects all `block` rows).
    pub fn isNoop(self: Rollback) bool {
        return self.committed == self.block;
    }

    /// Rows that must be replayed from the snapshot, i.e. `committed`.
    /// Only meaningful when `!isNoop()`.
    pub fn replayRows(self: Rollback) u32 {
        return self.committed;
    }

    pub fn init(accepted: u32, block: u32) Rollback {
        std.debug.assert(block >= 1);
        std.debug.assert(accepted < block); // a full accept is `accepted == block - 1`
        return .{ .accepted = accepted, .block = block, .committed = accepted + 1 };
    }
};

/// Copy a GDN layer's live conv/state bytes into its snapshot.
///
/// Called ONCE per verify round, before `forwardChunk(verify)` advances the
/// state by `block` rows.  Both operands are plain f32 slices here: this is
/// host-side bookkeeping, and `snapshot` is always a host staging buffer (or,
/// on a device backend, a separate device region copied by `copyRange` -- the
/// backend decides, this function only forwards).
///
/// The `conv_len` field is snapshotted too: `forwardGdnPrelude` does not
/// advance it (`gdnPrefillCore` owns that), but `reset()` and future callers
/// may, and restoring only the buffers would leave the two inconsistent.
pub fn takeSnapshot(dst: *LayerSnapshot, conv: []const f32, state: []const f32, conv_len: u32) void {
    std.debug.assert(dst.conv.len == conv.len);
    std.debug.assert(dst.state.len == state.len);
    @memcpy(dst.conv, conv);
    @memcpy(dst.state, state);
    dst.conv_len = conv_len;
}

/// Put the snapshot back.  After this the layer's carried state is exactly
/// what it was before the verify chunk, so the caller can replay.
pub fn restoreSnapshot(src: LayerSnapshot, conv: []f32, state: []f32) u32 {
    std.debug.assert(src.conv.len == conv.len);
    std.debug.assert(src.state.len == state.len);
    @memcpy(conv, src.conv);
    @memcpy(state, src.state);
    return src.conv_len;
}

test "Rollback: committed is accepted+1, and full accept is a noop" {
    // gamma = block - 1 drafts; committing k drafts means k+1 rows.
    const t = Rollback.init(0, 16);
    try std.testing.expectEqual(@as(u32, 1), t.committed);
    try std.testing.expect(!t.isNoop());

    const full = Rollback.init(15, 16);
    try std.testing.expectEqual(@as(u32, 16), full.committed);
    try std.testing.expect(full.isNoop());

    // The last partial accept still needs a replay.
    const last = Rollback.init(14, 16);
    try std.testing.expectEqual(@as(u32, 15), last.committed);
    try std.testing.expect(!last.isNoop());
}

test "Rollback: full-accept commit equals block for every block size" {
    var block: u32 = 2;
    while (block <= 32) : (block += 1) {
        const full = Rollback.init(block - 1, block);
        try std.testing.expectEqual(block, full.committed);
        try std.testing.expect(full.isNoop());
        // One short of full always replays.
        const one_short = Rollback.init(block - 2, block);
        try std.testing.expectEqual(block - 1, one_short.committed);
        try std.testing.expect(!one_short.isNoop());
    }
}

test "Rollback: replayRows is the committed count, not the accepted count" {
    // The caller replays `committed` rows from the snapshot.  Replaying
    // `accepted` would drop the bonus token's own state update, which is the
    // same off-by-one as committing too few rows.
    const r = Rollback.init(3, 16);
    try std.testing.expectEqual(@as(u32, 4), r.replayRows());
    try std.testing.expect(r.replayRows() != r.accepted);
}

// ---------------------------------------------------------------------------
// Equivalence test: restore+replay == "had only run `committed` rows"
// ---------------------------------------------------------------------------
//
// This is the test that makes the whole rollback worth writing: it pins the
// primitive to the property the engine actually relies on, rather than to a
// copy of its own implementation.  The oracle below is deliberately a naive
// scalar transcription of the four GDN steps (decay / kv-mem / outer / read),
// mirroring `engine.zig`'s `gdnDecay`, `gdnKvMem`, `gdnOuter`, `gdnStateOut`
// (SPEC A4-A5) -- and the conv step mirrors `elementwise.convStepKernel`.
//
// A "row" here is (k, v, q, gamma, beta, conv input).  Two runs are compared:
//
//   A) run ALL `block` rows, snapshot taken before, then restore + replay
//      `committed` rows
//   B) run ONLY the first `committed` rows from the same initial state
//
// A and B must be bit-identical in both the conv history and the 128x128
// state.  If the snapshot misses a byte or the replay count is off by one,
// this fails.

const oracle = struct {
    fn convStep(conv: []f32, y: []f32, w: []const f32) void {
        // conv[taps=4] over one channel-block; `conv` is channel-major history
        // of the last 3 raw samples, byte-order identical to convStepKernel.
        for (y, 0..) |*yv, i| {
            const h = conv[i * 3 ..][0..3];
            const cur = yv.*;
            yv.* = h[0] * w[i * 4] + h[1] * w[i * 4 + 1] + h[2] * w[i * 4 + 2] + cur * w[i * 4 + 3];
            h[0] = h[1];
            h[1] = h[2];
            h[2] = cur;
        }
    }

    fn step(sh: []f32, k: []const f32, v: []const f32, q: []const f32,
            km: []f32, delta: []f32, out: []f32, gamma: f32, beta: f32, vd: usize) void {
        // (1) decay
        for (sh) |*s| s.* *= gamma;
        // (2) km[j] = sum_i sh[i*vd+j] * k[i]
        @memset(km, 0);
        for (k, 0..) |ki, i| {
            for (0..vd) |j| km[j] += sh[i * vd + j] * ki;
        }
        // (3) delta = beta*(v-km); S += k (x) delta
        for (0..vd) |j| delta[j] = (v[j] - km[j]) * beta;
        for (k, 0..) |ki, i| {
            for (0..vd) |j| sh[i * vd + j] += ki * delta[j];
        }
        // (4) out[j] = sum_i sh[i*vd+j] * q[i]
        @memset(out, 0);
        for (q, 0..) |qi, i| {
            for (0..vd) |j| out[j] += sh[i * vd + j] * qi;
        }
    }
};

test "restore+replay is bit-identical to a fresh run of the committed prefix" {
    const alloc = std.testing.allocator;
    const nk = 3; // k/q width (small but not degenerate)
    const vd = 8; // value width
    const conv_dim = 10; // conv channels per row
    const taps: usize = 4;
    const hist = conv_dim * (taps - 1);

    var prng = std.Random.DefaultPrng.init(0xD5_91_4E);
    const rnd = prng.random();

    // Deterministic per-row data (k, v, q, gamma, beta, conv input).
    const block: usize = 8;
    const kdata = try alloc.alloc(f32, block * nk);
    defer alloc.free(kdata);
    const vdata = try alloc.alloc(f32, block * vd);
    defer alloc.free(vdata);
    const qdata = try alloc.alloc(f32, block * nk);
    defer alloc.free(qdata);
    const ydata = try alloc.alloc(f32, block * conv_dim);
    defer alloc.free(ydata);
    // `convStepKernel` rewrites `y` IN PLACE (the row becomes silu(conv)), so
    // run B needs its own copy of the same inputs -- reusing `ydata` would feed
    // B the already-convolved values of run A.
    const ydataB = try alloc.alloc(f32, block * conv_dim);
    defer alloc.free(ydataB);
    // Run B needs a THIRD copy: run A's replay consumes `ydataB`.
    const ydataC = try alloc.alloc(f32, block * conv_dim);
    defer alloc.free(ydataC);
    const gammas = try alloc.alloc(f32, block);
    defer alloc.free(gammas);
    const betas = try alloc.alloc(f32, block);
    defer alloc.free(betas);
    for (kdata) |*x| x.* = rnd.float(f32) * 2 - 1;
    for (vdata) |*x| x.* = rnd.float(f32) * 2 - 1;
    for (qdata) |*x| x.* = rnd.float(f32) * 2 - 1;
    for (ydata) |*x| x.* = rnd.float(f32) * 2 - 1;
    @memcpy(ydataB, ydata);
    @memcpy(ydataC, ydata);
    for (gammas) |*x| x.* = 0.9 + rnd.float(f32) * 0.1;
    for (betas) |*x| x.* = rnd.float(f32) * 0.5;

    const wconv = try alloc.alloc(f32, conv_dim * taps);
    defer alloc.free(wconv);
    for (wconv) |*x| x.* = rnd.float(f32) * 2 - 1;

    // Shared initial state (both runs start here).
    const sh0 = try alloc.alloc(f32, nk * vd);
    defer alloc.free(sh0);
    for (sh0) |*x| x.* = rnd.float(f32) * 0.25;
    const cv0 = try alloc.alloc(f32, hist);
    defer alloc.free(cv0);
    for (cv0) |*x| x.* = rnd.float(f32) * 2 - 1;

    const km = try alloc.alloc(f32, vd);
    defer alloc.free(km);
    const delta = try alloc.alloc(f32, vd);
    defer alloc.free(delta);
    const out = try alloc.alloc(f32, vd);
    defer alloc.free(out);

    // Two scratch state copies (A: verify+rollback, B: committed-only oracle),
    // allocated ONCE outside the trial loop -- a `defer` inside the loop would
    // leak one pair per trial.
    const shA = try alloc.alloc(f32, nk * vd);
    defer alloc.free(shA);
    const cvA = try alloc.alloc(f32, hist);
    defer alloc.free(cvA);
    const shB = try alloc.alloc(f32, nk * vd);
    defer alloc.free(shB);
    const cvB = try alloc.alloc(f32, hist);
    defer alloc.free(cvB);
    const snap_conv = try alloc.alloc(f32, hist);
    defer alloc.free(snap_conv);
    const snap_state = try alloc.alloc(f32, nk * vd);
    defer alloc.free(snap_state);

    var acc: u32 = 0;
    while (acc + 1 < block) : (acc += 1) {
        // fresh copy of the shared initial state for this trial
        @memcpy(shA, sh0);
        @memcpy(cvA, cv0);

        const rb = Rollback.init(acc, @intCast(block));
        var snap: LayerSnapshot = .{ .conv = snap_conv, .state = snap_state, .conv_len = 0 };
        takeSnapshot(&snap, cvA, shA, 0);

        // Run A, part 1: the FULL block (what a verify chunk does).
        for (0..block) |t| {
            const y = ydata[t * conv_dim ..][0..conv_dim];
            oracle.convStep(cvA, y, wconv);
            oracle.step(shA, kdata[t * nk ..][0..nk], vdata[t * vd ..][0..vd],
                qdata[t * nk ..][0..nk], km, delta, out, gammas[t], betas[t], vd);
        }

        // Run A, part 2: rollback (skipped when the whole block was accepted,
        // i.e. `committed == block`).
        //
        // The replay must consume FRESH conv inputs (`ydataB`), not the rows
        // part 1 left behind: `convStepKernel` overwrites `y` with silu(conv),
        // so replaying from `ydata` would feed the layer pre-convolved values.
        // This mirrors reality, where a replay re-runs `forwardGdnPrelude` over
        // the same token ids and therefore re-derives the same raw y.
        if (!rb.isNoop()) {
            _ = restoreSnapshot(snap, cvA, shA);
            for (0..rb.replayRows()) |t| {
                const y = ydataB[t * conv_dim ..][0..conv_dim];
                oracle.convStep(cvA, y, wconv);
                oracle.step(shA, kdata[t * nk ..][0..nk], vdata[t * vd ..][0..vd],
                    qdata[t * nk ..][0..nk], km, delta, out, gammas[t], betas[t], vd);
            }
        }

        // Run B: only the committed prefix, from the same initial state.
        @memcpy(shB, sh0);
        @memcpy(cvB, cv0);
        for (0..rb.committed) |t| {
            const y = ydataC[t * conv_dim ..][0..conv_dim];
            oracle.convStep(cvB, y, wconv);
            oracle.step(shB, kdata[t * nk ..][0..nk], vdata[t * vd ..][0..vd],
                qdata[t * nk ..][0..nk], km, delta, out, gammas[t], betas[t], vd);
        }

        try std.testing.expectEqualSlices(f32, cvB, cvA);
        try std.testing.expectEqualSlices(f32, shB, shA);
    }
}
