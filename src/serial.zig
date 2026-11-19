//! DSpine serial refinement (SPECULATIVE DECODING, stage 4).
//!
//! The draft's 5-layer forward produces, for every row of a block, a
//! `top_k`-wide candidate set -- but a block is a CHAIN, not 15 independent
//! picks: the `feature_message` the bridge applies to row `r` depends on which
//! candidate row `r-1` emitted.  Scoring each candidate in isolation therefore
//! cannot see the constraint, and the reference does not do it either.
//!
//! `serial_table` turns the chain into a lookup problem.  Write `s[j][k]` for
//! "the score of candidate `k` at row `r`, GIVEN row `r-1` chose lane `j`".
//! The reference's derivation (`sglang_dspine/kernels/dspine.py:211`) is
//!
//!     base[j]  = <LmHead[cand[r][j]], norm_w * raw[row]>
//!     s[j][k]  = base[k] + rms[row] * sum_d m[k][d] * TG[cand[r][k]][d]
//!     m[k][:]  = sigmoid(query + gate[sid[j]] + gate_bias) * value[sid[j]]
//!     sid[j]   = (r == 0) ? anchor : cand[r-1][j]
//!
//! whose ONLY dependence on the previous lane `j` is through `sid[j]`.  The two
//! per-vocabulary tables
//!
//!     TG[v]  = bf16(LmHead[v] * norm_w) @ W_last          [R]
//!     TSV[v] = [G_p; A] @ bf16(sqrt(C) * Codebook[v])     [2R]
//!
//! fold the H- and C-wide operands down to R so the per-row work is O(K*R)
//! after one pass over K lm_head rows (without them it would be O(K*H) plus an
//! R-wide re-projection per candidate per row).
//!
//! `serial_chain` (`:287`) then walks the table: start at lane `idx[0][0]`,
//! emit that token, look up `idx[r+1][chosen]`, repeat.  The emitted chain is
//! what gets verified against the target -- greedy acceptance takes the longest
//! matching prefix plus one bonus token (`sampling/.../caps.py:35`).
//!
//! SCOPE OF THIS FILE: the two per-vocabulary tables plus the two table
//! kernels, as plain Zig.  Stage 4 is validated against the MLX formulation on
//! small synthetic shapes here; MLX is the authority (not the Triton port).

const std = @import("std");
const backend = @import("backend.zig");
const backend_cpu = @import("backend_cpu.zig");
const gguf = @import("gguf.zig");

const Buf = backend.Buf;
const LinearW = backend.LinearW;

/// bf16 rounding (round-to-nearest-even), matching `dspine.c`'s
/// `q35_f32_to_bf16`.  The reference stores both tables as bf16; keeping f32
/// would score a different function than the reference emits.
pub fn roundBf16(x: f32) f32 {
    const bits: u32 = @bitCast(x);
    if ((bits & 0x7fff_ffff) > 0x7f80_0000) return x; // NaN / inf pass through
    const lsb = (bits >> 16) & 1;
    const rounded = bits +% 0x7fff +% lsb;
    return @bitCast(rounded & 0xffff_0000);
}

pub fn toBf16Bits(x: f32) u16 {
    return @truncate(@as(u32, @bitCast(roundBf16(x))) >> 16);
}

pub fn fromBf16Bits(h: u16) f32 {
    return @bitCast(@as(u32, h) << 16);
}

/// The two per-vocabulary tables `build_serial_tables`
/// (`sglang_dspine/kernels/dspine.py:323`) materialises.
///
/// Both are indexed by TOKEN ID, so they are `vocab` rows wide; that is the
/// only stage-4 state that scales with the vocabulary (762 MB at 27B:
/// 248320 * 512 * 2 B for `tg`, twice that for `tsv`).
pub const Tables = struct {
    /// `TG[v] = bf16(LmHead[v] * norm_w) @ W_last` -- `[vocab][rank]`.
    ///
    /// The `* norm_w` is ELEMENTWISE, not an inner product: the reference is
    /// `torch.matmul((lm_head_w * nwf).to(bf16), wb)` with `lm_head_w [V,H]`,
    /// `wb [H,R]`, so the bf16 rounding lands on each of the H products before
    /// the R-wide contraction.
    tg: []u16,
    /// `TSV[v] = [G_p; A] @ bf16(sqrt(C) * Codebook[v])` -- `[vocab][2*rank]`.
    tsv: []u16,
    vocab: usize,
    rank: usize,
    code_len: usize,

    // ---- what those rows are computed FROM, and which are already done ----
    //
    // The reference builds all `V` rows up front (`build_serial_tables`).  That
    // is 650 G multiplies at 27B -- 12 minutes on one core -- and a decode step
    // only ever indexes the couple of hundred ids its candidates happen to hit.
    // So the buffers are still full-size (762 MB, the same memory the reference
    // spends) but a row is only CONTRACTED on first use, and the result is
    // cached.  Same arithmetic per row; the difference is whether unused rows
    // are ever paid for.
    tg_done: std.DynamicBitSetUnmanaged,
    tsv_done: std.DynamicBitSetUnmanaged,
    /// `[H]`: the target's terminal norm.
    norm_w: []const f32,
    /// `[R][H]`: `write_w` transposed -- see the note in `fillRow`.
    wt: []const f32,
    /// `[2R][C]`: `[predecessor_gate; prev]`.
    pv_w: []const f32,
    lm_head: LinearW,
    codebook: LinearW,

    pub fn deinit(self: *Tables, alloc: std.mem.Allocator) void {
        alloc.free(self.tg);
        alloc.free(self.tsv);
        alloc.free(self.wt);
        self.tg_done.deinit(alloc);
        self.tsv_done.deinit(alloc);
        self.* = undefined;
    }

    /// `TG[v]` and `TSV[v]`, computing either on first use.
    pub fn ensure(self: *Tables, alloc: std.mem.Allocator, v: usize) !void {
        if (!self.tg_done.isSet(v)) {
            try self.fillTg(alloc, v);
            self.tg_done.set(v);
        }
        if (!self.tsv_done.isSet(v)) {
            try self.fillTsv(alloc, v);
            self.tsv_done.set(v);
        }
    }

    fn fillTg(self: *Tables, alloc: std.mem.Allocator, v: usize) !void {
        const hidden = self.norm_w.len;
        const rank = self.rank;
        const row = try alloc.alloc(f32, hidden);
        defer alloc.free(row);
        dequantRowInto(row, self.lm_head, v, hidden);
        // Round the H products to bf16 FIRST -- that is where the reference's
        // `.to(torch.bfloat16)` sits -- then contract against the R columns.
        for (row, self.norm_w) |*a, b| a.* = roundBf16(a.* * b);
        for (0..rank) |d| {
            var acc: f32 = 0;
            for (row, self.wt[d * hidden ..][0..hidden]) |a, b| acc += a * b;
            self.tg[v * rank + d] = toBf16Bits(acc);
        }
    }

    fn fillTsv(self: *Tables, alloc: std.mem.Allocator, v: usize) !void {
        const c = self.code_len;
        const rank = self.rank;
        const cb_row = try alloc.alloc(f32, c);
        defer alloc.free(cb_row);
        const scaled = try alloc.alloc(f32, c);
        defer alloc.free(scaled);
        const scale = @sqrt(@as(f32, @floatFromInt(c)));
        dequantRowInto(cb_row, self.codebook, v, c);
        for (cb_row, 0..) |x, k| scaled[k] = roundBf16(scale * x);
        for (0..2 * rank) |d| {
            var acc: f32 = 0;
            for (self.pv_w[d * c ..][0..c], scaled) |a, b| acc += a * b;
            self.tsv[v * 2 * rank + d] = toBf16Bits(acc);
        }
    }

    /// `TG[v]`, the gate side of the last-layer write.  Caller must `ensure(v)`.
    pub fn tgRow(self: Tables, v: usize) []const u16 {
        return self.tg[v * self.rank ..][0..self.rank];
    }

    /// `TSV[v][0..R]` -- the predecessor-gate weights.  Caller must `ensure(v)`.
    pub fn tsvGate(self: Tables, v: usize) []const u16 {
        return self.tsv[v * 2 * self.rank ..][0..self.rank];
    }

    /// `TSV[v][R..2R]` -- the predecessor VALUE weights.  Caller must `ensure(v)`.
    pub fn tsvValue(self: Tables, v: usize) []const u16 {
        return self.tsv[v * 2 * self.rank + self.rank ..][0..self.rank];
    }
};

/// `build_serial_tables` in the MLX formulation
/// (`dspine_mlx/dspine_mlx/model.py`, `build_serial_tables`).
///
/// * `lm_head`  -- the TARGET's `[V, H]` output projection
/// * `norm_w`   -- the DRAFT's terminal norm weight, `[H]`
/// * `write_w`  -- the LAST layer's `feature_write_weights[last]`, `[H][R]`
/// * `pv_w`     -- `[predecessor_gate; prev]` concatenated, `[2R][C]`
/// * `codebook` -- `[V][C]`, row-major
///
/// Allocates the table storage but contracts NO rows: see `Tables.ensure`.
pub fn buildTables(
    alloc: std.mem.Allocator,
    lm_head: LinearW,
    norm_w: []const f32,
    write_w: []const f32,
    pv_w: []const f32,
    codebook: LinearW,
    vocab: usize,
    hidden: usize,
    rank: usize,
    code_len: usize,
) !Tables {
    std.debug.assert(norm_w.len == hidden);
    std.debug.assert(write_w.len == hidden * rank);
    std.debug.assert(pv_w.len == 2 * rank * code_len);

    // `write_w` arrives as [H][R] (the GGUF row layout) but every contraction
    // walks it once per hidden element at a FIXED d, so the naive access
    // `write_w[h * rank + d]` strides by `rank` floats -- one cache line per
    // multiply.  Transposing to [R][H] makes the inner loop contiguous, which
    // is the difference between ~12 minutes and ~1.5 for a full vocabulary.
    const wt = try alloc.alloc(f32, rank * hidden);
    errdefer alloc.free(wt);
    for (0..hidden) |h| {
        for (0..rank) |d| wt[d * hidden + h] = write_w[h * rank + d];
    }

    const tg = try alloc.alloc(u16, vocab * rank);
    errdefer alloc.free(tg);
    const tsv = try alloc.alloc(u16, vocab * 2 * rank);
    errdefer alloc.free(tsv);
    var tg_done = try std.DynamicBitSetUnmanaged.initEmpty(alloc, vocab);
    errdefer tg_done.deinit(alloc);
    var tsv_done = try std.DynamicBitSetUnmanaged.initEmpty(alloc, vocab);
    errdefer tsv_done.deinit(alloc);

    return .{
        .tg = tg,
        .tsv = tsv,
        .vocab = vocab,
        .rank = rank,
        .code_len = code_len,
        .tg_done = tg_done,
        .tsv_done = tsv_done,
        .norm_w = norm_w,
        .wt = wt,
        .pv_w = pv_w,
        .lm_head = lm_head,
        .codebook = codebook,
    };
}


/// Dequantise one whole row of `w` into `dst` (`cols` = the row width).
///
/// Delegates to `backend_cpu.dequantRowInto`, which already walks the row one
/// 256-element row-block at a time -- the contract `Exec.dequantRow` has and
/// the reason a wide row must not be handed to a single `dequantRowFn` call
/// (AGENTS.md 5.2).  Kind-level: this is the CPU oracle path, not a device one.
pub fn dequantRowInto(dst: []f32, w: LinearW, row: usize, cols: usize) void {
    std.debug.assert(dst.len == cols);
    switch (w) {
        .f32 => |m| @memcpy(dst, m[row * cols ..][0..cols]),
        .quant => |q| backend_cpu.dequantRowInto(dst, q, row),
    }
}

pub const Stash = struct {
    /// `[rows][hidden]` bf16-rounded residual stream BEFORE the last write.
    raw: []f32,
    /// `[rows]` the RMS that write was scaled by.
    rms: []const f32,
    /// `[rows][rank]` the receiver query (`packed[:, :R]`), bf16-rounded.
    query: []const f32,
    rows: usize,
};

/// `serial_table` (`sglang_dspine/kernels/dspine.py:350`).
///
/// `cand` is `[rows][k]` token ids, ASCENDING within a row (the reference's
/// `topk_ids` contract; ties in `argmax` then resolve to the larger id, which
/// is what `return_indices_tie_break_left` does on an ascending row).
///
/// `idx[n][j]` = the best candidate lane at row `n` given the previous row
/// chose lane `j`; row 0 of a block has the block's anchor as its sender, so
/// every `j` there agrees.
pub fn serialTable(
    out_idx: []i16,
    stash: Stash,
    norm_w: []const f32,
    cand: []const u32,
    anchors: []const u32,
    lm_head: LinearW,
    /// Taken by pointer: rows are contracted on first use (`Tables.ensure`), and
    /// a round only ever needs the handful of ids its candidates name.
    tables: *Tables,
    alloc: std.mem.Allocator,
    gate_b: []const f32,
    block_size: usize,
    hidden: usize,
    k: usize,
    scratch: struct {
        /// `[hidden]` one dequantised lm_head row, reused per candidate.
        w_rows: []f32,
        /// `[hidden]` = `norm_w * raw[row]`, hoisted out of the candidate loop.
        scored: []f32,
    },
) void {
    const t = block_size;
    const rk = tables.rank;
    const n_rows = stash.rows / t * (t - 1);
    std.debug.assert(out_idx.len >= n_rows * k);

    for (0..n_rows) |n| {
        const b = n / (t - 1);
        const r = n % (t - 1);
        const row = b * t + 1 + r;
        const anchor: u32 = anchors[b];

        // ---- base[j] = <LmHead[cand[n][j]], norm_w * raw[row]> ----
        // `u = norm_w * raw[row]` is computed once (the reference hoists it out
        // of the h-block loop the same way); then each candidate costs one
        // dequantised lm_head row and one H-wide dot.
        for (0..hidden) |h| scratch.scored[h] = stash.raw[row * hidden + h] * norm_w[h];
        var base: [MAX_K]f32 = undefined;
        std.debug.assert(k <= MAX_K);
        for (0..k) |j| {
            const tok = cand[n * k + j];
            dequantRowInto(scratch.w_rows, lm_head, tok, hidden);
            var acc: f32 = 0;
            for (scratch.w_rows[0..hidden], scratch.scored[0..hidden]) |a, u| acc += a * u;
            base[j] = acc;
        }

        // ---- sid[j] = (r == 0) ? anchor : cand[n-1][j] ----
        // `prev` is the SAME LANE of the previous row: choosing lane j at row
        // r-1 is exactly what makes sid[j] the sender for this row.
        const prev_row = if (n >= 1) n - 1 else 0;
        const rms = stash.rms[row];

        // Materialise every table row this (row, lane) block will read.  Doing
        // it here rather than inside the inner loop keeps the contraction cost
        // out of the K*K scoring nest; `ensure` is idempotent and cached.
        for (0..k) |j| {
            const sid: u32 = if (r == 0) anchor else cand[prev_row * k + j];
            tables.ensure(alloc, sid) catch @panic("serial table allocation failed");
        }
        for (0..k) |kc| tables.ensure(alloc, cand[n * k + kc]) catch @panic("serial table allocation failed");

        for (0..k) |j| {
            const sid: u32 = if (r == 0) anchor else cand[prev_row * k + j];
            const g_send = tables.tsvGate(sid);
            const v_send = tables.tsvValue(sid);

            // ---- m[:] = sigmoid(query + gate[sid] + gate_b) * value[sid] ----
            // Recomputed per `j`: `m` depends on `sid`, and `sid` is the only
            // thing `j` changes.  K*R work per (n, j) and K=16, R=512 -- the
            // reference does the same with a tl.dot over the tile.
            var best: f32 = -std.math.inf(f32);
            var bestk: usize = 0;
            for (0..k) |kc| {
                const tok = cand[n * k + kc];
                const tg = tables.tgRow(tok);
                var acc: f32 = 0;
                for (0..rk) |d| {
                    const g = 1.0 / (1.0 + @exp(-(stash.query[row * rk + d] +
                        fromBf16Bits(g_send[d]) + gate_b[d])));
                    // NO bf16 round here.  `serial_table` builds `m` in f32 from
                    // bf16 table entries -- the only narrowing points are TSV and
                    // TG (`propose.py:207-214`: `tsv_sid = self.tsv[sid].astype(f32)`,
                    // `m = sigmoid(...) * tvv`, no cast around it).  Rounding `m`
                    // would be a cast the reference does not perform, and this
                    // table is what picks the chain.
                    const m = g * fromBf16Bits(v_send[d]);
                    acc += m * fromBf16Bits(tg[d]);
                }
                const s = base[kc] + rms * acc;
                if (s > best) {
                    best = s;
                    bestk = kc;
                }
            }
            out_idx[n * k + j] = @intCast(bestk);
        }
    }
}

/// `selector_top_k` is 16 on the shipped checkpoints, and the table layout is
/// sized for that; stage 4 does not need to grow it.
pub const MAX_K = 64;


/// `serial_chain` (`sglang_dspine/kernels/dspine.py:287`).
///
/// Walks the table: `j` starts at `idx[0][0]`, and each emitted token's own
/// lane is what selects the next row.  `out` holds `block_size - 1` tokens.
pub fn serialChain(
    out: []u32,
    idx: []const i16,
    cand: []const u32,
    block_size: usize,
    k: usize,
) void {
    const t = block_size;
    const gamma = t - 1;
    std.debug.assert(out.len >= gamma);
    std.debug.assert(idx.len >= gamma * k);
    var j: usize = @intCast(idx[0]);
    for (0..gamma) |r| {
        out[r] = cand[r * k + j];
        const next = if (r + 1 < gamma) @as(usize, @intCast(idx[(r + 1) * k + j])) else j;
        j = next;
    }
}
