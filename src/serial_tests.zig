//! Tests for the DSpine serial refinement (stage 4).
//!
//! Every expected value is recomputed from the reference FORMULA, not from
//! `serial.zig`'s own helpers: an oracle that calls the code under test only
//! proves the code is self-consistent.  The formula below is transcribed from
//! `sglang_dspine/kernels/dspine.py:211` (`_serial_table`) with the tensor
//! algebra written out longhand.
//!
//! Shapes are tiny (4-wide hidden, 2-wide rank, ...) so the expected values can
//! be checked by hand when something disagrees.

const std = @import("std");
const testing = std.testing;
const serial = @import("serial.zig");
const gguf = @import("gguf.zig");
const backend = @import("backend.zig");

/// bf16 round-to-nearest-even, written independently of `serial.roundBf16`'s
/// bit trick: round through the value itself so an error in the bit trick shows
/// up as a disagreement rather than cancelling out.
fn refBf16(x: f32) f32 {
    if (!std.math.isFinite(x) or x == 0) return x;
    // bf16 keeps the top 16 bits: 8 explicit mantissa bits.  Find the two
    // representable neighbours of `x` and take the nearer, ties to even.
    const ax = @abs(x);
    const e: i32 = @intFromFloat(@floor(std.math.log2(ax)));
    const step: f32 = std.math.ldexp(@as(f32, 1.0), e - 7);
    const lo = @floor(x / step) * step;
    const hi = lo + step;
    const dl = @abs(x - lo);
    const dh = @abs(hi - x);
    if (dl < dh) return lo;
    if (dh < dl) return hi;
    // Tie: the mantissa's low bit decides. `@mod` on a value/step ratio gives
    // the integer mantissa's parity.
    return if (@mod(@round(lo / step), 2.0) == 0) lo else hi;
}

fn bf(x: f32) f32 {
    return refBf16(x);
}

fn bits(x: f32) u16 {
    return @truncate(@as(u32, @bitCast(bf(x))) >> 16);
}

fn unb(h: u16) f32 {
    return @bitCast(@as(u32, h) << 16);
}

test "serial: bf16 rounding matches the reference scan" {
    // RNE through the bit trick and through the value scan must agree on a
    // spread that hits every case: exact, below half, exactly half (to even),
    // above half, and negatives.
    const cases = [_]f32{ 0, 1, -1, 0.5, 3, 1.0009765625, 1.001953125, 1.00048828125, 2.5, -2.5, 1e-8, 65504, 123.456, -0.3333 };
    for (cases) |x| {
        try testing.expectEqual(bits(x), serial.toBf16Bits(x));
    }
}

/// `build_serial_tables` longhand (`dspine_mlx/model.py`).
///   TG[v][d]  = sum_h bf16(LmHead[v][h] * norm_w[h]) * W_last[h][d]
///   TSV[v][d] = sum_c pv_w[d][c] * bf16(sqrt(C) * Codebook[v][c])
fn refBuildTables(
    a: std.mem.Allocator,
    lm: []const f32,
    norm_w: []const f32,
    write_w: []const f32,
    pv_w: []const f32,
    codebook: []const f32,
    vocab: usize,
    h: usize,
    r: usize,
    c: usize,
) !struct { tg: []f32, tsv: []f32 } {
    const tg = try a.alloc(f32, vocab * r);
    const tsv = try a.alloc(f32, vocab * 2 * r);
    const scale = @sqrt(@as(f32, @floatFromInt(c)));
    for (0..vocab) |v| {
        var prod: f32 = 0;
        _ = &prod;
        for (0..r) |d| {
            var acc: f32 = 0;
            for (0..h) |hh| acc += bf(lm[v * h + hh] * norm_w[hh]) * write_w[hh * r + d];
            tg[v * r + d] = bf(acc);
        }
        for (0..2 * r) |d| {
            var acc: f32 = 0;
            for (0..c) |cc| acc += pv_w[d * c + cc] * bf(scale * codebook[v * c + cc]);
            tsv[v * 2 * r + d] = bf(acc);
        }
    }
    return .{ .tg = tg, .tsv = tsv };
}

/// `_serial_table` longhand (`kernels/dspine.py:211`), transcribed line by line.
fn refSerialTable(
    a: std.mem.Allocator,
    raw: []const f32,
    rms: []const f32,
    query: []const f32,
    cand: []const u32,
    anchors: []const u32,
    lm: []const f32,
    tg: []const f32,
    tsv: []const f32,
    gate_b: []const f32,
    norm_w: []const f32,
    t: usize,
    h: usize,
    r: usize,
    k: usize,
) ![]usize {
    const rows = raw.len / h;
    const n_rows = rows / t * (t - 1);
    const idx = try a.alloc(usize, n_rows * k);
    for (0..n_rows) |n| {
        const b = n / (t - 1);
        const rr_ = n % (t - 1);
        const row = b * (t - 1 + 1) + 1 + rr_;
        const anchor: u32 = anchors[b];

        // base[j] = sum_h LmHead[cand[n][j]][h] * norm_w[h] * raw[row][h]
        var base: [32]f32 = undefined;
        for (0..k) |j| {
            const id = cand[n * k + j];
            var acc: f32 = 0;
            for (0..h) |hh| acc += lm[@as(usize, id) * h + hh] * norm_w[hh] * raw[row * h + hh];
            base[j] = acc;
        }
        for (0..k) |j| {
            const prev_id: u32 = if (n >= 1) cand[(n - 1) * k + j] else 0;
            const sid: u32 = if (rr_ == 0) anchor else prev_id;
            var best: f32 = -std.math.inf(f32);
            var bestk: usize = 0;
            for (0..k) |kc| {
                const id = cand[n * k + kc];
                var acc: f32 = 0;
                for (0..r) |d| {
                    const g = 1.0 / (1.0 + @exp(-(query[row * r + d] +
                        tsv[@as(usize, sid) * 2 * r + d] + gate_b[d])));
                    const m = bf(g * tsv[@as(usize, sid) * 2 * r + r + d]);
                    acc += m * tg[@as(usize, id) * r + d];
                }
                const s = base[kc] + rms[row] * acc;
                if (s > best) {
                    best = s;
                    bestk = kc;
                }
            }
            idx[n * k + j] = bestk;
        }
    }
    return idx;
}

fn fill(a: std.mem.Allocator, n: usize, seed: u64, amp: f32) ![]f32 {
    const out = try a.alloc(f32, n);
    var st: u64 = seed;
    for (out) |*v| {
        st = st *% 6364136223846793005 +% 1442695040888963407;
        const u: f32 = @floatFromInt((st >> 40) & 0xffffff);
        v.* = (u / 16777216.0 * 2.0 - 1.0) * amp;
    }
    return out;
}

test "serial: buildTables matches the longhand MLX formulation" {
    const a = testing.allocator;
    const vocab = 7;
    const h = 8;
    const r = 4;
    const c = 6;

    const lm = try fill(a, vocab * h, 11, 2.0);
    defer a.free(lm);
    const norm_w = try fill(a, h, 12, 1.5);
    defer a.free(norm_w);
    const write_w = try fill(a, h * r, 13, 1.0);
    defer a.free(write_w);
    const pv_w = try fill(a, 2 * r * c, 14, 1.0);
    defer a.free(pv_w);
    const codebook = try fill(a, vocab * c, 15, 1.0);
    defer a.free(codebook);

    var tables = try serial.buildTables(a, .{ .f32 = lm }, norm_w, write_w, pv_w, .{ .f32 = codebook }, vocab, h, r, c);
    defer tables.deinit(a);
    // Rows are built lazily; this test is about the arithmetic, so touch them all.
    for (0..vocab) |v| try tables.ensure(a, v);

    const want = try refBuildTables(a, lm, norm_w, write_w, pv_w, codebook, vocab, h, r, c);
    defer a.free(want.tg);
    defer a.free(want.tsv);

    for (0..vocab) |v| {
        for (0..r) |d| {
            try testing.expectEqual(bits(want.tg[v * r + d]), tables.tg[v * r + d]);
        }
        for (0..2 * r) |d| {
            try testing.expectEqual(bits(want.tsv[v * 2 * r + d]), tables.tsv[v * 2 * r + d]);
        }
    }
}

test "serial: serialTable matches the longhand triton formulation" {
    const a = testing.allocator;
    const t = 4; // block size
    const h = 8;
    const r = 4;
    const blk = 2; // blocks in the batch
    const rows = blk * t;
    const k = 3;
    const vocab = 9;

    const raw = try fill(a, rows * h, 21, 1.0);
    defer a.free(raw);
    const rms = try fill(a, rows, 22, 0.5);
    defer a.free(rms);
    const query = try fill(a, rows * r, 23, 1.0);
    defer a.free(query);
    const gate_b = try fill(a, r, 24, 0.5);
    defer a.free(gate_b);
    const norm_w = try fill(a, h, 25, 1.0);
    defer a.free(norm_w);
    const lm = try fill(a, vocab * h, 26, 1.0);
    defer a.free(lm);
    const c = 5; // code length

    // Candidates: ascending within a row (the `topk_ids` contract), and the
    // two blocks use disjoint id sets so a cross-block index bug shows up.
    const n_rows = blk * (t - 1);
    const cand = try a.alloc(u32, n_rows * k);
    defer a.free(cand);
    for (0..n_rows) |n| {
        for (0..k) |j| {
            const base: u32 = if (n / (t - 1) == 0) 0 else @intCast(vocab - 4);
            cand[n * k + j] = base + @as(u32, @intCast(j));
        }
    }
    const anchors = [_]u32{ 5, 6 };

    // Build the tables through the PRODUCTION path so `serialTable` exercises
    // `ensure` (a test that pre-fills the rows and marks them done would keep
    // passing if the lazy store never ran -- it did, before this was fixed).
    const write_w = try fill(a, h * r, 29, 1.0);
    defer a.free(write_w);
    const pv_w = try fill(a, 2 * r * c, 30, 1.0);
    defer a.free(pv_w);
    const codebook = try fill(a, vocab * c, 31, 1.0);
    defer a.free(codebook);
    const norm_w_cb = norm_w;
    var tables = try serial.buildTables(
        a, .{ .f32 = lm }, norm_w_cb, write_w, pv_w, .{ .f32 = codebook }, vocab, h, r, c,
    );
    defer tables.deinit(a);

    // Oracle: the same two tables from `refBuildTables`, and the expected table
    // from `refSerialTable` fed those -- so a lazy row that is never filled or
    // filled differently now changes the answer.
    const ref = try refBuildTables(a, lm, norm_w, write_w, pv_w, codebook, vocab, h, r, c);
    defer a.free(ref.tg);
    defer a.free(ref.tsv);
    const want = try refSerialTable(
        a, raw, rms, query, cand, &anchors, lm, ref.tg, ref.tsv, gate_b, norm_w, t, h, r, k,
    );
    defer a.free(want);

    const got = try a.alloc(i16, n_rows * k);
    defer a.free(got);

    const w_rows = try a.alloc(f32, h);
    defer a.free(w_rows);
    const scored = try a.alloc(f32, h);
    defer a.free(scored);

    serial.serialTable(
        got, .{ .raw = raw, .rms = rms, .query = query, .rows = rows },
        norm_w, cand, &anchors, .{ .f32 = lm }, &tables, a, gate_b, t, h, k,
        .{ .w_rows = w_rows, .scored = scored },
    );

    for (0..n_rows * k) |i| {
        try testing.expectEqual(@as(i16, @intCast(want[i])), got[i]);
    }
}

test "serial: serialChain walks the table the reference way" {
    // Table, laid out [row][lane].  Row 0's lanes all agree (its sender is the
    // anchor); later rows depend on the lane the previous row picked, which is
    // what makes this a chain and not 3 independent picks.
    const t = 4;
    const k = 3;
    const cand = [_]u32{ 10, 11, 12, 20, 21, 22, 30, 31, 32 };
    const idx = [_]i16{
        1, 1, 1, // row 0: every lane picks candidate 1 -> emit 11
        2, 0, 1, // row 1: lane 1 (what row 0 picked) -> 1
        0, 2, 1, // row 2: lane 1 (what row 1 picked) -> 1
    };
    var out: [3]u32 = undefined;
    serial.serialChain(&out, &idx, &cand, t, k);
    // j=1 -> emit 11; next = idx[1][1] = 0 -> emit 20; next = idx[2][0] = 0.
    try testing.expectEqualSlices(u32, &[_]u32{ 11, 20, 30 }, &out);

    // A different first lane must change the whole tail: that is the property a
    // per-row independent argmax would NOT have.
    const idx2 = [_]i16{
        2, 2, 2, // every lane of row 0 -> candidate 2 -> emit 12
        0, 2, 1, // row 1: lane 2 -> 1
        0, 2, 1, // row 2: lane 2 -> candidate 2
    };
    serial.serialChain(&out, &idx2, &cand, t, k);
    // j=2 -> emit 12; next = idx2[1][2] = 1 -> emit 21; next = idx2[2][1] = 2
    // -> emit 32.
    try testing.expectEqualSlices(u32, &[_]u32{ 12, 21, 32 }, &out);
}
