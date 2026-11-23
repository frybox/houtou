//! Cross-check for the DSPINE code bridge: the C kernels in
//! `src/kernels/x86/dspine.c` against the Zig oracles in `src/spec_ref.zig`.
//!
//! The two are independent transcriptions of the same Triton source
//! (`sglang_dspine/kernels/dspine.py`), so agreement means the transcription
//! is at least self-consistent.  What this file is really guarding is the
//! ELEVEN bf16 ROUNDING POINTS: a missing one shifts a few ulps, which is
//! invisible to any "finite / in range" check and silently degrades draft
//! acceptance instead of breaking anything.
//!
//! Sensitivities that MUST fail here (each was verified by injection):
//!   * dropping the bf16 narrowing in `prepare_read`'s ADD branch
//!   * collapsing `feature_finish`'s three roundings into one
//!   * using `x * (1/(rms+eps))` instead of `x / (rms+eps)` in prepare_read
//!   * normalising the anchor rows in `normalize_codes`
//!   * `prev = row-1` without the `max(...,0)` clamp (row 0 reads OOB)

const std = @import("std");
const ref = @import("spec_ref.zig");
const testing = std.testing;
const gguf = @import("gguf.zig");

const C = struct {
    extern fn q35_f32_to_bf16(x: f32) f32;

    extern fn q35_dspine_prepare_read(x: [*]const f32, residual: ?[*]const f32, raw: [*]f32, normed: [*]f32, rms_out: ?[*]f32, rows: c_long, h: c_long, eps: f32, add: c_int) void;
    extern fn q35_dspine_normalize_codes(projected: [*]const f32, codebook: [*]const f32, anchors: ?[*]const i32, codes: [*]f32, rows: c_long, codes_n: c_long, t: c_long, eps: f32, row0: c_long) void;
    extern fn q35_dspine_feature_message(packed_buf: [*]const f32, packed_prev: ?[*]const f32, bias: [*]const f32, message: [*]f32, rows: c_long, r: c_long, t: c_long, row0: c_long) void;
    extern fn q35_dspine_feature_finish(raw: [*]const f32, delta: [*]const f32, rms: ?[*]const f32, valid: ?[*]const u8, weight: ?[*]const f32, output: [*]f32, normed: ?[*]f32, rows: c_long, h: c_long, t: c_long, norm_eps: f32, write: c_int, norm: c_int, row0: c_long) void;
};

/// Deterministic, wide-dynamic-range fill.  A uniform [0,1) fill would hide
/// rounding bugs: bf16's 8-bit mantissa only loses bits when the exponent is
/// large enough for the low mantissa to matter, so the values have to span
/// several octaves and include negatives.
fn fill(a: std.mem.Allocator, n: usize, seed: u64, scale: f32) ![]f32 {
    var prng = std.Random.DefaultPrng.init(seed);
    const r = prng.random();
    const v = try a.alloc(f32, n);
    for (v) |*x| {
        const u = r.float(f32) * 2.0 - 1.0;
        // exponent jitter: multiply by 2^(k) for k in about [-8, 8]
        const k = r.float(f32) * 16.0 - 8.0;
        x.* = u * scale * std.math.pow(f32, 2.0, k);
    }
    return v;
}

test "dspine: f32->bf16 matches Zig's RNE cast" {
    var prng = std.Random.DefaultPrng.init(7);
    const r = prng.random();
    for (0..20000) |_| {
        const x = (r.float(f32) * 2.0 - 1.0) * std.math.pow(f32, 2.0, r.float(f32) * 60.0 - 30.0);
        try testing.expectEqual(ref.roundBf16(x), C.q35_f32_to_bf16(x));
    }
    // The exact ties and the special values that a naive truncation gets wrong.
    for ([_]f32{ 0.0, -0.0, 1.0, -1.0, 1.0 / 3.0, 1e-38, 3.3895314e38 }) |x| {
        try testing.expectEqual(ref.roundBf16(x), C.q35_f32_to_bf16(x));
    }
}

test "dspine: prepare_read with and without residual add" {
    const a = testing.allocator;
    const rows = 17; // deliberately not a multiple of T
    const h = 512; // narrower than 5120: the arithmetic is per-element
    const eps: f32 = 1e-6;

    const x = try fill(a, rows * h, 11, 0.5);
    defer a.free(x);
    const res = try fill(a, rows * h, 12, 0.5);
    defer a.free(res);

    for ([_]bool{ false, true }) |add| {
        const want = try ref.prepareRead(a, x, if (add) res else null, rows, h, eps);
        defer a.free(want.raw);
        defer a.free(want.normed);
        defer a.free(want.rms);

        const got_raw = try a.alloc(f32, rows * h);
        defer a.free(got_raw);
        const got_normed = try a.alloc(f32, rows * h);
        defer a.free(got_normed);
        const got_rms = try a.alloc(f32, rows);
        defer a.free(got_rms);

        C.q35_dspine_prepare_read(x.ptr, if (add) res.ptr else null, got_raw.ptr, got_normed.ptr, got_rms.ptr, rows, h, eps, @intFromBool(add));

        // BIT-exact, not approximate: the whole design of this kernel is that
        // every operation has a defined rounding.
        try testing.expectEqualSlices(f32, want.raw, got_raw);
        try testing.expectEqualSlices(f32, want.normed, got_normed);
        try testing.expectEqualSlices(f32, want.rms, got_rms);
    }
}

test "dspine: normalize_codes keeps anchor rows un-normalised" {
    const a = testing.allocator;
    const rows = 16;
    const c = 1024;
    const t = 16;
    const eps: f32 = 1e-6;

    const projected = try fill(a, rows * c, 21, 0.3);
    defer a.free(projected);
    const codebook = try fill(a, 64 * c, 22, 0.5);
    defer a.free(codebook);
    const anchors = [_]u32{ 3, 5 }; // row 0 is anchor -> codebook[3]

    const want = try ref.normalizeCodes(a, projected, codebook, &anchors, rows, c, t, eps, 0);
    defer a.free(want);
    const got = try a.alloc(f32, rows * c);
    defer a.free(got);

    var ai: [2]i32 = .{ 3, 5 };
    C.q35_dspine_normalize_codes(projected.ptr, codebook.ptr, &ai, got.ptr, rows, c, t, eps, 0);
    try testing.expectEqualSlices(f32, want, got);

    // The anchor row must equal `bf16(codebook[3] * sqrt(C))` exactly -- a
    // version that normalises every row (the natural "simplification") differs
    // here, and so does one that skips the bf16 narrowing of the product.
    const scale = @sqrt(@as(f32, @floatFromInt(c)));
    for (0..c) |k| {
        try testing.expectEqual(ref.roundBf16(codebook[3 * c + k] * scale), got[k]);
    }
    // ...and a non-anchor row must NOT equal its projection.
    try testing.expect(!std.mem.eql(f32, projected[c..][0..c], got[c..][0..c]));
}

test "dspine: feature_message gate/prev wiring" {
    const a = testing.allocator;
    const rows = 16;
    const r = 512;
    const t = 16;

    const packed_buf = try fill(a, rows * 3 * r, 31, 1.0);
    defer a.free(packed_buf);
    const bias = try fill(a, r, 32, 0.1);
    defer a.free(bias);

    const want = try ref.featureMessage(a, packed_buf, bias, rows, r, t, 0, null);
    defer a.free(want);
    const got = try a.alloc(f32, rows * r);
    defer a.free(got);
    C.q35_dspine_feature_message(packed_buf.ptr, null, bias.ptr, got.ptr, rows, r, t, 0);
    try testing.expectEqualSlices(f32, want, got);

    // Row 0 is an anchor: all zero, despite `prev = 0` reading row 0's own
    // sender/value (a `prev = row-1` without the clamp would read before the
    // buffer and, on a good day, produce garbage instead of crashing).
    for (got[0..r]) |v| try testing.expectEqual(@as(f32, 0.0), v);

    // Row 3 must consume row 2's sender/value slots, not row 3's.  Swapping
    // the offsets is the single most likely transcription slip here and is
    // silent otherwise (both slots hold same-distribution numbers).
    const k: usize = 5;
    const query = packed_buf[3 * 3 * r + k];
    const sender = packed_buf[2 * 3 * r + r + k];
    const value = packed_buf[2 * 3 * r + 2 * r + k];
    const gate = 1.0 / (1.0 + @exp(-(query + sender + bias[k])));
    // `message` is bf16 in the reference (bridge.py:139).
    try testing.expectEqual(ref.roundBf16(gate * value), got[3 * r + k]);
}

test "dspine: feature_finish rounds three times on the NORM path" {
    const a = testing.allocator;
    const rows = 16;
    const h = 512;
    const t = 16;
    const norm_eps: f32 = 1e-6;

    const raw = try fill(a, rows * h, 41, 1.0);
    defer a.free(raw);
    const delta = try fill(a, rows * h, 42, 1.0);
    defer a.free(delta);
    const rms = try fill(a, rows, 43, 1.0);
    defer a.free(rms);
    const weight = try fill(a, h, 44, 1.0);
    defer a.free(weight);
    const valid = try a.alloc(bool, rows);
    defer a.free(valid);
    @memset(valid, true);
    valid[5] = false; // one invalid row: its edge must collapse

    const want = try ref.featureFinish(a, raw, delta, rms, valid, weight, rows, h, t, norm_eps, true, true);
    defer a.free(want.output);
    defer a.free(want.normed);

    const got_out = try a.alloc(f32, rows * h);
    defer a.free(got_out);
    const got_normed = try a.alloc(f32, rows * h);
    defer a.free(got_normed);
    const valid_u8 = try a.alloc(u8, rows);
    defer a.free(valid_u8);
    for (valid, 0..) |v, i| valid_u8[i] = @intFromBool(v);

    C.q35_dspine_feature_finish(raw.ptr, delta.ptr, rms.ptr, valid_u8.ptr, weight.ptr, got_out.ptr, got_normed.ptr, rows, h, t, norm_eps, 1, 1, 0);
    try testing.expectEqualSlices(f32, want.output, got_out);
    try testing.expectEqualSlices(f32, want.normed, got_normed);

    // Anchor row 0: no update, so output == bf16(raw).
    const r0: f32 = raw[0];
    try testing.expectEqual(ref.roundBf16(r0), got_out[0]);
    // Row 5 is marked invalid: also no update.
    try testing.expectEqual(ref.roundBf16(raw[5 * h]), got_out[5 * h]);
    // A normal row DOES move.
    try testing.expect(got_out[h] != ref.roundBf16(raw[h]));
}

test "dspine: bf16 matvec on the real draft file" {
    // The draft's weights are bf16, a family no other test in this repo feeds
    // a real model file, so it is worth pinning down.  Two categorical checks:
    //
    //   1. every element of a row is the bf16->f32 left-shift of the raw bytes
    //      (`dotBf16F32`) and `dequantRow` agree bit-for-bit;
    //   2. a one-hot activation reproduces that single weight element exactly,
    //      which pins the row stride and the element order inside the kernel.
    //
    // The full-dot error is NOT usable as a criterion here: these projections
    // cancel hard (|sum| << L1), so a legitimate tree reduction differs from a
    // sequential f32 sum by ~1e-4 relative, and an exponent-jittered input
    // makes that 1e-3.  That is f32 conditioning, not a kernel defect.
    const a = testing.allocator;
    const path = "/mnt/Qwen3.8-27B-GGUF/dspine/Qwen3.8-27B-DSpine-b16-Q8_0.gguf";
    const io = testing.io;
    const file = std.Io.Dir.cwd().openFile(io, path, .{}) catch return error.SkipZigTest;
    defer file.close(io);
    const size = try file.stat(io);
    var mm = try std.Io.File.MemoryMap.create(io, file, .{
        .len = @intCast(size.size),
        .protection = .{ .read = true },
    });
    defer mm.destroy(io);
    var f = try gguf.parse(a, mm.memory);
    defer f.deinit();

    var cpu = @import("backend_cpu.zig").CpuBackend.init(a, &f);
    defer cpu.deinit();
    const exec = cpu.exec();

    const w = try exec.loadLinear("dspine_gate.weight");
    const q = w.quant;
    try testing.expectEqual(gguf.TensorType.bf16, q.ttype);
    try testing.expectEqual(@as(usize, 512), q.rows);
    try testing.expectEqual(@as(usize, 1024), q.cols());

    const cols = q.cols();
    const out = try a.alloc(f32, q.rows);
    defer a.free(out);
    try exec.allocScratch(cols);

    // (1) element decode, on a row that is not the first (row stride matters)
    const probe_row: usize = 401;
    const wide = try a.alloc(f32, cols);
    defer a.free(wide);
    // `Exec.dequantRow` writes exactly one 256-element row-block -- at 1024 wide
    // that is a quarter of the row (AGENTS.md 5.2), so use the reference
    // dequantiser, which walks `blocks_per_row`.
    const deqWide = struct {
        fn into(dst: []f32, qq: @TypeOf(q), r: usize) void {
            const rbe = qq.rowBlockElems();
            const blk = qq.blockBytes();
            const per_rb = rbe / qq.blockElems();
            const src = qq.row(r);
            var b: usize = 0;
            while (b < qq.blocks_per_row) : (b += 1) {
                @import("backend_cpu.zig").dequantRowFnFor(qq.ttype).?(src[b * per_rb * blk ..], dst[b * rbe ..][0..rbe]);
            }
        }
    };
    deqWide.into(wide, q, probe_row);
    const raw = q.row(probe_row);
    for (0..cols) |k| {
        const h = std.mem.readInt(u16, raw[k * 2 ..][0..2], .little);
        try testing.expectEqual(@as(f32, @bitCast(@as(u32, h) << 16)), wide[k]);
    }

    // (2) one-hot: the dot must return exactly one weight element.  Probing
    // every 32-element boundary of the row plus the two ends catches any
    // off-by-one in the element or row stride.
    const onehot = try a.alloc(f32, cols);
    defer a.free(onehot);
    @memset(onehot, 0);
    for (0..cols / 32) |blk| {
        for ([_]usize{ 0, 7, 31 }) |off| {
            const probe = blk * 32 + off;
            onehot[probe] = 1.0;
            exec.matvec(.{}, .{ .ptr = out.ptr, .len = q.rows }, w, .{ .ptr = onehot.ptr, .len = cols });
            for ([_]usize{ 0, probe_row, q.rows - 1 }) |r| {
                deqWide.into(wide, q, r);
                const expect = wide[probe];
                try testing.expectEqual(expect, out[r]);
            }
            onehot[probe] = 0.0;
        }
    }
}

test "dspine: single-row bridge calls match a whole-block call" {
    // The bridge runs one row at a time (`rows=1`, `row0=row`) while the
    // reference kernels are written for a whole block.  Feeding row `r` alone
    // must give byte-identical results to feeding the block and reading row
    // `r` -- otherwise a stray `row * ...` index (which is what the first
    // version of these kernels had) reads and writes past the caller's
    // buffer, silently corrupting whatever follows it.  `packed_prev` carries
    // the predecessor row that a single-row call cannot see.
    const a = testing.allocator;
    const rows = 16;
    const r = 64;
    const c = 96;
    const t = 16;

    const packed_buf = try fill(a, rows * 3 * r, 71, 1.0);
    defer a.free(packed_buf);
    const bias = try fill(a, r, 72, 1.0);
    defer a.free(bias);
    const projected = try fill(a, rows * c, 73, 1.0);
    defer a.free(projected);
    const codebook = try fill(a, 4 * c, 74, 1.0);
    defer a.free(codebook);
    const anchors = [_]u32{ 0, 1, 2, 3 };
    const anchors_u = anchors;

    // Whole-block reference.
    const want_msg = try ref.featureMessage(a, packed_buf, bias, rows, r, t, 0, null);
    defer a.free(want_msg);
    const want_codes = try ref.normalizeCodes(a, projected, codebook, &anchors_u, rows, c, t, 1e-6, 0);
    defer a.free(want_codes);

    for (1..rows) |row| {
        const prev = try a.alloc(f32, 3 * r);
        defer a.free(prev);
        @memcpy(prev, packed_buf[(row - 1) * 3 * r ..][0 .. 3 * r]);
        const got = try a.alloc(f32, r);
        defer a.free(got);
        C.q35_dspine_feature_message(
            packed_buf.ptr + row * 3 * r,
            prev.ptr,
            bias.ptr,
            got.ptr,
            1,
            r,
            t,
            @intCast(row),
        );
        try testing.expectEqualSlices(f32, want_msg[row * r ..][0..r], got);

        const got_codes = try a.alloc(f32, c);
        defer a.free(got_codes);
        // `anchors` holds one entry per anchor row inside the block.
        const one_anchor = [_]i32{@intCast(anchors[row / t])};
        C.q35_dspine_normalize_codes(
            projected.ptr + row * c,
            codebook.ptr,
            &one_anchor,
            got_codes.ptr,
            1,
            c,
            t,
            1e-6,
            @intCast(row),
        );
        try testing.expectEqualSlices(f32, want_codes[row * c ..][0..c], got_codes);
    }
}
