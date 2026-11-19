//! Zig scalar oracles for the DSPINE code bridge (`kernels/x86/dspine.c`).
//!
//! These are written from the REFERENCE (sglang_dspine/kernels/dspine.py), not
//! from the C unit, so `zig build test` compares two independent transcriptions
//! of the same Triton source rather than a copy against a copy.  That is the
//! only way a bf16 rounding point can be caught: getting one wrong shifts a
//! handful of ulps, which no "is it finite / is it in range" test sees.
//!
//! Deliberately slow and literal.

const std = @import("std");

/// The reduction order shared by all three of our implementations (this oracle,
/// `kernels/x86/dspine.c` `q35_dspine_sum_sq`, and `kernels/cuda/dspine.cu`
/// `block_reduce`).
///
/// This is the ONE thing these oracles do NOT derive independently, and that is
/// deliberate.  The reference sums with `tl.sum`, whose lowering depends on
/// Triton's thread layout for the target; that order changes between Triton
/// releases and between GPU architectures, so it is not a stable property to
/// test against -- only the bf16 cast points and the expression order are.
///
/// The contract is therefore: ELEMENTWISE arithmetic and every bf16 cast is
/// checked independently against the reference (see the callers of
/// `roundBf16`), while the reduction follows one shared tree so the three
/// implementations can be compared bit-for-bit with each other.  Mirroring the
/// CUDA `block_reduce`: 256 "threads", each accumulating its strided slice in
/// f32, then a butterfly fold over 32 lanes per warp and again over the 8 warps
/// (unused lanes contributing +0.0).
fn fold32(v: [32]f32) f32 {
    var w = v;
    var off: usize = 16;
    while (off > 0) : (off >>= 1) {
        for (0..off) |lane| w[lane] += w[lane + off];
    }
    return w[0];
}

/// `sum(v[i]^2)` in the shared tree order.
fn sumSq(v: []const f32) f32 {
    const threads = 256;
    const warps = threads / 32;
    var warp: [warps]f32 = undefined;
    for (0..warps) |wid| {
        var lane_vals: [32]f32 = undefined;
        for (0..32) |lane| {
            const tid = wid * 32 + lane;
            var acc: f32 = 0;
            var i = tid;
            while (i < v.len) : (i += threads) acc += v[i] * v[i];
            lane_vals[lane] = acc;
        }
        warp[wid] = fold32(lane_vals);
    }
    var wide: [32]f32 = undefined;
    for (0..32) |i| wide[i] = if (i < warps) warp[i] else 0.0;
    return fold32(wide);
}

/// f32 -> bf16 -> f32, round-to-nearest-even.
///
/// Zig 0.17 has no `bf16` type, so this is the bit-level RNE that torch's
/// `.to(torch.bfloat16)` performs: add a rounding bias built from the bits being
/// discarded (so an exact tie goes to even), then truncate.  Agreeing with the
/// maths, not with the C unit's spelling, is the point -- `dspine_tests.zig`
/// compares the two on 20000 random values plus the tie cases.
pub fn roundBf16(x: f32) f32 {
    const u: u32 = @bitCast(x);
    // NaN / inf pass through untouched: adding a bias to inf overflows the
    // exponent, and the reference never feeds them here.
    if (!std.math.isFinite(x)) return x;
    const bias: u32 = 0x7FFF + ((u >> 16) & 1);
    const r: u32 = (u +% bias) & 0xFFFF_0000;
    return @bitCast(r);
}

pub const ReadResult = struct {
    /// [rows*H] pre-norm values (after the optional residual add)
    raw: []f32,
    /// [rows*H] `raw / (rms + eps)`
    normed: []f32,
    /// [rows] the rms of each row
    rms: []f32,
};

/// `prepare_read` (kernels/dspine.py:11).
pub fn prepareRead(
    a: std.mem.Allocator,
    x: []const f32,
    residual: ?[]const f32,
    rows: usize,
    h: usize,
    eps: f32,
) !ReadResult {
    const raw = try a.alloc(f32, rows * h);
    errdefer a.free(raw);
    const normed = try a.alloc(f32, rows * h);
    errdefer a.free(normed);
    const rms = try a.alloc(f32, rows);

    const flt_min_normal: f32 = @bitCast(@as(u32, 0x0080_0000));
    for (0..rows) |row| {
        const src = x[row * h ..][0..h];
        const dst = raw[row * h ..][0..h];
        for (src, 0..) |v, i| {
            // ALWAYS narrowed (dspine_mlx/bridge.py:112), not only when adding.
            const summed = if (residual) |r| v + r[row * h + i] else v;
            dst[i] = roundBf16(summed);
        }
        const mean: f32 = sumSq(dst) / @as(f32, @floatFromInt(h));
        // `tl.maximum(mean, 1.1754943508222875e-38)`
        const clipped = @max(mean, flt_min_normal);
        const r = @sqrt(clipped);
        rms[row] = r;
        const denom = r + eps;
        for (dst, 0..) |v, i| normed[row * h + i] = v / denom;
    }
    return .{ .raw = raw, .normed = normed, .rms = rms };
}

/// `normalize_codes` (kernels/dspine.py:41).  Anchor rows (`row % T == 0`)
/// bypass normalisation and read the codebook directly.
pub fn normalizeCodes(
    a: std.mem.Allocator,
    projected: []const f32,
    codebook: []const f32,
    anchors: []const u32,
    rows: usize,
    c: usize,
    t: usize,
    eps: f32,
    row0: usize,
) ![]f32 {
    const codes = try a.alloc(f32, rows * c);
    const scale = @sqrt(@as(f32, @floatFromInt(c)));
    for (0..rows) |i| {
        // `row` is the position INSIDE THE BLOCK (it drives the anchor test);
        // every buffer offset walks with `i`, because the caller's arrays only
        // hold `rows` rows.  `anchors[i]` likewise indexes this call's rows.
        const row = row0 + i;
        const dst = codes[i * c ..][0..c];
        if (t > 0 and row % t == 0) {
            const tok = anchors[i];
            const src = codebook[@as(usize, tok) * c ..][0..c];
            for (src, 0..) |v, k| dst[k] = roundBf16(v * scale);
            continue;
        }
        const src = projected[i * c ..][0..c];
        const norm: f32 = @sqrt(sumSq(src));
        for (src, 0..) |v, k| dst[k] = roundBf16(scale * v / (norm + eps));
    }
    return codes;
}

/// `feature_message` (kernels/dspine.py:63).
pub fn featureMessage(
    a: std.mem.Allocator,
    packed_buf: []const f32,
    bias: []const f32,
    rows: usize,
    r: usize,
    t: usize,
    row0: usize,
    packed_prev: ?[]const f32,
) ![]f32 {
    const msg = try a.alloc(f32, rows * r);
    for (0..rows) |i| {
        const row = row0 + i;
        for (0..r) |k| {
            // The predecessor's packed triple: the previous row of THIS call,
            // or `packed_prev` when the caller feeds one row at a time, or the
            // row itself on row 0 (the reference clamps, and the mask zeroes).
            const prow: []const f32 = if (i > 0)
                packed_buf[(i - 1) * 3 * r ..]
            else if (packed_prev) |p|
                p
            else
                packed_buf;
            const query = packed_buf[i * 3 * r + k];
            const sender = prow[r + k];
            const value = prow[2 * r + k];
            const gate = 1.0 / (1.0 + @exp(-(query + sender + bias[k])));
            msg[i * r + k] = if (row % t == 0) 0.0 else roundBf16(gate * value);
        }
    }
    return msg;
}

pub const FinishResult = struct {
    /// [rows*H] the updated residual stream, already bf16-rounded
    output: []f32,
    /// [rows*H] `norm(output) * weight`, or empty when `norm` is false
    normed: []f32,
};

/// `feature_finish` (kernels/dspine.py:85).
pub fn featureFinish(
    a: std.mem.Allocator,
    raw: []const f32,
    delta: []const f32,
    rms: []const f32,
    valid: ?[]const bool,
    weight: ?[]const f32,
    rows: usize,
    h: usize,
    t: usize,
    norm_eps: f32,
    write: bool,
    norm: bool,
) !FinishResult {
    const out = try a.alloc(f32, rows * h);
    errdefer a.free(out);
    const normed = if (norm) try a.alloc(f32, rows * h) else try a.alloc(f32, 0);
    errdefer a.free(normed);

    for (0..rows) |row| {
        const prev = if (row > 0) row - 1 else 0;
        var edge = row % t != 0;
        if (valid) |v| edge = edge and v[row] and v[prev];
        const update_scale: f32 = if (edge and write) rms[row] else 0.0;
        for (0..h) |i| {
            out[row * h + i] = roundBf16(raw[row * h + i] + update_scale * delta[row * h + i]);
        }
        if (!norm) continue;
        const variance: f32 = sumSq(out[row * h ..][0..h]) / @as(f32, @floatFromInt(h));
        const inv = 1.0 / @sqrt(variance + norm_eps);
        for (0..h) |i| {
            const y = roundBf16(out[row * h + i] * inv);
            normed[row * h + i] = roundBf16(y * (if (weight) |w| w[i] else 1.0));
        }
    }
    return .{ .output = out, .normed = normed };
}
