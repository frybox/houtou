//! Cross-checks for KV-cache type support (`--kv-type f32|f16|q8_0|q4_0`).
//!
//! KV rows are stored in the WEIGHT layouts, so the C kernels
//! (src/kernels/x86/kv.c) must match the weight-side oracles
//! (quant_dequant.zig) exactly:
//!   * quantize: C bytes vs the Zig `quantizeQ4_0`/`quantizeQ8_0` — the
//!     q4_0 pair is written expression-for-expression and must be
//!     byte-identical; the q8_0 pair differs in one historical detail (C
//!     multiplies by the reciprocal, Zig divides), so its quants may differ
//!     by 1 while the f16 scale bytes stay bit-identical.
//!   * dequant: C vs the weight-side `dequantQ4_0`/`dequantizeQ8_0` on the
//!     SAME bytes — bit-identical (this is what pins the KV layout to the
//!     weight layout, including the q4_0 j / 16+j nibble packing).
//!   * attention: `backend_cpu.attnDecodeKv` (dequant-on-read over the byte
//!     cache) vs `ew.bufAttnDecode` over the C-dequantized rows — bit-identical
//!     — and vs the f32 reference — within the per-format element error bound.
//!   * `kvAppendRows` row addressing: only the addressed rows change.

const std = @import("std");
const backend = @import("backend.zig");
const backend_cpu = @import("backend_cpu.zig");
const quant = @import("quant_dequant.zig");
const ew = @import("elementwise.zig");
const c_kernels = @import("c_kernels.zig");
const Buf = backend.Buf;

const N: usize = 256; // 8 x 32-elem blocks: dequantQ4_0 / dequantizeQ8_0 take 256

fn randomRow(alloc: std.mem.Allocator, rand: std.Random, n: usize) ![]f32 {
    const x = try alloc.alloc(f32, n);
    for (x) |*v| v.* = rand.float(f32) * 3.0 - 1.5;
    return x;
}

fn maxAbs(x: []const f32) f32 {
    var m: f32 = 0;
    for (x) |v| m = @max(m, @abs(v));
    return m;
}

test "kv q8_0: C quantize/dequant match the weight-side oracles" {
    const alloc = std.testing.allocator;
    var prng = std.Random.DefaultPrng.init(7);
    const rand = prng.random();

    const x = try randomRow(alloc, rand, N);
    defer alloc.free(x);
    const cbytes = try alloc.alloc(u8, N / 32 * 34);
    defer alloc.free(cbytes);
    const zbytes = try alloc.alloc(u8, N / 32 * 34);
    defer alloc.free(zbytes);

    c_kernels.quantizeQ8_0(N, x.ptr, cbytes.ptr);
    try quant.quantizeQ8_0(x, zbytes);

    // f16 scale bytes: both are the RN f32->f16 cast of amax/127 —
    // bit-identical.
    for (0..8) |ib| {
        const b = ib * 34;
        try std.testing.expectEqual(cbytes[b], zbytes[b]);
        try std.testing.expectEqual(cbytes[b + 1], zbytes[b + 1]);
        // quants: C rounds x*inv, Zig rounds x/d32 — a last-ulp difference
        // may flip one LSB, nothing more.
        for (0..32) |k| {
            const dq = @as(i32, @intCast(@as(i8, @bitCast(cbytes[b + 2 + k])))) -
                @as(i32, @intCast(@as(i8, @bitCast(zbytes[b + 2 + k]))));
            try std.testing.expect(@abs(dq) <= 1);
        }
    }

    // C dequant vs the weight-side dequant on the SAME C bytes — bit-identical.
    const cq = try alloc.alloc(f32, N);
    defer alloc.free(cq);
    c_kernels.kvDequantQ8_0(N, cbytes.ptr, cq.ptr);
    const zq = try alloc.alloc(f32, N);
    defer alloc.free(zq);
    try quant.dequantizeQ8_0(cbytes, zq);
    try std.testing.expect(std.mem.eql(f32, cq, zq));

    // Round trip: |deq - x| <= d32/2 + 127*|d_f16 - d32| per block.
    for (0..8) |ib| {
        const b = ib * 34;
        const d: f16 = @bitCast(std.mem.readInt(u16, cbytes[b..][0..2], .little));
        const df: f32 = @floatCast(d);
        const d32: f32 = if (maxAbs(x[ib * 32 ..][0..32]) > 0)
            maxAbs(x[ib * 32 ..][0..32]) / 127.0
        else
            1.0;
        const bound = d32 / 2 + 127 * @abs(df - d32);
        for (0..32) |k| {
            const i = ib * 32 + k;
            try std.testing.expect(@abs(cq[i] - x[i]) <= bound);
        }
    }
}

test "kv q4_0: C quantize/dequant match the weight-side oracles" {
    const alloc = std.testing.allocator;
    var prng = std.Random.DefaultPrng.init(11);
    const rand = prng.random();

    const x = try randomRow(alloc, rand, N);
    defer alloc.free(x);
    const cbytes = try alloc.alloc(u8, N / 32 * 18);
    defer alloc.free(cbytes);
    const zbytes = try alloc.alloc(u8, N / 32 * 18);
    defer alloc.free(zbytes);

    c_kernels.kvQuantQ4_0(N, x.ptr, cbytes.ptr);
    try quant.quantizeQ4_0(x, zbytes);

    // Expression-for-expression pair: byte-identical.
    try std.testing.expectEqualSlices(u8, cbytes, zbytes);

    // C dequant vs the WEIGHT-side dequantQ4_0 on the SAME bytes — this is
    // the layout pin (18 B blocks, elem j low nibble / elem 16+j high nibble).
    const cq = try alloc.alloc(f32, N);
    defer alloc.free(cq);
    c_kernels.kvDequantQ4_0(N, cbytes.ptr, cq.ptr);
    const zq = try alloc.alloc(f32, N);
    defer alloc.free(zq);
    quant.dequantQ4_0(cbytes, zq);
    try std.testing.expect(std.mem.eql(f32, cq, zq));

    // Round trip: |deq - x| <= d32 + 8*|d_f16 - d32| per block.  The amax
    // element itself lands on the clamp (round(x*inv)+8 == 16 -> 15), so its
    // error is ~d32, not the interior d32/2.
    for (0..8) |ib| {
        const b = ib * 18;
        const d: f16 = @bitCast(std.mem.readInt(u16, cbytes[b..][0..2], .little));
        const df: f32 = @floatCast(d);
        const d32: f32 = if (maxAbs(x[ib * 32 ..][0..32]) > 0)
            maxAbs(x[ib * 32 ..][0..32]) / 8.0
        else
            1.0;
        const bound = d32 + 8 * @abs(df - d32);
        for (0..32) |k| {
            const i = ib * 32 + k;
            try std.testing.expect(@abs(cq[i] - x[i]) <= bound);
        }
    }
}

test "kv f16: C quantize/dequant is the exact f32->f16 cast" {
    const alloc = std.testing.allocator;
    var prng = std.Random.DefaultPrng.init(23);
    const rand = prng.random();

    const x = try randomRow(alloc, rand, N);
    defer alloc.free(x);
    const cbytes = try alloc.alloc(u8, N * 2);
    defer alloc.free(cbytes);
    c_kernels.kvQuantF16(N, x.ptr, cbytes.ptr);

    for (0..N) |i| {
        const h: f16 = @bitCast(std.mem.readInt(u16, cbytes[2 * i..][0..2], .little));
        try std.testing.expect(std.mem.eql(u16, &.{@bitCast(h)}, &.{@bitCast(@as(f16, @floatCast(x[i])))}));
    }

    // Dequant is the exact inverse: f16 -> f32.
    const cq = try alloc.alloc(f32, N);
    defer alloc.free(cq);
    c_kernels.kvDequantF16(N, cbytes.ptr, cq.ptr);
    for (0..N) |i| {
        const h: f16 = @bitCast(std.mem.readInt(u16, cbytes[2 * i..][0..2], .little));
        try std.testing.expect(std.mem.eql(f32, &.{cq[i]}, &.{@as(f32, @floatCast(h))}));
        // half-ulp: |deq - x| <= 2^-11 * max(|x|, |deq|)
        const bound = @max(@abs(x[i]), @abs(cq[i])) * (1.0 / 2048.0);
        try std.testing.expect(@abs(cq[i] - x[i]) <= bound);
    }
}

fn kvAttnTest(alloc: std.mem.Allocator, t: backend.KvType, rel_bound: f32) !void {
    // heads=8, gqa=2 -> kv_heads=4, hd=64 -> kv_dim=256 (= N above, so the
    // weight-side dequants can read whole rows).
    const heads: usize = 8;
    const gqa: usize = 2;
    const hd: usize = 64;
    const kv_dim: usize = 256;
    const slot: usize = 5; // rows 0..5 live
    const max_rows: usize = 8; // two extra rows: row addressing must not spill
    const scale: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(hd)));

    var prng = std.Random.DefaultPrng.init(31);
    const rand = prng.random();

    const q = try randomRow(alloc, rand, heads * hd);
    defer alloc.free(q);
    const gate = try randomRow(alloc, rand, heads * hd);
    defer alloc.free(gate);
    const k_f32 = try randomRow(alloc, rand, max_rows * kv_dim);
    defer alloc.free(k_f32);
    const v_f32 = try randomRow(alloc, rand, max_rows * kv_dim);
    defer alloc.free(v_f32);
    const att = try alloc.alloc(f32, max_rows);
    defer alloc.free(att);
    const gout_ref = try alloc.alloc(f32, heads * hd);
    defer alloc.free(gout_ref);
    const gout_q = try alloc.alloc(f32, heads * hd);
    defer alloc.free(gout_q);
    const k_deq = try alloc.alloc(f32, (slot + 1) * kv_dim);
    defer alloc.free(k_deq);
    const v_deq = try alloc.alloc(f32, (slot + 1) * kv_dim);
    defer alloc.free(v_deq);
    const row_bytes: usize = backend.kvRowBytes(kv_dim, t);
    const flen: usize = (max_rows * row_bytes + 3) / 4;
    const kbytes = try alloc.alloc(u8, 4 * flen);
    defer alloc.free(kbytes);
    const vbytes = try alloc.alloc(u8, 4 * flen);
    defer alloc.free(vbytes);
    const kview: []f32 = @as([*]f32, @ptrCast(@alignCast(kbytes.ptr)))[0..flen];
    const vview: []f32 = @as([*]f32, @ptrCast(@alignCast(vbytes.ptr)))[0..flen];
    var scratch = backend_cpu.KvDequantScratch{};
    defer {
        if (scratch.k.len > 0) alloc.free(scratch.k);
        if (scratch.v.len > 0) alloc.free(scratch.v);
    }
    try scratch.ensure(alloc, max_rows * kv_dim);

    // f32 reference over the original rows.
    ew.bufAttnDecode(Buf.wrap(q), Buf.wrap(gate), Buf.wrap(k_f32), Buf.wrap(v_f32),
        Buf.wrap(att), Buf.wrap(gout_ref), slot, heads, gqa, hd, kv_dim, scale);

    // Write rows 0..max_rows-1 through kvAppendRows (the vKvAppend body).
    backend_cpu.kvAppendRows(t, Buf.wrap(kview), Buf.wrap(k_f32), 0, kv_dim, max_rows);
    backend_cpu.kvAppendRows(t, Buf.wrap(vview), Buf.wrap(v_f32), 0, kv_dim, max_rows);

    // Dequant-on-read attention over the byte cache.
    backend_cpu.attnDecodeKv(t, Buf.wrap(q), Buf.wrap(gate), kbytes.ptr, vbytes.ptr,
        Buf.wrap(att), Buf.wrap(gout_q), slot, heads, gqa, hd, kv_dim, scale, &scratch);

    // Exact twin: C-dequant rows 0..slot, then the f32 oracle over them.  Same
    // code path and same dequant as attnDecodeKv, so bit-identical.
    for (0..slot + 1) |r| {
        switch (t) {
            .f32 => unreachable,
            .f16 => {
                c_kernels.kvDequantF16(kv_dim, kbytes.ptr + r * row_bytes, k_deq[r * kv_dim..].ptr);
                c_kernels.kvDequantF16(kv_dim, vbytes.ptr + r * row_bytes, v_deq[r * kv_dim..].ptr);
            },
            .q8_0 => {
                c_kernels.kvDequantQ8_0(kv_dim, kbytes.ptr + r * row_bytes, k_deq[r * kv_dim..].ptr);
                c_kernels.kvDequantQ8_0(kv_dim, vbytes.ptr + r * row_bytes, v_deq[r * kv_dim..].ptr);
            },
            .q4_0 => {
                c_kernels.kvDequantQ4_0(kv_dim, kbytes.ptr + r * row_bytes, k_deq[r * kv_dim..].ptr);
                c_kernels.kvDequantQ4_0(kv_dim, vbytes.ptr + r * row_bytes, v_deq[r * kv_dim..].ptr);
            },
        }
    }
    const gout_exact = try alloc.alloc(f32, heads * hd);
    defer alloc.free(gout_exact);
    ew.bufAttnDecode(Buf.wrap(q), Buf.wrap(gate), Buf.wrap(k_deq), Buf.wrap(v_deq),
        Buf.wrap(att), Buf.wrap(gout_exact), slot, heads, gqa, hd, kv_dim, scale);
    try std.testing.expect(std.mem.eql(f32, gout_q, gout_exact));

    // vs the f32 reference: within the format's element error bound.
    const mv = maxAbs(v_f32);
    for (gout_q, gout_ref) |a, b| {
        try std.testing.expect(@abs(a - b) <= rel_bound * mv);
    }
}

test "kv attn: f16 KV vs f32 reference" {
    const alloc = std.testing.allocator;
    try kvAttnTest(alloc, .f16, 2e-3);
}

// Format-fidelity bounds vs the f32 reference (the bit-exact twin check is
// the correctness pin; these catch a layout or quantize bug, which would move
// the output by O(mv), not O(format error)).  Measured worst on this shape:
// f16 1.3e-4, q8_0 2.2e-3, q4_0 4.8e-2 (rel to max|v|).
test "kv attn: q8_0 KV vs f32 reference" {
    const alloc = std.testing.allocator;
    try kvAttnTest(alloc, .q8_0, 1e-2);
}

test "kv attn: q4_0 KV vs f32 reference" {
    const alloc = std.testing.allocator;
    try kvAttnTest(alloc, .q4_0, 1.2e-1);
}

test "kvAppendRows: only the addressed rows change" {
    const alloc = std.testing.allocator;
    var prng = std.Random.DefaultPrng.init(47);
    const rand = prng.random();

    const kv_dim: usize = 128;
    const rows: usize = 6;
    const slot: usize = 2;
    const n: usize = 3; // rows 2,3,4
    const src = try randomRow(alloc, rand, n * kv_dim);
    defer alloc.free(src);

    const types: [4]backend.KvType = .{ .f32, .f16, .q8_0, .q4_0 };
    for (types) |t| {
        const row_bytes = backend.kvRowBytes(kv_dim, t);
        const flen = (rows * row_bytes + 3) / 4;
        const bytes = try alloc.alloc(u8, 4 * flen);
        defer alloc.free(bytes);
        // Sentinel fill: any row kvAppendRows did NOT address stays 0xA5.
        @memset(bytes, 0xA5);
        const kview: []f32 = @as([*]f32, @ptrCast(@alignCast(bytes.ptr)))[0..flen];
        backend_cpu.kvAppendRows(t, Buf.wrap(kview), Buf.wrap(src), slot, kv_dim, n);
        for (0..slot * row_bytes) |i| try std.testing.expectEqual(@as(u8, 0xA5), bytes[i]);
        const tail = (slot + n) * row_bytes;
        for (tail..bytes.len) |i| try std.testing.expectEqual(@as(u8, 0xA5), bytes[i]);
        // Row `slot` round-trips to the src row (within the format bound).
        const deq = try alloc.alloc(f32, kv_dim);
        defer alloc.free(deq);
        switch (t) {
            .f32 => @memcpy(deq, src[0..kv_dim]),
            .f16 => c_kernels.kvDequantF16(kv_dim, bytes.ptr + slot * row_bytes, deq.ptr),
            .q8_0 => c_kernels.kvDequantQ8_0(kv_dim, bytes.ptr + slot * row_bytes, deq.ptr),
            .q4_0 => c_kernels.kvDequantQ4_0(kv_dim, bytes.ptr + slot * row_bytes, deq.ptr),
        }
        // Per-element dequant error, rel to the row's max|src|:
        //   f16  2^-11*|x| <= 5e-4*rowmax
        //   q8_0 d32/2 + 127*ulp(d) <= 0.0041*rowmax
        //   q4_0 d32 (the amax element clamps 16 -> 15) <= 1.0034*rowmax/8
        const rel: f32 = switch (t) {
            .f32 => 0.0,
            .f16 => 2e-3,
            .q8_0 => 6e-3,
            else => 0.13,
        };
        const bound = rel * maxAbs(src[0..kv_dim]);
        for (deq, src[0..kv_dim]) |a, b| try std.testing.expect(@abs(a - b) <= bound);
    }
}

test "kvRowBytes / kvTypeFits geometry" {
    try std.testing.expectEqual(@as(usize, 2048), backend.kvRowBytes(512, .f32));
    try std.testing.expectEqual(@as(usize, 1024), backend.kvRowBytes(512, .f16));
    try std.testing.expectEqual(@as(usize, 544), backend.kvRowBytes(512, .q8_0)); // 16 x 34
    try std.testing.expectEqual(@as(usize, 288), backend.kvRowBytes(512, .q4_0)); // 16 x 18
    try std.testing.expectEqual(@as(usize, 1088), backend.kvRowBytes(1024, .q8_0)); // 32 x 34
    try std.testing.expectEqual(@as(usize, 576), backend.kvRowBytes(1024, .q4_0)); // 32 x 18
    try std.testing.expect(backend.kvTypeFits(1024, .q8_0));
    try std.testing.expect(backend.kvTypeFits(1024, .q4_0));
    try std.testing.expect(backend.kvTypeFits(1024, .f16));
    try std.testing.expect(backend.kvTypeFits(1024, .f32));
    try std.testing.expect(!backend.kvTypeFits(1000, .q8_0)); // 1000 % 32 = 8
    try std.testing.expect(!backend.kvTypeFits(1000, .q4_0));
}
