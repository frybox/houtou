//! Qwen3.5 CPU scalar inference engine (Q8_0 GGUF only).
//!
//! Linear weights stay in their native Q8_0 blocks (zero-copy views into the
//! mmapped GGUF); matvecs quantize the activation on the fly and accumulate
//! per-block dot products in i32. Norms and elementwise ops stay f32.
//!
//! Numerics follow ref/docs/SPEC.md (authoritative) and ref/docs/GDN_IMPL.md;
//! control flow mirrors ref/c/qwen35_engine.c. GGUF-specific notes:
//!   - GGUF tensor layout differs from the C ref's safetensors (see briefing):
//!     linear layers have attn_qkv [hidden, nk*kd|nk*kd|nv*kd] = [Q|K|V],
//!     attn_gate [hidden, nv*kd] (= z), ssm_alpha [hidden, nv], ssm_beta
//!     [hidden, nk], ssm_a [nv] is already -exp(A_log) (negative values,
//!     verified numerically against SPEC A7's A_log.exp() list), ssm_conv1d
//!     is F32 [4, conv_dim] channel-major taps.
//!     0.8B: hidden=1024, nv=nk=16, kd=128 (conv_dim 6144); 4B: hidden=2560,
//!     nv=32, nk=16, kd=128 (conv_dim 8192).
//!   - GGUF Qwen3_5RMSNorm weights are already (1+w) folded at conversion time
//!     (verified: blk.0.attn_norm[0] == 2.0, q_norm mean ~ 1.4), so they are
//!     applied directly; the gated ssm_norm (Qwen3_5RMSNormGated) is also
//!     pre-folded and applied directly.
//!   - llama.cpp conversion (conversion/qwen.py:394-404) fixes +1 for all
//!     Qwen3_5RMSNorm-family weights and stores ssm_a = -A_log.exp().

const std = @import("std");

/// `$QWEN35_SPEC_FASTROLLBACK=0`：A/B 切回整段重放。快速路径已是默认，
/// 因为已验证它与重放逐位一致。
pub var spec_fast_rollback = true;

/// `$QWEN35_SPEC_RBDBG`：打印快速回滚路径的决策与状态比对。
pub var cudart_debug_rollback = false;

/// `$QWEN35_SPEC_DUMMYALLOC`: DIAGNOSTIC.  Allocates one throwaway device
/// region during speculative setup, to test whether the MTP path's output
/// depends on the device allocator's layout (it should not).
pub var spec_dummy_alloc = false;
const gguf = @import("../gguf.zig");
const backend = @import("../backend.zig");
const backend_cpu = @import("../backend_cpu.zig");
const sampler = @import("../sampler.zig");
const Io = std.Io;

/// The engine hands backend-owned activation regions to every op.  It wraps its
/// host slices here (CPU path); a device backend resolves its own base.
const Buf = backend.Buf;

/// Re-export so the CPU and CUDA hosts share one sampler implementation.
pub const Sampler = sampler.Sampler;

pub const Config = struct {
    n_layers: u32 = 24,
    full_attention_interval: u32 = 4,
    hidden: u32 = 1024,
    n_heads: u32 = 8,
    n_kv_heads: u32 = 2,
    head_dim: u32 = 256,
    n_ff: u32 = 3584,
    rms_eps: f32 = 1e-6,
    vocab: u32 = 248320,
    rope_freq_base: f32 = 1e7,
    /// B9/B10: [11,11,10] mrope sections on 32 freq pairs (rotary_dim=64);
    /// 4th section 0 = padding to 64 dims. Pure text => all three grids equal.
    rope_sections: [4]u32 = .{ 11, 11, 10, 0 },
    /// rotary_dim = head_dim * 0.25 = 64 (un-rotated tail = 192).
    rotary_dim: u32 = 64,
    // GDN
    gdn_n_heads: u32 = 16, // linear_num_value_heads (0.8B: 16, 4B: 32)
    /// k/q head count (ssm.group_count); v-head h uses k-head h / (nv/nk).
    /// 0.8B: 16 (nv == nk), 4B: 16 (nv/nk = 2).
    gdn_num_k_heads: u32 = 16,
    gdn_head_dim: u32 = 128,
    conv_kernel: u32 = 4,
    eos_token_id: u32 = 248046,

    pub fn isFullAttention(self: Config, layer: u32) bool {
        return self.full_attention_interval > 0 and
            layer % self.full_attention_interval == self.full_attention_interval - 1;
    }
};

// ---------------------------------------------------------------------------
// Scalar math (f32; SPEC C15: all norms computed in f32)
// ---------------------------------------------------------------------------

/// RMSNorm, out = x * rsqrt(mean(x^2) + eps) * w. eps inside the sum (SPEC).
/// The sum of squares stays scalar (exact order); the elementwise scale is
/// vectorized.
pub fn rmsNorm(out: []f32, x: []const f32, w: []const f32, eps: f32) void {
    std.debug.assert(out.len == x.len and out.len == w.len);
    var ss: f32 = 0;
    for (x) |v| ss += v * v;
    const scale = 1.0 / @sqrt(ss / @as(f32, @floatFromInt(x.len)) + eps);
    const sv: @Vector(GV, f32) = @splat(scale);
    const xp: [*]const f32 = x.ptr;
    const wp: [*]const f32 = w.ptr;
    const op: [*]f32 = out.ptr;
    for (0..out.len / GV) |b| {
        const o = b * GV;
        vstoreF32(GV, op + o, vloadF32(GV, xp + o) * sv * vloadF32(GV, wp + o));
    }
    for (out.len - out.len % GV..out.len) |i| out[i] = x[i] * scale * w[i];
}

pub fn sigmoid(x: f32) f32 {
    return 1.0 / (1.0 + @exp(-x));
}

/// Elementwise sigmoid over a whole slice (bit-identical to the scalar form).
fn sigmoidVec(x: []f32) void {
    const one: @Vector(GV, f32) = @splat(1.0);
    const p: [*]f32 = x.ptr;
    for (0..x.len / GV) |b| {
        const o = b * GV;
        vstoreF32(GV, p + o, one / (one + @exp(-vloadF32(GV, p + o))));
    }
}

/// out[i] *= silu(g[i]) for the whole slice.
fn mulSiluVec(out: []f32, gate: []const f32) void {
    const one: @Vector(GV, f32) = @splat(1.0);
    const op: [*]f32 = out.ptr;
    const gp: [*]const f32 = gate.ptr;
    for (0..out.len / GV) |b| {
        const o = b * GV;
        const g = vloadF32(GV, gp + o);
        vstoreF32(GV, op + o, vloadF32(GV, op + o) * (g / (one + @exp(-g))));
    }
}

pub fn silu(x: f32) f32 {
    return x / (1.0 + @exp(-x));
}

/// torch F.softplus default beta=1, threshold=20 (SPEC A4.4).
pub fn softplus(x: f32) f32 {
    return if (x > 20.0) x else std.math.log1p(@exp(x));
}

pub fn softmax(x: []f32) void {
    var max: f32 = -std.math.floatMax(f32);
    for (x) |v| max = @max(max, v);
    var sum: f32 = 0;
    for (x) |*v| {
        v.* = @exp(v.* - max);
        sum += v.*;
    }
    for (x) |*v| v.* /= sum;
}

/// FLA l2norm (SPEC A3): x * rsqrt(sum(x^2) + eps), eps inside the sum.
/// Sum of squares stays scalar (exact order); the scale is vectorized.
pub fn l2norm(x: []f32) void {
    var ss: f32 = 0;
    for (x) |v| ss += v * v;
    const inv = 1.0 / @sqrt(ss + 1e-6);
    scaleVec(x, inv);
}

// ---------------------------------------------------------------------------
// GDN delta-rule state kernels, @Vector-ized (SPEC A4)
// ---------------------------------------------------------------------------
//
// The state is [nv, kd, vd] row-major with kd == vd == 128, so every row is
// an exact number of 16-element f32 vectors.  All four per-head operations
// (decay, S^T k, outer-product update, S^T q) are written as @Vector(16, f32)
// loads/stores; the accumulation order per output element is unchanged from
// the scalar version, so results are bit-identical.  Loads/stores are
// unaligned because the state buffer is only 4-byte aligned.

inline fn vloadF32(comptime N: usize, p: [*]const f32) @Vector(N, f32) {
    const ptr: *align(1) const [N]f32 = @ptrCast(p);
    return @bitCast(ptr.*);
}

inline fn vstoreF32(comptime N: usize, p: [*]f32, v: @Vector(N, f32)) void {
    const ptr: *align(1) [N]f32 = @ptrCast(p);
    ptr.* = @bitCast(v);
}

const GV = 16; // f32 lanes per vector used by the GDN kernels

/// Elementwise x *= s (order-preserving, bit-identical to the scalar loop).
fn scaleVec(x: []f32, s: f32) void {
    const sv: @Vector(GV, f32) = @splat(s);
    const p: [*]f32 = x.ptr;
    for (0..x.len / GV) |b| {
        const o = b * GV;
        vstoreF32(GV, p + o, vloadF32(GV, p + o) * sv);
    }
    for (x.len - x.len % GV..x.len) |i| x[i] *= s;
}

/// (1) S *= gamma.
fn gdnDecay(sh: []f32, gamma: f32) void {
    const g: @Vector(GV, f32) = @splat(gamma);
    const p: [*]f32 = sh.ptr;
    for (0..sh.len / GV) |b| {
        const o = b * GV;
        vstoreF32(GV, p + o, vloadF32(GV, p + o) * g);
    }
    for (sh.len - sh.len % GV..sh.len) |i| sh[i] *= gamma;
}

/// (2) km[j] = sum_i sh[i*vd + j] * k[i].  km must be vd long; vd % GV == 0.
/// Accumulates in place in km (same per-element order as the scalar loop).
fn gdnKvMem(sh: []const f32, k: []const f32, km: []f32, vd: usize) void {
    std.debug.assert(vd % GV == 0 and km.len == vd);
    @memset(km, 0);
    const p: [*]const f32 = sh.ptr;
    const q: [*]f32 = km.ptr;
    for (0..k.len) |i| {
        const ki: @Vector(GV, f32) = @splat(k[i]);
        const row = p + i * vd;
        for (0..vd / GV) |b| {
            const o = b * GV;
            vstoreF32(GV, q + o, vloadF32(GV, q + o) + vloadF32(GV, row + o) * ki);
        }
    }
}

/// (3) delta = beta * (v - km);  S += k (outer) delta.
fn gdnOuter(sh: []f32, k: []const f32, v: []const f32, km: []const f32, delta: []f32, beta: f32, vd: usize) void {
    const b16: @Vector(GV, f32) = @splat(beta);
    const vp: [*]const f32 = v.ptr;
    const kp: [*]const f32 = km.ptr;
    const dp: [*]f32 = delta.ptr;
    for (0..vd / GV) |b| {
        const o = b * GV;
        vstoreF32(GV, dp + o, (vloadF32(GV, vp + o) - vloadF32(GV, kp + o)) * b16);
    }
    const p: [*]f32 = sh.ptr;
    for (0..k.len) |i| {
        const ki: @Vector(GV, f32) = @splat(k[i]);
        const row = p + i * vd;
        for (0..vd / GV) |b| {
            const o = b * GV;
            vstoreF32(GV, row + o, vloadF32(GV, row + o) + ki * vloadF32(GV, dp + o));
        }
    }
}

/// (4) out[j] = sum_i sh[i*vd + j] * q[i]  (row-wise accumulation, same order
/// as the scalar column loop).  Accumulates in place in out.
fn gdnStateOut(sh: []const f32, q: []const f32, out: []f32, vd: usize) void {
    std.debug.assert(vd % GV == 0 and out.len == vd);
    @memset(out, 0);
    const p: [*]const f32 = sh.ptr;
    const o: [*]f32 = out.ptr;
    for (0..q.len) |i| {
        const qi: @Vector(GV, f32) = @splat(q[i]);
        const row = p + i * vd;
        for (0..vd / GV) |b| {
            const ob = b * GV;
            vstoreF32(GV, o + ob, vloadF32(GV, o + ob) + vloadF32(GV, row + ob) * qi);
        }
    }
}

/// Fused GDN state update: (1) decay + (2) kv-mem in ONE pass over the state,
/// then (3) delta + outer-update and (4) S^T q in a SECOND pass.  Semantically
/// the same as gdnDecay; gdnKvMem; gdnOuter; gdnStateOut back to back, but the
/// 128x128 (2.1 MB) state is traversed twice instead of four times, and km/out
/// are accumulated in vector registers rather than load/store'd per row.
/// Fast path requires vd == 128 (all Qwen3.5 variants); otherwise it falls back
/// to the four separate kernels.
inline fn gdnFused(sh: []f32, k: []const f32, v: []const f32, q: []const f32, km: []f32, delta: []f32, out: []f32, gamma: f32, beta: f32, vd: usize) void {
    std.debug.assert(k.len == q.len and km.len == vd and out.len == vd and delta.len == vd);
    if (vd != 128) {
        gdnDecay(sh, gamma);
        gdnKvMem(sh, k, km, vd);
        gdnOuter(sh, k, v, km, delta, beta, vd);
        gdnStateOut(sh, q, out, vd);
        return;
    }
    const W = 4; // 128 / 32 lanes per @Vector(32, f32)
    const sp: [*]f32 = sh.ptr;
    const gp: [*]f32 = km.ptr;
    const op: [*]f32 = out.ptr;
    const g: @Vector(32, f32) = @splat(gamma);

    // Pass 1: read-only sweep, km += (S[i] * gamma) * k[i], accumulated in
    // registers.  The decayed rows are NOT stored: pass 2 recomputes them.
    var acc: [W]@Vector(32, f32) = undefined;
    for (0..W) |b| acc[b] = @splat(@as(f32, 0.0));
    for (0..k.len) |i| {
        const row = sp + i * vd;
        const ki: @Vector(32, f32) = @splat(k[i]);
        for (0..W) |b| acc[b] += vloadF32(32, row + b * 32) * ki;
    }
    // km is stored already decayed (S is overwritten below), so callers that
    // read km afterwards see the same value as gdnKvMem after gdnDecay.
    for (0..W) |b| {
        acc[b] *= g;
        vstoreF32(32, gp + b * 32, acc[b]);
    }

    // delta = beta * (v - km)
    const bp: [*]const f32 = v.ptr;
    const dp: [*]f32 = delta.ptr;
    const bb: @Vector(32, f32) = @splat(beta);
    for (0..W) |b| {
        const o = b * 32;
        vstoreF32(32, dp + o, (vloadF32(32, bp + o) - vloadF32(32, gp + o)) * bb);
    }

    // Pass 2: S[i] = S[i] * gamma + k[i] * delta (one store), out += S[i] * q[i].
    var dvec: [W]@Vector(32, f32) = undefined;
    for (0..W) |b| dvec[b] = vloadF32(32, dp + b * 32);
    var oacc: [W]@Vector(32, f32) = undefined;
    for (0..W) |b| oacc[b] = @splat(@as(f32, 0.0));
    for (0..k.len) |i| {
        const row = sp + i * vd;
        const ki: @Vector(32, f32) = @splat(k[i]);
        const qi: @Vector(32, f32) = @splat(q[i]);
        for (0..W) |b| {
            const o = b * 32;
            const r = vloadF32(32, row + o) * g + ki * dvec[b];
            vstoreF32(32, row + o, r);
            oacc[b] += r * qi;
        }
    }
    for (0..W) |b| vstoreF32(32, op + b * 32, oacc[b]);
}

/// Vectorized silu over a whole slice: x / (1 + exp(-x)).
fn siluVec(x: []f32) void {
    const p: [*]f32 = x.ptr;
    const one: @Vector(GV, f32) = @splat(1.0);
    for (0..x.len / GV) |b| {
        const o = b * GV;
        const xv = vloadF32(GV, p + o);
        vstoreF32(GV, p + o, xv / (one + @exp(-xv)));
    }
    for (x.len - x.len % GV..x.len) |i| x[i] = silu(x[i]);
}

/// out[i] *= sigmoid(g[i]) for the whole slice.
fn mulSigmoidVec(out: []f32, gate: []const f32) void {
    const one: @Vector(GV, f32) = @splat(1.0);
    const op: [*]f32 = out.ptr;
    const gp: [*]const f32 = gate.ptr;
    for (0..out.len / GV) |b| {
        const o = b * GV;
        const g = vloadF32(GV, gp + o);
        vstoreF32(GV, op + o, vloadF32(GV, op + o) * (one / (one + @exp(-g))));
    }
    for (out.len - out.len % GV..out.len) |i| out[i] *= sigmoid(gate[i]);
}

/// Activation context for the f64 reference: `x` is the serial input
/// (k == 1) or the [k, cols] row-major batch input (k > 1, out is [k, rows]).
const DumpX = struct { x: []const f32, k: usize };

/// f64 golden reference error for one matvec of any ttype: dequantize each
/// weight row with the pure Zig scalar dequant (dequantToRows' per-block
/// dequantRowFn, verified byte-identical to the C scalar kernels) and dot
/// against the ORIGINAL x in f64.  Returns
/// max_i |out[i] - ref[i]| / max_i |ref[i]|, i.e. the maximum relative error
/// of `out` against the dequantized-f32/f64 ground truth.
///
/// Reusable for tests and for validating new kernels (f32, Q8_0 and every
/// packed K/IQ ttype) without any debug-print scaffolding.
/// Negative return = reference itself not computable (unknown ttype / OOB x).
pub fn refErrF64(w: Engine.LinearW, out: []const f32, dx: DumpX, cols: usize) f64 {
    const k: usize = @max(1, dx.k);
    const rows = out.len / k;
    if (rows * k != out.len or rows == 0) return -1;
    var rowbuf: [256]f32 = undefined;
    var max_ref: f64 = 0;
    var max_err: f64 = 0;
    var vi: usize = 0;
    while (vi < out.len) : (vi += 1) {
        const t = vi / rows;
        const rr = vi % rows;
        const xoff = t * cols;
        if (xoff + cols > dx.x.len) return -3; // activation slice OOB
        var acc: f64 = 0;
        switch (w) {
            .quant => |q| {
                // One `deq` call per row-block, every ttype including Q8_0 --
                // deliberately the SAME framing as `backend_cpu.dequantRowInto`,
                // because that shared shape is what makes the two comparable at
                // all: this function is the f64 oracle matvec results are checked
                // against, so it must not be a second, independently written
                // decode of the same bytes.
                //
                // Q8_0 needs no branch of its own here.  The reason it has one in
                // `dequantRowInto` -- "a row is not a whole number of 256-element
                // units" -- does not apply, because this loop does not iterate in
                // 256-element units: it iterates `rowBlockElems()`, which for Q8_0
                // IS its native 32-element block (`gguf.rowBlockElems`), so
                // `per_rb == 1` and one 34-byte block goes to `deq` per call.
                // `dequantRowFnFor(.q8_0)` is registered to make that work.
                //
                // Keying on `blockElems()` alone WOULD be wrong: f16/bf16 have
                // blockElems == 1 (one element per 2-byte block), so a per-block
                // loop asks a 256-wide dequantizer for a 1-wide output.
                const bb = q.blockBytes(); // one block of this type
                const rbe = q.rowBlockElems();
                const per_rb = rbe / q.blockElems(); // blocks per row-block
                const deq = backend_cpu.dequantRowFnFor(q.ttype) orelse return -2;
                for (0..q.blocks_per_row) |b| {
                    const i = rr * q.blocks_per_row + b;
                    const span = per_rb * bb;
                    if (i * span + span > q.data.len) return -4;
                    deq(q.data[i * span ..][0..span], rowbuf[0..rbe]);
                    for (0..rbe) |j| acc += @as(f64, rowbuf[j]) * @as(f64, dx.x[xoff + b * rbe + j]);
                }
            },
            .f32 => |m| {
                if (rr * cols + cols > m.len) return -4;
                const rowv = m[rr * cols ..][0..cols];
                for (0..cols) |c| acc += @as(f64, rowv[c]) * @as(f64, dx.x[xoff + c]);
            },
        }
        const e = @abs(acc - @as(f64, out[vi]));
        if (e > max_err) max_err = e;
        const a = @abs(acc);
        if (a > max_ref) max_ref = a;
    }
    if (max_ref == 0) return 0;
    return max_err / max_ref;
}

/// Row-major [rows, cols] matvec: out[r] = sum_c mat[r*cols + c] * x[c].


// ---------------------------------------------------------------------------
// Q8_0 block format (ggml: 34 bytes per 32 elements: f16 scale + 32 x i8)
// ---------------------------------------------------------------------------

/// Block geometry of a quantized ttype: ONE source, `gguf.TensorType
/// .blockLayout()`, which mirrors ggml's `ggml_blck_size` / `ggml_type_size`
/// pair (`ggml.c` type table; per-block sizes asserted in ggml-common.h e.g.
/// `static_assert(sizeof(block_q8_0) == ...)`).  Every block-elems / block-bytes
/// number in engine.zig is derived here -- there is deliberately no
/// "row bytes" table any more: a row is `blocks_per_row * blockElemBytes()`.
///
/// Panics on a layout-less ttype: 0 was the old "unknown" sentinel and each
/// caller turned it into its own error, but every caller is either guarded by
/// `gguf.File` rejecting unknown types at load (`gguf.zig` `isKnown()`) or
/// behind an exhaustive dispatch switch, so reaching here means a real bug.
pub const blockElemsOf = gguf.blockElems;

/// Packed bytes of one block of `t` (`blockLayout().bytes`).
pub const blockElemBytes = gguf.blockBytes;

pub const q8_0_block_elems = blockElemsOf(.q8_0);
pub const q8_0_block_bytes = blockElemBytes(.q8_0);

/// Bytes `q8_scratch` must hold to quantize a matvec INPUT of `cols` elements
/// (ceil(cols/32) blocks of 34 bytes).  Every Q8_0 matvec writes its quantized
/// input into this scratch, so the scratch is sized by the widest INPUT, never
/// by an output row.
pub fn q8ScratchBytes(cols: usize) usize {
    return (cols + q8_0_block_elems - 1) / q8_0_block_elems * q8_0_block_bytes;
}

/// Bytes `q8k_scratch` must hold to Q8_K-quantize a matvec INPUT of `cols`
/// elements (292 bytes per 256 elements: f32 scale + 256 x i8 + 16 x i16 bsums).
pub fn q8kScratchBytes(cols: usize) usize {
    const be = blockElemsOf(.q8_k);
    return (cols + be - 1) / be * blockElemBytes(.q8_k);
}

/// The widest Q8_0 matvec INPUT in a model of this shape, derived from the
/// config.  Each Q8_0 matvec quantizes its input activation into q8_scratch,
/// so the scratch must cover the widest input:
///   ffn_down        <- n_ff        (mlp_h)
///   attn_output     <- q_dim       (gout, full-attention layers)
///   ssm_out         <- value_dim    (gout, GDN layers)
///   token_embd / lm_head / in_proj / ffn_gate / ffn_up <- hidden
/// Sizing to conv_dim / 2*q_dim (the OUTPUT widths of attn_qkv / attn_q) is
/// the classic miss: n_ff is the widest input in every Qwen3.5 config
/// (27B: n_ff=17408 > conv_dim=10240 > q_dim=6144 > hidden=5120), so ffn_down
/// writes past the arena end -- assert in Debug, silent heap corruption in
/// ReleaseFast.  Same root cause as the CUDA-side 2B/4B fix (main_cuda.zig:
/// max(hidden, value_dim, q_dim, n_ff)).
pub fn maxQ8Cols(c: Config) usize {
    const n_ff: usize = c.n_ff;
    const hidden: usize = c.hidden;
    const value_dim: usize = @as(usize, c.gdn_n_heads) * c.gdn_head_dim;
    const q_dim: usize = @as(usize, c.n_heads) * c.head_dim;
    return @max(@max(n_ff, hidden), @max(value_dim, q_dim));
}

/// Anti-regression guard: every Q8_0 matvec's input length must fit in
/// q8_scratch.  Returns the widest input seen, so a miss is reported at load
/// time instead of as an out-of-bounds write inside the first matvec.
pub fn checkQ8Scratch(scratch: []const u8, cols: usize) !void {
    if (scratch.len < q8ScratchBytes(cols)) return error.Q8ScratchTooSmall;
}

// ---------------------------------------------------------------------------
// Q8_0-native matvec: weights stay in quantized blocks (zero-copy mmap
// view), dot products accumulate in i32 per block and rescale by the f16
// block scale. x is f32, quantized on the fly into a scratch buffer.
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Generic per-row dequant: a tensor of a K-quantized type is a sequence of
// fixed-size rows (each `row_bytes` bytes == 256 elements); dequantize each
// row into its 256 f32 values and append to `out`. `out.len` must equal the
// element count of the tensor (rows * 256).
// ---------------------------------------------------------------------------


// ---------------------------------------------------------------------------
// RoPE (SPEC B9/B10/B11): rotary_dim=64, cross-half pairing (i, i+32),
// cos/sin layout [c0..c31, c0..c31], inv_freq[j] = base^(-2j/64).
// Only dims 0..63 rotate; dims 64..255 pass through unchanged.
// mrope sections [11,11,10] only reorder WHICH section uses which grid
// position; for pure text all grids are identical, so sections only
// determine the frequency grouping: pairs 0..10 use pos of grid dim0,
// 11..21 grid dim1, 22..31 grid dim2 -- all == pos here (B10).
// ---------------------------------------------------------------------------

/// Rotate a 64-dim rotary block in place (cross-half pairing).
pub fn applyRope64(x: []f32, pos: u32, base: f32) void {
    const half = 32;
    // freq j uses exponent 2j/rotary_dim with rotary_dim=64
    for (0..half) |j| {
        const theta = @as(f32, @floatFromInt(pos)) * std.math.pow(f32, base, -@as(f32, @floatFromInt(2 * j)) / 64.0);
        const c = @cos(theta);
        const s = @sin(theta);
        const a = x[j];
        const b = x[j + half];
        x[j] = a * c - b * s;
        x[j + half] = a * s + b * c;
    }
}

// ---------------------------------------------------------------------------
// Engine
// ---------------------------------------------------------------------------

pub var prof_io: Io = undefined;

/// Disable the K-token prefill chunking (QWEN35_NO_BATCH); every token then
/// runs through the single-token path.
pub var no_batch = false;
/// Enable the per-phase timing report (QWEN35_PROF).
pub var prof = false;

pub fn profReport() void {
    const tot: f64 = prof_emb + prof_gdn + prof_full + prof_mlp + prof_lm;
    std.debug.print("PROF emb={d:.2}ms gdn={d:.2}ms full={d:.2}ms mlp={d:.2}ms lm={d:.2}ms total={d:.2}\n", .{ prof_emb / 1e6, prof_gdn / 1e6, prof_full / 1e6, prof_mlp / 1e6, prof_lm / 1e6, tot / 1e6 });
    backend_cpu.pool_profReport();
}

fn profNs(io: Io, t0: Io.Clock.Timestamp) f64 {
    return @as(f64, @floatFromInt(t0.durationTo(Io.Clock.Timestamp.now(io, .awake)).raw.nanoseconds));
}

pub var prof_gdn: f64 = 0;
pub var prof_full: f64 = 0;
pub var prof_mlp: f64 = 0;
pub var prof_lm: f64 = 0;
pub var prof_emb: f64 = 0;

pub const Engine = struct {
    alloc: std.mem.Allocator,
    cfg: Config,
    g: gguf.File,
    mm: std.Io.File.MemoryMap,
    file: std.Io.File,
    io: std.Io,
    /// set by `load` when it created the Threaded instance; freed in deinit
    owns_io: ?*std.Io.Threaded = null,

    token_embd: LinearW, // [vocab, hidden] row-major (embedding)
    /// untied lm_head (27B+); null in the Engine means "tied" -> use
    /// token_embd.  GGUF `output.weight` has the SAME [vocab, hidden]
    /// row-major layout as token_embd.
    output_weight: ?LinearW,
    output_norm: backend.Buf, // [hidden]
    /// Hardware boundary: matvecs, weight loading, dequantization and the
    /// activation scratch the kernels need.  The engine never learns which
    /// implementation this is -- CPU today, CUDA later.
    exec: backend.Exec,
    layers: []LayerWeights,
    kv: []KvCache,
    gdn: []GdnState,
    /// Pre-verify snapshot for speculative decoding; empty until
    /// `specSnapshotAlloc` runs.
    spec_snap: []SnapLayer = &.{},
    /// Trailing MTP / nextn prediction block (`blk.<n_layers>.*`), or null when
    /// the model has none (0.8B/2B/4B) or `spec` was never initialised.
    ///
    /// SPECULATIVE DECODING (stage 1): the nextn block is a full decoder layer
    /// plus a `eh_proj` that folds (embedding, hidden) back to `hidden`, and it
    /// shares the main model's `output_norm` / lm_head.  It is the cheapest
    /// possible drafter -- the weights ship with the target, so there is nothing
    /// to download, and one forward gives one speculative token instead of the
    /// 15 a DSpine-style block drafter proposes.
    nextn: ?Nextn = null,
    /// worker count (1 = serial); set by load from `n_threads`
    n_threads: usize = 1,

    /// Transient activations live in backend-owned buffers
    /// (`Exec.allocBuf`): the engine names a `Buf` (base + length), never a
    /// host slice, so a device backend can hand back device memory without the
    /// engine changing.  This step (P2a) still gets host memory from both
    /// backends; every `exec.*` call therefore receives a `Buf` directly, and
    /// the few places the engine itself reads scalar elements go through
    /// `.slice()`.
    ///
    /// NOTE: these regions are owned by the backend (`Exec.freeScratch`
    /// releases them all at once).  The engine must never `alloc.free` one.
    x: backend.Buf, // hidden residual stream
    x1: backend.Buf, // hidden normed input
    x2: backend.Buf, // hidden output projection target
    qkv: backend.Buf, // GDN conv_dim / reused for q,k,v of full layers
    z: backend.Buf, // gate (GDN value_dim / full-attn q_dim)
    ab: backend.Buf, // a|b (2*nv scalars)
    gdn_out: backend.Buf, // GDN output / full-attn attention out
    q: backend.Buf, // full-attn q
    k: backend.Buf, // full-attn k
    v: backend.Buf, // full-attn v
    attn: backend.Buf, // full-attn scores [max_seq]
    mlp_h: backend.Buf, // n_ff
    mlp_h1: backend.Buf, // n_ff
    logits: backend.Buf,
    delta: backend.Buf, // gdn_n_heads * 128: PER-HEAD scratch for gdnStepRows
    kv_mem: backend.Buf, // gdn_n_heads * 128: per-head, same reason
    /// Per-layer GDN decay scalars (`gdnGamma`'s output), `gdn_n_heads` long.
    /// A backend region because `gdnGamma` runs on the backend under a device
    /// backend, and `gdnStepRows` consumes it there as a `Buf` -- nothing reads
    /// it back to the host.
    gamma_scratch: backend.Buf,
    /// Prefill twin of `gamma_scratch`: `nbatch * gdn_n_heads`, one gamma row per
    /// token of the current chunk, written by `gdnPrefillCore` and consumed by
    /// `forwardGdnCore`.  Separate from `gamma_scratch` so the serial decode path
    /// keeps its single-row region untouched.
    gdn_prefill_gamma: backend.Buf,
    /// 每层 GDN 在 verify 块上的**原始投影**缓存（pre-conv 的 `qkv` 与 `ab`）。
    ///
    /// 这是让推测解码的回滚变便宜的东西：verify 已经把整块每一行都算过，
    /// 而被接受的前缀就是这些行的前缀，所以从快照恢复后只把那些行重走一遍
    /// `gdnStepRowsChunk` 就能精确复现状态——不必重跑 65 层投影。
    /// 实测：MTP 每 32 个 token 要 7951 次批量 matvec（plain 只有 496），
    /// 多出来的全在重放里。
    ///
    /// 必须缓存 pre-conv 的原始投影，不能缓存 conv 之后的字节：卷积历史由
    /// prelude 推进，且无法从 conv 的**输出**反推，所以缓存 post-conv 会让
    /// 回滚退回"重跑整条链"。见 `specRollbackAndReplay`。
    gdn_in: []GdnIn = &.{},
    /// 当前 chunk 是否是推测解码的 verify。只有 verify 需要 GDN 状态快照；
    /// prefill 与普通 decode 恒为 false，走不带快照的 chunk 内核。
    in_spec_verify: bool = false,
    /// `$QWEN35_SPEC_RBDBG` 用的常驻比对缓冲（两份 GDN `state` 的 HOST 副本）。
    /// 只在打开诊断时由 `allocSpecSnapshot` 分配一次。`copyOut` 的目标是
    /// HOST 内存，所以这两个是普通切片，不是 `Buf`（设备地址）。
    /// 每轮现场 `alloc` 3 MB 既没必要，也会让分配点混进热路径。
    spec_dbg_a: []f32 = &.{},
    spec_dbg_b: []f32 = &.{},
    /// Number of tokens batched per prefill chunk (K-token batched projection:
    /// each layer's weights are read once per chunk and reused across K tokens
    /// through cache).  0 => the batched path is unavailable.
    nbatch: usize = 0,
    /// K-slot scratch for the batched prefill path.  `xK[t*hidden..]` is the
    /// residual stream of token t of the current chunk; `x1K[t*hidden..]` the
    /// RMS-normed input; `qkvK/gateK/abK/qK/kK/vK` the batched projection
    /// outputs; `goutK/x2K` per-token attention/GDN output and out-projection
    /// result; `mlpK/mlpK1` the ffn intermediates; `q8kK/q8K` the per-token
    /// quantized activations for the batched matvecs.
    xK: backend.Buf = backend.Buf.empty,
    x1K: backend.Buf = backend.Buf.empty,
    qkvK: backend.Buf = backend.Buf.empty,
    gateK: backend.Buf = backend.Buf.empty,
    abK: backend.Buf = backend.Buf.empty,
    qK: backend.Buf = backend.Buf.empty,
    kK: backend.Buf = backend.Buf.empty,
    vK: backend.Buf = backend.Buf.empty,
    goutK: backend.Buf = backend.Buf.empty,
    x2K: backend.Buf = backend.Buf.empty,
    mlpK: backend.Buf = backend.Buf.empty,
    mlpK1: backend.Buf = backend.Buf.empty,
    /// `spec_verify_rows * vocab` f32: every row's lm_head output for a batched
    /// speculative verify (`forwardChunkVerify`).  The single-row `logits`
    /// buffer cannot hold a block's worth.
    logitsK: backend.Buf = backend.Buf.empty,
    /// Top-k order-statistic workspace (quickselect); sized vocab.
    sample_scratch: backend.Buf = backend.Buf.empty,
    /// Sampling policy applied to every produced token.  Default is greedy
    /// (temperature 0), i.e. exactly the historical behaviour; the CLI sets
    /// it from -t/-k/-s.
    sampler: Sampler = Sampler.init(.{}),
    prefill_ms: f64 = 0,

    // ---- speculative-decoding taps (DSpine port, stage 0) ----
    //
    // The DSpine draft consumes the target's hidden state at a FIXED set of
    // layer indices (`dflash_config.target_layers`, `[2,17,32,47,62]` for the
    // 27B) and never looks at any other layer.  `forwardChunk` can therefore
    // copy those layers' OUTPUT hidden state into `tap_hidden`, which the
    // draft's `fc` projection then turns into its 5H -> H context.
    //
    // LAYOUT: `tap_hidden` is `taps.len * nbatch * hidden`, i.e. one
    // `nbatch * hidden` plane per tap, tap-major.  Tap `i` covers layer
    // `tap_layers[i]`, and inside that plane row `t` is token `t` of the
    // chunk -- matching `dspine_mlx`'s `_LayerHook`, which stores the layer's
    // WHOLE output tensor (`mx.concatenate(storage, axis=-1)` keeps every
    // position; the draft needs features for all block rows, not just the
    // last).  The `nbatch` stride (not `k`) is so a short final chunk still
    // lands at a fixed offset and the reader does not have to know `k`.
    //
    // Disabled by default: `tap_hidden` is `Buf.empty`, so the normal
    // prefill/decode path pays ONE usize compare per layer and nothing else.
    //
    // ORDER: `tap_layers` must be ASCENDING (it is the draft's
    // `target_layers`); tap i is written from layer `tap_layers[i]`.
    tap_layers: []const u32 = &.{},
    tap_hidden: backend.Buf = backend.Buf.empty,
    /// Rolling tap ring for the spec context window (`specRingEnable`).
    ring: []f32 = &.{},
    ring_ctx: usize = 0,
    ring_next: usize = 0,
    ring_filled: usize = 0,
    /// `$QWEN35_SPEC_TRACE`: print each round's first proposal next to the
    /// target's own pick.  Set by `main.zig`; purely diagnostic.
    spec_trace_dbg: bool = false,
    /// `$QWEN35_SPEC_POSSTAT`: buckets each round by how many draft ids the
    /// verifier accepted (`0..max_draft`).
    spec_posstat: bool = false,
    spec_pos_reached: [64]usize = @splat(0),
    /// `$QWEN35_SPEC_RECALL`: the draft's `selector_top_k`.  Sized by the caller
    /// (`main.zig`) before `generateSpec`, because the engine must not import
    /// the draft module.
    spec_recall_k: usize = 0,
    /// `$QWEN35_SPEC_RECALL`: failures split by cause.  `miss_select` = the
    /// target's token WAS in the candidate set but stage 4 picked another lane;
    /// `miss_forward` = it was not in the set at all.
    recall_miss_select: usize = 0,
    recall_miss_forward: usize = 0,
    /// Which backend `exec` drives.  Only consulted by the CPU-only debug
    /// helpers (`allocHiddenRow`), which must refuse a device buffer.
    backend_kind: backend.Kind = .cpu,


    /// Packed weight types are defined by the GGUF module (`gguf.QuantRow` is
    /// pure layout); the backend owns `LinearW` because whether a weight lives
    /// packed in host memory, expanded to f32, or uploaded to device memory is
    /// a backend decision.
    pub const QuantRow = gguf.QuantRow;
    pub const LinearW = backend.LinearW;

    /// [rows, cols] so cols = len / rows.
    fn matvecCols(w: LinearW, out_len: usize) usize {
        return switch (w) {
            .quant => |q| q.cols(),
            .f32 => |m| m.len / out_len,
        };
    }

    const LayerWeights = struct {
        attn_norm: backend.Buf,
        post_attention_norm: backend.Buf,
        // GDN-only
        attn_qkv: ?LinearW = null,
        attn_gate: ?LinearW = null,
        ssm_alpha: ?LinearW = null, // a (decay input)
        ssm_beta: ?LinearW = null, // b (beta input)
        ssm_conv1d: backend.Buf = backend.Buf.empty, // F32 [4, conv_dim]: w[ch*k + t]
        ssm_dt_bias: backend.Buf = backend.Buf.empty, // [nv], already softplus-anchored
        ssm_a: backend.Buf = backend.Buf.empty, // [nv], already -exp(A_log)
        ssm_norm: backend.Buf = backend.Buf.empty, // [128]
        ssm_out: ?LinearW = null,
        // full-attention-only
        attn_q: ?LinearW = null, // [1024, 4096] = per-head [q|gate] interleaved
        attn_k: ?LinearW = null,
        attn_v: ?LinearW = null,
        attn_q_norm: backend.Buf = backend.Buf.empty, // [256]
        attn_k_norm: backend.Buf = backend.Buf.empty, // [256]
        attn_output: ?LinearW = null,
        // shared
        ffn_gate: LinearW,
        ffn_up: LinearW,
        ffn_down: LinearW,
    };

    /// KV cache for one full-attention layer: `max_seq` rows of `kv_dim`
    /// floats each, in the backend's own address space (device memory under
    /// `--backend cuda`).  `len` is the number of rows written so far.
    ///
    /// P2c: these are `backend.Buf`, not `[]f32`, and that is what lets
    /// `attnDecode` read the rows without a round trip -- the engine never
    /// dereferences them, it only takes `span()` sub-windows and hands them to
    /// backend slots.
    const KvCache = struct {
        k: backend.Buf,
        v: backend.Buf,
        len: u32,
    };

    /// Causal conv state: [conv_dim] history of the last K-1=3 raw samples
    /// (channel-major, same order as the C ref's per-channel rows).
    /// delta-rule state: [nv, kd, kd] row-major S[k][v].
    /// Same story as `KvCache` (P2c): backend-owned regions, addressed only
    /// through `span()`.  `state` is [nv, kd, vd] row-major, one `kd*vd`
    /// sub-window per value head.
    /// Trailing MTP / nextn prediction block (SPECULATIVE DECODING, stage 1).
    ///
    /// The block is a normal decoder layer at index `cfg.n_layers` (27B:
    /// blk.64) PLUS four MTP-only tensors.  Its input is not the residual
    /// stream: it is
    ///
    ///     h = eh_proj( [ enorm(embedding(next_token)) ; hnorm(hidden) ] )
    ///
    /// (embedding FIRST -- `qwen.py`'s remapper names them
    /// `pre_fc_norm_embedding` / `pre_fc_norm_hidden`, and deepseek.py builds
    /// the fused matrix with `torch.cat((e_proj, h_proj), dim=1)`).  The layer
    /// then runs attention + MLP as usual and its output is normed by
    /// `shared_head_norm` before the SHARED lm_head / `output_norm`, i.e. the
    /// drafter does not own a head.
    ///
    /// `eh_proj` is `[2*hidden, hidden]` in GGUF (= torch `[hidden, 2*hidden]`),
    /// so one matvec over a `2*hidden` input produces the layer input directly.
    const Nextn = struct {
        /// `[2*hidden]` staging for `[enorm(embed); hnorm(hidden)]`.
        input: backend.Buf,
        eh_proj: LinearW,
        enorm: backend.Buf, // [hidden]
        hnorm: backend.Buf, // [hidden]
        shared_head_norm: backend.Buf, // [hidden]
        /// Layer index of the block inside `layers`/`kv` (== `cfg.n_layers`).
        li: usize,
    };

    const GdnState = struct {
        conv: backend.Buf,
        conv_len: u32,
        /// 活跃的状态切片视图（`[nv, kd, vd]`）。它指向 `state_all` 的第
        /// `rs_idx` 块：推测解码回滚时只改这个视图，不拷贝、不重算。
        /// 普通解码时恒为第 0 块。
        state: backend.Buf,
        /// `(1 + spec_rs_seq)` 份 `[nv, kd, vd]` 连续状态缓冲。第 0 块是常规
        /// 状态，第 `t` 块（t>=1）是"跑完 verify 第 t 行之后"的快照。
        ///
        /// 这是 llama.cpp `n_rs_seq` 的同一布局（`src/llama-memory-recurrent.cpp`
        /// 的 `n_rows = mem_size * (1 + n_rs_seq)`）：状态缓冲多开 K 份，
        /// verify 时内核逐行写不同块，回滚只把 `rs_idx` 改到对应块。
        /// 相比"回滚时重跑 rows 行递推"，这省掉 73.9% 的 MTP 时间（实测）。
        state_all: backend.Buf,
        /// 当前活跃块号。0 = 常规状态；1..K = verify 第 t 行之后的状态。
        rs_idx: u32,
    };

    /// One layer's pre-verify state, for the speculative rollback
    /// (`specSnapshotTake` / `specRollbackAndReplay`).  Host memory on both
    /// backends: `copyRange` moves device<->host, and the bytes are touched
    /// once per round, not per token.
    const SnapLayer = struct {
        /// Backend regions, NOT host slices: the snapshot is taken and restored
        /// inside the per-round speculative loop, and on CUDA a host copy means
        /// 230 MB of PCIe per round (48 GDN layers x [conv + 128x128 state]).
        /// Keeping both sides as `Buf` makes the whole thing `copyRange`
        /// (device-to-device), which is what the ablation measured: dropping
        /// the transfer entirely took 27B MTP decode from 19.1 to 32.1 tok/s.
        conv: backend.Buf,
        state: backend.Buf,
        conv_len: u32,
        kv_len: u32,
    };

    pub const max_seq: u32 = 4096;

    /// 一层 GDN 的状态内核输入缓存（见 `gdn_in`）。
    pub const GdnIn = struct {
        /// 该层 `matvecBatchK` 产出的**原始** `[q|k|v]` 投影行，`K * conv_dim`。
        /// 刻意缓存 pre-conv 的字节：回滚要重跑 prelude，让卷积历史只前进
        /// `accepted` 步，而不是 verify 的 `k` 步。
        qkv: backend.Buf = backend.Buf.empty,
        /// 每 token 的原始 `[a|b]` 投影，`K * 2 * nv`（即 prelude 的输入）。
        gamma: backend.Buf = backend.Buf.empty,
    };

    /// Rows `forwardChunkVerify` accepts.  A spec block is `block_size` (16 on
    /// the 27B), so the verify buffer is `16 * vocab` f32 -- distinct from the
    /// prefill chunk `nbatch` (1024), which would make it 1 GB.
    pub const spec_verify_rows: usize = 16;

    /// GDN 状态快照槽数 K。一次 verify 块最多 `spec_verify_rows` 行，接受
    /// `rows <= verify_rows - 1` 行，而槽 `rows-1` 是"跑完第 rows 行之后"，
    /// 所以需要 `spec_verify_rows - 1` 个 快照块。取 16 覆盖到 verify 的
    /// 全部行数，与 llama.cpp 的 `n_rs_seq = draft.n_max` 同一个量级。
    pub const spec_rs_seq: usize = spec_verify_rows;

    /// One decl per `backend.Kind`, holding the concrete backend type
    /// `selectBackend` picks for it.
    ///
    /// `load` cannot call `selectBackend(kind)` directly: it returns a `type`,
    /// so its argument must be comptime-known, while the CLI kind is a runtime
    /// value ("types are not available at runtime").  Naming the map as a
    /// *namespace of decls* keeps it comptime-known at the call site, so
    /// `@field(BackendTypes, @tagName(k))` works for a runtime `k`.
    /// (Fields would not do: a struct field cannot hold a type as its value.)
    const BackendTypes = struct {
        pub const cpu = backend.selectBackend(.cpu);
        pub const cuda = backend.selectBackend(.cuda);
    };

    /// Map a runtime backend kind to its concrete backend struct type.
    ///
    /// `kind` is `comptime` on purpose: callers that have a runtime kind must
    /// instantiate once per tag (see the `switch` in `load`), because a `type`
    /// cannot be produced at runtime.  The `inline else` prong captures the tag
    /// at comptime, which is what lets `@field` name one instantiation per
    /// `Kind`.  No `else`: a new `Kind` fails to compile here instead of
    /// silently running CPU code.
    fn backendImplFor(comptime kind: backend.Kind) type {
        return switch (kind) {
            inline else => |k| @field(BackendTypes, @tagName(k)),
        };
    }

    /// Debug switch: force a SIMD tier (baseline|avx2|avx512) before load.
    /// Force a SIMD tier (QWEN35_SIMD_TIER).  Static because it is applied
    /// before the backend exists (the tier only changes which C kernels the
    /// dispatch tables point at), so it forwards to the CPU backend's setter.
    pub fn setSimdTier(name: []const u8) void {
        backend_cpu.setSimdTier(name);
    }

    /// `spec` = 调用方是否要在本次运行里跑推测解码（MTP）。
    ///
    /// 它决定 `gdn_in`/`spec_snap` 是否在加载时分配：让不需要推测解码的运行
    /// 完全不付这块显存（27B 上 6 MB 缓存 + 31 MB 状态快照）。加载时就能
    /// 确定的事（模型有 nextn 块 + 用户开了开关）不应该拖到第一次
    /// `generateSpec` 才去分配。
    pub fn load(alloc: std.mem.Allocator, path: []const u8, io: std.Io, n_threads: usize, backend_kind: backend.Kind, kv_type: backend.KvType, spec: bool) !Engine {
        var nt = n_threads;
        if (nt == 0) {
            const ncpu = std.Thread.getCpuCount() catch 1;
            nt = @min(16, ncpu);
        }
        const file = std.Io.Dir.cwd().openFile(io, path, .{}) catch return error.OpenFailed;
        errdefer file.close(io);
        const stat = file.stat(io) catch return error.OpenFailed;
        const file_size: usize = @intCast(stat.size);
        var mm = std.Io.File.MemoryMap.create(io, file, .{
            .len = file_size,
            .protection = .{ .read = true },
        }) catch |e2| {
            file.close(io);
            return e2;
        };
        errdefer mm.destroy(io);

        var g = try gguf.parse(alloc, mm.memory);
        errdefer g.deinit();

        // Create the hardware backend now that the file is parsed: the CPU
        // backend keeps `&g` to resolve tensor names into mmap views.  Its
        // allocation lives in the same arena as the engine.
        //
        // `switch` because the concrete type depends on the *runtime* kind:
        // `selectBackend` returns a `type`, which cannot be produced at
        // runtime, so each prong instantiates its own backend.  `inline else`
        // means a new `Kind` is handled automatically, and a missing entry in
        // `BackendTypes` is then a compile error rather than a silent fallback.
        //
        // The struct itself is not kept: `exec` is all the engine needs, and
        // its `ptr` keeps the box alive (same arena as the engine).
        const exec = switch (backend_kind) {
            inline else => |k| blk: {
                const BackendImpl = backendImplFor(k);
                const p = try alloc.create(BackendImpl);
                p.* = BackendImpl.init(alloc, &g);
                break :blk p.exec();
            },
        };

        // KV cache format for this run (CLI `--kv-type`, default f16).  The
        // engine only forwards it: allocation, quantization and dequantization
        // all live in the backends, which own the cache bytes.
        exec.setKvType(kv_type);

        var cfg = Config{};
        // block_count includes the trailing MTP/nextn prediction layer(s)
        // (27B: block_count=65, nextn=1 -> main decoder = layers 0..63).
        // The 0.8B/2B/4B files have no nextn key -> 0 -> unchanged.
        //
        // The count shrinks because blk.64 is a prediction head, not a decoder
        // layer.  It is no longer unused: `Nextn` (below) loads it as a
        // speculative drafter, stored in the SAME `layers`/`kv` arrays at
        // index `cfg.n_layers`.  `cfg.n_layers` stays 64 so every existing
        // loop, cache sizing and `isFullAttention` call keeps its meaning.
        var n_nextn: u32 = 0;
        if (g.getUint("qwen35.block_count")) |v| {
            cfg.n_layers = @intCast(v);
            if (g.getUint("qwen35.nextn_predict_layers")) |n| n_nextn = @intCast(n);
        }
        if (n_nextn > 0 and n_nextn < cfg.n_layers) cfg.n_layers -= n_nextn;
        if (g.getUint("qwen35.embedding_length")) |v| cfg.hidden = @intCast(v);
        if (g.getUint("qwen35.feed_forward_length")) |v| cfg.n_ff = @intCast(v);
        if (g.getUint("qwen35.attention.head_count")) |v| cfg.n_heads = @intCast(v);
        if (g.getUint("qwen35.attention.head_count_kv")) |v| cfg.n_kv_heads = @intCast(v);
        if (g.getUint("qwen35.attention.key_length")) |v| cfg.head_dim = @intCast(v);
        if (g.getUint("qwen35.full_attention_interval")) |v| cfg.full_attention_interval = @intCast(v);
        if (g.getFloat("qwen35.attention.layer_norm_rms_epsilon")) |v| cfg.rms_eps = @floatCast(v);
        if (g.getUint("qwen35.ssm.state_size")) |v| cfg.gdn_head_dim = @intCast(v);
        if (g.getUint("qwen35.ssm.time_step_rank")) |_| {} // informational
        if (g.getUint("qwen35.ssm.conv_kernel")) |v| cfg.conv_kernel = @intCast(v);
        // qwen35.ssm.inner_size = value_dim = num_v_heads * head_dim
        if (g.getUint("qwen35.ssm.inner_size")) |v| {
            cfg.gdn_n_heads = @intCast(v / cfg.gdn_head_dim);
        }
        // qwen35.ssm.group_count = k/q head count (== v heads on 0.8B, half on 4B)
        if (g.getUint("qwen35.ssm.group_count")) |v| {
            cfg.gdn_num_k_heads = @intCast(v);
        }
        if (g.getFloat("qwen35.rope.freq_base")) |v| cfg.rope_freq_base = @floatCast(v);
        if (g.getUint("qwen35.rope.dimension_count")) |v| {
            cfg.rotary_dim = @intCast(v);
        }
        if (g.findKey("qwen35.rope.dimension_sections")) |v| {
            if (v.* == .array and v.array.element_type == .int32 and v.array.count == 4) {
                var r = gguf.arrayReader(&g, v.array);
                for (&cfg.rope_sections) |*sec| {
                    sec.* = @intCast(try gguf.arrayElemU64(&r, .int32));
                }
            }
        }
        if (g.getUint("tokenizer.ggml.eos_token_id")) |v| cfg.eos_token_id = @intCast(v);
        if (g.getUint("tokenizer.ggml.vocab_size")) |v| cfg.vocab = @intCast(v);

        var e = Engine{
            .alloc = alloc,
            .cfg = cfg,
            .backend_kind = backend_kind,
            .g = g,
            .mm = mm,
            .file = file,
            .io = io,
            .owns_io = null,
            .token_embd = .{ .f32 = &.{} },
            .output_weight = null,
            .output_norm = backend.Buf.empty,
            .exec = exec,
            .layers = &.{},
            .kv = &.{},
            .gdn = &.{},
            .n_threads = 1,
            .x = backend.Buf.empty,
            .x1 = backend.Buf.empty,
            .x2 = backend.Buf.empty,
            .qkv = backend.Buf.empty,
            .z = backend.Buf.empty,
            .ab = backend.Buf.empty,
            .gdn_out = backend.Buf.empty,
            .q = backend.Buf.empty,
            .k = backend.Buf.empty,
            .v = backend.Buf.empty,
            .attn = backend.Buf.empty,
            .mlp_h = backend.Buf.empty,
            .mlp_h1 = backend.Buf.empty,
            .logits = backend.Buf.empty,
            .delta = backend.Buf.empty,
            .kv_mem = backend.Buf.empty,
            .gamma_scratch = backend.Buf.empty,
            .gdn_prefill_gamma = backend.Buf.empty,
        };
        errdefer e.freeAll();

        e.n_threads = nt;
        // The thread pool belongs to the backend: it is how the *hardware*
        // parallelizes a matvec, and a GPU backend has no use for it.
        exec.setThreads(io, nt);

        e.token_embd = try exec.loadLinear("token_embd.weight");
        // 27B+ has an untied lm_head (`output.weight`, same [vocab,hidden]
        // layout); 0.8B/2B/4B tie it to the embedding (key absent -> null).
        e.output_weight = exec.loadLinear("output.weight") catch |err| switch (err) {
            error.MissingTensor => null,
            else => return err,
        };
        e.output_norm = try exec.loadVecF32("output_norm.weight");

        const n_layers = cfg.n_layers;
        // One EXTRA slot for the trailing nextn block when it exists: it lives
        // at index `n_layers` (blk.<n_layers>.*), so keeping it in the same
        // array lets every per-layer routine index it exactly like a decoder
        // layer.  `e.layers.len` is therefore NOT `cfg.n_layers` on 27B --
        // use `n_layers` (or `cfg.n_layers`) for loops over decoder layers.
        const n_slots = n_layers + @intFromBool(n_nextn > 0);
        e.layers = try alloc.alloc(LayerWeights, n_slots);
        // Every slot MUST start from the struct's declared defaults.  `alloc`
        // hands back uninitialised memory, and while the decoder loop fills all
        // of 0..n_layers, the nextn slot at index `n_layers` is only given the
        // fields its own block uses -- `ssm_conv1d` / `ssm_dt_bias` / `ssm_a` /
        // `ssm_norm` / `ssm_out` stay whatever the allocator left there.  They
        // are never read (the block is full-attention), but `deinit`'s
        // `freeLayerTyped` frees them unconditionally, and `vFreeVec` only
        // skips `len == 0` -- a garbage length walks off into the heap.
        // MEASURED: general protection fault inside `free` on EVERY shutdown of
        // a 27B run, i.e. the process always core-dumped after printing its
        // results, which is easy to miss when the harness only greps stdout.
        for (e.layers) |*slot| slot.* = .{
            .attn_norm = backend.Buf.empty,
            .post_attention_norm = backend.Buf.empty,
            .ffn_gate = undefined,
            .ffn_up = undefined,
            .ffn_down = undefined,
        };
        var loaded: usize = 0;
        errdefer for (e.layers[0..loaded]) |lw| e.freeLayerWeights(lw);
        for (0..n_layers) |li| {
            var b1: [64]u8 = undefined;
            var b2: [64]u8 = undefined;
            const idx: u32 = @intCast(li);
            const lw = LayerWeights{
                .attn_norm = try exec.loadVecF32(std.fmt.bufPrint(&b1, "blk.{d}.attn_norm.weight", .{idx}) catch unreachable),
                .post_attention_norm = try exec.loadVecF32(std.fmt.bufPrint(&b2, "blk.{d}.post_attention_norm.weight", .{idx}) catch unreachable),
                .ffn_gate = try exec.loadLinear(tname(&b1, "blk.{d}.ffn_gate.weight", idx)),
                .ffn_up = try exec.loadLinear(tname(&b2, "blk.{d}.ffn_up.weight", idx)),
                .ffn_down = try exec.loadLinear(tname(&b1, "blk.{d}.ffn_down.weight", idx)),
            };
            e.layers[li] = lw;
            loaded += 1;
        }

        // caches (alloc + init)
        e.kv = try alloc.alloc(KvCache, n_slots);
        e.gdn = try alloc.alloc(GdnState, n_slots);
        var inited: usize = 0;
        errdefer e.freeCachesUpTo(inited);
        const kv_dim: usize = @as(usize, cfg.n_kv_heads) * cfg.head_dim;
        if (!backend.kvTypeFits(kv_dim, kv_type)) {
            // Block-quantized KV needs whole 32-element blocks per row; the
            // named failure is better than a silently truncated last block.
            return error.KvTypeNotSupported;
        }
        const gdn_kv: usize = @as(usize, cfg.gdn_n_heads) * cfg.gdn_head_dim * cfg.gdn_head_dim;
        // 状态块数：1 块常规 + `spec_rs_seq` 块 verify 快照。不开推测解码时
        // 只有 1 块，与改动前完全一样（多出来的只有 `state_all` 这个字段）。
        const spec_rs_slabs: usize = 1 + (if (spec) spec_rs_seq else 0);
        const gdn_conv_dim: usize =
            2 * @as(usize, cfg.gdn_num_k_heads) * cfg.gdn_head_dim +
            @as(usize, cfg.gdn_n_heads) * cfg.gdn_head_dim;
        for (0..n_layers) |li| {
            if (cfg.isFullAttention(@intCast(li))) {
                const region = try exec.allocKv(kv_dim, max_seq);
                e.kv[li] = .{ .k = region.k, .v = region.v, .len = 0 };
                e.gdn[li] = .{ .conv = backend.Buf.empty, .conv_len = 0, .state = backend.Buf.empty, .state_all = backend.Buf.empty, .rs_idx = 0 };
            } else {
                e.kv[li] = .{ .k = backend.Buf.empty, .v = backend.Buf.empty, .len = 0 };
                e.gdn[li] = .{
                    .conv = try exec.allocBuf(gdn_conv_dim * (@as(usize, cfg.conv_kernel) - 1)),
                    .conv_len = 0,
                    .state_all = try exec.allocBuf(gdn_kv * spec_rs_slabs),
                    .state = backend.Buf.empty,
                    .rs_idx = 0,
                };
                // 活跃视图 = 第 0 块。
                e.gdn[li].state = e.gdn[li].state_all.span(0, gdn_kv);
                // The delta-rule state must start at zero.  The region is
                // backend memory (device under `--backend cuda`), so the reset
                // goes through the backend: a host `@memset` cannot reach it.
                // 所有块一起清零：快照块在写入前也会被读（见 `rs_idx`）。
                exec.zeroBuf(e.gdn[li].state_all);
            }
            inited += 1;
        }

        // Nextn block slot (`blk.<n_layers>.*`): a FULL-ATTENTION layer with
        // its own KV cache.  It must NOT share the main cache -- the drafter
        // runs over speculative tokens at positions the target has not
        // committed, so those rows would corrupt the target's attention.
        //
        // `cfg.isFullAttention(n_layers)` is WRONG here (64 % 4 == 0 -> it says
        // GDN), which is why the block's type is decided by the tensors it
        // actually ships, not by the periodic formula.
        if (n_nextn > 0) {
            const li = n_layers;
            const region = try exec.allocKv(kv_dim, max_seq);
            e.kv[li] = .{ .k = region.k, .v = region.v, .len = 0 };
            e.gdn[li] = .{ .conv = backend.Buf.empty, .conv_len = 0, .state = backend.Buf.empty, .state_all = backend.Buf.empty, .rs_idx = 0 };
            inited += 1;
        }

        // load the per-type weights now that layer type is known
        var loaded2: usize = 0;
        errdefer for (e.layers[0..loaded2]) |*lw| e.freeLayerTyped(lw);
        for (0..n_layers) |li| {
            const idx: u32 = @intCast(li);
            var b1: [64]u8 = undefined;
            var b2: [64]u8 = undefined;
            if (cfg.isFullAttention(@intCast(li))) {
                e.layers[li].attn_q = try exec.loadLinear(tname(&b1, "blk.{d}.attn_q.weight", idx));
                e.layers[li].attn_k = try exec.loadLinear(tname(&b2, "blk.{d}.attn_k.weight", idx));
                e.layers[li].attn_v = try exec.loadLinear(tname(&b1, "blk.{d}.attn_v.weight", idx));
                e.layers[li].attn_q_norm = try exec.loadVecF32(tname(&b2, "blk.{d}.attn_q_norm.weight", idx));
                e.layers[li].attn_k_norm = try exec.loadVecF32(tname(&b1, "blk.{d}.attn_k_norm.weight", idx));
                e.layers[li].attn_output = try exec.loadLinear(tname(&b2, "blk.{d}.attn_output.weight", idx));
            } else {
                e.layers[li].attn_qkv = try exec.loadLinear(tname(&b1, "blk.{d}.attn_qkv.weight", idx));
                e.layers[li].attn_gate = try exec.loadLinear(tname(&b2, "blk.{d}.attn_gate.weight", idx));
                e.layers[li].ssm_alpha = try exec.loadLinear(tname(&b1, "blk.{d}.ssm_alpha.weight", idx));
                e.layers[li].ssm_beta = try exec.loadLinear(tname(&b2, "blk.{d}.ssm_beta.weight", idx));
                e.layers[li].ssm_conv1d = try exec.loadVecF32(tname(&b1, "blk.{d}.ssm_conv1d.weight", idx));
                e.layers[li].ssm_dt_bias = try exec.loadVecF32(tname(&b2, "blk.{d}.ssm_dt.bias", idx));
                e.layers[li].ssm_a = try exec.loadVecF32(tname(&b1, "blk.{d}.ssm_a", idx));
                e.layers[li].ssm_norm = try exec.loadVecF32(tname(&b2, "blk.{d}.ssm_norm.weight", idx));
                e.layers[li].ssm_out = try exec.loadLinear(tname(&b1, "blk.{d}.ssm_out.weight", idx));
            }
            loaded2 += 1;
        }

        // ---- nextn / MTP drafter block (speculative decoding, stage 1) ----
        // Same full-attention weight set as the decoder layers (the block ships
        // attn_q/k/v/q_norm/k_norm/output, no ssm_*), loaded at index
        // `n_layers`, plus the four MTP-only tensors.  Loaded even though
        // nothing consumes it yet: `spec` needs it present to arm the drafter,
        // and a model without one stays null (`n_nextn == 0`).
        if (n_nextn > 0) {
            const li = n_layers;
            const idx: u32 = @intCast(n_layers);
            var b1: [64]u8 = undefined;
            var b2: [64]u8 = undefined;
            // The two layer norms are loaded by the FIRST loop above, which only
            // walks the decoder layers -- the nextn slot gets them here, or
            // `forwardFull`/`forwardMlp` would scale by an empty weight vector.
            e.layers[li].attn_norm = try exec.loadVecF32(tname(&b1, "blk.{d}.attn_norm.weight", idx));
            e.layers[li].post_attention_norm = try exec.loadVecF32(tname(&b2, "blk.{d}.post_attention_norm.weight", idx));
            e.layers[li].ffn_gate = try exec.loadLinear(tname(&b1, "blk.{d}.ffn_gate.weight", idx));
            e.layers[li].ffn_up = try exec.loadLinear(tname(&b2, "blk.{d}.ffn_up.weight", idx));
            e.layers[li].ffn_down = try exec.loadLinear(tname(&b1, "blk.{d}.ffn_down.weight", idx));
            e.layers[li].attn_q = try exec.loadLinear(tname(&b1, "blk.{d}.attn_q.weight", idx));
            e.layers[li].attn_k = try exec.loadLinear(tname(&b2, "blk.{d}.attn_k.weight", idx));
            e.layers[li].attn_v = try exec.loadLinear(tname(&b1, "blk.{d}.attn_v.weight", idx));
            e.layers[li].attn_q_norm = try exec.loadVecF32(tname(&b2, "blk.{d}.attn_q_norm.weight", idx));
            e.layers[li].attn_k_norm = try exec.loadVecF32(tname(&b1, "blk.{d}.attn_k_norm.weight", idx));
            e.layers[li].attn_output = try exec.loadLinear(tname(&b2, "blk.{d}.attn_output.weight", idx));
            loaded2 += 1;
            // `eh_proj` is [2*hidden, hidden]; `Nextn` owns it because it is
            // not part of a decoder layer's weight set.
            const eh_proj = try exec.loadLinear(tname(&b1, "blk.{d}.nextn.eh_proj.weight", idx));
            errdefer exec.freeLinear(eh_proj);
            e.nextn = Nextn{
                .input = try exec.allocBuf(2 * cfg.hidden),
                .eh_proj = eh_proj,
                .enorm = try exec.loadVecF32(tname(&b2, "blk.{d}.nextn.enorm.weight", idx)),
                .hnorm = try exec.loadVecF32(tname(&b1, "blk.{d}.nextn.hnorm.weight", idx)),
                .shared_head_norm = try exec.loadVecF32(tname(&b2, "blk.{d}.nextn.shared_head_norm.weight", idx)),
                .li = li,
            };
        }

        // scratch
        // scratch: backend-owned regions (release = Exec.freeScratch at teardown)
        e.x = try e.exec.allocBuf(cfg.hidden);
        e.x1 = try e.exec.allocBuf(cfg.hidden);
        e.x2 = try e.exec.allocBuf(cfg.hidden);
        const gdn_nv: usize = @as(usize, cfg.gdn_n_heads);
        const gdn_kd: usize = @as(usize, cfg.gdn_head_dim);
        const gdn_value_dim: usize = gdn_nv * gdn_kd;
        // q8_scratch holds the Q8_0-quantized INPUT x of each Q8_0 matvec, so
        // its size is driven by the widest matvec INPUT (cols), not by any
        // output row: ffn_down's input is n_ff, attn_output's is q_dim,
        // ssm_out's is gdn_value_dim, token_embd / lm_head / in_proj /
        // ffn_gate / ffn_up's is hidden.
        // Sizing to conv_dim / 2*q_dim (the OUTPUT widths of attn_qkv / attn_q)
        // under-allocates ffn_down's scratch, because n_ff is the widest input
        // in every Qwen3.5 config (27B: n_ff=17408 vs conv_dim=10240,
        // q_dim=6144, hidden=5120): the quantize loop writes past the end of
        // the arena, so the assert fires in Debug and it silently corrupts the
        // heap in ReleaseFast.  Same root cause as the CUDA-side 2B/4B fix
        // (see main_cuda.zig: max(hidden, value_dim, q_dim, n_ff)).
        //
        // Size from the config, then MEASURE the widest input over the weights
        // that were actually loaded and assert the scratch covers it.  A
        // sizing miss (n_ff left out of the max, a new layer shape, a config
        // key read wrong) then fails here at load time with a named error,
        // instead of as an out-of-bounds write inside the first matvec.
        const max_q8_cols = maxQ8Cols(cfg);
        try e.exec.allocScratch(max_q8_cols);

        // measured widest input, straight from the loaded tensors
        var measured: usize = 0;
        measured = @max(measured, matvecCols(e.token_embd, cfg.vocab));
        if (e.output_weight) |ow| measured = @max(measured, matvecCols(ow, cfg.vocab));
        for (0..n_layers) |li| {
            const lw = &e.layers[li];
            if (cfg.isFullAttention(@intCast(li))) {
                const q_dim: usize = @as(usize, cfg.n_heads) * cfg.head_dim;
                const o_kv_dim: usize = @as(usize, cfg.n_kv_heads) * cfg.head_dim;
                measured = @max(measured, matvecCols(lw.attn_q.?, q_dim));
                measured = @max(measured, matvecCols(lw.attn_k.?, o_kv_dim));
                measured = @max(measured, matvecCols(lw.attn_v.?, o_kv_dim));
                measured = @max(measured, matvecCols(lw.attn_output.?, cfg.hidden));
            } else {
                const conv_dim: usize = 2 * @as(usize, cfg.gdn_num_k_heads) * cfg.gdn_head_dim + gdn_value_dim;
                measured = @max(measured, matvecCols(lw.attn_qkv.?, conv_dim));
                measured = @max(measured, matvecCols(lw.attn_gate.?, gdn_value_dim));
                measured = @max(measured, matvecCols(lw.ssm_alpha.?, gdn_nv));
                measured = @max(measured, matvecCols(lw.ssm_beta.?, gdn_nv));
                measured = @max(measured, matvecCols(lw.ssm_out.?, gdn_value_dim));
            }
            measured = @max(measured, matvecCols(lw.ffn_gate, cfg.n_ff));
            measured = @max(measured, matvecCols(lw.ffn_up, cfg.n_ff));
            measured = @max(measured, matvecCols(lw.ffn_down, cfg.hidden));
        }
        std.debug.assert(measured == max_q8_cols);

        e.qkv = try e.exec.allocBuf(@max(gdn_conv_dim, 2 * @as(usize, cfg.n_heads) * cfg.head_dim));
        e.z = try e.exec.allocBuf(@max(gdn_value_dim, @as(usize, cfg.n_heads) * cfg.head_dim));
        e.ab = try e.exec.allocBuf(2 * gdn_nv);
        e.gdn_out = try e.exec.allocBuf(@max(gdn_value_dim, @as(usize, cfg.n_heads) * cfg.head_dim));
        e.q = try e.exec.allocBuf(@as(usize, cfg.n_heads) * cfg.head_dim);
        e.k = try e.exec.allocBuf(@as(usize, cfg.n_kv_heads) * cfg.head_dim);
        e.v = try e.exec.allocBuf(@as(usize, cfg.n_kv_heads) * cfg.head_dim);
        e.attn = try e.exec.allocBuf(max_seq);
        e.mlp_h = try e.exec.allocBuf(cfg.n_ff);
        e.mlp_h1 = try e.exec.allocBuf(cfg.n_ff);
        e.logits = try e.exec.allocBuf(cfg.vocab);
        // `nv` slices of `vd` each, NOT one shared `vd` buffer: `gdnStepRows`
        // gives every head its own block, and `heads` blocks writing one shared
        // km/delta slice would be a silent race (wrong tokens, no crash).
        //
        // ONE TOKEN is all these need: nothing reads them (they are scratch the
        // kernel fills), so the decode path keeps the minimal size.  The prefill
        // chunk path needs `K` rows and REALLOCATES them below -- see the
        // `gdnStepRowsChunk` note there.
        e.delta = try e.exec.allocBuf(gdn_nv * gdn_kd);
        e.kv_mem = try e.exec.allocBuf(gdn_nv * gdn_kd);
        e.gamma_scratch = try e.exec.allocBuf(@as(usize, cfg.gdn_n_heads));
        e.sample_scratch = try e.exec.allocBuf(cfg.vocab);

        // ---- K-token batched prefill scratch (K = 1024) ----
        // K is the number of prompt tokens folded into one batched chunk, and it
        // is THE knob that decides prefill throughput: every layer's weight
        // matrix is streamed from HBM once per chunk, so the total weight
        // traffic is (n_prompt / K) x model_bytes.  Nothing else on the prefill
        // path scales that way.
        //
        // Measured on 27B IQ3_XXS, 525-token prompt (tok/s, this kernel only
        // needs Q35_MMQ_MAX_TOKENS >= K):
        //     K =  128 -> 170.3   525 = 5 chunks, i.e. 51.7 GB of weight reads
        //     K =  256 -> 184.1
        //     K =  512 -> 187.3   525 = 2 chunks
        //     K = 1024 -> 200.5   525 = 1 chunk,  i.e. 10.3 GB
        // The gain is sublinear in the traffic ratio (5x fewer bytes, 18%
        // faster) because at these sizes the kernel is latency-bound, not
        // bandwidth-bound: one launch costs almost the same whether it serves
        // 105 or 525 tokens (2.68 vs 2.83 ms/call on the same shapes).  Bucket
        // sizes 4x the prompt would stop helping; 1024 covers the realistic
        // prompt range and keeps 4 chunks for a 4k-token prompt.
        //
        // The buffers below are all sized from K, so raising it further is a
        // matter of activation memory only (measured: bufs_MiB 22 -> 157 at
        // K=1024 on the 27B).
        const K: usize = 1024;
        // `gdnStepRowsChunk` runs the state recurrence for ALL K tokens of a
        // chunk in one launch, striding km/delta per row, so those two scratch
        // buffers must cover K rows (gdn_nv * gdn_kd per row).  Sized here rather
        // than with the other GDN scratch above because `K` is defined here.
        //
        // `no_batch` (decode) still uses the one-row `gdnStepRows` and only ever
        // touches the first row, so allocating the full chunk unconditionally
        // costs 2 * K * 16 * 128 * 4 B = 16 MiB of device scratch and nothing
        // else.  A one-row buffer would make the backend's length gate reject
        // the chunk call, which silently falls back to the CPU oracle -- correct
        // output, ~10x slower, and it then walks off the end of the buffer.
        e.delta = try e.exec.allocBuf(K * gdn_nv * gdn_kd);
        e.kv_mem = try e.exec.allocBuf(K * gdn_nv * gdn_kd);
        const q_dimK: usize = @as(usize, cfg.n_heads) * cfg.head_dim;
        const kv_dimK: usize = @as(usize, cfg.n_kv_heads) * cfg.head_dim;
        e.nbatch = if (no_batch) 0 else K;
        e.xK = try e.exec.allocBuf(K * cfg.hidden);
        e.x1K = try e.exec.allocBuf(K * cfg.hidden);
        e.qkvK = try e.exec.allocBuf(K * @max(gdn_conv_dim, 2 * q_dimK));
        e.gateK = try e.exec.allocBuf(K * @max(gdn_value_dim, q_dimK));
        e.abK = try e.exec.allocBuf(K * 2 * gdn_nv);
        e.qK = try e.exec.allocBuf(K * q_dimK);
        e.kK = try e.exec.allocBuf(K * kv_dimK);
        e.vK = try e.exec.allocBuf(K * kv_dimK);
        e.goutK = try e.exec.allocBuf(K * @max(gdn_value_dim, q_dimK));
        // Sized for a full chunk even when `nbatch == 0` (then it is unused):
        // `forwardGdnPrelude` only runs on the batched path.  Must be `K`, not a
        // literal: the chunk size is the one knob that decides how many times
        // the weights are streamed, and a stale 128 here aborts the run as soon
        // as K grows (assert in `span`).
        e.gdn_prefill_gamma = try e.exec.allocBuf(K * @as(usize, cfg.gdn_n_heads));
        e.x2K = try e.exec.allocBuf(K * cfg.hidden);
        e.mlpK = try e.exec.allocBuf(K * cfg.n_ff);
        e.mlpK1 = try e.exec.allocBuf(K * cfg.n_ff);
        e.logitsK = try e.exec.allocBuf(spec_verify_rows * cfg.vocab);
        // 推测解码回滚的每层投影缓存（见 `gdn_in`）。加载时就分配：
        // "要不要跑推测解码"在 `load` 之前就定了（`spec` 参数 + 模型是否有
        // nextn 块），把这块拖到第一次 `generateSpec` 只会让分配点离使用点
        // 更远，也让 `specSnapshotAlloc` 混着做两件不相干的事。
        try e.allocGdnIn(alloc, spec);
        try e.exec.allocScratchK(max_q8_cols, K);
        // 推测解码的状态快照（`spec_snap`）。与 `gdn_in` 同一原则：要不要跑
        // 推测解码在加载时就已确定，快照尺寸（`gdn[i].conv/state` 的长度）也
        // 在逐层分配时定下来了，所以一起在这里分配。普通运行两者都不分配。
        try e.allocSpecSnapshot(alloc, spec);
        prof_io = io;
        return e;
    }

    fn tname(buf: []u8, comptime fmt: []const u8, idx: u32) []const u8 {
        return std.fmt.bufPrint(buf, fmt, .{idx}) catch unreachable;
    }


    /// Release the two norm vectors every layer has.  The `LinearW` fields are
    /// zero-copy views into the GGUF mapping (both the `.quant` and `.f32`
    /// arms), so there is nothing to release for those: the views die with the
    /// mapping.  The norm vectors are backend regions (`loadVecF32`), so they
    /// are released through the backend, not the engine allocator.
    fn freeLayerWeights(self: *Engine, lw: LayerWeights) void {
        self.exec.freeVec(lw.attn_norm);
        self.exec.freeVec(lw.post_attention_norm);
    }

    fn freeLayerTyped(self: *Engine, lw: *LayerWeights) void {
        // The `?LinearW` fields (attn_qkv / ssm_alpha / ffn_gate / ...) are all
        // zero-copy views into the mapping -- nothing to free.  The 1-D f32
        // vectors are backend regions (`loadVecF32`); a `Buf.empty` (a layer
        // type that does not use that vector) is a no-op in `freeVec`, so all
        // six are released unconditionally.
        self.exec.freeVec(lw.ssm_conv1d);
        self.exec.freeVec(lw.ssm_dt_bias);
        self.exec.freeVec(lw.ssm_a);
        self.exec.freeVec(lw.ssm_norm);
        self.exec.freeVec(lw.attn_q_norm);
        self.exec.freeVec(lw.attn_k_norm);
    }

    /// Release the KV / GDN regions for the first `n` layers.
    ///
    /// P2c: these are backend regions now, so there is nothing per-cache to
    /// free here -- `Exec.allocBuf` hands out regions that `Exec.freeScratch`
    /// releases all at once at teardown.  What remains is the *reset* the
    /// caller may want, and even that is a no-op: a re-`load` builds a fresh
    /// backend.  The `O(n)` host frees that used to live here are gone with
    /// the host allocations.
    ///
    /// Robust to partially-constructed state: `load`'s `errdefer freeAll` may
    /// run before kv/gdn are allocated (an unsupported tensor type fails at
    /// `loadLinear` first), in which case `self.kv`/`self.gdn` are still the
    /// empty sentinel.
    fn freeCachesUpTo(self: *Engine, n: usize) void {
        const nk = @min(n, self.kv.len);
        for (self.kv[0..nk]) |*c| c.* = .{ .k = backend.Buf.empty, .v = backend.Buf.empty, .len = 0 };
        const ng = @min(n, self.gdn.len);
        for (self.gdn[0..ng]) |*s| s.* = .{ .conv = backend.Buf.empty, .conv_len = 0, .state = backend.Buf.empty, .state_all = backend.Buf.empty, .rs_idx = 0 };
    }

    /// Number of layer slots actually allocated: decoder layers plus the nextn
    /// block when present.  `cfg.n_layers` counts ONLY decoder layers (it is
    /// what every per-layer loop and `isFullAttention` mean by "layer"), so
    /// anything that walks the whole `layers`/`kv`/`gdn` arrays must use this.
    fn nSlots(self: *const Engine) usize {
        return self.cfg.n_layers + @intFromBool(self.nextn != null);
    }

    fn freeAll(self: *Engine) void {
        const a = self.alloc;
        // `token_embd` / `output_weight` are zero-copy views into the mapping;
        // the layer `LinearW`s are the same (see `freeLayerWeights`).
        self.exec.freeVec(self.output_norm);
        self.exec.freeScratch();
        if (self.layers.len > 0) {
            for (self.layers) |lw| self.freeLayerWeights(lw);
            for (self.layers) |*lw| self.freeLayerTyped(lw);
            a.free(self.layers);
        }
        self.freeCachesUpTo(self.nSlots());
        if (self.kv.len > 0) a.free(self.kv);
        if (self.gdn.len > 0) a.free(self.gdn);
        // `gdn_in` 与 `spec_snap` 只有 host 侧的数组本身要在这里回收：它们指向的
        // `qkv`/`gamma`/`conv`/`state` 是 `allocBuf` 区域，上面 `freeScratch()`
        // 已经连带释放。
        if (self.gdn_in.len > 0) a.free(self.gdn_in);
        if (self.spec_snap.len > 0) a.free(self.spec_snap);
        if (self.spec_dbg_a.len > 0) a.free(self.spec_dbg_a);
        if (self.spec_dbg_b.len > 0) a.free(self.spec_dbg_b);
        // `tap_layers` is a host allocation made by `specTapEnable` (the
        // `tap_hidden` region itself is backend scratch and goes with
        // `freeScratch()` above).
        if (self.tap_layers.len > 0) a.free(self.tap_layers);
        // Transient activations are backend-owned (`Exec.allocBuf`), so
        // `exec.freeScratch()` above already released every one of them: the
        // engine must NOT free them individually (double free).
    }

    pub fn deinit(self: *Engine) void {
        self.freeAll();
        self.g.deinit();
        self.mm.destroy(self.io);
        self.file.close(self.io);
        // destroy the Threaded instance iff `load` created it
        if (self.owns_io) |t| {
            self.alloc.destroy(t);
            t.deinit();
        }
    }

    // -----------------------------------------------------------------
    // Forward: one token at position `pos`
    // -----------------------------------------------------------------

    /// Run the whole model for one token and return the raw logits (no
    // -----------------------------------------------------------------
    // Forward: one token at position `pos`
    // -----------------------------------------------------------------

    // Forward: one token at position `pos`
    // -----------------------------------------------------------------

    /// Run the whole model for one token and return the raw logits (no
    // -----------------------------------------------------------------
    // Forward: one token at position `pos`
    // -----------------------------------------------------------------

    /// Run the whole model for one token and return the raw logits as a backend
    /// `Buf` -- device memory on the GPU backend, so the caller must go through
    /// `Exec.argmax` / `Exec.sample` (or `copyToHost` explicitly) rather than
    /// indexing it.  (P2d: this used to return `self.logits.slice()`, which was
    /// the host dereference that forced a D2H of the whole vocabulary per token.)
    /// No sampling, no PRNG draw.  Prefill uses this so the sampler's coin
    /// stream is untouched by the prompt.
    pub fn forwardLogits(self: *Engine, token: u32, pos: u32) Buf {
        const c = self.cfg;
        const hidden: usize = c.hidden;
        const io = self.io;
        if (prof) {
            prof_gdn = 0;
            prof_full = 0;
            prof_mlp = 0;
            prof_lm = 0;
            prof_emb = 0;
        }
        const te = Io.Clock.Timestamp.now(io, .awake);
        // embedding lookup: dequantize the token row straight into x
        self.embedTo(token, self.x.span(0, hidden));
        if (prof) prof_emb += profNs(io, te);

        // One decode step, from the first layer to the logits, is the only
        // stretch of this engine whose kernels are known before they run: the
        // weights are resident, every shape is fixed, and `pos`/`slot` are the
        // only arguments that move.  A device backend can therefore record it
        // once and replay it, and it needs to be told where the stretch starts
        // and ends -- it never sees the model's control flow, only single ops.
        //
        // The boundary STARTS AFTER the embedding lookup on purpose: that lookup
        // picks its row on the HOST (`base + token * row_bytes`), so a captured
        // dequantize node would re-read the first token's row forever.  One
        // dequantize row is a few microseconds, so keeping it out of the graph
        // costs nothing measurable and removes a whole class of staleness.
        //
        // The boundary ENDS at `stepEnd`, which the caller invokes -- and the
        // caller samples BEFORE that, so the sampler is inside the span.  That
        // is what lets a device backend put sampling in its captured graph (the
        // way llama.cpp keeps the sampler as a node of the model graph) instead
        // of paying a launch + full-vocabulary D2H after the graph has finished.
        // The token id still has to reach the host before the next step can
        // exist, so the download itself happens at `stepEnd`; see
        // `VTable.sampledToken` for the split.
        self.exec.stepBegin(pos);
        // `self.layers` has one extra slot when the model ships a nextn block
        // (see `nSlots`); a normal forward must stop at the decoder layers, or
        // it would run the drafter as layer 64 of every token.
        for (self.layers[0..self.cfg.n_layers], 0..) |*lw, li| {
            if (c.isFullAttention(@intCast(li))) {
                const a = Io.Clock.Timestamp.now(io, .awake);
                self.forwardFull(lw, li, pos);
                if (prof) prof_full += profNs(io, a);
            } else {
                const a = Io.Clock.Timestamp.now(io, .awake);
                self.forwardGdn(lw, li);
                if (prof) prof_gdn += profNs(io, a);
            }
            const a = Io.Clock.Timestamp.now(io, .awake);
            self.forwardMlp(lw);
            if (prof) prof_mlp += profNs(io, a);

            // Speculative tap, single-token form: row 0 of the plane.  Written
            // after the MLP's residual add, which is the layer's output -- the
            // value `forwardChunk` copies too (`Engine.tap_hidden` layout).
            //
            // This is what lets a spec round reuse the hidden states the TARGET
            // already computed during ordinary decode / verification instead of
            // re-forwarding its context window every round.  At 16 rows that
            // re-forward cost 16 x the target's per-token work per round, which
            // measured 651 s for 32 tokens before this existed.
            if (self.tap_hidden.len != 0) {
                for (self.tap_layers, 0..) |tl, ti| {
                    if (tl == li) {
                        self.exec.copyRange(
                            self.tap_hidden.span(ti * self.nbatch * self.cfg.hidden, self.cfg.hidden),
                            self.x.span(0, self.cfg.hidden),
                        );
                        break;
                    }
                }
            }
        }
        const b = Io.Clock.Timestamp.now(io, .awake);

        self.exec.rmsnorm(self.x1, self.x, self.output_norm, c.rms_eps);
        // lm_head: untied `output.weight` when present, else tied token_embd.
        // Both are [vocab, hidden] row-major.
        if (self.output_weight) |ow| {
            switch (ow) {
                .quant, .f32 => self.exec.matvec(self.logits, ow, self.x1),
            }
        } else {
            // tied lm_head: logits = W_emb @ x1 with W_emb row-major [vocab, hidden]
            switch (self.token_embd) {
                .quant => |qrow| self.exec.matvec(self.logits, .{ .quant = qrow }, self.x1),
                .f32 => |m| self.exec.matvec(self.logits, .{ .f32 = m }, self.x1),
            }
        }
        if (prof) prof_lm += profNs(io, b);
        return self.logits;
    }

    pub fn profReport() void {
        const tot: f64 = prof_gdn + prof_full + prof_mlp + prof_lm;
        std.debug.print("PROF emb={d:.2}ms gdn={d:.2}ms full={d:.2}ms mlp={d:.2}ms lm={d:.2}ms total={d:.2}\n", .{ prof_emb / 1e6, prof_gdn / 1e6, prof_full / 1e6, prof_mlp / 1e6, prof_lm / 1e6, tot / 1e6 });
    }

    /// Forward + sample one token.  Greedy (argmax) when the sampler's
    /// temperature is 0 — the historical default.
    pub fn step(self: *Engine, token: u32, pos: u32) u32 {
        // One decode step, from the embedding lookup to the logits, is the only
        // stretch of this engine whose kernels are known before they run: weights
        // are resident, every shape is fixed, and `pos`/`slot` are the only
        // arguments that move.  A device backend can therefore record it once and
        // replay it, and it needs to be told where the stretch starts and ends --
        // it never sees the model's control flow, only individual ops.
        //
        // The boundary deliberately EXCLUDES `sample`: its result is the next
        // token, so it has to reach the host before the next step exists.  A
        // backend that ignores the hint (the CPU one does) is unaffected.
        const logits = self.forwardLogits(token, pos);
        // SAMPLE INSIDE THE SPAN, DOWNLOAD AT THE BOUNDARY.  A backend that
        // captures the step into a CUDA graph must record the sampler as part of
        // it; the 4-byte token download cannot be captured, so it happens in
        // `stepEnd` and `sampledToken` collects it.  For the CPU backend
        // `sample` is synchronous and `sampledToken` returns its argument
        // unchanged, so this split costs nothing there.
        const hinted = self.exec.sample(logits, self.sample_scratch, &self.sampler);
        self.exec.stepEnd();
        return self.exec.sampledToken(hinted);
    }

    /// Dequantize the embedding row of `token` into `dst` (hidden wide).
    fn embedTo(self: *Engine, token: u32, dst: Buf) void {
        const hidden: usize = self.cfg.hidden;
        std.debug.assert(dst.len == hidden);
        // Expanding one row to f32 is the one place the model needs *values*
        // rather than a dot product, so it is a backend call: which
        // dequantizer runs (Zig reference on CPU, a device kernel on GPU) is
        // the backend's business.  Q8_0's 32-element framing is handled
        // there rather than special-cased here.
        switch (self.token_embd) {
            .f32 => |m| self.exec.copyRange(dst, Buf.wrap(m[token * hidden ..][0..hidden])),
            .quant => |qrow| self.exec.dequantRow(dst, qrow, token),
        }
    }

    // -----------------------------------------------------------------
    // K-token batched prefill chunk
    // -----------------------------------------------------------------

    /// Run `tokens.len` (<= K) tokens of prefill starting at position `pos0`
    /// and return the logits of the LAST token.  Every weight-reading linear
    /// projection (qkv / gate / alpha-beta / q / k / v / out-proj / ffn
    /// gate-up-down) is executed ONCE per chunk as a K-token batched matvec,
    /// so each layer's weight bytes pass through cache once for K tokens; the
    /// state kernels (GDN conv + 128x128 state, full-attention scores, KV
    /// cache writes) run token-by-token in exactly the serial order, so the
    /// KV cache and GDN state end up bit-identical to the serial prefill.
    pub fn forwardChunk(self: *Engine, tokens: []const u32, pos0: u32) Buf {
        const k = tokens.len;
        std.debug.assert(k > 0 and k <= self.nbatch);
        // A single-token chunk is just a serial step, but it is reached from
        // `generate`'s PREFILL loop, which (unlike the decode loop) does not call
        // `exec.stepEnd`.  Delegating straight to `forwardLogits` would therefore
        // open a backend decode-graph capture (`stepBegin`) that nothing ever
        // closes, and the capture would run on into the first real decode step --
        // the graph would then hold two tokens' worth of nodes and replay garbage
        // from the first replayed token.  Measured 2026-11: with a 17-token
        // prompt the capture recorded 120 rope64 nodes (60 from pos 16 and 60 from
        // pos 17) instead of 60.  Close the span here so prefill can never leak
        // one into the decode boundary.
        if (k == 1) {
            const out = self.forwardLogits(tokens[0], pos0);
            self.exec.stepEnd();
            return out;
        }

        const c = self.cfg;
        const hidden: usize = c.hidden;
        const q_dim: usize = @as(usize, c.n_heads) * c.head_dim;
        const kv_dim: usize = @as(usize, c.n_kv_heads) * c.head_dim;
        const nv: usize = c.gdn_n_heads;
        const nk: usize = c.gdn_num_k_heads;
        const kd: usize = c.gdn_head_dim;
        const conv_dim: usize = 2 * nk * kd + nv * kd;
        const value_dim: usize = nv * kd;
        const n_ff: usize = c.n_ff;

        var t: usize = 0;
        while (t < k) : (t += 1) self.embedTo(tokens[t], self.xK.span(t * hidden, hidden));

        // `self.layers` has one extra slot when the model ships a nextn block
        // (see `nSlots`); a normal forward must stop at the decoder layers, or
        // it would run the drafter as layer 64 of every token.
        for (self.layers[0..self.cfg.n_layers], 0..) |*lw, li| {
            // ---- RMSNorm of the whole chunk in ONE launch, then the batched
            // projections (weight bytes read once for the whole chunk) ----
            // `rmsnormRows` is `k` consecutive single-row blocks of `k_rmsnorm`:
            // each block reduces its own row with the same order, so the result
            // is bit-identical to the per-token loop it replaces.  The
            // per-token loop issued `k` launches of a 5 us kernel per layer per
            // norm, which was 5.6% of prefill device time on 27B IQ3_XXS.
            self.exec.rmsnormRows(self.x1K.span(0, k * hidden), self.xK.span(0, k * hidden), lw.attn_norm, k, c.rms_eps);

            if (c.isFullAttention(@intCast(li))) {
                self.exec.matvecBatchK(self.qkvK.span(0, k * 2 * q_dim), lw.attn_q.?, self.x1K, k);
                self.exec.matvecBatchK(self.kK.span(0, k * kv_dim), lw.attn_k.?, self.x1K, k);
                self.exec.matvecBatchK(self.vK.span(0, k * kv_dim), lw.attn_v.?, self.x1K, k);
                // ---- RoPE ----
                // MOVED: both q's and k's rotation now happen inside
                // `forwardFullChunk`, AFTER the q_norm/k_norm pass.  Doing them
                // here was wrong twice over:
                //   * `qK` is still empty at this point (it is produced by the
                //     deinterleave inside `forwardFullChunk`), so q was rotated
                //     before it existed and the real q was never rotated;
                //   * `kK` WAS populated, but rotating it before `k_norm` means
                //     the norm then rescales each head -- and since a head's
                //     norm depends on the rotated values' sum of squares, the
                //     rotation is not preserved.  The reference order
                //     (`forwardFull`) is norm -> rope.
                // Symptoms this caused: row 0 of a chunk was exact (pos 0 is
                // the RoPE identity) and every later row drifted -- 5.9e-3 in
                // the first full-attention layer's output, enough to change the
                // first sampled token of a 17-token prompt.
                t = 0;
                self.forwardFullChunk(lw, li, k);
                self.exec.matvecBatchK(self.x2K.span(0, k * hidden), lw.attn_output.?, self.goutK, k);
            } else {
                self.exec.matvecBatchK(self.qkvK.span(0, k * conv_dim), lw.attn_qkv.?, self.x1K, k);
                self.exec.matvecBatchK(self.gateK.span(0, k * value_dim), lw.attn_gate.?, self.x1K, k);
                self.exec.matvecBatchK(self.abK.span(0, k * nv), lw.ssm_alpha.?, self.x1K, k);
                self.exec.matvecBatchK(self.abK.span(k * nv, k * nv), lw.ssm_beta.?, self.x1K, k);
                // 为推测解码回滚保留 PRE-prelude 的投影。必须是原始行：
                // 回滚只对 `rows` 个 token 重跑 prelude，而 prelude 会推进卷积
                // 历史并就地重写 `qkv`/`ab`——这两样都无法从 POST-prelude 的
                // 字节恢复（实测：缓存那些字节会让 conv 停在 `k` 深，而 state
                // 已退到 `rows`）。
                if (li < self.gdn_in.len) {
                    const gi = &self.gdn_in[li];
                    if (gi.qkv.len >= k * conv_dim and gi.gamma.len >= k * 2 * nv) {
                        self.exec.copyRange(gi.qkv.span(0, k * conv_dim), self.qkvK.span(0, k * conv_dim));
                        self.exec.copyRange(gi.gamma.span(0, k * 2 * nv), self.abK.span(0, k * 2 * nv));
                    }
                }
                t = 0;
                // Chunk-wide GDN prelude: conv+silu, gamma/beta and the
                // l2norm/scale passes do not touch the running state, and
                // `gdnStepRows` only READS qkv, so all of it can run before the
                // first state update.  The conv history is the one ordering
                // constraint and it stays ascending inside the call.  This is
                // what turns ~14 launches per token into 4 per chunk plus the
                // per-token state step.
                self.forwardGdnPrelude(lw, li, k);
                self.forwardGdnCoreChunk(li, k);
                // Chunk-wide gated RMSNorm + silu gate, hoisted out of the
                // token loop above; `ssm_out` is the first reader of `goutK`.
                self.forwardGdnGated(lw, k);
                self.exec.matvecBatchK(self.x2K.span(0, k * hidden), lw.ssm_out.?, self.goutK, k);
            }

            // ---- residual add: the `k` rows are CONSECUTIVE (`xK`/`x2K` are
            // `k * hidden` contiguous), so one call covers the chunk.  This was a
            // per-token loop (one launch per token per layer: 3353 launches for
            // a 525-token prefill, which PROFILE showed at 2.7%).  `addResid` is
            // elementwise with no per-row state, so the merge is exact.
            self.exec.addResid(self.xK.span(0, k * hidden), self.x2K.span(0, k * hidden));

            // ---- MLP: chunk-wide norm, batched gate/up, chunk-wide SwiGLU,
            // batched down, residual ----
            self.exec.rmsnormRows(self.x1K.span(0, k * hidden), self.xK.span(0, k * hidden), lw.post_attention_norm, k, c.rms_eps);
            self.exec.matvecBatchK(self.mlpK.span(0, k * n_ff), lw.ffn_gate, self.x1K, k);
            self.exec.matvecBatchK(self.mlpK1.span(0, k * n_ff), lw.ffn_up, self.x1K, k);
            // SwiGLU over the whole chunk: `gate[i] = silu(gate[i]) * up[i]` is
            // elementwise and each token's `n_ff` run is independent, so one
            // launch over `k * n_ff` is the same arithmetic as `k` launches of
            // an `n_ff`-long one.
            self.exec.swiglu(self.mlpK.span(0, k * n_ff), self.mlpK1.span(0, k * n_ff));
            self.exec.matvecBatchK(self.x2K.span(0, k * hidden), lw.ffn_down, self.mlpK, k);
            self.exec.addResid(self.xK.span(0, k * hidden), self.x2K.span(0, k * hidden));

            // ---- speculative-decoding tap: copy this layer's OUTPUT hidden
            // state into row `ti` of `tap_hidden`.  Placed after the second
            // `addResid` so it sees the same value the next layer's
            // `rmsnormRows` will consume -- i.e. the layer's output, which is
            // what DSpine's `fc` projection was trained on.
            //
            // Guarded by `tap_hidden.len != 0` (disabled by default).  `ti` is
            // found by a linear scan over `tap_layers`, which is at most 5
            // entries, once per layer: ~320 compares per chunk, not per token.
            if (self.tap_hidden.len != 0) {
                for (self.tap_layers, 0..) |tl, ti| {
                    if (tl == li) {
                        self.exec.copyRange(
                            self.tap_hidden.span(ti * self.nbatch * hidden, k * hidden),
                            self.xK.span(0, k * hidden),
                        );
                        break;
                    }
                }
            }
        }

        // ---- last token: final RMSNorm + lm_head on a single token (serial
        // path, so the first sampled logits are bit-identical) ----
        self.exec.rmsnorm(self.x1, self.xK.span((k - 1) * hidden, hidden), self.output_norm, c.rms_eps);
        if (self.output_weight) |ow| {
            switch (ow) {
                .quant, .f32 => self.exec.matvec(self.logits, ow, self.x1),
            }
        } else {
            switch (self.token_embd) {
                .quant => |qrow| self.exec.matvec(self.logits, .{ .quant = qrow }, self.x1),
                .f32 => |m| self.exec.matvec(self.logits, .{ .f32 = m }, self.x1),
            }
        }
        return self.logits;
    }

    /// Batched speculative verify: every row's lm_head output, `k * vocab`.
    ///
    /// Row r must equal `forwardLogits(tokens[r], pos0 + r)` -- the chunk's row
    /// r has consumed exactly that prefix.  Verifying a draft block by calling
    /// `forwardLogits` once per token makes speculation a net loss (1 + accepted
    /// full passes per round), which is why this exists.
    pub fn forwardChunkVerify(self: *Engine, tokens: []const u32, pos0: u32) Buf {
        const k = tokens.len;
        std.debug.assert(k > 0 and k <= self.nbatch and k <= spec_verify_rows);
        const c = self.cfg;
        const hidden: usize = c.hidden;

        // Layers, exactly as `forwardChunk` runs them.  Row k-1 also gets its
        // lm_head there, which the batched projection below recomputes.
        // 让 GDN 分支写每行的状态快照；`specRollbackAndReplay` 据此只改视图。
        self.in_spec_verify = true;
        defer self.in_spec_verify = false;
        _ = self.forwardChunk(tokens, pos0);
        if (k == 1) {
            self.exec.copyRange(self.logitsK.span(0, c.vocab), self.logits);
            return self.logitsK.span(0, c.vocab);
        }
        // Normalise every row and project them all through lm_head.
        self.exec.rmsnormRows(self.x1K.span(0, k * hidden), self.xK.span(0, k * hidden), self.output_norm, k, c.rms_eps);
        const w = self.output_weight orelse self.token_embd;
        self.exec.matvecBatchK(self.logitsK.span(0, k * c.vocab), w, self.x1K.span(0, k * hidden), k);
        return self.logitsK.span(0, k * c.vocab);
    }

    /// Greedy argmax over the last logits.  Forwards to the backend so the logits
    /// never have to be readable from the host (doc comment on `Exec.argmax`);
    /// the engine no longer carries its own copy of this loop.
    pub fn argmax(self: *Engine) u32 {
        return self.exec.argmax(self.logits);
    }

    // ------------- GDN block (SPEC A, GDN_IMPL §2) -------------

    fn forwardGdn(self: *Engine, lw: *LayerWeights, li: usize) void {
        const c = self.cfg;
        const hidden: usize = c.hidden;
        const nv: usize = c.gdn_n_heads; // value heads (0.8B: 16, 4B: 32)
        const nk: usize = c.gdn_num_k_heads; // k/q heads (0.8B: 16, 4B: 16)
        const kd: usize = c.gdn_head_dim; // 128
        const vd: usize = c.gdn_head_dim; // 128
        const conv_dim: usize = 2 * nk * kd + nv * kd; // [Q|K|V] = nk*kd + nk*kd + nv*kd
        const kq_dim: usize = nk * kd; // Q and K sections are nk*kd wide each
        const value_dim: usize = nv * vd; // 2048 (0.8B) / 4096 (4B)

        self.exec.rmsnorm(self.x1, self.x, lw.attn_norm, c.rms_eps);

        // in_proj_qkv [hidden, conv_dim]: [Q|K|V] contiguous (SPEC A1);
        // Q,K are nk*kd wide, V is nv*kd wide
        const qkv = self.qkv.span(0, conv_dim);
        self.exec.matvec(qkv, lw.attn_qkv.?, self.x1);

        // z projection (attn_gate [hidden, value_dim])
        const z = self.z.span(0, value_dim);
        self.exec.matvec(z, lw.attn_gate.?, self.x1);

        // a/b projections: ssm_alpha = a (decay), ssm_beta = b (beta input),
        // one scalar per VALUE head (C ref: gdn_a/gdn_b/A/dt all sized
        // gdn_vheads; GGUF rows are [hidden, nv] on 0.8B/2B/4B alike)
        //
        // ONE `matvecPair` instead of two `matvec` calls.  `ssm_alpha` and
        // `ssm_beta` are both [hidden, nv] of the same type and sit back to
        // back in the GGUF (verified for all 48 layers: `alpha.off +
        // alpha.size == beta.off`), i.e. they ARE one `2*nv x hidden` matrix,
        // and `ab` is already the contiguous output.  The backend proves the
        // adjacency before fusing and falls back to the two calls otherwise.
        //
        // Two launches here cost more than they compute: each is 48x5120 q8_0
        // = 255 KB, and 255 KB reads in 3.1-4.4 us at 60-85 GB/s for every
        // block shape tested (/tmp/shp/bw2.cu) -- that is the launch floor, not
        // bandwidth.  96 such launches per token (~0.3 ms, 1.6%) halve to 48.
        const ab = self.ab.span(0, 2 * nv);
        self.exec.matvecPair(ab, lw.ssm_alpha.?, lw.ssm_beta.?, self.x1);

        // causal conv1d with left-pad K-1, silu after (SPEC A2, C ref 520-556)
        // convStep handles history + silu for all channels; beta/gamma
        // are computed after, on the RAW a/b projections (not conv'd).
        const st = &self.gdn[li];
        self.exec.convStep(st.conv, qkv, lw.ssm_conv1d, c.conv_kernel);

        // beta = sigmoid(b); g = A * softplus(a + dt_bias) with A = ssm_a
        // (GGUF ssm_a is already -exp(A_log), i.e. negative, verified.)
        //
        // `beta` is written IN PLACE into the b half of `ab`, and `gamma` is
        // this layer's `nv` decay scalars.  Both are backend regions, and they
        // STAY there: `gdnStepRows` takes them as `Buf`s and a device backend
        // reads its own memory, so a token makes no gamma/beta PCIe trip at all
        // (it used to be 2 * nv * 4 B down and 2 * nv * 4 B back up per GDN
        // layer -- `Exec.copyToHost` per layer, which is now off this path
        // entirely).
        //
        // The two regions change contents every layer every token, so they must
        // never be handed to a pointer-keyed "upload once" cache; see
        // `VTable.gdnStepRows`.
        const beta = self.ab.span(nv, nv); // b half, overwritten with sigmoid(b)
        const gamma = self.gamma_scratch.span(0, nv);
        self.exec.gdnGamma(ab.span(0, nv), beta, lw.ssm_a, lw.ssm_dt_bias, gamma, nv);

        const state = st.state; // [nv, kd, vd] row-major S[k][v]
        const gout = self.gdn_out.span(0, value_dim);
        // L2 normalize q,k (eps inside), then scale q by 1/sqrt(128), for a
        // whole head-run at a time instead of one head at a time.  Conditions:
        //   * the q heads are `qkv[0..kq_dim]`, the k heads `qkv[kq_dim .. 2*kq_dim]`
        //     and the v heads `qkv[2*kq_dim .. 3*kq_dim]` -- contiguous runs of
        //     `kd`-wide rows;
        //   * every row gets the same eps / the same `qscale`;
        //   * each row's normalisation is independent (own sum of squares).
        // `l2normRows` is `n` consecutive single-row `k_l2norm` blocks and
        // `scaleVec` is elementwise, so both are the same arithmetic per row.
        //
        // ACTIVE ON THIS MODEL: nv (48) > nk (16), so head `h` maps to k head
        // `h % nk` and each q/k row is visited `nv / nk` = 3 times, in the order
        // 0..nk-1 repeated.  `l2norm` is NOT idempotent (a second pass divides by
        // sqrt(1 + eps) again), so the repetitions are part of the semantics and
        // must be kept: one `l2normRows` over the `nk` rows, `nk_reps` times,
        // reproduces exactly the sequence of per-row calls the loop made.
        //
        // NO `nv/nk` REPETITION.  The per-head loop this replaced visited row
        // `h % nk` once per v head, i.e. `nv/nk` = 3 times per k head, and
        // `l2normRows(span(0, kq_dim), kd)` normalises each of the `nk` rows
        // ONCE -- so reproducing that loop needed 3 repeats.  That is the wrong
        // operation, not just a slow way to express it: the reference
        // (llama.cpp `llama_model_qwen3next::graph::build_layer_gdn`,
        // src/models/qwen3next.cpp:503-511) applies `build_gdn_l2_norm` to a
        // `(head_k_dim, num_k_heads)` view -- **16** rows -- and only THEN
        // broadcasts q/k up to `num_v_heads` with `ggml_repeat_4d`
        // (qwen3next.cpp:520-528).  Normalising 16 rows 3 times is not the same
        // as normalising them once and repeating: `l2norm` is not idempotent,
        // so passes 2 and 3 multiply the row by `rsqrt(1 + 1e-6)` ≈ 1-5e-7
        // again.  The error was too small to flip a greedy token, which is why
        // no test caught it, but it is both WRONG and 3x the work:
        // ncu measured `k_l2norm` at 15.68 ms / 7452 calls over 20 tokens
        // (2.1 us each for a 128-float reduction), and 2/3 of those calls were
        // this repetition.
        if (gdnHeadsTiled(nv, nk)) {
            const qscale = 1.0 / @sqrt(@as(f32, @floatFromInt(kd)));
            // One launch instead of three (l2normRows q, l2normRows k,
            // scaleVec q).  Same arithmetic in the same order -- the CPU
            // backend's `l2normQkScale` literally calls those three, and the
            // device kernel reproduces them per row.  See `backend.Exec`.
            self.exec.l2normQkScale(qkv, nk, kd, kq_dim, qscale);
        }
        // Fallback for a layout the batched passes above/below cannot express
        // (`nv` not a multiple of `nk`): the per-head form this function has
        // always used.  On this model nv/nk == 3, so this is dead code kept for
        // shapes the kernel does not model.
        if (!gdnHeadsTiled(nv, nk)) {
            for (0..nv) |h| {
                const kh = h % nk;
                const q_h = qkv.span(kh * kd, kd);
                const k_h = qkv.span(kq_dim + kh * kd, kd);
                self.exec.l2norm(q_h);
                self.exec.l2norm(k_h);
                const qscale = 1.0 / @sqrt(@as(f32, @floatFromInt(kd)));
                self.exec.scaleVec(q_h, qscale);
            }
        }

        // (1)+(2) decay & kv_mem in one pass, (3)+(4) outer update & S^T q in a
        // second pass (see gdnFused) -- for all `nv` heads in ONE backend call.
        //
        // This was a per-head `exec.gdnStep` loop: 48 launches per token per GDN
        // layer, which the PROFILE put at ~15% of slot time.  `gdnStepRows` runs
        // the identical per-head computation with block h on the device (the
        // CPU backend keeps the loop, as the oracle), so the tokens do not
        // move.
        //
        // The kv-head mapping `kh = h % nk` (GGUF v slots are in "tiled" order,
        // converter `_reorder_v_heads`: slot h <-> HF v head (h%nk)*nvp + h/nk)
        // now lives in the slot -- see `VTable.gdnStepRows`.  `km`/`delta` are
        // per-head slices of `nv*vd` scratch: with one shared `vd` buffer the
        // 48 blocks would race silently.
        self.exec.gdnStepRows(
            state,
            qkv.span(kq_dim, kq_dim),
            qkv.span(2 * kq_dim, value_dim),
            qkv.span(0, kq_dim),
            self.kv_mem,
            self.delta,
            gout,
            gamma,
            beta,
            nv,
            nk,
            kd,
            vd,
        );

        // gated RMSNorm: norm -> weight -> silu(gate) (SPEC A6).
        //
        // The `nv` norm rows are consecutive (`gout` is `nv * vd` wide) with one
        // shared `ssm_norm` row, so this is the three-condition batch.
        self.exec.rmsnormRows(gout.span(0, value_dim), gout.span(0, value_dim), lw.ssm_norm, nv, c.rms_eps);
        // `mulSilu` is elementwise with no reduction and no per-row state, so the
        // `nv` heads collapse into this one call over the same `nv * vd` run --
        // unlike the norm merges above, there is no repeat-count subtlety here.
        self.exec.mulSilu(gout.span(0, value_dim), z.span(0, value_dim));

        // out projection [2048 -> 1024]
        self.exec.matvec(self.x2.span(0, hidden), lw.ssm_out.?, gout);
        self.exec.addResid(self.x.span(0, hidden), self.x2.span(0, hidden));
    }

    /// Whether the GDN per-head loop visits the q/k runs as `nk` contiguous
    /// rows repeated `nv / nk` times, in that order -- the shape the batched
    /// normalisation reproduces.
    ///
    /// The per-head loop maps v head `h` to k head `h % nk` and walks
    /// `h = 0 .. nv-1`, so the rows it touches are `0,1,..,nk-1` repeated
    /// `nv / nk` times.  That is expressible as `nv / nk` batched passes over
    /// the `nk`-row run -- exactly what `forwardGdn` / `forwardGdnCore` do --
    /// whenever `nv` is a whole multiple of `nk`.  Anything else (a partial
    /// final group) falls back to the per-head calls.
    inline fn gdnHeadsTiled(nv: usize, nk: usize) bool {
        return nk != 0 and nv % nk == 0;
    }

    /// Causal conv step, HOST-slice form: the unit test of this file drives it
    /// directly (the forward path calls `exec.convStep` with backend regions
    /// instead, and never reaches this function).
    ///
    /// out[i] = sum_t w[i][t] * hist[i][t], where hist = [3 previous raw
    /// samples (oldest..newest), current raw sample].
    fn convStep(self: *Engine, conv: []f32, qkv: []f32, wconv: []const f32, K: usize) void {
        _ = self;
        std.debug.assert(K == 4);
        const h: [*]f32 = conv.ptr;
        const w: [*]const f32 = wconv.ptr;
        const out: [*]f32 = qkv.ptr;
        for (0..qkv.len) |i| {
            const hp = h + i * 3;
            const cur = out[i];
            // 4 taps per channel in one @Vector multiply; the sum keeps the
            // exact scalar order so results are bit-identical.
            const hv: @Vector(4, f32) = .{ hp[0], hp[1], hp[2], cur };
            const wv: @Vector(4, f32) = .{ w[i * 4], w[i * 4 + 1], w[i * 4 + 2], w[i * 4 + 3] };
            const prod: [4]f32 = @bitCast(hv * wv);
            out[i] = prod[0] + prod[1] + prod[2] + prod[3];
            // shift: drop oldest, append current raw
            hp[0] = hp[1];
            hp[1] = hp[2];
            hp[2] = cur;
        }
        siluVec(qkv);
    }

    // ------------- Full attention block (SPEC B) -------------

    // `forwardFullCore` (the per-token body) was removed when `forwardFullChunk`
    // replaced it: the per-row form now lives in the CPU backend as
    // `vAttnChunk`, which IS the old loop and is the oracle the chunk kernel is
    // checked against (`/tmp/gate_attn_chunk.cu`).  Keeping a second copy here
    // would mean two per-row implementations to keep in sync for no caller.

    /// Chunk-wide twin of `forwardFullCore`: the whole chunk's full-attention
    /// layer in ONE `attnChunk` call instead of `k` `attnDecode` calls.
    ///
    /// WHAT MOVED.  `forwardFullCore`'s per-token body had four parts:
    ///   1. `deinterleaveQG`   [q|gate] split for token t
    ///   2. `rmsnormRows` x2   q_norm over `heads` rows, k_norm over `kv_heads`
    ///   3. `copyRangeSlot` x2 KV cache write for token t
    ///   4. `attnDecode`       scores -> softmax -> PV -> gate
    /// Only (4) touches the growing window, so only (4) is what `attnChunk`
    /// replaces.  (1)-(3) were already cheap single launches; they stay per
    /// token here because each one writes the row that token r's attention
    /// needs, and the cache rows must ALL be present before `attnChunk` runs.
    /// Folding (1)-(3) into the chunk kernel is a separate change.
    ///
    /// CORRECTNESS: `attnChunk(slot0, k)` is defined as the k per-row
    /// `attnDecode(slot0+r)` calls, and the CUDA kernel was gated bit-identical
    /// against exactly that (`/tmp/gate_attn_chunk.cu`, 23 rows / 4 heads /
    /// slot0=37: worst abs diff 0).  The cache is filled in ascending order
    /// first, so row r sees the same `[0, slot0+r]` prefix it saw before.
    fn forwardFullChunk(self: *Engine, lw: *LayerWeights, li: usize, k: usize) void {
        const c = self.cfg;
        const heads: usize = c.n_heads;
        const kv_heads: usize = c.n_kv_heads;
        const hd: usize = c.head_dim;
        const q_dim: usize = heads * hd;
        const kv_dim: usize = kv_heads * hd;
        const gqa: usize = heads / kv_heads;

        const cache = &self.kv[li];
        std.debug.assert(cache.len + k <= max_seq);
        const slot0: usize = cache.len;

        // [q|gate] split for the whole chunk in ONE launch (was `k` launches of
        // 24 heads each: 940 per 525-token prefill, 1.2% of device time).
        self.exec.deinterleaveQGStrided(self.qK.span(0, k * q_dim), self.gateK.span(0, k * q_dim),
            self.qkvK.span(0, k * 2 * q_dim), heads, hd, k);

        // q's RoPE, chunk-wide.  It CANNOT move up to `forwardChunk` with k's:
        // `qK` does not exist until the deinterleave above has run.  It also
        // has to run AFTER the q_norm/k_norm pair below, because the reference
        // order (`forwardFull`) is norm -> rope; rotating first and
        // normalising after renormalises the rotation away.
        //
        // `slot0` is the chunk's absolute start, so row r rotates at
        // `slot0 + r`, which equals the `pos0 + r` the caller used for k.
        //
        // Pass 1: everything row r needs written, for EVERY row, in ascending
        // order.  This is the old per-token body minus the attention itself.
        //
        // Both ops here are independent across rows, so they run CHUNK-WIDE (one
        // launch each) instead of once per token:
        //   * q_norm/k_norm: "rows" is already a batch dimension of
        //     `rmsnormRows` -- row r of qK is one head of token r, and every row
        //     uses the same weight row, so `k * heads` rows in one call is the
        //     identical per-row arithmetic (same eps, same single-pass sum).
        //   * KV write: `cache.k` rows [slot0, slot0+k) are exactly the chunk's
        //     `kK[0 .. k*kv_dim)` f32 rows -- so this is ONE `kvAppend` of `k`
        //     rows, not k per-token appends (the backend strides its own row
        //     bytes; f32 KV makes it the old single f32 copy).
        //
        // Doing these per token cost 525 * 2 rmsnorm + 525 * 2 KV launches per
        // full-attention layer (16 layers => 33,600 launches for a 525-token
        // prefill).  Measured: prefill had ~36k launches and was runnable at
        // 50% GPU occupancy, i.e. half the wall clock was the host submitting
        // launches.  RoPE already ran chunk-wide in `forwardChunk`.
        //
        // `kvAppend`'s explicit `slot`/`rows` serve the same purpose: the
        // DECODE graph retargets a captured KV append row by row
        // (`g_op_seen.kv_append`), and prefill does not go through the graph.
        self.exec.rmsnormRows(self.qK.span(0, k * q_dim), self.qK.span(0, k * q_dim),
            lw.attn_q_norm, k * heads, c.rms_eps);
        self.exec.rmsnormRows(self.kK.span(0, k * kv_dim), self.kK.span(0, k * kv_dim),
            lw.attn_k_norm, k * kv_heads, c.rms_eps);
        self.exec.rope64Rows(self.qK.span(0, k * q_dim), @intCast(slot0), self.cfg.rope_freq_base, k, heads, hd);
        self.exec.rope64Rows(self.kK.span(0, k * kv_dim), @intCast(slot0), self.cfg.rope_freq_base, k, kv_heads, hd);
        self.exec.kvAppend(cache.k, self.kK.span(0, k * kv_dim), slot0, kv_dim, k);
        self.exec.kvAppend(cache.v, self.vK.span(0, k * kv_dim), slot0, kv_dim, k);
        cache.len += @intCast(k);

        // Pass 2: scores -> softmax -> PV -> gate for all k rows, one launch.
        const scale = 1.0 / @sqrt(@as(f32, @floatFromInt(hd)));
        self.exec.attnChunk(self.qK.span(0, k * q_dim), self.gateK.span(0, k * q_dim),
            cache.k, cache.v,
            self.attn.span(0, slot0 + k), self.goutK.span(0, k * q_dim),
            slot0, k, heads, gqa, hd, kv_dim, scale);
    }

    /// Batched-chunk twin of `forwardGdn`: qkv / gate / alpha / beta were
    /// batched into qkvK / gateK / abK; this runs the per-token conv + GDN
    /// state kernel for slot `t` (updating the shared conv history and 128x128
    /// Chunk-wide GDN prelude -- the per-token work in `forwardGdnCore` that does
    /// not depend on the running state, hoisted out of the token loop.
    ///
    /// What it replaces, per token, per GDN layer: one `convStep` (2 launches),
    /// one `gdnGamma`, and `nk_reps` x (2 `l2normRows` + 1 `scaleVec`).  On the
    /// 27B model that is 14 launches per token per layer; for a 525-token
    /// prefill that is ~59k launches, all of them 1 KB of work.
    ///
    /// CORRECTNESS: this is a pure code motion, not a new algorithm.  Every
    /// call below is the identical expression `forwardGdnCore` used to make for
    /// token `t`, in the same order, on the same regions -- `gdnStepRows` only
    /// reads qkv (it writes state/km/delta/out), so nothing the prelude writes is
    /// consumed before the state loop starts.  The conv is the one real
    /// cross-token dependency (token t's taps read the history token t-1 left
    /// behind) and it stays ascending, inside the backend call.
    fn forwardGdnPrelude(self: *Engine, lw: *LayerWeights, li: usize, k: usize) void {
        const c = self.cfg;
        const nv: usize = c.gdn_n_heads;
        const nk: usize = c.gdn_num_k_heads;
        const kd: usize = c.gdn_head_dim;
        const conv_dim: usize = 2 * nk * kd + nv * kd;
        const kq_dim: usize = nk * kd;

        const st = &self.gdn[li];
        // The scratch is `nv` wide and every token writes its own row, so it is
        // sized `k * nv` by `forwardChunk`'s caller.  `st.conv` stays single-token
        // (the history is shared across the chunk, by design).
        const gamma = self.gdn_prefill_gamma.span(0, k * nv);
        self.exec.gdnPrefillCore(
            self.qkvK.span(0, k * conv_dim),
            st.conv,
            lw.ssm_conv1d,
            self.abK.span(0, 2 * k * nv),
            lw.ssm_a,
            lw.ssm_dt_bias,
            gamma,
            k,
            conv_dim,
            0, // ab_a_offset: alpha half starts at 0
            k * nv, // ab_b_offset: beta half starts at k*nv
            nv, // ab_stride: token t at t*nv inside each half
            nv, // gamma_stride
            kq_dim,
            nv,
            nk,
            kd,
            c.conv_kernel,
            // `elementwise.l2norm` pins eps at 1e-6 (inside the sum, SPEC A3);
            // the CUDA l2norm slots pass the same constant.  There is no config
            // knob for it, so this is not a choice: passing anything else would
            // diverge from the CPU oracle.
            1e-6,
        );
    }

    /// Chunk-wide state update: the SAME per-token recurrence as
    /// `forwardGdnCore` did, but for all `k` tokens in ONE backend call.
    ///
    /// WHY: the steps are sequential (`state` is the carried recurrence), so
    /// fusing them adds no parallelism -- what it removes is the LAUNCH.  The
    /// per-token loop issued one launch per token per GDN layer: 10,500 for a
    /// 525-token prefill at 48 layers, with the kernel itself doing ~4 us of
    /// work each.  PROFILE put the device time of those launches at 42 ms while
    /// the wall clock they occupied was 641 ms of a 1752 ms prefill -- the host
    /// could not submit them fast enough.  Deleting the loop outright
    /// (measurement only, wrong results) took prefill 299 -> 453 tok/s, which
    /// prices the whole overhead at ~594 ms.
    ///
    /// CORRECTNESS: this is pure code motion, not an algorithm change.  The
    /// device kernel is `q35_gdn_step_rows_kernel`'s body verbatim with the
    /// token index as a loop variable (same __fmul_rn/__fadd_rn pinning, same
    /// i-order, same two-pass structure), and the CPU oracle for the new slot
    /// is literally the old `rows` successive calls.  Every operand that used to
    /// be sliced per token is now sliced per row inside the call.
    ///
    /// Layout note: `qkvK` is `k * conv_dim` row-major, so q/k/v for the whole
    /// chunk are three strided views of it (stride `conv_dim`), NOT three
    /// contiguous blocks -- which is why the chunk slot takes explicit strides
    /// instead of inferring them from the buffer length.
    fn forwardGdnCoreChunk(self: *Engine, li: usize, k: usize) void {
        const c = self.cfg;
        const nv: usize = c.gdn_n_heads;
        const nk: usize = c.gdn_num_k_heads;
        const kd: usize = c.gdn_head_dim;
        const vd: usize = c.gdn_head_dim;
        const conv_dim: usize = 2 * nk * kd + nv * kd;
        const kq_dim: usize = nk * kd;
        const value_dim: usize = nv * vd;

        if (k == 0) return;
        // conv + silu, beta = sigmoid(b), gamma, and the l2norm/scale passes were
        // all done for the whole chunk by `forwardGdnPrelude`; qkv and abK are
        // already in their final form here.  Only the state update is per token,
        // and it is now per token inside the kernel rather than per launch.
        //
        // Every argument is the CHUNK-RESIDENT buffer, not token 0's slice: the
        // backend strides it.  That is what makes this one launch instead of `k`.
        const beta_base: usize = k * nv; // the b half of abK follows the a half
        // 推测解码的 verify 走带快照的版本：内核在每行结束后把整份 state 写进
        // `state_all` 的第 t+1 块。回滚因此只需把 `state` 视图指过去。
        // 其它情形（prefill、普通 decode）传空 snap，与改动前完全一致。
        const want_snap = self.in_spec_verify and self.gdn[li].state_all.len >= (spec_rs_seq + 1) * nv * kd * vd;
        if (want_snap) {
            self.exec.gdnStepRowsSnap(
                self.gdn[li].state,
                self.qkvK.span(kq_dim, k * conv_dim - kq_dim),
                self.qkvK.span(2 * kq_dim, k * conv_dim - 2 * kq_dim),
                self.qkvK.span(0, k * conv_dim),
                self.kv_mem,
                self.delta,
                self.goutK.span(0, k * value_dim),
                self.gdn_prefill_gamma.span(0, k * nv),
                self.abK.span(beta_base, k * nv),
                self.gdn[li].state_all.span(nv * kd * vd, (spec_rs_seq) * nv * kd * vd),
                nv,
                nk,
                kd,
                vd,
                k,
                conv_dim,
                conv_dim,
                conv_dim,
            );
            return;
        }
        self.exec.gdnStepRowsChunk(
            self.gdn[li].state,
            self.qkvK.span(kq_dim, k * conv_dim - kq_dim),
            self.qkvK.span(2 * kq_dim, k * conv_dim - 2 * kq_dim),
            self.qkvK.span(0, k * conv_dim),
            self.kv_mem,
            self.delta,
            self.goutK.span(0, k * value_dim),
            self.gdn_prefill_gamma.span(0, k * nv),
            self.abK.span(beta_base, k * nv),
            nv,
            nk,
            kd,
            vd,
            k,
            conv_dim, // k per-token stride (qkv rows interleave q, k, gate)
            conv_dim, // v per-token stride
            conv_dim, // q per-token stride
        );
    }

    /// Chunk-wide tail of the GDN block: the gated RMSNorm (SPEC A6, norm ->
    /// weight) followed by `silu(gate)` for every token in ONE call each.
    ///
    /// CORRECTNESS: this is pure code motion out of `forwardGdnCore`'s token
    /// loop.  Both kernels are elementwise with no per-row state, and every row
    /// they touch (`goutK[t*value_dim ..][value_dim]`, `gateK[t*value_dim ..]`)
    /// was written by token t's own `gdnStepRows` and is read by nothing else in
    /// between -- the only cross-token dependency in the GDN block is the conv
    /// history / 128x128 state, and that stays in the serial `gdnStepRows` loop.
    /// `rmsnormRows` reduces each row independently in the same order, and
    /// `mulSilu` is elementwise over arbitrary `n`, so both produce bit-identical
    /// bytes to the per-token calls they replace.
    ///
    /// This MUST run after the last `forwardGdnCore` of the chunk and before
    /// `ssm_out` reads `goutK`.
    fn forwardGdnGated(self: *Engine, lw: *LayerWeights, k: usize) void {
        const c = self.cfg;
        const nv: usize = c.gdn_n_heads;
        const value_dim: usize = nv * c.gdn_head_dim;
        // The `nv` norm rows of a token are consecutive (`gout` is `nv * vd`
        // wide) with one shared `ssm_norm` row, so `k * nv` rows is the same
        // three-condition batch case, `k` tokens deep.
        self.exec.rmsnormRows(self.goutK.span(0, k * value_dim), self.goutK.span(0, k * value_dim), lw.ssm_norm, k * nv, c.rms_eps);
        self.exec.mulSilu(self.goutK.span(0, k * value_dim), self.gateK.span(0, k * value_dim));
    }

    fn forwardFull(self: *Engine, lw: *LayerWeights, li: usize, pos: u32) void {
        const c = self.cfg;
        const hidden: usize = c.hidden;
        const heads: usize = c.n_heads;
        const kv_heads: usize = c.n_kv_heads;
        const hd: usize = c.head_dim; // 256
        const q_dim: usize = heads * hd; // 2048
        const kv_dim: usize = kv_heads * hd; // 512
        const gqa: usize = heads / kv_heads; // 4

        self.exec.rmsnorm(self.x1, self.x, lw.attn_norm, c.rms_eps);

        // q_proj [1024, 4096]: view(8, 512) then chunk -> per-head [q|gate]
        const qg = self.qkv.span(0, 2 * q_dim);
        self.exec.matvec(qg, lw.attn_q.?, self.x1);
        const q = self.q.span(0, q_dim);
        const gate = self.z.span(0, q_dim);
        // Batched split (see `forwardFullCore`): the GGUF `attn_q` rows are
        // `[h][q_head | gate_head]`, and `q`/`gate` are both consumed per head,
        // so this is a layout change no single memcpy can express -- the slot
        // just does it in one launch instead of `2*heads`.
        self.exec.deinterleaveQG(q, gate, qg, heads, hd);
        self.exec.matvec(self.k.span(0, kv_dim), lw.attn_k.?, self.x1);
        self.exec.matvec(self.v.span(0, kv_dim), lw.attn_v.?, self.x1);

        // q_norm/k_norm BEFORE RoPE (B12); GGUF weights pre-folded (1+w)
        //
        // ONE `rmsnormRows` for all heads, not one `rmsnorm` per head.  Head h
        // of q occupies [h*hd, (h+1)*hd), so the heads are contiguous and the
        // row-batched slot is exactly the loop it replaces -- on the CPU oracle
        // it is *literally* that loop (`vRmsnormRows`), so this cannot change a
        // single value.  On CUDA `rmsnormRows` is capture-safe (no graph gate),
        // so the decode graph sees one node here instead of `heads` of them.
        //
        // RoPE stays per head on purpose: `dRope64_body` reads {pos, base} from
        // a device buffer so its nodes need no retargeting, but `rope64_rows`
        // launches a different kernel with no host-stub identity for the graph
        // to match.  See `dRope64Rows_body`'s capture guard.
        self.exec.rmsnormRows(q.span(0, heads * hd), q.span(0, heads * hd), lw.attn_q_norm, heads, c.rms_eps);
        self.exec.rmsnormRows(self.k.span(0, kv_heads * hd), self.k.span(0, kv_heads * hd), lw.attn_k_norm, kv_heads, c.rms_eps);
        // RoPE on first 64 dims of each q/k head -- ONE call per tensor, not
        // one per head.  `rope64Rows(x, 1, heads, hd)` is literally the
        // per-head loop it replaces (`k_rope64_rows` rotates row r at `pos+r`,
        // and here rows == 1), and the CUDA backend now has a
        // device-position twin of that kernel (`k_rope64_rows_dpos`) whose
        // host-stub address the decode graph recognises, so batching survives
        // the capture.
        //
        // MEASURED (27B IQ3_XXS decode, ncu): one launch per head meant
        // 40 + 8 = 48 rope launches per full-attn layer, 768 per token, and
        // `k_rope64_dpos` was 12.84 ms / 6188 calls over 20 tokens -- 2.07 us
        // per launch for 64 cos+sin, and 27.9% of all non-matvec kernel time.
        self.exec.rope64Rows(q.span(0, heads * hd), pos, c.rope_freq_base, 1, heads, hd);
        self.exec.rope64Rows(self.k.span(0, kv_heads * hd), pos, c.rope_freq_base, 1, kv_heads, hd);

        // KV cache: quantize this token's rows into the cache (the backend
        // strides its own layout; `slot` is the per-token row the decode
        // graph retargets, exactly as it did for the old copy append).
        const cache = &self.kv[li];
        std.debug.assert(cache.len < max_seq);
        const slot: usize = cache.len;
        self.exec.kvAppend(cache.k, self.k.span(0, kv_dim), slot, kv_dim, 1);
        self.exec.kvAppend(cache.v, self.v.span(0, kv_dim), slot, kv_dim, 1);
        cache.len += 1;

        // attention: kv = h // gqa (B13), scale = hd^-0.5
        const scale = 1.0 / @sqrt(@as(f32, @floatFromInt(hd)));
        const gout = self.gdn_out.span(0, q_dim);
        // scores -> softmax -> PV -> sigmoid gate in one backend call
        self.exec.attnDecode(q.span(0, q_dim), gate.span(0, q_dim),
            cache.k, cache.v,
            self.attn.span(0, slot + 1), gout.span(0, q_dim),
            slot, heads, gqa, hd, kv_dim, scale);
        // o_proj [2048 -> 1024]
        self.exec.matvec(self.x2.span(0, hidden), lw.attn_output.?, gout);
        self.exec.addResid(self.x.span(0, hidden), self.x2.span(0, hidden));
    }

    // ------------- nextn / MTP drafter (SPECULATIVE DECODING, stage 1) ------

    /// Run the nextn block once and leave ONE token's logits in `self.logits`.
    ///
    /// Inputs:
    ///   * `hidden_buf` -- the target's `output_norm`-ed hidden state for the
    ///     token at position p.  NOT the raw last-layer residual stream:
    ///     `qwen35.cpp:210` builds `t_h_nextn` as
    ///     `build_norm(inpL, output_norm, ...)`, and that normed value is what
    ///     the drafter's `hnorm` receives.  Passing the raw residual instead
    ///     keeps every intermediate finite but drops the hit rate to a few
    ///     percent (MEASURED: 12.2 / 6.1 / 14.3 / 0.0 % over four prompts);
    ///     with the norm applied it is ~25 %.
    ///   * `next_token` -- the token at position p+1, i.e. the one the target
    ///     has ALREADY committed after the hidden state above.  The drafter
    ///     therefore predicts the token after it (position p+2), which is what
    ///     `common/speculative.cpp`'s draft driver does: it pairs each batch
    ///     token `t.id` with `h_tgt + (k-1)*n_embd`, the target embedding
    ///     shifted right by one position.
    ///   * `draft_pos` -- KV position for the drafter's OWN cache.
    ///
    /// The block is a full-attention layer, so this reuses `forwardFull` /
    /// `forwardMlp` unchanged: they read `self.x1` (normed input) and write
    /// `self.x` (residual stream), which is exactly the contract the decoder
    /// loop uses.
    ///
    /// ORDER (from `qwen.py`'s remapper + `deepseek.py`'s torch.cat): the
    /// embedding half comes FIRST, then the hidden half.  Getting this backwards
    /// produces a plausible-looking but wrong draft, so it is asserted by
    /// construction here (`input[0..hidden]` = enorm(e)).
    ///
    /// Returns error.NoNextn when the model has no drafter.
    ///
    /// CHAINING (the reason `out_hidden` exists): a single MTP head drafts
    /// several tokens by feeding its OWN output back in, one step at a time --
    /// `common/speculative.cpp`'s non-`chain_heads` path re-runs this graph per
    /// draft token with `h_row` taken from the drafter's previous step.  Pass
    /// the same buffer as `hidden_buf` and `out_hidden` to chain in place
    /// (`null` skips the copy).  `self.x1` holds exactly the value to carry:
    /// `shared_head_norm` of this step's output, which is the drafter's
    /// counterpart of the trunk's `t_h_nextn`.
    pub fn nextnPredict(
        self: *Engine,
        hidden_buf: backend.Buf,
        next_token: u32,
        draft_pos: u32,
        out_hidden: ?backend.Buf,
    ) !void {
        const nn = self.nextn orelse return error.NoNextn;
        const c = self.cfg;
        const hidden: usize = c.hidden;
        const li = nn.li;
        const lw = &self.layers[li];

        // h = eh_proj([enorm(e); hnorm(h)])  -- one matvec over a 2*hidden input.
        //
        // `enorm`/`hnorm` are RMSNorms, and they are applied to the two halves
        // INDEPENDENTLY (each over its own `hidden` values), not jointly over
        // the concatenation: GGUF ships them as two separate [hidden] weights,
        // and `pre_fc_norm_embedding` / `pre_fc_norm_hidden` are two distinct
        // modules in the reference.
        self.embedTo(next_token, self.x1.span(0, hidden));
        self.exec.rmsnorm(self.x1.span(0, hidden), self.x1.span(0, hidden), nn.enorm, c.rms_eps);
        self.exec.copyRange(nn.input.span(0, hidden), self.x1.span(0, hidden));
        self.exec.rmsnorm(self.x1.span(0, hidden), hidden_buf.span(0, hidden), nn.hnorm, c.rms_eps);
        self.exec.copyRange(nn.input.span(hidden, hidden), self.x1.span(0, hidden));

        // The projected value becomes the residual stream for the block.
        self.exec.matvec(self.x.span(0, hidden), nn.eh_proj, nn.input.span(0, 2 * hidden));

        // Full-attention layer + MLP, over the drafter's OWN KV cache.
        //
        // `forwardFull` derives the cache index from `li`, which is
        // `cfg.n_layers` -- a slot the decoder loop never touches, so the
        // target's KV is untouched by construction.
        self.forwardFull(lw, li, draft_pos);
        self.forwardMlp(lw);

        // Shared head: `shared_head_norm` then the TARGET's `output_norm`?  No --
        // the reference applies the MTP block's own norm and then reuses the
        // model's lm_head on that normed value.  `output_norm` belongs to the
        // main stack and is already applied once for the target logits, so the
        // drafter's equivalent is `shared_head_norm` alone.
        self.exec.rmsnorm(self.x1.span(0, hidden), self.x.span(0, hidden), nn.shared_head_norm, c.rms_eps);
        // Hand the normed output back for the next chained step BEFORE it gets
        // overwritten by anything else in this function (nothing does, but the
        // copy makes the contract explicit and costs one `hidden` memcpy).
        if (out_hidden) |oh| self.exec.copyRange(oh.span(0, hidden), self.x1.span(0, hidden));
        if (self.output_weight) |ow| {
            switch (ow) {
                .quant, .f32 => self.exec.matvec(self.logits, ow, self.x1.span(0, hidden)),
            }
        } else {
            switch (self.token_embd) {
                .quant => |qrow| self.exec.matvec(self.logits, .{ .quant = qrow }, self.x1.span(0, hidden)),
                .f32 => |m| self.exec.matvec(self.logits, .{ .f32 = m }, self.x1.span(0, hidden)),
            }
        }
    }

    // ------------- MLP (SPEC C16) -------------

    fn forwardMlp(self: *Engine, lw: *LayerWeights) void {
        const c = self.cfg;
        const hidden: usize = c.hidden;
        self.exec.rmsnorm(self.x1, self.x, lw.post_attention_norm, c.rms_eps);
        self.exec.matvec(self.mlp_h, lw.ffn_gate, self.x1);
        self.exec.matvec(self.mlp_h1, lw.ffn_up, self.x1);
        self.exec.swiglu(self.mlp_h, self.mlp_h1);
        self.exec.matvec(self.x2.span(0, hidden), lw.ffn_down, self.mlp_h);
        self.exec.addResid(self.x.span(0, hidden), self.x2.span(0, hidden));
    }

    // -----------------------------------------------------------------
    // Generation
    // -----------------------------------------------------------------

    pub fn reset(self: *Engine) void {
        for (self.kv) |*cache| cache.len = 0;
        for (self.gdn) |*st| {
            if (st.conv.len > 0) self.exec.zeroBuf(st.conv);
            if (st.state.len > 0) self.exec.zeroBuf(st.state);
            st.conv_len = 0;
        }
    }

    // -----------------------------------------------------------------
    // Speculative-decoding rollback (stage 2)
    // -----------------------------------------------------------------

    /// 在 `load` 里分配推测解码回滚用的每层投影缓存（`gdn_in`）。
    ///
    /// `spec = false`、模型没有 nextn 块（`e.nextn == null`，此时
    /// `generateSpec` 会直接退回普通解码）、或 `nbatch == 0`（批量路径不可用）
    /// 时什么都不做，`gdn_in` 保持空 —— 这些运行完全不付这块显存。
    /// 27B 上它是 48 层 × (16×10240 + 16×96) floats ≈ 6 MB。
    ///
    /// 调用点必须在 `e.nextn` 赋值之后（`load` 的逐层分配之后）。
    fn allocGdnIn(self: *Engine, alloc: std.mem.Allocator, spec: bool) !void {
        if (!spec or self.nextn == null or self.nbatch == 0 or self.gdn.len == 0) return;
        const c = self.cfg;
        const nv: usize = c.gdn_n_heads;
        const nk: usize = c.gdn_num_k_heads;
        const kd: usize = c.gdn_head_dim;
        const conv_dim: usize = 2 * nk * kd + nv * kd;
        // `@min` 的结果类型会窄化到整数常量的位宽（16 -> `u5`），
        // 于是 `rows * 2 * nv` 在 ReleaseFast 下静默回绕成 0
        // （实测：`gamma` 拿到长度 0，缓存从未写入）。乘积必须留在 `usize` 里。
        const rows: usize = @min(spec_verify_rows, self.nbatch);
        const ins = try alloc.alloc(GdnIn, self.gdn.len);
        // 中途失败时只回收 host 数组：`allocBuf` 的区域归 `exec.freeScratch()`，
        // 不能在这里单独 free（会与它二次释放）。
        errdefer alloc.free(ins);
        @memset(ins, .{});
        for (ins) |*gi| {
            gi.qkv = try self.exec.allocBuf(rows * conv_dim);
            gi.gamma = try self.exec.allocBuf(rows * 2 * nv);
        }
        self.gdn_in = ins;
    }

    /// 在 `load` 里分配一轮推测解码回滚所需的快照。
    ///
    /// 快照保存的是"verify 一个块之前"每个携带状态的地方，回滚时据此恢复
    /// （`spec_rollback.zig` 的文件头解释了为什么没有增量逆运算可选）：
    ///   * full-attention 层：只存 `KvCache.len`。ROWS 不拷贝——下一次 verify
    ///     会覆盖它们，而且越过 `len` 的行从不被读，所以回退计数就是全部撤销。
    ///   * GDN 层：`conv`（每通道最后 K-1 个原始样本）与 `state`
    ///     （[nv, kd, vd] 的 delta 规则矩阵）整块拷贝。两者都按非线性方式被消费
    ///     （卷积会移位，delta 规则除以一个运行中的范数），所以便宜地求逆。
    ///
    /// 27B（48 个 GDN 层）约 31.4 MB，全程一次。
    ///
    /// `spec = false` / 无 nextn 块 / `nbatch == 0` 时不分配，`spec_snap` 保持空。
    /// 必须在 `e.gdn` 逐层分配之后调用（尺寸取自它）。
    fn allocSpecSnapshot(self: *Engine, alloc: std.mem.Allocator, spec: bool) !void {
        if (!spec or self.nextn == null or self.gdn.len == 0) return;
        if (spec_dummy_alloc) {
            // `$QWEN35_SPEC_DUMMYALLOC`：在推测解码的设置阶段预留一块一次性设备区域。
            // 用来检查 MTP 路径的输出**不**依赖设备分配器的布局：开着它 token 流
            // 仍逐位一致（实测 3/3），所以两个构建间行为变化是真的逻辑差异，
            // 不是地址运气。
            //
            // 不要在这里碰 `Buf.slice()`：CUDA 上 `Buf` 是**设备**地址，从主机侧
            // 写它是非法访问，进程会静默死掉（实测）。
            _ = try self.exec.allocBuf(48 * 16 * 10240);
        }
        const n = self.nSlots();
        const snaps = try alloc.alloc(SnapLayer, n);
        errdefer alloc.free(snaps);
        for (snaps, 0..) |*s, i| {
            s.* = .{ .conv = backend.Buf.empty, .state = backend.Buf.empty, .conv_len = 0, .kv_len = 0 };
            if (i < self.gdn.len) {
                const g = self.gdn[i];
                // 后端区域，与活缓冲的 residency 一致：CUDA 上是 VRAM，
                // 所以每轮的拷贝是 D2D。
                if (g.conv.len > 0) s.conv = try self.exec.allocBuf(g.conv.len);
                if (g.state.len > 0) s.state = try self.exec.allocBuf(g.state.len);
            }
        }
        self.spec_snap = snaps;

        // `$QWEN35_SPEC_RBDBG` 的比对缓冲：一份第一层 GDN 的 `state`，HOST 侧
        // （`copyOut` 的目标），一次分配、每轮复用。
        if (cudart_debug_rollback and self.gdn.len > 0 and self.gdn[0].state.len > 0) {
            const len = self.gdn[0].state.len;
            self.spec_dbg_a = try alloc.alloc(f32, len);
            errdefer alloc.free(self.spec_dbg_a);
            self.spec_dbg_b = try alloc.alloc(f32, len);
        }
    }

    /// Copy the live per-token state into the snapshot.
    pub fn specSnapshotTake(self: *Engine) void {
        for (self.spec_snap, 0..) |*s, i| {
            if (i < self.kv.len) s.kv_len = self.kv[i].len;
            if (i < self.gdn.len) {
                const g = self.gdn[i];
                // Both sides are backend regions, so this is a plain copy in the
                // backend's own address space (D2D on CUDA).  It used to be
                // `copyOut` into a HOST slice, which is what made every
                // speculative round push ~230 MB over PCIe.
                if (g.conv.len > 0) self.exec.copyRange(s.conv, g.conv);
                if (g.state.len > 0) self.exec.copyRange(s.state, g.state);
                s.conv_len = g.conv_len;
            }
        }
    }

    /// Wind the state back to the snapshot, then advance it by `rows` tokens of
    /// `tokens` so it reflects a verify block that was `rows` long instead of
    /// the full block.
    ///
    /// The replay exists because the conv history and the delta-rule matrix are
    /// non-linear in the number of steps: restoring the snapshot and re-running
    /// the accepted prefix reproduces exactly the state the model would have had
    /// (`spec_rollback.zig` explains why an incremental undo is not available).
    /// `tokens` holds the ACCEPTED ids; their logits are discarded.
    pub fn specRollbackAndReplay(self: *Engine, tokens: []const u32, pos: u32) void {
        const rows = tokens.len;
        if (!spec_fast_rollback or self.gdn_in.len != self.gdn.len) {
            // 参照路径：恢复快照，然后逐 token 重跑整个 forward。
            for (self.spec_snap, 0..) |*s, i| {
                if (i < self.kv.len) self.kv[i].len = s.kv_len;
                if (i < self.gdn.len) {
                    const g = &self.gdn[i];
                    // 设备到设备，与 `specSnapshotTake` 对称。快照就是放在
                    // 后端内存里，所以这里永远不跨 PCIe。
                    if (g.conv.len > 0) self.exec.copyRange(g.conv, s.conv);
                    if (g.state.len > 0) self.exec.copyRange(g.state, s.state);
                    g.conv_len = s.conv_len;
                }
            }
            for (tokens, 0..) |t, r| _ = self.forwardLogits(t, pos + @as(u32, @intCast(r)));
            return;
        }
        if (cudart_debug_rollback) {
            // `$QWEN35_SPEC_RBDBG`：用两个可逐字节比较的方式重建一次状态。
            // 先恢复到快照、跑 `specRollbackFast`（prelude + 状态内核），再恢复
            // 快照、用参照重放算同一件事；两者不一致就指名是重建的问题，
            // 因为重放自己会恢复每一个承载状态的缓冲。最终以重放的结果继续。
            //
            // 这里的 D2H 缓冲是 `allocSpecSnapshot` 在加载时一次性分配的：
            // 每次现场 `alloc` 3 MB 会把分配点混进热路径，而且失败时若
            // `return` 就会静默跳过真正的回滚。
            if (self.spec_dbg_a.len == 0 or self.spec_dbg_b.len == 0) {
                self.specRollbackFast(rows);
                return;
            }
            self.specRestoreSnapshot();
            self.specRollbackFast(rows);
            self.exec.copyOut(self.spec_dbg_a, self.gdn[0].state);
            self.specRestoreSnapshot();
            for (tokens, 0..) |t, r| _ = self.forwardLogits(t, pos + @as(u32, @intCast(r)));
            self.exec.copyOut(self.spec_dbg_b, self.gdn[0].state);
            var nd: usize = 0;
            var w: f32 = 0;
            for (self.spec_dbg_a, self.spec_dbg_b) |x, y| {
                const d = @abs(x - y);
                if (d > 1e-6) nd += 1;
                if (d > w) w = d;
            }
            std.debug.print("[rbfast] rows={d} L0 state fast-vs-replay ndiff={d}/{d} w={e}\n", .{
                rows, nd, self.spec_dbg_a.len, w,
            });
            return;
        }
        self.specRollbackFast(rows);
    }

    /// 把 GDN 状态回滚到 `rows` 个 token，**不重放 forward**。
    ///
    /// verify 已经把整块每一行都跑过状态内核，而被接受的前缀就是这些行的
    /// 前缀，所以：
    ///
    ///   * KV：只改长度。verify 已追加全部 `k` 行，而越过 `len` 的行从不被读，
    ///     所以 `snapshot + rows` 就是整个回滚。
    ///   * GDN 的 state 与 conv：从 verify 前的快照恢复，然后用**缓存输入**
    ///     (`gdn_in`) 把被接受的行重新走一遍 `gdnStepRowsChunk`——每层一次内核
    ///     调用，而不是整个 65 层 forward。
    ///
    /// 卷积历史在这里不需要单独处理：`gdn_in` 存的是**原始投影（pre-conv）**，
    /// 所以对 accepted 行重跑 prelude 会让 `conv` 恰好前进 `rows` 步，并就地
    /// 重写 `qkv`/`ab`，随后状态内核看到的就是它本来会看到的输入。
    /// （缓存 post-conv 的行是错的——实测：卷积历史无法从它自己的输出反推，
    /// 于是 `conv` 停在 `k` 深而 state 已退到 `rows`。）
    fn specRestoreSnapshot(self: *Engine) void {
        for (self.spec_snap, 0..) |*sn, i| {
            if (i < self.kv.len) self.kv[i].len = sn.kv_len;
            if (i < self.gdn.len) {
                const g = &self.gdn[i];
                if (g.conv.len > 0) self.exec.copyRange(g.conv, sn.conv);
                if (g.state.len > 0) self.exec.copyRange(g.state, sn.state);
                g.conv_len = sn.conv_len;
            }
        }
    }

    fn specRollbackFast(self: *Engine, rows: usize) void {
        if (rows == 0) return;
        const c = self.cfg;
        const nv: usize = c.gdn_n_heads;
        const nk: usize = c.gdn_num_k_heads;
        const kd: usize = c.gdn_head_dim;
        const vd: usize = c.gdn_head_dim;
        const conv_dim: usize = 2 * nk * kd + nv * kd;
        const kq_dim: usize = nk * kd;
        const value_dim: usize = nv * vd;

        for (self.spec_snap, 0..) |*sn, i| {
            if (i < self.kv.len) self.kv[i].len = sn.kv_len + @as(u32, @intCast(rows));
            if (i >= self.gdn.len) continue;
            const g = &self.gdn[i];
            if (g.state.len == 0) continue;
            var used_snap = false;
            // 卷积历史必须从 verify 前恢复（它无法反推，见上面的注释）。
            if (g.conv.len > 0) self.exec.copyRange(g.conv, sn.conv);
            g.conv_len = sn.conv_len;
            // state：verify 的内核已经在每行结束后把整份状态写进 `state_all`
            // 的第 t+1 块，所以回滚**只是把视图指过去**——不拷贝、不重算。
            // 这正是 llama.cpp `rs_idx` 的做法（`llama-memory-recurrent.cpp:825`
            // 的 `cell_id = rs_idx_cur * size + i`）。`rows` 行的状态在槽 `rows`。
            if (rows <= spec_rs_seq and g.state_all.len >= (rows + 1) * nv * kd * vd) {
                g.rs_idx = @intCast(rows);
                g.state = g.state_all.span(rows * nv * kd * vd, nv * kd * vd);
                used_snap = true;
            } else {
                // 行数超出快照容量（不该发生，verify 块 <= spec_verify_rows）：
                // 退回"恢复快照 + 重跑递推"，正确但慢。
                self.exec.copyRange(g.state, sn.state);
                used_snap = false;
            }
            if (i >= self.gdn_in.len) continue;
            const gi = &self.gdn_in[i];
            if (gi.qkv.len < rows * conv_dim or gi.gamma.len < rows * 2 * nv) continue;
            const lw = &self.layers[i];
            if (lw.attn_qkv == null) continue;
            // 1. 用**恢复后的**卷积历史，对 accepted 行重跑 prelude。这一步让
            //    conv 前进 `rows` 步（而不是 verify 的 `k` 步），并就地重算
            //    gamma/beta。
            self.exec.copyRange(self.qkvK.span(0, rows * conv_dim), gi.qkv.span(0, rows * conv_dim));
            self.exec.copyRange(self.abK.span(0, rows * 2 * nv), gi.gamma.span(0, rows * 2 * nv));
            self.exec.gdnPrefillCore(
                self.qkvK.span(0, rows * conv_dim),
                g.conv,
                lw.ssm_conv1d,
                self.abK.span(0, 2 * rows * nv),
                lw.ssm_a,
                lw.ssm_dt_bias,
                self.gdn_prefill_gamma.span(0, rows * nv),
                rows,
                conv_dim,
                0,
                rows * nv,
                nv,
                nv,
                kq_dim,
                nv,
                nk,
                kd,
                c.conv_kernel,
                1e-6,
            );
            // 2. 状态递推只在快照分支没命中时才需要：那时 `g.state` 就是
            //    verify 前的快照，要重新推进 `rows` 行。用了快照就不做。
            if (used_snap) continue;
            self.exec.gdnStepRowsChunk(
                g.state,
                self.qkvK.span(kq_dim, rows * conv_dim - kq_dim),
                self.qkvK.span(2 * kq_dim, rows * conv_dim - 2 * kq_dim),
                self.qkvK.span(0, rows * conv_dim),
                self.kv_mem,
                self.delta,
                self.goutK.span(0, rows * value_dim),
                self.gdn_prefill_gamma.span(0, rows * nv),
                self.abK.span(rows * nv, rows * nv),
                nv,
                nk,
                kd,
                vd,
                rows,
                conv_dim,
                conv_dim,
                conv_dim,
            );
        }
    }

    /// Enable speculative-decoding hidden-state taps (DSpine port, stage 0).
    ///
    /// Allocates `layers.len * nbatch * hidden` floats and makes `forwardChunk`
    /// copy layer `layers[i]`'s output hidden state into plane `i`.  Must be
    /// called AFTER `load` (it needs the backend and `cfg`) and before the
    /// first `forwardChunk` that should be observed.  `layers` must be
    /// ascending and every index must be a real layer.
    ///
    /// Idempotent in the sense that calling it twice re-allocates into the
    /// backend scratch arena; that memory is released wholesale by
    /// `exec.freeScratch()` in `freeAll` (like every other `allocBuf` region),
    /// so this must NOT free it individually -- doing so would double-free.
    ///
    /// COST: the copy is `k * hidden` floats per tapped layer per chunk.  For
    /// the 27B (hidden=5120, 5 taps, k=16 verify) that is 5 * 16 * 5120 * 4 B
    /// = 1.6 MB per verify round, i.e. ~0.15 ms at 10 GB/s -- against an
    /// ~11 ms verify, so it is noise.  It only runs in `forwardChunk`, never in
    /// the single-token decode path.
    /// The output projection as a `[vocab, hidden]` linear: the untied
    /// `output.weight` when the checkpoint has one, else the tied embedding
    /// table.  Same selection `forwardLogits` makes; exposed because the
    /// speculative drafter scores candidates with it (stage 4's
    /// `build_serial_tables` and `topk_ids`).
    /// `lm_head @ x` for a HOST row `x` (`hidden` floats), logits into `out`
    /// (`vocab`).
    ///
    /// This is the DSpine draft's `topk_ids` scoring: the head lives in the
    /// target's backend, so the draft has to borrow it.  The row is uploaded
    /// into engine scratch, projected, and the logits come back -- one H2D, one
    /// launch, one D2H per candidate row (`block_size - 1` = 15 of them per spec
    /// round), against 248320 host-side dequant+dot calls if the draft did it
    /// itself (measured 28 s per round).
    pub fn lmHeadFromHostRow(self: *Engine, out: []f32, x: []const f32) void {
        const hidden: usize = self.cfg.hidden;
        std.debug.assert(x.len == hidden);
        std.debug.assert(out.len >= self.cfg.vocab);
        // `x1` is engine scratch exactly `hidden` wide and is rewritten by every
        // forward anyway, so borrowing it here needs no new allocation.
        const row = self.x1;
        self.exec.copyFromHost(row, x[0..hidden]);
        switch (self.output_weight orelse self.token_embd) {
            .quant => |qrow| self.exec.matvec(self.logits, .{ .quant = qrow }, row),
            .f32 => |m| self.exec.matvec(self.logits, .{ .f32 = m }, row),
        }
        self.exec.copyOut(out[0..self.cfg.vocab], self.logits);
    }

    pub fn lmHead(self: *const Engine) LinearW {
        if (self.output_weight) |ow| return ow;
        return switch (self.token_embd) {
            .quant => |qrow| .{ .quant = qrow },
            .f32 => |m| .{ .f32 = m },
        };
    }

    pub fn specTapEnable(self: *Engine, layers: []const u32) !void {
        self.tap_layers = &.{};
        self.tap_hidden = backend.Buf.empty;
        if (layers.len == 0) return;
        // `nbatch` is 0 when batching is disabled (`QWEN35_NO_BATCH`), and a
        // zero-sized tap buffer would silently disable the taps.  Refuse rather
        // than allocate nothing: the caller asked for observation.
        if (self.nbatch == 0) return error.BatchingDisabled;
        var prev: u32 = 0;
        for (layers, 0..) |l, i| {
            if (l >= self.layers.len) return error.TapLayerOutOfRange;
            if (i > 0 and l <= prev) return error.TapLayersNotAscending;
            prev = l;
        }
        self.tap_hidden = try self.exec.allocBuf(layers.len * self.nbatch * self.cfg.hidden);
        // Copy the indices: the caller's slice may be a temporary.
        const owned = try self.alloc.dupe(u32, layers);
        self.tap_layers = owned;
    }

    /// Which backend this engine drives (debug/verification helpers use it).
    pub fn backendKind(self: *const Engine) backend.Kind {
        return self.backend_kind;
    }

    /// Allocate a one-row hidden-state buffer (debug/verification helper).
    ///
    /// CPU BACKEND ONLY: it hands back a host-readable `Buf` and
    /// `copyHiddenRowIn` memcpys into it, which is invalid on a device buffer
    /// (`Buf.slice()` there is a device pointer; a host `@memcpy` segfaults --
    /// observed as a core dump with `QWEN35_SPEC_TAPS=1 --backend cuda`).
    pub fn allocHiddenRow(self: *Engine) !backend.Buf {
        // Backend-agnostic: on CUDA this is a device region and the caller must
        // only hand it to backend slots (`copyRange`/`rmsnorm`/`nextnPredict`),
        // never to `slice()`.  The host-readback helpers below still refuse a
        // device backend.
        return self.exec.allocBuf(self.cfg.hidden);
    }

    /// `argmax` over the engine's current `logits`, wherever they live.
    ///
    /// On CUDA this is a device reduction that returns a 4-byte index, so the
    /// MTP drafter scores its proposals without a vocab-sized D2H per token.
    pub fn argmaxLogits(self: *Engine) u32 {
        return self.exec.argmax(self.logits);
    }

    /// Argmax of the current logits plus the top-1 softmax probability.
    ///
    /// `p` is what the MTP drafter's early stop keys on (llama.cpp's `p_min`,
    /// `common/speculative.cpp:1789`).  Only the CUDA backend can produce it
    /// cheaply; every other backend reports 1.0, which disables the stop rather
    /// than changing behaviour.
    pub fn argmaxConfLogits(self: *Engine, p: *f32) u32 {
        return self.exec.argmaxConf(self.logits, p);
    }

    /// Copy a host f32 row into `dst` (debug/verification helper).  See
    /// `allocHiddenRow` for the CPU-only restriction.
    pub fn copyHiddenRowIn(self: *Engine, dst: backend.Buf, row: []const f32) void {
        _ = self;
        std.debug.assert(dst.len == row.len);
        @memcpy(dst.slice(), row);
    }

    /// Debug/verification helper: run `output_norm` + lm_head over an arbitrary
    /// hidden row that the caller supplies, leaving the result in `self.logits`.
    /// Used by the `QWEN35_SPEC_TAPS` self-check to prove a tapped last-layer
    /// hidden reproduces `forwardLogits`' logits -- i.e. that the tap really is
    /// the final layer's output and not some other layer's.
    pub fn logitsFromHidden(self: *Engine, h: backend.Buf) void {
        const c = self.cfg;
        self.exec.rmsnorm(self.x1, h, self.output_norm, c.rms_eps);
        if (self.output_weight) |ow| {
            switch (ow) {
                .quant, .f32 => self.exec.matvec(self.logits, ow, self.x1),
            }
        } else {
            switch (self.token_embd) {
                .quant => |qrow| self.exec.matvec(self.logits, .{ .quant = qrow }, self.x1),
                .f32 => |m| self.exec.matvec(self.logits, .{ .f32 = m }, self.x1),
            }
        }
    }

    /// Copy the engine's current `logits` into HOST slice `dst` (`vocab` wide).
    /// Debug/verification helper: the MTP drafter scores its own proposals from
    /// `nextnPredict`, which leaves them in `logits` (`self.logits`), and the
    /// spec loop needs them on the host to argmax.  CPU-only, like the other
    /// host-readback helpers here.
    pub fn copyLogitsToHost(self: *Engine, dst: []f32) !void {
        if (self.backend_kind != .cpu) return error.HostOnly;
        std.debug.assert(dst.len >= self.cfg.vocab);
        self.exec.copyOut(dst[0..self.cfg.vocab], self.logits);
    }

    /// Read-back accessor for tests: the tap plane of `layers[i]`.
    /// `rows * hidden` floats starting at plane `i`, where `rows <= nbatch`.
    /// The target's embedding table as a row view.  The DSpine draft has no
    /// embedding of its own (the reference calls the target's
    /// `embed_tokens`), so `draft.load` needs this rather than a copy.
    /// `max_seq` the engine sized its KV caches for; a draft attached to this
    /// target must not outrun it.
    /// Prefill chunk width (`0` when K-token batching is disabled).  A spec
    /// round's context window is bounded by it: the draft's taps come out of one
    /// `forwardChunk`, and that call refuses more than this many rows.
    pub fn batchLimit(self: *const Engine) usize {
        return if (self.nbatch == 0) 1 else self.nbatch;
    }

    pub fn maxSeqForSpec(self: *const Engine) usize {
        _ = self;
        return max_seq;
    }

    pub fn embedRows(self: *const Engine) !gguf.QuantRow {
        return switch (self.token_embd) {
            .quant => |qrow| qrow,
            .f32 => error.EmbeddingNotQuantized,
        };
    }

    /// Enable the ROLLING tap ring: every `forwardLogits` / `forwardChunk`
    /// appends its rows' tap values, and `specContextRing` hands back the last
    /// `ctx_rows` of them.
    ///
    /// Why a ring at all.  The draft needs the target's hidden states for the
    /// tokens that ALREADY precede the block.  The obvious implementation is to
    /// re-forward that window through the target every round, which costs one
    /// target prefill of `ctx_rows` tokens per round -- measured at 651 s for 32
    /// generated tokens (0.05 tok/s against a 47 tok/s baseline).  A ring makes
    /// it free: those hidden states were computed anyway, during the previous
    /// round's verification, and are simply kept.
    ///
    /// Host memory, `taps * ctx_rows * hidden` f32 = 5 x 16 x 5120 x 4 = 1.6 MB.
    pub fn specRingEnable(self: *Engine, alloc: std.mem.Allocator, ctx_rows: usize) !void {
        if (self.ring.len > 0) return;
        if (self.tap_layers.len == 0) return error.TapsNotEnabled;
        self.ring_ctx = ctx_rows;
        self.ring = try alloc.alloc(f32, self.tap_layers.len * ctx_rows * self.cfg.hidden);
        self.ring_filled = 0;
    }

    /// Push the current tap planes (row 0 of each) into the ring.  Called right
    /// after every single-token forward.
    pub fn specRingPush(self: *Engine) void {
        if (self.ring.len == 0) return;
        const hidden: usize = self.cfg.hidden;
        const ctx = self.ring_ctx;
        const at = self.ring_next;
        for (0..self.tap_layers.len) |ti| {
            const src = self.tap_hidden.span(ti * self.nbatch * hidden, hidden);
            const dst = self.ring[ti * ctx * hidden + at * hidden ..][0..hidden];
            self.exec.copyOut(dst, src);
        }
        self.ring_next = (at + 1) % ctx;
        if (self.ring_filled < ctx) self.ring_filled += 1;
    }

    /// The ring's contents in CHRONOLOGICAL order, flattened tap-major to match
    /// `Draft.projectContext`'s `tap` argument.  `out` must be
    /// `taps * ctx_rows * hidden`.
    ///
    /// Only the rows actually pushed are meaningful; the caller passes
    /// `specRingFilled()` as the row count.
    pub fn specRingRead(self: *Engine, out: []f32) void {
        const hidden: usize = self.cfg.hidden;
        const ctx = self.ring_ctx;
        const nt = self.tap_layers.len;
        const n = self.ring_filled;
        const start = (self.ring_next + ctx - n) % ctx;
        for (0..nt) |ti| {
            for (0..n) |r| {
                const src_at = (start + r) % ctx;
                @memcpy(
                    out[ti * ctx * hidden + r * hidden ..][0..hidden],
                    self.ring[ti * ctx * hidden + src_at * hidden ..][0..hidden],
                );
            }
        }
    }

    /// Seed the ring from the tap planes a PREFILL chunk just wrote: those hold
    /// `rows` consecutive tokens' hidden states for every tapped layer, which is
    /// exactly the draft's left context after the prompt.
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
                const src = self.tap_hidden.span(ti * self.nbatch * hidden + (skip + r) * hidden, hidden);
                self.exec.copyOut(self.ring[ti * ctx * hidden + at * hidden ..][0..hidden], src);
            }
        }
        self.ring_next = (self.ring_next + take) % ctx;
        self.ring_filled = @min(self.ring_filled + take, ctx);
    }

    pub fn specRingFilled(self: *const Engine) usize {
        return self.ring_filled;
    }

    pub fn specTapPlane(self: *const Engine, i: usize, rows: usize) backend.Buf {
        std.debug.assert(i < self.tap_layers.len);
        std.debug.assert(rows <= self.nbatch);
        return self.tap_hidden.span(i * self.nbatch * self.cfg.hidden, rows * self.cfg.hidden);
    }

    /// Copy tap plane `i`'s first `rows` rows into HOST slice `dst`
    /// (`rows * hidden` floats).
    ///
    /// The draft lives on its own backend -- it may be the CPU one while the
    /// target is on the GPU, or vice versa -- and `Draft.projectContext` takes a
    /// host slice, so the planes have to come back over PCIe either way.  That
    /// is `rows * hidden * 4` bytes per tap plane per spec round (33 x 5120 x 4
    /// x 5 = 3.4 MB), which is the price of the two models not sharing a
    /// device; the kernel-level A/B that would avoid it is not worth it while
    /// stage 4 is still a host loop.
    pub fn specTapPlaneToHost(self: *Engine, dst: []f32, i: usize, rows: usize) void {
        const hidden: usize = self.cfg.hidden;
        std.debug.assert(dst.len >= rows * hidden);
        // `copyOut`, not `copyRange`: the destination is HOST memory, and
        // `copyRange` decides the direction from the source alone (a device
        // source + a host destination is a device-to-device move as far as it
        // can tell, which writes through a host address).
        self.exec.copyOut(dst[0 .. rows * hidden], self.specTapPlane(i, rows));
    }

    /// Tap plane `i`, first `rows` rows, each passed through `output_norm` --
    /// the exact value llama.cpp exports as `t_h_nextn` and feeds to the
    /// drafter (`qwen35.cpp:210` norms the trunk's last residual stream with
    /// `output_norm` before the MTP block sees it).
    /// Returns host memory (`rows * hidden`), so it is CPU-BACKEND ONLY: on CUDA
    /// the plane lives in device memory.  Debug/self-check use only.
    /// `out = lm_head @ h` for one already-normed `[hidden]` row.
    ///
    /// Exposed for the drafter's self-check, which has to tell two failures
    /// apart that give the same acceptance rate: a broken draft forward pass
    /// and a broken candidate scoring path.  Uses the same `matvec` as
    /// `logitsFromHidden`, MINUS its terminal rmsnorm -- callers hand in a row
    /// that has already been normed (see `specTapNormedPlane`), so norming
    /// twice here would be a silent difference from the engine's own path.
    pub fn selfCheckLogits(self: *Engine, out: []f32, h: []const f32) void {
        const c = self.cfg;
        const dst = self.exec.allocBuf(c.vocab) catch unreachable;
        defer self.exec.freeScratch();
        _ = dst;
        const src = self.exec.allocBuf(c.hidden) catch unreachable;
        @memcpy(src.slice()[0..c.hidden], h[0..c.hidden]);
        const lw = self.lmHead();
        const tmp = self.exec.allocBuf(c.vocab) catch unreachable;
        self.exec.matvec(tmp, lw, src);
        @memcpy(out[0..c.vocab], tmp.slice()[0..c.vocab]);
    }

    pub fn specTapNormedPlane(self: *const Engine, i: usize, rows: usize) ![]f32 {
        const hidden: usize = self.cfg.hidden;
        const plane = self.specTapPlane(i, rows).slice();
        const w = self.output_norm.slice();
        const out = try self.alloc.alloc(f32, rows * hidden);
        for (0..rows) |r| {
            rmsNorm(out[r * hidden ..][0..hidden], plane[r * hidden ..][0..hidden], w, self.cfg.rms_eps);
        }
        return out;
    }

    /// Device-side twin of `specTapNormedPlane`: `output_norm` applied to tap
    /// row `row` of plane `i`, written into the caller's buffer `dst` (at least
    /// `hidden` long, `Buf` so it can live on the device).
    ///
    /// The MTP drafter consumes exactly this value, and it must stay on the
    /// device: `specTapNormedPlane` returns a HOST slice (it calls `slice()` on
    /// the tap plane), which on CUDA dereferences a device pointer.  One
    /// `rmsnorm` launch keeps the whole draft on the GPU.
    pub fn specTapNormedRowInto(self: *Engine, dst: backend.Buf, i: usize, row: usize) void {
        const hidden: usize = self.cfg.hidden;
        std.debug.assert(dst.len >= hidden);
        std.debug.assert(row < self.nbatch);
        // Row `row` of plane `i`: the plane is `nbatch` rows, so the offset is
        // `row * hidden` inside it.
        const src = self.specTapPlane(i, self.nbatch).span(row * hidden, hidden);
        self.exec.rmsnorm(dst.span(0, hidden), src, self.output_norm, self.cfg.rms_eps);
    }

    pub const GenResult = struct {
        tokens: []u32,
        prefill_ms: f64,
        /// average wall time per generated token (ms)
        ms_per_token: f64,
        /// total wall time spent in decode steps, forward passes only (ms)
        decode_ms: f64,
        /// true when generation stopped because the next sampled token was EOS
        hit_eos: bool,
        /// true when generation stopped because the KV cache (`Engine.max_seq`)
        /// ran out before `n_predict` tokens were produced
        hit_context_limit: bool = false,
        /// Speculative decoding counters (all zero when `generate` ran the
        /// plain loop).  `proposed` counts draft tokens offered to the verifier,
        /// `accepted` how many of them the target agreed with, and `rounds` how
        /// many verify passes were spent -- `accepted / proposed` is the
        /// acceptance rate, and `accepted / rounds` is the mean accepted length.
        spec_proposed: usize = 0,
        spec_accepted: usize = 0,
        spec_rounds: usize = 0,
        /// `QWEN35_SPEC_POSSTAT`: rounds bucketed by accepted count, so
        /// "chain[0] always right, chain[1] never" is distinguishable from
        /// "uniformly weak".
        spec_pos_reached: [64]usize = @splat(0),
        /// Per-row candidate recall (`$QWEN35_SPEC_RECALL`): entry `r` is
        /// `recall_hit[r] / recall_tries[r]`, the share of rounds that reached
        /// block row `r+1` whose candidate set contained the target's own pick.
        spec_recall_hit: [64]usize = @splat(0),
        spec_recall_tries: [64]usize = @splat(0),
        /// `$QWEN35_SPEC_RECALL`: how each rejected row failed.
        spec_miss_select: usize = 0,
        spec_miss_forward: usize = 0,
    };

    /// Prefill prompt (forward-only, no PRNG draws), then generate up to
    /// `n_predict` **new** tokens with `self.sampler` (greedy argmax when its
    /// temperature is 0 — the historical default), stopping at EOS or when the
    /// KV cache is full (`Engine.max_seq`).  The first generated token is
    /// sampled from the last prefill position, so the PRNG stream starts at the
    /// same point on CPU and CUDA.  Caller owns result.tokens.
    ///
    /// `n_predict` counts generated tokens, not the total sequence length.  The
    /// two used to be conflated (`while (pos < max_tokens)` with `pos` starting
    /// at `prompt.len`), so `-n 48` emitted `48 - prompt.len` tokens and `-n`
    /// below the prompt length emitted nothing at all.  Anything comparing our
    /// token counts against llama.cpp (whose `-n` already means "predict N")
    /// must keep them separate.
    pub fn generate(
        self: *Engine,
        alloc: std.mem.Allocator,
        prompt: []const u32,
        n_predict: usize,
        progress: ?*const fn (ctx: ?*anyopaque, tok: u32) void,
        progress_ctx: ?*anyopaque,
    ) !GenResult {
        // Restart the PRNG at its seed so every `generate` call (and every
        // process started with the same -s) samples identically.  The prompt
        // itself never draws (forwardLogits below), so the first coin spent
        // is the first generated token — on CPU and CUDA alike.
        self.sampler.reseed();
        self.reset();

        // The KV cache is `Engine.max_seq` slots; the prompt must fit and leave
        // room for at least the sampled-but-not-yet-appended token, otherwise
        // the prefill loop would run past the end of the cache.
        if (prompt.len >= max_seq) {
            std.debug.print(
                "generate: prompt is {d} tokens, but the context holds {d}\n",
                .{ prompt.len, max_seq },
            );
            return error.ContextLengthExceeded;
        }

        const io = self.io;
        // Profiling covers prefill too: prof_phase used to be set only after
        // the prefill loop, so every prefill sample reported calls=0 and the
        // batched kernels -- exactly the part worth profiling -- were invisible.
        self.exec.profReset();
        self.exec.profPhase(true);
        const t_start = Io.Clock.Timestamp.now(io, .awake);
        var last_logits: Buf = Buf.empty;
        if (self.nbatch == 0) {
            for (prompt, 0..) |t, i| last_logits = self.forwardLogits(t, @intCast(i));
        } else {
            // K-token batched prefill: every layer's weights are read once per
            // chunk (batched projections) and reused across the chunk's K
            // tokens; the KV cache / GDN state updates happen token-by-token in
            // the same order as the serial loop, so the resulting logits of the
            // last prompt token are bit-identical to the serial prefill.
            var i: usize = 0;
            while (i < prompt.len) {
                const n = @min(self.nbatch, prompt.len - i);
                last_logits = self.forwardChunk(prompt[i .. i + n], @intCast(i));
                i += n;
            }
        }
        const t_after = Io.Clock.Timestamp.now(io, .awake);
        self.prefill_ms = @as(f64, @floatFromInt(t_start.durationTo(t_after).raw.nanoseconds)) / 1e6;
        // Same boundary for the backend's own accounting.  A CPU backend ignores
        // it; the CUDA backend closes its GPU-side prefill window here instead of
        // at `profPhase(false)`, which only fires after the decode loop and would
        // fold decode into the prefill numbers.
        self.exec.profPrefillEnd();

        var out: std.ArrayList(u32) = .empty;
        errdefer out.deinit(alloc);
        var pos: u32 = @intCast(prompt.len);
        var generated: usize = 0;
        var gen_ns: i96 = 0;
        var hit_eos = false;
        var hit_context_limit = false;
        // prof_phase stays on across prefill and decode (set above); do not
        // reset here or the prefill samples are thrown away.
        var next: u32 = self.exec.sample(last_logits, self.sample_scratch, &self.sampler);
        // Two independent limits: how many tokens the caller asked for, and how
        // many slots the KV cache has left.  `pos` advances regardless.
        while (generated < n_predict) {
            if (pos >= max_seq) {
                hit_context_limit = true;
                break;
            }
            if (next == self.cfg.eos_token_id) {
                hit_eos = true;
                break;
            }
            try out.append(alloc, next);
            if (progress) |f| f(progress_ctx, next);
            const t0 = Io.Clock.Timestamp.now(io, .awake);
            next = self.step(next, pos);
            gen_ns += t0.durationTo(Io.Clock.Timestamp.now(io, .awake)).raw.nanoseconds;
            pos += 1;
            generated += 1;
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
        };
    }

    /// The draft callback a speculative round calls to get proposals.
    ///
    /// `context` is the trailing window of ALREADY-COMMITTED token ids (`ctx_rows`
    /// of them, the block's left context); `anchor` is the id the block starts
    /// from.  The callee fills `out` with up to `out.len` candidate ids and
    /// returns how many it wrote.  Returns 0 to skip drafting this round (the
    /// caller then runs one plain step, so a drafter that cannot answer costs
    /// nothing but the call).
    pub const DraftFn = *const fn (
        ctx: *anyopaque,
        context: []const u32,
        anchor: u32,
        /// Absolute position the block's FIRST row sits at (`anchor`'s position
        /// + 1).  The draft's RoPE is absolute, so this cannot be inferred from
        /// the row index.
        block_pos: u32,
        out: []u32,
    ) usize;

    /// `QWEN35_SPEC_RECALL`: the draft's candidate sets for the round it just
    /// proposed, `[row][k]` with row 0 = block row 1 (the anchor row is never
    /// predicted).  Called right after `draft_fn` returns, before verification,
    /// so the caller can test each row's candidates against the token the target
    /// actually picks.  Separate from `DraftFn` because the candidate table is a
    /// diagnostic, not part of the proposal contract.
    pub const CandFn = *const fn (
        ctx: *anyopaque,
        /// Filled with `rows * cfg.spec_recall_k` ids, row-major.
        cand: []u32,
        rows: usize,
    ) void;

    /// Speculative decoding: draft proposes, this loop verifies.
    ///
    /// SHAPE OF A ROUND (DSpine / dspine_mlx `generate.py`):
    ///   1. the draft proposes up to `draft.len` ids for positions
    ///      `pos+1 .. pos+draft.len`, conditioned on the committed context;
    ///   2. the target runs them through ONE verify pass -- here literally
    ///      `draft.len` `forwardLogits` calls, so every state kernel sees the
    ///      same per-token order the plain loop would give it and the resulting
    ///      logits are identical to a serial run (which is why acceptance is
    ///      meaningful: the target is not approximating itself);
    ///   3. the accepted prefix is the longest one where the target's own
    ///      greedy pick matches the proposal.  One extra token (the target's own
    ///      choice at the first mismatch, or at the end of an all-accepted
    ///      block) is committed too -- that is the `accepted + 1` of the
    ///      reference;
    ///   4. the state is wound back to `accepted` rows (`specRollbackAndReplay`)
    ///      because the verify already pushed all `draft.len` of them.
    ///
    /// WHY THE VERIFY IS PER-TOKEN AND NOT BATCHED: `forwardChunk` would compute
    /// the same logits from the same weights, but its GDN/attention state update
    /// is the thing the rollback has to undo, and a batched verify makes that
    /// state a function of the CHUNK LENGTH rather than of the accepted prefix.
    /// The replay in `specRollbackAndReplay` already handles the per-token case
    /// exactly, so batching would trade a measurable risk for a launch count.
    /// It is also not the current bottleneck: the draft forward is.
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
        /// `QWEN35_SPEC_RECALL` only: reports the round's candidate table so
        /// the caller can score recall.  Null (the default) costs nothing.
        cand_fn: ?CandFn,
        cand_ctx: ?*anyopaque,
    ) !GenResult {
        self.sampler.reseed();
        self.reset();

        if (prompt.len >= max_seq) {
            std.debug.print(
                "generateSpec: prompt is {d} tokens, but the context holds {d}\n",
                .{ prompt.len, max_seq },
            );
            return error.ContextLengthExceeded;
        }
        const io = self.io;
        self.exec.profReset();
        self.exec.profPhase(true);
        const t_start = Io.Clock.Timestamp.now(io, .awake);
        var last_logits: Buf = Buf.empty;
        if (self.nbatch == 0) {
            for (prompt, 0..) |t, i| last_logits = self.forwardLogits(t, @intCast(i));
        } else {
            var i: usize = 0;
            while (i < prompt.len) {
                const n = @min(self.nbatch, prompt.len - i);
                last_logits = self.forwardChunk(prompt[i .. i + n], @intCast(i));
                i += n;
            }
        }
        const t_after = Io.Clock.Timestamp.now(io, .awake);
        self.prefill_ms = @as(f64, @floatFromInt(t_start.durationTo(t_after).raw.nanoseconds)) / 1e6;
        self.exec.profPrefillEnd();

        // The prompt's own hidden states are the draft's first context window;
        // without this the first round drafts from a single row (`nctx=1`) and
        // nothing it proposes can be right.
        if (self.nbatch != 0) self.specRingFillFromChunk(prompt.len, 0);

        var out: std.ArrayList(u32) = .empty;
        errdefer out.deinit(alloc);
        // The committed id stream INCLUDING the prompt: the draft needs real
        // left context, and after a few rounds most of it is generated text.
        var history: std.ArrayList(u32) = .empty;
        defer history.deinit(alloc);
        try history.appendSlice(alloc, prompt);

        var pos: u32 = @intCast(prompt.len);
        var generated: usize = 0;
        var gen_ns: i96 = 0;
        var hit_eos = false;
        var hit_context_limit = false;
        var next: u32 = self.exec.sample(last_logits, self.sample_scratch, &self.sampler);
        var proposed_total: usize = 0;
        var accepted_total: usize = 0;
        var rounds: usize = 0;
        // `QWEN35_SPEC_RECALL` bookkeeping: per block-row position (0 = block
        // row 1), how often the target's own pick was among that row's
        // candidates.  Row position `r` is reached in a round only when rows
        // 0..r-1 were accepted, so a low count at `r` also reflects the
        // rounds that never got there -- `recall_tries` records that.
        var recall_hit: [64]usize = @splat(0);
        var recall_tries: [64]usize = @splat(0);
        const recall_k = self.spec_recall_k;

        const draft_buf = try alloc.alloc(u32, max_draft);
        defer alloc.free(draft_buf);
        // `QWEN35_SPEC_RECALL` scratch: `max_draft * k` candidate ids, `k` being
        // the draft's `selector_top_k`.  Allocated lazily on the first round
        // (the callback reports `k`, which the engine does not otherwise know).
        var cand_buf: []u32 = &.{};
        if (cand_fn != null and self.spec_recall_k > 0) {
            cand_buf = try alloc.alloc(u32, max_draft * self.spec_recall_k);
        }

        while (generated < n_predict) {
            if (pos >= max_seq) {
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
            // `next` sits at `pos` and is already committed.  The block the
            // draft may fill occupies positions `pos+1 ..`.
            const block_pos = pos + 1;
            const room = @min(max_draft, @min(n_predict - generated - 1, max_seq - block_pos));

            var advanced: usize = 1;
            // Set when the verify block already advanced the target over this
            // position; the plain `step` below must then be skipped, or the
            // target moves TWO rows while `pos` only advances one.
            var verified = false;
            if (room > 0) {
                const from = if (history.items.len > ctx_rows) history.items.len - ctx_rows else 0;
                const n_draft = draft_fn(draft_ctx, history.items[from..], next, block_pos, draft_buf[0..room]);
                if (n_draft > 0) {
                    rounds += 1;
                    proposed_total += n_draft;
                    verified = true;
                    // ---- verification uses the ORDINARY step ----
                    // The first draft id is scored by target's own `step` for
                    // position `pos`; every further one by the step for the
                    // position before it.  Reusing `step` (rather than a bespoke
                    // argmax walk) keeps the accepted path bit-identical to plain
                    // decode, which is the whole point: an accepted block must
                    // produce exactly the tokens greedy decode would have.
                    self.specSnapshotTake();
                    // ---- BATCHED VERIFY: one forward for the whole block ----
                    // Rows are `[next, draft[0], draft[1], ...]`, so row r's
                    // logits are the prediction that FOLLOWS row r's input --
                    // exactly what `draft_buf[r]` must be compared against.
                    var vbuf: [spec_verify_rows]u32 = undefined;
                    var n_verify: usize = 0;
                    const room_verify = @min(n_draft + 1, spec_verify_rows);
                    vbuf[n_verify] = next;
                    n_verify += 1;
                    while (n_verify < room_verify) : (n_verify += 1) {
                        vbuf[n_verify] = draft_buf[n_verify - 1];
                    }
                    const vlogits = self.forwardChunkVerify(vbuf[0..n_verify], pos);
                    // ONE read-back for every row's pick.  Asking per row meant
                    // five blocking D2H per round, each waiting on the whole
                    // round's launches (measured: 6.5 ms of host wait per 9 us
                    // kernel).
                    var vpicks: [spec_verify_rows]u32 = undefined;
                    self.exec.argmaxRows(vlogits, n_verify, vpicks[0..n_verify]);
                    self.specRingPush();
                    var committed: usize = 0;
                    var pending: u32 = vpicks[0];
                    if (self.spec_trace_dbg) {
                        std.debug.print("ROUND pos={d} anchor={d} draft[0]={d} target[0]={d} n={d} rows={d} vbuf:", .{
                            pos, next, draft_buf[0], pending, n_draft, n_verify,
                        });
                        for (0..n_verify) |r| std.debug.print(" {d}", .{vbuf[r]});
                        std.debug.print("  argmax:", .{});
                        for (0..n_verify) |r| std.debug.print(" {d}", .{vpicks[r]});
                        std.debug.print("\n", .{});
                    }
                    // `QWEN35_SPEC_RECALL`: fetch this round's candidate table.
                    // It is scored inside the verify loop below, where the
                    // target's own pick for each row is actually available.
                    const have_cand = cand_fn != null;
                    if (cand_fn) |cf| cf(cand_ctx.?, cand_buf[0 .. n_draft * recall_k], n_draft);
                    while (committed < n_draft) : (committed += 1) {
                        const cand = draft_buf[committed];
                        // `QWEN35_SPEC_RECALL`: `pending` is now the target's
                        // own pick for row `committed`, and the draft's candidate
                        // set for that row is `cand_buf[committed]`.  Recall =
                        // "was the target's token even OFFERED", which is the
                        // gate that separates a scoring defect from a forward
                        // defect: high recall + wrong pick means stage 4 picked
                        // badly; low recall means the block forward never
                        // produced the right hidden state.
                        if (have_cand and committed < recall_hit.len) {
                            recall_tries[committed] += 1;
                            var offered = false;
                            for (cand_buf[committed * recall_k ..][0..recall_k]) |c| {
                                if (c == pending) {
                                    recall_hit[committed] += 1;
                                    offered = true;
                                    break;
                                }
                            }
                            // `QWEN35_SPEC_RECALL` split: an offered-but-rejected
                            // token is a *selection* loss (stage 4 scored it below
                            // another candidate); a token that was never offered
                            // is a *forward* loss (the block never produced it).
                            if (pending != cand) {
                                if (offered) self.recall_miss_select += 1 else self.recall_miss_forward += 1;
                            }
                        }
                        if (pending != cand) break;
                        if (generated + 1 + committed >= n_predict) break;
                        try out.append(alloc, cand);
                        try history.append(alloc, cand);
                        if (progress) |f| f(progress_ctx, cand);
                        // The NEXT row's logits are already in `vlogits` -- that
                        // is the point of the batched verify, so there is no
                        // forward call here any more.
                        pending = vpicks[committed + 1];
                    }
                    accepted_total += committed;
                    // `QWEN35_SPEC_POSSTAT`: how far down the chain each round
                    // gets.  A chain whose first element is right and whose
                    // second never is has a different shape from one that is
                    // uniformly weak, and `accepted/rounds` cannot tell them
                    // apart.
                    if (self.spec_posstat) {
                        self.spec_pos_reached[committed] += 1;
                    }
                    // ALWAYS wind the state back, including the all-rejected
                    // case (`committed == 0`).  The batched verify pushes the
                    // whole `[next, draft...]` block into the KV cache, so
                    // leaving it there when nothing was accepted puts `pos +
                    // n_verify` rows in the cache while `pos` advances by one --
                    // every later round's `slot0` is then past the end of the
                    // real prefix.  (The per-token verify this replaced fed only
                    // `next` at `pos` and stopped there when `draft[0]` was
                    // rejected, so it never had to undo anything.)
                    //
                    // The replay starts at `next` (position `pos`), not at the
                    // first drafted token: that is the row the next round's
                    // chunk expects to find already in the cache.
                    {
                        var replay_buf: [spec_verify_rows]u32 = undefined;
                        replay_buf[0] = next;
                        @memcpy(replay_buf[1 .. committed + 1], draft_buf[0..committed]);
                        self.specRollbackAndReplay(replay_buf[0 .. committed + 1], pos);
                    }
                    advanced = committed + 1;
                    next = pending;
                }
            }
            if (advanced == 1 and !verified) {
                // Nothing was drafted this round: one ordinary step, exactly as
                // plain decode would do.
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
            .spec_pos_reached = self.spec_pos_reached,
            .spec_recall_hit = recall_hit,
            .spec_recall_tries = recall_tries,
            .spec_miss_select = self.recall_miss_select,
            .spec_miss_forward = self.recall_miss_forward,
        };
    }

    /// Ask the backend to print its **complete** profile report once (per-type
    /// matvec counters, `PROF-SLOT`, `PROF-COPY`; see
    /// `backend.Exec.profReport`).
    ///
    /// A one-line pass-through on purpose: `engine.zig` must not know which
    /// hardware backend is in use (AGENTS.md §1.1 -- it imports no kernel or
    /// device module), so the caller (`main.zig`) hands the decision to the
    /// backend through the `Exec` vtable instead.  Callers gate this on their
    /// own debug switch; a CPU backend prints only its matvec table.
    ///
    /// Call it after `generate` returns: that is the only point where the run
    /// is over and the counters are final (the engine is arena-allocated, so
    /// `deinit` is not reached on the CLI path).
    pub fn profBackendReport(self: *Engine) void {
        self.exec.profReport();
    }
};

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

test "rms norm: x*rsqrt(mean(x^2)+eps)*w" {
    const w = [_]f32{ 1, 1 };
    var out: [2]f32 = undefined;
    rmsNorm(&out, &.{ 3, 4 }, &w, 0);
    const inv = 1.0 / @sqrt(12.5);
    try std.testing.expectApproxEqAbs(@as(f32, 3 * inv), out[0], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 4 * inv), out[1], 1e-6);
    const w2 = [_]f32{ 2, 0.5 };
    rmsNorm(&out, &.{ 3, 4 }, &w2, 0);
    try std.testing.expectApproxEqAbs(@as(f32, 3 * inv * 2), out[0], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 4 * inv * 0.5), out[1], 1e-6);
}

test "silu / sigmoid / softplus reference values" {
    try std.testing.expectApproxEqAbs(@as(f32, 0), silu(0), 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0.7310586), silu(1), 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, -0.26894143), silu(-1), 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 0.5), sigmoid(0), 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, @log(@as(f32, 2))), softplus(0), 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 25), softplus(25), 1e-5); // threshold branch
}

test "softmax sums to one and is shift-stable" {
    var x = [_]f32{ 1, 2, 1000, -5 };
    softmax(&x);
    var sum: f32 = 0;
    for (x) |v| sum += v;
    try std.testing.expectApproxEqAbs(@as(f32, 1), sum, 1e-5);
    try std.testing.expect(x[2] > 0.99);
}

test "l2norm eps inside the sum" {
    var x = [_]f32{ 3, 4 };
    l2norm(&x);
    // 1/sqrt(25 + 1e-6): almost exactly 0.2
    try std.testing.expectApproxEqAbs(@as(f32, 3 * 0.2), x[0], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 4 * 0.2), x[1], 1e-6);
}

test "rope64: pos 0 identity, cross-half pairing, tail untouched" {
    var x: [256]f32 = undefined;
    for (&x, 0..) |*p, i| p.* = @floatFromInt(i);
    applyRope64(x[0..64], 0, 1e7);
    for (0..64) |i| try std.testing.expectApproxEqAbs(@as(f32, @floatFromInt(i)), x[i], 1e-4);
    for (64..256) |i| try std.testing.expectEqual(@as(f32, @floatFromInt(i)), x[i]);

    // pos 1: pair (0, 32) rotates by theta=1 (inv_freq[0]=1)
    var y: [256]f32 = undefined;
    for (&y, 0..) |*p, i| p.* = @floatFromInt(i);
    applyRope64(y[0..64], 1, 1e7);
    // SPEC B10 cos/sin table: cos(theta=1)=0.54030234, sin=0.84147096
    // y[0] = 0*cos - 32*sin ; y[32] = 0*sin + 32*cos
    try std.testing.expectApproxEqAbs(@as(f32, -32 * 0.84147096), y[0], 1e-3);
    try std.testing.expectApproxEqAbs(@as(f32, 32 * 0.54030234), y[32], 1e-3);
    // pair (1, 33): theta = 0.60429639 (inv_freq[1])
    const c1: f32 = 0.82290208;
    const s1: f32 = 0.56818324;
    try std.testing.expectApproxEqAbs(1 * c1 - 33 * s1, y[1], 1e-3);
    try std.testing.expectApproxEqAbs(1 * s1 + 33 * c1, y[33], 1e-3);
}

test "rope64: inv_freq[31] matches 1e7^-62/64" {
    // pair (31,63) rotates with inv_freq[31] = 1e7^(-62/64) = 1.6548171e-07
    var x: [64]f32 = undefined;
    @memset(&x, 0);
    x[31] = 1;
    applyRope64(&x, 1000000, 1e7);
    const expected_cos = @cos(@as(f32, 1000000) * 1.6548171e-07);
    try std.testing.expectApproxEqAbs(expected_cos, x[31], 1e-4);
    // also check pos=0 gives cos(theta0)=cos(1e6) for the pair (0,32)
    var y: [64]f32 = undefined;
    @memset(&y, 0);
    y[0] = 1;
    applyRope64(&y, 1000000, 1e7);
    try std.testing.expectApproxEqAbs(@cos(@as(f32, 1000000)), y[0], 1e-4);
}

test "config layer typing" {
    const c = Config{};
    try std.testing.expect(!c.isFullAttention(0));
    try std.testing.expect(!c.isFullAttention(2));
    try std.testing.expect(c.isFullAttention(3));
    try std.testing.expect(c.isFullAttention(23));
}

test "convStep: causal left-pad semantics" {
    // single channel with taps [a0,a1,a2,a3]; feed three tokens manually
    const alloc = std.testing.allocator;
    const conv = try alloc.alloc(f32, 4);
    defer alloc.free(conv);
    @memset(conv, 0);
    var eng = Engine{
        .alloc = alloc,
        .cfg = .{},
        .g = undefined,
        .mm = undefined,
        .file = undefined,
        .io = undefined,
        .token_embd = .{ .f32 = &.{} },
        .output_weight = null,
        .output_norm = backend.Buf.empty,
        .exec = undefined,
        .layers = &.{},
        .kv = &.{},
        .gdn = &.{},
        .x = backend.Buf.empty,
        .x1 = backend.Buf.empty,
        .x2 = backend.Buf.empty,
        .qkv = backend.Buf.empty,
        .z = backend.Buf.empty,
        .ab = backend.Buf.empty,
        .gdn_out = backend.Buf.empty,
        .q = backend.Buf.empty,
        .k = backend.Buf.empty,
        .v = backend.Buf.empty,
        .attn = backend.Buf.empty,
        .mlp_h = backend.Buf.empty,
        .mlp_h1 = backend.Buf.empty,
        .logits = backend.Buf.empty,
        .delta = backend.Buf.empty,
        .kv_mem = backend.Buf.empty,
        .gamma_scratch = backend.Buf.empty,
        .gdn_prefill_gamma = backend.Buf.empty,
        .prefill_ms = 0,
    };
    // taps: w = [1, 0.5, 0.25, 2]
    const wc = [_]f32{ 1, 0.5, 0.25, 2 };
    var qkv = [_]f32{1};
    eng.convStep(conv, &qkv, &wc, 4);
    // token0: state all zero -> out = 1*2 = 2, silu(2)
    try std.testing.expectApproxEqAbs(@as(f32, 2.0 / (1.0 + @exp(@as(f32, -2)))), qkv[0], 1e-5);
    qkv[0] = 10; // token1
    eng.convStep(conv, &qkv, &wc, 4);
    // window: [0,0,1(raw t0),10] -> 0*1+0*.5+1*.25+10*2 = 20.25
    try std.testing.expectApproxEqAbs(silu(@as(f32, 20.25)), qkv[0], 1e-4);
    qkv[0] = 100; // token2
    eng.convStep(conv, &qkv, &wc, 4);
    // window: [0,1,10,100] -> 0+0.5+2.5+200 = 203
    try std.testing.expectApproxEqAbs(silu(@as(f32, 203)), qkv[0], 1e-3);
}

test "gdn delta rule one step, hand computed" {
    // 1 head, kd=2, vd=2. S starts 0.
    // q=[1,0], k=[0,1], v=[3,4], beta=0.5, gamma=0.9
    // (1) S *= 0.9 -> 0
    // (2) kv_mem = S^T k = [0,0]
    // (3) delta = 0.5*[3,4] = [1.5,2]; S += k outer delta = [[0,0],[1.5,2]]
    // (4) o = S^T q = row0 = [0,0]
    var state: [4]f32 = @splat(0);
    for (&state) |*p| p.* *= 0.9;
    const k = [_]f32{ 0, 1 };
    var delta: [2]f32 = undefined;
    var km: [2]f32 = undefined;
    @memset(&km, 0);
    for (0..2) |i| {
        for (0..2) |j| km[j] += state[i * 2 + j] * k[i];
    }
    const v = [_]f32{ 3, 4 };
    const beta: f32 = 0.5;
    for (0..2) |j| delta[j] = (v[j] - km[j]) * beta;
    for (0..2) |i| {
        for (0..2) |j| state[i * 2 + j] += k[i] * delta[j];
    }
    const q = [_]f32{ 1, 0 };
    var o: [2]f32 = undefined;
    for (0..2) |j| {
        var sum: f32 = 0;
        for (0..2) |i| sum += state[i * 2 + j] * q[i];
        o[j] = sum;
    }
    try std.testing.expectEqual(@as(f32, 0), o[0]);
    try std.testing.expectEqual(@as(f32, 0), o[1]);
    try std.testing.expectEqual(@as(f32, 0), state[0]);
    try std.testing.expectEqual(@as(f32, 0), state[1]);
    try std.testing.expectEqual(@as(f32, 1.5), state[2]);
    try std.testing.expectEqual(@as(f32, 2), state[3]);

    // second token with q=[0,1]: o = S^T q = row1 = [1.5,2]
    for (0..2) |j| {
        var sum: f32 = 0;
        for (0..2) |i| sum += state[i * 2 + j] * k[i]; // reuse k=[0,1] as q
        o[j] = sum;
    }
    try std.testing.expectEqual(@as(f32, 1.5), o[0]);
    try std.testing.expectEqual(@as(f32, 2), o[1]);
}

test "q8ScratchBytes: ceil(cols/32) blocks of 34 bytes" {
    try std.testing.expectEqual(@as(usize, 34), q8ScratchBytes(32));
    try std.testing.expectEqual(@as(usize, 34), q8ScratchBytes(1));
    try std.testing.expectEqual(@as(usize, 68), q8ScratchBytes(33));
    try std.testing.expectEqual(@as(usize, 68), q8ScratchBytes(64));
    // 27B ffn_down input: n_ff = 17408 = 544 blocks
    try std.testing.expectEqual(@as(usize, 544 * 34), q8ScratchBytes(17408));
}

test "gdn chunk state: rows=r == r single steps (the rollback's premise)" {
    // SPECULATIVE ROLLBACK DEPENDS ON THIS, and nothing tested it before.
    //
    // A round verifies a block of `k` rows, then keeps only `accepted + 1`.  The
    // cheap rollback replaces "replay the accepted prefix through the whole
    // model" with "restore the pre-verify state and re-step `rows` rows through
    // the state kernel".  That is only valid if the chunk kernel's `rows`-row
    // run is exactly the same recurrence as `rows` single steps -- this test is
    // that claim, stated once, on the kernel pair the engine actually calls.
    //
    // The first version of the rollback change broke acceptance 47.6% -> 2.8%
    // and this test did not exist; every one of those failures would have been
    // caught here in milliseconds instead of by a 27B run.
    const alloc = std.testing.allocator;
    const heads = 2;
    const nk = 2;
    const kd = 128;
    const vd = 128;
    const rows = 5;
    const kb = try alloc.alloc(f32, rows * nk * kd);
    defer alloc.free(kb);
    const vb = try alloc.alloc(f32, rows * heads * vd);
    defer alloc.free(vb);
    const qb = try alloc.alloc(f32, rows * nk * kd);
    defer alloc.free(qb);
    const gb = try alloc.alloc(f32, rows * heads);
    defer alloc.free(gb);
    const bb = try alloc.alloc(f32, rows * heads);
    defer alloc.free(bb);
    var prng = std.Random.DefaultPrng.init(0x9E3779B97F4A7C15);
    const rnd = prng.random();
    for (kb) |*x| x.* = rnd.float(f32) * 2 - 1;
    for (vb) |*x| x.* = rnd.float(f32) * 2 - 1;
    for (qb) |*x| x.* = rnd.float(f32) * 2 - 1;
    for (gb) |*x| x.* = 0.80 + rnd.float(f32) * 0.18;
    for (bb) |*x| x.* = 0.25 + rnd.float(f32) * 0.5;

    const st0 = try alloc.alloc(f32, heads * kd * vd);
    defer alloc.free(st0);
    for (st0) |*x| x.* = (rnd.float(f32) - 0.5) * 0.1;

    var g: gguf.File = undefined;
    var cpu = backend_cpu.CpuBackend.init(alloc, &g);
    const ex = cpu.exec();

    // Reference: `rows` separate single-token steps (what the serial decode
    // path does), each with its own km/delta/out scratch.
    const st_ref = try alloc.alloc(f32, st0.len);
    defer alloc.free(st_ref);
    @memcpy(st_ref, st0);
    const km = try alloc.alloc(f32, heads * vd);
    defer alloc.free(km);
    const dl = try alloc.alloc(f32, heads * vd);
    defer alloc.free(dl);
    const o = try alloc.alloc(f32, heads * vd);
    defer alloc.free(o);
    for (0..rows) |t| {
        for (0..heads) |h| {
            const kv = h % nk;
            gdnFused(
                st_ref[h * kd * vd ..][0 .. kd * vd],
                kb[t * nk * kd + kv * kd ..][0..kd],
                vb[t * heads * vd + h * vd ..][0..vd],
                qb[t * nk * kd + kv * kd ..][0..kd],
                km[h * vd ..][0..vd],
                dl[h * vd ..][0..vd],
                o[h * vd ..][0..vd],
                gb[t * heads + h],
                bb[t * heads + h],
                vd,
            );
        }
    }

    // Candidate: one chunk call over `rows`, from the same starting state.
    const st_chunk = try alloc.alloc(f32, st0.len);
    defer alloc.free(st_chunk);
    @memcpy(st_chunk, st0);
    const kmc = try alloc.alloc(f32, rows * heads * vd);
    defer alloc.free(kmc);
    const dlc = try alloc.alloc(f32, rows * heads * vd);
    defer alloc.free(dlc);
    const oc = try alloc.alloc(f32, rows * heads * vd);
    defer alloc.free(oc);
    ex.gdnStepRowsChunk(
        Buf.wrap(st_chunk),
        Buf.wrap(kb),
        Buf.wrap(vb),
        Buf.wrap(qb),
        Buf.wrap(kmc),
        Buf.wrap(dlc),
        Buf.wrap(oc),
        Buf.wrap(gb),
        Buf.wrap(bb),
        heads,
        nk,
        kd,
        vd,
        rows,
        nk * kd,
        heads * vd,
        nk * kd,
    );

    // 1. the carried state must match exactly -- loose equality would hide
    //    precisely the kind of mistake this guards (a wrong row offset).
    for (st_ref, st_chunk) |a, b| try std.testing.expectApproxEqAbs(a, b, 1e-6);
    // 2. and so must each row's delta, which is the only per-row quantity the
    //    reference keeps: re-run the reference one row at a time and compare
    //    against the chunk's `dlc[t]`.
    const st_walk = try alloc.alloc(f32, st0.len);
    defer alloc.free(st_walk);
    @memcpy(st_walk, st0);
    for (0..rows) |t| {
        for (0..heads) |h| {
            const kv = h % nk;
            gdnFused(
                st_walk[h * kd * vd ..][0 .. kd * vd],
                kb[t * nk * kd + kv * kd ..][0..kd],
                vb[t * heads * vd + h * vd ..][0..vd],
                qb[t * nk * kd + kv * kd ..][0..kd],
                km[h * vd ..][0..vd],
                dl[h * vd ..][0..vd],
                o[h * vd ..][0..vd],
                gb[t * heads + h],
                bb[t * heads + h],
                vd,
            );
            for (0..vd) |j| {
                try std.testing.expectApproxEqAbs(dl[h * vd + j], dlc[t * heads * vd + h * vd + j], 1e-6);
            }
        }
    }
}

test "gdn prelude: chunk(rows=r) == r single-token steps, conv history included" {
    // The other half of the rollback's premise.  Re-stepping the state is only
    // enough if the CONV HISTORY also reflects exactly `rows` rows: the chunk
    // prelude advances it over the whole verify block, so the rollback has to
    // re-run the prelude for the accepted prefix from the pre-verify snapshot.
    //
    // Missing this is what made an earlier attempt land at 2.8% acceptance, and
    // it is invisible to the state-only test above -- hence a separate test that
    // compares the conv history, the rewritten ab, and the qkv rows.
    const alloc = std.testing.allocator;
    // The reference prelude is written with the elementwise primitives the CPU
    // backend itself uses (`backend_cpu.vGdnPrefillCore` spells the same
    // per-token sequence), imported HERE rather than at module scope: the
    // engine must not depend on a backend module (AGENTS.md §1.1).
    const ew = @import("../elementwise.zig");
    const nv = 2;
    const nk = 2;
    const kd = 8;
    const conv_dim = 2 * nk * kd + nv * kd;
    const kq_dim = nk * kd;
    const conv_k = 4;
    const rows = 5;

    const qkv = try alloc.alloc(f32, rows * conv_dim);
    defer alloc.free(qkv);
    const ab = try alloc.alloc(f32, rows * 2 * nv);
    defer alloc.free(ab);
    const wconv = try alloc.alloc(f32, conv_dim * conv_k);
    defer alloc.free(wconv);
    const ssm_a = try alloc.alloc(f32, nv);
    defer alloc.free(ssm_a);
    const dt_bias = try alloc.alloc(f32, nv);
    defer alloc.free(dt_bias);
    var prng = std.Random.DefaultPrng.init(0x1234_5678_9ABC_DEF0);
    const rnd = prng.random();
    for (qkv) |*x| x.* = rnd.float(f32) * 2 - 1;
    for (ab) |*x| x.* = rnd.float(f32) * 2 - 1;
    for (wconv) |*x| x.* = (rnd.float(f32) - 0.5) * 0.25;
    for (ssm_a) |*x| x.* = -0.5 - rnd.float(f32);
    for (dt_bias) |*x| x.* = rnd.float(f32) * 0.1;

    var g: gguf.File = undefined;
    var cpu = backend_cpu.CpuBackend.init(alloc, &g);
    const ex = cpu.exec();

    // Reference: one prelude step per token, sharing ONE conv history -- exactly
    // what the serial decode path does.
    const conv_ref = try alloc.alloc(f32, conv_dim * (conv_k - 1));
    defer alloc.free(conv_ref);
    @memset(conv_ref, 0);
    const qkv_ref = try alloc.alloc(f32, rows * conv_dim);
    defer alloc.free(qkv_ref);
    @memcpy(qkv_ref, qkv);
    const ab_ref = try alloc.alloc(f32, rows * 2 * nv);
    defer alloc.free(ab_ref);
    @memcpy(ab_ref, ab);
    const gam_ref = try alloc.alloc(f32, rows * nv);
    defer alloc.free(gam_ref);
    const qscale: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(kd)));
    for (0..rows) |t| {
        const y = Buf.wrap(qkv_ref[t * conv_dim ..][0..conv_dim]);
        ew.bufConvStep(Buf.wrap(conv_ref), y, Buf.wrap(wconv), conv_k);
        ew.bufGdnGamma(
            Buf.wrap(ab_ref[0 * nv + t * nv ..][0..nv]),
            Buf.wrap(ab_ref[rows * nv + t * nv ..][0..nv]),
            Buf.wrap(ssm_a),
            Buf.wrap(dt_bias),
            Buf.wrap(gam_ref[t * nv ..][0..nv]),
            nv,
        );
        var h: usize = 0;
        while (h < nk) : (h += 1) ew.bufL2norm(y.span(h * kd, kd));
        h = 0;
        while (h < nk) : (h += 1) ew.bufL2norm(y.span(kq_dim + h * kd, kd));
        var j: usize = 0;
        while (j < kq_dim) : (j += kd) ew.bufScaleVec(y.span(j, kd), qscale);
    }

    // Candidate: ONE prelude call over `rows`, from the same zero history.
    const conv_c = try alloc.alloc(f32, conv_dim * (conv_k - 1));
    defer alloc.free(conv_c);
    @memset(conv_c, 0);
    const qkv_c = try alloc.alloc(f32, rows * conv_dim);
    defer alloc.free(qkv_c);
    @memcpy(qkv_c, qkv);
    const ab_c = try alloc.alloc(f32, rows * 2 * nv);
    defer alloc.free(ab_c);
    @memcpy(ab_c, ab);
    const gam_c = try alloc.alloc(f32, rows * nv);
    defer alloc.free(gam_c);
    ex.gdnPrefillCore(
        Buf.wrap(qkv_c),
        Buf.wrap(conv_c),
        Buf.wrap(wconv),
        Buf.wrap(ab_c),
        Buf.wrap(ssm_a),
        Buf.wrap(dt_bias),
        Buf.wrap(gam_c),
        rows,
        conv_dim,
        0,
        rows * nv,
        nv,
        nv,
        kq_dim,
        nv,
        nk,
        kd,
        conv_k,
        1e-6,
    );

    // The conv history is the part a state-only rollback cannot restore.
    for (conv_ref, conv_c) |a, b| try std.testing.expectApproxEqAbs(a, b, 1e-6);
    // The convolved+silu'd rows, which the state kernel then consumes.
    for (qkv_ref, qkv_c) |a, b| try std.testing.expectApproxEqAbs(a, b, 1e-6);
    // beta, rewritten in place, and the decay scalars.
    for (ab_ref, ab_c) |a, b| try std.testing.expectApproxEqAbs(a, b, 1e-6);
    for (gam_ref, gam_c) |a, b| try std.testing.expectApproxEqAbs(a, b, 1e-6);
}

test "gdn rollback: chunk prelude output + state kernel over rows<k == serial" {
    // THE GAP the two kernel tests left.  "gdn chunk state" pins the state
    // kernel and "gdn prelude" pins the prelude, but the rollback consumes one's
    // OUTPUT through the other, and THAT composition is what was wrong: on a
    // device run, the rebuilt state differed from the replayed state on ~92% of
    // elements while the conv history matched on 100% -- i.e. the prelude was
    // right and the state kernel was being fed the wrong slices.
    //
    // This test reproduces the exact engine call shape: one prelude over
    // `k` tokens into a buffer laid out `[q | k | v]` per row, then the state
    // kernel over the FIRST `rows` rows with `stride = conv_dim`, compared
    // against `rows` serial single-token steps that used the same prelude rows.
    const alloc = std.testing.allocator;
    const nv = 3;
    const nk = 2;
    // vd must be a multiple of the GDN kernel's vector width (16).
    const kd = 16;
    const vd = 16;
    const conv_k = 4;
    const conv_dim = 2 * nk * kd + nv * vd;
    const kq_dim = nk * kd;
    const k = 5;
    const rows = 3;

    const raw = try alloc.alloc(f32, k * conv_dim);
    defer alloc.free(raw);
    const ab = try alloc.alloc(f32, k * 2 * nv);
    defer alloc.free(ab);
    const wconv = try alloc.alloc(f32, conv_dim * conv_k);
    defer alloc.free(wconv);
    const ssm_a = try alloc.alloc(f32, nv);
    defer alloc.free(ssm_a);
    const dt_bias = try alloc.alloc(f32, nv);
    defer alloc.free(dt_bias);
    var prng = std.Random.DefaultPrng.init(0xC0FFEE_1234);
    const rnd = prng.random();
    for (raw) |*x| x.* = rnd.float(f32) * 2 - 1;
    for (ab) |*x| x.* = rnd.float(f32) * 2 - 1;
    for (wconv) |*x| x.* = (rnd.float(f32) - 0.5) * 0.25;
    for (ssm_a) |*x| x.* = -0.5 - rnd.float(f32);
    for (dt_bias) |*x| x.* = rnd.float(f32) * 0.1;

    var g: gguf.File = undefined;
    var cpu = backend_cpu.CpuBackend.init(alloc, &g);
    const ex = cpu.exec();

    // Shared prelude over all `k` rows, from a zero conv history.  This is the
    // chunk path the engine uses; its output buffer is REUSED by the state call.
    const qkv_chunk = try alloc.alloc(f32, k * conv_dim);
    defer alloc.free(qkv_chunk);
    @memcpy(qkv_chunk, raw);
    const ab_chunk = try alloc.alloc(f32, k * 2 * nv);
    defer alloc.free(ab_chunk);
    @memcpy(ab_chunk, ab);
    const gam_chunk = try alloc.alloc(f32, k * nv);
    defer alloc.free(gam_chunk);
    const conv_ref = try alloc.alloc(f32, conv_dim * (conv_k - 1));
    defer alloc.free(conv_ref);
    @memset(conv_ref, 0);
    ex.gdnPrefillCore(
        Buf.wrap(qkv_chunk),
        Buf.wrap(conv_ref),
        Buf.wrap(wconv),
        Buf.wrap(ab_chunk),
        Buf.wrap(ssm_a),
        Buf.wrap(dt_bias),
        Buf.wrap(gam_chunk),
        k,
        conv_dim,
        0,
        k * nv,
        nv,
        nv,
        kq_dim,
        nv,
        nk,
        kd,
        conv_k,
        1e-6,
    );

    const st0 = try alloc.alloc(f32, nv * kd * vd);
    defer alloc.free(st0);
    for (st0) |*x| x.* = (rnd.float(f32) - 0.5) * 0.1;

    // A: the state kernel over the FIRST `rows` rows of that buffer, using the
    // engine's slices and `stride = conv_dim` (what `specRebuildAccepted` does).
    const st_a = try alloc.alloc(f32, st0.len);
    defer alloc.free(st_a);
    @memcpy(st_a, st0);
    const km = try alloc.alloc(f32, rows * nv * vd);
    defer alloc.free(km);
    const dl = try alloc.alloc(f32, rows * nv * vd);
    defer alloc.free(dl);
    const out = try alloc.alloc(f32, rows * nv * vd);
    defer alloc.free(out);
    const qkv_view = Buf.wrap(qkv_chunk);
    ex.gdnStepRowsChunk(
        Buf.wrap(st_a),
        qkv_view.span(kq_dim, rows * conv_dim - kq_dim),
        qkv_view.span(2 * kq_dim, rows * conv_dim - 2 * kq_dim),
        qkv_view.span(0, rows * conv_dim),
        Buf.wrap(km),
        Buf.wrap(dl),
        Buf.wrap(out),
        Buf.wrap(gam_chunk[0 .. rows * nv]),
        Buf.wrap(ab_chunk[k * nv ..][0 .. rows * nv]),
        nv,
        nk,
        kd,
        vd,
        rows,
        conv_dim,
        conv_dim,
        conv_dim,
    );

    // B: `rows` serial steps over the SAME prelude rows.
    const st_b = try alloc.alloc(f32, st0.len);
    defer alloc.free(st_b);
    @memcpy(st_b, st0);
    const km1 = try alloc.alloc(f32, nv * vd);
    defer alloc.free(km1);
    const dl1 = try alloc.alloc(f32, nv * vd);
    defer alloc.free(dl1);
    const o1 = try alloc.alloc(f32, nv * vd);
    defer alloc.free(o1);
    for (0..rows) |t| {
        const row = qkv_chunk[t * conv_dim ..][0..conv_dim];
        for (0..nv) |h| {
            // head h reads its q/k from `h % nk` (GQA-style repeat), v from h.
            const kq = h % nk;
            gdnFused(
                st_b[h * kd * vd ..][0 .. kd * vd],
                row[kq_dim + kq * kd ..][0..kd], // k lives after the q half
                row[2 * kq_dim + h * vd ..][0..vd],
                row[kq * kd ..][0..kd], // q is the first half
                
                km1[h * vd ..][0..vd],
                dl1[h * vd ..][0..vd],
                o1[h * vd ..][0..vd],
                gam_chunk[t * nv + h],
                ab_chunk[k * nv + t * nv + h],
                vd,
            );
        }
    }

    for (st_a[0 .. nv * kd * vd], st_b[0 .. nv * kd * vd]) |x, y| {
        try std.testing.expectApproxEqAbs(y, x, 1e-6);
    }
}

test "maxQ8Cols: widest matvec INPUT is n_ff, not conv_dim / 2*q_dim" {
    // 27B (read from the GGUF metadata): n_ff=17408 > conv_dim=10240 >
    // q_dim=value_dim=6144 > hidden=5120
    const c27 = Config{
        .hidden = 5120,
        .n_heads = 24,
        .head_dim = 256,
        .n_ff = 17408,
        .gdn_n_heads = 48, // ssm.inner_size 6144 / state_size 128
        .gdn_head_dim = 128,
        .gdn_num_k_heads = 16, // ssm.group_count
    };
    // the old (buggy) sizing: conv_dim = 2*16*128 + 48*128 = 10240
    const conv_dim27 = 2 * @as(usize, c27.gdn_num_k_heads) * c27.gdn_head_dim +
        @as(usize, c27.gdn_n_heads) * c27.gdn_head_dim;
    try std.testing.expectEqual(@as(usize, 10240), conv_dim27);
    try std.testing.expect(maxQ8Cols(c27) > conv_dim27); // guard: n_ff must win
    try std.testing.expectEqual(@as(usize, 17408), maxQ8Cols(c27));

    // 0.8B defaults: n_ff=3584 is the widest INPUT (conv_dim=6144 is an
    // OUTPUT width of attn_qkv, whose input is hidden=1024 -- sizing the
    // scratch to conv_dim over-allocates here and under-allocates on 27B).
    const c08 = Config{};
    try std.testing.expectEqual(@as(usize, 3584), maxQ8Cols(c08));
    const conv_dim08 = 2 * @as(usize, c08.gdn_num_k_heads) * c08.gdn_head_dim +
        @as(usize, c08.gdn_n_heads) * c08.gdn_head_dim;
    try std.testing.expectEqual(@as(usize, 6144), conv_dim08);
    try std.testing.expect(conv_dim08 > maxQ8Cols(c08)); // old sizing: too big
}

test "checkQ8Scratch: rejects a scratch too small for the widest input" {
    const alloc = std.testing.allocator;
    const need = q8ScratchBytes(17408);
    const good = try alloc.alloc(u8, need);
    defer alloc.free(good);
    try checkQ8Scratch(good, 17408);
    // the old conv_dim-based sizing (10240 cols) is 2448 bytes short
    const bad = try alloc.alloc(u8, q8ScratchBytes(10240));
    defer alloc.free(bad);
    try std.testing.expectError(error.Q8ScratchTooSmall, checkQ8Scratch(bad, 17408));
    // exact fit is accepted, one byte short is not
    try std.testing.expectError(error.Q8ScratchTooSmall, checkQ8Scratch(good[0 .. need - 1], 17408));
}

test "matvecCols: reads input width from Q8_0 / K-IQ / f32 weights" {
    // Q8_0: blocks_per_row * 32 (native 32-element blocks)
    const q8 = Engine.LinearW{ .quant = .{ .data = &.{}, .ttype = .q8_0, .rows = 5120, .blocks_per_row = 544 } };
    try std.testing.expectEqual(@as(usize, 17408), Engine.matvecCols(q8, 5120));
    // K/IQ: blocks_per_row * 256 (native 256-element blocks)
    const q = Engine.LinearW{ .quant = .{ .data = &.{}, .ttype = .q4_k, .rows = 5120, .blocks_per_row = 68 } };
    try std.testing.expectEqual(@as(usize, 17408), Engine.matvecCols(q, 5120));
    // f32: flat [rows, cols] -> len / rows (small shape, 4 rows x 256 cols)
    var f32buf: [4 * 256]f32 = undefined;
    const f = Engine.LinearW{ .f32 = &f32buf };
    try std.testing.expectEqual(@as(usize, 256), Engine.matvecCols(f, 4));
}
