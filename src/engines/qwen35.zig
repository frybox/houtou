//! Qwen3.5 推理引擎（架构层：不碰任何硬件细节）。
//!
//! 这里的代码只描述 Qwen3.5 的模型结构与执行顺序；每个算子经 `backend.Exec`
//! 落到具体后端（CPU 标量/ SIMD，或 CUDA）。权重以 GGUF 原生块格式零拷贝常驻
//! （mmap），激活按需量化，逐块点积在 i32 上累加；norm 与 elementwise 保持 f32。
//!
//! 数值规范以 ref/docs/SPEC.md（权威）与 ref/docs/GDN_IMPL.md 为准；控制流
//! 对齐 ref/c/qwen35_engine.c。GGUF 相关的差异：
//!   - GGUF 张量布局与 C 参照的 safetensors 不同（见简报）：
//!     线性层是 attn_qkv [hidden, nk*kd|nk*kd|nv*kd] = [Q|K|V]，
//!     attn_gate [hidden, nv*kd]（即 z），ssm_alpha [hidden, nv]，
//!     ssm_beta [hidden, nk]，ssm_a [nv] 已经是 -exp(A_log)（负值，已按
//!     SPEC A7 的 A_log.exp() 数值核对过），ssm_conv1d 是 F32 [4, conv_dim]
//!     的通道优先抽头。
//!     0.8B：hidden=1024，nv=nk=16，kd=128（conv_dim 6144）；4B：hidden=2560，
//!     nv=32，nk=16，kd=128（conv_dim 8192）。
//!   - GGUF 里的 Qwen3_5RMSNorm 权重在转换时已经折入了 (1+w)
//!     （已核对：blk.0.attn_norm[0] == 2.0，q_norm 均值约 1.4），所以直接使用；
//!     带门的 ssm_norm（Qwen3_5RMSNormGated）同样已预折叠，直接使用。
//!   - llama.cpp 的转换脚本（conversion/qwen.py:394-404）为所有
//!     Qwen3_5RMSNorm 家族权重固定加 1，并存储 ssm_a = -A_log.exp()。

const std = @import("std");

/// `$QWEN35_SPEC_FASTROLLBACK=0`：A/B 切回整段重放。快速路径已是默认，
/// 因为已验证它与重放逐位一致。
pub var spec_fast_rollback = true;

/// `$QWEN35_SPEC_RBDBG`：打印快速回滚路径的决策与状态比对。
pub var cudart_debug_rollback = false;

/// `$QWEN35_SPEC_DUMMYALLOC`：**诊断用**。在推测解码的准备阶段多分配一块用完
/// 即弃的设备区域，用来检验 MTP 路径的输出是否依赖于设备分配器的布局（理应不
/// 依赖）。
pub var spec_dummy_alloc = false;
const gguf = @import("../gguf.zig");
const backend = @import("../backend.zig");
const backend_cpu = @import("../backend_cpu.zig");
const sampler = @import("../sampler.zig");
const Io = std.Io;

/// 引擎把后端拥有的激活区域交给每个算子。CPU 路径在这里包装主机切片；设备后端
/// 自己解析基址。
const Buf = backend.Buf;

/// 再导出，好让 CPU 与 CUDA 两个宿主共用同一份采样器实现。
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
    /// B9/B10：[11,11,10] 的 mrope 分段，作用在 32 对频率上（rotary_dim=64）；
    /// 第 4 段为 0，是把维度补到 64 的填充。纯文本时三组网格相同。
    rope_sections: [4]u32 = .{ 11, 11, 10, 0 },
    /// rotary_dim = head_dim * 0.25 = 64（不旋转的尾部是 192）。
    rotary_dim: u32 = 64,
    // GDN
    gdn_n_heads: u32 = 16, // linear_num_value_heads (0.8B: 16, 4B: 32)
    /// k/q 的头数（ssm.group_count）；v 头 h 使用 k 头 h / (nv/nk)。
    /// 0.8B：16（nv == nk），4B：16（nv/nk = 2）。
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
// 标量数学（f32；SPEC C15：所有 norm 都用 f32 计算）
// ---------------------------------------------------------------------------

/// RMSNorm，out = x * rsqrt(mean(x^2) + eps) * w。eps 加在平方和里面（SPEC）。
/// 平方和保持标量（顺序精确）；逐元素的缩放做向量化。
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

/// 对整个切片做逐元素 sigmoid（与标量形式逐位相同）。
fn sigmoidVec(x: []f32) void {
    const one: @Vector(GV, f32) = @splat(1.0);
    const p: [*]f32 = x.ptr;
    for (0..x.len / GV) |b| {
        const o = b * GV;
        vstoreF32(GV, p + o, one / (one + @exp(-vloadF32(GV, p + o))));
    }
}

/// 对整个切片做 out[i] *= silu(g[i])。
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

/// 与 torch 的 F.softplus 默认参数一致：beta=1、threshold=20（SPEC A4.4）。
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

/// FLA 的 l2norm（SPEC A3）：x * rsqrt(sum(x^2) + eps)，eps 加在求和里面。
/// 平方和保持标量（顺序精确）；缩放做向量化。
pub fn l2norm(x: []f32) void {
    var ss: f32 = 0;
    for (x) |v| ss += v * v;
    const inv = 1.0 / @sqrt(ss + 1e-6);
    scaleVec(x, inv);
}

// ---------------------------------------------------------------------------
// GDN delta-rule 的状态 kernel，已用 @Vector 向量化（SPEC A4）
// ---------------------------------------------------------------------------
//
// 状态是 [nv, kd, vd] 行优先，且 kd == vd == 128，所以每一行正好是整数个
// 16 元素 f32 向量。每个 head 的四步操作（decay、S^T k、外积更新、S^T q）都写成
// @Vector(16, f32) 的 load/store；每个输出元素的累加顺序与标量版一致，所以结果
// 逐位相同。load/store 是不对齐的，因为状态缓冲只有 4 字节对齐。

inline fn vloadF32(comptime N: usize, p: [*]const f32) @Vector(N, f32) {
    const ptr: *align(1) const [N]f32 = @ptrCast(p);
    return @bitCast(ptr.*);
}

inline fn vstoreF32(comptime N: usize, p: [*]f32, v: @Vector(N, f32)) void {
    const ptr: *align(1) [N]f32 = @ptrCast(p);
    ptr.* = @bitCast(v);
}

const GV = 16; // f32 lanes per vector used by the GDN kernels

/// 逐元素 x *= s（保持顺序，与标量循环逐位相同）。
fn scaleVec(x: []f32, s: f32) void {
    const sv: @Vector(GV, f32) = @splat(s);
    const p: [*]f32 = x.ptr;
    for (0..x.len / GV) |b| {
        const o = b * GV;
        vstoreF32(GV, p + o, vloadF32(GV, p + o) * sv);
    }
    for (x.len - x.len % GV..x.len) |i| x[i] *= s;
}

/// (1) S *= gamma。
fn gdnDecay(sh: []f32, gamma: f32) void {
    const g: @Vector(GV, f32) = @splat(gamma);
    const p: [*]f32 = sh.ptr;
    for (0..sh.len / GV) |b| {
        const o = b * GV;
        vstoreF32(GV, p + o, vloadF32(GV, p + o) * g);
    }
    for (sh.len - sh.len % GV..sh.len) |i| sh[i] *= gamma;
}

/// (2) km[j] = sum_i sh[i*vd + j] * k[i]。km 必须长为 vd；vd % GV == 0。
/// 就地累加到 km（每个元素的累加顺序与标量循环相同）。
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

/// (3) delta = beta * (v - km)；S += k ⊗ delta（外积）。
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

/// (4) out[j] = sum_i sh[i*vd + j] * q[i]（按行累加，顺序与标量的按列循环
/// 相同）。就地累加到 out。
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

/// 融合的 GDN 状态更新：(1) decay + (2) kv-mem 合成一次状态遍历，随后
/// (3) delta + 外积更新 与 (4) S^T q 合成第二次遍历。语义上与依次调用
/// gdnDecay; gdnKvMem; gdnOuter; gdnStateOut 完全相同，但那个 128x128（2.1 MB）
/// 的状态只遍历两次而不是四次，且 km/out 在向量寄存器里累加，而不是逐行
/// load/store。快速路径要求 vd == 128（所有 Qwen3.5 变体都满足）；否则回退到
/// 那四个独立 kernel。
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

    // 第 1 遍：只读扫描，km += (S[i] * gamma) * k[i]，在寄存器里累加。衰减后的
    // 行**不**写回：第 2 遍会重新算一遍。
    var acc: [W]@Vector(32, f32) = undefined;
    for (0..W) |b| acc[b] = @splat(@as(f32, 0.0));
    for (0..k.len) |i| {
        const row = sp + i * vd;
        const ki: @Vector(32, f32) = @splat(k[i]);
        for (0..W) |b| acc[b] += vloadF32(32, row + b * 32) * ki;
    }
    // km 写出的已经是衰减后的值（S 稍后会被覆盖），所以事后读 km 的调用方看到
    // 的与"先 gdnDecay 再 gdnKvMem"一致。
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

/// 关闭 K-token 的 prefill 分块（QWEN35_NO_BATCH）；此后每个 token 都走单
/// token 路径。
pub var no_batch = false;
/// 打开逐阶段计时报表（QWEN35_PROF）。
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
    /// 由 `load` 在创建 Threaded 实例时设置；在 deinit 里释放。
    owns_io: ?*std.Io.Threaded = null,

    token_embd: LinearW, // [vocab, hidden] row-major (embedding)
    /// 不共享的 lm_head（27B 及以上）；在 Engine 里为 null 表示"共享"，即用
    /// token_embd。GGUF 的 `output.weight` 与 token_embd 是**相同**的
    /// [vocab, hidden] 行优先布局。
    output_weight: ?LinearW,
    output_norm: backend.Buf, // [hidden]
    /// 硬件边界：matvec、权重加载、反量化，以及 kernel 需要的激活 scratch。引擎
    /// 永远不知道背后是哪一份实现。
    exec: backend.Exec,
    /// 当前这次前向正在被录制进（或从中重放）的计算图。由 `forwardLogits` /
    /// `forwardChunk` 设置，它们是唯一的入口：从这里到下一个 `graphEnd` 之间的
    /// 每个算子都读它。直通值（`ptr == null`）让每个算子的行为与"图还不存在时"
    /// 完全一致，CPU 后端与捕获失败的情况用的就是它。
    ///
    /// 做成字段而不是逐层传参：层内的辅助函数有 15 层深，否则它们都得为了往下
    /// 传这一个参数而多一个形参。
    ///
    /// 初值是**直通**图（`ptr == null`），但控制块不空：`embedTo` 在图活跃时
    /// 从 `ctl.tok[0]` 取行号（见 `VTable.dequantRow`），所以任何调用
    /// `embedTo` 的路径都必须先把当步的 `pos`/`tok` 写进来 —— `step` 由
    /// `graphBegin` 填，其余单行入口用 `directCtl`。`kind`/`slot` 与 `step`
    /// 用的一致，CPU 后端因此不需要分叉。
    graph: backend.Graph = .{ .kind = .decode, .slot = graph_slot_decode },
    /// `step` 产出的 token。在图的范围之内写出，好让后端能在设备上采样
    /// （`Exec.sample`），随后由 `step` 读取。
    sampled: u32 = 0,
    layers: []LayerWeights,
    kv: []KvCache,
    gdn: []GdnState,
    /// 推测解码用的 verify 前快照；在 `specSnapshotAlloc` 跑过之前是空的。
    spec_snap: []SnapLayer = &.{},
    /// 尾部的 MTP / nextn 预测块（`blk.<n_layers>.*`）；模型没有这个块时
    /// （0.8B/2B/4B）为 null，`spec` 从未初始化时也为 null。
    ///
    /// 推测解码（第 1 阶段）：nextn 块是一个完整的 decoder 层，外加一个
    /// `eh_proj` 把 (embedding, hidden) 折回 `hidden`，它共享主模型的
    /// `output_norm` / lm_head。这是最便宜的一种草稿头 —— 权重随目标模型一起
    /// 发布，没有东西要下载，一次 forward 给出一个投机 token，而不是 DSpine
    /// 那类块级草稿头一次提 15 个。
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
    /// 每个 prefill 块批量处理的 token 数（K-token 批量投影：每层的权重每块只读
    /// 一次，然后经缓存供 K 个 token 复用）。0 表示批量路径不可用。
    nbatch: usize = 0,
    /// 批量 prefill 路径的 K 槽 scratch。`xK[t*hidden..]` 是当前块第 t 个 token
    /// 的残差流；`x1K[t*hidden..]` 是 RMS norm 之后的输入；`qkvK/gateK/abK/`
    /// `qK/kK/vK` 是批量投影的输出；`goutK/x2K` 是逐 token 的 attention/GDN 输出
    /// 与 out-projection 结果；`mlpK/mlpK1` 是 ffn 中间量；`q8kK/q8K` 是批量
    /// matvec 用的逐 token 量化激活。
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
    /// `spec_verify_rows * vocab` 个 f32：批量推测验证（`forwardChunkVerify`）时
    /// 每一行的 lm_head 输出。单行的 `logits` 缓冲装不下一整块。
    logitsK: backend.Buf = backend.Buf.empty,
    /// top-k 的次序统计工作区（quickselect）；尺寸为 vocab。
    sample_scratch: backend.Buf = backend.Buf.empty,
    /// 施加到每个产出 token 上的采样策略。默认是 greedy（temperature 0），也就是
    /// 与历史行为完全一致；CLI 由 -t/-k/-s 设置它。
    sampler: Sampler = Sampler.init(.{}),
    prefill_ms: f64 = 0,

    // ---- 推测解码的抽头（DSpine 移植，第 0 阶段）----
    //
    // DSpine 草稿头只消费目标模型在**固定**一组层索引上的 hidden state
    // （`dflash_config.target_layers`，27B 是 `[2,17,32,47,62]`），其它层一概不
    // 看。于是 `forwardChunk` 可以把这些层的**输出** hidden state 拷进
    // `tap_hidden`，草稿头的 `fc` 投影再把它变成 5H -> H 的上下文。
    //
    // 布局：`tap_hidden` 是 `taps.len * nbatch * hidden`，即每个抽头一片
    // `nbatch * hidden`，抽头优先。抽头 `i` 对应层 `tap_layers[i]`，片内第 `t` 行
    // 就是块内第 `t` 个 token —— 与 `dspine_mlx` 的 `_LayerHook` 一致：它存的是该
    // 层**整张**输出张量（`mx.concatenate(storage, axis=-1)` 保留每个位置；草稿
    // 头需要块内所有行的特征，不只是最后一行）。用 `nbatch` 而不是 `k` 做步长，
    // 是为了让较短的末块仍落在固定偏移上，读者不必知道 `k`。
    //
    // 默认关闭：`tap_hidden` 是 `Buf.empty`，所以正常的 prefill/decode 路径每层
    // 只付一次 usize 比较，别的什么也不付。
    //
    // 顺序：`tap_layers` 必须**升序**（它就是草稿头的 `target_layers`）；抽头 i
    // 从层 `tap_layers[i]` 写出。
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

    /// 图槽号：**由引擎指定**，后端只把它记进 `Graph.slot`。三种 kind 必须在
    /// 不同槽上（节点集合不同，共用会让每步都失配重捕），而槽号走句柄而不是
    /// 后端的"当前槽"，嵌套 span（草稿块在 verify 里面）才能各自正确收尾。
    pub const graph_slot_decode: u8 = 0;
    pub const graph_slot_chunk: u8 = 1;
    pub const graph_slot_nextn: u8 = 2;

    /// `forwardLogits` 在**没有 span** 时用的控制块。
    ///
    /// 嵌入的行号在图活跃时取自 `ctl.tok[0]`（见 `embedTo`）。prefill 的尾巴
    /// 块、自检与调试路径直接调 `forwardLogits`，手里没有后端发出的句柄，
    /// 但位置是已知的主机标量，所以在这里拼一个直通控制块 —— 它是**值**，
    /// CPU 后端直接读，CUDA 的直通路径不发图也不发布它，行为与改动前相同。
    fn directCtl(self: *Engine, pos: u32, token: u32, rows: u32) backend.GraphCtl {
        var tt: [backend.ctl_tokens]u32 = @splat(0);
        tt[0] = token;
        return .{
            .pos = pos,
            .rope_base_bits = @bitCast(self.cfg.rope_freq_base),
            .kv_slot = pos,
            .attn_slot = pos,
            .rows = rows,
            .tok = tt,
        };
    }

    /// GDN 状态快照槽数 K。一次 verify 块最多 `spec_verify_rows` 行，接受
    /// `rows <= verify_rows - 1` 行，而槽 `rows-1` 是"跑完第 rows 行之后"，
    /// 所以需要 `spec_verify_rows - 1` 个 快照块。取 16 覆盖到 verify 的
    /// 全部行数，与 llama.cpp 的 `n_rs_seq = draft.n_max` 同一个量级。
    pub const spec_rs_seq: usize = spec_verify_rows;

    /// Slabs in `gdn[i].state_all`, in order:
    ///
    ///   [0]                  the live state every forward pass reads and writes
    ///   [1 .. 1+spec_rs_seq] the per-row verify snapshots (slot t)
    ///
    /// The live state ALWAYS lives in slab 0.  A rollback used to re-point
    /// `gdn[i].state` at slab `rows` and leave it there; the next verify then
    /// handed that slab both as the running state AND as the base of the
    /// snapshot region (which starts at slab 1), so the CPU fallback's
    /// `@memcpy(snap[..], state[..])` aliased and panicked (`@memcpy arguments
    /// alias`, measured on 27B IQ3_XXS + MTP).  Keeping the snapshot region
    /// disjoint from slab 0 is what makes the two views independent.
    ///
    /// The cost of the fix is one slab copy per rollback (`copyRange`), i.e. a
    /// 3 MB D2D on the 27B -- the rollback path already copies the conv history
    /// and the prelude inputs, so this does not change its order of magnitude.
    pub const snap_slab_base: usize = 1;

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
                exec.zeroBuf(e.graph, e.gdn[li].state_all);
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
    /// **不开 span**：`step` 才是 `.decode` 的开启点。这个函数只读
    /// `self.graph`（默认直通图 = 每个算子照原样发射），于是同一个函数在
    /// "直发"（prefill 的尾巴块）和"被 `step` 的 span 收进去"两种情形下都
    /// 成立。它不做采样 —— 采样器由 `step` 在图内调用。
    pub fn forwardLogits(self: *Engine, token: u32, pos: u32) Buf {
        // `token` 只在这一步的 embedding 里用到，而 embedding 归调用方
        // （见 `step` / prefill 循环）。保留参数是为了签名和调用点一致。
        _ = token;
        const c = self.cfg;
        const io = self.io;
        if (prof) {
            prof_gdn = 0;
            prof_full = 0;
            prof_mlp = 0;
            prof_lm = 0;
            prof_emb = 0;
        }
        // embedding 由 `step` 在开图之前做好（查表在主机上选行，录进图会
        // 永远重读录制那一行）；prefill 的尾巴块也由调用方自己做。
        // 模型带 nextn 块时 `self.layers` 会多出一个槽（见 `nSlots`）；普通的
        // 前向必须停在 decoder 层，否则每个 token 都会把草稿头当作第 64 层再跑
        // 一遍。
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

            // 推测解码抽头的单 token 形式：用平面的第 0 行。在 MLP 的残差相加之后
            // 写出 —— 那正是该层的输出，也是 `forwardChunk` 拷的值（`tap_hidden`
            // 的布局）。
            //
            // 正是这一点让每一轮推测可以复用**目标模型**在普通 decode / 验证时
            // 已经算出来的 hidden state，而不必每轮把它的上下文窗口重新前向一遍。
            // 按 16 行算，那次重前向每轮要付 16 倍的目标逐 token 工作量；在它存在
            // 之前，32 个 token 实测耗时 651 秒。
            if (self.tap_hidden.len != 0) {
                for (self.tap_layers, 0..) |tl, ti| {
                    if (tl == li) {
                        self.exec.copyRange(self.graph, 
                            self.tap_hidden.span(ti * self.nbatch * self.cfg.hidden, self.cfg.hidden),
                            self.x.span(0, self.cfg.hidden),
                        );
                        break;
                    }
                }
            }
        }
        const b = Io.Clock.Timestamp.now(io, .awake);

        self.exec.rmsnorm(self.graph, self.x1, self.x, self.output_norm, c.rms_eps);
        // lm_head：有独立的 `output.weight` 就用它，否则用共享的 token_embd。
        // 两者都是 [vocab, hidden] 行优先。
        if (self.output_weight) |ow| {
            switch (ow) {
                .quant, .f32 => self.exec.matvec(self.graph, self.logits, ow, self.x1),
            }
        } else {
            // 共享 lm_head：logits = W_emb @ x1，W_emb 为 [vocab, hidden] 行优先
            switch (self.token_embd) {
                .quant => |qrow| self.exec.matvec(self.graph, self.logits, .{ .quant = qrow }, self.x1),
                .f32 => |m| self.exec.matvec(self.graph, self.logits, .{ .f32 = m }, self.x1),
            }
        }
        if (prof) prof_lm += profNs(io, b);
        return self.logits;
    }

    pub fn profReport() void {
        const tot: f64 = prof_gdn + prof_full + prof_mlp + prof_lm;
        std.debug.print("PROF emb={d:.2}ms gdn={d:.2}ms full={d:.2}ms mlp={d:.2}ms lm={d:.2}ms total={d:.2}\n", .{ prof_emb / 1e6, prof_gdn / 1e6, prof_full / 1e6, prof_mlp / 1e6, prof_lm / 1e6, tot / 1e6 });
    }

    /// 前向 + 采样一个 token。采样器的 temperature 为 0 时就是 greedy（argmax）
    /// —— 历史默认行为。
    ///
    /// **`.decode` span 的唯一开启点。** `forwardLogits` 只读 `self.graph`，
    /// 采样器由这里在 span 内调用（图的最后一个节点），读回 token 的那次 D2H
    /// 留在 `graphEnd` 之后。
    pub fn step(self: *Engine, token: u32, pos: u32) u32 {
        const c = self.cfg;
        // 一次 decode 步骤 —— 从 embedding 查表到 logits —— 是整个引擎里唯一
        // 一段"kernel 在跑之前就已知"的区间：权重已常驻、每个形状固定，只有
        // `pos`/`slot` 在动。所以设备后端可以录一次、之后重放，而它需要被告知
        // 这段区间的起止 —— 它看不到模型的控制流，只看得到单个算子。
        //
        // embedding 是 span **内**的第一个 op：本步的 token 先写进控制块的
        // `tok[0]`，后端在设备上按它选行（见 `embedTo`），于是节点实参全是
        // 模型常量，重放时嵌的是当步的 token。行号若留在主机侧当实参，录进图
        // 之后会永远重读录制那一行。
        self.graph = self.exec.graphBegin(.decode, .{
            .pos = pos,
            .rope_base_bits = @bitCast(c.rope_freq_base),
            .kv_slot = pos,
            .attn_slot = pos,
            .rows = 1,
            .tok = blk: {
                var tt: [backend.ctl_tokens]u32 = @splat(0);
                tt[0] = token;
                break :blk tt;
            },
        }, graph_slot_decode);
        self.embedTo(token, self.x.span(0, c.hidden));
        // 采样器在区间**之内**运行（图的最后一个节点）；对 CPU 后端来说它只是
        // 一次普通调用。读回 token 的那次 D2H 必须在 `graphEnd` **之后**：
        // capture 期间不能拷贝，录进图又会让每一 token 都读回录制时的那个 id。
        _ = self.forwardLogits(token, pos);
        self.sampled = self.exec.sample(self.graph, self.logits, self.sample_scratch, &self.sampler);
        self.exec.graphEnd(&self.graph);
        // 收尾后回到直通控制块：`forwardLogits` 是公开入口，其它调用点（prefill
        // 尾巴、自检）靠它拿到当步的 `tok`/`pos`，而不是一个清零的结构。
        self.graph = .{ .kind = .decode, .slot = graph_slot_decode, .ctl = self.directCtl(pos, token, 1) };
        return self.exec.sampledToken();
    }

    /// 单 token 前向，**自带 embedding**。给调试/自检路径用（它们只想要
    /// logits，不关心 span）。它自己拼一个直通控制块，因为这里没有后端发出的
    /// 句柄 —— 见 `directCtl`。
    pub fn forwardTokenEmbedded(self: *Engine, token: u32, pos: u32) Buf {
        self.graph = .{ .kind = .decode, .slot = graph_slot_decode, .ctl = self.directCtl(pos, token, 1) };
        self.embedTo(token, self.x.span(0, self.cfg.hidden));
        return self.forwardLogits(token, pos);
    }

    /// 把 `token` 的 embedding 行反量化进 `dst`（宽度为 hidden）。
    ///
    /// 调用方必须**先** `graphBegin`（把 `token` 写进 `ctl.tok[0]`）再调它：
    /// 图活跃时行号由后端从设备控制块读（见 `VTable.dequantRow`），重放时嵌的
    /// 是当步的 token，而不是录制那一轮的行。直通图（prefill、CPU）用实参。
    fn embedTo(self: *Engine, token: u32, dst: Buf) void {
        const hidden: usize = self.cfg.hidden;
        std.debug.assert(dst.len == hidden);
        // 把一行展开成 f32 是模型里唯一需要**具体数值**（而不是点积）的地方，
        // 所以它是一次后端调用：用哪个反量化器（CPU 上是 Zig 参照、GPU 上是设备
        // kernel）是后端的事。Q8_0 的 32 元素分帧也在那里处理，不在这里特判。
        switch (self.token_embd) {
            .f32 => |m| self.exec.copyRange(self.graph, dst, Buf.wrap(m[token * hidden ..][0..hidden])),
            .quant => |qrow| self.exec.dequantRow(self.graph, dst, qrow, token),
        }
    }

    /// 把 `tokens` 的 embedding 行展开进 `xK` 的前 `k` 行。
    ///
    /// **必须在开 capture 之前调用。** capture 期间 op 不执行，`dequantRow`
    /// 会被录成节点而 `xK` 停在上一轮的内容，于是整块前向算在旧输入上。
    /// `forwardLogits` 的 anchor 也是同样的道理（放在 embedding 之后）。
    fn embedChunk(self: *Engine, tokens: []const u32) void {
        const hidden: usize = self.cfg.hidden;
        // `$QWEN35_EMB_ROWS=1`（A/B）：走多行设备端路径。只对 `tokens.len <=
        // ctl_tokens` 成立（控制块的行号数组只有那么长）；prefill 的 k 可达
        // 1024，那里只能逐行。
        const use_rows = @import("../cudart.zig").environ_map != null and
            @import("../cudart.zig").environ_map.?.get("QWEN35_EMB_ROWS") != null and
            tokens.len <= backend.ctl_tokens;
        if (use_rows) {
            switch (self.token_embd) {
                .quant => |qrow| {
                    var tt: [backend.ctl_tokens]u32 = @splat(0);
                    for (tokens, 0..) |t, i| tt[i] = t;
                    self.exec.dequantRows(.{
                        .kind = self.graph.kind,
                        .ctl = .{ .rows = @intCast(tokens.len), .tok = tt },
                        .slot = self.graph.slot,
                    }, self.xK.span(0, tokens.len * hidden), qrow);
                    return;
                },
                .f32 => {},
            }
        }
        for (tokens, 0..) |t, i| self.embedTo(t, self.xK.span(i * hidden, hidden));
    }

    // -----------------------------------------------------------------
    // K-token 批量 prefill 块
    // -----------------------------------------------------------------
    /// 从位置 `pos0` 开始跑 `tokens.len`（<= K）个 token 的 prefill，返回**最后
    /// 一个** token 的 logits。每个读权重的线性投影（qkv / gate / alpha-beta /
    /// q / k / v / out-proj / ffn gate-up-down）每块只执行**一次** K-token 批量
    /// matvec，于是每层的权重字节为 K 个 token 只过一遍 cache；状态 kernel
    /// （GDN 的 conv + 128x128 状态、full-attention 打分、KV cache 写入）仍按
    /// 完全串行的顺序逐 token 跑，所以 KV cache 与 GDN 状态与串行 prefill 逐位
    /// 相同。
    pub fn forwardChunk(self: *Engine, tokens: []const u32, pos0: u32) Buf {
        const k = tokens.len;
        // `forwardChunk` 与 `forwardLogits` 是**并列**的两种形状：前者 k 行块、
        // 后者单行 decode。二者互不调用 —— 谁被用由调用方（prefill 循环 /
        // `step` / `forwardChunkVerify` / `nextnPredict`）按行数决定。
        //
        // 这里不做 `k == 1` 的转发：让一个函数在内部改换成另一种形状，正是
        // "同一个槽里出现两次 `graphBegin`"那类问题的来源（第二次
        // `StreamBeginCapture` 会被运行时拒绝，第一次开下的 capture 永远没人关）。
        // 断言是唯一的表达方式；调用方必须自己分流。
        std.debug.assert(k > 1 and k <= self.nbatch);

        // 位置**不从这里传**：每一层的位置都来自 `self.graph.ctl` 的设备控制块
        // （`pos` / `kv_slot` / `attn_slot` / `rows`），由开 span 的那一方发布。
        // 块内第 t 行的位置是 `pos + t`，由 kernel 从控制块推导。
        //
        // 这个参数因此没有消费者 —— 它保留有两个原因：调用方（prefill 循环、
        // `forwardChunkVerify`）本来就持有它、传出去更便于阅读；而 `step` /
        // `verify` / `nextn` 三者的 `pos0` 各自不同，签名一致才好对照。
        // 一旦确认没有读者，可以连同调用点一起删掉。
        _ = pos0;

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

        // **不开 span**：`forwardChunk` 有两个调用方（prefill 的块循环与
        // `forwardChunkVerify`），只有后者要录制，而它自己会开 `.verify`。
        // 这里只用 `self.graph` 的当前值（默认直通图 = 每个算子照原样发射），
        // 于是同一个函数在"直发"和"被外层 span 收进去"两种情形下都成立。
        //
        // embedding 在 span **内**做：行号从设备控制块读（`g.ctl.tok[r]`），
        // 实参全是模型常量，所以 capture 期间也成立 —— 重放时嵌入的是当步的
        // token，而不是录制那一轮的行。调用方负责在 `graphBegin` 的 ctl 里填好
        // `tok[0..k]`；直通图上没人读 ctl，后端退回逐行 path。
        self.embedChunk(tokens);

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
            self.exec.rmsnormRows(self.graph, self.x1K.span(0, k * hidden), self.xK.span(0, k * hidden), lw.attn_norm, k, c.rms_eps);

            if (c.isFullAttention(@intCast(li))) {
                self.exec.matvecBatchK(self.graph, self.qkvK.span(0, k * 2 * q_dim), lw.attn_q.?, self.x1K, k);
                self.exec.matvecBatchK(self.graph, self.kK.span(0, k * kv_dim), lw.attn_k.?, self.x1K, k);
                self.exec.matvecBatchK(self.graph, self.vK.span(0, k * kv_dim), lw.attn_v.?, self.x1K, k);
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
                self.forwardFullChunk(lw, li, k);
                self.exec.matvecBatchK(self.graph, self.x2K.span(0, k * hidden), lw.attn_output.?, self.goutK, k);
            } else {
                self.exec.matvecBatchK(self.graph, self.qkvK.span(0, k * conv_dim), lw.attn_qkv.?, self.x1K, k);
                self.exec.matvecBatchK(self.graph, self.gateK.span(0, k * value_dim), lw.attn_gate.?, self.x1K, k);
                self.exec.matvecBatchK(self.graph, self.abK.span(0, k * nv), lw.ssm_alpha.?, self.x1K, k);
                self.exec.matvecBatchK(self.graph, self.abK.span(k * nv, k * nv), lw.ssm_beta.?, self.x1K, k);
                // 为推测解码回滚保留 PRE-prelude 的投影。必须是原始行：
                // 回滚只对 `rows` 个 token 重跑 prelude，而 prelude 会推进卷积
                // 历史并就地重写 `qkv`/`ab`——这两样都无法从 POST-prelude 的
                // 字节恢复（实测：缓存那些字节会让 conv 停在 `k` 深，而 state
                // 已退到 `rows`）。
                if (li < self.gdn_in.len) {
                    const gi = &self.gdn_in[li];
                    if (gi.qkv.len >= k * conv_dim and gi.gamma.len >= k * 2 * nv) {
                        // 长度同样不能是主机实参（见 tap 那段）：图活跃时按
                        // `ctl.rows` 拷，直通时用 `k`。
                        if (self.graph.ptr != null) {
                            self.exec.copyRowsCtl(self.graph, gi.qkv, self.qkvK, conv_dim);
                            self.exec.copyRowsCtl(self.graph, gi.gamma, self.abK, 2 * nv);
                        } else {
                            self.exec.copyRange(self.graph, gi.qkv.span(0, k * conv_dim), self.qkvK.span(0, k * conv_dim));
                            self.exec.copyRange(self.graph, gi.gamma.span(0, k * 2 * nv), self.abK.span(0, k * 2 * nv));
                        }
                    }
                }
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
                self.exec.matvecBatchK(self.graph, self.x2K.span(0, k * hidden), lw.ssm_out.?, self.goutK, k);
            }

            // ---- residual add: the `k` rows are CONSECUTIVE (`xK`/`x2K` are
            // `k * hidden` contiguous), so one call covers the chunk.  This was a
            // per-token loop (one launch per token per layer: 3353 launches for
            // a 525-token prefill, which PROFILE showed at 2.7%).  `addResid` is
            // elementwise with no per-row state, so the merge is exact.
            self.exec.addResid(self.graph, self.xK.span(0, k * hidden), self.x2K.span(0, k * hidden));

            // ---- MLP: chunk-wide norm, batched gate/up, chunk-wide SwiGLU,
            // batched down, residual ----
            self.exec.rmsnormRows(self.graph, self.x1K.span(0, k * hidden), self.xK.span(0, k * hidden), lw.post_attention_norm, k, c.rms_eps);
            self.exec.matvecBatchK(self.graph, self.mlpK.span(0, k * n_ff), lw.ffn_gate, self.x1K, k);
            self.exec.matvecBatchK(self.graph, self.mlpK1.span(0, k * n_ff), lw.ffn_up, self.x1K, k);
            // SwiGLU over the whole chunk: `gate[i] = silu(gate[i]) * up[i]` is
            // elementwise and each token's `n_ff` run is independent, so one
            // launch over `k * n_ff` is the same arithmetic as `k` launches of
            // an `n_ff`-long one.
            self.exec.swiglu(self.graph, self.mlpK.span(0, k * n_ff), self.mlpK1.span(0, k * n_ff));
            self.exec.matvecBatchK(self.graph, self.x2K.span(0, k * hidden), lw.ffn_down, self.mlpK, k);
            self.exec.addResid(self.graph, self.xK.span(0, k * hidden), self.x2K.span(0, k * hidden));

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
                        // 行数走**设备控制块**（`copyRowsCtl`），不是 `k`。
                        // `copyRange` 的长度是主机实参，录进图之后被烘死成录制
                        // 那一轮的 k；k 从 2 变 3 时第三行不写，草稿头读到旧
                        // hidden，接受率塌掉（实测 62.5% → 12.5%）并很快 EOS。
                        // 目标区按 `nbatch` 给上界（它本来就那么大），实际写多少
                        // 由 ctl.rows 决定。
                        // 图活跃时行数取控制块（`copyRowsCtl`），直通时用 `k`
                        // —— 直通图的 ctl 是默认值，`rows` 恒为 1。
                        if (self.graph.ptr != null) {
                            self.exec.copyRowsCtl(self.graph,
                                self.tap_hidden.span(ti * self.nbatch * hidden, self.nbatch * hidden),
                                self.xK.span(0, self.nbatch * hidden),
                                hidden,
                            );
                        } else {
                            self.exec.copyRange(self.graph,
                                self.tap_hidden.span(ti * self.nbatch * hidden, k * hidden),
                                self.xK.span(0, k * hidden),
                            );
                        }
                        break;
                    }
                }
            }
        }

        // ---- last token: final RMSNorm + lm_head on a single token (serial
        // path, so the first sampled logits are bit-identical) ----
        self.exec.rmsnorm(self.graph, self.x1, self.xK.span((k - 1) * hidden, hidden), self.output_norm, c.rms_eps);
        if (self.output_weight) |ow| {
            switch (ow) {
                .quant, .f32 => self.exec.matvec(self.graph, self.logits, ow, self.x1),
            }
        } else {
            switch (self.token_embd) {
                .quant => |qrow| self.exec.matvec(self.graph, self.logits, .{ .quant = qrow }, self.x1),
                .f32 => |m| self.exec.matvec(self.graph, self.logits, .{ .f32 = m }, self.x1),
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
    ///
    /// **这里是 `.verify` span 的唯一开启点。** `forwardChunk` 自己不开 span
    /// （prefill 也调它，而 prefill 一律直发），所以 verify 的行数 k 是这张图
    /// 的一部分 —— 换一个 k 就是另一张图，由后端的槽按 `rows` 校验。
    pub fn forwardChunkVerify(self: *Engine, tokens: []const u32, pos0: u32) Buf {
        const k = tokens.len;
        // k == 1 是**非法输入**，不是一种退化形状：这张图按行数建槽（见
        // `xdPhaseBegin` 的 `xd_rows` 校验），录进一行 decode 的节点集合后，
        // 下一轮真来 k 行就重放不成立。`generateSpec` 只会传 >= 2（`vbuf` 的
        // 首行固定是 anchor，且只有 `n_draft > 0` 才进这一路），其余调用点
        // 也都在任何 span 之外；所以这里断言而不是转发。
        std.debug.assert(k > 1 and k <= self.nbatch and k <= spec_verify_rows);
        const c = self.cfg;
        const hidden: usize = c.hidden;

        // 控制块：`pos` 是这一块第一行的位置，`kv_slot`/`attn_slot` 同它，
        // `rows` 是 k（后端据此判断槽里那张图还成不成立）。
        //
        // `$QWEN35_VERIFY_NOSPAN=1`（调试）：完全不碰图，`self.graph` 保持直通。
        // 用来把"图机制有问题"与"图的内容有问题"分开。
        const nospan = @import("../cudart.zig").environ_map != null and
            @import("../cudart.zig").environ_map.?.get("QWEN35_VERIFY_NOSPAN") != null;
        var tok: [backend.ctl_tokens]u32 = @splat(0);
        for (tokens, 0..) |t, i| tok[i] = t;
        if (!nospan) {
            self.graph = self.exec.graphBegin(.verify, .{
                .pos = pos0,
                .rope_base_bits = @bitCast(c.rope_freq_base),
                .kv_slot = pos0,
                .attn_slot = pos0,
                .rows = @intCast(k),
                .tok = tok,
            }, graph_slot_chunk);
        }
        defer {
            if (!nospan) {
                self.exec.graphEnd(&self.graph);
                // 回到直通句柄并带上本块末行的 ctl：`forwardChunkVerify` 返回后
                // 调用方（`generateSpec` 的 `draft_fn`）还会发直通 op，而它们
                // 读 `ctl.tok[0]`。留 `.{}` 会把行号清零，嵌入变成 token 0
                // —— 实测：MTP 接受率从 56.5% 掉到 41.7%。
                self.graph = .{ .kind = .decode, .slot = graph_slot_decode, .ctl = self.directCtl(pos0 + @as(u32, @intCast(k - 1)), tokens[k - 1], 1) };
            }
        }

        // Layers, exactly as `forwardChunk` runs them.  Row k-1 also gets its
        // lm_head there, which the batched projection below recomputes.
        // 让 GDN 分支写每行的状态快照；`specRollbackAndReplay` 据此只改视图。
        self.in_spec_verify = true;
        defer self.in_spec_verify = false;
        // 这一段落在 `.verify` span 里，但后端默认**不把 `.verify` 录成图**
        // （见 `dGraphBegin` 的说明）。
        _ = self.forwardChunk(tokens, pos0);
        // Normalise every row and project them all through lm_head.
        self.exec.rmsnormRows(self.graph, self.x1K.span(0, k * hidden), self.xK.span(0, k * hidden), self.output_norm, k, c.rms_eps);
        const w = self.output_weight orelse self.token_embd;
        self.exec.matvecBatchK(self.graph, self.logitsK.span(0, k * c.vocab), w, self.x1K.span(0, k * hidden), k);
        return self.logitsK.span(0, k * c.vocab);
    }

    /// Greedy argmax over the last logits.  Forwards to the backend so the logits
    /// never have to be readable from the host (doc comment on `Exec.argmax`);
    /// the engine no longer carries its own copy of this loop.
    pub fn argmax(self: *Engine) u32 {
        return self.exec.argmax(self.graph, self.logits);
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

        self.exec.rmsnorm(self.graph, self.x1, self.x, lw.attn_norm, c.rms_eps);

        // in_proj_qkv [hidden, conv_dim]：[Q|K|V] 连续排布（SPEC A1）；
        // Q、K 宽 nk*kd，V 宽 nv*kd
        const qkv = self.qkv.span(0, conv_dim);
        self.exec.matvec(self.graph, qkv, lw.attn_qkv.?, self.x1);

        // z 投影（attn_gate [hidden, value_dim]）
        const z = self.z.span(0, value_dim);
        self.exec.matvec(self.graph, z, lw.attn_gate.?, self.x1);

        // a/b 投影：ssm_alpha 即 a（衰减），ssm_beta 即 b（beta 的输入），每个
        // **value head** 一个标量（C 参照里 gdn_a/gdn_b/A/dt 都按 gdn_vheads
        // 定尺寸；GGUF 的行在 0.8B/2B/4B 上都是 [hidden, nv]）
        //
        // 用**一次** `matvecPair` 而不是两次 `matvec`。`ssm_alpha` 与
        // `ssm_beta` 都是 [hidden, nv] 同类型，且在 GGUF 里紧邻存放（48 层全都
        // 核对过：`alpha.off + alpha.size == beta.off`），也就是说它们**就是**一张
        // `2*nv x hidden` 的矩阵，而 `ab` 本来就是连续的输出。后端在融合之前会先
        // 证明相邻性，否则退回两次调用。
        //
        // 这里两次发射的开销大于它们算的东西：每个是 48x5120 q8_0 = 255 KB，而
        // 在所有试过的 block 形状下（/tmp/shp/bw2.cu），读 255 KB 要 3.1–4.4 us、
        // 带宽只有 60–85 GB/s —— 那是发射下限，不是带宽。这样的发射每 token 有
        // 96 次（约 0.3 ms，1.6%），减半到 48 次。
        const ab = self.ab.span(0, 2 * nv);
        self.exec.matvecPair(self.graph, ab, lw.ssm_alpha.?, lw.ssm_beta.?, self.x1);

        // 因果 conv1d，左侧补 K-1，之后接 silu（SPEC A2，C 参照 520-556）
        // convStep 对所有通道处理历史 + silu；beta/gamma 在其后基于**原始** a/b
        // 投影计算（不经过 conv）。
        const st = &self.gdn[li];
        self.exec.convStep(self.graph, st.conv, qkv, lw.ssm_conv1d, c.conv_kernel);

        // beta = sigmoid(b)；g = A * softplus(a + dt_bias)，其中 A = ssm_a
        // （GGUF 的 ssm_a 已经是 -exp(A_log)，即负值，已核对）。
        //
        // `beta` **就地**写进 `ab` 的 b 那一半，`gamma` 是本层的 `nv` 个衰减标量。
        // 两者都是后端区域，而且**一直留在那里**：`gdnStepRows` 以 `Buf` 形式接收
        // 它们，设备后端读自己的内存，所以一个 token 完全不会为 gamma/beta 走一趟
        // PCIe（过去是每个 GDN 层 2*nv*4 B 下行加 2*nv*4 B 上行 —— 每层一次
        // `Exec.copyToHost`，现在已彻底离开这条路径）。
        //
        // 这两块区域每层每 token 都换内容，所以绝不能交给按指针做键的"只上传一次"
        // 缓存；见 `VTable.gdnStepRows`。
        const beta = self.ab.span(nv, nv); // b half, overwritten with sigmoid(b)
        const gamma = self.gamma_scratch.span(0, nv);
        self.exec.gdnGamma(self.graph, ab.span(0, nv), beta, lw.ssm_a, lw.ssm_dt_bias, gamma, nv);

        const state = st.state; // [nv, kd, vd] row-major S[k][v]
        const gout = self.gdn_out.span(0, value_dim);
        // 对 q、k 做 L2 normalize（eps 在求和里面），然后给 q 乘 1/sqrt(128)，
        // 一次处理一整段 head，而不是一次一个 head。前提条件是：
        //   * the q heads are `qkv[0..kq_dim]`, the k heads `qkv[kq_dim .. 2*kq_dim]`
        //     and the v heads `qkv[2*kq_dim .. 3*kq_dim]` -- contiguous runs of
        //     `kd`-wide rows;
        //   * 每一行用同样的 eps / 同样的 `qscale`；
        //   * 每一行的归一化互相独立（各自的平方和）。
        // `l2normRows` 就是 `n` 个连续的单行 `k_l2norm` block，`scaleVec` 是逐
        // 元素操作，所以两者在每行上的算术完全相同。
        //
        // **不做** `nv/nk` 次重复。这里替代掉的逐 head 循环会让第 `h % nk` 行每个
        // v head 被访问一次（即每个 k head 3 次），而
        // `l2normRows(span(0, kq_dim), kd)` 只把 `nk` 行各归一化**一次** —— 想复现
        // 那个循环就得重复 3 次。但那是**错误的**运算，不只是表达得慢：参照实现
        // （llama.cpp `llama_model_qwen3next::graph::build_layer_gdn`，
        // src/models/qwen3next.cpp:503-511）是对一个 `(head_k_dim, num_k_heads)`
        // 视图 —— **16** 行 —— 做 `build_gdn_l2_norm`，然后才用 `ggml_repeat_4d`
        // 把 q/k 广播到 `num_v_heads`（qwen3next.cpp:520-528）。把 16 行归一化
        // 3 次与"归一化一次再重复"并不等价：`l2norm` 不是幂等的，第 2、3 遍会把
        // 该行再乘一次 `rsqrt(1 + 1e-6)` ≈ 1-5e-7。这个误差小到不会翻转 greedy
        // 的 token，所以没有测试抓到它 —— 但它既**算错了**又做了 3 倍的活：ncu
        // 实测 `k_l2norm` 在 20 个 token 里 15.68 ms / 7452 次调用（每次 2.1 us
        // 做一次 128 元素的归约），其中 2/3 的调用就是这个重复。
        if (gdnHeadsTiled(nv, nk)) {
            const qscale = 1.0 / @sqrt(@as(f32, @floatFromInt(kd)));
            // 一次发射取代三次（l2normRows q、l2normRows k、scaleVec q）。同样的
            // 算术、同样的顺序 —— CPU
            // backend's `l2normQkScale` literally calls those three, and the
            // device kernel reproduces them per row.  See `backend.Exec`.
            self.exec.l2normQkScale(self.graph, qkv, nk, kd, kq_dim, qscale);
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
                self.exec.l2norm(self.graph, q_h);
                self.exec.l2norm(self.graph, k_h);
                const qscale = 1.0 / @sqrt(@as(f32, @floatFromInt(kd)));
                self.exec.scaleVec(self.graph, q_h, qscale);
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
        self.exec.gdnStepRows(self.graph, 
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
        self.exec.rmsnormRows(self.graph, gout.span(0, value_dim), gout.span(0, value_dim), lw.ssm_norm, nv, c.rms_eps);
        // `mulSilu` is elementwise with no reduction and no per-row state, so the
        // `nv` heads collapse into this one call over the same `nv * vd` run --
        // unlike the norm merges above, there is no repeat-count subtlety here.
        self.exec.mulSilu(self.graph, gout.span(0, value_dim), z.span(0, value_dim));

        // out projection [2048 -> 1024]
        self.exec.matvec(self.graph, self.x2.span(0, hidden), lw.ssm_out.?, gout);
        self.exec.addResid(self.graph, self.x.span(0, hidden), self.x2.span(0, hidden));
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
        self.exec.deinterleaveQGStrided(self.graph, self.qK.span(0, k * q_dim), self.gateK.span(0, k * q_dim),
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
        self.exec.rmsnormRows(self.graph, self.qK.span(0, k * q_dim), self.qK.span(0, k * q_dim),
            lw.attn_q_norm, k * heads, c.rms_eps);
        self.exec.rmsnormRows(self.graph, self.kK.span(0, k * kv_dim), self.kK.span(0, k * kv_dim),
            lw.attn_k_norm, k * kv_heads, c.rms_eps);
        self.exec.rope64Rows(self.graph, self.qK.span(0, k * q_dim), @intCast(slot0), self.cfg.rope_freq_base, k, heads, hd);
        self.exec.rope64Rows(self.graph, self.kK.span(0, k * kv_dim), @intCast(slot0), self.cfg.rope_freq_base, k, kv_heads, hd);
        self.exec.kvAppend(self.graph, cache.k, self.kK.span(0, k * kv_dim), slot0, kv_dim, k);
        self.exec.kvAppend(self.graph, cache.v, self.vK.span(0, k * kv_dim), slot0, kv_dim, k);
        cache.len += @intCast(k);

        // Pass 2: scores -> softmax -> PV -> gate for all k rows, one launch.
        const scale = 1.0 / @sqrt(@as(f32, @floatFromInt(hd)));
        self.exec.attnChunk(self.graph, self.qK.span(0, k * q_dim), self.gateK.span(0, k * q_dim),
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
        self.exec.gdnPrefillCore(self.graph, 
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
            self.exec.gdnStepRowsSnap(self.graph, 
                self.gdn[li].state,
                self.qkvK.span(kq_dim, k * conv_dim - kq_dim),
                self.qkvK.span(2 * kq_dim, k * conv_dim - 2 * kq_dim),
                self.qkvK.span(0, k * conv_dim),
                self.kv_mem,
                self.delta,
                self.goutK.span(0, k * value_dim),
                self.gdn_prefill_gamma.span(0, k * nv),
                self.abK.span(beta_base, k * nv),
                self.gdn[li].state_all.span(snap_slab_base * nv * kd * vd, (spec_rs_seq) * nv * kd * vd),
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
        self.exec.gdnStepRowsChunk(self.graph, 
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
        self.exec.rmsnormRows(self.graph, self.goutK.span(0, k * value_dim), self.goutK.span(0, k * value_dim), lw.ssm_norm, k * nv, c.rms_eps);
        self.exec.mulSilu(self.graph, self.goutK.span(0, k * value_dim), self.gateK.span(0, k * value_dim));
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

        self.exec.rmsnorm(self.graph, self.x1, self.x, lw.attn_norm, c.rms_eps);

        // q_proj [1024, 4096]：先 view(8, 512) 再切块 -> 每头的 [q|gate]
        const qg = self.qkv.span(0, 2 * q_dim);
        self.exec.matvec(self.graph, qg, lw.attn_q.?, self.x1);
        const q = self.q.span(0, q_dim);
        const gate = self.z.span(0, q_dim);
        // 批量拆分（见 `forwardFullCore`）：GGUF 的 `attn_q` 行是
        // `[h][q_head | gate_head]`，而 `q`/`gate` 都按 head 消费，所以这是一次
        // 任何单个 memcpy 都表达不了的布局变换 —— 这个 slot 一次发射就做完，而不是
        // `2*heads` 次。
        self.exec.deinterleaveQG(self.graph, q, gate, qg, heads, hd);
        self.exec.matvec(self.graph, self.k.span(0, kv_dim), lw.attn_k.?, self.x1);
        self.exec.matvec(self.graph, self.v.span(0, kv_dim), lw.attn_v.?, self.x1);

        // q_norm/k_norm 在 RoPE **之前**（B12）；GGUF 权重已预折叠 (1+w)。
        //
        // 所有 head 共用**一次** `rmsnormRows`，而不是每 head 一次 `rmsnorm`。
        // q 的第 h 个 head 占 [h*hd, (h+1)*hd)，所以各 head 是连续的，按行批量的
        // slot 正好就是它替代的那个循环 —— 在 CPU 参照上它**就是**那个循环
        // （`vRmsnormRows`），所以不可能改变任何一个数值。在 CUDA 上
        // `rmsnormRows` 是 capture 安全的（不受图门控），所以 decode 图在这里
        // 只看到一个节点，而不是 `heads` 个。
        self.exec.rmsnormRows(self.graph, q.span(0, heads * hd), q.span(0, heads * hd), lw.attn_q_norm, heads, c.rms_eps);
        self.exec.rmsnormRows(self.graph, self.k.span(0, kv_heads * hd), self.k.span(0, kv_heads * hd), lw.attn_k_norm, kv_heads, c.rms_eps);
        // 对每个 q/k head 的前 64 个分量做 RoPE —— 每个张量**一次**调用，而不是
        // 每 head 一次。`rope64Rows(x, 1, heads, hd)` 正好就是它替代的逐 head
        // 循环（`k_rope64_rows` 让第 r 行在 `pos+r` 处旋转，而这里 rows == 1），
        // 且 CUDA 后端现在有这个 kernel 的设备位置孪生版
        // （`k_rope64_rows_dpos`），其 host-stub 地址能被 decode 图识别，所以
        // 批量发射在捕获下依然成立。
        //
        // 实测（27B IQ3_XXS decode，ncu）：每 head 一次发射意味着每个 full-attn
        // 层 40 + 8 = 48 次 rope 发射、每 token 768 次，而 `k_rope64_dpos` 在 20
        // 个 token 里花了 12.84 ms / 6188 次调用 —— 每次调用 2.07 us 只为算 64 个
        // cos+sin，占全部非 matvec kernel 时间的 27.9%。
        self.exec.rope64Rows(self.graph, q.span(0, heads * hd), pos, c.rope_freq_base, 1, heads, hd);
        self.exec.rope64Rows(self.graph, self.k.span(0, kv_heads * hd), pos, c.rope_freq_base, 1, kv_heads, hd);

        // KV cache：把这个 token 的行量化进 cache（后端自己按它的布局定步长；
        // `slot` 是逐 token 的行号，decode 图不再重定向节点，而是让 kernel 从
        // 设备控制块读它）。
        const cache = &self.kv[li];
        std.debug.assert(cache.len < max_seq);
        const slot: usize = cache.len;
        self.exec.kvAppend(self.graph, cache.k, self.k.span(0, kv_dim), slot, kv_dim, 1);
        self.exec.kvAppend(self.graph, cache.v, self.v.span(0, kv_dim), slot, kv_dim, 1);
        cache.len += 1;

        // attention: kv = h // gqa (B13), scale = hd^-0.5
        const scale = 1.0 / @sqrt(@as(f32, @floatFromInt(hd)));
        const gout = self.gdn_out.span(0, q_dim);
        // scores -> softmax -> PV -> sigmoid gate in one backend call
        self.exec.attnDecode(self.graph, q.span(0, q_dim), gate.span(0, q_dim),
            cache.k, cache.v,
            self.attn.span(0, slot + 1), gout.span(0, q_dim),
            slot, heads, gqa, hd, kv_dim, scale);
        // o_proj [2048 -> 1024]
        self.exec.matvec(self.graph, self.x2.span(0, hidden), lw.attn_output.?, gout);
        self.exec.addResid(self.graph, self.x.span(0, hidden), self.x2.span(0, hidden));
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

        // 草稿块这一趟自己的图。它是**独立的一槽**（`GraphKind.nextn`）：
        // 节点集合与 decode 不同（反量化的是嵌入行、下一层是 nextn 层），
        // 共用会让两边每步都失配重捕。
        //
        // 逐轮变化的量只有嵌入 token（`tok`，设备控制块里读）和位置
        // （`pos`/`kv_slot`/`attn_slot` 三个标量，同样走控制块），所以第一次
        // 之后每轮就是一次 `GraphLaunch`。
        //
        // 草稿块的算子本身 capture 安全：嵌入走 `dequantize_rows_ctl`（行号从
        // 控制块读，`tok[0]` 就是本轮要嵌的 token），其余是 matvec / rmsnorm /
        // attention 的设备位置形态。
        //
        // **串行，不嵌套**：`generateSpec` 先 `draft_fn`（本函数）再
        // `forwardChunkVerify`，两者在时间上分开；另外两个调用点
        // （`main.zig` 的 SPEC-NEXTN 自检）也都在任何 span 之外。所以这里
        // 不需要保存/恢复外层句柄 —— 那正是 `ctl_host` 只按后端存一份、
        // 恢复时漏掉一半状态的来源（实测：草稿槽覆盖 pos 后 verify 的 rope
        // 读到别人的 pos，MTP 接受率 0%）。
        self.graph = self.exec.graphBegin(.nextn, .{
            .pos = draft_pos,
            .rope_base_bits = @bitCast(c.rope_freq_base),
            .kv_slot = draft_pos,
            .attn_slot = draft_pos,
            .rows = 1,
            .tok = blk: {
                var tt: [backend.ctl_tokens]u32 = @splat(0);
                tt[0] = next_token;
                break :blk tt;
            },
        }, graph_slot_nextn);
        defer {
            self.exec.graphEnd(&self.graph);
            // 回到直通句柄，而不是某个"外层"图：调用方进来时 `self.graph`
            // 只可能是直通（见上面的串行约定）。
            self.graph = .{ .kind = .decode, .slot = graph_slot_decode, .ctl = self.directCtl(draft_pos, next_token, 1) };
        }

        // h = eh_proj([enorm(e); hnorm(h)])  -- one matvec over a 2*hidden input.
        //
        // `enorm`/`hnorm` are RMSNorms, and they are applied to the two halves
        // INDEPENDENTLY (each over its own `hidden` values), not jointly over
        // the concatenation: GGUF ships them as two separate [hidden] weights,
        // and `pre_fc_norm_embedding` / `pre_fc_norm_hidden` are two distinct
        // modules in the reference.
        self.embedTo(next_token, self.x1.span(0, hidden));
        self.exec.rmsnorm(self.graph, self.x1.span(0, hidden), self.x1.span(0, hidden), nn.enorm, c.rms_eps);
        self.exec.copyRange(self.graph, nn.input.span(0, hidden), self.x1.span(0, hidden));
        self.exec.rmsnorm(self.graph, self.x1.span(0, hidden), hidden_buf.span(0, hidden), nn.hnorm, c.rms_eps);
        self.exec.copyRange(self.graph, nn.input.span(hidden, hidden), self.x1.span(0, hidden));

        // The projected value becomes the residual stream for the block.
        self.exec.matvec(self.graph, self.x.span(0, hidden), nn.eh_proj, nn.input.span(0, 2 * hidden));

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
        self.exec.rmsnorm(self.graph, self.x1.span(0, hidden), self.x.span(0, hidden), nn.shared_head_norm, c.rms_eps);
        // Hand the normed output back for the next chained step BEFORE it gets
        // overwritten by anything else in this function (nothing does, but the
        // copy makes the contract explicit and costs one `hidden` memcpy).
        if (out_hidden) |oh| self.exec.copyRange(self.graph, oh.span(0, hidden), self.x1.span(0, hidden));
        if (self.output_weight) |ow| {
            switch (ow) {
                .quant, .f32 => self.exec.matvec(self.graph, self.logits, ow, self.x1.span(0, hidden)),
            }
        } else {
            switch (self.token_embd) {
                .quant => |qrow| self.exec.matvec(self.graph, self.logits, .{ .quant = qrow }, self.x1.span(0, hidden)),
                .f32 => |m| self.exec.matvec(self.graph, self.logits, .{ .f32 = m }, self.x1.span(0, hidden)),
            }
        }
    }

    // ------------- MLP (SPEC C16) -------------

    fn forwardMlp(self: *Engine, lw: *LayerWeights) void {
        const c = self.cfg;
        const hidden: usize = c.hidden;
        self.exec.rmsnorm(self.graph, self.x1, self.x, lw.post_attention_norm, c.rms_eps);
        self.exec.matvec(self.graph, self.mlp_h, lw.ffn_gate, self.x1);
        self.exec.matvec(self.graph, self.mlp_h1, lw.ffn_up, self.x1);
        self.exec.swiglu(self.graph, self.mlp_h, self.mlp_h1);
        self.exec.matvec(self.graph, self.x2.span(0, hidden), lw.ffn_down, self.mlp_h);
        self.exec.addResid(self.graph, self.x.span(0, hidden), self.x2.span(0, hidden));
    }

    // -----------------------------------------------------------------
    // Generation
    // -----------------------------------------------------------------

    pub fn reset(self: *Engine) void {
        for (self.kv) |*cache| cache.len = 0;
        for (self.gdn) |*st| {
            if (st.conv.len > 0) self.exec.zeroBuf(self.graph, st.conv);
            if (st.state.len > 0) self.exec.zeroBuf(self.graph, st.state);
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
                if (g.conv.len > 0) self.exec.copyRange(self.graph, s.conv, g.conv);
                if (g.state.len > 0) self.exec.copyRange(self.graph, s.state, g.state);
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
                    if (g.conv.len > 0) self.exec.copyRange(self.graph, g.conv, s.conv);
                    if (g.state.len > 0) self.exec.copyRange(self.graph, g.state, s.state);
                    g.conv_len = s.conv_len;
                }
            }
            for (tokens, 0..) |t, r| {
                self.graph = .{ .kind = .decode, .slot = graph_slot_decode, .ctl = self.directCtl(pos + @as(u32, @intCast(r)), t, 1) };
                self.embedTo(t, self.x.span(0, self.cfg.hidden));
                _ = self.forwardLogits(t, pos + @as(u32, @intCast(r)));
            }
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
            self.exec.copyOut(self.graph, self.spec_dbg_a, self.gdn[0].state);
            self.specRestoreSnapshot();
            for (tokens, 0..) |t, r| {
                self.graph = .{ .kind = .decode, .slot = graph_slot_decode, .ctl = self.directCtl(pos + @as(u32, @intCast(r)), t, 1) };
                self.embedTo(t, self.x.span(0, self.cfg.hidden));
                _ = self.forwardLogits(t, pos + @as(u32, @intCast(r)));
            }
            self.exec.copyOut(self.graph, self.spec_dbg_b, self.gdn[0].state);
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
                if (g.conv.len > 0) self.exec.copyRange(self.graph, g.conv, sn.conv);
                if (g.state.len > 0) self.exec.copyRange(self.graph, g.state, sn.state);
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
            if (g.conv.len > 0) self.exec.copyRange(self.graph, g.conv, sn.conv);
            g.conv_len = sn.conv_len;
            // state：verify 的内核在**每行跑完后**把整份状态写进快照槽 `t`
            // （0 基，见 `q35_gdn_step_rows_chunk_kernel` 的 `slot t` 注释与
            // `vGdnStepRowsSnap` 的 `@memcpy`）。所以「跑完 `rows` 行之后」的
            // 状态在**槽 `rows-1`**，不是槽 `rows`。
            //
            // 两个已知的坑（都实测过）：
            //   1. 槽号差一。读槽 `rows` 拿到的是 kernel 根本没写过的那一格
            //      （它只写到 `nsnap-1 = rows-1`），内容是上一轮的残留。
            //   2. 快照区与活状态块重叠。`state` 必须始终是第 0 块；用「把
            //      state 指到快照块」代替拷贝，会让下一轮 verify 把同一格既
            //      当活状态又当快照基址，CPU 回退的 `@memcpy` 直接 panic。
            if (rows >= 1 and rows <= spec_rs_seq and
                g.state_all.len >= (snap_slab_base + rows) * nv * kd * vd)
            {
                g.rs_idx = @intCast(rows);
                const src = g.state_all.span((snap_slab_base + rows - 1) * nv * kd * vd, nv * kd * vd);
                g.state = g.state_all.span(0, nv * kd * vd);
                self.exec.copyRange(self.graph, g.state, src);
                used_snap = true;
            } else {
                // 行数超出快照容量（不该发生，verify 块 <= spec_verify_rows）：
                // 退回"恢复快照 + 重跑递推"，正确但慢。
                self.exec.copyRange(self.graph, g.state, sn.state);
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
            self.exec.copyRange(self.graph, self.qkvK.span(0, rows * conv_dim), gi.qkv.span(0, rows * conv_dim));
            self.exec.copyRange(self.graph, self.abK.span(0, rows * 2 * nv), gi.gamma.span(0, rows * 2 * nv));
            self.exec.gdnPrefillCore(self.graph, 
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
            self.exec.gdnStepRowsChunk(self.graph, 
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
        self.exec.copyFromHost(self.graph, row, x[0..hidden]);
        switch (self.output_weight orelse self.token_embd) {
            .quant => |qrow| self.exec.matvec(self.graph, self.logits, .{ .quant = qrow }, row),
            .f32 => |m| self.exec.matvec(self.graph, self.logits, .{ .f32 = m }, row),
        }
        self.exec.copyOut(self.graph, out[0..self.cfg.vocab], self.logits);
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
        return self.exec.argmax(self.graph, self.logits);
    }

    /// Argmax of the current logits plus the top-1 softmax probability.
    ///
    /// `p` is what the MTP drafter's early stop keys on (llama.cpp's `p_min`,
    /// `common/speculative.cpp:1789`).  Only the CUDA backend can produce it
    /// cheaply; every other backend reports 1.0, which disables the stop rather
    /// than changing behaviour.
    pub fn argmaxConfLogits(self: *Engine, p: *f32) u32 {
        return self.exec.argmaxConf(self.graph, self.logits, p);
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
        self.exec.rmsnorm(self.graph, self.x1, h, self.output_norm, c.rms_eps);
        if (self.output_weight) |ow| {
            switch (ow) {
                .quant, .f32 => self.exec.matvec(self.graph, self.logits, ow, self.x1),
            }
        } else {
            switch (self.token_embd) {
                .quant => |qrow| self.exec.matvec(self.graph, self.logits, .{ .quant = qrow }, self.x1),
                .f32 => |m| self.exec.matvec(self.graph, self.logits, .{ .f32 = m }, self.x1),
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
        self.exec.copyOut(self.graph, dst[0..self.cfg.vocab], self.logits);
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
            self.exec.copyOut(self.graph, dst, src);
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
                self.exec.copyOut(self.graph, self.ring[ti * ctx * hidden + at * hidden ..][0..hidden], src);
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
        self.exec.copyOut(self.graph, dst[0 .. rows * hidden], self.specTapPlane(i, rows));
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
        self.exec.matvec(self.graph, tmp, lw, src);
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
        self.exec.rmsnorm(self.graph, dst.span(0, hidden), src, self.output_norm, self.cfg.rms_eps);
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
            for (prompt, 0..) |t, i| {
                self.graph = .{ .kind = .decode, .slot = graph_slot_decode, .ctl = self.directCtl(@intCast(i), t, 1) };
                self.embedTo(t, self.x.span(0, self.cfg.hidden));
                last_logits = self.forwardLogits(t, @intCast(i));
            }
        } else {
            // K-token batched prefill: every layer's weights are read once per
            // chunk (batched projections) and reused across the chunk's K
            // tokens; the KV cache / GDN state updates happen token-by-token in
            // the same order as the serial loop, so the resulting logits of the
            // last prompt token are bit-identical to the serial prefill.
            var i: usize = 0;
            while (i < prompt.len) {
                const n = @min(self.nbatch, prompt.len - i);
                // 尾巴块可能只有 1 个 token（`prompt.len % nbatch == 1`）。两种
                // 形状是**并列**的：`forwardChunk` 只断言 k > 1、内部不转发，
                // 单行一律由调用方走 `forwardLogits`。
                // embedding 归调用方：`forwardLogits` 和 `forwardChunk` 都
                // 不开 span、也不选行。
                if (n == 1) {
                    last_logits = blk: {
                        self.graph = .{ .kind = .decode, .slot = graph_slot_decode, .ctl = self.directCtl(@intCast(i), prompt[i], 1) };
                        self.embedTo(prompt[i], self.x.span(0, self.cfg.hidden));
                        break :blk self.forwardLogits(prompt[i], @intCast(i));
                    };
                } else {
                    last_logits = blk: {
                        self.embedChunk(prompt[i .. i + n]);
                        break :blk self.forwardChunk(prompt[i .. i + n], @intCast(i));
                    };
                }
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
        _ = self.exec.sample(self.graph, last_logits, self.sample_scratch, &self.sampler);
        var next: u32 = self.exec.sampledToken();
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
            for (prompt, 0..) |t, i| {
                self.graph = .{ .kind = .decode, .slot = graph_slot_decode, .ctl = self.directCtl(@intCast(i), t, 1) };
                self.embedTo(t, self.x.span(0, self.cfg.hidden));
                last_logits = self.forwardLogits(t, @intCast(i));
            }
        } else {
            var i: usize = 0;
            while (i < prompt.len) {
                const n = @min(self.nbatch, prompt.len - i);
                // 尾巴块可能只有 1 个 token（`prompt.len % nbatch == 1`）。两种
                // 形状是**并列**的：`forwardChunk` 只断言 k > 1、内部不转发，
                // 单行一律由调用方走 `forwardLogits`。
                // embedding 归调用方：`forwardLogits` 和 `forwardChunk` 都
                // 不开 span、也不选行。
                if (n == 1) {
                    last_logits = blk: {
                        self.graph = .{ .kind = .decode, .slot = graph_slot_decode, .ctl = self.directCtl(@intCast(i), prompt[i], 1) };
                        self.embedTo(prompt[i], self.x.span(0, self.cfg.hidden));
                        break :blk self.forwardLogits(prompt[i], @intCast(i));
                    };
                } else {
                    last_logits = blk: {
                        self.embedChunk(prompt[i .. i + n]);
                        break :blk self.forwardChunk(prompt[i .. i + n], @intCast(i));
                    };
                }
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
        _ = self.exec.sample(self.graph, last_logits, self.sample_scratch, &self.sampler);
        var next: u32 = self.exec.sampledToken();
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
                    self.exec.argmaxRows(self.graph, vlogits, n_verify, vpicks[0..n_verify]);
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
    ex.gdnStepRowsChunk(.{},
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
    ex.gdnPrefillCore(.{},
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
    ex.gdnPrefillCore(.{},
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
    ex.gdnStepRowsChunk(.{},
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
