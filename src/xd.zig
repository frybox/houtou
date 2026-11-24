//! `src/kernels/cuda/xd.cu` 的 Zig 侧声明（显式计算图的 op 表）。
//!
//! 这里只声明 C 侧的四个入口与两个查询函数，**不做任何算术**：op 表、形状、
//! 发射、重定向全在 `xd.cu`。Zig 侧需要知道的只有一件事——「这些 op 的编号
//! 在表里是什么」，而编号由 `xd.cu` 的 `enum xd_op` 与 `xd_op_name` 唯一定义。
//!
//! 为什么单独一个文件而不是塞进 `backend_cuda.zig`：`backend_cuda.zig` 已经在
//! 编译期用 `cuda_linked` 分支处理「没有 kernel 的构建」。这里沿用同样的模式，
//! 但把它隔离出来，这样 `graph_tests.zig` 能 import 它而不拖进整个后端。
const std = @import("std");
const build_options = @import("build_options");

pub const cuda_linked = build_options.cuda_linked;

/// `xd_slot.kind` 的取值，与 `xd.h` 的 `enum xd_kind` 逐项对应。
pub const Kind = enum(c_int) { f32 = 0, i32 = 1, u32 = 2, ptr = 3 };

/// 一个实参槽，与 `xd.h` 的 `struct xd_slot` 二进制一致。
///
/// `src` 是一个八字节的 union：指针类实参用它存放**值本身**（设备地址）；
/// 立即数类实参用它指向宿主的来源单元格（`*const f32` / `*const c_int` /
/// `*const c_uint`），`xd_apply` 每次重放前从那里重新读一遍。`src = null`
/// 表示「字面量」，重放时沿用 `v`。
pub const Src = extern union {
    f: ?*const f32,
    i: ?*const c_int,
    u: ?*const c_uint,
    p: ?*anyopaque,
};

pub const Val = extern union {
    f: f32,
    i: c_int,
    u: c_uint,
};

pub const Slot = extern struct {
    kind: c_int,
    src: Src,
    v: Val,
};

/// 发射几何，与 `xd.h` 的 `struct xd_shape` 一致。
pub const Shape = extern struct {
    gx: c_uint,
    gy: c_uint,
    gz: c_uint,
    bx: c_uint,
    by: c_uint,
    bz: c_uint,
    smem: c_uint,
};

/// 与 `xd.h` 的 `XD_MAX_SLOTS` / `XD_MAX_NODES` 一致。
pub const max_slots = 20;
pub const max_nodes = 4;

/// 一个已录制的 op，与 `xd.h` 的 `struct xd_node` 二进制一致。
pub const Node = extern struct {
    op: c_int,
    n: c_int,
    nnodes: c_int,
    gnode: [max_nodes]?*anyopaque,
    slots: [max_slots]Slot,
    vals: [max_slots]Slot,
    argv: [max_slots]?*anyopaque,
    gate_argv: [4]?*anyopaque,
    derived: [max_slots]c_int,
    /// 本 op 会发射的 CUDA 内核（重放时用它把「自己的节点」同上一个 op 的
    /// 节点区分开）。`attn_decode_dyn` 有两个。
    kern: [max_nodes]?*anyopaque,
};

/// op 编号，与 `xd.h` 的 `enum xd_op` 一致。Zig 侧只按名字用两个
/// （`attn_decode_dyn` 的双发射、`set_int1`），但整套列出来，这样增删 op 时
/// 不一致会在编译期暴露（`comptime` 检查在下方）。
pub const Op = enum(c_int) {
    mul_mat_vec = 0,
    mul_mat_vec_f32,
    mul_mat_vec_w16,
    quantize_q8_0,
    quantize_q8_k,
    rope64_dpos,
    rope64_rows_dpos,
    attn_decode_dyn,
    kv_quant_dslot,
    gdn_step_rows,
    gdn_gamma,
    rmsnorm,
    swiglu,
    silu,
    add_resid,
    copy_range,
    scale_vec,
    zero_buf,
    deinterleave_qg,
    argmax_batch,
    set_int1,
};

/// 表里的 op 数（`xd.h` 的 `XD_OP_COUNT`）。S3 的接线要用它做一次
/// `comptime` 对照：`xd_op_count()` 与 `@typeInfo(Op).@"enum".fields.len`
/// 必须相等，否则某一边加了 op 而另一边没加。
pub const op_count: usize = std.enums.values(Op).len;

const C = if (cuda_linked) struct {
    extern fn xd_op_count() c_int;
    extern fn xd_op_exists(op: c_int) c_int;
    extern fn xd_op_name(op: c_int) [*:0]const u8;
    extern fn xd_launch_direct(op: c_int, a: [*]const Slot, stream: ?*anyopaque) c_int;
    extern fn xd_capture(op: c_int, a: [*]const Slot, n: c_int, stream: ?*anyopaque, gx: c_uint, gy: c_uint, gz: c_uint, bx: c_uint, by: c_uint, bz: c_uint, smem: c_uint, out: *Node) c_int;
    extern fn xd_apply(node: *Node, gexec: ?*anyopaque) c_int;

    // 逐 op 录制入口：参数表写在 `xd.cu`（与消费它的表行同一文件），Zig 侧
    // 只按 kernel 自己的参数顺序调用，不需要知道槽位编号。
    extern fn xd_rec_rope64_dpos(stream: ?*anyopaque, out: *Node, d_x: ?[*]f32, heads: c_int, hd: c_int, d_ctl: ?*const anyopaque) c_int;
    extern fn xd_rec_kv_quant_dslot(stream: ?*anyopaque, out: *Node, d_x: ?[*]const f32, d_out: ?*anyopaque, d_slot: ?*const c_int, nblocks_per_row: c_int, rows_this: c_int, row_bytes: c_int) c_int;
    extern fn xd_rec_gdn_step_rows(stream: ?*anyopaque, out: *Node, d_state: ?[*]f32, state_stride: c_int, d_k: ?[*]const f32, k_stride: c_int, d_v: ?[*]const f32, v_stride: c_int, d_q: ?[*]const f32, q_stride: c_int, d_km: ?[*]f32, d_delta: ?[*]f32, d_out: ?[*]f32, d_gamma: ?[*]const f32, d_beta: ?[*]const f32, heads: c_int, nk: c_int, kd: c_int, vd: c_int) c_int;
    extern fn xd_rec_gdn_gamma(stream: ?*anyopaque, out: *Node, d_a_raw: ?[*]const f32, d_b_raw: ?[*]f32, d_ssm_a: ?[*]const f32, d_dt_bias: ?[*]const f32, d_gamma: ?[*]f32, nv: c_int) c_int;
    extern fn xd_rec_rmsnorm(stream: ?*anyopaque, out: *Node, d_out: ?[*]f32, d_x: ?[*]const f32, d_w: ?[*]const f32, rows: c_int, n: c_int, eps: f32) c_int;
    extern fn xd_rec_swiglu(stream: ?*anyopaque, out: *Node, d_gate: ?[*]f32, d_up: ?[*]const f32, n: c_int) c_int;
    extern fn xd_rec_silu(stream: ?*anyopaque, out: *Node, d_x: ?[*]f32, n: c_int) c_int;
    extern fn xd_rec_add_resid(stream: ?*anyopaque, out: *Node, d_dst: ?[*]f32, d_src: ?[*]const f32, n: c_int) c_int;
    extern fn xd_rec_copy_range(stream: ?*anyopaque, out: *Node, d_dst: ?[*]f32, d_src: ?[*]const f32, n: c_int) c_int;
    extern fn xd_rec_scale_vec(stream: ?*anyopaque, out: *Node, d_x: ?[*]f32, s: f32, n: c_int) c_int;
    extern fn xd_rec_zero_buf(stream: ?*anyopaque, out: *Node, d_buf: ?[*]f32, n: c_int) c_int;
    extern fn xd_rec_deinterleave_qg(stream: ?*anyopaque, out: *Node, d_q: ?[*]f32, d_gate: ?[*]f32, d_src: ?[*]const f32, heads: c_int, hd: c_int) c_int;
    extern fn xd_rec_argmax(stream: ?*anyopaque, out: *Node, d_x: ?[*]const f32, n: c_int, d_out: ?*c_int) c_int;
    extern fn xd_rec_argmax_batch(stream: ?*anyopaque, out: *Node, d_x: ?[*]const f32, n: c_int, rows: c_int, d_out: ?[*]c_int) c_int;
    extern fn xd_rec_set_int1(stream: ?*anyopaque, out: *Node, d: ?[*]c_int, v: c_int) c_int;

    fn count() usize {
        return @intCast(xd_op_count());
    }
    fn exists(op: Op) bool {
        return xd_op_exists(@intFromEnum(op)) != 0;
    }
    fn name(op: Op) [:0]const u8 {
        return std.mem.span(xd_op_name(@intFromEnum(op)));
    }
    fn launch(op: Op, slots: []const Slot, stream: ?*anyopaque) c_int {
        return xd_launch_direct(@intFromEnum(op), slots.ptr, stream);
    }
    fn record(op: Op, slots: []const Slot, stream: ?*anyopaque, out: *Node) c_int {
        // 几何传 0：`xd_capture` 用表自己的 `shape()`，见 xd.h 的说明。
        return xd_capture(@intFromEnum(op), slots.ptr, @intCast(slots.len), stream, 0, 0, 0, 0, 0, 0, 0, out);
    }
    fn replay(node: *Node, gexec: ?*anyopaque) c_int {
        return xd_apply(node, gexec);
    }
} else struct {
    // 没有 CUDA kernel 的构建：所有入口都是 no-op，查询函数返回 0 / "?"。
    // `cuda_linked` 是 comptime 已知的，所以这些函数体不会被链接进最终
    // 二进制；它们只需要存在，让 `graph_tests.zig` 能引用本模块。
    fn count() usize {
        return 0;
    }
    fn exists(_: Op) bool {
        return false;
    }
    fn name(_: Op) [:0]const u8 {
        return "?";
    }
    fn launch(_: Op, _: []const Slot, _: ?*anyopaque) c_int {
        return 0;
    }
    fn record(_: Op, _: []const Slot, _: ?*anyopaque, _: *Node) c_int {
        return 0;
    }
    fn replay(_: *Node, _: ?*anyopaque) c_int {
        return 0;
    }
};

pub fn opCount() usize {
    return C.count();
}

pub fn opExists(op: Op) bool {
    return C.exists(op);
}

pub fn opName(op: Op) [:0]const u8 {
    return C.name(op);
}

/// 直接发射一个 op（warm 路径）。`slots` 的下标即 `xd.cu` 表里该 op 的实参
/// 顺序。返回 0 是 `cudaSuccess`。
pub fn launchDirect(op: Op, slots: []const Slot, stream: ?*anyopaque) c_int {
    return C.launch(op, slots, stream);
}

/// 在已打开的 capture 内录制一个 op，并把节点信息写进 `out`。
pub fn capture(op: Op, slots: []const Slot, stream: ?*anyopaque, out: *Node) c_int {
    return C.record(op, slots, stream, out);
}

/// 重放前刷新实参并重定向所有节点。
pub fn apply(node: *Node, gexec: ?*anyopaque) c_int {
    return C.replay(node, gexec);
}

/// 便于测试/诊断：把一个 `Slot` 设成字面 `f32`。
pub fn setF32(s: *Slot, src: ?*const f32, v: f32) void {
    s.* = .{ .kind = @intFromEnum(Kind.f32), .src = .{ .f = src }, .v = .{ .f = v } };
}

pub fn setI32(s: *Slot, src: ?*const c_int, v: c_int) void {
    s.* = .{ .kind = @intFromEnum(Kind.i32), .src = .{ .i = src }, .v = .{ .i = v } };
}

pub fn setU32(s: *Slot, src: ?*const c_uint, v: c_uint) void {
    s.* = .{ .kind = @intFromEnum(Kind.u32), .src = .{ .u = src }, .v = .{ .u = v } };
}

pub fn setPtr(s: *Slot, p: ?*anyopaque) void {
    s.* = .{ .kind = @intFromEnum(Kind.ptr), .src = .{ .p = p }, .v = .{ .u = 0 } };
}

// ==== 逐 op 录制入口（S3c）==================================================
//
// 参数顺序是 **kernel 自己的**，与槽位编号无关：槽位表写在 `xd.cu`，紧挨
// 消费它的表行。CPU 构建（`cuda_linked == false`）下这些是 no-op：没有任何
// 代码会在没有 CUDA kernel 时进入录制期。

pub fn recRope64Dpos(stream: ?*anyopaque, out: *Node, d_x: ?[*]f32, heads: c_int, hd: c_int, d_ctl: ?*const anyopaque) c_int {
    if (comptime cuda_linked) return C.xd_rec_rope64_dpos(stream, out, d_x, heads, hd, d_ctl);
    return 0;
}
pub fn recKvQuantDslot(stream: ?*anyopaque, out: *Node, d_x: ?[*]const f32, d_out: ?*anyopaque, d_slot: ?*const c_int, nblocks_per_row: c_int, rows_this: c_int, row_bytes: c_int) c_int {
    if (comptime cuda_linked) return C.xd_rec_kv_quant_dslot(stream, out, d_x, d_out, d_slot, nblocks_per_row, rows_this, row_bytes);
    return 0;
}
pub fn recGdnStepRows(stream: ?*anyopaque, out: *Node, d_state: ?[*]f32, state_stride: c_int, d_k: ?[*]const f32, k_stride: c_int, d_v: ?[*]const f32, v_stride: c_int, d_q: ?[*]const f32, q_stride: c_int, d_km: ?[*]f32, d_delta: ?[*]f32, d_out: ?[*]f32, d_gamma: ?[*]const f32, d_beta: ?[*]const f32, heads: c_int, nk: c_int, kd: c_int, vd: c_int) c_int {
    if (comptime cuda_linked) return C.xd_rec_gdn_step_rows(stream, out, d_state, state_stride, d_k, k_stride, d_v, v_stride, d_q, q_stride, d_km, d_delta, d_out, d_gamma, d_beta, heads, nk, kd, vd);
    return 0;
}
pub fn recGdnGamma(stream: ?*anyopaque, out: *Node, d_a_raw: ?[*]const f32, d_b_raw: ?[*]f32, d_ssm_a: ?[*]const f32, d_dt_bias: ?[*]const f32, d_gamma: ?[*]f32, nv: c_int) c_int {
    if (comptime cuda_linked) return C.xd_rec_gdn_gamma(stream, out, d_a_raw, d_b_raw, d_ssm_a, d_dt_bias, d_gamma, nv);
    return 0;
}
pub fn recRmsnorm(stream: ?*anyopaque, out: *Node, d_out: ?[*]f32, d_x: ?[*]const f32, d_w: ?[*]const f32, rows: c_int, n: c_int, eps: f32) c_int {
    if (comptime cuda_linked) return C.xd_rec_rmsnorm(stream, out, d_out, d_x, d_w, rows, n, eps);
    return 0;
}
pub fn recSwiglu(stream: ?*anyopaque, out: *Node, d_gate: ?[*]f32, d_up: ?[*]const f32, n: c_int) c_int {
    if (comptime cuda_linked) return C.xd_rec_swiglu(stream, out, d_gate, d_up, n);
    return 0;
}
pub fn recSilu(stream: ?*anyopaque, out: *Node, d_x: ?[*]f32, n: c_int) c_int {
    if (comptime cuda_linked) return C.xd_rec_silu(stream, out, d_x, n);
    return 0;
}
pub fn recAddResid(stream: ?*anyopaque, out: *Node, d_dst: ?[*]f32, d_src: ?[*]const f32, n: c_int) c_int {
    if (comptime cuda_linked) return C.xd_rec_add_resid(stream, out, d_dst, d_src, n);
    return 0;
}
pub fn recCopyRange(stream: ?*anyopaque, out: *Node, d_dst: ?[*]f32, d_src: ?[*]const f32, n: c_int) c_int {
    if (comptime cuda_linked) return C.xd_rec_copy_range(stream, out, d_dst, d_src, n);
    return 0;
}
pub fn recScaleVec(stream: ?*anyopaque, out: *Node, d_x: ?[*]f32, s: f32, n: c_int) c_int {
    if (comptime cuda_linked) return C.xd_rec_scale_vec(stream, out, d_x, s, n);
    return 0;
}
pub fn recZeroBuf(stream: ?*anyopaque, out: *Node, d_buf: ?[*]f32, n: c_int) c_int {
    if (comptime cuda_linked) return C.xd_rec_zero_buf(stream, out, d_buf, n);
    return 0;
}
pub fn recDeinterleaveQg(stream: ?*anyopaque, out: *Node, d_q: ?[*]f32, d_gate: ?[*]f32, d_src: ?[*]const f32, heads: c_int, hd: c_int) c_int {
    if (comptime cuda_linked) return C.xd_rec_deinterleave_qg(stream, out, d_q, d_gate, d_src, heads, hd);
    return 0;
}
pub fn recArgmax(stream: ?*anyopaque, out: *Node, d_x: ?[*]const f32, n: c_int, d_out: ?*c_int) c_int {
    if (comptime cuda_linked) return C.xd_rec_argmax(stream, out, d_x, n, d_out);
    return 0;
}
pub fn recArgmaxBatch(stream: ?*anyopaque, out: *Node, d_x: ?[*]const f32, n: c_int, rows: c_int, d_out: ?[*]c_int) c_int {
    if (comptime cuda_linked) return C.xd_rec_argmax_batch(stream, out, d_x, n, rows, d_out);
    return 0;
}
pub fn recSetInt1(stream: ?*anyopaque, out: *Node, d: ?[*]c_int, v: c_int) c_int {
    if (comptime cuda_linked) return C.xd_rec_set_int1(stream, out, d, v);
    return 0;
}
