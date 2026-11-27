//! 显式计算图里「宿主侧能纯数据验证」的那一半。
//!
//! 不需要 GPU，所以在 `make test`（没有 -Dcuda 的默认构建）里也跑：
//!
//!   * `xd.zig` 的布局：`Slot` / `Node` 必须与 `xd.h` 的 C 结构二进制一致。
//!     这两个结构是跨语言 ABI，布局错了就是「图里某个实参读到别的字节」的
//!     静默错，只有在真机重放时才表现出来；这里用 size/align/offset 断言把它
//!     钉死。
//!   * `backend.GraphCtl` 与设备控制块（`xd.h` 的 `XD_CTL_INTS`）的字段布局。
//!
//! 状态机那部分（`graph_registry.zig`）已经删除：录制与重放现在是引擎侧的
//! 显式动作（`beginRecord` / `endRecord` / `launch`），后端不再维护
//! absent/warm/rec/live 这套自动推演。

const std = @import("std");
const xd = @import("xd.zig");
const backend = @import("backend.zig");

// ---------------------------------------------------------------------------
// 跨语言布局
// ---------------------------------------------------------------------------

test "backend.GraphCtl 与 xd.h 的 XD_CTL_INTS 保持一致" {
    // 设备控制块是 `int32_t[XD_CTL_INTS]` =
    //   `XD_SCTL_BASE(21) + XD_SCTL_INTS(16)` = 37。
    // 前 21 个是位置/嵌入行号，后 16 个是采样器字段（见 `sctl_base`）。
    // Zig 侧多一个字段就是读写相邻内存，少一个则某段读不到。
    try std.testing.expectEqual(@as(usize, 37), @sizeOf(backend.GraphCtl) / 4);
    try std.testing.expectEqual(@as(usize, 4), @alignOf(backend.GraphCtl));
    // 字段顺序即布局：tok 从第 6 个 int（下标 5）开始，是 16 个一组的数组。
    var ctl = backend.GraphCtl{};
    ctl.tok[0] = 12345;
    ctl.tok[15] = 678;
    const ints: [*]const i32 = @ptrCast(&ctl);
    try std.testing.expectEqual(@as(i32, 12345), ints[5]);
    try std.testing.expectEqual(@as(i32, 678), ints[5 + 15]);
    try std.testing.expectEqual(@as(usize, 16), backend.ctl_tokens);
    // 采样段紧跟其后：`sctl_base` 必须等于 5 + 16，`token_out` 落在
    // `sctl_base + 6`（与 sample.cu 的 `CTL_TOKEN_OUT` 同一个事实）。
    try std.testing.expectEqual(@as(usize, 21), backend.sctl_base);
    try std.testing.expectEqual(@as(usize, backend.sctl_base * 4), @offsetOf(backend.GraphCtl, "vocab"));
    try std.testing.expectEqual(@as(usize, (backend.sctl_base + 6) * 4), @offsetOf(backend.GraphCtl, "token_out"));
}

test "xd.Slot matches xd.h's xd_slot" {
    // 8 字节对齐 + 24 字节：kind(4) + padding(4) + src union(8) + v union(4) + padding(4)
    try std.testing.expectEqual(@as(usize, 24), @sizeOf(xd.Slot));
    try std.testing.expectEqual(@as(usize, 8), @alignOf(xd.Slot));
}

test "xd.Node matches xd.h's xd_node" {
    // 每个字段的偏移都钉死，与 xd.h 顶部的 `_Static_assert` 一一对应。
    // `struct xd_node` 里最大的成员是 `bool`/指针，整体对齐按 8 走；`gnode`
    // 数组在 12 处补齐到 16。任何一边改了字段顺序，这里或那边的编译期断言就会
    // 失败——而不是等到真机上重放到「某个实参读到别的字节」。
    const N = xd.Node;
    try std.testing.expectEqual(@as(usize, 1312), @sizeOf(N));
    try std.testing.expectEqual(@as(usize, 8), @alignOf(N));
    try std.testing.expectEqual(@as(usize, 0), @offsetOf(N, "op"));
    try std.testing.expectEqual(@as(usize, 4), @offsetOf(N, "n"));
    try std.testing.expectEqual(@as(usize, 8), @offsetOf(N, "nnodes"));
    try std.testing.expectEqual(@as(usize, 16), @offsetOf(N, "gnode"));
    try std.testing.expectEqual(@as(usize, 48), @offsetOf(N, "slots"));
    try std.testing.expectEqual(@as(usize, 528), @offsetOf(N, "vals"));
    try std.testing.expectEqual(@as(usize, 1008), @offsetOf(N, "argv"));
    try std.testing.expectEqual(@as(usize, 1168), @offsetOf(N, "gate_argv"));
    try std.testing.expectEqual(@as(usize, 1200), @offsetOf(N, "derived"));
    try std.testing.expectEqual(@as(usize, 1280), @offsetOf(N, "kern"));
}

test "xd.Op ids and the table length agree" {
    // 两边各自是 `enum xd_op` 与 `xd_op_count()`：名字一一对应。
    // 没有 CUDA kernel 的构建里 `xd_op_count()` 是 0，这一条就只查 Zig 侧。
    if (!xd.cuda_linked) return;
    try std.testing.expectEqual(xd.op_count, xd.opCount());
    for (std.enums.values(xd.Op)) |op| {
        try std.testing.expect(xd.opExists(op));
        const name = xd.opName(op);
        // 每行的 key 是它的 Zig 名字（`set_int1` 之类），不是内核符号名。
        try std.testing.expectEqualStrings(@tagName(op), name);
    }
}
