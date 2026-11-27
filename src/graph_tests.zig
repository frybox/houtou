//! S1 的验收：显式计算图里「宿主侧能纯数据验证」的那一半。
//!
//! 这两部分都不需要 GPU，所以在 `make test`（没有 -Dcuda 的默认构建）里也跑：
//!
//!   1. `graph_registry.zig` 的状态机：每个状态迁移恰好落在一个分支上，
//!      并且 `live` 之后永远不再录制（录制一次重放 N 次才是收益的来源）。
//!   2. `xd.zig` 的布局：`Slot` / `Node` 必须与 `xd.h` 的 C 结构二进制一致。
//!      这两个结构是跨语言 ABI，布局错了就是「图里某个实参读到别的字节」的
//!      静默错，只有在真机重放时才表现出来；这里用编译期的 size/align 断言把它
//!      钉死。
//!
//! 不在本文件里的：`xd_capture` / `xd_apply` 与 CUDA 的交互。那需要真机，
//! 由 S3 的 `QWEN35_XD_STRICT=1`（replay 与 eager 逐位比对）承担。

const std = @import("std");
const reg = @import("graph_registry.zig");
const xd = @import("xd.zig");
const backend = @import("backend.zig");

// ---------------------------------------------------------------------------
// 状态机
// ---------------------------------------------------------------------------

test "registry: warm -> record -> replay, and never re-records" {
    var r = reg.Registry.init();
    defer r.deinit(std.testing.allocator);

    const key = reg.Key{ .signature = 0x1234, .kind = .decode, .rows = 1 };
    const e = try r.slot(std.testing.allocator, key);
    try std.testing.expectEqual(reg.State.absent, e.state);
    try std.testing.expectEqual(reg.Transition.direct, reg.resolve(e.state));

    // 第 1 次：warm（直接发射），不录制。
    e.state = .warm;
    try std.testing.expectEqual(reg.Transition.record, reg.resolve(e.state));

    // 第 2 次：录制成功。
    e.state = reg.afterRecord(true);
    try std.testing.expectEqual(reg.State.live, e.state);

    // 第 3.. 次：全部重放，且状态不再迁移。
    for (0..5) |_| {
        try std.testing.expectEqual(reg.Transition.replay, reg.resolve(e.state));
        e.state = reg.afterReplay(true, e.state);
        try std.testing.expectEqual(reg.State.live, e.state);
    }
}

test "registry: a failed record gives up permanently" {
    var r = reg.Registry.init();
    defer r.deinit(std.testing.allocator);
    const e = try r.slot(std.testing.allocator, .{ .signature = 1, .kind = .decode, .rows = 1 });
    e.state = .warm;
    e.state = reg.afterRecord(false);
    try std.testing.expectEqual(reg.State.failed, e.state);
    // 失败之后永远是直接发射，不会再进录制/重放。
    for (0..3) |_| try std.testing.expectEqual(reg.Transition.direct, reg.resolve(e.state));
}

test "registry: a failed replay abandons the graph" {
    try std.testing.expectEqual(reg.State.failed, reg.afterReplay(false, .live));
    // 失败之前的任何状态，失败后都一样是 failed。
    try std.testing.expectEqual(reg.State.failed, reg.afterReplay(false, .warm));
    // 成功不做迁移。
    try std.testing.expectEqual(reg.State.live, reg.afterReplay(true, .live));
}

test "registry: entries are keyed by (signature, kind, rows)" {
    var r = reg.Registry.init();
    defer r.deinit(std.testing.allocator);

    const a = try r.slot(std.testing.allocator, .{ .signature = 7, .kind = .decode, .rows = 1 });
    a.node_count = 42;
    const b = try r.slot(std.testing.allocator, .{ .signature = 7, .kind = .decode, .rows = 1 });
    try std.testing.expect(a == b); // 同一个 key 返回同一个条目
    try std.testing.expectEqual(@as(usize, 42), b.node_count);

    // decode/rows=1 与 chunk/rows=1 是两张不同的图。
    const c = try r.slot(std.testing.allocator, .{ .signature = 7, .kind = .chunk, .rows = 1 });
    try std.testing.expect(a != c);
    try std.testing.expectEqual(@as(usize, 2), r.entries.items.len);
}

test "registry: disableAll turns every graph into the direct path" {
    var r = reg.Registry.init();
    defer r.deinit(std.testing.allocator);
    const e = try r.slot(std.testing.allocator, .{ .signature = 9, .kind = .decode, .rows = 1 });
    e.state = .live;
    r.disableAll();
    try std.testing.expectEqual(reg.State.failed, e.state);
    try std.testing.expectEqual(reg.Transition.direct, reg.resolve(e.state));
}

// ---------------------------------------------------------------------------
// 跨语言布局
// ---------------------------------------------------------------------------

test "GraphCtl is a 64-byte control block" {
    // `graph_registry.GraphCtl`（S1 的 registry 布局）仍然按 64 B 一次刷完；
    // 真正接在设备上的那块是 `backend.GraphCtl`（6 个 int32），见下一条用例。
    try std.testing.expectEqual(@as(usize, 64), @sizeOf(reg.GraphCtl));
    try std.testing.expectEqual(@as(usize, 4), @alignOf(reg.GraphCtl));
}

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
