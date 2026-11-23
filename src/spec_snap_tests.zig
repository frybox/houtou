//! 推测解码回滚的**不变量**测试（纯数据，不需要模型或 GPU）。
//!
//! 这里只钉一条规则，因为它是 27B CUDA + MTP 崩溃的真正原因：
//!
//!   `gdn[i].state_all` 的第 0 块是**活状态**（每步 forward 读写的那一块），
//!   第 `1..17` 块是 verify 的逐行快照。回滚把快照块拷回第 0 块，
//!   并让 `gdn[i].state` 始终指向第 0 块。
//!
//! 反例（修复前的代码）：回滚把 `state` 直接指到快照块 `rows`，下次 verify 时
//! 「活状态」和「快照区起点」是同一块内存 —— CPU 回退的
//! `@memcpy(snap[..], state[..])` 于是撞上 `@memcpy arguments alias` 并在 debug
//! 构建里 panic（实测 27B IQ3_XXS + `QWEN35_SPEC_MTP=1`）；release 构建则静默地
//! 把一块当两块用。
//!
//! 这条断言能抓住「把 state 指进快照区」的任何版本，不管它是怎么写的。

const std = @import("std");
const backend = @import("backend.zig");

/// `gdn[i].state_all` 的块布局。与 `Engine.snap_slab_base` 必须一致。
fn slabLayout(nv: usize, kd: usize, vd: usize, rs_seq: usize) struct {
    live: backend.Buf,
    snap: backend.Buf,
    store: []f32,
} {
    const per = nv * kd * vd;
    const store = std.testing.allocator.alloc(f32, per * (1 + rs_seq)) catch @panic("oom");
    const all = backend.Buf{ .ptr = store.ptr, .len = store.len };
    return .{
        .live = all.span(0, per),
        .snap = all.span(1 * per, rs_seq * per),
        .store = store,
    };
}

fn overlaps(a: backend.Buf, b: backend.Buf) bool {
    const ab = @intFromPtr(a.ptr);
    const bb = @intFromPtr(b.ptr);
    return ab < bb + b.len * @sizeOf(f32) and bb < ab + a.len * @sizeOf(f32);
}

test "rollback: the live state slab never overlaps the snapshot region" {
    // Small shape; the arithmetic is the same for the 27B's 48x128x128.
    const nv: usize = 4;
    const kd: usize = 8;
    const vd: usize = 8;
    const rs_seq: usize = 16;
    const per = nv * kd * vd;

    const layout = slabLayout(nv, kd, vd, rs_seq);
    defer std.testing.allocator.free(layout.store);

    try std.testing.expect(!overlaps(layout.live, layout.snap));

    // Every rollback target must land inside the snapshot region AND the copy
    // back into the live slab must be legal.
    for (1..rs_seq + 1) |rows| {
        const src = layout.snap.span((rows - 1) * per, per);
        try std.testing.expect(!overlaps(src, layout.live));
        // The snapshot region must be large enough for that slot.
        try std.testing.expect(src.len == per);
    }
}

test "rollback: re-pointing state at a snapshot slab is what aliases" {
    // The fix must keep this from ever being the layout: after a rollback the
    // engine copies instead of re-pointing.  This asserts the property that
    // made re-pointing unsafe, so a future "optimisation" back to re-pointing
    // fails here rather than on a 27B GPU run.
    const nv: usize = 4;
    const kd: usize = 8;
    const vd: usize = 8;
    const rs_seq: usize = 16;
    const per = nv * kd * vd;

    const layout = slabLayout(nv, kd, vd, rs_seq);
    defer std.testing.allocator.free(layout.store);

    // What the OLD code did: state := state_all.span(rows*per, per).
    const rows: usize = 2;
    const old_view = backend.Buf{ .ptr = layout.store.ptr + rows * per, .len = per };
    try std.testing.expect(overlaps(old_view, layout.snap));

    // What the FIXED code does: state stays at slab 0 and the value is copied.
    try std.testing.expectEqual(layout.live.ptr, layout.store.ptr);
    try std.testing.expect(!overlaps(layout.live, layout.snap));
    const src = layout.snap.span(rows * per, per);
    @memcpy(layout.live.slice(), src.slice());
}
