//! 显式计算图（XG）的宿主侧状态机：S1 阶段只做「表 + 状态」。
//!
//! 本文件不发射任何内核，也不认识 CUDA。它只定义两样东西：
//!
//!   1. `GraphCtl`：一张图的**逐 step 标量**（pos / slot / rows / ...），
//!      主机侧就是这张表的真身；`backend_cuda.zig` 在重放前把它整块 H2D 到
//!      设备控制块上，所以任何「每 token 会变的标量」都从这里读，不经过
//!      `cudaGraphExecKernelNodeSetParams`。
//!
//!   2. `Registry`：按 `(signature, kind, rows)` 索引的图条目，每个条目是一个
//!      **三态状态机**（`absent → warm → live`）。它与 `xd.cu` 的 op 表是两件
//!      不同的事：`xd.cu` 说「这个 op 怎么发射」，这里说「这张图现在处于哪一步」。
//!
//! 为什么状态机在宿主侧而不是在后端里：同一段状态机要同时服务 decode、verify
//! 和 MTP 的 draft（三种形状、三种同步策略）。放在这里它就是一个纯数据测试可以
//! 覆盖的对象（`graph_tests.zig`），不需要 GPU 或 CUDA 上下文。
//!
//! 状态的语义（与 llama.cpp `ggml_backend_cuda_graph_compute` 的 warmup 协议
//! 对齐，见 plan §1.2）：
//!
//!   absent → warm   第 1 次调用：**直接发射**，不建图。目的只有一个——把所有
//!                   惰性分配与首次缓存走到稳态，这样录制时不会在 capture 里
//!                   出现 cudaMalloc（那是非法的）。
//!   warm   → live   第 2 次调用：打开 capture、逐 op 录制（同时发射）、
//!                   实例化；然后**立刻用这张图 launch 一次**，让被录制的那
//!                   一轮真正执行（capture 期间内核不会跑）。从这一轮起，
//!                   每一轮的执行方式都只有一种：一次 `cudaGraphLaunch`。
//!   live             第 3 次及以后：刷控制块 + 少量 region 重定向 + launch。
//!
//! `failed` 是一次性的：任何一步失败都退回「直接发射」，不再重试——重试只会在
//! 同一条坏路径上反复失败，且会把失败藏进每 token 的噪声里。

const std = @import("std");
const backend = @import("backend.zig");

/// 图按什么形状区分。`decode` 是单行前向（`forwardGdn/forwardFull/forwardMlp`），
/// `chunk` 是 K 行前向（`forwardChunkRun`，prefill 与 verify 共用）。
///
/// MTP 的 draft 不单列：它走的是 `decode` 的行数（rows = 1），差别只在
/// `GraphCtl` 的 `next_token` / `draft_pos` 字段——那两个字段本来就是逐 step
/// 标量，所以 draft 与 decode 共用同一张图不需要新的 kind（plan §8.1）。
pub const Kind = enum { decode, chunk };

pub const State = enum {
    /// 没有这张图，下一次调用走 warm。
    absent,
    /// warm 已完成，下一次调用录制。
    warm,
    /// 已实例化，可以直接重放。
    live,
    /// 这条路已经失败过：永久退回直接发射。
    failed,
};

/// 一张图的逐 step 标量。**字段顺序就是设备控制块的布局**，`xd.cu` 里的
/// 内核通过 `const int *` 读它；改动顺序必须同步改设备侧（目前只有 S3 引入
/// 的 `GraphCtl` 内核参数会读，S1 还没有消费者）。
///
/// 全部是 `i32`：一块 64 B 的控制块，一次异步 H2D 就能刷完。
pub const GraphCtl = extern struct {
    /// 本 step 的位置（rope 的 `pos`，以及 kv 写入的 `slot`）。
    pos: i32 = 0,
    /// KV 写入的起始行。decode 恒等于 `pos`，verify/chunk 是 `slot0`。
    slot: i32 = 0,
    /// 本 step 的行数（decode = 1，verify = k）。内核的 grid 不走它——grid
    /// 在录制时固定——它只给需要按行数提前返回的变体用。
    rows: i32 = 0,
    /// MTP draft 的输入 token / 位置。非 draft 路径不使用，但占位保证布局
    /// 对所有 kind 相同。
    next_token: i32 = -1,
    draft_pos: i32 = -1,
    /// 采样器写回、`graphEnd` 之后由宿主读走的 token。
    sampled_token: i32 = -1,
    _pad: [10]i32 = @splat(0),
};

comptime {
    // 控制块必须能被一次 4 字节对齐的 memcpy 整体搬走，且大小是 4 的倍数。
    std.debug.assert(@sizeOf(GraphCtl) == 64);
    std.debug.assert(@alignOf(GraphCtl) == 4);
}

/// 表里属于哪一档。S1 只用到 `resolve`，`plan` 是给 S3 的调用点写清楚的。
pub const Transition = union(enum) {
    /// 直接发射（第一次 warm、或已 failed）。
    direct,
    /// 录制（warm 之后的那一次）。
    record,
    /// 重放（live 之后的所有次）。
    replay,
    /// 第一次录制失败：这一次仍要直接发射，并且以后都不再试。
    give_up,
};

pub fn resolve(state: State) Transition {
    return switch (state) {
        .absent => .direct,
        .warm => .record,
        .live => .replay,
        .failed => .direct,
    };
}

/// 录制完成后的状态迁移：成功 → live，失败 → failed。
pub fn afterRecord(ok: bool) State {
    return if (ok) .live else .failed;
}

/// 重放失败：一张已 live 的图重放失败，说明节点表或流已经和实例化时的
/// 不一致，重试只会得到同样的失败，所以直接放弃。
pub fn afterReplay(ok: bool, prev: State) State {
    if (ok) return prev;
    return .failed;
}

/// 一个 registry 条目。`exec` / `graph` 是不透明句柄（CUDA 侧所有），S1 不
/// 使用它们，但字段先定下来，S3 的接线就只加实现不改形状。
pub const Entry = struct {
    signature: u64,
    kind: Kind,
    rows: usize,
    state: State = .absent,
    /// `cudaGraphExec_t` / `cudaGraph_t`；S1 恒为 null。
    exec: ?*anyopaque = null,
    graph: ?*anyopaque = null,
    /// 录制出的节点数。S1 的 `XD_DUMP` 打印它，S3 的 `XD_STRICT` 拿它做
    /// 「图没录漏」的第一道对照。
    node_count: usize = 0,
    /// 这张图启动过多少次重放。诊断用，和现有 `stat_g_replay` 同一口径。
    replays: usize = 0,

    pub fn key(self: Entry) Key {
        return .{ .signature = self.signature, .kind = self.kind, .rows = self.rows };
    }
};

/// 图的 key。`signature` 是「本模型 + kv 类型 + batch 配置」的指纹，
/// 由 `allocScratchK` 时算一次（S3），只用于防止同一进程里两次 `Engine.load`
/// 的图被混用；逐 op 的 shape 不进 key（plan §1.3）。
pub const Key = struct {
    signature: u64,
    kind: Kind,
    rows: usize,
};

/// 一个小 registry。条目数量是「几种 kind × 几个 rows」，个位数，所以是线性
/// 查找而不是哈希表——同时也让内存布局可预测（不在热路径上分配）。
pub const Registry = struct {
    entries: std.ArrayList(Entry) = .empty,

    pub fn init() Registry {
        return .{};
    }

    pub fn deinit(self: *Registry, alloc: std.mem.Allocator) void {
        self.entries.deinit(alloc);
        self.* = .{};
    }

    /// 找到（或登记）一个条目。`state` 只在新建时是 `absent`。
    pub fn slot(self: *Registry, alloc: std.mem.Allocator, k: Key) !*Entry {
        for (self.entries.items) |*e| {
            if (e.signature == k.signature and e.kind == k.kind and e.rows == k.rows) return e;
        }
        try self.entries.append(alloc, .{ .signature = k.signature, .kind = k.kind, .rows = k.rows });
        return &self.entries.items[self.entries.items.len - 1];
    }

    pub fn find(self: *const Registry, k: Key) ?*const Entry {
        for (self.entries.items) |*e| {
            if (e.signature == k.signature and e.kind == k.kind and e.rows == k.rows) return e;
        }
        return null;
    }

    /// 让每个条目都退回「直接发射」，不释放句柄（后端自己释放）。
    /// 用于 `QWEN35_CUDA_GRAPH=0` 的 A/B：开关一拧，所有图立即失活。
    pub fn disableAll(self: *Registry) void {
        for (self.entries.items) |*e| {
            if (e.state != .failed) e.state = .failed;
        }
    }
};
