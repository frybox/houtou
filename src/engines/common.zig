//! Shared driver code for every engine under `src/engines/`.
//!
//! Why an interface and not shared state: the two engines' `Engine` structs are
//! 49 (qwen35) and 27 (qwen3) fields, and only TEN of those are touched by the
//! text-generation loop.  Sharing a struct would make the smaller engine carry
//! the larger one's speculative-decoding, GDN and nextn state forever; sharing
//! an interface costs one vtable of function pointers and keeps each engine's
//! state private.
//!
//! What actually lives here, and why it can be architecture-neutral:
//!
//!   * `generate` -- prefill, then sample-and-step until EOS / context limit /
//!     n_predict.  It only needs "run these prompt ids", "advance one token" and
//!     "sample from the logits", none of which care how a layer is built.
//!   * `Driver` -- that narrow contract.  An engine implements it by returning
//!     a value whose functions call back into the engine; nothing here knows
//!     what an `Engine` is.
//!
//! What deliberately does NOT live here: `forwardLogits`, `step`, `reset`,
//! `argmax`, KV-cache ownership, sampling policy.  Those differ per engine
//! (qwen35 batches its prefill through `forwardChunk` and owns a GDN state
//! machine; qwen3 is uniform full attention), and a premature abstraction over
//! them would have to take a flag for each difference -- which is how the
//! `isFullAttention`/`full_attention_interval` tangle in `qwen35.zig` started.

const std = @import("std");
const backend = @import("../backend.zig");
const sampler = @import("../sampler.zig");
const Io = std.Io;

const Buf = backend.Buf;

/// What `generate` needs from an engine.  Implemented by each engine in
/// `Driver.ops`; the functions take the engine pointer as `ctx`.
pub const Driver = struct {
    ctx: *anyopaque,
    ops: *const Ops,

    pub const Ops = struct {
        /// Restart from a clean state (KV cache, recurrent state, PRNG).
        reset: *const fn (ctx: *anyopaque) void,
        /// Feed `prompt` (ids start at position 0) and return the LOGITS of the
        /// last prompt position.  Engines with a batched prefill do that here.
        prefill: *const fn (ctx: *anyopaque, prompt: []const u32) Buf,
        /// Advance one token at `pos`, leaving the logits of that position in
        /// the engine's logits buffer.
        step: *const fn (ctx: *anyopaque, token: u32, pos: u32) void,
        /// Sample from the current logits.
        sample: *const fn (ctx: *anyopaque) u32,
        /// How many slots the KV cache holds.
        max_seq: *const fn (ctx: *anyopaque) u32,
        eos_id: u32,
    };

    pub fn reset(self: Driver) void {
        self.ops.reset(self.ctx);
    }
    pub fn prefill(self: Driver, prompt: []const u32) Buf {
        return self.ops.prefill(self.ctx, prompt);
    }
    pub fn step(self: Driver, token: u32, pos: u32) void {
        self.ops.step(self.ctx, token, pos);
    }
    pub fn sample(self: Driver) u32 {
        return self.ops.sample(self.ctx);
    }
    pub fn maxSeq(self: Driver) u32 {
        return self.ops.max_seq(self.ctx);
    }
};

pub const GenResult = struct {
    tokens: []u32,
    prefill_ms: f64,
    decode_ms: f64,
    ms_per_token: f64,
    hit_eos: bool,
    hit_context_limit: bool,
};

/// Prompt in, generated ids out.  Greedy or sampled depending on the engine's
/// sampler; the loop itself never inspects the logits.
///
/// The order here is load-bearing and matches both engines' previous
/// hand-written loops:
///   1. `reset`, so a second `generate` on the same engine is independent;
///   2. prefill, whose last position yields the FIRST sampled token;
///   3. append that token, then `step` it to get the next sample -- so a token
///      is never fed back at the position it was sampled from.
pub fn generate(
    drv: Driver,
    alloc: std.mem.Allocator,
    io: Io,
    prompt: []const u32,
    n_predict: usize,
    progress: ?*const fn (ctx: ?*anyopaque, tok: u32) void,
    progress_ctx: ?*anyopaque,
) !GenResult {
    const ctx_max = drv.maxSeq();
    if (prompt.len >= ctx_max) {
        std.debug.print(
            "generate: prompt is {d} tokens, but the context holds {d}\n",
            .{ prompt.len, ctx_max },
        );
        return error.ContextLengthExceeded;
    }

    drv.reset();

    const t_prefill = Io.Clock.Timestamp.now(io, .awake);
    _ = drv.prefill(prompt);
    // `$QWEN35_PREFILL_SYNC`: a device backend submits the chunk's kernels
    // asynchronously, so the wall clock above stops as soon as the LAST launch
    // is queued -- on CUDA that reported "892 tokens in 0.007 s".  `sample` is
    // the first call that must read a result back (it downloads the token id),
    // so it is also the first point where the device is known to be idle.
    // Doing it here makes the timer measure the prefill instead of the
    // submission, and the sampled token is kept so the loop below does not
    // re-draw it.
    var pre_sampled: ?u32 = null;
    if (@import("../cudart.zig").environ_map != null and
        @import("../cudart.zig").environ_map.?.get("QWEN35_PREFILL_SYNC") != null)
    {
        pre_sampled = drv.sample();
    }
    const prefill_ms = @as(f64, @floatFromInt(
        t_prefill.durationTo(Io.Clock.Timestamp.now(io, .awake)).raw.nanoseconds,
    )) / 1e6;

    var out: std.ArrayList(u32) = .empty;
    errdefer out.deinit(alloc);
    var pos: u32 = @intCast(prompt.len);
    var generated: usize = 0;
    var gen_ns: i96 = 0;
    var hit_eos = false;
    var hit_context_limit = false;

    var next: u32 = pre_sampled orelse drv.sample();
    while (generated < n_predict) {
        if (pos >= ctx_max) {
            hit_context_limit = true;
            break;
        }
        if (next == drv.ops.eos_id) {
            hit_eos = true;
            break;
        }
        try out.append(alloc, next);
        if (progress) |f| f(progress_ctx, next);
        const t0 = Io.Clock.Timestamp.now(io, .awake);
        drv.step(next, pos);
        next = drv.sample();
        gen_ns += t0.durationTo(Io.Clock.Timestamp.now(io, .awake)).raw.nanoseconds;
        pos += 1;
        generated += 1;
    }

    const per: f64 = if (out.items.len > 0)
        @as(f64, @floatFromInt(gen_ns)) / 1e6 / @as(f64, @floatFromInt(out.items.len))
    else
        0;
    return .{
        .tokens = try out.toOwnedSlice(alloc),
        .prefill_ms = prefill_ms,
        .decode_ms = @as(f64, @floatFromInt(gen_ns)) / 1e6,
        .ms_per_token = per,
        .hit_eos = hit_eos,
        .hit_context_limit = hit_context_limit,
    };
}

test "generate stops on EOS without appending it" {
    const T = struct {
        var logits: [4]f32 = .{ 0, 0, 0, 0 };
        var seq: u32 = 0;
        var n_step: usize = 0;
        var samples: usize = 0;

        fn reset(_: *anyopaque) void {
            seq = 0;
            n_step = 0;
            samples = 0;
        }
        fn prefill(_: *anyopaque, prompt: []const u32) Buf {
            seq = @intCast(prompt.len);
            return Buf.empty;
        }
        fn step(_: *anyopaque, _: u32, p: u32) void {
            n_step += 1;
            seq = p + 1;
        }
        fn sample(_: *anyopaque) u32 {
            // Non-EOS for the first two calls, EOS on the third.  Counting
            // samples (not positions) is what makes the intent explicit: the
            // loop must append exactly two tokens and then stop.
            samples += 1;
            return if (samples >= 3) 2 else 1;
        }
        fn maxSeq(_: *anyopaque) u32 {
            return 64;
        }
        var ops: Driver.Ops = .{
            .reset = reset,
            .prefill = prefill,
            .step = step,
            .sample = sample,
            .max_seq = maxSeq,
            .eos_id = 2,
        };
    };
    var threaded = std.Io.Threaded.init(std.testing.allocator, .{});
    defer threaded.deinit();
    var self: u8 = 0;
    const drv = Driver{ .ctx = &self, .ops = &T.ops };
    const res = try generate(drv, std.testing.allocator, threaded.io(), &.{ 7, 8, 9 }, 16, null, null);
    defer std.testing.allocator.free(res.tokens);
    // Samples: pos3 -> 1 (appended), pos4 -> 1 (appended), pos5 -> 2 = EOS.
    try std.testing.expectEqual(@as(usize, 2), res.tokens.len);
    try std.testing.expect(res.hit_eos);
    try std.testing.expect(!res.hit_context_limit);
    try std.testing.expectEqual(@as(usize, 2), T.n_step);
}

test "generate stops at the context limit, not past it" {
    const T = struct {
        var seq: u32 = 0;
        fn reset(_: *anyopaque) void {
            seq = 0;
        }
        fn prefill(_: *anyopaque, prompt: []const u32) Buf {
            seq = @intCast(prompt.len);
            return Buf.empty;
        }
        fn step(_: *anyopaque, _: u32, p: u32) void {
            seq = p + 1;
        }
        fn sample(_: *anyopaque) u32 {
            return 5;
        }
        fn maxSeq(_: *anyopaque) u32 {
            return 8;
        }
        var ops: Driver.Ops = .{
            .reset = reset,
            .prefill = prefill,
            .step = step,
            .sample = sample,
            .max_seq = maxSeq,
            .eos_id = 99,
        };
    };
    var threaded = std.Io.Threaded.init(std.testing.allocator, .{});
    defer threaded.deinit();
    var self: u8 = 0;
    const drv = Driver{ .ctx = &self, .ops = &T.ops };
    // 4-token prompt in an 8-slot cache -> 4 more tokens fit, then the limit.
    const res = try generate(drv, std.testing.allocator, threaded.io(), &.{ 1, 2, 3, 4 }, 100, null, null);
    defer std.testing.allocator.free(res.tokens);
    try std.testing.expectEqual(@as(usize, 4), res.tokens.len);
    try std.testing.expect(res.hit_context_limit);
    try std.testing.expect(!res.hit_eos);
}
