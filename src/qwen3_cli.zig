//! `houtou-qwen3`: run the Qwen3 engine directly.
//!
//! Why a separate binary: `main.zig`'s `run` path is written against the
//! Qwen3.5 engine (spec decoding, tap ring, nextn).  Generalising that call site
//! is the `engines/common.zig` work; until then this is how the second engine is
//! exercised end to end.  It is deliberately minimal -- prompt in, tokens out --
//! so that a wrong output points at the engine, not at CLI plumbing.
//!
//!     houtou-qwen3 <model.gguf> <prompt> [-n N] [--backend cpu|cuda]

const std = @import("std");
const Io = std.Io;

const houtou = @import("houtou");

pub fn main(init: std.process.Init) !void {
    const arena: std.mem.Allocator = init.arena.allocator();
    const io = init.io;

    var stdout_buffer: [1 << 16]u8 = undefined;
    var stdout_writer: Io.File.Writer = .init(.stdout(), io, &stdout_buffer);
    const out = &stdout_writer.interface;

    const args = try init.minimal.args.toSlice(arena);
    if (args.len < 3) {
        try out.print(
            "usage: {s} <model.gguf> <prompt> [-n N] [--backend cpu|cuda]\n",
            .{args[0]},
        );
        try out.flush();
        return error.MissingArgument;
    }
    const path = args[1];
    const prompt_text = args[2];

    var n_predict: usize = 32;
    var backend_kind: houtou.backend.Kind = .cpu;
    var i: usize = 3;
    while (i < args.len) : (i += 1) {
        if (std.mem.eql(u8, args[i], "-n") and i + 1 < args.len) {
            n_predict = try std.fmt.parseInt(usize, args[i + 1], 10);
            i += 1;
        } else if (std.mem.eql(u8, args[i], "--backend") and i + 1 < args.len) {
            backend_kind = std.meta.stringToEnum(houtou.backend.Kind, args[i + 1]) orelse {
                try out.print("error: unknown backend '{s}'\n", .{args[i + 1]});
                try out.flush();
                return error.InvalidArgument;
            };
            i += 1;
        } else {
            try out.print("error: unknown argument '{s}'\n", .{args[i]});
            try out.flush();
            return error.InvalidArgument;
        }
    }

    // Tokenizer comes from the same GGUF; the engine reads its own copy of the
    // mapping.
    var file = try Io.Dir.cwd().openFile(io, path, .{});
    defer file.close(io);
    const stat = try file.stat(io);
    var mm = try file.createMemoryMap(io, .{ .len = @intCast(stat.size), .protection = .{ .read = true } });
    defer mm.destroy(io);
    var g = try houtou.gguf.parse(arena, mm.memory);
    defer g.deinit();

    var tok = try houtou.tokenizer.Tokenizer.initFromGguf(arena, &g);
    defer tok.deinit();

    const rendered = try houtou.tokenizer.Tokenizer.renderChatTurn(arena, prompt_text, false);
    const prompt = try tok.encode(arena, rendered);
    try out.print("[prompt: {d} tokens]\n", .{prompt.len});
    try out.flush();

    var e = houtou.engines.qwen3.Engine.load(
        arena,
        path,
        io,
        0,
        backend_kind,
        .f16,
        houtou.engines.qwen3.Engine.max_seq,
    ) catch |err| {
        try out.print("error: failed to load '{s}': {s}\n", .{ path, @errorName(err) });
        try out.flush();
        return err;
    };
    defer e.deinit();
    try out.print(
        "[qwen3: layers={d} hidden={d} heads={d}/{d} hd={d} rotary={d} ff={d}]\n",
        .{ e.cfg.n_layers, e.cfg.hidden, e.cfg.n_heads, e.cfg.n_kv_heads, e.cfg.head_dim, e.cfg.rotary_dim, e.cfg.n_ff },
    );
    try out.flush();

    const StreamCtx = struct {
        tok: *houtou.tokenizer.Tokenizer,
        out: *Io.Writer,
        arena: std.mem.Allocator,
        fn emit(ctx: ?*anyopaque, token_id: u32) void {
            const self: *@This() = @ptrCast(@alignCast(ctx.?));
            const piece = self.tok.decode(self.arena, &.{token_id}) catch null;
            if (piece) |s| {
                if (s.len > 0) {
                    self.out.print("{s}", .{s}) catch {};
                } else {
                    self.out.print("[{d}]", .{token_id}) catch {};
                }
            } else {
                self.out.print("[{d}]", .{token_id}) catch {};
            }
            self.out.flush() catch {};
        }
    };
    var stream_ctx = StreamCtx{ .tok = &tok, .out = out, .arena = arena };

    const tokens = try e.generate(arena, prompt, n_predict, StreamCtx.emit, &stream_ctx);
    try out.print("\n\n[{d} tokens generated]\n", .{tokens.len});
    try out.flush();
}
