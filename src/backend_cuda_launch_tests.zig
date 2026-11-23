//! Structural guard for the `Launch` seam in `backend_cuda.zig`.
//!
//! `Launch` is `if (cuda_linked) <aliases of the real launchers> else <stubs
//! that report "rebuild with -Dcuda=true">`.  `cuda_linked` is comptime-known, so
//! exactly ONE of those structs exists in any given build and no type check can
//! ever notice the two lists disagreeing.  Both failure modes are real and both
//! were live in the tree at the same time:
//!
//!   * aliased in the CUDA branch, no stub  -> `zig build` (without -Dcuda) fails
//!     with `struct 'Launch__struct_N' has no member named 'set_int1'` at the call
//!     site, i.e. the CPU build is broken by CUDA-side work;
//!   * a call site naming `Kernels.q35_launch_X(...)` instead of `Launch.X(...)`
//!     -> it bypasses the seam, the stub is dead code, and the CPU-only build
//!     fails at LINK time with `undefined symbol: q35_launch_X`.
//!
//! The second is the nastier one: it compiles in the branch it belongs to and only
//! ever shows up in the build nobody runs while working on CUDA code.
//!
//! This is deliberately a *source-text* check -- "two mutually exclusive branches
//! agree" is not expressible as a type assertion.  It runs in both builds.

const std = @import("std");

const src_text = @embedFile("backend_cuda.zig");

const LAUNCH_HEAD = "const Launch = if (cuda_linked) struct {";
const LAUNCH_ELSE = "} else struct {";
const LAUNCH_END = "\n};";

/// The two branch bodies of the `Launch` struct, as text.
fn launchBranches() struct { cuda: []const u8, stub: []const u8 } {
    const hi = std.mem.indexOf(u8, src_text, LAUNCH_HEAD) orelse
        @panic("backend_cuda.zig: `Launch` no longer opens with the expected line. Update LAUNCH_HEAD here -- and keep this test: it is the only thing comparing the two branches.");
    const after = hi + LAUNCH_HEAD.len;
    const mi = std.mem.indexOfPos(u8, src_text, after, LAUNCH_ELSE) orelse
        @panic("backend_cuda.zig: `Launch` no longer has an `} else struct {` fallback branch.");
    const ti = std.mem.indexOfPos(u8, src_text, mi + LAUNCH_ELSE.len, LAUNCH_END) orelse
        @panic("backend_cuda.zig: could not find the end of the `Launch` struct.");
    return .{ .cuda = src_text[after..mi], .stub = src_text[mi + LAUNCH_ELSE.len .. ti] };
}

const List = std.ArrayList([]const u8);

/// Member names declared in a `Launch` branch body.
/// `.alias` -> `const <name> = Kernels.<...>;`   (the CUDA branch)
/// `.stub`  -> `fn <name>(...)`                  (the fallback branch)
///
/// Matched on the trimmed line so a reindent cannot silently shrink a list to
/// empty; the caller asserts both lists are non-trivial before comparing.
fn members(body: []const u8, kind: enum { alias, stub }, alloc: std.mem.Allocator) ![]const []const u8 {
    var out: List = .empty;
    errdefer out.deinit(alloc);
    var it = std.mem.splitScalar(u8, body, '\n');
    while (it.next()) |raw| {
        const line = std.mem.trim(u8, raw, " \t\r");
        switch (kind) {
            .alias => {
                // Two legal shapes since S3c threads `g` through the seam:
                //   const <name> = Kernels.<...>;                  (plain alias)
                //   fn <name>(g: backend.Graph, ...) { ... Kernels... }  (forwarder)
                // The forwarder is still ONE place per kernel: it is what lets a
                // launch consult the graph state before reaching the extern.
                if (std.mem.startsWith(u8, line, "const ")) {
                    const eq = std.mem.indexOfScalar(u8, line, '=') orelse continue;
                    if (!std.mem.startsWith(u8, std.mem.trimStart(u8, line[eq + 1 ..], " "), "Kernels.")) continue;
                    try out.append(alloc, std.mem.trim(u8, line["const ".len..eq], " "));
                } else if (std.mem.startsWith(u8, line, "fn ") and std.mem.indexOf(u8, line, "g: backend.Graph") != null) {
                    const op = std.mem.indexOfScalar(u8, line, '(') orelse continue;
                    try out.append(alloc, std.mem.trim(u8, line["fn ".len..op], " "));
                }
            },
            .stub => {
                if (!std.mem.startsWith(u8, line, "fn ")) continue;
                const op = std.mem.indexOfScalar(u8, line, '(') orelse continue;
                try out.append(alloc, std.mem.trim(u8, line["fn ".len..op], " "));
            },
        }
    }
    return out.toOwnedSlice(alloc);
}

fn join(list: []const []const u8, buf: []u8) []const u8 {
    var n: usize = 0;
    for (list, 0..) |x, i| {
        if (i > 0 and n + 1 <= buf.len) {
            buf[n] = ' ';
            n += 1;
        }
        const take = @min(x.len, buf.len - n);
        @memcpy(buf[n..][0..take], x[0..take]);
        n += take;
    }
    return buf[0..n];
}

fn has(list: []const []const u8, name: []const u8) bool {
    for (list) |x| if (std.mem.eql(u8, x, name)) return true;
    return false;
}

test "Launch: every CUDA alias has a fallback stub, and vice versa" {
    const br = launchBranches();
    const aliased = try members(br.cuda, .alias, std.testing.allocator);
    defer std.testing.allocator.free(aliased);
    const stubs = try members(br.stub, .stub, std.testing.allocator);
    defer std.testing.allocator.free(stubs);

    // Guards against the markers above matching a degenerate region, which would
    // make the comparison below vacuously pass.
    try std.testing.expect(aliased.len >= 30);
    try std.testing.expect(stubs.len >= 30);

    var missing: List = .empty;
    defer missing.deinit(std.testing.allocator);
    for (aliased) |a| if (!has(stubs, a)) try missing.append(std.testing.allocator, a);

    var orphan: List = .empty;
    defer orphan.deinit(std.testing.allocator);
    for (stubs) |s| if (!has(aliased, s)) try orphan.append(std.testing.allocator, s);

    var namebuf: [4096]u8 = undefined;
    if (missing.items.len != 0) {
        std.debug.print(
            "\nLaunch: aliased in the CUDA branch but MISSING a fallback stub: {s}\n" ++
                "  -> `zig build` (no -Dcuda) fails with 'no member named ...' at any call site.\n",
            .{join(missing.items, &namebuf)},
        );
        return error.MissingStub;
    }
    if (orphan.items.len != 0) {
        std.debug.print(
            "\nLaunch: stub with no CUDA-branch alias: {s}\n" ++
                "  -> dead code; that is how `conv_step_rows` & friends hid a seam bypass.\n",
            .{join(orphan.items, &namebuf)},
        );
        return error.OrphanStub;
    }
    try std.testing.expectEqual(aliased.len, stubs.len);
}

/// `const <name> = Kernels.<something>;` -- a seam alias, not a call.
/// Requires an `=` and the statement to end at the `;`, so a call expression
/// stored in a local (`const m = Kernels.q35_X(...)`) is NOT matched.
fn isAlias(line: []const u8) bool {
    // A forwarder (`fn <name>(g: backend.Graph, ...)`) is the seam too: its body
    // is where the extern is reached, and it exists in BOTH branches.
    if (std.mem.startsWith(u8, line, "fn ") and std.mem.indexOf(u8, line, "g: backend.Graph") != null) return true;
    if (!std.mem.startsWith(u8, line, "const ")) return false;
    const eq = std.mem.indexOfScalar(u8, line, '=') orelse return false;
    const rhs = std.mem.trim(u8, line[eq + 1 ..], " \t");
    if (!std.mem.startsWith(u8, rhs, "Kernels.")) return false;
    // `Kernels.name;` or `Kernels.name ,` -- never `Kernels.name(`.
    const name = rhs["Kernels.".len..];
    const end = std.mem.indexOfScalar(u8, name, ';') orelse return false;
    return std.mem.indexOfScalar(u8, name[0..end], '(') == null;
}

test "Launch: no call site bypasses the seam with Kernels.q35_*" {
    // Legal places for `Kernels.q35_*`: the `extern fn` declarations (no
    // `Kernels.` prefix, so they cannot match) and the `const ... = Kernels.
    // q35_*;` alias lines inside the CUDA branch.  Anything else is a call site,
    // and a call site that names the extern directly has no fallback.
    //
    // The prefix is `Kernels.q35_` and NOT `Kernels.q35_launch_`: query-style
    // helpers are just as linkable-only-with-CUDA as the launchers.  A real miss
    // slipped through under the narrower prefix -- `q35_attn_chunk_max_rows` was
    // called as `Kernels.q35_attn_chunk_max_rows` from `dAttnChunk` while
    // `Launch.attn_chunk` had a stub, so `make cpu` failed with
    // `undefined symbol: q35_attn_chunk_max_rows` even though this test passed.
    var it = std.mem.splitScalar(u8, src_text, '\n');
    var line_no: usize = 0;
    // True while inside a `fn <name>(g: backend.Graph, ...)` body.  Forwarders
    // are one-liners (`{`, body, `}`), so tracking the brace depth is enough.
    var in_forwarder = false;
    var bad: [32]usize = undefined;
    var nbad: usize = 0;
    while (it.next()) |raw| {
        line_no += 1;
        // Only the CODE part of the line: `Kernels.q35_*` written in prose
        // (there is such a sentence in `backend_cuda.zig`'s own comment) is not a
        // call site.
        const code = raw[0 .. std.mem.indexOf(u8, raw, "//") orelse raw.len];
        const line = std.mem.trim(u8, code, " \t\r");
        if (line.len == 0) continue;
        if (std.mem.startsWith(u8, line, "fn ") and std.mem.indexOf(u8, line, "g: backend.Graph") != null) {
            in_forwarder = true;
            if (std.mem.indexOfScalar(u8, line, '}') != null) in_forwarder = false;
            continue;
        }
        if (in_forwarder) {
            if (std.mem.indexOfScalar(u8, line, '}') != null) in_forwarder = false;
            continue;
        }
        if (std.mem.indexOf(u8, line, "Kernels.q35_") == null) continue;
        // An ALIAS is `const <name> = Kernels.<...>;` and nothing else.  Testing
        // merely for a leading `const ` also swallowed ordinary
        // `const m = Kernels.q35_X(...)` call sites -- which is exactly how the
        // `q35_attn_chunk_max_rows` bypass below stayed invisible for a release.
        if (isAlias(line)) continue; // that IS the seam
        // The BODY of a forwarder: `return Kernels.q35_launch_X(...)` inside the
        // `fn <name>(g: backend.Graph, ...)` that the branch declares.  That line
        // is the seam's implementation, not a bypass -- but only while it sits
        // inside a forwarder, which `inForwarder` tracks.
        if (in_forwarder) continue;
        // `q35_func_*` returns a raw function pointer and is only ever used from
        // inside `if (cuda_linked)`, so the CPU branch never analyses those lines
        // (see `resolve()` in the graph-capture section).  They are a different
        // seam -- `Launch` exists for launch calls, not for node classification --
        // and they need no stub.
        if (std.mem.indexOf(u8, line, "Kernels.q35_func_") != null) continue;
        if (nbad < bad.len) bad[nbad] = line_no;
        nbad += 1;
    }
    if (nbad != 0) {
        std.debug.print(
            "\nLaunch seam bypassed: {d} call site(s) name `Kernels.q35_*` directly.\n" ++
                "  -> the CPU-only build links against a symbol that only exists with -Dcuda=true.\n" ++
                "     Call `Launch.<name>` instead, and give it a stub if it lacks one.\n",
            .{nbad},
        );
        for (bad[0..@min(nbad, bad.len)]) |ln| std.debug.print("     backend_cuda.zig:{d}\n", .{ln});
        return error.LaunchSeamBypassed;
    }
}
