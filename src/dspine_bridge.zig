//! Zig side of the DSpine code-bridge kernels (`src/kernels/cuda/dspine.cu`).
//!
//! Kept as its own tiny module rather than folded into `backend_cuda.zig` for
//! the same reason `src/kernels/x86/dspine.c` is a separate C unit: these four
//! kernels are defined by their bf16 rounding points, and keeping their
//! declaration, their host twin and their device twin adjacent is what makes
//! "the two backends agree" checkable by reading three small files.
//!
//! The device twin is only LINKED when the binary was built with `-Dcuda=true`
//! (see `build.zig`'s `cuda_sources`).  `dspineLinkedIn` exposes that fact so a
//! CPU-only build can refuse to select the CUDA draft instead of failing at link
//! time.
const std = @import("std");

/// True when the four `q35_launch_dspine_*` symbols exist in this binary.
pub const dspineLinkedIn = @import("build_options").cuda_linked;

pub const Launchers = struct {
    pub extern fn q35_launch_dspine_prepare_read(
        d_x: ?[*]const f32,
        d_residual: ?[*]const f32,
        d_raw: ?[*]f32,
        d_normed: ?[*]f32,
        d_rms: ?[*]f32,
        rows: c_int,
        h: c_int,
        eps: f32,
        add: c_int,
        stream: ?*anyopaque,
    ) callconv(.c) c_int;

    pub extern fn q35_launch_dspine_normalize_codes(
        d_projected: ?[*]const f32,
        d_codebook: ?[*]const f32,
        d_anchors: ?[*]const i32,
        d_codes: ?[*]f32,
        rows: c_int,
        c: c_int,
        t: c_int,
        eps: f32,
        row0: c_int,
        stream: ?*anyopaque,
    ) callconv(.c) c_int;

    pub extern fn q35_launch_dspine_feature_message(
        d_packed: ?[*]const f32,
        d_packed_prev: ?[*]const f32,
        d_bias: ?[*]const f32,
        d_message: ?[*]f32,
        rows: c_int,
        r: c_int,
        t: c_int,
        row0: c_int,
        stream: ?*anyopaque,
    ) callconv(.c) c_int;

    pub extern fn q35_launch_dspine_feature_finish(
        d_raw: ?[*]const f32,
        d_delta: ?[*]const f32,
        d_rms: ?[*]const f32,
        d_valid: ?[*]const u8,
        d_weight: ?[*]const f32,
        d_output: ?[*]f32,
        d_normed: ?[*]f32,
        rows: c_int,
        h: c_int,
        t: c_int,
        norm_eps: f32,
        write: c_int,
        norm: c_int,
        row0: c_int,
        stream: ?*anyopaque,
    ) callconv(.c) c_int;
};

/// Which side of the bridge runs the four kernels.  Stage 3 (the 5-layer draft
/// forward plus the code bridge) is fully device-capable; stage 4 (the serial
/// refinement) is not -- it walks `lm_head`'s quantized rows on the host
/// (`src/serial.zig`) and cannot be moved without rewriting it.  The two stages
/// already meet at an explicit host boundary (`stash_raw`/`stash_rms`), so the
/// split costs one D2H per block and nothing else.
pub const Kind = enum { cpu, cuda };
