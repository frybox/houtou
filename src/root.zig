//! houtou: a Zig inference engine for frontier open models.
//!
//! Module root. Everything public here is reachable as `@import("houtou").xxx`.

const std = @import("std");

pub const gguf = @import("gguf.zig");
pub const backend = @import("backend.zig");
pub const backend_cpu = @import("backend_cpu.zig");
pub const backend_cuda = @import("backend_cuda.zig");
pub const cudart = @import("cudart.zig");
pub const tokenizer = @import("tokenizer.zig");
/// Per-architecture inference engines.  Each one owns the layer stack of one
/// model family and shares the rest (backends, GGUF, sampling) with the others.
pub const engines = struct {
    /// Architecture-neutral driver code (the generation loop and the narrow
    /// contract each engine exposes to it).
    pub const common = @import("engines/common.zig");
    pub const qwen35 = @import("engines/qwen35.zig");
    pub const qwen3 = @import("engines/qwen3.zig");
};
/// Historical alias: before the `engines/` split there was exactly one engine,
/// and `main.zig` and the tests still reach it as `qwen35.engine`.  Kept so the
/// move did not have to touch every call site at once.
pub const engine = engines.qwen35;
pub const quant_dequant = @import("quant_dequant.zig");
pub const pool = @import("pool.zig");
pub const dspine_bridge = @import("dspine_bridge.zig");
pub const draft = @import("draft.zig");
pub const spec_rollback = @import("spec_rollback.zig");
pub const serial = @import("serial.zig");

// The IQ lookup tables are defined here once and consumed by the C kernels via
// `extern const` (src/kernels/x86/iq_tables.h).  Importing them from the root
// module guarantees the symbols are emitted into the final binary regardless of
// which Zig code path reaches them.
pub const iq_tables = @import("iq_tables.zig");

test {
    std.testing.refAllDecls(@This());
}

test {
    _ = @import("iq_table_tests.zig");
}

// C-kernel parity coverage (restored after `33e2bb8` unmounted them).
// Each file compares the C kernels (`simd_c.zig`, extern symbols from
// src/kernels/x86/) against the Zig scalar oracle (`simd_ref.zig`), and against
// the engine's own `rowDot*` on real packed rows from testdata/udtest.
// This is the only automated guard for the AGENTS.md §4 symbol-table hazard
// (`ksigns_iq2xs` vs `keven_signs_q2xs`): no test catches a swapped table if
// these files are unmounted.
test {
    _ = @import("simd_tests32.zig");
}

// Chunk-wide prefill slots (`attnChunk`, `deinterleaveQGStrided`): each is
// checked against the per-row calls it replaced, which is the whole contract of
// a launch merge.  See the file header for why an off-by-one row is the failure
// mode worth a test.
test {
    _ = @import("chunk_slot_tests.zig");
}

// Speculative-decoding rollback (DSpine port, stage 2): the GDN/conv snapshot +
// replay primitive, pinned to "restore + replay(committed) is bit-identical to a
// fresh run of only those rows".  That equivalence is the entire contract the
// engine relies on, and an off-by-one in `committed` is its failure mode.
test {
    _ = @import("spec_rollback.zig");
    _ = @import("draft.zig");
    _ = @import("spec_ref.zig");
    _ = @import("dspine_tests.zig");
    _ = @import("serial_tests.zig");
    // The Qwen3 engine's own checks: the architecture gate, the full-head
    // rotary span, and that `applyRope` really rotates past 64 dims (the one
    // difference from Qwen3.5 that `rope64Rows` cannot express).
    _ = @import("engines/common.zig");
    _ = @import("engines/qwen3.zig");
}

test {
    _ = @import("simd_tests256.zig");
}

// KV-cache type support (`--kv-type f32|f16|q8_0|q4_0`): the C row
// quantize/dequant kernels (src/kernels/x86/kv.c) are checked against the
// weight-side oracles in quant_dequant.zig (byte-exact where the pair was
// written expression-for-expression, bit-exact dequant on shared bytes), and
// the dequant-on-read attention path (`backend_cpu.attnDecodeKv`) against the
// f32 reference within each format's element error bound.
test {
    _ = @import("kv_tests.zig");
}

test {
    _ = @import("simd_tests_iq_a.zig");
}

test {
    _ = @import("simd_tests_iq_b.zig");
}

test {
    _ = @import("simd_tests_iq_c.zig");
    _ = @import("simd_tests_f32.zig");
}

test {
    _ = @import("backend_cpu.zig");
    // Referenced so the trampoline is type-checked by `zig build test` even
    // before anything selects `.cuda`; the CUDA backend has no test of its own
    // (it is a pure forwarding layer) so this is its only compile-time guard.
    _ = @import("backend_cuda.zig");
}

test {
    // `Launch` (backend_cuda.zig) is a two-branch comptime table: real aliases vs
    // fallback stubs.  Only one branch exists per build, so nothing can notice the
    // lists drifting -- and a call site that names `Kernels.q35_launch_*` instead
    // of `Launch.*` bypasses the seam and breaks `make cpu` at link time.  Both
    // happened; these are the structural checks that make them fail a test run.
    _ = @import("backend_cuda_launch_tests.zig");
}

test {
    // `engine.refErrF64` (the f64 matvec oracle) is called by nothing in the
    // engine, and Zig analyses unreferenced functions lazily -- so `zig build
    // test` never even type-checked it.  Its Q8_0 branch called
    // `q.row_blocks()`, a method that does not exist (`QuantRow.rowBlocks`), i.e.
    // the oracle for the Q8_0 cross-check did not compile.  This file references
    // it, which is the guard: adding a ttype without registering its row fn now
    // fails a build instead of silently returning -2.
    _ = @import("referr_tests.zig");
}

test {
    // `elementwise.zig` is the slot body of the elementwise kernels; import it
    // here so its tests (notably the `bufDeinterleaveQG` oracle comparison,
    // which pins the batched q/gate split to the per-head `copyRange` sequence
    // it replaced) actually run.  Without this line the file's `test` blocks
    // are dead code: nothing else imports it as a module.
    _ = @import("elementwise.zig");
}

