// xd_cpu.cpp -- the CPU-only build's copy of the explicit-graph kernel table.
//
// WHY THIS FILE EXISTS.  The table lives in `xd.cu`, whose other half is CUDA:
// it launches device kernels through their function pointers.  A build without
// `-Dcuda=true` has no CUDA headers on the include path and must not link any
// `__global__` symbol, so `xd.cu` is only ever compiled by nvcc.  This shim is
// the same source with the kernel half switched off by `Q35_XD_NO_KERNELS`; what
// remains is the ABI, the op roster and the argument constructors, which the
// Zig module needs in every build (see src/xd.zig).
//
// It is a `.cpp` (not a `.cu`) on purpose: Zig's own C++ front end picks the
// language from the extension, and a `.cu` file makes it take the CUDA path,
// which needs the toolkit's runtime wrapper.
//
// Keep this file trivial: its only job is the `#include` and the `#define`, so
// there is exactly one copy of the table's source.
#define Q35_XD_NO_KERNELS 1
#include "xd.cu"
