// caps.cu -- CUDA device capability probe, the single source of GPU hardware
// facts for src/kernels/cuda (mirror of src/kernels/x86/cpu_caps.c).
//
// The facts are probed ONCE, lazily, on the first q35_gpu_cc() /
// q35_gpu_sm_count() / q35_gpu_name() call; every later read is a cached
// value, safe to call from a launcher on every call (the probe is one
// cudaGetDeviceProperties, no kernel launch, no device allocation).
//
// Device 0 is the primary context the whole backend uses, so the probe works
// both before and after the backend's own context setup; a failure just
// leaves the facts at zero and callers keep their defaults.
//
// Consumers today: only host-side launchers.  q35_launch_mul_mat_vec
// (mmvq.cu) picks the warp-per-row block shape from the compute capability --
// sm_89 gets the measured-best shape, any other compute capability gets an
// occupancy-formula shape.  No device code lives here, so this TU adds
// nothing to the device image.
#include "api.h"
#include <cuda_runtime.h>
#include <stdio.h>
#include <string.h>

struct q35GpuCaps {
    int cc_major;             // 0/0 = probe failed
    int cc_minor;
    int sm_count;
    int max_threads_per_block;
    int max_threads_per_sm;
    int max_blocks_per_sm;
    /// `cudaDevAttrCanMapHostMemory`: 能否把 `cudaHostAlloc(..., Mapped)` 的
    /// 主机内存映射进 CUDA 地址空间（`cudaHostGetDevicePointer`）。
    /// 架构要求是 compute capability >= 1.1，2060（7.5）与 4090（8.9）都满足；
    /// 探测而不是假定的理由是不支持的设备上 `HostGetDevicePointer` 会失败，
    /// 而那发生在一次分配里，报错位置离真正的原因很远。
    int can_map_host;
    char name[256];
};
static q35GpuCaps q35_caps;
static int q35_caps_state = 0;  // 0 = not probed, 1 = probed (result above)

static void q35_caps_probe(void) {
    if (q35_caps_state) return;
    q35_caps_state = 1;
    cudaDeviceProp p;
    if (cudaGetDeviceProperties(&p, 0) != cudaSuccess) {
        // No usable device: report "unknown", and clear the sticky error so
        // the next real launch's error check is clean.
        cudaGetLastError();
        memset(&q35_caps, 0, sizeof(q35_caps));
        return;
    }
    q35_caps.cc_major = p.major;
    q35_caps.cc_minor = p.minor;
    q35_caps.sm_count = p.multiProcessorCount;
    q35_caps.max_threads_per_block = p.maxThreadsPerBlock;
    q35_caps.max_threads_per_sm = p.maxThreadsPerMultiProcessor;
    q35_caps.max_blocks_per_sm = p.maxBlocksPerMultiProcessor;
    q35_caps.can_map_host = p.canMapHostMemory;
    snprintf(q35_caps.name, sizeof(q35_caps.name), "%s", p.name);
}

// Compute capability as major*100 + minor (809 == sm_89, 900 == sm_90).
// -1 when the probe found no device.
extern "C" int q35_gpu_cc(void) {
    q35_caps_probe();
    if (q35_caps.cc_major == 0 && q35_caps.cc_minor == 0) return -1;
    return q35_caps.cc_major * 100 + q35_caps.cc_minor;
}

// SM count of device 0; 0 when the probe found no device.
extern "C" int q35_gpu_sm_count(void) {
    q35_caps_probe();
    return q35_caps.sm_count;
}

// 1 when this device can map host memory (`cudaHostAlloc` + `cudaHostGetDevicePointer`),
// 0 otherwise or when the probe found no device.  See `q35GpuCaps.can_map_host`.
extern "C" int q35_gpu_can_map_host(void) {
    q35_caps_probe();
    return q35_caps.can_map_host;
}

// Device name of device 0 ("unknown" when the probe found no device).
extern "C" const char *q35_gpu_name(void) {
    q35_caps_probe();
    return q35_caps.name[0] ? q35_caps.name : "unknown";
}
