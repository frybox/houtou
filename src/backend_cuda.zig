//! GPU 后端：前向通路里的每一个数值 kernel 都在设备上跑。
//!
//! 为什么要有这个文件（而不是做一个 `backend_cpu` 的 CUDA 版本）：同一个
//! `zig-out/bin/qwen35` 二进制必须能在**没有装 CUDA** 的主机上跑起来。两套
//! 机制共同保证这一点，二者不能混为一谈：
//!
//!   * **kernel 是链接进来的。** `src/kernels/cuda/*.cu` 由
//!     `zig build -Dcuda=true` 编成目标文件，链接进可执行文件；下面每个入口
//!     都是一个顶层的 `extern fn q35_launch_*`，由链接器在构建期解析。这里
//!     没有 kernel 的 `.so`、没有 `dlopen`、也没有运行期符号查找。纯 CPU 构建
//!     根本不会引用它们（没有东西调用这个后端，见 `main.zig` 的 `--backend`
//!     解析），那些目标文件也不在链接命令行上。
//!   * **libcudart 是 dlopen 的**（`cudart.zig`），所以纯 CPU 主机从不需要对它
//!     的 DT_NEEDED。那是唯一剩下的运行期库依赖，而 `--backend cpu` 在任何
//!     没装 CUDA 的主机上照样能跑。
//!
//! 与 `backend_cpu` 的分工（刻意如此，见 AGENTS.md §1.1）：
//!
//!   * 内嵌的 `CpuBackend` 仍然是**元数据**工作的权威：解析 GGUF 张量、构建
//!     `QuantRow` 视图、以及它在加载期校验的几何。那是发生在 mmap 上的启动
//!     期工作，不是前向计算。只有 `.f32` 张量会被拓宽成 f32；f16/bf16 权重
//!     保持它们原生的 2 字节形态（见 `backend_cpu.zig` 的 `loadLinear`），按
//!     每元素 2 B 上传，在 kernel 内部由 `mmvq.cu` 的 w16 系列拓宽。
//!   * 引擎要求的每一个**数值**操作 —— matvec、K-token 批量 matvec、按行
//!     反量化、以及它们需要的激活 scratch —— 都由 `src/kernels/cuda` 里的
//!     kernel 在设备上完成。这里**没有** CPU 算术兜底：按张量混用后端会让
//!     数值分歧无法归因，也让每次 profile 失去意义。
//!
//! 权重驻留：`loadLinear` 上传 CPU 加载器接受的所有 `LinearW.quant` 类型
//! （含 f16/bf16，按 2 B/elem）以及 `.f32` 张量，并把设备指针记进
//! `dev_rows`。上传失败是**硬性**加载错误，绝不会静默退回 CPU ——
//! `--backend cuda` 必须名副其实。
//!
//! 内存预算：后端不去"智胜"驱动。`cudaMalloc` 失败时会连同张量名、类型、
//! 字节数和设备剩余空闲内存（`cudaMemGetInfo`）一起报出来，然后从
//! `loadLinear` 转成 `error.OutOfMemory`，于是加载中止，而不是继续走一条
//! CPU/GPU 混合路径（既慢，归因又静默地不同）。装不下的模型需要
//! `--backend cpu` 或更小的量化；`$QWEN35_CUDA_MAX_DEV_MIB` 给上传设上限，
//! 让失败尽早发生、并可归因到预算而不是某次瞬时分配。

const std = @import("std");
const backend = @import("backend.zig");
const graph_registry = @import("graph_registry.zig");
const xd = @import("xd.zig");
const cpu_mod = @import("backend_cpu.zig");
const gguf = @import("gguf.zig");
const cudart = @import("cudart.zig");
const sampler = @import("sampler.zig");

/// `build.zig` 注入的构建期事实（`addOptions("build_options")`）。
/// `cuda_linked` 就是那个控制 "`src/kernels/cuda/*.cu` 是否进可执行文件的链接
/// 命令行" 的 `-Dcuda` 值。
const build_options = @import("build_options");

const mem = std.mem;
const Io = std.Io;

// ==== 链接期的 kernel 绑定 ===================================================
//
// src/kernels/cuda/api.h 里每个入口对应一个 `extern`，名字与签名照抄头文件。
// `zig build -Dcuda=true` 把那些 `.cu` 目标文件放上链接命令行，所以缺一个或
// 签名对不上的 launcher 是**构建期**错误，而不是运行期的 `MissingSymbol` ——
// 这正是"链接进来"而非"dlopen 一个 .so"的全部意义。运行期无法重新绑定，因此
// 这些声明必须手工跟随 api.h；下面的注释复述了头文件的契约，好让任何一边的
// 改动在另一边看得见。
//
// ABI 对应关系：`int` -> `c_int`，`const void *` -> `?*const anyopaque`，
// `void *` -> `?*anyopaque`，`float *` -> `?[*]f32`，`const float *` ->
// `?[*]const f32`，`unsigned long long` -> `c_ulonglong`。
//
// 调用点统一走下面的 `Kernels` 命名空间（`const kern = Launch;`），纯粹是为了
// 让发射点读起来仍是 `kern.mul_mat_vec(...)`；它不持有状态，编译成直接调用。
const Kernels = struct {
    /// `nblocks` 以 **32 元素**块为单位计数（api.h 的 Q35_API_QK8_0）。
    extern fn q35_launch_quantize_q8_0(d_x: ?[*]const f32, d_out: ?*anyopaque, nblocks: c_int, stream: ?*anyopaque) callconv(.c) c_int;
    /// `nblocks` 以 **256 元素**块为单位计数。
    extern fn q35_launch_quantize_q8_K(d_x: ?[*]const f32, d_out: ?*anyopaque, nblocks: c_int, stream: ?*anyopaque) callconv(.c) c_int;
    /// 同上，另外还写出 dp4a prefill kernel 读取的 `block_q8_1_mmq` 重排
    /// （每 256 元素块 288 B）。见 mmvq.cu 的 `q35_k_quantize_q8_K`。
    extern fn q35_launch_quantize_q8_K_mmq(d_x: ?[*]const f32, d_out: ?*anyopaque, d_mmq: ?*anyopaque, nblocks: c_int, stream: ?*anyopaque) callconv(.c) c_int;
    /// 该类型在 prefill 下的 matvec 走 dp4a kernel 时返回 1 —— 那条路需要 mmq
    /// 激活布局，而不是 q8_K。
    extern fn q35_mmq_dp4a_wants(ttype: c_int) c_int;
    /// mmq 激活布局下每个 256 元素块的字节数（288）。
    extern fn q35_mmq_act_block_bytes() c_int;

    /// 整张量反量化；`nblocks` 按该类型**原生**块计数。
    extern fn q35_launch_dequantize(ttype: c_int, d_src: ?*const anyopaque, d_dst: ?[*]f32, nblocks: c_int, stream: ?*anyopaque) callconv(.c) c_int;
    /// 一行，共 `cols` 个元素。行偏移由调用方在设备侧做指针算术完成（见
    /// `vDequantRow`），所以这里只需要 `cols`。
    extern fn q35_launch_dequantize_row(ttype: c_int, d_src_row: ?*const anyopaque, d_dst: ?[*]f32, cols: c_int, stream: ?*anyopaque) callconv(.c) c_int;
    /// 同上，但行号从每步控制块读（`ctl[XD_CTL_TOKEN]`）。MTP 草稿块的嵌入用：
    /// 那一行的行号每轮都变，不能作为 kernel 实参录进图。
    extern fn q35_launch_dequantize_row_dtoken(ttype: c_int, d_src_base: ?*const anyopaque, d_dst: ?[*]f32, cols: c_int, row_bytes: c_int, d_ctl: ?*const anyopaque, stream: ?*anyopaque) callconv(.c) c_int;

    /// 表驱动的量化 matvec：`d_out[t*rows + r] = W[r] . act_t`。
    /// 实参顺序照 api.h：rows、cols、tokens，**然后**才是 ttype。
    extern fn q35_launch_mul_mat_vec(d_w: ?*const anyopaque, d_xq: ?*const anyopaque, d_out: ?[*]f32, rows: c_int, cols: c_int, tokens: c_int, ttype: c_int, stream: ?*anyopaque) callconv(.c) c_int;
    /// f32 权重矩阵（embedding / norm）的同形状版本。
    extern fn q35_launch_mul_mat_vec_f32(d_w: ?[*]const f32, d_x: ?[*]const f32, d_out: ?[*]f32, rows: c_int, cols: c_int, tokens: c_int, stream: ?*anyopaque) callconv(.c) c_int;
    /// f16/bf16 权重保持原生 2 字节形态，乘 f32 激活。
    /// `d_w` 是 rows*cols 个 u16（每元素 2 B）—— 设备上**没有** f32 副本，反量化
    /// 在 kernel 内部完成（mmvq.cu 的 w16 头部）。
    extern fn q35_launch_mul_mat_vec_w16(d_w: ?*const anyopaque, d_x: ?[*]const f32, d_out: ?[*]f32, rows: c_int, cols: c_int, tokens: c_int, ttype: c_int, stream: ?*anyopaque) callconv(.c) c_int;

    /// 把十二张 IQ 码本上传到设备（src/kernels/cuda/iq_tables.cu）。在任何触及
    /// IQ 类型的 kernel 之前**必须**成功：那些 kernel 通过只有本调用才会填的
    /// `__device__` 指针读表，所以漏掉或失败的上传会让 IQ matvec 索引到垃圾
    /// （输出静默错误，而不是崩溃）。幂等；返回 cudaError_t。
    extern fn q35_launch_init_tables(stream: ?*anyopaque) callconv(.c) c_int;
    /// 释放码本的设备副本（进程级，寿命与后端相同）。
    extern fn q35_launch_deinit_tables() callconv(.c) void;
    /// 驻留在设备上的码本字节数（成功上传之前为 0）。
    extern fn q35_iq_tables_bytes() callconv(.c) c_ulonglong;

    /// 纯 elementwise 激活（src/kernels/cuda/elementwise.cu）。`n` 是 f32 的
    /// **元素**个数，缓冲区连续，这四个都与 CPU 实现逐位相同 —— 为什么必须保持
    /// 这一点见该文件的注释。
    extern fn q35_launch_scale_vec(d_x: ?[*]f32, s: f32, n: c_int, stream: ?*anyopaque) callconv(.c) c_int;
    extern fn q35_launch_l2norm_qk_scale(d_x: ?[*]f32, nk: c_int, kd: c_int, k_offset: c_int, eps: f32, qscale: f32, stream: ?*anyopaque) callconv(.c) c_int;
    extern fn q35_launch_add_resid(d_dst: ?[*]f32, d_src: ?[*]const f32, n: c_int, stream: ?*anyopaque) callconv(.c) c_int;
    extern fn q35_launch_copy_range(d_dst: ?[*]f32, d_src: ?[*]const f32, n: c_int, stream: ?*anyopaque) callconv(.c) c_int;
    extern fn q35_launch_deinterleave_qg(d_q: ?[*]f32, d_gate: ?[*]f32, d_src: ?[*]const f32, heads: c_int, hd: c_int, stream: ?*anyopaque) callconv(.c) c_int;
    extern fn q35_launch_deinterleave_qg_strided(d_q: ?[*]f32, d_gate: ?[*]f32, d_src: ?[*]const f32, heads: c_int, hd: c_int, tokens: c_int, stream: ?*anyopaque) callconv(.c) c_int;
    extern fn q35_launch_zero_buf(d_buf: ?[*]f32, n: c_int, stream: ?*anyopaque) callconv(.c) c_int;

    /// Norm（src/kernels/cuda/norm.cu）、激活 + conv（src/kernels/cuda/act.cu）
    /// 以及 rotary（src/kernels/cuda/rope.cu）。
    ///
    /// `rows` 个 `n` 元素的 f32 块；`d_w`/`d_wconv` 是长度 `n` 的逐层权重，每一行
    /// 复用同一份。这些**都不是**与 CPU 逐位相同的：归约用块树，超越函数来自
    /// 设备运行时而非 glibc。逐 kernel 的实测误差见 api.h 与各文件头部。
    extern fn q35_launch_rmsnorm(d_out: ?[*]f32, d_x: ?[*]const f32, d_w: ?[*]const f32, rows: c_int, n: c_int, eps: f32, stream: ?*anyopaque) callconv(.c) c_int;
    extern fn q35_launch_l2norm(d_x: ?[*]f32, rows: c_int, n: c_int, eps: f32, stream: ?*anyopaque) callconv(.c) c_int;
    extern fn q35_launch_silu(d_x: ?[*]f32, n: c_int, stream: ?*anyopaque) callconv(.c) c_int;
    extern fn q35_launch_mul_silu(d_out: ?[*]f32, d_gate: ?[*]const f32, n: c_int, stream: ?*anyopaque) callconv(.c) c_int;
    extern fn q35_launch_mul_sigmoid(d_out: ?[*]f32, d_gate: ?[*]const f32, n: c_int, stream: ?*anyopaque) callconv(.c) c_int;
    extern fn q35_launch_swiglu(d_gate: ?[*]f32, d_up: ?[*]const f32, n: c_int, stream: ?*anyopaque) callconv(.c) c_int;
    extern fn q35_launch_softmax(d_x: ?[*]f32, n: c_int, stream: ?*anyopaque) callconv(.c) c_int;
    extern fn q35_launch_conv_step(d_conv: ?[*]f32, d_y: ?[*]f32, d_wconv: ?[*]const f32, n: c_int, stream: ?*anyopaque) callconv(.c) c_int;
    /// 最大 logit 的下标；并列时取第一个下标。
    extern fn q35_launch_argmax(d_x: ?[*]const f32, n: c_int, d_out: ?*c_int, stream: ?*anyopaque) callconv(.c) c_int;
    extern fn q35_launch_argmax_batch(d_x: ?[*]const f32, n: c_int, rows: c_int, d_out: ?[*]c_int, stream: ?*anyopaque) callconv(.c) c_int;
    /// `argmax` 之外还给出胜出者的 logit 与该行的 `sum(exp(x - max))`，供 MTP
    /// 草稿头的置信度提前停止使用（见 api.h）。
    extern fn q35_launch_argmax_conf(d_x: ?[*]const f32, n: c_int, d_out: ?[*]c_int, stream: ?*anyopaque) callconv(.c) c_int;

    // 设备采样器（sample.cu）。逐 token 的参数 —— temperature、top_k、top_p、
    // PRNG 状态 —— 经 `d_ctl` 传递：一块**常驻**设备控制块，每次重放之前由一个
    // 64 字节 H2D 重写，所以捕获下来的图节点永远不需要重定向它的参数。
    // `n`（vocab）是发射实参，因为它决定网格形状且从不改变。
    extern fn q35_launch_sample(d_logits: ?[*]const f32, n: c_int, d_ctl: ?*c_int, d_scratch: ?*anyopaque, stream: ?*anyopaque) callconv(.c) c_int;
    extern fn q35_launch_sample_init(d_ctl: ?*?*anyopaque, d_scratch: ?*?*anyopaque) callconv(.c) c_int;
    extern fn q35_launch_sample_deinit(d_ctl: ?*?*anyopaque, d_scratch: ?*?*anyopaque) callconv(.c) void;
    extern fn q35_sample_scratch_ints() callconv(.c) c_int;
    extern fn q35_sample_ctl_ints() callconv(.c) c_int;

    // P2c：两个状态机。gdn_step 每个 value head 一次发射，attn_decode 一次发射
    // （每 head 一个 block）；见各自的 kernel 头部。
    extern fn q35_launch_gdn_step(d_state: ?[*]f32, d_k: ?[*]const f32, d_v: ?[*]const f32, d_q: ?[*]const f32, d_km: ?[*]f32, d_delta: ?[*]f32, d_out: ?[*]f32, gamma: f32, beta: f32, vd: c_int, stream: ?*anyopaque) callconv(.c) c_int;
    extern fn q35_launch_gdn_step_rows(d_state: ?[*]f32, state_stride: c_int, d_k: ?[*]const f32, k_stride: c_int, d_v: ?[*]const f32, v_stride: c_int, d_q: ?[*]const f32, q_stride: c_int, d_km: ?[*]f32, d_delta: ?[*]f32, d_out: ?[*]f32, d_gamma: ?[*]const f32, d_beta: ?[*]const f32, heads: c_int, nk: c_int, kd: c_int, vd: c_int, stream: ?*anyopaque) callconv(.c) c_int;
    extern fn q35_launch_gdn_step_rows_chunk(d_state: ?[*]f32, state_stride: c_int, d_k: ?[*]const f32, k_stride: c_int, d_v: ?[*]const f32, v_stride: c_int, d_q: ?[*]const f32, q_stride: c_int, d_km: ?[*]f32, km_stride: c_int, d_delta: ?[*]f32, delta_stride: c_int, d_out: ?[*]f32, out_stride: c_int, d_gamma: ?[*]const f32, gamma_stride: c_int, d_beta: ?[*]const f32, beta_stride: c_int, heads: c_int, nk: c_int, kd: c_int, vd: c_int, rows: c_int, stream: ?*anyopaque) callconv(.c) c_int;
    extern fn q35_launch_gdn_step_wy(d_state: ?[*]f32, state_stride: c_int, d_k: ?[*]const f32, k_stride: c_int, d_v: ?[*]const f32, v_stride: c_int, d_q: ?[*]const f32, q_stride: c_int, d_km: ?[*]f32, km_stride: c_int, d_delta: ?[*]f32, delta_stride: c_int, d_out: ?[*]f32, out_stride: c_int, d_gamma: ?[*]const f32, gamma_stride: c_int, d_beta: ?[*]const f32, beta_stride: c_int, heads: c_int, nk: c_int, kd: c_int, vd: c_int, rows: c_int, stream: ?*anyopaque) callconv(.c) c_int;
    extern fn q35_launch_gdn_step_rows_snap(d_state: ?[*]f32, state_stride: c_int, d_k: ?[*]const f32, k_stride: c_int, d_v: ?[*]const f32, v_stride: c_int, d_q: ?[*]const f32, q_stride: c_int, d_km: ?[*]f32, km_stride: c_int, d_delta: ?[*]f32, delta_stride: c_int, d_out: ?[*]f32, out_stride: c_int, d_gamma: ?[*]const f32, gamma_stride: c_int, d_beta: ?[*]const f32, beta_stride: c_int, d_snap: ?[*]f32, nsnap: c_int, heads: c_int, nk: c_int, kd: c_int, vd: c_int, rows: c_int, stream: ?*anyopaque) callconv(.c) c_int;
    extern fn q35_launch_gdn_gamma(d_a_raw: ?[*]const f32, d_b_raw: ?[*]f32, d_ssm_a: ?[*]const f32, d_dt_bias: ?[*]const f32, d_gamma: ?[*]f32, nv: c_int, stream: ?*anyopaque) callconv(.c) c_int;

    // conv_step / gdn_gamma / l2norm / scale_vec 的 prefill（批量）孪生版本。
    // 它们存在只为把逐 token 的 GDN 前奏压缩成每块少数几次发射 —— 它们**不属于**
    // decode 图（prefill 从不捕获），所以是直接调用、刻意不放进 `Launch`。
    // 每一个都与它所替代的逐 token kernel 逐位相同。
    extern fn q35_launch_conv_step_rows(d_conv: ?[*]f32, d_y: ?[*]f32, d_wconv: ?[*]const f32, n: c_int, tokens: c_int, y_stride: c_int, stream: ?*anyopaque) callconv(.c) c_int;
    extern fn q35_launch_gdn_gamma_rows(d_a_raw: ?[*]const f32, d_b_raw: ?[*]f32, d_ssm_a: ?[*]const f32, d_dt_bias: ?[*]const f32, d_gamma: ?[*]f32, nv: c_int, tokens: c_int, a_stride: c_int, b_stride: c_int, gamma_stride: c_int, stream: ?*anyopaque) callconv(.c) c_int;
    extern fn q35_launch_l2norm_strided(d_x: ?[*]f32, row_offset: c_int, rows: c_int, n: c_int, eps: f32, tokens: c_int, row_stride: c_int, stream: ?*anyopaque) callconv(.c) c_int;
    extern fn q35_launch_scale_vec_strided(d_x: ?[*]f32, offset: c_int, s: f32, n: c_int, tokens: c_int, stride: c_int, stream: ?*anyopaque) callconv(.c) c_int;
    extern fn q35_launch_attn_decode(d_q: ?[*]const f32, d_gate: ?[*]const f32, d_kcache: ?[*]const f32, d_vcache: ?[*]const f32, d_att: ?[*]f32, d_gout: ?[*]f32, slot: c_int, heads: c_int, gqa: c_int, hd: c_int, kv_dim: c_int, scale: f32, kv_kind: c_int, stream: ?*anyopaque) callconv(.c) c_int;
    extern fn q35_launch_attn_chunk(d_q: ?[*]const f32, d_gate: ?[*]const f32, d_kcache: ?[*]const f32, d_vcache: ?[*]const f32, d_gout: ?[*]f32, slot0: c_int, rows: c_int, heads: c_int, gqa: c_int, hd: c_int, kv_dim: c_int, scale: f32, kv_kind: c_int, stream: ?*anyopaque) callconv(.c) c_int;
    extern fn q35_attn_chunk_max_rows(slot0: c_int, hd: c_int, kv_kind: c_int) callconv(.c) c_int;
    /// warp 分块 / 切分的 decode attention（实测数字见 api.h）。
    extern fn q35_launch_attn_decode_coal(d_q: ?[*]const f32, d_gate: ?[*]const f32, d_kcache: ?[*]const f32, d_vcache: ?[*]const f32, d_att: ?[*]f32, d_gout: ?[*]f32, slot: c_int, heads: c_int, gqa: c_int, hd: c_int, kv_dim: c_int, scale: f32, kv_kind: c_int, stream: ?*anyopaque) callconv(.c) c_int;
    extern fn q35_launch_attn_decode_split_dyn(d_q: ?[*]const f32, d_gate: ?[*]const f32, d_kcache: ?[*]const f32, d_vcache: ?[*]const f32, d_att: ?[*]f32, d_gout: ?[*]f32, d_part: ?[*]f32, d_slot: ?[*]const c_int, heads: c_int, gqa: c_int, hd: c_int, kv_dim: c_int, scale: f32, split: c_int, kv_kind: c_int, stream: ?*anyopaque) callconv(.c) c_int;
    extern fn q35_func_attn_score_split_dyn() callconv(.c) ?*anyopaque;
    extern fn q35_func_attn_pv_split_dyn() callconv(.c) ?*anyopaque;
    extern fn q35_func_attn_softmax_dyn() callconv(.c) ?*anyopaque;
    extern fn q35_func_attn_reduce() callconv(.c) ?*anyopaque;
    extern fn q35_launch_attn_decode_split2(d_q: ?[*]const f32, d_gate: ?[*]const f32, d_kcache: ?[*]const f32, d_vcache: ?[*]const f32, d_att: ?[*]f32, d_gout: ?[*]f32, d_part: ?[*]f32, slot: c_int, heads: c_int, gqa: c_int, hd: c_int, kv_dim: c_int, scale: f32, split: c_int, kv_kind: c_int, stream: ?*anyopaque) callconv(.c) c_int;
    /// 同一 kernel 的"设备 `slot`"形态；见 `dAttnDecode_body`。
    extern fn q35_launch_attn_decode_dyn(d_q: ?[*]const f32, d_gate: ?[*]const f32, d_kcache: ?[*]const f32, d_vcache: ?[*]const f32, d_att: ?[*]f32, d_gout: ?[*]f32, d_slot: ?[*]const c_int, heads: c_int, gqa: c_int, hd: c_int, kv_dim: c_int, scale: f32, kv_kind: c_int, stream: ?*anyopaque) callconv(.c) c_int;
    /// 用单线程 kernel 写一个 int32（图安全的逐 token 标量）。
    extern fn q35_launch_set_int1(d: ?[*]c_int, v: c_int, stream: ?*anyopaque) callconv(.c) c_int;
    extern fn q35_launch_copy_range_dslot(dst_base: ?[*]f32, stride: c_int, src: ?[*]const f32, rows: c_int, n: c_int, ctl: ?[*]const c_int, stream: ?*anyopaque) callconv(.c) c_int;
    extern fn q35_launch_xd_ctl_publish(d_ctl: ?*anyopaque, pos: c_int, base_bits: c_int, kv_slot: c_int, attn_slot: c_int, rows: c_int, stream: ?*anyopaque) callconv(.c) c_int;
    /// 对 `heads` 个长度为 `hd` 的头做 rope64（只有前 64 个分量旋转）。
    extern fn q35_launch_rope64(d_x: ?[*]f32, heads: c_int, hd: c_int, pos: c_int, base: f32, stream: ?*anyopaque) callconv(.c) c_int;
    extern fn q35_launch_rope64_dpos(d_x: ?[*]f32, heads: c_int, hd: c_int, d_dyn: ?*const anyopaque, stream: ?*anyopaque) callconv(.c) c_int;
    extern fn q35_launch_rope64_setenv(d_dyn: ?*anyopaque, pos: c_int, base: f32, stream: ?*anyopaque) callconv(.c) c_int;
    /// 按行批量的 rope64：`rows` 个 token（`pos0..pos0+rows-1`），每个宽
    /// `heads * hd` 个 float。**仅 prefill 使用**（见 `VTable.rope64Rows` 的说明）。
    extern fn q35_launch_rope64_rows(d_x: ?[*]f32, rows: c_int, heads: c_int, hd: c_int, pos0: c_int, base: f32, stream: ?*anyopaque) callconv(.c) c_int;
    /// 转子嵌入，旋转宽度运行期可设（Qwen3 旋转整个头；Qwen3.5/3.8 旋 256 中的
    /// 64 个）。见 `api.h`。
    extern fn q35_launch_rope_rows_dim(d_x: ?[*]f32, rows: c_int, heads: c_int, hd: c_int, pos0: c_int, base: f32, rotary_dim: c_int, stream: ?*anyopaque) callconv(.c) c_int;
    extern fn q35_launch_rope64_rows_dpos(d_x: ?[*]f32, rows: c_int, heads: c_int, hd: c_int, d_dyn: ?*const anyopaque, stream: ?*anyopaque) callconv(.c) c_int;
    extern fn q35_func_rope64_rows_dpos() callconv(.c) ?*anyopaque;
    /// `q35_launch_rope64` 所发射的那个设备 kernel 的 host-stub 地址。捕获下来
    /// 的图节点会把这个值原样报在 `KernelNodeParams.func` 里，所以拿它做比较
    /// 就是 decode 图路径"找出每 token 需要更新 `pos` 的那个节点"的办法。
    /// 改读实参值并不等价：节点的 `kernelParams` 数组没有长度，探测它会读过
    /// 实参较少的 kernel 的末尾并触发段错误（实测）。
    extern fn q35_func_rope64() callconv(.c) ?*anyopaque;
    extern fn q35_func_rope64_dpos() callconv(.c) ?*anyopaque;
    extern fn q35_func_attn_decode() callconv(.c) ?*anyopaque;
    extern fn q35_func_attn_decode_dyn() callconv(.c) ?*anyopaque;
    extern fn q35_func_copy_range() callconv(.c) ?*anyopaque;
    extern fn q35_launch_kv_quant(d_x: ?[*]const f32, d_out: ?*anyopaque, nblocks: c_int, kind: c_int, stream: ?*anyopaque) callconv(.c) c_int;
    extern fn q35_func_kv_quant() callconv(.c) ?*anyopaque;
    extern fn q35_launch_attn_chunk_dyn(d_q: ?[*]const f32, d_gate: ?[*]const f32, d_kcache: ?[*]const f32, d_vcache: ?[*]const f32, d_gout: ?[*]f32, d_slot0: ?[*]const c_int, rows: c_int, heads: c_int, gqa: c_int, hd: c_int, kv_dim: c_int, scale: f32, kv_kind: c_int, stream: ?*anyopaque) callconv(.c) c_int;
    extern fn q35_attn_chunk_dyn_smem(hd: c_int, kv_kind: c_int) callconv(.c) c_ulonglong;
    extern fn q35_launch_kv_quant_dslot(d_x: ?[*]const f32, d_out_base: ?*anyopaque, d_slot: ?*const c_int, nblocks_per_row: c_int, rows_this: c_int, row_bytes: c_int, kind: c_int, stream: ?*anyopaque) callconv(.c) c_int;
    extern fn q35_func_kv_quant_dslot() callconv(.c) ?*anyopaque;
};

/// decode attention 的行区间切分因子。`0` 关闭（每 head 一个 block，2026-11 之前
/// 的行为），`2..16` 把每个 head 的**两条**行遍历都摊到这么多 block 上。
///
/// 默认 `16`，因为它是纯收益且不改变任何结果：score 求和保持串行的 `j` 顺序，
/// softmax 保持 CPU 的 max/串行求和/相除顺序，PV 的 t 求和是唯一的二级树 ——
/// 而连它在探针上也是逐位相同的（每个 head、slot 0/5/63/300/516-519/530/700/
/// 1023 对单 block 参照均为 `mismatched=0 worst=0`）。每 head 单 block 在这块
/// 114 SM 的 GPU 上是 24 个 block（21% 占用率），此时每个 lane 要串行走完一个
/// 放不进 L2 的 KV cache 的 `n` 全部行，实测 **33 GB/s**；切分后达到
/// **325 GB/s**。
///
/// `$QWEN35_CUDA_ATTN_SPLIT` 可以覆盖它（置 0 关闭，用于 A/B）。
fn attnSplitFromEnv() usize {
    const v = cudart.getenv("QWEN35_CUDA_ATTN_SPLIT") orelse return 16;
    const n = std.fmt.parseInt(usize, std.mem.trim(u8, v, " \t"), 10) catch return 16;
    return @min(n, 16);
}


// ==== 链接进来 vs 纯 CPU 构建 ===============================================
//
// 上面那些 launcher 只存在于 `-Dcuda=true` 的构建里（正是那种构建把
// `src/kernels/cuda/*.cu` 放上链接命令行的）。普通的 `zig build` 仍然会编译
// 本文件 —— `engine.zig` 会为 `--backend cuda` 无条件实例化 `CudaBackend`，而
// CLI 无论哪种构建都会宣传这个开关 —— 于是在纯 CPU 构建里，每个发射点都会成为
// 未定义符号。
//
// 判别依据是**构建期常量**，不是符号探测：`build.zig` 把 `-Dcuda` 以
// `build_options.cuda_linked` 传给 Zig 代码（与决定 nvcc 目标文件是否交给
// `exe.root_module.addObjectFile` 的是同一个 `cuda_enabled` 变量）。`else`
// 分支是一个 stub，它把情况报出来，而不是调用一片虚无；这个选择是 comptime
// 的，所以纯 CPU 二进制里也不会留下对 launcher 的悬空调用。
//
// 符号存在性探测在这里**不管用**，也不得再引入。
// `@hasDecl(Kernels, "q35_launch_mul_mat_vec")` 看起来能回答"这个链接进来了吗"，
// 其实不能：上面的 `fn` 是 **`Kernels` 结构体内部的声明**，而 `@hasDecl` 只报告
// 那个结构体有没有这个成员。实测（2026-10）：即使 C 定义已经链接进同一个可执行
// 文件，它依然是 `false`，于是 `Launch` 在所有构建里（**包括** `-Dcuda=true`）
// 都塌到 stub 分支 —— `--backend cuda` 打印"这个可执行文件没带 CUDA kernel"，
// 而 `nm` 明明列着九个 `T q35_launch_*`。真正的链接期检查必须去问链接器
// （例如弱引用），而 Zig 在 comptime 不暴露这个能力；构建标志精确、便宜，且与
// 链接本身同源。
const cuda_linked = build_options.cuda_linked;

/// 纯 CPU 构建下每个 stub 报出的原因。刻意指向构建开关，因为那是唯一能修好它
/// 的东西。
const no_cuda_kernels_msg = "cuda backend: this executable was built without CUDA kernels (rebuild with -Dcuda=true, or use --backend cpu)";

fn stubNoCudaKernels() c_int {
    std.debug.print("error: {s}\n", .{no_cuda_kernels_msg});
    // 返回非零（cudaErrorInvalidDeviceFunction），好让调用方的 `rc != cudaSuccess`
    // 分支触发、把消息暴露出来，而不是带着垃圾继续跑。
    return 98;
}

/// 决定 `Launch.*` 调用在图的范围内该做什么的唯一一处。
///
/// 语义只有一条：**`g.ptr` 非空表示"这一步由图在跑，主机侧不要再发射"**。
///
///   off / rec  `dGraphBegin` 返回 `ptr == null` 的直通图 → 这里返回 false，
///              op 照常发射。`rec` 期间这在 CUDA 里是期望行为：capture 会把
///              每一次发射记进图（参照项目同样如此，capture 期没有任何特殊
///              分支）。
///   live       `dGraphBegin` 在 `GraphLaunch` 之后返回带 `ptr` 的图 → 这里
///              返回 true，op 被跳过。不跳过就是在默认流上把整步再算一遍，
///              结果会和图的输出交错。
///
/// 这样就不需要 op 表，也不需要按 op 分别记账：判断依据是"这一 token 的
/// 工作是否已经由图承担"。
/// 当前正在执行的那一层图。它由 **每个 op 的定时包装**（`vMatvec`、
/// `dAddResid` …）在调用 body 之前压入、之后弹回，而不是由"当前 span"全局
/// 决定 —— 后者在嵌套 span（草稿块在 verify 里面）下会拿错槽。
///
/// 为什么需要它：`graphGate` 只能看到 `Graph.replayed`，而 verify 那段是逐
/// kernel 直发（`replayed == false`）。草稿图重放时它的 op 同样是 `false`，
/// 必须靠"槽 2 是 `.live`"来抑制，否则草稿的每一步会在主机上再算一遍。
var xg_cur_slot: u8 = 0;
/// 压入 `slot`，返回先前那个（供 `defer` 恢复）。
inline fn xg_push_slot(slot: u8) u8 {
    const prev = xg_cur_slot;
    xg_cur_slot = slot;
    return prev;
}
var xg_backend: ?*CudaBackend = null;

fn graphGate(g: backend.Graph) bool {
    // 草稿图：录制期一律照常发射（CUDA 才录得到），重放期按槽位状态抑制。
    if (g.kind == .nextn) {
        const self = xg_backend orelse return g.replayed;
        // 槽数组在堆上，分配失败时是空切片（见 `g_slots` 的注释）。
        if (xg_cur_slot >= self.g_slots.len) return g.replayed;
        const s = self.g_slots[xg_cur_slot];
        if (s.xd_capturing) return false;
        if (s.xd_phase == .live) return true;
    }
    return g.replayed;
}

/// 编译期解析的 launcher 表。`cuda_linked` 是 comptime 已知的，所以每个字段都
/// 塌成 extern 或 stub 之一 —— 运行期没有间接跳转
/// run time.
const Launch = if (cuda_linked) struct {
    const quantize_q8_0 = Kernels.q35_launch_quantize_q8_0;
    const quantize_q8_k = Kernels.q35_launch_quantize_q8_K;
    const quantize_q8_k_mmq = Kernels.q35_launch_quantize_q8_K_mmq;
    const mmq_dp4a_wants = Kernels.q35_mmq_dp4a_wants;
    const mmq_act_block_bytes = Kernels.q35_mmq_act_block_bytes;
    fn dequantize(g: backend.Graph, ttype: c_int, d_src: ?*const anyopaque, d_dst: ?[*]f32, nblocks: c_int, stream: ?*anyopaque) c_int {
        if (graphGate(g)) return 0;
        return Kernels.q35_launch_dequantize(ttype, d_src, d_dst, nblocks, stream);
    }
    fn dequantize_row(g: backend.Graph, ttype: c_int, d_src_row: ?*const anyopaque, d_dst: ?[*]f32, cols: c_int, stream: ?*anyopaque) c_int {
        if (graphGate(g)) return 0;
        return Kernels.q35_launch_dequantize_row(ttype, d_src_row, d_dst, cols, stream);
    }
    /// 行号从每步控制块读的行反量化（MTP 草稿块的嵌入）。见 `api.h`：
    /// 行号若是实参，录进图之后重放永远嵌入第一轮那个 token。
    fn dequantize_row_dtoken(g: backend.Graph, ttype: c_int, d_src_base: ?*const anyopaque, d_dst: ?[*]f32, cols: c_int, row_bytes: c_int, d_ctl: ?*const anyopaque, stream: ?*anyopaque) c_int {
        if (graphGate(g)) return 0;
        return Kernels.q35_launch_dequantize_row_dtoken(ttype, d_src_base, d_dst, cols, row_bytes, d_ctl, stream);
    }
    fn mul_mat_vec(g: backend.Graph, d_w: ?*const anyopaque, d_xq: ?*const anyopaque, d_out: ?[*]f32, rows: c_int, cols: c_int, tokens: c_int, ttype: c_int, stream: ?*anyopaque) c_int {
        if (graphGate(g)) return 0;
        return Kernels.q35_launch_mul_mat_vec(d_w, d_xq, d_out, rows, cols, tokens, ttype, stream);
    }
    fn mul_mat_vec_f32(g: backend.Graph, d_w: ?[*]const f32, d_x: ?[*]const f32, d_out: ?[*]f32, rows: c_int, cols: c_int, tokens: c_int, stream: ?*anyopaque) c_int {
        if (graphGate(g)) return 0;
        return Kernels.q35_launch_mul_mat_vec_f32(d_w, d_x, d_out, rows, cols, tokens, stream);
    }
    fn mul_mat_vec_w16(g: backend.Graph, d_w: ?*const anyopaque, d_x: ?[*]const f32, d_out: ?[*]f32, rows: c_int, cols: c_int, tokens: c_int, ttype: c_int, stream: ?*anyopaque) c_int {
        if (graphGate(g)) return 0;
        return Kernels.q35_launch_mul_mat_vec_w16(d_w, d_x, d_out, rows, cols, tokens, ttype, stream);
    }
    const init_tables = Kernels.q35_launch_init_tables;
    const deinit_tables = Kernels.q35_launch_deinit_tables;
    const tables_bytes = Kernels.q35_iq_tables_bytes;
    fn scale_vec(g: backend.Graph, d_x: ?[*]f32, s: f32, n: c_int, stream: ?*anyopaque) c_int {
        if (graphGate(g)) return 0;
        return Kernels.q35_launch_scale_vec(d_x, s, n, stream);
    }
    fn l2norm_qk_scale(g: backend.Graph, d_x: ?[*]f32, nk: c_int, kd: c_int, k_offset: c_int, eps: f32, qscale: f32, stream: ?*anyopaque) c_int {
        if (graphGate(g)) return 0;
        return Kernels.q35_launch_l2norm_qk_scale(d_x, nk, kd, k_offset, eps, qscale, stream);
    }
    /// 是转发器，不是别名。`g` 决定这次发射是被录制、被重放还是直接执行（见
    /// `graphDispatch`）；其余参数就是 kernel 自己的参数，所以调用点上的
    /// `Launch.x(a, b, c)` 读起来和它包装的 launcher 一模一样。
    fn add_resid(g: backend.Graph, dst: ?[*]f32, src: ?[*]const f32, n: c_int, stream: ?*anyopaque) c_int {
        if (graphGate(g)) return 0;
        return Kernels.q35_launch_add_resid(dst, src, n, stream);
    }
    fn copy_range(g: backend.Graph, d_dst: ?[*]f32, d_src: ?[*]const f32, n: c_int, stream: ?*anyopaque) c_int {
        if (graphGate(g)) return 0;
        return Kernels.q35_launch_copy_range(d_dst, d_src, n, stream);
    }
    fn kv_quant(g: backend.Graph, d_x: ?[*]const f32, d_out: ?*anyopaque, nblocks: c_int, kind: c_int, stream: ?*anyopaque) c_int {
        if (graphGate(g)) return 0;
        return Kernels.q35_launch_kv_quant(d_x, d_out, nblocks, kind, stream);
    }
    fn kv_quant_dslot(g: backend.Graph, d_x: ?[*]const f32, d_out_base: ?*anyopaque, d_slot: ?*const c_int, nblocks_per_row: c_int, rows_this: c_int, row_bytes: c_int, kind: c_int, stream: ?*anyopaque) c_int {
        if (graphGate(g)) return 0;
        return Kernels.q35_launch_kv_quant_dslot(d_x, d_out_base, d_slot, nblocks_per_row, rows_this, row_bytes, kind, stream);
    }
    fn attn_chunk_dyn(g: backend.Graph, d_q: ?[*]const f32, d_gate: ?[*]const f32, d_kcache: ?[*]const f32, d_vcache: ?[*]const f32, d_gout: ?[*]f32, d_slot0: ?[*]const c_int, rows: c_int, heads: c_int, gqa: c_int, hd: c_int, kv_dim: c_int, scale: f32, kv_kind: c_int, stream: ?*anyopaque) c_int {
        if (graphGate(g)) return 0;
        return Kernels.q35_launch_attn_chunk_dyn(d_q, d_gate, d_kcache, d_vcache, d_gout, d_slot0, rows, heads, gqa, hd, kv_dim, scale, kv_kind, stream);
    }
    const attn_chunk_dyn_smem = Kernels.q35_attn_chunk_dyn_smem;
    const func_kv_quant_dslot = Kernels.q35_func_kv_quant_dslot;
    fn deinterleave_qg(g: backend.Graph, d_q: ?[*]f32, d_gate: ?[*]f32, d_src: ?[*]const f32, heads: c_int, hd: c_int, stream: ?*anyopaque) c_int {
        if (graphGate(g)) return 0;
        return Kernels.q35_launch_deinterleave_qg(d_q, d_gate, d_src, heads, hd, stream);
    }
    fn deinterleave_qg_strided(g: backend.Graph, d_q: ?[*]f32, d_gate: ?[*]f32, d_src: ?[*]const f32, heads: c_int, hd: c_int, tokens: c_int, stream: ?*anyopaque) c_int {
        if (graphGate(g)) return 0;
        return Kernels.q35_launch_deinterleave_qg_strided(d_q, d_gate, d_src, heads, hd, tokens, stream);
    }
    fn zero_buf(g: backend.Graph, d_buf: ?[*]f32, n: c_int, stream: ?*anyopaque) c_int {
        if (graphGate(g)) return 0;
        return Kernels.q35_launch_zero_buf(d_buf, n, stream);
    }
    fn rmsnorm(g: backend.Graph, d_out: ?[*]f32, d_x: ?[*]const f32, d_w: ?[*]const f32, rows: c_int, n: c_int, eps: f32, stream: ?*anyopaque) c_int {
        if (graphGate(g)) return 0;
        return Kernels.q35_launch_rmsnorm(d_out, d_x, d_w, rows, n, eps, stream);
    }
    fn l2norm(g: backend.Graph, d_x: ?[*]f32, rows: c_int, n: c_int, eps: f32, stream: ?*anyopaque) c_int {
        if (graphGate(g)) return 0;
        return Kernels.q35_launch_l2norm(d_x, rows, n, eps, stream);
    }
    fn silu(g: backend.Graph, d_x: ?[*]f32, n: c_int, stream: ?*anyopaque) c_int {
        if (graphGate(g)) return 0;
        return Kernels.q35_launch_silu(d_x, n, stream);
    }
    fn mul_silu(g: backend.Graph, d_out: ?[*]f32, d_gate: ?[*]const f32, n: c_int, stream: ?*anyopaque) c_int {
        if (graphGate(g)) return 0;
        return Kernels.q35_launch_mul_silu(d_out, d_gate, n, stream);
    }
    fn mul_sigmoid(g: backend.Graph, d_out: ?[*]f32, d_gate: ?[*]const f32, n: c_int, stream: ?*anyopaque) c_int {
        if (graphGate(g)) return 0;
        return Kernels.q35_launch_mul_sigmoid(d_out, d_gate, n, stream);
    }
    fn swiglu(g: backend.Graph, d_gate: ?[*]f32, d_up: ?[*]const f32, n: c_int, stream: ?*anyopaque) c_int {
        if (graphGate(g)) return 0;
        return Kernels.q35_launch_swiglu(d_gate, d_up, n, stream);
    }
    fn softmax(g: backend.Graph, d_x: ?[*]f32, n: c_int, stream: ?*anyopaque) c_int {
        if (graphGate(g)) return 0;
        return Kernels.q35_launch_softmax(d_x, n, stream);
    }
    fn conv_step(g: backend.Graph, d_conv: ?[*]f32, d_y: ?[*]f32, d_wconv: ?[*]const f32, n: c_int, stream: ?*anyopaque) c_int {
        if (graphGate(g)) return 0;
        return Kernels.q35_launch_conv_step(d_conv, d_y, d_wconv, n, stream);
    }
    fn argmax(g: backend.Graph, d_x: ?[*]const f32, n: c_int, d_out: ?*c_int, stream: ?*anyopaque) c_int {
        if (graphGate(g)) return 0;
        return Kernels.q35_launch_argmax(d_x, n, d_out, stream);
    }
    fn argmax_batch(g: backend.Graph, d_x: ?[*]const f32, n: c_int, rows: c_int, d_out: ?[*]c_int, stream: ?*anyopaque) c_int {
        if (graphGate(g)) return 0;
        return Kernels.q35_launch_argmax_batch(d_x, n, rows, d_out, stream);
    }
    fn argmax_conf(g: backend.Graph, d_x: ?[*]const f32, n: c_int, d_out: ?[*]c_int, stream: ?*anyopaque) c_int {
        if (graphGate(g)) return 0;
        return Kernels.q35_launch_argmax_conf(d_x, n, d_out, stream);
    }
    fn sample(g: backend.Graph, d_logits: ?[*]const f32, n: c_int, d_ctl: ?*c_int, d_scratch: ?*anyopaque, stream: ?*anyopaque) c_int {
        if (graphGate(g)) return 0;
        return Kernels.q35_launch_sample(d_logits, n, d_ctl, d_scratch, stream);
    }
    const sample_init = Kernels.q35_launch_sample_init;
    const sample_deinit = Kernels.q35_launch_sample_deinit;
    const sample_scratch_ints = Kernels.q35_sample_scratch_ints;
    const sample_ctl_ints = Kernels.q35_sample_ctl_ints;
    fn attn_decode(g: backend.Graph, d_q: ?[*]const f32, d_gate: ?[*]const f32, d_kcache: ?[*]const f32, d_vcache: ?[*]const f32, d_att: ?[*]f32, d_gout: ?[*]f32, slot: c_int, heads: c_int, gqa: c_int, hd: c_int, kv_dim: c_int, scale: f32, kv_kind: c_int, stream: ?*anyopaque) c_int {
        if (graphGate(g)) return 0;
        return Kernels.q35_launch_attn_decode(d_q, d_gate, d_kcache, d_vcache, d_att, d_gout, slot, heads, gqa, hd, kv_dim, scale, kv_kind, stream);
    }
    fn attn_chunk(g: backend.Graph, d_q: ?[*]const f32, d_gate: ?[*]const f32, d_kcache: ?[*]const f32, d_vcache: ?[*]const f32, d_gout: ?[*]f32, slot0: c_int, rows: c_int, heads: c_int, gqa: c_int, hd: c_int, kv_dim: c_int, scale: f32, kv_kind: c_int, stream: ?*anyopaque) c_int {
        if (graphGate(g)) return 0;
        return Kernels.q35_launch_attn_chunk(d_q, d_gate, d_kcache, d_vcache, d_gout, slot0, rows, heads, gqa, hd, kv_dim, scale, kv_kind, stream);
    }
    const attn_chunk_max_rows = Kernels.q35_attn_chunk_max_rows;
    fn attn_decode_dyn(g: backend.Graph, d_q: ?[*]const f32, d_gate: ?[*]const f32, d_kcache: ?[*]const f32, d_vcache: ?[*]const f32, d_att: ?[*]f32, d_gout: ?[*]f32, d_slot: ?[*]const c_int, heads: c_int, gqa: c_int, hd: c_int, kv_dim: c_int, scale: f32, kv_kind: c_int, stream: ?*anyopaque) c_int {
        if (graphGate(g)) return 0;
        return Kernels.q35_launch_attn_decode_dyn(d_q, d_gate, d_kcache, d_vcache, d_att, d_gout, d_slot, heads, gqa, hd, kv_dim, scale, kv_kind, stream);
    }
    fn attn_decode_coal(g: backend.Graph, d_q: ?[*]const f32, d_gate: ?[*]const f32, d_kcache: ?[*]const f32, d_vcache: ?[*]const f32, d_att: ?[*]f32, d_gout: ?[*]f32, slot: c_int, heads: c_int, gqa: c_int, hd: c_int, kv_dim: c_int, scale: f32, kv_kind: c_int, stream: ?*anyopaque) c_int {
        if (graphGate(g)) return 0;
        return Kernels.q35_launch_attn_decode_coal(d_q, d_gate, d_kcache, d_vcache, d_att, d_gout, slot, heads, gqa, hd, kv_dim, scale, kv_kind, stream);
    }
    fn attn_decode_split2(g: backend.Graph, d_q: ?[*]const f32, d_gate: ?[*]const f32, d_kcache: ?[*]const f32, d_vcache: ?[*]const f32, d_att: ?[*]f32, d_gout: ?[*]f32, d_part: ?[*]f32, slot: c_int, heads: c_int, gqa: c_int, hd: c_int, kv_dim: c_int, scale: f32, split: c_int, kv_kind: c_int, stream: ?*anyopaque) c_int {
        if (graphGate(g)) return 0;
        return Kernels.q35_launch_attn_decode_split2(d_q, d_gate, d_kcache, d_vcache, d_att, d_gout, d_part, slot, heads, gqa, hd, kv_dim, scale, split, kv_kind, stream);
    }
    fn attn_decode_split_dyn(g: backend.Graph, d_q: ?[*]const f32, d_gate: ?[*]const f32, d_kcache: ?[*]const f32, d_vcache: ?[*]const f32, d_att: ?[*]f32, d_gout: ?[*]f32, d_part: ?[*]f32, d_slot: ?[*]const c_int, heads: c_int, gqa: c_int, hd: c_int, kv_dim: c_int, scale: f32, split: c_int, kv_kind: c_int, stream: ?*anyopaque) c_int {
        if (graphGate(g)) return 0;
        return Kernels.q35_launch_attn_decode_split_dyn(d_q, d_gate, d_kcache, d_vcache, d_att, d_gout, d_part, d_slot, heads, gqa, hd, kv_dim, scale, split, kv_kind, stream);
    }
    const set_int1 = Kernels.q35_launch_set_int1;
    const copy_range_dslot = Kernels.q35_launch_copy_range_dslot;
    const xd_ctl_publish = Kernels.q35_launch_xd_ctl_publish;
    fn gdn_step(g: backend.Graph, d_state: ?[*]f32, d_k: ?[*]const f32, d_v: ?[*]const f32, d_q: ?[*]const f32, d_km: ?[*]f32, d_delta: ?[*]f32, d_out: ?[*]f32, gamma: f32, beta: f32, vd: c_int, stream: ?*anyopaque) c_int {
        if (graphGate(g)) return 0;
        return Kernels.q35_launch_gdn_step(d_state, d_k, d_v, d_q, d_km, d_delta, d_out, gamma, beta, vd, stream);
    }
    fn gdn_step_rows(g: backend.Graph, d_state: ?[*]f32, state_stride: c_int, d_k: ?[*]const f32, k_stride: c_int, d_v: ?[*]const f32, v_stride: c_int, d_q: ?[*]const f32, q_stride: c_int, d_km: ?[*]f32, d_delta: ?[*]f32, d_out: ?[*]f32, d_gamma: ?[*]const f32, d_beta: ?[*]const f32, heads: c_int, nk: c_int, kd: c_int, vd: c_int, stream: ?*anyopaque) c_int {
        if (graphGate(g)) return 0;
        return Kernels.q35_launch_gdn_step_rows(d_state, state_stride, d_k, k_stride, d_v, v_stride, d_q, q_stride, d_km, d_delta, d_out, d_gamma, d_beta, heads, nk, kd, vd, stream);
    }
    fn gdn_step_rows_chunk(g: backend.Graph, d_state: ?[*]f32, state_stride: c_int, d_k: ?[*]const f32, k_stride: c_int, d_v: ?[*]const f32, v_stride: c_int, d_q: ?[*]const f32, q_stride: c_int, d_km: ?[*]f32, km_stride: c_int, d_delta: ?[*]f32, delta_stride: c_int, d_out: ?[*]f32, out_stride: c_int, d_gamma: ?[*]const f32, gamma_stride: c_int, d_beta: ?[*]const f32, beta_stride: c_int, heads: c_int, nk: c_int, kd: c_int, vd: c_int, rows: c_int, stream: ?*anyopaque) c_int {
        if (graphGate(g)) return 0;
        return Kernels.q35_launch_gdn_step_rows_chunk(d_state, state_stride, d_k, k_stride, d_v, v_stride, d_q, q_stride, d_km, km_stride, d_delta, delta_stride, d_out, out_stride, d_gamma, gamma_stride, d_beta, beta_stride, heads, nk, kd, vd, rows, stream);
    }
    fn gdn_step_rows_snap(g: backend.Graph, d_state: ?[*]f32, state_stride: c_int, d_k: ?[*]const f32, k_stride: c_int, d_v: ?[*]const f32, v_stride: c_int, d_q: ?[*]const f32, q_stride: c_int, d_km: ?[*]f32, km_stride: c_int, d_delta: ?[*]f32, delta_stride: c_int, d_out: ?[*]f32, out_stride: c_int, d_gamma: ?[*]const f32, gamma_stride: c_int, d_beta: ?[*]const f32, beta_stride: c_int, d_snap: ?[*]f32, nsnap: c_int, heads: c_int, nk: c_int, kd: c_int, vd: c_int, rows: c_int, stream: ?*anyopaque) c_int {
        if (graphGate(g)) return 0;
        return Kernels.q35_launch_gdn_step_rows_snap(d_state, state_stride, d_k, k_stride, d_v, v_stride, d_q, q_stride, d_km, km_stride, d_delta, delta_stride, d_out, out_stride, d_gamma, gamma_stride, d_beta, beta_stride, d_snap, nsnap, heads, nk, kd, vd, rows, stream);
    }
    fn gdn_step_wy(g: backend.Graph, d_state: ?[*]f32, state_stride: c_int, d_k: ?[*]const f32, k_stride: c_int, d_v: ?[*]const f32, v_stride: c_int, d_q: ?[*]const f32, q_stride: c_int, d_km: ?[*]f32, km_stride: c_int, d_delta: ?[*]f32, delta_stride: c_int, d_out: ?[*]f32, out_stride: c_int, d_gamma: ?[*]const f32, gamma_stride: c_int, d_beta: ?[*]const f32, beta_stride: c_int, heads: c_int, nk: c_int, kd: c_int, vd: c_int, rows: c_int, stream: ?*anyopaque) c_int {
        if (graphGate(g)) return 0;
        return Kernels.q35_launch_gdn_step_wy(d_state, state_stride, d_k, k_stride, d_v, v_stride, d_q, q_stride, d_km, km_stride, d_delta, delta_stride, d_out, out_stride, d_gamma, gamma_stride, d_beta, beta_stride, heads, nk, kd, vd, rows, stream);
    }
    fn gdn_gamma(g: backend.Graph, d_a_raw: ?[*]const f32, d_b_raw: ?[*]f32, d_ssm_a: ?[*]const f32, d_dt_bias: ?[*]const f32, d_gamma: ?[*]f32, nv: c_int, stream: ?*anyopaque) c_int {
        if (graphGate(g)) return 0;
        return Kernels.q35_launch_gdn_gamma(d_a_raw, d_b_raw, d_ssm_a, d_dt_bias, d_gamma, nv, stream);
    }
    // 仅 prefill 使用的按行/跨步孪生版本。列在这里（并在 stub 分支里以同样顺序
    // 列出），好让 `Launch` 成为 launcher 与它"无 CUDA"回退之间唯一的接缝：
    // `dGdnPrefillCore` 的调用点曾直接写 `Kernels.q35_launch_*`，那让这 4 个 stub
    // 变成死代码，并让 `zig build`（不带 -Dcuda）在**链接期**因未定义符号失败
    // —— 这类失败是上面的类型错误抓不到的。
    fn conv_step_rows(g: backend.Graph, d_conv: ?[*]f32, d_y: ?[*]f32, d_wconv: ?[*]const f32, n: c_int, tokens: c_int, y_stride: c_int, stream: ?*anyopaque) c_int {
        if (graphGate(g)) return 0;
        return Kernels.q35_launch_conv_step_rows(d_conv, d_y, d_wconv, n, tokens, y_stride, stream);
    }
    fn gdn_gamma_rows(g: backend.Graph, d_a_raw: ?[*]const f32, d_b_raw: ?[*]f32, d_ssm_a: ?[*]const f32, d_dt_bias: ?[*]const f32, d_gamma: ?[*]f32, nv: c_int, tokens: c_int, a_stride: c_int, b_stride: c_int, gamma_stride: c_int, stream: ?*anyopaque) c_int {
        if (graphGate(g)) return 0;
        return Kernels.q35_launch_gdn_gamma_rows(d_a_raw, d_b_raw, d_ssm_a, d_dt_bias, d_gamma, nv, tokens, a_stride, b_stride, gamma_stride, stream);
    }
    fn l2norm_strided(g: backend.Graph, d_x: ?[*]f32, row_offset: c_int, rows: c_int, n: c_int, eps: f32, tokens: c_int, row_stride: c_int, stream: ?*anyopaque) c_int {
        if (graphGate(g)) return 0;
        return Kernels.q35_launch_l2norm_strided(d_x, row_offset, rows, n, eps, tokens, row_stride, stream);
    }
    fn scale_vec_strided(g: backend.Graph, d_x: ?[*]f32, offset: c_int, s: f32, n: c_int, tokens: c_int, stride: c_int, stream: ?*anyopaque) c_int {
        if (graphGate(g)) return 0;
        return Kernels.q35_launch_scale_vec_strided(d_x, offset, s, n, tokens, stride, stream);
    }
    fn rope64(g: backend.Graph, d_x: ?[*]f32, heads: c_int, hd: c_int, pos: c_int, base: f32, stream: ?*anyopaque) c_int {
        if (graphGate(g)) return 0;
        return Kernels.q35_launch_rope64(d_x, heads, hd, pos, base, stream);
    }
    fn rope64_dpos(g: backend.Graph, d_x: ?[*]f32, heads: c_int, hd: c_int, d_ctl: ?*const anyopaque, stream: ?*anyopaque) c_int {
        if (graphGate(g)) return 0;
        return Kernels.q35_launch_rope64_dpos(d_x, heads, hd, d_ctl, stream);
    }
    const rope64_setenv = Kernels.q35_launch_rope64_setenv;
    fn rope64_rows(g: backend.Graph, d_x: ?[*]f32, rows: c_int, heads: c_int, hd: c_int, pos0: c_int, base: f32, stream: ?*anyopaque) c_int {
        if (graphGate(g)) return 0;
        return Kernels.q35_launch_rope64_rows(d_x, rows, heads, hd, pos0, base, stream);
    }
    fn rope_rows_dim(g: backend.Graph, d_x: ?[*]f32, rows: c_int, heads: c_int, hd: c_int, pos0: c_int, base: f32, rotary_dim: c_int, stream: ?*anyopaque) c_int {
        if (graphGate(g)) return 0;
        return Kernels.q35_launch_rope_rows_dim(d_x, rows, heads, hd, pos0, base, rotary_dim, stream);
    }
    fn rope64_rows_dpos(g: backend.Graph, d_x: ?[*]f32, rows: c_int, heads: c_int, hd: c_int, d_ctl: ?*const anyopaque, stream: ?*anyopaque) c_int {
        if (graphGate(g)) return 0;
        return Kernels.q35_launch_rope64_rows_dpos(d_x, rows, heads, hd, d_ctl, stream);
    }
} else struct {
    fn quantize_q8_0(d_x: ?[*]const f32, d_out: ?*anyopaque, nblocks: c_int, stream: ?*anyopaque) callconv(.c) c_int {
        _ = .{ d_x, d_out, nblocks, stream };
        return stubNoCudaKernels();
    }
    fn quantize_q8_k(d_x: ?[*]const f32, d_out: ?*anyopaque, nblocks: c_int, stream: ?*anyopaque) callconv(.c) c_int {
        _ = .{ d_x, d_out, nblocks, stream };
        return stubNoCudaKernels();
    }
    fn quantize_q8_k_mmq(d_x: ?[*]const f32, d_out: ?*anyopaque, d_mmq: ?*anyopaque, nblocks: c_int, stream: ?*anyopaque) callconv(.c) c_int {
        _ = .{ d_x, d_out, d_mmq, nblocks, stream };
        return stubNoCudaKernels();
    }
    /// 没有 CUDA kernel 也就没有 dp4a kernel，所以调用方保持普通 q8_K 路径。
    /// 返回 0 是不需要第二块缓冲的那个答案。
    fn mmq_dp4a_wants(ttype: c_int) c_int {
        _ = ttype;
        return 0;
    }
    /// 从不使用（这里 `mmq_dp4a_wants` 恒为 0），保留是为了让 `Launch` 在两个
    /// 分支里形状一致。
    fn mmq_act_block_bytes() c_int {
        return 288;
    }
    fn dequantize(g: backend.Graph, ttype: c_int, d_src: ?*const anyopaque, d_dst: ?[*]f32, nblocks: c_int, stream: ?*anyopaque) c_int {
        _ = g;
        _ = .{ ttype, d_src, d_dst, nblocks, stream };
        return stubNoCudaKernels();
    }
    fn dequantize_row(g: backend.Graph, ttype: c_int, d_src_row: ?*const anyopaque, d_dst: ?[*]f32, cols: c_int, stream: ?*anyopaque) c_int {
        _ = g;
        _ = .{ ttype, d_src_row, d_dst, cols, stream };
        return stubNoCudaKernels();
    }
    fn dequantize_row_dtoken(g: backend.Graph, ttype: c_int, d_src_base: ?*const anyopaque, d_dst: ?[*]f32, cols: c_int, row_bytes: c_int, d_ctl: ?*const anyopaque, stream: ?*anyopaque) c_int {
        _ = g;
        _ = .{ ttype, d_src_base, d_dst, cols, row_bytes, d_ctl, stream };
        return stubNoCudaKernels();
    }
    fn mul_mat_vec(g: backend.Graph, d_w: ?*const anyopaque, d_xq: ?*const anyopaque, d_out: ?[*]f32, rows: c_int, cols: c_int, tokens: c_int, ttype: c_int, stream: ?*anyopaque) c_int {
        _ = g;
        _ = .{ d_w, d_xq, d_out, rows, cols, tokens, ttype, stream };
        return stubNoCudaKernels();
    }
    fn mul_mat_vec_f32(g: backend.Graph, d_w: ?[*]const f32, d_x: ?[*]const f32, d_out: ?[*]f32, rows: c_int, cols: c_int, tokens: c_int, stream: ?*anyopaque) c_int {
        _ = g;
        _ = .{ d_w, d_x, d_out, rows, cols, tokens, stream };
        return stubNoCudaKernels();
    }
    fn mul_mat_vec_w16(g: backend.Graph, d_w: ?*const anyopaque, d_x: ?[*]const f32, d_out: ?[*]f32, rows: c_int, cols: c_int, tokens: c_int, ttype: c_int, stream: ?*anyopaque) c_int {
        _ = g;
        _ = .{ d_w, d_x, d_out, rows, cols, tokens, ttype, stream };
        return stubNoCudaKernels();
    }
    fn init_tables(stream: ?*anyopaque) callconv(.c) c_int {
        _ = stream;
        return stubNoCudaKernels();
    }
    fn deinit_tables() callconv(.c) void {
        // `init_tables` 失败时已经报过了，这里保持安静。
    }
    fn tables_bytes() callconv(.c) c_ulonglong {
        return 0;
    }
    fn scale_vec(g: backend.Graph, d_x: ?[*]f32, s: f32, n: c_int, stream: ?*anyopaque) c_int {
        _ = g;
        _ = .{ d_x, s, n, stream };
        return stubNoCudaKernels();
    }
    fn add_resid(g: backend.Graph, d_dst: ?[*]f32, d_src: ?[*]const f32, n: c_int, stream: ?*anyopaque) c_int {
        _ = g;
        _ = .{ d_dst, d_src, n, stream };
        return stubNoCudaKernels();
    }
    fn copy_range(g: backend.Graph, d_dst: ?[*]f32, d_src: ?[*]const f32, n: c_int, stream: ?*anyopaque) c_int {
        _ = g;
        _ = .{ d_dst, d_src, n, stream };
        return stubNoCudaKernels();
    }
    fn kv_quant(g: backend.Graph, d_x: ?[*]const f32, d_out: ?*anyopaque, nblocks: c_int, kind: c_int, stream: ?*anyopaque) c_int {
        _ = g;
        _ = .{ d_x, d_out, nblocks, kind, stream };
        return stubNoCudaKernels();
    }
    fn attn_chunk_dyn(g: backend.Graph, d_q: ?[*]const f32, d_gate: ?[*]const f32, d_kcache: ?[*]const f32, d_vcache: ?[*]const f32, d_gout: ?[*]f32, d_slot0: ?[*]const c_int, rows: c_int, heads: c_int, gqa: c_int, hd: c_int, kv_dim: c_int, scale: f32, kv_kind: c_int, stream: ?*anyopaque) c_int {
        _ = g;
        _ = .{ d_q, d_gate, d_kcache, d_vcache, d_gout, d_slot0, rows, heads, gqa, hd, kv_dim, scale, kv_kind, stream };
        return stubNoCudaKernels();
    }
    fn attn_chunk_dyn_smem(hd: c_int, kv_kind: c_int) callconv(.c) c_ulonglong {
        _ = .{ hd, kv_kind };
        return 0;
    }
    fn kv_quant_dslot(g: backend.Graph, d_x: ?[*]const f32, d_out_base: ?*anyopaque, d_slot: ?*const c_int, nblocks_per_row: c_int, rows_this: c_int, row_bytes: c_int, kind: c_int, stream: ?*anyopaque) c_int {
        _ = g;
        _ = .{ d_x, d_out_base, d_slot, nblocks_per_row, rows_this, row_bytes, kind, stream };
        return stubNoCudaKernels();
    }
    fn func_kv_quant_dslot() callconv(.c) ?*anyopaque {
        return null;
    }
    fn deinterleave_qg(g: backend.Graph, d_q: ?[*]f32, d_gate: ?[*]f32, d_src: ?[*]const f32, heads: c_int, hd: c_int, stream: ?*anyopaque) c_int {
        _ = g;
        _ = .{ d_q, d_gate, d_src, heads, hd, stream };
        return stubNoCudaKernels();
    }
    fn zero_buf(g: backend.Graph, d_buf: ?[*]f32, n: c_int, stream: ?*anyopaque) c_int {
        _ = g;
        _ = .{ d_buf, n, stream };
        return stubNoCudaKernels();
    }
    fn rmsnorm(g: backend.Graph, d_out: ?[*]f32, d_x: ?[*]const f32, d_w: ?[*]const f32, rows: c_int, n: c_int, eps: f32, stream: ?*anyopaque) c_int {
        _ = g;
        _ = .{ d_out, d_x, d_w, rows, n, eps, stream };
        return stubNoCudaKernels();
    }
    fn l2norm(g: backend.Graph, d_x: ?[*]f32, rows: c_int, n: c_int, eps: f32, stream: ?*anyopaque) c_int {
        _ = g;
        _ = .{ d_x, rows, n, eps, stream };
        return stubNoCudaKernels();
    }
    fn silu(g: backend.Graph, d_x: ?[*]f32, n: c_int, stream: ?*anyopaque) c_int {
        _ = g;
        _ = .{ d_x, n, stream };
        return stubNoCudaKernels();
    }
    fn mul_silu(g: backend.Graph, d_out: ?[*]f32, d_gate: ?[*]const f32, n: c_int, stream: ?*anyopaque) c_int {
        _ = g;
        _ = .{ d_out, d_gate, n, stream };
        return stubNoCudaKernels();
    }
    fn mul_sigmoid(g: backend.Graph, d_out: ?[*]f32, d_gate: ?[*]const f32, n: c_int, stream: ?*anyopaque) c_int {
        _ = g;
        _ = .{ d_out, d_gate, n, stream };
        return stubNoCudaKernels();
    }
    fn swiglu(g: backend.Graph, d_gate: ?[*]f32, d_up: ?[*]const f32, n: c_int, stream: ?*anyopaque) c_int {
        _ = g;
        _ = .{ d_gate, d_up, n, stream };
        return stubNoCudaKernels();
    }
    fn softmax(g: backend.Graph, d_x: ?[*]f32, n: c_int, stream: ?*anyopaque) c_int {
        _ = g;
        _ = .{ d_x, n, stream };
        return stubNoCudaKernels();
    }
    fn conv_step(g: backend.Graph, d_conv: ?[*]f32, d_y: ?[*]f32, d_wconv: ?[*]const f32, n: c_int, stream: ?*anyopaque) c_int {
        _ = g;
        _ = .{ d_conv, d_y, d_wconv, n, stream };
        return stubNoCudaKernels();
    }
    fn argmax_batch(g: backend.Graph, d_x: ?[*]const f32, n: c_int, rows: c_int, d_out: ?[*]c_int, stream: ?*anyopaque) c_int {
        _ = g;
        _ = .{ d_x, n, rows, d_out, stream };
        return stubNoCudaKernels();
    }
    fn argmax(g: backend.Graph, d_x: ?[*]const f32, n: c_int, d_out: ?*c_int, stream: ?*anyopaque) c_int {
        _ = g;
        _ = .{ d_x, n, d_out, stream };
        return stubNoCudaKernels();
    }
    fn argmax_conf(g: backend.Graph, d_x: ?[*]const f32, n: c_int, d_out: ?[*]c_int, stream: ?*anyopaque) c_int {
        _ = g;
        _ = .{ d_x, n, d_out, stream };
        return stubNoCudaKernels();
    }
    fn sample(g: backend.Graph, d_logits: ?[*]const f32, n: c_int, d_ctl: ?*c_int, d_scratch: ?*anyopaque, stream: ?*anyopaque) c_int {
        _ = g;
        _ = .{ d_logits, n, d_ctl, d_scratch, stream };
        return stubNoCudaKernels();
    }
    fn sample_init(d_ctl: ?*?*anyopaque, d_scratch: ?*?*anyopaque) callconv(.c) c_int {
        _ = .{ d_ctl, d_scratch };
        return stubNoCudaKernels();
    }
    fn sample_deinit(d_ctl: ?*?*anyopaque, d_scratch: ?*?*anyopaque) callconv(.c) void {
        _ = .{ d_ctl, d_scratch };
    }
    fn sample_scratch_ints() callconv(.c) c_int {
        return 0;
    }
    fn sample_ctl_ints() callconv(.c) c_int {
        return 0;
    }
    fn attn_decode(g: backend.Graph, d_q: ?[*]const f32, d_gate: ?[*]const f32, d_kcache: ?[*]const f32, d_vcache: ?[*]const f32, d_att: ?[*]f32, d_gout: ?[*]f32, slot: c_int, heads: c_int, gqa: c_int, hd: c_int, kv_dim: c_int, scale: f32, kv_kind: c_int, stream: ?*anyopaque) c_int {
        _ = g;
        _ = .{ d_q, d_gate, d_kcache, d_vcache, d_att, d_gout, slot, heads, gqa, hd, kv_dim, scale, kv_kind, stream };
        return stubNoCudaKernels();
    }
    fn attn_chunk(g: backend.Graph, d_q: ?[*]const f32, d_gate: ?[*]const f32, d_kcache: ?[*]const f32, d_vcache: ?[*]const f32, d_gout: ?[*]f32, slot0: c_int, rows: c_int, heads: c_int, gqa: c_int, hd: c_int, kv_dim: c_int, scale: f32, kv_kind: c_int, stream: ?*anyopaque) c_int {
        _ = g;
        _ = .{ d_q, d_gate, d_kcache, d_vcache, d_gout, slot0, rows, heads, gqa, hd, kv_dim, scale, kv_kind, stream };
        return stubNoCudaKernels();
    }
    /// 没有 chunk kernel 就不分块：返回 0 让 `dAttnChunk` 把 `maxrows` 留成 0、
    /// 完全跳过按片路径。在这里返回正数才是危险的答案 —— 分片循环接下来就会去
    /// 调上面那个被 stub 掉的 `attn_chunk`。
    fn attn_chunk_max_rows(slot0: c_int, hd: c_int, kv_kind: c_int) c_int {
        _ = .{ slot0, hd, kv_kind };
        return 0;
    }
    fn deinterleave_qg_strided(g: backend.Graph, d_q: ?[*]f32, d_gate: ?[*]f32, d_src: ?[*]const f32, heads: c_int, hd: c_int, tokens: c_int, stream: ?*anyopaque) c_int {
        _ = g;
        _ = .{ d_q, d_gate, d_src, heads, hd, tokens, stream };
        return stubNoCudaKernels();
    }
    // 顺序与签名都和上面 `Kernels` 的别名一致：CUDA 分支暴露的每个 launcher 在
    // 这里都必须有对应成员，否则以后新加的调用点会让 `zig build`（不带 -Dcuda）
    // 报 `no member named ...` —— `set_int1` 和三个 `attn_decode_*` 变体正是这样
    // 漏过去的：它们被起了别名、被调用，而这个结构体已经很久没人读了。
    fn attn_decode_dyn(g: backend.Graph, d_q: ?[*]const f32, d_gate: ?[*]const f32, d_kcache: ?[*]const f32, d_vcache: ?[*]const f32, d_att: ?[*]f32, d_gout: ?[*]f32, d_slot: ?[*]const c_int, heads: c_int, gqa: c_int, hd: c_int, kv_dim: c_int, scale: f32, kv_kind: c_int, stream: ?*anyopaque) c_int {
        _ = g;
        _ = .{ d_q, d_gate, d_kcache, d_vcache, d_att, d_gout, d_slot, heads, gqa, hd, kv_dim, scale, kv_kind, stream };
        return stubNoCudaKernels();
    }
    fn attn_decode_coal(g: backend.Graph, d_q: ?[*]const f32, d_gate: ?[*]const f32, d_kcache: ?[*]const f32, d_vcache: ?[*]const f32, d_att: ?[*]f32, d_gout: ?[*]f32, slot: c_int, heads: c_int, gqa: c_int, hd: c_int, kv_dim: c_int, scale: f32, kv_kind: c_int, stream: ?*anyopaque) c_int {
        _ = g;
        _ = .{ d_q, d_gate, d_kcache, d_vcache, d_att, d_gout, slot, heads, gqa, hd, kv_dim, scale, kv_kind, stream };
        return stubNoCudaKernels();
    }
    fn attn_decode_split2(g: backend.Graph, d_q: ?[*]const f32, d_gate: ?[*]const f32, d_kcache: ?[*]const f32, d_vcache: ?[*]const f32, d_att: ?[*]f32, d_gout: ?[*]f32, d_part: ?[*]f32, slot: c_int, heads: c_int, gqa: c_int, hd: c_int, kv_dim: c_int, scale: f32, split: c_int, kv_kind: c_int, stream: ?*anyopaque) c_int {
        _ = g;
        _ = .{ d_q, d_gate, d_kcache, d_vcache, d_att, d_gout, d_part, slot, heads, gqa, hd, kv_dim, scale, split, kv_kind, stream };
        return stubNoCudaKernels();
    }
    fn attn_decode_split_dyn(g: backend.Graph, d_q: ?[*]const f32, d_gate: ?[*]const f32, d_kcache: ?[*]const f32, d_vcache: ?[*]const f32, d_att: ?[*]f32, d_gout: ?[*]f32, d_part: ?[*]f32, d_slot: ?[*]const c_int, heads: c_int, gqa: c_int, hd: c_int, kv_dim: c_int, scale: f32, split: c_int, kv_kind: c_int, stream: ?*anyopaque) c_int {
        _ = g;
        _ = .{ d_q, d_gate, d_kcache, d_vcache, d_att, d_gout, d_part, d_slot, heads, gqa, hd, kv_dim, scale, split, kv_kind, stream };
        return stubNoCudaKernels();
    }
    fn set_int1(d: ?[*]c_int, v: c_int, stream: ?*anyopaque) callconv(.c) c_int {
        _ = .{ d, v, stream };
        return stubNoCudaKernels();
    }
    fn copy_range_dslot(dst_base: ?[*]f32, stride: c_int, src: ?[*]const f32, rows: c_int, n: c_int, ctl: ?[*]const c_int, stream: ?*anyopaque) callconv(.c) c_int {
        _ = .{ dst_base, stride, src, rows, n, ctl, stream };
        return stubNoCudaKernels();
    }
    fn xd_ctl_publish(d_ctl: ?*anyopaque, pos: c_int, base_bits: c_int, kv_slot: c_int, attn_slot: c_int, rows: c_int, stream: ?*anyopaque) callconv(.c) c_int {
        _ = .{ d_ctl, pos, base_bits, kv_slot, attn_slot, rows, stream };
        return stubNoCudaKernels();
    }
    fn gdn_step(g: backend.Graph, d_state: ?[*]f32, d_k: ?[*]const f32, d_v: ?[*]const f32, d_q: ?[*]const f32, d_km: ?[*]f32, d_delta: ?[*]f32, d_out: ?[*]f32, gamma: f32, beta: f32, vd: c_int, stream: ?*anyopaque) c_int {
        _ = g;
        _ = .{ d_state, d_k, d_v, d_q, d_km, d_delta, d_out, gamma, beta, vd, stream };
        return stubNoCudaKernels();
    }
    fn gdn_step_rows(g: backend.Graph, d_state: ?[*]f32, state_stride: c_int, d_k: ?[*]const f32, k_stride: c_int, d_v: ?[*]const f32, v_stride: c_int, d_q: ?[*]const f32, q_stride: c_int, d_km: ?[*]f32, d_delta: ?[*]f32, d_out: ?[*]f32, d_gamma: ?[*]const f32, d_beta: ?[*]const f32, heads: c_int, nk: c_int, kd: c_int, vd: c_int, stream: ?*anyopaque) c_int {
        _ = g;
        _ = .{ d_state, state_stride, d_k, k_stride, d_v, v_stride, d_q, q_stride, d_km, d_delta, d_out, d_gamma, d_beta, heads, nk, kd, vd, stream };
        return stubNoCudaKernels();
    }
    fn gdn_step_rows_chunk(g: backend.Graph, d_state: ?[*]f32, state_stride: c_int, d_k: ?[*]const f32, k_stride: c_int, d_v: ?[*]const f32, v_stride: c_int, d_q: ?[*]const f32, q_stride: c_int, d_km: ?[*]f32, km_stride: c_int, d_delta: ?[*]f32, delta_stride: c_int, d_out: ?[*]f32, out_stride: c_int, d_gamma: ?[*]const f32, gamma_stride: c_int, d_beta: ?[*]const f32, beta_stride: c_int, heads: c_int, nk: c_int, kd: c_int, vd: c_int, rows: c_int, stream: ?*anyopaque) c_int {
        _ = g;
        _ = .{ d_state, state_stride, d_k, k_stride, d_v, v_stride, d_q, q_stride, d_km, km_stride, d_delta, delta_stride, d_out, out_stride, d_gamma, gamma_stride, d_beta, beta_stride, heads, nk, kd, vd, rows, stream };
        return stubNoCudaKernels();
    }
    fn gdn_step_rows_snap(g: backend.Graph, d_state: ?[*]f32, state_stride: c_int, d_k: ?[*]const f32, k_stride: c_int, d_v: ?[*]const f32, v_stride: c_int, d_q: ?[*]const f32, q_stride: c_int, d_km: ?[*]f32, km_stride: c_int, d_delta: ?[*]f32, delta_stride: c_int, d_out: ?[*]f32, out_stride: c_int, d_gamma: ?[*]const f32, gamma_stride: c_int, d_beta: ?[*]const f32, beta_stride: c_int, d_snap: ?[*]f32, nsnap: c_int, heads: c_int, nk: c_int, kd: c_int, vd: c_int, rows: c_int, stream: ?*anyopaque) c_int {
        _ = g;
        _ = .{ d_state, state_stride, d_k, k_stride, d_v, v_stride, d_q, q_stride, d_km, km_stride, d_delta, delta_stride, d_out, out_stride, d_gamma, gamma_stride, d_beta, beta_stride, d_snap, nsnap, heads, nk, kd, vd, rows, stream };
        return stubNoCudaKernels();
    }
    fn gdn_step_wy(g: backend.Graph, d_state: ?[*]f32, state_stride: c_int, d_k: ?[*]const f32, k_stride: c_int, d_v: ?[*]const f32, v_stride: c_int, d_q: ?[*]const f32, q_stride: c_int, d_km: ?[*]f32, km_stride: c_int, d_delta: ?[*]f32, delta_stride: c_int, d_out: ?[*]f32, out_stride: c_int, d_gamma: ?[*]const f32, gamma_stride: c_int, d_beta: ?[*]const f32, beta_stride: c_int, heads: c_int, nk: c_int, kd: c_int, vd: c_int, rows: c_int, stream: ?*anyopaque) c_int {
        _ = g;
        _ = .{ d_state, state_stride, d_k, k_stride, d_v, v_stride, d_q, q_stride, d_km, km_stride, d_delta, delta_stride, d_out, out_stride, d_gamma, gamma_stride, d_beta, beta_stride, heads, nk, kd, vd, rows, stream };
        return stubNoCudaKernels();
    }
    fn gdn_gamma(g: backend.Graph, d_a_raw: ?[*]const f32, d_b_raw: ?[*]f32, d_ssm_a: ?[*]const f32, d_dt_bias: ?[*]const f32, d_gamma: ?[*]f32, nv: c_int, stream: ?*anyopaque) c_int {
        _ = g;
        _ = .{ d_a_raw, d_b_raw, d_ssm_a, d_dt_bias, d_gamma, nv, stream };
        return stubNoCudaKernels();
    }
    // 仅 prefill 使用的孪生版本，现在和其它 launcher 一样经 `Launch` 到达
    // （见 CUDA 分支里的说明：它们曾绕过这条接缝）。
    fn conv_step_rows(g: backend.Graph, d_conv: ?[*]f32, d_y: ?[*]f32, d_wconv: ?[*]const f32, n: c_int, tokens: c_int, y_stride: c_int, stream: ?*anyopaque) c_int {
        _ = g;
        _ = .{ d_conv, d_y, d_wconv, n, tokens, y_stride, stream };
        return stubNoCudaKernels();
    }
    fn gdn_gamma_rows(g: backend.Graph, d_a_raw: ?[*]const f32, d_b_raw: ?[*]f32, d_ssm_a: ?[*]const f32, d_dt_bias: ?[*]const f32, d_gamma: ?[*]f32, nv: c_int, tokens: c_int, a_stride: c_int, b_stride: c_int, gamma_stride: c_int, stream: ?*anyopaque) c_int {
        _ = g;
        _ = .{ d_a_raw, d_b_raw, d_ssm_a, d_dt_bias, d_gamma, nv, tokens, a_stride, b_stride, gamma_stride, stream };
        return stubNoCudaKernels();
    }
    fn l2norm_strided(g: backend.Graph, d_x: ?[*]f32, row_offset: c_int, rows: c_int, n: c_int, eps: f32, tokens: c_int, row_stride: c_int, stream: ?*anyopaque) c_int {
        _ = g;
        _ = .{ d_x, row_offset, rows, n, eps, tokens, row_stride, stream };
        return stubNoCudaKernels();
    }
    fn scale_vec_strided(g: backend.Graph, d_x: ?[*]f32, offset: c_int, s: f32, n: c_int, tokens: c_int, stride: c_int, stream: ?*anyopaque) c_int {
        _ = g;
        _ = .{ d_x, offset, s, n, tokens, stride, stream };
        return stubNoCudaKernels();
    }
    fn rope64(g: backend.Graph, d_x: ?[*]f32, heads: c_int, hd: c_int, pos: c_int, base: f32, stream: ?*anyopaque) c_int {
        _ = g;
        _ = .{ d_x, heads, hd, pos, base, stream };
        return stubNoCudaKernels();
    }
    fn rope64_dpos(g: backend.Graph, d_x: ?[*]f32, heads: c_int, hd: c_int, d_ctl: ?*const anyopaque, stream: ?*anyopaque) c_int {
        _ = g;
        _ = .{ d_x, heads, hd, d_ctl, stream };
        return stubNoCudaKernels();
    }
    fn rope64_setenv(d_dyn: ?*anyopaque, pos: c_int, base: f32, stream: ?*anyopaque) callconv(.c) c_int {
        _ = .{ d_dyn, pos, base, stream };
        return stubNoCudaKernels();
    }
    fn rope64_rows(g: backend.Graph, d_x: ?[*]f32, rows: c_int, heads: c_int, hd: c_int, pos0: c_int, base: f32, stream: ?*anyopaque) c_int {
        _ = g;
        _ = .{ d_x, rows, heads, hd, pos0, base, stream };
        return stubNoCudaKernels();
    }
    fn rope_rows_dim(g: backend.Graph, d_x: ?[*]f32, rows: c_int, heads: c_int, hd: c_int, pos0: c_int, base: f32, rotary_dim: c_int, stream: ?*anyopaque) c_int {
        _ = g;
        _ = .{ d_x, rows, heads, hd, pos0, base, rotary_dim, stream };
        return stubNoCudaKernels();
    }
    fn rope64_rows_dpos(g: backend.Graph, d_x: ?[*]f32, rows: c_int, heads: c_int, hd: c_int, d_ctl: ?*const anyopaque, stream: ?*anyopaque) c_int {
        _ = g;
        _ = .{ d_x, rows, heads, hd, d_ctl, stream };
        return stubNoCudaKernels();
    }
    fn l2norm_qk_scale(g: backend.Graph, d_x: ?[*]f32, nk: c_int, kd: c_int, k_offset: c_int, eps: f32, qscale: f32, stream: ?*anyopaque) c_int {
        _ = g;
        _ = .{ d_x, nk, kd, k_offset, eps, qscale, stream };
        return stubNoCudaKernels();
    }
};

// ==== 激活的几何 =============================================================

/// 针对 `ttype` 的 matvec 所需要的激活块大小（以**元素**计）。
///
/// 有两种格式而不是一种，因为量化 matvec 是按权重类型表驱动的，激活块必须与
/// 权重块匹配：
///   * Q8_0 权重用 32 元素块 -> Q8_0 激活（34 B / 32）。
///   * BF16/F16 权重（每元素一个 2 字节块，被框架成 256 元素的 GGUF row-block）
///     同样由 Q8_0 家族做点积 —— 见 `backend_cpu.actFamily`，那是这件事的唯一
///     来源。
///   * 本后端加载的其它打包类型一律用 256 元素块（Q4_K/Q6_K/IQ*/... 全都消费
///     Q8_K）-> Q8_K（292 B / 256）。
///
/// 只按 `gguf.blockElems` 分支曾是一个真实 bug：BF16/F16 的 `blockElems == 1`，
/// 于是它们落进 Q8_K 家族，launcher 把激活按 292 B / 256 元素分配，而权重点积
/// 读的是 Q8_0 块。
fn actBlockElems(ttype: gguf.TensorType) usize {
    return if (cpu_mod.actFamily(ttype) == .q8_0)
        gguf.blockElems(.q8_0)
    else
        gguf.blockElems(.q8_k);
}

/// `ttype` 的 matvec 输入下，一个激活块的打包字节数。
fn actBlockBytes(ttype: gguf.TensorType) usize {
    return if (cpu_mod.actFamily(ttype) == .q8_0)
        gguf.blockBytes(.q8_0)
    else
        gguf.blockBytes(.q8_k);
}

/// 宽 `cols` 的输入所需的量化激活字节数：整块，向上取整的方式与
/// `backend_cpu.q8ScratchBytes` / `q8kScratchBytes` 完全一致，因此 CPU 后端接收
/// 过的配置在这里不会把设备缓冲撑爆。
fn actBytes(ttype: gguf.TensorType, cols: usize) usize {
    const be = actBlockElems(ttype);
    return (cols + be - 1) / be * actBlockBytes(ttype);
}

fn oomMsg(what: []const u8) void {
    std.debug.print("cuda backend: {s} device buffer allocation failed\n", .{what});
}

/// 某个时间戳到现在的纳秒数。`durationTo(...).raw.nanoseconds` 是 `i96`，而
/// 计数器是 `u64` 且只会累加（单调时钟），所以这里钳位而不是触发陷阱。

fn durNs(t0: Io.Clock.Timestamp, io: Io) u64 {
    const d = t0.durationTo(Io.Clock.Timestamp.now(io, .awake)).raw.nanoseconds;
    return if (d > 0) @intCast(@min(d, std.math.maxInt(u64))) else 0;
}

// ---- 逐 slot 的性能剖析（仅 $QWEN35_PROF） --------------------------------

/// `backend.Exec` 的各个 slot，每个一项；另外还有两个权重/scratch slot —— 它们
/// 不算在每 token 的预算里，但值得一看（那是加载期与分配的开销）。
///
/// 顺序就是报告顺序，所以回读 `PROF-SLOT` 转储时热项排在前面；不过
/// `reportSlotStats` 最终还是会按总耗时排序。
pub const Slot = enum(u8) {
    matvec,
    matvec_batchk,
    dequant_row,
    load_linear,
    rmsnorm,
    rope64,
    rope64_rows,
    l2norm,
    attn_decode,
    /// 块级 prefill attention（`VTable.attnChunk`）。与 `attn_decode` 分开，好让
    /// 剖析把两种形状分开显示：prefill 每层应该降到少数几次调用，而 decode 保持
    /// 每 token 一次。
    attn_chunk,
    gdn_gamma,
    gdn_prelude,
    gdn_step,
    conv_step,
    silu,
    mul_silu,
    mul_sigmoid,
    softmax,
    swiglu,
    scale_vec,
    add_resid,
    copy_range,
    deinterleave_qg,
    zero_buf,
    copy_from_host,
    copy_to_host,
    argmax,
    sample,
    alloc_buf,
    alloc_scratch,
    alloc_scratch_k,
    const_buf,
};

/// `PROF-COPY` 明细所用的元素个数分桶。固定而非动态：这个模型实际发出的宽度只有
/// 少数几个不同的值，而固定数组让这些插桩在热路径上不产生分配。
///
/// 顺序只在"`copyBinOf` 按序扫描它"这一点上有意义；边界是升序的，所以第一个
/// 命中就是最紧的桶。
const copy_bin_bounds = [_]usize{ 128, 256, 512, 1024, 2048, 4096 };
const copy_bin_count = copy_bin_bounds.len + 1;


/// 一个 `copyRange` 家族：调用了多少次，以及在主机上花了多久。
const CopyBin = struct {
    n: u32 = 0,
    ns: u64 = 0,
    /// 搬运的元素总数；除以 `n` 就是每次调用的负载 —— 一个桶借此暴露出它是
    /// 每头 512 B 的那一族，还是整行 embedding。
    elems: u64 = 0,
};

/// 给定元素个数对应最紧的那个桶（`copy_bin_count-1` = 大于最后一个边界，例如
/// 27B 上的一次 `kv_dim` 缓存写）。
fn copyBinOf(elems: usize) usize {
    for (copy_bin_bounds, 0..) |b, i| {
        if (elems <= b) return i;
    }
    return copy_bin_bounds.len;
}

/// `Slot` 的条目数；用于给 `stat_ns_slot` / `stat_n_slot` 定尺寸。
///
/// 写死而不是从 `@typeInfo` 推导：Zig 0.17 重做了 `@typeInfo`，而 `slotName` 的
/// 穷尽 `switch` 已经让"漏掉一个枚举成员"变成编译错误，所以这里只需保持在最后
/// 一个枚举值（`const_buf`）之后加一即可。
const slot_count = @intFromEnum(Slot.const_buf) + 1;

/// `$QWEN35_PROF_GPU` 用于逐 slot 设备计时的 event 对数量。一个 decode token 会
/// 发出约 600 次计时 slot 调用（0.8B 实测 599 次），所以 1024 对足够覆盖一个
/// token 并有余量；溢出的 token 会计入 `stat_gpu_overflow`，而不是被静默丢弃。
///
/// 27B 需要的多得多：一个 decode token 发出约 2273 次计时 slot 调用（由
/// PROF-SLOT 实测：111k 次调用 / 49 token）。1024 对时池子溢出了 93566，导致
/// 27B 上的 PROF-GPU 明细不可用，设备侧的归因只能从 `QWEN35_ABLATE` 反推而无法
/// 实测。4096 对可以带余量覆盖一个 27B token。这是一条只用于调试的路径（环境
/// 变量 `$QWEN35_PROF_GPU`）；event 由 `gpuEvEnsure` 惰性创建、析构时销毁，所以
/// 提高这个数在不做 profiling 时没有运行期代价。
///
/// 再次提高：27B decode token 发出约 2273 次**计时**调用，但每次调用会占一个对，
/// 而池子还会被排空那个 slot 消耗，所以 4096 仍然溢出（实测：一次 60 token 的
/// 运行溢出 7519，也就是 0.5 ms 以下的归因一直在被静默丢弃）。16384 是下一个有
/// 真实余量的 2 的幂；它只在调试路径上花费 2 * 16384 * sizeof(cudaEvent_t)。
const gpu_ev_capacity = 16384;

/// `PROF-SLOT` 那一行用的可读 slot 名。按 `@intFromEnum` 索引，所以一个
/// `comptime` 校验过的长度把它们与枚举绑在一起。
fn slotName(s: Slot) []const u8 {
    return switch (s) {
        .matvec => "matvec",
        .matvec_batchk => "mv_batchk",
        .dequant_row => "dequant_row",
        .load_linear => "load_linear",
        .rmsnorm => "rmsnorm",
        .rope64 => "rope64",
        .rope64_rows => "rope64_rows",
        .l2norm => "l2norm",
        .attn_decode => "attn_decode",
        .attn_chunk => "attn_chunk",
        .gdn_gamma => "gdn_gamma",
        .gdn_prelude => "gdn_prelude",
        .gdn_step => "gdn_step",
        .conv_step => "conv_step",
        .silu => "silu",
        .mul_silu => "mul_silu",
        .mul_sigmoid => "mul_sigmoid",
        .softmax => "softmax",
        .swiglu => "swiglu",
        .scale_vec => "scale_vec",
        .add_resid => "add_resid",
        .copy_range => "copy_range",
        .deinterleave_qg => "deinterleave_qg",
        .zero_buf => "zero_buf",
        .copy_from_host => "copy_from_host",
        .copy_to_host => "copy_to_host",
        .argmax => "argmax",
        .sample => "sample",
        .alloc_buf => "alloc_buf",
        .alloc_scratch => "alloc_scratch",
        .alloc_scratch_k => "alloc_scratch_k",
        .const_buf => "const_buf",
    };
}

/// True for the slots whose forwarder drains the stream, so the recorded host
/// interval really contains device execution.  Everything else records
/// submission cost only (see the note on `stat_ns_slot`).
fn slotSyncs(s: Slot) bool {
    return switch (s) {
        // MAY drain -- but in the steady state it does not.  `downloadOut` (a
        // blocking `cudaMemcpy` D2H) only runs when the caller handed us a HOST
        // output buffer; when `dBufPtr(out)` resolves, the kernel writes that
        // device region directly and no copy back happens.  The engine passes
        // device buffers on the whole decode path, so this flag marks a
        // capability, not an observed wait: measured with per-call probes, not
        // one of the ~24k matvecs in a 48-token decode of the 27B model blocked
        // on the host, and the slot's cost is the launch (~0.03 ms/call).  Only
        // the first-touch constant uploads (one per layer weight, ~353 of them)
        // wait, and those land in the prefill chunk, not in decode.
        .matvec, .matvec_batchk, .dequant_row => true,
        // Reads an index / floats back to the host.
        .argmax, .copy_to_host, .sample => true,
        // Host-side bookkeeping, no device work to attribute.
        .load_linear, .const_buf, .alloc_buf, .alloc_scratch, .alloc_scratch_k => false,
        else => false,
    };
}

/// One device copy of a host weight.  `ttype` is null for the `.f32` tensors
/// (which have no quant type); `rows`/`cols` are stored because an f32 slice
/// carries no geometry of its own and both matvec paths need the launch shape.
const DevRow = struct {
    host: [*]const u8,
    dev: *anyopaque,
    bytes: usize,
    ttype: ?gguf.TensorType,
    rows: usize,
    cols: usize,
};

const Geometry = struct { rows: usize, cols: usize };

/// Resolved kernel entry points.  They are link-time symbols from
/// `src/kernels/cuda/*.cu` (never from cudart), so `Kernels` above is the single
/// place to audit against `src/kernels/cuda/api.h`.
///
/// ABI, as landed in `src/kernels/cuda/api.h`:
///   ttype   : `@intFromEnum(gguf.TensorType)` -- GGML `u32` tag.  api.h's
///             `Q35_TT_*` values are copied from that enum, so the numbers agree
///             without a translation table (they are even in the same order).
///   nblocks : count of *type-native* blocks (cols / blockElems), not rows, and
///             not elements -- 32 for the q8_0/q4_*/q5_*/iq4_nl family, 256 for
///             the K/IQ family.
///   stream  : `cudaStream_t` as an opaque pointer; null = default stream.
/// Everything returns `cudaError_t` as an int (0 == success).
/// State of the IQ codebook upload (see `CudaBackend.uploadIqTables`).
/// `failed` is terminal on purpose: the upload is one cudaMalloc + twelve
/// cudaMemcpy + one kernel, so a failure is a device/driver-level problem, not
/// something a retry would fix, and the caller must not proceed to index the
/// (never-written) device pointers.
const IqTablesState = enum { pending, done, failed };

/// The two `dRope64` implementations, only for the startup log line.
const RopePath = enum { host, device };


// Buffers for formatted error/description strings.  Static because
// `describe`/`name` return slices with no allocator in scope (same pattern as
// the CPU backend, which reports through `std.debug.print`).
var msg_buf: [512]u8 = undefined;
/// One `dAllocBuf` region: the cudaMalloc base and its length in f32 elements.
/// `isDevBuf` needs both to recognise a `span()`ed view of our own memory.
const DevRegion = struct { ptr: *anyopaque, len: usize };

/// Upper bound on the number of transformer layers whose `rope64` /
/// `attn_decode` nodes the graph tracks.  The engine's default is 24 and
/// Capacity for the per-token node arrays, **not** a layer count.
///
/// The name is a leftover from the first cut, which assumed one `rope64` node
/// per layer; the engine actually calls `rope64` once per head-bearing tensor
/// (`q` and `k`, and again per head for some variants), so the node count for a
/// decode step is a small multiple of `n_layers`, not `n_layers`.  0.8B needs 48
/// and 27B needs several hundred, which is why the old 64 tripped the capacity
/// check on the first real run and the feature switched itself off -- correct,
/// but useless.
///
/// Each entry costs a few hundred bytes (args + params + ptr arrays), so 1024 is
/// ~800 KiB -- acceptable for a backend that already holds the weights.
///
/// Measured on the 27B (65 layers, 24 q-heads, 4 kv-heads, interval 4): a decode
/// token captures **448 `rope64` + 16 `attn_decode` + 32 kv-append** nodes.  The
/// earlier claim here ("128 rope64, room to spare") was off by 3.5x and made the
/// 256 capacity silently reject the graph on every 27B run.  Exceeding the
/// capacity is not a correctness problem: the graph path switches itself off and
/// falls back to direct launches, so a too-small value is invisible in the exit
/// code and only shows up as decode throughput.  0.8B needs 48.
const graph_max_layers: usize = 1024;

/// One flag per op class the captured decode step can contain.  See
/// `CudaBackend.g_op_seen` for why the gating exists at all.
const GhOpFlags = struct {
    matvec: bool = false,
    rmsnorm: bool = false,
    rmsnorm_rows: bool = false,
    add_resid: bool = false,
    scale_vec: bool = false,
    silu_in_place: bool = false,
    mul_silu: bool = false,
    mul_sigmoid: bool = false,
    swiglu: bool = false,
    softmax: bool = false,
    zero_buf: bool = false,
    gdn_step: bool = false,
    gdn_step_rows: bool = false,
    gdn_gamma: bool = false,
    conv_step: bool = false,
    attn_decode: bool = false,
    rope64: bool = false,
    rope64_rows: bool = false,
    kv_append: bool = false,
};

/// Node budget for one captured decode step.  Used for the per-function arrays
/// (a 0.8B decode step is ~800 nodes) and for the ordering buffers
/// (`g_order_nodes`), which are the reason it is this large.
const capture_nodes_max = 4096;

/// See the `g_state` field comment in `CudaBackend` for the transitions.
const GraphState = enum { off, warm, rec, live };

/// How many argument slots `argHoldsU32`/`argHoldsInt` will inspect.  The
/// widest launcher in this backend (`gdn_step_rows`) takes 19 arguments, so 24
/// is comfortably above every real signature while keeping a linear scan of an
/// argument array trivially cheap (it runs once per kernel node, once per
/// process, at capture time).
/// Largest kernel signature this backend launches.  `attn_decode`'s device
/// kernel takes 13 arguments and `gdn_step_rows`' takes 17; 24 leaves headroom
/// and costs `24 * 16` bytes per tracked node, i.e. nothing.
const kernel_max_args: usize = 24;

/// 一张 XG decode 图最多记录多少个 op。实测 27B 一个 decode token 的发射
/// 次数约 3000（`QWEN35_PROF` 各 slot 的 calls 之和），4096 留出余量。


/// 每步设备控制块里 `kv_slot` 的 int 下标（`xd.h` 的 `XD_CTL_KV_SLOT`）。
///
/// 设备块是 `int32_t[6]`，字段顺序**必须**与 `backend.GraphCtl` 一致；那边
/// 有 `@sizeOf` 断言，这里补一条顺序断言 —— 设备变体（如 `attn_chunk_dyn`）
/// 按固定下标读它，错一位就是静默读到别的标量。
const ctl_kv_slot: usize = 2;

comptime {
    // `GraphCtl` 里 `kv_slot` 的字节偏移必须正好是这个下标处的 i32。
    const GC = @import("backend.zig").GraphCtl;
    if (@offsetOf(GC, "kv_slot") != ctl_kv_slot * @sizeOf(c_int)) {
        @compileError("ctl_kv_slot 与 GraphCtl 的字段顺序不符");
    }
}

/// 槽数：decode / verify / nextn 各一张。槽号必须和 `engines/qwen35.zig` 的
/// `graph_slot_*` 常量一致（引擎传进来，后端只做边界检查）。
const graph_slot_count: usize = 3;

/// One argument slot of a tracked kernel: `width` bytes of value storage, so a
/// f32 (4) and a 64-bit pointer (8) both fit without a variant type.  Alignment
/// is 8 because the widest thing stored is a pointer.
const ArgSlot = struct { bytes: [8]u8 align(8) = @splat(0) };

/// One graph's captured state.
///
/// Three instances exist:
///   * slot 0 -- the DECODE graph: one `forwardLogitsSampled` (1 row).
///   * slot 1 -- the VERIFY graph: one `forwardChunkVerify` (k rows).
///   * slot 2 -- the NEXTN graph: one MTP draft block (1 row).
///
/// They cannot share one graph: the node set (and every `ne`/`nb` the
/// backend compares in `graphUpdateRequired`) differs between a 1-row
/// and a k-row forward, so a shared slot would re-capture on every
/// switch.  The stream and the two ordering events ARE shared
/// (`g_stream`/`g_ev`/`g_ev_out`): capture and replay on one stream is
/// legal for any number of graphs.
const GraphSlot = struct {
    state: GraphState = .off,
    rope_dyn: ?*anyopaque = null,
    rope_dyn_pos: u32 = 0xFFFF_FFFF,
    rope_dyn_base: f32 = 0.0,
    attn_dyn: ?*anyopaque = null,
    /// 设备端 int：attnChunk 的起始行。graph 重放时从它读。
    d_attn_slot: ?*anyopaque = null,
    /// Host 镜像，只有值变了才 H2D。
    d_attn_slot_host: c_int = -1,
    graph: ?*anyopaque = null,
    exec: ?*anyopaque = null,
    rope_nodes: [graph_max_layers]?*anyopaque = @splat(null),
    rope_n: usize = 0,
    attn_nodes: [graph_max_layers]?*anyopaque = @splat(null),
    attn_n: usize = 0,
    rope_args: [graph_max_layers]c_uint = @splat(0),
    rope_base: [graph_max_layers]f32 = @splat(0),
    attn_args: [graph_max_layers]c_int = @splat(0),
    rope_params: [graph_max_layers]cudart.KernelNodeParams = @splat(undefined),
    attn_params: [graph_max_layers]cudart.KernelNodeParams = @splat(undefined),
    rope_argv: [graph_max_layers][kernel_max_args]ArgSlot = @splat(@splat(.{})),
    attn_argv: [graph_max_layers][kernel_max_args]ArgSlot = @splat(@splat(.{})),
    rope_kptr: [graph_max_layers][kernel_max_args]?*anyopaque = @splat(@splat(null)),
    attn_kptr: [graph_max_layers][kernel_max_args]?*anyopaque = @splat(@splat(null)),
    expect_rope: usize = 0,
    expect_attn: usize = 0,
    rec_pos: u32 = 0,
    rec_slot: usize = 0,
    base_captured: f32 = 0,
    kv_dst_captured: [graph_max_layers]?*anyopaque = @splat(null),
    kv_stride: [graph_max_layers]usize = @splat(0),
    expect_kv: usize = 0,
    order_nodes: [capture_nodes_max]?*anyopaque = undefined,
    order_n: usize = 0,
    cluster_n: usize = 0,
    cluster_appends: usize = 0,
    kv_partner: [graph_max_layers]usize = @splat(0),
    kv_nodes: [graph_max_layers]?*anyopaque = @splat(null),
    kv_n: usize = 0,
    kv_params: [graph_max_layers]cudart.KernelNodeParams = @splat(undefined),
    kv_argv: [graph_max_layers][kernel_max_args]ArgSlot = @splat(@splat(.{})),
    kv_kptr: [graph_max_layers][kernel_max_args]?*anyopaque = @splat(@splat(null)),
    kv_dst_cells: [graph_max_layers]?*anyopaque = @splat(null),
    capturing: bool = false,
    capturing_step: bool = false,
    op_seen: GhOpFlags = .{},
    failed: bool = false,

    // ---- 显式计算图（XG，S3c）--------------------------------------------
    //
    // 与上面那批旧机制字段并存，直到 S3d 把它们删掉。`xd_phase` 非 `.off`
    // 时走 XG 的路，旧字段一律不看。
    //
    // 没有 op 表、没有节点列表：capture 期间 op 照常发射，CUDA 自己记录，
    // 重放就是一次 `GraphLaunch`。这是 llama.cpp / ds4 / Strata 三者共同的
    // 结构，也是唯一能覆盖 `matvec` 那种几何由运行时分发决定的 op 的做法。
    xd_phase: XdPhase = .off,
    xd_gexec: ?*anyopaque = null,
    xd_graph: ?*anyopaque = null,
    /// 本次重放是否已经把图交给 GPU：只有越过 launch 的失败才意味着
    /// "状态已推进、不能再重跑本步"。
    xd_launched: bool = false,
    /// capture 已经打开、尚未关闭。防止同一调用栈里重入 `graphBegin`。
    xd_capturing: bool = false,
    /// 这一槽的图是按多少行录的（`GraphCtl.rows` 的原值）。只有 `.verify`
    /// 需要它：它的行数是草稿块长度，逐轮会变，而一张图的节点数和 buffer
    /// 行数都烘死在录制那一刻。
    xd_rows: u32 = 0,
};

/// XG 一张图的状态。`warm` 是"下一步录制"（`absent → warm → live`），
/// 与 `graph_registry.zig` 的状态机同义，但因为 `graph_registry` 的 key
/// 是 `(signature, kind, rows)` 而 decode 只有一种，这里先用单槽状态。
const XdPhase = enum { off, rec, live };

pub const CudaBackend = struct {
    /// Ring size for the per-call timing events; see `ev_pool`.
    ///
    /// Measured: with 256 slots (128 usable pairs) the host lapped the device
    /// and 228 of 496 samples were dropped on a 525-token prefill.  That is
    /// itself the useful reading -- the host queues a whole prefill's worth of
    /// kernels before the GPU catches up -- so the ring is sized at 2048 slots
    /// (1024 pairs, two prefills) and the drop counter is a real alarm.
    pub const EV_POOL: usize = 2048;

    /// Upper bound on a believable single `matvecBatchK` interval, in ns.
    /// 50 ms is >30x the largest real kernel here (1.6 ms), so it only fires on
    /// a broken measurement, never on a slow but genuine one.
    pub const MAX_BK_NS: u64 = 50_000_000;

    alloc: std.mem.Allocator,
    /// Metadata/geometry authority (see the file comment).  Never used for
    /// arithmetic -- the one exception is the opt-in `$QWEN35_MVCHECK`
    /// cross-check in `mvCheck`, which is diagnostics and cannot affect output.
    cpu: cpu_mod.CpuBackend,
    /// The GGUF being served.  Needed only by `vLoadLinear` to recover the
    /// rows/cols of a `.f32` weight: `LinearW.f32` is a flat slice with no
    /// geometry, and the device launches need `cols`.  Every quant row -- f16
    /// and bf16 included -- carries its own rows/cols, so none of them uses this.
    g: *const gguf.File,
    /// KV cache storage format for this load (`setKvType` before the first
    /// `allocKv`): `allocKv` sizes, `kvAppend` writes, attention reads.
    /// Mirrored into the embedded CPU fallback (it reads the same cache on
    /// the CPU net path).
    kv_type: backend.KvType = .f32,
    cu: ?cudart.Open = null,
    /// Why device init failed; re-reported from `allocScratch` so `Engine.load`
    /// turns it into a load error instead of silently running on the CPU.
    init_error: []const u8 = "cuda backend: not initialized",

    /// Whether the IQ codebooks are on the device.  `pending` -> `done` on the
    /// first successful `uploadIqTables`, `failed` if that never happened (and
    /// then stays put: see `uploadIqTables`).
    iq_tables_state: IqTablesState = .pending,

    /// Host data pointer -> device copy.  Keyed by `q.data.ptr` (mmap view) or the
    /// arena slice pointer (f32), so a `LinearW` handed back to us resolves
    /// without threading a cookie through the `Exec` interface.
    dev_rows: std.ArrayList(DevRow) = .empty,
    /// Sum of successful uploads: the resident weight footprint.
    stat_uploaded_bytes: usize = 0,
    /// Bytes moved host->device / device->host on the PER-TOKEN path, i.e. every
    /// `cudaMemcpy` this file issues after the weights are resident.  P2c's
    /// whole point is that this collapses: before it, every elementwise slot
    /// bounced its operands through staging, so each operator cost 2 transfers.
    /// Reported by `$QWEN35_STAT` so the claim is measurable rather than stated.
    stat_h2d_bytes: usize = 0,
    stat_d2h_bytes: usize = 0,
    /// How many times a numeric slot fell back to a host round trip because the
    /// caller passed a *host* `Buf` instead of a device region (`dBufPtr` null).
    /// These are the only two paths that cost a memcpy per call after P2d, and
    /// on the engine's per-token path both must stay at zero -- which is what
    /// makes "no per-token H2D/D2H" an assertion rather than a hope.
    /// `rope64`'s own round trip is counted separately (`stat_rope_trips`).
    stat_host_act_uploads: usize = 0,
    stat_host_out_downloads: usize = 0,
    /// `dRope64` round trips (one H2D + one D2H per rotated head, per token).
    /// The deliberate remainder described at that function: it buys bit-exact
    /// rotation, and the report quantifies it instead of hiding it.  Always 0
    /// under `$QWEN35_CUDA_ROPE=device`.
    stat_rope_trips: usize = 0,
    /// `dRope64` heads rotated by the device kernel.  Its complement to the
    /// total rotation count is `stat_rope_trips`, so the `CUDA-M` line says
    /// which of the two paths actually ran rather than which one was asked for.
    stat_rope_devices: usize = 0,
    /// `$QWEN35_STAT_TALKY` grouping state: last uploaded element count and how
    /// many consecutive uploads had that size.
    debug_h2d_len: usize = 0,
    debug_h2d_same: usize = 0,
    /// `$QWEN35_SYNC=1`: barrier after every matvec, to test whether a result
    /// depends on an unsynchronized producer.  Diagnostic only -- correctness
    /// does not need it (default stream ordering + synchronous memcpy).
    sync_debug: bool = false,
    /// Device time per slot, so the `CUDA-M` line answers "is the GPU actually
    /// doing the work, and where does it go".
    stat_ns_matvec: u64 = 0,
    stat_ns_batchk: u64 = 0,
    stat_ns_dequant: u64 = 0,
    stat_n_matvec: usize = 0,
    stat_n_batchk: usize = 0,
    stat_n_dequant: usize = 0,
    /// Per-quant-type matvec counters, indexed by `@intFromEnum(TensorType)`.
    /// `TensorType` has holes, so index by the raw tag and size for the largest
    /// value we count (tq2_0 = 35).
    stat_by_type: [64]usize = @splat(0),
    /// Nanoseconds spent per backing type, summed over the same calls that
    /// `stat_by_type` counts.  Diagnostics only ($QWEN35_PROF): answers "which
    /// quant type owns the matvec time", which the count alone cannot.
    stat_ns_by_type: [64]u64 = @splat(0),
    /// Which types the `$QWEN35_MVCHECK` cross-check has already covered (see
    /// `mvCheck`).  Per type, not per process: a model has one type per tensor
    /// family, so a single sample would leave e.g. iq4_xs unchecked.
    mvcheck_seen: [64]bool = @splat(false),
    /// Clock for the slot timings.  Supplied by `setThreads`, which the engine
    /// calls before the first matvec (same contract as `backend_cpu.prof_io`);
    /// only read when `$QWEN35_PROF` output is wanted.
    io: Io = undefined,

    // ---- GPU-side timing ($QWEN35_PROF only) --------------------------------
    //
    // `durNs()` above is host wall clock wrapped around an *asynchronous*
    // launch, so `t_mv_ms` / `t_bk_ms` measure submission, not execution.  The
    // fields here come from CUDA events, which timestamp a point in the stream;
    // `ev_end - ev_begin` is what the device actually ran.
    //
    // Cost model, and why this is PROF-gated: a record is ~1 us, but the
    // *synchronize* needed to read an elapsed time drains the queue.  Measuring
    // every batchk call therefore serializes the whole prefill; the fields below
    // are only maintained when `ev_on` is set, and `ev_on` is only set when
    // `$QWEN35_PROF` was in the environment.  With PROF off, `gpuTimingOn()`
    // returns false before any event is created or touched.
    /// Same condition as every other `$QWEN35_PROF` check in this file.
    ev_on: bool = false,
    /// Created once (`ensureEvents`) and reused, never per token: an
    /// `cudaEventCreateWithFlags` in the hot path would cost more than the
    /// kernels being measured.
    ev_created: bool = false,
    /// Around one `matvecBatchK` call (one weight, one chunk of K tokens).
    ///
    /// These are the ONLY pair recorded per call; the interval is not read here
    /// any more (see `EV_POOL`).  Kept because `batchKTimingBegin` uses them to
    /// find the prefill window start on the first call.
    ev_bk_a: ?*anyopaque = null,
    /// ttype of the batchK call currently bracketed by `ev_bk_a`/`ev_bk_b`, so
    /// `batchKTimingEnd` can attribute the device interval to a type.  Only
    /// meaningful while `gpuTimingOn()`; written by `batchKTimingBegin`.
    bk_ttype: i32 = -1,
    /// Ring of event pairs for the per-call batchK intervals, plus the pending
    /// samples that still have to be read.
    ///
    /// The point of the ring is to avoid a host sync per kernel.  The old code
    /// did `cudaEventSynchronize` in `batchKTimingEnd`, once per weight per
    /// chunk, which drains the queue every ~1.3 ms and costs 41% of prefill
    /// throughput (436 -> 309 tok/s) -- so the numbers it produced described a
    /// run that does not exist, and `bk_gpu_ms` was really an upper bound that
    /// silently included the stall.  Recording into a rotating pair and reading
    /// it one call later (or at the end of the window) keeps the queue deep.
    ///
    /// 256 pairs x 2 events covers a whole 525-token prefill's 249 matvecs
    /// twice over, so in practice every sample is already complete when read.
    ev_pool: [EV_POOL]?*anyopaque = @splat(null),
    ev_pool_wr: usize = 0,
    /// Samples recorded but not yet read: (slot, ttype).
    ev_pending: [EV_POOL]struct { slot: u32 = 0, ttype: i32 = -1, live: bool = false } = @splat(.{}),
    ev_pending_n: usize = 0,
    /// Begin/End pairing counters.  `batchKTimingBegin` and `batchKTimingEnd`
    /// must fire the same number of times; a mismatch means some early return
    /// in `matvecBatchK` skipped its End, in which case the ring slot is reused
    /// while still pending and the interval covers unrelated work -- the
    /// double-counting that made low-frequency types read 20-30x too slow.
    stat_n_bk_begin: usize = 0,
    stat_n_bk_end: usize = 0,
    /// Samples the ring had to throw away because the device was `EV_POOL/2`
    /// launches behind.  Reported so a low `bk_n` is never mistaken for the
    /// model issuing fewer matvecs.
    stat_n_ev_dropped: usize = 0,
    /// Intervals rejected by the `MAX_BK_NS` sanity clamp.  Non-zero means the
    /// per-call timing is not trustworthy for this run.
    stat_n_ev_suspect: usize = 0,
    /// `$QWEN35_PROF`: per-type device ns / call count from the batchK events.
    /// `stat_ns_batchk_gpu` is the same intervals summed over all types; these
    /// say WHERE they went, which is what picks the next optimisation target.
    stat_ns_bk_by_type: [64]u64 = @splat(0),
    stat_n_bk_by_type: [64]u32 = @splat(0),
    ev_bk_b: ?*anyopaque = null,
    /// Around the whole prefill window: `profReset` -> first `matvecBatchK` of
    /// the decode phase.  The wall clock version of the same window is
    /// `stat_ns_prefill_wall`, so the difference is the *host* gap that no
    /// kernel covers.
    ev_pre_a: ?*anyopaque = null,
    ev_pre_b: ?*anyopaque = null,
    /// Around one `argmax` launch + its sync D2H copy.  `ev_arg_a` is recorded
    /// on the device *before* the argmax kernel, `ev_arg_b` *after* it, so
    /// `ev_arg_b - ev_arg_a` is the tail of the queue as the device saw it.
    /// Measured only under `$QWEN35_PROF`; see `dArgmax_body`.
    ev_arg_a: ?*anyopaque = null,
    ev_arg_b: ?*anyopaque = null,
    /// Marks the device position at the END of the previous token (recorded right
    /// after that token's argmax copy).  `tok_end - prev_tok_end` is one whole
    /// decode token as the device saw it: every kernel of that token.  The gap
    /// between that and the host's wall clock for the same token is the host's
    /// own share (submission + the blocking copy).
    ev_tok_end: ?*anyopaque = null,
    ev_prev_tok_end: ?*anyopaque = null,
    tok_end_valid: bool = false,
    stat_ns_token_dev: u64 = 0,
    stat_n_token_dev: u64 = 0,
    /// Device time of the argmax kernel's own launch window, summed (`$QWEN35_PROF`).
    stat_ns_argmax_gpu: u64 = 0,
    /// Host wall clock spent inside `dArgmax_body`, summed (`$QWEN35_PROF`).  The
    /// difference against `stat_ns_argmax_gpu` is the host's share of the drain:
    /// driver submission + the blocking copy, rather than device execution.
    stat_ns_argmax_host: u64 = 0,
    stat_n_argmax_timed: u64 = 0,
    /// True between `profReset` and the event read that closes the prefill
    /// window; decides whether a `matvecBatchK` call is prefill or decode.
    prefill_open: bool = false,
    /// `ev_pre_a` has been recorded (the window's first device position).
    ev_pre_started: bool = false,
    /// Host clock origin of the prefill window (`openPrefillWindow`).
    prefill_t0: Io.Clock.Timestamp = undefined,
    /// Host clock origin of the batchk span inside the prefill window.
    bk_wall_started: bool = false,
    bk_wall_t0: Io.Clock.Timestamp = undefined,
    /// GPU kernel time summed over every timed `matvecBatchK` call, in ns.  The
    /// span recorded is the call's first recorded point to its last, so it
    /// covers the activation quantize kernels *and* `mul_mat_vec` -- i.e. all
    /// the device work that call issued (not just the matmul).
    stat_ns_batchk_gpu: u64 = 0,
    /// Number of batchk calls that contributed to `stat_ns_batchk_gpu`.
    stat_n_batchk_gpu: usize = 0,
    /// Device event span of the whole prefill window (first prefill record to
    /// the event that closes it), in ns.  This is a *span*, not busy time: it
    /// includes device idle while the host is submitting, so it should match
    /// `stat_ns_prefill_wall` and is reported as a consistency check.
    stat_ns_prefill_span: u64 = 0,
    /// Host wall clock of that same prefill window, in ns.
    stat_ns_prefill_wall: u64 = 0,
    /// Host wall clock from the first to the last `matvecBatchK` of prefill:
    /// `wall - batchk_gpu` is then the host-side slack (launch gaps, the
    /// quantize launches' submission, the synchronous memcpys).
    stat_ns_prefill_bk_wall: u64 = 0,
    /// `cudaEventElapsedTime` resolution guard: number of reads that returned a
    /// cudart error (normally 0).
    stat_n_ev_err: usize = 0,
    /// Kernel launches issued by the batchk path during prefill (K quantizes +
    /// one mul_mat_vec per call).  A single `+= 1` per launch, so counting it
    /// costs nothing measurable and works with or without events -- which is
    /// what makes the launch *rate* comparable between a PROF run and a plain
    /// one.
    stat_n_launch_bk: usize = 0,
    /// Host time spent inside `cudaEventSynchronize` while measuring.  This IS
    /// the interference the events add, so it is reported rather than assumed
    /// small: an event read drains the queue, so every timed call waits for its
    /// own kernels before returning.
    stat_ns_ev_sync: u64 = 0,

    // ---- per-slot host wall clock ($QWEN35_PROF only) ----------------------
    //
    // `t_mv_ms` / `t_bk_ms` / `t_dq_ms` only cover the three slots that had a
    // counter.  Everything else the engine calls through `backend.Exec`
    // (rmsnorm, rope64, attention, the GDN state updates, the elementwise
    // kernels, lm_head, argmax) was invisible, so the prefill budget could not
    // be closed: wall minus (matvec + batchk + dequant) left ~2/3 of the
    // window unaccounted for.
    //
    // Every forwarder below now wraps its body in `slotEnter`/`slotExit`, which
    // are `QWEN35_PROF`-gated (a bool test when off, so the plain path is
    // unchanged).  These are HOST wall-clock intervals around the forwarder
    // call, which is exactly what the prefill window is measured in -- so the
    // slots and `stat_ns_prefill_wall` are directly comparable.
    //
    // Caveat, important when reading the table: the CUDA forwarders launch on
    // the default stream and do NOT drain it.  For those slots the interval
    // measures *submission* (python-side dispatch cost plus the memcpys and any
    // implicit sync), not device execution; the GPU time they queued shows up
    // instead inside whichever later slot drains the stream.  Slots whose
    // forwarder does synchronize -- `argmax`, `copyToHost`, and the batchK /
    // matvec paths that call `downloadOut` -- DO include their GPU time.  The
    // per-slot line marks the draining slots; see `slotSyncs`.
    stat_ns_slot: [slot_count]u64 = @splat(0),
    stat_n_slot: [slot_count]u32 = @splat(0),
    /// Set while a timed slot is open, so `slotExit` can be paired.  Not
    /// re-entrant by design: the forwarders do not call each other except via
    /// explicit `CpuBackend.vtable.*` calls, which bypass the timers.
    slot_depth: u32 = 0,
    /// Wall clock spent inside timed slots, all slots summed (`QWEN35_PROF`
    /// only).  `pre_wall_ms - slot_total_ms` is what the engine itself spent
    /// outside any `Exec` call.
    stat_ns_slot_total: u64 = 0,

    // ---- per-slot DEVICE time ($QWEN35_PROF_GPU) ---------------------------
    //
    // WHY THIS EXISTS: the `QWEN35_PROF` slot table is HOST wall clock, and for
    // the 28 non-draining slots that is submission time only -- their queued GPU
    // work surfaces in whichever later slot drains the stream.  So the table can
    // say "matvec = 9 ms/token" without saying whether that is 187 kernel
    // submissions (the launch-bound regime, fixable with a graph) or 187 kernels
    // that really ran 9 ms on the device (the bandwidth-bound regime, fixable
    // only with a faster kernel).  A/B on `QWEN35_CUDA_GRAPH` settled the
    // question numerically (graph=1 survives CUDA_LAUNCH_BLOCKING at 122 tok/s
    // while graph=0 drops to 78), so the device number is what is left to
    // measure.
    //
    // HOW: one pair of `cudaEventCreateWithFlags(..., EventDisableTiming)` per
    // timed slot, recorded on the slot's launch stream at enter/exit.  Reading
    // an elapsed time REQUIRES the pair to be complete, and each pair is
    // recorded on the stream the launch goes to, so `b - a` is the device time
    // between the two records: the slot's own kernels plus any device work
    // already queued ahead of them.  That is exactly why the read happens once
    // per token, at the argmax drain, when the stream is empty -- a pair read in
    // the middle of a token would measure the queue, not the slot.
    //
    // WHY A POOL AND NOT PER-TOKEN CREATION: `cudaEventCreateWithFlags` in the
    // hot path costs as much as the launch being measured.  The pool is created
    // once, and a pair is reused as soon as its interval has been read.
    // HOW: one pair of `cudaEventCreateWithFlags(..., EventDisableTiming)` per
    // timed slot, recorded on the slot's launch stream at enter/exit.  Reading
    // an elapsed time REQUIRES the pair to be complete, and each pair is
    // recorded on the stream the launch goes to, so `b - a` is the device time
    // between the two records: the slot's own kernels plus any device work
    // already queued ahead of them.  That is exactly why the read happens once
    // per token, at the argmax drain, when the stream is empty -- a pair read in
    // the middle of a token would measure the queue, not the slot.
    //
    // WHY A POOL AND NOT PER-TOKEN CREATION: `cudaEventCreateWithFlags` in the
    // hot path costs as much as the launch being measured.  The pool is created
    // once, and a pair is reused as soon as its interval has been read.
    ev_pair_a: [gpu_ev_capacity]?*anyopaque = @splat(null),
    ev_pair_b: [gpu_ev_capacity]?*anyopaque = @splat(null),
    gpu_ev_ready: bool = false,
    /// Pool usage for the token currently in flight.
    gpu_ev_used: usize = 0,
    /// Slot id per pair, so a read can be attributed.
    gpu_ev_slot: [gpu_ev_capacity]u8 = @splat(0),
    /// Device ns per slot, summed over all tokens read.
    stat_gpu_ns_slot: [slot_count]u64 = @splat(0),
    stat_gpu_n_slot: [slot_count]u32 = @splat(0),
    /// Pairs that did not fit in the pool for this token (counted, not dropped).
    stat_gpu_overflow: usize = 0,
    /// TEMP(ablation): bitmap of `Slot`s whose launch is SKIPPED.  Driven by
    /// `QWEN35_ABLATE=<comma-separated slot names>`.  Model output is garbage
    /// when this is on; the ONLY thing read is the tok/s drop, which is how a
    /// slot's real device contribution is attributed when nsys is unavailable.
    ablate_bits: u32 = 0,
    /// `EventElapsedTime` failures while reading the pool.
    stat_gpu_ev_err: usize = 0,
    /// Whole-token device ns from the pool (sum of all pairs), for the
    /// reconciliation against `stat_ns_token_dev` (the token-end event pair).
    stat_gpu_ns_token_sum: u64 = 0,
    stat_gpu_n_token: usize = 0,
    /// Set when `$QWEN35_PROF_GPU` is in the environment.
    gpu_ev_on: bool = false,
    /// Pair index opened by the current `slotEnter`, consumed by its `slotExit`.
    gpu_ev_open: ?usize = null,

    /// `copyRange` broken down by *length* and *route* (`QWEN35_PROF` only).
    ///
    /// The slot total says `copy_range` is expensive but not *which* call site
    /// pays: the engine issues it from nine places at four very different
    /// widths (a 512-B per-head de-interleave, a `kv_dim` KV-cache write, a
    /// one-shot embedding H2D, and the per-row GDN copies).  Bucketing by
    /// element count makes each family its own row, because in this model the
    /// widths do not overlap: 128 = one head, 512/1024 = q/gate or the GDN
    /// `kv_mem` slices, 2048+ = kv_dim (27B: 8*256) and embedding rows.
    ///
    /// The route split matters separately from the width: a d2d move is a
    /// kernel launch, a host->device move is a `cudaMemcpy`, and they have
    /// different per-call costs for the same byte count.  Counting them
    /// together is what made the P2b-era table unreadable.
    stat_copy_bins: [copy_bin_count]CopyBin = @splat(.{}),

    // ---- reusable device buffers -------------------------------------------
    //
    // One set, grown on demand, never shrunk: matvec is the decode hot path and a
    // `cudaMalloc` there costs far more than the kernel.  `ensureBuf` is the only
    // allocation site reachable from matvec/matvecBatchK/dequantRow.
    /// f32 activations (one token, or k*cols for a batch).
    d_act: ?*anyopaque = null,
    d_act_bytes: usize = 0,
    /// Quantized activations (Q8_0 or Q8_K, whichever the weight type needs).
    d_xq: ?*anyopaque = null,
    d_xq_bytes: usize = 0,
    /// Second activation buffer holding the `block_q8_1_mmq` reshape, written by
    /// the same quantization pass.  Only allocated when a dp4a type is launched
    /// (see `q35_mmq_dp4a_wants`); the mma path never touches it.  Both buffers
    /// are needed because one prefill chunk mixes types -- e.g. iq3_xxs on dp4a
    /// next to iq3_s on mma -- and they share `d_xq` otherwise.
    d_xqm: ?*anyopaque = null,
    d_xqm_bytes: usize = 0,
    /// f32 matvec output (rows, or k*rows).
    d_out: ?*anyopaque = null,
    d_out_bytes: usize = 0,
    /// f32 row-dequantization target (cols floats).
    d_deq: ?*anyopaque = null,
    d_deq_bytes: usize = 0,
    /// Single `int` cell the argmax kernel writes its index into.  It has to be
    /// device memory (the launcher stores through `d_out`), and it is the only
    /// per-token download the greedy path makes: 4 bytes.
    d_idx: ?*anyopaque = null,
    d_idx_bytes: usize = 0,
    /// 3 f32-or-int cells `[index, best_logit_bits, exp_sum_bits]` for
    /// `q35_launch_argmax_conf`.  Separate from `d_idx` because that one is
    /// exactly 4 bytes and pre-allocated in `allocScratch`.
    d_conf: ?*anyopaque = null,
    d_conf_bytes: usize = 0,

    // ---- device sampler (src/kernels/cuda/sample.cu) ------------------------
    //
    // `d_sample_ctl` is a RESIDENT 64-byte control block holding every
    // per-token sampler input (temperature, top_k, top_p, PRNG state) plus the
    // result.  It exists so the sampler can live INSIDE the captured decode
    // graph: a captured node keeps the parameters it was recorded with, so all
    // of them have to arrive through memory instead of through `SetParams`.
    // `d_sample_scratch` is the kernel's working set (histogram, per-block
    // extrema, the f64 mass), sized once by `q35_sample_scratch_ints()`.
    //
    // Both are allocated lazily by `ensureSampleBufs` BEFORE any capture opens,
    // because `cudaMalloc` is illegal inside a capture interval.
    d_sample_ctl: ?*anyopaque = null,
    d_sample_scratch: ?*anyopaque = null,
    /// Host row for the `$QWEN35_CUDA_SAMPLE_CPU` A/B, grown on demand.
    h_sample_ab: ?[]f32 = null,
    /// Host scratch for the same A/B: the CPU sampler sorts through it.
    h_sample_ab_scratch: ?[]f32 = null,
    /// `$QWEN35_CUDA_SAMPLE_CPU`: route `sample` to the CPU reference on the
    /// device's own logits, so the two samplers can be compared without the
    /// matvec divergence in the way.  Debug only; default off.
    sample_cpu_ab: bool = false,
    /// The token `sample` produced on the current step, collected by
    /// `sampledToken` after `stepEnd` has done the download.  The split exists
    /// because the download cannot be captured but the sampler must be.
    sample_token: u32 = 0,
    /// The sampler whose PRNG state the pending download must advance.  A bare
    /// pointer on purpose: the engine owns the object and outlives every step
    /// (it is allocated with the engine's arena), so the backend must not copy
    /// or free it -- only write `state` through it, exactly as `dSample` would.

    // ---- decode CUDA graph ($QWEN35_CUDA_GRAPH) -----------------------------
    //
    // WHY THIS EXISTS: a decode token is ~700 kernel launches of which almost
    // none is dominated by its own execution time -- the 0.8B runs at ~8.2 ms
    // per token on this box, and a measured 240-launch probe moved 6.57 -> 6.12
    // ms/token purely by replacing the host's per-launch submission with one
    // graph replay.  The saving is a fixed per-kernel submit cost, so a token
    // with more launches gains proportionally more.
    //
    // WHY A GRAPH AND NOT A STREAM: every launch in this file passes `null`,
    // i.e. the LEGACY default stream, and CUDA forbids capturing it.  So the
    // graph is captured on a private non-default stream (`g_stream`) and every
    // `Launch.*` call inside the captured forward takes that stream instead of
    // `null` -- see `gSlice`-style stream substitution at the call sites.
    // Atomicity of a token is preserved: legacy-default-stream work is
    // implicitly ordered against all blocking streams, but during capture there
    // IS no other stream doing anything, and the replay is the only work
    // queued, so ordering between the graph's own nodes is what the captured
    // dependencies already encode.
    //
    // WHAT IS NOT IN THE GRAPH: `dArgmax_body`'s blocking D2H.  A synchronous
    // copy cannot be captured, and more fundamentally its result has to reach
    // the host before the next token's `pos`/`slot` are even known -- that is a
    // real structural dependency, not an implementation detail.  So the graph
    // spans exactly one `forwardLogits`, and argmax runs after it, directly.
    //
    // STATE MACHINE (`g_state`):
    //   off     -- feature disabled or already gave up; direct launches only.
    //   warm    -- feature on but the capture has not started.
    //   rec     -- inside `cudaStreamBeginCapture`..`EndCapture`; every launch
    //              must be redirectable to `g_stream`.
    //   live    -- `g_exec` holds a replayable graph.

    /// 三张图的槽。槽号由引擎经 `Graph.slot` 传进来（`xdPhaseBegin` 只看
    /// `g.slot`），后端没有"当前槽"这种全局状态：
    /// 0 = decode（1 行）、1 = verify（k 行）、2 = nextn（草稿块）。
    /// 形状不同就不能共用一槽：节点集合一变，每步都要重捕，等于没有图。
    ///
    /// **堆上分配，不内联。** `GraphSlot` 里还留着旧机制的一大堆
    /// `[graph_max_layers][kernel_max_args]` 参数数组，单个槽有上百 KB；三个
    /// 内联进 `CudaBackend` 会把它撑到接近线程栈上限（`CpuBackend` 是按值嵌在
    /// 引擎里的，引擎就活在主线程栈上）。实测：给槽加一个 4 字节字段就让
    /// `CudaBackend.init` 在 `--backend cpu` 下段错误，地址紧贴栈顶
    /// （0x7fff73e27e10）。`init` 里分配，失败时留空切片（`gs()` 的调用方都已
    /// 做了边界检查）。
    g_slots: []GraphSlot = &.{},
    /// `sample` 的结果。图活跃时由 `dSample` 留空（只有 kernel 进图），
    /// token 在 `graphEnd` 之后由 `dSampledToken` 读回时填入。
    sampled: u32 = 0,

    /// 本步的采样器。`collectSampledToken` 除了取 id 还要把 kernel 推进过的
    /// PRNG 状态写回这里，而 `graphEnd` 之后 engine 不再传它，所以在
    /// `dSample` 里记下指针（engine 的采样器活的比一步长）。
    sample_sp: ?*sampler.Sampler = null,

    /// 见 `gOpGate`：非 graph 管辖段。
    g_direct: bool = false,

    /// 设备端的一个 int：当前 KV 写入的起始行号。`q35_launch_kv_quant_dslot`
    /// 从它读行号，所以 graph 捕获下来的指针在重放时仍然有效（每 token 只需
    /// 更新这一个 int）。非 graph 路径不使用它。
    /// The device control block (see `ensureCtl`).  ONE per backend, shared by
    /// every graph: the blocks carry per-STEP values, so two graphs never need
    /// them at the same time.
    ctl_dev: ?*anyopaque = null,
    /// Host mirror of the device control block (`backend.GraphCtl`), written by
    /// `publishCtl` once per step.  Kept so a step that would publish the SAME
    /// values (e.g. `rope64` per head) does not launch the writer again.
    ctl_host: backend.GraphCtl = .{},
    /// True once `ctl_host` matches what is in the device block.
    ctl_valid: bool = false,
    /// 上一次 `sample` 的 logits 长度。`dGraphBegin` 在 XG 重放前要用它发布
    /// 采样参数，而那时引擎还没调用本步的 `sample`。
    sample_vocab: usize = 0,
    /// 草稿块录制期间记住这一趟的 f32 logits（`d_deq`）。`dSampledToken` 在图
    /// 关闭后用一次阻塞 D2H 把它读回主机、交给 CPU 采样器算 token。
    /// 用完即清；其余路径为 null。
    sample_logits: ?*anyopaque = null,


    d_kv_slot: ?*anyopaque = null,
    /// Host 镜像，用来判断是否需要重新 H2D（只有值变了才写）。
    d_kv_slot_host: c_int = -1,
    /// Non-default stream the graph was captured on and is replayed on.
    /// Created lazily (a CUDA build that never enables the feature pays
    /// nothing), never the legacy default stream -- capture of that one is
    /// rejected by the runtime with `cudaErrorStreamCaptureUnsupported`.
    g_stream: ?*anyopaque = null,
    /// Event used purely for ORDERING the graph against the legacy default
    /// stream (where the embedding lookup runs).  The graph's stream is
    /// non-blocking, so the two have no implicit ordering: without this the
    /// replay could start before the embedding wrote its input.
    g_ev: ?*anyopaque = null,
    /// Second ordering event, kept separate from `g_ev` on purpose: `g_ev`
    /// records the DEFAULT stream (to make the graph stream wait for the
    /// embedding) and `g_ev_out` records the GRAPH stream (to make the host wait
    /// for the replay).  Sharing one event between the two directions makes each
    /// `Record` clobber the other direction's pending wait inside one token;
    /// two events remove that coupling for the cost of one extra handle.
    g_ev_out: ?*anyopaque = null,
    /// Two device int32 {pos, bit pattern of base}: the per-token inputs of
    /// `k_rope64_dpos`.  Allocated on first use and then just written, so the
    /// 448 rope nodes of a 27B decode graph carry a constant pointer and never
    /// need `GraphExecKernelNodeSetParams`.  See `dRope64_body`.
    /// Host alias of `g_rope_dyn` (the mapped pinned page).  Writing here is how
    /// a token's `pos`/`base` reaches `k_rope64_dpos`.
    /// What is currently published in `g_rope_dyn`, so the writer kernel runs
    /// once per token instead of once per head.
    /// What `d_xq` currently holds (see `actQValid`).  `n == 0` means empty.
    q_hit: usize = 0,
    q_miss: usize = 0,
    /// QWEN35_GRTIME: pure graph-replay device time (no host wait inside).
    gr_ev: [2]?*anyopaque = .{ null, null },
    gr_ms: f64 = 0,
    gr_n: usize = 0,
    /// QWEN35_MVTIME: per-matvec device time accumulator (events, not wall).
    mv_ev: [2]?*anyopaque = .{ null, null },
    mv_ns: u64 = 0,
    mv_on: bool = false,
    act_q: struct {
        src: ?[*]const f32 = null,
        n: usize = 0,
        cols: usize = 0,
        fam: cpu_mod.ActFamily = .q8_0,
    } = .{},
    /// Device int holding the decode `slot` (the KV prefix length minus one).
    /// `attn_decode` reads it instead of taking `slot` as a launch argument, so
    /// the captured node's argument block AND its `shared_mem_bytes` are
    /// constants and the per-token/per-layer `GraphExecKernelNodeSetParams` --
    /// which cost 14.3 ms of the 27B's 33.8 ms decode -- disappears entirely.
    /// Allocated once, written once per token by `set_int1`.
    /// PV partials for the split attention path: `heads * split * hd` floats.
    d_att_part: ?*anyopaque = null,
    d_att_part_bytes: usize = 0,
    /// Per-head probability scratch for the split attention path.  The engine
    /// hands every head the SAME `slot + 1` floats (`self.attn`), because the
    /// reference kernel runs one block per head and each block re-derives the
    /// same scores -- but the split path runs the softmax as its OWN kernel, one
    /// block per head, so writing all heads into one shared row would race.
    /// Sized `heads * Q35_ATTN_MAX_ROWS` so the stride matches the kernel.
    d_att_prob: ?*anyopaque = null,
    d_att_prob_bytes: usize = 0,
    /// Rows-per-head reduction factor for the split attention path
    /// (`$QWEN35_CUDA_ATTN_SPLIT`, 0 = off -> coalesced kernel).
    attn_split: usize = 0,
    /// The captured graph, kept only between EndCapture and Instantiate.
    /// Replayable instantiation.  This is the only thing used per token.
    /// Kernel nodes to retarget every token, in the order they were launched.
    /// Both are variables of the *decode step*, not of the model:
    ///   - `rope64`'s `pos` advances once per token (one launch per layer);
    ///   - `attn_decode`'s `slot` advances once per token (one launch per full
    ///     attention layer).
    /// Nothing else in a decode token changes its arguments -- `conv_step`
    /// shifts its window IN the kernel (fixed pointers), the GDN state kernels
    /// read fixed pointers, and matvec's operands are all fixed regions of the
    /// KV cache / weights.  That was checked against every `Launch.*` call site
    /// in this file, not assumed.
    /// Argument blocks handed to `cudaGraphExecKernelNodeSetParams`.  Each is
    /// the *value* of one changed argument (`pos`, `slot`), so these are four
    /// bytes each and outlive the call only in the sense that the runtime
    /// copies them -- but they must stay addressable between the `SetParams`
    /// and it returning, which a plain field guarantees.
    /// `rope64`'s `base` (theta), the OTHER per-token argument of that kernel.
    /// It is not a constant of the model: the engine re-derives it each step
    /// (context-length dependent RoPE scaling), so a snapshot would rotate
    /// every replayed token with the capture token's theta.
    /// Per-node parameter definitions captured at capture time (grid/block/func
    /// plus a pointer to the argument array).  Rebuilt rather than mutated
    /// because `cudaGraphKernelNodeGetParams` may return either a
    /// `kernelParams` array or an `extra` blob and only the former is safe to
    /// hand back to `SetParams`.
    /// Owned copies of every tracked node's argument VALUES, one
    /// `[kernel_max_args]ArgSlot` per node.  `cudaGraphKernelNodeGetParams`
    /// hands back a `kernelParams` array that belongs to the runtime and that we
    /// must not write through; instead the values are copied here at capture
    /// time and the `kernelParams` pointer handed to `SetParams` points into
    /// this array.  `pos`/`slot` occupy one slot each and that slot's pointer is
    /// retargeted at `g_rope_args[i]`/`g_attn_args[i]`, so a later token only has
    /// to store a new value in that field.
    /// The pointers handed to `SetParams` as `kernelParams`: one per tracked
    /// node, each pointing at that node's `g_*_argv[i]` row above.
    /// How many `rope64` / `attn_decode` launches the engine issued while the
    /// capture was open.  Reconciled against the number of matched nodes before
    /// the graph is trusted: a shortfall means a launch produced no node we can
    /// retarget, so replay would feed it a stale `pos`/`slot` -- wrong numbers,
    /// no error.  See `graphBuild`.
    /// `pos`/`slot` the capture recorded, used to prove the node identification
    /// actually found the right nodes before trusting the graph (see
    /// `graphSelfCheck`).
    /// The rope64 `base`/theta the graph was CAPTURED with.  It is a model
    /// constant (`rope_freq_base`), so the captured value stays right for every
    /// replay -- recorded rather than assumed so a future RoPE regime change is
    /// a visible mismatch instead of silent garbage.
    /// KV-cache writes captured from `copyRange`.  Their DESTINATION pointer is
    /// `cache.{k,v} + slot * kv_dim`, i.e. it advances by exactly one row per
    /// token, so a plain snapshot would append every later token's k/v to the
    /// capture token's slot -- each token then attends to a cache that never
    /// grew, which is wrong numbers with no error at all.  `g_copy_base[i]` is
    /// the address the capture used and `g_copy_stride[i]` the per-token byte
    /// step, so replay can recompute it.
    /// Destination of each KV append the capture issued, in order, plus its
    /// byte stride per token (`kv_dim * sizeof(f32)`).  `g_kv_expect` is the
    /// count the engine declared; it has to equal the number of nodes found, or
    /// the graph is dropped (see `graphBuild`).
    /// `g_order_nodes[0..g_order_n]` is the captured decode step's nodes in
    /// EXECUTION order, recovered by `graphCaptureOrder` -- see the note there
    /// for why the list `cudaGraphGetNodes` returns cannot be used as-is.

    /// Append counter of the layer cluster currently being walked by
    /// `graphBuild`: 0 before the cluster's first append, 1 after it.  See
    /// `kvSlotFromCluster`.

    /// For every KV append captured, the slot that the OTHER append of the same
    /// layer group belongs to: during a capture the engine issues a group's two
    /// appends consecutively, so the second one of a cluster is the partner of
    /// the first (see `kvSlotFromCluster`).
    /// Per-node cell holding the destination POINTER for `SetParams`.  A token
    /// advances it by `g_kv_stride[i]` bytes, so this is the one argument that
    /// `copyArgs` must leave pointing at our storage rather than copying.
    /// True while a capture is being assembled for the token whose argument
    /// values are (`g_rec_pos`, `g_rec_slot`); argmax's blocking copy and any
    /// host fallback clear it so a doomed capture is abandoned instead of
    /// instantiated.
    /// True between `stepBegin` opening a capture and `stepEnd` closing it.
    /// Distinct from `g_capturing`, which a mid-step abandonment clears: this
    /// one records *ownership* of the open capture so `stepEnd` knows it has to
    /// close something (or clean up after an abandonment) rather than being
    /// called on an uncaptured step.
    /// WHICH ops the captured graph actually contains.  Set during capture by
    /// `gOpGate`, consumed during replay to suppress the host-side launch.
    ///
    /// This is not an optimisation -- it is what makes replay correct.  A
    /// capture does NOT stop the engine: `forwardLogits` keeps calling every
    /// op's body, and a body that launches unconditionally emits a second copy
    /// of the same work onto the default stream while the graph replays the
    /// first copy on `g_stream`.  The two then interleave on the same buffers
    /// and the token's result is neither copy: measured 2026-11, 187 host
    /// matvec launches per replayed token, and `gdn.k` already holding the
    /// graph's output when the host path read it.  Recursive state (GDN) is
    /// advanced TWICE per token that way, which is why the first replayed token
    /// was already numerically wrong.
    ///
    /// Recording the flags from an actual capture (rather than guessing a list)
    /// is deliberate: an op that never entered the graph must keep running on
    /// the host, and one that did must not.
    /// Counters, printed once when the graph first goes live and again at
    /// deinit, so a report can state whether replay actually happened.
    stat_g_replay: usize = 0,
    stat_g_abandon: usize = 0,
    /// Set when `$QWEN35_CUDA_GRAPH` said "off" or the first capture failed;
    /// makes the decision one-way so a per-token retry cannot thrash.
    // No gamma/beta staging buffer any more: `gdnGamma` writes both INTO
    // backend regions (`gamma_scratch`, and the b half of `ab` in place) and
    // `gdnStepRows` consumes them straight from those device pointers.  The
    // scratch that used to live here existed only to re-upload host stack
    // arrays, which cost 2 * heads floats of H2D per layer per token plus a
    // matching D2H to produce them.  Those two regions must never be treated
    // as read-only weights either: their contents change every layer every
    // token, so an upload-once pointer-keyed copy would serve stale data.
    /// Every region `dAllocBuf` handed to the engine (activations, KV cache,
    /// GDN state).  The `Buf` values themselves are the engine's; this list is
    /// only so `freeScratch` / `deinit` can release them, since a `Buf` carries
    /// no ownership flag.
    /// Allocations from `dAllocBuf`, as (base, element count).  The count is not
    /// decoration: `isDevBuf` has to answer for Bufs the engine built with
    /// `span()`, whose pointer is base + offset and therefore never equals the
    /// base.  Comparing pointers alone made every strided device view look like
    /// host memory, which silently sent device-to-device copies through the CPU
    /// host bounce (correct answers, ~2.5 s of needless PCIe on a 355-token
    /// prefill).
    dev_bufs: std.ArrayList(DevRegion) = .empty,
    // ==== construction ======================================================

    /// The engine's factory contract (`engine.backendImplFor` calls
    /// `BackendImpl.init(alloc, g)`).  Returns a value rather than an error on
    /// purpose: `init` has no way to report one, so a host without a usable GPU
    /// gets a backend whose `allocScratch` fails with a named error -- that is
    /// where `Engine.load` can turn it into a load failure.
    pub fn init(alloc: std.mem.Allocator, g: *const gguf.File) CudaBackend {
        var self = CudaBackend{
            .alloc = alloc,
            .cpu = cpu_mod.CpuBackend.init(alloc, g),
            .g = g,
        };
        // 槽在堆上（见 `g_slots` 的注释）。分配失败留空切片：`gs()` 的调用方
        // 都会先查边界，此时图一律直通，模型照常跑。
        self.g_slots = alloc.alloc(GraphSlot, graph_slot_count) catch &.{};
        for (self.g_slots) |*s| s.* = .{};

        // Ask cudart for the device before touching weights, so a host with no GPU
        // fails here (and `allocScratch` reports it) rather than after a partial
        // upload.
        var cand: cudart.Open = undefined;
        cudart.open(&cand, 3 << 30) catch |e| {
            self.init_error = std.fmt.bufPrint(
                &msg_buf,
                "cuda backend: device unavailable ({s}); use --backend cpu{s}{s}",
                .{
                    @errorName(e),
                    if (cand.missing().len > 0) "; missing cudart symbols: " else "",
                    cand.missing(),
                },
            ) catch "cuda backend: device unavailable";
            return self;
        };
        self.cu = cand;
        self.sync_debug = cudart.getenv("QWEN35_SYNC") != null;
        self.ablateInit();

        // The launchers themselves need no setup: they are link-time symbols, so
        // "the kernels are there" is enforced by the build.  The one run-time
        // precondition left is the IQ codebook upload, which `allocScratch`
        // performs (see `uploadIqTables`) because it is the first hook the engine
        // calls after this.
        return self;
    }

    pub fn exec(self: *CudaBackend) backend.Exec {
        return .{ .ptr = self, .vtable = &vtable };
    }

    /// One-shot *complete* profile report: `CUDA-M [...]` + `CUDA-M  types:`
    /// + `CUDA-M  time:` + the `PROF-SLOT` and `PROF-COPY` tables.
    ///
    /// The reason this entry exists instead of relying on the existing
    /// `reportStats` call sites: those fire at internal trigger points
    /// (`% 500` matvecs, the first matvec, the closing of the prefill window)
    /// and in `deinit`, but the engine is arena-allocated so `deinit` never
    /// runs on the CLI path.  A long decode that stops before the next `% 500`
    /// boundary -- or any BF16/f32 run, whose `matvec` branch never calls
    /// `reportStats` at all -- therefore ended with a single header line and no
    /// tables.  This is called once, by `Engine.profBackendReport`, right after
    /// `generate` returns (the CLI gates it on `$QWEN35_PROF`).
    ///
    /// Unconditional by design: the *caller* owns the debug switch, so the
    /// semantics here are "print the whole thing now" and nothing else.  Purely
    /// diagnostic -- it reads counters and prints, touching no device state.
    pub fn profReport(self: *CudaBackend) void {
        self.reportStats("final");
    }

    pub fn deinit(self: *CudaBackend) void {
        self.reportStats("deinit");
        // The graph's stream is a non-default stream and every object it owns
        // (exec, graph, stream) must die before the context is released below.
        self.graphDestroy();
        if (self.cu) |*cu| {
            // Events first: they belong to the context that is about to be
            // released, and each one was created exactly once in
            // `ensureEvents`, so this is the matching destroy.
            self.freeEvents();
            for (self.dev_rows.items) |e| _ = cu.lib.Free(e.dev);
            if (self.d_act) |p| _ = cu.lib.Free(p);
            if (self.d_xq) |p| _ = cu.lib.Free(p);
            if (self.d_out) |p| _ = cu.lib.Free(p);
            if (self.d_deq) |p| _ = cu.lib.Free(p);
            for (self.dev_bufs.items) |r| _ = cu.lib.Free(r.ptr);
            self.dev_bufs.clearRetainingCapacity();
            self.d_act = null;
            self.d_xq = null;
            self.d_out = null;
            self.d_deq = null;
            // The codebook buffer is process-wide (it lives in the linked-in
            // iq_tables.cu, not in this struct), so release it here and reset the
            // flag: that re-arms the one-shot upload for any backend created
            // later in this process.
            if (self.iq_tables_state == .done) {
                Launch.deinit_tables();
                self.iq_tables_state = .pending;
            }
            cu.lib.close();
            self.cu = null;
        }
        self.dev_rows.deinit(self.alloc);
        self.dev_bufs.deinit(self.alloc);
        self.cpu.deinit();
    }

    // ==== device plumbing ===================================================

    fn self_(ctx: *anyopaque) *CudaBackend {
        return @ptrCast(@alignCast(ctx));
    }

    /// 每步的控制块（计划 §1.5 的 `cell`）。**一块**设备缓冲装着捕获下来的
    /// kernel 在执行时需要的每一个标量：
    ///
    ///   [XD_CTL_POS]        rope 的相位基准
    ///   [XD_CTL_BASE_BITS]  rope theta 的位模式
    ///   [XD_CTL_KV_SLOT]    这一步写入的第一行 KV
    ///   [XD_CTL_ATTN_SLOT]  full attention 必须覆盖到的最后一行 KV
    ///   [XD_CTL_ROWS]       这一步产生多少行
    ///
    /// 惰性分配；只有分配失败时才为 null，那时每个消费方退回自己那套
    /// 逐标量（capture 不安全）的路径。
    fn ensureCtl(self: *CudaBackend) ?*anyopaque {
        if (self.ctl_dev) |p| return p;
        const cu = &(self.cu orelse return null);
        // 只有设备块。主机侧没有暂存块，也不做 pinned 分配：发布走一次阻塞
        // `Memcpy`（见 `publishCtl`），源就是调用栈上的一个局部结构体。
        //
        // 为什么不需要 pinned：pinned 只对 `MemcpyAsync` 有意义（运行时不保证
        // pageable 内存的异步性，会退化成内部暂存 + 隐式同步）。这里这次 H2D
        // 的下游就是紧随其后的图重放，没有任何能与它重叠的工作，异步省不下
        // 时间，却要多一块常驻的 page-locked 内存和它的清理。
        //
        // 也不用 mapped 内存：GPU 读映射主机内存可能命中 GPU 侧陈旧缓存，而
        // `pos`/`kv_slot` 必须每步读到当步的值。
        //
        // 发布必须发生在 capture 打开**之前**：图里的 kernel 只读这块设备内存，
        // 不再有任何搬运节点。历史教训：早先用 `k_xd_ctl_publish` 发布，它在
        // capture 期被录成节点，重放时把录制那一步的 pos 写回（实测每步都退回
        // 17）；后来换成 capture 内的 `MemcpyAsync`，同样被录成 memcpy 节点。
        var p: ?*anyopaque = null;
        // cudaMalloc inside a capture is illegal, so a capture must not get here.
        // Every caller runs at `stepBegin`/`graphBegin`, i.e. before any capture.
        const rc = cu.lib.Malloc(&p, @sizeOf(backend.GraphCtl));
        if (rc != cudart.cudaSuccess or p == null) {
            self.ewLaunchFail("ctl alloc", rc);
            return null;
        }
        self.ctl_dev = p;
        self.ctl_valid = false;
        return p;
    }

    /// 把**这一步**的控制块发布出去，并记住它。
    ///
    /// 对同一组值幂等：`rope64` 每个 head 调一次，在 27B 上会把同样的
    /// `{pos, base}` 发布 448 次，所以主机镜像会把第一次之外的都短路掉。
    /// `force` 绕过这个短路（新的一步必须重新发布，即使值碰巧重复 —— 图里
    /// 其它节点可能在 `rope64` 被调用之前就读它们）。
    fn publishCtl(self: *CudaBackend, ctl: backend.GraphCtl, stream: ?*anyopaque, force: bool) bool {
        _ = stream;
        if (!force and self.ctl_valid and std.meta.eql(self.ctl_host, ctl)) return true;
        const p = self.ensureCtl() orelse return false;
        const cu = &(self.cu orelse return false);
        // 一次阻塞 H2D，源是调用栈上的局部结构体。
        //
        // 为什么是阻塞而不是 `MemcpyAsync`：这次拷贝的下游就是紧接着的
        // `GraphLaunch`（图里的 kernel 立刻要读这块设备内存），中间没有任何
        // 能与它重叠的工作。异步既省不下时间，又强制源内存必须 page-locked
        // —— 那是 `cudaMemcpyAsync` 的 API 要求，不是 GPU 在跑这段拷贝。
        //
        // 为什么这里不发 kernel：一个 `k_xd_ctl_publish` 节点在 capture 期会被
        // 录进图，重放时把录制那一步的值写回设备（实测 `pos` 每步都退回 17）。
        const rc = cu.lib.Memcpy(p, @ptrCast(&ctl), @sizeOf(backend.GraphCtl), cudart.MemcpyHostToDevice);
        if (rc != cudart.cudaSuccess) {
            self.ewLaunchFail("ctl publish", rc);
            return false;
        }
        self.stat_h2d_bytes += @sizeOf(backend.GraphCtl);
        self.ctl_host = ctl;
        self.ctl_valid = true;
        return true;
    }

    /// Invalidate the host mirror so the next `publishCtl` writes unconditionally.
    fn ctlInvalidate(self: *CudaBackend) void {
        self.ctl_valid = false;
    }

    fn gpuReady(self: *CudaBackend) bool {
        return self.cu != null;
    }

    fn needHelp(self: *CudaBackend) []const u8 {
        if (!self.gpuReady()) return self.init_error;
        return "cuda backend: kernel launch failed";
    }

    /// Grow `buf` to at least `need` bytes.
    ///
    /// Rounding up to 1 MiB keeps a varying prefill width (1..k tokens) from
    /// reallocating every call, and the buffer is never shrunk because matvec is on
    /// the decode hot path.  This is the **only** `cudaMalloc` reachable from
    /// `matvec` / `matvecBatchK` / `dequantRow`; everything else in those slots is a
    /// launch or a `cudaMemcpy`.
    fn ensureBuf(cu: *cudart.Lib, buf: *?*anyopaque, cur: *usize, need: usize) !void {
        if (buf.* != null and cur.* >= need) return;
        const want = mem.alignForward(usize, need, 1 << 20);
        if (buf.*) |p| _ = cu.Free(p);
        var np: ?*anyopaque = null;
        const rc = cu.Malloc(&np, want);
        if (rc != cudart.cudaSuccess) {
            buf.* = null;
            cur.* = 0;
            return error.OutOfMemory;
        }
        buf.* = np;
        cur.* = want;
    }

    fn findDev(self: *CudaBackend, host: [*]const u8) ?*DevRow {
        for (self.dev_rows.items) |*e| {
            if (e.host == host) return e;
        }
        return null;
    }

    /// Free device memory for the allocation-failure message.  0 when the query
    /// itself fails, in which case the caller leaves it out.
    fn freeDevMiB(cu: *cudart.Lib) usize {
        var freeb: usize = 0;
        var total: usize = 0;
        if (cu.MemGetInfo(&freeb, &total) != cudart.cudaSuccess) return 0;
        return freeb >> 20;
    }

    /// `$QWEN35_CUDA_MAX_DEV_MIB`: optional upload cap, so a model that will not fit
    /// fails during load with a budget message instead of a driver error deep into
    /// the layer stack.  Unset or unparsable = no cap.
    fn budgetExceeded(self: *CudaBackend) bool {
        const v = cudart.getenv("QWEN35_CUDA_MAX_DEV_MIB") orelse return false;
        const mib = std.fmt.parseInt(usize, mem.trim(u8, v, " "), 10) catch return false;
        return self.stat_uploaded_bytes >= (mib << 20);
    }

    /// `$QWEN35_CUDA_ROPE=host|device`: which implementation `dRope64` uses.
    /// Default `device` (zero-copy `q35_launch_rope64`); `host` is the escape
    /// hatch back to the CPU body, which is bit-exact against the reference but
    /// pays one D2H+H2D pair per head per layer per token.
    ///
    /// Measured on Qwen3.8-27B IQ3_XXS / 4090, 354-token prompt, 3 runs each
    /// (`scripts/bench.sh` style A/B, `$QWEN35_CUDA_ROPE` the only variable):
    /// prefill 10.20-10.54 -> 11.35-11.41 tok/s (+10.6%), generate 7.15-7.35 ->
    /// 7.48-7.50 tok/s (+4.4%).  The gain is smaller on generate because the
    /// per-head staging is amortised differently there, and it is NOT a scaling
    /// claim: as the matvec kernels improve, rope's share grows and so will this.
    ///
    /// This does NOT preserve host/device token identity, and is not meant to.
    /// The device libm's `powf` differs from glibc's by ~1 ULP, which is an
    /// absolute phase error of order `pos * 1.2e-7` rad; 65 layers of q/k
    /// rotation amplify it past the argmax margin well before pos 1000 (measured:
    /// first divergence at generated token 34 with a 17-token prompt, pos ~50;
    /// from token 1 with a 354-token prompt).  Each path is still deterministic
    /// on its own (3 runs, identical md5), so A/B debugging stays possible --
    /// what is lost is cross-path comparability, the same thing llama.cpp's own
    /// CPU-vs-CUDA backends do not offer either.
    ///
    /// Read once and cached: this is on the per-token path, and `getenv` per
    /// head per layer per token is exactly the kind of syscall this backend
    /// exists to avoid.  Any other value (including unset) keeps the default.
    fn ropeOnDevice() bool {
        const S = struct {
            var cached: ?bool = null;
        };
        if (S.cached) |v| return v;
        var on = true;
        if (cudart.getenv("QWEN35_CUDA_ROPE")) |raw| {
            const v = mem.trim(u8, raw, " \t\r\n");
            if (mem.eql(u8, v, "host")) {
                on = false;
            } else if (!mem.eql(u8, v, "device")) {
                std.debug.print(
                    "cuda backend: QWEN35_CUDA_ROPE='{s}' is neither 'host' nor 'device'; using device\n",
                    .{v},
                );
            }
            std.debug.print("cuda backend: rope64 = {s} ({s})\n", .{
                @tagName(if (on) RopePath.device else RopePath.host), "QWEN35_CUDA_ROPE",
            });
        }
        S.cached = on;
        return on;
    }

    /// `QWEN35_GDN_WY=0` forces the per-token GDN kernel (the reference the WY
    /// form was validated against); anything else, including unset, uses WY.
    /// Cached because it is read per GDN layer, ~50 times per prefill.
    fn gdnWyEnabled() bool {
        const S = struct {
            var cached: ?bool = null;
        };
        if (S.cached) |v| return v;
        var on = true;
        if (cudart.getenv("QWEN35_GDN_WY")) |raw| {
            const v = mem.trim(u8, raw, " \t\r\n");
            if (v.len == 1 and v[0] == '0') on = false;
        }
        S.cached = on;
        return on;
    }

    // ==== decode CUDA graph ==================================================
    //
    // Read `g_state`'s comment for the design; this section is the mechanics.
    //
    // THREE INVARIANTS the code below exists to uphold, each one a silent
    // wrong-numbers bug if broken:
    //
    //  1. NOTHING inside the graph may allocate.  `cudaMalloc` during capture is
    //     legal in relax mode but produces a graph whose replay reuses the same
    //     address -- correct only if the allocation never moves.  `ensureBuf`
    //     *does* move (it frees and reallocates when the size grows), so a
    //     matvec that grew `d_out` after capture would leave every subsequent
    //     node writing through a dangling pointer.  `graphReserve` therefore
    //     forces every scratch buffer this backend can grow up to its
    //     prefill-sized maximum BEFORE the capture opens, and
    //     `graphed-ensureBuf` asserts rather than growing during `rec`.
    //
    //  2. NOTHING inside the graph may be synchronous.  `cudaMemcpy` (the
    //     blocking flavour, used by `dCopyToHost`/`dCopyFromHost` and by the
    //     `rope64` host path) invalidates a capture outright.  That is why
    //     capture forces the device rope path and why `dArgmax_body` is outside
    //     the graph's span by construction.
    //
    //  3. The graph must contain ALL of one `forwardLogits` and nothing else.
    //     Node identification below is by argument value, not by position,
    //     because `cudaGraphGetNodes`'s order is unspecified.

    /// Whether `$QWEN35_CUDA_GRAPH` allows the feature.  Unset means ON: the
    /// measurement this was written from shows a monotone win and the fallback
    /// is automatic, so requiring an opt-in would just hide it.  `=0` (or any
    /// of false/no/off) turns it fully off -- that is the A/B switch and the
    /// escape hatch, and with it set nothing below ever runs.
    var g_graph_cached: ?bool = null;
    var g_xd_cached: ?bool = null;

    /// 槽 `i`。**没有"当前槽"这种全局状态**：调用方一律把 `Graph.slot` 传进
    /// 来（见 `backend.Graph.slot` 的说明 —— 嵌套 span 下全局当前槽会让外层
    /// `graphEnd` 收尾到错误的槽上）。
    inline fn gs(self: anytype, slot: u8) if (@TypeOf(self) == *CudaBackend) *GraphSlot else *const GraphSlot {
        std.debug.assert(slot < self.g_slots.len);
        return &self.g_slots[slot];
    }

    /// XG（显式计算图）的开关，**默认开启**。`$QWEN35_XD=0` 退回旧机制
    /// （`GraphSlot` 的 `rec`/`live` + `GraphExecKernelNodeSetParams`），
    /// 那是 A/B 与排障用的对照路径，S6 会删掉。
    ///
    /// 默认值是怎么定的（2026-11 实测，27B IQ3_XXS / 0.8B Q8_0，64 token）：
    /// 图内 1573 个节点、**0 个非 kernel 节点**，token 流与旧机制逐位一致，
    /// 27B decode 51.7 → 54.2 tok/s（+4.8%）。旧机制每 token 要按节点重建
    /// 实参，XG 一次 `GraphLaunch` 即可，省下的是纯主机侧开销。
    fn xdWanted() bool {
        if (g_xd_cached) |v| return v;
        var on = true;
        if (cudart.getenv("QWEN35_XD")) |raw| {
            const v = mem.trim(u8, raw, " \t\r\n");
            if (v.len != 0) {
                on = !(mem.eql(u8, v, "0") or mem.eql(u8, v, "false") or
                    mem.eql(u8, v, "no") or mem.eql(u8, v, "off"));
                std.debug.print("cuda backend: explicit compute graph = {s} ($QWEN35_XD)\n", .{
                    if (on) "on" else "off",
                });
            }
        }
        g_xd_cached = on;
        return on;
    }

    fn graphWanted() bool {
        if (g_graph_cached) |v| return v;
        var on = true;
        if (cudart.getenv("QWEN35_CUDA_GRAPH")) |raw| {
            const v = mem.trim(u8, raw, " \t\r\n");
            if (v.len != 0) {
                on = !(mem.eql(u8, v, "0") or mem.eql(u8, v, "false") or
                    mem.eql(u8, v, "no") or mem.eql(u8, v, "off"));
                std.debug.print("cuda backend: decode CUDA graph = {s} ($QWEN35_CUDA_GRAPH)\n", .{
                    if (on) "on" else "off",
                });
            }
        }
        g_graph_cached = on;
        return on;
    }

    /// The stream every kernel launch in the captured forward must use.
    ///
    /// `null` (the legacy default stream) everywhere EXCEPT while a capture is
    /// open, when it must be the private capture stream.  Centralising it here
    /// rather than at ~28 call sites means there is exactly one place that can
    /// forget, and the `g_capturing`/`g_state` pair makes forgetting a loud
    /// capture error (`cudaErrorStreamCaptureUnsupported`) instead of a graph
    /// that is missing nodes.
    /// The stream kernel launches should use right now (`null` = the default
    /// stream).  Public so a caller that owns its own launches -- the DSpine
    /// draft's code bridge (`src/draft.zig`) -- can put them on the same stream
    /// as the backend's matvecs and get the ordering for free.
    pub fn currentStream(self: *CudaBackend) ?*anyopaque {
        return self.launchStream();
    }

    /// Copy `n` f32 from device region `src` into HOST slice `dst`.
    ///
    /// `Exec.copyRange` is a device-to-device move on this backend (and a
    /// memcpy of host memory for a host->device or host-only pair, which does
    /// not help a caller that owns a raw `[]f32`).  A caller holding host memory
    /// -- the DSpine draft's serial stash -- needs this instead; passing a host
    /// pointer to `copyRange` would make `copy_range` write THROUGH a host
    /// address and abort the context with an illegal access (measured).
    pub fn copyToHost(self: *CudaBackend, dst: []f32, src: backend.Buf) void {
        if (dst.len < src.len or self.cu == null) return;
        const cu = &self.cu.?;
        _ = cu.lib.Memcpy(dst.ptr, @ptrCast(src.ptr), src.len * @sizeOf(f32), cudart.MemcpyDeviceToHost);
    }

    /// Copy arbitrary host bytes into a device buffer.  Needed for the small
    /// non-f32 control cells a caller owns (the DSpine bridge's anchor-id
    /// table); `copyFromHost` is the typed form of the same thing.
    pub fn copyBytesToDevice(self: *CudaBackend, dst: *anyopaque, src: []const u8) void {
        if (self.cu == null) return;
        const cu = &self.cu.?;
        _ = cu.lib.Memcpy(dst, src.ptr, src.len, cudart.MemcpyHostToDevice);
    }

    /// Copy HOST slice `src` into device region `dst`.  The mirror of
    /// `copyToHost`; see its comment.
    pub fn copyFromHost(self: *CudaBackend, dst: backend.Buf, src: []const f32) void {
        if (src.len < dst.len or self.cu == null) return;
        const cu = &self.cu.?;
        _ = cu.lib.Memcpy(@ptrCast(dst.ptr), src.ptr, dst.len * @sizeOf(f32), cudart.MemcpyHostToDevice);
    }

    /// Block until every launch on `currentStream()` has completed.  Public for
    /// the same reason as `currentStream`: the DSpine draft issues its own
    /// launches and then needs the results on the host.
    pub fn syncCurrentStream(self: *CudaBackend) void {
        if (self.cu) |*cu| {
            _ = cu.lib.StreamSynchronize(self.launchStream());
        }
    }

    fn launchStream(self: *CudaBackend) ?*anyopaque {
        // XG 的录制也必须在 `g_stream` 上，否则 `xd_capture` 看到的流不在
        // capture 状态，录制直接失败（实测：rc=1 在第一个 op）。
        //
        // 这里保持"任何槽在录就用图流"：capture 只有一条流，且同一时刻只
        // 允许一个槽进 `.rec`。真正需要区分"哪张图"的判断在 `capturing(g)`。
        for (self.g_slots) |s| {
            if (s.xd_phase == .rec) return self.g_stream;
        }
        return switch (self.gs(0).state) {
            .rec => self.g_stream,
            else => null,
        };
    }

    /// True when a launch happening right now is being captured, i.e. when its
    /// arguments must be recorded for the per-token retarget.
    /// 这张图此刻正在被录吗。
    ///
    /// 必须按图问：不同槽的 capture 状态不同，问"后端有没有在录"会把 A 图
    /// 的录制误当成 B 图的（草稿块就嵌在 verify 的调用栈里面）。
    fn capturing(self: *const CudaBackend, g: backend.Graph) bool {
        if (g.slot >= self.g_slots.len) return false;
        return self.g_slots[g.slot].xd_capturing;
    }

    /// 后端级问题：**有没有任何**槽在录。只给拿不到图的统计路径用。
    fn anySlotCapturing(self: *const CudaBackend) bool {
        for (self.g_slots) |s| {
            if (s.xd_capturing) return true;
        }
        return false;
    }

    /// True when this step's work is (or will be) inside a graph, i.e. the op
    /// must take the device-control-block form so a replayed node reads fresh
    /// per-token scalars.  Covers BOTH mechanisms: the old `state` and XG's
    /// `xd_phase`.  Getting this wrong is silent: a node captured with a
    /// frozen `pos`/`slot` re-runs the first token's value every replay.
    /// 有没有任何一个槽是 `.live`（诊断路径用；decode 只看槽 0）。
    fn anySlotLive(self: *const CudaBackend) bool {
        for (self.g_slots) |s| {
            if (s.state == .live) return true;
        }
        return false;
    }

    /// 这张图是（或即将是）`rec`/`live` 吗 —— 即它的 op 必须走设备控制块
    /// 形态，好让重放的节点读到当步的标量。同样按图问，理由同 `capturing`。
    fn graphActive(self: *const CudaBackend, g: backend.Graph) bool {
        if (g.slot >= self.g_slots.len) return false;
        const ph = self.g_slots[g.slot].xd_phase;
        return ph == .rec or ph == .live;
    }

    /// Suppress a host-side launch when the captured graph already contains it.
    ///
    /// Returns false while a capture is open (after recording that this op made
    /// it into the graph) and false while no graph is live; returns true only
    /// for a REPLAYED step, whose work the graph has already done on `g_stream`.
    /// Without this the host path re-runs the whole step on the default stream
    /// next to the graph -- see `g_op_seen`.
    fn gOpGate(self: *CudaBackend, g: backend.Graph, seen: *bool) bool {
        // `g_direct`：当前这段代码不在任何 graph 的管辖范围内（MTP 的 draft
        // 循环就是这样跑的）。它必须真正执行，而不是被当成"已由 graph 重放
        // 吸收"而跳过 —— 否则 draft 什么都不算，verify 拿到垃圾。
        if (self.g_direct) return false;
        switch (self.gs(g.slot).state) {
            .rec => {
                seen.* = true;
                return false;
            },
            .live => return seen.*,
            else => return false,
        }
    }

    /// Called from every place that would otherwise make a capture invalid:
    /// a blocking copy, a host fallback, a launch error.  Abandons the capture
    /// for this token so the caller falls back to direct launches.  `why` is
    /// printed once (not per token) -- a per-token flood would bury the real
    /// message and cost measurable time in the very path being measured.
    /// 后端级：放弃**任何**正在录的槽。只给拿不到具体图的路径用
    /// （`dGraphDirect` 进入直发段时）。
    fn graphAbandonAny(self: *CudaBackend, why: []const u8) void {
        var i: u8 = 0;
        while (i < self.g_slots.len) : (i += 1) {
            if (!self.g_slots[i].xd_capturing) continue;
            self.graphAbandon(.{ .slot = i }, why);
            return;
        }
    }

    /// 放弃**这张图**的 capture。
    ///
    /// 按图而不是"扫描所有槽"：嵌套 span 下（草稿块在 verify 里面）扫描会
    /// 关掉外层的 capture 而把内层的留着，结果两个槽的状态都不对。
    fn graphAbandon(self: *CudaBackend, g: backend.Graph, why: []const u8) void {
        if (g.slot >= self.g_slots.len) return;
        if (!self.g_slots[g.slot].xd_capturing) return;
        self.stat_g_abandon += 1;
        if (self.stat_g_abandon == 1) {
            std.debug.print(
                "cuda backend: decode graph abandoned this token ({s}); falling back to direct launches\n",
                .{why},
            );
        }
        // ORDER MATTERS: the capture has to be CLOSED first, because until
        // `cudaStreamEndCapture` returns the stream is still "capturing" and any
        // subsequent launch on it fails with `cudaErrorStreamCaptureUnsupported`.
        self.g_slots[g.slot].xd_capturing = false;
        if (self.g_slots[g.slot].xd_phase == .rec) self.g_slots[g.slot].xd_phase = .off;
        if (self.cu) |*cu| {
            var cap: ?*anyopaque = null;
            _ = cu.lib.StreamEndCapture(self.g_stream, &cap);
            // THE ABANDONED CAPTURE STILL HAS TO RUN.  Capture mode records
            // launches instead of executing them, so this token has had no kernel
            // at all: no KV row appended, no GDN state advanced.  Instantiate and
            // launch once, which replays exactly the kernels the direct path would
            // have issued, so the step is performed and the numbers stay right
            // even though no graph is kept.
            if (cap) |gg| {
                var ab_exec: ?*anyopaque = null;
                if (cu.lib.GraphInstantiate(&ab_exec, gg, 0) == cudart.cudaSuccess and ab_exec != null) {
                    if (cu.lib.GraphLaunch(ab_exec.?, self.g_stream) != cudart.cudaSuccess) {
                        _ = cu.lib.GetLastError();
                    }
                    if (self.g_stream) |st| _ = cu.lib.StreamSynchronize(st);
                    _ = cu.lib.GraphExecDestroy(ab_exec.?);
                }
                _ = cu.lib.GraphDestroy(gg);
            }
        }
    }

    /// Bring every growable scratch buffer up to the size a *prefill* call would
    /// need, so no `ensureBuf` can fire inside the capture (invariant 1).
    ///
    /// WHY THE PREFILL SIZE: the decode step's needs are a strict subset of a
    /// prefill chunk's, and prefill has already run `allocScratchK` by the time
    /// generation starts.  Re-reserving with the same numbers here is not
    /// redundant -- it makes the requirement local and checkable instead of
    /// "the engine happens to have called allocScratchK first", and it also
    /// covers `d_idx`, which nothing pre-grows today because 4 bytes never
    /// needed it.
    fn graphReserve(self: *CudaBackend, max_cols: usize) !void {
        if (!self.gpuReady()) return error.BackendUnavailable;
        const cu = &self.cu.?;
        // q8_k is the wider of the two activation formats; reserving it for
        // both means `ensureBuf` at matvec time can never grow `d_xq` either.
        try ensureBuf(&cu.lib, &self.d_act, &self.d_act_bytes, max_cols * @sizeOf(f32));
        try ensureBuf(&cu.lib, &self.d_xq, &self.d_xq_bytes, actBytes(.q8_k, max_cols));
        try ensureBuf(&cu.lib, &self.d_out, &self.d_out_bytes, max_cols * @sizeOf(f32));
        try ensureBuf(&cu.lib, &self.d_deq, &self.d_deq_bytes, max_cols * @sizeOf(f32));
        try ensureBuf(&cu.lib, &self.d_idx, &self.d_idx_bytes, @sizeOf(c_int));
        // The reservation is what makes capture safe, so it is also what arms the
        // feature: before this point the XG phase stays `.off` because a capture
        // that early would have to allocate scratch inside the graph.
        //
        // `$QWEN35_XD=1` gates it, DEFAULT OFF.  The XG path is under
        // construction: with it on, a decode step currently loses its capture at
        // the first op that is not in `xd.cu`'s table (measured: op #157 of the
        // 0.8B step, right after the layer's MLP, with `cudaErrorIllegalAddress`
        // 700 left on the stream).  Turning it on by default would mean the
        // engine silently produced a garbled token stream -- see the "xg: replay"
        // note on `xd_launched`.  It is armed only when explicitly asked for,
        // so `QWEN35_CUDA_GRAPH=0` and the plain path stay the default.
        if (self.gs(0).xd_phase == .off and xdWanted()) self.gs(0).xd_phase = .rec;
        // `.verify` 槽同样 arm：`forwardChunkVerify` 只在 MTP 下被调用，
        // 不 arm 的话它永远停在 `.off`，第一轮录不到就再也不会重放。
        if (xdWanted()) self.gs(1).xd_phase = .rec;
    }

    /// Open the capture.  `pos` and `slot` are this token's values, recorded so
    /// node identification can verify itself afterwards.
    ///
    /// Returns false when the capture could not be opened (no stream, runtime
    /// refused); the caller then runs the token with direct launches.  A false
    /// return is NOT permanent -- a later token may try again -- except when
    /// `g_failed` was set after a real failure.
    fn graphSyncFromDefault(self: *CudaBackend) void {
        const cu = &(self.cu orelse return);
        const st = self.g_stream orelse return;
        const ev = self.g_ev_out orelse return;
        if (cu.lib.EventRecord(ev, null) != cudart.cudaSuccess) return;
        _ = cu.lib.StreamWaitEvent(st, ev, 0);
    }

    // ==== 显式计算图（XG）的状态机 ==========================================
    //
    // `off → warm → rec → live`，与 `graph_registry.zig` 的语义一致：
    //
    //   off   直接发射，不录制。
    //   warm  直接发射；`dGraphEnd` 把这步当作"惰性分配已稳态"的证据，转 rec。
    //         必须走这一步：capture 内出现 `cudaMalloc` 是非法的，而首个
    //         decode token 会分配 KV 量化缓冲等。
    //   rec   转发器开 capture 并逐个 `xd_capture`；`dGraphEnd` 关 capture、
    //         instantiate，然后**立刻 launch 一次**——capture 不执行 kernel，
    //         不 launch 这一步就没有任何计算发生过。
    //   live  每步 `apply` 全部节点 + 一次 `GraphLaunch`；转发器被 `xdGate`
    //         抑制，主机侧不再发射。
    //
    // 任何一步失败都转 `off`（不是 `.failed` 的永久放弃：`warm` 期的失败
    // 只说明"这次还不该录"，下一 token 再试一次就够；真的反复失败会表现为
    // 每 token 重试一次 capture，用 `QWEN35_CUDA_GRAPH=0` 关掉即可）。

    /// 开始/推进这一步。返回带 `ptr` 的图（`live`/`rec`）或直通图。
    ///
    /// 槽号来自**句柄**（`g.slot`），不是后端的"当前槽"。`graphBegin` 与
    /// `graphEnd` 会嵌套（草稿块在 verify 里面），全局当前槽会让外层的收尾
    /// 落到内层的槽上 —— 实测：草稿槽永远停在 `.off`，草稿的 op 全落到主机侧
    /// 又算一遍，接受率 0%。
    fn xdPhaseBegin(self: *CudaBackend, g: backend.Graph) backend.Graph {
        const s = self.gs(g.slot);
        const out_node = struct {
            fn make(gg: backend.Graph, ss: *GraphSlot, replayed: bool) backend.Graph {
                return .{
                    .kind = gg.kind,
                    .ctl = gg.ctl,
                    .slot = gg.slot,
                    .ptr = @ptrCast(ss),
                    .replayed = replayed,
                };
            }
        }.make;
        s.xd_launched = false;
        // prefill 的块长度是数据相关的（`min(nbatch, 剩余 token)`），而一张图
        // 只对一种行数成立；按形状建图最多要 nbatch-1 张。prefill 的发射开销
        // 在 238 ms 的块里上界只有 6.3%（AGENTS.md §5.7 的账本），所以
        // prefill 一律直通 —— 它现在压根不经过这里（`forwardChunk` 自己不开
        // span，只有 `forwardChunkVerify` 开 `.verify`）。
        //
        // `.verify` 走到这里只可能来自 `$QWEN35_XD_VERIFY=1`（默认不建图，
        // 见 `dGraphBegin`）。它的行数逐轮会变，所以这里按 `rows` 校验：一张
        // 图只对一种行数成立，行数变了必须重录，否则会重放一张节点数与 buffer
        // 行数都不匹配的图 —— 那是静默算错，不是报错。
        if (s.xd_phase == .live and s.xd_rows != g.ctl.rows) {
            if (s.xd_gexec) |e| {
                if (self.cu) |*cu| _ = cu.lib.GraphExecDestroy(e);
                s.xd_gexec = null;
            }
            s.xd_phase = .rec;
        }
        s.xd_rows = g.ctl.rows;
        switch (s.xd_phase) {
            .off => return g,
            .rec => {
                // 直通图：capture 期 op 必须照常发射，CUDA 才会记录它们；
                // 若返回带 ptr 的图，`graphGate` 会把每次发射都跳过，图就是空的。
                if (s.xd_capturing) return out_node(g, s, false);
                if (!self.xdStartCapture(s)) return g;
                return out_node(g, s, false);
            },
            .live => {
                // `$QWEN35_XD_NOREPLAY=1`：每一步都重新录制而不是重放。调试用，
                // 用来把"图的内容错"与"重放这个动作错"分开。
                if (cudart.getenv("QWEN35_XD_NOREPLAY") != null) {
                    if (s.xd_gexec) |e| {
                        if (self.cu) |*cu| _ = cu.lib.GraphExecDestroy(e);
                        s.xd_gexec = null;
                    }
                    s.xd_phase = .rec;
                    if (!self.xdStartCapture(s)) return g;
                    return out_node(g, s, false);
                }
                if (!self.xdReplay(s)) {
                    // 重放失败后**不能**退回主机侧直接发射：图可能已经执行
                    // （状态被推进一次），重跑整步会把状态推进两次。所以只有
                    // "图没启动"这一种失败允许降级；见 `xd_launched`。
                    s.xd_phase = .off;
                    if (s.xd_launched) return out_node(g, s, true);
                    return g;
                }
                // `.replayed = true` 是抑制主机侧发射的唯一依据：图的每一个节点
                // 都会在本步执行，主机再发一遍就是在默认流上把整步算第二次。
                return out_node(g, s, true);
            },
        }
    }

    fn xdPhaseEnd(self: *CudaBackend, g: backend.Graph) void {
        const s = self.gs(g.slot);
        switch (s.xd_phase) {
            .rec => self.xdFinishCapture(s),
            // `live` 的 `GraphLaunch` 已经在 `xdPhaseBegin` 里发过，没有收尾。
            else => {},
        }
    }

    /// 开 capture。失败返回 false，`xd_phase` 归 `.off`。
    fn xdStartCapture(self: *CudaBackend, slot: *GraphSlot) bool {
        const cu = &(self.cu orelse return false);
        if (!self.xdEnsureStream()) {
            slot.xd_phase = .off;
            return false;
        }
        // 先排空：capture 开始前默认流上的 embedding 拷贝必须已经落定，否则
        // `StreamBeginCapture` 直接拒绝（实测 rc = "operation cannot be
        // performed in the present state"）。ds4 与 Strata 也是先同步再开。
        // 同步失败也要清掉错误状态，后面才看得懂真正的失败点。
        if (cu.lib.StreamSynchronize(null) != cudart.cudaSuccess) _ = cu.lib.GetLastError();
        self.graphSyncFromDefault();
        // Global 而非 ThreadLocal：本后端在 capture 期间混用了默认流（embed 之后
        // 的 `launchStream()` 在 `.rec` 时是 `g_stream`，但部分路径仍显式传 null），
        // ThreadLocal 会把那些发射当成非法操作。ds4 用的是 Global。
        // Global 而非 ThreadLocal：本后端在 capture 期间混用了默认流（embed 之后
        // 的 `launchStream()` 在 `.rec` 时是 `g_stream`，但部分路径仍显式传 null），
        // ThreadLocal 会把那些发射当成非法操作。ds4 用的是 Global。
        const rc = cu.lib.StreamBeginCapture(self.g_stream, cudart.StreamCaptureModeGlobal);
        if (rc != cudart.cudaSuccess) {
            self.ewLaunchFail("xg: begin capture", rc);
            slot.xd_phase = .off;
            return false;
        }
        slot.xd_capturing = true;
        return true;
    }

    /// 关 capture：实例化、launch 一次（capture 不执行 kernel）、转 `live`。
    /// `$QWEN35_XD_NODES`: 打印录进图的 kernel 节点分布（按 `func` 地址聚合）。
    ///
    /// 用途是把"图里到底有什么"变成可读的事实。重放输出异常时，第一个要
    /// 问的问题是"图里是不是混进了不该有的节点"或"某个节点少了"，而这在
    /// 主机侧只看得到地址；把次数和 PROF 的发射次数对照即可判断。
    fn dumpGraphNodes(cu: *cudart.Open, graph: *anyopaque) void {
        var n: usize = 0;
        if (cu.lib.GraphGetNodes(graph, null, &n) != cudart.cudaSuccess or n == 0) return;
        const alloc = std.heap.page_allocator;
        const nodes = alloc.alloc(?*anyopaque, n) catch return;
        defer alloc.free(nodes);
        var got: usize = n;
        if (cu.lib.GraphGetNodes(graph, nodes.ptr, &got) != cudart.cudaSuccess) return;

        const Key = usize;
        var counts = std.AutoHashMap(Key, usize).init(alloc);
        defer counts.deinit();
        var non_kernel: usize = 0;
        for (nodes[0..got]) |nd| {
            var ty: c_int = 0;
            if (cu.lib.GraphNodeGetType(nd, &ty) != cudart.cudaSuccess) continue;
            if (ty != 0) { // cudaGraphNodeTypeKernel == 0
                non_kernel += 1;
                continue;
            }
            var p: cudart.KernelNodeParams = undefined;
            if (cu.lib.GraphKernelNodeGetParams(nd, &p) != cudart.cudaSuccess) continue;
            const k: Key = @intFromPtr(p.func);
            const gop = counts.getOrPut(k) catch continue;
            if (!gop.found_existing) gop.value_ptr.* = 0;
            gop.value_ptr.* += 1;
        }
        std.debug.print("[xg] graph has {d} nodes ({d} non-kernel kernels, {d} distinct funcs)\n", .{
            got, non_kernel, counts.count(),
        });
    }

    fn xdFinishCapture(self: *CudaBackend, s: *GraphSlot) void {
        const cu = &(self.cu orelse return);
        if (cudart.getenv("QWEN35_XD_REPLAYDBG") != null) {
            std.debug.print("[xg] finish capture, phase={s}\n", .{@tagName(s.xd_phase)});
        }
        var graph: ?*anyopaque = null;
        s.xd_capturing = false;
        const erc = cu.lib.StreamEndCapture(self.g_stream, &graph);
        if (erc != cudart.cudaSuccess or graph == null) {
            if (graph) |gr| _ = cu.lib.GraphDestroy(gr);
            self.ewLaunchFail("xg: end capture", erc);
            s.xd_phase = .off;
            return;
        }
        var gexec_new: ?*anyopaque = null;
        const irc = cu.lib.GraphInstantiate(&gexec_new, graph.?, 0);
        std.debug.print("[xg] instantiate rc={d} err={s}\n", .{ irc, cu.lib.GetErrorString(cu.lib.GetLastError()) });
        if (irc != cudart.cudaSuccess or gexec_new == null) {
            _ = cu.lib.GraphDestroy(graph.?);
            self.ewLaunchFail("xg: instantiate", irc);
            s.xd_phase = .off;
            return;
        }
        if (cudart.getenv("QWEN35_XD_NODES") != null) dumpGraphNodes(cu, graph.?);
        // 源图可以立刻销毁：本实现不保留任何节点句柄（无 op 表、无 apply），
        // 重放只认 `gexec`。参照项目里只有 ds4 保留 `graph`，那是因为它要用
        // `cudaGraphAddKernelNode` 手工改图。
        _ = cu.lib.GraphDestroy(graph.?);
        s.xd_gexec = gexec_new;
        s.xd_phase = .live;
        // capture 只记录不执行：这一 token 至今没有任何 kernel 跑过，必须
        // launch 一次，否则 KV/GDN 状态都停在上一个 token。
        const lrc = cu.lib.GraphLaunch(gexec_new, self.g_stream.?);
        if (lrc != cudart.cudaSuccess) {
            self.ewLaunchFail("xg: first launch", lrc);
            s.xd_phase = .off;
            _ = cu.lib.GraphExecDestroy(gexec_new.?);
            s.xd_gexec = null;
        }
    }

    /// 重放：先按序 `apply` 每个节点（刷新实参），再 launch，最后把默认流
    /// 与图流排序。
    ///
    /// 收尾那一步不是可选的。`GraphLaunch` 是异步的、落在 `g_stream` 上，
    /// 而 `launchStream()` 在图活着的时候返回默认流；图之后的所有工作
    /// （lm_head matvec、读它结果的 argmax）因此会和图并发，除非显式排序。
    /// 旧机制在这里踩过：`backend_cuda.zig` 的 `graphReplayInner` 注释记录了
    /// "参数全冻结、每次跑出不同 token"，只加这一道 barrier 就逐位一致。
    fn xdReplay(self: *CudaBackend, s: *GraphSlot) bool {
        const cu = &(self.cu orelse return false);
        const gexec_live = s.xd_gexec orelse return false;
        // 没有 `apply`：参照项目（llama.cpp / ds4 / Strata）重放前都不改节点。
        // 逐 token 变化的标量由设备控制块在 kernel 内读取（S3a/S3b），其余
        // 实参在 capture 时就已固化成常量区域。
        const st = self.g_stream orelse return false;
        if (cudart.getenv("QWEN35_XD_REPLAYDBG") != null) {
            std.debug.print("[xg] replay pos={d} kv={d} attn={d} rows={d}\n", .{
                self.ctl_host.pos, self.ctl_host.kv_slot, self.ctl_host.attn_slot, self.ctl_host.rows,
            });
        }
        // `dGraphBegin` 在本步开始时把控制块写到 DEFAULT 流上（publish 是普通
        // kernel launch）。图在 `g_stream` 上读同一个块，两条流之间没有隐式
        // 顺序，所以重放前必须显式对齐；否则图会读到上一 token 的 pos/slot。
        self.graphSyncFromDefault();
        s.xd_launched = true;
        const lrc = cu.lib.GraphLaunch(gexec_live, st);
        if (lrc != cudart.cudaSuccess) {
            self.ewLaunchFail("xg: replay launch", lrc);
            return false;
        }
        // 事件等待两侧都发：事件排在默认流上；`StreamWaitEvent(null, ...)`
        // 不被支持时由 `EventSynchronize` 兜底。
        if (self.g_ev_out) |ev| {
            if (cu.lib.EventRecord(ev, st) == cudart.cudaSuccess) {
                _ = cu.lib.StreamWaitEvent(null, ev, 0);
                _ = cu.lib.EventSynchronize(ev);
            }
        }
        _ = cu.lib.StreamSynchronize(st);
        return true;
    }

    /// 图用的私有流。与旧机制共用 `g_stream`（它们不会同时活着：XG 的
    /// `xd_phase` 非 `.off` 时旧的 `state` 恒为 `.off`）。
    fn xdEnsureStream(self: *CudaBackend) bool {
        const cu = &(self.cu orelse return false);
        if (self.g_stream == null) {
            var st: ?*anyopaque = null;
            const rc = cu.lib.StreamCreateWithFlags(&st, cudart.StreamNonBlocking);
            if (rc != cudart.cudaSuccess or st == null) {
                self.ewLaunchFail("xg: stream create", rc);
                return false;
            }
            self.g_stream = st;
        }
        if (self.g_ev_out == null) {
            var ev: ?*anyopaque = null;
            const erc = cu.lib.EventCreate(&ev, cudart.EventDisableTiming);
            if (erc != cudart.cudaSuccess or ev == null) {
                self.ewLaunchFail("xg: ordering event create", erc);
                return false;
            }
            self.g_ev_out = ev;
        }
        return true;
    }

    /// The mirror image, for the END of the step: whatever the graph produced
    /// (the KV row it just appended, the GDN state it just advanced) has to be
    /// visible to the default stream before the next step's embedding and before
    /// `sample`.  The default stream is the LEGACY one, and — the trap this
    /// used to fall into — `cudaStreamWaitEvent(0, ...)` is not merely "the
    /// cheapest correct barrier", it is *rejected*: the legacy stream cannot be
    /// made to wait on an event, the call returns `cudaErrorInvalidResourceHandle`
    /// and the `_ =` below threw that away, so the barrier silently did not
    /// exist.  Everything after this point (the argmax D2H that reads the logits,
    /// the next token's embedding write) then raced the graph.  Measured: with
    /// every node's arguments frozen the replayed text still changed from run to
    /// run, and `CUDA_LAUNCH_BLOCKING=1` made the same run correct.  Blocking the
    /// host on the event is what actually orders the two streams; the overlap it
    /// gives up is the host-side sampling, which cannot start before this point
    /// anyway.
    /// Free the graph state.  Idempotent; called from `deinit`.
    fn graphDestroy(self: *CudaBackend) void {
        if (self.cu) |*cu| {
            for (self.g_slots) |*sl| {
                if (sl.xd_gexec) |e| {
                    _ = cu.lib.GraphExecDestroy(e);
                    sl.xd_gexec = null;
                }
                sl.xd_phase = .off;
                sl.xd_capturing = false;
            }
            if (self.gs(0).xd_gexec) |e| {
                _ = cu.lib.GraphExecDestroy(e);
                self.gs(0).xd_gexec = null;
            }
            if (self.g_ev_out) |ev| {
                _ = cu.lib.EventDestroy(ev);
                self.g_ev_out = null;
            }
            if (self.g_ev) |ev| {
                _ = cu.lib.EventDestroy(ev);
                self.g_ev = null;
            }
            if (self.g_stream) |s| {
                _ = cu.lib.StreamDestroy(s);
                self.g_stream = null;
            }
        }
        for (self.g_slots) |*sl| {
            sl.xd_phase = .off;
            sl.xd_capturing = false;
        }
        if (self.stat_g_replay != 0 or self.stat_g_abandon != 0) {
            std.debug.print("cuda: graph replays={d} abandoned-tokens={d}\n", .{
                self.stat_g_replay, self.stat_g_abandon,
            });
        }
    }

    /// Copy one activation vector to `d_act`, growing the buffer as needed.
    fn uploadAct(self: *CudaBackend, x: []const f32) void {
        const cu = &self.cu.?;
        const bytes = x.len * @sizeOf(f32);
        ensureBuf(&cu.lib, &self.d_act, &self.d_act_bytes, bytes) catch {
            std.debug.print("cuda backend: activation upload buffer allocation failed ({d} B)\n", .{bytes});
            return;
        };
        const rc = cu.lib.Memcpy(self.d_act, x.ptr, bytes, cudart.MemcpyHostToDevice);
        self.stat_h2d_bytes += bytes;
        self.stat_host_act_uploads += 1;
        if (rc != cudart.cudaSuccess) {
            std.debug.print("cuda backend: activation upload failed: {s}\n", .{std.mem.span(cu.lib.errStr(rc))});
        }
    }

    /// Launch the f16/bf16 (w16) matvec.  `d_w` is the resident 2-byte-per-element
    /// weights, `d_x` the f32 activation.  Writes straight into `outBuf` when that
    /// is already device memory (the CUDA-graph decode path), otherwise into the
    /// shared staging buffer and downloads.  Returns the launch result code so the
    /// caller reports it; 0 (cudaSuccess) means the work is queued/done.
    fn launchW16(self: *CudaBackend, g: backend.Graph, d_w: ?*const anyopaque, d_x: ?[*]const f32,
                 ttype: gguf.TensorType, rows: usize, cols: usize, tokens: usize,
                 outBuf: backend.Buf) c_int {
        const n = tokens * rows;
        if (dBufPtr(outBuf)) |pd| {
            return Launch.mul_mat_vec_w16(g, d_w, d_x, pd, @intCast(rows), @intCast(cols),
                                          @intCast(tokens), @intCast(@backingInt(ttype)),
                                          self.launchStream());
        }
        ensureBuf(&self.cu.?.lib, &self.d_out, &self.d_out_bytes, n * @sizeOf(f32)) catch {
            std.debug.print("cuda backend: w16 matvec: device output allocation failed ({d} B)\n", .{n * @sizeOf(f32)});
            // Negative, not a cudaSuccess value, so the caller's error path fires
            // (the same shape `stubNoCudaKernels` uses).
            return -1;
        };
        const rc = Launch.mul_mat_vec_w16(g, d_w, d_x, @ptrCast(@alignCast(self.d_out)),
                                          @intCast(rows), @intCast(cols), @intCast(tokens),
                                          @intCast(@backingInt(ttype)), self.launchStream());
        // The staging branch must download (same contract as the f32 branch: a
        // download abandons any open graph capture through `dCopyToHost`).
        if (rc == cudart.cudaSuccess) self.downloadOut(outBuf.slice()[0..n]);
        return rc;
    }

    fn downloadOut(self: *CudaBackend, out: []f32) void {
        const cu = &self.cu.?;
        const bytes = out.len * @sizeOf(f32);
        const rc = cu.lib.Memcpy(out.ptr, self.d_out, bytes, cudart.MemcpyDeviceToHost);
        self.stat_d2h_bytes += bytes;
        self.stat_host_out_downloads += 1;
        if (rc != cudart.cudaSuccess) {
            std.debug.print("cuda backend: result download failed: {s}\n", .{std.mem.span(cu.lib.errStr(rc))});
        }
    }

    /// Quantize one token's activation into `self.d_xq`, in the format the weight
    /// type needs (Q8_0 for 32-element-block weights, Q8_K otherwise).  Returns the
    /// byte count written.
    ///
    /// The activation is read straight from a device pointer, which after P2d is
    /// normally the caller's own resident buffer (`xBuf`), not a `d_act` staging
    /// area -- staging is only written when a caller hands over a host slice.
    fn quantizeActFrom(self: *CudaBackend, g: backend.Graph, d_src: [*]const f32, ttype: gguf.TensorType, cols: usize) !usize {
        const cu = &self.cu.?;
        const nb = cols / actBlockElems(ttype);
        std.debug.assert(nb * actBlockElems(ttype) == cols);
        const kern = Launch;
        const rc = if (actBlockElems(ttype) == gguf.blockElems(.q8_0))
            kern.quantize_q8_0(d_src, self.d_xq.?, @intCast(nb), self.launchStream())
        else
            kern.quantize_q8_k(d_src, self.d_xq.?, @intCast(nb), self.launchStream());
        if (rc != cudart.cudaSuccess) {
            if (self.capturing(g)) self.graphAbandon(g, "quantize activation launch failed");
            std.debug.print("cuda backend: quantize({s}) launch failed: {s}\n", .{
                @tagName(ttype), std.mem.span(cu.lib.errStr(rc)),
            });
            return error.LaunchFailed;
        }
        return nb * actBlockBytes(ttype);
    }

    /// Memoisation for `quantizeActFrom`: does `d_xq` already hold the
    /// activation of `src` quantised to `fam` at width `cols`?  Cleared by
    /// `stepBegin` and by every host write into an activation buffer
    /// (`copyFromHost`), the only two ways those bytes can change under us.
    fn actQValid(self: *CudaBackend, src: [*]const f32, fam: cpu_mod.ActFamily, cols: usize) bool {
        const k = self.act_q;
        // NB: no ttype in the key.  The packed layout is a function of the
        // FAMILY alone (`actFamily` collapses every 256-element block type onto
        // Q8_K), so q4_k -> iq4_xs -> iq3_xxs in one phase must hit.  Keying on
        // the ttype too made every one of those transitions a MISS and put the
        // quantize back on the critical path: measured 103 hits / 399 misses per
        // token, which is the 9.6 ms/token the replay bench reproduces exactly
        // (matvecs alone 22.7 ms, +quantize-per-matvec 32.2 ms).
        return k.n != 0 and k.src == src and k.cols == cols and k.fam == fam;
    }

    fn actQMark(self: *CudaBackend, src: [*]const f32, fam: cpu_mod.ActFamily, cols: usize) void {
        self.act_q = .{ .src = src, .n = 1, .cols = cols, .fam = fam };
    }

    fn actQClear(self: *CudaBackend) void {
        self.act_q = .{ .src = null, .n = 0, .cols = 0, .fam = .q8_0 };
    }

    // ==== slot: loadLinear / freeLinear =====================================

    /// Upload every weight the CPU loader accepted, packed or f32.
    ///
    /// The CPU loader stays authoritative for geometry, mmap resolution and type
    /// support; it already reports missing/unknown tensors with the `LoadError` the
    /// engine expects.  This function adds exactly one decision: does the tensor
    /// fit on the device, and if not, say so loudly.
    fn vLoadLinear_body(ctx: *anyopaque, name: []const u8) backend.LoadError!backend.LinearW {
        const self = self_(ctx);
        const w = try self.cpu.exec().loadLinear(name);
        errdefer self.cpu.exec().freeLinear(w);

        if (!self.gpuReady()) {
            std.debug.print("cuda backend: refusing to load '{s}' on the CPU ({s})\n", .{
                name, self.needHelp(),
            });
            return error.BackendUnavailable;
        }
        const cu = &self.cu.?;

        // Host bytes to upload, plus what the launcher needs to know about the
        // tensor's shape and type.
        const holder: []const u8 = switch (w) {
            .quant => |q| q.data,
            .f32 => |m| mem.sliceAsBytes(m),
        };
        const ttype: ?gguf.TensorType = switch (w) {
            .quant => |q| q.ttype,
            .f32 => null,
        };
        // Geometry: the QuantRow view knows rows/cols for a packed weight.  A
        // `.f32` tensor is a flat slice, so it is registered as one row of `len`
        // columns; the engine only sends f32 weights through `dequantRow` or a
        // `matvec` whose x is exactly that wide.
        const geom = switch (w) {
            .quant => |q| Geometry{ .rows = q.rows, .cols = q.cols() },
            // A `.f32` weight is `LinearW.f32`, a flat f32 slice with no
            // geometry of its own; recover cols/rows from the tensor's GGUF dims
            // (GGML row-major: dims[0] is the row length).  Falls back to a
            // single row so a missing tensor still reports a clear shape error
            // from the matvec assert rather than silently using the wrong cols.
            // f16/bf16 never land here: they are `QuantRow`s with the GGUF
            // 256-element row framing, so `.quant` above already gives the shape.
            .f32 => |m| if (self.g.findTensor(name)) |t|
                Geometry{
                    .rows = if (t.dims.len > 1) @intCast(t.nElems() / @max(t.dims[0], 1)) else 1,
                    .cols = @intCast(t.dims[0]),
                }
            else
                Geometry{ .rows = 1, .cols = m.len },
        };
        const bytes = holder.len;
        if (bytes == 0) return error.MissingTensor;

        if (self.budgetExceeded()) {
            std.debug.print(
                "cuda backend: $QWEN35_CUDA_MAX_DEV_MIB budget exceeded by '{s}' ({d} MiB; {d} MiB uploaded already)\n",
                .{ name, bytes >> 20, self.stat_uploaded_bytes >> 20 },
            );
            return error.OutOfMemory;
        }

        var dp: ?*anyopaque = null;
        const rc = cu.lib.Malloc(&dp, bytes);
        if (rc != cudart.cudaSuccess) {
            // No silent CPU retreat: a partially resident model would be both slow
            // and impossible to attribute.  Name the tensor, type, size and what is
            // left on the device, then stop the load.
            std.debug.print(
                "cuda backend: cudaMalloc({d} MiB) for '{s}' ({s}) failed: {s}; {d} MiB free on the device\n" ++
                    "       a cuda run must fit entirely in device memory; use --backend cpu if it does not\n",
                .{
                    bytes >> 20,
                    name,
                    if (ttype) |t| @tagName(t) else "f32",
                    std.mem.span(cu.lib.errStr(rc)),
                    freeDevMiB(&cu.lib),
                },
            );
            return error.OutOfMemory;
        }

        const rc2 = cu.lib.Memcpy(dp, holder.ptr, bytes, cudart.MemcpyHostToDevice);
        if (rc2 != cudart.cudaSuccess) {
            _ = cu.lib.Free(dp);
            std.debug.print("cuda backend: upload of '{s}' ({d} MiB) failed: {s}\n", .{
                name, bytes >> 20, std.mem.span(cu.lib.errStr(rc2)),
            });
            return error.OutOfMemory;
        }

        self.dev_rows.append(self.alloc, .{
            .host = holder.ptr,
            .dev = dp.?,
            .bytes = bytes,
            .ttype = ttype,
            .rows = geom.rows,
            .cols = geom.cols,
        }) catch {
            _ = cu.lib.Free(dp);
            return error.OutOfMemory;
        };
        self.stat_uploaded_bytes += bytes;
        return w;
    }

    fn vFreeLinear(ctx: *anyopaque, w: backend.LinearW) void {
        const self = self_(ctx);
        const host: [*]const u8 = switch (w) {
            .quant => |q| q.data.ptr,
            .f32 => |m| mem.sliceAsBytes(m).ptr,
        };
        if (self.cu) |cu| {
            var i: usize = 0;
            while (i < self.dev_rows.items.len) : (i += 1) {
                if (self.dev_rows.items[i].host == host) {
                    _ = cu.lib.Free(self.dev_rows.items[i].dev);
                    _ = self.dev_rows.swapRemove(i);
                    break;
                }
            }
        }
        // Host side: both arms are now zero-copy views into the GGUF mapping
        // (`.quant` and `.f32` alike), so there is nothing host-side to release
        // here -- only the device copy above.
        self.cpu.exec().freeLinear(w);
    }

    // ==== slot: loadVecF32 ==================================================

    /// Load a 1-D f32 weight vector by name and hand back a device-resident
    /// `Buf`.  llama.cpp-style residency: the tensor is uploaded exactly once,
    /// at load time, and the returned `Buf` IS the residency -- its `ptr` is
    /// the device pointer the elementwise kernels (`rmsnorm`, `convStep`,
    /// `gdnGamma`, ...) read.  There is no host copy and no first-use upload:
    /// the old `devConst` path is gone.
    ///
    /// The widening/dequant is the CPU loader's single definition (f16/bf16/
    /// q8_0 -> f32); we upload those bytes and then release the host slice, so
    /// the weight lives only on the device from here on.
    fn vLoadVecF32(ctx: *anyopaque, name: []const u8) backend.LoadError!backend.Buf {
        const self = self_(ctx);
        const host = self.cpu.exec().loadVecF32(name) catch |err| return err;
        errdefer self.cpu.exec().freeVec(host);
        if (host.len == 0) return backend.Buf.empty;
        if (!self.gpuReady()) {
            return error.BackendUnavailable;
        }
        const cu = &self.cu.?;
        const bytes = host.len * @sizeOf(f32);
        var dp: ?*anyopaque = null;
        const rc = cu.lib.Malloc(&dp, bytes);
        if (rc != cudart.cudaSuccess) {
            std.debug.print("cuda backend: 1-D weight '{s}' ({d} B) cudaMalloc failed: {s}\n", .{
                name, bytes, std.mem.span(cu.lib.errStr(rc)),
            });
            return error.OutOfMemory;
        }
        const host_bytes = mem.sliceAsBytes(host.slice());
        const rc2 = cu.lib.Memcpy(dp, host_bytes.ptr, bytes, cudart.MemcpyHostToDevice);
        if (rc2 != cudart.cudaSuccess) {
            _ = cu.lib.Free(dp);
            std.debug.print("cuda backend: 1-D weight '{s}' ({d} B) upload failed: {s}\n", .{
                name, bytes, std.mem.span(cu.lib.errStr(rc2)),
            });
            return error.OutOfMemory;
        }
        // Uploaded: the host slice is dead; the `Buf` we hand back is the device
        // region.  (Success return, so the `errdefer` above does NOT run.)
        self.cpu.exec().freeVec(host);
        const dev: [*]f32 = @ptrCast(@alignCast(dp.?));
        return backend.Buf{ .ptr = dev, .len = host.len };
    }

    fn vFreeVec(ctx: *anyopaque, v: backend.Buf) void {
        const self = self_(ctx);
        if (v.len == 0) return; // `Buf.empty` sentinel: nothing was uploaded.
        if (self.cu) |cu| {
            _ = cu.lib.Free(@ptrCast(v.ptr));
        }
        // Context already torn down: the device memory died with it.
    }

    // ==== slot: dequantRow ==================================================

    /// Expand one row of a packed weight on the device.
    ///
    /// The source row is read *on the device*: the kernel gets `dev + row *
    /// rowBytes` (device-side pointer arithmetic, no host round trip), so a 27B
    /// embedding table is never re-uploaded per token.  The `cols` f32 result goes
    /// straight into `dstBuf`'s device region -- P2d removed the `downloadOut`
    /// that used to push every embedding row and lm_head row back through the
    /// host.  Buffers are the shared pair, grown by `ensureBuf`, so a repeated
    /// call does no `cudaMalloc`.
    ///
    /// The engine calls this once per row (token embedding, lm_head), so it is a
    /// latency path rather than a throughput one.
    fn vDequantRow_body(g: backend.Graph, ctx: *anyopaque, dstBuf: backend.Buf, w: gguf.QuantRow, row: usize) void {
        const self = self_(ctx);
        if (!self.gpuReady()) {
            std.debug.print("cuda backend: dequantRow: {s}\n", .{self.needHelp()});
            self.clearDevBuf(dstBuf);
            return;
        }
        const e = self.findDev(w.data.ptr) orelse {
            // Only reachable for a weight never loaded through this backend.  Say
            // so rather than compute a plausible-looking row on the CPU -- exactly
            // the class of silent divergence in AGENTS.md §5.2.
            std.debug.print("cuda backend: dequantRow: a '{s}' weight is not resident on the device\n", .{@tagName(w.ttype)});
            self.clearDevBuf(dstBuf);
            return;
        };
        const cu = &self.cu.?;
        const cols = w.cols();
        const row_bytes = w.rowBytes();
        std.debug.assert(dstBuf.len >= cols);
        std.debug.assert(row < w.rows);

        const out_bytes = cols * @sizeOf(f32);
        // Destination: the caller's own device region when it has one (the normal
        // case), otherwise the shared `d_out` plus the one download that a host
        // caller still needs.
        const dst_dev: ?[*]f32 = dBufPtr(dstBuf);
        const d_out: [*]f32 = dst_dev orelse blk: {
            ensureBuf(&cu.lib, &self.d_out, &self.d_out_bytes, out_bytes) catch {
                std.debug.print("cuda backend: dequantRow: cannot allocate {d} B of device output\n", .{out_bytes});
                self.clearDevBuf(dstBuf);
                return;
            };
            break :blk @ptrCast(@alignCast(self.d_out));
        };

        const t0 = Io.Clock.Timestamp.now(self.io, .awake);
        // Kernel entry points are link-time externs, so the "table" is just the
        // namespace of those declarations -- see `Kernels` at the top of the file.
        const kern = Launch;
        const src: [*]const u8 = @ptrCast(e.dev);
        // 草稿块（`nextn` 图）的嵌入行号每轮都变。它不能作为实参：录进图之后
        // 重放永远嵌入第一轮那个 token，所以改成从设备控制块读
        // （`ctl[XD_CTL_TOKEN]`），实参只剩模型常量。decode 的嵌入在图外，
        // 行号照旧走指针算术。
        const rc = if (g.kind == .nextn and self.graphActive(g) and self.ctl_dev != null)
            kern.dequantize_row_dtoken(
                g,
                @intCast(@backingInt(w.ttype)),
                @ptrCast(e.dev),
                d_out,
                @intCast(cols),
                @intCast(row_bytes),
                self.ctl_dev.?,
                self.launchStream(),
            )
        else blk: {
            const d_row = src + row * row_bytes;
            break :blk kern.dequantize_row(g,
                @intCast(@backingInt(w.ttype)),
                d_row,
                d_out,
                @intCast(cols),
                self.launchStream(),
            );
        };
        if (rc != cudart.cudaSuccess) {
            if (self.capturing(g)) self.graphAbandon(g, "dequantize_row launch failed");
            std.debug.print("cuda backend: dequantize_row({s}) launch failed: {s}\n", .{
                @tagName(w.ttype), std.mem.span(cu.lib.errStr(rc)),
            });
            self.clearDevBuf(dstBuf);
            return;
        }
        if (dst_dev == null) {
            const rc2 = cu.lib.Memcpy(dstBuf.slice().ptr, d_out, out_bytes, cudart.MemcpyDeviceToHost);
            if (rc2 != cudart.cudaSuccess) {
                std.debug.print("cuda backend: dequantRow download failed: {s}\n", .{std.mem.span(cu.lib.errStr(rc2))});
                @memset(dstBuf.slice(), 0);
                return;
            }
            // Only a caller that handed us host memory gets a D2H here; the
            // engine's embedding / lm_head buffers are device regions, so this
            // counter staying at 0 is the proof the rows never came back.
            self.stat_d2h_bytes += out_bytes;
            self.stat_host_out_downloads += 1;
        }
        self.stat_ns_dequant += durNs(t0, self.io);
        self.stat_n_dequant += 1;
    }

    // ==== slot: matvec ======================================================

    /// `out = W * x` entirely on the device: quantize x in place (or, for the
    /// one legitimate host caller, upload it once), table-driven matvec, result
    /// written straight into `outBuf`'s device region.
    ///
    /// P2d: no `slice()` and no `downloadOut` on either leg.  `xBuf` / `outBuf`
    /// are backend regions (`dAllocBuf`), so the kernel reads and writes them
    /// directly; the only host->device copy left is `uploadAct`, reached solely
    /// when a caller genuinely hands over a host slice (`dBufPtr` null).
    /// `ttype` only chooses the activation format and quantizer -- never a
    /// backend, and never a CPU branch.
    /// `matvec` twice over ADJACENT weights as one launch, falling back to two
    /// `vMatvec_body` calls whenever adjacency cannot be proven (see
    /// `backend.Exec.matvecPair` for why the 27B's ssm_alpha/ssm_beta pair
    /// makes this worth doing at all).
    ///
    /// The legality test is on the DEVICE ranges: same tensor type, same shape,
    /// and `a_bytes` past `a.dev` landing exactly on `b.dev`.  GGUF alignment
    /// padding between the two tensors would break that (it does not here --
    /// checked for all 48 layers), and gluing across a pad would make the kernel
    /// read the pad as a block, so the test is exact rather than approximate.
    fn vMatvecPair_body(g: backend.Graph, ctx: *anyopaque, outBuf: backend.Buf, a: backend.LinearW, b: backend.LinearW, xBuf: backend.Buf) void {
        const self = self_(ctx);
        const r = outBuf.len / 2;
        if (outBuf.len % 2 != 0 or r == 0) {
            vMatvec_body(g, ctx, outBuf, a, xBuf);
            return;
        }
        if (pairViewInit(self, a, b, r)) |pair| {
            vMatvec_body(g, ctx, outBuf, .{ .quant = pair }, xBuf);
            return;
        }
        vMatvec_body(g, ctx, outBuf.span(0, r), a, xBuf);
        vMatvec_body(g, ctx, outBuf.span(r, r), b, xBuf);
    }

    /// The fused `QuantRow` for `matvecPair`, or null when the two weights are
    /// not provably one contiguous `2r x cols` block of the same type.  Exists
    /// so the decision is one readable predicate instead of seven early
    /// returns, and so the check is unit-testable on its own.
    fn pairViewInit(
        self: *CudaBackend,
        a: backend.LinearW,
        b: backend.LinearW,
        r: usize,
    ) ?gguf.QuantRow {
        const qa = switch (a) {
            .quant => |q| q,
            else => return null,
        };
        const qb = switch (b) {
            .quant => |q| q,
            else => return null,
        };
        const ea = self.findDev(qa.data.ptr) orelse return null;
        const eb = self.findDev(qb.data.ptr) orelse return null;
        if (ea.ttype != eb.ttype or qa.ttype != qb.ttype) return null;
        if (ea.rows != r or eb.rows != r) return null;
        if (ea.cols != eb.cols or qa.blocks_per_row != qb.blocks_per_row) return null;
        if (@intFromPtr(ea.dev) + ea.bytes != @intFromPtr(eb.dev)) return null;
        // The bytes must also be contiguous in the host view the kernel was
        // uploaded from, so re-check on that side too: `findDev` keys on the
        // host pointer, and a device-side match with a host-side gap would mean
        // the upload played games we are not modelling.
        if (qa.data.len != ea.bytes or qb.data.len != eb.bytes) return null;
        if (@intFromPtr(qa.data.ptr) + qa.data.len != @intFromPtr(qb.data.ptr)) return null;
        return .{
            .data = qa.data.ptr[0 .. qa.data.len + qb.data.len],
            .ttype = qa.ttype,
            .rows = 2 * r,
            .blocks_per_row = qa.blocks_per_row,
        };
    }

    fn vMatvec_body(g: backend.Graph, ctx: *anyopaque, outBuf: backend.Buf, w: backend.LinearW, xBuf: backend.Buf) void {
        const self = self_(ctx);
        if (graphGate(g)) return;
        if (!self.gpuReady()) {
            std.debug.print("cuda backend: matvec: {s}\n", .{self.needHelp()});
            self.clearDevBuf(outBuf);
            return;
        }
        const cu = &self.cu.?;
        const t0 = Io.Clock.Timestamp.now(self.io, .awake);

        // 设备激活指针：调用点本来就在设备侧时（整条逐 token 路径）用常驻缓冲，
        // 否则一次性上传到 `d_act`（首次使用只在主机上的张量）。
        const d_x: [*]const f32 = blk: {
            if (dBufPtr(xBuf)) |p| break :blk @ptrCast(p);
            if (xBuf.len != 0) self.uploadAct(xBuf.slice());
            break :blk @ptrCast(@alignCast(self.d_act.?));
        };

        switch (w) {
            .f32 => |m| {
                const e = self.findDev(mem.sliceAsBytes(m).ptr) orelse {
                    std.debug.print("cuda backend: matvec: f32 weight is not resident on the device\n", .{});
                    self.clearDevBuf(outBuf);
                    return;
                };
                const rows = e.rows;
                const cols = e.cols;
                std.debug.assert(xBuf.len >= cols and outBuf.len >= rows);
                if (dBufPtr(outBuf)) |pd| {
                    const rc = Launch.mul_mat_vec_f32(g, 
                        @ptrCast(@alignCast(e.dev)),
                        d_x,
                        pd,
                        @intCast(rows),
                        @intCast(cols),
                        1,
                        self.launchStream(),
                    );
                    if (rc != cudart.cudaSuccess) {
                        if (self.capturing(g)) self.graphAbandon(g, "mul_mat_vec_f32 launch failed");
                        std.debug.print("cuda backend: mul_mat_vec_f32 launch failed: {s}\n", .{std.mem.span(cu.lib.errStr(rc))});
                        self.clearDevBuf(outBuf);
                        return;
                    }
                } else {
                    ensureBuf(&cu.lib, &self.d_out, &self.d_out_bytes, rows * @sizeOf(f32)) catch {
                        std.debug.print("cuda backend: matvec: device output allocation failed\n", .{});
                        self.clearDevBuf(outBuf);
                        return;
                    };
                    const rc = Launch.mul_mat_vec_f32(g, 
                        @ptrCast(@alignCast(e.dev)),
                        d_x,
                        @ptrCast(@alignCast(self.d_out)),
                        @intCast(rows),
                        @intCast(cols),
                        1,
                        self.launchStream(),
                    );
                    if (rc != cudart.cudaSuccess) {
                        if (self.capturing(g)) self.graphAbandon(g, "mul_mat_vec_f32 launch failed");
                        std.debug.print("cuda backend: mul_mat_vec_f32 launch failed: {s}\n", .{std.mem.span(cu.lib.errStr(rc))});
                        self.clearDevBuf(outBuf);
                        return;
                    }
                    // 这个分支会下载结果，而下载会经 `dCopyToHost` 放弃任何已经
                    // 打开的 capture；不需要另外检查。
                }
                self.stat_ns_matvec += durNs(t0, self.io);
                self.stat_by_type[@backingInt(gguf.TensorType.f32)] += 1;
                self.stat_n_matvec += 1;
                return;
            },
            .quant => |q| {
                const e = self.findDev(q.data.ptr) orelse {
                    std.debug.print("cuda backend: matvec: a '{s}' weight is not resident on the device\n", .{@tagName(q.ttype)});
                    self.clearDevBuf(outBuf);
                    return;
                };
                const cols = q.cols();
                const rows = q.rows;
                std.debug.assert(xBuf.len >= cols and outBuf.len >= rows);

                // f16/bf16：不做激活量化，也不拷 f32 权重。权重以原生 2 字节
                // 形态常驻，kernel 拿它们逐元素对齐引擎的 f32 激活做加宽
                // （mmvq.cu 的 `q35_k_mul_mat_vec_w16`）。
                if (q.ttype == .f16 or q.ttype == .bf16) {
                    const rcw = self.launchW16(g, e.dev, d_x, q.ttype, rows, cols, 1, outBuf);
                    if (rcw != cudart.cudaSuccess) {
                        if (self.capturing(g)) self.graphAbandon(g, "mul_mat_vec_w16 launch failed");
                        std.debug.print("cuda backend: mul_mat_vec({s}) launch failed: {s}\n", .{
                            @tagName(q.ttype), std.mem.span(cu.lib.errStr(rcw)),
                        });
                        self.clearDevBuf(outBuf);
                        return;
                    }
                    self.stat_ns_matvec += durNs(t0, self.io);
                    self.stat_by_type[@backingInt(q.ttype)] += 1;
                    self.stat_n_matvec += 1;
                    return;
                }

                if (cudart.getenv("QWEN35_MVLIST") != null) {
                    std.debug.print("[mvl] {s} rows={d} cols={d}\n", .{ @tagName(q.ttype), rows, cols });
                }
                const xq_bytes = actBytes(q.ttype, cols);
                ensureBuf(&cu.lib, &self.d_xq, &self.d_xq_bytes, xq_bytes) catch {
                    std.debug.print("cuda backend: matvec: device activation allocation failed ({d} B)\n", .{xq_bytes});
                    self.clearDevBuf(outBuf);
                    return;
                };
                // 每个（来源、宽度、激活**族**）只量化一次，而不是每个 matvec
                // 一次。引擎给同一阶段的每个 matvec 传的是**同一个**激活缓冲
                // （27B 实测：一个 token 的 497 次 matvec 只碰 3 个不同的缓冲），
                // 而打包形态只取决于族 —— `actFamily` 把所有 256 元素块类型都
                // 归到同一个 Q8_K 布局 —— 于是同一批字节曾在一个 token 里被重复
                // 量化上百次。
                //
                // 为什么这件事要紧：量化是排在 matvec **关键路径**上的一次发射，
                // 它排空之前 GPU 起不了那 60 多微秒的权重流量。17408x5120 的
                // iq3_xxs 实测：单独 matvec 54.9 us（666 GB/s），matvec 排在它
                // 自己的量化之后 75.2 us（486 GB/s）—— 三分之一的带宽丢在重复
                // 工作上。
                //
                // 之所以成立，是因为 matvec 只**读** `x`（engine.zig 把投影写进
                // 别的缓冲），且 `stepBegin` 每 token 清一次这个缓存。按族而不是
                // 按 ttype 做键，正是它合法的原因：q4_k / iq4_xs / iq3_xxs 用的
                // 是同一个 Q8_K 激活，所以在它们之间切换**不能**让它失效。
                const fam = cpu_mod.actFamily(q.ttype);
                const _hit = self.actQValid(d_x, fam, cols);
                if (cudart.getenv("QWEN35_QSTAT") != null) {
                    self.q_hit += @intFromBool(_hit);
                    self.q_miss += @intFromBool(!_hit);
                }
                if (!_hit) {
                    _ = self.quantizeActFrom(g, d_x, q.ttype, cols) catch {
                        self.clearDevBuf(outBuf);
                        return;
                    };
                    self.actQMark(d_x, fam, cols);
                }
                if (dBufPtr(outBuf)) |pd| {
                    // 实参顺序照 api.h：(d_w, d_xq, d_out, rows, cols, tokens, ttype, stream)。
                    const rc = Launch.mul_mat_vec(g, 
                        e.dev,
                        self.d_xq,
                        pd,
                        @intCast(rows),
                        @intCast(cols),
                        1,
                        @intCast(@backingInt(q.ttype)),
                        self.launchStream(),
                    );
                    if (rc != cudart.cudaSuccess) {
                        if (self.capturing(g)) self.graphAbandon(g, "mul_mat_vec launch failed");
                        std.debug.print("cuda backend: mul_mat_vec({s}) launch failed: {s}\n", .{
                            @tagName(q.ttype), std.mem.span(cu.lib.errStr(rc)),
                        });
                        self.clearDevBuf(outBuf);
                        return;
                    }
                } else {
                    ensureBuf(&cu.lib, &self.d_out, &self.d_out_bytes, rows * @sizeOf(f32)) catch {
                        std.debug.print("cuda backend: matvec: device output allocation failed\n", .{});
                        self.clearDevBuf(outBuf);
                        return;
                    };
                    const rc = Launch.mul_mat_vec(g, 
                        e.dev,
                        self.d_xq,
                        @ptrCast(@alignCast(self.d_out)),
                        @intCast(rows),
                        @intCast(cols),
                        1,
                        @intCast(@backingInt(q.ttype)),
                        self.launchStream(),
                    );
                    if (rc != cudart.cudaSuccess) {
                        if (self.capturing(g)) self.graphAbandon(g, "mul_mat_vec launch failed");
                        std.debug.print("cuda backend: mul_mat_vec({s}) launch failed: {s}\n", .{
                            @tagName(q.ttype), std.mem.span(cu.lib.errStr(rc)),
                        });
                        self.clearDevBuf(outBuf);
                        return;
                    }
                    // 同上：`downloadOut` 会执行那次阻塞 D2H，而它本身就会放弃
                    // capture，所以这里不需要额外动作。
                }
                self.stat_ns_matvec += durNs(t0, self.io);
                self.stat_by_type[@backingInt(q.ttype)] += 1;
                self.stat_ns_by_type[@backingInt(q.ttype)] += durNs(t0, self.io);
                self.stat_n_matvec += 1;
                if (cudart.getenv("QWEN35_PROF") != null and self.stat_n_matvec % 500 == 0) {
                    self.reportStats("matvec");
                }
                if (cudart.getenv("QWEN35_PROF") != null and self.stat_n_matvec == 1) {
                    // `reportProfile` 在 CLI 路径上没有调用点，而引擎是 arena
                    // 分配的（所以 `deinit` 从不执行）。于是在第一次 GPU matvec 上
                    // 报一次设备信息：这是"kernel 真的跑了、没有回退到主机"的
                    // 可观测证据。
                    self.reportStats("first gpu matvec");
                }
                self.mvCheck(outBuf, q, xBuf);
            },
        }
        // `$QWEN35_SYNC`：每次 matvec 之后严格串行化。纯粹是用来回答一致性问题的
        // 诊断开关 —— 如果某个结果依赖了一个没同步的生产者，插入这道屏障会让
        // 那个 bug 确定性地出现或消失。正确性上并不需要：本文件的每次发射和每次
        // 拷贝都在同一条流上，
        // which orders them, and each `cudaMemcpy` is the synchronous form, so
        // it blocks until all prior work finished.
        if (self.sync_debug) _ = self.cu.?.lib.DeviceSynchronize();
    }

    /// `$QWEN35_MVCHECK`: **debug-only cross-check, not a computation path**.
    ///
    /// This is the only place in the file that calls the CPU matvec.  It is
    /// limited to the first matvec of a run, behind an env var, purely to compare
    /// device and host results (the class of bug in AGENTS.md §5.2: a kernel that
    /// is silently wrong for one quant type).  `outBuf` already holds the device
    /// result and is never overwritten; pulling it down for the comparison is
    /// the diagnostic's own cost (only with `QWEN35_MVCHECK` set).
    fn mvCheck(self: *CudaBackend, outBuf: backend.Buf, q: gguf.QuantRow, xBuf: backend.Buf) void {
        self.mvCheckVariant(outBuf, q, xBuf, false, 1);
    }

    /// `$QWEN35_MVCHECK` for the batch-K path: same comparison, CPU reference is
    /// `matvecBatchK` with the same K, so a per-token mismatch inside the chunk is
    /// visible instead of hidden behind token 0.
    fn mvCheckVariant(self: *CudaBackend, outBuf: backend.Buf, q: gguf.QuantRow, xBuf: backend.Buf, batchk: bool, k: usize) void {
        if (cudart.getenv("QWEN35_MVCHECK") == null) return;
        // `$QWEN35_MVCHECK_ONLY=iq1_s` restricts the check to one type: the
        // batch-K reference needs the CPU scratch shaped for `matvecBatchK`, and
        // not every type survives that in every launch order, so a targeted run
        // is the useful one when hunting a single kernel.
        if (cudart.getenv("QWEN35_MVCHECK_ONLY")) |only| {
            if (!std.mem.eql(u8, only, @tagName(q.ttype))) return;
        }
        if (self.mvcheck_seen[@backingInt(q.ttype)]) return;
        self.mvcheck_seen[@backingInt(q.ttype)] = true;

        // The CPU matvec needs its own activation scratch, and this backend never
        // calls `CpuBackend.allocScratch` (the device owns those buffers).  Hand it
        // a throwaway pair sized the way `backend_cpu.vAllocScratch` would, so the
        // reference run cannot assert on a zero-length scratch.
        const cols = q.cols();
        const cpu = &self.cpu.exec();
        cpu.allocScratch(cols) catch return;
        defer cpu.freeScratch();
        // `allocScratch` sizes one token; a batch-K launch needs K activation
        // slots, so grow them here (same `q8ScratchBytes` / `q8kScratchBytes`
        // arithmetic as `backend_cpu.vAllocScratch`, through the public sizes).
        if (batchk and k > 1) {
            self.cpu.q8K = self.alloc.alloc(u8, k * cpu_mod.q8ScratchBytes(cols)) catch return;
            self.cpu.q8kK = self.alloc.alloc(u8, k * cpu_mod.q8kScratchBytes(cols)) catch return;
        }
        // `dev` holds the device result, `cput` the CPU reference.
        const dev = self.alloc.alloc(f32, outBuf.len) catch return;
        defer self.alloc.free(dev);
        // The device result is pulled down for the comparison only: MVCHECK is a
        // debug switch, never part of the compute path (and it is what remains of
        // this file's D2H traffic when it is set).
        if (dBufPtr(outBuf)) |pd| {
            const rc = self.cu.?.lib.Memcpy(
                @ptrCast(dev.ptr),
                @ptrCast(pd),
                dev.len * @sizeOf(f32),
                cudart.MemcpyDeviceToHost,
            );
            if (rc != cudart.cudaSuccess) return;
            self.stat_d2h_bytes += dev.len * @sizeOf(f32);
        } else @memcpy(dev, outBuf.slice());

        // Reference activations: device-resident in the normal case, so they have
        // to come down too -- the CPU matvec can only read host memory.
        const x_dev = dBufPtr(xBuf) != null;
        const x = self.alloc.alloc(f32, xBuf.len) catch return;
        defer self.alloc.free(x);
        if (x_dev) {
            const rc = self.cu.?.lib.Memcpy(
                @ptrCast(x.ptr),
                @ptrCast(xBuf.ptr),
                x.len * @sizeOf(f32),
                cudart.MemcpyDeviceToHost,
            );
            if (rc != cudart.cudaSuccess) return;
            self.stat_d2h_bytes += x.len * @sizeOf(f32);
        } else @memcpy(x, xBuf.slice());

        const cput = self.alloc.alloc(f32, outBuf.len) catch return;
        defer self.alloc.free(cput);
        if (batchk) cpu.matvecBatchK(.{}, backend.Buf.wrap(cput), .{ .quant = q }, backend.Buf.wrap(x), k) else cpu.matvec(.{}, backend.Buf.wrap(cput), .{ .quant = q }, backend.Buf.wrap(x));
        // NOT `cpu.exec()` above: that is the diagnostic's OWN backend, whose
        // scratch is sized here, one token wide -- `matvecBatchK` wants K slots.
        // Those CPUs are never fully initialised (`setThreads` etc.), so this runs
        // serially (`pool = null`) instead.
        var bad: usize = 0;
        for (dev, 0..) |a, i| {
            // Relative tolerance, not bit equality: the device kernel may reorder
            // the accumulation, so an exact match is not expected even when
            // correct.
            const denom = @max(@abs(cput[i]), 1e-3);
            if (@abs(a - cput[i]) / denom > 2e-2) {
                if (bad < 3) std.debug.print("  MVCHECK row: dev={d} cpu={d}\n", .{ a, cput[i] });
                bad += 1;
            }
        }
        std.debug.print("MVCHECK[{s}] rows={d} cols={d} mismatched={d} batchk={} k={d}\n", .{
            @tagName(q.ttype), q.rows, q.cols(), bad, batchk, k,
        });
    }

    // ==== slot: matvecBatchK ================================================

    /// K prefetched tokens at once, all on the device.
    ///
    /// Activation layout matches the VTable contract (`out[j*rows + r]`) and the
    /// CPU reference: `xK` is `K` row-major token rows of `cols` floats, uploaded
    /// in one `cudaMemcpy`, quantized as `K` consecutive block groups in one buffer
    /// (`d_xq`, group `j` at `j * actBytes(ttype, cols)`), then a single launch
    /// computes `K*rows` outputs and one download returns them.
    fn vMatvecBatchK_body(g: backend.Graph, ctx: *anyopaque, outBuf: backend.Buf, w: backend.LinearW, xKBuf: backend.Buf, k: usize) void {
        const self = self_(ctx);
        if (!self.gpuReady()) {
            std.debug.print("cuda backend: matvecBatchK: {s}\n", .{self.needHelp()});
            self.clearDevBuf(outBuf);
            return;
        }
        const cu = &self.cu.?;
        const t0 = Io.Clock.Timestamp.now(self.io, .awake);

        // Which device output to write: the caller's own region (normal) or the
        // shared `d_out` when the caller handed us host memory.
        const out_dev: ?[*]f32 = dBufPtr(outBuf);

        switch (w) {
            .f32 => |m| {
                const e = self.findDev(mem.sliceAsBytes(m).ptr) orelse {
                    std.debug.print("cuda backend: matvecBatchK: f32 weight is not resident on the device\n", .{});
                    self.clearDevBuf(outBuf);
                    return;
                };
                const rows = e.rows;
                const cols = e.cols;
                std.debug.assert(xKBuf.len >= k * cols and outBuf.len >= k * rows);
                // `$QWEN35_PROF`: same device-side clock as the quant path.  This
                // branch serves `.f32` weights only; f16/bf16 take the w16 branch
                // below, so a model whose linear weights are 2-byte (BF16/F16) or
                // packed never reaches it.
                if (self.gpuTimingOn()) self.batchKTimingBegin();
                const act_bytes = k * cols * @sizeOf(f32);
                const d_x: [*]const f32 = blk: {
                    if (dBufPtr(xKBuf)) |p| break :blk @ptrCast(p);
                    ensureBuf(&cu.lib, &self.d_act, &self.d_act_bytes, act_bytes) catch return oomMsg("batchK activation");
                    const rc0 = cu.lib.Memcpy(self.d_act, xKBuf.slice().ptr, act_bytes, cudart.MemcpyHostToDevice);
                    if (rc0 != cudart.cudaSuccess) {
                        std.debug.print("cuda backend: batchK f32 activation upload failed: {s}\n", .{std.mem.span(cu.lib.errStr(rc0))});
                        self.clearDevBuf(outBuf);
                        return;
                    }
                    self.stat_h2d_bytes += act_bytes;
                    break :blk @ptrCast(@alignCast(self.d_act.?));
                };
                var d_out: [*]f32 = undefined;
                if (out_dev) |p| {
                    d_out = p;
                } else {
                    ensureBuf(&cu.lib, &self.d_out, &self.d_out_bytes, k * rows * @sizeOf(f32)) catch return oomMsg("batchK output");
                    d_out = @ptrCast(@alignCast(self.d_out.?));
                }
                const rc = Launch.mul_mat_vec_f32(g, 
                    @ptrCast(@alignCast(e.dev)),
                    d_x,
                    d_out,
                    @intCast(rows),
                    @intCast(cols),
                    @intCast(k),
                    null,
                );
                if (rc != cudart.cudaSuccess) {
                    std.debug.print("cuda backend: mul_mat_vec_f32(K={d}) launch failed: {s}\n", .{
                        k, std.mem.span(cu.lib.errStr(rc)),
                    });
                    self.clearDevBuf(outBuf);
                    return;
                }
                if (out_dev == null) self.downloadOut(outBuf.slice()[0 .. k * rows]);
                self.stat_ns_batchk += durNs(t0, self.io);
                self.stat_n_batchk += 1;
                if (self.gpuTimingOn()) self.batchKTimingEnd();
                return;
            },
            .quant => |q| {
                const e = self.findDev(q.data.ptr) orelse {
                    std.debug.print("cuda backend: matvecBatchK: a '{s}' weight is not resident on the device\n", .{@tagName(q.ttype)});
                    self.clearDevBuf(outBuf);
                    return;
                };
                const cols = q.cols();
                const rows = q.rows;
                const per_tok = actBytes(q.ttype, cols);
                const nb_tok = cols / actBlockElems(q.ttype);
                std.debug.assert(xKBuf.len >= k * cols and outBuf.len >= k * rows);
                // `$QWEN35_PROF`: start the device-side clock for this call.
                // Must run before the H2D below, so a synchronous upload is
                // counted as host slack rather than as GPU work; the recorded
                // point is ordered after any earlier queued kernel anyway.
                if (self.gpuTimingOn()) {
                    // Attribute this call's device interval to its type; see
                    // `stat_ns_bk_by_type`.  Set before the event so a failure
                    // in between still names the type that was issued.
                    self.bk_ttype = @intCast(@backingInt(q.ttype));
                    self.batchKTimingBegin();
                }

                // Activation source: the resident K-row buffer when the engine
                // already keeps it on the device (the normal case after P2d),
                // otherwise one H2D for all K rows.
                const base_act: [*]const f32 = blk: {
                    if (dBufPtr(xKBuf)) |p| break :blk @ptrCast(p);
                    const act_bytes = k * cols * @sizeOf(f32);
                    ensureBuf(&cu.lib, &self.d_act, &self.d_act_bytes, act_bytes) catch return oomMsg("batchK activation");
                    const rc0 = cu.lib.Memcpy(self.d_act, xKBuf.slice().ptr, act_bytes, cudart.MemcpyHostToDevice);
                    if (rc0 != cudart.cudaSuccess) {
                        std.debug.print("cuda backend: batchK activation upload failed: {s}\n", .{std.mem.span(cu.lib.errStr(rc0))});
                        self.clearDevBuf(outBuf);
                        return;
                    }
                    self.stat_h2d_bytes += act_bytes;
                    break :blk @ptrCast(@alignCast(self.d_act.?));
                };
                // f16/bf16: no activation quantization at all.  The K f32 rows
                // are already resident (`base_act`), the kernel widens the
                // 2-byte weights against them, and `d_xq` is never touched --
                // so this branch skips the quantize launch and its buffer
                // entirely.
                if (q.ttype == .f16 or q.ttype == .bf16) {
                    var d_out_w: [*]f32 = undefined;
                    if (out_dev) |p| {
                        d_out_w = p;
                    } else {
                        ensureBuf(&cu.lib, &self.d_out, &self.d_out_bytes, k * rows * @sizeOf(f32)) catch return oomMsg("batchK output");
                        d_out_w = @ptrCast(@alignCast(self.d_out.?));
                    }
                    const rcw = Launch.mul_mat_vec_w16(g, 
                        e.dev,
                        base_act,
                        d_out_w,
                        @intCast(rows),
                        @intCast(cols),
                        @intCast(k),
                        @intCast(@backingInt(q.ttype)),
                        self.launchStream(),
                    );
                    if (rcw != cudart.cudaSuccess) {
                        std.debug.print("cuda backend: mul_mat_vec({s}, K={d}) launch failed: {s}\n", .{
                            @tagName(q.ttype), k, std.mem.span(cu.lib.errStr(rcw)),
                        });
                        self.clearDevBuf(outBuf);
                        return;
                    }
                    if (out_dev == null) self.downloadOut(outBuf.slice()[0 .. k * rows]);
                    self.stat_ns_batchk += durNs(t0, self.io);
                    self.stat_n_batchk += 1;
                    if (self.gpuTimingOn()) self.batchKTimingEnd();
                    return;
                }

                ensureBuf(&cu.lib, &self.d_xq, &self.d_xq_bytes, per_tok * k) catch return oomMsg("batchK quantized activation");
                var d_out: [*]f32 = undefined;
                if (out_dev) |p| {
                    d_out = p;
                } else {
                    ensureBuf(&cu.lib, &self.d_out, &self.d_out_bytes, k * rows * @sizeOf(f32)) catch return oomMsg("batchK output");
                    d_out = @ptrCast(@alignCast(self.d_out.?));
                }

                // Quantize ALL k tokens in ONE launch.
                //
                // The k activation rows are contiguous in `base_act` (k*cols
                // f32) and the destination slots are contiguous in `d_xq`
                // (k*per_tok bytes, `per_tok == nb_tok * actBlockBytes`), so
                // the whole batch is one run of `k * nb_tok` blocks with the
                // SAME stride the quantizers already use -- no per-token offset
                // arithmetic on either side.  Both device quantizers
                // (q35_k_quantize_q8_K in mmvq.cu, q35_k_quantize_q8_0 in
                // convert.cu) are one-thread-per-block with `blockIdx`-derived
                // indices and read nothing outside `[0, nblocks)`, so this is
                // bit-exact vs. the old one-launch-per-token loop.
                const base_xq: [*]u8 = @ptrCast(self.d_xq.?);
                const q8_0_act = actBlockElems(q.ttype) == gguf.blockElems(.q8_0);
                const kern = Launch;
                // `k >= 2` on this path (the caller routes k == 1 to
                // `forwardLogits`, which is a separate shape) and `cols >= 1`
                // with cols a multiple of actBlockElems, so `total_blocks >= 2`:
                // the merged count can never be 0 and can never be smaller than
                // the per-token one.
                const total_blocks = nb_tok * k;
                std.debug.assert(k >= 2 and nb_tok >= 1);
                // The dp4a kernel reads `block_q8_1_mmq`, not q8_K, and used to
                // re-transpose on every launch.  When it is enabled for this
                // type the quantizer emits that reshape in the SAME pass (one
                // extra store per value, no extra f32 load), and the kernel does
                // a contiguous tile copy -- which is what llama.cpp does by
                // having quantize_mmq_q8_1 write the layout directly.  q8_K is
                // still produced because other types in the same chunk and the
                // decode path consume it.
                const wants_mmq = !q8_0_act and Launch.mmq_dp4a_wants(@intCast(@backingInt(q.ttype))) != 0;
                if (wants_mmq) {
                    ensureBuf(&cu.lib, &self.d_xqm, &self.d_xqm_bytes, @as(usize, @intCast(total_blocks)) * @as(usize, @intCast(Launch.mmq_act_block_bytes()))) catch return oomMsg("batchK mmq activations");
                }
                const rc = if (q8_0_act)
                    kern.quantize_q8_0(base_act, @ptrCast(base_xq), @intCast(total_blocks), null)
                else if (wants_mmq)
                    kern.quantize_q8_k_mmq(base_act, @ptrCast(base_xq), self.d_xqm, @intCast(total_blocks), null)
                else
                    kern.quantize_q8_k(base_act, @ptrCast(base_xq), @intCast(total_blocks), null);
                if (rc != cudart.cudaSuccess) {
                    // No single token index to blame any more: report the whole
                    // quantize that failed (all k rows, nb_tok blocks each).
                    std.debug.print("cuda backend: batchK quantize({s}) failed for k={d} tok x nblocks={d} (total {d} blocks): {s}\n", .{
                        @tagName(q.ttype), k, nb_tok, total_blocks, std.mem.span(cu.lib.errStr(rc)),
                    });
                    self.clearDevBuf(outBuf);
                    return;
                }
                // Diagnostics only.  This used to be +1 per token (K per call);
                // it is now +1 for the merged quantize, so one matvecBatchK call
                // costs 2 launches (quantize + mul_mat_vec) -- see the
                // `bk_launch` field on the `CUDA-M` line.
                self.stat_n_launch_bk += 1;

                const kern_mm = Launch;
                // New binding (shadowing the quantize's `rc` above): the old
                // per-token loop scoped that one to the loop body, so merging
                // it out would otherwise redeclare `rc` in this block.
                // The dp4a kernel takes the mmq layout, everything else q8_K.
                // `d_xqm` is non-null exactly when `wants_mmq` was set above (it
                // is `ensureBuf`'d right there), and `q35_mmq_dp4a_wants` is the
                // same predicate the kernel uses to pick its branch, so the two
                // cannot disagree.
                const rc_mm = kern_mm.mul_mat_vec(g,
                    e.dev,
                    if (wants_mmq) self.d_xqm else self.d_xq,
                    d_out,
                    @intCast(rows),
                    @intCast(cols),
                    @intCast(k),
                    @intCast(@backingInt(q.ttype)),
                    null,
                );
                if (rc_mm != cudart.cudaSuccess) {
                    std.debug.print("cuda backend: mul_mat_vec({s}, K={d}) launch failed: {s}\n", .{
                        @tagName(q.ttype), k, std.mem.span(cu.lib.errStr(rc_mm)),
                    });
                    self.clearDevBuf(outBuf);
                    return;
                }
                if (out_dev == null) self.downloadOut(outBuf.slice()[0 .. k * rows]);
                self.stat_ns_batchk += durNs(t0, self.io);
                self.stat_by_type[@backingInt(q.ttype)] += 1;
                self.stat_ns_by_type[@backingInt(q.ttype)] += durNs(t0, self.io);
                self.stat_n_batchk += 1;
                // The matmul itself: with the merged quantize above this makes
                // the per-call launch count 2, independent of k.
                self.stat_n_launch_bk += 1;
                // `$QWEN35_PROF`: GPU-side time of what this call just issued
                // (quantize TK + mul_mat_vec).  The drain is inside `evElapsedNs`.
                if (self.gpuTimingOn()) self.batchKTimingEnd();
                // Interim dump: the CLI path never runs `deinit` (the engine is
                // arena-allocated), so without a periodic report the only profile
                // line is the one printed on the first matvec.  Every 200th chunk
                // keeps generation-phase numbers visible without spamming.
                if (cudart.getenv("QWEN35_PROF") != null and self.stat_n_batchk % 200 == 0) {
                    self.reportStats("batchk");
                }
                // Token 0 of the chunk is enough: all K rows run through the same
                // kernel, so a per-type mismatch shows up on the first one.
                self.mvCheckVariant(outBuf, q, xKBuf, true, k);
            },
        }
    }

    // ==== slots: device scratch =============================================

    /// Reserve device scratch for one token.
    ///
    /// Sizes follow `backend_cpu.q8ScratchBytes` / `q8kScratchBytes` exactly (34 B
    /// per 32 elements, 292 B per 256) so a config the CPU backend accepted cannot
    /// overflow here.  This is also where a failed device init becomes a load
    /// error: `init()` cannot report one (the factory contract is
    /// `init(alloc, g) -> Backend`), and quietly running the whole model on the CPU
    /// after `--backend cuda` would be a lie.
    fn dAllocScratch_body(ctx: *anyopaque, max_cols: usize) anyerror!void {
        const self = self_(ctx);
        if (!self.gpuReady()) return error.BackendUnavailable;
        try self.uploadIqTables();
        const cu = &self.cu.?;
        try ensureBuf(&cu.lib, &self.d_act, &self.d_act_bytes, max_cols * @sizeOf(f32));
        // Both activation formats up front: which one a layer needs is only known at
        // matvec time, and a mid-model allocation is what this call exists to avoid.
        try ensureBuf(&cu.lib, &self.d_xq, &self.d_xq_bytes, actBytes(.q8_k, max_cols));
        try ensureBuf(&cu.lib, &self.d_out, &self.d_out_bytes, max_cols * @sizeOf(f32));
        try ensureBuf(&cu.lib, &self.d_deq, &self.d_deq_bytes, max_cols * @sizeOf(f32));
        // `d_idx` (the argmax result cell) is 4 bytes and was allocated lazily on
        // the first argmax.  It must not be allocated *during* a captured step,
        // so it is reserved here with everything else.
        try ensureBuf(&cu.lib, &self.d_idx, &self.d_idx_bytes, @sizeOf(c_int));
        // PV partials for the split attention path.  Sized for the widest head
        // count the config can produce at the largest split, allocated here so
        // the split path never allocates inside a capture.
        self.attn_split = attnSplitFromEnv();
        if (self.attn_split > 0) {
            try ensureBuf(&cu.lib, &self.d_att_part, &self.d_att_part_bytes, 64 * self.attn_split * 512 * @sizeOf(f32));
            // `heads * Q35_ATTN_MAX_ROWS` floats; 64 heads x 8192 = 2 MB.
            try ensureBuf(&cu.lib, &self.d_att_prob, &self.d_att_prob_bytes, 64 * 8192 * @sizeOf(f32));
        }
        // Same reasoning for the device `slot` cell that `attn_decode` reads:
        // allocating it on first use would land inside a capture, where
        // `cudaMalloc` is illegal ("operation not permitted when stream is
        // capturing").  Reserved here so the dyn path is always ready.
        _ = self.ensureCtl();
    }

    /// K-token variant: the same buffers scaled by `k`.  `d_out`/`d_xq` dominate, so
    /// a wide prefill chunk shows up in the `CUDA-M` buffer total.
    fn dAllocScratchK_body(ctx: *anyopaque, max_cols: usize, k: usize) anyerror!void {
        const self = self_(ctx);
        if (!self.gpuReady()) return error.BackendUnavailable;
        try self.uploadIqTables();
        const cu = &self.cu.?;
        try ensureBuf(&cu.lib, &self.d_act, &self.d_act_bytes, k * max_cols * @sizeOf(f32));
        try ensureBuf(&cu.lib, &self.d_xq, &self.d_xq_bytes, k * actBytes(.q8_k, max_cols));
        try ensureBuf(&cu.lib, &self.d_out, &self.d_out_bytes, k * max_cols * @sizeOf(f32));
        // The decode-graph path runs after prefill, so THIS call is where the
        // scratch reaches its final size.  Re-assert the reservation for the
        // non-K sizes too: a decode step needs `max_cols`-sized `d_deq`/`d_idx`,
        // which `allocScratch` already provided, but making the requirement
        // explicit here means a future caller that only ever calls
        // `allocScratchK` still cannot allocate inside a capture.
        try self.graphReserve(max_cols);
    }

    /// Upload the twelve IQ codebooks to the device, at most once per process.
    ///
    /// WHY HERE: the tables are read by name through `__device__` pointers that
    /// only `q35_launch_init_tables` fills in (src/kernels/cuda/iq_tables.cu).
    /// A kernel that runs before the upload indexes address 0 -- wrong numbers,
    /// no crash, and the failure looks like a bad quant kernel -- so the upload
    /// has to be provably earlier than the first matvec/dequantRow.  Both
    /// scratch-allocation hooks are the place: `Engine.load` calls
    /// `exec.allocScratch(cols)` while opening the model, before it has any
    /// weights resident or computes anything (src/engine.zig, `load`), and
    /// `allocScratchK` only ever follows it.  Making this the first statement
    /// after the `gpuReady` check means no device buffer -- IQ ones included --
    /// exists before the codebooks do.
    ///
    /// Cached through `iq_tables_state` rather than re-called every time: the
    /// C side is idempotent too, but this way a failed upload is reported once
    /// instead of on every later matvec.
    fn uploadIqTables(self: *CudaBackend) anyerror!void {
        switch (self.iq_tables_state) {
            .done => return,
            // A previous attempt failed.  Do not retry: the reason was already
            // recorded in `init_error` and repeated failures of the same kind
            // would only spam the log.  Report the same error to the caller.
            .failed => return error.IqTablesNotUploaded,
            .pending => {},
        }
        // `gpuReady` is checked by both callers before they get here, so the
        // runtime handle exists; no separate "kernels loaded" guard is needed
        // because they are link-time symbols.
        const rc = Launch.init_tables(null);
        if (rc != cudart.cudaSuccess) {
            std.debug.print(
                "error: cuda backend: IQ codebook upload failed (cudaError {d}: {s})\n" ++
                    "       IQ quant types cannot run on the GPU without it; use --backend cpu\n",
                .{ rc, std.mem.span(self.cu.?.lib.GetErrorName(rc)) },
            );
            self.iq_tables_state = .failed;
            return error.IqTablesNotUploaded;
        }
        self.iq_tables_state = .done;
    }

    /// `allocBuf` for the device backend: a `cudaMalloc` region, so every
    /// `backend.Buf` the engine holds (activations, KV cache, GDN state) is
    /// DEVICE memory and the slots can hand `buf.ptr` straight to a kernel.
    ///
    /// This is the P2c change: P2b forwarded to the embedded CPU backend, which
    /// returned host memory and forced every slot to bounce its operands
    /// through staging.  The list of handed-out regions lives here (`dev_bufs`)
    /// because `freeScratch` must release them all at once and `backend.Buf`
    /// carries no ownership flag.
    ///
    /// Failure is a hard error, not a fallback: the engine's caller has no
    /// hybrid path any more (device memory is where the data must live), so a
    /// failed `cudaMalloc` has to surface as a load/allocation failure.
    fn dAllocBuf_body(ctx: *anyopaque, len: usize) anyerror!backend.Buf {
        const self = self_(ctx);
        if (len == 0) return backend.Buf.empty;
        const cu = if (self.cu) |*c| c else {
            std.debug.print("cuda backend: allocBuf({d}): {s}\n", .{ len, self.needHelp() });
            return error.DeviceUnavailable;
        };
        const bytes = len * @sizeOf(f32);
        var p: ?*anyopaque = null;
        const rc = cu.lib.Malloc(&p, bytes);
        if (rc != cudart.cudaSuccess or p == null) {
            std.debug.print("cuda backend: allocBuf({d} floats = {d} B) failed: {s}\n", .{
                len, bytes, std.mem.span(cu.lib.errStr(rc)),
            });
            return error.OutOfMemory;
        }
        // A `Buf` is a `[]f32`-shaped view; the pointer is reused as-is, which
        // is what every slot then casts back to a device pointer.
        const slice: []f32 = @as([*]f32, @ptrCast(@alignCast(p.?)))[0..len];
        self.dev_bufs.append(self.alloc, .{ .ptr = p.?, .len = len }) catch {
            _ = cu.lib.Free(p.?);
            return error.OutOfMemory;
        };
        return backend.Buf.wrap(slice);
    }

    /// `VTable.setKvType`: one format per model load, set before the first
    /// `allocKv`.  Mirrored into the embedded CPU fallback so the CPU net path
    /// reads the cache in the same layout.
    fn vSetKvType(ctx: *anyopaque, t: backend.KvType) void {
        const self = self_(ctx);
        self.kv_type = t;
        self.cpu.setKvType(t);
    }

    /// `VTable.allocKv`: one layer's K and V, `max_seq` rows of `kv_dim`
    /// elements in `kv_type`.  Each region is a `cudaMalloc` of the f32-aligned
    /// size (`4 * ceil(bytes/4)`) so the returned `[]f32` view is exactly the
    /// allocation; the bytes are the cache in `kv_type` and only this backend
    /// interprets them.  Both regions go into `dev_bufs` so `freeScratch`
    /// releases them (a `Buf` carries no ownership flag).
    fn vAllocKv(ctx: *anyopaque, kv_dim: usize, max_seq: usize) anyerror!backend.KvRegion {
        const self = self_(ctx);
        const cu = if (self.cu) |*c| c else {
            std.debug.print("cuda backend: allocKv: {s}\n", .{self.needHelp()});
            return error.DeviceUnavailable;
        };
        const row_bytes = backend.kvRowBytes(kv_dim, self.kv_type);
        const bytes = row_bytes * max_seq;
        const flen = (bytes + 3) / 4;
        const total = 4 * flen;
        var kp: ?*anyopaque = null;
        const rc1 = cu.lib.Malloc(&kp, total);
        if (rc1 != cudart.cudaSuccess or kp == null) {
            std.debug.print("cuda backend: allocKv k region ({d} B) failed: {s}\n", .{ total, std.mem.span(cu.lib.errStr(rc1)) });
            return error.OutOfMemory;
        }
        var vp: ?*anyopaque = null;
        const rc2 = cu.lib.Malloc(&vp, total);
        if (rc2 != cudart.cudaSuccess or vp == null) {
            _ = cu.lib.Free(kp.?);
            std.debug.print("cuda backend: allocKv v region ({d} B) failed: {s}\n", .{ total, std.mem.span(cu.lib.errStr(rc2)) });
            return error.OutOfMemory;
        }
        const kview: []f32 = @as([*]f32, @ptrCast(@alignCast(kp.?)))[0..flen];
        const vview: []f32 = @as([*]f32, @ptrCast(@alignCast(vp.?)))[0..flen];
        self.dev_bufs.append(self.alloc, .{ .ptr = kp.?, .len = flen }) catch {
            _ = cu.lib.Free(kp.?);
            _ = cu.lib.Free(vp.?);
            return error.OutOfMemory;
        };
        self.dev_bufs.append(self.alloc, .{ .ptr = vp.?, .len = flen }) catch {
            _ = self.dev_bufs.pop();
            _ = cu.lib.Free(kp.?);
            _ = cu.lib.Free(vp.?);
            return error.OutOfMemory;
        };
        return .{ .k = backend.Buf.wrap(kview), .v = backend.Buf.wrap(vview) };
    }

    /// `VTable.kvAppend`: quantize `rows` f32 rows (in `src`, `kv_dim` wide,
    /// row-major) into the cache `dst` starting at row `slot`.
    ///
    /// f32 is the historical `copyRange`/`copyRangeSlot` path, byte-identical
    /// and graph-identical (the same `copy_range` node the decode graph
    /// retargets).  A quantized format launches `q35_k_kv_quant` (kind
    /// 0=f16/1=q8_0/2=q4_0); while a capture is open the destination address is
    /// recorded so the graph retargets it one row per token exactly like the
    /// f32 append.  The CPU path is the correctness net for a broken/missing
    /// device (mirrors `dCopyRange`), never the normal route.
    fn dKvAppend_body(g: backend.Graph, ctx: *anyopaque, dst: backend.Buf, src: backend.Buf, slot: usize, kv_dim: usize, rows: usize) void {
        const self = self_(ctx);
        if (graphGate(g)) return;
        if (rows == 0) return;
        // f32: the historical path, unchanged (and graph-identical) -- except
        // that a replayed/recorded step takes its ROW from the control block so
        // the captured destination pointer stays a constant region base.
        if (self.kv_type == .f32) {
            if (self.graphActive(g) and self.ctl_dev != null) {
                if (dBufPtr(dst)) |pd| {
                    if (dBufPtr(src)) |ps| {
                        const n = rows * kv_dim;
                        const rc0 = Launch.copy_range_dslot(pd, @intCast(kv_dim), ps,
                            @intCast(rows), @intCast(n),
                            @ptrCast(@alignCast(self.ctl_dev.?)), self.launchStream());
                        if (rc0 == cudart.cudaSuccess) return;
                        if (self.capturing(g)) self.graphAbandon(g, "copy_range_dslot launch failed");
                        self.ewLaunchFail("copy_range_dslot", rc0);
                    }
                }
            }
            if (rows == 1) {
                CudaBackend.dCopyRangeSlot_body(g, ctx, dst.span(slot * kv_dim, kv_dim), src, slot);
            } else {
                CudaBackend.dCopyRange_body(g, ctx, dst.span(slot * kv_dim, rows * kv_dim), src);
            }
            return;
        }
        const kind: c_int = switch (self.kv_type) {
            .f16 => 0,
            .q8_0 => 1,
            else => 2,
        };
        const row_bytes = backend.kvRowBytes(kv_dim, self.kv_type);
        const nblocks = rows * (kv_dim / 32);
        if (self.cu == null or kv_dim % 32 != 0) {
            cpu_mod.CpuBackend.vtable.kvAppend(.{}, @ptrCast(&self.cpu), dst, src, slot, kv_dim, rows);
            return;
        }
        // `dst` 是这一层的**整块**区域（f32 视图）；第 `slot` 行的起始位置是
        // 基址之后 `slot * row_bytes` 字节。
        const dst_bytes: [*]u8 = @ptrCast(@alignCast(dst.ptr));
        const out_ptr: [*]u8 = dst_bytes + slot * row_bytes;
        // 行号一律从**设备控制块**读（`XD_CTL_KV_SLOT`），capture 与 replay 都走
        // 这条路，且覆盖全部量化 KV 类型（f16 / q8_0 / q4_0）。节点实参固定、
        // 控制块每步由主机刷新，所以录进图的 `k_kv_quant_dslot` 在重放时读到的
        // 是**当步**的行号。
        //
        // 两条历史教训：
        //   * 旧的 `out_ptr = base + slot * row_bytes` 写法只在 `state == .live`
        //     时才切到设备行号，capture 期把录制那一步的指针固化进图 —— 实测：
        //     每次重放都往 KV[17] 覆盖写，attn 读 0..pos 拿到旧行，第二次重放
        //     起 logits 全错。
        //   * 只让 f16 走这条路也不够：q8_0/q4_0 会落回下面的
        //     `kv_quant(..., out_ptr, ...)`，而 `out_ptr` 是主机侧算出的行地址，
        //     捕获后永远是录制那一步的行（实测：`--kv-type q8_0` 时 XG 与直接
        //     发射从第十几个 token 起分叉，而 f32/f16 已经逐位一致）。
        // `rows == 1` 曾是这里的门槛：kernel 其实**一直**支持多行
        // （`slot = ctl[KV_SLOT] + row`），只是旧代码只把它接到单行上。放开到
        // 任意 `rows` 之后，k 行前向（verify）也不必再走下面的主机侧
        // `out_ptr` —— 那条路径把"基址 + 录制那一行的偏移"烘进图里，重放时
        // 每一轮都往同一段 KV 覆盖写。实测（verify、k=2）：两行 argmax 完全
        // 相同、anchor 不推进、MTP 接受率 60% → 0%。
        if (self.graphActive(g) and self.ctl_dev != null) {
            const nbp = @as(c_int, @intCast(kv_dim / 32));
            const rc0 = Launch.kv_quant_dslot(g, src.ptr, @ptrCast(dst_bytes),
                @ptrCast(@alignCast(self.ctl_dev.?)), nbp, @intCast(rows),
                @intCast(row_bytes), kind, self.launchStream());
            if (rc0 == cudart.cudaSuccess) return;
            self.ewLaunchFail("kv_quant_dslot", rc0);
        }
        const rc = Launch.kv_quant(g, src.ptr, @ptrCast(out_ptr), @intCast(nblocks), kind, self.launchStream());
        if (rc == cudart.cudaSuccess) return;
        if (self.capturing(g)) self.graphAbandon(g, "kv_quant launch failed");
        self.ewLaunchFail("kv_quant", rc);
        cpu_mod.CpuBackend.vtable.kvAppend(.{}, @ptrCast(&self.cpu), dst, src, slot, kv_dim, rows);
    }

    /// Release every region `dAllocBuf` handed out, plus the reusable scratch
    /// (`d_act`/`d_xq`/`d_out`/`d_deq`) and the resident constant copies.
    ///
    /// Called once, by the engine at teardown (`exec.freeScratch()`), never per
    /// token.  `dev_bufs` is sorted-free because the order does not matter; the
    /// `Buf` values the engine still holds become dangling here, which is why
    /// the engine must not touch them afterwards (it does not: teardown follows).
    fn dFreeScratch(ctx: *anyopaque) void {
        const self = self_(ctx);
        const cu = if (self.cu) |*c| c else return;
        for (self.dev_bufs.items) |r| _ = cu.lib.Free(r.ptr);
        self.dev_bufs.clearRetainingCapacity();
        if (self.d_act) |p| _ = cu.lib.Free(p);
        if (self.d_xq) |p| _ = cu.lib.Free(p);
        if (self.d_xqm) |p| _ = cu.lib.Free(p);
        if (self.d_out) |p| _ = cu.lib.Free(p);
        if (self.d_deq) |p| _ = cu.lib.Free(p);
        self.d_act = null;
        self.d_act_bytes = 0;
        self.d_xq = null;
        self.d_xq_bytes = 0;
        self.d_xqm = null;
        self.d_xqm_bytes = 0;
        self.d_out = null;
        self.d_out_bytes = 0;
        self.d_deq = null;
        self.d_deq_bytes = 0;
    }

    // ==== slots: pure elementwise (P2b) =====================================
    //
    // scaleVec / addResid / copyRange / zeroBuf are the first kernels to move
    // off the CPU.  None of them reduces, so each element is one IEEE single
    // operation and the GPU result must be BIT-EXACT with the CPU one (that is
    // what keeps the token stream identical); see
    // src/kernels/cuda/elementwise.cu.
    //
    // P2c: ACTIVATIONS ARE DEVICE MEMORY NOW.  `dAllocBuf` hands out
    // `cudaMalloc` regions, so a `backend.Buf` is already a device pointer and
    // every slot below passes `buf.ptr` straight to the launcher.  The P2b
    // staging triple and its `d_stage` slots are gone: there is exactly one
    // legal host->device path left in this file,
    // `vLoadVecF32` (read-only 1-D weight vectors, uploaded once at load time),
    // and no device->host path on the per-token execution at all -- the only D2H
    // left is `downloadOut` for
    // matvec results, which is the values the sampler needs.
    //
    // `dBufPtr` below is the whole replacement: it is `ensureGpu` + a pointer
    // cast, kept as one helper so the "is the device up, is this buffer empty"
    // check cannot drift between slots.

    /// Device pointer for a resident `backend.Buf`, or null for an empty buffer.
    ///
    /// No copy, no allocation, no ownership: after P2c the `Buf` IS the device
    /// region.  Null means "nothing to do" -- `Buf.empty()` is a legal value
    /// (`engine.zig` uses it for absent optional state) and the launchers all
    /// treat `n <= 0` as a no-op, so the null return simply skips the launch.
    ///
    /// The CPU fallback in the slots that call this is now a *correctness
    /// net*, not a hybrid path: if the runtime is down the buffers handed out by
    /// `dAllocBuf` would not exist either, so reaching it means something is
    /// already broken and the CPU path is the honest answer.
    fn dBufPtr(buf: backend.Buf) ?[*]f32 {
        if (buf.len == 0) return null;
        return @ptrCast(@alignCast(buf.ptr));
    }

    /// Zero a device `Buf` without a host round trip (the error path of every
    /// numeric slot: an empty result is the honest failure signal, and the CPU
    /// fallback of `elementwise` cannot be used because the buffer is device
    /// memory).  Falls back to the CPU memset only when the pointer is not a
    /// device region at all, i.e. the caller really handed us host memory.
    fn clearDevBuf(self: *CudaBackend, buf: backend.Buf) void {
        if (buf.len == 0) return;
        if (self.cu) |*c| {
            if (dBufPtr(buf)) |p| {
                if (c.lib.Memset(@ptrCast(p), 0, buf.len * @sizeOf(f32)) == cudart.cudaSuccess) return;
            }
        }
        @memset(buf.slice(), 0);
    }

    /// Failure text for a launch that returned a cudaError_t.
    fn ewLaunchFail(self: *CudaBackend, what: []const u8, rc: c_int) void {
        const cu = self.cu orelse return;
        std.debug.print("cuda backend: {s} launch failed: {s}\n", .{
            what, std.mem.span(cu.lib.errStr(rc)),
        });
    }

    // The four slots.  Each is a FORWARDER rather than an alias to
    // `cpu_mod.CpuBackend.vtable.*`, for the same reason `dAllocBuf` is: `ctx`
    // is a `*CudaBackend` and these need it (device scratch + stream).  Every
    // one of them degrades to the embedded CPU backend on any device-side
    // failure, so a broken/missing GPU still produces correct numbers instead of
    // silently leaving the buffer untouched -- none of these slots can report an
    // error through the `Exec` signature, and panicking the process would be a
    // worse outcome than a slow correct answer.

    /// Whether `$QWEN35_L2QK_CHECK` has already run (it is a first-call-only
    /// probe, so it stays off the hot path).
    var l2qk_checked: bool = false;
    /// A CPU backend for the probe only; the check runs once, at init-ish time,
    /// and `l2normQkScale`'s CPU implementation touches none of these fields
    /// (it is three elementwise/rms calls on the buffer it is handed), so the
    /// placeholder allocator and model handle are never dereferenced.
    var l2qk_cpu: cpu_mod.CpuBackend = .{ .alloc = std.heap.page_allocator, .g = undefined };

    /// The eps the GDN L2 normalisation uses.  Must match the `1e-6` that
    /// `dL2normRows_body` hardcodes (and `engine.l2norm`), because the fused
    /// kernel is claimed to be bit-identical to that three-call sequence.
    const gdn_l2_eps: f32 = 1e-6;

    fn dL2normQkScale_body(g: backend.Graph, ctx: *anyopaque, x: backend.Buf, nk: usize, kd: usize, k_offset: usize, qscale: f32) void {
        const self = self_(ctx);
        const nb = nk * kd;
        if (nk != 0 and kd != 0 and k_offset >= nb and x.len >= k_offset + nb) {
            if (self.cu != null) {
                if (dBufPtr(x)) |px| {
                    // `$QWEN35_L2QK_CHECK=1`: on the FIRST call only, run the
                    // three-call CPU composition on a copy of the same input and
                    // compare element-wise against the fused kernel's output.
                    // This is the kernel-level cross-check the fused path needs
                    // and MVCHECK cannot give (it only covers quantized matvec).
                    // First-call-only keeps it off the hot path -- the check
                    // itself needs a blocking D2H and a host round trip.
                    if (!l2qk_checked) {
                        if (cudart.getenv("QWEN35_L2QK_CHECK")) |v| {
                            if (std.mem.eql(u8, v, "1")) {
                                l2qk_checked = true;
                                if (self.cu) |*cu| l2qkCheck(&cu.lib, px, nk, kd, k_offset, qscale);
                            }
                        }
                    }
                    const rc = Launch.l2norm_qk_scale(g, px, @intCast(nk), @intCast(kd), @intCast(k_offset), gdn_l2_eps, qscale, self.launchStream());
                    if (rc == cudart.cudaSuccess) return;
                    self.ewLaunchFail("l2norm_qk_scale", rc);
                } else return;
            }
        }
        cpu_mod.CpuBackend.vtable.l2normQkScale(.{}, @ptrCast(&self.cpu), x, nk, kd, k_offset, qscale);
    }

    /// `$QWEN35_L2QK_CHECK` support: copy the device input, run the fused
    /// kernel into one copy and the CPU three-call composition into the other,
    /// and report the worst per-element error.  The two are claimed to be
    /// bit-identical (each row's sum of squares is one block over that row, the
    /// same `1/sqrt(ss+eps)`, and `scaleVec` reads the row the norm just wrote),
    /// so anything above f32 rounding noise means the fused kernel drifted.
    fn l2qkCheck(lib: *cudart.Lib, px: [*]f32, nk: usize, kd: usize, k_offset: usize, qscale: f32) void {
        const nb = nk * kd;
        const total = k_offset + nb;
        const bytes = total * 4;
        const host = std.heap.page_allocator.alloc(f32, total) catch return;
        defer std.heap.page_allocator.free(host);
        const dev = std.heap.page_allocator.alloc(f32, total) catch return;
        defer std.heap.page_allocator.free(dev);
        _ = lib.Memcpy(host.ptr, px, bytes, cudart.MemcpyDeviceToHost);
        @memcpy(dev, host);
        const cpu_buf = backend.Buf{ .ptr = @ptrCast(dev.ptr), .len = total };
        cpu_mod.CpuBackend.vtable.l2normQkScale(.{}, @ptrCast(&l2qk_cpu), cpu_buf, nk, kd, k_offset, qscale);
        // Device side: run the kernel on `px` itself, then read it back.
        // `$QWEN35_L2QK_CHECK` 是同步的逐元素对拍，永远不参与图，所以传直通图。
        _ = Launch.l2norm_qk_scale(backend.Graph.direct, px, @intCast(nk), @intCast(kd), @intCast(k_offset), gdn_l2_eps, qscale, null);
        _ = lib.DeviceSynchronize();
        _ = lib.Memcpy(host.ptr, px, bytes, cudart.MemcpyDeviceToHost);
        var worst: f32 = 0;
        var worst_i: usize = 0;
        var bad: usize = 0;
        for (0..total) |i| {
            const d = @abs(host[i] - dev[i]);
            const denom = @max(@abs(host[i]), @abs(dev[i]));
            const rel = if (denom > 0) d / denom else d;
            if (rel > worst) {
                worst = rel;
                worst_i = i;
            }
            if (rel > 1e-6) bad += 1;
        }
        std.debug.print("L2QKCHECK nk={d} kd={d} k_off={d} total={d} worst={e:.3} at {d} (>1e-6: {d})\n", .{ nk, kd, k_offset, total, worst, worst_i, bad });
    }

    fn dScaleVec_body(g: backend.Graph, ctx: *anyopaque, x: backend.Buf, s: f32) void {
        const self = self_(ctx);
        if (graphGate(g)) return;
        if (self.cu != null) {
            if (dBufPtr(x)) |px| {
                const rc = Launch.scale_vec(g, px, s, @intCast(x.len), self.launchStream());
                if (rc == cudart.cudaSuccess) return;
                self.ewLaunchFail("scale_vec", rc);
            } else return;
        }
        cpu_mod.CpuBackend.vtable.scaleVec(.{}, @ptrCast(&self.cpu), x, s);
    }

    fn dAddResid_body(g: backend.Graph, ctx: *anyopaque, dst: backend.Buf, src: backend.Buf) void {
        const self = self_(ctx);
        if (graphGate(g)) return;
        // Equal lengths are the engine's contract (`elementwise.bufAddResid`
        // zips the two slices).  A mismatch is an engine bug, so refuse to guess
        // rather than silently truncating to the shorter.
        if (src.len != dst.len) {
            std.debug.print("cuda backend: addResid length mismatch (dst={d} src={d}); using the CPU path\n", .{ dst.len, src.len });
            cpu_mod.CpuBackend.vtable.addResid(.{}, @ptrCast(&self.cpu), dst, src);
            return;
        }
        if (self.cu != null) {
            if (dBufPtr(dst)) |pd| {
                if (dBufPtr(src)) |ps| {
                    const rc = Launch.add_resid(g, pd, ps, @intCast(dst.len), self.launchStream());
                    if (rc == cudart.cudaSuccess) return;
                    self.ewLaunchFail("add_resid", rc);
                } else return;
            } else return;
        }
        cpu_mod.CpuBackend.vtable.addResid(.{}, @ptrCast(&self.cpu), dst, src);
    }

    /// True when this `Buf` is NOT one of ours, i.e. it wraps ordinary host
    /// memory (`Buf.wrap` of a `[]f32`, which the engine still does for f32
    /// token embeddings and for anything it owns itself).  The test is whether
    /// the pointer is in the `dev_bufs` list -- the only way a device region
    /// comes into existence, since `dAllocBuf` is the sole allocator.
    ///
    /// This exists because `Buf` cannot tell host from device by looking at
    /// it, and two slots legitimately receive host memory:
    ///   * `copyRange(device, Buf.wrap(host_embedding_row))` -- the f32
    ///     embedding lookup (engine.zig `embedToken`),
    ///   * `dCopyFromHost`'s callers.
    /// Treating such a `src` as a device pointer is an immediate illegal
    /// access -- caught in the P2c acceptance run on the Q8_K_XL model, whose
    /// embedding table is f32 (the Q6_K one is quantized and goes through
    /// `dequantRow`, which is why it passed first).
    fn isDevBuf(self: *CudaBackend, b: backend.Buf) bool {
        const lo: usize = @intFromPtr(b.ptr);
        const hi: usize = lo + b.len * @sizeOf(f32);
        for (self.dev_bufs.items) |r| {
            const rlo: usize = @intFromPtr(r.ptr);
            const rhi: usize = rlo + r.len * @sizeOf(f32);
            // Containment, not equality: `Buf.span` advances the pointer, so a
            // strided view of our own allocation is still ours.
            if (lo >= rlo and hi <= rhi) return true;
        }
        return false;
    }

    /// KV append.  Byte-identical to `dCopyRange_body`: the whole `copy_range`
    /// launch is a graph node like any other, so a replay re-runs it with the
    /// pointers it was captured with.  `slot` is part of the caller's contract
    /// (`kvAppend` decides which row) and is not needed here.
    fn dCopyRangeSlot_body(g: backend.Graph, ctx: *anyopaque, dst: backend.Buf, src: backend.Buf, slot: usize) void {
        _ = slot;
        if (graphGate(g)) return;
        CudaBackend.dCopyRange_body(g, ctx, dst, src);
    }

    fn dCopyRange_body(g: backend.Graph, ctx: *anyopaque, dst: backend.Buf, src: backend.Buf) void {
        const self = self_(ctx);
        // Same contract as `addResid`: `elementwise.bufCopyRange` is a plain
        // `@memcpy`, which requires equal lengths.  Mismatch = engine bug ->
        // CPU path, not a silent partial copy.
        if (src.len != dst.len) {
            std.debug.print("cuda backend: copyRange length mismatch (dst={d} src={d}); using the CPU path\n", .{ dst.len, src.len });
            cpu_mod.CpuBackend.vtable.copyRange(.{}, @ptrCast(&self.cpu), dst, src);
            return;
        }
        if (self.cu != null) {
            if (dBufPtr(dst)) |pd| {
                if (self.isDevBuf(src)) {
                    if (dBufPtr(src)) |ps| {
                        // Both device-resident: a pure device-to-device move.
                        const rc = Launch.copy_range(g, pd, ps, @intCast(dst.len), self.launchStream());
                        if (rc == cudart.cudaSuccess) return;
                        self.ewLaunchFail("copy_range", rc);
                    } else return;
                } else {
                    // `src` is host memory (an f32 embedding row).  Same kernel,
                    // opposite direction: this is the documented H2D path.
                    CudaBackend.dCopyFromHost(.{}, @ptrCast(self), dst, src.slice()[0..dst.len]);
                    return;
                }
            } else return;
        }
        cpu_mod.CpuBackend.vtable.copyRange(.{}, @ptrCast(&self.cpu), dst, src);
    }

    fn dZeroBuf_body(g: backend.Graph, ctx: *anyopaque, buf: backend.Buf) void {
        const self = self_(ctx);
        if (graphGate(g)) return;
        if (self.cu != null) {
            if (dBufPtr(buf)) |p| {
                const rc = Launch.zero_buf(g, p, @intCast(buf.len), self.launchStream());
                if (rc == cudart.cudaSuccess) return;
                self.ewLaunchFail("zero_buf", rc);
            } else return;
        }
        // Device path unavailable: still give dst the zero semantics it was
        // promised.  This touches host memory only when the buffer IS host
        // memory (i.e. the CPU fallback path), which is the honest reading.
        @memset(buf.slice(), 0);
    }

    // ==== slots: the two host<->device bridges ==============================
    //
    // These are the only two slots that can cross PCIe:
    //
    //   copyFromHost -- a token's f32 embedding row (unquantized token_embd
    //                   only), once per token, `hidden` floats.
    //   copyToHost   -- diagnostic only: nothing on the per-token path calls
    //                   it any more.  The last user was the per-layer
    //                   gamma/beta read-back, which `gdnStepRows` now takes as
    //                   device regions (`Buf`) instead of host `f32` arrays.
    //
    // Everything else in this file is device-to-device.

    fn dCopyFromHost_body(g: backend.Graph, ctx: *anyopaque, dst: backend.Buf, src: []const f32) void {
        const self = self_(ctx);
        // The bytes behind `dst` just changed, so anything quantised from it is
        // stale.  Conservative and free: the next matvec re-quantises.
        self.actQClear();
        // A blocking copy INVALIDATES an open capture (the runtime returns
        // `cudaErrorStreamCaptureUnsupported` and the graph is left empty), so
        // the capture is abandoned here instead: this token then runs on the
        // direct path and the next one may capture again.  Detection has to be
        // here rather than at each engine call site because the engine has no
        // idea a capture is open -- by design (AGENTS.md 1.1: it must not know
        // about the hardware's execution strategy).
        if (self.capturing(g)) self.graphAbandon(g, "host-to-device copy inside forward");
        if (self.cu != null and src.len != 0 and src.len <= dst.len) {
            if (dBufPtr(dst)) |pd| {
                const cu = &self.cu.?;
                const rc = cu.lib.Memcpy(
                    @ptrCast(pd),
                    @ptrCast(src.ptr),
                    src.len * @sizeOf(f32),
                    cudart.MemcpyHostToDevice,
                );
                self.stat_h2d_bytes += src.len * @sizeOf(f32);
                // `$QWEN35_STAT_TALKY`: breakdown of who still hands this backend
                // host memory.  Grouped by element count, which identifies the
                // caller (hidden=1024 vs head_dim=256).
                if (cudart.getenv("QWEN35_STAT_TALKY") != null) {
                    if (self.debug_h2d_len == src.len) {
                        self.debug_h2d_same += 1;
                    } else {
                        std.debug.print("H2DCOPY len={d} count={d}\n", .{ self.debug_h2d_len, self.debug_h2d_same });
                        self.debug_h2d_len = src.len;
                        self.debug_h2d_same = 1;
                    }
                }
                if (rc == cudart.cudaSuccess) return;
                self.ewLaunchFail("copyFromHost", rc);
            } else return;
        }
        // A length mismatch is an engine bug; let the CPU reference report it.
        cpu_mod.CpuBackend.vtable.copyFromHost(.{}, @ptrCast(&self.cpu), dst, src);
    }

    fn dCopyToHost_body(g: backend.Graph, ctx: *anyopaque, dst: []f32, src: backend.Buf) void {
        const self = self_(ctx);
        std.debug.assert(dst.len <= src.len);
        if (self.capturing(g)) self.graphAbandon(g, "device-to-host copy inside forward");
        if (self.cu != null) {
            if (dBufPtr(src)) |ps| {
                // Blocking on purpose: this slot is for small diagnostic
                // regions now (no per-token caller), and a blocking copy into
                // pageable host memory is both simpler and, at that size,
                // indistinguishable in cost from an async one.
                const cu = &self.cu.?;
                const rc = cu.lib.Memcpy(
                    @ptrCast(dst.ptr),
                    @ptrCast(ps),
                    dst.len * @sizeOf(f32),
                    cudart.MemcpyDeviceToHost,
                );
                self.stat_d2h_bytes += dst.len * @sizeOf(f32);
                if (rc == cudart.cudaSuccess) return;
                self.ewLaunchFail("copyToHost", rc);
            } else return;
        }
        cpu_mod.CpuBackend.vtable.copyOut(.{}, @ptrCast(&self.cpu), dst, src);
    }

    // ==== slots: norms / activations (P2b, second batch) ====================
    //
    // `dBufPtr -> launch`, with the CPU slot as the fallback on ANY failure.
    //
    // Two properties worth stating:
    //
    //   * These are NOT bit-exact with the CPU.  rmsnorm/l2norm/softmax reduce
    //     in a different order (a block tree vs the CPU's serial loop), and
    //     every transcendental here (`expf`, `powf`, `cosf`, `sinf`) is a
    //     different implementation in the device runtime than in glibc.  The
    //     measured figures are in the .cu headers.
    //   * The read-only weight operand (`w` in rmsnorm, `wconv` in convStep) is
    //     a per-layer constant re-passed on every token, so it IS a device `Buf`
    //     uploaded exactly once at load time (`loadVecF32`); each call just reads
    //     the resident device pointer -- no per-token upload.

    fn dRmsnorm_body(g: backend.Graph, ctx: *anyopaque, out: backend.Buf, x: backend.Buf, w: backend.Buf, eps: f32) void {
        const self = self_(ctx);
        if (graphGate(g)) return;
        // The CPU reference asserts out.len == x.len == w.len (one row).  The
        // kernel supports several rows (w reused per row) for the shape the
        // engine does not currently produce, so the only mismatch worth
        // refusing is a partial row.
        if (x.len != 0 and w.len != 0 and x.len % w.len == 0 and x.len == out.len and self.cu != null) {
            if (dBufPtr(x)) |px| {
                if (dBufPtr(out)) |po| {
                    if (dBufPtr(w)) |pw| {
                        const rows: c_int = @intCast(x.len / w.len);
                        const rc = Launch.rmsnorm(g, po, px, pw, rows, @intCast(w.len), eps, self.launchStream());
                        if (rc == cudart.cudaSuccess) return;
                        self.ewLaunchFail("rmsnorm", rc);
                    } else return;
                } else return;
            } else return;
        }
        cpu_mod.CpuBackend.vtable.rmsnorm(.{}, @ptrCast(&self.cpu), out, x, w, eps);
    }

    /// Row-batched `rmsnorm`: one launch, `rows` blocks, `w` reused per row.
    /// `norm.cu` `k_rmsnorm` already indexes `x`/`out` by `blockIdx.x * n`, so
    /// nothing in the kernel changes -- each block reduces its own row exactly
    /// as it did when it was the only block.  Rows are therefore bit-identical
    /// to the per-row launches (the reduction order within a row is untouched).
    ///
    /// Falls back to the per-row loop (`rmsnormRows` semantics) whenever the
    /// geometry does not hold, which is what the CPU side does anyway.
    fn dRmsnormRows_body(g: backend.Graph, ctx: *anyopaque, out: backend.Buf, x: backend.Buf, w: backend.Buf, rows_in: usize, eps: f32) void {
        const self = self_(ctx);
        if (graphGate(g)) return;
        const n = w.len;
        if (rows_in != 0 and n != 0 and x.len == rows_in * n and out.len == rows_in * n and self.cu != null) {
            if (dBufPtr(x)) |px| {
                if (dBufPtr(out)) |po| {
                    if (dBufPtr(w)) |pw| {
                        const rc = Launch.rmsnorm(g, po, px, pw, @intCast(rows_in), @intCast(n), eps, self.launchStream());
                        if (rc == cudart.cudaSuccess) {
                            return;
                        }
                        self.ewLaunchFail("rmsnorm_rows", rc);
                    } else return;
                } else return;
            } else return;
        }
        cpu_mod.CpuBackend.vtable.rmsnormRows(.{}, @ptrCast(&self.cpu), out, x, w, rows_in, eps);
    }

    fn dL2norm_body(g: backend.Graph, ctx: *anyopaque, x: backend.Buf) void {
        const self = self_(ctx);
        if (self.cu != null) {
            if (dBufPtr(x)) |px| {
                // eps 1e-6 is elementwise.zig `l2norm`'s constant; it is passed
                // explicitly so the CPU reference stays the single definition.
                const rc = Launch.l2norm(g, px, 1, @intCast(x.len), 1e-6, self.launchStream());
                if (rc == cudart.cudaSuccess) return;
                self.ewLaunchFail("l2norm", rc);
            } else return;
        }
        cpu_mod.CpuBackend.vtable.l2norm(.{}, @ptrCast(&self.cpu), x);
    }

    /// Row-batched `l2norm`: one launch, `rows = x.len / n` blocks, each
    /// normalising its own `n`-element row with its own eps-inside-the-sum
    /// reduction.  `k_l2norm` already strides by `blockIdx.x * n`, so this is
    /// the same arithmetic as `x.len / n` single-row launches.
    fn dL2normRows_body(g: backend.Graph, ctx: *anyopaque, x: backend.Buf, n_in: usize) void {
        const self = self_(ctx);
        const n = n_in;
        if (n != 0 and x.len % n == 0 and self.cu != null) {
            if (dBufPtr(x)) |px| {
                const rows: c_int = @intCast(x.len / n);
                const rc = Launch.l2norm(g, px, rows, @intCast(n), 1e-6, self.launchStream());
                if (rc == cudart.cudaSuccess) return;
                self.ewLaunchFail("l2norm_rows", rc);
            } else return;
        }
        // No batching possible (unknown row width): the whole-slice reading is
        // the single-row one, which is what `dL2norm_body` means.
        cpu_mod.CpuBackend.vtable.l2normRows(.{}, @ptrCast(&self.cpu), x, n);
    }

    fn dSiluInPlace_body(g: backend.Graph, ctx: *anyopaque, x: backend.Buf) void {
        const self = self_(ctx);
        if (graphGate(g)) return;
        if (self.cu != null) {
            if (dBufPtr(x)) |px| {
                const rc = Launch.silu(g, px, @intCast(x.len), self.launchStream());
                if (rc == cudart.cudaSuccess) return;
                self.ewLaunchFail("silu", rc);
            } else return;
        }
        cpu_mod.CpuBackend.vtable.siluInPlace(.{}, @ptrCast(&self.cpu), x);
    }

    fn dMulSilu_body(g: backend.Graph, ctx: *anyopaque, out: backend.Buf, gate: backend.Buf) void {
        const self = self_(ctx);
        if (graphGate(g)) return;
        if (gate.len != out.len) {
            std.debug.print("cuda backend: mulSilu length mismatch (out={d} gate={d}); using the CPU path\n", .{ out.len, gate.len });
            cpu_mod.CpuBackend.vtable.mulSilu(.{}, @ptrCast(&self.cpu), out, gate);
            return;
        }
        if (self.cu != null) {
            if (dBufPtr(out)) |po| {
                if (dBufPtr(gate)) |pg| {
                    const rc = Launch.mul_silu(g, po, pg, @intCast(out.len), self.launchStream());
                    if (rc == cudart.cudaSuccess) return;
                    self.ewLaunchFail("mul_silu", rc);
                } else return;
            } else return;
        }
        cpu_mod.CpuBackend.vtable.mulSilu(.{}, @ptrCast(&self.cpu), out, gate);
    }

    fn dMulSigmoid_body(g: backend.Graph, ctx: *anyopaque, out: backend.Buf, gate: backend.Buf) void {
        const self = self_(ctx);
        if (graphGate(g)) return;
        if (gate.len != out.len) {
            std.debug.print("cuda backend: mulSigmoid length mismatch (out={d} gate={d}); using the CPU path\n", .{ out.len, gate.len });
            cpu_mod.CpuBackend.vtable.mulSigmoid(.{}, @ptrCast(&self.cpu), out, gate);
            return;
        }
        if (self.cu != null) {
            if (dBufPtr(out)) |po| {
                if (dBufPtr(gate)) |pg| {
                    const rc = Launch.mul_sigmoid(g, po, pg, @intCast(out.len), self.launchStream());
                    if (rc == cudart.cudaSuccess) return;
                    self.ewLaunchFail("mul_sigmoid", rc);
                } else return;
            } else return;
        }
        cpu_mod.CpuBackend.vtable.mulSigmoid(.{}, @ptrCast(&self.cpu), out, gate);
    }

    fn dSwiglu_body(g: backend.Graph, ctx: *anyopaque, gate: backend.Buf, up: backend.Buf) void {
        const self = self_(ctx);
        if (graphGate(g)) return;
        if (up.len != gate.len) {
            std.debug.print("cuda backend: swiglu length mismatch (gate={d} up={d}); using the CPU path\n", .{ gate.len, up.len });
            cpu_mod.CpuBackend.vtable.swiglu(.{}, @ptrCast(&self.cpu), gate, up);
            return;
        }
        if (self.cu != null) {
            if (dBufPtr(gate)) |pg| {
                if (dBufPtr(up)) |pu| {
                    const rc = Launch.swiglu(g, pg, pu, @intCast(gate.len), self.launchStream());
                    if (rc == cudart.cudaSuccess) return;
                    self.ewLaunchFail("swiglu", rc);
                } else return;
            } else return;
        }
        cpu_mod.CpuBackend.vtable.swiglu(.{}, @ptrCast(&self.cpu), gate, up);
    }

    fn dSoftmax_body(g: backend.Graph, ctx: *anyopaque, x: backend.Buf) void {
        const self = self_(ctx);
        if (graphGate(g)) return;
        if (self.cu != null) {
            if (dBufPtr(x)) |px| {
                const rc = Launch.softmax(g, px, @intCast(x.len), self.launchStream());
                if (rc == cudart.cudaSuccess) return;
                self.ewLaunchFail("softmax", rc);
            } else return;
        }
        cpu_mod.CpuBackend.vtable.softmax(.{}, @ptrCast(&self.cpu), x);
    }

    // ==== sampling (P2c: logits are device memory) ==========================
    //
    // The engine keeps `logits` in a backend region, so both of these used to
    // dereference a device pointer through the CPU implementation.  argmax is
    // a pure max and therefore goes to the device exactly (see the kernel);
    // `sample` stays on the CPU because the sampler is a sampling POLICY
    // (temperature, top-k/p, RNG) whose reference implementation must not be
    // duplicated, and it needs the whole distribution.

    /// Rows `dArgmaxRows` will accept in one call (the spec verify block).
    /// Bounds the stack array the read-back lands in; the engine's own block
    /// size is 16 for both DSpine and MTP.
    const spec_argmax_rows: usize = 16;

    /// `rows` argmaxes in ONE launch and ONE read-back.
    ///
    /// The single-row slot blocks on a 4-byte D2H, which for the batched
    /// speculative verify meant five copies per round.  Measured on 27B MTP:
    /// 36 such calls totalled 235 ms of host wait against 9 us of kernel time --
    /// each one waited on the whole preceding round's launches, so paying that
    /// wait five times per round was pure loss.
    fn dArgmaxRows(g: backend.Graph, ctx: *anyopaque, logits: backend.Buf, rows: usize, out: []u32) void {
        const self = self_(ctx);
        const xg_prev = xg_push_slot(g.slot);
        defer xg_cur_slot = xg_prev;
        std.debug.assert(out.len >= rows and rows > 0);
        if (self.cu == null or logits.len == 0 or logits.len % rows != 0) {
            // Host memory (or a malformed region): the CPU oracle is the
            // definition, so fall back to it rather than refusing.
            cpu_mod.CpuBackend.vtable.argmaxRows(.{}, @ptrCast(&self.cpu), logits, rows, out);
            return;
        }
        const px = dBufPtr(logits) orelse {
            cpu_mod.CpuBackend.vtable.argmaxRows(.{}, @ptrCast(&self.cpu), logits, rows, out);
            return;
        };
        const cu = &self.cu.?;
        const vocab: usize = logits.len / rows;
        const bytes = rows * @sizeOf(c_int);
        ensureBuf(&cu.lib, &self.d_idx, &self.d_idx_bytes, bytes) catch {
            cpu_mod.CpuBackend.vtable.argmaxRows(.{}, @ptrCast(&self.cpu), logits, rows, out);
            return;
        };
        const rc = Launch.argmax_batch(g, px, @intCast(vocab), @intCast(rows), @ptrCast(@alignCast(self.d_idx)), null);
        if (rc != cudart.cudaSuccess) {
            self.ewLaunchFail("argmax_batch", rc);
            cpu_mod.CpuBackend.vtable.argmaxRows(.{}, @ptrCast(&self.cpu), logits, rows, out);
            return;
        }
        var host: [spec_argmax_rows]c_int = undefined;
        std.debug.assert(rows <= spec_argmax_rows);
        const rc2 = cu.lib.Memcpy(&host, self.d_idx, bytes, cudart.MemcpyDeviceToHost);
        if (rc2 != cudart.cudaSuccess) {
            self.ewLaunchFail("argmax_batch index download", rc2);
            cpu_mod.CpuBackend.vtable.argmaxRows(.{}, @ptrCast(&self.cpu), logits, rows, out);
            return;
        }
        for (0..rows) |r| out[r] = @intCast(host[r]);
    }

    /// `Exec.argmaxConf`: argmax plus the top-1 softmax probability.
    ///
    /// One launch and one 12-byte read-back, i.e. the same cost as `argmax`
    /// itself.  The probability feeds the MTP drafter's early stop, which is
    /// llama.cpp's `p_min` (`common/speculative.cpp:1789`): a drafted token
    /// costs a full verify row whether or not it is accepted, so a low-confidence
    /// draft is worth abandoning before it is ever proposed.
    pub fn argmaxConf(self: *CudaBackend, logits: backend.Buf, out_p: *f32) u32 {
        out_p.* = 1.0;
        if (self.cu == null or logits.len == 0) {
            return cpuConfFallback(self, logits, out_p);
        }
        const px = dBufPtr(logits) orelse return cpuConfFallback(self, logits, out_p);
        const cu = &self.cu.?;
        ensureBuf(&cu.lib, &self.d_conf, &self.d_conf_bytes, 3 * @sizeOf(c_int)) catch return 0;
        // 直通图：本函数紧跟一次阻塞 D2H（读回 12 字节），不可能被捕获。
        const rc = Launch.argmax_conf(backend.Graph.direct, px, @intCast(logits.len), @ptrCast(@alignCast(self.d_conf)), null);
        if (rc != cudart.cudaSuccess) {
            self.ewLaunchFail("argmax_conf", rc);
            return cpuConfFallback(self, logits, out_p);
        }
        var host: [3]c_int = undefined;
        if (cu.lib.Memcpy(&host, self.d_conf, 3 * @sizeOf(c_int), cudart.MemcpyDeviceToHost) != cudart.cudaSuccess) {
            return cpuConfFallback(self, logits, out_p);
        }
        const sum: f32 = @bitCast(host[2]);
        // `sum` is already `exp(x - max)` summed, so the probability IS `1/sum`
        // -- the numerator is `exp(max - max) = 1`.
        if (sum > 0 and std.math.isFinite(sum)) out_p.* = 1.0 / sum;
        return @intCast(host[0]);
    }

    fn cpuConfFallback(self: *CudaBackend, logits: backend.Buf, out_p: *f32) u32 {
        _ = self;
        // Host logits (or a device failure): compute both on the host, where the
        // reference already lives.
        const row = logits.slice();
        if (row.len == 0) return 0;
        var best: f32 = -std.math.inf(f32);
        var bi: u32 = 0;
        for (row, 0..) |v, i| {
            if (v > best) {
                best = v;
                bi = @intCast(i);
            }
        }
        var sum: f64 = 0;
        for (row) |v| sum += @exp(@as(f64, v - best));
        out_p.* = if (sum > 0) @floatCast(1.0 / sum) else 1.0;
        return bi;
    }

    fn dArgmaxConf(g: backend.Graph, ctx: *anyopaque, logits: backend.Buf, p: *f32) u32 {
        _ = g;
        const self = self_(ctx);
        if (self.anySlotLive()) p.* = 1.0; // the graph already ran it
        return self.argmaxConf(logits, p);
    }

    fn dArgmax_body(g: backend.Graph, ctx: *anyopaque, logits: backend.Buf) u32 {
        const self = self_(ctx);
        if (self.cu != null and logits.len != 0) {
            if (dBufPtr(logits)) |px| {
                // The launcher writes through `d_out`, so it MUST be device
                // memory: the previous build passed the address of a stack
                // `c_int`, which the kernel dereferenced as a GPU pointer.  A
                // dedicated 4-byte device cell is reused (grown once) and only
                // the final index -- 4 bytes -- comes back per token.
                const cu = &self.cu.?;
                ensureBuf(&cu.lib, &self.d_idx, &self.d_idx_bytes, @sizeOf(c_int)) catch return 0;
                // $QWEN35_PROF: split the per-token drain into "device was busy"
                // and "host was waiting".  `ev_arg_a` is recorded before the
                // argmax launch and `ev_arg_b` right after it; both are queue
                // positions, so `b - a` is what the device spent finishing that
                // tail.  The host wall clock around the blocking copy is what
                // the CPU actually paid.  Both are only touched when
                // gpuTimingOn(), so the default path is one predictable branch.
                const timed = self.gpuTimingOn();
                var host_t0: ?Io.Clock.Timestamp = null;
                if (timed) {
                    if (!self.ensureEvents()) {
                        self.ev_on = false;
                    } else {
                        host_t0 = Io.Clock.Timestamp.now(self.io, .awake);
                        if (cu.lib.EventRecord(self.ev_arg_a, null) != cudart.cudaSuccess) {
                            self.stat_n_ev_err += 1;
                        }
                    }
                }
                const rc = Launch.argmax(g, px, @intCast(logits.len), @ptrCast(@alignCast(self.d_idx)), null);
                if (rc != cudart.cudaSuccess) {
                    self.ewLaunchFail("argmax", rc);
                    return 0;
                }
                if (timed and host_t0 != null) {
                    if (cu.lib.EventRecord(self.ev_arg_b, null) != cudart.cudaSuccess) {
                        self.stat_n_ev_err += 1;
                    }
                }
                var host_idx: c_int = 0;
                // Measured 2026-11: a pinned `MemcpyAsync` + stream sync is
                // 5191us -> 5100us per token, i.e. noise.  The wait here is the
                // device finishing the 240 kernels this token already queued
                // (8.25 ms of real device work against an 8.20 ms host token),
                // not copy overhead, so the plain blocking copy stays.
                const rc2 = cu.lib.Memcpy(&host_idx, self.d_idx, @sizeOf(c_int), cudart.MemcpyDeviceToHost);
                if (rc2 != cudart.cudaSuccess) {
                    self.ewLaunchFail("argmax index download", rc2);
                    return 0;
                }
                if (timed) {
                    if (host_t0) |t0| {
                        self.stat_ns_argmax_host += durNs(t0, self.io);
                        self.stat_n_argmax_timed += 1;
                    }
                    // The events are now complete (the copy above waited on the
                    // whole queue), so reading the interval synchronizes nothing
                    // extra.  Guarded: `evElapsedNs` returns null on error.
                    if (self.evElapsedNs(self.ev_arg_a, self.ev_arg_b)) |ns| {
                        self.stat_ns_argmax_gpu += ns;
                    }
                    // Record the device position that closes THIS token, then
                    // compare it against the previous token's close.  That spans
                    // every kernel of the token, which is the number that has to
                    // be compared with the host wall clock.
                    if (cu.lib.EventRecord(self.ev_tok_end, null) == cudart.cudaSuccess) {
                        if (self.tok_end_valid) {
                            if (self.evElapsedNs(self.ev_prev_tok_end, self.ev_tok_end)) |ns| {
                                self.stat_ns_token_dev += ns;
                                self.stat_n_token_dev += 1;
                            }
                        }
                        if (cu.lib.EventRecord(self.ev_prev_tok_end, null) != cudart.cudaSuccess) {
                            self.stat_n_ev_err += 1;
                        }
                        self.tok_end_valid = true;
                    } else self.stat_n_ev_err += 1;
                }
                self.stat_d2h_bytes += @sizeOf(c_int);
                if (self.gpu_ev_on) self.gpuEvReadToken();
                return @intCast(host_idx);
            } else return 0;
        }
        return cpu_mod.CpuBackend.vtable.argmax(.{}, @ptrCast(&self.cpu), logits);
    }

    /// Grow the A/B host row to at least `n` floats.  Only ever called from the
    /// debug branch, i.e. outside any capture, so a plain allocation is fine.
    fn ensureSampleAbHost(self: *CudaBackend, n: usize) bool {
        if (self.h_sample_ab) |b| {
            if (b.len >= n) return true;
        }
        const alloc = std.heap.page_allocator;
        if (self.h_sample_ab) |b| alloc.free(b);
        self.h_sample_ab = alloc.alloc(f32, n) catch {
            self.h_sample_ab = null;
            return false;
        };
        return true;
    }

    /// 分配（一次）采样控制块与 kernel scratch。必须在任何 capture 打开**之前**
    /// 跑：capture 区间内 `cudaMalloc` 非法，而采样 kernel 的第一次发射就会落在
    /// 那里。
    ///
    /// 控制块是**设备常驻**的，由两次重放之间一次普通 H2D 刷新。采样 kernel 在
    /// 执行时从它读 temperature / top_k / top_p / PRNG 种子，并把选中的 token id
    /// 写回同一块，所以捕获下来的采样节点永远不需要 `SetParams`。
    ///
    /// 刷新刻意**不**发生在 capture 之内（那会被录成节点，重放时永远用录制那
    /// 一步的值），也**不**发生在图 launch 之后（那时采样已经跑完）。它在
    /// `graphBegin` 里发到图自己的流上，也就是下一次 `cudaGraphLaunch` 之前 ——
    /// 正是"主机在两次重放之间写、图在重放期间读"这个窗口。
    ///
    /// 主机侧没有常驻缓冲：发布/读回都直接对着设备块做一次阻塞 `Memcpy`，
    /// 源/目标是栈上的局部结构体。这样既不需要 pinned 分配，也就不存在
    /// "GPU 读到映射内存陈旧副本"这一类问题。
    fn ensureSampleBufs(self: *CudaBackend) bool {
        if (self.cu == null) return false;
        if (self.d_sample_ctl == null or self.d_sample_scratch == null) {
            const rc = Launch.sample_init(&self.d_sample_ctl, &self.d_sample_scratch);
            if (rc != cudart.cudaSuccess) {
                self.ewLaunchFail("sample init", rc);
                return false;
            }
        }
        return true;
    }

    /// `q35_sample_ctl`（`src/kernels/cuda/sample.cu`）的镜像。字段顺序就是那次
    /// 64 字节 H2D 的线格式——两边要么一起改，要么都不改。
    const SampleCtl = extern struct {
        vocab: c_int = 0,
        temperature_bits: c_int = 0,
        top_k: c_int = 0,
        top_p_bits: c_int = 0,
        rng_lo: c_int = 0,
        rng_hi: c_int = 0,
        token_out: c_int = 0,
        rng_out_lo: c_int = 0,
        rng_out_hi: c_int = 0,
        thresh_bits: c_int = 0,
        flags: c_int = 0,
        sel_prefix: c_int = 0,
        sel_mask: c_int = 0,
        sel_rank: c_int = 0,
        reserved0: c_int = 0,
        reserved1: c_int = 0,
    };

    /// 把这一步的采样参数发布进设备控制块：一次 64 字节 H2D，不发 kernel、
    /// 不做分配。
    ///
    /// `sp.seed` 不是每个 token 都重读：`Sampler.state` 是活的 PRNG 位置，每抽
    /// 一个 token 恰好推进一次（kernel 把推进后的值写回 `rng_out_*`，
    /// `dSample_body` 收集 token 时再把它拷进 `sp.state`）。在这里发布 `state`
    /// 正是"重放的流是接着往下走、而不是从头开始"的原因。
    fn publishSampleCtl(self: *CudaBackend, logits_len: usize, sp: *const sampler.Sampler, stream: ?*anyopaque) bool {
        _ = stream;
        const cu = &(self.cu orelse return false);
        const d_ctl = self.d_sample_ctl orelse return false;
        var hc: SampleCtl = .{};
        hc.vocab = @intCast(logits_len);
        hc.temperature_bits = @bitCast(sp.temperature);
        hc.top_k = @intCast(sp.top_k);
        hc.top_p_bits = @bitCast(sp.top_p);
        const seed: u64 = sp.state;
        hc.rng_lo = @bitCast(@as(u32, @truncate(seed)));
        hc.rng_hi = @bitCast(@as(u32, @truncate(seed >> 32)));
        // 一 次 64 字节阻塞 H2D，源是栈上的局部结构体。
        //
        // 与 `publishCtl` 同理：拷贝的下游就是图 launch（采样 kernel 是图的
        // 最后一个节点），没有可重叠的工作，用 `MemcpyAsync` 只会强制源内存
        // 必须 page-locked（那是 `cudaMemcpyAsync` 的 API 要求，不是"拷贝在
        // GPU 上跑"）。
        //
        // 但时机比形态更要紧：这条拷贝必须发生在 capture 打开之前。落在
        // capture 内会被录成 memcpy 节点，重放时永远用录制那一步的参数；
        // 落在 `GraphLaunch` 之后又太晚（采样已经跑完）。
        const rc = cu.lib.Memcpy(d_ctl, @ptrCast(&hc), @sizeOf(SampleCtl), cudart.MemcpyHostToDevice);
        if (rc != cudart.cudaSuccess) {
            self.ewLaunchFail("sample ctl publish", rc);
            return false;
        }
        self.stat_h2d_bytes += @sizeOf(SampleCtl);
        return true;
    }

    /// 把采样 kernel 写进设备常驻块的结果读回主机：token id 与推进后的 PRNG。
    ///
    /// 一次 64 字节阻塞 D2H（一个 cache line）。读回整块而不只读 token id，
    /// 是为了顺带拿到 `rng_out_*` —— 下一个 token 的 `publishSampleCtl` 才能
    /// 从同一个 PRNG 位置继续。
    fn collectSampledToken(self: *CudaBackend, sp: *sampler.Sampler) u32 {
        const cu = &(self.cu orelse return self.sample_token);
        const d_ctl = self.d_sample_ctl orelse return self.sample_token;
        var hc: SampleCtl = .{};
        const rc = cu.lib.Memcpy(@ptrCast(&hc), d_ctl, @sizeOf(SampleCtl), cudart.MemcpyDeviceToHost);
        if (rc != cudart.cudaSuccess) {
            self.ewLaunchFail("sample token download", rc);
            return self.sample_token;
        }
        self.stat_d2h_bytes += @sizeOf(SampleCtl);
        // `Sampler.sample` consumes exactly one `nextU64` per non-greedy draw
        // (sampler.zig:120 `nextF32`), and the kernel does the same, so the state
        // it wrote back IS the position the CPU reference would be at.
        if (sp.temperature > 0.0) {
            sp.state = (@as(u64, @as(u32, @bitCast(hc.rng_out_hi))) << 32) | @as(u32, @bitCast(hc.rng_out_lo));
        }
        const tok: u32 = @intCast(@as(u32, @bitCast(hc.token_out)));
        self.sample_token = tok;
        if (self.gpu_ev_on) self.gpuEvReadToken();
        return tok;
    }

    /// 设备采样。
    ///
    /// 整套策略 —— greedy argmax、temperature、top-k、top-p、PRNG 推进 —— 都在
    /// `sample.cu` 里跑。调用方式上有两点要紧：
    ///
    ///   * 发射走 `launchStream()`，也就是录制 decode 图时的那条 capture 流，
    ///     于是采样器像 matvec 一样成为那张图的一个节点。重放时下面的门控会
    ///     抑制主机侧发射，因为图里已经包含它了。
    ///   * 捕获路径上**不**在这里下载结果。D2H 无法被捕获；由 `stepEnd` 在图
    ///     关闭之后执行它，`sampledToken` 再把值交给引擎。直接发射路径上下载
    ///     当场发生，`sampledToken` 是恒等。
    fn dSample_body(g: backend.Graph, ctx: *anyopaque, logits: backend.Buf, scratch: backend.Buf, sp: *sampler.Sampler) u32 {
        const self = self_(ctx);
        // A/B 开关：在 GPU 自己产出的 logits 上跑 CPU 参照采样器。
        //
        // This is the ONLY way to compare the two samplers end to end, because
        // every other comparison runs into the CPU/CUDA matvec divergence first
        // (f32 accumulation order, documented in AGENTS.md §4) and the samplers
        // are then fed different logits.  With this set, both runs consume the
        // identical device logits row, so any token difference is the sampler's
        // fault alone.  Read once per process, on the host, before any capture.
        if (self.sample_cpu_ab and self.capturing(g)) {
            // A blocking D2H inside a capture is illegal and would silently
            // corrupt the graph.  Refuse loudly: this switch is for A/B on the
            // direct path only, and `graphAbandon` is the documented way to say
            // "this capture cannot continue".
            self.graphAbandon(g, "QWEN35_CUDA_SAMPLE_CPU inside a capture");
        }
        if (self.sample_cpu_ab) {
            // CPU 采样器直接对行下标，所以它需要一份**主机**副本：这里 `logits`
            // 是设备指针。`h_sample_ab` 按需增长并复用，拷贝是阻塞的 —— 整个
            // 分支都是调试用 A/B，永远不是计算路径。
            if (!self.ensureSampleAbHost(logits.len)) {
                return cpu_mod.CpuBackend.vtable.sample(.{}, @ptrCast(&self.cpu), logits, scratch, sp);
            }
            const dst = self.h_sample_ab.?[0..logits.len];
            // 内联的阻塞 D2H（`dCopyToHost_body` 是同级结构体的方法，从这里按名
            // 字够不到）。
            const cu = &(self.cu orelse return cpu_mod.CpuBackend.vtable.sample(
                .{}, @ptrCast(&self.cpu), logits, scratch, sp,
            ));
            const ps = dBufPtr(logits) orelse return cpu_mod.CpuBackend.vtable.sample(
                .{}, @ptrCast(&self.cpu), logits, scratch, sp,
            );
            const rc = cu.lib.Memcpy(
                @ptrCast(dst.ptr), @ptrCast(ps), dst.len * @sizeOf(f32), cudart.MemcpyDeviceToHost,
            );
            if (rc != cudart.cudaSuccess) {
                self.ewLaunchFail("sample ab row download", rc);
                return cpu_mod.CpuBackend.vtable.sample(.{}, @ptrCast(&self.cpu), logits, scratch, sp);
            }
            const host_row = backend.Buf.wrap(dst);
            // `scratch` 也会传给 CPU 采样器，而这个后端在用的时候引擎给的那份
            // 是**设备**缓冲；不给参照一份同样长度的主机 scratch，它就会去排序
            // 一个设备指针。
            if (self.h_sample_ab_scratch == null or self.h_sample_ab_scratch.?.len < logits.len) {
                const alloc = std.heap.page_allocator;
                if (self.h_sample_ab_scratch) |b| alloc.free(b);
                self.h_sample_ab_scratch = alloc.alloc(f32, logits.len) catch null;
            }
            const hs = self.h_sample_ab_scratch orelse
                return cpu_mod.CpuBackend.vtable.sample(.{}, @ptrCast(&self.cpu), host_row, scratch, sp);
            return cpu_mod.CpuBackend.vtable.sample(
                .{}, @ptrCast(&self.cpu), host_row, backend.Buf.wrap(hs[0..logits.len]), sp,
            );
        }
        // 设备采样器有自己的 scratch（`d_sample_scratch`）；引擎提供的那份只有
        // 下面的 CPU 回退路径才需要。
        if (self.cu == null or logits.len == 0) {
            return cpu_mod.CpuBackend.vtable.sample(.{}, @ptrCast(&self.cpu), logits, scratch, sp);
        }
        // 直发段（MTP 草稿块）里的采样：没有图在管这一段，所以参数走阻塞 H2D
        // 发布，kernel 照常发射，结果立刻 D2H 读回 —— 与图外 decode 完全同一条
        // 路径，只是没有 capture 能让它被录进去。
        if (self.g_direct) {
            if (!self.ensureSampleBufs()) {
                return cpu_mod.CpuBackend.vtable.sample(.{}, @ptrCast(&self.cpu), logits, scratch, sp);
            }
            const pdx = dBufPtr(logits) orelse
                return cpu_mod.CpuBackend.vtable.sample(.{}, @ptrCast(&self.cpu), logits, scratch, sp);
            self.sample_sp = sp;
            self.sample_vocab = logits.len;
            if (!self.publishSampleCtl(logits.len, sp, null)) return 0;
            const rc_s = Launch.sample(backend.Graph.direct, pdx, @intCast(logits.len), @ptrCast(@alignCast(self.d_sample_ctl)), self.d_sample_scratch, null);
            if (rc_s != cudart.cudaSuccess) {
                self.ewLaunchFail("sample (direct)", rc_s);
                return 0;
            }
            self.sampled = self.collectSampledToken(sp);
            return self.sampled;
        }
        // 活的 S1 图在上一次重放时已经在 `g_stream` 上跑过采样器了；仍然由
        // `sample` 把 token 交回去，所以下载发生在这里。（S3 会把这件事换成图
        // 自己在 `graphEnd` 里读控制块；在新路径上支持捕获之前，S1 图从不为
        // live，这个分支不可达。）
        const replayed = self.gs(0).state == .live;
        if (replayed) {
            return self.collectSampledToken(sp);
        }
        // XG 重放：图已在 `graphBegin` 里 launch 完，采样参数也已经在那里发布
        // 进设备控制块（见 `dGraphBegin`）。这里**什么都不能再写** —— 尤其不能
        // 再 `publishSampleCtl`，那条路径会把控制块整体重置，正好覆盖图刚写下的
        // `CTL_FLAGS` 与 `CTL_TOKEN_OUT`（实测：重放后 token 永远读回 0）。
        if (g.replayed) {
            self.sample_sp = sp;
            self.sample_vocab = logits.len;
            return 0;
        }
        // 录制期（草稿块）：这一趟的 f32 logits 常驻在 `d_deq` 里，`dSampledToken`
        // 在 `graphEnd` 之后要拿它做那次阻塞 D2H。只记指针：它不随步变，也不会
        // 在 capture 里过期（`graphReserve` 提前把它分配好了）。
        if (self.graphActive(g) and self.capturing(g)) {
            self.sample_sp = sp;
            self.sample_vocab = logits.len;
            self.sample_logits = if (dBufPtr(logits) != null) self.d_deq else null;
        }
        if (!self.ensureSampleBufs()) {
            return cpu_mod.CpuBackend.vtable.sample(.{}, @ptrCast(&self.cpu), logits, scratch, sp);
        }
        const px = dBufPtr(logits) orelse
            return cpu_mod.CpuBackend.vtable.sample(.{}, @ptrCast(&self.cpu), logits, scratch, sp);
        self.sample_sp = sp;
        self.sample_vocab = logits.len;
        const stream = self.launchStream(); // 录制期间就是 capture 流
        // 录制期不再从这里发布：`cudaMemcpyAsync` 落在 capture 区间内会被录成
        // 一个 **memcpy 节点**，重放时把录制那一步的参数写回设备块（实测：
        // `QWEN35_XD_NODES` 报出 1 个 non-kernel 节点，正是它）。参数已由
        // `dGraphBegin` 在图外、图 launch 之前发布好，这里只负责把 kernel 录进去。
        if (!self.capturing(g) and !self.publishSampleCtl(logits.len, sp, stream)) {
            return cpu_mod.CpuBackend.vtable.sample(.{}, @ptrCast(&self.cpu), logits, scratch, sp);
        }
        const rc = Launch.sample(g,
            px, @intCast(logits.len), @ptrCast(@alignCast(self.d_sample_ctl)), self.d_sample_scratch, stream,
        );
        if (rc != cudart.cudaSuccess) {
            self.ewLaunchFail("sample", rc);
            return cpu_mod.CpuBackend.vtable.sample(.{}, @ptrCast(&self.cpu), logits, scratch, sp);
        }
        // 采样 kernel 本身是图里的最后一个节点，但**读回 token 的 D2H 不能
        // 进图**：录进去的话重放永远返回录制时那个 id（实测每 token 重复）。
        // 所以图活跃时这里只发射、返回 0，由 `dGraphEnd` 在图关闭后做那一次
        // 4 字节读取（`sampledToken` 再把结果交给 engine）。
        // 直接发射路径没有图，就地读回，行为和以前完全一致。
        if (self.graphActive(g)) return 0;
        self.sampled = self.collectSampledToken(sp);
        return self.sampled;
    }

    /// 本步 `sample` 产出的 token，在 `stepEnd` 把它下载回来之后可用。

    // ==== slots: the two state machines (P2c) ===============================
    //
    // These are the two slots that P2c exists for.  Until KV and the GDN state
    // became backend regions they COULD NOT be device slots at all: their
    // arguments are large slices of state (a 64 KB GDN state per head, up to
    // `slot+1` KV rows per layer), so a staging backend would have had to move
    // megabytes per launch -- orders of magnitude worse than leaving them on
    // the CPU.  With both resident, the launches are free of traffic.
    //
    // The numeric contract is in the kernel headers (src/kernels/cuda/gdn.cu,
    // attn.cu): accumulation order follows the CPU reference element by
    // element, and neither is bit-exact (FMA contraction on the device side).

    fn dGdnStep_body(g: backend.Graph, ctx: *anyopaque, state: backend.Buf, k: backend.Buf, v: backend.Buf,
                q: backend.Buf, km: backend.Buf, delta: backend.Buf, out: backend.Buf,
                gamma: f32, beta: f32, vd: usize) void {
        const self = self_(ctx);
        if (graphGate(g)) return;
        // vd != 128 is the CPU `gdnFused`'s slow path (four separate kernels);
        // the device kernel is written for the fast path only, so route the odd
        // shape to the CPU rather than launching a kernel that would refuse.
        if (vd == 128 and self.cu != null and
            state.len >= vd * vd and k.len >= vd and q.len >= vd and
            v.len >= vd and km.len >= vd and delta.len >= vd and out.len >= vd)
        {
            if (dBufPtr(state)) |pstate| {
                if (dBufPtr(k)) |pk| {
                    if (dBufPtr(v)) |pv| {
                        if (dBufPtr(q)) |pq| {
                            if (dBufPtr(km)) |pkm| {
                                if (dBufPtr(delta)) |pdelta| {
                                    if (dBufPtr(out)) |pout| {
                                        const rc = Launch.gdn_step(g, pstate, pk, pv, pq, pkm, pdelta, pout, gamma, beta, @intCast(vd), self.launchStream());
                                        if (rc == cudart.cudaSuccess) return;
                                        self.ewLaunchFail("gdn_step", rc);
                                    } else return;
                                } else return;
                            } else return;
                        } else return;
                    } else return;
                } else return;
            } else return;
        }
        cpu_mod.CpuBackend.vtable.gdnStep(.{}, @ptrCast(&self.cpu), state, k, v, q, km, delta, out, gamma, beta, vd);
    }

    /// Batched `dGdnStep_body`: one launch for all `heads`, otherwise the same
    /// per-head call.  Shaped like `dRmsnormRows_body` / `dL2normRows_body`:
    /// check every precondition, launch, and on launch failure log once and
    /// fall through to the CPU reference -- which for this slot is the per-head
    /// loop, so a fallback costs `heads` host calls, never a wrong answer.
    ///
    /// The precondition set is the batch's geometry, not a heuristic: the
    /// per-head views the kernel computes must lie inside the buffers the
    /// caller handed us (a head whose slice overruns would read a neighbouring
    /// head's state, which is silent garbage rather than a fault).
    fn dGdnStepRows_body(g: backend.Graph, ctx: *anyopaque, state: backend.Buf, k: backend.Buf, v: backend.Buf,
                         q: backend.Buf, km: backend.Buf, delta: backend.Buf, out: backend.Buf,
                         gamma: backend.Buf, beta: backend.Buf,
                         heads: usize, nk: usize, kd: usize, vd: usize) void {
        const self = self_(ctx);
        if (graphGate(g)) return;
        if (vd == 128 and kd == 128 and heads != 0 and nk != 0 and nk <= heads and
            gamma.len >= heads and beta.len >= heads and
            state.len >= heads * kd * vd and k.len >= nk * kd and q.len >= nk * kd and
            v.len >= heads * vd and out.len >= heads * vd and
            km.len >= heads * vd and delta.len >= heads * vd and
            self.cu != null)
        {
            if (dBufPtr(state)) |pstate| {
                if (dBufPtr(k)) |pk| {
                    if (dBufPtr(v)) |pv| {
                        if (dBufPtr(q)) |pq| {
                            if (dBufPtr(km)) |pkm| {
                                if (dBufPtr(delta)) |pdelta| {
                                    if (dBufPtr(out)) |pout| {
                                        // gamma/beta are backend regions
                                        // (`gdnGamma`'s output and its in-place
                                        // sigmoid(b) write-back), so the kernel
                                        // reads them where they already live:
                                        // no H2D, no D2H, nothing per token.
                                        // They must NOT be cached as read-only
                                        // weights (pointer-keyed, upload-once):
                                        // their contents change per layer per
                                        // token, which would silently freeze
                                        // layer 0's coefficients for the run.
                                        if (dBufPtr(gamma)) |pgamma| {
                                            if (dBufPtr(beta)) |pbeta| {
                                                const rc = Launch.gdn_step_rows(g, 
                                                    pstate, @intCast(kd * vd),
                                                    pk, @intCast(kd),
                                                    pv, @intCast(vd),
                                                    pq, @intCast(kd),
                                                    pkm, pdelta, pout,
                                                    pgamma, pbeta,
                                                    @intCast(heads), @intCast(nk),
                                                    @intCast(kd), @intCast(vd), self.launchStream(),
                                                );
                                                if (rc == cudart.cudaSuccess) return;
                                                if (self.capturing(g)) self.graphAbandon(g, "gdn_step_rows launch failed");
                                                self.ewLaunchFail("gdn_step_rows", rc);
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        cpu_mod.CpuBackend.vtable.gdnStepRows(.{}, @ptrCast(&self.cpu), state, k, v, q, km, delta, out, gamma, beta, heads, nk, kd, vd);
    }

    fn dGdnGamma_body(g: backend.Graph, ctx: *anyopaque, a_raw: backend.Buf, b_raw: backend.Buf, ssm_a: backend.Buf,
                 dt_bias: backend.Buf, gamma: backend.Buf, nv: usize) void {
        const self = self_(ctx);
        if (graphGate(g)) return;
        // Every operand here is a backend region.  The two weight rows
        // (`ssm_a`, `dt_bias`) are per-layer constants and, like the norm
        // weights, are device `Buf`s uploaded exactly once at load time
        // (`loadVecF32`); the rest are live per-token regions.
        if (self.cu != null and nv != 0 and
            a_raw.len >= nv and b_raw.len >= nv and ssm_a.len >= nv and
            dt_bias.len >= nv and gamma.len >= nv)
        {
            if (dBufPtr(a_raw)) |pa| {
                if (dBufPtr(b_raw)) |pb| {
                    if (dBufPtr(gamma)) |pg| {
                        if (dBufPtr(ssm_a)) |sa| {
                            if (dBufPtr(dt_bias)) |db| {
                                const rc = Launch.gdn_gamma(g, pa, pb, sa, db, pg, @intCast(nv), self.launchStream());
                                if (rc == cudart.cudaSuccess) return;
                                self.ewLaunchFail("gdn_gamma", rc);
                            } else return;
                        } else return;
                    } else return;
                } else return;
            } else return;
        }
        cpu_mod.CpuBackend.vtable.gdnGamma(.{}, @ptrCast(&self.cpu), a_raw, b_raw, ssm_a, dt_bias, gamma, nv);
    }

    /// CUDA form of `VTable.gdnPrefillCore`: the whole chunk's conv+silu, gamma /
    /// beta and l2norm/scale passes in FOUR launches instead of `tokens` x ~14.
    ///
    /// Order matters and is preserved: conv first (token t's taps need what
    /// token t-1 shifted in -- the kernel walks the tokens ascending inside one
    /// thread per channel), then gamma (reads the RAW a/b projections, which conv
    /// does not touch), then the norm/scale passes (read the conv output).
    /// `gdnStepRows` afterwards only READS qkv, so hoisting all of this ahead of
    /// the state loop is order-preserving, not just commute-safe.
    ///
    /// Falls back to the CPU body when any region is not device-resident or a
    /// launch fails -- same policy as every other forwarder here (numbers stay
    /// right, only the speedup is lost).
    fn dGdnPrefillCore_body(g: backend.Graph, ctx: *anyopaque, qkv: backend.Buf, conv: backend.Buf,
                            wconv: backend.Buf, ab: backend.Buf, ssm_a: backend.Buf,
                            dt_bias: backend.Buf, gamma: backend.Buf, tokens: usize,
                            conv_dim: usize, ab_a_offset: usize, ab_b_offset: usize,
                            ab_stride: usize, gamma_stride: usize, kq_dim: usize,
                            nv: usize, nk: usize, kd: usize, conv_k: usize, eps: f32) void {
        const self = self_(ctx);
        const stream = self.launchStream();
        const nk_reps = if (nk != 0) nv / nk else 0;
        const qscale: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(kd)));

        // Shape gates: conv is fixed at 4 taps (elementwise.zig `convStepKernel`
        // asserts it), the q/k runs must be `nk` whole `kd` rows, and the tiled
        // layout is what makes the batched norm legal at all.
        const shaped = tokens > 0 and nv > 0 and nk > 0 and kd > 0 and
            conv_k == 4 and nv % nk == 0 and nk_reps >= 1 and
            kq_dim == nk * kd and
            conv_dim >= 2 * kq_dim and
            qkv.len >= tokens * conv_dim and
            ab.len >= @max(ab_a_offset, ab_b_offset) + tokens * ab_stride and
            gamma.len >= tokens * gamma_stride;
        if (self.cu != null and shaped) {
            if (dBufPtr(qkv)) |pq| {
                if (dBufPtr(conv)) |pc| {
                    if (dBufPtr(wconv)) |pw| {
                        if (dBufPtr(ab)) |pab| {
                            if (dBufPtr(ssm_a)) |sa| {
                                if (dBufPtr(dt_bias)) |db| {
                                    if (dBufPtr(gamma)) |pg| {
                                        // 1. conv + silu for every token, one shared history.
                                        var rc = Launch.conv_step_rows(g, 
                                            pc, pq, pw, @intCast(conv_dim), @intCast(tokens),
                                            @intCast(conv_dim), stream);
                                        if (rc != cudart.cudaSuccess) {
                                            self.ewLaunchFail("conv_step_rows", rc);
                                            return;
                                        }
                                        // 2. beta (in place) + gamma, one row per token.
                                        rc = Launch.gdn_gamma_rows(g, 
                                            pab + ab_a_offset, pab + ab_b_offset, sa, db, pg,
                                            @intCast(nv), @intCast(tokens),
                                            @intCast(ab_stride), @intCast(ab_stride),
                                            @intCast(gamma_stride), stream);
                                        if (rc != cudart.cudaSuccess) {
                                            self.ewLaunchFail("gdn_gamma_rows", rc);
                                            return;
                                        }
                                        // 3. the q/k norm + q scale pass, ONCE.
                                        //
                                        // The `nk_reps` repetition this used to do
                                        // is wrong, and the CPU path
                                        // (`backend_cpu.vGdnPrefillCore`) never had
                                        // it: `l2norm` is NOT idempotent (a second
                                        // pass divides by `sqrt(1 + eps)` again), so
                                        // repeating it `nv/nk` times (3 on the 27B)
                                        // silently rescales q/k on every GDN layer.
                                        // The reference normalises a
                                        // `(kd, nk)` view ONCE and only then
                                        // broadcasts q/k up to `nv` heads
                                        // (llama.cpp `qwen3next.cpp:503-528`), which
                                        // is what the CPU oracle models.
                                        {
                                            rc = Launch.l2norm_strided(g, 
                                                pq, 0, @intCast(nk), @intCast(kd), eps,
                                                @intCast(tokens), @intCast(conv_dim), stream);
                                            if (rc != cudart.cudaSuccess) {
                                                self.ewLaunchFail("l2norm_strided_q", rc);
                                                return;
                                            }
                                            rc = Launch.l2norm_strided(g, 
                                                pq, @intCast(kq_dim), @intCast(nk), @intCast(kd),
                                                eps, @intCast(tokens), @intCast(conv_dim), stream);
                                            if (rc != cudart.cudaSuccess) {
                                                self.ewLaunchFail("l2norm_strided_k", rc);
                                                return;
                                            }
                                            rc = Launch.scale_vec_strided(g, 
                                                pq, 0, qscale, @intCast(kq_dim),
                                                @intCast(tokens), @intCast(conv_dim), stream);
                                            if (rc != cudart.cudaSuccess) {
                                                self.ewLaunchFail("scale_vec_strided", rc);
                                                return;
                                            }
                                        }
                                        return;
                                    } else return;
                                } else return;
                            } else return;
                        } else return;
                    } else return;
                } else return;
            } else return;
        }
        cpu_mod.CpuBackend.vtable.gdnPrefillCore(
            .{},
            @ptrCast(&self.cpu), qkv, conv, wconv, ab, ssm_a, dt_bias, gamma, tokens,
            conv_dim, ab_a_offset, ab_b_offset, ab_stride, gamma_stride, kq_dim,
            nv, nk, kd, conv_k, eps);
    }

    /// The `kv_kind` the attention kernels decode the cache with: -1 for f32
    /// (the original path), else the `kind` the KV quantizer was launched with
    /// (0=f16, 1=q8_0, 2=q4_0).
    fn kvKind(self: *CudaBackend) c_int {
        return switch (self.kv_type) {
            .f32 => -1,
            .f16 => 0,
            .q8_0 => 1,
            .q4_0 => 2,
        };
    }

    fn dAttnDecode_body(g: backend.Graph, ctx: *anyopaque, q: backend.Buf, gate: backend.Buf, kcache: backend.Buf,
                   vcache: backend.Buf, att: backend.Buf, gout: backend.Buf, slot: usize,
                   heads: usize, gqa: usize, hd: usize, kv_dim: usize, scale: f32) void {
        const self = self_(ctx);
        if (graphGate(g)) return;
        // `kcache`/`vcache` 是本层 cache 的前 `slot+1` 行，所以它们的尺寸是
        // (slot+1)*kv_dim 而不是 max_seq*kv_dim（引擎传的就是这段跨度）。`att`
        // 需要 slot+1 个 float。
        //
        // `heads * hd <= gout.len` 与 `q.len >= heads*hd` 是引擎的契约；违反就是
        // bug，CPU 参照在切片时会直接陷阱，所以宁可落到它那里，也不要在错误的
        // 尺寸上发射。
        const nrows = slot + 1;
        // cache 区域是**字节分配的 f32 视图**，所以一行打包数据在这里只有
        // `kvRowBytes/4` 个元素宽。拿 `nrows * kv_dim` 去比会把每一种量化 cache
        // 都判为不合格、落进 CPU 参照，而它会去读**设备**指针 —— 这就是 2026-11
        // 那次 prefill 段错误（q8_0，slot0=1024）。`kv_dim` 本身仍是 kernel 的
        // 逻辑行宽，只有长度检查这一处做换算。
        const cache_row_elems: usize = (backend.kvRowBytes(kv_dim, self.kv_type) + 3) / 4;
        if (self.cu != null and heads != 0 and hd != 0 and
            q.len >= heads * hd and gout.len >= heads * hd and
            att.len >= nrows and kcache.len >= nrows * cache_row_elems and vcache.len >= nrows * cache_row_elems)
        {
            if (dBufPtr(q)) |pq| {
                if (dBufPtr(gate)) |pgate| {
                    if (dBufPtr(kcache)) |pk| {
                        if (dBufPtr(vcache)) |pv| {
                            if (dBufPtr(att)) |patt| {
                                if (dBufPtr(gout)) |pgout| {
                                    // 只要这次发射可能被捕获（或正在被重放），就走
                                    // 设备 slot 路径。那条路上节点的实参和它的
                                    // `shared_mem_bytes` 都是常量，所以重放永远不需要
                                    // 为它调 `GraphExecKernelNodeSetParams`。27B 实测：
                                    // 每层每 token 重定向这一个节点要花掉 33.8 ms
                                    // 里的 14.3 ms（43%），而 kernel 实际只跑 1.4 ms。
                                    // 标量 `slot` 形态留给直接发射路径 —— 那里没有
                                    // 节点可重定向，较小的 shared 块是白拿的。
                                    // 判定必须覆盖**两套**机制：旧的 `state` 与
                                    // XG 的 `xd_phase`。只认 `state` 时，XG 的
                                    // `.live` 重放会落进标量 `slot` 分支 —— 那一
                                    // 分支把 slot 当 kernel 实参，录进图后每次重放
                                    // 都按录制那一步的 KV 长度算 attention。
                                    const want_dyn = graphWanted() and
                                        (self.gs(0).state == .live or self.gs(0).state == .rec or self.graphActive(g));
                                    if (want_dyn) {
                                        if (self.ensureCtl()) |pdyn| {
                                            // **不要**在这里发布 `slot`。capture 期间
                                            // `launchStream()` 就是 capture 流，此时
                                            // 发出的发布会变成一个图节点，把这个 token
                                            // 的值烧进它的实参 —— 而每次重放那个节点都
                                            // 会把新写的值覆盖掉，等于把整张图钉死在录制
                                            // 那个 token 的 KV 长度上。（实测：前 6 个
                                            // token 一致，正文恰好在重放开始时发散，
                                            // 而设备控制块读回来的正是录制时的值。）
                                            // 发布由 `dGraphBegin` 负责，它在
                                            // `GraphLaunch` 之前、图外把控制块写好。
                                            const sslot = @as([*]c_int, @ptrCast(@alignCast(pdyn)));
                                            // 启用时优先走切分版设备 slot 路径：四个
                                            // 实参恒定的节点，仍然零重定向，而每个
                                            // head 的行遍历摊到 `split` 个 block 上，
                                            // 而不是一个。
                                            if (self.attn_split > 1) {
                                                if (self.d_att_part) |ppart| {
                                                    const s2 = @as([*]f32, @ptrCast(@alignCast(ppart)));
                                                    const pprob = if (self.d_att_prob) |pp| @as([*]f32, @ptrCast(@alignCast(pp))) else patt;
                                                    const rcs = Launch.attn_decode_split_dyn(g, pq, pgate, pk, pv, pprob, pgout, s2, sslot, @intCast(heads), @intCast(gqa), @intCast(hd), @intCast(kv_dim), scale, @intCast(self.attn_split), self.kvKind(), self.launchStream());
                                                    if (rcs == cudart.cudaSuccess) {
                                                        // 四个节点（score、softmax、
                                                        // PV、reduce+gate），都属于
                                                        // `.attn_dyn` 类：没有一个会被
                                                        // 重定向。节点普查必须看到全部
                                                        // 四个。
                                                        return;
                                                    }
                                                    if (self.capturing(g)) self.graphAbandon(g, "attn_decode split_dyn launch failed");
                                                    self.ewLaunchFail("attn_decode_split_dyn", rcs);
                                                }
                                            }
                                            const rc = Launch.attn_decode_dyn(g, pq, pgate, pk, pv, patt, pgout, sslot, @intCast(heads), @intCast(gqa), @intCast(hd), @intCast(kv_dim), scale, self.kvKind(), self.launchStream());
                                            if (rc == cudart.cudaSuccess) {
                                                // 只计数、不记录：该节点被归为
                                                // `.attn_dyn` 且从不重定向，但节点普查
                                                // 要求记账一致。
                                                return;
                                            }
                                            if (self.capturing(g)) self.graphAbandon(g, "attn_decode dyn launch failed");
                                            self.ewLaunchFail("attn_decode_dyn", rc);
                                        }
                                    }
                                    // 逐 token 变化的只有 `slot`：cache 的**指针**是固定
                                    // 区域，引擎只是交出一个更长的前缀，所以只要把新
                                    // slot 告诉 kernel，捕获下来的节点就一直有效。
                                    // 这就是那唯一的一次重定向。
                                    //
                                    // 设了 `$QWEN35_CUDA_ATTN_SPLIT` 时改走按行切分的
                                    // 变体：它把**两条**行遍历都摊到每 head 那么多
                                    // block 上 —— 因为每 head 单 block 在 114 SM 上只有
                                    // 24 个 block（21%），此时每个 lane 串行走完 `n`
                                    // 全部行（KV cache 不在 L2 时实测 33 GB/s）。
                                    if (self.attn_split > 1) {
                                        if (self.d_att_part) |ppart| {
                                            const s2 = @as([*]f32, @ptrCast(@alignCast(ppart)));
                                            // 用 `d_att_prob` 而**不是** `patt`：见该
                                            // 字段的说明。
                                            const pprob = if (self.d_att_prob) |pp| @as([*]f32, @ptrCast(@alignCast(pp))) else patt;
                                            const rc2 = Launch.attn_decode_split2(g, pq, pgate, pk, pv, pprob, pgout, s2, @intCast(slot), @intCast(heads), @intCast(gqa), @intCast(hd), @intCast(kv_dim), scale, @intCast(self.attn_split), self.kvKind(), self.launchStream());
                                            if (rc2 == cudart.cudaSuccess) return;
                                            self.ewLaunchFail("attn_decode split", rc2);
                                        }
                                    }
                                    const rc = Launch.attn_decode(g, pq, pgate, pk, pv, patt, pgout, @intCast(slot), @intCast(heads), @intCast(gqa), @intCast(hd), @intCast(kv_dim), scale, self.kvKind(), self.launchStream());
                                    if (rc == cudart.cudaSuccess) {
                                        return;
                                    }
                                    // 失败的发射不能进图：否则捕获下来的图会缺掉整整
                                    // 一层的 attention。宁可放弃，也不要实例化。
                                    if (self.capturing(g)) self.graphAbandon(g, "attn_decode launch failed");
                                    self.ewLaunchFail("attn_decode", rc);
                                } else return;
                            } else return;
                        } else return;
                    } else return;
                } else return;
            } else return;
        }
        cpu_mod.CpuBackend.vtable.attnDecode(.{}, @ptrCast(&self.cpu), q, gate, kcache, vcache, att, gout, slot, heads, gqa, hd, kv_dim, scale);
    }

    // rope64：两条路径，由 `$QWEN35_CUDA_ROPE` 在运行期选择。
    //
    //   host   (DEFAULT, unchanged from before) -- D2H, `elementwise.zig`
    //          `applyRope64` on the CPU, H2D.  The one remaining D2H+H2D pair on
    //          the per-token path: a head's 64 floats, twice, per q/k tensor per
    //          layer (~64 KB per token for the 0.8B: 48 rotations x 64 x 4 B x
    //          2 directions).  Bit-identical to the CPU backend *by
    //          construction*, because it IS the CPU backend.  These trips are
    //          counted in `stat_rope_trips` so `CUDA-M` shows the A/B really
    //          switched.
    //   device -- `q35_launch_rope64` (`src/kernels/cuda/rope.cu`), zero copy,
    //          `stat_rope_trips` stays 0.  NOT bit-exact; by how much is the
    //          number this section used to guess at.
    //
    // HISTORY (the reason `host` is the default): a device kernel was written,
    // measured and withdrawn.  What that measurement established:
    //
    //   The phase is `theta = pos * base^(-2j/64)`, an ABSOLUTE angle of order
    //   `pos`.  glibc's `powf` and the device runtime's differ by up to 1 ULP
    //   *relative to theta*, which at pos = 1e6 is ~1e-2 radians of phase --
    //   and `cos(theta)` then differs by up to 1e-2 outright, i.e. 5e-4
    //   relative on the rotated value (measured then: max relative error
    //   5.2e-4, max abs 2.5e-5 over 200 positions x 256-element heads).
    //   Recomputing theta in double precision on the device does NOT fix it
    //   (measured: still 5.2e-4) because the host's f32 `powf` is itself the
    //   lower-precision side and cannot be matched, only approximated.
    //
    // What that measurement did NOT establish is the sentence that followed it
    // ("far past what the token stream tolerates"): that was an inference, not
    // a test, and `pos` in that sweep went far past anything `Engine.max_seq`
    // (= 4096) can reach.  The switch exists so the claim can be measured on
    // the real token stream instead of argued about: run the same prompt with
    // `QWEN35_CUDA_ROPE=host` and `=device` and diff the token ids.
    //
    // If this ever goes zero-copy by default, the fix is to accept the drift
    // deliberately (with the measured number), NOT to rewrite the reference.
    /// Lazily allocate the two-int {pos, base} channel for `k_rope64_dpos`.
    /// Returns the DEVICE address the kernel reads, or null on failure (caller
    /// then falls back to the constant-argument kernel, so a missing 8 bytes
    /// degrades to "retarget per token" rather than to wrong math).
    ///
    /// The two rejected designs are worth recording, because both look right:
    ///   1. `cudaMalloc` + `cudaMemcpyAsync(H2D)` on the launch stream -- a
    ///      host-to-device copy is not capturable, so the first captured
    ///      `rope64` failed and the whole decode graph was abandoned
    ///      (`rope64 launch failed: operation failed due to a previous error
    ///      during capture`).
    ///   2. `cudaHostAlloc(Mapped)` + CPU stores -- no copy and capture-safe,
    ///      but every rope kernel then reads the value over PCIe: measured
    ///      26.9 -> 12.2 tok/s on the 27B.
    /// What works is a plain DEVICE buffer written by a 1-thread kernel on the
    /// launch stream (`k_rope64_setenv`), which captures like any other node and
    /// costs one launch per token instead of 448 retargets.

    /// Lazily allocate the device int that carries the decode `slot` to
    /// `q35_attn_decode_kernel_dyn`.  Returns null only if the allocation fails.

    /// Publish this token's `{pos, base}` into the device channel, but only when
    /// it actually changed (the engine calls `rope64` once per head, so the same
    /// token would otherwise publish 448 times).  Returns true if it launched the
    /// writer, i.e. if the caller must not assume the values are already there.

    fn dRope64_body(g: backend.Graph, ctx: *anyopaque, x: backend.Buf, pos: u32, base: f32) void {
        const self = self_(ctx);
        if (graphGate(g)) return;
        const max_head: usize = 256;
        var host: [max_head]f32 = undefined;
        const n = @min(x.len, max_head);
        if (self.cu == null or n == 0 or n != x.len) {
            // Same length guard as the rest: rope is a per-head call (hd = 128
            // or 64 in every Qwen3.5 variant), so anything larger means the
            // engine's shape changed and the CPU reference must own it.
            cpu_mod.CpuBackend.vtable.rope64(.{}, @ptrCast(&self.cpu), x, pos, base);
            return;
        }
        if (ropeOnDevice()) {
            // The launcher works on `heads * hd`; `x` is exactly one head, so
            // heads = 1 and hd = n.  The engine calls this once per head.
            if (dBufPtr(x)) |px| {
                // BOTH `pos` and `base` are variables of the decode step, not
                // constants: `pos` advances one per token, and `base` is the
                // context-dependent theta the engine derives per step (it
                // switches once the position passes the model's
                // `rope_freqs`/scaling threshold).  A captured node carries a
                // snapshot of both, so both are recorded here and retargeted
                // by `graphReplay` -- retargeting only `pos` rotates q/k with
                // the capture token's base, which is a wrong angle on every
                // replay (and if the swap happens to occur *during* capture it
                // is wrong from the first replayed token).
                // DEVICE-SIDE `pos` (see `k_rope64_dpos` in rope.cu).  The
                // kernel reads {pos, base} out of a two-int device buffer, so
                // the node's parameters never change and the decode graph can
                // replay 448 rope nodes on a 27B without a single
                // `cudaGraphExecKernelNodeSetParams`.  Measured on the 27B
                // IQ3_XXS: retargeting those 448 nodes per token was the
                // largest host cost in decode -- 17.5 `cuEventSynchronize` and
                // ~1.9k libcuda calls per token, with the GPU idle 64% of the
                // wall time waiting for the host to hand it the next token.
                //
                // The buffer is written with an ASYNC copy on the same stream,
                // never a blocking one: a blocking `cuMemcpy` inside an open
                // capture invalidates the graph (`cudaErrorStreamCaptureUnsupported`),
                // which is what `copyFromHost` has to guard against.
                if (self.ensureCtl()) |ctl_dev| {
                    // `pos` / `base` reach `k_rope64_dpos` through the shared
                    // per-step block, published once in `graphBegin`.  The
                    // engine still passes them per call, so a step that changes
                    // the theta mid-way (the rope-scaling threshold) publishes
                    // again here -- `publishCtl` compares against the host
                    // mirror, so the 448 per-head calls cost one launch in
                    // total.
                    _ = self.publishCtl(.{
                        .pos = pos,
                        .rope_base_bits = @bitCast(base),
                        .kv_slot = self.ctl_host.kv_slot,
                        .attn_slot = self.ctl_host.attn_slot,
                        .rows = self.ctl_host.rows,
                    }, self.launchStream(), false);
                    const rc = Launch.rope64_dpos(g, px, 1, @intCast(n), ctl_dev, self.launchStream());
                    if (rc == cudart.cudaSuccess) {
                        self.stat_rope_devices += 1;
                        return;
                    }
                    if (self.capturing(g)) self.graphAbandon(g, "rope64_dpos launch failed");
                    self.ewLaunchFail("rope64_dpos", rc);
                }
                const rc = Launch.rope64(g, px, 1, @intCast(n), @bitCast(pos), base, self.launchStream());
                if (rc == cudart.cudaSuccess) {
                    self.stat_rope_devices += 1;
                    return;
                }
                if (self.capturing(g)) self.graphAbandon(g, "rope64 launch failed");
                self.ewLaunchFail("rope64", rc);
            }
            // No device pointer (or the launch failed): the CPU path below is
            // still correct, so fall through to it rather than leave `x`
            // unrotated.  `stat_rope_trips` then counts this head as the host
            // trip it actually was.
        }
        CudaBackend.dCopyToHost(.{}, @ptrCast(self), host[0..n], x);
        cpu_mod.CpuBackend.vtable.rope64(.{}, @ptrCast(&self.cpu), backend.Buf.wrap(host[0..n]), pos, base);
        CudaBackend.dCopyFromHost(.{}, @ptrCast(self), x, host[0..n]);
        // One H2D + one D2H per rotated head, per position.  Counted so the
        // report can state this remainder's exact share instead of guessing.
        self.stat_rope_trips += 1;
    }

    /// Row-batched `rope64` for prefill: ONE launch for `rows` tokens x `heads`
    /// heads instead of `rows * heads` launches.  The engine's prefill loop used
    /// to issue 28 per-token calls per full-attention layer (24 q heads + 4 kv
    /// heads); `k_rope64_rows` indexes rows and heads off a single flat block
    /// id and rotates row r at `pos + r`, so row r is the same arithmetic as the
    /// single-row call it replaces.
    ///
    /// NEVER used during a decode capture: this slot would add a node the graph
    /// cannot retarget (it matches `q35_func_rope64` node-by-node), so it bails
    /// out to the per-row loop when a capture is live.  Prefill does not capture.
    fn dRope64Rows_body(g: backend.Graph, ctx: *anyopaque, x: backend.Buf, pos: u32, base: f32, rows_in: usize, heads_in: usize, hd_in: usize) void {
        const self = self_(ctx);
        if (graphGate(g)) return;
        const row_len = heads_in * hd_in;
        if (!ropeOnDevice() or
            heads_in == 0 or hd_in < 64 or rows_in == 0 or
            row_len == 0 or x.len != rows_in * row_len or self.cu == null)
        {
            cpu_mod.CpuBackend.vtable.rope64Rows(.{}, @ptrCast(&self.cpu), x, pos, base, rows_in, heads_in, hd_in);
            return;
        }
        // Under a capture this must be the DEVICE-position variant: the graph
        // retargets a rope node by its host-stub address, and only the `_dpos`
        // kernels have one registered (`q35_func_rope64_dpos` /
        // `q35_func_rope64_rows_dpos`).  This used to bail to the CPU here,
        // which is why decode never batched rope at all and paid one launch per
        // head -- see `k_rope64_rows_dpos`'s comment in rope.cu.
        //
        // `pos`/`base` are published into the same device channel the per-head
        // path uses (`ropeSetEnv` runs once per token, not once per call), so
        // the captured node reads them fresh on every replay.
        if (self.graphActive(g)) {
            if (self.ensureCtl()) |ctl_dev| {
                _ = self.publishCtl(.{
                    .pos = pos,
                    .rope_base_bits = @bitCast(base),
                    .kv_slot = self.ctl_host.kv_slot,
                    .attn_slot = self.ctl_host.attn_slot,
                    .rows = self.ctl_host.rows,
                }, self.launchStream(), false);
                if (dBufPtr(x)) |px| {
                    const rc = Launch.rope64_rows_dpos(g, px, @intCast(rows_in), @intCast(heads_in), @intCast(hd_in), ctl_dev, self.launchStream());
                    if (rc == cudart.cudaSuccess) {
                        return;
                    }
                    if (self.capturing(g)) self.graphAbandon(g, "rope64_rows_dpos launch failed");
                    self.ewLaunchFail("rope64_rows_dpos", rc);
                }
            }
            cpu_mod.CpuBackend.vtable.rope64Rows(.{}, @ptrCast(&self.cpu), x, pos, base, rows_in, heads_in, hd_in);
            return;
        }
        if (dBufPtr(x)) |px| {
            const rc = Launch.rope64_rows(g, px, @intCast(rows_in), @intCast(heads_in), @intCast(hd_in), @bitCast(pos), base, self.launchStream());
            if (rc == cudart.cudaSuccess) return;
            self.ewLaunchFail("rope64_rows", rc);
        }
        cpu_mod.CpuBackend.vtable.rope64Rows(.{}, @ptrCast(&self.cpu), x, pos, base, rows_in, heads_in, hd_in);
    }

    fn dConvStep_body(g: backend.Graph, ctx: *anyopaque, conv: backend.Buf, y: backend.Buf, wconv: backend.Buf, k: usize) void {
        const self = self_(ctx);
        if (graphGate(g)) return;
        // `conv` is the GDN sliding window and is BOTH read and written (the
        // kernel shifts it in place); `y` likewise (read as the 4th tap, then
        // overwritten).  Both are device-resident, so both are passed directly.
        //
        // `k` is the conv width and the kernel hardcodes 4 taps (as does
        // elementwise.zig `convStepKernel`, which asserts it).  A model with a
        // different window must reach the CPU rather than get a wrong answer.
        if (k != 4) {
            std.debug.print("cuda backend: convStep kernel width {d} != 4; using the CPU path\n", .{k});
            cpu_mod.CpuBackend.vtable.convStep(.{}, @ptrCast(&self.cpu), conv, y, wconv, k);
            return;
        }
        if (conv.len == 3 * y.len and wconv.len == 4 * y.len and self.cu != null) {
            if (dBufPtr(conv)) |pc| {
                if (dBufPtr(y)) |py| {
                    if (dBufPtr(wconv)) |pw| {
                        const rc = Launch.conv_step(g, pc, py, pw, @intCast(y.len), self.launchStream());
                        if (rc == cudart.cudaSuccess) return;
                        self.ewLaunchFail("conv_step", rc);
                    } else return;
                } else return;
            } else return;
        }
        cpu_mod.CpuBackend.vtable.convStep(.{}, @ptrCast(&self.cpu), conv, y, wconv, k);
    }

    // ==== profiling / introspection =========================================

    /// Bytes of IQ codebook currently on the device (0 when the upload has not
    /// happened).  Asked of the kernel object rather than cached in Zig so the
    /// number can never disagree with what the kernels actually index.
    fn iqTablesBytes(self: *CudaBackend) u64 {
        if (self.iq_tables_state != .done) return 0;
        return @intCast(Launch.tables_bytes());
    }

    // ---- GPU-side timing helpers ($QWEN35_PROF only) -----------------------
    //
    // Every entry point is a no-op unless `ev_on` is set, and `ev_on` is only
    // set from `dProfReset`, which the engine calls once before prefill.  So the
    // default path checks a bool and returns: no event is created, recorded or
    // waited on, and no stream is drained.

    /// Create the four events once.  Idempotent; a failure disables the whole
    /// facility rather than silently reporting zeros, because a zero GPU time
    /// would look like "the GPU is idle" -- the opposite of the truth.
    fn ensureEvents(self: *CudaBackend) bool {
        if (self.ev_created) return true;
        const lib = if (self.cu) |*cu| &cu.lib else return false;
        var made: [8]?*anyopaque = .{ null, null, null, null, null, null, null, null };
        const names = [_][:0]const u8{ "ev_bk_a", "ev_bk_b", "ev_pre_a", "ev_pre_b", "ev_arg_a", "ev_arg_b", "ev_tok_end", "ev_prev_tok_end" };
        for (&made, names) |*slot, nm| {
            var h: ?*anyopaque = null;
            // `cudaEventDefault` (flags 0): the default deliberately keeps host
            // sync enabled, which `EventSynchronize`/`ElapsedTime` need.
            if (lib.EventCreate(&h, 0) != cudart.cudaSuccess or h == null) {
                // Bounded cleanup: destroy whatever was already made so a
                // partial setup cannot leak into the next attempt.
                for (made) |m| {
                    if (m) |p| _ = lib.EventDestroy(p);
                }
                std.debug.print("cuda backend: $QWEN35_PROF event timing disabled ({s} create failed)\n", .{nm});
                return false;
            }
            slot.* = h;
        }
        self.ev_bk_a = made[0];
        self.ev_bk_b = made[1];
        self.ev_pre_a = made[2];
        self.ev_pre_b = made[3];
        self.ev_arg_a = made[4];
        self.ev_arg_b = made[5];
        self.ev_tok_end = made[6];
        self.ev_prev_tok_end = made[7];
        // Per-call interval ring (see `ev_pool`).  Created here so a failure
        // disables profiling for the whole run instead of half of it.
        for (&self.ev_pool, 0..) |*slot, i| {
            var h: ?*anyopaque = null;
            if (lib.EventCreate(&h, 0) != cudart.cudaSuccess or h == null) {
                for (self.ev_pool[0..i]) |q| {
                    if (q) |pp| _ = lib.EventDestroy(pp);
                }
                self.ev_pool = @splat(null);
                std.debug.print("cuda backend: $QWEN35_PROF event timing disabled (ev_pool create failed)\n", .{});
                return false;
            }
            slot.* = h;
        }
        self.ev_pool_wr = 0;
        self.ev_pending_n = 0;
        self.ev_created = true;
        return true;
    }

    fn freeEvents(self: *CudaBackend) void {
        if (!self.ev_created) return;
        const lib = if (self.cu) |*cu| &cu.lib else null;
        if (lib) |l| {
            for (self.ev_pool) |ep| {
                if (ep) |pp| _ = l.EventDestroy(pp);
            }
            for ([_]?*anyopaque{ self.ev_bk_a, self.ev_bk_b, self.ev_pre_a, self.ev_pre_b, self.ev_arg_a, self.ev_arg_b, self.ev_tok_end, self.ev_prev_tok_end }) |p| {
                if (p) |q| _ = l.EventDestroy(q);
            }
        }
        self.ev_bk_a = null;
        self.ev_bk_b = null;
        self.ev_pre_a = null;
        self.ev_pre_b = null;
        self.ev_arg_a = null;
        self.ev_arg_b = null;
        self.ev_tok_end = null;
        self.ev_prev_tok_end = null;
        self.ev_pool = @splat(null);
        self.ev_pool_wr = 0;
        self.ev_pending_n = 0;
        self.ev_created = false;
    }

    /// True when the caller should record events at all.  Deliberately reads
    /// the env var as well: `profReset` is the only intended setter, but a
    /// second line of defence keeps a stray `ev_on = true` from turning into a
    /// default-path sync on a build where PROF is off.
    fn gpuTimingOn(self: *const CudaBackend) bool {
        return self.ev_on and self.cu != null;
    }

    /// Start timing one `backend.Exec` slot; returns the clock origin to hand
    /// to `slotExit`, or null when profiling is off (the returned value is then
    /// never used, so the hot path is one predictable branch and no clock read).
    ///
    /// Nesting is tracked by `slot_depth` only so `slot_total` does not
    /// double-count if a forwarder ever calls another forwarder; today the
    /// forwarders call `CpuBackend.vtable.*` directly, which bypasses these.
    inline fn slotEnter(self: *CudaBackend) ?Io.Clock.Timestamp {
        // `$QWEN35_PROF_GPU` opens one device-timing pair for the outermost slot.
        // The slot id is NOT known here -- every forwarder calls `slotEnter()`
        // with no argument and hands the slot to `slotExit` -- so `gpuEvEnter`
        // takes only the pair index and `slotExit` attributes it.  A field, not a
        // returned value: the 32 forwarders pass only the host `t0` back.
        self.gpu_ev_open = self.gpuEvEnter();
        if (!self.ev_on) {
            self.slot_depth += 1;
            return null;
        }
        self.slot_depth += 1;
        return Io.Clock.Timestamp.now(self.io, .awake);
    }

    /// Close a slot opened by `slotEnter` and add the interval to its counter.
    /// `t0 == null` (profiling off) is a no-op.
    inline fn slotExit(self: *CudaBackend, s: Slot, t0: ?Io.Clock.Timestamp) void {
        // Close the device-timing pair first: the closing record has to land on
        // the same stream the launch used, which `launchStream()` still reports
        // correctly here (the capture state has not changed during the slot).
        self.gpuEvExit(self.gpu_ev_open, s);
        self.gpu_ev_open = null;
        // INVALIDATE THE QUANTISED-ACTIVATION CACHE for the slots that can
        // WRITE an activation buffer.  Matvecs only READ their `x`; the engine
        // reuses a single `x1` across phases (`rmsnorm(x1) -> 4 matvecs ->
        // rmsnorm(x1, post_attention_norm) -> 2 matvecs ...`), so a cached hit
        // surviving a norm would hand the second phase the FIRST phase's
        // quantised activation.
        //
        // The list is an ALLOW-list of writers, not "everything but matvec".
        // Clearing on every other slot threw the cache away several times per
        // phase, which put the quantize back on the critical path: measured
        // 176 hits / 326 misses per token and four distinct activations being
        // re-quantised 180/80/80/60 times.  The slots NOT listed here
        // (`rope64`, `attn_decode`, `gdn_step`, `conv_step`, `gdn_gamma`,
        // `scale_vec`, `zero_buf`, `deinterleave_qg`, `silu`, `softmax`,
        // `argmax`, `sample`, the allocators) operate on the projection
        // buffers (`qkv`, `z`, `ab`, `mlp_h`, the KV cache) -- never on the
        // hidden-state buffer a matvec is about to consume -- so keeping the
        // cache across them is sound.  Anything added to the engine that writes
        // the ACTIVATION of a pending matvec must be added here too.
        //
        // This must run before the `t0 orelse return` below because the
        // profiling toggle is off by default.
        switch (s) {
            .rmsnorm, .l2norm, .add_resid, .copy_range, .copy_from_host, .swiglu, .mul_silu, .dequant_row, .load_linear, .const_buf => self.act_q.n = 0,
            else => {},
        }
        const start = t0 orelse return;
        const ns = durNs(start, self.io);
        const i = @intFromEnum(s);
        self.stat_ns_slot[i] += ns;
        self.stat_n_slot[i] += 1;
        // Only the outermost interval counts toward the window total, so the
        // "accounted" number stays comparable with `stat_ns_prefill_wall`.
        self.slot_depth -= 1;
        if (self.slot_depth == 0) self.stat_ns_slot_total += ns;
    }

    /// Create the event pool for `$QWEN35_PROF_GPU` once.  Events are created
    /// with timing ENABLED (flags 0): `EventElapsedTime` needs it, and
    /// `EventDisableTiming` pairs return `cudaErrorInvalidResourceHandle`.
    fn gpuEvEnsure(self: *CudaBackend) bool {
        if (self.gpu_ev_ready) return true;
        const lib = if (self.cu) |*cu| &cu.lib else return false;
        var made: usize = 0;
        for (0..gpu_ev_capacity) |i| {
            var a: ?*anyopaque = null;
            var b: ?*anyopaque = null;
            if (lib.EventCreate(&a, 0) != cudart.cudaSuccess or a == null) break;
            self.ev_pair_a[i] = a;
            made += 1;
            if (lib.EventCreate(&b, 0) != cudart.cudaSuccess or b == null) break;
            self.ev_pair_b[i] = b;
            made += 1;
            self.gpu_ev_ready = true;
        }
        if (!self.gpu_ev_ready) {
            // Bounded cleanup: destroy what was created so a partial pool cannot
            // leak, and say what we got rather than silently measuring nothing.
            for (0..gpu_ev_capacity) |i| {
                if (self.ev_pair_a[i]) |p| { _ = lib.EventDestroy(p); self.ev_pair_a[i] = null; }
                if (self.ev_pair_b[i]) |p| { _ = lib.EventDestroy(p); self.ev_pair_b[i] = null; }
            }
            std.debug.print("cuda backend: $QWEN35_PROF_GPU event pool disabled (created {d} of {d} events)\n",
                .{ made, gpu_ev_capacity * 2 });
            return false;
        }
        return true;
    }

    /// Read every complete pair of the token in flight and add each interval to
    /// its slot.  Called at the argmax drain, where the blocking D2H has already
    /// completed the stream, so `EventElapsedTime` on any pair of this token
    /// returns without waiting: the read costs one host call per kernel, not one
    /// sync per kernel.
    fn gpuEvReadToken(self: *CudaBackend) void {
        const lib = if (self.cu) |*cu| &cu.lib else return;
        var sum: u64 = 0;
        for (0..self.gpu_ev_used) |i| {
            const a = self.ev_pair_a[i];
            const b = self.ev_pair_b[i];
            if (a == null or b == null) continue;
            var ms: f32 = 0;
            if (lib.EventElapsedTime(&ms, a, b) == cudart.cudaSuccess and ms >= 0) {
                const ns: u64 = @intFromFloat(@as(f64, ms) * 1_000_000.0);
                const s = self.gpu_ev_slot[i];
                self.stat_gpu_ns_slot[s] += ns;
                self.stat_gpu_n_slot[s] += 1;
                sum += ns;
            } else self.stat_gpu_ev_err += 1;
            // Reset for reuse: events stay allocated, only their recorded
            // position is discarded by the next token's record.
            self.gpu_ev_slot[i] = 0;
        }
        self.stat_gpu_ns_token_sum += sum;
        self.stat_gpu_n_token += 1;
        self.gpu_ev_used = 0;
    }

    /// Open one GPU timing pair for a slot.  Only the outermost slot gets a pair
    /// (`slot_depth == 0`), so nested slots cannot double-count device time.
    /// The slot id is filled in by `slotExit` (which is the only place that knows
    /// it), so this takes none.
    /// Returns the pair index, or null when timing is off, capture is open (an
    /// event record inside a graph is not what we want to measure), the pool is
    /// exhausted (counted as overflow), or event creation failed.
    /// TEMP(ablation): true when slot `s`'s launch is to be SKIPPED.  Model
    /// output is garbage while any compute slot is ablated; the only consumer of
    /// this is the tok/s delta, which attributes a slot's device cost.
    inline fn ablated(self: *CudaBackend, s: Slot) bool {
        return (self.ablate_bits >> @intFromEnum(s)) & 1 != 0;
    }

    /// Parse `QWEN35_ABLATE="slot,slot,..."` into `ablate_bits` (names are the
    /// `PROF-SLOT` names).  Unknown names are counted and reported once.
    fn ablateInit(self: *CudaBackend) void {
        // Value, not mere presence: `FOO=0` must mean OFF, like every other
        // switch in the table (`QWEN35_CUDA_GRAPH=0` is documented as "disable").
        self.sample_cpu_ab = if (cudart.getenv("QWEN35_CUDA_SAMPLE_CPU")) |v|
            v.len > 0 and !std.mem.eql(u8, v, "0")
        else
            false;
        const raw = cudart.getenv("QWEN35_ABLATE") orelse return;
        var it = std.mem.splitScalar(u8, std.mem.trim(u8, raw, " \t\r\n"), ',');
        var n_ablated: usize = 0;
        while (it.next()) |tok| {
            const t = std.mem.trim(u8, tok, " \t");
            if (t.len == 0) continue;
            const s = std.meta.stringToEnum(Slot, t) orelse continue;
            self.ablate_bits |= @as(u32, 1) << @intCast(@intFromEnum(s));
            n_ablated += 1;
        }
        if (n_ablated != 0) std.debug.print("cuda backend: $QWEN35_ABLATE skips {d} slot(s) (model output is garbage; tok/s delta is the measurement)\n", .{n_ablated});
    }

    inline fn gpuEvEnter(self: *CudaBackend) ?usize {
        if (!self.gpu_ev_on) return null;
        if (self.anySlotCapturing()) return null;
        // A live decode graph replays one graph per token and suppresses the
        // host-side launches (`gOpGate`), so there is nothing per slot to record
        // -- the pairs would measure empty intervals.  The graph path reports
        // its own totals via `stat_ns_token_dev`; disable the pool instead of
        // printing a table of zeros.
        if (self.gs(0).state == .live) return null;
        if (self.slot_depth != 0) return null;
        if (!self.gpuEvEnsure()) {
            self.gpu_ev_on = false;
            return null;
        }
        if (self.gpu_ev_used >= gpu_ev_capacity) {
            self.stat_gpu_overflow += 1;
            return null;
        }
        const i = self.gpu_ev_used;
        self.gpu_ev_used += 1;
        self.gpu_ev_slot[i] = 255; // set by slotExit once the slot is known
        const lib = &self.cu.?.lib;
        if (lib.EventRecord(self.ev_pair_a[i], self.launchStream()) != cudart.cudaSuccess) {
            self.stat_gpu_ev_err += 1;
            self.gpu_ev_used -= 1;
            return null;
        }
        return i;
    }

    /// Close a pair opened by `gpuEvEnter`, attributing it to slot `s`; the pair
    /// is read at the token drain.
    inline fn gpuEvExit(self: *CudaBackend, i: ?usize, s: Slot) void {
        const idx = i orelse return;
        const lib = &self.cu.?.lib;
        self.gpu_ev_slot[idx] = @intFromEnum(s);
        if (lib.EventRecord(self.ev_pair_b[idx], self.launchStream()) != cudart.cudaSuccess) {
            self.stat_gpu_ev_err += 1;
            // A pair whose closing record never landed cannot be read; free the
            // slot by moving the last live pair into it.
            const last = self.gpu_ev_used - 1;
            self.ev_pair_a[idx] = self.ev_pair_a[last];
            self.ev_pair_b[idx] = self.ev_pair_b[last];
            self.gpu_ev_slot[idx] = self.gpu_ev_slot[last];
            self.gpu_ev_used = last;
        }
    }

    /// Milliseconds between two recorded events, waiting for the second.
    ///
    /// The wait is a HOST BLOCKING SYNCHRONISATION, so it must never appear on a
    /// per-kernel profiling path: on the 525-token prefill it costs 41% of
    /// throughput (436 -> 309 tok/s), i.e. the numbers it produces describe a
    /// run that does not exist.  Kept only for the few call sites that genuinely
    /// need the result now (prefill window end, token boundary).  Per-kernel
    /// timings go through `evElapsedNsNoWait` against a pooled event pair.
    /// Returns null on error (and counts it) so the caller leaves the totals
    /// alone instead of adding a bogus 0.
    fn evElapsedNs(self: *CudaBackend, a: ?*anyopaque, b: ?*anyopaque) ?u64 {
        const lib = if (self.cu) |*cu| &cu.lib else return null;
        const t_sync = Io.Clock.Timestamp.now(self.io, .awake);
        defer self.stat_ns_ev_sync += durNs(t_sync, self.io);
        if (lib.EventSynchronize(b) != cudart.cudaSuccess) {
            self.stat_n_ev_err += 1;
            return null;
        }
        return self.evDeltaNs(a, b);
    }

    /// Non-blocking read of a completed event pair.  Returns null when the
    /// second event has not finished yet, WITHOUT waiting -- the caller then
    /// leaves the sample out rather than stalling the stream to collect it.
    /// This is what makes the per-kernel numbers honest; see `EV_POOL`.
    fn evElapsedNsNoWait(self: *CudaBackend, a: ?*anyopaque, b: ?*anyopaque) ?u64 {
        const lib = if (self.cu) |*cu| &cu.lib else return null;
        if (lib.EventQuery(b) != cudart.cudaSuccess) return null;
        return self.evDeltaNs(a, b);
    }

    /// Shared tail of both readers: the actual `cudaEventElapsedTime`, which
    /// does NOT synchronise on its own.
    fn evDeltaNs(self: *CudaBackend, a: ?*anyopaque, b: ?*anyopaque) ?u64 {
        const lib = if (self.cu) |*cu| &cu.lib else return null;
        var ms: f32 = 0;
        if (lib.EventElapsedTime(&ms, a, b) != cudart.cudaSuccess) {
            self.stat_n_ev_err += 1;
            return null;
        }
        if (!(ms >= 0)) {
            self.stat_n_ev_err += 1;
            return null;
        }
        return @intFromFloat(@as(f64, ms) * 1_000_000.0);
    }

    /// Read the pending per-call batchK samples that have completed.
    ///
    /// `wait` is for the end of a profile window: it blocks once (a single
    /// `EventSynchronize` on the NEWEST pending sample, not one per kernel) so
    /// the last few intervals are not lost.  The steady-state call passes false
    /// and simply leaves unready samples queued.
    fn drainBkEvents(self: *CudaBackend, wait: bool) void {
        const lib = if (self.cu) |*cu| &cu.lib else return;
        if (self.ev_pending_n == 0) return;
        if (wait) {
            const last = self.ev_pending[self.ev_pending_n - 1];
            if (lib.EventSynchronize(self.ev_pool[last.slot + 1]) != cudart.cudaSuccess) {
                self.stat_n_ev_err += 1;
            }
        }
        var kept: usize = 0;
        var i: usize = 0;
        while (i < self.ev_pending_n) : (i += 1) {
            const pend = self.ev_pending[i];
            const a = self.ev_pool[pend.slot];
            const b = self.ev_pool[pend.slot + 1];
            if (self.evElapsedNsNoWait(a, b)) |ns| {
                // A single batchK call cannot take tens of milliseconds: the
                // largest shape measured here is 1.6 ms.  Anything above
                // `MAX_BK_NS` means the pair spans work that is not this call
                // (a Begin whose End was skipped, or a ring slot reused while
                // still pending), so it is counted and dropped instead of
                // being added to the totals.
                if (ns > MAX_BK_NS) {
                    // Drop, do not retry: the sample will never shrink.
                    self.stat_n_ev_suspect += 1;
                    continue;
                }
                self.stat_ns_batchk_gpu += ns;
                self.stat_n_batchk_gpu += 1;
                const ti: usize = @intCast(pend.ttype);
                if (ti < self.stat_ns_bk_by_type.len) {
                    self.stat_ns_bk_by_type[ti] +|= ns;
                    self.stat_n_bk_by_type[ti] +|= 1;
                }
            } else {
                self.ev_pending[kept] = pend;
                kept += 1;
            }
        }
        self.ev_pending_n = kept;
    }

    /// Begin the device-side clock for one `matvecBatchK` call.  Records into the
    /// default stream, so the timestamp is taken *after* every kernel already
    /// queued by earlier calls -- the gap between two calls is therefore
    /// attributed to whoever issues work next, never smeared into the kernel
    /// timings.
    ///
    /// The prefill window's start event (`ev_pre_a`) is recorded here, on the
    /// first call issued while the window is open: that is the first device
    /// position that belongs to prefill.
    fn batchKTimingBegin(self: *CudaBackend) void {
        if (!self.ensureEvents()) {
            self.ev_on = false;
            return;
        }
        const lib = if (self.cu) |*cu| &cu.lib else return;
        if (self.prefill_open and !self.ev_pre_started) {
            if (lib.EventRecord(self.ev_pre_a, null) != cudart.cudaSuccess) {
                self.stat_n_ev_err += 1;
            }
            self.ev_pre_started = true;
        }
        // Record the call's start into the current ring pair.  `ev_bk_a` is
        // still recorded on the very first call only, as the prefill window
        // origin (it is a single event, so it cannot hold per-call data).
        const wr = self.ev_pool_wr;
        self.stat_n_bk_begin += 1;
        if (lib.EventRecord(self.ev_pool[wr], null) != cudart.cudaSuccess) self.stat_n_ev_err += 1;
    }

    /// Close the device-side clock for one `matvecBatchK` call.
    ///
    /// `ev_bk_b - ev_bk_a` is everything that call issued to the device.
    fn batchKTimingEnd(self: *CudaBackend) void {
        const lib = if (self.cu) |*cu| &cu.lib else return;
        const wr = self.ev_pool_wr;
        self.stat_n_bk_end += 1;
        // Close the ring pair the matching `batchKTimingBegin` opened.
        if (lib.EventRecord(self.ev_pool[wr + 1], null) != cudart.cudaSuccess) self.stat_n_ev_err += 1;
        // `EV_POOL/2` pairs can be outstanding, so the ring must not be
        // wrapped onto a pair that is still pending.  With 128 pairs and the
        // host never more than a few launches ahead of the device, a full ring
        // means the device is far behind only if something else is wrong --
        // drop the OLDEST sample (and count it) rather than corrupt the slot.
        if (self.ev_pending_n == self.ev_pending.len) {
            self.stat_n_ev_dropped += 1;
            std.mem.copyForwards(@TypeOf(self.ev_pending[0]), self.ev_pending[0 .. self.ev_pending.len - 1], self.ev_pending[1..]);
            self.ev_pending_n -= 1;
        }
        self.ev_pending[self.ev_pending_n] = .{ .slot = @intCast(wr), .ttype = self.bk_ttype, .live = true };
        self.ev_pending_n += 1;
        self.ev_pool_wr = (wr + 2) % EV_POOL;
        // Collect whatever has already completed.  A sample that is not ready
        // yet stays pending -- dropping it would bias the totals toward the
        // cheap calls, waiting for it is what this whole ring exists to avoid.
        self.drainBkEvents(false);
        // Host wall clock spanned by the batchk calls of the prefill window.  A
        // monotonic host clock, incremented under the same `prefill_open` gate,
        // so this measures "first timed batchk entry -> last timed batchk exit"
        // and its complement (`pre_wall - pre_bkwall`) is the part of prefill
        // that issued no matvec.
        if (self.prefill_open) {
            const now = Io.Clock.Timestamp.now(self.io, .awake);
            if (!self.bk_wall_started) {
                self.bk_wall_t0 = now;
                self.bk_wall_started = true;
            }
            const d = self.bk_wall_t0.durationTo(now).raw.nanoseconds;
            self.stat_ns_prefill_bk_wall = if (d > 0) @intCast(@min(d, std.math.maxInt(u64))) else 0;
        }
    }

    /// Open the prefill window.  The engine calls `profReset` immediately before
    /// prefill, so the host wall clock starts here too.
    fn openPrefillWindow(self: *CudaBackend) void {
        if (!self.ensureEvents()) {
            self.ev_on = false;
            return;
        }
        self.prefill_open = true;
        self.ev_pre_started = false;
        self.bk_wall_started = false;
        self.prefill_t0 = Io.Clock.Timestamp.now(self.io, .awake);
    }

    /// Close the prefill window: record the end event, wait for it, and read out
    /// both the device span and the host wall clock of the same window.
    ///
    /// Called from the first single-token `matvec` after prefill (the decode
    /// path), and again, harmlessly, from `dProfPhase(false)` -- the second call
    /// finds the window already shut.  Waiting for the end event here is what
    /// the measurement costs: it drains the queue, so the following decode
    /// launch cannot overlap the read.  That cost lands *after* the number it
    /// measures, and the report quantifies the interference by comparing a
    /// PROF run against a plain one.
    fn closePrefillWindow(self: *CudaBackend) void {
        if (!self.prefill_open) return;
        self.prefill_open = false;
        const now = Io.Clock.Timestamp.now(self.io, .awake);
        const wall = self.prefill_t0.durationTo(now).raw.nanoseconds;
        self.stat_ns_prefill_wall = if (wall > 0) @intCast(@min(wall, std.math.maxInt(u64))) else 0;
        if (!self.ev_created or !self.ev_pre_started) {
            self.reportStats("prefill done");
            return;
        }
        if (self.cu) |*cu| {
            if (cu.lib.EventRecord(self.ev_pre_b, null) != cudart.cudaSuccess) {
                self.stat_n_ev_err += 1;
            } else if (self.evElapsedNs(self.ev_pre_a, self.ev_pre_b)) |ns| {
                self.stat_ns_prefill_span = ns;
            }
        }
        // One blocking pass so the last intervals of the window are counted.
        // Everything before this was collected non-blocking by the ring, so the
        // queue stayed deep for the whole run -- which is the entire point.
        self.drainBkEvents(true);
        // `reportProfile` has no call site on the CLI path (the engine is
        // arena-allocated, so `deinit` never runs), so this is the only place the
        // fully-formed decomposition is ever printed.
        self.reportStats("prefill done");
    }

    /// Map a `Q35_TT_*` code (api.h) back to a printable name for `CUDA-BKT`.
    /// The CUDA type codes are NOT contiguous and NOT in GGUF order (see the
    /// table in api.h), so a positional table here would mislabel every type --
    /// which is exactly the mistake AGENTS.md warns about for microbenchmarks.
    fn bkTypeName(code: u8) []const u8 {
        return switch (code) {
            0 => "f32", 1 => "f16", 2 => "q4_0", 3 => "q4_1",
            6 => "q5_0", 7 => "q5_1", 8 => "q8_0", 9 => "q8_1",
            10 => "q2_k", 11 => "q3_k", 12 => "q4_k", 13 => "q5_k", 14 => "q6_k",
            15 => "q8_k", 16 => "iq2_xxs", 17 => "iq2_xs", 18 => "iq3_xxs",
            19 => "iq1_s", 20 => "iq4_nl", 21 => "iq3_s", 22 => "iq2_s",
            23 => "iq4_xs", 29 => "iq1_m", 30 => "bf16",
            else => "?",
        };
    }

    /// `$QWEN35_PROF`: one `CUDA-M [...]` line with what actually ran on the device,
    /// plus a per-type breakdown, so "did this model mix backends?" is answerable
    /// from the log.  Uploaded MiB is the resident weight footprint; bufs_MiB is the
    /// reusable activation/output scratch.
    fn reportStats(self: *CudaBackend, when: []const u8) void {
        if (self.cu == null) return;
        if (self.stat_n_batchk_gpu > 0) {
            // Per-type device cost of the batchK calls.  Without this the only
            // number available is the average over all 500-odd launches in a
            // 525-token prefill, which is dominated by whichever type is
            // biggest and hides whether ANY of them is running at the
            // standalone-kernel rate.  Emitted only when the events were on.
            std.debug.print("CUDA-BKT [{s}]", .{when});
            var seen: usize = 0;
            var order: [64]usize = undefined;
            var ti: usize = 0;
            while (ti < 64) : (ti += 1) {
                if (self.stat_n_bk_by_type[ti] != 0) {
                    order[seen] = ti;
                    seen += 1;
                }
            }
            std.mem.sort(usize, order[0..seen], self, struct {
                fn lt(ctx: *CudaBackend, a: usize, b: usize) bool {
                    return ctx.stat_ns_bk_by_type[a] > ctx.stat_ns_bk_by_type[b];
                }
            }.lt);
            for (order[0..seen]) |ix| {
                const ns = self.stat_ns_bk_by_type[ix];
                const n = self.stat_n_bk_by_type[ix];
                std.debug.print(" {s}({d})={d}us/{d}={d}us", .{
                    bkTypeName(@intCast(ix)), ix,
                    ns / 1000, n, ns / 1000 / n,
                });
            }
            std.debug.print("\n", .{});
        }
        // `d_xqm` is only non-zero on the opt-in dp4a path, so it usually adds
        // nothing here; counting it keeps the number honest when it does.
        const buf_bytes = self.d_act_bytes + self.d_xq_bytes + self.d_xqm_bytes + self.d_out_bytes + self.d_deq_bytes;
        std.debug.print(
            "CUDA-M  [{s}] matvec={d} batchk={d} dequant_row={d} up_MiB={d} h2d_kB={d} d2h_kB={d} resident_rows={d} bufs_MiB={d} iq_tbl_kB={d} t_mv_ms={d} t_bk_ms={d} t_dq_ms={d} hostpaths act={d} out={d} rope={d} rope_dev={d} bk_gpu_ms={d} bk_n={d} pre_wall_ms={d} pre_bkwall_ms={d} pre_span_ms={d} bk_launch={d} ev_sync_ms={d} ev_dropped={d} ev_suspect={d} bk_begin={d} bk_end={d} gpu_ev_err={d}\n",
            .{
                when,
                self.stat_n_matvec,
                self.stat_n_batchk,
                self.stat_n_dequant,
                self.stat_uploaded_bytes >> 20,
                self.stat_h2d_bytes >> 10,
                self.stat_d2h_bytes >> 10,
                self.dev_rows.items.len,
                buf_bytes >> 20,
                self.iqTablesBytes() >> 10,
                self.stat_ns_matvec / 1_000_000,
                self.stat_ns_batchk / 1_000_000,
                self.stat_ns_dequant / 1_000_000,
                // Non-zero here means a numeric slot was handed host memory.  On
                // the per-token path all three must be 0 except `rope`, which is
                // the documented remainder.  `rope` / `rope_dev` partition the
                // same rotations between the two `$QWEN35_CUDA_ROPE` paths, so
                // the line itself proves which one ran (device: rope=0,
                // rope_dev>0; host: the reverse).
                self.stat_host_act_uploads,
                self.stat_host_out_downloads,
                self.stat_rope_trips,
                self.stat_rope_devices,
                // GPU-side numbers from CUDA events ($QWEN35_PROF only; 0
                // without it, because no event is ever recorded then).
                //   bk_gpu_ms     device time summed over timed batchk calls --
                //                 the actual GPU execution of the matmuls +
                //                 their activation quantize kernels
                //   bk_n          how many calls that sum covers
                //   pre_wall_ms   host wall clock of the prefill window
                //   pre_bkwall_ms host wall spanned by the batchk calls
                //   pre_gpu_span_ms
                //                 device event span of the whole window; equals
                //                 pre_wall_ms up to the drain, so it is a
                //                 sanity check (they must agree), not the busy
                //                 number
                // Decomposition of pre_wall:
                //   bk_gpu        GPU busy in matvecBatchK
                //   bk_host       pre_bkwall - bk_gpu  (host slack between
                //                 batchk calls: quantize/upload/download/launch)
                //   non_bk        pre_wall - pre_bkwall (attention, GDN, rmsnorm,
                //                 sampling -- issued outside matvecBatchK)
                self.stat_ns_batchk_gpu / 1_000_000,
                self.stat_n_batchk_gpu,
                self.stat_ns_prefill_wall / 1_000_000,
                self.stat_ns_prefill_bk_wall / 1_000_000,
                self.stat_ns_prefill_span / 1_000_000,
                self.stat_n_launch_bk,
                self.stat_ns_ev_sync / 1_000_000,
                self.stat_n_ev_dropped,
                self.stat_n_ev_suspect,
                self.stat_n_bk_begin,
                self.stat_n_bk_end,
                self.stat_n_ev_err,
            },
        );
        // Per-type breakdown, only for types that actually ran.  Assembled with
        // `bufPrint` into a growing slice: `std.io` writers are gone in Zig 0.17.
        var line: [448]u8 = undefined;
        var n: usize = 0;
        n += (std.fmt.bufPrint(line[n..], "CUDA-M  types:", .{}) catch return).len;
        var any = false;
        for (0..self.stat_by_type.len) |i| {
            const cnt = self.stat_by_type[i];
            if (cnt == 0) continue;
            const tt: gguf.TensorType = @fromBackingInt(@intCast(@as(u32, @intCast(i))));
            n += (std.fmt.bufPrint(line[n..], " {s}={d}", .{ @tagName(tt), cnt }) catch break).len;
            any = true;
        }
        if (!any) n += (std.fmt.bufPrint(line[n..], " (none)", .{}) catch line[n..n]).len;
        std.debug.print("{s}\n", .{line[0..n]});

        // Per-type wall time, largest first.  `stat_ns_matvec` includes the
        // blocking activation upload and the output download, so these sums add
        // up to more than the kernel time -- that is deliberate: the question
        // being answered is "where does the wall clock go per type".
        var tline: [448]u8 = undefined;
        var tn: usize = 0;
        tn += (std.fmt.bufPrint(tline[tn..], "CUDA-M  time:", .{}) catch return).len;
        const total: u64 = blk: {
            var s: u64 = 0;
            for (self.stat_ns_by_type) |v| s += v;
            break :blk s;
        };
        for (0..self.stat_by_type.len) |i| {
            const ns = self.stat_ns_by_type[i];
            if (ns == 0) continue;
            const tt: gguf.TensorType = @fromBackingInt(@intCast(@as(u32, @intCast(i))));
            const pct: u64 = if (total == 0) 0 else ns * 100 / total;
            tn += (std.fmt.bufPrint(tline[tn..], " {s}={d}ms/{d}%", .{
                @tagName(tt), ns / 1_000_000, pct,
            }) catch break).len;
        }
        std.debug.print("{s}\n", .{tline[0..tn]});

        self.reportSlotStats();
    }

    /// `PROF-SLOT`: one line per `backend.Exec` slot, largest total first, plus
    /// the reconciliation against the prefill window.
    ///
    /// `slot_sum_ms` adds the host wall clock of every timed slot call,
    /// `unaccounted_ms = pre_wall_ms - slot_sum_ms` is the prefill loop itself
    /// (engine arithmetic, the sampler, the batchK chunk scheduler, the token
    /// loop bookkeeping), i.e. exactly the part that never crosses the `Exec`
    /// vtable.  A large unaccounted number means a *path*, not a slot, is still
    /// missing from the table.
    ///
    /// Fields per line: `name calls total_ms pct sync`.  `sync=1` marks the
    /// slots whose forwarder can drain the stream, so their ms MAY include
    /// device execution; `sync=0` slots measure submission only and their
    /// queued GPU work surfaces inside the later draining slot (or in
    /// `bk_gpu_ms` for the batchK path).  Note that `sync` is a static property
    /// of the slot, NOT evidence that this run blocked: `matvec` is flagged
    /// because its forwarder contains `downloadOut`, but that path only runs
    /// for a host output buffer, which the decode path never passes -- see the
    /// note on `slotSyncs`.  So `sync=1` on `matvec` does not by itself mean
    /// host waiting, and `total_ms` there is launch cost, not a stall.  That is the one caveat needed to read the table as
    /// "where did the wall clock go": submission time is real host time and is
    /// never double-counted, but it is not the same question as GPU busy time.
    fn reportSlotStats(self: *CudaBackend) void {
        if (self.cu == null) return;
        // Sort indices by total, descending.  N=28, so an insertion sort into a
        // fixed array beats anything fancier and allocates nothing.
        var order: [slot_count]u8 = undefined;
        var n: usize = 0;
        for (0..slot_count) |i| {
            if (self.stat_n_slot[i] == 0) continue;
            order[n] = @intCast(i);
            n += 1;
        }
        var i: usize = 1;
        while (i < n) : (i += 1) {
            const cur = order[i];
            const cur_ns = self.stat_ns_slot[cur];
            var j: usize = i;
            while (j > 0 and self.stat_ns_slot[order[j - 1]] < cur_ns) : (j -= 1) {
                order[j] = order[j - 1];
            }
            order[j] = cur;
        }

        const wall_ns = self.stat_ns_prefill_wall;
        const sum_ns = self.stat_ns_slot_total;
        std.debug.print(
            "PROF-SLOT window_ms={d} slot_sum_ms={d} unaccounted_ms={d} accounted_pct={d}.{d}% slots={d} note=\"sync=0 slots are launch-only; their GPU time is counted by the draining slot\"\n",
            .{
                wall_ns / 1_000_000,
                sum_ns / 1_000_000,
                (wall_ns -| sum_ns) / 1_000_000,
                // Fraction of the prefill window covered by timed slots.  Shown
                // as a percentage so a reader does not have to divide two ms
                // numbers, and so a *regression* in coverage (a new engine path
                // that skips the vtable) shows up as a falling number here.
                if (wall_ns == 0) 0 else (sum_ns * 1000 / wall_ns) / 10,
                if (wall_ns == 0) 0 else (sum_ns * 1000 / wall_ns) % 10,
                n,
            },
        );
        for (order[0..n]) |idx| {
            const s: Slot = @enumFromInt(idx);
            const ns = self.stat_ns_slot[idx];
            const pct: u64 = if (sum_ns == 0) 0 else ns * 1000 / sum_ns;
            std.debug.print(
                "PROF-SLOT {s:<15} calls={d:<7} total_ms={d:<6} pct={d}.{d}% sync={d}\n",
                .{
                    slotName(s),
                    self.stat_n_slot[idx],
                    ns / 1_000_000,
                    pct / 10,
                    pct % 10,
                    @intFromBool(slotSyncs(s)),
                },
            );
        }
        self.reportCopyStats();
        self.reportGpuSlotStats();
        self.reportArgmaxDrain();
    }

    /// `PROF-GPU`: the same slot table as `PROF-SLOT`, but measured by the device.
    ///
    /// `dev_us` is the interval the GPU recorded between the slot's opening and
    /// closing event records.  Because every record sits on the stream the launch
    /// goes to, the interval includes device work queued ahead of the slot -- so
    /// the per-slot numbers of a token SUM to that token's device time, and the
    /// sum is printed against `stat_ns_token_dev` (the independent token-end pair)
    /// as the reconciliation.  The difference between the two is the device idle
    /// the engine left between slots: the launch-bound signature.
    fn reportGpuSlotStats(self: *CudaBackend) void {
        const divCeil = struct {
            fn f(a: u64, b: u64) u64 {
                return if (b == 0) 0 else (a + b - 1) / b;
            }
        }.f;
        if (!self.gpu_ev_on) return;
        const tokens: usize = if (self.stat_gpu_n_token == 0) 1 else self.stat_gpu_n_token;
        var order: [slot_count]u8 = undefined;
        var n: usize = 0;
        for (0..slot_count) |i| {
            if (self.stat_gpu_n_slot[i] == 0) continue;
            order[n] = @intCast(i);
            n += 1;
        }
        var i: usize = 1;
        while (i < n) : (i += 1) {
            const cur = order[i];
            const cur_ns = self.stat_gpu_ns_slot[cur];
            var j: usize = i;
            while (j > 0 and self.stat_gpu_ns_slot[order[j - 1]] < cur_ns) : (j -= 1) {
                order[j] = order[j - 1];
            }
            order[j] = cur;
        }
        const sum_ns = self.stat_gpu_ns_token_sum;
        const tok_ns = self.stat_ns_token_dev;
        std.debug.print(
            "PROF-GPU tokens={d} gpu_sum_ms={d} tok_dev_ms={d} idle_ms={d} overflow={d} ev_err={d} note=\"dev_us = device interval per slot, per token\"\n",
            .{
                self.stat_gpu_n_token,
                sum_ns / 1_000_000,
                tok_ns / 1_000_000,
                (tok_ns -| sum_ns) / 1_000_000,
                self.stat_gpu_overflow,
                self.stat_gpu_ev_err,
            },
        );
        for (order[0..n]) |idx| {
            const s: Slot = @enumFromInt(idx);
            const ns = self.stat_gpu_ns_slot[idx];
            const calls = self.stat_gpu_n_slot[idx];
            std.debug.print(
                "PROF-GPU {s:<15} calls/tok={d:<5} dev_us/tok={d:<8} avg_us/call={d:<6} pct={d}.{d}%\n",
                .{
                    slotName(s),
                    divCeil(calls, tokens),
                    divCeil(ns, tokens * 1_000),
                    divCeil(ns, calls * 1_000),
                    if (sum_ns == 0) 0 else (ns * 1000 / sum_ns) / 10,
                    if (sum_ns == 0) 0 else (ns * 1000 / sum_ns) % 10,
                },
            );
        }
    }

    /// `PROF-DRAIN`: the per-token argmax drain split into the device's share and
    /// the host's share.
    ///
    /// `PROF-SLOT argmax` gives one number for the whole slot, but that number
    /// mixes two very different costs: the device finishing the queue the token
    /// already issued (unavoidable compute that would happen anyway) and the
    /// host sitting in a blocking D2H copy (pure lost overlap).  This line
    /// separates them, because a plan that shrinks one does nothing for the
    /// other -- e.g. a CUDA graph attacks submission overhead, which shows up
    /// here as device-side tail, not as the copy.
    fn reportArgmaxDrain(self: *CudaBackend) void {
        if (self.stat_n_argmax_timed == 0) return;
        const n = self.stat_n_argmax_timed;
        const host_ns = self.stat_ns_argmax_host;
        const dev_ns = self.stat_ns_argmax_gpu;
        // Per call, in microseconds, so the numbers are comparable across runs
        // with different token counts.
        const host_us = host_ns / n / 1000;
        const dev_us = dev_ns / n / 1000;
        // Device share of what the host waited for.  <100% means the host paid
        // for more than the device was actually executing (submission, driver,
        // copy latency); that gap is what a graph replay can reclaim.
        const dev_pct: u64 = if (host_ns == 0) 0 else (dev_ns * 100) / host_ns;
        // Whole-token device time: `ev_prev_tok_end .. ev_tok_end` spans every
        // kernel of one decode token.  Compared against the per-token host wall
        // clock this says whether the host or the device is the ceiling.
        const tok_dev_us = if (self.stat_n_token_dev == 0) 0 else self.stat_ns_token_dev / self.stat_n_token_dev / 1000;
        std.debug.print(
            "PROF-DRAIN argmax calls={d} host_us={d} dev_us={d} host_ms={d} dev_ms={d} dev_pct={d}% tok_dev_us={d} note=\"host_us is the blocking copy wait; dev_us is the argmax kernel alone; tok_dev_us is ALL device work for one token\"\n",
            .{ n, host_us, dev_us, host_ns / 1_000_000, dev_ns / 1_000_000, dev_pct, tok_dev_us },
        );
    }

    /// `PROF-COPY`: `copyRange` narrowed down by element count.
    ///
    /// `PROF-SLOT` gives one line for the whole slot; this says which width is
    /// paying for it.  Printed after the slot table because it is a drill-down,
    /// and only when at least one call happened (a run that never copies -- the
    /// all-quantized-embedding models -- prints nothing).
    ///
    /// Fields: `elems<= bound: calls= total_ms= bytes_each=`.  `bytes_each` is
    /// the per-call payload, which is what makes a 512-B-per-call family
    /// visible as such instead of being averaged with the embedding row.
    fn reportCopyStats(self: *CudaBackend) void {
        if (self.cu == null) return;
        var total: u32 = 0;
        for (self.stat_copy_bins) |b| total += b.n;
        if (total == 0) return;
        for (self.stat_copy_bins, 0..) |b, i| {
            if (b.n == 0) continue;
            const bound = if (i < copy_bin_bounds.len) copy_bin_bounds[i] else 0;
            std.debug.print(
                "PROF-COPY elems<={d:<5} calls={d:<7} total_ms={d:<6} bytes_each={d}\n",
                .{
                    // 0 marks the overflow bucket ("bigger than the last bound").
                    bound,
                    b.n,
                    b.ns / 1_000_000,
                    b.elems / b.n * @sizeOf(f32),
                },
            );
        }
    }

    fn vName(ctx: *anyopaque) []const u8 {
        const self = self_(ctx);
        if (self.cu) |cu| {
            return std.fmt.bufPrint(&msg_buf, "cuda ({s}, sm {d}.{d})", .{
                cu.devName(), cu.cc / 10, cu.cc % 10,
            }) catch "cuda";
        }
        return "cuda (unavailable)";
    }

    fn vDescribe(ctx: *anyopaque) []const u8 {
        const self = self_(ctx);
        if (self.cu) |cu| {
            return std.fmt.bufPrint(
                &msg_buf,
                "cuda sm {d}.{d} {s}, {d} MiB free; all quantized types on device",
                .{ cu.cc / 10, cu.cc % 10, cu.devName(), cu.free_bytes >> 20 },
            ) catch "cuda";
        }
        return self.init_error;
    }

    fn dSetTier(ctx: *anyopaque, tier_name: []const u8) void {
        // 设备上每种格式只有一份实现；CPU 的 ISA tier 现在只影响元数据加载器。
        // 接受并忽略，好让 `--tier` 在那一部分保持诊断价值。
        _ = ctx;
        _ = tier_name;
    }

    /// 显式计算图（XG）的钩子。S2 返回直通图；捕获/重放在 S3 接线。CUDA 后端
    /// 仍保留 S1 的 registry，好让 A/B 开关（`QWEN35_CUDA_GRAPH`）与诊断输出
    /// 依然能描述"接下来会发生什么"。
    fn dGraphBegin(ctx: *anyopaque, kind: backend.GraphKind, ctl: backend.GraphCtl, slot: u8) backend.Graph {
        const self = self_(ctx);
        if (!self.gpuReady()) return .{ .kind = kind, .ctl = ctl, .slot = slot };
        // 槽号来自调用方（见 `backend.Graph.slot`）。越界只会是引擎的笔误，
        // 直接落回直通图并报一次，而不是悄悄写到别的槽上。
        if (slot >= self.g_slots.len) {
            std.debug.print("cuda backend: graphBegin slot {d} out of range\n", .{slot});
            return .{ .kind = kind, .ctl = ctl, .slot = slot };
        }
        // 在其它任何东西跑起来之前，先把这一步的标量发布出去。这是每步唯一能
        // 安全分配控制块的地方（capture 区间内 `cudaMalloc` 非法），也是唯一
        // 知道"这一步开始了"的地方。
        //
        // 这次写发生在图 launch 之前，图里的 kernel 只读那块设备内存；不带图的
        // 步骤则按自己的流顺序读同一个块（同一条流）。
        //
        // 发布失败不是致命的：每个消费方都有逐标量回退，所以这一步照样跑（只是
        // capture 不安全，也就是它不会被录进图）。
        _ = self.publishCtl(ctl, null, true);
        const g = backend.Graph{ .kind = kind, .ctl = ctl, .slot = slot };

        // XG 服务"节点集合可重放"的三种前向：`.decode`（1 行）、`.verify`
        // （k 行，k 由调用方声明、写进控制块）、`.nextn`（草稿块，1 行）。
        // prefill 的块长度是数据相关的，按形状建图最多要 nbatch-1 张；而
        // prefill 现在压根不经过这里（`forwardChunk` 自己不开 span），它的
        // 发射开销在 238 ms 的块里上界只有 6.3%。
        if (cudart.getenv("QWEN35_XD_REPLAYDBG") != null) {
            std.debug.print("[xg] graphBegin kind={s} slot={d} phase={s}\n", .{ @tagName(kind), slot, @tagName(self.gs(slot).xd_phase) });
        }
        if (kind != .decode and kind != .verify and kind != .nextn) return g;
        if (!graphWanted() or !xdWanted()) return g;
        // `.verify` 默认**不进图**：把 k 行前向录成一张可重放的图，要求它的
        // 每一个逐轮标量都走设备控制块。目前只覆盖到 KV 写入行与 attn 起始行，
        // `matvecBatchK` 的激活/输出 scratch 仍按录制时那一次固化 —— 实测
        // （27B IQ3_XXS，MTP draft=2）：图里只剩"尾部 rmsnormRows + lm_head"
        // 两步时接受率也是 0%（直发 60%），说明缺口不止一处。
        //
        // prefill 不走这里，它的发射开销在 238 ms 的块里上界只有 6.3%，所以
        // "verify 不建图"并不亏；把剩下的标量逐个搬到 ctl 之后再打开。
        // `$QWEN35_XD_VERIFY=1` 可以强行打开，用来继续那条工作。
        if (kind == .verify and cudart.getenv("QWEN35_XD_VERIFY") == null) return g;
        if (kind != .decode) return self.xdPhaseBegin(g);
        // 采样参数在**图 launch 之前**拷进设备常驻控制块。这是整个重放循环里
        // 唯一正确的位置：
        //   * 图里的 `k_plan` / `k_draw` / `k_coin` 在重放时从该块读
        //     temperature / top_k / top_p / PRNG，读的必须是**这一步**的值；
        //   * 而 `sample` 这个 op 的调用点在图 launch 之后（`dSample_body`
        //     那时再写就晚了 —— 会把图刚写下的 `CTL_FLAGS`/`CTL_TOKEN_OUT`
        //     整体重置，实测读回永远是 0）；
        //   * 写进 capture 区间更不行，那会被录成节点，重放永远用录制值。
        // 所以就是用户定的形态：两次重放之间 host H2D 一次，重放期间图只读。
        //
        // 拷贝发到 `g_stream`（图所在流）而不是默认流：图 launch 也在这条流
        // 上，先入先出保证它读到的是这次写进去的值。
        //
        // 引擎每步传的是同一个 `*Sampler`（`dSample_body` 里记下的
        // `sample_sp`），所以这里拿到的 temperature/top-k/top-p/PRNG 就是当步的。
        if (self.gs(slot).xd_phase == .live and kind == .decode) {
            if (self.sample_sp) |sp| {
                if (self.d_sample_ctl != null) {
                    _ = self.publishSampleCtl(self.sample_vocab, sp, self.g_stream);
                }
            }
        }
        const out = self.xdPhaseBegin(g);
        return out;
    }

    fn dGraphEnd(ctx: *anyopaque, g: *backend.Graph) void {
        const self = self_(ctx);
        if (g.ptr == null) return;
        // 收尾用的是**这张图自己的槽**，所以嵌套 span（草稿块在 verify 里面）
        // 各自关各自的 capture，外层不会被内层提前关掉。
        self.xdPhaseEnd(g.*);
        g.ptr = null;
    }

    /// 图关闭之后取回本步 `sample` 写进设备控制块的那个 id。
    ///
    /// 为什么必须在这里而不是在 `sample` 里：这读是一次阻塞 D2H，capture
    /// 期间非法；把它录进图又会让每次重放都返回录制时那个 token（实测：
    /// 24 个 token 全是同一个 id）。`sample` 的 kernel 仍然是图的最后一个
    /// 节点，这里只负责把结果搬回主机。
    fn dSampledToken(ctx: *anyopaque) u32 {
        const self = self_(ctx);
        const sp = self.sample_sp orelse return self.sampled;
        // 采样 kernel 刚由图/默认流写出，读之前要确保它执行完。
        if (self.cu) |*cu| {
            if (self.g_stream) |st| _ = cu.lib.StreamSynchronize(st);
            if (cu.lib.StreamSynchronize(null) != cudart.cudaSuccess) _ = cu.lib.GetLastError();
        }
        // 草稿块的采样器在 capture 期间**没有**被录进去（那不是 `sample.cu` 的
        // 节点，而是 `argmax_batch`），所以结果得在这里用 CPU 参照从
        // `d_deq` 里那份 f32 logits 算出来。读一次 `vocab` 个 float：草稿块
        // 一轮只有一次，与 decode 的一 token 一次同量级。
        if (self.sample_logits) |pl| {
            if (self.h_sample_ab_scratch == null or self.h_sample_ab_scratch.?.len < self.sample_vocab) {
                const alloc = std.heap.page_allocator;
                if (self.h_sample_ab_scratch) |b| alloc.free(b);
                self.h_sample_ab_scratch = alloc.alloc(f32, self.sample_vocab) catch null;
            }
            if (self.h_sample_ab_scratch) |hs| {
                if (self.cu) |*cu| {
                    if (cu.lib.Memcpy(hs.ptr, pl, self.sample_vocab * @sizeOf(f32), cudart.MemcpyDeviceToHost) == cudart.cudaSuccess) {
                        const row = backend.Buf.wrap(hs[0..self.sample_vocab]);
                        self.sampled = cpu_mod.CpuBackend.vtable.sample(
                            .{}, @ptrCast(&self.cpu), row, backend.Buf.wrap(hs[0..self.sample_vocab]), sp,
                        );
                        self.sample_logits = null;
                        return self.sampled;
                    }
                    _ = cu.lib.GetLastError();
                }
            }
            self.sample_logits = null;
        }
        self.sampled = self.collectSampledToken(sp);
        return self.sampled;
    }

    fn dSetThreads(ctx: *anyopaque, io: std.Io, nt: usize) void {
        // 主机上只跑（非计算的）元数据加载器；线程池除此之外并不使用，但仍然转告
        // CPU 后端，好让它的 profile 行保持有意义。
        const self = self_(ctx);
        self.io = io;
        self.cpu.exec().setThreads(io, nt);
    }

    /// 进入/离开"图外的直发段"（MTP 草稿块）。见 `VTable.graphDirect`。
    ///
    /// 进入：关掉当前正在录的 capture（把已录的节点丢掉、改用直发，原因见
    /// `graphAbandon`），并让每种 kind 的图都转成 `.off` —— 草稿块每轮换嵌入
    /// token，它的节点实参无法固化成常量，录进去只会重放第一轮。
    ///
    /// 这不只是"少录一段"的问题：草稿块跑在 verify 的 `forwardChunk` 调用栈
    /// **里面**，旧机制下它的节点会被记进 verify 那张图（实测：第一轮就验证出
    /// token 0 而不是 119016）。整段直发还它本来该有的样子。
    ///
    /// 离开：清掉"图断了"的标记。下一次 `graphReserve` 会重新把 decode 置回
    /// `.rec`（幂等），于是草稿块的代价不泄漏到 decode。
    fn dGraphDirect(ctx: *anyopaque, on: bool) void {
        const self = self_(ctx);
        if (on) {
            // **只置标记，不动任何槽。**
            //
            // 这段的语义是"接下来这些 op 不属于任何可重放单元"，不是"把已有
            // 的图作废"。以前这里会 `GraphExecDestroy` 全部槽并把 `xd_phase`
            // 归 `.off` —— 而 MTP 的整段 `generateSpec` 都在这个 span 里，
            // 于是 verify 的图永远建不起来（实测 `[xg] graphBegin kind=chunk
            // slot=1 phase=off`，一次都没录过）。真要把某个槽作废，用
            // `graphAbandon` 或把那个槽的 phase 单独置掉。
            self.g_direct = true;
            return;
        }
        self.g_direct = false;
    }

    /// prefill 循环的结束点：主机侧唯一能确知"prefill 已经做完"的地方。这里记录
    /// 的 event 仍然排在默认流里每一个 prefill kernel 之后，所以等这个结束 event
    /// 不会把统计区间截短。
    fn dProfPrefillEnd(ctx: *anyopaque) void {
        const self = self_(ctx);
        self.closePrefillWindow();
    }

    // 注：这里原有两段悬空的 doc comment（"Open a decode step" /
    // "Close a decode step"），它们描述的是已被删除的 S1 生命周期（含已经不存在
    // 的 warm 阶段），重构后没有对应函数。真正的开/关逻辑见 `dGraphBegin` 与
    // `dGraphEnd`。

    /// `VTable.selectGraphSlot`：切换 graph 状态槽。0 = decode，1 = verify。
    ///
    /// 两份 graph 形状不同（1 行 vs k 行），共用一槽会让
    /// `graphUpdateRequired` 每次切换都为真，等于每步重捕获。
    /// `VTable.graphDirect`：进入/离开"非 graph 管辖"段。MTP 的 draft 循环在
    /// verify 的 graph 之外运行，这段里的每个 op 都必须真正执行。


    fn dProfPhase(ctx: *anyopaque, on: bool) void {
        const self = self_(ctx);
        self.cpu.exec().profPhase(on);
        // 只是兜底：prefill 窗口通常已经被第一次 decode `matvec` 关掉
        // （`closePrefillWindow` 幂等）。这里接住的是 prefill 一次 decode matvec
        // 都没发过的退化情形（例如 `-n 0`），好让 `CUDA-M [prefill done]` 那一行
        // 照样出现。
        if (!on) self.closePrefillWindow();
    }

    fn dProfReset(ctx: *anyopaque) void {
        const self = self_(ctx);
        self.stat_ns_matvec = 0;
        self.stat_ns_batchk = 0;
        self.stat_ns_dequant = 0;
        self.stat_n_matvec = 0;
        self.stat_n_batchk = 0;
        self.stat_n_dequant = 0;
        self.stat_h2d_bytes = 0;
        self.stat_d2h_bytes = 0;
        self.stat_host_act_uploads = 0;
        self.stat_host_out_downloads = 0;
        self.stat_rope_trips = 0;
        self.stat_by_type = @splat(0);
        self.stat_ns_by_type = @splat(0);
        // GPU-side counters and the prefill window.  `ev_on` is set here and
        // nowhere else, and `profReset` is only reached from
        // `Engine.generate`, which runs it unconditionally -- so the gate is the
        // environment variable, checked once, and the default path never
        // creates an event.
        self.stat_ns_batchk_gpu = 0;
        self.stat_n_batchk_gpu = 0;
        self.stat_ns_prefill_span = 0;
        self.stat_ns_prefill_wall = 0;
        self.stat_ns_prefill_bk_wall = 0;
        self.stat_n_ev_err = 0;
        self.stat_n_ev_dropped = 0;
        self.stat_n_ev_suspect = 0;
        self.stat_n_bk_begin = 0;
        self.stat_n_bk_end = 0;
        self.ev_pending_n = 0;
        self.ev_pool_wr = 0;
        self.stat_n_launch_bk = 0;
        self.stat_ns_ev_sync = 0;
        // Per-slot timers (QWEN35_PROF only; the counters stay zero when off).
        self.stat_ns_slot = @splat(0);
        self.stat_n_slot = @splat(0);
        self.stat_ns_slot_total = 0;
        self.slot_depth = 0;
        // Per-slot DEVICE time ($QWEN35_PROF_GPU).  The pool is created lazily
        // on the first timed slot, not here, so a run without the variable never
        // touches an event.
        self.stat_gpu_ns_slot = @splat(0);
        self.stat_gpu_n_slot = @splat(0);
        self.stat_gpu_overflow = 0;
        self.stat_gpu_ev_err = 0;
        self.stat_gpu_ns_token_sum = 0;
        self.stat_gpu_n_token = 0;
        self.gpu_ev_used = 0;
        self.gpu_ev_open = null;
        self.gpu_ev_on = cudart.getenv("QWEN35_PROF_GPU") != null and self.cu != null;
        if (self.gpu_ev_on) {
            std.debug.print("cuda backend: $QWEN35_PROF_GPU per-slot device timing on ({d} event pairs)\n", .{gpu_ev_capacity});
        }
        if (cudart.getenv("QWEN35_PROF") != null) {
            self.ev_on = self.cu != null;
            if (self.ev_on) self.openPrefillWindow();
        } else {
            self.ev_on = false;
        }
        self.cpu.exec().profReset();
    }

    fn dReportProfile(ctx: *anyopaque) void {
        const self = self_(ctx);
        self.cpu.exec().reportProfile();
        self.reportStats("report");
    }

    /// `Exec.profReport`: the CUDA backend's complete one-shot dump.  Reaches
    /// the same `reportStats("final")` as the public `CudaBackend.profReport`;
    /// the engine only ever sees the `Exec`, so the vtable path is the one the
    /// CLI uses.
    fn dProfReport(ctx: *anyopaque) void {
        const self = self_(ctx);
        self.profReport();
    }
// ==== slot timing wrappers ($QWEN35_PROF only) ==============================
//
// One thin forwarder per `backend.Exec` slot.  Each one opens a clock, calls
// the unmodified `*_body` implementation, and closes the clock on the way out
// (`defer` covers the early-return and error paths inside the bodies, which is
// why the bodies keep their own `return` statements untouched).
//
// Why wrap instead of instrumenting in place: the bodies have several
// `return`/`catch return` exits each; adding a `slotExit` at every one of them
// is where a profiling patch starts lying.  A wrapper cannot miss an exit.
//
// With `$QWEN35_PROF` unset, `slotEnter` returns null after one predictable
// branch and `slotExit` returns immediately, so the plain path costs a call
// and a not-taken branch -- no clock read, no counter write.

    /// The `loadLinear` slot, timed (`$QWEN35_PROF`).
    fn vLoadLinear(ctx: *anyopaque, name: []const u8) backend.LoadError!backend.LinearW {
        const self = self_(ctx);
        const t0 = self.slotEnter();
        defer self.slotExit(.load_linear, t0);
        return vLoadLinear_body(ctx, name);
    }

    /// The `dequantRow` slot, timed (`$QWEN35_PROF`).
    fn vDequantRow(g: backend.Graph, ctx: *anyopaque, dstBuf: backend.Buf, w: gguf.QuantRow, row: usize) void {
        const self = self_(ctx);
        const xg_prev = xg_push_slot(g.slot);
        defer xg_cur_slot = xg_prev;
        const t0 = self.slotEnter();
        defer self.slotExit(.dequant_row, t0);
        vDequantRow_body(g, ctx, dstBuf, w, row);
    }

    /// The `matvec` slot, timed (`$QWEN35_PROF`).
    fn vMatvec(g: backend.Graph, ctx: *anyopaque, outBuf: backend.Buf, w: backend.LinearW, xBuf: backend.Buf) void {
        const self = self_(ctx);
        const xg_prev = xg_push_slot(g.slot);
        defer xg_cur_slot = xg_prev;
        const t0 = self.slotEnter();
        defer self.slotExit(.matvec, t0);
        if (self.ablated(.matvec)) return ;
        vMatvec_body(g, ctx, outBuf, w, xBuf);
    }

    /// The `matvecPair` slot; charged to `matvec` since it IS a matvec launch
    /// (or two), and a separate slot would only split the same total.
    fn vMatvecPair(g: backend.Graph, ctx: *anyopaque, outBuf: backend.Buf, a: backend.LinearW, b: backend.LinearW, xBuf: backend.Buf) void {
        const self = self_(ctx);
        const xg_prev = xg_push_slot(g.slot);
        defer xg_cur_slot = xg_prev;
        const t0 = self.slotEnter();
        defer self.slotExit(.matvec, t0);
        if (self.ablated(.matvec)) return ;
        vMatvecPair_body(g, ctx, outBuf, a, b, xBuf);
    }

    /// The `matvecBatchK` slot, timed (`$QWEN35_PROF`).
    fn vMatvecBatchK(g: backend.Graph, ctx: *anyopaque, outBuf: backend.Buf, w: backend.LinearW, xKBuf: backend.Buf, k: usize) void {
        const self = self_(ctx);
        const xg_prev = xg_push_slot(g.slot);
        defer xg_cur_slot = xg_prev;
        const t0 = self.slotEnter();
        defer self.slotExit(.matvec_batchk, t0);
        if (self.ablated(.matvec_batchk)) return ;
        vMatvecBatchK_body(g, ctx, outBuf, w, xKBuf, k);
    }

    /// The `allocScratch` slot, timed (`$QWEN35_PROF`).
    fn dAllocScratch(ctx: *anyopaque, max_cols: usize) anyerror!void {
        const self = self_(ctx);
        const t0 = self.slotEnter();
        defer self.slotExit(.alloc_scratch, t0);
        return dAllocScratch_body(ctx, max_cols);
    }

    /// The `allocScratchK` slot, timed (`$QWEN35_PROF`).
    fn dAllocScratchK(ctx: *anyopaque, max_cols: usize, k: usize) anyerror!void {
        const self = self_(ctx);
        const t0 = self.slotEnter();
        defer self.slotExit(.alloc_scratch_k, t0);
        return dAllocScratchK_body(ctx, max_cols, k);
    }

    /// The `allocBuf` slot, timed (`$QWEN35_PROF`).
    fn dAllocBuf(ctx: *anyopaque, len: usize) anyerror!backend.Buf {
        const self = self_(ctx);
        const t0 = self.slotEnter();
        defer self.slotExit(.alloc_buf, t0);
        return dAllocBuf_body(ctx, len);
    }

    /// The GDN q/k normalisation as one launch (see `backend.Exec`).
    /// Not a timed slot of its own: it replaces `l2norm_rows` x2 + `scale_vec`,
    /// whose slots it would otherwise have to charge in fractions.
    fn dL2normQkScale(g: backend.Graph, ctx: *anyopaque, x: backend.Buf, nk: usize, kd: usize, k_offset: usize, qscale: f32) void {
        dL2normQkScale_body(g, ctx, x, nk, kd, k_offset, qscale);
    }

    /// The `scaleVec` slot, timed (`$QWEN35_PROF`).
    fn dScaleVec(g: backend.Graph, ctx: *anyopaque, x: backend.Buf, s: f32) void {
        const self = self_(ctx);
        const xg_prev = xg_push_slot(g.slot);
        defer xg_cur_slot = xg_prev;
        const t0 = self.slotEnter();
        defer self.slotExit(.scale_vec, t0);
        if (self.ablated(.scale_vec)) return ;
        dScaleVec_body(g, ctx, x, s);
    }

    /// The `addResid` slot, timed (`$QWEN35_PROF`).
    fn dAddResid(g: backend.Graph, ctx: *anyopaque, dst: backend.Buf, src: backend.Buf) void {
        const self = self_(ctx);
        const xg_prev = xg_push_slot(g.slot);
        defer xg_cur_slot = xg_prev;
        const t0 = self.slotEnter();
        defer self.slotExit(.add_resid, t0);
        if (self.ablated(.add_resid)) return ;
        dAddResid_body(g, ctx, dst, src);
    }

    /// The `copyRange` slot, timed (`$QWEN35_PROF`).
    fn dCopyRange(g: backend.Graph, ctx: *anyopaque, dst: backend.Buf, src: backend.Buf) void {
        const self = self_(ctx);
        const xg_prev = xg_push_slot(g.slot);
        defer xg_cur_slot = xg_prev;
        const t0 = self.slotEnter();
        // Per-width accounting, alongside the slot total.  The bucket comes from
        // `src.len` alone, so it is chosen on the way in and does not need the
        // route decision the body makes; two defers keep the body call in the
        // tail position so the fast path is unchanged when profiling is off.
        if (t0) |t| {
            const b = &self.stat_copy_bins[copyBinOf(src.len)];
            b.n += 1;
            b.elems += src.len;
            dCopyRange_body(g, ctx, dst, src);
            const ns = durNs(t, self.io);
            b.ns += ns;
            self.slotExit(.copy_range, t);
            return;
        }
        dCopyRange_body(g, ctx, dst, src);
    }

    /// KV append, counted in the same slot as `copyRange` (it IS a copy; the
    /// only difference is that the capture records its destination).  Kept
    /// apart from `dCopyRange` so the recording cannot be reached from the
    /// generic path, where a moving destination does not exist.
    fn dCopyRangeSlot(g: backend.Graph, ctx: *anyopaque, dst: backend.Buf, src: backend.Buf, slot: usize) void {
        const self = self_(ctx);
        const xg_prev = xg_push_slot(g.slot);
        defer xg_cur_slot = xg_prev;
        const t0 = self.slotEnter();
        if (t0) |t| {
            const b = &self.stat_copy_bins[copyBinOf(src.len)];
            b.n += 1;
            b.elems += src.len;
            dCopyRangeSlot_body(g, ctx, dst, src, slot);
            const ns = durNs(t, self.io);
            b.ns += ns;
            self.slotExit(.copy_range, t);
            return;
        }
        dCopyRangeSlot_body(g, ctx, dst, src, slot);
    }

    /// The `deinterleaveQG` slot, timed (`$QWEN35_PROF`).
    ///
    /// Every argument is a device region in the P2c engine (`qK`/`z`/`qkv` are
    /// all backend allocations), so this is the one launch that replaces
    /// `2*heads` `copyRange` launches.  Not a `cudaMemcpy`: the two halves of a
    /// head are interleaved with the *other* half, so no single linear copy
    /// can produce the dense layout -- hence the kernel.
    fn dDeinterleaveQG(g: backend.Graph, ctx: *anyopaque, dst_q: backend.Buf, dst_gate: backend.Buf, src: backend.Buf, heads: usize, hd: usize) void {
        const self = self_(ctx);
        const xg_prev = xg_push_slot(g.slot);
        defer xg_cur_slot = xg_prev;
        const t0 = self.slotEnter();
        defer self.slotExit(.deinterleave_qg, t0);
        if (self.ablated(.deinterleave_qg)) return ;
        if (self.cu != null) {
            if (dBufPtr(dst_q)) |pq| {
                if (dBufPtr(dst_gate)) |pg| {
                    if (dBufPtr(src)) |ps| {
                        const rc = Launch.deinterleave_qg(g, pq, pg, ps, @intCast(heads), @intCast(hd), self.launchStream());
                        if (rc == cudart.cudaSuccess) return;
                        self.ewLaunchFail("deinterleave_qg", rc);
                    }
                }
            }
        }
        // CPU fallback: either there is no device, or a pointer was not a
        // device region (host `Buf`s reach this vtable only in tests).
        cpu_mod.CpuBackend.vtable.deinterleaveQG(.{}, @ptrCast(&self.cpu), dst_q, dst_gate, src, heads, hd);
    }

    /// Chunk-wide `deinterleaveQG`: `tokens` rows in one launch.  Same move as
    /// `dDeinterleaveQG`, so this is exact; it exists because the per-row form
    /// cost 940 launches per 525-token prefill (1.2% of device time at 6 us).
    fn dDeinterleaveQGStrided(g: backend.Graph, ctx: *anyopaque, dst_q: backend.Buf, dst_gate: backend.Buf,
                              src: backend.Buf, heads: usize, hd: usize, tokens: usize) void {
        const self = self_(ctx);
        const t0 = self.slotEnter();
        defer self.slotExit(.deinterleave_qg, t0);
        if (self.ablated(.deinterleave_qg)) return;
        const row_len = heads * hd;
        const shapes_ok = tokens != 0 and heads != 0 and hd != 0 and
            dst_q.len >= tokens * row_len and dst_gate.len >= tokens * row_len and
            src.len >= tokens * 2 * row_len;
        if (self.cu != null and shapes_ok) {
            if (dBufPtr(dst_q)) |pq| {
                if (dBufPtr(dst_gate)) |pg| {
                    if (dBufPtr(src)) |ps| {
                        const rc = Launch.deinterleave_qg_strided(g, pq, pg, ps,
                            @intCast(heads), @intCast(hd), @intCast(tokens), self.launchStream());
                        if (rc == cudart.cudaSuccess) return;
                        self.ewLaunchFail("deinterleave_qg_strided", rc);
                    }
                }
            }
        }
        cpu_mod.CpuBackend.vtable.deinterleaveQGStrided(.{}, @ptrCast(&self.cpu), dst_q, dst_gate, src, heads, hd, tokens);
    }

    /// The `zeroBuf` slot, timed (`$QWEN35_PROF`).
    fn dZeroBuf(g: backend.Graph, ctx: *anyopaque, buf: backend.Buf) void {
        const self = self_(ctx);
        const xg_prev = xg_push_slot(g.slot);
        defer xg_cur_slot = xg_prev;
        const t0 = self.slotEnter();
        defer self.slotExit(.zero_buf, t0);
        if (self.ablated(.zero_buf)) return ;
        dZeroBuf_body(g, ctx, buf);
    }

    /// The `copyFromHost` slot, timed (`$QWEN35_PROF`).
    fn dCopyFromHost(g: backend.Graph, ctx: *anyopaque, dst: backend.Buf, src: []const f32) void {
        const self = self_(ctx);
        const xg_prev = xg_push_slot(g.slot);
        defer xg_cur_slot = xg_prev;
        const t0 = self.slotEnter();
        defer self.slotExit(.copy_from_host, t0);
        if (self.ablated(.copy_from_host)) return ;
        dCopyFromHost_body(g, ctx, dst, src);
    }

    /// The `copyToHost` slot, timed (`$QWEN35_PROF`).
    fn dCopyToHost(g: backend.Graph, ctx: *anyopaque, dst: []f32, src: backend.Buf) void {
        const self = self_(ctx);
        const xg_prev = xg_push_slot(g.slot);
        defer xg_cur_slot = xg_prev;
        const t0 = self.slotEnter();
        defer self.slotExit(.copy_to_host, t0);
        dCopyToHost_body(g, ctx, dst, src);
    }

    /// The `rmsnorm` slot, timed (`$QWEN35_PROF`).
    fn dRmsnorm(g: backend.Graph, ctx: *anyopaque, out: backend.Buf, x: backend.Buf, w: backend.Buf, eps: f32) void {
        const self = self_(ctx);
        const xg_prev = xg_push_slot(g.slot);
        defer xg_cur_slot = xg_prev;
        const t0 = self.slotEnter();
        defer self.slotExit(.rmsnorm, t0);
        if (self.ablated(.rmsnorm)) return ;
        dRmsnorm_body(g, ctx, out, x, w, eps);
    }

    /// Row-batched twin of `dRmsnorm`.  Charged to the `rmsnorm` slot so the
    /// `PROF-SLOT` total for rmsnorm stays comparable across the change (only
    /// its `calls` count moves, which is the point of the merge).
    fn dRmsnormRows(g: backend.Graph, ctx: *anyopaque, out: backend.Buf, x: backend.Buf, w: backend.Buf, rows_in: usize, eps: f32) void {
        const self = self_(ctx);
        const xg_prev = xg_push_slot(g.slot);
        defer xg_cur_slot = xg_prev;
        const t0 = self.slotEnter();
        defer self.slotExit(.rmsnorm, t0);
        if (self.ablated(.rmsnorm)) return ;
        dRmsnormRows_body(g, ctx, out, x, w, rows_in, eps);
    }

    /// `l2norm(slot)`, timed (`$QWEN35_PROF`).
    fn dL2norm(g: backend.Graph, ctx: *anyopaque, x: backend.Buf) void {
        const self = self_(ctx);
        const xg_prev = xg_push_slot(g.slot);
        defer xg_cur_slot = xg_prev;
        const t0 = self.slotEnter();
        defer self.slotExit(.l2norm, t0);
        if (self.ablated(.l2norm)) return;
        dL2norm_body(g, ctx, x);
    }

    /// Row-batched twin of `dL2norm`; charged to the `l2norm` slot.
    fn dL2normRows(g: backend.Graph, ctx: *anyopaque, x: backend.Buf, n: usize) void {
        const self = self_(ctx);
        const xg_prev = xg_push_slot(g.slot);
        defer xg_cur_slot = xg_prev;
        const t0 = self.slotEnter();
        defer self.slotExit(.l2norm, t0);
        dL2normRows_body(g, ctx, x, n);
    }

    /// The `siluInPlace` slot, timed (`$QWEN35_PROF`).
    fn dSiluInPlace(g: backend.Graph, ctx: *anyopaque, x: backend.Buf) void {
        const self = self_(ctx);
        const xg_prev = xg_push_slot(g.slot);
        defer xg_cur_slot = xg_prev;
        const t0 = self.slotEnter();
        defer self.slotExit(.silu, t0);
        if (self.ablated(.silu)) return ;
        dSiluInPlace_body(g, ctx, x);
    }

    /// The `mulSilu` slot, timed (`$QWEN35_PROF`).
    fn dMulSilu(g: backend.Graph, ctx: *anyopaque, out: backend.Buf, gate: backend.Buf) void {
        const self = self_(ctx);
        const xg_prev = xg_push_slot(g.slot);
        defer xg_cur_slot = xg_prev;
        const t0 = self.slotEnter();
        defer self.slotExit(.mul_silu, t0);
        if (self.ablated(.mul_silu)) return ;
        dMulSilu_body(g, ctx, out, gate);
    }

    /// The `mulSigmoid` slot, timed (`$QWEN35_PROF`).
    fn dMulSigmoid(g: backend.Graph, ctx: *anyopaque, out: backend.Buf, gate: backend.Buf) void {
        const self = self_(ctx);
        const xg_prev = xg_push_slot(g.slot);
        defer xg_cur_slot = xg_prev;
        const t0 = self.slotEnter();
        defer self.slotExit(.mul_sigmoid, t0);
        if (self.ablated(.mul_sigmoid)) return ;
        dMulSigmoid_body(g, ctx, out, gate);
    }

    /// The `swiglu` slot, timed (`$QWEN35_PROF`).
    fn dSwiglu(g: backend.Graph, ctx: *anyopaque, gate: backend.Buf, up: backend.Buf) void {
        const self = self_(ctx);
        const xg_prev = xg_push_slot(g.slot);
        defer xg_cur_slot = xg_prev;
        const t0 = self.slotEnter();
        defer self.slotExit(.swiglu, t0);
        if (self.ablated(.swiglu)) return ;
        dSwiglu_body(g, ctx, gate, up);
    }

    /// The `softmax` slot, timed (`$QWEN35_PROF`).
    fn dSoftmax(g: backend.Graph, ctx: *anyopaque, x: backend.Buf) void {
        const self = self_(ctx);
        const xg_prev = xg_push_slot(g.slot);
        defer xg_cur_slot = xg_prev;
        const t0 = self.slotEnter();
        defer self.slotExit(.softmax, t0);
        if (self.ablated(.softmax)) return ;
        dSoftmax_body(g, ctx, x);
    }

    /// The `argmax` slot, timed (`$QWEN35_PROF`).
    fn dArgmax(g: backend.Graph, ctx: *anyopaque, logits: backend.Buf) u32 {
        const self = self_(ctx);
        const xg_prev = xg_push_slot(g.slot);
        defer xg_cur_slot = xg_prev;
        const t0 = self.slotEnter();
        defer self.slotExit(.argmax, t0);
        if (self.ablated(.argmax)) return 0;
        return dArgmax_body(g, ctx, logits);
    }

    /// The `sample` slot, timed (`$QWEN35_PROF`).
    fn dSample(g: backend.Graph, ctx: *anyopaque, logits: backend.Buf, scratch: backend.Buf, sp: *sampler.Sampler) u32 {
        const self = self_(ctx);
        const xg_prev = xg_push_slot(g.slot);
        defer xg_cur_slot = xg_prev;
        const t0 = self.slotEnter();
        defer self.slotExit(.sample, t0);
        if (self.ablated(.sample)) return 0;
        return dSample_body(g, ctx, logits, scratch, sp);
    }

    /// The `gdnStep` slot, timed (`$QWEN35_PROF`).
    fn dGdnStep(g: backend.Graph, ctx: *anyopaque, state: backend.Buf, k: backend.Buf, v: backend.Buf,
                q: backend.Buf, km: backend.Buf, delta: backend.Buf, out: backend.Buf,
                gamma: f32, beta: f32, vd: usize) void {
        const self = self_(ctx);
        const t0 = self.slotEnter();
        defer self.slotExit(.gdn_step, t0);
        if (self.ablated(.gdn_step)) return ;
        dGdnStep_body(g, ctx, state, k, v, q, km, delta, out, gamma, beta, vd);
    }

    /// The batched `gdnStep` slot, timed under the SAME `.gdn_step` slot as the
    /// single-row one: it does the same work for all heads, so the `PROF-SLOT`
    /// line stays comparable across the change (calls drops by `heads`x, which
    /// is the check that the batching actually took effect).
    fn dGdnStepRows(g: backend.Graph, ctx: *anyopaque, state: backend.Buf, k: backend.Buf, v: backend.Buf,
                    q: backend.Buf, km: backend.Buf, delta: backend.Buf, out: backend.Buf,
                    gamma: backend.Buf, beta: backend.Buf,
                    heads: usize, nk: usize, kd: usize, vd: usize) void {
        const self = self_(ctx);
        const t0 = self.slotEnter();
        defer self.slotExit(.gdn_step, t0);
        if (self.ablated(.gdn_step)) return ;
        dGdnStepRows_body(g, ctx, state, k, v, q, km, delta, out, gamma, beta, heads, nk, kd, vd);
    }

    /// Chunk-wide `gdnStepRows` (`VTable.gdnStepRowsChunk`): the same state
    /// recurrence for `rows` tokens in ONE launch.
    ///
    /// The steps are sequential -- `state` is carried -- so this does not add
    /// parallelism.  What it removes is the launch: the per-token loop issued
    /// 10,500 launches for a 525-token prefill (48 GDN layers), whose device
    /// time was 42 ms but whose wall clock was 641 ms.  Deleting the loop
    /// outright (measurement only) took prefill 299 -> 453 tok/s.
    fn dGdnStepRowsChunk_body(g: backend.Graph, ctx: *anyopaque, state: backend.Buf, k: backend.Buf, v: backend.Buf,
                              q: backend.Buf, km: backend.Buf, delta: backend.Buf, out: backend.Buf,
                              gamma: backend.Buf, beta: backend.Buf,
                              heads: usize, nk: usize, kd: usize, vd: usize, rows: usize,
                              k_stride: usize, v_stride: usize, q_stride: usize) void {
        const self = self_(ctx);
        // Same shape gate as the per-row body, plus the strides the chunk form
        // needs.  The last row must be fully inside each buffer, so the length
        // check uses (rows-1)*stride + the row width -- not rows*stride, which
        // would reject a view whose tail exactly ends the buffer.
        const need = struct {
            fn n(r: usize, s: usize, w: usize) usize {
                return if (r == 0) 0 else (r - 1) * s + w;
            }
        }.n;
        if (vd == 128 and kd == 128 and heads != 0 and nk != 0 and nk <= heads and rows != 0 and
            gamma.len >= rows * heads and beta.len >= rows * heads and
            state.len >= heads * kd * vd and
            k.len >= need(rows, k_stride, nk * kd) and q.len >= need(rows, q_stride, nk * kd) and
            v.len >= need(rows, v_stride, heads * vd) and
            out.len >= rows * heads * vd and
            km.len >= rows * heads * vd and delta.len >= rows * heads * vd and
            self.cu != null)
        {
            if (dBufPtr(state)) |pstate| {
                if (dBufPtr(k)) |pk| {
                    if (dBufPtr(v)) |pv| {
                        if (dBufPtr(q)) |pq| {
                            if (dBufPtr(km)) |pkm| {
                                if (dBufPtr(delta)) |pdelta| {
                                    if (dBufPtr(out)) |pout| {
                                        if (dBufPtr(gamma)) |pgamma| {
                                            if (dBufPtr(beta)) |pbeta| {
                                                // WY / UT-transform form: same
                                                // result, blocks of C tokens
                                                // instead of one at a time.
                                                // `QWEN35_GDN_WY=0` keeps the
                                                // per-token kernel for A/B (that
                                                // kernel stays as the reference
                                                // these numbers were checked
                                                // against).
                                                const use_wy = gdnWyEnabled();
                                                const rc = if (use_wy)
                                                    Launch.gdn_step_wy(g, 
                                                        pstate, @intCast(kd * vd),
                                                        pk, @intCast(k_stride),
                                                        pv, @intCast(v_stride),
                                                        pq, @intCast(q_stride),
                                                        pkm, @intCast(heads * vd),
                                                        pdelta, @intCast(heads * vd),
                                                        pout, @intCast(heads * vd),
                                                        pgamma, @intCast(heads),
                                                        pbeta, @intCast(heads),
                                                        @intCast(heads), @intCast(nk),
                                                        @intCast(kd), @intCast(vd), @intCast(rows),
                                                        self.launchStream(),
                                                    )
                                                else
                                                    Launch.gdn_step_rows_chunk(g, 
                                                        pstate, @intCast(kd * vd),
                                                        pk, @intCast(k_stride),
                                                        pv, @intCast(v_stride),
                                                        pq, @intCast(q_stride),
                                                        pkm, @intCast(heads * vd),
                                                        pdelta, @intCast(heads * vd),
                                                        pout, @intCast(heads * vd),
                                                        pgamma, @intCast(heads),
                                                        pbeta, @intCast(heads),
                                                        @intCast(heads), @intCast(nk),
                                                        @intCast(kd), @intCast(vd), @intCast(rows),
                                                        self.launchStream(),
                                                    );
                                                // A shape the WY kernel declines
                                                // (it only takes kd==128 and
                                                // vd%64==0) falls back rather
                                                // than failing the whole step.
                                                if (!use_wy or rc == cudart.cudaSuccess) {
                                                    if (rc == cudart.cudaSuccess) return;
                                                    self.ewLaunchFail("gdn_step_rows_chunk", rc);
                                                }
                                                const rc2 = Launch.gdn_step_rows_chunk(g, 
                                                    pstate, @intCast(kd * vd),
                                                    pk, @intCast(k_stride),
                                                    pv, @intCast(v_stride),
                                                    pq, @intCast(q_stride),
                                                    pkm, @intCast(heads * vd),
                                                    pdelta, @intCast(heads * vd),
                                                    pout, @intCast(heads * vd),
                                                    pgamma, @intCast(heads),
                                                    pbeta, @intCast(heads),
                                                    @intCast(heads), @intCast(nk),
                                                    @intCast(kd), @intCast(vd), @intCast(rows),
                                                    self.launchStream(),
                                                );
                                                if (rc2 == cudart.cudaSuccess) return;
                                                self.ewLaunchFail("gdn_step_rows_chunk", rc2);
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        // Fallback: the CPU oracle walks the rows one at a time, so a shape or
        // residency miss costs speed, never correctness.
        cpu_mod.CpuBackend.vtable.gdnStepRowsChunk(.{}, @ptrCast(&self.cpu), state, k, v, q, km, delta, out, gamma, beta, heads, nk, kd, vd, rows, k_stride, v_stride, q_stride);
    }

    /// `VTable.gdnStepRowsSnap`：chunk 递推 + 每行一份状态快照。
    ///
    /// 永远走逐 token 内核，不用 WY：WY 按 C 个 token 分块求解，只有块尾才把
    /// `state` 写回，块内"第 r 行之后的状态"不作为整块存在。要在那里发快照
    /// 就得逐行重新导出状态——正是这里要避免的活。
    ///
    /// 代价：只有 decode 的 verify 会调（`rows <= spec_verify_rows`），prefill
    /// 热路径完全不受影响。
    fn dGdnStepRowsSnap_body(g: backend.Graph, ctx: *anyopaque, state: backend.Buf, k: backend.Buf, v: backend.Buf,
                             q: backend.Buf, km: backend.Buf, delta: backend.Buf, out: backend.Buf,
                             gamma: backend.Buf, beta: backend.Buf, snap: backend.Buf,
                             heads: usize, nk: usize, kd: usize, vd: usize, rows: usize,
                             k_stride: usize, v_stride: usize, q_stride: usize) void {
        const self = self_(ctx);
        const per = heads * kd * vd;
        if (vd == 128 and kd == 128 and heads != 0 and nk != 0 and nk <= heads and rows != 0 and
            gamma.len >= rows * heads and beta.len >= rows * heads and
            state.len >= per and snap.len >= rows * per and
            out.len >= rows * heads * vd and
            km.len >= rows * heads * vd and delta.len >= rows * heads * vd and
            self.cu != null)
        {
            if (dBufPtr(state)) |pstate| {
                if (dBufPtr(k)) |pk| {
                    if (dBufPtr(v)) |pv| {
                        if (dBufPtr(q)) |pq| {
                            if (dBufPtr(km)) |pkm| {
                                if (dBufPtr(delta)) |pdelta| {
                                    if (dBufPtr(out)) |pout| {
                                        if (dBufPtr(gamma)) |pgamma| {
                                            if (dBufPtr(beta)) |pbeta| {
                                                if (dBufPtr(snap)) |psnap| {
                                                    const rc = Launch.gdn_step_rows_snap(g, 
                                                        pstate, @intCast(kd * vd),
                                                        pk, @intCast(k_stride),
                                                        pv, @intCast(v_stride),
                                                        pq, @intCast(q_stride),
                                                        pkm, @intCast(heads * vd),
                                                        pdelta, @intCast(heads * vd),
                                                        pout, @intCast(heads * vd),
                                                        pgamma, @intCast(heads),
                                                        pbeta, @intCast(heads),
                                                        psnap, @intCast(rows),
                                                        @intCast(heads), @intCast(nk),
                                                        @intCast(kd), @intCast(vd), @intCast(rows),
                                                        self.launchStream(),
                                                    );
                                                    if (rc == cudart.cudaSuccess) return;
                                                    self.ewLaunchFail("gdn_step_rows_snap", rc);
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        // 形状或驻留不满足：走 CPU oracle（逐行 + 逐行快照），只慢不错。
        cpu_mod.CpuBackend.vtable.gdnStepRowsSnap(.{}, @ptrCast(&self.cpu), state, k, v, q, km, delta, out, gamma, beta, snap, heads, nk, kd, vd, rows, k_stride, v_stride, q_stride);
    }

    fn dGdnStepRowsSnap(g: backend.Graph, ctx: *anyopaque, state: backend.Buf, k: backend.Buf, v: backend.Buf,
                        q: backend.Buf, km: backend.Buf, delta: backend.Buf, out: backend.Buf,
                        gamma: backend.Buf, beta: backend.Buf, snap: backend.Buf,
                        heads: usize, nk: usize, kd: usize, vd: usize, rows: usize,
                        k_stride: usize, v_stride: usize, q_stride: usize) void {
        const self = self_(ctx);
        const t0 = self.slotEnter();
        defer self.slotExit(.gdn_step, t0);
        if (self.ablated(.gdn_step)) return ;
        dGdnStepRowsSnap_body(g, ctx, state, k, v, q, km, delta, out, gamma, beta, snap, heads, nk, kd, vd, rows, k_stride, v_stride, q_stride);
    }

    fn dGdnStepRowsChunk(g: backend.Graph, ctx: *anyopaque, state: backend.Buf, k: backend.Buf, v: backend.Buf,
                         q: backend.Buf, km: backend.Buf, delta: backend.Buf, out: backend.Buf,
                         gamma: backend.Buf, beta: backend.Buf,
                         heads: usize, nk: usize, kd: usize, vd: usize, rows: usize,
                         k_stride: usize, v_stride: usize, q_stride: usize) void {
        const self = self_(ctx);
        const t0 = self.slotEnter();
        defer self.slotExit(.gdn_step, t0);
        if (self.ablated(.gdn_step)) return ;
        dGdnStepRowsChunk_body(g, ctx, state, k, v, q, km, delta, out, gamma, beta, heads, nk, kd, vd, rows, k_stride, v_stride, q_stride);
    }

    /// The `gdnGamma` slot, timed (`$QWEN35_PROF`).
    fn dGdnGamma(g: backend.Graph, ctx: *anyopaque, a_raw: backend.Buf, b_raw: backend.Buf, ssm_a: backend.Buf,
                 dt_bias: backend.Buf, gamma: backend.Buf, nv: usize) void {
        const self = self_(ctx);
        const t0 = self.slotEnter();
        defer self.slotExit(.gdn_gamma, t0);
        if (self.ablated(.gdn_gamma)) return ;
        dGdnGamma_body(g, ctx, a_raw, b_raw, ssm_a, dt_bias, gamma, nv);
    }

    /// The prefill-only GDN prelude, timed separately so a profile shows
    /// directly how many launches the per-token form used to cost.
    fn dGdnPrefillCore(g: backend.Graph, ctx: *anyopaque, qkv: backend.Buf, conv: backend.Buf,
                       wconv: backend.Buf, ab: backend.Buf, ssm_a: backend.Buf,
                       dt_bias: backend.Buf, gamma: backend.Buf, tokens: usize,
                       conv_dim: usize, ab_a_offset: usize, ab_b_offset: usize,
                       ab_stride: usize, gamma_stride: usize, kq_dim: usize,
                       nv: usize, nk: usize, kd: usize, conv_k: usize, eps: f32) void {
        const self = self_(ctx);
        const t0 = self.slotEnter();
        defer self.slotExit(.gdn_prelude, t0);
        dGdnPrefillCore_body(g, ctx, qkv, conv, wconv, ab, ssm_a, dt_bias, gamma, tokens,
            conv_dim, ab_a_offset, ab_b_offset, ab_stride, gamma_stride, kq_dim,
            nv, nk, kd, conv_k, eps);
    }

    /// The `attnDecode` slot, timed (`$QWEN35_PROF`).
    fn dAttnDecode(g: backend.Graph, ctx: *anyopaque, q: backend.Buf, gate: backend.Buf, kcache: backend.Buf,
                   vcache: backend.Buf, att: backend.Buf, gout: backend.Buf, slot: usize,
                   heads: usize, gqa: usize, hd: usize, kv_dim: usize, scale: f32) void {
        const self = self_(ctx);
        const t0 = self.slotEnter();
        defer self.slotExit(.attn_decode, t0);
        if (self.ablated(.attn_decode)) return ;
        dAttnDecode_body(g, ctx, q, gate, kcache, vcache, att, gout, slot, heads, gqa, hd, kv_dim, scale);
    }

    /// `attnChunk` -- the chunk-wide prefill attention, timed separately
    /// (`$QWEN35_PROF` shows `attn_chunk` beside `attn_decode`).
    ///
    /// Falls back to the per-row `dAttnDecode_body` whenever the device path is
    /// unavailable or the caller's buffers do not match the chunk contract, so
    /// a size violation degrades to known-good behaviour instead of a launch on
    /// bad pointers.  The CUDA kernel was gated bit-identical against exactly
    /// that per-row form (`/tmp/gate_attn_chunk.cu`).
    fn dAttnChunk(g: backend.Graph, ctx: *anyopaque, q: backend.Buf, gate: backend.Buf, kcache: backend.Buf,
                  vcache: backend.Buf, att: backend.Buf, gout: backend.Buf, slot0: usize,
                  rows: usize, heads: usize, gqa: usize, hd: usize, kv_dim: usize,
                  scale: f32) void {
        const self = self_(ctx);
        const t0 = self.slotEnter();
        defer self.slotExit(.attn_chunk, t0);
        if (self.ablated(.attn_chunk)) return;

        const row_len = heads * hd;
        const n_tot = slot0 + rows;
        // Bytes one cache row occupies in the format in use.  A cached region is
        // an f32 VIEW over its byte allocation, so a packed row is only
        // `row_bytes/4` elements long here; checking against `n_tot * kv_dim`
        // would compare bytes with elements and reject every quantized cache.
        // MEASURED (2026-11): with q8_0 the check was false at all times, so
        // every prefill silently took the per-row fallback below, and at
        // slot0=1024 that path handed device pointers to the CPU dequant.
        const cache_row_elems: usize = (backend.kvRowBytes(kv_dim, self.kv_type) + 3) / 4;
        // The gate kernel that the per-row path runs separately is folded into
        // the chunk kernel, so this path is only taken when EVERY input lines up.
        const shapes_ok = rows != 0 and heads != 0 and hd != 0 and gqa != 0 and
            q.len >= rows * row_len and gate.len >= rows * row_len and
            gout.len >= rows * row_len and
            kcache.len >= n_tot * cache_row_elems and vcache.len >= n_tot * cache_row_elems;

        // A chunk appended to a long KV cache can need more `s_scores` than the
        // 48 KB dynamic budget allows; `q35_attn_chunk_max_rows` reports the
        // slice size, and slicing is safe because row r depends only on earlier
        // rows and the cache prefix it needs.
        var maxrows: usize = 0;
        if (self.cu != null and shapes_ok) {
            const m = Launch.attn_chunk_max_rows(@intCast(slot0), @intCast(hd), self.kvKind());
            if (m > 0) maxrows = @intCast(m);
        }

        if (maxrows != 0) {
            // CUDA-graph 重放：起始行必须从设备读，dynamic shared 固定最大值。
            // 否则捕获时固化的 `slot0`/`smem` 会让每个 chunk 都要重新捕获。
            //
            // **起始行的来源是每步的设备控制块**（`ctl + XD_CTL_KV_SLOT`），
            // 不是每槽另开一块 int：那个块已经由 `dGraphBegin` 里的
            // `publishCtl` 在 capture 之外写好了，rope 也在读它。以前这里走
            // `s.d_attn_slot`，而那个指针从来没有被分配过（只有声明），所以
            // 这段路径一次都没生效 —— k 行前向只能走 per-slice 的
            // `attn_chunk`，它的 `slot0` 是普通实参、被烘死在图里。decode 只有
            // 1 行，且每轮别的地方会重录，问题不明显；verify 的 k 行起始行每轮
            // 都在前进，于是每一轮都读到录制那一行（实测：两行 argmax 完全
            // 相同、anchor 不推进、接受率 0%）。
            if (self.graphActive(g) and self.ensureCtl() != null and
                rows <= maxrows and self.kvKind() == 0 and n_tot <= 12288)
            {
                if (dBufPtr(q)) |pq| {
                    if (dBufPtr(gate)) |pgate| {
                        if (dBufPtr(kcache)) |pk| {
                            if (dBufPtr(vcache)) |pv| {
                                if (dBufPtr(gout)) |pgout| {
                                    // `d_slot0` 指向控制块的第 KV_SLOT 个 int。
                                    const ctl_base: [*]const c_int = @ptrCast(@alignCast(self.ctl_dev.?));
                                    const rc = Launch.attn_chunk_dyn(
                                        g,
                                        pq, pgate, pk, pv, pgout,
                                        @ptrCast(ctl_base + ctl_kv_slot),
                                        @intCast(rows), @intCast(heads), @intCast(gqa),
                                        @intCast(hd), @intCast(kv_dim), scale,
                                        self.kvKind(), self.launchStream());
                                    if (rc == cudart.cudaSuccess) return;
                                    self.ewLaunchFail("attn_chunk_dyn", rc);
                                }
                            }
                        }
                    }
                }
            }

            if (dBufPtr(q)) |pq| {
                if (dBufPtr(gate)) |pgate| {
                    if (dBufPtr(kcache)) |pk| {
                        if (dBufPtr(vcache)) |pv| {
                            if (dBufPtr(gout)) |pgout| {
                                var done: usize = 0;
                                while (done < rows) {
                                    // `maxrows` shrinks as `slot0` grows, so
                                    // recompute it per slice rather than
                                    // trusting the first value.
                                    const m2 = Launch.attn_chunk_max_rows(@intCast(slot0 + done), @intCast(hd), self.kvKind());
                                    if (m2 <= 0) break;
                                    const take = @min(rows - done, @as(usize, @intCast(m2)));
                                    const rc = Launch.attn_chunk(g, 
                                        pq + done * row_len, pgate + done * row_len,
                                        pk, pv,
                                        pgout + done * row_len,
                                        @intCast(slot0 + done), @intCast(take),
                                        @intCast(heads), @intCast(gqa), @intCast(hd),
                                        @intCast(kv_dim), scale, self.kvKind(), self.launchStream());
                                    if (rc != 0) {
                                        std.debug.print("cuda backend: attn_chunk launch failed ({d})\n", .{rc});
                                        break;
                                    }
                                    done += take;
                                }
                                if (done == rows) return;
                            }
                        }
                    }
                }
            }
        }
        // Fall back row by row (bad shapes, no device, or a rejected slice).
        var r: usize = 0;
        while (r < rows) : (r += 1) {
            dAttnDecode_body(g, ctx,
                q.span(r * row_len, row_len),
                gate.span(r * row_len, row_len),
                kcache, vcache, att,
                gout.span(r * row_len, row_len),
                slot0 + r, heads, gqa, hd, kv_dim, scale);
        }
    }

    /// `rope64(slot)`, timed (`$QWEN35_PROF`).
    fn dRope64(g: backend.Graph, ctx: *anyopaque, x: backend.Buf, pos: u32, base: f32) void {
        const self = self_(ctx);
        const xg_prev = xg_push_slot(g.slot);
        defer xg_cur_slot = xg_prev;
        const t0 = self.slotEnter();
        defer self.slotExit(.rope64, t0);
        if (self.ablated(.rope64)) return;
        dRope64_body(g, ctx, x, pos, base);
    }

    /// The prefill row-batched `rope64` slot, timed (`$QWEN35_PROF`).
    /// `ropeRows` (variable rotary width).  NOT implemented on the device yet.
    ///
    /// It cannot delegate to the CPU twin: `vRopeRows` takes a host slice, and
    /// on this backend `x` is a DEVICE address, so the CPU path dereferences
    /// device memory and dies without a stack trace.  Refusing loudly is the
    /// only honest option until a device kernel exists -- and the engine that
    /// needs it (`engines/qwen3.zig`, full-head rotation) is CPU-verified, so
    /// a clear error is strictly better than a segfault.
    ///
    /// To implement: a `k_rope_rows` kernel taking `(x, rows, heads, hd,
    /// rotary_dim, pos, base)`; `rotary_dim` is uniform across the vector so it
    /// can be a template/register argument rather than a runtime branch.
    /// Rotary embedding with a runtime rotary width: `rows` rows x `heads`
    /// heads of `hd` floats, rotating each head's leading `rotary_dim`.
    ///
    /// Qwen3.5/3.8 reach `rope64Rows` (rotary width 64); Qwen3 rotates the whole
    /// head, so it reaches this slot with `rotary_dim == hd == 128`.  The two
    /// kernels are the same arithmetic with the width as a parameter, and
    /// `rotary_dim == 64` reproduces `rope64Rows` bit for bit (see rope.cu).
    ///
    /// PREFILL ONLY in intent, but Qwen3 has no batched prefill yet, so this is
    /// also reached from decode.  There is no device-position twin of the kernel
    /// (unlike `rope64Rows`, which has `_dpos` variants the decode graph
    /// retargets), so if a capture ever IS open the node would replay the first
    /// token's arguments forever.  Refuse to record it: abandon the capture and
    /// run this op on the host instead of silently rotating with a stale
    /// `pos`/`base`.
    ///
    /// NOTE: on the Qwen3 engine this branch is currently unreachable -- its
    /// decode step never manages to open a capture (measured: no "decode graph
    /// live" line and no abandon message with or without this kernel).  It is
    /// kept because that is a property of `graphBegin` for that engine, not of
    /// this op, and it must not become a silent corruption if it changes.
    fn dRopeRows(g: backend.Graph, ctx: *anyopaque, x: backend.Buf, pos: u32, base: f32, rows: usize, heads: usize, hd: usize, rotary_dim: usize) void {
        const self = self_(ctx);
        const xg_prev = xg_push_slot(g.slot);
        defer xg_cur_slot = xg_prev;
        const t0 = self.slotEnter();
        defer self.slotExit(.rope64_rows, t0);
        if (self.ablated(.rope64_rows)) return;
        dRopeRows_body(g, ctx, x, pos, base, rows, heads, hd, rotary_dim);
    }

    fn dRopeRows_body(g: backend.Graph, ctx: *anyopaque, x: backend.Buf, pos: u32, base: f32, rows: usize, heads: usize, hd: usize, rotary_dim: usize) void {
        const self = self_(ctx);
        const row_len = heads * hd;
        if (rotary_dim == 64) {
            // Same width as `rope64Rows`: delegate so the 3.5/3.8 path keeps
            // exactly one implementation (and one set of graph-identity rules).
            return dRope64Rows_body(g, ctx, x, pos, base, rows, heads, hd);
        }
        if (!ropeOnDevice() or rows == 0 or heads == 0 or
            rotary_dim < 2 or rotary_dim > hd or
            row_len == 0 or x.len != rows * row_len or self.cu == null)
        {
            cpu_mod.CpuBackend.vtable.ropeRows(.{}, @ptrCast(&self.cpu), x, pos, base, rows, heads, hd, rotary_dim);
            return;
        }
        // A capture would replay this node with the FIRST token's arguments and
        // there is no `_dpos` variant to retarget, so refuse to record it: tear
        // the capture down and run the op with a DIRECT launch (never the CPU
        // fallback -- `x` is a device pointer here, and the CPU twin would
        // dereference it as host memory).
        if (self.capturing(g)) {
            self.graphAbandon(g, "ropeRows (variable rotary_dim) is not capture-safe");
        }
        if (dBufPtr(x)) |px| {
            const rc = Launch.rope_rows_dim(g, px, @intCast(rows), @intCast(heads), @intCast(hd), @bitCast(pos), base, @intCast(rotary_dim), self.launchStream());
            if (rc == cudart.cudaSuccess) return;
            self.ewLaunchFail("rope_rows_dim", rc);
        }
        cpu_mod.CpuBackend.vtable.ropeRows(.{}, @ptrCast(&self.cpu), x, pos, base, rows, heads, hd, rotary_dim);
    }


    fn dRope64Rows(g: backend.Graph, ctx: *anyopaque, x: backend.Buf, pos: u32, base: f32, rows_in: usize, heads_in: usize, hd_in: usize) void {
        const self = self_(ctx);
        const xg_prev = xg_push_slot(g.slot);
        defer xg_cur_slot = xg_prev;
        const t0 = self.slotEnter();
        defer self.slotExit(.rope64_rows, t0);
        if (self.ablated(.rope64_rows)) return;
        dRope64Rows_body(g, ctx, x, pos, base, rows_in, heads_in, hd_in);
    }

    /// The `convStep` slot, timed (`$QWEN35_PROF`).
    fn dConvStep(g: backend.Graph, ctx: *anyopaque, conv: backend.Buf, y: backend.Buf, wconv: backend.Buf, k: usize) void {
        const self = self_(ctx);
        const xg_prev = xg_push_slot(g.slot);
        defer xg_cur_slot = xg_prev;
        const t0 = self.slotEnter();
        defer self.slotExit(.conv_step, t0);
        if (self.ablated(.conv_step)) return ;
        dConvStep_body(g, ctx, conv, y, wconv, k);
    }

};

// ==== the Exec vtable ========================================================

pub const vtable = backend.Exec.VTable{
    // Device: weights and every numeric kernel.
    .loadLinear = CudaBackend.vLoadLinear,
    .freeLinear = CudaBackend.vFreeLinear,
    .dequantRow = CudaBackend.vDequantRow,
    .matvec = CudaBackend.vMatvec,
    .matvecPair = CudaBackend.vMatvecPair,
    .matvecBatchK = CudaBackend.vMatvecBatchK,
    .allocScratch = CudaBackend.dAllocScratch,
    .allocScratchK = CudaBackend.dAllocScratchK,
    .freeScratch = CudaBackend.dFreeScratch,
    // Host, non-computational only: thread-pool bookkeeping for the metadata loader
    // and the timing counters it prints.
    .setThreads = CudaBackend.dSetThreads,
    .profPhase = CudaBackend.dProfPhase,
    .profReset = CudaBackend.dProfReset,
    .profPrefillEnd = CudaBackend.dProfPrefillEnd,
    .graphBegin = CudaBackend.dGraphBegin,
    .graphEnd = CudaBackend.dGraphEnd,
    .graphDirect = CudaBackend.dGraphDirect,
    .setTier = CudaBackend.dSetTier,
    .reportProfile = CudaBackend.dReportProfile,
    .profReport = CudaBackend.dProfReport,
    .name = CudaBackend.vName,
    .describe = CudaBackend.vDescribe,
    // Elementwise / state slots: ALL on the device.
    //
    // P2b left seven of these on the CPU because every call was an H2D + D2H
    // round trip of a few KB, and for the small hot buffers that round trip cost
    // more than the CPU work it replaced (measured: binding all nine sank
    // Generate from 38.7 to 22.1 tok/s).  P2c removed the round trip -- both
    // sides are device memory now -- so the bindings below cost a launch and
    // nothing else, and the measurement inverts (see the report).
    //
    // rope64 stays on the CPU on purpose: it is the one slot whose device
    // implementation cannot match the CPU bit-for-bit (sin/cos of a different
    // libm), and a per-head 64-element call is not worth any divergence in the
    // token stream.  See `dRope64`'s comment.
    .rmsnorm = CudaBackend.dRmsnorm,
    .rmsnormRows = CudaBackend.dRmsnormRows,
    .rope64 = CudaBackend.dRope64,
    .rope64Rows = CudaBackend.dRope64Rows,
    .ropeRows = CudaBackend.dRopeRows,
    .l2norm = CudaBackend.dL2norm,
    .l2normRows = CudaBackend.dL2normRows,
    .siluInPlace = CudaBackend.dSiluInPlace,
    .mulSilu = CudaBackend.dMulSilu,
    .mulSigmoid = CudaBackend.dMulSigmoid,
    .softmax = CudaBackend.dSoftmax,
    .swiglu = CudaBackend.dSwiglu,
    .convStep = CudaBackend.dConvStep,
    .gdnStep = CudaBackend.dGdnStep,
    .gdnStepRows = CudaBackend.dGdnStepRows,
    .gdnStepRowsChunk = CudaBackend.dGdnStepRowsChunk,
    .gdnStepRowsSnap = CudaBackend.dGdnStepRowsSnap,
    .loadVecF32 = CudaBackend.vLoadVecF32,
    .freeVec = CudaBackend.vFreeVec,
    .attnDecode = CudaBackend.dAttnDecode,
    .attnChunk = CudaBackend.dAttnChunk,
    .gdnGamma = CudaBackend.dGdnGamma,
    .gdnPrefillCore = CudaBackend.dGdnPrefillCore,
    .copyFromHost = CudaBackend.dCopyFromHost,
    .argmax = CudaBackend.dArgmax,
    .argmaxConf = CudaBackend.dArgmaxConf,
    .argmaxRows = CudaBackend.dArgmaxRows,
    .sample = CudaBackend.dSample,
    .sampledToken = CudaBackend.dSampledToken,
    // The four pure elementwise slots.  Forwarders, NOT aliases: they need
    // `ctx` to reach the device, and the ctx here is a `*CudaBackend`
    // (aliasing a CPU body would work only because those bodies ignore ctx).
    // Each still falls back to the CPU path on a device-side failure.
    .scaleVec = CudaBackend.dScaleVec,
    .l2normQkScale = CudaBackend.dL2normQkScale,
    .addResid = CudaBackend.dAddResid,
    .copyRange = CudaBackend.dCopyRange,
    .copyOut = CudaBackend.dCopyToHost,
    .copyRangeSlot = CudaBackend.dCopyRangeSlot,
    .deinterleaveQG = CudaBackend.dDeinterleaveQG,
    .deinterleaveQGStrided = CudaBackend.dDeinterleaveQGStrided,
    .zeroBuf = CudaBackend.dZeroBuf,
    // Activations are PROPER device regions after P2c, so `allocBuf` is a real
    // `cudaMalloc` and `freeScratch` releases them (`dAllocBuf`/`dFreeScratch`).
    .allocBuf = CudaBackend.dAllocBuf,
    // KV cache: one format per load, device-resident.
    .setKvType = CudaBackend.vSetKvType,
    .allocKv = CudaBackend.vAllocKv,
    .kvAppend = CudaBackend.dKvAppend_body,

};
