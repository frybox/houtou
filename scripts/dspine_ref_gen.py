#!/usr/bin/env python3
"""Generate ground truth for the four DSPINE bridge kernels from the REFERENCE
Triton implementation (`sglang_dspine/kernels/dspine.py`), then write .npy
files that `scripts/dspine_c_check.c` compares the scalar C port against.

The C port is not allowed to drift even by one ulp: the reference runs every
kernel with `enable_fp_fusion=False` and casts to bf16 at eleven specific
points, so "numerically close" is not the contract -- bit equality is.

Run with the reference's own venv (it already has torch+triton on the 4090D):

    /mnt/llm-dev/dspine/venv/bin/python scripts/dspine_ref_gen.py /tmp/dspine_ref

Shapes come from the shipped b16 checkpoint: C=1024, R=512, H=5120, T=16.
Numeric inputs are the actual bf16 weight/activation distributions (normal
scaled), not uniform noise: the rms division and the sigmoid both have
ulp-level behaviour that depends on magnitude.
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, "/mnt/llm-dev/dspine")
from sglang_dspine.kernels import dspine as K  # noqa: E402

torch.manual_seed(0)

C, R, H, T = 1024, 512, 5120, 16
EPS = 1e-6
out_dir = sys.argv[1] if len(sys.argv) > 1 else "/tmp/dspine_ref"
os.makedirs(out_dir, exist_ok=True)
dev = "cuda"


def save(name, t):
    """Store as f32.  A bf16 value is exactly representable in f32, so this
    loses nothing and keeps the C harness free of a bf16 reader.

    Integer inputs (anchors, valid masks) stay integral: the C port indexes a
    table with them, and f32 would round large ids.
    """
    t = t.detach().cpu()
    arr = t.numpy() if t.dtype in (torch.int32, torch.int8, torch.bool) else t.float().numpy()
    np.save(os.path.join(out_dir, name + ".npy"), arr)


def bf16(x):
    return x.to(torch.bfloat16)


def randn_bf16(*shape, scale=1.0):
    return bf16(torch.randn(*shape, device=dev) * scale)


# --------------------------------------------------------------------------
# prepare_read  (kernels/dspine.py:11)
#   raw    = bf16(x + residual) if ADD else bf16(x)
#   rms    = sqrt(max(mean(raw^2), FLT_MIN_NORMAL))
#   normed = raw / (rms + EPS)
# Rows: one full block plus a couple, and both the ADD and no-ADD arms.
# --------------------------------------------------------------------------
for add in (0, 1):
    m = T + 3
    x = randn_bf16(m, H, scale=1.7)
    res = randn_bf16(m, H, scale=0.9) if add else None
    raw = torch.empty_like(x)
    normed = torch.empty_like(x)
    rms = torch.empty(m, device=dev, dtype=torch.float32)
    K.prepare_read[m,](
        x,
        x if res is None else res,
        raw,
        normed,
        rms,
        H,
        EPS,
        bool(add),
        torch.tensor(0).item() or 8192,
        enable_fp_fusion=False,
    )
    tag = f"add{add}"
    save(f"prepare_read_{tag}_x", x)
    if res is not None:
        save(f"prepare_read_{tag}_res", res)
    save(f"prepare_read_{tag}_raw", raw)
    save(f"prepare_read_{tag}_normed", normed)
    save(f"prepare_read_{tag}_rms", rms)
    print(f"prepare_read add={add}: rms mean {rms.mean().item():.6f}")

# --------------------------------------------------------------------------
# normalize_codes  (kernels/dspine.py:41)
#   anchor rows (row % T == 0) take the codebook entry * sqrt(C), NOT renormed;
#   every other row is sqrt(C) * z / (||z|| + EPS).
# --------------------------------------------------------------------------
m = T + 2
projected = torch.randn(m, C, device=dev, dtype=torch.float32)
codebook = randn_bf16(4096, C, scale=0.5)
anchors = torch.tensor([7, 1234, 3, 99], device=dev, dtype=torch.int32)
codes = torch.empty(m, C, device=dev, dtype=torch.bfloat16)
K.normalize_codes[m,](
    projected,
    codebook,
    anchors,
    codes,
    C,
    T,
    True,
    EPS,
    torch.tensor(0).item() or 1024,
    enable_fp_fusion=False,
)
save("normalize_codes_projected", projected)
save("normalize_codes_codebook", codebook)
save("normalize_codes_anchors", anchors)
save("normalize_codes_codes", codes)
print("normalize_codes: ok")

# --------------------------------------------------------------------------
# feature_message  (kernels/dspine.py:63)
#   query  = packed[row,  :R]
#   sender = packed[row-1, R:2R]      prev = max(row-1, 0)
#   value  = packed[row-1, 2R:3R]
#   msg    = 0 on anchor rows, else bf16(sigmoid(query+sender+bias) * value)
# --------------------------------------------------------------------------
m = T + 4
packed = bf16(torch.randn(m, 3 * R, device=dev) * 1.3).float()
bias = torch.randn(R, device=dev, dtype=torch.float32)
message = torch.empty(m, R, device=dev, dtype=torch.bfloat16)
K.feature_message[m,](
    packed,
    bias,
    message,
    R,
    T,
    torch.tensor(0).item() or 512,
    enable_fp_fusion=False,
)
save("feature_message_packed", packed)
save("feature_message_bias", bias)
save("feature_message_message", message)
print("feature_message: ok")

# --------------------------------------------------------------------------
# feature_finish  (kernels/dspine.py:85)
#   edge   = (row % T != 0) & valid[row] & valid[row-1]
#   out    = bf16(raw + rms*delta*edge)
#   normed = bf16(bf16(out * rsqrt(mean(out^2)+eps)) * weight)
# The NORM branch has THREE consecutive bf16 roundings; collapsing them changes
# ~26% of elements (measured), so this is the tightest of the four.
# --------------------------------------------------------------------------
m = T + 2
raw = randn_bf16(m, H, scale=1.1)
delta = torch.randn(m, H, device=dev, dtype=torch.float32)
rms = torch.rand(m, device=dev, dtype=torch.float32) * 2.0 + 0.01
weight = torch.randn(H, device=dev, dtype=torch.float32)
valid = torch.ones(m, device=dev, dtype=torch.bool)
valid[5] = False  # exercise the valid[] gate on a non-anchor row
output = torch.empty_like(raw)
normed = torch.empty_like(raw)
K.feature_finish[m,](
    raw,
    delta,
    rms,
    valid,
    weight,
    output,
    normed,
    H,
    T,
    True,
    True,
    EPS,
    torch.tensor(0).item() or 8192,
    enable_fp_fusion=False,
)
save("feature_finish_raw", raw)
save("feature_finish_delta", delta)
save("feature_finish_rms", rms)
save("feature_finish_weight", weight)
save("feature_finish_valid", valid.to(torch.int8))
save("feature_finish_output", output)
save("feature_finish_normed", normed)
print("feature_finish: ok")

print(f"\nwrote ground truth to {out_dir}")
