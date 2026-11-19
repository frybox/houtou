#!/usr/bin/env python3
"""Run the REFERENCE Triton `serial_table` on the SAME inputs our Zig kernel saw.

Magnitude probes could not separate "my table/scoring is wrong" from "this
checkpoint is weak".  Feeding the reference MY stash, cand and tables is the
only discriminating experiment: if the reference picks the same lane, the
transcription is faithful and the low acceptance is a property of the draft; if
it picks another lane, the defect is in my transcription.

The id space is compacted to the ids that actually appear in `cand` (plus the
anchors), so `lm_head_w` is a few hundred rows instead of 248320 -- the scoring
only ever touches `LmW[ids]`, so this is an exact reduction, not an
approximation.  `TG`/`TSV` are permuted the same way.

    /mnt/llm-dev/dspine/venv/bin/python scripts/s4_ref_compare.py /tmp/s4dump
"""
import sys

import numpy as np
import torch

sys.path.insert(0, "/mnt/llm-dev/dspine")
from sglang_dspine.kernels import dspine as K  # noqa: E402

dev = "cuda"
d = sys.argv[1] if len(sys.argv) > 1 else "/tmp/s4dump"

meta = {}
with open(f"{d}/meta.txt") as f:
    for line in f:
        k, v = line.strip().split("=")
        meta[k] = int(v)
T, K_, H, R, C, rows, n_rows, V = (
    meta["T"], meta["K"], meta["H"], meta["R"], meta["C"],
    meta["rows"], meta["n_rows"], meta["V"],
)
print(f"T={T} K={K_} H={H} R={R} C={C} rows={rows} n_rows={n_rows} V={V}")


def f32(name, shape=None):
    t = torch.from_numpy(np.fromfile(f"{d}/{name}", dtype=np.float32).copy()).to(dev)
    return t.reshape(shape) if shape else t


def u16_as_bf16(name, shape=None):
    raw = np.fromfile(f"{d}/{name}", dtype=np.uint16)
    as_f32 = (raw.astype(np.uint32) << 16).view(np.float32)
    t = torch.from_numpy(as_f32.copy()).to(dev).to(torch.bfloat16)
    return t.reshape(shape) if shape else t


raw = f32("raw.f32", (rows, H)).to(torch.bfloat16)
rms = f32("rms.f32")
query = f32("query.f32", (rows, R)).to(torch.bfloat16)
gate_b = f32("gate_b.f32")
cand = torch.from_numpy(np.fromfile(f"{d}/cand.i32", dtype=np.int32).copy()).to(dev).long().reshape(n_rows, K_)
tg = u16_as_bf16("tg.u16", (V, R))
tsv = u16_as_bf16("tsv.u16", (V, 2 * R))
anchors = torch.from_numpy(np.fromfile(f"{d}/anchors.i32", dtype=np.int32).copy()).to(dev).long()
lm_rows = f32("lm_head_rows.f32", (anchors.numel() + 0, H)) if False else None
lm_ids = torch.from_numpy(np.fromfile(f"{d}/lm_head_ids.i32", dtype=np.int32).copy()).to(dev).long()
lm_vals = f32("lm_head_rows.f32").reshape(lm_ids.numel(), H).to(torch.bfloat16)

print("\n--- input magnitudes ---")
rf = raw.float()
print(f"raw    max={rf.abs().max().item():.4e}  rms={rf.pow(2).mean().sqrt().item():.4e}")
print(f"rms    min={rms.min().item():.4e} max={rms.max().item():.4e}")
print(f"query  max={query.float().abs().max().item():.4e}")
print(f"tg     max={tg.float().abs().max().item():.4e}")
print(f"tsv    max={tsv.float().abs().max().item():.4e}")
print(f"anchors {anchors.tolist()}")

# ---- compact the id space to {anchors} U {cand} U {lm_head_ids} ----
ids = sorted(set(anchors.tolist()) | set(cand.flatten().tolist()) | set(lm_ids.tolist()))
id2c = {v: i for i, v in enumerate(ids)}
VC = len(ids)
print(f"compacted id space: {VC} ids")

cand_c = torch.tensor([[id2c[v] for v in row] for row in cand.tolist()], dtype=torch.long, device=dev)
anchors_c = torch.tensor([id2c[v] for v in anchors.tolist()], dtype=torch.long, device=dev)

# lm_head for the compacted space: rows we dumped, zero elsewhere.  Rows that are
# never scored cannot affect `argmax`, so the zeros are inert.
lm = torch.zeros((VC, H), dtype=torch.bfloat16, device=dev)
for pos, gid in enumerate(lm_ids.tolist()):
    lm[id2c[gid]] = lm_vals[pos]

tg_c = torch.zeros((VC, R), dtype=torch.bfloat16, device=dev)
tsv_c = torch.zeros((VC, 2 * R), dtype=torch.bfloat16, device=dev)
for gid in ids:
    tg_c[id2c[gid]] = tg[gid]
    tsv_c[id2c[gid]] = tsv[gid]

input_ids = torch.zeros(rows, dtype=torch.long, device=dev)
for b in range(rows // T):
    input_ids[b * T] = anchors_c[b]

packed = torch.zeros((rows, 3 * R), dtype=torch.bfloat16, device=dev)
packed[:, :R] = query

print("\n--- reference Triton serial_table on OUR stash ---")
ref_idx = K.serial_table(
    raw, f32("norm_w.f32"), cand_c, input_ids, lm, tg_c, tsv_c, packed, rms, gate_b, T,
).cpu()

ours = np.fromfile(f"{d}/idx.i16", dtype=np.int16).reshape(n_rows, K_)
print("ref  idx[0]:", ref_idx[0].tolist())
print("ours idx[0]:", ours[0].tolist())
same = int((ref_idx.numpy() == ours).sum())
print(f"idx agreement: {same}/{n_rows * K_}")

np.save(f"{d}/ref_idx.npy", ref_idx.numpy())
print(f"\nwrote {d}/ref_idx.npy")

# ---- longhand score for row 0, both sides, to see WHICH term diverges ----
print("\n--- row 0 score, longhand over the compacted space ---")
b = 0
row = b * T + 1
x = raw[row].float()
nw = f32("norm_w.f32")
u = x * nw

base = torch.stack([(lm[cand_c[0, j]].float() * u).sum() for j in range(K_)])
cand_row = cand_c[0]
anchor_id = anchors_c[0]

sid = anchor_id                      # r == 0 -> sender is the anchor
g_send = tsv_c[sid, :R].float()
v_send = tsv_c[sid, R:].float()
q = query[row].float()
gb = gate_b
m = torch.sigmoid(q + g_send + gb) * v_send

scores = torch.zeros(K_, K_)
for kc in range(K_):
    tg_k = tg_c[cand_row[kc]].float()
    acc = (m * tg_k).sum()
    scores[:, kc] = base[kc] + rms[row] * acc
print("base   :", [f"{v:.4e}" for v in base.tolist()[:6]])
print("score  :", [f"{v:.4e}" for v in scores[0].tolist()[:6]])
print("longhand argmax:", int(scores[0].argmax()), " ref:", int(ref_idx[0, 0]), " ours:", int(ours[0, 0]))

print("\n--- component rms, row 0, lane 0 ---")
nw = f32("norm_w.f32")
u = raw[row].float() * nw
lm0 = lm[cand_c[0, 0]].float()
print(f"norm_w rms={nw.pow(2).mean().sqrt().item():.4e}")
print(f"u      rms={u.pow(2).mean().sqrt().item():.4e}")
print(f"lm0    rms={lm0.pow(2).mean().sqrt().item():.4e}")
print(f"base0  ={(lm0*u).sum().item():.4e}")

print("\n--- dump integrity ---")
print("lm_head_ids:", lm_ids.tolist()[:8], "... n=", lm_ids.numel())
print("lm_vals rows:", lm_vals.shape)
rms_per_row = lm_vals.float().pow(2).mean(dim=1).sqrt()
print("first 8 row rms:", [f"{v:.4e}" for v in rms_per_row[:8].tolist()])
lane0_id = int(cand[0, 0])
pos = int((lm_ids == lane0_id).nonzero()[0])
print(f"lane0 id={lane0_id} pos_in_dump={pos} rms={rms_per_row[pos].item():.4e} -> lm[id2c] rms={lm[id2c[lane0_id]].float().pow(2).mean().sqrt().item():.4e}")

# ---- chain walk on BOTH idx tables ----
def walk(idx_t, cand_t, t, k):
    n_rows = idx_t.shape[0]
    out = []
    j = int(idx_t[0, 0])
    for r in range(min(n_rows, t - 1)):
        out.append(int(cand_t[r, j]))
        if r + 1 < n_rows:
            j = int(idx_t[r + 1, j])
    return out

print("\n--- chain walk ---")
print("  ours:", walk(ours.astype(np.int64), cand.cpu().numpy(), T, K_))
print("  ref :", walk(ref_idx.numpy(), cand.cpu().numpy(), T, K_))
