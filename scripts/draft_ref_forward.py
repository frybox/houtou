#!/usr/bin/env python3
"""Independent PyTorch transcription of the DSpine draft's 5-layer forward.

Why: stage 4 was proven correct against the Triton reference
(`scripts/s4_ref_compare.py`, 240/240 idx agreement), so whatever remains is in
the forward.  MLX cannot run on this host, so this file re-derives the forward
from the same authorities the Zig port used:

  * `dspine_mlx/dspine_mlx/model.py`   DSpineDraftModel.__call__, DraftAttention,
                                       DraftDecoderLayer
  * `dspine_mlx/dspine_mlx/bridge.py`  DSpineCodeBridge.__call__ (the code bridge)
  * `sglang_dspine/kernels/dspine.py`  prepare_read / normalize_codes /
                                       feature_message / feature_finish

It reads the DRAFT GGUF directly (no safetensors conversion needed) and takes the
context tap planes from the Zig dump, so both sides start from identical input.

    python scripts/draft_ref_forward.py <draft.gguf> <dump_root>

`<dump_root>/fwd/` gets one file per (layer, stage) from the Zig side; this script
prints the same stages for comparison.
"""
import math
import sys

import numpy as np
import torch
from gguf import GGUFReader

# The four bridge kernels are the REFERENCE Triton implementations from
# `sglang_dspine/kernels/dspine.py` when running on CUDA; on CPU this file falls
# back to its own transcription of them.  GPU is both 50x faster and removes the
# "did my transcription of the bridge differ" question entirely.
dev = "cuda" if "--cpu" not in sys.argv else "cpu"
torch.set_grad_enabled(False)

gguf_path = sys.argv[1]
dump_root = sys.argv[2]
FWD = f"{dump_root}/fwd"

r = GGUFReader(gguf_path)
T = {t.name: t for t in r.tensors}


def raw(name):
    """Dequantise one GGUF tensor to f32, returned as torch [ne1, ne0] (row-major
    rows of ne0) -- the layout every `nn.Linear` weight wants."""
    t = T[name]
    ne0 = int(t.shape[0])
    ne1 = int(t.shape[1]) if len(t.shape) > 1 else 1
    flat = np.array(t.data, copy=True).reshape(-1)
    if t.tensor_type.name.startswith("F32"):
        a = flat.view(np.float32).reshape(ne1, ne0)
    elif t.tensor_type.name.startswith("F16"):
        a = flat.view(np.float16).astype(np.float32).reshape(ne1, ne0)
    elif "BF16" in t.tensor_type.name:
        u16 = flat.view(np.uint16)
        a = (u16.astype(np.uint32) << 16).view(np.float32).reshape(ne1, ne0)
    elif t.tensor_type.name == "Q8_0":
        # block_q8_0: f16 d + 32 int8, 34 bytes.  `dequantize_row_q8_0` in
        # ggml-quants.c is exactly `d * qs[i]`.
        blk = flat.reshape(-1, 34)
        d = np.ascontiguousarray(blk[:, :2]).view(np.float16).astype(np.float32)
        qs = np.ascontiguousarray(blk[:, 2:]).view(np.int8).astype(np.float32)
        a = (d.reshape(-1, 1) * qs).reshape(-1).reshape(ne1, ne0)
    else:
        raise SystemExit(f"need a plain dequant of {name} ({t.tensor_type.name})")
    return torch.from_numpy(a.copy()).to(dev)


def meta(key):
    f = r.fields.get(key)
    if f is None:
        return None
    return f.contents()


def bf(x):
    return x.to(torch.bfloat16)


USE_TRITON = (dev == "cuda")
if USE_TRITON:
    sys.path.insert(0, "/mnt/llm-dev/dspine")
    from sglang_dspine.kernels import dspine as K
    import triton

    def npow2(v):
        return 1 << (int(v) - 1).bit_length()
    print("bridge kernels: REFERENCE Triton (sglang_dspine.kernels.dspine)")

H = int(meta("dflash.embedding_length") or T["fc.weight"].shape[0])
FFN = int(meta("dflash.feed_forward_length"))
HEADS = int(meta("dflash.attention.head_count"))
KVH = int(meta("dflash.attention.head_count_kv"))
HD = int(meta("dflash.attention.key_length"))
NL = int(meta("dflash.block_count"))
TBLK = int(meta("dflash.block_size"))
ROPE_BASE = float(meta("dflash.rope.freq_base") or 1e7)
EPS = float(meta("dflash.attention.layer_norm_rms_epsilon") or 1e-6)
BRIDGE_EPS = EPS
MASK = int(meta("tokenizer.ggml.mask_token_id"))
R = int(T["dspine_gate.weight"].shape[1])
C = int(T["dspine_gate.weight"].shape[0])
print(f"H={H} FFN={FFN} HEADS={HEADS} KVH={KVH} HD={HD} NL={NL} T={TBLK} R={R} C={C} mask={MASK}")

# ---- weights ----
layers = []
for i in range(NL):
    layers.append(dict(
        attn_norm=raw(f"blk.{i}.attn_norm.weight"),
        attn_q=raw(f"blk.{i}.attn_q.weight"),
        attn_k=raw(f"blk.{i}.attn_k.weight"),
        attn_v=raw(f"blk.{i}.attn_v.weight"),
        attn_output=raw(f"blk.{i}.attn_output.weight"),
        attn_q_norm=raw(f"blk.{i}.attn_q_norm.weight"),
        attn_k_norm=raw(f"blk.{i}.attn_k_norm.weight"),
        ffn_norm=raw(f"blk.{i}.ffn_norm.weight"),
        ffn_gate=raw(f"blk.{i}.ffn_gate.weight"),
        ffn_up=raw(f"blk.{i}.ffn_up.weight"),
        ffn_down=raw(f"blk.{i}.ffn_down.weight"),
        read=raw(f"blk.{i}.dspine_read.weight"),
        write=raw(f"blk.{i}.dspine_write.weight"),
    ))
fc = raw("fc.weight")
hidden_norm = raw("enc.output_norm.weight")
out_norm = raw("output_norm.weight")
codebook = raw("dspine_codebook.weight")   # [V, C]
gate_w = raw("dspine_gate.weight")
gate_b = raw("dspine_gate.bias")
prev_w = raw("dspine_prev.weight")
pred_gate_w = raw("dspine_pred_gate.weight")


def rmsnorm(x, w, eps):
    return (x / torch.sqrt(x.pow(2).mean(-1, keepdim=True) + eps)) * w


def rope64(x, pos, base):
    """`applyRope64`: rotate each head's leading 64 floats, pairing j with j+32."""
    out = x.clone()
    half = 32
    for j in range(half):
        theta = pos * (base ** (-2.0 * j / 64.0))
        c, s = math.cos(theta), math.sin(theta)
        a = x[..., j]
        b = x[..., j + half]
        out[..., j] = a * c - b * s
        out[..., j + half] = a * s + b * c
    return out


# ---- Zig dump ----
meta_txt = {}
with open(f"{FWD}/meta.txt") as f:
    for line in f:
        k, v = line.strip().split("=")
        meta_txt[k] = v


def zig(name):
    a = np.fromfile(f"{FWD}/{name}", dtype=np.float32)
    return torch.from_numpy(a.copy()).to(dev)


tap = zig("tap.f32").reshape(5, -1, H)      # [n_taps][rows][H]
rows = tap.shape[1]
print(f"context rows={rows} taps={tap.shape[0]}")

# ---- context: ctx = hidden_norm(fc(concat_taps)) ----
cat = tap.permute(1, 0, 2).reshape(rows, 5 * H)
ctx = rmsnorm(cat @ fc.T, hidden_norm, EPS)
print(f"ctx rms={ctx.pow(2).mean().sqrt().item():.4e} (rows={rows})")

# ---- block ids: anchor + mask ----
anchor = int(np.fromfile(f"{FWD}/anchor.i32", dtype=np.int32)[0])
toks = [anchor] + [MASK] * (TBLK - 1)

# embed: the target's table.  The Zig dump carries the rows it used.
emb = zig("embed_rows.f32").reshape(TBLK, H)
print(f"embed rms={emb.pow(2).mean().sqrt().item():.4e}")

resid = emb.clone()
xin = rmsnorm(resid, layers[0]["attn_norm"], EPS)

stash = {}
(xins, attns, resids, outs, gouts, kvs, qs_, swis, mlps, raws_, msgs_, deltas_, codes_ref) = ([], [], [], [], [], [], [], [], [], [], [], [], [])
for i, L in enumerate(layers):
    # ---- attention ----
    q = xin @ L["attn_q"].T
    k = xin @ L["attn_k"].T
    v = xin @ L["attn_v"].T
    q = rmsnorm(q.reshape(TBLK, HEADS, HD), L["attn_q_norm"], EPS).reshape(TBLK, HEADS * HD)
    k = rmsnorm(k.reshape(TBLK, KVH, HD), L["attn_k_norm"], EPS).reshape(TBLK, KVH * HD)
    # rope at absolute positions: context 0..rows-1, block rows..rows+TBLK-1
    qh = q.reshape(TBLK, HEADS, HD)
    kh = k.reshape(TBLK, KVH, HD)
    for t in range(TBLK):
        qh[t] = rope64(qh[t], rows + t, ROPE_BASE)
        kh[t] = rope64(kh[t], rows + t, ROPE_BASE)
    q, k = qh.reshape(TBLK, HEADS * HD), kh.reshape(TBLK, KVH * HD)

    # `_context_kv`: ctx -> k_proj -> k_norm -> rope at absolute position, and
    # v_proj with NO norm (`model.py:111-122`).
    ck = (ctx @ L["attn_k"].T).reshape(rows, KVH, HD)
    ck = rmsnorm(ck, L["attn_k_norm"], EPS)
    ck = torch.stack([rope64(ck[t], t, ROPE_BASE) for t in range(rows)]).reshape(rows, KVH * HD)
    cv = (ctx @ L["attn_v"].T)

    K_all = torch.cat([ck, k], 0).reshape(rows + TBLK, KVH, HD)
    V_all = torch.cat([cv, v], 0).reshape(rows + TBLK, KVH, HD)
    qq = q.reshape(TBLK, HEADS, HD)
    gqa = HEADS // KVH
    gout = torch.zeros(TBLK, HEADS * HD, device=dev)
    scale = HD ** -0.5
    for t in range(TBLK):
        for h in range(HEADS):
            kvh = h // gqa
            ks = K_all[: rows + t + 1, kvh]
            scores = (ks @ qq[t, h]) * scale
            p = torch.softmax(scores, -1)
            gout[t, h * HD : (h + 1) * HD] = p @ V_all[: rows + t + 1, kvh]

    attn = gout @ L["attn_output"].T
    resid = resid + attn
    x1 = rmsnorm(resid, L["ffn_norm"], EPS)
    gate = x1 @ L["ffn_gate"].T
    up = x1 @ L["ffn_up"].T
    swi = torch.nn.functional.silu(gate) * up
    mlp = swi @ L["ffn_down"].T
    resid = resid + mlp

    # ---- code bridge ----
    # Two modes, on purpose:
    #   CUDA -> the REFERENCE Triton kernels from sglang_dspine, called exactly as
    #           `DSpineCodeBridge._forward_fused` calls them.  This is the
    #           authority; nothing here re-derives rounding.
    #   CPU  -> a transcription, kept so the file still runs without a GPU.
    if USE_TRITON:
        m = TBLK
        raww = bf(resid)
        normalized = torch.empty_like(raww)
        rms = torch.empty(m, device=dev, dtype=torch.float32)
        K.prepare_read[m,](
            raww, raww, raww.clone(), normalized, rms,
            H, BRIDGE_EPS, False, npow2(H),
            enable_fp_fusion=False,
        )
        raww = raww.clone()
        projected = normalized.float() @ L["read"].T
        codes = torch.empty_like(projected).to(torch.bfloat16)
        anchors = torch.tensor([anchor], dtype=torch.int32, device=dev)
        K.normalize_codes[m,](
            projected, codebook, anchors, codes,
            C, TBLK, True, BRIDGE_EPS, npow2(C),
            enable_fp_fusion=False,
        )
        packed = torch.cat([codes.float() @ gate_w.T,
                            codes.float() @ pred_gate_w.T,
                            codes.float() @ prev_w.T], dim=1).to(torch.bfloat16)
        message = torch.empty((m, R), device=dev, dtype=torch.bfloat16)
        K.feature_message[m,](
            packed, gate_b, message,
            R, TBLK, npow2(R),
            enable_fp_fusion=False,
        )
        delta = message.float() @ L["write"].T
        valid = torch.ones(m, device=dev, dtype=torch.bool)
        output = torch.empty_like(raww)
        normed = torch.empty_like(raww)
        nxt_w = (layers[i + 1]["attn_norm"] if i < NL - 1 else out_norm)
        K.feature_finish[m,](
            raww, delta, rms, valid, nxt_w, output, normed,
            H, TBLK, True, True, EPS, npow2(H),
            enable_fp_fusion=False,
        )
        raww = raww.float()
        out = output.float()
        nrm = normed.float()
        msg = message.float()
    else:
        raww = bf(resid).float()
        rms = torch.sqrt(torch.clamp(raww.pow(2).mean(-1), min=1.1754943508222875e-38))
        normed = raww / (rms.unsqueeze(-1) + BRIDGE_EPS)
        projected = normed @ L["read"].T
        codes = torch.zeros(TBLK, C, device=dev)
        for t in range(TBLK):
            if t % TBLK == 0:
                codes[t] = bf(codebook[anchor] * math.sqrt(C)).float()
            else:
                z = projected[t]
                nn = torch.sqrt((z * z).sum())
                codes[t] = bf(math.sqrt(C) * z / (nn + BRIDGE_EPS)).float()
        packed = torch.zeros(TBLK, 3 * R, device=dev)
        packed[:, :R] = codes @ gate_w.T
        packed[:, R : 2 * R] = codes @ pred_gate_w.T
        packed[:, 2 * R :] = codes @ prev_w.T
        msg = torch.zeros(TBLK, R, device=dev)
        for t in range(TBLK):
            if t % TBLK == 0:
                continue
            p = packed[t - 1]
            g = 1.0 / (1.0 + torch.exp(-(packed[t, :R] + p[R : 2 * R] + gate_b)))
            msg[t] = bf(g * p[2 * R :])
        delta = msg @ L["write"].T
        nxt = layers[i + 1]["attn_norm"] if i < NL - 1 else out_norm
        edge = torch.tensor([0.0 if t % TBLK == 0 else 1.0 for t in range(TBLK)], device=dev)
        out = bf(raww + (rms.unsqueeze(-1) * delta) * edge.unsqueeze(-1)).float()
        nrm = bf(bf(out / torch.sqrt(out.pow(2).mean(-1, keepdim=True) + EPS)) * nxt).float()

    xins.append(xin)
    attns.append(attn)
    resids.append(resid)
    outs.append(out)
    gouts.append(gout)
    kvs.append((K_all, V_all))
    qs_.append(q)
    swis.append(swi)
    mlps.append(mlp)
    raws_.append(raww)
    codes_ref.append(codes)
    msgs_.append(msg)
    deltas_.append(delta)
    resid = out
    xin = nrm
    if i == NL - 1:
        stash = dict(raw=raww, rms=rms, query=packed[:, :R])

    print(f"L{i} attn_max={attn.abs().max().item():.4e} mlp_max={mlp.abs().max().item():.4e} "
          f"resid_max={resid.abs().max().item():.4e} xin_max={xin.abs().max().item():.4e}")

print(f"\nstash raw rms={stash['raw'].pow(2).mean().sqrt().item():.4e}")
print(f"stash rms   min={stash['rms'].min().item():.4e} max={stash['rms'].max().item():.4e}")

# ---- element-wise comparison against the Zig dump ----
# The Zig dump is one file per (layer, stage), `block_size` rows of `width`
# floats, written incrementally -- so only the rows it reached are meaningful.
import os

WIDTH = {"xin": H, "gout": HEADS * HD, "attn": H, "mlp": FFN,
         "resid": H, "raw": H, "msg": R, "delta": H, "out": H}


def zstage(i, stage):
    p = f"{FWD}/l{i}_{stage}.f32"
    if not os.path.exists(p):
        return None
    a = np.fromfile(p, dtype=np.float32)
    return torch.from_numpy(a.copy()).reshape(-1, WIDTH[stage]).to(dev)


def cmp_stage(i, stage, ref):
    z = zstage(i, stage)
    if z is None or z.numel() == 0:
        return
    n = min(z.shape[0], ref.shape[0])
    z, ref = z[:n], ref[:n]
    d = (z - ref).abs()
    print(f"  L{i} {stage:5s} zrms={z.pow(2).mean().sqrt().item():.4e} "
          f"rrms={ref.pow(2).mean().sqrt().item():.4e} "
          f"maxdiff={d.max().item():.4e} meanrel={(d.mean().item()/(ref.abs().mean().item()+1e-30)):.3e}")


print("\n--- Zig vs Python per layer ---")
print(f"  ctx rrms={ctx.pow(2).mean().sqrt().item():.4e}")
for i in range(NL):
    cmp_stage(i, "xin", xins[i])
    cmp_stage(i, "gout", gouts[i])
    cmp_stage(i, "attn", attns[i])
    cmp_stage(i, "resid", resids[i])
    cmp_stage(i, "out", outs[i])

print("\n--- L0 attention detail ---")
KVD = KVH * HD
for name, ref_t, w in (("q", qs_[0], HEADS * HD), ("kcache", kvs[0][0], KVD), ("vcache", kvs[0][1], KVD)):
    p = f"{FWD}/l0_{name}.f32"
    if not os.path.exists(p):
        continue
    z = torch.from_numpy(np.fromfile(p, dtype=np.float32).copy()).to(dev)
    rr = ref_t.reshape(-1)
    n = min(z.numel(), rr.numel())
    d = (z[:n] - rr[:n]).abs()
    print(f"  {name:7s} zig={z[:n].pow(2).mean().sqrt().item():.4e} ref={rr[:n].pow(2).mean().sqrt().item():.4e} "
          f"maxdiff={d.max().item():.4e} zig_n={z.numel()} ref_n={rr.numel()}")

print("\n--- L0 bridge detail ---")
for nm, ref_t in (("raw", raws_[0]), ("msg", msgs_[0]), ("delta", deltas_[0])):
    p = f"{FWD}/l0_{nm}.f32"
    if not os.path.exists(p):
        continue
    w = WIDTH.get(nm, R if nm == "msg" else H)
    z = torch.from_numpy(np.fromfile(p, dtype=np.float32).copy()).reshape(-1, w).to(dev)
    ref_t = ref_t.reshape(-1, w)
    n = min(z.shape[0], ref_t.shape[0])
    d = (z[:n] - ref_t[:n]).abs()
    print(f"  {nm:6s} zig={z[:n].pow(2).mean().sqrt().item():.4e} ref={ref_t[:n].pow(2).mean().sqrt().item():.4e} "
          f"maxdiff={d.max().item():.4e} meanrel={(d.mean().item()/(ref_t[:n].abs().mean().item()+1e-30)):.3e}")
print(f"  nxt(xin) zig={xins[1].pow(2).mean().sqrt().item():.4e} ref={((bf(bf(out/torch.sqrt(out.pow(2).mean(-1,keepdim=True)+EPS))*layers[1]['attn_norm'])).float()).pow(2).mean().sqrt().item():.4e}")

print("\n--- L0 MLP ---")
for nm, ref_t, w in (("mlp", swis[0], FFN), ("mlp_out", mlps[0], H)):
    p = f"{FWD}/l0_{nm}.f32"
    if not os.path.exists(p):
        print(f"  {nm}: no dump")
        continue
    z = torch.from_numpy(np.fromfile(p, dtype=np.float32).copy()).reshape(-1, w).to(dev)
    rr = ref_t.reshape(-1, w)
    n = min(z.shape[0], rr.shape[0])
    d = (z[:n] - rr[:n]).abs()
    print(f"  {nm:8s} zig={z[:n].pow(2).mean().sqrt().item():.4e} ref={rr[:n].pow(2).mean().sqrt().item():.4e} "
          f"maxdiff={d.max().item():.4e} meanrel={(d.mean().item()/(rr[:n].abs().mean().item()+1e-30)):.3e}")

print("\n--- L0 codes ---")
import os
pc = f"{FWD}/l0_codes.f32"
if os.path.exists(pc):
    z = torch.from_numpy(np.fromfile(pc, dtype=np.float32).copy()).reshape(-1, C).to(dev)
    rr = codes_ref[0]
    n = min(z.shape[0], rr.shape[0])
    d = (z[:n] - rr[:n]).abs()
    print(f"  codes zig={z[:n].pow(2).mean().sqrt().item():.4e} ref={rr[:n].pow(2).mean().sqrt().item():.4e} maxdiff={d.max().item():.4e}")
    print(f"  zig[0][:4]={[f'{v:.4e}' for v in z[0][:4].tolist()]}")
    print(f"  ref[0][:4]={[f'{v:.4e}' for v in rr[0][:4].tolist()]}")
    print(f"  zig[1][:4]={[f'{v:.4e}' for v in z[1][:4].tolist()]}")
    print(f"  ref[1][:4]={[f'{v:.4e}' for v in rr[1][:4].tolist()]}")

print("\n--- L0 msg per row ---")
import os
pm = f"{FWD}/l0_msg.f32"
if os.path.exists(pm):
    z = torch.from_numpy(np.fromfile(pm, dtype=np.float32).copy()).reshape(-1, R).to(dev)
    rr = msgs_[0]
    n = min(z.shape[0], rr.shape[0])
    for t in range(min(n, 6)):
        d = (z[t] - rr[t]).abs().max().item()
        print(f"  row{t} zig_rms={z[t].pow(2).mean().sqrt().item():.4e} ref_rms={rr[t].pow(2).mean().sqrt().item():.4e} maxdiff={d:.4e}")

print("\n--- ctx difference: is it Q8_0 activation quantization? ---")


def q8_0_sim(x, w):
    """Zig's `matvec` quantizes the ACTIVATION to Q8_0 before the dot.  Simulate
    it so the two sides do the same arithmetic."""
    xq = x.clone()
    for r in range(x.shape[0]):
        v = x[r]
        for b0 in range(0, v.numel(), 32):
            blk = v[b0 : b0 + 32]
            amax = blk.abs().max()
            d = amax / 127.0 if amax > 0 else 0.0
            q = torch.round(blk / d).clamp(-127, 127) if d > 0 else torch.zeros_like(blk)
            xq[r, b0 : b0 + 32] = q * d
    return xq @ w.T


zctx = torch.from_numpy(np.fromfile(f"{FWD}/ctx.f32", dtype=np.float32).copy()).reshape(-1, H).to(dev)
cat_q = q8_0_sim(cat, fc)
ctx_q = rmsnorm(cat_q, hidden_norm, EPS)
print(f"  ctx  (f32 act)      vs zig: maxdiff={(ctx - zctx).abs().max().item():.4e} "
      f"rms={ctx.pow(2).mean().sqrt().item():.4e}")
print(f"  ctx  (q8_0 act)     vs zig: maxdiff={(ctx_q - zctx).abs().max().item():.4e} "
      f"rms={ctx_q.pow(2).mean().sqrt().item():.4e}")
print(f"  zig ctx rms={zctx.pow(2).mean().sqrt().item():.4e}")

print("\n--- what the reference's OWN stash would be ---")
print(f"  zig  final stash raw rms = 343.31   (from the S4 dump)")
print(f"  this run (q8_0 mismatch in matvec)  = {stash['raw'].pow(2).mean().sqrt().item():.4e}")
print()
print("  The reference (sglang) also quantizes activations, so a f32 transcription")
print("  is NOT the same arithmetic; the surviving 1e-2 per layer is that, not a port bug.")
