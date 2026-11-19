#!/usr/bin/env python3
"""For every idx entry where our stage 4 and the reference disagree, print how
far apart the two candidates' scores are.

The point: a transcription bug produces a disagreement between candidates that
are clearly separated (the loser may be far down the list).  A *tie* -- two
candidates whose scores differ by far less than bf16's resolution (3.9e-3) --
is what `argmax` over `bf16` values does when the association order differs, and
says nothing about the algorithm.

    python scripts/s4_tie_report.py <dump_root_or_call_dir> [...]
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, "/mnt/llm-dev/dspine")

dev = "cuda"
torch.set_grad_enabled(False)
BF16_RES = 2.0 ** -8  # 3.9e-3 relative, the format's resolution


def load(d):
    meta = {}
    for line in open(f"{d}/meta.txt"):
        k, v = line.strip().split("=")
        meta[k] = int(v)
    T, K, H, R = meta["T"], meta["K"], meta["H"], meta["R"]
    V, rows, n_rows = meta["V"], meta["rows"], meta["n_rows"]

    def f32(n, s=None):
        t = torch.from_numpy(np.fromfile(f"{d}/{n}", dtype=np.float32).copy()).to(dev)
        return t.reshape(s) if s else t

    raw = f32("raw.f32", (rows, H)).to(torch.bfloat16)
    rms = f32("rms.f32")
    query = f32("query.f32", (rows, R)).to(torch.bfloat16)
    gb = f32("gate_b.f32")
    nw = f32("norm_w.f32")
    cand = torch.from_numpy(
        np.fromfile(f"{d}/cand.i32", dtype=np.int32).copy()).to(dev).long().reshape(n_rows, K)
    anchors = torch.from_numpy(
        np.fromfile(f"{d}/anchors.i32", dtype=np.int32).copy()).to(dev).long()
    lm_ids = torch.from_numpy(
        np.fromfile(f"{d}/lm_head_ids.i32", dtype=np.int32).copy()).to(dev).long()
    lm_vals = f32("lm_head_rows.f32").reshape(lm_ids.numel(), H).to(torch.bfloat16)
    tg = torch.from_numpy(
        (np.fromfile(f"{d}/tg.u16", dtype=np.uint16).astype(np.uint32) << 16)
        .view(np.float32).copy()).to(dev).to(torch.bfloat16).reshape(V, R)
    tsv = torch.from_numpy(
        (np.fromfile(f"{d}/tsv.u16", dtype=np.uint16).astype(np.uint32) << 16)
        .view(np.float32).copy()).to(dev).to(torch.bfloat16).reshape(V, 2 * R)

    ids = sorted(set(anchors.tolist()) | set(cand.flatten().tolist())
                 | set(lm_ids.tolist()))
    id2c = {v: i for i, v in enumerate(ids)}
    VC = len(ids)
    cand_c = torch.tensor([[id2c[v] for v in r] for r in cand.tolist()], device=dev)
    anchors_c = torch.tensor([id2c[v] for v in anchors.tolist()], device=dev)
    lm = torch.zeros((VC, H), dtype=torch.bfloat16, device=dev)
    for p, g in enumerate(lm_ids.tolist()):
        lm[id2c[g]] = lm_vals[p]
    tg_c = torch.zeros((VC, R), dtype=torch.bfloat16, device=dev)
    tsv_c = torch.zeros((VC, 2 * R), dtype=torch.bfloat16, device=dev)
    for g in ids:
        tg_c[id2c[g]] = tg[g]
        tsv_c[id2c[g]] = tsv[g]
    return dict(T=T, K=K, H=H, R=R, rows=rows, n_rows=n_rows, raw=raw, rms=rms,
                query=query, gb=gb, nw=nw, cand_c=cand_c, anchors_c=anchors_c,
                lm=lm, tg_c=tg_c, tsv_c=tsv_c, ids=ids)


def score_row(m, r_words, lane):
    """The reference's own `s[j][k]`, transcribed (propose.py / kernels.py)."""
    T, K, H, R = m["T"], m["K"], m["H"], m["R"]
    row = r_words + 1  # serialTable: row = b*T + 1 + r  (b = 0 here)
    u = m["raw"][row].float() * m["nw"]
    base = torch.stack([(m["lm"][m["cand_c"][r_words, j]].float() * u).sum()
                        for j in range(K)])
    sid = m["anchors_c"][0] if r_words == 0 else m["cand_c"][r_words - 1, lane]
    g_send = m["tsv_c"][sid, :R].float()
    v_send = m["tsv_c"][sid, R:].float()
    q = m["query"][row].float()
    msg = torch.sigmoid(q + g_send + m["gb"]) * v_send
    s = torch.stack([base[kc] + m["rms"][row]
                     * (msg * m["tg_c"][m["cand_c"][r_words, kc]].float()).sum()
                     for kc in range(K)])
    return s.cpu().numpy()


def main():
    args = sys.argv[1:]
    calls = []
    for a in args:
        if os.path.isdir(a) and os.path.exists(f"{a}/meta.txt"):
            calls.append(a)
        else:
            calls += sorted((os.path.join(a, d) for d in os.listdir(a)
                             if d.startswith("call")),
                            key=lambda s: int(os.path.basename(s)[4:]))
    n_diff = n_tie = 0
    worst = 0.0
    for c in calls:
        if not os.path.exists(f"{c}/ref_idx.npy"):
            continue
        ours = np.fromfile(f"{c}/idx.i16", dtype=np.int16).reshape(-1, 16)
        ref = np.load(f"{c}/ref_idx.npy")
        d = np.argwhere(ours != ref)
        if not len(d):
            continue
        m = load(c)
        for r, lane in d:
            s = score_row(m, int(r), int(lane))
            a, b = int(ours[r, lane]), int(ref[r, lane])
            hi, lo = max(s[a], s[b]), min(s[a], s[b])
            rel = (hi - lo) / max(abs(hi), 1e-30)
            n_diff += 1
            if rel < BF16_RES:
                n_tie += 1
            worst = max(worst, rel)
            print(f"{os.path.basename(c):8s} row={r:2d} lane={lane:2d} "
                  f"ours={a:2d} ref={b:2d}  |gap|={hi - lo:.3e}  rel={rel:.2e}"
                  f"{'  TIE' if rel < BF16_RES else '  <-- REAL DIVERGENCE'}")
    print(f"\nfixture: {n_diff} disagreeing entries, {n_tie} of them ties "
          f"(rel < bf16 resolution {BF16_RES:.1e}), worst rel gap {worst:.2e}")


if __name__ == "__main__":
    main()
