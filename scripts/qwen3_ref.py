#!/usr/bin/env python3
"""Independent reference forward for Qwen3, to diff against `engines/qwen3.zig`.

Why Q5_0: it is the simplest quantized layout in the set (`block_q5_0` = f16 d,
32 packed 5-bit values, one extra 32-bit high-bit plane), so a NumPy
dequantizer is a handful of lines and cannot itself be the bug.  The engine is
then compared layer by layer; the first layer whose residual rms disagrees
points at the operator.

    python scripts/qwen3_ref.py <model.gguf> [--layers N] [--token ID]
"""
import argparse
import struct
import sys

import numpy as np

# GGML tensor types used by the Qwen3 GGUFs we have.
F32, F16, Q5_0, Q8_0 = 0, 1, 6, 8
Q4_K, Q6_K = 12, 14
NAMES = {F32: "F32", F16: "F16", Q5_0: "Q5_0", Q8_0: "Q8_0", Q4_K: "Q4_K", Q6_K: "Q6_K"}


class Gguf:
    def __init__(self, path):
        self.f = open(path, "rb")
        assert self.f.read(4) == b"GGUF"
        self.version, n_tensors, n_kv = struct.unpack("<IQQ", self.f.read(20))
        self.kv = {}
        for _ in range(n_kv):
            k = self._str()
            t = struct.unpack("<I", self.f.read(4))[0]
            self.kv[k] = self._value(t)
        self.tensors = {}
        for _ in range(n_tensors):
            name = self._str()
            n_dims = struct.unpack("<I", self.f.read(4))[0]
            dims = struct.unpack(f"<{n_dims}Q", self.f.read(8 * n_dims))
            ttype = struct.unpack("<I", self.f.read(4))[0]
            offset = struct.unpack("<Q", self.f.read(8))[0]
            self.tensors[name] = (dims, ttype, offset)
        self.data_start = self.f.tell()
        self.data_start += (-self.data_start) % 32

    def _str(self):
        n = struct.unpack("<Q", self.f.read(8))[0]
        return self.f.read(n).decode("utf-8", "replace")

    def _value(self, t):
        if t == 8:
            return self._str()
        if t == 9:
            et = struct.unpack("<I", self.f.read(4))[0]
            n = struct.unpack("<Q", self.f.read(8))[0]
            return [self._value(et) for _ in range(n)]
        fmt = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i",
               6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d"}[t]
        return struct.unpack(fmt, self.f.read(struct.calcsize(fmt)))[0]

    def raw(self, name):
        dims, ttype, off = self.tensors[name]
        n = 1
        for d in dims:
            n *= d
        per = {F32: 4, F16: 2, Q5_0: 22, Q8_0: 34, Q4_K: 144, Q6_K: 210}
        if ttype not in per:
            raise SystemExit(f"{name}: unsupported {NAMES.get(ttype, ttype)}")
        elem = {F32: 1, F16: 1, Q5_0: 32, Q8_0: 32, Q4_K: 256, Q6_K: 256}[ttype]
        nbytes = n // elem * per[ttype]
        self.f.seek(self.data_start + off)
        return self.f.read(nbytes), dims, ttype

    def deq(self, name):
        """[rows, cols] float32, row-major (cols = dims[0], the GGML fastest axis)."""
        buf, dims, ttype = self.raw(name)
        cols = dims[0]
        rows = 1
        for d in dims[1:]:
            rows *= d
        if ttype == F32:
            return np.frombuffer(buf, dtype=np.float32).reshape(rows, cols)
        if ttype == F16:
            return np.frombuffer(buf, dtype=np.float16).astype(np.float32).reshape(rows, cols)
        if ttype == Q5_0:
            return deq_q5_0(buf, rows, cols)
        if ttype == Q8_0:
            return deq_q8_0(buf, rows, cols)
        if ttype == Q6_K:
            return deq_q6_k(buf, rows, cols)
        raise SystemExit(f"{name}: deq not implemented for {NAMES[ttype]}")


def deq_q5_0(buf, rows, cols):
    """block_q5_0 { f16 d; u8 qh[4]; u8 qs[16]; } over 32 elements.

    `qh` holds the 5th bit of every value: bit j for value j, bit j+16 for
    value j+16 (ggml-quants.c `dequantize_row_q5_0`).  Note the numpy `.view`
    on the 4-byte slice must produce a per-block scalar, hence the `[:, 0]`.
    """
    blk = np.frombuffer(buf, dtype=np.uint8).reshape(-1, 22)
    d = np.ascontiguousarray(blk[:, :2]).view(np.float16).astype(np.float32)[:, 0]
    qh = np.ascontiguousarray(blk[:, 2:6]).view(np.uint32)[:, 0]
    qs = blk[:, 6:22]
    out = np.empty((blk.shape[0], 32), dtype=np.float32)
    for j in range(16):
        h0 = (((qh >> np.uint32(j)) & np.uint32(1)).astype(np.uint8)) << 4
        h1 = (((qh >> np.uint32(j + 16)) & np.uint32(1)).astype(np.uint8)) << 4
        out[:, j] = (((qs[:, j] & 0x0F) | h0).astype(np.int16) - 16).astype(np.float32)
        out[:, j + 16] = (((qs[:, j] >> 4) | h1).astype(np.int16) - 16).astype(np.float32)
    out *= d[:, None]
    return out.reshape(rows, cols).copy()


def deq_q8_0(buf, rows, cols):
    """block_q8_0 { f16 d; i8 qs[32]; }"""
    blk = np.frombuffer(buf, dtype=np.uint8).reshape(-1, 34)
    d = np.ascontiguousarray(blk[:, :2]).view(np.float16).astype(np.float32)
    qs = np.ascontiguousarray(blk[:, 2:]).view(np.int8).astype(np.float32)
    return (d * qs).reshape(rows, cols).copy()


def deq_q6_k(buf, rows, cols):
    """block_q6_K { u8 ql[128]; u8 qh[64]; i8 scales[16]; f16 d; } over 256.

    Layout from ggml-quants.c `dequantize_row_q6_K`.  Each of 16 sub-blocks of
    16 values has its own signed 6-bit scale; the low 4 bits come from `ql`, the
    high 2 from `qh`.  Splitting the 128-element `ql` into two 64-halves is what
    makes the per-value indices non-obvious, so the loop mirrors the reference
    literally.
    """
    blk = np.frombuffer(buf, dtype=np.uint8).reshape(-1, 210)
    d = np.ascontiguousarray(blk[:, 208:210]).view(np.float16).astype(np.float32)[:, 0]
    ql = blk[:, :128]
    qh = blk[:, 128:192]
    sc = np.ascontiguousarray(blk[:, 192:208]).view(np.int8).astype(np.float32)
    out = np.empty((blk.shape[0], 256), dtype=np.float32)
    for n in range(2):  # two halves of 128 values
        for l in range(32):
            idx = 16 * n + l
            h = qh[:, 32 * n + l]
            q1 = ((ql[:, 64 * n + l] & 0x0F) | (((h >> 0) & 3) << 4)).astype(np.int8) - 32
            q2 = ((ql[:, 64 * n + 32 + l] & 0x0F) | (((h >> 2) & 3) << 4)).astype(np.int8) - 32
            q3 = ((ql[:, 64 * n + l] >> 4) | (((h >> 4) & 3) << 4)).astype(np.int8) - 32
            q4 = ((ql[:, 64 * n + 32 + l] >> 4) | (((h >> 6) & 3) << 4)).astype(np.int8) - 32
            s1 = d * sc[:, n * 8 + l // 16]
            s2 = d * sc[:, n * 8 + 2 + l // 16]
            s3 = d * sc[:, n * 8 + 4 + l // 16]
            s4 = d * sc[:, n * 8 + 6 + l // 16]
            out[:, idx] = s1 * q1
            out[:, idx + 32] = s2 * q2
            out[:, idx + 64] = s3 * q3
            out[:, idx + 96] = s4 * q4
    return out.reshape(rows, cols).copy()


def rms(x, w, eps):
    """RMSNorm; `x` is [..., dim] and `w` is [dim]."""
    return x / np.sqrt((x * x).mean(-1, keepdims=True) + eps) * w


def apply_rope(x, pos, base, n_heads, hd):
    """x: [n_heads*hd]; cross-half pairing over the WHOLE head (Qwen3)."""
    x = x.reshape(n_heads, hd).astype(np.float64).copy()
    half = hd // 2
    j = np.arange(half)
    inv = base ** (-2.0 * j / hd)
    theta = pos * inv
    c, s = np.cos(theta), np.sin(theta)
    a = x[:, :half].copy()
    b = x[:, half:].copy()
    x[:, :half] = a * c - b * s
    x[:, half:] = a * s + b * c
    return x.reshape(-1).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--token", type=int, default=9707)
    ap.add_argument("--layers", type=int, default=3)
    a = ap.parse_args()

    g = Gguf(a.model)
    cfg = dict(
        n_layers=g.kv["qwen3.block_count"],
        hidden=g.kv["qwen3.embedding_length"],
        n_heads=g.kv["qwen3.attention.head_count"],
        n_kv=g.kv["qwen3.attention.head_count_kv"],
        n_ff=g.kv["qwen3.feed_forward_length"],
        hd=g.kv.get("qwen3.attention.key_length", 128),
        eps=g.kv["qwen3.attention.layer_norm_rms_epsilon"],
        base=g.kv["qwen3.rope.freq_base"],
    )
    if isinstance(cfg["base"], list):
        cfg["base"] = 1e6
    print("cfg:", cfg)
    print("tensor types:", {NAMES[t[1]] for t in g.tensors.values()})

    H, NH, NKV, HD = cfg["hidden"], cfg["n_heads"], cfg["n_kv"], cfg["hd"]
    FF = cfg["n_ff"]
    eps, base = cfg["eps"], cfg["base"]

    def W(n):
        return g.deq(n)

    def norm(n):
        """A 1-D norm vector: GGUFs store them as a single row."""
        v = W(n)
        return v[0] if v.ndim == 2 else v

    emb = g.deq("token_embd.weight")  # [vocab, hidden]
    print(f"embedding row rms={np.sqrt((emb[a.token] ** 2).mean()):.5f}")
    x = emb[a.token].copy()

    for li in range(min(a.layers, cfg["n_layers"])):
        p = f"blk.{li}"
        an = norm(f"{p}.attn_norm.weight")
        hn = rms(x, an, eps)
        q = W(f"{p}.attn_q.weight") @ hn
        k = W(f"{p}.attn_k.weight") @ hn
        v = W(f"{p}.attn_v.weight") @ hn
        qn = norm(f"{p}.attn_q_norm.weight") + 1.0
        kn = norm(f"{p}.attn_k_norm.weight") + 1.0
        q = rms(q.reshape(NH, HD), qn, eps).reshape(-1)
        k = rms(k.reshape(NKV, HD), kn, eps).reshape(-1)
        q = apply_rope(q, 0, base, NH, HD)
        k = apply_rope(k, 0, base, NKV, HD)
        # Single token: softmax over one key is 1.0, so each query head's output
        # equals the V of the KV head it maps to (`head // gqa`).  The result is
        # CONCATENATED over all 32 query heads -> 32*hd, not the 8 KV heads.
        gqa = NH // NKV
        gout = np.concatenate([v[(h // gqa) * HD:(h // gqa + 1) * HD] for h in range(NH)])
        ao = W(f"{p}.attn_output.weight") @ gout
        x = x + ao
        resid_after_attn = np.sqrt((x ** 2).mean())

        fn = norm(f"{p}.ffn_norm.weight")
        hn2 = rms(x, fn, eps)
        gate = W(f"{p}.ffn_gate.weight") @ hn2
        up = W(f"{p}.ffn_up.weight") @ hn2
        act = gate / (1.0 + np.exp(-gate)) * up
        mlp = W(f"{p}.ffn_down.weight") @ act
        x = x + mlp

        print(f"L{li}: xnorm={np.sqrt((hn2**2).mean()):.4f} "
              f"attn_out={np.sqrt((ao**2).mean()):.4f} "
              f"after_attn={resid_after_attn:.4f} "
              f"mlp={np.sqrt((mlp**2).mean()):.4f} "
              f"x={np.sqrt((x**2).mean()):.4f}")

    on = norm("output_norm.weight") + 1.0
    hf = rms(x, on, eps)
    print(f"final normed rms={np.sqrt((hf**2).mean()):.4f}")
    out = g.deq("output.weight") @ hf
    top = np.argsort(-out)[:5]
    print("logits rms=%.4f max=%.4f" % (np.sqrt((out**2).mean()), out.max()))
    print("top5:", [(int(i), float(out[i])) for i in top])


if __name__ == "__main__":
    sys.exit(main())
