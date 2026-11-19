#!/usr/bin/env python3
"""Convert a DSpine draft checkpoint (HF safetensors) to the GGUF this port reads.

Why this exists
---------------
`scripts/draft_ref_forward.py` and the reference pipelines load DSpine drafts from
HF `model.safetensors`; `src/draft.zig` loads them from GGUF.  For
Qwen3.8-27B-DSpine-b16 a GGUF already existed (`aj9o9/Qwen3.8-27B-DSpine-b16-GGUF`),
but for Qwen3-8B-DSpine-b16 neither `llama.cpp`'s `convert_hf_to_gguf.py` nor any
published repo provides one, so the 8B acceptance could not be measured at all.

The contract this reproduces is not invented: it is exactly the metadata and
tensor naming of the 27B draft GGUF that `src/draft.zig` is verified against
(see `docs/dspine-stage3-cuda.md`).  Every field below is read from the draft's
own HF `config.json`, so the same script handles both checkpoints.

    python scripts/dspine_hf_to_gguf.py <hf_dir> <out.gguf> [--tokenizer <gguf>]

`--tokenizer` copies the tokenizer fields + `tokenizer.chat_template` from an
existing target GGUF, because the draft GGUF carries them and the CLI reads
`tokenizer.ggml.mask_token_id` from it.
"""
import argparse
import json
import sys

import numpy as np
import torch
from gguf import GGUFWriter, GGUFReader, GGUFValueType
from safetensors import safe_open

BF16 = 30
F32 = 0


def bf16_bits(t: torch.Tensor) -> np.ndarray:
    """[n, m] -> uint16 bits of the bf16 rounding, as GGUF stores them."""
    return t.to(torch.bfloat16).contiguous().view(torch.int16).numpy().view(np.uint16)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("hf_dir")
    ap.add_argument("out")
    ap.add_argument("--tokenizer", default=None)
    args = ap.parse_args()

    cfg = json.load(open(f"{args.hf_dir}/config.json"))
    df = cfg["dflash_config"]
    bridge = df["selected_prefix_code_bridge"]

    H = int(cfg["hidden_size"])
    NL = int(cfg["num_hidden_layers"])
    NH = int(cfg["num_attention_heads"])
    NKV = int(cfg["num_key_value_heads"])
    HD = int(cfg["head_dim"])
    FFN = int(cfg["intermediate_size"])
    V = int(cfg["vocab_size"])
    T = int(cfg["block_size"])
    K = int(cfg["code_bridge_second_pass"]["head_topk"])
    TAPS = [int(x) for x in df["target_layer_ids"]]
    R = int(bridge["rank"])
    C = int(bridge["code_size"])
    ROPE = float(cfg.get("rope_theta") or cfg["rope_parameters"]["rope_theta"])
    EPS = float(cfg["rms_norm_eps"])
    MASK = int(df["mask_token_id"])
    WINDOW = cfg.get("sliding_window")

    print(f"draft: L={NL} H={H} ffn={FFN} heads={NH}/{NKV}x{HD} V={V} "
          f"R={R} C={C} T={T} K={K} taps={TAPS} rope={ROPE} window={WINDOW}")

    w = GGUFWriter(args.out, "dflash")

    w.add_block_count(NL)
    w.add_context_length(int(cfg["max_position_embeddings"]))
    w.add_embedding_length(H)
    w.add_feed_forward_length(FFN)
    w.add_head_count(NH)
    w.add_head_count_kv(NKV)
    w.add_key_length(HD)
    w.add_value_length(HD)
    w.add_rope_freq_base(ROPE)
    w.add_layer_norm_rms_eps(EPS)
    w.add_uint32("dflash.block_size", T)
    w.add_uint32("dflash.selector_top_k", K)
    w.add_array("dflash.target_layers", TAPS)
    w.add_bool("dflash.attention.causal", True)
    # rope sections: the ported kernel rotates the leading 64 of each 128-wide
    # head (`applyRope64`), which is `[64, 0, 0, 0]` for head_dim 128.
    w.add_array("dflash.rope.dimension_sections", [64, 0, 0, 0])
    if WINDOW is not None:
        w.add_uint32("dflash.attention.sliding_window", int(WINDOW))
        # `dflash.attention.sliding_window_pattern` is one bool per LAYER in the
        # 27B file (`[True]*5`), but `src/draft.zig` never reads it: the port has
        # no window mask and relies on the context window being far shorter than
        # any window.  Emit nothing here so the field cannot be misread.
    w.add_uint32("tokenizer.ggml.mask_token_id", MASK)

    # ---- tensors ----
    W = {}
    with safe_open(f"{args.hf_dir}/model.safetensors", "pt") as f:
        for name in f.keys():
            W[name] = f.get_tensor(name)
    print(f"read {len(W)} tensors")

    def put_bf16(gguf_name, tensor):
        assert tensor.dtype == torch.bfloat16, (gguf_name, tensor.dtype)
        # GGUF stores `ne[0]` as the FASTEST-varying dimension, and `add_tensor`
        # takes the numpy shape verbatim, so a torch [out, in] weight goes in
        # as-is: ne[0] = in (the row length `dimOf(name, 0)` wants), ne[1] = out.
        # Transposing here was measured wrong -- all 50 matrix tensors came out
        # [in, out] against the reference file.
        #
        # numpy has no bfloat16, so the bits are reinterpreted as int16 first.
        a = tensor.contiguous().view(torch.int16).numpy().view(np.uint16)
        w.add_tensor(gguf_name, np.ascontiguousarray(a), raw_dtype=BF16)
        print(f"  {gguf_name:36s} bf16 {list(tensor.shape)}")

    def put_f32(gguf_name, tensor):
        a = tensor.float().contiguous()
        w.add_tensor(gguf_name, a.numpy(), raw_dtype=F32)
        print(f"  {gguf_name:36s} f32  {list(a.shape)}")

    put_bf16("fc.weight", W["fc.weight"])                       # [H, taps*H]
    put_f32("enc.output_norm.weight", W["hidden_norm.weight"])  # `hidden_norm`
    put_f32("output_norm.weight", W["norm.weight"])             # `self.norm`

    for i in range(NL):
        p = f"layers.{i}"
        put_f32(f"blk.{i}.attn_norm.weight", W[f"{p}.input_layernorm.weight"])
        put_f32(f"blk.{i}.ffn_norm.weight", W[f"{p}.post_attention_layernorm.weight"])
        put_f32(f"blk.{i}.attn_q_norm.weight", W[f"{p}.self_attn.q_norm.weight"])
        put_f32(f"blk.{i}.attn_k_norm.weight", W[f"{p}.self_attn.k_norm.weight"])
        put_bf16(f"blk.{i}.attn_q.weight", W[f"{p}.self_attn.q_proj.weight"])
        put_bf16(f"blk.{i}.attn_k.weight", W[f"{p}.self_attn.k_proj.weight"])
        put_bf16(f"blk.{i}.attn_v.weight", W[f"{p}.self_attn.v_proj.weight"])
        put_bf16(f"blk.{i}.attn_output.weight", W[f"{p}.self_attn.o_proj.weight"])
        put_bf16(f"blk.{i}.ffn_gate.weight", W[f"{p}.mlp.gate_proj.weight"])
        put_bf16(f"blk.{i}.ffn_up.weight", W[f"{p}.mlp.up_proj.weight"])
        put_bf16(f"blk.{i}.ffn_down.weight", W[f"{p}.mlp.down_proj.weight"])

    # The bridge is a SINGLE top-level module, not per-layer: its weights carry a
    # layer axis (`feature_write_weights` is [NL, H, R], `early_read_weights`
    # [NL-1, C, H]).  The GGUF spreads them across `blk.{i}.*` instead.
    FW = W["selected_prefix_bridge.feature_write_weights"]     # [NL, H, R]
    ER = W["selected_prefix_bridge.early_read_weights"]        # [NL-1, C, H]
    for i in range(NL):
        put_bf16(f"blk.{i}.dspine_write.weight", FW[i])

    # `read`: the LAST layer uses the shared `bridge.read_weight`, the others the
    # per-layer `early_read_weights[i]` (`bridge.py:layer_read_weight`).
    for i in range(NL):
        # The LAST layer shares `read_weight`; earlier layers have their own
        # `early_read_weights[i]` (`bridge.py:layer_read_weight`).
        t = W["selected_prefix_bridge.read_weight"] if i == NL - 1 else ER[i]
        put_bf16(f"blk.{i}.dspine_read.weight", t)

    put_bf16("dspine_gate.weight", W["selected_prefix_bridge.gate.weight"])
    # HF stores the bias in BF16; the 27B GGUF stores it F32 (`dspine_gate.bias`
    # is read as f32 by `draft.load`).  Widening is exact.
    put_f32("dspine_gate.bias", W["selected_prefix_bridge.gate.bias"])
    put_bf16("dspine_pred_gate.weight",
             W["selected_prefix_bridge.predecessor_gate.weight"])
    put_bf16("dspine_prev.weight", W["selected_prefix_bridge.prev.weight"])
    # `dspine_codebook.weight` is stored [C, V] (dim0 = code row length C).
    put_bf16("dspine_codebook.weight", W["selected_prefix_bridge.codebook"])

    # ---- general.* ----
    w.add_name(f"{cfg.get('_name_short', args.hf_dir.rstrip('/').split('/')[-1])}")
    w.add_uint32("general.file_type", 32)
    w.add_string("general.organization", "Tencent")
    w.add_string("general.finetune", "DSpine-b16")
    w.add_string("general.base_model.0.organization", "Qwen")
    w.add_string("general.base_model.0.name", cfg.get("_base_name", ""))
    w.add_uint32("general.base_model.count", 1)
    w.add_array("general.tags", ["dspine", "speculative-decoding", "draft-model"])

    # ---- tokenizer, borrowed from a target GGUF ----
    if args.tokenizer:
        tr = GGUFReader(args.tokenizer)
        copied = 0
        for k, f in tr.fields.items():
            if not k.startswith("tokenizer."):
                continue
            if k in ("tokenizer.chat_template",):
                w.add_string(k, f.contents() if isinstance(f.contents(), str) else "")
                copied += 1
                continue
            if k in ("tokenizer.ggml.tokens", "tokenizer.ggml.merges",
                     "tokenizer.ggml.token_type", "tokenizer.ggml.scores"):
                continue
            try:
                w.add_key_value(k, f.contents(), f.types[1] if len(f.types) > 1 else f.types[0])
                copied += 1
            except Exception as e:  # noqa: BLE001
                print(f"  tokenizer field {k} skipped: {e}")
        print(f"copied {copied} tokenizer fields from {args.tokenizer}")

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    print(f"wrote {args.out}")


if __name__ == "__main__":
    sys.exit(main())
