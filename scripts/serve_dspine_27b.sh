#!/bin/bash
# Start the official sglang + DSpine stack on ONE 24 GiB 4090 D.
#
# Deviations from the README's command, and why:
#   * `--tp-size 1` is mandatory (DSPINE refuses tp>1), and 27B BF16 is 50 GiB,
#     so the target has to be the INT4 (W4A16) checkpoint: 12.6 GiB weights +
#     2.37 GiB BF16 lm_head (DSPINE requires it unquantized) + 4.06 GiB draft
#     = 19.0 GiB before KV cache.
#   * `--mem-fraction-static 0.80` (the README value) leaves too little for the
#     CUDA graphs on a 24 GiB card; 0.72 is what fits.
#   * `--context-length` is reduced: this measurement needs ~100 tokens of
#     context, not 128k, and the GDN state cache is the other big allocation.
#
# Usage: ./scripts/serve_dspine_27b.sh [gpu_id] [port]
set -eu
GPU=${1:-4}
PORT=${2:-32126}
MODEL=${MODEL:-/mnt/tmp/awq27}
DRAFT=${DRAFT:-/mnt/tmp/dspine/DSpine-b16-HF}

exec env CUDA_VISIBLE_DEVICES="$GPU" \
  /mnt/llm-dev/dspine/venv/bin/python -m sglang.launch_server \
    --model-path "$MODEL" \
    --speculative-algorithm DSPINE \
    --speculative-draft-model-path "$DRAFT" \
    --speculative-num-draft-tokens 16 \
    --tp-size 1 \
    --dtype bfloat16 \
    --attention-backend fa3 \
    --speculative-draft-attention-backend fa3 \
    --linear-attn-backend triton \
    --enable-linear-replayssm-spec \
    --mamba-ssm-dtype float32 \
    --mamba-radix-cache-strategy extra_buffer \
    --max-mamba-cache-size 64 \
    --mem-fraction-static 0.72 \
    --context-length 4096 \
    --max-running-requests 4 \
    --cuda-graph-max-bs-decode 4 \
    --sampling-backend flashinfer \
    --trust-remote-code \
    --host 127.0.0.1 \
    --port "$PORT"
