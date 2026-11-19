#!/usr/bin/env python3
"""Build identical-input prompts for a benchmark-style acceptance measurement.

The problem this solves: this CLI takes raw text (`qwen35 run <gguf> <prompt>`),
while the official number is measured on chat-templated benchmark items.  Feeding
a bare `def is_prime(n):` and comparing the result to the published HumanEval
figure is not a like-for-like comparison -- the chat template and the full
item prompt both constrain the continuation far more.

So the prompt is rendered here with the model's OWN `tokenizer.chat_template`
(read from the GGUF) and handed over as text, which reproduces exactly what a
serving stack would tokenize.

    python scripts/bench_prompts.py --dataset humaneval --limit 20 --out /mnt/tmp/bench/he
    python scripts/bench_prompts.py --dataset gsm8k    --limit 20 --out /mnt/tmp/bench/gsm8k
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
from gguf import GGUFReader

PKG = "/mnt/llm-dev/dspine/venv/lib/python3.12/site-packages"


def chat_template(gguf_path):
    r = GGUFReader(gguf_path)
    f = r.fields.get("tokenizer.chat_template")
    return f.contents() if f is not None else None


def render(tmpl, user_text):
    from jinja2 import Environment

    env = Environment(trim_blocks=False, lstrip_blocks=False)
    # `generation`/`endgeneration` are Qwen3.8's markers around the assistant
    # body; the plain-Jinja rendering used here only needs them to be no-ops.
    env.globals["raise_exception"] = lambda m: (_ for _ in ()).throw(ValueError(m))
    fn = env.from_string(tmpl)
    return fn.render(
        messages=[{"role": "user", "content": user_text}],
        add_generation_prompt=True,
        enable_thinking=False,
        tools=None,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["humaneval", "gsm8k"], required=True)
    ap.add_argument("--limit", type=int, default=20)
    ap.add_argument("--gguf", default="/mnt/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-IQ3_XXS.gguf")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    if args.dataset == "humaneval":
        df = pd.read_parquet("/mnt/tmp/he.parquet").head(args.limit)
        items = [(row.task_id, row.prompt) for row in df.itertuples()]
    else:
        path = "/mnt/tmp/gsm8k.parquet"
        df = pd.read_parquet(path).head(args.limit)
        items = [(f"gsm8k/{i}", row.question) for i, row in enumerate(df.itertuples())]

    tmpl = chat_template(args.gguf)
    if tmpl is None:
        print("no chat_template in the GGUF; falling back to raw prompts")
        rendered = [(i, t) for i, t in items]
    else:
        rendered = [(i, render(tmpl, t)) for i, t in items]

    index = []
    for i, (task_id, text) in enumerate(rendered):
        p = os.path.join(args.out, f"{i:02d}.txt")
        with open(p, "w") as fh:
            fh.write(text)
        index.append(dict(idx=i, task_id=task_id, path=p, chars=len(text)))
    with open(os.path.join(args.out, "index.json"), "w") as fh:
        json.dump(index, fh, indent=1, ensure_ascii=False)
    print(f"wrote {len(index)} prompts to {args.out}")
    print(f"sample ({index[0]['task_id']}, {index[0]['chars']} chars):")
    print(repr(rendered[0][1][:400]))


if __name__ == "__main__":
    sys.exit(main())
