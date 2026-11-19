#!/usr/bin/env python3
"""Run a batch of chat-templated prompts through `qwen35 run` and average acceptance.

The official figure is a benchmark average; a single ad-hoc prompt is not
comparable to it.  This walks the prompts written by `bench_prompts.py`, runs
each with spec decoding on, and reports the mean acceptance length
(`tokens/round`, the same unit the DSpine model card publishes) together with
the per-prompt spread.

    python scripts/bench_run.py --dir /mnt/tmp/bench/he --gguf <target.gguf> \
        --draft <draft.gguf> --n 128 --gpu 4 --out /tmp/bench_he.json
"""
import argparse
import json
import os
import re
import subprocess
import sys

BIN = "/mnt/llm-dev/qwen35/zig-out/bin/qwen35"
SPEC = re.compile(r"rounds=(\d+) proposed=(\d+) accepted=(\d+)")
POS = re.compile(r"^Spec-pos:(.*)$", re.M)


def run_one(prompt_path, gguf, draft, n, gpu):
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["QWEN35_SPEC_GEN"] = "1"
    env["QWEN35_SPEC_DRAFT"] = "0"
    env["QWEN35_SPEC_POSSTAT"] = "1"
    env["QWEN35_DRAFT"] = draft
    text = open(prompt_path).read()
    p = subprocess.run(
        [BIN, "run", gguf, text, "-n", str(n), "--backend", "cuda"],
        capture_output=True, text=True, env=env, timeout=3600,
    )
    m = SPEC.search(p.stdout)
    if not m:
        return None
    rounds, proposed, accepted = (int(x) for x in m.groups())
    rec = dict(rounds=rounds, proposed=proposed, accepted=accepted,
               accept_rate=accepted / proposed if proposed else 0.0,
               tokens_per_round=(accepted + rounds) / rounds if rounds else 0.0)
    pm = POS.search(p.stdout)
    if pm:
        rec["pos"] = {int(k): int(v) for k, v in
                      re.findall(r"acc(\d+)=(\d+)", pm.group(1))}
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--draft", required=True)
    ap.add_argument("--n", type=int, default=128)
    ap.add_argument("--gpu", default="4")
    ap.add_argument("--out", required=True)
    ap.add_argument("--skip", type=int, default=0)
    args = ap.parse_args()

    index = json.load(open(os.path.join(args.dir, "index.json")))[args.skip:]
    results = []
    for it in index:
        r = run_one(it["path"], args.gguf, args.draft, args.n, args.gpu)
        tag = "ok" if r else "NO OUTPUT"
        if r:
            print(f'{it["task_id"]:16s} L={r["tokens_per_round"]:.2f} '
                  f'rate={100*r["accept_rate"]:.1f}% rounds={r["rounds"]}', flush=True)
        else:
            print(f'{it["task_id"]:16s} {tag}', flush=True)
        results.append(dict(task_id=it["task_id"], **(r or {})))
        # Dump after every item: the summary below used to raise on a typo and
        # lose the whole run's data.
        json.dump(dict(items=results), open(args.out, "w"), indent=1)

    ok = [r for r in results if "tokens_per_round" in r]
    if ok:
        Ls = sorted(r["tokens_per_round"] for r in ok)
        rates = sorted(r["accept_rate"] for r in ok)
        mean_L = sum(Ls) / len(Ls)
        med = Ls[len(Ls) // 2] if len(Ls) % 2 else (Ls[len(Ls)//2 - 1] + Ls[len(Ls)//2]) / 2
        print(f"\n{len(index)} items: {len(ok)} measured")
        print(f"  mean acceptance length = {mean_L:.3f}")
        print(f"  median                 = {med:.3f}   (min {Ls[0]:.2f}, max {Ls[-1]:.2f})")
        print(f"  mean accept rate       = {100*sum(rates)/len(rates):.2f}%")
        hist = {}
        for r in ok:
            for k, v in (r.get("pos") or {}).items():
                hist[k] = hist.get(k, 0) + v
        if hist:
            tot = sum(hist.values())
            print("  accepted-count histogram: " +
                  " ".join(f"acc{k}={100*v/tot:.1f}%" for k, v in sorted(hist.items())))
        json.dump(dict(items=results, mean_L=mean_L, median_L=med,
                       mean_rate=sum(rates)/len(rates), histogram=hist),
                  open(args.out, "w"), indent=1)
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
