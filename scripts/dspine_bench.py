#!/usr/bin/env python3
"""Ask an sglang DSpine server for its acceptance statistics.

The point of this script is to produce the ONE number the port investigation did
not have: the official pipeline's acceptance on the same target, read from the
server rather than inferred from our own implementation.

sglang reports speculative-decoding acceptance through `/metrics` (counter
`sglang:spec_accept_length` and friends) and, more directly, through the
per-request `meta_info` when `--enable-cache-report`/usage is on.  Both are tried
so the number is reported even if only one is available in this build.

    python scripts/dspine_bench.py --port 32126 --n 32 \
        --prompt "写一个快速排序函数"
"""
import argparse
import json
import statistics
import sys
import urllib.request


def post(port, path, payload):
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}{path}",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=1800) as r:
        return json.loads(r.read())


def get(port, path):
    with urllib.request.urlopen(f"http://127.0.0.1:{port}{path}", timeout=60) as r:
        return r.read().decode()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=32126)
    ap.add_argument("--n", type=int, default=32)
    ap.add_argument("--prompt", default="写一个快速排序函数")
    ap.add_argument("--model", default=None)
    args = ap.parse_args()

    # Snapshot before/after so the counters can be differenced; they are
    # server-cumulative and include any earlier request.
    before_metrics = get(args.port, "/metrics")
    payload = {
        "text": args.prompt,
        "sampling_params": {"temperature": 0, "max_new_tokens": args.n},
        "return_meta_info": True,
    }
    if args.model:
        payload["model"] = args.model

    out = post(args.port, "/generate", payload)
    meta = out.get("meta_info", {}) or {}
    text = out.get("text", "")
    print(f"generated {len(text)} chars")
    print(f"text: {text[:200]!r}")

    interesting = {
        k: v for k, v in meta.items()
        if any(s in k.lower() for s in
               ("spec", "accept", "draft", "verify", "decode", "completion"))
    }
    if interesting:
        print("\nmeta_info (spec-related):")
        for k in sorted(interesting):
            print(f"  {k} = {interesting[k]}")
    else:
        print("\nno spec fields in meta_info; keys:", sorted(meta)[:20])

    after = get(args.port, "/metrics")
    # Difference the spec counters against the pre-request snapshot.
    def spec_counters(txt):
        out = {}
        for line in txt.splitlines():
            if line.startswith("#") or not line.strip():
                continue
            name = line.split("{")[0].split()[0]
            if "spec" not in name.lower() and "accept" not in name.lower():
                continue
            try:
                out[name] = out.get(name, 0.0) + float(line.rsplit(" ", 1)[1])
            except (ValueError, IndexError):
                continue
        return out

    b, a = spec_counters(before_metrics), spec_counters(after)
    if a:
        print("\n/metrics (spec counters, delta for this request):")
        for k in sorted(a):
            print(f"  {k} = {a[k] - b.get(k, 0.0):.0f}   (cumulative {a[k]:.0f})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
