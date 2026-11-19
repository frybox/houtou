#!/usr/bin/env python3
"""Resumable, bandwidth-safe range downloader for the AWQ-INT4 27B checkpoint.

The shell version had two defects that cost real bandwidth:

  * a retry re-`curl`ed the whole range into the same `-o` file, discarding what
    had already arrived (curl truncates the output file), and
  * `pkill -f curl` and friends matched unrelated processes.

This one owns a fixed set of part files, resumes each with `-C -`, and never
starts more than one transfer per part.  It is safe to stop and restart.

    python scripts/fetch_ranges.py --out /mnt/tmp/awq27 --parts 8
"""
import argparse
import json
import os
import subprocess
import sys
import time

REPO = "cyankiwi/Qwen3.8-27B-AWQ-INT4"
SHARDS = [
    "model-00001-of-00005.safetensors",
    "model-00002-of-00005.safetensors",
    "model-00003-of-00005.safetensors",
    "model-00004-of-00005.safetensors",
    "model-00005-of-00005.safetensors",
]
SMALL = ["config.json", "generation_config.json", "model.safetensors.index.json",
         "tokenizer.json", "tokenizer_config.json", "vocab.json",
         "preprocessor_config.json", "video_preprocessor_config.json"]


def api_files():
    """{filename: size} from the ModelScope listing (the mirrors do not answer HEAD)."""
    url = f"https://modelscope.cn/api/v1/models/{REPO}/repo/files?Revision=master"
    out = subprocess.run(["curl", "-s", "--max-time", "120", url],
                         capture_output=True, text=True).stdout
    d = json.loads(out)
    return {f["Name"]: int(f.get("Size", 0)) for f in d["Data"]["Files"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/mnt/tmp/awq27")
    ap.add_argument("--parts", type=int, default=8)
    ap.add_argument("--host", default="https://modelscope.cn")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    sizes = api_files()
    procs = []
    for name in SMALL:
        dst = os.path.join(args.out, name)
        if os.path.exists(dst) and os.path.getsize(dst) == sizes.get(name, -1):
            continue
        procs.append((subprocess.Popen(
            ["curl", "-sL", "--retry", "5", "--max-time", "900",
             "-o", dst, f"{args.host}/models/{REPO}/resolve/master/{name}"]), None))

    active = []
    for shard in SHARDS:
        size = sizes.get(shard, 0)
        if size <= 0:
            print(f"!! unknown size for {shard}", flush=True)
            continue
        chunk = (size + args.parts - 1) // args.parts
        for p in range(args.parts):
            start = p * chunk
            end = min(start + chunk - 1, size - 1)
            want = end - start + 1
            part = os.path.join(args.out, f"{shard}.part{p}")
            if os.path.exists(part) and os.path.getsize(part) == want:
                continue
            active.append(dict(shard=shard, p=p, start=start, end=end, want=want,
                               part=part, proc=None))

    print(f"{len(active)} ranges to fetch, {len(procs)} small files", flush=True)

    def spawn(item):
        have = os.path.getsize(item["part"]) if os.path.exists(item["part"]) else 0
        if have >= item["want"]:
            return
        # Resume by asking for exactly the bytes still missing and APPENDING them.
        # `-C -` is not usable together with an explicit `-r`, and a bare `-r a-b`
        # with `-o` truncates the file and writes at offset 0 -- that silently
        # discarded 8.5 GB of progress on the first attempt at this.
        lo = item["start"] + have
        log = subprocess.DEVNULL
        item["proc"] = subprocess.Popen(
            ["curl", "-sL", "--max-time", "7200", "-r", f'{lo}-{item["end"]}',
             f'{args.host}/models/{REPO}/resolve/master/{item["shard"]}'],
            stdout=open(item["part"], "ab"), stderr=log)

    for it in active:
        spawn(it)

    last = time.time()
    tol = time.time()
    while True:
        live = [it for it in active
                if not (os.path.exists(it["part"]) and os.path.getsize(it["part"]) == it["want"])]
        for it in live:
            if it["proc"] is not None and it["proc"].poll() is None:
                continue
            got = os.path.getsize(it["part"]) if os.path.exists(it["part"]) else 0
            if got == it["want"]:
                continue
            if got > it["want"]:
                # An upstream hiccup appended too much: that part cannot be
                # trusted, so start it over rather than keep a corrupt tail.
                print(f"!! {it['shard']} part{it['p']} overshot ({got}>{it['want']}), restarting",
                      flush=True)
                os.remove(it["part"])
                it["proc"] = None
                continue
            if time.time() - tol > 6 * 3600:
                print(f"!! giving up on {it['shard']} part{it['p']}", flush=True)
                continue
            spawn(it)
        if not live:
            break
        if time.time() - last > 60:
            total = sum(os.path.getsize(it["part"]) for it in active
                        if os.path.exists(it["part"]))
            allb = sum(it["want"] for it in active)
            print(f"[{time.strftime('%H:%M:%S')}] {total/1e9:.2f} / {allb/1e9:.2f} GB "
                  f"({100*total/allb:.0f}%), {len(live)} ranges live", flush=True)
            last = time.time()
        time.sleep(2)

    for pr, _ in procs:
        pr.wait()

    rc = 0
    for shard in SHARDS:
        out = os.path.join(args.out, shard)
        with open(out, "wb") as fh:
            for p in range(args.parts):
                with open(os.path.join(args.out, f"{shard}.part{p}"), "rb") as ph:
                    fh.write(ph.read())
        got, want = os.path.getsize(out), sizes[shard]
        if got != want:
            print(f"!! {shard}: {got} != {want}")
            rc = 1
        else:
            for p in range(args.parts):
                os.remove(os.path.join(args.out, f"{shard}.part{p}"))
            print(f"ok {shard} ({got} bytes)")
    return rc


if __name__ == "__main__":
    sys.exit(main())
