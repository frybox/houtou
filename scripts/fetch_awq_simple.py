#!/usr/bin/env python3
"""Download the AWQ shards with `curl -C -`, one process per shard.

Why not range-split again: the previous attempt split each shard into 8 byte
ranges and concatenated them, which produced files whose first 8 bytes were
float data instead of the safetensors header length -- the parts had been
appended in the wrong order (an earlier aborted run had already written to
them).  Manual offset bookkeeping is the part that broke, not the transfer.

`curl -C -` tracks the offset itself from the file's current size and refuses to
write out of order, so resume is correct by construction.  Throughput is lower
per connection, so every shard runs concurrently.
"""
import json
import os
import subprocess
import sys
import time

REPO = "cyankiwi/Qwen3.8-27B-AWQ-INT4"
SHARDS = [f"model-0000{i}-of-00005.safetensors" for i in range(1, 6)]
SMALL = ["config.json", "generation_config.json", "model.safetensors.index.json",
         "tokenizer.json", "tokenizer_config.json", "vocab.json",
         "preprocessor_config.json", "video_preprocessor_config.json"]
HOST = "https://modelscope.cn"


def api_sizes():
    url = f"{HOST}/api/v1/models/{REPO}/repo/files?Revision=master"
    out = subprocess.run(["curl", "-s", "--max-time", "120", url],
                         capture_output=True, text=True).stdout
    return {f["Name"]: int(f.get("Size", 0)) for f in json.loads(out)["Data"]["Files"]}


def safe_header_ok(path):
    """A safetensors file starts with a little-endian uint64 header length."""
    if not os.path.exists(path):
        return False
    with open(path, "rb") as fh:
        head = fh.read(8)
    if len(head) < 8:
        return False
    n = int.from_bytes(head, "little")
    if not (0 < n < 100_000_000):
        return False
    with open(path, "rb") as fh:
        fh.seek(8)
        blob = fh.read(n)
    return blob.startswith(b"{") and blob.rstrip().endswith(b"}")


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else "/mnt/tmp/awq27"
    os.makedirs(out, exist_ok=True)
    sizes = api_sizes()

    for name in SMALL:
        dst = os.path.join(out, name)
        if os.path.exists(dst) and os.path.getsize(dst) == sizes.get(name, -1):
            continue
        subprocess.run(["curl", "-sL", "--retry", "5", "--max-time", "900",
                        "-o", dst, f"{HOST}/models/{REPO}/resolve/master/{name}"])

    # Discard anything that failed the header test: `-C -` would happily append
    # to a corrupt prefix.
    for name in SHARDS:
        dst = os.path.join(out, name)
        if os.path.exists(dst) and not safe_header_ok(dst):
            print(f"removing corrupt {name}", flush=True)
            os.remove(dst)

    procs = {}
    for name in SHARDS:
        dst = os.path.join(out, name)
        if os.path.exists(dst) and os.path.getsize(dst) == sizes[name]:
            print(f"{name} already complete", flush=True)
            continue
        procs[name] = subprocess.Popen(
            ["curl", "-sL", "--retry", "10", "--retry-delay", "3",
             "--max-time", "20000", "-C", "-", "-o", dst,
             f"{HOST}/models/{REPO}/resolve/master/{name}"])

    print(f"{len(procs)} shard downloads started", flush=True)
    t0 = time.time()
    last = 0.0
    while procs:
        for name, pr in list(procs.items()):
            if pr.poll() is not None:
                del procs[name]
        if time.time() - last > 60:
            have = sum(os.path.getsize(os.path.join(out, s)) for s in SHARDS
                       if os.path.exists(os.path.join(out, s)))
            want = sum(sizes[s] for s in SHARDS)
            print(f"[{time.strftime('%H:%M:%S')}] {have/1e9:.2f}/{want/1e9:.2f} GB "
                  f"({100*have/want:.1f}%) {len(procs)} live", flush=True)
            last = time.time()
        time.sleep(3)

    rc = 0
    for name in SHARDS:
        dst = os.path.join(out, name)
        got = os.path.getsize(dst) if os.path.exists(dst) else 0
        hdr = safe_header_ok(dst)
        if got != sizes[name] or not hdr:
            print(f"!! {name}: {got} != {sizes[name]} header_ok={hdr}", flush=True)
            rc = 1
        else:
            print(f"ok {name} ({got} bytes, header ok)", flush=True)
    print(f"done in {time.time()-t0:.0f}s rc={rc}", flush=True)
    return rc


if __name__ == "__main__":
    sys.exit(main())
