#!/usr/bin/env python3
"""Read the raw activation dump produced by `QWEN35_DRAFT_DUMP=<dir>`.

Why this script exists
----------------------
The DSpine draft's 5-layer forward is being validated against a reference
implementation, and the only reliable way to find where two implementations
diverge is to compare intermediate tensors, not final outputs (the final output
is a 248320-way argmax; two different bugs look identical there).

The dump is deliberately dumb -- flat little-endian f32, one file per
(stage, layer) -- because the writer lives in a Zig release build with no
serialisation library.  All the layout knowledge lives here instead.

Layout
------
Every stage is emitted once per block row, for each of the two passes the CLI
runs (the draft's own block, then the same block with the target's hidden
injected).  So each file is:

    [pass][row][data]

with `pass` the outermost index.  Use `load(path, pass=0, row=r)` to get one
contiguous vector.

Stage shapes are derived from the model config, which this script reads from the
GGUF side-car dump the CLI prints, or takes from `--cfg`.
"""

import argparse
import struct
import sys
from pathlib import Path

# (elements per emit) keyed by stage name, for the shipped b16 checkpoint.
# q_dim = heads * hd = 32 * 128, kv_dim = kv_heads * hd = 8 * 128.
SHAPES = {
    "embed": 5120,
    "xin": 5120,
    "q": 4096,
    "k": 1024,
    "v": 1024,
    "attn": 4096,
    "h": 5120,
    "swiglu": 17408,
    "layer_out": 5120,
    "bridge_raw": 5120,
    "bridge_codes": 1024,
    "bridge_packed": 1536,
    "bridge_msg": 512,
    "bridge_delta": 5120,
}

# Layer count per stage: the bridge stages run once per layer too, but embedding
# has no layer dimension (it is emitted with layer 0).
LAYERED = {"embed": False}


def stages(directory):
    files = sorted(directory.glob("*.f32"))
    return [f.stem for f in files]


def load(directory, stage, pass_index=0, row=0, block_size=16):
    """One [shape] vector of `stage` for the given pass and row."""
    path = directory / f"{stage}.f32"
    raw = path.read_bytes()
    n = len(raw) // 4
    width = SHAPES[stage.rsplit(".", 1)[0]]
    if n % width:
        raise SystemExit(f"{path}: {n} floats is not a multiple of {width}")
    frames = n // width
    if frames % block_size:
        raise SystemExit(
            f"{path}: {frames} frames is not a multiple of block_size={block_size}"
        )
    passes = frames // block_size
    if pass_index >= passes:
        raise SystemExit(f"{path}: only {passes} passes recorded")
    at = (pass_index * block_size + row) * width
    return struct.unpack_from(f"<{width}f", raw, at * 4)


def stats(vec):
    return min(vec), max(vec), (sum(v * v for v in vec) / len(vec)) ** 0.5


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("directory", type=Path)
    ap.add_argument("--block-size", type=int, default=16)
    ap.add_argument("--pass", dest="pass_index", type=int, default=0)
    ap.add_argument("--row", type=int, default=None)
    args = ap.parse_args()

    if not args.directory.is_dir():
        raise SystemExit(f"{args.directory} is not a directory")

    for name in stages(args.directory):
        path = args.directory / f"{name}.f32"
        n = path.stat().st_size // 4
        base = name.rsplit(".", 1)[0]
        width = SHAPES.get(base)
        if width is None:
            print(f"{name:22s} {n:>10d} floats  (unknown stage)")
            continue
        frames = n // width
        passes = frames // args.block_size
        rows = [args.row] if args.row is not None else range(min(3, args.block_size))
        for r in rows:
            vec = load(args.directory, name, args.pass_index, r, args.block_size)
            lo, hi, rms = stats(vec)
            print(
                f"{name:22s} pass={passes} row={r:<2d} "
                f"min={lo:>12.4f} max={hi:>12.4f} rms={rms:>10.4f}"
            )


if __name__ == "__main__":
    sys.exit(main())
