#!/usr/bin/env python3
"""Run the reference Triton `serial_table` on EVERY dumped spec round.

`scripts/s4_ref_compare.py` compares one round.  This walks all `call*/`
directories under a dump root and reports the agreement per round plus a total,
so "our stage 4 matches the reference" is a claim about N rounds rather than
about the one that happened to be inspected.

The verdict this supports: if our idx table and the reference's agree on every
round of a real rollout, then stage 4's transcription is not what limits
acceptance -- the decision the reference makes on the same stash is the same one
we make, right or wrong.

    python scripts/s4_ref_sweep.py <dump_root>
"""
import os
import subprocess
import sys

root = sys.argv[1] if len(sys.argv) > 1 else "/mnt/tmp/dspine/s4real3"
calls = sorted(
    (d for d in os.listdir(root) if d.startswith("call")),
    key=lambda s: int(s[4:]),
)
print(f"{len(calls)} rounds under {root}\n")

total_same = total_n = 0
mismatched = []
for c in calls:
    d = os.path.join(root, c)
    if not os.path.exists(os.path.join(d, "meta.txt")):
        print(f"{c}: no meta.txt (truncated dump), skipped")
        continue
    out = subprocess.run(
        [sys.executable, "scripts/s4_ref_compare.py", d],
        capture_output=True, text=True,
    )
    if out.returncode != 0:
        print(f"{c}: FAILED\n{out.stderr.strip()[-400:]}")
        mismatched.append((c, "error"))
        continue
    agree = chain = None
    for line in out.stdout.splitlines():
        if "idx agreement:" in line:
            agree = line.split(":", 1)[1].strip()
        if line.strip().startswith("ref :"):
            chain = line.split(":", 1)[1].strip()
    if agree is None:
        print(f"{c}: could not parse output")
        mismatched.append((c, "parse"))
        continue
    same, n = (int(x) for x in agree.replace("/", " ").split())
    total_same += same
    total_n += n
    flag = "" if same == n else "   <-- MISMATCH"
    print(f"{c}: idx {agree}  chain{chain}{flag}")
    if same != n:
        mismatched.append((c, agree))

print()
if total_n:
    print(f"TOTAL: {total_same}/{total_n} idx entries agree "
          f"({100.0 * total_same / total_n:.2f} %)")
print(f"rounds with a mismatch: {len(mismatched)}"
      + (f" -> {mismatched}" if mismatched else ""))
sys.exit(0 if not mismatched else 1)
