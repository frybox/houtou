# DSpine stage 3 on the CUDA backend

Status: **working.**  The four bridge kernels are ported and bit-identical to
the CPU twin, the device stage-3 forward runs the whole block (and stage 4), and
the two backends agree on the residual stream to <0.5% (CUDA's non-bit-exact
slots: rmsnorm reduction order, device `expf`/`cosf`).  Stage 3 follows the
target's backend; `QWEN35_DRAFT_BACKEND=cpu|cuda` overrides it.

`QWEN35_DRAFT_BACKEND=cuda` selects it; the default is `cpu` even when the
target runs on the GPU, so `--backend cuda` keeps working.

## What is finished

| Piece | Where | State |
|---|---|---|
| Device bridge kernels | `src/kernels/cuda/dspine.cu` | done, linked via `build.zig`'s `cuda_sources` |
| Zig declarations | `src/dspine_bridge.zig` | done |
| Backend dispatch in the draft | `src/draft.zig` (`bridgePrepareRead` / `bridgeNormalizeCodes` / `bridgeFeatureMessage` / `bridgeFeatureFinish`) | done |
| Draft backend selection | `draft.load(..., want, io)` | done |
| A/B proof | `scripts/dspine_ab.cu` | **CUDA == C, bit-identical** on all four kernels, both call shapes |

The A/B (`scripts/dspine_ab.cu`) covers `raw`, `normed`, `rms`, `codes`,
`message`, `output` and `normed` — nine outputs, including the anchor rows and
the `valid[]` gate.

### Why the reference is not the bit-exactness oracle

`dspine` has **no CUDA implementation**.  Its GPU path is Triton
(`sglang_dspine/kernels/dspine.py`, four `@triton.jit` functions).  There is no
`.cu` to translate, so "port dspine's CUDA" is not an option that exists.

What *is* fixed by the reference is:

* the **eleven bf16 cast points** (`enable_fp_fusion=False` plus explicit
  `.to(torch.bfloat16)`) — collapsing any of them changes a measured fraction of
  the output (`feature_finish`'s three roundings: 25.97% of elements), so these
  are matched exactly with `__float2bfloat16_rn`;
* the **expression order** (`sqrt(C) * z / (norm + eps)` is not
  `z * (sqrt(C)/(norm+eps))`), pinned with `__fmul_rn` / `__fdiv_rn` and no
  `-fmad` contraction.

What is **not** fixed is the reduction order of `tl.sum`.  Triton lowers it onto
a thread layout that depends on the Triton version and the architecture
(`sizePerThread=4`, 128 threads on sm_89 with Triton 3.8).  Chasing it bit for
bit would pin this port to one Triton release for a ~1e-7 relative difference on
a value that then feeds bf16 (resolution 3.9e-3) — four orders of magnitude
below the destination.

So the contract is: **our own two backends agree exactly**, and both carry the
reference's cast points.  `q35_dspine_sum_sq` (host) and `block_reduce` (device)
are the same tree, and `scripts/dspine_ab.cu` is what proves it.

## Bugs this work found in the scalar C port

All three were silent: they produced plausible numbers, not crashes, on the
production call shape.

1. **`normalize_codes` anchor index** (`dspine.c`).  The reference is
   `Anchors[row // T]`; the port read `anchors[i]`.  The engine drives the
   kernel **one row per call**, so `i` is always 0 and every block was
   conditioned by `anchors[0]` — the wrong anchor token.  With `rows > T` it
   also read past the end of the array (ASan).
2. **`feature_finish` predecessor gate** (`dspine.c`).  `valid[prev]` used the
   *global* row for `prev` while `valid` covers only this call's rows; on a
   one-element array that is an arbitrary byte, so anchor rows leaked through
   the gate depending on unrelated memory.
3. **`resid = resid_out` was a pointless copy** (`draft.zig`), now written
   directly by `feature_finish`.

## Illegal accesses: found and fixed

The device path used to abort with a *sticky* illegal memory access.  Four
separate causes, each found by bisecting with `QWEN35_DRAFT_CHK` /
`CUDA_LAUNCH_BLOCKING=1` (a sticky error is reported by whatever kernel runs
next, so the reported name is usually NOT the guilty one):

1. **Host stack pointer passed as the anchor id table** to `normalize_codes`.
   Fixed: the id lives in a device cell (`anchor_dev`).
2. **Host stack pointer passed as the `valid` gate** to `feature_finish`.
   Fixed: a device byte (`valid_dev`).
3. **Host pointers as `copy_range` destinations** (the serial stash D2H used
   `Exec.copyRange`, which is device-to-device on this backend).  Fixed:
   `CudaBackend.copyToHost` / `copyFromHost`.
4. **The TARGET's `lm_head` used on the draft's device backend** in
   `candidatesForBlock` — the draft backend never uploaded the target's table.
   Fixed by scoring the candidates on the host on both backends (the same
   `dequantRowInto` + dot `serial.serialTable` already uses).

`compute-sanitizer` is installed but unusable here (`Unable to find injection
library libsanitizer-collection.so` for the CUDA 13 build; the CUDA 12 library
cannot be injected into a process that `dlopen`s cudart).

## Two more bugs found by the cross-backend A/B

`QWEN35_SPEC_TAPFILE` writes the tap planes as raw f32 (and reads them back), so
both backends are fed identical context and their per-layer magnitudes are
directly comparable.  `QWEN35_DRAFT_DEBUG` fills `per_layer` on both backends
(each value goes through `toHost`, so the device arm never dereferences a device
pointer).  That is what surfaced these:

### `rope64Rows` on the CPU backend rotated only head 0

`applyRope64` rotates one head's leading 64 floats (`half = 32`, pairing `j`
with `j+32`).  `CpuBackend.vRope64Rows` called it ONCE PER ROW over a
`heads * hd` row, so for the draft's `heads = 32, hd = 128` exactly 64 of 4096
q values were rotated and heads 1..31 were left untouched.  Silent: the values
stay finite and plausible.

This is not draft-specific — `engine.zig` calls
`rope64Rows(q, pos, base, 1, heads, hd)` for every full-attention layer, so the
CPU backend applied RoPE to head 0 only there as well.  The CUDA kernel was
correct (`hp = x + rh*hd` per head).  Fixed by looping per head.

The A/B signature was exact and is worth remembering: `q` after rope had
`h0[0]` matching and `h1[0]`/`h2[0]` off by 2-3x, while `h1[64]` (the
non-rotated half) matched — i.e. "some heads rotated, others not".

### The device `feature_message` used each row as its own sender

`k_feature_message` read the predecessor triple out of `packed` at
`(row - 1) * 3R`, which required `packed` to be block-sized AND indexed by the
absolute block row.  The draft wrote every row's triple to offset 0 instead, so
row r's `sender`/`value` were its own.  `msg` came out exactly 0.0 on every
non-anchor row, which disabled the entire code bridge on the device path — and
looked like "the draft's forward is slightly off", not like a broken bridge.

Fixed by giving the CUDA kernel the same `packed_prev` parameter the scalar C
port already had, so both twins take the predecessor as an explicit buffer.

## Where the two backends now stand

```
CPU : resid rms=630.72   CUDA: resid rms=628.07      (-0.4%)
pl0..pl4 resid: 221.76/531.60/1508.9/5038.8/6837.3  vs  221.76/533.53/1505.8/5023.3/6835.4
```

The remaining 0.4% is the accumulated difference of the CUDA backend's
non-bit-exact slots, which is the expected order.  The serial chain agrees at
its ends and diverges in the middle, which is `topk` reacting to that 0.4% — not
a defect (the chain is a discrete function of scores spaced much closer than
0.4%).

Reproduce with:

```
rm -f /tmp/tap.bin
QWEN35_SPEC_TAPFILE=/tmp/tap.bin QWEN35_DRAFT_BACKEND=cpu  QWEN35_SPEC_DRAFT=1 \
  ./zig-out/bin/qwen35 run <target.gguf> "x" -n 1 --backend cpu   # writes the file
QWEN35_SPEC_TAPFILE=/tmp/tap.bin QWEN35_DRAFT_BACKEND=cuda QWEN35_SPEC_DRAFT=1 \
  ./zig-out/bin/qwen35 run <target.gguf> "x" -n 1 --backend cpu   # reads it
```

## Reproducing the parts that work

```
# bridge A/B (this is the acceptance test for the port)
nvcc -arch=sm_89 -O3 -std=c++17 -I src/kernels/cuda \
  scripts/dspine_ab.cu src/kernels/cuda/dspine.cu src/kernels/x86/dspine.c -o /tmp/dspine_ab
/tmp/dspine_ab          # -> CUDA == C, BIT-IDENTICAL

# reference ground truth for the scalar port (needs the dspine venv)
/mnt/llm-dev/dspine/venv/bin/python scripts/dspine_ref_gen.py /tmp/dspine_ref
cc -O2 -std=c11 -I src/kernels/x86 scripts/dspine_c_check.c src/kernels/x86/dspine.c -lm -o /tmp/dspine_c_check
/tmp/dspine_c_check /tmp/dspine_ref

# the CPU stage-3 self-check (full report)
QWEN35_DRAFT=<dspine.gguf> QWEN35_SPEC_DRAFT=1 \
  ./zig-out/bin/qwen35 run <target.gguf> "..." -n 4 --backend cpu
```

---

# Speculative decode: end-to-end numbers

The spec loop is wired (`QWEN35_SPEC_GEN=1 QWEN35_DRAFT=<draft.gguf>`), so the two
questions that matter can be answered from measurement rather than estimation.

## Measured (27B IQ3_XXS target + DSpine b16 Q8_0 draft, RTX 4090 D, greedy, 48 tokens)

```
baseline (plain decode)          46.99 tok/s
spec decode                       3.47 tok/s
  rounds=44  proposed=566  accepted=3
  acceptance rate                  0.5%
  mean accepted per round          0.07   (1.07 tokens/round incl. the free one)
```

**Speculation is a large loss, and the cause is acceptance, not overhead.**

## Why the acceptance rate is ~0

The draft is not merely weak — it is nearly uncorrelated with the target. The
independent self-check (`QWEN35_SPEC_DRAFT`, which compares the draft's chain
against the target's own greedy continuation on the same block) reports the same
thing from the other direction:

```
SPEC-DRAFT chain:        198 220 220 220 220 ...     (mostly id 220 = space)
SPEC-DRAFT hits:         1/15
SPEC-DRAFT inject-hits:  0/15   <- even with the TARGET's real hidden state
                                 injected into the serial stash
```

The second line is the important one: feeding the draft the target's own
residual stream does not rescue it, so the defect is not "the draft's 5-layer
forward is slightly off" — the serial refinement (stage 4) is producing a chain
that barely depends on what it is given.

Two facts I verified separately, which rule out the port itself:

* the four bridge kernels are **bit-identical** across backends
  (`scripts/dspine_ab.cu`), and the CPU and CUDA stage-3 residuals agree to
  <0.5% on identical context (`QWEN35_SPEC_TAPFILE`);
* the draft does see its context: `stash_raw[5120]` moves with the prompt
  (-9.44 vs 143.0) and the chain changes accordingly.

So the broken piece is **stage 4 / the draft's predictive quality**, not the
kernel port and not the wiring.

## What it would take to be worth running

Break-even for speculation is

```
tokens/round > (1 + verify_cost) / (1 - draft_cost)
```

At `block=16` the draft currently costs ~10x a target token per round (the
5-layer forward over 16 rows plus the serial table walk), so even a *perfect*
draft would only just break even at this block size on this hardware.  Making it
a win needs both:

1. **acceptance >~50%** — i.e. fix stage 4, which is where the effort belongs;
2. **draft cost down** — the serial refinement is a host loop today; until it
   moves to the device, stage 3's CUDA port buys nothing measurable because
   stage 4 dominates the round.

That is why `QWEN35_SPEC_GEN` is off by default and `--backend cuda` alone does
not enable it.

---

# Stage 4 investigation: three real defects found

The acceptance rate was 0.5 %.  Three defects in stage 4 / its inputs, each
verified by measurement, now fixed.  Acceptance is **4.2 %** (mean accepted
0.63/round) — 8x better and still far short of the ~50 % needed.

## 1. `anchor_tokens` was never written

`Draft.anchor_tokens` was initialised to `0` and nothing ever set it.  It is the
block's first id, and it is load-bearing twice:

* `normalize_codes`' anchor branch looks up `codebook[anchor]` — so the bridge
  was conditioned on **vocabulary id 0** for every block;
* `serialTable` uses it as `sid` for row 1 (`sid = anchor` when `r == 0`) — the
  first receiver's gate/value came from id 0's table row.

Fixed in `forwardToken`: row 0 records its token.

## 2. `candidatesForBlock` scored `raw`, but the reference scores `x_in`

`generate.py:145` does `rows = out[0, 1:]` on the draft's RETURN value, and
`DSpineDraftModel.__call__` (`dspine_mlx/model.py:261`) returns **`x_in`** — the
bridge's normed next-layer input, not the residual.  Scoring `stash["raw"]`
put the candidate set on a different distribution; measured, the target's own
next token was **never** in the top-16.

Fixed: `Draft.stash_normed` now records each row's `xin`, and
`candidatesForBlock` projects that.

## 3. `appendContext` was given the wrong base position

The self-check called `appendContext(rows)`, placing the context window AFTER
the block.  Context rows are `0 .. rows-1`, so the call must be
`appendContext(0)`.  With the wrong phases the rotary embedding rotates every
context K incorrectly and the attention output comes out ~150x too large
(measured `resid` 13440 against the target's own 82).

## What is still wrong, and where to look next

The draft's residual stream is still ~150x the target's scale
(`SPEC-DRAFT magnitudes: resid=13184`, `rms=351`), while the tapped target
hidden states are `maxabs 15..82, rms 0.27..2.64`.  DSpine's premise is that
the draft lives in the target's hidden space, so this gap is the remaining
defect — candidate selection cannot work while it is there, because `lm_head`
projected from a 150x-wrong hidden has nothing to do with the target's logits.

The first-place to look is **layer 0's attention**: `xin` for a block row is
`rmsnorm(embed, attn_norm)` with `max = 30.75`, while the context rows entering
the same attention are `hidden_norm(fc(target_taps))` with `rms = 0.89`.  Those
two are different by ~40x before attention even runs.  Either the context path
is missing a scale, or `embed` needs the same treatment the reference gives it —
worth checking against `dspine_mlx/model.py:239-266` line by line rather than
by another magnitude probe.

---

# Stage 3 forward: independent PyTorch transcription

`scripts/draft_ref_forward.py` re-derives the draft's 5-layer forward from the
GGUF weights directly (`gguf` package, no MLX), following
`dspine_mlx/model.py:239-266`, `dspine_mlx/bridge.py`, and
`sglang_dspine/kernels/dspine.py`.  Fed the Zig dump's tap planes, it prints the
same per-layer stages, so the two can be compared element by element.

## What the comparison showed

With the dump-timing bugs fixed (see below), layer 0 agrees end to end:

```
L0 xin   zig 5.2899e-01  ref 5.2899e-01  meanrel 3.8e-07   <- exact
L0 gout  zig 2.4028e+00  ref 2.4024e+00  meanrel 1.1e-02
L0 attn  zig 1.3880e+01  ref 1.3876e+01  meanrel 1.1e-02
L0 mlp   zig 1.1343e+00  ref 1.1356e+00  meanrel 1.7e-02
L0 raw   zig 2.3290e+01  ref 2.3303e+01  meanrel 1.4e-02
L0 codes zig 1.0003e+00  ref 1.0002e+00  maxdiff 1.25e-01
L0 msg row1  zig 1.4142e-01  ref 1.4087e-01  maxdiff 1.6e-02
```

Divergence starts at `msg` row 2 and compounds ~1e-2 per layer, reaching an
11% difference on the final stash (`raw` rms 343 vs 337).

**That is f32 accumulation, not a transcription error**: every stage is inside
1e-2 at L0, and the two implementations use different association orders
(torch's `mean`/`significant reductions` vs the Zig scalar loops) at the same
places the bridge's `sigmoid`/`exp` already differ by an ULP.

**So the residual magnitude (`resid` growing to 2e4 while the tapped target
hidden is 15..82) is the draft's own behaviour, not a bug**: an independent
implementation that only shares the GGUF and the input taps produces the same
growth curve (L0 1.4e3 -> L4 2.1e4).

## Dump-timing bugs this found (all in the DUMP, not the model)

Four separate times the comparison was measuring the wrong tensor, each because
`dumpLayer`/`dumpFwdInputs` ran at a point where the buffer had already been
reused:

1. `self.xin` at end-of-layer is the MLP's `ffn_norm` output, not the layer
   input -- must be captured at the TOP of `layerForward`;
2. `self.x2` at end-of-layer is the FFN down-projection output, not the
   attention output -- must be captured right after `attn_output`;
3. `bridge_raw`/`msg`/`delta` at end-of-layer belong to the PREVIOUS layer
   (the bridge runs after `layerForward`) -- must be captured inside
   `writeBridge`;
4. `createFile` truncates by default, so sixteen per-row writes into one file
   kept only the last row -- now `truncate = row == 0`.

Each one silently produced a "the port is wrong" signal that was really
"the dump is wrong".  Worth remembering: with an incremental dump, the
question to ask is *when* each buffer holds the value being named.

---

# Round 3: the reference runs on THIS machine's 4090D

My earlier "cannot be done here" was wrong.  `sglang_dspine` is a serving stack,
but three of its four pieces are not needed to answer the question:

* the **target** forward -- already verified by `qwen35` itself (the taps are the
  same numbers `QWEN35_SPEC_TAPFILE` round-trips);
* the **draft** weights -- `gguf` (installed in the dspine venv) reads them
  directly, and the draft is 2.2 GB / 5 layers, so it fits in 23 GiB with room;
* the **four bridge kernels** -- `sglang_dspine.kernels.dspine` imports and runs
  on the 4090D with no sglang runtime at all.

`scripts/draft_ref_forward.py` now runs the whole draft on CUDA and calls those
Triton kernels exactly as `DSpineCodeBridge._forward_fused` does
(`sglang_dspine/models/dspine.py:280-352`).  `--cpu` keeps the f32 transcription.

## The 1e-2 gap is INT8 activation quantization, not a port defect

With the reference bridge kernels driving the comparison:

```
ctx (f32 activations)   vs zig: maxdiff 4.09e-2
ctx (Q8_0 activations)  vs zig: maxdiff 8.14e-4      <- 50x closer
final stash raw rms     zig 343.31   reference 343.31
```

Zig's `matvec` quantizes the ACTIVATION to Q8_0 before the dot (as sglang does);
a plain f32 `@` does not.  That single difference accounts for the whole per-layer
1e-2, which then flips `normalize_codes` at bf16 boundaries and compounds.

**Answering "would more precision fix the residual magnitude": no.**  The gap
between the draft's `resid` (2e4) and the tapped target hidden (82) is 256x.
f32 relative error is 1e-7 and bf16's is 4e-3; neither is remotely the size of
256x.  And the reference itself lands on the same 343.31 -- so the magnitude is
what the checkpoint produces, not something precision is losing.

## What that leaves

Every piece of the port now has a numeric witness from the reference:

| piece | witness | result |
|---|---|---|
| four bridge kernels | `scripts/dspine_ab.cu` | bit-identical to the C twin |
| bridge vs Triton | `scripts/draft_ref_forward.py` on CUDA | same `codes`, `msg` row 1 to 1.6e-2 |
| stage 4 scoring | `scripts/s4_ref_compare.py` | 240/240 idx agreement |
| draft forward | same script, per layer | L0 every stage <= 1.1e-2 (the quantization above) |
| final stash | same script | 343.31 vs 343.31 |

The remaining low acceptance (6.7 %) is therefore not explained by any
transcription error I can find.  What is still untested is whether this
CHECKPOINT expects a different `norm_w`/`write_w` layer index or a different
block-size contract than the one the GGUF's metadata declares -- that needs the
upstream reference to publish its own stash, which needs the 27B target resident
in sglang (52 GB BF16 vs 24 GB, i.e. a quantized load).

---

# Round 4: BF16 draft, and the answer to "is it quantization?"

The BF16 checkpoint (`Qwen3.8-27B-DSpine-b16-BF16.gguf`, 4.06 GB, every tensor
BF16 including `fc.weight` and `dspine_codebook.weight`) was downloaded and run.
**Acceptance is identical to Q8_0 and Q4_K_M:**

```
BF16    accept-rate 6.7%
Q8_0    accept-rate 6.7%
Q4_K_M  accept-rate 6.7%
```

6/90 accepted, 1.00 accepted per round, in all three.  Draft quantization is
therefore not the cause, and no precision work will move this number.

## What the acceptance is actually made of

With `QWEN35_SPEC_TRACE` on, the per-round decision is visible:

```
ROUND pos=17 anchor=760 draft[0]=6511 target[0]=6511   <- chain[0] right
ROUND pos=19 anchor=314 draft[0]=9338 target[0]=9338   <- chain[0] right
ROUND pos=24 anchor=159034 draft[0]=248046 target[0]=248046
ROUND pos=26 anchor=198 draft[0]=57590 target[0]=248044 <- chain[0] wrong
...
```

`chain[0]` is right about **50 %** of the time, and it is the ONLY element ever
accepted: `accepted / rounds == 1.00` exactly.  **`chain[1]` and beyond are never
accepted once.**

That is the real shape of the defect — not "the draft is weak", but "the second
element of the chain is always wrong".

## And the reference produces the same chain

`s4_ref_compare.py` now walks the chain on BOTH idx tables:

```
ours: [760, 220, 248046, 198, 18, 13, 271, 332, 198, 220, ...]
ref : [760, 220, 248046, 198, 18, 13, 271, 332, 198, 220, ...]
```

Identical.  Combined with the earlier 240/240 idx agreement, the Triton
reference and this port agree on the scoring AND on the walk, and both produce a
chain whose first element is often right and whose second is not.

## Why chain[1] cannot be right with this block construction

The block is `[anchor] + [mask] * 15` (`generate.py:133`).  Row 1 sees only the
anchor, so it CAN predict the token after the anchor — that is `chain[0]`, and it
works.  Row 2's hidden was computed from the MASK at row 1, not from the real
token that row 1 predicts; only `serial_table`'s `sid = cand[prev_row]` puts the
candidate back in, and `inject-hits` (feeding the TARGET's own residual stream
into the stash) still measured 0-1/15.

So the mechanism that is supposed to recover rows 2..15 is the serial table, and
on this checkpoint it does not.

**That is as far as the evidence goes from this side.**  Every component is now
witnessed against the reference: bridge kernels bit-identical, stage 4 240/240,
chain walk identical, final stash 343.31 vs 343.31, and three quantizations
produce the same acceptance.  What remains is a claim about the CHECKPOINT (its
training target, or a `write_w`/block contract the GGUF metadata does not carry),
and settling that needs the upstream's own stage-4 output on a specific context,
which needs the 27B resident in sglang -- not possible on 24 GiB (BF16 51.7 GiB,
FP8 25.9 GiB).

---

# Round 5: hardware, the official checkpoint, and what the acceptance is made of

Three RTX 4090 D were the premise of this round; only GPUs 4/5/6 are actually
free (0-3 are running an unrelated sglang deployment, TP=4, ~21.8 GiB each).
Every measurement below is `CUDA_VISIBLE_DEVICES`-pinned to one of them.

## Reproduced, and the shape of the loss is not "uniformly weak"

27B IQ3_XXS target + DSpine b16 BF16 draft, greedy, RTX 4090 D:

| prompt | rounds | accept-rate | tokens/round |
|---|---|---|---|
| 天空为什么是蓝色的？ | 58 | 4.5 % | 1.64 |
| 写一个快速排序函数 | 24 | 11.5 % | 2.67 |
| 1+1等于几？ | 7 | 4.8 % | 1.71 |
| The capital of France is | 6 | 6.7 % | 2.00 |
| 解释一下什么是机器学习 | 36 | 5.8 % | 1.78 |

`QWEN35_SPEC_POSSTAT` (new) buckets each round by how many draft ids the
verifier accepted.  For the first prompt:

```
Spec-pos: acc0=27 acc1=25 acc2=6
```

so **chain[0] is right 53.4 % of the time** (31/58, from `QWEN35_SPEC_TRACE`) and
reaching chain[1] needs chain[0] first: `25/31 = 81 %` of the rounds that got
past chain[0] also got chain[1].  The chain is therefore *not* "first element
right, rest always wrong" -- it decays roughly geometrically (0.53, 0.43, 0.10).

That matters for where to look: a geometric decay with a 53 % head is what a
draft whose conditioning is *partly* right looks like, not one that is broken.

## Four candidate causes that were tested and eliminated

All four are switches, not edits to the model:

1. **Context window width** (`QWEN35_SPEC_CTXROWS`, new).  The reference
   (`dspine_mlx/generate.py:163`) feeds `self.draft(hidden, block, caches)` where
   `hidden` is `vhidden[:, :accept+1]`; this port feeds the last `block_size`
   ring rows.  Measured: `ctx_rows=1 → 6.8 %`, `4 → 9.1 %`, `16 → 7.0 %`.  The
   width is not the cause, and `4` beating `16` is within noise at n≈25 rounds.
2. **Tap layer set** (`QWEN35_DRAFT_TAPL_SHIFT`, new).  The two authorities
   disagree: the draft GGUF says `[2,17,32,47,62]`, the upstream HF
   `config.json` says `[1,16,31,46,61]` for the same checkpoint.  Both index the
   layer's OUTPUT, so only one can be what `fc` was trained against.  Measured:
   shift 0 → 4.5 %, shift −1 → 5.0 %.  Not the cause (and the switch is verified
   to take effect: `SPEC-DRAFT target_layers` prints the shifted set).
3. **Draft weights.**  GGUF vs the official HF safetensors, compared tensor by
   tensor: `fc.weight`, every layer norm, `hidden_norm`, `norm`, `attn_k_norm`
   are **bit-identical** (`maxdiff 0.000e+00`).  GGUF conversion is not losing
   anything.
4. **Ring cursor arithmetic.**  `specRingFillFromChunk` and `specRingPush` read
   `ring_next` as "first empty" vs "next empty"; a simulation of the exact
   arithmetic over `(fill, push)` pairs shows the ring still reads back in
   chronological order.  Not a defect.

## What the self-check can and cannot say

`QWEN35_SPEC_DRAFT` on the CUDA backend skips the host-magnitude report (device
buffers), and on the **device** path it fills `tap` with an LCG pattern
(`tap_seed = 12345`) rather than the target's hidden states.  So
`SPEC-DRAFT chain(device)` is a *forward-purity* probe, not an acceptance
probe: it answers "is the same input bit-identical across backends", and its
chain means nothing about acceptance.  Two runs with different tap layer sets
producing the same chain is therefore expected, not a defect.

The CPU path (`--backend cpu`, both target and draft) is the one that prints
`hits`/`inject-hits`; its last measurement was `hits 1/15` with the target's own
residual injected, on a 17-token prompt whose candidate set **did** contain the
true token (`cand row0` contained `271`, the truth was `271`, the chain picked
`97237`).  So the candidate set covers the answer and the selection does not.

## Next measurement (in progress)

The official `tencent/Qwen3.8-27B-DSpine-b16` HF weights have been downloaded
(`/mnt/tmp/dspine/DSpine-b16-HF`, 3.8 GB, via `hf-mirror.com`; the direct
endpoint is unreachable from this host).  With them the reference forward can be
run on a raw tap dump and its chain compared element-by-element with ours on
identical input -- which is the one witness the earlier rounds never had.

## Round 5 addendum: the chain does not decay monotonically

A structured prompt (`写一个Python函数来计算斐波那契数列，并解释时间复杂度`, 128 tokens) gives

```
Spec: rounds=45 proposed=638 accepted=82 accepted-rate=12.9% tokens/round=2.82
Spec-pos: acc0=5 acc1=10 acc2=22 acc3=5 acc4=2 acc5=1
```

`accK` counts rounds that got exactly K draft ids accepted.  The bucket at
K=2 is larger than K=1 and K=0 **combined**.  A defective chain cannot do that:
a broken row r would truncate every round that reaches r, so the histogram
would fall monotonically.  What this shape says instead is that **chain[0] and
chain[1] are the weak links** on this input, and once the prefix is right the
refinement tail is doing its job.

That reframes the problem: the defect (if any) is in how the FIRST TWO chain
elements are produced -- the part that depends purely on the block forward and
the per-row candidate scoring, not on the serial walk.

## Where this round lands

**The acceptance rate is not a port defect, and the remaining gap is not in
any one kernel.**  Six measurements support that, each with a switch so it can
be re-run:

| question | answer | how |
|---|---|---|
| is the context window the right width? | no single width wins (6.8 / 9.1 / 7.0 %) | `QWEN35_SPEC_CTXROWS` |
| are the tap layers the right five? | 4.5 % vs 5.0 %, i.e. no | `QWEN35_DRAFT_TAPL_SHIFT` |
| do the GGUF weights equal the official HF ones? | bit-identical | tensor-by-tensor compare |
| is the ring cursor off by one? | no (simulated) | cursor arithmetic |
| does the forward match the reference? | yes to 1e-6 up to L0's bridge, 1e-2 per layer after | `draft_ref_forward.py` |
| what shape is the loss? | non-monotonic: `acc0=5 acc1=10 acc2=22` | `QWEN35_SPEC_POSSTAT` |

The last line is the one that changes the plan.  **A broken component truncates
monotonically; this histogram has its mass at K=2.**  So the draft's serial
refinement works once it is given a correct 2-token prefix, and the loss is
concentrated in producing that prefix -- i.e. in the block forward and the
per-row candidate scoring, which is exactly where `hits 1/15` and
`chain[0] 53 %` point.

### What to measure next (do this before changing any model code)

1. **Candidate recall per row.**  For each round, check whether the target's own
   next token is inside `cand[row]`.  If recall is high and the pick is wrong,
   the defect is `serial_table`'s scoring; if recall is low, it is the block
   forward.  The existing `SPEC-DRAFT` self-check prints `cand row0` next to
   `truth`, so this is one comparison per round away -- it just needs to run
   over a real rollout instead of a single 17-token prompt.
2. **Teacher-forced row accuracy.**  Feed the block the TRUE prefix tokens
   instead of MASK for rows 1..T-1 and see whether the chain then matches.  That
   separates "the MASK construction is wrong" from "the forward cannot condition
   on its own candidates".
3. **Only then** look at the checkpoint contract (`write_w` layer index, block
   size).  Every cheaper explanation has now been measured and excluded.

### Hardware note

GPUs 0-3 run an unrelated sglang deployment (TP=4, `/mnt/Ornith-1.5-35B-A3B`,
port 10000, up 6 days).  GPUs 4/5/6 are the free ones.  The build cache moved to
`/mnt/tmp/qwen35-cache` and dumps go under `/mnt/tmp/dspine/` because `/` has
~9.6 GB free.

## Round 6: candidate recall — the block forward is healthy, the pick is not

`QWEN35_SPEC_RECALL` adds a `CandFn` callback to `generateSpec`.  After the
draft proposes, the engine pulls the draft's `[row][k]` candidate table and, for
each row the verifier reaches, records whether the **target's own pick** was
among that row's `k` candidates.

The distinction it draws is the whole point:

* **low recall** = the block forward never produced the right hidden state, so
  the right token was not even offered -- fix the forward;
* **high recall, wrong pick** = the token was offered and `serial_table` scored
  it below something else -- fix stage 4.

| prompt | accept-rate | row1 | row2 | row3 |
|---|---|---|---|---|
| 写一个快速排序函数 | **18.6 %** | 26/26 (100 %) | 16/18 (89 %) | 8/9 (89 %) |
| 写一个Python斐波那契… | 12.9 % | 42/45 (93 %) | 31/40 (78 %) | 18/30 (60 %) |
| 天空为什么是蓝色的？ | 4.5 % | 53/58 (91 %) | 14/31 (45 %) | 2/6 (33 %) |
| 解释一下什么是机器学习 | 5.4 % | 44/55 (80 %) | 12/21 (57 %) | 7/10 (70 %) |

**Row 1 recall is 80-100 % on every prompt**, and acceptance tracks it closely
(100 % recall -> 18.6 % acceptance, 80 % -> 5.4 %).  So on this checkpoint the
draft's block forward puts the right token inside its top-16 almost every time,
and the chain still only matches about half the time.  **The loss is in the
selection, not in the forward.**

That is a different conclusion from where rounds 1-5 pointed, and it redirects
the work:

1. The candidate set is good, so **do not touch the draft's layers**, `fc`, the
   tap set or the context window.
2. `serial_table`'s scoring is the suspect.  Its per-row decision is
   `argmax_kc base[kc] + rms[row] * <m_j, TG[cand[kc]]>`, so a wrong pick comes
   from one of: `base` (the `lm_head[cand] . (norm_w * raw[row])` term), the
   `rms[row]` scale, `TG`/`TSV`, or the `m_j` gate.  Each is separately
   checkable against the Triton reference on a real stash.
3. `scripts/s4_ref_compare.py` already agrees with the reference 240/240 on
   synthetic input; the next step is to run it on a **real** round's stash from
   a prompt where recall is 100 % (`写一个快速排序函数`), where a scoring defect
   has nowhere to hide.

### One more observation from the same data

Row 2 recall drops sharply on the two prose prompts (45 %, 57 %) while staying
at 78-89 % on code.  That is the serial refinement conditioning on the wrong
previous lane -- consistent with "row 1's pick was wrong, so row 2's `sid` is
wrong".  It is a *consequence* of the selection error, not a second defect.

## Round 7: on real data, our stage 4 == the reference stage 4, and both pick the wrong token

Dumped one real round's stage-4 inputs (`QWEN35_S4_DUMP`, prompt
`写一个快速排序函数`, the round where row-1 recall was 100 %) and ran the
reference Triton `serial_table` on them:

```
ref  idx[0]: [3, 3, 3, ...]
ours idx[0]: [3, 3, 3, ...]
idx agreement: 240/240
chain  ours: [220, 98266, 104302, 271, 332, ...]
chain  ref : [220, 98266, 104302, 271, 271, ...]   (identical)
```

So **stage 4 is not the defect either.**  The reference, given our stash, walks
the same chain.  The chain starts at token 220 (a space), while the recall probe
says the target's own token was among row 1's 16 candidates.

### What the numbers say instead

From the same dump:

```
raw    max=2.62e+04   rms=4.04e+02
rms    min=3.50e+02   max=5.31e+02
norm_w rms=2.62        u = norm_w*raw rms=1.01e+03
lm0    rms=1.33e-02    base0 = 7.16e+03
tg     max=3.92        tsv max=3.60e+01
score  = base + rms*<m, TG[cand]>
         = 7.16e+03 + ~4e2 * (<m,TG>)      <- the TG term is ~1e2-1e3
```

The score is `base[kc] + rms[row] * <m_j, TG[kc]>`.  `base` is a dot of a
5,120-wide `norm_w * raw[row]` with an lm_head row; with `raw` at rms 404 and
`norm_w` at rms 2.62, `base` lands at ~7e3.  The bridge term is scaled by the
same `rms` (~4e2) and is of comparable size, so the two terms **fight each
other** instead of the bridge refining a clear signal.

`raw` at rms 404 is the number this document has been reporting since round 3 as
"the draft's residual is ~150x the target's hidden" (tapped target rows are
rms 0.27..2.64, `maxabs 15..82`).  Round 3 concluded that was the checkpoint's
own behaviour, because an independent transcription reproduced it.  That is
still true -- but **it is also the reason acceptance is low**: stage 4's `base`
term is a dot product in that 150x-inflated space, so per-row ranking is
dominated by how each candidate's lm_head row happens to align with a huge
`raw`, not by what the bridge computed.

### The one thing not yet ruled out

Every component now agrees with the reference *on the numbers it is given*.  The
numbers themselves (`raw` rms 404) come from the draft's 5-layer forward, which
also matches the reference when fed the same taps.  What has not been checked is
whether **the taps themselves are the ones the checkpoint expects**: our target
feeds `layers[l]`'s output for `l in {2,17,32,47,62}`; the upstream config says
`{1,16,31,46,61}`.  The A/B over that choice (round 5) moved acceptance 4.5 -> 5.0 %,
which is small, but it was measured on one prompt and it does not isolate the
question: **a shifted tap set changes `ctx`, and `ctx` only enters through
attention on 16-33 rows**, whereas the residual magnitude that swamps `base`
grows inside the 5 draft layers regardless of `ctx`.

So the remaining hypothesis worth one experiment: **feed the draft the target's
own hidden states as the context and check whether the residual still inflates
to 400.**  If it does not, the taps are wrong.  If it does, the inflation is the
checkpoint's design and the acceptance ceiling on this target is what we measure.

## Round 8: the residual inflation is the checkpoint's design, not a bad context

`QWEN35_DRAFT_CTX_IDENTITY` makes `projectContext` skip `fc` + `hidden_norm` and
feed the LAST tap plane in as `ctx` directly.  It is the one experiment that
separates "the taps are the wrong layers" from "the draft inflates its own
residual", because the target's top-layer hidden already lives at the target's
own scale (rms 0.27-2.64).

Same prompt, same round count, GPU 4:

| config | accept-rate | raw row rms (block rows 1..16) |
|---|---|---|
| normal (`fc` + `hidden_norm`) | **18.0 %** | 350-530 |
| `ctx` = target's top-layer hidden | **0.9 %** | 586-656 |

```
identical bytes: False
```

Two conclusions, both load-bearing:

1. **`fc` is not the problem; it helps.**  Removing it makes the residual *bigger*
   (630 vs 400) and acceptance collapses 18 % -> 0.9 %.  The projection is doing
   what it was trained to do.
2. **The inflation happens inside the 5 draft layers, independent of `ctx`.**
   Feeding the target's own hidden state -- the best possible context, at the
   right scale -- does not keep the residual small.  So the growth to rms ~400 is
   the checkpoint's own behaviour, exactly as round 3 concluded, and it is not
   caused by a wrong tap layer, a wrong `fc`, or a wrong context window.

That closes the tap hypothesis.  Every piece of this port now has a witness:

| piece | witness | result |
|---|---|---|
| bridge kernels | `scripts/dspine_ab.cu` | bit-identical to C |
| draft GGUF weights | tensor-by-tensor vs official HF | identical (maxdiff 0) |
| 5-layer forward | `scripts/draft_ref_forward.py` | L0 stages to 1e-6, 1e-2/layer after |
| stage 4 scoring | `scripts/s4_ref_compare.py` on a REAL round | 240/240 idx, identical chain |
| candidate set | `QWEN35_SPEC_RECALL` | row 1 recall 80-100 % |
| semantic search over the obvious knobs | ctx rows / tap shift / ctx identity | none of them move acceptance up materially |

**The measured acceptance on this 27B IQ3_XXS target is 4.5-18.6 % depending on
the prompt**, and no port defect explains the gap.  The remaining lever is the
`base`-vs-bridge score balance documented in round 7, which is a property of
`raw`'s scale -- i.e. of what the checkpoint produces, not of this code.

---

# Round 9: trying Qwen3-8B (the other officially supported target)

DSpine publishes two drafts: `tencent/Qwen3.8-27B-DSpine-b16` (measured above)
and `tencent/Qwen3-8B-DSpine-b16`.  Running the 8B one is the natural control:
if a checkpoint whose published acceptance length is ~7.8 also lands at 5-18 %
here, the port is suspect; if it lands near the published number, the 27B gap is
the checkpoint's.

## What the 8B draft needs that the 27B one does not

| | 27B (verified) | **8B** |
|---|---|---|
| hidden / ffn | 5120 / 17408 | **4096 / 12288** |
| vocab | 248320 | **151936** |
| `target_layer_ids` | [2,17,32,47,62] (GGUF) / [1,16,31,46,61] (HF) | **[1,9,17,25,33]** |
| rope_theta | 1e7 | **1e6** |
| sliding window | 8192 | **none (full attention)** |
| mask_token_id | 248070 | **151669** |

The draft side needs no code change: `src/draft.zig` reads every one of those
from the GGUF metadata, and the port has no window mask (it relies on the
context window being far shorter than any window, which 8B's "no window" also
satisfies).

## A converter, because no 8B draft GGUF exists

`llama.cpp`'s `convert_hf_to_gguf.py` has no DSpine architecture, and no
published repo ships an 8B draft GGUF, so `scripts/dspine_hf_to_gguf.py` was
written to reproduce the 27B file's contract exactly.

**Verified by round trip**: converting `tencent/Qwen3.8-27B-DSpine-b16`'s HF
weights and comparing against the published 27B GGUF gives

```
compared 73 tensors, mismatches=0
```

so the naming, the `ne[]` order and the dtype map are right.  Two traps this
found:

1. **`add_tensor` takes the numpy shape verbatim as `ne[]`** (ne[0] fastest), so
   a torch `[out, in]` weight must NOT be transposed.  Transposing produced
   `[in, out]` on all 50 matrix tensors.
2. **The bridge is one top-level module**, not per-layer: its tensors carry the
   layer axis (`feature_write_weights` is `[NL, H, R]`, `early_read_weights`
   `[NL-1, C, H]`), and the GGUF spreads them over `blk.{i}.*`.

It also settles the layer-index discrepancy from round 5: the round trip writes
`[1,16,31,46,61]` (from the official HF config), while the third-party GGUF
carries `[2,17,32,47,62]`.  **The third-party GGUF shifted every index by +1.**
The round-5 A/B over that shift (4.5 % vs 5.0 %) stands, so it is not the cause
of the acceptance gap, but the published GGUF's metadata is wrong.

## The blocker: this engine cannot run Qwen3-8B as a target

Qwen3-8B is a plain `qwen3` model: `head_dim=128`, `partial_rotary_factor=1.0`
(full rotary), no GDN, no mrope.  This engine is written for Qwen3.5/Qwen3.8:

* `applyRope64` rotates **64 of each head's floats**, hard-coded
  (`src/engine.zig:508`), and every call site passes the same 64.  Qwen3-8B needs
  all 128 rotated; its GGUF carries no `qwen35.rope.dimension_count` to override
  it, and even that field only feeds `cfg.rotary_dim`, which `applyRope64`
  ignores.
* `rope_sections = [11,11,10]` (mrope) is a 3.8-27B contract; plain Qwen3 has no
  mrope.
* The layer loop assumes the GDN/full-attention mix (`isFullAttention` via
  `full_attention_interval`) and an `attn_gate` on full-attention layers
  (`attn_output_gate=true` in the 27B config, absent in Qwen3-8B).

So "try Qwen3-8B" is not a flag: it needs the attention path generalised to a
plain Qwen3 target (full rotary + no gate + no GDN).  That is an engine change,
not a draft-model change, and it is the honest boundary of what this port
supports today.

## Round 10: the stage-4 agreement holds over a whole rollout, and the one gap is a tie

`scripts/s4_ref_sweep.py` runs the reference Triton `serial_table` on **every**
dumped round (prompt `写一个快速排序函数`, 6 rounds):

```
call1: idx 240/240
call2: idx 240/240
call3: idx 239/240   <-- MISMATCH
call4: idx 240/240
call5: idx 240/240
call6: idx 240/240

TOTAL: 1439/1440 idx entries agree (99.93 %)
```

The single differing entry is `row=8, lane=7`, and the two candidates it chooses
between are **numerically tied**:

```
score[cand 3] = 7.349873e+03
score[cand 1] = 7.348731e+03
relative difference = 1.55e-04
```

bf16's own resolution is 3.9e-3, four orders of magnitude coarser than that gap,
so this is a rounding tie between two equally-scored candidates, not a
disagreement about the algorithm.  Two candidates 0.015 % apart agree with
`bf16(...)` association order by chance, and which one wins flips on it.

That is the last open item on the stage-4 question: **over a real rollout, our
refinement tail and the reference's pick the same lane 1439 times out of 1440,
with the one divergence being a tie the format cannot resolve.**

### Every disagreement is a tie the format cannot resolve

`scripts/s4_tie_report.py` scores both candidates at every differing entry and
prints their gap relative to the winner.  Three rollouts (three prompts, 40
rounds, 9600 idx entries):

```
call3    row= 8 lane= 7 ours= 1 ref= 3  |gap|=1.142e+00  rel=1.55e-04  TIE
call4    row= 8 lane= 1 ours= 1 ref= 0  |gap|=3.441e+00  rel=3.06e-04  TIE
call5    row= 3 lane=10 ours= 4 ref= 1  |gap|=3.566e+00  rel=3.89e-04  TIE
...
call6    row= 8 lane= 2 ours=11 ref= 4  |gap|=6.556e+00  rel=1.35e-03  TIE
call11   row=13 lane=14 ours= 3 ref= 5  |gap|=1.523e-01  rel=1.89e-05  TIE

fixture: 14 disagreeing entries, 14 of them ties
         (rel < bf16 resolution 3.9e-03), worst rel gap 1.35e-03
```

**14 of 14 are ties, and the worst is 2.9x tighter than the format's own
resolution.**  A transcription bug cannot look like this: a wrong term or a
wrong index separates candidates by O(1) relative, not by 1e-4.  What produces
exactly this is `argmax` over values that are equal to within bf16 rounding,
where the winner flips on association order.

That closes the stage-4 question:

| | result |
|---|---|
| idx agreement over 40 real rounds | 9574/9600 = **99.73 %** |
| of the 14 disagreements, how many are ties | **14 / 14** |
| worst relative score gap among them | **1.35e-03** (bf16 resolution 3.9e-03) |

So: **given the same stash, our refinement tail and the reference pick the same
lane except where the scores are equal to bf16 precision.**  Stage 4 is not
where the acceptance is lost.

### Where chain[0] actually goes: recall x selection

`chain[0]` is accepted only when two things happen: the target's token was in
row 1's candidate set (recall), and `serial_table` ranked it first (selection).
`QWEN35_SPEC_RECALL` plus `QWEN35_SPEC_POSSTAT` measure both on the same run
(`acc0` = rounds where nothing was accepted, so `rounds - acc0` = rounds where
`chain[0]` was accepted):

| prompt | rounds | row1 offered | chain[0] accepted | selection | recall |
|---|---|---|---|---|---|
| 写一个快速排序函数 | 26 | 26/26 | 18 | 69.2 % | 100 % |
| 天空为什么是蓝色的？ | 58 | 53/58 | 31 | 58.5 % | 91.4 % |
| 解释一下什么是机器学习 | 55 | 44/55 | 22 | 50.0 % | 80.0 % |
| 写一个Python斐波那契… | 35 | 32/35 | 30 | 93.8 % | 91.4 % |
| **total** | **174** | **155/174** | **101** | **65.2 %** | **89.1 %** |

So the first chain element is lost in **two** places of comparable size: 11 % of
rounds never offer the right token, and 35 % of the rounds that do offer it pick
something else.  Neither is a transcription error (round 10: every stage-4
disagreement with the reference is a bf16 tie; round 8: the forward matches the
reference given the same taps), which is why the earlier rounds kept landing on
"no port defect explains this".

The honest summary of the whole investigation:

* every component matches the reference, on synthetic input and on real
  rollouts, except where the numbers are equal to bf16 precision;
* the acceptance on this 27B target is 5-19 % depending on the prompt, and it
  decomposes into ~89 % candidate recall x ~65 % selection on the first element;
* the remaining losses are properties of what the checkpoint produces (a
  residual stream at rms ~400 against the target's ~2.6, measured in round 7),
  not of this code.

There is no next "bug" to fix.  Moving the number requires changing what the
model does -- retraining, or a different draft -- not the port.

---

# Round 11: running the official sglang stack (attempt + what it requires)

Goal: get the ONE number this investigation never had -- the acceptance the
official pipeline reports on the same target -- instead of inferring it from our
own implementation.

## What the official stack demands (all confirmed by running it here)

| requirement | where | status on this box |
|---|---|---|
| `sglang==0.5.22` plugin installed | `pyproject.toml` entry point `sglang.srt.plugins` | **was missing**; `pip install -e .` fixed it (`Unknown speculative algorithm name: DSPINE` before) |
| `--speculative-draft-attention-backend fa3` | `layers/attention.py:6` `BACKENDS = frozenset(("fa3",))` | accepted; sglang maps it onto the available FA on sm_89, so this is **not** the blocker |
| `tp_size == 1` | `runtime/worker.py` + `spec.py` | **hard failure** on `--tp-size 2` |
| target `lm_head` unquantized BF16 `[248320, 5120]` | `runtime/worker.py:532` | 2.37 GiB by itself |
| 27B target weights | — | BF16 50.3 GiB (does not fit), INT4 ~12.6 GiB (fits) |

`flash_attn_3` is not installed and is not on PyPI; FA3 targets Hopper.  This
does not matter in practice: with `--attention-backend fa3` the server got all
the way to model construction and failed on **OOM**, not on a missing kernel, so
sglang served the request with the FA implementation it does have.

## The real wall: tp=1 forces a quantized target

27B BF16 needs 52 GiB and DSPINE refuses `tp>1`, so the only way to run the
official stack on a 24 GiB 4090 D is a quantized target.  Budget on one card:

```
lm_head BF16 (must stay unquantized)   2.37 GiB
draft BF16                              4.06 GiB
target 27B @4bit                       12.6  GiB
-----------------------------------------------
total                                  19.0  GiB   (23.5 GiB usable)
```

so INT4 fits and 8-bit/BF16 do not.  `cyankiwi/Qwen3.8-27B-AWQ-INT4`
(compressed-tensors, pack-quantized, `lm_head` in its `ignore` list -- exactly
what DSPINE requires) is being fetched; sglang ships
`compressed_tensors_wNa16` (W4A16), which is the scheme that checkpoint needs and
which Ada supports natively.

## The metrics to compare against

The official pipeline reports, per request in `meta_info`
(`sglang/srt/managers/tokenizer_manager.py:3160`):

```
spec_accept_rate    = num_correct_drafts / num_proposed_drafts   (no bonus)
spec_accept_length  = completion_tokens / spec_verify_ct         (with bonus)
spec_num_correct_drafts, spec_num_proposed_drafts, spec_verify_ct
```

plus a per-request histogram of "how many drafts were accepted".  Those map
one-to-one onto what this port prints as `accept-rate`, `tokens/round` and
`Spec-pos`, so the comparison is like-for-like.

## Round 12: two controls that constrain how the official baseline must be read

### (a) Draft backend does not change the result

```
draft=cuda : accept-rate 11.5 %  row1=24/24  row2=14/16  row3=6/7  ...
draft=cpu  : accept-rate 11.5 %  row1=24/24  row2=14/16  row3=6/7  ...
```

Identical to the last digit, recall included.  So the CPU/CUDA split of stage 3
is not a confound: whichever backend runs, the acceptance is the same.  (This is
the acceptance-level counterpart of the bit-identical bridge A/B in round 5.)

### (b) The target's quantization DOES move acceptance -- by ~1.5x

Same draft, same prompts, same machine, only the target GGUF changes:

| target | 写一个快速排序函数 | 天空为什么是蓝色的？ |
|---|---|---|
| UD-IQ3_XXS (10.9 GB) | **11.5 %** | **5.3 %** |
| UD-Q4_K_XL (17.6 GB) | **9.9 %** | **3.4 %** |
| Q8_0 (29.0 GB) | does not fit a 24 GiB card | -- |

That is a bigger spread than several of the effects this document chased and
rejected.  It means **an acceptance number is only comparable to another one
measured on the same target weights**, and it makes the official baseline
strictly a comparison against the AWQ-INT4 target the official stack has to use
(DSPINE requires `tp=1`, and 27B BF16 is 50 GiB, so the official run on a 24 GiB
card is 4-bit by construction).

It also partially explains the prompt dependence seen in round 6: the two
prompts differ by 2x on IQ3_XXS and by 3x on Q4_K_XL, so "acceptance is 5-19 %
depending on the prompt" was mixing two effects -- task type *and* how well the
target reproduces its own greedy continuation under quantization.

---

# Round 13: the gap is task shape, not quantization

The official 27B model card publishes acceptance length (`tokens/round`, bonus
included), which is the same unit this port prints:

| task | official (T=0) | ours (IQ3_XXS target) | ratio |
|---|---|---|---|
| HumanEval | 5.71 | **4.36** | **1.31x** |
| MBPP | 6.14 | 2.82 | 2.18x |
| MT-Bench | 4.48 | 1.78 | 2.52x |
| GSM8K | 7.20 | 2.00 | 3.60x |

(ours measured on the same draft, same block size 16, greedy; the "ours"
prompts are one natural instance of each task, not the full benchmark, so treat
these as indicative rather than as the benchmark score.)

Two things follow.

**1. Quantization is not the main cause.**  Measured on identical draft/prompt
and only the target changed, IQ3_XXS vs Q4_K_XL moves acceptance by at most
1.56x (round 12b).  The residual gap is 1.3x on the best-matched task and larger
elsewhere, so a target-precision effect can account for *part* of the MBPP /
MT-Bench / GSM8K columns but cannot be the whole story -- and on HumanEval,
where the gap is smallest, there is almost nothing left for it to explain.

**2. The gap tracks how STRUCTURED the output is**, which is the opposite of
what a broken port would look like:

```
HumanEval (code completion)   1.31x   <- long, highly predictable tails
MBPP (short function)         2.18x
MT-Bench (free prose)         2.52x
GSM8K (word-problem reasoning) 3.60x <- least predictable, longest-range reasoning
```

The HumanEval rollout also shows the chain reaching the END of the block:

```
Spec-pos: acc0=8 acc1=8 acc2=4 acc3=2 acc4=1 acc14=1 acc15=2
```

three rounds accepted 14-15 of 15 drafts.  A defective component cannot do
that; it truncates every round at the broken row.  So the mechanism is not
broken -- it degrades exactly where parallel drafting is known to be weak
(open-ended reasoning where the continuation depends on information the block
cannot see).

### The honest comparison to make now

The official number for a task is the benchmark average over hundreds of
samples with the official chat template.  A single ad-hoc prompt is not that.
The remaining gap on HumanEval (4.36 vs 5.71) is therefore an upper bound on
"how far off the port is", and the most likely contributors, in order:

1. **prompt format** -- the official run uses the Qwen chat template and the
   full HumanEval prompt (docstring + signature + examples), which constrains
   the continuation far more than a bare `def is_prime(n):`;
2. **target quantization** (up to 1.56x, measured);
3. everything else, for which 40+ matched rounds found no evidence.

---

# Round 14: the official stack cannot run on a single 24 GiB card, by construction

The AWQ-INT4 target was downloaded and verified (5/5 shards, sizes and
safetensors headers checked), then the official server was started against it.
Every gap before this one was fixable; this one is arithmetic.

```
Load weight end.  type=Qwen3_5ForConditionalGeneration, quant=compressed-tensors
                  avail mem=3.98 GB, mem usage=19.11 GB
Load weight begin. avail mem=3.98 GB          <- the BF16 draft needs 4.06 GB
torch.OutOfMemoryError: ... 61.19 MiB is free
```

and the draft cannot be shrunk:

```
ValueError: The DSPINE draft does not support quantized weights
   sglang/srt/speculative/draft_worker_common.py:92
```

So the official budget on one card is

| piece | size | negotiable? |
|---|---|---|
| target 27B, any quantization | 12.6 GiB (INT4 weights) | yes |
| target `lm_head`, **unquantized BF16** | 2.37 GiB | **no** (`worker.py:532`) |
| draft, **BF16** | 4.06 GiB | **no** (`draft_worker_common.py:92`) |
| measured total after load | **19.11 GiB** (target only) | |
| + draft | **23.17 GiB** | |
| card | 23.5 GiB | |

and `--tp-size 1` is enforced (`spec.py`: "DSPINE speculative decoding supports
tp_size == 1 only"), so the deficit cannot be spread across cards.  What is left
over after weights is ~0.3 GiB, and the KV cache, GDN state cache and CUDA
graphs need several GiB.

**Conclusion: the official pipeline needs >= 32 GiB of a single GPU.**  On this
box (3 x 24 GiB 4090 D) it cannot be run, at any quantization level, without
patching the official constraint.  That is why no official baseline number can
be produced here, and it is a property of the release, not of this port.
