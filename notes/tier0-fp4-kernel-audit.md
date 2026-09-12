# Tier 0 — auditing 0xBakeer's FP4 MoE kernel with no checkpoint

2026-09-12, one GB10 (sm_121), torch 2.13.0+cu130, repo at `8b68fdd`.

## Why this is possible at all

Their kernels are standalone: `tools/fp4_moe.py`, `cb3.py`, `decode_attn.py` import only torch and
triton; `cb3_moe.py` imports its sibling. `fp4_moe.py` ships **both** `moe_forward` and
`moe_forward_reference` (dequant + torch matmul, "mirroring `inference/model.py::Expert.forward`"),
and `ExpertArena.load_slot` takes plain uint8 tensors. So their own three checks run with **random
weights instead of the 475 GiB checkpoint** — `tools/test_fp4_moe_random.py`, which is their test
logic with `load_arena()` swapped.

Packed bytes are uniform random (every byte is two valid e2m1 nibbles, so in-distribution). UE8M0
scale bytes decode as `2^(b-127)`, drawn from 119–126 so the dynamic range resembles real weight
scales rather than spanning 2^±127 and producing inf.

## Result: the FP4 MoE kernel is clean here

| T | max\|diff\| | rel err | max\|ref\| |
|---|---|---|---|
| 1 | 1.60e+01 | **2.82e-03** | 3328 |
| 8 | 1.60e+01 | 2.90e-03 | 4128 |
| 32 | 1.90e+01 | 3.20e-03 | 3952 |
| 64 | 3.20e+01 | **4.41e-03** | 5664 |

- **Determinism: 4/4 bit-exact** over repeated identical calls at T = 1, 8, 64.
- **Chunk invariance: no prefix** of a 64-row batch differs from the same rows of the full batch.

## What that says about their two open questions

**The unexplained gap.** `LIMITATIONS.md` says the graphed decode path "differs by a few percent
relative and is not yet explained". This kernel contributes **0.3–0.4 %** against its own reference,
rising mildly with batch (2.8e-3 → 4.4e-3 from T=1 to 64). So either the gap is accumulation of this
across 40 layers, or it is **not in this kernel** — attention, routing composition, or the graph
itself. That is a narrowing, not an answer, and it is worth telling them which.

**Their spec-lossless gate.** `engine/test_spec_lossless.py` has no recorded passing run. If it
fails, **it is not because this kernel is nondeterministic** — it is bit-exact and chunk-invariant
here. That removes one candidate before they spend time on it.

## Caveats, stated because they matter

- **Random weights are not their weights.** Uniform nibbles have a different distribution from
  trained FP4 experts, and the reference is *their own* torch port, not DeepSeek's tilelang. This
  measures kernel-vs-reference consistency, not correctness against DeepSeek.
- Single run of each cell. The determinism check is 5 calls in one process, which does not test
  across restarts — and our own `mtp-restart-instability` found up to 1.83× timing spread across
  restarts on a different stack.
- Only `fp4_moe` so far. **CB3 is their shipped expert format** (`EXPERT_FORMAT=cb3`, keep 0.44), and
  `cb3_moe.py` needs a `CB3Arena` plus the packer in `cb3.py`, so it is a larger harness. That is the
  one that matters most and is not done.

## Tier 1, in flight

`tools/test_fp4_moe.py:17` hardcodes **one** shard, `model-00003-of-00048.safetensors`, and reads
`layers.0.ffn.experts.{e}.` from it — **6.88 GiB**, verified against the HF API, not 475.3 GiB.
Downloading at a 6 MB/s cap. That gives their own test on real layer-0 experts, and a
random-vs-real control on the numbers above.
