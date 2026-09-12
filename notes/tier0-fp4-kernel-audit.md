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

---

# Tier 1 — the same checks on real layer-0 experts (2026-09-12)

One shard, `model-00003-of-00048.safetensors`, **6.88 GiB**, sha256 verified against HF's `lfs.oid`
(`e1281f85…d4c9`) — not a size check, which aria2 preallocation makes meaningless. Ran the repo's
**own** `tools/test_fp4_moe.py` unmodified with `MODEL_DIR` pointed at it, 32 expert slots.

## 1. It passes, and it validates the random-weight harness

| T | Tier 0 (random) | Tier 1 (real weights) |
|---|---|---|
| 1 | 2.82e-03 | **4.35e-03** |
| 6 / 8 | 2.90e-03 | **4.33e-03** |
| 64 | 4.41e-03 | **4.46e-03** |
| 512 | — | **4.99e-03** |

Same order of magnitude throughout, real weights ~1.5× higher at small T and converging by T=64.
**So the no-download harness is a fair proxy** — which matters, because it means future kernel work
here does not need the checkpoint. Chunk invariance and run-to-run bit-identity: **PASS** on real
weights too, confirming Tier 0.

## 2. We independently reproduce their expert-bandwidth figure

| T | distinct experts | pipelined | GB/s | their "% of 273" |
|---|---|---|---|---|
| 1 | 6 | 0.671 ms | 168.2 | 61.6 % |
| 6 | 30 | 2.911 ms | **193.7** | 71.0 % |
| 64 | 32 | 3.324 ms | 181.0 | 66.3 % |
| 512 | 32 | 9.290 ms | 64.8 | 23.7 % |

The user's brief quotes the CB3 kernels at **~186 GB/s**. We measure **168–194 GB/s** for FP4 in the
same regime on a second GB10. That is independent corroboration of the number their "near the
bandwidth floor" conclusion rests on.

## 3. …and their headroom is smaller than their own metric suggests

Their `% peak` column divides by **273 GB/s**, the theoretical figure. Our own `bwprobe.py` measures
this box at **212.8–215.0 GB/s achievable** on decode-shaped reads. Against achievable rather than
theoretical, the T=6/30-expert row is **~90 %**, not 71 %.

That *strengthens* their argument rather than weakening it: the expert kernel is nearer the floor
than they claim, so conventional kernel tuning has even less to give than they think. Caveat: bwprobe
is a GEMV read pattern and this is gathered expert weights, so treat ~215 as indicative, not as the
same measurement.

**The T=512 row is worth their attention** — 64.8 GB/s, 23.7 %. At that batch the kernel is presumably
compute-bound rather than bandwidth-bound, in which case GB/s is the wrong axis and the column reads
as a regression when it may not be one. Worth labelling, not worth alarm.
