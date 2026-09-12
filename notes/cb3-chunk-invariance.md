# CB3, the shipped expert format, is not chunk-invariant — and is not tested for it

2026-09-12, one GB10 (sm_121), repo at `8b68fdd`. Harnesses:
`tools/test_cb3_moe_random.py` (no checkpoint) and `tools/test_cb3_moe_real.py` (layer-0 experts
from the one 6.88 GiB shard).

## Why CB3 and not FP4

Tier 0/1 covered `fp4_moe` only. **CB3 is what v0.4.0 actually serves** (`EXPERT_FORMAT=cb3`,
keep 0.44), and `engine/v41_engine.py:370` calls `C3.moe_forward_v3` with `C3.CB3ArenaV2` — which is
exactly the path tested here, verified at source before measuring.

Method mirrors their own `test_cb3_moe.py`: a `CB3ArenaV2` and an `ExpertArena` holding **the same**
8-level requantized weights (`sim.requant_packed`), so the comparison isolates the CB3 kernel from
the 3-bit quantization loss instead of conflating them.

## Two passes, one failure

| check | random weights | real layer-0 weights |
|---|---|---|
| CB3 kernel vs FP4 kernel, identical 8-level weights | 0.0 / 6.2e-05 / 0.0 (T=1/6/64) | 2.8e-05 / 1.7e-05 / 0.0 |
| determinism, 5 identical calls | **4/4 bit-exact** | **4/4 bit-exact** |
| **chunk invariance** | **fails at prefixes 1, 2, 4, 8** | **fails at prefixes 2, 4, 8** |

The kernel itself is excellent — against the FP4 kernel on identical weights it is essentially
**exact**, so the format costs nothing numerically, and CB3 is 14.45 MB/slot against FP4's 18.80
(ratio 0.769). Determinism is clean. The failure is specifically that **a prefix of a batch does not
reproduce the same rows of the full batch**, at small prefixes only; 16 and 32 pass.

## Why this is a finding and not a nitpick

1. **They engineered for this property and documented it.** `LIMITATIONS.md`: *"`engine/model.py` is
   now bit-exact under chunking … **identical, not merely close**"*, achieved by disabling
   `allow_bf16_reduced_precision_reduction` and forcing fixed 16-row tiles, because *"cuBLAS picks
   tiling AND split-K from M, and for several shapes even a row's offset inside the tile changes its
   last bits (8 and 16 are offset-invariant, 32/64/128 are not)"*. `CHANGELOG` repeats it.
2. **Our failure pattern matches their own explanation** — it breaks below 16 rows and holds at 16
   and 32, which is what a tile/BM choice keyed on M would do.
3. **Their CB3 test does not check it.** Zero hits for chunk/prefix/invariance in
   `tools/test_cb3_moe.py`; only `test_fp4_moe.py` checks it, and FP4 passes (we confirmed at Tier 1).
   So the property is verified for the format they do **not** ship and unverified for the one they do.
4. **The regime it breaks in is the one they rely on.** Their own FP4 test says why it matters: *"the
   engine prefills a prompt in chunks and re-runs speculative blocks after a rollback, so a prefix
   must match."* A **6-token** speculative block re-run sits squarely inside the failing T ≤ 8 range.

**A plausible link to their open problem, offered as a hypothesis, not a claim:**
`engine/test_spec_lossless.py` exists and has **no recorded passing run** anywhere in `RESULTS.md`,
and `LIMITATIONS.md` still concedes an unexplained few-percent gap between the graphed decode path
and `Model.forward`. A non-chunk-invariant MoE on the shipped format would produce exactly that
symptom. We have not tested that chain end to end — we cannot, without the 510 GB.

## What we did not establish

- Only layer-0 experts, 8 slots, one box, one run per cell.
- We did not trace *which* line makes it M-dependent; `_UP_CFG`/`_DOWN_CFG` are keyed by a block size
  and `_pick_bm(P)` exists, so a tiling choice from M is the obvious suspect — unverified.
- Whether it actually breaks generation. It is a bit-exactness failure, which their own standard
  calls a defect, but the output-level consequence is untested.
