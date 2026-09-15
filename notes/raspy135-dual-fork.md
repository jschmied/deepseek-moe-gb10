# raspy135's dual-Spark fork: what we checked, what we took, what it saved us from trying

*Reviewed 2026-09-15. `raspy135/deepseek-v41-flash-spark_dual`, forked from 0xBakeer 2026-09-13,
118 commits ahead / 101 behind, merge base `8b68fdde1`. One person, two days, 0 stars, and about
60 % of the delta is single-box kernel work that has nothing to do with the second box. Treat it as
a snapshot to mine, not a dependency — upstream has since gone a different direction entirely
(contribution-based expert ranking, narrow keep-set profiles) with zero overlap.*

## Their headline recommendation does NOT apply to us — checked, not assumed

`e01cfd307` fixes `build_routing` reserving and scanning block space per **arena slot** instead of
per distinct expert touched: 164 ms -> 0.44 ms, prefill 226 -> 406 tok/s. A 1.8x prefill for +17/-1
lines is worth checking hard.

We already do it, by a different route. `build_routing` costs `NB = ceil(P/BM) + min(n_slots, P)`,
and `moe_forward_prefill` remaps slot ids through `inv` before routing, so it passes the **scratch**
arena (384) and not the expert arena (5,728); the batched fallback passes `b <= 32`. Distinct
experts per chunk is ~362 against 384, so ~6 % of a small number is all that is left.

We hit the same wall once and fixed it elsewhere: `build_routing_small`'s docstring records "the
Triton router costs ~29 ms per step with a 4,800-slot arena", which is why decode has its own
builder.

## What we took

**`tl.dot_scaled`** (`e760f4599`): hand e2m1 codes plus UE8M0 scales straight to the tensor core
instead of decoding nibbles in software and calling `tl.dot`. They measure 19.0 -> 27.3 TFLOPS
isolated (1.44x), **1.19x in-phase, error vs the dequantised reference identical**. Our
`_moe_up` 5.88 s + `_moe_down` 3.11 s + `_fp4_linear` 1.64 s is 10.6 s of a ~40 s GPU-busy prefill.
`tl.dot_scaled` exists in our Triton 3.7.1. Port queued, gated, with the scale-layout question as an
explicit stop condition.

**Their prefix-cache invariance test design**, worth stealing verbatim: a resumed prefill must equal
a cold prefill *under the same kernel*, byte for byte, **and** the test must assert the cache was
actually used (`prefix_cached_tokens == 2259`, deliberately not a multiple of the 2,048 chunk).
Equality alone proves nothing, because a broken cache leaves both runs cold and agreeing.

## What it saved us from trying — the most valuable part of the repo

From `docs/gotchas.md` (their 147 added lines) and `docs/dual-spark-plan.md`:

| they tried | result |
| --- | --- |
| wider verify block `DSV41_BLOCK=9` | GPU utilisation up, decode **27 % slower** (162.7 -> 217.4 ms/step). Distinct experts/layer 21.1 -> 29.9, bytes/step 7.94 -> 11.23 GB; acceptance flat (the DSpark head is trained at block 5). *"Fuller tiles are not the goal; tokens per byte read is."* |
| pinned staging for the engram H2D | works, engram wait 58 -> 11 ms/step, **decode rate does not move** — the host just blocks in verify instead. We measured the same null independently. |
| engram read throttles / pacing | added, then removed: "both measured harmful" |
| dense tensor-parallel | 19.56 -> 14.39 tok/s, TTFT 14.9 -> 19.8 s |
| autotuning the FP4 MoE kernel | 36 BM x BN x warps x stages combinations, **nothing** beat the shipped default. The win was a primitive (`dot_scaled`), not tuning. |
| fp8 for the indexer score buffer | top-512 overlap 75.5 % vs bf16's 98.3 %; also `topk`/`amax`/`masked_fill` are unimplemented for `float8_e4m3fn`, and with no infinity `-inf` masking saturates to -448 by accident of the value range |
| `DSV41_PRUNE_HALFLIFE=2e6` | churned ~100 experts/request at a 0.9 % miss rate to fix nothing; `2e7` cut churn 5x **and** halved the miss rate |

And one measurement trap we are exposed to: their `decode_accounting` timed the eager path while
CUDA graphs replayed a capture, so 46-75 % showed as "unaccounted" and looked like an idle GPU.
**A host timer around a graph replay measures queueing, not work.** Four kernel rewrites were
benchmarked against the wrong phase before the real (routing) bug was even suspected.

## Their numbers, and why they are not comparable to ours

README: prefill ~1047 tok/s on 4,513 tokens, decode 13-24 tok/s, 256k context, 28.4 % residency.
The one machine-readable run (`results/bench-session-end.json`) is honest — `isl 8192, osl 512,
runs 3, warmup 1, ignore_eos`, engine-side counters — and gives TTFT 14,919 ms, **decode 19.56
tok/s**, prefill 285 tok/s, accept 3.56.

But it carries `expert_hit_rate: 1.0, nvme_gb: 0.0, prune_keep: 0.55`: **all-resident pruned mode
with zero NVMe streaming**. Our 6.3 tok/s is the full-router streaming configuration. Different
experiment, not a different result. Their 512-token generation does at least avoid the
short-generation trap that makes our own 64-token runs read 1.4-2.5 tok/s.

Could not establish: their per-box NVMe O_DIRECT rate (their own plan still lists it "not yet
measured"), and the provenance of the README's 13-24 tok/s range — only the 19.56 point has a file.
