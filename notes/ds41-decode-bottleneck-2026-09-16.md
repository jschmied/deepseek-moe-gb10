# DS4.1 decode: where the step goes, and what that closes (2026-09-16)

All numbers from the engines' own counters and the loader's own `nvme_start`/`nvme_end` brackets.
Never `/proc/diskstats`. Engine work is on `deepseek-v41-flash-spark`, branch
`feat/enginev2-realleaves`; the long-form version with per-commit scope notes is in that branch's
`notes/read-depth-and-oracle.md`.

## The headline

The decode step is **not** compute waiting on I/O. It is a device that is asked for nothing about
half the time.

| | |
|---|---|
| depth 0 (no expert read in flight), 40 GB arena | **44.3 %** of the span |
| same, 79 GB arena | **51.6 %** |
| GPU busy per step (CORRECTED, see below) | **135 ms** of ~578 ms |
| oracle prefetch, horizon 1, **same bytes read** | **+37.8 %** (40 GB), **+36.0 %** (79 GB) |

The oracle reads 1896 records against the null arm's 1895 — the entire win is *when* the reads
start, not how many. And `ready_hit` 96 against `late_hit` 2292 says almost nothing arrives
finished: one layer of lead is ~417 us of compute against a ~12 ms read, so the value is head
start, not readiness. Horizon 2 adds only 4 points.

Depth-zero **rises** as the cache improves (45.9 % -> 51.6 % going 40 -> 79 GB). A better cache
means fewer misses per layer, which means less work available to keep the device busy. The pipe
gets emptier as the engine gets faster.

## CORRECTION: the "3.8 % compute ceiling" was wrong by 8x

Everything this note said about compute being negligible came from `GPU_BUSY_S = 3.0` over a 104 s
span. That number counted `CUPTI_ACTIVITY_KIND_KERNEL` only, while the profiler ran without
`--cuda-graph-trace=node`. Nsight then records each CUDA graph LAUNCH as a single activity and
emits no kernel rows for its nodes -- and decode runs entirely in captured graphs. So the table
held only the non-graph work.

Queried directly from the existing report:

| | |
|---|---|
| span | 104.08 s |
| `CUPTI_ACTIVITY_KIND_KERNEL` | 2.99 s = 2.9 % |
| `CUPTI_ACTIVITY_KIND_GRAPH_TRACE` | **21.29 s = 20.5 %** |
| union (real GPU busy) | **24.28 s = 23.3 %** |

So GPU busy is **135 ms/step, not 16.7 ms**, and compute/IO overlap is bounded at **~23 %, not
3.8 %**. The tell was in job 175's own table the whole time: `_moe_up_kernel` reported **40
launches** -- exactly one step's 40 layers -- across ~180 steps. Only the eager warm-up pass was
ever counted.

This probably also explains the open "136 vs 451 ms/step 3.3x gap": 135 ms/step is GPU busy
including graph nodes, 451 ms is the e2e step, and the difference is real idle rather than a
measurement discrepancy.

WHAT SURVIVES UNCHANGED: the oracle margin, the depth-zero fractions and the capacity/skew results.
None of them come from CUPTI -- they are wall-clock and the loader's own nvme brackets.

WHAT IS NOW SUSPECT: the per-phase split (`F_IND` 0.0105, `SHARE_MOE` 0.185, `SHARE_ATTN_HC` 0.020)
was computed by kernel NAME over that same undercounted table, so it describes the non-graph
population. Every conclusion derived from it -- including "forking the shared expert is worth
0.04 %" -- needs the phase split redone with node tracing before it can be quoted.

Fixed in `scripts/profile-early-submit.sh` (adds `--cuda-graph-trace=node`) and
`tools/prefill_budget.py` (unions the graph table, idempotently).

## Why the layer cannot fill its own pipe

A decode step verifies a 6-token block, so a layer routes 6 x top-6 = 36 pairs -> **21.9 distinct
experts**, of which ~1.7 miss. The per-layer miss distribution is:

    0 misses 23 %   1: 30 %   2: 23 %   3: 13 %   4: 6 %   5: 3 %

**53 % of layers issue at most one read**, against a device that only saturates near 2 concurrent.
For most layers the work to fill the pipe does not exist within the layer at all. Depth has to come
from outside it: lookahead, or concurrency across requests.

(Measured 23 % zero-miss layers against p^22 = 16.8 % and Poisson 18.0 % — misses cluster.)

## What is closed, and the one number that explains it

Prediction of expert IDENTITY, by every route tried:

| predictor | future-miss recall | fetch precision |
|---|---|---|
| popularity (baseline) | 3.0 % | 6.2 % |
| transition table, h=1 | 2.7 % | 6.4 % |
| prev-step-miss | 0.4 % | 1.1 % |
| **DSpark drafter -> backbone miss** | **0.4 %** | **0.5 %** |

The mechanism is capacity, not novelty. Over a run the working set converges — new (layer, expert)
pairs collapse from 63 to 3.1 per step — but it converges to **11,087 pairs, 72 % of the model**;
the median layer touches **276 of 384 experts**. **76 % of misses (code trace: 83 %) are CAPACITY
misses**: seen before, evicted, wanted again. The arena holds 23-25 % of the working set.

So LRU already absorbs everything predictable, and what is left to predict is the warm tail, where
the distribution is flat. This is *not* a short-prompt artifact — the predictor study scored only
the last 40 % of each trace, already converged — and longer context makes it worse, not better.

**Correction worth carrying:** an earlier draft of this said routing was "near-uniform", citing
entropy 8.52 of 8.58 bits. Wrong distribution and a bad statistic. Realized accesses are heavily
skewed: entropy **6.89**, **30 experts of 384 hold half the accesses**, the top-139 hold **93.3 %**.
That is why published pruning profiles at keep ~0.36 work. Entropy is a poor skew detector over 384
symbols — a top-36 % holding 60 % still scores ~8.42.

## What the levers actually are

| lever | measured |
|---|---|
| arena capacity 56.75 -> 82 GB | 3.71 -> 6.26 tok/s, hit 0.796 -> 0.894 |
| `age_over_freq` vs LRU (server) | 1.88/2.13/2.14 vs 1.74/1.85/1.86 tok/s |
| oracle prefetch h=1 | +36 to +38 %, same bytes |
| Belady vs LRU (1500 slots) | +19.0 pp hit, 47 % less NVMe |
| protection (segmented vs LRU) | 90.7 % vs 90.5 % — near null |
| compute/IO overlap, all of it | bounded at **~23 %** (was misstated as 3.8 %) |

Protection is near-null because LRU already keeps the hot set *when it fits*. At 40 GB it does not:
the hot quartile is 96 x 40 = 3,840 pairs against 2,767 slots, which is why 33.6 % of misses there
are top-quartile experts. At 79 GB (5,679 slots) it fits.

## Carried forward

- **Concurrency** is the one depth lever the entropy/capacity result does not touch: more requests
  in flight means more misses per layer to issue, with nothing predicted.
- **A better 2-bit expert format** is the largest untested lever. CB2 slot 9.992 MB against CB3's
  14.455 = 0.691x, i.e. -31 % bytes per miss *and* +45 % resident experts at once. Our CB2 costs
  -2.16 pp top-1 (softening signature: NLL down, entropy 1.71 -> 1.93, top-1 0.8067 -> 0.7940), but
  that is *our* CB2, not a 2-bit ceiling — external evidence has an unpruned 2.05 bpw beating a
  3 bpw build with the coldest 25 % of experts deleted.
- **`DSV41_BLOCK=3`** is untested and the trend favours it: BLOCK=7 measures 4.83 tok/s / 266 GB
  against BLOCK=5's 6.08 / 221, and the BLOCK=3 arm fails with `index 3 is out of bounds for
  dimension 0 with size 3` — a real bug blocking the only arm that tests narrower.
