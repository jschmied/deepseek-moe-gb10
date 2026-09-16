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
| GPU kernel time per step | 16.7 ms of ~437 ms |
| oracle prefetch, horizon 1, **same bytes read** | **+37.8 %** (40 GB), **+36.0 %** (79 GB) |

The oracle reads 1896 records against the null arm's 1895 — the entire win is *when* the reads
start, not how many. And `ready_hit` 96 against `late_hit` 2292 says almost nothing arrives
finished: one layer of lead is ~417 us of compute against a ~12 ms read, so the value is head
start, not readiness. Horizon 2 adds only 4 points.

Depth-zero **rises** as the cache improves (45.9 % -> 51.6 % going 40 -> 79 GB). A better cache
means fewer misses per layer, which means less work available to keep the device busy. The pipe
gets emptier as the engine gets faster.

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
| compute/IO overlap, all of it | bounded at 3.8 % |

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
