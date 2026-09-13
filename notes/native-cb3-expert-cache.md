# Proposal: a native on-disk CB3 expert cache

**Origin: the user, 2026-09-13.** Recorded verbatim in substance below, then checked against the
code and today's measurements. The short verdict: the mechanism is real, the code confirms every
structural claim, and our own numbers make it a decode optimization rather than only a startup one —
though the headline arithmetic uses a miss rate about 3× ours.

## The observation

For an LRU miss today the expert is read from NVMe in its **original packed FP4 checkpoint
representation, even when the active arena is CB3**. `ExpertStore._read_leased` reads 18.8 MB per
expert by O_DIRECT — packed FP4 weights plus UE8M0 scales — and hands those FP4 tensors to
`arena.load_slot()`, where `CB3ArenaV2` converts them to CB3 on the way into the slot
(`engine/experts.py:272` → `tools/cb3_moe.py:412`, which calls `fp4_to_cb3_v2`).

So a CB3 cache miss is:

```text
NVMe → 18.8 MB FP4 + scales → pinned CPU staging → 18.8 MB H2D → GPU
     → FP4→CB3 packing → CB3 arena slot (~14.45 MB)
```

A native on-disk CB3 expert would instead be:

```text
NVMe → ~14.45 MB CB3 → pinned CPU staging → ~14.45 MB H2D → CB3 arena slot
```

Three gains at once, and possibly a fourth:

1. ~23 % less NVMe bandwidth per miss.
2. ~23 % less staging and H2D traffic.
3. No FP4→CB3 GPU packing on every miss.
4. **One contiguous extent instead of two runs.** The safetensors layout puts all the scale tensors
   near the front of the shard and all the weight tensors far behind, so an expert is two disk runs
   ~585 MB apart — `ShardFile.expert_runs` exists precisely to cope with that. A purpose-built cache
   can store `[header][w1 CB3][s1][w3 CB3][s3][w2 CB3][s2]` as one record, so one expert becomes one
   O_DIRECT extent.

Implementation: a persistent, expert-major, contiguous CB3 cache file, and `_load_into_slot()`
detects that format and copies straight into `CB3ArenaV2`, bypassing `fp4_to_cb3_v2()` entirely. That
attacks the 183 s warm start and the ongoing swap cost with one change.

## Checked against the code and the checkpoint

Every structural claim holds:

| | bytes | vs FP4 |
|---|---|---|
| FP4 checkpoint expert | 18,800,640 (17.93 MiB) | — |
| CB3 arena slot | 14,454,784 (13.79 MiB) | **0.769** |
| CB2 arena slot | 9,992,192 (9.53 MiB) | 0.531 |

The saving is **23.1 %**, exactly as stated. The two-run layout is confirmed from the shard headers:
a 1.05 MiB scale run and a 16.88 MiB weight run, each starting unaligned so O_DIRECT overreads one
page per run.

## What our own measurements make of it

The worked example uses **100 swapped experts/token**. Our measured figure at the shipped 98 GB arena
is **31.0 misses/token** (§1), so the effect is about a third of the headline — still material:

| | today (FP4 on disk) | native CB3 on disk |
|---|---|---|
| bytes/token from NVMe | 31.0 × 18.80 MB = **0.582 GB** | 31.0 × 14.45 MB = **0.448 GB** |
| at the measured 5.2 GB/s single-expert read rate | 112 ms/token | **86 ms/token** |

**~26 ms/token saved on a path whose whole NVMe term is 112 ms** — and that is before the removed
GPU packing and before the two-runs-to-one change. Against the ~1,108 ms/step of the streaming path
it is smaller than the schedule fixes (≈1.25–1.35×), but it is *orthogonal* to them and it composes:
the schedule work makes the reads overlap, this makes them smaller.

The point about relative difficulty is the strongest part of the argument and I agree with it:
0xBakeer already worked hard to get 4.7–5.5 GB/s out of this SSD path (we measure 5.0–5.6 at one
expert in flight, 6.8 at two), and **removing 23 % of the bytes is much easier than finding another
23 % of bandwidth**, which §3 shows is not there to find.

## Two things to settle before building it

* **Disk.** A full CB3 cache for all 15,360 experts is **222.0 GB** against the 288.8 GB the FP4
  experts occupy today. Keeping both means 511 + 222 = 733 GB, which does not fit beside everything
  else on a 916 GB disk. Deleting the FP4 experts and treating CB3 as the serving source gives
  222 + 203 (Engram) + 19 (dense) = **444 GB, i.e. 67 GB less than today** — but then the checkpoint
  on the box is a lossy derived artifact and the FP4 original has to live on the backup server. That
  is a fine trade, it just has to be a decision rather than a side effect.
* **It is a cache of a lossy transform.** CB3 is fitted per row by `CodebookSim`, so the cache is
  only valid for the codebook version that produced it. It needs a format/version stamp checked at
  load, or a `CodebookSim` change silently serves stale quantization — the same class of trap as our
  unstamped stored measurements.

## Where it sits against the other levers

It stacks with, and does not overlap, the two byte levers already on the list: **2-bit scales**
(bit-exact, −4.7 % of a CB3 slot, no quality gate) would make the cached record 13.77 MB and take the
saving from 23.1 % to **26.8 %**; a **CB2 tier** would make it 9.99 MB and 46.9 %, at a real quality
cost. And it is the only one of the three that also removes work (the per-miss repack) rather than
only bytes.

Related: [[ds41-measured-2026-09-13]] §1 (0.582 GB/token), §3 (the read path is not the bottleneck),
§6c (CB2's measured wall ratio 0.801).
