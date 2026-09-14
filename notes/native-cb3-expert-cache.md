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

---

## Scale survey: done, all 40 layers, and 3 bits is exactly lossless

`tools/scale_survey.py`, 2026-09-13. Reads **only** the scale region of each layer shard — one
contiguous ~425 MB span near the front — straight off the backup server with a remote `dd`, so it
never copies a 7.4 GB shard and never needs the checkpoint local. 40 layers, 384 experts each,
**149,422,080 rows**, 17.0 GB read, ~4 minutes.

| | |
|---|---|
| distinct exponent values per layer | 6–11 |
| order-0 entropy | 0.975–1.031 bits/byte |
| worst intra-row range, any layer | **7** |
| rows with range ≤ 7 | **100.000000 %** |
| rows with range ≤ 3 | 99.7965 % (layer 39) – 100.0000 % |

**A 1-byte row base plus 3 bits per group is exactly lossless on every row of every layer. No escape
path is needed.** The 3-layer sample generalised; layer 39 is the worst case and it still fits. Note
2 bits does *not* — ≤3 fails on 0.2 % of layer 39's rows — so 3 is the width, not 2.

### The record, frozen

Per expert: w1 and w3 are 2304 rows × 160 groups → 1 + 60 = 61 B/row; w2 is 5120 × 72 → 1 + 27 = 28.
Scales go **1,105,920 → 424,448 B**, saving 681,472.

```text
record  = [w1 CB3 | s1 3-bit][w3 CB3 | s3 3-bit][w2 CB3 | s2 3-bit]  = 13,773,312 B
padded to 4096                                                        = 13,774,848 B  (1,536 B pad)
record i at offset i * 13,774,848,   i = layer*384 + expert,   no index
```

| | per-expert bytes | vs FP4 | full cache | slots at a 98 GB arena |
|---|---|---|---|---|
| FP4 checkpoint (today) | 18,800,640 | 1.000 | 288.8 GB | — |
| CB3, 8-bit scales | 14,454,784 | 0.769 | 222.0 GB | 6,779 = 44.1 % |
| **CB3, 3-bit scales** | **13,774,848** | **0.733** | **211.6 GB** | **7,114 = 46.3 %** |

So the disk cache saves **26.7 % per miss**, not 23.1 %, and the whole thing is 211.6 GB.

### Two separable wins, and only one of them is cheap

* **Disk record only.** Store 3-bit scales in the cache, expand to 8-bit while writing into the
  arena. The kernel is untouched. This is what the cache needs, it is bit-exact, and the expansion
  replaces the FP4→CB3 conversion that a miss pays today — strictly less work, not more.
* **Arena too.** Compressing the scales *in memory* is the +2.2 pp coverage and −4.7 % kernel-bytes
  item, but the CB3 kernel reads `s1/s2/s3` directly, so it needs a decode in the inner loop. Given
  §6b — the kernel is occupancy-limited, not byte-limited, below ~38 experts in flight — that one
  should be measured before it is built, and it is no longer obviously free.

### Disk arithmetic, updated

On-box with a CB3 cache and the FP4 experts deleted: 211.6 + 203 (Engram) + 19 (dense) = **434 GB**,
against 511 GB today, leaving ~130 GB free on this disk. The encode streams one layer shard at a time
from the backup server, so peak local footprint during the build is ~227 GB — the full 510 GB
checkpoint never has to be here at all. Read cost ~48 minutes of LAN plus GPU conversion.

---

## Built, wired into the engine, and measured — the win is 6–12×, and not for the reason expected

2026-09-13. `tools/scale_codec.py`, `tools/cb3_cache_build.py`, `tools/cb3_cache_bench.py`,
`tools/test_cb3_cache.py`; engine side on branch `feat/cb3-disk-cache` of the fork
(`engine/cb3_cache.py` + 43 lines in `engine/experts.py`).

**Correctness first.** The codec round-trips bit-exactly on real scales in both numpy and torch. A
cache built for 8 experts, loaded through `CB3Cache` into a `CB3ArenaV2`, is **byte-identical in all
twelve planes** to the same experts loaded the FP4 way, and `moe_forward_v3` output is bit-identical
(max |delta| 0.0). The builder asserts the scale round-trip on **every** expert, not a sample, and
reads back one record in 32.

**Then the benchmark**, both arms through the engine's own `ExpertStore._load_into_slot`, 96 loads,
median of 3, 64-slot CB3 arena, layer 20:

| arm | conc | MB/load | ms/load | GB/s |
|---|---|---|---|---|
| FP4 checkpoint | 1 | 18.81 | 25.54 | 0.74 |
| FP4 checkpoint | 2 | 18.81 | 24.92 | 0.75 |
| FP4 checkpoint | 4 | 18.81 | 25.28 | 0.74 |
| **native CB3 cache** | 1 | 13.77 | **4.14** | 3.33 |
| **native CB3 cache** | 2 | 13.77 | **2.23** | 6.18 |
| **native CB3 cache** | 4 | 13.77 | **2.04** | 6.76 |

**6.2× at one expert in flight, 12.4× at four.** That is far past the 1.27× the byte count predicts,
so the bytes are not what is driving it. Decomposed:

| | ms |
|---|---|
| read 18.8 MB at the measured 5.5 GB/s | 3.4 |
| **`arena.load_slot` — `fp4_to_cb3_v2` over three tensors + H2D** | **20.9** |
| one 3-bit scale plane expanded on the device | 0.064 |

**86 % of a CB3 miss today is the repack, not the read.** The 26.7 % byte saving is the minor term;
the major one is that the cached record is already in the arena's layout. Two independent checks
agree: the FP4 arm does not improve at all from conc 1 to 4 (the repack serialises on the GPU while
the reads do not), and the repo's own warm start of 6,160 experts at 183 s is 29.7 ms/expert, which
is this cost.

### Scope — be careful with this number

The 6–12× is a property of **a CB3 arena taking misses**. It does not apply everywhere:

* **Pruned all-resident** (the shipped `PRUNE_KEEP=0.44 EXPERT_FORMAT=cb3`) has no decode misses at
  all, so this only shortens the 183 s warm start — to roughly 30 s.
* **An FP4 arena** pays no repack, so there the win would be the bytes alone, ~1.27×.
* **Unpruned streaming with a CB3 arena** is where it is worth 6–12× per miss. Whether the 2.68 tok/s
  baseline ran CB3 or FP4 experts is **not stated** in the run's env block and I have not confirmed
  it. That has to be settled before any end-to-end claim.

There is also a compounding case if that baseline ran FP4: a CB3 cache lets the arena be CB3 —
**7,114 slots at 98 GB against 5,213**, i.e. 46.3 % coverage against 33.9 % — *without* paying the
repack that made CB3 unattractive for a streaming arena in the first place. More coverage, fewer
bytes per miss, and no repack, from one change.

### Build cost

44.4 s per layer of 384 experts (115.6 ms/expert, dominated by the same `fp4_to_cb3_v2`), so ~30 min
of conversion for all 40 layers, overlapped with ~48 min of LAN transfer: **call it an hour**. Peak
local footprint is the cache plus two shards; the 510 GB checkpoint is never resident.

## Where the code lives, and why

The **format** — writer, reader, codec, correctness test — lives together in the fork, because a
record written by one version and read by another is the failure mode this whole note is about:

| | |
|---|---|
| `deepseek-v41-flash-spark/tools/scale_codec.py` | the 3-bit codec, numpy + torch, and its identity string |
| `deepseek-v41-flash-spark/tools/cb3_cache_build.py` | the writer |
| `deepseek-v41-flash-spark/engine/cb3_cache.py` | the reader, importing the same codec module |
| `deepseek-v41-flash-spark/tools/test_cb3_cache.py` | bit-identity against the FP4 path |

They ship and version as one unit, on branch `feat/cb3-disk-cache`, and the reader now refuses a
cache whose manifest carries a different `format_version` or `codec` string rather than serving
skewed quantization silently. The reader no longer has its own copy of the bit math — it calls
`scale_codec.unpack_torch`, so there is exactly one definition.

The **evidence and the measurement** stay in this repo, because they are about deciding, not shipping:

| | |
|---|---|
| `deepseek-moe-gb10/tools/scale_survey.py` | the proof that 3 bits is lossless on all 40 layers |
| `deepseek-moe-gb10/tools/cb3_cache_bench.py` | the FP4-vs-cache A/B through `ExpertStore` |

Rule of thumb for next time: **if the engine has to agree with it at runtime, it goes in the fork.**

---

## Built for real: 211.6 GB, verified, and what it is actually worth

2026-09-13, 16:16. All 40 layers, 15,360 experts, **211,581,665,280 B (197 GiB)**, ~50 minutes,
layer shards streamed from the backup server and deleted behind so the 510 GB checkpoint was never
resident. Every expert's scale round-trip asserted during the build; one record in 32 read back and
compared. `tools/test_cb3_cache.py` against the finished artifact: **all twelve planes byte-identical
to the FP4 path, `moe_forward_v3` output bit-identical, max |delta| 0.0**.

The manifest was written by the pre-relocation module, so `format_version` and `codec` were stamped
in afterwards — the record bytes are unaffected, only the identity strings were added.

### Load path, on the real artifact

Same layer, both arms through `ExpertStore._load_into_slot`:

| arm | conc 1 | conc 2 | conc 4 |
|---|---|---|---|
| FP4 checkpoint (18.81 MB) | 25.54 ms | 24.57 | 24.99 |
| native CB3 cache (13.77 MB) | 4.38 | 2.40 | 2.28 |

And reads scattered over the whole 197 GiB file, which a real miss stream would be — the single-layer
test could not show this:

| conc | 1 | 2 | 4 | 8 | 12 |
|---|---|---|---|---|---|
| ms/load | 3.58 | 2.77 | 3.05 | 3.01 | 2.98 |
| GB/s | 3.84 | 4.97 | 4.52 | 4.58 | 4.62 |

**Scatter costs nothing** — 3.58 ms against 4.38 for the single-layer case at conc 1. The record being
one aligned extent is doing its job.

### What it is worth, stated honestly against the right baseline

The 6–11× is **CB3-arena-with-cache against CB3-arena-without**. Nobody runs the latter: the measured
streaming baseline is `kernel triton-fp4` (RESULTS §4), and an FP4 arena pays no repack. So that
number is not a speedup over the shipped configuration — it is the reason CB3 was **unusable** as a
streaming arena, now removed.

Against the FP4 baseline the value is that a CB3 slot is smaller, so the same arena bytes hold more
experts. Simulated on the real trace, held out, `transient_slots=8`, global warm start:

| arena | slot | slots | coverage | misses/token | **GB/token** |
|---|---|---|---|---|---|
| 73.8 GB (the baseline's own) | FP4 18.80 MB | 3,925 | 25.6 % | 42.56 | **0.800** |
| 73.8 GB | **CB3 13.77 MB** | 5,357 | 34.9 % | 29.94 | **0.563** |
| 98 GB | FP4 | 5,212 | 33.9 % | 30.96 | 0.582 |
| 98 GB | **CB3** | 7,114 | 46.3 % | 19.70 | **0.370** |

**1.42× less NVMe traffic at the same arena bytes**, 1.57× at 98 GB. The measured baseline moves
916 MB/token and decodes at 2.68 tok/s with essentially all of the 188 s decode being NVMe, so
0.704× traffic is **≈3.8 tok/s, +42 %** — *derived*, chaining a simulated miss rate onto their
measured baseline, and the sim runs 13 % low on absolute misses (42.6 against their 48.7) so it is
ratios that carry, not levels.

Two honest caveats. The MoE kernel changes from `triton-fp4` to the CB3 kernel; §6c measured CB3
against CB2 but never against FP4, so that side is unpriced. And the end-to-end number needs the
engine actually running, which is now unblocked: **434 GB on box with the cache against 511 raw**,
351 GB free right now.

---

## The cache alone does not unblock the engine — the dense pack does

Finishing the cache exposed something the disk arithmetic had missed. With `DSV41_CB3_CACHE` set the
engine never reads an expert from the checkpoint, but `engine/model.py::Weights` still loads
**attention, shared-expert, gate and hyper-connection** weights by name through the safetensors
index, and those live inside the same 7.4 GB layer shards. So "434 GB on box" was wrong: serving
still implied 296 GB of layer shards, and 296 + 203 + 197 does not fit on this disk.

The fix is cheap because of how the shards are laid out. Per layer the non-expert tensors are only
**181 MB and form exactly two contiguous runs** (measured on the real files), so they can be pulled
by byte range instead of streaming the shard:

| | |
|---|---|
| `tools/dense_pack_build.py` | 40 per-layer packs, **6.88 GB total**, ~1 min of LAN against 296 GB and 48 min |
| verification | all 37 tensors of layer 20 re-read from the pack are **bit-identical** to the shard, dtypes and shapes included |
| `tools/lean_dir_build.py` | merges the dense packs over the original index, symlinks the shards still needed whole, writes a model dir the engine can open unmodified |

The packs are assembled byte-for-byte with the original dtype strings rather than round-tripped
through torch, so no dtype mapping can go wrong.

One deliberate choice: **expert tensors keep their original shard filenames in the merged index, and
those files are absent.** With the cache on they are never opened, so if that ever stops being true
the failure is a loud missing file rather than a silent wrong answer.

### What a serving box actually needs

| | GB |
|---|---|
| CB3 expert cache | 211.6 |
| dense packs (40 layers) | 6.9 |
| Engram tables (shards 47, 48) | 203.1 |
| embeddings, head, Engram aux (shards 1, 2, 43–46) | 9.5 |
| **total** | **431** |

against **511 GB** for the raw checkpoint — and, more usefully, the 296 GB of layer shards never has
to be resident at any point, including during the build.

---

## It runs. First engine start of this project, and three corrections it produced

2026-09-13 17:22. `./start.sh` against the lean directory with `DSV41_CB3_CACHE` set, unpruned
(no `PRUNE_KEEP`), `EXPERT_FORMAT=cb3`, `MAX_SEQ=32768`, arena auto.

```
17:21:55 weights: all non-expert weights on GPU in 77s      <- the dense packs
17:24:10 arena 79.9 GB = 5530 cb3 expert slots (36% of all routed experts, auto)
17:24:31 warm start done: 5130 experts resident (74.2 GB, 77.9 GB read) in 15s
17:24:31 ready
```

**Up in 103 s from cold.** A greedy code generation returns correct, well-formed Python.

### Warm start: 183 s → 15 s, measured

The repo's stored warm start is **183 s for 6,160 experts** (29.7 ms each). On the cache: **5,130
experts in 15 s**, 2.9 ms each — **~10×**, and it is the first end-to-end confirmation of the
`_load_into_slot` microbenchmark rather than another bench of the same thing.

### Correction 1: `TRANSIENT_SLOTS=8` is unusable for unpruned streaming

The first start came up and then failed every request with
`RuntimeError: transient ring exhausted: more experts in one call than transient_slots`.
Prefill misses go to the ring, and `env.example` says it plainly two hundred lines away from the
value: *"A prefill chunk touches ~370 of the 384 experts of every layer."* The ring must hold one
resolve call's worth of misses, so **400 — the code default — is correct, and the 8 in `env.example`
is only valid for the pruned all-resident profile where nothing misses at all.**

So §2's recommendation ("change `transient_slots`'s code default from 400 to 8") is **withdrawn**. It
came from simulating decode misses only, at an arena where prefill was never modelled. The sim also
modelled the ring as evicting FIFO when full; the real engine *raises*. That arm was not realisable.

### Correction 2: the arena slot is 14.45 MB, not 13.77

The 3-bit scales are a **disk** format. The in-memory arena still holds 8-bit scales, exactly as
designed — so 79.9 GB is 5,530 slots at 14,454,784 B, not 5,800 at 13,774,848. My projection quietly
used the disk figure for the arena. Corrected coverage at this arena: **36.0 % against FP4's 27.7 %**
at the same bytes, rather than the 46.3 % I wrote for a 98 GB arena.

### Correction 3: "the engine never opens the layer shards" is now tested, not assumed

40 of the 88 files the index names are absent, and the server reached `ready` and served a
generation. The deliberate choice to leave expert tensors pointing at missing files held up.

## End to end: 6.16 tok/s against a stored 2.68 — and why that number is not yet clean

`bench/bench.py --workload code --runs 2 --osl 512 --ignore-eos`, the same invocation behind the
stored baseline. 1 warmup + 2 measured: **5.49 / 6.28 / 6.04 tok/s**.

| | baseline `[stored, 2026-09-10]` | this run `[measured today]` |
|---|---|---|
| decode | 2.68 tok/s | **6.16** |
| TPOT | 374 ms | 162.5 |
| TTFT | 11,050 ms | 8,699 |
| expert hit rate | 0.830 | 0.896 |
| NVMe per run | 530.3 GB | **208.7** |
| accept_len | 3.03 | 3.59 |

**2.30×.** My derived estimate was +42 %; the measurement is +130 %, so I underestimated, and the
excess is in the part I cannot yet attribute.

**The confound, stated before the number gets quoted.** The baseline ran **FP4** experts
(`kernel triton-fp4`); this run is **CB3**. Three things changed at once:

1. the cache — no `fp4_to_cb3_v2` per miss, 13.77 MB reads in one extent;
2. coverage — a CB3 slot is smaller, so the same arena holds 5,530 experts instead of ~4,250, 36.0 %
   against 27.7 %;
3. **the arithmetic** — 3-bit experts are not 4-bit experts.

`accept_len` moving 3.03 → 3.59 is the tell: the cache cannot change acceptance, so (3) is doing
some of this. Whatever the split, **this is not a pure speed measurement — it is a different model.**

The clean A/B is unpruned FP4 on today's build, and it is **not runnable**: it needs the 296 GB of
layer shards deleted during the cache build, and only 141 GB is free. Two ways to get the
attribution without them, in increasing cost:

* **Match coverage, not memory.** Pin `ARENA_GB` so the CB3 arena holds exactly the baseline's 3,926
  slots (56.8 GB). That neutralises (2) with one env var and no download, leaving (1)+(3). If decode
  stays far above 2.68, coverage was not the story.
* **Build an FP4 cache in the same record format** (288.8 GB — does not fit today) to isolate (1)
  from (3) exactly.

Until one of those runs, the honest claim is narrow: **the unpruned full-router path went from 2.68
to 6.16 tok/s**, which is what a user gets, with the model's expert precision changed from 4-bit to
3-bit as part of the change.

## The confound, removed — and the quality gate CB3 was expected to fail

**Coverage-matched run.** Pinning `ARENA_GB=56.75` gives **exactly 3,926 CB3 slots**, the baseline's
own slot count and 25.6 % coverage, so the only remaining differences are the delivery and the
arithmetic:

| config | slots | coverage | hit rate | NVMe/run | accept_len | **decode** |
|---|---|---|---|---|---|---|
| FP4 baseline `[stored 2026-09-10]` | 3,926 | 25.6 % | 0.830 | 530.3 GB | 3.03 | **2.68** |
| CB3 cache, **matched coverage** | 3,926 | 25.6 % | 0.806 | 367.9 GB | 3.68 | **3.94** |
| CB3 cache, same memory | 5,530 | 36.0 % | 0.896 | 208.7 GB | 3.59 | **6.16** |

**The 2.30× splits into 1.47× at fixed coverage and a further 1.56× from coverage.** Two things worth
noting in the middle row: CB3's hit rate is slightly *worse* than FP4's at identical slot counts
(0.806 vs 0.830), so none of this is better caching; and NVMe falls 530.3 → 367.9 GB, a ratio of
1.44 against the 1.365 the smaller read predicts, so bytes account for essentially all of the
traffic drop.

What the 1.47× is *not* purely: `accept_len` is 3.68 against the baseline's 3.03. A cache cannot
change acceptance, so 3-bit expert arithmetic is contributing throughput here as well as bytes, and I
cannot separate those two without an FP4 cache in the same record format (288.8 GB, does not fit).

**Free-generation gate: 5/5 pass.**

| prompt | words | distinct ratio | max line repeat | finish |
|---|---|---|---|---|
| html page | 344 | 0.439 | 1 | length |
| python LRU + tests | 511 | 0.282 | 1 | length |
| English essay | 731 | 0.536 | 1 | **stop** |
| German prose | 396 | 0.667 | 1 | **stop** |
| reasoning | 73 | 0.644 | 1 | **stop** |

This is the gate CB3 **failed** at keep 40 % — distinct-token ratio 0.03, `<!DOCTYPE>` to the cap.
Unpruned CB3 passes it comfortably, three of five terminating naturally. That is independent support
for the repo's own root cause (`e28d0d7`: *"Expert pruning, not the engine, is what degenerates long
generations"*), from a configuration nobody had run: the 3-bit format was never the problem, the
truncated router was.

It is a structural gate, not a quality benchmark — it says the model is not degenerating, not that
3-bit costs nothing. NLL against FP4 and a real agent turn are still owed.

## The NLL half of that debt, paid (2026-09-14)

Measured: `ds41-measured-2026-09-13.md` §16. Paired teacher-forced comparison against the FP4 arm on
the same 17,704 tokens — **coding top-1 81.55 % → 80.67 %** (McNemar 239/138, z = 5.20, p = 2e-7);
general top-1 unchanged (46.88 → 46.82, z = 0.18). CB3's *lower* NLL is not a win: its predictive
entropy is higher in both categories, and temperature alone on the FP4 arm's own logits reaches
pooled NLL 1.656 at T = 1.40 versus CB3's 1.759 at T = 1. The requantization noise softens the
distribution, and NLL rewards that.

So the honest pitch for this cache is capacity and I/O, not free quality: 211.6 GB instead of
296 GB on disk and a 6–12× cheaper miss, for about **one point of coding top-1**. A real agent turn
is still owed; NLL is closed.
