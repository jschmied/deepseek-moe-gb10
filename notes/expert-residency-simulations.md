# Expert residency: what the routing traces actually say

Findings ds-03…ds-12, 2026-09-13. All simulated offline against **real routing traces**, held out,
before any engine work. Method per the user: capture the access pattern with the router **unmasked**,
then price strategies in simulation and only then build.

**Read this caveat first.** Every number below is measured on **Qwen3.8-Flash-Next** (512 experts,
top-10, 48 MoE layers), not DeepSeek-V4.1 (384, top-6, 40 layers). Qwen is the harder covering
problem, and it is the model we can actually run today. The *directions* are what transfer; the
levels do not. `tools/*_sim.py` take any npz of `{name}__routed [tokens, layers, topk]` +
`{name}__meta [n_prompt, n_gen]`, which is also the shape 0xBakeer's `tools/expert_trace.py` emits,
so DS4.1 traces drop straight in.

## How the trace was captured — no patching needed

vLLM has this natively: `enable_return_routed_experts=True` makes `CompletionOutput.routed_experts`
an `[seq_len, layer, topk]` array covering **prompt and generated tokens concatenated**
(`routed_experts_prompt_start`'s docstring: "Default 0 returns routing for all prompt tokens"), so
the prefill/decode split is an index at `prompt_len`. Capture binds on our flashinfer_cutlass NVFP4
path — `RoutedExpertsManager CPU buffer: 0.16 GB (slots=169344, layers=48, top_k=10, uint16)`.

Data: `notes/data/qwen-routing-short.npz` (10 requests, 22–53-token instructions) and
`qwen-routing-long.npz` (6 requests, 888–4,445 tokens of real CUDA/Python/prose/docs/YAML/JSON).

---

## ds-03 — a prompt-derived overlay beats a static union, most at the worst case

The user's proposal: replace one global keep-set with `core + request-specific overlay`, chosen from
the full router's picks during prefill. Held out: core from the *other* requests, overlay from **only
the held-out request's prompt**, scored on **decode-phase** selections.

| | short prompts (n=10) | long prompts (n=6) |
|---|---|---|
| static 44 % union | 49.5 – 81.4 % | 73.1 – 86.8 % |
| **38 % core + 6 % overlay** | **68.0 – 83.4 %** | **79.3 – 86.3 %** |

Same total budget (225 vs 226 slots/layer). The gain is **at the floor** — +18.5 pp short, +6.2 pp
long — and the ceiling barely moves. That is exactly the "unknown workload silently degrades"
failure mode, and it is the case the overlay fixes.

**Do not read the short/long difference as "longer prompts help less."** The two sets differ in *two*
ways: the long set is also task-homogeneous (all "read this document and answer"), which favours a
static union. Confounded; not a clean length comparison.

## ds-04 — 44 % is inherited, and the curve has no knee there

| total residency | coverage (uniform, held-out) |
|---|---|
| 20 % | 33.9 – 57.1 % |
| 30 % | 52.6 – 72.8 % |
| **44 %** | **73.1 – 86.8 %** |
| 60 % | 88.2 – 95.7 % |

30→44 % buys ~15 points, 44→60 % another ~13. **44 % is where 98 GB of arena landed, not a property
of the routing.** Every additional point still pays.

## ds-05 — per-layer reallocation is nearly worthless, though the spread is real

Concentration varies a lot by depth: at a 44 % keep the top experts cover **68.6 % of picks in the
worst layer and 96.6 % in the best**, and early layers (0–11) are the diffuse ones. Uniform
allocation ignores all of it. But moving slots to where they help most barely pays:

| total | uniform | global (their alternative) | waterfill (greedy-optimal) |
|---|---|---|---|
| 44 % | 73.1 – 86.8 % | 73.8 – 88.0 % | 73.6 – 88.1 % |

**~0.5–1.3 points.** This independently reproduces their measured `global ≈ uniform` result and
explains it: the per-layer curves are similarly shaped, so a slot moved is a slot lost elsewhere.
**The layer axis is close to exhausted — do not spend effort there.**

## ds-06 — the static core is the wrong design; a plain LRU of the same size wins

All arms at **identical 44 % total residency**:

| arm | coverage | misses **or** loads / token |
|---|---|---|
| static 44 % frozen | 73.1 – 86.8 % | 63.2 – 129.2 *(masked, no traffic)* |
| core 43 % + LRU 1 % | 79.6 – 89.9 % | 48.3 – 98.1 |
| core 22 % + LRU 22 % | 91.9 – 94.0 % | 28.7 – 39.1 |
| **adaptive 0 % + LRU 44 %** | **93.4 – 94.8 %** | **24.8 – 31.5** |

Monotonic: the more of the budget is adaptive, the better. **~20 points over the frozen keep-set**,
and the static core contributes nothing at any mix. Their `TRANSIENT_SLOTS=8` is two orders of
magnitude too small to register — the 43 %+1 % arm barely moves.

**The two right-hand columns are different currencies and must not be compared directly.** Static's
63–129 are picks *masked away*: zero traffic, silent quality cost. Adaptive's 25–32 are **real
fetches** — ×14.45 MB ≈ **360–460 MB/token**, ~13–16 tok/s at 6 GB/s, i.e. *below* their current
17–37. So this is a **quality/speed trade, not a free win**. What the simulation settles is that if
the router is to be unmasked, LRU dominates the static design, and the frozen core is the part to
drop rather than tune.

## ds-07 — eviction policy is irrelevant at small capacity; capacity is everything

At core 43 % + 5 dynamic slots/layer:

| policy | coverage | loads/token |
|---|---|---|
| ring (FIFO, their design) | 79.1 – 89.8 % | 49.2 – 100.2 |
| LRU | 79.6 – 89.9 % | 48.4 – 98.1 |
| LFU | 74.9 – 88.0 % | 57.5 – 120.7 |
| **Belady (optimal, unimplementable)** | **79.1 – 89.8 %** | 49.2 – 100.2 |

**LRU ties Belady.** With ~10 misses/token/layer against 5 slots the cache thrashes, so eviction
order cannot matter. LFU is actively worse — it clings to early-frequent entries. **Choose capacity,
not policy.**

*Correction recorded:* an earlier pass labelled its dynamic set a "transient ring" but implemented
`set.pop()` — arbitrary eviction that could drop *core* entries too. It scored 91.6–93.4 %, which was
not a ring result at all but an accidental measurement of a **fully adaptive cache**. Mislabelled,
but it is what led to ds-06.

## ds-08 — their I/O is well built; the gap is queue depth, not read shape

From `engine/experts.py`: `O_DIRECT preadv` (no page-cache pollution — correct here, since on GB10
page cache *is* GPU memory), **two coalesced runs per expert instead of six** ("six separate preadv
costs six O_DIRECT round trips"), pinned aligned staging, two thread pools.

* **Overread is negligible**: ≤4095 B per run edge × 2 runs ≈ **8 KB on 14.45 MB = 0.06 %**, and
  expert sizes are exact page multiples (14,454,784 = 3529×4096; 18,800,640 = 4590×4096).
* **Not optimal**: decode achieves **2.5 GB/s** against a device that gives 5.6 single-threaded;
  prefill 3.8. Their `experts.py:77` names the cause — "a decode step misses only about one expert
  per layer, so nothing else is in flight".
* **Channel change will not help.** `libcufile.so` is installed and loads, but **`nvidia_fs` is not
  loaded**, so GPUDirect Storage would fall back to a host bounce — what they already do. GDS solves
  a PCIe-crossing problem this box does not have.

> **CORRECTED 2026-09-13** — two claims in this finding were wrong, both from numbers quoted without
> re-measuring. (a) The staging→slot bounce is **not** "2 % of the cost at 273 GB/s": a pinned H2D
> copy runs at **59.4 GB/s**, so at 852 MB of misses per step it is ~14 ms, not noise. (b) The
> `read_chunk_mb=4` reasoning was backwards. Chunking does not raise effective queue depth, because
> a thread issues its chunks **serially**; measured, 4 MiB chunks cost **41 % at one expert in flight
> and 14 % at two** against reading the whole expert in one `pread`, and **two in flight already
> reach the device ceiling** of ~6.8 GB/s. The lever is a *larger* read, not more of them.
> Full measurements: [gb10-arena-io-measured.md](gb10-arena-io-measured.md).

**Interaction worth noting:** ds-06's adaptive design needs 25–32 loads/token across 48 layers —
*more* concurrency than today's ~1 miss/layer. Higher miss rates are *better* for the device given
the QD curve, so the residency policy and the I/O ceiling push the same way.

---

## ds-09 — the arena SPLIT is the lever, not the keep-set's contents

`tools/prefill_warm_sim.py`, leave-one-out over the long trace (6 requests, prefill 888–4,445
tokens). Every arm has the **same total capacity**; only the static/dynamic split and the static
core's contents differ. LRU on the dynamic slots throughout.

Coverage %, mean over the held-out requests:

| arena | dyn share | global core | pure LRU (no core) | oracle core |
|---|---|---|---|---|
| 44 % | 5 %  | 88.8 | 91.4 | 97.1 |
| 44 % | 10 % | 90.4 | 91.4 | 96.8 |
| 44 % | 25 % | **92.1** | 91.4 | 95.9 |
| 44 % | 50 % | **92.4** | 91.4 | 94.4 |
| 44 % | 100 % | 91.4 | 91.4 | 91.4 |
| 20 % | 5 %  | 53.8 | **78.8** | 81.1 |
| 20 % | 10 % | 62.2 | **78.8** | 82.2 |
| 20 % | 25 % | 69.4 | **78.8** | 82.1 |
| 20 % | 50 % | 74.1 | **78.8** | 81.0 |
| 20 % | 100 % | 78.8 | 78.8 | 78.8 |

Two things fall out, and they point opposite ways depending on how tight the arena is.

* **At a roomy 44 % the split has an interior optimum at 25–50 % dynamic** (92.1–92.4 %), worth
  **+1.7…+2.0 pp over a 10 % transient allowance** and +3.6 over 5 %. 0xBakeer ships
  `TRANSIENT_SLOTS=8`, well inside the losing region. This is a constant, not a redesign.
* **At a tight 20 % the static core is actively harmful.** Pure LRU over the whole arena scores
  78.8 % against the frozen core's 53.8–74.1 % — **+4.7 to +25 pp** — and no split recovers it. The
  frozen keep-set only earns its slots when there is enough capacity that the globally-hot experts
  really are this request's hot experts too.

DS4.1 at its 98 GB arena sits at 44 % of experts, but that figure is a memory artefact on a steep
curve (ds-04); anything that squeezes the arena moves it toward the regime where the keep-set is a
liability. Ship the split as a runtime knob, not a compile-time constant.

## ds-10 — warming the cache from the request's own prefill: real, small, and not worth it

The idea was free domain adaptation: rank the initial arena on the routing observed during *this*
request's prefill, which is known before a single token is decoded and is drawn from exactly the
domain the decode will be in.

It works, and it is too small to build:

| trace | prefill tokens | arena / dyn | global | prefill-warmed | delta |
|---|---|---|---|---|---|
| long | 888–4,445 | 44 % / 10 % | 90.4 | 91.2 | **+0.8 pp** |
| long | 888–4,445 | 20 % / 5 %  | 53.8 | 58.5 | **+4.7 pp** |
| short | 22–53    | 44 % / 10 % | 85.6 | 84.0 | **−1.7 pp** |

And it needs a lot of prompt to pay. Truncating the window at cap 20 %/dyn 5 %, where the effect is
largest: 64 tokens **−0.9 pp**, 256 tokens +0.7, 1,024 tokens +4.7, all tokens +4.7. So the whole
effect arrives between 256 and 1,024 prompt tokens, and short prompts are actively worse off.

The decisive comparison is not against `global` though — it is against **doing nothing but enlarging
the dynamic fraction**, which is free. Pure LRU beats prefill-warming at 20 % (78.8 vs 58.5) and
matches it at 44 % (91.4 vs 91.2). **Closed: prefill-warming is dominated by ds-09 at every point
measured.** Blending prefill and global ranks (25/50/75 %) was also tested and lands between the
two parents every time — no interaction to exploit.

## ds-11 — where the remaining headroom actually is

The oracle arm ranks the static core on the held-out request's own **decode** histogram — knowledge
no serving system has. It brackets what any smarter *initialisation* could ever win:

* At 44 % / 5 % dynamic: **97.1 %** against the best real arm's 92.4 — **~5 pp of headroom**.
* At 20 %: 81.1 % against pure LRU's 78.8 — **~2 pp**.

So under memory pressure adaptivity has already collected nearly everything on the table, and
further work on *which experts to hold* is worth about two points. With a roomy arena there are five
points left, and they are only reachable by **predicting** the decode distribution — which is the
drafter-driven-prefetch question, not a cache-policy question. That is the honest case for spending
effort there, and it is much smaller than the ~20 points ds-06 already banked.

---

## What this changes

1. **Drop the frozen keep-set** in favour of an adaptive cache of the same size (ds-06). Biggest
   single effect measured, ~20 points.
2. **Do not tune per-layer allocation** (ds-05) or the eviction policy (ds-07). Both are ~1 point.
3. **Do not treat 44 % as a design point** (ds-04) — it is a memory artefact on a steep curve.
4. **Stop splitting the expert read** (ds-08 as corrected): `read_chunk_mb=4` costs 41 % at the
   concurrency decode actually runs at. Then batch a layer's misses — the target is only two in
   flight, which is where the device already saturates. Coalescing and alignment are already right.
5. The prompt overlay (ds-03) still helps at the worst case, but **a plain LRU of equal size beats it
   outright**, so the sticky-session variant is the version worth keeping — adaptivity is the active
   ingredient, not prompt-derived prediction.

6. **Ship the static/dynamic split as a knob** (ds-09) and **do not build prefill-warming**
   (ds-10); the remaining initialisation headroom is ~2 pp under pressure, ~5 pp with a roomy arena
   (ds-11), and only prediction can reach it.

**Everything above is unvalidated on DS4.1.** The next step is a trace from
`tools/expert_trace.py` with `prune_keep` unset — it streams one 7.4 GB layer shard at a time, needs
neither the 510 GB resident nor the 189 GB engram shards (rows come via HTTP ranges), and is
`--resume`-able as shards land. Then re-run these same simulators against it.
