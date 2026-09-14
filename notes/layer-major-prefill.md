# Layer-major prefill: the plan, what verified, and where the prize actually is

User proposal, 2026-09-13. Recorded because it is the largest remaining prefill idea and because
checking it turned up two things that change how to run it.

## What verified against the code

| claim | status |
|---|---|
| prefill is chunk-major: each chunk runs the whole layer stack | **confirmed** — `v41_engine.py:680` `for s in range(0, P, MAX_CHUNK): m.forward(..., encoder_only=True)` |
| there is an encoder/decoder split; only L0..candidate-source run per chunk | **confirmed** — `model.py:637` `last = a.candidate_source_layer if encoder_only else n_layers - 1` |
| CB3 prefill converts touched experts back to FP4 on every call | **confirmed** — `cb3_moe.py:732`, and the scratch arena is reused but its *contents* are per call |
| `lease_s` / `h2d_s` / `load_wait_s` are already exported | **confirmed** — `v41_engine.py:662` (`load_wait_s` aliases `load_s`) |

And the reason the unpack exists is documented and measured by them: *"at decode there is one block per
expert and CB3 wins (0.79× the FP4 time); at prefill shapes there are one to twelve … and CB3 measured
**2.3–6.7× the FP4 time**"*. So the unpack is not an oversight, it is a deliberate trade — which is
why the A/B below matters before any rewrite.

## Provenance: the cited numbers are 0xBakeer's, not ours

The proposal quotes 291/326/369 tok/s at 512/1024/2048 chunks, CB3 up 41.0 ms and down 22.5 ms of a
116.6 ms step, and the `num_warps=8` miscompile. **None of those are our measurements.** They are in
**0xBakeer's `NOTES.md`** (lines 1088, 1589, 1997) and `RESULTS.md`, which makes them real but
*stored*, and taken on a configuration that differs from ours in four ways that all bear on these
kernels: **`max_seq` 8192 against our 32768**, **`kernel triton-fp4` against our `triton-cb3`**,
**spec off against on**, and **27.9 % resident against 35.8 %**.

This matters because four survey claims died today from exactly this — building on a stored number
without re-measuring. Our own prefill-chunk sweep is queued and not yet run, and **we have never
profiled our kernel mix at all**. So the plan's *ordering* is sound; its *magnitudes* are not ours yet.

## Where the prize actually is: NVMe, not the unpack

The proposal frames the win as amortising the CB3→FP4 unpack across chunks. Priced against our own
measurements, that is the smaller half:

| | |
|---|---|
| 11,366-token prompt at `MAX_CHUNK=2048` | 5.5 chunks, each running L0..L20 |
| measured prefill NVMe | **341 GB at 2.6 GB/s = 131 s** |
| redundant unpack traffic (21 × 370 × 33.25 MB × 4.5) | 1.18 TB ≈ **6 s** of the 131 |
| if each layer loaded its experts **once** rather than once per chunk | ~61 GB → **~24 s** |

**The unpack is worth about 6 s; not re-reading the experts is worth about 107.** Same rewrite
delivers both, but the thing to instrument first is expert loads per chunk, not `unpack_s`.

One detail suggests the design already anticipated this: the transient ring is **400 slots** and a
2048-token chunk touches **~370 experts of a layer**. The ring is sized almost exactly for one
layer's population — which is what layer-major needs and what chunk-major cannot use.

## Revised order

**0. `DSV41_CB3_PREFILL=direct` vs `fp4` — free, and it is P0.5.** The switch already exists
(`cb3_moe.py:397`), explicitly "for measurement". The proposal budgets a day for this A/B; it is one
env var and one bench run. Queued as `sweep-cb3-prefill`.

**1. The two-chunk / one-layer diagnostic**, as proposed: run two chunks through one MoE layer,
retain the FP4 scratch and the expert batches between them, measure. But instrument **expert loads
and `load_wait_s`**, not only `unpack_s` — per the arithmetic above.

**2. Layer-major transpose** if 1 holds up. Two BF16 `[P,4,5120]` buffers are ~0.57 GB at 7k tokens
and ~2.6 GB at 32k, against ~7.2 GB for a layer's 384 FP4 experts, and we have ~20 GB free.

Then the rest of the proposal's order stands as written: event-based miss staging, SM121 CB3
autotune plus down/reduce fusion, shared-expert overlap, delayed mHC seam fusion, Engram early H2D,
and the candidate-only indexer last — that one only pays above ~100k context, and our own
`sweep-max-seq` will say what the current full-width scan costs us at 32k first.

---

## P0.5, measured — but it answers a narrower question than the plan asked

`DSV41_CB3_PREFILL` A/B, median of 3 runs per arm, fresh server each, same 62-token prompt:

| arm | TTFT | TPOT | decode | expert hit | arena slots |
|---|---|---|---|---|---|
| `fp4` — unpack CB3→FP4, run the FP4 kernel (**default**) | **8.18 s** | 160.0 ms | 6.25 tok/s | 0.8966 | 5,669 |
| `direct` — run the CB3 kernel at prefill shapes | **10.98 s** | 158.8 | 6.30 | 0.9008 | 5,714 |

**The shipped default is right: unpacking wins TTFT by 34 %.** Decode is unchanged (6.25 vs 6.30,
inside the spread), which is the expected signature of a prefill-only switch and a useful check that
the knob does what it says. The result is if anything understated — the `direct` arm happened to get
a slightly *larger* arena and a *better* hit rate and still lost prefill by 2.8 s.

This confirms 0xBakeer's stored "2.3–6.7× the FP4 time at prefill shapes" **directionally** on our
configuration. Our 1.34× is much smaller than their kernel-level range because TTFT also contains
attention, Engram and the dense path; the MoE kernel is only part of it.

**What it does not answer.** The plan's P0.5 was `CB3 → unpack → FP4` against **native FP4 read from
disk** → FP4. I measured `CB3 → unpack → FP4` against **CB3 run directly**. Those are different
pairs: mine settles *whether the unpack earns its keep*, not *whether a second on-disk FP4 source
would beat it*. That one still needs the 296 GB of FP4 experts we deleted, so it stays open — and it
is now lower priority, because:

**The result strengthens the layer-major case rather than replacing it.** Prefill must unpack, the
unpack is per call, and layer-major is exactly what amortises a per-call cost across chunks. Had
`direct` won there would have been no unpack left to amortise and the transpose would have been worth
only its NVMe half.

---

## The two-chunk diagnostic: unpack-once is 33 % SLOWER, and the reason matters

4 chunks × 2048 tokens through one layer, 192 distinct experts, resident throughout, median of 3:

| arm | ms | GB unpacked |
|---|---|---|
| A: re-unpack the population per chunk (**what the engine does today**) | **316.2** | 14.44 |
| B: unpack once into a full-population scratch, reuse across chunks | **421.7** | 3.61 |

**Arm B moves a quarter of the bytes and takes a third longer.** So the redundant unpack is not the
cost — and the hypothesis that motivated the whole transpose does not survive its own first test.

**The likely mechanism, and it is the interesting part.** `moe_forward_prefill` unpacks in batches of
32 into a **0.6 GB** scratch and runs the FP4 kernel over that batch immediately. The unpack is
therefore doubling as a **prefetch**: it writes each expert into a small buffer microseconds before
the kernel reads it. Arm B unpacks all 192 into a **3.6 GB** arena once, so from chunk 2 onward the
kernel reads memory that has long gone cold. Removing the "redundant" work removed the locality that
made the kernel fast.

**What this experiment does NOT settle, and I should have separated it.** Arm B changed two things at
once: it stopped re-unpacking *and* it abandoned 32-expert batching. A cleaner arm B keeps the
batching — iterating the same 32-expert windows over slices of a pre-unpacked arena — and would say
whether the loss is the cold read or the batch size. Queued as `prefill-reuse-probe-batched`. Until
that runs, the honest statement is **"unpack once with one big scratch is worse"**, not "reuse is
worthless".

**What it does not touch at all**: the NVMe half. This probe ran on resident experts by design, so
the 341 GB → ~61 GB of experts *re-read from disk per chunk* is untouched and remains the larger
prize — by our own arithmetic ~107 s of a 131 s prefill against the unpack's ~6 s. The transpose's
case now rests entirely on that half, which is the half the probe could not measure.

Worth noting the shape of the surprise: this is the second time today that a "redundant" operation
turned out to be load-bearing. The first was `DSV41_CB3_PREFILL`, where running the CB3 kernel
directly — and skipping the unpack entirely — cost 34 % of TTFT.

---

## The chunk sweep, on a prompt long enough to see it — and this is the case for the transpose

`sweep-prefill-chunk-long`, 11,344-token prompt, 3 arms:

| `DSV41_PREFILL_CHUNK` | chunks | TTFT | NVMe | MB/prompt-token | GB per chunk |
|---|---|---|---|---|---|
| 1024 | 12 | 174.5 s | **530.8 GB** | 46.8 | 44.2 |
| 2048 (default) | 6 | **118.5 s** | 331.1 | 29.2 | 55.2 |
| 8192 | 2 | 137.7 s | **139.2 GB** | 12.3 | 69.6 |

**Two things, and they point opposite ways.**

**NVMe is near-linear in chunk *count*** — 44/55/70 GB per chunk across a 12× range of chunk size.
That is the re-read made visible: **74 % of prefill I/O at chunk 1024 is the same experts being read
again**, and it falls to 58 % less at 8192. The premise behind the layer-major transpose is now
measured rather than derived.

**But TTFT is non-monotonic and the default is already optimal.** 8192 reads 58 % less than 2048 and
is **16 % slower**. So the compute cost of a large chunk overtakes the I/O it saves somewhere just
past 2048, and the naive lever — "use bigger chunks" — is exhausted at the shipped value. 0xBakeer's
`MAX_CHUNK = 2048` is sitting on the optimum.

**Which is precisely the argument for layer-major.** The transpose gets the I/O saving by *reordering*
rather than *enlarging*: each layer's expert population is read once per prompt while the chunk stays
at the compute-optimal 2048. Extrapolating the GB-per-chunk line to one effective chunk gives ~70 GB
against the default's 331 — **~4.8× less prefill I/O at no compute penalty**.

That reframes the reuse probe's negative result. The probe measured the **unpack** half on resident
experts and found unpack-once 33 % slower; this measures the **I/O** half and finds 74 % of it
redundant. Both halves of the original proposal are now measured, and they disagree: the unpack is
load-bearing and should stay batched, the re-reads are waste and are worth ~4.8×. A transpose that
keeps the 32-expert unpack batching *inside* a layer-major loop takes the second without losing the
first.

*(The decode column, 1.43–1.53 tok/s, is 64-token generations again — not comparable to the 6.24
reference, same caveat as §14 of the serving profile.)*

---

## The oracle, before the rewrite (2026-09-14)

Third-party suggestion, and it is the right order of work: measure the ceiling from real routes
before writing the transpose. Expert identity is `(layer, expert)` and there is no useful
cross-layer reuse, so *"hold this layer's experts until every chunk has consumed them"* is very
nearly the **exact** upper bound for layer-major — not a loose global Belady bound. That is what
makes this oracle worth trusting.

**Recording.** `engine/experts.py` now writes one JSON line per `resolve()` call under
`DSV41_ROUTE_LOG=<path>` (off by default, a few hundred short lines for a whole prompt):
`{"i", "L", "pf", "uniq", "miss"}` — what this (layer, chunk) wanted, and which of those were not
resident. Everything the oracles need is in that.

**Replaying.** `tools/prefill_io_oracle.py` produces, from one log:

| | what it is |
|---|---|
| current | Σ over calls of \|miss\| — what the engine reads today |
| layer-union | Σ over **layers** of \|∪ miss\| — each missing expert read once per layer |
| layer-union, cold arena | Σ over **layers** of \|∪ uniq\| — ignores LRU warmth, so it does not move between runs (and sits *above* the residency-aware number) |

plus a delivery-time bound at the **measured** CB3 cache latencies (4.14 / 2.23 / 2.04 ms per load
at queue depth 1 / 2 / 4), which separates *the gain from not re-reading* (the bytes column) from
*the gain from queue depth* (QD1 → QD4 on the same bytes). With `--compute-s` it also prints
`max(compute, delivery)` per layer — flagged in the output as an assumption, because the engine does
not time layers separately and the compute is spread evenly.

**One thing the oracle makes clear that the byte count alone does not.** The layer-union figure is
reached by *reordering*, not by residency: each expert batch is read once, used by every chunk, then
discarded. So layer-major does **not** require holding a layer's 384 experts (5.55 GB of CB3) at
once — it needs the existing 32-expert working set (0.46 GB CB3 + 0.60 GB unpacked FP4). That is
also what keeps the 32-expert unpack batching intact, which the reuse probe showed is load-bearing.

**And the state it does have to hold is not a constraint.** `dim = 5120`, `hc_mult = 4`, so a
layer's token state is `[T, 4, 5120]` bf16: 0.46 GB at 11.3k tokens, 0.91 GB at 22k, **1.34 GB at the
full 32k**, plus a megabyte of `pre_mix`. Against ~20 GB free. The design question is the routing
and gather bookkeeping, not memory.

Status: tool written and its arithmetic checked on a synthetic log; the real recording waits for the
GPU (`longctx-profile` is running). Numbers to follow.

## The oracle, measured (2026-09-14) — **3.88×**, and it is prefill-only

`route-log-oracle`: one 11,344-token prompt through the engine with `DSV41_ROUTE_LOG` on, replayed
offline. 145 prefill `resolve()` calls, 40 layers, 6 chunks per layer.

| | loads | GB | vs current |
|---|---|---|---|
| current (engine today) | 23,798 | **327.8** | 1.00× |
| layer-union oracle | 6,131 | **84.5** | **3.88×** |
| layer-union, cold arena | 10,487 | 144.5 | 2.27× |

**Re-read fraction: 74.2 %** — the chunk sweep's derived 74 % reproduced exactly, now from routes
instead of from a byte extrapolation. And the oracle's "current" of 327.8 GB lands on the engine's
own measured 331.1 GB of NVMe for the same request, which is the cross-check that says the recorder
is seeing what the drive is doing.

**Correction to this note's own extrapolation.** The chunk-sweep section above projected "~70 GB
against the default's 331 — ~4.8× less prefill I/O". The exact answer from real routes is **84.5 GB
and 3.88×**. The extrapolation was 21 % optimistic, because it ran the GB-per-chunk line down to one
effective chunk and that line is not straight at the end.

**Delivery-time bound** at the measured CB3 latencies:

| | QD1 | QD2 | QD4 |
|---|---|---|---|
| current | 98.5 s | 53.1 s | **48.5 s** |
| layer-union | 25.4 s | 13.7 s | **12.5 s** |

Against a measured TTFT of **118.7 s**. So at queue depth 4, delivery is 41 % of the wall and the
transpose takes it to 11 %: **TTFT 118.7 → ~82.7 s, 1.44×**, if compute is unchanged and the engine
already achieves QD4. It does not — §13 measured NVMe at 2.3–3.0 GB/s of an available 5.0–6.8, and
this request ran at 331 GB / 118.7 s = **2.79 GB/s** — so the real win is larger and the ceiling is
set by the loader pipeline. The two levers compose rather than compete: the transpose removes 243 GB
of reads, the pipeline decides how much of what remains is hidden.

**Decode is untouched, and that is a result.**

| | loads | GB |
|---|---|---|
| current | 1,126 | 15.5 |
| layer-union | 1,126 | 15.5 |

**Re-read fraction 0.0 %.** One token per step means one chunk per layer, so there is nothing to
transpose. Layer-major is a prefill lever only — which is the right place for it, since prefill is
79 % of an agent turn, but it means none of this touches the 6.24 tok/s headline.

**Verdict: build it.** 3.88× on the dominant phase, bit-exact in intent (the same weights, a
different visiting order), and it needs no residency — each 32-expert batch is read once, used by
every chunk, discarded.

## At 27k the oracle is **9.38×** — and the floor does not move (2026-09-14)

`route-log-oracle-27k`, one 27,200-token prompt, same recorder and replay:

| | loads | GB | vs current |
|---|---|---|---|
| current (engine today) | 56,096 | **772.7** | 1.00× |
| layer-union oracle | 5,978 | **82.3** | **9.38×** |
| layer-union, cold arena | 9,437 | 130.0 | 5.94× |

**Re-read fraction 89.3 %**, against 74.2 % at 11.3k. And the cross-check holds: the oracle's
"current" 772.7 GB lands on the engine's own measured **787.6 GB** of NVMe for the same request
(98 %), so the recorder is still seeing what the drive does.

**The result that matters is not 9.38× — it is that the layer-union column barely moved.**

| prompt | chunks | current | layer-union | ratio |
|---|---|---|---|---|
| 11,344 tok | 6 | 327.8 GB | **84.5 GB** | 3.88× |
| 27,200 tok | 14 | 772.7 GB | **82.3 GB** | 9.38× |

Current I/O grows 2.36× with the prompt; the layer-union floor **falls slightly**, 84.5 → 82.3 GB.
That is what the structure predicts and it is worth stating plainly: the floor is bounded by the
number of *distinct* `(layer, expert)` pairs a prompt touches, which saturates near the full
15,360 — not by chunk count. So **layer-major makes prefill expert I/O essentially
context-independent**, and the speedup grows linearly with prompt length because the thing it
replaces does.

Extrapolating the same way the engine's own curve runs (TTFT ≈ tokens/104 + 15 s), the ratio at
32k would be ~11×, and the floor still ~82 GB.

**Delivery bound** at the measured CB3 latencies, against a measured TTFT of **280.5 s**:

| | QD1 | QD2 | QD4 |
|---|---|---|---|
| current | 232.2 s | 125.1 s | **114.4 s** |
| layer-union | 24.7 s | 13.3 s | **12.2 s** |

So at QD4 delivery is 41 % of the wall — the same share as at 11.3k — and the transpose takes it to
4 %: **TTFT 280.5 → ~178 s, 1.57×, on delivery alone**, with the rest belonging to the loader
pipeline.

*(Log shape check: 305 prefill `resolve()` calls, which is 21 encoder layers × 14 chunks plus one
replay pass over layers 21–39, not the 40 × 14 a naive reading would expect. CED runs the encoder
half chunked and the decoder half once over the window tail, so the decoder layers have no re-read
to remove — the 89.3 % is all in the encoder. The decode half of the log came back empty at 16
output tokens; the recorder's buffer is flushed on close and the job stops the server, so short
tails can be lost. Not load-bearing here.)*
