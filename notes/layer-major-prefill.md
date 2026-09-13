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
