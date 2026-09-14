# A three-stage layer scheduler for layer-major prefill

Design, 2026-09-14, from a long exchange with a third party. Nothing here is built except the
instrumentation. Every item exploits a dependency that is **already absent** — no prediction, no
overfetch, the same experts and the same arithmetic.

## Where the barrier is today

`ExpertStore.resolve` ends with

```python
list(self.pool.map(lambda ks: self._load_into_slot(*ks), to_load))
```

The `list()` is a full barrier: every miss in the layer must land before a single slot is returned.
`encoder_prefill_layer_major` then runs all chunk attention/routing, unions the routes, hits that
barrier, and only then applies the MoE. So the layer is strictly `route → wait → compute`.

## The target

```
A: attention/router over chunks, in order   (routes appear incrementally)
B: async expert delivery                    (resident immediately; NVMe → H2D → ready queues)
C: FFN execution                            (shared expert immediately; routed batches as they land)
```

A, B and C can all be live at once. The only hard barrier is each chunk's final hidden state at
layer L before that chunk advances to L+1.

## P0 — submit reads from early chunks, measured

The claim: after chunk 0 has routed, its expert ids are final, so their reads can start while
chunks 1..n are still doing attention. **Measured on the recorded route log** (`route-11k.jsonl`,
21 chunked encoder layers, 6 chunks each, union **362 experts/layer**):

| | fraction of the layer's entire expert set already known |
|---|---|
| after chunk 0 | **85.4 %** (worst layer 76.4 %) |
| after chunk 1 | 93.0 % (88.0 %) |
| after chunk 2 | 96.3 % (93.0 %) |
| after chunk 5 | 100 % |

One 2,048-token chunk at top-6 names five-sixths of everything the layer will want.

**Sizing.** ~153 of the 362 are misses → **~312 ms of delivery per layer** at the measured QD4
latency. Per-layer attention is roughly 1.2 s over 6 chunks ≈ 193 ms per chunk, so after chunk 0
routes there is about **966 ms of remaining attention to hide ~265 ms of reads under**.

**So incremental submission alone can plausibly hide all prefill expert delivery** — which would make
the resident-vs-streamed split largely redundant *for prefill*. The lead time exceeds the delivery
it has to cover. This is P0 and it is the cheapest of the three.

## P0/P1 — the shared expert has no routing dependency at all

`moe_apply` runs `routed = self.moe_fn(...)` then `shared = R.expert_ffn(y, w.sh_w1, w.sh_w2,
w.sh_w3, ...)`, serially. `expert_ffn` takes only `y`. It depends on no route, no slot, no NVMe and
no routed result, yet today it sits on the critical path *while routed experts are stalled*.

## P1 — resident routed experts, while misses stream

Covered in `ds41-measured-2026-09-13.md` §21's terms: the kernel already early-returns on `slot < 0`
(`tools/cb3_moe.py:130`, `:176`), so masking pending experts to −1 gives a partial pass with no
kernel change. And the refactor should be `moe_prepare` / `moe_launch_ready` / `moe_finish` rather
than calling `moe_forward` twice and adding, because

```python
P = T * K
parts = torch.empty((P, DIM), dtype=torch.float32, device=dev)
...
return parts.view(K, T, DIM).sum(dim=0).to(torch.bfloat16)
```

every `(token, expert)` pair owns its own fp32 row and the reduction is one sum plus one bf16 cast.
Filling rows in any order is therefore **bit-identical**, not merely close. Two `moe_forward` calls
added together would be two roundings.

**One correctness detail:** `parts` is `torch.empty`. That is safe today only because `slots` is
never negative in practice. The moment rows can go unlaunched, `moe_prepare` must allocate
**`torch.zeros`** or `moe_finish` must assert full coverage.

## Slot lifetime: per-slot completion, and exact Belady

Today `_transient_slot_for` is a pure FIFO ring and safety comes from one blanket barrier in the
loader — `stream.wait_stream(compute)`, i.e. wait for *everything* queued on the compute stream.
Replacing that with a CUDA event recorded after the launch that reads slot *s*, which the copy
stream waits on before overwriting *s*, is strictly finer and makes "computation done → slot vacant"
a tracked property.

It only holds if the loop is **expert-batch-outermost within a layer** (`for batch: for chunk:
compute; release batch`). Recycling while still iterating chunks would force a re-read and undo
layer-major's 5.56×.

**And when the fetch list exceeds the ring, this is the one place Belady is exact.** §5 measured
Belady against LRU at **1.76×** and dismissed it because it needs the entire future. Under
layer-major the entire future *of the layer* is known at resolve time — which expert each remaining
chunk still wants is simply read off the routes already computed. So eviction can pick the slot with
no further use, or the furthest next use, instead of FIFO. That turns a bound we measured and
shelved into an implementable policy, and it is what makes a **small** ring viable:

| ring | frees | LRU slots gained | coverage |
|---|---|---|---|
| 128 | 3.93 GB | +272 | +1.8 pp |
| **64** | **4.86 GB** | **+336** | **+2.2 pp** |

At the measured arena curve (0.226 tok/s per pp) a 64-slot ring is worth **≈ +0.5 tok/s of decode**,
arriving free with the prefill pipeline.

With 400 slots and a 362-expert union, a layer fits but two layers do not — so without per-slot
events the layer boundary stays a hard barrier. P0 and this are complementary, not alternatives.

## Ranked

| work | overlaps with | priority |
|---|---|---|
| reads discovered from early chunks | later-chunk attention/router | **P0** |
| shared expert | expert I/O + routed MoE | **P0/P1** |
| resident routed experts | missing-expert I/O | **P1** |
| early-chunk FFN | later-chunk attention | P1 |
| Engram row prefetch | previous layers | P2 — *low-priority I/O*: it shares the SSD with reads worth far more, and it is 0.64 s of a decode where `resolve()` is 84 s |
| HC FFN mix vs router gate | each other | P2 — both need only post-attention `h` |
| Q vs KV projections | each other | P2 |
| LM head vs DSpark seed | each other | P3 |

The P2/P3 GPU-only forks are deferred deliberately: on a part measured at **31–33 % GPU-busy** the
question is not whether they are independent but whether they contend, and that needs a profile we
do not have.

## Decode is explicitly out of scope

A decode step is 6 positions × 6 experts at an ~89 % hit rate — a handful of misses per layer.
Splitting a tiny MoE into extra launches could cost more than it hides. Build all of this in
`encoder_prefill_layer_major` only, and leave `moe()` alone until a measurement says otherwise.

## The per-layer split, measured (2026-09-14)

`DSV41_LM_PHASES=1`, 12,621-token prompt, 7 chunks, layer-major:

```
[layer-major phases] 21 encoder layers, 34.0s:
    attn+route 18.1s (53%)    resolve 6.3s (19%)    ffn 9.6s (28%)
```

Per layer: **attn+route 0.86 s, resolve 0.30 s, FFN 0.46 s.** My pre-measurement estimates had
delivery at ~312 ms/layer (right) and attention at ~1.2 s/layer (high — it is 0.86).

**This resizes P0.** After chunk 0 routes — about 123 ms into the layer — there are ~0.74 s of
remaining attention to hide 0.30 s of resolve under: **2.4× more lead time than the delivery needs**.
So P0 can plausibly hide *all* of resolve, and resolve is **19 %** of layer-major prefill. That is
34.0 s → ~27.7 s, about **1.23×** — the honest ceiling, and smaller than the oracle framing implied
because layer-major already took the large cut. The three stages together are worth roughly a
quarter of prefill, not a multiple of it.

One thing visible per layer and worth keeping: **L2 spends 1.36 s in attn+route against ~0.80 for
its neighbours.** Layer 2 is a `kv_source` layer, so that is the compressor. If attention ever
becomes the target, the four source layers are where it lives.

## Built so far: the instrumentation, and only that

`DSV41_LM_PHASES=1` reports per encoder layer the split between attention+route, resolve, and FFN.
It goes in **first**, before any of the three stages, because every claim above is about where time
goes *inside* a layer and `route_s` / `load_s` / `moe_s` are per-request totals that cannot see it.
§7, §10c and §19 each went wrong by measuring an aggregate and inferring a mechanism; this is the
instrument that stops the same thing happening to the scheduler. It costs a device sync per layer,
so it is a diagnostic and never a serving setting.

---

# The MTP / DSpark backlog — costed, and deferred

Five async boundaries on the drafter path, proposed 2026-09-14. All five verified against
`engine/fastdecode.py`; two of the proposer's own estimates moved on contact with the code.

## The ceiling, and why it is not permanent

§21 splits decode wall into **75–76 % NVMe wait, 18 % GPU-routing wait, 1.5 % bookkeeping**.
Everything else — attention, the MoE kernel, engram, LM head, draft, seeds — is the residual **~5 %**,
and *the entire MTP path lives inside it*.

| | MTP share of wall | ceiling | decode |
|---|---|---|---|
| today | 5.0 % | **1.05×** | 6.24 → 6.57 tok/s |
| after a loader pipeline running reads at the device rate | 8.3 % | **1.09×** | 10.39 → 11.34 tok/s |

So **1.05× is the current-profile ceiling, not a permanent one.** If the loader removes most of the
NVMe stall, decode wall falls to ~60 % of today (≈1.67× on its own) and the MTP share nearly doubles.
The correct standing wording is: *deferred until after loader pipelining; current maximum benefit
≈1.05×, but re-profile, because its share grows as expert-delivery stalls are removed.*

## Order, after two corrections from the code

1. **Greedy-only draft graph.** The one item that *removes* work rather than rescheduling two
   consumers of the same GB10 resources. `_draft()` computes, per draft position, `argmax`, a full
   `softmax` over V, `log`, a Gumbel add, a second `argmax` and a `d_probs` write — then
   `torch.where(temp > 0, …)` throws half of it away. Five positions × ~129k vocab, and the lean
   greedy verify path never reads `q`. Safest and largest of the set.
2. **LM head ∥ MTP seed preparation.** `_final()` runs the head, then `main_proj`, then three `wkv`
   + ring writes, serially; the two branches share only the final target hidden state. No arithmetic
   changes — a pure scheduling A/B.
3. **GPU-side acceptance → prelaunch the next draft**, removing the target → host → draft bubble.
   The current path stays for grammar and penalties, which modify logits outside the target graph.
4. **Pipeline seed0/1/2 against MTP0/1/2**, and benchmark the concatenated `[wkv0;wkv1;wkv2]` GEMM
   against three concurrent small ones — at 31–33 % GPU-busy one larger GEMM plausibly wins.
5. **`main_proj` early — demoted to near-worthless, and this is the correction that matters.** The
   proposal called it "possibly the largest MTP overlap", conditional on the last
   `dspark_target_layer_id` sitting well before L39. It does not: `dspark_target_layer_ids =
   [37, 38, 39]`, and the snapshot is taken at the **top** of each such layer
   (`fastdecode.py:321-323`, before `residual = h`). So `main_hidden` completes at the top of **L39**
   and the overlap window is exactly one layer-block — **~4 ms of a ~160 ms step, 2.5 %.** Do it only
   if it falls out of (4) for free, and **do not** attempt the split-`main_proj`-by-input-block
   variant: it changes GEMM shapes and FP accumulation order for a 2.5 % window.
6. **Routed ∥ shared inside the MTP layer.** `_draft()` serializes `out = moe_fn(...)` then
   `out += expert_ffn(...)`, both depending only on `y`, with a fully resident drafter arena so
   there is no NVMe correctness question — but both want the same tensor cores. Profile first.

## Reporting discipline for the scheduling rewrites

Adopted for the layer-major validation and everything after it: report **three separate axes**, never
one verdict.

```
performance   TTFT / prefill tok/s / NVMe GB
numerics      token agreement, or the first divergence
semantics     the rare-identifier gate
```

Conflating them is how a methodology artifact reads as a model regression: the identifier gate scored
10/14 on *both* arms of an A/B before it was fixed, and 14/14 after — the model never moved. A
scheduling change that preserves the semantic gate and delivers the I/O reduction is acceptable even
with minor token drift from changed execution order; that is a different decision from "bit-exact",
and the report should let someone make it.

---

# P0 MEASURED: early submission is ~1.03×, and the premise does not hold

`early-submit-abcd`, four arms × three separate processes on separate servers, prompt cache off,
12,624-token prompt. Medians of starts 2–3 (start 1 is the cold outlier in every arm).

| arm | TTFT | encoder total | attn+route | resolve | ffn |
|---|---|---|---|---|---|
| **A** `off` | 57.0 s | 51.8 s | 18.9 s | **15.1 s** | 17.9 s |
| **B** `synconly` | 57.8 s | 52.7 s | 21.1 s | 13.6 s | 18.0 s |
| **C** `all` | **55.2 s** | **50.2 s** | 25.4 s | **6.7 s** | 18.1 s |
| **D** `chunk0` | 55.7 s | 50.7 s | 21.6 s | 10.9 s | 18.1 s |

## The attribution

| | attn+route | encoder total |
|---|---|---|
| **B − A** — the per-chunk route-id D2H barrier and bookkeeping, alone | **+2.2 s** | +0.8 s |
| **C − B** — genuine overlap contention, on top of the barriers | **+4.3 s** | −2.4 s |
| **D − A** — one barrier, ~85 % submitted early | +2.8 s | −1.1 s |

So the external review's hypothesis was **partly right**: the seven new barriers really do cost
**2.2 s** of the 6.5 s that all-chunk submission added to `attn+route`. But they are only a third of
it. The other **4.3 s is real contention** — the reads and the attention genuinely compete on this
part, which has one memory system for both.

## And `chunk0` is worse than `all`, which was the opposite of the prediction

The argument for `chunk0` was that chunk 0 alone names 85.4 % of a layer's expert set, so one
barrier should buy nearly all the overlap. It does buy most of it — but the 15 % tail it leaves
behind costs more than the six barriers it saves: resolve is **10.9 s against `all`'s 6.7 s**, and
the totals come out 50.7 vs 50.2 s. The barriers were never the dominant term.

## The verdict, and it is a negative

**Encoder total 51.8 → 50.2 s. TTFT 57.0 → 55.2 s. About 1.03×** — and my earlier single-run figure
of 1.07× was optimistic, as single runs on this test tend to be.

The mechanism is not in doubt: resolve falls **15.1 → 6.7 s**, 56 % of the stall genuinely removed.
It is the *premise* that fails. Hiding an SSD read under attention on GB10 costs about as much as it
saves, because both want the same unified memory. **"Overlap the I/O with compute" is not free here,
and this is the first direct measurement of that on this box.**

## What it means for the rest of the plan

This is the cheapest member of the loader-pipeline family and the one with no quality question
attached — and it returns 3 %. That should be priced into the rest before anyone spends a day on
them:

* the **75 % NVMe wait** in decode is real, but the ~2.2× headroom implied by 3.2 GB/s of an
  available 5.0–6.8 assumed overlapping is free. On this evidence it is not;
* **stage C** (FFN executing as expert batches land) overlaps *more* I/O with *more* compute, so it
  inherits this contention rather than escaping it;
* the **shared-expert overlap** is the exception worth keeping: it overlaps compute with compute, not
  with I/O, so this result does not bear on it.

**Disposition:** `DSV41_EARLY_SUBMIT` stays in the tree, off by default, modes intact. It is a
working instrument for asking this question again after anything that changes the memory-system
picture. It is not worth switching on for 3 %.
