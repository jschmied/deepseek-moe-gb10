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

## `IO_THREADS` under overlap: flat, and that is the evidence

`iothreads-under-overlap`, 4/6/8/12 workers with `EARLY_SUBMIT=all`, three separate processes each,
medians of starts 2–3:

| io_threads | TTFT | prefill | NVMe | encoder | resolve |
|---|---|---|---|---|---|
| **4** | **54.2 s** | 233.2 tok/s | 75.1 GB | 49.6 s | 9.8 s |
| 6 | 55.1 | 228.9 | 75.6 | 50.6 | 9.3 |
| 8 | 55.6 | 227.2 | 76.2 | 50.9 | 9.0 |
| 12 | 55.1 | 228.9 | 76.0 | 50.2 | **6.7 s** |

**Tripling the workers moves `resolve` by 3.1 s and the wall by 1.4 s — a 2.7 % spread, with 4
marginally best.** If NVMe bandwidth were the binding constraint, 4 workers would be clearly worse:
we measured 4.14 ms per load at queue depth 1 against 2.04 at depth 4. It is not worse. The extra
concurrency is not reaching the device.

## Why the drive and the GPU are both idle: we gate our own copies

`_load_into_slot` and `_load_into_slot_cached` both do, on the worker thread, **after** the NVMe read
has landed:

```python
with torch.cuda.stream(stream):
    stream.wait_stream(compute)      # wait for EVERYTHING queued on compute right now
    load_slot(..., non_blocking=True)
stream.synchronize()                 # park this worker until the copy finishes
```

`wait_stream(compute)` is a blanket barrier. Under `EARLY_SUBMIT` the main thread is queuing
attention for chunks 1..n, so a read landing during chunk 3's attention makes its H2D wait for
chunks 0–3's attention to **complete** — and then `stream.synchronize()` parks the worker. Twelve
workers park behind attention, the pool stops issuing reads, and the drive goes idle. That is the
idle NVMe and the flat `io_threads` curve, from one cause.

**It was free in the design it was written for.** Chunk-major `resolve()` blocked anyway, so nothing
was on the compute stream while loads ran; the blanket wait cost nothing. `EARLY_SUBMIT` is the
first caller for which compute *is* running, and the conservatism becomes the thing that prevents
the overlap. The comment states the real requirement — *"the previous layer's MoE kernel may still
be reading the slot we are about to overwrite"* — which is a **per-slot** dependency implemented as
a global one.

**So the per-slot completion events are not a ring-size optimisation. They are the enabler.** They
were filed under "shrink the transient ring, ≈ +0.5 tok/s"; they are actually the fix for the 4.3 s.

**And the 1.03 × verdict needs softening.** "The premise fails" was wrong. The honest statement is
that **the premise is untested**: the implementation gated its own overlap behind a barrier inherited
from the blocking design. Whether hiding SSD reads under attention works on GB10 is still open, and
one diagnostic arm with that wait removed settles it.

## Three changes that only work together

| | alone | together |
|---|---|---|
| per-slot completion events | the last expert arrives slightly sooner | the H2D stops waiting on unrelated attention |
| traffic-ordered submission | irrelevant — the MoE needs the *last* expert | the most valuable experts land in the first waves |
| partial MoE (`moe_launch_ready`) | still gated behind attention | compute starts on wave 1 instead of wave 13 |

Measured support for the middle row: within a layer, ordering the **misses** by traffic puts
**28.5 %** of the layer's blocked token-expert pairs in the first 32 reads, against **12.9 %** for the
arbitrary expert-id order `np.unique` currently produces (51.1 % vs 26 % at 64). Notable because the
misses *are* the cold tail — the hot experts are resident — so skew was not guaranteed to survive there.

## The `wait_stream` hypothesis is REFUTED — and it reinstates the contention verdict

`waitstream-diag`, `EARLY_SUBMIT=all`, three separate processes per arm, medians of starts 2–3.
Both arms **PASS token equality** (21/21 identical), so the unsafe arm is a result, not corruption.

| | TTFT | encoder | attn+route | resolve | **FFN** |
|---|---|---|---|---|---|
| `wait_stream(compute)` **present** | 54.4 s | **49.7 s** | 24.8 s | 6.9 s | **17.9 s** |
| `wait_stream(compute)` **removed** | 61.5 s | **56.9 s** | 25.7 s | 7.5 s | **23.7 s** |
| | | **+14.5 %** | +3.8 % | +8.0 % | **+32.0 %** |

**Removing the barrier makes it worse, and the cost lands in the FFN**, not in `attn+route` where my
hypothesis put it. That is not a shape the "the barrier gates our overlap" story predicts at all.

### What it actually shows

With the barrier, an expert's H2D is serialised *after* whatever compute was queued, so the copy and
the kernels each get the memory system to themselves. Remove it and the copies genuinely overlap the
MoE kernels — **and the MoE kernels slow by 32 %.** The overlap is real; it is simply not free,
because on this part the CPU and GPU share one memory system and an 13.77 MB H2D per expert is
memory traffic that the kernels also need.

**So "GPU 31–33 % busy" and "NVMe at 2.3–3.0 of 5.0–6.8 GB/s" do not mean there is headroom to
overlap into.** SM occupancy is not memory-system occupancy. A kernel can sit at a third of the SMs
and still be taking most of the bandwidth, and this is the first direct evidence of it on GB10:
the same kernels, the same data, 32 % slower purely from copies running alongside.

### The correction I owe

After the `io_threads` result I wrote that the 1.03 × verdict should be softened — that *"the premise
is untested, because the implementation gated its own overlap behind an inherited barrier"*. That was
wrong. The barrier is **load-bearing**: it is worth more than the overlap it prevents, by 7.2 s of
49.7. The original reading of the A/B/C/D run — that hiding an SSD read under compute costs about as
much as it saves — stands, and now has a direct measurement behind it rather than a subtraction.

Two lessons, both about me rather than the engine:

* I found a plausible mechanism in the code and promoted it to an explanation before testing it. The
  code evidence was real (the barrier *is* a per-slot dependency implemented globally); the inference
  from it was not.
* The flat `io_threads` curve is equally consistent with "workers parked behind a barrier" and
  "memory system saturated". I read it as the first because I had just found the barrier.

### What this costs the plan

The loader-pipeline family is **capped by memory bandwidth, not by scheduling**. Stage C (FFN
executing as expert batches land) overlaps *more* copies with *more* kernels, so it inherits this
penalty rather than escaping it — on this evidence it could be net negative.

Still standing, because neither overlaps a copy with a kernel:

* **shared-expert overlap** — compute against compute;
* **traffic-ordered submission** — only useful with partial MoE, which is now doubtful;
* **layer-major itself**, which *removes* 5.4× of the copies rather than rescheduling them. That is
  the lever that worked, and the reason is now clear: on a shared memory system the only reliable win
  is moving fewer bytes.

### Correction, same evening: it is not bandwidth, and I named a mechanism again without isolating it

The section above concludes *"capped by memory bandwidth, not by scheduling"*. The arithmetic does
not support that, and I should have done it before writing the word.

Per encoder layer, during the FFN phase (12,624-token prompt, 21 layers, 17.9 s of FFN):

| | |
|---|---|
| expert weights read from the arena | 5.23 GB |
| `parts` `[T·K, 5120]` fp32 | 1.55 GB |
| `h` `[T·K, 2304]` bf16 | 0.35 GB |
| `y` | 0.13 GB |
| **total, over 0.85 s per layer** | **7.26 GB → ~9 GB/s** |

The overlapping H2D copies add 4.99 GB per layer, ~6 GB/s. **Combined that is ~6 % of this box's
measured 240 GB/s stream rate.** Nothing here is bandwidth-saturated, so "memory bandwidth" is the
wrong label for a 32 % FFN slowdown.

**What it could be, none of it isolated:**

* **L2 pollution** — each copy streams 13.77 MB through the cache hierarchy, which is large against
  any plausible L2, and the MoE kernel's working set is evicted between tiles.
* **Memory-controller latency interference** rather than throughput — concurrent copy-engine and SM
  traffic raising each other's access latency well below saturation.
* **Driver-level contention** — twelve worker threads each doing `stream` operations and a
  `synchronize`, delaying the main thread's kernel launches. Weak support: `io_threads=4` was
  marginally the *best* arm.

**That this is unresolved changes the conclusion.** "Capped by bandwidth" would be a hardware wall.
L2 pollution or launch contention are implementation problems with implementation answers — a
non-temporal or chunked copy path, fewer workers, a different staging route. So the loader-pipeline
family is **not** closed; it is blocked on a mechanism we have not identified, and identifying it
needs a profiler rather than another A/B.

**And this is the second time this evening I promoted a plausible mechanism to an explanation.**
First `wait_stream`, refuted by its own diagnostic; then "bandwidth", refuted by five minutes of
arithmetic I could have done first. The rule I keep re-deriving: a mechanism is a hypothesis until
something measures *it*, not its consequence. The consequence here — copies slow the FFN by 32 % —
is solid and reproduces across three processes. Everything I have said about *why* is not.
