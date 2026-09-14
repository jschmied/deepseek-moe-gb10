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
