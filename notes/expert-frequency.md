# How skewed is DS4.1's expert usage, really?

From our own **unmasked** trace (`notes/data/ds41-routing-unmasked.npz`): 57 requests, 40 layers,
384 experts, **3,183,360 decode routing decisions**. 0xBakeer ships `coverage.png` and
`layer_hist.png`, but those came from a trace taken through a *masked* router — this is what the
model wants, not what it was permitted. `tools/expert_frequency.py`; full output in
`notes/data/expert-frequency-20260913.txt`.

## The keep-set curve

Share of routing captured by the top N experts **of each layer**, mean over 40 layers:

| N | % of experts | mean coverage | worst layer | best layer |
|---|---|---|---|---|
| 32 | 8.3 % | 44.6 % | 28.0 % | 54.8 % |
| 64 | 16.7 % | 61.4 % | 44.0 % | 69.9 % |
| 96 | 25.0 % | 72.0 % | 56.4 % | 78.6 % |
| 128 | 33.3 % | 79.6 % | 66.3 % | 85.2 % |
| **154** (sayyidfareed's K154) | 40.1 % | **84.4 %** | **73.2 %** | 89.0 % |
| **169** (0xBakeer's keep 0.44) | 44.0 % | **86.7 %** | 76.7 % | 90.7 % |
| 192 | 50.0 % | 89.9 % | 81.6 % | 93.1 % |
| 256 | 66.7 % | 96.1 % | 92.4 % | 97.6 % |
| 320 | 83.3 % | 99.2 % | 98.3 % | 99.7 % |

**It is skewed, but far less than a keep-set needs.** Per-layer entropy is **7.43 bits of the 8.58 a
uniform router would give** — the distribution is much closer to flat than to Zipfian. For contrast,
the Engram tables are genuinely Zipfian (top 1M of 384M rows = 62 % of lookups, per the field
survey); experts are not.

The tail structure is real though: the hottest expert of a layer runs **45.8× the median** (9.0× to
95.5× across layers). So there is a hot core — it is just that the core is small and everything
below it is a long, fat middle rather than a cliff.

## What that says about pruning

Keeping 40 % of experts leaves **15.6 % of routing decisions outside the set on an average layer and
26.8 % on the worst one**. That is the cost a pruned deployment pays *before* any question of which
domain the keep-set was ranked on, and it is consistent with what we measured directly against
sayyidfareed's shipped K154: 9.0 % of coding picks masked, **60.6 % of general picks masked**.

The spread across layers is the part nobody seems to exploit. At N=154 the best layer is at 89.0 %
and the worst at 73.2 %; a uniform per-layer budget spends the same slots on a layer that needs far
fewer and one that needs far more. Note this is **not** the per-layer *arena* allocation ds-05
measured and killed (~1 point) — that was residency of a cache; this is the routable set.

## Coding against general

Top-25 % sets per layer, Jaccard: **0.067–0.185, mean 0.117**. At the wider keep 0.44 the same trace
gave 0.280 (ds-09), so the disagreement sharpens as the set narrows, which is what you would expect
and which makes a single merged keep-set worse the tighter the budget gets.

## The curve itself

Layer 20, all 384 experts sorted by use, log₂ scale:

```
    0- 95 |@@@@@%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#########################################********************|
   96-191 |*********************************************+++++++++++++++++++++++++++++++++++++++++++++++++++|
  192-287 |+++++++++++++++++++++++++++++++++++++++=======================================================--|
  288-383 |---------------------------------:::::::::::::::::::::................                          |
  '@' = 3,249 uses, ' ' = 0
```

No cliff anywhere — a short hot head, then a very long shoulder. That shape is the argument for
**caching** experts (which adapts to whatever this request needs) over **pruning** them (which must
commit in advance), and it is why ds-06's adaptive LRU beat the frozen keep-set by ~20 points at
equal size.

*A rendered plot needs matplotlib, which is not in the serving venv; not installing it while the
night queue is running through that same interpreter.*

## Pruning gate, first pass (2026-09-14) — active, passing at 600 tokens, and under-powered

`prune-keep-cliff`, five arms, `gen_gate.py` at 600 tokens, temperature 0.6, CB3 + dense FP4 + fp8
head. **Every arm passed 5/5, including `PRUNE_KEEP=0.40`, which is the arm the stored record says
failed** (`native-cb3-expert-cache.md`: distinct-token ratio 0.03, `<!DOCTYPE>` to the cap).

**First question: was pruning even on?** Yes, and I checked rather than assumed after initially
concluding the opposite from the arena lines. `/health` reports `prune_keep = 0.4` on a direct
pruned start, and the five arms produce **different outputs** (keep-0.40 essay 478 words at 0.617
distinct; keep-0.60 essay 432 at 0.664; python 260 / 220 / 156 / 150 words across 0.40 / 0.44 / 0.60
/ 0.80). So the router really is restricted and the run is not void.

**Second question, and it is why the result cannot yet be believed: 3 of 5 prompts hit the token cap
in every arm.** The stored failure is described as degenerating *long* generations. A gate that
truncates at 600 tokens stops before the failure mode it is looking for has room to appear. Re-queued
at **2000 tokens**.

**Third thing, worth recording separately: keep 0.40 did not go all-resident.** The keep set is 6,144
experts = 88.8 GB against an auto arena of 83.0 GB / 5,744 slots, so the engine pruned the router and
kept streaming — `resident_expert_pct` stayed at 37.4 % in every arm. That is the same arithmetic as
pruned-all-resident and a completely different speed, so none of these arms says anything about the
337 tok/s prefill or 36.6 tok/s decode that raspy135 reports for that mode. **Only keep ≤ 0.44 can be
all-resident at all** on this box (0.44 needs 97.7 GB, 0.55 needs 122.1, 0.70 needs 155.4), and even
0.44 needs the arena pinned above its auto value.

**Harness note.** The first attempt failed all five arms with "server never came up", and the real
error was in `logs/server.log`: *"pruned mode needs the per-layer trace npz files next to
trace_stats"*. `.env` pointed at `trace-full-20260910`, whose `trace/` the streaming trace driver
deletes layer by layer as it runs. Stats regenerated for `trace-unmasked-20260913`, which still has
all 40. **The keep set therefore comes from a different trace than the stored failure used**, which
is a second live hypothesis for the disagreement and should be stated whenever this result is quoted.
