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
