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

## 0xBakeer v0.5.0 (2026-09-14): frequency ranking is the wrong axis, and our gate cannot see the failure

He moved from `8b68fdd` to `45a0caf` "Release 0.5.0" in a day, and the shape of the change matters
more to us than the numbers. The recipe:

```
DSV41_PRUNE_SOURCE=saliency    # rank by contribution, not frequency
DSV41_PRUNE_RANK=maxmin        # give slots to the worst-served topic
PRUNE_KEEP=0.40 / 0.36         # 154 or 139 experts a layer
```

**1. Contribution ranking beats frequency ranking, and the failure mode is one our gate scores as a
pass.** Frequency-ranked keep-0.40 *"corrupts rare tokens at subword boundaries —* `clearTimeout`
*written* `cleartimeout`*,* `OSError` *as* `oenerror`*"*. Contribution ranking —
`gate_weight(t,e) · ‖expert_e(x_t)‖₂` summed over the tokens routed to the expert — removes it:
*"Not one rare token corrupts under contribution ranking, where frequency wrote* `ttimerid` *and*
`color-s-s-mode`*"*.

**Everything we rank is frequency-ranked**, including `coverage.json`'s `counts` and therefore every
keep-set in `prune-keep-cliff`. And — this is the part that invalidates the reading, not just the
ranking — **`gen_gate.py` cannot detect this failure at all.** A corrupted identifier is still a
distinct token, still repeats no line, still lets the model finish. Distinct-token ratio and line
repeats are blind to it by construction. That is why our keep-0.40 arm "passed 5/5" while his
frequency-ranked keep-0.40 was corrupting identifiers. **The two results do not contradict; ours was
measured with an instrument that cannot see his failure.**

`tools/token_integrity.py` is the missing instrument: ask for specific rare identifiers
(`clearTimeout`, `XMLHttpRequest`, `itertools.groupby`, `pthread_mutex_trylock`,
`__builtin_expect`, `os.posix_fadvise`) at temperature 0 and check they come back byte-exact, with a
separate "mangled" class for an identifier that is present but case- or separator-corrupted.

**2. Substitution beats dropping — measured, and it corrects me.** Earlier today I wrote that
substituting a wrong expert is *"arguably the more violent of the two"* compared with dropping.
Measured the other way: drop mode scores **0 of 6** on Frontend at keep 0.36 with thinking on,
against substitution's 7 of 10. Dropping zeros the non-resident experts and `norm_topk_prob`
renormalises over roughly four survivors of six, with a shared-expert-only fallthrough on about one
token in twenty. Substitution won decisively.

**3. One keep-set for everything does not work, which is our Jaccard result from the other side.**
*"Everything"* (37 topics) scores **4 of 39**; *"Programming, broadly"* (15 topics) **4 of 16**. Seven
narrow profiles score **39 of 61** on contribution ranking against 34 of 61 on frequency. Our
own measurement said the same thing structurally: per-layer top-25 % sets have Jaccard **0.067–0.185**
across categories, so a merged keep-set is worst exactly where the budget is tightest.

**4. Even at its best this is 39 of 61 — 64 %.** It is a real quality cost, not a free lunch, and it
is scored on his own gates rather than a public benchmark.

**5. The limitation that matters most for us: the 0.40 all-resident configuration cannot hold a
filled context.** *"No request in any of these gates prefilled anything like 262,144 tokens; the
longest prompt in the whole suite is 58 words."* And a 195k-token prefill *"took MemAvailable to
0.8 GB and the watchdog killed the engine after 582 s"*.

So his configuration and ours are **not competitors for the same job**. His is short prompts, fast
decode, one narrow task at a time. Ours is the faithful full router at long context — where we
measured 27,200 tokens at 261.3 s cold and **~22 s cached**, and where his config dies. That is the
honest framing for both, and it means the 36.6 tok/s is not a number we are losing to.

## The cliff, measured at 2000 tokens — and where the profile-free band actually sits

**`prune-keep-cliff`, second run.** The 600-token pass was under-powered exactly as suspected; at
2000 tokens the failure appears and it reproduces the stored one. Frequency-ranked keep-sets from
`trace-unmasked-20260913`, CB3 + dense FP4 + fp8 head, the `html` prompt is the one that breaks:

| `PRUNE_KEEP` | experts/layer | html words | distinct ratio | line rep | verdict |
|---|---|---|---|---|---|
| **0.40** | 154 | 971 | **0.056** | 1 | **DEGENERATE** |
| **0.44** | 169 | 852 | **0.095** | 4 | **DEGENERATE** |
| 0.60 | 230 | 552 | 0.471 | 2 | pass 5/5 |
| 0.80 | 307 | 582 | 0.510 | 2 | pass 5/5 |

*(the unpruned control was still running when this was written; its stored json is from the killed
600-token attempt and must not be read as this run's row.)*

**The cliff is between 0.44 and 0.60**, and 0.056/0.095 against the stored 0.03 is the same failure,
found with a gate three times longer. That is the whole reason the first pass looked clean.

### Does one universal keep-set remove the need for task profiles?

`tools/universal_keepset.py`, same trace, both rankings. Per size: Jaccard between the two
categories' own top-N sets, and what fraction of each category's **gate-weight mass** survives a
single universal top-N set built on pooled traffic.

| N/layer | % | Jaccard c-vs-g | shared of each | universal keeps: coding | general |
|---|---|---|---|---|---|
| 96 | 25 % | 0.129 | 22.9 % | 77.3 % | 55.8 % |
| 139 | 36 % | 0.216 | 35.5 % | 85.5 % | 71.4 % |
| **154** | 40 % | 0.248 | 39.8 % | 87.7 % | **75.7 %** (worst layer 74.0) |
| 169 | 44 % | 0.291 | 45.0 % | 89.6 % | 79.5 % |
| **230** | 60 % | 0.467 | 63.7 % | 95.4 % | **91.4 %** (worst layer 89.2) |
| 307 | 80 % | 0.714 | 83.3 % | 99.1 % | 98.5 % |

**Three results.**

1. **The hypothesis is right.** The hot experts are strongly domain-dependent (J = 0.129 at the top
   25 %) and a moderate set is mostly shared (J = 0.714 at 80 %). A universal set at 230/layer
   retains 95.4 % / 91.4 % of each category's gate-weight mass against 87.7 % / **75.7 %** at 154.
   The jump is where it was predicted to be.
2. **The `1 − J` reading is wrong and the correction matters.** For equal-size sets the fraction of
   one set also in the other is `2J/(1+J)`, not `1 − J`. At J = 0.70 that is **82.4 % shared, 17.6 %
   replaced** — not 30 %. The table carries the corrected column.
3. **Gate-weight ranking is NOT a usable saliency proxy.** Ranking by summed gate weight instead of
   count changes almost nothing: J 0.248 → 0.265 at N = 154, and identical retention (87.7 % / 75.7 %
   → 87.7 % / 77.6 %). So whatever contribution ranking buys upstream comes from the
   `‖expert_e(x_t)‖₂` term, **not** from the gate weight — and that term is not in our trace. Testing
   contribution ranking honestly needs a new trace pass that records expert output norms.

### Where this leaves the all-resident idea

The two measurements meet, and they meet badly for CB3:

| | fits in ~83–88 GB? | survives the gate? |
|---|---|---|
| 154/layer (keep 0.40) | CB3 **89.0 GB** — marginal | **no** |
| 169/layer (keep 0.44) | CB3 97.7 GB — no | **no** |
| 230/layer (keep 0.60) | CB3 133.0 GB — no | yes |

**The band that works cannot be resident, and the band that fits degenerates.** So pruned
all-resident is closed for us *in CB3*.

It is **not** closed in CB2: 220/layer is **87.9 GB** and 230/layer is 91.9 GB, against ~88 GB
reachable with the dense/head savings. 220/layer is 57.3 % keep — above the 0.44 that degenerates and
just under the 0.60 that passes. That is the one configuration on this box where a profile-free,
all-resident, full-quality-band design is arithmetically possible, and it rests entirely on what
2-bit experts cost. `cb2-nll` is queued and is now the gating measurement for the whole direction.

## Per-layer bit sensitivity: **no signal in weight space** (2026-09-14)

`cb-perlayer-sensitivity`, all 40 layers, hottest 24 experts each, activation-weighted by summed
gate usage from the unmasked trace. Relative requantization error against the FP4 weights the
checkpoint actually holds:

| | min | max | spread |
|---|---|---|---|
| err @ 3 bits | 0.0450 | 0.0475 | **1.06×** |
| err @ 2 bits | 0.1369 | 0.1458 | **1.06×** |
| ratio err2/err3 | 2.985 | 3.068 | — |
| **marginal cost of the third bit** (err2 − err3) | 0.0919 | 0.0983 | **1.07×** |

**Every layer costs the same.** The third bit is worth +0.0919 to +0.0983 of relative error
*everywhere*, a 7 % spread end to end, and the only mild outlier is layer 39 (+0.0983, and it is the
last layer). There is no cheap layer to raid and no expensive layer to protect.

**So the idea does not survive on this metric.** Moving bits between layers cannot help if the bits
cost the same in every layer. `ds41-measured-2026-09-13.md` §20's uniform CB2 result stands as the
price of 2-bit experts, and the all-resident direction stays closed.

**And the control is interesting.** MiaAI-Lab's EXL3 build puts K=2 on layers **18–22** specifically.
Ranked by marginal cost here, those layers come out 2nd, 5th, 15th, 10th and 22nd of 40 — i.e.
**scattered, not clustered at the cheap end**. If their choice were explained by weight-space error
they would occupy the first five ranks. They do not.

**What that does and does not license.** It does not say their choice is wrong — it says *this
proxy cannot see whatever they used*. Trellis quantizers in that family select on output or
curvature sensitivity (Hessian/gradient-weighted), not on raw weight distance, and those are
different quantities: a layer whose weights quantize cleanly can still sit where the network is most
sensitive to the perturbation. Measuring that needs per-layer output sensitivity — one forward pass
per layer with a perturbation, not a weight comparison — which is a real experiment and not this one.

**Cost of finding out: ~100 minutes and no trace.** The value was in *not* spending a 150-minute
paired trace on an assignment chosen by a metric that turns out to be flat. Recorded as a negative
with the data in `notes/data/perlayer-quant-error.json`.
