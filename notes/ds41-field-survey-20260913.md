# The DeepSeek-V4.1 field, four days after release — and where it contradicts us

Survey 2026-09-13. The model shipped 2026-09-10; in four days the field went from ~3 efforts to
~40 GitHub repos and ~97 HF derivatives. Three of our four standing beliefs are contradicted by
other people's measurements, and two things we listed as untried have been done by several groups.

## The item to act on first — and our fork is clear of it

**HF discussion [#12](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/discussions/12)**
(unmerged): DeepSeek's own reference `Indexer.forward` publishes `shared_attn.index_k` only inside
`if self.owns_k and latent is not None`. On an incomplete compression group the source does not
republish but the forward still reads the slot — which holds **a later source layer's cache from the
previous forward**. Hits source layers 2, 8, 14. Reproduced independently twice: malaiwah
([#41](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/discussions/41)) measures max abs logit
diff **1.991 and argmax agreement 34/48** unmodified against **4.029e-5 and 48/48** fixed, on
prefill-then-decode — i.e. every agent turn. PipeNetwork found the same from the MLX side.

**Checked ours: structurally immune.** The reference keeps **one shared slot**. 0xBakeer's rewrite
keeps a **per-layer** cache (`Caches.ik[L]`, `engine/model.py:195`) and rebinds `st["ik"] = c.ik[L]`
in layer order inside a `st` dict that is created fresh per forward (`fastdecode.py:447/460/577`,
`"ik": None`). A layer that does not publish therefore inherits the most recent *earlier* publisher
of the same forward, which is the intended sharing — never a later layer's stale cache. The
definitive test is still malaiwah's prefill-vs-one-shot logit comparison; the structural argument is
strong enough to deprioritise it.

## Where the field contradicts us

**"Unpruned streaming on one Spark is NVMe-bandwidth-bound" — contradicted, and we had already half
corrected it.** JigSawPT on one RTX 5090 measures the drive delivering **10.04 GB/s at the queue
depth decode actually produces** while the engine extracts **4.33 — 43 %**, with only **3.78 requests
in flight per layer**, because a layer's expert reads cannot be issued until that layer's router has
run. They published an explicit retraction of their own bandwidth hypothesis. atbender's M1 Mac mini:
reads are **18 % of wall**. This matches our own profile exactly — GPU 31–33 % busy, NVMe at 2.3–3.0
of an available 5.0–6.8 GB/s — and the right phrase is **latency- and serialization-bound**, with the
per-layer router barrier as the named mechanism.

**"12.7 GB/token caps a 128 GB box at 19–22 tok/s" — the traffic figure is confirmed, the causal
story is not.** atbender's zero-cache run reads **13.04 GB per decode token**, 3 % off ours. But
nobody is near a bandwidth cap: the 5090's all-resident ceiling is **21.3 tok/s on ~1.79 TB/s**
(bandwidth would predict ~140), an M3 Ultra with everything resident gets 6.5, and PipeNetwork
measured the batch-1 expert GEMM at **5.2 % of the card's bandwidth** against llama.cpp's 72 %. So
19–22 is a correct upper bound that current kernels do not reach, and our own resident estimate
landing near it is a coincidence of a slow kernel meeting a slow bus.

**"3-bit codebook is fine unpruned" — supported for codebooks, refuted for affine, and the
interaction we assert is still untested by anyone.** PipeNetwork: uniform 3-bit **affine** experts
with the **full** router → ppl 1.4e7 and degenerate loops. But three independent 3-bit-class
**codebook** builds are clean: bot-lab-21 EXL3 3.51 bpw (**NLL Δ −0.0032 ± 0.0055, HumanEval/+
0.951/0.921**), drowzeys TR3-Hybrid 3.22 bpw (**KL 0.032, top-1 0.984**). Their own mechanism
explains the split — FP4's levels are non-uniform, affine's are evenly spaced, "which no group size
fixes". That is the argument for CB3, and it now has outside support. **Nobody has measured 3-bit ×
pruned together**, so our specific claim stands untested by others.

## Two things we called untried that are not

**Engram compression: five groups, three days.** HLWQ-Q4 (Hadamard + 16 Lloyd-Max centroids)
203 → **101 GB**, and it is the only one **served**: prose NLL improves, **code NLL worsens
0.1086 → 0.1233 and top-1 drops 97.5 → 93.8 %**. LibertAI NVFP4 → 97.6 GiB, never evaluated
end-to-end. PipeNetwork's affine ladder: **6-bit is free** (divergence 0.1945 vs FP8's 0.1948),
4-bit costs +7.3 %. bot-lab-21 measured **fp4 Engram rows cost no NLL** (Δ −0.003 ± 0.003). The two
that measured **disagree**, and there is no task eval anywhere.

**But Engram is a capacity problem, not a speed one** — five independent confirmations that it is
**0.6–3.4 % of a decode step**, and that the real cost is ~16× page amplification (256 B rows into
4 KiB pages), not the 203 GB. The untried lever is bot-lab-21's Zipfian measurement: **top 1M of
384M rows = 62 % of lookups, top 5M = 89 %, top 100M = 92.7 % held out.**

## Worth stealing

* **nktlabs' engram-disk-prestage**: hash eagerly in `prepare_inputs`, one `preadv` into a pinned
  buffer, one H2D, so CUDA graphs stay on. **26.1 → 80.2 tok/s c=1** on 2× RTX PRO 6000. Nine files.
* **drowzeys' TR3-Hybrid allocation**: trellis the tail at K=3, leave the **64 highest-error experts
  per layer at native MXFP4**. Better fidelity than uniform 3.5 bpw at *lower* average bits.
* **JigSawPT's batched Engram prefetch**: addresses are functions of token ids, so one batched OS
  prefetch call needs no predictor — **10.437 → 0.917 ms/token, bit-exact**.
* Two negative results that save us time: perfect-oracle expert prefetch pays only at **5 tokens of
  lookahead** (−19/−7/−16 % at 1–4), and **LRU vs Belady is a 0 % margin** at their working set —
  both matching what we measured independently.

## Operational warnings

* **vLLM's merged Engram `cpu_offload` frees zero bytes on GB10** — CPU and GPU share one pool. The
  upstream fix everyone else gets does nothing here; only true on-disk works.
* **CUDA graphs are a correctness hazard on sm_121 for this model**, found three ways. CiphemonJY's
  step probe caught **17 NaN layers in 997 steps** while a greedy gate, vision/tools and a 60-request
  promo gate all PASSED — and graphs were slower in aggregate anyway.
* **"A greedy output gate cannot detect a feature being silently disabled"**: with Engram rows forced
  to zero, their token-exact greedy gate still returned PASS. Our five-prompt gate has the same blind
  spot.
* **Never benchmark this model on counting prompts** — the drafter accepts them wholesale and decode
  inflates ~2.6× (19.2 prose vs 49.4 counting on one box).
* A **GB10 "slow state"** exists: decode-shaped GEMV at 70 vs 230 GB/s, invisible to `nvidia-smi`,
  triggered by a long idle. Not reproduced on our firmware (BIOS 5.36_0ACUM027 / 580.173.02) in 9
  runs across 8 nodes, but it taints single-cell c=1 figures elsewhere.
* **"QSA" is not this model's terminology** — 0 hits in 57 READMEs. It is CSA2 plus a Hierarchical
  Sparse Indexer. Searching for QSA finds nothing.

## Where we stand

We and sayyidfareed remain **the only single-Spark efforts at usable speed**; the only other is
llama.cpp Q2_K at 2.3–2.8 tok/s. 0xBakeer is still at `8b68fdde` with no commits since 09-12, all
nine forks are zero commits ahead on their default branch, and **the entire issue tracker and its one
PR are ours, with no maintainer reply**.
