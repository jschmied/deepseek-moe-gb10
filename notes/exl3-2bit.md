# EXL3 at ~2 bits: error per bit and decode speed on one GB10 (2026-10-11)

Question (user): we never tried EXL3; what tok/s would a ~2-bit EXL3 reach, and how much quality does it cost?
Nothing was downloaded whole: every tensor came by HTTP byte range (`tools/hf_range.py`).

## Error per bit, same experts (`tools/exl3_vs_fp4.py`, `data/exl3-vs-fp4-20261011.json`)

60 tensors: layers 2/10/20/30/38, the 3 hottest experts of the unmasked trace plus 1 random cold one, w1/w2/w3.
rel = ||W_q - W||^2 / ||W||^2 against DeepSeek's FP4 weights, float64; EXL3 decoded by TensorFold's reference
decoder; CB2/CB3 by `engine/codebook_sim.py`. **Proxy**: identity Hessian. EXL3's LDLQ minimises error against
activations, so on real outputs EXL3 should look better than this, not worse.

| format | bits/weight | median rel |
| --- | --- | --- |
| CB3 (ours) | 3.25 | 0.046 |
| EXL3 K3 (all packs pooled) | 3.0 | 0.022-0.027 |
| CB2 (ours) | 2.25 | 0.138 |
| EXL3 K2 | 2.0 | 0.089-0.105 |
| EXL3 K1 | 1.0 | 0.40 |
| EXL3 K4 | 4.0 | 0.005-0.007 |

EXL3 halves CB3's error at fewer bits, and a uniform 2-bit EXL3 lands between CB3 and CB2 (~0.10 vs 0.046 / 0.138).

Allocations of the published packs (mean K over the sample): SAGE 1.59 gives K1 to 34 of 60 tensors, layers 30/38
~1.1-1.25 bits and layer 10 2.33, hot experts more than cold (1.83 vs 1.53); its card reports KL 0.19, top-1 91.2 %
(code 97.9 %) vs FP4. Mia 2.9 is K3 except layer 20 at K2 (their "18-22 at K2"). jayleaton 2.8: K2 early, K3 late.

## Decode speed per bit (`tools/ds41_engine_sim.py`, arena 83 GB, transient 400, global warm, held-out stream)

The sim reproduces the measured CB3 traffic (396 MB/token simulated vs 386 measured). Measured split at CB3:
41 ms/token fixed + NVMe at 3.1 GB/s.

| bits | experts resident | miss/token | MB/token | tok/s (3.1 GB/s) | tok/s (loader pipeline, ~6 GB/s) |
| --- | --- | --- | --- | --- | --- |
| 3.27 (CB3) | 37 % | 27.4 | 396 | 5.9 (measured 6.0) | ~9 |
| 3.0 | 41 % | 23.5 | 312 | 7.0 | ~10 |
| 2.5 | 49 % | 17.0 | 188 | 9.8 | ~14 |
| 2.0 | 61 % | 9.9 | 88 | 14 | ~18 |
| 1.59 | 77 % | 3.9 | 27 | 20 | ~22 |

Caveats: the fixed 41 ms assumes the EXL3 expert kernel costs what CB3's does (trellis decode is more ALU per
weight; at ~6 verify rows it should not bind, unmeasured). Cold start is much worse (1.59 bits: 14.7 miss/token
cold vs 3.9 warm). The trace is short.

## What it would take

- A pack: uniform K2 needs calibration activations, i.e. the FP4 model run layer by layer (428 GB source; not on
  disk any more), or take SAGE 1.59's experts (109 GB) by range fetch. Either way the lean non-expert checkpoint
  and Engram (~196 GB FP8) are needed again: ~300-450 GB of disk, today 41 GB free.
- Engine: an EXL3 grouped-expert kernel in the fork (TensorFold's Python `cuda/exl3/experts*.cu` runs on sm_121),
  K read per tensor.
- Refinement research (iterative trellis refinement, joint w1/w3/w2): needs real Hessians, so the same
  calibration run; with an identity Hessian LDLQ's tiles are independent and refinement has nothing to trade.

## Fitting the whole model in 128 GB: the field and the budget (2026-10-11)

Budget on the GB10: ~10 GB OS + runtime, ~5 GB KV and activations, non-expert weights ~17 GB at Q8 (MixedQ2's
card: 169.9 GB total, 152.9 GB experts) or ~9 GB at 4-bit. That leaves **~100 GB for the routed experts**
(543.6 B weights). Engram (196 B, 203-209 GB at FP8/Q8) stays on NVMe in every option, as it does today.

Field (HF cards, nobody runs it resident on one 128 GB box):

| pack | what | experts | quality stated |
| --- | --- | --- | --- |
| vcruz305 SAGE EXL3 1.59 | mixed K1-K6, no pruning | 108.9 GB | KL 0.19 vs FP4, top-1 91.2 % (code 97.9) |
| vcruz305 SAGE EXL3 3.30 | mixed | 224.9 GB | KL 0.091, top-1 94.0 % |
| LibertAIDAI REAP | prune, FP4 kept | 272 / 256 / 240 / 192 kept | ppl +4.1 % / +5.5 % / +7.1 % / +14.7 % |
| tacos4me Prune-2xRTX6K | 224 kept, FP4 | ~130 GB | GSM8K 198/200, GPQA-D 92.9 % (keep set overlaps eval) |
| apetersson MixedQ2 GGUF | IQ2_XXS/Q2_K | 152.9 GB (2.25 bpw) | none |
| antirez Q2 GGUF | IQ2_XXS + Q2_K | streamed on a 128 GB Mac | none |

Combinations that fit ~100 GB of experts:

| option | experts kept | bits/weight needed | expected loss |
| --- | --- | --- | --- |
| A mixed EXL3, no pruning | 384 | 1.47 | a bit past SAGE 1.59 (KL ~0.2) |
| B REAP 272 + mixed EXL3 | 272 (0.71) | 2.08 | ppl +4 % from pruning + ~2-bit error |
| C REAP 256 + mixed EXL3 | 256 (0.67) | 2.2 | ppl +5.5 % + error a bit below B's |
| D REAP 224 + EXL3 | 224 (0.58) | 2.5 | larger pruning loss, smaller quant loss |
| (today's direction) stream, 2-bit EXL3 | 384 | 2.0 | no pruning loss; ~14 tok/s, not resident |

Closed or out of reach here: shared-base + low-rank delta (our delta spectrum: NO SAVING on DS4.1), sub-1-bit
(QMoE-style, LittleBit) needs training, lossless entropy coding of trellis codes gains ~nothing.
Our own frequency-ranked pruning gate: keep 0.44 degenerates, 0.60 passes; REAP's saliency ranking is better than
frequency (256: +5.5 % mean-saliency vs +14 % balanced on text).

## Expert merging: closed in weight space (`tools/merge_align.py`, `data/merge-align-20261011.json`)

delta_spectrum.py averaged experts without aligning their hidden neurons, which merging methods (MC-SMoE,
HC-SMoE) do first. Aligned now: Hungarian matching of the 2,304 neurons (w1 row, w3 row, w2 column) for every pair
of 10 experts (8 hottest + 2 cold) in layers 10 and 30, 90 pairs, FP4 originals.

rel = ||F_a - P F_b||^2 / ||F_a||^2: unaligned median 2.006, **aligned min 1.871 / median 1.948 / max 2.016**.
Alignment buys ~3 %; the experts stay as far apart as unrelated matrices (2.0). Averaging a pair would cost each
~0.47 of its norm, against ~0.10 for EXL3 2-bit. Weight-space merging is closed on DS4.1; output-space merging
(HC-SMoE clusters by expert outputs) would need real activations and is unlikely to rescue weights this
orthogonal.
