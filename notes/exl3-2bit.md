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
