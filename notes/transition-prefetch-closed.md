# Cross-layer transition prediction in the PREFETCH role: closed

Status 2026-09-16, job 270 on `45641f9` (enginev2). This was the last open question on the DS4.1
loader line. It is closed negative, and the mechanism is understood.

## The verdict

Production cache, 5,328 LRU slots, 25 steps, settled cohort accounting:

| arm | steps/s | pred-prec | fetch-prec | of its own oracle win |
| --- | --- | --- | --- | --- |
| null | 4.84 | — | — | — |
| perfect oracle d=1 | 6.30 | 100 % | 100 % | — |
| perfect oracle d=8 | 7.96 | 100 % | 100 % | — |
| **table d=1 topn=8** | 4.89 | 66.9 % | **3.7 %** | **+3.5 %** |
| table d=1 topn=16 | 4.63 | 53.7 % | 3.7 % | −14.3 % |
| **table d=8 topn=8** | 4.83 | 65.3 % | **3.3 %** | **−0.1 %** |
| table d=8 topn=16 | 4.58 | 51.9 % | 2.8 % | −8.3 % |

The best arm captures **+3.5 %** of the oracle win. Every other arm is negative. Under a pressured
cache (1,200 slots) they are catastrophic: −101 % to −290 %.

## Why, and it is not "the table is bad"

**Prediction accuracy is fine and holds with distance** — 66.9 % at d=1, 65.3 % at d=8. Expert
identity IS predictable eight layers ahead. The table is not the problem.

**The conversion is the problem.** Over FETCHES the same predictor is at 3.3–3.7 %. The cache runs
at 92.7 %, so a correct prediction is usually already resident: it costs no I/O and gains nothing.
A wrong one is essentially never resident, so every error buys a read, a slot and an eviction. The
issued mix is therefore dominated by the errors, and a 66 %-accurate predictor produces ~3 % useful
fetches.

That is a property of the cache hit rate, not of the predictor, and no training fixes it.

## What a working predictor would have to do

Not "be more accurate". It would have to predict the **miss** set — the 7.3 % tail — and an
activation-ranked predictor ranks that tail LAST. Job 205 tried exactly that (fit on misses instead
of activations) and every arm was negative too.

## Three targets, all exhausted

* activation-targeted, d=1 (jobs 195/200/215/220) — 1.5 % of the oracle win at best
* miss-targeted (job 205) — negative at every topn
* activation-targeted, d=4 and d=8 (jobs 265/270) — +3.5 % at best

## Scope

Calibrated model, not real CB3/NVMe/GPU leaves. One trace, two cache sizes, 25 steps, one rep.
Ranks arms; does not predict the box. The perfect oracle is worth +30 % (d=1) to +64 % (d=8) in the
same model, so the CEILING is real and large — what is absent is any counting-based predictor that
reaches it.

## Measurement history, because it matters here

This job ran SIX times. Five were discarded because the accounting changed underneath them, and
every one of those bugs biased in the PREDICTOR'S FAVOUR: speculative inserts credited a phantom
use count; wrong prefetches stayed resident; `issued` was the fetch-precision denominator instead
of reads that started; a correct-but-evicted prefetch counted as a timely hit; the finite window
right-censored long horizons. The negative result survived all five corrections, which is the
strongest form this conclusion can take. Superseded logs are kept as `270-*.PRE-<sha>.log`.
