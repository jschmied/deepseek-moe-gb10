# Fused prefill attention + gather fusion: the quality gate, measured

Status 2026-09-16. Jobs 250 / 255 / 260. The speed side was already known: **40.4 s vs 53.6 s**
prefill. This settles what it costs in quality, and corrects two things the gate had been carrying.

## The gate did not need a paired NLL, and could not have had one

`tools/paired_nll.py` pairs `expert_trace.py` arms. Those run **`v41_ref`**. The fusion flags
(`DSV41_ATTN_FUSED_PREFILL`, `DSV41_ATTN_GATHER_FUSED`) live in **`engine/model.py`** — the serving
path. The trace harness never sees them, so "needs a paired-NLL verdict" named an instrument that
does not apply.

## Token comparison is not an instrument here either (job 250)

Same prompt, temperature 0, each arm twice:

| comparison | first divergence |
| --- | --- |
| arm A (fusion off) run 1 vs run 2 | char 153 |
| arm B (fusion on) run 1 vs run 2 | char 244 |
| A vs B | char 244 |

**Neither arm reproduces itself**, and the cross-arm difference is *smaller* than arm A's
disagreement with itself. Any verdict read off emitted tokens would have been noise.

## What does work: offline logits, teacher-forced (jobs 255, 260)

No server, no sampling. One weight load serves every arm — `model.py:402` reads the fusion flags as
module globals at **call** time. `m.c.rollback(0)` rewinds between arms (`begin_prompt()` only drops
the replay buffer; it does not touch `c.len`).

12 documents from `corpus/trace_corpus_v3.jsonl`, 4,506 paired positions:

| arm | maxabs | top-1 agree | ΔNLL (paired) | SE | t |
| --- | --- | --- | --- | --- | --- |
| eager vs eager | 0.000e+00 | 100.000 % | 0 | 0 | — |
| eager vs fused | 1.612e+01 | 99.048 % | −2.842e-03 | 9.23e-04 | −3.1 |
| eager vs fused+gather | 1.612e+01 | 99.048 % | −2.842e-03 | 9.23e-04 | −3.1 |

**The floor is exactly zero**: prefill is bit-deterministic, unlike decode. So the floor is a
control on the harness, not a noise band, and every other row reads at face value.

**The gather fusion is bit-identical end to end** — `fused vs fused+gather` was 0 on every metric in
job 255, confirming `engine/test_gather_fused_attn.py` independently. It carries no numerical risk
and could ship on its own.

## The NLL "win" is calibration, not information

The fused arm scores *better* on NLL, which is the trap [[lower-nll-can-be-softening]] describes.
Checked rather than assumed:

```
baseline NLL 0.702931  entropy 0.4577
fused    NLL 0.700088  entropy 0.4598   <- SOFTER
best temperature on the BASELINE alone: T=1.015 -> NLL 0.699847
```

T=1.015 on the baseline reaches **below** the fused arm's NLL. Temperature alone accounts for the
whole difference, so the fused arm is not better — its distribution is softer, and a softer
distribution scores better under NLL while being no better at picking the mode.

## Verdict

* **Gather fusion: no quality cost at all.** Bit-identical.
* **Fused prefill attention:** the real, measured change is **0.95 % of positions flipping their
  top-1 token** (99.048 % agreement) with a max logit deviation of 16.1. There is **no evidence it
  is better** (the NLL gain is softening) and **no evidence it is worse**.

So the trade is 24.6 % off prefill against ~1 % of top-1 tokens changing, with quality otherwise
unmeasurable at this sample size. That is a judgement call, not a measurement, and it is not made
here.

## Scope

Teacher-forced prefill logits only, one corpus, 4,506 positions, single run per arm (the floor is
exactly zero, so repetition adds nothing). Says nothing about decode, where job 250 showed the
engine is not self-reproducible at temperature 0.
