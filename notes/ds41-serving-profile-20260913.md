# What the running engine actually does: memory, precision, and where the time goes

2026-09-13, measured on the live server (unpruned, CB3 experts from the native cache, arena auto,
`MAX_SEQ=32768`, DSpark on). `tools/longprobe.py` samples `/proc/diskstats` and `nvidia-smi` under a
single request and reads the engine's own `x_engine_stats` off the final stream chunk.

## Memory

| | GiB |
|---|---|
| unified pool visible | 121 |
| expert arena (auto) | **79.5** = 5,501 CB3 slots of 14.45 MB = 35.8 % of all 15,360 pairs |
| everything else resident (dense, KV, runtime, DSpark) | ~19 |
| page cache | ~18 |
| **available** | **~20** |

The arena takes 79.5 of the 88.3 GB the engine sees free, leaving its `KEEP_FREE_GB=6` floor. There
is no meaningful headroom to enlarge it. The 203 GB of Engram tables stay on NVMe and are read by
range — that is the ~18 GiB of page cache doing its job.

## Precision, by component — and two shipped options we are not using

From the checkpoint headers and the engine's own config line:

| component | bytes/layer | stored as | notes |
|---|---|---|---|
| routed experts (384) | 7,219 MB | **I8 + F8_E8M0** — FP4 packed two per byte, UE8M0 group scales | served as **CB3 3-bit** from our cache |
| attention (wq_a/b, wo_a/b, wkv) | 137.7 MB | F8_E4M3 + F8_E8M0, some BF16/F32 | **`dense_fp4: off`** |
| shared expert w1/w2/w3 | 35.4 MB | F8_E4M3 + F8_E8M0 | never quantized further |
| router gate | 3.9 MB | BF16 + F32 | precision is load-bearing: it picks the experts |
| hyper-connection mixers | 3.9 MB | **F32** | |
| `embed.weight` | 1,323.8 MB | BF16 | ≤6 rows read per token |
| `head.weight` | 1,323.8 MB | BF16 | **`head_fmt: bf16`**, read in full every step |

The engine reports `dense_fp4: off` and `head_fmt: bf16`. Both have shipped alternatives
(`DSV41_DENSE_FP4=attn,wo_a` is `env.example`'s own default, and an FP8 head is in the tree) and
neither is on in this configuration. That is **6.9 GB/step of attention and 1.3 GB/step of head at
full width**, on the memory side, untouched — an easy A/B nobody has run on the unpruned path.

## Where the time goes

One 11,366-token prompt, 71 tokens out, sampled at 2 Hz:

| phase | wall | GPU busy | NVMe read | read rate |
|---|---|---|---|---|
| prefill | 131.7 s | **31.4 %** | 341.1 GB | 2.60 GB/s |
| decode | 45.8 s | **33.1 %** | 105.1 GB | 2.27 GB/s |

Repeated: prefill 114.0 s at 39.5 % GPU and 3.00 GB/s, decode 30.2 % and 2.36 GB/s.

**The GPU is idle roughly two thirds of the time in both phases, and NVMe delivers 2.3–3.0 GB/s
against a device that gives 5.0–5.6 single-threaded and 6.8 at two in flight.** That is the whole
story: the engine is neither compute-bound nor device-bound, it is schedule-bound, which is what the
offline analysis predicted and this is the first end-to-end confirmation on the real engine.

Prefill costs **30 MB of NVMe per prompt token** (341 GB / 11,366) and runs at 86–99 tok/s.

## A finding I withdrew before reporting it

Short requests issued right after the long prompt decoded at 1.5 tok/s against the benchmark's 6.16,
and the obvious reading was that a long prefill destroys the warm-started LRU. It does not. Those
runs generated 19–96 tokens, and the engine's counters show **839 prefill misses** against 922 decode
misses for a 19-token generation — the prefill cost was being amortised over almost nothing. Re-running
the full 512-token benchmark *after* the long prompts gives **6.12 tok/s** against 6.16 before them,
with hit rate 0.8977 against 0.8959. Nothing was destroyed.

The real lesson is about the metric, not the engine: **decode tok/s measured over a short generation
is mostly a prefill measurement.** Any comparison has to fix the output length.

---

## The long-context curve, measured — and what the prompt cache is worth

`longctx-profile`, 2026-09-14. Three arms landed; the fourth asked for ~44k tokens against
`MAX_SEQ` 32768 and produced no output, crashing the tool on a `None` TTFT (guarded now — it reports
and continues).

| prompt tokens | TTFT | ms per prompt token | prefill rate | NVMe | MB per prompt token |
|---|---|---|---|---|---|
| 2,922 | 44.1 s | 15.1 | 66.3 tok/s | 113.6 GB | 38.9 |
| 5,892 | 56.5 s | 10.4 | 104.3 | 166.5 | 28.3 |
| 11,344 | 119.2 s | 10.5 | 95.2 | 345.9 | 30.5 |
| 22,210 | **213.9 s** | 9.6 | 103.8 | **598.8 GB** | 27.0 |
| 27,200 | **261.3 s** | 9.5 | 104.1 | **730.2 GB** | 26.8 |

**Prefill gets more efficient with length** — 66 → 104 tok/s, 38.9 → 27.0 MB per prompt token — which
is the chunking and the expert working set amortising. But the absolute cost does not care: a
**22k-token prompt costs 214 seconds and 599 GB before the first token appears.**

### What an extend-only prompt cache is worth, from these numbers

A three-turn agent conversation at ~11k context, 200 tokens out per turn, at the measured 6.24 tok/s:

| | |
|---|---|
| today | 3 × 119 s prefill + 3 × 32 s decode = **454 s** |
| extend-only cache | 1 × 119 s + 2 × ~2 s + 3 × 32 s = **219 s** |
| | **2.1× on the whole conversation** |

And it grows with both turn count and context: at 22k context the same three turns go from 738 s to
262 s, **2.8×**. Every decode lever measured tonight — the arena curve at +65 % steps/s being the
largest — acts only on the 32 s per turn that decode occupies.

That is the case for putting the prompt cache first, now with a measured curve under it rather than
the single 11k point it rested on before.

**One caveat on the decode column** (1.43 / 1.97 / 2.11 tok/s): those generations were 64 tokens, so
prefill misses amortise over almost nothing — the same trap as §10. They are not comparable to the
6.24 reference and should not be read as a context-length decode curve.


## The 5,892-token point, and why the 28k arm is still missing (2026-09-14)

`longctx-profile` re-run added the row above. Prefill rate over the four points: **66.3, 104.3,
95.2, 103.8 tok/s** at 2.9k / 5.9k / 11.3k / 22.2k tokens — flat from ~6k on, so prefill throughput
saturates near **100 tok/s** and TTFT is essentially linear in prompt length past that. The 2.9k
point being slower is the fixed per-request cost showing through on a short prompt, not a trend.

MB per prompt token falls monotonically — 38.9, 28.3, 30.5, 27.0 — which is the expert LRU warming
across the prompt, not an economy of scale in the reads.

**The 27,200-token point landed on the re-run**: 261.3 s TTFT, 104.1 tok/s, **730.2 GB of NVMe for
one prompt**. Prefill rate across all five points is 66.3 / 104.3 / 95.2 / 103.8 / **104.1** tok/s —
dead flat from ~6k on, so the curve is a straight line and TTFT can be read off as
`prompt_tokens / 104` seconds plus a fixed ~15 s. MB per prompt token settles at **26.8–27.0** once
the LRU is warm.

**The first attempt's 28k arm returned NO OUTPUT, and the cause was the harness, not the engine.** `longctx_profile.py`
takes a *word* target and emits `0.75 x target` words; the measured word→token ratio on this corpus
is **1.39–1.47**, so `--target 28000` asks for roughly **41,000 tokens** against a 32,768 context. The
tool's guard caught it and carried on instead of crashing the sweep, which is the behaviour added
after the first run's `ttft=None` crash. To land near 28k tokens the target is **~19,500**; requeued
that way.

Stated because it is the number the prompt cache is measured against: at 22,210 tokens this engine
spends **213.9 s before the first generated token**, every turn.
