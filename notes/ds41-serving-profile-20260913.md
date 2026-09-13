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
