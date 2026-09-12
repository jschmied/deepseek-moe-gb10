# DeepSeek MoE on one DGX Spark (GB10)

Research notes for running and compressing DeepSeek MoE models on a single GB10 (sm_121, 128 GB
unified, aarch64) under vLLM. Sibling of [`qwen38-flash-next-gb10`](https://github.com/jschmied/qwen38-flash-next-gb10),
and it inherits that repo's method: name the differing cell, state void conditions before the run,
report ranges not means, and write down the results that went the wrong way.

**Private.** Nothing here is published unless it is moved deliberately.

## Layout

| | |
|---|---|
| `notes/` | findings, numbered and dated; `notes/data/` holds summaries, never raw multi-MB arrays |
| `notes/method.md` | measurement discipline, carried over from the Qwen repo |
| `bench/` `tools/` `scripts/` | runners and harnesses (never commit `.sh` runners that carry credentials) |

## Open question this repo exists for

Whether **shared-base + low-rank expert deltas** (D²-MoE) or **shared low-rank factors across
experts** (MoE-SVD) can cut routed-expert bandwidth on a DeepSeek MoE that is *already* quantized.
See `notes/moe-expert-compression.md` — the short version is that the published ratios are measured
against FP16 and do not obviously survive stacking onto a 3-bit representation.
