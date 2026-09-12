# TODO

## Live

- **Survey the current DeepSeek open-weights landscape** (started 2026-09-12, running as a subagent).
  Newest models incl. any V4.x, expert geometry from each `config.json` rather than blog copy,
  quantized community releases that fit 128 GB, vLLM support status for the newest architectures,
  and any published DeepSeek-specific expert-compression work. Results land in `notes/the-field.md`.

## Queued

- **Decide the shared-base + delta question cheaply first.** SVD spectrum of `W_expert − W_base` on
  a few layers of the target checkpoint; if the deltas are not strongly low-rank at these shapes the
  question closes offline. See `notes/moe-expert-compression.md`.
- **Establish what the binding constraint actually is** on GB10 for the target model before
  optimising expert bandwidth. On Qwen3.8-Flash-Next it turned out 69 % of single-stream decode was
  BF16 GEMV on unquantized *dense* weights, and compressing experts would have addressed the wrong
  thing.

## Standing cautions carried from the Qwen repo

- Published compression ratios are against FP16. We are not at FP16.
- `notes/method.md` applies here: name the differing cell, write the void conditions before the run,
  ranges not means, and record the results that went the wrong way.
