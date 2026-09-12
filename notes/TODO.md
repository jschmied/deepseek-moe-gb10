# TODO

## Live

- ~~Survey the DeepSeek open-weights landscape~~ **DONE 2026-09-12** → `notes/the-field.md`.

## Next, and it is cheap

- **Test vllm PR #56509 on this box.** V4.1-Flash cannot start on SM120/SM121 (issue #56461: SWA
  cache hardcoded `block_size=32` vs FlashInfer SM120 sparse-MLA page-64 only). The fix PR is **open
  and explicitly hardware-untested** — its author is "relying on CI and upstream review for SM120
  validation". We have a GB10. **The 475 GiB size blocker does not apply**: #56461 was reproduced
  with `--load-format dummy`, so the startup path is testable without weights. Cost is a venv clone
  and a few minutes. Report to the PR needs the user's go.

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
