# TODO

## Done 2026-09-12

- Landscape survey → `notes/the-field.md` (V4.1-Flash exists, 2026-09-10, 475.3 GiB, 384+1 experts).
- 0xBakeer repo assessed at `8b68fdd` → `notes/0xbakeer-repo-state.md`; three brief items superseded.
- **Tier 0** FP4 kernel audit, no checkpoint → `notes/tier0-fp4-kernel-audit.md`.
- **Tier 1** their own test on real layer-0 experts (one 6.88 GiB shard, sha256 verified).
- **CB3 chunk invariance**: fails at prefixes 2/4/8 on the shipped path → `notes/cb3-chunk-invariance.md`.
- Consolidated our two issues into **#3**; opened **PR #4** (first on that repo).

## Live — waiting on other people

- **0xBakeer #3** (issue) and **#4** (PR): both open, **zero comments**, opened today. No owner reply
  on anything we have filed there, ever. Nothing to do but wait.

## Next, and the situation changed under it

- **Test vllm PR #56509 on this box.** Still cheap — `--load-format dummy` means the 475 GiB does not
  apply, so it is a venv clone and a few minutes.
  **New as of 2026-09-12 02:10:** @munakaya reproduced the failure on **RTX PRO 6000 Blackwell
  (SM120, cc 12.0, x86_64)** and reports **#56509 does not fix it**. So:
  - the PR is in trouble — its author is still blocked asking for a `ready`/`verified` label, and now
    has a report that the fix does not work;
  - our datapoint is *still* distinct: that was **sm_120, x86_64**; we are **sm_121, aarch64**, and
    the PR special-cases "device capability family 120", which includes 121;
  - the interesting question is no longer "does it fix GB10" but **what the residual failure is on
    sm_121**, which nobody has posted.
  Reporting needs the user's go.

## Queued research (no box time, no downloads)

- **Decide the shared-base + delta question cheaply**: SVD spectrum of `W_expert − W_base` on a few
  layers. If the deltas are not strongly low-rank at these shapes it closes offline. Note we now know
  V4.1 experts are **already fp4** and the architecture **already has a shared expert** (384+1), so
  two of the three legs of the original argument are weaker than assumed
  (`notes/moe-expert-compression.md`).
- **Establish the binding constraint** before optimising expert bandwidth. On Flash-Next 69 % of
  single-stream decode was BF16 GEMV on unquantized *dense* weights; compressing experts there would
  have fixed the wrong thing.

## Blocked on the user

- **Tier 2**, ~50 GB partial checkpoint (4 layers + embeddings + tokenizer) for per-layer route
  agreement against `Model.forward`. Queued in qnext as `dsv41-tier2`, `needs_user: true`.

## Standing cautions

- Published compression ratios are against FP16. V4.1 is already fp4.
- `notes/method.md`: name the differing cell, void conditions before the run, ranges not means,
  record what went the wrong way.
