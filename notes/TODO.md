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

- ~~**Decide the shared-base + delta question cheaply**~~ — **CLOSED 2026-09-12.** Deltas need 92–96 %
  of full rank on both DS V4.1 L0 and Qwen L24: within 1–2 ranks of `W` itself, within 2–37 of noise.
  The Qwen row's apparent pass was a shape artefact the random control caught. See
  `notes/moe-expert-compression.md`; posted to 0xBakeer #3.

- **Establish the binding constraint** before optimising expert bandwidth. On Flash-Next 69 % of
  single-stream decode was BF16 GEMV on unquantized *dense* weights; compressing experts there would
  have fixed the wrong thing. Still open, still no box time.

- **NEW, and the best-leveraged thing we hold: the CSA2 indexer on layer 2.** Shard 5 carries
  `layers.2.attn.indexer.{wk,k_norm,weights_proj}` and `layers.2.attn.compressor.wgate` — the real
  sparse-attention weights, on disk and checksum-verified. Layers 0/1/3 have no indexer. This is the
  same machinery as Qwen's QSA, where we have det-207…229 and two upstream threads. No download.

## Blocked on the user

- **Nothing for Tier 2 any more — that entry was wrong.** It said ~50 GB had to be downloaded. The
  seven verified shards already *are* the partial checkpoint: **layers 0, 1, 2, 3 complete** (384
  experts each, routers `ffn.gate.{weight,bias,bias_vl}`, attention), plus embeddings (shard 2),
  `head`+`norm` (shard 43) and the vision tower (shard 1). `config.json`, the index and both
  tokenizer files were fetched 2026-09-12 (a few MB). Per-layer route agreement is runnable now.
  Note `dsv41-tier2` was never actually in the qnext queue despite this file claiming it was.

- **Engram: a whole-shard fetch is impossible, and that is now measured.** The engram tensors are not
  missing from layers 1/14 — they live in **shards 47 and 48**, `94.56 GiB each, 189.13 GiB total`,
  against **85 GB free**. So `engram_rows.py`'s HTTP-range approach is not an optimisation, it is the
  only option. Config: `engram_vocab_size 16,000,000`, `num_embeddings [384006168, 384016682]`,
  `n_heads 8`, `head_dim 256`, `max_ngram_size 4`, compressed vocab 99,092.

## Standing cautions

- Published compression ratios are against FP16. V4.1 is already fp4.
- `notes/method.md`: name the differing cell, void conditions before the run, ranges not means,
  record what went the wrong way.
