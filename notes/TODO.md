# TODO

## THE BIGGEST ITEM: there is no prompt cache — 2026-09-13

`V41Engine.generate()` calls `self._reset()` unconditionally (`engine/v41_engine.py:608`), so **every
request re-prefills its whole context from scratch**. `LIMITATIONS.md` confirms batch size 1 and one
serialising lock. Confirmed in data taken for other reasons: the benchmark ran the same 62-token
prompt three times for TTFT 8,485 / 8,369 / 8,101 ms — no decay — and two identical 11,366-token
prompts back to back gave 131.7 s then 114.3 s, which is expert-LRU warming, not prefix reuse.

**What it costs.** Prefill measured at 86–99 tok/s and **30 MB of NVMe per prompt token**. A realistic
agent turn — 10k context, 200-token reply — is ~120 s of prefill against ~32 s of decode: **79 % of
the wall, paid again every turn on almost entirely identical tokens.** Every decode lever measured
today (the 1.42× from the CB3 cache, +3.4 % from dense FP4, a hypothetical +13 % kernel repack) acts
on the other 21 %.

**The cheap version is very cheap.** A radix prefix cache is a big change, but this engine is
single-sequence and serialised, and the agent case is always *turn N+1's prompt = turn N's prompt +
reply + new user text*. So the whole win is **do not reset when the new prompt extends the previous
one**: compare against the retained token ids, `Caches.rollback(n)` to the divergence point — and
rollback already exists, because speculative rejection needs it.

Two known constraints: the compressor works in groups of `compress_ratio`, so a reusable prefix must
be truncated to a group boundary (`LIMITATIONS.md` already flags that rollback only restores the
pending token inside the last chunk); and the Engram hash state is a function of token ids alone, so
it replays deterministically and costs nothing. Memory is not a constraint — `ckv` at 32k/ratio-2 is
~655 MB over 40 layers and `ik` ~164 MB, so a full 32k context is roughly **1 GB against the ~20 GB
free**.

Sequence: `longctx-profile` (queued) measures the curve it has to beat, then build it.

## Actionable prefill and decode levers, ranked — 2026-09-13

Everything here is measured today unless marked. See `ds41-measured-2026-09-13.md`,
`ds41-serving-profile-20260913.md`, `native-cb3-expert-cache.md`, `ds41-field-survey-20260913.md`.

**Prefill (79 % of an agent turn, and barely touched)**

1. **Prompt cache, extend-only** — above. Removes prefill entirely for turns 2..n.
2. **`DSV41_PREFILL_CHUNK`** (queued). `env.example` says the miss term is chunks × layers × 7 GB, so
   quadrupling the chunk should quarter it. Default 2048, never swept here.
3. **Engram prestage** — hash eagerly in `prepare_inputs`, one `preadv` into a pinned buffer, one
   H2D, graphs stay on. Somebody else measured **26.1 → 80.2 tok/s c=1** from this pattern; nine files.
4. **The indexer scores all positions when four of eight layers only consume 16,384 candidates**, and
   materialises a `[T,32,N]` intermediate twice. Bit-exact, and roughly halves a long-context step.
5. **QSA `float4` key rows** — someone measured prefill 359 → 174 ms from load width alone.

**Decode**

6. **Unblock the per-layer serialisation.** GPU is 31–33 % busy and NVMe runs at 2.3–3.0 of an
   available 5.0–6.8 GB/s, in *both* phases. Independently confirmed by JigSawPT: a drive giving
   10.04 GB/s at decode's own queue depth while the engine extracts 4.33, because a layer's reads
   wait on its router. Worth ~1.25–1.35× and bit-exact.
7. **CB2 at higher coverage** — 9,992,192 B/slot, wall ratio 0.801 measured against CB3. Trades
   quality for coverage; needs the full gate.
8. **3-bit scales in the arena, not just on disk** — +2.2 pp coverage, but the kernel reads the scales
   in its inner loop and §6b showed it is occupancy-limited, so measure before building.
9. **Widen the verify block** (queued) — free on bytes and free on the kernel; acceptance is the only
   thing that can make it not free, which is what `sweep-block-width` settles.
10. **Batching** — the dense chain is flat to M=48, so every non-expert byte amortises: ~+49 %
    aggregate at 8 sequences. Needs the streaming store to survive concurrency, and the engine is
    single-sequence today.

**Measured dead, do not spend on these**: `DSV41_IO_THREADS` and `DSV41_READ_CHUNK_MB` (offline;
`sweep-io-threads` re-checks the first now that a miss is 5× cheaper), per-layer arena budgets,
prefill-warmed keep-sets, drafter-driven expert prefetch, cross-layer prefetch, eviction policy,
GPUDirect Storage, io_uring, the CB3 arena repack, low-rank expert deltas, expert merging.


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

## Closed 2026-09-14

- **`MAX_SEQ` is not a decode lever — the 7 % was the arena.** With `ARENA_GB` pinned at 79 GB,
  4096 and 32768 are 0.1 % apart (6.074 vs 6.080 tok/s) across an 8× range of indexer columns, and
  the hit rate does not move. `ds41-measured-2026-09-13.md` §17; §7's reading is withdrawn there.
  **Consequence: vLLM #56686's class of fix buys us nothing at short context**, and "lower
  `MAX_SEQ` for speed" is off the table — it only ever worked by leaving more room for experts, so
  raise `ARENA_GB` directly instead. Untouched by this: the indexer's cost at 20k real positions
  (prefill item 4), which this 62-token bench cannot see.

- **CB3 quality — settled.** Paired teacher-forced comparison against FP4 on the same 17,704
  tokens: coding top-1 **81.55 % → 80.67 %** (McNemar 239/138, z = 5.20, p = 2e-7), general top-1
  null. CB3's *lower* NLL is a calibration artifact — its entropy is higher, and temperature alone
  on the FP4 arm reaches a better NLL than CB3 does. See `ds41-measured-2026-09-13.md` §16 and
  `tools/paired_nll.py` in the fork. **State the trade when offering 3-bit: about one point of
  coding top-1 for 211.6 GB instead of 296 GB and a 6–12× cheaper miss.** Still owed: one real
  agent turn, not more NLL.

## Standing cautions

- Published compression ratios are against FP16. V4.1 is already fp4.
- A lower-precision arm with a *lower* NLL is almost always softening, not information. Check the
  predictive entropy, and sweep one temperature on the baseline's own logits before believing it.
- `notes/method.md`: name the differing cell, void conditions before the run, ranges not means,
  record what went the wrong way.

## Residency simulations — done 2026-09-13 (see notes/expert-residency-simulations.md)

ds-03..ds-08 on real Qwen routing traces, held out, before any engine work:

- **Drop the frozen keep-set.** Adaptive LRU of the SAME size beats it by ~20 points (93.4-94.8 % vs
  73.1-86.8 %). The static core contributes nothing at any mix. Their `TRANSIENT_SLOTS=8` is ~2
  orders of magnitude too small to register.
- **Do not tune per-layer allocation** (~1 point, reproduces their `global ≈ uniform`) or the
  **eviction policy** (LRU ties Belady — capacity is the variable, not policy).
- **44 % is a memory artefact**, not a design point: the coverage curve has no knee there.
- **I/O: overread is 0.06 %, reads are already coalesced 6→2, alignment is right.** The gap is queue
  depth (2.5 of 5.5 GB/s; ~1 miss per layer in flight). GPUDirect will not help — `nvidia_fs` is not
  loaded and the host bounce is ~2 % on unified memory.
- The prompt overlay helps at the worst case (+18.5 pp short / +6.2 pp long) but loses to a plain
  LRU of equal size, so **adaptivity is the active ingredient**, not prompt-derived prediction.

**Caveat on all of it: measured on Qwen (512/top-10/48L), not DS4.1 (384/top-6/40L).** Directions
transfer, levels do not.

**Next:** trace DS4.1 unmasked with `tools/expert_trace.py` (`prune_keep` unset). It streams one
7.4 GB layer shard at a time — needs neither the 510 GB resident nor the 189 GB engram shards, and
resumes as shards land — then re-run `tools/*_sim.py` against it.
