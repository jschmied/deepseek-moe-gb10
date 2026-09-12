# Upstream posting log — DeepSeek work

### 0xBakeer/deepseek-v41-flash-spark PR #4, 2026-09-12 (user go "do pr, state prominent about risk")

First PR to that repo (it had zero). Two commits:
1. `tools/test_cb3_moe_random.py` — CB3 kernel test needing **no checkpoint**. Passes. Useful to the
   repo's 6 forks, since the real test wants 510 GB on local NVMe.
2. Chunk-invariance check added to `tools/test_cb3_moe.py` in their `ok()` idiom — **fails today** at
   prefixes 2/4/8 on real layer-0 weights (1/2/4/8 random), passing at 16/32.

**Risk stated at the top of the body, as instructed:** the PR turns their suite red, deliberately;
offered to drop commit 2 and file an issue instead; noted their `image.yml` has no GPU runner so CI
should be unaffected but asked them to confirm, since we are reading their setup from outside. Also
stated what was not established — one box, one run per cell, layer-0 only, the M-dependence not
traced to a line, and **no evidence it corrupts generation**.

Engine path verified before claiming: `v41_engine.py:370` → `moe_forward_v3` + `CB3ArenaV2`.
→ https://github.com/0xBakeer/deepseek-v41-flash-spark/pull/4

Repo is **public** (60 stars, 6 forks, MIT) — not private, which is why the risk framing matters.

### 0xBakeer/deepseek-v41-flash-spark issue #3, 2026-09-12 15:5x CEST (user go "post comment to #3")

Comment on the consolidated issue, not a new thread — the repo has had no owner reply on anything we
have filed, so a fourth item would have been noise. **Shared-base + low-rank delta expert compression
does not pay on an already-4-bit checkpoint**, measured on V4.1's own layer-0 `w1` (sha256-verified
shard) plus Qwen L24 `down_proj` for shape contrast. Deltas need 92–96 % of full rank: within 1–2
ranks of W itself, within 2–37 of noise.

Stated the trap explicitly: the Qwen row's 616 < 640 "pass" is an artefact — random scores 618, also
under break-even — so on 4:1 experts clearing break-even measures the shape, not the weights. Our own
first verdict line printed USABLE for it; the random control caught it.

Limits stated: one box, layer 0 only, 16 experts per arm, bf16 SVD, break-even specific to shared-V
bf16 factors. Draft: `notes/upstream/delta-spectrum-0xbakeer.md`.
→ https://github.com/0xBakeer/deepseek-v41-flash-spark/issues/3#issuecomment-5646328174
