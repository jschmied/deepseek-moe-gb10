# 0xBakeer/deepseek-v41-flash-spark — our base for V4.1 work

Designated the base for our DeepSeek-V4.1-Flash effort, 2026-09-12. This file records the user's
read of the repo at head **`4871b93`**; a file-level survey is in progress separately.

We have prior history with this repo: we posted a correction to its **issue #1** (NVFP4 is not
losslessly compressible by 12–25 % — order-0 entropy 3.969/4.000, combined floor 2.8 %) and rewrote
**issue #2** (NVMe read shape: 64 KiB went 0.49 → 3.90 GB/s past QD4, an 8.0× effect).

## The headline is correctness, not another speed win

**The fast decode router was wrong.** The reference router uses FP32 activations × FP32 gate weights;
FastDecoder had converted the gate to BF16. With identical inputs, **11 % of expert selections
differed at layer 0**, and up to **31 %** in deeper layers. After reverting to FP32:

| | before | after |
|---|---|---|
| layer-0 route agreement | — | **100 %** |
| deep-layer agreement | — | ~94–100 % |
| final logit relative error vs reference | 0.049 | **0.0121** |

The HTML repetition case disappears with it. A second numerical issue: the custom fused attention
kernel, with FP4 dense projections + FP8 head, could drive greedy generation into pathological
repetition — now **off by default**. Its isolated attention error was small but compounded through
routing.

**This invalidates the published speed frontier.** The 22.9 / 23.4 / 25.9 tok/s runs all predate the
FP32-router fix, so they were measured while routing a materially different model. They should not be
quoted as production throughput. A clean number requires: FP32 router · fused attention **off** ·
CB3 keep = 40 % · FP4 attention + `wo_a` · FP8 head. **The repo does not yet show such a benchmark.**

## Two levers measured and closed

- **Router top-k.** top-6 → top-5 cuts expert bytes 12.12 → 10.17 GB per verify step and verify time
  114.9 → 106.6 ms, but general-text NLL worsens **+0.025** against the repo's **+0.015** acceptance
  threshold. top-4 far worse. **top-6 stays.**
- **Fixed verify widths.** Draft widths 3/5/7 → ~41.0 / 39.7 / 39.7 ms per accepted token. The
  checkpoint's 5-draft / 6-position block is at the **static optimum**: 7 buys nothing, 3 loses
  acceptance.

## The MoE traffic correction, and why it matters to us

Six consecutive verify tokens do **not** touch ~30 distinct experts per layer. They average
**20.96**, because routing overlaps heavily between adjacent tokens. So routed experts move ~**12.1 GB
per verify step**, and the CB3 kernels take ~**65 ms at ~186 GB/s**.

That is close to the bandwidth floor, which is the important part: **conventional kernel tuning has
little left to give here.** It also sharpens the shared-base/delta question in
`moe-expert-compression.md` — a method that reduces *bytes per distinct expert touched* is the only
kind that helps, and 20.96 rather than 30 is a smaller denominator than we assumed.

## Still open

**Adaptive DSpark verification is not refuted.** The repo rejected *fixed* block lengths 4/6/8 only.
It never tested the actual idea: choosing 4/6/8 **dynamically per step from the confidence head**,
which already exists. The fixed averages being nearly tied makes adaptive selection *more* plausible,
not less — different steps have different prefix-survival curves, so a global 4 is bad and a global 8
pointless while per-step choice could still win.

## Priority order the user set

1. **Correctness baseline first.** `spec-on greedy == spec-off greedy` on a real prompt suite;
   per-layer route agreement against `Model.forward`.
2. **Then** re-establish verify/draft timing and tok/s for the exact post-fix config.
3. **Only then** more speed work. "Stop optimising the 114 ms number for a moment."

`test_spec_lossless.py` is the right gate, and the reason is worth keeping: **teacher-forced NLL
missed both the router error and the decode-loop failure**, because it never exercises speculative
decoding. That is the same lesson as our own `acceptance-is-not-quality` and the Thai/Devanagari
probe work — a metric that does not run the real path cannot catch a defect in it.

The newest commit is orthogonal: xgrammar-constrained DSML tool calls, off by default pending
real-weight verification. The repo correctly separates that formatting problem from the numerical
router bug.

---

# UPDATE 2026-09-12 — the repo moved seven commits past `4871b93` (HEAD `8b68fdd`)

A file-level survey found the tree has advanced, and **three items above are superseded**. Keeping
the original section as the record of what we believed, with the corrections here.

## Superseded

1. **The CB3 3-bit default was withdrawn** (`96915a7`, `RESULTS.md` §3.6). It scored *better*
   teacher-forced and then emitted `<!DOCTYPE><!DOCTYPE><!DOCTYPE>`. Root cause (`e28d0d7`): the
   keep-set was ranked on a **50-document corpus containing no HTML/JS/CSS/SQL**, so markup experts
   never fired, ranked cold, and were pruned.
   *This is our own lesson in someone else's repo.* A calibration corpus decides what survives, and a
   teacher-forced metric cannot see what it never asked the model to generate — the same shape as our
   Thai/Devanagari probe work and `acceptance-is-not-quality`.
2. **The acceptance gate changed.** No longer +0.015 nats teacher-forced; it is now a
   **free-generation gate** — 900–2,000 tokens, distinct-token ratio > 0.25, line-repeat < 30 %,
   plus structural checks. The user's "correctness baseline before speed" instinct is what the repo
   independently concluded.
3. **A clean post-fix benchmark now exists** (`RESULTS.md` §4.3, v0.4.0-wip), for very nearly the
   config the brief specified: `PRUNE_KEEP=0.44`, `EXPERT_FORMAT=cb3`, `ARENA_GB=98`, union trace,
   `DSV41_DENSE_FP4=attn,wo_a`, `DSV41_HEAD_FMT=fp8`, `DSV41_FUSED_ATTN=0`. 44.1 % resident, hit rate
   1.0, **no NVMe during decode**: **36.6 tok/s (HTML) → 17.1 (prose)**, 18.6 thinking-on; prefill
   337 tok/s on 5,014 tokens. `env.example` on main now ships these defaults.

Note the keep fraction moved 0.40 → **0.44** and the arena 90.5 → **98 GB** — presumably the
markup-expert repair.

## What it is, and the one fact that decides everything

**A standalone engine, not a vLLM fork.** ~7.4k lines of pure Python plus Triton JIT kernels, **no
C++/CUDA, no build step**, MIT. torch 2.13.0+cu130, Triton as shipped with it. `sm_121a` is required
because `tools/fp4_moe.py` emits `cvt.rn.f16x2.e2m1x2`.

**It bypasses vLLM entirely** — no import, no dependency, its own attention, indexer, KV cache and
speculative loop. **So vLLM #56461 (SWA `block_size=32` vs page-64-only FlashInfer SM120 sparse-MLA)
does not apply**, and neither does waiting on the untested fix PR #56509. That is the strongest
argument for building on this rather than on vLLM for V4.1.

## The blocker that is ours, not theirs: disk

Needs `deepseek-ai/DeepSeek-V4.1-Flash` at **510 GB** across 48 shards, on **≥600 GB local NVMe** —
O_DIRECT, so no NFS and no overlayfs. **We have 101 GB free.** The full path is not open to us today.

**But there is a partial-checkpoint path**, and it is the actionable finding: `Weights(..., n_layers=NL,
load_mtp=False)` (`engine/model.py:79`), used by `engine/test_layers.py` to run a 4-layer smoke test
"on the shards present", with `tools/engram_rows.py` fetching only the corpus's Engram rows over HTTP
range requests. **~4 layer shards + embeddings + tokenizer + `inference/` ≈ 50 GB** exercises the
model math and chunk-invariance without the 510 GB. It is not advertised as a supported mode. At
101 GB free that fits — and it is the only version of this that fits.

## Things to know before trusting any number in it

- **Every figure is N=1**, one box, one day, no repeats.
- Nothing is bit-exact against DeepSeek's own tilelang reference — only against **this repo's own
  torch port**.
- `LIMITATIONS.md` still concedes the graphed decode path **"is not numerically equal to
  `Model.forward` … differs by a few percent relative and is not yet explained"** — post-fix.
- **`test_spec_lossless.py` exists but no passing run of it is recorded anywhere** in `RESULTS.md`.
  The gate the user correctly identified as the right one has not been shown to pass.
- The whole v0.4.0 quality story rests on a **5-prompt heuristic gate**, not a benchmark suite.
- Housekeeping smells: dead `self.gate_bf16` / `self.mtp_gate_bf16` still allocated and read nowhere
  (`fastdecode.py:144-145`) — the removed BF16 router's corpse; and three disagreeing version strings
  (`VERSION` 0.1.0-wip, `CITATION.cff` 0.2.0-wip, `CHANGELOG` 0.3.0-wip).
- Container image **has never been built or run**, self-declared.

## Confirmed at source (still true)

Router is FP32 — `fastdecode.py:325`, `scores = F.softplus(R.mm(y.float(), self.W.layers[L].gate_w)).sqrt()`,
drafter likewise at `:380`. Fused attention off — `fastdecode.py:51`, `DSV41_FUSED_ATTN` defaults `"0"`.
Adaptive DSpark genuinely absent: `conf_proj` is loaded (`model.py:178`) and reported, but adaptive
verification is off at `model.py:711`. **The open lever is real.**

Repo state: two issues, **both ours**, both open, no owner reply; zero PRs.

## Our fork and local clone (2026-09-12)

| | |
|---|---|
| fork | `jschmied/deepseek-v41-flash-spark` — **created automatically by `gh` at 11:27 when PR #4 was opened**, not by hand |
| local clone | `~/git/deepseek-v41-flash-spark` (14 MB; their `results/*/trace/*.npz` are gitignored, so a clone is small) |
| remotes | `origin` = our fork, `upstream` = `0xBakeer/deepseek-v41-flash-spark` |
| branches | `main` (in sync at `8b68fdde188f`), `test/cb3-chunk-invariance` (open PR #4) |

Workflow for further contributions there:

```bash
cd ~/git/deepseek-v41-flash-spark
git fetch upstream && git merge --ff-only upstream/main      # or the merge-upstream API
git switch -c <topic> && ...                                 # then push to origin, PR to upstream
```

**Keep `main` fast-forwarded before sharing any `compare/main...branch` link** — otherwise the compare
lists every upstream commit as well (upstream-post skill).

Note the scratch clone used for the ds-02 coverage analysis was a separate `--depth 1` copy of
*upstream*; `notes/data/0xbakeer-coverage-8b68fdd.json` is the pinned artifact from it, so that result
does not depend on either clone surviving.
