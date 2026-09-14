# DeepSeek MoE on one DGX Spark (GB10)

Research notes for serving **DeepSeek-V4.1-Flash** — 40 layers, 384 routed experts + 1 shared,
top-6, FP4, a 510 GB checkpoint — on a **single** GB10 (sm_121, 128 GB unified, aarch64), with the
full router intact rather than pruned.

Sibling of [`qwen38-flash-next-gb10`](https://github.com/jschmied/qwen38-flash-next-gb10), and it
inherits that repo's method: name the differing cell, state void conditions before the run, report
ranges not means, take three starts in separate processes, and **write down the results that went
the wrong way**. A good deal of what is here is a withdrawal.

The engine work lives in a fork of [0xBakeer/deepseek-v41-flash-spark](https://github.com/0xBakeer/deepseek-v41-flash-spark);
this repo is the measurements, the tools that took them, and the reasoning.

**Branches in the fork.** Work happens on **`one-spark-full-router`**. `main` stays pinned to
upstream `8b68fdd` so the delta reads cleanly, and `feat/cb3-disk-cache` is kept alive at the same
commit because notes and commit messages link into it — it started as just the on-disk expert cache
and had grown into the whole engine programme, which is why the trunk got a name that says what it
is. Nothing is ever proposed upstream.

**This repository is public.** It was described as private here until 2026-09-14, which was simply
wrong and is the kind of line that invites putting something in a repo that should not be in one.
What actually keeps it safe is a scan before every commit —
`git grep -nI -E "sk-[A-Za-z0-9_-]{16,}|hf_[A-Za-z0-9]{30,}|BEGIN [A-Z ]*PRIVATE KEY"` — plus the
rule that `.sh` runners are never committed, because they are where host names, paths and
credentials collect. Both have held: the full history scans clean, and no `.sh` has ever been added.
Measurements, tools and reasoning belong here; checkpoints, credentials and runners do not.

## The position, as measured

The checkpoint does not fit: 15,360 routed experts at 14.45 MB (CB3) is 222 GB against ~83 GB of
arena. Everyone else's answer is to prune the router to a task-specific keep-set and hold it
resident. Ours is to keep all 384 experts routable and stream the misses, and the question this
repo exists to answer is whether that can be made fast enough to prefer.

| lever | status |
|---|---|
| **Extend-only prompt cache** | **built.** Turn 2 of a conversation costs **~21 s at any context** — 2.85× / 5.83× / 9.12× at 5.9k / 11.4k / 22.2k tokens. TTFT stops being linear in the prompt |
| **Layer-major encoder prefill** | **built.** Each layer's experts read once per prompt instead of once per chunk; the load floor is **context-independent**. Validation in flight |
| Loader pipeline | not built. Targets the 75 % of decode wall spent waiting on NVMe; ~2.2× of headroom (3.2 GB/s of an available 5.0–6.8) |
| Cross-layer expert prediction | measured at 32 % of misses converted at 1.5× overfetch, 7× the popularity baseline — below the 70–80 % bar, revisit after the pipeline |

**Where the time goes** (`ds41-measured-2026-09-13.md` §21, three processes per arm): decode is
**75–76 %** waiting for NVMe, **18 %** waiting for the GPU to name the layer's experts, **1.5 %**
host bookkeeping. Prefill is ~79 % of an agent turn and runs at 95–104 tok/s flat from ~6k tokens on.

**What quality costs.** CB3 (3-bit experts) costs **0.88 pp of coding top-1**; CB2 costs **2.16 pp**
and is not a serving candidate. Pruning degenerates between keep 0.44 and 0.60 — and the band that
survives needs 133 GB, so **pruned all-resident is closed on this box**.

## Layout

| | |
|---|---|
| `notes/` | findings, numbered and dated; `notes/data/` holds summaries, never raw multi-MB arrays |
| `notes/method.md` | measurement discipline, carried over from the Qwen repo |
| `notes/TODO.md` | the live ordering |
| `tools/` | the harnesses below — all runnable against a live engine or an offline trace |
| `bench/` `scripts/` | runners (never commit a `.sh` that carries credentials) |

## Where to start reading

| note | what it settles |
|---|---|
| `ds41-measured-2026-09-13.md` | the numbered result log, §1–21, including every withdrawal |
| `layer-major-prefill.md` | the transpose: oracle, then the built engine measurement |
| `prompt-cache.md` | design, the three bounds on the resume point, and what it is worth |
| `native-cb3-expert-cache.md` | the 3-bit on-disk expert cache, 211.6 GB, bit-identical |
| `expert-frequency.md` | the keep-set curve, the pruning cliff, and whether one universal set can replace task profiles |
| `the-field.md` | who else is doing this, and which of their numbers are comparable |
| `ds41-serving-profile-20260913.md` | the long-context curve, 2.9k → 27.2k tokens |

## Tools

Measurement harnesses, not a library. The ones that answer a question on their own:

| | |
|---|---|
| `qnext.py` | the job queue: one job per invocation, completion by the job's own sentinel, never on a wall clock |
| `prefill_io_oracle.py` | replays a recorded route log into current / layer-union / cold-arena byte counts — what a transpose is worth, before writing one |
| `paired_nll.py` *(in the fork)* | pairs two quantization arms token by token, with McNemar and a temperature control |
| `universal_keepset.py` | whether one universal keep-set can replace per-task profiles, and at what size |
| `xlayer_predict.py` | cross-layer expert prediction scored on misses converted, not on accuracy |
| `gen_gate.py` / `token_integrity.py` | degeneration and rare-identifier corruption — two different failure modes, and the first cannot see the second |
| `step_breakdown.py` | decode decomposed into NVMe wait, GPU-routing wait and bookkeeping |
| `env_sweep.py` / `longctx_profile.py` / `promptcache_profile.py` | the sweeps behind most cells above |

## Two standing cautions

- **Published compression ratios are against FP16. This checkpoint is already FP4** at ~2.6
  effective bits, so 3-bit and 2-bit are below what is actually stored, not comfortably above it.
- **A lower-precision arm with a *lower* NLL is almost always softening, not information.** Check
  the predictive entropy and sweep one temperature on the baseline before believing it. This caught
  a wrong verdict on CB3 and again on CB2.
