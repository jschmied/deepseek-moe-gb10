# DS4.1 on our GB10: environment, file placement, and the plan

Written 2026-09-12 while the 475 GiB checkpoint downloads to PBS (ETA Sun ~11:20–12:25 at the
degraded WAN rate; ~1.7 h if the 100 Mbit port is fixed).

## 1. What we already hold

| where | what |
|---|---|
| `/opt/llm/models/dsv41-shards` (31 GB) | shards 1,2,3,4,5,6,43 — **layers 0–3 complete** (384 experts each, routers), embeddings, `head`+`norm`, vision tower. All sha256-verified against HF `lfs.oid`. |
| same | `config.json`, `model.safetensors.index.json`, `tokenizer.json`, `tokenizer_config.json` (fetched 2026-09-12) |
| PBS `hf/deepseek-ai--DeepSeek-V4.1-Flash` | full checkpoint, in progress, `SOURCE.json` pinned at revision `dba1be0a40aa`, 88 files, all 88 verifiable |
| `~/git/deepseek-v41-flash-spark` | our fork's clone; `origin`=fork, `upstream`=0xBakeer, main at `8b68fdd` |
| `notes/data/0xbakeer-coverage-8b68fdd.json` | their committed per-layer routing histogram — the ds-02 input, pinned so it survives clone deletion |

Note: `scratchpad/dsv41/tools` is a **copy of their tools**, not ours. The only tool we have written
for this track is `tools/delta_spectrum.py`.

## 2. Environment — their native path is the right one for us

They document two paths and say "they are the same server": a venv, or a container
(`docker run --gpus all --ipc=host --ulimit memlock=-1`). **Take the venv.** Their pin is
`torch==2.13.0+cu130`, which is exactly what `vllm-venv-fnmain3` already runs, and a venv keeps the
instrumentation this project depends on — `source_toggle` patching `site-packages` between A/B arms,
reading module source to confirm a patch applies, swapping `_C_det.so` in the live path. Their
container would be opaque to all of it.

If the container is ever used here it will likely need **`--cap-add=SYS_PTRACE`**, which their docs do
not list: PLE offload's `rebuild_cuda_tensor` needs `pidfd_getfd`, and without it the engine dies
~10 min in with only `Failed core proc(s): {}` (our `flashnext-vllm-working-config` memory).

Also fix on setup: `env.example` ships `ARENA_GB`/`TRACE_STATS`/`TRANSIENT_SLOTS`/`KEEP_FREE_GB`
**empty** with the v0.4.0-wip block commented out, and `start.sh:160` picks the trace by
`ls … | sort | tail -1`, which lands on `trace-union` only by lexicographic luck. Set them explicitly:
`PRUNE_KEEP=0.44 EXPERT_FORMAT=cb3 ARENA_GB=98 TRANSIENT_SLOTS=8 KEEP_FREE_GB=6`.

## 3. File placement — and the space fork this forces

Their engine reads experts out of the checkpoint with `O_DIRECT` on **every miss**, so the 510 GB must
be on local NVMe: not PBS, not NFS, not overlayfs (`docs/install.md:12`). CB3 does not help — it is
packed on the GPU at warm start "so nothing on disk changes".

| step | +GB | GB10 free |
|---|---|---|
| now | — | 232 |
| `archive-work` (calib 91 + bf16-work 52 + 27b-nvfp4 22) | 165 | 397 |
| `archive-prod` (qwen38-flash-next-nvfp4 126) | 126 | 523 |
| venvs, after capturing their hand edits as diffs | 102 | 625 |

**523 GB does not comfortably hold a 510 GB checkpoint.** So running their engine here means also
giving up the venvs — i.e. the box stops being a Qwen serving/experiment environment. Transfer from
PBS over the measured 96.8 MB/s LAN is ~1.5 h once the space exists.

**This is a strategic choice, not a logistics detail, and it should be made deliberately:**

* **Path A — run their engine.** Costs the Qwen environment. Buys end-to-end DS4.1 on one box, and the
  ability to test ideas 1/4/5/6 for real.
* **Path B — offline analysis only.** ds-01 and ds-02 were both produced from their *committed
  artifacts* plus our 4 layers, with no checkpoint and no engine. The top transplant item (b12x MXFP8
  kernels) needs no DS4.1 at all. Costs nothing, but cannot measure their engine's behaviour.

## 4. Optimisation plan, ordered by evidence we already have

1. **CSA2 indexer on layer 2 — do this first, it needs nothing.** Shard 5 carries
   `layers.2.attn.indexer.{wk,k_norm,weights_proj}` and `attn.compressor.wgate`; `compress_ratios`
   starts `[0,0,2,…]` so **layer 2 is the first indexed layer**. This is the same machinery as Qwen's
   QSA, where we hold det-207…234 and two upstream threads. Highest leverage per unit of work.
2. **Request-specific expert overlay (their idea 5) — the data now supports it.** ds-02 measured
   cross-domain misses of **43.78 %** (coding keep-set → general accesses) and **52.24 %** the other
   way. The union is a poor compromise, not a superset. An overlay chosen from the prefill routing
   histogram attacks exactly that.
3. **Layer-major prefill (their idea 6).** Their own profiling shows the long-prompt cost is repeated
   CB3→FP4 unpacking, 291 → 326 → 369 tok/s as chunk goes 512 → 1024 → 2048. Note chunk 2048 is
   *already* their default, so that curve is a demonstration of the cost, not an available speedup.
4. **DSpark confidence scheduling (their idea 4) — blocked on a prerequisite.** The confidence head is
   computed at `engine/model.py:722` and explicitly not acted on, and the *shipped* decode path is the
   CUDA-graph `FastDecoder`, which never evaluates `conf_proj` at all. It must be wired into the served
   path before any offline simulation has traces to consume.
5. **`--speculative-dspark-align-verify-tokens-to-graph-tier`** — fill the verify window to the graph
   tier the forward is padded to anyway. All four multi-Spark repos leave this lever inert, including
   rhys101's 82.73 tok/s run (`speculative_dspark_sps_table_path: null`).

**Closed, do not reopen:** hybrid resident+SSD with an unmasked router (their idea 1) — ds-02 measured
16.9 % access-weighted miss in-sample, above their own 10–15 % "SSD dominates" line, ≈0.59 GB/token,
a 5–12 tok/s ceiling. And the hash-routed-early-layer budget (idea 3) — ds-01: all 40 layers carry a
learned `ffn.gate.weight`, zero hash tensors; that belongs to the other V4 checkpoint.

**Note the tension:** idea 1 wants the residency set that minimises cold traffic (= frequency ranking,
which is already miss-optimal by construction), while REAP saliency (idea 2) deliberately keeps
rarely-selected-but-high-contribution experts. REAP cannot beat frequency on *traffic*; it is a
quality-at-fixed-residency claim. Do not treat them as additive.

## 5. Next steps, in order

1. Finish `archive-work`, then `mtp-remeasure3`, then `archive-prod` (chained; `archive-prod` must wait
   because `mtp-remeasure3` needs the prod checkpoint).
2. **Decide Path A vs Path B.** Needs the user.
3. Build the CSA2-indexer probe against layers 2 (works today, no download, no engine).
4. Design `pr56509_test.py` properly and un-park it — a venv clone plus the #56509 diff plus a
   `--load-format dummy` init on sm_121; `config.json` is now local. Currently blocked by qnext's
   artifact guard, which is correct.
5. When the download finishes: `hfverify.py` against `SOURCE.json` (88 files, two hash kinds) **before**
   trusting it. The `fnbf16` stage in the same archive is missing one shard of 132 and printed
   `== ALL DONE ==` anyway.
