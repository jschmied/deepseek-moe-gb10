# Speedup ideas from three parallel surveys, 2026-09-13

Three agents, three angles (kernels / memory-IO / speculation-scheduling), each given our measured
numbers. **All three corrected a premise I had supplied**, which is the main lesson of the exercise
and is recorded at the end.

Provenance convention adopted from the memory/IO survey and applied throughout:
**[measured today]** on this box this session · **[stored]** from our artifacts, needs a build stamp ·
**[external]** published by someone else.

---

## Hardware facts that were WRONG in our own notes — now measured

| what | we believed | **[measured today]** |
|---|---|---|
| NVMe large sequential O_DIRECT | 5.5 GB/s, "queue depth is the whole game" | **6.9 GB/s** at 4 threads × 4 MB, **4.42 GB/s** from one thread. *Refined 2026-09-13 on the real shards at the engine's own shape:* one expert in flight gives **3.97 GB/s in 4 MiB chunks but 5.58 in one `pread`**, and two in flight reach **6.82** either way — so the read size matters more than the depth |
| the QD curve | 3.5k IOPS @ QD1 → 112k @ QD64 | that is the **4 KB** curve. Large reads self-generate ~115 in-flight commands per expert (`max_hw_sectors_kb = 128`), so **QD is not the story for expert reads** |
| pinned host→device copy | "a memcpy at ~273 GB/s" | **59.4 GB/s**, flat across 1/2/4/8 streams; in-pool D2D is 241 GB/s r+w |
| pageable H2D | — | **6.2 GB/s at 14–64 MB, 1.3 at 512 MB — below the NVMe.** NVIDIA confirms this as a DGX Spark defect |
| 4 KB random | CPU-bound near 112k | 61.5k IOPS at **1.4 of 20 cores** — not CPU-bound, so io_uring's route to 112k is unproven here |

**The decisive measurement.** Replaying the DS4.1 engine's exact expert-read shape (14.45 MB in 4 MB
chunks, pinned staging lease, H2D on a per-thread stream, `stream.synchronize()`):

| experts in flight | with staging + H2D + sync | without |
|---|---|---|
| 1 | **3.73 GB/s** | 5.16 |
| 2 | 6.76 | 6.54 |
| 4–48 | 6.5–6.8 | 6.4–6.7 |

A decode layer misses ~0.5–1.6 experts, so the engine sits at the worst point — and it measures
**2.68 GB/s [stored]**, below even the C=1 figure. **Roughly half the load phase is host-side residue
and device idle, not transfer. The SSD is not the bottleneck; the schedule is.**

---

## Runnable today — no download, no root, hours each

| # | idea | risk | cost |
|---|---|---|---|
| 1 | **`--prefix-match-unit 64`** — finding 141's ~1,026 recomputed tokens/turn exist because the hash boundary is 1,600 back. Upstream's own hybrid measurement: 2nd-turn latency −28%. Supersedes the fp8-SSM trade that cost 127/2,504 predictions | none | 2 h |
| 2 | **`rejection_sample_method: "block"`** — block-level verification, optimal-transport-optimal, same output distribution. Never tried; `probabilistic` was null and `--use-fp64-gumbel` cost 4.6% | none by construction | 2 h |
| 3 | **`num_speculative_tokens_per_batch_size`** e.g. `[[1,2,2],[3,16,0]]` — our `cg.txt` shows MTP collapsing 18.9→10.7 tok/s at c=4 because MambaManager charges spec blocks per request | none in the k=0 band | 2 h |
| 4 | **Right-size the KV pool** — 30.99–33.47 GiB ≈ 967k tokens against a 262k max context. On this box the marginal GiB is worth more as absence of memory pressure. **Do NOT quantize KV**: our FP8-KV result (×1.72 pool, no speed), an independent GB10 q4_0 result (−37% decode at 110k), and vLLM's own Spark guidance all agree | none | hours |
| 5 | **`CUBLASLT_WORKSPACE_SIZE=131072`** — 338 calls of an *Ampere* WMMA kernel (5.0% of a 30k prefill) are being selected on a Blackwell part, a classic too-small-workspace symptom | bit-exact if it only reselects | 30 min |
| 6 | ~~**Host-pinned arena kernel test**~~ — **DONE 2026-09-13, and it changed the answer twice over.** An SM reads a `cudaHostAlloc` arena at **228–230 GB/s**, 81% of device memory's 281, not the ~59 GB/s that was predicted; and `O_DIRECT` into it is free while `cudaMalloc` returns `EFAULT`. But the bandwidth arithmetic then makes the zero-copy arena **near-neutral** (−4.4 ms of 318/step) and a *loss* above 95.2% coverage. See [gb10-arena-io-measured.md](gb10-arena-io-measured.md) | n/a | done |
| 7 | ~~**Stop chunking the expert read**~~ — **SUSPENDED, my claim, my error.** The 41% is real for *serial* chunking; `engine/experts.py:219` issues an expert's chunks **in parallel** on a second pool, so it does not apply here. Faithful re-run (`tools/expert_read_bench.py`, the engine's own reader) pending. What stands: one expert read is 5.6 GB/s, two concurrent reads 6.8 — the ceiling — so batching a layer's misses needs a queue depth of 2 | — | pending |
| 8 | **Suffix-drafting oracle**, CPU only — replay real agent trajectories against `SuffixDecodingCache` and get the acceptance curve for a *free* drafter before building anything | n/a | 1 d |

## The biggest single lever, and it is bit-exact

**Fused multi-step draft decode.** Derived from the MTP numbers we measured today:
`k=3 − k=2 = 12.8 ms` = one extra draft step, and `k=2 − no-spec = 23.7 ms ≈ 2 × 12.8`, leaving
**≈0 ms to widen the verify window from 1 to 4 tokens**. Drafts are expensive; verify width is free.

Cause verified in our own tree: `supports_draft_decode_metadata_update` defaults False
(`v1/attention/backend.py:608`) and only `flash_attn`/`triton_attn` implement it — **not** QSA. Our
prod log says so directly: *"Fused multi-step draft decode is not supported by attention backend(s)
QWEN4_EXP_EXP_QSA_STATE; falling back to rebuilding attention metadata between draft steps."* Plus
*"PIECEWISE cudagraphs are not supported for draft decodes"* — the target body runs under a graph, the
drafter does not.

If a graphed, metadata-stable draft step reaches ~4 ms: **k=2 → ~29.8 tok/s (+23%)**, k=3 → ~30.9
(+27%) and k=3 stops being a regression. **Bit-exact and mechanically provable** — assert every field
of the updated metadata equals the rebuilt metadata. **Half-day attribution test first**: time
`_multi_step_decode` vs `_generate_draft` alone; if the rebuild is under ~2 ms of the 12.8, the cost is
the eager forward and the fix is a cudagraph-mode line, not a builder.

## Needs the Qwen checkpoint restored (~22 min from PBS)

**Take the PLE off swap and put it on NVMe deliberately.** Today the box runs 119–120/121 GiB and the
serve *depends on* `/swapfile-fnext` paging out ~27 GiB of cold PLE rows — NVMe-backed already, via the
worst mechanism available: no `MADV_RANDOM`, no readahead control, no batching, and it OOM-killed the
worker twice. The PLE costs **16 rows = 2.5 KB per token** while holding **47.68 GiB of a 121 GiB
pool**. `mmap` + `MADV_RANDOM` + gather in `prepare_inputs` (row ids depend on token ids alone) deletes
the offload worker, the semaphore, the `SYS_PTRACE` requirement and the swapfile dependency.
`read_ahead_kb` is **128 [measured today]**, so `MADV_RANDOM` is load-bearing: without it the kernel
faults a ~64 KiB window per 160-byte row. Two same-silicon external points suggest **+15–25%**.
**Bit-exact.** Half-day kill test: mmap one shard, replay 2,000 tokens of the gather pattern, measure
warm latency and RSS.

Also here: **product-quantize the PLE** to ~8 B/row → **47.68 GiB → ~2.6 GiB**, the resident
alternative. Strictly better than HashK, whose polynomial re-hash + mean-pooling is stuck at a
reconstruction cosine of exactly **0.50** (the 1/√R limit) because merging colliding rows destroys row
identity. **Gate on FREE GENERATION, never teacher-forced loss** — CB3 measured *better* held-out
(1.5384/3.2087 vs 1.5705/3.3790) and then emitted `<!DOCTYPE>` until the cap.

> **NO LONGER DOWNLOAD-BLOCKED, and largely superseded, 2026-09-13.** All 40 layer shards were
> already on the backup server, and `expert_trace.py` streams them one at a time, so DS4.1's routing
> was traced on the box without the checkpoint ever being resident. The measurements are in
> [ds41-measured-2026-09-13.md](ds41-measured-2026-09-13.md) and they change this section's ranking:
> the real traffic is **0.582 GB/token at a 34 % arena**, not 271 MB at 44 %, so the path stays
> bandwidth-bound after any schedule fix; **`DSV41_IO_THREADS` and `DSV41_READ_CHUNK_MB` are both
> measured dead**; the engine's read path already gives 5.0–6.4 GB/s so the missing 2× is the
> per-layer blocking `pool.map` in `resolve()`; and **drafter-driven prediction moves from last place
> to near the top**, because the oracle headroom on DS4.1 is 5–6× the bytes rather than Qwen's
> ~5 points.

## DOWNLOAD-BLOCKED — needs DS4.1 (7 of 48 shards local, 521 GB free, thin margin)

Bit-exact, in order of value:

- **Fix the expert-read schedule.** Cheapest first: (a) raise `DSV41_IO_THREADS` from 12 — a 5-minute
  env A/B, and if it moves anything at all that *proves* the semaphore rather than the device was the
  limit; (b) stop holding the staging lease across the H2D + `stream.synchronize()` — hand the buffer to
  a completion thread so the read pool never waits on a CUDA stream; (c) `preadv`+`O_DIRECT` straight
  into a `cudaHostAlloc` arena, deleting the upload entirely — **measured, viable, but near-neutral**;
  (d) issue a whole verify block's misses
  for a layer as one batch — the engine already sees 20.96 distinct experts/layer, so the union is
  naturally wide, it just is not in flight together. Takes the full-quality unpruned path from 2.68
  toward **10–16 tok/s**, against the pruned config's 23.5.
- ~~**Adaptive LRU warmed by the prompt's own routing.**~~ **Simulated 2026-09-13 (ds-09…ds-11), and
  the prefill-warming half is closed.** It is real but small — +0.8 pp at a roomy arena, +4.7 pp at a
  tight one, **−1.7 pp on short prompts** — and it needs 256–1,024 prompt tokens before it pays at all.
  Crucially it is *dominated by doing nothing but enlarging the dynamic fraction*, which is free. What
  survives, and is bigger: **the static/dynamic split is the lever**. At a roomy 44% arena the optimum
  is 25–50% dynamic, worth +1.7…2.0 pp over the 10% transient allowance 0xBakeer ships; at a tight 20%
  arena a **plain LRU with no static core beats the frozen keep-set by +4.7 to +25 pp**. Ship the split
  as a runtime knob.
- **Prefetch n-gram rows the moment token ids exist.** 48 rows × 264 B per token, addressed by token id
  alone; a verify block's addresses are known the instant the drafter emits. 288 random 264-B reads is
  4.7 ms at the device's 61.5k IOPS, against a stored 58.9 ms step of which 48.5 ms is the Engram wait —
  so that wait is scheduling, not capacity. Warm steady-state gain is probably small; the value is cold
  start and long-context prefill, and it becomes load-bearing once the expert reads get 2.2× faster.
- **Drafter-driven expert prefetch.** Ranked last and pushed lower by today's own numbers: at 12.8 ms
  per draft step and ~0 ms to widen the verify window, buying lookahead by drafting *deeper* is
  expensive while buying queue depth by drafting *wider* is free, so the batching item above dominates
  it on the same axis. One layer of lookahead cannot cover an NVMe fetch anyway (6 experts × 14.45 MB ≈
  14 ms at 6 GB/s); the only published cold-NVMe number is +8%. Kill it offline for a day's work: replay
  a trace, count how many of the target's layer-L experts the drafter already names for that position,
  and stop under ~60%.

## Small quality loss — real candidates, and the gate they must pass

Both of these trade accuracy for residency, so neither may be judged on held-out loss alone. This repo
has already been burned by exactly that: CB3 3-bit measured *better* held-out (1.5384/3.2087 vs
1.5705/3.3790), was predicted to three decimals by the simulator, and then emitted `<!DOCTYPE>` until
the output cap. The gate is NLL/token **and** the five-prompt free-generation check (distinct-token
ratio > 0.25, structural intactness) **and** NIAH **and** a real agent turn.

- **Product-quantize the PLE at ~8 B/row** (Qwen; runnable today). Trainless k-means: 160 dims → 8
  subvectors × 20 dims × 256 centroids, 47.68 GiB → ~2.6 GiB, dequant on gather. This is the resident
  alternative to the swap-removal above: no page-cache dependency and no cold-TTFT cliff, at the cost of
  a real quality question. It beats HashK by construction — HashK's re-hash plus mean-pooling sits at a
  reconstruction cosine of exactly 0.50, the 1/√R mean-pooling limit, because merging colliding rows
  destroys row identity, while PQ preserves it. Ready-made fallbacks already exist and are measured by
  others: trainless INT4 g16 at 32 GB and NVFP4 g16 at 28.8 GB (knowledge 92.2–92.9 vs base 92.2,
  tool-calling 77.7–78.7 vs 79.2), and an NVFP4 PLE at 26.8 GiB behind a plugin. Half-day kill: fit the
  PQ, reconstruct, measure per-head cosine offline; below ~0.9 there is no reason to prefer it to the
  bit-exact route.
- **Hot/cold mixed precision for DS4.1 experts** (download-blocked). Not more uniform compression —
  that is spent, the checkpoint is FP4-native at ~2.6 effective bits and CB3 at 14.45 MB is itself under
  a cloud from the degeneration episode. Instead keep the hot ~44% at native FP4 and demote only the
  cold tail, i.e. exactly the experts that will be streamed anyway. Published shape recovers 77.57%
  average against INT4's 78.11% and static INT2's 73.09% at the same budget, no retraining. Cuts the
  miss bytes only: 271 MB/token → ~205–230, i.e. 13–16 tok/s → ~16–20.

An adjacent lever worth more than drafter prefetch: capacity budgeting that drops experts whose
*aggregated* routing weight across a whole block is negligible. Much cheaper in quality than cutting
global top-k, which we rejected at k=5 for +0.025 nats on prose.

## Killed outright

- **GPUDirect Storage.** NVIDIA's release notes verbatim: *"On DGX Spark, GPUDirect Storage is
  supported only in compatibility mode. **Do not load `nvidia-fs`, because doing so can result in
  errors.**"* The package `nvidia-spark-default-remove-nvidia-fs-pkg` is installed specifically to stop
  it auto-loading, and there is a known nvidia-fs NVMe bug on Linux ≥6.17 — we run 6.17.0-1031. Even if
  it loaded, `cudaMalloc` memory is not coherently reachable by PCIe on this box, so cuFile compat is
  one memcpy *worse* than plain O_DIRECT into `cudaHostAlloc`.
- **io_uring.** Its exotic flags are 4 KB-IOPS features. Our large-read path saturates at 6.9 GB/s with
  two plain threads; the small-read path does 61.5k IOPS at 1.4 of 20 cores. Neither n-gram path needs
  more than ~1.2k IOPS.
- **The M=7503 blockwise-FP8 collapse** (see below) and, with it, "close the blockwise-vs-per-tensor
  gap": both rested on `fp8bench.txt` from 2026-09-03. Re-measured: **163–170 TF flat**, and blockwise
  is now within ~6% of per-tensor.
- **Tree drafting** — removed from vLLM main, feature request closed as not planned.
- **NVMe I/O from inside a CUDA graph** via `cudaLaunchHostFunc`. Legal and safe across replays,
  but the stream counts as *idle* for the duration of the callback, host funcs across streams may
  serialize onto one CUDA worker thread, and the Python bindings are broken (only the first replay
  calls the host function, later replays segfault). Use a host node to *signal* an existing I/O
  worker, never to hold a `pread`.

## An unresolved conflict worth one A/B

Verify-width cost. Qwen **[measured by us today]**: ≈0 ms per verify position. DS4.1 **[stored,
2026-09-11]**: block 4 → 110.7 ms, block 6 → 124.8, block 8 → 137.0, i.e. **+13.1 ms per +2
positions**. Both can be true — Qwen's experts are resident and small, while widening a DS4.1 block
adds ~7 distinct experts per layer × 40 layers of arena traffic. But it decides whether wider blocks
are free or are the main cost, so it should be disambiguated rather than assumed.

## What this exercise actually established

Three agents, three corrections to context I supplied:

1. **`fp8bench.txt` (2026-09-03) no longer describes this build.** Its M=7503 collapse — the largest
   single lever in the whole survey, an estimated −35 to −40% TTFT — does not reproduce. One hour of
   checking avoided a 4–6 day project, and killed a second idea resting on the same file.
2. **"Agentic speed is TTFT-bound" is the dense 27B's rule, not Flash-Next's.** Warm, TTFT is 0.59 s
   against 5.35 s of decode — **~10% of the turn**, not 53–69%. I applied the right rule to the wrong
   model.
3. **Four hardware constants in our notes were wrong**, including two I had repeated confidently in
   this session (the 273 GB/s host bounce and "queue depth is the whole game").

**The common cause is that our stored measurements carry no build stamp or date at the point of use.**
Every one of these was a true measurement that stopped being true. Fix: stamp `notes/data/*` with the
vLLM version and date, and treat any figure older than the last vLLM bump as needing re-measurement
before it enters a decision.
