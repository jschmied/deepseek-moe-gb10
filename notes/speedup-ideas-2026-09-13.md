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
| NVMe large sequential O_DIRECT | 5.5 GB/s, "queue depth is the whole game" | **6.9 GB/s**, saturated at 4 threads × 4 MB, and **4.42 GB/s from ONE thread** |
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
| 6 | **Host-pinned arena kernel test** — allocate one tensor `cudaMalloc` vs `cudaHostAlloc`, time a Triton MoE kernel over each. Decides whether a zero-copy NVMe→SM arena is possible at all | n/a | 30 min |
| 7 | **Suffix-drafting oracle**, CPU only — replay real agent trajectories against `SuffixDecodingCache` and get the acceptance curve for a *free* drafter before building anything | n/a | 1 d |

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

## DOWNLOAD-BLOCKED — needs DS4.1 (7 of 48 shards local, 521 GB free, thin margin)

Fix the expert-read schedule (2.68 → ~6 GB/s, taking the full-quality unpruned path from 2.68 toward
**10–16 tok/s**); adaptive LRU warmed by the prompt's own routing; n-gram row prefetch; hot/cold mixed
precision; drafter-driven expert prefetch.

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
