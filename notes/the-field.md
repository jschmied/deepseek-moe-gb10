# DeepSeek landscape for one GB10 — survey 2026-09-12

Verified against the HF API and each repo's raw `config.json` rather than blog copy. Figures marked
*inferred* were not confirmed at source.

## V4.1-Flash exists, and it is two days old

`deepseek-ai/DeepSeek-V4.1-Flash`, published **2026-09-10**, MIT.

| | |
|---|---|
| params | 552B backbone + 196B Engram (HF index 763.2B) |
| active/token | **8B prefill / 16B decode** |
| routed experts | **384 + 1 shared**, 6 active |
| expert intermediate / hidden | **2304 / 5120** |
| layers | 40 (20 causal-encoder + 20 decoder) |
| native context | 1,048,576 |
| on disk | **475.3 GiB** |

New architecture (`DeepseekV41ForCausalLM`, `model_type: deepseek_v41`), not a V4 variant: CED
encoder–decoder, CSA2 sparse attention, Engram lookup memory (`engram_layer_ids [1,14]`, 16M-entry
vocab), 890 B/token global KV, and a **built-in DSpark drafter** (`dspark_n_routed_experts: 128`).

**It ships already quantized**: `quant_method: fp8`, `expert_dtype: fp4`, `scale_fmt: ue8m0`,
32×32 weight blocks (V4 used 128×128).

Others: V4-Pro-0813 (1.65T), V4-Flash-Vision-Exp (304.6B), V4-Flash-0731 (304.2B, 13B active,
256+1 experts). No V4.1-Pro, no V4.1-Flash-Base, no V4.2.

## Two blockers, and they are independent

**1. Size.** Nothing under 128 GB exists for V4.1 — the smallest anything is a 198.3 GiB MLX REAP
2-bit. NVFP4 ports are 386–491 GiB.

**2. It cannot start on our hardware.** vLLM merged V4.1 on main (#56214, 2026-09-11), but
[**#56461**](https://github.com/vllm-project/vllm/issues/56461) (open, 2026-09-11) reports
**DeepSeek-V4.1-Flash cannot serve on SM120/SM121 (GB10)**: the SWA cache is hardcoded to
`block_size=32` while FlashInfer's SM120 sparse-MLA decode only has a page-64 kernel, and ratio-1
layers hand DeepGEMM SM120 `block_kv=128` where it accepts 64. Reproduced on a DGX Spark with
`--load-format dummy`. The official V4.1 recipe is verified on H200, GB200, GB300, MI350X — **not
GB10**.

### The opening

Fix PR [**#56509**](https://github.com/vllm-project/vllm/pull/56509) is **open and hardware-untested** —
its author says they are "relying on CI and upstream review for SM120 validation". We have the
hardware they lack.

And the size blocker does not apply to that test: #56461 was reproduced with **`--load-format dummy`**,
which allocates the architecture without real weights. So the startup path on sm_121 is testable here
for the price of a venv and a few minutes, with no 475 GiB download. That is the single cheapest
high-value thing in this survey.

## What could actually run here

Realistic target is a pruned **V4-Flash-0731**, not V4.1:

| repo | size | format | GB10 claim |
|---|---|---|---|
| `Baekpica/…-120B-REAM-104E-NVFP4` | **65.3 GiB** | NVFP4**A16** weight-only, REAM to 104 experts | claims GB10-validated, but via a **vLLM 0.26.0 fork**; card self-declares a smoke-test arithmetic failure / repetition |
| `0xSero/…-spark` | 99.5 GiB | REAP + EXL3 3.0bpw | TP4 rank-sliced; **stock vLLM cannot load it** |
| `Laplace1313/…-EXL3-3bpw` | 101.8 GiB | JA-tuned REAP | card states stock vLLM/SGLang cannot start it |
| `unsloth/…-GGUF` UD-Q3_K_XL | 119.4 GiB | GGUF | llama.cpp only |

Every mainstream NVFP4 (nvidia, amd, RedHatAI: 148–164 GiB) is over budget.

## Expert compression, and what is actually implemented

- **REAP** (router-weighted expert pruning, [arXiv:2510.13999](https://arxiv.org/abs/2510.13999)) and
  **REAM** (merge-then-prune, [arXiv:2604.04356](https://arxiv.org/html/2604.04356v1)) are what every
  sub-128 GB build above uses.
- **D²-MoE**, **MoBE**, **MoE-I²**, **RS-MoE** — research code only.
- **None of these are vLLM features.** A search of vllm-project/vllm for REAP / expert-merging /
  D2-MoE returns **zero** issues or PRs. Pruning and merging work because their output is an ordinary
  smaller MoE that loads through the normal path. **Delta/low-rank methods are not like that** — they
  need a custom vLLM layer, which is the cost my `moe-expert-compression.md` note was pointing at.

## One thing this changes about the shared-base idea

DeepSeek **already has a shared expert** (384 **+ 1**), read once per token per layer. So the
"evaluate the base once regardless of how many experts fire" property is *already* in the
architecture; a D²-MoE-style base would be a second, different shared object (a Fisher-weighted
merge of the routed experts). The gain is therefore strictly over the routed-expert traffic, which is
what the original argument claimed — but it means the architecture is not naive about this, and the
easy part of the win is already taken.

Also: V4.1's experts are **already `fp4`**. The FP16-baseline caveat in
`moe-expert-compression.md` applies with full force.

---

## The V4.1-on-Spark landscape, verified 2026-09-12 ~15:10 UTC

Correction to our own framing: we had been treating 0xBakeer as *the* V4.1-on-Spark repo. It is the
primary repo for the **single-Spark** problem specifically — its own description says "on a single
DGX Spark (GB10): resident hot experts + NVMe streaming, DSpark" — but it is neither the first nor
the only line of work. All figures below are from the GitHub API, not from the repos' own claims.

| repo | focus | created (UTC) | last push | ★ | forks |
|---|---|---|---|---|---|
| tonyd2wild/…-vLLM-DGX-Spark | 3–4× Spark, vLLM | **09-10 07:51** | 09-12 02:50 | 56 | 9 |
| 0xBakeer/deepseek-v41-flash-spark | **1× Spark** | 09-10 18:40 | 09-12 08:36 | 65 | 7 |
| MiaAI-Lab/…-DGX-Sparks | 3–4× Spark, SGLang | 09-11 07:33 | **09-11 11:00** | 60 | **15** |
| magicbear/…-SGLang-DGX-Spark | 4–8× Spark, SGLang | 09-11 01:06 | **09-11 08:10** | **1** | 0 |
| rhys101/…-SGLang-DGX-Spark-4-NVMe | 4× Spark, SGLang | **09-12 14:02** | 09-12 15:01 | 0 | 0 |

Tony's repo does predate Bakeer's by ~11 hours, exactly as the user said.

**Three calibrations the star/push data adds:**

1. **magicbear has 1 star, 0 forks and has not been pushed since 09-11 08:10.** The user's table
   credits it with "extensive benchmarking and kernel work"; whatever is in there, it has essentially
   no third-party validation and is not being maintained. Treat its numbers as single-source.
2. **rhys101 was created today at 14:02 UTC and pushed at 15:01** — about an hour old when we looked,
   0 stars. Its "82.73 code / 52.62 prose tok/s C1 on four Sparks" is therefore a brand-new,
   unreplicated claim. Worth reading for method, not for citing as a figure.
3. **MiaAI-Lab has the most forks (15) but is a day stale.** Forks-to-stars ratio suggests people are
   actually running it.

Only Bakeer, Tony and rhys101 pushed anything today.

**How we will use them.** Per the user: monitor all, and for each finding ask *can this be
transplanted into a single-node engine?* — the five areas that transfer despite TP/EP being
unavailable to us are DSpark scheduling, FlashInfer kernels (sm_120/121), Engram/NVMe handling,
prefill memory management, and FP8/MXFP4 kernels. Anything whose benefit comes from splitting the
model across boxes does not transfer and should not be logged as an option for us.

Add to the watch list: tonyd2wild, MiaAI-Lab (we already watch #19/#23), magicbear, rhys101.

## Transplant survey of the four non-Bakeer repos (2026-09-12)

All figures below are **as reported by those repos**, read from their trees; we have verified none of
them on our box. Marked TRANSPLANTABLE only where the benefit does not come from splitting the model
across Sparks.

### Ranked candidates for a single Spark

1. **`b12x` MXFP8/FP4 kernels for small-M dense projections — the biggest measured win anywhere.**
   `MiaAI-Lab/adapter/mxfp8_b12x.py` + `README.md:194,204-208`: ~230 dense FP8 projections per step at
   M=6 go **52 ms (Triton) → 50 (FlashInfer CUTLASS SM120) → 17 ms (b12x warp-level MMA)**; her total
   step 118 → 82 ms, largest single contributor. Cause: the checkpoint uses 32×32 `ue8m0` block
   scales and CUTLASS pads M 6→128. Pure kernel selection, **zero TP content**.
   `local-inference-lab/b12x` is a live SM120/121 CuTe-DSL library (213★, pushed today) whose
   `AGENTS.md` states the contract we have been measuring by hand: *"W4A16 means BF16 activations with
   inline FP4/NVFP4 weight dequantization."* Relevant to us **independent of V4.1**.

2. **The GB10 hidden slow state — see the separate entry below. Highest expected value for us.**

3. **`row_store.cpp` + `cudaLaunchHostFunc`: host-side IO *inside* a CUDA graph.**
   `MiaAI-Lab/adapter/row_store.cpp` (byte-identical in rhys101), 429 lines of C, no CUDA in the hot
   path. Packed 264 B rows (FP8 weight + E8M0 scale adjacent), 4096 B header, row 0 page-aligned ⇒ one
   `pread` per miss. Queue depth sized from a measured GB10 NVMe curve (**~3.5k IOPS at QD1 → ~112k at
   QD64**) ⇒ 96 IO threads; `kChunk = 1` deliberately, to keep in-flight reads high. Costs ~1–3 ms per
   decode step for both Engram layers against an ~82 ms step. rhys101 shows it bit-exact across 12
   changing-input graph replays. **This is the general answer to "graph capture vs a host lookup" and
   applies to any offloaded table — our PLE included.**

4. **`top_k_per_row_decode` beats `persistent_topk` on GB10, with numbers.**
   `tonyd2wild/patch/sm12x-indexer-topk/RESULTS.md`: exact index-set match vs `torch.topk` at widths
   600–300,000 and **1.6–3.6× faster in every cell** (6 rows @ 4,096: 96.1 → 59.8 µs; 48 rows @
   65,536: 810.6 → 337.7 µs). Structural cause: wants **128 KB smem, GB10 has 99 KB**, and it
   oversubscribes 48 SMs above ~48 CTAs/row. **This is our ground** — det-217 measured the same
   101,376 B optin ceiling from the other direction, and det-222/224 were a shared-memory budget bug
   in exactly this kernel family.

5. **`--speculative-dspark-align-verify-tokens-to-graph-tier`: free verify tokens.**
   `MiaAI-Lab/boot.py:321-323` — *"Fills each step's verify window up to the cuda-graph tier the
   forward is padded to anyway: free verification at the same cost."* Engine-agnostic idea. **Every
   one of the four leaves DSpark's ragged-verify scheduler off or inert** — rhys101's headline 82.73
   tok/s run has `speculative_dspark_sps_table_path: null`, and Tony disables adaptive verification
   deliberately (padded speculative batches can hang SM120 sparse MLA, FlashInfer #5015, open).

**Runners-up:** `prefill_empty_cache.py` (chunked prefill reserves memory ~quadratically because the
caching allocator cannot serve a slightly larger next chunk from a smaller cached block — 8 GiB
reserved with nothing live after 64 chunks; verbatim transplantable); the `force_deep_gemm_metadata`
one-liner for ratio-1/2 indexers on SM120; and a unified-memory weight-loading deadlock on GB10
(`cudaMemcpyAsync` on mmap-backed CPU weights stuck in `pthread_rwlock_wrlock`; 778 s → 576 s).

**Explicitly NOT transplantable:** rhys101's PR #36655 "native H8/H16 decode" backport buys nothing at
TP1 (with all 64 heads local the native path *is* the padded-64 path), and Mia's NCCL-buffer work
(4.7 GiB → 139 MB pinned) is purely multi-node.

### Two independent confirmations of our own findings

- **Fused MoE finalize.** `MiaAI-Lab/README.md:249-256` ships `SGLANG_FLASHINFER_MOE_FUSED_FINALIZE=0`
  for atomic bf16 adds in the fused finalize, autotuner-selected for one bucket, ~1 nat of first-token
  logprob drift, ~0.3 ms/step to disable. That is **our PR #54948 finding reached independently on a
  different engine.**
- **FlashInfer autotune workspace OOM at high `gpu-memory-utilization`.** magicbear had to drop 0.80 →
  0.65 on vLLM or the driver returned `NV_ERR_NO_MEMORY`. Matches our
  `flashinfer-jit-oom-after-driver-upgrade` memory.

### Where the field contradicts 0xBakeer

He names none of the four repos. Beyond the slow state, he appears unaware of the b12x dense-GEMM
route, the in-graph `cudaLaunchHostFunc` Engram lookup, and the measured top-k timings (he knows the
128 KB vs 99 KB *constraint*, `docs/gotchas.md:124-131`, but has no numbers). Two direct conflicts:

- **O_DIRECT for Engram rows.** `gotchas.md:74-79` says do not — "48 random 264-byte reads per token,
  O_DIRECT on those is all overhead" — and uses buffered `preadv` + a small row cache. Mia/rhys101 do
  exactly that, but on a **repacked page-aligned shard** where each row is one aligned read. Different
  premises, reconcilable; on unified memory his buffered reads cost page cache that *is* GPU memory.
- **Engram row cache.** He keeps one; Mia sets `DSV41_CACHE_GIB=0` (~0 % reuse) and rhys101 refuses to
  boot with a cache configured, with telemetry showing **0 hits / 31,582 misses**.

### Corrections to our own landscape table

- **MiaAI-Lab is a 3-Spark repo.** `README.md:136-137`: every published measurement is TP3; the TP4
  profile is config-validated only, never booted. Do not cite Mia numbers as 4-Spark.
- **magicbear's content is the most rigorous kernel A/B of the four** despite 1★ — it refuses to
  attribute a prose regression to the kernel because acceptance moved (2.08–3.92 prose vs 5.5–5.88
  counting). Judge it on content, not reach.
- **rhys101 is a spin-off** of `rhys101/…-vLLM-DGX-Spark-8`, and its 82.73/52.62 figures are
  well-documented (n=5, sd 0.179) with prompt bytes SHA-256-pinned identical to Tony's. Its
  **prefill** advantage over Tony's vLLM on identical prompts is arguably the bigger result —
  2,245–3,032 vs 902–1,539 tok/s — and given our `agentic-speed-is-ttft-bound` finding, that is the
  half that would matter to us.
- **Acceptance is dominated by prompt content, not configuration**: 3.55–3.89 across three independent
  4-Spark fleets, ~3.0 on Bakeer's single box, but 5.5–5.88 counting vs 2.08–3.92 prose within one
  run. **No cross-repo tok/s comparison is meaningful unless matched on prompt** — the same lesson as
  our `acceptance-is-not-quality`.

---

## Field check 2026-09-14 — one new item, and the reason the others moved

**No new faithful full-router one-Spark competitor.** 0xBakeer is unchanged at `8b68fdd`
(2026-09-12); sayyidfareed's K154 has had only install/profile cleanup, no new performance work.
Our branch is still the only one running the unpruned 384-expert router on one box with a native
3-bit on-disk expert cache.

### What is new

| item | status | what it is worth to us |
|---|---|---|
| **SGLang #39370** | opened 2026-09-14, on `dsv4.1` not `main` | supersedes #39301/#39336; combines small-row DSpark decode with **large-prefill** tuning — packed router ids, split argmax, ≤8-row MXFP8 epilogue, parallel FlashMLA scheduling, bounded candidate graphs, tuned prefill GEMMs, Q RoPE/store, WO-A layout, mHC combine/RMSNorm. Combined numbers still pending |
| **FlashInfer #5191** | open, author asks for GB10/SM121 validation | the DSV4.1 shared-expert projection at **M=6**: CUTLASS 15.53 µs vs native b12x **3.242 µs, 4.79×** on SM120 |
| **FlashInfer #4955** | merged | native **NVFP4 sparse MLA on SM120/121**, 384 B/token; 1.42–1.67× prefill kernel, 1.31× decode kernel |
| **FlashInfer #4661** | merged | FMHA v2 prefill **validated on DGX Spark / SM121**, 26 targeted tests incl. graph and async-enqueue |
| **DeepGEMM #394** | still open | FP4 packet layout, +9.5 % at 1 token, +4.4 % at 8, ~0.3 % at 8k, bit-identical |

### Two of these we can size against our own measurements

**#5191's shape is in our family, and M=6 is not going to move.** Confirmed from the checkpoint:
the shared expert is `w1 (2304, 5120)` / `w2 (5120, 2304)`, `n_shared_experts = 1`, so N = 5120 is
our `dim`. And **M = 6 is exactly `T_VERIFY`** — the checkpoint-native verify width, which our own
§9 shows is the optimum (width 8 is 20 % worse). So the shape they are optimising is the shape we
will keep. Their K = 576 is 2304/4 and looks like a split-K factor rather than our raw GEMM; read
the PR before claiming it is bit-for-bit ours.

Why it is still not next: §8 measured the GPU **31–33 % busy**. A 4.79× on a GEMM that sits hidden
under expert waits buys close to nothing until the streaming path is fixed. Same discount applies to
#4661 and to the DeepGEMM packet layout.

**#4955 is a speed item, not a memory item — and the memory case is off by an order of magnitude.**
Measured from the allocation at `max_seq = 32768`:

| | |
|---|---|
| `ckv` + `ik`, all four `kv_source_layers` `[2, 8, 14, 20]` | 104.9 MB |
| window ring, 40 × 4096 × 512 × bf16 | 167.8 MB |
| `mtp_win` | 12.6 MB |
| **entire cache state** | **285.2 MB = 19.7 CB3 slots** |

Quantizing *all* of it to NVFP4 frees ~214 MB — **15 CB3 slots, 0.4 experts per layer**. Only four
layers hold compressed KV at all; everything above layer 20 projects its global KV from layer 20
under CED. Any argument of the form "free a few hundred MB and hold more experts" is dead on this
architecture, including the version of it in our own TODO. What #4955 may still be worth is the
**1.42–1.67× prefill kernel** — though note their own serving numbers are much weaker than their
kernel numbers (9 % at 8192/1, 3.67 % at 8192/256, steady decode 0.57 % *slower*), which is the same
idle-GPU discount we measure.

### The sequencing point worth keeping

SGLang #39370's large-prefill half becomes interesting **after** layer-major, not before. Our oracle
says the transpose removes 243 GB of redundant expert I/O from an 11.3k prefill (327.8 → 84.5 GB,
3.88×); once that is gone, dense/attention/mHC is the majority of TTFT rather than a rounding error,
and kernel tuning on the prefill path starts paying. Before then it is optimising the 41 % of prefill
wall that delivery does not already own.

**Order, as it stands after the night's measurements:** prompt cache validation → re-profile decode
on the fixed counter (§10c was withdrawn) → layer-major prefill → loader pipeline → profile →
FlashInfer SM121 prefill/MLA and #5191 → CB2/remap experiments → packet layout.
