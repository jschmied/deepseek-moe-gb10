# The idea ledger

**Why this file exists.** Ideas have been lost several times — closed in a chat, never written down,
then re-derived weeks later or, worse, re-closed on different evidence. **Nothing leaves this file
until it is proved bad, and when it is, the proof is recorded next to it.** An idea with a null
result stays here with the null attached; that is the whole point, because a null measured under
the wrong conditions is not a closure (see the three entries marked RE-OPENED below, all of which
were "settled" and were not).

Status vocabulary, used strictly:

| status | meaning |
|---|---|
| **SHIPPED** | on in production, with the evidence that put it there |
| **READY** | verified, not yet switched on, and why not |
| **OPEN** | not yet measured, or measured inconclusively |
| **NULL** | measured, no effect, under stated conditions — NOT the same as closed |
| **CLOSED** | measured and refuted, with the mechanism understood |
| **RE-OPENED** | was closed, and the closure turned out not to hold |

Last reviewed 2026-09-17.

---

## SHIPPED

| idea | evidence |
|---|---|
| `DSV41_EVICT_POLICY=age_over_freq` | +8 % real tokens over LRU (job 360). age/(1+use count). |
| CB3 3-bit codebook experts | the serving format; 13,774,848 B/record |
| `DSV41_DENSE_FP4=attn,wo_a`, `DSV41_HEAD_FMT=fp8` | dense/attn FP4, fp8 head |
| `DSV41_HC_KERNEL=1` (split-K fp32 HC mixing) | on by default in code |
| Out-of-order H2D completion | job 375 null on speed (+0.4 %), kept because it is simpler, tested, and removes a failure mode at higher H2D depth |
| Immutable `LoadTicket` + two-deque work queue | review item 6; simplification, not speed |
| `DSV41_SHARED_FIRST=1` | **2026-09-17.** Bitwise identical (0.000e+00 on logits/h/pre_mix/y over 12 real decode steps). +0.98 % over 12 arms (jobs 410+415, t≈2.25, p≈0.05). Shipped under "neutral-or-better and opens a later gain": the routed-MoE split reuses this seam. |

## READY — verified, not switched on

| idea | why not on |
|---|---|
| `global_barrier=False` (v2 `Policy`) | +0.97 % over 6 arms, token-equal (job 420). **Not applicable to the production server**, which runs the v1 fastdecode path; this is a v2 `Policy` field. Keep for the v2 engine. |
| `ARENA_GB=86` | +10.1 % over 79 GB (jobs 380/381), replicated. **Already effectively realised**: the server's auto-sizer yields ~84.5 GB at `KEEP_FREE_GB=12`. Setting it explicitly at `MAX_SEQ=32768` is UNMEASURED — our arms ran `max_seq=8192`, and the job-150 incident (4 GiB free → watchdog stopped the server) is exactly this failure mode. Measure under production context before pinning. |

## OPEN — not yet measured, ranked by expected value

1. **Co-activation-ordered file layout.** Largest non-pruning number in the field: reads/token
   1418 → 775 → ~370 (2.23× cold decode I/O, llama.cpp #18758); 36× fewer page faults from
   expert-contiguous layout (#27149). For us the mechanism differs — our reads are already single
   contiguous 13.77 MB O_DIRECT extents, so there are no page faults to save — but if co-activated
   experts were ADJACENT on disk, the ~2 misses per layer could merge into one larger read.
   **Distinct from the CLOSED "read coalescing"**, which was about merging the 6 planes *within*
   one expert (already done, 6→2, overread 0.06 %).
2. **Cross-layer gate.** Run layer L+1's *existing* router on layer L's residual. Computes the real
   router on a stale input; it infers nothing from routing statistics, so **our entropy result does
   not bear on it** — that is the one thing separating it from everything in CLOSED below.
   Evidence: 97 % accuracy / 4.1× decode at 60–64 experts (arXiv 2502.12224); HOBBIT ~90 % at two
   and three layers ahead; but at 256 experts the only measurement is 73.6 % recall and **no
   throughput gain** (colibri #200 — because their cache held 2 experts per layer; we have the
   opposite problem). Nobody has measured it at 384. **Testable offline against traces we already
   have, before any engine work.**
3. **Fused decode attention** (`DSV41_FUSED_ATTN=1`). In-tree, defaults OFF. Targets `graph A` =
   46.1 ms/step = 37 % of device time, which **can never be overlapped** (A(L) must finish before
   layer L's expert ids exist). No scheduler change reaches it; only a faster kernel does.
   Job 430 runs the paired-NLL verdict. Not bitwise, so quality gates it.
4. **Mixed-precision miss path (HOBBIT's mechanism).** Second lower-bit copy per expert; on a miss
   whose router weight is low, read the small record. Bytes come straight off the critical path.
   HOBBIT attributes 1.19–1.57× to dynamic precision alone vs ~1.05× to prefetching, ≤1 % accuracy
   cost, gated by an importance proxy correlating 0.99 with true output magnitude. Scales with our
   13.77 MB record size. Not a global quant change — a selective miss tier.
5. **AdapMoE adaptive gating.** ~25 % fewer experts activated per token, no reported accuracy loss,
   1.35×. Cuts our 19 reads/token directly.
6. **Least-Stale eviction** (SpecMD). Two priority queues: evict experts from previous forward
   passes first, protect current-pass and prefetch-selected, FIFO by layer position within each.
   Collision miss at 5 % capacity 1.6–1.9 % vs LRU 4.5–12.6 %. Cheap drop-in A/B vs age_over_freq.
   Caveat: SpecMD is emulated (A100 with throttled bandwidth), not real storage.
7. **Overfetch top-(k+δ) with confidence-adaptive δ.** ETH measured 98–99 % hit from
   over-provisioning alone. Fits us specifically because the pipe is idle 44–84 % of the time, so
   the bandwidth is free. Pairs with (2) — alone there is nothing to overfetch, since the current
   layer's top-6 is already exact.
8. **Engram row layout / IOPS.** 576 preads per step to deliver 76 KB: two reads per row from
   widely separated regions (256 B weights at `w_off + r*256`, 8 B scale at `s_off + r*8`), and the
   8-byte read pulls a whole 4 KB page. Row cache is `cache_max = 200_000` (52.8 MB), fill-once, no
   eviction, **and has no hit counter** — so nobody knows whether it works. Counter staged.
   Bound it first: preload every touched row into RAM (45.6 MB over 600 steps) so engram device
   traffic is zero and logits stay identical; if `wait_reads` does not move, the 203 GB interleave
   rewrite is unjustified.
9. **Saliency-ranked keep-sets.** `gate_weight × output_norm`, not frequency. Upstream's table:
   at 75 % keep 0.440 vs 0.082; at 50 % keep 0.429 vs **0.000**. This matters because our pruning
   closure may rest on the wrong ranking — see RE-OPENED.
10. **`DSV41_CB3_SCRATCH_SLOTS=384`.** Our records say "unpack scratch 384" SHIPPED, and it is not
    set anywhere; the default is 0. Either the record is wrong or the setting was lost. Job 10
    measured ttft 54.6–55.4 s with 384 vs 58.6 without. **Re-verify and then actually ship it.**
11. **Device read depth.** Our "4.89 GB/s matches the bare device" rests on a TWO-POINT ladder
    (1 and 2 readers), which cannot see a knee further out. A contended sweep gave 3.19/3.78/5.05/
    4.76/4.78/4.87 across depths 1–24 — flat from 4. Published Spark figures are 11.1 GB/s, but for
    the **internal 4 TB** variant; this box is `ESL01TBTLCZ` on a 916 GB partition, a different
    part. Job 425 sweeps depth cleanly and records the drive.
12. **Concurrency.** More requests in flight means more misses per layer, raising read depth with
    no prediction at all. The one lever the entropy result does not touch. Changes the product
    (batch serving), so it is a product decision, not just an engine one.

### Added 2026-09-17 from the NVIDIA NVFP4 quant-map comparison

`nvidia/DeepSeek-V4.1-Flash-NVFP4` quantises ONLY the routed experts (NVFP4 W4A4, group 16, two-level
E4M3-per-16 + F32-per-tensor). Its exclude list, verbatim from `hf_quant_config.json`:

    "exclude_modules": ["*.attn.*", "*.ffn.shared_experts.*", "head", "mtp.*"]

Checked against ours, and **we diverge on two of the four, not three** -- our `_FP4_GROUP_OF` puts
`ffn.shared_experts.*` in a `shared` group that `DSV41_DENSE_FP4=attn,wo_a` does NOT enable, so our
shared expert is FP8 and already agrees with them. (A research summary claimed we CB3 it; we do not.
The CB3 arena holds routed experts only.)

13. **FP4 on the attention path.** OPEN. They had attention at MXFP8 and went to 4 bits on ZERO
    attention tensors on this architecture; we FP4 `wq_a/wq_b/wkv/wo_b` plus `wo_a`. `wo_a` is
    `[8192,4096]` inside a low-rank `o_lora_rank: 1024` split, so its error is not averaged away by
    a wide reduction. Never measured. Job 435 runs paired NLL, FP8 vs FP4, with the temperature
    sweep, and reports what the FP8 arm costs in arena slots so the trade is explicit.
14. **LM head at BF16 instead of FP8.** OPEN, low priority. They exclude `head` and ship BF16
    `[129280,5120]`; we use `DSV41_HEAD_FMT=fp8`. Our own `lm-head-precision-and-humming-012` note
    already cleared head activation precision as a loss source, so this is a cheap alignment rather
    than a suspected defect. Costs ~1.3 GB.

**Two things the comparison VALIDATES, worth not re-litigating:** their engram rows are byte-identical
in layout to ours (256 B E4M3 + 8 B UE8M0 per-32 = 264 B/row), and their Hyper-Connection tensors are
F32, matching ours. Also note their published accuracy table baselines against **MXFP4, not BF16**,
and carries no KL or top-1-agreement figure -- so it is not a quality receipt we can borrow.

### 2026-09-17: the prefill reserve is NOT reclaimable, and ARENA_GB=86 is decode-only

Job 485. At `ARENA_GB=86`, `max_seq=32768`, after load: torch allocated 94.3 GiB, MemAvailable
18.6 GiB. One 4096-token prefill chunk then drove MemAvailable to **1.3 GB** and the engine's own
guard fired:

    FATAL: host MemAvailable 1.3 GB stayed below the 2.5 GB floor for 3.0 s

So item 5 of the review plan is a **NO**: `MAX_CHUNK * 5e6 = 20.5 GB` is calibrated about right,
not slack, and chunk 8192 (41 GB) stays out of reach -- which also removes the one lever that
coupled the decode memory work to prefill (see `dsv41-enginev2/notes/prefill-in-v2-plan.md`).

**AND IT PUTS A CAVEAT ON THE +10.1 % ARENA RESULT.** That was measured with `bench_tokens`, whose
prompt is three sentences. `ARENA_GB=86` is safe for DECODE-ONLY benchmarking and is NOT safe for
serving long prompts. The auto-sizer's ~84.5 GB exists because it subtracts this reserve. Do not
pin ARENA_GB=86 in production on the strength of the decode number.

Method note: the first two runs of 485 were wrong in ways the job hid -- `forward()` asserts
`T <= MAX_CHUNK` and chunking is the caller's job, and `sys.path` put the v2 worktree (same repo,
different branch, also has `engine/`) ahead of the spark checkout. Both surfaced only after the
job stopped doing `python ... | grep`, which makes the exit status GREP's. That pattern has now
hidden a failure in three separate job scripts (430, 470, 485).

## NULL — measured, no effect, under stated conditions

| idea | result | condition that could change it |
|---|---|---|
| Lock removal (~1080/step) | +0.08 % (job 370) | Measured behind a dominating physical barrier — see RE-OPENED |
| Out-of-order completer | +0.4 % (job 375) | `staging peak 8` vs `h2d_inflight 2`; would matter at higher H2D depth |
| `global_barrier=False` | +0.97 % (job 420) | marginal; the 0–2 % branch |
| Graph segments at engram layers | "no measurable gain either way" (upstream) | — |
| `--prune-select global` | measured worse than uniform (upstream) | — |
| Per-layer arena allocation | ~1 point (ds-05) | — |
| CPU core pinning | +0.029 % against 0.751 % drift, sign flipped | — |

## NEEDS REPEATING — measured under conditions now known to be wrong

Job 420 showed every measurement before it ran with `global_barrier=True`, where `_wait()` blocks
until `_demand == 0` and `_demand` falls only after `handle.synchronize()`. That does not merely add
overhead: it **amplifies the cost of a miss**, because the driver waited for every outstanding copy
rather than this layer's. Anything whose effect depends on the driver NOT already being blocked was
measured in the one regime that would hide it.

| result | why it is suspect | action |
|---|---|---|
| **Job 375**, out-of-order completer, +0.4 % "null" | its mechanism is a finished copy held behind a running one, which costs nothing while the driver waits for all copies anyway | **job 455** re-runs it at gb=0 AND gb=1 in one job |
| **The oracle margin**, +37.8 % cold / +20.4 % warm (sections 3/11), and the "+81-84 % perfect oracle" built on it | both arms paid the barrier, but the oracle has fewer demand misses and the barrier multiplies what each miss costs, so its advantage is **likely overstated**. This is the load-bearing number behind "prediction is worth 1.2x-50x an async loader" in the standing argument | re-measure at gb=0 before that framing is quoted again |
| Job 370, lock removal, 0.08 % | same masking | **not worth the box time** — 420 already bounds the whole family at ~1 % |

**Re-baseline, not re-measure:** `DSV41_SHARED_FIRST=1` is now in production `.env`, so any
SERVER-side number from before 2026-09-17 is against a different configuration. Engine-side jobs set
their env explicitly and are unaffected.

**Does NOT need repeating** — both arms shared the condition, so the comparison holds: the arena
result (+10.1 %), `age_over_freq` (+8 %, measured on real tokens rather than trace replay; the
replay-based eviction work was already withdrawn), and the per-graph budget (A 46.1 / B 69.4 /
S 10.4, which replicated across jobs 400 and 405 and never depended on the seam placement).

**Void, already replaced:** job 430's throughput (token inequality, 1734 vs 1768, and blocked on the
missing checkpoint regardless) and job 445 (unpropagated `h`; replaced by job 450).

## BLOCKED — cannot be measured on this box as it stands

**Paired per-token NLL is not runnable here.** `tools/paired_nll.py` consumes
`state/after_layer39.pt` from `tools/expert_trace.py`, and expert_trace needs the FULL 48-shard
checkpoint (`model-000NN-of-00048.safetensors`). This box has only `~/dsv41-lean` (6.5 GB: dense
shards; the routed experts live in the CB3 cache file), and 198 GB free against a ~476 GB
checkpoint. Job 430 died on exactly this — `FileNotFoundError: .../model-00003-of-00048.safetensors`
— and job 435 was withdrawn before it could fail the same way.

This is why "fused attention + gather fusion still need a paired-NLL verdict" has stayed open. It
is **blocked on a download, not on neglect**, and every quality question below inherits the block:

- **`DSV41_FUSED_ATTN=1`** — not bitwise, so it needs a quality verdict. Job 430's throughput arms
  ran anyway and are NOT a valid comparison: the fused arm emitted **1734 tokens against 1768**, a
  different token stream, so 6.78 vs 6.56 compares two different computations. The job header said
  token equality gates it and the script did not enforce it. **Void; do not quote those numbers.**
- **FP4 on the attention path** (item 13) — same gate, same block.
- Any future non-bitwise kernel or precision change.

**The way out, in preference order:**
1. Build a paired-NLL harness that runs on the LEAN dir through the decode path. The pieces exist:
   `falsify_shared_first.py` already drives the real fastdecode graphs from the lean dir, and
   `RealLeaves` already has a teacher-forced `next_block` mode. What is missing is scoring
   teacher-forced next-token NLL through fastdecode rather than through `model.py`'s eager path —
   and it MUST be the fastdecode path, because `FUSED_ATTN` lives in `fastdecode._attention` and an
   eager-path measurement would not exercise it at all.
2. Fetch the full checkpoint (~476 GB; does not fit in 198 GB free without removing something).

Until one of those lands, **no non-bitwise change can be shipped**, and the ship rule stands:
bitwise-identical switches only.

## CLOSED — measured and refuted, mechanism understood

- **Expert IDENTITY prediction** by co-occurrence, recurrence, or the DSpark drafter. Our entropy
  6.89–7.12 bits over accesses; the predictable mass is already resident (LRU captures the skew, so
  what is left to predict is the flat tail). Now independently confirmed three ways: expert-sniper
  measured **97 % recall@16 with zero throughput gain**; the shi3z argument that any accurate
  predictor names experts that are *already cached*, so prediction is anti-correlated with what
  needs predicting; and arXiv 2505.16056, which ranks **DeepSeekMoE-with-shared-experts last
  (36.94)** on routing consistency and gives the architectural reason — shared experts shrink the
  usable combination space and rigid load-balancing suppresses local patterns. **We have both.**
- **Expert substitution / "buddy" experts.** SpecMD: *"Substitution Score fails across all
  configurations."* BuddyMoE's own best case is +10 % for 2.1–5.4 % accuracy loss.
- **Chasing the Belady gap with a better heuristic.** The gap is real (39–45 % of misses) and 84 %
  of it is future-victim ranking, but the obvious predicted-next-use policy measured **worse than
  LFRU** (Spearman −0.207, anticorrelated).
- **Training the router for cacheability.** Pre-registered negative result (arXiv 2608.18261):
  59–60 % miss reduction at +2.1–3.1 % perplexity against a ≤1 % bar.
- **Shadow/draft-model lookahead** (OD-MoE, MoE-SpeQ, SP-MoE, SPICE). Ruled out on **capacity**,
  not principle: needs a second model with resident experts; at 3 bits our pool is ~212 GB.
  Exception worth a note: DraftExpert replaces the draft's routed experts with one trained draft
  expert per layer and its gains were *larger* on flash-resident configs (1.52–1.61×).
- **Multi-batch pipelining** (Klotski, PreScope). Klotski states single-batch decode gets minimal
  benefit; PreScope's headline is batch-64. We are single-stream.
- **GPUDirect Storage / cuFile.** Vendor-disabled on DGX Spark: `nvidia-spark-default-remove-nvidia-fs-pkg`
  removes the loader at first boot, `gdscheck -p` reports every backend Unsupported, and our
  `cufile.log` says `running in compatible mode` — a host bounce buffer plus a copy, i.e. what we
  already do through a heavier API. Architecturally moot anyway on unified memory. BaM/GIDS need
  PCIe P2P BAR mapping and have no aarch64 port.
- **Read coalescing** (within one expert): already 6→2 runs, overread 0.06 %, alignment right.
- Segmented caches; row-block CB2/CB3 tiering; CB2 whole-expert; expert prediction heads in the
  PROTECTION role (perfect oracle 0.4 pp).

## RE-OPENED — closures that did not hold

1. **"Python synchronisation is not material" (job 370).** Unsupported. `bench_tokens` runs
   `Policy()` = V1, where `global_barrier` makes `_wait()` block on `_demand == 0`, decremented only
   after `handle.synchronize()`. What 370 removed sat behind a barrier that dominated it; the
   experiment could not have returned anything else. Correct statement: 370 (locks, 0.08 %) plus
   420 (the barrier, 0.97 %) says host-side scheduling is worth ~1 %, because the step is 269 ms of
   read wait out of 443.
2. **The shared-expert overlap (jobs 395/400/405, three nulls).** All three measured a reordering
   INSIDE the post-wait region: the driver's `shared()` call sat between the two wait branches, and
   under V1 the first branch fired, so the shared expert was enqueued *after* the reads had landed.
   With the seam moved ahead of both, `layer_a` falls 5.86 ms and the wall follows.
   **Lesson: a gate proving the output is bitwise identical says nothing about whether the code ran
   where you think it did.**
3. **Pruning at keep 0.44.** Closed on measured quality — at K154, 9.0 % of coding picks masked but
   **60.6 % of general picks masked**. But our own analysis is frequency-ranked throughout, and the
   third-party keep-set we tested was of unknown ranking. Upstream's saliency-vs-frequency table
   says frequency collapses to 0.000 coverage at 50 % keep while saliency holds 0.429. **The input
   that would change the verdict has changed.** Re-test with saliency ranking before treating this
   as settled either way.

## Measurement traps to re-read before designing any job here

- **Count tokens via `usage.completion_tokens`, never SSE chunks.** With spec decode one chunk is a
  STEP, ~3 tokens. Upstream's bench README names this explicitly as a 3–5× error.
- **Per-token trace replay inflates LRU by +27.4 % and LFRU by +28.5 % but LFU by only 3.4 %** — it
  inverts the policy ranking. Replay event-atomically (all top-k of a step as one event).
- **A lower NLL can be softening**, not information. Sweep the baseline's temperature;
  `paired_nll.py` does this.
- **Check the baseline arm is physically realizable.** The biggest error of this project was an
  oracle whose no-prediction arm got compute that ran before the misses were knowable.
- **A single A/B pair is not a result on this box.** Same-config arms span 1.4–1.8 %.
- **The device is bimodal**: 4.8 or 6.8 GB/s for identical work, one run in four or five. Exclude
  or replicate; never compare across jobs.
