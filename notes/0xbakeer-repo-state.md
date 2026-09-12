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
