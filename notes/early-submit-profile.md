# The early-submit profile: the loader-overlap family is closed, and not by a bandwidth wall

*2026-09-14, after enabling GPU performance counters and rebooting. Three arms, one start each,
`nsys` with `--gpu-metrics-set=gb20b`, prompt cache off, 12,624-token prompt, 16 output tokens.
Reports: `~/ds41-queue/logs/nsys-early-{off,chunk0,all}.nsys-rep`. Tool: `tools/nsys_early_submit.py`.*

## What the question was, and why it could not be asked

The open question from the 09-14 handover was: *"overlapping expert copies with MoE kernels slows
the FFN by 32 %, at ~6 % of this box's 240 GB/s stream rate -- is that bandwidth, L2 pollution, or
launch contention?"*, with "does the GPU memory-bandwidth row rise across the arms" as the test.

**That row does not exist on this box, by any route.** The `gb20b` metric set has no DRAM counter
(clocks, copy engines, GR/SM/Tensor active, warps in flight -- that is all of it), `--soc-metrics`
answers *"The feature is not supported on this system"*, `nvidia-smi dmon`'s `mem %` is stubbed on
this iGPU, and `ncu` is not installed. So the question was re-posed on counters that do exist:
stalled-but-resident warps (memory system) vs gaps between kernels (launch contention) vs copy
engines at their ceiling, with the GPC clock as a throttle check that would invalidate all three.

## 1. There is no FFN slowdown to explain

Inside the MoE kernels, every discriminator is flat:

| arm | kernels | mean us | busy ms | clk MHz | sm_act | sm_issue | tensor | warps |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| off | 3042 | 2956.3 | 8993.0 | 2512.0 | 97.9 | 21.7 | 23.3 | 16.9 |
| chunk0 | 3042 | 2844.3 | 8652.3 | 2505.5 | 97.9 | 22.6 | 24.3 | 16.9 |
| all | 3042 | 2943.7 | 8954.8 | 2503.9 | 97.9 | 21.9 | 23.5 | 16.9 |

Identical launch counts. `_moe_up_kernel` 3864.5 / 3684.6 / 3824.1 us and `_moe_down_kernel`
2048.0 / 2004.0 / 2063.4 us -- `all` is within +-1 % of `off` on both. No throttle, no gaps
(SMs Active 97.9 % in all three), no change in occupancy or issue rate.

**The 32 % was measured on `moe_s`, a host-side wall clock around the call, not on the kernels.**
It is not a GPU phenomenon: at kernel level the effect is absent. Whatever `moe_s` was charging,
it was host wait, and the three mechanisms proposed for it were all explanations of something that
does not happen on the device.

## 2. Early submission does overlap -- with attention, never with MoE, and that is correct

Copy engines read 0.0 % inside MoE windows in every arm, `all` included. That is not a failure:
`join_pending()` precedes `moe_apply` by construction, so a copy can never be in flight there. The
overlap it is designed for is with attention, and against *all* kernels it is real and large:

| arm | memcpy total | overlapped with GPU work | share of copy time |
| --- | ---: | ---: | ---: |
| off | 2960.6 ms | 100.3 ms | 3.4 % |
| all | 3645.8 ms | 1025.2 ms | 28.1 % |

A 10x increase in realised overlap. The mechanism works.

## 3. And it cannot matter, because the thing it hides is 3 % of the time

| arm | span s | GPU busy s | busy % | copy s | copy % | hidden s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| off | 93.2 | 40.3 | 43.3 | 2.95 | 3.2 | 0.10 |
| chunk0 | 92.6 | 41.8 | 45.1 | 3.29 | 3.5 | 0.68 |
| all | 94.2 | 41.8 | 44.3 | 3.61 | 3.8 | 1.03 |

The entire H2D copy time in a 93-second prefill is **2.95 s**. That is the whole prize for any
loader-scheduling change, ever. `all` already hides 1.03 s of it, leaving ~2.6 s (2.8 % of TTFT)
for a perfect scheduler -- and overlapping *inflates the copies themselves by 22 %* (2.95 -> 3.61 s),
which eats a third of what it wins. End to end the arms are indistinguishable:

    off 83.0 s | chunk0 82.1 s | all 84.7 s     (12,624 tok, 152.1 / 153.7 / 149.1 tok/s prefill)

3.2 % spread across three single starts, with `all` slowest. Within restart noise; they cannot be
ranked. **The loader-pipeline family is closed** -- not because hardware caps it, but because it
optimises 3 % of the wall clock. `DSV41_STAGE_BUFS` and the read-early/copy-late A/B are both
answers to a question worth 2.8 % at its theoretical maximum, and should not be built.

## 4. Where prefill time actually goes

**The GPU is idle 55 % of the prefill** (busy 40.3 s of 93.2 s), and H2D explains 3 % of it. The
remaining ~50 s is host-side: NVMe reads and the work around them. That is the only object left
worth attacking, and it is 17x larger than everything the early-submit family was arguing about.

Also visible, and not previously on any list: `_cb3_unpack_kernel` is the **single largest GPU
consumer** at 7.44 s (off), ahead of `_moe_up_kernel` at 5.88 s. It is pure load-path overhead --
unpacking 3-bit rows into the arena layout. 7.4 s of GPU time, 8 % of the prefill, spent on the
format rather than on the model. That is a bigger prize than the entire copy-scheduling question.

## What this supersedes

* The handover's open question ("is it bandwidth, L2, or launch contention?") is void: the premise
  (a 32 % FFN slowdown) does not exist at kernel level.
* `notes/layer-scheduler.md`'s loader-pipeline items and the queued `DSV41_STAGE_BUFS` work: closed,
  ceiling 2.8 %.
* The reviewer's read-early/copy-late three-arm experiment: no longer worth running for its own
  sake; it would resolve read-side vs copy-side within a 3 % budget.
