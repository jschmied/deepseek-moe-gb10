# Shared base + low-rank deltas on an already-quantized MoE

Opened 2026-09-12. Sources read directly (PDFs, not abstracts) during the Qwen work on the same day;
carried here because the question is really about DeepSeek.

## The two papers, with their baselines stated

| | what it does | headline | baseline |
|---|---|---|---|
| **D²-MoE** ([arXiv 2502.17298](https://arxiv.org/abs/2502.17298), ICML'25) | shared base weight + per-expert low-rank delta (SVD), plus semi-dynamic structured pruning of the base | "40–60 % compression", >13 % over other compressors | **FP16 parameter count** |
| **MoE-SVD** ([PMLR v267](https://proceedings.mlr.press/v267/li25az.html), ICML'25) | selective SVD; one shared V across experts, top-k U matrices | "60 % compression, 1.5× inference" | **FP16 parameter count** |

Both are training-free. Both were tested on **8, 16 and 64 experts** (Mixtral-8×7B, Phi-3.5-MoE,
DeepSeekMoE-16B, Qwen2-57B-A14B). Neither was tested at the expert counts modern DeepSeek MoEs use.

## The three things that decide whether this helps us

**1. The ratios are against FP16, and we are not at FP16.** "60 % compression" means 8-bit-equivalent.
Against a 3-bit CB3 representation that is *larger*, not smaller. The published number cannot be
read as a further 60 % off what we already have.

**2. MoE-SVD is the only one that actually tested the stack, and the result is sobering.** Its
Table 8 combines MoE-SVD with GPTQ and reports that **MoE-SVD (4-bit) is on par with plain GPTQ
(3-bit)** on both memory and accuracy. So the SVD buys roughly one bit over a 4-bit baseline — and
if the starting point is *already* 3-bit, that is precisely the regime where the gain has been spent.
D²-MoE does not test it at all: its conclusion lists "parameter quantization" as **future work**.

**3. The one paper validated at our expert count reports far less.** [MoBE (arXiv 2508.05257)](https://arxiv.org/pdf/2508.05257),
Aug 2026, tests Qwen3-235B-A22B, **DeepSeek-V3-0324** and Kimi-K2 — and gets **24–30 % at 1–2 %
accuracy loss**, while noting that prior methods lose **7–14 % relatively even at modest rates** on
these models. It also states that applying **D²-MoE to models of that scale is "computationally
prohibitive on an 8×H100 machine"**, which is a practical bound on ever building one here.

## Where the idea is genuinely attractive anyway

The structural argument is better than the arithmetic. If the router normalises to a fixed route
scale, a shared base is read **once per token per layer** regardless of how many experts fire, so
the saving scales with *distinct experts touched*, not with expert size. On a measured pattern of
~21 distinct routed experts per layer, that is a much stronger lever than any scalar quantizer,
because scalar quantization pays per expert and this does not.

That is the part worth testing. The 55 %-bandwidth figure that motivated this is illustrative and
assumes the compression composes with an already-3-bit representation — which is exactly the
assumption point 2 contradicts.

## Cheapest test that could refute it

Before any implementation: take the target checkpoint's routed experts, compute the SVD spectrum of
`W_expert − W_shared_base` for a handful of layers, and ask what rank retains (say) 99 % of the
delta's energy. If the deltas are not strongly low-rank at these shapes, nothing downstream matters
and the question closes for the cost of one offline script. Only if they are does the kernel question
(two small GEMMs versus one, and whether vLLM's fused-MoE path can host a factorised expert at all)
become worth asking.

## Not yet established here

- The target model's actual expert geometry (see `notes/TODO.md` — survey in progress).
- Whether the routed-expert traffic is the binding constraint on GB10 for this model, or whether it
  is dense/attention-bound the way Flash-Next turned out to be (69 % of single-stream there was
  BF16 GEMV on unquantized dense weights, not experts).

---

# RESULT 2026-09-12 — the deltas are not low-rank. The direction closes.

`tools/delta_spectrum.py`, one GB10, offline, no server. 16 experts per arm, SVD spectrum of
`W_expert − mean(W)` against two controls: the spectrum of **W itself** (if experts are already
full-rank, a full-rank delta is not news) and a **random matrix** of the same shape (the floor any
structure claim must clear). Real DS V4.1 layer-0 experts from the checksum-verified shard, and real
Qwen L24 `down_proj` from the NVFP4 checkpoint.

| | max rank | **delta @99 %** | W itself | random floor | break-even |
|---|---|---|---|---|---|
| DeepSeek-V4.1 L0 `w1` (2304×5120) | 2304 | **2121** (92 %) | 2119 | 2158 | 1280 |
| Qwen Flash-Next L24 `down_proj` (2560×640) | 640 | **616** (96 %) | 615 | 618 | 640 |

**There is no exploitable low-rank structure in the deltas.** They sit within **1–2 ranks of the
expert matrices themselves** and within **2–37 ranks of pure noise**. At 99 % energy they need
92–96 % of full rank — that is not a compression scheme, it is a rounding error.

**The Qwen arm's apparent pass is an artefact, and the control caught it.** 616 < 640 looks like a
saving until you notice the random matrix scores **618** — also under break-even. On experts that
rectangular (4:1), shared-V factorisation "saves bytes" for *random data*, so passing break-even
measures the shape, not the weights. The script's verdict logic has been fixed to require beating the
random floor and W itself by a margin; it previously printed `USABLE` for this row.

## What this settles

The D²-MoE / MoE-SVD premise — that expert weights decompose into a shared base plus a *low-rank*
per-expert delta — **does not hold on either of these checkpoints**. That closes the direction before
any kernel work, any custom vLLM layer, and any of the 40–60 % headline numbers, for the cost of one
offline script. It is consistent with the three earlier objections and now supersedes them: the
ratios were against FP16, MoE-SVD's own Table 8 put 4-bit + SVD on par with plain 3-bit, and MoBE
reported only 24–30 % on DeepSeek-V3-class models.

It also retires the original bandwidth argument. A shared base read once per token per layer only
helps if the per-expert remainder is cheap, and the remainder here is ~95 % of a full-rank matrix.

## What it does not settle

- **The base is a plain mean**, not D²-MoE's Fisher-weighted merge. A better base would shift the
  numbers — but `W` itself is already ~full-rank and the delta tracks it within 2 ranks, so no choice
  of base makes differences between near-full-rank, mutually dissimilar matrices low-rank.
- One layer per model, 16 experts, one projection each (`w1` / `down_proj`). Layer 0 and layer 24 may
  not be typical, though 48 matrices agreeing this tightly is not a marginal signal.
- Says nothing about **pruning or merging** (REAP, REAM), which are what every sub-128 GB DeepSeek
  build actually uses and which do not assume low rank.
