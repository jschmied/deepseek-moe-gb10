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
