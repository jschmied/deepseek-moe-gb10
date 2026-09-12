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

Posted to 0xBakeer/deepseek-v41-flash-spark issue #3 on 2026-09-12: https://github.com/0xBakeer/deepseek-v41-flash-spark/issues/3#issuecomment-5646328174

---

# ds-01 (2026-09-12) — idea 3 does NOT apply to V4.1: there are no hash-routed layers

The user's idea 3 proposed a non-uniform layer budget (layers 0–2 at 100 %, the rest ~39.6 %) on the
grounds that "the first hash-routed layers map token IDs directly to experts, so calibration
frequency is inherently biased", and asked to verify that this transfers to V4.1 **first**. It does
not. Checked against the real checkpoint, not the V4 tooling's description.

| evidence | result |
|---|---|
| `text_config` routing keys | `topk_method = noaux_tc`, `scoring_func = sqrtsoftplus`, top-6 of 384 + 1 shared |
| hash-ish tensors in the 48-shard index (`hash\|bucket\|token2expert\|expert_map`) | **0** |
| layers with a learned `ffn.gate.weight` `[384, 5120]` BF16 | **40 of 40** |
| dense layers with no router | **none** |

Layers 0, 1 and 2 each carry an ordinary learned router. The hash-routed early layers belong to the
*other* V4 checkpoint (256 experts/layer) that the third-party REAP adapter targets — the same
checkpoint the user already warned not to transfer absolute results from. **So the non-uniform budget
has no justification here and idea 3 should be dropped for V4.1**, before any implementation.

## But the underlying worry has a real V4.1 analogue — and it is small

Every router carries **two** biases: `ffn.gate.bias` and `ffn.gate.bias_vl` (both `[384]` f32, all 40
layers). A second, vision-language routing correction. A text-only calibration trace never exercises
it, which is exactly the user's concern — a token population absent from the corpus getting no
evidence — arriving through a different mechanism.

Ranking experts by the bias **alone** looks alarming: `corr(bias, bias_vl) = −0.30`, top-6 overlap
0/6. **That reading is wrong and I nearly shipped it.** A constant offset cancels in top-k, and these
biases are mostly constant offset (mean +9.83 vs +21.21) with tiny spread. Against the actual router
logits:

| | spread | vs logit std 2.97 |
|---|---|---|
| `bias` | 0.0359 | 1.2 % |
| `bias_vl` | 0.0994 | 3.3 % |

Selecting top-6 with one bias versus the other, over 512 random unit-RMS directions: **mean overlap
5.75/6, identical for 75.6 % of tokens, never worse than 4/6.** So the VL path moves ~4 % of expert
slots and touches one token in four. Real, measurable, and **not** a basis for a 100 %-resident early
layer budget.

**Assumption stated:** random directions, not real hidden states, so this bounds the scale rather
than simulating routing. Feeding real token embeddings through the layer-0 router would sharpen it.

## Architecture facts recovered while doing this (all from the checkpoint)

40 layers, all MoE. `n_routed_experts 384`, `num_experts_per_tok 6`, `n_shared_experts 1`,
`norm_topk_prob true`, `routed_scaling_factor 1.5`. DSpark: `dspark_target_layer_ids [37,38,39]`,
`n_routed_experts 128`, `num_experts_per_tok 3`, `markov_rank 256`, `block_size 5`,
`num_nextn_predict_layers 3`. CSA2: `index_source_layer_ids [2,8,14,20,24,28,32,36]`,
`kv_source_layer_ids [2,8,14,20]`, `index_topk 512`, `candidate_topk_blocks 2048`,
`sliding_window 128`; `compress_ratios` starts `[0,0,2,2,…]`, so layers 0–1 are uncompressed and
**layer 2 is the first indexed layer** — which is why shard 5 carries the indexer weights and
layers 0/1/3 do not. `num_experts_per_tok 6` is the "expert #6" of idea 7.
