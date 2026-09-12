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
