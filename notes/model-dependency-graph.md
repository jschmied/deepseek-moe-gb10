# DeepSeek-V4.1-Flash: what depends on what

The **model's** dependency structure, read out of `inference/config.json` and `tools/v41_ref.py` —
not our engine's schedule. Every number below is from the checkpoint.

This exists because almost every scheduling argument we have had turned on a dependency question:
*can this start before that finishes?* The answer is a property of the model, and it is not obvious
from the layer count.

## The shape in one picture

```mermaid
flowchart TB
  tok["token ids"] --> emb["embed → h [T, 4, 5120]<br/>hc_mult = 4 parallel residual copies"]
  emb --> L0

  subgraph ENC["ENCODER — layers 0..20 (writes all global KV)"]
    direction TB
    L0["L0<br/>dense"] --> L1["L1 · ENGRAM"] --> L2["L2 ★ KV+IDX source<br/>ratio 2"]
    L2 --> L3_7["L3..L7<br/>read L2's ckv/ik"]
    L3_7 --> L8["L8 ★ KV+IDX source"] --> L9_13["L9..L13"]
    L9_13 --> L14["L14 ★ KV+IDX source · ENGRAM"] --> L15_19["L15..L19"]
    L15_19 --> L20["L20 ★ KV+IDX+CANDIDATE source<br/>ratio 1 — the CED boundary"]
  end

  subgraph DEC["DECODER — layers 21..39 (no new global KV)"]
    direction TB
    L21_23["L21..L23<br/>reuse L20's top-k verbatim"] --> L24_39["L24..L39<br/>search inside L20's candidate pool<br/>own indexer at 24/28/32/36"]
  end

  L20 --> L21_23
  L24_39 --> D37["L37,L38,L39<br/>DSpark snapshots"]
  D37 --> norm["final norm → head → logits"]
  D37 -."mean over hc, concat 3×5120"..-> mainh["main_hidden [T, 15360]"]
  mainh --> mtp["main_proj → 3 MTP layers<br/>128 draft experts, top-3, block 5"]

  eg[("Engram n-gram tables<br/>hash = f(token ids only)")] -.-> L1
  eg -.-> L14
```

## Inside one layer — the only true serial chain

```mermaid
flowchart LR
  h["h [T,4,5120]"] --> hcA["hc_mixes<br/>(attn)"]
  h --> preA["hc_pre + rmsnorm"]
  hcA --> preA
  preA --> attn["ATTENTION"]
  attn --> postA["hc_post<br/>+ residual"]
  postA --> hcF["hc_mixes<br/>(ffn)"]
  postA --> preF["hc_pre + rmsnorm → y"]
  hcF --> preF
  preF --> router["router<br/>top-6 of 384"]
  preF --> shared["SHARED expert<br/>no routing dependency"]
  router --> experts["6 routed experts"]
  experts --> sum["Σ in fp32"]
  shared --> sum
  sum --> postF["hc_post → h'"]
```

**The shared expert needs only `y`.** It does not depend on the router, on any slot, or on any
routed result — yet it is computed after them. That is an implementation choice, not a model
constraint.

**The six routed experts are independent of each other.** Their contributions are summed, so there
is no requirement that they execute together — which is what makes partial/streamed MoE legal.

## Attention's two independent branches

```mermaid
flowchart LR
  x["normed x"] --> q["Q: wq_a → q_norm → wq_b → RoPE<br/>64 heads × 512"]
  x --> kv["KV: compress → ckv (latent 512)<br/>+ ik (index 128)"]
  q --> idx["indexer: score ckv rows,<br/>pick candidates"]
  kv --> idx
  idx --> gather["gather k rows + SWA window (128)"]
  gather --> out["attention out"]
```

Q and KV share only `x`. Nothing in the Q path needs the KV path until the indexer.

## The cross-layer reuse that makes CED possible

| what | written at | read by |
|---|---|---|
| compressed KV `ckv`, index `ik` | **only layers 2, 8, 14, 20** | every layer up to the next source |
| indexer parameters | 2, 8, 14, 20, **24, 28, 32, 36** | the three layers after each |
| **candidate pool + top-k** | **layer 20** | 21–23 reuse its top-k verbatim; 24–39 search inside its pool |
| Engram rows | tables, keyed by **token ids alone** | layers **1 and 14** |
| DSpark `main_hidden` | layers **37, 38, 39** (snapshot taken *before* the block) | `main_proj` → the 3 MTP layers |

**Four layers of forty write global KV.** That is why the decoder half can be replayed over just the
last 128 positions, and why the whole prompt's KV state is 105 MB rather than the ~1 GB a
40-layer cache would need.

**Engram hashes depend on token ids only** — not on any hidden state. So they are computable for the
entire prompt before layer 0 runs, and they replay deterministically for a cached prefix.

## What this licenses, and what it forbids

**Legal** (dependency genuinely absent):
- Reordering chunks against layers — layer-major prefill. Each layer's experts are needed only by
  that layer.
- Computing the shared expert while routed experts are still arriving.
- Computing some routed experts before others; the sum is order-independent in fp32 accumulation.
- Prefetching Engram rows arbitrarily early.
- Issuing a chunk's expert reads the moment *that chunk* has routed.

**Forbidden** (real dependency):
- Chunk *k+1*'s attention before chunk *k*'s attention **at the same layer** — it reads the KV chunk
  *k* wrote.
- Any layer > 20's attention before layer 20 has produced its candidate pool.
- Layer L+1 before layer L's `h` is complete.
- The MTP draft before the accepted token exists — the drafter is seeded from it.

**Conditional:**
- Chunk *k*'s FFN need not precede chunk *k+1*'s attention at the same layer. Its output is needed
  only at layer L+1. This is a genuine pipeline boundary that layer-major creates and we do not use.

## Sizes, for judging what is worth overlapping

| | |
|---|---|
| routed experts | 384 × 40 = **15,360**, 14.45 MB each in CB3 = 222 GB |
| distinct experts one layer touches on a 2k-token chunk | **~300 of 384**; a whole layer's union ≈ **362** |
| a layer's token state | `[T, 4, 5120]` bf16 — 1.34 GB at 32k |
| whole KV + index + window state at 32k | **285 MB** (4 source layers, not 40) |
| Engram tables | 203 GB on disk, read by HTTP range / NVMe per token |
