# TODO: DeepSeek-V4.1-Flash on one Spark in TensorFold - existing EXL3 ~2.9 bpw + saliency pruning

Status: PARKED as a TODO (2026-10-12). Nothing downloaded, nothing built.

## Idea

0xBakeer's recipe (PRUNE_KEEP=0.44, CB3 3-bit codebooks, all resident, no NVMe at decode: 36.6 tok/s HTML,
17.1 prose) with a better quant: an existing EXL3 build instead of CB3, pruned to what fits one GB10.
EXL3 at ~3 bits has about half CB3's error (our measurements: K3 rel. error 0.022-0.027 vs CB3 0.046).
No own quantization needed.

## Facts gathered

- No DS4.1 weights locally or on PBS (ds41-archive holds metadata/logs only; downloads deleted).
- HF EXL3 builds: Mia-AiLab 2.9 bpw 210.7 GB (what bertholomus serves), jayleaton 2.8 bpw 203.8 GB,
  Mia-AiLab 3.0 bpw 219.3 GB; TensorFold/DeepSeek-V4.1-Flash-MLX-2bit-MTP 238.8 GB.
- One Spark leaves ~88 GB for experts -> keep ~0.40-0.45 of each layer's experts at ~2.9 bpw.
- TensorFold: bertholomus's `dsv41` family (#559; TP2/TP4 only, EXL3 experts, DSpark, Engram) and
  jayleaton's upstream rewrite (#671-#727, CPU-checked only). No expert-streaming tier, no router mask.
- Quality history: frequency-ranked keep 0.44 masked 60.6 % of general picks (closed); 0xBakeer's
  saliency ranking (gate weight x output norm) holds 0.43 coverage at 50 % keep where frequency gives 0.00.
  Their first keep-set came from a corpus without markup -> `<!DOCTYPE>` loops; corpus coverage is the risk.

## Steps

1. [ ] Read bertholomus `deepseek-v41-zig`: can `dsv41` run on one GPU (TP1)? Where would a router mask go
   (pruned experts out of top-k, renormalize)? Code only, no download.
2. [ ] Read 0xBakeer's keep-set format and how its saliency keep-sets were traced (corpus incl. markup,
   code, multilingual); pick keep levels that fit (~0.40-0.45 at 2.9 bpw).
3. [ ] Go/no-go, then ask before downloading Mia-AiLab 2.9 bpw (210.7 GB; disk had 253 GB free).
4. [ ] TF: single-GPU dsv41 + router mask + load only kept experts.
5. [ ] Quality: real-text NLL / top-1 and 0xBakeer's free-generation gate (900-2,000 tokens, distinct-token
   ratio > 0.25, line repeat < 30 %, structural checks) on code, prose, markup, multilingual. No unpruned
   reference fits one box; compare with published numbers, or borrow a 2-node setup.
6. [ ] Speed: decode with DSpark drafting vs 0xBakeer's 17-37 tok/s and our enginev2's 6.03 tok/s.
