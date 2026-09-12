#!/usr/bin/env python3
"""Is the shared-base + low-rank delta idea alive on these models? One offline SVD, both checkpoints.

D2-MoE stores a shared base plus a per-expert low-rank delta; MoE-SVD shares factors across experts.
Both report 40-60% against an FP16 parameter count. Our models are already 4-bit, so the only
question that matters is: at what rank does the DELTA retain its energy, and is that rank below the
break-even against the 4-bit bytes we already pay?

Break-even, shared-V, per-expert U held at bf16 (U is inter x r):
  Qwen 640x2560   fp4 0.82 MB  -> r < 640  of max 640   (every rank below full saves)
  DS   2304x5120  fp4 5.90 MB  -> r < 1280 of max 2304  (must beat 56% of full rank)

Controls that decide whether a low-rank delta is even news:
  * spectrum of W itself -- if the experts are already low-rank, a low-rank delta says nothing new
  * a RANDOM matrix of the same shape -- the floor any "structure" claim must clear

Reads only what is already on disk: the Qwen NVFP4 checkpoint, and DS V4.1 layer-0 experts from the
one verified shard. No downloads, no server, CPU/GPU-light.
"""
import json, sys, torch

DEV = "cuda" if torch.cuda.is_available() else "cpu"
NE = int(sys.argv[1]) if len(sys.argv) > 1 else 16


def ranks(M):
    """rank retaining 95% / 99% of squared singular-value energy, and max rank"""
    s = torch.linalg.svdvals(M.float())
    e = (s * s).cumsum(0) / (s * s).sum()
    return int((e < 0.95).sum()) + 1, int((e < 0.99).sum()) + 1, min(M.shape)


def report(tag, W, break_even):
    base = W.mean(0)
    d95 = d99 = w95 = w99 = 0
    for i in range(W.shape[0]):
        a, b, mx = ranks(W[i] - base); d95 += a; d99 += b
        a, b, _ = ranks(W[i]);          w95 += a; w99 += b
    n = W.shape[0]
    r95, r99, mx = ranks(torch.randn_like(W[0]))
    print(f"\n  {tag}  {tuple(W.shape[1:])}, {n} experts, max rank {mx}")
    print(f"    {'':<22}{'rank@95%':>10}{'rank@99%':>10}")
    print(f"    {'delta (W - base)':<22}{d95//n:>10}{d99//n:>10}")
    print(f"    {'W itself (control)':<22}{w95//n:>10}{w99//n:>10}")
    print(f"    {'random (floor)':<22}{r95:>10}{r99:>10}")
    print(f"    break-even rank vs the 4-bit bytes we already pay: {break_even}")
    # A break-even test alone is not enough: on a very rectangular expert a RANDOM matrix can also
    # come in under break-even, so "passes" would mean nothing. Require the delta to beat the random
    # floor and W itself by a margin, and name the artefact when it does not.
    d, w = d99 // n, w99 // n
    beats_be, beats_noise, beats_W = d < break_even, d < r99 * 0.9, d < w * 0.9
    if beats_be and beats_noise and beats_W:
        verdict = "USABLE"
    elif beats_be:
        verdict = (f"ARTEFACT -- under break-even, but random scores {r99}, which would also pass"
                   if r99 < break_even else
                   f"NOT STRUCTURE -- under break-even but within {abs(d-w)} ranks of W itself")
    else:
        verdict = "NO SAVING"
    print(f"    -> delta@99% {d} vs break-even {break_even}, W {w}, random {r99}: **{verdict}**")
    return beats_be and beats_noise and beats_W


alive = []
# ---- DeepSeek V4.1, real layer-0 experts from the verified shard
try:
    sys.path.insert(0, "/tmp/claude-1000/-home-jschmied-git-dgx-spark-setup-guide/"
                       "24a6e32a-d571-4261-96e4-1463dea3f47a/scratchpad/dsv41/tools")
    import fp4_moe as F4
    from safetensors import safe_open
    SH = "/opt/llm/models/dsv41-shards/model-00003-of-00048.safetensors"
    ar = F4.ExpertArena(NE, DEV)
    with safe_open(SH, "pt", device="cpu") as f:
        for e in range(NE):
            p = f"layers.0.ffn.experts.{e}."
            ar.load_slot(e, *[f.get_tensor(p + n) for n in
                              ("w1.weight", "w1.scale", "w2.weight", "w2.scale", "w3.weight", "w3.scale")])
    W = torch.stack([ar.dequant_slot(e)[0] for e in range(NE)])   # w1 (gate)
    alive.append(report("DeepSeek-V4.1 layer-0 w1", W, 1280))
    del ar, W; torch.cuda.empty_cache()
except Exception as ex:
    print(f"  DS arm skipped: {type(ex).__name__}: {str(ex)[:140]}")

# ---- Qwen3.8-Flash-Next, NVFP4 experts
try:
    sys.path.insert(0, "/opt/llm/runners/lh")
    import nvfp4pack as NP
    from safetensors import safe_open
    MD = "/opt/llm/models/qwen38-flash-next-nvfp4"
    wm = json.load(open(f"{MD}/model.safetensors.index.json"))["weight_map"]
    key = f"model.language_model.layers.24.mlp.experts.0.down_proj.weight"
    shard = wm[key]
    rows = []
    with safe_open(f"{MD}/{shard}", "pt", device="cpu") as f:
        for e in range(NE):
            p = f"model.language_model.layers.24.mlp.experts.{e}.down_proj."
            rows.append(NP.decode(f.get_tensor(p + "weight"),
                                  f.get_tensor(p + "weight_scale"),
                                  f.get_tensor(p + "weight_scale_2")).to(DEV))
    W = torch.stack(rows)
    alive.append(report("Qwen Flash-Next L24 down_proj", W, 640))
except Exception as ex:
    print(f"  Qwen arm skipped: {type(ex).__name__}: {str(ex)[:140]}")

print(f"\n  Any arm where the delta beats break-even? {'YES' if any(alive) else 'NO'}")
print("  NO on both closes the shared-base/delta direction offline, before any kernel work.")
print("== ALL DONE ==")
