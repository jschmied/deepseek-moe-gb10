#!/usr/bin/env python3
"""Which layers tolerate fewer bits? Activation-weighted requantization error, per layer.

MiaAI-Lab's EXL3 build of this checkpoint spends bits per LAYER as well as per tensor role --
routed experts K=3 except layers 18-22 at K=2. Ours is uniform 3-bit, and uniform 2-bit costs
2.16 pp of coding top-1 (ds41-measured §20), which closed the all-resident direction. The question
that measurement did not answer is whether the bits can be MOVED rather than uniformly removed.

Answering it with traces would be 40 layers x 150 min per assignment. This is the cheap first half:
no forward pass, no logits. For each layer it requantizes every routed expert through the same
`CodebookSim` the reference path uses and reports

    err(L, b) = sum_e  usage_e * ||Q_b(W_e) - W_e||_F^2  /  sum_e usage_e * ||W_e||_F^2

where usage_e is the expert's summed GATE WEIGHT over a real trace, so a layer whose error sits on
experts nobody routes to is not punished for it. The ratio err(L,2)/err(L,3) is the quantity that
decides where a bit is cheap.

This is a PROXY. Weight-space error is not output error, and the only way to price an assignment is
the paired teacher-forced trace. Its job is to pick which assignment is worth that trace.

  python tools/perlayer_quant_error.py --shard-dir ... --trace ... [--layers 0-39] [--experts 24]
"""
import argparse, glob, json, os, re, sys

import numpy as np
import torch


_FP4 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
                     -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0])


def _deq(w: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    """packed FP4 codes + ue8m0 scales -> the real weight values.

    `requant_packed` returns packed CODES, not values. Comparing those byte patterns is meaningless
    -- a code moving 7 -> 8 flips the sign bit and reads as a huge delta while 0 -> 6 reads as
    small. Scored that way, 2 bits came out with LESS error than 3, which is impossible and is what
    caught it on the first row.
    """
    N, K2 = w.shape
    lo = (w & 0x0F).long()
    hi = ((w >> 4) & 0x0F).long()
    codes = torch.stack([lo, hi], dim=-1).reshape(N, K2 * 2)
    vals = _FP4.to(w.device)[codes]
    scale = torch.exp2(s.float() - 127.0).repeat_interleave(32, dim=1)
    return vals * scale


def usage_by_expert(trace_dir, L, n_experts=384):
    f = os.path.join(trace_dir, f"layer{L}.npz")
    if not os.path.exists(f):
        return np.ones(n_experts)
    d = np.load(f)
    u = np.zeros(n_experts)
    np.add.at(u, d["indices"].astype(np.int64).reshape(-1), d["weights"].astype(np.float64).reshape(-1))
    return u


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard-dir", required=True)
    ap.add_argument("--trace", required=True, help="dir with layer*.npz, for the usage weights")
    ap.add_argument("--layers", default="0-39")
    ap.add_argument("--experts", type=int, default=24, help="hottest N experts per layer to score")
    ap.add_argument("--bits", default="2,3")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out")
    a = ap.parse_args()
    sys.path.insert(0, os.path.expanduser("~/git/deepseek-v41-flash-spark"))
    from safetensors import safe_open
    from engine.codebook_sim import CodebookSim

    lo, hi = (int(x) for x in a.layers.split("-"))
    bits = [int(b) for b in a.bits.split(",")]
    sims = {b: CodebookSim(b, a.device) for b in bits}
    index = json.load(open(os.path.join(a.shard_dir, "model.safetensors.index.json")))["weight_map"]

    print(f"  {'layer':>5} " + " ".join(f"{'err@%d' % b:>10}" for b in bits) + f"{'ratio 2/3':>11} {'experts':>8}",
          flush=True)
    rows = []
    for L in range(lo, hi + 1):
        u = usage_by_expert(a.trace, L)
        hot = np.argsort(-u)[:a.experts]
        num = {b: 0.0 for b in bits}
        den = 0.0
        handles = {}
        ok = True
        for e in hot.tolist():
            for nm in ("w1", "w2", "w3"):
                key = f"layers.{L}.ffn.experts.{e}.{nm}.weight"
                sk = f"layers.{L}.ffn.experts.{e}.{nm}.scale"
                fn = index.get(key)
                path = os.path.join(a.shard_dir, fn) if fn else None
                if not path or not os.path.exists(path):
                    ok = False
                    break
                if fn not in handles:
                    handles[fn] = safe_open(path, "pt")
                w = handles[fn].get_tensor(key).to(a.device)
                s = handles[fn].get_tensor(sk).to(a.device)
                # deq once, then compare each width against it -- requant_packed wants RAW bytes,
                # and .float() on an F8_E8M0 scale DECODES it (the bug that collapsed the CB3 trace)
                wu, su = w.view(torch.uint8), s.view(torch.uint8)
                ref = _deq(wu, su)                      # the weights the checkpoint actually holds
                for b in bits:
                    q = _deq(sims[b].requant_packed(wu, su), su)
                    num[b] += float(u[e]) * (q - ref).pow(2).sum().item()
                den += float(u[e]) * ref.pow(2).sum().item()
                del ref
            if not ok:
                break
        for h in handles.values():
            del h
        if not ok or den == 0:
            print(f"  {L:>5}   shard not local, skipped", flush=True)
            continue
        vals = {b: num[b] / den for b in bits}
        r = vals[2] / vals[3] if 3 in vals and vals[3] > 0 else float("nan")
        rows.append({"layer": L, **{f"err{b}": vals[b] for b in bits}, "ratio_2_3": r})
        print(f"  {L:>5} " + " ".join(f"{vals[b]:>10.5f}" for b in bits) + f"{r:>11.3f} {len(hot):>8}",
              flush=True)
    if rows:
        rs = sorted(rows, key=lambda x: x["ratio_2_3"])
        print(f"\n  cheapest layers to drop to 2 bits: " +
              ", ".join(str(x["layer"]) for x in rs[:8]), flush=True)
        print(f"  most expensive:                    " +
              ", ".join(str(x["layer"]) for x in rs[-8:]), flush=True)
    if a.out:
        json.dump(rows, open(a.out, "w"), indent=1)
    print("== ALL DONE ==", flush=True)


if __name__ == "__main__":
    main()
