#!/usr/bin/env python3
"""Weight-space error of published EXL3 packs and of our CB2/CB3 against DeepSeek's FP4 experts, same experts.

For each sampled (layer, expert, w1/w2/w3) it range-fetches the FP4 original and the same tensor from every
EXL3 pack (tools/hf_range.py, no shard downloads), decodes EXL3 with TensorFold's float64 reference decoder
(tensorfold/cuda/exl3/format.py), requantizes FP4 through CodebookSim for CB2/CB3, and reports

    rel = ||W_q - W||_F^2 / ||W||_F^2          (W = the FP4 weight, float64)

with the bits each format stores for the tensor. Experts are the hottest of the unmasked routing trace
(notes/data/ds41-routing-unmasked.npz), plus a few cold ones.

This is a PROXY: weight error with an identity Hessian. EXL3 minimises error against calibration activations
(LDLQ), so weight-space error can rank it below its real output quality. Read it as a curve per format, not a
quality verdict.
"""
import argparse, json, os, sys, time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.expanduser("~/git/tensorfold/src"))
sys.path.insert(0, os.path.expanduser("~/git/deepseek-v41-flash-spark"))
from hf_range import Remote  # noqa: E402
from tensorfold.cuda.exl3 import format as exl3  # noqa: E402

FP4 = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0])
PACKS = {
    "sage1.59": "vcruz305/DSV4.1-Flash-SAGE-EXL3-1.59bpw",
    "jay2.8": "jayleaton/DeepSeek-V4.1-Flash-EXL3-2.8bpw",
    "mia2.9": "Mia-AiLab/DeepSeek-V4.1-Flash-EXL3-2.9bpw",
    "sage3.30": "vcruz305/DSV4.1-Flash-SAGE-EXL3-3.30bpw",
}


def fp4(w: np.ndarray, s: np.ndarray) -> np.ndarray:
    w = w.view(np.uint8)
    codes = np.stack([w & 0x0F, w >> 4], axis=-1).reshape(w.shape[0], -1)
    return FP4[codes] * np.repeat(np.exp2(s.astype(np.float64) - 127.0), 32, axis=1)


def exl3_weight(R: Remote, p: str):
    t = R.get(p + ".trellis")
    bits = exl3.bits_of(t.shape)
    cb = "mul1" if R.has(p + ".mul1") else ("mcg" if R.has(p + ".mcg") else "3inst")
    w = exl3.dequantize(t, R.get(p + ".suh"), R.get(p + ".svh"), bits, cb)  # [K, N]
    return w.T, bits, cb


def sample(trace: str, layers, hot: int, cold: int, seed: int = 0):
    z = np.load(trace)
    count = np.zeros((40, 384))
    for k in z.files:
        if k.endswith("__routed"):
            r = z[k].astype(np.int64)  # [tokens, layers, topk]
            for L in range(r.shape[1]):
                np.add.at(count[L], r[:, L].reshape(-1), 1)
    rng = np.random.default_rng(seed)
    out = []
    for L in layers:
        order = np.argsort(-count[L])
        out += [(L, int(e), float(count[L, e])) for e in order[:hot]]
        out += [(L, int(e), float(count[L, e])) for e in rng.choice(order[hot:], cold, replace=False)]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", default="2,10,20,30,38")
    ap.add_argument("--hot", type=int, default=3)
    ap.add_argument("--cold", type=int, default=1)
    ap.add_argument("--packs", default=",".join(PACKS))
    ap.add_argument("--trace", default=os.path.expanduser("~/git/deepseek-moe-gb10/notes/data/ds41-routing-unmasked.npz"))
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from engine.codebook_sim import CodebookSim
    sims = {b: CodebookSim(b, "cuda") for b in (2, 3)}
    src = Remote("deepseek-ai/DeepSeek-V4.1-Flash")
    packs = {k: Remote(PACKS[k]) for k in a.packs.split(",")}
    rows = []
    for L, e, uses in sample(a.trace, [int(x) for x in a.layers.split(",")], a.hot, a.cold):
        for w in ("w1", "w2", "w3"):
            p = f"layers.{L}.ffn.experts.{e}.{w}"
            t0 = time.time()
            ww, ss = src.get(p + ".weight"), src.get(p + ".scale")
            W = fp4(ww, ss)
            norm = float((W * W).sum())
            row = {"layer": L, "expert": e, "proj": w, "uses": uses, "shape": list(W.shape)}
            for b, sim in sims.items():  # CB keeps the FP4 scales: b + 8/32 bits a weight
                wt, st = torch.from_numpy(ww.view(np.uint8)).cuda(), torch.from_numpy(ss.view(np.uint8)).cuda()
                q = sim.requant_packed(wt, st).cpu().numpy()
                d = fp4(q, ss) - W
                row[f"cb{b}"] = {"bits": b + 0.25, "rel": float((d * d).sum()) / norm}
            for name, R in packs.items():
                if not R.has(p + ".trellis"):
                    continue
                Q, bits, cb = exl3_weight(R, p)
                d = Q - W
                row[name] = {"bits": bits, "cb": cb, "rel": float((d * d).sum()) / norm}
            rows.append(row)
            print(json.dumps(row), f"{time.time() - t0:.1f}s", flush=True)
    json.dump(rows, open(a.out, "w"))


if __name__ == "__main__":
    main()
