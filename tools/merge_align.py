#!/usr/bin/env python3
"""Are DeepSeek-V4.1's routed experts mergeable once their hidden neurons are aligned?

delta_spectrum.py found no shared base (W - mean(W) as high-rank as W itself), but it averaged experts without
aligning them, and an expert's intermediate neurons are permutation-symmetric: neuron j is (w1[j], w3[j], w2[:, j])
and any reordering computes the same function. Merging methods (MC-SMoE, HC-SMoE) align before they average.

For pairs of one layer's experts this matches neurons by the Hungarian algorithm on the dot product of their
concatenated (w1 row, w3 row, w2 column) and reports

    rel = ||F_a - P F_b||^2 / ||F_a||^2      unaligned (P = I) and aligned

~2 means unrelated (equal norms, orthogonal), 0 identical. Averaging the pair costs each member rel / 4 of its
own norm. For scale: EXL3 2-bit costs ~0.10 (notes/exl3-2bit.md). Weight distance is a strict proxy: two experts
can compute similar outputs on the inputs they actually get while differing in weights.
"""
import argparse, itertools, json, os, sys

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, os.path.dirname(__file__))
from hf_range import Remote  # noqa: E402
from exl3_vs_fp4 import fp4, sample  # noqa: E402


def neurons(R: Remote, L: int, e: int) -> torch.Tensor:
    p = f"layers.{L}.ffn.experts.{e}"
    w = {n: fp4(R.get(f"{p}.{n}.weight"), R.get(f"{p}.{n}.scale")) for n in ("w1", "w3", "w2")}
    return torch.from_numpy(np.concatenate([w["w1"], w["w3"], w["w2"].T], axis=1)).cuda()  # [2304, 3 * 5120]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", default="10,30")
    ap.add_argument("--hot", type=int, default=8)
    ap.add_argument("--cold", type=int, default=2)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    R = Remote("deepseek-ai/DeepSeek-V4.1-Flash")
    trace = os.path.expanduser("~/git/deepseek-moe-gb10/notes/data/ds41-routing-unmasked.npz")
    rows = []
    for L in [int(x) for x in a.layers.split(",")]:
        picks = sample(trace, [L], a.hot, a.cold)
        F = {e: neurons(R, L, e) for _, e, _ in picks}
        for x, y in itertools.combinations(F, 2):
            A, B = F[x], F[y]
            na = float((A * A).sum())
            un = float(((A - B) ** 2).sum()) / na
            col = linear_sum_assignment((A @ B.T).double().cpu().numpy(), maximize=True)[1]
            al = float(((A - B[torch.from_numpy(col).cuda()]) ** 2).sum()) / na
            rows.append({"layer": L, "a": x, "b": y, "unaligned": un, "aligned": al})
            print(json.dumps(rows[-1]), flush=True)
    json.dump(rows, open(a.out, "w"))
    v = np.array([r["aligned"] for r in rows])
    print(f"aligned rel: min {v.min():.3f}  median {np.median(v):.3f}  max {v.max():.3f}; "
          f"unaligned median {np.median([r['unaligned'] for r in rows]):.3f}")


if __name__ == "__main__":
    main()
