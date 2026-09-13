#!/usr/bin/env python3
"""Expert usage distribution from our own UNMASKED DS4.1 trace.

The question under every keep-set, pruning and residency argument is how skewed routing actually is.
0xBakeer ships coverage.png and layer_hist.png from their trace, but that trace was taken through a
masked router; ours is unmasked, so this is what the model wants rather than what it was allowed.

Reports, per layer and pooled: the sorted frequency curve, what share of all routing the top-N
experts capture, and how far the coding and general halves of the corpus disagree.
"""
import argparse, collections, os
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--trace", default=os.path.expanduser(
    "~/git/deepseek-moe-gb10/notes/data/ds41-routing-unmasked.npz"))
ap.add_argument("--decode-only", action="store_true", default=True)
a = ap.parse_args()

Z = np.load(a.trace)
names = sorted({k.split("__")[0] for k in Z.files})
L = Z[f"{names[0]}__routed"].shape[1]
E = int(max(Z[f"{n}__routed"].max() for n in names)) + 1

hist = np.zeros((L, E), np.int64)
by_cat = {"code": np.zeros((L, E), np.int64), "gen": np.zeros((L, E), np.int64)}
for n in names:
    r = Z[f"{n}__routed"]; npr = int(Z[f"{n}__meta"][0])
    dec = r[npr:] if a.decode_only else r
    cat = "code" if n.startswith("code") else "gen"
    for l in range(L):
        b = np.bincount(dec[:, l, :].ravel(), minlength=E)
        hist[l] += b; by_cat[cat][l] += b

tot = hist.sum()
print(f"  {len(names)} requests, {L} layers, {E} experts, {tot:,} routing decisions (decode only)")
print(f"  uniform would be {tot/(L*E):,.0f} per (layer, expert)\n")

# pooled top-N coverage: the keep-set question
pooled = np.sort(hist, axis=1)[:, ::-1]            # per layer, descending
cum = pooled.cumsum(axis=1) / pooled.sum(axis=1, keepdims=True)
print(f"  share of routing captured by the top N experts OF EACH LAYER (mean over 40 layers):")
print(f"  {'N':>5} {'% of E':>7} {'mean':>7} {'min layer':>10} {'max layer':>10}")
for N in (32, 64, 96, 128, 154, 169, 192, 256, 320):
    c = cum[:, N - 1]
    print(f"  {N:>5} {100*N/E:>6.1f}% {100*c.mean():>6.1f}% {100*c.min():>9.1f}% {100*c.max():>9.1f}%")

# how flat is it really
p = hist / hist.sum(axis=1, keepdims=True)
ent = -(p * np.log2(np.maximum(p, 1e-12))).sum(axis=1)
print(f"\n  per-layer entropy: {ent.min():.2f}-{ent.max():.2f} bits of {np.log2(E):.2f} "
      f"(uniform); mean {ent.mean():.2f}")
ratio = pooled[:, 0] / np.maximum(pooled[:, E // 2], 1)
print(f"  hottest expert / median expert, per layer: {ratio.min():.1f}x - {ratio.max():.1f}x, "
      f"median {np.median(ratio):.1f}x")

# coding vs general disagreement -- the cross-domain keep-set problem
print(f"\n  coding vs general, top-25% sets per layer:")
k = E // 4
js = []
for l in range(L):
    cs = set(np.argsort(-by_cat['code'][l])[:k].tolist())
    gs = set(np.argsort(-by_cat['gen'][l])[:k].tolist())
    js.append(len(cs & gs) / len(cs | gs))
js = np.array(js)
print(f"    Jaccard {js.min():.3f}-{js.max():.3f}, mean {js.mean():.3f}")

# an actual curve, in text
print(f"\n  sorted frequency, layer 20 (log2 count, 384 experts left to right):")
row = pooled[min(20, L - 1)]
bars = " .:-=+*#%@"
mx = np.log2(max(row.max(), 1))
line = "".join(bars[min(int(len(bars) * np.log2(max(v, 1)) / mx), len(bars) - 1)] for v in row)
for i in range(0, E, 96):
    print(f"    {i:>3}-{min(i+95,E-1):>3} |{line[i:i+96]}|")
print(f"    scale: '{bars[-1]}' = {row.max():,} uses, ' ' = 0")
print("== ALL DONE ==")
