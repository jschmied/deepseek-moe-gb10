#!/usr/bin/env python3
"""How wide is the expert union over a speculative verify block, measured?

The GB/token arithmetic in our notes rests on a stored figure of "20.96 distinct experts per layer
per verify block". That number decides whether a wider block is nearly free (more queue depth, the
same bytes) or the main cost (more bytes). This measures it from the routing trace: for block sizes
k = 1..8, the mean number of DISTINCT experts a layer touches over k consecutive decode tokens.

Also reports the marginal cost of widening -- experts added per extra position -- which is the
quantity the verify-width conflict between Qwen and DS4.1 is actually about.
"""
import argparse, numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--trace", required=True)
ap.add_argument("--max-block", type=int, default=8)
a = ap.parse_args()

Z = np.load(a.trace)
names = sorted({k.split("__")[0] for k in Z.files})
reqs = [(n, Z[f"{n}__routed"], int(Z[f"{n}__meta"][0])) for n in names]
L, K = reqs[0][1].shape[1], reqs[0][1].shape[2]
E = int(max(r.max() for _, r, _ in reqs)) + 1
print(f"  {len(reqs)} requests, {L} layers, {E} experts, top-{K}")

print(f"\n  {'block k':>7} {'distinct/layer':>15} {'per step (40L)':>15} {'marginal':>10} {'per token':>11}")
prev = None
for k in range(1, a.max_block + 1):
    tot, cnt = 0, 0
    for _, r, npr in reqs:
        dec = r[npr:]
        n = dec.shape[0] - k + 1
        if n <= 0:
            continue
        for l in range(L):
            col = dec[:, l, :]
            for i in range(0, n, k):                 # non-overlapping blocks
                tot += len(np.unique(col[i:i + k]))
                cnt += 1
    m = tot / cnt
    marg = "" if prev is None else f"{m - prev:+.2f}"
    print(f"  {k:>7} {m:15.2f} {m * L:15.1f} {marg:>10} {m * L / k:11.1f}")
    prev = m
print("\n  'per token' = distinct experts a step touches divided by the tokens it verifies;")
print("  flat means widening is free in bytes, rising means it is not.")
print("== ALL DONE ==")
