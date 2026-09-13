#!/usr/bin/env python3
"""Is 44% right, and should every layer get the same share?

Two assumptions in 0xBakeer's design, neither derived: the TOTAL residency (44%) and the UNIFORM
per-layer split. Their build_keep_masks offers `uniform` (per-layer top-n) and `global` (one
cross-layer ranking, min_per_layer=24) and measured global WORSE -- but those are two points, not a
search. This sweeps the budget and compares three allocations at EQUAL total slots:

  uniform      every layer gets the same count      (their shipped default)
  global       one cross-layer frequency ranking    (their measured-worse option)
  waterfill    slots go where the MARGINAL coverage gain is highest, greedily

Hold-out throughout: the keep-set never sees the request it is scored on.
"""
import numpy as np, os

Z = np.load(os.environ.get("SIM_IN", "/opt/llm/runners/results/overlay-capture-long.npz"))
names = sorted({k.split("__")[0] for k in Z.files})
data = {n: (Z[f"{n}__routed"], int(Z[f"{n}__meta"][0])) for n in names}
L, K = data[names[0]][0].shape[1], data[names[0]][0].shape[2]
E = int(max(r.max() for r, _ in data.values())) + 1

def hist(r, lo, hi):
    h = np.zeros((L, E), np.int64)
    for l in range(L):
        np.add.at(h[l], r[lo:hi, l, :].ravel(), 1)
    return h

# ---- per-layer concentration on the whole trace -------------------------------------------------
gh_all = np.zeros((L, E), np.int64)
for n in names:
    r, _ = data[n]; gh_all += hist(r, 0, len(r))
print(f"  {len(names)} requests, {L} layers, {E} experts, top-{K}")
print("\n  per-layer concentration (share of picks covered by the top 44% of experts):")
tops = []
for l in range(L):
    s = np.sort(gh_all[l])[::-1]
    tops.append(100.0 * s[:round(0.44*E)].sum() / max(s.sum(), 1))
tops = np.array(tops)
for lo, hi in ((0, 12), (12, 24), (24, 36), (36, 48)):
    print(f"    layers {lo:>2}-{hi-1:<2}  top-44% covers {tops[lo:hi].min():5.1f}-{tops[lo:hi].max():5.1f}%")
print(f"    spread across layers: {tops.min():.1f}% .. {tops.max():.1f}%  <- uniform ignores this")

def coverage(r, npr, keep):
    tot = ok = 0
    for l in range(L):
        ids = r[npr:, l, :].ravel(); tot += ids.size
        ok += int(np.isin(ids, list(keep[l])).sum())
    return 100.0 * ok / tot

def alloc_uniform(gh, total): return [total // L] * L
def alloc_global(gh, total):
    norm = gh / np.maximum(gh.sum(axis=1, keepdims=True), 1)
    flat = [(norm[l, e], l) for l in range(L) for e in range(E)]
    flat.sort(reverse=True)
    c = [24] * L                                   # their min_per_layer floor
    rem = total - sum(c)
    for _, l in flat:
        if rem <= 0: break
        if c[l] < E: c[l] += 1; rem -= 1
    return c
def alloc_waterfill(gh, total):
    """give each next slot to the layer whose next expert carries the most picks"""
    order = [np.sort(gh[l])[::-1] for l in range(L)]
    c = [1] * L; rem = total - L
    import heapq
    heap = [(-order[l][1], l) for l in range(L) if len(order[l]) > 1]
    heapq.heapify(heap)
    while rem > 0 and heap:
        gain, l = heapq.heappop(heap); c[l] += 1; rem -= 1
        if c[l] < E: heapq.heappush(heap, (-order[l][c[l]], l))
    return c

print("\n  coverage at equal total slots (held-out, ranges over requests):")
for frac in (0.20, 0.30, 0.44, 0.60):
    total = round(frac * E) * L
    row = {}
    for nm, fn in (("uniform", alloc_uniform), ("global", alloc_global), ("waterfill", alloc_waterfill)):
        cov = []
        for held in names:
            gh = np.zeros((L, E), np.int64)
            for o in names:
                if o != held:
                    ro, _ = data[o]; gh += hist(ro, 0, len(ro))
            c = fn(gh, total)
            keep = [set(np.argsort(-gh[l], kind="stable")[:c[l]].tolist()) for l in range(L)]
            r, npr = data[held]; cov.append(coverage(r, npr, keep))
        row[nm] = (min(cov), max(cov))
    print(f"    {100*frac:>3.0f}% total: " + "  ".join(f"{k} {v[0]:5.1f}-{v[1]:5.1f}%" for k, v in row.items()))
print("== ALL DONE ==")
