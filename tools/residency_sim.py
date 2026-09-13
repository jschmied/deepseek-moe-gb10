#!/usr/bin/env python3
"""Simulate expert-residency strategies against a real routing trace, before building anything.

The point (user, 2026-09-13): stream/trace with the router UNMASKED, get the full access pattern,
then decide the optimum offline. A masked engine reports hit rate 1.0 by construction and can never
tell you what it is missing -- ds-02 measured 16.9% of an unmasked router's picks falling outside
0xBakeer's 44% keep-set.

Input: an npz of {name}__routed [tokens, layers, topk] + {name}__meta [n_prompt, n_gen].
Works on the Qwen captures today; the same shape comes out of 0xBakeer's tools/expert_trace.py
(per-token top-6 ids per layer), so DS4.1 traces drop straight in.

Reported per strategy, as RANGES over held-out requests:
  coverage   % of DECODE expert selections served from the resident set
  cold/tok   expert loads per decoded token -> this is what becomes SSD traffic
  churn      distinct experts admitted over the request (arena write pressure)
"""
import numpy as np, os, sys, collections

Z = np.load(os.environ.get("SIM_IN", "/opt/llm/runners/results/overlay-capture-long.npz"))
names = sorted({k.split("__")[0] for k in Z.files})
data = {n: (Z[f"{n}__routed"], int(Z[f"{n}__meta"][0])) for n in names}
L, K = data[names[0]][0].shape[1], data[names[0]][0].shape[2]
E = int(max(r.max() for r, _ in data.values())) + 1
print(f"  trace: {len(names)} requests, {L} layers, {E} experts, top-{K}")

def hist(r, lo, hi):
    h = np.zeros((L, E), np.int64)
    for l in range(L):
        np.add.at(h[l], r[lo:hi, l, :].ravel(), 1)
    return h

def topset(h, n):
    return [set(np.argsort(-h[l], kind="stable")[:n].tolist()) for l in range(L)]

def score(r, npr, keep_per_layer, dynamic=None):
    """coverage / cold-per-token / churn over the DECODE phase."""
    tot = miss = 0
    admitted = [set() for _ in range(L)]
    live = [set(s) for s in keep_per_layer]
    for t in range(npr, len(r)):
        for l in range(L):
            for e in r[t, l, :]:
                e = int(e); tot += 1
                if e not in live[l]:
                    miss += 1
                    if dynamic is not None:
                        dynamic(live[l], admitted[l], e)
    ntok = max(len(r) - npr, 1)
    return 100.0 * (tot - miss) / tot, miss / ntok, sum(len(a) for a in admitted) / L

def lru_admit(cap):
    def f(live, admitted, e):
        live.add(e); admitted.add(e)
        if len(live) > cap:                      # evict arbitrary (set order) -- crude LRU stand-in
            live.pop()
    return f

for frac, ov, label in ((0.44, 0.00, "static 44% frequency"),
                        (0.44, 0.06, "38% core + 6% prompt overlay"),
                        (0.44, 0.00, "static 44% + transient ring (1%)"),
                        (1.00, 0.00, "ORACLE: all experts resident")):
    nk = round((frac - ov) * E); no = round(ov * E)
    cov, cold, churn = [], [], []
    for held in names:
        r, npr = data[held]
        gh = np.zeros((L, E), np.int64)
        for o in names:
            if o != held:
                ro, _ = data[o]; gh += hist(ro, 0, len(ro))
        core = topset(gh, nk)
        if no:
            ph = hist(r, 0, npr)
            keep = [core[l] | {int(e) for e in
                     [x for x in np.argsort(-ph[l], kind="stable") if x not in core[l]][:no]}
                    for l in range(L)]
        elif frac >= 1.0:
            keep = [set(range(E)) for _ in range(L)]
        else:
            keep = topset(gh, nk + no)
        dyn = lru_admit(nk + no + round(0.01 * E)) if "transient" in label else None
        c, m, ch = score(r, npr, keep, dyn)
        cov.append(c); cold.append(m); churn.append(ch)
    print(f"  {label:<32} cov {min(cov):5.1f}-{max(cov):5.1f}%   cold/tok {min(cold):5.2f}-{max(cold):5.2f}   churn {min(churn):5.1f}-{max(churn):5.1f}")
print("== ALL DONE ==")
