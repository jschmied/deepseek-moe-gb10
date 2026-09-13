#!/usr/bin/env python3
"""Is a static core needed at all, or is a fully adaptive cache of the same size better?

Follow-up to a mislabelled result: what scored 91.6-93.4% was not "core + 1% ring" but a fully
adaptive cache of the same TOTAL size, because arbitrary eviction could drop core entries too. That
is a different design worth pricing properly. All arms have IDENTICAL total residency.

  static        44% by frequency from the other requests, frozen          (0xBakeer today)
  core+dyn      43% static core + 1% LRU
  half+half     22% static core + 22% LRU
  adaptive      0% core, 44% LRU, warm-started from the frequency ranking
"""
import numpy as np, os, collections

Z = np.load(os.environ.get("SIM_IN", "/opt/llm/runners/results/overlay-capture-long.npz"))
names = sorted({k.split("__")[0] for k in Z.files})
data = {n: (Z[f"{n}__routed"], int(Z[f"{n}__meta"][0])) for n in names}
L, E = data[names[0]][0].shape[1], int(max(r.max() for r, _ in data.values())) + 1
TOTAL = round(0.44 * E)

def hist(r, lo, hi):
    h = np.zeros((L, E), np.int64)
    for l in range(L):
        np.add.at(h[l], r[lo:hi, l, :].ravel(), 1)
    return h

def run(seq, core, warm, cap):
    """core = frozen set; warm = initial LRU contents; cap = LRU capacity."""
    hits = loads = 0
    dyn = collections.OrderedDict((e, 1) for e in warm)
    for e in seq:
        if e in core: hits += 1; continue
        if e in dyn:  hits += 1; dyn.move_to_end(e); continue
        loads += 1
        if len(dyn) >= cap and dyn: dyn.popitem(last=False)
        if cap: dyn[e] = 1
    return hits, loads

arms = [("static  44% frozen",        TOTAL, 0),
        ("core 43% + LRU  1%",  round(0.43*E), TOTAL-round(0.43*E)),
        ("core 22% + LRU 22%",  round(0.22*E), TOTAL-round(0.22*E)),
        ("adaptive 0% + LRU 44%",           0, TOTAL)]
print(f"  {len(names)} requests, {L} layers, {E} experts | total residency {TOTAL} = {100*TOTAL/E:.0f}% in every arm")
for label, ncore, ndyn in arms:
    cov, ld = [], []
    for held in names:
        gh = np.zeros((L, E), np.int64)
        for o in names:
            if o != held:
                ro, _ = data[o]; gh += hist(ro, 0, len(ro))
        H = D = T = 0
        r, npr = data[held]
        for l in range(L):
            rank = np.argsort(-gh[l], kind="stable")
            core = set(rank[:ncore].tolist())
            warm = [int(x) for x in rank[ncore:ncore+ndyn]]      # warm-start LRU from frequency
            seq = [int(x) for x in r[npr:, l, :].ravel()]
            h, d = run(seq, core, warm, ndyn)
            H += h; D += d; T += len(seq)
        cov.append(100.0*H/T); ld.append(D/max(len(r)-npr, 1))
    print(f"  {label:<24} coverage {min(cov):5.1f}-{max(cov):5.1f}%   loads/tok {min(ld):6.1f}-{max(ld):6.1f}")
print("== ALL DONE ==")
