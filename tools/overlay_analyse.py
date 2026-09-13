#!/usr/bin/env python3
"""Does a prompt-derived overlay beat a larger static union? Offline, on the captured routing.

Compares, per held-out request, decode-phase expert coverage of:
    A) static  K%  union core   (built from the OTHER requests)
    B) (K-O)% core + O% overlay chosen from THIS request's PROMPT routing only

HOLD-OUT IS THE WHOLE POINT. ds-02's lesson: a keep-set scored on the corpus it was built from is
circular. The core here never sees the request it is evaluated on.
"""
import numpy as np, os, sys, itertools

Z = np.load(os.environ.get("OV_OUT", "/opt/llm/runners/results/overlay-capture.npz"))
names = sorted({k.split("__")[0] for k in Z.files})
data = {n: (Z[f"{n}__routed"], Z[f"{n}__meta"]) for n in names}
L = data[names[0]][0].shape[1]
E = int(max(r.max() for r, _ in data.values())) + 1
print(f"  {len(names)} requests, {L} MoE layers, {E} experts seen, top-{data[names[0]][0].shape[2]}")

def hist(routed, lo, hi):
    """per-layer expert counts over token rows [lo:hi)"""
    h = np.zeros((L, E), dtype=np.int64)
    sl = routed[lo:hi]
    for l in range(L):
        np.add.at(h[l], sl[:, l, :].ravel(), 1)
    return h

def topset(h, n):
    return [set(np.argsort(-h[l], kind="stable")[:n].tolist()) for l in range(L)]

def coverage(routed, lo, hi, keep):
    tot = ok = 0
    sl = routed[lo:hi]
    for l in range(L):
        ids = sl[:, l, :].ravel()
        tot += ids.size
        ok  += sum(1 for e in ids if e in keep[l])
    return 100.0 * ok / max(tot, 1)

for K, O in ((0.44, 0.00), (0.44, 0.06), (0.38, 0.06), (0.34, 0.10)):
    nk, no = round((K - O) * E), round(O * E)
    rows = []
    for held in names:
        r, m = data[held]; npr = int(m[0])
        # core from the OTHER requests only, prompt+decode
        gh = np.zeros((L, E), dtype=np.int64)
        for other in names:
            if other == held: continue
            ro, mo = data[other]; gh += hist(ro, 0, len(ro))
        core = topset(gh, nk)
        if no:
            ph = hist(r, 0, npr)                      # THIS request's prompt routing only
            keep = []
            for l in range(L):
                cold = [e for e in np.argsort(-ph[l], kind="stable") if e not in core[l]][:no]
                keep.append(core[l] | set(int(e) for e in cold))
        else:
            keep = topset(gh, nk + no)                # static: spend the whole budget on the core
        rows.append(coverage(r, npr, len(r), keep))    # DECODE-phase coverage only
    lbl = f"core {100*(K-O):.0f}% + overlay {100*O:.0f}%" if no else f"static {100*K:.0f}% union"
    print(f"  {lbl:<28} decode coverage {min(rows):5.2f}-{max(rows):5.2f}%  (n={len(rows)} held-out)")
print("== ALL DONE ==")
