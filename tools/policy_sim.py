#!/usr/bin/env python3
"""Which eviction policy for the dynamic slots? FIFO ring vs LRU vs LFU vs Belady.

My first pass labelled the dynamic set a "transient ring" (after 0xBakeer's TRANSIENT_SLOTS=8) but
implemented set.pop() -- arbitrary eviction, neither a ring nor LRU. That scored 91.6-93.4%, which is
therefore an unprincipled floor, not a policy result. This compares the real ones at equal capacity:

  ring    FIFO round-robin over the transient slots  (their shipped design)
  lru     evict least recently used
  lfu     evict least frequently used within the request
  belady  evict the entry reused furthest in the future -- OPTIMAL, unimplementable, an upper bound

The static core is identical in every arm and never sees the held-out request.
"""
import numpy as np, os, collections, heapq

Z = np.load(os.environ.get("SIM_IN", "/opt/llm/runners/results/overlay-capture-long.npz"))
names = sorted({k.split("__")[0] for k in Z.files})
data = {n: (Z[f"{n}__routed"], int(Z[f"{n}__meta"][0])) for n in names}
L, E = data[names[0]][0].shape[1], int(max(r.max() for r, _ in data.values())) + 1
CORE = round(0.43 * E)          # static core
DYN  = round(0.01 * E)          # dynamic slots per layer

def hist(r, lo, hi):
    h = np.zeros((L, E), np.int64)
    for l in range(L):
        np.add.at(h[l], r[lo:hi, l, :].ravel(), 1)
    return h

def run(seq, core, cap, policy, future=None):
    """seq: list of expert ids for ONE layer over decode. returns (hits, loads)."""
    hits = loads = 0
    dyn = collections.OrderedDict()      # id -> freq
    fifo = collections.deque()
    for i, e in enumerate(seq):
        if e in core:
            hits += 1; continue
        if e in dyn:
            hits += 1
            if policy == "lru": dyn.move_to_end(e)
            if policy == "lfu": dyn[e] += 1
            continue
        loads += 1
        if len(dyn) >= cap:
            if policy == "ring":   dyn.pop(fifo.popleft(), None)
            elif policy == "lru":  dyn.popitem(last=False)
            elif policy == "lfu":  dyn.pop(min(dyn, key=dyn.get), None)
            elif policy == "belady":
                nxt = {}
                for k in dyn:
                    nxt[k] = future[k].get(i, 1 << 30)
                dyn.pop(max(nxt, key=nxt.get), None)
        dyn[e] = 1
        if policy == "ring": fifo.append(e)
    return hits, loads

print(f"  {len(names)} requests, {L} layers, {E} experts | core {CORE} + dyn {DYN} = {100*(CORE+DYN)/E:.0f}%")
res = {p: ([], []) for p in ("ring", "lru", "lfu", "belady")}
for held in names:
    gh = np.zeros((L, E), np.int64)
    for o in names:
        if o != held:
            ro, _ = data[o]; gh += hist(ro, 0, len(ro))
    core = [set(np.argsort(-gh[l], kind="stable")[:CORE].tolist()) for l in range(L)]
    r, npr = data[held]
    for p in res:
        H = Ld = T = 0
        for l in range(L):
            seq = [int(x) for x in r[npr:, l, :].ravel()]
            fut = None
            if p == "belady":                      # next-use index per expert
                fut = collections.defaultdict(dict)
                nxt = {}
                for i in range(len(seq) - 1, -1, -1):
                    fut[seq[i]][i] = nxt.get(seq[i], 1 << 30); nxt[seq[i]] = i
            h, ld = run(seq, core[l], DYN, p, fut)
            H += h; Ld += ld; T += len(seq)
        res[p][0].append(100.0 * H / T); res[p][1].append(Ld / max(len(r) - npr, 1))
for p in ("ring", "lru", "lfu", "belady"):
    c, ld = res[p]
    print(f"  {p:<7} coverage {min(c):5.1f}-{max(c):5.1f}%   loads/tok {min(ld):6.1f}-{max(ld):6.1f}")
print("== ALL DONE ==")
