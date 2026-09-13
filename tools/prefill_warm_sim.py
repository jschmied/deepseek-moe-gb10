#!/usr/bin/env python3
"""Does seeding the expert cache from THIS request's own prefill beat a global keep-set?

ds-06 showed adaptive LRU beats a frozen global keep-set by ~20 points at equal capacity.
This asks the next question: where should the cache START?  At serve time the prefill of the
current request is already known before a single token is decoded, and it is drawn from exactly
the domain the decode will be in.  Arms, all at identical capacity and all LRU on the dynamic
slots -- only the INITIAL core differs:

  global    core ranked on the other requests' full traces        (what a frozen keep-set does)
  prefill   core ranked on THIS request's prefill only            (free domain adaptation)
  blend-f   fraction f of the core from prefill, rest from global
  oracle    core ranked on THIS request's own DECODE tokens       (cheating; upper bound)
  cold      no core at all, pure LRU over the whole capacity      (floor for "is the core worth it")

Leave-one-out over the requests in the trace.  Reports coverage (% of expert activations already
resident) and loads per decode token.
"""
import numpy as np, os, sys, collections

TRACE = sys.argv[1] if len(sys.argv) > 1 else "notes/data/qwen-routing-long.npz"
CAP_FRAC = float(os.environ.get("CAP", "0.44"))     # total arena as a fraction of all experts
DYN_FRAC = float(os.environ.get("DYN", "0.10"))     # share of the arena left dynamic
PRE_N    = int(os.environ.get("PRE_N", "0"))        # use only the first N prefill tokens (0 = all)

Z = np.load(TRACE)
names = sorted({k.split("__")[0] for k in Z.files})
data = {n: (Z[f"{n}__routed"], int(Z[f"{n}__meta"][0])) for n in names}
L = data[names[0]][0].shape[1]
E = int(max(r.max() for r, _ in data.values())) + 1
CAP = round(CAP_FRAC * E)
DYN = max(1, round(DYN_FRAC * CAP))
CORE = CAP - DYN


def hist(r, lo, hi):
    h = np.zeros((L, E), np.int64)
    for l in range(L):
        np.add.at(h[l], r[lo:hi, l, :].ravel(), 1)
    return h


def rank(h, l, k, exclude=()):
    order = np.argsort(-h[l], kind="stable")
    out = []
    for e in order:
        if len(out) >= k:
            break
        if e in exclude:
            continue
        out.append(int(e))
    return out


def lru_run(seq, core, cap):
    """core = set of always-resident ids; cap = dynamic slots, LRU."""
    hits = loads = 0
    dyn = collections.OrderedDict()
    for e in seq:
        if e in core:
            hits += 1
            continue
        if e in dyn:
            hits += 1
            dyn.move_to_end(e)
            continue
        loads += 1
        if len(dyn) >= cap:
            dyn.popitem(last=False)
        dyn[e] = 1
    return hits, loads


BLENDS = [0.25, 0.50, 0.75]
arms = ["global", "prefill"] + [f"blend-{b:.2f}" for b in BLENDS] + ["oracle", "cold"]
res = {a: ([], []) for a in arms}

print(f"trace {TRACE}")
print(f"  {len(names)} requests, {L} layers, {E} experts")
print(f"  arena {CAP} = core {CORE} + dyn {DYN}  ({100*CAP/E:.0f}% of experts)"
      f"   prefill window {'all' if PRE_N == 0 else PRE_N}")

for held in names:
    gh = np.zeros((L, E), np.int64)
    for o in names:
        if o != held:
            ro, _ = data[o]
            gh += hist(ro, 0, len(ro))
    r, npr = data[held]
    pre_hi = npr if PRE_N == 0 else min(npr, PRE_N)
    ph = hist(r, 0, pre_hi)
    dh = hist(r, npr, len(r))

    cores = {}
    cores["global"] = [set(rank(gh, l, CORE)) for l in range(L)]
    cores["prefill"] = [set(rank(ph, l, CORE)) for l in range(L)]
    cores["oracle"] = [set(rank(dh, l, CORE)) for l in range(L)]
    cores["cold"] = [set() for _ in range(L)]
    for b in BLENDS:
        k = round(b * CORE)
        cl = []
        for l in range(L):
            p = rank(ph, l, k)
            g = rank(gh, l, CORE - len(p), exclude=set(p))
            cl.append(set(p) | set(g))
        cores[f"blend-{b:.2f}"] = cl

    ntok = len(r) - npr
    for a in arms:
        H = Ld = T = 0
        for l in range(L):
            seq = [int(x) for x in r[npr:, l, :].ravel()]
            slots = DYN if a != "cold" else CAP
            h, ld = lru_run(seq, cores[a][l], slots)
            H += h; Ld += ld; T += len(seq)
        res[a][0].append(100.0 * H / T)
        res[a][1].append(Ld / ntok)

print()
base = float(np.mean(res["global"][0]))
for a in arms:
    c, ld = res[a]
    print(f"  {a:<11} coverage {np.mean(c):5.1f}%  [{min(c):5.1f}-{max(c):5.1f}]"
          f"   loads/tok {np.mean(ld):6.1f}   delta {np.mean(c)-base:+5.1f} pp")
print()
print("  per request (coverage %):")
hdr = "    " + "".join(f"{a[:9]:>10}" for a in arms)
print(hdr)
for i, n in enumerate(names):
    print(f"    {n:<10}" + "".join(f"{res[a][0][i]:10.1f}" for a in arms))
print("== ALL DONE ==")
