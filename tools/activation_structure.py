#!/usr/bin/env python3
"""activation_structure.py -- is the expert set we ALREADY KNOW we need shaped so we can exploit it?

`tools/prefetch_oracle.py` closed the forward question: short-history recurrence names 0.0 % of
LRU's misses, because LRU already retains ~57 tokens and the misses are experts unused for a median
of 45 tokens. This tool asks the orthogonal question, which needs no prediction at all: **the moment
the router has resolved, we hold the exact list of experts this layer must fetch. Is that list
structured?** If it is, the win is in the read path (fewer, larger NVMe reads), not in the
predictor, and it is bit-exact by construction.

Four measurements, cheapest and most decisive first.

  Q1  CO-ACTIVATION. Within a layer, do experts co-occur across tokens more than their marginal
      rates imply -- and, the part that actually matters, do the MISSES co-occur? A recurring small
      group of co-missing experts could be stored adjacently and read as one record.
  Q2  CONTIGUITY IN SLOT ORDER. Missed expert ids at one resolve call: numerically clustered, or
      scattered over 0..383? Coalesce runs at gap tolerance t and price the overfetch. Q2b then asks
      whether a LEARNED per-layer expert ordering (a repack of the on-disk order, fitted on the
      first half of the trace and scored on the second) makes the same misses coalescable.
  Q3  TOKEN-IDENTITY CONDITIONING. Needs token ids. This log does not carry them; the tool says so
      and skips rather than inventing a proxy.
  Q4  SLOW STATE. Jaccard between a layer's active set at t and t+d. Calibrates the 45-token
      horizon from the other side.

NULLS -- every structural claim here has one, because every one of them is meaningless without it.

* Independence null (Q1, analytic). Under independent shuffling of each expert's column,
  co-occurrence of a pair is Hypergeometric(T, n_e, n_f): mean n_e*n_f/T, var
  n_e*n_f*(T-n_e)*(T-n_f) / (T^2*(T-1)). Gives a per-pair z.
* Curveball null (Q1, decisive). The independence null is confounded: set size per token varies
  (8..36 experts), and a fixed set size of k out of 384 also induces a small NEGATIVE pair
  correlation on its own. Curveball (Strona et al. 2014) trades elements between two token rows,
  preserving BOTH the per-token set size AND every expert's marginal rate exactly, while destroying
  any pair structure. The statistic compared against it is the dispersion sum_{e<f} C_ef^2 and the
  number of DISTINCT pairs that ever co-occur -- the total number of co-occurring pairs is fixed by
  the row sums and therefore cannot differ from the null, which is why "experts co-occur" is not a
  measurement.
* Uniform-ids null (Q2). Draw the same number of missed ids per call uniformly without replacement
  from 0..383 and coalesce identically. Reported next to every real number.
* Random-permutation null (Q2b). The learned repack is scored against a random per-layer
  permutation on the same held-out half, so "the reordering helped" cannot be an artifact of
  reordering per se.
* Random-pair null (Q4). Jaccard between two tokens of the same layer drawn at random: the
  asymptote a decaying curve must be compared against.

PRICING. A miss is 13,774,848 B of NVMe on the shipped server (one native CB3 record,
notes/native-cb3-expert-cache.md). Coalescing two wanted experts across one unwanted slot buys one
read and costs one whole extra record. Every coalescing arm below therefore reports BOTH reads and
bytes, and the verdict ranks on the break-even: a read-size change has to buy more bandwidth than
the overfetch throws away. Measured context (notes/ds41-measured-2026-09-13.md sec.13,
notes/gb10-arena-io-measured.md): the engine achieves 2.3-3.0 GB/s against 5.0-5.6 GB/s for a single
large pread and 6.8 GB/s at two in flight.

LAYOUT CAVEAT, and it bounds Q2 entirely. "Coalescing" assumes experts of one layer sit in expert-id
order in one contiguous on-disk region, so that ids e and e+1 are adjacent records. That is true of
the PROPOSED native CB3 cache (notes/native-cb3-expert-cache.md, "expert-major, contiguous"). Today's
FP4 safetensors shards store each expert as TWO runs ~585 MB apart (all scales in front, all weights
behind; `ShardFile.expert_runs`), so a coalesced range today would be two coalesced ranges, and
whether experts are id-ordered within each group was not verifiable from the lean checkout on this
box. Read Q2 as "what the repacked layout would buy", not as a patch to the shipped reader.

Usage:
    activation_structure.py ~/ds41-queue/logs/route-decode.jsonl [--reps 10] [--seed 0]
CPU only; no GPU, no server.
"""

from __future__ import annotations

import argparse
import collections
import json
import random
import sys

import numpy as np

N_EXPERTS = 384
N_LAYERS = 40
EXPERT_BYTES = 13_774_848        # one native CB3 disk record; what a miss really costs


# --------------------------------------------------------------------------- load

def load(path):
    """-> (decode_calls, has_token_ids, cyclic_violations).

    decode_calls: [(layer, tuple(active ids), tuple(missed ids)), ...] in forward order, pf lines
    dropped. The log is cyclic 0..39, so call index // 40 is the token (decode step) index -- the
    tool verifies that rather than assuming it.
    """
    calls, fields = [], set()
    prev_L = None
    bad = 0
    for line in open(path):
        line = line.strip()
        if not line:
            continue
        o = json.loads(line)
        fields |= set(o.keys())
        if o.get("pf"):
            continue
        L = o["L"]
        if prev_L is not None and L != (prev_L + 1) % N_LAYERS:
            bad += 1
        prev_L = L
        calls.append((L, tuple(o["uniq"]), tuple(o["miss"])))
    has_tok = bool(fields & {"tok", "token", "token_id", "tid", "t"})
    return calls, has_tok, bad, fields


def by_layer(calls, which):
    """-> {layer: [set of ids per token]}; which=1 active, 2 missed."""
    d = collections.defaultdict(list)
    for c in calls:
        d[c[0]].append(set(c[which]))
    return d


# --------------------------------------------------------------------------- Q1 machinery

def cooc(rows, n=N_EXPERTS):
    """Co-occurrence matrix over tokens for one layer. rows = [set of ids]."""
    X = np.zeros((len(rows), n), dtype=np.float32)
    for t, s in enumerate(rows):
        if s:
            X[t, list(s)] = 1.0
    return X.T @ X, X


def dispersion(C):
    """sum_{e<f} C_ef^2 and the number of distinct pairs with C_ef > 0."""
    off = C.copy()
    np.fill_diagonal(off, 0.0)
    return float((off * off).sum()) / 2.0, int((off > 0).sum()) // 2


def curveball(rows, rng, sweeps=5):
    """Randomise preserving BOTH row sums (set size per token) and column sums (expert rates)."""
    rows = [set(s) for s in rows]
    T = len(rows)
    for _ in range(sweeps * T):
        i, j = rng.randrange(T), rng.randrange(T)
        if i == j:
            continue
        a, b = rows[i], rows[j]
        shared = a & b
        A, B = a - shared, b - shared
        if not A or not B:
            continue
        pool = list(A) + list(B)
        rng.shuffle(pool)
        k = len(A)
        rows[i] = shared | set(pool[:k])
        rows[j] = shared | set(pool[k:])
    return rows


def q1_layer(rows, rng, reps):
    """-> dict of real stats, null stats, and the analytic-z tail fraction."""
    T = len(rows)
    C, _ = cooc(rows)
    n_e = np.diag(C).copy()
    disp, distinct = dispersion(C)

    # analytic independence (hypergeometric) z per pair
    outer = np.outer(n_e, n_e)
    mean = outer / max(T, 1)
    var = outer * np.outer(T - n_e, T - n_e) / (T * T * max(T - 1, 1))
    with np.errstate(invalid="ignore", divide="ignore"):
        Z = (C - mean) / np.sqrt(var)
    np.fill_diagonal(Z, 0.0)
    Z = np.nan_to_num(Z)
    iu = np.triu_indices(N_EXPERTS, 1)
    z = Z[iu]
    tail_real = float((z >= 3.0).mean())

    obs_pairs = float(sum(len(s) * (len(s) - 1) / 2 for s in rows))
    exp_pairs = float((mean[iu]).sum())

    d_null, p_null, tail_null = [], [], []
    for _ in range(reps):
        r = curveball(rows, rng)
        Cn, _ = cooc(r)
        dn, pn = dispersion(Cn)
        d_null.append(dn)
        p_null.append(pn)
        with np.errstate(invalid="ignore", divide="ignore"):
            Zn = (Cn - mean) / np.sqrt(var)
        np.fill_diagonal(Zn, 0.0)
        tail_null.append(float((np.nan_to_num(Zn)[iu] >= 3.0).mean()))
    return dict(T=T, disp=disp, distinct=distinct, obs_pairs=obs_pairs, exp_pairs=exp_pairs,
                tail_real=tail_real, d_null=np.array(d_null), p_null=np.array(p_null),
                tail_null=float(np.mean(tail_null)) if tail_null else 0.0)


def group_containment(per_layer, rng, reps, K):
    """The literal form of "the misses are drawn from a small recurring group".

    For each layer take the K most-missed experts and ask how often a multi-miss call's ENTIRE miss
    set lies inside that group. The null is curveball, which preserves every expert's miss rate
    exactly -- so the same K experts are the top K in the null too, and any excess is grouping and
    not popularity. That separation is the whole point: a hot group is already reachable by pinning,
    a *recurring co-missing* group is what a repack would need.
    """
    real = tot = 0
    nulls = []
    for L, rows in per_layer.items():
        c = collections.Counter()
        for s in rows:
            c.update(s)
        if not c:
            continue
        top = set(e for e, _ in c.most_common(K))
        multi = [s for s in rows if len(s) >= 2]
        tot += len(multi)
        real += sum(1 for s in multi if s <= top)
        acc = []
        for _ in range(reps):
            r = curveball(rows, rng)
            acc.append(sum(1 for s in r if len(s) >= 2 and s <= top))
        nulls.append(np.mean(acc))
    null = float(np.sum(nulls))
    return real, null, tot


def q1_report(title, per_layer, rng, reps, note=""):
    tot = collections.Counter()
    rows_out = []
    agg_d = agg_dn = agg_dn2 = 0.0
    agg_p = agg_pn = 0.0
    agg_obs = agg_exp = 0.0
    tail_r = tail_n = 0.0
    nl = 0
    for L in sorted(per_layer):
        rows = per_layer[L]
        if sum(len(s) for s in rows) == 0:
            continue
        r = q1_layer(rows, rng, reps)
        nl += 1
        agg_d += r["disp"]
        agg_dn += r["d_null"].mean()
        agg_dn2 += r["d_null"].std()
        agg_p += r["distinct"]
        agg_pn += r["p_null"].mean()
        agg_obs += r["obs_pairs"]
        agg_exp += r["exp_pairs"]
        tail_r += r["tail_real"]
        tail_n += r["tail_null"]
        rows_out.append((L, r))
    print(f"  {title}")
    if note:
        print(f"  {note}")
    print(f"  pairs that co-occur, observed / independence-expected : {agg_obs / agg_exp:6.3f}  "
          f"({agg_obs:,.0f} vs {agg_exp:,.0f})")
    print(f"    ^ this ratio is NOT evidence: sum of co-occurring pairs is fixed by the per-token")
    print(f"      set sizes, so it only reports how far set-size variance pushes it off 1.000.")
    zscore = (agg_d - agg_dn) / agg_dn2 if agg_dn2 else float("nan")
    print(f"  dispersion  sum C_ef^2   real {agg_d:14,.0f}   curveball null "
          f"{agg_dn:14,.0f} +- {agg_dn2:,.0f}   excess {100 * (agg_d / agg_dn - 1):+6.2f} %  "
          f"z {zscore:+.1f}")
    print(f"  distinct pairs ever seen  real {agg_p:14,.0f}   curveball null {agg_pn:14,.0f}"
          f"                            {100 * (agg_p / agg_pn - 1):+6.2f} %")
    print(f"  pairs at analytic z>=3    real {100 * tail_r / nl:6.3f} %  curveball null "
          f"{100 * tail_n / nl:6.3f} %   (the null's own false-positive rate at the same threshold)")
    print()
    return rows_out


# --------------------------------------------------------------------------- Q2 machinery

def coalesce(ids, tol):
    """Merge sorted ids into ranges tolerating up to `tol` unwanted slots between neighbours.

    -> (n_reads, n_slots_fetched). Slots fetched counts every record inside every emitted range,
    wanted or not: that is the overfetch, and it is charged at the full 13.77 MB record.
    """
    if not ids:
        return 0, 0
    s = sorted(ids)
    reads, fetched = 1, 1
    lo = prev = s[0]
    for e in s[1:]:
        if e - prev - 1 <= tol:
            fetched += e - prev
        else:
            reads += 1
            fetched += 1
            lo = e
        prev = e
    return reads, fetched


def q2_scan(miss_lists, tols):
    """-> {tol: (reads, slots_fetched)} plus the wanted count."""
    wanted = sum(len(m) for m in miss_lists)
    out = {}
    for t in tols:
        R = F = 0
        for m in miss_lists:
            r, f = coalesce(m, t)
            R += r
            F += f
        out[t] = (R, F)
    return wanted, out


def q2_uniform_null(sizes, tols, rng, reps=5):
    """Same per-call miss counts, ids uniform without replacement from 0..383."""
    acc = {t: [np.zeros(reps), np.zeros(reps)] for t in tols}
    for r in range(reps):
        lists = [rng.sample(range(N_EXPERTS), k) if k else [] for k in sizes]
        _, out = q2_scan(lists, tols)
        for t in tols:
            acc[t][0][r] = out[t][0]
            acc[t][1][r] = out[t][1]
    return {t: (acc[t][0].mean(), acc[t][1].mean()) for t in tols}


def gap_hist(miss_lists):
    """Distribution of gaps between consecutive sorted missed ids, real."""
    g = []
    for m in miss_lists:
        s = sorted(m)
        g += [s[i + 1] - s[i] for i in range(len(s) - 1)]
    return np.array(g)


# --------------------------------------------------------------------------- Q2b machinery

def seriate(weight, deg):
    """Greedy nearest-neighbour seriation of a co-miss graph -> list of expert ids, best first.

    Start at the expert with the largest co-miss degree, then repeatedly append the unplaced expert
    with the largest co-miss weight to the one just placed (ties by total degree). Cheap, and good
    enough to answer "is there ANY ordering that makes these reads coalescable" -- if greedy finds
    nothing, a better seriation is not going to find a factor.
    """
    order = []
    placed = np.zeros(N_EXPERTS, dtype=bool)
    cur = int(np.argmax(deg))
    order.append(cur)
    placed[cur] = True
    for _ in range(N_EXPERTS - 1):
        w = weight[cur].copy()
        w[placed] = -1.0
        best = w + 1e-6 * np.where(placed, -1e9, deg)
        cur = int(np.argmax(best))
        order.append(cur)
        placed[cur] = True
    return order


def q2b(calls, tols, rng, reps=5):
    """Fit per-layer expert permutations on the first half, score coalescing on the second.

    Three orderings, because "the repack does not help" has to survive more than one repack:
      identity  -- today's expert id order, the baseline
      seriation -- greedy nearest-neighbour on the co-miss graph (pairs that miss together become
                   neighbours on disk)
      hot-first -- simply sort by miss frequency, so the experts that miss often are packed into one
                   dense region. Cruder than seriation and much less noisy at these counts: a co-miss
                   count here is mostly 0 or 1, while a miss count is a solid number.
      random    -- the control for "reordering at all".
    """
    per_layer = collections.defaultdict(list)
    for L, _u, m in calls:
        per_layer[L].append(sorted(m))
    half = {L: len(v) // 2 for L, v in per_layer.items()}

    arms = {k: collections.defaultdict(list) for k in ("identity", "seriation", "hot-first")}
    randp = [collections.defaultdict(list) for _ in range(reps)]

    for L, v in per_layer.items():
        train, test = v[:half[L]], v[half[L]:]
        W = np.zeros((N_EXPERTS, N_EXPERTS), dtype=np.float32)
        freq = np.zeros(N_EXPERTS, dtype=np.float32)
        for m in train:
            for a in m:
                freq[a] += 1.0
                for b in m:
                    if a != b:
                        W[a, b] += 1.0
        deg = W.sum(1)

        def posmap(order):
            p = np.zeros(N_EXPERTS, dtype=np.int32)
            for i, e in enumerate(order):
                p[e] = i
            return p

        pos_s = posmap(seriate(W, deg))
        pos_f = posmap(list(np.argsort(-freq)))
        rperms = []
        for _ in range(reps):
            q = list(range(N_EXPERTS))
            rng.shuffle(q)
            rperms.append(posmap(q))
        for m in test:
            arms["identity"][L].append(m)
            arms["seriation"][L].append([int(pos_s[e]) for e in m])
            arms["hot-first"][L].append([int(pos_f[e]) for e in m])
            for r in range(reps):
                randp[r][L].append([int(rperms[r][e]) for e in m])

    flat = lambda d: [m for L in sorted(d) for m in d[L]]
    wanted, out = None, {}
    for k, d in arms.items():
        w, o = q2_scan(flat(d), tols)
        wanted = w
        out[k] = o
    o_r = [q2_scan(flat(randp[r]), tols)[1] for r in range(reps)]
    out["random"] = {t: (np.mean([o[t][0] for o in o_r]), np.mean([o[t][1] for o in o_r]))
                     for t in tols}
    return wanted, out


def frontier(wanted, out, tols, levels):
    """reads at a MATCHED overfetch, linearly interpolated across the tolerance ladder.

    Comparing two orderings at the same `t` is not a fair comparison -- a denser packing buys reads
    by spending more bytes at the same t. The only honest axis is reads at equal bytes.
    """
    pts = sorted((100.0 * (out[t][1] / wanted - 1.0), float(out[t][0])) for t in tols)
    res = {}
    for lv in levels:
        if lv <= pts[0][0]:
            res[lv] = pts[0][1]
            continue
        if lv >= pts[-1][0]:
            res[lv] = pts[-1][1]
            continue
        for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
            if x0 <= lv <= x1:
                res[lv] = y0 + (y1 - y0) * (lv - x0) / (x1 - x0) if x1 > x0 else y0
                break
    return res


# --------------------------------------------------------------------------- Q4

def q4(per_layer, ds, rng, reps=200):
    real, null = {}, {}
    for d in ds:
        num = cnt = 0.0
        for rows in per_layer.values():
            for t in range(len(rows) - d):
                a, b = rows[t], rows[t + d]
                if not a and not b:
                    continue
                num += len(a & b) / max(len(a | b), 1)
                cnt += 1
        real[d] = num / cnt if cnt else 0.0
    num = cnt = 0.0
    for rows in per_layer.values():
        T = len(rows)
        for _ in range(reps):
            i, j = rng.randrange(T), rng.randrange(T)
            if i == j:
                continue
            a, b = rows[i], rows[j]
            if not a and not b:
                continue
            num += len(a & b) / max(len(a | b), 1)
            cnt += 1
    null = num / cnt if cnt else 0.0
    return real, null


# --------------------------------------------------------------------------- main

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("log")
    ap.add_argument("--reps", type=int, default=10, help="curveball randomisations per layer")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    rng = random.Random(a.seed)
    np.random.seed(a.seed)

    calls, has_tok, bad, fields = load(a.log)
    acts = sum(len(c[1]) for c in calls)
    miss = sum(len(c[2]) for c in calls)
    tokens = len(calls) / N_LAYERS
    msizes = [len(c[2]) for c in calls]
    zero = sum(1 for m in msizes if m == 0)
    one = sum(1 for m in msizes if m == 1)

    print(f"\n  {a.log}")
    print(f"  {len(calls):,} decode resolve calls ({bad} out-of-cycle), {acts:,} activations, "
          f"{miss:,} misses, {tokens:.1f} decode steps")
    print(f"  {acts / len(calls):.2f} active experts per call, {miss / len(calls):.2f} misses per "
          f"call, miss rate {100 * miss / acts:.2f} %")
    print(f"  misses per call: {100 * zero / len(calls):.1f} % are 0, {100 * one / len(calls):.1f} % "
          f"are exactly 1, {100 * (len(calls) - zero - one) / len(calls):.1f} % are >= 2 "
          f"(only those can be coalesced at all)")
    print(f"  one miss = {EXPERT_BYTES / 2**20:.2f} MiB of NVMe\n")

    print("=" * 100)
    print("  Q1  CO-ACTIVATION AND MISS CLUSTERING")
    print("=" * 100)
    act_L = by_layer(calls, 1)
    miss_L = by_layer(calls, 2)
    q1_report("ACTIVE sets, per layer, over decode steps", act_L, rng, a.reps)
    q1_report("MISSED sets, per layer, over decode steps", miss_L, rng, a.reps,
              note="the one that matters: a recurring small group of co-missing experts is "
                   "coalescable.")

    # how concentrated are the misses at all -- the operational version of "small recurring group"
    print("  miss concentration per layer (could a few experts BE the miss set?)")
    covs = []
    for L in sorted(miss_L):
        c = collections.Counter()
        for s in miss_L[L]:
            c.update(s)
        tot = sum(c.values())
        if not tot:
            continue
        mc = [v for _, v in c.most_common()]
        covs.append((L, len(mc), sum(mc[:16]) / tot, sum(mc[:32]) / tot, sum(mc[:64]) / tot))
    d16 = np.mean([c[2] for c in covs])
    d32 = np.mean([c[3] for c in covs])
    d64 = np.mean([c[4] for c in covs])
    nd = np.mean([c[1] for c in covs])
    print(f"    distinct experts ever missed per layer: {nd:.0f} of 384")
    print(f"    share of a layer's misses from its top 16 / 32 / 64 most-missed experts: "
          f"{100 * d16:.1f} % / {100 * d32:.1f} % / {100 * d64:.1f} %")
    print(f"    uniform-over-the-{nd:.0f}-touched null would give "
          f"{100 * 16 / nd:.1f} % / {100 * 32 / nd:.1f} % / {100 * 64 / nd:.1f} %")
    for K in (32, 64):
        r, nl, tot = group_containment(miss_L, rng, max(3, a.reps // 2), K)
        print(f"    multi-miss calls whose WHOLE miss set is inside the layer's top-{K}: "
              f"{100 * r / tot:5.2f} % real vs {100 * nl / tot:5.2f} % curveball "
              f"(same miss rates, no grouping) -> x{r / nl:.2f}")
    print()

    print("=" * 100)
    print("  Q2  CONTIGUITY OF MISSES IN SLOT ORDER")
    print("=" * 100)
    lists = [sorted(c[2]) for c in calls]
    sizes = [len(m) for m in lists]
    g = gap_hist(lists)
    gn = []
    for _ in range(5):
        gn.append(gap_hist([rng.sample(range(N_EXPERTS), k) if k else [] for k in sizes]))
    gn = np.concatenate(gn)
    print(f"  gaps between consecutive sorted missed ids within one call ({len(g):,} gaps)")
    print(f"    {'':12s} {'real':>10s} {'uniform null':>14s}")
    for q, lab in ((0.10, "p10"), (0.25, "p25"), (0.50, "p50"), (0.75, "p75"), (0.90, "p90")):
        print(f"    {lab:12s} {np.quantile(g, q):10.1f} {np.quantile(gn, q):14.1f}")
    print(f"    {'mean':12s} {g.mean():10.1f} {gn.mean():14.1f}")
    for t in (1, 2, 4, 8):
        print(f"    gap <= {t:<2d}     {100 * (g <= t).mean():9.2f} % {100 * (gn <= t).mean():13.2f} %")
    print()

    tols = [0, 1, 2, 4, 8, 16, 32]
    wanted, out = q2_scan(lists, tols)
    nullout = q2_uniform_null(sizes, tols, rng, reps=5)
    base = out[0][0]
    ceiling = sum(1 for m in lists if m)
    print(f"  coalescing runs at gap tolerance t (t = unwanted slots tolerated between two wanted).")
    print(f"  {'t':>3s} {'reads':>9s} {'reads/call':>11s} {'vs t=0':>8s} {'slots':>9s} "
          f"{'overfetch':>10s} {'MiB/step':>9s} {'|':>2s} {'null reads':>11s} {'null over':>10s}")
    for t in tols:
        R, F = out[t]
        nR, nF = nullout[t]
        print(f"  {t:3d} {R:9,d} {R / len(calls):11.3f} {100 * (R / base - 1):7.1f} % {F:9,d} "
              f"{100 * (F / wanted - 1):9.1f} % {F * EXPERT_BYTES / tokens / 2**20:9.1f} {'|':>2s} "
              f"{nR:11,.0f} {100 * (nF / wanted - 1):9.1f} %")
    print(f"  wanted records {wanted:,}; one read per call that misses anything would be "
          f"{ceiling:,} reads ({100 * (ceiling / base - 1):.1f} % vs t=0) -- the ceiling of ANY "
          f"within-call coalescing, at whatever overfetch it took.")

    # price it. coalescing is a win only if the bigger read buys more bandwidth than the overfetch
    # throws away: time = bytes / rate, so we need rate_ratio > byte_ratio.
    print(f"\n  PRICED. Coalescing pays only if the larger read raises achieved bandwidth by more")
    print(f"  than it adds bytes: need rate x{out[8][1] / wanted:.3f} at t=8, "
          f"x{out[16][1] / wanted:.3f} at t=16, x{out[32][1] / wanted:.3f} at t=32.")
    print(f"  Measured headroom (notes/gb10-arena-io-measured.md sec.3): ONE whole-expert O_DIRECT")
    print(f"  pread (18.8 MB, already larger than a CB3 record) gives 5.58 GB/s and the device")
    print(f"  ceiling at two concurrent reads is 6.82 GB/s -- so the")
    print(f"  largest bandwidth factor any read-size change can buy is x1.22, and the engine already")
    print(f"  runs ~6 reads in flight. Every tolerance above is on the wrong side of that.\n")

    print("=" * 100)
    print("  Q2b REPACK: does a LEARNED per-layer expert order make the misses contiguous?")
    print("=" * 100)
    print("  Four per-layer orderings, all fitted on the FIRST half of each layer's steps and scored")
    print("  on the held-out second: identity (today), greedy seriation of the co-miss graph,")
    print("  hot-first (sort by miss count), and random = the control for 'reordering at all'.")
    w2, arms = q2b(calls, tols, rng, reps=5)
    names = ["identity", "seriation", "hot-first", "random"]
    print(f"  {'t':>3s} " + " ".join(f"{n + ' reads':>17s}" for n in names))
    for t in tols:
        cells = " ".join(f"{arms[n][t][0]:11,.0f} {100 * (arms[n][t][1] / w2 - 1):4.0f} %"
                         for n in names)
        print(f"  {t:3d} " + cells)
    levels = [5.0, 20.0, 50.0, 100.0]
    print(f"\n  reads at MATCHED overfetch (interpolated) -- the only fair comparison between orders")
    print(f"  {'overfetch':>10s} " + " ".join(f"{n:>13s}" for n in names))
    fr = {n: frontier(w2, arms[n], tols, levels) for n in names}
    for lv in levels:
        print(f"  {lv:9.0f} % " + " ".join(f"{fr[n][lv]:13,.0f}" for n in names))
    print(f"  held-out wanted records {w2:,}, identity t=0 reads {arms['identity'][0][0]:,}\n")

    print("=" * 100)
    print("  Q3  TOKEN-IDENTITY CONDITIONING")
    print("=" * 100)
    print(f"  fields present in the log: {sorted(fields)}")
    if not has_tok:
        print("  NO TOKEN ID in this trace. Not measurable here, and there is no honest proxy: the")
        print("  call index gives position, not identity, and the active set cannot stand in for the")
        print("  token without assuming the very thing the question asks. SKIPPED -- it needs a")
        print("  re-trace with the token id logged per resolve call.\n")

    print("=" * 100)
    print("  Q4  SLOW STATE / DRIFT")
    print("=" * 100)
    ds = [1, 2, 4, 8, 16, 32, 64, 128]
    real, null = q4(act_L, ds, rng)
    print(f"  mean Jaccard of a layer's ACTIVE set at step t vs t+d")
    print(f"  {'d':>5s} {'Jaccard':>9s} {'excess over random-pair null':>30s}")
    for d in ds:
        print(f"  {d:5d} {real[d]:9.4f} {real[d] / null:29.2f}x")
    print(f"  random-pair null (two steps of the same layer, drawn at random): {null:.4f}")
    print()

    print("=" * 100)
    print("  SCOPE: one decode trace, {:,} steps, the shipped arena's own hit/miss decisions.".format(
        int(tokens)))
    print("  Nothing here is measured at another cache size, another corpus, or in prefill.")
    print("=" * 100)
    return 0


if __name__ == "__main__":
    sys.exit(main())
