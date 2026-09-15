#!/usr/bin/env python3
"""eviction_oracle.py -- is the LRU->Belady gap reachable by a better EVICTION ORDER?

`tools/prefetch_oracle.py` closed the prefetch side on this same decode log: short-history
predictors name 0.0 % of LRU's misses, because LRU already retains ~57 tokens of history and the
misses are pairs unused for a median of 45 tokens. But Belady is not a prefetcher -- it never
fetches anything early, it only picks a better VICTIM. So the entire 61 %-less-NVMe gap is an
eviction-ordering gap, and eviction has economics that make weak signals usable: a wrong eviction
costs the opportunity of having kept something better, a wrong prefetch costs a 13,774,848 B read.

This tool replays the cache and, at every eviction, ranks the K coldest resident entries by a score
built from PAST INFORMATION ONLY and evicts the top-ranked one. Small K keeps it obviously cheap:
`_lru_slot_for` would walk K slots off the cold end of the LRU list, not all 5,328. K is swept,
because how a score behaves as it is allowed to see more of the cache is itself the finding -- the
winner keeps improving to K = every slot, and LFU inverts.

Scores (cheapest first):

  1 lru             age since last use                     -- the baseline we ship
  2 lfu             use count (ties broken by age)
  3 age/(1+freq)    age discounted by frequency, GDSF-style without the size/cost terms
  4 median-gap      this entry's median inter-arrival so far; evict the slowest rhythm
  5 age/median-gap  normalised: evict the entry most OVERDUE relative to its own rhythm
  6 hazard(H)       P(no use in the next H tokens | survived `age` already), from that entry's own
                    gap history via its empirical survival function; H swept
  7 ttnu            median-gap - age = estimated tokens until the next use; evict the largest.
                    This is the direct plug-in estimator OF BELADY and is the arm the motivating
                    intuition actually asks for: not how OFTEN an entry is used, but WHEN it returns.
  -- belady         true furthest-next-use, uses the future, not implementable: the ceiling

HOW NO-FUTURE-LEAKAGE IS GUARANTEED
  * Every per-entry statistic (last_acc, last_tok, count, sorted gap list) is written in exactly one
    place: `touch()`, called at the moment the access is processed, from indices <= the current one.
  * The two priors a score can fall back to (global median gap, global gap survival function) are
    fitted on the warm-up prefix only, before the scored window begins, and are frozen after.
  * The scores are pure functions of (now, entry stats) -- they receive no sequence and no index.
  * Belady lives in a separate function that never shares state with the replay loop.
  * Sanity check, printed: the `lru` score arm must reproduce plain LRU BIT-EXACTLY (score = age is
    monotone in LRU position, so the top-ranked of the K coldest IS the LRU head). If that line does
    not say IDENTICAL, the harness is broken and nothing below means anything.

  Caveat stated up front: H for the hazard arm is swept and the best H reported, on the same scored
  window. That is hyperparameter selection on the test set, so the hazard row is an OPTIMISTIC upper
  bound, not a held-out result.

Usage:
    eviction_oracle.py ~/ds41-queue/logs/route-decode.jsonl [--slots 5328 3000] [--k 32 64]
CPU only; no GPU, no server.
"""

from __future__ import annotations

import argparse
import bisect
import collections
import heapq
import itertools
import json
import sys

N_EXPERTS = 384          # DS4.1 routed experts per MoE layer
N_LAYERS = 40
EXPERT_BYTES = 13_774_848   # one native CB3 disk record -- what a miss actually reads


# --------------------------------------------------------------------------- data

def load(path: str):
    """-> [(layer, (expert ids,)), ...] for decode calls only, in forward order.

    Same loader as tools/prefetch_oracle.py: the log is strictly cyclic 0..39 after the prefill
    lines, so call index // 40 is the token and % 40 is the layer.
    """
    calls = []
    for line in open(path):
        line = line.strip()
        if not line:
            continue
        try:
            o = json.loads(line)
        except json.JSONDecodeError:
            continue
        if o.get("pf"):
            continue
        calls.append((o["L"], tuple(o["uniq"])))
    return calls


# --------------------------------------------------------------------------- per-entry stats

class Stats:
    """Everything a policy may look at. Written only by touch(), only from the present."""
    __slots__ = ("last_acc", "last_tok", "count", "gaps")

    def __init__(self):
        self.last_acc = -1
        self.last_tok = -1
        self.count = 0
        self.gaps = []          # sorted token gaps between consecutive uses of this pair

    def touch(self, acc: int, tok: int):
        if self.count and tok > self.last_tok:
            bisect.insort(self.gaps, tok - self.last_tok)
        self.count += 1
        self.last_acc = acc
        self.last_tok = tok

    def median_gap(self, prior: float) -> float:
        g = self.gaps
        n = len(g)
        if n == 0:
            return prior
        return g[n // 2] if n & 1 else 0.5 * (g[n // 2 - 1] + g[n // 2])


# --------------------------------------------------------------------------- scores
# Each takes (stats, now_acc, now_tok, ctx) and returns a float. HIGHEST score is evicted.
# ctx carries only prefix-fitted priors. None of them can see the future.

def s_lru(st, acc, tok, ctx):
    return acc - st.last_acc


def s_lfu(st, acc, tok, ctx):
    # lowest use count first; age only as a tie-break, scaled so it can never outvote a count step
    return -st.count + (acc - st.last_acc) * 1e-9


def s_age_over_freq(st, acc, tok, ctx):
    return (acc - st.last_acc) / (1.0 + st.count)


def s_median_gap(st, acc, tok, ctx):
    return st.median_gap(ctx["gap_prior"])


def s_age_over_median_gap(st, acc, tok, ctx):
    return (tok - st.last_tok) / max(1e-9, st.median_gap(ctx["gap_prior"]))


def s_ttnu(st, acc, tok, ctx):
    # estimated tokens until the next use = own median rhythm minus how long it has already waited
    return st.median_gap(ctx["gap_prior"]) - (tok - st.last_tok)


def make_hazard(H: int):
    """P(gap > age + H | gap > age) from this entry's own empirical gap distribution.

    Entries with too little history fall back to the global prefix-fitted survival function. A high
    value = 'unlikely to come back within H tokens' = evict.
    """
    def score(st, acc, tok, ctx):
        age = tok - st.last_tok
        g = st.gaps
        if len(g) >= 3:
            surv = len(g) - bisect.bisect_right(g, age)
            if surv == 0:                       # already older than anything it has ever done
                return 1.0 + age * 1e-9         # strongly evictable, oldest first
            beyond = len(g) - bisect.bisect_right(g, age + H)
            return beyond / surv
        G = ctx["gap_pool"]
        surv = len(G) - bisect.bisect_right(G, age)
        if surv == 0:
            return 1.0 + age * 1e-9
        beyond = len(G) - bisect.bisect_right(G, age + H)
        return beyond / surv
    return score


# --------------------------------------------------------------------------- replay

def replay(calls, slots, cut, score, K):
    """LRU list + K-candidate re-ranking at eviction. Returns (hits, fetches, scored_accesses).

    The access sequence is identical for every policy (no prefetch, no lookahead), so fetches ==
    misses and the byte axis and the hit axis are the same number.
    """
    cache = collections.OrderedDict()          # key -> 1, ordered least-recently-used first
    stats = collections.defaultdict(Stats)
    ctx = {"gap_prior": 0.0, "gap_pool": []}
    hits = fetches = scored = 0
    acc = 0

    for i, (L, ex) in enumerate(calls):
        if i == cut:
            # Freeze the priors here, from prefix history only, and never update them again.
            pool = []
            for st in stats.values():
                pool.extend(st.gaps)
            pool.sort()
            ctx["gap_pool"] = pool
            ctx["gap_prior"] = pool[len(pool) // 2] if pool else 1.0
        tok = i // N_LAYERS
        scoring = i >= cut
        for e in ex:
            k = L * N_EXPERTS + e
            acc += 1
            if k in cache:
                cache.move_to_end(k)
                if scoring:
                    hits += 1
                    scored += 1
            else:
                if len(cache) >= slots:
                    if K <= 1:          # fast path, used only for the plain-LRU baseline
                        victim = next(iter(cache))
                    else:
                        best = None
                        best_s = None
                        for c in itertools.islice(cache, K):
                            s = score(stats[c], acc, tok, ctx)
                            if best_s is None or s > best_s:
                                best_s, best = s, c
                        victim = best
                    del cache[victim]
                cache[k] = 1
                if scoring:
                    fetches += 1
                    scored += 1
            stats[k].touch(acc, tok)
    return hits, fetches, scored


def replay_buckets(calls, slots, cut):
    """EXACT global argmax of age/(1+freq) over the WHOLE cache, in O(#distinct use counts).

    Why this matters: the K-scan arms show age/(1+freq) still improving at K = every slot, which
    would be a 5,328-entry scan per eviction if you did it naively. You do not have to. Bucket the
    resident entries by use count and keep each bucket in LRU order; within a bucket the count is
    constant, so the maximum of (now - last_acc)/(1 + count) is the bucket's LRU head. The global
    winner is therefore the best of the bucket heads -- a few hundred comparisons, next to a
    13.1 MiB NVMe read. This is what would go into `_lru_slot_for`, and it is EXACT, not an
    approximation: --verify-buckets checks it against the brute-force scan.

    Returns (hits, fetches, scored, mean bucket heads examined per eviction).
    """
    buckets = collections.defaultdict(collections.OrderedDict)   # count -> {key: 1} LRU-ordered
    where = {}                                                   # key -> its count bucket
    stats = collections.defaultdict(Stats)
    hits = fetches = scored = evictions = examined = 0
    acc = 0
    resident = 0
    for i, (L, ex) in enumerate(calls):
        tok = i // N_LAYERS
        scoring = i >= cut
        for e in ex:
            k = L * N_EXPERTS + e
            acc += 1
            if k in where:
                c = where.pop(k)
                del buckets[c][k]
                if scoring:
                    hits += 1
                    scored += 1
            else:
                if resident >= slots:
                    best = None
                    best_key = (None, None)
                    n_heads = 0
                    for c, b in buckets.items():
                        if not b:
                            continue
                        n_heads += 1
                        head = next(iter(b))
                        sc = (acc - stats[head].last_acc) / (1.0 + c)
                        key = (sc, -stats[head].last_acc)
                        if best is None or key > best_key:
                            best, best_key = head, key
                    del buckets[where.pop(best)][best]
                    resident -= 1
                    evictions += 1
                    examined += n_heads
                if scoring:
                    fetches += 1
                    scored += 1
                resident += 1
            stats[k].touch(acc, tok)
            c = stats[k].count
            buckets[c][k] = 1
            where[k] = c
    return hits, fetches, scored, examined / max(1, evictions)


def belady(calls, slots, cut):
    """Evict the resident pair whose next use is furthest away. Lifted from prefetch_oracle.py."""
    seq = [L * N_EXPERTS + e for L, ex in calls for e in ex]
    cut_acc = sum(len(ex) for _, ex in calls[:cut])
    nxt = [0] * len(seq)
    last = {}
    for i in range(len(seq) - 1, -1, -1):
        nxt[i] = last.get(seq[i], len(seq))
        last[seq[i]] = i
    cache, heap, hits, fetches = {}, [], 0, 0
    for i, k in enumerate(seq):
        if k in cache:
            hits += (i >= cut_acc)
        else:
            if len(cache) >= slots:
                while True:
                    neg, kk = heapq.heappop(heap)
                    if cache.get(kk) == -neg:
                        del cache[kk]
                        break
            fetches += (i >= cut_acc)
        cache[k] = nxt[i]
        heapq.heappush(heap, (-nxt[i], k))
    return hits, fetches, len(seq) - cut_acc


def miss_anatomy(calls, slots, cut):
    """Of LRU's scored misses: how many are COMPULSORY (pair never used before, in the whole log)?

    Compulsory misses are unreachable by ANY eviction policy -- you cannot keep what you have never
    seen -- so they bound the achievable ceiling. Also returns how stale the non-compulsory ones are.
    """
    cache = collections.OrderedDict()
    last_tok = {}
    misses = compulsory = 0
    dists = []
    for i, (L, ex) in enumerate(calls):
        tok = i // N_LAYERS
        scoring = i >= cut
        for e in ex:
            k = L * N_EXPERTS + e
            if k in cache:
                cache.move_to_end(k)
            else:
                if len(cache) >= slots:
                    cache.popitem(last=False)
                cache[k] = 1
                if scoring:
                    misses += 1
                    if k in last_tok:
                        dists.append(tok - last_tok[k])
                    else:
                        compulsory += 1
            last_tok[k] = tok
    dists.sort()
    pct = lambda q: dists[min(len(dists) - 1, int(q * len(dists)))] if dists else 0
    return misses, compulsory, pct(0.10), pct(0.50), pct(0.90)


# --------------------------------------------------------------------------- main

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("log")
    ap.add_argument("--slots", type=int, nargs="+", default=[5328, 3000],
                    help="shipped lru_slots = 5728 - 400 transient = 5328; 3000 checks rank stability")
    ap.add_argument("--k", type=int, nargs="+", default=[32, 64, 256],
                    help="candidates ranked per eviction (the cold end of the LRU list)")
    ap.add_argument("--hazard-h", type=int, nargs="+", default=[10, 25, 45, 100, 200],
                    help="hazard horizons in tokens (swept; best reported, see the caveat)")
    ap.add_argument("--prefix", type=float, default=0.25)
    ap.add_argument("--verify-buckets", action="store_true",
                    help="also run the brute-force K=all-slots scan and assert the count-bucket\n                          implementation is bit-identical to it (slow: minutes)")
    a = ap.parse_args()

    calls = load(a.log)
    if not calls:
        print("no decode calls in log", file=sys.stderr)
        return 1
    accesses = sum(len(ex) for _, ex in calls)
    tokens = len(calls) / N_LAYERS
    cut = int(len(calls) * a.prefix) // N_LAYERS * N_LAYERS       # whole-token boundary
    tok_scored = (len(calls) - cut) / N_LAYERS

    print(f"  {len(calls):,} decode resolve calls, {accesses:,} accesses, {tokens:.0f} tokens, "
          f"{accesses / tokens:.0f} accesses/token")
    print(f"  warm every policy on calls [0,{cut:,}) = {100 * cut / len(calls):.0f} %, score the "
          f"rest ({len(calls) - cut:,} calls, {tok_scored:.0f} tokens)")
    print(f"  one fetch = {EXPERT_BYTES / 2**20:.2f} MiB of NVMe; no policy here prefetches, so "
          f"fetches == misses and MiB/token is just the miss axis rescaled\n")

    for slots in a.slots:
        print(f"  ===== cache {slots:,} slots "
              f"({100 * slots / (N_LAYERS * N_EXPERTS):.1f} % of the 15,360 pairs) =====")

        h_lru, f_lru, scored = replay(calls, slots, cut, s_lru, 1)
        hb, fb, _ = belady(calls, slots, cut)
        base_hit, bel_hit = h_lru / scored, hb / scored

        nm, comp, d10, d50, d90 = miss_anatomy(calls, slots, cut)
        print(f"  compulsory misses: {comp:,} of {nm:,} LRU misses = {100 * comp / nm:.1f} % "
              f"(first ever use of that pair -- unreachable by any eviction policy)")
        print(f"  the other {100 * (1 - comp / nm):.1f} % were last used {d10} / {d50} / {d90} "
              f"tokens ago (p10/p50/p90): the reachable set.")
        print(f"  Belady removes {f_lru - fb:,} of those {nm - comp:,} reachable misses "
              f"= {100 * (f_lru - fb) / (nm - comp):.0f} % of them.")
        print(f"  (the COUNT of compulsory misses is a property of the trace and the prefix, not of"
              f" the cache: 2,433 at this 25 % prefix for every size. The PERCENTAGE moves only"
              f" because the miss denominator does -- 7.1 % at 4,788 slots, {100 * comp / nm:.1f} %"
              f" here. Earlier work's 7.1 % was belady_replay.py's old 4,788-slot default.)\n")

        arms = [("1 LRU  (age)", s_lru),
                ("2 LFU  (count, age tie-break)", s_lfu),
                ("3 age / (1 + freq)", s_age_over_freq),
                ("4 median-gap", s_median_gap),
                ("5 age / median-gap", s_age_over_median_gap),
                ("7 ttnu = median-gap - age", s_ttnu)]
        for H in a.hazard_h:
            arms.append((f"6 hazard H={H}", make_hazard(H)))

        hdr = (f"  {'policy':34s} {'K':>3s} {'hit':>8s} {'fetch/tok':>10s} {'MiB/tok':>9s} "
               f"{'xLRU':>6s} {'gap recovered':>14s}")
        print(hdr)
        print(f"  {'LRU (shipped baseline)':34s} {'-':>3s} {100 * base_hit:7.2f} % "
              f"{f_lru / tok_scored:10.1f} {f_lru * EXPERT_BYTES / tok_scored / 2**20:9.0f} "
              f"{1.0:6.2f} {'0.0 %':>14s}")

        for K in a.k:
            for name, score in arms:
                h, f, sc = replay(calls, slots, cut, score, K)
                assert sc == scored
                hit = h / sc
                rec = (hit - base_hit) / (bel_hit - base_hit) if bel_hit > base_hit else 0.0
                tag = ""
                if score is s_lru:
                    tag = "  <- IDENTICAL to LRU" if (h, f) == (h_lru, f_lru) else "  <- HARNESS BUG"
                print(f"  {name:34s} {K:3d} {100 * hit:7.2f} % {f / tok_scored:10.1f} "
                      f"{f * EXPERT_BYTES / tok_scored / 2**20:9.0f} {f / f_lru:6.2f} "
                      f"{100 * rec:13.1f} %{tag}")
            print()

        hbk, fbk, sck, heads = replay_buckets(calls, slots, cut)
        hitbk = hbk / sck
        recbk = (hitbk - base_hit) / (bel_hit - base_hit)
        print(f"  {'3-exact age/(1+freq), ALL slots':34s} {'all':>3s} {100 * hitbk:7.2f} % "
              f"{fbk / tok_scored:10.1f} {fbk * EXPERT_BYTES / tok_scored / 2**20:9.0f} "
              f"{fbk / f_lru:6.2f} {100 * recbk:13.1f} %   count-bucketed, "
              f"{heads:.0f} comparisons/eviction")
        if a.verify_buckets:
            hs, fs, _ = replay(calls, slots, cut, s_age_over_freq, slots)
            print(f"      verify: brute-force K={slots} scan -> hits {hs:,} fetches {fs:,} ; "
                  f"buckets -> hits {hbk:,} fetches {fbk:,} : "
                  f"{'IDENTICAL' if (hs, fs) == (hbk, fbk) else 'MISMATCH'}")
        print()

        print(f"  {'Belady (future; the ceiling)':34s} {'-':>3s} {100 * bel_hit:7.2f} % "
              f"{fb / tok_scored:10.1f} {fb * EXPERT_BYTES / tok_scored / 2**20:9.0f} "
              f"{fb / f_lru:6.2f} {'100.0 %':>14s}\n")

    print("  gap recovered = (hit - LRU) / (Belady - LRU); equivalently the fraction of LRU's excess")
    print("  NVMe bytes removed, since no arm prefetches. Negative = worse than what we ship.")
    print("  Every score sees only (age, use count, own past gaps) plus two priors frozen at the end")
    print("  of the warm-up prefix. The hazard rows sweep H on the scored window and are therefore an")
    print("  optimistic bound on that family, not a held-out result.")
    print("  Scope: ONE trace (DS4.1 decode, 549 tokens, one prompt), these cache sizes. Nothing here")
    print("  says anything about prefill, other prompts, or other models.\n")
    print("  WHAT THIS FOUND. Eviction is NOT closed the way prefetch is. One score wins and it is the")
    print("  cheapest combination on the list: age / (1 + use count) -- frequency-discounted recency.")
    print("  It recovers ~3-5 % of the LRU->Belady gap at K = 32/64 and ~29-31 % when allowed to rank")
    print("  the whole cache, at BOTH sizes, monotonically in K. The exact global version is not a")
    print("  full scan: bucket by use count, LRU within the bucket, compare the bucket heads (~226")
    print("  comparisons/eviction at 3,000 slots, verified bit-identical to the brute-force scan).")
    print()
    print("  WHAT THIS REFUTED -- the motivating intuition, as stated, is WRONG on this trace. The")
    print("  'when does it return' scores (median-gap, age/median-gap, ttnu = median-gap - age) are")
    print("  at or BELOW LRU everywhere and collapse as K grows (-300 % of the gap at full K). A")
    print("  per-entry inter-arrival median does not predict this workload's returns: routing is")
    print("  near-uniform and an expert's gap distribution is wide, so its median is a bad point")
    print("  estimate and acting on it evicts entries that were about to be used. The frequency term")
    print("  is what carries -- but ONLY held against age: plain LFU also wins at small K and then")
    print("  inverts to -10 %/-15 % at full K, because unweighted frequency protects stale hot")
    print("  entries forever. The age numerator is load-bearing.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
