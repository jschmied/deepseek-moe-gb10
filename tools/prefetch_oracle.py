#!/usr/bin/env python3
"""prefetch_oracle.py -- how much of the LRU->Belady gap does REAL lookahead reach?

`tools/belady_replay.py` on this same decode log says: plain LRU 90.5 % hit, Belady 96.3 %, i.e.
misses 9.50 % -> 3.70 % = 61 % less NVMe traffic. Decode is ~78 % expert-fetch wait, so that is the
largest single prize in the project -- and Belady needs the entire future, so it is not buildable.

This asks the buildable question instead: **what does a predictor we could actually evaluate in time
get us?** Each predictor here corresponds to a lookahead the engine really has:

  P1  same layer, previous token   -- known ONE FULL TOKEN ahead (~3.6 s of wall clock at today's
                                      decode speed), more lookahead than any drafter offers
  P2  same layer, union of last k  -- buys coverage with bytes; k in {1,2,4,8}
  P3  the previous resolve call    -- S(L-1,t), only ~90 ms of lookahead but no cross-token
                                      assumption; for L=0 that is layer 39 of t-1
  P4  static per-layer frequency   -- fitted on the held-out prefix; the baseline any predictor must
                                      beat, run at each P2 arm's byte volume so it is same-budget

Why "bytes/token", not hit rate. A prefetch that misses costs exactly what a demand miss costs:
18,800,640 B read from NVMe in the original packed FP4 layout (notes/native-cb3-expert-cache.md).
A predictor that fetches 3x the bytes to remove 1x the misses is a loss even though its hit rate
looks better. Every arm below therefore reports total fetched bytes normalised to LRU = 1.00, and
the verdict line ranks on that.

Measurement rules, learned the hard way:

* Warm every policy on the SAME 25 % prefix and score only the suffix. An earlier belady_replay.py
  scored LRU from an empty cache while the static policies started resident; that manufactured a
  1.0 pp "win" for the segmented policy which vanished when fixed (external review, 2026-09-15).
  Here the prefix is rounded to a multiple of n_layers so the warm cache ends on a token boundary
  and P1/P2 are not fed a half-token of history.
* A prefetch is modelled as an insert into the same LRU, issued when the prediction actually becomes
  available (`lead` resolve calls before the target) and NOT immediately before the access it
  serves. At lead 40 a prefetched slot must survive a whole token of other layers' traffic -- that
  is the honest cost and it is where naive prefetch simulations cheat.
* It displaces the LRU tail, i.e. the coldest resident (layer, expert) pairs. We count the damage
  directly: `stolen` = accesses that plain LRU hit and this policy missed, computed by replaying the
  identical access sequence through both and diffing per access.
* `wasted` = prefetched entries evicted before anything demanded them: pure NVMe traffic for nothing.
* P1-touch is a control with NO bytes at all: it only moves the predicted set to the MRU end if it
  is already resident. If P1's gain came from recency re-ordering rather than from fetching, this
  arm shows it -- and a byte-free win would be the better thing to build.

Usage:
    prefetch_oracle.py ~/ds41-queue/logs/route-decode.jsonl [--slots 5328] [--prefix 0.25]
CPU only; no GPU, no server.
"""

from __future__ import annotations

import argparse
import collections
import heapq
import json
import sys

N_EXPERTS = 384          # DS4.1 routed experts per MoE layer
N_LAYERS = 40
# What a miss actually reads: the expert in its original packed FP4 checkpoint layout, six tensors,
# 17.93 MiB of payload (notes/native-cb3-expert-cache.md). The CB3 arena slot it lands in is
# 14,454,784 B, but the arena is filled FROM the checkpoint, so NVMe traffic is priced at the FP4
# size. A prefetch costs exactly the same as a demand miss -- that is the whole tension here.
# What a MISS actually costs in NVMe bytes on the shipped server: one native CB3 record, not the
# packed-FP4 checkpoint expert. The arena SLOT is 14,454,784 B (scales expanded) but the disk read
# is the record. Using the FP4 number inflated every absolute MiB/token by 1.36x; the normalised
# xLRU column was unaffected, since every policy is multiplied by the same constant.
# External review, 2026-09-15.
EXPERT_BYTES = 13_774_848


def load(path: str):
    """-> [(layer, (expert ids,)), ...] for decode calls only, in forward order.

    The log opens with 40 pf=1 lines (one prefill call per layer) and then runs strictly cyclic
    0,1,...,39,0,1,... -- verified, 0 out-of-order positions in 21,956 calls -- so call index // 40
    is the token index and call index % 40 is the layer.
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


# --------------------------------------------------------------------------- predictors

def build_predictors(calls, cut, ks):
    """-> {name: (lead_in_calls, [set_of_global_keys_per_call])}.

    A predictor entry at index n is the set proposed FOR call n; `lead` says how many calls earlier
    it is knowable, which is when the simulator issues it.
    """
    n = len(calls)
    per_layer = collections.defaultdict(list)          # layer -> [set of expert ids per token]
    idx_of = {}                                        # (layer, token) -> call index
    for i, (L, ex) in enumerate(calls):
        idx_of[(L, len(per_layer[L]))] = i
        per_layer[L].append(set(ex))

    preds = {}

    # P1/P2: same layer, union of the last k tokens. Knowable at the (L, t-k) call at the latest,
    # but the tightest binding one is (L, t-1) -> exactly one token = 40 calls of lookahead.
    for k in ks:
        sets = [frozenset()] * n
        for L, steps in per_layer.items():
            for t in range(k, len(steps)):
                u = set()
                for j in range(1, k + 1):
                    u |= steps[t - j]
                sets[idx_of[(L, t)]] = frozenset(L * N_EXPERTS + e for e in u)
        preds[f"P2 same-layer union k={k}" if k > 1 else "P1 same-layer prev token"] = (N_LAYERS, sets)

    # P3: the immediately preceding resolve call, carried across the layer boundary by expert index.
    # Only ~90 ms of lookahead (one layer), but it needs no cross-token assumption at all.
    sets = [frozenset()] * n
    for i in range(1, n):
        L = calls[i][0]
        sets[i] = frozenset(L * N_EXPERTS + e for e in calls[i - 1][1])
    preds["P3 prev layer, same token"] = (1, sets)

    # P4: static per-layer frequency prior fitted on the held-out prefix only. Sized to each P2 arm's
    # mean set size so it spends the same bytes -- a bigger static set is not a fairer baseline, it
    # is a different (and much more expensive) policy.
    freq = collections.defaultdict(collections.Counter)
    for L, ex in calls[:cut]:
        freq[L].update(ex)
    for k in ks:
        name = f"P2 same-layer union k={k}" if k > 1 else "P1 same-layer prev token"
        sizes = [len(s) for s in preds[name][1][cut:] if s]
        m = max(1, round(sum(sizes) / len(sizes)))
        top = {L: frozenset(L * N_EXPERTS + e for e, _ in c.most_common(m)) for L, c in freq.items()}
        sets = [top.get(calls[i][0], frozenset()) for i in range(n)]
        preds[f"P4 static top-{m}/layer"] = (1, sets)

    return preds


# --------------------------------------------------------------------------- simulators

def simulate(calls, slots, cut, pred=None, lead=0, touch_only=False):
    """LRU with optional prefetch-into-LRU. Returns a dict of counters + a per-access hit bytearray.

    `touch_only` issues no fetches: it only refreshes recency for already-resident predicted keys.
    That is the zero-byte control for P1.
    """
    cache = collections.OrderedDict()
    pending = set()                       # prefetched, resident, not yet demanded
    hits = demand_fetch = pre_fetch = wasted = pre_useful = 0
    per_access = bytearray()
    n = len(calls)

    for i in range(n):
        L, ex = calls[i]
        scoring = i >= cut
        for e in ex:
            k = L * N_EXPERTS + e
            if k in cache:
                cache.move_to_end(k)
                if k in pending:
                    pending.discard(k)
                    if scoring:
                        pre_useful += 1
                if scoring:
                    hits += 1
                    per_access.append(1)
            else:
                if len(cache) >= slots:
                    ev, _ = cache.popitem(last=False)
                    if ev in pending:
                        pending.discard(ev)
                        if scoring:
                            wasted += 1
                cache[k] = 1
                if scoring:
                    demand_fetch += 1
                    per_access.append(0)

        # Issue the prefetch for the call whose prediction becomes available now. At lead=40 the
        # fetched slot then has to survive 39 other layer calls before it is used.
        tgt = i + lead
        if pred is not None and lead and tgt < n:
            P = pred[tgt]
            for k in P:
                if k in cache:
                    if touch_only:
                        cache.move_to_end(k)
                    continue
                if touch_only:
                    continue
                if len(cache) >= slots:
                    ev, _ = cache.popitem(last=False)
                    if ev in pending:
                        pending.discard(ev)
                        if scoring:
                            wasted += 1
                cache[k] = 1
                pending.add(k)
                if scoring:
                    pre_fetch += 1

    return {
        "hits": hits, "accesses": len(per_access),
        "demand_fetch": demand_fetch, "pre_fetch": pre_fetch,
        "wasted": wasted, "pre_useful": pre_useful, "trace": per_access,
    }


def belady(calls, slots, cut):
    """Evict the resident pair whose next use is furthest away. Max-heap with lazy deletion."""
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


# --------------------------------------------------------------------------- reporting

def overlap_table(calls, ks):
    """Raw set overlap, before any cache is involved.

    Printed FIRST and on purpose: if |S(L,t) & S(L,t-1)| / |S(L,t)| were low, every arm below is
    dead and nothing else needs running. The `chance` column is |P| / 384 -- what a predictor that
    names |P| experts at random would score. An overlap at chance carries no information at all,
    whatever the absolute number looks like.
    """
    per_layer = collections.defaultdict(list)
    for L, ex in calls:
        per_layer[L].append(set(ex))
    rows = []
    for k in ks:
        num = den = vol = steps = 0
        for steps_L in per_layer.values():
            for t in range(k, len(steps_L)):
                u = set()
                for j in range(1, k + 1):
                    u |= steps_L[t - j]
                num += len(steps_L[t] & u)
                den += len(steps_L[t])
                vol += len(u)
                steps += 1
        name = "same layer, prev token" if k == 1 else f"same layer, union last {k}"
        rows.append((name, num / den, vol / steps))
    # P3 measured the same way: does layer L reuse the expert INDICES layer L-1 just used?
    num = den = vol = 0
    for i in range(1, len(calls)):
        p = set(calls[i - 1][1])
        num += len(set(calls[i][1]) & p)
        den += len(calls[i][1])
        vol += len(p)
    rows.append(("prev layer, same token", num / den, vol / (len(calls) - 1)))
    return rows


def miss_anatomy(calls, slots, cut):
    """Why LRU misses: is the wanted expert new, or merely old?

    slots / insertions-per-token is a rough retention horizon: roughly how many tokens of history
    the cache holds for free. It is only a proxy (hits also push entries down the stack), so the
    load-bearing number is the exact one: how many tokens ago a missed pair was last used. Any
    predictor built from the last k tokens can only name experts inside that window -- which is
    exactly why the same-layer arms below fetch nothing at all.
    """
    cache = collections.OrderedDict()
    last_tok = {}
    compulsory = 0
    dists = []
    misses = 0
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
    def pct(q):
        return dists[min(len(dists) - 1, int(q * len(dists)))] if dists else 0
    return misses, compulsory, pct(0.10), pct(0.50), pct(0.90)


def miss_coverage(calls, slots, cut, preds):
    """Of the demand misses PLAIN LRU takes, what fraction does each predictor name?

    This is the number that decides everything. An earlier note argued previous-step prefetch is a
    no-op "by construction" because the previous step's experts are still resident -- i.e. the
    predictor names only what is already there. This measures that claim instead of asserting it.
    """
    cache = collections.OrderedDict()
    named = collections.Counter()
    misses = 0
    for i, (L, ex) in enumerate(calls):
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
                    for name, (_, sets) in preds.items():
                        if k in sets[i]:
                            named[name] += 1
    return misses, named


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("log")
    ap.add_argument("--slots", type=int, nargs="+", default=[5328, 3000],
                    help="shipped lru_slots = n_slots - transient = 5728 - 400 = 5328; 3000 checks rank stability")
    ap.add_argument("--prefix", type=float, default=0.25, help="warm-up / fitting fraction")
    a = ap.parse_args()

    calls = load(a.log)
    if not calls:
        print("no decode calls in log", file=sys.stderr)
        return 1
    accesses = sum(len(ex) for _, ex in calls)
    tokens = len(calls) / N_LAYERS
    # Round the prefix to a whole token so the warm cache ends on a layer-39 boundary and P1/P2 do
    # not start scoring on a half-built history.
    cut = int(len(calls) * a.prefix) // N_LAYERS * N_LAYERS
    ks = [1, 2, 4, 8]

    print(f"  {len(calls):,} decode resolve calls, {accesses:,} accesses, "
          f"{tokens:.1f} tokens, {accesses / tokens:.0f} accesses/token")
    print(f"  warm on calls [0,{cut:,}) = {100 * cut / len(calls):.0f} %, score the rest "
          f"({len(calls) - cut:,} calls, {(len(calls) - cut) / N_LAYERS:.0f} tokens)")
    print(f"  one fetch = {EXPERT_BYTES / 2**20:.2f} MiB of NVMe (one native CB3 disk record)\n")

    print("  RAW SET OVERLAP -- no cache involved, the cheap kill-shot check")
    print(f"  {'predictor':32s} {'|S&P|/|S|':>10s} {'|P| experts':>12s} {'chance':>8s}")
    for name, ov, vol in overlap_table(calls, ks):
        print(f"  {name:32s} {100 * ov:9.1f} % {vol:11.1f} {100 * vol / N_EXPERTS:7.1f} %")
    print()

    preds = build_predictors(calls, cut, ks)

    for slots in a.slots:
        print(f"  ===== cache {slots:,} slots "
              f"({100 * slots / (N_LAYERS * N_EXPERTS):.1f} % of the 15,360 pairs) =====")

        base = simulate(calls, slots, cut)
        scored = base["accesses"]
        h_lru = base["hits"] / scored
        f_lru = base["demand_fetch"]
        hb, fb, _ = belady(calls, slots, cut)
        h_bel = hb / scored
        bytes_lru = f_lru * EXPERT_BYTES
        tok_scored = (len(calls) - cut) / N_LAYERS

        m_lru, m_bel = 1 - h_lru, 1 - h_bel
        print(f"  LRU {100 * h_lru:.2f} % hit, {f_lru / tok_scored:.1f} fetches/token, "
              f"{bytes_lru / tok_scored / 2**20:,.0f} MiB/token")
        print(f"  Belady {100 * h_bel:.2f} % hit, {fb / tok_scored:.1f} fetches/token, "
              f"{fb * EXPERT_BYTES / tok_scored / 2**20:,.0f} MiB/token "
              f"= {100 * (1 - fb / f_lru):.0f} % less NVMe (the unreachable ceiling)\n")

        nm, comp, d10, d50, d90 = miss_anatomy(calls, slots, cut)
        print(f"  LRU retention horizon ~ {slots:,} slots / {f_lru / tok_scored:.1f} insertions per "
              f"token = ~{slots / (f_lru / tok_scored):.0f} tokens of history held for free.")
        print(f"  its misses: {100 * comp / nm:.1f} % compulsory (pair never touched before); the "
              f"rest were last used {d10} / {d50} / {d90} tokens ago (p10/p50/p90).\n")

        n_miss, named = miss_coverage(calls, slots, cut, preds)
        print(f"  of LRU's {n_miss:,} demand misses, the predictor already names:")
        for name in preds:
            print(f"    {name:32s} {100 * named[name] / n_miss:5.1f} %")
        print()

        hdr = (f"  {'policy':32s} {'hit':>7s} {'gap':>6s} {'fetch/tok':>10s} {'MiB/tok':>9s} "
               f"{'xLRU':>6s} {'stolen':>8s} {'wasted':>8s}")
        print(hdr)
        rows = [("LRU (shipped)", base, 0)]
        rows.append(("P1 touch-only, zero bytes",
                     simulate(calls, slots, cut, preds["P1 same-layer prev token"][1],
                              N_LAYERS, touch_only=True), N_LAYERS))
        for name, (lead, sets) in preds.items():
            rows.append((name, simulate(calls, slots, cut, sets, lead), lead))

        for name, r, _lead in rows:
            h = r["hits"] / scored
            tot = r["demand_fetch"] + r["pre_fetch"]
            # "gap closed" on the miss axis: Belady is 1.00 by definition, LRU 0.00. Negative means
            # the arm is worse than what we already ship.
            gap = (h - h_lru) / (h_bel - h_lru) if h_bel > h_lru else 0.0
            stolen = sum(1 for x, y in zip(base["trace"], r["trace"]) if x and not y)
            print(f"  {name:32s} {100 * h:6.2f} % {100 * gap:5.0f} % {tot / tok_scored:10.1f} "
                  f"{tot * EXPERT_BYTES / tok_scored / 2**20:9.0f} {tot / f_lru:6.2f} "
                  f"{stolen:8,d} {r['wasted']:8,d}")
        print()

    print("  gap = fraction of the LRU->Belady hit-rate gap closed; xLRU = total NVMe bytes vs LRU.")
    print("  stolen = accesses LRU hit that this policy missed (prefetch evicted a live entry).")
    print("  wasted = prefetched slots evicted before anything asked for them.")
    print("  The deciding column is MiB/token. Anything with xLRU > 1.00 moves MORE bytes than the")
    print("  policy we already ship, which is the opposite of the point.\n")
    print("  VERDICT -- SHORT-HISTORY RECURRENCE PREDICTORS ARE CLOSED, and that is all this tool")
    print("  tests. The mechanism: LRU already retains tens of tokens of history, so every expert a")
    print("  k<=8-token predictor can name is ALREADY RESIDENT -- 0.0 % of LRU's misses. The misses")
    print("  are experts NOT used for a median of 45 tokens, so the lookahead needed is ~45 tokens,")
    print("  not the 1 token P1 has or the 5-8 a drafter gives.")
    print()
    print("  WHAT THIS DOES *NOT* CLOSE: P3 here is the naive identity map (expert e at L-1 predicts")
    print("  expert e at L) and is at chance -- 5.5 % overlap against a 5.7 % random baseline. That is")
    print("  NOT the cross-layer predictor we already measured. notes/ds41-measured-2026-09-13.md")
    print("  section 18 fitted per-source-layer TRANSITION TABLES and converted 32.4 % of misses at")
    print("  1.5x overfetch, seven times the 4.3 % popularity baseline -- real signal, rejected on")
    print("  economics (+118 % bytes), not on absence. A learned transition table is a strictly")
    print("  stronger predictor than anything here; do not cite this tool as having closed it.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
