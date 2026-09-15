#!/usr/bin/env python3
"""transition_prefetch.py -- the cross-layer transition predictor, filtered through the cache.

notes/ds41-measured-2026-09-13.md section 18 measured a learned per-source-layer transition table
T_L[a,b] ~ P(b in E_{L+d} | a in E_L), scored as sum over the active set. It recalls 30.5 % of layer
L+1's activations against a 14.4 % static-popularity baseline -- real signal, roughly double
popularity, and it survived the "pooled table" bug that had made it look like chance.

It was then rejected on economics: broad prefetch at 4-layer lookahead, N=24 candidates per layer,
converted ~2.3 % of blocking misses for ~+39 % traffic. Two things were wrong with that rejection
and this tool fixes both.

1. THE PREDICTOR WAS NOT CACHE-AWARE. It predicted what layer L+d would USE. But ~22 experts are
   active per call and only ~1.7 of them MISS -- the rest are already resident, so most of the
   prefetch budget was spent re-fetching things we already hold. The right target is "which of
   L+d's activations will be NONRESIDENT", which requires running the predictor's candidate list
   through the simulated cache at the instant the prefetch is issued, and discarding every
   candidate that is resident right then. That is the `if k in cache: continue` in simulate().

2. THE COST MODEL WAS WRONG. A CORRECT prefetch costs ZERO extra bytes: that expert was going to be
   read from NVMe anyway, just as a blocking miss instead. Prefetching only adds traffic when it is
   WRONG. So "+39 % traffic" is not the price of the conversions, it is the price of the mistakes,
   and the two must be counted separately. Here `total = demand_fetch + pre_fetch`, so a useful
   prefetch moves one read from the demand column to the prefetch column and the total is
   unchanged; only wasted prefetches push xLRU above 1.00.

Consequence for the sweep: N = 1, 2, 4, 8, not 24/48/96. Each layer has ~1.7 real misses, so any
N far above that is waste by construction no matter how good the ranking is.

Two further axes, added after the first pass showed the cache-aware filter alone was nearly a null:

3. HOW LONG A WRONG PREFETCH KEEPS THE SLOT IT STOLE is a policy choice, not a fact, and it turned
   out to dominate everything else. Three placements are crossed with (N, d): A insert at MRU (what
   "prefetch into the LRU" means, and the arm that hurts), B probationary FIFO, C eager discard at
   the moment the target call resolves. See `simulate()`'s `place` argument.

4. A PREFETCH COMPETES FOR I/O WITH THE MISSES THE ENGINE IS WAITING ON RIGHT NOW, which no pure
   cache simulation can see. Class `IO` prices it with the box's measured device numbers, and the
   headline there is TOTAL BLOCKING WAIT per token, not misses remaining: a policy that converts
   30 % of misses while delaying the rest can be net negative, and P1 below is exactly that.

Machinery is reused from tools/prefetch_oracle.py rather than re-derived: the same call loader, the
same LRU replay with prefetch-into-LRU and eviction accounting, the same EXPERT_BYTES price, the
same warm-on-the-same-prefix rule, and the same `stolen` diff. The one thing that is new is the
residency filter at issue time, and the report axis (blocking misses per token, not hit rate).

Measurement rules kept from that tool, all of them learned the hard way:
  * Fit the tables on a held-out 25 % prefix and score only the suffix. An earlier tool in this repo
    fitted and scored on the same data and manufactured a result.
  * Warm every policy on the SAME prefix, rounded to a whole token so the warm cache ends on a
    layer-39 boundary.
  * The prefetch is issued `d` resolve calls before the target, which is the lookahead the engine
    really has -- not immediately before the access it serves.
  * A prefetch displaces the LRU tail. `stolen` = accesses plain LRU hit that this policy missed.
  * `wasted` = prefetched entries evicted before anything demanded them.

Usage:
    transition_prefetch.py ~/ds41-queue/logs/route-decode.jsonl [--slots 5328 3000]
CPU only; no GPU, no server.
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
import time

import numpy as np

from prefetch_oracle import belady

N_EXPERTS = 384
N_LAYERS = 40
# One native CB3 disk record -- what a miss actually reads on the shipped server, and therefore
# exactly what a prefetch costs when it is wrong (prefetch_oracle.py, corrected 2026-09-15).
EXPERT_BYTES = 13_774_848
# The full ranking is precomputed per call: "top N nonresident" is taken literally, with no depth
# cap. A cap looks harmless and is not -- at 5,328 slots a depth-96 list was exhausted on 181k
# issues, because the highest-scoring candidates are also the most popular ones and therefore the
# most likely to be RESIDENT. Those arms then quietly prefetched fewer than N and looked both
# cheaper and weaker than the policy they claimed to measure.
CAND_DEPTH = N_EXPERTS

# Feasibility constants, all measured elsewhere and only arithmetic here.
PLACE_NAME = {"mru": "A mru", "prob": "B prob", "discard": "C discard"}
LAYER_MS = 90.0        # one resolve call of wall clock at today's decode speed, 3.6 s per token
# Device numbers, measured on this box with whole-expert O_DIRECT preads:
B_SINGLE = 5.58e9      # one read in flight
B_TOTAL = 6.82e9       # two in flight -- the SSD is essentially saturated at ~2 concurrent reads,
                       # so beyond that concurrency buys queueing, not throughput
# The share of a resolve call that is NOT load-wait: section 19 measured 79 % load-wait / 21 % route,
# so this is the compute the engine can overlap a prefetch with.
COMPUTE_MS = LAYER_MS * 0.21


def load(path: str):
    """-> [(layer, (expert ids,)), ...] for decode calls only, in forward order.

    Identical to prefetch_oracle.load. The log opens with 40 pf=1 prefill calls and then runs
    strictly cyclic 0..39, so call index // 40 is the token and call index % 40 is the layer.
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


# --------------------------------------------------------------------------- the predictor

def fit_tables(calls, cut, dists):
    """Per-SOURCE-LAYER transition tables, fitted on calls[:cut] only.

    T[(L, d)][a, b] = P(b active at call j+d | a active at call j, layer(j) = L).

    Per source layer, NOT pooled over all (i, i+d) pairs: section 18 records that pooling halves the
    recall (16.8 % vs 45.1 %) and lands the predictor exactly on the popularity baseline, which is
    what made the first attempt look like a clean kill.

    Distance is measured in RESOLVE CALLS, not in layers, so d wraps across the token boundary on
    its own statistics (source layer 39, d=1 predicts the next token's layer 0). That keeps "d" the
    literal lookahead budget -- d layer-times of compute -- for every source layer, instead of
    leaving the last d layers of each token with no arm at all.
    """
    tables = {}
    for d in dists:
        cnt = np.zeros((N_LAYERS, N_EXPERTS, N_EXPERTS), np.float32)
        rows = np.zeros((N_LAYERS, N_EXPERTS), np.float32)
        for j in range(cut - d):
            L, ex = calls[j]
            tgt = calls[j + d][1]
            a = np.fromiter(ex, np.int64, len(ex))
            b = np.fromiter(tgt, np.int64, len(tgt))
            cnt[L][np.ix_(a, b)] += 1.0
            rows[L][a] += 1.0
        with np.errstate(invalid="ignore", divide="ignore"):
            T = cnt / np.maximum(rows, 1.0)[:, :, None]
        for L in range(N_LAYERS):
            tables[(L, d)] = T[L]
    return tables


def fit_popularity(calls, cut):
    """Static per-layer frequency prior, fitted on the same held-out prefix. The control."""
    freq = np.zeros((N_LAYERS, N_EXPERTS), np.float32)
    for L, ex in calls[:cut]:
        freq[L][np.fromiter(ex, np.int64, len(ex))] += 1.0
    return freq


def rank_candidates(calls, tables, freq, d):
    """-> (trans_order, pop_order): int16 [n_calls, CAND_DEPTH] candidate rankings for target i+d.

    The RANKING does not depend on the cache -- only the residency FILTER does -- so it is computed
    once here and shared by every N and every cache size. Entries are expert ids at the target
    layer; the caller turns them into global keys.
    """
    n = len(calls)
    trans = np.zeros((n, CAND_DEPTH), np.int16)
    pop = np.zeros((n, CAND_DEPTH), np.int16)
    pop_order = np.argsort(-freq, axis=1, kind="stable")[:, :CAND_DEPTH].astype(np.int16)
    for i in range(n - d):
        L, ex = calls[i]
        tl = calls[i + d][0]
        a = np.fromiter(ex, np.int64, len(ex))
        score = tables[(L, d)][a].sum(axis=0)
        trans[i] = np.argsort(-score, kind="stable")[:CAND_DEPTH]
        pop[i] = pop_order[tl]
    return trans, pop


# --------------------------------------------------------------------------- the I/O model

class IO:
    """A processor-sharing NVMe model with a finite number of staging leases.

    MODELLED, NOT MEASURED. It exists to stop the cache study from scoring latency wins the I/O
    path cannot deliver: a prefetch for layer L+d competes for bandwidth AND for a staging lease
    with the blocking misses of layer L that the engine is waiting on right now.

      * `cap` concurrent reads (the DSV41_IO_THREADS pinned buffers). Admission is FIFO with NO
        priority for blocking reads -- that is the point: a prefetch holding a lease makes a
        blocking miss wait even when bandwidth is free.
      * bandwidth is shared: with n reads in flight each gets min(B_SINGLE, B_TOTAL / n), so one
        read runs at 5.58 GB/s and any n >= 2 splits 6.82 GB/s. Concurrency beyond ~2 adds
        queueing, not throughput, exactly as measured.
    """

    def __init__(self, cap):
        self.cap = cap
        self.t = 0.0
        self.active = {}                   # key -> remaining bytes
        self.q = collections.deque()       # waiting for a lease

    def _admit(self):
        while len(self.active) < self.cap and self.q:
            self.active[self.q.popleft()] = float(EXPERT_BYTES)

    def busy(self, k):
        return k in self.active or k in self.q

    def free_leases(self):
        return max(0, self.cap - len(self.active) - len(self.q))

    def submit(self, k):
        if self.busy(k):
            return
        self.q.append(k)
        self._admit()

    def _step(self, max_dt):
        if not self.active:
            self.t += max_dt
            return max_dt
        rate = min(B_SINGLE, B_TOTAL / len(self.active))
        dt = min(max_dt, min(self.active.values()) / rate)
        for k in list(self.active):
            self.active[k] -= rate * dt
            if self.active[k] <= 1e-3:
                del self.active[k]
        self.t += dt
        self._admit()
        return dt

    def advance(self, dt):
        rem = dt
        while rem > 1e-12:
            rem -= self._step(rem)

    def finish(self, keys):
        """Advance until every key in `keys` has completed. -> elapsed seconds (the blocking wait)."""
        t0 = self.t
        pend = [k for k in keys if self.busy(k)]
        while pend:
            if not self.active and not self.q:
                break
            self._step(float("inf"))
            pend = [k for k in pend if self.busy(k)]
        return self.t - t0


# --------------------------------------------------------------------------- the simulator

def simulate(calls, slots, cut, order=None, d=0, topn=0, place="mru",
             io_cap=None, issue="always"):
    """LRU with cache-aware prefetch. Returns counters plus a per-access hit bytearray.

    At call i, after layer L's activations are known and the cache reflects them, the ranked
    candidate list for call i+d is walked from the top; every candidate ALREADY RESIDENT is skipped
    (it needs no fetch, and spending a slot of budget on it is the exact mistake the broad version
    made), and the first `topn` nonresident ones are fetched.

    `place` is the PLACEMENT / RETENTION axis: how long a WRONG prefetch is allowed to occupy the
    slot it stole. That is a policy choice, not a fact, and it is where `stolen` is decided.

      "mru"      A -- ordinary LRU insertion at the head. A wrong prefetch then ages out over the
                 full retention horizon (~82 tokens at 5,328 slots). The pessimistic arm, and the
                 one the plain "prefetch into the LRU" spec implies.
      "prob"     B -- probationary: prefetches land in a separate FIFO queue that is the first
                 thing evicted; a demand hit promotes the entry into the main LRU at MRU. No
                 detection of any kind, the classic 2Q/LIRS A1 trick.
                 NOTE, and this is why it is a separate queue rather than a literal insert at the
                 LRU tail of one list: with a single list and a full cache, "insert at the tail"
                 makes each prefetch the immediate victim of the NEXT insertion, so a batch of N
                 evicts all but its own last member and arm B would be degenerate by construction.
                 The FIFO queue evicts the OLDEST speculative entry instead, which is what
                 "probationary" actually means.
      "discard"  C -- eager discard: every prefetch is tagged with the call it was predicted for.
                 When that call resolves, an entry that is not in its `uniq` is evicted on the spot
                 and the slot returned. Exact, and it bounds a wrong prefetch's residency to d
                 resolve calls (~90 ms each) instead of tens of tokens. A prefetch that IS used
                 becomes an ordinary main-cache entry. Insertion is at MRU, so C is exactly A plus
                 the discard.

    `spec` tracks entries that are prefetched, resident, and whose target call has not resolved yet
    -- the real capacity cost of speculation, sampled once per call.

    `io_cap` switches on the I/O model (class IO above), which prices the CONTENTION a pure cache
    simulation cannot see. With it on, each call submits its demand misses as blocking reads, waits
    for them (and for any prefetch of its own experts that is still in flight -- partial credit,
    the wait is only the remainder), then overlaps COMPUTE_MS of compute with whatever is still
    running. `issue` is the design choice being tested:

      "always"  P1 -- issue the prefetch as soon as the prediction is available, i.e. alongside
                this call's blocking misses. It competes for bandwidth and for leases.
      "idle"    P2 -- hold it until this call's blocking reads have all completed, then issue only
                as many as there are FREE LEASES. Anything that does not fit is dropped outright
                and never occupies a cache slot either. P2 can only fill genuine gaps, never steal.
    """
    main = collections.OrderedDict()      # the LRU proper
    prob = collections.OrderedDict()      # probationary FIFO, arm B only
    pending = set()                       # prefetched, resident, not yet demanded
    spec = set()                          # prefetched, resident, target call not yet resolved
    tag = {}                              # key -> the call index it was prefetched for
    by_target = collections.defaultdict(list)
    hits = demand_fetch = pre_fetch = wasted = pre_useful = discarded = 0
    deepest = short = 0
    spec_sum = spec_max = spec_n = 0
    per_access = bytearray()
    n = len(calls)
    io = IO(io_cap) if io_cap else None
    blocking_wait = 0.0
    dropped = 0

    def evict(scoring):
        nonlocal wasted
        if prob:
            ev, _ = prob.popitem(last=False)   # oldest speculative first
        else:
            ev, _ = main.popitem(last=False)   # coldest resident
        spec.discard(ev)
        if ev in pending:
            pending.discard(ev)
            if scoring:
                wasted += 1

    for i in range(n):
        L, ex = calls[i]
        scoring = i >= cut
        needed = []
        for e in ex:
            k = L * N_EXPERTS + e
            if k in main:
                main.move_to_end(k)
                hit = True
            elif k in prob:
                del prob[k]                    # promote out of probation into the LRU proper
                main[k] = 1
                hit = True
            else:
                hit = False
            if hit:
                if io and io.busy(k):
                    needed.append(k)       # prefetched but still in flight: wait for the remainder
                spec.discard(k)
                if k in pending:
                    pending.discard(k)
                    if scoring:
                        pre_useful += 1
                if scoring:
                    hits += 1
                    per_access.append(1)
            else:
                if len(main) + len(prob) >= slots:
                    evict(scoring)
                main[k] = 1
                if io:
                    io.submit(k)
                    needed.append(k)
                if scoring:
                    demand_fetch += 1
                    per_access.append(0)

        # Arm C: this call has now resolved, so every entry prefetched FOR it that it did not use is
        # known wrong. Evict it and return the slot. (Anything it did use left `pending` just above.)
        for k in by_target.pop(i, ()):
            spec.discard(k)
            if place == "discard" and k in pending and k in main:
                del main[k]
                pending.discard(k)
                if scoring:
                    wasted += 1
                    discarded += 1

        def do_issue(limit):
            nonlocal pre_fetch, short, deepest, dropped
            tgt = i + d
            if order is None or not topn or tgt >= n:
                return
            if limit is not None and limit <= 0:
                if scoring:
                    dropped += topn
                return
            want = topn if limit is None else min(topn, limit)
            base_k = calls[tgt][0] * N_EXPERTS
            got = 0
            row = order[i]
            for pos in range(CAND_DEPTH):
                k = base_k + int(row[pos])
                if k in main or k in prob:         # <-- the cache-aware filter
                    continue
                if len(main) + len(prob) >= slots:
                    evict(scoring)
                if place == "prob":
                    prob[k] = 1
                else:
                    main[k] = 1
                pending.add(k)
                spec.add(k)
                tag[k] = tgt
                by_target[tgt].append(k)
                if io:
                    io.submit(k)
                if scoring:
                    pre_fetch += 1
                got += 1
                if got >= want:
                    deepest = max(deepest, pos + 1)
                    break
            else:
                # The candidate list ran out before `want` nonresident entries were found: this arm
                # then prefetches fewer than N and is UNDERSTATED. Reported, never silent.
                short += 1
                deepest = CAND_DEPTH
            if scoring:
                dropped += topn - got

        if io is None or issue == "always":
            do_issue(None)
        if io:
            w = io.finish(needed)
            if scoring:
                blocking_wait += w
            if issue == "idle":
                do_issue(io.free_leases())
            io.advance(COMPUTE_MS / 1000.0)

        if scoring:
            spec_sum += len(spec)
            spec_max = max(spec_max, len(spec))
            spec_n += 1

    return {
        "hits": hits, "accesses": len(per_access), "demand_fetch": demand_fetch,
        "pre_fetch": pre_fetch, "wasted": wasted, "pre_useful": pre_useful, "discarded": discarded,
        "trace": per_access, "deepest": deepest, "short": short,
        "spec_avg": spec_sum / max(spec_n, 1), "spec_max": spec_max,
        "wait": blocking_wait, "dropped": dropped, "t": io.t if io else 0.0,
    }


def diff(base_trace, trace):
    """-> (converted, stolen). Exact, per access, against the plain-LRU replay of the same stream.

    converted = accesses LRU MISSED and this policy HIT: blocking misses actually removed.
    stolen    = accesses LRU HIT and this policy MISSED: blocking misses this policy ADDED by
                evicting a live entry to make room for a prefetch.
    """
    conv = sto = 0
    for x, y in zip(base_trace, trace):
        if y and not x:
            conv += 1
        elif x and not y:
            sto += 1
    return conv, sto


# --------------------------------------------------------------------------- main

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("log")
    ap.add_argument("--slots", type=int, nargs="+", default=[5328, 3000],
                    help="shipped lru_slots = 5728 - 400 transient = 5328; 3000 checks rank stability")
    ap.add_argument("--prefix", type=float, default=0.25, help="fitting / warm-up fraction")
    ap.add_argument("--ns", type=int, nargs="+", default=[1, 2, 4, 8])
    ap.add_argument("--dists", type=int, nargs="+", default=[1, 2, 4])
    ap.add_argument("--io-caps", type=int, nargs="+", default=[2, 6],
                    help="staging leases in the I/O model: 2 = the SSD saturation point, 6 = what the engine runs")
    ap.add_argument("--places", nargs="+", default=["mru", "prob", "discard"],
                    help="placement/retention of a prefetched entry: A mru, B probationary, C eager discard")
    a = ap.parse_args()

    calls = load(a.log)
    if not calls:
        print("no decode calls in log", file=sys.stderr)
        return 1
    n = len(calls)
    accesses = sum(len(ex) for _, ex in calls)
    tokens = n / N_LAYERS
    cut = int(n * a.prefix) // N_LAYERS * N_LAYERS
    tok_scored = (n - cut) / N_LAYERS

    print(f"  {n:,} decode resolve calls, {accesses:,} activations, {tokens:.0f} tokens, "
          f"{accesses / tokens:.1f} activations/token")
    print(f"  fit + warm on calls [0,{cut:,}) = {100 * cut / n:.0f} %; score the suffix "
          f"({n - cut:,} calls, {tok_scored:.0f} tokens)")
    print(f"  one fetch = {EXPERT_BYTES:,} B = {EXPERT_BYTES / 2**20:.2f} MiB of NVMe\n")

    t0 = time.time()
    tables = fit_tables(calls, cut, a.dists)
    freq = fit_popularity(calls, cut)
    orders = {d: rank_candidates(calls, tables, freq, d) for d in a.dists}
    print(f"  tables fitted on the prefix only, {len(tables)} per-source-layer tables, "
          f"{time.time() - t0:.1f} s\n")

    for slots in a.slots:
        print(f"  ===== LRU cache, {slots:,} slots "
              f"({100 * slots / (N_LAYERS * N_EXPERTS):.1f} % of the 15,360 (layer, expert) pairs) =====")
        base = simulate(calls, slots, cut)
        scored = base["accesses"]
        m_lru = base["demand_fetch"]
        bytes_lru = m_lru * EXPERT_BYTES
        _, f_bel, _ = belady(calls, slots, cut)
        print(f"  plain LRU: {100 * base['hits'] / scored:.2f} % hit, "
              f"{m_lru:,} blocking misses in the scored window = {m_lru / tok_scored:.2f} per token, "
              f"{bytes_lru / tok_scored / 2**20:.1f} MiB/token")
        print(f"  Belady floor (needs the whole future, unbuildable): {f_bel / tok_scored:.2f} blocking "
              f"misses per token. That is the entire budget any policy can play for.\n")

        hdr = (f"  {'policy':11s} {'place':9s} {'N':>2s} {'d':>2s} {'blk miss/tok':>13s} {'vs LRU':>8s} "
               f"{'recall':>7s} {'wasted':>7s} {'prec':>6s} {'MiB/tok':>8s} {'xLRU':>6s} "
               f"{'stolen':>7s} {'spec avg':>9s} {'max':>5s}")
        print(hdr)
        print(f"  {'plain LRU':11s} {'-':9s} {'-':>2s} {'-':>2s} {m_lru / tok_scored:13.2f} {'-':>8s} "
              f"{'-':>7s} {'-':>7s} {'-':>6s} {bytes_lru / tok_scored / 2**20:8.1f} {1.00:6.2f} "
              f"{'-':>7s} {'-':>9s} {'-':>5s}")

        deepest = short = 0
        for d in a.dists:
            trans_order, pop_order = orders[d]
            for label, order in (("transition", trans_order), ("popularity", pop_order)):
                for place in a.places:
                    for N in a.ns:
                        r = simulate(calls, slots, cut, order, d, N, place)
                        deepest = max(deepest, r["deepest"])
                        short += r["short"]
                        conv, sto = diff(base["trace"], r["trace"])
                        tot = r["demand_fetch"] + r["pre_fetch"]
                        pre = r["pre_fetch"]
                        prec = r["pre_useful"] / pre if pre else 0.0
                        print(f"  {label:11s} {PLACE_NAME[place]:9s} {N:2d} {d:2d} "
                              f"{r['demand_fetch'] / tok_scored:13.2f} "
                              f"{100 * (r['demand_fetch'] - m_lru) / m_lru:+7.1f}% "
                              f"{100 * conv / m_lru:6.1f}% {r['wasted']:7,d} "
                              f"{100 * prec:5.1f}% {tot * EXPERT_BYTES / tok_scored / 2**20:8.1f} "
                              f"{tot / m_lru:6.2f} {sto:7,d} {r['spec_avg']:9.1f} {r['spec_max']:5d}")
                    print()
        print(f"  candidate list: deepest position used {deepest} of {CAND_DEPTH}, "
              f"{short:,} issues ran short of N (0 = no arm above was truncated)\n")

    print("  blk miss/tok = BLOCKING misses per token left after prefetching -- the headline, because")
    print("    it is the latency the engine actually waits on (decode is ~78 % expert-fetch wait).")
    print("  recall  = of plain LRU's blocking misses, the fraction this policy turned into hits")
    print("            (exact per-access diff against the LRU replay of the same stream).")
    print("  useful/wasted = prefetched entries later demanded / evicted before anyone asked.")
    print("  MiB/tok, xLRU = TOTAL NVMe bytes, demand + prefetch. A useful prefetch is byte-neutral:")
    print("    it moves one read from the demand column to the prefetch column. Only WRONG")
    print("    prefetches push xLRU above 1.00.")
    print("  stolen  = accesses LRU hit that this policy missed, because a prefetch evicted a live")
    print("            entry. It is already inside blk miss/tok; shown so the damage is visible.")
    print("  spec avg/max = slots held by prefetched entries whose target call has not resolved yet:")
    print("            the standing capacity cost of speculation.")

    # ------------------------------------------------------------------ the I/O contention model
    slots = a.slots[0]
    print(f"  ===== I/O CONTENTION MODEL, {slots:,} slots, placement C (eager discard) =====")
    print("  MODELLED, NOT MEASURED -- do not quote these as measurements. Assumptions, all stated:")
    print(f"    * one whole-expert O_DIRECT pread = {EXPERT_BYTES:,} B; one in flight runs at "
          f"{B_SINGLE / 1e9:.2f} GB/s,")
    print(f"      any n >= 2 in flight SHARE {B_TOTAL / 1e9:.2f} GB/s (the SSD saturates at ~2 "
          f"concurrent reads),")
    print("      so concurrency past 2 buys queueing, not throughput.")
    print("    * `cap` staging leases (DSV41_IO_THREADS pinned buffers), FIFO admission with NO")
    print("      priority for blocking reads: a prefetch holding a lease makes a miss wait.")
    print(f"    * a call waits for its own misses, then overlaps {COMPUTE_MS:.1f} ms of compute")
    print("      (the 21 % of a resolve call that is not load-wait) with whatever is still running.")
    print("    * a demand for an expert whose prefetch is still in flight waits only for the")
    print("      REMAINDER of that read -- partial credit, not a full miss.")
    print("    * the model prices DEVICE time only. It does not reproduce the engine's ~90 ms per")
    print("      layer, because most of that is scheduling gaps, not the device. Read the wait")
    print("      column as a ratio between policies, never as a forecast of decode wall.\n")

    print(f"  {'issue':9s} {'N':>2s} {'d':>2s} {'cap':>4s} {'blk miss/tok':>13s} {'recall':>7s} "
          f"{'wait ms/tok':>12s} {'vs LRU':>8s} {'pref/tok':>9s} {'drop/tok':>9s} {'stolen':>7s}")
    for cap in a.io_caps:
        b_io = simulate(calls, slots, cut, io_cap=cap)
        w_lru = b_io["wait"] * 1e3 / tok_scored
        m_io = b_io["demand_fetch"]
        print(f"  {'LRU':9s} {'-':>2s} {'-':>2s} {cap:4d} {m_io / tok_scored:13.2f} {'-':>7s} "
              f"{w_lru:12.2f} {'-':>8s} {'-':>9s} {'-':>9s} {'-':>7s}")
        for d in a.dists:
            for issue in ("always", "idle"):
                for N in a.ns:
                    r = simulate(calls, slots, cut, orders[d][0], d, N, "discard", cap, issue)
                    conv, sto = diff(b_io["trace"], r["trace"])
                    w = r["wait"] * 1e3 / tok_scored
                    print(f"  {('P1 ' + issue) if issue == 'always' else 'P2 idle':9s} {N:2d} {d:2d} "
                          f"{cap:4d} {r['demand_fetch'] / tok_scored:13.2f} "
                          f"{100 * conv / m_io:6.1f}% {w:12.2f} {100 * (w - w_lru) / w_lru:+7.1f}% "
                          f"{r['pre_fetch'] / tok_scored:9.1f} {r['dropped'] / tok_scored:9.1f} "
                          f"{sto:7,d}")
            print()
    print("  wait ms/tok = TOTAL BLOCKING WAIT per token: the headline here, because a policy that")
    print("    converts misses but delays the ones it does not convert can be net negative.")
    print("  pref/tok, drop/tok = prefetches issued / suppressed for want of a free lease (P2 only).")
    print()

    read_ms = EXPERT_BYTES / (B_SINGLE * 1e-9 * 1e9) * 1e3
    print(f"\n  FEASIBILITY -- does the prefetch have time to land?")
    print(f"  One resolve call is ~{LAYER_MS:.0f} ms of wall clock ({40 * LAYER_MS / 1000:.1f} s per token), "
          f"so d=1 buys ~{LAYER_MS:.0f} ms, d=2 ~{2 * LAYER_MS:.0f} ms, d=4 ~{4 * LAYER_MS:.0f} ms.")
    print(f"  One {EXPERT_BYTES:,} B O_DIRECT read alone at {B_SINGLE / 1e9:.2f} GB/s is {read_ms:.2f} ms.")
    print(f"  {'N':>2s} {'d':>2s} {'issue time':>11s} {'window':>8s} {'duty':>7s} {'in flight':>10s}")
    for d in a.dists:
        for N in a.ns:
            print(f"  {N:2d} {d:2d} {N * read_ms:10.1f} ms {d * LAYER_MS:7.0f} ms "
                  f"{100 * N * read_ms / (d * LAYER_MS):6.1f}% {N * d:10d}")
    print("  In-flight is the issue-side bound N*d; the staging leases cap the real number lower.")
    print("  So the WINDOW is never the constraint: even N=8 at d=1 spends 22 % of one layer's wall")
    print("  clock issuing, and a d=4 prefetch has 360 ms to cover a 2.5 ms read. What constrains")
    print("  this is not time, it is the two things the tables above measure: the slot a wrong")
    print("  prefetch occupies, and the bandwidth/lease it takes from a miss already on the critical")
    print("  path. Both are fixable by policy -- eager discard for the first, idle-only issue for")
    print("  the second -- which is why the C/P2 arms are the only ones that win on both axes.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
