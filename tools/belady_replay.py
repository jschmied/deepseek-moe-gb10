#!/usr/bin/env python3
"""belady_replay.py -- what is the expert cache actually leaving on the table?

det-198 (qwen38-flash-next-gb10/notes/determinism-investigation.md) ran this for a 48x512 MoE and
concluded: plain global LRU 90.1 %, Belady 95.6 %, a protected static set 84.8 % -- i.e. the
segmented / "task profile" family is WORSE than what we already run, because routing entropy is
near-uniform and all the hit rate is temporal locality. That was a different model. This replays
the same question on DS4.1's own routing, captured with DSV41_ROUTE_LOG.

The point is the gap, not the ranking: LRU-to-Belady is the entire budget available to ANY smarter
policy. If it is small, every policy idea is dead and the only lever left is moving fewer bytes.

Policies:
  LRU              what the engine runs today
  Belady / MIN     evict the entry whose next use is furthest away (unimplementable, the ceiling)
  static-ranked    keep the globally hottest pairs, never evict (the "universal keepset" idea)
  segmented        protected static set + LRU over the rest (the "task profile" idea)

Frequency-ranked policies are ranked on a held-out PREFIX and scored on the rest; without that
split they are oracles too, and det-198 records that the in-sample version reverses the ranking.

Usage:
    belady_replay.py route.jsonl [--slots 4788] [--decode-only] [--rank-frac 0.25]
"""

from __future__ import annotations

import argparse
import collections
import heapq
import json
import sys

N_EXPERTS = 384


def load(path: str, decode_only: bool):
    """-> a flat access sequence of (layer, expert) ids, in forward order."""
    seq = []
    calls = 0
    for line in open(path):
        line = line.strip()
        if not line:
            continue
        try:
            o = json.loads(line)
        except json.JSONDecodeError:
            continue
        if decode_only and o.get("pf"):
            continue
        calls += 1
        L = o["L"]
        seq.extend(L * N_EXPERTS + e for e in o["uniq"])
    return seq, calls


def lru(seq, slots, score_from: int = 0):
    """`score_from` warms the cache on seq[:score_from] without counting those accesses."""
    cache = collections.OrderedDict()
    hits = 0
    for i, k in enumerate(seq):
        if k in cache:
            cache.move_to_end(k)
            hits += (i >= score_from)
        else:
            if len(cache) >= slots:
                cache.popitem(last=False)
            cache[k] = 1
    return hits


def belady(seq, slots, score_from: int = 0):
    """Evict the resident entry whose next use is furthest in the future.

    next_use[i] = the next index where seq[i] recurs, or +inf. The heap is max-by-next_use with
    lazy deletion: a popped entry whose recorded next_use is stale is simply discarded.
    """
    nxt = [len(seq)] * len(seq)
    last = {}
    for i in range(len(seq) - 1, -1, -1):
        k = seq[i]
        nxt[i] = last.get(k, len(seq))
        last[k] = i
    cache, heap, hits = {}, [], 0
    for i, k in enumerate(seq):
        if k in cache:
            hits += (i >= score_from)
        elif len(cache) >= slots:
            while True:                       # lazy deletion
                neg, kk = heapq.heappop(heap)
                if cache.get(kk) == -neg:
                    del cache[kk]
                    break
        cache[k] = nxt[i]
        heapq.heappush(heap, (-nxt[i], k))
    return hits


def static_ranked(seq, slots, rank_frac):
    cut = int(len(seq) * rank_frac)
    freq = collections.Counter(seq[:cut])
    keep = {k for k, _ in freq.most_common(slots)}
    return sum(1 for k in seq[cut:] if k in keep), len(seq) - cut


def segmented(seq, slots, rank_frac, protect_frac=0.5):
    """A protected static set of the hottest pairs, LRU over the remaining slots."""
    cut = int(len(seq) * rank_frac)
    freq = collections.Counter(seq[:cut])
    n_prot = int(slots * protect_frac)
    prot = {k for k, _ in freq.most_common(n_prot)}
    cache = collections.OrderedDict()
    room = slots - len(prot)
    hits = 0
    for k in seq[cut:]:
        if k in prot:
            hits += 1
            continue
        if k in cache:
            cache.move_to_end(k)
            hits += 1
        else:
            if len(cache) >= room:
                cache.popitem(last=False)
            cache[k] = 1
    return hits, len(seq) - cut


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("log")
    ap.add_argument("--slots", type=int, default=4788, help="LRU slots (engine: n_slots - transient)")
    ap.add_argument("--decode-only", action="store_true")
    ap.add_argument("--rank-frac", type=float, default=0.25)
    a = ap.parse_args()

    seq, calls = load(a.log, a.decode_only)
    if not seq:
        print("no accesses (is this a prefill-only log and --decode-only set?)", file=sys.stderr)
        return 1
    distinct = len(set(seq))
    print(f"  {len(seq):,} accesses over {calls:,} resolve calls, {distinct:,} distinct "
          f"(layer, expert) pairs of {40 * N_EXPERTS:,}")
    print(f"  cache {a.slots:,} slots = {100 * a.slots / (40 * N_EXPERTS):.1f} % of all pairs\n")

    cut = int(len(seq) * a.rank_frac)
    scored = len(seq) - cut
    h_lru_full = lru(seq, a.slots)
    h_bel_full = belady(seq, a.slots)
    # Warm LRU and Belady on the SAME prefix the ranked policies rank on, and count hits only over
    # the suffix. Scoring them from an empty cache while static_ranked and segmented start with
    # their chosen set already resident is not apples-to-apples: it hands the static policies a free
    # warm cache and charges the adaptive ones every compulsory miss. That bias is large enough to
    # invent a result -- it is why an earlier run had `segmented` beating LRU by 1.0 pp, which was
    # the warm-up, not the policy. External review, 2026-09-15.
    h_lru = lru(seq, a.slots, score_from=cut)
    h_bel = belady(seq, a.slots, score_from=cut)
    h_st, _ = static_ranked(seq, a.slots, a.rank_frac)
    h_sg, _ = segmented(seq, a.slots, a.rank_frac)

    print(f"  {'policy':28s} {'hit rate':>9s} {'miss rate':>10s}")
    for name, h in (("static trace-ranked", h_st), ("segmented (protected + LRU)", h_sg),
                    ("global LRU", h_lru), ("Belady (full oracle)", h_bel)):
        print(f"  {name:28s} {100 * h / scored:8.1f} % {100 * (1 - h / scored):9.2f} %")
    gap = (h_bel - h_lru) / scored
    m_lru, m_bel = 1 - h_lru / scored, 1 - h_bel / scored
    print(f"\n  LRU -> Belady: +{100 * gap:.1f} pp hit, but misses {100 * m_lru:.2f} % -> "
          f"{100 * m_bel:.2f} % = {100 * (1 - m_bel / m_lru):.0f} % less NVMe traffic at the ceiling.")
    print(f"  (whole-sequence, for reference: LRU {100 * h_lru_full / len(seq):.1f} %, "
          f"Belady {100 * h_bel_full / len(seq):.1f} %)")
    ent = 0.0
    import math
    c = collections.Counter(e % N_EXPERTS for e in seq)
    tot = sum(c.values())
    for v in c.values():
        p = v / tot
        ent -= p * math.log2(p)
    print(f"  routing entropy {ent:.2f} of {math.log2(N_EXPERTS):.2f} bits "
          f"(near-uniform = no hot set to protect, so only recency can win)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
