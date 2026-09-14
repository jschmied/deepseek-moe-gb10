#!/usr/bin/env python3
"""Prefill I/O oracles, replayed offline from a recorded route log.

Answers "what is the layer-major transpose actually worth" without writing it. The engine records
one line per `ExpertStore.resolve()` call under `DSV41_ROUTE_LOG=<path>`:

    {"i": call, "L": layer, "pf": 1, "uniq": [expert ids wanted], "miss": [ids not resident]}

Expert identity is `(layer, expert)` and there is no cross-layer reuse, so "hold this layer's
experts until every chunk has consumed them" is very nearly the exact upper bound for layer-major
-- not a loose global Belady bound. Three byte figures come out of the same log:

  current        sum over calls of |miss|                 -- what the engine reads today
  layer-union    sum over LAYERS of |union of miss|       -- each missing expert read once per layer
  cold-arena     sum over LAYERS of |union of uniq|       -- every expert of the layer read once
                                                             even if the LRU happens to hold it, so
                                                             this number does not move with warmth
                                                             (it is ABOVE the residency-aware one)

and a delivery-time bound at the measured CB3 cache latencies, which separates the gain from
avoiding re-reads (bytes) from the gain from queue depth (ms/load at QD 1 vs 4).

  python tools/prefill_io_oracle.py route.jsonl [--compute-s S] [--record-bytes N]
"""
import argparse
import collections
import json
import sys

# engine/cb3_cache.py record size; the FP4 checkpoint path reads 18,800,640 instead.
CB3_RECORD = 13_774_848
FP4_RECORD = 18_800_640
# Measured, notes/native-cb3-expert-cache.md: ExpertStore._load_into_slot on the native CB3 cache,
# 96 loads, median of 3. Queue depth -> ms per load.
MS_PER_LOAD = {1: 4.14, 2: 2.23, 4: 2.04}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log")
    ap.add_argument("--record-bytes", type=int, default=CB3_RECORD)
    ap.add_argument("--compute-s", type=float, default=None,
                    help="prefill seconds NOT spent waiting on expert loads; enables the "
                         "max(compute, delivery) pipelined bound")
    ap.add_argument("--decode", action="store_true", help="score decode calls instead of prefill")
    a = ap.parse_args()

    want = 0 if a.decode else 1
    calls = [json.loads(l) for l in open(a.log) if l.strip()]
    calls = [c for c in calls if c["pf"] == want]
    if not calls:
        print(f"no {'decode' if a.decode else 'prefill'} calls in {a.log}")
        return 1

    per_layer_miss = collections.defaultdict(set)
    per_layer_uniq = collections.defaultdict(set)
    current = 0
    calls_per_layer = collections.Counter()
    for c in calls:
        L = c["L"]
        current += len(c["miss"])
        per_layer_miss[L] |= set(c["miss"])
        per_layer_uniq[L] |= set(c["uniq"])
        calls_per_layer[L] += 1

    union = sum(len(v) for v in per_layer_miss.values())
    arch = sum(len(v) for v in per_layer_uniq.values())
    chunks = max(calls_per_layer.values())
    R = a.record_bytes

    print(f"{len(calls)} {'decode' if a.decode else 'prefill'} resolve() calls over "
          f"{len(per_layer_miss)} layers, {chunks} chunks per layer, record {R/1e6:.2f} MB\n")
    rows = [("current (engine today)", current),
            ("layer-union oracle", union),
            ("layer-union, COLD arena (no LRU warmth)", arch)]
    print(f"  {'':44} {'loads':>8} {'GB':>8} {'vs current':>11}")
    for name, n in rows:
        print(f"  {name:44} {n:>8} {n*R/1e9:>8.1f} {current/max(n,1):>10.2f}x")

    print(f"\n  re-read fraction today: {100*(current-union)/max(current,1):.1f} % of loads are an "
          f"expert this layer had already fetched in an earlier chunk")

    print(f"\n  delivery time bound (measured CB3 ms/load, notes/native-cb3-expert-cache.md):")
    print(f"  {'':44} " + "".join(f"{'QD%d' % q:>10}" for q in sorted(MS_PER_LOAD)))
    for name, n in rows:
        print(f"  {name:44} " + "".join(f"{n*MS_PER_LOAD[q]/1000:>9.1f}s" for q in sorted(MS_PER_LOAD)))

    if a.compute_s is not None:
        # Per layer, a perfectly pipelined prefill cannot finish before the later of its compute
        # and its delivery. Compute is split evenly across layers, which is an assumption, not a
        # measurement -- the engine does not time layers separately.
        nL = len(per_layer_miss)
        comp = a.compute_s / nL
        print(f"\n  pipelined bound, max(compute, delivery) per layer, compute {a.compute_s:.1f}s "
              f"spread evenly over {nL} layers ({comp*1000:.0f} ms each) -- the split is ASSUMED:")
        cur_loads = collections.Counter()
        for c in calls:
            cur_loads[c["L"]] += len(c["miss"])
        for q in sorted(MS_PER_LOAD):
            for name, loads in (("current", cur_loads),
                                ("layer-union", {L: len(v) for L, v in per_layer_miss.items()})):
                tot = sum(max(comp, loads[L] * MS_PER_LOAD[q] / 1000) for L in per_layer_miss)
                print(f"    QD{q}  {name:14} {tot:6.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
