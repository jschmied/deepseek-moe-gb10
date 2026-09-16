#!/usr/bin/env python3
"""GPU busy fraction inside one NVTX range of an nsys report.

Answers exactly one question: with a WARM cache, where the NVMe pipe is empty because the work is
already resident rather than because anything is stalled, does the GPU actually run without pause?

Two things this gets right that the earlier budget did not:
  * GRAPH NODES COUNT. Without --cuda-graph-trace=node, Nsight records each graph LAUNCH as a
    single activity and emits no kernel rows for its nodes; decode runs entirely in captured
    graphs, so CUPTI_ACTIVITY_KIND_KERNEL alone saw only the non-graph work and undercounted GPU
    busy by 8x. Both tables are unioned here, and merge() makes that idempotent whichever way the
    capture was taken.
  * THE RANGE. A trace of the harness covers a 65 s engine load, a warm start and a warm-up decode.
    Busy % over that span is about none of the things being asked. Only the named NVTX range counts.
"""
import sqlite3, sys

db, rng = sys.argv[1], (sys.argv[2] if len(sys.argv) > 2 else "timed_decode")
c = sqlite3.connect(db)


def tables():
    return {r[0] for r in c.execute("select name from sqlite_master where type='table'")}


T = tables()
sid = None
for t in ("NVTX_EVENTS",):
    if t in T:
        q = c.execute(f"select start, end, text from {t} where text like ?", (f"%{rng}%",)).fetchall()
        if q:
            sid = max(q, key=lambda r: (r[1] or 0) - r[0])
if sid is None:
    print(f"  no NVTX range matching {rng!r}; was the harness run with the marker, and -t nvtx?")
    sys.exit(1)
lo, hi, _ = sid
span = (hi - lo) / 1e9


def merge(rows):
    rows = sorted(rows)
    tot, cs, ce = 0, None, None
    for s, e in rows:
        if cs is None:
            cs, ce = s, e
        elif s <= ce:
            ce = max(ce, e)
        else:
            tot += ce - cs
            cs, ce = s, e
    if cs is not None:
        tot += ce - cs
    return tot


def rows(tab):
    if tab not in T:
        return []
    return list(c.execute(f"select start, end from {tab} where end > ? and start < ?", (lo, hi)))


k = rows("CUPTI_ACTIVITY_KIND_KERNEL")
g = rows("CUPTI_ACTIVITY_KIND_GRAPH_TRACE")
mc = rows("CUPTI_ACTIVITY_KIND_MEMCPY")
busy = merge([(max(s, lo), min(e, hi)) for s, e in k + g]) / 1e9
kern = merge([(max(s, lo), min(e, hi)) for s, e in k]) / 1e9
gph = merge([(max(s, lo), min(e, hi)) for s, e in g]) / 1e9
cpy = merge([(max(s, lo), min(e, hi)) for s, e in mc]) / 1e9
print(f"  range {rng!r}: {span:.2f} s, {len(k)} kernel rows, {len(g)} graph rows")
print(f"  KERNEL only        {kern:6.2f} s  {kern / span * 100:5.1f} %")
print(f"  GRAPH_TRACE only   {gph:6.2f} s  {gph / span * 100:5.1f} %")
print(f"  GPU busy (union)   {busy:6.2f} s  {busy / span * 100:5.1f} %   <- the answer")
print(f"  MEMCPY             {cpy:6.2f} s  {cpy / span * 100:5.1f} %")
print(f"  GPU IDLE           {span - busy:6.2f} s  {(span - busy) / span * 100:5.1f} %")
