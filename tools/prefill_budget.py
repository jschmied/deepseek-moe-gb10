#!/usr/bin/env python3
"""prefill_budget.py -- where a prefill's wall clock actually goes, from an nsys report.

This is the decomposition that re-ordered the whole 2026-09-14 work queue. Ranking optimisations by
GPU kernel time alone pointed at attention copies worth 2.15 s; this table showed the main thread
blocked 47.6 s of 93.2 and launch overhead at 11.97 s, which is a different night's work.

The main thread is the critical path: its wall time IS the prefill. So classify that thread's time
into (a) inside a CUDA API call, (b) inside a syscall, (c) neither, and then split the blocked part
by what the machine was doing meanwhile -- an NVMe read in flight, a kernel running, or nothing at
all. "Nothing at all" is pure overhead and cannot be blamed on the device.

Usage:  prefill_budget.py report.nsys-rep
"""

from __future__ import annotations

import bisect
import os
import subprocess
import sys


def sqlite_for(rep: str) -> str:
    db = rep.replace(".nsys-rep", ".sqlite")
    if not os.path.exists(db) or os.path.getmtime(db) < os.path.getmtime(rep):
        print(f"  exporting {os.path.basename(rep)} (minutes) ...", flush=True)
        subprocess.run(["nsys", "export", "--type", "sqlite", "--force-overwrite", "true",
                        "-o", db, rep], check=True, stdout=subprocess.DEVNULL)
    return db


def merge(rows):
    m = []
    for a, b in sorted(rows):
        if m and a <= m[-1][1]:
            m[-1][1] = max(m[-1][1], b)
        else:
            m.append([a, b])
    return m


def cover(win, other) -> float:
    st = [a for a, _ in other]
    en = [b for _, b in other]
    tot = 0
    for a, b in win:
        i = bisect.bisect_right(st, a) - 1
        while i < len(st):
            if i >= 0 and st[i] < b:
                tot += max(0, min(b, en[i]) - max(a, st[i]))
            i += 1
            if i >= len(st) or st[i] >= b:
                break
    return tot / 1e9


def main() -> int:
    import sqlite3
    if len(sys.argv) != 2:
        print(__doc__.strip().splitlines()[-1], file=sys.stderr)
        return 2
    c = sqlite3.connect(sqlite_for(sys.argv[1]))
    names = dict(c.execute("select id,value from StringIds"))
    sid = {v: k for k, v in names.items()}
    lo, hi = c.execute("select min(start),max(end) from CUPTI_ACTIVITY_KIND_KERNEL").fetchone()
    span = (hi - lo) / 1e9
    main_tid = c.execute("select globalTid from CUPTI_ACTIVITY_KIND_RUNTIME "
                         "group by 1 order by count(*) desc limit 1").fetchone()[0]

    api = merge(c.execute("select start,end from CUPTI_ACTIVITY_KIND_RUNTIME where globalTid=?", (main_tid,)))
    osr = merge(c.execute("select start,end from OSRT_API where globalTid=?", (main_tid,)))
    in_api = sum(b - a for a, b in api) / 1e9
    in_os = sum(b - a for a, b in osr) / 1e9
    print(f"\n  prefill span {span:.1f} s, main thread {main_tid}\n")
    print(f"  {'inside a CUDA API call':32s} {in_api:7.1f} s ({100 * in_api / span:4.0f} %)")
    print(f"  {'inside a syscall':32s} {in_os:7.1f} s ({100 * in_os / span:4.0f} %)")
    print(f"  {'neither (pure Python)':32s} {span - in_api - in_os:7.1f} s "
          f"({100 * (span - in_api - in_os) / span:4.0f} %)")

    print("\n  top CUDA API by time on the main thread:")
    for nid, n, t in c.execute(
            "select nameId,count(*),sum(end-start) from CUPTI_ACTIVITY_KIND_RUNTIME "
            "where globalTid=? group by 1 order by 3 desc limit 6", (main_tid,)):
        nm = names.get(nid, str(nid))
        print(f"    {nm[:30]:30s} {n:9,} {t / 1e9:7.2f} s {t / n / 1e3:8.1f} us")

    wait_id = sid.get("sem_wait")
    if wait_id is None:
        print("\n  (no sem_wait in this capture)")
        return 0
    wait = merge(c.execute("select start,end from OSRT_API where globalTid=? and nameId=?",
                           (main_tid, wait_id)))
    W = sum(b - a for a, b in wait) / 1e9
    reads = merge(list(c.execute("select start,end from OSRT_API where nameId=?", (sid["preadv64v2"],)))
                  + list(c.execute("select start,end from OSRT_API where nameId=?", (sid["pread64"],))))
    kern = merge(c.execute("select start,end from CUPTI_ACTIVITY_KIND_KERNEL"))
    cp = merge(c.execute("select start,end from CUPTI_ACTIVITY_KIND_MEMCPY"))
    r, k, m = cover(wait, reads), cover(wait, kern), cover(wait, cp)
    anyb = cover(wait, merge([list(x) for x in reads] + [list(x) for x in kern] + [list(x) for x in cp]))
    print(f"\n  main thread blocked in sem_wait: {W:.1f} s ({100 * W / span:.0f} % of the prefill)")
    print(f"    with an NVMe read in flight  {r:7.1f} s ({100 * r / W:4.0f} %)")
    print(f"    with a GPU kernel running    {k:7.1f} s ({100 * k / W:4.0f} %)")
    print(f"    with an H2D copy in flight   {m:7.1f} s ({100 * m / W:4.0f} %)")
    print(f"    with NOTHING at all          {W - anyb:7.1f} s ({100 * (W - anyb) / W:4.0f} %)  <- pure overhead")

    busy = sum(b - a for a, b in kern) / 1e9
    nl = c.execute("select count(*) from CUPTI_ACTIVITY_KIND_KERNEL").fetchone()[0]
    print(f"\n  GPU busy {busy:.1f} s ({100 * busy / span:.0f} % of span) over {nl:,} kernel launches")
    return 0


if __name__ == "__main__":
    sys.exit(main())
