#!/usr/bin/env python3
"""nsys_early_submit.py -- read the three EARLY_SUBMIT arms and decide the FFN-slowdown mechanism.

The question was "does memory bandwidth rise as FFN time rises". It cannot be asked on this box:
the gb20b GPU metric set has no DRAM counter, --soc-metrics is unsupported here, nvidia-smi dmon's
mem% is stubbed on this iGPU and ncu is not installed. What IS available separates the hypotheses
on a different axis, measured INSIDE the selected kernels' time windows:

  SM Issue % down, Compute Warps in Flight flat, kernel duration up
      -> warps resident but stalled: a memory-system stall (L2 or interference).
  SMs Active down with gaps, kernel duration flat
      -> launch/scheduling contention, not memory.
  Copy Engine Active % at its ceiling while the FFN inflates
      -> the copy engines are the constraint.
  GPC clock down
      -> neither: the part is throttling, and every other number is downstream of that.

Usage:
    nsys_early_submit.py --list                      # top kernels per arm, to pick the regex
    nsys_early_submit.py --kernel-re 'moe|grouped'   # the comparison
"""

from __future__ import annotations

import argparse
import bisect
import os
import sqlite3
import subprocess
import sys

ARMS = ("off", "chunk0", "all")
LOGS = os.path.expanduser("~/ds41-queue/logs")
# metricName -> short label. Throughput % metrics are plain 0..100 integers.
WANT = {
    "GPC Clock Frequency [MHz]": "clk",
    "SMs Active [Throughput %]": "sm_act",
    "SM Issue [Throughput %]": "sm_issue",
    "Tensor Active [Throughput %]": "tensor",
    "Compute Warps in Flight [Throughput %]": "warps",
    "Sync Copy Engine Active [Throughput %]": "ce_sync",
    "Async Copy Engine Active 0 [Throughput %]": "ce_async",
}


def sqlite_for(rep: str) -> str:
    db = rep.replace(".nsys-rep", ".sqlite")
    if not os.path.exists(db) or os.path.getmtime(db) < os.path.getmtime(rep):
        print(f"  exporting {os.path.basename(rep)} -> sqlite (minutes) ...", flush=True)
        subprocess.run(["nsys", "export", "--type", "sqlite", "--force-overwrite", "true",
                        "-o", db, rep], check=True, stdout=subprocess.DEVNULL)
    return db


def clock_fix(v: int) -> float:
    """Clocks are an unsigned Hz value stored in a signed 32-bit column; % metrics are plain.

    Verified on a toy capture: the raw -1768172296 wraps to 2526795000 Hz = 2526.8 MHz.
    """
    if v < 0:
        v += 1 << 32
    return v / 1e6


def kernels(c: sqlite3.Connection):
    return list(c.execute(
        "select s.value, count(*), sum(k.end-k.start) "
        "from CUPTI_ACTIVITY_KIND_KERNEL k join StringIds s on s.id=k.shortName "
        "group by 1 order by 3 desc"))


def intervals(c: sqlite3.Connection, pattern: str):
    rows = list(c.execute(
        "select k.start, k.end from CUPTI_ACTIVITY_KIND_KERNEL k join StringIds s on s.id=k.shortName "
        "where s.value regexp ? order by k.start", (pattern,)))
    # merge overlapping windows (several streams run the same kernel family concurrently)
    merged = []
    for a, b in rows:
        if merged and a <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    return rows, merged


def metrics_in(c: sqlite3.Connection, merged, names: dict):
    starts = [a for a, _ in merged]
    ends = [b for _, b in merged]
    out = {}
    for mid, label in names.items():
        tot = n = 0.0
        for ts, v in c.execute("select timestamp, value from GPU_METRICS where metricId=?", (mid,)):
            i = bisect.bisect_right(starts, ts) - 1
            if i >= 0 and ts <= ends[i]:
                tot += clock_fix(v) if label == "clk" else v
                n += 1
        out[label] = (tot / n if n else float("nan"), int(n))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true", help="print the top kernels per arm and exit")
    ap.add_argument("--kernel-re", default=None, help="SQLite REGEXP over the kernel short name")
    ap.add_argument("--top", type=int, default=15)
    a = ap.parse_args()

    reps = [(m, f"{LOGS}/nsys-early-{m}.nsys-rep") for m in ARMS]
    reps = [(m, r) for m, r in reps if os.path.exists(r)]
    if not reps:
        print(f"no reports in {LOGS}", file=sys.stderr)
        return 1

    if a.list:
        for m, rep in reps:
            c = sqlite3.connect(sqlite_for(rep))
            print(f"\n=== {m} ===")
            print(f"  {'kernel':50s} {'count':>8s} {'total ms':>10s} {'mean us':>9s}")
            for name, cnt, tot in kernels(c)[:a.top]:
                print(f"  {name[:50]:50s} {cnt:8d} {tot/1e6:10.1f} {tot/cnt/1e3:9.1f}")
        return 0

    if not a.kernel_re:
        print("--kernel-re is required (run --list first)", file=sys.stderr)
        return 2

    import re as _re
    rows = []
    for m, rep in reps:
        c = sqlite3.connect(sqlite_for(rep))
        c.create_function("regexp", 2, lambda p, s: 1 if s and _re.search(p, s) else 0)
        ids = {mid: WANT[nm] for mid, nm in c.execute(
            "select metricId, metricName from TARGET_INFO_GPU_METRICS") if nm in WANT}
        ks, merged = intervals(c, a.kernel_re)
        if not ks:
            print(f"{m}: no kernel matched {a.kernel_re!r}", file=sys.stderr)
            continue
        busy = sum(b - x for x, b in merged)
        rows.append((m, len(ks), sum(b - x for x, b in ks) / len(ks) / 1e3, busy / 1e6,
                     metrics_in(c, merged, ids)))

    labels = ["clk", "sm_act", "sm_issue", "tensor", "warps", "ce_sync", "ce_async"]
    print(f"\n  inside kernels matching {a.kernel_re!r}\n")
    print(f"  {'arm':8s} {'kernels':>8s} {'mean us':>9s} {'busy ms':>9s} " +
          " ".join(f"{l:>9s}" for l in labels))
    for m, n, mean_us, busy_ms, met in rows:
        print(f"  {m:8s} {n:8d} {mean_us:9.1f} {busy_ms:9.1f} " +
              " ".join(f"{met[l][0]:9.1f}" for l in labels))
    print("\n  clk MHz; every other column is a throughput %.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
