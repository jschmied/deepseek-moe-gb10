#!/usr/bin/env python3
"""Sweep one engine env var against the standard benchmark, restarting the server for each value.

Replaces four near-identical shell loops. It exists because the first one I wrote by hand piped
stderr through grep, so the failing arm printed nothing and the loop marched on as though it had
worked. Here every arm's full output is kept, an arm that does not produce a MEDIAN line is recorded
as FAILED rather than skipped, and the sentinel at the end reflects that.

  env_sweep.py --var DSV41_BLOCK --values 3,5,7,9 --label blockwidth
"""
import argparse, json, os, re, subprocess, sys, time

ap = argparse.ArgumentParser()
ap.add_argument("--repo", default=os.path.expanduser("~/git/deepseek-v41-flash-spark"))
ap.add_argument("--var", required=True)
ap.add_argument("--values", required=True, help="comma separated; the literal word 'unset' clears it")
ap.add_argument("--label", required=True)
ap.add_argument("--runs", type=int, default=3)
ap.add_argument("--osl", type=int, default=512)
ap.add_argument("--outdir", default=os.path.expanduser("~/ds41-queue/logs"))
ap.add_argument("--base", default="http://127.0.0.1:8001")
a = ap.parse_args()

BASE_ENV = {"DSV41_CB3_CACHE": os.path.expanduser("~/dsv41-cb3/experts-cb3-s3.bin"),
            "DSV41_DENSE_FP4": "attn,wo_a", "DSV41_HEAD_FMT": "fp8"}
os.makedirs(a.outdir, exist_ok=True)


def sh(cmd, env=None, timeout=None):
    e = dict(os.environ); e.update(BASE_ENV); e.update(env or {})
    return subprocess.run(cmd, cwd=a.repo, env=e, capture_output=True, text=True, timeout=timeout)


def wait_health(limit=420):
    t0 = time.time()
    while time.time() - t0 < limit:
        r = subprocess.run(["curl", "-sf", "-m", "3", a.base + "/health"], capture_output=True)
        if r.returncode == 0:
            return True
        time.sleep(10)
    return False


rows, failed = [], 0
vals = [v.strip() for v in a.values.split(",")]
print(f"  sweeping {a.var} over {vals}, {a.runs} runs of {a.osl} tokens each", flush=True)
for v in vals:
    sh(["./stop.sh"]); time.sleep(4)
    env = {} if v == "unset" else {a.var: v}
    if v == "unset":
        os.environ.pop(a.var, None)
    sh(["./start.sh", "--no-wait"], env=env)
    if not wait_health():
        print(f"  {a.var}={v}: SERVER DID NOT COME UP", flush=True); failed += 1; continue
    r = sh([os.path.expanduser("~/vllm-venv-main-dflash2/bin/python"), "bench/bench.py",
            "--workload", "code", "--runs", str(a.runs), "--osl", str(a.osl), "--ignore-eos",
            "--base", a.base, "--label", f"{a.label}-{v}"], env=env, timeout=7200)
    out = r.stdout + r.stderr
    open(os.path.join(a.outdir, f"{a.label}-{v}.log"), "w").write(out)
    m = re.search(r"MEDIAN.*?decode=\s*([\d.]+)", out) or re.search(r"decode=([\d.]+) tok/s", out)
    md = re.search(r"MEDIAN\s+ttft=([\d.]+).*?tpot=([\d.]+).*?decode=([\d.]+)", out)
    acc = re.search(r"ENGINE\s+accept_len=([\d.]+)\s+expert_hit=([\d.]+)\s+nvme=([\d.]+)", out)
    if md:
        rows.append((v, float(md.group(1)) / 1000, float(md.group(2)), float(md.group(3)),
                     float(acc.group(1)) if acc else None, float(acc.group(2)) if acc else None,
                     float(acc.group(3)) if acc else None))
        print(f"  {a.var}={v:<6} ttft {rows[-1][1]:6.2f}s  tpot {rows[-1][2]:6.1f}ms  "
              f"decode {rows[-1][3]:5.2f} tok/s  accept {rows[-1][4]}  hit {rows[-1][5]}  "
              f"nvme {rows[-1][6]} GB", flush=True)
    else:
        failed += 1
        print(f"  {a.var}={v:<6} FAILED (rc={r.returncode}, no MEDIAN) -> "
              f"{a.label}-{v}.log", flush=True)
        print("      " + "\n      ".join(out.strip().splitlines()[-4:]), flush=True)

json.dump({"var": a.var, "rows": rows, "failed": failed},
          open(os.path.join(a.outdir, f"{a.label}.json"), "w"), indent=1)
# leave the box on the reference configuration whatever happened
sh(["./stop.sh"]); time.sleep(4)
os.environ.pop(a.var, None)
sh(["./start.sh", "--no-wait"]); wait_health()
print(f"\n  {len(rows)} arms measured, {failed} failed")
print("== ALL DONE ==" if failed == 0 else "== VOID ==   some arms failed; see the logs")
