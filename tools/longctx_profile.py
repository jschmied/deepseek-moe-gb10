#!/usr/bin/env python3
"""TTFT, NVMe and GPU duty against prompt length -- the baseline a prompt cache would remove.

This engine calls _reset() on every request, so an agent turn re-prefills its whole context each
time. Prefill measured 86-99 tok/s and 30 MB of NVMe per prompt token, which on a 10k-token turn is
~79% of the wall. Before building an extend-only prefix cache it is worth having the curve it has to
beat, measured rather than extrapolated from one point.
"""
import json, os, subprocess, sys, time, urllib.request

BASE = "http://127.0.0.1:8001/v1/chat/completions"
LENS = [int(x) for x in (sys.argv[1].split(",") if len(sys.argv) > 1 else "2000,8000,16000,32000".split(","))]
OUT = int(sys.argv[2]) if len(sys.argv) > 2 else 64
src = open(os.path.expanduser("~/git/deepseek-v41-flash-spark/NOTES.md")).read().split()


def dev_read():
    for l in open("/proc/diskstats"):
        f = l.split()
        if f[2] == "nvme0n1":
            return int(f[5]) * 512
    return 0


print(f"  {'target':>7} {'prompt tok':>11} {'TTFT s':>8} {'tok/s pre':>10} {'NVMe GB':>9} {'MB/tok':>8} {'decode':>7}")
rows = []
for n in LENS:
    words = src * (n // len(src) + 2)
    q = " ".join(words[:int(n * 0.75)])
    req = urllib.request.Request(BASE, method="POST", headers={"Content-Type": "application/json"},
        data=json.dumps({"model": "deepseek-v4.1-flash",
                         "messages": [{"role": "user", "content": "Summarise in one sentence.\n\n" + q}],
                         "temperature": 0.6, "max_tokens": OUT, "stream": True}).encode())
    r0 = dev_read(); t0 = time.time(); ttft = None; k = 0; st = None
    with urllib.request.urlopen(req, timeout=7200) as resp:
        for raw in resp:
            l = raw.decode().strip()
            if not l.startswith("data: "):
                continue
            b = l[6:]
            if b == "[DONE]":
                break
            o = json.loads(b)
            if o.get("x_engine_stats"):
                st = o["x_engine_stats"]
            d = (o.get("choices") or [{}])[0].get("delta") or {}
            if d.get("content"):
                k += 1
                if ttft is None:
                    ttft = time.time() - t0; r_at_ttft = dev_read()
    tot = time.time() - t0
    pt = (st or {}).get("prompt_tokens", n)
    if ttft is None:
        # no content ever arrived -- almost always the prompt exceeding max_seq. Report and carry on
        # rather than crashing the whole sweep on its last arm, which is what the first run did.
        print(f"  {n:>7} {pt:>11}  NO OUTPUT (prompt over max_seq, or server error)", flush=True)
        continue
    pre_gb = (r_at_ttft - r0) / 1e9
    rows.append(dict(target=n, prompt_tokens=pt, ttft=ttft, prefill_tok_s=pt / ttft,
                     prefill_nvme_gb=pre_gb, mb_per_prompt_tok=pre_gb * 1000 / pt,
                     decode_tok_s=k / max(tot - ttft, 1e-9)))
    r = rows[-1]
    print(f"  {n:>7} {pt:>11} {r['ttft']:>8.1f} {r['prefill_tok_s']:>10.1f} "
          f"{r['prefill_nvme_gb']:>9.1f} {r['mb_per_prompt_tok']:>8.1f} {r['decode_tok_s']:>7.2f}", flush=True)
json.dump(rows, open(os.path.expanduser("~/ds41-queue/logs/longctx.json"), "w"), indent=1)
print("\n  a prompt cache removes the TTFT column for every turn after the first in a conversation.")
print("== ALL DONE ==")
