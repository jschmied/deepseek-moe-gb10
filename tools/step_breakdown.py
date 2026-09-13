#!/usr/bin/env python3
"""Where does a decode step actually go? From the engine's own counters, not a profiler.

Two plans have now been ranked on a stored 63-of-117 ms CB3 split taken on an all-resident config.
Ours is 68% idle on NVMe, so the shares differ and the ranking follows the shares. v41_engine.py
already exports every phase timer on x_engine_stats; this just reads them and prints the split.
"""
import json, urllib.request

BASE = "http://127.0.0.1:8001/v1/chat/completions"
KEYS = ["attn_s", "moe_s", "kernel_s", "route_s", "load_wait_s", "lease_s", "h2d_s",
        "nvme_read_s", "engram_s", "engram_read_s", "resolve_s"]


def run(prompt, n):
    req = urllib.request.Request(BASE, method="POST", headers={"Content-Type": "application/json"},
        data=json.dumps({"model": "deepseek-v4.1-flash", "messages": [{"role": "user", "content": prompt}],
                         "temperature": 0.6, "max_tokens": n, "stream": True}).encode())
    st = None; k = 0
    with urllib.request.urlopen(req, timeout=3600) as r:
        for raw in r:
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
    return st, k


for tag, p, n in (("short", "Write a Python LRU cache with tests.", 400),
                  ("long", "Explain transformer attention in detail, with worked arithmetic.", 400)):
    st, k = run(p, n)
    if not st:
        print(f"  {tag}: no x_engine_stats"); continue
    acc = st.get("accept_len_mean") or 1.0
    steps = max(k / acc, 1)
    print(f"\n  {tag}: {k} tokens, accept {acc}, ~{steps:.0f} steps, "
          f"hit {st.get('expert_hit_rate')}, nvme {st.get('nvme_gb')} GB")
    print(f"  {'counter':<16} {'total s':>9} {'ms/step':>9}")
    for key in KEYS:
        if key in st:
            print(f"  {key:<16} {st[key]:>9.2f} {1000*st[key]/steps:>9.2f}")
print("== ALL DONE ==")
