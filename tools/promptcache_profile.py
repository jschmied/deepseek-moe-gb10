#!/usr/bin/env python3
"""What the extend-only prompt cache is worth at real context.

The first measurement (1.60x) came from a 3,416-token turn, which is the design's WORST case: the
resume point is a prefill-chunk boundary at `lcp - window_size` rounded down, so a short turn throws
away most of what it could have reused. That the win grows with context is arithmetic, not a
measurement, until this runs.

Two turns per length. Turn 1 is a cold request. Turn 2 sends turn 1's prompt + turn 1's reply + a
short follow-up, which is exactly the agent shape. Each length is measured twice: once with the
server's cache live (warm), once after a cache-busting request that diverges at position 0 (cold).

  python tools/promptcache_profile.py 4000,8000,16000
"""
import json
import os
import sys
import time
import urllib.request

BASE = "http://127.0.0.1:8001/v1/chat/completions"
LENS = [int(x) for x in (sys.argv[1].split(",") if len(sys.argv) > 1 else "4000,8000".split(","))]
src = open(os.path.expanduser("~/git/deepseek-v41-flash-spark/NOTES.md")).read().split()


def ask(messages, max_tokens):
    req = urllib.request.Request(
        BASE, method="POST", headers={"Content-Type": "application/json"},
        data=json.dumps({"model": "deepseek-v4.1-flash", "messages": messages,
                         "temperature": 0.0, "max_tokens": max_tokens}).encode())
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=7200) as r:
        o = json.loads(r.read())
    st = o.get("x_engine_stats", {}) or {}
    return o["choices"][0]["message"]["content"] or "", st, time.time() - t0


print(f"  {'target':>7} {'turn':>6} {'prompt tok':>11} {'reused':>8} {'prefilled':>10} "
      f"{'prefill s':>10} {'nvme GB':>8}", flush=True)
rows = []
for n in LENS:
    words = src * (n // len(src) + 2)
    doc = " ".join(words[: int(n * 0.75)])
    m1 = [{"role": "user", "content": "Here is a document.\n\n" + doc +
           "\n\nName one topic it covers, in one short sentence."}]
    for arm in ("cold", "warm"):
        if arm == "cold":
            # bust the cache: a prompt that diverges at position 0, so nothing can be reused
            ask([{"role": "user", "content": "Say OK."}], 4)
        r1, s1, _ = ask(m1, 40)
        print(f"  {n:>7} {'1/'+arm:>6} {s1.get('prompt_tokens'):>11} "
              f"{s1.get('prompt_cache_reused', 0):>8} {s1.get('prompt_tokens_prefilled', '?'):>10} "
              f"{s1.get('prefill_s'):>10} {s1.get('nvme_gb'):>8}", flush=True)
        m2 = m1 + [{"role": "assistant", "content": r1},
                   {"role": "user", "content": "Now name a second topic, in one short sentence."}]
        r2, s2, _ = ask(m2, 40)
        print(f"  {n:>7} {'2/'+arm:>6} {s2.get('prompt_tokens'):>11} "
              f"{s2.get('prompt_cache_reused', 0):>8} {s2.get('prompt_tokens_prefilled', '?'):>10} "
              f"{s2.get('prefill_s'):>10} {s2.get('nvme_gb'):>8}", flush=True)
        rows.append({"target": n, "arm": arm, "turn1": s1, "turn2": s2})
json.dump(rows, open(os.path.expanduser("~/ds41-queue/logs/promptcache.json"), "w"), indent=1)

print("\n  speedup on turn 2, warm against cold:", flush=True)
for n in LENS:
    w = next(r for r in rows if r["target"] == n and r["arm"] == "warm")["turn2"]
    c = next(r for r in rows if r["target"] == n and r["arm"] == "cold")["turn2"]
    if w.get("prefill_s") and c.get("prefill_s"):
        print(f"    {n:>7}: {c['prefill_s']:.1f}s -> {w['prefill_s']:.1f}s  "
              f"({c['prefill_s']/max(w['prefill_s'],1e-9):.2f}x), reused "
              f"{w.get('prompt_cache_reused',0)} of {w.get('prompt_tokens')}", flush=True)
print("== ALL DONE ==", flush=True)
