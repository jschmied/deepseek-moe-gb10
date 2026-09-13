#!/usr/bin/env python3
"""Does Engram actually reach the forward? The inverse of a degeneration gate.

A token-exact greedy gate cannot detect a feature being silently disabled: the field demonstrated
exactly that by zeroing the Engram rows and watching their gate still return PASS. We serve Engram
in a way nobody else does -- 203 GB on NVMe, 48 rows per token by byte range, hashed from token ids --
so a wrong offset or a stale hash state would be invisible to every quality check we run, and would
show up only as a small speedup.

So: run identical greedy prompts against the server, compare with the hashes taken while
DSV41_ENGRAM_ABLATE=1. The output MUST change. If it does not, either the rows are not reaching the
forward or the model does not use them, and both are things we need to know.

Usage: engram_ablation.py <hashfile-out> [--compare <hashfile-in>]
"""
import argparse, hashlib, json, urllib.request

BASE = "http://127.0.0.1:8001/v1/chat/completions"
PROMPTS = [
    ("code", "Write a Python function to compute the nth Fibonacci number iteratively. Code only."),
    ("prose", "In exactly three sentences, explain why the sky is blue."),
    ("fact", "List the first eight prime numbers, comma separated, nothing else."),
    ("quote", "Complete this exactly: 'It was the best of times, it was'"),
    ("recall", "The capital of Burkina Faso is"),
]

ap = argparse.ArgumentParser()
ap.add_argument("out")
ap.add_argument("--compare", default=None)
ap.add_argument("--tokens", type=int, default=120)
a = ap.parse_args()


def go(p):
    req = urllib.request.Request(BASE, method="POST", headers={"Content-Type": "application/json"},
        data=json.dumps({"model": "deepseek-v4.1-flash", "messages": [{"role": "user", "content": p}],
                         "temperature": 0, "top_p": 1.0, "max_tokens": a.tokens}).encode())
    with urllib.request.urlopen(req, timeout=1800) as r:
        return json.load(r)["choices"][0]["message"]["content"]


cur = {}
for tag, p in PROMPTS:
    txt = go(p)
    cur[tag] = hashlib.sha256(txt.encode()).hexdigest()[:12]
    print(f"  {tag:<7} {cur[tag]}  {txt[:60].replace(chr(10),' ')!r}")
json.dump(cur, open(a.out, "w"))

if a.compare:
    ref = json.load(open(a.compare))
    changed = [k for k in cur if ref.get(k) != cur[k]]
    print(f"\n  {len(changed)}/{len(cur)} prompts changed when Engram was ablated: {changed}")
    for k in cur:
        print(f"    {k:<7} ref {ref.get(k)}  now {cur[k]}  {'CHANGED' if ref.get(k) != cur[k] else 'IDENTICAL'}")
    if not changed:
        print("\n  == VOID ==  Engram made NO difference to any prompt. Either the rows are not")
        print("  reaching the forward, or this corpus does not exercise them. Do not treat any")
        print("  quality result on this build as having covered Engram.")
        raise SystemExit(0)
print("== ALL DONE ==")
