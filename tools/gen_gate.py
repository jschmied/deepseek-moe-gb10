#!/usr/bin/env python3
"""Free-generation degeneration gate.

Teacher-forced NLL cannot see the failure mode that matters for expert substitution: the model
keeps assigning reasonable probability to the next token of a *given* text while its own free
generation collapses into repetition. That is exactly how `PRUNE_KEEP=0.40` failed
(notes/native-cb3-expert-cache.md: distinct-token ratio 0.03, `<!DOCTYPE>` repeated to the cap)
while its NLL would have looked unremarkable.

So this generates freely and scores the output structurally:

  distinct ratio   unique whitespace tokens / total -- collapses toward 0 under repetition
  max line repeat  longest run of identical consecutive lines
  finish           "stop" (the model chose to end) or "length" (hit the cap)

The pass line used for CB3 was: distinct ratio > 0.25 on code, > 0.4 on prose, no line repeated
more than twice, and at least some prompts terminating naturally.

  python tools/gen_gate.py [--max-tokens 600] [--label NAME] [--out results.json]
"""
import argparse
import json
import os
import sys
import time
import urllib.request

BASE = "http://127.0.0.1:8001/v1/chat/completions"

PROMPTS = [
    ("html", "Write a complete single-file HTML page for a tic-tac-toe game. Output only the HTML."),
    ("python", "Write a Python LRU cache class with unit tests. Output only code."),
    ("essay", "Write a short essay on why sailing ships lost to steam, in English."),
    ("german", "Schreibe eine kurze Geschichte ueber einen Leuchtturmwaerter im Winter."),
    ("reasoning", "A farmer has 17 sheep. All but 9 run away. How many are left? Explain briefly."),
]


def gen(prompt, max_tokens, timeout=1800):
    req = urllib.request.Request(
        BASE, method="POST", headers={"Content-Type": "application/json"},
        data=json.dumps({"model": "deepseek-v4.1-flash",
                         "messages": [{"role": "user", "content": prompt}],
                         "temperature": 0.6, "top_p": 0.95,
                         "max_tokens": max_tokens}).encode())
    with urllib.request.urlopen(req, timeout=timeout) as r:
        o = json.loads(r.read())
    c = o["choices"][0]
    return c["message"]["content"] or "", c.get("finish_reason", "?"), o.get("x_engine_stats", {})


def score(text):
    w = text.split()
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    run = best = 1
    for i in range(1, len(lines)):
        run = run + 1 if lines[i] == lines[i - 1] else 1
        best = max(best, run)
    return {"words": len(w),
            "distinct_ratio": round(len(set(w)) / max(len(w), 1), 3),
            "max_line_repeat": best if lines else 0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-tokens", type=int, default=600)
    ap.add_argument("--label", default="run")
    ap.add_argument("--out")
    a = ap.parse_args()

    print(f"  {'prompt':>10} {'words':>6} {'distinct':>9} {'line rep':>9} {'finish':>8}", flush=True)
    rows, failed = [], 0
    for name, p in PROMPTS:
        t0 = time.time()
        try:
            text, fin, st = gen(p, a.max_tokens)
        except Exception as e:                              # a dead server is not a quality verdict
            print(f"  {name:>10}  REQUEST FAILED: {type(e).__name__}: {e}", flush=True)
            failed += 1
            continue
        s = score(text)
        s.update(prompt=name, finish=fin, label=a.label, seconds=round(time.time() - t0, 1),
                 tok_s=st.get("decode_tok_s"))
        rows.append(s)
        # the pass line from the CB3 gate; code is allowed a lower ratio than prose
        floor = 0.25 if name in ("html", "python") else 0.40
        bad = s["distinct_ratio"] < floor or s["max_line_repeat"] > 2
        print(f"  {name:>10} {s['words']:>6} {s['distinct_ratio']:>9.3f} {s['max_line_repeat']:>9} "
              f"{fin:>8}  {'<-- DEGENERATE' if bad else ''}", flush=True)
        if bad:
            failed += 1
    n_stop = sum(1 for r in rows if r["finish"] == "stop")
    print(f"\n  {len(rows) - failed}/{len(PROMPTS)} pass, {n_stop} terminated naturally", flush=True)
    if a.out:
        json.dump({"label": a.label, "rows": rows, "failed": failed}, open(a.out, "w"), indent=1)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
