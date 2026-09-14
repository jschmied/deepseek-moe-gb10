#!/usr/bin/env python3
"""Rare-identifier corruption gate — the failure mode our degeneration gate cannot see.

0xBakeer's v0.5.0 RESULTS reports that frequency-ranked pruning at keep 0.40 "corrupts rare tokens
at subword boundaries — `clearTimeout` written `cleartimeout`, `OSError` as `oenerror`". Note what
that does to `gen_gate.py`'s statistics: **nothing**. A corrupted identifier is still a distinct
token, still does not repeat a line, and still lets the model finish. Our gate scores it as a pass.

So this asks the model to emit specific rare identifiers and checks they come back byte-exact. It is
a much narrower instrument than a benchmark and a much sharper one than a repetition detector.

  python tools/token_integrity.py [--label NAME] [--out r.json]
"""
import argparse
import json
import sys
import urllib.request

BASE = "http://127.0.0.1:8001/v1/chat/completions"

# (prompt, identifiers that MUST appear byte-exact). Chosen to sit at subword boundaries the way
# the reported corruptions do: camelCase, embedded capitals, underscores, dotted paths.
CASES = [
    ("Write a JavaScript function that uses clearTimeout, setInterval, XMLHttpRequest and "
     "requestAnimationFrame. Output only code.",
     ["clearTimeout", "setInterval", "XMLHttpRequest", "requestAnimationFrame"]),
    ("Write a Python function that catches OSError and NotImplementedError, uses "
     "itertools.groupby and collections.OrderedDict. Output only code.",
     ["OSError", "NotImplementedError", "itertools.groupby", "collections.OrderedDict"]),
    ("Write a Python snippet using numpy.ascontiguousarray, torch.nn.functional.softmax and "
     "os.posix_fadvise. Output only code.",
     ["numpy.ascontiguousarray", "torch.nn.functional.softmax", "os.posix_fadvise"]),
    ("Write a C snippet using pthread_mutex_trylock, __builtin_expect and O_DIRECT. Output only code.",
     ["pthread_mutex_trylock", "__builtin_expect", "O_DIRECT"]),
]


def gen(prompt, max_tokens=400):
    req = urllib.request.Request(
        BASE, method="POST", headers={"Content-Type": "application/json"},
        data=json.dumps({"model": "deepseek-v4.1-flash",
                         "messages": [{"role": "user", "content": prompt}],
                         "temperature": 0.0, "max_tokens": max_tokens}).encode())
    with urllib.request.urlopen(req, timeout=1800) as r:
        o = json.loads(r.read())
    return o["choices"][0]["message"]["content"] or ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", default="run")
    ap.add_argument("--out")
    a = ap.parse_args()
    rows, miss_tot, want_tot = [], 0, 0
    print(f"  {'case':>6} {'wanted':>7} {'exact':>6} missing", flush=True)
    for i, (prompt, idents) in enumerate(CASES):
        try:
            text = gen(prompt)
        except Exception as e:
            print(f"  {i:>6}  REQUEST FAILED: {type(e).__name__}: {e}", flush=True)
            return 2
        missing = [w for w in idents if w not in text]
        # a near-miss is the interesting case: the identifier is there but case- or
        # separator-mangled, which is exactly the reported corruption
        near = []
        low = text.lower().replace("_", "").replace("-", "").replace(".", "")
        for w in missing:
            if w.lower().replace("_", "").replace("-", "").replace(".", "") in low:
                near.append(w)
        rows.append({"case": i, "wanted": idents, "missing": missing, "mangled": near})
        miss_tot += len(missing); want_tot += len(idents)
        print(f"  {i:>6} {len(idents):>7} {len(idents)-len(missing):>6} "
              f"{', '.join(missing) if missing else '-'}"
              f"{'   MANGLED: ' + ', '.join(near) if near else ''}", flush=True)
    print(f"\n  {want_tot - miss_tot}/{want_tot} identifiers byte-exact, label {a.label}", flush=True)
    if a.out:
        json.dump({"label": a.label, "rows": rows, "exact": want_tot - miss_tot, "wanted": want_tot},
                  open(a.out, "w"), indent=1)
    return 1 if miss_tot else 0


if __name__ == "__main__":
    sys.exit(main())
