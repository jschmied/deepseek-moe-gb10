#!/usr/bin/env python3
"""Where does a decode step actually go? From the engine's own counters, not a profiler.

FIRST VERSION WAS WRONG and this is the fix. The counters on x_engine_stats are cumulative over the
WHOLE request -- prefill included -- so dividing them by the decode step count attributes thousands
of prefill expert loads to decode. It produced nvme_read_s of 131 s inside a request that took tens,
and a negative kernel_s.

So: same prompt twice, different max_tokens, and difference the counters. Prefill is identical in
both, so the delta is decode alone.

Two further traps this exposed, both worth knowing before reading any of these numbers:
  * `read_s`, `h2d_s` and `lease_s` are accumulated INSIDE the io-thread pool (experts.py:233, 291,
    309), so they are thread-seconds summed over 12 workers, not wall. Compare them to each other,
    never to a wall clock.
  * `route_s`, `load_s`, `resolve_s` (experts.py:460-465) and `moe_s`, `attn_s` (model.py:512, 524)
    are main-thread and ARE wall.
  * `kernel_s = moe_s - resolve_s` is only meaningful in the pruned all-resident configuration. On
    the streaming path resolve_s is large and the subtraction goes negative.
"""
import json, urllib.request

BASE = "http://127.0.0.1:8001/v1/chat/completions"
# decode_only carries the RAW counter names (ZERO_STATS), not the exported aliases:
# load_wait_s is load_s there, nvme_read_s is read_s. Asking for the aliases silently
# dropped the two biggest I/O terms from the first decode-only run.
KEYS = ["attn_s", "moe_s", "sync_s", "route_s", "load_s", "lease_s", "h2d_s",
        "read_s", "engram_s", "resolve_s"]


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


WALL = {"route_s", "load_s", "resolve_s", "moe_s", "attn_s", "engram_s", "sync_s"}
# A task the model does not finish in 130 tokens. Generation length is content-limited, not
# max_tokens-limited, and a short generation right after a restart misses ~3x more per step than
# the steady state -- which makes its SHARES fine and its LEVELS unrepresentative.
PROMPT = ("Write a complete Python module implementing a B-tree with insert, delete, search and "
          "range scan, with full docstrings, type hints, and a test suite covering at least eight "
          "cases. Output only code.")
import sys
st, k = run(PROMPT, int(sys.argv[1]) if len(sys.argv) > 1 else 512)
d = (st or {}).get("decode_only")
if not d:
    print("  no decode_only in x_engine_stats -- the engine predates the prefill snapshot"); raise SystemExit(1)
acc = st.get("accept_len_mean") or 1.0
steps = max(k / acc, 1)
print(f"\n  {k} tokens, accept {acc}, ~{steps:.0f} decode steps, hit {st.get('expert_hit_rate')}, "
      f"decode {st.get('decode_tok_s')} tok/s over {st.get('decode_s')} s")
print(f"  counters below are DECODE ONLY, from the engine's prefill snapshot.\n")
print(f"  {'counter':<16} {'total s':>9} {'ms/step':>9}  kind")
for key in KEYS:
    if key in d:
        kind = "wall" if key in WALL else "thread-seconds (12 workers)"
        print(f"  {key:<16} {d[key]:>9.2f} {1000*d[key]/steps:>9.2f}  {kind}")
if "bytes_read" in d:
    print(f"\n  decode NVMe {d['bytes_read']/1e9:.1f} GB over {k} tokens "
          f"= {d['bytes_read']/1e6/k:.0f} MB/token")
mps = d.get("misses", 0) / steps
print(f"  misses {d.get('misses')} = {mps:.0f}/step, prefill_misses {d.get('prefill_misses')} "
      f"(must be 0 -- if not, the server is running pre-605f09d code)")
# The benchmark's steady state is ~133 misses/step. Well above that means the arena has not caught
# up with THIS request's experts, so the shares below are usable and the levels are not.
if mps > 200:
    print(f"  WARNING: {mps:.0f} misses/step against the benchmark's ~133. Levels are not the "
          f"serving levels; read the shares only.")
dw = st.get("decode_s") or 0
if dw:
    print(f"\n  share of decode wall ({dw:.1f} s): resolve {100*d.get('resolve_s',0)/dw:.0f} %, "
          f"of which load-wait {100*d.get('load_s',0)/max(d.get('resolve_s',1e-9),1e-9):.0f} % "
          f"and route bookkeeping {100*d.get('route_s',0)/max(d.get('resolve_s',1e-9),1e-9):.0f} %")
    print(f"  attn_s/moe_s read 0 on the fast path: the CUDA-graph decode never updates m.stats, "
          f"so this is NOT evidence that compute is free.")
print("== ALL DONE ==")
