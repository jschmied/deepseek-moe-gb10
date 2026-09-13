#!/usr/bin/env python3
"""Convert expert_trace.py's per-layer output into the npz the residency simulators read.

expert_trace writes one file per layer, each covering the whole corpus as a flat token
stream.  The simulators want one entry per *request*: `{id}__routed [tokens, layers, topk]`
and `{id}__meta [n_prompt, n_gen]`.  Sequence boundaries come from meta.json (seqs are
written in corpus order); the prefill/decode split is the `<|Assistant|>` token, id 128804,
which is where a served request would stop prefilling and start decoding.
"""
import argparse, json, os, sys
import numpy as np

ASSISTANT = 128804

ap = argparse.ArgumentParser()
ap.add_argument("--trace-dir", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--min-tokens", type=int, default=64, help="skip sequences shorter than this")
a = ap.parse_args()

meta = json.load(open(os.path.join(a.trace_dir, "meta.json")))
tdir = os.path.join(a.trace_dir, "trace")
layers = sorted(int(f[5:-4]) for f in os.listdir(tdir) if f.startswith("layer") and f.endswith(".npz"))
if layers != list(range(len(layers))):
    sys.exit(f"layers are not a prefix run: {layers}")
print(f"  {len(layers)} layers, {meta['n_tokens']} tokens, {meta['n_seqs']} sequences")

idx = np.stack([np.load(os.path.join(tdir, f"layer{l}.npz"))["indices"] for l in layers], axis=1)
tok = np.load(os.path.join(tdir, "layer0.npz"))["token"]
print(f"  routed {idx.shape} {idx.dtype}   experts seen {int(idx.max())+1}")

out, off, skipped = {}, 0, 0
for s in meta["seqs"]:
    n = s["n"]; lo, hi = off, off + n; off = hi
    if n < a.min_tokens:
        skipped += 1
        continue
    t = tok[lo:hi]
    hit = np.nonzero(t == ASSISTANT)[0]
    npr = int(hit[0]) + 1 if len(hit) else n // 2       # fallback: half
    if npr >= n - 8:                                     # nothing to decode
        skipped += 1
        continue
    name = s["id"].replace("__", "_")
    out[f"{name}__routed"] = idx[lo:hi].astype(np.int16)
    out[f"{name}__meta"] = np.array([npr, n - npr], dtype=np.int64)

np.savez_compressed(a.out, **out)
kept = len(out) // 2
pre = [int(out[k][0]) for k in out if k.endswith("__meta")]
gen = [int(out[k][1]) for k in out if k.endswith("__meta")]
print(f"  wrote {kept} requests to {a.out} ({os.path.getsize(a.out)/1e6:.1f} MB), skipped {skipped}")
print(f"  prompt tokens {min(pre)}-{max(pre)} (median {int(np.median(pre))}), "
      f"decode tokens {min(gen)}-{max(gen)} (median {int(np.median(gen))})")
print("== ALL DONE ==")
