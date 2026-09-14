#!/usr/bin/env python3
"""Can the UE8M0 group scales be stored in 3 bits + a per-row base, losslessly, on every layer?

A CB3 expert slot is 14,454,784 B of which 1,105,920 B (7.6%) is UE8M0 group exponents -- one byte
per group of 32 weights. If the exponents inside one output row span a small range, the row can carry
a 1-byte base and 3 bits per group instead, which is exactly lossless when the range is <= 7.

A 3-layer sample said range <= 7 on 100.0000% of rows. This checks all 40, because the codec has to
be lossless everywhere or it needs a per-row escape. It reads ONLY the scale region of each layer
shard (one contiguous span of ~425 MB near the front of the file) straight off the backup server, so
it never needs the 7.4 GB shard local and never needs the checkpoint resident.

Emits, per layer: distinct exponent values, order-0 entropy, and the row-range distribution.
"""
import argparse, io, json, os, struct, subprocess, sys
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--host", default=os.environ.get("DSV41_SHARD_HOST", ""),
                help="user@host of the shard store; set DSV41_SHARD_HOST rather than hard-coding one")
ap.add_argument("--key", default="/home/jschmied/.ssh/id_ed25519")
ap.add_argument("--dir", default="/mnt/bulk/hf/deepseek-ai--DeepSeek-V4.1-Flash")
ap.add_argument("--layers", default="0-39")
ap.add_argument("--experts", type=int, default=0, help="0 = all 384")
a = ap.parse_args()
lo, hi = (int(x) for x in a.layers.split("-"))

SSH = ["ssh", "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=no", "-i", a.key, a.host]


def remote_range(path, off, n):
    """Read n bytes at off from a remote file, without copying the file. dd rather than a remote
    python one-liner: embedded newlines do not survive the remote shell."""
    cmd = ["dd", f"if={path}", "bs=4M", f"skip={off}", f"count={n}",
           "iflag=skip_bytes,count_bytes", "status=none"]
    p = subprocess.run(SSH + cmd, stdout=subprocess.PIPE, check=True)
    if len(p.stdout) != n:
        raise IOError(f"short remote read: {len(p.stdout)} of {n} at {off}")
    return p.stdout


def header(path):
    raw = remote_range(path, 0, 8)
    n = struct.unpack("<Q", raw)[0]
    return json.loads(remote_range(path, 8, n)), 8 + n


def row_ranges(buf, rows, groups):
    x = np.frombuffer(buf, dtype=np.uint8).reshape(rows, groups).astype(np.int16)
    return x.max(1) - x.min(1), x


print(f"  layers {lo}-{hi}, {'all 384' if not a.experts else a.experts} experts each, scale region only")
print(f"  {'layer':>5} {'MB read':>8} {'rows':>10} {'distinct':>9} {'entropy':>8} "
      f"{'max range':>10} {'<=3':>9} {'<=7':>9}")
worst = 0
tot_rows = tot_le7 = 0
for L in range(lo, hi + 1):
    path = f"{a.dir}/model-{L+3:05d}-of-00048.safetensors"
    hdr, base = header(path)
    ents = []
    for k, v in hdr.items():
        if k == "__metadata__" or ".ffn.experts." not in k or not k.endswith(".scale"):
            continue
        e = int(k.split(".experts.")[1].split(".")[0])
        if a.experts and e >= a.experts:
            continue
        s, t = v["data_offsets"]
        ents.append((base + s, base + t, tuple(v["shape"])))
    ents.sort()
    span_lo, span_hi = ents[0][0], ents[-1][1]
    buf = remote_range(path, span_lo, span_hi - span_lo)

    vals = np.zeros(256, dtype=np.int64)
    rng_hist = np.zeros(64, dtype=np.int64)
    nrows = 0
    for s, t, shape in ents:
        r, g = shape
        rr, x = row_ranges(buf[s - span_lo:t - span_lo], r, g)
        vals += np.bincount(x.ravel(), minlength=256)
        rng_hist += np.bincount(np.clip(rr, 0, 63), minlength=64)
        nrows += r
    p = vals[vals > 0] / vals.sum()
    ent = float(-(p * np.log2(p)).sum())
    mx = int(np.nonzero(rng_hist)[0].max())
    le3 = 100.0 * rng_hist[:4].sum() / nrows
    le7 = 100.0 * rng_hist[:8].sum() / nrows
    worst = max(worst, mx); tot_rows += nrows; tot_le7 += rng_hist[:8].sum()
    print(f"  {L:>5} {(span_hi-span_lo)/1e6:8.1f} {nrows:>10,} {int((vals>0).sum()):>9} "
          f"{ent:8.3f} {mx:>10} {le3:8.4f}% {le7:8.4f}%")

print(f"\n  worst intra-row range over all layers: {worst}")
print(f"  rows with range <= 7 (3 bits + base, lossless): {100.0*tot_le7/tot_rows:.6f}% of {tot_rows:,}")
print("== ALL DONE ==")
