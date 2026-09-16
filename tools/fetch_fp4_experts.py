#!/usr/bin/env python3
"""Fetch a FEW routed experts' original FP4 weights out of the 428 GB checkpoint on the backup box.

The lean checkpoint on the GB10 has no routed-expert safetensors -- the CB3 cache replaced them --
so any quantizer study that starts from CB3 is measuring a requantization of a requantization. The
originals are on the backup server, but the shard holding one layer is 7.4 GB and local disk has
~134 GB free, so pulling shards per layer does not scale.

safetensors puts an 8-byte little-endian header length, then a JSON header giving every tensor's
[start, end) within the data blob. So each tensor is a byte range and can be fetched on its own:
one expert's w1+w3 weights and scales are ~12.6 MB against the 7.4 GB shard that contains them.

Reads only; nothing is written on the remote side. Key auth, never a password on argv.
"""
import argparse, json, os, re, struct, subprocess, sys

import torch

REMOTE = "root@10.0.0.70"
RDIR = "/mnt/bulk/hf/deepseek-ai--DeepSeek-V4.1-Flash"
KEY = os.path.expanduser("~/.ssh/id_ed25519")
DT = {"I8": torch.uint8, "F8_E8M0": torch.uint8, "BF16": torch.bfloat16}

ap = argparse.ArgumentParser()
ap.add_argument("--layers", default="0,10,20,30,39")
ap.add_argument("--experts", type=int, default=4)
ap.add_argument("--out", default=os.path.expanduser("~/dsv41-fp4-partial/experts.pt"))
ap.add_argument("--index", default=os.path.expanduser("~/dsv41-fp4-partial/model.safetensors.index.json"))
a = ap.parse_args()


def ssh(cmd):
    return subprocess.run(["ssh", "-i", KEY, "-o", "BatchMode=yes", REMOTE, cmd],
                          capture_output=True, check=True).stdout


_hdr_cache = {}


def header(shard):
    if shard not in _hdr_cache:
        n = struct.unpack("<Q", ssh(f"dd if={RDIR}/{shard} bs=8 count=1 status=none")[:8])[0]
        raw = ssh(f"dd if={RDIR}/{shard} bs=1M skip=8 count={n} "
                  f"iflag=skip_bytes,count_bytes status=none")
        _hdr_cache[shard] = (json.loads(raw), 8 + n)
    return _hdr_cache[shard]


def fetch(shard, name):
    h, base = header(shard)
    e = h[name]
    lo, hi = e["data_offsets"]
    raw = ssh(f"dd if={RDIR}/{shard} bs=1M skip={base + lo} count={hi - lo} "
              f"iflag=skip_bytes,count_bytes status=none")
    assert len(raw) == hi - lo, f"{name}: got {len(raw)} of {hi - lo} bytes"
    t = torch.frombuffer(bytearray(raw), dtype=DT[e["dtype"]]).reshape(e["shape"])
    return t.clone()


idx = json.load(open(a.index))["weight_map"]
out, got = {}, 0
for L in [int(x) for x in a.layers.split(",")]:
    for e in range(a.experts):
        for mat in ("w1", "w3"):
            for part in ("weight", "scale"):
                key = f"layers.{L}.ffn.experts.{e}.{mat}.{part}"
                if key not in idx:
                    continue
                out[key] = fetch(idx[key], key)
                got += out[key].numel() * out[key].element_size()
    print(f"  layer {L}: {got/1e6:7.1f} MB so far", flush=True)
os.makedirs(os.path.dirname(a.out), exist_ok=True)
torch.save(out, a.out)
print(f"  {len(out)} tensors, {got/1e6:.1f} MB -> {a.out}")
