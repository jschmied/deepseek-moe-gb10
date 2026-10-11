#!/usr/bin/env python3
"""Single safetensors tensors out of a Hugging Face repo by byte range: the header once a shard, then only the bytes.

    t = Remote("jayleaton/DeepSeek-V4.1-Flash-EXL3-2.8bpw").get("layers.10.ffn.experts.7.w1.trellis")  # numpy

The shard is never downloaded. Headers are cached in memory a repo; tensors are not cached.
"""
import json, struct, urllib.request

import numpy as np

DT = {"I8": np.int8, "U8": np.uint8, "I16": np.int16, "U16": np.uint16, "I32": np.int32, "U32": np.uint32,
      "F16": np.float16, "F32": np.float32, "F64": np.float64, "BF16": np.uint16, "F8_E8M0": np.uint8,
      "F8_E4M3": np.uint8, "BOOL": np.uint8}


def _range(url: str, start: int, end: int) -> bytes:
    req = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end - 1}"})
    with urllib.request.urlopen(req, timeout=300) as r:
        data = r.read()
    if len(data) != end - start:
        raise IOError(f"{url}: asked {end - start} bytes, got {len(data)}")
    return data


class Remote:
    def __init__(self, repo: str, revision: str = "main"):
        self.repo, self.rev = repo, revision
        url = f"https://huggingface.co/{repo}/resolve/{revision}/model.safetensors.index.json"
        with urllib.request.urlopen(url, timeout=120) as r:
            self.map = json.load(r)["weight_map"]
        self.heads = {}

    def url(self, shard: str) -> str:
        return f"https://huggingface.co/{self.repo}/resolve/{self.rev}/{shard}"

    def header(self, shard: str):
        if shard not in self.heads:
            n = struct.unpack("<Q", _range(self.url(shard), 0, 8))[0]
            self.heads[shard] = (8 + n, json.loads(_range(self.url(shard), 8, 8 + n)))
        return self.heads[shard]

    def has(self, name: str) -> bool:
        return name in self.map

    def info(self, name: str):
        base, h = self.header(self.map[name])
        return h[name]

    def get(self, name: str) -> np.ndarray:
        shard = self.map[name]
        base, h = self.header(shard)
        e = h[name]
        a, b = e["data_offsets"]
        raw = _range(self.url(shard), base + a, base + b)
        return np.frombuffer(raw, dtype=DT[e["dtype"]]).reshape(e["shape"]).copy()
