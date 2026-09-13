#!/usr/bin/env python3
"""Two chunks through one MoE layer: re-unpack per chunk (today) vs unpack once and reuse.

The diagnostic that decides whether the layer-major CED transpose is worth building, without
rewriting execution order. `moe_forward_prefill` unpacks the layer's distinct experts CB3->FP4 into
a 32-expert scratch on EVERY call, so a prompt of C chunks unpacks the same population C times.
Arm B unpacks once into a full-population scratch and runs the same FP4 kernels against it.

Measured per the note: the redundant unpack is the smaller half (~6 s of a 131 s prefill); the
larger half is experts being re-READ from NVMe per chunk. This probe isolates the unpack half, which
is the part that needs no I/O and can therefore be measured cleanly on resident experts. Treat the
result as a floor on the transpose's value, not an estimate of it.
"""
import argparse, os, statistics, sys, time
import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--repo", default=os.path.expanduser("~/git/deepseek-v41-flash-spark"))
ap.add_argument("--experts", type=int, default=192, help="distinct experts a layer's chunk touches")
ap.add_argument("--chunks", type=int, default=4)
ap.add_argument("--tokens", type=int, default=2048, help="tokens per chunk")
ap.add_argument("--reps", type=int, default=3)
a = ap.parse_args()

sys.path.insert(0, a.repo); sys.path.insert(0, os.path.join(a.repo, "tools"))
import cb3_moe as C3
import fp4_moe as F4
from engine.codebook_sim import CodebookSim

dev = torch.device("cuda")
arena = C3.CB3ArenaV2(a.experts, dev); arena.sim = CodebookSim(3, dev)
for n in ("w1_lo", "w1_hi", "w3_lo", "w3_hi", "w2_lo", "w2_hi"):
    getattr(arena, n).fill_(0x5A)
for n in ("s1", "s3", "s2"):
    getattr(arena, n).fill_(120)
for n in ("w1_cb", "w3_cb", "w2_cb"):
    getattr(arena, n).random_(0, 16)

T, K = a.tokens, 6
rng = np.random.default_rng(3)
chunks = [torch.tensor(rng.integers(0, a.experts, size=(T, K)), dtype=torch.int32, device=dev)
          for _ in range(a.chunks)]
x = torch.randn(T, C3.DIM, dtype=torch.bfloat16, device=dev)
w = torch.full((T, K), 1.0 / K, dtype=torch.float32, device=dev)


def arm_a():
    """today: moe_forward_prefill per chunk, so the population is unpacked once per chunk."""
    for s in chunks:
        C3.moe_forward_prefill(x, s, w, arena)


def arm_b():
    """layer-major: unpack the whole population once, then run the FP4 kernels per chunk."""
    big = F4.ExpertArena(a.experts, dev)
    sel = torch.arange(a.experts, dtype=torch.int32, device=dev)
    C3._unpack_into(arena, sel, big)                     # ONE unpack for the layer
    P = T * K
    BM = F4._pick_bm(P)
    bn1, nw1, ns1 = F4._UP_CFG[BM]; bn2, nw2, ns2 = F4._DOWN_CFG[BM]
    h = torch.empty((P, C3.INTER), dtype=torch.bfloat16, device=dev)
    parts = torch.empty((P, C3.DIM), dtype=torch.float32, device=dev)
    for s in chunks:
        block_slot, block_pair, NB = F4.build_routing(s, a.experts, BM)
        F4._moe_up_kernel[(NB, C3.INTER // bn1)](
            x, big.w1, big.s1, big.w3, big.s3, h, w.reshape(-1), block_slot, block_pair,
            x.stride(0), h.stride(0), 10.0, TOPK=K, N=C3.INTER, K=C3.DIM, BM=BM, BN=bn1,
            num_warps=nw1, num_stages=ns1)
        F4._moe_down_kernel[(NB, C3.DIM // bn2)](
            h, big.w2, big.s2, parts, block_slot, block_pair, h.stride(0), parts.stride(0),
            TOPK=K, N=C3.DIM, K=C3.INTER, BM=BM, BN=bn2, NTOK=T, num_warps=nw2, num_stages=ns2)
    del big
    torch.cuda.empty_cache()


def timeit(fn):
    fn(); torch.cuda.synchronize()
    out = []
    for _ in range(a.reps):
        t0 = time.perf_counter(); fn(); torch.cuda.synchronize()
        out.append(time.perf_counter() - t0)
    return statistics.median(out) * 1e3


EB = 18_800_640
print(f"  {a.chunks} chunks x {a.tokens} tokens, {a.experts} distinct experts per chunk")
ta = timeit(arm_a); tb = timeit(arm_b)
print(f"  {'arm':<34} {'ms':>9} {'unpacked GB':>12}")
print(f"  {'A: re-unpack per chunk (today)':<34} {ta:9.1f} {a.chunks*a.experts*EB/1e9:12.2f}")
print(f"  {'B: unpack once, reuse':<34} {tb:9.1f} {a.experts*EB/1e9:12.2f}")
print(f"\n  reuse is {ta/tb:.2f}x on this layer  ({(1-tb/ta)*100:+.1f}% wall)")
print(f"  NOTE: resident experts only -- this is the unpack half. The NVMe half (experts re-READ")
print(f"  per chunk, ~341 GB measured on an 11.3k prompt) is larger and is not in this number.")
print("== ALL DONE ==")
