#!/usr/bin/env python3
"""Baseline for the CB3 arena-layout question: what does the SHIPPED kernel actually do?

The claim under test is that `_cb3v3_up_kernel` runs well below the device's gather rate because
each program reads eight independent base pointers (`w1/w3 x {lo,hi,cb,s}`), and that interleaving
the arena per (slot, n-block) would recover it. Before writing a packer, this measures the shipped
path itself: `moe_forward_v3` over a real `CB3ArenaV2`, at the real decode shape, swept over arena
footprint -- because the effect is reported to be absent at a small arena and only to appear at the
~90 GB the engine really runs.

It also re-measures the stored 168.7 GB/s (2026-09-11) on today's build, which is the number the
whole +13% estimate is anchored to and which has never been re-checked.

Bytes are counted the way the kernel reads them: `build_routing` sorts the (token, expert) pairs by
slot, so each of the NB blocks reads one slot's w1+w3 in the up kernel and its w2 in the down
kernel -- NB x bytes_per_slot in total.

Real weights are loaded into a few slots for the correctness check; the rest of the arena is filled
with a cheap pattern, which is legitimate here because the kernel's cost is footprint and access
pattern, not content. Codebook bytes are kept in range so the table lookup stays in bounds.
"""
import argparse, os, statistics, sys, time
import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--repo", default=os.path.expanduser("~/git/deepseek-v41-flash-spark"))
ap.add_argument("--shard", default="/opt/llm/models/dsv41-shards/model-00023-of-00048.safetensors")
ap.add_argument("--slots", default="384,1024,2048,4096,5212")
ap.add_argument("--real-slots", type=int, default=4, help="slots filled with real weights (correctness)")
ap.add_argument("--block", type=int, default=6, help="verify block width; decode shape is [block, 6]")
ap.add_argument("--distinct", type=int, default=23, help="distinct experts a layer touches (trace: 22.56 at k=6)")
ap.add_argument("--reps", type=int, default=30)
ap.add_argument("--starts", type=int, default=3)
a = ap.parse_args()

sys.path.insert(0, a.repo)
sys.path.insert(0, os.path.join(a.repo, "tools"))
import cb3_moe as C3
import fp4_moe as F4
from engine.codebook_sim import CodebookSim

dev = torch.device("cuda")
sim = CodebookSim(3, dev)
torch.manual_seed(0)


def fill(arena):
    """Fault the whole arena in with in-range bytes. Content is irrelevant to the timing."""
    for name in ("w1_lo", "w1_hi", "w3_lo", "w3_hi", "w2_lo", "w2_hi"):
        getattr(arena, name).fill_(0x5A)
    for name in ("s1", "s3", "s2"):
        getattr(arena, name).fill_(120)          # a real UE8M0 exponent (measured: 119-122)
    for name in ("w1_cb", "w3_cb", "w2_cb"):
        getattr(arena, name).random_(0, 16)      # codebook entries index the 16-entry FP4 table


def load_real(arena, n):
    from safetensors import safe_open
    got = 0
    with safe_open(a.shard, framework="pt") as f:
        keys = [k for k in f.keys() if k.endswith(".w1.weight") and ".ffn.experts." in k]
        keys.sort()
        for k in keys[:n]:
            p = k[: -len("w1.weight")]
            w1, s1 = f.get_tensor(p + "w1.weight"), f.get_tensor(p + "w1.scale")
            w2, s2 = f.get_tensor(p + "w2.weight"), f.get_tensor(p + "w2.scale")
            w3, s3 = f.get_tensor(p + "w3.weight"), f.get_tensor(p + "w3.scale")
            arena.load_slot(got, w1, s1, w2, s2, w3, s3)
            got += 1
    return got


def check(arena, n_real):
    """The shipped kernel against the repo's own dequant+matmul reference on the same slots."""
    T, K = 2, 2
    x = torch.randn(T, C3.DIM, dtype=torch.bfloat16, device=dev)
    slots = torch.tensor([[0, 1], [1, 0]], dtype=torch.int32, device=dev)[:T, :K]
    wgt = torch.full((T, K), 0.5, dtype=torch.float32, device=dev)
    out = C3.moe_forward_v3(x, slots, wgt, arena)
    ref = F4.moe_forward_reference(x, slots, wgt, arena)
    return float((out.float() - ref.float()).norm() / ref.float().norm())


def bench(arena, slots_n):
    """Rotate over several routing draws so the touched set changes between reps, as it does in the
    engine (a different 6 experts per layer). NB from build_routing is an UPPER BOUND
    (ceil(P/BM) + min(n_slots,P)) with unused blocks marked -1 and skipped, so bytes are counted from
    the DISTINCT slots actually touched, not from NB."""
    T, K = a.block, 6
    x = torch.randn(T, C3.DIM, dtype=torch.bfloat16, device=dev)
    rng = np.random.default_rng(7)
    draws = []
    for _ in range(16):
        pool = rng.choice(slots_n, size=min(a.distinct, slots_n), replace=False)
        ids = rng.choice(pool, size=(T, K))
        draws.append((torch.tensor(ids, dtype=torch.int32, device=dev), len(np.unique(ids))))
    wgt = torch.full((T, K), 1.0 / K, dtype=torch.float32, device=dev)
    BM = C3._pick_bm(T * K)
    NB = C3.build_routing(draws[0][0], arena.slots, BM)[2]
    mean_distinct = sum(d for _, d in draws) / len(draws)
    bytes_per_rep = mean_distinct * arena.bytes_per_slot

    for i in range(8):
        C3.moe_forward_v3(x, draws[i % len(draws)][0], wgt, arena)
    torch.cuda.synchronize()
    out = []
    for _ in range(a.starts):
        t0 = time.perf_counter()
        for i in range(a.reps):
            C3.moe_forward_v3(x, draws[i % len(draws)][0], wgt, arena)
        torch.cuda.synchronize()
        out.append((time.perf_counter() - t0) / a.reps)
    ms = statistics.median(out) * 1e3
    return ms, bytes_per_rep / statistics.median(out) / 1e9, NB, mean_distinct, bytes_per_rep


print(f"  shard {os.path.basename(a.shard)}   block {a.block} x top-6, {a.distinct} distinct experts")
print(f"  {'slots':>6} {'arena GB':>9} {'NBmax':>6} {'distinct':>8} {'read MB':>9} {'ms':>8} {'GB/s':>8}   rel err")
for s in [int(x) for x in a.slots.split(",")]:
    try:
        arena = C3.CB3ArenaV2(s, dev)
        arena.sim = sim
        fill(arena)
        nreal = load_real(arena, a.real_slots) if a.real_slots else 0
        err = check(arena, nreal) if nreal >= 2 else float("nan")
        ms, gbs, NB, nd, nb = bench(arena, s)
        print(f"  {s:>6} {s*arena.bytes_per_slot/1e9:9.1f} {NB:>6} {nd:8.1f} {nb/1e6:9.1f} {ms:8.3f} {gbs:8.1f}   {err:.2e}")
        del arena
        torch.cuda.empty_cache()
    except torch.cuda.OutOfMemoryError:
        print(f"  {s:>6}  OOM"); torch.cuda.empty_cache(); break
print("== ALL DONE ==")
