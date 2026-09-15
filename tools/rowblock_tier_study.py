#!/usr/bin/env python3
"""Is a row-block-adaptive CB2/CB3 mix worth building? One offline pass over the native CB3 cache.

Whole-expert CB2 is priced and rejected: -2.16 pp of coding top-1 against FP4 and a further -1.27 pp
against CB3 itself (ds41-measured 2026-09-13 section 20). The finer idea is to pick the tier per
32-row OUTPUT BLOCK instead of per expert, so that only blocks which barely notice lose the high bit
plane. That is only worth a format, a packer and a kernel if a LARGE fraction of blocks is genuinely
insensitive. This tool answers that and nothing else -- no format, no kernel.

Sizes per expert (tools/cb3_moe.py): FP4 18,800,640 B, CB3 14,454,784 B, CB2 9,992,192 B. Dropping
the hi plane for one 32-row block saves 32*K/8 bytes plus 4 codebook bytes per row (8 entries -> 4):
w1/w3 (K=5120) 20,608 B per block, 72 blocks each; w2 (K=2304) 9,344 B per block, 160 blocks. All
304 blocks converted is 4,462,592 B, which is exactly CB3 - CB2, so the accounting closes.

THE METRIC, and why this shape of it.
  Reference W = the CB3-dequantised weight VALUES -- what the engine serves today. Per 32-row block,
  relative Frobenius error ||Q - W||_F / ||W||_F, with Q the block requantised to the candidate tier.
  Value space, not packed codes: tools/perlayer_quant_error.py carries the scar tissue for that --
  scored on code bytes, a 7 -> 8 move flips the sign bit and reads as a huge delta while 0 -> 6 reads
  as small, and 2 bits came out with LESS error than 3, which is impossible.

  The cache is the only local source of routed experts (the FP4 shards are not on this box -- 40 of
  88 files of ~/dsv41-lean are absent and the shard store did not answer), so the reference IS CB3.
  CB3's own error against it is therefore EXACTLY ZERO by construction: the row codebook already
  restricts every weight to 8 grid levels, so re-picking an 8-subset costs nothing. That makes the
  literal "CB2 within 1.5x / 2x / 4x of CB3's block error" uncomputable from this source -- the
  denominator is 0 -- so the budgets below are multiples of the matrix's MEDIAN block error instead,
  which is the same question asked against a denominator that exists: how deep is the cheap tail?

  Zero is also a weak monotonicity check, so the check here is the whole bit ladder: CB1, CB2 and
  CB3 all requantised through the same code path, with err(CB1) > err(CB2) > err(CB3) required per
  block. Err(CB3) == 0 exactly is itself a real result -- it says the reader, the scale codec and the
  requantiser agree with the packer that wrote the file.

  Weight error is not output error. This is the same proxy perlayer_quant_error.py uses, for the same
  reason: it costs minutes instead of the 150 min/layer a paired teacher-forced trace costs. Its job
  is to kill the idea cheaply, not to price it.

Reads only ~13.8 MB per sampled expert out of a 211 GB file by O_DIRECT. No server, no full model.

  python tools/rowblock_tier_study.py --sample 8
"""
import argparse
import os
import sys

import numpy as np
import torch

DIM = 5120       # model hidden size: K of w1/w3, N of w2
INTER = 2304     # expert intermediate size: N of w1/w3, K of w2
GROUP = 32       # weights per UE8M0 scale group
ROWS_PER_BLOCK = 32

FP4_BYTES, CB3_BYTES, CB2_BYTES = 18_800_640, 14_454_784, 9_992_192

ap = argparse.ArgumentParser()
ap.add_argument("--cache", default=os.path.expanduser("~/dsv41-cb3/experts-cb3-s3.bin"))
ap.add_argument("--spark-repo", default=os.path.expanduser("~/git/deepseek-v41-flash-spark"))
ap.add_argument("--sample", type=int, default=16, help="number of (layer, expert) records to read")
ap.add_argument("--layers", default="0-39")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
ap.add_argument("--budgets", type=float, nargs="+", default=[0.25, 0.5, 0.667, 1.0],
                help="block-error thresholds, as multiples of that matrix's median block error")
a = ap.parse_args()

sys.path.insert(0, os.path.join(a.spark_repo, "engine"))
sys.path.insert(0, os.path.join(a.spark_repo, "tools"))
import cb3 as CB3                                    # noqa: E402  packer/dequant, the production one
import scale_codec as SC                             # noqa: E402
from cb3_cache import CB3Cache, ALIGN                # noqa: E402
from codebook_sim import CodebookSim                 # noqa: E402

DEV = torch.device(a.device)

# name -> (rows, K); the cache stores the planes in the arena's own order, one row per output row.
MATS = (("w1", INTER, DIM), ("w3", INTER, DIM), ("w2", DIM, INTER))
SCALE_PLANE = {"w1": "s1", "w3": "s3", "w2": "s2"}


def block_bytes_saved(K: int) -> int:
    """Bytes a 32-row block gives up when it drops to CB2: the hi plane plus half the codebook."""
    return ROWS_PER_BLOCK * (K // 8) + ROWS_PER_BLOCK * 4


@torch.no_grad()
def requant_values(codes: torch.Tensor, scale_u8: torch.Tensor, sim) -> torch.Tensor:
    """codes: long [N, K] FP4 grid codes; scale_u8: uint8 [N, K/32] UE8M0 -> dequantised fp32 [N, K]
    after restricting every row to the best 2^bits-level subset of the FP4 grid.

    Same selection as engine/codebook_sim.CodebookSim.requant_packed -- histogram weighted by the
    SQUARED group scale, one small GEMM against the precomputed per-subset cost -- but it returns
    values rather than packed codes, because packed codes are exactly what must not be compared.
    """
    N, K = codes.shape
    s = torch.exp2(scale_u8.float() - 127.0).repeat_interleave(GROUP, 1)   # [N, K]
    hist = torch.zeros(N, 16, device=codes.device, dtype=torch.float32).scatter_add_(1, codes, s * s)
    best = (hist @ sim.cost.T).argmin(dim=1)                               # [N]
    near = sim.near[best][torch.arange(N, device=codes.device)[:, None], codes]
    return CB3.FP4_TABLE.to(codes.device)[near].float() * s


@torch.no_grad()
def block_err(Q: torch.Tensor, W: torch.Tensor, nb: int):
    """-> (relative Frobenius error per 32-row block, squared absolute error per block)."""
    d2 = (Q - W).pow(2).view(nb, -1).sum(1)
    w2 = W.pow(2).view(nb, -1).sum(1)
    return torch.sqrt(d2 / w2.clamp_min(1e-30)), d2


def planes_of(staged: torch.Tensor, cache: CB3Cache, name: str, rows: int, K: int):
    """Slice one matrix's four planes out of a staged record and expand its scales."""
    def span(p, shape):
        lo, hi = cache.planes[p]
        assert hi - lo == shape[0] * shape[1], f"{p}: manifest span {hi - lo} != {shape}"
        return staged[lo:hi].view(*shape)

    lo = span(f"{name}_lo", (rows, K // 4)).to(DEV)
    hi = span(f"{name}_hi", (rows, K // 8)).to(DEV)
    cb = span(f"{name}_cb", (rows, 8)).to(DEV)
    sp = SCALE_PLANE[name]
    groups = K // GROUP
    sc = span(sp, (rows, SC.packed_row_bytes(groups))).to(DEV)
    return lo, hi, cb, SC.unpack_torch(sc, groups)


def main():
    cache = CB3Cache(a.cache, device=DEV)
    n_layers, n_exp = int(cache.man["n_layers"]), cache.n_experts
    l0, l1 = (int(x) for x in a.layers.split("-"))
    layers = list(range(l0, min(l1, n_layers - 1) + 1))

    # STRIDE across the layer range, do not take the first N: quantization sensitivity is a depth
    # story on this checkpoint (MiaAI-Lab's EXL3 build spends K=2 only on layers 18-22), so a sample
    # of 8 that only saw layers 0-7 would answer a different question. Seeded, so a rerun reads the
    # same experts and two arms are comparable.
    rng = np.random.default_rng(a.seed)
    picks = [(layers[(i * len(layers)) // a.sample], int(rng.integers(n_exp)))
             for i in range(a.sample)]

    sims = {b: CodebookSim(b, str(DEV)) for b in (1, 2, 3)}

    raw = np.empty(cache.record + ALIGN, dtype=np.uint8)
    off = (-raw.ctypes.data) % ALIGN                 # O_DIRECT needs the destination page-aligned
    host = raw[off:off + cache.record]

    print("== row-block tier study: how many 32-row blocks can drop CB3's high bit plane? ==")
    print(f"  cache   {a.cache}  ({cache.record:,} B/record, {n_layers} layers x {n_exp} experts)")
    print(f"  sample  {a.sample} records, layers {l0}-{l1}, seed {a.seed}, device {DEV}")
    print("  metric  relative Frobenius error per 32-row block, on DEQUANTISED VALUES,")
    print("          reference = the CB3 weights the engine serves today")

    rows_out = []          # one dict per block
    ladder = {1: 0, 2: 0, 3: 0}
    nblocks = 0
    for (layer, exp) in picks:
        cache.read_into(memoryview(host), layer, exp)
        staged = torch.from_numpy(host)
        for name, rows, K in MATS:
            lo, hi, cb, sc = planes_of(staged, cache, name, rows, K)
            codes = CB3.unpack_cb3_v2(lo, hi, cb)                       # long [rows, K], 0..15
            W = CB3.FP4_TABLE.to(DEV)[codes].float() * \
                torch.exp2(sc.float() - 127.0).repeat_interleave(GROUP, 1)
            nb = rows // ROWS_PER_BLOCK
            e = {}
            for bits in (3, 2, 1):
                rel, sq = block_err(requant_values(codes, sc, sims[bits]), W, nb)
                e[bits] = (rel, sq)
            # The ladder must be monotone in the bit or the metric is wrong; err(3) is exactly 0
            # because the row codebook already holds only 8 levels, so ">=" is the honest test there.
            ladder[3] += int((e[3][0] == 0).sum())
            ladder[2] += int((e[2][0] > e[3][0]).sum())
            ladder[1] += int((e[1][0] > e[2][0]).sum())
            nblocks += nb
            bs = block_bytes_saved(K)
            rel2, sq2 = e[2][0].cpu().numpy(), e[2][1].double().cpu().numpy()
            for i in range(nb):
                rows_out.append(dict(layer=layer, expert=exp, mat=name, blk=i,
                                     rel=float(rel2[i]), sq=float(sq2[i]), saved=bs))
            del lo, hi, cb, sc, codes, W, e
        if DEV.type == "cuda":
            torch.cuda.empty_cache()
    cache.close()

    print(f"\n  METRIC CHECK -- bit ladder over all {nblocks:,} sampled blocks")
    print(f"    err(CB3) == 0 exactly     {ladder[3]:>8,} / {nblocks:,}   reference is self-consistent")
    print(f"    err(CB2)  > err(CB3)      {ladder[2]:>8,} / {nblocks:,}")
    print(f"    err(CB1)  > err(CB2)      {ladder[1]:>8,} / {nblocks:,}")
    ok = ladder[3] == nblocks and ladder[2] == nblocks and ladder[1] == nblocks
    print(f"    MONOTONICITY CHECK {'PASSED' if ok else 'FAILED -- the metric is wrong, stop here'}")
    if not ok:
        return 1

    rel = np.array([r["rel"] for r in rows_out])
    mats = np.array([r["mat"] for r in rows_out])
    lays = np.array([r["layer"] for r in rows_out])
    sq = np.array([r["sq"] for r in rows_out])
    saved = np.array([r["saved"] for r in rows_out], dtype=np.int64)

    qs = [1, 10, 25, 50, 75, 90, 99]
    print("\n  CB2 BLOCK ERROR, pooled over the sample     (relative Frobenius, percent)")
    print(f"    {'matrix':<8}{'blocks':>8}" + "".join(f"{'p' + str(q):>8}" for q in qs) + f"{'p90/p10':>9}")
    for name, _, _ in MATS:
        m = mats == name
        p = np.percentile(rel[m], qs) * 100
        print(f"    {name:<8}{int(m.sum()):>8}" + "".join(f"{v:>8.2f}" for v in p) +
              f"{p[5] / max(p[1], 1e-9):>9.2f}")

    print("\n  BUDGETS -- threshold as a multiple of that matrix's median block error")
    print(f"    {'matrix':<8}{'budget':>8}{'blocks<=':>10}{'B saved/exp':>13}{'of CB3':>9}{'damage':>9}")
    for name, rows, K in MATS:
        m = mats == name
        med = float(np.median(rel[m]))
        tot_sq = sq[m].sum()
        # bytes are PER EXPERT: this matrix drops at most `rows / 32` blocks, so a block fraction
        # converts straight into a byte fraction of that matrix's full CB3 -> CB2 saving.
        full = block_bytes_saved(K) * (rows // ROWS_PER_BLOCK)
        for b in a.budgets:
            sel = m & (rel <= b * med)
            frac = int(sel.sum()) / max(int(m.sum()), 1)
            print(f"    {name:<8}{b:>8.2f}{frac * 100:>9.1f}%{int(frac * full):>13,}"
                  f"{frac * full / CB3_BYTES * 100:>8.1f}%{sq[sel].sum() / tot_sq * 100:>8.1f}%")

    print("\n  PER LAYER -- median CB2 block error, percent (blocks in the sample)")
    print(f"    {'layer':<8}" + "".join(f"{n:>10}" for n, _, _ in MATS) + f"{'records':>9}")
    for L in sorted(set(lays.tolist())):
        cells = []
        for name, _, _ in MATS:
            m = (lays == L) & (mats == name)
            cells.append(f"{np.median(rel[m]) * 100:>10.2f}" if m.any() else f"{'-':>10}")
        n_rec = len([1 for (l, _) in picks if l == L])
        print(f"    {L:<8}" + "".join(cells) + f"{n_rec:>9}")

    # The decision curve. Greedy by damage per byte saved, pooled across w1/w3/w2 -- a real mixed
    # format would choose blocks globally, not per matrix, so this is the best case available to it.
    print("\n  EXPERT-LEVEL TRADE-OFF -- convert the cheapest blocks first (damage per byte saved)")
    print("    all-CB2 is the 100 % row: 9,992,192 B and a measured -1.27 pp coding top-1 vs CB3")
    order = np.argsort(sq / saved)
    cum_saved = np.cumsum(saved[order]) / a.sample        # bytes per expert
    cum_dmg = np.cumsum(sq[order]) / sq.sum()
    n = len(order)
    print(f"    {'blocks':>8}{'B saved/exp':>13}{'expert size':>13}{'of CB3':>9}{'damage':>9}{'proxy pp':>10}")
    for f in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0):
        i = min(int(f * n) - 1, n - 1)
        b = cum_saved[i]
        print(f"    {f * 100:>7.0f}%{int(b):>13,}{int(CB3_BYTES - b):>13,}"
              f"{b / CB3_BYTES * 100:>8.1f}%{cum_dmg[i] * 100:>8.1f}%{cum_dmg[i] * 1.27:>10.2f}")
    print("    proxy pp = damage share x the measured -1.27 pp of all-CB2 vs CB3. Weight-space error")
    print("    is NOT output error; this only ranks, it does not price. Add ~38 B/expert of tier bits.")

    # VERDICT. A per-block tier only pays if the cheap blocks are genuinely cheap, so the test is
    # whether the greedy curve BEATS THE DIAGONAL: convert half the blocks and, if the choice is
    # worth anything, carry well under half the damage. The diagonal is what a coin flip gets, and a
    # format that only matches a coin flip is 304 tier bits, a variable-length hi plane and a second
    # kernel for nothing. Two conditions, both of which the shape of the distribution must clear.
    half = cum_dmg[min(int(0.5 * n) - 1, n - 1)]
    spread = float(np.percentile(rel, 90) / np.percentile(rel, 10))
    print(f"\n  VERDICT")
    print(f"    damage carried by the cheapest HALF of blocks   {half * 100:5.1f} %   (needs < 25 %)")
    print(f"    p90/p10 spread of the block error               {spread:5.2f}     (needs > 1.50)")
    worth = half < 0.25 and spread > 1.5
    print(f"    row-block-adaptive CB2/CB3 worth building?      {'YES' if worth else 'NO'}")
    if not worth:
        print("    The blocks are interchangeable: every 32-row block loses the same share of its")
        print("    norm to the dropped plane, so choosing WHICH blocks drop it buys nothing over")
        print("    choosing at random, and choosing at random is whole-expert CB2 pro rata -- which")
        print("    is already priced at -1.27 pp vs CB3. No format, no packer, no kernel.")
    print("\n== ALL DONE ==")
    return 0


if __name__ == "__main__":
    sys.exit(main())
