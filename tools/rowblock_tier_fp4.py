#!/usr/bin/env python3
"""Is a row-block-adaptive CB2/CB3 mix worth building? Asked against the ORIGINAL FP4 shards.

tools/rowblock_tier_study.py asked a neighbouring question and its verdict was withdrawn (see the
SCOPE LIMIT in that file). It derived CB2 from CB3-DEQUANTISED weights, because the FP4 shards were
not on the box when it was written. Two consequences made its answer the wrong one:

  * the optimal 4-level subset is NOT nested inside the optimal 8-level subset, so deriving CB2 from
    CB3 restricts CB2 to levels CB3 happened to keep. That can flatten exactly the heterogeneity the
    study was looking for. This tool measures the non-nesting rate instead of assuming it away.
  * it forced err(CB3) == 0 by construction, so the incremental error dE = E_CB2 - E_CB3 collapsed
    to E_CB2 and could not be ranked per byte saved -- which is the only quantity the format trades.

The proposed format is, per 32-row output block, independently:

    block A:  FP4 -> optimal CB3 (8 levels/row)      block B:  FP4 -> optimal CB2 (4 levels/row)

so this tool quantises each block both ways DIRECTLY FROM FP4 and reports dE per byte saved.

THE METRIC.
  Per 32-row output block, the absolute squared reconstruction error against the checkpoint's own
  FP4 values, in VALUE space:  SSE_b = ||Q_b - W_b||_F^2.  Value space, not packed codes:
  tools/perlayer_quant_error.py carries the scar tissue -- scored on code bytes a 7 -> 8 move flips
  the sign bit and reads as a huge delta while 0 -> 6 reads as small, and 2 bits came out with LESS
  error than 3, which is impossible.

  Absolute and not relative, because the ranked quantity has to be ADDITIVE: "the cheapest 50 % of
  blocks carries X % of the incremental damage" is only a sentence if the damage of a set of blocks
  is the sum of their damages. Relative Frobenius error is not. Under whitened activations the
  expected output squared error is proportional to ||dW||_F^2, so absolute SSE is also the honest
  proxy: a block whose weights are small really is cheaper to degrade. The relative view the
  withdrawn study reported is printed alongside, so the two are comparable.

  dE_b = SSE_b(CB2) - SSE_b(CB3) is the incremental damage of demoting block b, and

      dE per byte saved = dE_b / bytes_saved(K)

  is what a block-selecting packer would sort on. bytes_saved is 32*K/8 (the dropped high bit plane)
  + 32*4 (codebook, one FP4 code per byte, 8 entries -> 4); 20,608 B for a w1/w3 block (K=5120),
  9,344 B for a w2 block (K=2304). 72 + 72 + 160 = 304 blocks per expert, 4,462,592 B if all of them
  convert, which is exactly CB3_BYTES - CB2_BYTES, so the accounting closes.

WHAT CodebookSim ACTUALLY OPTIMISES -- checked before trusting it, as instructed.
  engine/codebook_sim.py builds hist[row, level] = sum over 32-groups of scale_g^2 * count(level),
  and cost[subset, level] = min_{m in subset} (v_level - v_m)^2 on the RAW FP4 grid values, then
  picks argmin over all C(16, 2^B) subsets of hist @ cost.T. Since the group scale is a positive
  common factor of every weight in its group, min_m scale^2 (v - v_m)^2 = scale^2 min_m (v - v_m)^2:
  the nearest member does not depend on the scale, and the scale^2 weight makes the objective
  EXACTLY sum_w (W_w - Q_w)^2 = ||W_row - Q_row||_F^2 in value space. So it is the exact minimiser
  of the metric used here, over all subsets -- not a greedy or k-means approximation.

  Two properties that matter for this study: (1) the codebook is per ROW and spans the FULL row
  (K=5120 or 2304), so a 32-row block's error is the sum of 32 independent row optima and there is
  no block-level coupling to model; (2) it is purely weight-space -- no activation statistics, no
  output sensitivity. That is the same cheap-rejector caveat perlayer_quant_error.py carries: two
  blocks with identical reconstruction error can differ in output effect. Its job is to kill the
  idea cheaply, not to price it.

THE DECISION RULE, applied in the output.
  If p90/p10 of dE/byte is ~1.0 AND the cheapest 50 % (by bytes saved) carries ~40-50 % of the
  incremental damage, the blocks are interchangeable and the idea is CLOSED -- a selector cannot
  beat a coin flip, so mixed tiering is just uniform CB2 on half the expert. Heterogeneity worth a
  format, a packer and a kernel means a LONG CHEAP TAIL: the cheapest half of the bytes has to buy
  its savings at a small fraction of the damage.

Reads one expert at a time (w1+w3+w2 = ~18.8 MB of FP4 plus scales) out of a 7.4 GB shard.

  python tools/rowblock_tier_fp4.py --shard /opt/llm/models/dsv41-shards/model-00012-of-00048.safetensors \
      --layer 9 --experts 4 --out /tmp/rb-l9.jsonl
  python tools/rowblock_tier_fp4.py --summary /tmp/rb-*.jsonl
"""
import argparse
import glob
import json
import os
import re
import sys

import numpy as np

DIM = 5120        # model hidden size: K of w1/w3, N of w2
INTER = 2304      # expert intermediate size: N of w1/w3, K of w2
GROUP = 32        # weights per UE8M0 scale group
ROWS_PER_BLOCK = 32

FP4_BYTES, CB3_BYTES, CB2_BYTES = 18_800_640, 14_454_784, 9_992_192
MATS = ("w1", "w3", "w2")                       # w1/w3 are [INTER, DIM]; w2 is [DIM, INTER]


def block_bytes_saved(K: int) -> int:
    """Bytes a 32-row block gives up when it drops from CB3 to CB2: the high bit plane (K/8 per row)
    plus half the codebook (8 -> 4 entries, one FP4 code per byte). 20,608 B at K=5120, 9,344 at
    K=2304; 72+72+160 blocks per expert sum to 4,462,592 B = CB3_BYTES - CB2_BYTES exactly."""
    return ROWS_PER_BLOCK * (K // 8) + ROWS_PER_BLOCK * 4


# ---------------------------------------------------------------------------------------------
# measurement
# ---------------------------------------------------------------------------------------------

def _deq(w_u8, s_u8, torch):
    """packed FP4 codes [N, K/2] + UE8M0 scales [N, K/32] -> fp32 weight VALUES [N, K].

    s_u8 must already be .view(torch.uint8): .float() on an F8_E8M0 tensor DECODES it, which is the
    bug that collapsed the CB3 trace (perlayer_quant_error.py carries the same warning).
    """
    from engine.codebook_sim import FP4_VALS
    N, K2 = w_u8.shape
    lo = (w_u8 & 0x0F).long()
    hi = ((w_u8 >> 4) & 0x0F).long()               # low nibble = EVEN element (v41_ref convention)
    codes = torch.stack([lo, hi], dim=-1).reshape(N, K2 * 2)
    vals = FP4_VALS.to(w_u8.device)[codes]
    scale = torch.exp2(s_u8.float() - 127.0).repeat_interleave(GROUP, dim=1)
    return vals * scale, codes


def _used_levels(codes, torch):
    """[N, K] FP4 codes -> bool [N, 16], which grid levels the row's codebook actually emits."""
    N = codes.shape[0]
    p = torch.zeros(N, 16, dtype=torch.bool, device=codes.device)
    return p.scatter_(1, codes, True)


def measure_expert(f, layer, expert, sims, dev, torch):
    """-> list of per-block dicts for this expert's w1, w3, w2, plus (nested, rows) counters."""
    out, n_nested, n_rows = [], 0, 0
    for name in MATS:
        w = f.get_tensor(f"layers.{layer}.ffn.experts.{expert}.{name}.weight").to(dev).view(torch.uint8)
        s = f.get_tensor(f"layers.{layer}.ffn.experts.{expert}.{name}.scale").to(dev).view(torch.uint8)
        N, K = w.shape[0], w.shape[1] * 2
        assert N % ROWS_PER_BLOCK == 0, f"{name}: {N} rows is not a whole number of 32-row blocks"
        W, _ = _deq(w, s, torch)

        sse, used = {}, {}
        for bits in (3, 2):
            # the PRODUCTION packer, on the raw FP4 bytes -- not a local re-derivation, and not
            # chained off the CB3 result, which is the whole point of this tool existing
            q_codes = sims[bits].requant_packed(w, s)
            Q, qc = _deq(q_codes, s, torch)
            nb = N // ROWS_PER_BLOCK
            sse[bits] = (Q - W).pow(2).double().view(nb, -1).sum(1)
            used[bits] = _used_levels(qc, torch)
            del q_codes, Q, qc
        w2n = W.pow(2).double().view(N // ROWS_PER_BLOCK, -1).sum(1)

        # Non-nesting: levels CB2 emits that CB3 does not. Undercounts (a codebook entry can go
        # unused), so any nonzero rate already falsifies the withdrawn study's nesting assumption.
        n_nested += int((used[2] & ~used[3]).sum(1).eq(0).sum())
        n_rows += N

        bs = block_bytes_saved(K)
        s3, s2, wn = (t.cpu().numpy() for t in (sse[3], sse[2], w2n))
        for i in range(N // ROWS_PER_BLOCK):
            out.append(dict(layer=int(layer), expert=int(expert), mat=name, blk=i, K=K,
                            sse3=float(s3[i]), sse2=float(s2[i]), w2=float(wn[i]), saved=bs))
        del w, s, W, sse, used, w2n
        if dev.type == "cuda":
            torch.cuda.empty_cache()
    return out, n_nested, n_rows


# ---------------------------------------------------------------------------------------------
# aggregation
# ---------------------------------------------------------------------------------------------

QS = [1, 10, 50, 90, 99]


def summarise(rows):
    sse3 = np.array([r["sse3"] for r in rows])
    sse2 = np.array([r["sse2"] for r in rows])
    w2 = np.array([r["w2"] for r in rows])
    saved = np.array([r["saved"] for r in rows], dtype=np.float64)
    mat = np.array([r["mat"] for r in rows])
    dE = sse2 - sse3
    dpb = dE / saved
    rel3 = np.sqrt(sse3 / np.maximum(w2, 1e-300))
    rel2 = np.sqrt(sse2 / np.maximum(w2, 1e-300))

    print(f"\n  SANITY CHECKS over all {len(rows):,} blocks")
    n_ord = int((sse3 < sse2).sum())
    print(f"    err(CB3) < err(CB2)          {n_ord:>8,} / {len(rows):,}"
          f"   {'OK' if n_ord == len(rows) else 'FAILED -- the metric is wrong, stop here'}")
    print(f"    err(CB3) relative, min/med/max  {rel3.min() * 100:.2f}% / {np.median(rel3) * 100:.2f}%"
          f" / {rel3.max() * 100:.2f}%")
    zero3 = int((rel3 < 1e-6).sum())
    print(f"    err(CB3) ~ 0 (rel < 1e-6)    {zero3:>8,} / {len(rows):,}"
          f"   {'OK -- quantising from FP4' if zero3 == 0 else 'FAILED -- source is CB3, not FP4'}")
    print(f"    err(CB2) relative, min/med/max  {rel2.min() * 100:.2f}% / {np.median(rel2) * 100:.2f}%"
          f" / {rel2.max() * 100:.2f}%")

    print("\n  dE PER BYTE SAVED     (SSE_CB2 - SSE_CB3, per byte the demotion buys)")
    hdr = f"    {'matrix':<8}{'blocks':>8}" + "".join(f"{'p' + str(q):>11}" for q in QS) + f"{'p90/p10':>9}"
    print(hdr)
    for name in ("w1", "w3", "w2", "ALL"):
        m = np.ones(len(rows), bool) if name == "ALL" else (mat == name)
        if not m.any():
            continue
        p = np.percentile(dpb[m], QS)
        print(f"    {name:<8}{int(m.sum()):>8,}" + "".join(f"{v:>11.3e}" for v in p)
              + f"{p[3] / max(p[1], 1e-300):>9.2f}")

    # The withdrawn study's view, for comparison: the INCREMENTAL relative Frobenius error. Kept
    # because that is the number readers have in their head (it reported p1 32.0 %, p99 33.5 %).
    d_rel = (rel2 - rel3) * 100
    p = np.percentile(d_rel, QS)
    print(f"\n  for comparison, incremental RELATIVE error (rel2 - rel3, percentage points)")
    print(f"    {'ALL':<8}{len(rows):>8,}" + "".join(f"{v:>11.3f}" for v in p)
          + f"{p[3] / max(p[1], 1e-300):>9.2f}")

    # Cheapest-first: sort by dE/byte and walk. The x axis is BYTES SAVED, not block count, because
    # that is what the format buys and it makes the null exact: with identical dE/byte everywhere,
    # 25 % of the bytes costs exactly 25 % of the damage.
    o = np.argsort(dpb)
    cb, cd = np.cumsum(saved[o]), np.cumsum(dE[o])
    cb, cd = cb / cb[-1], cd / cd[-1]
    print("\n  CHEAPEST-FIRST  (blocks sorted by dE/byte; share of total incremental damage taken)")
    for frac in (0.25, 0.50, 0.75):
        j = int(np.searchsorted(cb, frac))
        j = min(j, len(cd) - 1)
        print(f"    cheapest {frac * 100:.0f}% of bytes saved  ->  {cd[j] * 100:5.1f}% of the damage"
              f"   (null = {frac * 100:.0f}.0%)")

    ratio = np.percentile(dpb, 90) / max(np.percentile(dpb, 10), 1e-300)
    j50 = min(int(np.searchsorted(cb, 0.50)), len(cd) - 1)
    half = cd[j50] * 100
    print("\n  VERDICT (decision rule from the brief)")
    print("    rule: p90/p10 ~ 1.0 AND cheapest 50% carries ~40-50% of the damage  =>  CLOSED.")
    print("          heterogeneity worth building for means a LONG CHEAP TAIL.")
    print(f"    measured: p90/p10 = {ratio:.2f},  cheapest 50% of bytes carries {half:.1f}% of the damage")
    if ratio < 1.5 and 35.0 <= half <= 55.0:
        print("    => CLOSED on this evidence: the blocks are interchangeable, a selector cannot")
        print("       beat a coin flip, and mixed tiering degenerates to uniform CB2 on half the expert.")
    elif ratio >= 1.5 and half < 35.0:
        print("    => OPEN: there is a cheap tail. Next step is activation-weighted dE, not a packer.")
    else:
        print("    => AMBIGUOUS: the two halves of the rule disagree; do not quote a verdict from this.")
    print("\n    one layer decides nothing -- depth is a real axis on this checkpoint (MiaAI-Lab's")
    print("    EXL3 build spends K=2 only on layers 18-22). Stream the shards and re-run --summary.")


# ---------------------------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", help="one model-*.safetensors holding the layer's experts")
    ap.add_argument("--layer", type=int)
    ap.add_argument("--experts", type=int, default=16, help="how many experts to sample from that layer")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", help="append one JSON line per 32-row block here")
    ap.add_argument("--summary", nargs="+", help="aggregate previously written --out files and stop")
    ap.add_argument("--spark-repo", default=os.path.expanduser("~/git/deepseek-v41-flash-spark"))
    ap.add_argument("--device", default=None)
    a = ap.parse_args()

    if a.summary:
        files = [p for pat in a.summary for p in sorted(glob.glob(pat))] or a.summary
        rows = []
        for p in files:
            with open(p) as fh:
                rows += [json.loads(ln) for ln in fh if ln.strip()]
        if not rows:
            print("no blocks found", file=sys.stderr)
            return 1
        lay = sorted({r["layer"] for r in rows})
        exp = {(r["layer"], r["expert"]) for r in rows}
        print("== row-block tier, quantised from ORIGINAL FP4 (aggregate) ==")
        print(f"  {len(files)} file(s), {len(exp)} experts over {len(lay)} layer(s): {lay}")
        summarise(rows)
        return 0

    if not a.shard or a.layer is None:
        ap.error("--shard and --layer are required unless --summary is given")

    sys.path.insert(0, a.spark_repo)
    import torch
    from safetensors import safe_open
    from engine.codebook_sim import CodebookSim

    dev = torch.device(a.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    f = safe_open(a.shard, "pt", device="cpu")
    have = sorted({int(m.group(1)) for m in
                   (re.match(rf"layers\.{a.layer}\.ffn\.experts\.(\d+)\.w1\.weight$", k) for k in f.keys())
                   if m})
    if not have:
        print(f"no layer-{a.layer} routed experts in {a.shard}", file=sys.stderr)
        return 1
    # Seeded sample, not the first N: expert ids are not ordered by anything, but a rerun must read
    # the same experts so two arms stay comparable.
    rng = np.random.default_rng(a.seed)
    picks = sorted(rng.choice(np.array(have), size=min(a.experts, len(have)), replace=False).tolist())

    sims = {b: CodebookSim(b, str(dev)) for b in (3, 2)}

    print("== row-block tier study, quantised from the ORIGINAL FP4 shard ==")
    print(f"  shard   {a.shard}")
    print(f"  layer   {a.layer}, {len(have)} routed experts present, sampling {len(picks)} (seed {a.seed})")
    print(f"  device  {dev}")
    print("  metric  absolute squared reconstruction error vs the FP4 values, per 32-row block;")
    print("          CB3 and CB2 quantised INDEPENDENTLY from FP4 (this is the fix over")
    print("          rowblock_tier_study.py, which derived CB2 from CB3 and forced err(CB3)=0)")

    rows, nested, nrows = [], 0, 0
    fh = open(a.out, "a") if a.out else None
    for e in picks:
        r, nn, nr = measure_expert(f, a.layer, e, sims, dev, torch)
        nested += nn
        nrows += nr
        rows += r
        if fh:
            for d in r:
                fh.write(json.dumps(d, separators=(",", ":")) + "\n")
            fh.flush()
    if fh:
        fh.close()
        print(f"\n  wrote {len(rows):,} block records to {a.out}")

    print(f"\n  NESTING -- rows whose CB2 levels all lie inside its CB3 levels: {nested:,} / {nrows:,}"
          f" ({100.0 * nested / max(nrows, 1):.1f}%)")
    print("    the withdrawn study assumed 100% by construction; every row below that is a row it")
    print("    could not have represented.")
    summarise(rows)
    return 0


if __name__ == "__main__":
    sys.exit(main())
