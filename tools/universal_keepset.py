#!/usr/bin/env python3
"""Does one universal keep-set remove the need for task profiles above some size?

The hypothesis (third party, 2026-09-14): the top experts are strongly domain-dependent, but most
of a moderately sized resident set is shared, so at ~200-220 experts/layer a single universal set
might retain nearly all of every domain's important mass -- whereas at 139 it clearly cannot.

Measured here on the unmasked trace (17,704 positions, categories `coding` and `general`), two ways,
because the ranking is the thing 0xBakeer's 0.5.0 changed:

  frequency   rank experts by how often the router picked them        (what all our keep-sets use)
  gateweight  rank by the SUM OF GATE WEIGHTS routed to them          (a saliency proxy we already
              have: `weights` is in the trace; true saliency also needs ||expert(x)||_2, which is not)

Reports, per keep size N:
  * Jaccard between the two categories' own top-N sets      -- how domain-dependent the ranking is
  * fraction of each category's gate-weight mass retained by ONE universal top-N set built on
    pooled traffic                                          -- what a profile-free design would cost
"""
import argparse, glob, os, re
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", required=True)
    ap.add_argument("--sizes", default="96,139,154,169,192,230,268,307")
    a = ap.parse_args()
    files = sorted(glob.glob(os.path.join(a.trace, "layer*.npz")),
                   key=lambda p: int(re.search(r"layer(\d+)", p).group(1)))
    assert files, a.trace
    N_E = 384
    sizes = [int(x) for x in a.sizes.split(",")]

    # per layer, per category: counts and gate-weight mass
    cnt = {c: np.zeros((len(files), N_E)) for c in ("coding", "general")}
    wsum = {c: np.zeros((len(files), N_E)) for c in ("coding", "general")}
    for L, f in enumerate(files):
        d = np.load(f)
        idx, w, cat = d["indices"].astype(np.int64), d["weights"].astype(np.float64), d["category"]
        for c in ("coding", "general"):
            m = cat == c
            np.add.at(cnt[c][L], idx[m].reshape(-1), 1.0)
            np.add.at(wsum[c][L], idx[m].reshape(-1), w[m].reshape(-1))
    pooled_cnt = cnt["coding"] + cnt["general"]
    pooled_w = wsum["coding"] + wsum["general"]

    for rank_name, cat_rank, pooled_rank in (("frequency", cnt, pooled_cnt),
                                             ("gateweight", wsum, pooled_w)):
        print(f"\n=== ranked by {rank_name} "
              f"({'what our keep-sets use' if rank_name=='frequency' else 'saliency proxy'}) ===")
        print(f"  {'N/layer':>8} {'% of 384':>9} {'Jaccard c-vs-g':>15} {'shared of each':>15} "
              f"{'universal keeps: coding':>24} {'general':>9}")
        for N in sizes:
            js, keep_c, keep_g = [], [], []
            for L in range(len(files)):
                tc = set(np.argsort(-cat_rank["coding"][L], kind="stable")[:N].tolist())
                tg = set(np.argsort(-cat_rank["general"][L], kind="stable")[:N].tolist())
                js.append(len(tc & tg) / len(tc | tg))
                uni = np.argsort(-pooled_rank[L], kind="stable")[:N]
                # what fraction of THIS category's gate-weight mass survives the universal set
                keep_c.append(wsum["coding"][L][uni].sum() / max(wsum["coding"][L].sum(), 1e-9))
                keep_g.append(wsum["general"][L][uni].sum() / max(wsum["general"][L].sum(), 1e-9))
            J = float(np.mean(js))
            shared = 2 * J / (1 + J)      # equal-size sets: |I|/|K|, NOT 1 - J
            print(f"  {N:>8} {100*N/N_E:>8.1f}% {J:>15.3f} {100*shared:>14.1f}% "
                  f"{100*np.mean(keep_c):>23.1f}% {100*np.mean(keep_g):>8.1f}%"
                  f"{'   <- worst layer %.1f%%' % (100*min(keep_c)) if N in (154,230) else ''}")
    print("\n  `shared of each` is 2J/(1+J), the fraction of ONE set that is also in the other for\n"
          "  equal-size sets. It is NOT 1-J: J=0.70 means 82.4 % shared, not 70 %.")


if __name__ == "__main__":
    main()
