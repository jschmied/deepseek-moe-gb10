#!/usr/bin/env python3
"""Cross-layer expert prefetch: can layer i's routes name layer i+d's experts in time?

The proposal (third party, 2026-09-14) is a transition-table predictor -- no neural net:

    M[i,j][a,b] ~ P(b active at layer j | a active at layer i)
    score_j(b)  = bias_j(b) + sum_{a in top6_i} g_a * M[i,j][a,b]

evaluated not on exact token->expert accuracy but on the SET a decode step needs, and scored by the
metric that actually matters for prefetch: **the fraction of real cache MISSES converted into
prefetched hits at a bounded overfetch budget**. Wrong predictions cost bandwidth, never output.

§5 of ds41-measured-2026-09-13.md already measured a co-occurrence predictor at 30.5 % recall of
layer L's top-6 from L-1 (29.9 / 28.8 / 26.6 at d = 2 / 4 / 8) against a 14.4 % popularity baseline,
and costed it at 2.3-4 % of blocking misses removed. This re-runs it on the *set* metric with the
weighted score and the residency filter, because that framing is different enough to deserve its
own number rather than an argument from the old one.

  python tools/xlayer_predict.py --trace <dir with layer*.npz> --meta <meta.json>
"""
import argparse
import json
import os

import numpy as np

BLOCK = 6          # verify positions per decode step, i.e. what one prefetch must cover
N_EXPERTS = 384
ARENA_SLOTS = 5465  # the engine's measured arena: 5,465 of 15,360 (layer, expert) pairs = 35.6 %


def load(trace, n_layers):
    return [np.load(os.path.join(trace, f"layer{L}.npz"))["indices"].astype(np.int32)
            for L in range(n_layers)], \
           [np.load(os.path.join(trace, f"layer{L}.npz"))["weights"].astype(np.float32)
            for L in range(n_layers)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", required=True)
    ap.add_argument("--meta", required=True)
    ap.add_argument("--layers", type=int, default=40)
    ap.add_argument("--dists", default="1,2,4,8,12,16")
    ap.add_argument("--budgets", default="1.0,1.25,1.5,2.0")
    ap.add_argument("--arena-slots", type=int, default=ARENA_SLOTS,
                    help="0 = score the whole true set, which is the metric §5 used")
    a = ap.parse_args()

    idx, wts = load(a.trace, a.layers)
    T = idx[0].shape[0]
    meta = json.load(open(a.meta))
    # split by SEQUENCE, not by position: a co-occurrence table trained on the same sequence it is
    # scored on reads its own answer back out.
    bounds, p = [], 0
    for s in meta["seqs"]:
        bounds.append((s["id"], p, p + s["n"])); p += s["n"]
    assert p == T, (p, T)
    n_tr = int(0.6 * len(bounds))
    tr = np.zeros(T, bool)
    for _, lo, hi in bounds[:n_tr]:
        tr[lo:hi] = True
    te = ~tr
    print(f"{T} positions, {len(bounds)} sequences -> {tr.sum()} train / {te.sum()} test")

    # the arena the engine actually runs: hottest (layer, expert) pairs over the whole trace
    freq = np.zeros((a.layers, N_EXPERTS), np.int64)
    for L in range(a.layers):
        np.add.at(freq[L], idx[L].reshape(-1), 1)
    flat = np.argsort(-freq.reshape(-1), kind="stable")[:a.arena_slots]
    resident = np.zeros((a.layers, N_EXPERTS), bool)
    resident.reshape(-1)[flat] = True
    print(f"resident set: {resident.sum()} of {a.layers*N_EXPERTS} pairs "
          f"({100*resident.sum()/(a.layers*N_EXPERTS):.1f} %), "
          f"covering {100*freq[resident].sum()/freq.sum():.1f} % of routing decisions\n")

    dists = [int(x) for x in a.dists.split(",")]
    budgets = [float(x) for x in a.budgets.split(",")]
    print(f"  {'d':>3} {'true/step':>10} {'miss/step':>10} " +
          "".join(f"{('x%.2f' % b):>9}" for b in budgets) + f"{'popular':>9}")
    for d in dists:
        srcs = [i for i in range(a.layers - d)]
        # transition counts, pooled over all (i, i+d) pairs: one table per distance, which is what
        # makes this cheap enough to be interesting at all
        # ONE TABLE PER SOURCE LAYER, not one per distance. Pooling every (i, i+d) pair into a
        # single table costs about half the recall -- the transitions are layer-specific, which is
        # the whole reason the signal exists.
        M = {}
        pop = np.zeros((a.layers, N_EXPERTS), np.float64)
        for i in srcs:
            j = i + d
            si, sj = idx[i][tr], idx[j][tr]
            code = (si[:, :, None] * N_EXPERTS + sj[:, None, :]).reshape(-1)
            m = np.bincount(code, minlength=N_EXPERTS * N_EXPERTS).reshape(
                N_EXPERTS, N_EXPERTS).astype(np.float32)
            M[i] = m / np.maximum(m.sum(1, keepdims=True), 1.0)
        for L in range(a.layers):
            np.add.at(pop[L], idx[L][tr].reshape(-1), 1.0)

        pos = np.where(te)[0]
        pos = pos[: (len(pos) // BLOCK) * BLOCK].reshape(-1, BLOCK)
        hit = {b: 0 for b in budgets}
        tot_miss = tot_true = 0
        pop_hit = 0
        for i in srcs:
            j = i + d
            for blk in pos:
                true = np.unique(idx[j][blk])
                miss = true[~resident[j][true]]
                if len(miss) == 0:
                    continue
                src = idx[i][blk].reshape(-1)
                g = wts[i][blk].reshape(-1)
                sc = (M[i][src] * g[:, None]).sum(0) + 1e-6 * pop[j] / max(pop[j].sum(), 1)
                sc[resident[j]] = -1.0          # never spend a prefetch on a resident expert
                order = np.argsort(-sc, kind="stable")
                pord = np.argsort(-np.where(resident[j], -1.0, pop[j]), kind="stable")
                for b in budgets:
                    N = max(1, int(round(b * len(miss))))
                    hit[b] += len(np.intersect1d(order[:N], miss, assume_unique=False))
                N0 = max(1, int(round(budgets[-2] * len(miss))))
                pop_hit += len(np.intersect1d(pord[:N0], miss))
                tot_miss += len(miss); tot_true += len(true)
        nb = max(len(pos) * len(srcs), 1)
        print(f"  {d:>3} {tot_true/nb:>10.1f} {tot_miss/nb:>10.1f} " +
              "".join(f"{100*hit[b]/max(tot_miss,1):>8.1f}%" for b in budgets) +
              f"{100*pop_hit/max(tot_miss,1):>8.1f}%")
    print("\n  columns are RECALL OF ACTUAL MISSES at that overfetch budget; `popular` is the static\n"
          "  frequency baseline at the second-largest budget. The stop-line for this avenue was 70-80 %.")


if __name__ == "__main__":
    main()
