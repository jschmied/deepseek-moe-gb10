#!/usr/bin/env python3
"""Simulate DeepSeek-V4.1-Flash's ACTUAL expert residency, as engine/experts.py implements it.

Earlier residency work here modelled a frozen keep-set plus a small adaptive remainder. That is
not what 0xBakeer's engine does. `ExpertLRU` keeps ONE global LRU over `(layer, expert)` keys
spanning all 40 layers, prewarmed by `warm_start()` from a global ranking, plus a FIFO ring of
`transient_slots` (default 400) that absorbs **prefill** misses so a long prompt cannot evict the
decode working set. Decode misses evict from the LRU; a transient entry hit during decode is
promoted into the LRU.

So the open questions are not "keep-set or LRU" -- it is already LRU. They are:
  * does the global warm start earn its slots, or would a cold LRU converge anyway?
  * is `transient_slots=400` the right size, given every one of them is taken from the LRU?
  * is one global pool better than per-layer budgets at the same total size?
  * what is the real decode miss rate, and therefore GB/token, at the shipped arena size?

The LRU persists across requests, as it does in a served process.
"""
import argparse, collections, json, os
import numpy as np

EXPERT_BYTES = 3 * (2304 * 2560 + 2304 * 160)      # 18,800,640 -- engine/experts.py:46


class ExpertLRU:
    """Faithful to engine/experts.py: global LRU + prefill transient ring."""

    def __init__(self, n_slots, transient_slots, per_layer=False, n_layers=40):
        self.transient_slots = transient_slots
        self.lru_slots = n_slots - transient_slots
        assert self.lru_slots > 0, f"transient_slots {transient_slots} >= arena {n_slots}"
        self.per_layer = per_layer
        self.n_layers = n_layers
        if per_layer:
            self.pools = [collections.OrderedDict() for _ in range(n_layers)]
            self.cap = self.lru_slots // n_layers
        else:
            self.lru = collections.OrderedDict()
        self.ring = collections.OrderedDict()          # FIFO, (layer,expert) -> 1
        self.hits = self.decode_miss = self.prefill_miss = self.promoted = 0

    def _pool(self, layer):
        return self.pools[layer] if self.per_layer else self.lru

    def _cap(self):
        return self.cap if self.per_layer else self.lru_slots

    def _lru_insert(self, key, layer):
        p = self._pool(layer)
        if len(p) >= self._cap():
            p.popitem(last=False)
        p[key] = 1

    def resolve(self, layer, experts, prefill):
        p = self._pool(layer)
        for e in experts:
            key = (layer, e)
            if key in p:
                self.hits += 1
                p.move_to_end(key)
                continue
            if key in self.ring:
                self.hits += 1
                if not prefill:                        # promote out of the ring
                    del self.ring[key]
                    self._lru_insert(key, layer)
                    self.promoted += 1
                continue
            if prefill and self.transient_slots:
                self.prefill_miss += 1
                if len(self.ring) >= self.transient_slots:
                    self.ring.popitem(last=False)
                self.ring[key] = 1
            elif prefill:               # transient=0: prefill competes for the LRU directly
                self.prefill_miss += 1
                self._lru_insert(key, layer)
            else:
                self.decode_miss += 1
                self._lru_insert(key, layer)

    def warm(self, ranked):
        p = None
        for key in ranked[: self.lru_slots]:
            self._lru_insert(key, key[0])


def load(path):
    Z = np.load(path)
    names = sorted({k.split("__")[0] for k in Z.files})
    return [(n, Z[f"{n}__routed"], int(Z[f"{n}__meta"][0])) for n in names]


def rank_global(reqs, exclude):
    c = collections.Counter()
    for n, r, _ in reqs:
        if n == exclude:
            continue
        L, K = r.shape[1], r.shape[2]
        for l in range(L):
            for e, cnt in zip(*np.unique(r[:, l, :], return_counts=True)):
                c[(l, int(e))] += int(cnt)
    return [k for k, _ in c.most_common()]


def run(reqs, n_slots, transient, warm_ranked, per_layer=False, n_layers=40):
    lru = ExpertLRU(n_slots, transient, per_layer, n_layers)
    if warm_ranked is not None:
        lru.warm(warm_ranked)
    dec_tok = 0
    for name, r, npr in reqs:
        L = r.shape[1]
        for l in range(L):                              # prefill: one resolve per layer
            lru.resolve(l, np.unique(r[:npr, l, :]).tolist(), True)
        for t in range(npr, r.shape[0]):                # decode: token by token
            for l in range(L):
                lru.resolve(l, np.unique(r[t, l, :]).tolist(), False)
            dec_tok += 1
    tot = lru.hits + lru.decode_miss + lru.prefill_miss
    return dict(coverage=100.0 * lru.hits / tot, decode_miss=lru.decode_miss,
                miss_per_tok=lru.decode_miss / dec_tok,
                gb_per_tok=lru.decode_miss * EXPERT_BYTES / dec_tok / 1e9,
                promoted=lru.promoted, prefill_miss=lru.prefill_miss, dec_tok=dec_tok)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", required=True)
    ap.add_argument("--arena-gb", type=float, default=98.0)
    ap.add_argument("--slot-bytes", type=int, default=EXPERT_BYTES)
    ap.add_argument("--transient", default="0,8,100,400,1000")
    ap.add_argument("--held-out", type=int, default=8, help="last N requests are the measured stream")
    a = ap.parse_args()

    reqs = load(a.trace)
    n_layers = reqs[0][1].shape[1]
    E = int(max(r.max() for _, r, _ in reqs)) + 1
    n_slots = int(a.arena_gb * 1e9 // a.slot_bytes)
    print(f"  {len(reqs)} requests, {n_layers} layers, {E} experts, "
          f"{sum(r.shape[0] for _, r, _ in reqs)} tokens")
    print(f"  arena {a.arena_gb:.0f} GB / {a.slot_bytes/1e6:.2f} MB = {n_slots} slots "
          f"= {100*n_slots/(E*n_layers):.1f}% of {E*n_layers} (layer, expert) pairs")

    stream = reqs[-a.held_out:]
    ranked = rank_global(reqs, exclude=None)
    ranked_ho = rank_global(reqs[:-a.held_out], exclude=None)

    print(f"\n  measured stream: last {len(stream)} requests, "
          f"{sum(r.shape[0]-npr for _, r, npr in stream)} decode tokens\n")
    print(f"  {'transient':>9} {'warm start':>18}  {'coverage':>9} {'miss/tok':>9} {'GB/tok':>8} {'promoted':>9}")
    for tr in [int(x) for x in a.transient.split(",")]:
        if tr >= n_slots:
            continue
        for label, rk in (("cold", None), ("global (held-out)", ranked_ho)):
            r = run(stream, n_slots, tr, rk, n_layers=n_layers)
            print(f"  {tr:>9} {label:>18}  {r['coverage']:8.2f}% {r['miss_per_tok']:9.2f} "
                  f"{r['gb_per_tok']:8.3f} {r['promoted']:9d}")
    print()
    for pl in (False, True):
        r = run(stream, n_slots, 400, ranked_ho, per_layer=pl, n_layers=n_layers)
        print(f"  pool {'per-layer' if pl else 'global   '}  coverage {r['coverage']:6.2f}%  "
              f"miss/tok {r['miss_per_tok']:6.2f}  GB/tok {r['gb_per_tok']:6.3f}")
    print("== ALL DONE ==")
