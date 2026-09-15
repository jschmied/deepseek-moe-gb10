#!/usr/bin/env python3
"""eviction_oracle.py -- is the LRU->Belady gap reachable by a better EVICTION ORDER?

`tools/prefetch_oracle.py` closed the prefetch side on this same decode log: short-history
predictors name 0.0 % of LRU's misses, because LRU already retains ~57 tokens of history and the
misses are pairs unused for a median of 45 tokens. But Belady is not a prefetcher -- it never
fetches anything early, it only picks a better VICTIM. So the entire 61 %-less-NVMe gap is an
eviction-ordering gap, and eviction has economics that make weak signals usable: a wrong eviction
costs the opportunity of having kept something better, a wrong prefetch costs a 13,774,848 B read.

This tool replays the cache and, at every eviction, ranks the K coldest resident entries by a score
built from PAST INFORMATION ONLY and evicts the top-ranked one. Small K keeps it obviously cheap:
`_lru_slot_for` would walk K slots off the cold end of the LRU list, not all 5,328. K is swept,
because how a score behaves as it is allowed to see more of the cache is itself the finding -- the
winner keeps improving to K = every slot, and LFU inverts.

Scores (cheapest first):

  1 lru             age since last use                     -- the baseline we ship
  2 lfu             use count (ties broken by age)
  3 age/(1+freq)    age discounted by frequency, GDSF-style without the size/cost terms
  4 median-gap      this entry's median inter-arrival so far; evict the slowest rhythm
  5 age/median-gap  normalised: evict the entry most OVERDUE relative to its own rhythm
  6 hazard(H)       P(no use in the next H tokens | survived `age` already), from that entry's own
                    gap history via its empirical survival function; H swept
  7 ttnu            median-gap - age = estimated tokens until the next use; evict the largest.
                    This is the direct plug-in estimator OF BELADY and is the arm the motivating
                    intuition actually asks for: not how OFTEN an entry is used, but WHEN it returns.
  -- belady         true furthest-next-use, uses the future, not implementable: the ceiling

HOW NO-FUTURE-LEAKAGE IS GUARANTEED
  * Every per-entry statistic (last_acc, last_tok, count, sorted gap list) is written in exactly one
    place: `touch()`, called at the moment the access is processed, from indices <= the current one.
  * The two priors a score can fall back to (global median gap, global gap survival function) are
    fitted on the warm-up prefix only, before the scored window begins, and are frozen after.
  * The scores are pure functions of (now, entry stats) -- they receive no sequence and no index.
  * Belady lives in a separate function that never shares state with the replay loop.
  * Sanity check, printed: the `lru` score arm must reproduce plain LRU BIT-EXACTLY (score = age is
    monotone in LRU position, so the top-ranked of the K coldest IS the LRU head). If that line does
    not say IDENTICAL, the harness is broken and nothing below means anything.

  Caveat stated up front: H for the hazard arm is swept and the best H reported, on the same scored
  window. That is hyperparameter selection on the test set, so the hazard row is an OPTIMISTIC upper
  bound, not a held-out result.

PART 2 -- THE WORKLOAD PART 1 COULD NOT SEE (--mode switched)

  Part 1 replays ONE continuous decode stream, and on that workload age/(1+count) recovers 31 % of
  the Belady gap. The server A/B agreed on decode (+15 %) and then, on the third prefill repetition
  of the same arm, read 275.5 GB where the previous repetition read 87.6 GB. A single continuous
  stream is structurally blind to that: use counts persist across requests, prefill hits do not tick
  them, so a count only becomes WRONG once the working set changes -- which never happens here.

  Part 2 therefore replays SEVERAL requests back to back (--pattern ABABA over two captures of
  different prompts), with nothing reset between them, and models what the engine actually does at a
  request boundary: the prefill, the 400-slot transient ring its misses land in, the promotions
  decode makes out of that ring, and the fact that a prefill hit refreshes self.lru's order but not
  _last_acc. It measures prefill NVMe reads per request -- the axis that blew up -- next to the
  decode axis part 1 optimised.

  The decays it sweeps all keep the count an INTEGER with FEW values, because that is what makes the
  shipped victim search exact and cheap (compare bucket heads, ~228 per eviction, not 5,328 slots):
      A  cap      score = age / (1 + min(count, C)),  C swept; C=0 is LRU, C=inf is today
      B  halve    count //= 2 for every counter every H decode ticks, H swept
      C  2-gen    uses in the last W ticks as (this window + previous window), capped
      D  per-req  decay at the request boundary instead of on a timer -- the server knows where one
                  is, so this needs no swept constant at all
      E  pf-age   orthogonal: let a prefill hit refresh last_acc (never the count), which is what
                  the shipped LRU path gets for free and the bucket path does not
  --verify checks every one of them against a brute-force argmax over all residents, and checks that
  cap C=0 + pf-age reproduces the shipped LRU path bit-exactly.

Usage:
    eviction_oracle.py ~/ds41-queue/logs/route-decode.jsonl [--slots 5328 3000] [--k 32 64]
    eviction_oracle.py A.jsonl --log-b B.jsonl --mode switched --pattern ABABA [--verify]
    eviction_oracle.py A.jsonl --mode decay-continuous      # the part-2 arms on part 1's workload
CPU only; no GPU, no server.
"""

from __future__ import annotations

import argparse
import bisect
import collections
import heapq
import itertools
import json
import sys

N_EXPERTS = 384          # DS4.1 routed experts per MoE layer
N_LAYERS = 40
EXPERT_BYTES = 13_774_848   # one native CB3 disk record -- what a miss actually reads


# --------------------------------------------------------------------------- data

def load(path: str):
    """-> [(layer, (expert ids,)), ...] for decode calls only, in forward order.

    Same loader as tools/prefetch_oracle.py: the log is strictly cyclic 0..39 after the prefill
    lines, so call index // 40 is the token and % 40 is the layer.
    """
    calls = []
    for line in open(path):
        line = line.strip()
        if not line:
            continue
        try:
            o = json.loads(line)
        except json.JSONDecodeError:
            continue
        if o.get("pf"):
            continue
        calls.append((o["L"], tuple(o["uniq"])))
    return calls


# --------------------------------------------------------------------------- per-entry stats

class Stats:
    """Everything a policy may look at. Written only by touch(), only from the present."""
    __slots__ = ("last_acc", "last_tok", "count", "gaps")

    def __init__(self):
        self.last_acc = -1
        self.last_tok = -1
        self.count = 0
        self.gaps = []          # sorted token gaps between consecutive uses of this pair

    def touch(self, acc: int, tok: int):
        if self.count and tok > self.last_tok:
            bisect.insort(self.gaps, tok - self.last_tok)
        self.count += 1
        self.last_acc = acc
        self.last_tok = tok

    def median_gap(self, prior: float) -> float:
        g = self.gaps
        n = len(g)
        if n == 0:
            return prior
        return g[n // 2] if n & 1 else 0.5 * (g[n // 2 - 1] + g[n // 2])


# --------------------------------------------------------------------------- scores
# Each takes (stats, now_acc, now_tok, ctx) and returns a float. HIGHEST score is evicted.
# ctx carries only prefix-fitted priors. None of them can see the future.

def s_lru(st, acc, tok, ctx):
    return acc - st.last_acc


def s_lfu(st, acc, tok, ctx):
    # lowest use count first; age only as a tie-break, scaled so it can never outvote a count step
    return -st.count + (acc - st.last_acc) * 1e-9


def s_age_over_freq(st, acc, tok, ctx):
    return (acc - st.last_acc) / (1.0 + st.count)


def s_median_gap(st, acc, tok, ctx):
    return st.median_gap(ctx["gap_prior"])


def s_age_over_median_gap(st, acc, tok, ctx):
    return (tok - st.last_tok) / max(1e-9, st.median_gap(ctx["gap_prior"]))


def s_ttnu(st, acc, tok, ctx):
    # estimated tokens until the next use = own median rhythm minus how long it has already waited
    return st.median_gap(ctx["gap_prior"]) - (tok - st.last_tok)


def make_hazard(H: int):
    """P(gap > age + H | gap > age) from this entry's own empirical gap distribution.

    Entries with too little history fall back to the global prefix-fitted survival function. A high
    value = 'unlikely to come back within H tokens' = evict.
    """
    def score(st, acc, tok, ctx):
        age = tok - st.last_tok
        g = st.gaps
        if len(g) >= 3:
            surv = len(g) - bisect.bisect_right(g, age)
            if surv == 0:                       # already older than anything it has ever done
                return 1.0 + age * 1e-9         # strongly evictable, oldest first
            beyond = len(g) - bisect.bisect_right(g, age + H)
            return beyond / surv
        G = ctx["gap_pool"]
        surv = len(G) - bisect.bisect_right(G, age)
        if surv == 0:
            return 1.0 + age * 1e-9
        beyond = len(G) - bisect.bisect_right(G, age + H)
        return beyond / surv
    return score


# --------------------------------------------------------------------------- replay

def replay(calls, slots, cut, score, K):
    """LRU list + K-candidate re-ranking at eviction. Returns (hits, fetches, scored_accesses).

    The access sequence is identical for every policy (no prefetch, no lookahead), so fetches ==
    misses and the byte axis and the hit axis are the same number.
    """
    cache = collections.OrderedDict()          # key -> 1, ordered least-recently-used first
    stats = collections.defaultdict(Stats)
    ctx = {"gap_prior": 0.0, "gap_pool": []}
    hits = fetches = scored = 0
    acc = 0

    for i, (L, ex) in enumerate(calls):
        if i == cut:
            # Freeze the priors here, from prefix history only, and never update them again.
            pool = []
            for st in stats.values():
                pool.extend(st.gaps)
            pool.sort()
            ctx["gap_pool"] = pool
            ctx["gap_prior"] = pool[len(pool) // 2] if pool else 1.0
        tok = i // N_LAYERS
        scoring = i >= cut
        for e in ex:
            k = L * N_EXPERTS + e
            acc += 1
            if k in cache:
                cache.move_to_end(k)
                if scoring:
                    hits += 1
                    scored += 1
            else:
                if len(cache) >= slots:
                    if K <= 1:          # fast path, used only for the plain-LRU baseline
                        victim = next(iter(cache))
                    else:
                        best = None
                        best_s = None
                        for c in itertools.islice(cache, K):
                            s = score(stats[c], acc, tok, ctx)
                            if best_s is None or s > best_s:
                                best_s, best = s, c
                        victim = best
                    del cache[victim]
                cache[k] = 1
                if scoring:
                    fetches += 1
                    scored += 1
            stats[k].touch(acc, tok)
    return hits, fetches, scored


def replay_buckets(calls, slots, cut):
    """EXACT global argmax of age/(1+freq) over the WHOLE cache, in O(#distinct use counts).

    Why this matters: the K-scan arms show age/(1+freq) still improving at K = every slot, which
    would be a 5,328-entry scan per eviction if you did it naively. You do not have to. Bucket the
    resident entries by use count and keep each bucket in LRU order; within a bucket the count is
    constant, so the maximum of (now - last_acc)/(1 + count) is the bucket's LRU head. The global
    winner is therefore the best of the bucket heads -- a few hundred comparisons, next to a
    13.1 MiB NVMe read. This is what would go into `_lru_slot_for`, and it is EXACT, not an
    approximation: --verify-buckets checks it against the brute-force scan.

    Returns (hits, fetches, scored, mean bucket heads examined per eviction).
    """
    buckets = collections.defaultdict(collections.OrderedDict)   # count -> {key: 1} LRU-ordered
    where = {}                                                   # key -> its count bucket
    stats = collections.defaultdict(Stats)
    hits = fetches = scored = evictions = examined = 0
    acc = 0
    resident = 0
    for i, (L, ex) in enumerate(calls):
        tok = i // N_LAYERS
        scoring = i >= cut
        for e in ex:
            k = L * N_EXPERTS + e
            acc += 1
            if k in where:
                c = where.pop(k)
                del buckets[c][k]
                if scoring:
                    hits += 1
                    scored += 1
            else:
                if resident >= slots:
                    best = None
                    best_key = (None, None)
                    n_heads = 0
                    for c, b in buckets.items():
                        if not b:
                            continue
                        n_heads += 1
                        head = next(iter(b))
                        sc = (acc - stats[head].last_acc) / (1.0 + c)
                        key = (sc, -stats[head].last_acc)
                        if best is None or key > best_key:
                            best, best_key = head, key
                    del buckets[where.pop(best)][best]
                    resident -= 1
                    evictions += 1
                    examined += n_heads
                if scoring:
                    fetches += 1
                    scored += 1
                resident += 1
            stats[k].touch(acc, tok)
            c = stats[k].count
            buckets[c][k] = 1
            where[k] = c
    return hits, fetches, scored, examined / max(1, evictions)


def belady(calls, slots, cut):
    """Evict the resident pair whose next use is furthest away. Lifted from prefetch_oracle.py."""
    seq = [L * N_EXPERTS + e for L, ex in calls for e in ex]
    cut_acc = sum(len(ex) for _, ex in calls[:cut])
    nxt = [0] * len(seq)
    last = {}
    for i in range(len(seq) - 1, -1, -1):
        nxt[i] = last.get(seq[i], len(seq))
        last[seq[i]] = i
    cache, heap, hits, fetches = {}, [], 0, 0
    for i, k in enumerate(seq):
        if k in cache:
            hits += (i >= cut_acc)
        else:
            if len(cache) >= slots:
                while True:
                    neg, kk = heapq.heappop(heap)
                    if cache.get(kk) == -neg:
                        del cache[kk]
                        break
            fetches += (i >= cut_acc)
        cache[k] = nxt[i]
        heapq.heappush(heap, (-nxt[i], k))
    return hits, fetches, len(seq) - cut_acc


def miss_anatomy(calls, slots, cut):
    """Of LRU's scored misses: how many are COMPULSORY (pair never used before, in the whole log)?

    Compulsory misses are unreachable by ANY eviction policy -- you cannot keep what you have never
    seen -- so they bound the achievable ceiling. Also returns how stale the non-compulsory ones are.
    """
    cache = collections.OrderedDict()
    last_tok = {}
    misses = compulsory = 0
    dists = []
    for i, (L, ex) in enumerate(calls):
        tok = i // N_LAYERS
        scoring = i >= cut
        for e in ex:
            k = L * N_EXPERTS + e
            if k in cache:
                cache.move_to_end(k)
            else:
                if len(cache) >= slots:
                    cache.popitem(last=False)
                cache[k] = 1
                if scoring:
                    misses += 1
                    if k in last_tok:
                        dists.append(tok - last_tok[k])
                    else:
                        compulsory += 1
            last_tok[k] = tok
    dists.sort()
    pct = lambda q: dists[min(len(dists) - 1, int(q * len(dists)))] if dists else 0
    return misses, compulsory, pct(0.10), pct(0.50), pct(0.90)


# ===========================================================================================
# PART 2 -- CROSS-REQUEST COUNT POLLUTION, AND INTEGER DECAYS THAT SURVIVE THE BUCKET TRICK
# ===========================================================================================
# Part 1 (above) replays ONE continuous decode stream. That workload structurally cannot see the
# defect this part is about: inside a single request a use count that never decays is never WRONG,
# only old. The server A/B disagreed with part 1 exactly there -- decode +15 %, but one prefill
# repetition of the age_over_freq arm read 275.5 GB against its own previous 87.6 GB.
#
# WHAT THE ENGINE ACTUALLY DOES (engine/experts.py of deepseek-v41-flash-spark, read, not assumed):
#   * LRU pool = 5,328 slots. Transient ring = 400 slots, FIFO; prefill misses land there.
#   * decode hit on an LRU resident : clock += 1, count += 1, last_acc = clock.
#   * decode hit on a RING resident : promoted into the LRU without a re-read -- and the donor slot
#     the ring gets back has to come out of the LRU, so a promotion also EVICTS.
#   * decode miss                   : clock += 1, score the bucket heads, evict, insert, touch.
#   * prefill hit on an LRU resident: self.lru.move_to_end(key) and NOTHING else. Deliberate: a
#     prefill chunk touches nearly every expert of a layer, so ticking the counter there would push
#     every resident up by one per chunk and drown the decode signal the policy was fitted on.
#   * prefill miss                  : transient ring. Never evicts from the LRU, never counted.
#
# That last pair hides a SECOND pollution channel, independent of the counts, because the two
# shipped policies read recency out of two different places:
#   - policy "lru"           takes its victim from self.lru, whose order prefill hits DO refresh;
#   - policy "age_over_freq" takes its victim from _last_acc, which only decode ever writes.
# So under the shipped LRU a prefill protects its own working set for the request that follows, and
# under age_over_freq it does not. The arms below separate the two channels rather than assume
# which one fired: `pf-age` = let a prefill hit refresh last_acc (still no count tick).
#
# All of it is modelled here, ring and promotions included, because the quantity that blew up in
# the A/B is prefill NVMe bytes, and prefill bytes are decided entirely by what the PREVIOUS
# request's eviction policy left resident.

RING_SLOTS = 400          # engine default transient_slots


def load_requests(path: str):
    """-> (prefill calls, decode calls) for one logged request.

    A prefill line's `uniq` is one resolve() call = one (layer, chunk); these two captures have
    exactly one chunk per layer, so the prefill is 40 calls of ~86-90 experts each.
    """
    pf, dec = [], []
    for line in open(path):
        line = line.strip()
        if not line:
            continue
        try:
            o = json.loads(line)
        except json.JSONDecodeError:
            continue
        (pf if o.get("pf") else dec).append((o["L"], tuple(o["uniq"])))
    return pf, dec


def build_workload(streams: dict, pattern: str, pf_chunks: int = 1):
    """pattern 'ABABA' -> [(tag, prefill, decode, is_new_request), ...].

    A request is its prefill followed by its decode, and NOTHING is reset between requests: not the
    cache, not the ring, not the use counts, not the clock. That is the point -- the server does not
    reset them either, and `_use_count` outlives eviction by design.

    pf_chunks > 1 repeats the 40-layer prefill sequence, CHUNK-major (all 40 layers, then all 40
    again). That is the real prefill's shape and the reason prefill can re-read: between two chunks
    of layer L, 39 other layers' misses cycle through the 400-slot transient ring, so whatever
    layer L put there is gone and is read again. The captures have exactly one chunk per layer, so
    every chunk here requests the same per-layer union -- which is precisely the engine's own
    description of a prefill re-read ("`uniq` seen again in a later chunk of the SAME layer"), but
    it is an approximation of chunk-to-chunk routing drift and is only used as a sensitivity knob.
    """
    return [(c, streams[c][0] * pf_chunks, streams[c][1], True) for c in pattern]


# --------------------------------------------------------------------------- integer decays
# Every policy here maps a resident to a SMALL INTEGER bucket. Not a style preference: the shipped
# victim search compares only bucket HEADS (227.8 comparisons per eviction instead of 5,328) and it
# is exact only because the score is monotone in age WITHIN a bucket. Float counts would make
# thousands of singleton buckets and turn the search back into a full scan, so an exponential decay
# -- the obvious fix -- is not shippable here at any quality.

class Decay:
    """Base = today's behaviour: count = uses ever, bucket = count, no decay at all."""

    def __init__(self, name, pf_refresh=False):
        self.name = name
        self.pf_refresh = pf_refresh      # may a prefill hit refresh last_acc? (never the count)
        self.count = collections.defaultdict(int)
        self.rebuilds = 0
        self.decayed = 0                  # counters rewritten by the decay, total

    def touch(self, key):
        self.count[key] += 1

    def bucket(self, key):
        return self.count[key]

    def after_tick(self, clock):
        return False

    def on_request(self):
        return False


class Cap(Decay):
    """A. score = age / (1 + min(count, C)). Buckets 0..C, so at most C+1 of them, forever.

    C = 0 is age-only (LRU read off the decode clock), C = None is today's unbounded count.
    """

    def __init__(self, C, pf_refresh=False):
        super().__init__(f"cap C={'inf' if C is None else C}", pf_refresh)
        self.C = C

    def bucket(self, key):
        c = self.count[key]
        return c if self.C is None or c < self.C else self.C


class Halve(Decay):
    """B. every H decode ticks, count //= 2 for EVERY counter, resident or not.

    Not only residents: `_use_count` is keyed by (layer, expert) and survives eviction, so an entry
    that is evicted and re-fetched comes back carrying its old count. Halving just the residents
    would leave exactly the stale counters that cause the problem. The table is <= 15,360 entries.
    """

    def __init__(self, H, C=None, pf_refresh=False):
        super().__init__(f"halve H={H // 1000}k" + ("" if C is None else f" cap {C}"), pf_refresh)
        self.H = H
        self.C = C

    def bucket(self, key):
        c = self.count[key]
        return c if self.C is None or c < self.C else self.C

    def after_tick(self, clock):
        if clock % self.H:
            return False
        n = 0
        for k, v in self.count.items():
            if v:
                self.count[k] = v >> 1
                n += 1
        self.rebuilds += 1
        self.decayed += n
        return True


class ReqDecay(Decay):
    """D. decay at the REQUEST BOUNDARY instead of on a timer.

    The server knows where a request starts -- that is the event the pollution is defined against,
    so this needs no clock, no threshold and no swept constant. 'halve' -> //= 2, 'reset' -> 0.
    """

    def __init__(self, mode, C=None, pf_refresh=False):
        super().__init__(f"per-request {mode if isinstance(mode, str) else '>>%d' % mode}"
                         + ("" if C is None else f" cap {C}"), pf_refresh)
        self.mode = mode          # "reset", "halve", or an integer right-shift depth
        self.C = C

    def bucket(self, key):
        c = self.count[key]
        return c if self.C is None or c < self.C else self.C

    def on_request(self):
        n = len(self.count)
        if self.mode == "reset":
            self.count.clear()
        else:
            sh = 1 if self.mode == "halve" else self.mode
            for k, v in self.count.items():
                if v:
                    self.count[k] = v >> sh
        self.rebuilds += 1
        self.decayed += n
        return True


class TwoGen(Decay):
    """C. 'uses in the last W ticks', carried as two small integers: this window and the previous.

    bucket = min(cur + prev, C). One counter table is swapped and one cleared per window boundary
    instead of every counter being rewritten, and the value stays an integer bounded by C. This is
    the windowed-frequency reading of the brief, not an approximation of it.
    """

    def __init__(self, W, C=8, pf_refresh=False):
        super().__init__(f"2-gen W={W // 1000}k cap {C}", pf_refresh)
        self.W = W
        self.C = C
        self.prev = collections.defaultdict(int)

    def bucket(self, key):
        c = self.count[key] + self.prev[key]
        return c if c < self.C else self.C

    def after_tick(self, clock):
        if clock % self.W:
            return False
        self.prev = self.count
        self.count = collections.defaultdict(int)
        self.rebuilds += 1
        return True


# --------------------------------------------------------------------------- the request-level sim

class Sim:
    """LRU pool + transient ring + count buckets, with the engine's exact access semantics.

    victim_mode:
      'lru'    -- head of self.lru: the shipped default path. self.lru's order IS refreshed by
                  prefill hits, which is the channel the bucket policies do not have.
      'bucket' -- argmax of (clock - last_acc) / (1 + bucket) over bucket heads: the shipped
                  age_over_freq path. last_acc is written by decode only, plus prefill hits iff the
                  policy sets pf_refresh.
      'belady' -- furthest next use among residents. Uses the future; a ceiling, not a policy.
      'brute'  -- same score as 'bucket' but by scanning every resident. Only for --verify.
    """

    def __init__(self, slots, ring_slots, policy, victim_mode, nxt=None):
        self.slots = slots
        self.ring_slots = ring_slots
        self.pol = policy
        self.mode = victim_mode
        self.lru = collections.OrderedDict()          # key -> True, LRU order incl. prefill hits
        self.ring = collections.OrderedDict()         # key -> True, FIFO, no refresh on hit
        self.last_acc = {}                            # decode clock of the last decode touch
        self.buckets = collections.defaultdict(collections.OrderedDict)
        self.where = {}
        self.clock = 0
        self.ai = 0                                   # index into the flat access sequence
        self.nxt = nxt
        self.nextuse = {}
        self.heap = []
        self.evictions = 0
        self.examined = 0
        self.max_heads = 0
        self.skips = 0

    # ---- bucket bookkeeping
    def _seat(self, key):
        b = self.pol.bucket(key)
        old = self.where.get(key)
        if old is not None:
            bb = self.buckets[old]
            bb.pop(key, None)
            if not bb:
                del self.buckets[old]
        self.buckets[b][key] = True              # appended = newest of its bucket
        self.where[key] = b

    def _unseat(self, key):
        b = self.where.pop(key, None)
        if b is None:
            return
        bb = self.buckets[b]
        bb.pop(key, None)
        if not bb:
            del self.buckets[b]

    def rebuild(self):
        """Every bucket index may have changed at once. Re-seat residents in last_acc order so each
        bucket stays ordered oldest-first, which is what makes its head that bucket's maximum."""
        res = sorted(self.where, key=lambda k: self.last_acc[k])
        self.buckets = collections.defaultdict(collections.OrderedDict)
        self.where = {}
        for k in res:
            b = self.pol.bucket(k)
            self.buckets[b][k] = True
            self.where[k] = b

    # ---- victim selection
    def _victim(self, used):
        if self.mode == "lru":
            for k in self.lru:
                if k not in used:
                    return k
                self.skips += 1
            return None
        if self.mode == "belady":
            parked, victim = [], None
            while self.heap:
                neg, k = heapq.heappop(self.heap)
                if k not in self.lru or self.nextuse.get(k) != -neg:
                    continue
                if k in used:
                    parked.append((neg, k))
                    self.skips += 1
                    continue
                victim = k
                break
            for item in parked:
                heapq.heappush(self.heap, item)
            return victim
        now = self.clock
        best, best_score, best_acc = None, -1.0, 0
        if self.mode == "brute":
            for k in self.where:
                if k in used:
                    continue
                acc = self.last_acc[k]
                sc = (now - acc) / (1.0 + self.pol.bucket(k))
                if best is None or sc > best_score or (sc == best_score and acc < best_acc):
                    best, best_score, best_acc = k, sc, acc
            self.examined += len(self.where)
            return best
        heads = 0
        for c, b in self.buckets.items():
            for k in b:                          # oldest first: the first ELIGIBLE entry is this
                heads += 1                       # bucket's maximum under the `used` constraint
                if k in used:
                    self.skips += 1
                    continue
                acc = self.last_acc[k]
                sc = (now - acc) / (1.0 + c)
                if best is None or sc > best_score or (sc == best_score and acc < best_acc):
                    best, best_score, best_acc = k, sc, acc
                break
        self.examined += heads
        if heads > self.max_heads:
            self.max_heads = heads
        return best

    def _insert(self, key, used):
        if len(self.lru) >= self.slots:
            v = self._victim(used)
            if v is None:
                raise RuntimeError("every resident is protected by this call")
            del self.lru[v]
            self._unseat(v)
            self.nextuse.pop(v, None)
            self.evictions += 1
        self.lru[key] = True

    def _future(self, key):
        """Record this access's next-use for the Belady arm. Only for LRU residents."""
        if self.nxt is None:
            return
        n = self.nxt[self.ai]
        self.nextuse[key] = n
        if self.mode == "belady":
            heapq.heappush(self.heap, (-n, key))

    # ---- the two access paths
    def prefill_call(self, L, uniq, st):
        for e in uniq:
            key = L * N_EXPERTS + e
            if key in self.lru:
                self.lru.move_to_end(key)             # engine: move_to_end and nothing else
                st["pf_hits"] += 1
                if self.pol.pf_refresh:
                    self.last_acc[key] = self.clock
                    self._seat(key)
                self._future(key)
            elif key in self.ring:
                st["pf_hits"] += 1                    # ring hit: no read, and no refresh (FIFO)
            else:
                st["pf_fetches"] += 1
                if len(self.ring) >= self.ring_slots:
                    self.ring.popitem(last=False)
                self.ring[key] = True
            self.ai += 1

    def decode_call(self, L, ex, st):
        used = set()
        for e in ex:
            key = L * N_EXPERTS + e
            self.clock += 1
            if key in self.lru:
                self.lru.move_to_end(key)
                st["hits"] += 1
            elif key in self.ring:
                del self.ring[key]                    # promotion: a hit, and also an eviction
                st["hits"] += 1
                st["promoted"] += 1
                self._insert(key, used)
            else:
                st["fetches"] += 1
                self._insert(key, used)
            self.pol.touch(key)
            self.last_acc[key] = self.clock
            self._seat(key)
            self._future(key)
            used.add(key)
            self.ai += 1
            if self.pol.after_tick(self.clock):
                self.rebuild()


def flatten(workload):
    """The exact access order the sim walks, turned into a next-use table for the Belady arm."""
    seq = []
    for _, pf, dec, _ in workload:
        for L, uniq in pf:
            seq.extend(L * N_EXPERTS + e for e in uniq)
        for L, ex in dec:
            seq.extend(L * N_EXPERTS + e for e in ex)
    nxt = [0] * len(seq)
    last = {}
    for i in range(len(seq) - 1, -1, -1):
        nxt[i] = last.get(seq[i], len(seq))
        last[seq[i]] = i
    return nxt


def run_workload(workload, slots, policy, victim_mode, warm, nxt=None):
    """Replay the whole multi-request workload. -> (per-request rows, totals over scored requests)."""
    sim = Sim(slots, RING_SLOTS, policy, victim_mode, nxt)
    rows = []
    tot = {"hits": 0, "fetches": 0, "pf_fetches": 0, "pf_hits": 0, "promoted": 0, "tokens": 0.0}
    for r, (tag, pf, dec, is_new) in enumerate(workload):
        if r and is_new and policy.on_request():
            sim.rebuild()
        st = collections.Counter()
        for L, uniq in pf:
            sim.prefill_call(L, uniq, st)
        e0, x0 = sim.evictions, sim.examined
        for L, ex in dec:
            sim.decode_call(L, ex, st)
        toks = len(dec) / N_LAYERS
        rows.append({"r": r, "tag": tag, "warm": r < warm, "tokens": toks,
                     "pf_fetches": st["pf_fetches"], "pf_hits": st["pf_hits"],
                     "hits": st["hits"], "fetches": st["fetches"], "promoted": st["promoted"],
                     "heads": (sim.examined - x0) / max(1, sim.evictions - e0)})
        if r >= warm:
            for k in ("hits", "fetches", "pf_fetches", "pf_hits", "promoted"):
                tot[k] += st[k]
            tot["tokens"] += toks
    tot["max_heads"] = sim.max_heads
    tot["skips"] = sim.skips
    tot["heads"] = sim.examined / max(1, sim.evictions)
    tot["rebuilds"] = policy.rebuilds
    tot["decayed"] = policy.decayed
    return rows, tot


# --------------------------------------------------------------------------- main

def continuous_study(a) -> int:
    calls = load(a.log)
    if not calls:
        print("no decode calls in log", file=sys.stderr)
        return 1
    accesses = sum(len(ex) for _, ex in calls)
    tokens = len(calls) / N_LAYERS
    cut = int(len(calls) * a.prefix) // N_LAYERS * N_LAYERS       # whole-token boundary
    tok_scored = (len(calls) - cut) / N_LAYERS

    print(f"  {len(calls):,} decode resolve calls, {accesses:,} accesses, {tokens:.0f} tokens, "
          f"{accesses / tokens:.0f} accesses/token")
    print(f"  warm every policy on calls [0,{cut:,}) = {100 * cut / len(calls):.0f} %, score the "
          f"rest ({len(calls) - cut:,} calls, {tok_scored:.0f} tokens)")
    print(f"  one fetch = {EXPERT_BYTES / 2**20:.2f} MiB of NVMe; no policy here prefetches, so "
          f"fetches == misses and MiB/token is just the miss axis rescaled\n")

    for slots in a.slots:
        print(f"  ===== cache {slots:,} slots "
              f"({100 * slots / (N_LAYERS * N_EXPERTS):.1f} % of the 15,360 pairs) =====")

        h_lru, f_lru, scored = replay(calls, slots, cut, s_lru, 1)
        hb, fb, _ = belady(calls, slots, cut)
        base_hit, bel_hit = h_lru / scored, hb / scored

        nm, comp, d10, d50, d90 = miss_anatomy(calls, slots, cut)
        print(f"  compulsory misses: {comp:,} of {nm:,} LRU misses = {100 * comp / nm:.1f} % "
              f"(first ever use of that pair -- unreachable by any eviction policy)")
        print(f"  the other {100 * (1 - comp / nm):.1f} % were last used {d10} / {d50} / {d90} "
              f"tokens ago (p10/p50/p90): the reachable set.")
        print(f"  Belady removes {f_lru - fb:,} of those {nm - comp:,} reachable misses "
              f"= {100 * (f_lru - fb) / (nm - comp):.0f} % of them.")
        print(f"  (the COUNT of compulsory misses is a property of the trace and the prefix, not of"
              f" the cache: 2,433 at this 25 % prefix for every size. The PERCENTAGE moves only"
              f" because the miss denominator does -- 7.1 % at 4,788 slots, {100 * comp / nm:.1f} %"
              f" here. Earlier work's 7.1 % was belady_replay.py's old 4,788-slot default.)\n")

        arms = [("1 LRU  (age)", s_lru),
                ("2 LFU  (count, age tie-break)", s_lfu),
                ("3 age / (1 + freq)", s_age_over_freq),
                ("4 median-gap", s_median_gap),
                ("5 age / median-gap", s_age_over_median_gap),
                ("7 ttnu = median-gap - age", s_ttnu)]
        for H in a.hazard_h:
            arms.append((f"6 hazard H={H}", make_hazard(H)))

        hdr = (f"  {'policy':34s} {'K':>3s} {'hit':>8s} {'fetch/tok':>10s} {'MiB/tok':>9s} "
               f"{'xLRU':>6s} {'gap recovered':>14s}")
        print(hdr)
        print(f"  {'LRU (shipped baseline)':34s} {'-':>3s} {100 * base_hit:7.2f} % "
              f"{f_lru / tok_scored:10.1f} {f_lru * EXPERT_BYTES / tok_scored / 2**20:9.0f} "
              f"{1.0:6.2f} {'0.0 %':>14s}")

        for K in a.k:
            for name, score in arms:
                h, f, sc = replay(calls, slots, cut, score, K)
                assert sc == scored
                hit = h / sc
                rec = (hit - base_hit) / (bel_hit - base_hit) if bel_hit > base_hit else 0.0
                tag = ""
                if score is s_lru:
                    tag = "  <- IDENTICAL to LRU" if (h, f) == (h_lru, f_lru) else "  <- HARNESS BUG"
                print(f"  {name:34s} {K:3d} {100 * hit:7.2f} % {f / tok_scored:10.1f} "
                      f"{f * EXPERT_BYTES / tok_scored / 2**20:9.0f} {f / f_lru:6.2f} "
                      f"{100 * rec:13.1f} %{tag}")
            print()

        hbk, fbk, sck, heads = replay_buckets(calls, slots, cut)
        hitbk = hbk / sck
        recbk = (hitbk - base_hit) / (bel_hit - base_hit)
        print(f"  {'3-exact age/(1+freq), ALL slots':34s} {'all':>3s} {100 * hitbk:7.2f} % "
              f"{fbk / tok_scored:10.1f} {fbk * EXPERT_BYTES / tok_scored / 2**20:9.0f} "
              f"{fbk / f_lru:6.2f} {100 * recbk:13.1f} %   count-bucketed, "
              f"{heads:.0f} comparisons/eviction")
        if a.verify_buckets:
            hs, fs, _ = replay(calls, slots, cut, s_age_over_freq, slots)
            print(f"      verify: brute-force K={slots} scan -> hits {hs:,} fetches {fs:,} ; "
                  f"buckets -> hits {hbk:,} fetches {fbk:,} : "
                  f"{'IDENTICAL' if (hs, fs) == (hbk, fbk) else 'MISMATCH'}")
        print()

        print(f"  {'Belady (future; the ceiling)':34s} {'-':>3s} {100 * bel_hit:7.2f} % "
              f"{fb / tok_scored:10.1f} {fb * EXPERT_BYTES / tok_scored / 2**20:9.0f} "
              f"{fb / f_lru:6.2f} {'100.0 %':>14s}\n")

    print("  gap recovered = (hit - LRU) / (Belady - LRU); equivalently the fraction of LRU's excess")
    print("  NVMe bytes removed, since no arm prefetches. Negative = worse than what we ship.")
    print("  Every score sees only (age, use count, own past gaps) plus two priors frozen at the end")
    print("  of the warm-up prefix. The hazard rows sweep H on the scored window and are therefore an")
    print("  optimistic bound on that family, not a held-out result.")
    print("  Scope: ONE trace (DS4.1 decode, 549 tokens, one prompt), these cache sizes. Nothing here")
    print("  says anything about prefill, other prompts, or other models.\n")
    print("  WHAT THIS FOUND. Eviction is NOT closed the way prefetch is. One score wins and it is the")
    print("  cheapest combination on the list: age / (1 + use count) -- frequency-discounted recency.")
    print("  It recovers ~3-5 % of the LRU->Belady gap at K = 32/64 and ~29-31 % when allowed to rank")
    print("  the whole cache, at BOTH sizes, monotonically in K. The exact global version is not a")
    print("  full scan: bucket by use count, LRU within the bucket, compare the bucket heads (~226")
    print("  comparisons/eviction at 3,000 slots, verified bit-identical to the brute-force scan).")
    print()
    print("  WHAT THIS REFUTED -- the motivating intuition, as stated, is WRONG on this trace. The")
    print("  'when does it return' scores (median-gap, age/median-gap, ttnu = median-gap - age) are")
    print("  at or BELOW LRU everywhere and collapse as K grows (-300 % of the gap at full K). A")
    print("  per-entry inter-arrival median does not predict this workload's returns: routing is")
    print("  near-uniform and an expert's gap distribution is wide, so its median is a bad point")
    print("  estimate and acting on it evicts entries that were about to be used. The frequency term")
    print("  is what carries -- but ONLY held against age: plain LFU also wins at small K and then")
    print("  inverts to -10 %/-15 % at full K, because unweighted frequency protects stale hot")
    print("  entries forever. The age numerator is load-bearing.")
    return 0


ARMS = [
        ("LRU  (shipped default)", lambda: Cap(0, pf_refresh=True), "lru", True),
        ("age/(1+count)  SHIPPED afq", lambda: Cap(None), "bucket", True),
        ("A  cap C=1", lambda: Cap(1), "bucket", False),
        ("A  cap C=2", lambda: Cap(2), "bucket", False),
        ("A  cap C=4", lambda: Cap(4), "bucket", False),
        ("A  cap C=8", lambda: Cap(8), "bucket", False),
        ("A  cap C=16", lambda: Cap(16), "bucket", False),
        ("A  cap C=32", lambda: Cap(32), "bucket", False),
        ("B  halve H=2k", lambda: Halve(2000), "bucket", False),
        ("B  halve H=8k", lambda: Halve(8000), "bucket", False),
        ("B  halve H=32k", lambda: Halve(32000), "bucket", False),
        ("B  halve H=128k", lambda: Halve(128000), "bucket", False),
        ("C  halve H=8k + cap 8", lambda: Halve(8000, 8), "bucket", False),
        ("C  2-gen W=8k cap 8", lambda: TwoGen(8000, 8), "bucket", False),
        ("C  2-gen W=32k cap 8", lambda: TwoGen(32000, 8), "bucket", False),
        ("D  per-request halve", lambda: ReqDecay("halve"), "bucket", False),
        ("D  per-request reset", lambda: ReqDecay("reset"), "bucket", False),
        ("E  afq + pf-age", lambda: Cap(None, pf_refresh=True), "bucket", True),
        ("E  cap C=4 + pf-age", lambda: Cap(4, pf_refresh=True), "bucket", False),
        ("E  halve H=8k + pf-age", lambda: Halve(8000, pf_refresh=True), "bucket", False),
        ("E  per-req reset + pf-age", lambda: ReqDecay("reset", pf_refresh=True), "bucket", False),
        ("cap C=0 (age only, no pf-age)", lambda: Cap(0), "bucket", False),
        ("F  per-req reset + cap 8", lambda: ReqDecay("reset", 8), "bucket", False),
        ("F  per-req reset + cap 16", lambda: ReqDecay("reset", 16), "bucket", False),
        ("F  per-req reset + cap 32", lambda: ReqDecay("reset", 32), "bucket", False),
        ("F  per-req reset + cap 64", lambda: ReqDecay("reset", 64), "bucket", False),
        ("F  per-req halve + cap 32", lambda: ReqDecay("halve", 32), "bucket", False),
        ("F  per-req reset cap 32 pf-age", lambda: ReqDecay("reset", 32, pf_refresh=True),
         "bucket", True),
        ("F  halve H=32k + cap 32", lambda: Halve(32000, 32), "bucket", False),
        ("G  per-req >>1 cap 32 pf-age", lambda: ReqDecay(1, 32, pf_refresh=True), "bucket", False),
        ("G  per-req >>2 cap 32 pf-age", lambda: ReqDecay(2, 32, pf_refresh=True), "bucket", False),
        ("G  per-req >>3 cap 32 pf-age", lambda: ReqDecay(3, 32, pf_refresh=True), "bucket", False),
        ("G  per-req >>2 cap 64 pf-age", lambda: ReqDecay(2, 64, pf_refresh=True), "bucket", False),
        ("G  per-req >>2 uncapped pf-age", lambda: ReqDecay(2, None, pf_refresh=True), "bucket", False),
    ]




def _fmt_req_table(rows):
    out = ["    req  stream  role    prefill fetch   prefill GB   decode hit   dec fetch/tok   "
           "promoted   heads/evict"]
    for r in rows:
        acc = r["hits"] + r["fetches"]
        hit = 100.0 * r["hits"] / max(1, acc)
        out.append(f"    {r['r']:3d}  {r['tag']:>6s}  {'warm' if r['warm'] else 'scored':6s} "
                   f"{r['pf_fetches']:14,d} {r['pf_fetches'] * EXPERT_BYTES / 1e9:12.1f} "
                   f"{hit:12.2f} % {r['fetches'] / max(1e-9, r['tokens']):14.1f} "
                   f"{r['promoted']:10,d} {r['heads']:13.0f}")
    return "\n".join(out)


def switched_study(a) -> int:
    streams = {}
    for tag, path in (("A", a.log), ("B", a.log_b)):
        if path is None:
            continue
        pf, dec = load_requests(path)
        streams[tag] = (pf, dec)
        print(f"  stream {tag} = {path}")
        print(f"    prefill {len(pf)} resolve calls, {sum(len(u) for _, u in pf):,} accesses, "
              f"{len(set((L, e) for L, u in pf for e in u)):,} distinct (layer,expert) pairs")
        print(f"    decode  {len(dec):,} resolve calls, {sum(len(u) for _, u in dec):,} accesses, "
              f"{len(dec) / N_LAYERS:.0f} tokens")
    missing = set(a.pattern) - set(streams)
    if missing:
        print(f"pattern {a.pattern} needs streams {sorted(missing)} that were not supplied",
              file=sys.stderr)
        return 1
    pa = set((L, e) for L, u in streams["A"][1] for e in u)
    if "B" in streams:
        pb = set((L, e) for L, u in streams["B"][1] for e in u)
        print(f"\n  decode working sets: |A| {len(pa):,}  |B| {len(pb):,}  |A n B| {len(pa & pb):,} "
              f"= {100 * len(pa & pb) / len(pa | pb):.1f} % of the union (Jaccard)")
        pfa = set((L, e) for L, u in streams["A"][0] for e in u)
        pfb = set((L, e) for L, u in streams["B"][0] for e in u)
        print(f"  prefill sets:        |A| {len(pfa):,}  |B| {len(pfb):,}  |A n B| {len(pfa & pfb):,} "
              f"= {100 * len(pfa & pfb) / len(pfa | pfb):.1f} %")

    workload = build_workload(streams, a.pattern, a.pf_chunks)
    warm = a.warm
    tok_scored = sum(len(d) / N_LAYERS for _, _, d, _ in workload[warm:])
    n_scored = len(workload) - warm
    print(f"\n  workload = {a.pattern}: {len(workload)} requests, each = its prefill then its decode."
          f"\n  Nothing is reset between them -- not the cache, not the ring, not the use counts, not"
          f"\n  the clock -- which is the only reason the defect can appear at all.")
    print(f"  warm on the first {warm} request(s), score the remaining {n_scored} "
          f"({tok_scored:.0f} decode tokens)\n")

    nxt = flatten(workload)

    arms = list(ARMS)
    if a.arms:
        want = set(a.arms)
        arms = [x for x in arms if any(w in x[0] for w in want)]

    results = {}
    print(f"  {'policy':30s} {'dec hit':>8s} {'dec f/tok':>10s} {'pf fetch/req':>13s} "
          f"{'worst pf':>9s} {'MiB/tok':>9s} {'xLRU B':>7s} {'dec gap':>8s} {'tot gap':>8s} "
          f"{'heads':>6s} {'max':>5s}")

    base = bel = None
    for label, mk, mode, _detail in arms:
        rows, tot = run_workload(workload, a.slots, mk(), mode, warm, None)
        results[label] = (rows, tot)
        if label.startswith("LRU"):
            base = tot
    rows_b, tot_b = run_workload(workload, a.slots, Cap(0, pf_refresh=True), "belady", warm, nxt)
    results["Belady (future; the ceiling)"] = (rows_b, tot_b)
    bel = tot_b

    def line(label, tot, rows):
        acc = tot["hits"] + tot["fetches"]
        hit = tot["hits"] / max(1, acc)
        tb = tot["fetches"] + tot["pf_fetches"]
        pf_req = tot["pf_fetches"] / n_scored
        worst = max(r["pf_fetches"] for r in rows if not r["warm"])
        bh = base["hits"] / max(1, base["hits"] + base["fetches"])
        beh = bel["hits"] / max(1, bel["hits"] + bel["fetches"])
        bt = base["fetches"] + base["pf_fetches"]
        bet = bel["fetches"] + bel["pf_fetches"]
        dgap = (hit - bh) / (beh - bh) if beh > bh else 0.0
        tgap = (bt - tb) / (bt - bet) if bt > bet else 0.0
        print(f"  {label:30s} {100 * hit:7.2f} % {tot['fetches'] / tok_scored:10.1f} "
              f"{pf_req:13,.0f} {worst:9,d} "
              f"{tb * EXPERT_BYTES / tok_scored / 2**20:9.0f} {tb / bt:7.2f} "
              f"{100 * dgap:7.1f} % {100 * tgap:7.1f} % {tot['heads']:6.0f} {tot['max_heads']:5d}")

    for label, _mk, _m, _d in arms:
        line(label, results[label][1], results[label][0])
    line("Belady (future; the ceiling)", tot_b, rows_b)

    print()
    print("  dec hit / dec f/tok  = DECODE accesses only, the axis part 1 optimised.")
    print("  pf fetch/req         = prefill NVMe reads per scored request -- the axis that blew up")
    print("                         in the A/B (87.6 GB -> 275.5 GB). 'worst pf' is the single worst")
    print("                         scored request, i.e. the offline analogue of 'rep 3'.")
    print("  MiB/tok / tot gap    = prefill AND decode bytes together, over decode tokens.")
    print("  heads / max          = bucket heads compared per eviction, mean and worst = the")
    print("                         implementation cost, and the number of live buckets.")
    print()

    for label, _mk, _m, detail in arms:
        if detail:
            print(f"  --- per request: {label}")
            print(_fmt_req_table(results[label][0]))
            print()
    if a.detail:
        for label in a.detail:
            for k in results:
                if label in k:
                    print(f"  --- per request: {k}")
                    print(_fmt_req_table(results[k][0]))
                    print()
    print("  --- per request: Belady (future; the ceiling)")
    print(_fmt_req_table(rows_b))
    print()

    if a.verify:
        print("  verify: bucket-head argmax vs a brute-force scan of every resident, on a truncated")
        print("  workload (the full one takes hours at 5,328 comparisons per eviction).")
        short = [(t, pf, d[:1500], n) for t, pf, d, n in workload]
        for label, mk in (("afq (cap inf)", lambda: Cap(None)),
                          ("cap C=4", lambda: Cap(4)),
                          ("halve H=8k", lambda: Halve(8000)),
                          ("2-gen W=8k cap 8", lambda: TwoGen(8000, 8)),
                          ("per-request reset", lambda: ReqDecay("reset"))):
            _, t1 = run_workload(short, a.slots, mk(), "bucket", warm)
            _, t2 = run_workload(short, a.slots, mk(), "brute", warm)
            same = (t1["hits"], t1["fetches"], t1["pf_fetches"]) == \
                   (t2["hits"], t2["fetches"], t2["pf_fetches"])
            print(f"    {label:22s} buckets {t1['hits']:,}/{t1['fetches']:,}  "
                  f"brute {t2['hits']:,}/{t2['fetches']:,}  "
                  f"{'IDENTICAL' if same else 'MISMATCH -- the shortcut is not exact'}")
        r1, _ = run_workload(workload, a.slots, Cap(0, pf_refresh=True), "lru", warm)
        r2, _ = run_workload(workload, a.slots, Cap(0, pf_refresh=True), "bucket", warm)
        same = all((x["hits"], x["fetches"], x["pf_fetches"]) ==
                   (y["hits"], y["fetches"], y["pf_fetches"]) for x, y in zip(r1, r2))
        print(f"    {'cap C=0 + pf-age':22s} must reproduce the shipped LRU path exactly: "
              f"{'IDENTICAL' if same else 'MISMATCH -- the harness is broken'}")
        print()
    print("  WHAT PART 2 FOUND -- 1. THE THRASH DOES NOT REPRODUCE OFFLINE, IN EITHER SWITCH SHAPE.")
    print("  Replayed at the shipped 5,328 slots with the real prefill, ring and promotion rules, the")
    print("  age_over_freq arm reads FEWER prefill bytes than LRU in EVERY scored request of both")
    print("  patterns, and the figure is flat across repetitions, not drifting:")
    print("      ABABA  prefill fetches/request   LRU 1,814 2,188 1,814 2,188 | afq 1,702 1,914 1,670 1,909")
    print("      AAAAAA prefill fetches/request   LRU 1,373 x5              | afq 1,288 1,287 1,287 1,288 1,288")
    print("  AAAAAA is the A/B's own shape -- the same prompt prefilled again and again with the cache,")
    print("  the ring and the counts carried over -- and there is no rep 3. Nothing here reaches even")
    print("  1.1x of its own previous repetition, let alone the A/B's 3.14x.")
    print()
    print("  So the stated diagnosis is NOT supported by replay. Two things do fit the A/B numbers and")
    print("  neither is an eviction-policy effect:")
    print("    * 275.5 / 87.6 = 3.14, and that prefill control is a 9,000-token prompt at")
    print("      DSV41_PREFILL_CHUNK=4096 = exactly 3 chunks. `--pf-chunks 3` reproduces that shape")
    print("      here: prefill fetches x3.00 for EVERY policy alike (LRU 1,373 -> 4,119, afq 1,288 ->")
    print("      3,863). A prefill that re-reads once per chunk is a transient-ring / chunk-ordering")
    print("      failure -- the ring is what makes chunk 2 of a layer free -- and the ring is policy-")
    print("      independent code.")
    print("    * that GB column is not the store's own counter. tools/longctx_profile.py reads")
    print("      /proc/diskstats for nvme0n1, so it charges the arm with every other read on the")
    print("      device in that window, swap included. The store's `stats['bytes_read']` was not the")
    print("      number reported.")
    print("  Before any of this is rewritten, re-run that control with stats['bytes_read'] logged per")
    print("  rep and DSV41_LAYER_MAJOR pinned, on an otherwise idle box. If bytes_read is flat, the")
    print("  275.5 GB was never the store's.")
    print()
    print("  2. WHAT DOES DEGRADE WITH UPTIME IS THE VICTIM SEARCH, NOT THE HIT RATE. Unbounded counts")
    print("  mean unbounded live buckets, and the bucket-head scan grows monotonically per request:")
    print("      AAAAAA heads/eviction  228 -> 528 -> 759 -> 951 -> 1,119 -> 1,271   (still climbing)")
    print("      ABABA  heads/eviction  228 -> 457 -> 626 -> 791 ->   932")
    print("  228 was measured on ONE request and is the number the shipped docstring quotes. The")
    print("  shortcut's whole claim is that it is O(distinct use counts) and not O(lru_slots) = 5,328;")
    print("  on this trajectory it reaches 5,328 after a few thousand more decode tokens of uptime.")
    print("  That is real, it is reproducible, and a cap fixes it for free.")
    print()
    print("  3. THE DECAY SWEEP, read across all four workloads (fraction of the LRU->Belady gap in")
    print("  TOTAL bytes; 'heads' is the live-bucket count = the implementation cost):")
    print()
    print("      policy                         cont-A  cont-B   ABABA  AAAAAA   heads (mean/max)")
    print("      LRU (shipped)                     0.0     0.0     0.0     0.0   0 / 0")
    print("      age/(1+count)  = afq today       31.1    26.3    17.8    42.4   191-804 / 335-1381 +")
    print("      A  cap C=32 alone                30.9    26.2    12.4    19.6   26-32 / 32-33")
    print("      B  halve H=32k                   19.2    20.5    16.1    20.3   50-54 / 71-74")
    print("      B  halve H=128k                  29.7    26.9    17.8    34.7   127-184 / 220-257")
    print("      C  2-gen W=32k cap 8             12.7    16.7    12.4    13.0   8 / 9")
    print("      D  per-request reset             31.1*   26.3*   24.1    27.0   191-228 / 335-408")
    print("      G  per-req >>2 cap 32 + pf-age   30.9    26.2    21.1    37.3   32 / 32-33")
    print("      F  per-req reset cap 32 + pf-age 30.9    26.2    24.9    27.0   30-32 / 32")
    print("      (+) still climbing at the end of the run.  (*) identical to afq BY CONSTRUCTION --")
    print("      a continuous trace has no request boundary, so a boundary decay never fires. That")
    print("      one line is the whole reason part 1 could not see any of this.")
    print()
    print("  Read it as three separate facts:")
    print("    * A CAP ALONE IS NOT FREE. cap C=32 costs 0.2 pp on a single request (counts barely")
    print("      reach 32 in 412 tokens) but 5 pp and 23 pp once several requests have run and counts")
    print("      have grown past the cap. Capping an undecayed counter throws away live ranking.")
    print("    * A TIMER IS THE WRONG CLOCK. halve H=2k/8k degenerate to LRU (-1.0 to +0.1 %) and")
    print("      H=128k degenerates to afq. Nothing in between beats a decay tied to the request")
    print("      boundary, because the event the count goes stale at is a request switch, not the")
    print("      passage of 32,768 ticks. The server already knows where a boundary is; no swept")
    print("      constant is needed. Same for 2-gen: its window is the same wrong clock.")
    print("    * THE DECAY IS WHAT MAKES THE CAP FREE. Decay at the boundary keeps counts inside the")
    print("      cap's range, so C=32 stops binding: reset+cap32 == reset uncapped (27.0 / 27.0) at")
    print("      a sixth of the buckets. The two belong together; neither works alone.")
    print("  pf-age (letting a prefill hit refresh last_acc, never the count) is worth +0.4 pp at most")
    print("  and is not the fix either -- but it is two lines, it never loses, and it removes the one")
    print("  real asymmetry between the shipped LRU path and the shipped bucket path.")
    print()
    print("  4. RECOMMENDATION -- per-request `count >>= 2`, cap C=32, plus pf-age. 33 buckets, ever.")
    print("  It is a hedge, and the hedge is the point: how much a decay is worth depends entirely on")
    print("  whether the next request wants different experts, and a server cannot know in advance.")
    print("      vs afq on a working-set SWITCH (ABABA)    21.1 % vs 17.8 % of the gap   better")
    print("      vs afq on a REPEATED prompt (AAAAAA)      37.3 % vs 42.4 %              worse by 5 pp")
    print("      vs afq on ONE request (cont-A / cont-B)   30.9 / 26.2 vs 31.1 / 26.3    within noise")
    print("      victim search                             33 buckets vs 1,381 and climbing")
    print("  It does NOT cure by degenerating to LRU: it keeps 30.9 of afq's 31.1 points on the very")
    print("  trace part 1 was fitted on. If the mix is known to be repeated prompts (an agent loop")
    print("  re-prefilling its own context), `>>= 1` or no decay at all is better and the cap alone is")
    print("  what should ship; if it is many separate conversations, full reset wins (24.9 %).")
    print()
    print("  SCOPE. Two traces (549-token prose, 444-token code decode, one prompt each), their two")
    print("  40-call prefills, 5,328 LRU slots + 400 ring slots, patterns ABABA / AAAAAA / AB / ABA,")
    print("  warm on the first request. The request switch is a WHOLE-PROMPT switch between two")
    print("  captures whose decode sets overlap 71.5 % and whose prefill sets overlap 42.2 %; a")
    print("  sharper switch (different language, different model context) is not represented. The")
    print("  prefill model is one chunk per layer because that is what the captures hold -- --pf-chunks")
    print("  scales it and every policy scales with it identically. Nothing here is a wall-clock")
    print("  measurement and nothing here has been through the engine.")
    return 0


def continuous_decay_sweep(a) -> int:
    """The same arms on the ORIGINAL single continuous decode stream, no prefill, no switch.

    This is the control the part-1 study already ran, re-run through the part-2 engine so the two
    columns are comparable: a decay that cures the switched workload by degenerating into LRU would
    show up here as 'gap recovered' collapsing back to 0 %.
    """
    _pf, dec = load_requests(a.log)
    cut = int(len(dec) * a.prefix) // N_LAYERS * N_LAYERS
    workload = [("warm", [], dec[:cut], True), ("A", [], dec[cut:], False)]
    tok_scored = (len(dec) - cut) / N_LAYERS
    print(f"  ORIGINAL continuous trace {a.log}: {len(dec):,} decode calls, "
          f"{len(dec) / N_LAYERS:.0f} tokens, no prefill, no request boundary.")
    print(f"  warm on [0,{cut:,}) = {100 * cut / len(dec):.0f} %, score the rest "
          f"({tok_scored:.0f} tokens)\n")
    nxt = flatten(workload)
    _, base = run_workload(workload, a.slots, Cap(0, pf_refresh=True), "lru", 1)
    _, bel = run_workload(workload, a.slots, Cap(0, pf_refresh=True), "belady", 1, nxt)
    bh = base["hits"] / (base["hits"] + base["fetches"])
    beh = bel["hits"] / (bel["hits"] + bel["fetches"])
    print(f"  {'policy':30s} {'hit':>8s} {'fetch/tok':>10s} {'MiB/tok':>9s} {'xLRU':>6s} "
          f"{'gap recovered':>14s} {'heads':>6s} {'max':>5s}")

    def line(label, tot):
        hit = tot["hits"] / (tot["hits"] + tot["fetches"])
        f = tot["fetches"]
        rec = (hit - bh) / (beh - bh) if beh > bh else 0.0
        print(f"  {label:30s} {100 * hit:7.2f} % {f / tok_scored:10.1f} "
              f"{f * EXPERT_BYTES / tok_scored / 2**20:9.0f} {f / base['fetches']:6.2f} "
              f"{100 * rec:13.1f} % {tot['heads']:6.0f} {tot['max_heads']:5d}")

    line("LRU  (shipped default)", base)
    arms = list(ARMS)
    if a.arms:
        arms = [x for x in arms if any(w in x[0] for w in a.arms)]
    for label, mk, mode, _d in arms:
        if label.startswith("LRU"):
            continue
        _, tot = run_workload(workload, a.slots, mk(), mode, 1)
        line(label, tot)
    line("Belady (future; the ceiling)", bel)
    print()
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("log", help="stream A (the prose decode capture)")
    ap.add_argument("--log-b", default=None, help="stream B, a capture of a DIFFERENT prompt")
    ap.add_argument("--mode", default="continuous",
                    choices=["continuous", "switched", "decay-continuous"],
                    help="continuous = the part-1 study; switched = the multi-request pollution "
                         "study; decay-continuous = the part-2 arms on the part-1 workload")
    ap.add_argument("--pattern", default="ABABA",
                    help="request order for --mode switched, letters indexing the streams")
    ap.add_argument("--warm", type=int, default=1, help="leading requests that are not scored")
    ap.add_argument("--pf-chunks", type=int, default=1,
                    help="repeat each request's 40-layer prefill this many times, chunk-major; the "
                         "captures hold one chunk per layer and a real prefill has several")
    ap.add_argument("--arms", nargs="*", default=None, help="substring filter on the arm labels")
    ap.add_argument("--detail", nargs="*", default=None,
                    help="also print the per-request table for these arms")
    ap.add_argument("--verify", action="store_true",
                    help="check the bucket-head argmax against a brute-force scan of every resident")
    ap.add_argument("--slots", type=int, nargs="+", default=[5328, 3000],
                    help="shipped lru_slots = 5728 - 400 transient = 5328; 3000 checks rank stability")
    ap.add_argument("--k", type=int, nargs="+", default=[32, 64, 256],
                    help="candidates ranked per eviction (the cold end of the LRU list)")
    ap.add_argument("--hazard-h", type=int, nargs="+", default=[10, 25, 45, 100, 200],
                    help="hazard horizons in tokens (swept; best reported, see the caveat)")
    ap.add_argument("--prefix", type=float, default=0.25)
    ap.add_argument("--verify-buckets", action="store_true",
                    help="part-1 only: assert the count-bucket implementation is bit-identical to "
                         "the brute-force K=all-slots scan (slow: minutes)")
    a = ap.parse_args()
    if a.mode == "continuous":
        return continuous_study(a)
    a.slots = a.slots[0]
    if a.mode == "switched":
        return switched_study(a)
    return continuous_decay_sweep(a)


if __name__ == "__main__":
    sys.exit(main())
