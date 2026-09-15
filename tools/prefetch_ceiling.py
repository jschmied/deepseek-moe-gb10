#!/usr/bin/env python3
"""prefetch_ceiling.py -- the UPPER BOUND on expert prefetch, and which HALF of it is prediction.

WHY THIS TOOL EXISTS, AND WHY IT COMES BEFORE ANY MORE PREDICTOR WORK.

We have measured learned prefetch predictors three times (same-layer recurrence in
tools/prefetch_oracle.py, cross-layer transition tables in tools/transition_prefetch.py) and each
time rejected them on economics. That is the wrong order of work. The right first question is the
CEILING: give an oracle PERFECT knowledge of exactly which (layer, expert) pairs the next H resolve
calls will MISS, prefetch precisely those and nothing else, and measure what that is worth. If a
perfect oracle is worth little, every predictor in the family is dead at once and no amount of
accuracy can rescue it -- exactly how expert-prediction heads were closed in the PROTECTION role
(a perfect protector recovered 0.4 pp of the Belady gap, so no predictor could).

But "perfect oracle" alone answers the wrong question, because it bundles two unknowns that cost
wildly different amounts to build. So every table below reports FOUR arms, not one:

  0  TODAY        no lookahead, SYNCHRONOUS resolve: issue this call's misses, wait, then compute.
                  This is the shipped path and it is pinned to the measured 1.73 steps/s.
  a  SUBSYSTEM    no lookahead, ASYNCHRONOUS issue + full overlap of this call's own compute with
                  its own reads. NO PREDICTION AT ALL -- the engine only ever knows the misses the
                  current router has already produced. This is what the PLUMBING alone is worth,
                  and `resolve(defer=True)` / the EARLY_SUBMIT path in engine/experts.py already
                  does the submit-and-join, so it is the cheap thing to build.
  b  ORACLE+IDEAL horizon H, plus an idealised subsystem: unbounded staging pool, a cache slot
                  charged only when the read COMPLETES, speculative issue at both ends of a call.
                  Device concurrency and bandwidth are still physics and still apply.
  c  ORACLE+REAL  horizon H on the subsystem we could actually ship: a finite pinned staging pool
                  (DSV41_IO_THREADS) that a prefetch HOLDS for its whole read, a cache slot
                  allocated BEFORE the read starts, and issue only at a layer boundary, because
                  there is no background issue path outside a resolve call today.

THE DECISIVE COMPARISON IS (b) MINUS (a). That is what PREDICTION is worth once the plumbing
exists. If (a) is already most of (b), the predictor is irrelevant and the lever is the loader.

READS ARE NOT PREEMPTIBLE, and the model obeys that. You cannot abort an in-flight 13.77 MB
O_DIRECT pread. So "a blocking miss takes priority" can only mean two things here, and both are
implemented as such: a demand read is admitted ahead of QUEUED speculative reads, and speculative
reads are not ISSUED while a demand miss is outstanding (the `gated` arm). A speculative read
already in flight runs to completion and keeps taking its share of bandwidth from the miss that
arrives after it started. That residual is the irreducible cost of speculating, and the ungated arm
prices what dropping the rule costs.

THE UNIT, stated once and loudly, because a unit error here cost most of a day:

    one RESOLVE CALL  = one LAYER
    40 resolve calls  = one DECODE STEP
    one decode step   = ~3.6 emitted tokens (MTP accept length)

Every per-step column is per DECODE STEP = per 40 resolve calls. The older tools in
deepseek-moe-gb10/tools call this same quantity "per token"; it is not, it is per step, and the
published baselines (64.6 fetches, 849 MiB) are per STEP.

HORIZON. H is in RESOLVE CALLS, so H=40 is exactly one decode step of lookahead.
H = 1, 2, 4, 8, 16, 40, 80, 400 (ten steps). The large horizons are the point: our own miss anatomy
says the misses are experts unused for a MEDIAN of ~57 steps, while an 8-call window is 0.2 of a
step. A study that stops at H=8 under-reports the ceiling by construction.

WHAT IS REUSED, not re-derived (deepseek-moe-gb10/tools):
  * prefetch_oracle.py     -- the call loader, EXPERT_BYTES, the warm-on-a-whole-step prefix rule,
                              the per-access hit trace and the `stolen` diff, and belady().
  * eviction_oracle.py     -- replay_buckets()'s EXACT global argmax of age/(1+count), as a cache
                              class so a prefetch can insert into it. Validated against the
                              published figure before anything else is printed.
  * transition_prefetch.py -- the processor-sharing IO class, the blocking-wait headline, the
                              free-lease issue policy. Extended here with a demand/speculative
                              queue split, a staging pool separate from device concurrency, a
                              completion callback (so a slot can be charged at completion instead
                              of at issue), and time-weighted in-flight accounting.

Usage:
    prefetch_ceiling.py [logs...] [--slots 5328] [--prefix 0.25]
CPU only; no GPU, no server. Latency is MODELLED, never measured -- the tool prints its assumptions
before it prints any number.
"""

from __future__ import annotations

import argparse
import collections
import heapq
import json
import os
import random
import sys

N_EXPERTS = 384          # DS4.1 routed experts per MoE layer
N_LAYERS = 40            # resolve calls per decode step
ACCEPT = 3.6             # emitted tokens per decode step (MTP accept length)

# What a miss actually reads: one native CB3 disk record. Corrected in prefetch_oracle.py on
# 2026-09-15 (the packed-FP4 number had inflated every absolute MiB figure by 1.36x).
EXPERT_BYTES = 13_774_848

# Device numbers, measured on this box with whole-expert O_DIRECT preads.
B_SINGLE = 5.58e9        # one read in flight
B_TOTAL = 6.82e9         # two in flight; the SSD is essentially saturated at ~2 concurrent reads,
                         # so n >= 2 splits 6.82 GB/s and more concurrency buys queueing only.

# The engine as it ships, MEASURED: 1.73 decode steps/s = 6.2 tok/s at accept 3.6, and decode is
# ~78 % expert-fetch wait. The remaining 22 % is compute, spread over the 40 resolve calls.
STEPS_PER_S_MEASURED = 1.73
FETCH_WAIT_FRAC = 0.78
STEP_MS_MEASURED = 1000.0 / STEPS_PER_S_MEASURED
WAIT_MS_MEASURED = STEP_MS_MEASURED * FETCH_WAIT_FRAC
COMPUTE_MS = STEP_MS_MEASURED * (1.0 - FETCH_WAIT_FRAC) / N_LAYERS   # per resolve call

# The pinned staging pool the engine runs today (DSV41_IO_THREADS). A prefetch HOLDS one of these
# for the whole duration of its read, which is why the pool size is a real constraint on arm (c)
# and not a modelling detail.
POOL_TODAY = 48
CAP_TODAY = 6            # reads the engine actually keeps in flight
BIG = 1 << 30

HORIZONS = [1, 2, 4, 8, 16, 40, 80, 400]


def load(path: str):
    """-> [(layer, (expert ids,)), ...] for DECODE calls only, in forward order.

    Identical to prefetch_oracle.load. The log opens with 40 pf=1 prefill calls and then runs
    strictly cyclic 0..39, so call index // 40 is the decode step and call index % 40 is the layer.
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


# --------------------------------------------------------------------------- eviction policies
# Both expose the same interface so the prefetch path is identical for either base policy:
#   __contains__, __len__, hit(k, acc), insert_demand(k, acc), insert_spec(k, acc), evict().
# The only thing a prefetch does differently from a demand fetch is insert_spec: it takes the slot
# without counting as a USE. That is invisible to LRU and load-bearing for age/(1+count).


class LRUCache:
    """Plain LRU -- the policy we ship. A prefetch inserts at MRU, the placement that PROTECTS it,
    which is the correct and most favourable choice for a PERFECT oracle: it never fetches anything
    wrong, so there is nothing to keep on probation."""

    label = "LRU"

    def __init__(self, slots):
        self.slots = slots
        self.od = collections.OrderedDict()

    def __contains__(self, k):
        return k in self.od

    def __len__(self):
        return len(self.od)

    def set_acc(self, acc):
        pass

    def hit(self, k, acc):
        self.od.move_to_end(k)

    def insert_demand(self, k, acc):
        self.od[k] = 1

    insert_spec = insert_demand

    def evict(self):
        k, _ = self.od.popitem(last=False)
        return k


class AgeFreqCache:
    """EXACT global argmax of age/(1+count), bucketed by use count.

    Lifted from eviction_oracle.replay_buckets: resident entries are bucketed by use count, each
    bucket kept in LRU order; within a bucket the count is constant, so the maximum of
    (now - last_acc)/(1 + count) is that bucket's LRU head and the global victim is the best head.
    A few hundred comparisons, EXACT, not an approximation.

    Per-key stats persist across eviction, exactly as in the original, so a pair that returns keeps
    its count. A SPECULATIVE insert refreshes last_acc but does NOT tick the count: a prefetch is
    not a use, and pretending otherwise would hand the oracle free protection it has not earned.
    """

    label = "age/(1+count)"

    def __init__(self, slots):
        self.slots = slots
        self.buckets = collections.defaultdict(collections.OrderedDict)   # count -> LRU-ordered
        self.where = {}                                                   # resident key -> bucket
        self.last_acc = collections.defaultdict(lambda: -1)
        self.count = collections.defaultdict(int)
        self.resident = 0
        self._acc = 0

    def __contains__(self, k):
        return k in self.where

    def __len__(self):
        return self.resident

    def set_acc(self, acc):
        self._acc = acc

    def _seat(self, k, c):
        self.buckets[c][k] = 1
        self.where[k] = c

    def _unseat(self, k):
        del self.buckets[self.where.pop(k)][k]

    def hit(self, k, acc):
        self._unseat(k)
        self.count[k] += 1
        self.last_acc[k] = acc
        self._seat(k, self.count[k])

    def insert_demand(self, k, acc):
        self.count[k] += 1
        self.last_acc[k] = acc
        self._seat(k, self.count[k])
        self.resident += 1

    def insert_spec(self, k, acc):
        self.last_acc[k] = acc
        self._seat(k, self.count[k])
        self.resident += 1

    def evict(self):
        best = None
        best_key = None
        for c, b in self.buckets.items():
            if not b:
                continue
            head = next(iter(b))
            la = self.last_acc[head]
            key = ((self._acc - la) / (1.0 + c), -la)
            if best is None or key > best_key:
                best, best_key = head, key
        self._unseat(best)
        self.resident -= 1
        return best


def make_cache(policy, slots):
    return LRUCache(slots) if policy == "lru" else AgeFreqCache(slots)


# --------------------------------------------------------------------------- the I/O model

class IO:
    """Processor-sharing NVMe: finite device concurrency, a finite pinned staging pool, two
    admission classes, and NO PREEMPTION.

    MODELLED, NOT MEASURED. Its job is to stop a pure cache study from booking latency wins the I/O
    path cannot deliver: a speculative read for call i+H competes for bandwidth AND for a pinned
    staging buffer with the blocking miss of call i that the engine is waiting on right now.

      * `cap`  reads IN FLIGHT. Bandwidth is shared: with n in flight each gets
        min(B_SINGLE, B_TOTAL / n), so one read runs at 5.58 GB/s and any n >= 2 splits 6.82 GB/s.
      * `pool` total OUTSTANDING reads (in flight + queued). A read holds its pinned staging buffer
        from submit to completion, so this is the real DSV41_IO_THREADS constraint.
      * two admission queues. A demand read is admitted ahead of every QUEUED speculative read.
        It canNOT preempt one already in flight -- an O_DIRECT pread in progress has no cheap abort
        -- so an in-flight speculative read keeps taking its share of bandwidth from a miss that
        arrives after it started.
    """

    def __init__(self, cap, pool, on_complete=None):
        self.cap = cap
        self.pool = pool
        self.t = 0.0
        self.active = {}                      # key -> remaining bytes
        self.qhi = collections.deque()        # demand reads
        self.qlo = collections.deque()        # speculative reads
        self.busy_set = set()
        self.on_complete = on_complete
        self.int_active = 0.0                 # time integral of in-flight count
        self.peak = 0

    def _admit(self):
        while len(self.active) < self.cap and (self.qhi or self.qlo):
            q = self.qhi if self.qhi else self.qlo
            self.active[q.popleft()] = float(EXPERT_BYTES)
        if len(self.active) > self.peak:
            self.peak = len(self.active)

    def busy(self, k):
        return k in self.busy_set

    def free_buffers(self):
        return max(0, self.pool - len(self.busy_set))

    def submit(self, k, demand):
        if k in self.busy_set:
            return
        self.busy_set.add(k)
        (self.qhi if demand else self.qlo).append(k)
        self._admit()

    def _step(self, max_dt):
        n = len(self.active)
        if not n:
            self.t += max_dt
            return max_dt
        rate = min(B_SINGLE, B_TOTAL / n)
        dt = min(max_dt, min(self.active.values()) / rate)
        done = []
        for k in list(self.active):
            self.active[k] -= rate * dt
            if self.active[k] <= 1e-3:
                del self.active[k]
                self.busy_set.discard(k)
                done.append(k)
        self.t += dt
        self.int_active += n * dt
        if done and self.on_complete is not None:
            for k in done:
                self.on_complete(k)
        self._admit()
        return dt

    def advance(self, dt):
        rem = dt
        while rem > 1e-12:
            rem -= self._step(rem)

    def finish(self, keys):
        """Advance until every key in `keys` has completed. -> elapsed seconds = blocking wait."""
        t0 = self.t
        pend = [k for k in keys if self.busy(k)]
        while pend:
            if not self.active and not self.qhi and not self.qlo:
                break
            self._step(float("inf"))
            pend = [k for k in pend if self.busy(k)]
        return self.t - t0


# --------------------------------------------------------------------------- the simulator

def simulate(calls, slots, cut, policy, H=0, cap=CAP_TODAY, pool=None,
             gated=True, overlap=True, slot_on_issue=True, recall=1.0, seed=12345):
    """Replay `calls` through `policy` with a PERFECT prefetch oracle of horizon H resolve calls.

    THE ORACLE. At a layer boundary, after call i has resolved, the oracle considers the window of
    calls [i+1, i+H]. It knows their exact contents. It issues a fetch for exactly those
    (layer, expert) pairs that
        * are used somewhere in that window, AND
        * are NOT resident in the cache right now, AND
        * are NOT already in flight,
    in order of FIRST USE inside the window (earliest need first), and for nothing else. That is
    literally "prefetch exactly what will miss": precision is 1.0 by construction, so the total
    NVMe byte count is unchanged unless capacity pressure evicts a prefetch before it is used.

    Knobs, and which arm each one belongs to:

      overlap       False = the SHIPPED synchronous path: submit this call's misses, wait for them,
                    then compute. True = asynchronous issue with the call's own compute fully
                    overlapped with its own reads. H=0 + overlap=True is arm (a), the value of the
                    subsystem with NO prediction whatsoever.
      gated         True  = speculative reads are issued only at a layer boundary AFTER this call's
                    demand misses have completed, i.e. never while a miss is outstanding. This is
                    the only realizable form of "blocking misses take priority", because an
                    in-flight read cannot be aborted.
                    False = issued at both ends of the call, including before the demand misses are
                    submitted. Prices what dropping the rule costs.
      pool          total outstanding reads (staging buffers held from submit to completion).
                    POOL_TODAY for the realizable arm, BIG for the idealised one.
      slot_on_issue True  = a prefetched expert needs its cache slot allocated BEFORE its read
                    starts, so it occupies capacity while in flight (realizable).
                    False = the slot is charged only on completion (idealised).
      recall        < 1.0 DEGRADES the oracle: it names only that fraction of the true future
                    misses, chosen at random, and never names anything false. That is still an
                    UPPER BOUND on a real predictor at the same recall, because a real one also
                    has precision < 1 and pays extra bytes and extra stolen slots for its false
                    positives. The sweep answers the only question that matters once the ceiling
                    is known: how accurate would a predictor have to be to be worth building.

    Returns counters plus a per-access hit bytearray for the exact `stolen` / `converted` diff.
    """
    if pool is None:
        pool = cap
    rng = random.Random(seed)
    cache = make_cache(policy, slots)

    n = len(calls)
    keys = [tuple(L * N_EXPERTS + e for e in ex) for L, ex in calls]

    pending = set()          # prefetched, resident, not yet demanded -- the standing capacity cost
    inflight_spec = set()    # slot_on_issue=False: read running, slot not yet charged
    uses = {}                # key -> deque of call indices inside the current window
    heap = []                # (first use in window, key), lazily validated

    hits = demand_fetch = pre_fetch = wasted = pre_useful = 0
    per_access = bytearray()
    blocking_wait = 0.0
    spec_sum = spec_max = spec_n = 0
    acc = 0
    scoring = False

    def on_evict(ev):
        nonlocal wasted
        if ev in pending:
            pending.discard(ev)
            if scoring:
                wasted += 1
        dq = uses.get(ev)
        if dq:                                  # still wanted inside the window -> candidate again
            heapq.heappush(heap, (dq[0], ev))

    def seat_spec(k):
        if len(cache) >= slots:
            on_evict(cache.evict())
        cache.insert_spec(k, acc)
        pending.add(k)

    def on_complete(k):
        if k in inflight_spec:
            inflight_spec.discard(k)
            seat_spec(k)

    io = IO(cap, pool, None if slot_on_issue else on_complete)

    def w_add(j):
        if j >= n:
            return
        for k in keys[j]:
            dq = uses.get(k)
            if dq is None:
                uses[k] = collections.deque((j,))
                heapq.heappush(heap, (j, k))
            else:
                dq.append(j)

    def w_rem(j):
        if j >= n:
            return
        for k in keys[j]:
            dq = uses.get(k)
            if dq and dq[0] == j:
                dq.popleft()
                if dq:
                    heapq.heappush(heap, (dq[0], k))
                else:
                    del uses[k]

    def issue():
        """Issue speculative reads, earliest-need first, into free staging buffers."""
        nonlocal pre_fetch
        budget = io.free_buffers()
        got = 0
        while got < budget and heap:
            j, k = heap[0]
            dq = uses.get(k)
            if dq is None or dq[0] != j:
                heapq.heappop(heap)
                continue
            heapq.heappop(heap)
            if k in cache or io.busy(k):
                continue                        # nothing to fetch; on_evict re-arms it if needed
            if recall < 1.0 and rng.random() >= recall:
                continue                        # this true miss is one the predictor does not name
            if slot_on_issue:
                seat_spec(k)
            else:
                inflight_spec.add(k)
            io.submit(k, demand=False)
            if scoring:
                pre_fetch += 1
            got += 1

    if H:
        for j in range(1, H + 1):               # prime the window to [1, H]
            w_add(j)

    io_t_cut = io_int_cut = 0.0
    for i in range(n):
        if i == cut:
            io_t_cut, io_int_cut = io.t, io.int_active
        scoring = i >= cut
        cache.set_acc(acc)

        if H and not gated:
            issue()

        needed = []
        for k in keys[i]:
            acc += 1
            cache.set_acc(acc)
            if k in inflight_spec:
                # Idealised-slot arm: the read is running and the expert IS coming. Charge the slot
                # now and count it as a hit that waits only for the remainder of its own read.
                inflight_spec.discard(k)
                if len(cache) >= slots:
                    on_evict(cache.evict())
                cache.insert_demand(k, acc)
                needed.append(k)
                if scoring:
                    hits += 1
                    pre_useful += 1
                    per_access.append(1)
            elif k in cache:
                cache.hit(k, acc)
                if io.busy(k):
                    needed.append(k)            # prefetched but still in flight: wait the remainder
                if k in pending:
                    pending.discard(k)
                    if scoring:
                        pre_useful += 1
                if scoring:
                    hits += 1
                    per_access.append(1)
            else:
                if len(cache) >= slots:
                    on_evict(cache.evict())
                cache.insert_demand(k, acc)
                io.submit(k, demand=True)
                needed.append(k)
                if scoring:
                    demand_fetch += 1
                    per_access.append(0)

        if overlap:
            io.advance(COMPUTE_MS / 1000.0)     # the call's own compute runs while its reads run
        w = io.finish(needed)
        if scoring:
            blocking_wait += w
        if H:
            issue()                             # the layer boundary: the only issue point we have
        if not overlap:
            io.advance(COMPUTE_MS / 1000.0)     # shipped path: compute strictly after the wait

        if H:
            w_rem(i + 1)
            w_add(i + 1 + H)

        if scoring:
            held = len(pending) + len(inflight_spec)
            spec_sum += held
            if held > spec_max:
                spec_max = held
            spec_n += 1

    span = io.t - io_t_cut
    return {
        "hits": hits, "accesses": len(per_access), "demand_fetch": demand_fetch,
        "pre_fetch": pre_fetch, "wasted": wasted, "pre_useful": pre_useful,
        "trace": per_access, "wait": blocking_wait,
        "spec_avg": spec_sum / max(spec_n, 1), "spec_max": spec_max,
        "mean_inflight": (io.int_active - io_int_cut) / span if span > 0 else 0.0,
        "peak_inflight": io.peak,
    }


def cache_only(calls, slots, cut, policy):
    """The shipped cache with no I/O model at all -- used solely to reproduce the published
    hit-rate / fetches-per-step baselines, which is how this harness earns the right to be read."""
    cache = make_cache(policy, slots)
    hits = fetches = 0
    acc = 0
    trace = bytearray()
    for i, (L, ex) in enumerate(calls):
        scoring = i >= cut
        for e in ex:
            k = L * N_EXPERTS + e
            acc += 1
            cache.set_acc(acc)
            if k in cache:
                cache.hit(k, acc)
                if scoring:
                    hits += 1
                    trace.append(1)
            else:
                if len(cache) >= slots:
                    cache.evict()
                cache.insert_demand(k, acc)
                if scoring:
                    fetches += 1
                    trace.append(0)
    return {"hits": hits, "demand_fetch": fetches, "accesses": len(trace), "trace": trace}


def belady(calls, slots, cut):
    """Evict the resident pair whose next use is furthest away. Lifted from prefetch_oracle.py."""
    seq = [L * N_EXPERTS + e for L, ex in calls for e in ex]
    cut_acc = sum(len(ex) for _, ex in calls[:cut])
    nxt = [0] * len(seq)
    last = {}
    for i in range(len(seq) - 1, -1, -1):
        nxt[i] = last.get(seq[i], len(seq))
        last[seq[i]] = i
    cache, hp, hits, fetches = {}, [], 0, 0
    for i, k in enumerate(seq):
        if k in cache:
            hits += (i >= cut_acc)
        else:
            if len(cache) >= slots:
                while True:
                    neg, kk = heapq.heappop(hp)
                    if cache.get(kk) == -neg:
                        del cache[kk]
                        break
            fetches += (i >= cut_acc)
        cache[k] = nxt[i]
        heapq.heappush(hp, (-nxt[i], k))
    return hits, fetches, len(seq) - cut_acc


def diff(base_trace, trace):
    """-> (converted, stolen), exact, per access, against the base replay of the same stream.

    converted = accesses the base policy MISSED and this one HIT: blocking misses actually removed.
    stolen    = accesses the base policy HIT and this one MISSED, because a speculative entry
                evicted a live one. For a PERFECT oracle this is pure capacity damage.
    """
    conv = sto = 0
    for x, y in zip(base_trace, trace):
        if y and not x:
            conv += 1
        elif x and not y:
            sto += 1
    return conv, sto


# --------------------------------------------------------------------------- reporting

def assumptions():
    print("  ASSUMPTIONS -- EVERY LATENCY BELOW IS MODELLED, NOT MEASURED. Do not quote one as a")
    print("  measurement. The inputs to the model are measured; the output is arithmetic.")
    print(f"    * one expert record = {EXPERT_BYTES:,} B = {EXPERT_BYTES / 2**20:.2f} MiB, read as")
    print(f"      one whole-expert O_DIRECT pread. One read alone = {B_SINGLE / 1e9:.2f} GB/s; two")
    print(f"      in flight = {B_TOTAL / 1e9:.2f} GB/s. The device is essentially SATURATED at ~2")
    print(f"      concurrent reads, so n in flight each get min({B_SINGLE / 1e9:.2f}, "
          f"{B_TOTAL / 1e9:.2f}/n) GB/s: past 2,")
    print("      concurrency buys queueing, not throughput.")
    print(f"    * device concurrency `cap` swept at 2 (the saturation point) and {CAP_TODAY} (what")
    print(f"      the engine runs). Pinned staging pool = {POOL_TODAY} (DSV41_IO_THREADS), held by a")
    print("      read from submit to completion, so a prefetch's buffer is unavailable to a miss.")
    print("    * NO PREEMPTION. An in-flight 13.77 MB pread cannot be aborted. Priority therefore")
    print("      means only (i) a demand read is admitted ahead of QUEUED speculative reads, and")
    print("      (ii) in the `gated` arm no speculative read is ISSUED while a miss is outstanding.")
    print("      A speculative read already running keeps its share of bandwidth regardless.")
    print("    * issue happens only at a layer boundary -- there is no background issue path")
    print("      outside a resolve call today.")
    print("    * UNITS: one resolve call = one LAYER; 40 calls = one decode STEP; one step emits")
    print(f"      ~{ACCEPT} tokens. Every per-step column is per 40 resolve calls.")
    print(f"    * the shipped engine, MEASURED: {STEPS_PER_S_MEASURED} steps/s = "
          f"{STEP_MS_MEASURED:.0f} ms/step = {STEPS_PER_S_MEASURED * ACCEPT:.1f} tok/s, of which")
    print(f"      {100 * FETCH_WAIT_FRAC:.0f} % is expert-fetch wait = {WAIT_MS_MEASURED:.0f} "
          f"ms/step; the remaining {COMPUTE_MS:.2f} ms per")
    print("      resolve call is compute, which is what an async subsystem gets to overlap.")
    print("    * the oracle has ZERO wrong prefetches by construction, so it adds ZERO extra bytes.")
    print("      Its only costs are the slot it holds and the buffer/bandwidth it takes.")
    print()


ARMS = [
    # (tag, label, H-swept?, kwargs)
    ("0", "TODAY -- synchronous, no lookahead (the shipped path)", False,
     dict(H=0, cap=CAP_TODAY, pool=POOL_TODAY, gated=True, overlap=False, slot_on_issue=True)),
    ("a", "SUBSYSTEM ONLY -- async issue + compute overlap, NO prediction", False,
     dict(H=0, cap=CAP_TODAY, pool=POOL_TODAY, gated=True, overlap=True, slot_on_issue=True)),
    ("b", "ORACLE + IDEALISED subsystem (unbounded pool, slot charged on completion)", True,
     dict(cap=CAP_TODAY, pool=BIG, gated=False, overlap=True, slot_on_issue=False)),
    ("c", "ORACLE + REALIZABLE subsystem (pool 48, slot on issue, gated, cap 6)", True,
     dict(cap=CAP_TODAY, pool=POOL_TODAY, gated=True, overlap=True, slot_on_issue=True)),
    ("c2", "ORACLE + REALIZABLE at device concurrency 2", True,
     dict(cap=2, pool=POOL_TODAY, gated=True, overlap=True, slot_on_issue=True)),
    ("cu", "ORACLE + REALIZABLE, issue gate REMOVED (prices the gate)", True,
     dict(cap=CAP_TODAY, pool=POOL_TODAY, gated=False, overlap=True, slot_on_issue=True)),
]


def run_trace(path, slots, prefix, horizons):
    calls = load(path)
    if not calls:
        print(f"  {path}: no decode calls", file=sys.stderr)
        return
    n = len(calls)
    accesses = sum(len(ex) for _, ex in calls)
    cut = int(n * prefix) // N_LAYERS * N_LAYERS          # whole decode steps only
    steps = (n - cut) / N_LAYERS

    print(f"  ========== {os.path.basename(path)} ==========")
    print(f"  {n:,} decode resolve calls = {n / N_LAYERS:.1f} decode steps "
          f"(~{n / N_LAYERS * ACCEPT:.0f} emitted tokens), {accesses:,} activations, "
          f"{accesses / (n / N_LAYERS):.1f} per step")
    print(f"  warm on calls [0,{cut:,}) = {100 * cut / n:.0f} %; score the suffix "
          f"({n - cut:,} calls = {steps:.0f} decode steps)")
    print(f"  cache {slots:,} slots = {100 * slots / (N_LAYERS * N_EXPERTS):.1f} % of the "
          f"{N_LAYERS * N_EXPERTS:,} (layer, expert) pairs")
    print()

    # ------------------------------------------------------------------ harness validation
    print("  HARNESS VALIDATION -- reproduce the published baselines before trusting a new number.")
    bases = {}
    ok_all = True
    for pol, want, nm in (("lru", (92.67, 64.6, 849), "LRU"),
                          ("agefreq", (94.07, 52.3, 687), "age/(1+count)")):
        b = cache_only(calls, slots, cut, pol)
        hit = 100 * b["hits"] / b["accesses"]
        fps = b["demand_fetch"] / steps
        mibs = b["demand_fetch"] * EXPERT_BYTES / steps / 2**20
        ok = abs(hit - want[0]) < 0.02 and abs(fps - want[1]) < 0.1 and abs(mibs - want[2]) < 1.5
        ok_all &= ok
        bases[pol] = b
        print(f"    {nm:14s} {hit:6.2f} % hit / {fps:5.1f} fetches per step / {mibs:5.0f} MiB per "
              f"step   published {want[0]} / {want[1]} / {want[2]}   "
              f"{'MATCH' if ok else 'DIFFERS'}")
    hb, fb, _ = belady(calls, slots, cut)
    print(f"    {'Belady':14s} {100 * hb / bases['lru']['accesses']:6.2f} % hit / {fb / steps:5.1f} "
          f"fetches per step / {fb * EXPERT_BYTES / steps / 2**20:5.0f} MiB per step   "
          f"(needs the whole future; unbuildable floor)")
    if not ok_all:
        print("    ^ the published baselines are for route-decode.jsonl at 5,328 slots / 25 % prefix.")
        print("      On any OTHER trace or cache size a difference is expected and is not a defect;")
        print("      on that trace a mismatch would mean the tables below must not be trusted.")
    gb_step = bases["lru"]["demand_fetch"] * EXPERT_BYTES / steps / 1e9
    print(f"    device utilisation sanity: {bases['lru']['demand_fetch'] / steps:.1f} fetches/step "
          f"over the MEASURED {STEP_MS_MEASURED:.0f} ms step is")
    print(f"      only {1000 * gb_step / STEP_MS_MEASURED:.2f} GB/s of the {B_TOTAL / 1e9:.2f} GB/s "
          f"the device delivers. There IS idle bandwidth, which is")
    print("      the premise of the whole study.")
    print()

    for pol in ("lru", "agefreq"):
        base = bases[pol]
        scored = base["accesses"]
        m_base = base["demand_fetch"]
        polname = LRUCache.label if pol == "lru" else AgeFreqCache.label

        # Arm 0 sets the calibration: it IS the shipped path, so it is pinned to 1.73 steps/s.
        r0 = simulate(calls, slots, cut, pol, **ARMS[0][3])
        wait0 = 1000 * r0["wait"] / steps
        comp_ms = COMPUTE_MS * N_LAYERS
        k_opt = WAIT_MS_MEASURED / wait0 if wait0 > 0 else 0.0
        resid = max(0.0, WAIT_MS_MEASURED - wait0)

        print(f"  ----- base eviction policy: {polname} ({100 * base['hits'] / scored:.2f} % hit, "
              f"{m_base / steps:.1f} blocking misses/step) -----")
        print(f"  Modelled arm-0 wait is {wait0:.1f} ms/step against a MEASURED "
              f"{WAIT_MS_MEASURED:.0f} ms/step, so the pure-bandwidth")
        print(f"  model is {k_opt:.1f}x optimistic about the real read path (syscall, buffer copy, "
              f"HtoD and")
        print("  bookkeeping are not in it). The implied rates are therefore printed as a BRACKET:")
        print(f"    OPT  scales every arm's wait by that same {k_opt:.1f}x -- i.e. assumes ALL the "
              f"unmodelled")
        print("         overhead is background work that overlap hides too.")
        print(f"    PESS holds {resid:.0f} ms/step of it fixed as serial critical-path work and "
              f"moves only the")
        print("         transfer time. Arm 0 reproduces 1.73 steps/s in both, by construction.")
        print()
        hdr = (f"  {'arm':>3s} {'H':>4s} {'hit':>7s} {'blk/step':>9s} {'conv':>6s} {'stol':>5s} "
               f"{'wast':>5s} {'wait/step':>10s} {'vs 0':>8s} {'OPT st/s':>9s} {'tok/s':>6s} "
               f"{'PESS st/s':>10s} {'tok/s':>6s} {'pf slots':>9s} {'max':>5s} {'infl':>5s} "
               f"{'pk':>3s}")

        def row(tag, H, r):
            conv, sto = diff(base["trace"], r["trace"])
            w = 1000 * r["wait"] / steps
            opt = 1000.0 / (w * k_opt + comp_ms)
            pess = 1000.0 / (w + resid + comp_ms)
            print(f"  {tag:>3s} {H:4d} {100 * r['hits'] / scored:6.2f} % "
                  f"{r['demand_fetch'] / steps:9.1f} {100 * conv / m_base:5.1f}% {sto:5,d} "
                  f"{r['wasted']:5,d} {w:10.1f} "
                  f"{(100 * (w - wait0) / wait0 if wait0 else 0):+7.1f}% {opt:9.2f} "
                  f"{opt * ACCEPT:6.2f} {pess:10.2f} {pess * ACCEPT:6.2f} {r['spec_avg']:9.1f} "
                  f"{r['spec_max']:5d} {r['mean_inflight']:5.2f} {r['peak_inflight']:3d}")

        best = None
        for tag, label, swept, kw in ARMS:
            print(f"  {tag}  {label}")
            print(hdr)
            if not swept:
                r = r0 if tag == "0" else simulate(calls, slots, cut, pol, **kw)
                row(tag, 0, r)
                if tag == "a":
                    wait_a = 1000 * r["wait"] / steps
            else:
                for H in horizons:
                    r = simulate(calls, slots, cut, pol, H=H, **kw)
                    row(tag, H, r)
                    w = 1000 * r["wait"] / steps
                    if tag == "c" and (best is None or w < best[1]):
                        best = (H, w)
            print()

        # ---------------------------------------------------------- the decomposition, spelled out
        bh, bw = best
        print(f"  WHERE THE {wait0:.0f} ms/step OF MODELLED BLOCKING WAIT GOES ({polname}):")
        print(f"    {wait0 - wait_a:6.1f} ms/step ({100 * (wait0 - wait_a) / wait0:4.1f} %) removed "
              f"by the SUBSYSTEM ALONE -- async issue + compute")
        print("                          overlap, arm (a), NO PREDICTION OF ANY KIND.")
        print(f"    {wait_a - bw:6.1f} ms/step ({100 * (wait_a - bw) / wait0:4.1f} %) removed on top "
              f"of that by a PERFECT oracle at its best")
        print(f"                          horizon (arm c, H={bh}). This is the ENTIRE budget any")
        print("                          predictor is competing for.")
        print(f"    {bw:6.1f} ms/step ({100 * bw / wait0:4.1f} %) irreducible: the bytes still have "
              f"to cross the device.")
        print()

        # ---------------------------------------------------------- how accurate is accurate enough
        kwc = dict(ARMS[3][3])
        print(f"  RECALL SWEEP -- arm (c) at H={bh}, oracle degraded to name only `recall` of the true")
        print("  future misses and NOTHING false. Perfect precision, so every row is still an UPPER")
        print("  BOUND on a real predictor at that recall: a real one also pays for false positives")
        print("  in extra bytes and extra stolen slots. Read it as 'what would we have to hit'.")
        print(f"  {'recall':>7s} {'blk/step':>9s} {'conv':>6s} {'wait/step':>10s} "
              f"{'of oracle win':>14s} {'OPT st/s':>9s} {'PESS st/s':>10s}")
        win = wait_a - bw
        for rc in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0):
            r = simulate(calls, slots, cut, pol, H=bh, recall=rc, **kwc)
            w = 1000 * r["wait"] / steps
            conv, _ = diff(base["trace"], r["trace"])
            got = (wait_a - w) / win if win > 0 else 0.0
            print(f"  {rc:7.2f} {r['demand_fetch'] / steps:9.1f} {100 * conv / m_base:5.1f}% "
                  f"{w:10.1f} {100 * got:13.1f} % "
                  f"{1000.0 / (w * k_opt + comp_ms):9.2f} {1000.0 / (w + resid + comp_ms):10.2f}")
        print()
    print()


def legend():
    print("  COLUMNS")
    print("    arm       0 shipped sync path / a async+overlap with NO prediction / b oracle on an")
    print("              idealised subsystem / c the same oracle on a subsystem we could ship /")
    print("              c2 same at device concurrency 2 / cu same but WITHOUT the issue gate.")
    print("    H         oracle horizon in RESOLVE CALLS. 40 = one decode step. 0 = no lookahead.")
    print("    blk/step  BLOCKING misses per decode step left after prefetching. Decode is ~78 %")
    print("              expert-fetch wait, so this is what the engine actually waits on.")
    print("    conv      of the base policy's blocking misses, the fraction this arm turned into")
    print("              hits (exact per-access diff against the base replay of the same stream).")
    print("    stol      accesses the base policy HIT that this arm MISSED, because a speculative")
    print("              entry evicted a live one. Already inside blk/step; shown so it is visible.")
    print("    wast      prefetched entries evicted before anything demanded them. For a PERFECT")
    print("              oracle this is not prediction error -- it is purely capacity pressure.")
    print("    wait/step MODELLED total blocking wait per decode step, ms. THE HEADLINE.")
    print("    OPT/PESS  implied decode steps/s and emitted tok/s at accept 3.6, the two ends of the")
    print("              calibration bracket described above each table. The truth is inside it, and")
    print("              WHICH END is an empirical question this tool cannot settle.")
    print("    pf slots  cache slots held by prefetched-but-not-yet-used entries, mean over the")
    print("              scored window / max. The standing capacity cost of speculation.")
    print("    infl/pk   mean (time-weighted) and peak reads in flight -- is the schedule feasible?")
    print()
    print("  HOW TO READ IT. (a) minus (0) is what the LOADER is worth with no predictor at all.")
    print("  (b) minus (a) is what PREDICTION is worth once the plumbing exists -- that difference,")
    print("  not (b) itself, is the budget any predictor is competing for. (c) is what survives the")
    print("  constraints we actually have, and (b) minus (c) is what better plumbing would buy on")
    print("  top. (cu) minus (c) prices the issue gate.")
    print()
    print("  SCOPE. Decode only, two traces (prose and code-heavy), one cache size (5,328 slots =")
    print("  the shipped lru_slots), device concurrency 2 and 6, staging pool 48. Cache behaviour is")
    print("  an exact replay of measured routing; LATENCY IS MODELLED, NOT MEASURED, from three")
    print("  measured device numbers and the measured decode split. Nothing here was run on a GPU.")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("logs", nargs="*", default=[
        os.path.expanduser("~/ds41-queue/logs/route-decode.jsonl"),
        os.path.expanduser("~/ds41-queue/logs/route-decode-code.jsonl"),
    ])
    ap.add_argument("--slots", type=int, default=5328,
                    help="shipped lru_slots = 5728 n_slots - 400 transient")
    ap.add_argument("--prefix", type=float, default=0.25, help="warm-up fraction")
    ap.add_argument("--horizons", type=int, nargs="+", default=HORIZONS)
    a = ap.parse_args()

    print()
    print("  prefetch_ceiling.py -- the UPPER BOUND on expert prefetch, and which HALF is")
    print("  prediction. A PERFECT oracle knows exactly which (layer, expert) pairs the next H")
    print("  resolve calls will MISS and prefetches precisely those, nothing else: zero wasted")
    print("  prefetches by construction, therefore ZERO extra NVMe bytes. What is left is the value")
    print("  of OVERLAP -- and arm (a) splits off the part of that value needing no prediction.")
    print()
    assumptions()

    for path in a.logs:
        if not os.path.exists(path):
            print(f"  (skipping {path}: not present)\n")
            continue
        run_trace(path, a.slots, a.prefix, a.horizons)

    legend()
    return 0


if __name__ == "__main__":
    sys.exit(main())
