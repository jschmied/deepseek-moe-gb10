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

CORRECTION 2026-09-15, AND IT DOES REVERSE THE PREVIOUS COMMIT. The first version of this tool
gave the no-prediction arm a whole LAYER of compute to overlap its own reads with:

    queue the current call's misses ; io.advance(COMPUTE_MS) ; wait for the rest

That is not the dependency order the engine has. `engine/fastdecode.py` lines 13-15:

    "Per backbone layer there are two graphs: A = attention + HC + router (ends with the expert
     ids), then the host resolves expert slots (LRU / NVMe), then B = MoE + shared expert + HC
     residual."

The expert ids do not exist until the END of graph A. A loader with no prediction cannot overlap
its reads with attention/HC/router, because that compute has already run by the time the misses are
knowable. It can only overlap with post-router work that does NOT depend on the routed experts.
The old arm handed the loader a window only PREDICTION can physically produce, which biased the
whole study toward the loader and against prediction -- which is exactly the conclusion it drew.
So the per-layer compute is now split into three dependency phases:

  C_pre   attention + HC + router. Runs BEFORE this layer's misses are known. Unusable by a
          non-predictive loader; usable by a predictor that issued H layers earlier.
  C_ind   post-router work independent of the routed experts. The SHARED EXPERT is the main one:
          graph B computes it alongside the routed MoE. THE ONLY window a non-predictive loader
          has.
  C_dep   routed MoE + combine. Cannot start until the routed experts have arrived.

    no predictor :  C_pre ; issue demand misses ; overlap ONLY with C_ind ; wait ; C_dep
    oracle at H  :  issue H layers earlier ; ... ; C_pre ; wait only for unfinished reads ;
                    C_ind + C_dep

THE SPLIT IS NOT MEASURED YET. A decode nsys profile (queue job 165) will measure it. So it is not
hardcoded to a guess: C_ind is parameterised as a fraction f of the per-layer compute and SWEPT,
f in {0.02, 0.05, 0.10, 0.20, 0.35, 0.50}, and every conclusion below is reported as a function of
f, together with the f at which the verdict flips. The shared expert is 1 of 385 experts, so small
f is the likely regime -- but the tool reports what the numbers do, not what anyone expects.
The C_pre / C_dep split of the remaining (1 - f) is immaterial to every arm except the ungated
pricing arm (cu), because the compute between two consecutive waits always sums to one layer; it is
fixed at half and half and is not swept.

But "perfect oracle" alone answers the wrong question, because it bundles two unknowns that cost
wildly different amounts to build. So every table below reports FOUR arms, not one:

  0  TODAY        no lookahead, SYNCHRONOUS resolve: issue this call's misses, wait, then compute.
                  This is the shipped path and it is pinned to the measured 1.73 steps/s.
  a  SUBSYSTEM    no lookahead, ASYNCHRONOUS issue, overlapping this call's own reads with C_ind
                  ONLY -- the corrected chronology above. NO PREDICTION AT ALL: the engine only
                  ever knows the misses the current router has already produced. This is what the
                  PLUMBING alone is worth, and `resolve(defer=True)` / the EARLY_SUBMIT path in
                  engine/experts.py already does the submit-and-join, so it is the cheap thing to
                  build.
  a! SUPERSEDED   the old, wrong arm (a): the same loader given a FULL layer of compute to overlap
                  into. Kept and printed so the size of the scheduling error is visible rather
                  than quietly corrected away.
  b  ORACLE+IDEAL horizon H, plus an idealised subsystem: unbounded staging pool, a cache slot
                  charged only when the read COMPLETES, speculative issue at both ends of a call.
                  Device concurrency and bandwidth are still physics and still apply.
  c  ORACLE+REAL  horizon H on the subsystem we could actually ship: a finite pinned staging pool
                  (DSV41_IO_THREADS) that a prefetch HOLDS for its whole read, a cache slot
                  allocated BEFORE the read starts, and issue only at a layer boundary, because
                  there is no background issue path outside a resolve call today.

THE DECISIVE COMPARISON IS THE ORACLE MINUS (a). That is what PREDICTION is worth once the
plumbing exists. If (a) is already most of it, the predictor is irrelevant and the lever is the
loader. (b) minus (a) is the ceiling on that; (c) minus (a) is the version that could ship, and it
is (c) minus (a) that the decisive table reports, at every f, in the same currency -- per cent of
arm-0's blocking wait -- that the superseded commit used, so the two are directly comparable.

CONCURRENCY IS NOT FREE, and the model now charges for it. Overlapping H2D expert copies with
compute is the thing `stream.wait_stream(compute)` prevents; we removed that barrier once and
measured the run 14.5 % SLOWER end to end (notes, commit dfb312c). The accompanying "and the FFN
pays it, +32 %" claim from that same commit was RETRACTED by cd3be98: an nsys profile found MoE
kernel time flat within +-1 % across all three EARLY_SUBMIT arms, SMs Active 97.9 %, clocks flat --
it was a host-side wall clock, not a device effect. So the penalty here is calibrated to the 14.5 %
TOTAL and never to the 32 %. THE PENALTY MODEL, stated so it can be argued with:

    any compute that runs while at least one expert read is in flight is charged
    GAMMA = 0.145 of its own duration as extra wall-clock time.

Three properties of that model, each a deliberate choice:
  * it is a LOWER bound on the per-unit stretch. In the calibrating run only part of the compute
    overlapped a copy, so spreading the same 14.5 % over a smaller overlapped base would imply a
    larger GAMMA. A sensitivity row at GAMMA in {0, 0.145, 0.30} is printed.
  * the penalty is extra wall time and does NOT advance the I/O clock. Contention slows both sides;
    crediting the read with the window its own contention created would let it free-ride on its
    own damage.
  * it applies to every arm identically, and it therefore costs the ORACLE arms MOST, because they
    are the arms that keep reads in flight under compute. That is the correct direction: this
    correction moves the comparison toward prediction, and the penalty moves it back.

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
    prefetch_ceiling.py [logs...] [--slots 5328] [--prefix 0.25] [--f-sweep ...] [--gamma 0.145]
CPU only; no GPU, no server. Latency is MODELLED, never measured, and the compute split f is not
even modelled from a measurement -- it is an unmeasured parameter that is swept. The tool prints
its assumptions before it prints any number.
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

# --- the dependency split of one layer's compute (engine/fastdecode.py lines 13-15) -------------
# C_pre + C_ind + C_dep = COMPUTE_MS. f = C_ind / COMPUTE_MS is NOT MEASURED -- queue job 165 (a
# decode nsys profile) will measure it -- so it is swept and every verdict is a function of it.
# The shared expert is 1 of 385 experts, so the small-f end is the likely regime.
F_SWEEP = [0.02, 0.05, 0.10, 0.20, 0.35, 0.50]
F_REF = 0.05             # the f the detailed per-arm tables are printed at
# The C_pre / C_dep split of the remaining (1-f) is immaterial to every arm except (cu): the
# compute between two consecutive waits always sums to exactly one layer. Fixed, not swept.
PRE_SHARE = 0.5

# Contention penalty. Calibrated to the ONE end-to-end measurement we have of letting H2D run
# concurrently with compute: removing stream.wait_stream(compute) cost 14.5 % overall (dfb312c).
# The "FFN +32 %" from that same commit is RETRACTED (cd3be98: MoE kernels flat within +-1 %,
# SMs Active 97.9 %, clocks flat -- host-side wall clock, not a device effect) and is NOT used.
GAMMA = 0.145
GAMMA_SWEEP = [0.0, 0.145, 0.30]


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
        """Run the clock for `dt` seconds of compute. -> the part of `dt` during which at least one
        read was IN FLIGHT, i.e. the compute that was exposed to copy contention and is therefore
        charged the GAMMA penalty by the caller."""
        rem = dt
        busy = 0.0
        while rem > 1e-12:
            was_busy = bool(self.active)
            d = self._step(rem)
            if was_busy:
                busy += d
            rem -= d
        return busy

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
             gated=True, overlap=True, slot_on_issue=True, recall=1.0, seed=12345,
             f_ind=F_REF, gamma=GAMMA, full_layer_window=False):
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

      overlap       False = the SHIPPED synchronous path: C_pre, submit this call's misses, wait
                    for them, then C_ind + C_dep. True = asynchronous issue, with the call's own
                    reads overlapped against C_ind ONLY -- the corrected chronology, since the
                    misses do not exist until the router at the end of C_pre has produced them.
                    H=0 + overlap=True is arm (a), the value of the subsystem with NO prediction.
      f_ind         C_ind as a fraction of one layer's compute. UNMEASURED; swept.
      full_layer_window
                    True reinstates the SUPERSEDED behaviour: the loader overlaps a whole layer of
                    compute with reads it could not have known about that early. Arm (a!) only,
                    kept so the size of the scheduling error is printed rather than hidden.
      gamma         contention penalty: compute running while >=1 read is in flight is charged this
                    fraction of its own duration as extra wall time. Calibrated to the measured
                    14.5 % cost of removing stream.wait_stream (dfb312c); the retracted 32 % FFN
                    figure (cd3be98) is not used.
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

    if full_layer_window:
        c_pre = c_dep = 0.0
        c_ind = COMPUTE_MS / 1000.0
    else:
        c_ind = f_ind * COMPUTE_MS / 1000.0
        c_pre = PRE_SHARE * (1.0 - f_ind) * COMPUTE_MS / 1000.0
        c_dep = (1.0 - PRE_SHARE) * (1.0 - f_ind) * COMPUTE_MS / 1000.0

    hits = demand_fetch = pre_fetch = wasted = pre_useful = 0
    per_access = bytearray()
    blocking_wait = 0.0
    contended = 0.0          # seconds of compute that ran with >=1 read in flight
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
            issue()                             # ungated: also issue before C_pre

        # ---- C_pre: attention + HC + router. THIS LAYER'S MISSES ARE NOT KNOWN YET, so a
        # non-predictive loader has nothing to issue here; only reads a predictor issued earlier
        # are running. (Arms 0 and a therefore find the device idle across this phase.)
        b = io.advance(c_pre)
        if scoring:
            contended += b

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

        # ---- C_ind: post-router work that does NOT depend on the routed experts (the shared
        # expert is the main one). THE ONLY WINDOW A NON-PREDICTIVE LOADER HAS.
        if overlap:
            b = io.advance(c_ind)
            if scoring:
                contended += b
        w = io.finish(needed)
        if scoring:
            blocking_wait += w
        if H:
            issue()                             # the layer boundary: the only issue point we have
        # ---- C_dep: routed MoE + combine, which needed the experts. Arm 0 folds C_ind in here,
        # because the shipped path computes strictly after the wait.
        b = io.advance(c_dep if overlap else c_ind + c_dep)
        if scoring:
            contended += b

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
        "contended": contended, "pen": gamma * contended,
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
    print("    * DEPENDENCY ORDER (engine/fastdecode.py:13-15): graph A = attention + HC + router")
    print("      ENDS with the expert ids; only then does the host resolve slots; graph B = MoE +")
    print("      shared expert + HC residual. So one layer's compute splits into C_pre (before the")
    print("      misses are knowable), C_ind (post-router, expert-INDEPENDENT -- the shared expert")
    print("      -- the only window a non-predictive loader has) and C_dep (routed MoE + combine,")
    print("      which needs the experts). A loader with no prediction overlaps C_ind and nothing")
    print("      else; a predictor that issued H layers earlier overlaps all of it.")
    print(f"    * f = C_ind / C_layer IS NOT MEASURED. Queue job 165 (a decode nsys profile) will")
    print(f"      measure it. It is swept over {F_SWEEP} and every verdict below")
    print("      is reported as a function of f. The shared expert is 1 of 385, so small f is the")
    print("      likely regime. C_pre : C_dep is fixed at 1:1 and is immaterial to every arm but")
    print("      (cu), since the compute between two waits always sums to one layer.")
    print("    * CONTENTION PENALTY, stated so it can be argued with: compute that runs while >= 1")
    print(f"      expert read is in flight is charged GAMMA = {GAMMA} of its own duration as extra")
    print("      wall time. Calibrated to the ONE end-to-end measurement of concurrent H2D we")
    print("      have: removing stream.wait_stream(compute) cost 14.5 % overall (dfb312c). The")
    print("      '+32 % FFN' from that same commit is RETRACTED (cd3be98: MoE kernels flat within")
    print("      +-1 %, SMs Active 97.9 %, clocks flat -- a host-side wall clock, not a device")
    print("      effect) and is NOT used. The penalty is extra wall time and does NOT advance the")
    print("      I/O clock, and it is charged to every arm alike -- which costs the ORACLE arms")
    print("      most, since they are the ones computing with reads in flight.")
    print("    * UNITS: one resolve call = one LAYER; 40 calls = one decode STEP; one step emits")
    print(f"      ~{ACCEPT} tokens. Every per-step column is per 40 resolve calls.")
    print(f"    * the shipped engine, MEASURED: {STEPS_PER_S_MEASURED} steps/s = "
          f"{STEP_MS_MEASURED:.0f} ms/step = {STEPS_PER_S_MEASURED * ACCEPT:.1f} tok/s, of which")
    print(f"      {100 * FETCH_WAIT_FRAC:.0f} % is expert-fetch wait = {WAIT_MS_MEASURED:.0f} "
          f"ms/step; the remaining {COMPUTE_MS:.2f} ms per")
    print("      resolve call is compute. A PREDICTOR gets to overlap all of it; a loader with no")
    print("      prediction gets only the C_ind slice of it, which is the point of the f sweep.")
    print("    * the oracle has ZERO wrong prefetches by construction, so it adds ZERO extra bytes.")
    print("      Its only costs are the slot it holds and the buffer/bandwidth it takes.")
    print()


ARMS = [
    # (tag, label, H-swept?, kwargs)
    ("0", "TODAY -- synchronous, no lookahead (the shipped path)", False,
     dict(H=0, cap=CAP_TODAY, pool=POOL_TODAY, gated=True, overlap=False, slot_on_issue=True)),
    ("a", "SUBSYSTEM ONLY -- async issue, overlap C_ind ONLY, NO prediction (CORRECTED)", False,
     dict(H=0, cap=CAP_TODAY, pool=POOL_TODAY, gated=True, overlap=True, slot_on_issue=True)),
    ("a!", "SUPERSEDED arm (a) -- the same loader given a FULL layer to overlap into. The misses "
     "do\n      not exist that early; this row exists only to show the size of the old error.",
     False,
     dict(H=0, cap=CAP_TODAY, pool=POOL_TODAY, gated=True, overlap=True, slot_on_issue=True,
          full_layer_window=True)),
    ("b", "ORACLE + IDEALISED subsystem (unbounded pool, slot charged on completion)", True,
     dict(cap=CAP_TODAY, pool=BIG, gated=False, overlap=True, slot_on_issue=False)),
    ("c", "ORACLE + REALIZABLE subsystem (pool 48, slot on issue, gated, cap 6)", True,
     dict(cap=CAP_TODAY, pool=POOL_TODAY, gated=True, overlap=True, slot_on_issue=True)),
    ("c2", "ORACLE + REALIZABLE at device concurrency 2", True,
     dict(cap=2, pool=POOL_TODAY, gated=True, overlap=True, slot_on_issue=True)),
    ("cu", "ORACLE + REALIZABLE, issue gate REMOVED (prices the gate)", True,
     dict(cap=CAP_TODAY, pool=POOL_TODAY, gated=False, overlap=True, slot_on_issue=True)),
]


def run_trace(path, slots, prefix, horizons, f_sweep=F_SWEEP, gamma=GAMMA):
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

    memo = {}

    def sim(pol, **kw):
        key = (pol, tuple(sorted((k, v) for k, v in kw.items())))
        if key not in memo:
            memo[key] = simulate(calls, slots, cut, pol, gamma=gamma, **kw)
        return memo[key]

    arm_kw = {tag: kw for tag, _lbl, _sw, kw in ARMS}

    for pol in ("lru", "agefreq"):
        base = bases[pol]
        scored = base["accesses"]
        m_base = base["demand_fetch"]
        polname = LRUCache.label if pol == "lru" else AgeFreqCache.label
        comp_ms = COMPUTE_MS * N_LAYERS

        # Arm 0 sets the calibration: it IS the shipped path, so it is pinned to 1.73 steps/s.
        # It is also f-independent (it overlaps nothing) and pays no contention penalty (it never
        # computes with a read in flight), so the calibration survives the correction untouched.
        r0 = sim(pol, f_ind=F_REF, **arm_kw["0"])
        wait0 = 1000 * r0["wait"] / steps
        pen0 = 1000 * r0["pen"] / steps
        k_opt = WAIT_MS_MEASURED / wait0 if wait0 > 0 else 0.0
        resid = max(0.0, WAIT_MS_MEASURED - wait0)

        def t_opt(w, pen):
            return w * k_opt + comp_ms + pen

        def t_pess(w, pen):
            return w + resid + comp_ms + pen

        def wp(r):
            return 1000 * r["wait"] / steps, 1000 * r["pen"] / steps

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
        print("         transfer time. Arm 0 reproduces 1.73 steps/s in both, by construction")
        print(f"         (its contention penalty is {pen0:.2f} ms/step: it never computes with a "
              f"read in flight).")
        print("  The contention penalty is compute-side, so it is NOT scaled by the bracket; it is")
        print("  added to both ends unchanged.")
        print()
        print(f"  DETAIL TABLE at f = {F_REF:.2f} (C_ind = {F_REF * COMPUTE_MS:.2f} ms of the "
              f"{COMPUTE_MS:.2f} ms layer), gamma = {gamma:.3f}.")
        hdr = (f"  {'arm':>3s} {'H':>4s} {'hit':>7s} {'blk/step':>9s} {'conv':>6s} {'stol':>5s} "
               f"{'wast':>5s} {'wait/step':>10s} {'pen':>6s} {'vs 0':>8s} {'OPT st/s':>9s} "
               f"{'tok/s':>6s} {'PESS st/s':>10s} {'tok/s':>6s} {'pf slots':>9s} {'max':>5s} "
               f"{'infl':>5s} {'pk':>3s}")

        def row(tag, H, r):
            conv, sto = diff(base["trace"], r["trace"])
            w, pen = wp(r)
            opt = 1000.0 / t_opt(w, pen)
            pess = 1000.0 / t_pess(w, pen)
            print(f"  {tag:>3s} {H:4d} {100 * r['hits'] / scored:6.2f} % "
                  f"{r['demand_fetch'] / steps:9.1f} {100 * conv / m_base:5.1f}% {sto:5,d} "
                  f"{r['wasted']:5,d} {w:10.1f} {pen:6.1f} "
                  f"{(100 * (w + pen - wait0 - pen0) / (wait0 + pen0) if wait0 else 0):+7.1f}% "
                  f"{opt:9.2f} {opt * ACCEPT:6.2f} {pess:10.2f} {pess * ACCEPT:6.2f} "
                  f"{r['spec_avg']:9.1f} {r['spec_max']:5d} {r['mean_inflight']:5.2f} "
                  f"{r['peak_inflight']:3d}")

        best = None
        r_a_ref = None
        for tag, label, swept, kw in ARMS:
            print(f"  {tag:>3s}  {label}")
            print(hdr)
            if not swept:
                r = sim(pol, f_ind=F_REF, **kw)
                row(tag, 0, r)
                if tag == "a":
                    r_a_ref = r
            else:
                for H in horizons:
                    r = sim(pol, H=H, f_ind=F_REF, **kw)
                    row(tag, H, r)
                    if tag == "c":
                        st = t_opt(*wp(r))
                        if best is None or st < best[1]:
                            best = (H, st)
            print()

        wait_a, pen_a = wp(r_a_ref)
        bh = best[0]
        r_c_ref = sim(pol, H=bh, f_ind=F_REF, **arm_kw["c"])
        wait_c, pen_c = wp(r_c_ref)

        # ---------------------------------------------------------- the decomposition, spelled out
        print(f"  WHERE THE {wait0:.0f} ms/step OF MODELLED BLOCKING WAIT GOES AT f = {F_REF:.2f} "
              f"({polname}):")
        print(f"    {wait0 - wait_a:6.1f} ms/step ({100 * (wait0 - wait_a) / wait0:4.1f} %) removed "
              f"by the SUBSYSTEM ALONE -- async issue")
        print("                          overlapped against C_ind ONLY, arm (a), NO PREDICTION.")
        print(f"    {wait_a - wait_c:6.1f} ms/step ({100 * (wait_a - wait_c) / wait0:4.1f} %) "
              f"removed on top of that by a PERFECT oracle at its")
        print(f"                          best horizon (arm c, H={bh}). This is the ENTIRE budget")
        print("                          any predictor is competing for.")
        print(f"    {wait_c:6.1f} ms/step ({100 * wait_c / wait0:4.1f} %) irreducible: the bytes "
              f"still have to cross the device.")
        print("  Contention penalty on top, NOT part of the wait: "
              f"a {pen_a:.1f} ms/step, c {pen_c:.1f} ms/step.")
        print()

        # ---------------------------------------------------------- the decisive table over f
        print(f"  ===== DECISIVE TABLE ({polname}): WHAT PERFECT KNOWLEDGE H LAYERS AHEAD ADDS OVER")
        print("  A REALIZABLE LOADER THAT MAY ONLY OVERLAP C_ind.")
        print("  Currency: PERCENT OF ARM-0's MODELLED BLOCKING WAIT -- deliberately the same")
        print("  currency the superseded commit used ('the loader removes 49-70 %, a perfect")
        print("  oracle adds 9-23 % on top'), so these rows are directly comparable to it.")
        print("  Rows: f = C_ind / C_layer, the post-router expert-INDEPENDENT share of a layer's")
        print("  compute -- UNMEASURED; queue job 165 (decode nsys) will measure it.")
        print("  Cells: 100 x (wait_a - wait_c) / wait_0 at horizon H -- the oracle's ADDITION on")
        print("  top of the loader, arm (c), the realizable subsystem. =====")
        hh = "".join(f"{('H=' + str(H)):>7s}" for H in horizons)
        print(f"  {'f':>5s} {'loader':>7s} |{hh} | {'bestH':>5s} {'oracle+':>8s} {'ideal+':>7s} "
              f"{'ratio':>6s} {'e2e OPT':>8s} {'e2e PESS':>9s}")
        flip = []
        for f in f_sweep:
            r_a = sim(pol, f_ind=f, **arm_kw["a"])
            wa, pa = wp(r_a)
            sa_opt, sa_pess = t_opt(wa, pa), t_pess(wa, pa)
            loader_pp = 100 * (wait0 - wa) / wait0
            cells = []
            bestf = None
            for H in horizons:
                r_c = sim(pol, H=H, f_ind=f, **arm_kw["c"])
                wc, pc = wp(r_c)
                cells.append(100 * (wa - wc) / wait0)
                if bestf is None or wc < bestf[1]:
                    bestf = (H, wc, pc)
            Hb, wc_b, pc_b = bestf
            oracle_pp = 100 * (wa - wc_b) / wait0
            ideal_pp = max(100 * (wa - wp(sim(pol, H=H, f_ind=f, **arm_kw["b"]))[0]) / wait0
                           for H in horizons)
            ratio = (wait0 - wa) / (wa - wc_b) if wa - wc_b > 1e-9 else float("inf")
            g_opt = 100 * (sa_opt / t_opt(wc_b, pc_b) - 1.0)
            g_pess = 100 * (sa_pess / t_pess(wc_b, pc_b) - 1.0)
            flip.append((f, oracle_pp, loader_pp, g_opt, g_pess, Hb, ratio))
            cs = "".join(f"{c:+6.1f}%" for c in cells)
            print(f"  {f:5.2f} {loader_pp:+6.1f}% |{cs} | {Hb:5d} {oracle_pp:+7.1f}% "
                  f"{ideal_pp:+6.1f}% {ratio:5.2f}x {g_opt:+7.1f}% {g_pess:+8.1f}%")
        print("    loader   = what arm (a) removes on its own, 100 x (wait_0 - wait_a) / wait_0.")
        print("               The superseded chronology put this at 49-70 %.")
        print("    ideal+   = the same addition against the IDEALISED subsystem, arm (b): the")
        print("               ceiling if the plumbing were perfect too.")
        print("    ratio    = loader / oracle+, the quantity the superseded commit called '3-6x'.")
        print("    e2e      = end-to-end decode-rate speedup of (c) over (a) at the same f, both")
        print("               ends of the calibration bracket, contention penalty included in")
        print("               both. The bracket is wide because it is a bracket on the UNMODELLED")
        print("               read overhead, which is 3-4x the modelled transfer; it is not a")
        print("               statement about prediction.")
        print()

        # ---------------------------------------------------------- the verdict as a function of f
        print(f"  VERDICT AS A FUNCTION OF f ({polname}). The accepted thresholds, in the currency")
        print("  above: an addition of +10-20 % leaves predictor work CLOSED; +30-50 % makes a")
        print("  router-logit prediction head one of the highest-upside decode experiments left.")
        for f, oracle_pp, loader_pp, g_opt, g_pess, Hb, ratio in flip:
            if oracle_pp >= 30.0:
                v = "TRAIN ONE"
            elif oracle_pp >= 20.0:
                v = "borderline"
            else:
                v = "stays CLOSED"
            print(f"    f = {f:4.2f}  loader {loader_pp:+5.1f} %  oracle adds {oracle_pp:+6.1f} % "
                  f"at H={Hb:3d}  ratio {ratio:5.2f}x  e2e {g_opt:+6.1f}/{g_pess:+5.1f} %   {v}")
        closed = [f for f, o, *_ in flip if o < 30.0]
        opened = [f for f, o, *_ in flip if o >= 30.0]
        if not opened:
            print("    The +30 % line is not crossed at any f in the sweep: the verdict does NOT")
            print("    flip, and predictor work stays closed on this trace and policy.")
        elif not closed:
            print(f"    The +30 % line is crossed at EVERY f in the sweep, including the largest")
            print(f"    tested, f = {f_sweep[-1]:.2f}. There is no flip point inside the sweep: the")
            print("    verdict is 'train one' across the whole plausible range, and it only gets")
            print("    stronger as f falls, which is the regime the shared expert (1 of 385) puts")
            print("    us in.")
        else:
            print(f"    The verdict flips between f = {max(opened):.2f} and f = {min(closed):.2f}.")
            print("    A LARGER f means a larger window for the non-predictive loader, so the")
            print("    verdict moves toward 'closed' as f RISES and toward 'train one' as it falls.")
        print()

        # ---------------------------------------------------------- contention sensitivity
        print("  CONTENTION SENSITIVITY. gamma does not change any schedule, only the wall clock")
        print("  charged for compute that ran with a read in flight, so this is exact, not a")
        print(f"  re-simulation. Rows: gain of (c) over (a) at each f, OPT end.")
        print(f"  {'f':>5s}" + "".join(f"{('g=' + format(g, '.3f')):>10s}" for g in GAMMA_SWEEP))
        for f in f_sweep:
            r_a = sim(pol, f_ind=f, **arm_kw["a"])
            bH = [x for x in flip if x[0] == f][0][5]
            r_c = sim(pol, H=bH, f_ind=f, **arm_kw["c"])
            cells = []
            for g in GAMMA_SWEEP:
                sa = t_opt(1000 * r_a["wait"] / steps, g * 1000 * r_a["contended"] / steps)
                sc = t_opt(1000 * r_c["wait"] / steps, g * 1000 * r_c["contended"] / steps)
                cells.append(100 * (sa / sc - 1.0))
            print(f"  {f:5.2f}" + "".join(f"{c:+9.1f}%" for c in cells))
        print("    gamma = 0 removes the penalty entirely; 0.145 is the calibration; 0.30 is the")
        print("    direction a per-unit calibration would move it. Higher gamma HURTS the oracle,")
        print("    because the oracle is the arm that keeps reads in flight under compute.")
        print()

        # ---------------------------------------------------------- how accurate is accurate enough
        print(f"  RECALL SWEEP -- arm (c) at H={bh}, f={F_REF:.2f}, oracle degraded to name only")
        print("  `recall` of the true future misses and NOTHING false. Perfect precision, so every")
        print("  row is still an UPPER BOUND on a real predictor at that recall: a real one also")
        print("  pays for false positives in extra bytes and extra stolen slots.")
        print(f"  {'recall':>7s} {'blk/step':>9s} {'conv':>6s} {'wait/step':>10s} "
              f"{'of oracle win':>14s} {'OPT st/s':>9s} {'PESS st/s':>10s} {'vs (a) OPT':>11s}")
        win = wait_a - wait_c
        sa_opt_ref = t_opt(wait_a, pen_a)
        for rc in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0):
            r = sim(pol, H=bh, f_ind=F_REF, recall=rc, **arm_kw["c"])
            w, pen = wp(r)
            conv, _ = diff(base["trace"], r["trace"])
            got = (wait_a - w) / win if win > 0 else 0.0
            print(f"  {rc:7.2f} {r['demand_fetch'] / steps:9.1f} {100 * conv / m_base:5.1f}% "
                  f"{w:10.1f} {100 * got:13.1f} % "
                  f"{1000.0 / t_opt(w, pen):9.2f} {1000.0 / t_pess(w, pen):10.2f} "
                  f"{100 * (sa_opt_ref / t_opt(w, pen) - 1):+10.1f}%")
        print()
    print()


def legend():
    print("  COLUMNS")
    print("    arm       0 shipped sync path / a async loader overlapping C_ind ONLY, with NO")
    print("              prediction / a! the SUPERSEDED version of (a) that was handed a whole")
    print("              layer to overlap into / b oracle on an idealised subsystem / c the same")
    print("              oracle on a subsystem we could ship / c2 same at device concurrency 2 /")
    print("              cu same but WITHOUT the issue gate.")
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
    print("    pen       contention penalty, ms/step: GAMMA x the compute that ran with at least")
    print("              one read in flight. Extra wall time, not wait; added to both brackets")
    print("              unscaled. Arm 0 pays zero, which is why the calibration is unaffected.")
    print("    OPT/PESS  implied decode steps/s and emitted tok/s at accept 3.6, the two ends of the")
    print("              calibration bracket described above each table. The truth is inside it, and")
    print("              WHICH END is an empirical question this tool cannot settle.")
    print("    pf slots  cache slots held by prefetched-but-not-yet-used entries, mean over the")
    print("              scored window / max. The standing capacity cost of speculation.")
    print("    infl/pk   mean (time-weighted) and peak reads in flight -- is the schedule feasible?")
    print()
    print("  HOW TO READ IT. (a) minus (0) is what the LOADER is worth with no predictor at all,")
    print("  and (a!) minus (a) is what the superseded chronology had wrongly credited to it.")
    print("  (b) minus (a) is what PREDICTION is worth once the plumbing exists -- that difference,")
    print("  not (b) itself, is the budget any predictor is competing for. (c) is what survives the")
    print("  constraints we actually have, and (b) minus (c) is what better plumbing would buy on")
    print("  top. (cu) minus (c) prices the issue gate. THE DECISIVE TABLE is (c) over (a) at the")
    print("  same f, end to end, because that is the realizable predictor against the realizable")
    print("  loader; every cell in it moves with f, and f is not measured yet.")
    print()
    print("  SCOPE. Decode only, two traces (prose and code-heavy), one cache size (5,328 slots =")
    print("  the shipped lru_slots), device concurrency 2 and 6, staging pool 48. Cache behaviour is")
    print("  an exact replay of measured routing; LATENCY IS MODELLED, NOT MEASURED, from three")
    print("  measured device numbers and the measured decode split. AND IT IS NOW ALSO PARAMETRIC")
    print("  IN AN UNMEASURED COMPUTE SPLIT: f = C_ind / C_layer is swept, not measured, until the")
    print("  decode nsys profile (queue job 165) lands. The contention penalty is calibrated to a")
    print("  single end-to-end A/B. Nothing here was run on a GPU.")


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
    ap.add_argument("--f-sweep", type=float, nargs="+", default=F_SWEEP,
                    help="C_ind as a fraction of one layer's compute; UNMEASURED, so it is swept")
    ap.add_argument("--gamma", type=float, default=GAMMA,
                    help="contention penalty on compute that runs with a read in flight")
    a = ap.parse_args()

    print()
    print("  prefetch_ceiling.py -- the UPPER BOUND on expert prefetch, and which HALF is")
    print("  prediction. A PERFECT oracle knows exactly which (layer, expert) pairs the next H")
    print("  resolve calls will MISS and prefetches precisely those, nothing else: zero wasted")
    print("  prefetches by construction, therefore ZERO extra NVMe bytes. What is left is the value")
    print("  of OVERLAP -- and arm (a) splits off the part of that value needing no prediction.")
    print()
    print("  CORRECTED 2026-09-15, AND THE CORRECTION REVERSES THE PREVIOUS RESULT. The first")
    print("  version let the no-prediction loader overlap a WHOLE LAYER of compute with reads it")
    print("  could not have known about that early: the expert ids do not exist until the router")
    print("  at the end of graph A has run (engine/fastdecode.py:13-15). A non-predictive loader")
    print("  can only overlap post-router, expert-INDEPENDENT work. That share is not measured")
    print("  yet, so it is swept as f and every verdict is a function of f. Arm (a!) reproduces")
    print("  the superseded arm so the size of the error is on the page, not hidden.")
    print()
    assumptions()

    for path in a.logs:
        if not os.path.exists(path):
            print(f"  (skipping {path}: not present)\n")
            continue
        run_trace(path, a.slots, a.prefix, a.horizons, a.f_sweep, a.gamma)

    legend()
    return 0


if __name__ == "__main__":
    sys.exit(main())
