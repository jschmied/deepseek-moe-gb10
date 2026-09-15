# Engine v2: an event-driven expert loader

*Drafted 2026-09-15, after a decode profile showed the GPU busy **3 % of the span** while the main
thread sat blocked 75 %. This is a design note, not a plan of record. Nothing here is built.*

## Why a v2 rather than more tuning

Every local optimisation this week landed on the same number. Thread counts swept 6/12 → 48/96:
**1.7 %**. Chunk size, eviction policy, coalescing, prefetch: the achieved read rate stays ~2.78 of a
measured 6.82 GB/s. The reason is structural, and the profile names it:

| decode, 104 s span | |
| --- | --- |
| GPU busy | **3.0 s (3 %)** |
| main thread blocked in `sem_wait` | **78.1 s (75 %)** |
| ... with an NVMe read in flight | 47.4 s |
| ... with no tracked device work at all | 26.2 s |
| `cudaMemcpyAsync` | **19.90 s over 21,132 calls, 942 us each** |

The engine is **fork-join**: `_layer_ab` is `_layer_a(L)` -> `resolve()` -> `_layer_b(L)`, and
`resolve` ends in `list(self.pool.map(...))`. The main thread is the scheduler and the loader is a
function it calls and waits inside. Nothing can issue work except a call that immediately blocks on
it, which is why the device cannot be kept busy at any thread count.

## The four false dependencies, each measured

1. **`stream.wait_stream(compute)`** is a per-slot dependency implemented as *wait for all compute* --
   and the event is recorded in the sink, AFTER the ~5 ms read, so under EARLY_SUBMIT it waits on
   attention queued *during* that read: precisely the work it existed to overlap. This is the likely
   mechanism behind the unexplained 14.5 % and behind the 942 us memcpys.
2. **The staging lease** is held from submit to completion, so a demand miss can block on a *buffer*
   while bandwidth is idle.
3. **`join_pending()`** is a barrier over every pending read, when layer L's MoE needs only layer L's
   experts.
4. **`_layer_b` computes the routed MoE before the shared expert**, so the one piece of
   expert-independent post-router work cannot overlap the reads. Today nothing overlaps at all.

## Shape

    router (producer)                loader service (continuous)         consumer
    ---------------------            ---------------------------        ------------------
    publishes (layer, ids, gen)  ->  work queue                          waits on the slot
                                     reserve slot (host, cheap)          handles IT needs
                                     read -> staging                     per-slot ready event
                                     handoff -> release lease early
                                     H2D -> record per-slot event   ->

Principles, each traceable to a measured defect above:

* **Per-slot readiness, not a global barrier.** A CUDA event per slot; a consumer waits on the slots
  it uses. Separates *correctness ordering* (this slot's previous reader must finish) from
  *bandwidth policy* (when H2D may run). Those are different concerns the current code fuses.
* **Release the lease at handoff.** Buffer count stops being coupled to read duration.
* **Backpressure on device queue depth, not thread count.** The device saturates near 2 concurrent
  reads; threads exist to hide per-read latency, not to add bandwidth. CAVEAT: that 2 was measured
  while H2D was serialised behind compute, so re-measure it once (1) is fixed before treating it as
  a design constant.
* **The loader runs continuously.** It is a service fed by a queue, not a callee. This is what makes
  any lookahead -- structural or predicted -- expressible at all.

## What v2 keeps from v1, verbatim

The kernels and the storage layer are measured-good and are not in question: CB3 v3 PTX kernels, the
unpack path and its layer cache, the FP4 MoE kernels, `ShardFile.expert_runs` and the O_DIRECT read
geometry, the CB3 native cache reader, the arena, and the `age/(1+count)` eviction policy. v2 is a
rewrite of the *execution and scheduling* layer only.

## How it gets validated, and why that is unusually cheap here

`engine/test_io_path.py` already pins 11 invariants with mutation checks -- torn slots, lease
conservation under exceptions, join completeness after a raising load, cross-layer safety, the
two-pool deadlock requirement, and the slot-reuse ordering that only the stream barrier provides.
Those are semantics, not implementation, so **they are v2's acceptance tests, written before v2
exists**. A rewrite normally has to discover what the old system guaranteed; here it is executable.

Sequence:
1. v2 as a **fully mocked simulation** -- no GPU, no model, no NVMe -- driven by the real captured
   route traces, validated against those invariants and against v1's measured fetch counts.
2. Swap in v1's real logic block by block behind the same `ExpertStore` interface.
3. v1 stays the default. Every arm is an A/B on the box, warm reps only, engine counters only.

## Open questions this design must NOT assume away

* **f**, the expert-independent fraction of per-layer compute (job 175). The shared expert is 1 of
  385; if f is ~0.02 the structural overlap window is small and the value of the restructuring rests
  on the other three dependencies, not on overlap.
* **The 26.2 s of untracked wait.** CPU work is invisible to the syscall tables, so "no device work"
  is an upper bound on idleness, not proof of it. If that time is real CPU work, a queue-based
  loader inherits it rather than removing it.
* **Whether 2-deep device concurrency survives** once H2D is not serialised behind compute.

Designing around an unmeasured constant is how the arena sizer got a 0.82 fudge that cost three
jobs. These three are measurements, not assumptions, and two of them are already queued.
