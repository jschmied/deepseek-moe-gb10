# Where an expert arena should live on GB10, measured

2026-09-13, on the idle box (no serving, load 0.01, 113 GiB free). Three programs, sources in the
session scratchpad, every number the mean of **three starts** with the spread quoted. These replace
four constants our notes had been carrying without a build stamp — the same failure mode that nearly
bought a 4–6 day project off `fp8bench.txt`.

Method: `hostmem_bw.cu` times a grid-stride `float4` read kernel (48 SMs × 32 blocks × 256 threads)
over an 8 GiB arena allocated three ways, plus an "expert gather" kernel where each of 640 blocks
walks one 13.79 MiB region — the DS4.1 expert size and roughly a decode step's arena traffic.
`odirect_dest.cu` issues `O_DIRECT` `pread`s of one expert at random 4 KiB-aligned offsets in the
**real DS4.1 shards** we have locally (31 GB, shards 3–6), into each destination.

## 1. An SM reads pinned host memory at 81 % of device-memory speed

| arena allocation | stream read | expert gather |
|---|---|---|
| `cudaMalloc` (device) | 240.0–240.7 GB/s | **281.4–283.0 GB/s** |
| `cudaHostAlloc` (mapped, SM-readable) | 212.2–214.7 | **228.3–230.0** |
| `cudaMallocManaged` | 160.7–161.4 | 187.7–187.9 |

`cudaHostGetDevicePointer` returns the identical pointer (`unifiedAddressing 1`,
`pageableMemoryAccess 1`), so nothing crosses PCIe — all three land in the same LPDDR5X and the
spread is caching/coherency attributes, not topology.

**This refutes the prediction that a host-pinned arena would read at the H2D copy rate (~59 GB/s)
and therefore halve the MoE kernel.** It is 19 % slower, not 2× slower. Managed memory is the one to
avoid: 33 % off device speed, for no benefit here.

## 2. `O_DIRECT` into an SM-readable buffer costs nothing; into device memory it is impossible

Chunk 4 MiB, per-destination, threads 1→8:

| destination | T=1 | T=2 | T=4 | T=8 |
|---|---|---|---|---|
| `posix_memalign` (plain) | 3.74 | 5.05 | 5.15 | 4.96 GB/s |
| `cudaHostAlloc` (SM-readable) | 3.76 | 5.06 | 5.31 | 5.07 |
| `cudaMallocManaged` | 3.81 | — | 5.37 | — |
| `cudaMalloc` (device) | **EFAULT (errno 14)** | | | |

Making the landing zone readable by the SMs is free. Device memory cannot be a DMA target from the
block layer at all, which is also the whole reason GPUDirect Storage has nothing to offer here.

## 3. `read_chunk_mb=4` is the wrong constant — it costs 41 % at decode concurrency

Same expert read, varying only the chunk the engine splits it into. Three reps:

| chunk | T=1 | T=2 | T=4 |
|---|---|---|---|
| 1 MiB | 2.58 | 3.42 | 5.52 |
| **4 MiB** (shipped) | 3.79 / 3.98 / 4.15 → **3.97** | 5.50 / 6.12 / 6.27 → 5.96 | 5.56–6.80 |
| 8 MiB | 4.85 / 4.97 / 5.28 → 5.03 | 6.77 / 6.85 / 6.77 → 6.80 | 6.63–6.77 |
| **whole expert, one `pread`** | 5.48 / 5.65 / 5.61 → **5.58** | 6.80 / 6.83 / 6.83 → **6.82** | 6.58–6.69 |

A decode step misses ~0.5–1.6 experts per layer, so the engine lives at **T=1–2**, exactly where the
chunking hurts: **+41 % at T=1 and +14 % at T=2 from simply not splitting the read.** By T=4
everything converges to ~6.7 GB/s, which is why a pure "queue depth" framing missed this — the loss
only exists at low concurrency, and low concurrency is the case that matters.

Two consequences. First, this is a **one-constant change**, not a redesign. Second, it lowers the bar
for batching a layer's misses: **two experts in flight already reach the device ceiling** (6.82 of
~6.8 GB/s), so that work needs to buy a queue depth of 2, not 8.

## 4. So should the arena be host-pinned? Marginally, and it stops paying as the cache improves

With the arena on the host you delete the H2D copy but pay 19 % on every arena read. Per decode step,
using the stored 12.12 GB of arena traffic and 852 MB of misses at ~93 % coverage:

* device arena: 12.12 / 281.6 = **43.0 ms** kernel, plus 0.852 / 59.4 = **14.3 ms** copy → 57.3 ms
* host arena: 12.12 / 229.0 = **52.9 ms** kernel, no copy → 52.9 ms

**Net −4.4 ms/step out of ~318 — about 1.4 %.** Break-even is a miss rate of **4.85 %**: above ~95.2 %
coverage the host arena is a *loss*, and ds-09/ds-06 are pushing coverage in exactly that direction.

The conclusion is not "don't do it" but "do it for the right reason". The bandwidth arithmetic is a
wash. What a host-pinned arena actually removes is **structural**: the staging lease held across the
H2D and `stream.synchronize()`, the twelve pinned buffers, and the serialisation that makes the
engine measure 2.68 GB/s against a device that gives 5.6 single-threaded. Fix the lease first — it is
cheaper and it keeps the faster arena.

## What this changes in the survey

* Idea 2c (`O_DIRECT` straight into a `cudaHostAlloc` arena) is **viable but near-neutral**, not the
  large win it was ranked as, and not the dead end the kernel test was expected to prove.
* A new item outranks most of the list: **stop chunking expert reads**, +41 % on the read term at
  decode concurrency, one constant.
* Ideas 2b (drop the staging lease) and 2d (batch a layer's misses) get *cheaper* — the target queue
  depth is 2.
* GPUDirect Storage stays dead, now for a reason measured here rather than quoted: device memory
  returns `EFAULT` to the block layer.
