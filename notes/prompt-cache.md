# Extend-only prompt cache for the DS4.1 engine

**Why this is the top item.** `V41Engine.generate()` called `self._reset()` unconditionally, so every
request re-prefilled its whole context. Measured: prefill runs at 86–99 tok/s and 27–47 MB of NVMe
per prompt token, so a 10k-context agent turn with a 200-token reply is ~120 s of prefill against
~32 s of decode — **79 % of the wall, paid again every turn on almost entirely identical tokens.**
Every decode lever we have measured (the CB3 cache's 1.42×, dense FP4's +3.4 %) acts on the other
21 %.

## What makes it cheap on *this* engine

Three properties, all checked in the code rather than assumed:

1. **One sequence, one lock.** No paged allocator and no sharing between requests, so there is
   nothing to design: the cache is simply the caches that are already there, not cleared.
2. **`window_size` is 128 and there are only four KV source layers** (`[2, 8, 14, 20]`). The only
   state that does not survive a jump backwards is the compressor's `pending`, at two `[512]` fp32
   rows per source layer — **16 KB per checkpoint**. Every prefill chunk boundary can be
   checkpointed for nothing.
3. **The Engram hash state is position-indexed.** `NgramHashState.cache` is `[B, max_seq_len]` of
   compressed token ids, written at `start_pos:start_pos+seqlen` and read back by `gather` at
   `positions - shift`. A reused prefix has identical ids at identical positions, so it replays
   correctly with no work and no bookkeeping. This was an assumption in the TODO; it is now read
   out of `inference/engram.py:167-172`.

## The three things that bound how far back it can resume

Not one of these is obvious from the outside, and each is a wrong answer if ignored:

* **The longest common prefix, on token ids** — never on lengths. A chat template can rewrite
  earlier turns, so `len(previous prompt)` is not a safe resume point.
* **The SWA replay tail.** `decoder_replay` runs layers 21..39 over the last `window_size` prompt
  positions and takes its inputs from `Model._rep`, which **only the encoder prefill pass fills** —
  decode steps never append to it. So at least `window_size` positions must be re-encoded, and the
  resume point is the last chunk boundary at or below `lcp - window_size`. With `MAX_CHUNK = 2048`
  that re-encodes between 128 and 2176 tokens: on a 10k turn, still 5–6× less prefill.
* **The window ring.** `RING = 4096`, and SWA at the resume point reads back 128 positions that the
  *previous* request wrote. If the prefix diverges far behind the end of a long context, those ring
  rows have been overwritten even though the ids match. Guarded explicitly
  (`len(ctx) - (start - window) > RING` → no resume), because the failure mode is silent garbage,
  not an exception.

## What it is not

**Not bit-identical to a full prefill.** The reused positions were computed under a different chunk
alignment, so their GEMM shapes differed. Same class of drift as any other reshaping — but it is
drift, and the test asserts *token* agreement, not bitwise agreement.

**Not on by default.** `DSV41_PROMPT_CACHE=1`. Every number in `notes/` was taken without it, and it
changes what a request computes.

## Implementation

* `engine/model.py` — `Caches.checkpoint(n)` / `Caches.resume_at(n)`. `resume_at` also drops every
  checkpoint past `n`, since those describe positions the resumed prefill is about to rewrite with
  different tokens; keeping them would let a later request resume onto a prefix that was never
  written. The existing `rollback()` is untouched: it reaches back only into the last chunk, which
  is all speculative rejection needs.
* `engine/v41_engine.py` — `_reset()` split into cache-clearing and `_reset_stats()`, so a resumed
  request still reports its own I/O instead of accumulating across a conversation.
  `_resume_point(prompt)` decides, before anything is cleared. The prefill loops start at
  `self._resumed_from` and checkpoint each boundary.
* Stats: `prompt_cache_reused` and `prompt_tokens_prefilled` are exported, and **`prefill_tok_s` is
  now over what was actually prefilled** — dividing the full prompt by the time to prefill a suffix
  would report a speedup that is really just work not done.
* The retained ids satisfy `ctx_ids == (prompt + out)[:caches.len]`; the last emitted token is
  always one the decode loop has not forwarded yet. If that invariant fails the cache is dropped
  rather than resumed onto.
* `engine/test_prompt_cache.py` — two turns of a real conversation; turn 2 warm must emit the same
  tokens as turn 2 on a cleared engine, and must actually have resumed (a test that silently takes
  the cold path would pass for the wrong reason).

## First measurement (2026-09-14, `engine/test_prompt_cache.py`)

Real two-turn conversation, greedy, CB3 + `DSV41_DENSE_FP4=attn,wo_a` + fp8 head, `max_seq=16384`:

| turn 2 | reused | prefilled | prefill | NVMe |
|---|---|---|---|---|
| warm (resumed) | 2048 of 3416 | 1368 | **10.19 s** | 28.78 GB |
| cold (cache cleared) | 0 | 3416 | 16.32 s | 27.71 GB |

**18 tokens identical warm and cold** — the correctness bar. TTFT **1.60×** on a 3.4k-token turn,
which is the *worst* case for this design: the resume point is a 2048 chunk boundary, so a short
turn throws away most of what it could have reused. The win grows with context — a 22k turn resumes
within 2176 tokens of the end instead of re-prefilling 22k.

NVMe barely moves (28.8 vs 27.7 GB) because the expert LRU is shared between the two arms; the warm
arm reads *more* per prefilled token, not less. So this is a compute win at the arena coverage we
run, not an I/O win. The I/O half is the layer-major transpose's job, and the two compose.

## At real context: **TTFT stops being linear in the prompt** (2026-09-14)

`promptcache-longctx`, three conversation lengths, greedy. The right baseline for "turn 2 with the
cache" is a full prefill of a prompt that length — turn 1 of the cache-busted arm, within 1 % of
turn 2's length:

| turn-2 tokens | prefill, no cache | prefill, cached | | NVMe, no cache | cached | |
|---|---|---|---|---|---|---|
| 5,955 | 62.9 s | 22.0 s | **2.85×** | 202.3 GB | 79.8 GB | 2.54× |
| 11,383 | 118.6 s | 20.3 s | **5.83×** | 360.6 GB | 66.0 GB | 5.46× |
| 22,249 | 203.6 s | 22.3 s | **9.12×** | 578.1 GB | 70.8 GB | 8.17× |

**The ratio is not the result — the constant is.** Turn 2 costs **20.3–22.3 s at every length**,
because the resumed suffix is always the reply plus the follow-up plus chunk-boundary rounding:
1,859 / 1,143 / 1,769 tokens. So the cache converts TTFT from *linear in context* to *constant in
the increment*, and the speedup only looks bigger at long context because the thing it replaces
grows. Extrapolated to the 27,200-token point of the long-context curve, turn 2 would be ~22 s
against 261.3 s — **~12×** — and still ~22 s at any context the engine can hold.

Reused is always a multiple of 2,048 (4,096 / 10,240 / 20,480), which is the chunk-boundary rounding
working as designed. The residual 21 s is ~1,800 tokens at 86 tok/s, consistent with the measured
prefill rate; its ~40 MB per prefilled token against the steady-state 27 is the suffix's experts
not being warm.

**A flaw in the first version of this measurement, recorded because the number it printed looked
plausible.** The script's "cold" arm busts the cache with a short unrelated prompt and then runs
turn 1 — which repopulates it, so turn 2 of the "cold" arm resumes too. Comparing warm turn 2
against "cold" turn 2 therefore compared two cached turns and printed **1.04–1.07×**: a believable
small number that would have read as "the cache barely helps at long context" and buried the result.
The genuinely cold rows (`reused 0`) were in the same table. Fixed in the tool.

## It composes with layer-major (2026-09-14)

The first layer-major commit took exactly one checkpoint, at the resume point — position 0 on a
first turn — and `_resume_point` rejects `start < MAX_CHUNK`. So `DSV41_LAYER_MAJOR=1` **silently
disabled this cache**: no error, no slowdown, just no resume, ever. Since the cache is worth more on
agent traffic than the transpose is (154 s → ~21 s per turn against 154 → 72), shipping layer-major
as a default would have traded the larger lever for the smaller one invisibly.

Fixed by assembling the checkpoints inside the pass (fork `dc74db5`): `Caches.checkpoint(n)` wants
every layer's compressor state as of *n*, and chunk-major can take it in one call because the whole
stack is at *n* at that moment — layer-major never is, since layer L is at the end of the prompt
while L+1 is still at the start. So each layer's `pending` is recorded as it crosses each boundary
and the checkpoints are assembled at the end.

**Measured with both features on** (`layer-major-cache-ab`):

| turn-2 tokens | cold prefill | cached | | reused |
|---|---|---|---|---|
| 5,955 | 42.4 s | **21.6 s** | 1.96× | 4,096 |
| 11,383 | 57.7 s | **20.2 s** | 2.86× | 10,240 |
| 22,249 | 91.8 s | **22.2 s** | 4.14× | 20,480 |

**`reused` is non-zero at every length**, which is the assertion the fix exists for. And the *cold*
column is layer-major working underneath: 91.8 s at 22k against the **203.6 s** the same arm
measured chunk-major. So the two compose as intended —

* **cold turn**: 203.6 → 91.8 s, from the transpose;
* **every later turn**: **~21 s**, from the cache, and still flat in context.

The cached column is unchanged from the cache's own measurement (22.0 / 20.3 / 22.3 s), which is the
right outcome: the cache resumes past the prefill either way, so layer-major cannot help a turn that
barely prefills — and, now, cannot hurt it either.

## Token equality with both features on (2026-09-14)

The half of `layer-major-cache-ab` that VOIDed on runner ordering, re-run: 12,621-token prompt,
`DSV41_PROMPT_CACHE=1` and `DSV41_LAYER_MAJOR=1` both live.

```
chunk-major: prefill 30.209s, nvme 77.92 GB, prefill misses 4116
layer-major: prefill 10.387s, nvme 18.36 GB, prefill misses 1333
PASS: 20 tokens identical chunk-major and layer-major
prefill expert loads 4116 -> 1333 = 3.09x fewer; NVMe 77.92 -> 18.36 GB
```

So the transpose still emits the same tokens with the cache's checkpoint assembly running inside its
hot loop. Both halves of the composition question are now green: the cache resumes under
layer-major, and layer-major is token-identical under the cache.

*(The runner's earlier VOID was ordering: it built a second engine while the server still held
83 GB, and the upstream pre-flight check refused it at MemAvailable 3.6 GB. The fix is to stop the
server and wait on MemAvailable first — and the check refusing rather than loading for three minutes
and dying is exactly why it was worth cherry-picking.)*
