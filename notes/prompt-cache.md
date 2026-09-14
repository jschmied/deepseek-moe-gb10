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
