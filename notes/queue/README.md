# The DS4.1 work queue

Same tool as the Qwen queue (`qnext`), pointed at a second queue file. It was already generic — it
runs an arbitrary `cmd` array and gates on prereqs, artifact existence, `needs_user`, a lock, and the
job's **own `== ALL DONE ==` sentinel** rather than elapsed time. Two changes were needed:

* **Per-queue lock and logs.** They now follow the queue file (`QNEXT_LOCK` / `QNEXT_LOGDIR` default
  to the queue's directory), so the DS4.1 queue and the Qwen one cannot claim each other's jobs or
  overwrite each other's logs. `QNEXT_BUSY_GLOB` makes the systemd exclusion configurable too.
* **A `guard` field.** A prereq says "another job finished"; a guard says "the world is in the right
  state *now*". DS4.1 jobs need the server **up**, which no prereq can express. It is a shell
  command that must exit 0.

Usage:

```
QNEXT_QUEUE=~/ds41-queue/queue.json tools/qnext.py --dry-run   # what would start, and why not the rest
QNEXT_QUEUE=~/ds41-queue/queue.json tools/qnext.py             # claim one job, run to its sentinel, stop
```

## Why this is worth the twenty minutes

Three of today's failures are ones the queue catches by construction:

* **The block sweep died silently.** I ran it with stderr filtered through `grep`, the block-3 arm
  errored, and the loop marched on to block 7 as though nothing had happened — I only noticed because
  the output was missing. qnext records a job that exits without a sentinel as **`unknown`, never
  `done`**.
* **The bench ran with a flag it does not have** (`--base-url`), wasting a cycle. A job that cannot
  reach its sentinel is caught the same way.
* **`test_cb3_cache.py` was invoked against a cache file that did not exist yet** during development.
  That is exactly `NO ARTIFACT`, and the dry run above still reports it for the two tools in the
  queue that are not written yet.

And the fourth is the one it does *not* catch, which is worth saying: nothing here would have caught
`TRANSIENT_SLOTS=8` being wrong for streaming, because that job started, ran and printed its
sentinel. A queue enforces that work happened, not that it was correct.

## Current contents

| job | gate | what it decides |
|---|---|---|
| `ds41-block-sweep` | guard: server up | acceptance vs verify width 4/8/10 — the last open question on whether widening is free |
| `ds41-engram-ablation` | guard: server up | zero the Engram rows and confirm the output **changes** — closes the gate blind spot the field demonstrated |
| `ds41-free3-fit` | offline | free 3-bit row codebook by DP against CB3's E2M1 subset (−17 % weight error at equal bytes) |
| `ds41-nll-cb3-vs-fp4` | **needs_user** | held-out NLL of CB3 against FP4; wants the 296 GB of layer shards back against 141 GB free |
