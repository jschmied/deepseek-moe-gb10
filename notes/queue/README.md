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

## Running it unattended: what the watchdog does and does not cover

The heartbeat cron is the watchdog. It fires every 30 minutes, refuses to start a second job while
one is running, and is told to diagnose a `void`/`unknown` from its log before re-queuing rather than
restarting blindly. Two holes had to be closed before a night run was safe:

* **`expect_min` was documentation.** `subprocess.run` had no timeout, so a hung job would have held
  the queue until morning with nothing to break the tie. It is now enforced at **3× expect_min** —
  generous enough never to kill a slow-but-working job, short enough that one wedged process cannot
  cost eight hours. Negative-tested: a `sleep 600` at `expect_min: 0` is killed and recorded **void,
  rc=-9**, not done.
* **Nothing restarted the server.** Every measurement job guards on `/health`, so a dead server would
  have failed every guard and stalled the whole night silently. Jobs now carry an optional `recover`
  command, run **once** before the guard is re-checked; all six server-dependent jobs restart the
  engine on the reference configuration.

Still not covered, and worth knowing:

* **The cron is session-only.** If this Claude session ends, the watchdog ends with it. The queue
  file and its state survive on disk, so the queue can be resumed by hand.
* **A job that fails the same way twice** will be re-queued twice unless the tick reads the log. That
  is a judgement the heartbeat prompt asks for; it is not enforced by the tool.
* **Correctness.** As above: a queue enforces that work happened, not that it was right.

## 2026-09-14: a job was killed by the next claim attempt, and the log could not say so

`longctx-profile` vanished 10–14 minutes into a 60-minute run. Driver and child both gone, queue
left at `state: running` with no `finished`/`rc`, **log 0 bytes**.

**Cause.** The driver was launched as `tmux kill-session -t ds41q; tmux new-session -d -s ds41q
"... qnext.py ..."` — one recycled session name. A later claim attempt ran the same line, and the
`kill-session` killed the session that was running the job. The fingerprint is in
`qnext-drive.log`: the `== qnext: longctx-profile ==` header is followed by `nothing runnable`,
which is what a *second* claim prints once the first has marked the job running. qnext's own lock
does not help — the first process was already past it and the kill came from outside.

**Two fixes, both in.**

1. **Launch the driver with `setsid`, not inside a named tmux session.** A driver that is not a
   child of any session cannot be taken down by session bookkeeping:
   ```
   setsid env QNEXT_QUEUE=... PYTHONUNBUFFERED=1 python3 tools/qnext.py \
       >> ~/ds41-queue/logs/qnext-drive.log 2>&1 < /dev/null &
   ```
2. **`PYTHONUNBUFFERED=1` on every job** (`qnext.py` now sets it in the subprocess env). The log
   was empty because `longctx_profile.py`'s header `print()` had no `flush=True`, so a job that had
   been running for ten minutes was **byte-for-byte indistinguishable from one that died on its
   first instruction**. That ambiguity is what made this take a diagnosis instead of a glance.
   `longctx_profile.py` also now prints `... requesting` before each arm, so a stalled arm is
   visible rather than inferred.

**The general rule this earns:** a long job's liveness must be readable from its log alone. If the
only way to tell a running job from a dead one is `pgrep`, the harness is under-instrumented — and
`pgrep` is exactly what is not available when reading the log an hour later.
