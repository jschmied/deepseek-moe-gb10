# Our two issues on 0xBakeer/deepseek-v41-flash-spark — status 2026-09-12

Both filed 2026-09-11, **both still open, no owner reply on either**, zero PRs on the repo.

| | |
|---|---|
| **#1** | "Unpruned path: ~35 % is available from fetch/compute overlap, plus three quality-exact levers" — a cost model plus four levers. Our own comment (09-11 09:48) **withdrew lever 3**: NVFP4 is not losslessly compressible; order-0 nibble entropy 3.969/4.000 over 73.7 M nibbles, combined floor **2.8 %**. |
| **#2** | "Corrected: queue depth dominates…" — **revision 2**, itself withdrawing a third claim in the opposite direction. QD1→QD128 is **8.0×** at 64 KiB (0.49 → 3.90 GB/s); 256 KiB reaches 5.88 GB/s, 92 % of what 18 MiB reads get. Plus a PS on the Samsung PM9E1 Gen5 as a drop-in for the Gen4 Phison. |

## Both have been partly overtaken, and #1 has two stale inputs

**The v0.4.0 config does not touch NVMe during decode.** `RESULTS.md` §4.3: keep 0.44, arena 98 GB,
44.1 % resident, **hit rate 1.0, no NVMe during decode**. Our two issues are both about the storage
path:

- **#2 is now a load-time and streaming-mode result, not a decode result.** The 8× queue-depth
  finding stands as measured and still governs the 510 GB load and the unpruned/streaming path — but
  it no longer touches the configuration they now ship. Worth saying so ourselves rather than letting
  it read as advice about the current default.
- **#1 is explicitly about the *unpruned* path**, which is not where the repo now operates. Still
  valid for that path; less relevant to the default.

**And #1 quotes two numbers that have since moved:**

| input | as filed | current repo | effect |
|---|---|---|---|
| verify time | 168 ms | **114.9 ms** (top-6) | compute 61.0 → **43.3 ms/token** |
| the measured baseline it reproduces | "15.2–15.7 resident", "3.5–4.0 unpruned" | pre-FP32-router | those runs routed a materially different model |

Re-running our own arithmetic with the current verify time:

| | as filed | corrected |
|---|---|---|
| compute ceiling | 16.4 tok/s | **23.1 tok/s** |
| serialised | 4.5 tok/s | 4.8 tok/s |
| overlapped | 6.1 tok/s | 6.1 tok/s |
| **fetch : compute** | 2.7 : 1 | **3.8 : 1** |

**The conclusion survives and in fact strengthens** — fetch dominates compute by a wider margin than
we claimed, so the overlap argument is stronger, not weaker. But we should not leave two stale inputs
standing in a public issue when we know they moved, particularly the baseline, which was measured
before the router was fixed.

## What that implies for a third issue

Do not file one framed against the old operating point. If we contribute again it should be against
**v0.4.0 as shipped** — where storage is out of the decode loop and the open questions are quality
gates and correctness, not bytes. Our four candidate ideas (see below) are all in that space, which
is the right space now.

Also worth weighing: two issues, no reply, no PRs on the repo at all. A third issue may not land
either. A **pull request** — or simply running their own `test_spec_lossless.py` and reporting
whether it passes — may carry further than more analysis.

---

## Resolved 2026-09-12: merged into issue #3

**Deletion was not available** — our `viewerPermission` on that repo is `READ`, and GitHub requires
admin/maintain to delete an issue. As author we can only close. So: opened the merged issue first,
then closed #1 and #2 with a comment pointing at it.

- **#3 OPEN** — "Consolidating #1 and #2: a quality-gate blind spot, the spec-lossless gate, and
  corrected storage numbers" → https://github.com/0xBakeer/deepseek-v41-flash-spark/issues/3
- **#1 CLOSED** (not planned) → comment 5645143760
- **#2 CLOSED** (not planned) → comment 5645144000

Repo state re-read immediately before posting: HEAD `8b68fdde1`, pushed 2026-09-12 08:36
("Prefill: device slot table for the chunked path"). Still no owner reply on anything.

**What #3 leads with**, deliberately changed from the storage framing of #1/#2: the free-generation
gate cannot catch fluent-wrong output, with our Marathi-for-Hindi case as the evidence (7 words,
distinct-token ratio 1.0, zero repeated lines — passes every heuristic in it), and the prediction
that non-Latin-script experts are the next casualty of the same corpus-coverage mechanism that
pruned the markup experts. Then the temp-0 warning about `test_spec_lossless.py`, the two methods for
the `Model.forward` gap, the 1.83× restart spread against their N=1, and finally the storage results
scoped down to load-time/streaming with #1's cost model corrected in the open.

Everything attributed to our model (Qwen3.8-Flash-Next NVFP4 under vLLM), never asserted about theirs.
