POSTED 2026-09-12 as issue #3 (user go "remove them and replace with one merged one").
Target: 0xBakeer/deepseek-v41-flash-spark, new issue; then close #1 and #2 pointing here.
(Deletion is not available to us — viewerPermission READ, no admin/maintain.)
---
Consolidating my #1 and #2 into one issue and closing both. Two reasons: they were written against
an operating point v0.4.0 has moved past, and #1 quotes two numbers that have since changed. Rather
than leave stale arithmetic standing in your tracker, here is the corrected version plus the things
I think are worth more than the storage analysis was.

Different box, same class: single GB10 / DGX Spark, sm_121. I run Qwen3.8-Flash-Next NVFP4 under
vLLM, not V4.1 — so everything below is either a measurement on *my* model offered as a transferable
failure mode, or a method, never a claim about yours.

## 1. Your new free-generation gate cannot catch the failure it replaced

The `<!DOCTYPE>` case and its root cause — a 50-doc keep-set corpus with no HTML/JS/CSS/SQL, so
markup experts never fired, ranked cold, got pruned — is the most useful thing in the repo. But the
gate you replaced the +0.015-nats threshold with tests for **degeneration**: distinct-token ratio,
line-repeat, structural checks.

Expert pruning also produces **fluent, wrong** output, which that gate passes.

Measured on my model (stock NVFP4, unmodified), asked to copy a Hindi sentence back verbatim:

```
want  उपयोगकर्ता को पहले लॉग इन करना होगा।
got   उपयोगकर्ताने पहिले लॉग इन करणे आवश्यक आहे.
```

The second is **Marathi** — a different language. 7 words, distinct-token ratio **1.0**, zero
repeated lines, structurally clean. Every heuristic in your gate passes it.

**The prediction, and it is cheap to check.** If markup experts were pruned for never firing in a
50-doc corpus, non-Latin-script experts are the next casualty of the same mechanism, and this gate
would not see it. An exact-copy probe over a few scripts (Devanagari, Thai, Arabic, Hebrew, plus ZWJ
emoji) scores one thing — byte-exact copy or not — and catches it in one run. In my own work a
build with two broken export contracts hit 12/12 Devanagari corruption where the reference was
clean, so the readout does discriminate. Cost to you: a dozen prompts, one generation each.

One trap if you build it: score **exact copy**, not character-level diffs. My first metric counted
"duplicated combining marks", which conflates substitution, duplication and translation — it produced
two findings I had to retract before I looked at an actual output string.

## 2. `spec-on greedy == spec-off greedy` may not be passable as written

`test_spec_lossless.py` is the right gate and the reason is right — teacher-forced NLL never
exercises the decode loop, which is exactly why it missed your BF16 router. But on an MoE at
`temperature: 0`, greedy is not reproducible *at all* without work. Same ~5,700-token prompt, five
runs, nothing to do with speculation:

| max_tokens | distinct outputs of 5 | first divergence |
|---|---|---|
| 32 | 3 | char 52 |
| 512 | **5** | **char 9** |
| 2,000 | **5** | char 9 |

So the gate can fail for reasons that are not the spec path, and it would look like the spec path.
What it took me to get bit-stability: a deterministic top-k (the stock radix select is not
tie-stable), a bit-stable MoE finalize (`use_fused_finalize=False`), an align-block fix, and a
semaphore reset in the offload path. If your gate fails, I would check reproducibility with
speculation **off** first, before believing it says anything about speculation.

## 3. The unexplained few-percent `Model.forward` gap — two things that worked for me

`LIMITATIONS.md` still concedes this post-fix. I chased the same shape for a day:

- **Read the executed kernel roster from logs, not from dispatch code.** I made three wrong claims in
  one morning reasoning about selection logic that the run did not take.
- **Check whether compile/autotune cache state alone moves the output.** For me, the *identical* venv,
  patch state and flags with only a different cache root produced a **different answer** — one arm a
  matra substitution, the other a whole-sentence paraphrase. If your Triton kernels autotune, cold
  versus warm cache is a candidate for a gap that "isn't explained".

## 4. On N=1

Every number in `RESULTS.md` is single-run. I measured up to **1.83× spread on speculative-decode
timings across restarts with no configuration change whatsoever**. Three starts before quoting a rate
is not pedantry on this hardware — it is the difference between 36.6 vs 17.1 tok/s being a workload
effect or partly restart noise.

## 5. The storage findings from #1 and #2 — still true, now narrower

v0.4.0 reports hit rate 1.0 and **no NVMe during decode**, so these govern the 510 GB load and the
streaming/unpruned path, not the config you now ship. Stating that myself:

- **Queue depth dominates read size.** 64 KiB goes 0.49 → 3.90 GB/s from QD1 to QD128 (**8.0×**);
  256 KiB reaches 5.88 GB/s, 92 % of what 18 MiB reads get. My original "read size dominates,
  io_uring cannot help" was an artefact of stopping the sweep at QD4.
- **NVFP4 is not losslessly compressible** — withdrawn lever from #1, measured over 73.7 M nibbles:
  order-0 entropy **3.969 of 4.000**, combined floor **2.8 %**. Not worth a decoder.
- **#1's cost model, corrected.** It used 168 ms verify; you now report 114.9 ms at top-6. So compute
  is 43.3 ms/token (23.1 tok/s ceiling), not 61.0 (16.4), and fetch:compute is **3.8 : 1**, not
  2.7 : 1. The conclusion strengthens — fetch dominates by more than I claimed — but the baseline it
  reproduced was measured before your router fix, so treat the fit as coincidental.

Happy to run any of the above here if it is useful; I have the box and no stake in the answer.
