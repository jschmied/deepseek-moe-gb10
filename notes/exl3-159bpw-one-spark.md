# The EXL3 1.59 bpw one-Spark recipe: 2-3x our throughput, and what it costs (2026-09-16)

Sources: `github.com/vcruz305/DeepSeek-V4.1-Flash-EXL3-DGX-Spark-recipe/tree/main/one-spark-tp1`
and `huggingface.co/vcruz305/DSV4.1-Flash-SAGE-EXL3-1.59bpw`. Everything below is their published
claim unless marked otherwise.

## The result

| | tok/s, one DGX Spark |
|---|---|
| decode, no drafter, aliased | 13.96 - 14.19 |
| decode, no drafter, copied to CUDA | 15.13 - 15.22 |
| **decode + DSpark drafter** | **17.53 median, 19.82 mean** |
| repeat prompt | 20.11 - 24.67 |

Acceptance 0.889, draft block 5. **Our production is 6.26 tok/s**, so this is 2.2-3.2x.

## The mechanism is not a better scheduler. It is not streaming the experts.

All 384 routed experts are RESIDENT -- 108.86 GB of experts, aliased out of `mmap` through GB10's
ATS addressing mode rather than copied. No expert cache, no eviction policy, no NVMe expert path.
Everything this repo has spent the day on -- read depth, capacity misses, prefetch, eviction --
exists because we stream at ~3 bits. They removed the problem by going to 1.59.

The bit allocation is real and verifiable: the trellis tensor shapes give a weighted mean of
**1.5920 bits/weight**, reproducing their claim. It is sensitivity-shaped, not uniform -- **59 % of
expert tensors are at K1, one bit per weight** -- with w2 favoured (mean K 1.72) over w1/w3 (1.53 /
1.52), and per-layer means from 1.95 at layer 19 down to 1.31 at layers 38-39.

Everything else is left alone: attention and shared experts FP8, head and embedding BF16, Engram
FP8 byte-identical passthrough.

## They did not remove streaming. They moved it.

The artifact is **330.39 GB**, of which **Engram is 203 GB** -- and the card says plainly that
*"the Engram tables are read from disk"*. The recipe's "~107 GiB resident" is the pack MINUS
Engram, which its memory arithmetic never mentions.

That inverts our design rather than beating it: we keep experts streamed (13.77 MB records, few
and huge) and Engram on disk at ~0.003 % of decode bytes; they keep experts resident and stream
Engram (264 B rows, many and tiny). Two very different I/O profiles, and the tiny-row one is far
friendlier to a device that saturates at ~2 concurrent reads.

## The quality cost, which the card publishes and the recipe does not

Teacher-forced against the **native FP4 release checkpoint** (not BF16), full-vocabulary KL per
token, 32 held-out sequences x 4096 tokens:

| pack | expert bytes | KL vs FP4 | top-1 | top-5 |
|---|---|---|---|---|
| 1.59 bpw | 108.9 GB | 0.1885 | **91.2 %** | 77.5 % |
| 3.30 bpw sibling | 224.9 GB | 0.0910 | 94.0 % | 83.0 % |
| FP4 reference | native | 0 | 100 % | 100 % |

So **8.8 % of top-1 tokens flip** at 1.59 bpw, and 6.0 % even at 3.30. By subset, general text is
worst (top-1 82.0 %, KL 0.409) and code best (97.9 %, KL 0.054). KL is flat across the context
window, so the loss does not compound with position.

For scale: our own fused-attention work rejected a change that flipped **0.95 %** of top-1 tokens.
These are an order of magnitude beyond that.

## What is absent from both documents

Calibration corpus and the SAGE allocator method (explicitly withheld), any PPL/NLL, any task
benchmark, any BF16 reference, quantizer version pin, TTFT, token-counting method, prompt/output
lengths, run counts, concurrency above batch 1. The benchmarked artifact is a locally re-laid
64-byte-aligned build that is not published, and the driver script is not in the repo, so the
throughput numbers cannot be reproduced from the repo alone.

Two internal contradictions: the recipe claims a "~14 GiB drafter" where the artifact's `mtp.*`
tensors total 7.39 GiB, and its memory story omits the 189 GiB of Engram the card says is streamed.

## What this means for us

1. **Residency beats scheduling, if the quality holds.** 2-3x is larger than every scheduling lever
   we have measured combined (oracle prefetch +20 % warm, +36 % cold; policy null; overlap ceiling
   +80 % of a step we cannot fully overlap).
2. **The quality question is the whole question**, and it is one we are equipped to answer: we have
   `paired_nll.py` and the FP4 reference on the box. Their 8.8 % top-1 flip is a published number
   we can reproduce our own version of, on our own corpus, against our own CB3.
3. **Our CB3 sits at 3 bits with no published top-1 flip rate against FP4.** Measuring that is the
   missing comparison: if CB3 flips ~1 % where 1.59 bpw flips 8.8 %, the trade is quantified and
   the decision becomes a product judgement rather than a guess.
4. The sensitivity-shaped allocation (K1-K6, w2 favoured, layer 19 protected) is a strong hint that
   our uniform CB3 leaves bits on the table -- consistent with today's finding that freeing CB3's
   codebook is worth 17.9 % of output error at unchanged size.
