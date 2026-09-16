# The CB2 deficit is the bit count, not the calibration (2026-09-16)

Our CB2 is not a general 2-bit quantizer. `engine/codebook_sim.py` picks, per matrix ROW, the best
4 of the 16 fixed E2M1 grid levels, scored on pure weight error:

    hist[level]   = sum over weights of scale_g^2 * [code == level]
    cost(subset)  = sum_level hist[level] * min_{m in subset} (v_level - v_m)^2

Two obvious weaknesses -- it never sees an activation, and the levels are locked to a geometric
grid. Both were argued to explain the -2.16 pp result. **Both were measured and neither does.**

## Measurement

Real FP4 expert weights, fetched by byte range from the 428 GB checkpoint on the backup box (the
lean checkpoint has no routed-expert safetensors, so starting from the CB3 arena would have been a
requantization of a requantization). Real FFN inputs captured by hooking `moe_fn`, whose first
argument IS the vector every routed expert's w1/w3 sees. Metric is output error,
`||(W - Wq) x|| / ||W x||`, on those activations -- 3 layers, 2 experts, w1 and w3.

| variant | mean output error | closes of the CB2 -> CB3 gap |
|---|---|---|
| CB3 (3 bits, grid) | 0.18788 | — |
| CB2 (2 bits, grid, weight error) | 0.32759 | — |
| CB2 + activation weighting | 0.32660 | **0.7 %** |
| CB2 + FREE levels (Lloyd-Max, 4 arbitrary values/row) | 0.31834 | **6.6 %** |

Activation weighting is one line (`wgt *= E[x_j^2]`) and keeps format, packing and kernel. Free
levels keep 2 bits/weight and the per-row codebook but drop the grid constraint -- not
representable by today's packing, which rebuilds an FP4 nibble from the index, so it was simulated
precisely to bound what a format change could buy.

## Why both fail

The codebook is chosen per ROW of W (output dim) and the activation weighting is over COLUMNS
(input dim). One row's 4 levels must serve all 5,120 columns, so reweighting which columns matter
barely moves the histogram. That is the 0.7 %.

And free levels only buy 6.6 % because the remaining gap is the BIT COUNT. Lloyd-Max on a Gaussian
gives 9.30 dB at 2 bits against 14.62 dB at 3, a factor 1.85 in RMS error; measured here,
free-level CB2 / CB3 = 0.31834 / 0.18788 = **1.69**. Free-level CB2 is already at its scalar bound.

## What this closes, and what it points at

CLOSED: improving CB2 by calibration or by codebook placement. There is ~7 % available inside
scalar-per-row quantization and the deficit is ~74 %.

POINTS AT: the only way to 2-bit quality is a representation that escapes the scalar bound --
vector or trellis coding over groups of weights (QTIP / EXL3 mul1 K=2), which is exactly where the
external evidence pointed. Now we have our own reason: not "someone else's 2-bit is better" but
"nothing left to win within scalar."

COROLLARY, untested: the ratio 1.69 is BELOW the 1.85 the scalar bound predicts, which means CB3 is
further from ITS bound than free-level CB2 is from its own -- CB3 is grid-constrained too. Freeing
CB3's levels would be a quality gain at unchanged size, and unlike the CB2 case the format already
stores 8 codebook bytes per row.

## CORRECTED 2026-09-16 after review: the free-level arm was broken twice

Everything below this heading from the first two sections' free-level numbers onward was wrong, in
the same direction each time -- understating what a better codebook buys. Two defects, both found
by review:

1. **`requant_free` removed the group scales.** It fitted centroids to fully SCALED weights and
   returned them directly, so that arm had one codebook per row covering all 160 groups and no
   per-32 scale at all. It was not a relaxation of CB2, it was a different and weaker format.
2. **Lloyd-Max with quantile init was not solving the problem.** At 4 levels it scored WORSE than
   the exhaustive 4-of-16 grid search it was meant to beat (0.338 against 0.331) -- a solver
   artifact reported as "free levels do not help". With only 16 distinct source levels the weighted
   k-centroid problem is a 1-D partition and dynamic programming solves it exactly in O(16^2 k).

Corrected -- group scales kept, exact DP, same weights and activations:

| levels | bits | output error | vs grid CB3 |
|---|---|---|---|
| 4 | 2.00 | 0.32095 | +68.9 % |
| 6 | 2.58 | 0.21636 | +13.9 % |
| 8 | 3.00 | **0.15601** | **-17.9 %** |

**CB3 IS grid-constrained, by 17.9 %.** My original corollary was right and the broken
implementation is what appeared to refute it. Freeing CB3's levels lowers output error by 17.9 % at
the SAME bit width -- and plausibly at the same size, because the format already carries 8 codebook
bytes per row, which is exactly 8 fp8 values. Only the kernel's decode changes: an fp8 table lookup
instead of nibble-to-FP4. That is now the cheapest quality lever on the page and it was hidden
behind two bugs.

The 2.58-bit point is +13.9 % against CB3, not the +23.2 % reported.

WHAT DOES NOT SURVIVE: the Gaussian rate-distortion argument built on those numbers. It assumed a
continuous source and a scalar bound, and was used to claim a 2-bit trellis "must" stay ~23 % worse.
That was a hypothesis dressed as a bound, and with the corrected curve it does not even reproduce
its own anchor. Withdrawn.

WHAT STILL STANDS: activation weighting is worth 0.4 % -- it was never the free-level arm that
carried that result, and the structural reason holds (the codebook is per ROW, the weighting per
COLUMN). But "calibration is closed" overstates a diagonal-weighting test on w1/w3 only; w2 and a
non-diagonal objective are untested.

## The free-level width curve closes the corollary too (job 330, corrected)

The corollary above predicted CB3 was grid-constrained and that freeing its levels would be quality
at unchanged size. **Measured, and wrong.** Free-level Lloyd-Max at several codebook sizes, same
real FP4 weights, same real activations, 5 layers x 3 experts:

| levels | bits | output error | vs grid CB3 |
|---|---|---|---|
| 4 | 2.00 | 0.32244 | **+69.7 %** |
| 6 | 2.58 | 0.23418 | **+23.2 %** |
| 8 | 3.00 | 0.18885 | **-0.6 %** |
| grid CB3 (8 of 16) | 3.00 | 0.19003 | — |

Free levels at 3 bits beat the grid by **0.6 %**. So CB3's grid constraint costs essentially
nothing, and freeing it is not a lever. At 4 levels the same constraint costs 6 % -- the grid only
binds when the codebook is small relative to it.

And there is no width between 2 and 3 bits that matches CB3: 2.58 free bits is still +23 % worse.

## Why this checkpoint is not QTIP's setting

**The source is already 4-bit.** These experts are stored as E2M1 codes -- 16 distinct values per
row, times a UE8M0 scale per 32. So "CB3" is not 3-bit quantization of a continuous weight, it is
choosing 8 of 16 available levels, and "CB2" is choosing 4 of 16. That is why free-vs-grid vanishes
at 8 levels: with a 16-level source, 8 free levels and 8 grid levels have almost the same reach.

This matters for the trellis/QTIP direction, and the continuous-source arithmetic sets the ceiling:

* Gaussian rate-distortion gives 6.02 dB per bit, so R=2 is 12.04 dB; scalar Lloyd-Max at 2 bits
  reaches 9.30 dB and at 3 bits 14.62 dB.
* So an OPTIMAL 2-bit vector quantizer -- the rate-distortion bound, which no implementation
  reaches -- is still **2.58 dB short of 3-bit scalar**. Trellis coding cannot make 2 bits match 3.
* Converting that bound to this measurement: the full VQ gain at 2 bits would take 0.32244 down to
  about 0.235, which is where the 2.58-bit free-scalar point already sits (0.23418) -- **+23 %
  against CB3, not parity**.

The measured curve and the information-theoretic bound agree, which is the useful part: K=2 trellis
should be expected to land near "2.6 effective bits", not near CB3.

So the honest framing for a QTIP/EXL3 project here is a TRADE, not a free win: roughly +23 % output
error for a 31 % smaller record (9.99 against 14.45 MB) and ~45 % more resident experts. Given
capacity is the largest measured lever on this box, that trade may well be worth taking -- but it
must be gated on paired NLL and a long free-generation check, not on the size arithmetic.

Two further cautions specific to us, neither of which QTIP's published results address:
* Their numbers quantize BF16 originals. Re-quantizing an FP4-native checkpoint has strictly less
  headroom, and the 8-level result above is direct evidence that the 16-level source is already
  binding.
* Incoherence processing (random rotations) is where much of QTIP's gain comes from, and it
  reshapes the source distribution. On a source that is already a 16-point grid times block scales,
  how much of that gain survives is unknown and is the first thing to measure.
