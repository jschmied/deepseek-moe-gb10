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
