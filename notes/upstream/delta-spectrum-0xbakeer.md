DRAFT — needs the user's go. 0xBakeer/deepseek-v41-flash-spark, comment on issue #3 (2026-09-12).

**Shared-base + low-rank delta expert compression does not pay on an already-4-bit checkpoint.**
Measuring it before building it, in case anyone here is tempted by the D²-MoE / MoE-SVD headlines.

SVD spectrum of `W_expert − mean(W)`, 16 experts per arm, real weights — DS V4.1 layer-0 `w1` from
the checksum-verified shard, and Qwen Flash-Next L24 `down_proj` from an NVFP4 checkpoint for shape
contrast. Break-even is the rank at which a shared-V bf16 factorisation stops saving bytes against
the 4-bit weights already being paid for.

| | max rank | delta @99 % | W itself | random floor | break-even |
|---|---|---|---|---|---|
| DS V4.1 L0 `w1` (2304×5120) | 2304 | **2121** (92 %) | 2119 | 2158 | 1280 |
| Qwen FN L24 `down_proj` (2560×640) | 640 | **616** (96 %) | 615 | 618 | 640 |

The deltas sit within **1–2 ranks of the expert matrices themselves** and within **2–37 ranks of pure
noise**. At 99 % energy they need 92–96 % of full rank.

The Qwen row is the one worth dwelling on: 616 < 640 reads as a saving until you see that a **random
matrix of the same shape scores 618** — also under break-even. On experts that rectangular (4:1),
shared-V factorisation "saves bytes" for random data, so clearing break-even measures the shape, not
the weights. Our first verdict line printed USABLE for that row; the random control is what caught it.

**Not established:** one box, layer-0 and one Qwen layer only, 16 experts per arm, bf16 SVD, and a
break-even derived for shared-V bf16 factors specifically — a different factorisation has a different
break-even. This says the *premise* (a low-rank per-expert delta) does not hold on these weights; it
does not evaluate any published implementation's other machinery.
