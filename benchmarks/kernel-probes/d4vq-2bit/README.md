# d=4 vector quantization at 2 bits/weight vs MQ2-Lloyd

Tests whether a 4-dimensional VQ can beat MQ2-Lloyd at **identical bytes**.

## Why d=4

MQ2-Lloyd is a *scalar* quantizer, and its fit is already within 0.045 dB of
the optimal scalar fit — so its remaining error is structural, not a fitting
problem. Scalar quantization of a Gaussian sits ~2.7 dB off the Shannon bound
at 2 bits; most of that is space-filling loss that only higher dimensions
recover.

At 2 bits/dim, d=4 needs `2^(2*4) = 256` codewords → exactly **1 byte per 4
weights** → 64 B payload per 256-weight group, the same as MQ2's. The codebook
is a universal Gaussian constant (zero wire cost), so the full 8 B header is
free for `[f32 master][4 x E4M3 g64 sub-scale]`.

**72 B/group, 2.25 bpw — byte-identical to MQ2-Lloyd.**

## Quality (real weights, FWHT-rotated)

| tensor | MQ2-Lloyd | d4-VQ | gain |
|---|---|---|---|
| LFM2.5-350M in_proj | 9.51 dB | 10.21 | +0.70 |
| LFM2.5-350M out_proj | 9.48 dB | 10.18 | +0.69 |
| Qwen3.5-0.8B in_proj_qkv (held out) | 9.57 dB | 10.23 | +0.66 |
| Qwen3.5-0.8B in_proj_z (held out) | 9.47 dB | 10.21 | +0.73 |

The codebook was fitted on **pure synthetic Gaussian noise, no model data**.
A codebook fitted on actual LFM2.5 weights scores *worse* (10.07-10.10) — same
CLT argument as the 1-D case: post-FWHT everything is the same Gaussian, so
fitting per-model only adds noise.

## Decode cost (gfx1201, weighted over the Qwen3.5-0.8B per-token GEMV stack)

| codebook placement | LDS | excl lm_head | incl lm_head |
|---|---|---|---|
| f32 in LDS, filled per block | 4 KB | **+108% .. +112%** | +121% |
| fp16 in LDS, filled per block | 2 KB | +20% .. +24% | +16% |
| **`__constant__`, no LDS, no fill** | 0 | **+0.9% .. +1.0%** | **−1.0%** |

## The finding: lookups per weight decides where the codebook lives

This **inverts** the result from the MQ4N probe next door, where indexing
`__constant__` with a divergent index was catastrophic (+23% mean, +96% worst)
and LDS was free.

The difference is lookups per weight. MQ4N does one lookup *per weight*, so
gather cost is paid 8x per thread per group and a 16-float LDS fill is
negligible. d=4 VQ does one lookup per **four** weights — 4x fewer gathers —
while its LDS table is 64x larger, so the per-block fill (1024 floats) becomes
the dominant cost and the gather becomes affordable.

Rule of thumb: small table + frequent lookups → LDS. Large table + amortized
lookups → `__constant__`.

## Verdict

+0.70 dB at identical bytes, at roughly decode parity — but **borderline**
against a 1% budget. Two shapes regress (`fused_qkv` +5.7%, `fused_gate_up`
+4.3%), offset by `down_proj` (−5.3%) and `lm_head` (−6.2%). Both regressing
shapes are K=1024, i.e. only 4 groups per row, so there is little work to hide
the gather latency behind. Processing multiple rows per block would likely fix
them; not attempted here.

Quality numbers are weight-MSE. Nothing here is a coherence claim.
