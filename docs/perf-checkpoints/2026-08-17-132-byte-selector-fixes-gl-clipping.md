# 132 bytes of 4-byte-aligned layout buys the fix GL needed: max-normalisation + a per-block codebook selector

- **Date:** 2026-08-17
- **Lifecycle:** `historical`
- **Fixture:** 69,632 real post-FWHT 256-blocks — `layers.20.mlp.down_proj` of the
  `Qwen3.8-27B` bf16 parent, engine FWHT (sign seeds 42 / 1042), per-block
  normalisation as `gl_encode_block` applies it
- **Disposition:** design study. Selects the qt=43 format; supersedes and
  **cancels** the GL_CB4R span-widening experiment. Codec MSE reported, never
  ranked on. No KLD claim until the artifact is scored on hiptrx.

## Where this starts

MQ4-GL (qt=40) loses KLD by 11.28% (WT2) and 27.52% (v6 selector) while winning
codec MSE by 22.69%. The prior checkpoint localised the cause: GL's outermost
level of 2.7326 clips the dominant coefficient of **82.57% of blocks**, which is
only 0.606% of weights — invisible to MSE, decisive for logits.

Two candidate fixes existed. One was measured wrong; the other was suggested by
the maintainer and is right.

## The obvious fix is wrong: widening the span makes everything worse

`GL_CB4R` pinned the outermost level at 4.0 with the inner 7 magnitudes
constrained-Lloyd-refitted, which on paper cuts clipped blocks from 82.57% to
1.29%. Measured, it is worse on **all three** proxies:

| variant | B | overall MSE | tail-1% MSE | max-coef relerr |
|---|---|---|---|---|
| GL_CB4 shipped | 130 | 1.1627e-06 | 1.3921e-05 | 10.859% |
| GL_CB4R span 4.0 | 130 | 1.399e-06 | 1.969e-05 | **14.921%** |

It made the very quantity it targeted worse. Pinning level 15 at 4.0 without
adding levels opens a coarse gap across 2.7–4.0, and that gap is exactly where
block maxima live (max\|z\| p50 = 2.999). Range without resolution is not a fix.
**The experiment was cancelled before it consumed a scoring slot.**

## What affine is actually doing right

| format | B | bpw | overall MSE | tail-1% MSE | max-coef relerr |
|---|---|---|---|---|---|
| affine (qt=1) | 136 | 4.2500 | 1.4642e-06 | **7.8961e-07** | **0.000%** |
| GL_CB4 (qt=40) | 130 | 4.0625 | **1.1627e-06** | 1.3921e-05 | 10.859% |

Affine's max-coefficient error is **exactly zero** — min/max fitting makes the
block extreme representable *by construction* — and its tail-1% MSE is **17.6×**
better, while it loses overall MSE by 20.7%. That is the trade in one line: GL
serves 99.4% of weights better and the decisive 0.6% far worse.

## Fix 1 — max-normalise. Zero bytes.

Redefine what the existing 2-byte fp16 scale field *means*: store the block's
`max|coefficient|` instead of its RMS. Codebook level 15 becomes `+1.0` and lands
exactly on the block maximum.

| variant | B | bpw | overall MSE | tail-1% MSE | max-coef relerr |
|---|---|---|---|---|---|
| GL_CB4 rms (shipped) | 130 | 4.0625 | 1.1627e-06 | 1.3921e-05 | 10.859% |
| **max-norm + refit cb** | **130** | **4.0625** | 1.1975e-06 | **6.3938e-06** | **0.000%** |

Clipping eliminated structurally, tail **2.18× better**, for 3.0% of overall MSE
and **not one additional byte**.

## Fix 2 — spend 2 spare bytes on a per-block profile selector

132 B = 128 B indices + 2 B fp16 scale + 1 B selector + 1 B pad → 4.125 bpw,
still **2.9% under affine's 136 B**, and **4-byte aligned**. Alignment is not a
cost here: the previously measured AoS-B 132 B layout ran **505.1 GB/s vs the
130 B SoA layout's 492.0** (bandwidth ratio 0.866 vs 0.843).

The selector picks one of 16 level profiles, so each block chooses its own point
on the bulk-resolution ↔ tail-fidelity tradeoff — the axis that separates Lloyd
(dense near zero, sparse at the extremes) from uniform (even everywhere).

| candidate | B | bpw | overall MSE | tail-1% MSE | max-coef relerr |
|---|---|---|---|---|---|
| affine (qt=1) | 136 | 4.2500 | 1.4642e-06 | 7.8961e-07 | 0.000% |
| GL_CB4 (qt=40) | 130 | 4.0625 | 1.1627e-06 | 1.3921e-05 | 10.859% |
| max-norm only | 130 | 4.0625 | 1.1975e-06 | 6.3938e-06 | 0.000% |
| + 4b selector, linear family | 132 | 4.1250 | 1.1132e-06 | **5.0063e-06** | 0.000% |
| **+ 4b selector, designed family** | **132** | **4.1250** | **1.0807e-06** | 5.1552e-06 | **0.000%** |

The designed family (generalized Lloyd over the *family* — alternating
block→profile assignment with per-profile constrained refit, outermost pinned at
±1.0) is **7.1% better than shipped GL on overall MSE, 2.70× better on the tail,
26.2% better than affine on overall MSE, and exactly non-clipping**. All 16
profiles are live; max share 14.8%; the uniform end takes 0.00%, so the family
spans the useful range with none of it wasted.

Both families are retained. The designed one wins overall MSE; the linear one
wins the tail (5.0063e-06 vs 5.1552e-06). Since KLD is tail-dominated, the
MSE-optimal family member is not automatically the KLD-optimal one, so the
selector between them is an env override rather than a build-time choice.

Constants: [`2026-08-17-gl-cb4s-family-constants.rs.txt`](2026-08-17-gl-cb4s-family-constants.rs.txt).

## Standing warning

Every number above is a **proxy**. On this exact format a 22.69% codec-MSE win
became an 11.28% KLD loss, so none of this is a quality claim. What is different
this time is that the proxy being optimised — max-coefficient error and tail MSE
— is the one measured to track the observed KLD failure, rather than the one
measured not to. The claim stands or falls on hiptrx:

| arm | B/group | bpw | WT2 KLD | v6sel KLD |
|---|---|---|---|---|
| mq4 uniform affine | 136 | 4.2500 | 0.043776 | 0.587566 |
| mq4gl GL_CB4 | 130 | 4.0625 | 0.048713 | 0.749238 |
| mq4sel qt=43 | 132 | 4.1250 | pending | pending |

If qt=43 does not reach affine's KLD, the tensor-global-codebook family has been
given its best shot and should be retired rather than tuned further.

---

## Amendment — the pad byte was worth spending: 64 profiles, not 16

The 132 B layout was specified as 128 B indices + 2 B scale + 1 B selector +
**1 B pad**. Those pad bits are free, so the selector width was swept on all
69,632 blocks:

| selector bits | profiles | overall MSE | tail-1% MSE | live / K | max share |
|---|---|---|---|---|---|
| 0 (max-norm only) | 1 | 1.1800e-06 | 5.3238e-06 | 1 / 1 | 100.00% |
| 2 | 4 | 1.1217e-06 | 5.3911e-06 | 4 / 4 | 41.22% |
| 4 | 16 | 1.0854e-06 | 5.1746e-06 | 16 / 16 | 16.63% |
| **6** | **64** | **1.0501e-06** | **4.8093e-06** | **62 / 64** | 7.01% |

6 bits buys **3.3% overall MSE and 8.0% tail** over 4 bits, and the tail is the
KLD-relevant axis. 62 of 64 profiles stay live at a 7.01% max share, so 64 is
near saturation and 256 is not worth pursuing. Same bytes, same alignment,
strictly better — the format is 6-bit-selector.

Final measured position:

| format | B | bpw | overall MSE | tail-1% MSE | max-coef relerr |
|---|---|---|---|---|---|
| affine qt=1 | 136 | 4.2500 | 1.4642e-06 | 7.8961e-07 | 0.000% |
| GL_CB4 qt=40 | 130 | 4.0625 | 1.1627e-06 | 1.3921e-05 | 10.859% |
| **SEL qt=43** | **132** | **4.1250** | **1.0501e-06** | **4.8093e-06** | **0.000%** |

9.7% better overall MSE than shipped GL, **2.89× better tail**, structurally
non-clipping, and 28.3% better overall MSE than affine at 2.9% fewer bytes.

Constants: [`2026-08-17-gl-cb4s64-family-constants.rs.txt`](2026-08-17-gl-cb4s64-family-constants.rs.txt).

### The 64-entry table needs no LDS

The initial kernel spec called for LDS-staging the codebook. That is wrong for
this access pattern, and at 64 profiles (4 KiB f32) it would have been actively
harmful: `__launch_bounds__(32, 16)` puts 16 waves on a CU, and 16 × 4 KiB
exactly saturates the 64 KiB LDS budget, costing occupancy.

It is unnecessary because **the selector is per-group and therefore wave-uniform**
— a 32-thread workgroup processes one 256-weight group, so every lane shares one
profile. The 16 live levels load into SGPRs via scalar loads (8 × `s_load_b64`
from `sel * 64` bytes): no LDS allocation, no per-lane table traffic, no barrier,
occupancy untouched. Pressure moves to SGPRs, so the kernel report must include
the SGPR count alongside VGPRs and the zero-spill check.
