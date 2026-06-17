# RoughQuant — Phase 2e: energy analysis + correction of the PCA "win"

**VERDICT (corrected, final for 0.8B): no RoughQuant variant beats mq4 — foldable
OR not.** Two user catches drove this correction: (1) a non-monotonic bug in the
protected quantizers inflated the earlier "PCA beats mq4" result; (2) PPL is too
noisy (pointwise) to resolve small effects — KLD/energy needed. The root cause is
now quantified: **the energy concentration RoughQuant needs exists only in the
un-foldable per-weight eigenbasis; the raw/foldable basis has no exploitable
concentration.**

## The bug (user-caught) and its correction

The masked/protected QTIP+mq4 quantizers **zeroed** protected channels before the
FWHT+quant (to tighten the bulk range), then restored them. This is *non-monotonic*:
`mq4+protect` scored WORSE than `mq4` at some fracs (f0.015=30.73 > f0.0=29.44),
which is impossible if protection only adds precision. Fixed → quantize the full
group, then **overwrite** protected positions exact. Proven correct: protect=100%
→ PPL 26.1745 = bf16 floor exactly; monotone-decreasing at scale.

PPL is bit-exact deterministic (verified 3× identical), so the small-frac
non-monotonicity that remained is **pointwise-PPL nonmonotonicity** of tiny weight
perturbations, not a bug — and it's why PPL can't resolve the knee.

**Correction to de-risk A:** the headline "per-weight PCA beats mq4" was largely
the zeroing trick (which happened to clean the bulk when the protected channels
are large outliers). With the monotonic overwrite:

| config (bf16 embed) | ~avg-bits | PPL |
|---|---|---|
| bf16 floor | 16 | 26.17 |
| mq4 | 4.25 | **29.07** |
| PCA b3 f0.0 (no protect) | 3.13 | 33.63 |
| **PCA b3 f0.03 (overwrite, monotonic)** | ~3.5 | **29.37** ✗ |
| PCA b3 f0.03 (old zeroing, non-monotonic) | ~3.5 | 27.90 (artifact) |
| foldable shared rotation (de-risk B) | ~3.5 | 30.68 |
| permutation / channel-consistent r+w | ~3.5 | ~30.5 |

Protection still helps a lot (33.63→29.37 by protecting 3% of eigenvectors —
concentration is real), but the monotonic result does **not** beat mq4, and the
27.90 needed the non-monotonic outlier-zeroing trick AND is unfoldable regardless.

## Energy concentration analysis (the root cause; no GPU, no PPL noise)

`scripts/roughquant_energy_cdf.py` — cumulative energy CDF from the Hessian
(E[x²]) + bf16 weights. "Energy" = output-relevant `‖W[:,c]‖²·E[x_c²]`.

**RAW residual channels (the FOLDABLE basis):** essentially uniform — NO knee.

| top channels | % energy |
|---|---|
| 1% | 21% |
| 10% | 46% |
| 50% | 77% |
| 90% energy needs | 77% of channels |

Product (W²·E[x²]) ≈ activation (E[x²]) CDFs — so weight-aware saliency doesn't
change the picture; energy is spread either way.

**EIGENBASIS (the per-weight, dense, NOT-foldable basis):** sharply concentrated.

| input | top 1% eig | top 10% eig | 90% energy in |
|---|---|---|---|
| out_proj (k2048) | 81% | 94% | 4.6% of eig |
| in_proj_qkv (k1024) | 42% | 72% | 38% |
| gate_proj (k1024) | 41% | 68% | 40% |
| down_proj (k3584) | 31% | 69% | 31% |

## Why this reconciles every result

- **PCA-protect helps** (33.63→29.37): top eigenvectors hold concentrated energy,
  so protecting a few recovers a lot. Real, but starts from a bad rotated-bulk
  base and nets out above mq4.
- **Foldable variants lose** (shared rotation 30.68, permutation/channel ~30.5):
  the raw/foldable basis has no concentration (top 1% = 21%), so there's nothing
  cheap to protect; and one shared rotation can't match each input's own eigenbasis.
- **mq4+protect ≈ neutral at small fracs:** mq4's per-256 FWHT already handles
  generic raw-basis incoherence; no concentrated tail to exploit at 4-bit.
- **The knee** you sought: absent in the foldable basis, sharp only in the
  un-foldable per-weight eigenbasis. That is the fundamental tension.

## Bottom line

On Qwen3.5-0.8B, RoughQuant's energy-concentration premise is only realizable via
per-weight dense PCA rotations that don't fold for free, and even those (done
monotonically) don't beat mq4. The one un-foreclosed idea is a **learned
block-diagonal rotation** (per-256-block PCA, foldable in-kernel like mq4's FWHT
but data-fitted) — it could capture *some* eigenbasis concentration while folding.
mq4 already uses a *random* per-256 Hadamard; whether a *learned* block rotation
beats it is the open question (and the user's chosen next direction). Cross-model
(7B/9B) remains the other speculative avenue.

## Artifacts

- `scripts/roughquant_energy_cdf.py` (energy CDF analysis).
- `main.rs`: overwrite fix (monotonic); `HIPFIRE_RQ4_BULK={mq4,void}`,
  `HIPFIRE_RQ4_SALIENCY={diag,wnorm,product}`; `mq4_simquant_masked`,
  `bf16_colnorm2`; void(prune) mode.
- KLD tooling note: `eval_hipfire` can't eval bf16-weight candidates on gfx1151
  (unregistered lm_head gemv variant); the energy-CDF analysis is the noise-free
  substitute for the "where do we lose energy" question.
