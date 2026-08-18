# MQ4-GL clips the largest coefficient in 82.6% of blocks — the mechanism behind its KLD loss

- **Date:** 2026-08-17
- **Lifecycle:** `historical`
- **Fixture:** `model.language_model.layers.20.mlp.down_proj.weight` from the
  `Qwen3.8-27B` bf16 parent at `~/models/Qwen3.8-27B` (shape `[5120, 17408]`,
  first 2048 rows → 139,264 post-FWHT 256-blocks)
- **Method:** raw safetensors parse, bf16→f32 widening (`u16 << 16`), FWHT-256
  with the engine's sign tables (xorshift seeds 42 / 1042), per-block RMS
  normalisation — the exact transform `gl_encode_block` applies
- **Disposition:** diagnostic. Explains the `2ec9c5703` / `db80c8bbd` KLD
  result; motivates a range-widened codebook. Not a perf claim.

## Question

MQ4-GL wins codec MSE by 22.69% and loses KLD by 11.28% (WT2) / 27.52% (v6
selector). Codec MSE was already shown non-predictive; that says the objective
is wrong but not *which* weights are being served badly. This measures it.

## Result — GL's fixed range clips the dominant coefficient in most blocks

`GL_CB4`'s outermost level is **2.7326**, the Lloyd-optimal outermost magnitude
for a unit Gaussian at 16 levels. Real post-FWHT blocks routinely exceed it:

| quantity | value |
|---|---|
| fraction of **all weights** beyond ±2.7326 | **0.606%** |
| **blocks whose largest coefficient is clipped** | **82.57%** |
| per-block max \|z\| | mean 3.048, p50 3.002, p99 4.061, max 5.788 |
| overshoot when clipped | mean 1.108×, max 2.118× |
| clipped coefficients per block | mean 1.55 (0 in 17.4%, 1 in 34.5%, 2 in 29.4%, 3 in 13.8%, max 8) |

**Uniform affine cannot clip at all** — it fits per-block min/max by
construction. That is the structural asymmetry between the two formats, and it
is exactly the 0.1875 bpw GL "saves" by dropping the per-block zero-point and
range fit.

## Why this reconciles the two metrics

Only 0.606% of weights are clipped, and only by ~11% on average, so **codec MSE
barely registers it** — GL still wins MSE by 22.69% on the strength of the other
99.4%. But the clipped value is the **largest-magnitude coefficient in 5 of
every 6 blocks**, which is precisely what moves logits. KLD is dominated by it.

That is the whole observed pattern, with no appeal to level *shape*:

- GL wins every unweighted metric (MSE, NLL, PPL) — the bulk is better served.
- GL loses KLD — the dominant coefficient is truncated in most blocks.
- The KLD gap widens from +11.28% (prose) to +27.52% (conversation), because a
  more peaked distribution depends more heavily on the dominant weights.

## Two hypotheses killed on the way

**The outlier is not the DC coefficient.** A Hadamard transform's first row is
all-ones, so an unrandomised transform would concentrate the block mean into
coefficient 0. Measured position of each block's largest \|z\| is essentially
**uniform** — max share 0.43% against a 0.39% uniform baseline, spread over all
256 indices. The randomised sign flips (seeds 42/1042) destroy DC concentration,
which is what they exist for. So a fixed-position outlier slot is not available,
and with mean 1.55 clipped per block a single slot would not suffice regardless.

**Importance weighting cannot fix this.** It repositions levels *within* the
existing range; it does not extend the range. Weighting toward high-mass regions
pulls levels toward the bulk and makes clipping **worse**. Compounding it, the
FWHT Gaussianises every block — the previously measured 1.8% ceiling on per-block
codebooks says the optimal level shape is universal — so reweighting Gaussian
blocks by per-block importance leaves the pooled shape, the only thing a global
codebook sees, essentially unchanged.

## The lever that is actually available: range, at zero byte cost

| outermost level | blocks clipped | weights clipped |
|---|---|---|
| **2.7326** (current, MSE-optimal) | **82.57%** | 0.606% |
| 3.2 | 28.70% | 0.127% |
| 3.6 | 6.95% | 0.028% |
| **4.0** | **1.29%** | 0.005% |
| 4.5 | 0.13% | 0.001% |

Widening costs no bytes — the codebook is 16 f32 constants — but it necessarily
coarsens the bulk at fixed level count, so **unweighted MSE will get worse**.
Under the KLD-only criterion that is an acceptable trade, and it is the point:
the codebook being MSE-optimal *is* the defect.

### This retro-explains an earlier result

A Lloyd refit on real post-FWHT data was measured to go **narrower** (span 2.231
vs 2.7326) and score 50% worse on MSE, and was filed as "real-data Lloyd is
worse." Correct on the number, wrong on the reading: MSE was the wrong judge.
Narrower means **more** clipping, so that variant should be expected to score
worse on KLD than GL_CB4, not better. Real-data Lloyd is the wrong direction;
the range must be pinned wider deliberately, against the MSE gradient.

## Next

Build `GL_CB4R`: 16-entry antisymmetric codebook, outermost magnitude pinned at
4.0, remaining 7 magnitudes constrained-Lloyd-optimised for a unit Gaussian.
Quantize and score against both references. Success is KLD only:

| arm | WT2 KLD | v6sel KLD |
|---|---|---|
| mq4 uniform affine | 0.043776 | 0.587566 |
| mq4gl GL_CB4 | 0.048713 | 0.749238 |
| mq4glr GL_CB4R | pending | pending |

If KLD improves while MSE worsens, the clipping explanation is confirmed. If it
does not, clipping is not the mechanism and GL should be retired.
