# RoughQuant — Phase 2g: CORRECTION — outliers are real, shared, and foldable

**This corrects the wrong "energy is spread / no foldable concentration" claim in
phase2e/2f.** Prompted by user skepticism that our results didn't match the
quantization literature (AWQ, super-weight, SpQR, QuIP all find sharp salient
concentration). They were right — the earlier reading was an analysis artifact.

## What was wrong

phase2e aggregated the per-channel energy across all 138 residual readers and saw
a ~linear CDF (top 1% = 21%), concluding "energy is spread, concentration only in
the un-foldable per-weight eigenbasis." That aggregation was the error:

- **Per-tensor, the activations have strong outlier features** (literature-
  consistent): `diag(H)` max/median = 7–283× (gate_proj 283×, kurtosis 391;
  layer-12 in_proj 187×). Textbook LLM.int8/AWQ outliers. (Hessian collection is
  the standard `H += xᵀx` on post-norm layer inputs — correct.)
- **The outlier channels are SHARED across layers, not per-tensor-disjoint.** The
  union of each input-point's top-2% outlier set across all 48 residual
  input-points is only **75 channels** (of 1024). Jaccard overlap between mid/late
  layers is 0.4–0.9; layer 0 is the exception (its own unique outliers, overlap
  ~0.05). So ~75 residual dims are persistent outliers — a small, SHARED, and
  therefore **foldable** set.

Aggregating energy mixed layer-0's unique huge outliers into the ranking and
flattened the shared structure into a fake-linear CDF. And de-risk B tested a
shared *rotation* (must match every layer's eigenbasis — fails), NOT shared
channel *protection* (keep ~75 dims precise everywhere — viable).

## Corrected result: foldable outlier protection works (KLD vs bf16)

| config | avg-bits | KLD |
|---|---|---|
| mq4 | 4.25 | 0.162 |
| mq4 + protect top 5% (shared, bf16) | 4.84 | 0.084 |
| mq4 + protect top 7% | 5.07 | 0.076 |
| mq4 + protect top 10% | 5.43 | 0.072 |
| mq4 + protect top 15% | 6.01 | 0.057 |
| mq6 (uniform 6-bit) | 6.25 | 0.0084 |

Protecting the shared outlier set (foldable, no rotation) **halves mq4's KLD** for
+0.6 bits and beats the crude mq4→mq6 linear interpolation at ~5 bits. So the
energy-concentration premise IS real and IS foldable — contrary to phase2e.

## But uniform still wins at iso-bits (the deployment verdict stands)

At ~6 bits, uniform **mq6 (0.0084) beats protect-15% (0.057) by 6.8×**. Reason,
now correctly understood:
1. **mq4 already does incoherence processing** (per-256 FWHT + grouping), which
   captures most of the outlier benefit for free. The papers' protection wins are
   over *naive RTN*; against an FWHT-grouped 4-bit baseline the marginal benefit
   is small.
2. **The bulk dominates the bit budget.** The non-outlier ~93% still carries ~1/3
   of the error at 4-bit (soft knee: top 20% = 2/3, bulk = 1/3). Only uniform
   bit-increase fixes the bulk; protection doesn't touch it.

So spending a bit budget on uniform precision (mq6) beats spending it protecting a
few outliers, on this model. RoughQuant's premise is sound and literature-aligned;
it just doesn't beat a strong already-incoherent uniform baseline at iso-bits.

## Reconciliation with the literature (the answer to "why don't our results match")

The papers are right: outliers exist (we confirm: ~75 shared dims, max/med up to
283×) and protecting them helps (we confirm: mq4 KLD halved). The apparent
conflict was (a) our wrong aggregated-CDF reading, and (b) comparing against mq4's
FWHT baseline, which already does what AWQ-style scaling/protection does, so the
incremental win is small and uniform-bit-increase dominates.

## Open / next

- **Q8 (not bf16) protection** would cut the protection bit-cost ~½ (protect at
  8-bit ≈ lossless for those channels); could shift the Pareto. Untested.
- **Persistence-based selection** (rank by outlier-frequency across layers, not
  aggregated energy) may protect the shared set more efficiently than the
  energy ranking (which over-weights layer-0's unique outliers). Untested.
- **Cross-model (7B/9B):** does a bigger model's outlier/bulk balance shift enough
  that protection beats uniform? The reconciliation predicts probably not (any
  FWHT-grouped baseline captures the outliers), but scale changes redundancy.
- Supersedes the "no concentration" framing in phase2e/2f; the deployment verdict
  (uniform ≥ protection at iso-bits on a strong baseline) is unchanged.
