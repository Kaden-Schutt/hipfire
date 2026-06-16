# RoughQuant — Phase 1: top-k column protection (no rotation)

**VERDICT: PROCEED to Phase 2.** The super-weight protection premise is
confirmed emphatically — protecting a tiny fraction of salient input columns
moves PPL by up to **97%** at 3-bit. BUT no un-rotated config reaches the mq4
gate (29.08), and 2-bit is hopeless without rotation even with protection. This
is the expected, healthy result: it isolates **rotation** (Phase 2) as the
lever that makes the bulk quantizable below 4 bits.

## Method

`hipfire-quantize --format roughquant-sim` (added this phase, `main.rs`):
per 2D weight, rank the `k` input columns by saliency, protect the top
`protect_frac · k` at full precision (left as bf16 in the sim), crush the rest
to a `bulk_bits` symmetric-uniform grid (per row, per 256-col group, absmax over
non-protected entries), bake back into bf16, eval PPL via the normal forward.

- **Saliency = diag(H) = E[x_c²]** from the Hessian sidecar (output-aware:
  quant noise on a high-energy input channel costs more output error — CMPQ's
  "quant-error impact"). Full coverage: 186/186 tensors used diag(H), 0 proxy.
- **No rotation** — this phase deliberately omits PCA/Hadamard to isolate the
  protection effect alone.
- Corpus `wikitext2-1024s-2048ctx.txt` (md5 `83b0205a…`), ctx 2048, warmup 8,
  2039 scored. Sweep: `scripts/roughquant_sweep.sh`. Env knobs:
  `HIPFIRE_RQ_{PROTECT_FRAC,BULK_BITS,GROUP}`.

## Results

Baselines (Phase 0): bf16 **26.17**, mq4 (gate) **29.08**, qtip3sim-ldlq 31.42.

| bulk_bits | protect_frac | avg-bits (est) | NLL/tok | PPL |
|---|---|---|---|---|
| 4 | 0.0   | 4.06 | 3.749  | 42.47   |
| 4 | 0.015 | 4.24 | 3.483  | **32.55** |
| 3 | 0.0   | 3.06 | 8.289  | 3978.5  |
| 3 | 0.015 | 3.26 | 4.657  | 105.36  |
| 3 | 0.06  | 3.84 | 4.288  | 72.81   |
| 2 | 0.0   | 2.06 | 15.353 | 4.65e6  |
| 2 | 0.005 | 2.13 | 15.523 | 5.51e6  |
| 2 | 0.015 | 2.27 | 15.276 | 4.31e6  |
| 2 | 0.03  | 2.48 | 13.601 | 8.07e5  |
| 2 | 0.06  | 2.90 | 12.245 | 2.08e5  |

## Reading

1. **Protection works, hugely, at 3–4 bit.** 3-bit: 3978 → 105 (−97%) by
   protecting just 1.5% of columns; → 72.8 at 6%. 4-bit: 42.5 → 32.6 (−23%) at
   1.5%. The diag(H)-ranked salient columns carry vastly disproportionate
   output energy — the super-weight / ResQ premise holds.
2. **2-bit is hopeless without rotation, protection notwithstanding.** Even at
   6% protected, 2-bit uniform stays at PPL 2e5 (broken). The *bulk* columns'
   own heavy-tailed distributions can't be represented by a 2-bit uniform grid;
   no choice of *which* columns to protect fixes the columns you don't. This is
   the canonical motivation for incoherence rotation (QTIP/ResQ): rotation
   Gaussianizes the bulk so a low-bit grid becomes viable.
3. **No un-rotated config beats mq4.** Best is 4-bit + 1.5% = 32.55 at 4.24
   avg-bits — *more* bits than mq4 yet +12% PPL, because mq4 already FWHT-rotates
   its 4-bit grid. Protection alone is necessary but not sufficient; rotation is
   the competitiveness lever.

## Implications for Phase 2

- The RoughQuant thesis (fp32-protected subspace + sub-4-bit bulk ≈ mq4)
  **cannot** be validated without rotation. Phase 2 must add it:
  PCA-rotate into the eigenbasis of C, bin by eigenvalue, within-tier Hadamard
  to Gaussianize each tier, then quantize per tier.
- Saliency-by-diag(H) is the right column ranking and is already wired and
  cheap. Phase 2 generalizes from a binary protect/bulk split to a multi-tier
  eigenvalue partition.
- The QTIP trellis (`qtip.rs`, already Gaussian-input-valid post-rotation) is
  the natural bulk format for 2–3 bit tiers; the no-rotation uniform grid used
  here is only a Phase-1 probe.

## Artifacts

- Code: `crates/hipfire-quantize/src/main.rs` (`roughquant_sim_tensor` +
  `roughquant-sim` post-pass), `scripts/roughquant_sweep.sh`.
- Generated `.hfq` were transient (quantize→PPL→delete); none committed.
- Fixtures: model `Qwen3.5-0.8B`, Hessian `~/.hipfire/hessians/qwen3.5-0.8b.hessian.bin`.
