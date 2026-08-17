# GPTQ PSD-projection rescue on MI300X — works, and rocSOLVER confirmed, but the EVD was not the bottleneck (gfx942)

**Lifecycle:** `historical`. Fixture-bound measured evidence. It is **not** a
current default, an automatic baseline, a product claim, or an admission
decision.

**Disposition:** The PSD-projection fallback (commits `0fba538b5`, `5c45efbd8`)
does what it was built to do:

- **Every previously-hard-failing tensor is now rescued.** `hard_failed = 0`
  across 10 layers, against **10 hard failures** at the same point before the fix,
  each of which had silently become round-to-nearest.
- **The rocSOLVER GPU eigensolver is confirmed running on MI300X**, not silently
  falling back: `6 PSD projection via GPU rocsolver_dsyevd`, `0 via CPU faer eigh`.
- **But the end-to-end speedup is only ~1.37x**, not the ~180x the eigendecomposition
  alone would imply. The EVD was never the dominant cost. See § "Amdahl".

---

## Fixture identity

| field | value |
|---|---|
| producer | MI300X, `gfx942`, CDNA3, 235 GB RAM, 205 GB VRAM — **PRODUCER ONLY** |
| binary | `hipfire-quantize` @ `aebf90c90`, md5 prefix `f0932bac806b` |
| calibration | `qwen3.8-27b.calibv6-814d8fd.hfq`, 31,481,139,200 B, md5 `9e2a7ab00e0b982e6793f4d2b0b86379` |
| parent | `/scratch/parents/qwen3.8-27b`, 18/18 shards |
| recipe | `--format mq4 --q8-router --hessian <calib> --ldlq`, `HIPFIRE_Q8_CLASSES=lm_head,embed`, **no `--awq`** so GPTQ is isolated |

---

## The rescue works

Pre-fix run (CPU, no projection), by layer 10:

```
rescued=0   hard_failed=10   fired=22
gptq: Cholesky failed for layers.0.mlp.down_proj (K=17408): failed even at
      damp=2.967069e-2 (diag mean=2.967069e-2); skip GPTQ for this tensor;
      falling back to RTN MQ4
```

Post-fix run (CPU EVD), by layer 10:

```
rescued=10   nonconv=0   hard_failed=0   fired=22
```

Post-fix run (GPU EVD), by layer 1:

```
rescued=6    nonconv=0   hard_failed=0   fired=10
rescued by K: 1x K=17408, 5x K=5120
6 PSD projection via GPU rocsolver_dsyevd
0 PSD projection via CPU faer eigh
```

Example rescue — the same tensor that previously exhausted its damping cap and
fell to RTN:

```
gptq: PSD projection rescued K=17408 Hessian
      (lambda_min=-1.076990e-1 before projection); Cholesky succeeded at damp=1.000000e-2
```

`damp=1e-2` is the **default**. After projection the tensor needs no unusual
damping at all.

Pre-fix, roughly 62% of the model was not receiving GPTQ: 10 of 21 tensors failed
outright, and 3 more "succeeded" only at near-maximal damping with
`clamps=0/31457280` — i.e. the correction had collapsed, which is RTN wearing a
GPTQ label. That second category is why a naive fired/attempted ratio understates
the damage.

---

## rocSOLVER path, confirmed on hardware

`rocsolver_dsyevd` bound alongside the existing `dpotrf`/`dtrtri`/`dgemm`, with
`ROCBLAS_EVECT_ORIGINAL` (211) and `ROCBLAS_FILL_LOWER` (122) to match the CPU
path's `Side::Lower`.

Reconstruction exploits the clipping: once `S` is clipped to `>= 0`, `sqrt(S)` is
real, so `U*S*U^T = (U*sqrt(S))*(U*sqrt(S))^T`. That is one `rocblas_dscal` column
scale plus a **single** `rocblas_dgemm(N,T)`, symmetric by construction — cheaper
than a diagonal scale plus a two-sided product, and everything stays on device.

Corroborating timing: the PSD test suite runs in **1.30 s on MI300X** versus
**234.60 s** on a host with no AMD GPU (CPU fallback) — a ~180x gap on the
eigendecomposition itself, consistent with the GPU doing the work.

The two paths log distinguishably (`via GPU rocsolver_dsyevd` /
`via CPU faer eigh`) precisely so that a silent fallback presents as a logged
regression rather than as a hang.

---

## Amdahl — the EVD was not the bottleneck

| run | elapsed to reach layer 1 |
|---|---|
| CPU EVD | 18:52 |
| GPU EVD | 13:43 |

**1.37x end-to-end**, against ~180x on the eigendecomposition in isolation. The
conclusion is that the EVD was a minor term:

- `compute_damped_inv_cholesky_upper` **already** ran its `dpotrf` → `dtrtri` →
  `dgemm` chain on the GPU before this change, so the Cholesky path was never the
  CPU cost either.
- What now dominates is the **blocked §3.2 sequential column update**, still on
  CPU at `O(M * K^2)`. For `down_proj` that is `5120 * 17408^2 = 1.55e12` ops per
  tensor, and there are 64 of them.

Projected total at the observed rate: **~14.6 hours** for the full 27B artifact.
That is tractable but it means the next optimization target is the weight update,
not the linear algebra. Moving the EVD to the GPU was necessary — the CPU EVD
would have added ~18 hours on top — but it was not sufficient, and this record
exists so that is not rediscovered.

---

## What this record does NOT yet establish

**The artifact has not been scored.** The claim "GPTQ beats its non-GPTQ twin on
KLD" is unproven here; the run is still producing. Scoring must happen on hiptrx
(`gfx1201`), never on this producer, because uncalibrated MQ4 is degenerate on
gfx942 (KLD ~12.7 / PPL ~1.8e6) and any number from that host is worthless.

Baselines it will have to beat, all at 15,662,615,552 bytes on the WT2 reference:

| arm | KLD |
|---|---|
| native HFQM calib v6 + AWQ 0.55 | 0.086790 |
| native HFQM calib v5 + AWQ 0.55 | 0.087921 |
| **off-the-shelf imatrix + AWQ 0.55** | **0.043776** |

Note the bar is the third row, not the first two. An earlier finding established
that consumed off-the-shelf imatrix beats native calibration by 1.98x at identical
size and identical decode speed, and that native calibration is Pareto-dominated by
three arms including one that is smaller, faster **and** better. GPTQ is a
different mechanism (cross-column error compensation via the Hessian inverse, which
an imatrix cannot express), so it is not redundant with that finding — but it must
clear 0.043776, not 0.086790, to matter.

Scoring must use **both** the WT2 prose tripwire and the v6 conversation selector:
prose compresses chat-model arm margins ~3x relative and has already inverted a PPL
ranking in this campaign.
