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

---

## Amendment — the mechanism works perfectly and the ARTIFACT IS DEGENERATE

The 27B run completed. Every mechanism this record was written to validate did
exactly what it was built to do, across all 64 layers:

```
rescued=23   hard_failed=0   gpu_evd=23   nonconv=0
artifact: 15,655,791,616 B, md5 prefix a5c1b76c0c5dad, "Done: 15655.8 MB written"
```

Zero hard failures. Every rescue on the rocSOLVER GPU path. No eigensolver
non-convergence. Clamp rates modest and sane: min 0.000%, **median 0.776%**, max
1.394% across 496 tensors. The packed-code oracle emitted no mismatch, no warning,
no NaN, no assert.

**And the artifact is degenerate.** Scored on hiptrx `gfx1201`:

| arm | ref | KLD | NLL | PPL |
|---|---|---|---|---|
| `gptqpsd` | WT2 | **8.367924** | 10.124605 | **24949.4105** |
| `gptqpsd` | v6 selector | **8.601600** | 9.444278 | 12635.6596 |
| `mq4` (same size, no GPTQ) | WT2 | 0.043776 | 1.857666 | 6.4088 |
| uncalibrated MQ4 | WT2 | 0.066668 | — | — |

KLD 8.37 against a uniform-output ceiling of `ln(248320) = 12.42`. PPL 24,949
against 6.41. **GPTQ made the model roughly 125x worse than doing nothing at
all** — plain RTN MQ4 scores 0.066668.

### What this means for the fix, stated precisely

The PSD projection is **not** implicated by this evidence, and is also **not
exonerated** by it. What is established:

- The projection does what it claims: it converts hard Cholesky failures into
  successes at default damping, on GPU, at scale, with no non-convergence.
- Enabling GPTQ on ~62% more tensors made the output much worse. Before the fix,
  those tensors silently fell back to RTN — which, given RTN scores 0.066668 and
  GPTQ scores 8.37, means **the silent fallback was protecting the model from a
  broken GPTQ implementation.**

That is the uncomfortable reading and it is the honest one. A fix that removes a
fallback is only an improvement if the primary path is correct, and this evidence
says the primary path is not.

### What was ruled out

- **Basis mismatch — RULED OUT.** MQ4 is FWHT-rotated while the stored Hessian is
  raw pre-rotation, which looked like the obvious culprit. But
  `gptq_pipeline_mq4g256` takes `h_unrot_f32` and applies
  `fwht_similarity_per_256(&mut h, ...)` at `gptq.rs:1656`, rotating the Hessian
  into the weight basis, exactly as its module doc specifies
  (`H_target = FWHT_per_256_similarity(diag(1/s) · H_unrot · diag(1/s))`). Since a
  similarity transform preserves eigenvalues, the bf16-induced negative
  eigenvalues survive it consistently, so the PSD analysis remains valid in the
  rotated basis too.
- **AWQ absence.** This run deliberately omitted `--awq` to isolate GPTQ, so the
  `diag(1/s)` term is identity. That is the intended configuration, not a defect,
  and identity scaling cannot explain a 125x regression.
- **Numerical blowup.** Clamp rates peak at 1.394%. Weights are not being driven
  out of range.

### Correcting an earlier interpretation in this record

An earlier note read the clamp rate *rising with depth* (0.236% at layer 0 to
0.998% at layer 10) as "the signature of genuine sequential error compensation."
That inference is withdrawn. Error accumulating with depth is equally the
signature of a compensation term with the **wrong sign or wrong magnitude**
accumulating with depth. The observation does not distinguish the two, and given
the final KLD it more likely indicated the latter.

### The isolation test, and what survives the producer's teardown

The decisive experiment is whether GPTQ is broken *independently* of the PSD
projection. Two cross-architecture arms completed with the projection **never
firing**:

| model | bytes | rescued | hard_failed | md5 |
|---|---|---|---|---|
| `lfm2.5-350m` | 229,474,032 | **0** | 0 | `7fcd86de6836` |
| `lfm2.5-1.2b` | 698,645,168 | **0** | 0 | `48e4bb612ef6` |

`rescued=0` on both is itself a clean confirmation of the linear-in-`K` law: at
small `K`, `-lambda_min/mean(diag(H))` is far below the damping cap, so no rescue
is needed and the fallback is correctly inert.

Because PSD projection never ran on these two, **if their artifacts are also
degenerate then GPTQ is broken on its own** and the projection is fully exonerated.
Both were pulled off the producer before teardown (md5-verified against the
producer's own record) specifically to preserve that test. Scoring them needs
lfm2.5 KLD references, which do not exist yet.

A `gemma4-12b` GPTQ run was in flight at teardown (`rescued=1, hard_failed=0`) and
did not finish; `gemma4-12b.ref_wt2.bin` and `ref_ag.bin` exist on the Hub, so that
remains the cleanest future isolation route on a mid-size model.

### Disposition change

The heading of this record — "works, and rocSOLVER confirmed" — stands as to the
**mechanism** and must not be read as endorsing the **output**. Until GPTQ itself
is validated end-to-end against a known-good reference, `--ldlq` should not be used
to produce shippable artifacts. Note that LDLQ was already recorded elsewhere in
this campaign as "never completed", i.e. it appears never to have been validated
end-to-end at any point; this run is the first evidence of what it actually
produces, and the answer is garbage.

---

## Amendment 2 — the likely cause is the COLLECTION HOST, not GPTQ

The previous amendment's leading reading ("GPTQ is broken") is **withdrawn as the
primary hypothesis**. A simpler explanation covers every datapoint.

Scoring host is confirmed correct — `GPU dev 0: gfx1201 (34.2 GB VRAM, HIP 7.14)`
in the score logs, so 8.367924 is a valid gfx1201 measurement of the artifact and
is not itself a gfx942 artefact.

But the **artifact was produced from Hessians collected on gfx942**, and gfx942 has
a documented faulty forward path — see
`2026-08-16-amendment-qwen38-mq4-collapse-is-a-gfx942-execution-bug` and the
degenerate uncalibrated-MQ4 behaviour on that arch. Calibration statistics are
gathered *by running forward passes*. If the forward path is wrong, the statistics
are wrong.

Sort every arm by how much of the gfx942-collected statistic it consumes:

| arm | statistics source | what it consumes | WT2 KLD |
|---|---|---|---|
| uncalibrated MQ4 | none | nothing | 0.066668 |
| off-the-shelf imatrix + AWQ | llama.cpp, **never gfx942** | per-channel scale | **0.043776** |
| native imatrix + AWQ (v6) | gfx942 forward | diagonal only | 0.086790 |
| native full Hessian + GPTQ | gfx942 forward | full off-diagonal structure | **8.367924** |

The ordering is monotone in *dependence on gfx942-derived data*, and the only arm
that beats doing nothing is the only one whose statistics never touched that host.
AWQ reads just the diagonal and lands mildly worse than nothing; GPTQ reads the
entire off-diagonal structure and lands catastrophically worse. That is exactly the
dose-response curve a corrupt collection host would produce, and it requires no
defect in GPTQ at all.

This also retro-explains the earlier finding that **native calibration is
Pareto-dominated by three arms including one that is smaller, faster and better**.
That was recorded as a fact about AWQ. It is more likely a fact about **where the
calibration was collected**.

### Consequences

1. **Do not conclude GPTQ is broken.** The evidence is equally consistent with GPTQ
   faithfully propagating corrupt second-order statistics. The `lfm2.5` isolation
   arms cannot settle this either, since their Hessians came from the same host.
2. **The 179.5 GB of calibration on `hipfire-models/hipfire-calib-data` is
   suspect**, not merely lossy. Its README documents the bf16 PSD issue; it should
   additionally warn that every artifact was collected on gfx942, whose forward path
   is known-faulty, and that no artifact derived from it has ever beaten
   uncalibrated on this model.
3. **The correct next experiment is to re-collect calibration on gfx1201** — the
   host that scores correctly — and re-run both AWQ and GPTQ against it. Until then
   native calibration has never been tested on trustworthy input, and neither has
   GPTQ.
4. **The PSD projection remains validated as a mechanism** and unaffected by this:
   it converts hard Cholesky failures into default-damping successes on GPU at
   scale. What it was fed is the open question.

### On the spend

The campaign bought ~12 h of gfx942 time to collect ~180 GB of second-order
statistics, and the arm that wins is the one that ignored all of it in favour of a
13.6 MB off-the-shelf GGUF imatrix. Producer-only discipline correctly kept gfx942
out of the *scoring* path; what was missed is that a faulty forward path
contaminates the *collection* path just as surely, and collection is the more
expensive half. That is the transferable lesson: a host disqualified from measuring
is thereby disqualified from calibrating, because calibration is measurement.
