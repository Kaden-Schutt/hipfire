# TODO — Learned rotations for W4A4 (SpinQuant) via hipfire-train

Status: **design / not started.** High-leverage: this is what makes 4-bit
**activations** usable, unlocking the `iu4×iu4` matrix path (~2× the compute of
`iu4×iu8` in the compute-bound regime). Ref: SpinQuant, arXiv:2405.16406
(`Quantization-research/SpinQuant/`). Built on the same `hipfire-train` autograd
the GuidedQuant thread uses.

## Why (the payoff)

- **Compute.** hipfire's `Oq4G256` (qt=34) is the real **W4A4** path (int4
  weight × int4 activation, `iu4·iu4` GEMM). `OqPlusG256` (qt=33) is the W4A8
  fallback (nibble-expand to int8, `iu8` GEMM). On the RDNA WMMA, `iu4×iu4` is
  ~2× `iu4×iu8` **in the compute-bound regime** (prefill / batched serving; GEMM
  matrix-core-bound). Decode stays weight-**bandwidth**-bound, so the direct win
  is prefill/batch throughput + the activation-side memory/KV traffic halving —
  not single-stream decode tok/s.
- **The blocker is A4 quality, not the kernel.** Per the repo's own note, we
  keep experts/important tensors on W4A8 because *4-bit activations are usually
  unusable* — activation outliers (measured kurtosis > 200 in LLMs) blow up the
  4-bit grid. Fixing A4 quality flips the default from `iu4×iu8` → `iu4×iu4`.
- **Rotation is the fix, and learning it is the multiplier.** SpinQuant W4A4
  end-to-end SNR (dB): `R=I: −2.9` → `random/fixed rotation: +0.9` → **learned
  rotation: +6.8**. hipfire today uses a **fixed** per-256-group FWHT (the
  QuaRot / +0.9 dB tier). Learning the rotation is the path to the +6.8 tier,
  closing W4A4KV4 to ~2.9 pts of FP (Llama-2-7B) and beating QuaRot most on the
  hard models (Llama-3).

## Background: SpinQuant mechanism (recap)

The full-precision transformer is **rotation-invariant**: inserting orthonormal
`R` and its inverse at matched points leaves FP output unchanged, but changes
the *quantized* output. SpinQuant places four:

| R | where | absorbed? | hipfire status |
|---|---|---|---|
| **R1** | residual stream (embed out; reversed before attn/FFN nonlinearity) | **merged into weights** (fold RMSNorm scale into next weight, à la SliceGPT) | **missing** (new) |
| **R2** | value matrix + o_proj input, head-wise `[D_head,D_head]` | **merged** | **missing** (new) |
| **R3** | KV-cache (Q·K), online Hadamard | online FWHT | partial (KV quant path) |
| **R4** | down_proj input, online Hadamard | online FWHT | **already have** (down FWHT) |

R1+R2 are *mergeable* ⇒ **zero new params, zero runtime cost** once baked in.
Only R1/R2 are **learned**; R3/R4 stay fixed Hadamard (online, cheap). Learning:
minimize the **quantized network's CE loss** over `{R1,R2}` (weights frozen) via
**Cayley SGD on the Stiefel manifold** (orthonormal-preserving update
`R' = (I−α/2·Y)⁻¹(I+α/2·Y)R`, `Y = Ĝ−Ĝᵀ` skew-symmetric from the projected
gradient; ~2× SGD cost/iter, computed by fixed-point iteration to avoid the
inverse). ~0.26% of params, ~100–200 iters on ~800 WikiText sequences.
Key identity: `∂(quantized output)/∂R` is **nonzero only when quantization is
present** — learning R is purely aligning the quant grid to the data.

## What hipfire already has (reuse)

- **Autograd + differentiable quantized forward** — `hipfire-train`
  (`TrainTensor`, `model_forward`/`model_*_backward`), and `oqplus_quant.rs`
  which reproduces the **OQ+ W4 damage as a differentiable fp32 round-trip**
  (FWHT → clip-search scale → int4 → dequant → inverse FWHT). That is the
  SpinQuant training forward's weight side.
- **CE loss** on calib text (`cross_entropy`, `calib_guided`'s tokenizer path).
- **FWHT** (`hipfire-primitives::fwht`, `mq_rotate_x`) for R3/R4.
- **The W4A4 kernel** (`Oq4G256` `iu4·iu4` GEMM) and its loader/repack.
- **Rotation-invariance precedent** — the MQ/OQ formats already fold a rotation
  into the weight offline; R1/R2 absorption is the same idea at residual scope.

## What to build (phased)

**Phase 0 — rotation-invariant model transform.** Fold each RMSNorm scale `α`
into the following weight so the block is rotation-invariant, then verify a
random `R1` leaves the **FP** output bit-identical (the invariance is the
correctness contract). Do this in `hipfire-train`'s llama model (dense, tied).

**Phase 1 — R1 with a FIXED Hadamard, mergeable.** Insert `R1` (residual,
`[hidden,hidden]`) as a fixed random Hadamard, merged into embed/attn/FFN input
weights. Quantize OQ4 (real W4A4) and confirm: (a) FP invariance holds, (b) the
`iu4·iu4` path runs, (c) A4 SNR/KLD improves vs no-R1. This reproduces the
QuaRot/+0.9 dB tier and validates the plumbing before learning.

**Phase 2 — Cayley-SGD learn R1.** Add a small Stiefel-manifold optimizer
(`crates/hipfire-train/src/ops/` + `optim.rs`): Cayley update with fixed-point
inverse, orthonormality asserted each step. Objective = quantized-CE over calib
sequences with `Q(W R1⁻¹) Q(R1 X)` in the forward (weight side via
`oqplus_quant` sim, activation side via an A4 sim-quant to be added). Weights
frozen; ~100–200 iters. Expect the +6.8 dB tier.

**Phase 3 — add R2 (head-wise), learn {R1,R2} jointly.** `R2 [D_head,D_head]`
on V + o_proj input; merged. This is the "W4A8-gap-closing" pair even before R3/R4.

**Phase 4 — R3/R4 online Hadamard.** R4 = the existing down FWHT (verify it's
positioned as SpinQuant's R4). R3 for KV4 (when we do 4-bit KV). Both fixed.

**Phase 5 — export + validate.** Merge learned R1/R2 into the weights at
quantize time, emit an `Oq4G256` `.hfq` with a rotation sidecar for R3/R4 seeds.
Validate end-to-end: **W4A4 KLD / zero-shot vs fixed-rotation OQ4**, and
**measure the prefill/batched throughput** of `iu4·iu4` vs `iu4·iu8`.

## New pieces (net)

1. `hipfire-train` rotation-invariant model transform (RMSNorm-scale fold).
2. Learnable `R1 [hidden,hidden]`, `R2 [D_head,D_head]` params + their merge.
3. **A4 activation sim-quant** in the training forward (int4 per-group activation
   round-trip; the weight side `oqplus_quant` already exists).
4. **Cayley-SGD optimizer** on the Stiefel manifold (small, ~2× SGD/iter).
5. Quantize-time **merge + export** of learned R1/R2 into `Oq4G256`.
6. A rotation-SNR probe (analog of `--ldlq-probe`) for fast per-tensor iteration.

## Constraints / notes

- **LLaMA-dense only** (hipfire-train's model) initially; MoE (minimax/qwen35-moe)
  needs trainable models — separate lift.
- The **A4 path must be the real runtime quantizer** (int4 activations, per-group
  scale) so the learned rotation optimizes the deployed grid, not a proxy.
- **Decode is bandwidth-bound** → the W4A4 compute win is prefill/batched serving;
  frame benchmarks accordingly (do NOT expect single-stream decode tok/s gains
  beyond the activation-traffic reduction).
- **Composes with GuidedQuant/LDLQ:** rotation is the *first-order* W4A4 enabler
  (free at runtime, mergeable); the Guided/LDLQ Hessian is the *second-order* W4
  refinement *within* the better-conditioned rotated basis. Do rotation first.
- Cayley SGD needs the loss to actually depend on R through the quantizer — verify
  the gradient is nonzero (it is only under quantization; a good unit check).

## Success criteria

- FP rotation-invariance bit-exact (Phase 0/1).
- Learned-R1 OQ4 W4A4 SNR ≥ fixed-Hadamard by a clear margin (target the SpinQuant
  ~+5.9 dB learned-over-random delta, scaled to our setup).
- W4A4 KLD within a small gap of W4A8 on the tiny-quant battery.
- `iu4·iu4` prefill throughput ≈ 2× `iu4·iu8` on gfx1151, with quality now
  acceptable — flipping the default activation path.
