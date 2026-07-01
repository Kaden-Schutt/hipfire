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

**Phase 0 — rotation-invariant model transform. ✅ DONE.** Fold each RMSNorm
scale `α` into the following weight so the block is rotation-invariant, then
verify a random `R1` leaves the **FP** output bit-identical (the invariance is
the correctness contract). Do this in `hipfire-train`'s llama model (dense, tied).
- Landed as `crates/hipfire-train/src/rotation.rs`: `Rotation` (orthonormal
  `[h,h]`; `identity`/`random`(Gram–Schmidt Gaussian)/`hadamard`(random-sign
  Sylvester, the Phase-1 fixed rotation)) + `apply_r1(gpu, model, R)` which folds
  norm1→{q,k,v}, norm2→{gate,up}, final_norm→lm_head, then reader-rotates
  (`W Rᵀ`) q/k/v/gate/up/embed/head and writer-rotates (`R W`) o_proj/down_proj.
  Unties the head first (input-embed vs α_f-folded head diverge under tie); the
  tied backward stays a Phase-2 item.
- Verified: CPU unit tests (orthonormality, reader/writer invariance identities)
  + GPU probe `examples/rotation_invariance_probe.rs` on gfx1151: random-R1
  `max|Δlogit|=2.6e-6`, fold-only `2.4e-7` — invariant to fp reassociation.

**Phase 1 — R1 with a FIXED Hadamard, mergeable.** Insert `R1` (residual,
`[hidden,hidden]`) as a fixed random Hadamard, merged into embed/attn/FFN input
weights. Quantize OQ4 (real W4A4) and confirm: (a) FP invariance holds, (b) the
`iu4·iu4` path runs, (c) A4 SNR/KLD improves vs no-R1. This reproduces the
QuaRot/+0.9 dB tier and validates the plumbing before learning.
- **(a) DONE** via Phase 0 (`apply_r1` with `Rotation::hadamard`).
- **(c) A4 SNR — DONE.** New `crates/hipfire-train/src/a4_quant.rs`: `a4_simquant`
  (per-256-group symmetric int4 absmax round-trip = the runtime A4 grid) + `snr_db`
  + `rotation::rotate_rows`. Key metric lesson: raw-activation reconstruction SNR
  is a *bad* proxy (its Frobenius norm is dominated by the outliers, which
  quantize well — it rewards keeping them and ignores the crushed bulk); the
  faithful metric is **end-to-end output SNR through the weight**, where the
  crushed bulk propagates. CPU test (`hadamard_beats_identity_end_to_end`):
  identity 9.35 dB → Hadamard 20.97 (+11.6). GPU probe
  `examples/rotation_a4_snr_probe.rs` on real Supra-50M: q_proj 14.4→22.7 (+8.3),
  gate_proj 9.95→18.6 (+8.6). Rotation moves the int4-activation grid from
  marginal to usable, as SpinQuant predicts. Hadamard ≈ random (Hadamard the
  canonical fixed choice).
- **(b) DONE** (real `iu4·iu4` end-to-end). Working kernel copy
  `kernels/src/gemm_iu4_i32_wmma_r1.hip` (symbol `gemm_iu4_i32_wmma_r1`, own
  `Gpu::gemm_iu4_i32_wmma_r1` wrapper) — byte-identical to production, so the
  learned-rotation work can evolve it (fuse R4 FWHT / dequant epilogue) without
  touching the Oq4 forward. `parity_gemm_iu4_i32_wmma_r1`: EXACT vs int ref +
  bit-identical to production. Probe `examples/w4a4_r1_probe.rs` runs the real
  Oq4 W4A4 recipe (FWHT-256 signs 42/1042 → clip-search int4 weight / absmax int4
  act → grouped iu4 GEMM → f32 rescale) on real Supra-50M, GPU == CPU-sim exactly
  (int GEMM is exact). Result (mean q_proj SQNR): **naive W4A4 13.1 dB → +per-group
  FWHT recipe 20.1 dB → +fixed Hadamard R1 19.95 dB (−0.17)**.
- **KEY FINDING:** hipfire's Oq4G256 recipe *already* applies a per-256-group
  FWHT, which already captures the fixed-rotation (QuaRot) tier — so a second
  *fixed* data-agnostic R1 adds ~nothing on top. The +6.8 dB SpinQuant payoff
  requires a **learned** R1 (Phase 2); `w4a4_r1_probe` is the baseline it must
  beat. (Note the earlier A4-SNR probe's +8.3 dB was vs *no* rotation at all,
  i.e. it re-derives the per-group-FWHT benefit; the marginal value of a fixed R1
  over the deployed FWHT recipe is what's ~0 here.)
- Toolchain note: the `_r1` kernel isn't in the precompiled cache, so it JITs —
  set `ROCM_PATH=$HOME/.venv/.../_rocm_sdk_core` (has `lib/llvm/bin/clang++`); the
  runtime-only `/opt/rocm` lacks the compiler.

**Phase 2 — Cayley-SGD learn R1. ✅ DONE (learned > fixed demonstrated).**
Stiefel-manifold optimizer landed in `crates/hipfire-train/src/learn_rotation.rs`.
- **Optimizer** `cayley_step`: `A = Ĝ − Ĝᵀ` (`Ĝ = G Rᵀ`), Cayley update
  `R' = (I + α/2 A)⁻¹(I − α/2 A) R` via fixed-point inverse (no explicit solve).
  `A` is Frobenius-normalized so `lr` is a bounded per-step rotation *angle*
  (raw-gradient magnitude otherwise breaks the fixed-point contraction → NaN);
  periodic `reorthonormalize` (new on `Rotation`) guards drift from the
  approximate inverse. Unit-tested: `cayley_solves_procrustes` (L↓ >100×, stays
  orthonormal) + `learns_to_reduce_kurtosis`.
- **Objective — the key design call.** A plain STE on the quantized recon loss
  has a ~zero gradient w.r.t. `R` (the clean `X Rᵀ·R Wᵀ = X Wᵀ` term is
  R-invariant; STE zeroes the noise term — the loss looks flat exactly where it
  isn't). So we minimize the **differentiable incoherence proxy**: per-element
  4th moment `L(R)=Σ(X Rᵀ)⁴`. Orthonormal `R` fixes the 2nd moment, so this
  minimizes kurtosis → flattens the outlier tails that crush the int4 grid, with
  a clean dense gradient `G = 4(X̃∘³)ᵀX`. (This is a surrogate for SpinQuant's
  quantized-CE; the CE-through-the-net path is a later option if needed.)
- **Result** (`examples/learned_r1_probe.rs`, real Supra-50M, learn on stacked
  xn1+xn2, kurtosis Σx⁴ 1.06e7→had 5.4e5→**learned 2.8e5**, orthonormality 6e-7):
  A4 output SNR q_proj **identity 14.4 → fixed-Hadamard 22.7 → learned 27.2 dB**
  (**+4.5 dB over fixed**), gate_proj 9.95 → 18.6 → 23.1 (+4.5). The learned-over-
  fixed delta matches SpinQuant's ~+5.9 dB learned-over-random claim.
- **OPEN follow-ups:** (i) compose the learned R1 with the deployed per-group
  FWHT recipe and re-measure through `w4a4_r1_probe` (does learned beat the ~20 dB
  FWHT baseline where fixed R1 didn't?); (ii) the tied-head backward deferred from
  Phase 0 if we move to a true quantized-CE objective; (iii) bake the learned R1
  into an `Oq4G256` export (Phase 5).

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
