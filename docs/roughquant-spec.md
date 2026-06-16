# RoughQuant — energy-concentrating mixed-precision weight quant (build spec)

**Status:** design (2026-06-17). Derived from ResQ (2412.14363), adapted to
hipfire (weight-only, GQA, multi-tier, fp32 super-bin) + the "roughquant" lever.

## Lineage / what's new

- **ResQ** (2412.14363): PCA-rotate into the eigenbasis of the activation
  covariance `XXᵀ`; keep the top-`r` high-variance subspace at high precision
  (8-bit), the rest at 4-bit; random rotation *within* each subspace to
  Gaussianize. Proven error-optimal split. Beats SpinQuant −33% wikitext PPL.
- **ROSAQ** (2506.13472): same PCA-salient-channel idea, FP16 top / INT3-4 rest.
- **Super-weight** (2411.07191): a tiny set of weights gate the model; protect
  them or PPL explodes 3+ orders of magnitude. Sets the *floor* on the top bin.
- **CMPQ** (2410.13056): select the protected set by **quant-error impact**, not
  raw magnitude.
- **NEW here (RoughQuant):** the *opposite of SmoothQuant*. SmoothQuant migrates
  difficulty out of activations into weights to make uniform quant work.
  RoughQuant deliberately **concentrates energy INTO the high-precision subspace**
  (extract the smooth/dominant bulk into a tiny fp32 low-rank part) so the
  *residual* is near-zero-variance and crushes to 1-2 bits. Accept **fp32** (not
  8-bit) on the protected subspace because it's a fraction of a percent of
  columns — the fp32 cost is far less than the savings from pushing the bulk
  below 4-bit.

## The math (weight-only specialization)

Layer output `Y = X·W`, `X ∈ ℝ^{n×d}` (activations, kept high precision —
weight-only quant), `W ∈ ℝ^{d×d_out}`.

1. **Basis** `U = P · blockdiag(R_1…R_T)`:
   - `P` = eigenvectors of `C = XᵀX` (the per-layer covariance we already collect
     as the LDLQ Hessian), sorted by eigenvalue. Eigenvalues = importance rank.
   - `R_t` = random orthogonal (Hadamard) rotation within tier `t` → Gaussianizes
     each tier (**this is also what makes the QTIP codebook valid per tier** —
     qtip needs Gaussian input; the within-tier rotation provides it).
2. **Multi-tier partition** (generalizes ResQ's binary r): sort the `d` coords by
   eigenvalue, cut into `T` tiers, tier `t` → bit-width `b_t` ∈
   {fp32, bf16, 8, 4, 3, 2, 1, void}. Tier boundaries = the budget knob (swept).
   - `void` tier = structured prune of the dead tail (lowest eigenvalues).
   - Top tier(s) = fp32/bf16 super-bin; floor = the super-weight channels.
3. **RoughQuant concentration lever:** push energy into the top tier so the low
   tiers' residual variance shrinks → enables 1-2 bit. Two forms to test:
   - (a) **rank/size:** grow the fp32 low-rank part until the residual is flat.
   - (b) **per-tier scale:** scale top-tier coords up / bulk down (folds into the
     projection); harmless on fp32, finer effective grid on the bulk.
   - **Economics (the thesis to verify):** avg-bits ≈ `(Σ_t n_t·b_t)/d`. e.g.
     top `d/64` (1.5%) @ fp32 = 0.5 avg-bit; bulk 98.5% @ 2-bit = 1.97 → **~2.5
     avg-bits with an fp32-protected dominant-energy subspace**, vs 4-bit uniform.
     Worth it iff PPL at ~2.5 avg-bits ≈ 4-bit-uniform PPL. (ResQ's d/8 @ 8-bit ≈
     4.5 avg-bits is a known-good but less aggressive anchor.)
4. **Orthogonality wins (from ResQ):** cross-tier products vanish → runtime is
   `T` *same-precision* partial GEMMs accumulated (fp32 top is a tiny dense GEMM;
   bulk is 1-3 bit), OR a single fused mixed-bit kernel. Numerically invariant at
   inf precision.

## Folding (runtime cost) — the make-or-break, solved by ResQ

- **`U_A` at block boundaries** folds into `o_proj`/`down_proj` (right-mult) +
  `q/k/v/gate/up` (`U_Aᵀ` left-mult) + embed/head → **zero runtime cost**.
- **`U_D` for down_proj** can't fold past the activation fn → runtime **Hadamard**
  (hipfire already has FWHT kernels; cheap). down_proj kept uniform low-bit.
- GQA: the weight projections (`U_A`) are arch-agnostic. Head-wise K/V projections
  (ResQ `U_B/U_C`) are a KV concern → defer to Phase D (separate from weights).

## hipfire reuse

- **Calibration / `C = XᵀX`:** the per-layer Hessian the LDLQ path already
  collects (HFHS sidecar). One artifact → `P` (rotation), eigenvalues (importance
  bins), and LDLQ feedback.
- **Within-tier Hadamard:** existing `cpu_fwht_256` + FWHT GEMV machinery.
- **Low-tier format:** QTIP-3/2 trellis (the within-tier rotation Gaussianizes →
  codebook valid). Mid tiers: MQ4/Q8. Top: fp32/bf16 dense.
- **Bins concept:** generalizes the existing `QuantLevel` enum from per-tensor to
  per-column-group.

## De-risk order (sim before kernels — repo methodology)

1. **CPU sim, no rotation yet:** real layer + collected `C`. Rank channels by
   `diag` proxy, protect top-k at fp32/bf16, quantize rest, dequant → PPL via the
   normal forward. Sweep k. Confirms the super-weight-protection thesis cheaply.
2. **Add PCA rotation:** eigendecompose `C`, rotate, re-bin by eigenvalue,
   within-tier Hadamard, quantize per tier, dequant → PPL. Sweep tier count +
   boundaries + top-tier fp32-vs-bf16 + roughquant concentration strength. Find
   the avg-bit/PPL frontier. **Gate:** does ~2.5 avg-bit ≈ 4-bit-uniform PPL?
3. **Only if the frontier wins:** build the fold (offline, into adjacent weights)
   + the runtime down_proj Hadamard + the per-tier GEMV (multi-launch or fused).
   Coherence + fresh-probe perf.

## Open questions to resolve in the sim

- Top-bin: **fp32 vs bf16** — does fp32 buy enough residual-flattening to drop the
  bulk a further bit? (roughquant's core claim)
- Tier count on small models (launch cost) vs big models (where it amortizes).
- PCA basis is per-activation (shared by `q/k/v/gate/up`); per-tensor refinement
  (different bins per weight sharing one rotation) — needed or not?
- Super-weights as a sparse scalar exception bin (SpQR-style) vs a full fp32
  column — which is cheaper for the lone scalars.
