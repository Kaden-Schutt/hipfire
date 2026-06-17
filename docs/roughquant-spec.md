# RoughQuant — energy-concentrating mixed-precision weight quant (build spec)

**Status:** **POSITIVE at low bits (2026-06-17, phase2h) — reverses the earlier
negative.** Foldable Q8-protection of the shared ~75-channel outlier set + mq4
bulk beats uniform by ~25% KLD at ~4.4–4.5 avg-bits (win shrinks toward higher
bits). Reached only after user skepticism uncovered 4 compounding artifacts
(zeroing bug, energy-aggregation error, bf16-protection wasting half its bits,
PPL pointwise noise). Literature-consistent (AWQ/super-weight). SIM result; needs
packed format + offline fold + coherence + cross-model before shipping. See
`docs/roughquant/phase2h-foldable-win.md`. (Earlier phases below retained for the
record; their "no foldable win" verdicts are SUPERSEDED by phase2g/2h.)

_Prior status line (superseded):_ **CONCLUDED — NOT DEPLOYABLE on Qwen3.5-0.8B (2026-06-17).** Sim
phases 0–2 + de-risks A/B done — see `docs/roughquant/phase{0,1,2}.md`. Two
independent failures: (1) the headline thesis (≈2.5 avg-bit ≈ 4-bit PPL) is
FALSIFIED — 2-bit bulk non-viable (2.55 bit → PPL 47.85 vs mq4 29.08); (2) the
real sub-4-bit quality edge (per-weight PCA: 28.28 < mq4 29.08, iso-bit, de-risk
A) EVAPORATES under the folding constraint (de-risk B: foldable shared residual
rotation → 30.68, worse than mq4). The win existed only with per-weight dense
rotations that cannot fold for free; the deployable form is strictly dominated by
mq4. **The foldable design space is now fully swept (phases 2c/2d): shared
rotation, permutation (read), and channel-consistent read+write protection all
land at ~29.4–30.7 PPL, all dominated by mq4** — the write-side lever genuinely
helps (~1 PPL) but decorrelation is the missing ingredient and it doesn't fold.
**CORRECTION (phase2e):** a non-monotonic bug (zero-before-quant) inflated the
earlier "PCA beats mq4 (27.90/28.28)" headline; with the fixed monotonic
quantizer PCA b3 f0.03 = 29.37, which does NOT beat mq4 (29.07). The energy-CDF
analysis (`scripts/roughquant_energy_cdf.py`, see `phase2e`) shows why: raw/
foldable-basis energy is ~uniform (top 1% of residual channels = 21%, no knee);
concentration exists ONLY in the un-foldable per-weight eigenbasis (top 1% of
eigenvalues = 42–81%). So NO variant — foldable or not — beats mq4 on 0.8B.
Phase 3 NOT pursued. **CORRECTION (phase2g):** the "energy is spread / no foldable
concentration" reading was WRONG — an aggregation artifact + PPL noise. Per-tensor
outliers are strong (max/med up to 283×, kurtosis 391) and SHARED across layers
(~75 persistent residual dims; union of top-2% over 48 input-points = 75/1024).
Foldable shared-outlier protection HALVES mq4's KLD (0.162→0.084 at +0.6 bits) —
literature-consistent. BUT uniform mq6 still dominates at iso-bits (0.0084 vs
protect-15% 0.057 at ~6b) because mq4's per-256 FWHT already does incoherence
processing (the papers' protection wins are over naive RTN) and the bulk carries
~1/3 of the error (only uniform-bit-increase fixes it). Deployment verdict
unchanged (uniform ≥ protection on a strong baseline); the mechanism is now
correctly understood & reconciled with the literature. Untested: Q8 (vs bf16)
protection, persistence-based selection, cross-model 7B/9B. Tooling: KLD now the
default metric (`perplexity --dump-ref/--kld-ref`); `scripts/roughquant_energy_cdf.py`.
Derived from ResQ (2412.14363), adapted to hipfire (weight-only, GQA, multi-tier,
fp32 super-bin) + the "roughquant" lever.

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
  → **Moot so far:** the bulk could not drop to 2-bit on the 0.8B at all (2-bit
  PPL ≥ 35 even with rotation+protection), so the further-bit-drop the fp32 bin
  was meant to enable did not materialize. fp32-vs-bf16 untested because the
  bf16 sim already keeps the protected subspace effectively exact for PPL.
- Tier count on small models (launch cost) vs big models (where it amortizes).
  → **Untested across models** (0.8B only). The 0.8B verdict: a binary split
  (protect + single bulk tier) already shows the win is bounded at ~3.5 bits;
  multi-tier unlikely to help here. May differ on big models — deferred.
- PCA basis is per-activation (shared by `q/k/v/gate/up`); per-tensor refinement
  (different bins per weight sharing one rotation) — needed or not?
  → **THE pivotal open question for deployability.** The sim used a per-weight
  dense PCA basis, which does NOT fold for free. Whether a *shared* per-activation
  basis (foldable, ResQ-style) preserves the PPL win is de-risk B and must be
  settled before any kernel work.
- Super-weights as a sparse scalar exception bin (SpQR-style) vs a full fp32
  column — which is cheaper for the lone scalars.
  → Untested; the column-granular protection used here is the full-column form.

## Sim verdict (2026-06-17)

- **Phase 0** (`docs/roughquant/phase0.md`): baselines — bf16 26.17, mq4
  (4-bit gate) 29.08, QTIP-3-LDLQ 31.42.
- **Phase 1** (`phase1.md`): top-k column protection (no rotation) — super-weight
  premise confirmed (3-bit 3978→105 at 1.5% protect), but no un-rotated config
  beats mq4; rotation is the competitiveness lever.
- **Phase 2** (`phase2.md`): PCA rotation + protection — headline 2.5-bit gate
  FALSIFIED (2.55 bit → 47.85); best `b3 f0.03` = 27.90 at ~3.5 bit beats mq4 and
  beats iso-bit QTIP-3-LDLQ by 11%. Win confounded by bf16 embed/lm_head and
  contingent on foldable shared rotation. **De-risks gating Phase 3:** (A) iso-bit
  embed comparison; (B) shared/foldable rotation preserves the win. Phase 3
  kernels NOT to be built until A+B hold + human go/no-go.
