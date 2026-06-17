# RoughQuant — Phase 2: PCA rotation + frontier sweep

**FINAL VERDICT (post de-risk A+B): NOT DEPLOYABLE on Qwen3.5-0.8B. STOP.**
The headline 2.5-bit thesis is falsified; the modest sub-4-bit quality edge is
real and iso-bit (de-risk A) but **evaporates under the folding constraint**
(de-risk B): the deployable (foldable shared-rotation) form scores PPL 30.68 —
**worse than mq4's 29.08** — while also incurring runtime rotation cost for
o_proj/down_proj. The win existed only with per-weight dense rotations that
cannot fold for free. Do NOT build Phase 3. (Cross-model recheck on a 7B/9B is
the only remaining avenue and is speculative — see NEXT-STEPS.)

---

_Original phase-2 verdict (pre de-risk B), kept for the record:_ headline thesis
falsified, modest sub-4-bit win exists but confounded and not-yet-deployable.

- The spec's headline gate — *"does ~2.5 avg-bit ≈ 4-bit-uniform PPL?"* — **fails**
  on Qwen3.5-0.8B: at 2.55 avg-bits RoughQuant gives PPL 47.85 vs mq4's 29.08.
  2-bit QTIP bulk is too lossy on this model even with PCA rotation + protection.
- A **real but modest** win exists at ~3.5 avg-bits: `b3 f0.03` = **27.90** beats
  mq4 (**29.08**, ~4.25 bits) — ~0.7 bit cheaper at slightly better PPL.
- **BUT** two confounds make this not-yet-a-win (below). Both are cheap to settle
  and must be settled before any Phase 3 kernel/format work.

## Method

`hipfire-quantize --format roughquant2-sim` (added this phase):
1. Per 2D weight, eigendecompose the activation Hessian `C=XᵀX` (sidecar) →
   basis `P` (columns = input directions, sorted by eigenvalue desc).
2. Rotate `W̃ = W·P` (faer matmul). Columns of W̃ are now energy-ranked.
3. Protect the leading `protect_frac·k` columns (highest energy) at full
   precision — saved & zeroed before QTIP so they don't inflate group scales,
   restored after (per-column granularity; `qtip_simquant_protected`).
4. QTIP-trellis the bulk (the per-256 FWHT supplies the within-tier Hadamard +
   low-bit format).
5. Inverse-rotate `W_q = W̃_q·Pᵀ`, bake to bf16. Normal forward → PPL.

Round-trip identity check (protect_frac=1.0, no quant): PPL 26.61 vs bf16 26.17
(+1.7%, from f32-matmul + bf16 storage) — confirms the rotation math is correct.

Corpus/ctx as Phase 0/1. Sweep: `scripts/roughquant2_sweep.sh`. Env:
`HIPFIRE_RQ2_{PROTECT_FRAC,BULK_BITS,DAMP}`. Cost ~10 min/config (QTIP
beam-encode over 752M params dominates).

## Results (vs mq4 gate 29.08, bf16 floor 26.17; PPL is deterministic)

| bulk_bits | protect | avg-bits (est) | PPL | vs mq4 |
|---|---|---|---|---|
| 3 | 0.0   | 3.13 | 33.63 | +16% |
| 3 | 0.015 | 3.32 | 31.53 | +8%  |
| **3 | 0.03**  | **3.52** | **27.90** | **−4% ✓** |
| 3 | 0.06  | 3.90 | 28.65 | −1% ✓ |
| 2 | 0.0   | 2.13 | 397.9 | ✗ |
| 2 | 0.015 | 2.34 | 53.26 | ✗ |
| 2 | 0.03  | 2.55 | 47.85 | ✗ (headline-gate point) |
| 2 | 0.06  | 2.96 | 42.64 | ✗ |
| 2 | 0.12  | 3.79 | 35.48 | +22% |

Reference (no PCA): qtip3sim-plain 34.41, qtip3sim-ldlq 31.42. PCA rotation +
protection (b3 f0.03 = 27.90) beats LDLQ-QTIP-3 (31.42) by 11% — rotation into
the eigenbasis + protecting the top-energy subspace genuinely helps.

## Reading

1. **Bulk bit-width dominates protection.** b2 f0.12 (3.79 bits, 35.48) is worse
   than b3 f0.03 (3.52 bits, 27.90). Spending bits on a 3-bit bulk beats spending
   them on protecting more columns of a 2-bit bulk. The "crush the bulk to 1-2
   bits" half of the RoughQuant thesis does **not** hold on this model.
2. **Protection has a sweet spot, then hurts (on avg-bits).** b3: best PPL at
   f0.03 (27.90); f0.06 is slightly worse PPL *and* more bits. Beyond ~3% the
   marginal protected column isn't worth its 16 bits.
3. **The energy-concentration premise is real but bounded.** PCA + top-subspace
   protection beats both plain and LDLQ QTIP-3 — the eigenbasis is the right
   frame and the top columns are worth protecting. It just doesn't extend down
   to 2-bit on a 0.8B.

## Two confounds — one settled, one is the blocker

1. **Embed/lm_head precision — SETTLED (de-risk A PASSED).** The `*-sim`
   post-pass leaves embed/lm_head at bf16; mq4 uses Q8 (~20% of params on a tied
   0.8B). Re-ran `b3 f0.03` with `HIPFIRE_RQ2_Q8_EMBED=1` (8-bit per-256-group
   uniform on embed/lm_head, matching mq4): **PPL 28.28** (vs 27.90 bf16-embed).
   The win shrank slightly but **still beats mq4 (29.08) by ~2.8% at ~0.7 fewer
   bits**. The win is real and iso-bit — NOT an embed artifact.
2. **The PCA rotation does not fold for free (the deployability blocker).** The win
   assumes the rotation is zero-cost at runtime. mq4's FWHT rotation IS free
   (folded into the GEMV that rotates x). RoughQuant uses a **dense per-weight
   k×k PCA basis P**. A dense per-weight rotation applied at runtime is a k×k
   matmul per weight per token — catastrophic, erasing any bit-saving. It is
   free ONLY if P is **shared across all weights reading one residual-stream
   point** and folds into the producing weight (ResQ's U_A). The sim used a
   **dense per-weight** P (186 distinct rotations) — each weight's own optimal
   basis. That does NOT fold: applying it at runtime is a [k×k] matvec per weight
   per token (~100M MACs/token across the model), eroding the perf benefit of the
   ~0.7 saved bits. mq4's FWHT rotation is free precisely because it is a fixed
   structured (Hadamard) transform baked into the GEMV. **The open question:** does
   a single SHARED rotation of the residual stream (foldable into embed +
   o_proj/down_proj outputs + lm_head, ResQ-style) preserve the win? A shared
   rotation is far more constrained than 186 per-weight ones and will capture
   less benefit — this is the research crux, and a redesign, not a quick sweep.

## De-risk B — DONE, FAILED (the deployability verdict)

Foldability analysis first (Hessian inspection, no GPU):
- Same-input weights have **identical** Hessians (cos = 1.000000 for
  `in_proj_{qkv,z,b,a}` and for `gate/up`; unrelated pairs cos ≈ 0.034). So
  within-input-group rotation sharing is **already free** and the per-weight
  phase-2 result already reflects it — the win was NOT inflated by per-q/k/v
  rotations.
- Rotations can only be shared among **same-input, same-k** weights. By k:
  k=1024 = all d_model residual readers (in_proj_*, gate/up, q/k/v) → one
  foldable residual rotation; k=2048 (o_proj/out_proj) and k=3584 (down_proj)
  read internal activations → need a runtime rotation regardless.

Experiment (`HIPFIRE_RQ2_SHARE_RESID=1`): force all 138 k=1024 weights onto ONE
global rotation aggregated from their summed Hessians (the foldable ResQ-U_A
design); keep o_proj/down_proj per-weight. `b3 f0.03`, Q8 embed:

| rotation | avg-bits | PPL | vs mq4 (29.08) |
|---|---|---|---|
| per-weight (NOT foldable) | ~3.5 | 28.28 | −2.8% ✓ |
| **shared residual (foldable)** | ~3.5 | **30.68** | **+5.5% ✗** |

**The win evaporates under folding.** The single shared residual rotation is
suboptimal for every individual input-point (attn-input vs mlp-input vs the
evolving cross-layer residual have different covariance), and that 2.4-PPL
penalty is larger than the entire margin over mq4. The deployable form (30.68) is
strictly dominated by mq4 (29.08) — worse PPL **and** still needing runtime
rotations for o_proj/down_proj. The phase-2 edge was an artifact of per-weight
dense rotations that cannot fold for free.

## CONCLUSION: RoughQuant not deployable on Qwen3.5-0.8B — STOP

Two independent failures on this model: (1) 2-bit bulk non-viable (headline
thesis), (2) the sub-4-bit quality edge does not survive the folding constraint.
The deployable form loses to existing mq4. No Phase 3.

## NEXT-STEPS

1. **De-risk A — DONE, PASSED** (iso-bit: 28.28 < 29.08).
2. **De-risk B — DONE, FAILED** (foldable: 30.68 > 29.08). Deployability killed.
3. **Phase 3 — NOT pursued.** Would ship a format strictly worse than mq4.
4. **Only remaining avenue (speculative): cross-model.** Both failures are on a
   0.8B. A 7B/9B has more redundancy — 2-bit might become viable AND the shared-
   rotation penalty might shrink (a bigger residual stream may admit a better
   single rotation). Requires a fresh Hessian (~1–3 h collect via
   `scripts/collect_hessian.py`) + a slower sweep. Given two clean negatives
   here, pursue only if there's an independent reason to expect big models to
   differ qualitatively. Not auto-run.

## Artifacts

- Code: `crates/hipfire-quantize/src/roughquant.rs` (PCA basis + rotate via
  faer), `main.rs` (`qtip_simquant_protected` + `roughquant2-sim` post-pass),
  `scripts/roughquant2_sweep.sh`.
- Generated `.hfq` transient (quantize→PPL→delete). Fixtures as Phase 0.
