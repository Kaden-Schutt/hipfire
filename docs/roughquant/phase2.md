# RoughQuant — Phase 2: PCA rotation + frontier sweep

**VERDICT: HEADLINE THESIS FALSIFIED, modest sub-4-bit win exists but is
CONFOUNDED and not yet deployable. → Do NOT build Phase 3 kernels yet; resolve
two cheap de-risks first (see NEXT-STEPS).**

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

## STATUS: stopped here for a human go/no-go (2026-06-17)

De-risk A passed → the mechanism win is real and iso-bit (~0.7 bit at iso-PPL,
~2.8% better than mq4; +11% over QTIP-3-LDLQ). De-risk B (foldable shared
rotation) is unresolved and is the deployability make-or-break. Building it is a
substantial redesign (global/shared residual-stream rotation + fold + packed
per-tier format + kernels) for a **modest ~0.7-bit payoff**. Per the project
methodology (don't build large speculative kernels for a contingent win), this is
left for a human decision rather than an unattended build.

## NEXT-STEPS (gating Phase 3)

1. **De-risk A — DONE, PASSED.** Iso-bit (Q8 embed): b3 f0.03 = 28.28 < mq4 29.08.
2. **De-risk B (the blocker, a redesign): shared/foldable rotation.** Collect ONE
   residual-stream Hessian per block boundary (or one global), apply a single
   shared rotation foldable into embed + all o_proj/down_proj outputs + lm_head,
   and confirm b3 f0.03's PPL edge survives the shared constraint. If shared-P
   loses the edge → win not deployable, STOP. If it holds → Phase 3 is justified.
3. **Only if B holds:** offline fold + down_proj runtime FWHT + real packed
   per-tier format (not bf16 sim) + coherence + fresh-probe perf on a perf-class
   model. Large effort; confirm the payoff is worth it first.
4. **Cross-model check (orthogonal):** the 2-bit failure and bounded ~3.5-bit win
   are on a 0.8B; bigger models have more redundancy and may push the frontier
   lower. Worth re-running phase 2 on a 7B/9B before concluding 2-bit is dead.

## Artifacts

- Code: `crates/hipfire-quantize/src/roughquant.rs` (PCA basis + rotate via
  faer), `main.rs` (`qtip_simquant_protected` + `roughquant2-sim` post-pass),
  `scripts/roughquant2_sweep.sh`.
- Generated `.hfq` transient (quantize→PPL→delete). Fixtures as Phase 0.
