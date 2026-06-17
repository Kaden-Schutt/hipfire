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

## Two confounds — settle before ANY Phase 3 work

1. **Embed/lm_head precision is unequal (inflates the win).** The `*-sim`
   post-pass leaves embed/lm_head at **bf16**; mq4 quantizes them to **Q8**. On a
   0.8B with tied embeddings, that tensor is ~20% of params getting a free
   precision boost in RoughQuant's favor. The avg-bits estimate (3.52) counts
   only the 2D transformer weights and ignores this. **The 27.90-vs-29.08 win is
   not iso-bit.** Re-run with embed/lm_head Q8 (or mq4 with bf16 embed) for an
   honest comparison — the ~0.7-bit edge may shrink or vanish.
2. **The PCA rotation may not fold for free (could erase the perf win).** The win
   assumes the rotation is zero-cost at runtime. mq4's FWHT rotation IS free
   (folded into the GEMV that rotates x). RoughQuant uses a **dense per-weight
   k×k PCA basis P**. A dense per-weight rotation applied at runtime is a k×k
   matmul per weight per token — catastrophic, erasing any bit-saving. It is
   free ONLY if P is **shared across all weights reading one residual-stream
   point** and folds into the producing weight (ResQ's U_A). q/k/v/gate/up share
   their input, so a shared P is plausible; down_proj needs a runtime Hadamard
   (spec-acknowledged). **This is unvalidated.** Until a shared-and-foldable
   rotation is shown to keep the PPL win, the win is a sim artifact.

## NEXT-STEPS (gating Phase 3)

1. **De-risk A (cheap, ~1 sweep): iso-bit embed.** Add embed/lm_head Q8 to the
   roughquant2-sim path (or a bf16-embed mq4 baseline) and re-measure b3 f0.03.
   If the win survives → continue; if it vanishes → RoughQuant ≈ mq4, STOP.
2. **De-risk B (cheap, ~1 sweep): shared/foldable rotation.** Compute ONE P per
   residual-stream input (shared by q/k/v and by gate/up) instead of per-weight,
   confirm the b3 f0.03 PPL win holds. If shared-P loses the edge, the win is not
   deployable, STOP.
3. **Only if A and B both hold:** design the offline fold (shared U_A into
   adjacent weights + residual skip), the down_proj runtime FWHT, a real packed
   per-tier format (not bf16 sim), then coherence + fresh-probe perf on a
   perf-class model. This is a large, intricate effort for a ~0.7-bit payoff —
   worth a human go/no-go decision, not an unattended build.

## Artifacts

- Code: `crates/hipfire-quantize/src/roughquant.rs` (PCA basis + rotate via
  faer), `main.rs` (`qtip_simquant_protected` + `roughquant2-sim` post-pass),
  `scripts/roughquant2_sweep.sh`.
- Generated `.hfq` transient (quantize→PPL→delete). Fixtures as Phase 0.
