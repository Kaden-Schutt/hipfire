# Methodology — AWQ+GPTQ Sub-0.10 KLD Investigation

## The math

### One-shot AWQ+GPTQ (v3 winner)

**Runtime invariant**: hipfire AWQ-aware kernels apply `x/s` element-wise before the FWHT-256 rotation + MQ4 GEMV, so:

```
(W·diag(s)) · FWHT(x/s) = W·x        (FWHT is orthogonal, self-inverse cancels in the GEMM)
```

**Quantize-time pipeline**:
1. **Imatrix**: collect `Σ_token act²[j]` per input channel from BF16 forward pass over calibration corpus
2. **AWQ scales** (paper formula): `s[j] = RMS_act[j]^α`, geo-mean normalized to 1.0
3. **Source pre-scale**: `W' = W · diag(s)` per AWQ-eligible tensor (input-side projections only on F1: q/k/v/gate/up/in_proj_*)
4. **FWHT-256 rotation**: applied to `W'` rows over input dim → `W'' = W'·H^T`
5. **MQ4 quantize**: per-256-group min-max → `Q(W'')`
6. **Sidecar emit**: 1D F16 tensor `<name>.awq_scale.weight` with `s` values

**AWQ-aware GPTQ pass** (the v3 contribution):
1. Load AWQ-base model (step 6 output)
2. Load AWQ scales `s` from sidecars
3. Transform Hessian: `H_z = diag(1/s) · H_x · diag(1/s)` (FWHT is orthogonal, no extra term needed)
4. Pre-scale source weights: `values = W · diag(s)` (the missing step from path 1 / path 2 v2)
5. GPTQ solve: per-column quantize+propagate, minimizing `||values − Q(values)||² · H_z`
6. Write output `.hfq` with corrected weights + preserved AWQ sidecars

**Why both transforms are necessary**:
- Hessian-only transform (path 2 v2) fails: GPTQ targets raw `W` but runtime expects `W·s` weights → per-channel corruption
- Source-only transform (no Hessian): GPTQ minimizes `||W·s − Q(W·s)||²·||x||²` but runtime applies `x/s` → wrong objective, sub-optimal
- Both together = optimization in runtime basis, validated empirically at KLD 0.1257

### Iterative AWQ+GPTQ (tier 3, in progress)

The one-shot pipeline above derives `s⁽⁰⁾` from BF16-collected imatrix. But the final quantized model sees activations that differ slightly from BF16's because upstream layers are also quantized. The Krasnoselskii-Mann alternating optimization captures this:

```
Round 0: s⁽⁰⁾ = AWQ(H_BF16, α)
         Q⁽⁰⁾ = AWQ-aware-GPTQ(W·s⁽⁰⁾, H_BF16_transformed)
Round k: H⁽ᵏ⁾ = imatrix(model = Q⁽ᵏ⁻¹⁾)             # forward pass through quantized model
         s_raw = AWQ(H⁽ᵏ⁾, α)
         s⁽ᵏ⁾ = (1−β)·s⁽ᵏ⁻¹⁾ + β·s_raw            # KM damping; β ∈ [0.3, 0.7]
         Q⁽ᵏ⁾ = AWQ-aware-GPTQ(W·s⁽ᵏ⁾, H⁽ᵏ⁾_transformed)
Stop:    ‖s⁽ᵏ⁾ − s⁽ᵏ⁻¹⁾‖_2 / ‖s⁽ᵏ⁻¹⁾‖_2 < ε  (default 0.01)
         or k ≥ k_max (default 6)
```

Fixed point: `s* = argmin_s ‖W·X(s,Q*) − Q*(W·s)·diag(1/s)·X(s,Q*)‖²`

## The investigation arc (what was tried, in order)

### Phase 1: harness validation

**Goal**: confirm eval_hipfire + the kldref harness produce numbers that match REPORT-9b.md's prior reference data on origin/master HEAD post-PR-#273.

Cross-checks (c512 q8 prefill on gfx1201, all four match REPORT-9b.md to within 0.001 KLD):
- `flat-mq4` = 0.3215 (REPORT: 0.3215) ✓
- `kmd2-q8conv1d` = 0.1605 (REPORT: 0.1605) ✓
- `cand151-gptq-all-compatible` = 0.1565 (REPORT: 0.1565) ✓
- `pr266-repro-q8` (AWQ alone, F1) = 0.1867 (new) — establishes AWQ baseline

Engine drift is **zero across 41 master commits** between 2026-05-15 (REPORT) and 2026-05-18. Safe to compare new numbers against REPORT historical numbers.

### Phase 2: naive AWQ+GPTQ stack — catastrophic failure (path 1)

Approach: run `mq4_masked_calib.py quantize --method gptq` on the F1-AWQ-base model with the existing raw-x Hessian. Expectation: small lift from GPTQ on top of AWQ.

**Result: KLD 1.7634, PPL 49.9 — 5.5× WORSE than flat-mq4.**

Root cause: AWQ pre-scaled weights `W' = W·s` are stored in the file. GPTQ reads them, but minimizes `||W − Q_new(W)||² · H_x` (where W is the BF16 source, NOT the pre-scaled). At runtime, the kernel applies `x/s` divide expecting weights to be `W·s`, but GPTQ "corrected" them toward `W`. Mismatch → per-channel corruption.

REPORT-9b.md had warned about this: *"A real stack would need the GPTQ solver to use AWQ-clipped values or AWQ scale/code initialization directly."* Confirmed empirically.

### Phase 3: AWQ-aware Hessian-only — still broken (path 2 v2)

Approach: extend the GPTQ pipeline with `--awq-aware-hessian PATH` flag (Codex agent, commit `c5825848` on `awq-aware-hessian` branch). Transform the Hessian: `H_z = diag(1/s) · H · diag(1/s)`. The math: GPTQ should minimize over the runtime activation `z = x/s`, and `H_z = E[z^T z] = diag(1/s) · H_x · diag(1/s)`.

**Result: KLD 1.7531, PPL 49.1 — statistically identical to path 1. Still broken.**

Root cause: `values` (source weights passed to GPTQ) was still raw `W`, not `W·s`. GPTQ minimized `||W − Q_new(W)||² · H_z` — the Hessian was right, but the optimization target was wrong. Runtime then computed `Q_new(W) · (x/s)`, off by factor `s` per channel.

### Phase 4: AWQ-aware GPTQ — v3 winner (path 2 v3)

Approach: add the missing source pre-scale. Commit `f0bcfabd` on `awq-aware-hessian` adds `values *= scale[None, :]` so GPTQ now solves the runtime objective: `min ||W·s − Q(W·s)||² · H_z`.

**Result: KLD 0.1257, PPL 9.31, p99 13.7. Same 5.0 GB as flat-mq4. New 9B Pareto winner.**

Beats prior best `cand151-gptq-all-compatible` (0.1565 KLD, 5.56 GB) by **−20% KLD at 90% of size**.

### Phase 5: F2 stack — Pareto-dominated by F1

Approach: rebase onto post-PR-#273 master to pick up F2 AWQ (extends scope from input-side to also include o_proj/down_proj/out_proj). Run AWQ-aware GPTQ on the F2 base at α=0.5.

**Result: KLD 0.1386, PPL 9.50.** Worse than v3 on BOTH axes. α=0.55 (PR #273's sweet spot for PPL) yielded KLD 0.1396 / PPL 9.32 — recovered PPL but KLD got marginally worse.

Hypothesis: the 64 new output-side sidecars aren't compatible with AWQ-aware GPTQ source pre-scaling without further math — possibly per-output-axis instead of per-input-axis convention, or the AWQ-aware kernels have different normalization. Not investigated further since F1 v3 strictly Pareto-dominates.

### Phase 6: AutoAWQ formula — catastrophic failure

Approach: extend hipfire-quantize with `--awq-formula autoawq` adding the weight-magnitude term: `s = RMS_act^α · RMS_w^(1-α)` (PR #266 design doc deferred this as "small-effect, can be added later"). Commit `85f6d055` on `awq-aware-hessian`.

**Result at α=0.5: KLD 1.8257, PPL 44.08 — catastrophic.**

Root cause: the weight-magnitude term widens the per-input-channel dynamic range of `s` (because RMS_w varies more than RMS_act in trained networks). After FWHT-256 mixing + MQ4 G=256 quantization, the wider scale distribution stretches per-group min-max ranges past what GPTQ can correct. PR #266 design doc warning was explicit: *"keeps the post-AWQ-scaled weight tensor's overall magnitude in the same range as the input — important for the downstream MQ4 min-max scale fitter not to suddenly compress/expand its dynamic range based on alpha."*

The AutoAWQ formula assumes scalar-precision GEMM (no rotation, no group quant), and doesn't compose with hipfire's FWHT+MQ4 stack. Different `α` would need fundamentally different math.

### Phase 7: alpha sensitivity sweep — global α is exhausted

Approach: 5-point sweep of paper formula α ∈ {no-AWQ, 0.3, 0.5, 0.7, 1.0} + AWQ-aware GPTQ, parallelized on 4 R9700 GPUs (one variant per GPU at c512 q8 prefill).

**Result**: U-shaped on KLD with **v3 (α=0.5) at the minimum**:
- α=∞ (no AWQ): KLD 0.2686, PPL 8.91
- α=0.3: KLD 0.1514, PPL 9.34
- **α=0.5 (v3): KLD 0.1257, PPL 9.31** ← minimum
- α=0.7: KLD 0.1663, PPL 8.97
- α=1.0: KLD 0.2032, PPL 8.89

PPL is monotonically decreasing toward higher α. The KLD-PPL inversion is real. Global α tuning produces no further KLD lift; **the next lever must be per-tensor**.

See `alpha-sweep.md` for the full data + interpretation.

### Phase 8: iterative AWQ+GPTQ (in progress)

Approach: alternate AWQ scale derivation and GPTQ solve, refreshing the imatrix from the partial-quantized model's outputs each round. The fixed point is self-consistent — AWQ scales match the actual runtime activation distribution post-GPTQ.

Cost per round: ~5-10 min Hessian collection (4 GPUs in parallel) + ~4 min GPTQ (1 GPU) + ~24 min bench (1 GPU, optional per round). 4 rounds with bench-each ≈ 2-3h total.

Expected lift: 5-20% from literature norms for iterative quantization. v3's 0.1257 × 0.85 = 0.107; × 0.80 = 0.101; × 0.75 = 0.094. **Sub-0.10 lands if iteration captures ≥18% lift**, which is plausible but not guaranteed.

Codex implementation on branch `iterative-awq-gptq` (commit `f286bade`). Parity tests pass on both k9lin (CPU) and hiptrx (hipfire-rocm env):
- Identity round = byte-identical to one-shot
- Damping=0 = no scale evolution
- Synthetic 3-round damped FPI: deltas halve per round (0.0796 → 0.0393 → 0.0195) ✓

## Why this approach is sound

1. **Math correctness verified empirically**: v3's KLD 0.1257 vs no-AWQ-no-GPTQ KLD 0.2686 → −53% lift from AWQ; vs no-GPTQ-with-AWQ KLD 0.1867 → −33% lift from GPTQ. Both contribute, no double-counting.
2. **Wire format preserved**: 184 sidecars (~1.4 MB) + standard MQ4G256 weights = 5.0 GB on disk, byte-aligned with flat MQ4. Inference perf within ~0% of flat (one extra fused element-wise divide per AWQ-eligible weight, BW-free).
3. **Reproducible**: every step uses pinned input (Hessian, imatrix, calibration corpus, BF16 reference) and exact commands recorded in `repro-recipe.md`.
4. **Tested cross-machine**: parity gates pass on both k9lin (Codex's dev machine) and hiptrx (the bench machine).

## What we ruled out (and why)

| Lever | Why dropped |
|---|---|
| K-map promotions (MQ6 for v_proj/down_proj) | User constraint — keep flat-MQ4 wire format. K-map adds 1+ GB disk. |
| AutoAWQ formula (weight-magnitude term) | Tested — KLD 1.83, catastrophic. Doesn't compose with FWHT+MQ4 G=256 |
| F2 expansion (output-side AWQ) | Tested at α=0.5 and α=0.55 — Pareto-dominated by F1 v3 on both axes |
| Q8 conv1d→Q8 input/MLP weights | User constraint — implies bigger disk, would also nullify the AWQ contribution |
| Different GPTQ damping / refit_iters | Not tried; small expected lever; iterative pipeline is more general |
| Larger imatrix Hessian (c64 → c256) | Not tried; iterative pipeline subsumes by re-collecting per round |

## What this investigation does NOT claim

- Sub-0.10 is guaranteed — iterative rounds may converge above 0.10 if v3 is already near the AWQ+GPTQ local optimum
- These numbers transfer to non-9B Qwen models (per-model retuning may be needed)
- Decode tok/s is identical to flat-mq4 (not measured directly; expected to match within ~0% based on kernel design but should be confirmed if it's a ship axis)

## Phase 8a: iterative pipeline (run-001) — AWQ scope mismatch discovered

**Approach**: spawned Codex to implement an `iterate` subcommand on `mq4_masked_calib.py` chaining the KM steps. Branch `iterative-awq-gptq` at commit `f286bade`. Parity tests passed in synthetic (identity-round byte-match, damping=0 stability, 3-round damped FPI deltas halving).

**Real-data run** with `--awq-alpha 0.5 --damping 0.5 --max-rounds 4 --bench-each-round`:

| Round | KLD c512 | PPL | scale-delta vs prev | Elapsed | Notes |
|---:|---:|---:|---:|---:|---|
| 0 | 0.6999 | 11.66 | — | 1810 s | should match v3's 0.1257 but doesn't |
| 1 | 0.4521 | 8.75 | 0.039 (3.9%) | 1945 s | iterating but converging to wrong fixed point |
| 2-3 | aborted | — | — | — | killed once scope bug diagnosed |

**Root cause**: Codex's `selected_iterate_targets` selected ALL 91 packable_flat_mq4 tensors from the mask as the AWQ scope — **including lm_head and other output-side projections**. On master without PR-#273 F2 kernels, those tensors have no runtime `x/s` inverse (they don't pass through the AWQ-aware fused rotate kernel). Applying AWQ pre-scaling produces `(W·s)·x ≠ W·x` per-channel corruption — same failure-mode signature as Path 1.

v3 keeps AWQ and GPTQ scopes separate:
- AWQ scope: 184 tensors (Rust quantizer's `awq_eligible(name)` F1 whitelist — suffix-match on q/k/v_proj, gate/up_proj, in_proj_*, router, mlp.gate)
- GPTQ scope: 67 tensors (a different `mask.json` produced by `mq4_masked_calib.py mask` step)

Codex's iterate conflated these → 91 sidecars total, wrong subset, corruption on the misaligned ones.

**Fix** (commit `63ba8aa1`): added `_is_awq_eligible_f1` in `scripts/mq4_masked_calib.py` mirroring Rust's `awq_eligible()` F1 scope exactly. `selected_iterate_targets` now filters by this predicate, so AWQ scales are only computed for input-side projections.

**Lesson**: when porting AWQ-eligibility logic across languages, mirror the source-of-truth function exactly (suffix-match list). Don't infer eligibility from adjacent metadata like `packable_flat_mq4` — that's a GPTQ-target marker, not an AWQ-eligibility marker.

## Phase 8b: iterative pipeline (run-002) — F1-scope-fixed

Same command as run-001 but with the F1-scope fix at commit `63ba8aa1`. Expected round 0 KLD: close to v3's 0.1257 (some residual difference from in-process imatrix collection vs unsloth's pre-collected imatrix).

(Results populating to [results.md](results.md) as monitor fires.)
