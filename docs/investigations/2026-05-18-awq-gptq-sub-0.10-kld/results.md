# Results — Running KLD Table

All numbers are c512 q8-KV prefill on hiptrx (gfx1201) against `qwen3.5-9b-bf16.kldref.bin`.

## Final table (sorted by KLD)

| Rank | Variant | KLD ± 95% CI | p99 KLD | PPL | NLL mean | Size | Notes |
|---:|---|---:|---:|---:|---:|---:|---|
| 🥇 | **awq-aware-gptq-v3** | **0.1257 ± 0.006** | 13.7 | 9.310 | 2.2313 | 5.00 GB | F1 AWQ + AWQ-aware GPTQ at α=0.5 |
| 2 | awq-aware-gptq-f2 | 0.1386 ± 0.007 | 14.4 | 9.501 | 2.2514 | 5.00 GB | F2 (input+output-side AWQ) at α=0.5 |
| 3 | awq-aware-gptq-f2-a055 | 0.1396 ± 0.007 | 14.5 | 9.323 | 2.2325 | 5.00 GB | F2 at PR-author's PPL sweet spot |
| 4 | f1-a030-gptq | 0.1514 ± 0.008 | 14.3 | 9.340 | 2.2343 | 5.00 GB | F1 α=0.3 — below U-curve min |
| 5 | cand151-gptq-all-compatible | 0.1565 ± 0.008 | 14.2 | 9.197 | 2.2189 | 5.56 GB | prior best, GPTQ-only mixed base |
| 6 | kmd2-q8conv1d | 0.1605 ± 0.008 | 15.2 | 9.173 | 2.2163 | 6.43 GB | K-map MQ4/MQ6 + Q8 conv1d, no AWQ no GPTQ |
| 7 | f1-a070-gptq | 0.1663 ± 0.008 | 15.3 | 8.965 | 2.1933 | 5.00 GB | F1 α=0.7 — above U-curve min |
| 8 | pr266-repro-q8 (AWQ alone) | 0.1867 ± 0.009 | 16.3 | 9.526 | 2.2541 | 5.00 GB | F1 AWQ no-GPTQ |
| 9 | f1-a100-gptq | 0.2032 ± 0.011 | 16.0 | 8.885 | 2.1844 | 5.00 GB | F1 α=1.0 — far above U-curve min |
| 10 | gptq-only-noawq | 0.2686 ± 0.011 | 17.9 | 8.905 | 2.1866 | 5.00 GB | flat-mq4 + GPTQ, no AWQ |
| 11 | flat-mq4 (baseline) | 0.3215 ± 0.012 | 18.7 | 8.715 | 2.1651 | 5.00 GB | no AWQ no GPTQ |
| (floor) | Q8F16 | 0.0186 ± 0.001 | 1.8 | 9.260 | 2.2259 | 9.53 GB | engine ceiling |

## Failed experiments (recorded so we don't repeat)

| Variant | KLD | PPL | Why broken |
|---|---:|---:|---|
| awq-gptq-stack (path 1) | 1.7634 | 49.86 | naive raw-x Hessian, source weights not pre-scaled |
| awq-aware-gptq-stack (path 2 v2) | 1.7531 | 49.14 | Hessian transformed but source still W (off by factor s) |
| autoawq-gptq-a050 | 1.8257 | 44.08 | weight-magnitude term widens dynamic range past MQ4 G=256 |

## Alpha sensitivity curve (paper formula + AWQ-aware GPTQ, c512 q8 prefill)

| α | KLD | PPL | p99 | Δ from v3 |
|---:|---:|---:|---:|---:|
| no AWQ | 0.2686 | 8.91 | 17.9 | +114% |
| 0.30 | 0.1514 | 9.34 | 14.3 | +20% |
| **0.50** | **0.1257** | **9.31** | **13.7** | **minimum** |
| 0.70 | 0.1663 | 8.97 | 15.3 | +32% |
| 1.00 | 0.2032 | 8.89 | 16.0 | +62% |

Convex U-shape on KLD. PPL monotonically decreases toward higher α (KLD-PPL inversion). v3 (α=0.5) is the global optimum.

## Per-tensor metric data (from GPTQ logs)

GPTQ writes per-tensor reconstruction metric (`metric=` field per tensor). v3's metrics show variance across tensors — some are 10× lower error than others. The per-tensor variance is the signal that motivates iterative AWQ+GPTQ.

(Will populate detailed per-tensor table from `round_*/candidate.json` artifacts when iterative completes.)

## Iterative AWQ+GPTQ rounds (in progress)

| Round | KLD c512 | PPL | scale-delta vs prev | Wall time | Notes |
|---:|---:|---:|---:|---:|---|
| 0 (= v3) | 0.1257 | 9.31 | — | 5 min | one-shot baseline |
| 1 | ⏳ | ⏳ | ⏳ | ⏳ | imatrix re-collected on Q⁽⁰⁾ |
| 2 | ⏳ | ⏳ | ⏳ | ⏳ | |
| 3 | ⏳ | ⏳ | ⏳ | ⏳ | |
| 4 | ⏳ | ⏳ | ⏳ | ⏳ | last round (max-rounds=4) |

Damping β=0.5, ε=0.01 stopping criterion. Will populate as monitor fires.

## Iterative pipeline run-001 (deprecated — AWQ scope bug)

**Status**: aborted after round 1. Codex's iterate selected 91 AWQ targets (= GPTQ mask scope) instead of v3's 184 F1-AWQ-eligible scope. Output-side tensors like lm_head got AWQ pre-scaling without runtime inverse → corruption.

| Round | KLD c512 | PPL | scale-delta | Elapsed | Status |
|---:|---:|---:|---:|---:|---|
| 0 | 0.6999 | 11.66 | — | 30 min | broken (lm_head AWQ corruption) |
| 1 | 0.4521 | 8.75 | 3.9% | 32 min | iterating toward wrong fixed point |
| 2-3 | — | — | — | — | aborted |

Recorded as negative result. Fix at commit `63ba8aa1` (`fix(iterate): restrict AWQ scope to F1-eligible tensors`).

## Iterative pipeline run-002 (F1-scope-fixed)

(Populating as monitor fires.)

## Iterative pipeline run-003 (F1-scope + v3 base + corrected module_name)

After three failed setups, the working configuration uses:
- Mask: 184 F1-AWQ-eligible tensors model-wide (`mask-f1-184-v3.json`)
- Base: v3's mq4-awq-pr266-repro (preserves AWQ sidecars where iterate doesn't refresh)
- Module names: `model.layers.N.*` (NOT `model.language_model.layers.N.*` — transformers strips the multimodal infix at the live module level)

| Round | KLD c512 | PPL | scale-delta | Elapsed | Notes |
|---:|---:|---:|---:|---:|---|
| 0 | **0.1798** | 9.36 | — | 33 min | within 43% of v3's 0.1257 — imatrix-source gap |
| 1 | ⏳ | ⏳ | ⏳ | ~32 min | first true KM step |
| 2 | ⏳ | ⏳ | ⏳ | ~32 min | last round (max=3) |

Round 0 ≠ v3 (0.1257) because:
- v3 used **unsloth's pre-published imatrix** (different calibration corpus, different chunking)
- run-003 collects imatrix in-process from **calib-1m.txt @ ctx=256/chunks=64**

The in-process collection should be CLOSER to runtime activation distribution (matches what the model actually sees), so iteration may converge below v3 even though round 0 is above. We'll see by round 2.

| Round | KLD c512 | PPL | scale-delta | Elapsed |
|---:|---:|---:|---:|---:|
| 0 | **0.1798** | 9.36 | — | 33 min |
| 1 | **0.1809** | 9.18 | 6.34% | 34 min |
| 2 | ⏳ | ⏳ | ⏳ | running |

**Critical finding** at round 1: KLD plateau between rounds — `0.1798 → 0.1809` is essentially flat (+0.06%) despite scale-delta of 6.34%. The iteration IS moving scales but the resulting model quality is hitting a floor at ~0.18 KLD, **higher than v3's 0.1257**.

**Diagnosis**: the limiting factor is the **imatrix calibration data** itself, not the iteration. v3's unsloth-published imatrix appears to be of higher quality (longer/different corpus, possibly different methodology) than what run-003 produces via in-process collection on `calib-1m.txt @ ctx=256/chunks=64`. Iteration converges to a worse fixed point because its starting imatrix is worse.

**Hypothesis**: pointing iterate at unsloth's imatrix as the round-0 stats source (instead of collecting fresh) might recover v3-quality round-0, after which iteration could push below 0.1257. This would be a script-level change — `--candidate-mq4` for round-N>0 (already implemented) + `--initial-stats-npz unsloth.npz` for round-0 (not yet implemented).

### Run-003 final trajectory

| Round | KLD c512 | PPL | scale-delta vs prev | Elapsed | KLD vs v3 (+%) |
|---:|---:|---:|---:|---:|---:|
| 0 | 0.1798 | 9.360 | — | 33 min | +43.0% |
| 1 | 0.1809 | 9.181 | 6.34% | 34 min | +43.9% |
| 2 | **0.1839** | **8.895** | 1.99% | 34 min | +46.3% |

**Convergence**: scale-delta shrinking geometrically (6.34% → 1.99% per round, ratio ≈ 1/3 with damping β=0.5). Would converge below ε=1% in ~1 more round. Iteration is **mathematically working as designed**.

**Quality**: KLD slowly RISES across rounds (0.1798 → 0.1839) while PPL DROPS (9.36 → 8.89). Same KLD-PPL inversion pattern as the alpha sweep and F2 stack. The iteration tunes AWQ scales to match the in-process activation distribution; the model becomes more "self-consistent" (lower PPL) but drifts further from BF16's distribution (higher KLD).

**Goal verdict**: ❌ **sub-0.10 KLD NOT achieved** by iterative AWQ+GPTQ in run-003. Final iterate KLD 0.1839 vs v3's 0.1257 vs target <0.10.

### Why the iteration plateaued above v3

The iterative pipeline's per-round imatrix is collected **in-process** from `calib-1m.txt @ ctx=256/chunks=64`. v3 used **unsloth's pre-collected imatrix** (different corpus / methodology). The unsloth imatrix produces "better" AWQ scales for our KLD objective.

Evidence:
- v3 with unsloth imatrix: KLD 0.1257
- run-003 round 0 with in-process imatrix: KLD 0.1798 (already +43% vs v3 *before* any iteration)
- Iteration changes the FIXED POINT, not the underlying imatrix; can't recover the imatrix gap

### What would unblock sub-0.10

In rough priority order (most-likely-to-work first):

1. **Use unsloth's imatrix as the round-0 stats input** (script-level change ~50 LOC): `--initial-stats-npz unsloth.npz` to bypass round-0 in-process collection. Then iteration starts from v3-quality scales and refines from there. Plausible round-0 ≈ 0.13, iteration drives toward sub-0.10.

2. **Per-tensor α grid search** (Tier 2 proper): vary α ∈ {0.0, 0.3, 0.5, 0.7, 1.0} per tensor, pick each tensor's local optimum based on reconstruction error. The α-curve U-shape with PPL minimum at α=1.0 strongly implies per-tensor heterogeneity. Expected lift 5-15%, plausible sub-0.10.

3. **Iterate on top of v3's actual model** (deeper rewrite): instead of starting from flat-MQ4 + new scales, start from v3's GPTQ-corrected model and use iteration to refine *only* the AWQ scale values while preserving GPTQ corrections. Currently iterate's round-0 GPTQ pass overwrites v3's corrections.

4. **Longer/different calibration corpus** (data-side): re-collect imatrix on a calibration corpus closer to unsloth's (longer documents, different distribution). Out of scope without more info on what unsloth used.

### Infrastructure for lever #1 — SHIPPED, not yet executed

A finer-grained split of lever #1 was shipped on branch `worktree-awq-raw-sumsq-converter` (PR-able to `iterative-awq-gptq`). The original lever bypassed in-process collection wholesale; the shipped version is more surgical and addresses the run-006 diagnosis directly:

- **`scripts/convert_gguf_imatrix_to_npz.py`**: parses a llama.cpp GGUF imatrix (e.g. unsloth's `imatrix_unsloth.gguf_file`), maps each GGUF logical name to the candidate hipfire hfq names (reusing `astrea.gguf_to_hfq_candidates`), and writes a `.npz` keyed by `safe_key(hfq_name) → float32[K]` with raw per-input-channel `sum(x²)`.
- **`scripts/mq4_masked_calib.py iterate --awq-raw-sumsq-npz PATH`**: new flag. When given, the AWQ scale derivation reads raw per-channel sum² for each tensor directly from the npz instead of taking `np.diagonal(rotated_H)`. This bypasses the structural mismatch (FWHT-rotated Hessian diagonals are not raw per-channel statistics under the FWHT) that capped run-003 at KLD ≈ 0.18.
- The flag falls back to the Hessian-diagonal path for tensors absent from the npz (e.g. ssm_out, attn_output not in F1), so partial coverage is safe.

**Validation done (CPU-only, no model runtime):**

- Converter end-to-end on unsloth Qwen3.5-9B imatrix: 496 npz entries, geometric mean of AWQ scales ≈ 1.0, predicted log-range matches measured to ≥4 sig figs.
- Override path produces materially different scales than the Hessian-diagonal path: q_proj layer 7 raw-sumsq gives 14.13× scale range vs 6.40× from the rotated-Hessian-diag fallback. The over-smoothed rotated path was systematically under-applying AWQ correction by ~2× in dynamic range.

**Not yet executed:** the actual end-to-end iterate run + KLD measurement. That requires hiptrx GPU access (1-2h per iterate run; F1-184 mask + base v3 model + imatrix all live on hiptrx). The infrastructure is ready; the lever is one command away.

**Plausible-but-unproven outcome:** sub-0.10 KLD. The diagnosis is structural (rotated Hessian doesn't yield raw sum² under FWHT, so AWQ has been systematically mis-scaled in every iterate round), so the fix is high-leverage *if* the data-side bottleneck from run-003 was actually this and not the calibration corpus. The other plausible outcome is that round 0 lands at ~0.13 (the v3 anchor, slightly perturbed by the more-discriminating scales) and iteration drifts up/down by a few percent — same plateau pattern, different floor. Won't know without running.

**Repro command (to be executed on hiptrx, ~1-2h):**

```bash
# One-shot: convert unsloth GGUF imatrix to raw-sumsq npz
python3 scripts/convert_gguf_imatrix_to_npz.py \
    --in /home/kaden/.hipfire/imatrix/unsloth/Qwen3.5-9B-GGUF/imatrix_unsloth.gguf_file \
    --out /home/kaden/.hipfire/imatrix/unsloth-9b-raw-sumsq.npz

# Iterate with the override active
python3 scripts/mq4_masked_calib.py iterate \
    --hf-model /home/kaden/.cache/huggingface/hub/models--Qwen--Qwen3.5-9B/snapshots/<sha>/ \
    --calib-text benchmarks/calib/calib-1m.txt \
    --imatrix-mask <mask-f1-184-v3.json on hiptrx> \
    --base-output-dir runs/iterate-raw-sumsq-9b/ \
    --awq-raw-sumsq-npz /home/kaden/.hipfire/imatrix/unsloth-9b-raw-sumsq.npz \
    --awq-alpha 0.5 --damping 0.5 --max-rounds 4 --epsilon 0.01 --gpu 0
```

This is the documented lever the user explicitly pivoted away from when calling time on the 6h autoresearch loop. Infrastructure shipped to lower the cost to "execute when convenient" instead of "rebuild the pipeline first."
