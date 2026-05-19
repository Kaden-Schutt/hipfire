# Qwen3.5/3.6-A3B MoE coherence investigation

**Date:** 2026-05-10 (in progress) — **resolved 2026-05-17**
**Trigger:** Validating PR #228 (rmsnorm fix for MoE final norm) re-exposed a pre-existing `<think>` infinite-loop spiral on Qwen3.6-35B-A3B reasoning prompts. The spiral exists at correct GemmaRMSNorm scale and partly resists the env-var workaround that restores under-scaled behavior. This document tracks the investigation into the underlying precision attractor.

## Resolution (2026-05-17)

The underlying attractor was **not** a precision cliff in the MoE / router / final-norm path. It was the daemon's `repeat_penalty` default of 1.3 over a 128-token window penalizing legitimately repeated chain-of-thought formatting tokens (bullet markers, indentation), dropping the trajectory off the model's well-trained reasoning path into a self-doubt / number-hallucination attractor. The pattern matched llama.cpp's `--repeat-penalty 1.0` and HF transformers' `generate(repetition_penalty=1.0)` defaults; both produce clean structured CoT on the same prompts at greedy decode. Commit `9b4ab74a` (PR #267) flipped the daemon default from 1.3 → 1.0, dissolving the spiral on Qwen3.6-35B-A3B at correct GemmaRMSNorm scale without any precision-path change. A/B verified on `/local/hipfire/qwen3.6-35b-a3b.mq4`:

- **rp=1.0, no workaround:** clean step-by-step reasoning to 60(t+2)=90t → t=4 hours ✓
- **rp=1.3 (prior default), no workaround:** classic self-doubt spiral ("Wait, re-reading prompt…", hallucinated numbers, looping Step B/C/D)
- **rp=1.0 + `HIPFIRE_QWEN_MOE_FINAL_NORM_RAW=1`:** equivalent quality to rp=1.0 plain

The env-var fallback (`HIPFIRE_QWEN_MOE_FINAL_NORM_RAW`) is therefore redundant and was removed together with this investigation closure. The router-precision / fast-path-eligibility hypotheses below are not falsified, but are no longer load-bearing for the documented spiral; the precision interactions remain a real consideration for future MoE quality work but they're not the reasoning-loop cause.

Reproducer: `scripts/test_pr228_spiral_check.sh` (A/B/C against the same prompt + daemon).

---

## Original (pre-resolution) investigation follows.

## TL;DR (live, will update)

The A3B `<think>` reasoning attractor is **not** a single bug. It's a **regime-dependent interaction** between three independently-tuned variables:

1. **Final-norm magnitude** — controlled by hipfire's `load_norm_weight` (`+1.0` baking) vs. `load_norm_weight_raw` (no bake). Engine commit `1e01c0b` worked around a 3.6-A3B spiral by under-scaling the final norm; PR #228 reverted that workaround for correctness, exposing the regime change.
2. **Router precision** — controlled by quantizer commit `ee1be8a` (May 6 19:23 UTC), which promotes the MoE router (`mlp.gate.weight`) and `shared_expert_gate.weight` from MQ4 to Q8 by default. Was added to fix issue #171 (a *different* attractor — cosine similarity 152/256 expert rows below 0.99 at HFQ4G256).
3. **Engine fast-path eligibility** — `qwen35.rs:1934-1948` requires uniform MQ4 across router + shared expert + gate-side weights for the fused `fused_qkvza_hfq4g256` 4-way GEMV. Q8 router or MQ6 shared expert disqualifies the model from the fast path; it then takes a less-tested `weight_gemv` fallback whose softmax/topk numerics differ.

The spiral fires whenever these three variables land in incompatible combinations. The May 6 era's pre-ee1be8a + pre-rmsnorm-fix file landed in a stable-by-accident regime: 4-bit router + under-scaled norm + fast-path eligible. PR #228 takes us out of that regime; if we revert ee1be8a (restoring 4-bit router) but keep the rmsnorm fix, we get a different combination still to test.

## Empirical timeline

### Phase 1: rmsnorm fix audit (PR #228 prep)

After reading vLLM, llama.cpp, and HF transformers source, established that Qwen3.5/3.6 use `GemmaRMSNorm` whose forward is `x * (1 + w) * rms`. Hipfire's `load_norm_weight` bakes `+= 1.0` at load to match (effective scale `(1+w)`), but commit `1e01c0b` introduced a special case `load_norm_weight_raw` for MoE final norms that skipped the bake — under-scaling the final norm by ~38% on Qwen3.6-A3B.

The maintainer's commit message stated this fixed a `<think>` spiral. PR #228 removed the under-scaling (matching vLLM/llama.cpp/HF) with an opt-in `HIPFIRE_QWEN_MOE_FINAL_NORM_RAW=1` flag for users hitting the spiral.

### Phase 2: smoke test on existing A3B file

Tested `/local/hipfire/qwen3.6-35b-a3b.mq4` (timestamp May 6, 17:03 UTC) on the train-pursuit reasoning prompt at temp=0:

| Mode | Result |
|---|---|
| Default (rmsnorm fix active) | spiral, content empty after `<think>` strip |
| Workaround (`HIPFIRE_QWEN_MOE_FINAL_NORM_RAW=1`) | 1327 chars coherent reasoning ✓ |
| Workaround at temp=0.6, top_p=0.95 | 1374 chars coherent ✓ |

Confirms the spiral is real, the workaround is effective, and PR #228 ships correctly with the workaround as a documented escape hatch.

### Phase 3: PR #214 alternating-mode hypothesis test

Hypothesis from glm-5 review: PR #214 (K-map alternating mode, promoting expert tensors from MQ4 to MQ6 in edge + every-3rd middle layers) might suppress the spiral by reducing expert quantization noise (PPL 8K 25 → 19.96 per the PR's own bench).

Re-quantized A3B from safetensors with `--kmap-mode alternating`. Result on same prompt: **spiral persists** in both default and workaround modes. Simple "Paris" prompt works on the same file; reasoning consistently collapses.

### Phase 4: vLLM cross-reference

Per `docs/plans/qwen35-moe-precision-vllm-comparison.md`, vLLM keeps router and `shared_expert_gate` always **unquantized FP16/BF16** (`quant_config=None` in `qwen3_next.py:122,130-136`). `topk_weights` allocated FP32 always (`fused_topk_router.py:81-82`). vLLM's design contract is the **opposite** of hipfire's K-map Rule 3 (Q8 router promotion).

This raised the hypothesis that hipfire's Q8-router promotion (added by ee1be8a) might be hostile to MoE coherence at correct logit magnitude.

### Phase 5: --no-kmap baseline test

Quantized A3B with `--no-kmap` from safetensors (no K-map promotions, all expert tensors at MQ4). Result on reasoning prompt:

| Mode | Result |
|---|---|
| Default | spiral, 263 tokens, content empty |
| Workaround | spiral, 272 tokens, content empty |
| Simple "Paris" prompt | works ✓ |

Same spiral as the K-map alternating file. So K-map's expert-promotion logic is **not** the cause of the spiral.

### Phase 6: compare_hfq pinpoints the difference

Built a `compare_hfq` tool (`crates/hipfire-runtime/examples/compare_hfq.rs`) that diffs two .hfq files tensor-by-tensor with NRMSE for differing tensors. Output for the May 6 baseline vs the May 10 no-kmap file:

```
common tensors: 21093
EXACT (byte-equal):       20792
QT_DIFFER:                301
BYTES_DIFFER non-dequant:   0
Dequant'd diffs (NRMSE):    0
```

**The 301 QT_DIFFER tensors are exactly the routers and shared_expert_gates** for every layer:

```
model.language_model.layers.0.mlp.gate.weight                  qt13 → qt3 (MQ4 → Q8)
model.language_model.layers.0.mlp.shared_expert_gate.weight    qt13 → qt3
... [repeating for every layer] ...
```

**Everything else is byte-identical** (same FWHT seeds, same bytes, same quantization for all the non-router weights).

### Phase 7: identify the introducing commit

```
$ git log --oneline --since="2026-05-06" --until="2026-05-10" -- crates/hipfire-quantize/
4fd0b05 fix(quantize): K-map edge layers — FFN-only for dense, attn+FFN for MoE (#205)
914e356 feat(quantize): per-tensor mixed precision K-map (#196) (#199)
ee1be8a fix(quantizer): Q8 router default for MoE models — fixes #171 attractor on 3.6-A3B
```

Commit `ee1be8a` (May 6, 19:23 UTC) introduced the always-on Q8 router for MoE models, with this rationale in the commit message:

> 4-bit quantization of MoE router weights destroys routing precision on Qwen3.6-A3B: 152/256 expert rows drop below 0.99 cosine similarity at HFQ4G256 (3× worse MSE than llama.cpp Q4_K_M). This causes structural attractors (repetition loops) on multi-paragraph prompts under greedy decoding.

The original `qwen3.6-35b-a3b.mq4` file was quantized at 17:03 UTC, **2.5 hours before** ee1be8a landed. So the original file is a **pre-ee1be8a artifact** with 4-bit (MQ4) router; every file we quantized today has Q8 router.

### Phase 8: the regime-collision picture

The maintainer's commit ee1be8a fixed one attractor (issue #171: 4-bit router cosine mismatch on multi-paragraph prompts) by promoting the router to Q8. Validation in the commit was done **with the engine's then-current `load_norm_weight_raw` MoE final-norm path** — i.e., under-scaled.

Our PR #228 takes the final norm to its correct GemmaRMSNorm magnitude. This puts the model in a regime that ee1be8a's bench never tested. The Q8 router that fixes one attractor at under-scaled magnitude **trades it for another attractor at correct magnitude**.

The original May 6 file remains coherent in workaround mode because both the rmsnorm fix and ee1be8a are reverted in that combination — it's a snapshot of the only stable regime in the matrix:

| Final norm | Router precision | Smoke test outcome |
|---|---|---|
| Under-scaled (pre-#228 / workaround flag) | MQ4 (pre-ee1be8a) | **stable** ✓ (the May 6 baseline) |
| Under-scaled (workaround flag) | Q8 (post-ee1be8a) | spiral |
| Correct (#228 default) | MQ4 (pre-ee1be8a) | **TBD** (testing now) |
| Correct (#228 default) | Q8 (post-ee1be8a) | spiral |

The TBD row is the experiment running right now: re-quantize with `--no-q8-router` flag (added to the worktree to opt out of ee1be8a's promotion) and test in default rmsnorm mode. **If it produces coherent reasoning, the answer is "ship the rmsnorm fix in default mode + revert ee1be8a + find a different fix for issue #171."** If it spirals too, then we have a deeper precision-attractor problem that neither variable can fully suppress.

## Hypotheses and what they predict

### H1: Q8 router is the actual culprit (under both norm regimes)

Predicts: 4-bit-router + correct-norm produces coherent reasoning. Workaround flag becomes unnecessary. Issue #171 needs an alternative fix (e.g. FP16 router per vLLM, or Q4_K_M-style sub-block scales for the router specifically).

### H2: There's a deeper interaction; neither single variable explains it

Predicts: 4-bit-router + correct-norm still spirals. Multiple variables compound. Need to investigate engine fast-path numerics, softmax precision contracts, or attention-precision regression at correct logit magnitude.

### H3: A3B reasoning is fundamentally fragile at temp=0

Per community reports (QwenLM/Qwen3.6 issues #88, #145; HF Qwen3.6-35B-A3B discussion #19), the loop reproduces at BF16 with Qwen's published reference sampling params. May not be fully fixable in any quantized regime — but Qwen recommends `temp=0.6, top_p=0.95` for thinking-mode, which we tested and saw spiral at on the new files.

H1 is the simplest. Phase 9 below tests it directly.

## Phase 9 — DONE, hypothesis H1 REJECTED

Quantized A3B with `--no-q8-router --no-kmap` (output:
`/local/hipfire/qwen3.6-35b-a3b-mq4-router.hfq`, 18.70 GB).

`compare_hfq` against the May 6 baseline:
```
common tensors: 21093
EXACT (byte-equal):       21093
QT_DIFFER:                0
BYTES_DIFFER non-dequant: 0
Dequant'd diffs (NRMSE):  0
```

**The new file is byte-identical to the May 6 baseline.** Same FWHT
seeds, same quantization, same everything.

Smoke test on the train-pursuit reasoning prompt:

| Mode | Result |
|---|---|
| Default (rmsnorm fix active) | **spiral, 295 tokens, content empty** |
| Workaround (`HIPFIRE_QWEN_MOE_FINAL_NORM_RAW=1`) | **coherent, 1381 chars** ✓ |

**H1 is rejected.** 4-bit router does NOT fix the default-mode spiral.
The Q8 router promotion (ee1be8a) is not the cause — it was a
red herring from the compare_hfq diff. The same byte-identical file
that produces coherent output in workaround mode (matching May 6
behavior exactly) still spirals at correct GemmaRMSNorm scale.

**The discriminating variable is exclusively the rmsnorm fix itself.**
Q8 vs MQ4 router, K-map vs no K-map, alternating vs uniform — none of
these affect the outcome. What matters is whether the final norm runs
at under-scaled magnitude (workaround flag, pre-#228) or correct
magnitude (PR #228 default).

## Phase 10: KLD-quality reframe

User-provided KLD measurements (gfx1100/gfx1151, Qwen3.5-9B variants):

| Variant | Source | Size GB | Mean KLD | p99 KLD | PPL |
|---|---|---|---|---|---|
| 9B-UD-Q3_K_XL | unsloth (3-bit dynamic) | 5.05 | **0.141** | 13.9 | 8.67 |
| 9B-MQ4-uniform | hipfire (4-bit) | 5.20 | **0.876** | 20.0 | n/a |
| 9B-MQ3-Lloyd | hipfire (3-bit Lloyd) | 4.40 | 1.691 | 18.0 | 33.98 |
| 9B-MQ3-uniform | hipfire (3-bit) | 4.10 | 2.622 | 17.4 | 85.25 |

**Hipfire's MQ4 has 6.2× worse KLD than unsloth's Q3** despite being
one bit higher. This is a structural quantization-quality gap.

The maintainer's own ee1be8a commit message acknowledges the
mechanism:

> HFQ4G256 uses a single FP32 scale+zero per 256-element group. Q4_K_M
> uses 8 sub-blocks of 32 with independent 6-bit scales/mins, giving
> much better local adaptation. The router weight has non-uniform
> distribution across hidden_dim, and the coarse HFQ4 grouping loses
> critical precision on expert rows that determine top-8 selection.

That same coarse grouping affects every weight in the model, not just
the router. The Q8 router was a band-aid for one tensor; the broader
problem is hipfire's MQ format itself.

## Reframed root cause

The `<think>` spiral on hipfire A3B at correct GemmaRMSNorm magnitude
is **a downstream symptom of hipfire's MQ4 quantization quality gap**,
not a router-specific or norm-specific issue. Three reinforcing data
points:

1. **Community reports (Qwen issues #145, HF discussion #19) show the
   loop reproduces at BF16** — A3B is fundamentally precision-fragile
   regardless of quantization.
2. **Kaitchup's truncation rate doubles at 4-bit** (full-precision
   ~30% → Intel INT4 35B-A3B ~70%) — quantization amplifies the
   underlying fragility.
3. **Hipfire's MQ4 KLD is 6.2× worse than unsloth's Q3.** When
   community quant-noise tolerances assume Q4_K_M-quality
   distributions, hipfire's MQ4 sits closer to a hypothetical "Q3.5"
   in effective precision — and the A3B routing path can't tolerate
   that much noise at correct logit magnitude.

The under-scaled-norm workaround happened to land in a regime where
the lower output magnitude masked enough noise to keep A3B routing
below the attractor threshold. PR #228's correctness fix removes the
mask; the underlying noise becomes load-bearing.

## What this means for the engine-pass plan

The original three-step plan (C → A → B) needs revision:

- **Option C (engine fast-path tweak) — superseded.** Even byte-identical
  weights spiral, so no fast-path/fallback adjustment will fix this.
- **Option A (read fast-path numerics) — limited value.** The bug is
  upstream of the routing path; weights themselves carry too much
  noise.
- **Option B (instrument-and-measure) — partially relevant.** Useful
  for confirming where in the forward pass the per-token logits begin
  to drift, but won't address the root cause.

The actual engine-pass agenda is now:

1. **Close the MQ format quality gap.** Implement sub-block scales
   (Q4_K_M-style: 8 sub-blocks of 32 with 6-bit scales) in hipfire's
   MQ format. Empirically the dominant precision lever per the KLD
   data and per ee1be8a's own analysis.

2. **Adopt vLLM's routing-path precision contract** as a defense in
   depth: FP16 router + FP32 topk_weights, even if MQ format
   improves. The vLLM authors clearly tested Q8 router and decided FP16
   was structurally necessary.

3. **Add a KLD-vs-reference quality gate** (not just PPL) that catches
   regressions like hipfire's MQ4 vs unsloth Q4_K_M before they ship.
   Currently hipfire validates byte-equality and PPL, but not KLD —
   the latter is what catches "this quant fundamentally compromises
   routing precision" earlier.

4. **In the meantime: PR #228 ships with the workaround flag** as
   documented escape hatch. A3B reasoning users opt out of correct
   GemmaRMSNorm magnitude until hipfire's quant quality is upgraded.

## Hypothesis status update

| Hypothesis | Verdict |
|---|---|
| H1: Q8 router is the bug | **Rejected** — byte-identical 4-bit-router file spirals identically |
| H2: Multi-variable interaction | **Refined** — single root cause is MQ format quality; rmsnorm scale is the trigger that exposes it |
| H3: A3B fundamentally fragile at temp=0 | **Confirmed** — community reports independent corroboration; quantization amplifies (Kaitchup 30%→70%) |

Path forward is fundamentally a **quantization quality** investigation,
not an engine-side numerics audit.

Currently re-quantizing A3B with the worktree quantizer + `--no-q8-router` flag. This produces a file with:
- Uniform MQ4 across all weights (including router and shared_expert_gate)
- No K-map promotions
- Norm storage convention identical to the May 6 baseline

Expected smoke-test cases (4 conditions):

| Mode | Prompt | Expected if H1 true | Expected if H2 true |
|---|---|---|---|
| Default rmsnorm | reasoning | **coherent** | spiral |
| Default rmsnorm | "Paris" | coherent | coherent |
| Workaround | reasoning | coherent (matches May 6 baseline) | depends |
| Workaround | "Paris" | coherent | coherent |

If H1 holds (default + reasoning works), the 4-bit-router file becomes a candidate replacement for the existing /local/hipfire/qwen3.6-35b-a3b.mq4, and ee1be8a's Q8-router promotion needs revisiting at the design level (e.g., gate the promotion behind a flag that defaults off until paired with a coherence audit).

## Engine-pass implications

Regardless of Phase 9 outcome, the investigation has surfaced concrete engine-side issues:

1. **`gate_side_mq4` precondition is fragile.** The fast-path qkvza GEMV requires uniform MQ4 weights for router + shared_expert_gate + shared.gate + shared.up. Any quantization-side change (K-map, ee1be8a, alternating mode) that promotes any of these tensors silently disqualifies the model from the fast path. The fallback `weight_gemv` path's softmax/topk numerics haven't been validated against the same coherence bar.

2. **Engine's Q8 router path is implicit.** The maintainer's commit ee1be8a noted "no performance impact — router GEMV is [256, 2048]." That's true for throughput, but **the engine's MoE softmax/topk path was never explicitly validated for stability with a Q8 router under correct GemmaRMSNorm magnitude.** The validation in ee1be8a's commit message was at under-scaled magnitude.

3. **vLLM's design (FP16 router + FP32 topk_weights) is the long-term reference.** Even if H1 holds and 4-bit router works at correct magnitude, FP16 router would be more robust across regimes. Implementing FP16 router in hipfire requires a new fast-path kernel (existing `fused_qkvza_hfq4g256` is hardcoded to 4-bit weights).

4. **The 1-ULP softmax-renorm structural-attractor issue (`qwen35.rs:1996-2003`) was previously mitigated for one symptom.** Our `<think>` spiral may be a residual of the same class at a different precision-cliff edge.

## Files referenced

- `docs/plans/qwen35-moe-rmsnorm-fix.md` — PR #228 audit
- `docs/plans/qwen35-moe-precision-vllm-comparison.md` — vLLM cross-reference
- `docs/plans/qwen35-gguf-moe-bridge.md` — GGUF conversion follow-up doc
- `crates/hipfire-runtime/examples/dump_norms.rs` — diagnostic tool (PR #228)
- `crates/hipfire-runtime/examples/compare_hfq.rs` — diff tool (this investigation)
- `crates/hipfire-runtime/examples/query_tensor.rs` — quick lookup tool (this investigation)
- `crates/hipfire-arch-qwen35/src/qwen35.rs:1934-1948` — `gate_side_mq4` precondition
- `crates/hipfire-arch-qwen35/src/qwen35.rs:1996-2003` — 1-ULP softmax-attractor mitigation comment
- ee1be8a — Q8-router promotion (the variable we're now isolating)
- `1e01c0b` — original MoE final-norm under-scale workaround (reverted by PR #228)
