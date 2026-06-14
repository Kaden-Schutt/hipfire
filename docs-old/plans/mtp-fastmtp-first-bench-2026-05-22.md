# FastMTP first end-to-end bench — recipe works mechanically, training data too small to translate

**Date:** 2026-05-22
**Branch:** `mtp-hiptrx-rocprof`
**Hardware:** MI300X (1× VF) for training; k9lin (7900 XTX) for bench
**Model:** Qwen3.6-27B (user-selected over 3.5)

## TL;DR

End-to-end FastMTP pipeline shipped and validated. Training mechanically
works: loss drops, offline argmax agreement improves from 74.46% → 90.28%
on held-out prompts (+15.82pp) in 176 seconds on 1× MI300X. **But the
offline metric did NOT transfer to runtime acceptance** — trained MTP is
actually 10-15% slower than stock in the Goal A recipe (compressed-serial
+ p_min=0.65), and identical in the full-vocab path.

Likely root cause: tiny training corpus (9 hipfire prompts + 500 wikitext
samples = 509 total) caused distribution mismatch. Bench is on Python LRU;
training was wikitext-heavy. Compressed-serial gates aggressively on
logit confidence + vocab subset — small distribution shifts get amplified.

## Pipeline shipped (scripts/mtp_train/)

| File | Purpose |
|---|---|
| `mtp_module.py` | Custom nn.Module loader for Qwen's stock MTP block (HF Transformers ignores `mtp.*` weights via `_keys_to_ignore_on_load_unexpected`). 130 LOC. Wraps Qwen3_5DecoderLayer with the pre_fc_norm + fc concat. |
| `mtp_baseline.py` | Measure pretrained-MTP next-token agreement on held-out prompts |
| `mtp_train_0p8b.py` | FastMTP fine-tune for Qwen3.5-0.8B (smoke; +10.71pp in 70s) |
| `mtp_train_27b.py` | FastMTP fine-tune for Qwen3.6-27B (real; +15.82pp in 176s) |

Plus on mi300:
- `/tmp/export_mtp_to_safetensors.py` — convert PyTorch checkpoint → safetensors with `mtp.*` keys
- existing `mtp_extract` binary handles safetensors → .mtp (MQ4G256) conversion

End-to-end conversion path works: `.pt → safetensors → mtp_extract → .mtp (215 MiB)`.

## Empirical results

### Offline (mi300, MTP next-token argmax agreement on 5 held-out prompts)

| | Pretrained (baseline) | After 500 steps FastMTP |
|---|---|---|
| Overall agree | 74.46% | **90.28% (+15.82pp)** |
| LRU prompt | 84.42% | 95.67% (+11.26pp) |
| HumanEval | 70.49% | 90.98% (+20.49pp) |
| Agentic | 62.96% | 88.89% (+25.93pp) |

Loss: 1.57 → 0.51 over 500 steps. Cosine LR 5e-5 with 50-step warmup.

### Runtime (k9lin gfx1100, mtp_only_demo on LRU prompt)

| Recipe | Stock | Trained | Delta |
|---|---|---|---|
| K=2 p_min=0.65 compressed-serial (Goal A) | **51.46 / 55.15 / 48.68 tok/s** τ=2.56-2.71 | 45.26 / 45.20 / 45.67 tok/s τ=2.46-2.49 | **−10 to −15%** |
| K=2 full-vocab (no compressed-serial, no p_min) | 26.34 tok/s τ=2.00 | 26.33 tok/s τ=2.00 | 0% |
| K=4 p_min=0.5 full-vocab | 25.93-26.39 τ=1.97-2.07 | 24.5-24.77 τ=1.96-2.0 | −2 to −7% |

**Key observation:** In the Goal A recipe, `replay_skipped` drops from
74-79% (stock) to 60-62% (trained). Trained MTP triggers full-accept LESS
often. This is the opposite of what we'd expect from +15.82pp agreement.

## Why the offline metric didn't transfer

Three plausible mechanisms:

1. **Distribution mismatch.** Training was 500 wikitext + 9 hipfire prompts.
   Bench is on Python LRU. The MTP shifted toward prose; compressed-serial
   gates on top-16K vocab where prose tokens may not overlap perfectly
   with Python tokens.

2. **Logit shape vs argmax.** Offline metric is argmax (only the top token).
   Runtime uses logit distribution (`mtp-p-min=0.65` thresholds on max
   logit probability). Training with CE on hard labels can shift the
   ENTIRE distribution shape — argmax agreement up, but max-prob may
   actually drop (smoothed distribution).

3. **Compressed-serial vocab subset.** The cvs sidecar restricts MTP's
   output to top-16K tokens. Our training used full-vocab labels. The
   trained weights may have shifted mass toward tokens OUTSIDE the top-16K,
   degrading compressed-serial acceptance.

## What this is and isn't

**This IS:**
- A validated end-to-end FastMTP pipeline (load → train → convert → bench)
- Proof the recipe mechanically works on 1× MI300X in ~3 minutes wall
- A working module loader for Qwen's `mtp.*` weights (HF doesn't ship one)
- The right architecture for scale-up

**This is NOT:**
- A win on the runtime metric (Goal A τ regressed by ~10%)
- Sufficient training data for production (509 samples is ~1000× too small)
- A proof that bigger training won't transfer either

## Next experiments to size the real bet

Priority order:

1. **Bigger code-heavy corpus.** Generate 5K-10K self-distilled samples
   from code datasets (`bigcode/the-stack-smol`, HumanEval+, MBPP-class).
   Re-train, re-bench. ~3-6 hours on 1× MI300X. If runtime acceptance
   improves with same training recipe, distribution mismatch is the cause.

2. **Diagnose logit shape.** Compare stock vs trained MTP logit distributions
   on bench prompts: entropy, top-1 prob, top-5 prob mass. If trained MTP
   has lower top-1 prob (smoothed), that's the p_min interaction. ~30 min.

3. **Try Unsloth's pre-trained MTP GGUF.** If `unsloth/Qwen3.6-27B-MTP-GGUF`
   loads cleanly and improves runtime acceptance, that's a free win and
   confirms training quality matters more than method. ~1-2 hours.

4. **Compressed-vocab-aware loss.** Mask training loss to top-16K vocab.
   If runtime acceptance improves disproportionately, the cvs subset is
   the cause. ~2 hours code + train.

5. **Recursive multi-step CE.** Implement FastMTP's recursive loss (predict
   t+1, t+2, ...). Should help K>2 paths. ~4-6 hours.

## Concrete next session pickup

The pipeline is on `mtp-hiptrx-rocprof` under `scripts/mtp_train/`. Checkpoint
at `/workspace/mtp-fastmtp/qwen3.6-27b-fastmtp-500steps.pt` on mi300,
exported `.mtp` at `/workspace/mtp-fastmtp/qwen3.6-27b-trained-500.mtp`.
Copied to k9lin at `/tmp/qwen3.6-27b-trained-500.mtp` for further iteration.

Stock 3.6-27B baseline on k9lin (so future comparisons have a fixed reference):
- LRU full-vocab K=2 no flags: **26.3 tok/s τ=2.00**
- LRU Goal A recipe (K=2 p_min=0.65 compressed-serial): **51.5 tok/s τ=2.6**
  (vs 3.5-27B Goal A: 60.44 — 3.6 is ~15% slower per-token but should support
  higher K)

To resume: implement experiment #1 (bigger code corpus). Self-distill 5-10K
prompts from `bigcode/the-stack-smol` + HumanEval+ on mi300, re-train,
re-bench. Expected wall: 1-2 hours.

## Cost summary

- 1× MI300X rental time used: ~30 min (mostly probing + 2 training runs)
- Estimated remaining bench budget: $4-8 per training+conversion+bench iteration
- Full FastMTP recipe scale-up (10K samples × 3 epochs): ~$30-50, ~3-6 hour wall
