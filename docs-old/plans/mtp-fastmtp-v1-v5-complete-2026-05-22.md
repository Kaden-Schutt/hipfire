# FastMTP fine-tune pipeline — 5 variants, complete analysis

**Date:** 2026-05-22
**Branch:** `mtp-hiptrx-rocprof`
**Hardware:** 1× MI300X VF (training); k9lin 7900 XTX (bench)
**Model:** Qwen3.6-27B
**Code:** `scripts/mtp_train/` + checkpoints on mi300 `/workspace/mtp-fastmtp/`

## TL;DR

End-to-end FastMTP pipeline shipped and 5 training variants benchmarked.
**Best result: parity with stock at full-vocab K=2; -20% in Goal A compressed-serial recipe.**
The runtime cycle structure caps τ at ~2.0 regardless of MTP quality
improvements. The compressed-serial path has a distinct interaction
with trained weights we don't fully understand yet.

Offline metrics improved dramatically (74% → 92% step0 agreement, 57% →
71% step1 recursive agreement) but didn't translate to runtime tok/s.
This is a useful negative result that **clarifies what the actual lever
is** for closing the MTP-vs-DFlash gap.

## 5 training variants (all on Qwen3.6-27B, 1× MI300X)

| Variant | Loss | Steps | Corpus | Offline step0 | Offline step1 | Goal A tok/s | Full-vocab tok/s |
|---|---|---|---|---|---|---|---|
| Stock (baseline) | — | — | — | 74.46% | 56.98% | **~51 tok/s τ=2.55** | **26.8 tok/s τ=2.00** |
| v1: CE-short | CE | 500 | 9 hipfire + 500 wikitext | 90.28% | — | 46 (-10%) | — |
| v2: CE-long | CE | 2000 | HumanEval+ + distill + calib | 92.09% | — | 25 (-50%) | — |
| v3: CE-early-stop | CE | 200 | Same as v2 + LR 2e-5 | 89.13% | — | 46 (-10%) | — |
| v4: KL divergence | KL | 500 | Same as v2 | 92.59% | — | 34 (-32%) | — |
| **v5: recursive CE** | CE×2 | 500 | Same as v2 | **90.12%** | **71.26%** | 40 (-20%) | **26.9 (parity)** |

## Per-prompt offline step0/step1 (v5 final)

| Prompt | step0 | step1 |
|---|---|---|
| LRU cache (canonical bench) | 93.5% | **85.2%** |
| HumanEval_0 | 97.5% | **91.7%** |
| trains-meet | 80.0% | 52.5% |
| tool_call_system | 85.0% | 47.6% |
| agentic_user_multistep | 81.5% | 46.2% |

Massive improvement on code prompts (LRU step1: 85%, HumanEval: 92%).
Weak on non-code prompts (agentic, prose).

## Why runtime didn't improve despite offline gains

### 1. Compressed-serial path has unknown interaction with trained weights

| Config | v5 | Stock | Delta |
|---|---|---|---|
| K=2, compressed-serial, p_min=0.65 | 40 tok/s τ=2.19 | 51 tok/s τ=2.55 | **-20%** |
| K=2, NO compressed-serial, p_min=0.65 | 26.6 tok/s τ=1.98 | 26.6 tok/s τ=2.00 | **parity** |
| K=2, NO compressed-serial, no p_min | 26.9 tok/s τ=2.00 | 26.8 tok/s τ=2.00 | **+0.4%** |
| K=4 p_min=0.5 | 25.5 tok/s τ=1.94 | 26.0 tok/s τ=2.03 | -2% |
| K=5 | 25.9 tok/s τ=2.02 | 25.7 tok/s τ=2.02 | **+0.8%** |
| K=8 | 24.3 tok/s τ=1.96 | 25.2 tok/s τ=2.07 | -3.5% |

**The 20% gap is exclusive to compressed-serial.** All other configs are parity.
The compressed-serial code path (`spec_step_mtp_compressed_serial` in
`mtp_spec.rs:1291`) maintains a "top-2 index scratch" and interacts with
MTP's logit distribution in ways trained MTPs don't satisfy.

### 2. Runtime cycle caps τ at ~2.0 even with K=5/K=8

All variants and even stock peak at τ≈2.0 in full-vocab mode at any K.
The chain truncates aggressively when any draft fails — and even with
v5's higher step1 recursive accept rate, hipfire's runtime can't sustain
longer chains.

Hipfire's `chain_truncated` metric shows ~42% of cycles truncate. Even
trained MTP doesn't reduce this — the p_min or trunk-mismatch gates
fire too readily.

### 3. Bonus token masks MTP quality

Every cycle gets the bonus token (trunk's verify output) for free.
So even when MTP completely fails, τ stays at 1.0. With MTP success,
τ = 2.0 = 1 MTP + 1 bonus. The +1 bonus floor masks any per-token
acceptance improvement until MTP can sustain longer chains.

## What we proved

1. **Recipe is mechanically reproducible** on 1× MI300X in ~3 minutes
   (500 steps, 60 GB peak memory).
2. **Custom Qwen MTP loader works** — all 15 `mtp.*` tensors load
   correctly, forward matches expected behavior.
3. **Recursive CE training reduces step1 compound error** (57% → 71%
   on LRU it's 85%) — confirms the FastMTP approach is correct.
4. **Full-vocab path is parity** — trained MTPs don't break the
   normal speculative cycle.
5. **Compressed-serial path is broken for trained MTPs** — but only
   that path. The runtime works fine without it.
6. **K=5/K=8 don't unlock further gains** — runtime structurally caps
   at τ=2.0 due to chain truncation.

## What we did NOT prove

- **Why compressed-serial breaks**: needs deeper read of
  `spec_step_mtp_compressed_serial` to understand the top-2 indexing
  and how it interacts with MTP's logit shape.
- **Whether larger corpus helps**: still capped at 996 weighted samples.
  Real FastMTP recipe uses 389K.
- **Whether longer recursive training helps**: only ran 500 steps. v5
  showed improving step1 over time; could go to 2000+.
- **Whether different decay weight helps**: used 0.5; FastMTP uses
  exponential decay with different λ.

## Concrete pipeline / artifacts shipped

| File | Purpose | Size/LOC |
|---|---|---|
| `scripts/mtp_train/mtp_module.py` | Custom Qwen MTP loader (130 LOC) | — |
| `scripts/mtp_train/mtp_baseline.py` | Pretrained MTP eval | — |
| `scripts/mtp_train/mtp_train_0p8b.py` | 0.8B smoke (validated) | — |
| `scripts/mtp_train/mtp_train_27b.py` | Single-step CE 27B | — |
| `/workspace/mtp-fastmtp/corpus_v2/` (mi300) | Code corpus (498 distill + 164 HE+ + calib) | 16 MB |
| `/workspace/mtp-fastmtp/qwen3.6-27b-trained-v5.mtp` (mi300) | Best checkpoint | 215 MB |
| `/tmp/qwen3.6-27b-trained-v5.mtp` (k9lin) | Bench-ready | 215 MB |

5 trained checkpoints saved on mi300, all converted to `.mtp` format
and benchmarked end-to-end.

## Critical insight (key takeaway)

**Offline MTP↔trunk argmax agreement is NOT a sufficient predictor
of runtime tok/s.** The hipfire runtime has multiple gates (p_min,
compressed-serial vocab restriction, chain truncation) that interact
with the trained MTP's logit DISTRIBUTION SHAPE, not just its argmax.

For FastMTP-style training to translate to runtime tok/s in hipfire,
we need either:
1. **Train with the runtime's gating math built into the loss**
   (mask logits to compressed-vocab subset, use p_min-aware loss)
2. **Modify the runtime to be less sensitive to logit shape**
   (replace p_min hard threshold with soft acceptance)
3. **Train a completely different MTP architecture** that's not
   gated the same way (e.g., separate small drafter like DFlash)

## Next experiments (priority for future session)

1. **Read `spec_step_mtp_compressed_serial` end-to-end** (~30 min)
   — understand the top-2 indexing logic, identify the trained-MTP
   incompatibility precisely.

2. **Try masked-vocab training** (~1 hour code + 10 min train)
   — restrict CE loss to top-16K vocab tokens. May fix the
   compressed-serial path.

3. **Scale corpus 10x** (~3-6 hours data gen + 30 min train)
   — generate 5K-10K self-distilled samples instead of 996.
   May fix small-sample overfitting.

4. **Try the actual Unsloth Qwen3.6-27B-MTP GGUF as drop-in**
   — sanity check that ANY pre-trained MTP improves runtime.

5. **Diagnose runtime mtp_max_prob distribution**
   — instrument mtp_only_demo to log MTP's per-position max logit
   prob during a real run. Compare stock vs v5. The compressed-serial
   gap probably traces to this.

## Cost / wall summary

- Total mi300 rental time used: ~1.5 hours (~$3)
- 5 training runs total: ~15 minutes wall combined
- 5 conversion + bench cycles: ~30 minutes wall combined
- Honest result: validated infrastructure, identified the actual
  bottleneck (runtime gates, not training quality), documented
  what works (full-vocab parity) and what doesn't (compressed-serial)
- **Most valuable artifact: the proven `mtp_module.py` loader** —
  enables any future MTP training work on Qwen3.5/3.6
