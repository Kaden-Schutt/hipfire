# MTP-vs-DFlash gap: why DFlash works, and the recipe to fix MTP

**Date:** 2026-05-21
**Branch:** `mtp-hiptrx-rocprof`
**Status:** Research synthesis. No implementation yet.

## TL;DR

Our MTP solo (τ=2.74, 60 tok/s on 7900 XTX) sits exactly on the
**vanilla untrained-MTP plateau** seen across public reports
(AtomicBot llama.cpp PR #22673 measured 72% accept ≈ τ 2.5-3.0;
NodeNestor 47.5% on consumer GPU). External implementations that
report DFlash-class MTP gains (Unsloth 1.4-2.2×, FastMTP 2.03×,
EAGLE-3 5.6-6.6×) all share **one differentiator: a fine-tuned
draft head**, not a fancier dispatch architecture. Our kernel was
fine — the head weights are the gap.

**Recommended next step**: replicate FastMTP's recipe on our existing
Qwen3.5-27B MTP block. Self-distillation from the trunk, 300-400K
samples, exponentially-decayed recursive CE, freeze trunk. Expected
τ uplift 2.74 → 5-6; tok/s 60 → 110-160. **8-24 wall hours on 4×
R9700, infra already in place** (`scripts/distill/` from tonight's
Track A work).

## Why DFlash works — the architectural insight (from agent #2 trace)

DFlash's drafter is NOT autoregressive. It is a 5-layer **block
diffusion model** that produces ALL B=15 draft tokens in **ONE
forward pass** using:

1. **Non-causal self-attention** across the block — each of 15 draft
   positions attends to every other position simultaneously.
2. **Cross-attention against target hidden states** — each draft
   position reads the actual frozen target's hidden states from prior
   committed tokens.
3. **F16 weights through `gemm_f16_batched_lmhead` at M=16** — the
   draft forward is one batched WMMA call at peak tile utilization.

This means:
- 15 candidates produced in ~5 ms (one drafter forward) vs MTP's
  K candidates requiring K separate forwards through the native head.
- Each candidate is informed by full bidirectional block context AND
  full target context — no autoregressive compounding error.
- The verify uses the same trunk kernels as MTP, but at M=16 (100%
  WMMA tile utilization) vs MTP's M=2-3 (12-18% tile utilization).

The agent's per-cycle dispatch count (`docs/plans/mtp-hiptrx-rocprof-2026-05-21.md`)
confirms: DFlash and MTP have **near-identical cycle walls** (51 ms
vs 45 ms). The 3× tok/s gap is purely in **commits per cycle** — 12
for DFlash vs 3.74 for MTP. Same wall, 3× more output.

## Why MTP loses — the structural mismatch

Qwen3.5's native MTP block is a **single transformer layer** trained
to predict ONE token ahead given the trunk's last hidden state.
Multi-step prediction is implemented in `mtp_spec.rs` by recursively
feeding the previous step's hidden output back into the same block.
Three structural problems:

1. **Autoregressive compounding error**: each step k conditions on
   the prior sampled draft. Errors accumulate fast — per-draft
   acceptance is 65-70% but the chain truncates at K=2 because
   conditional acceptance at K=3 drops below the p_min=0.65 threshold.
2. **No bidirectional context**: draft k can't see drafts k+1..K-1
   that would inform its prediction.
3. **Each step is M=1**: per-draft GEMV class kernels at small-M
   ceiling. Even fully optimized, kernel work caps at +10-15%.

The MTP head as-shipped by Qwen is a **placeholder** — it's a one-step
predictor that we're abusing for multi-step. AtomicBot's NodeNestor
gets 47.5% accept on the same architecture (matches our τ=2.74
ballpark). To get τ=5-8 the head itself needs to be **trained for
recursive multi-step coherence**.

## Two paths forward (ranked by EV)

### Path A: FastMTP-style MTP head fine-tune (RECOMMENDED)

Take Qwen's stock MTP block as starting weights. Freeze trunk + lm_head.
Train ONLY the MTP block on self-distilled data with FastMTP's
recursive-CE loss. Reference: https://github.com/Tencent-BAC/FastMTP
`sft.sh` for exact ms-swift command.

**Recipe (from FastMTP arxiv 2509.18362 + DeepSeek-V3 §3.2):**
- Starting weights: Qwen3.5-27B native MTP block (frozen until just
  the MTP layer params)
- Loss: `L = Σ_k exp(-λk) · CE(mtp_k_logits, trunk_argmax_{t+k})`
  with λ ≈ 0.5, k=1..N (recursive multi-step)
- Data: self-distill 300-400K samples from Qwen3.5-27B trunk at
  seq_len=4096. Code-heavy mix (HumanEval+, MBPP-class, plus
  conversation/reasoning).
- Optimizer: AdamW lr=5e-5 cosine, warmup 0.05, 3 epochs
- Batch: per-device bs=1 × grad-accum 8 = effective bs=32 across 4 GPUs
- Framework: PyTorch + Transformers + ms-swift OR direct HF Trainer

**Expected: τ 2.74 → 5-6** (matches FastMTP's 2.03× / EAGLE-3's
training-time-test would push to 7-8 but needs more infra)
**Inference: tok/s 60 → 110-160** (clears Goal A 80+ with margin,
puts composition Goal B 230+ in reach with current K=2 dispatch).

### Path B: EAGLE-3-style separate drafter (HIGHER CEILING, HIGHER COST)

Train a separate small "draft transformer" with EAGLE-3's training-
time-test trick. ShareGPT 68K + UltraChat 200K + code corpus.
Reference: SpecForge (LMSYS) on AMD MI300X — proven on ROCm.

**Expected: τ 5.6-6.6** (per EAGLE-3 paper on Llama/Vicuna/DSR1-8B-70B)
**Cost: ~3× Path A** (separate model architecture + training pipeline
+ inference integration like DFlash already needs)

This is essentially "build a second DFlash drafter, but trained
EAGLE-3-style instead of block-diffusion-style." Skip unless Path A
underperforms.

### Path C (NOT recommended): Drop in Unsloth's pre-trained MTP GGUF

`unsloth/Qwen3.6-27B-MTP-GGUF` exists. It's Qwen's untouched MTP
weights re-bundled with imatrix calibration. Worth a 1-hour test as a
sanity check (does ANY drop-in MTP weight change τ?) but the discussion
thread confirms it's the same weights we already have. No expected
lift.

## Why FastMTP recipe over DeepSeek-V3 recipe

DeepSeek-V3 co-trains MTP during pretraining (14.8T tokens). We can't
afford that. FastMTP takes Qwen/DeepSeek's pre-trained head as start
and fine-tunes with 389K self-distilled samples. That's the cheap
reproducible recipe. Reported result: 2.03× speedup, ~τ 5-6 from a
DSV3-scale model.

## Hardware feasibility on 4× R9700

Per agent #3 estimate (FastMTP paper has H100 reference, R9700 ≈
0.4× H100 bf16 throughput on flash-attn):

| Phase | Wall time on 4× R9700 |
|---|---|
| Self-distill data generation (389K × seq_len=4096) | 8-12 hours |
| MTP-block-only training (3 epochs, ~18K steps) | 8-12 hours |
| Conversion to `.mtp` + bench | 1-2 hours |
| **Total** | **~20-26 hours wall** |

Memory: Qwen3.5-27B bf16 (~54 GB) sharded across 2 GPUs leaves 74 GB
for trainable MTP params (~600M, Adam state ~6 GB), activations,
optimizer. **Fits comfortably in 128 GB pool.**

No PyTorch+ROCm experience required on our side beyond the existing
distill scripts. SpecForge AMD tutorial
(https://rocm.docs.amd.com/projects/ai-developer-hub/en/latest/notebooks/pretrain/SpecForge_SGlang.html)
demonstrates `torchrun --nproc_per_node 8 ... train_*.py` works on
MI300X — R9700/gfx1201 is newer than MI300X for the relevant ops.

## What we already have (don't rebuild)

From tonight's Phase 1 Track A (`project_mtp_hiptrx_session_2026_05_21`):

- `scripts/distill/run_distill_parallel.sh` — multi-GPU distill harness on 4× R9700
- `~/.hipfire/distill_artifacts_2026_05_21/distill_raw_249prompts.tar.gz` (70K argmax tokens, too small but format proven)
- `scripts/distill/aggregate_argmax.py` — argmax label extraction
- `crates/hipfire-runtime/examples/mtp_extract.rs` — converts trained weights → `.mtp` format

What we DON'T have (Path A blockers):
1. **Larger prompt corpus** (need 300-400K prompts, currently 249)
2. **PyTorch training script** for Qwen3.5-27B MTP block FastMTP-style
3. **Recursive-CE loss implementation** (current distill is one-token-ahead CE only)
4. **ROCm PyTorch env on hiptrx** (untested for 27B; may need install)

## Risk register

| Risk | Mitigation |
|---|---|
| ROCm PyTorch unstable for Qwen3.5-27B bf16 on R9700 | Test bf16 forward first; fall back to MI300x rental ($) if blocked |
| FastMTP recipe doesn't transfer Qwen3.6→Qwen3.5 | Recipe is architecture-agnostic per DeepSeek-V3 §3.2; same MTP block shape |
| 389K samples insufficient | FastMTP claims yes; if not, scale to 1M (extends data-gen wall by 2-3×) |
| Trained head breaks coherence gate | Self-distill from same trunk preserves distribution; verify with `coherence-gate.sh` post-training |
| τ uplift smaller than projected | Path A is still the cheapest test; Path B (EAGLE-3) is the fallback at 3× cost |

## Concrete starting checklist

If next session opts to execute Path A:

1. **Day 1 morning** — read FastMTP `sft.sh` line-by-line; map their
   parameter names to Qwen3.5-27B; verify ROCm PyTorch fwd/bwd works
   on 4× R9700 with bf16 Qwen3.5-27B.
2. **Day 1 afternoon** — generate prompt corpus (300-400K prompts; mix
   of HF code datasets `bigcode/the-stack-smol-xl`, conversation
   `lmsys/sharegpt`, reasoning `open-thoughts/OpenThoughts-114k-math`).
3. **Day 1 evening + Day 2** — self-distill: run trunk forwards on
   prompts via hipfire daemon (extends existing `scripts/distill/`
   pipeline). Store (input_ids, target_argmax, target_hidden) tuples.
4. **Day 3** — write/adapt FastMTP training loop targeting Qwen3.5-27B
   MTP block. Freeze trunk. Implement recursive-CE loss.
5. **Day 4** — train (12 hours wall). Save checkpoints every epoch.
6. **Day 5** — convert checkpoint to `.mtp` via `mtp_extract`, bench
   MTP solo + composition vs baseline. Coherence gate. Commit on
   success.

Total: **5 day overhead** (1 day setup, 3 days execution, 1 day
validation). Expected outcome: MTP solo at τ ≥ 5, tok/s ≥ 100.
Composition (Goal B) at 230+ becomes realistic.

## Sources

**Primary references:**
- FastMTP paper: https://arxiv.org/abs/2509.18362
- FastMTP repo: https://github.com/Tencent-BAC/FastMTP
- DeepSeek-V3 §3.2 (MTP canonical loss): https://arxiv.org/html/2412.19437v1
- EAGLE-3 paper: https://arxiv.org/html/2503.01840v1
- SpecForge AMD tutorial: https://rocm.docs.amd.com/projects/ai-developer-hub/en/latest/notebooks/pretrain/SpecForge_SGlang.html
- NodeNestor injection plumbing: https://github.com/NodeNestor/qwen3.5-27b-mtp-llamacpp
- llama.cpp MTP PR #22673: https://github.com/ggml-org/llama.cpp/pull/22673
- NeuraLiying reference trainer: https://github.com/NeuraLiying/mtp_train

**Internal references:**
- `docs/plans/mtp-hiptrx-rocprof-2026-05-21.md` — DFlash vs MTP rocprof
- `docs/plans/mtp-cycle-anatomy.md` — per-phase timing
- `docs/plans/mtp-gate-up-wmma-ceiling-2026-05-21.md` — kernel-opt close-out
- `crates/hipfire-arch-qwen35/src/speculative.rs:2475` — `spec_step_dflash`
- `crates/hipfire-arch-qwen35/src/mtp_spec.rs` — `spec_step_mtp_compressed_serial`
- `crates/hipfire-runtime/src/dflash.rs` — drafter loader
- `~/.hipfire/distill_artifacts_2026_05_21/` (hiptrx) — captured distill corpus

**Hard-falsified or de-prioritized:**
- Drop-in Unsloth GGUF (Path C): same weights, no expected lift
- Tree MTP composition: 3-7× worse than linear (memory)
- Full-vocab MTP head: identical to compressed sidecar (memory)
- Further gate_up_wmma kernel optimization: caps at ~10% lift (this session)
