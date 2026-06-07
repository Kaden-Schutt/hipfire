<!--
SPDX-License-Identifier: Apache-2.0
Copyright (c) 2026 Kaden Schutt
hipfire — see LICENSE and NOTICE in the project root.
-->
# DFlash trainer (hipfire-native) — design & build plan

**Status: PLAN — not started.** Authored 2026-05-29 during the MiniMax-M2 DFlash
scoping. This is the design + sequencing for a Rust/HIP-native trainer for DFlash
speculative-decode *draft* models.

## TL;DR

Build a hipfire-native trainer so every hipfire target arch can get a *matched*
DFlash drafter on-device, without leaving the Rust/hipfire world (no
SpecForge/sglang/PyTorch in the loop). The justification is **infrastructure for
arch-intake automation + full pipeline ownership** — *not* training speed.
Two reasons speed is the wrong frame:

- The drafter backward is **dense bf16 GEMM**, which rocBLAS already saturates on
  MI300X. hipfire's edge (quantization + fused *inference* kernels) doesn't apply to
  full-precision training.
- The dominant cost is **running the target to produce the distillation signal**, not
  the drafter backward. The 600M drafter's own forward+backward is ~8 h on 8×MI300X;
  the "3–6 days" in published Kimi recipes is the **1T target** + a 1.2B drafter +
  long context. For a 229B target + ~600M drafter, expect **~a day**.

So: build it for ownership and reuse across arches, and because hipfire is genuinely
advantaged at the *data-gen* phase (fast quantized target + existing hidden-state
dumping). Do **not** put it on the critical path of the first MiniMax drafter.

## Background

**DFlash** (z-lab, [arXiv 2602.06036](https://arxiv.org/abs/2602.06036)) is a block-
diffusion draft model for speculative decoding: a small decoder drafts a *block* of B
tokens in parallel; the target verifies them in one batched pass; the longest matching
prefix is accepted. z-lab ships trained drafters on HF (MiniMax-M2.5/M2.7 are
**manually gated**) and open *inference* backends (transformers/sglang/vLLM/MLX), but
the *training* recipe is "coming soon." The community trainer is
[SpecForge](https://github.com/sgl-project/SpecForge) (sgl-project); florianleibert's
`kimi-k26-dflash-mi300x` README documents a 3-phase SpecForge pipeline on MI300X.

A drafter is **per-(target arch)** — it conditions on that target's hidden states and
borrows its (frozen) embed + lm_head. So each new hipfire arch wants its own.

## The drafter we train

Matches hipfire's existing inference contract (`hipfire-runtime/src/dflash.rs`,
`DflashWeights::load`, arch_id 20):

- **Arch:** 5–6 layer Qwen3-style decoder + an `fc` projector that maps concatenated
  target hidden states (from N chosen target layers) → draft hidden. Block-causal /
  block-diffusion with `mask_token_id` (block_size 16 train, 8 infer).
- **Shares (frozen):** the target's embedding + lm_head — the drafter has *no* vocab
  head of its own; it emits hidden states and the caller applies the target lm_head.
- **Size (MiniMax, hidden 3072):** ~600M params, BF16 → ~1.1–1.2 GB. (Kimi's is 1.2B.)
- **Output:** B hidden rows → target lm_head → B candidate tokens.

A trained drafter is converted to hfq via the existing `dflash_convert` binary
(`--mq4`/`--mq3`/`--keep-f32`) and run through the hipfire DFlash inference path.

## The 3-phase pipeline

1. **Data-gen ("regenerate") — hipfire's real advantage.** Run the *target* over a
   corpus (PerfectBlend-style, ~1M samples) to produce target-distribution responses
   and capture hidden states at the chosen `target_layer_ids`. hipfire already has the
   fast quantized target + `dump_minimax_hidden_states` (per-layer hidden capture).
   This is the dominant compute; doing it in hipfire avoids re-standing-up the 229B
   target in sglang. **Cache caveat:** full hiddens for 1M × 4K × hidden × N layers is
   hundreds of TB — *cannot* cache naively. Stream from the target during training, or
   cache a managed subset / re-run per epoch on the fast quantized target.
2. **Train.** 6-epoch block-diffusion distillation: drafter forward → block of logits
   via target lm_head → block-diffusion loss vs the target's tokens/distribution →
   backward → Adam. ~8 h for 600M on 8×MI300X @ ~35% MFU.
3. **Convert + validate.** `dflash_convert` → hfq; `scripts/coherence-gate-dflash.sh`
   (q8 KV, max=256, byte-identical prompt + md5); measure τ (acceptance length).

## What hipfire must add (the trainer subsystem)

hipfire is inference-only — **no backward, no optimizer**. The trainer is a *bounded,
fixed-arch* subsystem (not a general autograd framework):

- **Forward — HAVE IT.** Drafter forward = `dflash.rs::draft_forward`. Target forward +
  hidden capture = `dump_minimax_hidden_states` / the arch forward.
- **Backward (hand-coded for the fixed drafter arch):**
  - transposed-GEMM backward — reuse the existing GEMM (dW, dX are GEMMs with transposes).
  - rmsnorm backward; SwiGLU/MLP backward; the `fc` projector backward.
  - **attention backward via gradient checkpointing** — recompute the *flash-attn
    forward* (already owned) per layer in the backward, materialize that one layer's
    scores transiently, run a standard softmax+matmul backward. **No fused
    flash-attn-backward kernel needed** — the scary kernel is sidestepped.
  - embedding + lm_head are **frozen** (no grad) — they're the target's.
  - block-diffusion loss + backward (objective per the DFlash paper).
- **Optimizer:** Adam (elementwise; m/v in fp32). ~4.8 GB state for 600M — fine.
- **Driver:** data loader over (target hiddens, target tokens); micro-batching + grad
  accumulation; bf16 compute + fp32 master weights; activation checkpointing; LR
  schedule + warmup; checkpoint save/resume; metric logging (loss, per-block accept).
- **Distributed:** data-parallel across the 8 MI300X (gradient all-reduce). TP not
  needed (drafter is tiny); the *target* in data-gen may shard (separate concern,
  tracked with the hiptrx TP work).

## Cost & sizing (MiniMax)

- Drafter training FLOPs ≈ `6·N·D` = 6 × 0.6e9 × (1M samples × 4K ctx × 6 epochs ≈ 28B
  tokens) ≈ **1e20 FLOPs** → ~8 h on 8×MI300X @ 35% MFU.
- Data-gen (229B target over the corpus) is the larger phase but hipfire-fast; total
  ≈ **~1 day**, not the Kimi "3–6 days" (those are 1T-target / 1.2B-drafter numbers).
- Optimizer + bf16 master + activations: comfortably within a single MI300X's 192 GB
  for a 600M drafter; data-parallel for throughput.

## Sequencing & scope

1. **Inference first (the keystone, needed regardless of drafter source):** the hipfire
   DFlash *inference* path for minimax — a **batched verify forward** (process B
   positions at once; minimax is batch-1 even for prefill today), `hidden_rb`
   extraction at `target_layer_ids`, batched lm_head, `spec_step_dflash_minimax`
   (port from qwen35, *drop* the DeltaNet rollback — minimax is pure FA), daemon
   registration. **Bonus: the batched forward also fixes minimax's slow per-token-AR
   prefill.** No new GPU kernels (batches the existing pipeline). ~3–5 days.
2. **First drafter — don't block on the trainer.** Either the hybrid (hipfire data-gen
   → SpecForge train) or z-lab access if granted. Gets MiniMax-DFlash live soonest.
3. **The Rust trainer — deliberate infra investment** (this doc). ~1–2 weeks for the
   subsystem, justified by reuse across every future arch (the intake-automation goal),
   not by minimax. Build the backward/optimizer/driver, then train a minimax drafter
   end-to-end as the first customer.
4. **Generalize:** parameterize the trainer over the target arch (qwen35 / deepseek4 /
   minimax / future) so "quantize + train a drafter" becomes a standard intake step.

## Open questions / risks

- **Block-diffusion objective details** — the exact loss + mask schedule from the DFlash
  paper; reimplement faithfully. *Biggest unknown.*
- **Hidden-state cache** — too big to materialize fully; pick stream-vs-cache strategy.
- **SpecForge MiniMax support** — for the hybrid path, does SpecForge already target the
  MiniMax arch or does it need adding?
- **MFU on the small drafter** — many small ops; fusion + CUDA-graph-equivalent capture
  to keep the GPUs fed.
- **Validation** — τ + `coherence-gate-dflash.sh`; a matched drafter should reach
  ~60–75% acceptance (st=8 viable) vs ~50% for a cross-version one.

## Milestones

1. hipfire DFlash *inference* for minimax (batched verify forward + spec-step + daemon) — also a prefill win.
2. hipfire target-conditioning data-gen pipeline (reuse `dump_minimax_hidden_states` + generation).
3. Trainer core: backward kernels (checkpointed attention) + Adam + block-diffusion loss + driver.
4. E2E: train a MiniMax drafter → `dflash_convert` → validate τ + coherence-gate-dflash.
5. Generalize over target arch → an arch-intake step.

## Next program: MTP (after DFlash)

MTP (native multi-token prediction) is the follow-on, and it **reuses most of this**:

- **Shared keystone.** MTP verification also needs the batched target forward + the
  spec-step loop (milestone 1). Build it once for DFlash; MTP rides on it.
- **Shared trainer.** MiniMax's checkpoint has `use_mtp:true` but **no `mtp.*` weights**
  — so MTP means *training/extracting* an MTP head, which is exactly what the DFlash
  trainer's data-gen + small-model-training infra does. The MTP head is even simpler
  than the DFlash drafter (no block diffusion; conditions on the single last hidden +
  next-token embedding, deepseek4-style `mtp_forward`).
- **Why DFlash first:** the z-lab DFlash artifact + SpecForge give a head start (drafter
  for free if access lands), and DFlash-solo has historically been the higher-leverage
  decode-accel lever. MTP is viable for MoE (A3B hit ~260 tok/s on R9700), so it's worth
  doing — just second, recycling the DFlash batched-forward + trainer groundwork.

## References

- DFlash paper: arXiv 2602.06036 · repo: github.com/z-lab/dflash · SpecForge: github.com/sgl-project/SpecForge
- florianleibert/kimi-k26-dflash-mi300x (SpecForge MI300X recipe; st=2, NUMA off, max_num_seqs=32)
- hipfire: `crates/hipfire-runtime/src/dflash.rs` (drafter, arch-agnostic), `crates/hipfire-arch-qwen35/src/speculative.rs` (`spec_step_dflash`, the qwen35-coupled bits to port/strip), `crates/hipfire-arch-qwen35/src/qwen35.rs::forward_prefill_batch_with_pbs` (batched-verify template), `crates/hipfire-arch-minimax/src/forward.rs` (batch-1 only — needs the batched forward), `crates/hipfire-quantize/src/bin/dflash_convert.rs` (draft → hfq, arch_id 20)
