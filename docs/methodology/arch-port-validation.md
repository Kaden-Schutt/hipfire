<!--
SPDX-License-Identifier: Apache-2.0
Copyright (c) 2026 Kaden Schutt
hipfire — see LICENSE and NOTICE in the project root.
-->
# Architecture-port validation methodology

How to bring up a new model architecture in hipfire and **prove the forward
pass is correct cheaply** — before spending GPU-hours quantizing the real
weights or chasing coherence on a 200B model. This is the method used for the
MiniMax-M2 port (2026-05-29); it caught 3 real bugs in seconds-per-iteration on
a tiny model with no GPU-hours wasted.

> TL;DR: build a **dimension-faithful tiny random-weight oracle** from the
> *reference* implementation, compare **per-layer hidden states** (not final
> logits), and when a layer diverges, **bisect within the layer** and
> **precision-sweep** to separate a real arch bug from quantization noise.

## Why this works

A new arch port is mostly *plumbing existing kernels in the right order*. The
failure modes are: wrong norm convention, wrong RoPE convention, wrong
gate/up split, wrong routing math, a kernel-shape/`k_top` mismatch, a
dtype/rotation mismatch. **All of these produce a per-layer cosine well below
1.0; quantization noise does not.** So a per-layer cosine harness against a
trusted reference localizes the bug to a layer and a sub-block in minutes.

Doing this on a *tiny* model (2 layers, hidden 256) means each iteration is
~5 s and needs no real weights — you decouple *arch correctness* from *weight
download* and *quantization quality*.

## The loop

1. **Build a tiny reference oracle from the reference implementation.**
   Use the HF `transformers` (or upstream) modeling code — the *ground truth* —
   to instantiate a small random-weight model and dump its per-layer hidden
   states. Use `scripts/gen_tiny_oracle.py` (adapt the per-arch block). Critical
   dim rules (see Pitfalls): keep the *real* `head_dim`/`rotary_dim`, make every
   2D weight `k % group_size == 0`, and match any **hardcoded kernel `k_top`**.

2. **Dump per-layer POST-residual hidden states from both sides** in the shared
   `HFHS` binary format (`magic "HFHS\0\0\0\0"`, `<IIII>` = n_layers, n_pos,
   hidden, reserved, then `n_layers × [n_pos, hidden]` f32, row-major). The HF
   side is in the gen script; the hipfire side is a tiny example mirroring
   `crates/hipfire-arch-*/examples/dump_*_hidden_states.rs` (run `decode_step`
   per position with a per-layer capture hook). Convention: capture the
   residual **after each decoder layer, before the final norm** — match it on
   both sides.

3. **Compare with `scripts/compare_hidden_states.py`.** It prints per-layer
   `rms`, `rel_L2`, `mean_cos`, `min_cos`. Read the drift profile:
   - cosine ≈ 1.0 (≥0.999 for Q8-grade, ≥0.99 for 4-bit experts) → correct.
   - cosine flat across layers but < 1 → uniform per-layer error (quant noise
     *or* a per-layer systematic bug — disambiguate in step 5).
   - cosine craters at a specific layer / compounds → structural bug there.

4. **Bisect within the diverging layer.** Add a second capture point (e.g.
   post-attention, pre-MoE) gated by an env var, and dump the **isolated
   block output** (`block_out = post_block − pre_block`). MiniMax: post-attn
   was 0.9999 (attention correct) but isolated `moe_out` cosine was 0.07
   (orthogonal) → the bug was entirely in the MoE.

5. **Precision-sweep to separate quant-noise from arch-bug.** Re-quantize the
   suspect block at a *higher* precision (e.g. 4-bit → 6-bit) and re-compare.
   - error shrinks ∝ precision → it was quant noise; the arch is fine.
   - error **unchanged** → structural bug, independent of bit-width. (MiniMax:
     MQ4→MQ6 didn't move the 0.96 → proved it wasn't quant.)

6. **Stage-dump + cross-check in Python to root-cause.** Dump the block's
   intermediates (router logits, top-k indices/weights, gate/up, down) from
   hipfire and recompute the *same* in numpy/torch from the F32 weights.
   Compare each stage; the first stage that diverges is the bug. (MiniMax:
   routing indices matched but logits were 30× → F16 router hit the lm-head
   GEMM kernel; later, the expert gate matched but the final output was
   orthogonal + **non-deterministic** → a hardcoded-`k8` kernel reading past a
   top-2 buffer.)

7. **Resolve ambiguities by reading the kernel/dispatch source, not guessing.**
   RoPE convention (rotate_half vs interleaved), `route_scale`, FWHT-rotation
   matching between quantizer and `rotate_x_mq`, FP8 dequant multiply-vs-divide
   — every one was verified against the `.hip` kernel or a numeric reference
   (e.g. dequant a real tensor in torch → check the distribution is sane)
   before trusting it.

## Pitfall checklist (bake into the tiny-oracle dims)

These are the traps that cost iterations on MiniMax — encode them in the tiny
config so they can't bite:

- **Match hardcoded kernel `k_top`.** The indexed-MoE GEMV kernels are often
  hardcoded (`_k8_` → top-8) with no `k_top` parameter. A tiny model with a
  *different* top-k overflows the output buffers and reads garbage
  (non-deterministic!). Set the tiny `num_experts_per_tok` to the kernel's
  hardcoded value (and `num_local_experts` > k_top for real sparsity).
- **`k % group_size == 0`** for every quantized 2D weight (G256 → every dim a
  multiple of 256). The expert intermediate is the usual offender; size the
  tiny model so even the down projection is divisible.
- **Use the real `head_dim` / `rotary_dim`.** Attention/RoPE kernels are tuned
  for specific head_dim (e.g. 128); a tiny `head_dim=32` may hit an untested
  path. Shrink the *number* of heads/layers, not head_dim.
- **Keep routing/precision-sensitive tensors at Q8, never F16.** `weight_gemv`'s
  F16 arm dispatches the lm-head batched GEMM kernel, which is wrong for a
  router's tiny output dim (m = n_experts). Q8 (`gemv_q8_0`) is well-behaved at
  any m.
- **The reference oracle must match the *runtime* layout.** If the HF modeling
  stores experts packed (`gate_up_proj [E,2I,H]`) but your loader/kernels want
  split (`experts.E.w1/w2/w3`), re-split the saved tensors so the oracle and
  hipfire consume the same numbers (numerically identical, just reorganized).
- **Standard vs Gemma RMSNorm.** Check whether the norm is `weight * x̂` or
  `(1+weight) * x̂` before loading norm weights; a wrong `+1` corrupts every
  layer uniformly.

## Reusable artifacts

- `scripts/gen_tiny_oracle.py` — generalized tiny-oracle generator (adapt the
  marked per-arch block: config builder, the post-attention hook module path,
  any packed→split re-split). `scripts/gen_tiny_minimax.py` is the worked
  example.
- `scripts/compare_hidden_states.py` — arch-agnostic HFHS comparator.
- `crates/hipfire-arch-*/examples/dump_*_hidden_states.rs` — the hipfire-side
  per-layer dumper pattern (clone per arch; ~80 lines: load HFQ, per-token
  `decode_step` with a `capture: &mut [Vec<f32>]` hook, write HFHS).

## When this does NOT generalize

This validates *plumbing* against existing kernels. It assumes the new arch
**maps onto kernels that already exist** (attention family × MoE/quant family).
MiniMax-M2 needed **zero new kernels** — that was the enabler. An arch that
needs a *new* HIP kernel (e.g. a quant format with no indexed-decode MoE GEMV)
needs kernel-level oracles too, and the cosine harness can only validate it
once that kernel exists. Map the closest existing arch + the shared helpers
(via a codebase survey) *before* writing, to confirm the kernel coverage.

## Worked example: MiniMax-M2 (arch_id 10)

- Template: qwen35 GQA attention + deepseek4 sigmoid-bias MoE routing, **0 new
  kernels**.
- Oracle: HF `transformers` `MiniMaxM2`, tiny (2L, hidden 256, **head_dim 128 /
  rotary 64**, inter 256 [÷256], **16 experts top-8** [matches `_k8`]), packed→split.
- Bugs caught: (1) tiny used top-2 vs hardcoded-k8 kernel → buffer overflow;
  (2) F16 router → lm-head GEMM kernel → 30× logits; (3) initial confusion
  between quant-noise and bug, resolved by the MQ4→MQ6 precision sweep.
- Result: per-layer cosine **0.9996** (mq4 experts) / **0.9987** (mq2-lloyd),
  attention isolated 0.99990, routing exact — all on a tiny model, no GPU-hours.
