# Plan: qwen3.5 support in the training system (hipfire-train)

Status: SCOPING (not started). Date 2026-06-18. Author: autonomous (/loop).

## Goal

Bring the **qwen3.5** architecture into `hipfire-train` so the un-fused training
path can run it — for (a) a differentiable **teacher / activation source** (e.g.
the real PFlash drafter target, recovery-FT of qwen3.5 quants, probes), and
eventually (b) full fine-tuning. Today `hipfire-train` is hardcoded dense-LLaMA
(`model.rs::LlamaModel`, `loader::load_llama_fp32`, `block.rs` = one LLaMA block);
there is no arch abstraction.

## qwen3.5 architecture (from `crates/hipfire-arch-qwen35`)

A **hybrid**: per-layer type is either **Gated DeltaNet (linear attention)** or
**full attention**, plus **MoE** (or dense FFN). Config shape (A3B numbers):

- **Full-attention layers:** GQA `n_heads=8`, `n_kv_heads=2`, `head_dim=256`,
  **partial rotary** `partial_rotary_factor=0.25` (only 64 of 256 dims get RoPE),
  optional **attention output gate** (`attn_output_gate`), QK-norm. (M-RoPE for VL
  variants — out of scope for text training.)
- **Gated DeltaNet layers:** `linear_num_key_heads=16`, `linear_num_value_heads=16`,
  `linear_key_head_dim=128`, `linear_value_head_dim=128`, **short causal conv**
  `conv_kernel_dim=4`, delta-rule recurrence with gating (beta / decay).
- **FFN:** MoE `num_experts=256`, `num_experts_per_tok=8`,
  `moe_intermediate_size=512` + a **shared expert** (`shared_expert_intermediate_size`),
  OR dense (`hidden_dim=3584`) for non-MoE variants.
- vocab ≈ 248k; pre-norm; SwiGLU experts.

## Op inventory — have vs. need (fwd **and** bwd)

**Already in `hipfire-train/ops` (gradchecked fwd+bwd):** rmsnorm, linear (x/w),
rope (full half-split), swiglu, GQA attention (sdpa), cross-entropy, AdamW,
pflash_score. These cover a dense-LLaMA block end-to-end.

**Need to build (each fwd + bwd, finite-diff gradchecked):**
1. **Partial-rotary RoPE** — rotate only the first `rotary_dim = 0.25·head_dim`
   dims, pass-through the rest. Small variant of the existing rope kernels.
2. **QK-norm** — per-head RMSNorm on q,k before RoPE. Reuse rmsnorm over a
   reshaped view; bwd is rmsnorm bwd.
3. **Attention output gate** — `out = attn · sigmoid(Wg·x)`; element-wise gate
   fwd+bwd (sigmoid + mul, both already exist as kernels — compose).
4. **Gated DeltaNet** (the crux):
   - short causal **depthwise conv1d** (kernel=4) fwd+bwd;
   - **delta-rule linear-attention recurrence** fwd+bwd — for training this wants
     the **chunked parallel** form (not the O(T) scan) to be GPU-efficient and to
     have a tractable backward. This is the single biggest item.
   - input-dependent **gating** (beta, decay) fwd+bwd.
5. **MoE** — router (linear → top-k softmax) fwd+bwd; token scatter/gather;
   per-expert SwiGLU FFN fwd+bwd; weighted combine; **shared expert** add. The
   inference side has all the forward kernels (`moe_*`, `gemv_*_moe_*`); training
   needs the *differentiable* (dense-ish, fp32) versions + the router backward.

## Two scopes

- **Scope B — forward-only (recommended first).** Implement every *forward* above
  (no backward for DeltaNet/MoE), giving: real-target activation/K capture (PFlash
  teacher), KLD/logit probes, and a parity check vs the daemon. Unblocks the
  PFlash real-target label capture without the hardest backward work. Most new
  forwards are modest; DeltaNet forward (chunked) is the main effort.
- **Scope A — full training.** Add the backward for DeltaNet + MoE (+ router).
  Enables qwen3.5 recovery-FT / fine-tuning. Gated behind Scope B + per-op
  gradchecks.

## Refactor: arch abstraction in hipfire-train

Introduce a minimal seam so LLaMA and qwen3.5 coexist without forking the trainer:
- a `LayerKind { FullAttn(..), DeltaNet(..) }` enum + per-layer dispatch in a
  generalized `model_forward`, and an FFN enum `{ Dense(SwiGLU), Moe(..) }`;
- keep the gradchecked LLaMA ops as-is; qwen3.5 reuses rmsnorm/linear/swiglu/
  attention and adds the new ops. Prefer composition over a trait explosion.

## Config / loader

Parse qwen3.5 `config.json` (reference `hipfire-runtime`'s existing qwen3.5 config
+ `SafetensorsSource`); load weights fp32 like `load_llama_fp32`. Handle tied vs
untied head, per-layer-type tensor names, MoE expert tensors.

## Memory reality

A3B (35B) won't fit fp32 on a 24GB/45GB box. Prototype on a **small dense
qwen3.5** (0.8b / 4b) or a synthetic tiny config; the A3B path is for the big
boxes (hipx 96GB / hiptrx) and likely needs bf16 master weights, not fp32.

## Milestones

- **Q1 — config + fp32 loader** for a small qwen3.5 (CPU-checkable against a real
  `config.json`). No GPU.
- **Q2 — new forward ops**: partial-rope, qk-norm, attn-output-gate, conv1d,
  DeltaNet-forward (chunked), MoE-forward (dense fp32). Parity vs daemon logits
  (KLD) on a short prompt.
- **Q3 — Scope-B integration**: hybrid `model_forward` + activation/K capture;
  wire as a PFlash real-target teacher (replaces the Llama-3.2-3B stand-in).
- **Q4 — backward** (Scope A): conv1d, DeltaNet, MoE/router bwd, each gradchecked;
  then end-to-end qwen3.5 recovery-FT.

## Relationship to other plans

Complements (does not replace) the PFlash daemon-hook route in
`2026-06-18-pflash-qat-drafter.md` (P5). A native hipfire-train qwen3.5 forward
also gives a clean differentiable teacher + a qwen3.5 recovery-FT path for the
QTIP/MQ quant work. LFM2 (`hipfire-arch-lfm2moe` exists on the inference side) is
a parallel candidate with a similar conv+attention shape — much of the conv /
hybrid-scheduling scaffolding here would transfer.
