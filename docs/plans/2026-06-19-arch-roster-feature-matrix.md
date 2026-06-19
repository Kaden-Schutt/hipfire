# Arch roster & feature matrix (multi-family bring-up)

Status: **reference** — drives the family-seam refactor + per-family plans.
Built 2026-06-19 from the actual `config.json`s under `/srv/huggingface`.

## Why this exists

The family roster isn't "a few more transformers" — it spans AR transformers,
hybrid SSM+attention, pure SSM, and block-diffusion. The seam
(`docs/plans/2026-06-19-daemon-family-seam.md`) must be designed against this
diversity, not against transformers alone. This doc is the shared picture.

## Feature matrix (verified configs)

| family | model_type | mixer layers | FFN | generation | new infra forced |
|---|---|---|---|---|---|
| gemma3 *(WIP)* | gemma3_text | SWA + full attn, dual-θ (5:1) | GeGLU | AR | GeGLU, (1+w) norm, dual-θ SWA |
| gemma4 | gemma4_text | SWA+full via `layer_types` list | GeGLU **or MoE** (A4B, 128 exp) | AR | MoE-on-gemma |
| diffusion_gemma | diffusion_gemma_text | SWA+full attn (`layer_types`) | MoE (128 exp, A4B) | **block diffusion** | block-diffusion loop |
| nemotron_h | nemotron_h | **Mamba2 + attn** interleaved (`hybrid_override_pattern`) | dense **or MoE** (128–512 exp) | AR | **Mamba2 SSM+conv kernels** |
| mamba2 | ssm_cfg.layer=Mamba2 | **pure Mamba2** (no attn/KV) | — | AR | **Mamba2 SSM+conv kernels** |
| LFM2 *(WIP, arch_id 11)* | lfm2 / lfm2_moe | **short-conv** + attn | dense or MoE (32 exp) | AR | short-conv state |

Key per-config specifics:
- **gemma4**: head_dim 256, gelu_pytorch_tanh, sliding_window 512/1024, layer
  types as an explicit `layer_types: [sliding_attention|full_attention]` list
  (vs gemma3's `sliding_window_pattern` int). Dense 31B ≈ gemma3-31B; 26B-A4B
  + E2B/E4B are MoE. `Gemma4ForConditionalGeneration` (multimodal wrapper).
- **diffusion_gemma** (`DiffusionGemmaForBlockDiffusion`, 26B-A4B): gemma-shaped
  transformer layers (head_dim 256, GeGLU, SWA, MoE 128) but **block-diffusion**
  generation. Forward is reusable from gemma4; the loop is the novelty.
- **nemotron_h** (`NemotronHForCausalLM`): `hybrid_override_pattern` of M
  (Mamba2) / `*` (attention) / `-` (dense MLP) / E (MoE) per layer.
  mamba_head_dim 64–80, mamba_num_heads 64–128, ssm_state_size 128, conv_kernel
  4. Nano-4B dense-FFN; 30B-A3B (128 exp) + Super-120B (512 exp) MoE; Super also
  has an MTP head (`mtp_hybrid_override_pattern`). 131072 vocab.
- **mamba2** (state-spaces, 130m/2.7b): original (non-HF-transformers) checkpoint
  format — `ssm_cfg={'layer':'Mamba2'}`, no `architectures`, GPT-NeoX vocab
  50277. Pure SSM, the clean kernel-validation vehicle.
- **LFM2.5**: `conv` mixer layers (`conv_L_cache=3`, `conv_dim`) interleaved with
  attention; `lfm2_moe` (8B-A1B) adds 32-expert MoE.

## Two truths this forces on the seam

1. **Generation strategy ⊥ model-forward.** diffusion_gemma proves it: gemma
   transformer layers + block-diffusion loop. So the forward (run the layer
   stack → hidden/logits) must be **separable from the loop**. `SimpleAr` is the
   *AR strategy over a forward*, not the owner of the forward; a `BlockDiffusion`
   strategy reuses the same forward. `ServingBackend` (the boxed seam) is the
   strategy; `GenerateCtx` must not assume causal/KV.
2. **The layer stack is a heterogeneous per-layer mixer list.** mixer ∈
   {full-attn, SWA, Mamba2, short-conv} × FFN ∈ {SwiGLU, GeGLU, MoE}, selected
   per layer (nemotron `hybrid_override_pattern`, lfm2/gemma `layer_types`).
   qwen35's LA/FA hybrid is the existing precedent to generalize.

## Two big NEW infra investments (rest is composition)

- **Mamba2 SSM + conv1d kernels** — `mamba2` and `nemotron_h`. Build/validate on
  **pure mamba2 first** (no attn/MoE confounds), then nemotron_h composes them.
- **Block-diffusion generation loop** — `diffusion_gemma`, reusing gemma4 forward.

## Dependency-aware bring-up order

1. **gemma3** (WIP) — GeGLU/SWA/(1+w), `SimpleAr` + `ServingBackend`.
2. **seam wiring** — route qwen2 + gemma3 through it.
3. **mamba2 (pure)** — land SSM+conv kernels in isolation; validate `SimpleAr`
   on a no-KV recurrent arch.
4. **gemma4** — gemma3 + MoE (reuse qwen35-MoE) + `layer_types` form; cheap.
5. **nemotron_h** — Mamba2(3) + attn + MoE(4) + per-layer hybrid dispatch.
6. **diffusion_gemma** — gemma4 forward(4) + block-diffusion `ServingBackend`.
7. **LFM2** finish + shared-loader / Option-soup cleanup folded across.

Builds each new capability in its cheapest isolating context, composes upward,
and stresses the seam early (no-KV at #3, non-AR at #6).

## Cleanup that the roster justifies

- **Shared transformer loader**: `load_weight_tensor` / `load_norm` /
  `load_embed` / `load_lm_head` are now duplicated across qwen35 / qwen2 /
  dots-ocr / gemma3 (all carry `TODO(transformer-extraction)`). With 4+ more
  transformer families coming, extract into `hipfire_runtime::transformer` after
  gemma3 decodes (generalize on the working instance, not before).
- **Shared MoE FFN**: qwen35-MoE, gemma4, nemotron_h, lfm2_moe, diffusion_gemma
  all need MoE — converge on one expert-FFN building block (watch differing
  expert tensor layouts: stacked-3D vs per-expert-split).
