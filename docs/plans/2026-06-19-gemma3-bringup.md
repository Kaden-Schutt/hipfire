# Gemma3 bring-up plan (text-first, full-tower target)

Status: **active** — started 2026-06-19.
Owner: chaingun.

## Goal

Native hipfire support for the **Gemma3** family, with no GFX override and no
Python in the hot path (project Rules 1/5). Start with **`gemma3_text`** (decoder
only), then extend to the **full multimodal tower** (SigLIP vision → gemma3
decoder) for the MedGemma multimodal SKUs.

Bring-up target: **`medgemma-27b-text-it`** (`Gemma3ForCausalLM`, clean
`model.layers.*` names, no vision wrapper). Full-tower targets:
`medgemma-1.5-4b-it`, `medgemma-27b-it` (both `Gemma3ForConditionalGeneration`
= gemma3 text + `siglip_vision_model`).

`arch_id` allocation: **12 = gemma3 (text)**, **13 = gemma3-vl (vision+text)**.
(Existing: 0 llama, 1 qwen3/2, 5 qwen3.5, 6 qwen3.5-moe, 7 qwen2, 8 dots-ocr,
9 deepseek4, 10 minimax, 11 lfm2moe.)

## Reference configs (from /srv/huggingface, verified 2026-06-19)

| field | medgemma-27b-text-it | medgemma-1.5-4b-it (text_config) |
|---|---|---|
| hidden_size | 5376 | 2560 |
| num_hidden_layers | 62 | 34 |
| num_attention_heads | 32 | 8 |
| num_key_value_heads | 16 | (GQA) |
| head_dim | **128** | **256** |
| intermediate_size | 21504 | — |
| vocab_size | 262144 | 262144 |
| rms_norm_eps | 1e-6 | 1e-6 |
| rope_theta (global) | 1_000_000 | 1_000_000 |
| rope_local_base_freq | 10_000 | 10_000 |
| sliding_window | 1024 | 1024 |
| sliding_window_pattern | 6 (5 local : 1 global) | 6 |
| query_pre_attn_scalar | **168** (≠ head_dim!) | 256 (= head_dim) |
| final/attn_logit_softcapping | None | None |
| hidden_activation | gelu_pytorch_tanh | gelu_pytorch_tanh |
| attention_bias | False | — |

## Architecture delta vs the llama/qwen35 forward (what's actually different)

1. **Embedding scaled by √hidden_size** — the embed row is multiplied by
   `sqrt(hidden_size)` (bf16-rounded normalizer) before the first layer. Hook:
   at the embedding fill, or bake a scalar into metadata applied once per step.
2. **(1+w) zero-centered RMSNorm** — Gemma stores norm weights `w` and applies
   `(1+w)`. **Fix: bake `+1.0` into every RMSNorm weight at ingest** so the
   existing `rmsnorm_*` kernels run unchanged (norms are non-quantizable →
   stored F32/Q8, so the offset is exact). Record `gemma_norm_offset=1.0` in
   metadata for provenance.
3. **4 norms per layer** (vs llama's 2): `input_layernorm`,
   `post_attention_layernorm`, `pre_feedforward_layernorm`,
   `post_feedforward_layernorm`. The post-norms sit **between the projection and
   the residual add**:
   ```
   x = x + post_attn_norm( attn( input_norm(x) ) )
   x = x + post_ffn_norm ( mlp ( pre_ffn_norm(x) ) )
   ```
   → cannot reuse the fused `GemvResidual` for wo/down; need
   gemv(no-residual) → post_norm → explicit residual-add.
4. **QK-norm per head** (RMSNorm over head_dim on q and k) — reuse the existing
   `has_qk_norm` + `rmsnorm_batched` path (qwen35/llama already do this).
5. **head_dim independent of dim/n_heads** (128 @27b, 256 @4b) — config carries
   it explicitly (LlamaConfig already has `head_dim`).
6. **Custom attention scale** = `query_pre_attn_scalar^-0.5`. For 27b that is
   `1/√168`, **not** `1/√head_dim (=1/√128)`. The attention family must accept a
   scale override; default `1/√head_dim` is WRONG for 27b.
7. **Dual RoPE + sliding-window interleave**: layers cycle 5 local : 1 global
   (`layer_idx % sliding_window_pattern == pattern-1` → global). Local layers use
   θ=`rope_local_base_freq` (10000) and a **1024 sliding-window causal mask**;
   global layers use θ=`rope_theta` (1e6) and full causal. This is the largest
   forward change vs llama (single-θ, global-only).
8. **GeGLU `gelu_pytorch_tanh`** (not SwiGLU/silu): replace `silu_mul_f32` with a
   fused `gelu_mul_f32` (gelu-tanh of gate × up). A `gelu_tanh_f32` unary kernel
   already exists (`kernels/src/gelu_tanh.hip`); add the fused multiply variant.
9. **No soft-capping** in gemma3 (gemma2 had final=30 / attn=50) — nothing to do.
10. **Tied embeddings**, vocab 262144.

## Existing foundations to reuse

- `kernels/src/gelu_tanh.hip` + `Gpu::gelu_tanh_f32` — gelu-tanh activation.
- `superop.rs::AttnFlavor { window, qk_norm, q_scale_sqrt_hd, logit_softcap, rope }`
  — the dispatch surface for Gemma attention is already scaffolded (comment:
  "Gemma exercises the full surface"). Need to confirm/finish kernel backing.
- `has_qk_norm` + `rmsnorm_batched` (qwen35/llama).
- dispatch pipeline: `execute_steps` / `GemvFamily` / `RotationFamily` /
  `attention_family().run_attention`.
- EOS `<end_of_turn>` filter config + tests (`hipfire-generate::eos_filter`).
- Gemma BOS/chat-template hooks (`hipfire-prompt`: explicit `<bos>`).
- `hipfire-arch-llama` as the dense-forward template; `hipfire-arch-qwen35-vl`
  as the vision-tower-splice template for Phase 3.

## Phases (commit at each ✓)

### Phase 0 — Ingest + arch registration ✓
- Quantizer `model_type` map: `gemma3_text` → arch_id 12; `gemma3` (multimodal
  wrapper) → 12 for now (text tensors only; vision deferred to Phase 3 / id 13).
- Bake `+1.0` into all RMSNorm weights at ingest (input/post_attn/pre_ffn/
  post_ffn/q_norm/k_norm/final_norm). Emit `gemma_norm_offset` + `embed_scale`
  (= √hidden_size) + the Gemma config fields (head_dim, query_pre_attn_scalar,
  sliding_window, sliding_window_pattern, rope_theta, rope_local_base_freq,
  hidden_activation) into the HFQ metadata JSON.
- First artifact: `medgemma-27b-text-it` → `q8f16` (correctness-first; ~27 GB
  fits halo 128 GB GTT). Iterate format → mq4 once the forward is correct.

### Phase 1 — gemma3 text forward (core) ✓✓✓
1.1 New crate `hipfire-arch-gemma3`: `Gemma3Config`, `config_from_hfq`,
    `load_weights` (4 norms/layer + q/k norm + tied embed), `new_state`,
    `Architecture` impl (arch_id 12).
1.2 Forward: the 4-norm residual structure; embed √scale; gemv-no-residual →
    post_norm → explicit residual add.
1.3 Attention: per-layer local/global selection; custom scale
    `query_pre_attn_scalar^-0.5`; dual RoPE θ; 1024 sliding-window mask. Wire
    `AttnFlavor` → attention family (add masked-softmax + scale param if the
    kernel lacks it — **primary risk**).
1.4 GeGLU: add fused `gelu_mul_f32` kernel (mirror `silu_mul_f32`).
1.5 Daemon dispatch: arch_id 12 → Gemma3 loader/forward/state; gemma chat
    frame, `<end_of_turn>` EOS, explicit `<bos>`.
1.6 Validation: greedy decode medgemma-27b-text-it; fluent medical Q&A;
    `./scripts/coherence-gate.sh`.

### Phase 2 — quant formats, perf, eval
- mq4/mq6/q8 sweep; coherence gate; perplexity battery; register a gemma3
  coherence-gate model; warm tok/s on halo + k9lin.

### Phase 3 — full tower: SigLIP vision (gemma3 multimodal, arch_id 13)
- New crate `hipfire-arch-gemma3-vl`: SigLIP-so400m ViT (27 layers, 896² image,
  14² patch, GELU), multimodal projector → text embed space, image-token
  splice into the decoder input (mirror `hipfire-arch-qwen35-vl`). Targets
  medgemma-1.5-4b-it / medgemma-27b-it. Includes Gemma3 pan-and-scan preproc.

## Risks / open questions

- **Attention SWA + custom scale kernel backing** — `AttnFlavor` is a typed
  surface; confirm the attention family kernel actually applies `window` and a
  scale override, or add a masked-softmax variant. Biggest Phase-1 unknown.
- **27B iteration speed on the APU** — q8 ≈ 27 GB in GTT (fits) but slow to load;
  fallback is extracting the 4b text tower (`language_model.*`) for a faster dev
  loop.
- **(1+w) baking** is exact only if norms stay F32/Q8 (they do — non-quantizable
  per `should_quantize`).
