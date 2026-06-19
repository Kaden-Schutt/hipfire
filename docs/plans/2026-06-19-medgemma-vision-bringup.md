# Medgemma vision (gemma3-vl) bring-up plan

Status: **active** — started 2026-06-19, right after gemma3 text E1 landed.
Builds on: `2026-06-19-gemma3-bringup.md` (text decoder, done), the master plan's
Tier-1 vision InputAdapter. Reuses: `hipfire-arch-qwen35-vl` (vision kernels +
image preproc), `hipfire-arch-dots-ocr` (ViT + splice precedent).

## Goal

Native multimodal inference for **Gemma3ForConditionalGeneration** (medgemma):
image → SigLIP vision encoder → multimodal projector → splice into the **gemma3
text decoder (E1, reusable)**. Bring-up target **medgemma-1.5-4b-it** (4B text +
SigLIP — fast to iterate), then medgemma-27b-it. `arch_id = 13` (gemma3-vl).

## Architecture (verified from config + tensor index)

```
image 896×896×3
  └─ vision_tower.vision_model (SigLIP, model_type=siglip_vision_model):
       patch_embedding   Conv2d 3→1152, k=s=14  → 64×64 = 4096 patches × 1152
       + position_embedding (learned, 4096 × 1152)
       27× encoder layer: LN1 → self_attn(full/bidirectional, 16 heads) → +resid
                          LN2 → mlp(fc1 → gelu_tanh → fc2)               → +resid
       post_layernorm                                       → 4096 × 1152
  └─ multi_modal_projector:
       avg-pool 64×64 → 16×16 (kernel/stride 4)             → 256 × 1152
       mm_soft_emb_norm   (RMSNorm, gemma (1+w) convention) → 256 × 1152
       mm_input_projection_weight (linear 1152 → 2560)      → 256 × 2560  (= mm_tokens_per_image)
  └─ splice: replace the 256 image-placeholder tokens (image_token_index=262144,
       delimited by boi=255999 / eoi=256000) in the text stream with these 256
       image embeddings.
  └─ gemma3 text decoder (E1) over the combined embedding sequence.
```

Notes:
- SigLIP uses **LayerNorm (with bias)**, not RMSNorm; **gelu_pytorch_tanh** MLP;
  **bidirectional** attention (no causal mask); **learned** position embeddings
  (not RoPE). `vision_use_head=False` (no pooling head).
- Text tensors are `language_model.model.*`-prefixed (the text loader must strip
  `language_model.`); vision = `vision_tower.vision_model.*`; projector =
  `multi_modal_projector.*`.
- text_config: hidden 2560, 34 layers, head_dim 256, vocab 262208 (base 262144 +
  image/boi/eoi specials).

## Kernel reuse (the big leverage — SigLIP needs ~no new kernels)

qwen35-vl's `vision_forward` already uses: `layernorm_batched` (SigLIP LN),
`vit_attention_f32` (bidirectional), `gelu_tanh_f32`, `add_inplace_f32`
(residual), and a patch-embed path. SigLIP drops the 2D-RoPE (uses a learned
pos-embed add instead). So the encoder is composition of existing kernels.
Projector: avg-pool (small new kernel or strided-average via existing ops),
`rmsnorm_f32` ((1+w)-baked `mm_soft_emb_norm`), `weight_gemv` (projection).

## Phases (commit/push each ✓)

- **V0 — ingest.** Quantizer handles gemma3 multimodal: keep `language_model.`
  text tensors (loader strips prefix), ingest `vision_tower.*` + projector
  tensors, and extend the `(1+w)` norm bake to `mm_soft_emb_norm` (SigLIP LNs are
  standard LayerNorm — NOT baked). Tag `arch_id 13`.
- **V1 — SigLIP encoder.** New crate `hipfire-arch-gemma3-vl`: `SigLipConfig`,
  vision weight load, `vision_forward` (patch embed → +pos → 27 ViT layers →
  post_ln), modelled on qwen35-vl. Validate encoder output stats vs a reference.
- **V2 — projector.** avg-pool 4096→256 → `mm_soft_emb_norm` → `mm_input_projection`
  → 256 × 2560 image embeddings.
- **V3 — splice + text.** Add `forward_step_with_embed` to the gemma3 forward
  (insert a prebuilt embedding at image-placeholder positions, à la qwen2);
  reuse the gemma3 text decoder for the combined sequence.
- **V4 — image preprocessing.** 896² resize + normalize (adapt qwen35-vl
  `image.rs`); pan-and-scan deferred.
- **V5 — bring-up + validate.** `infer_gemma3_vl` example: image + prompt →
  coherent description. Eyeball coherence (medical image Q&A).

## Reuse map

- gemma3 text decoder + ingest + (1+w) baking — `hipfire-arch-gemma3` (E1).
- vision kernels (layernorm_batched / vit_attention_f32 / gelu_tanh / add) +
  image preproc — `hipfire-arch-qwen35-vl`.
- ViT-tower + splice structure — `hipfire-arch-dots-ocr` (Qwen2 + vision).

## Risks

- **Patch embed**: Conv2d k=s=14 = im2col + linear; confirm the existing
  patch-embed path matches SigLIP's layout (CHW, no overlap).
- **avg-pool 4096→256**: Gemma3's exact pooling (4×4 over the 64×64 grid) — verify
  ordering (row-major 64×64) before/after.
- **Bidirectional attention scale + no causal mask** in `vit_attention_f32` —
  confirm it's non-causal and scale = 1/√head_dim (head_dim = 1152/16 = 72).
- **Image token count**: prompt must reserve exactly 256 placeholders per image
  between boi/eoi; tokenizer/templating must match.

## V5 COMPLETE — multimodal pipeline validated (2026-06-19)

medgemma-1.5-4b-it (multimodal, arch_id 13, q8f16 text + BF16 vision/projector,
5.0 GB) decoded the brain-MRI fixture correctly via `infer_gemma3_vl`:

> "This is a brain MRI scan. The image shows the brain structure. There are
> several structures visible, including the cerebrum (the largest part of the
> brain), cerebellum, brainstem, and possibly parts of the skull. The image is
> in grayscale, which is typical for MRI scans…"

273-token prefill (256 image rows spliced), clean greedy decode, ec=0. The whole
multimodal path is numerically correct: SigLIP encoder (patch-embed → learned
pos → 27 ViT layers → post_ln) → projector (avg-pool 4096→256 → mm_soft_emb_norm
→ mm_input_projection) → image-token splice (`forward_step_with_embed`) → gemma3
text decoder, plus im2col preproc and the arch-13 multimodal ingest.

Two bugs found+fixed during bring-up:
1. Ingest dropped the vision tower (default `--include-vision` opt-in) →
   arch_id 13 now auto-includes it (`fix(quantize)`).
2. Vision/projector loaders used `tensor_data` (mmap) which returns None on the
   UMA APU (mmap dropped) → switched to `tensor_data_vec` (pread)
   (`fix(gemma3-vl)`).

Landed (pushed): SigLipConfig/Gemma3VlConfig, SigLIP weights+forward, projector,
multimodal loader bundle, image preproc, forward_step_with_embed splice, the
infer_gemma3_vl harness, arch-13 ingest, and both fixes.

**Follow-ups (not blocking):** GPU avg-pool (currently host-side), medgemma-27b-it,
daemon wiring (arch_id 13 ServingBackend), pan-and-scan for large images, and the
prompt-frame specials check (the manual `<bos>`/turn-token construction in the
example should move to the proper gemma chat template when daemon-wired).
