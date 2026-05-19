# hipfire-arch-qwen2

Plain Qwen2 dense text decoder for hipfire. arch_id = 7.

## Status

Rev 0: bring-up skeleton from `hipfire-arch-toy` template.
`config_from_hfq` and `load_weights` are stubs; forward pass is not yet
implemented. See `docs/plans/qwen_2.0_vlm_plus_dots_ocr.md` phase 1 for the full
implementation plan.

## Architecture

- GQA attention (e.g. 12 query heads, 2 KV heads for 1.5B)
- RMSNorm (eps=1e-6)
- SwiGLU FFN
- 1-D RoPE (theta = 1_000_000)
- `attention_bias = true` on Q/K/V projections (Qwen2 modeling default)
- Variable `tie_word_embeddings` (1.5B-Instruct: true; dots.ocr: false)

## Relation to other arches

- **Not LLaMA.** The LLaMA crate covers `arch_id = 1` for Qwen2/Qwen3
  today, but doesn't surface the Qwen2-specific QKV bias requirement.
  We claim a new slot rather than restructure that crate.
- **Not Qwen3.5.** Differs in q/k-RMSNorm-pre-RoPE (Qwen3-only), no
  DeltaNet, no MoE.
- **Reused by dots.ocr** (`hipfire-arch-dots-ocr`, arch_id=8) for the
  text path with no weight-key remap needed.

## Validation target

`Qwen2-1.5B-Instruct` from HuggingFace at
`/home/kread/.cache/huggingface/hub/models--Qwen--Qwen2-1.5B-Instruct/`.
