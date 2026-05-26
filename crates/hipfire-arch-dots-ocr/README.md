# hipfire-arch-dots-ocr

`Architecture` trait impl for **dots.ocr** — a Qwen2-VL-family
layout-analysis VLM from rednote-hilab. arch_id = 8.

Pairs a plain Qwen2 text decoder (delegated to
[`hipfire-arch-qwen2`](../hipfire-arch-qwen2)) with a custom 42-block
`DotsVisionTransformer` (RMSNorm + SwiGLU + 2-D RoPE + non-causal
attention) and a LayerNorm-based PatchMerger. Text decoding produces
JSON / Markdown / SVG layout output gated by a custom chat template.

## Bring-up status (phase 2a + 2b landed)

| component | status |
|---|---|
| Crate scaffold + `Architecture` trait | landed (rev 0) |
| `DotsOcrConfig` parser | landed (text via delegation; vision side parses with defaults fallback) |
| Text-side weight load | landed (delegates to `Qwen2Weights::load`) |
| Image preprocessing (smart-resize + CLIP normalise + patch transpose) | **landed** (phase 2b, rev 1) |
| Vision weight load | **stub** — lands in phase 2c |
| `vision_forward` (42-block ViT + merger) | **stub** — lands in phase 2c |
| Daemon load arm (arch_id = 8) | not started — phase 3 |
| `infer_dots_ocr.rs` driver example | not started — phase 3 |

See [`docs/plans/dots-ocr-prd.md`](../../docs/plans/dots-ocr-prd.md)
for the full bring-up plan, including the silent-failure trap in §2.7
(the patch reshape+transpose) and the OCR coherence gate in §5 phase 4.

## Relation to `hipfire-arch-qwen35-vl`

Closest analog. Key differences (the reasons this is a new crate, not
a `qwen35-vl` parametrisation):

- Vision blocks use **RMSNorm** (eps=1e-5), not LayerNorm.
- FFN is **SwiGLU** (fc1 + fc3 + fc2), not GELU MLP (fc1 + fc2).
- 2-D spatial RoPE (theta=10000) is applied to Q and K inside each
  block; qwen35-vl uses learned positional embeddings.
- PatchMerger pre-norm is **LayerNorm** (with bias), not RMSNorm.
- Patch embed weight is 4-D `[1536, 3, 14, 14]` and has bias; qwen35-vl
  has its own conv layout.
- All vision-block linears are **bias-free** (`use_bias=false` in the
  vision config); only `patch_embed.proj` and the merger MLP have bias
  on disk.

## Layout

```
src/
  lib.rs                — crate doc + module re-exports
  arch.rs               — impl Architecture for DotsOcr (arch_id=8)
  dots_ocr.rs           — Config, Weights, vision_forward
  image.rs              — smart-resize + patch extraction (phase 2b)
```
