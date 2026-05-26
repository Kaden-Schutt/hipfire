//! hipfire-arch-dots-ocr: dots.ocr layout-analysis VLM.
//!
//! Implements [`hipfire_runtime::arch::Architecture`] for dots.ocr, a
//! Qwen2-VL-family model that pairs a plain Qwen2 text decoder with a
//! custom 42-block `DotsVisionTransformer`. arch_id = 8. See
//! `docs/architecture-ids.md`.
//!
//! The text-side trait impl delegates entirely to
//! [`hipfire_arch_qwen2`] — dots.ocr stores text weights as `model.*`,
//! identical to plain Qwen2, so the Qwen2 loader and forward pass
//! work unchanged on the text path. The dots.ocr-specific code lives
//! in the vision tower ([`dots_ocr::vision_forward`]), image
//! preprocessing ([`image`]), and the prompt-frame + EOS overrides
//! (see [`arch`]).
//!
//! # Bring-up status (rev 4 — phase 2a + 2b + 2c-1..5a landed)
//!
//! - Crate scaffold + `Architecture` trait impl with arch_id=8 (2a).
//! - Text-side delegation to hipfire-arch-qwen2 (Config, Weights,
//!   State all wrap Qwen2 equivalents).
//! - Image preprocessing complete (2b): [`image::smart_resize`],
//!   [`image::clip_normalise`], [`image::extract_patches`], and
//!   [`image::preprocess_image`]. The §2.7 silent-failure trap
//!   (patch reshape+transpose) is gated by
//!   `image::tests::extract_patches_uses_grid_block_order` which
//!   verifies the 2×2-grouped-block-major enumeration against a
//!   synthetic per-pixel-tagged input — catches any drift to raster
//!   order independently of any GPU code.
//! - Vision weight loader complete (2c-2 → refactored 2c-4):
//!   [`dots_ocr::load_vision_weights`] reads patch_embed + 42 blocks
//!   + post-trunk norm + merger from an HFQ file. Linear weights are
//!   loaded as F16 on GPU (HFQ4 / Q8 / F32 sources dequantise at load
//!   time per the qwen35-vl pattern — vision is one-shot per image,
//!   so dequant-on-load is cheaper than wiring batched HFQ4 GEMM for
//!   every per-block linear). The fc1+fc3 → `fc13_proj` row
//!   concatenation (load-time SwiGLU fusion per plan §5 phase 2
//!   option (a)) lands here via `load_f16_or_dequant_concat_rows`.
//! - 2-D RoPE prep complete (2c-3): [`rope::build_rope_2d_tables`]
//!   builds per-patch cos/sin tables in the
//!   `[h-quarter, w-quarter, repeat]` layout dots.ocr expects.
//! - Vision GPU primitives complete (2c-4):
//!   * `rdna_compute::Gpu::rope_2d_halfsplit_f32` — applies the
//!     precomputed tables to Q/K (kernel:
//!     `kernels/src/rope_2d_halfsplit.hip`).
//!   * `dots_ocr::linear_f16` / `linear_f16_no_bias` — F16 GEMM +
//!     optional bias + transpose, matching the qwen35-vl pattern.
//! - `vision_forward` assembly complete (2c-5a):
//!   [`dots_ocr::vision_forward`] runs the full encoder + merger
//!   end-to-end on GPU — patch_embed + RMSNorm, 42 blocks (RMSNorm +
//!   fused QKV GEMM + 2-D RoPE via the new
//!   `rope_2d_halfsplit_qkv_interleaved_f32` kernel + non-causal
//!   `vit_attention_opt` + SwiGLU with load-time fc13 fusion), post-
//!   trunk RMSNorm, and the LayerNorm-based PatchMerger (free 2×2
//!   reshape thanks to the patch-order permutation in 2b+2c-3, MLP
//!   with GELU between layers). Per-stage byte-level validation
//!   against `benchmarks/references/dots_ocr_smoke_001_activations/`
//!   pends 2c-5b — requires a quantised dots.ocr HFQ and an
//!   `infer_dots_ocr` driver binary.
//!
//! Not yet wired: daemon load arm for arch_id=8, vision token
//! splicing, infer_dots_ocr.rs driver. Those follow phase 3 (assembly
//! + daemon plumbing).
//!
//! # See also
//!
//! - `docs/plans/dots-ocr-prd.md` — full bring-up plan.
//! - `hipfire-arch-qwen2` — the text-side delegate.
//! - `hipfire-arch-qwen35-vl` — sibling VL arch, closest analog for
//!   the daemon plumbing (image preprocessing + IMGPAD splicing).

pub mod arch;
pub mod dots_ocr;
pub mod image;
pub mod rope;

pub use arch::DotsOcr;
