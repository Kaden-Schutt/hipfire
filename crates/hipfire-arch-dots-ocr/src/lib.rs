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
//! # Bring-up status (rev 2 — phase 2a + 2b + 2c-2 landed)
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
//! - Vision weight loader complete (2c-2):
//!   [`dots_ocr::load_vision_weights`] reads patch_embed + 42 blocks
//!   + post-trunk norm + merger from an HFQ file. The fc1+fc3 →
//!   `fc13_proj` byte-level row concatenation (load-time SwiGLU
//!   fusion per plan §5 phase 2 option (a)) lands here. Helpers:
//!   `load_norm_weight_raw`, `load_bias_f32`, `load_weight_tensor`,
//!   `load_weight_tensor_concat_rows` — all carry
//!   `TODO(transformer-extraction)` markers next to their qwen2 dupes
//!   for the future consolidation PR.
//! - `vision_forward` still a stub — assembly lands in phase 2c-5
//!   after the 2-D RoPE prep helper (2c-3) and vision-shape primitive
//!   variants (2c-4).
//!
//! Not yet wired: daemon load arm for arch_id=8, vision token
//! splicing, infer_dots_ocr.rs driver. Those follow phase 3 (assembly
//! + daemon plumbing).
//!
//! # See also
//!
//! - `docs/plans/qwen_2.0_vlm_plus_dots_ocr.md` — full bring-up plan.
//! - `hipfire-arch-qwen2` — the text-side delegate.
//! - `hipfire-arch-qwen35-vl` — sibling VL arch, closest analog for
//!   the daemon plumbing (image preprocessing + IMGPAD splicing).

pub mod arch;
pub mod dots_ocr;
pub mod image;

pub use arch::DotsOcr;
