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
//! # Bring-up status (rev 0 — phase 2 skeleton)
//!
//! - Crate scaffold + `Architecture` trait impl with arch_id=8.
//! - Text-side delegation to hipfire-arch-qwen2 (Config, Weights,
//!   State all wrap Qwen2 equivalents).
//! - Vision tower types declared but `vision_forward` is a stub —
//!   forward pass lands in phase 2c.
//! - Image preprocessing module declared but not yet written —
//!   lands in phase 2b.
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
