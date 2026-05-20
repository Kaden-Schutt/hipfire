// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! hipfire-arch-qwen35-vl: Qwen3.5-VL vision-language architecture.
//!
//! Wraps the Qwen3.5 text decoder (in `hipfire-arch-qwen35`) with a SigLIP-2
//! ViT vision tower + spatial merger that produces visual tokens spliced
//! into the prompt as `<|vision_start|>` ... `<|image_pad|>×N` ...
//! `<|vision_end|>` before forwarding through the text path.
//!
//! Layout mirrors `hipfire-arch-qwen35` (PR 8):
//!   - `qwen35_vl` — the vision tower (config, weights, GPU forward).
//!   - `image`     — PNG/JPEG decode + smart-resize + patch extraction.
//!                   Lives here because every consumer of these helpers
//!                   today is a vl entry point (`infer_vl`, the vl branch
//!                   of `infer`, daemon's vl request path); moving them
//!                   keeps the runtime crate arch-agnostic.
//!   - `arch`      — `Architecture` trait impl for the vl bring-up triple
//!                   (config_from_hfq → load_weights → new_state).
//!
//! Forward dispatch is intentionally NOT trait-routed. `daemon.rs` and
//! `infer_vl.rs` call `qwen35_vl::vision_forward`, `qwen35::forward_scratch`,
//! etc. directly — see the parent arch crate's `arch.rs` doc-comment for
//! the rationale (forward signatures vary too much; static dispatch is
//! hot-path-friendly; the trait is bring-up scaffolding).
//!
//! When `deltanet` is off the crate compiles to a no-op stub, matching the
//! gating `engine::qwen35_vl` had pre-Phase-2.

#[cfg(feature = "deltanet")]
pub mod qwen35_vl;
#[cfg(feature = "deltanet")]
pub mod image;
#[cfg(feature = "deltanet")]
pub mod arch;

#[cfg(feature = "deltanet")]
pub use arch::Qwen35Vl;
