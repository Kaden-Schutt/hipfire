// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! hipfire-arch-qwen35: Qwen3.5 architecture (dense + MoE / A3B / A10B / A17B).
//!
//! This crate implements the [`hipfire_runtime::arch::Architecture`] trait
//! for Qwen3.5. It owns the model forward pass, weight loading, KV-state
//! layout, and the speculative-decoding glue that today is qwen35-specific
//! (`speculative.rs`, `pflash.rs`).
//!
//! Future work (per docs/plans/engine-modularization.prd Phase 2):
//!   - `speculative.rs` and `pflash.rs` will become arch-generic and move
//!     back into `hipfire-runtime`. They live here today because the
//!     existing impls are deeply coupled to `qwen35::*` symbols (config,
//!     weights, scratch, forward functions). PR 8 freezes the dep direction
//!     `arch-qwen35 → runtime`, but accepts that today's spec/pflash are
//!     not generic enough to live above the arch boundary.
//!
//! The `arch` module exposes the trait impl for use by the runtime's
//! daemon and other consumers via `hipfire_arch_qwen35::Qwen35`.

// Qwen3.5 is a hybrid DeltaNet + FullAttention architecture; all the
// runtime infrastructure it touches is `deltanet`-gated. When the parent
// build doesn't enable the feature, the crate is a no-op stub. This keeps
// `cargo build --no-default-features` working and matches the gating that
// was on `engine::qwen35` pre-Phase-2.
#[cfg(feature = "deltanet")]
pub mod arch;
#[cfg(feature = "deltanet")]
pub mod mtp_compose;
#[cfg(feature = "deltanet")]
pub mod mtp_head;
#[cfg(feature = "deltanet")]
pub mod mtp_probe;
#[cfg(feature = "deltanet")]
pub mod mtp_spec;
#[cfg(feature = "deltanet")]
pub mod pflash;
#[cfg(feature = "deltanet")]
pub mod qwen35;
#[cfg(feature = "deltanet")]
pub mod speculative;

/// Grammar-guided decoding for qwen35 tool-call format. Independent of
/// the deltanet feature gate — pure data-structure work, no GPU
/// dependencies. See module docs for design and the Pi turn-12
/// failure mode this prevents.
pub mod grammar;

#[cfg(feature = "deltanet")]
pub use arch::Qwen35;

#[cfg(feature = "deltanet")]
pub use mtp_compose::{spec_step_dflash_mtp_tree, MtpComposeTreeResult, MtpComposeTreeState};
