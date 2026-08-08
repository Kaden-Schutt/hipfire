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
pub mod carrier;
/// Qwen3.5 DFlash / DDTree speculative-decode state (`DflashState`,
/// `load_dflash_state`) and the `DflashSpeculator` impl of the arch-generic
/// `hipfire_runtime::spec::Speculator`. Deltanet-gated — it owns `ModelSlot`-
/// based draft verify.
#[cfg(feature = "deltanet")]
pub mod dflash_spec;
#[cfg(feature = "deltanet")]
pub(crate) mod layer_driver;
#[cfg(feature = "deltanet")]
pub mod mtp_compose;
#[cfg(feature = "deltanet")]
pub mod mtp_head;
#[cfg(feature = "deltanet")]
pub mod mtp_probe;
#[cfg(feature = "deltanet")]
pub mod mtp_spec;
/// Qwen3.5 `MtpDrafter` impl (the arch half of the unified MTP spec-decode
/// core). Deltanet-gated — it touches `ModelSlot` + `MtpSpecState`.
#[cfg(feature = "deltanet")]
pub mod mtp_speculator;
#[cfg(feature = "deltanet")]
pub(crate) mod paro_moe;
#[cfg(feature = "deltanet")]
pub mod pflash;
#[cfg(feature = "deltanet")]
pub mod qwen35;
/// Qwen3.5 impls of the arch-generic `hipfire_runtime::spec` seam
/// (`impl SpecTarget for ModelSlot`). Deltanet-gated — it touches `ModelSlot`.
#[cfg(feature = "deltanet")]
mod spec_impl;
#[cfg(feature = "deltanet")]
pub mod speculative;
#[cfg(feature = "deltanet")]
pub mod store;
/// Test-only access to the unchanged legacy Qwen3.5 forward sequence.
#[cfg(feature = "test-utils")]
pub mod test_utils;

/// Emulated EP2 harness (STEP-002 Task 8, Phase 2B): high-level
/// Single-vs-EP2 parity driver over the one-owner Frozen path.  Test-only,
/// non-default feature; exposes only `Ep2HarnessOptions`, `Ep2HarnessReport`,
/// and `run` — no store/tensor/raw-ownership types.
#[cfg(feature = "emulated-ep2-harness")]
pub mod ep2_harness;

/// Grammar-guided decoding for qwen35 tool-call format. Independent of
/// the deltanet feature gate — pure data-structure work, no GPU
/// dependencies. See module docs for design and the Pi turn-12
/// failure mode this prevents.
pub mod grammar;

/// Per-token spec-decode emission (`SpecEmit`). Pure CPU; named here because it
/// drives the qwen35 `grammar` matcher. Built via [`spec_emit::Qwen35Emit::from_ctx`].
pub mod spec_emit;

/// Env-var-driven fault injection for the Frozen construction path. Compiles
/// to a constant `None` in production builds (feature `frozen-fault-inject`).
mod frozen_fault_inject;

#[cfg(feature = "deltanet")]
pub use arch::Qwen35;

#[cfg(feature = "deltanet")]
pub use carrier::{free_qwen35_bundle, load_bundle as load_qwen35_bundle, Qwen35Bundle};
#[cfg(feature = "deltanet")]
pub use mtp_compose::{spec_step_dflash_mtp_tree, MtpComposeTreeResult, MtpComposeTreeState};
#[cfg(feature = "deltanet")]
pub use mtp_speculator::{build_qwen35_mtp_speculator, Qwen35MtpDrafter};
#[cfg(feature = "deltanet")]
pub use store::{
    load_qwen35_hfq_weights, load_qwen35_paro_weights, preflight_qwen35_frozen,
    Qwen35FrozenIneligible, Qwen35FrozenPlan, Qwen35FrozenPreflight, Qwen35LoadError,
    Qwen35MoeLoadFlags,
};
