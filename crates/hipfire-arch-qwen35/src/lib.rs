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
/// SP3 Task 2 — `forward_batch_slots`, the N-slot forward pass. A PARALLEL
/// entry point to `qwen35::forward_prefill_batch_with_pbs_opts` (never a
/// modification of it — see the module doc for why), routing attention
/// and KV-write through SP1's slot-aware `_slots` kernels and DeltaNet
/// through SP2's per-slot `DeltaNetState`. Q8_0-only; depends on `qwen35`
/// for weight/config/scratch types, hence deltanet-gated like it.
#[cfg(feature = "deltanet")]
pub mod forward_slots;
/// Qwen3.5 impls of the arch-generic `hipfire_runtime::spec` seam
/// (`impl SpecTarget for ModelSlot`). Deltanet-gated — it touches `ModelSlot`.
#[cfg(feature = "deltanet")]
mod spec_impl;
#[cfg(feature = "deltanet")]
pub mod speculative;

/// Grammar-guided decoding for qwen35 tool-call format. Independent of
/// the deltanet feature gate — pure data-structure work, no GPU
/// dependencies. See module docs for design and the Pi turn-12
/// failure mode this prevents.
pub mod grammar;

/// Per-token spec-decode emission (`SpecEmit`). Pure CPU; named here because it
/// drives the qwen35 `grammar` matcher. Built via [`spec_emit::Qwen35Emit::from_ctx`].
pub mod spec_emit;

/// `SlotBatch` — one forward step's ragged work across N slots. Pure CPU
/// data structure; no GPU dependencies. See module docs for the
/// per-slot-absolute `positions[]` invariant.
pub mod slot_batch;

/// `Scheduler` — decides what goes into each step's `SlotBatch`. Pure CPU
/// logic; no GPU dependencies. Round-robin, chunked prefill mixed with
/// decode; deliberately minimal — see module docs for why.
pub mod scheduler;

#[cfg(feature = "deltanet")]
pub use arch::Qwen35;

#[cfg(feature = "deltanet")]
pub use carrier::{free_qwen35_bundle, load_bundle as load_qwen35_bundle, Qwen35Bundle};
#[cfg(feature = "deltanet")]
pub use mtp_compose::{spec_step_dflash_mtp_tree, MtpComposeTreeResult, MtpComposeTreeState};
#[cfg(feature = "deltanet")]
pub use mtp_speculator::{build_qwen35_mtp_speculator, Qwen35MtpDrafter};
