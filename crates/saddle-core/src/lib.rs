// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! # saddle-core
//!
//! Target-agnostic model composition for the saddle substrate.
//!
//! ## Why this crate exists
//!
//! Measured on `8510ca5f2`, hipfire's compute-to-architecture line ratio was
//! **0.70 : 1** where llama.cpp's is **9.7 : 1** — hipfire carried more
//! architecture code than compute code, because every generic concern that
//! `ggml` owns once was reimplemented per architecture. `grammar.rs` existed
//! twice (3,935 lines for one model-agnostic concern); `spec_emit.rs`,
//! `spec_impl.rs` and `mtp_speculator.rs` each existed twice.
//!
//! This crate is where those concerns live once.
//!
//! ## Layering contract
//!
//! `saddle-core` sits **below** `hipfire-runtime` and below every
//! `hipfire-arch-*` crate. It may depend on `rdna-compute`, `hip-bridge`,
//! `serde` and `std` — nothing further. It must never depend on
//! `hipfire-runtime`, `hipfire-dispatch`, or any architecture crate.
//!
//! The rule that governs what belongs here: **abstract the model, never the
//! kernel.** Kernel specialization is what makes hipfire faster than portable
//! engines on RDNA; genericizing it would rebuild `ggml` and forfeit the
//! advantage. Only composition is shared.
//!
//! ## Explicitly not here
//!
//! Weight manifests and device placement. Those are multi-device *placement*,
//! which belongs to the parallelism work in PR #527 and is orthogonal to this
//! layer. See the lean-up map § 5b.
//!
//! ## Module ownership during the re-layering
//!
//! Each module below is filled by exactly one agent in wave 1 or wave 4. This
//! file and `Cargo.toml` are owned by the scaffold and by no agent, so that
//! concurrent work never collides here.

/// Constrained decoding / grammar matching. Model-agnostic: operates on token
/// ids and a grammar. Unified from `hipfire-arch-qwen35` (2,736 lines) and
/// `hipfire-arch-deepseek4` (1,199) — 13 and 1 arch-specific mentions
/// respectively. Lean-up map item **B1**.
pub mod grammar;

/// KV cache. Extracted from `hipfire-runtime/src/llama.rs` (11,999 lines,
/// `KvCache` at :5493), which is neither the right home nor
/// architecture-neutral. Lean-up map item **C1**.
pub mod kv;

/// Architecture capability contract. Replaces the `arch_id ==` branching in
/// `daemon.rs` (43 sites) and the 13-parameter `is_batch_eligible`, so
/// per-architecture policy is declared by the carrier rather than
/// reimplemented by the daemon. Lean-up map item **C3**.
pub mod caps;

/// Per-architecture sampling policy — default temperature, top-p and repeat
/// penalty. Currently duplicated verbatim at `daemon.rs:1310` and `:14618`.
/// Lean-up map item **C4**.
pub mod sampling;

/// Speculative-decode orchestration shared across architectures. Absorbs the
/// same-named file pairs each architecture reimplements — `spec_emit.rs`
/// (903 + 270), `spec_impl.rs` (629 + 1,026), `mtp_speculator.rs` (225 + 320).
/// The bulk of qwen35's speculation is genuinely arch-specific and stays put.
/// Sits behind the existing `SpecTarget` seam rather than replacing it.
/// Lean-up map item **B2**.
pub mod spec;
