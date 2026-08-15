// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! # hipfire-generate
//!
//! Per-architecture generation bodies, lifted out of the daemon binary.
//!
//! ## Why this crate exists
//!
//! `daemon.rs` carried 17 `generate_*` functions — roughly 25k lines — that
//! each reach into an architecture crate directly. That is the last thing
//! keeping the daemon's arch-crate reference count above zero.
//!
//! They could not simply move down. `hipfire-loader` has the 11 arch
//! dependencies but sits *below* `hipfire-engine`; `hipfire-engine` sits above
//! the loader but is deliberately arch-free (0 arch deps). The bodies need
//! both sides, so they need a layer above both:
//!
//! ```text
//! saddle-core -> hipfire-runtime -> hipfire-loader -> hipfire-engine
//!                -> hipfire-generate -> hipfire-daemon
//! ```
//!
//! ## Module ownership during the re-layering
//!
//! Grouped by architecture family so each module has exactly one owning agent
//! and the three can be filled concurrently without collision. This file and
//! `Cargo.toml` are scaffold-owned and belong to no agent.

/// Helpers shared by more than one family — the tail that made the first
/// three-way split fail. Single owner; families depend on it and never on
/// each other.
pub mod common;

/// Qwen3.5/3.6 family: the multi-token path, native MTP, speculative and
/// DFlash routes, and expert-parallel. The largest group —
/// `generate_multi` alone is ~5,300 lines.
pub mod qwen;

/// DeepSeek-V4 plus the smaller dense and MoE architectures: LFM2.5,
/// MiniMax-M2, Cohere2-MoE, Gemma4, Muse Glimmer, Qwen2, LLaMA.
pub mod dense;

/// Vision and OCR: Qwen3.5-VL and dots.ocr, including the text-only
/// dots.ocr path.
pub mod vision;
