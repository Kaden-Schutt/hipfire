// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! # hipfire-engine
//!
//! Architecture-neutral daemon machinery that sits **above** `hipfire-loader`
//! and **below** the product binary.
//!
//! This crate breaks the `daemon -> loader -> daemon` cycle described in
//! docs/governance/2026-08-15-hipfire-leanup-map.md §5b. The helpers the arch
//! crates need (`ContinuousBatchScheduler`, `emit_gen_start`,
//! `conversation_tokens` handling, `redline_*` bench helpers, prompt-frame
//! helpers) were trapped inside `crates/hipfire-runtime/examples/daemon.rs`.
//! Moving them down into `hipfire-loader` would create a cycle because
//! `hipfire-pflash` (and each arch crate) already depend on `hipfire-runtime`
//! and `hipfire-loader` depends on the arch crates. The fix is a layer above
//! the loader that the binary and the arch crates can both depend on:
//!
//! ```text
//! saddle-core -> hipfire-runtime -> hipfire-loader -> hipfire-engine -> (binary)
//! ```
//!
//! ## Layering contract
//!
//! `hipfire-engine` may depend on `hipfire-runtime`, `hipfire-loader`,
//! `saddle-core`, `hipfire-config`, `hipfire-dispatch`, `rdna-compute`,
//! `hip-bridge`. It **must not** depend on `hipfire-pflash` or any
//! `hipfire-arch-*` crate — those must be able to depend on it.
//!
//! Weight manifests and device placement stay out (PR #527).

pub mod emit;
pub mod prompt;
pub mod redline;
pub mod scheduler;
pub mod terminal;
