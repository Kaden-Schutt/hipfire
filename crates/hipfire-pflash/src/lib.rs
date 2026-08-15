// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! hipfire-pflash: PFlash speculative prefill compression — retained legacy research.
//!
//! This crate is the historical reproduction home for PFlash. It is not
//! mainline or production functionality; prefix caching supersedes it for
//! supported serving workloads. See `AGENTS.md` for policy.
//!
//! PFlash was previously `hipfire_arch_qwen35::pflash` (2,031 lines inside a
//! production arch crate). Lean-up map B3 evacuates it here so that policy
//! and location agree: the code lives outside `crates/hipfire-arch-*`.
//!
//! The crate is a consumer of `hipfire-arch-qwen35` for the hybrid-drafter
//! (DeltaNet + FullAttention) path; the arch crate does NOT depend on this
//! crate.

pub mod pflash;
