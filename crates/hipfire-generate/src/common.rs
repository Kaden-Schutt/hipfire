// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Helpers shared by more than one generation family.
//!
//! STUB — filled by the sequential D3 follow-up.
//!
//! ## Why this module exists
//!
//! The first D3 attempt split generation three ways (qwen / dense / vision) on
//! the assumption the families were independent. They are not: they share
//! roughly fifty helpers — `asst_turn_fingerprint`,
//! `production_fail_closed_rollback*`, `free_checkpoints`,
//! `emit_committed_event`, the `ds4_*` cache family, the `spec_*` family.
//!
//! Each agent duplicated what it needed to build in isolation, and both large
//! branches deferred de-duplication to a merge step nobody owned. File-level
//! ownership cannot partition a set of functions with a shared tail.
//!
//! This module is that tail, with a single owner. Families depend on it; they
//! never copy from it and never depend on each other.
