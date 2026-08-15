// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Speculative-decode orchestration shared across architectures.
//!
//! STUB — filled by lean-up map item **B2** (wave 5).
//!
//! Scope note, so this does not over-reach: most speculation code in the arch
//! crates is genuinely architecture-specific (qwen35's `speculative.rs` 6,449
//! lines, `mtp_spec.rs` 3,862, `mtp_head.rs` 2,484) and belongs where it is.
//! The duplication this module exists to absorb is the same-named file pairs
//! that each architecture reimplements:
//!
//! | file | qwen35 | deepseek4 |
//! |---|---:|---:|
//! | `spec_emit.rs` | 903 | 270 |
//! | `spec_impl.rs` | 629 | 1,026 |
//! | `mtp_speculator.rs` | 225 | 320 |
//!
//! An arch-erased seam already exists — `SpecTarget` in
//! `hipfire-runtime/src/spec.rs`, implemented by eight arch crates. This module
//! deduplicates implementations behind that contract; it does not replace it.
