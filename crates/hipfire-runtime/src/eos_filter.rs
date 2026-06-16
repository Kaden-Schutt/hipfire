// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire - see LICENSE and NOTICE in the project root.

//! Compatibility re-export for generation output filtering.
//!
//! The implementation lives in `hipfire-generate`; this module preserves the
//! `hipfire_runtime::eos_filter` path for source-compatible callers.

pub use hipfire_generate::eos_filter::*;
