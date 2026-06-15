// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire - see LICENSE and NOTICE in the project root.

//! Compatibility re-export for the GGUF artifact parser.
//!
//! The parser lives in `hipfire-model`; runtime keeps this module so existing
//! imports through `hipfire_runtime::gguf` remain source-compatible while the
//! model boundary is migrated.

pub use hipfire_model::gguf::*;
