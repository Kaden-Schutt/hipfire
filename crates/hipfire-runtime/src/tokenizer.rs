// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire - see LICENSE and NOTICE in the project root.

//! Compatibility re-export for the tokenizer implementation.
//!
//! Tokenizer parsing, encoding, decoding, and prompt normalization live in
//! `hipfire-model`; runtime keeps this module so existing imports through
//! `hipfire_runtime::tokenizer` remain source-compatible while callers move.

pub use hipfire_model::tokenizer::*;
