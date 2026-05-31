// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt

//! # hipfire-dispatch-tests
//!
//! Functional tests verifying kernel dispatch decisions per model family.
//! Organized as one module per model family, each testing the arch × quant
//! × cache-format matrix that determines which kernels are called.
//!
//! No GPU hardware required — all tests exercise pure dispatch logic:
//! `_for_arch()` functions, `is_batchable_la()`, `should_use_mmq()`,
//! `DType` predicates, and `ArchCaps` capability gates.

mod arch_caps;
mod dtype;
mod qwen35;
mod llama;
mod qwen2;
mod deepseek4;
