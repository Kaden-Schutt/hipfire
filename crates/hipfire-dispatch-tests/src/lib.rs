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

// Every module here holds only `#[test]` functions plus their helper imports,
// consts, and fns. Gate the declarations on `cfg(test)` so a plain
// `cargo build` (tests stripped) doesn't compile those helpers as dead code.
#[cfg(test)]
mod arch_caps;
#[cfg(test)]
mod deepseek4;
#[cfg(test)]
mod dtype;
#[cfg(test)]
mod llama;
#[cfg(test)]
mod qwen2;
#[cfg(test)]
mod qwen35;
