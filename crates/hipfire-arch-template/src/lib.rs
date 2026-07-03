// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! hipfire-arch-template: copy-paste template for a new model family (serving half).
//!
//! This crate is **not a real model**. It implements the
//! [`hipfire_runtime::arch::Architecture`] trait with hardcoded stub values, plus
//! the capability-layer wiring (`caps.rs`), so a contributor adding a new
//! architecture can copy the directory as a starting point and have a
//! workspace-clean build before wiring in real model code.
//!
//! The lean OFFLINE half (the `Ingest` quant-policy the quantizer links) lives in
//! the sibling `hipfire-arch-template-spec`. See `README.md` for the full
//! add-a-family checklist and `crates/hipfire-arch-qwen35/` for a production
//! reference.

pub mod arch;
pub mod caps;
pub mod template_model;

pub use arch::Template;
pub use caps::TEMPLATE_ARCH_ID;
