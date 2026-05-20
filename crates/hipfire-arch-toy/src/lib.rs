// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! hipfire-arch-toy: reference template for new arch crates.
//!
//! This crate is **not a real model**. It implements the
//! [`hipfire_runtime::arch::Architecture`] trait with hardcoded stub
//! values so a contributor adding a new architecture can copy the
//! directory as a starting point and have a workspace-clean build
//! before they wire in real model code.
//!
//! See `crates/hipfire-arch-toy/README.md` for what to keep, what to
//! replace, and rough effort estimates per component. For a full
//! production reference, read `crates/hipfire-arch-qwen35/`.

pub mod arch;
pub mod toy_model;

pub use arch::Toy;
