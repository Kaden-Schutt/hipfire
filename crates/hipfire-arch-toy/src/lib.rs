// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! hipfire-arch-toy: reference template for new arch crates.
//!
//! This crate is **not a real model** and **must never load**. It mirrors
//! the exact contract a shippable arch crate satisfies:
//!
//! 1. [`ToyBundle`] implements [`hipfire_runtime::arch_model::ArchModel`]
//!    (the arch-agnostic view the loader boxes) — `src/arch_model.rs`.
//! 2. [`load_toy_bundle`] has the exact signature a loader `Carrier` calls
//!    (`src/carrier.rs`), with an honestly stubbed body.
//! 3. [`Toy`] implements the intra-crate bring-up trait
//!    [`hipfire_runtime::arch::Architecture`] — `src/arch.rs`.
//!
//! It is deliberately UNSHIPPABLE: `arch_id` is 0xFF, no `Carrier` in the
//! loader claims 0xFF, and `load_toy_bundle` always returns `Err`. Copy this
//! directory, claim a real id, and fill in the bodies — `README.md` is the
//! step-by-step checklist, verified against the tree.

pub mod arch;
pub mod arch_model;
pub mod carrier;
pub mod toy_model;

pub use arch::Toy;
pub use arch_model::ToyBundle;
pub use carrier::load_toy_bundle;
