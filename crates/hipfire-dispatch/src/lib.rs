// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Björn Bösel
// hipfire — see LICENSE and NOTICE in the project root.
// hipfire-dispatch: unified kernel dispatch abstraction.
//
// One entry point per kernel family. Models never match on DType.
// The dispatch layer selects the correct kernel based on quant format,
// arch capabilities, and feature flags — all resolved at init time.

#![allow(
    clippy::doc_lazy_continuation,
    clippy::large_enum_variant,
    clippy::len_zero,
    clippy::manual_flatten,
    clippy::manual_is_multiple_of,
    clippy::needless_borrow,
    clippy::needless_range_loop,
    clippy::needless_return,
    clippy::new_without_default,
    clippy::nonminimal_bool,
    clippy::unnecessary_lazy_evaluations
)]

pub mod context;
pub mod families;
pub mod pipeline;
pub mod resource;
pub mod tables;
pub mod traits;
pub mod types;

#[cfg(test)]
mod tests;

#[cfg(test)]
mod coverage_tests;
