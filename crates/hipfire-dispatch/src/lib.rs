// SPDX-License-Identifier: MIT OR Apache-2.0
// hipfire-dispatch: unified kernel dispatch abstraction.
//
// One entry point per kernel family. Models never match on DType.
// The dispatch layer selects the correct kernel based on quant format,
// arch capabilities, and feature flags — all resolved at init time.

pub mod context;
pub mod families;
pub mod pipeline;
pub mod resource;
pub mod tables;
pub mod traits;
pub mod types;

#[cfg(test)]
mod tests;
