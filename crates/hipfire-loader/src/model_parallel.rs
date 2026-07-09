// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt

//! Classifies which parallelism axis a loaded model uses, so daemon dispatch
//! can `match` one value instead of a chain of `is_some()` early-returns.
//! Inc 0: classifier only (borrows nothing). The owning enum lands in Inc N+1.
//!
//! # Real dispatch precedence (verified against daemon.rs `fn generate()`)
//!
//! The axis-first early-return order in `generate()` (daemon.rs ~6891–7374):
//!   1. `m.tp.is_some()`       → line 6891
//!   2. `m.pp_dense.is_some()` → line 6940
//!   3. `m.ep.is_some()`       → line 6990
//!   4. `m.pp > 1`             → line 7374  (qwen35 pipeline-parallel)
//!   5. (everything else)      → Single
//!
//! NOTE: The brief guessed `ep > tp > pp_dense > pp_qwen35 > single`.
//! The REAL order is `tp > pp_dense > ep > pp_qwen35 > single`.
//! Both `priority()` and the test reflect the real order.

/// Which parallelism axis is active for a loaded model.
///
/// Variants are ordered by dispatch priority (highest → lowest).
#[allow(dead_code)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ModelParallelKind {
    /// Tensor-parallel dense multi-GPU (`m.tp.is_some()`).
    Tp,
    /// Pipeline-parallel dense multi-GPU (`m.pp_dense.is_some()`).
    PpDense,
    /// Expert-parallel MoE multi-GPU (`m.ep.is_some()`).
    Ep,
    /// Pipeline-parallel qwen35 single-GPU sharding (`m.pp > 1`).
    PpQwen35,
    /// No multi-GPU axis — standard single-GPU path.
    Single,
}

impl ModelParallelKind {
    /// Classify from the four axis-present flags in dispatch priority order.
    ///
    /// `flags` = `[tp_some, pp_dense_some, ep_some, pp_qwen35]`,
    /// matching the early-return order in `generate()` (daemon.rs ~6891–7374):
    ///   [0] tp_some     → `m.tp.is_some()`
    ///   [1] pp_dense    → `m.pp_dense.is_some()`
    ///   [2] ep_some     → `m.ep.is_some()`
    ///   [3] pp_qwen35   → `m.pp > 1`
    #[allow(dead_code)]
    pub fn priority(flags: &[bool; 4]) -> ModelParallelKind {
        match flags {
            [true, _, _, _] => ModelParallelKind::Tp,
            [_, true, _, _] => ModelParallelKind::PpDense,
            [_, _, true, _] => ModelParallelKind::Ep,
            [_, _, _, true] => ModelParallelKind::PpQwen35,
            _ => ModelParallelKind::Single,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kind_classifier_is_exhaustive_and_ordered() {
        // Real dispatch order from daemon.rs generate() (~6891–7374):
        // Tp wins over PpDense wins over Ep wins over PpQwen35 wins over Single.
        // flags = [tp_some, pp_dense_some, ep_some, pp_qwen35]
        assert_eq!(
            ModelParallelKind::priority(&[true, false, false, false]),
            ModelParallelKind::Tp
        );
        assert_eq!(
            ModelParallelKind::priority(&[false, true, false, false]),
            ModelParallelKind::PpDense
        );
        assert_eq!(
            ModelParallelKind::priority(&[false, false, true, false]),
            ModelParallelKind::Ep
        );
        assert_eq!(
            ModelParallelKind::priority(&[false, false, false, true]),
            ModelParallelKind::PpQwen35
        );
        assert_eq!(
            ModelParallelKind::priority(&[false, false, false, false]),
            ModelParallelKind::Single
        );
    }
}
