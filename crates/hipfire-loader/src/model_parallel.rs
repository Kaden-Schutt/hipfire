// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt

//! Classifies which parallelism axis a loaded model uses, so daemon dispatch
//! can `match` one value instead of a chain of `is_some()` early-returns.
//! Inc 0: classifier only (borrows nothing). The owning enum lands in Inc N+1.
//!
//! # Real dispatch precedence (verified against daemon.rs `fn generate()`)
//!
//! The axis-first early-return order in `generate()` (daemon.rs):
//!   1. `m.parallel` is Tp                   → TP path
//!   2. `m.parallel` is Pp(Dense)            → dense-PP path
//!   3. `m.parallel` is Ep(_)               → EP path (Task 5: migrated from m.ep)
//!   4. `m.parallel` is Pp(ArchResident(_)) → qwen35 pipeline-parallel (Task 6)
//!   5. (everything else)                    → Single
//!
//! NOTE: The brief guessed `ep > tp > pp_dense > pp_qwen35 > single`.
//! The REAL order is `tp > pp_dense > ep > pp_qwen35 > single`.
//! Both `priority()` and the test reflect the real order.

use hipfire_runtime::multi_gpu::Gpus;
use hipfire_runtime::{pp_serve::PpModel, tp_serve::TpModel};
use crate::EpState;

/// Which parallelism axis is active for a loaded model.
///
/// Variants are ordered by dispatch priority (highest → lowest).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ModelParallelKind {
    /// Tensor-parallel dense multi-GPU (`m.tp.is_some()`).
    Tp,
    /// Pipeline-parallel dense multi-GPU (`m.parallel` is `Pp(Dense)`).
    PpDense,
    /// Expert-parallel MoE multi-GPU (`m.parallel` is `Ep(_)`).
    Ep,
    /// Pipeline-parallel qwen35 arch-resident (`m.parallel` is `Pp(ArchResident)`).
    PpQwen35,
    /// No multi-GPU axis — standard single-GPU path.
    Single,
}

impl ModelParallelKind {
    /// Classify from the four axis-present flags in dispatch priority order.
    ///
    /// `flags` = `[tp_some, pp_dense_some, ep_some, pp_qwen35]`,
    /// matching the early-return order in `generate()` (daemon.rs):
    ///   [0] tp_some     → `m.parallel` is `Tp`
    ///   [1] pp_dense    → `m.parallel` is `Pp(Dense)`
    ///   [2] ep_some     → `m.parallel` is `Ep(_)` (Task 5)
    ///   [3] pp_qwen35   → `m.parallel` is `Pp(ArchResident(_))` (Task 6)
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

/// Where a loaded model runs — the parallelism axis. No variant names a model.
pub enum ModelParallel {
    Single,
    Tp(TpModel),
    Pp(PipelineImpl),
    Ep(EpState),
}

/// PP is one axis, two implementations.
pub enum PipelineImpl {
    Dense(PpModel),      // generic dense-llama PP driver (self-contained)
    ArchResident(Gpus),  // model in ModelState; only the mesh here (qwen35 today)
}

pub(crate) fn kind_is_pipelined(k: ModelParallelKind) -> bool {
    matches!(k, ModelParallelKind::PpDense | ModelParallelKind::PpQwen35)
}

impl ModelParallel {
    pub fn kind(&self) -> ModelParallelKind {
        match self {
            ModelParallel::Single => ModelParallelKind::Single,
            ModelParallel::Tp(_) => ModelParallelKind::Tp,
            ModelParallel::Pp(PipelineImpl::Dense(_)) => ModelParallelKind::PpDense,
            ModelParallel::Pp(PipelineImpl::ArchResident(_)) => ModelParallelKind::PpQwen35,
            ModelParallel::Ep(_) => ModelParallelKind::Ep,
        }
    }
    pub fn is_pipelined(&self) -> bool {
        kind_is_pipelined(self.kind())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kind_pipelined_truth_table() {
        use ModelParallelKind::*;
        assert!(kind_is_pipelined(PpDense));
        assert!(kind_is_pipelined(PpQwen35));
        assert!(!kind_is_pipelined(Single));
        assert!(!kind_is_pipelined(Tp));
        assert!(!kind_is_pipelined(Ep));
    }

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
