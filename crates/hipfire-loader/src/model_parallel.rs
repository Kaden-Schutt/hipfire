// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt

//! Classifies which parallelism axis a loaded model uses, so daemon dispatch
//! can `match` one value instead of a chain of `is_some()` early-returns.
//!
//! # Real dispatch precedence (verified against daemon.rs `fn generate()`)
//!
//! `generate()` does one `match m.parallel.kind()` (Task 7) with this order:
//!   1. `ModelParallel::Tp`                  → TP path   (dense_serve_via_ar_generate)
//!   2. `ModelParallel::Pp(Dense)`           → PP-dense  (dense_serve_via_ar_generate)
//!   3. `ModelParallel::Ep(_)`              → EP path   (generate_ep)
//!   4. `ModelParallel::Pp(ArchResident(_))`→ qwen35 PP (fall-through)
//!   5. `ModelParallel::Single`             → single-GPU (fall-through)
//!
//! Real order: `Tp > PpDense > Ep > PpQwen35 > Single`.

use hipfire_runtime::multi_gpu::Gpus;
use hipfire_runtime::{pp_serve::PpModel, tp_serve::TpModel};
use crate::EpState;

/// Which parallelism axis is active for a loaded model.
///
/// Variants are ordered by dispatch priority (highest → lowest).
/// Produced by `ModelParallel::kind()`; used in the `match` dispatch head
/// in `generate()` (daemon.rs) and the bench-prefill guard.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ModelParallelKind {
    /// Tensor-parallel dense multi-GPU (`ModelParallel::Tp`).
    Tp,
    /// Pipeline-parallel dense multi-GPU (`ModelParallel::Pp(Dense)`).
    PpDense,
    /// Expert-parallel MoE multi-GPU (`ModelParallel::Ep(_)`).
    Ep,
    /// Pipeline-parallel qwen35 arch-resident (`ModelParallel::Pp(ArchResident(…))`).
    PpQwen35,
    /// No multi-GPU axis — standard single-GPU path.
    Single,
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
    fn bench_reject_matches_legacy_predicate() {
        use ModelParallelKind::*;
        // legacy: pp>1 || ep || tp.  pp>1 is true iff PpDense or PpQwen35 (dense-PP sets
        // pp=mesh>=2; PpModel::load errors for pp<2). New: !matches!(Single).
        let legacy = |k| matches!(k, PpDense | PpQwen35 | Ep | Tp);
        for k in [Single, Tp, PpDense, Ep, PpQwen35] {
            assert_eq!(legacy(k), k != Single, "kind {k:?}");
        }
    }
}
