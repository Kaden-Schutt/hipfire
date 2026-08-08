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

use crate::parallel_capability::RawParallelRequest;
use crate::EpState;
pub use hipfire_runtime::moe_plan::{
    MoEExecutionKind, MoEExecutionPolicy, MoEExecutionPolicyError,
};
use hipfire_runtime::multi_gpu::Gpus;
use hipfire_runtime::{pp_serve::PpModel, tp_serve::TpModel};

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

impl From<ModelParallelKind> for MoEExecutionKind {
    fn from(kind: ModelParallelKind) -> Self {
        match kind {
            ModelParallelKind::Tp => Self::Tp,
            ModelParallelKind::Ep => Self::Ep,
            ModelParallelKind::Single
            | ModelParallelKind::PpDense
            | ModelParallelKind::PpQwen35 => Self::Single,
        }
    }
}

impl ModelParallelKind {
    pub fn moe_execution_kind(self) -> MoEExecutionKind {
        self.into()
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
    Dense(PpModel),     // generic dense-llama PP driver (self-contained)
    ArchResident(Gpus), // model in ModelState; only the mesh here (qwen35 today)
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

    /// Return the concrete topology owned by this model.
    pub(crate) fn topology(&self) -> RawParallelRequest {
        match self {
            ModelParallel::Single => RawParallelRequest::new(1, 1, 1),
            ModelParallel::Tp(model) => RawParallelRequest::new(1, model.tp(), 1),
            ModelParallel::Pp(PipelineImpl::Dense(model)) => {
                RawParallelRequest::new(model.pp(), 1, 1)
            }
            ModelParallel::Pp(PipelineImpl::ArchResident(gpus)) => {
                RawParallelRequest::new(gpus.devices.len(), 1, 1)
            }
            ModelParallel::Ep(state) => RawParallelRequest::new(1, 1, state.gpus.devices.len()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use hipfire_runtime::multi_gpu::{DeviceMesh, DimKind};

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

    #[test]
    fn execution_policy_rejects_axis_mesh_mismatch() {
        let result =
            MoEExecutionPolicy::new(ModelParallelKind::Tp, DeviceMesh::rect(&[(DimKind::Ep, 2)]));

        assert!(result.is_err());
    }

    #[test]
    fn execution_policy_accepts_pp_only_single_mesh() {
        let policy = MoEExecutionPolicy::new(
            MoEExecutionKind::Single,
            DeviceMesh::rect(&[(DimKind::Pp, 2)]),
        )
        .unwrap();

        assert_eq!(policy.kind(), MoEExecutionKind::Single);
        assert_eq!(policy.mesh().size_of(DimKind::Pp), 2);
    }

    #[test]
    fn execution_policy_accepts_tp_only_mesh() {
        let policy =
            MoEExecutionPolicy::new(MoEExecutionKind::Tp, DeviceMesh::rect(&[(DimKind::Tp, 2)]))
                .unwrap();

        assert_eq!(policy.kind(), MoEExecutionKind::Tp);
    }

    #[test]
    fn execution_policy_accepts_ep_only_mesh() {
        let policy =
            MoEExecutionPolicy::new(MoEExecutionKind::Ep, DeviceMesh::rect(&[(DimKind::Ep, 2)]))
                .unwrap();

        assert_eq!(policy.kind(), MoEExecutionKind::Ep);
    }

    #[test]
    fn execution_policy_ignores_pp_for_effective_moe_axis() {
        assert!(MoEExecutionPolicy::new(
            MoEExecutionKind::Tp,
            DeviceMesh::rect(&[(DimKind::Pp, 2), (DimKind::Tp, 2)])
        )
        .is_ok());
        assert!(MoEExecutionPolicy::new(
            MoEExecutionKind::Ep,
            DeviceMesh::rect(&[(DimKind::Pp, 2), (DimKind::Ep, 2)])
        )
        .is_ok());
    }

    #[test]
    fn execution_policy_accepts_size_one_axis() {
        assert!(MoEExecutionPolicy::new(
            MoEExecutionKind::Tp,
            DeviceMesh::rect(&[(DimKind::Tp, 1)])
        )
        .is_ok());
        assert!(MoEExecutionPolicy::new(
            MoEExecutionKind::Ep,
            DeviceMesh::rect(&[(DimKind::Ep, 1)])
        )
        .is_ok());
    }

    #[test]
    fn execution_policy_rejects_tp_ep_mesh() {
        assert!(MoEExecutionPolicy::new(
            MoEExecutionKind::Tp,
            DeviceMesh::rect(&[(DimKind::Tp, 2), (DimKind::Ep, 3)])
        )
        .is_err());
        assert!(MoEExecutionPolicy::new(
            MoEExecutionKind::Ep,
            DeviceMesh::rect(&[(DimKind::Tp, 2), (DimKind::Ep, 3)])
        )
        .is_err());
    }

    #[test]
    fn execution_policy_error_format_identifies_required_and_effective_axes() {
        let error = MoEExecutionPolicy::new(
            MoEExecutionKind::Tp,
            DeviceMesh::rect(&[(DimKind::Tp, 2), (DimKind::Ep, 3)]),
        )
        .unwrap_err();

        assert_eq!(
            error.to_string(),
            "MoE execution kind Tp requires TP=2 and rejects effective EP=3; mesh axes: TP=2, EP=3"
        );
    }

    #[test]
    fn moe_execution_kind_matches_topology() {
        use ModelParallelKind::*;

        assert_eq!(MoEExecutionKind::from(Single), MoEExecutionKind::Single);
        assert_eq!(MoEExecutionKind::from(PpDense), MoEExecutionKind::Single);
        assert_eq!(MoEExecutionKind::from(PpQwen35), MoEExecutionKind::Single);
        assert_eq!(MoEExecutionKind::from(Tp), MoEExecutionKind::Tp);
        assert_eq!(MoEExecutionKind::from(Ep), MoEExecutionKind::Ep);
    }
}
