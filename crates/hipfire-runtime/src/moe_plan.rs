// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt

//! Runtime-owned MoE execution policy and common program lowering.

use std::fmt;

use hipfire_dispatch::context::DispatchCtx;
use hipfire_dispatch::families::moe::{
    ExpertExecutionPlan, RouterPlan, RouterSelection, MOE_GROUPED_BLOCK_M,
};
use hipfire_dispatch::pipeline::{
    execute_steps_mesh, execute_steps_parallel, GemvInput, MoeActivationVariant, MoeProj,
    QwenDownMode, Step, StepCollective,
};
use hipfire_dispatch::types::DispatchError;
use rdna_compute::{DType, GpuTensor};

use crate::multi_gpu::{DeviceMesh, DimKind};
use crate::weight_manifest::{
    ExpertExecutionIdentity, ExpertGroupPlan, ExpertParallelism, ExpertPostCombineAllReduce,
};

/// Device-pointer table ABI for the MQ2 batched indexed protocol: the
/// `_batched_k4` kernels read `expert_ptrs` as `const unsigned long long*`
/// (see `kernels/src/gemv_mq2g256_lloyd_moe_{gate_up_indexed_batched_k4,
/// down_indexed_batched_k4}.hip`), so each expert entry is 8 bytes. This is
/// the repository's device-pointer representation — never host pointer size
/// by assumption.
const DEVICE_POINTER_BYTES: usize = 8;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MoEExecutionKind {
    Single,
    Tp,
    Ep,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum MoEExecutionPolicyError {
    MissingRequiredAxis {
        kind: MoEExecutionKind,
        required_axis: DimKind,
        mesh_axes: Vec<(DimKind, usize)>,
    },
    CompetingAxis {
        kind: MoEExecutionKind,
        required_axis: DimKind,
        required_size: usize,
        effective_axis: DimKind,
        effective_size: usize,
        mesh_axes: Vec<(DimKind, usize)>,
    },
    SingleHasEffectiveAxis {
        effective_axis: DimKind,
        effective_size: usize,
        mesh_axes: Vec<(DimKind, usize)>,
    },
}

fn axis_name(axis: DimKind) -> &'static str {
    match axis {
        DimKind::Pp => "PP",
        DimKind::Tp => "TP",
        DimKind::Ep => "EP",
    }
}

fn format_axes(axes: &[(DimKind, usize)]) -> String {
    if axes.is_empty() {
        "none".to_owned()
    } else {
        axes.iter()
            .map(|(axis, size)| format!("{}={size}", axis_name(*axis)))
            .collect::<Vec<_>>()
            .join(", ")
    }
}

impl fmt::Display for MoEExecutionPolicyError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MissingRequiredAxis {
                kind,
                required_axis,
                mesh_axes,
            } => write!(
                f,
                "MoE execution kind {kind:?} requires {required} axis, but no effective {required} axis is present; mesh axes: {axes}",
                required = axis_name(*required_axis),
                axes = format_axes(mesh_axes),
            ),
            Self::CompetingAxis {
                kind,
                required_axis,
                required_size,
                effective_axis,
                effective_size,
                mesh_axes,
            } => write!(
                f,
                "MoE execution kind {kind:?} requires {required}={required_size} and rejects effective {effective}={effective_size}; mesh axes: {axes}",
                required = axis_name(*required_axis),
                effective = axis_name(*effective_axis),
                axes = format_axes(mesh_axes),
            ),
            Self::SingleHasEffectiveAxis {
                effective_axis,
                effective_size,
                mesh_axes,
            } => write!(
                f,
                "MoE execution kind Single requires no effective TP/EP axis, but found effective {effective}={effective_size}; mesh axes: {axes}",
                effective = axis_name(*effective_axis),
                axes = format_axes(mesh_axes),
            ),
        }
    }
}

impl std::error::Error for MoEExecutionPolicyError {}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MoEExecutionPolicy {
    kind: MoEExecutionKind,
    mesh: DeviceMesh,
}

impl MoEExecutionPolicy {
    pub fn single() -> Self {
        Self {
            kind: MoEExecutionKind::Single,
            mesh: DeviceMesh::single(),
        }
    }

    pub fn new<K>(kind: K, mesh: DeviceMesh) -> Result<Self, MoEExecutionPolicyError>
    where
        K: Into<MoEExecutionKind>,
    {
        let kind = kind.into();
        let axes: Vec<(DimKind, usize)> = mesh
            .axes()
            .iter()
            .map(|axis| (axis.kind, axis.size))
            .collect();
        match kind {
            MoEExecutionKind::Single => {
                if let Some(&(effective_axis, effective_size)) = axes
                    .iter()
                    .find(|(axis, _)| matches!(axis, DimKind::Tp | DimKind::Ep))
                {
                    return Err(MoEExecutionPolicyError::SingleHasEffectiveAxis {
                        effective_axis,
                        effective_size,
                        mesh_axes: axes,
                    });
                }
            }
            MoEExecutionKind::Tp => validate_axis(kind, DimKind::Tp, DimKind::Ep, &axes)?,
            MoEExecutionKind::Ep => validate_axis(kind, DimKind::Ep, DimKind::Tp, &axes)?,
        }
        Ok(Self { kind, mesh })
    }

    pub fn kind(&self) -> MoEExecutionKind {
        self.kind
    }

    pub fn mesh(&self) -> &DeviceMesh {
        &self.mesh
    }

    pub fn axis(&self) -> Option<DimKind> {
        match self.kind {
            MoEExecutionKind::Single => None,
            MoEExecutionKind::Tp => Some(DimKind::Tp),
            MoEExecutionKind::Ep => Some(DimKind::Ep),
        }
    }

    pub fn rank_count(&self) -> usize {
        self.axis().map_or(1, |axis| self.mesh.size_of(axis))
    }
}

fn validate_axis(
    kind: MoEExecutionKind,
    required_axis: DimKind,
    competing_axis: DimKind,
    mesh_axes: &[(DimKind, usize)],
) -> Result<(), MoEExecutionPolicyError> {
    let Some(&(_, required_size)) = mesh_axes.iter().find(|(axis, _)| *axis == required_axis)
    else {
        return Err(MoEExecutionPolicyError::MissingRequiredAxis {
            kind,
            required_axis,
            mesh_axes: mesh_axes.to_vec(),
        });
    };
    if let Some(&(_, effective_size)) = mesh_axes.iter().find(|(axis, _)| *axis == competing_axis) {
        return Err(MoEExecutionPolicyError::CompetingAxis {
            kind,
            required_axis,
            required_size,
            effective_axis: competing_axis,
            effective_size,
            mesh_axes: mesh_axes.to_vec(),
        });
    }
    Ok(())
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MoeExecutorKind {
    SingleMesh,
    Parallel,
}

pub struct RoutedMoePhases<T> {
    pub router: Vec<T>,
    pub gate_up: Vec<T>,
    pub activation: Vec<T>,
    pub down: Vec<T>,
    pub combine: Vec<T>,
    pub finish: Vec<T>,
}

impl<T> RoutedMoePhases<T> {
    fn lengths(&self) -> [usize; 6] {
        [
            self.router.len(),
            self.gate_up.len(),
            self.activation.len(),
            self.down.len(),
            self.combine.len(),
            self.finish.len(),
        ]
    }

    fn into_steps(self) -> Vec<T> {
        let capacity = self.lengths().into_iter().sum();
        let mut steps = Vec::with_capacity(capacity);
        steps.extend(self.router);
        steps.extend(self.gate_up);
        steps.extend(self.activation);
        steps.extend(self.down);
        steps.extend(self.combine);
        steps.extend(self.finish);
        steps
    }
}

pub type RoutedMoeStepPhases<'a> = RoutedMoePhases<Step<'a>>;

/// The typed router/execution identity and per-rank phase programs of one MoE
/// group. The launch schedule (collectives, zeroing, conversion placement) is
/// derived exclusively from these concrete borrowed `Step` programs.
pub struct MoeProgramParts<'step> {
    pub router: RouterPlan<'step>,
    pub execution: ExpertExecutionPlan,
    pub ranks: Vec<RoutedMoeStepPhases<'step>>,
    /// Explicit deferred-combine marker: the expanded down producer has NO
    /// local combine because the architecture's next-layer fused consumer
    /// folds the partial (combine-next-RMS). Only this flag admits a
    /// zero-combine `ExpandedIndexed` program; ordinary programs must carry
    /// exactly one combine, and a deferred program carrying one is a
    /// double-add and is rejected. Never set together with a parallel policy.
    pub deferred_combine: bool,
}

/// A lowered MoE program sealed behind a private inner representation: the
/// launch schedule is derived exclusively from the concrete borrowed `Step`
/// programs by `lower_moe_steps`, so callers cannot authorize collectives,
/// zeroing, conversion, or reduction placement. The inner representation is
/// only constructible inside this module.
pub struct LoweredMoeProgram<'mesh, 'step> {
    inner: LoweredMoeProgramInner<'mesh, 'step>,
}

enum LoweredMoeProgramInner<'mesh, 'step> {
    Single {
        steps: Vec<Step<'step>>,
    },
    Parallel {
        mesh: &'mesh DeviceMesh,
        per_rank_steps: Vec<Vec<Step<'step>>>,
        collectives: Vec<StepCollective>,
        zero_before: Vec<bool>,
    },
}

// `Step` carries no `Debug`, so the sealed program prints schedule shape.
impl fmt::Debug for LoweredMoeProgram<'_, '_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.inner {
            LoweredMoeProgramInner::Single { steps } => f
                .debug_struct("LoweredMoeProgram")
                .field("kind", &"Single")
                .field("steps", &steps.len())
                .finish(),
            LoweredMoeProgramInner::Parallel {
                mesh,
                per_rank_steps,
                collectives,
                zero_before,
            } => f
                .debug_struct("LoweredMoeProgram")
                .field("kind", &"Parallel")
                .field("mesh", mesh)
                .field("per_rank_steps", &per_rank_steps.len())
                .field("collectives", collectives)
                .field("zero_before", zero_before)
                .finish(),
        }
    }
}

// Immutable inspection surface of the sealed lowered program. Read-only: no
// mutable refs, no raw `Step`/pointer/mesh exposure, no constructors or
// setters. Every accessor returns `None` for any invalid bound and never
// panics. Parallel schedules are shared vectors: the same `StepCollective`
// ref may serve every rank.
impl LoweredMoeProgram<'_, '_> {
    pub fn executor_kind(&self) -> MoeExecutorKind {
        match &self.inner {
            LoweredMoeProgramInner::Single { .. } => MoeExecutorKind::SingleMesh,
            LoweredMoeProgramInner::Parallel { .. } => MoeExecutorKind::Parallel,
        }
    }

    pub fn rank_count(&self) -> usize {
        match &self.inner {
            LoweredMoeProgramInner::Single { .. } => 1,
            LoweredMoeProgramInner::Parallel { per_rank_steps, .. } => per_rank_steps.len(),
        }
    }

    pub fn step_count(&self, rank: usize) -> Option<usize> {
        match &self.inner {
            LoweredMoeProgramInner::Single { steps } => (rank == 0).then_some(steps.len()),
            LoweredMoeProgramInner::Parallel { per_rank_steps, .. } => {
                per_rank_steps.get(rank).map(|steps| steps.len())
            }
        }
    }

    pub fn collective(&self, rank: usize, step: usize) -> Option<&StepCollective> {
        match &self.inner {
            // Single programs carry no parallel schedule vectors.
            LoweredMoeProgramInner::Single { .. } => None,
            LoweredMoeProgramInner::Parallel {
                per_rank_steps,
                collectives,
                ..
            } => per_rank_steps
                .get(rank)?
                .get(step)
                .and_then(|_| collectives.get(step)),
        }
    }

    pub fn zero_before(&self, rank: usize, step: usize) -> Option<bool> {
        match &self.inner {
            LoweredMoeProgramInner::Single { .. } => None,
            LoweredMoeProgramInner::Parallel {
                per_rank_steps,
                zero_before,
                ..
            } => per_rank_steps
                .get(rank)?
                .get(step)
                .and_then(|_| zero_before.get(step).copied()),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum MoeLowerError {
    GroupPolicyMismatch {
        group: String,
        layer: Option<usize>,
        parallelism: ExpertParallelism,
        policy: MoEExecutionKind,
        collective: Option<ExpertPostCombineAllReduce>,
    },
    GroupSizeMismatch {
        group: String,
        layer: Option<usize>,
        group_size: usize,
        policy_ranks: usize,
    },
    RankCountMismatch {
        group: String,
        layer: Option<usize>,
        expected: usize,
        actual: usize,
    },
    RankPhaseMismatch {
        group: String,
        layer: Option<usize>,
        rank: usize,
        expected: [usize; 6],
        actual: [usize; 6],
    },
    MissingPhase {
        group: String,
        layer: Option<usize>,
        phase: &'static str,
    },
    UnrecognizedDownStep {
        group: String,
        layer: Option<usize>,
        rank: usize,
        index: usize,
    },
    MultipleDownSteps {
        group: String,
        layer: Option<usize>,
        rank: usize,
    },
    DuplicateDownOp {
        group: String,
        layer: Option<usize>,
        rank: usize,
        index: usize,
        phase: &'static str,
    },
    MisplacedDownOp {
        group: String,
        layer: Option<usize>,
        rank: usize,
        index: usize,
        phase: &'static str,
    },
    DuplicateCombineOp {
        group: String,
        layer: Option<usize>,
        rank: usize,
        index: usize,
    },
    MisplacedCombineOp {
        group: String,
        layer: Option<usize>,
        rank: usize,
        index: usize,
        phase: &'static str,
    },
    UnexpectedCombineInversePerm {
        group: String,
        layer: Option<usize>,
        rank: usize,
        index: usize,
    },
    MissingCombineInversePerm {
        group: String,
        layer: Option<usize>,
        rank: usize,
        index: usize,
    },
    MissingScatter {
        group: String,
        layer: Option<usize>,
        rank: usize,
        index: usize,
    },
    DuplicateScatter {
        group: String,
        layer: Option<usize>,
        rank: usize,
        index: usize,
    },
    ScatterChainMismatch {
        group: String,
        layer: Option<usize>,
        rank: usize,
        index: usize,
        field: &'static str,
    },
    GroupedChainMismatch {
        group: String,
        layer: Option<usize>,
        rank: usize,
        index: usize,
        field: &'static str,
    },
    GroupedOpCountMismatch {
        group: String,
        layer: Option<usize>,
        rank: usize,
        op: &'static str,
        expected: usize,
        actual: usize,
    },
    ScatterSlotCountMismatch {
        group: String,
        layer: Option<usize>,
        rank: usize,
        index: usize,
        expected: usize,
        actual: usize,
    },
    ScatterBlockMismatch {
        group: String,
        layer: Option<usize>,
        rank: usize,
        index: usize,
        expected: usize,
        actual: usize,
    },
    ScatterExpertCountMismatch {
        group: String,
        layer: Option<usize>,
        rank: usize,
        index: usize,
        expected: usize,
        actual: usize,
    },
    ScatterMaxTotalMismatch {
        group: String,
        layer: Option<usize>,
        rank: usize,
        index: usize,
        field: &'static str,
        expected: usize,
        actual: usize,
    },
    StrayScatter {
        group: String,
        layer: Option<usize>,
        rank: usize,
        index: usize,
        phase: &'static str,
    },
    CombineCountMismatch {
        group: String,
        layer: Option<usize>,
        rank: usize,
        expected: usize,
        actual: usize,
    },
    UnrecognizedCombineStep {
        group: String,
        layer: Option<usize>,
        rank: usize,
        index: usize,
    },
    CombineDownSourceMismatch {
        group: String,
        layer: Option<usize>,
        rank: usize,
        index: usize,
    },
    CombineMetadataMismatch {
        group: String,
        layer: Option<usize>,
        rank: usize,
        index: usize,
        field: &'static str,
        expected: usize,
        actual: usize,
    },
    CombineOutputShapeMismatch {
        group: String,
        layer: Option<usize>,
        rank: usize,
        index: usize,
        expected: usize,
        actual: usize,
    },
    CombineAfterSelfCombiningDown {
        group: String,
        layer: Option<usize>,
        rank: usize,
        index: usize,
    },
    InvalidCombineDimensions {
        group: String,
        layer: Option<usize>,
        rank: usize,
        hidden: usize,
        batch_size: usize,
    },
    InvalidF32Shape {
        group: String,
        layer: Option<usize>,
        rank: usize,
        dim: usize,
    },
    MissingI64Conversion {
        group: String,
        layer: Option<usize>,
        rank: usize,
    },
    MisplacedI64Conversion {
        group: String,
        layer: Option<usize>,
        rank: usize,
        index: usize,
        phase: &'static str,
    },
    DuplicateI64Conversion {
        group: String,
        layer: Option<usize>,
        rank: usize,
        index: usize,
    },
    I64ConversionSourceMismatch {
        group: String,
        layer: Option<usize>,
        rank: usize,
        index: usize,
    },
    CapacityMismatch {
        group: String,
        layer: Option<usize>,
        rank: usize,
        index: usize,
        expected_bytes: usize,
        actual_bytes: usize,
    },
    I64DimensionMismatch {
        group: String,
        layer: Option<usize>,
        rank: usize,
        index: usize,
        expected: usize,
        actual: usize,
    },
    ArithmeticOverflow {
        group: String,
        layer: Option<usize>,
        rank: usize,
        index: usize,
        what: &'static str,
    },
    I64OnNonAdmittedAxis {
        group: String,
        layer: Option<usize>,
    },
    /// The deferred-expanded combine protocol (expanded down producer, zero
    /// local combine, architecture's next-layer fused consumer) is inherently
    /// rank-local: the consuming fused kernel folds the expanded partial on
    /// the same device, so a parallel axis has no valid all-reduce anchor.
    /// Refused instead of silently flattening a partial combine.
    DeferredCombineOnParallelAxis {
        group: String,
        layer: Option<usize>,
    },
    RankProtocolMismatch {
        group: String,
        layer: Option<usize>,
        rank: usize,
        expected: String,
        actual: String,
    },
    RouterIdentityMismatch {
        group: String,
        layer: Option<usize>,
        expected: String,
        actual: &'static str,
    },
    ExecutionIdentityMismatch {
        group: String,
        layer: Option<usize>,
        expected: String,
        actual: &'static str,
    },
    /// The plan's allowed-execution admission set is empty; lowering is
    /// refused before any membership or protocol check runs.
    EmptyAllowedExecutions {
        group: String,
        layer: Option<usize>,
    },
    /// The plan's allowed-execution admission set repeats a typed identity;
    /// `identity` is its canonical label.
    DuplicateAllowedExecutions {
        group: String,
        layer: Option<usize>,
        identity: &'static str,
    },
    /// A batched indexed I64 chain must contain exactly one indexed GateUp.
    BatchedI64GateUpCount {
        group: String,
        layer: Option<usize>,
        rank: usize,
        expected: usize,
        actual: usize,
    },
    /// A batched indexed I64 chain must contain exactly one `MoeActivation`.
    BatchedI64ActivationCount {
        group: String,
        layer: Option<usize>,
        rank: usize,
        expected: usize,
        actual: usize,
    },
    /// A batched indexed I64 chain field (batch_size, k_top, expert metadata,
    /// top-k index buffer, activation/down dataflow, or semantic phase)
    /// disagrees with the down step.
    BatchedI64ChainMismatch {
        group: String,
        layer: Option<usize>,
        rank: usize,
        index: usize,
        field: &'static str,
    },
    /// Activation row count must equal the checked `batch_size * k_top`
    /// product.
    BatchedI64ActivationRows {
        group: String,
        layer: Option<usize>,
        rank: usize,
        index: usize,
        expected: usize,
        actual: usize,
    },
    /// The batched I64 down output must be exactly `[batch_size, hidden]`.
    BatchedI64OutputShape {
        group: String,
        layer: Option<usize>,
        rank: usize,
        index: usize,
        expected: [usize; 2],
        actual: Vec<usize>,
    },
    /// MQ2 batched contractions must be 256-aligned (gate-up hidden and
    /// i64-down inter_local).
    BatchedI64Alignment {
        group: String,
        layer: Option<usize>,
        rank: usize,
        index: usize,
        field: &'static str,
        expected: usize,
        actual: usize,
    },
    /// Batched geometry (batch_size or k_top) must be nonzero; a zero value
    /// is never interpreted as the scalar protocol.
    BatchedI64ZeroGeometry {
        group: String,
        layer: Option<usize>,
        rank: usize,
        index: usize,
        field: &'static str,
    },
    /// A batched indexed I64 chain is MQ2G256Lloyd-only; a coherent chain
    /// with any other dtype is rejected at lowering, never deferred to a
    /// dispatch-time scalar fallback.
    BatchedI64Dtype {
        group: String,
        layer: Option<usize>,
        rank: usize,
        index: usize,
        expected: DType,
        actual: DType,
    },
    /// The routed k_top must not exceed the expert count.
    BatchedI64KTopExceedsExperts {
        group: String,
        layer: Option<usize>,
        rank: usize,
        index: usize,
        k_top: usize,
        n_experts: usize,
    },
    /// A batched indexed I64 geometry value or product cannot convert exactly
    /// to the kernel ABI integer width used by the MQ2 `_batched_k4`
    /// launchers (i32 kernargs for m/k/k_top; u32 grid-z for batch_size).
    BatchedI64AbiRange {
        group: String,
        layer: Option<usize>,
        rank: usize,
        index: usize,
        field: &'static str,
        max: usize,
        actual: usize,
    },
    /// A grouped chain's scatter is not in the router phase.
    MisplacedScatter {
        group: String,
        layer: Option<usize>,
        rank: usize,
        index: usize,
        phase: &'static str,
    },
    /// A grouped phase violates the binding grammar: router exactly
    /// `[MoeScatter]`; gate_up `[GivensRotateBatched?,
    /// GroupedMoeGemm(GateUp), MoeGateUpUnscatter]`; activation exactly
    /// `[MoeActivation]`; finish exactly empty.
    GroupedPhaseMismatch {
        group: String,
        layer: Option<usize>,
        rank: usize,
        phase: &'static str,
        expected: &'static str,
        index: usize,
    },
    /// A grouped permutation operation appears on a non-grouped protocol
    /// (indexed expanded, self-combining f32/i64).
    StrayGroupedOp {
        group: String,
        layer: Option<usize>,
        rank: usize,
        index: usize,
        phase: &'static str,
        op: &'static str,
    },
    /// A grouped chain's m_total_max is not a multiple of the frozen grouped
    /// block width (16): the tile count would truncate.
    GroupedTileAlignment {
        group: String,
        layer: Option<usize>,
        rank: usize,
        index: usize,
        m_total: usize,
        block_m: usize,
    },
    /// A grouped Paro Givens rotation dim must be a nonzero multiple of 128.
    GroupedGivensAlignment {
        group: String,
        layer: Option<usize>,
        rank: usize,
        index: usize,
        dim: usize,
        expected_alignment: usize,
    },
    /// A grouped chain's routed geometry must be nonzero before any chain or
    /// capacity arithmetic; a zero value is never a degenerate launch.
    GroupedZeroGeometry {
        group: String,
        layer: Option<usize>,
        rank: usize,
        index: usize,
        field: &'static str,
    },
    /// The routed k_top must not exceed the expert count.
    GroupedKTopExceedsExperts {
        group: String,
        layer: Option<usize>,
        rank: usize,
        index: usize,
        k_top: usize,
        n_experts: usize,
    },
}

impl fmt::Display for MoeLowerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::GroupPolicyMismatch {
                group,
                layer,
                parallelism,
                policy,
                collective,
                ..
            } => write!(
                f,
                "MoE group '{group}' layer {layer:?}: parallelism {parallelism:?} is incompatible with policy {policy:?} and collective {collective:?}"
            ),
            Self::GroupSizeMismatch {
                group,
                layer,
                group_size,
                policy_ranks,
                ..
            } => write!(
                f,
                "MoE group '{group}' layer {layer:?}: group_size={group_size} does not match policy ranks={policy_ranks}"
            ),
            Self::RankCountMismatch {
                group,
                layer,
                expected,
                actual,
                ..
            } => write!(
                f,
                "MoE group '{group}' layer {layer:?}: rank program count={actual} does not match expected={expected}"
            ),
            Self::RankPhaseMismatch {
                group,
                layer,
                rank,
                expected,
                actual,
                ..
            } => write!(
                f,
                "MoE group '{group}' layer {layer:?} rank {rank}: phase lengths {actual:?} differ from {expected:?}"
            ),
            Self::MissingPhase { group, layer, phase, .. } => {
                write!(f, "MoE group '{group}' layer {layer:?}: {phase} phase is empty")
            }
            Self::UnrecognizedDownStep {
                group,
                layer,
                rank,
                index,
                ..
            } => write!(
                f,
                "MoE group '{group}' layer {layer:?} rank {rank}: step at absolute index {index} is not a recognized down projection"
            ),
            Self::MultipleDownSteps {
                group,
                layer,
                rank,
                ..
            } => write!(
                f,
                "MoE group '{group}' layer {layer:?} rank {rank}: down phase must contain exactly one down projection"
            ),
            Self::DuplicateDownOp {
                group,
                layer,
                rank,
                index,
                phase,
                ..
            } => write!(
                f,
                "MoE group '{group}' layer {layer:?} rank {rank}: duplicate down projection at absolute index {index} in {phase} phase"
            ),
            Self::MisplacedDownOp {
                group,
                layer,
                rank,
                index,
                phase,
                ..
            } => write!(
                f,
                "MoE group '{group}' layer {layer:?} rank {rank}: down projection at absolute index {index} is misplaced in {phase} phase"
            ),
            Self::DuplicateCombineOp {
                group,
                layer,
                rank,
                index,
                ..
            } => write!(
                f,
                "MoE group '{group}' layer {layer:?} rank {rank}: duplicate combine step at absolute index {index}"
            ),
            Self::MisplacedCombineOp {
                group,
                layer,
                rank,
                index,
                phase,
                ..
            } => write!(
                f,
                "MoE group '{group}' layer {layer:?} rank {rank}: combine step at absolute index {index} is misplaced in {phase} phase"
            ),
            Self::UnexpectedCombineInversePerm {
                group,
                layer,
                rank,
                index,
                ..
            } => write!(
                f,
                "MoE group '{group}' layer {layer:?} rank {rank}: indexed expanded combine at absolute index {index} must not carry an inverse permutation"
            ),
            Self::MissingCombineInversePerm {
                group,
                layer,
                rank,
                index,
                ..
            } => write!(
                f,
                "MoE group '{group}' layer {layer:?} rank {rank}: grouped expanded combine at absolute index {index} requires an inverse permutation"
            ),
            Self::MissingScatter {
                group,
                layer,
                rank,
                index,
                ..
            } => write!(
                f,
                "MoE group '{group}' layer {layer:?} rank {rank}: grouped down at absolute index {index} has no preceding MoeScatter"
            ),
            Self::DuplicateScatter {
                group,
                layer,
                rank,
                index,
                ..
            } => write!(
                f,
                "MoE group '{group}' layer {layer:?} rank {rank}: duplicate MoeScatter step at absolute index {index}"
            ),
            Self::ScatterChainMismatch {
                group,
                layer,
                rank,
                index,
                field,
                ..
            } => write!(
                f,
                "MoE group '{group}' layer {layer:?} rank {rank}: MoeScatter at absolute index {index} {field} does not match the grouped down/combine chain"
            ),
            Self::GroupedChainMismatch {
                group,
                layer,
                rank,
                index,
                field,
                ..
            } => write!(
                f,
                "MoE group '{group}' layer {layer:?} rank {rank}: grouped chain {field} at absolute index {index} does not match the frozen scatter/gate-up/down contract"
            ),
            Self::GroupedOpCountMismatch {
                group,
                layer,
                rank,
                op,
                expected,
                actual,
                ..
            } => write!(
                f,
                "MoE group '{group}' layer {layer:?} rank {rank}: grouped chain requires exactly {expected} {op} op(s), found {actual}"
            ),
            Self::ScatterSlotCountMismatch {
                group,
                layer,
                rank,
                index,
                expected,
                actual,
                ..
            } => write!(
                f,
                "MoE group '{group}' layer {layer:?} rank {rank}: MoeScatter at absolute index {index} total_slots={actual} does not match checked batch_size*k_top={expected}"
            ),
            Self::ScatterBlockMismatch {
                group,
                layer,
                rank,
                index,
                expected,
                actual,
                ..
            } => write!(
                f,
                "MoE group '{group}' layer {layer:?} rank {rank}: MoeScatter at absolute index {index} block_m={actual} does not match the frozen grouped block width {expected}"
            ),
            Self::ScatterExpertCountMismatch {
                group,
                layer,
                rank,
                index,
                expected,
                actual,
                ..
            } => write!(
                f,
                "MoE group '{group}' layer {layer:?} rank {rank}: MoeScatter at absolute index {index} n_experts={expected} does not match grouped experts.n_experts={actual}"
            ),
            Self::ScatterMaxTotalMismatch {
                group,
                layer,
                rank,
                index,
                field,
                expected,
                actual,
                ..
            } => write!(
                f,
                "MoE group '{group}' layer {layer:?} rank {rank}: MoeScatter at absolute index {index} m_total_max={expected} does not match {field}.m_total={actual}"
            ),
            Self::StrayScatter {
                group,
                layer,
                rank,
                index,
                phase,
                ..
            } => write!(
                f,
                "MoE group '{group}' layer {layer:?} rank {rank}: stray MoeScatter at absolute index {index} in {phase} phase is not admitted by this protocol"
            ),
            Self::CombineCountMismatch {
                group,
                layer,
                rank,
                expected,
                actual,
                ..
            } => write!(
                f,
                "MoE group '{group}' layer {layer:?} rank {rank}: expanded down requires exactly {expected} combine step(s), found {actual}"
            ),
            Self::UnrecognizedCombineStep {
                group,
                layer,
                rank,
                index,
                ..
            } => write!(
                f,
                "MoE group '{group}' layer {layer:?} rank {rank}: step at absolute index {index} is not a recognized combine step"
            ),
            Self::CombineDownSourceMismatch {
                group,
                layer,
                rank,
                index,
                ..
            } => write!(
                f,
                "MoE group '{group}' layer {layer:?} rank {rank}: combine step at absolute index {index} does not consume the expanded down output"
            ),
            Self::CombineMetadataMismatch {
                group,
                layer,
                rank,
                index,
                field,
                expected,
                actual,
                ..
            } => write!(
                f,
                "MoE group '{group}' layer {layer:?} rank {rank}: combine step at absolute index {index} has {field}={actual}, expected {field}={expected} from the expanded down producer"
            ),
            Self::CombineOutputShapeMismatch {
                group,
                layer,
                rank,
                index,
                expected,
                actual,
                ..
            } => write!(
                f,
                "MoE group '{group}' layer {layer:?} rank {rank}: combine output at absolute index {index} has logical shape dimension {actual}, expected {expected}"
            ),
            Self::CombineAfterSelfCombiningDown {
                group,
                layer,
                rank,
                index,
                ..
            } => write!(
                f,
                "MoE group '{group}' layer {layer:?} rank {rank}: combine step at absolute index {index} follows a self-combining down and would double-accumulate"
            ),
            Self::InvalidCombineDimensions {
                group,
                layer,
                rank,
                hidden,
                batch_size,
                ..
            } => write!(
                f,
                "MoE group '{group}' layer {layer:?} rank {rank}: combine hidden={hidden} and batch_size={batch_size} must both be nonzero"
            ),
            Self::InvalidF32Shape {
                group,
                layer,
                rank,
                dim,
                ..
            } => write!(
                f,
                "MoE group '{group}' layer {layer:?} rank {rank}: self-combining f32 down output has invalid logical shape dimension {dim}"
            ),
            Self::MissingI64Conversion {
                group,
                layer,
                rank,
                ..
            } => write!(
                f,
                "MoE group '{group}' layer {layer:?} rank {rank}: int64 down projection has no matching ConvertI64ToF32 step"
            ),
            Self::MisplacedI64Conversion {
                group,
                layer,
                rank,
                index,
                phase,
                ..
            } => write!(
                f,
                "MoE group '{group}' layer {layer:?} rank {rank}: ConvertI64ToF32 at absolute index {index} in {phase} phase is misplaced; i64 conversions must follow the down step in the finish phase"
            ),
            Self::DuplicateI64Conversion {
                group,
                layer,
                rank,
                index,
                ..
            } => write!(
                f,
                "MoE group '{group}' layer {layer:?} rank {rank}: duplicate ConvertI64ToF32 step at absolute index {index}"
            ),
            Self::I64ConversionSourceMismatch {
                group,
                layer,
                rank,
                index,
                ..
            } => write!(
                f,
                "MoE group '{group}' layer {layer:?} rank {rank}: ConvertI64ToF32 at absolute index {index} does not read the int64 down output"
            ),
            Self::CapacityMismatch {
                group,
                layer,
                rank,
                index,
                expected_bytes,
                actual_bytes,
                ..
            } => write!(
                f,
                "MoE group '{group}' layer {layer:?} rank {rank}: step at absolute index {index} requires {expected_bytes} bytes of capacity but the buffer provides {actual_bytes}"
            ),
            Self::I64DimensionMismatch {
                group,
                layer,
                rank,
                index,
                expected,
                actual,
                ..
            } => write!(
                f,
                "MoE group '{group}' layer {layer:?} rank {rank}: ConvertI64ToF32 dimension {actual} at absolute index {index} does not match int64 down buffer dimension {expected}"
            ),
            Self::ArithmeticOverflow {
                group,
                layer,
                rank,
                index,
                what,
                ..
            } => write!(
                f,
                "MoE group '{group}' layer {layer:?} rank {rank}: arithmetic overflow computing {what} at absolute index {index}"
            ),
            Self::I64OnNonAdmittedAxis { group, layer } => write!(
                f,
                "MoE group '{group}' layer {layer:?}: int64 down projection is not admitted on a single-rank axis"
            ),
            Self::DeferredCombineOnParallelAxis { group, layer } => write!(
                f,
                "MoE group '{group}' layer {layer:?}: the deferred-expanded combine protocol is rank-local (the next-layer fused consumer folds the expanded partial on the same device); a parallel axis has no combine anchor"
            ),
            Self::RankProtocolMismatch {
                group,
                layer,
                rank,
                expected,
                actual,
            } => write!(
                f,
                "MoE group '{group}' layer {layer:?}: rank {rank} protocol {actual} does not match rank 0 protocol {expected}"
            ),
            Self::RouterIdentityMismatch {
                group,
                layer,
                expected,
                actual,
            } => write!(
                f,
                "MoE group '{group}' layer {layer:?}: router identity '{expected}' does not match actual router selection '{actual}'"
            ),
            Self::ExecutionIdentityMismatch {
                group,
                layer,
                expected,
                actual,
            } => write!(
                f,
                "MoE group '{group}' layer {layer:?}: execution identity '{expected}' does not match actual execution plan '{actual}'"
            ),
            Self::EmptyAllowedExecutions { group, layer } => write!(
                f,
                "MoE group '{group}' layer {layer:?}: allowed execution identities is empty"
            ),
            Self::DuplicateAllowedExecutions {
                group,
                layer,
                identity,
            } => write!(
                f,
                "MoE group '{group}' layer {layer:?}: duplicate allowed execution identity {identity}"
            ),
            Self::BatchedI64GateUpCount {
                group,
                layer,
                rank,
                expected,
                actual,
            } => write!(
                f,
                "MoE group '{group}' layer {layer:?} rank {rank}: batched indexed I64 chain requires {expected} indexed gate-up step(s), found {actual}"
            ),
            Self::BatchedI64ActivationCount {
                group,
                layer,
                rank,
                expected,
                actual,
            } => write!(
                f,
                "MoE group '{group}' layer {layer:?} rank {rank}: batched indexed I64 chain requires {expected} activation step(s), found {actual}"
            ),
            Self::BatchedI64ChainMismatch {
                group,
                layer,
                rank,
                index,
                field,
            } => write!(
                f,
                "MoE group '{group}' layer {layer:?} rank {rank}: batched indexed I64 chain {field} mismatch at absolute index {index}"
            ),
            Self::BatchedI64ActivationRows {
                group,
                layer,
                rank,
                index,
                expected,
                actual,
            } => write!(
                f,
                "MoE group '{group}' layer {layer:?} rank {rank}: activation at absolute index {index} spans {actual} routed rows, expected {expected}"
            ),
            Self::BatchedI64OutputShape {
                group,
                layer,
                rank,
                index,
                expected,
                actual,
            } => write!(
                f,
                "MoE group '{group}' layer {layer:?} rank {rank}: batched I64 output at absolute index {index} has shape {actual:?}, expected [{}, {}]",
                expected[0], expected[1]
            ),
            Self::BatchedI64Alignment {
                group,
                layer,
                rank,
                index,
                field,
                expected,
                actual,
            } => write!(
                f,
                "MoE group '{group}' layer {layer:?} rank {rank}: batched I64 contraction {field} at absolute index {index} is {actual}, expected {expected}-aligned"
            ),
            Self::BatchedI64ZeroGeometry {
                group,
                layer,
                rank,
                index,
                field,
            } => write!(
                f,
                "MoE group '{group}' layer {layer:?} rank {rank}: batched indexed I64 {field} at absolute index {index} must be nonzero"
            ),
            Self::BatchedI64Dtype {
                group,
                layer,
                rank,
                index,
                expected,
                actual,
            } => write!(
                f,
                "MoE group '{group}' layer {layer:?} rank {rank}: batched indexed I64 dtype {actual:?} at absolute index {index}, expected {expected:?} (MQ2G256Lloyd only)"
            ),
            Self::BatchedI64KTopExceedsExperts {
                group,
                layer,
                rank,
                index,
                k_top,
                n_experts,
            } => write!(
                f,
                "MoE group '{group}' layer {layer:?} rank {rank}: batched indexed I64 k_top {k_top} at absolute index {index} exceeds n_experts {n_experts}"
            ),
            Self::BatchedI64AbiRange {
                group,
                layer,
                rank,
                index,
                field,
                max,
                actual,
            } => write!(
                f,
                "MoE group '{group}' layer {layer:?} rank {rank}: batched indexed I64 {field} = {actual} at absolute index {index} exceeds the kernel ABI width {max}"
            ),
            Self::MisplacedScatter {
                group,
                layer,
                rank,
                index,
                phase,
            } => write!(
                f,
                "MoE group '{group}' layer {layer:?} rank {rank}: scatter at absolute index {index} is misplaced in {phase} phase; grouped chains require router exactly [MoeScatter]"
            ),
            Self::GroupedPhaseMismatch {
                group,
                layer,
                rank,
                phase,
                expected,
                index,
            } => write!(
                f,
                "MoE group '{group}' layer {layer:?} rank {rank}: grouped {phase} phase at absolute index {index} violates the binding grammar; expected {expected}"
            ),
            Self::StrayGroupedOp {
                group,
                layer,
                rank,
                index,
                phase,
                op,
            } => write!(
                f,
                "MoE group '{group}' layer {layer:?} rank {rank}: {op} at absolute index {index} is not admitted in {phase} phase on a non-grouped protocol"
            ),
            Self::GroupedTileAlignment {
                group,
                layer,
                rank,
                index,
                m_total,
                block_m,
            } => write!(
                f,
                "MoE group '{group}' layer {layer:?} rank {rank}: m_total_max {m_total} at absolute index {index} is not a multiple of the grouped block width {block_m}"
            ),
            Self::GroupedGivensAlignment {
                group,
                layer,
                rank,
                index,
                dim,
                expected_alignment,
            } => write!(
                f,
                "MoE group '{group}' layer {layer:?} rank {rank}: Givens dim {dim} at absolute index {index} is not aligned to {expected_alignment}"
            ),
            Self::GroupedZeroGeometry {
                group,
                layer,
                rank,
                index,
                field,
            } => write!(
                f,
                "MoE group '{group}' layer {layer:?} rank {rank}: grouped {field} at absolute index {index} must be nonzero"
            ),
            Self::GroupedKTopExceedsExperts {
                group,
                layer,
                rank,
                index,
                k_top,
                n_experts,
            } => write!(
                f,
                "MoE group '{group}' layer {layer:?} rank {rank}: grouped k_top {k_top} at absolute index {index} exceeds n_experts {n_experts}"
            ),
        }
    }
}

impl std::error::Error for MoeLowerError {}

/// Direct label mapping from the typed router selection to the canonical
/// contract label. No hashing, interning, or aliases: `Hash` maps to `hash`
/// only; host-completed hash routing must declare `precomputed` explicitly.
fn canonical_router_identity(selection: RouterSelection) -> &'static str {
    match selection {
        RouterSelection::SoftmaxTopK => "softmax_topk",
        RouterSelection::SigmoidTopK => "sigmoid_topk",
        RouterSelection::BiasAwareTopK => "bias_aware_topk",
        RouterSelection::Hash => "hash",
        RouterSelection::Precomputed => "precomputed",
    }
}

/// Exhaustive typed mapping from the dispatch execution plan to the
/// manifest-owned identity enum. The match is total: a new dispatch variant
/// fails to compile here until the manifest enum covers it.
impl From<ExpertExecutionPlan> for ExpertExecutionIdentity {
    fn from(execution: ExpertExecutionPlan) -> Self {
        match execution {
            ExpertExecutionPlan::IndexedQuantized => Self::IndexedQuantized,
            ExpertExecutionPlan::GroupedQuantized => Self::GroupedQuantized,
            ExpertExecutionPlan::PerExpertFallback => Self::PerExpertFallback,
        }
    }
}

/// Deterministic display of an allowed-execution admission set: the canonical
/// labels in declaration order, joined by ", ". Never re-derived by parsing.
fn allowed_executions_label(executions: &[ExpertExecutionIdentity]) -> String {
    executions
        .iter()
        .map(|identity| identity.canonical_label())
        .collect::<Vec<_>>()
        .join(", ")
}

/// Validate that a group's declared semantic identities exactly match the
/// typed router selection and execution plan. The router check runs first, so
/// a router mismatch is always reported before an execution mismatch.
#[expect(
    clippy::result_large_err,
    reason = "MoeLowerError deliberately carries rich group/layer/rank diagnostics; boxing would fragment error reporting"
)]
fn validate_program_identity(
    group: &ExpertGroupPlan,
    router_selection: RouterSelection,
    execution: ExpertExecutionPlan,
) -> Result<(), MoeLowerError> {
    let expected_router = &group.router_identity;
    let actual_router = canonical_router_identity(router_selection);
    if expected_router != actual_router {
        return Err(MoeLowerError::RouterIdentityMismatch {
            group: group.group.clone(),
            layer: group.layer,
            expected: expected_router.clone(),
            actual: actual_router,
        });
    }
    // Exact typed membership: the actual plan must be one of the declared
    // allowed execution identities. No label parsing — the typed plan maps
    // exhaustively onto the manifest enum and membership is checked directly.
    let actual_execution = ExpertExecutionIdentity::from(execution);
    if !group.allowed_executions.contains(&actual_execution) {
        return Err(MoeLowerError::ExecutionIdentityMismatch {
            group: group.group.clone(),
            layer: group.layer,
            expected: allowed_executions_label(&group.allowed_executions),
            actual: actual_execution.canonical_label(),
        });
    }
    Ok(())
}

#[expect(
    clippy::result_large_err,
    reason = "MoeLowerError deliberately carries rich group/layer/rank diagnostics; boxing would fragment error reporting"
)]
pub fn select_moe_executor(
    group: &ExpertGroupPlan,
    policy: &MoEExecutionPolicy,
) -> Result<MoeExecutorKind, MoeLowerError> {
    if group.group_size != policy.rank_count() {
        return Err(MoeLowerError::GroupSizeMismatch {
            group: group.group.clone(),
            layer: group.layer,
            group_size: group.group_size,
            policy_ranks: policy.rank_count(),
        });
    }
    match (group.parallelism, policy.kind(), group.collective) {
        (ExpertParallelism::Single, MoEExecutionKind::Single, None) => {
            Ok(MoeExecutorKind::SingleMesh)
        }
        (
            ExpertParallelism::TensorParallel,
            MoEExecutionKind::Tp,
            Some(ExpertPostCombineAllReduce::TensorParallel),
        )
        | (
            ExpertParallelism::ExpertParallel,
            MoEExecutionKind::Ep,
            Some(ExpertPostCombineAllReduce::ExpertParallel),
        ) => Ok(MoeExecutorKind::Parallel),
        _ => Err(MoeLowerError::GroupPolicyMismatch {
            group: group.group.clone(),
            layer: group.layer,
            parallelism: group.parallelism,
            policy: policy.kind(),
            collective: group.collective,
        }),
    }
}

/// Two tensors name the same buffer when they are the same borrowed object,
/// or when their non-null raw pointers and allocation sizes agree. Distinct
/// null test tensors are never equated.
fn same_buffer(left: &GpuTensor, right: &GpuTensor) -> bool {
    std::ptr::eq(left, right)
        || (!left.buf.as_ptr().is_null()
            && left.buf.as_ptr() == right.buf.as_ptr()
            && left.buf.size() == right.buf.size())
}

/// Private typed down-projection evidence classified globally from one rank's
/// concrete Steps across all six phase vectors:
/// - `ExpandedIndexed`: writes per-expert intermediates via the indexed
///   kernels; a separate combine (no inverse permutation) folds them.
/// - `ExpandedGrouped`: writes per-expert intermediates via the grouped WMMA
///   kernels after a MoeScatter; the combine carries the inverse permutation.
/// - `SelfCombiningF32`: the kernel folds the weighted combine into the EP
///   partial itself (f32).
/// - `SelfCombiningI64`: reproducible int64 accumulator; a later
///   `ConvertI64ToF32` converts the summed int64.
#[derive(Clone, Copy)]
enum DownEvidence<'a> {
    ExpandedIndexed {
        out: &'a GpuTensor,
        k_top: usize,
        batch_size: usize,
    },
    ExpandedGrouped {
        y: &'a GpuTensor,
        x: &'a GpuTensor,
        sorted_slot_index: &'a GpuTensor,
        expert_tile_ids: &'a GpuTensor,
        k_top: usize,
        batch_size: usize,
        m_total: usize,
        n_experts: usize,
        dtype_tags: Option<&'a GpuTensor>,
        force_mq4_fp16: bool,
        paro_i8: bool,
        paro_i8_k8: bool,
    },
    SelfCombiningF32 {
        out: &'a GpuTensor,
    },
    SelfCombiningI64 {
        out: &'a GpuTensor,
        /// The authoritative routed batch: batch 1 keeps the scalar protocol,
        /// batch > 1 must satisfy the batched chain invariants, batch 0
        /// rejects.
        batch_size: usize,
    },
}

/// Private, parser-derived launch protocol for the whole group. Callers never
/// see or construct this: the schedule is derived exclusively from the
/// concrete borrowed `Step` programs. Zeroing is invariant per variant (the
/// accumulate/zero semantics of each protocol shape), so no zero flags are
/// carried; `TpI64` needs no conversion anchor because the i64 all-reduce
/// lands on the down step itself.
enum ValidatedProtocol {
    Single,
    ParallelF32 {
        anchor: usize,
        dim: usize,
    },
    TpI64 {
        down: usize,
        dim: usize,
    },
    EpLocalI64 {
        down: usize,
        convert: usize,
        dim: usize,
    },
}

/// Typed protocol-kind discriminator for rank agreement. Compared directly;
/// formatted labels are produced only for `RankProtocolMismatch` diagnostics.
#[derive(Clone, Copy, PartialEq, Eq)]
enum RankProtocolKind {
    ExpandedIndexed,
    /// Expanded down producer with ZERO local combine: the architecture's
    /// next-layer fused consumer folds the partial (combine-next-RMS). Only
    /// admitted when `MoeProgramParts.deferred_combine` explicitly marks the
    /// deferred consumption; never produced by an ordinary program.
    ExpandedIndexedDeferred,
    ExpandedGrouped,
    SelfCombiningF32,
    SelfCombiningI64,
}

/// Exact pointer-free activation signature for the grouped cross-rank
/// comparison: every live scalar launch/numerical discriminator exposed by
/// each `MoeActivationVariant`, with buffer pointers excluded (presence or
/// exact bits only).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum GroupedActivationSignature {
    /// Minimax fused silu·mul·rotate: only the AWQ-scale presence is a
    /// launch discriminator (the scale tensor itself is rank-local).
    MinimaxFused { awq: bool },
    /// Ds4 clamp+rotate: the exact f32 swiglu limit travels by bit pattern.
    Ds4ClampRotate { swiglu_limit_bits: u32 },
    /// Qwen AWQ indexed: no scalar launch discriminators (awq_ptrs and
    /// topk_indices are rank-local pointer tables).
    QwenAwqIndexed,
    /// Qwen Paro: krot is the live scalar launch discriminator (pairs/theta/
    /// scales are rank-local sidecars).
    QwenParo { krot: usize },
}

fn grouped_activation_signature(step: &Step<'_>) -> GroupedActivationSignature {
    match step {
        Step::MoeActivation {
            variant: MoeActivationVariant::MinimaxFused { awq_scale },
            ..
        } => GroupedActivationSignature::MinimaxFused {
            awq: awq_scale.is_some(),
        },
        Step::MoeActivation {
            variant: MoeActivationVariant::Ds4ClampRotate { swiglu_limit },
            ..
        } => GroupedActivationSignature::Ds4ClampRotate {
            swiglu_limit_bits: swiglu_limit.to_bits(),
        },
        Step::MoeActivation {
            variant: MoeActivationVariant::QwenAwqIndexed { .. },
            ..
        } => GroupedActivationSignature::QwenAwqIndexed,
        Step::MoeActivation {
            variant: MoeActivationVariant::QwenParo { krot, .. },
            ..
        } => GroupedActivationSignature::QwenParo { krot: *krot },
        _ => unreachable!("activation pin only collects MoeActivation"),
    }
}

/// Typed gate-up or down grouped-GEMM launch controls: dtype, the three
/// kernel-selection levers, and the independent per-expert tag presence.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct GroupedGemmControls {
    dtype: DType,
    force_mq4_fp16: bool,
    paro_i8: bool,
    paro_i8_k8: bool,
    tags: bool,
}

/// Typed cross-rank signature of a validated grouped chain. Every scalar is
/// rank-invariant: disagreeing ranks could preserve the flattened collective
/// dim (batch*hidden) while corrupting per-rank kernel geometry. Populated
/// only from the per-rank grouped validation; rank-local buffer pointers are
/// intentionally absent. Derived values (activation_rows == batch*k_top,
/// Givens dim == expert_k) are carried as validated evidence alongside their
/// sources.
#[derive(Clone, Copy, PartialEq, Eq)]
struct GroupedSignature {
    batch_size: usize,
    k_top: usize,
    n_experts: usize,
    /// expert_m == unscatter/activation inter == grouped-down input width.
    inter: usize,
    /// expert_k == gate-up input == grouped-down output == combine hidden.
    hidden: usize,
    m_total: usize,
    block_m: usize,
    activation: GroupedActivationSignature,
    /// Routed rows == batch_size * k_top.
    activation_rows: usize,
    /// (krot, dim) when the typed Paro Givens is present.
    givens: Option<(usize, usize)>,
    gate_up: GroupedGemmControls,
    down: GroupedGemmControls,
}

/// Typed cross-rank signature of a validated batched indexed I64 chain
/// (`batch_size > 1`). Every scalar here is rank-invariant: disagreeing ranks
/// could otherwise preserve the flattened collective dim while corrupting the
/// per-rank kernel layout and the reduction. Populated only from the
/// per-rank batched chain validation (never for scalar chains). Buffer
/// pointers are intentionally absent — buffers are rank-local. The output
/// hidden is NOT independently sourced: the per-rank exact-shape check binds
/// `out.shape[1]` to expert_k, and `inter`/`activation_rows` are bound to
/// expert_m / batch*k_top by the per-rank equality checks; they still travel
/// in the signature as validated evidence.
#[derive(Clone, Copy, PartialEq, Eq)]
struct BatchedI64Signature {
    batch_size: usize,
    k_top: usize,
    n_experts: usize,
    /// expert_m == the activation's inter (per-rank equality enforced).
    inter: usize,
    /// expert_k == the gate-up hidden == the i64 output hidden.
    hidden: usize,
    dtype: DType,
    /// Routed rows == batch_size * k_top == the activation's rows.
    activation_rows: usize,
}

/// One rank's parsed protocol evidence, compared across ranks for agreement
/// on dimensions, protocol, and absolute indices. Tensor validation happens
/// locally during parsing; only the typed kind and absolute schedule indices
/// survive into the compared record.
struct RankProtocol {
    kind: RankProtocolKind,
    down_index: usize,
    combine_index: Option<usize>,
    convert_index: Option<usize>,
    dim: usize,
    /// Cross-rank signature for validated batched indexed I64 chains
    /// (`batch_size > 1`); `None` for every other protocol.
    batched: Option<BatchedI64Signature>,
    /// Cross-rank signature for validated grouped chains; `None` for every
    /// other protocol.
    grouped: Option<GroupedSignature>,
}

/// Diagnostic label for `RankProtocolMismatch`; not used for agreement.
fn rank_protocol_label(protocol: &RankProtocol) -> String {
    let kind = match protocol.kind {
        RankProtocolKind::ExpandedIndexed => "expanded_indexed",
        RankProtocolKind::ExpandedIndexedDeferred => "expanded_indexed_deferred",
        RankProtocolKind::ExpandedGrouped => "expanded_grouped",
        RankProtocolKind::SelfCombiningF32 => "self_combining_f32",
        RankProtocolKind::SelfCombiningI64 => "self_combining_i64",
    };
    let batched = match &protocol.batched {
        Some(sig) => format!(
            " batched[batch={} k_top={} n_experts={} inter={} hidden={} dtype={:?} rows={}]",
            sig.batch_size,
            sig.k_top,
            sig.n_experts,
            sig.inter,
            sig.hidden,
            sig.dtype,
            sig.activation_rows
        ),
        None => String::new(),
    };
    let grouped = match &protocol.grouped {
        Some(sig) => format!(
            " grouped[batch={} k_top={} n_experts={} inter={} hidden={} m_total={} block_m={} act={:?} rows={} givens={:?} gu={:?} dn={:?}]",
            sig.batch_size,
            sig.k_top,
            sig.n_experts,
            sig.inter,
            sig.hidden,
            sig.m_total,
            sig.block_m,
            sig.activation,
            sig.activation_rows,
            sig.givens,
            sig.gate_up,
            sig.down
        ),
        None => String::new(),
    };
    format!(
        "{kind} down={} combine={:?} convert={:?} dim={}{batched}{grouped}",
        protocol.down_index, protocol.combine_index, protocol.convert_index, protocol.dim
    )
}

fn same_rank_protocol(a: &RankProtocol, b: &RankProtocol) -> bool {
    a.kind == b.kind
        && a.down_index == b.down_index
        && a.combine_index == b.combine_index
        && a.convert_index == b.convert_index
        && a.dim == b.dim
        && a.batched == b.batched
        && a.grouped == b.grouped
}

/// One rank's globally classified protocol-bearing operations, each with its
/// absolute index and semantic phase identity.
struct RankOps<'a> {
    downs: Vec<(usize, &'static str, DownEvidence<'a>)>,
    combines: Vec<(usize, &'static str, &'a Step<'a>)>,
    scatters: Vec<(usize, &'static str, &'a Step<'a>)>,
    gate_ups: Vec<(usize, &'static str, &'a Step<'a>)>,
    indexed_gate_ups: Vec<(usize, &'static str, &'a Step<'a>)>,
    /// Specialized qwen indexed gate-up (`Step::MoeGateUpIndexed`): the
    /// mandatory pairing partner for indexed Paro pre-rotation Givens.
    specialized_gate_ups: Vec<(usize, &'static str, &'a Step<'a>)>,
    gate_up_unscatters: Vec<(usize, &'static str, &'a Step<'a>)>,
    givens: Vec<(usize, &'static str, &'a Step<'a>)>,
    activations: Vec<(usize, &'static str, &'a Step<'a>)>,
    conversions: Vec<(usize, &'static str, &'a Step<'a>)>,
}

/// Globally classify every protocol-bearing down projection, combine step,
/// MoeScatter, and `ConvertI64ToF32` across all six phase vectors. Grouped
/// `GateUp` GEMMs are not down operations.
fn classify_rank_ops<'a>(phases: &'a RoutedMoePhases<Step<'a>>) -> RankOps<'a> {
    let mut ops = RankOps {
        downs: Vec::new(),
        combines: Vec::new(),
        scatters: Vec::new(),
        gate_ups: Vec::new(),
        indexed_gate_ups: Vec::new(),
        specialized_gate_ups: Vec::new(),
        activations: Vec::new(),
        givens: Vec::new(),
        gate_up_unscatters: Vec::new(),
        conversions: Vec::new(),
    };
    let mut offset = 0usize;
    for (phase, steps) in [
        ("router", &phases.router),
        ("gate_up", &phases.gate_up),
        ("activation", &phases.activation),
        ("down", &phases.down),
        ("combine", &phases.combine),
        ("finish", &phases.finish),
    ] {
        for (index, step) in steps.iter().enumerate() {
            let absolute = offset + index;
            match step {
                Step::IndexedMoeGemv {
                    which: MoeProj::DownExpanded,
                    out,
                    k_top,
                    batch_size,
                    ..
                }
                | Step::MoeDownIndexed {
                    mode: QwenDownMode::Expanded,
                    out,
                    k_top,
                    batch_size,
                    ..
                } => ops.downs.push((
                    absolute,
                    phase,
                    DownEvidence::ExpandedIndexed {
                        out,
                        k_top: *k_top,
                        batch_size: *batch_size,
                    },
                )),
                Step::GroupedMoeGemm {
                    which: MoeProj::DownExpanded,
                    y,
                    x,
                    sorted_slot_index,
                    expert_tile_ids,
                    k_top,
                    batch_size,
                    m_total,
                    dtype_tags,
                    force_mq4_fp16,
                    paro_i8,
                    paro_i8_k8,
                    experts,
                    ..
                } => ops.downs.push((
                    absolute,
                    phase,
                    DownEvidence::ExpandedGrouped {
                        y,
                        x,
                        sorted_slot_index,
                        expert_tile_ids,
                        k_top: *k_top,
                        batch_size: *batch_size,
                        m_total: *m_total,
                        n_experts: experts.n_experts,
                        dtype_tags: *dtype_tags,
                        force_mq4_fp16: *force_mq4_fp16,
                        paro_i8: *paro_i8,
                        paro_i8_k8: *paro_i8_k8,
                    },
                )),
                Step::GroupedMoeGemm {
                    which: MoeProj::GateUp { .. },
                    ..
                } => ops.gate_ups.push((absolute, phase, step)),
                Step::IndexedMoeGemv {
                    which: MoeProj::GateUp { .. },
                    ..
                } => ops.indexed_gate_ups.push((absolute, phase, step)),
                Step::MoeGateUpIndexed { .. } => {
                    ops.specialized_gate_ups.push((absolute, phase, step))
                }
                Step::MoeActivation { .. } => ops.activations.push((absolute, phase, step)),
                Step::MoeGateUpUnscatter { .. } => {
                    ops.gate_up_unscatters.push((absolute, phase, step))
                }
                Step::GivensRotateBatched { .. } => ops.givens.push((absolute, phase, step)),
                Step::IndexedMoeGemv {
                    which: MoeProj::DownResidual { .. },
                    out,
                    ..
                }
                | Step::MoeDownIndexed {
                    mode: QwenDownMode::ResidualScaled { .. },
                    out,
                    ..
                } => ops
                    .downs
                    .push((absolute, phase, DownEvidence::SelfCombiningF32 { out })),
                Step::IndexedMoeGemv {
                    which: MoeProj::DownResidualI64 { .. },
                    out,
                    batch_size,
                    ..
                } => ops.downs.push((
                    absolute,
                    phase,
                    DownEvidence::SelfCombiningI64 {
                        out,
                        batch_size: *batch_size,
                    },
                )),
                Step::MoeCombine { .. } => ops.combines.push((absolute, phase, step)),
                Step::MoeScatter { .. } => ops.scatters.push((absolute, phase, step)),
                Step::ConvertI64ToF32 { .. } => ops.conversions.push((absolute, phase, step)),
                _ => {}
            }
        }
        offset += steps.len();
    }
    ops
}

/// Reject grouped permutation machinery on non-grouped protocols: scatter,
/// gate-up unscatter, and grouped gate-up are never admitted on indexed
/// expanded or self-combining programs.
#[expect(
    clippy::result_large_err,
    reason = "MoeLowerError deliberately carries rich group/layer/rank diagnostics; boxing would fragment error reporting"
)]
fn reject_stray_grouped_ops(
    group: &ExpertGroupPlan,
    rank: usize,
    ops: &RankOps<'_>,
) -> Result<(), MoeLowerError> {
    if let Some(&(index, phase, _)) = ops.scatters.first() {
        return Err(MoeLowerError::StrayScatter {
            group: group.group.clone(),
            layer: group.layer,
            rank,
            index,
            phase,
        });
    }
    if let Some(&(index, phase, _)) = ops.gate_up_unscatters.first() {
        return Err(MoeLowerError::StrayGroupedOp {
            group: group.group.clone(),
            layer: group.layer,
            rank,
            index,
            phase,
            op: "gate_up_unscatter",
        });
    }
    if let Some(&(index, phase, _)) = ops.gate_ups.first() {
        return Err(MoeLowerError::StrayGroupedOp {
            group: group.group.clone(),
            layer: group.layer,
            rank,
            index,
            phase,
            op: "gate_up",
        });
    }
    if let Some(&(index, phase, _)) = ops.givens.first() {
        // Indexed Paro pre-rotation is admitted ONLY when paired with the
        // specialized indexed gate-up (`Step::MoeGateUpIndexed`): exactly one
        // Givens in the decode router phase or the indexed-prefill gate_up
        // phase, with exactly one specialized gate-up in the gate_up phase.
        // Every other Givens placement (stray, unpaired, extra, or relocated)
        // is grouped permutation machinery and stays rejected.
        let paired = ops.specialized_gate_ups.len() == 1
            && ops.specialized_gate_ups[0].1 == "gate_up"
            && ops.givens.len() == 1
            && (phase == "router" || phase == "gate_up");
        if !paired {
            return Err(MoeLowerError::StrayGroupedOp {
                group: group.group.clone(),
                layer: group.layer,
                rank,
                index,
                phase,
                op: "givens",
            });
        }
    }
    Ok(())
}

/// Checked byte count for a grouped operand: `elements * bytes_per_element`
/// with a contextual arithmetic-overflow error.
#[expect(
    clippy::result_large_err,
    reason = "MoeLowerError deliberately carries rich group/layer/rank diagnostics; boxing would fragment error reporting"
)]
fn checked_byte_count(
    group: &ExpertGroupPlan,
    rank: usize,
    index: usize,
    what: &'static str,
    elements: usize,
    bytes_per_element: usize,
) -> Result<usize, MoeLowerError> {
    elements
        .checked_mul(bytes_per_element)
        .ok_or_else(|| MoeLowerError::ArithmeticOverflow {
            group: group.group.clone(),
            layer: group.layer,
            rank,
            index,
            what,
        })
}

/// Checked element-count product: `a * b` with a contextual overflow error.
#[expect(
    clippy::result_large_err,
    reason = "MoeLowerError deliberately carries rich group/layer/rank diagnostics; boxing would fragment error reporting"
)]
fn checked_elems(
    group: &ExpertGroupPlan,
    rank: usize,
    index: usize,
    what: &'static str,
    a: usize,
    b: usize,
) -> Result<usize, MoeLowerError> {
    a.checked_mul(b)
        .ok_or_else(|| MoeLowerError::ArithmeticOverflow {
            group: group.group.clone(),
            layer: group.layer,
            rank,
            index,
            what,
        })
}

/// Checked element-count sum: `a + b` with a contextual overflow error.
#[expect(
    clippy::result_large_err,
    reason = "MoeLowerError deliberately carries rich group/layer/rank diagnostics; boxing would fragment error reporting"
)]
fn checked_add_elems(
    group: &ExpertGroupPlan,
    rank: usize,
    index: usize,
    what: &'static str,
    a: usize,
    b: usize,
) -> Result<usize, MoeLowerError> {
    a.checked_add(b)
        .ok_or_else(|| MoeLowerError::ArithmeticOverflow {
            group: group.group.clone(),
            layer: group.layer,
            rank,
            index,
            what,
        })
}

/// Actual-buffer capacity gate for one grouped operand. `cap` is the
/// pointer-identity byte capacity (production: `buf.size()`).
#[expect(
    clippy::result_large_err,
    reason = "MoeLowerError deliberately carries rich group/layer/rank diagnostics; boxing would fragment error reporting"
)]
fn require_grouped_capacity(
    group: &ExpertGroupPlan,
    rank: usize,
    index: usize,
    cap: &impl Fn(&GpuTensor) -> usize,
    tensor: &GpuTensor,
    needed: usize,
) -> Result<(), MoeLowerError> {
    let actual = cap(tensor);
    if actual < needed {
        return Err(MoeLowerError::CapacityMismatch {
            group: group.group.clone(),
            layer: group.layer,
            rank,
            index,
            expected_bytes: needed,
            actual_bytes: actual,
        });
    }
    Ok(())
}

#[expect(
    clippy::result_large_err,
    reason = "MoeLowerError deliberately carries rich group/layer/rank diagnostics; boxing would fragment error reporting"
)]
fn parse_rank_protocol<'a, F>(
    group: &ExpertGroupPlan,
    rank: usize,
    deferred_combine: bool,
    phases: &RoutedMoePhases<Step<'a>>,
    cap: &F,
) -> Result<RankProtocol, MoeLowerError>
where
    F: Fn(&GpuTensor) -> usize,
{
    let lengths = phases.lengths();
    let down_offset = lengths[0] + lengths[1] + lengths[2];
    let combine_offset = down_offset + lengths[3];
    let ops = classify_rank_ops(phases);

    // ── Global down requirements ─────────────────────────────────────────
    // Exactly one protocol-bearing down projection, in the semantic down
    // phase, occupying it entirely. Zero/duplicate/misplaced are rejected
    // contextually before any launch schedule is built.
    let (evidence, down_index) = match ops.downs.as_slice() {
        [] => {
            if lengths[3] == 0 {
                return Err(MoeLowerError::MissingPhase {
                    group: group.group.clone(),
                    layer: group.layer,
                    phase: "down",
                });
            }
            return Err(MoeLowerError::UnrecognizedDownStep {
                group: group.group.clone(),
                layer: group.layer,
                rank,
                index: down_offset,
            });
        }
        [_first, second, ..] => {
            return Err(MoeLowerError::DuplicateDownOp {
                group: group.group.clone(),
                layer: group.layer,
                rank,
                index: second.0,
                phase: second.1,
            });
        }
        [only] => {
            if only.1 != "down" {
                return Err(MoeLowerError::MisplacedDownOp {
                    group: group.group.clone(),
                    layer: group.layer,
                    rank,
                    index: only.0,
                    phase: only.1,
                });
            }
            if lengths[3] != 1 {
                return Err(MoeLowerError::MultipleDownSteps {
                    group: group.group.clone(),
                    layer: group.layer,
                    rank,
                });
            }
            (only.2, only.0)
        }
    };

    match evidence {
        DownEvidence::ExpandedIndexed {
            k_top, batch_size, ..
        }
        | DownEvidence::ExpandedGrouped {
            k_top, batch_size, ..
        } => {
            // Normalize the producer buffer from the typed evidence.
            let producer = match evidence {
                DownEvidence::ExpandedIndexed { out, .. } => out,
                DownEvidence::ExpandedGrouped { y, .. } => y,
                _ => unreachable!("expanded branch only handles expanded evidence"),
            };
            // Any i64 conversion anywhere is misplaced in an expanded f32
            // program; it must never flatten into a launchable schedule.
            if let Some(&(index, phase, _)) = ops.conversions.first() {
                return Err(MoeLowerError::MisplacedI64Conversion {
                    group: group.group.clone(),
                    layer: group.layer,
                    rank,
                    index,
                    phase,
                });
            }
            // The combine phase may only contain MoeCombine steps.
            if let Some((index, _)) = phases
                .combine
                .iter()
                .enumerate()
                .find(|(_, step)| !matches!(step, Step::MoeCombine { .. }))
            {
                return Err(MoeLowerError::UnrecognizedCombineStep {
                    group: group.group.clone(),
                    layer: group.layer,
                    rank,
                    index: combine_offset + index,
                });
            }
            // Deferred-expanded protocol (explicit `MoeProgramParts`
            // marker only): the expanded down producer carries NO local
            // combine because the architecture's next-layer fused consumer
            // folds it (combine-next-RMS). A local combine under the flag
            // would double-add and is rejected; ordinary programs keep the
            // exact-one-combine contract.
            let deferred =
                deferred_combine && matches!(evidence, DownEvidence::ExpandedIndexed { .. });
            let combine_step = if deferred {
                if let Some(&(_, _, _)) = ops.combines.first() {
                    return Err(MoeLowerError::CombineCountMismatch {
                        group: group.group.clone(),
                        layer: group.layer,
                        rank,
                        expected: 0,
                        actual: ops.combines.len(),
                    });
                }
                None
            } else {
                // Exactly one combine, in the semantic combine phase.
                Some(match ops.combines.as_slice() {
                    [] => {
                        return Err(MoeLowerError::CombineCountMismatch {
                            group: group.group.clone(),
                            layer: group.layer,
                            rank,
                            expected: 1,
                            actual: 0,
                        });
                    }
                    [_first, second, ..] => {
                        return Err(MoeLowerError::DuplicateCombineOp {
                            group: group.group.clone(),
                            layer: group.layer,
                            rank,
                            index: second.0,
                        });
                    }
                    [only] if only.1 != "combine" => {
                        return Err(MoeLowerError::MisplacedCombineOp {
                            group: group.group.clone(),
                            layer: group.layer,
                            rank,
                            index: only.0,
                            phase: only.1,
                        });
                    }
                    [only] => only.2,
                })
            };
            // Combine metadata validation applies only to the ordinary path;
            // the deferred path has no combine to validate (dim stays 0: the
            // Single schedule never consults it). `combine_topk_weights` and
            // `combine_hidden` additionally feed the grouped chain checks
            // below; the grouped branch always carries a combine.
            let (dim, inverse_perm, combine_topk_weights, combine_hidden) =
                if let Some(combine_step) = combine_step {
                    let Step::MoeCombine {
                        down_out,
                        topk_weights: combine_topk_weights,
                        out: combine_out,
                        k,
                        hidden,
                        batch_size: combine_batch_size,
                        inverse_perm,
                        ..
                    } = combine_step
                    else {
                        unreachable!("combine scan only collects MoeCombine steps")
                    };
                    let (down_out, combine_out, combine_topk_weights) =
                        (*down_out, *combine_out, *combine_topk_weights);
                    // The combine must consume the expanded down producer
                    // (aliases with the same non-null pointer and size are
                    // accepted).
                    if !same_buffer(down_out, producer) {
                        return Err(MoeLowerError::CombineDownSourceMismatch {
                            group: group.group.clone(),
                            layer: group.layer,
                            rank,
                            index: combine_offset,
                        });
                    }
                    // The combine's shape metadata must match the producer's
                    // launch metadata exactly.
                    if *k != k_top {
                        return Err(MoeLowerError::CombineMetadataMismatch {
                            group: group.group.clone(),
                            layer: group.layer,
                            rank,
                            index: combine_offset,
                            field: "k",
                            expected: k_top,
                            actual: *k,
                        });
                    }
                    if *combine_batch_size != batch_size {
                        return Err(MoeLowerError::CombineMetadataMismatch {
                            group: group.group.clone(),
                            layer: group.layer,
                            rank,
                            index: combine_offset,
                            field: "batch_size",
                            expected: batch_size,
                            actual: *combine_batch_size,
                        });
                    }
                    if *hidden == 0 || *combine_batch_size == 0 {
                        return Err(MoeLowerError::InvalidCombineDimensions {
                            group: group.group.clone(),
                            layer: group.layer,
                            rank,
                            hidden: *hidden,
                            batch_size: *combine_batch_size,
                        });
                    }
                    // Collective dimension is the checked logical
                    // batch_size*hidden, never inferred from allocation capacity.
                    let dim = hidden.checked_mul(*combine_batch_size).ok_or_else(|| {
                        MoeLowerError::ArithmeticOverflow {
                            group: group.group.clone(),
                            layer: group.layer,
                            rank,
                            index: combine_offset,
                            what: "combine_dim",
                        }
                    })?;
                    // The combine output must hold exactly the collective
                    // dimension, with enough F32 capacity for it.
                    let combine_out_dim = combine_out
                        .shape
                        .iter()
                        .try_fold(1usize, |acc, &dim| acc.checked_mul(dim))
                        .ok_or_else(|| MoeLowerError::ArithmeticOverflow {
                            group: group.group.clone(),
                            layer: group.layer,
                            rank,
                            index: combine_offset,
                            what: "combine_out_shape",
                        })?;
                    if combine_out_dim != dim {
                        return Err(MoeLowerError::CombineOutputShapeMismatch {
                            group: group.group.clone(),
                            layer: group.layer,
                            rank,
                            index: combine_offset,
                            expected: dim,
                            actual: combine_out_dim,
                        });
                    }
                    let needed =
                        dim.checked_mul(4)
                            .ok_or_else(|| MoeLowerError::ArithmeticOverflow {
                                group: group.group.clone(),
                                layer: group.layer,
                                rank,
                                index: combine_offset,
                                what: "combine_out_bytes",
                            })?;
                    let actual = cap(combine_out);
                    if actual < needed {
                        return Err(MoeLowerError::CapacityMismatch {
                            group: group.group.clone(),
                            layer: group.layer,
                            rank,
                            index: combine_offset,
                            expected_bytes: needed,
                            actual_bytes: actual,
                        });
                    }
                    (dim, *inverse_perm, Some(combine_topk_weights), *hidden)
                } else {
                    (0, None, None, 0)
                };
            let (kind, grouped_signature) = match evidence {
                DownEvidence::ExpandedIndexed { .. } => {
                    // Scatter, gate-up unscatter, grouped gate-up, and
                    // unpaired Givens are never admitted on the indexed path
                    // (the specialized-gate-up-paired Paro pre-rotation is
                    // the sole Givens exception, decided inside
                    // reject_stray_grouped_ops).
                    reject_stray_grouped_ops(group, rank, &ops)?;
                    if inverse_perm.is_some() {
                        return Err(MoeLowerError::UnexpectedCombineInversePerm {
                            group: group.group.clone(),
                            layer: group.layer,
                            rank,
                            index: combine_offset,
                        });
                    }
                    (
                        if deferred {
                            RankProtocolKind::ExpandedIndexedDeferred
                        } else {
                            RankProtocolKind::ExpandedIndexed
                        },
                        None,
                    )
                }
                DownEvidence::ExpandedGrouped {
                    y: _,
                    x: down_x,
                    sorted_slot_index,
                    expert_tile_ids,
                    k_top: grouped_k_top,
                    batch_size: grouped_batch_size,
                    m_total: grouped_m_total,
                    n_experts: grouped_n_experts,
                    dtype_tags: grouped_dtype_tags,
                    force_mq4_fp16: dn_force_mq4_fp16,
                    paro_i8: dn_paro_i8,
                    paro_i8_k8: dn_paro_i8_k8,
                } => {
                    let perm =
                        inverse_perm.ok_or_else(|| MoeLowerError::MissingCombineInversePerm {
                            group: group.group.clone(),
                            layer: group.layer,
                            rank,
                            index: combine_offset,
                        })?;

                    // ── Binding phase grammar (G1) ─────────────────────────
                    // router exactly [MoeScatter]; gate_up
                    // [GivensRotateBatched?, GroupedMoeGemm(GateUp),
                    // MoeGateUpUnscatter]; activation exactly [MoeActivation];
                    // down exactly one grouped down; combine exactly one
                    // MoeCombine; finish exactly empty. No arbitrary extras or
                    // phase relocation.
                    // Absolute phase starts: router 0, gate_up router.len,
                    // activation +gate_up.len, down +activation.len, combine
                    // +down.len, finish +combine.len.
                    let router_offset = 0usize;
                    let gate_up_offset = router_offset + lengths[0];
                    let act_offset = gate_up_offset + lengths[1];
                    let finish_offset = combine_offset + lengths[4];
                    // Exactly one scatter, and it must live in the router
                    // phase.
                    let (scatter_index, scatter_step) = match ops.scatters.as_slice() {
                        [] => {
                            return Err(MoeLowerError::MissingScatter {
                                group: group.group.clone(),
                                layer: group.layer,
                                rank,
                                index: router_offset,
                            });
                        }
                        [_first, second, ..] => {
                            return Err(MoeLowerError::DuplicateScatter {
                                group: group.group.clone(),
                                layer: group.layer,
                                rank,
                                index: second.0,
                            });
                        }
                        [only] => {
                            if only.1 != "router" {
                                return Err(MoeLowerError::MisplacedScatter {
                                    group: group.group.clone(),
                                    layer: group.layer,
                                    rank,
                                    index: only.0,
                                    phase: only.1,
                                });
                            }
                            (only.0, only.2)
                        }
                    };
                    if phases.router.len() != 1 {
                        return Err(MoeLowerError::GroupedPhaseMismatch {
                            group: group.group.clone(),
                            layer: group.layer,
                            rank,
                            phase: "router",
                            expected: "exactly [MoeScatter]",
                            index: router_offset,
                        });
                    }
                    // Exactly one grouped gate-up and one gate-up unscatter
                    // globally (missing/duplicate), then the exact gate_up
                    // sequence with its optional leading Givens.
                    let (gate_up_index, _) = match ops.gate_ups.as_slice() {
                        [] => {
                            return Err(MoeLowerError::GroupedOpCountMismatch {
                                group: group.group.clone(),
                                layer: group.layer,
                                rank,
                                op: "gate_up",
                                expected: 1,
                                actual: 0,
                            });
                        }
                        [_first, _second, ..] => {
                            return Err(MoeLowerError::GroupedOpCountMismatch {
                                group: group.group.clone(),
                                layer: group.layer,
                                rank,
                                op: "gate_up",
                                expected: 1,
                                actual: ops.gate_ups.len(),
                            });
                        }
                        [only] => (only.0, only.2),
                    };
                    let (unscatter_index, _) = match ops.gate_up_unscatters.as_slice() {
                        [] => {
                            return Err(MoeLowerError::GroupedOpCountMismatch {
                                group: group.group.clone(),
                                layer: group.layer,
                                rank,
                                op: "gate_up_unscatter",
                                expected: 1,
                                actual: 0,
                            });
                        }
                        [_first, _second, ..] => {
                            return Err(MoeLowerError::GroupedOpCountMismatch {
                                group: group.group.clone(),
                                layer: group.layer,
                                rank,
                                op: "gate_up_unscatter",
                                expected: 1,
                                actual: ops.gate_up_unscatters.len(),
                            });
                        }
                        [only] => (only.0, only.2),
                    };
                    let (givens, gate_up_step, unscatter_step) = match phases.gate_up.as_slice() {
                        [givens @ Step::GivensRotateBatched { .. }, gu @ Step::GroupedMoeGemm {
                            which: MoeProj::GateUp { .. },
                            ..
                        }, us @ Step::MoeGateUpUnscatter { .. }] => (Some(givens), gu, us),
                        [gu @ Step::GroupedMoeGemm {
                            which: MoeProj::GateUp { .. },
                            ..
                        }, us @ Step::MoeGateUpUnscatter { .. }] => (None, gu, us),
                        _ => {
                            return Err(MoeLowerError::GroupedPhaseMismatch {
                                    group: group.group.clone(),
                                    layer: group.layer,
                                    rank,
                                    phase: "gate_up",
                                    expected: "[GivensRotateBatched,] GroupedMoeGemm(GateUp), MoeGateUpUnscatter",
                                    index: gate_up_offset,
                                });
                        }
                    };
                    let act_step = match phases.activation.as_slice() {
                        [act @ Step::MoeActivation { .. }] => act,
                        _ => {
                            return Err(MoeLowerError::GroupedPhaseMismatch {
                                group: group.group.clone(),
                                layer: group.layer,
                                rank,
                                phase: "activation",
                                expected: "exactly [MoeActivation]",
                                index: act_offset,
                            });
                        }
                    };
                    if !phases.finish.is_empty() {
                        return Err(MoeLowerError::GroupedPhaseMismatch {
                            group: group.group.clone(),
                            layer: group.layer,
                            rank,
                            phase: "finish",
                            expected: "exactly empty",
                            index: finish_offset,
                        });
                    }
                    let Step::GroupedMoeGemm {
                        experts: down_experts,
                        y: down_y_tensor,
                        ..
                    } = &phases.down[0]
                    else {
                        unreachable!("down evidence only collects grouped downs")
                    };
                    let (
                        gate_up_sorted,
                        gate_up_tiles,
                        gate_up_y,
                        gate_up_x,
                        gate_up_m_total,
                        gate_up_batch_size,
                        gate_up_k_top,
                        gu_dtype_tags,
                        gu_force_mq4_fp16,
                        gu_paro_i8,
                        gu_paro_i8_k8,
                        gate_up_experts,
                    ) = match gate_up_step {
                        Step::GroupedMoeGemm {
                            sorted_slot_index,
                            expert_tile_ids,
                            y,
                            which: MoeProj::GateUp { .. },
                            x,
                            m_total,
                            batch_size,
                            k_top,
                            dtype_tags: gu_dtype_tags,
                            force_mq4_fp16,
                            paro_i8,
                            paro_i8_k8,
                            experts,
                            ..
                        } => (
                            *sorted_slot_index,
                            *expert_tile_ids,
                            *y,
                            *x,
                            *m_total,
                            *batch_size,
                            *k_top,
                            *gu_dtype_tags,
                            *force_mq4_fp16,
                            *paro_i8,
                            *paro_i8_k8,
                            experts,
                        ),
                        _ => unreachable!("gate_up scan only collects GroupedMoeGemm GateUp"),
                    };
                    let (
                        unscatter_y_grouped,
                        unscatter_sorted,
                        unscatter_gate_batch,
                        unscatter_up_batch,
                        unscatter_inter,
                        unscatter_m_total,
                        unscatter_k_top,
                    ) = match unscatter_step {
                        Step::MoeGateUpUnscatter {
                            y_grouped,
                            sorted_slot_index,
                            gate_batch,
                            up_batch,
                            inter,
                            m_total,
                            k_top,
                        } => (
                            *y_grouped,
                            *sorted_slot_index,
                            *gate_batch,
                            *up_batch,
                            *inter,
                            *m_total,
                            *k_top,
                        ),
                        _ => {
                            unreachable!("gate-up unscatter scan only collects MoeGateUpUnscatter")
                        }
                    };
                    let (act_gate, act_up, act_rot_out, act_inter, act_rows) = match act_step {
                        Step::MoeActivation {
                            gate,
                            up,
                            rot_out,
                            inter,
                            k_top,
                            ..
                        } => (*gate, *up, *rot_out, *inter, *k_top),
                        _ => unreachable!("activation pin only collects MoeActivation"),
                    };
                    let (
                        scatter_topk,
                        scatter_counts,
                        scatter_offsets,
                        scatter_sorted,
                        scatter_tiles,
                        scatter_perm,
                        scatter_total_slots,
                        scatter_n_experts,
                        scatter_m_total_max,
                        scatter_block_m,
                    ) = match scatter_step {
                        Step::MoeScatter {
                            topk_indices,
                            expert_token_counts,
                            expert_offsets,
                            sorted_slot_index,
                            expert_tile_ids,
                            inverse_perm,
                            total_slots,
                            n_experts,
                            m_total_max,
                            block_m,
                            ..
                        } => (
                            *topk_indices,
                            *expert_token_counts,
                            *expert_offsets,
                            *sorted_slot_index,
                            *expert_tile_ids,
                            *inverse_perm,
                            *total_slots,
                            *n_experts,
                            *m_total_max,
                            *block_m,
                        ),
                        _ => unreachable!("scatter scan only collects MoeScatter"),
                    };
                    // Executable zero/range gates before any chain or
                    // capacity arithmetic: a zero routed geometry is never a
                    // degenerate launch. Zero batch/hidden are additionally
                    // rejected by the combine preamble's contextual
                    // InvalidCombineDimensions before this branch is reached.
                    for (field, value, index) in [
                        ("batch_size", grouped_batch_size, scatter_index),
                        ("k_top", grouped_k_top, scatter_index),
                        ("n_experts", grouped_n_experts, scatter_index),
                        ("inter", gate_up_experts.expert_m, gate_up_index),
                        ("hidden", down_experts.expert_k, down_index),
                        ("m_total", scatter_m_total_max, scatter_index),
                    ] {
                        if value == 0 {
                            return Err(MoeLowerError::GroupedZeroGeometry {
                                group: group.group.clone(),
                                layer: group.layer,
                                rank,
                                index,
                                field,
                            });
                        }
                    }
                    if grouped_k_top > grouped_n_experts {
                        return Err(MoeLowerError::GroupedKTopExceedsExperts {
                            group: group.group.clone(),
                            layer: group.layer,
                            rank,
                            index: scatter_index,
                            k_top: grouped_k_top,
                            n_experts: grouped_n_experts,
                        });
                    }
                    // Buffer chain: one scatter feeds the gate-up, the
                    // unscatter, and the down; the unscatter consumes the
                    // gate-up's y; the gate-up's up_out is the unscatter's
                    // up_batch; the activation consumes the unscatter's
                    // gate/up; the down consumes the activation's rot_out;
                    // the combine consumes the scatter's perm.
                    if !same_buffer(scatter_sorted, gate_up_sorted) {
                        return Err(MoeLowerError::ScatterChainMismatch {
                            group: group.group.clone(),
                            layer: group.layer,
                            rank,
                            index: gate_up_index,
                            field: "sorted_slot_index",
                        });
                    }
                    if !same_buffer(scatter_sorted, unscatter_sorted) {
                        return Err(MoeLowerError::ScatterChainMismatch {
                            group: group.group.clone(),
                            layer: group.layer,
                            rank,
                            index: unscatter_index,
                            field: "sorted_slot_index",
                        });
                    }
                    if !same_buffer(scatter_sorted, sorted_slot_index) {
                        return Err(MoeLowerError::ScatterChainMismatch {
                            group: group.group.clone(),
                            layer: group.layer,
                            rank,
                            index: down_index,
                            field: "sorted_slot_index",
                        });
                    }
                    if !same_buffer(scatter_tiles, gate_up_tiles) {
                        return Err(MoeLowerError::ScatterChainMismatch {
                            group: group.group.clone(),
                            layer: group.layer,
                            rank,
                            index: gate_up_index,
                            field: "expert_tile_ids",
                        });
                    }
                    if !same_buffer(scatter_tiles, expert_tile_ids) {
                        return Err(MoeLowerError::ScatterChainMismatch {
                            group: group.group.clone(),
                            layer: group.layer,
                            rank,
                            index: down_index,
                            field: "expert_tile_ids",
                        });
                    }
                    if !same_buffer(scatter_perm, perm) {
                        return Err(MoeLowerError::ScatterChainMismatch {
                            group: group.group.clone(),
                            layer: group.layer,
                            rank,
                            index: scatter_index,
                            field: "inverse_perm",
                        });
                    }
                    if !same_buffer(gate_up_y, unscatter_y_grouped) {
                        return Err(MoeLowerError::GroupedChainMismatch {
                            group: group.group.clone(),
                            layer: group.layer,
                            rank,
                            index: unscatter_index,
                            field: "y_grouped",
                        });
                    }
                    if !same_buffer(unscatter_gate_batch, act_gate) {
                        return Err(MoeLowerError::GroupedChainMismatch {
                            group: group.group.clone(),
                            layer: group.layer,
                            rank,
                            index: act_offset,
                            field: "activation_gate",
                        });
                    }
                    if !same_buffer(unscatter_up_batch, act_up) {
                        return Err(MoeLowerError::GroupedChainMismatch {
                            group: group.group.clone(),
                            layer: group.layer,
                            rank,
                            index: act_offset,
                            field: "activation_up",
                        });
                    }
                    if !same_buffer(act_rot_out, down_x) {
                        return Err(MoeLowerError::GroupedChainMismatch {
                            group: group.group.clone(),
                            layer: group.layer,
                            rank,
                            index: down_index,
                            field: "down_input",
                        });
                    }
                    // Optional Givens: at most one, first in gate_up (the
                    // sequence pin above), whose output feeds the grouped
                    // gate-up input with matching batch/dim.
                    let (givens_x, givens_pairs, givens_theta, givens_scales, givens_krot) =
                        match givens {
                            Some(Step::GivensRotateBatched {
                                x,
                                pairs,
                                theta,
                                scales,
                                krot,
                                ..
                            }) => (
                                Some(*x),
                                Some(*pairs),
                                Some(*theta),
                                Some(*scales),
                                Some(*krot),
                            ),
                            None => (None, None, None, None, None),
                            _ => unreachable!("sequence pin only collects GivensRotateBatched"),
                        };
                    if let Some(givens) = givens {
                        let Step::GivensRotateBatched {
                            out, batch, dim, ..
                        } = givens
                        else {
                            unreachable!("sequence pin only collects GivensRotateBatched")
                        };
                        // The rotation dim must be nonzero and 128-aligned
                        // before any chain or capacity math.
                        if *dim == 0 {
                            return Err(MoeLowerError::GroupedChainMismatch {
                                group: group.group.clone(),
                                layer: group.layer,
                                rank,
                                index: gate_up_offset,
                                field: "givens_dim",
                            });
                        }
                        if *dim % 128 != 0 {
                            return Err(MoeLowerError::GroupedGivensAlignment {
                                group: group.group.clone(),
                                layer: group.layer,
                                rank,
                                index: gate_up_offset,
                                dim: *dim,
                                expected_alignment: 128,
                            });
                        }
                        if !same_buffer(out, gate_up_x) {
                            return Err(MoeLowerError::GroupedChainMismatch {
                                group: group.group.clone(),
                                layer: group.layer,
                                rank,
                                index: gate_up_index,
                                field: "givens_out",
                            });
                        }
                        if *batch != gate_up_batch_size {
                            return Err(MoeLowerError::GroupedChainMismatch {
                                group: group.group.clone(),
                                layer: group.layer,
                                rank,
                                index: gate_up_index,
                                field: "givens_batch",
                            });
                        }
                        if *dim != gate_up_experts.expert_k {
                            return Err(MoeLowerError::GroupedChainMismatch {
                                group: group.group.clone(),
                                layer: group.layer,
                                rank,
                                index: gate_up_index,
                                field: "givens_dim",
                            });
                        }
                    }
                    // Scalar invariants of the frozen grouped contract.
                    if gate_up_k_top != grouped_k_top {
                        return Err(MoeLowerError::GroupedChainMismatch {
                            group: group.group.clone(),
                            layer: group.layer,
                            rank,
                            index: gate_up_index,
                            field: "k_top",
                        });
                    }
                    if unscatter_k_top != grouped_k_top {
                        return Err(MoeLowerError::GroupedChainMismatch {
                            group: group.group.clone(),
                            layer: group.layer,
                            rank,
                            index: unscatter_index,
                            field: "k_top",
                        });
                    }
                    if gate_up_batch_size != grouped_batch_size {
                        return Err(MoeLowerError::GroupedChainMismatch {
                            group: group.group.clone(),
                            layer: group.layer,
                            rank,
                            index: gate_up_index,
                            field: "batch_size",
                        });
                    }
                    // Cross-expert width agreement per the actual Step
                    // semantics: expert_m is the INTERMEDIATE width (gate-up
                    // fused output per stream, unscatter/activation inter,
                    // grouped-down input); expert_k is the MODEL HIDDEN width
                    // (gate-up input, grouped-down output, combine hidden).
                    if gate_up_experts.expert_m != down_experts.expert_m {
                        return Err(MoeLowerError::GroupedChainMismatch {
                            group: group.group.clone(),
                            layer: group.layer,
                            rank,
                            index: gate_up_index,
                            field: "inter",
                        });
                    }
                    if gate_up_experts.expert_k != down_experts.expert_k {
                        return Err(MoeLowerError::GroupedChainMismatch {
                            group: group.group.clone(),
                            layer: group.layer,
                            rank,
                            index: gate_up_index,
                            field: "hidden",
                        });
                    }
                    // The deinterleave inter and the activation inter must
                    // both equal the expert contraction width.
                    if unscatter_inter != gate_up_experts.expert_m {
                        return Err(MoeLowerError::GroupedChainMismatch {
                            group: group.group.clone(),
                            layer: group.layer,
                            rank,
                            index: unscatter_index,
                            field: "inter",
                        });
                    }
                    if act_inter != unscatter_inter {
                        return Err(MoeLowerError::GroupedChainMismatch {
                            group: group.group.clone(),
                            layer: group.layer,
                            rank,
                            index: act_offset,
                            field: "inter",
                        });
                    }
                    // The combine output width must equal the grouped down
                    // output width (expert_k). The grouped chain always
                    // carries its combine, so the hoisted value is present.
                    if combine_hidden != down_experts.expert_k {
                        return Err(MoeLowerError::CombineMetadataMismatch {
                            group: group.group.clone(),
                            layer: group.layer,
                            rank,
                            index: combine_offset,
                            field: "hidden",
                            expected: down_experts.expert_k,
                            actual: combine_hidden,
                        });
                    }
                    // Checked slot arithmetic: total_slots must equal the
                    // exact batch_size*k_top product, and the activation must
                    // span exactly those rows.
                    let slots = grouped_batch_size
                        .checked_mul(grouped_k_top)
                        .ok_or_else(|| MoeLowerError::ArithmeticOverflow {
                            group: group.group.clone(),
                            layer: group.layer,
                            rank,
                            index: scatter_index,
                            what: "scatter_slots",
                        })?;
                    if scatter_total_slots != slots {
                        return Err(MoeLowerError::ScatterSlotCountMismatch {
                            group: group.group.clone(),
                            layer: group.layer,
                            rank,
                            index: scatter_index,
                            expected: slots,
                            actual: scatter_total_slots,
                        });
                    }
                    if act_rows != slots {
                        return Err(MoeLowerError::GroupedChainMismatch {
                            group: group.group.clone(),
                            layer: group.layer,
                            rank,
                            index: act_offset,
                            field: "activation_rows",
                        });
                    }
                    // Frozen grouped block width: the dispatch crate's
                    // public MOE_GROUPED_BLOCK_M (16) is the binding value.
                    if scatter_block_m != MOE_GROUPED_BLOCK_M {
                        return Err(MoeLowerError::ScatterBlockMismatch {
                            group: group.group.clone(),
                            layer: group.layer,
                            rank,
                            index: scatter_index,
                            expected: MOE_GROUPED_BLOCK_M,
                            actual: scatter_block_m,
                        });
                    }
                    if scatter_n_experts != gate_up_experts.n_experts {
                        return Err(MoeLowerError::ScatterExpertCountMismatch {
                            group: group.group.clone(),
                            layer: group.layer,
                            rank,
                            index: scatter_index,
                            expected: scatter_n_experts,
                            actual: gate_up_experts.n_experts,
                        });
                    }
                    if scatter_n_experts != grouped_n_experts {
                        return Err(MoeLowerError::ScatterExpertCountMismatch {
                            group: group.group.clone(),
                            layer: group.layer,
                            rank,
                            index: scatter_index,
                            expected: scatter_n_experts,
                            actual: grouped_n_experts,
                        });
                    }
                    if scatter_m_total_max != gate_up_m_total {
                        return Err(MoeLowerError::ScatterMaxTotalMismatch {
                            group: group.group.clone(),
                            layer: group.layer,
                            rank,
                            index: scatter_index,
                            field: "gate_up",
                            expected: scatter_m_total_max,
                            actual: gate_up_m_total,
                        });
                    }
                    if scatter_m_total_max != unscatter_m_total {
                        return Err(MoeLowerError::ScatterMaxTotalMismatch {
                            group: group.group.clone(),
                            layer: group.layer,
                            rank,
                            index: scatter_index,
                            field: "gate_up_unscatter",
                            expected: scatter_m_total_max,
                            actual: unscatter_m_total,
                        });
                    }
                    if scatter_m_total_max != grouped_m_total {
                        return Err(MoeLowerError::ScatterMaxTotalMismatch {
                            group: group.group.clone(),
                            layer: group.layer,
                            rank,
                            index: scatter_index,
                            field: "down",
                            expected: scatter_m_total_max,
                            actual: grouped_m_total,
                        });
                    }
                    // ── Actual-buffer capacity gates (G1 remediation) ──────
                    // Every grouped operand sized from the kernel/launcher
                    // contracts: scatter [slots]i32 / [E]i32 / [E+1]i32 /
                    // [m_total_max]i32 / [m_total_max/block_m]i32 / [slots]i32;
                    // Givens [batch×dim]f32 in/out, pairs [krot×dim]i16,
                    // theta [krot×dim/2]f16, scales [dim]f16; gate-up input
                    // [batch×hidden]f32, pointer table [E]u64, fused y
                    // [m_total × 2*inter]f32; unscatter gate/up
                    // [slots × inter]f32; activation rot [slots × inter]f32;
                    // down input [slots × inter]f32, pointer table [E]u64,
                    // y [m_total × hidden]f32; combine topk weights
                    // [slots]f32. Shared buffers (sorted, tiles, perm,
                    // y_gate_up, y_down, gate/up, rot) are gated once at
                    // their primary role with the same byte math.
                    let inter = gate_up_experts.expert_m;
                    let hidden = down_experts.expert_k;
                    let m_total = scatter_m_total_max;
                    // `slots` is the checked batch*k_top product from the
                    // scalar section above; reused here, never recomputed.
                    // Checked ceil tile count; a bound that is not a multiple
                    // of the frozen block width would truncate the kernel
                    // tile count and is rejected contextually. For admitted
                    // aligned programs ceil == floor.
                    let tiles = m_total
                        .checked_add(MOE_GROUPED_BLOCK_M - 1)
                        .map(|v| v / MOE_GROUPED_BLOCK_M)
                        .ok_or_else(|| MoeLowerError::ArithmeticOverflow {
                            group: group.group.clone(),
                            layer: group.layer,
                            rank,
                            index: scatter_index,
                            what: "tile_count",
                        })?;
                    if m_total % MOE_GROUPED_BLOCK_M != 0 {
                        return Err(MoeLowerError::GroupedTileAlignment {
                            group: group.group.clone(),
                            layer: group.layer,
                            rank,
                            index: scatter_index,
                            m_total,
                            block_m: MOE_GROUPED_BLOCK_M,
                        });
                    }
                    // Every pre-byte product/sum is checked before the byte
                    // math; overflow reports the failing element count.
                    let offsets_elems = checked_add_elems(
                        group,
                        rank,
                        scatter_index,
                        "offsets_elems",
                        scatter_n_experts,
                        1,
                    )?;
                    let gu_input_elems = checked_elems(
                        group,
                        rank,
                        gate_up_index,
                        "gu_input_elems",
                        grouped_batch_size,
                        hidden,
                    )?;
                    let gu_y_inter_elems =
                        checked_elems(group, rank, gate_up_index, "gu_y_inter_elems", inter, 2)?;
                    let gu_y_elems = checked_elems(
                        group,
                        rank,
                        gate_up_index,
                        "gu_y_elems",
                        m_total,
                        gu_y_inter_elems,
                    )?;
                    let inter_elems =
                        checked_elems(group, rank, unscatter_index, "inter_elems", slots, inter)?;
                    let dn_y_elems =
                        checked_elems(group, rank, down_index, "dn_y_elems", m_total, hidden)?;
                    let slots4 =
                        checked_byte_count(group, rank, scatter_index, "slots_bytes", slots, 4)?;
                    let experts4 = checked_byte_count(
                        group,
                        rank,
                        scatter_index,
                        "counts_bytes",
                        scatter_n_experts,
                        4,
                    )?;
                    let offsets4 = checked_byte_count(
                        group,
                        rank,
                        scatter_index,
                        "offsets_bytes",
                        offsets_elems,
                        4,
                    )?;
                    let sorted4 =
                        checked_byte_count(group, rank, scatter_index, "sorted_bytes", m_total, 4)?;
                    let tiles4 =
                        checked_byte_count(group, rank, scatter_index, "tiles_bytes", tiles, 4)?;
                    let ptr8 = checked_byte_count(
                        group,
                        rank,
                        gate_up_index,
                        "pointer_table_bytes",
                        scatter_n_experts,
                        8,
                    )?;
                    let gu_input4 = checked_byte_count(
                        group,
                        rank,
                        gate_up_index,
                        "gu_input_bytes",
                        gu_input_elems,
                        4,
                    )?;
                    let gu_y4 = checked_byte_count(
                        group,
                        rank,
                        gate_up_index,
                        "gu_y_bytes",
                        gu_y_elems,
                        4,
                    )?;
                    let inter4 =
                        checked_byte_count(group, rank, act_offset, "inter_bytes", inter_elems, 4)?;
                    let dn_y4 =
                        checked_byte_count(group, rank, down_index, "dn_y_bytes", dn_y_elems, 4)?;
                    // Scatter operands (router phase).
                    for (tensor, needed) in [
                        (scatter_topk, slots4),
                        (scatter_counts, experts4),
                        (scatter_offsets, offsets4),
                        (scatter_sorted, sorted4),
                        (scatter_tiles, tiles4),
                        (scatter_perm, slots4),
                    ] {
                        require_grouped_capacity(group, rank, router_offset, &cap, tensor, needed)?;
                    }
                    // Optional Givens sidecars (typed Paro configuration).
                    // The Givens output aliases the gate-up input (already
                    // gated above with the identical batch*hidden*4 math).
                    if let (Some(x), Some(pairs), Some(theta), Some(scales), Some(krot)) = (
                        givens_x,
                        givens_pairs,
                        givens_theta,
                        givens_scales,
                        givens_krot,
                    ) {
                        let dim = gate_up_experts.expert_k;
                        let batch = grouped_batch_size;
                        let givens_io_elems = checked_elems(
                            group,
                            rank,
                            gate_up_offset,
                            "givens_io_elems",
                            batch,
                            dim,
                        )?;
                        let givens_pairs_elems = checked_elems(
                            group,
                            rank,
                            gate_up_offset,
                            "givens_pairs_elems",
                            krot,
                            dim,
                        )?;
                        let givens_theta_elems = checked_elems(
                            group,
                            rank,
                            gate_up_offset,
                            "givens_theta_elems",
                            krot,
                            dim / 2,
                        )?;
                        // scales holds dim f16 elements: no element-count
                        // product here, the byte math applies the f16 factor.
                        let givens_scales_elems = dim;
                        let givens_io4 = checked_byte_count(
                            group,
                            rank,
                            gate_up_offset,
                            "givens_io_bytes",
                            givens_io_elems,
                            4,
                        )?;
                        let pairs2 = checked_byte_count(
                            group,
                            rank,
                            gate_up_offset,
                            "givens_pairs_bytes",
                            givens_pairs_elems,
                            2,
                        )?;
                        let theta2 = checked_byte_count(
                            group,
                            rank,
                            gate_up_offset,
                            "givens_theta_bytes",
                            givens_theta_elems,
                            2,
                        )?;
                        let scales2 = checked_byte_count(
                            group,
                            rank,
                            gate_up_offset,
                            "givens_scales_bytes",
                            givens_scales_elems,
                            2,
                        )?;
                        for (tensor, needed) in [
                            (x, givens_io4),
                            (pairs, pairs2),
                            (theta, theta2),
                            (scales, scales2),
                        ] {
                            require_grouped_capacity(
                                group,
                                rank,
                                gate_up_offset,
                                &cap,
                                tensor,
                                needed,
                            )?;
                        }
                    }
                    // Gate-up: input, pointer table, fused output.
                    require_grouped_capacity(
                        group,
                        rank,
                        gate_up_index,
                        &cap,
                        gate_up_x,
                        gu_input4,
                    )?;
                    require_grouped_capacity(
                        group,
                        rank,
                        gate_up_index,
                        &cap,
                        gate_up_experts.gate_up_ptrs,
                        ptr8,
                    )?;
                    require_grouped_capacity(group, rank, gate_up_index, &cap, gate_up_y, gu_y4)?;
                    // Unscatter: gate/up (the y_grouped/sorted alias the
                    // gate-up y and scatter sorted, already gated).
                    require_grouped_capacity(
                        group,
                        rank,
                        unscatter_index,
                        &cap,
                        unscatter_gate_batch,
                        inter4,
                    )?;
                    require_grouped_capacity(
                        group,
                        rank,
                        unscatter_index,
                        &cap,
                        unscatter_up_batch,
                        inter4,
                    )?;
                    // Activation: rot_out (gate/up alias the unscatter
                    // buffers, already gated).
                    require_grouped_capacity(group, rank, act_offset, &cap, act_rot_out, inter4)?;
                    // Down: pointer table and grouped output (x aliases the
                    // activation rot_out, already gated).
                    require_grouped_capacity(
                        group,
                        rank,
                        down_index,
                        &cap,
                        down_experts.down_ptrs,
                        ptr8,
                    )?;
                    require_grouped_capacity(group, rank, down_index, &cap, down_y_tensor, dn_y4)?;
                    // Combine: topk weights (down_out/perm alias gated
                    // buffers; the final output keeps its generic check).
                    // The grouped chain always carries its combine.
                    require_grouped_capacity(
                        group,
                        rank,
                        combine_offset,
                        &cap,
                        combine_topk_weights.expect("grouped chain always carries a combine"),
                        slots4,
                    )?;
                    // Optional per-expert dtype-tag tables: each Some is
                    // independent and needs n_experts bytes at its own step
                    // index; neither implies the other.
                    if let Some(tags) = gu_dtype_tags {
                        require_grouped_capacity(
                            group,
                            rank,
                            gate_up_index,
                            &cap,
                            tags,
                            scatter_n_experts,
                        )?;
                    }
                    if let Some(tags) = grouped_dtype_tags {
                        require_grouped_capacity(
                            group,
                            rank,
                            down_index,
                            &cap,
                            tags,
                            scatter_n_experts,
                        )?;
                    }
                    // Typed cross-rank signature for the validated grouped
                    // chain: every rank-invariant launch discriminator, no
                    // rank-local buffer pointers. Givens dim == expert_k and
                    // activation rows == batch*k_top are per-rank bound; both
                    // the derived values and their sources travel.
                    let givens_signature = match givens {
                        Some(Step::GivensRotateBatched { krot, dim, .. }) => Some((*krot, *dim)),
                        None => None,
                        _ => unreachable!("sequence pin only collects GivensRotateBatched"),
                    };
                    (
                        RankProtocolKind::ExpandedGrouped,
                        Some(GroupedSignature {
                            batch_size: grouped_batch_size,
                            k_top: grouped_k_top,
                            n_experts: grouped_n_experts,
                            inter: gate_up_experts.expert_m,
                            hidden: down_experts.expert_k,
                            m_total: scatter_m_total_max,
                            block_m: scatter_block_m,
                            activation: grouped_activation_signature(act_step),
                            activation_rows: slots,
                            givens: givens_signature,
                            gate_up: GroupedGemmControls {
                                dtype: gate_up_experts.dtype,
                                force_mq4_fp16: gu_force_mq4_fp16,
                                paro_i8: gu_paro_i8,
                                paro_i8_k8: gu_paro_i8_k8,
                                tags: gu_dtype_tags.is_some(),
                            },
                            down: GroupedGemmControls {
                                dtype: down_experts.dtype,
                                force_mq4_fp16: dn_force_mq4_fp16,
                                paro_i8: dn_paro_i8,
                                paro_i8_k8: dn_paro_i8_k8,
                                tags: grouped_dtype_tags.is_some(),
                            },
                        }),
                    )
                }
                _ => unreachable!("expanded branch only handles expanded evidence"),
            };
            Ok(RankProtocol {
                kind,
                down_index,
                combine_index: if deferred { None } else { Some(combine_offset) },
                convert_index: None,
                dim,
                batched: None,
                grouped: grouped_signature,
            })
        }
        DownEvidence::SelfCombiningF32 { out } => {
            // Grouped permutation machinery is never admitted on the
            // self-combining f32 path.
            reject_stray_grouped_ops(group, rank, &ops)?;
            // Any i64 conversion anywhere is misplaced in an f32
            // self-combining program.
            if let Some(&(index, phase, _)) = ops.conversions.first() {
                return Err(MoeLowerError::MisplacedI64Conversion {
                    group: group.group.clone(),
                    layer: group.layer,
                    rank,
                    index,
                    phase,
                });
            }
            // Any combine anywhere would double-accumulate; the combine phase
            // must be empty too.
            if let Some(&(index, _, _)) = ops.combines.first() {
                return Err(MoeLowerError::CombineAfterSelfCombiningDown {
                    group: group.group.clone(),
                    layer: group.layer,
                    rank,
                    index,
                });
            }
            if lengths[4] != 0 {
                return Err(MoeLowerError::CombineAfterSelfCombiningDown {
                    group: group.group.clone(),
                    layer: group.layer,
                    rank,
                    index: combine_offset,
                });
            }
            // Logical dimension from the checked shape product; a padded
            // allocation never enlarges the collective dim.
            let dim = out
                .shape
                .iter()
                .try_fold(1usize, |acc, &dim| acc.checked_mul(dim))
                .ok_or_else(|| MoeLowerError::ArithmeticOverflow {
                    group: group.group.clone(),
                    layer: group.layer,
                    rank,
                    index: down_index,
                    what: "f32_shape",
                })?;
            if dim == 0 {
                return Err(MoeLowerError::InvalidF32Shape {
                    group: group.group.clone(),
                    layer: group.layer,
                    rank,
                    dim,
                });
            }
            let needed = dim
                .checked_mul(4)
                .ok_or_else(|| MoeLowerError::ArithmeticOverflow {
                    group: group.group.clone(),
                    layer: group.layer,
                    rank,
                    index: down_index,
                    what: "f32_bytes",
                })?;
            let actual = cap(out);
            if actual < needed {
                return Err(MoeLowerError::CapacityMismatch {
                    group: group.group.clone(),
                    layer: group.layer,
                    rank,
                    index: down_index,
                    expected_bytes: needed,
                    actual_bytes: actual,
                });
            }
            Ok(RankProtocol {
                kind: RankProtocolKind::SelfCombiningF32,
                down_index,
                combine_index: None,
                convert_index: None,
                dim,
                batched: None,
                grouped: None,
            })
        }
        DownEvidence::SelfCombiningI64 { out, batch_size } => {
            // Grouped permutation machinery is never admitted on the
            // self-combining i64 path.
            reject_stray_grouped_ops(group, rank, &ops)?;
            // Any combine anywhere would double-accumulate.
            if let Some(&(index, _, _)) = ops.combines.first() {
                return Err(MoeLowerError::CombineAfterSelfCombiningDown {
                    group: group.group.clone(),
                    layer: group.layer,
                    rank,
                    index,
                });
            }
            let conversions = &ops.conversions;
            let (convert_index, phase, convert_step) = match conversions.as_slice() {
                [] => {
                    return Err(MoeLowerError::MissingI64Conversion {
                        group: group.group.clone(),
                        layer: group.layer,
                        rank,
                    });
                }
                [_first, second, ..] => {
                    return Err(MoeLowerError::DuplicateI64Conversion {
                        group: group.group.clone(),
                        layer: group.layer,
                        rank,
                        index: second.0,
                    });
                }
                [only] => (only.0, only.1, only.2),
            };
            if phase != "finish" {
                return Err(MoeLowerError::MisplacedI64Conversion {
                    group: group.group.clone(),
                    layer: group.layer,
                    rank,
                    index: convert_index,
                    phase,
                });
            }
            let Step::ConvertI64ToF32 { src, dst, n } = convert_step else {
                unreachable!("conversion scan only collects ConvertI64ToF32 steps")
            };
            let (src, dst) = (*src, *dst);
            if !same_buffer(src, out) {
                return Err(MoeLowerError::I64ConversionSourceMismatch {
                    group: group.group.clone(),
                    layer: group.layer,
                    rank,
                    index: convert_index,
                });
            }
            // Logical dimension: batch 1 uses the checked shape product; a
            // batched chain must satisfy the full batched invariants first
            // (checked geometry, chain dataflow, alignment, operand
            // capacities) and yields its checked logical dimension plus the
            // typed cross-rank signature.
            let (down_dim, batched) = if batch_size == 1 {
                (
                    out.shape
                        .iter()
                        .try_fold(1usize, |acc, &dim| acc.checked_mul(dim))
                        .ok_or_else(|| MoeLowerError::ArithmeticOverflow {
                            group: group.group.clone(),
                            layer: group.layer,
                            rank,
                            index: convert_index,
                            what: "logical_shape",
                        })?,
                    None,
                )
            } else {
                let (dim, signature) = validate_batched_i64_chain(
                    group, rank, phases, down_index, &ops, &cap, batch_size, out,
                )?;
                (dim, Some(signature))
            };
            if *n == 0 || down_dim != *n {
                return Err(MoeLowerError::I64DimensionMismatch {
                    group: group.group.clone(),
                    layer: group.layer,
                    rank,
                    index: convert_index,
                    expected: down_dim,
                    actual: *n,
                });
            }
            // Checked byte arithmetic: an n whose n*8 or n*4 requirement
            // overflows usize is reported as arithmetic overflow before any
            // capacity comparison. Capacities are lower bounds only: padded
            // and non-multiple allocations pass.
            let needed_i64 = n
                .checked_mul(8)
                .ok_or_else(|| MoeLowerError::ArithmeticOverflow {
                    group: group.group.clone(),
                    layer: group.layer,
                    rank,
                    index: convert_index,
                    what: "i64_bytes",
                })?;
            let needed_f32 = n
                .checked_mul(4)
                .ok_or_else(|| MoeLowerError::ArithmeticOverflow {
                    group: group.group.clone(),
                    layer: group.layer,
                    rank,
                    index: convert_index,
                    what: "f32_bytes",
                })?;
            for (tensor, needed) in [(out, needed_i64), (src, needed_i64), (dst, needed_f32)] {
                let actual = cap(tensor);
                if actual < needed {
                    return Err(MoeLowerError::CapacityMismatch {
                        group: group.group.clone(),
                        layer: group.layer,
                        rank,
                        index: convert_index,
                        expected_bytes: needed,
                        actual_bytes: actual,
                    });
                }
            }
            Ok(RankProtocol {
                kind: RankProtocolKind::SelfCombiningI64,
                down_index,
                combine_index: None,
                convert_index: Some(convert_index),
                dim: *n,
                batched,
                grouped: None,
            })
        }
    }
}

/// Chain authority for a batched indexed I64 program (`batch_size > 1`):
/// exactly one indexed GateUp and one `MoeActivation`, matching nonzero
/// batch/k_top/expert metadata, the shared top-k index buffer, connected
/// gate-up → activation → down dataflow, routed rows equal to the checked
/// `batch_size * k_top` product, 256-aligned MQ2 contractions, an I64 output
/// of exactly `[batch_size, hidden]`, and real-buffer capacities for every
/// operand. Returns the checked logical output dimension plus the typed
/// cross-rank signature. Everything fails closed: zero geometry, overflow,
/// misalignment, disconnection, and undersized operands are contextual
/// errors, never a scalar fallback.
#[expect(
    clippy::result_large_err,
    clippy::too_many_arguments,
    reason = "MoeLowerError deliberately carries rich group/layer/rank diagnostics; boxing would fragment error reporting. The validation helper threads the same contextual (group, rank, ops, cap) bundle as its sibling validators."
)]
fn validate_batched_i64_chain<'a, F>(
    group: &ExpertGroupPlan,
    rank: usize,
    phases: &'a RoutedMoePhases<Step<'a>>,
    down_index: usize,
    ops: &RankOps<'a>,
    cap: &F,
    batch_size: usize,
    out: &'a GpuTensor,
) -> Result<(usize, BatchedI64Signature), MoeLowerError>
where
    F: Fn(&GpuTensor) -> usize,
{
    let Step::IndexedMoeGemv {
        experts: down_experts,
        which: MoeProj::DownResidualI64 {
            topk_weights: down_topk_weights,
        },
        topk_indices: down_topk,
        input: down_input,
        k_top: down_k_top,
        ..
    } = &phases.down[0]
    else {
        unreachable!("i64 evidence only collects IndexedMoeGemv DownResidualI64 steps")
    };

    // Nonzero geometry: a zero routed batch or k_top is rejected, never
    // interpreted as the scalar protocol.
    if batch_size == 0 {
        return Err(MoeLowerError::BatchedI64ZeroGeometry {
            group: group.group.clone(),
            layer: group.layer,
            rank,
            index: down_index,
            field: "batch_size",
        });
    }
    if *down_k_top == 0 {
        return Err(MoeLowerError::BatchedI64ZeroGeometry {
            group: group.group.clone(),
            layer: group.layer,
            rank,
            index: down_index,
            field: "k_top",
        });
    }
    // Expert geometry must be nonzero; a zero expert_m/expert_k would pass
    // the 256-alignment check (0 % 256 == 0) and silently collapse the
    // contraction.
    for (field, value) in [
        ("n_experts", down_experts.n_experts),
        ("expert_m", down_experts.expert_m),
        ("expert_k", down_experts.expert_k),
    ] {
        if value == 0 {
            return Err(MoeLowerError::BatchedI64ZeroGeometry {
                group: group.group.clone(),
                layer: group.layer,
                rank,
                index: down_index,
                field,
            });
        }
    }
    // Routing cannot select more experts than exist.
    if *down_k_top > down_experts.n_experts {
        return Err(MoeLowerError::BatchedI64KTopExceedsExperts {
            group: group.group.clone(),
            layer: group.layer,
            rank,
            index: down_index,
            k_top: *down_k_top,
            n_experts: down_experts.n_experts,
        });
    }

    // Exactly one indexed gate-up, in the semantic gate_up phase.
    let (gate_up_index, gate_up_phase, gate_up_step) = match ops.indexed_gate_ups.as_slice() {
        [] => {
            return Err(MoeLowerError::BatchedI64GateUpCount {
                group: group.group.clone(),
                layer: group.layer,
                rank,
                expected: 1,
                actual: 0,
            });
        }
        [_first, _second, ..] => {
            return Err(MoeLowerError::BatchedI64GateUpCount {
                group: group.group.clone(),
                layer: group.layer,
                rank,
                expected: 1,
                actual: ops.indexed_gate_ups.len(),
            });
        }
        [only] => (only.0, only.1, only.2),
    };
    if gate_up_phase != "gate_up" {
        return Err(MoeLowerError::BatchedI64ChainMismatch {
            group: group.group.clone(),
            layer: group.layer,
            rank,
            index: gate_up_index,
            field: "gate_up_phase",
        });
    }

    // Exactly one activation, in the semantic activation phase.
    let (act_index, act_phase, act_step) = match ops.activations.as_slice() {
        [] => {
            return Err(MoeLowerError::BatchedI64ActivationCount {
                group: group.group.clone(),
                layer: group.layer,
                rank,
                expected: 1,
                actual: 0,
            });
        }
        [_first, _second, ..] => {
            return Err(MoeLowerError::BatchedI64ActivationCount {
                group: group.group.clone(),
                layer: group.layer,
                rank,
                expected: 1,
                actual: ops.activations.len(),
            });
        }
        [only] => (only.0, only.1, only.2),
    };
    if act_phase != "activation" {
        return Err(MoeLowerError::BatchedI64ChainMismatch {
            group: group.group.clone(),
            layer: group.layer,
            rank,
            index: act_index,
            field: "activation_phase",
        });
    }

    let Step::IndexedMoeGemv {
        experts: gate_up_experts,
        topk_indices: gate_up_topk,
        input: gate_up_input,
        out: gate_out,
        which: MoeProj::GateUp { up_out },
        k_top: gate_up_k_top,
        batch_size: gate_up_batch,
    } = gate_up_step
    else {
        unreachable!("indexed gate-up scan only collects IndexedMoeGemv GateUp steps")
    };
    let Step::MoeActivation {
        gate: act_gate,
        up: act_up,
        rot_out: act_rot_out,
        inter: act_inter,
        k_top: act_rows,
        ..
    } = act_step
    else {
        unreachable!("activation scan only collects MoeActivation steps")
    };

    // Metadata must agree exactly with the down step.
    for (field, expected, actual) in [
        ("batch_size", batch_size, *gate_up_batch),
        ("k_top", *down_k_top, *gate_up_k_top),
        (
            "n_experts",
            down_experts.n_experts,
            gate_up_experts.n_experts,
        ),
        ("expert_m", down_experts.expert_m, gate_up_experts.expert_m),
        ("expert_k", down_experts.expert_k, gate_up_experts.expert_k),
    ] {
        if expected != actual {
            return Err(MoeLowerError::BatchedI64ChainMismatch {
                group: group.group.clone(),
                layer: group.layer,
                rank,
                index: gate_up_index,
                field,
            });
        }
    }
    if down_experts.dtype != gate_up_experts.dtype {
        return Err(MoeLowerError::BatchedI64ChainMismatch {
            group: group.group.clone(),
            layer: group.layer,
            rank,
            index: gate_up_index,
            field: "dtype",
        });
    }
    // The batched protocol is MQ2G256Lloyd-only. A coherent chain with any
    // other dtype (e.g. MQ3G256Lloyd) is rejected at lowering — the step
    // executor must never be asked to fall back to scalar kernels for a
    // batched program.
    if down_experts.dtype != DType::MQ2G256Lloyd {
        return Err(MoeLowerError::BatchedI64Dtype {
            group: group.group.clone(),
            layer: group.layer,
            rank,
            index: down_index,
            expected: DType::MQ2G256Lloyd,
            actual: down_experts.dtype,
        });
    }
    // The activation's inter must be the connected expert_m: nonzero, and
    // equal to the gate-up/down contraction width so the gate/up/rotation
    // capacities can be derived from trusted expert geometry.
    if *act_inter == 0 {
        return Err(MoeLowerError::BatchedI64ZeroGeometry {
            group: group.group.clone(),
            layer: group.layer,
            rank,
            index: act_index,
            field: "activation_inter",
        });
    }
    if *act_inter != down_experts.expert_m {
        return Err(MoeLowerError::BatchedI64ChainMismatch {
            group: group.group.clone(),
            layer: group.layer,
            rank,
            index: act_index,
            field: "activation_inter",
        });
    }

    // The same top-k index buffer feeds both projections.
    if !same_buffer(down_topk, gate_up_topk) {
        return Err(MoeLowerError::BatchedI64ChainMismatch {
            group: group.group.clone(),
            layer: group.layer,
            rank,
            index: down_index,
            field: "topk_indices",
        });
    }
    // Dataflow: the activation consumes the gate-up outputs and the down
    // consumes the activation's rotated output.
    if !same_buffer(act_gate, gate_out) {
        return Err(MoeLowerError::BatchedI64ChainMismatch {
            group: group.group.clone(),
            layer: group.layer,
            rank,
            index: act_index,
            field: "activation_gate",
        });
    }
    if !same_buffer(act_up, up_out) {
        return Err(MoeLowerError::BatchedI64ChainMismatch {
            group: group.group.clone(),
            layer: group.layer,
            rank,
            index: act_index,
            field: "activation_up",
        });
    }
    let down_input_tensor = match down_input {
        GemvInput::Raw(t) | GemvInput::Prerotated(t) => *t,
    };
    let gate_up_input_tensor = match gate_up_input {
        GemvInput::Raw(t) | GemvInput::Prerotated(t) => *t,
    };
    if !same_buffer(down_input_tensor, act_rot_out) {
        return Err(MoeLowerError::BatchedI64ChainMismatch {
            group: group.group.clone(),
            layer: group.layer,
            rank,
            index: down_index,
            field: "down_input",
        });
    }

    // Routed rows: the activation must span exactly the checked batch*k_top
    // product.
    let rows =
        batch_size
            .checked_mul(*down_k_top)
            .ok_or_else(|| MoeLowerError::ArithmeticOverflow {
                group: group.group.clone(),
                layer: group.layer,
                rank,
                index: act_index,
                what: "activation_rows",
            })?;
    if *act_rows != rows {
        return Err(MoeLowerError::BatchedI64ActivationRows {
            group: group.group.clone(),
            layer: group.layer,
            rank,
            index: act_index,
            expected: rows,
            actual: *act_rows,
        });
    }

    // Kernel ABI range (kernarg group): the MQ2 `_batched_k4` launchers
    // convert m/k/k_top to i32 kernargs (`m as i32`, `k as i32`,
    // `k_top as i32` in `deepseek4_gemv_mq2g256_lloyd_moe_gate_up_indexed_
    // batched_k4` and `moe_down_mq2g256_lloyd_residual_i64_indexed_batched`).
    // The gate-up m is the checked 2*expert_m product; every value must
    // convert exactly.
    let gate_up_m =
        down_experts
            .expert_m
            .checked_mul(2)
            .ok_or_else(|| MoeLowerError::ArithmeticOverflow {
                group: group.group.clone(),
                layer: group.layer,
                rank,
                index: gate_up_index,
                what: "gate_up_m",
            })?;
    let i32_max = i32::MAX as usize;
    for (field, value, index) in [
        ("gate_up_m", gate_up_m, gate_up_index),
        ("expert_k", down_experts.expert_k, gate_up_index),
        ("expert_m", down_experts.expert_m, down_index),
        ("k_top", *down_k_top, down_index),
    ] {
        if value > i32_max {
            return Err(MoeLowerError::BatchedI64AbiRange {
                group: group.group.clone(),
                layer: group.layer,
                rank,
                index,
                field,
                max: i32_max,
                actual: value,
            });
        }
    }

    // MQ2 batched contractions are 256-aligned: the gate-up hidden (expert_k)
    // and the i64-down inter_local (expert_m). The literal is the frozen
    // batched-protocol value from the MQ2 `_batched_k4` kernel contract.
    if gate_up_experts.expert_k % 256 != 0 {
        return Err(MoeLowerError::BatchedI64Alignment {
            group: group.group.clone(),
            layer: group.layer,
            rank,
            index: gate_up_index,
            field: "gate_up_hidden",
            expected: 256,
            actual: gate_up_experts.expert_k,
        });
    }
    if down_experts.expert_m % 256 != 0 {
        return Err(MoeLowerError::BatchedI64Alignment {
            group: group.group.clone(),
            layer: group.layer,
            rank,
            index: down_index,
            field: "down_inter_local",
            expected: 256,
            actual: down_experts.expert_m,
        });
    }

    // The I64 output is exactly [batch_size, hidden].
    let expected_shape = [batch_size, down_experts.expert_k];
    if out.shape.len() != 2
        || out.shape[0] != expected_shape[0]
        || out.shape[1] != expected_shape[1]
    {
        return Err(MoeLowerError::BatchedI64OutputShape {
            group: group.group.clone(),
            layer: group.layer,
            rank,
            index: down_index,
            expected: expected_shape,
            actual: out.shape.clone(),
        });
    }
    let logical_dim = out
        .shape
        .iter()
        .try_fold(1usize, |acc, &dim| acc.checked_mul(dim))
        .ok_or_else(|| MoeLowerError::ArithmeticOverflow {
            group: group.group.clone(),
            layer: group.layer,
            rank,
            index: down_index,
            what: "logical_shape",
        })?;

    // Kernel ABI range (grid-z group): both `_batched_k4` launchers launch
    // `[m as u32, k_top as u32, batch_size as u32]`; batch_size is ONLY a u32
    // grid dimension (never a kernarg), so its bound is u32::MAX. Checked
    // after the logical dimension so a shape-product overflow is still
    // reported as arithmetic overflow first.
    if batch_size > u32::MAX as usize {
        return Err(MoeLowerError::BatchedI64AbiRange {
            group: group.group.clone(),
            layer: group.layer,
            rank,
            index: down_index,
            field: "batch_size",
            max: u32::MAX as usize,
            actual: batch_size,
        });
    }

    // Operand capacities from the actual buffer sizes: top-k indices+weights
    // slots*4, gate-up input batch*hidden*4, gate/up/rotation
    // slots*inter_local*4, all checked against overflow.
    let slots_bytes = rows
        .checked_mul(4)
        .ok_or_else(|| MoeLowerError::ArithmeticOverflow {
            group: group.group.clone(),
            layer: group.layer,
            rank,
            index: down_index,
            what: "slots_bytes",
        })?;
    let gate_up_input_dim = batch_size
        .checked_mul(gate_up_experts.expert_k)
        .ok_or_else(|| MoeLowerError::ArithmeticOverflow {
            group: group.group.clone(),
            layer: group.layer,
            rank,
            index: gate_up_index,
            what: "gate_up_input_dim",
        })?;
    let gate_up_input_bytes =
        gate_up_input_dim
            .checked_mul(4)
            .ok_or_else(|| MoeLowerError::ArithmeticOverflow {
                group: group.group.clone(),
                layer: group.layer,
                rank,
                index: gate_up_index,
                what: "gate_up_input_bytes",
            })?;
    let inter_rows = rows.checked_mul(down_experts.expert_m).ok_or_else(|| {
        MoeLowerError::ArithmeticOverflow {
            group: group.group.clone(),
            layer: group.layer,
            rank,
            index: act_index,
            what: "inter_rows",
        }
    })?;
    let inter_bytes =
        inter_rows
            .checked_mul(4)
            .ok_or_else(|| MoeLowerError::ArithmeticOverflow {
                group: group.group.clone(),
                layer: group.layer,
                rank,
                index: act_index,
                what: "inter_bytes",
            })?;
    // Expert pointer tables: `n_experts` u64 device pointers per the kernel
    // ABI (8 bytes per entry).
    let pointer_table_bytes = down_experts
        .n_experts
        .checked_mul(DEVICE_POINTER_BYTES)
        .ok_or_else(|| MoeLowerError::ArithmeticOverflow {
            group: group.group.clone(),
            layer: group.layer,
            rank,
            index: down_index,
            what: "pointer_table_bytes",
        })?;
    for (tensor, needed, index) in [
        (
            gate_up_experts.gate_up_ptrs,
            pointer_table_bytes,
            gate_up_index,
        ),
        (down_experts.down_ptrs, pointer_table_bytes, down_index),
        (*down_topk, slots_bytes, down_index),
        (*down_topk_weights, slots_bytes, down_index),
        (gate_up_input_tensor, gate_up_input_bytes, gate_up_index),
        (*act_gate, inter_bytes, act_index),
        (*act_up, inter_bytes, act_index),
        (*act_rot_out, inter_bytes, act_index),
    ] {
        let actual = cap(tensor);
        if actual < needed {
            return Err(MoeLowerError::CapacityMismatch {
                group: group.group.clone(),
                layer: group.layer,
                rank,
                index,
                expected_bytes: needed,
                actual_bytes: actual,
            });
        }
    }
    Ok((
        logical_dim,
        BatchedI64Signature {
            batch_size,
            k_top: *down_k_top,
            n_experts: down_experts.n_experts,
            inter: down_experts.expert_m,
            hidden: down_experts.expert_k,
            dtype: down_experts.dtype,
            activation_rows: rows,
        },
    ))
}

/// Capacity-aware protocol parser: every rank is parsed before any
/// flattening, and every rank must agree on dimensions, protocol, and
/// absolute indices. `cap` yields a tensor's byte capacity.
#[expect(
    clippy::result_large_err,
    reason = "MoeLowerError deliberately carries rich group/layer/rank diagnostics; boxing would fragment error reporting"
)]
fn parse_step_protocol<'a, F>(
    group: &ExpertGroupPlan,
    policy: &MoEExecutionPolicy,
    ranks: &[RoutedMoeStepPhases<'a>],
    cap: F,
    deferred_combine: bool,
) -> Result<(ValidatedProtocol, RankProtocolKind), MoeLowerError>
where
    F: Fn(&GpuTensor) -> usize,
{
    if ranks.len() != group.group_size {
        return Err(MoeLowerError::RankCountMismatch {
            group: group.group.clone(),
            layer: group.layer,
            expected: group.group_size,
            actual: ranks.len(),
        });
    }
    let lengths = ranks[0].lengths();
    if lengths[1] == 0 {
        return Err(MoeLowerError::MissingPhase {
            group: group.group.clone(),
            layer: group.layer,
            phase: "gate_up",
        });
    }
    if lengths[2] == 0 {
        return Err(MoeLowerError::MissingPhase {
            group: group.group.clone(),
            layer: group.layer,
            phase: "activation",
        });
    }
    for (rank, phases) in ranks.iter().enumerate().skip(1) {
        if phases.lengths() != lengths {
            return Err(MoeLowerError::RankPhaseMismatch {
                group: group.group.clone(),
                layer: group.layer,
                rank,
                expected: lengths,
                actual: phases.lengths(),
            });
        }
    }

    let first = parse_rank_protocol(group, 0, deferred_combine, &ranks[0], &cap)?;
    for (rank, phases) in ranks.iter().enumerate().skip(1) {
        let next = parse_rank_protocol(group, rank, deferred_combine, phases, &cap)?;
        if !same_rank_protocol(&first, &next) {
            return Err(MoeLowerError::RankProtocolMismatch {
                group: group.group.clone(),
                layer: group.layer,
                rank,
                expected: rank_protocol_label(&first),
                actual: rank_protocol_label(&next),
            });
        }
    }

    let protocol = match first.kind {
        RankProtocolKind::ExpandedIndexed
        | RankProtocolKind::ExpandedGrouped
        | RankProtocolKind::SelfCombiningF32 => match policy.kind() {
            MoEExecutionKind::Single => ValidatedProtocol::Single,
            MoEExecutionKind::Tp | MoEExecutionKind::Ep => ValidatedProtocol::ParallelF32 {
                anchor: first.combine_index.unwrap_or(first.down_index),
                dim: first.dim,
            },
        },
        // Deferred-expanded: the zero-combine shape is inherently rank-local
        // (the next-layer fused consumer folds the partial on the same
        // device); a parallel axis has no combine anchor, so only the Single
        // executor is admitted.
        RankProtocolKind::ExpandedIndexedDeferred => match policy.kind() {
            MoEExecutionKind::Single => ValidatedProtocol::Single,
            MoEExecutionKind::Tp | MoEExecutionKind::Ep => {
                return Err(MoeLowerError::DeferredCombineOnParallelAxis {
                    group: group.group.clone(),
                    layer: group.layer,
                });
            }
        },
        RankProtocolKind::SelfCombiningI64 => {
            let convert = first
                .convert_index
                .expect("i64 evidence always sets a conversion");
            match policy.kind() {
                MoEExecutionKind::Tp => ValidatedProtocol::TpI64 {
                    down: first.down_index,
                    dim: first.dim,
                },
                MoEExecutionKind::Ep => ValidatedProtocol::EpLocalI64 {
                    down: first.down_index,
                    convert,
                    dim: first.dim,
                },
                MoEExecutionKind::Single => {
                    return Err(MoeLowerError::I64OnNonAdmittedAxis {
                        group: group.group.clone(),
                        layer: group.layer,
                    });
                }
            }
        }
    };
    Ok((protocol, first.kind))
}

/// Production entry: parse the launch protocol from the concrete borrowed
/// `Step` programs, measuring capacities from the real device buffers.
#[expect(
    clippy::result_large_err,
    reason = "MoeLowerError deliberately carries rich group/layer/rank diagnostics; boxing would fragment error reporting"
)]
fn validate_step_protocol<'a>(
    group: &ExpertGroupPlan,
    policy: &MoEExecutionPolicy,
    ranks: &[RoutedMoeStepPhases<'a>],
    deferred_combine: bool,
) -> Result<(ValidatedProtocol, RankProtocolKind), MoeLowerError> {
    parse_step_protocol(
        group,
        policy,
        ranks,
        |tensor| tensor.buf.size(),
        deferred_combine,
    )
}

/// Construct the sealed lowered program from the validated protocol. The
/// schedule (collectives, zeroing, conversion placement) is derived only from
/// `ValidatedProtocol` plus `ExpertGroupPlan.collective`.
fn construct_lowered<'mesh, 'step>(
    group: &ExpertGroupPlan,
    policy: &'mesh MoEExecutionPolicy,
    ranks: Vec<RoutedMoeStepPhases<'step>>,
    protocol: ValidatedProtocol,
) -> LoweredMoeProgram<'mesh, 'step> {
    let step_count = ranks[0].lengths().into_iter().sum();
    match protocol {
        ValidatedProtocol::Single => {
            let mut ranks = ranks;
            LoweredMoeProgram {
                inner: LoweredMoeProgramInner::Single {
                    steps: ranks
                        .pop()
                        .expect("validated one single-rank program")
                        .into_steps(),
                },
            }
        }
        ValidatedProtocol::ParallelF32 { anchor, dim } => {
            let mut collectives = (0..step_count)
                .map(|_| StepCollective::None)
                .collect::<Vec<_>>();
            let mut zero = vec![false; step_count];
            let axis = group
                .collective
                .expect("parallel executor selection validated collective")
                .axis();
            collectives[anchor] = StepCollective::AllReduce { kind: axis, dim };
            // The EP partial accumulates (combine or residual-fused down), so
            // the anchor is always zeroed before it runs.
            zero[anchor] = true;
            LoweredMoeProgram {
                inner: LoweredMoeProgramInner::Parallel {
                    mesh: policy.mesh(),
                    per_rank_steps: ranks.into_iter().map(RoutedMoePhases::into_steps).collect(),
                    collectives,
                    zero_before: zero,
                },
            }
        }
        ValidatedProtocol::TpI64 { down, dim } => {
            let mut collectives = (0..step_count)
                .map(|_| StepCollective::None)
                .collect::<Vec<_>>();
            let mut zero = vec![false; step_count];
            collectives[down] = StepCollective::AllReduceI64Tp { dim };
            // The i64 accumulator is pre-zeroed (8 bytes per element).
            zero[down] = true;
            LoweredMoeProgram {
                inner: LoweredMoeProgramInner::Parallel {
                    mesh: policy.mesh(),
                    per_rank_steps: ranks.into_iter().map(RoutedMoePhases::into_steps).collect(),
                    collectives,
                    zero_before: zero,
                },
            }
        }
        ValidatedProtocol::EpLocalI64 { down, convert, dim } => {
            let mut collectives = (0..step_count)
                .map(|_| StepCollective::None)
                .collect::<Vec<_>>();
            let mut zero = vec![false; step_count];
            collectives[down] = StepCollective::ZeroI64Only { dim };
            // The local i64 accumulator is pre-zeroed (8 bytes per element).
            zero[down] = true;
            let axis = group
                .collective
                .expect("parallel executor selection validated collective")
                .axis();
            // The FP32 EP all-reduce lands on the conversion step, which
            // writes its destination fresh.
            collectives[convert] = StepCollective::AllReduce { kind: axis, dim };
            zero[convert] = false;
            LoweredMoeProgram {
                inner: LoweredMoeProgramInner::Parallel {
                    mesh: policy.mesh(),
                    per_rank_steps: ranks.into_iter().map(RoutedMoePhases::into_steps).collect(),
                    collectives,
                    zero_before: zero,
                },
            }
        }
    }
}

/// Deterministic admission-set validation for a resolved group plan: the
/// allowed-execution collection must be non-empty and duplicate-free before
/// any executor selection, identity membership, protocol parsing, or schedule
/// construction runs. The duplicate is reported with its canonical label and
/// the group/layer context. Allocation-free declaration-order scan: each
/// identity is checked against every preceding one, so the FIRST duplicate in
/// declaration order is always the one reported. No string parsing — identity
/// equality is typed.
#[expect(
    clippy::result_large_err,
    reason = "MoeLowerError deliberately carries rich group/layer/rank diagnostics; boxing would fragment error reporting"
)]
fn validate_allowed_executions(group: &ExpertGroupPlan) -> Result<(), MoeLowerError> {
    if group.allowed_executions.is_empty() {
        return Err(MoeLowerError::EmptyAllowedExecutions {
            group: group.group.clone(),
            layer: group.layer,
        });
    }
    for (idx, identity) in group.allowed_executions.iter().enumerate() {
        if group.allowed_executions[..idx].contains(identity) {
            return Err(MoeLowerError::DuplicateAllowedExecutions {
                group: group.group.clone(),
                layer: group.layer,
                identity: identity.canonical_label(),
            });
        }
    }
    Ok(())
}

#[expect(
    clippy::result_large_err,
    reason = "MoeLowerError deliberately carries rich group/layer/rank diagnostics; boxing would fragment error reporting"
)]
pub fn lower_moe_steps<'mesh, 'step>(
    group: &ExpertGroupPlan,
    policy: &'mesh MoEExecutionPolicy,
    parts: MoeProgramParts<'step>,
) -> Result<LoweredMoeProgram<'mesh, 'step>, MoeLowerError> {
    validate_allowed_executions(group)?;
    select_moe_executor(group, policy)?;
    validate_program_identity(group, parts.router.selection(), parts.execution)?;
    let (protocol, kind) =
        validate_step_protocol(group, policy, &parts.ranks, parts.deferred_combine)?;
    // A concrete grouped protocol requires the GroupedQuantized execution
    // plan, even when IndexedQuantized is also declared: membership alone
    // must never admit a grouped chain mislabeled as indexed.
    if kind == RankProtocolKind::ExpandedGrouped
        && parts.execution != ExpertExecutionPlan::GroupedQuantized
    {
        return Err(MoeLowerError::ExecutionIdentityMismatch {
            group: group.group.clone(),
            layer: group.layer,
            expected: "grouped_quantized".to_owned(),
            actual: ExpertExecutionIdentity::from(parts.execution).canonical_label(),
        });
    }
    Ok(construct_lowered(group, policy, parts.ranks, protocol))
}

pub enum MoeExecutionTarget<'a> {
    Single {
        gpu: &'a mut rdna_compute::Gpu,
        ctx: &'a DispatchCtx,
    },
    Parallel {
        gpus: &'a mut crate::multi_gpu::Gpus,
    },
}

pub fn execute_lowered_moe(
    program: &LoweredMoeProgram<'_, '_>,
    target: MoeExecutionTarget<'_>,
) -> Result<(), DispatchError> {
    match (&program.inner, target) {
        (LoweredMoeProgramInner::Single { steps }, MoeExecutionTarget::Single { gpu, ctx }) => {
            execute_steps_mesh(&DeviceMesh::single(), gpu, ctx, steps)
        }
        (
            LoweredMoeProgramInner::Parallel {
                mesh,
                per_rank_steps,
                collectives,
                zero_before,
            },
            MoeExecutionTarget::Parallel { gpus },
        ) => {
            if gpus.devices.len() != mesh.n_devices() {
                return Err(DispatchError::Hip(format!(
                    "MoE target has {} devices, but lowered mesh requires {}",
                    gpus.devices.len(),
                    mesh.n_devices()
                )));
            }
            gpus.weight_origin_in(mesh, 0).map_err(|error| {
                DispatchError::Hip(format!("MoE target mesh identity mismatch: {error}"))
            })?;
            execute_steps_parallel(mesh, gpus, per_rank_steps, collectives, zero_before)
        }
        _ => Err(DispatchError::Hip(
            "MoE lowered program and execution target disagree".into(),
        )),
    }
}

#[cfg(test)]
fn collective_count(collectives: &[StepCollective]) -> usize {
    collectives
        .iter()
        .filter(|collective| {
            matches!(
                collective,
                StepCollective::AllReduce { .. } | StepCollective::AllReduceI64Tp { .. }
            )
        })
        .count()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::multi_gpu::{DeviceMesh, DimKind};
    use crate::tp_shard::ExpertAssign;
    use crate::weight_manifest::{
        ExpertGroupPlan, ExpertParallelism, ExpertPostCombineAllReduce, ExpertResourceRequirements,
        ExpertSourceLayout,
    };

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    enum Marker {
        Router,
        GateUp,
        Activation,
        Down,
        Combine,
        Finish,
    }

    fn group(parallelism: ExpertParallelism, group_size: usize) -> ExpertGroupPlan {
        let collective = match parallelism {
            ExpertParallelism::Single => None,
            ExpertParallelism::TensorParallel => Some(ExpertPostCombineAllReduce::TensorParallel),
            ExpertParallelism::ExpertParallel => Some(ExpertPostCombineAllReduce::ExpertParallel),
        };
        ExpertGroupPlan {
            group: "test".into(),
            layer: Some(0),
            n_experts: 4,
            group_size,
            parallelism,
            assignment: ExpertAssign::Stride,
            experts: Vec::new(),
            source_layout: ExpertSourceLayout::PackedFused {
                gate_up: "gate_up".into(),
                down: "down".into(),
                sidecars: Vec::new(),
            },
            resources: ExpertResourceRequirements {
                bytes_per_expert: 1,
                alignment: 1,
            },
            router: "router".into(),
            router_identity: "softmax_topk".into(),
            allowed_executions: vec![ExpertExecutionIdentity::IndexedQuantized],
            collective,
        }
    }

    fn group_with_identities(
        router_identity: &str,
        allowed: &[ExpertExecutionIdentity],
    ) -> ExpertGroupPlan {
        let mut group = group(ExpertParallelism::Single, 1);
        group.router = "router".into();
        group.router_identity = router_identity.into();
        group.allowed_executions = allowed.to_vec();
        group
    }

    fn phases() -> RoutedMoePhases<Marker> {
        RoutedMoePhases {
            router: vec![Marker::Router],
            gate_up: vec![Marker::GateUp],
            activation: vec![Marker::Activation],
            down: vec![Marker::Down],
            combine: vec![Marker::Combine],
            finish: vec![Marker::Finish],
        }
    }

    fn tp_policy(ranks: usize) -> MoEExecutionPolicy {
        MoEExecutionPolicy::new(
            MoEExecutionKind::Tp,
            DeviceMesh::rect(&[(DimKind::Tp, ranks)]),
        )
        .unwrap()
    }

    fn ep_policy(ranks: usize) -> MoEExecutionPolicy {
        MoEExecutionPolicy::new(
            MoEExecutionKind::Ep,
            DeviceMesh::rect(&[(DimKind::Ep, ranks)]),
        )
        .unwrap()
    }

    #[test]
    fn single_policy_has_no_axis_and_one_rank() {
        let policy = MoEExecutionPolicy::single();
        assert_eq!(policy.kind(), MoEExecutionKind::Single);
        assert_eq!(policy.axis(), None);
        assert_eq!(policy.rank_count(), 1);
    }

    #[test]
    fn tp_policy_uses_named_axis_and_rank_count() {
        let policy =
            MoEExecutionPolicy::new(MoEExecutionKind::Tp, DeviceMesh::rect(&[(DimKind::Tp, 2)]))
                .unwrap();
        assert_eq!(policy.axis(), Some(DimKind::Tp));
        assert_eq!(policy.rank_count(), 2);
    }

    #[test]
    fn ep_policy_uses_named_axis_and_rank_count() {
        let policy =
            MoEExecutionPolicy::new(MoEExecutionKind::Ep, DeviceMesh::rect(&[(DimKind::Ep, 3)]))
                .unwrap();
        assert_eq!(policy.axis(), Some(DimKind::Ep));
        assert_eq!(policy.rank_count(), 3);
    }

    #[test]
    fn pp_only_mesh_is_valid_for_single() {
        let policy = MoEExecutionPolicy::new(
            MoEExecutionKind::Single,
            DeviceMesh::rect(&[(DimKind::Pp, 2)]),
        )
        .unwrap();
        assert_eq!(policy.mesh().size_of(DimKind::Pp), 2);
    }

    #[test]
    fn policy_rejects_missing_required_axis() {
        assert!(matches!(
            MoEExecutionPolicy::new(MoEExecutionKind::Tp, DeviceMesh::rect(&[(DimKind::Ep, 2)])),
            Err(MoEExecutionPolicyError::MissingRequiredAxis { .. })
        ));
    }

    #[test]
    fn policy_rejects_competing_axis() {
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
    fn size_one_named_axes_are_valid() {
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
    fn lowerer_orders_all_phases_and_omits_optional_phases() {
        // Marker flattening proves phase ordering only; it never produces a
        // launchable lowered value (launch authority is derived exclusively
        // from concrete Steps by `lower_moe_steps`).
        assert_eq!(
            phases().into_steps(),
            vec![
                Marker::Router,
                Marker::GateUp,
                Marker::Activation,
                Marker::Down,
                Marker::Combine,
                Marker::Finish,
            ]
        );

        let mut omitted = phases();
        omitted.router.clear();
        omitted.combine.clear();
        assert_eq!(
            omitted.into_steps(),
            vec![
                Marker::GateUp,
                Marker::Activation,
                Marker::Down,
                Marker::Finish
            ]
        );
    }

    #[test]
    fn lowerer_rejects_policy_cardinality_and_phase_errors() {
        assert!(
            select_moe_executor(&group(ExpertParallelism::TensorParallel, 2), &ep_policy(2))
                .is_err()
        );
        assert!(
            select_moe_executor(&group(ExpertParallelism::ExpertParallel, 2), &tp_policy(2))
                .is_err()
        );
        assert!(
            select_moe_executor(&group(ExpertParallelism::TensorParallel, 3), &tp_policy(2))
                .is_err()
        );

        let mut malformed_single = group(ExpertParallelism::Single, 1);
        malformed_single.collective = Some(ExpertPostCombineAllReduce::TensorParallel);
        assert!(select_moe_executor(&malformed_single, &MoEExecutionPolicy::single()).is_err());
    }

    fn down_residual_f32_rank() -> RoutedMoeStepPhases<'static> {
        RoutedMoePhases {
            router: Vec::new(),
            gate_up: vec![gate_up_step()],
            activation: vec![activation_step()],
            down: vec![Step::IndexedMoeGemv {
                experts: expert_ref(),
                which: MoeProj::DownResidual {
                    topk_weights: synth_f32(4),
                },
                topk_indices: synth_i64(4),
                input: GemvInput::Raw(synth_f32(4)),
                out: synth_f32(4),
                k_top: 2,
                batch_size: 1,
            }],
            combine: Vec::new(),
            // Matches the i64 rank's phase lengths; a plain residual add, not
            // a conversion, so only the down evidence differs between ranks.
            finish: vec![Step::ResidualAdd {
                x: synth_f32(4),
                y: synth_f32(4),
                dim: 4,
            }],
        }
    }

    #[test]
    fn lowering_derives_f32_combine_collective_from_concrete_steps() {
        let group = group(ExpertParallelism::TensorParallel, 2);
        let policy = tp_policy(2);
        let rank = || {
            let out = synth_f32(4);
            RoutedMoePhases {
                router: Vec::new(),
                gate_up: vec![gate_up_step()],
                activation: vec![activation_step()],
                down: vec![Step::IndexedMoeGemv {
                    experts: expert_ref(),
                    which: MoeProj::DownExpanded,
                    topk_indices: synth_i64(4),
                    input: GemvInput::Raw(synth_f32(4)),
                    out,
                    k_top: 2,
                    batch_size: 1,
                }],
                // The combine must consume the expanded down producer.
                combine: vec![Step::MoeCombine {
                    down_out: out,
                    topk_weights: synth_f32(4),
                    out: synth_f32(4),
                    k: 2,
                    hidden: 4,
                    batch_size: 1,
                    inverse_perm: None,
                }],
                finish: Vec::new(),
            }
        };
        let lowered = lower_moe_steps(&group, &policy, parts(vec![rank(), rank()])).unwrap();
        let LoweredMoeProgramInner::Parallel {
            collectives,
            zero_before,
            ..
        } = lowered.inner
        else {
            panic!("expected parallel program");
        };
        // combine at absolute index 3; the TP collective lands exactly there.
        assert!(matches!(
            collectives[3],
            StepCollective::AllReduce {
                kind: DimKind::Tp,
                dim: 4
            }
        ));
        assert!(zero_before[3]);
        assert_eq!(collective_count(&collectives), 1);
    }

    #[test]
    fn lowering_derives_f32_self_combining_collective_from_concrete_steps() {
        let group = group(ExpertParallelism::ExpertParallel, 2);
        let policy = ep_policy(2);
        let lowered = lower_moe_steps(
            &group,
            &policy,
            parts(vec![down_residual_f32_rank(), down_residual_f32_rank()]),
        )
        .unwrap();
        let LoweredMoeProgramInner::Parallel {
            collectives,
            zero_before,
            ..
        } = lowered.inner
        else {
            panic!("expected parallel program");
        };
        // the residual-fused down at absolute index 2 IS the EP partial.
        assert!(matches!(
            collectives[2],
            StepCollective::AllReduce {
                kind: DimKind::Ep,
                dim: 4
            }
        ));
        assert!(zero_before[2]);
        assert_eq!(collective_count(&collectives), 1);
    }

    #[test]
    fn lowering_rejects_missing_phases() {
        let group = group(ExpertParallelism::Single, 1);
        let policy = MoEExecutionPolicy::single();
        let mut no_gate_up = i64_rank();
        no_gate_up.gate_up.clear();
        let err = lower_moe_steps(&group, &policy, parts(vec![no_gate_up])).unwrap_err();
        assert!(matches!(
            err,
            MoeLowerError::MissingPhase {
                phase: "gate_up",
                ..
            }
        ));

        let mut no_activation = i64_rank();
        no_activation.activation.clear();
        let err = lower_moe_steps(&group, &policy, parts(vec![no_activation])).unwrap_err();
        assert!(matches!(
            err,
            MoeLowerError::MissingPhase {
                phase: "activation",
                ..
            }
        ));

        let mut no_down = i64_rank();
        no_down.down.clear();
        let err = lower_moe_steps(&group, &policy, parts(vec![no_down])).unwrap_err();
        assert!(matches!(
            err,
            MoeLowerError::MissingPhase { phase: "down", .. }
        ));
    }

    #[test]
    fn lowering_rejects_i64_on_single_rank_axis() {
        let group = group(ExpertParallelism::Single, 1);
        let policy = MoEExecutionPolicy::single();
        let err = lower_moe_steps(&group, &policy, parts(vec![i64_rank()])).unwrap_err();
        assert!(matches!(err, MoeLowerError::I64OnNonAdmittedAxis { .. }));
        let display = err.to_string();
        assert!(display.contains("test"));
        assert!(display.contains("Some(0)"));
    }

    #[test]
    fn lowering_rejects_mixed_rank_protocols() {
        let group = group(ExpertParallelism::TensorParallel, 2);
        let policy = tp_policy(2);
        let err = lower_moe_steps(
            &group,
            &policy,
            parts(vec![i64_rank(), down_residual_f32_rank()]),
        )
        .unwrap_err();
        assert!(matches!(err, MoeLowerError::RankProtocolMismatch { .. }));
        let display = err.to_string();
        assert!(display.contains("test"));
        assert!(display.contains("Some(0)"));
        assert!(display.contains("rank 1"));
    }

    #[test]
    fn executor_selection_is_explicit() {
        assert_eq!(
            select_moe_executor(
                &group(ExpertParallelism::Single, 1),
                &MoEExecutionPolicy::single()
            )
            .unwrap(),
            MoeExecutorKind::SingleMesh
        );
        assert_eq!(
            select_moe_executor(&group(ExpertParallelism::TensorParallel, 2), &tp_policy(2))
                .unwrap(),
            MoeExecutorKind::Parallel
        );
        assert_eq!(
            select_moe_executor(&group(ExpertParallelism::ExpertParallel, 2), &ep_policy(2))
                .unwrap(),
            MoeExecutorKind::Parallel
        );
    }

    #[test]
    fn lowering_rejects_router_identity_mismatch() {
        // Both identities mismatch (declared sigmoid/grouped vs actual
        // softmax/indexed); the router check runs first, so the reported
        // error is always the RouterIdentityMismatch.
        let group =
            group_with_identities("sigmoid_topk", &[ExpertExecutionIdentity::GroupedQuantized]);
        let err = validate_program_identity(
            &group,
            RouterSelection::SoftmaxTopK,
            ExpertExecutionPlan::IndexedQuantized,
        )
        .unwrap_err();
        assert!(matches!(err, MoeLowerError::RouterIdentityMismatch { .. }));
        let display = err.to_string();
        assert!(display.contains("test"));
        assert!(display.contains("Some(0)"));
        assert!(display.contains("sigmoid_topk"));
        assert!(display.contains("softmax_topk"));
        assert!(!display.contains("execution"));
    }

    #[test]
    fn canonical_identity_labels_cover_every_typed_variant() {
        for (selection, expected) in [
            (RouterSelection::SoftmaxTopK, "softmax_topk"),
            (RouterSelection::SigmoidTopK, "sigmoid_topk"),
            (RouterSelection::BiasAwareTopK, "bias_aware_topk"),
            (RouterSelection::Hash, "hash"),
            (RouterSelection::Precomputed, "precomputed"),
        ] {
            assert_eq!(canonical_router_identity(selection), expected);
        }
        for (plan, expected) in [
            (ExpertExecutionPlan::IndexedQuantized, "indexed_quantized"),
            (ExpertExecutionPlan::GroupedQuantized, "grouped_quantized"),
            (
                ExpertExecutionPlan::PerExpertFallback,
                "per_expert_fallback",
            ),
        ] {
            assert_eq!(
                ExpertExecutionIdentity::from(plan).canonical_label(),
                expected
            );
        }
    }

    #[test]
    fn typed_execution_plans_map_exhaustively_to_manifest_identity() {
        for (plan, expected) in [
            (
                ExpertExecutionPlan::IndexedQuantized,
                ExpertExecutionIdentity::IndexedQuantized,
            ),
            (
                ExpertExecutionPlan::GroupedQuantized,
                ExpertExecutionIdentity::GroupedQuantized,
            ),
            (
                ExpertExecutionPlan::PerExpertFallback,
                ExpertExecutionIdentity::PerExpertFallback,
            ),
        ] {
            let identity = ExpertExecutionIdentity::from(plan);
            assert_eq!(identity, expected);
            // The mapped manifest identity carries the same canonical label
            // as the typed dispatch plan.
            assert_eq!(
                identity.canonical_label(),
                ExpertExecutionIdentity::from(plan).canonical_label()
            );
        }
    }

    #[test]
    fn lowering_rejects_execution_identity_mismatch() {
        let group =
            group_with_identities("sigmoid_topk", &[ExpertExecutionIdentity::GroupedQuantized]);
        let err = validate_program_identity(
            &group,
            RouterSelection::SigmoidTopK,
            ExpertExecutionPlan::IndexedQuantized,
        )
        .unwrap_err();
        assert!(matches!(
            err,
            MoeLowerError::ExecutionIdentityMismatch { .. }
        ));
        let display = err.to_string();
        assert!(display.contains("test"));
        assert!(display.contains("Some(0)"));
        assert!(display.contains("grouped_quantized"));
        assert!(display.contains("indexed_quantized"));
    }

    #[test]
    fn validate_program_identity_accepts_exact_execution_membership() {
        let group =
            group_with_identities("softmax_topk", &[ExpertExecutionIdentity::IndexedQuantized]);
        assert!(validate_program_identity(
            &group,
            RouterSelection::SoftmaxTopK,
            ExpertExecutionPlan::IndexedQuantized,
        )
        .is_ok());
    }

    #[test]
    fn validate_program_identity_rejects_execution_absent_from_membership() {
        let group =
            group_with_identities("softmax_topk", &[ExpertExecutionIdentity::IndexedQuantized]);
        let err = validate_program_identity(
            &group,
            RouterSelection::SoftmaxTopK,
            ExpertExecutionPlan::GroupedQuantized,
        )
        .unwrap_err();
        assert!(matches!(
            err,
            MoeLowerError::ExecutionIdentityMismatch { .. }
        ));
        let display = err.to_string();
        assert!(display.contains("indexed_quantized"));
        assert!(display.contains("grouped_quantized"));
    }

    #[test]
    fn validate_program_identity_accepts_indexed_and_grouped_when_both_declared() {
        // Qwen-style declaration: indexed+grouped both admitted.
        let group = group_with_identities(
            "softmax_topk",
            &[
                ExpertExecutionIdentity::IndexedQuantized,
                ExpertExecutionIdentity::GroupedQuantized,
            ],
        );
        assert!(validate_program_identity(
            &group,
            RouterSelection::SoftmaxTopK,
            ExpertExecutionPlan::IndexedQuantized,
        )
        .is_ok());
        assert!(validate_program_identity(
            &group,
            RouterSelection::SoftmaxTopK,
            ExpertExecutionPlan::GroupedQuantized,
        )
        .is_ok());
    }

    #[test]
    fn lowering_rejects_empty_allowed_executions_with_context() {
        // Direct-plan admission validation: an empty allowed-execution set
        // must be rejected with group/layer context BEFORE any membership
        // check runs (today it falls through to a misleading membership
        // mismatch instead).
        let mut group = group(ExpertParallelism::Single, 1);
        group.allowed_executions.clear();
        let err = lower_moe_steps(
            &group,
            &MoEExecutionPolicy::single(),
            parts(vec![down_residual_f32_rank()]),
        )
        .unwrap_err();
        let display = err.to_string();
        match err {
            MoeLowerError::EmptyAllowedExecutions { group, layer } => {
                assert_eq!(group, "test");
                assert_eq!(layer, Some(0));
            }
            other => panic!("expected EmptyAllowedExecutions, got {other:?}"),
        }
        assert!(display.contains("test"), "got: {display}");
        assert!(display.contains("Some(0)"), "got: {display}");
        assert!(
            display.contains("allowed execution identities is empty"),
            "empty admission set must be diagnosed explicitly, got: {display}"
        );
    }

    #[test]
    fn lowering_rejects_duplicate_allowed_executions_with_canonical_label() {
        // Direct-plan admission validation: a duplicated typed identity must
        // be rejected with its canonical label and group/layer context (today
        // the duplicate is silently accepted and lowering proceeds).
        let mut group = group(ExpertParallelism::Single, 1);
        group.allowed_executions = vec![
            ExpertExecutionIdentity::IndexedQuantized,
            ExpertExecutionIdentity::IndexedQuantized,
        ];
        let err = lower_moe_steps(
            &group,
            &MoEExecutionPolicy::single(),
            parts(vec![down_residual_f32_rank()]),
        )
        .unwrap_err();
        let display = err.to_string();
        match err {
            MoeLowerError::DuplicateAllowedExecutions {
                group,
                layer,
                identity,
            } => {
                assert_eq!(group, "test");
                assert_eq!(layer, Some(0));
                assert_eq!(identity, "indexed_quantized");
            }
            other => panic!("expected DuplicateAllowedExecutions, got {other:?}"),
        }
        assert!(display.contains("test"), "got: {display}");
        assert!(display.contains("Some(0)"), "got: {display}");
        assert!(display.contains("duplicate"), "got: {display}");
        assert!(
            display.contains("indexed_quantized"),
            "duplicate diagnostic must name the canonical label, got: {display}"
        );
    }

    #[test]
    fn hash_host_fallback_requires_explicit_precomputed_identity() {
        let group =
            group_with_identities("precomputed", &[ExpertExecutionIdentity::IndexedQuantized]);
        assert!(validate_program_identity(
            &group,
            RouterSelection::Precomputed,
            ExpertExecutionPlan::IndexedQuantized,
        )
        .is_ok());
        // Hash is not an alias for precomputed: the same group with an actual
        // Hash router still mismatches the explicitly declared identity.
        let err = validate_program_identity(
            &group,
            RouterSelection::Hash,
            ExpertExecutionPlan::IndexedQuantized,
        )
        .unwrap_err();
        assert!(matches!(err, MoeLowerError::RouterIdentityMismatch { .. }));
    }

    // ── Task 3: real-Step protocol derivation ──────────────────────────

    use hip_bridge::DeviceBuffer;
    use hipfire_dispatch::families::moe::MoeExpertRef;
    use hipfire_dispatch::pipeline::{GemvInput, MoeActivationVariant, MoeProj};
    use rdna_compute::{DType, GpuTensor};

    // Pointer-identity capacity registry for synthetic test tensors: each
    // leaked allocation registers its byte capacity (8 bytes per element for
    // i64 roles, 4 bytes per element for f32 roles).
    thread_local! {
        static SYNTH_CAPS: std::cell::RefCell<std::collections::HashMap<usize, usize>> =
            std::cell::RefCell::new(std::collections::HashMap::new());
    }

    /// Private test capacity closure: pointer-identity lookup returns the
    /// synthetic capacity registered for each test tensor, never a real
    /// device allocation and never a shared null.
    fn synthetic_capacity(tensor: &GpuTensor) -> usize {
        SYNTH_CAPS.with(|caps| {
            *caps
                .borrow()
                .get(&(tensor.buf.as_ptr() as usize))
                .expect("synthetic capacity registered for tensor")
        })
    }

    fn synth_with_bytes(dtype: DType, numel: usize, bytes: usize) -> &'static GpuTensor {
        let buffer = Box::leak(vec![0u8; bytes].into_boxed_slice());
        let tensor = Box::leak(Box::new(GpuTensor {
            buf: unsafe { DeviceBuffer::from_raw(buffer.as_mut_ptr().cast(), bytes) },
            shape: vec![numel],
            dtype,
        }));
        SYNTH_CAPS.with(|caps| {
            caps.borrow_mut()
                .insert(tensor.buf.as_ptr() as usize, bytes);
        });
        tensor
    }

    fn synth_i64(numel: usize) -> &'static GpuTensor {
        synth_with_bytes(DType::Raw, numel, numel * 8)
    }

    fn synth_f32(numel: usize) -> &'static GpuTensor {
        synth_with_bytes(DType::F32, numel, numel * 4)
    }

    fn expert_ref() -> &'static MoeExpertRef<'static> {
        expert_ref_with(4)
    }

    fn expert_ref_with(n_experts: usize) -> &'static MoeExpertRef<'static> {
        // Pointer tables are u64 entries (8 bytes per expert) per the kernel
        // ABI; the grouped capacity gates check them.
        let ptr_bytes = n_experts * DEVICE_POINTER_BYTES;
        Box::leak(Box::new(MoeExpertRef {
            gate_up_ptrs: synth_with_bytes(DType::Raw, n_experts, ptr_bytes),
            down_ptrs: synth_with_bytes(DType::Raw, n_experts, ptr_bytes),
            dummy_gate_up: None,
            dtype: DType::F32,
            n_experts,
            expert_m: 4,
            expert_k: 4,
            owned: &[],
        }))
    }

    fn gate_up_step() -> Step<'static> {
        Step::IndexedMoeGemv {
            experts: expert_ref(),
            which: MoeProj::GateUp {
                up_out: synth_f32(4),
            },
            topk_indices: synth_i64(4),
            input: GemvInput::Raw(synth_f32(4)),
            out: synth_f32(4),
            k_top: 2,
            batch_size: 1,
        }
    }

    fn activation_step() -> Step<'static> {
        Step::MoeActivation {
            variant: MoeActivationVariant::QwenAwqIndexed {
                awq_ptrs: synth_f32(4),
                topk_indices: synth_i64(4),
            },
            gate: synth_f32(4),
            up: synth_f32(4),
            rot_out: synth_f32(4),
            inter: 4,
            k_top: 2,
        }
    }

    fn down_i64_step() -> (Step<'static>, &'static GpuTensor) {
        let out = synth_i64(4);
        let step = Step::IndexedMoeGemv {
            experts: expert_ref(),
            which: MoeProj::DownResidualI64 {
                topk_weights: synth_f32(4),
            },
            topk_indices: synth_i64(4),
            input: GemvInput::Raw(synth_f32(4)),
            out,
            k_top: 2,
            batch_size: 1,
        };
        (step, out)
    }

    fn convert_step(src: &'static GpuTensor, n: usize) -> Step<'static> {
        Step::ConvertI64ToF32 {
            src,
            dst: synth_f32(n),
            n,
        }
    }

    fn i64_rank() -> RoutedMoeStepPhases<'static> {
        let (down, out) = down_i64_step();
        RoutedMoePhases {
            router: Vec::new(),
            gate_up: vec![gate_up_step()],
            activation: vec![activation_step()],
            down: vec![down],
            combine: Vec::new(),
            finish: vec![convert_step(out, 4)],
        }
    }

    fn parts(ranks: Vec<RoutedMoeStepPhases<'static>>) -> MoeProgramParts<'static> {
        MoeProgramParts {
            router: RouterPlan::SoftmaxTopK {
                scores: synth_f32(4),
                topk_indices: synth_i64(4),
                topk_weights: synth_f32(4),
                k_top: 2,
                normalize: true,
                route_scale: 1.0,
            },
            execution: ExpertExecutionPlan::IndexedQuantized,
            ranks,
            deferred_combine: false,
        }
    }

    fn deferred_parts(
        ranks: Vec<RoutedMoeStepPhases<'static>>,
        deferred_combine: bool,
    ) -> MoeProgramParts<'static> {
        let mut parts = parts(ranks);
        parts.deferred_combine = deferred_combine;
        parts
    }

    #[test]
    fn lowering_derives_tp_i64_collective_from_concrete_steps() {
        let group = group(ExpertParallelism::TensorParallel, 2);
        let policy = tp_policy(2);
        let lowered =
            lower_moe_steps(&group, &policy, parts(vec![i64_rank(), i64_rank()])).unwrap();
        let LoweredMoeProgramInner::Parallel {
            collectives,
            zero_before,
            ..
        } = lowered.inner
        else {
            panic!("expected parallel program");
        };
        // down at absolute index 2 (router 0 + gate_up 1 + activation 1);
        // conversion at absolute index 3.
        assert!(matches!(
            collectives[2],
            StepCollective::AllReduceI64Tp { dim: 4 }
        ));
        assert!(zero_before[2]);
        assert!(!zero_before[3]);
        assert_eq!(collective_count(&collectives), 1);

        // The capacity-aware parser honors the pointer-identity synthetic
        // capacity closure identically to the production buf.size() path.
        let (protocol, _) = parse_step_protocol(
            &group,
            &policy,
            &[i64_rank(), i64_rank()],
            synthetic_capacity,
            false,
        )
        .unwrap();
        assert!(matches!(protocol, ValidatedProtocol::TpI64 { .. }));
    }

    #[test]
    fn lowering_derives_ep_local_i64_zero_and_f32_collective() {
        let group = group(ExpertParallelism::ExpertParallel, 2);
        let policy = ep_policy(2);
        let lowered =
            lower_moe_steps(&group, &policy, parts(vec![i64_rank(), i64_rank()])).unwrap();
        let LoweredMoeProgramInner::Parallel {
            collectives,
            zero_before,
            ..
        } = lowered.inner
        else {
            panic!("expected parallel program");
        };
        // Local i64 zeroing at the down step, FP32 EP all-reduce at convert.
        assert!(matches!(
            collectives[2],
            StepCollective::ZeroI64Only { dim: 4 }
        ));
        assert!(zero_before[2]);
        assert!(matches!(
            collectives[3],
            StepCollective::AllReduce {
                kind: DimKind::Ep,
                dim: 4
            }
        ));
        assert!(!zero_before[3]);
        assert_eq!(collective_count(&collectives), 1);
    }

    #[test]
    fn lowering_rejects_tp_i64_without_matching_conversion() {
        let group = group(ExpertParallelism::TensorParallel, 2);
        let policy = tp_policy(2);
        let rank = || {
            let (down, _) = down_i64_step();
            RoutedMoePhases {
                router: Vec::new(),
                gate_up: vec![gate_up_step()],
                activation: vec![activation_step()],
                down: vec![down],
                combine: Vec::new(),
                finish: Vec::new(),
            }
        };
        let err = lower_moe_steps(&group, &policy, parts(vec![rank(), rank()])).unwrap_err();
        assert!(matches!(err, MoeLowerError::MissingI64Conversion { .. }));
        let display = err.to_string();
        assert!(display.contains("test"));
        assert!(display.contains("Some(0)"));
    }

    #[test]
    fn lowering_rejects_i64_conversion_source_mismatch() {
        let group = group(ExpertParallelism::TensorParallel, 2);
        let policy = tp_policy(2);
        // Conversion reads a distinct i64 buffer, not the down output.
        let rank = || {
            let (down, _) = down_i64_step();
            RoutedMoePhases {
                router: Vec::new(),
                gate_up: vec![gate_up_step()],
                activation: vec![activation_step()],
                down: vec![down],
                combine: Vec::new(),
                finish: vec![convert_step(synth_i64(4), 4)],
            }
        };
        let err = lower_moe_steps(&group, &policy, parts(vec![rank(), rank()])).unwrap_err();
        assert!(matches!(
            err,
            MoeLowerError::I64ConversionSourceMismatch { .. }
        ));
        let display = err.to_string();
        assert!(display.contains("test"));
        assert!(display.contains("Some(0)"));
    }

    #[test]
    fn lowering_rejects_i64_capacity_or_dimension_mismatch() {
        let group = group(ExpertParallelism::TensorParallel, 2);
        let policy = tp_policy(2);

        // Dimension mismatch: the i64 down buffer holds 4 elements (32 bytes)
        // but the conversion declares n=8.
        let rank = || {
            let (down, out) = down_i64_step();
            RoutedMoePhases {
                router: Vec::new(),
                gate_up: vec![gate_up_step()],
                activation: vec![activation_step()],
                down: vec![down],
                combine: Vec::new(),
                finish: vec![convert_step(out, 8)],
            }
        };
        let err = lower_moe_steps(&group, &policy, parts(vec![rank(), rank()])).unwrap_err();
        assert!(matches!(err, MoeLowerError::I64DimensionMismatch { .. }));

        // Capacity mismatch: the f32 destination holds 2 elements (8 bytes),
        // below the n*4 = 16 bytes the conversion requires.
        let rank = || {
            let (down, out) = down_i64_step();
            RoutedMoePhases {
                router: Vec::new(),
                gate_up: vec![gate_up_step()],
                activation: vec![activation_step()],
                down: vec![down],
                combine: Vec::new(),
                finish: vec![Step::ConvertI64ToF32 {
                    src: out,
                    dst: synth_f32(2),
                    n: 4,
                }],
            }
        };
        let err = lower_moe_steps(&group, &policy, parts(vec![rank(), rank()])).unwrap_err();
        assert!(matches!(err, MoeLowerError::CapacityMismatch { .. }));
    }

    #[test]
    fn lowering_rejects_duplicate_i64_conversion() {
        let group = group(ExpertParallelism::TensorParallel, 2);
        let policy = tp_policy(2);
        let rank = || {
            let (down, out) = down_i64_step();
            RoutedMoePhases {
                router: Vec::new(),
                gate_up: vec![gate_up_step()],
                activation: vec![activation_step()],
                down: vec![down],
                combine: Vec::new(),
                finish: vec![convert_step(out, 4), convert_step(out, 4)],
            }
        };
        let err = lower_moe_steps(&group, &policy, parts(vec![rank(), rank()])).unwrap_err();
        assert!(matches!(err, MoeLowerError::DuplicateI64Conversion { .. }));
        let display = err.to_string();
        assert!(display.contains("test"));
        assert!(display.contains("Some(0)"));
    }

    #[test]
    fn lowering_rejects_combine_after_self_combining_down() {
        let group = group(ExpertParallelism::TensorParallel, 2);
        let policy = tp_policy(2);
        let rank = || RoutedMoePhases {
            router: Vec::new(),
            gate_up: vec![gate_up_step()],
            activation: vec![activation_step()],
            down: vec![Step::IndexedMoeGemv {
                experts: expert_ref(),
                which: MoeProj::DownResidual {
                    topk_weights: synth_f32(4),
                },
                topk_indices: synth_i64(4),
                input: GemvInput::Raw(synth_f32(4)),
                out: synth_f32(4),
                k_top: 2,
                batch_size: 1,
            }],
            combine: vec![Step::MoeCombine {
                down_out: synth_f32(4),
                topk_weights: synth_f32(4),
                out: synth_f32(4),
                k: 2,
                hidden: 4,
                batch_size: 1,
                inverse_perm: None,
            }],
            finish: Vec::new(),
        };
        let err = lower_moe_steps(&group, &policy, parts(vec![rank(), rank()])).unwrap_err();
        assert!(matches!(
            err,
            MoeLowerError::CombineAfterSelfCombiningDown { .. }
        ));
        let display = err.to_string();
        assert!(display.contains("test"));
        assert!(display.contains("Some(0)"));
    }

    #[test]
    fn lowering_accepts_padded_and_non_multiple_i64_capacities() {
        let group = group(ExpertParallelism::TensorParallel, 2);
        let policy = tp_policy(2);
        // n=8 with a padded 72-byte source and a non-multiple 67-byte source:
        // both are >= n*8 = 64 bytes, so both must be admitted.
        for bytes in [72usize, 67usize] {
            let rank = || {
                let out = synth_with_bytes(DType::Raw, 8, bytes);
                let down = Step::IndexedMoeGemv {
                    experts: expert_ref(),
                    which: MoeProj::DownResidualI64 {
                        topk_weights: synth_f32(4),
                    },
                    topk_indices: synth_i64(4),
                    input: GemvInput::Raw(synth_f32(4)),
                    out,
                    k_top: 2,
                    batch_size: 1,
                };
                RoutedMoePhases {
                    router: Vec::new(),
                    gate_up: vec![gate_up_step()],
                    activation: vec![activation_step()],
                    down: vec![down],
                    combine: Vec::new(),
                    finish: vec![convert_step(out, 8)],
                }
            };
            let lowered = lower_moe_steps(&group, &policy, parts(vec![rank(), rank()])).unwrap();
            let LoweredMoeProgramInner::Parallel { collectives, .. } = lowered.inner else {
                panic!("expected parallel program");
            };
            assert!(matches!(
                collectives[2],
                StepCollective::AllReduceI64Tp { dim: 8 }
            ));
            assert_eq!(collective_count(&collectives), 1);
        }
    }

    #[test]
    fn same_buffer_accepts_same_object_null_and_rejects_distinct_nulls() {
        let null_tensor = Box::leak(Box::new(GpuTensor {
            buf: unsafe { DeviceBuffer::from_raw(std::ptr::null_mut(), 0) },
            shape: vec![0],
            dtype: DType::Raw,
        }));
        // The same borrowed null tensor is the same buffer by pointer identity.
        assert!(same_buffer(null_tensor, null_tensor));
        // Two distinct null tensors are never the same buffer.
        let other_null = Box::leak(Box::new(GpuTensor {
            buf: unsafe { DeviceBuffer::from_raw(std::ptr::null_mut(), 0) },
            shape: vec![0],
            dtype: DType::Raw,
        }));
        assert!(!same_buffer(null_tensor, other_null));
        assert!(!same_buffer(other_null, null_tensor));
    }

    #[test]
    fn lowering_rejects_early_conversion_even_with_valid_finish_conversion() {
        let group = group(ExpertParallelism::TensorParallel, 2);
        let policy = tp_policy(2);
        // One conversion in the activation phase plus one valid finish
        // conversion: two conversions anywhere must be rejected as duplicate.
        let rank = || {
            let (down, out) = down_i64_step();
            RoutedMoePhases {
                router: Vec::new(),
                gate_up: vec![gate_up_step()],
                activation: vec![activation_step(), convert_step(out, 4)],
                down: vec![down],
                combine: Vec::new(),
                finish: vec![convert_step(out, 4)],
            }
        };
        let err = lower_moe_steps(&group, &policy, parts(vec![rank(), rank()])).unwrap_err();
        assert!(matches!(err, MoeLowerError::DuplicateI64Conversion { .. }));
    }

    #[test]
    fn lowering_rejects_early_only_conversion_as_misplaced() {
        let group = group(ExpertParallelism::TensorParallel, 2);
        let policy = tp_policy(2);
        // The only conversion sits in the activation phase: a dedicated
        // contextual misplaced-conversion error, not a missing one.
        let rank = || {
            let (down, out) = down_i64_step();
            RoutedMoePhases {
                router: Vec::new(),
                gate_up: vec![gate_up_step()],
                activation: vec![activation_step(), convert_step(out, 4)],
                down: vec![down],
                combine: Vec::new(),
                finish: Vec::new(),
            }
        };
        let err = lower_moe_steps(&group, &policy, parts(vec![rank(), rank()])).unwrap_err();
        assert!(matches!(err, MoeLowerError::MisplacedI64Conversion { .. }));
        let display = err.to_string();
        assert!(display.contains("test"));
        assert!(display.contains("Some(0)"));
        assert!(display.contains("activation"));
    }

    #[test]
    fn lowering_rejects_i64_arithmetic_overflow_capacity() {
        let group = group(ExpertParallelism::TensorParallel, 2);
        let policy = tp_policy(2);
        // n exceeds usize::MAX/8: even a usize::MAX byte capacity cannot
        // satisfy the required n*8 bytes, and the requirement itself must be
        // reported as arithmetic overflow rather than admitted.
        let n = usize::MAX / 8 + 1;
        let rank = || {
            let out = Box::leak(Box::new(GpuTensor {
                buf: unsafe {
                    DeviceBuffer::from_raw(
                        Box::leak(vec![0u8; 8].into_boxed_slice())
                            .as_mut_ptr()
                            .cast(),
                        usize::MAX,
                    )
                },
                shape: vec![n],
                dtype: DType::Raw,
            }));
            let down = Step::IndexedMoeGemv {
                experts: expert_ref(),
                which: MoeProj::DownResidualI64 {
                    topk_weights: synth_f32(4),
                },
                topk_indices: synth_i64(4),
                input: GemvInput::Raw(synth_f32(4)),
                out,
                k_top: 2,
                batch_size: 1,
            };
            RoutedMoePhases {
                router: Vec::new(),
                gate_up: vec![gate_up_step()],
                activation: vec![activation_step()],
                down: vec![down],
                combine: Vec::new(),
                finish: vec![Step::ConvertI64ToF32 {
                    src: out,
                    dst: out,
                    n,
                }],
            }
        };
        let err = lower_moe_steps(&group, &policy, parts(vec![rank(), rank()])).unwrap_err();
        assert!(matches!(err, MoeLowerError::ArithmeticOverflow { .. }));
        let display = err.to_string();
        assert!(display.contains("test"));
        assert!(display.contains("Some(0)"));
    }

    #[test]
    fn lowering_rejects_overflowing_i64_shape_metadata_without_panic() {
        let group = group(ExpertParallelism::TensorParallel, 2);
        let policy = tp_policy(2);
        // The i64 down output's multidimensional shape overflows usize when
        // multiplied: validation must return Err, never panic or wrap.
        let rank = || {
            let out = Box::leak(Box::new(GpuTensor {
                buf: unsafe {
                    DeviceBuffer::from_raw(
                        Box::leak(vec![0u8; 64].into_boxed_slice())
                            .as_mut_ptr()
                            .cast(),
                        64,
                    )
                },
                shape: vec![usize::MAX, 2],
                dtype: DType::Raw,
            }));
            let down = Step::IndexedMoeGemv {
                experts: expert_ref(),
                which: MoeProj::DownResidualI64 {
                    topk_weights: synth_f32(4),
                },
                topk_indices: synth_i64(4),
                input: GemvInput::Raw(synth_f32(4)),
                out,
                k_top: 2,
                batch_size: 1,
            };
            RoutedMoePhases {
                router: Vec::new(),
                gate_up: vec![gate_up_step()],
                activation: vec![activation_step()],
                down: vec![down],
                combine: Vec::new(),
                finish: vec![Step::ConvertI64ToF32 {
                    src: out,
                    dst: out,
                    n: 8,
                }],
            }
        };
        let err = lower_moe_steps(&group, &policy, parts(vec![rank(), rank()])).unwrap_err();
        assert!(matches!(err, MoeLowerError::ArithmeticOverflow { .. }));
        assert!(err.to_string().contains("logical_shape"));
    }

    // ── DeepSeek batched indexed I64 chain (Phase 3 shared lane) ────────────
    // The runtime authority for a batched indexed I64 program: exactly one
    // indexed GateUp + one activation + one indexed DownResidualI64, with
    // matching batch/top-k/expert/index metadata, connected dataflow, checked
    // geometry, exact [batch, hidden] output, and real-buffer capacities.

    const BATCHED_B: usize = 2;
    const BATCHED_K: usize = 2;
    /// expert_m — the 256-aligned inter_local contraction dim of the batched
    /// MQ2 i64-down kernel.
    const BATCHED_INTER: usize = 256;
    /// expert_k — the 256-aligned hidden contraction dim of the batched MQ2
    /// gate-up kernel.
    const BATCHED_HIDDEN: usize = 256;

    fn synth_shape(dtype: DType, shape: Vec<usize>, bytes: usize) -> &'static GpuTensor {
        let buffer = Box::leak(vec![0u8; bytes].into_boxed_slice());
        let tensor = Box::leak(Box::new(GpuTensor {
            buf: unsafe { DeviceBuffer::from_raw(buffer.as_mut_ptr().cast(), bytes) },
            shape,
            dtype,
        }));
        SYNTH_CAPS.with(|caps| {
            caps.borrow_mut()
                .insert(tensor.buf.as_ptr() as usize, bytes);
        });
        tensor
    }

    fn synth_i32(numel: usize) -> &'static GpuTensor {
        synth_with_bytes(DType::Raw, numel, numel * 4)
    }

    /// Synthetic tensor with a TINY real allocation but a LARGE registered
    /// capacity (pointer-identity registry). For ABI-boundary fixtures whose
    /// coherent real buffers would be gigabytes; never dereferenced.
    fn synth_fake_capacity(dtype: DType, shape: Vec<usize>, capacity: usize) -> &'static GpuTensor {
        let buffer = Box::leak(vec![0u8; 8].into_boxed_slice());
        let tensor = Box::leak(Box::new(GpuTensor {
            buf: unsafe { DeviceBuffer::from_raw(buffer.as_mut_ptr().cast(), capacity) },
            shape,
            dtype,
        }));
        SYNTH_CAPS.with(|caps| {
            caps.borrow_mut()
                .insert(tensor.buf.as_ptr() as usize, capacity);
        });
        tensor
    }

    fn lloyd_expert_ref_with_tables(
        n_experts: usize,
        expert_m: usize,
        expert_k: usize,
        dtype: DType,
        gate_up_ptrs: &'static GpuTensor,
        down_ptrs: &'static GpuTensor,
    ) -> &'static MoeExpertRef<'static> {
        Box::leak(Box::new(MoeExpertRef {
            gate_up_ptrs,
            down_ptrs,
            dummy_gate_up: None,
            dtype,
            n_experts,
            expert_m,
            expert_k,
            owned: &[],
        }))
    }

    fn lloyd_expert_ref_with(
        n_experts: usize,
        expert_m: usize,
        expert_k: usize,
        dtype: DType,
    ) -> &'static MoeExpertRef<'static> {
        // The kernel ABI reads the expert pointer tables as `unsigned long
        // long` entries: 8 bytes per expert.
        let ptr_bytes = n_experts * DEVICE_POINTER_BYTES;
        lloyd_expert_ref_with_tables(
            n_experts,
            expert_m,
            expert_k,
            dtype,
            synth_with_bytes(DType::Raw, n_experts, ptr_bytes),
            synth_with_bytes(DType::Raw, n_experts, ptr_bytes),
        )
    }

    fn lloyd_expert_ref() -> &'static MoeExpertRef<'static> {
        lloyd_expert_ref_with(4, BATCHED_INTER, BATCHED_HIDDEN, DType::MQ2G256Lloyd)
    }

    /// A coherent batched indexed I64 chain: one batched GateUp (batch B,
    /// k_top K), one activation over B·K routed rows, one batched
    /// DownResidualI64 writing [B × hidden], and one conversion of
    /// n = B·hidden. Buffers alias exactly as the production DeepSeek chain
    /// will: gate-up outputs feed the activation, the activation's rot_out
    /// feeds the down, and both projections share the top-k index buffer.
    struct BatchedChain {
        experts: &'static MoeExpertRef<'static>,
        topk_indices: &'static GpuTensor,
        topk_weights: &'static GpuTensor,
        x_rot: &'static GpuTensor,
        gate_batch: &'static GpuTensor,
        up_batch: &'static GpuTensor,
        rot_batch: &'static GpuTensor,
        out: &'static GpuTensor,
        dst: &'static GpuTensor,
    }

    fn batched_chain() -> BatchedChain {
        BatchedChain {
            experts: lloyd_expert_ref(),
            topk_indices: synth_i32(BATCHED_B * BATCHED_K),
            topk_weights: synth_f32(BATCHED_B * BATCHED_K),
            x_rot: synth_f32(BATCHED_B * BATCHED_HIDDEN),
            gate_batch: synth_f32(BATCHED_B * BATCHED_K * BATCHED_INTER),
            up_batch: synth_f32(BATCHED_B * BATCHED_K * BATCHED_INTER),
            rot_batch: synth_f32(BATCHED_B * BATCHED_K * BATCHED_INTER),
            out: synth_shape(
                DType::Raw,
                vec![BATCHED_B, BATCHED_HIDDEN],
                BATCHED_B * BATCHED_HIDDEN * 8,
            ),
            dst: synth_f32(BATCHED_B * BATCHED_HIDDEN),
        }
    }

    impl BatchedChain {
        fn gate_up(&self, experts: &'static MoeExpertRef<'static>) -> Step<'static> {
            Step::IndexedMoeGemv {
                experts,
                which: MoeProj::GateUp {
                    up_out: self.up_batch,
                },
                topk_indices: self.topk_indices,
                input: GemvInput::Raw(self.x_rot),
                out: self.gate_batch,
                k_top: BATCHED_K,
                batch_size: BATCHED_B,
            }
        }

        fn activation(&self) -> Step<'static> {
            Step::MoeActivation {
                variant: MoeActivationVariant::Ds4ClampRotate { swiglu_limit: 30.0 },
                gate: self.gate_batch,
                up: self.up_batch,
                rot_out: self.rot_batch,
                inter: BATCHED_INTER,
                k_top: BATCHED_B * BATCHED_K,
            }
        }

        fn down(&self, experts: &'static MoeExpertRef<'static>) -> Step<'static> {
            Step::IndexedMoeGemv {
                experts,
                which: MoeProj::DownResidualI64 {
                    topk_weights: self.topk_weights,
                },
                topk_indices: self.topk_indices,
                input: GemvInput::Raw(self.rot_batch),
                out: self.out,
                k_top: BATCHED_K,
                batch_size: BATCHED_B,
            }
        }

        fn convert(&self) -> Step<'static> {
            Step::ConvertI64ToF32 {
                src: self.out,
                dst: self.dst,
                n: BATCHED_B * BATCHED_HIDDEN,
            }
        }

        fn rank(&self) -> RoutedMoeStepPhases<'static> {
            RoutedMoePhases {
                router: Vec::new(),
                gate_up: vec![self.gate_up(self.experts)],
                activation: vec![self.activation()],
                down: vec![self.down(self.experts)],
                combine: Vec::new(),
                finish: vec![self.convert()],
            }
        }
    }

    fn batched_i64_rank() -> RoutedMoeStepPhases<'static> {
        batched_chain().rank()
    }

    /// Lower a one-rank-mutated two-rank batched program; returns the error.
    fn lower_mutated_batched(phases: RoutedMoeStepPhases<'static>) -> MoeLowerError {
        let group = group(ExpertParallelism::TensorParallel, 2);
        let policy = tp_policy(2);
        lower_moe_steps(&group, &policy, parts(vec![phases, batched_i64_rank()])).unwrap_err()
    }

    /// Lower a two-rank batched program with the SAME mutation applied to both
    /// ranks; returns the error. Required wherever the mutation changes phase
    /// LENGTHS — the parser rejects length-mismatched ranks before any
    /// protocol parsing.
    fn lower_mutated_batched_pair(
        mutate: impl Fn(&mut RoutedMoeStepPhases<'static>),
    ) -> MoeLowerError {
        let group = group(ExpertParallelism::TensorParallel, 2);
        let policy = tp_policy(2);
        let mut first = batched_i64_rank();
        let mut second = batched_i64_rank();
        mutate(&mut first);
        mutate(&mut second);
        lower_moe_steps(&group, &policy, parts(vec![first, second])).unwrap_err()
    }

    #[test]
    fn lowering_derives_batched_tp_i64_collective_from_concrete_steps() {
        let group = group(ExpertParallelism::TensorParallel, 2);
        let policy = tp_policy(2);
        let lowered = lower_moe_steps(
            &group,
            &policy,
            parts(vec![batched_i64_rank(), batched_i64_rank()]),
        )
        .unwrap();
        let LoweredMoeProgramInner::Parallel {
            collectives,
            zero_before,
            per_rank_steps,
            ..
        } = lowered.inner
        else {
            panic!("expected parallel program");
        };
        // down at absolute index 2 (router 0 + gate_up 1 + activation 1);
        // conversion at absolute index 3. Exactly one zero, one
        // AllReduceI64Tp over batch*hidden, and one conversion.
        assert!(matches!(
            collectives[2],
            StepCollective::AllReduceI64Tp { dim: 512 }
        ));
        assert!(zero_before[2]);
        assert!(!zero_before[3]);
        assert_eq!(collective_count(&collectives), 1);
        for rank_steps in per_rank_steps {
            assert!(
                matches!(rank_steps[3], Step::ConvertI64ToF32 { .. }),
                "the finish phase must contain exactly the one conversion"
            );
            assert_eq!(rank_steps.len(), 4);
        }

        // The capacity-aware parser honors the pointer-identity synthetic
        // capacity closure identically to the production buf.size() path.
        let (protocol, _) = parse_step_protocol(
            &group,
            &policy,
            &[batched_i64_rank(), batched_i64_rank()],
            synthetic_capacity,
            false,
        )
        .unwrap();
        assert!(matches!(
            protocol,
            ValidatedProtocol::TpI64 { dim: 512, .. }
        ));
    }

    #[test]
    fn lowering_rejects_batched_i64_zero_batch() {
        let chain = batched_chain();
        let mut phases = chain.rank();
        phases.down[0] = Step::IndexedMoeGemv {
            experts: chain.experts,
            which: MoeProj::DownResidualI64 {
                topk_weights: chain.topk_weights,
            },
            topk_indices: chain.topk_indices,
            input: GemvInput::Raw(chain.rot_batch),
            out: chain.out,
            k_top: BATCHED_K,
            batch_size: 0,
        };
        let err = lower_mutated_batched(phases);
        assert!(matches!(
            err,
            MoeLowerError::BatchedI64ZeroGeometry {
                field: "batch_size",
                ..
            }
        ));
    }

    #[test]
    fn lowering_rejects_batched_i64_zero_k_top() {
        let chain = batched_chain();
        let mut phases = chain.rank();
        phases.down[0] = Step::IndexedMoeGemv {
            experts: chain.experts,
            which: MoeProj::DownResidualI64 {
                topk_weights: chain.topk_weights,
            },
            topk_indices: chain.topk_indices,
            input: GemvInput::Raw(chain.rot_batch),
            out: chain.out,
            k_top: 0,
            batch_size: BATCHED_B,
        };
        let err = lower_mutated_batched(phases);
        assert!(matches!(
            err,
            MoeLowerError::BatchedI64ZeroGeometry { field: "k_top", .. }
        ));
    }

    #[test]
    fn lowering_rejects_batched_i64_missing_gate_up() {
        // The gate_up phase holds an unclassified filler, so the phase is
        // non-empty while no indexed gate-up exists anywhere in the program.
        let err = lower_mutated_batched_pair(|phases| {
            phases.gate_up = vec![Step::ScoreActivation {
                scores: synth_i32(4),
                kind: hipfire_dispatch::pipeline::ScoreActKind::SqrtSoftplus,
            }];
        });
        assert!(matches!(
            err,
            MoeLowerError::BatchedI64GateUpCount {
                expected: 1,
                actual: 0,
                ..
            }
        ));
    }

    #[test]
    fn lowering_rejects_batched_i64_duplicate_gate_up() {
        let err = lower_mutated_batched_pair(|phases| {
            let chain = batched_chain();
            phases.gate_up = vec![chain.gate_up(chain.experts), chain.gate_up(chain.experts)];
        });
        assert!(matches!(
            err,
            MoeLowerError::BatchedI64GateUpCount {
                expected: 1,
                actual: 2,
                ..
            }
        ));
    }

    #[test]
    fn lowering_rejects_batched_i64_missing_activation() {
        // The activation phase holds an unclassified filler, so the phase is
        // non-empty while no activation exists anywhere in the program.
        let err = lower_mutated_batched_pair(|phases| {
            phases.activation = vec![Step::ScoreActivation {
                scores: synth_i32(4),
                kind: hipfire_dispatch::pipeline::ScoreActKind::SqrtSoftplus,
            }];
        });
        assert!(matches!(
            err,
            MoeLowerError::BatchedI64ActivationCount {
                expected: 1,
                actual: 0,
                ..
            }
        ));
    }

    #[test]
    fn lowering_rejects_batched_i64_misplaced_gate_up() {
        // The only indexed gate-up lives in the activation phase while the
        // gate_up phase holds an unclassified filler step.
        let err = lower_mutated_batched_pair(|phases| {
            let chain = batched_chain();
            phases.gate_up = vec![Step::ScoreActivation {
                scores: chain.topk_indices,
                kind: hipfire_dispatch::pipeline::ScoreActKind::SqrtSoftplus,
            }];
            phases.activation = vec![chain.gate_up(chain.experts), chain.activation()];
        });
        assert!(matches!(
            err,
            MoeLowerError::BatchedI64ChainMismatch {
                field: "gate_up_phase",
                ..
            }
        ));
    }

    fn batched_metadata_error(
        gate_up_batch: usize,
        gate_up_k_top: usize,
        gate_up_experts: &'static MoeExpertRef<'static>,
    ) -> MoeLowerError {
        let chain = batched_chain();
        let mut phases = chain.rank();
        phases.gate_up[0] = Step::IndexedMoeGemv {
            experts: gate_up_experts,
            which: MoeProj::GateUp {
                up_out: chain.up_batch,
            },
            topk_indices: chain.topk_indices,
            input: GemvInput::Raw(chain.x_rot),
            out: chain.gate_batch,
            k_top: gate_up_k_top,
            batch_size: gate_up_batch,
        };
        lower_mutated_batched(phases)
    }

    #[test]
    fn lowering_rejects_batched_i64_metadata_mismatch() {
        let err = batched_metadata_error(1, BATCHED_K, lloyd_expert_ref());
        assert!(matches!(
            err,
            MoeLowerError::BatchedI64ChainMismatch {
                field: "batch_size",
                ..
            }
        ));

        let err = batched_metadata_error(BATCHED_B, 3, lloyd_expert_ref());
        assert!(matches!(
            err,
            MoeLowerError::BatchedI64ChainMismatch { field: "k_top", .. }
        ));

        let err = batched_metadata_error(
            BATCHED_B,
            BATCHED_K,
            lloyd_expert_ref_with(8, BATCHED_INTER, BATCHED_HIDDEN, DType::MQ2G256Lloyd),
        );
        assert!(matches!(
            err,
            MoeLowerError::BatchedI64ChainMismatch {
                field: "n_experts",
                ..
            }
        ));

        let err = batched_metadata_error(
            BATCHED_B,
            BATCHED_K,
            lloyd_expert_ref_with(4, 128, BATCHED_HIDDEN, DType::MQ2G256Lloyd),
        );
        assert!(matches!(
            err,
            MoeLowerError::BatchedI64ChainMismatch {
                field: "expert_m",
                ..
            }
        ));

        let err = batched_metadata_error(
            BATCHED_B,
            BATCHED_K,
            lloyd_expert_ref_with(4, BATCHED_INTER, 128, DType::MQ2G256Lloyd),
        );
        assert!(matches!(
            err,
            MoeLowerError::BatchedI64ChainMismatch {
                field: "expert_k",
                ..
            }
        ));

        let err = batched_metadata_error(
            BATCHED_B,
            BATCHED_K,
            lloyd_expert_ref_with(4, BATCHED_INTER, BATCHED_HIDDEN, DType::F32),
        );
        assert!(matches!(
            err,
            MoeLowerError::BatchedI64ChainMismatch { field: "dtype", .. }
        ));
    }

    #[test]
    fn lowering_rejects_batched_i64_topk_index_buffer_mismatch() {
        let chain = batched_chain();
        let mut phases = chain.rank();
        // The down consumes a different (but adequately sized) index buffer.
        phases.down[0] = Step::IndexedMoeGemv {
            experts: chain.experts,
            which: MoeProj::DownResidualI64 {
                topk_weights: chain.topk_weights,
            },
            topk_indices: synth_i32(BATCHED_B * BATCHED_K),
            input: GemvInput::Raw(chain.rot_batch),
            out: chain.out,
            k_top: BATCHED_K,
            batch_size: BATCHED_B,
        };
        let err = lower_mutated_batched(phases);
        assert!(matches!(
            err,
            MoeLowerError::BatchedI64ChainMismatch {
                field: "topk_indices",
                ..
            }
        ));
    }

    #[test]
    fn lowering_rejects_batched_i64_disconnected_activation() {
        let chain = batched_chain();
        let mut phases = chain.rank();
        // The activation reads a different gate buffer (same capacity).
        phases.activation[0] = Step::MoeActivation {
            variant: MoeActivationVariant::Ds4ClampRotate { swiglu_limit: 30.0 },
            gate: synth_f32(BATCHED_B * BATCHED_K * BATCHED_INTER),
            up: chain.up_batch,
            rot_out: chain.rot_batch,
            inter: BATCHED_INTER,
            k_top: BATCHED_B * BATCHED_K,
        };
        let err = lower_mutated_batched(phases);
        assert!(matches!(
            err,
            MoeLowerError::BatchedI64ChainMismatch {
                field: "activation_gate",
                ..
            }
        ));

        let chain = batched_chain();
        let mut phases = chain.rank();
        phases.activation[0] = Step::MoeActivation {
            variant: MoeActivationVariant::Ds4ClampRotate { swiglu_limit: 30.0 },
            gate: chain.gate_batch,
            up: synth_f32(BATCHED_B * BATCHED_K * BATCHED_INTER),
            rot_out: chain.rot_batch,
            inter: BATCHED_INTER,
            k_top: BATCHED_B * BATCHED_K,
        };
        let err = lower_mutated_batched(phases);
        assert!(matches!(
            err,
            MoeLowerError::BatchedI64ChainMismatch {
                field: "activation_up",
                ..
            }
        ));
    }

    #[test]
    fn lowering_rejects_batched_i64_disconnected_down() {
        let chain = batched_chain();
        let mut phases = chain.rank();
        // The down consumes a different rotated buffer (same capacity).
        phases.down[0] = Step::IndexedMoeGemv {
            experts: chain.experts,
            which: MoeProj::DownResidualI64 {
                topk_weights: chain.topk_weights,
            },
            topk_indices: chain.topk_indices,
            input: GemvInput::Raw(synth_f32(BATCHED_B * BATCHED_K * BATCHED_INTER)),
            out: chain.out,
            k_top: BATCHED_K,
            batch_size: BATCHED_B,
        };
        let err = lower_mutated_batched(phases);
        assert!(matches!(
            err,
            MoeLowerError::BatchedI64ChainMismatch {
                field: "down_input",
                ..
            }
        ));
    }

    #[test]
    fn lowering_rejects_batched_i64_activation_rows_mismatch() {
        let chain = batched_chain();
        let mut phases = chain.rank();
        phases.activation[0] = Step::MoeActivation {
            variant: MoeActivationVariant::Ds4ClampRotate { swiglu_limit: 30.0 },
            gate: chain.gate_batch,
            up: chain.up_batch,
            rot_out: chain.rot_batch,
            inter: BATCHED_INTER,
            k_top: 5,
        };
        let err = lower_mutated_batched(phases);
        assert!(matches!(
            err,
            MoeLowerError::BatchedI64ActivationRows {
                expected: 4,
                actual: 5,
                ..
            }
        ));
    }

    #[test]
    fn lowering_rejects_batched_i64_non_256_aligned_contractions() {
        // Gate-up hidden (expert_k) misaligned; down inter_local stays aligned.
        let chain = batched_chain();
        let experts = lloyd_expert_ref_with(4, BATCHED_INTER, 128, DType::MQ2G256Lloyd);
        let mut phases = chain.rank();
        phases.gate_up[0] = Step::IndexedMoeGemv {
            experts,
            which: MoeProj::GateUp {
                up_out: chain.up_batch,
            },
            topk_indices: chain.topk_indices,
            input: GemvInput::Raw(chain.x_rot),
            out: chain.gate_batch,
            k_top: BATCHED_K,
            batch_size: BATCHED_B,
        };
        phases.down[0] = Step::IndexedMoeGemv {
            experts,
            which: MoeProj::DownResidualI64 {
                topk_weights: chain.topk_weights,
            },
            topk_indices: chain.topk_indices,
            input: GemvInput::Raw(chain.rot_batch),
            out: chain.out,
            k_top: BATCHED_K,
            batch_size: BATCHED_B,
        };
        let err = lower_mutated_batched(phases);
        assert!(matches!(
            err,
            MoeLowerError::BatchedI64Alignment {
                field: "gate_up_hidden",
                expected: 256,
                actual: 128,
                ..
            }
        ));

        // Down inter_local (expert_m) misaligned; gate-up hidden stays aligned.
        // The activation inter must track the connected (misaligned) expert_m
        // so the alignment check is what fires, not the inter mismatch.
        let chain = batched_chain();
        let experts = lloyd_expert_ref_with(4, 128, BATCHED_HIDDEN, DType::MQ2G256Lloyd);
        let mut phases = chain.rank();
        phases.gate_up[0] = Step::IndexedMoeGemv {
            experts,
            which: MoeProj::GateUp {
                up_out: chain.up_batch,
            },
            topk_indices: chain.topk_indices,
            input: GemvInput::Raw(chain.x_rot),
            out: chain.gate_batch,
            k_top: BATCHED_K,
            batch_size: BATCHED_B,
        };
        phases.activation[0] = Step::MoeActivation {
            variant: MoeActivationVariant::Ds4ClampRotate { swiglu_limit: 30.0 },
            gate: chain.gate_batch,
            up: chain.up_batch,
            rot_out: chain.rot_batch,
            inter: 128,
            k_top: BATCHED_B * BATCHED_K,
        };
        phases.down[0] = Step::IndexedMoeGemv {
            experts,
            which: MoeProj::DownResidualI64 {
                topk_weights: chain.topk_weights,
            },
            topk_indices: chain.topk_indices,
            input: GemvInput::Raw(chain.rot_batch),
            out: chain.out,
            k_top: BATCHED_K,
            batch_size: BATCHED_B,
        };
        let err = lower_mutated_batched(phases);
        assert!(matches!(
            err,
            MoeLowerError::BatchedI64Alignment {
                field: "down_inter_local",
                expected: 256,
                actual: 128,
                ..
            }
        ));
    }

    /// Lower a batched program whose chain is mutated by `mutate` (single
    /// rank; phase lengths unchanged) and return the error.
    fn batched_chain_error(
        mutate: impl Fn(&BatchedChain, &mut RoutedMoeStepPhases<'static>),
    ) -> MoeLowerError {
        let chain = batched_chain();
        let mut phases = chain.rank();
        mutate(&chain, &mut phases);
        lower_mutated_batched(phases)
    }

    #[test]
    fn lowering_rejects_batched_i64_activation_inter_mismatch() {
        // The activation's inter must equal the connected expert_m; a smaller
        // or different inter is incoherent with the gate-up/down kernels and
        // must never size the gate/up/rotation capacities.
        let err = batched_chain_error(|chain, phases| {
            phases.activation[0] = Step::MoeActivation {
                variant: MoeActivationVariant::Ds4ClampRotate { swiglu_limit: 30.0 },
                gate: chain.gate_batch,
                up: chain.up_batch,
                rot_out: chain.rot_batch,
                inter: BATCHED_INTER / 2,
                k_top: BATCHED_B * BATCHED_K,
            };
        });
        assert!(matches!(
            err,
            MoeLowerError::BatchedI64ChainMismatch {
                field: "activation_inter",
                ..
            }
        ));
    }

    #[test]
    fn lowering_rejects_batched_i64_mq3_chain() {
        // A fully coherent MQ3 batched chain must be rejected at lowering:
        // the batched protocol is MQ2G256Lloyd-only, never deferred to a
        // dispatch-time scalar fallback.
        let err = batched_chain_error(|chain, phases| {
            let experts =
                lloyd_expert_ref_with(4, BATCHED_INTER, BATCHED_HIDDEN, DType::MQ3G256Lloyd);
            phases.gate_up[0] = Step::IndexedMoeGemv {
                experts,
                which: MoeProj::GateUp {
                    up_out: chain.up_batch,
                },
                topk_indices: chain.topk_indices,
                input: GemvInput::Raw(chain.x_rot),
                out: chain.gate_batch,
                k_top: BATCHED_K,
                batch_size: BATCHED_B,
            };
            phases.down[0] = Step::IndexedMoeGemv {
                experts,
                which: MoeProj::DownResidualI64 {
                    topk_weights: chain.topk_weights,
                },
                topk_indices: chain.topk_indices,
                input: GemvInput::Raw(chain.rot_batch),
                out: chain.out,
                k_top: BATCHED_K,
                batch_size: BATCHED_B,
            };
        });
        assert!(matches!(
            err,
            MoeLowerError::BatchedI64Dtype {
                expected: DType::MQ2G256Lloyd,
                actual: DType::MQ3G256Lloyd,
                ..
            }
        ));
    }

    #[test]
    fn lowering_rejects_batched_i64_zero_geometry() {
        // n_experts = 0.
        let err = batched_chain_error(|chain, phases| {
            let experts =
                lloyd_expert_ref_with(0, BATCHED_INTER, BATCHED_HIDDEN, DType::MQ2G256Lloyd);
            phases.gate_up[0] = Step::IndexedMoeGemv {
                experts,
                which: MoeProj::GateUp {
                    up_out: chain.up_batch,
                },
                topk_indices: chain.topk_indices,
                input: GemvInput::Raw(chain.x_rot),
                out: chain.gate_batch,
                k_top: BATCHED_K,
                batch_size: BATCHED_B,
            };
            phases.down[0] = Step::IndexedMoeGemv {
                experts,
                which: MoeProj::DownResidualI64 {
                    topk_weights: chain.topk_weights,
                },
                topk_indices: chain.topk_indices,
                input: GemvInput::Raw(chain.rot_batch),
                out: chain.out,
                k_top: BATCHED_K,
                batch_size: BATCHED_B,
            };
        });
        assert!(matches!(
            err,
            MoeLowerError::BatchedI64ZeroGeometry {
                field: "n_experts",
                ..
            }
        ));

        // expert_m = 0 (zero is 256-divisible, so only the zero check catches
        // it; today the alignment check silently passes).
        let err = batched_chain_error(|chain, phases| {
            let experts = lloyd_expert_ref_with(4, 0, BATCHED_HIDDEN, DType::MQ2G256Lloyd);
            phases.gate_up[0] = Step::IndexedMoeGemv {
                experts,
                which: MoeProj::GateUp {
                    up_out: chain.up_batch,
                },
                topk_indices: chain.topk_indices,
                input: GemvInput::Raw(chain.x_rot),
                out: chain.gate_batch,
                k_top: BATCHED_K,
                batch_size: BATCHED_B,
            };
            phases.down[0] = Step::IndexedMoeGemv {
                experts,
                which: MoeProj::DownResidualI64 {
                    topk_weights: chain.topk_weights,
                },
                topk_indices: chain.topk_indices,
                input: GemvInput::Raw(chain.rot_batch),
                out: chain.out,
                k_top: BATCHED_K,
                batch_size: BATCHED_B,
            };
        });
        assert!(matches!(
            err,
            MoeLowerError::BatchedI64ZeroGeometry {
                field: "expert_m",
                ..
            }
        ));

        // expert_k = 0.
        let err = batched_chain_error(|chain, phases| {
            let experts = lloyd_expert_ref_with(4, BATCHED_INTER, 0, DType::MQ2G256Lloyd);
            phases.gate_up[0] = Step::IndexedMoeGemv {
                experts,
                which: MoeProj::GateUp {
                    up_out: chain.up_batch,
                },
                topk_indices: chain.topk_indices,
                input: GemvInput::Raw(chain.x_rot),
                out: chain.gate_batch,
                k_top: BATCHED_K,
                batch_size: BATCHED_B,
            };
            phases.down[0] = Step::IndexedMoeGemv {
                experts,
                which: MoeProj::DownResidualI64 {
                    topk_weights: chain.topk_weights,
                },
                topk_indices: chain.topk_indices,
                input: GemvInput::Raw(chain.rot_batch),
                out: chain.out,
                k_top: BATCHED_K,
                batch_size: BATCHED_B,
            };
        });
        assert!(matches!(
            err,
            MoeLowerError::BatchedI64ZeroGeometry {
                field: "expert_k",
                ..
            }
        ));

        // Activation inter = 0.
        let err = batched_chain_error(|chain, phases| {
            phases.activation[0] = Step::MoeActivation {
                variant: MoeActivationVariant::Ds4ClampRotate { swiglu_limit: 30.0 },
                gate: chain.gate_batch,
                up: chain.up_batch,
                rot_out: chain.rot_batch,
                inter: 0,
                k_top: BATCHED_B * BATCHED_K,
            };
        });
        assert!(matches!(
            err,
            MoeLowerError::BatchedI64ZeroGeometry {
                field: "activation_inter",
                ..
            }
        ));
    }

    #[test]
    fn lowering_rejects_batched_i64_k_top_exceeds_experts() {
        let err = batched_chain_error(|chain, phases| {
            phases.gate_up[0] = Step::IndexedMoeGemv {
                experts: chain.experts,
                which: MoeProj::GateUp {
                    up_out: chain.up_batch,
                },
                topk_indices: chain.topk_indices,
                input: GemvInput::Raw(chain.x_rot),
                out: chain.gate_batch,
                k_top: 8,
                batch_size: BATCHED_B,
            };
            phases.down[0] = Step::IndexedMoeGemv {
                experts: chain.experts,
                which: MoeProj::DownResidualI64 {
                    topk_weights: chain.topk_weights,
                },
                topk_indices: chain.topk_indices,
                input: GemvInput::Raw(chain.rot_batch),
                out: chain.out,
                k_top: 8,
                batch_size: BATCHED_B,
            };
        });
        assert!(matches!(
            err,
            MoeLowerError::BatchedI64KTopExceedsExperts {
                k_top: 8,
                n_experts: 4,
                ..
            }
        ));
    }

    #[test]
    fn lowering_rejects_batched_i64_kernel_abi_range() {
        // gate_up_m = 2*expert_m exceeds the i32 kernarg width used by
        // deepseek4_gemv_mq2g256_lloyd_moe_gate_up_indexed_batched_k4.
        let err = batched_chain_error(|chain, phases| {
            let experts = lloyd_expert_ref_with(
                4,
                i32::MAX as usize / 2 + 1,
                BATCHED_HIDDEN,
                DType::MQ2G256Lloyd,
            );
            phases.gate_up[0] = Step::IndexedMoeGemv {
                experts,
                which: MoeProj::GateUp {
                    up_out: chain.up_batch,
                },
                topk_indices: chain.topk_indices,
                input: GemvInput::Raw(chain.x_rot),
                out: chain.gate_batch,
                k_top: BATCHED_K,
                batch_size: BATCHED_B,
            };
            phases.activation[0] = Step::MoeActivation {
                variant: MoeActivationVariant::Ds4ClampRotate { swiglu_limit: 30.0 },
                gate: chain.gate_batch,
                up: chain.up_batch,
                rot_out: chain.rot_batch,
                inter: i32::MAX as usize / 2 + 1,
                k_top: BATCHED_B * BATCHED_K,
            };
            phases.down[0] = Step::IndexedMoeGemv {
                experts,
                which: MoeProj::DownResidualI64 {
                    topk_weights: chain.topk_weights,
                },
                topk_indices: chain.topk_indices,
                input: GemvInput::Raw(chain.rot_batch),
                out: chain.out,
                k_top: BATCHED_K,
                batch_size: BATCHED_B,
            };
        });
        assert!(matches!(
            &err,
            MoeLowerError::BatchedI64AbiRange {
                field: "gate_up_m",
                max,
                actual,
                ..
            } if *max == i32::MAX as usize && *actual == i32::MAX as usize + 1
        ));

        // expert_k exceeds the i32 kernarg width (gate-up k / down m).
        let err = batched_chain_error(|chain, phases| {
            let experts =
                lloyd_expert_ref_with(4, BATCHED_INTER, i32::MAX as usize + 1, DType::MQ2G256Lloyd);
            let out = synth_shape(DType::Raw, vec![BATCHED_B, i32::MAX as usize + 1], 64);
            phases.gate_up[0] = Step::IndexedMoeGemv {
                experts,
                which: MoeProj::GateUp {
                    up_out: chain.up_batch,
                },
                topk_indices: chain.topk_indices,
                input: GemvInput::Raw(chain.x_rot),
                out: chain.gate_batch,
                k_top: BATCHED_K,
                batch_size: BATCHED_B,
            };
            phases.down[0] = Step::IndexedMoeGemv {
                experts,
                which: MoeProj::DownResidualI64 {
                    topk_weights: chain.topk_weights,
                },
                topk_indices: chain.topk_indices,
                input: GemvInput::Raw(chain.rot_batch),
                out,
                k_top: BATCHED_K,
                batch_size: BATCHED_B,
            };
            phases.finish[0] = Step::ConvertI64ToF32 {
                src: out,
                dst: synth_f32(4),
                n: BATCHED_B * (i32::MAX as usize + 1),
            };
        });
        assert!(matches!(
            &err,
            MoeLowerError::BatchedI64AbiRange {
                field: "expert_k",
                max,
                actual,
                ..
            } if *max == i32::MAX as usize && *actual == i32::MAX as usize + 1
        ));

        // k_top exceeds the i32 kernarg width; n_experts is raised to match
        // so the k_top>n_experts check cannot mask the ABI boundary.
        let err = batched_chain_error(|chain, phases| {
            let k_top = i32::MAX as usize + 1;
            let experts = lloyd_expert_ref_with_tables(
                k_top,
                BATCHED_INTER,
                BATCHED_HIDDEN,
                DType::MQ2G256Lloyd,
                synth_fake_capacity(DType::Raw, vec![k_top], k_top * DEVICE_POINTER_BYTES),
                synth_fake_capacity(DType::Raw, vec![k_top], k_top * DEVICE_POINTER_BYTES),
            );
            phases.gate_up[0] = Step::IndexedMoeGemv {
                experts,
                which: MoeProj::GateUp {
                    up_out: chain.up_batch,
                },
                topk_indices: chain.topk_indices,
                input: GemvInput::Raw(chain.x_rot),
                out: chain.gate_batch,
                k_top,
                batch_size: BATCHED_B,
            };
            phases.activation[0] = Step::MoeActivation {
                variant: MoeActivationVariant::Ds4ClampRotate { swiglu_limit: 30.0 },
                gate: chain.gate_batch,
                up: chain.up_batch,
                rot_out: chain.rot_batch,
                inter: BATCHED_INTER,
                k_top: BATCHED_B * k_top,
            };
            phases.down[0] = Step::IndexedMoeGemv {
                experts,
                which: MoeProj::DownResidualI64 {
                    topk_weights: chain.topk_weights,
                },
                topk_indices: chain.topk_indices,
                input: GemvInput::Raw(chain.rot_batch),
                out: chain.out,
                k_top,
                batch_size: BATCHED_B,
            };
        });
        assert!(matches!(
            &err,
            MoeLowerError::BatchedI64AbiRange {
                field: "k_top",
                max,
                actual,
                ..
            } if *max == i32::MAX as usize && *actual == i32::MAX as usize + 1
        ));

        // batch_size exceeds the u32 grid-z width used by both `_batched_k4`
        // launchers.
        let err = batched_chain_error(|chain, phases| {
            let batch = u32::MAX as usize + 1;
            let out = synth_shape(DType::Raw, vec![batch, BATCHED_HIDDEN], 64);
            phases.gate_up[0] = Step::IndexedMoeGemv {
                experts: chain.experts,
                which: MoeProj::GateUp {
                    up_out: chain.up_batch,
                },
                topk_indices: chain.topk_indices,
                input: GemvInput::Raw(chain.x_rot),
                out: chain.gate_batch,
                k_top: BATCHED_K,
                batch_size: batch,
            };
            phases.activation[0] = Step::MoeActivation {
                variant: MoeActivationVariant::Ds4ClampRotate { swiglu_limit: 30.0 },
                gate: chain.gate_batch,
                up: chain.up_batch,
                rot_out: chain.rot_batch,
                inter: BATCHED_INTER,
                k_top: batch * BATCHED_K,
            };
            phases.down[0] = Step::IndexedMoeGemv {
                experts: chain.experts,
                which: MoeProj::DownResidualI64 {
                    topk_weights: chain.topk_weights,
                },
                topk_indices: chain.topk_indices,
                input: GemvInput::Raw(chain.rot_batch),
                out,
                k_top: BATCHED_K,
                batch_size: batch,
            };
            phases.finish[0] = Step::ConvertI64ToF32 {
                src: out,
                dst: synth_f32(4),
                n: batch * BATCHED_HIDDEN,
            };
        });
        assert!(matches!(
            &err,
            MoeLowerError::BatchedI64AbiRange {
                field: "batch_size",
                max,
                actual,
                ..
            } if *max == u32::MAX as usize && *actual == u32::MAX as usize + 1
        ));
    }

    #[test]
    fn lowering_rejects_batched_i64_gate_up_m_product_overflow() {
        // 2*expert_m overflows usize; the checked product must be reported as
        // arithmetic overflow before any alignment or capacity check.
        let err = batched_chain_error(|chain, phases| {
            let experts = lloyd_expert_ref_with(4, usize::MAX, BATCHED_HIDDEN, DType::MQ2G256Lloyd);
            phases.gate_up[0] = Step::IndexedMoeGemv {
                experts,
                which: MoeProj::GateUp {
                    up_out: chain.up_batch,
                },
                topk_indices: chain.topk_indices,
                input: GemvInput::Raw(chain.x_rot),
                out: chain.gate_batch,
                k_top: BATCHED_K,
                batch_size: BATCHED_B,
            };
            phases.activation[0] = Step::MoeActivation {
                variant: MoeActivationVariant::Ds4ClampRotate { swiglu_limit: 30.0 },
                gate: chain.gate_batch,
                up: chain.up_batch,
                rot_out: chain.rot_batch,
                inter: usize::MAX,
                k_top: BATCHED_B * BATCHED_K,
            };
            phases.down[0] = Step::IndexedMoeGemv {
                experts,
                which: MoeProj::DownResidualI64 {
                    topk_weights: chain.topk_weights,
                },
                topk_indices: chain.topk_indices,
                input: GemvInput::Raw(chain.rot_batch),
                out: chain.out,
                k_top: BATCHED_K,
                batch_size: BATCHED_B,
            };
        });
        assert!(matches!(
            err,
            MoeLowerError::ArithmeticOverflow {
                what: "gate_up_m",
                ..
            }
        ));
    }

    #[test]
    fn lowering_rejects_batched_i64_undersized_pointer_tables() {
        // Gate-up pointer table below n_experts * 8 bytes (16 < 32); the
        // down table is valid (32 bytes) so ONLY the gate-up table can fire.
        // The asserted absolute index 0 is the gate-up step's, identifying
        // the operand beyond the generic capacity bytes.
        let err = batched_chain_error(|chain, phases| {
            let experts = lloyd_expert_ref_with_tables(
                4,
                BATCHED_INTER,
                BATCHED_HIDDEN,
                DType::MQ2G256Lloyd,
                synth_f32(4),
                synth_f32(8),
            );
            phases.gate_up[0] = Step::IndexedMoeGemv {
                experts,
                which: MoeProj::GateUp {
                    up_out: chain.up_batch,
                },
                topk_indices: chain.topk_indices,
                input: GemvInput::Raw(chain.x_rot),
                out: chain.gate_batch,
                k_top: BATCHED_K,
                batch_size: BATCHED_B,
            };
            phases.down[0] = Step::IndexedMoeGemv {
                experts,
                which: MoeProj::DownResidualI64 {
                    topk_weights: chain.topk_weights,
                },
                topk_indices: chain.topk_indices,
                input: GemvInput::Raw(chain.rot_batch),
                out: chain.out,
                k_top: BATCHED_K,
                batch_size: BATCHED_B,
            };
        });
        assert!(matches!(
            err,
            MoeLowerError::CapacityMismatch {
                index: 0,
                expected_bytes: 32,
                actual_bytes: 16,
                ..
            }
        ));

        // Down pointer table below n_experts * 8 bytes (16 < 32); the
        // gate-up table is valid (32 bytes) so ONLY the down table can fire.
        // The asserted absolute index 2 is the down step's, identifying the
        // operand beyond the generic capacity bytes.
        let err = batched_chain_error(|chain, phases| {
            let experts = lloyd_expert_ref_with_tables(
                4,
                BATCHED_INTER,
                BATCHED_HIDDEN,
                DType::MQ2G256Lloyd,
                synth_f32(8),
                synth_f32(4),
            );
            phases.gate_up[0] = Step::IndexedMoeGemv {
                experts,
                which: MoeProj::GateUp {
                    up_out: chain.up_batch,
                },
                topk_indices: chain.topk_indices,
                input: GemvInput::Raw(chain.x_rot),
                out: chain.gate_batch,
                k_top: BATCHED_K,
                batch_size: BATCHED_B,
            };
            phases.down[0] = Step::IndexedMoeGemv {
                experts,
                which: MoeProj::DownResidualI64 {
                    topk_weights: chain.topk_weights,
                },
                topk_indices: chain.topk_indices,
                input: GemvInput::Raw(chain.rot_batch),
                out: chain.out,
                k_top: BATCHED_K,
                batch_size: BATCHED_B,
            };
        });
        assert!(matches!(
            err,
            MoeLowerError::CapacityMismatch {
                index: 2,
                expected_bytes: 32,
                actual_bytes: 16,
                ..
            }
        ));
    }

    #[test]
    fn lowering_rejects_batched_i64_output_shape_not_batch_hidden() {
        let chain = batched_chain();
        let bad_out = synth_shape(
            DType::Raw,
            vec![BATCHED_B * BATCHED_HIDDEN],
            BATCHED_B * BATCHED_HIDDEN * 8,
        );
        let mut phases = chain.rank();
        phases.down[0] = Step::IndexedMoeGemv {
            experts: chain.experts,
            which: MoeProj::DownResidualI64 {
                topk_weights: chain.topk_weights,
            },
            topk_indices: chain.topk_indices,
            input: GemvInput::Raw(chain.rot_batch),
            out: bad_out,
            k_top: BATCHED_K,
            batch_size: BATCHED_B,
        };
        phases.finish[0] = Step::ConvertI64ToF32 {
            src: bad_out,
            dst: chain.dst,
            n: BATCHED_B * BATCHED_HIDDEN,
        };
        let err = lower_mutated_batched(phases);
        assert!(matches!(
            err,
            MoeLowerError::BatchedI64OutputShape {
                expected: [2, 256],
                ..
            }
        ));
    }

    #[test]
    fn lowering_rejects_batched_i64_conversion_n_mismatch() {
        let chain = batched_chain();
        let mut phases = chain.rank();
        phases.finish[0] = Step::ConvertI64ToF32 {
            src: chain.out,
            dst: chain.dst,
            n: BATCHED_B * BATCHED_HIDDEN + 1,
        };
        let err = lower_mutated_batched(phases);
        assert!(matches!(
            err,
            MoeLowerError::I64DimensionMismatch {
                expected: 512,
                actual: 513,
                ..
            }
        ));
    }

    #[test]
    fn lowering_rejects_batched_i64_undersized_operands() {
        // top-k indices below slots*4 bytes.
        let chain = batched_chain();
        let undersized = synth_i32(BATCHED_B * BATCHED_K - 1);
        let mut phases = chain.rank();
        phases.gate_up[0] = Step::IndexedMoeGemv {
            experts: chain.experts,
            which: MoeProj::GateUp {
                up_out: chain.up_batch,
            },
            topk_indices: undersized,
            input: GemvInput::Raw(chain.x_rot),
            out: chain.gate_batch,
            k_top: BATCHED_K,
            batch_size: BATCHED_B,
        };
        phases.down[0] = Step::IndexedMoeGemv {
            experts: chain.experts,
            which: MoeProj::DownResidualI64 {
                topk_weights: chain.topk_weights,
            },
            topk_indices: undersized,
            input: GemvInput::Raw(chain.rot_batch),
            out: chain.out,
            k_top: BATCHED_K,
            batch_size: BATCHED_B,
        };
        let err = lower_mutated_batched(phases);
        assert!(matches!(
            err,
            MoeLowerError::CapacityMismatch {
                expected_bytes: 16,
                actual_bytes: 12,
                ..
            }
        ));

        // top-k weights below slots*4 bytes.
        let chain = batched_chain();
        let mut phases = chain.rank();
        phases.down[0] = Step::IndexedMoeGemv {
            experts: chain.experts,
            which: MoeProj::DownResidualI64 {
                topk_weights: synth_f32(BATCHED_B * BATCHED_K - 1),
            },
            topk_indices: chain.topk_indices,
            input: GemvInput::Raw(chain.rot_batch),
            out: chain.out,
            k_top: BATCHED_K,
            batch_size: BATCHED_B,
        };
        let err = lower_mutated_batched(phases);
        assert!(matches!(
            err,
            MoeLowerError::CapacityMismatch {
                expected_bytes: 16,
                actual_bytes: 12,
                ..
            }
        ));

        // Gate-up input below batch*hidden*4 bytes.
        let chain = batched_chain();
        let mut phases = chain.rank();
        phases.gate_up[0] = Step::IndexedMoeGemv {
            experts: chain.experts,
            which: MoeProj::GateUp {
                up_out: chain.up_batch,
            },
            topk_indices: chain.topk_indices,
            input: GemvInput::Raw(synth_f32(BATCHED_B * BATCHED_HIDDEN - 1)),
            out: chain.gate_batch,
            k_top: BATCHED_K,
            batch_size: BATCHED_B,
        };
        let err = lower_mutated_batched(phases);
        assert!(matches!(
            err,
            MoeLowerError::CapacityMismatch {
                expected_bytes: 2048,
                actual_bytes: 2044,
                ..
            }
        ));

        // Activation gate below slots*inter_local*4 bytes.
        let chain = batched_chain();
        let gate = synth_f32(BATCHED_B * BATCHED_K * BATCHED_INTER - 1);
        let mut phases = chain.rank();
        phases.gate_up[0] = Step::IndexedMoeGemv {
            experts: chain.experts,
            which: MoeProj::GateUp {
                up_out: chain.up_batch,
            },
            topk_indices: chain.topk_indices,
            input: GemvInput::Raw(chain.x_rot),
            out: gate,
            k_top: BATCHED_K,
            batch_size: BATCHED_B,
        };
        phases.activation[0] = Step::MoeActivation {
            variant: MoeActivationVariant::Ds4ClampRotate { swiglu_limit: 30.0 },
            gate,
            up: chain.up_batch,
            rot_out: chain.rot_batch,
            inter: BATCHED_INTER,
            k_top: BATCHED_B * BATCHED_K,
        };
        let err = lower_mutated_batched(phases);
        assert!(matches!(
            err,
            MoeLowerError::CapacityMismatch {
                expected_bytes: 4096,
                actual_bytes: 4092,
                ..
            }
        ));

        // Activation up below slots*inter_local*4 bytes.
        let chain = batched_chain();
        let up = synth_f32(BATCHED_B * BATCHED_K * BATCHED_INTER - 1);
        let mut phases = chain.rank();
        phases.gate_up[0] = Step::IndexedMoeGemv {
            experts: chain.experts,
            which: MoeProj::GateUp { up_out: up },
            topk_indices: chain.topk_indices,
            input: GemvInput::Raw(chain.x_rot),
            out: chain.gate_batch,
            k_top: BATCHED_K,
            batch_size: BATCHED_B,
        };
        phases.activation[0] = Step::MoeActivation {
            variant: MoeActivationVariant::Ds4ClampRotate { swiglu_limit: 30.0 },
            gate: chain.gate_batch,
            up,
            rot_out: chain.rot_batch,
            inter: BATCHED_INTER,
            k_top: BATCHED_B * BATCHED_K,
        };
        let err = lower_mutated_batched(phases);
        assert!(matches!(
            err,
            MoeLowerError::CapacityMismatch {
                expected_bytes: 4096,
                actual_bytes: 4092,
                ..
            }
        ));

        // Rotation output below slots*inter_local*4 bytes.
        let chain = batched_chain();
        let rot = synth_f32(BATCHED_B * BATCHED_K * BATCHED_INTER - 1);
        let mut phases = chain.rank();
        phases.activation[0] = Step::MoeActivation {
            variant: MoeActivationVariant::Ds4ClampRotate { swiglu_limit: 30.0 },
            gate: chain.gate_batch,
            up: chain.up_batch,
            rot_out: rot,
            inter: BATCHED_INTER,
            k_top: BATCHED_B * BATCHED_K,
        };
        phases.down[0] = Step::IndexedMoeGemv {
            experts: chain.experts,
            which: MoeProj::DownResidualI64 {
                topk_weights: chain.topk_weights,
            },
            topk_indices: chain.topk_indices,
            input: GemvInput::Raw(rot),
            out: chain.out,
            k_top: BATCHED_K,
            batch_size: BATCHED_B,
        };
        let err = lower_mutated_batched(phases);
        assert!(matches!(
            err,
            MoeLowerError::CapacityMismatch {
                expected_bytes: 4096,
                actual_bytes: 4092,
                ..
            }
        ));

        // I64 output below batch*hidden*8 bytes.
        let chain = batched_chain();
        let out = synth_shape(
            DType::Raw,
            vec![BATCHED_B, BATCHED_HIDDEN],
            BATCHED_B * BATCHED_HIDDEN * 8 - 4,
        );
        let mut phases = chain.rank();
        phases.down[0] = Step::IndexedMoeGemv {
            experts: chain.experts,
            which: MoeProj::DownResidualI64 {
                topk_weights: chain.topk_weights,
            },
            topk_indices: chain.topk_indices,
            input: GemvInput::Raw(chain.rot_batch),
            out,
            k_top: BATCHED_K,
            batch_size: BATCHED_B,
        };
        phases.finish[0] = Step::ConvertI64ToF32 {
            src: out,
            dst: chain.dst,
            n: BATCHED_B * BATCHED_HIDDEN,
        };
        let err = lower_mutated_batched(phases);
        assert!(matches!(
            err,
            MoeLowerError::CapacityMismatch {
                expected_bytes: 4096,
                actual_bytes: 4092,
                ..
            }
        ));
    }

    #[test]
    fn lowering_rejects_batched_i64_activation_rows_overflow() {
        // batch_size * k_top overflows usize; reported as arithmetic overflow,
        // never a panic or wrap.
        let topk = synth_i32(4);
        let gate = synth_f32(4);
        let up = synth_f32(4);
        let rot = synth_f32(4);
        let experts = lloyd_expert_ref();
        let rank = || {
            let out = synth_shape(DType::Raw, vec![usize::MAX, BATCHED_HIDDEN], 64);
            RoutedMoePhases {
                router: Vec::new(),
                gate_up: vec![Step::IndexedMoeGemv {
                    experts,
                    which: MoeProj::GateUp { up_out: up },
                    topk_indices: topk,
                    input: GemvInput::Raw(synth_f32(4)),
                    out: gate,
                    k_top: BATCHED_K,
                    batch_size: usize::MAX,
                }],
                activation: vec![Step::MoeActivation {
                    variant: MoeActivationVariant::Ds4ClampRotate { swiglu_limit: 30.0 },
                    gate,
                    up,
                    rot_out: rot,
                    inter: BATCHED_INTER,
                    // Never reached: the checked batch*k_top product overflows
                    // before this row count is compared.
                    k_top: 1,
                }],
                down: vec![Step::IndexedMoeGemv {
                    experts,
                    which: MoeProj::DownResidualI64 {
                        topk_weights: synth_f32(4),
                    },
                    topk_indices: topk,
                    input: GemvInput::Raw(rot),
                    out,
                    k_top: BATCHED_K,
                    batch_size: usize::MAX,
                }],
                combine: Vec::new(),
                finish: vec![Step::ConvertI64ToF32 {
                    src: out,
                    dst: out,
                    n: 1,
                }],
            }
        };
        let group = group(ExpertParallelism::TensorParallel, 2);
        let policy = tp_policy(2);
        let err = lower_moe_steps(&group, &policy, parts(vec![rank(), rank()])).unwrap_err();
        assert!(matches!(err, MoeLowerError::ArithmeticOverflow { .. }));
        assert!(err.to_string().contains("activation_rows"));
    }

    #[test]
    fn lowering_rejects_batched_i64_batch_hidden_overflow() {
        // batch_size * hidden overflows usize while the routed rows
        // (batch_size * k_top) still fit; the exact [batch, hidden] shape is
        // present but its logical product overflows.
        let batch = usize::MAX / BATCHED_HIDDEN + 1;
        let topk = synth_i32(4);
        let gate = synth_f32(4);
        let up = synth_f32(4);
        let rot = synth_f32(4);
        let experts = lloyd_expert_ref();
        let rank = || {
            let out = synth_shape(DType::Raw, vec![batch, BATCHED_HIDDEN], 64);
            RoutedMoePhases {
                router: Vec::new(),
                gate_up: vec![Step::IndexedMoeGemv {
                    experts,
                    which: MoeProj::GateUp { up_out: up },
                    topk_indices: topk,
                    input: GemvInput::Raw(synth_f32(4)),
                    out: gate,
                    k_top: 1,
                    batch_size: batch,
                }],
                activation: vec![Step::MoeActivation {
                    variant: MoeActivationVariant::Ds4ClampRotate { swiglu_limit: 30.0 },
                    gate,
                    up,
                    rot_out: rot,
                    inter: BATCHED_INTER,
                    k_top: batch,
                }],
                down: vec![Step::IndexedMoeGemv {
                    experts,
                    which: MoeProj::DownResidualI64 {
                        topk_weights: synth_f32(4),
                    },
                    topk_indices: topk,
                    input: GemvInput::Raw(rot),
                    out,
                    k_top: 1,
                    batch_size: batch,
                }],
                combine: Vec::new(),
                finish: vec![Step::ConvertI64ToF32 {
                    src: out,
                    dst: out,
                    n: 1,
                }],
            }
        };
        let group = group(ExpertParallelism::TensorParallel, 2);
        let policy = tp_policy(2);
        let err = lower_moe_steps(&group, &policy, parts(vec![rank(), rank()])).unwrap_err();
        assert!(matches!(err, MoeLowerError::ArithmeticOverflow { .. }));
        assert!(err.to_string().contains("logical_shape"));
    }

    // ── Cross-rank batched indexed I64 signature (Oracle quality) ──────────
    // Two ranks may disagree on per-rank geometry while the flattened
    // collective dim (batch*hidden) stays equal; the sealed schedule must
    // compare every rank-invariant scalar, not just the dim.

    /// A per-rank-valid batched indexed I64 chain parameterized by the
    /// signature scalars. `batch_size == 1` yields a per-rank-valid SCALAR
    /// i64 chain (no alignment/chain constraints) with the same phase shape.
    fn batched_sig_rank(
        batch_size: usize,
        k_top: usize,
        n_experts: usize,
        expert_m: usize,
        expert_k: usize,
    ) -> RoutedMoeStepPhases<'static> {
        let topk_indices = synth_i32(batch_size * k_top);
        let topk_weights = synth_f32(batch_size * k_top);
        let x_rot = synth_f32(batch_size * expert_k);
        let gate_batch = synth_f32(batch_size * k_top * expert_m);
        let up_batch = synth_f32(batch_size * k_top * expert_m);
        let rot_batch = synth_f32(batch_size * k_top * expert_m);
        let out = synth_shape(
            DType::Raw,
            vec![batch_size, expert_k],
            batch_size * expert_k * 8,
        );
        let dst = synth_f32(batch_size * expert_k);
        let experts = lloyd_expert_ref_with(n_experts, expert_m, expert_k, DType::MQ2G256Lloyd);
        RoutedMoePhases {
            router: Vec::new(),
            gate_up: vec![Step::IndexedMoeGemv {
                experts,
                which: MoeProj::GateUp { up_out: up_batch },
                topk_indices,
                input: GemvInput::Raw(x_rot),
                out: gate_batch,
                k_top,
                batch_size,
            }],
            activation: vec![Step::MoeActivation {
                variant: MoeActivationVariant::Ds4ClampRotate { swiglu_limit: 30.0 },
                gate: gate_batch,
                up: up_batch,
                rot_out: rot_batch,
                inter: expert_m,
                k_top: batch_size * k_top,
            }],
            down: vec![Step::IndexedMoeGemv {
                experts,
                which: MoeProj::DownResidualI64 { topk_weights },
                topk_indices,
                input: GemvInput::Raw(rot_batch),
                out,
                k_top,
                batch_size,
            }],
            combine: Vec::new(),
            finish: vec![Step::ConvertI64ToF32 {
                src: out,
                dst,
                n: batch_size * expert_k,
            }],
        }
    }

    fn lower_batched_pair_err(
        rank_a: RoutedMoeStepPhases<'static>,
        rank_b: RoutedMoeStepPhases<'static>,
    ) -> MoeLowerError {
        let group = group(ExpertParallelism::TensorParallel, 2);
        let policy = tp_policy(2);
        lower_moe_steps(&group, &policy, parts(vec![rank_a, rank_b])).unwrap_err()
    }

    /// Lower two ranks and run `f` on the sealed program while the mesh
    /// policy is still alive (the program borrows it).
    fn with_lowered_batched_pair<T>(
        rank_a: RoutedMoeStepPhases<'static>,
        rank_b: RoutedMoeStepPhases<'static>,
        f: impl FnOnce(&LoweredMoeProgram<'_, '_>) -> T,
    ) -> T {
        let group = group(ExpertParallelism::TensorParallel, 2);
        let policy = tp_policy(2);
        let lowered = lower_moe_steps(&group, &policy, parts(vec![rank_a, rank_b])).unwrap();
        f(&lowered)
    }

    #[test]
    fn lowering_rejects_cross_rank_batched_geometry_mismatch_with_equal_dim() {
        // Rank A: batch 2 × hidden 512 (dim 1024). Rank B: batch 4 × hidden
        // 256 (dim 1024). Both are per-rank valid (256-aligned); the flattened
        // collective dim is equal, so today the pair passes cross-rank — and
        // would run different per-rank kernel geometry. The signature must
        // reject it.
        let err = lower_batched_pair_err(
            batched_sig_rank(2, 2, 4, 256, 512),
            batched_sig_rank(4, 2, 4, 256, 256),
        );
        assert!(matches!(err, MoeLowerError::RankProtocolMismatch { .. }));
    }

    #[test]
    fn lowering_rejects_cross_rank_batched_k_top_mismatch_with_equal_dim() {
        // Rank A: batch 2, k_top 2 (rows 4). Rank B: batch 2, k_top 1
        // (rows 2). Both per-rank valid; the collective dim (batch*hidden =
        // 512) is equal, so today the pair passes cross-rank. The signature
        // must reject it.
        let err = lower_batched_pair_err(
            batched_sig_rank(2, 2, 4, 256, 256),
            batched_sig_rank(2, 1, 4, 256, 256),
        );
        assert!(matches!(err, MoeLowerError::RankProtocolMismatch { .. }));
    }

    #[test]
    fn lowering_rejects_cross_rank_batched_signature_mismatches() {
        // Table-driven single-field mismatches against the base geometry
        // (batch 2, k_top 2, n_experts 4, expert_m 256, expert_k 256; dim 512
        // everywhere). dtype is NOT in the table: per-rank validation admits
        // only MQ2G256Lloyd batched chains, so dtype is already rank-invariant
        // by construction (it still travels in the signature defensively).
        let base = (2usize, 2usize, 4usize, 256usize, 256usize);
        let cases: [(&str, (usize, usize, usize, usize, usize)); 4] = [
            ("n_experts", (2, 2, 8, 256, 256)),
            ("expert_m", (2, 2, 4, 512, 256)),
            // expert_k/output hidden: with both ranks batched, expert_k
            // changes the dim (dim = batch*expert_k), so the only coherent
            // equal-dim isolate compensates batch — rank B is a valid SCALAR
            // chain (batch 1, hidden 512). The signature's batch/hidden
            // disagreement is what fires.
            (
                "expert_k/output hidden (scalar-compensated)",
                (1, 2, 4, 256, 512),
            ),
            // rows are derived from batch*k_top per rank, so a rows mismatch
            // isolates via k_top (same dim).
            ("k_top/rows", (2, 1, 4, 256, 256)),
        ];
        for (name, rank_b_geometry) in cases {
            let (batch, k_top, n_experts, expert_m, expert_k) = rank_b_geometry;
            let err = lower_batched_pair_err(
                batched_sig_rank(base.0, base.1, base.2, base.3, base.4),
                batched_sig_rank(batch, k_top, n_experts, expert_m, expert_k),
            );
            assert!(
                matches!(err, MoeLowerError::RankProtocolMismatch { .. }),
                "{name} mismatch must be rejected cross-rank"
            );
        }
    }

    #[test]
    fn lowering_accepts_identical_batched_signatures_across_ranks() {
        // Two identical non-default batched geometries (batch 4, dim 1024)
        // lower to the sealed one-zero/one-all-reduce/one-conversion schedule.
        with_lowered_batched_pair(
            batched_sig_rank(4, 2, 4, 256, 256),
            batched_sig_rank(4, 2, 4, 256, 256),
            |lowered| {
                let LoweredMoeProgramInner::Parallel {
                    collectives,
                    zero_before,
                    ..
                } = &lowered.inner
                else {
                    panic!("expected parallel program");
                };
                assert!(matches!(
                    collectives[2],
                    StepCollective::AllReduceI64Tp { dim: 1024 }
                ));
                assert!(zero_before[2]);
                assert_eq!(collective_count(collectives), 1);
            },
        );
    }

    #[test]
    fn lowering_derives_batched_expanded_combine_dimension() {
        let group = group(ExpertParallelism::TensorParallel, 2);
        let policy = tp_policy(2);
        let rank = || {
            let out = synth_f32(4);
            RoutedMoePhases {
                router: Vec::new(),
                gate_up: vec![gate_up_step()],
                activation: vec![activation_step()],
                down: vec![Step::IndexedMoeGemv {
                    experts: expert_ref(),
                    which: MoeProj::DownExpanded,
                    topk_indices: synth_i64(4),
                    input: GemvInput::Raw(synth_f32(4)),
                    out,
                    k_top: 2,
                    batch_size: 4,
                }],
                combine: vec![Step::MoeCombine {
                    down_out: out,
                    topk_weights: synth_f32(4),
                    out: synth_f32(16),
                    k: 2,
                    hidden: 4,
                    batch_size: 4,
                    inverse_perm: None,
                }],
                finish: Vec::new(),
            }
        };
        let lowered = lower_moe_steps(&group, &policy, parts(vec![rank(), rank()])).unwrap();
        let LoweredMoeProgramInner::Parallel { collectives, .. } = lowered.inner else {
            panic!("expected parallel program");
        };
        // Batched expanded combine: the collective covers batch_size * hidden.
        assert!(matches!(
            collectives[3],
            StepCollective::AllReduce {
                kind: DimKind::Tp,
                dim: 16
            }
        ));
        assert_eq!(collective_count(&collectives), 1);
    }

    #[test]
    fn lowering_rejects_expanded_combine_dimension_overflow() {
        let group = group(ExpertParallelism::TensorParallel, 2);
        let policy = tp_policy(2);
        // hidden=0 and batch_size=0 are invalid; hidden*batch_size overflow is
        // arithmetic overflow. All rejected contextually, never a panic.
        for (hidden, batch_size, expected) in [
            (0usize, 4usize, "InvalidCombineDimensions"),
            (4, 0, "InvalidCombineDimensions"),
            (usize::MAX, 2, "ArithmeticOverflow"),
        ] {
            let rank = || {
                let out = synth_f32(4);
                RoutedMoePhases {
                    router: Vec::new(),
                    gate_up: vec![gate_up_step()],
                    activation: vec![activation_step()],
                    down: vec![Step::IndexedMoeGemv {
                        experts: expert_ref(),
                        which: MoeProj::DownExpanded,
                        topk_indices: synth_i64(4),
                        input: GemvInput::Raw(synth_f32(4)),
                        out,
                        k_top: 2,
                        batch_size,
                    }],
                    combine: vec![Step::MoeCombine {
                        down_out: out,
                        topk_weights: synth_f32(4),
                        out: synth_f32(16),
                        k: 2,
                        hidden,
                        batch_size,
                        inverse_perm: None,
                    }],
                    finish: Vec::new(),
                }
            };
            let err = lower_moe_steps(&group, &policy, parts(vec![rank(), rank()])).unwrap_err();
            let display = err.to_string();
            assert!(display.contains("test"));
            if expected == "InvalidCombineDimensions" {
                assert!(matches!(
                    err,
                    MoeLowerError::InvalidCombineDimensions { .. }
                ));
            } else {
                assert!(matches!(err, MoeLowerError::ArithmeticOverflow { .. }));
            }
        }
    }

    #[test]
    fn lowering_rejects_expanded_combine_consuming_different_down_buffer() {
        let group = group(ExpertParallelism::TensorParallel, 2);
        let policy = tp_policy(2);
        let rank = || {
            let producer = synth_f32(4);
            RoutedMoePhases {
                router: Vec::new(),
                gate_up: vec![gate_up_step()],
                activation: vec![activation_step()],
                down: vec![Step::IndexedMoeGemv {
                    experts: expert_ref(),
                    which: MoeProj::DownExpanded,
                    topk_indices: synth_i64(4),
                    input: GemvInput::Raw(synth_f32(4)),
                    out: producer,
                    k_top: 2,
                    batch_size: 1,
                }],
                // The combine consumes a distinct buffer, not the down output.
                combine: vec![Step::MoeCombine {
                    down_out: synth_f32(4),
                    topk_weights: synth_f32(4),
                    out: synth_f32(4),
                    k: 2,
                    hidden: 4,
                    batch_size: 1,
                    inverse_perm: None,
                }],
                finish: Vec::new(),
            }
        };
        let err = lower_moe_steps(&group, &policy, parts(vec![rank(), rank()])).unwrap_err();
        assert!(matches!(
            err,
            MoeLowerError::CombineDownSourceMismatch { .. }
        ));
        let display = err.to_string();
        assert!(display.contains("test"));
        assert!(display.contains("Some(0)"));
    }

    #[test]
    fn lowering_accepts_expanded_combine_alias_wrappers_over_same_pointer() {
        let group = group(ExpertParallelism::TensorParallel, 2);
        let policy = tp_policy(2);
        // Two distinct GpuTensor wrappers over the same non-null pointer and
        // allocation size are the same buffer: the combine may alias the
        // expanded down producer.
        let storage: &'static [u8] = Box::leak(vec![0u8; 16].into_boxed_slice());
        let rank = move || {
            let pointer = storage.as_ptr() as *mut std::ffi::c_void;
            let producer = Box::leak(Box::new(GpuTensor {
                buf: unsafe { DeviceBuffer::from_raw(pointer, 16) },
                shape: vec![4],
                dtype: DType::F32,
            }));
            let alias = Box::leak(Box::new(GpuTensor {
                buf: unsafe { DeviceBuffer::from_raw(pointer, 16) },
                shape: vec![4],
                dtype: DType::F32,
            }));
            assert!(same_buffer(producer, alias));
            RoutedMoePhases {
                router: Vec::new(),
                gate_up: vec![gate_up_step()],
                activation: vec![activation_step()],
                down: vec![Step::IndexedMoeGemv {
                    experts: expert_ref(),
                    which: MoeProj::DownExpanded,
                    topk_indices: synth_i64(4),
                    input: GemvInput::Raw(synth_f32(4)),
                    out: producer,
                    k_top: 2,
                    batch_size: 1,
                }],
                combine: vec![Step::MoeCombine {
                    down_out: alias,
                    topk_weights: synth_f32(4),
                    out: synth_f32(4),
                    k: 2,
                    hidden: 4,
                    batch_size: 1,
                    inverse_perm: None,
                }],
                finish: Vec::new(),
            }
        };
        assert!(lower_moe_steps(&group, &policy, parts(vec![rank(), rank()])).is_ok());
    }

    /// Indexed expanded rank: `IndexedMoeGemv(DownExpanded)` down with an
    /// optional combine. `givens_phase` inserts a Paro pre-rotation Givens
    /// into the named phase and `specialized_gate_up` swaps the plain indexed
    /// gate-up for the specialized `MoeGateUpIndexed` — mirroring the decode
    /// router-phase (Givens before the route) and indexed-prefill gate_up-phase
    /// (Givens preamble before the gate-up) builder shapes.
    fn expanded_rank(
        combine: bool,
        givens_phase: Option<&'static str>,
        specialized_gate_up: bool,
    ) -> RoutedMoeStepPhases<'static> {
        let out = synth_f32(4);
        // `Step` is not `Clone`, so each placement builds its own rotation.
        let givens = || Step::GivensRotateBatched {
            x: synth_f32(4),
            out: synth_f32(4),
            pairs: synth_f32(4),
            theta: synth_f32(4),
            scales: synth_f32(4),
            batch: 1,
            dim: 4,
            krot: 4,
        };
        let mut router = Vec::new();
        if givens_phase == Some("router") {
            router.push(givens());
        }
        let mut gate_up = if specialized_gate_up {
            vec![Step::MoeGateUpIndexed {
                experts: expert_ref(),
                topk_indices: synth_i64(4),
                x_rot: synth_f32(4),
                gate_batch: synth_f32(4),
                up_batch: synth_f32(4),
                k_top: 2,
                batch_size: 1,
                dtype_tags: None,
            }]
        } else {
            vec![gate_up_step()]
        };
        if givens_phase == Some("gate_up") {
            gate_up.insert(0, givens());
        }
        let mut activation = vec![activation_step()];
        if givens_phase == Some("activation") {
            activation.insert(0, givens());
        }
        RoutedMoePhases {
            router,
            gate_up,
            activation,
            down: vec![Step::IndexedMoeGemv {
                experts: expert_ref(),
                which: MoeProj::DownExpanded,
                topk_indices: synth_i64(4),
                input: GemvInput::Raw(synth_f32(4)),
                out,
                k_top: 2,
                batch_size: 1,
            }],
            combine: if combine {
                vec![Step::MoeCombine {
                    down_out: out,
                    topk_weights: synth_f32(4),
                    out: synth_f32(4),
                    k: 2,
                    hidden: 4,
                    batch_size: 1,
                    inverse_perm: None,
                }]
            } else {
                Vec::new()
            },
            finish: Vec::new(),
        }
    }

    #[test]
    fn lowering_accepts_deferred_expanded_with_zero_combines() {
        // The explicit deferred marker admits the combine-less expanded
        // program (the next-layer fused consumer folds the partial); the
        // Single schedule carries all three routed steps.
        let group = group(ExpertParallelism::Single, 1);
        let policy = MoEExecutionPolicy::single();
        let lowered = lower_moe_steps(
            &group,
            &policy,
            deferred_parts(vec![expanded_rank(false, None, false)], true),
        )
        .unwrap();
        assert_eq!(lowered.executor_kind(), MoeExecutorKind::SingleMesh);
        assert_eq!(lowered.step_count(0), Some(3));
    }

    #[test]
    fn lowering_rejects_zero_combine_expanded_without_defer() {
        // Ordinary programs keep the exact-one-combine contract: a missing
        // combine is still CombineCountMismatch, never silently deferred.
        let group = group(ExpertParallelism::Single, 1);
        let policy = MoEExecutionPolicy::single();
        let err = lower_moe_steps(
            &group,
            &policy,
            parts(vec![expanded_rank(false, None, false)]),
        )
        .unwrap_err();
        assert!(matches!(
            err,
            MoeLowerError::CombineCountMismatch {
                expected: 1,
                actual: 0,
                ..
            }
        ));
    }

    #[test]
    fn lowering_rejects_deferred_expanded_with_local_combine() {
        // A deferred producer carrying its own combine would double-add (the
        // next-layer fused consumer folds the partial again).
        let group = group(ExpertParallelism::Single, 1);
        let policy = MoEExecutionPolicy::single();
        let err = lower_moe_steps(
            &group,
            &policy,
            deferred_parts(vec![expanded_rank(true, None, false)], true),
        )
        .unwrap_err();
        assert!(matches!(
            err,
            MoeLowerError::CombineCountMismatch {
                expected: 0,
                actual: 1,
                ..
            }
        ));
    }

    #[test]
    fn lowering_rejects_deferred_expanded_on_parallel_axis() {
        // The deferred protocol is rank-local: the fused consumer folds the
        // partial on the same device, so Tp/Ep has no combine anchor.
        let group = group(ExpertParallelism::TensorParallel, 2);
        let policy = tp_policy(2);
        let err = lower_moe_steps(
            &group,
            &policy,
            deferred_parts(
                vec![
                    expanded_rank(false, None, false),
                    expanded_rank(false, None, false),
                ],
                true,
            ),
        )
        .unwrap_err();
        assert!(matches!(
            err,
            MoeLowerError::DeferredCombineOnParallelAxis { .. }
        ));
        let display = err.to_string();
        assert!(display.contains("test"));
        assert!(display.contains("Some(0)"));
    }

    #[test]
    fn lowering_accepts_indexed_paro_decode_givens() {
        // Decode Paro: the router-phase Givens pre-rotation paired with the
        // specialized `MoeGateUpIndexed` gate-up lowers as an indexed
        // expanded program.
        let group = group(ExpertParallelism::Single, 1);
        let policy = MoEExecutionPolicy::single();
        let lowered = lower_moe_steps(
            &group,
            &policy,
            parts(vec![expanded_rank(true, Some("router"), true)]),
        )
        .unwrap();
        assert_eq!(lowered.executor_kind(), MoeExecutorKind::SingleMesh);
    }

    #[test]
    fn lowering_accepts_indexed_paro_prefill_givens() {
        // Indexed prefill (Path2-disabled Paro): the gate_up-phase Givens
        // preamble paired with `MoeGateUpIndexed` also lowers.
        let group = group(ExpertParallelism::Single, 1);
        let policy = MoEExecutionPolicy::single();
        let lowered = lower_moe_steps(
            &group,
            &policy,
            parts(vec![expanded_rank(true, Some("gate_up"), true)]),
        )
        .unwrap();
        assert_eq!(lowered.executor_kind(), MoeExecutorKind::SingleMesh);
    }

    #[test]
    fn lowering_rejects_router_givens_without_specialized_gate_up() {
        // The decode router-phase Givens is only admitted when the
        // specialized indexed gate-up accompanies it; a plain indexed gate-up
        // keeps the Givens classified as stray grouped machinery.
        let group = group(ExpertParallelism::Single, 1);
        let policy = MoEExecutionPolicy::single();
        let err = lower_moe_steps(
            &group,
            &policy,
            parts(vec![expanded_rank(true, Some("router"), false)]),
        )
        .unwrap_err();
        assert!(matches!(
            err,
            MoeLowerError::StrayGroupedOp { op: "givens", .. }
        ));
    }

    #[test]
    fn lowering_rejects_relocated_givens_even_with_specialized_gate_up() {
        // The Givens must sit in the decode router phase or the indexed
        // prefill gate_up phase; the specialized gate-up does not license an
        // activation-phase Givens.
        let group = group(ExpertParallelism::Single, 1);
        let policy = MoEExecutionPolicy::single();
        let err = lower_moe_steps(
            &group,
            &policy,
            parts(vec![expanded_rank(true, Some("activation"), true)]),
        )
        .unwrap_err();
        assert!(matches!(
            err,
            MoeLowerError::StrayGroupedOp { op: "givens", .. }
        ));
    }

    #[test]
    fn lowering_uses_logical_shape_for_padded_self_combining_f32_dim() {
        let group = group(ExpertParallelism::ExpertParallel, 2);
        let policy = ep_policy(2);
        let rank = || {
            let out = synth_with_bytes(DType::F32, 8, 72);
            RoutedMoePhases {
                router: Vec::new(),
                gate_up: vec![gate_up_step()],
                activation: vec![activation_step()],
                down: vec![Step::IndexedMoeGemv {
                    experts: expert_ref(),
                    which: MoeProj::DownResidual {
                        topk_weights: synth_f32(4),
                    },
                    topk_indices: synth_i64(4),
                    input: GemvInput::Raw(synth_f32(4)),
                    out,
                    k_top: 2,
                    batch_size: 1,
                }],
                combine: Vec::new(),
                finish: Vec::new(),
            }
        };
        let lowered = lower_moe_steps(&group, &policy, parts(vec![rank(), rank()])).unwrap();
        let LoweredMoeProgramInner::Parallel { collectives, .. } = lowered.inner else {
            panic!("expected parallel program");
        };
        // The padded 72-byte allocation still means 8 logical elements.
        assert!(matches!(
            collectives[2],
            StepCollective::AllReduce {
                kind: DimKind::Ep,
                dim: 8
            }
        ));
        assert_eq!(collective_count(&collectives), 1);
    }

    #[test]
    fn lowering_rejects_invalid_self_combining_f32_shape_and_capacity() {
        let group = group(ExpertParallelism::ExpertParallel, 2);
        let policy = ep_policy(2);
        let rank = |shape: Vec<usize>, capacity: usize, bytes: usize| {
            let out = Box::leak(Box::new(GpuTensor {
                buf: unsafe {
                    DeviceBuffer::from_raw(
                        Box::leak(vec![0u8; bytes].into_boxed_slice())
                            .as_mut_ptr()
                            .cast(),
                        capacity,
                    )
                },
                shape,
                dtype: DType::F32,
            }));
            RoutedMoePhases {
                router: Vec::new(),
                gate_up: vec![gate_up_step()],
                activation: vec![activation_step()],
                down: vec![Step::IndexedMoeGemv {
                    experts: expert_ref(),
                    which: MoeProj::DownResidual {
                        topk_weights: synth_f32(4),
                    },
                    topk_indices: synth_i64(4),
                    input: GemvInput::Raw(synth_f32(4)),
                    out,
                    k_top: 2,
                    batch_size: 1,
                }],
                combine: Vec::new(),
                finish: Vec::new(),
            }
        };
        // Zero logical shape.
        let err = lower_moe_steps(
            &group,
            &policy,
            parts(vec![rank(vec![0], 16, 16), rank(vec![0], 16, 16)]),
        )
        .unwrap_err();
        assert!(matches!(err, MoeLowerError::InvalidF32Shape { .. }));

        // Overflows the logical shape product.
        let err = lower_moe_steps(
            &group,
            &policy,
            parts(vec![
                rank(vec![usize::MAX, 2], 64, 64),
                rank(vec![usize::MAX, 2], 64, 64),
            ]),
        )
        .unwrap_err();
        assert!(matches!(err, MoeLowerError::ArithmeticOverflow { .. }));

        // Insufficient capacity: 8 logical elements need 32 bytes, not 16.
        let err = lower_moe_steps(
            &group,
            &policy,
            parts(vec![rank(vec![8], 16, 16), rank(vec![8], 16, 16)]),
        )
        .unwrap_err();
        assert!(matches!(err, MoeLowerError::CapacityMismatch { .. }));
    }

    #[test]
    fn lowering_reports_rank_one_in_malformed_second_rank_error() {
        let group = group(ExpertParallelism::TensorParallel, 2);
        let policy = tp_policy(2);
        // Rank 0 is a valid i64 rank; rank 1 has identical phase lengths but
        // an unrecognized down step, so its parse error must name rank 1.
        let malformed = || RoutedMoePhases {
            router: Vec::new(),
            gate_up: vec![gate_up_step()],
            activation: vec![activation_step()],
            down: vec![gate_up_step()],
            combine: Vec::new(),
            finish: vec![convert_step(synth_i64(4), 4)],
        };
        let err =
            lower_moe_steps(&group, &policy, parts(vec![i64_rank(), malformed()])).unwrap_err();
        assert!(matches!(err, MoeLowerError::UnrecognizedDownStep { .. }));
        assert!(err.to_string().contains("rank 1"));
    }

    #[test]
    fn lowering_rejects_non_combine_step_in_combine_phase() {
        let group = group(ExpertParallelism::TensorParallel, 2);
        let policy = tp_policy(2);
        let rank = || {
            let out = synth_f32(4);
            RoutedMoePhases {
                router: Vec::new(),
                gate_up: vec![gate_up_step()],
                activation: vec![activation_step()],
                down: vec![Step::IndexedMoeGemv {
                    experts: expert_ref(),
                    which: MoeProj::DownExpanded,
                    topk_indices: synth_i64(4),
                    input: GemvInput::Raw(synth_f32(4)),
                    out,
                    k_top: 2,
                    batch_size: 1,
                }],
                combine: vec![activation_step()],
                finish: Vec::new(),
            }
        };
        let err = lower_moe_steps(&group, &policy, parts(vec![rank(), rank()])).unwrap_err();
        assert!(matches!(err, MoeLowerError::UnrecognizedCombineStep { .. }));
        let display = err.to_string();
        assert!(display.contains("test"));
        assert!(display.contains("Some(0)"));
    }

    #[test]
    fn lowering_rejects_expanded_combine_batch_size_mismatch() {
        let group = group(ExpertParallelism::TensorParallel, 2);
        let policy = tp_policy(2);
        let rank = || {
            let out = synth_f32(4);
            RoutedMoePhases {
                router: Vec::new(),
                gate_up: vec![gate_up_step()],
                activation: vec![activation_step()],
                down: vec![Step::IndexedMoeGemv {
                    experts: expert_ref(),
                    which: MoeProj::DownExpanded,
                    topk_indices: synth_i64(4),
                    input: GemvInput::Raw(synth_f32(4)),
                    out,
                    k_top: 2,
                    batch_size: 1,
                }],
                // The combine declares batch_size=4 while the producer ran 1.
                combine: vec![Step::MoeCombine {
                    down_out: out,
                    topk_weights: synth_f32(4),
                    out: synth_f32(16),
                    k: 2,
                    hidden: 4,
                    batch_size: 4,
                    inverse_perm: None,
                }],
                finish: Vec::new(),
            }
        };
        let err = lower_moe_steps(&group, &policy, parts(vec![rank(), rank()])).unwrap_err();
        assert!(matches!(
            err,
            MoeLowerError::CombineMetadataMismatch {
                field: "batch_size",
                expected: 1,
                actual: 4,
                ..
            }
        ));
    }

    #[test]
    fn lowering_rejects_expanded_combine_k_mismatch() {
        let group = group(ExpertParallelism::TensorParallel, 2);
        let policy = tp_policy(2);
        let rank = || {
            let out = synth_f32(4);
            RoutedMoePhases {
                router: Vec::new(),
                gate_up: vec![gate_up_step()],
                activation: vec![activation_step()],
                down: vec![Step::IndexedMoeGemv {
                    experts: expert_ref(),
                    which: MoeProj::DownExpanded,
                    topk_indices: synth_i64(4),
                    input: GemvInput::Raw(synth_f32(4)),
                    out,
                    k_top: 2,
                    batch_size: 1,
                }],
                // The combine folds k=3 while the producer ran k_top=2.
                combine: vec![Step::MoeCombine {
                    down_out: out,
                    topk_weights: synth_f32(4),
                    out: synth_f32(4),
                    k: 3,
                    hidden: 4,
                    batch_size: 1,
                    inverse_perm: None,
                }],
                finish: Vec::new(),
            }
        };
        let err = lower_moe_steps(&group, &policy, parts(vec![rank(), rank()])).unwrap_err();
        assert!(matches!(
            err,
            MoeLowerError::CombineMetadataMismatch {
                field: "k",
                expected: 2,
                actual: 3,
                ..
            }
        ));
    }

    #[test]
    fn lowering_rejects_undersized_expanded_combine_output_capacity() {
        let group = group(ExpertParallelism::TensorParallel, 2);
        let policy = tp_policy(2);
        let rank = || {
            let out = synth_f32(4);
            RoutedMoePhases {
                router: Vec::new(),
                gate_up: vec![gate_up_step()],
                activation: vec![activation_step()],
                down: vec![Step::IndexedMoeGemv {
                    experts: expert_ref(),
                    which: MoeProj::DownExpanded,
                    topk_indices: synth_i64(4),
                    input: GemvInput::Raw(synth_f32(4)),
                    out,
                    k_top: 2,
                    batch_size: 1,
                }],
                // dim = hidden*batch_size = 4 needs 16 bytes; the output has
                // the right 4-element shape but only 8 bytes of capacity.
                combine: vec![Step::MoeCombine {
                    down_out: out,
                    topk_weights: synth_f32(4),
                    out: synth_with_bytes(DType::F32, 4, 8),
                    k: 2,
                    hidden: 4,
                    batch_size: 1,
                    inverse_perm: None,
                }],
                finish: Vec::new(),
            }
        };
        let err = lower_moe_steps(&group, &policy, parts(vec![rank(), rank()])).unwrap_err();
        assert!(matches!(err, MoeLowerError::CapacityMismatch { .. }));
    }

    #[test]
    fn lowering_rejects_expanded_combine_output_shape_mismatch() {
        let group = group(ExpertParallelism::TensorParallel, 2);
        let policy = tp_policy(2);
        let rank = || {
            let out = synth_f32(4);
            RoutedMoePhases {
                router: Vec::new(),
                gate_up: vec![gate_up_step()],
                activation: vec![activation_step()],
                down: vec![Step::IndexedMoeGemv {
                    experts: expert_ref(),
                    which: MoeProj::DownExpanded,
                    topk_indices: synth_i64(4),
                    input: GemvInput::Raw(synth_f32(4)),
                    out,
                    k_top: 2,
                    batch_size: 1,
                }],
                // dim = hidden*batch_size = 4, but the output claims 8 elements.
                combine: vec![Step::MoeCombine {
                    down_out: out,
                    topk_weights: synth_f32(4),
                    out: synth_f32(8),
                    k: 2,
                    hidden: 4,
                    batch_size: 1,
                    inverse_perm: None,
                }],
                finish: Vec::new(),
            }
        };
        let err = lower_moe_steps(&group, &policy, parts(vec![rank(), rank()])).unwrap_err();
        assert!(matches!(
            err,
            MoeLowerError::CombineOutputShapeMismatch {
                expected: 4,
                actual: 8,
                ..
            }
        ));
    }

    /// Frozen grouped production shape: batch_size=4, k_top=8,
    /// total_slots=32, n_experts=8, block_m=16, m_total_max/m_total=160
    /// (N*8 + 8*16), intermediate width inter=4, hidden width=4. Binding
    /// grammar: router exactly [MoeScatter]; gate_up
    /// [GivensRotateBatched?, GroupedMoeGemm(GateUp), MoeGateUpUnscatter];
    /// activation [MoeActivation]; down [GroupedMoeGemm(DownExpanded)];
    /// combine [MoeCombine(inverse_perm)]; finish []. Every buffer is sized
    /// TRUTHFULLY from the kernel/launcher contracts (see the capacity
    /// checks): scatter topk [slots]i32, counts [E]i32, offsets [E+1]i32,
    /// sorted [m_total_max]i32, tiles [m_total_max/16]i32, perm [slots]i32;
    /// gate-up y [m_total × 2*inter]f32; unscatter gate/up [slots × inter];
    /// down y [m_total × hidden]f32; combine out [batch × hidden]f32. The
    /// grouped GateUp.up_out is UNUSED metadata and holds an unrelated
    /// sentinel. Chain buffers alias as production: unscatter y_grouped ==
    /// gate-up y; unscatter gate/up == activation gate/up; activation
    /// rot_out == down x; down y == combine down_out; scatter inverse_perm
    /// == combine inverse_perm.
    fn grouped_phases(
        sorted: &'static GpuTensor,
        tiles: &'static GpuTensor,
        perm: &'static GpuTensor,
        combine_perm: Option<&'static GpuTensor>,
    ) -> RoutedMoeStepPhases<'static> {
        let y_gate_up = synth_with_bytes(DType::F32, 160 * 8, 160 * 8 * 4);
        let y_down = synth_with_bytes(DType::F32, 160 * 128, 160 * 128 * 4);
        let experts = lloyd_expert_ref_with(8, 4, 128, DType::F32);
        let x_rot = synth_with_bytes(DType::F32, 4 * 128, 4 * 128 * 4);
        let gate_batch = synth_with_bytes(DType::F32, 32 * 4, 32 * 4 * 4);
        let up_batch = synth_with_bytes(DType::F32, 32 * 4, 32 * 4 * 4);
        let rot = synth_with_bytes(DType::F32, 32 * 4, 32 * 4 * 4);
        RoutedMoePhases {
            router: vec![Step::MoeScatter {
                topk_indices: synth_i32(32),
                expert_token_counts: synth_i32(8),
                expert_offsets: synth_i32(9),
                sorted_slot_index: sorted,
                expert_tile_ids: tiles,
                inverse_perm: perm,
                total_slots: 32,
                n_experts: 8,
                m_total_max: 160,
                block_m: 16,
            }],
            gate_up: vec![
                Step::GroupedMoeGemm {
                    experts,
                    // up_out is UNUSED metadata on the grouped path: an
                    // unrelated sentinel must be accepted.
                    which: MoeProj::GateUp {
                        up_out: synth_f32(4),
                    },
                    sorted_slot_index: sorted,
                    expert_tile_ids: tiles,
                    x: x_rot,
                    y: y_gate_up,
                    m_total: 160,
                    batch_size: 4,
                    k_top: 8,
                    dtype_tags: None,
                    force_mq4_fp16: false,
                    paro_i8: false,
                    paro_i8_k8: false,
                },
                Step::MoeGateUpUnscatter {
                    y_grouped: y_gate_up,
                    sorted_slot_index: sorted,
                    gate_batch,
                    up_batch,
                    inter: 4,
                    k_top: 8,
                    m_total: 160,
                },
            ],
            activation: vec![Step::MoeActivation {
                variant: MoeActivationVariant::QwenAwqIndexed {
                    awq_ptrs: synth_f32(4),
                    topk_indices: synth_i64(4),
                },
                gate: gate_batch,
                up: up_batch,
                rot_out: rot,
                inter: 4,
                k_top: 32,
            }],
            down: vec![Step::GroupedMoeGemm {
                experts,
                which: MoeProj::DownExpanded,
                sorted_slot_index: sorted,
                expert_tile_ids: tiles,
                x: rot,
                y: y_down,
                m_total: 160,
                batch_size: 4,
                k_top: 8,
                dtype_tags: None,
                force_mq4_fp16: false,
                paro_i8: false,
                paro_i8_k8: false,
            }],
            combine: vec![Step::MoeCombine {
                down_out: y_down,
                topk_weights: synth_f32(32),
                out: synth_f32(4 * 128),
                k: 8,
                hidden: 128,
                batch_size: 4,
                inverse_perm: combine_perm,
            }],
            finish: Vec::new(),
        }
    }

    /// Paro variant of the grouped fixture: the gate_up phase opens with the
    /// typed Givens rotation whose `out` aliases the grouped gate-up input.
    /// The sidecar tensors follow the real kernel contract
    /// (`givens_rotate_f32`): pairs [krot × dim] i16, theta [krot × dim/2]
    /// f16, scales [dim] f16; input/output [batch × dim] f32.
    fn grouped_phases_paro(
        sorted: &'static GpuTensor,
        tiles: &'static GpuTensor,
        perm: &'static GpuTensor,
        combine_perm: Option<&'static GpuTensor>,
    ) -> RoutedMoeStepPhases<'static> {
        let mut phases = grouped_phases(sorted, tiles, perm, combine_perm);
        let Step::GroupedMoeGemm { x, .. } = &mut phases.gate_up[0] else {
            unreachable!()
        };
        let x_rot = *x;
        phases.gate_up.insert(
            0,
            Step::GivensRotateBatched {
                x: synth_with_bytes(DType::F32, 4 * 128, 4 * 128 * 4),
                out: x_rot,
                pairs: synth_with_bytes(DType::Raw, 4 * 128, 4 * 128 * 2),
                theta: synth_with_bytes(DType::Raw, 4 * 64, 4 * 64 * 2),
                scales: synth_with_bytes(DType::Raw, 128, 128 * 2),
                batch: 4,
                dim: 128,
                krot: 4,
            },
        );
        phases
    }

    /// Group with BOTH indexed and grouped identities declared, plus TP
    /// collective authority: the grouped concrete protocol must resolve to
    /// GroupedQuantized even when IndexedQuantized is also declared.
    fn grouped_group() -> ExpertGroupPlan {
        let mut group = group(ExpertParallelism::TensorParallel, 2);
        group.allowed_executions = vec![
            ExpertExecutionIdentity::IndexedQuantized,
            ExpertExecutionIdentity::GroupedQuantized,
        ];
        group
    }

    /// `parts` with the concrete execution plan fixed to GroupedQuantized.
    fn grouped_parts(ranks: Vec<RoutedMoeStepPhases<'static>>) -> MoeProgramParts<'static> {
        let mut parts = parts(ranks);
        parts.execution = ExpertExecutionPlan::GroupedQuantized;
        parts
    }

    fn grouped_fixture() -> (&'static GpuTensor, &'static GpuTensor, &'static GpuTensor) {
        (
            synth_with_bytes(DType::Raw, 160, 160 * 4),
            synth_with_bytes(DType::Raw, 10, 10 * 4),
            synth_with_bytes(DType::Raw, 32, 32 * 4),
        )
    }

    #[test]
    fn lowering_derives_batched_grouped_expanded_lowering() {
        let group = grouped_group();
        let policy = tp_policy(2);
        let (sorted, tiles, perm) = grouped_fixture();
        let lowered = lower_moe_steps(
            &group,
            &policy,
            grouped_parts(vec![
                grouped_phases(sorted, tiles, perm, Some(perm)),
                grouped_phases(sorted, tiles, perm, Some(perm)),
            ]),
        )
        .unwrap();
        let LoweredMoeProgramInner::Parallel { collectives, .. } = lowered.inner else {
            panic!("expected parallel program");
        };
        // gate_up phase [scatter, grouped gate-up, unscatter] + activation +
        // down: the combine sits at absolute index 5.
        assert!(matches!(
            collectives[5],
            StepCollective::AllReduce {
                kind: DimKind::Tp,
                dim: 512
            }
        ));
        assert_eq!(collective_count(&collectives), 1);
    }

    #[test]
    fn lowering_rejects_grouped_scatter_total_slots_mismatch() {
        let group = grouped_group();
        let policy = tp_policy(2);
        let (sorted, tiles, perm) = grouped_fixture();
        let rank = || {
            let mut phases = grouped_phases(sorted, tiles, perm, Some(perm));
            let Step::MoeScatter { total_slots, .. } = &mut phases.router[0] else {
                unreachable!()
            };
            *total_slots = 33;
            phases
        };
        let err =
            lower_moe_steps(&group, &policy, grouped_parts(vec![rank(), rank()])).unwrap_err();
        assert!(matches!(
            err,
            MoeLowerError::ScatterSlotCountMismatch {
                expected: 32,
                actual: 33,
                ..
            }
        ));

        // Overflowing slot arithmetic is contextual overflow, never a panic.
        // batch=MAX passes the zero/range gates and trips the checked
        // batch*k product (via the combine preamble's checked dim first).
        let rank = || {
            let mut phases = grouped_phases(sorted, tiles, perm, Some(perm));
            let Step::GroupedMoeGemm { batch_size, .. } = &mut phases.gate_up[0] else {
                unreachable!()
            };
            *batch_size = usize::MAX;
            let Step::GroupedMoeGemm { batch_size, .. } = &mut phases.down[0] else {
                unreachable!()
            };
            *batch_size = usize::MAX;
            let Step::MoeCombine { batch_size, .. } = &mut phases.combine[0] else {
                unreachable!()
            };
            *batch_size = usize::MAX;
            phases
        };
        let err =
            lower_moe_steps(&group, &policy, grouped_parts(vec![rank(), rank()])).unwrap_err();
        assert!(matches!(err, MoeLowerError::ArithmeticOverflow { .. }));
    }

    #[test]
    fn lowering_rejects_grouped_block_mismatch() {
        let group = grouped_group();
        let policy = tp_policy(2);
        let (sorted, tiles, perm) = grouped_fixture();
        let rank = || {
            let mut phases = grouped_phases(sorted, tiles, perm, Some(perm));
            let Step::MoeScatter { block_m, .. } = &mut phases.router[0] else {
                unreachable!()
            };
            *block_m = 8;
            phases
        };
        let err =
            lower_moe_steps(&group, &policy, grouped_parts(vec![rank(), rank()])).unwrap_err();
        assert!(matches!(
            err,
            MoeLowerError::ScatterBlockMismatch {
                expected: 16,
                actual: 8,
                ..
            }
        ));
    }

    #[test]
    fn lowering_rejects_grouped_expert_count_mismatch() {
        let group = grouped_group();
        let policy = tp_policy(2);
        let (sorted, tiles, perm) = grouped_fixture();
        let rank = || {
            let mut phases = grouped_phases(sorted, tiles, perm, Some(perm));
            let Step::MoeScatter { n_experts, .. } = &mut phases.router[0] else {
                unreachable!()
            };
            *n_experts = 4;
            phases
        };
        let err =
            lower_moe_steps(&group, &policy, grouped_parts(vec![rank(), rank()])).unwrap_err();
        assert!(matches!(
            err,
            MoeLowerError::ScatterExpertCountMismatch {
                expected: 4,
                actual: 8,
                ..
            }
        ));
    }

    #[test]
    fn lowering_rejects_grouped_m_total_mismatch() {
        let group = grouped_group();
        let policy = tp_policy(2);
        let (sorted, tiles, perm) = grouped_fixture();
        for (field, index, value) in [
            ("down", 0usize, 200usize),
            ("down", 0, 120),
            ("gate_up", 1, 200),
            ("gate_up_unscatter", 1, 200),
        ] {
            let rank = || {
                let mut phases = grouped_phases(sorted, tiles, perm, Some(perm));
                if field == "down" {
                    let Step::GroupedMoeGemm { m_total, .. } = &mut phases.down[0] else {
                        unreachable!()
                    };
                    *m_total = value;
                } else if field == "gate_up" {
                    let Step::GroupedMoeGemm { m_total, .. } = &mut phases.gate_up[0] else {
                        unreachable!()
                    };
                    *m_total = value;
                } else {
                    let Step::MoeGateUpUnscatter { m_total, .. } = &mut phases.gate_up[1] else {
                        unreachable!()
                    };
                    *m_total = value;
                }
                let _ = index;
                phases
            };
            let err =
                lower_moe_steps(&group, &policy, grouped_parts(vec![rank(), rank()])).unwrap_err();
            assert!(
                matches!(
                    err,
                    MoeLowerError::ScatterMaxTotalMismatch {
                        field: f,
                        expected: 160,
                        actual: a,
                        ..
                    } if f == field && a == value
                ),
                "{field} m_total mismatch not reported"
            );
        }
    }

    #[test]
    fn lowering_rejects_grouped_gate_up_chain_mismatch() {
        let group = grouped_group();
        let policy = tp_policy(2);
        let (sorted, tiles, perm) = grouped_fixture();
        // The grouped gate-up reads a different sort table than the scatter.
        let rank = || {
            let mut phases = grouped_phases(sorted, tiles, perm, Some(perm));
            let Step::GroupedMoeGemm {
                sorted_slot_index,
                expert_tile_ids,
                ..
            } = &mut phases.gate_up[0]
            else {
                unreachable!()
            };
            *sorted_slot_index = synth_i64(32);
            *expert_tile_ids = synth_i64(8);
            phases
        };
        let err =
            lower_moe_steps(&group, &policy, grouped_parts(vec![rank(), rank()])).unwrap_err();
        assert!(matches!(
            err,
            MoeLowerError::ScatterChainMismatch {
                field: "sorted_slot_index",
                ..
            }
        ));

        // Distinct tile ids only.
        let rank = || {
            let mut phases = grouped_phases(sorted, tiles, perm, Some(perm));
            let Step::GroupedMoeGemm {
                expert_tile_ids, ..
            } = &mut phases.gate_up[0]
            else {
                unreachable!()
            };
            *expert_tile_ids = synth_i64(8);
            phases
        };
        let err =
            lower_moe_steps(&group, &policy, grouped_parts(vec![rank(), rank()])).unwrap_err();
        assert!(matches!(
            err,
            MoeLowerError::ScatterChainMismatch {
                field: "expert_tile_ids",
                ..
            }
        ));
    }

    #[test]
    fn lowering_rejects_grouped_unscatter_chain_mismatch() {
        let group = grouped_group();
        let policy = tp_policy(2);
        let (sorted, tiles, perm) = grouped_fixture();
        // The unscatter reads a different sort table than the scatter.
        let rank = || {
            let mut phases = grouped_phases(sorted, tiles, perm, Some(perm));
            let Step::MoeGateUpUnscatter {
                sorted_slot_index, ..
            } = &mut phases.gate_up[1]
            else {
                unreachable!()
            };
            *sorted_slot_index = synth_i64(32);
            phases
        };
        let err =
            lower_moe_steps(&group, &policy, grouped_parts(vec![rank(), rank()])).unwrap_err();
        assert!(matches!(
            err,
            MoeLowerError::ScatterChainMismatch {
                field: "sorted_slot_index",
                ..
            }
        ));

        // The unscatter consumes a different y_grouped than the gate-up wrote.
        let rank = || {
            let mut phases = grouped_phases(sorted, tiles, perm, Some(perm));
            let Step::MoeGateUpUnscatter { y_grouped, .. } = &mut phases.gate_up[1] else {
                unreachable!()
            };
            *y_grouped = synth_f32(160);
            phases
        };
        let err =
            lower_moe_steps(&group, &policy, grouped_parts(vec![rank(), rank()])).unwrap_err();
        assert!(matches!(
            err,
            MoeLowerError::GroupedChainMismatch {
                field: "y_grouped",
                ..
            }
        ));
    }

    #[test]
    fn lowering_rejects_missing_or_duplicate_grouped_ops() {
        let group = grouped_group();
        let policy = tp_policy(2);
        let (sorted, tiles, perm) = grouped_fixture();
        // Missing grouped gate-up.
        let rank = || {
            let mut phases = grouped_phases(sorted, tiles, perm, Some(perm));
            phases.gate_up.remove(0);
            phases
        };
        let err =
            lower_moe_steps(&group, &policy, grouped_parts(vec![rank(), rank()])).unwrap_err();
        assert!(matches!(
            err,
            MoeLowerError::GroupedOpCountMismatch {
                op: "gate_up",
                expected: 1,
                actual: 0,
                ..
            }
        ));

        // Missing unscatter.
        let rank = || {
            let mut phases = grouped_phases(sorted, tiles, perm, Some(perm));
            phases.gate_up.remove(1);
            phases
        };
        let err =
            lower_moe_steps(&group, &policy, grouped_parts(vec![rank(), rank()])).unwrap_err();
        assert!(matches!(
            err,
            MoeLowerError::GroupedOpCountMismatch {
                op: "gate_up_unscatter",
                expected: 1,
                actual: 0,
                ..
            }
        ));

        // Duplicate grouped gate-up.
        let rank = || {
            let mut phases = grouped_phases(sorted, tiles, perm, Some(perm));
            let experts = expert_ref_with(8);
            phases.gate_up.push(Step::GroupedMoeGemm {
                experts,
                which: MoeProj::GateUp {
                    up_out: synth_f32(4),
                },
                sorted_slot_index: sorted,
                expert_tile_ids: tiles,
                x: synth_f32(4),
                y: synth_f32(160),
                m_total: 160,
                batch_size: 4,
                k_top: 8,
                dtype_tags: None,
                force_mq4_fp16: false,
                paro_i8: false,
                paro_i8_k8: false,
            });
            phases
        };
        let err =
            lower_moe_steps(&group, &policy, grouped_parts(vec![rank(), rank()])).unwrap_err();
        assert!(matches!(
            err,
            MoeLowerError::GroupedOpCountMismatch {
                op: "gate_up",
                expected: 1,
                actual: 2,
                ..
            }
        ));

        // Duplicate unscatter.
        let rank = || {
            let mut phases = grouped_phases(sorted, tiles, perm, Some(perm));
            phases.gate_up.push(Step::MoeGateUpUnscatter {
                y_grouped: synth_f32(160),
                sorted_slot_index: sorted,
                gate_batch: synth_f32(4),
                up_batch: synth_f32(4),
                inter: 4,
                k_top: 8,
                m_total: 160,
            });
            phases
        };
        let err =
            lower_moe_steps(&group, &policy, grouped_parts(vec![rank(), rank()])).unwrap_err();
        assert!(matches!(
            err,
            MoeLowerError::GroupedOpCountMismatch {
                op: "gate_up_unscatter",
                expected: 1,
                actual: 2,
                ..
            }
        ));
    }

    #[test]
    fn lowering_rejects_grouped_gate_up_unscatter_before_gate_up() {
        let group = grouped_group();
        let policy = tp_policy(2);
        let (sorted, tiles, perm) = grouped_fixture();
        // The gate-up unscatter precedes the grouped gate-up in the gate_up
        // phase; the binding sequence is [givens?, gate-up, unscatter].
        let rank = || {
            let mut phases = grouped_phases(sorted, tiles, perm, Some(perm));
            phases.gate_up.swap(0, 1);
            phases
        };
        let err =
            lower_moe_steps(&group, &policy, grouped_parts(vec![rank(), rank()])).unwrap_err();
        assert!(matches!(
            err,
            MoeLowerError::GroupedPhaseMismatch {
                phase: "gate_up",
                index: 1,
                ..
            }
        ));
    }

    #[test]
    fn lowering_rejects_stray_scatter_in_indexed_expanded() {
        let group = group(ExpertParallelism::TensorParallel, 2);
        let policy = tp_policy(2);
        let rank = || {
            let out = synth_f32(4);
            RoutedMoePhases {
                router: Vec::new(),
                gate_up: vec![
                    gate_up_step(),
                    Step::MoeScatter {
                        topk_indices: synth_i64(4),
                        expert_token_counts: synth_i64(4),
                        expert_offsets: synth_i64(4),
                        sorted_slot_index: synth_i64(16),
                        expert_tile_ids: synth_i64(4),
                        inverse_perm: synth_i64(16),
                        total_slots: 16,
                        n_experts: 4,
                        m_total_max: 16,
                        block_m: 16,
                    },
                ],
                activation: vec![activation_step()],
                down: vec![Step::IndexedMoeGemv {
                    experts: expert_ref(),
                    which: MoeProj::DownExpanded,
                    topk_indices: synth_i64(4),
                    input: GemvInput::Raw(synth_f32(4)),
                    out,
                    k_top: 2,
                    batch_size: 1,
                }],
                combine: vec![Step::MoeCombine {
                    down_out: out,
                    topk_weights: synth_f32(4),
                    out: synth_f32(4),
                    k: 2,
                    hidden: 4,
                    batch_size: 1,
                    inverse_perm: None,
                }],
                finish: Vec::new(),
            }
        };
        let err = lower_moe_steps(&group, &policy, parts(vec![rank(), rank()])).unwrap_err();
        assert!(matches!(err, MoeLowerError::StrayScatter { .. }));
    }

    #[test]
    fn lowering_rejects_stray_scatter_in_self_combining() {
        let group = group(ExpertParallelism::TensorParallel, 2);
        let policy = tp_policy(2);
        // F32 self-combining with a stray scatter.
        let rank = || {
            let mut phases = down_residual_f32_rank();
            phases.activation.push(Step::MoeScatter {
                topk_indices: synth_i64(4),
                expert_token_counts: synth_i64(4),
                expert_offsets: synth_i64(4),
                sorted_slot_index: synth_i64(16),
                expert_tile_ids: synth_i64(4),
                inverse_perm: synth_i64(16),
                total_slots: 16,
                n_experts: 4,
                m_total_max: 16,
                block_m: 16,
            });
            phases
        };
        let err = lower_moe_steps(&group, &policy, parts(vec![rank(), rank()])).unwrap_err();
        assert!(matches!(err, MoeLowerError::StrayScatter { .. }));

        // I64 self-combining with a stray scatter.
        let rank = || {
            let mut phases = i64_rank();
            phases.activation.push(Step::MoeScatter {
                topk_indices: synth_i64(4),
                expert_token_counts: synth_i64(4),
                expert_offsets: synth_i64(4),
                sorted_slot_index: synth_i64(16),
                expert_tile_ids: synth_i64(4),
                inverse_perm: synth_i64(16),
                total_slots: 16,
                n_experts: 4,
                m_total_max: 16,
                block_m: 16,
            });
            phases
        };
        let err = lower_moe_steps(&group, &policy, parts(vec![rank(), rank()])).unwrap_err();
        assert!(matches!(err, MoeLowerError::StrayScatter { .. }));
    }

    #[test]
    fn lowering_rejects_indexed_expanded_with_inverse_perm() {
        let group = group(ExpertParallelism::TensorParallel, 2);
        let policy = tp_policy(2);
        let rank = || {
            let out = synth_f32(4);
            RoutedMoePhases {
                router: Vec::new(),
                gate_up: vec![gate_up_step()],
                activation: vec![activation_step()],
                down: vec![Step::IndexedMoeGemv {
                    experts: expert_ref(),
                    which: MoeProj::DownExpanded,
                    topk_indices: synth_i64(4),
                    input: GemvInput::Raw(synth_f32(4)),
                    out,
                    k_top: 2,
                    batch_size: 1,
                }],
                // Indexed expanded combines never take an inverse permutation.
                combine: vec![Step::MoeCombine {
                    down_out: out,
                    topk_weights: synth_f32(4),
                    out: synth_f32(4),
                    k: 2,
                    hidden: 4,
                    batch_size: 1,
                    inverse_perm: Some(synth_i64(4)),
                }],
                finish: Vec::new(),
            }
        };
        let err = lower_moe_steps(&group, &policy, parts(vec![rank(), rank()])).unwrap_err();
        assert!(matches!(
            err,
            MoeLowerError::UnexpectedCombineInversePerm { .. }
        ));
    }

    #[test]
    fn lowering_rejects_grouped_expanded_without_inverse_perm() {
        let group = group(ExpertParallelism::TensorParallel, 2);
        let policy = tp_policy(2);
        let (sorted, tiles, perm) = grouped_fixture();
        let err = lower_moe_steps(
            &group,
            &policy,
            parts(vec![
                grouped_phases(sorted, tiles, perm, None),
                grouped_phases(sorted, tiles, perm, None),
            ]),
        )
        .unwrap_err();
        assert!(matches!(
            err,
            MoeLowerError::MissingCombineInversePerm { .. }
        ));
    }

    #[test]
    fn lowering_rejects_grouped_combine_inverse_perm_chain_mismatch() {
        let group = grouped_group();
        let policy = tp_policy(2);
        let (sorted, tiles, perm) = grouped_fixture();
        let other_perm = synth_i64(32);
        let err = lower_moe_steps(
            &group,
            &policy,
            grouped_parts(vec![
                grouped_phases(sorted, tiles, perm, Some(other_perm)),
                grouped_phases(sorted, tiles, perm, Some(other_perm)),
            ]),
        )
        .unwrap_err();
        assert!(matches!(
            err,
            MoeLowerError::ScatterChainMismatch {
                field: "inverse_perm",
                ..
            }
        ));
    }

    // ── G1 grouped gate-up deinterleave protocol (Oracle quality) ───────────

    #[test]
    fn lowering_derives_grouped_paro_sequence() {
        // Paro: router [scatter], gate_up [Givens, grouped gate-up, gate-up
        // unscatter], activation, down, combine. The combine sits at absolute
        // index 6 (router 1 + gate_up 3 + activation 1 + down 1).
        let group = grouped_group();
        let policy = tp_policy(2);
        let (sorted, tiles, perm) = grouped_fixture();
        let lowered = lower_moe_steps(
            &group,
            &policy,
            grouped_parts(vec![
                grouped_phases_paro(sorted, tiles, perm, Some(perm)),
                grouped_phases_paro(sorted, tiles, perm, Some(perm)),
            ]),
        )
        .unwrap();
        let LoweredMoeProgramInner::Parallel { collectives, .. } = lowered.inner else {
            panic!("expected parallel program");
        };
        assert!(matches!(
            collectives[6],
            StepCollective::AllReduce {
                kind: DimKind::Tp,
                dim: 512
            }
        ));
        assert_eq!(collective_count(&collectives), 1);
    }

    #[test]
    fn lowering_rejects_grouped_chain_mislabeled_indexed_identity() {
        // A valid grouped concrete protocol MUST use GroupedQuantized even
        // when IndexedQuantized is also declared; the mislabeled program is
        // currently admitted because membership passes.
        let group = grouped_group();
        let policy = tp_policy(2);
        let (sorted, tiles, perm) = grouped_fixture();
        let err = lower_moe_steps(
            &group,
            &policy,
            parts(vec![
                grouped_phases(sorted, tiles, perm, Some(perm)),
                grouped_phases(sorted, tiles, perm, Some(perm)),
            ]),
        )
        .unwrap_err();
        assert!(matches!(
            err,
            MoeLowerError::ExecutionIdentityMismatch {
                expected,
                actual: "indexed_quantized",
                ..
            } if expected == "grouped_quantized"
        ));
    }

    #[test]
    fn lowering_rejects_grouped_gate_up_unscatter_in_wrong_phase() {
        let group = grouped_group();
        let policy = tp_policy(2);
        let (sorted, tiles, perm) = grouped_fixture();

        // In the router phase after the scatter.
        let rank = || {
            let mut phases = grouped_phases(sorted, tiles, perm, Some(perm));
            let us = phases.gate_up.remove(1);
            phases.router.push(us);
            phases
        };
        let err =
            lower_moe_steps(&group, &policy, grouped_parts(vec![rank(), rank()])).unwrap_err();
        assert!(matches!(
            err,
            MoeLowerError::GroupedPhaseMismatch {
                phase: "router",
                index: 0,
                ..
            }
        ));

        // In the activation phase after the activation.
        let rank = || {
            let mut phases = grouped_phases(sorted, tiles, perm, Some(perm));
            let us = phases.gate_up.remove(1);
            phases.activation.push(us);
            phases
        };
        let err =
            lower_moe_steps(&group, &policy, grouped_parts(vec![rank(), rank()])).unwrap_err();
        assert!(matches!(
            err,
            MoeLowerError::GroupedPhaseMismatch {
                phase: "gate_up",
                index: 1,
                ..
            }
        ));

        // In the finish phase.
        let rank = || {
            let mut phases = grouped_phases(sorted, tiles, perm, Some(perm));
            let us = phases.gate_up.remove(1);
            phases.finish.push(us);
            phases
        };
        let err =
            lower_moe_steps(&group, &policy, grouped_parts(vec![rank(), rank()])).unwrap_err();
        assert!(matches!(
            err,
            MoeLowerError::GroupedPhaseMismatch {
                phase: "gate_up",
                index: 1,
                ..
            }
        ));

        // In the down phase (the global down classification rejects the
        // two-step down phase first — same error before and after G1).
        let rank = || {
            let mut phases = grouped_phases(sorted, tiles, perm, Some(perm));
            let us = phases.gate_up.remove(1);
            phases.down.push(us);
            phases
        };
        let err =
            lower_moe_steps(&group, &policy, grouped_parts(vec![rank(), rank()])).unwrap_err();
        assert!(matches!(err, MoeLowerError::MultipleDownSteps { .. }));

        // In the combine phase (the combine-phase content check rejects it
        // first — same error before and after G1).
        let rank = || {
            let mut phases = grouped_phases(sorted, tiles, perm, Some(perm));
            let us = phases.gate_up.remove(1);
            phases.combine.push(us);
            phases
        };
        let err =
            lower_moe_steps(&group, &policy, grouped_parts(vec![rank(), rank()])).unwrap_err();
        assert!(matches!(err, MoeLowerError::UnrecognizedCombineStep { .. }));
    }

    #[test]
    fn lowering_rejects_grouped_chain_buffer_disconnects() {
        let group = grouped_group();
        let policy = tp_policy(2);
        let (sorted, tiles, perm) = grouped_fixture();

        // The activation consumes a different gate buffer.
        let rank = || {
            let mut phases = grouped_phases(sorted, tiles, perm, Some(perm));
            let Step::MoeActivation { gate, .. } = &mut phases.activation[0] else {
                unreachable!()
            };
            *gate = synth_f32(4);
            phases
        };
        let err =
            lower_moe_steps(&group, &policy, grouped_parts(vec![rank(), rank()])).unwrap_err();
        assert!(matches!(
            err,
            MoeLowerError::GroupedChainMismatch {
                field: "activation_gate",
                ..
            }
        ));

        // The activation consumes a different up buffer.
        let rank = || {
            let mut phases = grouped_phases(sorted, tiles, perm, Some(perm));
            let Step::MoeActivation { up, .. } = &mut phases.activation[0] else {
                unreachable!()
            };
            *up = synth_f32(4);
            phases
        };
        let err =
            lower_moe_steps(&group, &policy, grouped_parts(vec![rank(), rank()])).unwrap_err();
        assert!(matches!(
            err,
            MoeLowerError::GroupedChainMismatch {
                field: "activation_up",
                ..
            }
        ));

        // The down consumes a different rotated buffer than the activation
        // wrote.
        let rank = || {
            let mut phases = grouped_phases(sorted, tiles, perm, Some(perm));
            let Step::MoeActivation { rot_out, .. } = &mut phases.activation[0] else {
                unreachable!()
            };
            *rot_out = synth_f32(4);
            phases
        };
        let err =
            lower_moe_steps(&group, &policy, grouped_parts(vec![rank(), rank()])).unwrap_err();
        assert!(matches!(
            err,
            MoeLowerError::GroupedChainMismatch {
                field: "down_input",
                ..
            }
        ));
    }

    #[test]
    fn lowering_rejects_grouped_givens_disconnects() {
        let group = grouped_group();
        let policy = tp_policy(2);
        let (sorted, tiles, perm) = grouped_fixture();

        // The Givens output is not the grouped gate-up input.
        let rank = || {
            let mut phases = grouped_phases_paro(sorted, tiles, perm, Some(perm));
            let Step::GivensRotateBatched { out, .. } = &mut phases.gate_up[0] else {
                unreachable!()
            };
            *out = synth_f32(4);
            phases
        };
        let err =
            lower_moe_steps(&group, &policy, grouped_parts(vec![rank(), rank()])).unwrap_err();
        assert!(matches!(
            err,
            MoeLowerError::GroupedChainMismatch {
                field: "givens_out",
                ..
            }
        ));

        // The Givens batch disagrees with the grouped gate-up batch.
        let rank = || {
            let mut phases = grouped_phases_paro(sorted, tiles, perm, Some(perm));
            let Step::GivensRotateBatched { batch, .. } = &mut phases.gate_up[0] else {
                unreachable!()
            };
            *batch = 8;
            phases
        };
        let err =
            lower_moe_steps(&group, &policy, grouped_parts(vec![rank(), rank()])).unwrap_err();
        assert!(matches!(
            err,
            MoeLowerError::GroupedChainMismatch {
                field: "givens_batch",
                ..
            }
        ));

        // The Givens dim disagrees with the grouped gate-up input width.
        let rank = || {
            let mut phases = grouped_phases_paro(sorted, tiles, perm, Some(perm));
            let Step::GivensRotateBatched { dim, .. } = &mut phases.gate_up[0] else {
                unreachable!()
            };
            *dim = 256;
            phases
        };
        let err =
            lower_moe_steps(&group, &policy, grouped_parts(vec![rank(), rank()])).unwrap_err();
        assert!(matches!(
            err,
            MoeLowerError::GroupedChainMismatch {
                field: "givens_dim",
                ..
            }
        ));
    }

    #[test]
    fn lowering_rejects_grouped_scalar_mismatches() {
        let group = grouped_group();
        let policy = tp_policy(2);
        let (sorted, tiles, perm) = grouped_fixture();

        // The unscatter inter disagrees with the expert contraction width.
        let rank = || {
            let mut phases = grouped_phases(sorted, tiles, perm, Some(perm));
            let Step::MoeGateUpUnscatter { inter, .. } = &mut phases.gate_up[1] else {
                unreachable!()
            };
            *inter = 8;
            phases
        };
        let err =
            lower_moe_steps(&group, &policy, grouped_parts(vec![rank(), rank()])).unwrap_err();
        assert!(matches!(
            err,
            MoeLowerError::GroupedChainMismatch { field: "inter", .. }
        ));

        // The activation rows disagree with the checked slot count.
        let rank = || {
            let mut phases = grouped_phases(sorted, tiles, perm, Some(perm));
            let Step::MoeActivation { k_top, .. } = &mut phases.activation[0] else {
                unreachable!()
            };
            *k_top = 31;
            phases
        };
        let err =
            lower_moe_steps(&group, &policy, grouped_parts(vec![rank(), rank()])).unwrap_err();
        assert!(matches!(
            err,
            MoeLowerError::GroupedChainMismatch {
                field: "activation_rows",
                ..
            }
        ));

        // The combine hidden disagrees with the grouped down output width
        // (the combine output grows to stay shape-coherent, so the hidden
        // check is what fires).
        let rank = || {
            let mut phases = grouped_phases(sorted, tiles, perm, Some(perm));
            let Step::MoeCombine {
                hidden,
                out,
                batch_size,
                ..
            } = &mut phases.combine[0]
            else {
                unreachable!()
            };
            *hidden = 8;
            *out = synth_f32(32);
            *batch_size = 4;
            phases
        };
        let err =
            lower_moe_steps(&group, &policy, grouped_parts(vec![rank(), rank()])).unwrap_err();
        assert!(matches!(
            err,
            MoeLowerError::CombineMetadataMismatch {
                field: "hidden",
                ..
            }
        ));
    }

    // ── Typed grouped cross-rank signature (Oracle quality) ───────────────

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    enum GroupedActivationKind {
        MinimaxFused,
        Ds4ClampRotate,
        QwenAwqIndexed,
        QwenParo,
    }

    /// Coherent per-rank grouped program parameterized by the signature
    /// scalars; every buffer is sized truthfully from the launch contracts.
    #[derive(Clone, Copy)]
    struct GroupedSigParams {
        batch_size: usize,
        k_top: usize,
        n_experts: usize,
        inter: usize,
        hidden: usize,
        m_total: usize,
        givens: bool,
        givens_krot: usize,
        givens_dim: usize,
        gu_dtype: DType,
        down_dtype: DType,
        gu_tags: bool,
        down_tags: bool,
        gu_force_mq4_fp16: bool,
        gu_paro_i8: bool,
        gu_paro_i8_k8: bool,
        dn_force_mq4_fp16: bool,
        dn_paro_i8: bool,
        dn_paro_i8_k8: bool,
        activation: GroupedActivationKind,
        activation_awq: bool,
        activation_swiglu_limit: f32,
        activation_krot: usize,
    }

    impl Default for GroupedSigParams {
        fn default() -> Self {
            Self {
                batch_size: 4,
                k_top: 8,
                n_experts: 8,
                inter: 4,
                hidden: 128,
                m_total: 160,
                givens: false,
                givens_krot: 4,
                givens_dim: 128,
                gu_dtype: DType::F32,
                down_dtype: DType::F32,
                gu_tags: false,
                down_tags: false,
                gu_force_mq4_fp16: false,
                gu_paro_i8: false,
                gu_paro_i8_k8: false,
                dn_force_mq4_fp16: false,
                dn_paro_i8: false,
                dn_paro_i8_k8: false,
                activation: GroupedActivationKind::QwenAwqIndexed,
                activation_awq: false,
                activation_swiglu_limit: 30.0,
                activation_krot: 4,
            }
        }
    }

    fn grouped_sig_rank(p: &GroupedSigParams) -> RoutedMoeStepPhases<'static> {
        let slots = p.batch_size * p.k_top;
        let tiles = p.m_total / 16;
        let y_gate_up = synth_with_bytes(
            DType::F32,
            p.m_total * (2 * p.inter),
            p.m_total * (2 * p.inter) * 4,
        );
        let y_down = synth_with_bytes(DType::F32, p.m_total * p.hidden, p.m_total * p.hidden * 4);
        let x_rot = synth_with_bytes(
            DType::F32,
            p.batch_size * p.hidden,
            p.batch_size * p.hidden * 4,
        );
        let gate_batch = synth_with_bytes(DType::F32, slots * p.inter, slots * p.inter * 4);
        let up_batch = synth_with_bytes(DType::F32, slots * p.inter, slots * p.inter * 4);
        let rot = synth_with_bytes(DType::F32, slots * p.inter, slots * p.inter * 4);
        let gu_experts = lloyd_expert_ref_with(p.n_experts, p.inter, p.hidden, p.gu_dtype);
        let dn_experts = lloyd_expert_ref_with(p.n_experts, p.inter, p.hidden, p.down_dtype);
        let gu_tags = p
            .gu_tags
            .then(|| synth_with_bytes(DType::Raw, p.n_experts, p.n_experts));
        let dn_tags = p
            .down_tags
            .then(|| synth_with_bytes(DType::Raw, p.n_experts, p.n_experts));
        // Chain buffers are SHARED across their roles (scatter, gate-up,
        // unscatter, down, combine) exactly as the production grammar binds.
        let sorted = synth_with_bytes(DType::Raw, p.m_total, p.m_total * 4);
        let tile_buf = synth_with_bytes(DType::Raw, tiles, tiles * 4);
        let perm = synth_with_bytes(DType::Raw, slots, slots * 4);
        let activation = match p.activation {
            GroupedActivationKind::MinimaxFused => MoeActivationVariant::MinimaxFused {
                awq_scale: p.activation_awq.then(|| synth_f32(4)),
            },
            GroupedActivationKind::Ds4ClampRotate => MoeActivationVariant::Ds4ClampRotate {
                swiglu_limit: p.activation_swiglu_limit,
            },
            GroupedActivationKind::QwenAwqIndexed => MoeActivationVariant::QwenAwqIndexed {
                awq_ptrs: synth_f32(4),
                topk_indices: synth_i64(4),
            },
            GroupedActivationKind::QwenParo => MoeActivationVariant::QwenParo {
                pairs: synth_with_bytes(
                    DType::Raw,
                    p.activation_krot * p.givens_dim,
                    p.activation_krot * p.givens_dim * 2,
                ),
                theta: synth_with_bytes(
                    DType::Raw,
                    p.activation_krot * (p.givens_dim / 2),
                    p.activation_krot * (p.givens_dim / 2) * 2,
                ),
                scales: synth_with_bytes(DType::Raw, p.givens_dim, p.givens_dim * 2),
                krot: p.activation_krot,
            },
        };
        let mut phases = RoutedMoePhases {
            router: vec![Step::MoeScatter {
                topk_indices: synth_i32(slots),
                expert_token_counts: synth_i32(p.n_experts),
                expert_offsets: synth_i32(p.n_experts + 1),
                sorted_slot_index: sorted,
                expert_tile_ids: tile_buf,
                inverse_perm: perm,
                total_slots: slots,
                n_experts: p.n_experts,
                m_total_max: p.m_total,
                block_m: 16,
            }],
            gate_up: vec![
                Step::GroupedMoeGemm {
                    experts: gu_experts,
                    which: MoeProj::GateUp {
                        up_out: synth_f32(4),
                    },
                    sorted_slot_index: sorted,
                    expert_tile_ids: tile_buf,
                    x: x_rot,
                    y: y_gate_up,
                    m_total: p.m_total,
                    batch_size: p.batch_size,
                    k_top: p.k_top,
                    dtype_tags: gu_tags,
                    force_mq4_fp16: p.gu_force_mq4_fp16,
                    paro_i8: p.gu_paro_i8,
                    paro_i8_k8: p.gu_paro_i8_k8,
                },
                Step::MoeGateUpUnscatter {
                    y_grouped: y_gate_up,
                    sorted_slot_index: sorted,
                    gate_batch,
                    up_batch,
                    inter: p.inter,
                    k_top: p.k_top,
                    m_total: p.m_total,
                },
            ],
            activation: vec![Step::MoeActivation {
                variant: activation,
                gate: gate_batch,
                up: up_batch,
                rot_out: rot,
                inter: p.inter,
                k_top: slots,
            }],
            down: vec![Step::GroupedMoeGemm {
                experts: dn_experts,
                which: MoeProj::DownExpanded,
                sorted_slot_index: sorted,
                expert_tile_ids: tile_buf,
                x: rot,
                y: y_down,
                m_total: p.m_total,
                batch_size: p.batch_size,
                k_top: p.k_top,
                dtype_tags: dn_tags,
                force_mq4_fp16: p.dn_force_mq4_fp16,
                paro_i8: p.dn_paro_i8,
                paro_i8_k8: p.dn_paro_i8_k8,
            }],
            combine: vec![Step::MoeCombine {
                down_out: y_down,
                topk_weights: synth_f32(slots),
                out: synth_f32(p.batch_size * p.hidden),
                k: p.k_top,
                hidden: p.hidden,
                batch_size: p.batch_size,
                inverse_perm: Some(perm),
            }],
            finish: Vec::new(),
        };
        if p.givens {
            let Step::GroupedMoeGemm { x, .. } = &mut phases.gate_up[0] else {
                unreachable!()
            };
            let x_rot = *x;
            phases.gate_up.insert(
                0,
                Step::GivensRotateBatched {
                    x: synth_with_bytes(
                        DType::F32,
                        p.batch_size * p.givens_dim,
                        p.batch_size * p.givens_dim * 4,
                    ),
                    out: x_rot,
                    pairs: synth_with_bytes(
                        DType::Raw,
                        p.givens_krot * p.givens_dim,
                        p.givens_krot * p.givens_dim * 2,
                    ),
                    theta: synth_with_bytes(
                        DType::Raw,
                        p.givens_krot * (p.givens_dim / 2),
                        p.givens_krot * (p.givens_dim / 2) * 2,
                    ),
                    scales: synth_with_bytes(DType::Raw, p.givens_dim, p.givens_dim * 2),
                    batch: p.batch_size,
                    dim: p.givens_dim,
                    krot: p.givens_krot,
                },
            );
        }
        phases
    }

    #[test]
    fn lowering_rejects_cross_rank_grouped_geometry_mismatch_with_equal_dim() {
        // Rank A: batch4 × hidden128 (dim 512). Rank B: batch8 × hidden64
        // (dim 512). Both per-rank valid; the flattened collective dim is
        // equal, so before signature remediation the pair passed while per-rank
        // kernel geometry differed. The typed grouped signature must reject.
        let group = grouped_group();
        let policy = tp_policy(2);
        let rank_a = GroupedSigParams::default();
        let rank_b = GroupedSigParams {
            batch_size: 8,
            k_top: 4,
            hidden: 64,
            ..rank_a
        };
        let err = lower_moe_steps(
            &group,
            &policy,
            grouped_parts(vec![grouped_sig_rank(&rank_a), grouped_sig_rank(&rank_b)]),
        )
        .unwrap_err();
        assert!(matches!(err, MoeLowerError::RankProtocolMismatch { .. }));
    }

    #[test]
    fn lowering_rejects_cross_rank_grouped_signature_mismatches() {
        let group = grouped_group();
        let policy = tp_policy(2);
        let base = GroupedSigParams::default();
        let paro_base = GroupedSigParams {
            givens: true,
            ..base
        };
        // Each row isolates one signature discriminator with two per-rank
        // valid programs of equal flattened dim. `phase_masked` rows are
        // rejected by the rank-phase-length check (Givens presence changes
        // the gate_up phase length) before the signature comparison runs —
        // fail-closed, with the signature still carrying the evidence.
        let rows: [(&str, GroupedSigParams, GroupedSigParams, bool); 10] = [
            ("inter", base, GroupedSigParams { inter: 8, ..base }, false),
            (
                "n_experts",
                GroupedSigParams {
                    m_total: 288,
                    ..base
                },
                GroupedSigParams {
                    n_experts: 16,
                    m_total: 288,
                    ..base
                },
                false,
            ),
            (
                "m_total",
                base,
                GroupedSigParams {
                    m_total: 176,
                    ..base
                },
                false,
            ),
            ("givens presence", base, paro_base, true),
            (
                "givens krot",
                paro_base,
                GroupedSigParams {
                    givens_krot: 8,
                    ..paro_base
                },
                false,
            ),
            (
                "gate-up dtype",
                base,
                GroupedSigParams {
                    gu_dtype: DType::MQ4G256,
                    ..base
                },
                false,
            ),
            (
                "down dtype",
                base,
                GroupedSigParams {
                    down_dtype: DType::MQ4G256,
                    ..base
                },
                false,
            ),
            (
                "gate-up tag presence",
                base,
                GroupedSigParams {
                    gu_tags: true,
                    ..base
                },
                false,
            ),
            (
                "down tag presence",
                base,
                GroupedSigParams {
                    down_tags: true,
                    ..base
                },
                false,
            ),
            (
                "activation kind",
                base,
                GroupedSigParams {
                    activation: GroupedActivationKind::MinimaxFused,
                    ..base
                },
                false,
            ),
        ];
        for (name, rank_a, rank_b, phase_masked) in rows {
            let err = lower_moe_steps(
                &group,
                &policy,
                grouped_parts(vec![grouped_sig_rank(&rank_a), grouped_sig_rank(&rank_b)]),
            )
            .unwrap_err();
            if phase_masked {
                assert!(
                    matches!(err, MoeLowerError::RankPhaseMismatch { .. }),
                    "{name} must be rejected by the rank-phase-length check: {err:?}"
                );
            } else {
                assert!(
                    matches!(err, MoeLowerError::RankProtocolMismatch { .. }),
                    "{name} grouped signature mismatch must be rejected cross-rank: {err:?}"
                );
            }
        }
    }

    #[test]
    fn lowering_rejects_cross_rank_grouped_paro_geometry_mismatch_with_equal_dim() {
        // Both ranks Paro with 128-aligned hidden: A = batch2 × hidden256
        // (givens dim 256), B = batch4 × hidden128 (givens dim 128);
        // flattened dim 512 both. The grouped signature must reject (hidden
        // and Givens dim differ while the collective dim matches).
        let group = grouped_group();
        let policy = tp_policy(2);
        let rank_a = GroupedSigParams {
            batch_size: 2,
            hidden: 256,
            givens: true,
            givens_dim: 256,
            ..GroupedSigParams::default()
        };
        let rank_b = GroupedSigParams {
            batch_size: 4,
            hidden: 128,
            givens: true,
            givens_dim: 128,
            ..GroupedSigParams::default()
        };
        let err = lower_moe_steps(
            &group,
            &policy,
            grouped_parts(vec![grouped_sig_rank(&rank_a), grouped_sig_rank(&rank_b)]),
        )
        .unwrap_err();
        assert!(matches!(err, MoeLowerError::RankProtocolMismatch { .. }));
    }

    #[test]
    fn lowering_rejects_cross_rank_grouped_gemm_control_mismatches() {
        // Each grouped-GEMM launch control is a rank-invariant discriminator
        // even when every other scalar matches: equal-geometry pairs differing
        // only in one gate-up or down control must reject.
        let group = grouped_group();
        let policy = tp_policy(2);
        let base = GroupedSigParams::default();
        let rows: [(&str, GroupedSigParams); 6] = [
            (
                "gate-up force_mq4_fp16",
                GroupedSigParams {
                    gu_force_mq4_fp16: true,
                    ..base
                },
            ),
            (
                "gate-up paro_i8",
                GroupedSigParams {
                    gu_paro_i8: true,
                    ..base
                },
            ),
            (
                "gate-up paro_i8_k8",
                GroupedSigParams {
                    gu_paro_i8_k8: true,
                    ..base
                },
            ),
            (
                "down force_mq4_fp16",
                GroupedSigParams {
                    dn_force_mq4_fp16: true,
                    ..base
                },
            ),
            (
                "down paro_i8",
                GroupedSigParams {
                    dn_paro_i8: true,
                    ..base
                },
            ),
            (
                "down paro_i8_k8",
                GroupedSigParams {
                    dn_paro_i8_k8: true,
                    ..base
                },
            ),
        ];
        for (name, rank_b) in rows {
            let err = lower_moe_steps(
                &group,
                &policy,
                grouped_parts(vec![grouped_sig_rank(&base), grouped_sig_rank(&rank_b)]),
            )
            .unwrap_err();
            assert!(
                matches!(err, MoeLowerError::RankProtocolMismatch { .. }),
                "{name} grouped-GEMM control mismatch must be rejected: {err:?}"
            );
        }
    }

    #[test]
    fn lowering_rejects_cross_rank_grouped_activation_discriminant_mismatches() {
        // The exact pointer-free activation discriminants: Minimax AWQ
        // presence, Ds4Clamp swiglu-limit bits, and QwenParo krot are all
        // rank-invariant launch discriminators.
        let group = grouped_group();
        let policy = tp_policy(2);
        let base = GroupedSigParams::default();
        let rows: [(&str, GroupedSigParams, GroupedSigParams); 3] = [
            (
                "MinimaxFused awq presence",
                GroupedSigParams {
                    activation: GroupedActivationKind::MinimaxFused,
                    ..base
                },
                GroupedSigParams {
                    activation: GroupedActivationKind::MinimaxFused,
                    activation_awq: true,
                    ..base
                },
            ),
            (
                "Ds4ClampRotate swiglu bits",
                GroupedSigParams {
                    activation: GroupedActivationKind::Ds4ClampRotate,
                    ..base
                },
                GroupedSigParams {
                    activation: GroupedActivationKind::Ds4ClampRotate,
                    activation_swiglu_limit: 30.5,
                    ..base
                },
            ),
            (
                "QwenParo krot",
                GroupedSigParams {
                    activation: GroupedActivationKind::QwenParo,
                    ..base
                },
                GroupedSigParams {
                    activation: GroupedActivationKind::QwenParo,
                    activation_krot: 8,
                    ..base
                },
            ),
        ];
        for (name, rank_a, rank_b) in rows {
            let err = lower_moe_steps(
                &group,
                &policy,
                grouped_parts(vec![grouped_sig_rank(&rank_a), grouped_sig_rank(&rank_b)]),
            )
            .unwrap_err();
            assert!(
                matches!(err, MoeLowerError::RankProtocolMismatch { .. }),
                "{name} activation discriminant mismatch must be rejected: {err:?}"
            );
        }
    }

    #[test]
    fn lowering_accepts_identical_grouped_controls_and_activation_discriminants() {
        // Identical control and activation discriminants lower to the sealed
        // schedule.
        let group = grouped_group();
        let policy = tp_policy(2);
        for params in [
            GroupedSigParams {
                gu_force_mq4_fp16: true,
                dn_paro_i8: true,
                activation: GroupedActivationKind::Ds4ClampRotate,
                activation_swiglu_limit: 30.5,
                ..GroupedSigParams::default()
            },
            GroupedSigParams {
                activation: GroupedActivationKind::QwenParo,
                activation_krot: 8,
                givens: true,
                ..GroupedSigParams::default()
            },
        ] {
            let lowered = lower_moe_steps(
                &group,
                &policy,
                grouped_parts(vec![grouped_sig_rank(&params), grouped_sig_rank(&params)]),
            )
            .unwrap();
            let LoweredMoeProgramInner::Parallel { collectives, .. } = lowered.inner else {
                panic!("expected parallel program");
            };
            assert_eq!(collective_count(&collectives), 1);
        }
    }

    #[test]
    fn lowering_accepts_identical_grouped_signatures_across_ranks() {
        let group = grouped_group();
        let policy = tp_policy(2);
        for params in [
            GroupedSigParams::default(),
            GroupedSigParams {
                givens: true,
                ..GroupedSigParams::default()
            },
        ] {
            let lowered = lower_moe_steps(
                &group,
                &policy,
                grouped_parts(vec![grouped_sig_rank(&params), grouped_sig_rank(&params)]),
            )
            .unwrap();
            let LoweredMoeProgramInner::Parallel { collectives, .. } = lowered.inner else {
                panic!("expected parallel program");
            };
            let expected_dim = params.batch_size * params.hidden;
            assert_eq!(collective_count(&collectives), 1);
            assert!(collectives.iter().any(|c| matches!(
                c,
                StepCollective::AllReduce { dim, .. } if *dim == expected_dim
            )));
        }
    }

    #[test]
    fn lowering_rejects_grouped_zero_geometry() {
        let group = grouped_group();
        let policy = tp_policy(2);
        // Table-driven admitted zero geometry (k_top, n_experts, inter,
        // m_total) plus the k_top>n_experts range.
        let base = GroupedSigParams::default();
        let cases: [(&str, GroupedSigParams, &str); 5] = [
            ("k_top", GroupedSigParams { k_top: 0, ..base }, "k_top"),
            (
                "n_experts",
                GroupedSigParams {
                    n_experts: 0,
                    ..base
                },
                "n_experts",
            ),
            ("inter", GroupedSigParams { inter: 0, ..base }, "inter"),
            (
                "m_total",
                GroupedSigParams { m_total: 0, ..base },
                "m_total",
            ),
            (
                "k_top > n_experts",
                GroupedSigParams { k_top: 16, ..base },
                "range",
            ),
        ];
        for (name, params, kind) in cases {
            let err = lower_moe_steps(
                &group,
                &policy,
                grouped_parts(vec![grouped_sig_rank(&params), grouped_sig_rank(&params)]),
            )
            .unwrap_err();
            if kind == "range" {
                assert!(
                    matches!(
                        err,
                        MoeLowerError::GroupedKTopExceedsExperts {
                            k_top: 16,
                            n_experts: 8,
                            ..
                        }
                    ),
                    "{name} must reject k_top > n_experts: {err:?}"
                );
            } else {
                assert!(
                    matches!(
                        err,
                        MoeLowerError::GroupedZeroGeometry { field, .. } if field == kind
                    ),
                    "{name} must reject zero {kind}: {err:?}"
                );
            }
        }
    }

    #[test]
    fn lowering_accepts_grouped_sentinel_gate_up_up_out() {
        // The grouped GateUp.up_out is unused metadata: an unrelated sentinel
        // buffer must be accepted (the real chain is GateUp.y == unscatter
        // y_grouped and unscatter gate/up == activation gate/up).
        let group = grouped_group();
        let policy = tp_policy(2);
        let (sorted, tiles, perm) = grouped_fixture();
        let lowered = lower_moe_steps(
            &group,
            &policy,
            grouped_parts(vec![
                grouped_phases(sorted, tiles, perm, Some(perm)),
                grouped_phases(sorted, tiles, perm, Some(perm)),
            ]),
        )
        .unwrap();
        let LoweredMoeProgramInner::Parallel { collectives, .. } = lowered.inner else {
            panic!("expected parallel program");
        };
        assert!(matches!(
            collectives[5],
            StepCollective::AllReduce {
                kind: DimKind::Tp,
                dim: 512
            }
        ));
        assert_eq!(collective_count(&collectives), 1);
    }

    #[test]
    fn lowering_rejects_grouped_inter_width_mismatch() {
        // Intermediate width: gate-up expert_m == unscatter.inter ==
        // activation.inter == grouped-down expert_m. The down experts here
        // claim expert_m=8 while the gate-up/unscatter/activation run inter=4.
        let group = grouped_group();
        let policy = tp_policy(2);
        let (sorted, tiles, perm) = grouped_fixture();
        let rank = || {
            let mut phases = grouped_phases(sorted, tiles, perm, Some(perm));
            let Step::GroupedMoeGemm { experts, .. } = &mut phases.down[0] else {
                unreachable!()
            };
            *experts = lloyd_expert_ref_with(8, 8, 128, DType::F32);
            phases
        };
        let err =
            lower_moe_steps(&group, &policy, grouped_parts(vec![rank(), rank()])).unwrap_err();
        assert!(matches!(
            err,
            MoeLowerError::GroupedChainMismatch { field: "inter", .. }
        ));
    }

    #[test]
    fn lowering_rejects_grouped_hidden_width_mismatch() {
        // Hidden width: gate-up expert_k == grouped-down expert_k == combine
        // hidden. The down experts here claim expert_k=8 (and the combine
        // follows with hidden=8) while the gate-up runs expert_k=4.
        let group = grouped_group();
        let policy = tp_policy(2);
        let (sorted, tiles, perm) = grouped_fixture();
        let rank = || {
            let mut phases = grouped_phases(sorted, tiles, perm, Some(perm));
            let Step::GroupedMoeGemm { experts, .. } = &mut phases.down[0] else {
                unreachable!()
            };
            *experts = lloyd_expert_ref_with(8, 4, 256, DType::F32);
            phases
        };
        let err =
            lower_moe_steps(&group, &policy, grouped_parts(vec![rank(), rank()])).unwrap_err();
        assert!(matches!(
            err,
            MoeLowerError::GroupedChainMismatch {
                field: "hidden",
                ..
            }
        ));
    }

    #[test]
    fn lowering_rejects_grouped_undersized_scatter_operands() {
        let group = grouped_group();
        let policy = tp_policy(2);
        let (sorted, tiles, perm) = grouped_fixture();
        // topk_indices below slots*4.
        let rank = || {
            let mut phases = grouped_phases(sorted, tiles, perm, Some(perm));
            let Step::MoeScatter { topk_indices, .. } = &mut phases.router[0] else {
                unreachable!()
            };
            *topk_indices = synth_i32(31);
            phases
        };
        let err =
            lower_moe_steps(&group, &policy, grouped_parts(vec![rank(), rank()])).unwrap_err();
        assert!(matches!(
            err,
            MoeLowerError::CapacityMismatch {
                expected_bytes: 128,
                actual_bytes: 124,
                ..
            }
        ));

        // expert_token_counts below n_experts*4.
        let rank = || {
            let mut phases = grouped_phases(sorted, tiles, perm, Some(perm));
            let Step::MoeScatter {
                expert_token_counts,
                ..
            } = &mut phases.router[0]
            else {
                unreachable!()
            };
            *expert_token_counts = synth_i32(7);
            phases
        };
        let err =
            lower_moe_steps(&group, &policy, grouped_parts(vec![rank(), rank()])).unwrap_err();
        assert!(matches!(
            err,
            MoeLowerError::CapacityMismatch {
                expected_bytes: 32,
                actual_bytes: 28,
                ..
            }
        ));

        // expert_offsets below (n_experts+1)*4.
        let rank = || {
            let mut phases = grouped_phases(sorted, tiles, perm, Some(perm));
            let Step::MoeScatter { expert_offsets, .. } = &mut phases.router[0] else {
                unreachable!()
            };
            *expert_offsets = synth_i32(8);
            phases
        };
        let err =
            lower_moe_steps(&group, &policy, grouped_parts(vec![rank(), rank()])).unwrap_err();
        assert!(matches!(
            err,
            MoeLowerError::CapacityMismatch {
                expected_bytes: 36,
                actual_bytes: 32,
                ..
            }
        ));

        // sorted_slot_index below m_total_max*4.
        let rank = || {
            let mut phases = grouped_phases(sorted, tiles, perm, Some(perm));
            let Step::MoeScatter {
                sorted_slot_index, ..
            } = &mut phases.router[0]
            else {
                unreachable!()
            };
            let undersized = synth_with_bytes(DType::Raw, 159, 159 * 4);
            let Step::MoeScatter {
                sorted_slot_index, ..
            } = &mut phases.router[0]
            else {
                unreachable!()
            };
            *sorted_slot_index = undersized;
            for step in &mut phases.gate_up {
                match step {
                    Step::GroupedMoeGemm {
                        sorted_slot_index, ..
                    } => *sorted_slot_index = undersized,
                    Step::MoeGateUpUnscatter {
                        sorted_slot_index, ..
                    } => *sorted_slot_index = undersized,
                    _ => {}
                }
            }
            let Step::GroupedMoeGemm {
                sorted_slot_index, ..
            } = &mut phases.down[0]
            else {
                unreachable!()
            };
            *sorted_slot_index = undersized;
            phases
        };
        let err =
            lower_moe_steps(&group, &policy, grouped_parts(vec![rank(), rank()])).unwrap_err();
        assert!(matches!(
            err,
            MoeLowerError::CapacityMismatch {
                expected_bytes: 640,
                actual_bytes: 636,
                ..
            }
        ));

        // expert_tile_ids below (m_total_max/block_m)*4.
        let rank = || {
            let mut phases = grouped_phases(sorted, tiles, perm, Some(perm));
            let Step::MoeScatter {
                expert_tile_ids, ..
            } = &mut phases.router[0]
            else {
                unreachable!()
            };
            let undersized = synth_with_bytes(DType::Raw, 9, 9 * 4);
            *expert_tile_ids = undersized;
            let Step::GroupedMoeGemm {
                expert_tile_ids, ..
            } = &mut phases.gate_up[0]
            else {
                unreachable!()
            };
            *expert_tile_ids = undersized;
            let Step::GroupedMoeGemm {
                expert_tile_ids, ..
            } = &mut phases.down[0]
            else {
                unreachable!()
            };
            *expert_tile_ids = undersized;
            phases
        };
        let err =
            lower_moe_steps(&group, &policy, grouped_parts(vec![rank(), rank()])).unwrap_err();
        assert!(matches!(
            err,
            MoeLowerError::CapacityMismatch {
                expected_bytes: 40,
                actual_bytes: 36,
                ..
            }
        ));

        // inverse_perm below slots*4.
        let rank = || {
            let mut phases = grouped_phases(sorted, tiles, perm, Some(perm));
            let undersized = synth_with_bytes(DType::Raw, 31, 31 * 4);
            let Step::MoeScatter { inverse_perm, .. } = &mut phases.router[0] else {
                unreachable!()
            };
            *inverse_perm = undersized;
            let Step::MoeCombine { inverse_perm, .. } = &mut phases.combine[0] else {
                unreachable!()
            };
            *inverse_perm = Some(undersized);
            phases
        };
        let err =
            lower_moe_steps(&group, &policy, grouped_parts(vec![rank(), rank()])).unwrap_err();
        assert!(matches!(
            err,
            MoeLowerError::CapacityMismatch {
                expected_bytes: 128,
                actual_bytes: 124,
                ..
            }
        ));
    }

    #[test]
    fn lowering_rejects_grouped_undersized_gate_up_operands() {
        let group = grouped_group();
        let policy = tp_policy(2);
        let (sorted, tiles, perm) = grouped_fixture();
        // Gate-up input below batch*hidden*4.
        let rank = || {
            let mut phases = grouped_phases(sorted, tiles, perm, Some(perm));
            let Step::GroupedMoeGemm { x, .. } = &mut phases.gate_up[0] else {
                unreachable!()
            };
            *x = synth_with_bytes(DType::F32, 511, 511 * 4);
            phases
        };
        let err =
            lower_moe_steps(&group, &policy, grouped_parts(vec![rank(), rank()])).unwrap_err();
        assert!(matches!(
            err,
            MoeLowerError::CapacityMismatch {
                expected_bytes: 2048,
                actual_bytes: 2044,
                ..
            }
        ));

        // Gate-up expert pointer table below n_experts*8.
        let rank = || {
            let mut phases = grouped_phases(sorted, tiles, perm, Some(perm));
            let Step::GroupedMoeGemm { experts, .. } = &mut phases.gate_up[0] else {
                unreachable!()
            };
            *experts =
                lloyd_expert_ref_with_tables(8, 4, 128, DType::F32, synth_f32(4), synth_f32(8));
            phases
        };
        let err =
            lower_moe_steps(&group, &policy, grouped_parts(vec![rank(), rank()])).unwrap_err();
        assert!(matches!(
            err,
            MoeLowerError::CapacityMismatch {
                expected_bytes: 64,
                actual_bytes: 16,
                ..
            }
        ));

        // Fused grouped output below m_total*(2*inter)*4.
        let rank = || {
            let mut phases = grouped_phases(sorted, tiles, perm, Some(perm));
            let undersized = synth_with_bytes(DType::F32, 159 * 8, 159 * 8 * 4);
            let Step::GroupedMoeGemm { y, .. } = &mut phases.gate_up[0] else {
                unreachable!()
            };
            *y = undersized;
            let Step::MoeGateUpUnscatter { y_grouped, .. } = &mut phases.gate_up[1] else {
                unreachable!()
            };
            *y_grouped = undersized;
            phases
        };
        let err =
            lower_moe_steps(&group, &policy, grouped_parts(vec![rank(), rank()])).unwrap_err();
        assert!(matches!(
            err,
            MoeLowerError::CapacityMismatch {
                expected_bytes: 5120,
                actual_bytes: 5088,
                ..
            }
        ));
    }

    #[test]
    fn lowering_rejects_grouped_undersized_unscatter_operands() {
        let group = grouped_group();
        let policy = tp_policy(2);
        let (sorted, tiles, perm) = grouped_fixture();
        // gate_batch below slots*inter*4.
        let rank = || {
            let mut phases = grouped_phases(sorted, tiles, perm, Some(perm));
            let undersized = synth_with_bytes(DType::F32, 127, 127 * 4);
            let Step::MoeGateUpUnscatter { gate_batch, .. } = &mut phases.gate_up[1] else {
                unreachable!()
            };
            *gate_batch = undersized;
            let Step::MoeActivation { gate, .. } = &mut phases.activation[0] else {
                unreachable!()
            };
            *gate = undersized;
            phases
        };
        let err =
            lower_moe_steps(&group, &policy, grouped_parts(vec![rank(), rank()])).unwrap_err();
        assert!(matches!(
            err,
            MoeLowerError::CapacityMismatch {
                expected_bytes: 512,
                actual_bytes: 508,
                ..
            }
        ));

        // up_batch below slots*inter*4.
        let rank = || {
            let mut phases = grouped_phases(sorted, tiles, perm, Some(perm));
            let undersized = synth_with_bytes(DType::F32, 127, 127 * 4);
            let Step::MoeGateUpUnscatter { up_batch, .. } = &mut phases.gate_up[1] else {
                unreachable!()
            };
            *up_batch = undersized;
            let Step::MoeActivation { up, .. } = &mut phases.activation[0] else {
                unreachable!()
            };
            *up = undersized;
            phases
        };
        let err =
            lower_moe_steps(&group, &policy, grouped_parts(vec![rank(), rank()])).unwrap_err();
        assert!(matches!(
            err,
            MoeLowerError::CapacityMismatch {
                expected_bytes: 512,
                actual_bytes: 508,
                ..
            }
        ));
    }

    #[test]
    fn lowering_rejects_grouped_undersized_activation_operands() {
        let group = grouped_group();
        let policy = tp_policy(2);
        let (sorted, tiles, perm) = grouped_fixture();
        // rot_out below slots*inter*4 (gate/up alias the unscatter buffers,
        // which the unscatter capacity checks already cover).
        let rank = || {
            let mut phases = grouped_phases(sorted, tiles, perm, Some(perm));
            let undersized = synth_with_bytes(DType::F32, 127, 127 * 4);
            let Step::MoeActivation { rot_out, .. } = &mut phases.activation[0] else {
                unreachable!()
            };
            *rot_out = undersized;
            let Step::GroupedMoeGemm { x, .. } = &mut phases.down[0] else {
                unreachable!()
            };
            *x = undersized;
            phases
        };
        let err =
            lower_moe_steps(&group, &policy, grouped_parts(vec![rank(), rank()])).unwrap_err();
        assert!(matches!(
            err,
            MoeLowerError::CapacityMismatch {
                expected_bytes: 512,
                actual_bytes: 508,
                ..
            }
        ));
    }

    #[test]
    fn lowering_rejects_grouped_undersized_down_operands() {
        let group = grouped_group();
        let policy = tp_policy(2);
        let (sorted, tiles, perm) = grouped_fixture();
        // Down expert pointer table below n_experts*8.
        let rank = || {
            let mut phases = grouped_phases(sorted, tiles, perm, Some(perm));
            let Step::GroupedMoeGemm { experts, .. } = &mut phases.down[0] else {
                unreachable!()
            };
            *experts =
                lloyd_expert_ref_with_tables(8, 4, 128, DType::F32, synth_f32(8), synth_f32(4));
            phases
        };
        let err =
            lower_moe_steps(&group, &policy, grouped_parts(vec![rank(), rank()])).unwrap_err();
        assert!(matches!(
            err,
            MoeLowerError::CapacityMismatch {
                expected_bytes: 64,
                actual_bytes: 16,
                ..
            }
        ));

        // Grouped down output below m_total*hidden*4.
        let rank = || {
            let mut phases = grouped_phases(sorted, tiles, perm, Some(perm));
            let undersized = synth_with_bytes(DType::F32, 159 * 4, 159 * 4 * 4);
            let Step::GroupedMoeGemm { y, .. } = &mut phases.down[0] else {
                unreachable!()
            };
            *y = undersized;
            let Step::MoeCombine { down_out, .. } = &mut phases.combine[0] else {
                unreachable!()
            };
            *down_out = undersized;
            phases
        };
        let err =
            lower_moe_steps(&group, &policy, grouped_parts(vec![rank(), rank()])).unwrap_err();
        assert!(matches!(
            err,
            MoeLowerError::CapacityMismatch {
                expected_bytes: 81920,
                actual_bytes: 2544,
                ..
            }
        ));
    }

    #[test]
    fn lowering_rejects_grouped_undersized_combine_operands() {
        let group = grouped_group();
        let policy = tp_policy(2);
        let (sorted, tiles, perm) = grouped_fixture();
        // topk_weights below slots*4 (down_out/inverse_perm alias the down y
        // and the scatter perm, covered by those capacity checks; the final
        // output keeps its existing generic check).
        let rank = || {
            let mut phases = grouped_phases(sorted, tiles, perm, Some(perm));
            let Step::MoeCombine { topk_weights, .. } = &mut phases.combine[0] else {
                unreachable!()
            };
            *topk_weights = synth_f32(31);
            phases
        };
        let err =
            lower_moe_steps(&group, &policy, grouped_parts(vec![rank(), rank()])).unwrap_err();
        assert!(matches!(
            err,
            MoeLowerError::CapacityMismatch {
                expected_bytes: 128,
                actual_bytes: 124,
                ..
            }
        ));
    }

    #[test]
    fn lowering_rejects_grouped_undersized_givens_operands() {
        let group = grouped_group();
        let policy = tp_policy(2);
        let (sorted, tiles, perm) = grouped_fixture();
        // Givens input below batch*dim*4.
        let rank = || {
            let mut phases = grouped_phases_paro(sorted, tiles, perm, Some(perm));
            let Step::GivensRotateBatched { x, .. } = &mut phases.gate_up[0] else {
                unreachable!()
            };
            *x = synth_with_bytes(DType::F32, 511, 511 * 4);
            phases
        };
        let err =
            lower_moe_steps(&group, &policy, grouped_parts(vec![rank(), rank()])).unwrap_err();
        assert!(matches!(
            err,
            MoeLowerError::CapacityMismatch {
                expected_bytes: 2048,
                actual_bytes: 2044,
                ..
            }
        ));

        // Givens output below batch*dim*4.
        let rank = || {
            let mut phases = grouped_phases_paro(sorted, tiles, perm, Some(perm));
            let undersized = synth_with_bytes(DType::F32, 511, 511 * 4);
            let Step::GivensRotateBatched { out, .. } = &mut phases.gate_up[0] else {
                unreachable!()
            };
            *out = undersized;
            let Step::GroupedMoeGemm { x, .. } = &mut phases.gate_up[1] else {
                unreachable!()
            };
            *x = undersized;
            phases
        };
        let err =
            lower_moe_steps(&group, &policy, grouped_parts(vec![rank(), rank()])).unwrap_err();
        assert!(matches!(
            err,
            MoeLowerError::CapacityMismatch {
                expected_bytes: 2048,
                actual_bytes: 2044,
                ..
            }
        ));

        // pairs below krot*dim*2 (i16).
        let rank = || {
            let mut phases = grouped_phases_paro(sorted, tiles, perm, Some(perm));
            let Step::GivensRotateBatched { pairs, .. } = &mut phases.gate_up[0] else {
                unreachable!()
            };
            *pairs = synth_with_bytes(DType::Raw, 511, 511 * 2);
            phases
        };
        let err =
            lower_moe_steps(&group, &policy, grouped_parts(vec![rank(), rank()])).unwrap_err();
        assert!(matches!(
            err,
            MoeLowerError::CapacityMismatch {
                expected_bytes: 1024,
                actual_bytes: 1022,
                ..
            }
        ));

        // theta below krot*(dim/2)*2 (f16).
        let rank = || {
            let mut phases = grouped_phases_paro(sorted, tiles, perm, Some(perm));
            let Step::GivensRotateBatched { theta, .. } = &mut phases.gate_up[0] else {
                unreachable!()
            };
            *theta = synth_with_bytes(DType::Raw, 255, 255 * 2);
            phases
        };
        let err =
            lower_moe_steps(&group, &policy, grouped_parts(vec![rank(), rank()])).unwrap_err();
        assert!(matches!(
            err,
            MoeLowerError::CapacityMismatch {
                expected_bytes: 512,
                actual_bytes: 510,
                ..
            }
        ));

        // scales below dim*2 (f16).
        let rank = || {
            let mut phases = grouped_phases_paro(sorted, tiles, perm, Some(perm));
            let Step::GivensRotateBatched { scales, .. } = &mut phases.gate_up[0] else {
                unreachable!()
            };
            *scales = synth_with_bytes(DType::Raw, 127, 127 * 2);
            phases
        };
        let err =
            lower_moe_steps(&group, &policy, grouped_parts(vec![rank(), rank()])).unwrap_err();
        assert!(matches!(
            err,
            MoeLowerError::CapacityMismatch {
                expected_bytes: 256,
                actual_bytes: 254,
                ..
            }
        ));
    }

    #[test]
    fn lowering_rejects_grouped_nonaligned_m_total() {
        // m_total_max must be a multiple of the frozen block width 16; a
        // nonaligned bound would truncate the tile count.
        let group = grouped_group();
        let policy = tp_policy(2);
        let (sorted, tiles, perm) = grouped_fixture();
        let rank = || {
            let mut phases = grouped_phases(sorted, tiles, perm, Some(perm));
            let Step::MoeScatter { m_total_max, .. } = &mut phases.router[0] else {
                unreachable!()
            };
            *m_total_max = 161;
            let Step::GroupedMoeGemm { m_total, .. } = &mut phases.gate_up[0] else {
                unreachable!()
            };
            *m_total = 161;
            let Step::MoeGateUpUnscatter { m_total, .. } = &mut phases.gate_up[1] else {
                unreachable!()
            };
            *m_total = 161;
            let Step::GroupedMoeGemm { m_total, .. } = &mut phases.down[0] else {
                unreachable!()
            };
            *m_total = 161;
            phases
        };
        let err =
            lower_moe_steps(&group, &policy, grouped_parts(vec![rank(), rank()])).unwrap_err();
        assert!(matches!(
            err,
            MoeLowerError::GroupedTileAlignment {
                m_total: 161,
                block_m: 16,
                ..
            }
        ));
    }

    #[test]
    fn lowering_rejects_grouped_checked_arithmetic_overflow() {
        let group = grouped_group();
        let policy = tp_policy(2);
        let (sorted, tiles, perm) = grouped_fixture();

        // offsets = n_experts + 1 overflows.
        let rank = || {
            let mut phases = grouped_phases(sorted, tiles, perm, Some(perm));
            let Step::MoeScatter { n_experts, .. } = &mut phases.router[0] else {
                unreachable!()
            };
            *n_experts = usize::MAX;
            let Step::GroupedMoeGemm { experts, .. } = &mut phases.gate_up[0] else {
                unreachable!()
            };
            *experts = lloyd_expert_ref_with_tables(
                usize::MAX,
                4,
                128,
                DType::F32,
                synth_fake_capacity(DType::Raw, vec![1], usize::MAX),
                synth_fake_capacity(DType::Raw, vec![1], usize::MAX),
            );
            let Step::GroupedMoeGemm { experts, .. } = &mut phases.down[0] else {
                unreachable!()
            };
            *experts = lloyd_expert_ref_with_tables(
                usize::MAX,
                4,
                128,
                DType::F32,
                synth_fake_capacity(DType::Raw, vec![1], usize::MAX),
                synth_fake_capacity(DType::Raw, vec![1], usize::MAX),
            );
            phases
        };
        let err =
            lower_moe_steps(&group, &policy, grouped_parts(vec![rank(), rank()])).unwrap_err();
        assert!(matches!(
            err,
            MoeLowerError::ArithmeticOverflow {
                what: "offsets_elems",
                ..
            }
        ));

        // 2*inter overflows (gate-up fused output row width).
        let rank = || {
            let mut phases = grouped_phases(sorted, tiles, perm, Some(perm));
            let experts = lloyd_expert_ref_with(8, usize::MAX / 2 + 1, 128, DType::F32);
            let Step::GroupedMoeGemm { experts: gu, .. } = &mut phases.gate_up[0] else {
                unreachable!()
            };
            *gu = experts;
            let Step::MoeGateUpUnscatter { inter, .. } = &mut phases.gate_up[1] else {
                unreachable!()
            };
            *inter = usize::MAX / 2 + 1;
            let Step::MoeActivation {
                inter: act_inter, ..
            } = &mut phases.activation[0]
            else {
                unreachable!()
            };
            *act_inter = usize::MAX / 2 + 1;
            let Step::GroupedMoeGemm { experts: dn, .. } = &mut phases.down[0] else {
                unreachable!()
            };
            *dn = experts;
            phases
        };
        let err =
            lower_moe_steps(&group, &policy, grouped_parts(vec![rank(), rank()])).unwrap_err();
        assert!(matches!(
            err,
            MoeLowerError::ArithmeticOverflow {
                what: "gu_y_inter_elems",
                ..
            }
        ));

        // m_total * (2*inter) overflows.
        let rank = || {
            let mut phases = grouped_phases(sorted, tiles, perm, Some(perm));
            let m_total = (usize::MAX / 8 + 1) & !15usize;
            let Step::MoeScatter { m_total_max, .. } = &mut phases.router[0] else {
                unreachable!()
            };
            *m_total_max = m_total;
            let Step::GroupedMoeGemm { m_total: gu, .. } = &mut phases.gate_up[0] else {
                unreachable!()
            };
            *gu = m_total;
            let Step::MoeGateUpUnscatter { m_total: us, .. } = &mut phases.gate_up[1] else {
                unreachable!()
            };
            *us = m_total;
            let Step::GroupedMoeGemm { m_total: dn, .. } = &mut phases.down[0] else {
                unreachable!()
            };
            *dn = m_total;
            phases
        };
        let err =
            lower_moe_steps(&group, &policy, grouped_parts(vec![rank(), rank()])).unwrap_err();
        assert!(matches!(
            err,
            MoeLowerError::ArithmeticOverflow {
                what: "gu_y_elems",
                ..
            }
        ));

        // slots * inter overflows (inter_elems). batch=MAX/4 keeps the
        // combine preamble and gate-up input byte math representable; the
        // element phase overflows first.
        let rank = || {
            let mut phases = grouped_phases(sorted, tiles, perm, Some(perm));
            let batch = usize::MAX / 8;
            let Step::GroupedMoeGemm { batch_size, .. } = &mut phases.gate_up[0] else {
                unreachable!()
            };
            *batch_size = batch;
            let Step::GroupedMoeGemm { batch_size, .. } = &mut phases.down[0] else {
                unreachable!()
            };
            *batch_size = batch;
            let Step::MoeCombine {
                batch_size,
                out,
                hidden,
                ..
            } = &mut phases.combine[0]
            else {
                unreachable!()
            };
            *batch_size = batch;
            *out = synth_fake_capacity(DType::F32, vec![batch], batch * 4);
            *hidden = 1;
            let Step::MoeScatter { total_slots, .. } = &mut phases.router[0] else {
                unreachable!()
            };
            *total_slots = batch * 8;
            let Step::MoeActivation { k_top: rows, .. } = &mut phases.activation[0] else {
                unreachable!()
            };
            *rows = batch * 8;
            let experts = lloyd_expert_ref_with(8, 4, 1, DType::F32);
            let Step::GroupedMoeGemm { experts: gu, .. } = &mut phases.gate_up[0] else {
                unreachable!()
            };
            *gu = experts;
            let Step::GroupedMoeGemm { experts: dn, .. } = &mut phases.down[0] else {
                unreachable!()
            };
            *dn = experts;
            phases
        };
        let err =
            lower_moe_steps(&group, &policy, grouped_parts(vec![rank(), rank()])).unwrap_err();
        assert!(matches!(
            err,
            MoeLowerError::ArithmeticOverflow {
                what: "inter_elems",
                ..
            }
        ));

        // m_total * hidden overflows (dn_y_elems).
        let rank = || {
            let mut phases = grouped_phases(sorted, tiles, perm, Some(perm));
            let m_total = 1usize << 60;
            let Step::MoeScatter { m_total_max, .. } = &mut phases.router[0] else {
                unreachable!()
            };
            *m_total_max = m_total;
            let Step::GroupedMoeGemm { m_total: gu, .. } = &mut phases.gate_up[0] else {
                unreachable!()
            };
            *gu = m_total;
            let Step::MoeGateUpUnscatter { m_total: us, .. } = &mut phases.gate_up[1] else {
                unreachable!()
            };
            *us = m_total;
            let Step::GroupedMoeGemm { m_total: dn, .. } = &mut phases.down[0] else {
                unreachable!()
            };
            *dn = m_total;
            let experts = lloyd_expert_ref_with(8, 4, 16, DType::F32);
            let Step::GroupedMoeGemm { experts: gu2, .. } = &mut phases.gate_up[0] else {
                unreachable!()
            };
            *gu2 = experts;
            let Step::GroupedMoeGemm { experts: dn2, .. } = &mut phases.down[0] else {
                unreachable!()
            };
            *dn2 = experts;
            let Step::MoeCombine { hidden, out, .. } = &mut phases.combine[0] else {
                unreachable!()
            };
            *hidden = 16;
            *out = synth_f32(4 * 16);
            phases
        };
        let err =
            lower_moe_steps(&group, &policy, grouped_parts(vec![rank(), rank()])).unwrap_err();
        assert!(matches!(
            err,
            MoeLowerError::ArithmeticOverflow {
                what: "dn_y_elems",
                ..
            }
        ));

        // Givens krot * dim overflows (givens_pairs_elems).
        let rank = || {
            let mut phases = grouped_phases_paro(sorted, tiles, perm, Some(perm));
            let Step::GivensRotateBatched { krot, .. } = &mut phases.gate_up[0] else {
                unreachable!()
            };
            *krot = usize::MAX / 2;
            phases
        };
        let err =
            lower_moe_steps(&group, &policy, grouped_parts(vec![rank(), rank()])).unwrap_err();
        assert!(matches!(
            err,
            MoeLowerError::ArithmeticOverflow {
                what: "givens_pairs_elems",
                ..
            }
        ));
    }

    #[test]
    fn lowering_rejects_grouped_givens_zero_dim() {
        let group = grouped_group();
        let policy = tp_policy(2);
        let (sorted, tiles, perm) = grouped_fixture();
        let rank = || {
            let mut phases = grouped_phases_paro(sorted, tiles, perm, Some(perm));
            let Step::GivensRotateBatched { dim, .. } = &mut phases.gate_up[0] else {
                unreachable!()
            };
            *dim = 0;
            phases
        };
        let err =
            lower_moe_steps(&group, &policy, grouped_parts(vec![rank(), rank()])).unwrap_err();
        assert!(matches!(
            err,
            MoeLowerError::GroupedChainMismatch {
                field: "givens_dim",
                ..
            }
        ));
    }

    #[test]
    fn lowering_rejects_grouped_givens_non_128_dim() {
        let group = grouped_group();
        let policy = tp_policy(2);
        let (sorted, tiles, perm) = grouped_fixture();
        let rank = || {
            let mut phases = grouped_phases_paro(sorted, tiles, perm, Some(perm));
            let Step::GivensRotateBatched { dim, .. } = &mut phases.gate_up[0] else {
                unreachable!()
            };
            *dim = 64;
            phases
        };
        let err =
            lower_moe_steps(&group, &policy, grouped_parts(vec![rank(), rank()])).unwrap_err();
        assert!(matches!(
            err,
            MoeLowerError::GroupedGivensAlignment {
                dim: 64,
                expected_alignment: 128,
                ..
            }
        ));
    }

    #[test]
    fn lowering_accepts_grouped_independent_dtype_tags() {
        // dtype_tags are independent between the grouped GateUp and the
        // grouped Down: each Some needs its own n_experts-byte tag table,
        // neither implies the other.
        let group = grouped_group();
        let policy = tp_policy(2);
        let (sorted, tiles, perm) = grouped_fixture();
        for (gate_up_tags, down_tags) in [
            (None, None),
            (Some(synth_with_bytes(DType::Raw, 8, 8)), None),
            (None, Some(synth_with_bytes(DType::Raw, 8, 8))),
            (
                Some(synth_with_bytes(DType::Raw, 8, 8)),
                Some(synth_with_bytes(DType::Raw, 8, 8)),
            ),
        ] {
            let rank = || {
                let mut phases = grouped_phases(sorted, tiles, perm, Some(perm));
                let Step::GroupedMoeGemm { dtype_tags, .. } = &mut phases.gate_up[0] else {
                    unreachable!()
                };
                *dtype_tags = gate_up_tags;
                let Step::GroupedMoeGemm { dtype_tags, .. } = &mut phases.down[0] else {
                    unreachable!()
                };
                *dtype_tags = down_tags;
                phases
            };
            let err = lower_moe_steps(&group, &policy, grouped_parts(vec![rank(), rank()]));
            assert!(err.is_ok(), "tag combination must lower: {err:?}");
        }
    }

    #[test]
    fn lowering_rejects_grouped_undersized_dtype_tags() {
        let group = grouped_group();
        let policy = tp_policy(2);
        let (sorted, tiles, perm) = grouped_fixture();
        // Gate-up tag table one byte short of n_experts.
        let rank = || {
            let mut phases = grouped_phases(sorted, tiles, perm, Some(perm));
            let Step::GroupedMoeGemm { dtype_tags, .. } = &mut phases.gate_up[0] else {
                unreachable!()
            };
            *dtype_tags = Some(synth_with_bytes(DType::Raw, 8, 7));
            phases
        };
        let err =
            lower_moe_steps(&group, &policy, grouped_parts(vec![rank(), rank()])).unwrap_err();
        assert!(matches!(
            err,
            MoeLowerError::CapacityMismatch {
                expected_bytes: 8,
                actual_bytes: 7,
                ..
            }
        ));

        // Down tag table one byte short of n_experts.
        let rank = || {
            let mut phases = grouped_phases(sorted, tiles, perm, Some(perm));
            let Step::GroupedMoeGemm { dtype_tags, .. } = &mut phases.down[0] else {
                unreachable!()
            };
            *dtype_tags = Some(synth_with_bytes(DType::Raw, 8, 7));
            phases
        };
        let err =
            lower_moe_steps(&group, &policy, grouped_parts(vec![rank(), rank()])).unwrap_err();
        assert!(matches!(
            err,
            MoeLowerError::CapacityMismatch {
                expected_bytes: 8,
                actual_bytes: 7,
                ..
            }
        ));
    }

    #[test]
    fn lowering_rejects_stray_givens_in_indexed_expanded() {
        // A Givens rotation is grouped-Paro machinery; it is never admitted
        // on an indexed expanded protocol.
        let rank = || {
            let out = synth_f32(4);
            RoutedMoePhases {
                router: Vec::new(),
                gate_up: vec![
                    gate_up_step(),
                    Step::GivensRotateBatched {
                        x: synth_f32(4),
                        out: synth_f32(4),
                        pairs: synth_f32(4),
                        theta: synth_f32(4),
                        scales: synth_f32(4),
                        batch: 1,
                        dim: 4,
                        krot: 4,
                    },
                ],
                activation: vec![activation_step()],
                down: vec![Step::IndexedMoeGemv {
                    experts: expert_ref(),
                    which: MoeProj::DownExpanded,
                    topk_indices: synth_i64(4),
                    input: GemvInput::Raw(synth_f32(4)),
                    out,
                    k_top: 2,
                    batch_size: 1,
                }],
                combine: vec![Step::MoeCombine {
                    down_out: out,
                    topk_weights: synth_f32(4),
                    out: synth_f32(4),
                    k: 2,
                    hidden: 4,
                    batch_size: 1,
                    inverse_perm: None,
                }],
                finish: Vec::new(),
            }
        };
        let group = group(ExpertParallelism::TensorParallel, 2);
        let policy = tp_policy(2);
        let err = lower_moe_steps(&group, &policy, parts(vec![rank(), rank()])).unwrap_err();
        assert!(matches!(
            err,
            MoeLowerError::StrayGroupedOp { op: "givens", .. }
        ));
    }

    #[test]
    fn lowering_rejects_stray_grouped_ops_in_non_grouped_protocols() {
        // Indexed expanded with a stray gate-up unscatter.
        let rank = || {
            let out = synth_f32(4);
            RoutedMoePhases {
                router: Vec::new(),
                gate_up: vec![
                    gate_up_step(),
                    Step::MoeGateUpUnscatter {
                        y_grouped: synth_f32(4),
                        sorted_slot_index: synth_i64(4),
                        gate_batch: synth_f32(4),
                        up_batch: synth_f32(4),
                        inter: 4,
                        k_top: 2,
                        m_total: 16,
                    },
                ],
                activation: vec![activation_step()],
                down: vec![Step::IndexedMoeGemv {
                    experts: expert_ref(),
                    which: MoeProj::DownExpanded,
                    topk_indices: synth_i64(4),
                    input: GemvInput::Raw(synth_f32(4)),
                    out,
                    k_top: 2,
                    batch_size: 1,
                }],
                combine: vec![Step::MoeCombine {
                    down_out: out,
                    topk_weights: synth_f32(4),
                    out: synth_f32(4),
                    k: 2,
                    hidden: 4,
                    batch_size: 1,
                    inverse_perm: None,
                }],
                finish: Vec::new(),
            }
        };
        let group = group(ExpertParallelism::TensorParallel, 2);
        let policy = tp_policy(2);
        let err = lower_moe_steps(&group, &policy, parts(vec![rank(), rank()])).unwrap_err();
        assert!(matches!(
            err,
            MoeLowerError::StrayGroupedOp {
                op: "gate_up_unscatter",
                ..
            }
        ));

        // Indexed expanded with a stray grouped gate-up.
        let rank = || {
            let out = synth_f32(4);
            RoutedMoePhases {
                router: Vec::new(),
                gate_up: vec![
                    gate_up_step(),
                    Step::GroupedMoeGemm {
                        experts: expert_ref_with(8),
                        which: MoeProj::GateUp {
                            up_out: synth_f32(4),
                        },
                        sorted_slot_index: synth_i64(32),
                        expert_tile_ids: synth_i64(8),
                        x: synth_f32(4),
                        y: synth_f32(160),
                        m_total: 160,
                        batch_size: 4,
                        k_top: 8,
                        dtype_tags: None,
                        force_mq4_fp16: false,
                        paro_i8: false,
                        paro_i8_k8: false,
                    },
                ],
                activation: vec![activation_step()],
                down: vec![Step::IndexedMoeGemv {
                    experts: expert_ref(),
                    which: MoeProj::DownExpanded,
                    topk_indices: synth_i64(4),
                    input: GemvInput::Raw(synth_f32(4)),
                    out,
                    k_top: 2,
                    batch_size: 1,
                }],
                combine: vec![Step::MoeCombine {
                    down_out: out,
                    topk_weights: synth_f32(4),
                    out: synth_f32(4),
                    k: 2,
                    hidden: 4,
                    batch_size: 1,
                    inverse_perm: None,
                }],
                finish: Vec::new(),
            }
        };
        let err = lower_moe_steps(&group, &policy, parts(vec![rank(), rank()])).unwrap_err();
        assert!(matches!(
            err,
            MoeLowerError::StrayGroupedOp { op: "gate_up", .. }
        ));

        // F32 self-combining with a stray gate-up unscatter.
        let rank = || {
            let mut phases = down_residual_f32_rank();
            phases.activation.push(Step::MoeGateUpUnscatter {
                y_grouped: synth_f32(4),
                sorted_slot_index: synth_i64(4),
                gate_batch: synth_f32(4),
                up_batch: synth_f32(4),
                inter: 4,
                k_top: 2,
                m_total: 16,
            });
            phases
        };
        let err = lower_moe_steps(&group, &policy, parts(vec![rank(), rank()])).unwrap_err();
        assert!(matches!(
            err,
            MoeLowerError::StrayGroupedOp {
                op: "gate_up_unscatter",
                ..
            }
        ));

        // I64 self-combining with a stray gate-up unscatter.
        let rank = || {
            let mut phases = i64_rank();
            phases.activation.push(Step::MoeGateUpUnscatter {
                y_grouped: synth_f32(4),
                sorted_slot_index: synth_i64(4),
                gate_batch: synth_f32(4),
                up_batch: synth_f32(4),
                inter: 4,
                k_top: 2,
                m_total: 16,
            });
            phases
        };
        let err = lower_moe_steps(&group, &policy, parts(vec![rank(), rank()])).unwrap_err();
        assert!(matches!(
            err,
            MoeLowerError::StrayGroupedOp {
                op: "gate_up_unscatter",
                ..
            }
        ));
    }

    #[test]
    fn lowering_rejects_duplicate_combine_across_phases() {
        let group = group(ExpertParallelism::TensorParallel, 2);
        let policy = tp_policy(2);
        let rank = || {
            let out = synth_f32(4);
            let combine =
                |down_out: &'static GpuTensor, out_t: &'static GpuTensor| Step::MoeCombine {
                    down_out,
                    topk_weights: synth_f32(4),
                    out: out_t,
                    k: 2,
                    hidden: 4,
                    batch_size: 1,
                    inverse_perm: None,
                };
            RoutedMoePhases {
                router: Vec::new(),
                gate_up: vec![gate_up_step()],
                activation: vec![activation_step()],
                down: vec![Step::IndexedMoeGemv {
                    experts: expert_ref(),
                    which: MoeProj::DownExpanded,
                    topk_indices: synth_i64(4),
                    input: GemvInput::Raw(synth_f32(4)),
                    out,
                    k_top: 2,
                    batch_size: 1,
                }],
                combine: vec![combine(out, synth_f32(4))],
                // A second combine hides in the finish phase.
                finish: vec![combine(out, synth_f32(4))],
            }
        };
        let err = lower_moe_steps(&group, &policy, parts(vec![rank(), rank()])).unwrap_err();
        assert!(matches!(err, MoeLowerError::DuplicateCombineOp { .. }));
    }

    #[test]
    fn lowering_rejects_combine_only_in_wrong_phase() {
        let group = group(ExpertParallelism::TensorParallel, 2);
        let policy = tp_policy(2);
        let rank = || {
            let out = synth_f32(4);
            RoutedMoePhases {
                router: Vec::new(),
                gate_up: vec![gate_up_step()],
                activation: vec![activation_step()],
                down: vec![Step::IndexedMoeGemv {
                    experts: expert_ref(),
                    which: MoeProj::DownExpanded,
                    topk_indices: synth_i64(4),
                    input: GemvInput::Raw(synth_f32(4)),
                    out,
                    k_top: 2,
                    batch_size: 1,
                }],
                combine: Vec::new(),
                finish: vec![Step::MoeCombine {
                    down_out: out,
                    topk_weights: synth_f32(4),
                    out: synth_f32(4),
                    k: 2,
                    hidden: 4,
                    batch_size: 1,
                    inverse_perm: None,
                }],
            }
        };
        let err = lower_moe_steps(&group, &policy, parts(vec![rank(), rank()])).unwrap_err();
        assert!(matches!(
            err,
            MoeLowerError::MisplacedCombineOp {
                phase: "finish",
                ..
            }
        ));
    }

    #[test]
    fn lowering_rejects_duplicate_down_across_phases() {
        let group = group(ExpertParallelism::TensorParallel, 2);
        let policy = tp_policy(2);
        let rank = || {
            let out = synth_f32(4);
            RoutedMoePhases {
                router: Vec::new(),
                gate_up: vec![gate_up_step()],
                activation: vec![
                    activation_step(),
                    Step::IndexedMoeGemv {
                        experts: expert_ref(),
                        which: MoeProj::DownResidual {
                            topk_weights: synth_f32(4),
                        },
                        topk_indices: synth_i64(4),
                        input: GemvInput::Raw(synth_f32(4)),
                        out: synth_f32(4),
                        k_top: 2,
                        batch_size: 1,
                    },
                ],
                down: vec![Step::IndexedMoeGemv {
                    experts: expert_ref(),
                    which: MoeProj::DownExpanded,
                    topk_indices: synth_i64(4),
                    input: GemvInput::Raw(synth_f32(4)),
                    out,
                    k_top: 2,
                    batch_size: 1,
                }],
                combine: vec![Step::MoeCombine {
                    down_out: out,
                    topk_weights: synth_f32(4),
                    out: synth_f32(4),
                    k: 2,
                    hidden: 4,
                    batch_size: 1,
                    inverse_perm: None,
                }],
                finish: Vec::new(),
            }
        };
        let err = lower_moe_steps(&group, &policy, parts(vec![rank(), rank()])).unwrap_err();
        assert!(matches!(
            err,
            MoeLowerError::DuplicateDownOp { phase: "down", .. }
        ));
    }

    #[test]
    fn lowering_rejects_down_only_in_wrong_phase() {
        let group = group(ExpertParallelism::TensorParallel, 2);
        let policy = tp_policy(2);
        let rank = || RoutedMoePhases {
            router: Vec::new(),
            gate_up: vec![gate_up_step()],
            activation: vec![
                activation_step(),
                Step::IndexedMoeGemv {
                    experts: expert_ref(),
                    which: MoeProj::DownResidual {
                        topk_weights: synth_f32(4),
                    },
                    topk_indices: synth_i64(4),
                    input: GemvInput::Raw(synth_f32(4)),
                    out: synth_f32(4),
                    k_top: 2,
                    batch_size: 1,
                },
            ],
            down: Vec::new(),
            combine: Vec::new(),
            finish: Vec::new(),
        };
        let err = lower_moe_steps(&group, &policy, parts(vec![rank(), rank()])).unwrap_err();
        assert!(matches!(
            err,
            MoeLowerError::MisplacedDownOp {
                phase: "activation",
                ..
            }
        ));
    }

    #[test]
    fn broad_lowering_errors_include_group_context() {
        // GroupPolicyMismatch and GroupSizeMismatch carry group/layer.
        let err = select_moe_executor(&group(ExpertParallelism::TensorParallel, 2), &ep_policy(2))
            .unwrap_err();
        let display = err.to_string();
        assert!(display.contains("test"));
        assert!(display.contains("Some(0)"));
        let err = select_moe_executor(&group(ExpertParallelism::TensorParallel, 3), &tp_policy(2))
            .unwrap_err();
        assert!(err.to_string().contains("test"));

        // MissingPhase carries group/layer.
        let mut no_gate_up = i64_rank();
        no_gate_up.gate_up.clear();
        let err = lower_moe_steps(
            &group(ExpertParallelism::Single, 1),
            &MoEExecutionPolicy::single(),
            parts(vec![no_gate_up]),
        )
        .unwrap_err();
        assert!(err.to_string().contains("test"));

        // RankCountMismatch carries group/layer.
        let err = lower_moe_steps(
            &group(ExpertParallelism::TensorParallel, 2),
            &tp_policy(2),
            parts(vec![i64_rank(), i64_rank(), i64_rank()]),
        )
        .unwrap_err();
        assert!(err.to_string().contains("test"));

        // RankPhaseMismatch carries group/layer and the rank.
        let mut rank_1 = i64_rank();
        rank_1.gate_up.push(gate_up_step());
        let err = lower_moe_steps(
            &group(ExpertParallelism::TensorParallel, 2),
            &tp_policy(2),
            parts(vec![i64_rank(), rank_1]),
        )
        .unwrap_err();
        let display = err.to_string();
        assert!(display.contains("test"));
        assert!(display.contains("Some(0)"));
        assert!(display.contains("rank 1"));
    }

    #[test]
    fn inspection_single_reports_exact_shape_and_no_schedule() {
        let group = group(ExpertParallelism::Single, 1);
        let policy = MoEExecutionPolicy::single();
        let lowered =
            lower_moe_steps(&group, &policy, parts(vec![down_residual_f32_rank()])).unwrap();
        assert_eq!(lowered.executor_kind(), MoeExecutorKind::SingleMesh);
        assert_eq!(lowered.rank_count(), 1);
        assert_eq!(lowered.step_count(0), Some(4));
        assert_eq!(lowered.step_count(1), None);
        assert_eq!(lowered.step_count(usize::MAX), None);
        // Single programs carry no parallel schedule vectors: collective and
        // zero_before are ALWAYS None, even for in-range rank/step.
        assert!(lowered.collective(0, 0).is_none());
        assert!(lowered.collective(0, 3).is_none());
        assert!(lowered.collective(1, 0).is_none());
        assert!(lowered.collective(usize::MAX, usize::MAX).is_none());
        assert!(lowered.zero_before(0, 0).is_none());
        assert!(lowered.zero_before(0, 3).is_none());
        assert!(lowered.zero_before(usize::MAX, usize::MAX).is_none());
    }

    #[test]
    fn inspection_tp_i64_reports_typed_collective_and_zero_vectors() {
        let group = group(ExpertParallelism::TensorParallel, 2);
        let policy = tp_policy(2);
        let lowered =
            lower_moe_steps(&group, &policy, parts(vec![i64_rank(), i64_rank()])).unwrap();
        assert_eq!(lowered.executor_kind(), MoeExecutorKind::Parallel);
        assert_eq!(lowered.rank_count(), 2);
        assert_eq!(lowered.step_count(0), Some(4));
        assert_eq!(lowered.step_count(1), Some(4));
        assert_eq!(lowered.step_count(2), None);
        // Complete typed collective sequence on every rank (the schedule
        // vector is shared): [None, None, AllReduceI64Tp{dim:4}, None].
        for rank in [0usize, 1] {
            assert!(matches!(
                lowered.collective(rank, 0),
                Some(StepCollective::None)
            ));
            assert!(matches!(
                lowered.collective(rank, 1),
                Some(StepCollective::None)
            ));
            assert!(matches!(
                lowered.collective(rank, 2),
                Some(StepCollective::AllReduceI64Tp { dim: 4 })
            ));
            assert!(matches!(
                lowered.collective(rank, 3),
                Some(StepCollective::None)
            ));
        }
        // Complete zero vector: only the i64 accumulator step is pre-zeroed.
        assert_eq!(lowered.zero_before(0, 0), Some(false));
        assert_eq!(lowered.zero_before(0, 1), Some(false));
        assert_eq!(lowered.zero_before(0, 2), Some(true));
        assert_eq!(lowered.zero_before(0, 3), Some(false));
        assert_eq!(lowered.zero_before(1, 2), Some(true));
        // Exactly one AllReduceI64Tp and exactly one zero across the schedule.
        let mut collectives = 0usize;
        let mut zeroes = 0usize;
        for step in 0..4 {
            if matches!(
                lowered.collective(0, step),
                Some(StepCollective::AllReduceI64Tp { .. })
            ) {
                collectives += 1;
            }
            if lowered.zero_before(0, step) == Some(true) {
                zeroes += 1;
            }
        }
        assert_eq!(collectives, 1);
        assert_eq!(zeroes, 1);
    }

    #[test]
    fn inspection_ep_local_i64_reports_zero_and_ep_conversion_sequence() {
        let group = group(ExpertParallelism::ExpertParallel, 2);
        let policy = ep_policy(2);
        let lowered =
            lower_moe_steps(&group, &policy, parts(vec![i64_rank(), i64_rank()])).unwrap();
        assert_eq!(lowered.executor_kind(), MoeExecutorKind::Parallel);
        assert_eq!(lowered.rank_count(), 2);
        // Complete typed sequence on every rank: [None, None, ZeroI64Only,
        // AllReduce{Ep}] — the local i64 zeroing at the down step, the FP32 EP
        // all-reduce at the conversion step.
        for rank in [0usize, 1] {
            assert!(matches!(
                lowered.collective(rank, 0),
                Some(StepCollective::None)
            ));
            assert!(matches!(
                lowered.collective(rank, 1),
                Some(StepCollective::None)
            ));
            assert!(matches!(
                lowered.collective(rank, 2),
                Some(StepCollective::ZeroI64Only { dim: 4 })
            ));
            assert!(matches!(
                lowered.collective(rank, 3),
                Some(StepCollective::AllReduce {
                    kind: DimKind::Ep,
                    dim: 4
                })
            ));
        }
        // Complete zero vector: only the local i64 accumulator is pre-zeroed;
        // the EP all-reduce lands on the conversion which writes fresh.
        assert_eq!(lowered.zero_before(0, 0), Some(false));
        assert_eq!(lowered.zero_before(0, 1), Some(false));
        assert_eq!(lowered.zero_before(0, 2), Some(true));
        assert_eq!(lowered.zero_before(0, 3), Some(false));
        assert_eq!(lowered.zero_before(1, 2), Some(true));
        assert_eq!(lowered.zero_before(1, 3), Some(false));
        // Exactly one ZeroI64Only and exactly one Ep all-reduce.
        let mut zeros = 0usize;
        let mut ep_reduces = 0usize;
        for step in 0..4 {
            if matches!(
                lowered.collective(0, step),
                Some(StepCollective::ZeroI64Only { .. })
            ) {
                zeros += 1;
            }
            if matches!(
                lowered.collective(0, step),
                Some(StepCollective::AllReduce {
                    kind: DimKind::Ep,
                    ..
                })
            ) {
                ep_reduces += 1;
            }
        }
        assert_eq!(zeros, 1);
        assert_eq!(ep_reduces, 1);
    }

    #[test]
    fn inspection_reflects_private_inner_exactly() {
        // Parallel: every accessor must mirror the private stored vectors,
        // and every returned collective ref must point at the shared stored
        // element (the same ref serves every rank).
        let tp_group = group(ExpertParallelism::TensorParallel, 2);
        let policy = tp_policy(2);
        let lowered =
            lower_moe_steps(&tp_group, &policy, parts(vec![i64_rank(), i64_rank()])).unwrap();
        let LoweredMoeProgramInner::Parallel {
            per_rank_steps,
            collectives,
            zero_before,
            ..
        } = &lowered.inner
        else {
            panic!("expected parallel program");
        };
        assert_eq!(lowered.rank_count(), per_rank_steps.len());
        for rank in 0..lowered.rank_count() {
            assert_eq!(lowered.step_count(rank), Some(per_rank_steps[rank].len()));
            for step in 0..per_rank_steps[rank].len() {
                let collective = lowered.collective(rank, step).expect("in-range collective");
                assert!(
                    std::ptr::eq(collective, &collectives[step]),
                    "collective ref must be the shared stored element"
                );
                assert_eq!(
                    lowered.zero_before(rank, step),
                    Some(zero_before[step]),
                    "zero flag must mirror the private vector"
                );
            }
        }
        // Cross-rank ref identity: the same stored element serves every rank.
        for step in 0..per_rank_steps[0].len() {
            assert!(std::ptr::eq(
                lowered.collective(0, step).unwrap(),
                lowered.collective(1, step).unwrap(),
            ));
        }

        // Single: step_count mirrors the private single steps vector, and
        // schedule accessors are always None.
        let single_group = group(ExpertParallelism::Single, 1);
        let policy = MoEExecutionPolicy::single();
        let lowered = lower_moe_steps(
            &single_group,
            &policy,
            parts(vec![down_residual_f32_rank()]),
        )
        .unwrap();
        let LoweredMoeProgramInner::Single { steps } = &lowered.inner else {
            panic!("expected single program");
        };
        assert_eq!(lowered.step_count(0), Some(steps.len()));
        assert!(lowered.collective(0, 0).is_none());
        assert!(lowered.zero_before(0, 0).is_none());
    }

    #[test]
    fn inspection_bounds_never_panic_and_return_none() {
        let tp_group = group(ExpertParallelism::TensorParallel, 2);
        let policy = tp_policy(2);
        let lowered =
            lower_moe_steps(&tp_group, &policy, parts(vec![i64_rank(), i64_rank()])).unwrap();
        // rank == rank_count.
        assert_eq!(lowered.step_count(2), None);
        assert!(lowered.collective(2, 0).is_none());
        assert!(lowered.zero_before(2, 0).is_none());
        // step == step_count(rank).
        assert!(lowered.collective(0, 4).is_none());
        assert!(lowered.zero_before(0, 4).is_none());
        assert!(lowered.collective(1, 4).is_none());
        assert!(lowered.zero_before(1, 4).is_none());
        // usize::MAX on every axis, alone and combined.
        assert_eq!(lowered.step_count(usize::MAX), None);
        assert!(lowered.collective(usize::MAX, 0).is_none());
        assert!(lowered.collective(0, usize::MAX).is_none());
        assert!(lowered.collective(usize::MAX, usize::MAX).is_none());
        assert!(lowered.zero_before(usize::MAX, 0).is_none());
        assert!(lowered.zero_before(0, usize::MAX).is_none());
        assert!(lowered.zero_before(usize::MAX, usize::MAX).is_none());

        // Single program: identical no-panic bounds behavior.
        let single_group = group(ExpertParallelism::Single, 1);
        let policy = MoEExecutionPolicy::single();
        let lowered = lower_moe_steps(
            &single_group,
            &policy,
            parts(vec![down_residual_f32_rank()]),
        )
        .unwrap();
        assert_eq!(lowered.step_count(1), None);
        assert_eq!(lowered.step_count(usize::MAX), None);
        assert!(lowered.collective(0, 4).is_none());
        assert!(lowered.zero_before(0, 4).is_none());
        assert!(lowered.collective(usize::MAX, usize::MAX).is_none());
        assert!(lowered.zero_before(usize::MAX, usize::MAX).is_none());
    }

    #[test]
    fn inspection_parallel_schedule_missing_step_never_panics() {
        // Malformed inner: the per-rank step vector outruns the shared
        // schedule vectors. Lowering can never produce this (rank vectors are
        // validated to equal lengths and the schedule is derived from those
        // lengths), but the immutable inspection surface must still never
        // panic on the schedule-missing step.
        let policy = tp_policy(2);
        let malformed = LoweredMoeProgram {
            inner: LoweredMoeProgramInner::Parallel {
                mesh: policy.mesh(),
                per_rank_steps: vec![
                    (0..5).map(|_| gate_up_step()).collect(),
                    (0..5).map(|_| gate_up_step()).collect(),
                ],
                collectives: (0..4).map(|_| StepCollective::None).collect(),
                zero_before: vec![false; 4],
            },
        };
        // In-range rank/step for which the schedule vector has no entry:
        // None, never a panic.
        assert!(malformed.collective(0, 4).is_none());
        assert!(malformed.collective(1, 4).is_none());
        assert!(malformed.zero_before(0, 4).is_none());
        assert!(malformed.zero_before(1, 4).is_none());
        // Steps that DO have a schedule entry stay Some.
        assert!(matches!(
            malformed.collective(0, 0),
            Some(StepCollective::None)
        ));
        assert!(matches!(
            malformed.collective(1, 3),
            Some(StepCollective::None)
        ));
        assert_eq!(malformed.zero_before(0, 0), Some(false));
        assert_eq!(malformed.zero_before(1, 3), Some(false));
        // Bounds beyond the rank vector stay None.
        assert!(malformed.collective(0, 5).is_none());
        assert!(malformed.zero_before(0, 5).is_none());
        assert!(malformed.collective(2, 0).is_none());
        assert!(malformed.zero_before(2, 0).is_none());
        assert!(malformed.collective(usize::MAX, usize::MAX).is_none());
        assert!(malformed.zero_before(usize::MAX, usize::MAX).is_none());

        // A properly lowered valid schedule is unaffected.
        let tp_group = group(ExpertParallelism::TensorParallel, 2);
        let lowered =
            lower_moe_steps(&tp_group, &policy, parts(vec![i64_rank(), i64_rank()])).unwrap();
        assert_eq!(lowered.step_count(0), Some(4));
        for step in 0..4 {
            assert!(lowered.collective(0, step).is_some());
            assert!(lowered.zero_before(0, step).is_some());
        }
    }
}
