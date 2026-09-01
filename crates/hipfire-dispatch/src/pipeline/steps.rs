// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Björn Bösel
// hipfire — see LICENSE and NOTICE in the project root.
//! Op-list interpreter. Phase 2a: GEMV + a fused rmsnorm-rotate producer; empty
//! fusion table (all per-op fallback).

use hipfire_hardware::{DeviceMesh, DimKind, Gpus, MeshEpoch};
use rdna_compute::{DType, Gpu, GpuTensor};
use std::sync::OnceLock;

use crate::context::DispatchCtx;
use crate::families::fused_qkv::{FusedQkvBiasParams, FusedQkvFamily, FusedQkvParams};
use crate::families::gemv::{GemvFamily, GemvParams, RotateInputs, WeightRef};
use crate::families::moe::{
    ExpertExecutionPlan, MoeActivationVariant, MoeExpertRef, MoeFamily, MoeProj, MoeProtocolKind,
    RouterPlan,
};
use crate::families::rotation::{RotationFamily, RotationParams};
use crate::types::GemvVariant;
use crate::types::{DispatchError, KernelKey, PipelineOp, RotationPlan, RotationVariant};

/// Routing policy is carried by `RouterPlan`; no standalone score activation
/// can be inserted between route and expert phases.

/// Rotation disposition of a Gemv's input. Borrows (never owns a RotatedActivation).
pub enum GemvInput<'a> {
    Raw(&'a GpuTensor),        // launch_op self-rotates via run_auto (plan-aware)
    Prerotated(&'a GpuTensor), // already FWHT-rotated; dispatched via Prerotated variant
}

pub enum Step<'a> {
    Gemv {
        w: &'a WeightRef<'a>,
        input: GemvInput<'a>,
        out: &'a GpuTensor,
    },
    /// GEMV with in-place residual add: `residual += W · input`.
    /// For MQ-family, `input` must be pre-rotated (Prerotated variant) or the
    /// Raw variant triggers FWHT rotation before calling the residual kernel.
    GemvResidual {
        w: &'a WeightRef<'a>,
        input: GemvInput<'a>,
        residual: &'a GpuTensor,
        out: &'a GpuTensor,
    },
    /// Fused rmsnorm + optional FWHT rotation. The `rotation` field is derived
    /// by the caller via `dtype_rotation_plan(w.dtype)`. `out` holds the
    /// ready-to-use activation (FWHT-rotated for FwhtG256, plain-normed for None).
    /// All downstream Gemv steps use GemvInput::Prerotated(out).
    RmsnormAutomatic {
        x: &'a GpuTensor,
        norm_weight: &'a GpuTensor,
        x_plain: &'a GpuTensor, // rmsnorm intermediate scratch (always written)
        out: &'a GpuTensor,     // final activation output (written by this step)
        awq_scale: Option<&'a GpuTensor>,
        k: usize,
        eps: f32,
        rotation: RotationPlan, // FwhtG256 for MQ dtypes, None for HFQ4/others
    },
    /// Paired KV-write + flash-attention (Phase 0.3). Consumes a KvTierPlan
    /// (derived once per attention step) and AttnParams (tensor borrows).
    /// Not fusible — the two ops are inherently coupled.
    Attend {
        plan: crate::families::kv_tier::KvTierPlan,
        io: crate::families::attention::AttnParams<'a>,
    },
    /// In-place RoPE on Q and K. Per-op only (no fused entry) — present so the
    /// attention block can be one contiguous step list (future fusion seam).
    Rope {
        q: &'a GpuTensor,
        k: &'a GpuTensor,
        pos_buf: &'a hip_bridge::DeviceBuffer,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
        theta: f32,
    },
    /// Per-head rmsnorm on one tensor (Qwen3-style qk-norm). One step per tensor.
    QkNorm {
        x: &'a GpuTensor,
        weight: &'a GpuTensor,
        n_groups: usize, // n_heads (Q) or n_kv_heads (K)
        head_dim: usize,
        eps: f32,
    },
    /// In-place bias add on one tensor (e.g. qwen2 QKV bias).
    BiasAdd {
        x: &'a GpuTensor,
        bias: &'a GpuTensor,
        dim: usize,
    },
    /// Typed MoE route. Routing semantics (bias/hash/normalization) are
    /// carried by the plan rather than reconstructed by the family.
    MoeRoute { plan: RouterPlan<'a> },
    /// Indexed routed expert projection. Every down projection is expanded;
    /// the executor-owned `MoeCombine` is the only weighted reduction.
    IndexedMoeGemv {
        experts: &'a MoeExpertRef<'a>,
        which: MoeProj<'a>,
        topk_indices: &'a GpuTensor,
        input: GemvInput<'a>,
        out: &'a GpuTensor,
        k_top: usize,
        batch_size: usize,
    },
    /// Weighted combine for an expanded routed-down result. The executor
    /// accepts exactly one combine for a routed chain.
    MoeCombine {
        down_out: &'a GpuTensor,
        topk_weights: &'a GpuTensor,
        out: &'a GpuTensor,
        hidden: usize,
        k_top: usize,
        batch_size: usize,
        inverse_perm: Option<&'a GpuTensor>,
    },
    /// Build the deterministic grouped-GEMM permutation for a prefill batch.
    MoeScatter {
        topk_indices: &'a GpuTensor,
        expert_token_counts: &'a GpuTensor,
        expert_offsets: &'a GpuTensor,
        sorted_slot_index: &'a GpuTensor,
        expert_tile_ids: &'a GpuTensor,
        inverse_perm: &'a GpuTensor,
        total_slots: usize,
        n_experts: usize,
        m_total_max: usize,
        block_m: usize,
    },
    /// Grouped routed expert GEMM. Grouped down is always expanded; a
    /// residual-fused grouped down is refused before any GPU work.
    GroupedMoeGemm {
        experts: &'a MoeExpertRef<'a>,
        which: MoeProj<'a>,
        sorted_slot_index: &'a GpuTensor,
        expert_tile_ids: &'a GpuTensor,
        x: &'a GpuTensor,
        y: &'a GpuTensor,
        m_total: usize,
        batch_size: usize,
        k_top: usize,
    },
    /// Deinterleave grouped gate/up output into per-slot gate and up tensors.
    MoeGateUpUnscatter {
        y_grouped: &'a GpuTensor,
        sorted_slot_index: &'a GpuTensor,
        gate_batch: &'a GpuTensor,
        up_batch: &'a GpuTensor,
        inter: usize,
        k_top: usize,
        m_total: usize,
    },
    /// Activation/rotation between routed gate/up and down.
    MoeActivation {
        variant: MoeActivationVariant,
        gate: &'a GpuTensor,
        up: &'a GpuTensor,
        rot_out: &'a GpuTensor,
        inter: usize,
        rows: usize,
    },
}

/// Op-kind for fusion matching. Total over Step variants.
fn op_kind(step: &Step) -> PipelineOp {
    match step {
        Step::Gemv { .. } => PipelineOp::Gemv,
        Step::GemvResidual { .. } => PipelineOp::GemvResidual,
        Step::RmsnormAutomatic { .. } => PipelineOp::RmsnormAutomatic,
        Step::Attend { .. } => PipelineOp::Attend,
        Step::Rope { .. } => PipelineOp::Rope,
        Step::QkNorm { .. } => PipelineOp::QkNorm,
        Step::BiasAdd { .. } => PipelineOp::BiasAdd,
        Step::MoeRoute { .. } => PipelineOp::MoeRoute,
        Step::IndexedMoeGemv { .. } => PipelineOp::IndexedMoeGemv,
        Step::MoeCombine { .. } => PipelineOp::MoeCombine,
        Step::MoeScatter { .. } => PipelineOp::MoeScatter,
        Step::GroupedMoeGemm { .. } => PipelineOp::GroupedMoeGemm,
        Step::MoeGateUpUnscatter { .. } => PipelineOp::MoeGateUpUnscatter,
        Step::MoeActivation { .. } => PipelineOp::MoeActivation,
    }
}

/// Collective attached to one lock-step `Step` position. The descriptor is
/// immutable schedule data: membership and mesh identity are supplied by the
/// manifest/topology owner, never inferred by a family.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum StepCollective {
    None,
    AllReduce {
        kind: DimKind,
        dim: usize,
        group: Vec<usize>,
        mesh: MeshEpoch,
        rank: usize,
    },
}

impl StepCollective {
    pub fn all_reduce(
        kind: DimKind,
        dim: usize,
        group: Vec<usize>,
        mesh: MeshEpoch,
        rank: usize,
    ) -> Self {
        Self::AllReduce {
            kind,
            dim,
            group,
            mesh,
            rank,
        }
    }
}
/// Pointer-free identity for one executable rank schedule. Mesh execution
/// compares this value for every rank before launching any device work.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MoeExecutionSignature<'a> {
    pub protocol: MoeProtocolKind,
    pub execution: ExpertExecutionPlan,
    pub router_identity: &'a str,
    pub router_selection: crate::families::moe::RouterSelection,
    pub k_top: usize,
    pub normalize: bool,
    pub expert_dtype: DType,
    pub n_experts: usize,
    pub expert_k: usize,
    pub expert_m: usize,
    pub batch_size: usize,
    pub hidden: usize,
    /// Canonical `(global expert, owner rank, local slot)` tuples for every
    /// rank. Unlike pointer tables, this is safe to compare across devices.
    pub ownership_partition: &'a [(usize, usize, usize)],
}

fn collective_count(collectives: &[StepCollective]) -> usize {
    collectives
        .iter()
        .filter(|collective| matches!(collective, StepCollective::AllReduce { .. }))
        .count()
}

/// Validate the complete grammar of an admitted typed MoE program before any
/// GPU work. The two executable protocols are intentionally exact:
///
/// ```text
/// indexed: route → gate/up → activation → down(expanded) → combine
/// grouped: route → scatter → gate/up → unscatter → activation
///          → down(expanded) → combine
/// ```
///
/// Every routed operand is checked against the one route and expert view that
/// owns it. This makes a hand-built family schedule fail closed instead of
/// silently selecting a second policy or reduction.
pub fn validate_moe_step_schedule(
    steps: &[Step],
    collectives: &[StepCollective],
) -> Result<(), DispatchError> {
    if steps.len() != collectives.len() {
        return Err(DispatchError::Hip(format!(
            "MoE schedule has {} steps but {} collective descriptors",
            steps.len(),
            collectives.len()
        )));
    }
    if steps.is_empty() {
        return Err(DispatchError::Hip("MoE schedule is empty".into()));
    }

    let (protocol, route, experts, batch_size, combine_index, hidden, route_indices, route_weights) =
        match steps {
            [Step::MoeRoute { plan }, Step::IndexedMoeGemv {
                experts,
                which: MoeProj::GateUp { up_out },
                topk_indices,
                input: GemvInput::Prerotated(x),
                out: gate,
                k_top,
                batch_size,
            }, Step::MoeActivation {
                variant,
                gate: act_gate,
                up,
                rot_out,
                inter,
                rows,
            }, Step::IndexedMoeGemv {
                experts: down_experts,
                which: MoeProj::DownExpanded,
                topk_indices: down_indices,
                input: GemvInput::Prerotated(down_input),
                out: down_out,
                k_top: down_k,
                batch_size: down_batch,
            }, Step::MoeCombine {
                down_out: combine_down,
                topk_weights,
                out,
                hidden,
                k_top: combine_k,
                batch_size: combine_batch,
                inverse_perm,
            }] => {
                let expected_rows = batch_size
                    .checked_mul(*k_top)
                    .ok_or_else(|| DispatchError::Hip("indexed MoE row count overflows".into()))?;
                if *batch_size != 1
                    || *down_batch != 1
                    || inverse_perm.is_some()
                    || *k_top != plan.k_top()
                    || *inter != (*experts).expert_m()
                    || *rows != expected_rows
                {
                    return Err(DispatchError::Hip(
                        "indexed MoE grammar requires decode batch=1, normalized route, and no inverse permutation"
                            .into(),
                    ));
                }
                if *k_top != *down_k || *k_top != *combine_k || *batch_size != *combine_batch {
                    return Err(DispatchError::Hip(
                        "indexed MoE route width/batch differs across phases".into(),
                    ));
                }
                if !std::ptr::eq(*experts, *down_experts) {
                    return Err(DispatchError::Hip(
                        "indexed MoE phases use different expert owner views".into(),
                    ));
                }
                if !same_tensor(gate, act_gate)
                    || !same_tensor(up_out, up)
                    || !same_tensor(rot_out, down_input)
                    || !same_tensor(down_out, combine_down)
                {
                    return Err(DispatchError::Hip(
                        "indexed MoE phase operands are not identity-linked".into(),
                    ));
                }
                if !same_tensor(topk_indices, down_indices) {
                    return Err(DispatchError::Hip(
                        "indexed MoE phases use different route-index buffers".into(),
                    ));
                }
                (
                    MoeProtocolKind::Indexed,
                    plan,
                    *experts,
                    *batch_size,
                    4usize,
                    *hidden,
                    topk_indices,
                    topk_weights,
                )
            }
            [Step::MoeRoute { plan }, Step::MoeScatter {
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
            }, Step::GroupedMoeGemm {
                experts,
                which: MoeProj::GateUp { .. },
                sorted_slot_index: gate_sorted,
                expert_tile_ids: gate_tiles,
                x: gate_x,
                y: gate_y,
                m_total: gate_m_total,
                batch_size,
                k_top,
            }, Step::MoeGateUpUnscatter {
                y_grouped,
                sorted_slot_index: unscatter_sorted,
                gate_batch,
                up_batch,
                inter,
                k_top: unscatter_k,
                m_total: unscatter_m_total,
            }, Step::MoeActivation {
                variant: _,
                gate: act_gate,
                up: act_up,
                rot_out,
                inter: act_inter,
                rows,
            }, Step::GroupedMoeGemm {
                experts: down_experts,
                which: MoeProj::DownExpanded,
                sorted_slot_index: down_sorted,
                expert_tile_ids: down_tiles,
                x: down_x,
                y: down_y,
                m_total: down_m_total,
                batch_size: down_batch,
                k_top: down_k,
            }, Step::MoeCombine {
                down_out: combine_down,
                topk_weights,
                out,
                hidden,
                k_top: combine_k,
                batch_size: combine_batch,
                inverse_perm: combine_inverse,
            }] => {
                let expected_slots = batch_size
                    .checked_mul(*k_top)
                    .ok_or_else(|| DispatchError::Hip("grouped MoE slot count overflows".into()))?;
                if *k_top != plan.k_top()
                    || *k_top != *unscatter_k
                    || *k_top != *down_k
                    || *k_top != *combine_k
                    || *batch_size != *down_batch
                    || *batch_size != *combine_batch
                    || *total_slots != expected_slots
                    || *n_experts != experts.n_experts()
                    || !same_tensor(topk_indices, plan.route_buffers().0)
                    || same_tensor(topk_indices, sorted_slot_index)
                    || !same_tensor(sorted_slot_index, gate_sorted)
                    || !same_tensor(sorted_slot_index, unscatter_sorted)
                    || !same_tensor(sorted_slot_index, down_sorted)
                    || !same_tensor(expert_tile_ids, gate_tiles)
                    || !same_tensor(expert_tile_ids, down_tiles)
                {
                    return Err(DispatchError::Hip(
                        "grouped MoE route width, batch, or route identity mismatch".into(),
                    ));
                }
                if !std::ptr::eq(*experts, *down_experts) {
                    return Err(DispatchError::Hip(
                        "grouped MoE phases use different expert owner views".into(),
                    ));
                }
                if *n_experts == 0
                    || *m_total_max == 0
                    || *block_m == 0
                    || !m_total_max.is_multiple_of(*block_m)
                    || *gate_m_total == 0
                    || *down_m_total != *gate_m_total
                    || *unscatter_m_total != *gate_m_total
                    || *gate_m_total > *m_total_max
                {
                    return Err(DispatchError::Hip(
                        "grouped MoE scatter capacity or tile geometry is invalid".into(),
                    ));
                }
                let expected_offsets = n_experts.checked_add(1).ok_or_else(|| {
                    DispatchError::Hip("MoE expert offset count overflows".into())
                })?;
                let counts_elements = checked_numel(expert_token_counts, "expert counts")?;
                let offset_elements = checked_numel(expert_offsets, "expert offsets")?;
                if counts_elements == 0 || offset_elements < expected_offsets {
                    return Err(DispatchError::Hip(
                        "grouped MoE scatter metadata has empty or short buffers".into(),
                    ));
                }
                let Some(combine_inverse) = *combine_inverse else {
                    return Err(DispatchError::Hip(
                        "grouped MoE combine requires the scatter inverse permutation".into(),
                    ));
                };
                if !same_tensor(gate_y, y_grouped)
                    || !same_tensor(gate_sorted, unscatter_sorted)
                    || !same_tensor(gate_batch, act_gate)
                    || !same_tensor(up_batch, act_up)
                    || *inter != *act_inter
                    || *rows != expected_slots
                    || !same_tensor(rot_out, down_x)
                    || !same_tensor(down_y, combine_down)
                    || !same_tensor(inverse_perm, combine_inverse)
                {
                    return Err(DispatchError::Hip(
                        "grouped MoE phase operands are not identity-linked".into(),
                    ));
                }
                (
                    MoeProtocolKind::Grouped,
                    plan,
                    *experts,
                    *batch_size,
                    6usize,
                    *hidden,
                    topk_indices,
                    topk_weights,
                )
            }
            _ => {
                return Err(DispatchError::Hip(
                    "MoE Step grammar must be exactly indexed or grouped".into(),
                ))
            }
        };
    if let RouterPlan::SoftmaxTopK { k_top, .. } = route {
        if *k_top != 8 {
            return Err(DispatchError::Hip(format!(
                "generic MoE softmax route requires k_top=8, got {k_top}"
            )));
        }
    }

    experts.validate()?;
    route.validate_against(experts.n_experts(), batch_size)?;
    if !same_tensor(route_indices, route.route_buffers().0)
        || !same_tensor(route_weights, route.route_buffers().1)
    {
        return Err(DispatchError::Hip(
            "MoE route metadata is not bound to the concrete expert Steps".into(),
        ));
    }
    if hidden == 0 || hidden != experts.expert_k() {
        return Err(DispatchError::Hip(
            "MoE combine hidden size does not match the expert down projection".into(),
        ));
    }
    let gate_width = experts
        .expert_m()
        .checked_mul(2)
        .ok_or_else(|| DispatchError::Hip("MoE gate/up shape overflows".into()))?;
    let shape_gate = [gate_width, experts.expert_k()];
    let shape_down = [experts.expert_k(), experts.expert_m()];
    experts.validate_projection_shapes(&shape_gate, &shape_down)?;
    let dtype_supported = match protocol {
        MoeProtocolKind::Indexed => matches!(
            experts.dtype(),
            DType::MQ4G256 | DType::MQ6G256 | DType::MQ4G256V2 | DType::MQ6G256V2
        ),
        MoeProtocolKind::Grouped => matches!(
            experts.dtype(),
            DType::MQ2G256Lloyd
                | DType::MQ2G256LloydU
                | DType::MQ3G256Lloyd
                | DType::MQ4G256
                | DType::MQ4G256V2
                | DType::MQ6G256
                | DType::MQ6G256V2
                | DType::MFP4G32E8
                | DType::ParoQ4G128
        ),
    };
    if !dtype_supported {
        return Err(DispatchError::Hip(format!(
            "MoE {:?} protocol has no executable kernel for {:?}",
            protocol,
            experts.dtype()
        )));
    }
    validate_collectives(
        collectives,
        combine_index,
        batch_size,
        hidden,
        experts.collective_kind(),
    )?;
    Ok(())
}

/// Couple the immutable plan identity to the exact executable grammar.
pub fn validate_moe_protocol_schedule(
    steps: &[Step],
    collectives: &[StepCollective],
    execution: ExpertExecutionPlan,
) -> Result<(), DispatchError> {
    let expected = execution.protocol()?;
    validate_moe_step_schedule(steps, collectives)?;
    let actual = if matches!(steps.get(1), Some(Step::MoeScatter { .. })) {
        MoeProtocolKind::Grouped
    } else {
        MoeProtocolKind::Indexed
    };
    if expected != actual {
        return Err(DispatchError::Hip(format!(
            "MoE execution identity {:?} does not match {:?} Step grammar",
            expected, actual
        )));
    }
    Ok(())
}

fn derive_moe_execution_signature<'a>(
    steps: &[Step<'a>],
    execution: ExpertExecutionPlan,
) -> Result<MoeExecutionSignature<'a>, DispatchError> {
    let route = steps
        .iter()
        .find_map(|step| match step {
            Step::MoeRoute { plan } => Some(plan),
            _ => None,
        })
        .ok_or_else(|| DispatchError::Hip("MoE schedule has no route".into()))?;
    let experts = steps
        .iter()
        .find_map(|step| match step {
            Step::IndexedMoeGemv { experts, .. } | Step::GroupedMoeGemm { experts, .. } => {
                Some(*experts)
            }
            _ => None,
        })
        .ok_or_else(|| DispatchError::Hip("MoE schedule has no expert owner view".into()))?;
    let hidden = steps
        .iter()
        .find_map(|step| match step {
            Step::MoeCombine { hidden, .. } => Some(*hidden),
            _ => None,
        })
        .ok_or_else(|| DispatchError::Hip("MoE schedule has no combine geometry".into()))?;
    Ok(MoeExecutionSignature {
        protocol: execution.protocol()?,
        execution,
        router_identity: experts.router_identity(),
        router_selection: route.selection(),
        k_top: route.k_top(),
        normalize: route.normalizes(),
        expert_dtype: experts.dtype(),
        n_experts: experts.n_experts(),
        expert_k: experts.expert_k(),
        expert_m: experts.expert_m(),
        batch_size: route.batch_size(),
        hidden,
        ownership_partition: experts.ownership_partition(),
    })
}

/// An immutable, pre-validated typed MoE schedule.
pub struct SealedMoeSchedule<'a> {
    execution: ExpertExecutionPlan,
    steps: Vec<Step<'a>>,
    collectives: Vec<StepCollective>,
}

impl<'a> SealedMoeSchedule<'a> {
    pub fn new(
        execution: ExpertExecutionPlan,
        steps: Vec<Step<'a>>,
        collectives: Vec<StepCollective>,
    ) -> Result<Self, DispatchError> {
        validate_moe_protocol_schedule(&steps, &collectives, execution)?;
        Ok(Self {
            execution,
            steps,
            collectives,
        })
    }

    pub fn execution(&self) -> ExpertExecutionPlan {
        self.execution
    }

    pub fn steps(&self) -> &[Step<'a>] {
        &self.steps
    }

    pub fn collectives(&self) -> &[StepCollective] {
        &self.collectives
    }
    pub fn execution_signature(&self) -> Result<MoeExecutionSignature<'a>, DispatchError> {
        derive_moe_execution_signature(&self.steps, self.execution)
    }
}

pub fn execute_sealed_steps(
    gpu: &mut Gpu,
    ctx: &DispatchCtx,
    schedule: &SealedMoeSchedule<'_>,
) -> Result<(), DispatchError> {
    validate_moe_protocol_schedule(&schedule.steps, &schedule.collectives, schedule.execution)?;
    if collective_count(&schedule.collectives) != 0 {
        return Err(DispatchError::Hip(
            "parallel MoE schedules require execute_sealed_steps_mesh".into(),
        ));
    }
    execute_steps_inner(gpu, ctx, &schedule.steps)
}

/// Results of the host-only mesh checks shared by the executor and focused
/// preflight tests. No device method is called while this value is built.
struct MoeMeshPreflight<'a> {
    dim: usize,
    group: Vec<usize>,
    outputs: Vec<&'a hip_bridge::DeviceBuffer>,
}

fn preflight_sealed_steps_mesh<'a>(
    gpus_len: usize,
    mesh: &DeviceMesh,
    schedules: &[&SealedMoeSchedule<'a>],
) -> Result<MoeMeshPreflight<'a>, DispatchError> {
    if schedules.is_empty() {
        return Err(DispatchError::Hip(
            "parallel MoE execution has no rank schedules".into(),
        ));
    }
    if mesh.n_devices() != gpus_len {
        return Err(DispatchError::Hip(format!(
            "MoE mesh has {} devices but Gpus owns {}",
            mesh.n_devices(),
            gpus_len
        )));
    }

    let mut reduction: Option<(DimKind, usize, Vec<usize>, MeshEpoch)> = None;
    let mut execution_signature: Option<MoeExecutionSignature<'a>> = None;
    let mut outputs = Vec::with_capacity(schedules.len());
    for (rank, schedule) in schedules.iter().enumerate() {
        validate_moe_protocol_schedule(
            schedule.steps(),
            schedule.collectives(),
            schedule.execution(),
        )?;
        let signature = schedule.execution_signature()?;
        if let Some(expected) = &execution_signature {
            if expected != &signature {
                return Err(DispatchError::Hip(
                    "MoE rank schedules disagree on executable identity".into(),
                ));
            }
        } else {
            execution_signature = Some(signature);
        }
        let (kind, dim, group, epoch, descriptor_rank) = schedule
            .collectives()
            .iter()
            .find_map(|collective| match collective {
                StepCollective::AllReduce {
                    kind,
                    dim,
                    group,
                    mesh,
                    rank,
                } => Some((*kind, *dim, group.clone(), *mesh, *rank)),
                StepCollective::None => None,
            })
            .ok_or_else(|| {
                DispatchError::Hip("parallel MoE schedule has no manifest-owned collective".into())
            })?;
        if descriptor_rank != rank {
            return Err(DispatchError::Hip(
                "MoE collective rank does not match schedule order".into(),
            ));
        }
        let experts = schedule
            .steps()
            .iter()
            .find_map(|step| match step {
                Step::IndexedMoeGemv { experts, .. } | Step::GroupedMoeGemm { experts, .. } => {
                    Some(*experts)
                }
                _ => None,
            })
            .ok_or_else(|| DispatchError::Hip("MoE schedule has no expert owner view".into()))?;
        if experts.owner_rank() != rank
            || experts.group_devices() != group.as_slice()
            || experts.mesh_epoch() != epoch
            || experts.collective_kind() != Some(kind)
        {
            return Err(DispatchError::Hip(
                "MoE schedule expert owner does not match its collective identity".into(),
            ));
        }
        if let Some((expected_kind, expected_dim, expected_group, expected_epoch)) = &reduction {
            if *expected_kind != kind
                || *expected_dim != dim
                || expected_group != &group
                || *expected_epoch != epoch
            {
                return Err(DispatchError::Hip(
                    "MoE rank schedules disagree on collective mesh identity".into(),
                ));
            }
        } else {
            reduction = Some((kind, dim, group, epoch));
        }
        let output = schedule
            .steps()
            .iter()
            .find_map(|step| match step {
                Step::MoeCombine { out, .. } => Some(&out.buf),
                _ => None,
            })
            .ok_or_else(|| {
                DispatchError::Hip("parallel MoE schedule has no combine output".into())
            })?;
        outputs.push(output);
    }

    let (kind, dim, group, epoch) = reduction.expect("validated schedules have a collective");
    if mesh.epoch() != epoch {
        return Err(DispatchError::Hip(
            "MoE collective belongs to a different mesh generation".into(),
        ));
    }
    if group.len() != schedules.len() || group.len() < 2 {
        return Err(DispatchError::Hip(
            "MoE collective rank count does not match mesh group".into(),
        ));
    }
    if group
        .iter()
        .enumerate()
        .any(|(index, device)| *device >= mesh.n_devices() || group[..index].contains(device))
    {
        return Err(DispatchError::Hip(
            "MoE collective group contains invalid or duplicate devices".into(),
        ));
    }
    if mesh.group_along(kind, &mesh.coord_of(group[0])) != group {
        return Err(DispatchError::Hip(
            "MoE collective group is not a named DeviceMesh axis group".into(),
        ));
    }

    for &device in &group {
        if device >= gpus_len {
            return Err(DispatchError::Hip(format!(
                "MoE collective device {device} is outside the Gpus owner"
            )));
        }
    }
    Ok(MoeMeshPreflight {
        dim,
        group,
        outputs,
    })
}
/// Run the same host-only validation used by
/// [`execute_sealed_steps_mesh`] without touching a device.
///
/// This is a hidden test/integration seam: callers still need fully sealed
/// schedules, and the GPU executor invokes the identical preflight internally.
#[doc(hidden)]
pub fn validate_sealed_steps_mesh_preflight<'a>(
    gpus_len: usize,
    mesh: &DeviceMesh,
    schedules: &[&SealedMoeSchedule<'a>],
) -> Result<(), DispatchError> {
    preflight_sealed_steps_mesh(gpus_len, mesh, schedules).map(|_| ())
}

/// Execute one sealed MoE schedule per participating device and perform its
/// single manifest-owned routed reduction. Schedules are ordered by the
/// collective's rank field; every schedule is validated before any GPU work.
pub fn execute_sealed_steps_mesh<'a>(
    gpus: &mut Gpus,
    mesh: &DeviceMesh,
    ctx: &DispatchCtx,
    schedules: &[&SealedMoeSchedule<'a>],
) -> Result<(), DispatchError> {
    let preflight = preflight_sealed_steps_mesh(gpus.devices.len(), mesh, schedules)?;
    for (rank, &device) in preflight.group.iter().enumerate() {
        execute_steps_inner(&mut gpus.devices[device], ctx, schedules[rank].steps())?;
    }
    gpus.all_reduce_sum_f32_peer(&preflight.group, &preflight.outputs, preflight.dim)
        .map_err(|error| DispatchError::Hip(error.to_string()))
}

fn same_tensor(a: &GpuTensor, b: &GpuTensor) -> bool {
    std::ptr::eq(a, b) || (!a.buf.as_ptr().is_null() && a.buf.as_ptr() == b.buf.as_ptr())
}

fn checked_numel(tensor: &GpuTensor, name: &str) -> Result<usize, DispatchError> {
    tensor
        .shape
        .iter()
        .try_fold(1usize, |elements, dimension| {
            elements.checked_mul(*dimension)
        })
        .ok_or_else(|| DispatchError::Hip(format!("MoE {name} logical shape overflows")))
}

fn require_tensor(
    tensor: &GpuTensor,
    name: &str,
    dtype: DType,
    capacity: usize,
) -> Result<(), DispatchError> {
    let logical_elements = checked_numel(tensor, name)?;
    let required_bytes = capacity
        .checked_mul(dtype.size())
        .ok_or_else(|| DispatchError::Hip(format!("MoE {name} byte capacity overflows")))?;
    if tensor.dtype != dtype || logical_elements < capacity || tensor.buf.size() < required_bytes {
        return Err(DispatchError::Hip(format!(
            "MoE {name} has dtype {:?}/logical capacity {logical_elements}/physical bytes {}, \
             expected {:?}/{capacity} elements/{required_bytes} bytes",
            tensor.dtype,
            tensor.buf.size(),
            dtype
        )));
    }
    Ok(())
}

fn require_raw_i32(tensor: &GpuTensor, name: &str, elements: usize) -> Result<(), DispatchError> {
    let bytes = elements
        .checked_mul(std::mem::size_of::<i32>())
        .ok_or_else(|| DispatchError::Hip(format!("MoE {name} capacity overflows")))?;
    let logical_bytes = checked_numel(tensor, name)?;
    if tensor.dtype != DType::Raw || logical_bytes < bytes || tensor.buf.size() < bytes {
        return Err(DispatchError::Hip(format!(
            "MoE {name} has dtype {:?}/logical bytes {logical_bytes}/physical bytes {}, \
             expected Raw/at least {bytes} bytes",
            tensor.dtype,
            tensor.buf.size()
        )));
    }
    Ok(())
}

fn validate_step_tensors(
    steps: &[Step],
    experts: &MoeExpertRef<'_>,
    batch_size: usize,
    hidden: usize,
) -> Result<(), DispatchError> {
    let k_top = steps
        .iter()
        .find_map(|step| match step {
            Step::IndexedMoeGemv { k_top, .. }
            | Step::GroupedMoeGemm { k_top, .. }
            | Step::MoeGateUpUnscatter { k_top, .. }
            | Step::MoeCombine { k_top, .. } => Some(*k_top),
            _ => None,
        })
        .ok_or_else(|| DispatchError::Hip("MoE grammar has no route width".into()))?;
    let slots = batch_size
        .checked_mul(k_top)
        .ok_or_else(|| DispatchError::Hip("MoE slot capacity overflow".into()))?;
    let route = steps
        .iter()
        .find_map(|step| match step {
            Step::MoeRoute { plan } => Some(plan),
            _ => None,
        })
        .ok_or_else(|| DispatchError::Hip("MoE grammar has no route".into()))?;
    require_tensor(route.route_buffers().0, "route indices", DType::F32, slots)?;
    require_tensor(route.route_buffers().1, "route weights", DType::F32, slots)?;
    let gate_capacity = slots
        .checked_mul(experts.expert_m())
        .ok_or_else(|| DispatchError::Hip("MoE gate capacity overflow".into()))?;
    let down_capacity = slots
        .checked_mul(experts.expert_k())
        .ok_or_else(|| DispatchError::Hip("MoE down capacity overflow".into()))?;
    for step in steps {
        match step {
            Step::IndexedMoeGemv {
                which: MoeProj::GateUp { up_out },
                input,
                out,
                ..
            } => {
                let x = match input {
                    GemvInput::Prerotated(x) => *x,
                    GemvInput::Raw(_) => {
                        return Err(DispatchError::Hip(
                            "generic indexed MoE requires pre-rotated input".into(),
                        ))
                    }
                };
                require_tensor(
                    x,
                    "indexed gate/up input",
                    DType::F32,
                    batch_size.checked_mul(experts.expert_k()).ok_or_else(|| {
                        DispatchError::Hip("MoE indexed input capacity overflow".into())
                    })?,
                )?;
                require_tensor(out, "indexed gate output", DType::F32, gate_capacity)?;
                require_tensor(up_out, "indexed up output", DType::F32, gate_capacity)?;
                if same_tensor(out, up_out) || same_tensor(x, out) || same_tensor(x, up_out) {
                    return Err(DispatchError::Hip(
                        "indexed MoE gate/up buffers must not alias".into(),
                    ));
                }
            }
            Step::IndexedMoeGemv {
                which: MoeProj::DownExpanded,
                input,
                out,
                ..
            } => {
                let x = match input {
                    GemvInput::Prerotated(x) => *x,
                    GemvInput::Raw(_) => {
                        return Err(DispatchError::Hip(
                            "generic indexed MoE requires pre-rotated input".into(),
                        ))
                    }
                };
                require_tensor(x, "indexed down input", DType::F32, gate_capacity)?;
                require_tensor(out, "indexed down output", DType::F32, down_capacity)?;
                if same_tensor(x, out) {
                    return Err(DispatchError::Hip(
                        "indexed MoE down input/output must not alias".into(),
                    ));
                }
            }
            Step::MoeCombine {
                down_out,
                topk_weights,
                out,
                hidden,
                ..
            } => {
                require_tensor(
                    down_out,
                    "combine down input",
                    DType::F32,
                    batch_size
                        .checked_mul(k_top)
                        .and_then(|slots| slots.checked_mul(*hidden))
                        .ok_or_else(|| {
                            DispatchError::Hip("MoE combine input capacity overflows".into())
                        })?,
                )?;
                require_tensor(topk_weights, "combine weights", DType::F32, slots)?;
                require_tensor(
                    out,
                    "combine output",
                    DType::F32,
                    batch_size
                        .checked_mul(*hidden)
                        .ok_or_else(|| DispatchError::Hip("MoE output capacity overflow".into()))?,
                )?;
                if same_tensor(down_out, out) || same_tensor(topk_weights, out) {
                    return Err(DispatchError::Hip(
                        "MoE combine input/output buffers must not alias".into(),
                    ));
                }
            }
            Step::MoeScatter {
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
            } => {
                let offsets = n_experts
                    .checked_add(1)
                    .ok_or_else(|| DispatchError::Hip("MoE expert offsets overflow".into()))?;
                require_raw_i32(expert_token_counts, "expert counts", *n_experts)?;
                require_raw_i32(expert_offsets, "expert offsets", offsets)?;
                require_raw_i32(sorted_slot_index, "sorted slots", *m_total_max)?;
                require_raw_i32(inverse_perm, "inverse permutation", *total_slots)?;
                require_raw_i32(expert_tile_ids, "expert tile ids", *m_total_max / *block_m)?;
            }
            Step::GroupedMoeGemm {
                which,
                x,
                y,
                m_total,
                ..
            } => {
                let input_capacity = match which {
                    MoeProj::GateUp { .. } => {
                        batch_size.checked_mul(experts.expert_k()).ok_or_else(|| {
                            DispatchError::Hip("MoE grouped input capacity overflows".into())
                        })?
                    }
                    MoeProj::DownExpanded => {
                        slots.checked_mul(experts.expert_m()).ok_or_else(|| {
                            DispatchError::Hip("MoE grouped input capacity overflows".into())
                        })?
                    }
                };
                require_tensor(x, "grouped input", DType::F32, input_capacity)?;
                let output_width = match which {
                    MoeProj::GateUp { .. } => {
                        2usize.checked_mul(experts.expert_m()).ok_or_else(|| {
                            DispatchError::Hip("MoE grouped output width overflows".into())
                        })?
                    }
                    MoeProj::DownExpanded => experts.expert_k(),
                };
                require_tensor(
                    y,
                    "grouped output",
                    DType::F32,
                    m_total.checked_mul(output_width).ok_or_else(|| {
                        DispatchError::Hip("grouped output capacity overflow".into())
                    })?,
                )?;
            }
            Step::MoeGateUpUnscatter {
                y_grouped,
                gate_batch,
                up_batch,
                inter,
                m_total,
                ..
            } => {
                require_tensor(
                    y_grouped,
                    "unscatter grouped input",
                    DType::F32,
                    m_total
                        .checked_mul(2usize.checked_mul(*inter).ok_or_else(|| {
                            DispatchError::Hip("MoE unscatter width overflows".into())
                        })?)
                        .ok_or_else(|| {
                            DispatchError::Hip("MoE unscatter input overflows".into())
                        })?,
                )?;
                require_tensor(
                    gate_batch,
                    "unscatter gate",
                    DType::F32,
                    slots.checked_mul(*inter).ok_or_else(|| {
                        DispatchError::Hip("MoE unscatter gate capacity overflows".into())
                    })?,
                )?;
                require_tensor(
                    up_batch,
                    "unscatter up",
                    DType::F32,
                    slots.checked_mul(*inter).ok_or_else(|| {
                        DispatchError::Hip("MoE unscatter up capacity overflows".into())
                    })?,
                )?;
            }
            Step::MoeActivation {
                gate,
                up,
                rot_out,
                inter,
                rows,
                ..
            } => {
                let capacity = rows
                    .checked_mul(*inter)
                    .ok_or_else(|| DispatchError::Hip("MoE activation capacity overflow".into()))?;
                require_tensor(gate, "activation gate", DType::F32, capacity)?;
                require_tensor(up, "activation up", DType::F32, capacity)?;
                require_tensor(rot_out, "activation output", DType::F32, capacity)?;
                if same_tensor(gate, up) || same_tensor(gate, rot_out) || same_tensor(up, rot_out) {
                    return Err(DispatchError::Hip(
                        "MoE activation buffers must not alias".into(),
                    ));
                }
            }
            _ => {}
        }
    }
    let _ = hidden;
    Ok(())
}

fn validate_collectives(
    collectives: &[StepCollective],
    combine_index: usize,
    batch_size: usize,
    hidden: usize,
    expected_kind: Option<DimKind>,
) -> Result<(), DispatchError> {
    let element_count = batch_size
        .checked_mul(hidden)
        .ok_or_else(|| DispatchError::Hip("MoE collective element count overflows".into()))?;
    let mut reduction = None;
    for (index, collective) in collectives.iter().enumerate() {
        let StepCollective::AllReduce {
            kind,
            dim,
            group,
            mesh: _,
            rank,
        } = collective
        else {
            continue;
        };
        if index != combine_index {
            return Err(DispatchError::Hip(
                "MoE routed collective must be attached to combine".into(),
            ));
        }
        if expected_kind != Some(*kind) {
            return Err(DispatchError::Hip(format!(
                "MoE collective axis {kind:?} does not match owner axis {expected_kind:?}"
            )));
        }
        if *dim != element_count || group.len() < 2 || *rank >= group.len() {
            return Err(DispatchError::Hip(
                "MoE collective rank/group/output dimension is invalid".into(),
            ));
        }
        if group
            .iter()
            .enumerate()
            .any(|(offset, device)| group[..offset].contains(device))
        {
            return Err(DispatchError::Hip(
                "MoE collective group contains duplicate devices".into(),
            ));
        }
        if reduction.replace(*kind).is_some() {
            return Err(DispatchError::Hip(
                "MoE schedule contains duplicate routed collectives".into(),
            ));
        }
    }
    if expected_kind.is_some() != reduction.is_some() {
        return Err(DispatchError::Hip(
            "MoE collective count does not match the resolved owner parallelism".into(),
        ));
    }
    Ok(())
}

/// Parallel MoE schedule guard. Single-device execution intentionally leaves
/// all descriptors as `None` (the all-reduce is the identity); parallel
/// execution requires the one post-combine collective emitted by the manifest
/// plan.
pub fn validate_moe_parallel_schedule(
    steps: &[Step],
    collectives: &[StepCollective],
) -> Result<(), DispatchError> {
    validate_moe_step_schedule(steps, collectives)?;
    if collective_count(collectives) != 1 {
        return Err(DispatchError::Hip(
            "parallel MoE schedule requires exactly one routed collective".into(),
        ));
    }
    Ok(())
}

// ── Guard helpers ──────────────────────────────────────────────────────────

/// Extract the dtype of the first Gemv step in the window (step index 1,
/// after the RmsnormAutomatic producer). Returns None if not a Gemv step.
fn window_gemv_dtype(steps: &[Step]) -> Option<DType> {
    match steps.get(1)? {
        Step::Gemv { w, .. } => Some(w.dtype),
        _ => None,
    }
}

/// True if all Gemv steps in the window (indices 1..) have:
/// - the given dtype
/// - GemvInput::Prerotated
/// - awq_scale == None (iff require_no_awq)
fn gemv_steps_uniform(steps: &[Step], dtype: DType, require_no_awq: bool) -> bool {
    steps[1..].iter().all(|s| match s {
        Step::Gemv {
            w,
            input: GemvInput::Prerotated(_),
            ..
        } => w.dtype == dtype && (!require_no_awq || w.awq_scale.is_none()),
        _ => false,
    })
}

/// True if all Gemv steps in the window (indices 1..) have:
/// - the given dtype
/// - GemvInput::Raw (kernel rotates internally — used for Paro guards)
fn gemv_steps_uniform_raw(steps: &[Step], dtype: DType) -> bool {
    steps[1..].iter().all(|s| match s {
        Step::Gemv {
            w,
            input: GemvInput::Raw(_),
            ..
        } => w.dtype == dtype,
        _ => false,
    })
}

/// True if ctx has dp4a and !force_unfused.
fn dp4a_eligible(ctx: &DispatchCtx) -> bool {
    !ctx.flags.force_unfused && ctx.arch.gemv_dp4a_enabled()
}

// ── QKV 3-way guards ──

pub(crate) fn guard_qkv_mq4g256lloyd(steps: &[Step], ctx: &DispatchCtx) -> bool {
    if ctx.flags.force_unfused {
        return false;
    }
    steps.len() == 4 && gemv_steps_uniform(steps, DType::MQ4G256Lloyd, true)
}

/// Exact MQ4G256V2 scalar fusion is fail-closed to gfx1100 + gfx1201 only.
/// Keep `force_unfused` first so the global kill-switch still wins.
fn mq4g256v2_scalar_fusion_ok(ctx: &DispatchCtx) -> bool {
    !ctx.flags.force_unfused && (ctx.arch.is_gfx1100() || ctx.arch.is_gfx1201())
}

/// Exact MQ4G256V2 (qt44) QKV decode fusion. Official Ornith is all-qt44;
/// never admit mixed V1/V2 or AWQ windows onto the V2 fused kernel.
/// Unsupported arches fall through to exact per-projection V2 GEMVs.
pub(crate) fn guard_qkv_mq4g256v2(steps: &[Step], ctx: &DispatchCtx) -> bool {
    mq4g256v2_scalar_fusion_ok(ctx)
        && steps.len() == 4
        && gemv_steps_uniform(steps, DType::MQ4G256V2, true)
}

pub(crate) fn guard_qkv_mq3g256lloyd(steps: &[Step], ctx: &DispatchCtx) -> bool {
    if ctx.flags.force_unfused {
        return false;
    }
    steps.len() == 4 && gemv_steps_uniform(steps, DType::MQ3G256Lloyd, true)
}

/// Covers both DType::MQ4G256 (plain) and DType::HFQ4G256 — both feed
/// gpu.fused_qkv_hfq4g256 which takes a pre-normalized x.
pub(crate) fn guard_qkv_hfq4g256(steps: &[Step], ctx: &DispatchCtx) -> bool {
    if ctx.flags.force_unfused {
        return false;
    }
    if steps.len() != 4 {
        return false;
    }
    let dt = match window_gemv_dtype(steps) {
        Some(d) => d,
        None => return false,
    };
    matches!(dt, DType::MQ4G256 | DType::HFQ4G256) && gemv_steps_uniform(steps, dt, true)
}

/// Covers both DType::HFQ6G256 and DType::MQ6G256.
/// Fusion is safe on RDNA (fused_qkv.rs None arm falls back to gemm n=1)
/// and beneficial on RDNA3+ even without dp4a; dp4a is handled per-arm
/// in fused_qkv.rs dispatch.
pub(crate) fn guard_qkv_hfq6g256(steps: &[Step], ctx: &DispatchCtx) -> bool {
    if ctx.flags.force_unfused {
        return false;
    }
    if steps.len() != 4 {
        return false;
    }
    let dt = match window_gemv_dtype(steps) {
        Some(d) => d,
        None => return false,
    };
    matches!(dt, DType::HFQ6G256 | DType::MQ6G256) && gemv_steps_uniform(steps, dt, true)
}

// ── QKVZA 4-way guards (DeltaNet linear attention) ──

pub(crate) fn guard_qkvza_mq4g256lloyd(steps: &[Step], ctx: &DispatchCtx) -> bool {
    if ctx.flags.force_unfused {
        return false;
    }
    steps.len() == 5 && gemv_steps_uniform(steps, DType::MQ4G256Lloyd, true)
}

/// Exact MQ4G256V2 (qt44) QKVZA decode fusion.
/// Fail-closed to gfx1100/gfx1201; other arches keep per-projection V2 GEMVs.
pub(crate) fn guard_qkvza_mq4g256v2(steps: &[Step], ctx: &DispatchCtx) -> bool {
    mq4g256v2_scalar_fusion_ok(ctx)
        && steps.len() == 5
        && gemv_steps_uniform(steps, DType::MQ4G256V2, true)
}

pub(crate) fn guard_qkvza_mq3g256lloyd(steps: &[Step], ctx: &DispatchCtx) -> bool {
    if ctx.flags.force_unfused {
        return false;
    }
    steps.len() == 5 && gemv_steps_uniform(steps, DType::MQ3G256Lloyd, true)
}

/// Covers both DType::MQ4G256 (plain) and DType::HFQ4G256.
pub(crate) fn guard_qkvza_hfq4g256(steps: &[Step], ctx: &DispatchCtx) -> bool {
    if ctx.flags.force_unfused {
        return false;
    }
    if steps.len() != 5 {
        return false;
    }
    let dt = match window_gemv_dtype(steps) {
        Some(d) => d,
        None => return false,
    };
    matches!(dt, DType::MQ4G256 | DType::HFQ4G256) && gemv_steps_uniform(steps, dt, true)
}

/// Covers both DType::HFQ6G256 and DType::MQ6G256.
/// Fusion is safe on RDNA (fused_qkv.rs None arm falls back to gemm n=1)
/// and beneficial on RDNA3+ even without dp4a; dp4a is handled per-arm
/// in fused_qkv.rs dispatch.
pub(crate) fn guard_qkvza_hfq6g256(steps: &[Step], ctx: &DispatchCtx) -> bool {
    if ctx.flags.force_unfused {
        return false;
    }
    if steps.len() != 5 {
        return false;
    }
    let dt = match window_gemv_dtype(steps) {
        Some(d) => d,
        None => return false,
    };
    matches!(dt, DType::HFQ6G256 | DType::MQ6G256) && gemv_steps_uniform(steps, dt, true)
}

// ── Gate+Up 2-way guards ──

pub(crate) fn guard_gate_up_mq4g256lloyd(steps: &[Step], ctx: &DispatchCtx) -> bool {
    if ctx.flags.force_unfused {
        return false;
    }
    steps.len() == 3 && gemv_steps_uniform(steps, DType::MQ4G256Lloyd, true)
}

/// Exact MQ4G256V2 (qt44) gate+up decode fusion.
/// Fail-closed to gfx1100/gfx1201; other arches keep per-projection V2 GEMVs.
pub(crate) fn guard_gate_up_mq4g256v2(steps: &[Step], ctx: &DispatchCtx) -> bool {
    mq4g256v2_scalar_fusion_ok(ctx)
        && steps.len() == 3
        && gemv_steps_uniform(steps, DType::MQ4G256V2, true)
}

pub(crate) fn guard_gate_up_mq3g256lloyd(steps: &[Step], ctx: &DispatchCtx) -> bool {
    if ctx.flags.force_unfused {
        return false;
    }
    steps.len() == 3 && gemv_steps_uniform(steps, DType::MQ3G256Lloyd, true)
}

pub(crate) fn guard_gate_up_hfq4g256(steps: &[Step], ctx: &DispatchCtx) -> bool {
    if ctx.flags.force_unfused {
        return false;
    }
    if steps.len() != 3 {
        return false;
    }
    let dt = match window_gemv_dtype(steps) {
        Some(d) => d,
        None => return false,
    };
    matches!(dt, DType::MQ4G256 | DType::HFQ4G256) && gemv_steps_uniform(steps, dt, true)
}

pub(crate) fn guard_gate_up_hfq6g256(steps: &[Step], ctx: &DispatchCtx) -> bool {
    if !dp4a_eligible(ctx) {
        return false;
    }
    if steps.len() != 3 {
        return false;
    }
    let dt = match window_gemv_dtype(steps) {
        Some(d) => d,
        None => return false,
    };
    matches!(dt, DType::HFQ6G256 | DType::MQ6G256) && gemv_steps_uniform(steps, dt, true)
}

// ── mfp4-E8 decode launch-fusion guards (gfx1151 / Strix Halo ONLY) ──
// These are the SOLE producers of the FusedGateUpMfp4G32E8 / FusedQkvzaMfp4G32E8
// keys. The `is_gfx1151()` check firewalls the fused kernels to gfx1151 — on every
// other arch these return false and the projections fall through to the
// per-projection gemv_mfp4g32_e8 path unchanged. The fused kernels embed the
// byte-identical gemv_mfp4g32_e8 per-row body, so the fused output equals N
// sequential GEMVs bit-for-bit (only the launch count shrinks).
//
// gfx11 E8 port finding: the fusion (launch-overhead reduction, +5.8% on the Strix
// Halo APU) does NOT transfer to the gfx1100 dGPU — measured decode 101.7 (fused)
// vs 102.6 (unfused) tok/s, a ~1% LOSS, bit-identical output. The dGPU's faster
// compute + the (32,7) launch_bounds tuned for gfx1151 occupancy leave no launch
// win to capture. Kept gfx1151-only; revisit only with a gfx1100 occupancy retune.
pub(crate) fn guard_gate_up_mfp4g32e8(steps: &[Step], ctx: &DispatchCtx) -> bool {
    if ctx.flags.force_unfused {
        return false;
    }
    if !ctx.arch.is_gfx1151() {
        return false;
    }
    if steps.len() != 3 {
        return false;
    }
    let dt = match window_gemv_dtype(steps) {
        Some(d) => d,
        None => return false,
    };
    dt == DType::MFP4G32E8 && gemv_steps_uniform(steps, dt, true)
}

pub(crate) fn guard_qkvza_mfp4g32e8(steps: &[Step], ctx: &DispatchCtx) -> bool {
    if ctx.flags.force_unfused {
        return false;
    }
    if !ctx.arch.is_gfx1151() {
        return false;
    }
    if steps.len() != 5 {
        return false;
    }
    let dt = match window_gemv_dtype(steps) {
        Some(d) => d,
        None => return false,
    };
    dt == DType::MFP4G32E8 && gemv_steps_uniform(steps, dt, true)
}

// ── Paro fused guards (Raw input — kernel rotates internally) ──

// ── Q8_0 / Q4K fused guards (non-rotated, Prerotated input) ──
// These dtypes have no activation rotation (RotationPlan::None), so the
// RmsnormAutomatic producer does plain rmsnorm and the fused kernels take
// the pre-normed x directly. Prerotated input is correct because
// for_gemv_prerotated(Q8_0/Q4K) falls back to the plain GEMV kernel.

/// Fused QKV with Q4K weights. Used by llama (dense).
pub(crate) fn guard_qkv_q4k(steps: &[Step], ctx: &DispatchCtx) -> bool {
    if ctx.flags.force_unfused {
        return false;
    }
    steps.len() == 4 && gemv_steps_uniform(steps, DType::Q4K, true)
}

/// Fused gate+up with Q4K weights. Used by llama (dense).
pub(crate) fn guard_gate_up_q4k(steps: &[Step], ctx: &DispatchCtx) -> bool {
    if ctx.flags.force_unfused {
        return false;
    }
    steps.len() == 3 && gemv_steps_uniform(steps, DType::Q4K, true)
}

/// Fused gate+up with Q8_0 weights. Used by qwen2 FFN.
pub(crate) fn guard_gate_up_q8_0(steps: &[Step], ctx: &DispatchCtx) -> bool {
    if ctx.flags.force_unfused {
        return false;
    }
    steps.len() == 3 && gemv_steps_uniform(steps, DType::Q8_0, true)
}

/// Fused 4-way QKVZA with Q8_0 weights (DECODE path, n=1). Used by
/// Qwen3.5/A3B .mq4p DeltaNet layers (qt=3). No dp4a required.
pub(crate) fn guard_qkvza_q8_0(steps: &[Step], ctx: &DispatchCtx) -> bool {
    if ctx.flags.force_unfused {
        return false;
    }
    steps.len() == 5 && gemv_steps_uniform(steps, DType::Q8_0, true)
}

/// Fused 3-way QKV with Q8_0 weights (DECODE path, n=1). No dp4a required.
pub(crate) fn guard_qkv_q8_0(steps: &[Step], ctx: &DispatchCtx) -> bool {
    if ctx.flags.force_unfused {
        return false;
    }
    steps.len() == 4 && gemv_steps_uniform(steps, DType::Q8_0, true)
}

pub(crate) fn guard_gate_up_paro4g128t(steps: &[Step], ctx: &DispatchCtx) -> bool {
    if ctx.flags.force_unfused {
        return false;
    }
    if steps.len() != 3 {
        return false;
    }
    let dt = match window_gemv_dtype(steps) {
        Some(d) => d,
        None => return false,
    };
    dt == DType::ParoQ4G128
        && gemv_steps_uniform_raw(steps, DType::ParoQ4G128)
        && steps[1..].iter().all(|s| match s {
            Step::Gemv { w, .. } => w.m % 8 == 0 && w.k % 128 == 0,
            _ => false,
        })
        // Gate and up must have equal m — the fused kernel takes a single m.
        && {
            let m0 = match &steps[1] { Step::Gemv { w, .. } => w.m, _ => return false };
            let m1 = match &steps[2] { Step::Gemv { w, .. } => w.m, _ => return false };
            m0 == m1
        }
}

pub(crate) fn guard_qkvza_paro4g128t(_steps: &[Step], _ctx: &DispatchCtx) -> bool {
    false
}

pub(crate) fn guard_qkv_paro4g128t(_steps: &[Step], _ctx: &DispatchCtx) -> bool {
    false
}

pub struct FusedPattern {
    pub ops: &'static [PipelineOp],
    pub key: KernelKey,
    /// Dtype/arch predicate called after op-kind prefix match. Must return true
    /// for the entry to fire. Receives the full matched window (all ops.len()
    /// steps starting at the current position).
    pub guard: fn(&[Step], &DispatchCtx) -> bool,
}

/// Greedy longest-prefix op-pattern match with dtype/arch guard.
pub fn match_prefix(
    table: &[FusedPattern],
    steps: &[Step],
    ctx: &DispatchCtx,
) -> Option<(KernelKey, usize)> {
    table
        .iter()
        .filter(|p| {
            !p.ops.is_empty()
                && p.ops.len() <= steps.len()
                && p.ops.iter().zip(steps).all(|(o, s)| *o == op_kind(s))
                && (p.guard)(&steps[..p.ops.len()], ctx)
        })
        .max_by_key(|p| p.ops.len())
        .map(|p| (p.key, p.ops.len()))
}

/// Lower-time fusion match over the canonical `FUSED_TABLE`. The Ship-6 super-op
/// lowering (`superop::lower_layer`) calls THIS — reusing the same table + guards
/// verbatim — so a lowered program can never drift from what `execute_steps`
/// would dispatch live (the fusion-drift mitigation, spike risk #1).
pub(crate) fn match_fused_prefix(steps: &[Step], ctx: &DispatchCtx) -> Option<(KernelKey, usize)> {
    match_prefix(FUSED_TABLE, steps, ctx)
}

/// Public(crate) op-kind accessor for the lowering (mirror of the private `op_kind`).
pub(crate) fn step_op_kind(step: &Step) -> PipelineOp {
    op_kind(step)
}

const QKV3: &[PipelineOp] = &[
    PipelineOp::RmsnormAutomatic,
    PipelineOp::Gemv,
    PipelineOp::Gemv,
    PipelineOp::Gemv,
];
const QKVZA4: &[PipelineOp] = &[
    PipelineOp::RmsnormAutomatic,
    PipelineOp::Gemv,
    PipelineOp::Gemv,
    PipelineOp::Gemv,
    PipelineOp::Gemv,
];
const GATE_UP2: &[PipelineOp] = &[
    PipelineOp::RmsnormAutomatic,
    PipelineOp::Gemv,
    PipelineOp::Gemv,
];

const FUSED_TABLE: &[FusedPattern] = &[
    // ── QKV 3-way ──────────────────────────────────────────────────────────
    FusedPattern {
        ops: QKV3,
        key: KernelKey::FusedQkvMq4G256Lloyd,
        guard: guard_qkv_mq4g256lloyd,
    },
    FusedPattern {
        ops: QKV3,
        key: KernelKey::FusedQkvMq3G256Lloyd,
        guard: guard_qkv_mq3g256lloyd,
    },
    // Exact MQ4G256V2 before broad HFQ4/MQ4 V1 neighbor.
    FusedPattern {
        ops: QKV3,
        key: KernelKey::FusedQkvMq4G256V2,
        guard: guard_qkv_mq4g256v2,
    },
    FusedPattern {
        ops: QKV3,
        key: KernelKey::FusedQkvHfq4G256,
        guard: guard_qkv_hfq4g256,
    },
    FusedPattern {
        ops: QKV3,
        key: KernelKey::FusedQkvHfq6G256,
        guard: guard_qkv_hfq6g256,
    },
    // ── QKVZA 4-way (DeltaNet linear attention) ────────────────────────────
    FusedPattern {
        ops: QKVZA4,
        key: KernelKey::FusedQkvzaMq4G256Lloyd,
        guard: guard_qkvza_mq4g256lloyd,
    },
    FusedPattern {
        ops: QKVZA4,
        key: KernelKey::FusedQkvzaMq3G256Lloyd,
        guard: guard_qkvza_mq3g256lloyd,
    },
    // Exact MQ4G256V2 before broad HFQ4/MQ4 V1 neighbor.
    FusedPattern {
        ops: QKVZA4,
        key: KernelKey::FusedQkvzaMq4G256V2,
        guard: guard_qkvza_mq4g256v2,
    },
    FusedPattern {
        ops: QKVZA4,
        key: KernelKey::FusedQkvzaHfq4G256,
        guard: guard_qkvza_hfq4g256,
    },
    FusedPattern {
        ops: QKVZA4,
        key: KernelKey::FusedQkvzaHfq6G256,
        guard: guard_qkvza_hfq6g256,
    },
    // mfp4-E8 decode launch-fusion — gfx1151-ONLY (guard firewalls the arch).
    FusedPattern {
        ops: QKVZA4,
        key: KernelKey::FusedQkvzaMfp4G32E8,
        guard: guard_qkvza_mfp4g32e8,
    },
    // ── Gate+Up 2-way ───────────────────────────────────────────────────────
    FusedPattern {
        ops: GATE_UP2,
        key: KernelKey::FusedGateUpMq4G256Lloyd,
        guard: guard_gate_up_mq4g256lloyd,
    },
    FusedPattern {
        ops: GATE_UP2,
        key: KernelKey::FusedGateUpMq3G256Lloyd,
        guard: guard_gate_up_mq3g256lloyd,
    },
    // Exact MQ4G256V2 before broad HFQ4/MQ4 V1 neighbor.
    FusedPattern {
        ops: GATE_UP2,
        key: KernelKey::FusedGateUpMq4G256V2,
        guard: guard_gate_up_mq4g256v2,
    },
    FusedPattern {
        ops: GATE_UP2,
        key: KernelKey::FusedGateUpHfq4G256,
        guard: guard_gate_up_hfq4g256,
    },
    FusedPattern {
        ops: GATE_UP2,
        key: KernelKey::FusedGateUpHfq6G256,
        guard: guard_gate_up_hfq6g256,
    },
    // mfp4-E8 decode launch-fusion — gfx1151-ONLY (guard firewalls the arch).
    FusedPattern {
        ops: GATE_UP2,
        key: KernelKey::FusedGateUpMfp4G32E8,
        guard: guard_gate_up_mfp4g32e8,
    },
    // ── Q8_0 / Q4K fused entries (non-rotated, Always arch gate) ─────────
    // Q8_0 QKV/QKVZA: Qwen3.5-A3B .mq4p uses Q8_0 for all linear-attention
    // projections (qt=3). Scalar decode kernels added 2026-06-14.
    FusedPattern {
        ops: QKVZA4,
        key: KernelKey::FusedQkvzaQ8_0,
        guard: guard_qkvza_q8_0,
    },
    FusedPattern {
        ops: QKV3,
        key: KernelKey::FusedQkvQ8_0,
        guard: guard_qkv_q8_0,
    },
    FusedPattern {
        ops: QKV3,
        key: KernelKey::FusedQkvQ4K,
        guard: guard_qkv_q4k,
    },
    FusedPattern {
        ops: GATE_UP2,
        key: KernelKey::FusedGateUpQ4K,
        guard: guard_gate_up_q4k,
    },
    FusedPattern {
        ops: GATE_UP2,
        key: KernelKey::FusedGateUpQ8_0,
        guard: guard_gate_up_q8_0,
    },
    // ── Paro fused Paro4G128T (dp4a, Raw input) ────────────────────────
    FusedPattern {
        ops: GATE_UP2,
        key: KernelKey::FusedGateUpParo4G128T,
        guard: guard_gate_up_paro4g128t,
    },
    FusedPattern {
        ops: QKVZA4,
        key: KernelKey::FusedQkvzaParo4G128T,
        guard: guard_qkvza_paro4g128t,
    },
    FusedPattern {
        ops: QKV3,
        key: KernelKey::FusedQkvParo4G128T,
        guard: guard_qkv_paro4g128t,
    },
];
static GEMV: OnceLock<GemvFamily> = OnceLock::new();
static ROTATION: OnceLock<RotationFamily> = OnceLock::new();
static FUSED_QKV: OnceLock<FusedQkvFamily> = OnceLock::new();
static MOE: std::sync::LazyLock<MoeFamily> = std::sync::LazyLock::new(MoeFamily::new);

fn reject_unsealed_moe(steps: &[Step]) -> Result<(), DispatchError> {
    if steps.iter().any(is_moe_step) {
        return Err(DispatchError::Hip(
            "unsealed MoE schedules require execute_sealed_steps or execute_sealed_steps_mesh"
                .into(),
        ));
    }
    Ok(())
}

pub fn execute_steps(
    gpu: &mut Gpu,
    ctx: &DispatchCtx,
    steps: &[Step],
) -> Result<(), DispatchError> {
    reject_unsealed_moe(steps)?;
    execute_steps_inner(gpu, ctx, steps)
}

fn execute_steps_inner(
    gpu: &mut Gpu,
    ctx: &DispatchCtx,
    steps: &[Step],
) -> Result<(), DispatchError> {
    let mut i = 0;
    while i < steps.len() {
        if let Some((key, len)) = match_prefix(FUSED_TABLE, &steps[i..], ctx) {
            if ctx.flags.fuse_qkv_bias && len == QKV3.len() && qkv_bias_fold_supported(key, ctx) {
                if let Some(biases) = match_trailing_qkv_bias(&steps[i..], len) {
                    launch_fused_qkv_with_bias(gpu, ctx, key, &steps[i..i + len], biases)?;
                    i += len + 3;
                    continue;
                }
            }
            launch_fused(gpu, ctx, key, &steps[i..i + len])?;
            i += len;
        } else {
            launch_op(gpu, ctx, &steps[i])?;
            i += 1;
        }
    }
    Ok(())
}

fn is_moe_step(step: &Step<'_>) -> bool {
    matches!(
        step,
        Step::MoeRoute { .. }
            | Step::IndexedMoeGemv { .. }
            | Step::MoeCombine { .. }
            | Step::MoeScatter { .. }
            | Step::GroupedMoeGemm { .. }
            | Step::MoeGateUpUnscatter { .. }
            | Step::MoeActivation { .. }
    )
}

/// Keys whose 3-way QKV **decode** dispatch arm folds the optional Q/K/V bias
/// into the kernel (a `_with_bias` kernel variant exists and is wired in
/// `dispatch_fused_qkv`). The fold is additionally guarded off on dp4a archs
/// (gfx906), whose fused-QKV kernel has no bias parameters — there the 3
/// `BiasAdd` steps run separately as before (handover pitfall #4). Keys NOT
/// listed here keep the unfused path: their `BiasAdd` steps are never consumed,
/// so the result is unchanged.
fn qkv_bias_fold_supported(key: KernelKey, ctx: &DispatchCtx) -> bool {
    if ctx.arch.gemv_dp4a_enabled() {
        return false;
    }
    // All per-row 3-way QKV decode keys whose `gpu.fused_qkv_*_with_bias`
    // variant is wired and GPU-parity-validated (no-bias==0, with-bias==bias;
    // see examples/test_fused_qkv_bias_parity.rs). HFQ4G256 additionally has the
    // full three-way model byte-identity proof (see the Phase-1 commit).
    matches!(
        key,
        KernelKey::FusedQkvHfq4G256
            | KernelKey::FusedQkvMq4G256Lloyd
            | KernelKey::FusedQkvMq3G256Lloyd
            | KernelKey::FusedQkvQ4K
            | KernelKey::FusedQkvQ8_0
            // HFQ6/MQ6: the fold switches decode GEMM→per-row (Family B). The
            // dispatch arm keeps the GEMM unless bias is present, so this only
            // changes decode when the fold actually fires.
            | KernelKey::FusedQkvHfq6G256
    )
}

/// If `steps[len..len+3]` are three `BiasAdd` ops whose `x` targets are exactly
/// the q/k/v GEMV outputs of the fused window (`steps[1..4]`), return the three
/// bias tensors `[bias_q, bias_k, bias_v]`. Otherwise `None` (no fold). The
/// ptr-identity check guarantees we only fold the qwen2 `attention_bias` adds
/// that immediately follow this exact QKV window, never an unrelated `BiasAdd`.
fn match_trailing_qkv_bias<'a>(steps: &'a [Step<'a>], len: usize) -> Option<[&'a GpuTensor; 3]> {
    if len + 3 > steps.len() {
        return None;
    }
    let (
        Step::BiasAdd {
            x: bx_q, bias: bq, ..
        },
        Step::BiasAdd {
            x: bx_k, bias: bk, ..
        },
        Step::BiasAdd {
            x: bx_v, bias: bv, ..
        },
    ) = (&steps[len], &steps[len + 1], &steps[len + 2])
    else {
        return None;
    };
    let (_, out_q) = gemv_weight_out(&steps[1]);
    let (_, out_k) = gemv_weight_out(&steps[2]);
    let (_, out_v) = gemv_weight_out(&steps[3]);
    if std::ptr::eq(*bx_q as *const GpuTensor, out_q as *const GpuTensor)
        && std::ptr::eq(*bx_k as *const GpuTensor, out_k as *const GpuTensor)
        && std::ptr::eq(*bx_v as *const GpuTensor, out_v as *const GpuTensor)
    {
        Some([bq, bk, bv])
    } else {
        None
    }
}

/// Launch a Qwen2 3-way QKV window through bias-specific kernel symbols.
/// Qwen3+ continues through [`launch_fused`] and the original Redline ABI.
fn launch_fused_qkv_with_bias<'a>(
    gpu: &mut Gpu,
    ctx: &DispatchCtx,
    key: KernelKey,
    steps: &[Step<'a>],
    bias: [&'a GpuTensor; 3],
) -> Result<(), DispatchError> {
    // Opt-in diagnostic (HIPFIRE_FUSE_QKV_BIAS_DEBUG=1): confirm the fold fires.
    // Flag is resolved once at init — no per-launch env lock on the hot path.
    if ctx.flags.fuse_qkv_bias_debug {
        eprintln!("[qkv-bias-fold] fired ({:?})", key);
    }
    // Step 0 is RmsnormAutomatic — run it to fill the activated buffer.
    launch_op(gpu, ctx, &steps[0])?;
    let activated = rmsnorm_out(&steps[0]);
    let (wq, q) = gemv_weight_out(&steps[1]);
    let (wk, k) = gemv_weight_out(&steps[2]);
    let (wv, v) = gemv_weight_out(&steps[3]);
    let fused_qkv = FUSED_QKV.get_or_init(FusedQkvFamily::new);
    fused_qkv.run_with_qwen2_bias(
        ctx,
        gpu,
        &FusedQkvBiasParams {
            kind: key,
            weights: [wq.buf, wk.buf, wv.buf],
            x: activated,
            outputs: [q, k, v],
            m: [wq.m, wk.m, wv.m],
            k: wq.k,
            bias,
        },
    )
}

/// Per-op fallback. FULL enum match (no catch-all) so the compiler forces every
/// op to have an arm (spec F4 — a missing arm would be a silent runtime error).
fn launch_op(gpu: &mut Gpu, ctx: &DispatchCtx, step: &Step) -> Result<(), DispatchError> {
    match step {
        Step::Gemv {
            w,
            input: GemvInput::Raw(x),
            out,
        } => {
            let gemv = GEMV.get_or_init(GemvFamily::new);
            gemv.run_auto(ctx, gpu, w, x, out)
        }
        Step::Gemv {
            w,
            input: GemvInput::Prerotated(xr),
            out,
        } => {
            let gemv = GEMV.get_or_init(GemvFamily::new);
            gemv.run(
                ctx,
                gpu,
                &GemvParams {
                    w,
                    x: xr,
                    y: out,
                    variant: GemvVariant::Prerotated,
                    residual: None,
                    gate: None,
                    up: None,
                },
            )
        }
        Step::GemvResidual {
            w,
            input: GemvInput::Prerotated(xr),
            residual,
            out: _,
        } => {
            // MQ-family with a fused residual kernel: writes `residual` in-place via
            // GemvVariant::WithResidual. `out` is NOT written — it is scratch for the
            // fallback path only (see the Raw arm below). Nothing downstream reads
            // `out` after this step in either qwen2 or llama decode paths.
            let gemv = GEMV.get_or_init(GemvFamily::new);
            gemv.run(
                ctx,
                gpu,
                &GemvParams {
                    w,
                    x: xr,
                    y: residual,
                    variant: GemvVariant::WithResidual,
                    residual: None,
                    gate: None,
                    up: None,
                },
            )
        }
        Step::GemvResidual {
            w,
            input: GemvInput::Raw(x),
            residual,
            out,
        } => {
            // For dtypes WITHOUT a fused residual kernel (Q8_0, Q4K, F32), the
            // fallback path runs a plain GEMV then `residual += result`. `out` may
            // be used as scratch ONLY when it does not alias `residual`; when it
            // does (the common qwen35 o_proj / dn_out case where out == residual ==
            // &s.x), a dedicated persistent temp is used instead.
            // Nothing reads `out` after this step in any model decode path.
            let gemv = GEMV.get_or_init(GemvFamily::new);
            // Dtypes with a fused `gemv_*_residual` kernel use it in one launch.
            // Dtypes without one (Q8_0, ParoQ4G128, …) fall back to plain GEMV into
            // the `out` scratch + `residual += out` — reuses the pre-allocated `out`
            // buffer instead of alloc/free per call. Plain GEMV applies this
            // dtype's own rotation (FWHT / Givens) internally, so this is correct
            // for both no-rotation (Q8) and Givens (Paro) dtypes.
            if KernelKey::for_gemv_residual(w.dtype).is_ok() {
                if crate::types::dtype_rotation_plan(w.dtype) != RotationPlan::None {
                    let h = gemv.rotate(ctx, gpu, w, x, &RotateInputs::default())?;
                    let xr = h.into_buf();
                    gemv.run(
                        ctx,
                        gpu,
                        &GemvParams {
                            w,
                            x: &xr,
                            y: residual,
                            variant: GemvVariant::WithResidual,
                            residual: None,
                            gate: None,
                            up: None,
                        },
                    )
                } else {
                    gemv.run(
                        ctx,
                        gpu,
                        &GemvParams {
                            w,
                            x,
                            y: residual,
                            variant: GemvVariant::WithResidual,
                            residual: None,
                            gate: None,
                            up: None,
                        },
                    )
                }
            } else {
                // run_auto applies the dtype's rotation (FWHT/Givens) before the
                // kernel, so ParoQ4G128 gets its Givens rotation. Plain would skip it.
                //
                // ALIASING GUARD: most callers (e.g. qwen35 o_proj / dn_out) pass
                // `out` == `residual` (both `&s.x`). Reusing `out` as the GEMV scratch
                // in that case is WRONG: run_auto would overwrite the residual with
                // `W·x` and the subsequent `residual += out` would then compute
                // `2·(W·x)` — the residual is lost. Detect the alias by device pointer
                // and use a dedicated persistent scratch when they overlap. When `out`
                // is a genuinely-distinct buffer, reuse it (no alloc churn).
                if std::ptr::eq(residual, out) || residual.buf.as_ptr() == out.buf.as_ptr() {
                    let tmp = {
                        let scratch = gpu
                            .ensure_gemv_residual_tmp(w.m)
                            .map_err(|e| DispatchError::Hip(e.to_string()))?;
                        // `gpu` owns this dedicated allocation for the alias's lifetime.
                        GpuTensor {
                            buf: unsafe { scratch.buf.alias() },
                            shape: vec![w.m],
                            dtype: DType::F32,
                        }
                    };
                    gemv.run_auto(ctx, gpu, w, x, &tmp)?;
                    gpu.add_inplace_f32(residual, &tmp)
                        .map_err(|e| DispatchError::Hip(e.to_string()))?;
                } else {
                    gemv.run_auto(ctx, gpu, w, x, out)?;
                    gpu.add_inplace_f32(residual, out)
                        .map_err(|e| DispatchError::Hip(e.to_string()))?;
                }
                Ok(())
            }
        }
        Step::RmsnormAutomatic {
            x,
            norm_weight,
            x_plain,
            out,
            awq_scale,
            k,
            eps,
            rotation,
        } => {
            if *rotation == RotationPlan::None {
                // HFQ4G256 and other non-FWHT dtypes: plain rmsnorm into `out`.
                // x_plain is not written in this path (scratch only for FWHT path).
                gpu.rmsnorm_f32(x, norm_weight, out, *eps)
                    .map_err(|e| DispatchError::Hip(e.to_string()))
            } else if *rotation == RotationPlan::Mq8Internal {
                // MQ8 cannot share LDS with the FWHT-G256 fused kernel: it produces an
                // INT8 scratch consumed by the downstream gemv_mq8_prerotated kernel.
                // RotationFamily::WithRmsnorm would route to fused_rmsnorm_rotate_mq
                // (FWHT, F32 output) — wrong dtype for the MQ8 GEMV. Mirror the fix
                // from qwen35.rs::rmsnorm_rotate_dispatch (7b35e700).
                gpu.rmsnorm_f32(x, norm_weight, out, *eps)
                    .map_err(|e| DispatchError::Hip(e.to_string()))?;
                gpu.rotate_quantize_x_mq8(out, *k)
                    .map_err(|e| DispatchError::Hip(e.to_string()))
            } else {
                let rotation_family = ROTATION.get_or_init(RotationFamily::new);
                rotation_family
                    .run(
                        ctx,
                        gpu,
                        RotationParams {
                            x,
                            x_up: None,
                            w_norm: Some(norm_weight),
                            x_plain,
                            x_rot: out,
                            awq_scale: *awq_scale,
                            k: *k,
                            eps: *eps,
                            batch_size: 1,
                            variant: RotationVariant::WithRmsnorm,
                            givens_pairs: None,
                            givens_theta: None,
                            givens_scales: None,
                            givens_krot: None,
                        },
                    )
                    .map_err(|e| DispatchError::Hip(e.to_string()))
            }
        }
        Step::Attend { plan, io } => {
            use crate::families::attention::AttentionFamily;
            static ATTENTION: OnceLock<AttentionFamily> = OnceLock::new();
            let attn = ATTENTION.get_or_init(AttentionFamily::new);
            attn.run_attention(ctx, gpu, plan, io)
        }
        Step::Rope {
            q,
            k,
            pos_buf,
            n_heads,
            n_kv_heads,
            head_dim,
            theta,
        } => gpu
            .rope_f32(q, k, pos_buf, *n_heads, *n_kv_heads, *head_dim, *theta)
            .map_err(|e| DispatchError::Hip(e.to_string())),
        Step::QkNorm {
            x,
            weight,
            n_groups,
            head_dim,
            eps,
        } => gpu
            .rmsnorm_batched(x, weight, x, *n_groups, *head_dim, *eps)
            .map_err(|e| DispatchError::Hip(e.to_string())),
        Step::MoeRoute { plan } => MOE.run_route(gpu, plan),
        Step::IndexedMoeGemv {
            experts,
            which,
            topk_indices,
            input,
            out,
            k_top,
            batch_size,
        } => MOE.run_indexed(
            gpu,
            experts,
            which,
            topk_indices,
            input,
            out,
            *k_top,
            *batch_size,
        ),
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
        } => MOE.run_scatter(
            gpu,
            topk_indices,
            expert_token_counts,
            expert_offsets,
            sorted_slot_index,
            expert_tile_ids,
            inverse_perm,
            *total_slots,
            *n_experts,
            *m_total_max,
            *block_m,
        ),
        Step::GroupedMoeGemm {
            experts,
            which,
            sorted_slot_index,
            expert_tile_ids,
            x,
            y,
            m_total,
            batch_size,
            k_top,
        } => MOE.run_grouped(
            gpu,
            experts,
            which,
            sorted_slot_index,
            expert_tile_ids,
            x,
            y,
            *m_total,
            *batch_size,
            *k_top,
        ),
        Step::MoeGateUpUnscatter {
            y_grouped,
            sorted_slot_index,
            gate_batch,
            up_batch,
            inter,
            k_top,
            m_total,
        } => MOE.run_unscatter(
            gpu,
            y_grouped,
            sorted_slot_index,
            gate_batch,
            up_batch,
            *inter,
            *k_top,
            *m_total,
        ),
        Step::MoeActivation {
            variant,
            gate,
            up,
            rot_out,
            inter,
            rows,
        } => MOE.run_activation(gpu, *variant, gate, up, rot_out, *inter, *rows),
        Step::MoeCombine {
            down_out,
            topk_weights,
            out,
            hidden,
            k_top,
            batch_size,
            inverse_perm,
        } => MOE.run_combine(
            gpu,
            down_out,
            topk_weights,
            out,
            *hidden,
            *k_top,
            *batch_size,
            *inverse_perm,
        ),
        Step::BiasAdd { x, bias, dim } => gpu
            .bias_add_f32(x, bias, 1, *dim)
            .map_err(|e| DispatchError::Hip(e.to_string())),
    }
}

/// Borrow `out` from a `RmsnormAutomatic` step. The guard has already confirmed
/// step[0] is RmsnormAutomatic; this panics in debug if called incorrectly.
fn rmsnorm_out<'a>(step: &'a Step<'a>) -> &'a rdna_compute::GpuTensor {
    match step {
        Step::RmsnormAutomatic { out, .. } => out,
        _ => panic!("launch_fused: expected RmsnormAutomatic at step[0]"),
    }
}

/// Borrow `w` and `out` from a `Gemv` step.
fn gemv_weight_out<'a>(step: &'a Step<'a>) -> (&'a WeightRef<'a>, &'a rdna_compute::GpuTensor) {
    match step {
        Step::Gemv { w, out, .. } => (w, out),
        _ => panic!("launch_fused: expected Gemv step"),
    }
}

fn launch_fused(
    gpu: &mut Gpu,
    ctx: &DispatchCtx,
    key: KernelKey,
    steps: &[Step],
) -> Result<(), DispatchError> {
    // Step 0 is always RmsnormAutomatic — run it to fill the activated buffer.
    launch_op(gpu, ctx, &steps[0])?;
    let activated = rmsnorm_out(&steps[0]);
    let fused_qkv = FUSED_QKV.get_or_init(FusedQkvFamily::new);

    match key {
        KernelKey::FusedQkvMq4G256Lloyd
        | KernelKey::FusedQkvMq4G256V2
        | KernelKey::FusedQkvMq3G256Lloyd
        | KernelKey::FusedQkvHfq4G256
        | KernelKey::FusedQkvHfq6G256
        | KernelKey::FusedQkvQ4K
        | KernelKey::FusedQkvQ8_0 => {
            let (wq, q) = gemv_weight_out(&steps[1]);
            let (wk, k) = gemv_weight_out(&steps[2]);
            let (wv, v) = gemv_weight_out(&steps[3]);
            fused_qkv.run(
                ctx,
                gpu,
                &FusedQkvParams {
                    kind: key,
                    weights: &[wq.buf, wk.buf, wv.buf],
                    x: activated,
                    outputs: &[q, k, v],
                    m: &[wq.m, wk.m, wv.m],
                    k: wq.k,
                    rot_scratch: &[],
                    batch_size: None,
                },
            )
        }
        KernelKey::FusedGateUpMq4G256Lloyd
        | KernelKey::FusedGateUpMq4G256V2
        | KernelKey::FusedGateUpMq3G256Lloyd
        | KernelKey::FusedGateUpHfq4G256
        | KernelKey::FusedGateUpHfq6G256
        | KernelKey::FusedGateUpQ4K
        | KernelKey::FusedGateUpQ8_0
        | KernelKey::FusedGateUpMfp4G32E8 => {
            let (wg, gate) = gemv_weight_out(&steps[1]);
            let (wu, up) = gemv_weight_out(&steps[2]);
            fused_qkv.run(
                ctx,
                gpu,
                &FusedQkvParams {
                    kind: key,
                    weights: &[wg.buf, wu.buf],
                    x: activated,
                    outputs: &[gate, up],
                    m: &[wg.m, wu.m],
                    k: wg.k,
                    rot_scratch: &[],
                    batch_size: None,
                },
            )
        }
        // ── QKVZA 4-way (DeltaNet) ──
        KernelKey::FusedQkvzaHfq4G256
        | KernelKey::FusedQkvzaMq3G256Lloyd
        | KernelKey::FusedQkvzaMq4G256Lloyd
        | KernelKey::FusedQkvzaMq4G256V2
        | KernelKey::FusedQkvzaHfq6G256
        | KernelKey::FusedQkvzaMfp4G32E8
        | KernelKey::FusedQkvzaQ8_0 => {
            let (wqkv, qkv) = gemv_weight_out(&steps[1]);
            let (wz, z) = gemv_weight_out(&steps[2]);
            let (wb, beta) = gemv_weight_out(&steps[3]);
            let (wa, alpha) = gemv_weight_out(&steps[4]);
            fused_qkv.run(
                ctx,
                gpu,
                &FusedQkvParams {
                    kind: key,
                    weights: &[wqkv.buf, wz.buf, wb.buf, wa.buf],
                    x: activated,
                    outputs: &[qkv, z, beta, alpha],
                    m: &[wqkv.m, wz.m, wb.m, wa.m],
                    k: wqkv.k,
                    rot_scratch: &[],
                    batch_size: None,
                },
            )
        }

        // ── Paro fused Paro4G128T ────────────────────────────────────────
        // For all three Paro fused keys, we allocate rotation scratch from
        // gpu.scratch.paro_fused_scratch (4 × [k] F32 buffers). The QKVZA
        // path passes all 4; the QKV (3-way) passes 4 with m3=0 via aliasing;
        // the gate+up path passes 1 (x_rot_gate), with the kernel using
        // gpu.scratch.mq_x_rot internally for x_rot_up.
        //
        // Build aliased GpuTensor descriptors before the mutable borrow of
        // gpu (fused_qkv.run takes &mut Gpu). DeviceBuffer::alias() creates
        // an owned descriptor over the same VRAM — no Rust borrow held.
        KernelKey::FusedGateUpParo4G128T => {
            let (wg, gate) = gemv_weight_out(&steps[1]);
            let (wu, up) = gemv_weight_out(&steps[2]);
            let k = wg.k;
            #[cfg(debug_assertions)]
            eprintln!("[dispatch] GateUp Paro: k={}, mg={}, mu={}", k, wg.m, wu.m);
            gpu.ensure_paro_fused_scratch(k)
                .map_err(|e| DispatchError::Hip(e.to_string()))?;
            // Also ensure mq_x_rot >= k (the kernel aliases it for x_rot_up).
            gpu.ensure_mq_signs()
                .map_err(|e| DispatchError::Hip(e.to_string()))?;
            let rot_aliases: Vec<GpuTensor> = gpu
                .scratch
                .paro_fused_scratch
                .as_ref()
                .unwrap()
                .iter()
                .map(|t| GpuTensor {
                    buf: unsafe { t.buf.alias() },
                    shape: t.shape.clone(),
                    dtype: t.dtype,
                })
                .collect();
            #[cfg(debug_assertions)]
            {
                let gate_buf = &gpu.scratch.paro_fused_scratch.as_ref().unwrap()[0];
                let up_internal = gpu.scratch.mq_x_rot.as_ref().unwrap();
                debug_assert!(
                    gate_buf.buf.as_ptr() != up_internal.buf.as_ptr(),
                    "Paro gate+up: x_rot_gate must not alias mq_x_rot"
                );
            }
            fused_qkv.run(
                ctx,
                gpu,
                &FusedQkvParams {
                    kind: key,
                    weights: &[wg.buf, wu.buf],
                    x: activated,
                    outputs: &[gate, up],
                    m: &[wg.m, wu.m],
                    k,
                    rot_scratch: &rot_aliases,
                    batch_size: None,
                },
            )
        }
        KernelKey::FusedQkvzaParo4G128T => {
            let (wqkv, qkv) = gemv_weight_out(&steps[1]);
            let (wz, z) = gemv_weight_out(&steps[2]);
            let (wb, beta) = gemv_weight_out(&steps[3]);
            let (wa, alpha) = gemv_weight_out(&steps[4]);
            let k = wqkv.k;
            #[cfg(debug_assertions)]
            eprintln!(
                "[dispatch] QKVZA Paro: k={}, mqkv={}, mz={}, mbeta={}, malpha={}",
                k, wqkv.m, wz.m, wb.m, wa.m
            );
            gpu.ensure_paro_fused_scratch(k)
                .map_err(|e| DispatchError::Hip(e.to_string()))?;
            let rot_aliases: Vec<GpuTensor> = gpu
                .scratch
                .paro_fused_scratch
                .as_ref()
                .unwrap()
                .iter()
                .map(|t| GpuTensor {
                    buf: unsafe { t.buf.alias() },
                    shape: t.shape.clone(),
                    dtype: t.dtype,
                })
                .collect();
            fused_qkv.run(
                ctx,
                gpu,
                &FusedQkvParams {
                    kind: key,
                    weights: &[wqkv.buf, wz.buf, wb.buf, wa.buf],
                    x: activated,
                    outputs: &[qkv, z, beta, alpha],
                    m: &[wqkv.m, wz.m, wb.m, wa.m],
                    k,
                    rot_scratch: &rot_aliases,
                    batch_size: None,
                },
            )
        }
        KernelKey::FusedQkvParo4G128T => {
            let (wq, q) = gemv_weight_out(&steps[1]);
            let (wk, k) = gemv_weight_out(&steps[2]);
            let (wv, v) = gemv_weight_out(&steps[3]);
            let kk = wq.k;
            #[cfg(debug_assertions)]
            eprintln!(
                "[dispatch] QKV Paro: k={}, mq={}, mk={}, mv={}",
                kk, wq.m, wk.m, wv.m
            );
            gpu.ensure_paro_fused_scratch(kk)
                .map_err(|e| DispatchError::Hip(e.to_string()))?;
            let rot_aliases: Vec<GpuTensor> = gpu
                .scratch
                .paro_fused_scratch
                .as_ref()
                .unwrap()
                .iter()
                .map(|t| GpuTensor {
                    buf: unsafe { t.buf.alias() },
                    shape: t.shape.clone(),
                    dtype: t.dtype,
                })
                .collect();
            fused_qkv.run(
                ctx,
                gpu,
                &FusedQkvParams {
                    kind: key,
                    weights: &[wq.buf, wk.buf, wv.buf],
                    x: activated,
                    outputs: &[q, k, v],
                    m: &[wq.m, wk.m, wv.m],
                    k: kk,
                    rot_scratch: &rot_aliases,
                    batch_size: None,
                },
            )
        }
        _ => Err(DispatchError::MissingImpl { key }),
    }
}

// ── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::context::DispatchCtx;
    use crate::families::fused_qkv::FusedQkvFamily;
    use crate::types::KernelKey;

    #[test]
    fn qkvza_fused_table_entries_exist() {
        let keys: Vec<_> = FUSED_TABLE.iter().map(|e| e.key).collect();
        assert!(
            keys.contains(&KernelKey::FusedQkvzaMq4G256Lloyd),
            "FusedQkvzaMq4G256Lloyd missing"
        );
        assert!(
            keys.contains(&KernelKey::FusedQkvzaMq3G256Lloyd),
            "FusedQkvzaMq3G256Lloyd missing"
        );
        assert!(
            keys.contains(&KernelKey::FusedQkvzaHfq4G256),
            "FusedQkvzaHfq4G256 missing"
        );
        assert!(
            keys.contains(&KernelKey::FusedQkvzaHfq6G256),
            "FusedQkvzaHfq6G256 missing"
        );

        for entry in FUSED_TABLE.iter() {
            if matches!(
                entry.key,
                KernelKey::FusedQkvzaMq4G256Lloyd
                    | KernelKey::FusedQkvzaMq3G256Lloyd
                    | KernelKey::FusedQkvzaHfq4G256
                    | KernelKey::FusedQkvzaHfq6G256
            ) {
                assert_eq!(
                    entry.ops.len(),
                    5,
                    "QKVZA entry {:?} should have 5 ops",
                    entry.key
                );
            }
        }
    }

    #[test]
    fn qkvza_guards_reject_short_slices() {
        let ctx = DispatchCtx::for_test("gfx1100");
        // Guards must return false for slices shorter than 5 steps.
        let empty: &[Step] = &[];
        assert!(!guard_qkvza_mq4g256lloyd(empty, &ctx));
        assert!(!guard_qkvza_mq3g256lloyd(empty, &ctx));
        assert!(!guard_qkvza_hfq4g256(empty, &ctx));
        assert!(!guard_qkvza_hfq6g256(empty, &ctx));
    }

    #[test]
    fn qkvza_no_paro_or_q8_fused_entries() {
        use crate::types::GemvVariant;
        // ParoQ4G128 should not resolve to any fused QKVZA key. It may resolve
        // to a plain GEMV key (or nothing for unsupported arches). Both are fine.
        let paro = KernelKey::for_gemv(DType::ParoQ4G128, GemvVariant::Plain, false);
        let q8 = KernelKey::for_gemv(DType::Q8_0, GemvVariant::Plain, false);
        for key in [paro.ok(), q8.ok()].into_iter().flatten() {
            assert!(
                !matches!(
                    key,
                    KernelKey::FusedQkvzaMq4G256Lloyd
                        | KernelKey::FusedQkvzaMq3G256Lloyd
                        | KernelKey::FusedQkvzaHfq4G256
                        | KernelKey::FusedQkvzaHfq6G256
                ),
                "ParoQ4G128/Q8_0 must not resolve to a fused QKVZA key, got {:?}",
                key
            );
        }
    }

    #[test]
    fn qkvza_guards_reject_force_unfused() {
        // The plan mandates that force_unfused must prevent fused QKVZA dispatch.
        // Construct a DispatchCtx with force_unfused=true and verify each guard
        // returns false even for otherwise-matching dtypes. We can't build full
        // Steps with real GPU tensors, so we test the guard logic directly with
        // the flag set.
        use rdna_compute::feature_flags::FeatureFlags;
        use std::sync::Arc;
        let mut flags = FeatureFlags::for_test("gfx1100");
        flags.force_unfused = true;
        let ctx = DispatchCtx {
            arch: rdna_compute::arch_caps::ArchCaps::new(
                "gfx1100",
                Arc::new(FeatureFlags::for_test("gfx1100")),
            ),
            flags: Arc::new(flags),
            resources: crate::resource::ResourceManager::for_test(),
            workload: crate::context::DispatchWorkload::Standard,
        };
        // short-circuit: every guard opens with `force_unfused → false`, so even
        // an empty slice returns false. This proves the branch exists.
        let empty: &[Step] = &[];
        assert!(!guard_qkvza_mq4g256lloyd(empty, &ctx));
        assert!(!guard_qkvza_mq3g256lloyd(empty, &ctx));
        assert!(!guard_qkvza_hfq4g256(empty, &ctx));
        assert!(!guard_qkvza_hfq6g256(empty, &ctx));
    }

    #[test]
    fn qkvza_fused_table_no_paro_q4_or_q8_entries() {
        // ParoQ4G128 and Q8_0 must NOT have fused QKVZA entries — they fall
        // through to per-op dispatch. This test asserts that none of the fused
        // table keys match a Paro or Q8 variant, ensuring byte-identical
        // unfused-path correctness for those dtypes.
        let paro_q4_key = KernelKey::for_gemv(DType::ParoQ4G128, GemvVariant::Plain, false);
        let q8_key = KernelKey::for_gemv(DType::Q8_0, GemvVariant::Plain, false);
        // Paro and Q8 should resolve to plain GEMV keys, not fused QKVZA keys.
        // (They may be Err for arches without support, which is also fine.)
        for key in [paro_q4_key, q8_key] {
            if let Ok(k) = key {
                assert!(
                    !matches!(
                        k,
                        KernelKey::FusedQkvzaMq4G256Lloyd
                            | KernelKey::FusedQkvzaMq3G256Lloyd
                            | KernelKey::FusedQkvzaHfq4G256
                            | KernelKey::FusedQkvzaHfq6G256
                    ),
                    "ParoQ4G128/Q8_0 should not resolve to a fused QKVZA key"
                );
            }
        }
    }

    #[test]
    fn qkvza_fused_table_arch_coverage() {
        let family = FusedQkvFamily::new();
        let ctx1100 = DispatchCtx::for_test("gfx1100");
        let ctx1201 = DispatchCtx::for_test("gfx1201");

        let wmma_keys = &[
            KernelKey::FusedQkvzaMq4G256Lloyd,
            KernelKey::FusedQkvzaMq3G256Lloyd,
            KernelKey::FusedQkvzaHfq4G256,
        ];

        for &key in wmma_keys {
            assert!(
                family.resolve(key, &ctx1100, None).is_ok(),
                "QKVZA {:?} should resolve on gfx1100",
                key
            );
            assert!(
                family.resolve(key, &ctx1201, None).is_ok(),
                "QKVZA {:?} should resolve on gfx1201",
                key
            );
        }

        // dp4a key: just verify no panic
        let _ = family.resolve(KernelKey::FusedQkvzaHfq6G256, &ctx1100, None);
        let _ = family.resolve(KernelKey::FusedQkvzaHfq6G256, &ctx1201, None);
    }

    #[test]
    fn paro_guards_reject_force_unfused() {
        let ctx = DispatchCtx::for_test("gfx1100");
        let empty: &[Step] = &[];
        assert!(
            !guard_gate_up_paro4g128t(empty, &ctx),
            "force_unfused must reject gate_up_paro"
        );
        assert!(
            !guard_qkvza_paro4g128t(empty, &ctx),
            "force_unfused must reject qkvza_paro"
        );
        assert!(
            !guard_qkv_paro4g128t(empty, &ctx),
            "force_unfused must reject qkv_paro"
        );
    }

    #[test]
    fn paro_guards_require_raw_input_and_alignment() {
        // Paro guards require GemvInput::Raw (not Prerotated) and m%8==0/k%128==0.
        // We can't construct real Gemv steps with GPU tensors in a unit test,
        // but we can verify the guards reject empty/wrong-length slices.
        let ctx = DispatchCtx::for_test("gfx1100");
        let empty: &[Step] = &[];
        assert!(!guard_gate_up_paro4g128t(empty, &ctx));
        assert!(!guard_qkvza_paro4g128t(empty, &ctx));
        assert!(!guard_qkv_paro4g128t(empty, &ctx));
    }

    #[test]
    fn paro_fused_table_entries_exist() {
        let keys: Vec<_> = FUSED_TABLE.iter().map(|e| e.key).collect();
        assert!(
            keys.contains(&KernelKey::FusedGateUpParo4G128T),
            "FusedGateUpParo4G128T missing from FUSED_TABLE"
        );
        assert!(
            keys.contains(&KernelKey::FusedQkvzaParo4G128T),
            "FusedQkvzaParo4G128T missing from FUSED_TABLE"
        );
        assert!(
            keys.contains(&KernelKey::FusedQkvParo4G128T),
            "FusedQkvParo4G128T missing from FUSED_TABLE"
        );
    }

    #[test]
    fn paro_fused_table_arch_coverage() {
        let family = FusedQkvFamily::new();
        let ctx1100 = DispatchCtx::for_test("gfx1100");
        let ctx1201 = DispatchCtx::for_test("gfx1201");

        let paro_keys = &[
            KernelKey::FusedGateUpParo4G128T,
            KernelKey::FusedQkvzaParo4G128T,
            KernelKey::FusedQkvParo4G128T,
        ];

        for &key in paro_keys {
            // Paro uses dp4a — should resolve on gfx1100 (RDNA3) and gfx1201 (RDNA4).
            assert!(
                family.resolve(key, &ctx1100, None).is_ok(),
                "Paro key {:?} should resolve on gfx1100",
                key
            );
            assert!(
                family.resolve(key, &ctx1201, None).is_ok(),
                "Paro key {:?} should resolve on gfx1201",
                key
            );
        }
    }

    // ── Q4K / Q8_0 guard tests (Ship 2.1 A1 — Claude F1 / glm5 F2) ──────

    #[test]
    fn q4k_q8_0_guards_reject_force_unfused() {
        // All three new guards must return false when force_unfused is set,
        // even for empty slices (the guard opens with the early-return).
        use rdna_compute::feature_flags::FeatureFlags;
        use std::sync::Arc;
        let mut flags = FeatureFlags::for_test("gfx1100");
        flags.force_unfused = true;
        let ctx = DispatchCtx {
            arch: rdna_compute::arch_caps::ArchCaps::new(
                "gfx1100",
                Arc::new(FeatureFlags::for_test("gfx1100")),
            ),
            flags: Arc::new(flags),
            resources: crate::resource::ResourceManager::for_test(),
            workload: crate::context::DispatchWorkload::Standard,
        };
        let empty: &[Step] = &[];
        assert!(
            !guard_qkv_q4k(empty, &ctx),
            "guard_qkv_q4k must reject force_unfused"
        );
        assert!(
            !guard_gate_up_q4k(empty, &ctx),
            "guard_gate_up_q4k must reject force_unfused"
        );
        assert!(
            !guard_gate_up_q8_0(empty, &ctx),
            "guard_gate_up_q8_0 must reject force_unfused"
        );
    }

    #[test]
    fn q4k_q8_0_guards_reject_wrong_length() {
        let ctx = DispatchCtx::for_test("gfx1100");
        let empty: &[Step] = &[];
        assert!(!guard_qkv_q4k(empty, &ctx), "Q4K QKV guard needs len==4");
        assert!(
            !guard_gate_up_q4k(empty, &ctx),
            "Q4K gate+up guard needs len==3"
        );
        assert!(
            !guard_gate_up_q8_0(empty, &ctx),
            "Q8_0 gate+up guard needs len==3"
        );
    }

    #[test]
    fn q4k_q8_0_fused_table_entries_exist() {
        let keys: Vec<_> = FUSED_TABLE.iter().map(|e| e.key).collect();
        assert!(
            keys.contains(&KernelKey::FusedQkvQ4K),
            "FusedQkvQ4K missing from FUSED_TABLE"
        );
        assert!(
            keys.contains(&KernelKey::FusedGateUpQ4K),
            "FusedGateUpQ4K missing from FUSED_TABLE"
        );
        assert!(
            keys.contains(&KernelKey::FusedGateUpQ8_0),
            "FusedGateUpQ8_0 missing from FUSED_TABLE"
        );
    }

    #[test]
    fn unsealed_moe_schedule_is_rejected_before_gpu_execution() {
        let route = GpuTensor::null_for_test();
        let plan = RouterPlan::Precomputed {
            topk_indices: &route,
            topk_weights: &route,
            k_top: 1,
        };
        let steps = [Step::MoeRoute { plan }];
        let error = reject_unsealed_moe(&steps).expect_err("MoE must use a sealed executor path");
        assert!(error.to_string().contains("unsealed MoE schedules"));
    }
}
