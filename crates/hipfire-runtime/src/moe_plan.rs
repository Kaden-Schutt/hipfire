// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 hipfire contributors

//! Manifest-derived MoE placement, rank views, and storage ownership.
//!
//! This module is the only owner of the resolved expert placement contract.
//! It is CPU-only: source identities and logical shapes come from G3, topology
//! comes from G1, and architecture carriers bind borrowed pointer tables through
//! [`ExpertPlan::bind_expert_ref`]. No family receives an allocator or a store
//! representation, and no family-side teardown path exists.

use std::fmt;

use hipfire_dispatch::families::moe::{ExpertExecutionPlan, MoeExpertRef};
use hipfire_dispatch::pipeline::StepCollective;
use hipfire_hardware::{CollectiveHint, DeviceMesh, DimKind, MeshEpoch};
use rdna_compute::{DType, GpuTensor};

use crate::tp_shard::ExpertAssign;
use crate::weight_manifest::{
    collective_schedule, validate_expert_group_specs, ExpertGroupSpec, ExpertParallelism,
    ExpertResourceRequirements, ExpertSourceLayout, ShardPolicy, WeightEntry,
};

/// One logical expert and its deterministic owner-local slot.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ExpertPlacement {
    pub global_id: usize,
    /// Rank-local owner index within the named DeviceMesh group.
    pub owner: usize,
    pub local_slot: usize,
}

/// Logical dimensions shared by the fused gate/up and down projections.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ExpertShape {
    pub expert_m: usize,
    pub expert_k: usize,
    pub fused_gate_up: bool,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ExpertPlanError(String);

impl ExpertPlanError {
    fn new(message: impl Into<String>) -> Self {
        Self(message.into())
    }
}

impl fmt::Display for ExpertPlanError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for ExpertPlanError {}

struct ExpertPointerTables {
    gate_up: GpuTensor,
    down: GpuTensor,
    dummy_gate_up: Option<GpuTensor>,
}

struct ExpertStorageOwner {
    slots: Vec<ExpertPlacement>,
    resident: Vec<bool>,
    rank_tables: Vec<Option<ExpertPointerTables>>,
}

impl ExpertStorageOwner {
    fn new(slots: Vec<ExpertPlacement>, group_size: usize) -> Self {
        Self {
            resident: vec![false; slots.len()],
            slots,
            rank_tables: (0..group_size).map(|_| None).collect(),
        }
    }

    fn index_of(&self, placement: ExpertPlacement) -> Option<usize> {
        self.slots
            .iter()
            .position(|candidate| *candidate == placement)
    }

    fn rank_is_resident(&self, rank: usize) -> bool {
        self.slots
            .iter()
            .enumerate()
            .filter(|(_, placement)| placement.owner == rank)
            .all(|(index, _)| self.resident[index])
    }

    fn clear(&mut self) {
        self.resident.fill(false);
    }

    fn resident_count(&self) -> usize {
        self.resident.iter().filter(|resident| **resident).count()
    }
}

/// A sealed, manifest-derived expert plan.
///
/// Every placement, source identity, shape, resource requirement, rank-local
/// view, and collective row is private and fixed at construction. The only
/// mutable state is the owner transaction's resident bitmap; it is never
/// transferred to a family.
pub struct ExpertPlan {
    group: String,
    layer: Option<usize>,
    n_experts: usize,
    parallelism: ExpertParallelism,
    assignment: ExpertAssign,
    shape: ExpertShape,
    source_dtype: DType,
    source_layout: ExpertSourceLayout,
    resources: ExpertResourceRequirements,
    router: String,
    execution: String,
    execution_plan: ExpertExecutionPlan,
    mesh_epoch: MeshEpoch,
    group_devices: Vec<usize>,
    collective: Option<CollectiveHint>,
    collective_row: Option<String>,
    owner_views: Vec<Vec<usize>>,
    owner_partition: Vec<(usize, usize, usize)>,
    owner: ExpertStorageOwner,
}

impl fmt::Debug for ExpertPlan {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ExpertPlan")
            .field("group", &self.group)
            .field("layer", &self.layer)
            .field("n_experts", &self.n_experts)
            .field("parallelism", &self.parallelism)
            .field("assignment", &self.assignment)
            .field("shape", &self.shape)
            .field("source_dtype", &self.source_dtype)
            .field("source_layout", &self.source_layout)
            .field("resources", &self.resources)
            .field("router", &self.router)
            .field("execution", &self.execution)
            .field("execution_plan", &self.execution_plan)
            .field("mesh_epoch", &self.mesh_epoch)
            .field("group_devices", &self.group_devices)
            .field("collective", &self.collective)
            .field("collective_row", &self.collective_row)
            .field("resident_slots", &self.owner.resident_count())
            .finish()
    }
}

impl ExpertPlan {
    /// Resolve one expert declaration against the G3 logical manifest and G1
    /// named mesh. No GPU, source file, allocator, or `WeightStore` is touched.
    pub fn from_manifest(
        spec: &ExpertGroupSpec,
        manifest: &[WeightEntry],
        mesh: &DeviceMesh,
    ) -> Result<Self, ExpertPlanError> {
        validate_expert_group_specs(std::slice::from_ref(spec), manifest)
            .map_err(ExpertPlanError::new)?;
        validate_spec(spec, mesh)?;
        let (shape, source_dtype) = resolve_shape(spec, manifest)?;
        if !shape.fused_gate_up {
            return Err(ExpertPlanError::new(format!(
                "expert group '{}' uses separate gate/up sources, unsupported by the generic executor",
                spec.group
            )));
        }
        let execution_plan = parse_execution(&spec.execution, spec)?;
        validate_execution_dtype(execution_plan, source_dtype, spec)?;

        let group_devices = resolve_group_devices(spec, manifest, mesh);
        if group_devices.is_empty() {
            return Err(ExpertPlanError::new(format!(
                "expert group '{}' resolved to an empty mesh group",
                spec.group
            )));
        }
        let group_size = group_devices.len();
        let owner = resolve_placements(
            spec.n_experts,
            group_size,
            spec.parallelism,
            spec.assignment,
        );
        let owner_views = (0..group_size)
            .map(|rank| {
                owner
                    .iter()
                    .filter(|placement| placement.owner == rank)
                    .map(|placement| placement.global_id)
                    .collect()
            })
            .collect();
        let owner_partition = owner
            .iter()
            .map(|placement| (placement.global_id, placement.owner, placement.local_slot))
            .collect();
        let (collective, collective_row) = resolve_collective(spec, manifest)?;

        Ok(Self {
            group: spec.group.clone(),
            layer: spec.layer,
            n_experts: spec.n_experts,
            parallelism: spec.parallelism,
            assignment: spec.assignment,
            shape,
            source_dtype,
            source_layout: spec.source_layout.clone(),
            resources: spec.resources,
            router: spec.router.clone(),
            execution: spec.execution.clone(),
            execution_plan,
            mesh_epoch: mesh.epoch(),
            group_devices,
            collective,
            collective_row,
            owner_views,
            owner_partition,
            owner: ExpertStorageOwner::new(owner, group_size),
        })
    }

    pub fn group(&self) -> &str {
        &self.group
    }

    pub fn layer(&self) -> Option<usize> {
        self.layer
    }

    pub fn n_experts(&self) -> usize {
        self.n_experts
    }

    pub fn group_size(&self) -> usize {
        self.group_devices.len()
    }

    pub fn assignment(&self) -> ExpertAssign {
        self.assignment
    }

    pub fn parallelism(&self) -> ExpertParallelism {
        self.parallelism
    }

    pub fn shape(&self) -> ExpertShape {
        self.shape
    }

    pub fn source_dtype(&self) -> DType {
        self.source_dtype
    }

    pub fn source_layout(&self) -> &ExpertSourceLayout {
        &self.source_layout
    }

    pub fn resources(&self) -> ExpertResourceRequirements {
        self.resources
    }

    pub fn router(&self) -> &str {
        &self.router
    }

    pub fn execution(&self) -> &str {
        &self.execution
    }

    pub fn execution_plan(&self) -> ExpertExecutionPlan {
        self.execution_plan
    }

    pub fn mesh_epoch(&self) -> MeshEpoch {
        self.mesh_epoch
    }

    /// Global device IDs in the exact named-axis order used by collectives.
    pub fn group_devices(&self) -> &[usize] {
        &self.group_devices
    }

    /// The one ordered G3 manifest row that authorizes the routed reduction.
    pub fn collective_row(&self) -> Option<&str> {
        self.collective_row.as_deref()
    }

    /// The single post-combine collective implied by the declared parallelism.
    pub fn collective(&self) -> Option<CollectiveHint> {
        self.collective
    }

    pub fn placements(&self) -> &[ExpertPlacement] {
        &self.owner.slots
    }

    pub fn placement(&self, global_id: usize, owner: usize) -> Option<ExpertPlacement> {
        self.owner
            .slots
            .iter()
            .copied()
            .find(|placement| placement.global_id == global_id && placement.owner == owner)
    }

    pub fn owned_experts(&self, rank: usize) -> Result<&[usize], ExpertPlanError> {
        self.owner_views
            .get(rank)
            .map(Vec::as_slice)
            .ok_or_else(|| {
                ExpertPlanError::new(format!(
                    "expert group '{}' rank {rank} is outside group size {}",
                    self.group,
                    self.group_size()
                ))
            })
    }

    /// Commit the rank-local pointer tables to this owner. The plan validates
    /// the table ABI before retaining the tensors; binding later borrows only
    /// these committed values.
    pub fn commit_rank_tables(
        &mut self,
        rank: usize,
        gate_up: GpuTensor,
        down: GpuTensor,
        dummy_gate_up: Option<GpuTensor>,
    ) -> Result<(), ExpertPlanError> {
        if rank >= self.group_size() {
            return Err(ExpertPlanError::new(format!(
                "expert group '{}' rank {rank} is outside group size {}",
                self.group,
                self.group_size()
            )));
        }
        validate_pointer_table(&gate_up, "gate/up", self.n_experts)?;
        validate_pointer_table(&down, "down", self.n_experts)?;
        if let Some(dummy) = &dummy_gate_up {
            validate_dummy_table(dummy)?;
        }
        let tables = self
            .owner
            .rank_tables
            .get_mut(rank)
            .expect("rank checked above");
        if tables.is_some() {
            return Err(ExpertPlanError::new(format!(
                "expert group '{}' rank {rank} pointer tables are already committed",
                self.group
            )));
        }
        *tables = Some(ExpertPointerTables {
            gate_up,
            down,
            dummy_gate_up,
        });
        Ok(())
    }

    /// Bind the committed rank-local pointer tables to an opaque executable
    /// view. A rank cannot bind until its owned placements are resident.
    pub fn bind_expert_ref<'a>(&'a self, rank: usize) -> Result<MoeExpertRef<'a>, ExpertPlanError> {
        let owned = self.owned_experts(rank)?;
        if !self.owner.rank_is_resident(rank) {
            return Err(ExpertPlanError::new(format!(
                "expert group '{}' rank {rank} has nonresident placements",
                self.group
            )));
        }
        let tables = self
            .owner
            .rank_tables
            .get(rank)
            .and_then(Option::as_ref)
            .ok_or_else(|| {
                ExpertPlanError::new(format!(
                    "expert group '{}' rank {rank} has no committed pointer tables",
                    self.group
                ))
            })?;
        let collective_kind = match self.collective {
            Some(CollectiveHint::AllReduce { kind }) => Some(kind),
            _ => None,
        };
        // SAFETY: this private plan path has checked the committed rank table,
        // resident owner placements, and all metadata comes from `self`.
        let binding = unsafe {
            hipfire_dispatch::families::moe::MoeExpertRefBinding::from_validated_plan(
                &tables.gate_up,
                &tables.down,
                tables.dummy_gate_up.as_ref(),
                self.source_dtype,
                self.n_experts,
                self.shape.expert_m,
                self.shape.expert_k,
                owned,
                &self.owner_partition,
                &self.router,
                collective_kind,
                rank,
                &self.group_devices,
                self.mesh_epoch,
            )
        };
        let view = MoeExpertRef::from_binding(binding);
        view.validate()
            .map_err(|error| ExpertPlanError::new(error.to_string()))?;
        Ok(view)
    }

    /// Build the descriptor for the one collective attached to the combine
    /// position. Single is an explicit identity and emits `None`.
    pub fn step_collective(
        &self,
        rank: usize,
        dim: usize,
    ) -> Result<StepCollective, ExpertPlanError> {
        if rank >= self.group_size() {
            return Err(ExpertPlanError::new(format!(
                "expert group '{}' rank {rank} is outside group size {}",
                self.group,
                self.group_size()
            )));
        }
        if dim == 0 {
            return Err(ExpertPlanError::new(
                "expert collective output dimension must be nonzero",
            ));
        }
        match self.collective {
            None => Ok(StepCollective::None),
            Some(CollectiveHint::AllReduce { kind }) => Ok(StepCollective::all_reduce(
                kind,
                dim,
                self.group_devices.clone(),
                self.mesh_epoch,
                rank,
            )),
            Some(CollectiveHint::BandXfer { .. }) => Err(ExpertPlanError::new(
                "pipeline band transfer is not an expert reduction",
            )),
        }
    }

    /// Start one owner-controlled load transaction. A dropped or explicitly
    /// rolled-back transaction removes only its staged slots.
    pub fn begin_load(&mut self) -> ExpertLoadTxn<'_> {
        ExpertLoadTxn {
            owner: &mut self.owner,
            staged: Vec::new(),
            finished: false,
        }
    }

    /// Idempotent teardown. No family-side free path exists; all resident
    /// slots return to the owner baseline and repeated unloads are harmless.
    pub fn unload(&mut self) {
        self.owner.clear();
    }

    pub fn resident_slots(&self) -> usize {
        self.owner.resident_count()
    }

    pub fn allocated_slots(&self) -> usize {
        self.owner.slots.len()
    }
}

/// Owner-scoped transactional expert load state.
pub struct ExpertLoadTxn<'a> {
    owner: &'a mut ExpertStorageOwner,
    staged: Vec<usize>,
    finished: bool,
}

impl ExpertLoadTxn<'_> {
    /// Reserve one manifest placement. Duplicate reservations and reuse after
    /// commit/rollback are refused.
    pub fn reserve(&mut self, placement: ExpertPlacement) -> Result<(), ExpertPlanError> {
        if self.finished {
            return Err(ExpertPlanError::new(
                "expert load transaction is already finished",
            ));
        }
        let index = self.owner.index_of(placement).ok_or_else(|| {
            ExpertPlanError::new(format!("unknown expert placement {placement:?}"))
        })?;
        if self.owner.resident[index] {
            return Err(ExpertPlanError::new(format!(
                "expert placement {placement:?} is already resident"
            )));
        }
        self.owner.resident[index] = true;
        self.staged.push(index);
        Ok(())
    }

    /// Commit the staged reservations. The owner remains responsible for
    /// teardown; committing never transfers ownership to a family.
    pub fn commit(mut self) {
        self.finished = true;
    }

    /// Roll back staged reservations and close the transaction.
    pub fn rollback(&mut self) {
        for index in self.staged.drain(..) {
            self.owner.resident[index] = false;
        }
        self.finished = true;
    }
}

impl Drop for ExpertLoadTxn<'_> {
    fn drop(&mut self) {
        if !self.finished {
            for index in self.staged.drain(..) {
                self.owner.resident[index] = false;
            }
        }
    }
}

fn validate_spec(spec: &ExpertGroupSpec, mesh: &DeviceMesh) -> Result<(), ExpertPlanError> {
    if spec.group.is_empty() {
        return Err(ExpertPlanError::new("expert group identity is empty"));
    }
    if spec.n_experts == 0 {
        return Err(ExpertPlanError::new(format!(
            "expert group '{}' has no experts",
            spec.group
        )));
    }
    if spec.resources.bytes_per_expert == 0 {
        return Err(ExpertPlanError::new(format!(
            "expert group '{}' bytes_per_expert is zero",
            spec.group
        )));
    }
    if spec.resources.alignment == 0 || !spec.resources.alignment.is_power_of_two() {
        return Err(ExpertPlanError::new(format!(
            "expert group '{}' alignment={} is invalid",
            spec.group, spec.resources.alignment
        )));
    }
    spec.n_experts
        .checked_mul(spec.resources.bytes_per_expert)
        .ok_or_else(|| ExpertPlanError::new("expert resource capacity overflows usize"))?;
    if spec.router.is_empty() || spec.execution.is_empty() {
        return Err(ExpertPlanError::new(format!(
            "expert group '{}' router/execution identity is empty",
            spec.group
        )));
    }
    let required_axis = match spec.parallelism {
        ExpertParallelism::Single => None,
        ExpertParallelism::TensorParallel => Some(DimKind::Tp),
        ExpertParallelism::ExpertParallel => Some(DimKind::Ep),
    };
    let group_size = required_axis.map_or(1, |kind| mesh.size_of(kind));
    if let Some(kind) = required_axis {
        if !mesh.axes().iter().any(|axis| axis.kind == kind) {
            return Err(ExpertPlanError::new(format!(
                "expert group '{}' requires a named {:?} mesh axis",
                spec.group, kind
            )));
        }
        if group_size < 2 {
            return Err(ExpertPlanError::new(format!(
                "expert group '{}' parallel mesh group must have at least two ranks",
                spec.group
            )));
        }
    }
    if group_size == 0 {
        return Err(ExpertPlanError::new("expert group resolved to zero ranks"));
    }
    if matches!(spec.parallelism, ExpertParallelism::ExpertParallel)
        && !spec.n_experts.is_multiple_of(group_size)
    {
        return Err(ExpertPlanError::new(format!(
            "expert group '{}' n_experts={} is not divisible by group_size={group_size}",
            spec.group, spec.n_experts
        )));
    }
    Ok(())
}

fn parse_execution(
    execution: &str,
    spec: &ExpertGroupSpec,
) -> Result<ExpertExecutionPlan, ExpertPlanError> {
    match execution {
        "indexed_quantized" => Ok(ExpertExecutionPlan::IndexedQuantized),
        "grouped_quantized" => Ok(ExpertExecutionPlan::GroupedQuantized),
        "per_expert_fallback" => Err(ExpertPlanError::new(format!(
            "expert group '{}' uses PerExpertFallback, which is not a Step protocol",
            spec.group
        ))),
        other => Err(ExpertPlanError::new(format!(
            "expert group '{}' has unsupported execution identity '{other}'",
            spec.group
        ))),
    }
}

fn validate_execution_dtype(
    execution: ExpertExecutionPlan,
    dtype: DType,
    spec: &ExpertGroupSpec,
) -> Result<(), ExpertPlanError> {
    let supported = match execution {
        ExpertExecutionPlan::IndexedQuantized => matches!(
            dtype,
            DType::MQ4G256 | DType::MQ6G256 | DType::MQ4G256V2 | DType::MQ6G256V2
        ),
        ExpertExecutionPlan::GroupedQuantized => matches!(
            dtype,
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
        ExpertExecutionPlan::PerExpertFallback => false,
    };
    if supported {
        Ok(())
    } else {
        Err(ExpertPlanError::new(format!(
            "expert group '{}' execution {:?} has no generic kernel for source dtype {dtype:?}",
            spec.group, execution
        )))
    }
}

fn resolve_group_devices(
    spec: &ExpertGroupSpec,
    manifest: &[WeightEntry],
    mesh: &DeviceMesh,
) -> Vec<usize> {
    let n_layers = manifest
        .iter()
        .filter_map(|entry| entry.layer)
        .max()
        .map_or(1, |layer| layer.saturating_add(1));
    let mut coord = mesh.coord_of(0);
    if let Some(layer) = spec.layer {
        if let Some(index) = mesh.axes().iter().position(|axis| axis.kind == DimKind::Pp) {
            coord[index] = mesh.stage_for_layer(layer, n_layers);
        }
    }
    match spec.parallelism {
        ExpertParallelism::Single => vec![mesh.device_of(&coord)],
        ExpertParallelism::TensorParallel => mesh.group_along(DimKind::Tp, &coord),
        ExpertParallelism::ExpertParallel => mesh.group_along(DimKind::Ep, &coord),
    }
}

fn resolve_placements(
    n_experts: usize,
    group_size: usize,
    parallelism: ExpertParallelism,
    assignment: ExpertAssign,
) -> Vec<ExpertPlacement> {
    let capacity = match parallelism {
        ExpertParallelism::TensorParallel => n_experts.saturating_mul(group_size),
        ExpertParallelism::Single | ExpertParallelism::ExpertParallel => n_experts,
    };
    let mut next_slot = vec![0usize; group_size];
    let mut placements = Vec::with_capacity(capacity);
    for global_id in 0..n_experts {
        match parallelism {
            ExpertParallelism::Single => placements.push(ExpertPlacement {
                global_id,
                owner: 0,
                local_slot: global_id,
            }),
            ExpertParallelism::TensorParallel => {
                for owner in 0..group_size {
                    placements.push(ExpertPlacement {
                        global_id,
                        owner,
                        local_slot: global_id,
                    });
                }
            }
            ExpertParallelism::ExpertParallel => {
                let per = n_experts / group_size;
                let owner = match assignment {
                    ExpertAssign::Contiguous => global_id / per,
                    ExpertAssign::Stride => global_id % group_size,
                };
                let local_slot = next_slot[owner];
                next_slot[owner] += 1;
                placements.push(ExpertPlacement {
                    global_id,
                    owner,
                    local_slot,
                });
            }
        }
    }
    placements
}

fn source_names<'a>(layout: &'a ExpertSourceLayout) -> Vec<(&'static str, Vec<&'a str>)> {
    match layout {
        ExpertSourceLayout::PackedFused {
            gate_up,
            down,
            sidecars,
        } => vec![
            ("gate_up", vec![gate_up.as_str()]),
            ("down", vec![down.as_str()]),
            ("sidecar", sidecars.iter().map(String::as_str).collect()),
        ],
        ExpertSourceLayout::PackedSeparate {
            gate,
            up,
            down,
            sidecars,
        } => vec![
            ("gate", vec![gate.as_str()]),
            ("up", vec![up.as_str()]),
            ("down", vec![down.as_str()]),
            ("sidecar", sidecars.iter().map(String::as_str).collect()),
        ],
        ExpertSourceLayout::PerExpertFused {
            gate_up,
            down,
            sidecars,
        } => vec![
            ("gate_up", gate_up.iter().map(String::as_str).collect()),
            ("down", down.iter().map(String::as_str).collect()),
            ("sidecar", sidecars.iter().map(String::as_str).collect()),
        ],
        ExpertSourceLayout::PerExpertSeparate {
            gate,
            up,
            down,
            sidecars,
        } => vec![
            ("gate", gate.iter().map(String::as_str).collect()),
            ("up", up.iter().map(String::as_str).collect()),
            ("down", down.iter().map(String::as_str).collect()),
            ("sidecar", sidecars.iter().map(String::as_str).collect()),
        ],
    }
}

fn checked_numel(tensor: &GpuTensor, label: &str) -> Result<usize, ExpertPlanError> {
    tensor
        .shape
        .iter()
        .try_fold(1usize, |elements, dimension| {
            elements.checked_mul(*dimension)
        })
        .ok_or_else(|| ExpertPlanError::new(format!("{label} logical shape overflows")))
}

fn validate_pointer_table(
    tensor: &GpuTensor,
    label: &str,
    n_experts: usize,
) -> Result<(), ExpertPlanError> {
    let pointer_slots = n_experts
        .checked_mul(2)
        .ok_or_else(|| ExpertPlanError::new("pointer-table slot count overflows"))?;
    let required_bytes = pointer_slots
        .checked_mul(DType::F32.size())
        .ok_or_else(|| ExpertPlanError::new("pointer-table byte capacity overflows"))?;
    let logical_elements = checked_numel(tensor, label)?;
    if tensor.dtype != DType::F32
        || tensor.shape.as_slice() != [pointer_slots]
        || logical_elements < pointer_slots
        || tensor.buf.size() < required_bytes
    {
        return Err(ExpertPlanError::new(format!(
            "{label} pointer table must be F32 [{pointer_slots}] with {required_bytes} physical bytes"
        )));
    }
    Ok(())
}

fn validate_dummy_table(tensor: &GpuTensor) -> Result<(), ExpertPlanError> {
    let logical_elements = checked_numel(tensor, "dummy gate/up table")?;
    let required_bytes = logical_elements
        .checked_mul(DType::F32.size())
        .ok_or_else(|| ExpertPlanError::new("dummy gate/up table byte capacity overflows"))?;
    if tensor.dtype != DType::F32 || logical_elements == 0 || tensor.buf.size() < required_bytes {
        return Err(ExpertPlanError::new(
            "dummy gate/up table must be nonempty F32 storage with physical capacity",
        ));
    }
    Ok(())
}

fn entry_for<'a>(
    spec: &ExpertGroupSpec,
    manifest: &'a [WeightEntry],
    label: &str,
    name: &str,
) -> Result<&'a WeightEntry, ExpertPlanError> {
    manifest
        .iter()
        .find(|entry| entry.name == name && entry.layer == spec.layer)
        .ok_or_else(|| {
            ExpertPlanError::new(format!(
                "expert group '{}' layer {:?} missing {label} source '{name}'",
                spec.group, spec.layer
            ))
        })
}

fn check_shape(
    spec: &ExpertGroupSpec,
    label: &str,
    entry: &WeightEntry,
    per_expert: bool,
) -> Result<Vec<usize>, ExpertPlanError> {
    if !entry.dtype_constraint.accepts(entry.dtype) {
        return Err(ExpertPlanError::new(format!(
            "expert group '{}' {label} source '{}' violates dtype constraint",
            spec.group, entry.name
        )));
    }
    let shape = &entry.logical_shape;
    let valid_rank = if per_expert {
        shape.len() == 2
    } else {
        shape.len() == 3
    };
    if !valid_rank || shape.iter().any(|dim| *dim == 0) {
        return Err(ExpertPlanError::new(format!(
            "expert group '{}' layer {:?} {label} source '{}' has invalid logical_shape {:?}",
            spec.group, spec.layer, entry.name, shape
        )));
    }
    if !per_expert && shape[0] != spec.n_experts {
        return Err(ExpertPlanError::new(format!(
            "expert group '{}' layer {:?} {label} source '{}' has expert axis {}, expected {}",
            spec.group, spec.layer, entry.name, shape[0], spec.n_experts
        )));
    }
    Ok(if per_expert {
        shape.clone()
    } else {
        shape[1..].to_vec()
    })
}

fn resolve_shape(
    spec: &ExpertGroupSpec,
    manifest: &[WeightEntry],
) -> Result<(ExpertShape, DType), ExpertPlanError> {
    let per_expert = matches!(
        spec.source_layout,
        ExpertSourceLayout::PerExpertFused { .. } | ExpertSourceLayout::PerExpertSeparate { .. }
    );
    for (label, names) in source_names(&spec.source_layout) {
        for name in names {
            let entry = entry_for(spec, manifest, label, name)?;
            if !entry.dtype_constraint.accepts(entry.dtype) {
                return Err(ExpertPlanError::new(format!(
                    "expert group '{}' {label} source '{}' violates dtype constraint",
                    spec.group, name
                )));
            }
        }
    }
    let (gate_shapes, up_shapes, down_shapes, fused) = match &spec.source_layout {
        ExpertSourceLayout::PackedFused { gate_up, down, .. } => (
            vec![check_shape(
                spec,
                "gate_up",
                entry_for(spec, manifest, "gate_up", gate_up)?,
                false,
            )?],
            Vec::new(),
            vec![check_shape(
                spec,
                "down",
                entry_for(spec, manifest, "down", down)?,
                false,
            )?],
            true,
        ),
        ExpertSourceLayout::PackedSeparate { gate, up, down, .. } => (
            vec![check_shape(
                spec,
                "gate",
                entry_for(spec, manifest, "gate", gate)?,
                false,
            )?],
            vec![check_shape(
                spec,
                "up",
                entry_for(spec, manifest, "up", up)?,
                false,
            )?],
            vec![check_shape(
                spec,
                "down",
                entry_for(spec, manifest, "down", down)?,
                false,
            )?],
            false,
        ),
        ExpertSourceLayout::PerExpertFused { gate_up, down, .. } => (
            gate_up
                .iter()
                .map(|name| {
                    check_shape(
                        spec,
                        "gate_up",
                        entry_for(spec, manifest, "gate_up", name)?,
                        true,
                    )
                })
                .collect::<Result<_, _>>()?,
            Vec::new(),
            down.iter()
                .map(|name| {
                    check_shape(spec, "down", entry_for(spec, manifest, "down", name)?, true)
                })
                .collect::<Result<_, _>>()?,
            true,
        ),
        ExpertSourceLayout::PerExpertSeparate { gate, up, down, .. } => (
            gate.iter()
                .map(|name| {
                    check_shape(spec, "gate", entry_for(spec, manifest, "gate", name)?, true)
                })
                .collect::<Result<_, _>>()?,
            up.iter()
                .map(|name| check_shape(spec, "up", entry_for(spec, manifest, "up", name)?, true))
                .collect::<Result<_, _>>()?,
            down.iter()
                .map(|name| {
                    check_shape(spec, "down", entry_for(spec, manifest, "down", name)?, true)
                })
                .collect::<Result<_, _>>()?,
            false,
        ),
    };
    if gate_shapes.is_empty()
        || down_shapes.is_empty()
        || (!up_shapes.is_empty() && up_shapes[0] != gate_shapes[0])
    {
        return Err(ExpertPlanError::new(format!(
            "expert group '{}' layer {:?} projection source count/shape mismatch",
            spec.group, spec.layer
        )));
    }
    if gate_shapes.iter().any(|shape| shape != &gate_shapes[0])
        || up_shapes.iter().any(|shape| shape != &up_shapes[0])
        || down_shapes.iter().any(|shape| shape != &down_shapes[0])
    {
        return Err(ExpertPlanError::new(format!(
            "expert group '{}' layer {:?} per-expert projection shape mismatch",
            spec.group, spec.layer
        )));
    }
    let gate = &gate_shapes[0];
    let down = &down_shapes[0];
    if gate.len() != 2 || down.len() != 2 {
        return Err(ExpertPlanError::new(format!(
            "expert group '{}' layer {:?} projection shapes are incompatible",
            spec.group, spec.layer
        )));
    }
    let expert_m = if fused {
        if gate[0] % 2 != 0 {
            return Err(ExpertPlanError::new(format!(
                "expert group '{}' layer {:?} fused gate/up width is not even",
                spec.group, spec.layer
            )));
        }
        gate[0] / 2
    } else {
        gate[0]
    };
    let shape = ExpertShape {
        expert_m,
        expert_k: gate[1],
        fused_gate_up: fused,
    };
    if down != &[shape.expert_k, shape.expert_m] {
        return Err(ExpertPlanError::new(format!(
            "expert group '{}' layer {:?} projection shape mismatch: gate_up={gate:?}, down={down:?}",
            spec.group, spec.layer
        )));
    }
    let source_names = source_names(&spec.source_layout);
    let source_dtype = source_names
        .iter()
        .flat_map(|(_, names)| names.iter())
        .find_map(|name| {
            manifest
                .iter()
                .find(|entry| entry.name == *name && entry.layer == spec.layer)
                .map(|entry| entry.dtype)
        })
        .ok_or_else(|| ExpertPlanError::new("expert projection has no source dtype"))?;
    for (_, names) in source_names {
        for name in names {
            let entry = entry_for(spec, manifest, "projection", name)?;
            if entry.dtype != source_dtype && label_is_projection(name, &spec.source_layout) {
                return Err(ExpertPlanError::new(format!(
                    "expert group '{}' projection source '{}' dtype {:?} differs from {:?}",
                    spec.group, name, entry.dtype, source_dtype
                )));
            }
        }
    }
    Ok((shape, source_dtype))
}

fn label_is_projection(name: &str, layout: &ExpertSourceLayout) -> bool {
    match layout {
        ExpertSourceLayout::PackedFused { gate_up, down, .. } => name == gate_up || name == down,
        ExpertSourceLayout::PackedSeparate { gate, up, down, .. } => {
            name == gate || name == up || name == down
        }
        ExpertSourceLayout::PerExpertFused { gate_up, down, .. } => {
            gate_up.iter().any(|candidate| candidate == name)
                || down.iter().any(|candidate| candidate == name)
        }
        ExpertSourceLayout::PerExpertSeparate { gate, up, down, .. } => {
            gate.iter().any(|candidate| candidate == name)
                || up.iter().any(|candidate| candidate == name)
                || down.iter().any(|candidate| candidate == name)
        }
    }
}

fn down_names(layout: &ExpertSourceLayout) -> Vec<&str> {
    match layout {
        ExpertSourceLayout::PackedFused { down, .. }
        | ExpertSourceLayout::PackedSeparate { down, .. } => vec![down.as_str()],
        ExpertSourceLayout::PerExpertFused { down, .. }
        | ExpertSourceLayout::PerExpertSeparate { down, .. } => {
            down.iter().map(String::as_str).collect()
        }
    }
}

fn resolve_collective(
    spec: &ExpertGroupSpec,
    manifest: &[WeightEntry],
) -> Result<(Option<CollectiveHint>, Option<String>), ExpertPlanError> {
    let expected = match spec.parallelism {
        ExpertParallelism::Single => None,
        ExpertParallelism::TensorParallel => Some(DimKind::Tp),
        ExpertParallelism::ExpertParallel => Some(DimKind::Ep),
    };
    let Some(expected_kind) = expected else {
        return Ok((None, None));
    };
    let rows = collective_schedule(manifest);
    let mut selected = None;
    for name in down_names(&spec.source_layout) {
        let row = rows
            .iter()
            .find(|row| row.layer == spec.layer.unwrap_or(usize::MAX) && row.name == name)
            .ok_or_else(|| {
                ExpertPlanError::new(format!(
                    "expert group '{}' has no ordered G3 collective row for down source '{name}'",
                    spec.group
                ))
            })?;
        if !matches!(
            row.hint,
            CollectiveHint::AllReduce { kind } if kind == expected_kind
        ) {
            return Err(ExpertPlanError::new(format!(
                "expert group '{}' down source '{name}' collective {:?} does not match {:?}",
                spec.group, row.hint, expected_kind
            )));
        }
        if let Some(previous) = selected {
            if previous != row.hint {
                return Err(ExpertPlanError::new(format!(
                    "expert group '{}' down sources disagree on collective axis",
                    spec.group
                )));
            }
        } else {
            selected = Some(row.hint);
        }
    }
    Ok((
        selected,
        down_names(&spec.source_layout)
            .first()
            .map(|name| (*name).to_string()),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use hipfire_dispatch::families::moe::{
        MoeActivationVariant, MoeFamily, MoeProj, MoeProtocolKind, RouterPlan,
    };
    use hipfire_dispatch::pipeline::{GemvInput, Step, StepCollective};

    fn mesh_ep() -> DeviceMesh {
        DeviceMesh::rect(&[(DimKind::Ep, 2)]).expect("test mesh")
    }

    fn manifest() -> Vec<WeightEntry> {
        vec![
            WeightEntry::layer("router", 0, vec![4, 4], DType::F32, ShardPolicy::Replicate),
            WeightEntry::layer(
                "experts.gate_up",
                0,
                vec![4, 128, 64],
                DType::MQ4G256,
                ShardPolicy::ExpertSharded {
                    n_experts: 4,
                    assign: ExpertAssign::Stride,
                },
            ),
            WeightEntry::layer(
                "experts.down",
                0,
                vec![4, 64, 64],
                DType::MQ4G256,
                ShardPolicy::ExpertSharded {
                    n_experts: 4,
                    assign: ExpertAssign::Stride,
                },
            ),
        ]
    }

    fn manifest_for(parallelism: ExpertParallelism) -> Vec<WeightEntry> {
        let mut entries = manifest();
        if matches!(parallelism, ExpertParallelism::Single) {
            entries[1].policy = ShardPolicy::Replicate;
            entries[2].policy = ShardPolicy::Replicate;
        }
        entries
    }

    fn grouped_manifest_for(parallelism: ExpertParallelism) -> Vec<WeightEntry> {
        let mut entries = manifest_for(parallelism);
        entries[0].logical_shape = vec![4, 8];
        entries[1].logical_shape[0] = 8;
        entries[2].logical_shape[0] = 8;
        if matches!(parallelism, ExpertParallelism::ExpertParallel) {
            for entry in &mut entries[1..=2] {
                entry.policy = ShardPolicy::ExpertSharded {
                    n_experts: 8,
                    assign: ExpertAssign::Stride,
                };
            }
        }
        entries
    }


    fn separate_manifest() -> Vec<WeightEntry> {
        let mut entries = manifest();
        entries[1].name = "experts.gate".into();
        entries[1].logical_shape = vec![4, 64, 64];
        entries[1].policy = ShardPolicy::Replicate;
        entries[2].name = "experts.down".into();
        entries[2].policy = ShardPolicy::Replicate;
        entries.push(WeightEntry::layer(
            "experts.up",
            0,
            vec![4, 64, 64],
            DType::MQ4G256,
            ShardPolicy::Replicate,
        ));
        entries
    }

    fn separate_spec(execution: &str, parallelism: ExpertParallelism) -> ExpertGroupSpec {
        let mut value = spec(execution, parallelism);
        value.source_layout = ExpertSourceLayout::PackedSeparate {
            gate: "experts.gate".into(),
            up: "experts.up".into(),
            down: "experts.down".into(),
            sidecars: vec![],
        };
        value
    }

    fn spec(execution: &str, parallelism: ExpertParallelism) -> ExpertGroupSpec {
        ExpertGroupSpec {
            group: "block-0".into(),
            layer: Some(0),
            n_experts: 4,
            parallelism,
            assignment: ExpertAssign::Stride,
            source_layout: ExpertSourceLayout::PackedFused {
                gate_up: "experts.gate_up".into(),
                down: "experts.down".into(),
                sidecars: vec![],
            },
            resources: ExpertResourceRequirements {
                bytes_per_expert: 4096,
                alignment: 256,
            },
            router: "router".into(),
            execution: execution.into(),
        }
    }

    fn grouped_spec(execution: &str, parallelism: ExpertParallelism) -> ExpertGroupSpec {
        let mut value = spec(execution, parallelism);
        value.n_experts = 8;
        value
    }


    fn tensor_with_bytes(shape: Vec<usize>, dtype: DType, bytes: usize) -> GpuTensor {
        let mut tensor = GpuTensor::null_for_test();
        tensor.buf = unsafe { hip_bridge::DeviceBuffer::from_raw(std::ptr::null_mut(), bytes) };
        tensor.shape = shape;
        tensor.dtype = dtype;
        tensor
    }

    fn tensor(shape: Vec<usize>) -> GpuTensor {
        let elements = shape.iter().product::<usize>();
        tensor_with_bytes(
            shape,
            DType::F32,
            elements
                .checked_mul(DType::F32.size())
                .expect("test tensor bytes"),
        )
    }

    fn table(shape: Vec<usize>) -> GpuTensor {
        tensor(shape)
    }

    fn raw_i32(elements: usize) -> GpuTensor {
        let bytes = elements
            .checked_mul(std::mem::size_of::<i32>())
            .expect("test raw bytes");
        tensor_with_bytes(vec![bytes], DType::Raw, bytes)
    }

    fn resident_plan(
        execution: &str,
        parallelism: ExpertParallelism,
        mesh: &DeviceMesh,
    ) -> ExpertPlan {
        let (group_spec, group_manifest) = if execution == "grouped_quantized" {
            (
                grouped_spec(execution, parallelism),
                grouped_manifest_for(parallelism),
            )
        } else {
            (spec(execution, parallelism), manifest_for(parallelism))
        };
        let mut plan =
            ExpertPlan::from_manifest(&group_spec, &group_manifest, mesh).expect("test expert plan");
        let group_size = plan.group_size();
        let pointer_slots = plan.n_experts().checked_mul(2).expect("test pointer slots");
        for rank in 0..group_size {
            plan.commit_rank_tables(
                rank,
                table(vec![pointer_slots]),
                table(vec![pointer_slots]),
                None,
            )
            .expect("commit rank tables");
        }
        let placements = plan.placements().to_vec();
        let mut load = plan.begin_load();
        for placement in placements {
            load.reserve(placement).expect("reserve placement");
        }
        load.commit();
        plan
    }
    struct GroupedTensors {
        scores: GpuTensor,
        indices: GpuTensor,
        weights: GpuTensor,
        counts: GpuTensor,
        offsets: GpuTensor,
        sorted: GpuTensor,
        tiles: GpuTensor,
        inverse: GpuTensor,
        x: GpuTensor,
        grouped_gate: GpuTensor,
        gate_batch: GpuTensor,
        up_batch: GpuTensor,
        rot_batch: GpuTensor,
        grouped_down: GpuTensor,
        down_x: GpuTensor,
        out: GpuTensor,
    }

    fn grouped_tensors() -> GroupedTensors {
        GroupedTensors {
            scores: tensor(vec![2, 8]),
            indices: tensor(vec![2, 8]),
            weights: tensor(vec![2, 8]),
            counts: raw_i32(8),
            offsets: raw_i32(9),
            sorted: raw_i32(16),
            tiles: raw_i32(4),
            inverse: raw_i32(16),
            x: tensor(vec![2, 64]),
            grouped_gate: tensor(vec![16, 128]),
            gate_batch: tensor(vec![16, 64]),
            up_batch: tensor(vec![16, 64]),
            rot_batch: tensor(vec![16, 64]),
            grouped_down: tensor(vec![16, 64]),
            down_x: tensor(vec![16, 64]),
            out: tensor(vec![2, 64]),
        }
    }

    fn grouped_steps<'a>(
        experts: &'a MoeExpertRef<'a>,
        tensors: &'a GroupedTensors,
    ) -> Vec<Step<'a>> {
        vec![
            Step::MoeRoute {
                plan: RouterPlan::SoftmaxTopK {
                    scores: &tensors.scores,
                    topk_indices: &tensors.indices,
                    topk_weights: &tensors.weights,
                    k_top: 8,
                    normalize: true,
                },
            },
            Step::MoeScatter {
                topk_indices: &tensors.indices,
                expert_token_counts: &tensors.counts,
                expert_offsets: &tensors.offsets,
                sorted_slot_index: &tensors.sorted,
                expert_tile_ids: &tensors.tiles,
                inverse_perm: &tensors.inverse,
                total_slots: 16,
                n_experts: 8,
                m_total_max: 16,
                block_m: 4,
            },
            Step::GroupedMoeGemm {
                experts,
                which: MoeProj::GateUp {
                    up_out: &tensors.up_batch,
                },
                sorted_slot_index: &tensors.sorted,
                expert_tile_ids: &tensors.tiles,
                x: &tensors.x,
                y: &tensors.grouped_gate,
                m_total: 16,
                batch_size: 2,
                k_top: 8,
            },
            Step::MoeGateUpUnscatter {
                y_grouped: &tensors.grouped_gate,
                sorted_slot_index: &tensors.sorted,
                gate_batch: &tensors.gate_batch,
                up_batch: &tensors.up_batch,
                inter: 64,
                k_top: 8,
                m_total: 16,
            },
            Step::MoeActivation {
                variant: MoeActivationVariant::SiluMul,
                gate: &tensors.gate_batch,
                up: &tensors.up_batch,
                rot_out: &tensors.rot_batch,
                inter: 64,
                rows: 16,
            },
            Step::GroupedMoeGemm {
                experts,
                which: MoeProj::DownExpanded,
                sorted_slot_index: &tensors.sorted,
                expert_tile_ids: &tensors.tiles,
                x: &tensors.rot_batch,
                y: &tensors.grouped_down,
                m_total: 16,
                batch_size: 2,
                k_top: 8,
            },
            Step::MoeCombine {
                down_out: &tensors.grouped_down,
                topk_weights: &tensors.weights,
                out: &tensors.out,
                hidden: 64,
                k_top: 8,
                batch_size: 2,
                inverse_perm: Some(&tensors.inverse),
            },
        ]
    }

    fn indexed_steps<'a>(
        experts: &'a MoeExpertRef<'a>,
        indices: &'a GpuTensor,
        weights: &'a GpuTensor,
        x: &'a GpuTensor,
        gate: &'a GpuTensor,
        up: &'a GpuTensor,
        rot: &'a GpuTensor,
        down: &'a GpuTensor,
        out: &'a GpuTensor,
    ) -> Vec<Step<'a>> {
        vec![
            Step::MoeRoute {
                plan: RouterPlan::Precomputed {
                    topk_indices: indices,
                    topk_weights: weights,
                    k_top: 2,
                },
            },
            Step::IndexedMoeGemv {
                experts,
                which: MoeProj::GateUp { up_out: up },
                topk_indices: indices,
                input: GemvInput::Prerotated(x),
                out: gate,
                k_top: 2,
                batch_size: 1,
            },
            Step::MoeActivation {
                variant: MoeActivationVariant::SiluMul,
                gate,
                up,
                rot_out: rot,
                inter: 64,
                rows: 2,
            },
            Step::IndexedMoeGemv {
                experts,
                which: MoeProj::DownExpanded,
                topk_indices: indices,
                input: GemvInput::Prerotated(rot),
                out: down,
                k_top: 2,
                batch_size: 1,
            },
            Step::MoeCombine {
                down_out: down,
                topk_weights: weights,
                out,
                hidden: 64,
                k_top: 2,
                batch_size: 1,
                inverse_perm: None,
            },
        ]
    }

    #[test]
    fn grouped_chain_seals_with_distinct_route_and_sorted_slot_buffers() {
        let mesh = DeviceMesh::single().unwrap();
        let plan = resident_plan("grouped_quantized", ExpertParallelism::Single, &mesh);
        let experts = plan.bind_expert_ref(0).unwrap();
        let tensors = grouped_tensors();
        let steps = grouped_steps(&experts, &tensors);
        assert!(!std::ptr::eq(&tensors.indices, &tensors.sorted));

        let schedule = MoeFamily::new()
            .seal_steps(
                ExpertExecutionPlan::GroupedQuantized,
                steps,
                vec![StepCollective::None; 7],
            )
            .expect("grouped chain should seal");
        assert_eq!(schedule.execution(), ExpertExecutionPlan::GroupedQuantized);
        assert_eq!(schedule.steps().len(), 7);
        let signature = schedule.execution_signature().unwrap();
        assert_eq!(signature.protocol, MoeProtocolKind::Grouped);
        match (&schedule.steps()[0], &schedule.steps()[1]) {
            (
                Step::MoeRoute { plan },
                Step::MoeScatter {
                    topk_indices,
                    sorted_slot_index,
                    ..
                },
            ) => {
                assert!(std::ptr::eq(plan.route_buffers().0, *topk_indices));
                assert!(!std::ptr::eq(plan.route_buffers().0, *sorted_slot_index));
            }
            _ => panic!("sealed schedule changed grouped route/scatter order"),
        }
    }

    #[test]
    fn grouped_parallel_collective_covers_every_batched_output_element() {
        let mesh = mesh_ep();
        let plan = resident_plan(
            "grouped_quantized",
            ExpertParallelism::ExpertParallel,
            &mesh,
        );
        let experts = plan.bind_expert_ref(0).unwrap();
        let tensors = grouped_tensors();
        let family = MoeFamily::new();

        let mut short = vec![StepCollective::None; 7];
        short[6] = StepCollective::all_reduce(DimKind::Ep, 64, vec![0, 1], mesh.epoch(), 0);
        let error = match family.seal_steps(
            ExpertExecutionPlan::GroupedQuantized,
            grouped_steps(&experts, &tensors),
            short,
        ) {
            Ok(_) => panic!("batched grouped output must not reduce only one row"),
            Err(error) => error,
        };
        assert!(error.to_string().contains("output dimension"));

        let mut full = vec![StepCollective::None; 7];
        full[6] = StepCollective::all_reduce(DimKind::Ep, 128, vec![0, 1], mesh.epoch(), 0);
        let schedule = family
            .seal_steps(
                ExpertExecutionPlan::GroupedQuantized,
                grouped_steps(&experts, &tensors),
                full,
            )
            .expect("collective must cover batch_size * hidden elements");
        assert!(matches!(
            &schedule.collectives()[6],
            StepCollective::AllReduce { dim: 128, .. }
        ));
    }


    #[test]
    fn sealing_rejects_non_k8_softmax_routes() {
        let mesh = DeviceMesh::single().unwrap();
        let plan = resident_plan("grouped_quantized", ExpertParallelism::Single, &mesh);
        let experts = plan.bind_expert_ref(0).unwrap();
        let tensors = grouped_tensors();
        let mut steps = grouped_steps(&experts, &tensors);
        steps[0] = Step::MoeRoute {
            plan: RouterPlan::SoftmaxTopK {
                scores: &tensors.scores,
                topk_indices: &tensors.indices,
                topk_weights: &tensors.weights,
                k_top: 2,
                normalize: true,
            },
        };
        let error = match MoeFamily::new().seal_steps(
            ExpertExecutionPlan::GroupedQuantized,
            steps,
            vec![StepCollective::None; 7],
        ) {
            Ok(_) => panic!("generic softmax routing is only executable at k_top=8"),
            Err(error) => error,
        };
        assert!(error.to_string().contains("k_top=8"));
    }
    #[test]
    fn stride_assignment_and_named_group_are_deterministic() {
        let plan = ExpertPlan::from_manifest(
            &spec("indexed_quantized", ExpertParallelism::ExpertParallel),
            &manifest(),
            &mesh_ep(),
        )
        .unwrap();
        let owners: Vec<_> = plan
            .placements()
            .iter()
            .map(|placement| (placement.global_id, placement.owner, placement.local_slot))
            .collect();
        assert_eq!(owners, vec![(0, 0, 0), (1, 1, 0), (2, 0, 1), (3, 1, 1)]);
        assert_eq!(plan.group_devices(), &[0, 1]);
        assert_eq!(
            plan.collective(),
            Some(CollectiveHint::AllReduce { kind: DimKind::Ep })
        );
    }

    #[test]
    fn bound_view_uses_canonical_rank_owner() {
        let mut plan = ExpertPlan::from_manifest(
            &spec("indexed_quantized", ExpertParallelism::ExpertParallel),
            &manifest(),
            &mesh_ep(),
        )
        .unwrap();
        let gate = table(vec![8]);
        let down = table(vec![8]);
        plan.commit_rank_tables(1, gate, down, None).unwrap();
        {
            let placements: Vec<_> = plan
                .placements()
                .iter()
                .copied()
                .filter(|placement| placement.owner == 1)
                .collect();
            let mut load = plan.begin_load();
            for placement in placements {
                load.reserve(placement).unwrap();
            }
            load.commit();
        }
        let view = plan.bind_expert_ref(1).unwrap();
        assert_eq!(view.owned(), &[1, 3]);
        assert_eq!(view.n_experts(), 4);
    }

    #[test]
    fn rollback_then_reserve_reuse_and_repeated_teardown_are_safe() {
        let mut plan = ExpertPlan::from_manifest(
            &spec("indexed_quantized", ExpertParallelism::ExpertParallel),
            &manifest(),
            &mesh_ep(),
        )
        .unwrap();
        let first = plan.placements()[0];
        {
            let mut load = plan.begin_load();
            load.reserve(first).unwrap();
            load.rollback();
            assert_eq!(
                load.reserve(first)
                    .expect_err("a rolled-back transaction is closed")
                    .to_string(),
                "expert load transaction is already finished"
            );
        }
        assert_eq!(plan.resident_slots(), 0);
        {
            let mut load = plan.begin_load();
            load.reserve(first)
                .expect("rollback must release placement");
            load.commit();
        }
        assert_eq!(plan.resident_slots(), 1);
        plan.unload();
        plan.unload();
        assert_eq!(plan.resident_slots(), 0);
    }

    #[test]
    fn nonresident_and_mismatched_rank_owner_fail_closed() {
        let mesh = mesh_ep();
        let mut missing = ExpertPlan::from_manifest(
            &spec("grouped_quantized", ExpertParallelism::ExpertParallel),
            &manifest(),
            &mesh,
        )
        .unwrap();
        missing
            .commit_rank_tables(0, table(vec![8]), table(vec![8]), None)
            .unwrap();
        let error = missing
            .bind_expert_ref(0)
            .err()
            .expect("nonresident owner must not bind");
        assert!(error.to_string().contains("nonresident"));

        let plan = resident_plan(
            "grouped_quantized",
            ExpertParallelism::ExpertParallel,
            &mesh,
        );
        let experts = plan.bind_expert_ref(0).unwrap();
        assert_eq!(experts.owner_rank(), 0);
        let tensors = grouped_tensors();
        let steps = grouped_steps(&experts, &tensors);
        let mut collectives = vec![StepCollective::None; 7];
        collectives[6] =
            StepCollective::all_reduce(DimKind::Ep, 128, vec![0, 1], mesh.epoch(), 1);
        let schedule = MoeFamily::new()
            .seal_steps(ExpertExecutionPlan::GroupedQuantized, steps, collectives)
            .expect("collective descriptor is locally typed");
        let error = hipfire_dispatch::pipeline::steps::validate_sealed_steps_mesh_preflight(
            2,
            &mesh,
            &[&schedule],
        )
        .expect_err("rank-1 collective must not launch rank-0 owner view");
        assert!(error.to_string().contains("collective rank"));
    }

    #[test]
    fn sealed_route_rejects_physical_buffer_shorter_than_logical_shape() {
        let mesh = DeviceMesh::single().unwrap();
        let plan = resident_plan("grouped_quantized", ExpertParallelism::Single, &mesh);
        let experts = plan.bind_expert_ref(0).unwrap();
        let mut tensors = grouped_tensors();
        tensors.indices =
            tensor_with_bytes(vec![2, 8], DType::F32, 15 * DType::F32.size());
        let steps = grouped_steps(&experts, &tensors);
        let error = match MoeFamily::new().seal_steps(
            ExpertExecutionPlan::GroupedQuantized,
            steps,
            vec![StepCollective::None; 7],
        ) {
            Ok(_) => panic!("logical shape must not stand in for physical capacity"),
            Err(error) => error,
        };
        assert!(error
            .to_string()
            .contains("insufficient logical/physical capacity"));
    }

    #[test]
    fn mesh_preflight_rejects_cross_rank_protocol_disagreement() {
        let mesh = mesh_ep();
        let indexed_plan = resident_plan(
            "indexed_quantized",
            ExpertParallelism::ExpertParallel,
            &mesh,
        );
        let grouped_plan = resident_plan(
            "grouped_quantized",
            ExpertParallelism::ExpertParallel,
            &mesh,
        );
        let indexed_experts = indexed_plan.bind_expert_ref(0).unwrap();
        let grouped_experts = grouped_plan.bind_expert_ref(1).unwrap();

        let indexed_indices = tensor(vec![2]);
        let indexed_weights = tensor(vec![2]);
        let indexed_x = tensor(vec![64]);
        let indexed_gate = tensor(vec![2, 64]);
        let indexed_up = tensor(vec![2, 64]);
        let indexed_rot = tensor(vec![2, 64]);
        let indexed_down = tensor(vec![2, 64]);
        let indexed_out = tensor(vec![64]);
        let indexed = MoeFamily::new()
            .seal_steps(
                ExpertExecutionPlan::IndexedQuantized,
                indexed_steps(
                    &indexed_experts,
                    &indexed_indices,
                    &indexed_weights,
                    &indexed_x,
                    &indexed_gate,
                    &indexed_up,
                    &indexed_rot,
                    &indexed_down,
                    &indexed_out,
                ),
                {
                    let mut collectives = vec![StepCollective::None; 5];
                    collectives[4] =
                        StepCollective::all_reduce(DimKind::Ep, 64, vec![0, 1], mesh.epoch(), 0);
                    collectives
                },
            )
            .expect("indexed rank schedule should seal");

        let grouped_tensors = grouped_tensors();
        let grouped = MoeFamily::new()
            .seal_steps(
                ExpertExecutionPlan::GroupedQuantized,
                grouped_steps(&grouped_experts, &grouped_tensors),
                {
                    let mut collectives = vec![StepCollective::None; 7];
                    collectives[6] =
                        StepCollective::all_reduce(DimKind::Ep, 128, vec![0, 1], mesh.epoch(), 1);
                    collectives
                },
            )
            .expect("grouped rank schedule should seal");

        let error = hipfire_dispatch::pipeline::steps::validate_sealed_steps_mesh_preflight(
            2,
            &mesh,
            &[&indexed, &grouped],
        )
        .expect_err("mixed indexed/grouped ranks must not launch");
        assert!(error.to_string().contains("executable identity"));
    }

    #[test]
    fn parallel_sealing_rejects_duplicate_or_mixed_collectives() {
        let mesh = mesh_ep();
        let plan = resident_plan(
            "grouped_quantized",
            ExpertParallelism::ExpertParallel,
            &mesh,
        );
        let experts = plan.bind_expert_ref(0).unwrap();
        let tensors = grouped_tensors();
        let family = MoeFamily::new();

        let mut duplicate = vec![StepCollective::None; 7];
        duplicate[5] =
            StepCollective::all_reduce(DimKind::Ep, 128, vec![0, 1], mesh.epoch(), 0);
        duplicate[6] =
            StepCollective::all_reduce(DimKind::Ep, 128, vec![0, 1], mesh.epoch(), 0);
        let error = match family.seal_steps(
            ExpertExecutionPlan::GroupedQuantized,
            grouped_steps(&experts, &tensors),
            duplicate,
        ) {
            Ok(_) => panic!("a routed reduction cannot appear twice"),
            Err(error) => error,
        };

        assert!(error.to_string().contains("attached to combine"));

        let error = match family.seal_steps(
            ExpertExecutionPlan::GroupedQuantized,
            grouped_steps(&experts, &tensors),
            vec![StepCollective::None; 7],
        ) {
            Ok(_) => panic!("parallel owner must not silently use identity reduction"),
            Err(error) => error,
        };
        assert!(error.to_string().contains("collective count"));
    }

    #[test]
    fn single_plan_accepts_nonzero_pipeline_stage_device() {
        let mesh = DeviceMesh::rect(&[(DimKind::Pp, 2)]).unwrap();
        let mut staged_manifest = manifest_for(ExpertParallelism::Single);
        for entry in &mut staged_manifest {
            entry.layer = Some(1);
        }
        let mut staged_spec = spec("indexed_quantized", ExpertParallelism::Single);
        staged_spec.layer = Some(1);
        let mut plan =
            ExpertPlan::from_manifest(&staged_spec, &staged_manifest, &mesh).unwrap();
        assert_eq!(plan.group_devices(), &[1]);

        for rank in 0..plan.group_size() {
            plan.commit_rank_tables(rank, table(vec![8]), table(vec![8]), None)
                .unwrap();
        }
        let placements = plan.placements().to_vec();
        let mut load = plan.begin_load();
        for placement in placements {
            load.reserve(placement).unwrap();
        }
        load.commit();

        let experts = plan.bind_expert_ref(0).unwrap();
        assert_eq!(experts.owner_rank(), 0);
        assert_eq!(experts.group_devices(), &[1]);
        assert_eq!(experts.owned(), &[0, 1, 2, 3]);
    }
    #[test]
    fn per_expert_fallback_is_refused_at_plan_boundary() {
        let error = ExpertPlan::from_manifest(
            &spec("per_expert_fallback", ExpertParallelism::ExpertParallel),
            &manifest(),
            &mesh_ep(),
        )
        .expect_err("fallback is not a typed Step protocol");
        assert!(error.to_string().contains("PerExpertFallback"));
    }

    #[test]
    fn source_shape_mismatch_is_refused_before_owner_creation() {
        let mut malformed = manifest();
        malformed[1].logical_shape = vec![4, 64, 64];
        let error = ExpertPlan::from_manifest(
            &spec("indexed_quantized", ExpertParallelism::ExpertParallel),
            &malformed,
            &mesh_ep(),
        )
        .expect_err("gate/up and down shapes must agree");
        assert!(error.to_string().contains("shape mismatch"));
    }
    #[test]
    fn separate_layout_fails_closed_at_plan_boundary() {
        let mesh = DeviceMesh::single().unwrap();
        let error = ExpertPlan::from_manifest(
            &separate_spec("grouped_quantized", ExpertParallelism::Single),
            &separate_manifest(),
            &mesh,
        )
        .expect_err("generic executor must refuse separate projection sources");
        assert!(error.to_string().contains("separate gate/up sources"));
    }

    #[test]
    fn single_plan_emits_identity_collective() {
        let mut single_manifest = manifest();
        single_manifest[1].policy = ShardPolicy::Replicate;
        single_manifest[2].policy = ShardPolicy::Replicate;
        let mesh = DeviceMesh::single().unwrap();
        let plan = ExpertPlan::from_manifest(
            &spec("indexed_quantized", ExpertParallelism::Single),
            &single_manifest,
            &mesh,
        )
        .unwrap();
        assert_eq!(plan.collective(), None);
        assert_eq!(plan.step_collective(0, 64).unwrap(), StepCollective::None);
    }
}
