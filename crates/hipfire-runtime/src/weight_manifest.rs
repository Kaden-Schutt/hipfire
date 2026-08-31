// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Pure logical model declarations and device-mesh planning.
//!
//! A manifest describes *what* an architecture needs. [`plan_manifest`] resolves
//! those declarations against one already-admitted rectangular
//! [`hipfire_hardware::DeviceMesh`] and describes *where* each declaration and
//! synchronization point belongs. This module deliberately has no GPU, file,
//! carrier, quantizer, or allocation dependency; fulfillment is separate.
//!
//! The manifest is the single source of truth for collectives. A row-sharded
//! projection contributes one ordered tensor collective over `Tp`, an
//! expert-sharded projection contributes one over `Ep`, and pipeline boundaries
//! come from the mesh. Executors consume this schedule rather than add
//! family-local reductions.

use crate::tp_shard::ExpertAssign;
use hipfire_hardware::{CollectiveHint, DeviceMesh, DimKind};
use rdna_compute::DType;
use std::collections::HashSet;

/// Derive the collective required by one weight policy.
///
/// The returned hint is per declared operation. Two different row-sharded
/// operations in one layer are two distinct schedule entries and both execute
/// once.
#[inline]
pub fn collective_for_policy(policy: &ShardPolicy) -> Option<CollectiveHint> {
    match policy {
        ShardPolicy::RowShard { .. } => Some(CollectiveHint::AllReduce { kind: DimKind::Tp }),
        ShardPolicy::ExpertSharded { .. } => {
            Some(CollectiveHint::AllReduce { kind: DimKind::Ep })
        }
        ShardPolicy::ExpertTensorSharded { inner, .. } => collective_for_policy(inner),
        _ => None,
    }
}

/// Non-layer placement targets resolved from mesh stage coordinates.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum PinTarget {
    /// Token embedding, pinned to pipeline stage zero.
    Embed,
    /// Final norm/language head, pinned to the final pipeline stage.
    Output,
}

/// Optional placement override. It is separate from [`ShardPolicy`] so a tied
/// logical identity can be materialized at an output stage without changing
/// the source declaration.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum PlacementHint {
    /// Resolve placement from the policy and layer scope.
    #[default]
    Policy,
    /// Resolve placement from a mesh-derived pin target.
    Pin(PinTarget),
}

/// Source dtype acceptance. The logical manifest dtype remains an architecture
/// expectation; fulfillment preserves the source dtype and never silently
/// converts representation.
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum SourceDType {
    Any,
    Exact(DType),
    OneOf(Vec<DType>),
}

#[derive(Clone, PartialEq, Eq, Debug)]
pub struct DTypeConstraint {
    pub source: SourceDType,
}

impl DTypeConstraint {
    pub fn any_source() -> Self {
        Self {
            source: SourceDType::Any,
        }
    }

    pub fn source_exact(dtype: DType) -> Self {
        Self {
            source: SourceDType::Exact(dtype),
        }
    }

    pub fn source_from_sources(sources: Vec<DType>) -> Self {
        Self {
            source: SourceDType::OneOf(sources),
        }
    }

    pub fn accepts(&self, dtype: DType) -> bool {
        match &self.source {
            SourceDType::Any => true,
            SourceDType::Exact(expected) => *expected == dtype,
            SourceDType::OneOf(allowed) => allowed.contains(&dtype),
        }
    }
}

/// The block ordering of a fused projection.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum FusedQkvLayout {
    /// `[Q | K | V]`.
    Qkv,
    /// `[Q | gate]`.
    QGate,
    /// `[Q | K | V | Z]`.
    QkvZ,
}

/// How one logical tensor is projected onto mesh devices.
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum ShardPolicy {
    /// A complete tensor on every device in the owning compute grid.
    Replicate,
    /// Split the output dimension `axis` across `Tp`.
    ColumnShard { axis: usize },
    /// Split the input dimension `axis` across `Tp`; the consumer reduces.
    RowShard { axis: usize },
    /// Assign complete expert tensors across `Ep` ranks.
    ExpertSharded {
        n_experts: usize,
        assign: ExpertAssign,
    },
    /// Fused QKV projection with head-aware block boundaries.
    FusedQkv {
        q_heads: usize,
        kv_heads: usize,
        head_dim: usize,
        layout: FusedQkvLayout,
    },
    /// Per-head projection (DeltaNet state/projections).
    HeadSharded { n_heads: usize, head_dim: usize },
    /// Alias another logical source in the same manifest scope.
    Tied { source: String },
    /// Pin to a mesh-derived non-layer stage.
    Pin(PinTarget),
    /// Split vocabulary rows across `Tp`.
    VocabShard { axis: usize },
    /// Split each expert tensor across `Tp`. The inner policy is normally
    /// `ColumnShard { axis: 1 }` for gate/up or `RowShard { axis: 2 }` for down.
    ExpertTensorSharded {
        n_experts: usize,
        inner: Box<ShardPolicy>,
    },
}

/// A logical weight declaration. No source filename or GPU handle belongs
/// here; architecture carriers resolve those at fulfillment time.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct WeightEntry {
    pub name: String,
    pub layer: Option<usize>,
    pub logical_shape: Vec<usize>,
    pub dtype: DType,
    pub dtype_constraint: DTypeConstraint,
    pub placement: PlacementHint,
    pub policy: ShardPolicy,
}

impl WeightEntry {
    pub fn model(
        name: impl Into<String>,
        logical_shape: Vec<usize>,
        dtype: DType,
        policy: ShardPolicy,
    ) -> Self {
        Self::model_with_dtype_constraint(
            name,
            logical_shape,
            dtype,
            DTypeConstraint::any_source(),
            policy,
        )
    }

    pub fn model_with_dtype_constraint(
        name: impl Into<String>,
        logical_shape: Vec<usize>,
        dtype: DType,
        dtype_constraint: DTypeConstraint,
        policy: ShardPolicy,
    ) -> Self {
        Self {
            name: name.into(),
            layer: None,
            logical_shape,
            dtype,
            dtype_constraint,
            placement: PlacementHint::Policy,
            policy,
        }
    }

    pub fn layer(
        name: impl Into<String>,
        layer: usize,
        logical_shape: Vec<usize>,
        dtype: DType,
        policy: ShardPolicy,
    ) -> Self {
        Self::layer_with_dtype_constraint(
            name,
            layer,
            logical_shape,
            dtype,
            DTypeConstraint::any_source(),
            policy,
        )
    }

    pub fn layer_with_dtype_constraint(
        name: impl Into<String>,
        layer: usize,
        logical_shape: Vec<usize>,
        dtype: DType,
        dtype_constraint: DTypeConstraint,
        policy: ShardPolicy,
    ) -> Self {
        Self {
            name: name.into(),
            layer: Some(layer),
            logical_shape,
            dtype,
            dtype_constraint,
            placement: PlacementHint::Policy,
            policy,
        }
    }

    pub fn with_placement(mut self, placement: PlacementHint) -> Self {
        self.placement = placement;
        self
    }

    /// Stable identity used by source resolvers and store keys.
    pub fn identity(&self) -> (&str, Option<usize>) {
        (&self.name, self.layer)
    }
}

/// Per-layer state declaration. Actual cache representation remains in the
/// architecture/model owner; this records logical placement scope only.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub enum StateKind {
    Kv { quant: String },
    Recurrent,
    Conv,
}

#[derive(Clone, PartialEq, Eq, Debug)]
pub struct StateEntry {
    pub kind: StateKind,
    pub layer: usize,
}

impl StateEntry {
    pub fn new(kind: StateKind, layer: usize) -> Self {
        Self { kind, layer }
    }
}

/// One fully resolved weight placement.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct WeightPlacement {
    pub name: String,
    pub layer: Option<usize>,
    pub devices: Vec<usize>,
}

/// One ordered collective implied by one manifest operation.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct CollectiveScheduleEntry {
    pub name: String,
    pub layer: usize,
    pub hint: CollectiveHint,
}

/// Complete pure compilation of declarations against a mesh.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct ManifestPlan {
    pub weights: Vec<WeightPlacement>,
    /// State and the global devices on which that state is resident.
    pub state: Vec<(StateEntry, Vec<usize>)>,
    /// Ordered `(layer, hint)` schedule retained for executor integration.
    pub layer_collectives: Vec<(usize, CollectiveHint)>,
    /// Named schedule entries, allowing an executor to prove no operation was
    /// silently omitted or scheduled twice.
    pub collective_schedule: Vec<CollectiveScheduleEntry>,
    /// PP boundary hints in ascending after-layer order.
    pub band_xfers: Vec<(usize, CollectiveHint)>,
}

fn base_coord_for(entry: &WeightEntry, mesh: &DeviceMesh, n_layers: usize) -> Vec<usize> {
    let stage = match (entry.placement, &entry.policy, entry.layer) {
        (PlacementHint::Pin(PinTarget::Embed), _, _)
        | (PlacementHint::Policy, ShardPolicy::Pin(PinTarget::Embed), _) => 0,
        (PlacementHint::Pin(PinTarget::Output), _, _)
        | (PlacementHint::Policy, ShardPolicy::Pin(PinTarget::Output), _) => {
            mesh.size_of(DimKind::Pp).saturating_sub(1)
        }
        (PlacementHint::Policy, _, Some(layer)) => mesh.stage_for_layer(layer, n_layers),
        (PlacementHint::Policy, _, None) => 0,
    };
    let mut coord = mesh.coord_of(0);
    if let Some(index) = mesh.axes().iter().position(|axis| axis.kind == DimKind::Pp) {
        coord[index] = stage;
    }
    coord
}

/// Compute global placement without touching a source, GPU, or allocator.
pub fn placement_devices(entry: &WeightEntry, mesh: &DeviceMesh, n_layers: usize) -> Vec<usize> {
    let coord = base_coord_for(entry, mesh, n_layers);
    match &entry.policy {
        ShardPolicy::Pin(_) | ShardPolicy::Tied { .. } => vec![mesh.device_of(&coord)],
        ShardPolicy::ExpertSharded { .. } => mesh.group_along(DimKind::Ep, &coord),
        ShardPolicy::ExpertTensorSharded { .. } => mesh.group_along(DimKind::Tp, &coord),
        _ => mesh.stage_devices(&coord),
    }
}

/// Ordered per-operation collective schedule. This deliberately does not
/// deduplicate by `(layer, kind)`: two distinct row-sharded projections in one
/// layer represent two distinct output points and each must reduce once.
pub fn collective_schedule(manifest: &[WeightEntry]) -> Vec<CollectiveScheduleEntry> {
    manifest
        .iter()
        .filter_map(|entry| {
            let layer = entry.layer?;
            let hint = collective_for_policy(&entry.policy)?;
            Some(CollectiveScheduleEntry {
                name: entry.name.clone(),
                layer,
                hint,
            })
        })
        .collect()
}

/// Compact schedule view consumed by executor adapters.
pub fn layer_collectives(manifest: &[WeightEntry]) -> Vec<(usize, CollectiveHint)> {
    collective_schedule(manifest)
        .into_iter()
        .map(|entry| (entry.layer, entry.hint))
        .collect()
}

fn validate_shape(entry: &WeightEntry) -> Result<(), String> {
    if entry.name.is_empty() {
        return Err("manifest entry has an empty name".to_string());
    }
    if entry.logical_shape.is_empty() || entry.logical_shape.iter().any(|&d| d == 0) {
        return Err(format!(
            "{}[layer {:?}]: logical_shape {:?} must be non-empty",
            entry.name, entry.layer, entry.logical_shape
        ));
    }
    Ok(())
}

pub(crate) fn validate_weight_layers(
    manifest: &[WeightEntry],
    n_layers: usize,
) -> Result<(), String> {
    for entry in manifest {
        if let Some(layer) = entry.layer {
            if layer >= n_layers {
                return Err(format!(
                    "{} layer {} outside n_layers={n_layers}",
                    entry.name, layer
                ));
            }
        }
    }
    Ok(())
}

/// Validate logical shard math and tied source identity before fulfillment.
pub fn validate_manifest(manifest: &[WeightEntry], mesh: &DeviceMesh) -> Result<(), String> {
    let mut identities = HashSet::new();
    for entry in manifest {
        validate_shape(entry)?;
        if !identities.insert(entry.identity()) {
            return Err(format!(
                "duplicate manifest identity ('{}', {:?})",
                entry.name, entry.layer
            ));
        }
    }

    let tp = mesh.size_of(DimKind::Tp);
    for entry in manifest {
        let context = format!("{}[layer {:?}]", entry.name, entry.layer);
        match &entry.policy {
            ShardPolicy::ColumnShard { axis }
            | ShardPolicy::RowShard { axis }
            | ShardPolicy::VocabShard { axis } => {
                let dim = entry.logical_shape.get(*axis).ok_or_else(|| {
                    format!("{context}: shard axis {axis} outside logical shape")
                })?;
                if tp > 1 && dim % tp != 0 {
                    return Err(format!(
                        "{context}: shard dim {dim} (axis {axis}) not divisible by Tp={tp}"
                    ));
                }
            }
            ShardPolicy::FusedQkv {
                q_heads,
                kv_heads,
                head_dim,
                ..
            } => {
                if *q_heads == 0 || *kv_heads == 0 || *head_dim == 0 {
                    return Err(format!("{context}: fused QKV geometry must be non-zero"));
                }
                if tp > 1 && (q_heads % tp != 0 || kv_heads % tp != 0) {
                    return Err(format!(
                        "{context}: q_heads={q_heads}/kv_heads={kv_heads} not divisible by Tp={tp}"
                    ));
                }
            }
            ShardPolicy::HeadSharded { n_heads, head_dim } => {
                if *n_heads == 0 || *head_dim == 0 {
                    return Err(format!("{context}: head geometry must be non-zero"));
                }
                if tp > 1 && n_heads % tp != 0 {
                    return Err(format!(
                        "{context}: n_heads={n_heads} not divisible by Tp={tp}"
                    ));
                }
            }
            ShardPolicy::Tied { source } => {
                if source.is_empty() {
                    return Err(format!("{context}: tied source is empty"));
                }
                let source_entry = manifest
                    .iter()
                    .find(|candidate| candidate.name == *source && candidate.layer == entry.layer)
                    .ok_or_else(|| {
                        format!("{context}: Tied source '{source}' has no manifest entry in scope")
                    })?;
                if source_entry.identity() == entry.identity() {
                    return Err(format!("{context}: an entry cannot tie to itself"));
                }
                if source_entry.logical_shape != entry.logical_shape {
                    return Err(format!(
                        "{context}: tied source '{source}' shape {:?} does not match {:?}",
                        source_entry.logical_shape, entry.logical_shape
                    ));
                }
                if source_entry.dtype != entry.dtype {
                    return Err(format!(
                        "{context}: tied source '{source}' dtype {:?} does not match {:?}",
                        source_entry.dtype, entry.dtype
                    ));
                }
                if !entry.dtype_constraint.accepts(source_entry.dtype)
                    || !source_entry.dtype_constraint.accepts(entry.dtype)
                {
                    return Err(format!(
                        "{context}: tied source '{source}' violates the source dtype contract"
                    ));
                }
                if matches!(&source_entry.policy, ShardPolicy::Tied { .. }) {
                    return Err(format!(
                        "{context}: tied source '{source}' is itself tied; chains and cycles are unsupported"
                    ));
                }
            }
            ShardPolicy::ExpertSharded { n_experts, .. } => {
                if *n_experts == 0 || entry.logical_shape.first() != Some(n_experts) {
                    return Err(format!(
                        "{context}: logical_shape {:?} first dimension must equal n_experts={n_experts}",
                        entry.logical_shape
                    ));
                }
            }
            ShardPolicy::ExpertTensorSharded { n_experts, inner } => {
                if *n_experts == 0 || entry.logical_shape.first() != Some(n_experts) {
                    return Err(format!(
                        "{context}: ExpertTensorSharded shape {:?} must start with n_experts={n_experts}",
                        entry.logical_shape
                    ));
                }
                let axis = match inner.as_ref() {
                    ShardPolicy::ColumnShard { axis: 1 }
                    | ShardPolicy::RowShard { axis: 2 } => match inner.as_ref() {
                        ShardPolicy::ColumnShard { axis } | ShardPolicy::RowShard { axis } => *axis,
                        _ => unreachable!(),
                    },
                    ShardPolicy::ColumnShard { axis } | ShardPolicy::RowShard { axis } => {
                        return Err(format!(
                            "{context}: ExpertTensorSharded inner axis {axis} is incompatible with [expert, projection, hidden]"
                        ));
                    }
                    other => {
                        return Err(format!(
                            "{context}: ExpertTensorSharded inner policy {other:?} is unsupported"
                        ));
                    }
                };
                let dim = entry.logical_shape.get(axis).copied().ok_or_else(|| {
                    format!("{context}: ExpertTensorSharded axis {axis} outside shape")
                })?;
                if tp > 1 && dim % tp != 0 {
                    return Err(format!(
                        "{context}: ExpertTensorSharded dim {dim} (axis {axis}) not divisible by Tp={tp}"
                    ));
                }
            }
            ShardPolicy::Replicate | ShardPolicy::Pin(_) => {}
        }
    }
    Ok(())
}

/// Compile declarations against a mesh.
pub fn plan_manifest(
    weights: &[WeightEntry],
    state: &[StateEntry],
    mesh: &DeviceMesh,
    n_layers: usize,
) -> Result<ManifestPlan, String> {
    validate_weight_layers(weights, n_layers)?;
    validate_manifest(weights, mesh)?;
    let mut state_ids = HashSet::new();
    for entry in state {
        if entry.layer >= n_layers {
            return Err(format!(
                "state {:?} layer {} outside n_layers={n_layers}",
                entry.kind, entry.layer
            ));
        }
        if !state_ids.insert((&entry.kind, entry.layer)) {
            return Err(format!(
                "duplicate state declaration {:?}[layer {}]",
                entry.kind, entry.layer
            ));
        }
    }
    let schedule = collective_schedule(weights);
    let layer_collectives = schedule
        .iter()
        .map(|entry| (entry.layer, entry.hint))
        .collect();
    let weight_placements = weights
        .iter()
        .map(|entry| WeightPlacement {
            name: entry.name.clone(),
            layer: entry.layer,
            devices: placement_devices(entry, mesh, n_layers),
        })
        .collect();
    let state_placements = state
        .iter()
        .map(|entry| {
            let mut coord = mesh.coord_of(0);
            let stage = mesh.stage_for_layer(entry.layer, n_layers);
            if let Some(index) = mesh.axes().iter().position(|axis| axis.kind == DimKind::Pp) {
                coord[index] = stage;
            }
            (entry.clone(), mesh.stage_devices(&coord))
        })
        .collect();
    let band_xfers = (0..n_layers)
        .filter_map(|layer| mesh.band_xfer_after(layer, n_layers).map(|hint| (layer, hint)))
        .collect();
    Ok(ManifestPlan {
        weights: weight_placements,
        state: state_placements,
        layer_collectives,
        collective_schedule: schedule,
        band_xfers,
    })
}

// ── Logical expert source identity ─────────────────────────────────────────

/// How one logical expert group is distributed. This declaration is consumed
/// by the G5 executor-owned sealed plan; no rank assignment is resolved here.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ExpertParallelism {
    Single,
    TensorParallel,
    ExpertParallel,
}

/// Stable source identities for expert projections. These names are manifest
/// references, not on-disk paths; the carrier/source resolver owns translation
/// to an artifact namespace.
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum ExpertSourceLayout {
    PackedFused {
        gate_up: String,
        down: String,
        sidecars: Vec<String>,
    },
    PackedSeparate {
        gate: String,
        up: String,
        down: String,
        sidecars: Vec<String>,
    },
    PerExpertFused {
        gate_up: Vec<String>,
        down: Vec<String>,
        sidecars: Vec<String>,
    },
    PerExpertSeparate {
        gate: Vec<String>,
        up: Vec<String>,
        down: Vec<String>,
        sidecars: Vec<String>,
    },
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct ExpertResourceRequirements {
    pub bytes_per_expert: usize,
    pub alignment: usize,
}

/// Architecture-declared identity and source description of one expert group.
/// G5 derives rank ownership and seals the executor plan from this value.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct ExpertGroupSpec {
    pub group: String,
    pub layer: Option<usize>,
    pub n_experts: usize,
    pub parallelism: ExpertParallelism,
    pub assignment: ExpertAssign,
    pub source_layout: ExpertSourceLayout,
    pub resources: ExpertResourceRequirements,
    pub router: String,
    pub execution: String,
}

fn expert_context(spec: &ExpertGroupSpec) -> String {
    format!("expert group '{}' layer {:?}", spec.group, spec.layer)
}

fn source_names(layout: &ExpertSourceLayout) -> Vec<(&'static str, Vec<String>)> {
    match layout {
        ExpertSourceLayout::PackedFused {
            gate_up,
            down,
            sidecars,
        } => vec![
            ("gate_up", vec![gate_up.clone()]),
            ("down", vec![down.clone()]),
            ("sidecar", sidecars.clone()),
        ],
        ExpertSourceLayout::PackedSeparate {
            gate,
            up,
            down,
            sidecars,
        } => vec![
            ("gate", vec![gate.clone()]),
            ("up", vec![up.clone()]),
            ("down", vec![down.clone()]),
            ("sidecar", sidecars.clone()),
        ],
        ExpertSourceLayout::PerExpertFused {
            gate_up,
            down,
            sidecars,
        } => vec![
            ("gate_up", gate_up.clone()),
            ("down", down.clone()),
            ("sidecar", sidecars.clone()),
        ],
        ExpertSourceLayout::PerExpertSeparate {
            gate,
            up,
            down,
            sidecars,
        } => vec![
            ("gate", gate.clone()),
            ("up", up.clone()),
            ("down", down.clone()),
            ("sidecar", sidecars.clone()),
        ],
    }
}

fn manifest_entry<'a>(
    spec: &ExpertGroupSpec,
    manifest: &'a [WeightEntry],
    label: &str,
    name: &str,
) -> Result<&'a WeightEntry, String> {
    let context = expert_context(spec);
    if name.is_empty() {
        return Err(format!("{context}: {label} reference is empty"));
    }
    manifest
        .iter()
        .find(|entry| entry.name == name && entry.layer == spec.layer)
        .ok_or_else(|| format!("{context}: {label} reference '{name}' not found"))
}

fn source_policy_matches(
    spec: &ExpertGroupSpec,
    label: &str,
    policy: &ShardPolicy,
) -> bool {
    match spec.parallelism {
        ExpertParallelism::Single => matches!(
            policy,
            ShardPolicy::Replicate
                | ShardPolicy::Pin(_)
                | ShardPolicy::Tied { .. }
        ),
        ExpertParallelism::TensorParallel => match (label, policy) {
            (
                "gate_up" | "gate" | "up",
                ShardPolicy::ExpertTensorSharded { n_experts, inner },
            ) => {
                *n_experts == spec.n_experts
                    && matches!(inner.as_ref(), ShardPolicy::ColumnShard { axis: 1 })
            }
            ("down", ShardPolicy::ExpertTensorSharded { n_experts, inner }) => {
                *n_experts == spec.n_experts
                    && matches!(inner.as_ref(), ShardPolicy::RowShard { axis: 2 })
            }
            ("sidecar", ShardPolicy::Replicate | ShardPolicy::Tied { .. }) => true,
            _ => false,
        },
        ExpertParallelism::ExpertParallel => match (label, policy) {
            (
                "gate_up" | "gate" | "up" | "down",
                ShardPolicy::ExpertSharded { n_experts, assign },
            ) => *n_experts == spec.n_experts && *assign == spec.assignment,
            ("sidecar", ShardPolicy::Replicate | ShardPolicy::Tied { .. }) => true,
            _ => false,
        },
    }
}

fn source_shape_matches(
    spec: &ExpertGroupSpec,
    label: &str,
    per_expert: bool,
    entry: &WeightEntry,
) -> Result<(), String> {
    let context = expert_context(spec);
    if !source_policy_matches(spec, label, &entry.policy) {
        return Err(format!(
            "{context}: {label} source '{}' has incompatible policy {:?}",
            entry.name, entry.policy
        ));
    }
    if entry.logical_shape.len() < 2 {
        return Err(format!(
            "{context}: {label} source '{}' shape {:?} is too short",
            entry.name, entry.logical_shape
        ));
    }
    if !per_expert && entry.logical_shape.first() != Some(&spec.n_experts) {
        return Err(format!(
            "{context}: {label} source '{}' shape {:?} must start in n_experts={}",
            entry.name, entry.logical_shape, spec.n_experts
        ));
    }
    Ok(())
}

fn validate_expert_sources(
    spec: &ExpertGroupSpec,
    manifest: &[WeightEntry],
) -> Result<(), String> {
    let context = expert_context(spec);
    let router = manifest_entry(spec, manifest, "router", &spec.router)?;
    if !matches!(router.logical_shape.len(), 1 | 2)
        || router.logical_shape.last() != Some(&spec.n_experts)
    {
        return Err(format!(
            "{context}: router '{}' shape {:?} must end in n_experts={}",
            router.name, router.logical_shape, spec.n_experts
        ));
    }
    if !matches!(
        router.policy,
        ShardPolicy::Replicate | ShardPolicy::Pin(_) | ShardPolicy::Tied { .. }
    ) {
        return Err(format!(
            "{context}: router '{}' has incompatible policy {:?}",
            router.name, router.policy
        ));
    }
    let per_expert = matches!(
        spec.source_layout,
        ExpertSourceLayout::PerExpertFused { .. } | ExpertSourceLayout::PerExpertSeparate { .. }
    );
    if per_expert && spec.parallelism != ExpertParallelism::Single {
        return Err(format!(
            "{context}: per-expert source layout is only admitted for Single"
        ));
    }

    for (label, names) in source_names(&spec.source_layout) {
        if names.is_empty() {
            continue;
        }
        if per_expert && label != "sidecar" && names.len() != spec.n_experts {
            return Err(format!(
                "{context}: {label} source count={} != n_experts={}",
                names.len(),
                spec.n_experts
            ));
        }
        let mut seen = HashSet::new();
        let mut shape: Option<Vec<usize>> = None;
        for (index, name) in names.iter().enumerate() {
            if !seen.insert(name.as_str()) {
                return Err(format!(
                    "{context}: duplicate {label} source '{name}' at index {index}"
                ));
            }
            let entry = manifest_entry(spec, manifest, &format!("{label}[{index}]"), name)?;
            source_shape_matches(spec, label, per_expert, entry)?;
            if per_expert {
                if let Some(previous) = &shape {
                    if previous != &entry.logical_shape {
                        return Err(format!(
                            "{context}: {label}[{index}] shape {:?} differs from {:?}",
                            entry.logical_shape, previous
                        ));
                    }
                } else {
                    shape = Some(entry.logical_shape.clone());
                }
            }
        }
    }
    Ok(())
}

/// Validate logical expert source identities. Rank assignment remains owned by
/// G5; this function only proves source names, shapes, and scope are coherent.
pub fn validate_expert_group_specs(
    specs: &[ExpertGroupSpec],
    manifest: &[WeightEntry],
) -> Result<(), String> {
    let mut identities = HashSet::new();
    for entry in manifest {
        if !identities.insert(entry.identity()) {
            return Err(format!(
                "duplicate manifest identity ('{}', {:?})",
                entry.name, entry.layer
            ));
        }
    }
    let mut groups = HashSet::new();
    for spec in specs {
        let context = expert_context(spec);
        if spec.group.is_empty() || spec.router.is_empty() || spec.execution.is_empty() {
            return Err(format!("{context}: group/router/execution identities must be non-empty"));
        }
        if spec.n_experts == 0 || spec.resources.bytes_per_expert == 0 {
            return Err(format!("{context}: n_experts and bytes_per_expert must be non-zero"));
        }
        if spec.resources.alignment == 0 || !spec.resources.alignment.is_power_of_two() {
            return Err(format!("{context}: alignment must be a non-zero power of two"));
        }
        if !groups.insert((&spec.group, spec.layer)) {
            return Err(format!("{context}: duplicate group/layer identity"));
        }
        validate_expert_sources(spec, manifest)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn layer_entry(name: &str, layer: usize, policy: ShardPolicy) -> WeightEntry {
        WeightEntry::layer(name, layer, vec![8, 8], DType::F16, policy)
    }

    #[test]
    fn placement_and_boundaries_use_named_mesh() {
        let mesh = DeviceMesh::rect(&[(DimKind::Pp, 2), (DimKind::Tp, 2)]);
        let embed = WeightEntry::model(
            "token_embd",
            vec![32, 8],
            DType::F16,
            ShardPolicy::Pin(PinTarget::Embed),
        );
        let row = layer_entry("wo", 1, ShardPolicy::RowShard { axis: 1 });
        assert_eq!(placement_devices(&embed, &mesh, 4), vec![0]);
        assert_eq!(placement_devices(&row, &mesh, 4), vec![2, 3]);
        let plan = plan_manifest(
            &[layer_entry("wo", 0, ShardPolicy::RowShard { axis: 1 }), row],
            &[],
            &mesh,
            4,
        )
        .unwrap();
        assert_eq!(plan.layer_collectives.len(), 2);
        assert_eq!(
            plan.band_xfers,
            vec![(1, CollectiveHint::BandXfer { src: 0, dst: 1 })]
        );
    }

    #[test]
    fn schedule_is_ordered_per_operation_not_deduped() {
        let manifest = vec![
            layer_entry("wo", 0, ShardPolicy::RowShard { axis: 1 }),
            layer_entry("down", 0, ShardPolicy::RowShard { axis: 1 }),
        ];
        assert_eq!(
            layer_collectives(&manifest),
            vec![
                (0, CollectiveHint::AllReduce { kind: DimKind::Tp }),
                (0, CollectiveHint::AllReduce { kind: DimKind::Tp }),
            ]
        );
        assert_eq!(collective_schedule(&manifest)[0].name, "wo");
        assert_eq!(collective_schedule(&manifest)[1].name, "down");
    }

    #[test]
    fn validation_covers_divisibility_ties_and_expert_shape() {
        let tp3 = DeviceMesh::rect(&[(DimKind::Tp, 3)]);
        assert!(validate_manifest(
            &[layer_entry("wo", 0, ShardPolicy::RowShard { axis: 1 })],
            &tp3
        )
        .is_err());
        let tied = vec![
            WeightEntry::model(
                "embed",
                vec![8, 8],
                DType::F16,
                ShardPolicy::Pin(PinTarget::Embed),
            ),
            WeightEntry::model(
                "lm_head",
                vec![8, 8],
                DType::F16,
                ShardPolicy::Tied {
                    source: "embed".into(),
                },
            ),
        ];
        assert!(validate_manifest(&tied, &DeviceMesh::single()).is_ok());
        let bad_expert = WeightEntry::layer(
            "experts",
            0,
            vec![3, 8, 8],
            DType::F16,
            ShardPolicy::ExpertSharded {
                n_experts: 4,
                assign: ExpertAssign::Stride,
            },
        );
        assert!(validate_manifest(&[bad_expert], &DeviceMesh::single()).is_err());
    }

    #[test]
    fn expert_source_identity_and_shape_are_checked() {
        let manifest = vec![
            WeightEntry::layer(
                "router",
                0,
                vec![8, 4],
                DType::F16,
                ShardPolicy::Replicate,
            ),
            WeightEntry::layer(
                "gate_up",
                0,
                vec![4, 8, 8],
                DType::F16,
                ShardPolicy::ExpertSharded {
                    n_experts: 4,
                    assign: ExpertAssign::Stride,
                },
            ),
            WeightEntry::layer(
                "down",
                0,
                vec![4, 8, 8],
                DType::F16,
                ShardPolicy::ExpertSharded {
                    n_experts: 4,
                    assign: ExpertAssign::Stride,
                },
            ),
        ];
        let spec = ExpertGroupSpec {
            group: "ffn".into(),
            layer: Some(0),
            n_experts: 4,
            parallelism: ExpertParallelism::ExpertParallel,
            assignment: ExpertAssign::Stride,
            source_layout: ExpertSourceLayout::PackedFused {
                gate_up: "gate_up".into(),
                down: "down".into(),
                sidecars: Vec::new(),
            },
            resources: ExpertResourceRequirements {
                bytes_per_expert: 1024,
                alignment: 256,
            },
            router: "router".into(),
            execution: "moe.ffn".into(),
        };
        assert!(validate_expert_group_specs(&[spec], &manifest).is_ok());
        let bad = ExpertGroupSpec {
            source_layout: ExpertSourceLayout::PackedFused {
                gate_up: "missing".into(),
                down: "down".into(),
                sidecars: Vec::new(),
            },
            ..ExpertGroupSpec {
                group: "ffn2".into(),
                layer: Some(0),
                n_experts: 4,
                parallelism: ExpertParallelism::ExpertParallel,
                assignment: ExpertAssign::Stride,
                source_layout: ExpertSourceLayout::PackedFused {
                    gate_up: "gate_up".into(),
                    down: "down".into(),
                    sidecars: Vec::new(),
                },
                resources: ExpertResourceRequirements {
                    bytes_per_expert: 1024,
                    alignment: 256,
                },
                router: "router".into(),
                execution: "moe.ffn".into(),
            }
        };
        assert!(validate_expert_group_specs(&[bad], &manifest).is_err());
    }

    #[test]
    fn planning_rejects_weight_layer_at_n_layers_and_accepts_last_layer() {
        let mesh = DeviceMesh::single();
        let valid = layer_entry("w", 2, ShardPolicy::Replicate);
        assert!(plan_manifest(&[valid], &[], &mesh, 3).is_ok());
        let out_of_range = layer_entry("w", 3, ShardPolicy::Replicate);
        let error = plan_manifest(&[out_of_range], &[], &mesh, 3).unwrap_err();
        assert!(error.contains("outside n_layers=3"));
    }

    #[test]
    fn tied_entries_require_matching_representation_and_no_tied_chain() {
        let source = WeightEntry::model(
            "source",
            vec![8, 8],
            DType::F16,
            ShardPolicy::Replicate,
        );
        let shape_mismatch = WeightEntry::model(
            "shape_mismatch",
            vec![8, 4],
            DType::F16,
            ShardPolicy::Tied {
                source: "source".into(),
            },
        );
        assert!(validate_manifest(
            &[source.clone(), shape_mismatch],
            &DeviceMesh::single()
        )
        .is_err());

        let dtype_mismatch = WeightEntry::model(
            "dtype_mismatch",
            vec![8, 8],
            DType::F32,
            ShardPolicy::Tied {
                source: "source".into(),
            },
        );
        assert!(validate_manifest(
            &[source.clone(), dtype_mismatch],
            &DeviceMesh::single()
        )
        .is_err());

        let chained_source = WeightEntry::model(
            "chained_source",
            vec![8, 8],
            DType::F16,
            ShardPolicy::Tied {
                source: "source".into(),
            },
        );
        let chain = WeightEntry::model(
            "chain",
            vec![8, 8],
            DType::F16,
            ShardPolicy::Tied {
                source: "chained_source".into(),
            },
        );
        assert!(validate_manifest(
            &[source, chained_source, chain],
            &DeviceMesh::single()
        )
        .is_err());

        let cycle_a = WeightEntry::model(
            "cycle_a",
            vec![8, 8],
            DType::F16,
            ShardPolicy::Tied {
                source: "cycle_b".into(),
            },
        );
        let cycle_b = WeightEntry::model(
            "cycle_b",
            vec![8, 8],
            DType::F16,
            ShardPolicy::Tied {
                source: "cycle_a".into(),
            },
        );
        assert!(validate_manifest(&[cycle_a, cycle_b], &DeviceMesh::single()).is_err());
    }
}
