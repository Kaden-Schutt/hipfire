// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Declarative weight-placement manifest (Phase 2 of the device-mesh plan).
//!
//! An arch declares *what it needs* — for each tensor, a logical shape/dtype and
//! a [`ShardPolicy`] — and the engine (a later `fulfill_manifest(manifest, hfq,
//! mesh)` loop) owns *where it goes*: `placement = manifest (what) × mesh
//! (where)`. Because the engine slices each tensor to its `(stage, tp_rank)`
//! before the arch receives it, global sharded dims never enter arch code.
//!
//! These are **pure CPU data types** — no GPU, no HFQ dependency — so
//! `Architecture::weight_manifest` can be implemented and unit-tested for an
//! arch (transcribing its existing imperative loader) *before* the fulfillment
//! loop exists. See docs/superpowers/plans/2026-07-05-device-mesh-transparent-parallelism.md §4.

use crate::tp_shard::ExpertAssign;
use hipfire_hardware::{CollectiveHint, DeviceMesh, DimKind};
use rdna_compute::DType;

/// Derive the cross-device collective an op's output requires **from its weight
/// [`ShardPolicy`]** — the mini-partitioner that makes sharding a *single*
/// source of truth (declared once in the manifest) instead of a policy in the
/// manifest AND a hand-written hint at lowering (which risks a silent
/// forgotten-reduce). Row-parallel dense → all-reduce over `Tp`; expert-sharded
/// MoE → all-reduce over `Ep`. Column/replicate/pin/etc. need no output reduce.
/// (PP `BandXfer` is a per-layer-boundary concern, not per-op — handled by the
/// pipeline driver, not this map.)
pub fn collective_for_policy(policy: &ShardPolicy) -> Option<CollectiveHint> {
    match policy {
        ShardPolicy::RowShard { .. } => Some(CollectiveHint::AllReduce { kind: DimKind::Tp }),
        ShardPolicy::ExpertSharded { .. } => Some(CollectiveHint::AllReduce { kind: DimKind::Ep }),
        ShardPolicy::ExpertTensorSharded { inner, .. } => collective_for_policy(inner),
        _ => None,
    }
}

/// The pure "placement = manifest × mesh" computation: the global device ids a
/// weight entry lands on, before any GPU upload. This is the testable core of
/// `fulfill_manifest` (the "where"); the "how" (slice/upload the tensor to each
/// device) is the GPU-integration layer on top. A weight goes to the TP/EP
/// group of its owning pipeline stage (replicated, sharded, or expert-split);
/// `Pin`/`Tied` land on one device. Pure `Pp`/`Ep`/single meshes; composed
/// meshes are Phase 5b.
pub fn placement_devices(entry: &WeightEntry, mesh: &DeviceMesh, n_layers: usize) -> Vec<usize> {
    // Owning pipeline stage.
    let stage = match (&entry.placement, &entry.policy, entry.layer) {
        (PlacementHint::Pin(PinTarget::Embed), _, _) => 0,
        (PlacementHint::Pin(PinTarget::Output), _, _) => {
            mesh.size_of(DimKind::Pp).saturating_sub(1)
        }
        (PlacementHint::Policy, ShardPolicy::Pin(PinTarget::Embed), _) => 0,
        (PlacementHint::Policy, ShardPolicy::Pin(PinTarget::Output), _) => {
            mesh.size_of(DimKind::Pp).saturating_sub(1)
        }
        (PlacementHint::Policy, _, Some(l)) => mesh.stage_for_layer(l, n_layers),
        (PlacementHint::Policy, _, None) => 0,
    };
    // Coordinate with the Pp axis set to `stage`, others 0.
    let mut coord = mesh.coord_of(0);
    if let Some(idx) = mesh.axes().iter().position(|a| a.kind == DimKind::Pp) {
        coord[idx] = stage;
    }
    match &entry.policy {
        // Pinned/tied non-sharded weights land on exactly one device.
        ShardPolicy::Pin(_) | ShardPolicy::Tied { .. } => vec![mesh.device_of(&coord)],
        // Every replicated or sharded weight lands on the owning stage's full
        // compute grid. Placement is the "where" (which devices hold a copy or
        // slice); the shard axis and per-device bytes are the "how", resolved by
        // `fulfill_manifest` from the policy × mesh (see weight_store.rs). On a
        // mesh with no Tp axis a TP-shard policy has nothing to shard and
        // replicates across the grid — the EP-only fix.
        _ => mesh.stage_devices(&coord),
    }
}

/// The per-layer all-reduce schedule the executor injects, derived purely from
/// the manifest's sharded weights (single source of truth — see
/// [`collective_for_policy`]). Each `(layer, hint)` is a reduce a row-sharded or
/// expert-sharded weight in that layer implies; the executor applies it over the
/// mesh group at run time. PP `BandXfer` (inter-layer) comes from
/// [`hipfire_hardware::DeviceMesh::band_xfer_after`], not this per-op map.
pub fn layer_collectives(manifest: &[WeightEntry]) -> Vec<(usize, CollectiveHint)> {
    manifest
        .iter()
        .filter_map(|e| Some((e.layer?, collective_for_policy(&e.policy)?)))
        .collect()
}

/// A fully-resolved placement for one weight: the device ids it occupies.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct WeightPlacement {
    pub name: String,
    pub layer: Option<usize>,
    pub devices: Vec<usize>,
}

/// The complete, deterministic compilation of a (weight manifest, state
/// manifest, mesh) into everything the GPU-side `fulfill_manifest` + executor
/// need: where each weight/state lands, the per-layer all-reduce schedule, and
/// the PP band-transfer boundaries. This is the pure, unit-testable "compile"
/// step; `fulfill_manifest` is just the GPU execution of this plan.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct ManifestPlan {
    pub weights: Vec<WeightPlacement>,
    /// (state entry, device ids it occupies).
    pub state: Vec<(StateEntry, Vec<usize>)>,
    /// (layer, all-reduce hint) implied by that layer's sharded weights.
    pub layer_collectives: Vec<(usize, CollectiveHint)>,
    /// (after-layer, band-transfer hint) at PP stage boundaries.
    pub band_xfers: Vec<(usize, CollectiveHint)>,
}

/// Compile a manifest + mesh into a [`ManifestPlan`] (validates first). Pure —
/// no GPU. State co-resides with its layer's owning stage (replicated across
/// the stage's Tp group).
pub fn plan_manifest(
    weights: &[WeightEntry],
    state: &[StateEntry],
    mesh: &DeviceMesh,
    n_layers: usize,
) -> Result<ManifestPlan, String> {
    validate_manifest(weights, mesh)?;
    let w = weights
        .iter()
        .map(|e| WeightPlacement {
            name: e.name.clone(),
            layer: e.layer,
            devices: placement_devices(e, mesh, n_layers),
        })
        .collect();
    let s = state
        .iter()
        .map(|e| {
            let stage = mesh.stage_for_layer(e.layer, n_layers);
            let mut coord = mesh.coord_of(0);
            if let Some(idx) = mesh.axes().iter().position(|a| a.kind == DimKind::Pp) {
                coord[idx] = stage;
            }
            (e.clone(), mesh.stage_devices(&coord))
        })
        .collect();
    let band_xfers = (0..n_layers)
        .filter_map(|l| mesh.band_xfer_after(l, n_layers).map(|h| (l, h)))
        .collect();
    Ok(ManifestPlan {
        weights: w,
        state: s,
        layer_collectives: layer_collectives(weights),
        band_xfers,
    })
}

/// Validate a manifest against a mesh at **load time** (the plan's shape-only
/// safety, §6): every dim/head count a policy shards must divide evenly by its
/// group size, and every `Tied` source must name a real entry. Catches TP
/// shard-math bugs (a wrong-but-legal inner dim) as a load-time `Err` instead
/// of a token-1 GPU page fault. Pure CPU — no upload needed.
pub fn validate_manifest(manifest: &[WeightEntry], mesh: &DeviceMesh) -> Result<(), String> {
    let tp = mesh.size_of(DimKind::Tp);
    let names: std::collections::HashSet<&str> = manifest.iter().map(|e| e.name.as_str()).collect();
    for e in manifest {
        let ctx = || format!("{}[layer {:?}]", e.name, e.layer);
        match &e.policy {
            ShardPolicy::ColumnShard { axis } | ShardPolicy::RowShard { axis } => {
                let dim = e.logical_shape.get(*axis).copied().unwrap_or(0);
                if tp > 1 && dim % tp != 0 {
                    return Err(format!(
                        "{}: shard dim {dim} (axis {axis}) not divisible by Tp={tp}",
                        ctx()
                    ));
                }
            }
            ShardPolicy::FusedQkv {
                q_heads, kv_heads, ..
            } => {
                if tp > 1 && (q_heads % tp != 0 || kv_heads % tp != 0) {
                    return Err(format!(
                        "{}: q_heads={q_heads}/kv_heads={kv_heads} not divisible by Tp={tp}",
                        ctx()
                    ));
                }
            }
            ShardPolicy::HeadSharded { n_heads, .. } => {
                if tp > 1 && n_heads % tp != 0 {
                    return Err(format!(
                        "{}: n_heads={n_heads} not divisible by Tp={tp}",
                        ctx()
                    ));
                }
            }
            ShardPolicy::Tied { source } => {
                if !names.contains(source.as_str()) {
                    return Err(format!(
                        "{}: Tied source '{source}' has no manifest entry",
                        ctx()
                    ));
                }
            }
            ShardPolicy::ExpertTensorSharded { n_experts, inner } => {
                if e.logical_shape.len() != 3 || e.logical_shape.iter().any(|&dim| dim == 0) {
                    return Err(format!(
                        "{}: ExpertTensorSharded logical_shape {:?} must be 3D with no zero dimensions",
                        ctx(),
                        e.logical_shape
                    ));
                }
                if e.logical_shape.first().copied() != Some(*n_experts) {
                    return Err(format!(
                        "{}: ExpertTensorSharded logical_shape {:?} first dimension must equal n_experts={n_experts}",
                        ctx(),
                        e.logical_shape
                    ));
                }
                // Expert intermediate dim must be divisible by Tp and the
                // resulting slice must be a multiple of 256 (the quant group
                // size for MQ2G256/MQ3G256 experts).
                // logical_shape: [n_experts, 2*inter, hidden] (gate‖up) or
                // [n_experts, hidden, inter] (down).
                // Gate/up (ColumnShard): sharded dim is axis-1 (2*inter).
                // Down (RowShard): sharded dim is axis-2 (inter).
                let (axis, kind_name) = match inner.as_ref() {
                    ShardPolicy::ColumnShard { axis: 1 } => (1, "ColumnShard (2*inter)"),
                    ShardPolicy::RowShard { axis: 2 } => (2, "RowShard (inter)"),
                    ShardPolicy::ColumnShard { axis } | ShardPolicy::RowShard { axis } => {
                        return Err(format!(
                            "{}: ExpertTensorSharded inner shard axis {axis} is incompatible with the [n_experts, projection, hidden] layout",
                            ctx()
                        ));
                    }
                    inner => {
                        return Err(format!(
                            "{}: ExpertTensorSharded inner policy {inner:?} is incompatible with the [n_experts, projection, hidden] layout",
                            ctx()
                        ));
                    }
                };
                let d = e.logical_shape.get(axis).copied().unwrap_or(0);
                if tp > 1 && !(d % tp == 0 && (d / tp) % 256 == 0) {
                    return Err(format!(
                        "{}: ExpertTensorSharded {} dim {d} (axis {}) \
                         not divisible by Tp={tp} \
                         or slice {} not a multiple of 256",
                        ctx(),
                        kind_name,
                        axis,
                        d / tp
                    ));
                }
            }
            // Replicate / ExpertSharded (Stride tolerates uneven) / Pin / Vocab: no divisibility gate.
            _ => {}
        }
    }
    Ok(())
}

/// Non-layer placement targets (resolved against the mesh, not hardcoded).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum PinTarget {
    /// Token embedding — pinned to pipeline stage 0.
    Embed,
    /// Final norm + lm_head — pinned to the last stage (Megatron output
    /// convention); resolves to the mesh's output device.
    Output,
}

/// How a weight tensor is placed/sharded across a mesh axis. `FusedQKV` /
/// `HeadSharded` shard the **head axis** via `tp_shard`'s head-range math;
/// `ExpertSharded` carries the MoE packed-blob convention. Only genuinely
/// bespoke weights would need a future `Custom` escape (no known fleet example).
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum ShardPolicy {
    /// Full tensor on every rank in the group (attention when replicated,
    /// norms, biases).
    Replicate,
    /// Column-parallel (Megatron): split output dim `axis` across the TP group;
    /// no all-reduce on its own output.
    ColumnShard { axis: usize },
    /// Row-parallel (Megatron): split input dim `axis`; consumer op all-reduces.
    RowShard { axis: usize },
    /// MoE experts distributed across the group (`assign` policy); non-owned
    /// experts get the shared zeroed-dummy so they contribute 0 to the reduce.
    ExpertSharded {
        n_experts: usize,
        assign: ExpertAssign,
    },
    /// Fused QKV (GQA): split at the Q|K|V(|gate) block boundaries (`layout`),
    /// then shard each sub-block by head group via `q_head_range`/`kv_head_range`.
    FusedQkv {
        q_heads: usize,
        kv_heads: usize,
        head_dim: usize,
        layout: FusedQkvLayout,
    },
    /// Per-head weights (DeltaNet `w_alpha`/`w_beta`/`wz`) sharded on the head
    /// axis via `dn_value_head_range`.
    HeadSharded { n_heads: usize, head_dim: usize },
    /// Ties this logical tensor to another entry; fulfillment aliases when the
    /// source is local and materializes a copy when placement crosses devices.
    Tied { source: String },
    /// Pinned to a mesh-derived non-layer location (embed / output).
    Pin(PinTarget),
    /// TP logit sharding of lm_head along the vocab `axis`.
    VocabShard { axis: usize },
    /// Tensor-parallel MoE expert sharding: each rank holds a TP-sliced
    /// fraction of every expert's weight. `inner` = `ColumnShard` for gate‖up
    /// projections, `RowShard` for down projections; placement spans the Tp
    /// group (not Ep). Scaffolds manifest-transparent MoE loading where
    /// arch-imperative loaders hold the current GPU path.
    ExpertTensorSharded {
        n_experts: usize,
        inner: Box<ShardPolicy>,
    },
}

/// The fused-QKV block order an arch packs into one tensor (so the engine knows
/// where to cut before head-group sharding). Data, not code.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum FusedQkvLayout {
    /// `[Q | K | V]` concatenated (vanilla / GQA attention).
    Qkv,
    /// `[Q | gate]` (some DeltaNet fused projections).
    QGate,
    /// `[Q | K | V | Z]` — DeltaNet with a separate gate/normalization block.
    QkvZ,
}

/// One entry in an arch's weight manifest: a logical tensor + how to place it.
/// `layer` is `Some(idx)` for a per-layer weight (placed on that layer's stage)
/// or `None` for a model-level weight (embed/lm_head/final-norm).
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum SourceDType {
    /// The source may be any dtype accepted by the source/loader contract.
    Any,
    /// The source must have this dtype.
    Exact(DType),
    /// The source may have any one of these dtypes; fulfillment preserves the
    /// selected source dtype on the resident tensor.
    OneOf(Vec<DType>),
}

#[derive(Clone, PartialEq, Eq, Debug)]
pub struct DTypeConstraint {
    /// Dtype(s) accepted from the source/resolver side. Fulfillment validates
    /// this allow-list but preserves the source dtype on the resident tensor;
    /// this type deliberately does not promise conversion or a resident dtype.
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
}

/// Optional placement override independent of tensor identity/policy. This is
/// needed for a tied lm_head: its identity aliases token_embd, but its
/// resident copy belongs on the output PP stage.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum PlacementHint {
    Policy,
    Pin(PinTarget),
}

#[derive(Clone, PartialEq, Eq, Debug)]
pub struct WeightEntry {
    pub name: String,
    pub layer: Option<usize>,
    pub logical_shape: Vec<usize>,
    /// Logical dtype expected by the architecture. Fulfillment preserves the
    /// source dtype unless a separate conversion path explicitly changes it.
    pub dtype: DType,
    pub dtype_constraint: DTypeConstraint,
    pub placement: PlacementHint,
    pub policy: ShardPolicy,
}

impl WeightEntry {
    /// A model-level (non-layer) weight.
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

    /// A per-layer weight bound to `layer`.
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
}

/// The kind of per-layer state an arch holds — placed by the same mesh
/// projection as weights (co-resident with its layer's stage under PP,
/// replicated or head-sharded under TP). Collapses the ~15 format-exploded
/// `KvCache::*_multi` ctors + the DeltaNet `la_to_device` sidecar into one
/// keyed store (device-mesh plan §4).
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum StateKind {
    /// KV cache in a given quant mode (the quant string, e.g. "q8"/"fwht2").
    Kv { quant: String },
    /// Recurrent state (DeltaNet S-matrix) — head-sharded under TP.
    Recurrent,
    /// Conv state (lfm2moe short conv) — kernel_size-1 elems per conv layer.
    Conv,
}

/// One entry in an arch's *state* manifest. `layer` is the **global** layer
/// index (the store keys by global index, which is what defines the DeltaNet
/// LA-compact `la_to_device` sidecar out of existence — the LA-vs-full-attn
/// knowledge lives in manifest construction via `config.layer_types`).
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

/// How an architecture distributes one logical expert group.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ExpertParallelism {
    /// One rank owns and executes the complete expert group.
    Single,
    /// Every rank owns a compact slot for each expert's tensor-parallel slice.
    TensorParallel,
    /// Experts are assigned to ranks and executed on their owning rank.
    ExpertParallel,
}

/// The source representation of the expert tensors in the model artifact.
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum ExpertSourceLayout {
    /// Packed fused gate-up plus down projection.
    PackedFused {
        gate_up: String,
        down: String,
        sidecars: Vec<String>,
    },
    /// Packed separate gate, up, and down projections.
    PackedSeparate {
        gate: String,
        up: String,
        down: String,
        sidecars: Vec<String>,
    },
    /// One fused gate-up and down source per expert.
    PerExpertFused {
        gate_up: Vec<String>,
        down: Vec<String>,
        sidecars: Vec<String>,
    },
    /// Separate gate, up, and down source per expert.
    PerExpertSeparate {
        gate: Vec<String>,
        up: Vec<String>,
        down: Vec<String>,
        sidecars: Vec<String>,
    },
}

/// CPU-side resource constraints needed to admit an expert group.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct ExpertResourceRequirements {
    /// Resident bytes required by one expert (before any runtime paging).
    pub bytes_per_expert: usize,
    /// Required byte alignment of an expert's compact local slot.
    pub alignment: usize,
}

/// The only collectives admitted by an expert-group plan. The variant encodes
/// both post-combine ordering and the policy-derived collective axis.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ExpertPostCombineAllReduce {
    TensorParallel,
    ExpertParallel,
}

impl ExpertPostCombineAllReduce {
    pub const fn axis(self) -> DimKind {
        match self {
            Self::TensorParallel => DimKind::Tp,
            Self::ExpertParallel => DimKind::Ep,
        }
    }
}

fn post_combine_for_parallelism(
    parallelism: ExpertParallelism,
) -> Option<ExpertPostCombineAllReduce> {
    match parallelism {
        ExpertParallelism::Single => None,
        ExpertParallelism::TensorParallel => Some(ExpertPostCombineAllReduce::TensorParallel),
        ExpertParallelism::ExpertParallel => Some(ExpertPostCombineAllReduce::ExpertParallel),
    }
}

/// Architecture-declared description of one logical expert group.
///
/// The strings are stable manifest references: `router` names the router
/// weight/plan consumed by the existing router machinery and `execution` names
/// the arch execution plan. Keeping these references symbolic avoids coupling
/// this CPU-only manifest to feature-gated GPU or pager types.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct ExpertGroupSpec {
    /// Stable architecture identity for this logical MoE block.
    pub group: String,
    /// Manifest scope: `Some(layer)` for a layer-local group, `None` for a
    /// model-level group.
    pub layer: Option<usize>,
    pub n_experts: usize,
    pub parallelism: ExpertParallelism,
    pub assignment: ExpertAssign,
    pub source_layout: ExpertSourceLayout,
    pub resources: ExpertResourceRequirements,
    pub router: String,
    pub execution: String,
}

/// One resolved global expert id and its compact local slot on the owning
/// rank. `owner` is relative to the expert group, not a global device id.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct ExpertPlacement {
    pub global_id: usize,
    pub owner: usize,
    pub local_slot: usize,
}

/// The resolved expert-group plan consumed by later fulfillment/execution
/// layers. It deliberately contains no GPU handles.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct ExpertGroupPlan {
    pub group: String,
    pub layer: Option<usize>,
    pub n_experts: usize,
    pub parallelism: ExpertParallelism,
    pub assignment: ExpertAssign,
    pub experts: Vec<ExpertPlacement>,
    pub source_layout: ExpertSourceLayout,
    pub resources: ExpertResourceRequirements,
    pub router: String,
    pub execution: String,
    pub collective: Option<ExpertPostCombineAllReduce>,
}

fn expert_context(spec: &ExpertGroupSpec) -> String {
    format!("expert group '{}' layer {:?}", spec.group, spec.layer)
}

fn validate_expert_group_shape(spec: &ExpertGroupSpec, group_size: usize) -> Result<(), String> {
    let context = expert_context(spec);
    if group_size == 0 {
        return Err(format!("{context}: group_size=0 is invalid"));
    }
    if spec.n_experts == 0 {
        return Err(format!("{context}: n_experts=0 is invalid"));
    }

    match spec.parallelism {
        ExpertParallelism::Single => {
            if group_size != 1 {
                return Err(format!(
                    "{context}: Single requires group_size=1, got {group_size}"
                ));
            }
        }
        ExpertParallelism::TensorParallel | ExpertParallelism::ExpertParallel => {
            if spec.parallelism == ExpertParallelism::ExpertParallel
                && spec.n_experts % group_size != 0
            {
                return Err(format!(
                    "{context}: n_experts={} must divide evenly across group_size={group_size}",
                    spec.n_experts
                ));
            }
        }
    }
    if spec.resources.bytes_per_expert == 0 {
        return Err(format!("{context}: bytes_per_expert=0 is invalid"));
    }
    if spec.resources.alignment == 0 || !spec.resources.alignment.is_power_of_two() {
        return Err(format!(
            "{context}: alignment={} is invalid",
            spec.resources.alignment
        ));
    }
    if spec.group.is_empty() {
        return Err(format!("{context}: group reference '' is invalid"));
    }
    if spec.execution.is_empty() {
        return Err(format!("{context}: execution reference '' is invalid"));
    }
    Ok(())
}

/// Validate one expert group without checking manifest references.
pub fn validate_expert_group_spec(spec: &ExpertGroupSpec, group_size: usize) -> Result<(), String> {
    validate_expert_group_shape(spec, group_size)
}

fn validate_manifest_reference<'a>(
    spec: &ExpertGroupSpec,
    manifest: &'a [WeightEntry],
    label: &str,
    name: &str,
) -> Result<&'a WeightEntry, String> {
    let context = expert_context(spec);
    if name.is_empty() {
        return Err(format!("{context}: {label} reference '' is invalid"));
    }
    let mut found = None;
    for entry in manifest
        .iter()
        .filter(|entry| entry.name == name && entry.layer == spec.layer)
    {
        if found.is_some() {
            return Err(format!(
                "{context}: {label} reference '{name}' is ambiguous in manifest scope"
            ));
        }
        found = Some(entry);
    }
    found
        .ok_or_else(|| format!("{context}: {label} reference '{name}' not found in manifest scope"))
}

fn validate_source_policy(
    spec: &ExpertGroupSpec,
    entry: &WeightEntry,
    label: &str,
    role: ProjectionRole,
) -> Result<(), String> {
    let context = expert_context(spec);
    match spec.parallelism {
        ExpertParallelism::Single => {
            if !matches!(
                entry.policy,
                ShardPolicy::Replicate | ShardPolicy::Pin(_) | ShardPolicy::Tied { .. }
            ) {
                return Err(format!(
                    "{context}: {label} reference '{}' has incompatible policy {:?} for Single",
                    entry.name, entry.policy
                ));
            }
        }
        ExpertParallelism::TensorParallel => match &entry.policy {
            ShardPolicy::ExpertTensorSharded { n_experts, inner } => {
                if *n_experts != spec.n_experts {
                    return Err(format!(
                        "{context}: {label} reference '{}' embeds n_experts={} but spec requires {}",
                        entry.name, n_experts, spec.n_experts
                    ));
                }
                let expected = match role {
                    ProjectionRole::GateUp => ShardPolicy::ColumnShard { axis: 1 },
                    ProjectionRole::Down => ShardPolicy::RowShard { axis: 2 },
                };
                if inner.as_ref() != &expected {
                    return Err(format!(
                        "{context}: {label} reference '{}' has inner policy {:?}, expected {:?}",
                        entry.name, inner, expected
                    ));
                }
            }
            policy => {
                return Err(format!(
                    "{context}: {label} reference '{}' has incompatible policy {:?} for TensorParallel",
                    entry.name, policy
                ));
            }
        },
        ExpertParallelism::ExpertParallel => match &entry.policy {
            ShardPolicy::ExpertSharded { n_experts, assign } => {
                if *n_experts != spec.n_experts {
                    return Err(format!(
                        "{context}: {label} reference '{}' embeds n_experts={} but spec requires {}",
                        entry.name, n_experts, spec.n_experts
                    ));
                }
                if *assign != spec.assignment {
                    return Err(format!(
                        "{context}: {label} reference '{}' has assignment {:?}, expected {:?}",
                        entry.name, assign, spec.assignment
                    ));
                }
            }
            policy => {
                return Err(format!(
                    "{context}: {label} reference '{}' has incompatible policy {:?} for ExpertParallel",
                    entry.name, policy
                ));
            }
        },
    }
    Ok(())
}

fn validate_unique_per_expert_sources(
    spec: &ExpertGroupSpec,
    groups: &[(&str, &[String])],
) -> Result<(), String> {
    let mut names = std::collections::HashSet::new();
    for (label, refs) in groups {
        for (idx, name) in refs.iter().enumerate() {
            if !names.insert(name.as_str()) {
                return Err(format!(
                    "{}: duplicate per-expert source '{}' at {label}[{idx}]",
                    expert_context(spec),
                    name
                ));
            }
        }
    }
    Ok(())
}

#[derive(Clone, Copy)]
enum ProjectionRole {
    GateUp,
    Down,
}

fn validate_per_expert_projection(
    spec: &ExpertGroupSpec,
    manifest: &[WeightEntry],
    label: &str,
    names: &[String],
    role: ProjectionRole,
) -> Result<(), String> {
    let context = expert_context(spec);
    if names.len() != spec.n_experts {
        return Err(format!(
            "{context}: {label} source count={} does not match n_experts={}",
            names.len(),
            spec.n_experts
        ));
    }
    let mut shape: Option<&[usize]> = None;
    for (idx, name) in names.iter().enumerate() {
        let entry = validate_manifest_reference(spec, manifest, &format!("{label}[{idx}]"), name)?;
        validate_source_policy(spec, entry, &format!("{label}[{idx}]"), role)?;
        if entry.logical_shape.len() < 2 {
            return Err(format!(
                "{context}: {label}[{idx}] reference '{}' has incompatible logical_shape {:?}",
                entry.name, entry.logical_shape
            ));
        }
        if let Some(expected) = shape {
            if expected != entry.logical_shape.as_slice() {
                return Err(format!(
                    "{context}: {label}[{idx}] reference '{}' logical_shape {:?} differs from {:?}",
                    entry.name, entry.logical_shape, expected
                ));
            }
        } else {
            shape = Some(entry.logical_shape.as_slice());
        }
    }
    Ok(())
}

fn validate_packed_projection(
    spec: &ExpertGroupSpec,
    manifest: &[WeightEntry],
    label: &str,
    name: &str,
    role: ProjectionRole,
) -> Result<(), String> {
    let entry = validate_manifest_reference(spec, manifest, label, name)?;
    if entry.logical_shape.len() != 3 || entry.logical_shape.iter().any(|&dim| dim == 0) {
        return Err(format!(
            "{}: {label} reference '{}' logical_shape {:?} must be 3D with no zero dimensions",
            expert_context(spec),
            entry.name,
            entry.logical_shape
        ));
    }
    if entry.logical_shape.first().copied() != Some(spec.n_experts) {
        return Err(format!(
            "{}: {label} reference '{}' logical_shape {:?} is incompatible with n_experts={}",
            expert_context(spec),
            entry.name,
            entry.logical_shape,
            spec.n_experts
        ));
    }
    validate_source_policy(spec, entry, label, role)?;
    Ok(())
}

fn validate_sidecar_policy(
    spec: &ExpertGroupSpec,
    entry: &WeightEntry,
    label: &str,
) -> Result<(), String> {
    let context = expert_context(spec);
    match spec.parallelism {
        ExpertParallelism::Single => {
            if !matches!(
                entry.policy,
                ShardPolicy::Replicate | ShardPolicy::Pin(_) | ShardPolicy::Tied { .. }
            ) {
                return Err(format!(
                    "{context}: {label} reference '{}' has incompatible sidecar policy {:?}",
                    entry.name, entry.policy
                ));
            }
        }
        ExpertParallelism::TensorParallel => {
            if !matches!(entry.policy, ShardPolicy::Replicate) {
                return Err(format!(
                    "{context}: {label} reference '{}' has incompatible sidecar policy {:?}",
                    entry.name, entry.policy
                ));
            }
        }
        ExpertParallelism::ExpertParallel => match &entry.policy {
            ShardPolicy::Replicate => {}
            ShardPolicy::ExpertSharded { n_experts, assign }
                if *n_experts == spec.n_experts && *assign == spec.assignment => {}
            policy => {
                return Err(format!(
                    "{context}: {label} reference '{}' has incompatible sidecar policy {:?}",
                    entry.name, policy
                ));
            }
        },
    }
    Ok(())
}

fn validate_sidecars(
    spec: &ExpertGroupSpec,
    manifest: &[WeightEntry],
    sidecars: &[String],
) -> Result<(), String> {
    for (idx, name) in sidecars.iter().enumerate() {
        let label = format!("sidecar[{idx}]");
        let entry = validate_manifest_reference(spec, manifest, &label, name)?;
        validate_sidecar_policy(spec, entry, &label)?;
        if entry.logical_shape.first().copied() != Some(spec.n_experts) {
            return Err(format!(
                "{}: {label} reference '{}' logical_shape {:?} is incompatible with n_experts={}",
                expert_context(spec),
                entry.name,
                entry.logical_shape,
                spec.n_experts
            ));
        }
    }
    Ok(())
}

fn validate_expert_group_references(
    spec: &ExpertGroupSpec,
    manifest: &[WeightEntry],
) -> Result<(), String> {
    let router = validate_manifest_reference(spec, manifest, "router", &spec.router)?;
    if !matches!(router.logical_shape.len(), 1 | 2) {
        return Err(format!(
            "{}: router reference '{}' has logical_shape {:?}; router rank must be 1 or 2",
            expert_context(spec),
            router.name,
            router.logical_shape
        ));
    }
    if router.logical_shape.last().copied() != Some(spec.n_experts) {
        return Err(format!(
            "{}: router reference '{}' has logical_shape {:?}; last dimension must equal n_experts={}",
            expert_context(spec), router.name, router.logical_shape, spec.n_experts
        ));
    }
    if !matches!(
        router.policy,
        ShardPolicy::Replicate | ShardPolicy::Pin(_) | ShardPolicy::Tied { .. }
    ) {
        return Err(format!(
            "{}: router reference '{}' has incompatible policy {:?}",
            expert_context(spec),
            router.name,
            router.policy
        ));
    }
    let per_expert = matches!(
        spec.source_layout,
        ExpertSourceLayout::PerExpertFused { .. } | ExpertSourceLayout::PerExpertSeparate { .. }
    );
    if per_expert && spec.parallelism != ExpertParallelism::Single {
        return Err(format!(
            "{}: PerExpert source layout is unsupported for {:?}; global expert source placement is defined only for Single",
            expert_context(spec), spec.parallelism
        ));
    }
    match &spec.source_layout {
        ExpertSourceLayout::PackedFused {
            gate_up,
            down,
            sidecars,
        } => {
            validate_packed_projection(
                spec,
                manifest,
                "source gate_up",
                gate_up,
                ProjectionRole::GateUp,
            )?;
            validate_packed_projection(spec, manifest, "source down", down, ProjectionRole::Down)?;
            validate_sidecars(spec, manifest, sidecars)?;
        }
        ExpertSourceLayout::PackedSeparate {
            gate,
            up,
            down,
            sidecars,
        } => {
            validate_packed_projection(
                spec,
                manifest,
                "source gate",
                gate,
                ProjectionRole::GateUp,
            )?;
            validate_packed_projection(spec, manifest, "source up", up, ProjectionRole::GateUp)?;
            validate_packed_projection(spec, manifest, "source down", down, ProjectionRole::Down)?;
            validate_sidecars(spec, manifest, sidecars)?;
        }
        ExpertSourceLayout::PerExpertFused {
            gate_up,
            down,
            sidecars,
        } => {
            validate_unique_per_expert_sources(
                spec,
                &[("source gate_up", gate_up), ("source down", down)],
            )?;
            validate_per_expert_projection(
                spec,
                manifest,
                "source gate_up",
                gate_up,
                ProjectionRole::GateUp,
            )?;
            validate_per_expert_projection(
                spec,
                manifest,
                "source down",
                down,
                ProjectionRole::Down,
            )?;
            validate_sidecars(spec, manifest, sidecars)?;
        }
        ExpertSourceLayout::PerExpertSeparate {
            gate,
            up,
            down,
            sidecars,
        } => {
            validate_unique_per_expert_sources(
                spec,
                &[
                    ("source gate", gate),
                    ("source up", up),
                    ("source down", down),
                ],
            )?;
            validate_per_expert_projection(
                spec,
                manifest,
                "source gate",
                gate,
                ProjectionRole::GateUp,
            )?;
            validate_per_expert_projection(
                spec,
                manifest,
                "source up",
                up,
                ProjectionRole::GateUp,
            )?;
            validate_per_expert_projection(
                spec,
                manifest,
                "source down",
                down,
                ProjectionRole::Down,
            )?;
            validate_sidecars(spec, manifest, sidecars)?;
        }
    }
    Ok(())
}

/// Validate all architecture-declared expert groups against the weight
/// manifest, including stable group/layer identity and manifest scope.
pub fn validate_expert_group_specs(
    specs: &[ExpertGroupSpec],
    manifest: &[WeightEntry],
    group_size: usize,
) -> Result<(), String> {
    let mut manifest_names = std::collections::HashSet::new();
    for entry in manifest {
        if !manifest_names.insert((&entry.name, entry.layer)) {
            return Err(format!(
                "duplicate manifest (name, layer) ('{}', {:?})",
                entry.name, entry.layer
            ));
        }
    }
    let mut identities = std::collections::HashSet::new();
    for spec in specs {
        validate_expert_group_shape(spec, group_size)?;
        let context = expert_context(spec);
        if !identities.insert((&spec.group, spec.layer)) {
            return Err(format!("{context}: duplicate group/layer identity"));
        }
        validate_expert_group_references(spec, manifest)?;
    }
    Ok(())
}

/// Resolve an architecture-declared expert group for a group of `group_size`
/// ranks. Local slots are compact independently for each owner.
pub fn resolve_expert_group_plan(
    spec: &ExpertGroupSpec,
    manifest: &[WeightEntry],
    group_size: usize,
) -> Result<ExpertGroupPlan, String> {
    validate_expert_group_specs(std::slice::from_ref(spec), manifest, group_size)?;
    resolve_expert_group_plan_unchecked(spec, group_size)
}

/// Resolve multiple groups after validating their identities and manifest
/// references as one batch.
pub fn resolve_expert_group_plans(
    specs: &[ExpertGroupSpec],
    manifest: &[WeightEntry],
    group_size: usize,
) -> Result<Vec<ExpertGroupPlan>, String> {
    validate_expert_group_specs(specs, manifest, group_size)?;
    specs
        .iter()
        .map(|spec| resolve_expert_group_plan_unchecked(spec, group_size))
        .collect()
}

fn resolve_expert_group_plan_unchecked(
    spec: &ExpertGroupSpec,
    group_size: usize,
) -> Result<ExpertGroupPlan, String> {
    let mut next_slot = vec![0usize; group_size];
    let mut experts = Vec::with_capacity(spec.n_experts);
    for global_id in 0..spec.n_experts {
        match spec.parallelism {
            ExpertParallelism::Single => experts.push(ExpertPlacement {
                global_id,
                owner: 0,
                local_slot: global_id,
            }),
            ExpertParallelism::TensorParallel => {
                for owner in 0..group_size {
                    let local_slot = next_slot[owner];
                    next_slot[owner] += 1;
                    experts.push(ExpertPlacement {
                        global_id,
                        owner,
                        local_slot,
                    });
                }
            }
            ExpertParallelism::ExpertParallel => {
                let owner = match spec.assignment {
                    ExpertAssign::Contiguous => global_id / (spec.n_experts / group_size),
                    ExpertAssign::Stride => global_id % group_size,
                };
                let local_slot = next_slot[owner];
                next_slot[owner] += 1;
                experts.push(ExpertPlacement {
                    global_id,
                    owner,
                    local_slot,
                });
            }
        }
    }
    Ok(ExpertGroupPlan {
        group: spec.group.clone(),
        layer: spec.layer,
        n_experts: spec.n_experts,
        parallelism: spec.parallelism,
        assignment: spec.assignment,
        experts,
        source_layout: spec.source_layout.clone(),
        resources: spec.resources,
        router: spec.router.clone(),
        execution: spec.execution.clone(),
        collective: post_combine_for_parallelism(spec.parallelism),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn entry_constructors_set_layer_scope() {
        let e = WeightEntry::model(
            "token_embd",
            vec![152064, 4096],
            DType::F16,
            ShardPolicy::Pin(PinTarget::Embed),
        );
        assert_eq!(e.layer, None);
        assert!(matches!(e.policy, ShardPolicy::Pin(PinTarget::Embed)));

        let l = WeightEntry::layer(
            "wo",
            3,
            vec![4096, 4096],
            DType::F16,
            ShardPolicy::RowShard { axis: 1 },
        );
        assert_eq!(l.layer, Some(3));
        assert!(matches!(l.policy, ShardPolicy::RowShard { axis: 1 }));
    }

    #[test]
    fn dtype_constraints_describe_source_dtypes_only() {
        let raw =
            DTypeConstraint::source_from_sources(vec![DType::Q8_0, DType::F16, DType::ParoQ4G128]);
        assert_eq!(
            raw.source,
            SourceDType::OneOf(vec![DType::Q8_0, DType::F16, DType::ParoQ4G128])
        );

        let projection = WeightEntry::model(
            "projection",
            vec![8, 8],
            DType::F16,
            ShardPolicy::ColumnShard { axis: 0 },
        );
        assert_eq!(projection.dtype_constraint, DTypeConstraint::any_source());
    }

    #[test]
    fn plan_manifest_ties_placement_collectives_and_bands() {
        // 2-layer MoE-ish manifest: attention (wo row) + experts, KV state.
        let mut w = Vec::new();
        let mut st = Vec::new();
        for l in 0..2 {
            w.push(WeightEntry::layer(
                "wo",
                l,
                vec![8, 8],
                DType::F16,
                ShardPolicy::RowShard { axis: 1 },
            ));
            w.push(WeightEntry::layer(
                "experts",
                l,
                vec![4, 8, 8],
                DType::F16,
                ShardPolicy::ExpertSharded {
                    n_experts: 4,
                    assign: ExpertAssign::Stride,
                },
            ));
            st.push(StateEntry::new(
                StateKind::Kv {
                    quant: String::new(),
                },
                l,
            ));
        }
        // PP 2-stage mesh, 2 layers → one band boundary after layer 0.
        let pp = DeviceMesh::rect(&[(DimKind::Pp, 2)]);
        let plan = plan_manifest(&w, &st, &pp, 2).unwrap();
        // 4 weight placements, 2 state placements.
        assert_eq!(plan.weights.len(), 4);
        assert_eq!(plan.state.len(), 2);
        // layer-0 weights on stage 0 (device 0), layer-1 on stage 1 (device 1).
        let wo0 = plan
            .weights
            .iter()
            .find(|p| p.name == "wo" && p.layer == Some(0))
            .unwrap();
        assert_eq!(wo0.devices, vec![0]);
        let wo1 = plan
            .weights
            .iter()
            .find(|p| p.name == "wo" && p.layer == Some(1))
            .unwrap();
        assert_eq!(wo1.devices, vec![1]);
        // collectives: wo → Tp, experts → Ep, per layer (4 total).
        assert_eq!(plan.layer_collectives.len(), 4);
        // one band transfer after layer 0.
        assert_eq!(
            plan.band_xfers,
            vec![(0, CollectiveHint::BandXfer { src: 0, dst: 1 })]
        );
    }

    #[test]
    fn validate_manifest_catches_indivisible_and_dangling() {
        let tp3 = DeviceMesh::rect(&[(DimKind::Tp, 3)]);
        // 8 not divisible by Tp=3 → error at load.
        let bad = vec![WeightEntry::layer(
            "wo",
            0,
            vec![8, 8],
            DType::F16,
            ShardPolicy::RowShard { axis: 1 },
        )];
        assert!(validate_manifest(&bad, &tp3).is_err());
        // Divisible (Tp=2) → ok.
        let tp2 = DeviceMesh::rect(&[(DimKind::Tp, 2)]);
        assert!(validate_manifest(&bad, &tp2).is_ok());
        // Dangling Tied source → error.
        let dangling = vec![WeightEntry::model(
            "lm_head",
            vec![8, 8],
            DType::F16,
            ShardPolicy::Tied {
                source: "nope".into(),
            },
        )];
        assert!(validate_manifest(&dangling, &DeviceMesh::single()).is_err());
        // Tied to a present entry → ok.
        let tied_ok = vec![
            WeightEntry::model(
                "token_embd",
                vec![8, 8],
                DType::F16,
                ShardPolicy::Pin(PinTarget::Embed),
            ),
            WeightEntry::model(
                "lm_head",
                vec![8, 8],
                DType::F16,
                ShardPolicy::Tied {
                    source: "token_embd".into(),
                },
            ),
        ];
        assert!(validate_manifest(&tied_ok, &tp2).is_ok());
    }

    #[test]
    fn validate_manifest_rejects_expert_tensor_shape_expert_count_mismatch() {
        let manifest = vec![WeightEntry::layer(
            "experts",
            0,
            vec![3, 512, 8],
            DType::F16,
            ShardPolicy::ExpertTensorSharded {
                n_experts: 4,
                inner: Box::new(ShardPolicy::ColumnShard { axis: 1 }),
            },
        )];
        let err = validate_manifest(&manifest, &DeviceMesh::rect(&[(DimKind::Tp, 2)])).unwrap_err();
        assert!(err.contains("experts[layer Some(0)]"));
        assert!(err.contains("n_experts=4"));
    }

    #[test]
    fn validate_manifest_accepts_matching_expert_tensor_shape_expert_count() {
        let manifest = vec![WeightEntry::layer(
            "experts",
            0,
            vec![4, 512, 8],
            DType::F16,
            ShardPolicy::ExpertTensorSharded {
                n_experts: 4,
                inner: Box::new(ShardPolicy::ColumnShard { axis: 1 }),
            },
        )];
        assert!(validate_manifest(&manifest, &DeviceMesh::rect(&[(DimKind::Tp, 2)])).is_ok());
    }

    #[test]
    fn head_sharded_and_recurrent_conv_variants() {
        // DeltaNet HeadSharded (w_alpha/w_beta/wz): per-head shard, no own-output
        // all-reduce (the cross-head mix all-reduces on wo, like ColumnShard).
        let hs = ShardPolicy::HeadSharded {
            n_heads: 16,
            head_dim: 128,
        };
        assert_eq!(collective_for_policy(&hs), None);
        let e = WeightEntry::layer("w_alpha", 2, vec![16 * 128], DType::F16, hs);
        // HeadSharded shards on the Tp axis → spans the Tp group; on an Ep-only
        // mesh it replicates across the EP group.
        let tp = DeviceMesh::rect(&[(DimKind::Tp, 2)]);
        assert_eq!(placement_devices(&e, &tp, 4), vec![0, 1]);
        // On an Ep-only mesh a HeadSharded weight has no Tp axis to shard, so it
        // replicates across the whole EP group (each rank runs full attention).
        let ep = DeviceMesh::rect(&[(DimKind::Ep, 2)]);
        assert_eq!(placement_devices(&e, &ep, 4), vec![0, 1]);
        // FusedQkv QkvZ layout (DeltaNet fused projection) is expressible.
        let fq = ShardPolicy::FusedQkv {
            q_heads: 8,
            kv_heads: 2,
            head_dim: 256,
            layout: FusedQkvLayout::QkvZ,
        };
        assert_eq!(collective_for_policy(&fq), None);
        // Recurrent + Conv state kinds (DeltaNet S-matrix + short conv).
        assert!(matches!(
            StateEntry::new(StateKind::Recurrent, 2).kind,
            StateKind::Recurrent
        ));
        assert!(matches!(
            StateEntry::new(StateKind::Conv, 5).kind,
            StateKind::Conv
        ));
    }

    #[test]
    fn collective_derived_from_policy() {
        // Row-parallel → Tp all-reduce; expert → Ep all-reduce.
        assert_eq!(
            collective_for_policy(&ShardPolicy::RowShard { axis: 1 }),
            Some(CollectiveHint::AllReduce { kind: DimKind::Tp })
        );
        assert_eq!(
            collective_for_policy(&ShardPolicy::ExpertSharded {
                n_experts: 8,
                assign: ExpertAssign::Stride
            }),
            Some(CollectiveHint::AllReduce { kind: DimKind::Ep })
        );
        // Column-parallel / replicate / pin produce no output reduce.
        assert_eq!(
            collective_for_policy(&ShardPolicy::ColumnShard { axis: 0 }),
            None
        );
        assert_eq!(collective_for_policy(&ShardPolicy::Replicate), None);
        assert_eq!(
            collective_for_policy(&ShardPolicy::Pin(PinTarget::Embed)),
            None
        );
    }

    #[test]
    fn layer_collectives_from_toy_dense_manifest() {
        // Build a 2-layer dense manifest by hand (mirrors the toy arch): each
        // layer has wo + ffn_down row-parallel → 2 Tp all-reduces per layer.
        let mut m = Vec::new();
        for l in 0..2 {
            m.push(WeightEntry::layer(
                "wq",
                l,
                vec![8, 8],
                DType::F16,
                ShardPolicy::ColumnShard { axis: 0 },
            ));
            m.push(WeightEntry::layer(
                "wo",
                l,
                vec![8, 8],
                DType::F16,
                ShardPolicy::RowShard { axis: 1 },
            ));
            m.push(WeightEntry::layer(
                "ffn_down",
                l,
                vec![8, 32],
                DType::F16,
                ShardPolicy::RowShard { axis: 1 },
            ));
            m.push(WeightEntry::layer(
                "norm",
                l,
                vec![8],
                DType::F32,
                ShardPolicy::Replicate,
            ));
        }
        let sched = layer_collectives(&m);
        // 2 per layer × 2 layers = 4 Tp all-reduces; column/replicate contribute none.
        assert_eq!(sched.len(), 4);
        assert!(sched
            .iter()
            .all(|(_, h)| matches!(h, CollectiveHint::AllReduce { kind: DimKind::Tp })));
        assert_eq!(sched.iter().filter(|(l, _)| *l == 0).count(), 2);
        assert_eq!(sched.iter().filter(|(l, _)| *l == 1).count(), 2);
    }

    #[test]
    fn placement_where_by_mesh_and_policy() {
        let embed = WeightEntry::model(
            "e",
            vec![256, 8],
            DType::F16,
            ShardPolicy::Pin(PinTarget::Embed),
        );
        let out = WeightEntry::model(
            "lm",
            vec![256, 8],
            DType::F16,
            ShardPolicy::Pin(PinTarget::Output),
        );
        let wo = WeightEntry::layer(
            "wo",
            3,
            vec![8, 8],
            DType::F16,
            ShardPolicy::RowShard { axis: 1 },
        );
        let exp = WeightEntry::layer(
            "experts",
            3,
            vec![8, 8],
            DType::F16,
            ShardPolicy::ExpertSharded {
                n_experts: 8,
                assign: ExpertAssign::Stride,
            },
        );

        // Single-GPU: everything on device 0.
        let single = DeviceMesh::single();
        assert_eq!(placement_devices(&wo, &single, 4), vec![0]);

        // PP 2×1, 4 layers: layer 3 is on stage 1 → device 1; embed on 0; output on last (1).
        let pp = DeviceMesh::rect(&[(DimKind::Pp, 2)]);
        assert_eq!(placement_devices(&wo, &pp, 4), vec![1]);
        assert_eq!(placement_devices(&embed, &pp, 4), vec![0]);
        assert_eq!(placement_devices(&out, &pp, 4), vec![1]);

        // EP 1×4: experts span the whole Ep group; dense replicated over it too.
        let ep = DeviceMesh::rect(&[(DimKind::Ep, 4)]);
        assert_eq!(placement_devices(&exp, &ep, 4), vec![0, 1, 2, 3]);
    }

    #[test]
    fn state_entry_keyed_by_global_layer() {
        let s = StateEntry::new(StateKind::Kv { quant: "q8".into() }, 7);
        assert_eq!(s.layer, 7);
        assert!(matches!(s.kind, StateKind::Kv { .. }));
        let r = StateEntry::new(StateKind::Recurrent, 3);
        assert!(matches!(r.kind, StateKind::Recurrent));
    }

    #[test]
    fn expert_sharded_carries_assign() {
        let p = ShardPolicy::ExpertSharded {
            n_experts: 128,
            assign: ExpertAssign::Stride,
        };
        if let ShardPolicy::ExpertSharded { n_experts, assign } = p {
            assert_eq!(n_experts, 128);
            assert_eq!(assign, ExpertAssign::Stride);
        } else {
            panic!("wrong variant");
        }
    }

    #[test]
    fn ep_only_replicates_non_expert_weights() {
        let ep = DeviceMesh::rect(&[(DimKind::Ep, 2)]);
        // Replicate (deepseek4 attention/norm/router class) → every EP rank.
        let rep = WeightEntry::layer("attn_norm", 0, vec![8], DType::F32, ShardPolicy::Replicate);
        assert_eq!(placement_devices(&rep, &ep, 4), vec![0, 1]);
        // TP-shard policy (minimax attention class) → degenerates to replication
        // across the EP group; there is no Tp axis to shard along.
        let col = WeightEntry::layer(
            "wq",
            0,
            vec![8, 8],
            DType::F16,
            ShardPolicy::ColumnShard { axis: 0 },
        );
        assert_eq!(placement_devices(&col, &ep, 4), vec![0, 1]);
        // ExpertSharded still spans the whole EP group (sliced by expert at fulfill).
        let exp = WeightEntry::layer(
            "experts",
            0,
            vec![4, 8, 8],
            DType::F16,
            ShardPolicy::ExpertSharded {
                n_experts: 4,
                assign: ExpertAssign::Stride,
            },
        );
        assert_eq!(placement_devices(&exp, &ep, 4), vec![0, 1]);
    }

    fn expert_manifest(
        layer: Option<usize>,
        n_experts: usize,
        parallelism: ExpertParallelism,
    ) -> Vec<WeightEntry> {
        let gate_up_policy = match parallelism {
            ExpertParallelism::Single => ShardPolicy::Replicate,
            ExpertParallelism::TensorParallel => ShardPolicy::ExpertTensorSharded {
                n_experts,
                inner: Box::new(ShardPolicy::ColumnShard { axis: 1 }),
            },
            ExpertParallelism::ExpertParallel => ShardPolicy::ExpertSharded {
                n_experts,
                assign: ExpertAssign::Stride,
            },
        };
        let down_policy = match parallelism {
            ExpertParallelism::Single => ShardPolicy::Replicate,
            ExpertParallelism::TensorParallel => ShardPolicy::ExpertTensorSharded {
                n_experts,
                inner: Box::new(ShardPolicy::RowShard { axis: 2 }),
            },
            ExpertParallelism::ExpertParallel => ShardPolicy::ExpertSharded {
                n_experts,
                assign: ExpertAssign::Stride,
            },
        };
        vec![
            WeightEntry {
                name: "mlp.gate".into(),
                layer,
                logical_shape: vec![4, n_experts],
                dtype: DType::F16,
                dtype_constraint: DTypeConstraint::any_source(),
                placement: PlacementHint::Policy,
                policy: ShardPolicy::Replicate,
            },
            WeightEntry {
                name: "experts.gate_up".into(),
                layer,
                logical_shape: vec![n_experts, 4, 4],
                dtype: DType::F16,
                dtype_constraint: DTypeConstraint::any_source(),
                placement: PlacementHint::Policy,
                policy: gate_up_policy,
            },
            WeightEntry {
                name: "experts.down".into(),
                layer,
                logical_shape: vec![n_experts, 4, 4],
                dtype: DType::F16,
                dtype_constraint: DTypeConstraint::any_source(),
                placement: PlacementHint::Policy,
                policy: down_policy,
            },
        ]
    }

    fn expert_spec(parallelism: ExpertParallelism) -> ExpertGroupSpec {
        ExpertGroupSpec {
            group: "block-0".into(),
            layer: Some(0),
            n_experts: 4,
            parallelism,
            assignment: ExpertAssign::Stride,
            source_layout: ExpertSourceLayout::PackedFused {
                gate_up: "experts.gate_up".into(),
                down: "experts.down".into(),
                sidecars: Vec::new(),
            },
            resources: ExpertResourceRequirements {
                bytes_per_expert: 1024,
                alignment: 256,
            },
            router: "mlp.gate".into(),
            execution: "moe.feed_forward".into(),
        }
    }

    #[test]
    fn expert_group_single_has_zero_collectives() {
        let spec = expert_spec(ExpertParallelism::Single);
        let plan = resolve_expert_group_plan(
            &spec,
            &expert_manifest(Some(0), 4, ExpertParallelism::Single),
            1,
        )
        .unwrap();

        assert!(plan.collective.is_none());
        assert_eq!(plan.group, "block-0");
        assert_eq!(plan.layer, Some(0));
        assert!(plan.experts.iter().all(|expert| expert.owner == 0));
        assert_eq!(plan.experts[3].local_slot, 3);
    }

    #[test]
    fn expert_group_tp_has_one_post_combine_collective() {
        let spec = expert_spec(ExpertParallelism::TensorParallel);
        let plan = resolve_expert_group_plan(
            &spec,
            &expert_manifest(Some(0), 4, ExpertParallelism::TensorParallel),
            2,
        )
        .unwrap();

        assert_eq!(
            plan.collective,
            Some(ExpertPostCombineAllReduce::TensorParallel)
        );
    }

    #[test]
    fn expert_group_ep_has_one_post_combine_collective() {
        let spec = expert_spec(ExpertParallelism::ExpertParallel);
        let plan = resolve_expert_group_plan(
            &spec,
            &expert_manifest(Some(0), 4, ExpertParallelism::ExpertParallel),
            2,
        )
        .unwrap();

        assert_eq!(
            plan.collective,
            Some(ExpertPostCombineAllReduce::ExpertParallel)
        );
        assert_eq!(plan.experts[0].owner, 0);
        assert_eq!(plan.experts[1].owner, 1);
        assert_eq!(plan.experts[2].local_slot, 1);
    }

    #[test]
    fn expert_group_collective_authority_is_group_level() {
        let spec = expert_spec(ExpertParallelism::ExpertParallel);
        let plan = resolve_expert_group_plan(
            &spec,
            &expert_manifest(Some(0), 4, ExpertParallelism::ExpertParallel),
            2,
        )
        .unwrap();
        assert_eq!(plan.collective.unwrap().axis(), DimKind::Ep);
    }

    #[test]
    fn expert_group_tp_maps_each_expert_to_all_ranks_without_divisibility() {
        let mut spec = expert_spec(ExpertParallelism::TensorParallel);
        spec.n_experts = 3;

        let plan = resolve_expert_group_plan(
            &spec,
            &expert_manifest(Some(0), 3, ExpertParallelism::TensorParallel),
            2,
        )
        .unwrap();
        assert_eq!(plan.experts.len(), 6);
        for global_id in 0..3 {
            let owners_and_slots: Vec<_> = plan
                .experts
                .iter()
                .filter(|expert| expert.global_id == global_id)
                .map(|expert| (expert.owner, expert.local_slot))
                .collect();
            assert_eq!(owners_and_slots, vec![(0, global_id), (1, global_id)]);
        }
    }

    #[test]
    fn expert_group_identity_and_references_are_validated() {
        let manifest = expert_manifest(Some(0), 4, ExpertParallelism::ExpertParallel);
        let spec = expert_spec(ExpertParallelism::ExpertParallel);
        let duplicate = spec.clone();

        let err = validate_expert_group_specs(&[spec, duplicate], &manifest, 2).unwrap_err();
        assert!(err.contains("block-0"));
        assert!(err.contains("layer Some(0)"));
    }

    #[test]
    fn expert_group_rejects_zero_experts_and_zero_group_size() {
        let mut spec = expert_spec(ExpertParallelism::Single);
        spec.n_experts = 0;
        assert!(resolve_expert_group_plan(
            &spec,
            &expert_manifest(Some(0), 1, ExpertParallelism::Single),
            1
        )
        .is_err());

        spec.n_experts = 1;
        assert!(resolve_expert_group_plan(
            &spec,
            &expert_manifest(Some(0), 1, ExpertParallelism::Single),
            0
        )
        .is_err());
    }

    #[test]
    fn expert_group_rejects_multi_rank_single() {
        let spec = expert_spec(ExpertParallelism::Single);
        assert!(resolve_expert_group_plan(
            &spec,
            &expert_manifest(Some(0), 4, ExpertParallelism::Single),
            2
        )
        .is_err());
    }

    #[test]
    fn expert_group_rejects_invalid_resource_metadata() {
        let mut spec = expert_spec(ExpertParallelism::Single);
        spec.resources.bytes_per_expert = 0;
        let err = resolve_expert_group_plan(
            &spec,
            &expert_manifest(Some(0), 4, ExpertParallelism::Single),
            1,
        )
        .unwrap_err();
        assert!(err.contains("block-0"));
        assert!(err.contains("bytes_per_expert=0"));

        spec.resources.bytes_per_expert = 1;
        spec.resources.alignment = 3;
        let err = resolve_expert_group_plan(
            &spec,
            &expert_manifest(Some(0), 4, ExpertParallelism::Single),
            1,
        )
        .unwrap_err();
        assert!(err.contains("alignment=3"));
    }

    #[test]
    fn expert_group_tp_non_divisible_placement_is_valid() {
        let mut spec = expert_spec(ExpertParallelism::TensorParallel);
        spec.n_experts = 3;
        let plan = resolve_expert_group_plan(
            &spec,
            &expert_manifest(Some(0), 3, ExpertParallelism::TensorParallel),
            2,
        )
        .unwrap();
        assert_eq!(plan.experts.len(), 6);
    }

    #[test]
    fn expert_group_reference_errors_name_group_and_bad_value() {
        let manifest = expert_manifest(Some(0), 4, ExpertParallelism::Single);
        let mut spec = expert_spec(ExpertParallelism::Single);
        spec.router = "missing.router".into();

        let err = validate_expert_group_specs(&[spec.clone()], &manifest, 1).unwrap_err();
        assert!(err.contains("block-0"));
        assert!(err.contains("missing.router"));

        spec.router = "mlp.gate".into();
        spec.source_layout = ExpertSourceLayout::PackedFused {
            gate_up: "missing.source".into(),
            down: "experts.down".into(),
            sidecars: Vec::new(),
        };
        let err = validate_expert_group_specs(&[spec], &manifest, 1).unwrap_err();
        assert!(err.contains("missing.source"));

        let mut sidecar = expert_spec(ExpertParallelism::Single);
        sidecar.source_layout = ExpertSourceLayout::PackedFused {
            gate_up: "experts.gate_up".into(),
            down: "experts.down".into(),
            sidecars: vec![String::new()],
        };
        let err = validate_expert_group_specs(&[sidecar], &manifest, 1).unwrap_err();
        assert!(err.contains("sidecar[0]"));
    }

    #[test]
    fn expert_group_preserves_fused_separate_and_sidecar_references() {
        let mut manifest = expert_manifest(Some(0), 3, ExpertParallelism::Single);
        manifest.push(WeightEntry::layer(
            "experts.scale",
            0,
            vec![3, 4],
            DType::F16,
            ShardPolicy::Replicate,
        ));
        let mut spec = expert_spec(ExpertParallelism::Single);
        spec.n_experts = 3;
        spec.source_layout = ExpertSourceLayout::PackedFused {
            gate_up: "experts.gate_up".into(),
            down: "experts.down".into(),
            sidecars: vec!["experts.scale".into()],
        };
        let fused = resolve_expert_group_plan(&spec, &manifest, 1).unwrap();
        assert_eq!(fused.source_layout, spec.source_layout);

        for name in ["experts.gate", "experts.up"] {
            manifest.push(WeightEntry::layer(
                name,
                0,
                vec![3, 4, 4],
                DType::F16,
                ShardPolicy::Replicate,
            ));
        }
        let mut packed_separate = spec.clone();
        packed_separate.source_layout = ExpertSourceLayout::PackedSeparate {
            gate: "experts.gate".into(),
            up: "experts.up".into(),
            down: "experts.down".into(),
            sidecars: vec!["experts.scale".into()],
        };
        let packed_separate_plan =
            resolve_expert_group_plan(&packed_separate, &manifest, 1).unwrap();
        assert_eq!(
            packed_separate_plan.source_layout,
            packed_separate.source_layout
        );

        let names = ["experts.gate", "experts.up", "experts.down"];
        for prefix in names {
            for expert in 0..3 {
                manifest.push(WeightEntry::layer(
                    format!("{prefix}.{expert}"),
                    0,
                    vec![4, 4],
                    DType::F16,
                    ShardPolicy::Replicate,
                ));
            }
        }
        let mut separate = spec;
        separate.source_layout = ExpertSourceLayout::PerExpertSeparate {
            gate: (0..3).map(|e| format!("experts.gate.{e}")).collect(),
            up: (0..3).map(|e| format!("experts.up.{e}")).collect(),
            down: (0..3).map(|e| format!("experts.down.{e}")).collect(),
            sidecars: vec!["experts.scale".into()],
        };
        let per_expert = resolve_expert_group_plan(&separate, &manifest, 1).unwrap();
        assert_eq!(per_expert.source_layout, separate.source_layout);

        for expert in 0..3 {
            manifest.push(WeightEntry::layer(
                format!("experts.gate_up.{expert}"),
                0,
                vec![4, 4],
                DType::F16,
                ShardPolicy::Replicate,
            ));
        }
        let mut fused_per_expert = separate;
        fused_per_expert.source_layout = ExpertSourceLayout::PerExpertFused {
            gate_up: (0..3).map(|e| format!("experts.gate_up.{e}")).collect(),
            down: (0..3).map(|e| format!("experts.down.{e}")).collect(),
            sidecars: vec!["experts.scale".into()],
        };
        let fused_per_expert_plan =
            resolve_expert_group_plan(&fused_per_expert, &manifest, 1).unwrap();
        assert_eq!(
            fused_per_expert_plan.source_layout,
            fused_per_expert.source_layout
        );
    }

    #[test]
    fn expert_group_rejects_wrong_source_shape_and_policy() {
        let mut malformed = expert_manifest(Some(0), 4, ExpertParallelism::Single);
        malformed
            .iter_mut()
            .find(|entry| entry.name == "experts.gate_up")
            .unwrap()
            .logical_shape = vec![3, 4, 4];
        let spec = expert_spec(ExpertParallelism::Single);
        let err = validate_expert_group_specs(&[spec], &malformed, 1).unwrap_err();
        assert!(err.contains("experts.gate_up"));
        assert!(err.contains("logical_shape"));

        let mut wrong_policy = expert_manifest(Some(0), 4, ExpertParallelism::TensorParallel);
        wrong_policy
            .iter_mut()
            .find(|entry| entry.name == "experts.down")
            .unwrap()
            .policy = ShardPolicy::Replicate;
        let err = validate_expert_group_specs(
            &[expert_spec(ExpertParallelism::TensorParallel)],
            &wrong_policy,
            2,
        )
        .unwrap_err();
        assert!(err.contains("experts.down"));
        assert!(err.contains("incompatible policy"));
    }

    #[test]
    fn expert_group_rejects_empty_references_and_wrong_scope() {
        let manifest = expert_manifest(Some(0), 4, ExpertParallelism::Single);
        let mut empty = expert_spec(ExpertParallelism::Single);
        empty.router.clear();
        let err = validate_expert_group_specs(&[empty], &manifest, 1).unwrap_err();
        assert!(err.contains("block-0"));
        assert!(err.contains("reference ''"));

        let mut wrong_scope = expert_spec(ExpertParallelism::Single);
        wrong_scope.layer = Some(1);
        let err = validate_expert_group_specs(&[wrong_scope], &manifest, 1).unwrap_err();
        assert!(err.contains("layer Some(1)"));
        assert!(err.contains("mlp.gate"));
    }

    #[test]
    fn expert_group_collective_is_derived_from_parallelism() {
        for (parallelism, expected) in [
            (ExpertParallelism::Single, None),
            (
                ExpertParallelism::TensorParallel,
                Some(ExpertPostCombineAllReduce::TensorParallel),
            ),
            (
                ExpertParallelism::ExpertParallel,
                Some(ExpertPostCombineAllReduce::ExpertParallel),
            ),
        ] {
            let spec = expert_spec(parallelism);
            let plan = resolve_expert_group_plan(
                &spec,
                &expert_manifest(Some(0), 4, parallelism),
                if parallelism == ExpertParallelism::Single {
                    1
                } else {
                    2
                },
            )
            .unwrap();
            assert_eq!(plan.collective, expected);
        }
    }

    #[test]
    fn expert_group_rejects_mismatched_embedded_expert_policy_metadata() {
        let mut ep = expert_manifest(Some(0), 4, ExpertParallelism::ExpertParallel);
        if let ShardPolicy::ExpertSharded { n_experts, .. } = &mut ep[1].policy {
            *n_experts = 3;
        }
        let err =
            validate_expert_group_specs(&[expert_spec(ExpertParallelism::ExpertParallel)], &ep, 2)
                .unwrap_err();
        assert!(err.contains("experts.gate_up"));
        assert!(err.contains("n_experts"));

        let mut assignment = expert_manifest(Some(0), 4, ExpertParallelism::ExpertParallel);
        if let ShardPolicy::ExpertSharded { assign, .. } = &mut assignment[1].policy {
            *assign = ExpertAssign::Contiguous;
        }
        let err = validate_expert_group_specs(
            &[expert_spec(ExpertParallelism::ExpertParallel)],
            &assignment,
            2,
        )
        .unwrap_err();
        assert!(err.contains("assignment"));
        assert!(err.contains("Contiguous"));

        let mut tp = expert_manifest(Some(0), 4, ExpertParallelism::TensorParallel);
        if let ShardPolicy::ExpertTensorSharded { inner, .. } = &mut tp[1].policy {
            *inner = Box::new(ShardPolicy::Replicate);
        }
        let err =
            validate_expert_group_specs(&[expert_spec(ExpertParallelism::TensorParallel)], &tp, 2)
                .unwrap_err();
        assert!(err.contains("experts.gate_up"));
        assert!(err.contains("inner"));
    }

    #[test]
    fn expert_group_router_accepts_one_or_two_dimensions_and_rejects_wrong_last_dim() {
        let mut one_d = expert_manifest(Some(0), 4, ExpertParallelism::Single);
        one_d[0].logical_shape = vec![4];
        assert!(
            validate_expert_group_specs(&[expert_spec(ExpertParallelism::Single)], &one_d, 1)
                .is_ok()
        );

        let two_d = expert_manifest(Some(0), 4, ExpertParallelism::Single);
        assert!(
            validate_expert_group_specs(&[expert_spec(ExpertParallelism::Single)], &two_d, 1)
                .is_ok()
        );

        let mut wrong = two_d;
        wrong[0].logical_shape = vec![4, 5];
        let err = validate_expert_group_specs(&[expert_spec(ExpertParallelism::Single)], &wrong, 1)
            .unwrap_err();
        assert!(err.contains("router"));
        assert!(err.contains("logical_shape"));
        assert!(err.contains("n_experts=4"));
    }

    #[test]
    fn expert_group_rejects_per_expert_layouts_for_tp_and_ep() {
        let mut spec = expert_spec(ExpertParallelism::TensorParallel);
        spec.source_layout = ExpertSourceLayout::PerExpertFused {
            gate_up: (0..4).map(|e| format!("experts.gate_up.{e}")).collect(),
            down: (0..4).map(|e| format!("experts.down.{e}")).collect(),
            sidecars: Vec::new(),
        };
        let err = validate_expert_group_specs(
            &[spec.clone()],
            &expert_manifest(Some(0), 4, ExpertParallelism::TensorParallel),
            2,
        )
        .unwrap_err();
        assert!(err.contains("PerExpert"));

        spec.parallelism = ExpertParallelism::ExpertParallel;
        let err = validate_expert_group_specs(
            &[spec],
            &expert_manifest(Some(0), 4, ExpertParallelism::ExpertParallel),
            2,
        )
        .unwrap_err();
        assert!(err.contains("PerExpert"));
    }

    #[test]
    fn expert_group_rejects_duplicate_manifest_and_per_expert_sources() {
        let mut manifest = expert_manifest(Some(0), 4, ExpertParallelism::Single);
        manifest.push(manifest[1].clone());
        let err =
            validate_expert_group_specs(&[expert_spec(ExpertParallelism::Single)], &manifest, 1)
                .unwrap_err();
        assert!(err.contains("duplicate manifest"));
        assert!(err.contains("experts.gate_up"));

        let mut per = expert_spec(ExpertParallelism::Single);
        per.source_layout = ExpertSourceLayout::PerExpertFused {
            gate_up: vec![
                "experts.gate_up.0".into(),
                "experts.gate_up.0".into(),
                "experts.gate_up.2".into(),
                "experts.gate_up.3".into(),
            ],
            down: (0..4).map(|e| format!("experts.down.{e}")).collect(),
            sidecars: Vec::new(),
        };
        let mut per_manifest = expert_manifest(Some(0), 4, ExpertParallelism::Single);
        for name in [
            "experts.gate_up.0",
            "experts.gate_up.2",
            "experts.gate_up.3",
            "experts.down.0",
            "experts.down.1",
            "experts.down.2",
            "experts.down.3",
        ] {
            per_manifest.push(WeightEntry::layer(
                name,
                0,
                vec![4, 4],
                DType::F16,
                ShardPolicy::Replicate,
            ));
        }
        let err = validate_expert_group_specs(&[per], &per_manifest, 1).unwrap_err();
        assert!(err.contains("duplicate per-expert source"));
        assert!(err.contains("experts.gate_up.0"));
    }

    #[test]
    fn expert_group_accepts_same_manifest_name_at_different_layers() {
        let mut manifest = expert_manifest(Some(0), 4, ExpertParallelism::Single);
        manifest.extend(expert_manifest(Some(1), 4, ExpertParallelism::Single));
        let mut spec = expert_spec(ExpertParallelism::Single);
        spec.layer = Some(1);
        assert!(validate_expert_group_specs(&[spec], &manifest, 1).is_ok());
    }

    #[test]
    fn expert_group_rejects_swapped_or_wrong_tp_projection_axes() {
        let mut gate_axis = expert_manifest(Some(0), 4, ExpertParallelism::TensorParallel);
        if let ShardPolicy::ExpertTensorSharded { inner, .. } = &mut gate_axis[1].policy {
            *inner = Box::new(ShardPolicy::ColumnShard { axis: 0 });
        }
        let err = validate_expert_group_specs(
            &[expert_spec(ExpertParallelism::TensorParallel)],
            &gate_axis,
            2,
        )
        .unwrap_err();
        assert!(err.contains("source gate_up"));
        assert!(err.contains("axis: 1"));

        let mut swapped_gate_down = expert_manifest(Some(0), 4, ExpertParallelism::TensorParallel);
        if let ShardPolicy::ExpertTensorSharded { inner, .. } = &mut swapped_gate_down[1].policy {
            *inner = Box::new(ShardPolicy::RowShard { axis: 2 });
        }
        let err = validate_expert_group_specs(
            &[expert_spec(ExpertParallelism::TensorParallel)],
            &swapped_gate_down,
            2,
        )
        .unwrap_err();
        assert!(err.contains("source gate_up"));
        assert!(err.contains("ColumnShard"));

        let mut down_axis = expert_manifest(Some(0), 4, ExpertParallelism::TensorParallel);
        if let ShardPolicy::ExpertTensorSharded { inner, .. } = &mut down_axis[2].policy {
            *inner = Box::new(ShardPolicy::ColumnShard { axis: 1 });
        }
        let err = validate_expert_group_specs(
            &[expert_spec(ExpertParallelism::TensorParallel)],
            &down_axis,
            2,
        )
        .unwrap_err();
        assert!(err.contains("source down"));
        assert!(err.contains("RowShard"));
    }

    #[test]
    fn expert_group_sidecar_policies_are_separate_from_projection_policies() {
        let mut tp_replicated = expert_manifest(Some(0), 4, ExpertParallelism::TensorParallel);
        tp_replicated.push(WeightEntry::layer(
            "experts.scale",
            0,
            vec![4, 4],
            DType::F16,
            ShardPolicy::Replicate,
        ));
        let mut tp_spec = expert_spec(ExpertParallelism::TensorParallel);
        tp_spec.source_layout = ExpertSourceLayout::PackedFused {
            gate_up: "experts.gate_up".into(),
            down: "experts.down".into(),
            sidecars: vec!["experts.scale".into()],
        };
        assert!(validate_expert_group_specs(&[tp_spec.clone()], &tp_replicated, 2).is_ok());

        let mut ep_expert = expert_manifest(Some(0), 4, ExpertParallelism::ExpertParallel);
        ep_expert.push(WeightEntry::layer(
            "experts.scale",
            0,
            vec![4, 4],
            DType::F16,
            ShardPolicy::ExpertSharded {
                n_experts: 4,
                assign: ExpertAssign::Stride,
            },
        ));
        let mut ep_spec = expert_spec(ExpertParallelism::ExpertParallel);
        ep_spec.source_layout = ExpertSourceLayout::PackedFused {
            gate_up: "experts.gate_up".into(),
            down: "experts.down".into(),
            sidecars: vec!["experts.scale".into()],
        };
        assert!(validate_expert_group_specs(&[ep_spec], &ep_expert, 2).is_ok());

        let mut tp_expert = expert_manifest(Some(0), 4, ExpertParallelism::TensorParallel);
        tp_expert.push(WeightEntry::layer(
            "experts.scale",
            0,
            vec![4, 4],
            DType::F16,
            ShardPolicy::ExpertTensorSharded {
                n_experts: 4,
                inner: Box::new(ShardPolicy::ColumnShard { axis: 1 }),
            },
        ));
        let err = validate_expert_group_specs(&[tp_spec], &tp_expert, 2).unwrap_err();
        assert!(err.contains("sidecar[0]"));
        assert!(err.contains("ExpertTensorSharded"));
    }

    #[test]
    fn expert_group_tp_sidecar_rejects_pin_but_accepts_replicate() {
        let spec = expert_spec(ExpertParallelism::TensorParallel);
        let pin = WeightEntry::layer(
            "experts.scale",
            0,
            vec![4, 4],
            DType::F16,
            ShardPolicy::Pin(PinTarget::Embed),
        );
        assert!(validate_sidecar_policy(&spec, &pin, "sidecar[0]").is_err());

        let replicate = WeightEntry::layer(
            "experts.scale",
            0,
            vec![4, 4],
            DType::F16,
            ShardPolicy::Replicate,
        );
        assert!(validate_sidecar_policy(&spec, &replicate, "sidecar[0]").is_ok());
    }

    #[test]
    fn expert_group_tp_sidecar_rejects_tied_but_accepts_replicate() {
        let spec = expert_spec(ExpertParallelism::TensorParallel);
        let tied = WeightEntry::layer(
            "experts.scale",
            0,
            vec![4, 4],
            DType::F16,
            ShardPolicy::Tied {
                source: "source".into(),
            },
        );
        assert!(validate_sidecar_policy(&spec, &tied, "sidecar[0]").is_err());

        let replicate = WeightEntry::layer(
            "experts.scale",
            0,
            vec![4, 4],
            DType::F16,
            ShardPolicy::Replicate,
        );
        assert!(validate_sidecar_policy(&spec, &replicate, "sidecar[0]").is_ok());
    }

    #[test]
    fn expert_group_ep_sidecar_rejects_pin_but_accepts_replicate() {
        let spec = expert_spec(ExpertParallelism::ExpertParallel);
        let pin = WeightEntry::layer(
            "experts.scale",
            0,
            vec![4, 4],
            DType::F16,
            ShardPolicy::Pin(PinTarget::Output),
        );
        assert!(validate_sidecar_policy(&spec, &pin, "sidecar[0]").is_err());

        let replicate = WeightEntry::layer(
            "experts.scale",
            0,
            vec![4, 4],
            DType::F16,
            ShardPolicy::Replicate,
        );
        assert!(validate_sidecar_policy(&spec, &replicate, "sidecar[0]").is_ok());
    }

    #[test]
    fn expert_group_ep_sidecar_rejects_tied_but_accepts_replicate() {
        let spec = expert_spec(ExpertParallelism::ExpertParallel);
        let tied = WeightEntry::layer(
            "experts.scale",
            0,
            vec![4, 4],
            DType::F16,
            ShardPolicy::Tied {
                source: "source".into(),
            },
        );
        assert!(validate_sidecar_policy(&spec, &tied, "sidecar[0]").is_err());

        let replicate = WeightEntry::layer(
            "experts.scale",
            0,
            vec![4, 4],
            DType::F16,
            ShardPolicy::Replicate,
        );
        assert!(validate_sidecar_policy(&spec, &replicate, "sidecar[0]").is_ok());
    }

    #[test]
    fn expert_group_rejects_non_three_dimensional_or_zero_dim_packed_sources() {
        for shape in [vec![4], vec![4, 4], vec![4, 4, 4, 4], vec![4, 0, 4]] {
            let mut manifest = expert_manifest(Some(0), 4, ExpertParallelism::Single);
            manifest[1].logical_shape = shape;
            let err = validate_expert_group_specs(
                &[expert_spec(ExpertParallelism::Single)],
                &manifest,
                1,
            )
            .unwrap_err();
            assert!(err.contains("experts.gate_up"));
            assert!(err.contains("3D") || err.contains("zero"));
        }

        let tp = DeviceMesh::rect(&[(DimKind::Tp, 2)]);
        for shape in [vec![4, 512], vec![4, 512, 8, 1], vec![4, 0, 8]] {
            let manifest = vec![WeightEntry::layer(
                "experts",
                0,
                shape,
                DType::F16,
                ShardPolicy::ExpertTensorSharded {
                    n_experts: 4,
                    inner: Box::new(ShardPolicy::ColumnShard { axis: 1 }),
                },
            )];
            assert!(validate_manifest(&manifest, &tp).is_err());
        }

        let invalid_inner = vec![WeightEntry::layer(
            "experts",
            0,
            vec![4, 512, 8],
            DType::F16,
            ShardPolicy::ExpertTensorSharded {
                n_experts: 4,
                inner: Box::new(ShardPolicy::Replicate),
            },
        )];
        assert!(validate_manifest(&invalid_inner, &tp).is_err());
    }
}
