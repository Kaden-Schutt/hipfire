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
//!
//! ## Static expert policy vs effective projection
//!
//! A static manifest entry's [`ShardPolicy`] is authored once and describes
//! the *logical source layout* — MoE projections are typically declared as
//! [`ShardPolicy::ExpertSharded`] packed surrogates — which cannot
//! simultaneously describe Single, TP, and EP materialization.
//! [`resolve_expert_manifest_for_policy`] projects only the projection sources
//! explicitly claimed by an [`ExpertSourceLayout`] to the effective resident
//! placement for one exact execution policy, on a **clone** of the full static
//! manifest: the clone is strictly validated and resolved, its residual
//! per-weight collectives derived, and then it is dropped. The original
//! manifest and specs are never mutated. Projection-path TP eligibility
//! additionally enforces role-dimension divisibility by the exact TP rank
//! count and local slice width `% 256 == 0` for every rank count including
//! Tp=1 (where the frozen strict validator's `tp > 1` gate does not apply);
//! the strict validator and resolver semantics are unchanged.
//!
//! ## Logical source ownership vs loader fusion
//!
//! [`ExpertSourceLayout`] claims are *logical* source ownership: every
//! projection source (gate/up/down, per expert or packed) the group needs.
//! Loader fusion is *materialization* — a loader may fuse gate+up into one
//! runtime blob, but the manifest must still claim every logical source (a
//! fused runtime blob is declared `PackedFused`, separate logical gate/up/down
//! sources declare `PackedSeparate`). No source may be hidden: unclaimed
//! entries keep their static policy and remain visible in the residual
//! per-weight collective schedule.
//!
//! ## Collective authority
//!
//! A declared group's single post-combine collective derives solely from its
//! [`ExpertParallelism`]; projected source policies are placement evidence and
//! never become per-weight collective authority. The undeclared-family legacy
//! schedule ([`layer_collectives`]) remains policy-derived for manifests
//! without expert groups.

use crate::moe_plan::{MoEExecutionKind, MoEExecutionPolicy};
use crate::tp_shard::ExpertAssign;
use hipfire_hardware::{CollectiveHint, DeviceMesh, DimKind};
use rdna_compute::DType;

/// Derive the cross-device collective an op's output requires **from its weight
/// [`ShardPolicy`]** — the mini-partitioner that makes sharding a *single*
/// source of truth (declared once in the manifest) instead of a policy in the
/// manifest AND a hand-written hint at lowering (which risks a silent
/// forgotten-reduce). Row-parallel dense → all-reduce over `Tp`; expert-sharded
/// MoE → all-reduce over `Ep`. Column/replicate/pin/etc. need no output reduce.
/// This is the **undeclared-family** schedule: it maps every collective-bearing
/// policy, so [`layer_collectives`] stays complete for manifests that do not
/// declare expert groups. Declared expert groups instead use
/// [`layer_collectives_for_declared_groups`], which excludes their claimed
/// projections from this map.
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
/// mesh group at run time. This legacy schedule covers every collective-bearing
/// entry; manifests that declare expert groups should use
/// [`layer_collectives_for_declared_groups`] instead. PP `BandXfer`
/// (inter-layer) comes from [`hipfire_hardware::DeviceMesh::band_xfer_after`],
/// not this per-op map.
pub fn layer_collectives(manifest: &[WeightEntry]) -> Vec<(usize, CollectiveHint)> {
    manifest
        .iter()
        .filter_map(|e| Some((e.layer?, collective_for_policy(&e.policy)?)))
        .collect()
}

/// The per-layer all-reduce schedule for a manifest whose expert groups are
/// **declared** via [`ExpertGroupSpec`]s. Validates the declarations (shape,
/// group/layer identity, declared-source ownership, manifest references),
/// then retains the policy-derived schedule for every entry
/// except the expert projection sources claimed by the declared groups —
/// those are owned by the group plan's single post-combine collective, so they
/// must not schedule per-weight reduces here. Unclaimed entries keep their
/// [`collective_for_policy`] scheduling unchanged.
pub fn layer_collectives_for_declared_groups(
    manifest: &[WeightEntry],
    specs: &[ExpertGroupSpec],
    group_size: usize,
) -> Result<Vec<(usize, CollectiveHint)>, String> {
    validate_expert_group_specs(specs, manifest, group_size)?;
    let mut claimed: std::collections::HashSet<(&str, Option<usize>)> =
        std::collections::HashSet::new();
    for spec in specs {
        for name in expert_projection_sources(spec) {
            claimed.insert((name, spec.layer));
        }
    }
    Ok(manifest
        .iter()
        .filter_map(|e| {
            let layer = e.layer?;
            if claimed.contains(&(e.name.as_str(), e.layer)) {
                return None;
            }
            Some((layer, collective_for_policy(&e.policy)?))
        })
        .collect())
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
                if e.logical_shape.len() != 3 || e.logical_shape.contains(&0) {
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
                if tp > 1 && !(d % tp == 0 && (d / tp).is_multiple_of(256)) {
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
/// both post-combine ordering and the parallelism-derived collective axis.
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

/// Manifest-owned canonical execution identity of an expert group. The typed
/// dispatch `ExpertExecutionPlan` maps onto this enum exhaustively at lowering
/// (moe_plan's `From` impl); the manifest declares which members its group
/// admits. Canonical labels (`indexed_quantized`, `grouped_quantized`,
/// `per_expert_fallback`) are stable contract labels used by every diagnostic
/// and display path — never re-derived by string parsing.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum ExpertExecutionIdentity {
    IndexedQuantized,
    GroupedQuantized,
    PerExpertFallback,
}

impl ExpertExecutionIdentity {
    /// The stable canonical contract label of this identity.
    pub const fn canonical_label(self) -> &'static str {
        match self {
            Self::IndexedQuantized => "indexed_quantized",
            Self::GroupedQuantized => "grouped_quantized",
            Self::PerExpertFallback => "per_expert_fallback",
        }
    }
}

impl std::fmt::Display for ExpertExecutionIdentity {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.canonical_label())
    }
}

/// Architecture-declared description of one logical expert group.
///
/// `router` is a stable manifest weight reference consumed by the existing
/// router machinery; `router_identity` and `allowed_executions` are canonical
/// semantic identities (not manifest references) naming the routing algorithm
/// and the execution plans the group admits. Keeping these symbolic avoids
/// coupling this CPU-only manifest to feature-gated GPU or pager types.
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
    /// Manifest weight reference consumed by the router machinery (e.g. `mlp.gate`).
    pub router: String,
    /// Canonical routing algorithm identity: `softmax_topk`, `sigmoid_topk`,
    /// `bias_aware_topk`, `hash`, or `precomputed`.
    pub router_identity: String,
    /// Non-empty, duplicate-free set of execution identities this group
    /// admits. Lowering accepts exact typed membership only; the CPU
    /// fallback is never declared here (it lives outside lowering).
    pub allowed_executions: Vec<ExpertExecutionIdentity>,
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
    pub group_size: usize,
    pub parallelism: ExpertParallelism,
    pub assignment: ExpertAssign,
    pub experts: Vec<ExpertPlacement>,
    pub source_layout: ExpertSourceLayout,
    pub resources: ExpertResourceRequirements,
    /// Manifest weight reference consumed by the router machinery (e.g. `mlp.gate`).
    pub router: String,
    /// Canonical routing algorithm identity: `softmax_topk`, `sigmoid_topk`,
    /// `bias_aware_topk`, `hash`, or `precomputed`.
    pub router_identity: String,
    /// The execution identities the group admits (mirrors the spec's
    /// declaration; membership is checked at lowering, never by label parsing).
    pub allowed_executions: Vec<ExpertExecutionIdentity>,
    pub collective: Option<ExpertPostCombineAllReduce>,
}

fn expert_context(spec: &ExpertGroupSpec) -> String {
    format!("expert group '{}' layer {:?}", spec.group, spec.layer)
}

/// The gate/up/down expert projection references claimed by a declared group
/// across every [`ExpertSourceLayout`] variant — deliberately **not** the
/// router or sidecars. Used for declared-source ownership (each `(source,
/// layer)` may be claimed once) and for contextual schedule exclusion (the
/// declared-group collective schedule drops exactly these projections).
fn expert_projection_sources(spec: &ExpertGroupSpec) -> Vec<&str> {
    match &spec.source_layout {
        ExpertSourceLayout::PackedFused { gate_up, down, .. } => {
            vec![gate_up.as_str(), down.as_str()]
        }
        ExpertSourceLayout::PackedSeparate { gate, up, down, .. } => {
            vec![gate.as_str(), up.as_str(), down.as_str()]
        }
        ExpertSourceLayout::PerExpertFused { gate_up, down, .. } => gate_up
            .iter()
            .chain(down.iter())
            .map(String::as_str)
            .collect(),
        ExpertSourceLayout::PerExpertSeparate { gate, up, down, .. } => gate
            .iter()
            .chain(up.iter())
            .chain(down.iter())
            .map(String::as_str)
            .collect(),
    }
}

/// Validate one expert group's structural metadata and non-empty semantic
/// identities (group size, expert count, resources, group/router_identity
/// labels, and the non-empty duplicate-free allowed-execution admission set)
/// without resolving manifest references. Typed semantic identity matching
/// against the actual `RouterSelection` / `ExpertExecutionPlan` is moe_plan's
/// concern (Task 3), not manifest resolution.
fn validate_expert_group_metadata(spec: &ExpertGroupSpec, group_size: usize) -> Result<(), String> {
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
                && !spec.n_experts.is_multiple_of(group_size)
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
        return Err(format!("{context}: group identity '' is invalid"));
    }
    if spec.router_identity.is_empty() {
        return Err(format!("{context}: router identity '' is invalid"));
    }
    // The allowed-execution admission set must be non-empty and
    // duplicate-free; the duplicate is reported deterministically with its
    // canonical label. Allocation-free declaration-order scan: each identity
    // is checked against every preceding one, so the FIRST duplicate in
    // declaration order is always the one reported.
    if spec.allowed_executions.is_empty() {
        return Err(format!("{context}: allowed execution identities is empty"));
    }
    for (idx, identity) in spec.allowed_executions.iter().enumerate() {
        if spec.allowed_executions[..idx].contains(identity) {
            return Err(format!(
                "{context}: allowed execution identities contains duplicate '{}'",
                identity.canonical_label()
            ));
        }
    }
    Ok(())
}

/// Validate one expert group without checking manifest references.
pub fn validate_expert_group_spec(spec: &ExpertGroupSpec, group_size: usize) -> Result<(), String> {
    validate_expert_group_metadata(spec, group_size)
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

/// A *direct* collective-bearing placement policy on a claimed expert
/// projection competes with the group's single post-combine collective and
/// must be rejected before any generic incompatible-policy error. Top-level
/// `ExpertSharded` / `ExpertTensorSharded` placement policies are **not**
/// competing: they describe placement (and their inner/assigned collectives
/// are exactly what the group plan schedules once).
fn is_competing_per_weight_collective(policy: &ShardPolicy) -> bool {
    match policy {
        ShardPolicy::ExpertSharded { .. } | ShardPolicy::ExpertTensorSharded { .. } => false,
        policy => collective_for_policy(policy).is_some(),
    }
}

fn validate_source_policy(
    spec: &ExpertGroupSpec,
    entry: &WeightEntry,
    label: &str,
    role: ProjectionRole,
) -> Result<(), String> {
    let context = expert_context(spec);
    if is_competing_per_weight_collective(&entry.policy) {
        return Err(format!(
            "{context}: {label} reference '{}' has a competing per-weight collective (policy {:?}); expert-group collectives are group-level",
            entry.name, entry.policy
        ));
    }
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
    if entry.logical_shape.len() != 3 || entry.logical_shape.contains(&0) {
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
    // Zero dimensions are invalid at any rank (most importantly rank-two
    // `[n_experts, 0]`) and are rejected before the experts-first check.
    if router.logical_shape.contains(&0) {
        return Err(format!(
            "{}: router reference '{}' has logical_shape {:?}; router dimensions must be nonzero",
            expert_context(spec),
            router.name,
            router.logical_shape
        ));
    }
    // Routers are experts-first: rank one `[n_experts]`, rank two
    // `[n_experts, input_dim]`. Validation checks the FIRST dimension exactly
    // and rejects the transposed `[input_dim, n_experts]` form.
    if router.logical_shape.first().copied() != Some(spec.n_experts) {
        return Err(format!(
            "{}: router reference '{}' has logical_shape {:?}; first dimension must equal n_experts={}",
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

/// Deterministic declared-source ownership: every `(source, layer)` pair may
/// be claimed by exactly one declared group — and by at most one projection
/// within a single group (a group repeating a source is malformed on its own).
/// Specs are traversed in declaration order so the first claimant is stable,
/// and a cross-group conflict error names the source plus the first and
/// second groups. Routers and sidecars remain excluded: they are not expert
/// projections.
fn validate_declared_source_ownership(specs: &[ExpertGroupSpec]) -> Result<(), String> {
    let mut owners: std::collections::HashMap<(&str, Option<usize>), &str> =
        std::collections::HashMap::new();
    for spec in specs {
        for name in expert_projection_sources(spec) {
            let key = (name, spec.layer);
            match owners.get(&key) {
                None => {
                    owners.insert(key, spec.group.as_str());
                }
                Some(first) if *first == spec.group => {
                    return Err(format!(
                        "{}: repeats projection source '{}' within the same group",
                        expert_context(spec),
                        name
                    ));
                }
                Some(first) => {
                    return Err(format!(
                        "expert group '{}' layer {:?}: projection source '{}' at layer {:?} is already claimed by expert group '{}' (declared first)",
                        spec.group, spec.layer, name, spec.layer, first
                    ));
                }
            }
        }
    }
    Ok(())
}

/// The projection source claims of one group with their diagnostic role
/// labels, in declaration order. Mirrors [`expert_projection_sources`] but
/// carries the role (e.g. `source gate_up`, `source down[2]`) so ownership
/// diagnostics can name exactly which claim conflicts. Empty names are
/// skipped: they are refused by the reference validators with their own
/// 'reference is invalid' diagnostic.
fn projection_claims_with_roles(spec: &ExpertGroupSpec) -> Vec<(&str, String)> {
    let mut claims = Vec::new();
    match &spec.source_layout {
        ExpertSourceLayout::PackedFused { gate_up, down, .. } => {
            claims.push((gate_up.as_str(), "source gate_up".to_owned()));
            claims.push((down.as_str(), "source down".to_owned()));
        }
        ExpertSourceLayout::PackedSeparate { gate, up, down, .. } => {
            claims.push((gate.as_str(), "source gate".to_owned()));
            claims.push((up.as_str(), "source up".to_owned()));
            claims.push((down.as_str(), "source down".to_owned()));
        }
        ExpertSourceLayout::PerExpertFused { gate_up, down, .. } => {
            for (idx, name) in gate_up.iter().enumerate() {
                claims.push((name.as_str(), format!("source gate_up[{idx}]")));
            }
            for (idx, name) in down.iter().enumerate() {
                claims.push((name.as_str(), format!("source down[{idx}]")));
            }
        }
        ExpertSourceLayout::PerExpertSeparate { gate, up, down, .. } => {
            for (idx, name) in gate.iter().enumerate() {
                claims.push((name.as_str(), format!("source gate[{idx}]")));
            }
            for (idx, name) in up.iter().enumerate() {
                claims.push((name.as_str(), format!("source up[{idx}]")));
            }
            for (idx, name) in down.iter().enumerate() {
                claims.push((name.as_str(), format!("source down[{idx}]")));
            }
        }
    }
    claims.retain(|(name, _)| !name.is_empty());
    claims
}

/// A `(name, layer)`-keyed router/sidecar reference with its diagnostic role
/// label (`router`, `sidecar[i]`) and owning group.
type ReferenceClaim<'a> = ((&'a str, Option<usize>), String, &'a str);

/// Every router and sidecar reference of every declared group, keyed by
/// `(name, layer)` with its diagnostic role label (`router`, `sidecar[i]`) and
/// owning group, in declaration order. Repeated references (two groups sharing
/// one router entry) are legal and simply appear twice; the disjointness check
/// below only reports a conflict when a projection claim matches ANY of them.
fn router_and_sidecar_references(specs: &[ExpertGroupSpec]) -> Vec<ReferenceClaim<'_>> {
    let mut references = Vec::new();
    for spec in specs {
        references.push((
            (spec.router.as_str(), spec.layer),
            "router".to_owned(),
            spec.group.as_str(),
        ));
        let sidecars = match &spec.source_layout {
            ExpertSourceLayout::PackedFused { sidecars, .. }
            | ExpertSourceLayout::PackedSeparate { sidecars, .. }
            | ExpertSourceLayout::PerExpertFused { sidecars, .. }
            | ExpertSourceLayout::PerExpertSeparate { sidecars, .. } => sidecars.as_slice(),
        };
        for (idx, name) in sidecars.iter().enumerate() {
            references.push((
                (name.as_str(), spec.layer),
                format!("sidecar[{idx}]"),
                spec.group.as_str(),
            ));
        }
    }
    references
}

/// Projection ownership extends to router and sidecar references: an entry
/// claimed as an expert projection source by any group must never also be a
/// router or sidecar reference of ANY group (same-group or cross-group). The
/// check is projection-path-only — the strict validator does not run it, so
/// [`validate_expert_group_specs`] behavior stays equivalent. Traversal is in
/// declaration order (groups, then claims, then references) so the reported
/// conflict is deterministic; the diagnostic names the projection role/group
/// and the router/sidecar role/group. Router and sidecar entries are never
/// rewritten: the projection pass only touches claimed sources, and this
/// check guarantees no claimed source is also a router/sidecar entry.
fn validate_projection_sources_disjoint_from_references(
    specs: &[ExpertGroupSpec],
) -> Result<(), String> {
    let references = router_and_sidecar_references(specs);
    for spec in specs {
        for (name, role) in projection_claims_with_roles(spec) {
            if let Some((_, ref_role, ref_group)) = references
                .iter()
                .find(|((ref_name, ref_layer), _, _)| *ref_name == name && *ref_layer == spec.layer)
            {
                return Err(format!(
                    "expert group '{}' layer {:?}: {role} projection source '{name}' at layer {:?} is also referenced as {ref_role} by expert group '{ref_group}'",
                    spec.group, spec.layer, spec.layer
                ));
            }
        }
    }
    Ok(())
}

/// Static-policy-neutral prefix of the strict expert-group validation path:
/// the duplicate manifest `(name, layer)` identity check, per-group structural
/// metadata / non-empty semantic identities, and unique group/layer identity.
/// This prefix inspects no [`ShardPolicy`], so the policy-aware projection
/// entry point can run it on the static manifest BEFORE projection ownership
/// and rewriting — reporting the same diagnostics in the same order as the
/// strict path. The strict validator ([`validate_expert_group_specs`]) reuses
/// it verbatim so the two paths never diverge.
fn validate_expert_group_specs_prefix(
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
    // Stage 1: structural metadata, non-empty identities, unique group/layer.
    let mut identities = std::collections::HashSet::new();
    for spec in specs {
        validate_expert_group_metadata(spec, group_size)?;
        let context = expert_context(spec);
        if !identities.insert((&spec.group, spec.layer)) {
            return Err(format!("{context}: duplicate group/layer identity"));
        }
    }
    Ok(())
}

/// Validate all architecture-declared expert groups against the weight
/// manifest. Validation runs in three stages:
/// 1. the static-policy-neutral prefix
///    ([`validate_expert_group_specs_prefix`]): duplicate manifest identity,
///    per-group structural metadata and non-empty semantic identities, and
///    unique group/layer identity;
/// 2. declared-source ownership (each `(source, layer)` claimed once);
/// 3. manifest references, policies, router tensor, and sidecars
///    ([`validate_expert_group_references`]).
///
/// The duplicate `(source, layer)` claim is reported deterministically because
/// ownership runs before manifest-reference validation, which would otherwise error
/// first. Typed semantic identity matching against the actual
/// `RouterSelection` / `ExpertExecutionPlan` is moe_plan's concern (Task 3),
/// not manifest resolution.
pub fn validate_expert_group_specs(
    specs: &[ExpertGroupSpec],
    manifest: &[WeightEntry],
    group_size: usize,
) -> Result<(), String> {
    // Stage 1: static-policy-neutral prefix (duplicate manifest identity,
    // group metadata, unique group/layer identity).
    validate_expert_group_specs_prefix(specs, manifest, group_size)?;
    // Stage 2: declared-source ownership — each `(source, layer)` claimed once.
    validate_declared_source_ownership(specs)?;
    // Stage 3: manifest references, policies, router tensor, and sidecars.
    for spec in specs {
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
                for (owner, slot) in next_slot.iter_mut().enumerate() {
                    let local_slot = *slot;
                    *slot += 1;
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
        group_size,
        parallelism: spec.parallelism,
        assignment: spec.assignment,
        experts,
        source_layout: spec.source_layout.clone(),
        resources: spec.resources,
        router: spec.router.clone(),
        router_identity: spec.router_identity.clone(),
        allowed_executions: spec.allowed_executions.clone(),
        collective: post_combine_for_parallelism(spec.parallelism),
    })
}

/// The complete policy-aware resolution of a static expert manifest for one
/// exact execution policy.
///
/// `plans` are the resolved expert-group plans (one per declared spec); each
/// group's single post-combine collective derives solely from its
/// [`ExpertParallelism`] (see [`ExpertGroupPlan::collective`]). The projected
/// source policies are placement evidence, never extra per-weight collective
/// authority.
///
/// `layer_collectives` is the residual per-weight per-layer schedule derived
/// from the *projected* full manifest with every claimed projection source
/// excluded exactly once. Unclaimed entries (dense weights, routers, sidecars,
/// unclaimed expert sources) keep their static policy and remain visible here.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct ExpertManifestResolution {
    /// Resolved expert-group plans, one per declared spec, in declaration order.
    pub plans: Vec<ExpertGroupPlan>,
    /// Residual per-weight per-layer collectives from the projected full
    /// manifest, excluding claimed expert projection sources exactly once.
    pub layer_collectives: Vec<(usize, CollectiveHint)>,
}

/// Map an execution-policy kind to the manifest's group parallelism.
fn parallelism_for_kind(kind: MoEExecutionKind) -> ExpertParallelism {
    match kind {
        MoEExecutionKind::Single => ExpertParallelism::Single,
        MoEExecutionKind::Tp => ExpertParallelism::TensorParallel,
        MoEExecutionKind::Ep => ExpertParallelism::ExpertParallel,
    }
}

/// The effective resident placement one claimed packed expert projection must
/// carry under an exact execution policy, by logical role: gate/up projections
/// are column-sharded (axis 1) and down projections row-sharded (axis 2) under
/// TP; every packed projection replicates under Single and keeps the group's
/// exact `ExpertSharded` declaration under EP.
fn effective_projection_policy(
    spec: &ExpertGroupSpec,
    role: ProjectionRole,
    kind: MoEExecutionKind,
) -> ShardPolicy {
    match kind {
        MoEExecutionKind::Single => ShardPolicy::Replicate,
        MoEExecutionKind::Tp => ShardPolicy::ExpertTensorSharded {
            n_experts: spec.n_experts,
            inner: Box::new(match role {
                ProjectionRole::GateUp => ShardPolicy::ColumnShard { axis: 1 },
                ProjectionRole::Down => ShardPolicy::RowShard { axis: 2 },
            }),
        },
        MoEExecutionKind::Ep => ShardPolicy::ExpertSharded {
            n_experts: spec.n_experts,
            assign: spec.assignment,
        },
    }
}

/// Find the single manifest entry a claimed projection source resolves to in
/// the projection clone, mirroring [`validate_manifest_reference`]'s missing /
/// ambiguous / scope diagnostics so the projected clone is never silently
/// patched at the wrong (name, layer).
fn find_projection_entry_mut<'a>(
    spec: &ExpertGroupSpec,
    manifest: &'a mut [WeightEntry],
    label: &str,
    name: &str,
) -> Result<&'a mut WeightEntry, String> {
    let context = expert_context(spec);
    if name.is_empty() {
        return Err(format!("{context}: {label} reference '' is invalid"));
    }
    let matches: Vec<usize> = manifest
        .iter()
        .enumerate()
        .filter(|(_, entry)| entry.name == name && entry.layer == spec.layer)
        .map(|(idx, _)| idx)
        .collect();
    match matches.as_slice() {
        [] => Err(format!(
            "{context}: {label} reference '{name}' not found in manifest scope"
        )),
        [idx] => Ok(&mut manifest[*idx]),
        _ => Err(format!(
            "{context}: {label} reference '{name}' is ambiguous in manifest scope"
        )),
    }
}

/// Project one explicitly claimed packed expert projection source to its
/// effective resident placement. Only a static [`ShardPolicy::ExpertSharded`]
/// surrogate with the group's exact `n_experts` + assignment, or an
/// already-correct effective policy, is eligible. Wrong counts, assignments,
/// axes, nested policies, plain competing shards (including a direct
/// `RowShard`), and unsupported static `Pin`/`Tied` declarations are refused
/// here — before any strict validation or collective scheduling runs.
fn project_claimed_packed_source(
    spec: &ExpertGroupSpec,
    manifest: &mut [WeightEntry],
    label: &str,
    name: &str,
    role: ProjectionRole,
    kind: MoEExecutionKind,
) -> Result<(), String> {
    let context = expert_context(spec);
    let effective = effective_projection_policy(spec, role, kind);
    let entry = find_projection_entry_mut(spec, manifest, label, name)?;
    match &entry.policy {
        // Static `ExpertSharded` packed surrogate: projectable only with the
        // declaring group's exact expert count and assignment.
        ShardPolicy::ExpertSharded { n_experts, assign } => {
            if *n_experts != spec.n_experts {
                return Err(format!(
                    "{context}: {label} reference '{}' static ExpertSharded n_experts={n_experts} does not match spec n_experts={}",
                    entry.name, spec.n_experts
                ));
            }
            if *assign != spec.assignment {
                return Err(format!(
                    "{context}: {label} reference '{}' static ExpertSharded assignment {assign:?} does not match spec assignment {:?}",
                    entry.name, spec.assignment
                ));
            }
        }
        // Already-correct effective policies remain untouched; the strict
        // per-group validation below re-checks them unchanged.
        _ if entry.policy == effective => {}
        // Everything else is not projectable for this policy kind: wrong
        // axes/counts/assign, nested policies, plain competing shards, and
        // unsupported static Pin/Tied declarations all fail closed here.
        _ => {
            return Err(format!(
                "{context}: {label} reference '{}' static policy {:?} is not projectable to effective {:?}",
                entry.name, entry.policy, effective
            ));
        }
    }
    entry.policy = effective;
    Ok(())
}

/// Projection-path TP eligibility: every claimed packed projection source of
/// a TP policy must have its role dimension divisible by the exact TP rank
/// count AND a local slice width `% 256 == 0` — for EVERY rank count,
/// including Tp=1, where the frozen strict [`validate_manifest`] gate
/// (`tp > 1`) does not apply. Gate/up role dimension is axis 1 (the
/// projection width: `2*inter` for a fused gate_up, `inter` for separate
/// gate/up), down is axis 2 (`inter`). The check runs on the projected clone
/// after shape safety is established and before resolution; diagnostics name
/// the role, source, and group deterministically in declaration order.
/// Single/EP policies never reach this check (their effective placements do
/// not slice), and the frozen strict resolver / `validate_manifest` semantics
/// are unchanged.
fn validate_projected_tp_role_dimensions(
    specs: &[ExpertGroupSpec],
    manifest: &[WeightEntry],
    tp_ranks: usize,
) -> Result<(), String> {
    for spec in specs {
        // Only packed layouts project under TP; PerExpert layouts are refused
        // before this check, so they contribute no claims here.
        let claims: Vec<(&str, &str, usize)> = match &spec.source_layout {
            ExpertSourceLayout::PackedFused { gate_up, down, .. } => vec![
                (gate_up.as_str(), "source gate_up", 1),
                (down.as_str(), "source down", 2),
            ],
            ExpertSourceLayout::PackedSeparate { gate, up, down, .. } => vec![
                (gate.as_str(), "source gate", 1),
                (up.as_str(), "source up", 1),
                (down.as_str(), "source down", 2),
            ],
            ExpertSourceLayout::PerExpertFused { .. }
            | ExpertSourceLayout::PerExpertSeparate { .. } => Vec::new(),
        };
        for (name, label, axis) in claims {
            if name.is_empty() {
                continue;
            }
            let context = expert_context(spec);
            // Projection ownership already resolved the claim; the projected
            // clone carries the entry with the effective TP policy.
            let Some(entry) = manifest
                .iter()
                .find(|entry| entry.name == name && entry.layer == spec.layer)
            else {
                return Err(format!(
                    "{context}: {label} reference '{name}' not found in manifest scope"
                ));
            };
            let dim = entry.logical_shape.get(axis).copied().unwrap_or(0);
            if dim % tp_ranks != 0 || !(dim / tp_ranks).is_multiple_of(256) {
                return Err(format!(
                    "{context}: {label} reference '{}' projected TP role dim {dim} (axis {axis}) not divisible by Tp={tp_ranks} or local slice {} not a multiple of 256",
                    entry.name, dim / tp_ranks
                ));
            }
        }
    }
    Ok(())
}

/// Resolve a static expert manifest for one exact execution policy.
///
/// Static expert `ShardPolicy`s describe the logical source layout and cannot
/// describe Single, TP, and EP materialization simultaneously (see the module
/// docs). This entry point clones the FULL static manifest, rewrites only the
/// explicitly claimed expert projection sources to their effective resident
/// placement for `policy`, runs the strict existing full-manifest and
/// group/reference/source/policy validation against the exact policy mesh,
/// resolves the group plans, derives the residual per-weight layer collectives
/// with claimed sources excluded exactly once, and drops the clone. The
/// original manifest and specs remain equality-identical.
///
/// Fail-closed: every spec's [`ExpertParallelism`] must agree exactly with
/// `policy.kind()`; missing / ambiguous / wrong-layer / repeated / colliding
/// sources, wrong expert counts or assignments, malformed shapes, wrong
/// axes/ranks, TP divisibility and local-256 failures, direct competing
/// `RowShard`s, unsupported static `Pin`/`Tied`, PerExpert layouts under
/// TP/EP, and unrelated invalid manifest entries are all refused. Under TP,
/// every claimed packed source's role dimension (gate/up axis 1, down axis 2)
/// must divide by the exact TP rank count with a local slice width `% 256 ==
/// 0` for EVERY rank count, including Tp=1 — projection-path eligibility that
/// does not change the frozen strict validator. A claimed projection source
/// may never also be any group's router or sidecar reference (same-group or
/// cross-group): the diagnostic names the projection role/group and the
/// router/sidecar role/group. Router, sidecar, and unrelated entries are never
/// projected.
///
/// Diagnostic precedence matches the strict path: the static-policy-neutral
/// prefix (duplicate manifest identity, group metadata, unique group/layer
/// identity) runs before projection ownership, which runs before projection
/// eligibility, full-manifest validation, the exact-TP role-dimension
/// eligibility, and the unchanged complete group validator on the clone.
/// Existing strict resolution ([`resolve_expert_group_plans`]) remains
/// unchanged for Qwen-like control callers.
pub fn resolve_expert_manifest_for_policy(
    specs: &[ExpertGroupSpec],
    static_manifest: &[WeightEntry],
    policy: &MoEExecutionPolicy,
) -> Result<ExpertManifestResolution, String> {
    let kind = policy.kind();
    // 1. Every declared group's parallelism must agree exactly with the
    //    execution policy kind; a spec/kind mismatch is refused before any
    //    projection or validation runs.
    for spec in specs {
        if spec.parallelism != parallelism_for_kind(kind) {
            return Err(format!(
                "{}: parallelism {:?} does not match execution policy kind {kind:?}",
                expert_context(spec),
                spec.parallelism
            ));
        }
    }
    // 2. Static-policy-neutral prefix, shared verbatim with the strict path:
    //    duplicate manifest identity, group metadata, and unique group/layer
    //    identity all report before projection ownership or rewriting can
    //    mask them.
    let group_size = policy.rank_count();
    validate_expert_group_specs_prefix(specs, static_manifest, group_size)?;
    // 3. Projection ownership stays the deterministic authority: each
    //    `(source, layer)` may be claimed by exactly one group (and once
    //    within a group), and no claimed source may also be any group's
    //    router or sidecar reference (same-group or cross-group). Router and
    //    sidecar entries are never rewritten.
    validate_declared_source_ownership(specs)?;
    validate_projection_sources_disjoint_from_references(specs)?;
    // 4. Clone the FULL static manifest. The input manifest and specs are
    //    never mutated; the clone is dropped before return.
    let mut projected = static_manifest.to_vec();
    // 5. Project only the projection sources explicitly claimed by each
    //    ExpertSourceLayout, by logical role. PerExpert layouts are Single-only
    //    and preserved exactly; router, sidecars, and unrelated entries are
    //    never projected.
    for spec in specs {
        match &spec.source_layout {
            ExpertSourceLayout::PackedFused { gate_up, down, .. } => {
                project_claimed_packed_source(
                    spec,
                    &mut projected,
                    "source gate_up",
                    gate_up,
                    ProjectionRole::GateUp,
                    kind,
                )?;
                project_claimed_packed_source(
                    spec,
                    &mut projected,
                    "source down",
                    down,
                    ProjectionRole::Down,
                    kind,
                )?;
            }
            ExpertSourceLayout::PackedSeparate { gate, up, down, .. } => {
                project_claimed_packed_source(
                    spec,
                    &mut projected,
                    "source gate",
                    gate,
                    ProjectionRole::GateUp,
                    kind,
                )?;
                project_claimed_packed_source(
                    spec,
                    &mut projected,
                    "source up",
                    up,
                    ProjectionRole::GateUp,
                    kind,
                )?;
                project_claimed_packed_source(
                    spec,
                    &mut projected,
                    "source down",
                    down,
                    ProjectionRole::Down,
                    kind,
                )?;
            }
            ExpertSourceLayout::PerExpertFused { .. }
            | ExpertSourceLayout::PerExpertSeparate { .. } => {
                // PerExpert layouts are Single-only: their Replicate/Pin/Tied
                // sources are preserved exactly and never projected. Under
                // TP/EP the whole layout is refused.
                if kind != MoEExecutionKind::Single {
                    return Err(format!(
                        "{}: PerExpert source layout is unsupported for execution policy kind {kind:?}; global expert source placement is defined only for Single",
                        expert_context(spec)
                    ));
                }
            }
        }
    }
    // 6. Strict existing full-manifest validation on the projected clone
    //    against the exact policy mesh (TP divisibility + local-256
    //    alignment), then every existing group/reference/layer/source/policy/
    //    collective check.
    validate_manifest(&projected, policy.mesh())?;
    // 7. Projection-path TP eligibility after shape safety is established:
    //    every claimed packed source's role dimension must divide by the
    //    exact TP rank count with a local slice width % 256 == 0 — for EVERY
    //    rank count including Tp=1, where the strict validate_manifest gate
    //    (tp > 1) does not apply. Single/EP never reach this check.
    if kind == MoEExecutionKind::Tp {
        validate_projected_tp_role_dimensions(specs, &projected, group_size)?;
    }
    validate_expert_group_specs(specs, &projected, group_size)?;
    // 8. Private resolution. The group collective derives solely from
    //    ExpertParallelism (post_combine_for_parallelism); the projected
    //    source policies are placement evidence, never extra authority.
    let plans = specs
        .iter()
        .map(|spec| resolve_expert_group_plan_unchecked(spec, group_size))
        .collect::<Result<Vec<_>, _>>()?;
    // 9. Residual per-weight layer collectives from the projected full
    //    manifest, excluding the claimed projection sources exactly once.
    let mut claimed: std::collections::HashSet<(&str, Option<usize>)> =
        std::collections::HashSet::new();
    for spec in specs {
        for name in expert_projection_sources(spec) {
            claimed.insert((name, spec.layer));
        }
    }
    let layer_collectives = projected
        .iter()
        .filter_map(|entry| {
            let layer = entry.layer?;
            if claimed.contains(&(entry.name.as_str(), entry.layer)) {
                return None;
            }
            Some((layer, collective_for_policy(&entry.policy)?))
        })
        .collect();
    Ok(ExpertManifestResolution {
        plans,
        layer_collectives,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::moe_plan::{MoEExecutionKind, MoEExecutionPolicy};

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
        // ExpertTensorSharded recurses into its inner policy.
        assert_eq!(
            collective_for_policy(&ShardPolicy::ExpertTensorSharded {
                n_experts: 8,
                inner: Box::new(ShardPolicy::RowShard { axis: 2 }),
            }),
            Some(CollectiveHint::AllReduce { kind: DimKind::Tp })
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
                // Routers are experts-first: `[n_experts, input_dim]`.
                logical_shape: vec![n_experts, 4],
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
            router_identity: "softmax_topk".into(),
            allowed_executions: vec![ExpertExecutionIdentity::IndexedQuantized],
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
        assert_eq!(plan.group_size, 2);
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
    fn expert_group_rejects_per_weight_collectives() {
        // A TP group whose down projection carries a direct RowShard schedules
        // a per-weight Tp all-reduce that competes with the group's
        // post-combine collective; resolving the declared group must reject it
        // contextually instead of silently double-scheduling.
        let mut manifest = expert_manifest(Some(0), 4, ExpertParallelism::TensorParallel);
        manifest[2].policy = ShardPolicy::RowShard { axis: 2 };
        let err = resolve_expert_group_plan(
            &expert_spec(ExpertParallelism::TensorParallel),
            &manifest,
            2,
        )
        .unwrap_err();
        assert!(err.contains("block-0"));
        assert!(err.contains("layer Some(0)"));
        assert!(err.contains("experts.down"));
        assert!(err.contains("competing per-weight collective"));
    }

    #[test]
    fn expert_group_reports_source_conflict_before_reference_error() {
        let manifest = expert_manifest(Some(0), 4, ExpertParallelism::ExpertParallel);
        let mut first = expert_spec(ExpertParallelism::ExpertParallel);
        first.group = "block-0".into();
        let mut second = expert_spec(ExpertParallelism::ExpertParallel);
        second.group = "block-1".into();
        second.router = "missing.router".into();

        // The two specs claim the same expert projections at layer Some(0);
        // declared-source ownership must report the deterministic conflict
        // before the second spec's dangling router reference is ever resolved.
        let err = resolve_expert_group_plans(&[first, second], &manifest, 2).unwrap_err();
        assert!(err.contains("experts.gate_up"));
        assert!(err.contains("block-0"));
        assert!(err.contains("block-1"));
        assert!(err.contains("declared first"));
        assert!(!err.contains("missing.router"));
    }

    #[test]
    fn expert_group_rejects_repeated_source_within_one_group() {
        let manifest = expert_manifest(Some(0), 4, ExpertParallelism::ExpertParallel);
        let mut spec = expert_spec(ExpertParallelism::ExpertParallel);
        spec.source_layout = ExpertSourceLayout::PackedSeparate {
            gate: "experts.gate_up".into(),
            up: "experts.gate_up".into(),
            down: "experts.down".into(),
            sidecars: Vec::new(),
        };

        let err = validate_expert_group_specs(&[spec], &manifest, 2).unwrap_err();
        assert!(err.contains("block-0"));
        assert!(err.contains("layer Some(0)"));
        assert!(err.contains("experts.gate_up"));
        assert!(err.contains("repeats projection source"));
    }

    #[test]
    fn expert_group_reports_deterministic_source_conflict() {
        let manifest = expert_manifest(Some(0), 4, ExpertParallelism::ExpertParallel);
        let mut first = expert_spec(ExpertParallelism::ExpertParallel);
        first.group = "block-0".into();
        let mut second = expert_spec(ExpertParallelism::ExpertParallel);
        second.group = "block-1".into();

        // Both specs claim the same expert projections at layer Some(0);
        // declaration order decides who is reported as the first owner.
        let err = validate_expert_group_specs(&[first, second], &manifest, 2).unwrap_err();
        assert!(err.contains("experts.gate_up"));
        assert!(err.contains("layer Some(0)"));
        assert!(err.contains("block-0"));
        assert!(err.contains("block-1"));
        assert!(err.contains("already claimed by expert group 'block-0' (declared first)"));

        // Reversing declaration order must name the new first owner.
        let mut a = expert_spec(ExpertParallelism::ExpertParallel);
        a.group = "block-0".into();
        let mut b = expert_spec(ExpertParallelism::ExpertParallel);
        b.group = "block-1".into();
        let err = validate_expert_group_specs(&[b, a], &manifest, 2).unwrap_err();
        assert!(err.contains("experts.gate_up"));
        assert!(err.contains("layer Some(0)"));
        assert!(err.contains("block-0"));
        assert!(err.contains("block-1"));
        assert!(err.contains("already claimed by expert group 'block-1' (declared first)"));
    }

    #[test]
    fn declared_group_schedule_excludes_only_claimed_expert_sources() {
        let mut manifest = expert_manifest(Some(0), 4, ExpertParallelism::ExpertParallel);
        manifest.push(WeightEntry::layer(
            "wo",
            0,
            vec![8, 8],
            DType::F16,
            ShardPolicy::RowShard { axis: 1 },
        ));
        let spec = expert_spec(ExpertParallelism::ExpertParallel);

        // Legacy schedule: expert policies still schedule their Ep reduces, so
        // the expert manifest keeps its collectives alongside the dense wo.
        let legacy = layer_collectives(&manifest);
        assert!(!legacy.is_empty());
        assert!(legacy
            .iter()
            .any(|(_, h)| matches!(h, CollectiveHint::AllReduce { kind: DimKind::Ep })));

        // The contextual declared-group schedule excludes only the claimed
        // expert projections and retains the unclaimed wo reduce.
        let declared = layer_collectives_for_declared_groups(&manifest, &[spec], 2).unwrap();
        assert_eq!(
            declared,
            vec![(0, CollectiveHint::AllReduce { kind: DimKind::Tp })]
        );
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
    fn expert_group_rejects_empty_router_identity() {
        let manifest = expert_manifest(Some(0), 4, ExpertParallelism::Single);
        let mut spec = expert_spec(ExpertParallelism::Single);
        spec.router_identity.clear();
        let err = validate_expert_group_specs(&[spec], &manifest, 1).unwrap_err();
        assert!(err.contains("block-0"));
        assert!(err.contains("router identity"));
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
    fn expert_group_router_invariants_are_experts_first_and_nonzero() {
        let spec = || expert_spec(ExpertParallelism::Single);
        let mut manifest = expert_manifest(Some(0), 4, ExpertParallelism::Single);

        // Rank-one `[n_experts]` is accepted.
        manifest[0].logical_shape = vec![4];
        assert!(validate_expert_group_specs(&[spec()], &manifest, 1).is_ok());

        // Rank-two `[n_experts, input_dim]` is accepted, including non-square
        // input dims (the second axis is an unvalidated input dim).
        manifest[0].logical_shape = vec![4, 5];
        assert!(validate_expert_group_specs(&[spec()], &manifest, 1).is_ok());
        manifest[0].logical_shape = vec![4, 7];
        assert!(validate_expert_group_specs(&[spec()], &manifest, 1).is_ok());

        // The transposed `[input_dim, n_experts]` form is rejected by the
        // first-axis contract.
        manifest[0].logical_shape = vec![7, 4];
        let err = validate_expert_group_specs(&[spec()], &manifest, 1).unwrap_err();
        assert!(err.contains("router"), "got: {err}");
        assert!(err.contains("logical_shape"), "got: {err}");
        assert!(err.contains("n_experts=4"), "got: {err}");
        assert!(err.contains("first dimension"), "got: {err}");

        // Rank three is not a router (rank must be 1 or 2).
        manifest[0].logical_shape = vec![4, 4, 4];
        let err = validate_expert_group_specs(&[spec()], &manifest, 1).unwrap_err();
        assert!(err.contains("router rank must be 1 or 2"), "got: {err}");

        // Zero dimensions are rejected before the experts-first check —
        // most importantly rank-two `[n_experts, 0]` must not resolve.
        manifest[0].logical_shape = vec![4, 0];
        let err = validate_expert_group_specs(&[spec()], &manifest, 1).unwrap_err();
        assert!(err.contains("router"), "got: {err}");
        assert!(err.contains("zero"), "got: {err}");
    }

    #[test]
    fn expert_group_rejects_empty_allowed_executions() {
        let manifest = expert_manifest(Some(0), 4, ExpertParallelism::Single);
        let mut spec = expert_spec(ExpertParallelism::Single);
        spec.allowed_executions.clear();
        let err = validate_expert_group_specs(&[spec], &manifest, 1).unwrap_err();
        assert!(err.contains("block-0"));
        assert!(err.contains("execution identities"));
    }

    #[test]
    fn expert_group_rejects_duplicate_allowed_executions() {
        let manifest = expert_manifest(Some(0), 4, ExpertParallelism::Single);
        let mut spec = expert_spec(ExpertParallelism::Single);
        spec.allowed_executions = vec![
            ExpertExecutionIdentity::IndexedQuantized,
            ExpertExecutionIdentity::IndexedQuantized,
        ];
        let err = validate_expert_group_specs(&[spec], &manifest, 1).unwrap_err();
        assert!(err.contains("block-0"));
        assert!(err.contains("duplicate"));
        assert!(err.contains("indexed_quantized"));

        // Deterministic ordering: the FIRST duplicate in declaration order is
        // reported, not a later one.
        spec = expert_spec(ExpertParallelism::Single);
        spec.allowed_executions = vec![
            ExpertExecutionIdentity::GroupedQuantized,
            ExpertExecutionIdentity::IndexedQuantized,
            ExpertExecutionIdentity::GroupedQuantized,
        ];
        let err = validate_expert_group_specs(&[spec], &manifest, 1).unwrap_err();
        assert!(err.contains("duplicate 'grouped_quantized'"), "got: {err}");
    }

    #[test]
    fn expert_execution_identity_canonical_labels_are_deterministic() {
        for (identity, expected) in [
            (
                ExpertExecutionIdentity::IndexedQuantized,
                "indexed_quantized",
            ),
            (
                ExpertExecutionIdentity::GroupedQuantized,
                "grouped_quantized",
            ),
            (
                ExpertExecutionIdentity::PerExpertFallback,
                "per_expert_fallback",
            ),
        ] {
            assert_eq!(identity.canonical_label(), expected);
            assert_eq!(identity.to_string(), expected);
        }
        assert_eq!(
            ExpertExecutionIdentity::IndexedQuantized,
            ExpertExecutionIdentity::IndexedQuantized
        );
        assert_ne!(
            ExpertExecutionIdentity::GroupedQuantized,
            ExpertExecutionIdentity::PerExpertFallback
        );
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
        // A repeated per-expert source is a same-group duplicate `(source,
        // layer)` claim, so declared-source ownership reports it before the
        // per-expert reference pass runs.
        assert!(err.contains("repeats projection source"));
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

    // ------------------------------------------------------------------
    // Policy-aware effective projection
    // ------------------------------------------------------------------

    /// Static manifest in the packed-surrogate fixture pattern: every expert
    /// projection is declared `ExpertSharded` with the group's exact count and
    /// assignment, the router replicated. Shapes are truthful and non-square:
    /// gate_up is `[n_experts, 2*inter, hidden]`, down is
    /// `[n_experts, hidden, inter]`.
    fn projection_manifest(
        layer: Option<usize>,
        n_experts: usize,
        inter: usize,
        hidden: usize,
        assign: ExpertAssign,
    ) -> Vec<WeightEntry> {
        vec![
            WeightEntry {
                name: "mlp.gate".into(),
                layer,
                logical_shape: vec![n_experts, hidden],
                dtype: DType::F16,
                dtype_constraint: DTypeConstraint::any_source(),
                placement: PlacementHint::Policy,
                policy: ShardPolicy::Replicate,
            },
            WeightEntry {
                name: "experts.gate_up".into(),
                layer,
                logical_shape: vec![n_experts, 2 * inter, hidden],
                dtype: DType::F16,
                dtype_constraint: DTypeConstraint::any_source(),
                placement: PlacementHint::Policy,
                policy: ShardPolicy::ExpertSharded { n_experts, assign },
            },
            WeightEntry {
                name: "experts.down".into(),
                layer,
                logical_shape: vec![n_experts, hidden, inter],
                dtype: DType::F16,
                dtype_constraint: DTypeConstraint::any_source(),
                placement: PlacementHint::Policy,
                policy: ShardPolicy::ExpertSharded { n_experts, assign },
            },
        ]
    }

    fn projection_spec(parallelism: ExpertParallelism) -> ExpertGroupSpec {
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
            router_identity: "softmax_topk".into(),
            allowed_executions: vec![ExpertExecutionIdentity::IndexedQuantized],
        }
    }

    fn projection_policy(kind: MoEExecutionKind, ranks: usize) -> MoEExecutionPolicy {
        let mesh = match kind {
            MoEExecutionKind::Single => DeviceMesh::single(),
            MoEExecutionKind::Tp => DeviceMesh::rect(&[(DimKind::Tp, ranks)]),
            MoEExecutionKind::Ep => DeviceMesh::rect(&[(DimKind::Ep, ranks)]),
        };
        MoEExecutionPolicy::new(kind, mesh).unwrap()
    }

    fn kind_parallelism(kind: MoEExecutionKind) -> ExpertParallelism {
        match kind {
            MoEExecutionKind::Single => ExpertParallelism::Single,
            MoEExecutionKind::Tp => ExpertParallelism::TensorParallel,
            MoEExecutionKind::Ep => ExpertParallelism::ExpertParallel,
        }
    }

    #[test]
    fn projection_resolves_packed_surrogate_across_single_tp_ep() {
        let static_manifest = projection_manifest(Some(0), 4, 512, 64, ExpertAssign::Stride);
        for (kind, parallelism, ranks, collective) in [
            (MoEExecutionKind::Single, ExpertParallelism::Single, 1, None),
            (
                MoEExecutionKind::Tp,
                ExpertParallelism::TensorParallel,
                2,
                Some(ExpertPostCombineAllReduce::TensorParallel),
            ),
            (
                MoEExecutionKind::Ep,
                ExpertParallelism::ExpertParallel,
                2,
                Some(ExpertPostCombineAllReduce::ExpertParallel),
            ),
        ] {
            let spec = projection_spec(parallelism);
            let resolution = resolve_expert_manifest_for_policy(
                &[spec.clone()],
                &static_manifest,
                &projection_policy(kind, ranks),
            )
            .unwrap_or_else(|err| panic!("kind {kind:?}: {err}"));
            assert_eq!(resolution.plans.len(), 1);
            let plan = &resolution.plans[0];
            assert_eq!(plan.parallelism, parallelism);
            assert_eq!(plan.group_size, ranks);
            // The sole group collective derives from ExpertParallelism; the
            // projected source policies are placement evidence and contribute
            // no per-weight collectives to the residual schedule.
            assert_eq!(plan.collective, collective);
            assert!(
                resolution.layer_collectives.is_empty(),
                "kind {kind:?}: {:?}",
                resolution.layer_collectives
            );
            // Exact effective placements.
            match parallelism {
                ExpertParallelism::Single => {
                    assert!(plan
                        .experts
                        .iter()
                        .all(|expert| expert.owner == 0 && expert.local_slot == expert.global_id));
                }
                ExpertParallelism::TensorParallel => {
                    assert_eq!(plan.experts.len(), 8);
                    for global_id in 0..4 {
                        let slots: Vec<_> = plan
                            .experts
                            .iter()
                            .filter(|expert| expert.global_id == global_id)
                            .map(|expert| (expert.owner, expert.local_slot))
                            .collect();
                        assert_eq!(slots, vec![(0, global_id), (1, global_id)]);
                    }
                }
                ExpertParallelism::ExpertParallel => {
                    assert_eq!(plan.experts.len(), 4);
                    assert_eq!(plan.experts[0].owner, 0);
                    assert_eq!(plan.experts[1].owner, 1);
                    assert_eq!(plan.experts[2].owner, 0);
                    assert_eq!(plan.experts[3].owner, 1);
                }
            }
        }
    }

    #[test]
    fn projection_role_matrix_accepts_fused_and_separate_packed_layouts() {
        for (kind, ranks) in [
            (MoEExecutionKind::Single, 1),
            (MoEExecutionKind::Tp, 2),
            (MoEExecutionKind::Ep, 2),
        ] {
            let manifest = projection_manifest(Some(0), 4, 512, 64, ExpertAssign::Stride);
            let fused = projection_spec(kind_parallelism(kind));
            let resolution = resolve_expert_manifest_for_policy(
                &[fused.clone()],
                &manifest,
                &projection_policy(kind, ranks),
            )
            .unwrap_or_else(|err| panic!("fused under {kind:?}: {err}"));
            assert_eq!(resolution.plans.len(), 1);

            // PackedSeparate: gate and up carry the column role, down the row
            // role — the same effective mapping as the fused layout.
            let mut separate = fused;
            separate.source_layout = ExpertSourceLayout::PackedSeparate {
                gate: "experts.gate".into(),
                up: "experts.up".into(),
                down: "experts.down".into(),
                sidecars: Vec::new(),
            };
            let mut separate_manifest =
                projection_manifest(Some(0), 4, 512, 64, ExpertAssign::Stride);
            // PackedSeparate claims gate/up/down as separate logical sources.
            // Each separate gate/up projection is its own width-inter tensor
            // (512), not the fused 2*inter (1024) gate_up blob; down is
            // [n_experts, hidden, inter].
            separate_manifest.retain(|entry| entry.name != "experts.gate_up");
            separate_manifest.push(WeightEntry::layer(
                "experts.gate",
                0,
                vec![4, 512, 64],
                DType::F16,
                ShardPolicy::ExpertSharded {
                    n_experts: 4,
                    assign: ExpertAssign::Stride,
                },
            ));
            separate_manifest.push(WeightEntry::layer(
                "experts.up",
                0,
                vec![4, 512, 64],
                DType::F16,
                ShardPolicy::ExpertSharded {
                    n_experts: 4,
                    assign: ExpertAssign::Stride,
                },
            ));
            let resolution = resolve_expert_manifest_for_policy(
                &[separate],
                &separate_manifest,
                &projection_policy(kind, ranks),
            )
            .unwrap_or_else(|err| panic!("separate under {kind:?}: {err}"));
            assert_eq!(resolution.plans.len(), 1);
        }

        // Already-correct effective policies remain: a static Replicate packed
        // projection resolves under Single, and a static role-correct
        // ExpertTensorSharded resolves under TP without rewriting.
        let mut replicate = projection_manifest(Some(0), 4, 512, 64, ExpertAssign::Stride);
        replicate[1].policy = ShardPolicy::Replicate;
        replicate[2].policy = ShardPolicy::Replicate;
        assert!(resolve_expert_manifest_for_policy(
            &[projection_spec(ExpertParallelism::Single)],
            &replicate,
            &projection_policy(MoEExecutionKind::Single, 1),
        )
        .is_ok());

        let mut already_tp = projection_manifest(Some(0), 4, 512, 64, ExpertAssign::Stride);
        already_tp[1].policy = ShardPolicy::ExpertTensorSharded {
            n_experts: 4,
            inner: Box::new(ShardPolicy::ColumnShard { axis: 1 }),
        };
        already_tp[2].policy = ShardPolicy::ExpertTensorSharded {
            n_experts: 4,
            inner: Box::new(ShardPolicy::RowShard { axis: 2 }),
        };
        assert!(resolve_expert_manifest_for_policy(
            &[projection_spec(ExpertParallelism::TensorParallel)],
            &already_tp,
            &projection_policy(MoEExecutionKind::Tp, 2),
        )
        .is_ok());
    }

    #[test]
    fn projection_single_per_expert_preserves_policies_and_matches_strict_resolver() {
        // Qwen-like control: per-expert sources under Single are never
        // projected; Replicate/Pin/Tied static policies are preserved exactly,
        // and the result equals the existing strict resolver on the same
        // static manifest.
        let mut manifest = projection_manifest(Some(0), 4, 512, 64, ExpertAssign::Stride);
        manifest.retain(|entry| entry.name == "mlp.gate");
        for expert in 0..4 {
            manifest.push(WeightEntry::layer(
                format!("experts.gate.{expert}"),
                0,
                vec![64, 512],
                DType::F16,
                ShardPolicy::Replicate,
            ));
            manifest.push(WeightEntry::layer(
                format!("experts.up.{expert}"),
                0,
                vec![64, 512],
                DType::F16,
                ShardPolicy::Replicate,
            ));
            manifest.push(WeightEntry::layer(
                format!("experts.down.{expert}"),
                0,
                vec![512, 64],
                DType::F16,
                ShardPolicy::Replicate,
            ));
        }
        // Pin and Tied policies are preserved (never projected, never rejected).
        if let Some(entry) = manifest
            .iter_mut()
            .find(|entry| entry.name == "experts.gate.0")
        {
            entry.policy = ShardPolicy::Pin(PinTarget::Embed);
        }
        if let Some(entry) = manifest
            .iter_mut()
            .find(|entry| entry.name == "experts.up.2")
        {
            entry.policy = ShardPolicy::Tied {
                source: "experts.up.0".into(),
            };
        }
        let mut spec = projection_spec(ExpertParallelism::Single);
        spec.source_layout = ExpertSourceLayout::PerExpertSeparate {
            gate: (0..4)
                .map(|expert| format!("experts.gate.{expert}"))
                .collect(),
            up: (0..4)
                .map(|expert| format!("experts.up.{expert}"))
                .collect(),
            down: (0..4)
                .map(|expert| format!("experts.down.{expert}"))
                .collect(),
            sidecars: Vec::new(),
        };
        let resolution = resolve_expert_manifest_for_policy(
            &[spec.clone()],
            &manifest,
            &projection_policy(MoEExecutionKind::Single, 1),
        )
        .unwrap();
        let control = resolve_expert_group_plans(&[spec], &manifest, 1).unwrap();
        assert_eq!(resolution.plans, control);
        assert_eq!(resolution.plans[0].collective, None);
        assert!(resolution.layer_collectives.is_empty());
    }

    #[test]
    fn projection_rejects_per_expert_under_tp_and_ep() {
        let mut manifest = projection_manifest(Some(0), 4, 512, 64, ExpertAssign::Stride);
        manifest.retain(|entry| entry.name == "mlp.gate");
        for expert in 0..4 {
            manifest.push(WeightEntry::layer(
                format!("experts.gate_up.{expert}"),
                0,
                vec![64, 512],
                DType::F16,
                ShardPolicy::Replicate,
            ));
            manifest.push(WeightEntry::layer(
                format!("experts.down.{expert}"),
                0,
                vec![512, 64],
                DType::F16,
                ShardPolicy::Replicate,
            ));
        }
        for (kind, parallelism) in [
            (MoEExecutionKind::Tp, ExpertParallelism::TensorParallel),
            (MoEExecutionKind::Ep, ExpertParallelism::ExpertParallel),
        ] {
            let mut spec = projection_spec(parallelism);
            spec.source_layout = ExpertSourceLayout::PerExpertFused {
                gate_up: (0..4)
                    .map(|expert| format!("experts.gate_up.{expert}"))
                    .collect(),
                down: (0..4)
                    .map(|expert| format!("experts.down.{expert}"))
                    .collect(),
                sidecars: Vec::new(),
            };
            let err =
                resolve_expert_manifest_for_policy(&[spec], &manifest, &projection_policy(kind, 2))
                    .unwrap_err();
            assert!(err.contains("PerExpert"), "kind {kind:?}: {err}");
        }
    }

    #[test]
    fn projection_never_mutates_original_manifest_or_specs() {
        let manifest = projection_manifest(Some(0), 4, 512, 64, ExpertAssign::Stride);
        let specs = vec![projection_spec(ExpertParallelism::TensorParallel)];
        let manifest_before = manifest.clone();
        let specs_before = specs.clone();

        resolve_expert_manifest_for_policy(
            &specs,
            &manifest,
            &projection_policy(MoEExecutionKind::Tp, 2),
        )
        .unwrap();
        // A failing resolution must leave both inputs untouched as well.
        let err = resolve_expert_manifest_for_policy(
            &specs,
            &manifest,
            &projection_policy(MoEExecutionKind::Single, 1),
        )
        .unwrap_err();
        assert!(
            err.contains("does not match execution policy kind"),
            "got: {err}"
        );
        assert_eq!(manifest, manifest_before);
        assert_eq!(specs, specs_before);
    }

    #[test]
    fn projection_exact_tp_mesh_drives_divisibility_and_256_alignment() {
        let spec = projection_spec(ExpertParallelism::TensorParallel);
        // Truthful shapes: inter=512 gives gate_up axis-1 slice 1024/2=512 and
        // down axis-2 slice 512/2=256, both 256-aligned under Tp=2.
        let ok = projection_manifest(Some(0), 4, 512, 64, ExpertAssign::Stride);
        assert!(resolve_expert_manifest_for_policy(
            &[spec.clone()],
            &ok,
            &projection_policy(MoEExecutionKind::Tp, 2),
        )
        .is_ok());

        // The exact Tp mesh drives divisibility: 2*inter=1024 is not divisible
        // by Tp=3 even though the static surrogate itself is policy-agnostic.
        let err = resolve_expert_manifest_for_policy(
            &[spec.clone()],
            &ok,
            &projection_policy(MoEExecutionKind::Tp, 3),
        )
        .unwrap_err();
        assert!(err.contains("not divisible by Tp=3"), "got: {err}");

        // And local-256 alignment: inter=256 yields a down slice 256/2=128
        // under Tp=2, which is not a multiple of 256.
        let small = projection_manifest(Some(0), 4, 256, 64, ExpertAssign::Stride);
        let err = resolve_expert_manifest_for_policy(
            &[spec.clone()],
            &small,
            &projection_policy(MoEExecutionKind::Tp, 2),
        )
        .unwrap_err();
        assert!(err.contains("multiple of 256"), "got: {err}");

        // Malformed claimed-source shapes are refused by strict full-manifest
        // validation on the projected clone.
        let mut malformed = projection_manifest(Some(0), 4, 512, 64, ExpertAssign::Stride);
        malformed[1].logical_shape = vec![4, 0, 64];
        let err = resolve_expert_manifest_for_policy(
            &[spec],
            &malformed,
            &projection_policy(MoEExecutionKind::Tp, 2),
        )
        .unwrap_err();
        assert!(err.contains("zero") || err.contains("3D"), "got: {err}");
    }

    #[test]
    fn projection_rejects_wrong_static_count_assign_axes_nested_plain_shards_and_pin_tied() {
        let single_policy = projection_policy(MoEExecutionKind::Single, 1);
        let single_spec = projection_spec(ExpertParallelism::Single);

        // Static surrogate with the wrong expert count is not projectable.
        let mut count = projection_manifest(Some(0), 4, 512, 64, ExpertAssign::Stride);
        count[1].policy = ShardPolicy::ExpertSharded {
            n_experts: 3,
            assign: ExpertAssign::Stride,
        };
        let err =
            resolve_expert_manifest_for_policy(&[single_spec.clone()], &count, &single_policy)
                .unwrap_err();
        assert!(err.contains("experts.gate_up"), "got: {err}");
        assert!(err.contains("n_experts=3"), "got: {err}");

        // Static surrogate with the wrong assignment is not projectable.
        let mut assign = projection_manifest(Some(0), 4, 512, 64, ExpertAssign::Stride);
        assign[1].policy = ShardPolicy::ExpertSharded {
            n_experts: 4,
            assign: ExpertAssign::Contiguous,
        };
        let err =
            resolve_expert_manifest_for_policy(&[single_spec.clone()], &assign, &single_policy)
                .unwrap_err();
        assert!(err.contains("Contiguous"), "got: {err}");

        let tp_policy = projection_policy(MoEExecutionKind::Tp, 2);
        let tp_spec = projection_spec(ExpertParallelism::TensorParallel);

        // Plain shards on a claimed projection are refused — a direct
        // competing RowShard is reported by projection eligibility, not left
        // to the collective scheduler.
        let mut plain = projection_manifest(Some(0), 4, 512, 64, ExpertAssign::Stride);
        plain[2].policy = ShardPolicy::RowShard { axis: 2 };
        let err =
            resolve_expert_manifest_for_policy(&[tp_spec.clone()], &plain, &tp_policy).unwrap_err();
        assert!(err.contains("not projectable"), "got: {err}");
        assert!(
            !err.contains("competing per-weight collective"),
            "got: {err}"
        );

        // A static Replicate on a packed projection is not projectable under TP.
        let mut replicate = projection_manifest(Some(0), 4, 512, 64, ExpertAssign::Stride);
        replicate[1].policy = ShardPolicy::Replicate;
        let err = resolve_expert_manifest_for_policy(&[tp_spec.clone()], &replicate, &tp_policy)
            .unwrap_err();
        assert!(err.contains("not projectable"), "got: {err}");

        // Wrong inner axis on an already-ExpertTensorSharded source is refused.
        let mut axis = projection_manifest(Some(0), 4, 512, 64, ExpertAssign::Stride);
        axis[1].policy = ShardPolicy::ExpertTensorSharded {
            n_experts: 4,
            inner: Box::new(ShardPolicy::ColumnShard { axis: 0 }),
        };
        let err =
            resolve_expert_manifest_for_policy(&[tp_spec.clone()], &axis, &tp_policy).unwrap_err();
        assert!(err.contains("not projectable"), "got: {err}");

        // Nested ExpertTensorSharded inners are refused.
        let mut nested = projection_manifest(Some(0), 4, 512, 64, ExpertAssign::Stride);
        nested[1].policy = ShardPolicy::ExpertTensorSharded {
            n_experts: 4,
            inner: Box::new(ShardPolicy::ExpertTensorSharded {
                n_experts: 4,
                inner: Box::new(ShardPolicy::Replicate),
            }),
        };
        let err = resolve_expert_manifest_for_policy(&[tp_spec.clone()], &nested, &tp_policy)
            .unwrap_err();
        assert!(err.contains("not projectable"), "got: {err}");

        // Unsupported Pin/Tied static policies on packed projections are refused.
        let mut pin = projection_manifest(Some(0), 4, 512, 64, ExpertAssign::Stride);
        pin[1].policy = ShardPolicy::Pin(PinTarget::Embed);
        let err = resolve_expert_manifest_for_policy(&[single_spec.clone()], &pin, &single_policy)
            .unwrap_err();
        assert!(err.contains("not projectable"), "got: {err}");

        let mut tied = projection_manifest(Some(0), 4, 512, 64, ExpertAssign::Stride);
        tied[1].policy = ShardPolicy::Tied {
            source: "mlp.gate".into(),
        };
        let err =
            resolve_expert_manifest_for_policy(&[single_spec], &tied, &single_policy).unwrap_err();
        assert!(err.contains("not projectable"), "got: {err}");
    }

    #[test]
    fn projection_rejects_spec_policy_kind_mismatch() {
        let manifest = projection_manifest(Some(0), 4, 512, 64, ExpertAssign::Stride);
        let err = resolve_expert_manifest_for_policy(
            &[projection_spec(ExpertParallelism::Single)],
            &manifest,
            &projection_policy(MoEExecutionKind::Tp, 2),
        )
        .unwrap_err();
        assert!(err.contains("block-0"), "got: {err}");
        assert!(
            err.contains("does not match execution policy kind"),
            "got: {err}"
        );

        let err = resolve_expert_manifest_for_policy(
            &[projection_spec(ExpertParallelism::TensorParallel)],
            &manifest,
            &projection_policy(MoEExecutionKind::Single, 1),
        )
        .unwrap_err();
        assert!(
            err.contains("does not match execution policy kind"),
            "got: {err}"
        );

        let err = resolve_expert_manifest_for_policy(
            &[projection_spec(ExpertParallelism::ExpertParallel)],
            &manifest,
            &projection_policy(MoEExecutionKind::Tp, 2),
        )
        .unwrap_err();
        assert!(
            err.contains("does not match execution policy kind"),
            "got: {err}"
        );
    }

    #[test]
    fn projection_rejects_missing_ambiguous_repeated_cross_group_and_wrong_layer_sources() {
        let manifest = projection_manifest(Some(0), 4, 512, 64, ExpertAssign::Stride);
        let single_policy = projection_policy(MoEExecutionKind::Single, 1);

        // Missing claimed source.
        let mut missing = projection_spec(ExpertParallelism::Single);
        missing.source_layout = ExpertSourceLayout::PackedFused {
            gate_up: "experts.missing".into(),
            down: "experts.down".into(),
            sidecars: Vec::new(),
        };
        let err =
            resolve_expert_manifest_for_policy(&[missing], &manifest, &single_policy).unwrap_err();
        assert!(err.contains("experts.missing"), "got: {err}");
        assert!(err.contains("not found in manifest scope"), "got: {err}");

        // Ambiguous claimed source (duplicate (name, layer) manifest entries):
        // the static-policy-neutral prefix reports the duplicate manifest
        // identity before projection can reach the source finder.
        let mut ambiguous = projection_manifest(Some(0), 4, 512, 64, ExpertAssign::Stride);
        ambiguous.push(ambiguous[1].clone());
        let err = resolve_expert_manifest_for_policy(
            &[projection_spec(ExpertParallelism::Single)],
            &ambiguous,
            &single_policy,
        )
        .unwrap_err();
        assert!(
            err.contains("duplicate manifest (name, layer)"),
            "got: {err}"
        );

        // Repeated source within one group.
        let mut repeated = projection_spec(ExpertParallelism::Single);
        repeated.source_layout = ExpertSourceLayout::PackedSeparate {
            gate: "experts.gate_up".into(),
            up: "experts.gate_up".into(),
            down: "experts.down".into(),
            sidecars: Vec::new(),
        };
        let err =
            resolve_expert_manifest_for_policy(&[repeated], &manifest, &single_policy).unwrap_err();
        assert!(err.contains("repeats projection source"), "got: {err}");

        // Cross-group collision: the first declared group is the deterministic owner.
        let first = projection_spec(ExpertParallelism::Single);
        let mut second = projection_spec(ExpertParallelism::Single);
        second.group = "block-1".into();
        let err = resolve_expert_manifest_for_policy(&[first, second], &manifest, &single_policy)
            .unwrap_err();
        assert!(
            err.contains("already claimed by expert group 'block-0' (declared first)"),
            "got: {err}"
        );

        // Wrong layer: the claimed name exists only at another layer scope.
        let other_layer = projection_manifest(Some(1), 4, 512, 64, ExpertAssign::Stride);
        let err = resolve_expert_manifest_for_policy(
            &[projection_spec(ExpertParallelism::Single)],
            &other_layer,
            &single_policy,
        )
        .unwrap_err();
        assert!(err.contains("not found in manifest scope"), "got: {err}");
    }

    #[test]
    fn projection_residual_keeps_unrelated_dense_row_shard() {
        let mut manifest = projection_manifest(Some(0), 4, 512, 64, ExpertAssign::Stride);
        manifest.push(WeightEntry::layer(
            "wo",
            0,
            vec![8, 8],
            DType::F16,
            ShardPolicy::RowShard { axis: 1 },
        ));
        let resolution = resolve_expert_manifest_for_policy(
            &[projection_spec(ExpertParallelism::ExpertParallel)],
            &manifest,
            &projection_policy(MoEExecutionKind::Ep, 2),
        )
        .unwrap();
        // The unclaimed dense row shard keeps its Tp all-reduce; the claimed
        // expert projections contribute none.
        assert_eq!(
            resolution.layer_collectives,
            vec![(0, CollectiveHint::AllReduce { kind: DimKind::Tp })]
        );
    }

    #[test]
    fn projection_packed_separate_claims_gate_up_down_and_fused_does_not_hide_unclaimed_up() {
        let mut separate_manifest = projection_manifest(Some(0), 4, 512, 64, ExpertAssign::Stride);
        // PackedSeparate claims gate/up/down as separate logical sources. Each
        // separate gate/up projection is its own width-inter tensor (512), not
        // the fused 2*inter (1024) gate_up blob; down is
        // [n_experts, hidden, inter].
        separate_manifest.retain(|entry| entry.name != "experts.gate_up");
        separate_manifest.push(WeightEntry::layer(
            "experts.gate",
            0,
            vec![4, 512, 64],
            DType::F16,
            ShardPolicy::ExpertSharded {
                n_experts: 4,
                assign: ExpertAssign::Stride,
            },
        ));
        separate_manifest.push(WeightEntry::layer(
            "experts.up",
            0,
            vec![4, 512, 64],
            DType::F16,
            ShardPolicy::ExpertSharded {
                n_experts: 4,
                assign: ExpertAssign::Stride,
            },
        ));
        let mut separate = projection_spec(ExpertParallelism::ExpertParallel);
        separate.source_layout = ExpertSourceLayout::PackedSeparate {
            gate: "experts.gate".into(),
            up: "experts.up".into(),
            down: "experts.down".into(),
            sidecars: Vec::new(),
        };
        let resolution = resolve_expert_manifest_for_policy(
            &[separate],
            &separate_manifest,
            &projection_policy(MoEExecutionKind::Ep, 2),
        )
        .unwrap();
        // Gate, up, and down are all claimed and excluded exactly once: no
        // stray per-weight Ep collective remains.
        assert!(
            resolution.layer_collectives.is_empty(),
            "got: {:?}",
            resolution.layer_collectives
        );

        // A fused layout claims only gate_up + down. The separate experts.up
        // source is unclaimed and must remain visible in the residual schedule
        // — the projection never hides a source. Its standalone shape is the
        // truthful separate up width (inter = 512), not the fused 2*inter
        // (1024) gate_up width.
        let mut fused_manifest = projection_manifest(Some(0), 4, 512, 64, ExpertAssign::Stride);
        fused_manifest.push(WeightEntry::layer(
            "experts.up",
            0,
            vec![4, 512, 64],
            DType::F16,
            ShardPolicy::ExpertSharded {
                n_experts: 4,
                assign: ExpertAssign::Stride,
            },
        ));
        let fused = projection_spec(ExpertParallelism::ExpertParallel);
        let resolution = resolve_expert_manifest_for_policy(
            &[fused],
            &fused_manifest,
            &projection_policy(MoEExecutionKind::Ep, 2),
        )
        .unwrap();
        assert_eq!(
            resolution.layer_collectives,
            vec![(0, CollectiveHint::AllReduce { kind: DimKind::Ep })]
        );

        // Under Single and TP the same unclaimed source still keeps its static
        // Ep collective visible as the intentional family-completeness signal:
        // policy projection does not erase it, and no active-policy residual
        // rejection is applied.
        for kind in [MoEExecutionKind::Single, MoEExecutionKind::Tp] {
            let ranks = if kind == MoEExecutionKind::Tp { 2 } else { 1 };
            let spec = projection_spec(kind_parallelism(kind));
            let resolution = resolve_expert_manifest_for_policy(
                &[spec],
                &fused_manifest,
                &projection_policy(kind, ranks),
            )
            .unwrap_or_else(|err| panic!("kind {kind:?}: {err}"));
            assert_eq!(
                resolution.layer_collectives,
                vec![(0, CollectiveHint::AllReduce { kind: DimKind::Ep })],
                "kind {kind:?}"
            );
        }
    }

    #[test]
    fn projection_rejects_unrelated_invalid_entries() {
        let mut manifest = projection_manifest(Some(0), 4, 512, 64, ExpertAssign::Stride);
        manifest.push(WeightEntry::layer(
            "wo",
            0,
            vec![8, 7],
            DType::F16,
            ShardPolicy::RowShard { axis: 1 },
        ));
        let err = resolve_expert_manifest_for_policy(
            &[projection_spec(ExpertParallelism::TensorParallel)],
            &manifest,
            &projection_policy(MoEExecutionKind::Tp, 2),
        )
        .unwrap_err();
        assert!(err.contains("wo"), "got: {err}");
        assert!(err.contains("not divisible by Tp=2"), "got: {err}");
    }

    // ------------------------------------------------------------------
    // Projection ownership: router/sidecar disjointness and precedence
    // ------------------------------------------------------------------

    #[test]
    fn projection_rejects_same_group_router_alias() {
        // A projection source claimed by a group is never also that group's
        // router reference. Projection ownership reports the alias with both
        // roles and both groups named (the strict path would only reject the
        // 3D router shape later, so this must be refused before projection).
        let manifest = projection_manifest(Some(0), 4, 512, 64, ExpertAssign::Stride);
        let mut spec = projection_spec(ExpertParallelism::Single);
        spec.router = "experts.gate_up".into();
        let err = resolve_expert_manifest_for_policy(
            &[spec],
            &manifest,
            &projection_policy(MoEExecutionKind::Single, 1),
        )
        .unwrap_err();
        assert!(err.contains("source gate_up"), "got: {err}");
        assert!(err.contains("router"), "got: {err}");
        assert!(err.contains("block-0"), "got: {err}");
        assert!(err.contains("also referenced as"), "got: {err}");
    }

    #[test]
    fn projection_rejects_same_group_sidecar_alias_under_single_and_ep() {
        // A projection source claimed by a group is never also that group's
        // sidecar reference. The strict path would ACCEPT the aliased sidecar
        // (the projected Replicate/ExpertSharded policy satisfies the sidecar
        // policy and shape checks), so projection ownership is the only
        // authority that refuses it.
        for (kind, ranks) in [(MoEExecutionKind::Single, 1), (MoEExecutionKind::Ep, 2)] {
            let manifest = projection_manifest(Some(0), 4, 512, 64, ExpertAssign::Stride);
            let mut spec = projection_spec(kind_parallelism(kind));
            spec.source_layout = ExpertSourceLayout::PackedFused {
                gate_up: "experts.gate_up".into(),
                down: "experts.down".into(),
                sidecars: vec!["experts.gate_up".into()],
            };
            let err = resolve_expert_manifest_for_policy(
                &[spec],
                &manifest,
                &projection_policy(kind, ranks),
            )
            .unwrap_err();
            assert!(err.contains("source gate_up"), "kind {kind:?}: {err}");
            assert!(err.contains("sidecar[0]"), "kind {kind:?}: {err}");
            assert!(err.contains("block-0"), "kind {kind:?}: {err}");
            assert!(err.contains("also referenced as"), "kind {kind:?}: {err}");
        }
    }

    #[test]
    fn projection_rejects_cross_group_router_alias() {
        // Group A's claimed projection must not be group B's router reference.
        let mut manifest = projection_manifest(Some(0), 4, 512, 64, ExpertAssign::Stride);
        manifest.push(WeightEntry::layer(
            "experts.b_gate_up",
            0,
            vec![4, 1024, 64],
            DType::F16,
            ShardPolicy::ExpertSharded {
                n_experts: 4,
                assign: ExpertAssign::Stride,
            },
        ));
        manifest.push(WeightEntry::layer(
            "experts.b_down",
            0,
            vec![4, 64, 512],
            DType::F16,
            ShardPolicy::ExpertSharded {
                n_experts: 4,
                assign: ExpertAssign::Stride,
            },
        ));
        let mut first = projection_spec(ExpertParallelism::Single);
        first.group = "block-0".into();
        let mut second = projection_spec(ExpertParallelism::Single);
        second.group = "block-1".into();
        second.source_layout = ExpertSourceLayout::PackedFused {
            gate_up: "experts.b_gate_up".into(),
            down: "experts.b_down".into(),
            sidecars: Vec::new(),
        };
        second.router = "experts.gate_up".into();
        let err = resolve_expert_manifest_for_policy(
            &[first, second],
            &manifest,
            &projection_policy(MoEExecutionKind::Single, 1),
        )
        .unwrap_err();
        assert!(err.contains("source gate_up"), "got: {err}");
        assert!(err.contains("router"), "got: {err}");
        assert!(err.contains("block-0"), "got: {err}");
        assert!(err.contains("block-1"), "got: {err}");
        assert!(err.contains("also referenced as"), "got: {err}");
    }

    #[test]
    fn projection_rejects_cross_group_sidecar_alias() {
        // Group A's claimed projection must not be group B's sidecar reference.
        let mut manifest = projection_manifest(Some(0), 4, 512, 64, ExpertAssign::Stride);
        manifest.push(WeightEntry::layer(
            "experts.b_gate_up",
            0,
            vec![4, 1024, 64],
            DType::F16,
            ShardPolicy::ExpertSharded {
                n_experts: 4,
                assign: ExpertAssign::Stride,
            },
        ));
        manifest.push(WeightEntry::layer(
            "experts.b_down",
            0,
            vec![4, 64, 512],
            DType::F16,
            ShardPolicy::ExpertSharded {
                n_experts: 4,
                assign: ExpertAssign::Stride,
            },
        ));
        let mut first = projection_spec(ExpertParallelism::ExpertParallel);
        first.group = "block-0".into();
        let mut second = projection_spec(ExpertParallelism::ExpertParallel);
        second.group = "block-1".into();
        second.source_layout = ExpertSourceLayout::PackedFused {
            gate_up: "experts.b_gate_up".into(),
            down: "experts.b_down".into(),
            sidecars: vec!["experts.down".into()],
        };
        let err = resolve_expert_manifest_for_policy(
            &[first, second],
            &manifest,
            &projection_policy(MoEExecutionKind::Ep, 2),
        )
        .unwrap_err();
        assert!(err.contains("source down"), "got: {err}");
        assert!(err.contains("sidecar[0]"), "got: {err}");
        assert!(err.contains("block-0"), "got: {err}");
        assert!(err.contains("block-1"), "got: {err}");
        assert!(err.contains("also referenced as"), "got: {err}");
    }

    #[test]
    fn projection_accepts_distinct_router_and_sidecars_and_keeps_inputs_unchanged() {
        // Ordinary distinct router and sidecar references are not conflicts:
        // resolution succeeds and neither the manifest nor the specs change.
        let mut manifest = projection_manifest(Some(0), 4, 512, 64, ExpertAssign::Stride);
        manifest.push(WeightEntry::layer(
            "experts.scale",
            0,
            vec![4, 4],
            DType::F16,
            ShardPolicy::Replicate,
        ));
        let mut spec = projection_spec(ExpertParallelism::Single);
        spec.source_layout = ExpertSourceLayout::PackedFused {
            gate_up: "experts.gate_up".into(),
            down: "experts.down".into(),
            sidecars: vec!["experts.scale".into()],
        };
        let manifest_before = manifest.clone();
        let spec_before = spec.clone();
        let resolution = resolve_expert_manifest_for_policy(
            &[spec.clone()],
            &manifest,
            &projection_policy(MoEExecutionKind::Single, 1),
        )
        .unwrap();
        assert_eq!(resolution.plans.len(), 1);
        assert_eq!(manifest, manifest_before);
        assert_eq!(spec, spec_before);
    }

    #[test]
    fn projection_two_groups_legally_share_a_router() {
        // Repeated router references across groups are legal: the disjointness
        // check keys by (name, layer) and only reports a projection claim that
        // intersects ANY router/sidecar reference — a shared router entry is
        // not a conflict and never triggers the alias diagnostic.
        let mut manifest = projection_manifest(Some(0), 4, 512, 64, ExpertAssign::Stride);
        manifest.push(WeightEntry::layer(
            "experts.b_gate_up",
            0,
            vec![4, 1024, 64],
            DType::F16,
            ShardPolicy::ExpertSharded {
                n_experts: 4,
                assign: ExpertAssign::Stride,
            },
        ));
        manifest.push(WeightEntry::layer(
            "experts.b_down",
            0,
            vec![4, 64, 512],
            DType::F16,
            ShardPolicy::ExpertSharded {
                n_experts: 4,
                assign: ExpertAssign::Stride,
            },
        ));
        let mut first = projection_spec(ExpertParallelism::Single);
        first.group = "block-0".into();
        let mut second = projection_spec(ExpertParallelism::Single);
        second.group = "block-1".into();
        second.source_layout = ExpertSourceLayout::PackedFused {
            gate_up: "experts.b_gate_up".into(),
            down: "experts.b_down".into(),
            sidecars: Vec::new(),
        };
        let resolution = resolve_expert_manifest_for_policy(
            &[first, second],
            &manifest,
            &projection_policy(MoEExecutionKind::Single, 1),
        )
        .unwrap();
        assert_eq!(resolution.plans.len(), 2);
        assert!(resolution
            .plans
            .iter()
            .all(|plan| plan.router == "mlp.gate"));
    }

    #[test]
    fn projection_two_groups_legally_share_a_sidecar() {
        // Two groups referencing the same sidecar entry is legal: only a
        // projection claim intersecting a router/sidecar reference conflicts.
        let mut manifest = projection_manifest(Some(0), 4, 512, 64, ExpertAssign::Stride);
        manifest.push(WeightEntry::layer(
            "experts.scale",
            0,
            vec![4, 4],
            DType::F16,
            ShardPolicy::Replicate,
        ));
        manifest.push(WeightEntry::layer(
            "experts.b_gate_up",
            0,
            vec![4, 1024, 64],
            DType::F16,
            ShardPolicy::ExpertSharded {
                n_experts: 4,
                assign: ExpertAssign::Stride,
            },
        ));
        manifest.push(WeightEntry::layer(
            "experts.b_down",
            0,
            vec![4, 64, 512],
            DType::F16,
            ShardPolicy::ExpertSharded {
                n_experts: 4,
                assign: ExpertAssign::Stride,
            },
        ));
        let mut first = projection_spec(ExpertParallelism::Single);
        first.group = "block-0".into();
        first.source_layout = ExpertSourceLayout::PackedFused {
            gate_up: "experts.gate_up".into(),
            down: "experts.down".into(),
            sidecars: vec!["experts.scale".into()],
        };
        let mut second = projection_spec(ExpertParallelism::Single);
        second.group = "block-1".into();
        second.source_layout = ExpertSourceLayout::PackedFused {
            gate_up: "experts.b_gate_up".into(),
            down: "experts.b_down".into(),
            sidecars: vec!["experts.scale".into()],
        };
        let resolution = resolve_expert_manifest_for_policy(
            &[first, second],
            &manifest,
            &projection_policy(MoEExecutionKind::Single, 1),
        )
        .unwrap();
        assert_eq!(resolution.plans.len(), 2);
    }

    #[test]
    fn projection_same_source_name_at_different_layer_is_not_an_alias() {
        // Ownership and the router/sidecar disjointness are keyed by
        // (name, layer): the same source name at a different layer scope is a
        // distinct entry and never triggers alias rejection.
        let mut manifest = projection_manifest(Some(0), 4, 512, 64, ExpertAssign::Stride);
        manifest.extend(projection_manifest(
            Some(1),
            4,
            512,
            64,
            ExpertAssign::Stride,
        ));
        let mut first = projection_spec(ExpertParallelism::Single);
        first.layer = Some(0);
        let mut second = projection_spec(ExpertParallelism::Single);
        second.layer = Some(1);
        let resolution = resolve_expert_manifest_for_policy(
            &[first, second],
            &manifest,
            &projection_policy(MoEExecutionKind::Single, 1),
        )
        .unwrap();
        assert_eq!(resolution.plans.len(), 2);
    }

    #[test]
    fn projection_precedence_prefix_before_ownership_and_projection() {
        let manifest = projection_manifest(Some(0), 4, 512, 64, ExpertAssign::Stride);
        let single_policy = projection_policy(MoEExecutionKind::Single, 1);

        // Duplicate group identity + repeated source: the static-policy-neutral
        // prefix reports the duplicate group/layer identity before ownership
        // reports the repeated projection source.
        let mut first = projection_spec(ExpertParallelism::Single);
        first.group = "block-0".into();
        let mut second = projection_spec(ExpertParallelism::Single);
        second.group = "block-0".into();
        second.source_layout = ExpertSourceLayout::PackedSeparate {
            gate: "experts.gate_up".into(),
            up: "experts.gate_up".into(),
            down: "experts.down".into(),
            sidecars: Vec::new(),
        };
        let err = resolve_expert_manifest_for_policy(&[first, second], &manifest, &single_policy)
            .unwrap_err();
        assert!(err.contains("duplicate group/layer identity"), "got: {err}");

        // Invalid metadata + cross-group claim: the prefix reports the invalid
        // metadata before ownership reports the conflict.
        let mut invalid = projection_spec(ExpertParallelism::Single);
        invalid.group = "block-0".into();
        invalid.n_experts = 0;
        let mut claimant = projection_spec(ExpertParallelism::Single);
        claimant.group = "block-1".into();
        let err =
            resolve_expert_manifest_for_policy(&[invalid, claimant], &manifest, &single_policy)
                .unwrap_err();
        assert!(err.contains("n_experts=0"), "got: {err}");
        assert!(!err.contains("already claimed"), "got: {err}");

        // Duplicate manifest identity + claimed source: the prefix reports the
        // duplicate (name, layer) before projection reaches the source finder.
        let mut duplicated = projection_manifest(Some(0), 4, 512, 64, ExpertAssign::Stride);
        duplicated.push(duplicated[1].clone());
        let err = resolve_expert_manifest_for_policy(
            &[projection_spec(ExpertParallelism::Single)],
            &duplicated,
            &single_policy,
        )
        .unwrap_err();
        assert!(
            err.contains("duplicate manifest (name, layer)"),
            "got: {err}"
        );
    }

    #[test]
    fn projection_tp1_enforces_local_256_and_tp2_adds_divisibility() {
        // The exact TP mesh drives the local-256 contract for EVERY rank
        // count: under Tp=1 the aligned inter=256 fixture (gate_up axis-1
        // slice 512, down axis-2 slice 256) resolves, while Tp=2 rejects the
        // same fixture because the down slice 256/2=128 is not a multiple of
        // 256 — divisibility by the exact rank count is additionally enforced
        // under Tp>1.
        let spec = projection_spec(ExpertParallelism::TensorParallel);
        let small = projection_manifest(Some(0), 4, 256, 64, ExpertAssign::Stride);
        let resolution = resolve_expert_manifest_for_policy(
            &[spec.clone()],
            &small,
            &projection_policy(MoEExecutionKind::Tp, 1),
        )
        .unwrap();
        assert_eq!(resolution.plans[0].group_size, 1);
        assert_eq!(
            resolution.plans[0].collective,
            Some(ExpertPostCombineAllReduce::TensorParallel)
        );

        let err = resolve_expert_manifest_for_policy(
            &[spec],
            &small,
            &projection_policy(MoEExecutionKind::Tp, 2),
        )
        .unwrap_err();
        assert!(err.contains("multiple of 256"), "got: {err}");
    }

    #[test]
    fn projection_tp1_rejects_non_256_local_slices() {
        // Tp=1 performs no slicing, but the local slice width must still be a
        // multiple of 256: the effective resident tensor would carry a slice
        // the quant group size cannot represent. This is projection-path
        // eligibility only — the strict validator's tp > 1 gate is unchanged.
        let spec = projection_spec(ExpertParallelism::TensorParallel);
        let policy = projection_policy(MoEExecutionKind::Tp, 1);

        // inter=128: the gate_up axis-1 slice 256 is aligned, but the down
        // axis-2 slice 128 is not — the down role diagnostic fires.
        let width128 = projection_manifest(Some(0), 4, 128, 64, ExpertAssign::Stride);
        let err =
            resolve_expert_manifest_for_policy(&[spec.clone()], &width128, &policy).unwrap_err();
        assert!(err.contains("source down"), "got: {err}");
        assert!(err.contains("experts.down"), "got: {err}");
        assert!(err.contains("block-0"), "got: {err}");
        assert!(err.contains("multiple of 256"), "got: {err}");

        // inter=192: the gate_up axis-1 slice 384 is not aligned — the
        // gate/up role (axis 1) diagnostic fires before the down role.
        let width192 = projection_manifest(Some(0), 4, 192, 64, ExpertAssign::Stride);
        let err = resolve_expert_manifest_for_policy(&[spec], &width192, &policy).unwrap_err();
        assert!(err.contains("source gate_up"), "got: {err}");
        assert!(err.contains("experts.gate_up"), "got: {err}");
        assert!(err.contains("multiple of 256"), "got: {err}");
    }

    #[test]
    fn projection_tp_role_dim_check_leaves_single_and_ep_unaffected() {
        // The exact-TP role-dimension eligibility is projection-path and
        // TP-only: the same non-aligned-width fixture resolves under Single
        // (Replicate) and EP (ExpertSharded), where no slicing occurs.
        let manifest = projection_manifest(Some(0), 4, 128, 64, ExpertAssign::Stride);
        for (kind, ranks) in [(MoEExecutionKind::Single, 1), (MoEExecutionKind::Ep, 2)] {
            let spec = projection_spec(kind_parallelism(kind));
            let resolution = resolve_expert_manifest_for_policy(
                &[spec],
                &manifest,
                &projection_policy(kind, ranks),
            )
            .unwrap_or_else(|err| panic!("kind {kind:?}: {err}"));
            assert_eq!(resolution.plans.len(), 1);
        }
    }

    #[test]
    fn projection_rejects_plain_column_shard_projection_source() {
        // A bare ColumnShard on a claimed projection is a plain competing
        // shard: not a static surrogate, not the effective policy — refused.
        let mut tp_manifest = projection_manifest(Some(0), 4, 512, 64, ExpertAssign::Stride);
        tp_manifest[1].policy = ShardPolicy::ColumnShard { axis: 1 };
        let err = resolve_expert_manifest_for_policy(
            &[projection_spec(ExpertParallelism::TensorParallel)],
            &tp_manifest,
            &projection_policy(MoEExecutionKind::Tp, 2),
        )
        .unwrap_err();
        assert!(err.contains("not projectable"), "got: {err}");

        let mut single_manifest = projection_manifest(Some(0), 4, 512, 64, ExpertAssign::Stride);
        single_manifest[1].policy = ShardPolicy::ColumnShard { axis: 0 };
        let err = resolve_expert_manifest_for_policy(
            &[projection_spec(ExpertParallelism::Single)],
            &single_manifest,
            &projection_policy(MoEExecutionKind::Single, 1),
        )
        .unwrap_err();
        assert!(err.contains("not projectable"), "got: {err}");
    }

    #[test]
    fn projection_rejects_already_effective_tp_policy_outside_tp_and_replicate_under_ep() {
        // An already-TP-effective ExpertTensorSharded projection is effective
        // only under TP; under Single and EP it is refused rather than
        // reinterpreted.
        let mut manifest = projection_manifest(Some(0), 4, 512, 64, ExpertAssign::Stride);
        manifest[1].policy = ShardPolicy::ExpertTensorSharded {
            n_experts: 4,
            inner: Box::new(ShardPolicy::ColumnShard { axis: 1 }),
        };
        manifest[2].policy = ShardPolicy::ExpertTensorSharded {
            n_experts: 4,
            inner: Box::new(ShardPolicy::RowShard { axis: 2 }),
        };
        let err = resolve_expert_manifest_for_policy(
            &[projection_spec(ExpertParallelism::Single)],
            &manifest,
            &projection_policy(MoEExecutionKind::Single, 1),
        )
        .unwrap_err();
        assert!(err.contains("not projectable"), "got: {err}");

        let err = resolve_expert_manifest_for_policy(
            &[projection_spec(ExpertParallelism::ExpertParallel)],
            &manifest,
            &projection_policy(MoEExecutionKind::Ep, 2),
        )
        .unwrap_err();
        assert!(err.contains("not projectable"), "got: {err}");

        // Replicate is the Single-effective placement; under EP it is refused.
        let mut replicate = projection_manifest(Some(0), 4, 512, 64, ExpertAssign::Stride);
        replicate[1].policy = ShardPolicy::Replicate;
        replicate[2].policy = ShardPolicy::Replicate;
        let err = resolve_expert_manifest_for_policy(
            &[projection_spec(ExpertParallelism::ExpertParallel)],
            &replicate,
            &projection_policy(MoEExecutionKind::Ep, 2),
        )
        .unwrap_err();
        assert!(err.contains("not projectable"), "got: {err}");
    }

    #[test]
    fn projection_immutability_with_unrelated_tied_and_sidecar_entries() {
        // Richer immutability fixture: unrelated dense entry, a Tied model
        // entry, and a declared sidecar all survive both successful and
        // failing resolutions equality-identically.
        let mut manifest = projection_manifest(Some(0), 4, 512, 64, ExpertAssign::Stride);
        manifest.push(WeightEntry::layer(
            "wo",
            0,
            vec![8, 8],
            DType::F16,
            ShardPolicy::RowShard { axis: 1 },
        ));
        manifest.push(WeightEntry::model(
            "token_embd",
            vec![8, 8],
            DType::F16,
            ShardPolicy::Replicate,
        ));
        manifest.push(WeightEntry::model(
            "lm_head",
            vec![8, 8],
            DType::F16,
            ShardPolicy::Tied {
                source: "token_embd".into(),
            },
        ));
        manifest.push(WeightEntry::layer(
            "experts.scale",
            0,
            vec![4, 4],
            DType::F16,
            ShardPolicy::Replicate,
        ));
        let mut spec = projection_spec(ExpertParallelism::Single);
        spec.source_layout = ExpertSourceLayout::PackedFused {
            gate_up: "experts.gate_up".into(),
            down: "experts.down".into(),
            sidecars: vec!["experts.scale".into()],
        };
        let manifest_before = manifest.clone();
        let spec_before = spec.clone();
        resolve_expert_manifest_for_policy(
            &[spec.clone()],
            &manifest,
            &projection_policy(MoEExecutionKind::Single, 1),
        )
        .unwrap();
        let err = resolve_expert_manifest_for_policy(
            &[spec.clone()],
            &manifest,
            &projection_policy(MoEExecutionKind::Tp, 2),
        )
        .unwrap_err();
        assert!(
            err.contains("does not match execution policy kind"),
            "got: {err}"
        );
        assert_eq!(manifest, manifest_before);
        assert_eq!(spec, spec_before);
    }
}
