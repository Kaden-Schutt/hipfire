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
            ShardPolicy::ExpertTensorSharded { inner, .. } => {
                // Expert intermediate dim must be divisible by Tp and the
                // resulting slice must be a multiple of 256 (the quant group
                // size for MQ2G256/MQ3G256 experts).
                // logical_shape: [n_experts, 2*inter, hidden] (gate‖up) or
                // [n_experts, hidden, inter] (down).
                // Gate/up (ColumnShard): sharded dim is axis-1 (2*inter).
                // Down (RowShard): sharded dim is axis-2 (inter).
                let (axis, kind_name) = match inner.as_ref() {
                    ShardPolicy::ColumnShard { .. } => (1, "ColumnShard (2*inter)"),
                    ShardPolicy::RowShard { .. } => (2, "RowShard (inter)"),
                    _ => (1, "inner"),
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
}
