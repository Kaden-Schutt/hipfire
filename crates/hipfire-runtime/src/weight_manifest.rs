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
use hipfire_hardware::{CollectiveHint, DimKind};
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
        _ => None,
    }
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
    /// Aliases an already-placed tensor by name (tied lm_head / embeddings).
    Tied { source: String },
    /// Pinned to a mesh-derived non-layer location (embed / output).
    Pin(PinTarget),
    /// TP logit sharding of lm_head along the vocab `axis`.
    VocabShard { axis: usize },
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
pub struct WeightEntry {
    pub name: String,
    pub layer: Option<usize>,
    pub logical_shape: Vec<usize>,
    pub dtype: DType,
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
        Self {
            name: name.into(),
            layer: None,
            logical_shape,
            dtype,
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
        Self {
            name: name.into(),
            layer: Some(layer),
            logical_shape,
            dtype,
            policy,
        }
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
}
