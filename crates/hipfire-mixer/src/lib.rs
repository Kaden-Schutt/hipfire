// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! The per-layer token-mixer taxonomy and sequence-state *shape* for the
//! hipfire family seam (P2 of
//! `docs/plans/2026-06-23-seam-finish-and-mamba2.md`).
//!
//! # Why this crate exists
//!
//! hipfire grew with a monolithic, KV-cache-shaped notion of per-decode state:
//! `KvCache` (in `hipfire-runtime::kv`) holds `Vec<GpuTensor>` keyed by layer,
//! and hybrid families bolt a second parallel structure alongside it
//! (`DeltaNetState.{s_matrices, s_scales, conv_states}` in
//! `hipfire-arch-qwen35`). The honest model for the real roster — which mixes
//! full-attention, sliding-window attention, short-conv, DeltaNet, and Mamba-2
//! **per layer** — is a *heterogeneous per-layer mixer list*, not one KV cache.
//!
//! This crate owns the **neutral taxonomy** ([`MixerKind`]) and the
//! per-model [`MixerProfile`] that the serving layer queries instead of
//! branching on `arch_id` (e.g. `profile.needs_kv_cache()` replaces
//! `is_qwen35_family_arch_id(id)`). A **pure-SSM model (Mamba-2) reports
//! `needs_kv_cache() == false`** — the no-KV path falls straight out of the
//! taxonomy rather than being a special case.
//!
//! ## Scope of this increment
//!
//! This is the *taxonomy keystone only*: pure, GPU-free, no-dep types so it
//! builds in the no-GPU CI subset and nothing in the hot path is touched yet.
//! The per-layer **state buffers** (KV slab / conv-state ring / DeltaNet
//! S-matrix / Mamba-2 SSM state) are migrated onto a `MixerLayerState` model
//! built on top of this in the per-arch migration phases (P3–P5, P7). The
//! buffer layouts deliberately are NOT pinned here so the migration can reuse
//! the existing optimized `KvCache` / `DeltaNetState` allocations rather than
//! reallocating against a premature shape.

use serde::{Deserialize, Serialize};

/// The token-mixer kind of a single decoder layer.
///
/// This is the per-layer axis the heterogeneous layer stack is built from
/// (the FFN axis — dense SwiGLU / GeGLU / MoE — is orthogonal and lives with
/// the arch's FFN types). Selected per layer from the family's hybrid pattern
/// (`hybrid_override_pattern`, `layer_types`, qwen35's LA/FA split).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum MixerKind {
    /// Dense full causal attention (KV cache over the whole context).
    FullAttn,
    /// Sliding-window attention — KV cache bounded to `window` positions.
    Swa { window: usize },
    /// Depthwise causal short convolution (Mamba/LFM2 style); recurrent
    /// `kernel - 1` conv-state, no KV.
    ShortConv { kernel: usize },
    /// Gated DeltaNet linear attention (qwen35); recurrent S-matrix +
    /// short-conv state, no KV.
    DeltaNet,
    /// Mamba-2 selective SSM (SSD); recurrent SSM state + short-conv state,
    /// no KV.
    Mamba2,
}

impl MixerKind {
    /// Does this mixer maintain a KV cache? Only the attention variants do;
    /// recurrent mixers (short-conv, DeltaNet, Mamba-2) carry fixed-size
    /// recurrent state instead.
    pub fn uses_kv(self) -> bool {
        matches!(self, MixerKind::FullAttn | MixerKind::Swa { .. })
    }

    /// Does this mixer carry fixed-size recurrent state (advanced one step per
    /// token) rather than a growing KV cache?
    pub fn is_recurrent(self) -> bool {
        matches!(
            self,
            MixerKind::ShortConv { .. } | MixerKind::DeltaNet | MixerKind::Mamba2
        )
    }

    /// Does this mixer keep a depthwise short-conv state ring? True for the
    /// Mamba/LFM2/DeltaNet short conv that precedes the recurrence.
    pub fn uses_short_conv(self) -> bool {
        matches!(
            self,
            MixerKind::ShortConv { .. } | MixerKind::DeltaNet | MixerKind::Mamba2
        )
    }

    /// Stable lowercase tag for logs / config round-trips.
    pub fn tag(self) -> &'static str {
        match self {
            MixerKind::FullAttn => "full_attn",
            MixerKind::Swa { .. } => "swa",
            MixerKind::ShortConv { .. } => "short_conv",
            MixerKind::DeltaNet => "delta_net",
            MixerKind::Mamba2 => "mamba2",
        }
    }
}

/// The per-model layer-stack profile: the ordered list of per-layer mixer
/// kinds. The serving layer queries this for state-allocation decisions
/// instead of branching on `arch_id`.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct MixerProfile {
    /// One entry per decoder layer, in layer order.
    pub layers: Vec<MixerKind>,
}

impl MixerProfile {
    /// Build a profile from a per-layer kind list.
    pub fn new(layers: Vec<MixerKind>) -> Self {
        Self { layers }
    }

    /// Homogeneous profile — every layer is the same mixer kind (pure-attn
    /// llama/qwen2/gemma3-text, or pure-SSM mamba2).
    pub fn uniform(kind: MixerKind, n_layers: usize) -> Self {
        Self {
            layers: vec![kind; n_layers],
        }
    }

    /// Total decoder layers.
    pub fn n_layers(&self) -> usize {
        self.layers.len()
    }

    /// Does this model need a KV cache at all? **False for a pure-SSM model**
    /// (Mamba-2, pure DeltaNet, all-short-conv) — the no-KV path. True for any
    /// model with at least one attention layer (including hybrids like
    /// nemotron_h and qwen35).
    pub fn needs_kv_cache(&self) -> bool {
        self.layers.iter().any(|m| m.uses_kv())
    }

    /// Does any layer carry recurrent state (short-conv / DeltaNet / Mamba-2)?
    /// True for pure-SSM and for hybrids; false for a pure-transformer stack.
    pub fn has_recurrent_state(&self) -> bool {
        self.layers.iter().any(|m| m.is_recurrent())
    }

    /// A hybrid stack mixes attention and recurrent mixers (qwen35 LA/FA,
    /// nemotron_h, lfm2). Pure-attn and pure-SSM stacks are not hybrid.
    pub fn is_hybrid(&self) -> bool {
        self.needs_kv_cache() && self.has_recurrent_state()
    }

    /// Count layers whose mixer satisfies `pred` (e.g. how many KV layers to
    /// size the cache, how many recurrent layers to size SSM/conv state).
    pub fn count<F: Fn(MixerKind) -> bool>(&self, pred: F) -> usize {
        self.layers.iter().copied().filter(|&m| pred(m)).count()
    }

    /// Per-layer boolean: does layer `i` keep a KV cache? This is the mask the
    /// KV allocator consumes to skip recurrent (no-KV) layers in a hybrid stack
    /// — e.g. qwen35 allocates KV only for its `FullAttention` layers and skips
    /// the DeltaNet (`LinearAttention`) ones. For a pure-SSM model this is all
    /// `false`; for a pure-transformer stack, all `true`.
    pub fn kv_layer_mask(&self) -> Vec<bool> {
        self.layers.iter().map(|m| m.uses_kv()).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pure_ssm_needs_no_kv() {
        let p = MixerProfile::uniform(MixerKind::Mamba2, 24);
        assert!(!p.needs_kv_cache(), "pure mamba2 must not allocate KV");
        assert!(p.has_recurrent_state());
        assert!(!p.is_hybrid());
        assert_eq!(p.count(MixerKind::is_recurrent), 24);
    }

    #[test]
    fn pure_attn_needs_kv_no_recurrence() {
        let p = MixerProfile::uniform(MixerKind::FullAttn, 32);
        assert!(p.needs_kv_cache());
        assert!(!p.has_recurrent_state());
        assert!(!p.is_hybrid());
    }

    #[test]
    fn qwen35_style_hybrid() {
        // 3 DeltaNet (LA) layers : 1 FullAttn layer, repeated.
        let mut layers = Vec::new();
        for _ in 0..6 {
            layers.extend([
                MixerKind::DeltaNet,
                MixerKind::DeltaNet,
                MixerKind::DeltaNet,
                MixerKind::FullAttn,
            ]);
        }
        let p = MixerProfile::new(layers);
        assert!(p.is_hybrid());
        assert!(p.needs_kv_cache());
        assert_eq!(p.count(|m| m == MixerKind::FullAttn), 6);
        assert_eq!(p.count(MixerKind::is_recurrent), 18);
    }

    #[test]
    fn kv_layer_mask_matches_uses_kv() {
        // qwen35-style 3:1 DeltaNet:FullAttn block.
        let p = MixerProfile::new(vec![
            MixerKind::DeltaNet,
            MixerKind::DeltaNet,
            MixerKind::DeltaNet,
            MixerKind::FullAttn,
        ]);
        assert_eq!(p.kv_layer_mask(), vec![false, false, false, true]);
        // pure-SSM: no KV layers at all.
        assert_eq!(
            MixerProfile::uniform(MixerKind::Mamba2, 3).kv_layer_mask(),
            vec![false, false, false]
        );
    }

    #[test]
    fn kind_predicates_are_consistent() {
        for k in [
            MixerKind::FullAttn,
            MixerKind::Swa { window: 512 },
            MixerKind::ShortConv { kernel: 4 },
            MixerKind::DeltaNet,
            MixerKind::Mamba2,
        ] {
            // every kind is exactly one of KV-using xor recurrent
            assert_ne!(k.uses_kv(), k.is_recurrent(), "{k:?}");
        }
    }
}
