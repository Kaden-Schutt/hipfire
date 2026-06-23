// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Unified per-sequence decode state — the container half of the family-seam
//! `Mixer` state model (P2c of
//! `docs/plans/2026-06-23-seam-finish-and-mamba2.md`).
//!
//! # Why
//!
//! Today a loaded model's per-decode state is a *cluster* of parallel fields on
//! `LoadedModel`: `kv_cache: Option<KvCache>` for the attention layers, plus a
//! separate `DeltaNetState` (inside the qwen35 session state) for the recurrent
//! layers, plus assorted `q35_*` bookkeeping. A hybrid stack (qwen35 FA+DeltaNet)
//! carries both; a pure-attention stack carries only KV; a **pure-SSM stack
//! (Mamba-2) carries only recurrent state and no KV at all** — but there is no
//! single object that expresses "the state for this sequence" across those cases.
//!
//! [`SequenceState`] is that object: one container, keyed by the model's
//! [`MixerProfile`], owning an optional [`KvCache`] (attention layers) and an
//! optional boxed [`RecurrentMixerState`] (short-conv / DeltaNet / Mamba-2). The
//! **no-KV path falls out**: a profile with `needs_kv_cache() == false` builds a
//! `SequenceState` with `kv == None`.
//!
//! ## Deliberately a container, not a buffer-merge
//!
//! This does NOT fold `KvCache`'s 49 quant-codec constructors and
//! `DeltaNetState`'s S-matrix/conv storage into one buffer enum — those are
//! genuinely different (attention KV codecs vs. recurrent state) and stay in
//! their own types. The unification is at the *ownership / dispatch* level.
//!
//! ## Hot path stays monomorphized
//!
//! The boxed `dyn RecurrentMixerState` is a coarse, per-model handle. Arch
//! serving code recovers the concrete state (`&DeltaNetState`) for its
//! per-token forward via [`SequenceState::recurrent_as`] (an `Any` downcast),
//! so the inner loop never pays dyn dispatch — matching the seam's "forward
//! stays monomorphized" rule.

use hipfire_mixer::MixerProfile;
use rdna_compute::{Gpu, HipResult};

use crate::kv::KvCache;

/// Neutral, object-safe handle to a mixer's fixed-size recurrent state
/// (short-conv ring / DeltaNet S-matrix / Mamba-2 SSM state). Implemented by
/// the arch crate that owns the concrete type (e.g. `DeltaNetState` in
/// `hipfire-arch-qwen35`); the serving layer holds it as
/// `Box<dyn RecurrentMixerState>` inside a [`SequenceState`].
///
/// Kept minimal on purpose — it grows only as the access-site migration
/// (Slice 3+) reveals genuinely arch-neutral operations. Arch-specific
/// per-token work is reached via [`RecurrentMixerState::as_any`] downcast, not
/// by widening this trait.
pub trait RecurrentMixerState: Send {
    /// Zero all recurrent buffers — start of a fresh sequence (or a hard
    /// session reset). Snapshot/rollback for spec-decode stays arch-specific
    /// (recovered via [`Self::as_any`]).
    fn reset(&mut self, gpu: &mut Gpu) -> HipResult<()>;

    /// Downcast hook: recover the concrete recurrent-state type on the arch's
    /// monomorphized hot path.
    fn as_any(&self) -> &dyn std::any::Any;
    /// Mutable downcast hook.
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any;
    /// Owning downcast hook: recover the concrete recurrent state by value, e.g.
    /// `*state.into_any().downcast::<DeltaNetState>().unwrap()`. Used by the
    /// transitional [`SequenceState::into_parts`] swap bridge.
    fn into_any(self: Box<Self>) -> Box<dyn std::any::Any>;
}

/// The unified per-sequence decode state for one loaded model.
pub struct SequenceState {
    /// Per-layer token-mixer taxonomy — the source of truth for which layers
    /// are KV-backed vs recurrent, and whether a KV cache is needed at all.
    pub profile: MixerProfile,
    /// KV cache for the attention layers. `None` for a pure-SSM model
    /// (`profile.needs_kv_cache() == false`).
    pub kv: Option<KvCache>,
    /// Recurrent state for the short-conv / DeltaNet / Mamba-2 layers. `None`
    /// for a pure-attention model (`profile.has_recurrent_state() == false`).
    pub recurrent: Option<Box<dyn RecurrentMixerState>>,
}

impl SequenceState {
    /// Build a sequence state from its parts. In debug builds, asserts the
    /// `kv`/`recurrent` presence matches what the profile says the stack needs
    /// — the invariant the loader must uphold.
    pub fn new(
        profile: MixerProfile,
        kv: Option<KvCache>,
        recurrent: Option<Box<dyn RecurrentMixerState>>,
    ) -> Self {
        debug_assert_eq!(
            kv.is_some(),
            profile.needs_kv_cache(),
            "SequenceState: kv presence must match profile.needs_kv_cache()"
        );
        debug_assert_eq!(
            recurrent.is_some(),
            profile.has_recurrent_state(),
            "SequenceState: recurrent presence must match profile.has_recurrent_state()"
        );
        Self {
            profile,
            kv,
            recurrent,
        }
    }

    /// Does this sequence keep a KV cache? (Attention or hybrid model.)
    pub fn has_kv(&self) -> bool {
        self.kv.is_some()
    }

    /// Shared KV cache, if any.
    pub fn kv(&self) -> Option<&KvCache> {
        self.kv.as_ref()
    }

    /// Mutable KV cache, if any.
    pub fn kv_mut(&mut self) -> Option<&mut KvCache> {
        self.kv.as_mut()
    }

    /// Recover the concrete recurrent-state type `T` (e.g. `DeltaNetState`) for
    /// the arch's monomorphized hot path. `None` if there is no recurrent state
    /// or it is a different concrete type.
    pub fn recurrent_as<T: 'static>(&self) -> Option<&T> {
        self.recurrent.as_ref()?.as_any().downcast_ref::<T>()
    }

    /// Mutable concrete recurrent-state downcast.
    pub fn recurrent_as_mut<T: 'static>(&mut self) -> Option<&mut T> {
        self.recurrent.as_mut()?.as_any_mut().downcast_mut::<T>()
    }

    /// Reset the recurrent state (no-op for a pure-attention model).
    pub fn reset_recurrent(&mut self, gpu: &mut Gpu) -> HipResult<()> {
        if let Some(r) = self.recurrent.as_mut() {
            r.reset(gpu)?;
        }
        Ok(())
    }

    /// Consume the container into its raw `(kv, recurrent)` parts. Transitional
    /// bridge for the staged migration: the still-separate `LoadedModel`
    /// `kv_cache`/`dn_state` fields are fed from these until that side is unified
    /// too (Slice 3 / P6), at which point this collapses to a single move.
    pub fn into_parts(self) -> (Option<KvCache>, Option<Box<dyn RecurrentMixerState>>) {
        (self.kv, self.recurrent)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use hipfire_mixer::MixerKind;

    /// A GPU-free mock recurrent state for container tests. (A real `KvCache`
    /// or `DeltaNetState` needs a GPU, so the container's structural logic is
    /// tested against this stand-in; the no-KV / downcast paths need no GPU.)
    struct MockRecurrent {
        tag: u32,
    }
    impl RecurrentMixerState for MockRecurrent {
        fn reset(&mut self, _gpu: &mut Gpu) -> HipResult<()> {
            Ok(())
        }
        fn as_any(&self) -> &dyn std::any::Any {
            self
        }
        fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
            self
        }
        fn into_any(self: Box<Self>) -> Box<dyn std::any::Any> {
            self
        }
    }

    #[test]
    fn pure_ssm_shape_has_no_kv() {
        // profile = all Mamba-2 → needs_kv_cache() == false → kv None is valid.
        let profile = MixerProfile::uniform(MixerKind::Mamba2, 4);
        let st = SequenceState::new(profile, None, Some(Box::new(MockRecurrent { tag: 7 })));
        assert!(!st.has_kv());
        assert!(st.kv().is_none());
        assert_eq!(st.recurrent_as::<MockRecurrent>().unwrap().tag, 7);
        // wrong concrete type → None (downcast is type-checked).
        assert!(st.recurrent_as::<KvCache>().is_none());
    }

    #[test]
    fn empty_profile_has_neither() {
        // 0-layer profile: needs_kv_cache()==false AND has_recurrent_state()==
        // false, so the both-None container is valid and the debug_asserts hold.
        let mut st = SequenceState::new(MixerProfile::new(vec![]), None, None);
        assert!(!st.has_kv());
        assert!(st.kv_mut().is_none());
        assert!(st.recurrent_as_mut::<MockRecurrent>().is_none());
    }
}
