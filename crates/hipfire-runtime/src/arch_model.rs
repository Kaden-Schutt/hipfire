// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! The architecture-agnostic view of a loaded model.
//!
//! ## Why this exists
//!
//! `hipfire_loader::ModelState` is a closed enum with one variant per
//! architecture. Every crate that needed a scalar off a loaded model therefore
//! had to name all eleven — including the product entry point, which computed
//! *three integers* through a seven-arm architecture dispatch:
//!
//! ```text
//! let (dim, layers, vocab) = match m.state.as_ref() {
//!     Some(ModelState::Qwen35(b)) => (b.config.dim,         b.config.n_layers,          b.config.vocab_size),
//!     Some(ModelState::Qwen2(b))  => (b.config.hidden_size, b.config.num_hidden_layers, b.config.vocab_size),
//!     …
//! ```
//!
//! Seven arms, one tuple, and the only real difference is that some configs
//! spell it `dim`/`n_layers` and others `hidden_size`/`num_hidden_layers`. A
//! naming inconsistency was being paid for with architecture dispatch in the
//! daemon.
//!
//! Measured before this trait existed: the loader and daemon between them held
//! 93 `ModelState::` references, but touched only **seven distinct members** of
//! the bundles they unwrapped — `config` (26 hits, only ever for those three
//! scalars), `state`, `reset_session_state`, `kv_cache`/`kv`, `dn_state` (since
//! deleted as vestigial) and `weights` (free-on-unload). That is the whole
//! surface, and it is what this trait exposes.
//!
//! ## Why it lives in `hipfire-runtime`
//!
//! `hipfire-loader` depends on every `hipfire-arch-*` crate; the arch crates
//! must not depend on the loader. A trait that arch crates implement and the
//! loader consumes therefore cannot live in the loader — that is a cycle. It
//! also cannot live in `saddle-core`, which sits below the runtime and must not
//! know about `KvCache`. `hipfire-runtime` is the one layer both sides already
//! depend on, so it is where the contract belongs.
//!
//! ## What this is NOT
//!
//! Not a forward-pass abstraction. Generation stays in `hipfire-generate`,
//! which is the architecture composition root by design and legitimately names
//! arch crates. This trait exists so that *infrastructure* — load
//! acknowledgement, session reset, unload — stops branching on architecture.

use crate::llama::KvCache;
use rdna_compute::Gpu;

/// A loaded model, viewed without knowing its architecture.
///
/// Implemented by each architecture's bundle type in its own crate. The loader
/// stores `Box<dyn ArchModel>` so that adding an architecture does not edit a
/// closed enum, and the daemon asks questions instead of matching variants.
pub trait ArchModel: Send + std::any::Any {
    /// Hidden size. Spelled `dim` by some configs and `hidden_size` by others;
    /// the implementor resolves that, not the caller.
    fn dim(&self) -> usize;

    /// Number of decoder layers (`n_layers` / `num_hidden_layers`).
    fn n_layers(&self) -> usize;

    /// Vocabulary size.
    fn vocab_size(&self) -> usize;

    /// Short stable identifier, e.g. `"qwen35"`. Matches the key used by
    /// [`crate::reset_core`]'s inventory so the two cannot drift.
    fn arch_key(&self) -> &'static str;

    /// The model's KV cache, when it owns one directly.
    ///
    /// `None` is legitimate: some bundles keep the cache elsewhere, and callers
    /// must treat absence as "not applicable", never as an error.
    fn kv_cache_mut(&mut self) -> Option<&mut KvCache>;

    /// Drop per-session state so the next turn starts clean — recurrent state,
    /// conv rings, cache offsets. Position and conversation history are the
    /// caller's concern, not the model's.
    ///
    /// Default is a no-op because a pure-attention model with no recurrent
    /// state has nothing to reset, and forcing every implementor to write an
    /// empty body would obscure the ones that genuinely do work here.
    fn reset_session_state(&mut self, _gpu: &mut Gpu) -> Result<(), String> {
        Ok(())
    }

    /// Downcast hatch for the architecture composition root.
    ///
    /// `hipfire-generate` legitimately needs the concrete bundle to call a
    /// per-architecture forward pass — that is what a composition root does.
    /// This exists so it can keep doing that once `ModelState` is replaced by
    /// `Box<dyn ArchModel>`.
    ///
    /// Crucially it borrows only the receiver, so a caller can hold the
    /// downcast bundle and a disjoint `LoadedModel` field at the same time.
    /// A whole-struct accessor cannot: that distinction is why the accessor
    /// experiment converted 15 sites of 154 and this hatch is expected to do
    /// better.

    /// Validate that teardown on `gpu` would target the correct physical device.
    ///
    /// Read-only, performs zero GPU calls. Returning `Err` signals a wrong-device
    /// mismatch and allows the caller to preserve the model for retry on the
    /// correct device. The default is success so non-pilot arches need no change.
    fn validate_teardown_device(&self, _device_id: i32) -> Result<(), String> {
        Ok(())
    }

    /// Return every GPU buffer this model owns.
    ///
    /// Consumes the box: unload is terminal, and taking `self` by value makes
    /// use-after-free a compile error rather than a runtime one.
    fn free_gpu(self: Box<Self>, gpu: &mut Gpu);
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A bundle with no recurrent state and no directly-owned cache still
    /// satisfies the contract without writing either method — that is the point
    /// of the defaults, and a regression here would force boilerplate into
    /// every pure-attention arch crate.
    struct Minimal {
        dim: usize,
    }

    impl ArchModel for Minimal {
        fn dim(&self) -> usize {
            self.dim
        }
        fn n_layers(&self) -> usize {
            2
        }
        fn vocab_size(&self) -> usize {
            32
        }
        fn arch_key(&self) -> &'static str {
            "minimal"
        }
        fn kv_cache_mut(&mut self) -> Option<&mut KvCache> {
            None
        }
        fn free_gpu(self: Box<Self>, _gpu: &mut Gpu) {}
    }

    #[test]
    fn defaults_cover_a_stateless_arch() {
        let m = Minimal { dim: 8 };
        assert_eq!(m.dim(), 8);
        assert_eq!(m.arch_key(), "minimal");
    }

    #[test]
    fn trait_is_object_safe() {
        // The loader stores these behind a box; if this stops compiling the
        // whole design is void, so pin it rather than discovering it later.
        let m: Box<dyn ArchModel> = Box::new(Minimal { dim: 4 });
        assert_eq!(m.n_layers(), 2);
        assert_eq!(m.vocab_size(), 32);
    }
}
