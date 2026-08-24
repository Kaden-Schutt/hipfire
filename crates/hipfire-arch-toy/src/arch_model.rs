// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! `ArchModel` impl for the toy arch's bundle.
//!
//! `hipfire_runtime::arch_model::ArchModel` is THE arch-agnostic view of a
//! loaded model, and it is what the loader stores (`Box<dyn ArchModel>`).
//! A real arch crate implements it for its bundle type — see
//! `hipfire-arch-minimax/src/arch_model.rs` for the canonical small impl.
//!
//! Six required methods: `dim`, `n_layers`, `vocab_size`, `arch_key`,
//! `kv_cache_mut`, `free_gpu`. `reset_session_state` has a working default
//! (no-op); override it when your arch owns recurrent/conv state that must
//! drop between turns.

use hipfire_runtime::arch_model::ArchModel;
use hipfire_runtime::llama::KvCache;
use rdna_compute::Gpu;

use crate::toy_model::{ToyConfig, ToyState, ToyWeights};

/// Owned toy bundle — config + weights + state + eos. This is the shape a
/// real arch's `ModelState::<Arch>` variant payload takes (e.g.
/// `MiniMaxBundle` in `hipfire-arch-minimax/src/spec_impl.rs`).
pub struct ToyBundle {
    pub config: ToyConfig,
    pub weights: ToyWeights,
    pub state: ToyState,
    pub eos_tok: u32,
}

impl ArchModel for ToyBundle {
    fn dim(&self) -> usize {
        self.config.dim
    }

    fn n_layers(&self) -> usize {
        self.config.layers
    }

    fn vocab_size(&self) -> usize {
        self.config.vocab_size
    }

    /// Short stable identifier. In a real arch this string matches the row
    /// your arch claims in `hipfire_runtime::reset_core`'s retry inventory,
    /// if it claims one.
    fn arch_key(&self) -> &'static str {
        "toy"
    }

    /// Toy owns no runtime `KvCache`, so `None`. A real dense arch returns
    /// `Some(&mut self.state.kv)`; arches that keep KV inside their own
    /// state type return `None` here too (see `MuseGlimmerBundle`).
    fn kv_cache_mut(&mut self) -> Option<&mut KvCache> {
        None
    }

    /// Toy's `weights`/`state` are host-side stubs with no GPU buffers, so
    /// teardown is a no-op. A real arch frees EVERY GPU allocation it owns
    /// here — freeing less leaks VRAM across load cycles; freeing more
    /// double-frees (see the `MiniMaxBundle` impl and PR #566's note on
    /// `ModelState::MuseGlimmer`).
    fn free_gpu(self: Box<Self>, _gpu: &mut Gpu) {}
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn toy_bundle_satisfies_arch_model() {
        // Pins the trait surface the loader consumes: if a required method
        // is added or renamed, this fails to compile.
        let bundle = ToyBundle {
            config: ToyConfig {
                vocab_size: 256,
                dim: 8,
                layers: 1,
            },
            weights: ToyWeights {
                embeddings: vec![0.0; 256 * 8],
            },
            state: ToyState { token_count: 0 },
            eos_tok: 1,
        };
        assert_eq!(bundle.dim(), 8);
        assert_eq!(bundle.n_layers(), 1);
        assert_eq!(bundle.vocab_size(), 256);
        assert_eq!(bundle.arch_key(), "toy");
        // Object safety: the loader stores these boxed.
        let _: Box<dyn ArchModel> = Box::new(bundle);
    }
}
