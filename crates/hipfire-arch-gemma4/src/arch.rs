// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Kaden Schutt, Kate, Kevin Read
// hipfire — see LICENSE and NOTICE in the project root.

//! `Architecture` trait impl for Gemma 4 dense text (arch_id = 13).
//!
//! Thin marker + delegation, mirroring `hipfire-arch-minimax`'s `arch.rs`. The
//! forward pass is NOT on the trait — it lives as free functions in
//! `crate::forward` (hot-path static dispatch).

use crate::config::Gemma4Config;
use crate::gemma4::{Gemma4State, Gemma4Weights};
use hipfire_runtime::arch::{Architecture, EosFilterOverrides};
use hipfire_runtime::hfq::HfqFile;
use rdna_compute::Gpu;

/// Canonical arch_id for the Gemma 4 dense text family.
pub const ARCH_ID: u32 = 13;

/// Zero-sized type marker for Gemma 4 dense. Trait dispatch uses the type.
pub struct Gemma4;

impl Architecture for Gemma4 {
    type Weights = Gemma4Weights;
    type State = Gemma4State;
    type Config = Gemma4Config;

    fn arch_id() -> u32 {
        ARCH_ID
    }

    fn name() -> &'static str {
        "gemma4"
    }

    fn config_from_hfq(hfq: &HfqFile) -> Result<Self::Config, String> {
        Gemma4Config::from_hfq(hfq)
    }

    fn load_weights(
        hfq: &mut HfqFile,
        cfg: &Self::Config,
        gpu: &mut Gpu,
    ) -> Result<Self::Weights, String> {
        // The mmap-backed HfqFile is read-only at the syscall level; the loader
        // only reads tensor data, so the &mut is downgraded to &.
        Gemma4Weights::load(&*hfq, cfg, gpu)
    }

    fn new_state(gpu: &mut Gpu, cfg: &Self::Config) -> Result<Self::State, String> {
        Gemma4State::new(gpu, cfg)
    }

    /// Gemma 4 ends a turn on `<end_of_turn>` (token 106) in addition to the
    /// generic EOS. Hold back the `<end_` prefix so a partial decode doesn't
    /// leak marker bytes. Gemma is not a thinking-mode model, so no
    /// `<think>`-strip.
    fn eos_filter_overrides(_cfg: &Self::Config) -> EosFilterOverrides {
        EosFilterOverrides {
            stop_at: vec![b"<end_of_turn>".to_vec()],
            holdback_prefixes: vec![b"<end_".to_vec()],
            strip_think: Some(false),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gemma4_arch_id_and_name() {
        assert_eq!(Gemma4::arch_id(), 13);
        assert_eq!(Gemma4::name(), "gemma4");
        assert_eq!(ARCH_ID, 13);
    }
}
