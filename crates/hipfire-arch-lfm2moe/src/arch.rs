// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! `Architecture` trait impl for LFM2.5-MoE (arch_id = 11).
//!
//! Thin marker + delegation, mirroring `hipfire-arch-minimax`'s
//! `arch.rs`. The forward pass is NOT on the trait — it lives as free
//! functions in `crate::forward` (hot-path static dispatch), called
//! directly by the daemon's `arch_id == 11` generate branch.

use crate::config::Lfm2MoeConfig;
use crate::lfm2moe::{Lfm2MoeState, Lfm2MoeWeights};
use hipfire_runtime::arch::Architecture;
use hipfire_runtime::hfq::HfqFile;
use rdna_compute::Gpu;

/// Zero-sized type marker for LFM2.5-MoE. Trait dispatch uses the type.
pub struct Lfm2Moe;

impl Architecture for Lfm2Moe {
    type Weights = Lfm2MoeWeights;
    type State = Lfm2MoeState;
    type Config = Lfm2MoeConfig;

    /// Canonical architecture ID for LFM2.5-MoE.
    fn arch_id() -> u32 {
        11
    }

    fn name() -> &'static str {
        "lfm2moe"
    }

    fn config_from_hfq(hfq: &HfqFile) -> Result<Self::Config, String> {
        Lfm2MoeConfig::from_hfq(hfq)
    }

    fn load_weights(
        hfq: &mut HfqFile,
        cfg: &Self::Config,
        gpu: &mut Gpu,
    ) -> Result<Self::Weights, String> {
        Lfm2MoeWeights::load(hfq, cfg, gpu)
    }

    fn new_state(gpu: &mut Gpu, cfg: &Self::Config) -> Result<Self::State, String> {
        Lfm2MoeState::new(gpu, cfg)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lfm2moe_arch_id_and_name() {
        assert_eq!(Lfm2Moe::arch_id(), 11);
        assert_eq!(Lfm2Moe::name(), "lfm2moe");
    }
}
