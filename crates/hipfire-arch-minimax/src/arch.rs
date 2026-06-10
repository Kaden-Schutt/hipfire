// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! `Architecture` trait impl for MiniMax-M2 (arch_id = 10).
//!
//! Thin marker + delegation, mirroring `hipfire-arch-deepseek4`'s
//! `arch.rs`. The forward pass is NOT on the trait — it lives as free
//! functions in `crate::forward` (hot-path static dispatch), called
//! directly by the daemon's `arch_id == 10` generate branch.

use crate::minimax::{MiniMaxConfig, MiniMaxState, MiniMaxWeights};
use hipfire_runtime::arch::Architecture;
use hipfire_runtime::hfq::HfqFile;
use rdna_compute::Gpu;

/// Zero-sized type marker for MiniMax-M2. Trait dispatch uses the type.
pub struct MiniMaxM2;

impl Architecture for MiniMaxM2 {
    type Weights = MiniMaxWeights;
    type State = MiniMaxState;
    type Config = MiniMaxConfig;

    /// Canonical family marker. Reserved in docs/architecture-ids.md
    /// (next free after DeepSeek V4 = 9).
    fn arch_id() -> u32 {
        10
    }

    fn name() -> &'static str {
        "minimax"
    }

    fn config_from_hfq(hfq: &HfqFile) -> Result<Self::Config, String> {
        MiniMaxConfig::from_hfq(hfq)
    }

    fn load_weights(
        hfq: &mut HfqFile,
        cfg: &Self::Config,
        gpu: &mut Gpu,
    ) -> Result<Self::Weights, String> {
        MiniMaxWeights::load(hfq, cfg, gpu, None)
    }

    fn new_state(gpu: &mut Gpu, cfg: &Self::Config) -> Result<Self::State, String> {
        MiniMaxState::new(gpu, cfg)
    }

    // Optional overrides left at the ChatML defaults for the scaffold.
    // MiniMax-M2 ships its own chat template; revisit prompt_frame /
    // eos_filter overrides once tokenizer wiring lands.
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn minimax_arch_id_and_name() {
        assert_eq!(MiniMaxM2::arch_id(), 10);
        assert_eq!(MiniMaxM2::name(), "minimax");
    }
}
