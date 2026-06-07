// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! `Architecture` trait impl for Cohere2Moe (arch_id = 12).
//!
//! Thin marker + delegation, mirroring `hipfire-arch-minimax`'s `arch.rs`. The
//! forward pass is NOT on the trait — it lives as free functions in
//! `crate::forward` (hot-path static dispatch), to be called directly by the
//! daemon's `arch_id == 12` generate branch (daemon wiring is a follow-up; see
//! NEXT-STEPS.md).

use crate::cohere2moe::{CohereState, CohereWeights};
use crate::config::Cohere2MoeConfig;
use hipfire_runtime::arch::{Architecture, PromptFrameOverrides};
use hipfire_runtime::hfq::HfqFile;
use rdna_compute::Gpu;

/// Zero-sized type marker for Cohere2Moe (CohereLabs BLS-Mini-Code family).
pub struct Cohere2Moe;

impl Architecture for Cohere2Moe {
    type Weights = CohereWeights;
    type State = CohereState;
    type Config = Cohere2MoeConfig;

    /// Canonical family marker. Reserved in docs/architecture-ids.md (next free
    /// after LFM2.5-MoE = 11).
    fn arch_id() -> u32 {
        12
    }

    fn name() -> &'static str {
        "cohere2moe"
    }

    fn config_from_hfq(hfq: &HfqFile) -> Result<Self::Config, String> {
        Cohere2MoeConfig::from_hfq(hfq)
    }

    fn load_weights(
        hfq: &mut HfqFile,
        cfg: &Self::Config,
        gpu: &mut Gpu,
    ) -> Result<Self::Weights, String> {
        CohereWeights::load(hfq, cfg, gpu)
    }

    fn new_state(gpu: &mut Gpu, cfg: &Self::Config) -> Result<Self::State, String> {
        CohereState::new(gpu, cfg)
    }

    /// Cohere chat uses its own template (`<|START_OF_TURN_TOKEN|>` etc.), not
    /// ChatML. Until the tokenizer/template wiring lands, the oracle + dump
    /// paths drive raw token ids directly, so prompt framing is irrelevant for
    /// bring-up. Revisit once serve wiring lands.
    fn prompt_frame_overrides(_cfg: &Self::Config) -> PromptFrameOverrides {
        PromptFrameOverrides::default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cohere2moe_arch_id_and_name() {
        assert_eq!(Cohere2Moe::arch_id(), 12);
        assert_eq!(Cohere2Moe::name(), "cohere2moe");
    }
}
