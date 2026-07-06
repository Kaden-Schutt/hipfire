// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! `Architecture` trait impl for Cohere2-MoE (North-Mini-Code). arch_id = 12
//! (next free after lfm2/lfm2_moe = 11; see docs/architecture-ids.md).
//!
//! Maps onto hipfire's existing kernels with NO new GPU kernels: GQA +
//! interleaved (GPT-J) RoPE (skipped on the global/NoPE layers; sliding layers
//! use the windowed flash path) + Q8 KV attention, `rmsnorm_batched` (RMSNorm at
//! `rms_norm_eps` — NOT base Cohere2's mean-centered LayerNorm), `moe_topk_renorm_k8`
//! (sigmoid scoring, `norm_topk_prob=false`), and the qwen35/lfm2/minimax
//! indexed-MoE GEMV family for the MQ4/MQ6 expert tiers.

use crate::cohere2moe::{Cohere2MoeState, Cohere2MoeWeights};
use crate::config::Cohere2MoeConfig;
use hipfire_runtime::arch::{Architecture, EosFilterOverrides};
use hipfire_runtime::hfq::HfqFile;
use rdna_compute::Gpu;

/// Type marker for the Cohere2-MoE family (`arch_id = 12`).
pub struct Cohere2Moe;

impl Architecture for Cohere2Moe {
    type Weights = Cohere2MoeWeights;
    type State = Cohere2MoeState;
    type Config = Cohere2MoeConfig;

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
        Cohere2MoeWeights::load(hfq, cfg, gpu)
    }

    fn new_state(gpu: &mut Gpu, cfg: &Self::Config) -> Result<Self::State, String> {
        Cohere2MoeState::new(gpu, cfg)
    }

    /// Cohere2 ends a turn with `<|END_OF_TURN_TOKEN|>` (id 255001), not the
    /// ChatML `<|im_end|>`. Stop the visible stream on those bytes. (Prompt
    /// framing for the Cohere multi-template chat format is a serving-time
    /// follow-up; the KLD/PPL harness feeds raw token ids and bypasses it.)
    fn eos_filter_overrides(_cfg: &Self::Config) -> EosFilterOverrides {
        EosFilterOverrides {
            stop_at: vec![b"<|END_OF_TURN_TOKEN|>".to_vec()],
            holdback_prefixes: vec![b"<|END_OF_TURN_TOKEN".to_vec(), b"<|".to_vec()],
            strip_think: Some(false),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn arch_id_and_name() {
        assert_eq!(Cohere2Moe::arch_id(), 12);
        assert_eq!(Cohere2Moe::name(), "cohere2moe");
    }
}
