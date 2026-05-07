//! `Architecture` trait impl for Zaya.
//!
//! Phase 1 scaffold: every method either returns a typed default
//! (arch_id, name, overrides) or returns Err with a pointer to the
//! intake docs (config_from_hfq, load_weights, new_state).
//!
//! See `crates/hipfire-arch-toy/src/arch.rs` for the template, and
//! `crates/hipfire-arch-qwen35/src/arch.rs` for the production reference.

use crate::config::ZayaConfig;
use crate::state::ZayaState;
use crate::weights::ZayaWeights;
use hipfire_runtime::arch::{
    Architecture, EosFilterOverrides, LoopGuardOverrides, PromptFrameOverrides, SamplerOverrides,
};
use hipfire_runtime::hfq::HfqFile;
use rdna_compute::Gpu;

/// Type marker for the Zaya arch.
///
/// Zero-sized; trait dispatch is on the type, not a value. The actual
/// per-instance state lives in `ZayaState`, weights in `ZayaWeights`,
/// shape in `ZayaConfig`.
pub struct Zaya;

impl Architecture for Zaya {
    type Weights = ZayaWeights;
    type State = ZayaState;
    type Config = ZayaConfig;

    /// Reserved arch_id for the Zaya family. Existing IDs:
    /// 0 = LLaMA / Mistral, 1 = plain Qwen3 / Qwen2,
    /// 5 = Qwen3.5 dense, 6 = Qwen3.5/3.6 MoE,
    /// 7 = Zaya (this).
    /// 0xFF reserved for the `toy` template.
    /// Update `crates/hipfire-runtime/src/arch.rs` doc-comment ladder
    /// when this lands on master.
    fn arch_id() -> u32 {
        7
    }

    fn name() -> &'static str {
        "zaya"
    }

    /// Phase 1 stub. See `ZayaConfig::from_hfq`.
    fn config_from_hfq(hfq: &HfqFile) -> Result<Self::Config, String> {
        ZayaConfig::from_hfq(hfq)
    }

    /// Phase 1 stub. See `ZayaWeights::load`.
    fn load_weights(
        hfq: &mut HfqFile,
        cfg: &Self::Config,
        _gpu: &mut Gpu,
    ) -> Result<Self::Weights, String> {
        ZayaWeights::load(hfq, cfg)
    }

    /// Phase 1 stub. See `ZayaState::new`. The CCA recurrent state
    /// allocation is the headline Phase 6 deliverable.
    fn new_state(gpu: &mut Gpu, cfg: &Self::Config) -> Result<Self::State, String> {
        ZayaState::new(gpu, cfg)
    }

    // -- Optional overrides -------------------------------------------------
    // ZAYA1's chat template uses Gemma-style `<start_of_turn>` /
    // `<end_of_turn>` markers (eos_token_id=106 confirms Gemma tokenizer).
    // Override eos_filter to match. Sampler / loop_guard / prompt_frame
    // defaults are likely fine pending live coherence-gate validation.

    fn loop_guard_overrides(_cfg: &Self::Config) -> LoopGuardOverrides {
        LoopGuardOverrides::default()
    }

    fn sampler_overrides(_cfg: &Self::Config) -> SamplerOverrides {
        SamplerOverrides::default()
    }

    fn prompt_frame_overrides(_cfg: &Self::Config) -> PromptFrameOverrides {
        // ZAYA1 ships with a chat_template that uses Gemma turn markers,
        // not ChatML `<|im_start|>`. Override after Phase 1 read of the
        // tokenizer + chat_template artifacts (to be done alongside
        // reference dump generation on hiptrx).
        PromptFrameOverrides::default()
    }

    fn eos_filter_overrides(_cfg: &Self::Config) -> EosFilterOverrides {
        // ZAYA1 uses the Gemma tokenizer (eos_token_id=106 =
        // `<end_of_turn>`). Match the Gemma4 EOS filter pattern from
        // the existing port. Bytes filled in once Phase 1 confirms the
        // tokenizer's exact byte sequence for token 106.
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
    fn zaya_arch_id_is_seven() {
        assert_eq!(Zaya::arch_id(), 7);
        assert_eq!(Zaya::name(), "zaya");
    }

    #[test]
    fn zaya_eos_uses_gemma_end_of_turn() {
        let overrides = Zaya::eos_filter_overrides(&ZayaConfig::default());
        assert_eq!(overrides.stop_at, vec![b"<end_of_turn>".to_vec()]);
    }
}
