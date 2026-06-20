// SPDX-License-Identifier: Apache-2.0
// hipfire — Gemma3 Architecture + SimpleAr serving impl. See LICENSE / NOTICE.

//! `Architecture` bring-up triple for Gemma3 (`arch_id = 12`) and the
//! `Gemma3Backend: SimpleAr` serving impl (the dense-AR output strategy seed
//! for E2's `ServingBackend` seam). Mirrors `hipfire-arch-qwen2::arch`.

use hipfire_runtime::arch::{
    run_simple_ar, ArchCaps, Architecture, EosFilterOverrides, GenerateCtx, PromptFrameOverrides,
    ServeOutcome, ServingBackend, SimpleAr,
};
use hipfire_runtime::hfq::HfqFile;
use hipfire_runtime::tokenizer::Tokenizer;
use rdna_compute::{Gpu, GpuTensor};

use crate::config::Gemma3Config;
use crate::forward::{forward_step, Gemma3State};
use crate::weights::Gemma3Weights;

/// Zero-sized marker for the Gemma3 text family. `arch_id = 12` covers
/// `gemma3_text` (Gemma3ForCausalLM) and the text tower of `gemma3`
/// (Gemma3ForConditionalGeneration); vision is a separate adapter (arch_id 13).
pub struct Gemma3;

impl Architecture for Gemma3 {
    type Weights = Gemma3Weights;
    type State = Gemma3State;
    type Config = Gemma3Config;

    fn arch_id() -> u32 {
        12
    }

    fn name() -> &'static str {
        "gemma3"
    }

    fn config_from_hfq(hfq: &HfqFile) -> Result<Self::Config, String> {
        crate::config::config_from_hfq(hfq)
            .ok_or_else(|| "gemma3: failed to parse Gemma3Config from HFQ metadata".to_string())
    }

    fn load_weights(
        hfq: &mut HfqFile,
        cfg: &Self::Config,
        gpu: &mut Gpu,
    ) -> Result<Self::Weights, String> {
        Gemma3Weights::load(hfq, cfg, gpu)
    }

    fn new_state(gpu: &mut Gpu, cfg: &Self::Config) -> Result<Self::State, String> {
        Gemma3State::new(gpu, cfg)
    }

    // Gemma is NOT ChatML: it frames turns with `<start_of_turn>user …
    // <end_of_turn>\n<start_of_turn>model\n` and a leading `<bos>`. The daemon's
    // Jinja-template path (using the embedded chat_template) is the correct
    // framing once gemma3 is wired there (E2/E3); leave the ChatML default off.
    fn prompt_frame_overrides(_cfg: &Self::Config) -> PromptFrameOverrides {
        PromptFrameOverrides::default()
    }

    // medgemma-it / gemma3-it stop on `<end_of_turn>` (a special token the
    // tokenizer resolves to the model's EOS id); no `<think>` blocks.
    fn eos_filter_overrides(_cfg: &Self::Config) -> EosFilterOverrides {
        EosFilterOverrides {
            stop_at: vec![],
            holdback_prefixes: vec![],
            strip_think: Some(false),
        }
    }
}

// ── Serving seam: SimpleAr (dense-AR output strategy) ───────────────
// See docs/plans/2026-06-19-daemon-family-seam.md + the master plan. Gemma3
// bundles config/weights/state into a backend that the daemon drives through
// the object-safe SimpleAr surface, delegating to the per-token forward_step.

/// Owns the typed Gemma3 config/weights/state behind [`SimpleAr`]. Constructed
/// by the daemon once the bring-up triple has produced the parts.
pub struct Gemma3Backend {
    pub config: Gemma3Config,
    pub weights: Gemma3Weights,
    pub state: Gemma3State,
}

impl Gemma3Backend {
    pub fn new(config: Gemma3Config, weights: Gemma3Weights, state: Gemma3State) -> Self {
        Self {
            config,
            weights,
            state,
        }
    }
}

impl SimpleAr for Gemma3Backend {
    fn prefill(&mut self, gpu: &mut Gpu, tokens: &[u32]) -> Result<(), String> {
        // No batched prefill on this bring-up path: one token at a time
        // (forward_step tracks the KV write slot in state.next_pos). The final
        // call leaves next-token logits in state.logits.
        for &t in tokens {
            forward_step(gpu, &self.weights, &self.config, &mut self.state, t)
                .map_err(|e| format!("gemma3 prefill forward_step: {e:?}"))?;
        }
        Ok(())
    }

    fn decode_step(&mut self, gpu: &mut Gpu, token: u32, pos: usize) -> Result<(), String> {
        debug_assert_eq!(
            pos, self.state.next_pos,
            "gemma3 decode pos {pos} drifted from internal next_pos {}",
            self.state.next_pos
        );
        let _ = pos;
        forward_step(gpu, &self.weights, &self.config, &mut self.state, token)
            .map_err(|e| format!("gemma3 decode forward_step: {e:?}"))
    }

    fn logits(&self) -> &GpuTensor {
        &self.state.logits
    }

    fn vocab_size(&self) -> usize {
        self.config.vocab_size
    }
}

/// Gemma3 text is a plain dense-AR family (sliding-window + global attention is
/// internal to `forward_step`): its `ServingBackend` advertises no fast-path
/// caps and delegates the loop to the shared [`run_simple_ar`]. The vision tower
/// (arch_id 13) overrides `serve` for the image-token splice; this text path is
/// the splice-free base.
impl ServingBackend for Gemma3Backend {
    fn arch_id(&self) -> u32 {
        12
    }

    fn caps(&self) -> ArchCaps {
        ArchCaps::default()
    }

    fn eos_token(&self) -> u32 {
        self.config.eos_token_id
    }

    fn serve(
        &mut self,
        gpu: &mut Gpu,
        tok: &Tokenizer,
        ctx: &mut GenerateCtx,
    ) -> Result<ServeOutcome, String> {
        // gemma3 ends a chat turn with `<end_of_turn>`, which differs from
        // `config.eos_token_id` (`<eos>`). Stop on it so generation halts at the
        // turn boundary instead of leaking `<end_of_turn>` and running on.
        let eos = tok
            .special_token_id("<end_of_turn>")
            .unwrap_or(self.config.eos_token_id);
        run_simple_ar(gpu, self, tok, eos, ctx)
    }

    fn reset_session(&mut self, _gpu: &mut Gpu, _session_id: &str) -> Result<(), String> {
        // Single-session bring-up: rewind the KV cursor (O(1); slots overwrite).
        self.state.reset();
        Ok(())
    }

    fn unload(self: Box<Self>, gpu: &mut Gpu) {
        let b = *self;
        b.weights.free_gpu(gpu);
        b.state.free_gpu(gpu);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gemma3_arch_id_and_name() {
        assert_eq!(Gemma3::arch_id(), 12);
        assert_eq!(Gemma3::name(), "gemma3");
    }
}
