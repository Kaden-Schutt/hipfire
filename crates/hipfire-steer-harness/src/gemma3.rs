// SPDX-License-Identifier: Apache-2.0
//! Gemma3 adapter — implements `ModelHarness` over `hipfire-arch-gemma3`'s
//! in-process forward (so the global steer session + `maybe_steer_block` hook
//! work directly). Modeled on the crate's `infer_gemma3` example.

use hip_bridge::HipResult;
use hipfire_arch_gemma3 as g3;
use hipfire_runtime::hfq::HfqFile;
use hipfire_runtime::tokenizer::Tokenizer;
use hipfire_steer::commit_capture;
use hipfire_steer::driver::{ModelHarness, Prompt};
use rdna_compute::Gpu;

/// Gemma `<end_of_turn>` — stops greedy generation early.
const END_OF_TURN: u32 = 106;

pub struct Gemma3Harness {
    gpu: Gpu,
    // Kept alive: weights may reference the mmap-backed HFQ.
    _hfq: HfqFile,
    cfg: g3::Gemma3Config,
    weights: g3::Gemma3Weights,
    state: g3::forward::Gemma3State,
    tok: Tokenizer,
    max_new_tokens: usize,
}

impl Gemma3Harness {
    pub fn load(
        mut gpu: Gpu,
        mut hfq: HfqFile,
        max_seq: usize,
        max_new_tokens: usize,
    ) -> Result<Self, String> {
        let cfg = g3::config_from_hfq(&hfq).ok_or("gemma3: failed to parse config")?;
        // Note: the (1+w) RMSNorm bake lands in the weights at ingest and gemma3
        // uses a plain rmsnorm kernel; `gemma_norm_offset` is informational and
        // isn't stamped for the arch_id-13 (multimodal) wrapper, so we don't gate
        // on it (model coherence is verified separately).
        let tok = Tokenizer::from_hfq_metadata(&hfq.metadata_json)
            .map_err(|e| format!("gemma3: tokenizer not found: {e}"))?;
        // arch_id 13 (Gemma3-VL multimodal wrapper) nests the text decoder under
        // "language_model."; arch_id 12 (text-only) uses bare names. We load and
        // run only the text decoder via the (hooked) text forward — the vision
        // tower and the vl serving path are bypassed.
        let prefix = if hfq.arch_id == 13 {
            "language_model."
        } else {
            ""
        };
        let weights = g3::load_weights_prefixed(&mut hfq, &cfg, &mut gpu, prefix)
            .map_err(|e| format!("gemma3: load_weights failed: {e:?}"))?;
        let state = g3::forward::Gemma3State::new_with_max_seq(&mut gpu, &cfg, max_seq, false)
            .map_err(|e| format!("gemma3: Gemma3State::new failed: {e:?}"))?;
        Ok(Self {
            gpu,
            _hfq: hfq,
            cfg,
            weights,
            state,
            tok,
            max_new_tokens,
        })
    }

    /// Gemma chat framing. Gemma has no system role, so the system prompt is
    /// prepended to the user turn (Heretic does the same for Gemma).
    fn frame(&self, p: &Prompt) -> String {
        let user = if p.system.trim().is_empty() {
            p.user.clone()
        } else {
            format!("{}\n\n{}", p.system, p.user)
        };
        format!("<start_of_turn>user\n{user}<end_of_turn>\n<start_of_turn>model\n")
    }

    /// Reset to a clean state and prefill the framed prompt. Leaves
    /// `state.logits` = the next-token logits and `state.x` = the last-token
    /// residual (which the capture/apply hook taps per block).
    fn prefill(&mut self, p: &Prompt) -> HipResult<()> {
        self.state.reset();
        let ids = self.tok.encode(&self.frame(p));
        for &t in &ids {
            g3::forward_step(&mut self.gpu, &self.weights, &self.cfg, &mut self.state, t)?;
        }
        Ok(())
    }
}

impl ModelHarness for Gemma3Harness {
    fn num_layers(&self) -> usize {
        self.cfg.num_hidden_layers
    }

    fn hidden(&self) -> usize {
        self.cfg.hidden_size
    }

    fn run_forwards(&mut self, prompts: &[Prompt]) -> HipResult<()> {
        for p in prompts {
            // The capture hook observes each block per token; after the full
            // prefill `current` holds the last-token residuals — commit folds them.
            self.prefill(p)?;
            commit_capture();
        }
        Ok(())
    }

    fn first_token_logprobs(&mut self, prompts: &[Prompt]) -> HipResult<Vec<Vec<f32>>> {
        let mut out = Vec::with_capacity(prompts.len());
        for p in prompts {
            self.prefill(p)?;
            let logits = self.gpu.download_f32(&self.state.logits)?;
            out.push(log_softmax(&logits));
        }
        Ok(out)
    }

    fn generate(&mut self, prompts: &[Prompt]) -> HipResult<Vec<String>> {
        let mut out = Vec::with_capacity(prompts.len());
        for p in prompts {
            self.prefill(p)?;
            let mut ids = Vec::with_capacity(self.max_new_tokens);
            let mut next = self
                .gpu
                .argmax_f32(&self.state.logits, self.cfg.vocab_size)?;
            if next != END_OF_TURN {
                ids.push(next);
            }
            for _ in 1..self.max_new_tokens {
                next = g3::forward_step_greedy(
                    &mut self.gpu,
                    &self.weights,
                    &self.cfg,
                    &mut self.state,
                    next,
                )?;
                if next == END_OF_TURN {
                    break;
                }
                ids.push(next);
            }
            out.push(self.tok.decode(&ids));
        }
        Ok(out)
    }
}

/// Numerically stable log-softmax over a logit row (f64 accumulation).
fn log_softmax(logits: &[f32]) -> Vec<f32> {
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let sum: f64 = logits.iter().map(|&l| ((l - max) as f64).exp()).sum();
    let log_denom = max + sum.ln() as f32;
    logits.iter().map(|&l| l - log_denom).collect()
}
