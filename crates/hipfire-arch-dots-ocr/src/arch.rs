//! `Architecture` trait impl for dots.ocr (Qwen2-VL family).
//!
//! Single-arch pattern — unlike qwen35-vl (which splits its trait impl
//! across two crates so the vision tower is its own arch_id), dots.ocr's
//! `DotsOcr` impl is the **outer** arch that owns BOTH the text and
//! vision sides:
//!
//! - `Config` = [`dots_ocr::DotsOcrConfig`] (Qwen2 text-config + vision
//!   sub-config).
//! - `Weights` = [`dots_ocr::DotsOcrWeights`] (Qwen2Weights + vision
//!   weights side-by-side).
//! - `State` = [`hipfire_arch_qwen2::qwen2::Qwen2State`] — the vision
//!   tower is stateless one-shot (encode patches → visual tokens → free
//!   activations), so per-step state is just the text decoder's KV
//!   cache + scratch.
//!
//! Text-side `config_from_hfq` / `load_weights` / `new_state` delegate
//! straight into [`hipfire_arch_qwen2`]. Vision-side load happens inside
//! [`dots_ocr::DotsOcrWeights::load`].
//!
//! Forward dispatch (text + vision) lives as free `pub fn` in
//! [`crate::dots_ocr`] and [`crate::image`]; static dispatch in the
//! hot path, per the trait module's design.

use crate::dots_ocr::{DotsOcrConfig, DotsOcrWeights};
use hipfire_arch_qwen2::qwen2::Qwen2State;
use hipfire_runtime::arch::{Architecture, EosFilterOverrides, LoopGuardOverrides,
                            PromptFrameOverrides, SamplerOverrides};
use hipfire_runtime::hfq::HfqFile;
use rdna_compute::Gpu;

/// Zero-sized type marker for the dots.ocr arch.
pub struct DotsOcr;

impl Architecture for DotsOcr {
    type Weights = DotsOcrWeights;
    type State = Qwen2State;
    type Config = DotsOcrConfig;

    /// arch_id = 8 for the Qwen2-VL family (dots.ocr). See
    /// `docs/architecture-ids.md`.
    fn arch_id() -> u32 {
        8
    }

    fn name() -> &'static str {
        "dots-ocr"
    }

    fn config_from_hfq(hfq: &HfqFile) -> Result<Self::Config, String> {
        DotsOcrConfig::from_hfq(hfq)
    }

    fn load_weights(
        hfq: &mut HfqFile,
        cfg: &Self::Config,
        gpu: &mut Gpu,
    ) -> Result<Self::Weights, String> {
        DotsOcrWeights::load(hfq, cfg, gpu)
    }

    fn new_state(gpu: &mut Gpu, cfg: &Self::Config) -> Result<Self::State, String> {
        // Vision tower is stateless one-shot — encode patches, emit
        // visual tokens, free activations. Per-decode-step state is
        // just the Qwen2 text decoder's KV cache + scratch graph.
        Qwen2State::new(gpu, &cfg.text)
    }

    // ── Optional overrides ────────────────────────────────────────────
    //
    // dots.ocr uses a custom (non-ChatML) chat template and a different
    // primary EOS than the qwen35 default. Both `prompt_frame_overrides`
    // and `eos_filter_overrides` MUST diverge from defaults — see §2.5
    // of the bring-up plan.

    fn loop_guard_overrides(_cfg: &Self::Config) -> LoopGuardOverrides {
        // Layout-JSON output has short repeats (category names, bracket
        // patterns) but should not exceed the default n-gram threshold.
        // Tighten only if phase-4 coherence runs trip the default.
        LoopGuardOverrides::default()
    }

    fn sampler_overrides(_cfg: &Self::Config) -> SamplerOverrides {
        // No arch-specific blocked tokens at bring-up.
        SamplerOverrides::default()
    }

    fn prompt_frame_overrides(_cfg: &Self::Config) -> PromptFrameOverrides {
        // dots.ocr's chat template uses `<|user|>...<|endofuser|>` then
        // `<|assistant|>` — NOT ChatML's `<|im_start|>` / `<|im_end|>`.
        // The minijinja renderer in `hipfire-runtime` evaluates the
        // template from HFQ metadata if present, so this override stays
        // at default (`raw=None`) and the daemon picks up the custom
        // template via `resolve_chat_template`. Phase 3 verifies the
        // Jinja path produces the right framing on a smoke prompt; if
        // not, override `raw=Some(false)` and hand-roll the framing in
        // the daemon arm.
        PromptFrameOverrides::default()
    }

    fn eos_filter_overrides(_cfg: &Self::Config) -> EosFilterOverrides {
        // Primary EOS for an assistant turn is `<|endofassistant|>`
        // (id 151673). The wire-EOS `<|endoftext|>` (151643) also
        // terminates per `generation_config.json.eos_token_id =
        // [151643, 151673]`. The default ChatML `<|im_end|>` (151645)
        // never fires on a correct dots.ocr response.
        //
        // `holdback_prefixes` must include `<|end` so the streamer
        // doesn't leak the first few bytes of either marker before
        // disambiguating which stop-at sequence is in progress.
        //
        // `strip_think = Some(false)` — dots.ocr is an OCR model, not
        // thinking-mode; it does not emit <think> blocks.
        EosFilterOverrides {
            stop_at: vec![
                b"<|endofassistant|>".to_vec(),
                b"<|endoftext|>".to_vec(),
            ],
            holdback_prefixes: vec![b"<|end".to_vec()],
            strip_think: Some(false),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dots_ocr_arch_id_and_name() {
        assert_eq!(DotsOcr::arch_id(), 8);
        assert_eq!(DotsOcr::name(), "dots-ocr");
    }

    #[test]
    fn eos_overrides_include_endofassistant() {
        // The default EOS filter doesn't catch `<|endofassistant|>` —
        // the dots.ocr-specific override must add it. Regression guard
        // against drift back to the qwen35 default.
        //
        // We can't build a real DotsOcrConfig without an HFQ, so this
        // exercises the override by passing a manually-constructed
        // minimal cfg through the static method.
        use hipfire_arch_qwen2::qwen2::Qwen2Config;
        let text_cfg = Qwen2Config {
            hidden_size: 1536,
            num_hidden_layers: 28,
            num_attention_heads: 12,
            num_key_value_heads: 2,
            head_dim: 128,
            intermediate_size: 8960,
            vocab_size: 151936,
            max_position_embeddings: 131072,
            rope_theta: 1_000_000.0,
            rms_norm_eps: 1e-6,
            attention_bias: true,
            tie_word_embeddings: false,
            eos_token_id: 151673,
            eos_token_ids: vec![151643, 151673],
        };
        let cfg = DotsOcrConfig {
            text: text_cfg,
            vision: crate::dots_ocr::DotsVisionConfig::dots_ocr_defaults(),
        };
        let eos = DotsOcr::eos_filter_overrides(&cfg);
        assert!(eos.stop_at.iter().any(|b| b == b"<|endofassistant|>"));
        assert!(eos.stop_at.iter().any(|b| b == b"<|endoftext|>"));
        assert_eq!(eos.strip_think, Some(false));
        assert!(eos.holdback_prefixes.iter().any(|b| b == b"<|end"));
    }
}
