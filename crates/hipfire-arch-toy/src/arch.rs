// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! `Architecture` trait impl for the toy arch — minimum-viable reference.
//!
//! This file is the *shape* a new arch's `arch.rs` should take: a
//! zero-sized type marker (`Toy`), `impl Architecture for Toy`, and
//! every required method delegating into the arch's own model module
//! (here `toy_model`). Every method is a one-liner with a doc-comment
//! explaining what a real arch would do.
//!
//! For a fully wired production reference, read
//! `crates/hipfire-arch-qwen35/src/arch.rs`.

use crate::toy_model::{ToyConfig, ToyState, ToyWeights};
use hipfire_runtime::arch::{Architecture, EosFilterOverrides, LoopGuardOverrides,
                            PromptFrameOverrides, SamplerOverrides};
use hipfire_runtime::hfq::HfqFile;
use rdna_compute::Gpu;

/// Type marker for the toy arch. Zero-sized — no per-instance state.
/// A real arch's marker is exactly this shape (e.g. `Qwen35`, `Llama`).
/// Trait dispatch uses the type, not a value.
pub struct Toy;

impl Architecture for Toy {
    type Weights = ToyWeights;
    type State = ToyState;
    type Config = ToyConfig;

    /// Pick an unused `arch_id` for a real arch — reserve one in
    /// `docs/architecture-ids.md` if/when it lands. Existing IDs:
    /// 0 = LLaMA / Mistral, 1 = plain Qwen3 / Qwen2, 5 = Qwen3.5 dense,
    /// 6 = Qwen3.5/3.6 MoE. The `arch_id` returned here is the canonical
    /// marker for the family; the actual id loaded at runtime lives on
    /// `HfqFile::arch_id` and is dispatched by the daemon.
    fn arch_id() -> u32 {
        // 0xFF = "toy / reserved for the template". Never ship an HFQ
        // file with this arch_id; the daemon will not dispatch it.
        0xFF
    }

    fn name() -> &'static str {
        "toy"
    }

    /// In a real arch: parse model-shape constants out of
    /// `hfq.metadata_json` and emit a typed `Config`. See
    /// `hipfire_arch_qwen35::qwen35::config_from_hfq` for the pattern.
    fn config_from_hfq(hfq: &HfqFile) -> Result<Self::Config, String> {
        ToyConfig::from_hfq(hfq)
    }

    /// In a real arch: read every weight tensor out of `hfq`, upload
    /// to GPU memory in the appropriate quant format, and assemble
    /// per-layer `WeightTensor` arrays. See
    /// `hipfire_arch_qwen35::qwen35::load_weights` for the full pattern.
    /// The weight pager (lazy-load + LRU eviction) is wired through
    /// `WeightTensor` and is not arch-specific.
    fn load_weights(
        hfq: &mut HfqFile,
        cfg: &Self::Config,
        _gpu: &mut Gpu,
    ) -> Result<Self::Weights, String> {
        ToyWeights::load(hfq, cfg)
    }

    /// In a real arch: allocate GPU scratch buffers (KV cache, attention
    /// scratch, MoE expert workspace, recurrent state) sized by `cfg`.
    /// See `DeltaNetState::new` in `hipfire-arch-qwen35` (hybrid LA + FA)
    /// and `ForwardScratch::new` in `hipfire-runtime::llama` (dense FA).
    fn new_state(_gpu: &mut Gpu, cfg: &Self::Config) -> Result<Self::State, String> {
        ToyState::new(cfg)
    }

    // ── Optional overrides ────────────────────────────────────────────
    //
    // The trait's defaults assume Qwen3.5 family conventions (ChatML
    // framing with `<|im_start|>` / `<|im_end|>` markers, `<think>`
    // suppression, default n-gram thresholds, default sampler config).
    // Override only what diverges for your arch. The four override
    // structs are short enough to inline here as documentation;
    // see `hipfire_runtime::arch` for full field-level docs.

    /// Loop-guard overrides: tighten or loosen n-gram block thresholds.
    /// Example for a base model that legitimately repeats short phrases:
    /// `LoopGuardOverrides { ngram_threshold: Some(8), ngram_window: Some(256) }`.
    fn loop_guard_overrides(_cfg: &Self::Config) -> LoopGuardOverrides {
        LoopGuardOverrides::default()
    }

    /// Sampler overrides: per-arch blocked tokens and repeat-penalty.
    /// Example for an arch that uses a custom `<tool_call>` opener at
    /// token 99999:
    /// `SamplerOverrides { blocked_tokens: vec![99999], repeat_penalty: None }`.
    fn sampler_overrides(_cfg: &Self::Config) -> SamplerOverrides {
        SamplerOverrides::default()
    }

    /// Prompt-frame overrides: control assistant-prefix scheme.
    /// Example for a non-ChatML arch (raw-text completion model):
    /// `PromptFrameOverrides { raw: Some(true) }`.
    fn prompt_frame_overrides(_cfg: &Self::Config) -> PromptFrameOverrides {
        PromptFrameOverrides::default()
    }

    /// EOS-filter overrides: per-arch end-of-turn markers and visible-
    /// stream policy. Example for Gemma's `<end_of_turn>`:
    /// `EosFilterOverrides {
    ///     stop_at: vec![b"<end_of_turn>".to_vec()],
    ///     holdback_prefixes: vec![b"<end_".to_vec()],
    ///     strip_think: Some(false),
    /// }`.
    fn eos_filter_overrides(_cfg: &Self::Config) -> EosFilterOverrides {
        EosFilterOverrides::default()
    }
}

// ── Tests ────────────────────────────────────────────────────────────────
//
// Trivial sanity checks that the trait impl compiles and the stub
// values match the documentation. Real arches add far more (forward-
// pass golden-output tests against a CPU reference, KV-state shape
// checks, etc.). See `hipfire_arch_qwen35`'s test files for examples.
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn toy_arch_id_is_reserved() {
        assert_eq!(Toy::arch_id(), 0xFF);
        assert_eq!(Toy::name(), "toy");
    }
}
