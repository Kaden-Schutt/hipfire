// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Kaden Schutt, Kevin Read
// hipfire — see LICENSE and NOTICE in the project root.
//! `Architecture` trait implementation for Gemma 4.

use crate::gemma4::{self, Gemma4Config, Gemma4Scratch, Gemma4Weights};
use hipfire_runtime::arch::{Architecture, EosFilterOverrides, PromptFrameOverrides};
use hipfire_runtime::hfq::HfqFile;
use rdna_compute::Gpu;

/// Type marker for the Gemma 4 family (`arch_id = 12`).
///
/// Released sub-variants:
///   - `google/gemma-4-31B` / `gemma-4-31B-it` (31B dense)
///   - `google/gemma-4-26B-A4B` / `gemma-4-26B-A4B-it` (26B MoE A4B)
///   - `google/gemma-4-12B-it` (12B dense, unified audio+vision+text)
///   - `google/gemma-4-E4B-it` (8B Any-to-Any)
///   - `google/gemma-4-E2B-it` (5B Any-to-Any)
pub struct Gemma4;

impl Architecture for Gemma4 {
    type Weights = Gemma4Weights;
    type State = Gemma4Scratch;
    type Config = Gemma4Config;

    fn arch_id() -> u32 {
        12 // Gemma 4 family. Originally claimed 7 on the gemma4 branch;
           // qwen2 already occupies 7 on the dispatch-unification branch.
    }

    fn name() -> &'static str {
        "gemma4"
    }

    fn config_from_hfq(hfq: &HfqFile) -> Result<Self::Config, String> {
        gemma4::config_from_hfq(hfq)
            .ok_or_else(|| "gemma4: failed to parse config from HFQ metadata".to_string())
    }

    fn load_weights(
        hfq: &mut HfqFile,
        cfg: &Self::Config,
        gpu: &mut Gpu,
    ) -> Result<Self::Weights, String> {
        gemma4::load_weights(hfq, cfg, gpu)
            .map_err(|e| format!("gemma4: load_weights failed: {e:?}"))
    }

    fn new_state(gpu: &mut Gpu, cfg: &Self::Config) -> Result<Self::State, String> {
        Gemma4Scratch::new(gpu, cfg, 1)
            .map_err(|e| format!("gemma4: Gemma4Scratch::new failed: {e:?}"))
    }

    fn prompt_frame_overrides(_cfg: &Self::Config) -> PromptFrameOverrides {
        // Gemma 4 uses <|turn> / <turn|> framing (ids 105/106).
        // Daemon special-cases this until trait-dispatched framing lands.
        PromptFrameOverrides::default()
    }

    fn eos_filter_overrides(_cfg: &Self::Config) -> EosFilterOverrides {
        // Gemma 4 EOS tokens: <eos> (1) and <turn|> (106).
        // The daemon already handles EOS=1 universally; <turn|> is
        // added via the tokenizer/daemon framing. For now, no overrides.
        EosFilterOverrides::default()
    }
}
