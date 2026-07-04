// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! KLD-scoring seam for DeepSeek V4 Flash (arch_id 9). Served from loose
//! resident slots (`deepseek4_weights`/`deepseek4_config`/`deepseek4_state`)
//! rather than an owned `SimpleAr` backend, so it can't ride the blanket
//! `impl<T: SimpleAr> ChunkScoredForward`. [`DeepseekV4KldForward`] adapts the
//! loose `&weights`/`&config` to [`ChunkScoredForward`] directly, mirroring
//! `qwen35::forward_chunk_scored`.

use crate::deepseek4::{DeepseekV4Config, DeepseekV4State, DeepseekV4Weights};
use crate::forward::decode_step;
use hipfire_runtime::kld_eval::ChunkScoredForward;
use hipfire_rdna::Gpu;

/// Teacher-force `chunk` through a FRESH per-call state (MLA / compressed-KV
/// indexer + SWA + hyper-connection residuals), feeding one token per position
/// and yielding the just-fed token's logits for each scored position
/// `>= scoring_start`. A fresh `DeepseekV4State` starts at position 0 with lazy
/// (None) caches, so each chunk is independent — the `kld_self_score` ≈0 guard
/// verifies it.
fn forward_chunk_scored(
    gpu: &mut Gpu,
    weights: &DeepseekV4Weights,
    config: &DeepseekV4Config,
    chunk: &[u32],
    scoring_start: usize,
    mut at_scored: impl FnMut(usize, &[f32], usize),
) -> Result<(), String> {
    let n = chunk.len();
    let mut state = DeepseekV4State::new(config)?;
    for pos in 0..n.saturating_sub(1) {
        let logits = decode_step(config, weights, &mut state, gpu, chunk[pos], pos as u32)?;
        if pos >= scoring_start {
            at_scored(pos - scoring_start, &logits, chunk[pos + 1] as usize);
        }
    }
    Ok(())
}

/// Adapter making a resident DeepSeek V4 (the loose `weights`+`config` slots,
/// which don't implement [`hipfire_runtime::arch::SimpleAr`]) KLD-scorable
/// through the generic [`hipfire_runtime::kld_eval`] driver.
pub struct DeepseekV4KldForward<'a> {
    pub weights: &'a DeepseekV4Weights,
    pub config: &'a DeepseekV4Config,
}

impl ChunkScoredForward for DeepseekV4KldForward<'_> {
    fn forward_chunk_scored(
        &mut self,
        gpu: &mut Gpu,
        chunk: &[u32],
        scoring_start: usize,
        at_scored: &mut dyn FnMut(usize, &[f32], usize),
    ) -> Result<(), String> {
        forward_chunk_scored(
            gpu,
            self.weights,
            self.config,
            chunk,
            scoring_start,
            |j, lg, next| at_scored(j, lg, next),
        )
    }

    fn kld_vocab_size(&self) -> usize {
        self.config.vocab_size
    }
}
