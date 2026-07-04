// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! KLD-scoring seam for LFM2 (arch_id 11). LFM2 is served from loose resident
//! slots (`lfm2moe_weights`/`lfm2moe_config`/`lfm2moe_state`) rather than an
//! owned `SimpleAr` backend, so it can't ride the blanket
//! `impl<T: SimpleAr> ChunkScoredForward`. Instead [`Lfm2MoeKldForward`] adapts
//! the loose `&weights`/`&config` to [`ChunkScoredForward`] directly — the
//! calibration analogue of `Qwen35KldForward`/`Qwen35CalibBackend` and
//! structurally identical to `qwen35::forward_chunk_scored`.

use crate::config::Lfm2MoeConfig;
use crate::forward::decode_step;
use crate::lfm2moe::{Lfm2MoeState, Lfm2MoeWeights};
use hipfire_runtime::kld_eval::ChunkScoredForward;
use hipfire_rdna::Gpu;

/// Teacher-force `chunk` through a FRESH per-call state (KV + conv ring), feeding
/// one token per position and yielding the just-fed token's logits for each
/// scored position `>= scoring_start`. Fresh state per chunk is what makes chunks
/// independent — the contract `kld_self_score`'s ≈0 guard verifies.
fn forward_chunk_scored(
    gpu: &mut Gpu,
    weights: &Lfm2MoeWeights,
    config: &Lfm2MoeConfig,
    chunk: &[u32],
    scoring_start: usize,
    mut at_scored: impl FnMut(usize, &[f32], usize),
) -> Result<(), String> {
    let n = chunk.len();
    let mut state = Lfm2MoeState::new(gpu, config)?;
    for pos in 0..n.saturating_sub(1) {
        let logits = decode_step(config, weights, &mut state, gpu, chunk[pos], pos as u32)?;
        if pos >= scoring_start {
            at_scored(pos - scoring_start, &logits, chunk[pos + 1] as usize);
        }
    }
    Ok(())
}

/// Adapter making a resident LFM2 (the loose `weights`+`config` slots, which
/// don't implement [`hipfire_runtime::arch::SimpleAr`]) KLD-scorable through the
/// generic [`hipfire_runtime::kld_eval`] driver.
pub struct Lfm2MoeKldForward<'a> {
    pub weights: &'a Lfm2MoeWeights,
    pub config: &'a Lfm2MoeConfig,
}

impl ChunkScoredForward for Lfm2MoeKldForward<'_> {
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
