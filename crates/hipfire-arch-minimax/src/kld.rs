// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! KLD-scoring seam for MiniMax-M2 (arch_id 10). Served from loose resident
//! slots (`minimax_weights`/`minimax_config`/`minimax_state`) rather than an
//! owned `SimpleAr` backend, so it can't ride the blanket
//! `impl<T: SimpleAr> ChunkScoredForward`. [`MiniMaxKldForward`] adapts the loose
//! `&weights`/`&config` to [`ChunkScoredForward`] directly, mirroring
//! `qwen35::forward_chunk_scored`.

use crate::forward::decode_step;
use crate::minimax::{MiniMaxConfig, MiniMaxState, MiniMaxWeights};
use hipfire_runtime::kld_eval::ChunkScoredForward;
use hipfire_rdna::Gpu;

/// Teacher-force `chunk` through a FRESH per-call state (KV cache + partial-RoPE
/// attention), feeding one token per position and yielding the just-fed token's
/// logits for each scored position `>= scoring_start`. A fresh `MiniMaxState`
/// starts at position 0, so each chunk is independent — the `kld_self_score` ≈0
/// guard verifies it.
fn forward_chunk_scored(
    gpu: &mut Gpu,
    weights: &MiniMaxWeights,
    config: &MiniMaxConfig,
    chunk: &[u32],
    scoring_start: usize,
    mut at_scored: impl FnMut(usize, &[f32], usize),
) -> Result<(), String> {
    let n = chunk.len();
    let mut state = MiniMaxState::new(gpu, config)?;
    for pos in 0..n.saturating_sub(1) {
        let logits = decode_step(config, weights, &mut state, gpu, chunk[pos], pos as u32)?;
        if pos >= scoring_start {
            at_scored(pos - scoring_start, &logits, chunk[pos + 1] as usize);
        }
    }
    Ok(())
}

/// Adapter making a resident MiniMax-M2 (the loose `weights`+`config` slots,
/// which don't implement [`hipfire_runtime::arch::SimpleAr`]) KLD-scorable
/// through the generic [`hipfire_runtime::kld_eval`] driver.
pub struct MiniMaxKldForward<'a> {
    pub weights: &'a MiniMaxWeights,
    pub config: &'a MiniMaxConfig,
}

impl ChunkScoredForward for MiniMaxKldForward<'_> {
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
