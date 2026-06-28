// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
//
//! nemotron_h KLD-eval seam. The chunk loop / top-k reference / scoring /
//! aggregation are shared in `hipfire_kld::eval`; this module supplies only
//! nemotron's `forward_chunk_scored` (run the resident model over a chunk,
//! emitting per-scored-position logits) plus thin wrappers binding it to the
//! generic orchestrators — mirroring qwen35's seam so the daemon's `kld_eval`
//! op works for nemotron without duplicating the scoring code.

use crate::model::NemotronModel;
use hip_bridge::HipResult;
use hipfire_kld::{KldEvalOutcome, KldRefPayloads, RefArchive};
use rdna_compute::Gpu;

/// Run the resident model over `chunk` (per-token decode from fresh SSM/conv/KV
/// state) and invoke `at_scored(j, full_logits, actual_next)` for each scored
/// position `j` in `[scoring_start, n_ctx-1)`. The single forward path both
/// reference build and candidate scoring funnel through (≈0 self-score).
fn forward_chunk_scored(
    gpu: &mut Gpu,
    model: &mut NemotronModel,
    chunk: &[u32],
    scoring_start: usize,
    mut at_scored: impl FnMut(usize, &[f32], usize),
) -> HipResult<()> {
    model.reset(gpu)?;
    let n = chunk.len();
    for pos in 0..n.saturating_sub(1) {
        model.forward_gpu(gpu, chunk[pos], pos)?;
        if pos >= scoring_start {
            let lg = gpu.download_f32(model.logits_tensor())?;
            at_scored(pos - scoring_start, &lg, chunk[pos + 1] as usize);
        }
    }
    Ok(())
}

/// Self-consistency KLD against the resident nemotron model (no reload).
pub fn kld_eval_self_score(
    gpu: &mut Gpu,
    model: &mut NemotronModel,
    tokens: &[u32],
    n_ctx: usize,
    top_k: usize,
    max_chunks: Option<usize>,
    on_chunk: impl FnMut(usize, usize, usize, f32),
) -> HipResult<KldEvalOutcome> {
    hipfire_kld::eval::self_score(
        tokens,
        n_ctx,
        top_k,
        max_chunks,
        |chunk, scoring_start, emit| {
            forward_chunk_scored(gpu, model, chunk, scoring_start, |j, lg, nx| {
                emit(j, lg, nx)
            })
        },
        on_chunk,
    )
}

/// Build a KLD reference from the resident nemotron model.
pub fn kld_build_ref(
    gpu: &mut Gpu,
    model: &mut NemotronModel,
    tokens: &[u32],
    n_ctx: usize,
    top_k: usize,
    max_chunks: Option<usize>,
    on_chunk: impl FnMut(usize, usize, usize),
) -> HipResult<KldRefPayloads> {
    let n_vocab = model.config().vocab_size;
    hipfire_kld::eval::build_ref(
        tokens,
        n_ctx,
        top_k,
        n_vocab,
        max_chunks,
        |chunk, scoring_start, emit| {
            forward_chunk_scored(gpu, model, chunk, scoring_start, |j, lg, nx| {
                emit(j, lg, nx)
            })
        },
        on_chunk,
    )
}

/// Score the resident nemotron model against a persisted reference.
pub fn kld_score(
    gpu: &mut Gpu,
    model: &mut NemotronModel,
    archive: &RefArchive,
    max_chunks: Option<usize>,
    on_chunk: impl FnMut(usize, usize, usize, f32),
) -> HipResult<KldEvalOutcome> {
    hipfire_kld::eval::score(
        archive,
        max_chunks,
        |chunk, scoring_start, emit| {
            forward_chunk_scored(gpu, model, chunk, scoring_start, |j, lg, nx| {
                emit(j, lg, nx)
            })
        },
        on_chunk,
    )
}
