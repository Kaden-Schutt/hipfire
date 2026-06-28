// SPDX-License-Identifier: Apache-2.0
// hipfire — generic KLD-eval orchestration (arch-independent).
//
//! The chunk loop, top-k reference reduction, KLD-vs-reference scoring, and
//! aggregation live here ONCE. Each arch supplies only a `forward_chunk`
//! closure — run its forward over an `n_ctx` chunk and call `emit(j, logits,
//! next_token)` for every scored position `j` in `[n_ctx/2, n_ctx-1)`. qwen35,
//! nemotron, … differ only in that forward; the scoring math is shared so a
//! reference build and a candidate score cannot drift. Generic over the forward
//! error `E` so this crate stays GPU-independent (no `hip_bridge`/`rdna`).

use crate::{score_position, top_k_log_softmax, ChunkResult, RefArchive, RefBlock, TopKReduction};

/// Per-scored-position emit callback: `(j, full_logits, actual_next_token)`.
pub type Emit<'a> = dyn FnMut(usize, &[f32], usize) + 'a;

/// Aggregate KLD result over all scored chunks.
pub struct KldEvalOutcome {
    pub n_chunk: usize,
    pub total_scored: usize,
    pub mean_kld: f32,
    pub p99_kld: f32,
    pub mean_nll: f32,
    pub per_chunk: Vec<ChunkResult>,
}

/// Reference payloads built from a resident model (top-K log-softmax per scored
/// position). The daemon wraps these in a [`RefArchive`] with `RefMeta`.
pub struct KldRefPayloads {
    pub n_chunk: usize,
    pub n_ctx: usize,
    pub scored_per_chunk: usize,
    pub top_k: usize,
    pub n_vocab: usize,
    pub tokens: Vec<u32>,
    pub top_indices: Vec<u32>,
    pub top_log_probs: Vec<f32>,
    pub residual_mass: Vec<f32>,
}

fn mean_f32(v: &[f32]) -> f32 {
    if v.is_empty() {
        return 0.0;
    }
    (v.iter().map(|&x| x as f64).sum::<f64>() / v.len() as f64) as f32
}

fn p99_f32(v: &[f32]) -> f32 {
    if v.is_empty() {
        return 0.0;
    }
    let mut s = v.to_vec();
    s.sort_by(|a, b| a.total_cmp(b));
    let idx = (((s.len() as f64) * 0.99).ceil() as usize)
        .saturating_sub(1)
        .min(s.len() - 1);
    s[idx]
}

/// Self-consistency KLD against the resident model (no reload): pass 1 builds an
/// in-memory top-K reference from the model over each `n_ctx` chunk; pass 2
/// scores the SAME model against it through the identical `forward_chunk`. A
/// healthy run returns ≈0 on ANY arch (the cross-binary-drift guard).
pub fn self_score<E>(
    tokens: &[u32],
    n_ctx: usize,
    top_k: usize,
    max_chunks: Option<usize>,
    mut forward_chunk: impl FnMut(&[u32], usize, &mut Emit) -> Result<(), E>,
    mut on_chunk: impl FnMut(usize, usize, usize, f32),
) -> Result<KldEvalOutcome, E> {
    let scoring_start = n_ctx / 2;
    let n_chunk_avail = tokens.len() / n_ctx;
    let n_chunk = max_chunks.map_or(n_chunk_avail, |m| m.min(n_chunk_avail));

    let mut per_chunk = Vec::with_capacity(n_chunk);
    let mut global_kld_sum = 0.0f64;
    let mut total_scored = 0usize;
    let mut global_nll_sum = 0.0f64;
    let mut global_nll_n = 0usize;

    for c in 0..n_chunk {
        let chunk = &tokens[c * n_ctx..c * n_ctx + n_ctx];

        // Pass 1: reference top-K reductions for the scored positions.
        let mut reds: Vec<TopKReduction> = Vec::new();
        forward_chunk(chunk, scoring_start, &mut |_j, lg, _next| {
            reds.push(top_k_log_softmax(lg, top_k));
        })?;

        // Pass 2: score the same model against the in-memory reference.
        let mut klds: Vec<f32> = Vec::with_capacity(reds.len());
        let mut nlls: Vec<f32> = Vec::new();
        forward_chunk(chunk, scoring_start, &mut |j, lg, next| {
            let red = &reds[j];
            let rb = RefBlock {
                top_indices: &red.indices,
                top_log_probs: &red.log_probs,
                residual_mass: red.residual_mass,
            };
            let s = score_position(&rb, lg, next);
            klds.push(s.kld);
            if let Some(n) = s.nll {
                nlls.push(n);
            }
        })?;

        let mean_kld = mean_f32(&klds);
        global_kld_sum += klds.iter().map(|&x| x as f64).sum::<f64>();
        total_scored += klds.len();
        global_nll_sum += nlls.iter().map(|&x| x as f64).sum::<f64>();
        global_nll_n += nlls.len();
        on_chunk(c, n_chunk, klds.len(), mean_kld);
        per_chunk.push(ChunkResult {
            mean_kld: mean_kld as f64,
            p99_kld: p99_f32(&klds) as f64,
            mean_nll: mean_f32(&nlls) as f64,
        });
    }

    Ok(finish(n_chunk, total_scored, global_kld_sum, global_nll_sum, global_nll_n, per_chunk))
}

/// Build a KLD reference from the resident model: per `n_ctx` chunk, capture the
/// top-K log-softmax reduction at each scored position. One `forward_chunk` pass.
#[allow(clippy::too_many_arguments)]
pub fn build_ref<E>(
    tokens: &[u32],
    n_ctx: usize,
    top_k: usize,
    n_vocab: usize,
    max_chunks: Option<usize>,
    mut forward_chunk: impl FnMut(&[u32], usize, &mut Emit) -> Result<(), E>,
    mut on_chunk: impl FnMut(usize, usize, usize),
) -> Result<KldRefPayloads, E> {
    let scoring_start = n_ctx / 2;
    let scored_per_chunk = (n_ctx - 1).saturating_sub(scoring_start);
    let n_chunk_avail = tokens.len() / n_ctx;
    let n_chunk = max_chunks.map_or(n_chunk_avail, |m| m.min(n_chunk_avail));

    let mut out_tokens = Vec::with_capacity(n_chunk * n_ctx);
    let mut top_indices = Vec::with_capacity(n_chunk * scored_per_chunk * top_k);
    let mut top_log_probs = Vec::with_capacity(n_chunk * scored_per_chunk * top_k);
    let mut residual_mass = Vec::with_capacity(n_chunk * scored_per_chunk);

    for c in 0..n_chunk {
        let chunk = &tokens[c * n_ctx..c * n_ctx + n_ctx];
        out_tokens.extend_from_slice(chunk);
        forward_chunk(chunk, scoring_start, &mut |_j, lg, _next| {
            let r = top_k_log_softmax(lg, top_k);
            top_indices.extend_from_slice(&r.indices);
            top_log_probs.extend_from_slice(&r.log_probs);
            residual_mass.push(r.residual_mass);
        })?;
        on_chunk(c, n_chunk, scored_per_chunk);
    }

    Ok(KldRefPayloads {
        n_chunk,
        n_ctx,
        scored_per_chunk,
        top_k,
        n_vocab,
        tokens: out_tokens,
        top_indices,
        top_log_probs,
        residual_mass,
    })
}

/// Score the resident model against a persisted reference: forward over the
/// reference's embedded token stream and compute KLD per scored position against
/// the stored top-K blocks. Same `forward_chunk` the reference was built with, so
/// a same-model score returns ≈0.
pub fn score<E>(
    archive: &RefArchive,
    max_chunks: Option<usize>,
    mut forward_chunk: impl FnMut(&[u32], usize, &mut Emit) -> Result<(), E>,
    mut on_chunk: impl FnMut(usize, usize, usize, f32),
) -> Result<KldEvalOutcome, E> {
    let n_ctx = archive.meta.n_ctx;
    let scoring_start = archive.meta.scoring_start;
    let n_chunk = max_chunks.map_or(archive.meta.n_chunk, |m| m.min(archive.meta.n_chunk));

    let mut per_chunk = Vec::with_capacity(n_chunk);
    let mut global_kld_sum = 0.0f64;
    let mut total_scored = 0usize;
    let mut global_nll_sum = 0.0f64;
    let mut global_nll_n = 0usize;

    for c in 0..n_chunk {
        let chunk = &archive.tokens[c * n_ctx..c * n_ctx + n_ctx];
        let mut klds: Vec<f32> = Vec::new();
        let mut nlls: Vec<f32> = Vec::new();
        forward_chunk(chunk, scoring_start, &mut |j, lg, next| {
            let s = score_position(&archive.block(c, j), lg, next);
            klds.push(s.kld);
            if let Some(n) = s.nll {
                nlls.push(n);
            }
        })?;

        let mean_kld = mean_f32(&klds);
        global_kld_sum += klds.iter().map(|&x| x as f64).sum::<f64>();
        total_scored += klds.len();
        global_nll_sum += nlls.iter().map(|&x| x as f64).sum::<f64>();
        global_nll_n += nlls.len();
        on_chunk(c, n_chunk, klds.len(), mean_kld);
        per_chunk.push(ChunkResult {
            mean_kld: mean_kld as f64,
            p99_kld: p99_f32(&klds) as f64,
            mean_nll: mean_f32(&nlls) as f64,
        });
    }

    Ok(finish(n_chunk, total_scored, global_kld_sum, global_nll_sum, global_nll_n, per_chunk))
}

fn finish(
    n_chunk: usize,
    total_scored: usize,
    global_kld_sum: f64,
    global_nll_sum: f64,
    global_nll_n: usize,
    per_chunk: Vec<ChunkResult>,
) -> KldEvalOutcome {
    let mean_kld = if total_scored > 0 {
        (global_kld_sum / total_scored as f64) as f32
    } else {
        0.0
    };
    let mean_nll = if global_nll_n > 0 {
        (global_nll_sum / global_nll_n as f64) as f32
    } else {
        0.0
    };
    let chunk_means: Vec<f32> = per_chunk.iter().map(|c| c.mean_kld as f32).collect();
    KldEvalOutcome {
        n_chunk,
        total_scored,
        mean_kld,
        p99_kld: p99_f32(&chunk_means),
        mean_nll,
        per_chunk,
    }
}
