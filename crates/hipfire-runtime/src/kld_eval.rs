// SPDX-License-Identifier: Apache-2.0
// hipfire — arch-agnostic KLD / perplexity evaluation driver.
//
//! The single forward seam ([`ChunkScoredForward`]) + the self-score /
//! build-ref / score drivers over the pure [`hipfire_kld`] core. Any
//! autoregressive arch that implements [`crate::arch::SimpleAr`] is KLD-scorable
//! for free via the blanket impl — no per-arch eval copies. The daemon
//! `kld_eval` op and the `perplexity` example funnel through here, so reference
//! build and candidate scoring share ONE forward path on ONE arch.

use crate::arch::SimpleAr;
use rdna_compute::Gpu;

/// Forward a token chunk (teacher-forced, fresh per-chunk state) and yield
/// per-position logits over the scored window. The one seam every KLD driver
/// funnels through; one impl per arch (blanket over [`SimpleAr`]).
pub trait ChunkScoredForward {
    /// Run the model over `chunk` and invoke `at_scored(j, full_logits, next)`
    /// for each scored position `j` in `[scoring_start, chunk.len() - 1)`, where
    /// `full_logits` predicts token `chunk[scoring_start + j + 1]` and `next` is
    /// that actual next token. State is fresh per call.
    fn forward_chunk_scored(
        &mut self,
        gpu: &mut Gpu,
        chunk: &[u32],
        scoring_start: usize,
        at_scored: &mut dyn FnMut(usize, &[f32], usize),
    ) -> Result<(), String>;

    /// Vocabulary size (for reference provenance).
    fn kld_vocab_size(&self) -> usize;
}

/// Every autoregressive backend is KLD-scorable: prefill the first token, decode
/// the rest teacher-forced, downloading logits at each scored position.
/// `prefill` starts a fresh sequence (position 0), so chunks are independent —
/// the same contract serving relies on.
impl<T: SimpleAr> ChunkScoredForward for T {
    fn forward_chunk_scored(
        &mut self,
        gpu: &mut Gpu,
        chunk: &[u32],
        scoring_start: usize,
        at_scored: &mut dyn FnMut(usize, &[f32], usize),
    ) -> Result<(), String> {
        let n = chunk.len();
        if n < 2 {
            return Ok(());
        }
        for pos in 0..n - 1 {
            if pos == 0 {
                self.prefill(gpu, &chunk[..1])?;
            } else {
                self.decode_step(gpu, chunk[pos], pos)?;
            }
            if pos >= scoring_start {
                let lg = gpu
                    .download_f32(self.logits())
                    .map_err(|e| format!("kld forward: download logits: {e:?}"))?;
                at_scored(pos - scoring_start, &lg, chunk[pos + 1] as usize);
            }
        }
        Ok(())
    }

    fn kld_vocab_size(&self) -> usize {
        SimpleAr::vocab_size(self)
    }
}

/// Lets a borrowed backend (`&mut ZayaModel as &mut dyn ChunkScoredForward`) be
/// boxed alongside an owned adapter (`Qwen35KldForward`) into one
/// `Box<dyn ChunkScoredForward>` — the daemon's per-arch dispatch erases the
/// borrowed-vs-owned distinction this way.
impl ChunkScoredForward for &mut dyn ChunkScoredForward {
    fn forward_chunk_scored(
        &mut self,
        gpu: &mut Gpu,
        chunk: &[u32],
        scoring_start: usize,
        at_scored: &mut dyn FnMut(usize, &[f32], usize),
    ) -> Result<(), String> {
        (**self).forward_chunk_scored(gpu, chunk, scoring_start, at_scored)
    }
    fn kld_vocab_size(&self) -> usize {
        (**self).kld_vocab_size()
    }
}

/// Outcome of a KLD evaluation pass.
pub struct KldEvalOutcome {
    pub n_chunk: usize,
    pub total_scored: usize,
    pub mean_kld: f32,
    pub p99_kld: f32,
    pub mean_nll: f32,
    pub per_chunk: Vec<hipfire_kld::ChunkResult>,
}

/// Reference payloads built from the resident model. The caller wraps these in a
/// [`hipfire_kld::RefArchive`] with provenance metadata.
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

fn kld_mean_f32(v: &[f32]) -> f32 {
    if v.is_empty() {
        return 0.0;
    }
    (v.iter().map(|&x| x as f64).sum::<f64>() / v.len() as f64) as f32
}

fn kld_p99_f32(v: &[f32]) -> f32 {
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

/// Self-consistency KLD against the resident model, no reload. Pass 1 builds a
/// top-K reference from the loaded model over each `n_ctx` chunk; pass 2 scores
/// the SAME model against that in-memory reference via a second forward through
/// the identical seam. A healthy run returns ≈0 on ANY arch — the guard against
/// forward non-determinism / plumbing drift.
pub fn kld_self_score(
    fwd: &mut dyn ChunkScoredForward,
    gpu: &mut Gpu,
    tokens: &[u32],
    n_ctx: usize,
    top_k: usize,
    max_chunks: Option<usize>,
    mut on_chunk: impl FnMut(usize, usize, usize, f32),
) -> Result<KldEvalOutcome, String> {
    use hipfire_kld::{score_position, top_k_log_softmax, RefBlock, TopKReduction};

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
        fwd.forward_chunk_scored(gpu, chunk, scoring_start, &mut |_j, lg, _next| {
            reds.push(top_k_log_softmax(lg, top_k));
        })?;

        // Pass 2: score the same model against the in-memory reference.
        let mut klds: Vec<f32> = Vec::with_capacity(reds.len());
        let mut nlls: Vec<f32> = Vec::new();
        fwd.forward_chunk_scored(gpu, chunk, scoring_start, &mut |j, lg, next| {
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

        let mean_kld = kld_mean_f32(&klds);
        global_kld_sum += klds.iter().map(|&x| x as f64).sum::<f64>();
        total_scored += klds.len();
        global_nll_sum += nlls.iter().map(|&x| x as f64).sum::<f64>();
        global_nll_n += nlls.len();
        on_chunk(c, n_chunk, klds.len(), mean_kld);
        per_chunk.push(hipfire_kld::ChunkResult {
            mean_kld: mean_kld as f64,
            p99_kld: kld_p99_f32(&klds) as f64,
            mean_nll: kld_mean_f32(&nlls) as f64,
        });
    }

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

    Ok(KldEvalOutcome {
        n_chunk,
        total_scored,
        mean_kld,
        p99_kld: kld_p99_f32(&chunk_means),
        mean_nll,
        per_chunk,
    })
}

/// Build a KLD reference from the resident model: per `n_ctx` chunk, capture the
/// top-K log-softmax reduction at each scored position. Same forward seam as
/// `score`, so a same-model score returns ≈0.
pub fn kld_build_ref(
    fwd: &mut dyn ChunkScoredForward,
    gpu: &mut Gpu,
    tokens: &[u32],
    n_ctx: usize,
    top_k: usize,
    max_chunks: Option<usize>,
    mut on_chunk: impl FnMut(usize, usize, usize),
) -> Result<KldRefPayloads, String> {
    use hipfire_kld::top_k_log_softmax;

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
        fwd.forward_chunk_scored(gpu, chunk, scoring_start, &mut |_j, lg, _next| {
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
        n_vocab: fwd.kld_vocab_size(),
        tokens: out_tokens,
        top_indices,
        top_log_probs,
        residual_mass,
    })
}

/// Score the resident model against a persisted reference: forward over the
/// reference's embedded token stream and compute KLD per scored position against
/// the stored top-K blocks. Same forward seam as `build_ref`.
pub fn kld_score(
    fwd: &mut dyn ChunkScoredForward,
    gpu: &mut Gpu,
    archive: &hipfire_kld::RefArchive,
    max_chunks: Option<usize>,
    mut on_chunk: impl FnMut(usize, usize, usize, f32),
) -> Result<KldEvalOutcome, String> {
    use hipfire_kld::score_position;

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
        fwd.forward_chunk_scored(gpu, chunk, scoring_start, &mut |j, lg, next| {
            let s = score_position(&archive.block(c, j), lg, next);
            klds.push(s.kld);
            if let Some(n) = s.nll {
                nlls.push(n);
            }
        })?;

        let mean_kld = kld_mean_f32(&klds);
        global_kld_sum += klds.iter().map(|&x| x as f64).sum::<f64>();
        total_scored += klds.len();
        global_nll_sum += nlls.iter().map(|&x| x as f64).sum::<f64>();
        global_nll_n += nlls.len();
        on_chunk(c, n_chunk, klds.len(), mean_kld);
        per_chunk.push(hipfire_kld::ChunkResult {
            mean_kld: mean_kld as f64,
            p99_kld: kld_p99_f32(&klds) as f64,
            mean_nll: kld_mean_f32(&nlls) as f64,
        });
    }

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

    Ok(KldEvalOutcome {
        n_chunk,
        total_scored,
        mean_kld,
        p99_kld: kld_p99_f32(&chunk_means),
        mean_nll,
        per_chunk,
    })
}
