// SPDX-License-Identifier: Apache-2.0
//! PFlash per-block cosine-importance head (fp32 training twin).
//!
//! `score[b] = cosine(block_mean_K, last_token_K)` over the full kv_dim —
//! identical to production `pflash_score_q8_kv` / `compute_scores_batched`, so a
//! drafter trained against a target's block ranking is drop-in for PFlash's
//! existing scoring. Backward gives the gradient w.r.t. K for training.

use rdna_compute::{DType, Gpu, GpuTensor, HipResult};

/// Forward: `k` `[n_pos*kv_dim]` → `scores` `[n_blocks]`. `last_pos` is usually
/// `n_pos-1`. `n_blocks` defaults to `n_pos/block_size` at the call site.
pub fn pflash_score_forward(
    gpu: &mut Gpu,
    k: &GpuTensor,
    scores: &GpuTensor,
    n_pos: usize,
    kv_dim: usize,
    block_size: usize,
    n_blocks: usize,
    last_pos: usize,
) -> HipResult<()> {
    gpu.pflash_score_f32_fwd(k, scores, n_pos, kv_dim, block_size, n_blocks, last_pos)
}

/// Backward: `dscores` `[n_blocks]` → `dk` `[n_pos*kv_dim]`. Allocates and zeroes
/// `dk` internally (the kernel accumulates via atomics).
#[allow(clippy::too_many_arguments)]
pub fn pflash_score_backward(
    gpu: &mut Gpu,
    k: &GpuTensor,
    dscores: &GpuTensor,
    n_pos: usize,
    kv_dim: usize,
    block_size: usize,
    n_blocks: usize,
    last_pos: usize,
) -> HipResult<GpuTensor> {
    let dk = gpu.zeros(&[n_pos * kv_dim], DType::F32)?;
    gpu.pflash_score_f32_bwd(
        k, dscores, &dk, n_pos, kv_dim, block_size, n_blocks, last_pos,
    )?;
    Ok(dk)
}
