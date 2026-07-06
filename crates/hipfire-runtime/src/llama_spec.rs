// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! LLaMA-family per-position verify for model-free speculative decode.
//!
//! Kept in its own module (not in `llama.rs`) so it can be added without
//! touching that legacy file. The arch-generic `NgramSpeculator` reaches this
//! through `impl SpecTarget for LlamaBundle` (in `hipfire-arch-llama`).

use crate::llama::{
    argmax, forward_prefill_batch, forward_scratch_compute, forward_scratch_embed, is_batchable_la,
    weight_gemv, ForwardScratch, KvCache, LlamaConfig, LlamaWeights, PrefillBatchScratch,
};
use hip_bridge::HipResult;
use rdna_compute::Gpu;

/// Per-position greedy verify: run the target over `block` (length `n`) at
/// positions `[start_pos, start_pos + n)`, advancing `kv_cache` by `n`, and
/// return the target's greedy argmax at each position — `argmax[i]` is the token
/// predicted after consuming `block[0..=i]`.
///
/// Pure attention ⇒ no recurrent state to snapshot and the accepted-prefix KV is
/// already correct, so the speculator's `commit_prefix` is a no-op.
///
/// Fast path (the block-parallel win): when the block is batchable (`n >= 4`,
/// batchable weight dtypes, quantized KV, single chunk) one batched
/// [`forward_prefill_batch`] over the whole block leaves every row's hidden in
/// `pbs.x_batch`; we then do `n` cheap per-row `rmsnorm + lm_head + argmax`.
/// Shorter / ineligible blocks fall back to a per-token decode loop
/// (`forward_scratch_compute` already produces per-token logits).
///
/// The eligibility test mirrors `forward_prefill_batch`'s own (so the batched
/// call actually populates `pbs.x_batch` rather than silently taking its
/// per-token fallback); keep the two in sync.
#[allow(clippy::too_many_arguments)]
pub fn verify_block_argmax(
    gpu: &mut Gpu,
    weights: &LlamaWeights,
    config: &LlamaConfig,
    block: &[u32],
    start_pos: usize,
    kv_cache: &mut KvCache,
    scratch: &ForwardScratch,
    pbs: &PrefillBatchScratch,
) -> HipResult<Vec<u32>> {
    let n = block.len();
    let dim = config.dim;
    let mut out = Vec::with_capacity(n);

    const MIN_BATCH: usize = 4;
    let arch = gpu.arch.as_str();
    let kv_ok =
        kv_cache.quant_q8 || kv_cache.quant_asym2 || kv_cache.quant_asym3 || kv_cache.quant_asym4;
    let weights_ok = weights.layers.iter().all(|l| {
        is_batchable_la(l.wq.gpu_dtype, arch)
            && is_batchable_la(l.wk.gpu_dtype, arch)
            && is_batchable_la(l.wv.gpu_dtype, arch)
            && is_batchable_la(l.wo.gpu_dtype, arch)
            && is_batchable_la(l.w_gate.gpu_dtype, arch)
            && is_batchable_la(l.w_up.gpu_dtype, arch)
            && is_batchable_la(l.w_down.gpu_dtype, arch)
    });
    let eligible = crate::config::get().prefill_batched
        && n >= MIN_BATCH
        && n <= pbs.max_batch
        && kv_ok
        && weights_ok;

    if eligible {
        // Single batched forward (n <= pbs.max_batch ⇒ one chunk) populates
        // pbs.x_batch with all n rows of post-final-layer hidden. Its own
        // last-row lm_head is redundant here but cheap.
        forward_prefill_batch(
            gpu,
            weights,
            config,
            block,
            start_pos,
            kv_cache,
            scratch,
            Some(pbs),
        )?;
        for i in 0..n {
            let off_bytes = i * dim * 4;
            gpu.hip
                .memcpy_dtod_at(&scratch.x.buf, 0, &pbs.x_batch.buf, off_bytes, dim * 4)?;
            gpu.rmsnorm_f32(
                &scratch.x,
                &weights.output_norm,
                &scratch.tmp,
                config.norm_eps,
            )?;
            weight_gemv(gpu, &weights.output, &scratch.tmp, &scratch.logits)?;
            out.push(argmax(&gpu.download_f32(&scratch.logits)?));
        }
    } else {
        for (i, &tok) in block.iter().enumerate() {
            forward_scratch_embed(gpu, weights, config, tok, start_pos + i, scratch)?;
            forward_scratch_compute(gpu, weights, config, start_pos + i, kv_cache, scratch)?;
            out.push(argmax(&gpu.download_f32(&scratch.logits)?));
        }
    }
    Ok(out)
}
