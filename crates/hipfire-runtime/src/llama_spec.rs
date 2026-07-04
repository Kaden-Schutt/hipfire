// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! LLaMA-family per-position verify primitives for speculative decode.
//!
//! Kept in its own module (not in `llama.rs`) so the verify helpers can be added
//! without perturbing the forward hot path. These are lower-level GPU forward
//! helpers: they take `crate::llama` scratch/KV/weights types plus a `Gpu`, and
//! return argmax/logits `Vec`s (optionally capturing per-extract-layer hidden
//! into a caller-owned GPU buffer). They have NO dependency on any spec-decode
//! seam type — a later arch-llama `SpecTarget` impl calls these.

use crate::llama::{
    forward_prefill_batch_capture, forward_prefill_batch_tree, forward_scratch_compute,
    forward_scratch_embed, ForwardScratch, HiddenCaptureSink, KvCache, LlamaConfig, LlamaWeights,
    PrefillBatchScratch, PREFILL_MAX_BATCH,
};
use crate::sampler::argmax;
use crate::weights::weight_gemv;
use hip_bridge::HipResult;
use hipfire_rdna::{DType, Gpu, GpuTensor};

/// Whether a block of `n` tokens takes the batched verify forward (vs the
/// per-token fallback) — the path that populates `pbs.x_batch` and drives the
/// [`HiddenCaptureSink`]. Mirrors `forward_prefill_batch`'s own eligibility
/// (through the shared `transformer` seam) so a capture request is only made when
/// the batched call will actually run, plus the single-chunk bound.
fn batched_verify_eligible(
    gpu: &Gpu,
    weights: &LlamaWeights,
    kv_cache: &KvCache,
    n: usize,
    pbs: &PrefillBatchScratch,
) -> bool {
    let arch = gpu.arch.as_str();
    let batched_enabled = crate::config::get().prefill_batched;
    crate::transformer::llama_prefill_batchable(weights, kv_cache, arch, n, batched_enabled)
        && n <= pbs.max_batch
}

/// Per-position greedy verify: run the target over `block` (length `n`) at
/// positions `[start_pos, start_pos + n)`, advancing `kv_cache` by `n`, and
/// return the target's greedy argmax at each position — `argmax[i]` is the token
/// predicted after consuming `block[0..=i]`.
///
/// Fast path: when the block is batchable, one batched
/// [`forward_prefill_batch_capture`] over the whole block leaves every row's
/// hidden in `pbs.x_batch`; we then do `n` cheap per-row `rmsnorm + lm_head +
/// argmax`. Shorter / ineligible blocks fall back to a per-token decode loop.
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
    capture: Option<&mut HiddenCaptureSink>,
) -> HipResult<Vec<u32>> {
    verify_block_logits_or_argmax(
        gpu, weights, config, block, start_pos, kv_cache, scratch, pbs, capture, false, None, false,
    )
    .map(|VerifyOut { argmax, .. }| argmax)
}

/// Like [`verify_block_argmax`] but returns the FULL per-position target logits
/// (`block.len() × vocab_size`, row-major) instead of just the argmax. Used by
/// the temp>0 chain path (naive sampling draws from the per-position target
/// distribution). The logits are bit-identical to those `verify_block_argmax`
/// argmaxes internally.
#[allow(clippy::too_many_arguments)]
pub fn verify_block_logits(
    gpu: &mut Gpu,
    weights: &LlamaWeights,
    config: &LlamaConfig,
    block: &[u32],
    start_pos: usize,
    kv_cache: &mut KvCache,
    scratch: &ForwardScratch,
    pbs: &PrefillBatchScratch,
    capture: Option<&mut HiddenCaptureSink>,
) -> HipResult<Vec<f32>> {
    verify_block_logits_or_argmax(
        gpu, weights, config, block, start_pos, kv_cache, scratch, pbs, capture, true, None, false,
    )
    .map(|VerifyOut { logits, .. }| logits)
}

/// Like [`verify_block_argmax`] but captures the per-position extract-layer
/// residual hidden into the caller-owned GPU buffer `hidden_gpu` (position-major
/// `[n × extract_layers.len() × dim]` F32) instead of a host `Vec` — the
/// accepted-prefix-hidden reuse then stays entirely on-device (no D2H+H2D per
/// window).
///
/// Returns `(per-position argmax, captured)`. `captured` is `true` iff the
/// batched path ran (so all `block.len()` positions' hidden were written); the
/// per-token fallback captures nothing and returns `false`. When `false`,
/// `hidden_gpu` is left untouched.
#[allow(clippy::too_many_arguments)]
pub fn verify_block_argmax_capture_gpu(
    gpu: &mut Gpu,
    weights: &LlamaWeights,
    config: &LlamaConfig,
    block: &[u32],
    start_pos: usize,
    kv_cache: &mut KvCache,
    scratch: &ForwardScratch,
    pbs: &PrefillBatchScratch,
    extract_layers: &[usize],
    hidden_gpu: &GpuTensor,
) -> HipResult<(Vec<u32>, bool)> {
    let captured = !extract_layers.is_empty()
        && batched_verify_eligible(gpu, weights, kv_cache, block.len(), pbs);
    let mut empty: Vec<f32> = Vec::new();
    let mut sink = if captured {
        Some(HiddenCaptureSink {
            extract_layers,
            hidden: &mut empty,
            hidden_gpu: Some(hidden_gpu),
        })
    } else {
        None
    };
    let argmax = verify_block_logits_or_argmax(
        gpu,
        weights,
        config,
        block,
        start_pos,
        kv_cache,
        scratch,
        pbs,
        sink.as_mut(),
        false,
        None,
        true, // greedy: lazy prefix-stop (byte-identical committed, fewer lm_heads)
    )
    .map(|VerifyOut { argmax, .. }| argmax)?;
    Ok((argmax, captured))
}

/// Sampled (temp>0) counterpart of [`verify_block_argmax_capture_gpu`]: runs the
/// SAME batched forward + GPU-resident hidden capture, but draws each position's
/// token `t_i ~ p_T(temp, top_p)` (advancing `rng`) instead of argmax. Returns
/// `(per-position sampled tokens, captured)` with the same `captured` semantics.
///
/// Each position is drawn on-GPU via the fused `sample_top_p_pf` kernel — the
/// SAME sampler AR decode uses, so committed tokens are distribution-identical to
/// AR. At `temp <= 1e-6` the kernel collapses to argmax.
///
/// NOTE (chaingun adaptation): chaingun's `sample_top_p_pf` does not take a
/// `top_k`/`min_p` argument, so `top_k` is accepted for API parity but not
/// applied here (temp + top_p nucleus only).
#[allow(clippy::too_many_arguments)]
pub fn verify_block_sampled_capture_gpu(
    gpu: &mut Gpu,
    weights: &LlamaWeights,
    config: &LlamaConfig,
    block: &[u32],
    start_pos: usize,
    kv_cache: &mut KvCache,
    scratch: &ForwardScratch,
    pbs: &PrefillBatchScratch,
    extract_layers: &[usize],
    hidden_gpu: &GpuTensor,
    temp: f32,
    top_p: f32,
    top_k: usize,
    rng_state: &mut u64,
) -> HipResult<(Vec<u32>, bool)> {
    let _ = top_k; // chaingun sampler has no top_k arg; kept for API parity
    let captured = !extract_layers.is_empty()
        && batched_verify_eligible(gpu, weights, kv_cache, block.len(), pbs);
    let mut empty: Vec<f32> = Vec::new();
    let mut sink = if captured {
        Some(HiddenCaptureSink {
            extract_layers,
            hidden: &mut empty,
            hidden_gpu: Some(hidden_gpu),
        })
    } else {
        None
    };
    // Sampler scratch, allocated once for the whole block (freed below).
    let result_buf = gpu.alloc_tensor(&[2], DType::F32)?;
    let repeat_buf = gpu.alloc_tensor(&[1], DType::F32)?;
    let mut rng32 = *rng_state as u32;
    let out = verify_block_logits_or_argmax(
        gpu,
        weights,
        config,
        block,
        start_pos,
        kv_cache,
        scratch,
        pbs,
        sink.as_mut(),
        false, // no full-logit download; the GPU sampler returns picks directly
        Some(SampleCfg {
            temp,
            top_p,
            rng: &mut rng32,
            result_buf: &result_buf,
            repeat_buf: &repeat_buf,
        }),
        true, // sampled verify: lazy prefix-stop
    );
    *rng_state = rng32 as u64;
    let _ = gpu.free_tensor(result_buf);
    let _ = gpu.free_tensor(repeat_buf);
    Ok((out?.argmax, captured))
}

/// One single-pass TREE-masked verify, returning the FULL per-node target logits
/// (`tokens.len() × vocab_size`, row-major).
///
/// `tokens` is the linearized tree (slot 0 = seed), `mask_host` the `[n × n]`
/// row-major additive (`0.0`/`-inf`) tree-attention bias, and `depth_positions`
/// the per-slot DEPTH RoPE positions (`position + node.depth`). The whole tree is
/// verified in ONE batched forward; `capture` collects the per-extract-layer
/// residual rows for DFlash hidden conditioning. The mask is uploaded into a
/// scratch GPU tensor allocated + freed within the call.
#[allow(clippy::too_many_arguments)]
pub fn verify_tree_logits(
    gpu: &mut Gpu,
    weights: &LlamaWeights,
    config: &LlamaConfig,
    tokens: &[u32],
    mask_host: &[f32],
    depth_positions: &[i32],
    position: usize,
    kv_cache: &mut KvCache,
    scratch: &ForwardScratch,
    pbs: &PrefillBatchScratch,
    capture: Option<&mut HiddenCaptureSink>,
) -> HipResult<Vec<f32>> {
    let n = tokens.len();
    let dim = config.dim;
    let vocab = config.vocab_size;
    assert_eq!(
        mask_host.len(),
        n * n,
        "verify_tree_logits: mask_host len {} != n*n ({}*{})",
        mask_host.len(),
        n,
        n
    );

    // Upload the [n × n] additive mask into a scratch GPU tensor.
    let bias = gpu.alloc_tensor(&[n * n], DType::F32)?;
    let mask_bytes: &[u8] =
        unsafe { std::slice::from_raw_parts(mask_host.as_ptr() as *const u8, mask_host.len() * 4) };
    gpu.hip.memcpy_htod(&bias.buf, mask_bytes)?;

    // ONE tree-masked batched forward → every node's hidden lands in pbs.x_batch.
    let fwd = forward_prefill_batch_tree(
        gpu,
        weights,
        config,
        tokens,
        position,
        &bias,
        depth_positions,
        kv_cache,
        scratch,
        pbs,
        capture,
    );
    let _ = gpu.free_tensor(bias);
    fwd?;

    // Per-node rmsnorm + lm_head over the n hidden rows in pbs.x_batch.
    let mut logits_out: Vec<f32> = Vec::with_capacity(n * vocab);
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
        logits_out.extend_from_slice(&gpu.download_f32(&scratch.logits)?);
    }
    Ok(logits_out)
}

/// Output of the shared verify forward: per-position pick (argmax, or a sampled
/// token when `SampleCfg` was supplied); full logits only when `want_logits`.
struct VerifyOut {
    argmax: Vec<u32>,
    logits: Vec<f32>,
}

/// Per-position GPU sampling for the temp>0 verify. Each position's pick is drawn
/// on-GPU via the fused `sample_top_p_pf` kernel — the SAME sampler the AR decode
/// uses, so committed tokens are distribution-identical to AR.
struct SampleCfg<'a> {
    temp: f32,
    top_p: f32,
    /// xorshift/LCG state (u32, as the sampler kernel expects); advanced per draw.
    rng: &'a mut u32,
    /// Sampler scratch: `result_buf` `[2]` F32, `repeat_buf` `[1]` F32 (unused
    /// with `repeat_window=0`). Caller-owned so the loop allocates once.
    result_buf: &'a GpuTensor,
    repeat_buf: &'a GpuTensor,
}

/// Shared body for the verify helpers: one batched forward over `block`, then
/// per-row `rmsnorm + lm_head + argmax`. For the greedy path (`!want_logits`)
/// argmax is computed on GPU and only 4 bytes per position are downloaded; for
/// `want_logits=true` the full logit row is downloaded.
#[allow(clippy::too_many_arguments)]
fn verify_block_logits_or_argmax(
    gpu: &mut Gpu,
    weights: &LlamaWeights,
    config: &LlamaConfig,
    block: &[u32],
    start_pos: usize,
    kv_cache: &mut KvCache,
    scratch: &ForwardScratch,
    pbs: &PrefillBatchScratch,
    capture: Option<&mut HiddenCaptureSink>,
    want_logits: bool,
    mut sample: Option<SampleCfg>,
    // LAZY prefix stop: skip the per-row head for positions after the first
    // draft/pick mismatch (acceptance is a prefix). Committed output is identical,
    // just fewer lm_head GEMVs. Only safe for callers whose picks feed
    // accept_greedy_prefix; the plain verify_block_argmax/logits pass `false`.
    lazy: bool,
) -> HipResult<VerifyOut> {
    let n = block.len();
    let dim = config.dim;
    let vocab = config.vocab_size;
    let mut out = Vec::with_capacity(n);
    let mut logits_out: Vec<f32> = if want_logits {
        Vec::with_capacity(n * vocab)
    } else {
        Vec::new()
    };

    let eligible = batched_verify_eligible(gpu, weights, kv_cache, n, pbs);

    // Hidden capture only flows through the batched path; the per-token fallback
    // does not run the capturing per-layer loop. Clearing the sink for an
    // ineligible block yields the correct "not captured" signal (empty result).
    let capture = if !eligible { None } else { capture };

    if eligible {
        // Single batched forward (n <= pbs.max_batch ⇒ one chunk) populates
        // pbs.x_batch with all n rows of post-final-layer hidden. `capture` (if
        // Some) collects the per-extract-layer residual rows.
        forward_prefill_batch_capture(
            gpu,
            weights,
            config,
            block,
            start_pos,
            kv_cache,
            scratch,
            Some(pbs),
            capture,
        )?;

        // Per-row lm_head loop. For the greedy path we run the argmax on-GPU and
        // download only 4 bytes per position instead of the full vocab × 4.
        let argmax_one = if !want_logits && sample.is_none() {
            Some(gpu.alloc_tensor(&[1], DType::F32)?)
        } else {
            None
        };
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
            if let Some(sc) = sample.as_mut() {
                // temp>0: fused GPU sample (softmax+nucleus+draw) → 4-byte D2H.
                let tok = sample_one(gpu, &scratch.logits, vocab, sc)?;
                out.push(tok);
                if lazy && i + 1 < n && block[i + 1] != tok {
                    while out.len() < n {
                        out.push(u32::MAX);
                    }
                    break;
                }
            } else if let Some(ref ab) = argmax_one {
                // GPU argmax → 4-byte D2H (avoids the full vocab download).
                gpu.argmax_f32_batched(&scratch.logits, ab, vocab, 1)?;
                let mut raw = 0i32;
                let bytes =
                    unsafe { std::slice::from_raw_parts_mut(&mut raw as *mut i32 as *mut u8, 4) };
                gpu.hip.memcpy_dtoh(bytes, &ab.buf)?;
                let tok = raw as u32;
                out.push(tok);
                if lazy && i + 1 < n && block[i + 1] != tok {
                    while out.len() < n {
                        out.push(u32::MAX);
                    }
                    break;
                }
            } else {
                let row = gpu.download_f32(&scratch.logits)?;
                out.push(argmax(&row));
                logits_out.extend_from_slice(&row);
            }
        }
        if let Some(ab) = argmax_one {
            let _ = gpu.free_tensor(ab);
        }
    } else {
        for (i, &tok) in block.iter().enumerate() {
            forward_scratch_embed(gpu, weights, config, tok, start_pos + i, scratch)?;
            forward_scratch_compute(gpu, weights, config, start_pos + i, kv_cache, scratch)?;
            if let Some(sc) = sample.as_mut() {
                let pick = sample_one(gpu, &scratch.logits, vocab, sc)?;
                out.push(pick);
                if lazy && i + 1 < n && block[i + 1] != pick {
                    while out.len() < n {
                        out.push(u32::MAX);
                    }
                    break;
                }
            } else {
                let row = gpu.download_f32(&scratch.logits)?;
                out.push(argmax(&row));
                if want_logits {
                    logits_out.extend_from_slice(&row);
                }
            }
        }
    }
    Ok(VerifyOut {
        argmax: out,
        logits: logits_out,
    })
}

/// One fused GPU sample from `logits` (`[vocab]` F32) via `sample_top_p_pf` — the
/// same kernel AR decode uses. No repeat/presence/frequency penalty here (verify
/// is distribution-only; the emission layer owns penalties). Advances `sc.rng`.
fn sample_one(
    gpu: &mut Gpu,
    logits: &GpuTensor,
    vocab: usize,
    sc: &mut SampleCfg,
) -> HipResult<u32> {
    let top_p_eff = if sc.top_p > 0.0 {
        sc.top_p.min(1.0)
    } else {
        1.0
    };
    let (tok, new_rng) = gpu.sample_top_p_pf(
        logits,
        sc.result_buf,
        sc.repeat_buf,
        vocab,
        sc.temp,
        top_p_eff,
        *sc.rng,
        0,   // repeat_window (no penalty in verify)
        1.0, // repeat_penalty
        0.0, // presence_penalty
        0.0, // frequency_penalty
    )?;
    *sc.rng = new_rng;
    Ok(tok)
}

/// Apply the target lm_head (final-norm + output projection) to `n` rows of
/// pre-norm residual hidden states, returning `n × vocab_size` host-side f32
/// logits in row-major order.
///
/// `hidden_rows` must be an `F32` `GpuTensor` of length `n × dim` laid out
/// row-major (row `i` starts at byte offset `i * dim * 4`). `scratch` is used as a
/// single-row staging buffer. Mirrors the per-row lm_head loop in
/// [`verify_block_argmax`] exactly, so the returned logits are bit-identical.
pub fn lm_head_logits_n_rows(
    gpu: &mut Gpu,
    weights: &LlamaWeights,
    config: &LlamaConfig,
    hidden_rows: &GpuTensor,
    n: usize,
    scratch: &ForwardScratch,
) -> HipResult<Vec<f32>> {
    let dim = config.dim;
    let vocab = config.vocab_size;
    let mut out = Vec::with_capacity(n * vocab);
    for i in 0..n {
        let off_bytes = i * dim * 4;
        gpu.hip
            .memcpy_dtod_at(&scratch.x.buf, 0, &hidden_rows.buf, off_bytes, dim * 4)?;
        gpu.rmsnorm_f32(
            &scratch.x,
            &weights.output_norm,
            &scratch.tmp,
            config.norm_eps,
        )?;
        weight_gemv(gpu, &weights.output, &scratch.tmp, &scratch.logits)?;
        out.extend_from_slice(&gpu.download_f32(&scratch.logits)?);
    }
    Ok(out)
}

/// Default dense-tree node budget (matches the DSpark `dflash_generic` default).
const DEFAULT_TREE_BUDGET: usize = 8;

/// Linearized node count (clamped `budget` + seed) the dense tree-verify scratch
/// must hold, or `0` when the tree arm is disabled. A later `SpecTarget`
/// `new_spec_scratch` sizes `PrefillBatchScratch` to at least this, so a large
/// tree budget can't overflow the verify batch (the
/// `forward_prefill_batch_tree: tree size N exceeds max_batch` panic).
///
/// NOTE (chaingun adaptation): chaingun's `FeatureFlags` does not yet carry the
/// `dflash_tree`/`ddtree_budget` fields the DSpark source read, so this takes the
/// enable flag + optional budget as plain arguments. The clamp against
/// [`PREFILL_MAX_BATCH`] is preserved.
pub fn dense_tree_verify_nodes(dflash_tree: bool, ddtree_budget: Option<usize>) -> usize {
    if dflash_tree {
        ddtree_budget
            .unwrap_or(DEFAULT_TREE_BUDGET)
            .clamp(1, PREFILL_MAX_BATCH - 1)
            + 1
    } else {
        0
    }
}
