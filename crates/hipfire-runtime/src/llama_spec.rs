// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! LLaMA-family per-position verify for model-free speculative decode.
//!
//! Kept in its own module (not in `llama.rs`) so it can be added without
//! touching that legacy file. The arch-generic `NgramSpeculator` reaches this
//! through `impl SpecTarget for LlamaBundle` (in `hipfire-arch-llama`).

use crate::llama::{
    argmax, forward_prefill_batch_capture, forward_prefill_batch_tree, forward_scratch_compute,
    forward_scratch_embed, is_batchable_la, weight_gemv, ForwardScratch, HiddenCaptureSink,
    KvCache, LlamaConfig, LlamaWeights, PrefillBatchScratch,
};
use hip_bridge::HipResult;
use rdna_compute::{DType, Gpu, GpuTensor};

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
/// Whether a block of `n` tokens takes the batched verify forward (vs the
/// per-token fallback) — the path that populates `pbs.x_batch` and drives the
/// [`HiddenCaptureSink`]. Mirrors `forward_prefill_batch`'s own eligibility so
/// a capture request is only made when the batched call will actually run.
fn batched_verify_eligible(
    gpu: &Gpu,
    weights: &LlamaWeights,
    kv_cache: &KvCache,
    n: usize,
    pbs: &PrefillBatchScratch,
) -> bool {
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
    crate::config::get().prefill_batched
        && n >= MIN_BATCH
        && n <= pbs.max_batch
        && kv_ok
        && weights_ok
}

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
/// the temp>0 chain DFlash path (SpecInfer naive sampling draws from the per-
/// position target distribution rather than taking the argmax). The logits are
/// bit-identical to those `verify_block_argmax` argmaxes internally — both go
/// through the same single batched forward + per-row `rmsnorm + lm_head`.
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
/// residual hidden into the caller-owned GPU buffer `hidden_gpu`
/// (position-major `[n × extract_layers.len() × dim]` F32) instead of a host
/// `Vec` — the DSpark accepted-prefix-hidden reuse then stays entirely on-device
/// (no D2H+H2D per window).
///
/// Returns `(per-position argmax, captured)`. `captured` is `true` iff the
/// batched path ran (so all `block.len()` positions' hidden were written); the
/// per-token fallback (`block.len() < 4`, or ineligible dtypes/KV) captures
/// nothing and returns `false`, matching the host sink's empty-capture signal.
/// When `false`, `hidden_gpu` is left untouched.
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
        true, // DSpark greedy: lazy prefix-stop (byte-identical committed, fewer lm_heads)
    )
    .map(|VerifyOut { argmax, .. }| argmax)?;
    Ok((argmax, captured))
}

/// Sampled (temp>0) counterpart of [`verify_block_argmax_capture_gpu`]: runs the
/// SAME batched forward + GPU-resident hidden capture, but draws each position's
/// token `t_i ~ p_T(temp, top_p, top_k)` (advancing `rng`) instead of argmax.
/// Returns `(per-position sampled tokens, captured)` with the same `captured`
/// semantics (false ⇒ per-token fallback ran, `hidden_gpu` untouched).
///
/// Each position is drawn on-GPU via the fused `sample_top_p_pf` kernel (softmax
/// + nucleus + top_k + categorical in ONE launch, 4-byte D2H) — no b×vocab
/// download, no host softmax. It is the SAME sampler AR decode uses, so the
/// committed tokens are distribution-identical to AR temp-T decoding. At
/// `temp <= 1e-6` the kernel collapses to argmax.
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
            top_k,
            rng: &mut rng32,
            result_buf: &result_buf,
            repeat_buf: &repeat_buf,
        }),
        true, // DSpark sampled verify: lazy prefix-stop
    );
    *rng_state = rng32 as u64;
    let _ = gpu.free_tensor(result_buf);
    let _ = gpu.free_tensor(repeat_buf);
    Ok((out?.argmax, captured))
}

/// One single-pass TREE-masked verify, returning the FULL per-node target
/// logits (`tokens.len() × vocab_size`, row-major).
///
/// `tokens` is the linearized DDTree (slot 0 = seed token, slots `1..` =
/// `tree.nodes`), `mask_host` is the `[n × n]` row-major additive
/// (`0.0`/`-inf`) tree-attention bias, and `depth_positions` the per-slot DEPTH
/// RoPE positions (`position + node.depth`) — all from
/// [`crate::ddtree::linearize_tree_with_parents`]. The whole tree is verified in
/// ONE batched forward: Q/K RoPE rotates at the DEPTH positions (parent→child
/// distance 1) while the KV write + mask stay on contiguous slots, so a node's
/// logits equal a causal verify of that node's root-to-node chain (greedy-
/// LOSSLESS). `capture` collects the per-extract-layer residual rows for DFlash
/// hidden conditioning. The mask is uploaded into a scratch GPU tensor
/// allocated + freed within the call.
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
    let bias = gpu.alloc_tensor(&[n * n], rdna_compute::DType::F32)?;
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

/// Per-position GPU sampling for the temp>0 verify. When supplied to
/// [`verify_block_logits_or_argmax`], each position's pick is drawn on-GPU via
/// the fused `sample_top_p_pf` kernel (softmax + nucleus + top_k/min_p +
/// categorical in ONE launch, 4-byte D2H) — the SAME sampler the AR decode uses,
/// so the committed tokens are distribution-identical to AR. This replaces the
/// host-softmax path (b×vocab D2H + host exp) entirely.
struct SampleCfg<'a> {
    temp: f32,
    top_p: f32,
    top_k: usize,
    /// xorshift/LCG state (u32, as the sampler kernel expects); advanced per draw.
    rng: &'a mut u32,
    /// Sampler scratch: `result_buf` `[2]` F32, `repeat_buf` `[1]` F32 (unused
    /// with `repeat_window=0`). Caller-owned so the loop allocates once.
    result_buf: &'a GpuTensor,
    repeat_buf: &'a GpuTensor,
}

/// Shared body for [`verify_block_argmax`] / [`verify_block_logits`]: one batched
/// forward over `block`, then per-row `rmsnorm + lm_head + argmax`. For the greedy
/// path (`!want_logits`) argmax is computed on GPU and only 4 bytes per position
/// are downloaded; for `want_logits=true` the full logit row is downloaded for
/// SWOR / temperature sampling.
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
    // draft/pick mismatch (acceptance is a prefix). Committed output is
    // identical (accept_greedy_prefix only reads up to the mismatch), just fewer
    // lm_head GEMVs. Only safe for callers whose picks feed accept_greedy_prefix
    // (the DSpark capture paths) — the plain verify_block_argmax/logits have
    // consumers that read every pick, so they pass `false`.
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

    // DFlash hidden capture only flows through the batched path; the per-token
    // fallback below does not run the capturing per-layer loop.
    // When the block is too small for the batched path (n < MIN_BATCH), silently
    // skip the capture by clearing the capture sink — the caller checks whether
    // hidden_out is non-empty, so an empty result is the correct "not captured"
    // signal and does not break correctness.
    let capture = if !eligible { None } else { capture };

    if eligible {
        // Single batched forward (n <= pbs.max_batch ⇒ one chunk) populates
        // pbs.x_batch with all n rows of post-final-layer hidden. Its own
        // last-row lm_head is redundant here but cheap. `capture` (if Some)
        // collects the per-extract-layer residual rows for DFlash conditioning.
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

        // Per-row lm_head loop.  For the greedy path (!want_logits) we run the
        // argmax on-GPU and download only 4 bytes per position instead of the
        // full vocab × 4 (≈607 KB for Qwen3-8B vocab=151936).  The logit matrix
        // is never materialised to GPU memory, keeping the L2 cache clean for
        // the subsequent draft pass.
        //
        // Argmax tie-break identity: `argmax_f32_batched` uses strict `>` in
        // both the per-thread scan and the shared-memory reduction (see
        // `kernels/src/argmax_batched.hip`), so the lowest vocab index wins on a
        // tie — identical to the CPU `argmax()` which also uses strict `>`.
        //
        // For want_logits=true (SWOR / temp>0 sampling) we still download the
        // full logit row and do CPU argmax (caller needs the distribution).
        let argmax_one = if !want_logits && sample.is_none() {
            // 4-byte scratch; pool-resident (256-byte bucket) after the first use.
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
                // LAZY: acceptance is a prefix — once a position's sample differs
                // from its drafted token (`block[i+1]`), every later position is
                // rejected, so skip their (expensive) head+sample entirely. `tok`
                // is the committed correction; pad the tail so the pick vector is
                // full length (accept_greedy_prefix only reads up to this mismatch).
                if lazy && i + 1 < n && block[i + 1] != tok {
                    while out.len() < n {
                        out.push(u32::MAX);
                    }
                    break;
                }
            } else if let Some(ref ab) = argmax_one {
                // GPU argmax → 4-byte D2H (avoids 607 KB download per position).
                gpu.argmax_f32_batched(&scratch.logits, ab, vocab, 1)?;
                let mut raw = 0i32;
                let bytes =
                    unsafe { std::slice::from_raw_parts_mut(&mut raw as *mut i32 as *mut u8, 4) };
                gpu.hip.memcpy_dtoh(bytes, &ab.buf)?;
                let tok = raw as u32;
                out.push(tok);
                // LAZY prefix stop (greedy): same reasoning as the sample branch —
                // skip lm_head for positions after the first mismatch. Output is
                // byte-identical (accept reads only up to the mismatch).
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
                // LAZY prefix stop (see the batched branch above).
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
/// same kernel AR decode uses, so DSpark's committed tokens match the AR temp-T
/// distribution. No repeat/presence/frequency penalty here (verify is
/// distribution-only; the emission layer owns penalties). Advances `sc.rng`.
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
    let top_k = if sc.top_k > 0 {
        Some(sc.top_k as u32)
    } else {
        None
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
        top_k,
        None, // min_p
    )?;
    *sc.rng = new_rng;
    Ok(tok)
}

/// Apply the target lm_head (final-norm + output projection) to `n` rows of
/// pre-norm residual hidden states, returning `n × vocab_size` host-side f32
/// logits in row-major order.
///
/// `hidden_rows` must be an `F32` `GpuTensor` of length `n × dim` laid out
/// row-major (row `i` starts at byte offset `i * dim * 4`). `scratch` is used
/// as a single-row staging buffer — `scratch.x`, `scratch.tmp`, and
/// `scratch.logits` are overwritten on every iteration. Callers that need the
/// raw logits for SWOR sampling should call this instead of running argmax
/// inside the loop.
///
/// Concretely for each row `i`:
///   1. DtoD-copy row `i` of `hidden_rows` into `scratch.x` (single F32 vector).
///   2. `rmsnorm_f32(scratch.x, weights.output_norm, scratch.tmp, eps)`.
///   3. `weight_gemv(weights.output, scratch.tmp, scratch.logits)`.
///   4. Download `scratch.logits` and append to the output buffer.
///
/// This mirrors the per-row lm_head loop in `verify_block_argmax` exactly —
/// reusing the same scratch buffers and the same kernel dispatch path — so the
/// returned logits are bit-identical to what `verify_block_argmax` would compute
/// before taking `argmax`.
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

#[cfg(test)]
mod tests {
    use super::*;

    /// Verify that the CPU `argmax` and the GPU `argmax_f32_batched` kernel
    /// share the same tie-break rule (first/lower index on a tie).
    ///
    /// The GPU kernel (`kernels/src/argmax_batched.hip`) uses strict `>` in
    /// both the per-thread scan (`if (v > lmax)`) and the tree-reduction
    /// (`if (s[i+sz] > s[i])`), so ties resolve to the lowest vocabulary index.
    /// This is identical to the CPU `argmax()` which uses the same strict `>`
    /// with a fold over the slice. The tests below exercise the CPU half;
    /// the GPU half is structurally identical (verified by code inspection).
    #[test]
    fn argmax_tiebreak_first_index_wins() {
        // Unambiguous max at index 1
        assert_eq!(argmax(&[1.0, 5.0, 3.0]), 1);
        // Tie between index 0 and 2: first (lower) index wins
        assert_eq!(argmax(&[5.0, 3.0, 5.0]), 0);
        // Tie at end: earlier index wins
        assert_eq!(argmax(&[1.0, 5.0, 5.0]), 1);
        // All equal: index 0 wins (both GPU seed = −1e30, CPU seed = NEG_INFINITY)
        assert_eq!(argmax(&[2.0, 2.0, 2.0]), 0);
        // Single element
        assert_eq!(argmax(&[7.0]), 0);
    }
}
