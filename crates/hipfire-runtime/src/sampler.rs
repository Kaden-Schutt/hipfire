// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Logit-space sampling: top-p, temperature, repeat_penalty, and
//! single-token attractor blocking. Wraps GPU-dispatched sampling
//! kernels. One entry point for any caller (daemon, examples, future
//! arch ports).
//!
//! # Why this module
//!
//! Sampling primitives (top-p kernel call, repeat-penalty window upload,
//! attractor `-INF` writes, RNG threading) used to live across
//! `hipfire_runtime::llama` (CPU primitives + GPU launch wrappers) and
//! `crates/hipfire-daemon/src/main.rs` (call-site glue). New arch ports either
//! reached into llama.rs internals or duplicated the host-side prep.
//! This module gives every caller one entry point: [`sample`], with
//! [`SamplerConfig`] holding the policy knobs.
//!
//! # Behavior preservation
//!
//! [`sample`] is a pure call-site refactor. It delegates to the same
//! `Gpu::sample_top_p` kernel and the same `memcpy_htod` for the repeat
//! window and attractor `-INF` writes that the daemon used inline. The
//! same `(logits, history, temp, top_p, repeat_penalty, repeat_window,
//! blocked_tokens, rng_state)` tuple produces the same `next_token`
//! before and after PR 3.
//!
//! # Conditional vs unconditional blocking
//!
//! The unclosed-opener attractor block (#111) decides at the call site
//! which token to block (the opener) based on a depth count over recent
//! history. The decision lives at the call site; the resulting set of
//! token IDs is passed in as [`SamplerConfig::blocked_tokens`]. The
//! sampler treats them as unconditional `-INF` writes — it does not
//! reimplement the depth counter.

use crate::llama;
use rdna_compute::{Gpu, GpuTensor};

/// Re-exports of the CPU-side sampling primitives that still live in
/// `hipfire_runtime::llama`. Other examples (`infer_qwen35`, `run`, etc.)
/// continue to call them via the `llama::` path; this module exposes
/// them via `sampler::` so new code has a single import path.
pub use crate::llama::{
    apply_ngram_block, apply_repeat_penalty, apply_repeat_penalty_candidates,
    apply_special_token_attractor_block, apply_unclosed_attractor_block, argmax,
    reset_cpu_sampler_rng, sample_top_p as sample_top_p_cpu, sample_top_p_from_candidates,
    sampler_rng_restore, sampler_rng_snapshot, SamplingConfig,
};
pub use hipfire_generate::sampler::{collect_unclosed_attractor_blocks, SamplerConfig};

/// Sample one token from a GPU-resident `logits` tensor.
///
/// Pre-dispatch host work, in order (matches the daemon's pre-PR3
/// inline sequence so byte-identical token streams are preserved):
///
///  1. Upload the trailing `min(history.len(), repeat_window,
///     repeat_buf_capacity)` tokens of `history` into `repeat_buf`.
///  2. Write `-INF` to `logits` at every offset in
///     `cfg.blocked_tokens` (one 4-byte H2D copy each).
///  3. Launch `Gpu::sample_top_p` (top-K + softmax + top-p + RNG +
///     argmax-on-greedy, all on GPU). One 8-byte D2H syncs the
///     `(token, new_rng)` result.
///
/// `rng_state` is mutated in place. For greedy (`temperature == 0.0`)
/// the value is unused but is still threaded through the kernel.
///
/// # Buffer types
///
/// `logits` is the model's output logits tensor (shape `[vocab_size]`,
/// dtype F32). `sample_buf` and `repeat_buf` are scratch buffers from
/// `llama::ForwardScratch`; the caller owns them. This matches the
/// existing pre-PR3 daemon signature exactly — we do not redesign the
/// argument shape.
pub fn sample(
    gpu: &mut Gpu,
    logits: &GpuTensor,
    sample_buf: &GpuTensor,
    repeat_buf: &GpuTensor,
    vocab_size: usize,
    history: &[u32],
    cfg: &SamplerConfig,
    rng_state: &mut u32,
) -> u32 {
    // Step 1: upload the repeat-penalty window. The kernel reads
    // `repeat_tokens[0..effective_window]`, so we only have to upload
    // the tokens that will actually be read. An empty scope is a no-op
    // (matches the first-sample case in the daemon, which used to
    // skip the htod when `bytes0` was empty).
    let buf_cap_tokens = repeat_buf.buf.size() / 4;
    let window = cfg.repeat_window.min(buf_cap_tokens);
    let scope_start = history.len().saturating_sub(window);
    let scope = &history[scope_start..];
    if !scope.is_empty() {
        let bytes: Vec<u8> = scope.iter().flat_map(|t| t.to_ne_bytes()).collect();
        let _ = gpu.hip.memcpy_htod(&repeat_buf.buf, &bytes);
    }

    // Step 2: apply unconditional blocked tokens. One 4-byte H2D per
    // token. The daemon path used `gpu_block_attractor_unclosed` which
    // wrote `-INF` to a single offset only when the depth counter
    // tripped; here the caller has already done the depth math and
    // accumulated the token IDs into `cfg.blocked_tokens`.
    if !cfg.blocked_tokens.is_empty() {
        let neg_inf: [u8; 4] = f32::NEG_INFINITY.to_ne_bytes();
        for &tok in &cfg.blocked_tokens {
            if (tok as usize) < vocab_size {
                let _ = gpu
                    .hip
                    .memcpy_htod_offset(&logits.buf, (tok as usize) * 4, &neg_inf);
            }
        }
    }

    // Step 3: GPU sample. The kernel does:
    //   - top-K = 20 from raw logits
    //   - apply repeat_penalty over `repeat_buf[0..scope.len()]`
    //   - softmax(top-K) with temperature scaling
    //   - top-p truncation
    //   - RNG draw + argmax-on-greedy fallback
    //   - writeback (token_id, new_rng) to `sample_buf`
    //   - 8-byte D2H sync (returned by the wrapper)
    let (tok, new_rng) = gpu
        .sample_top_p_pf(
            logits,
            sample_buf,
            repeat_buf,
            vocab_size,
            cfg.temperature,
            cfg.top_p,
            *rng_state,
            scope.len(),
            cfg.repeat_penalty,
            cfg.presence_penalty,
            cfg.frequency_penalty,
        )
        .expect("sample_top_p kernel launch / readback failed");
    *rng_state = new_rng;
    tok
}

/// CPU-only fallback: same math as [`sample`] but operates on a host
/// `logits` slice. Used by the VL path (`generate_vl` in daemon.rs)
/// where the argmax/top-p selection runs after a CPU-side
/// `apply_ngram_block` + `apply_repeat_penalty` pass that has no GPU
/// equivalent.
///
/// This is a thin wrapper over `llama::apply_repeat_penalty` +
/// `llama::sample_top_p` that exists so call sites have one import
/// path; the math is unchanged.
pub fn sample_cpu(logits: &mut [f32], history: &[u32], cfg: &SamplerConfig) -> u32 {
    if cfg.repeat_penalty != 1.0 && cfg.repeat_window > 0 {
        llama::apply_repeat_penalty(logits, history, cfg.repeat_window, cfg.repeat_penalty);
    }
    if (cfg.presence_penalty > 0.0 || cfg.frequency_penalty > 0.0) && cfg.repeat_window > 0 {
        let start = history.len().saturating_sub(cfg.repeat_window);
        let mut counts = std::collections::HashMap::<u32, f32>::new();
        for &tok in &history[start..] {
            *counts.entry(tok).or_insert(0.0) += 1.0;
        }
        for (tok, count) in counts {
            if (tok as usize) < logits.len() {
                logits[tok as usize] -= cfg.frequency_penalty * count + cfg.presence_penalty;
            }
        }
    }
    for &tok in &cfg.blocked_tokens {
        if (tok as usize) < logits.len() {
            logits[tok as usize] = f32::NEG_INFINITY;
        }
    }
    llama::sample_top_p(logits, cfg.temperature, cfg.top_p)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sample_cpu_applies_presence_and_frequency_penalties() {
        let mut logits = vec![0.0_f32, 4.0, 3.5, 3.0];
        let cfg = SamplerConfig {
            temperature: 0.0,
            top_p: 1.0,
            repeat_penalty: 1.0,
            repeat_window: 8,
            presence_penalty: 1.0,
            frequency_penalty: 0.5,
            blocked_tokens: Vec::new(),
        };
        let tok = sample_cpu(&mut logits, &[1, 1, 2], &cfg);
        assert_eq!(tok, 3);
        assert!((logits[1] - 2.0).abs() < 1e-6);
        assert!((logits[2] - 2.0).abs() < 1e-6);
        assert!((logits[3] - 3.0).abs() < 1e-6);
    }

    #[test]
    fn sample_cpu_greedy_picks_argmax() {
        // sample_cpu with greedy SamplerConfig should return the
        // argmax of the logits — even when blocked_tokens or
        // repeat_penalty would otherwise mutate the slice.
        let mut logits = vec![1.0_f32, 5.0, 2.0, 7.0, 3.0];
        let cfg = SamplerConfig::greedy();
        let tok = sample_cpu(&mut logits, &[], &cfg);
        assert_eq!(tok, 3);
    }

    #[test]
    fn sample_cpu_blocks_tokens() {
        // A blocked token should never be the argmax even if it
        // started as the largest logit. The blocker is unconditional
        // (a -INF write) so the next-best token wins.
        let mut logits = vec![1.0_f32, 5.0, 2.0, 7.0, 3.0];
        let mut cfg = SamplerConfig::greedy();
        cfg.blocked_tokens = vec![3];
        let tok = sample_cpu(&mut logits, &[], &cfg);
        assert_eq!(tok, 1);
    }

    #[test]
    fn sample_cpu_blocked_tokens_out_of_range_skipped() {
        // Out-of-range token IDs are silently skipped — the GPU path
        // does the same `(tok as usize) < vocab_size` guard.
        let mut logits = vec![1.0_f32, 5.0, 2.0, 7.0, 3.0];
        let mut cfg = SamplerConfig::greedy();
        cfg.blocked_tokens = vec![999, 1234];
        let tok = sample_cpu(&mut logits, &[], &cfg);
        assert_eq!(tok, 3); // argmax unchanged
    }
}
