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
//! `examples/daemon.rs` (call-site glue). New arch ports either
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
    sample_top_p as sample_top_p_cpu, sample_top_p_from_candidates, sampler_rng_restore,
    sampler_rng_snapshot, SamplingConfig,
};

/// Sampler policy knobs for a single token sample.
///
/// `temperature == 0.0` is the greedy path (the kernel falls back to
/// argmax internally). `top_p == 1.0` disables nucleus truncation.
/// `repeat_penalty == 1.0` (with any `repeat_window`) is a no-op.
///
/// `blocked_tokens` are unconditional `-INF` writes applied directly to
/// the on-GPU logits buffer before the sampling kernel launches. The
/// daemon populates this list per-token from its unclosed-opener depth
/// counter (#111); a future caller could populate it from anywhere.
#[derive(Debug, Clone)]
pub struct SamplerConfig {
    /// 0.0 = greedy (kernel argmax fast path).
    pub temperature: f32,
    /// 1.0 = no nucleus truncation.
    pub top_p: f32,
    /// Top-k cap on candidate count BEFORE top-p / min-p truncation.
    /// `0` = disabled (kernel uses its built-in TOP_K=20 buffer cap).
    /// Otherwise clamped to `[1, 20]` by the kernel.
    /// HF generation_config recommends 20 for Qwen3.5/3.6 MoE.
    pub top_k: i32,
    /// Minimum probability relative to the modal token's probability
    /// (i.e. `min_p × P(top_token)`). `0.0` = disabled. Typical 0.05 for
    /// thinkers — drops the long tail without competing with top_p's
    /// cumulative cap. Applied after top_k, before top_p.
    pub min_p: f32,
    /// 1.0 = repeat-penalty disabled.
    pub repeat_penalty: f32,
    /// Tokens of recent history visible to the repeat-penalty kernel.
    /// Effective window is `min(history.len(), repeat_window)` and is
    /// also clipped to the GPU `repeat_buf` capacity by the caller.
    pub repeat_window: usize,
    /// OpenAI-style presence penalty: subtract this constant from the
    /// logit of any token that has appeared at least once in
    /// `repeat_window`. Distinct from `repeat_penalty` (which divides
    /// the logit by RP) — additive, applied once per unique token,
    /// independent of count and magnitude. `0.0` = disabled.
    /// Qwen3.5/3.6 generation_config recommends `1.5` for
    /// general-purpose thinking. Implemented as a host-side
    /// download → mutate-uniques → upload pre-pass over the full
    /// logits buffer, so it costs `vocab_size × 4 × 2` bytes of
    /// PCIe traffic per sample when active. Acceptable for low-
    /// volume sampling on the AR path; not yet wired into the
    /// DFlash spec-decode batched verify.
    pub presence_penalty: f32,
    /// Token IDs whose logit is unconditionally set to `-INF` before
    /// sampling. Used for the unclosed-opener attractor block (#111).
    pub blocked_tokens: Vec<u32>,
}

impl SamplerConfig {
    /// Greedy: temperature=0, top_p=1, repeat_penalty=1, no blocks.
    /// The kernel takes the argmax fast path; RNG state is unused.
    pub fn greedy() -> Self {
        Self {
            temperature: 0.0,
            top_p: 1.0,
            top_k: 0,
            min_p: 0.0,
            repeat_penalty: 1.0,
            repeat_window: 0,
            presence_penalty: 0.0,
            blocked_tokens: Vec::new(),
        }
    }

    /// HuggingFace-recommended sampling for Qwen3.5/3.6 MoE thinkers,
    /// matching `generation_config.json` (`temperature=1.0, top_k=20,
    /// top_p=0.95`) plus a conservative `min_p=0.05` to suppress the
    /// long-tail trash candidates that show up at MQ4 quant. No repeat
    /// penalty (HF doesn't recommend one for these models).
    pub fn hf_thinker() -> Self {
        Self {
            temperature: 1.0,
            top_p: 0.95,
            top_k: 20,
            min_p: 0.05,
            repeat_penalty: 1.0,
            repeat_window: 0,
            presence_penalty: 0.0,
            blocked_tokens: Vec::new(),
        }
    }
}

impl Default for SamplerConfig {
    /// Daemon-default: temperature=0.3, top_p=0.95, repeat_penalty=1.05.
    /// Mirrors the user-validated `RP=1.05` floor (CLAUDE.md memory:
    /// `feedback_repeat_penalty_default.md`). `repeat_window=128`
    /// matches `hipfire_runtime::llama::SamplingConfig::text_thinking()`.
    /// `top_k=0` and `min_p=0.0` keep current pre-top_k/min_p behavior;
    /// callers that want HF-recommended sampling should use
    /// `SamplerConfig::hf_thinker()` or set the fields explicitly.
    fn default() -> Self {
        Self {
            temperature: 0.3,
            top_p: 0.95,
            top_k: 0,
            min_p: 0.0,
            repeat_penalty: 1.05,
            repeat_window: 128,
            presence_penalty: 0.0,
            blocked_tokens: Vec::new(),
        }
    }
}

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

    // Step 2.5: presence-penalty host-side pre-pass. Subtract
    // `presence_penalty` once from the logit of any token that
    // appears at least once in the recent window. Implemented as
    // download → mutate → upload of the full logits buffer
    // because the GPU `sample_top_p` kernel doesn't yet have a
    // PP slot. Costs `vocab_size × 4 × 2` bytes of PCIe per sample
    // when active; on the AR path that's ~2 MB / token at vocab
    // 248k, tolerable for smoke tests at the 100 tok/s rate. Any
    // future caller wanting PP in the DFlash spec-decode batched
    // verify will need a kernel-side hook; this CPU path keeps the
    // patch small enough to validate the magnitude before spending
    // kernel time.
    //
    // The `Vec<u8>` host buffer is touched only at the unique-token
    // offsets via explicit `f32::from_ne_bytes` / `to_ne_bytes`
    // round-trips. We do NOT reinterpret the byte buffer as
    // `&mut [f32]` — `Vec<u8>` is only u8-aligned, and a wider
    // load through a misaligned pointer is UB. The per-unique-token
    // byte-juggling cost is negligible (~30 ns each) compared to
    // the PCIe round-trip that dominates this path.
    if cfg.presence_penalty != 0.0 && !scope.is_empty() {
        let logits_bytes = logits.buf.size();
        let mut host_bytes = vec![0u8; logits_bytes];
        let _ = gpu.hip.memcpy_dtoh(&mut host_bytes, &logits.buf);
        let n_logits = logits_bytes / 4;
        let mut seen = std::collections::HashSet::with_capacity(scope.len());
        for &t in scope {
            if seen.insert(t) && (t as usize) < n_logits {
                let off = (t as usize) * 4;
                let mut le = [0u8; 4];
                le.copy_from_slice(&host_bytes[off..off + 4]);
                let v = f32::from_ne_bytes(le) - cfg.presence_penalty;
                host_bytes[off..off + 4].copy_from_slice(&v.to_ne_bytes());
            }
        }
        let _ = gpu.hip.memcpy_htod(&logits.buf, &host_bytes);
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
        .sample_top_p(
            logits,
            sample_buf,
            repeat_buf,
            vocab_size,
            cfg.temperature,
            cfg.top_p,
            *rng_state,
            scope.len(),
            cfg.repeat_penalty,
            cfg.top_k,
            cfg.min_p,
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
    if cfg.presence_penalty != 0.0 && cfg.repeat_window > 0 && !history.is_empty() {
        let scope_start = history.len().saturating_sub(cfg.repeat_window);
        let scope = &history[scope_start..];
        let mut seen = std::collections::HashSet::with_capacity(scope.len());
        for &t in scope {
            if seen.insert(t) && (t as usize) < logits.len() {
                logits[t as usize] -= cfg.presence_penalty;
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

/// Compute the unclosed-opener attractor blocked-token list (#111).
///
/// Counts unclosed openers in the trailing `window` tokens of `history`
/// (as `opens - closes`, floored at zero). When the running depth
/// reaches `threshold`, the opener is appended to `out`. The
/// downstream sampler will write `-INF` to that token's logit so the
/// next sample cannot stack another nested opener.
///
/// `pairs` is a slice of `(open, close)` pairs (e.g.
/// `(<tool_call>, </tool_call>)` and `(<think>, </think>)`); only
/// pairs whose `open` clears the threshold contribute. With
/// `threshold = 2`, a second consecutive opener without an intervening
/// closer is the last one the decoder is allowed to emit.
///
/// Pure, no GPU work; the caller passes the result into
/// [`SamplerConfig::blocked_tokens`].
pub fn collect_unclosed_attractor_blocks(
    history: &[u32],
    pairs: &[(u32, u32)],
    window: usize,
    threshold: usize,
    out: &mut Vec<u32>,
) {
    if window == 0 || threshold == 0 {
        return;
    }
    let start = history.len().saturating_sub(window);
    let recent = &history[start..];
    for &(open_id, close_id) in pairs {
        let mut depth: i32 = 0;
        for &t in recent {
            if t == open_id {
                depth += 1;
            } else if t == close_id && depth > 0 {
                depth -= 1;
            }
        }
        if depth >= threshold as i32 {
            out.push(open_id);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn greedy_config_fields() {
        let g = SamplerConfig::greedy();
        assert_eq!(g.temperature, 0.0);
        assert_eq!(g.top_p, 1.0);
        assert_eq!(g.repeat_penalty, 1.0);
        assert_eq!(g.repeat_window, 0);
        assert!(g.blocked_tokens.is_empty());
    }

    #[test]
    fn default_config_fields() {
        let d = SamplerConfig::default();
        assert!((d.temperature - 0.3).abs() < 1e-6);
        assert!((d.top_p - 0.95).abs() < 1e-6);
        assert!((d.repeat_penalty - 1.05).abs() < 1e-6);
        assert_eq!(d.repeat_window, 128);
        assert!(d.blocked_tokens.is_empty());
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
    fn collect_unclosed_blocks_appends_when_depth_reached() {
        // history has 2 unclosed `<tool_call>` (id=10) — depth=2
        // hits threshold=2, so 10 should be in `out`. `<think>`
        // (id=20, close=21) has 1 unclosed → below threshold, not
        // appended.
        let history = [10u32, 99, 10, 5, 20, 7];
        let pairs = [(10u32, 11u32), (20u32, 21u32)];
        let mut out = Vec::new();
        collect_unclosed_attractor_blocks(&history, &pairs, 20, 2, &mut out);
        assert_eq!(out, vec![10]);
    }

    #[test]
    fn collect_unclosed_blocks_zero_threshold_is_noop() {
        let history = [10u32, 10, 10];
        let pairs = [(10u32, 11u32)];
        let mut out = Vec::new();
        collect_unclosed_attractor_blocks(&history, &pairs, 20, 0, &mut out);
        assert!(out.is_empty());
    }

    #[test]
    fn collect_unclosed_blocks_balanced_open_close_does_not_block() {
        // 2 opens, 2 closes → depth=0, never trips.
        let history = [10u32, 5, 11, 10, 7, 11];
        let pairs = [(10u32, 11u32)];
        let mut out = Vec::new();
        collect_unclosed_attractor_blocks(&history, &pairs, 20, 2, &mut out);
        assert!(out.is_empty());
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
