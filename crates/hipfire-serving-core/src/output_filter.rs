// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Prompt-input normalization, output/stop filtering, and the sampling-guard
//! attractor blockers. Extracted verbatim from the former `main.rs` monolith
//! (no behavior change).
//!
//! - `normalize_daemon_prompt` — collapse runaway whitespace before tokenizing.
//! - `chat_output_filter_from_profile` / `normalize_request_stop_sequences` —
//!   build the `EosFilter` stop/holdback set from the chat-template profile plus
//!   per-request stop sequences.
//! - `loop_guard_from_runtime_config` — n-gram loop guard from runtime config.
//! - `block_attractor_unclosed_cpu` / `gpu_block_attractor_token` — break
//!   structured-output special-token attractors (#111).

use hipfire_generate::eos_filter::{EosFilter, EosFilterConfig};
use hipfire_generate::loop_guard::LoopGuard;
use hipfire_prompt as prompt_frame;

use crate::model::LoadedModel;

/// Collapse runaway whitespace (3+ newlines → 2, etc.) before tokenizing, so
/// prompts that differ only in blank-line padding tokenize identically — a
/// material τ/throughput stabilizer (see CLAUDE.md "Prompt-structure τ
/// sensitivity"). Borrows unchanged when normalization is disabled
/// (`HIPFIRE_NORMALIZE_PROMPT=0` or the runtime config flag is off).
pub fn normalize_daemon_prompt(prompt: &str) -> std::borrow::Cow<'_, str> {
    if matches!(
        std::env::var("HIPFIRE_NORMALIZE_PROMPT").ok().as_deref(),
        Some("0") | Some("false") | Some("off") | Some("no")
    ) || !hipfire_runtime::config::get().normalize_prompt
    {
        return std::borrow::Cow::Borrowed(prompt);
    }

    hipfire_prompt::normalize_prompt_text_with_policy(prompt, true)
}

/// GPU-side attractor blockers for the AR generate path (#111).
///
/// MQ4 quant pressure makes structured-output special tokens (`<tool_call>`,
/// `<think>`) into self-reinforcing attractors: the model emits the same
/// special token hundreds of times in a row, never reaching the JSON body
/// (or in stacked-opener shapes that downstream regex parsers cannot
/// recover). The CPU-side `apply_ngram_block` is not in this path (its
/// per-token D2H + H2D would tank decode tok/s) and the GPU sampler's
/// repeat-penalty alone doesn't break a strong single-token loop fast
/// enough at the user-validated `RP=1.05` floor.
///
/// The unclosed-opener depth counter lives in
/// `hipfire_generate::sampler::collect_unclosed_attractor_blocks`; the resulting
/// blocked-token list is applied to the GPU logits buffer by
/// `hipfire_runtime::sampler::sample`
/// before the sampling kernel launches. The `gpu_block_attractor_token`
/// helper below is the simpler fallback for unpaired tokens — trips on
/// `count >= threshold` regardless of structure — kept here as
/// reference for a future per-token attractor block.
/// CPU-side counterpart that applies the same depth-tracking attractor
/// block directly to a freshly-downloaded logits vector. Avoids the
/// htod-memcpy + redownload roundtrip the GPU variant required per token.
pub fn block_attractor_unclosed_cpu(
    logits: &mut [f32],
    history: &[u32],
    open_id: u32,
    close_id: u32,
    window: usize,
    threshold: usize,
) {
    if window == 0 || threshold == 0 || open_id == close_id {
        return;
    }
    let start = history.len().saturating_sub(window);
    let mut depth: i32 = 0;
    for &t in &history[start..] {
        if t == open_id {
            depth += 1;
        } else if t == close_id && depth > 0 {
            depth -= 1;
        }
    }
    if depth >= threshold as i32 {
        if let Some(slot) = logits.get_mut(open_id as usize) {
            *slot = f32::NEG_INFINITY;
        }
    }
}

/// Build the n-gram repetition [`LoopGuard`] from the runtime config
/// (`ngram_loop_threshold` / `ngram_window`) — the decode-loop guard that breaks
/// degenerate verbatim repetition.
pub fn loop_guard_from_runtime_config() -> LoopGuard {
    let config = hipfire_runtime::config::get();
    LoopGuard::new(config.ngram_loop_threshold, config.ngram_window)
}

/// Simpler unpaired-token fallback to [`block_attractor_unclosed_cpu`]: writes
/// `-inf` straight into the GPU logits buffer for `tok_id` once it repeats
/// `>= threshold` times in the last `window` tokens, regardless of open/close
/// structure. Currently unused — kept as reference for a future per-token block.
#[allow(dead_code)]
pub fn gpu_block_attractor_token(
    gpu: &rdna_compute::Gpu,
    logits_buf: &hip_bridge::DeviceBuffer,
    history: &[u32],
    tok_id: u32,
    window: usize,
    threshold: usize,
) {
    if window == 0 || threshold == 0 {
        return;
    }
    let start = history.len().saturating_sub(window);
    let count = history[start..].iter().filter(|&&t| t == tok_id).count();
    if count >= threshold {
        let bytes: [u8; 4] = f32::NEG_INFINITY.to_ne_bytes();
        let _ = gpu
            .hip
            .memcpy_htod_offset(logits_buf, (tok_id as usize) * 4, &bytes);
    }
}

/// Build the streaming [`EosFilter`] for a turn: the chat-template profile's
/// `stop_at`/`holdback_prefixes` (defaulting to `<|im_end|>` when the profile has
/// none) unioned with the per-request stop sequences. Holdback prefixes let the
/// filter buffer a partial stop token rather than leaking it before the match
/// completes.
pub fn chat_output_filter(m: &LoadedModel, request_stop_sequences: &[String]) -> EosFilter {
    chat_output_filter_from_profile(m.chat_template_profile.as_ref(), request_stop_sequences)
}

pub fn chat_output_filter_from_profile(
    chat_template_profile: Option<&prompt_frame::ChatTemplateProfile>,
    request_stop_sequences: &[String],
) -> EosFilter {
    let (mut stop_at, mut holdback_prefixes) = chat_template_profile
        .map(|profile| {
            (
                profile
                    .stop_at
                    .iter()
                    .map(|s| s.as_bytes().to_vec())
                    .collect::<Vec<_>>(),
                profile
                    .holdback_prefixes
                    .iter()
                    .map(|s| s.as_bytes().to_vec())
                    .collect::<Vec<_>>(),
            )
        })
        .filter(|(stop_at, _)| !stop_at.is_empty())
        .unwrap_or_else(|| (vec![b"<|im_end|>".to_vec()], vec![b"<|im_end|>".to_vec()]));

    for stop in request_stop_sequences {
        if stop.is_empty() {
            continue;
        }
        let bytes = stop.as_bytes().to_vec();
        if !stop_at.iter().any(|existing| existing == &bytes) {
            stop_at.push(bytes.clone());
        }
        if !holdback_prefixes.iter().any(|existing| existing == &bytes) {
            holdback_prefixes.push(bytes);
        }
    }

    EosFilter::new(EosFilterConfig {
        stop_at,
        holdback_prefixes,
        ..Default::default()
    })
}

/// Parse and sanitize a request's `stop` field (string or array) the same way
/// the legacy server did: drop empties, cap at 4 sequences, truncate each to 64
/// bytes. Bounds adversarial/oversized input before it reaches the filter.
pub fn normalize_request_stop_sequences(value: Option<&serde_json::Value>) -> Vec<String> {
    let Some(value) = value else {
        return Vec::new();
    };
    let mut out = match value {
        serde_json::Value::String(s) => vec![s.clone()],
        serde_json::Value::Array(items) => items
            .iter()
            .filter_map(|item| item.as_str().map(ToOwned::to_owned))
            .collect::<Vec<_>>(),
        _ => Vec::new(),
    };
    out.retain(|s| !s.is_empty());
    out.truncate(4);
    for seq in &mut out {
        if seq.len() > 64 {
            seq.truncate(64);
        }
    }
    out
}

#[cfg(test)]
mod output_filter_tests {
    use super::*;
    use hipfire_generate::eos_filter::FilterAction;

    #[test]
    fn request_stop_sequences_are_normalized_like_bun() {
        let value = serde_json::json!(["", "END", "x".repeat(80), "A", "B", "C"]);
        let stops = normalize_request_stop_sequences(Some(&value));
        assert_eq!(stops.len(), 4);
        assert_eq!(stops[0], "END");
        assert_eq!(stops[1].len(), 64);
    }

    #[test]
    fn request_stop_sequence_stops_filter_output() {
        let stops = vec!["END".to_string()];
        let mut filter = chat_output_filter_from_profile(None, &stops);
        match filter.observe(b"hello END hidden") {
            FilterAction::StopEmit(bytes) => {
                assert_eq!(std::str::from_utf8(&bytes).unwrap(), "hello ");
            }
            other => panic!("expected emitted prefix before stop, got {other:?}"),
        }
        assert!(matches!(
            filter.observe(b" after"),
            FilterAction::Stop | FilterAction::Hold
        ));
    }
}
