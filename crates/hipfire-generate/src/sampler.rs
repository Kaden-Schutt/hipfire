// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire - see LICENSE and NOTICE in the project root.

//! Pure generation sampling policy contracts.
//!
//! GPU/CPU sampling execution still lives in `hipfire-runtime` because it
//! depends on runtime scratch tensors and kernel launch wrappers. This module
//! owns the policy values and token-history guard helpers shared by those
//! execution paths.

/// Sampler policy knobs for a single token sample.
///
/// `temperature == 0.0` is the greedy path. `top_p == 1.0` disables nucleus
/// truncation. `repeat_penalty == 1.0` is a no-op.
#[derive(Debug, Clone, PartialEq)]
pub struct SamplerConfig {
    /// 0.0 = greedy.
    pub temperature: f32,
    /// 1.0 = no nucleus truncation.
    pub top_p: f32,
    /// 1.0 = repeat-penalty disabled.
    pub repeat_penalty: f32,
    /// Tokens of recent history visible to repeat/frequency penalties.
    pub repeat_window: usize,
    /// OpenAI `presence_penalty`: flat logit subtraction applied once to any
    /// token that occurred within `repeat_window`. 0.0 = disabled.
    pub presence_penalty: f32,
    /// OpenAI `frequency_penalty`: logit subtraction scaled by the token's
    /// occurrence count within `repeat_window`. 0.0 = disabled.
    pub frequency_penalty: f32,
    /// Token IDs whose logit is unconditionally set to `-INF` before sampling.
    pub blocked_tokens: Vec<u32>,
}

impl SamplerConfig {
    /// Greedy: temperature=0, top_p=1, repeat_penalty=1, no blocks.
    pub fn greedy() -> Self {
        Self {
            temperature: 0.0,
            top_p: 1.0,
            repeat_penalty: 1.0,
            repeat_window: 0,
            presence_penalty: 0.0,
            frequency_penalty: 0.0,
            blocked_tokens: Vec::new(),
        }
    }
}

impl Default for SamplerConfig {
    /// Daemon-default: temperature=0.3, top_p=0.95, repeat_penalty=1.05.
    ///
    /// `repeat_window=128` matches the existing text-thinking sampler policy.
    fn default() -> Self {
        Self {
            temperature: 0.3,
            top_p: 0.95,
            repeat_penalty: 1.05,
            repeat_window: 128,
            presence_penalty: 0.0,
            frequency_penalty: 0.0,
            blocked_tokens: Vec::new(),
        }
    }
}

/// Compute the unclosed-opener attractor blocked-token list.
///
/// Counts unclosed openers in the trailing `window` tokens of `history` (as
/// `opens - closes`, floored at zero). When the running depth reaches
/// `threshold`, the opener is appended to `out`. The downstream runtime sampler
/// writes `-INF` to those token logits before drawing.
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
        assert_eq!(g.presence_penalty, 0.0);
        assert_eq!(g.frequency_penalty, 0.0);
        assert!(g.blocked_tokens.is_empty());
    }

    #[test]
    fn default_config_fields() {
        let d = SamplerConfig::default();
        assert!((d.temperature - 0.3).abs() < 1e-6);
        assert!((d.top_p - 0.95).abs() < 1e-6);
        assert!((d.repeat_penalty - 1.05).abs() < 1e-6);
        assert_eq!(d.repeat_window, 128);
        assert_eq!(d.presence_penalty, 0.0);
        assert_eq!(d.frequency_penalty, 0.0);
        assert!(d.blocked_tokens.is_empty());
    }

    #[test]
    fn collect_unclosed_blocks_appends_when_depth_reached() {
        let history = vec![10, 99, 10, 20, 21, 20];
        let pairs = vec![(10, 11), (20, 21)];
        let mut out = Vec::new();
        collect_unclosed_attractor_blocks(&history, &pairs, 20, 2, &mut out);
        assert_eq!(out, vec![10]);
    }

    #[test]
    fn collect_unclosed_blocks_zero_threshold_is_noop() {
        let history = vec![10, 10, 10];
        let pairs = vec![(10, 11)];
        let mut out = Vec::new();
        collect_unclosed_attractor_blocks(&history, &pairs, 20, 0, &mut out);
        assert!(out.is_empty());
    }

    #[test]
    fn collect_unclosed_blocks_balanced_open_close_does_not_block() {
        let history = vec![10, 10, 11, 11, 10];
        let pairs = vec![(10, 11)];
        let mut out = Vec::new();
        collect_unclosed_attractor_blocks(&history, &pairs, 20, 2, &mut out);
        assert!(out.is_empty());
    }
}
