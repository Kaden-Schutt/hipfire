// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Stateless / per-request guards on the emitted token stream.
//! Pure Rust, no GPU calls. Detects pathological generation patterns
//! and signals when generation should be force-stopped.
//!
//! The n-gram loop detector tracks 4-gram token sequences over a rolling
//! window of recently emitted tokens. When any 4-gram repeats more than
//! `ngram_threshold` times in the last `ngram_window` tokens, the guard
//! fires. This catches answer-phase repetition loops that the think cap
//! and repeat-penalty miss. Operates on token IDs only — no decode
//! overhead.
//!
//! Behavior is byte-identical to the inline detector previously in
//! `crates/hipfire-runtime/examples/daemon.rs`. The defaults are read from
//! `HIPFIRE_NGRAM_LOOP_THRESHOLD` (default 8, 0 = disabled) and
//! `HIPFIRE_NGRAM_WINDOW` (default 256). The threshold is `>=`, not `>`.

use std::collections::HashMap;

/// Why the guard signalled a stop.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StopReason {
    /// A 4-gram token sequence repeated `count` times within the active window.
    NgramRepeat {
        /// The 4-gram tokens in stream order.
        ngram: [u32; 4],
        /// Total occurrences of `ngram` inside the inspected window.
        count: usize,
    },
}

/// Per-request guard state. Cheap to construct — holds only configuration.
pub struct LoopGuard {
    ngram_threshold: usize,
    ngram_window: usize,
    enabled: bool,
}

impl LoopGuard {
    /// Construct a guard from environment variables.
    ///
    /// - `HIPFIRE_NGRAM_LOOP_THRESHOLD` (default 8): a 4-gram count of this
    ///   value or higher inside the window triggers the guard. Set to 0 to
    ///   disable the guard entirely.
    /// - `HIPFIRE_NGRAM_WINDOW` (default 256): how many trailing tokens to
    ///   inspect on each `check` call.
    pub fn from_config(config: &crate::config::RuntimeConfig) -> Self {
        Self::new(config.ngram_loop_threshold, config.ngram_window)
    }

    /// Construct with explicit threshold and window. `threshold = 0` disables
    /// the guard (matches the env-var convention).
    pub fn new(threshold: usize, window: usize) -> Self {
        Self {
            ngram_threshold: threshold,
            ngram_window: window,
            enabled: threshold > 0,
        }
    }

    /// A guard that never fires.
    pub fn off() -> Self {
        Self {
            ngram_threshold: 0,
            ngram_window: 0,
            enabled: false,
        }
    }

    /// Whether the guard is currently active.
    pub fn enabled(&self) -> bool {
        self.enabled
    }

    /// Inspect the streamed token slice and return `Some(StopReason)` when a
    /// loop pattern triggers. Pure function of `(self, streamed_tokens)`.
    ///
    /// Mirrors the prior daemon inline check exactly:
    /// - Disabled when `enabled == false`.
    /// - Skips when fewer than 4 tokens have been streamed.
    /// - Slices the trailing `ngram_window` tokens (`saturating_sub`).
    /// - Skips when the resulting slice has fewer than 4 tokens.
    /// - Counts every 4-gram via a `HashMap<[u32; 4], usize>` and triggers
    ///   when any count is `>= ngram_threshold`.
    pub fn check(&self, streamed_tokens: &[u32]) -> Option<StopReason> {
        if !self.enabled {
            return None;
        }
        if streamed_tokens.len() < 4 {
            return None;
        }
        let window_start = streamed_tokens.len().saturating_sub(self.ngram_window);
        let window = &streamed_tokens[window_start..];
        if window.len() < 4 {
            return None;
        }
        let mut ngram_counts = HashMap::<[u32; 4], usize>::new();
        for w in window.windows(4) {
            let key = [w[0], w[1], w[2], w[3]];
            *ngram_counts.entry(key).or_insert(0) += 1;
        }
        let (max_ngram, max_count) = ngram_counts
            .iter()
            .max_by_key(|(_, c)| **c)
            .map(|(k, c)| (*k, *c))
            .unwrap_or(([0; 4], 0));
        if max_count >= self.ngram_threshold {
            Some(StopReason::NgramRepeat {
                ngram: max_ngram,
                count: max_count,
            })
        } else {
            None
        }
    }

    /// Active inspection-window size. Useful for log/info messages so the
    /// caller can report the same `window.len()` value the inline code did.
    pub fn window_len(&self, streamed_tokens_len: usize) -> usize {
        let window_start = streamed_tokens_len.saturating_sub(self.ngram_window);
        streamed_tokens_len - window_start
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn threshold_zero_disables() {
        let guard = LoopGuard::new(0, 256);
        assert!(!guard.enabled());
        // Even an obvious loop must not trigger.
        let mut tokens: Vec<u32> = Vec::new();
        for _ in 0..50 {
            tokens.extend_from_slice(&[1, 2, 3, 4]);
        }
        assert_eq!(guard.check(&tokens), None);
    }

    #[test]
    fn eight_repeats_of_4gram_triggers() {
        let guard = LoopGuard::new(8, 256);
        let mut tokens: Vec<u32> = Vec::new();
        // Pad with non-repeating prefix so the trigger source is unambiguous.
        tokens.extend_from_slice(&[100, 101, 102, 103, 104]);
        // 8 repeats of [1, 2, 3, 4] — overlapping windows produce >= 8
        // occurrences of that 4-gram inside the rolling window.
        for _ in 0..8 {
            tokens.extend_from_slice(&[1, 2, 3, 4]);
        }
        match guard.check(&tokens) {
            Some(StopReason::NgramRepeat { ngram, count }) => {
                assert_eq!(ngram, [1, 2, 3, 4]);
                assert!(count >= 8, "expected count >= 8, got {}", count);
            }
            None => panic!("guard should have fired on 8× repeats of [1,2,3,4]"),
        }
    }

    #[test]
    fn old_4gram_outside_window_does_not_count() {
        // Threshold 8, window 16. We jam 8 repeats of [1,2,3,4] at the start
        // (outside the trailing 16-token window) and follow with 16 unique
        // tokens. The trailing window must contain only the unique tail and
        // therefore must not trigger.
        let guard = LoopGuard::new(8, 16);
        let mut tokens: Vec<u32> = Vec::new();
        for _ in 0..8 {
            tokens.extend_from_slice(&[1, 2, 3, 4]);
        }
        // 16 unique trailing tokens.
        for i in 200..216u32 {
            tokens.push(i);
        }
        assert_eq!(
            guard.check(&tokens),
            None,
            "old 4-gram outside window must not fire the guard"
        );
    }

    #[test]
    fn fewer_than_four_tokens_returns_none() {
        let guard = LoopGuard::new(8, 256);
        assert_eq!(guard.check(&[]), None);
        assert_eq!(guard.check(&[1]), None);
        assert_eq!(guard.check(&[1, 2]), None);
        assert_eq!(guard.check(&[1, 2, 3]), None);
    }
}
