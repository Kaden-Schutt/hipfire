// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire - see LICENSE and NOTICE in the project root.

//! Compatibility wrapper for generation loop-guard policy.
//!
//! The implementation lives in `hipfire-generate`; this module preserves the
//! `hipfire_runtime::loop_guard` path for source-compatible callers.

pub use hipfire_generate::loop_guard::StopReason;

/// Per-request generation loop guard.
pub struct LoopGuard {
    inner: hipfire_generate::loop_guard::LoopGuard,
}

impl LoopGuard {
    /// Construct a guard from the runtime's resolved configuration.
    pub fn from_config(config: &crate::config::RuntimeConfig) -> Self {
        Self::new(config.ngram_loop_threshold, config.ngram_window)
    }

    /// Construct with explicit threshold and window. `threshold = 0` disables
    /// the guard.
    pub fn new(threshold: usize, window: usize) -> Self {
        Self {
            inner: hipfire_generate::loop_guard::LoopGuard::new(threshold, window),
        }
    }

    /// A guard that never fires.
    pub fn off() -> Self {
        Self {
            inner: hipfire_generate::loop_guard::LoopGuard::off(),
        }
    }

    /// Whether the guard is currently active.
    pub fn enabled(&self) -> bool {
        self.inner.enabled()
    }

    /// Inspect streamed tokens and return a stop reason when a loop triggers.
    pub fn check(&self, streamed_tokens: &[u32]) -> Option<StopReason> {
        self.inner.check(streamed_tokens)
    }

    /// Active inspection-window size for logging.
    pub fn window_len(&self, streamed_tokens_len: usize) -> usize {
        self.inner.window_len(streamed_tokens_len)
    }
}
