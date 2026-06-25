// SPDX-License-Identifier: Apache-2.0
//! Warn-on-generic-fallback — the reference kernel layer's runtime coverage map.
//!
//! The reference kernel layer (docs/plans/2026-06-22-reference-kernel-layer.md) keeps
//! a boring, correct kernel for every (op × mode × precision) as the dispatch floor.
//! When the dispatcher runs one of those *because no optimized overlay matched*, it
//! calls [`warn_generic_once`]. That emits ONE stderr warning per distinct
//! `(op, precision, mode, arch)` tuple — hot loops never spam — turning fallbacks into
//! a readable signal: "where is optimization missing for this config?".
//!
//! Two side-uses fall out:
//!   * eval/CI can assert "no generic fallback on the hot path for config X" via
//!     [`generic_fallback_count`] (reset with [`reset_generic_warnings`]).
//!   * completeness-only tiers (e.g. W4A4 — kept for matrix coverage but known to be
//!     low-quality) pass [`Quality::CompletenessOnly`], so the warning flags that the
//!     selected path is not production-quality, not merely unoptimized.
//!
//! Silence with `HIPFIRE_WARN_GENERIC=0` (default: on, warn-once).

use std::collections::HashSet;
use std::sync::{Mutex, OnceLock};

/// Forward shape the kernel runs in. Decode = single-token (GEMV-shaped); Prefill =
/// batched (GEMM-shaped). They are distinct reference kernels.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum KernelMode {
    Decode,
    Prefill,
}

impl KernelMode {
    pub fn as_str(self) -> &'static str {
        match self {
            KernelMode::Decode => "decode",
            KernelMode::Prefill => "prefill",
        }
    }
}

/// Quality contract of the reference kernel being run.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Quality {
    /// Correct + reasonable; just lacks an optimized overlay for this arch.
    Reference,
    /// In the stack for matrix completeness only, KNOWN to produce poor output
    /// (e.g. W4A4 — the activation-precision cliff). Numerics-correct, not
    /// quality-gated; never the default, never claimed production.
    CompletenessOnly,
}

fn enabled() -> bool {
    static E: OnceLock<bool> = OnceLock::new();
    *E.get_or_init(|| std::env::var("HIPFIRE_WARN_GENERIC").ok().as_deref() != Some("0"))
}

fn seen() -> &'static Mutex<HashSet<String>> {
    static S: OnceLock<Mutex<HashSet<String>>> = OnceLock::new();
    S.get_or_init(|| Mutex::new(HashSet::new()))
}

/// Record (and, the first time per distinct tuple, warn about) a generic-reference
/// kernel selection. `op` e.g. "gate_up"; `precision` e.g. "W4A8"/"iu8"; `arch` e.g.
/// "gfx1103". Cheap + thread-safe; safe to call on every dispatch.
pub fn warn_generic_once(
    op: &str,
    precision: &str,
    mode: KernelMode,
    arch: &str,
    quality: Quality,
) {
    if !enabled() {
        return;
    }
    let key = format!("{op}|{precision}|{}|{arch}", mode.as_str());
    let first = {
        let mut s = seen().lock().unwrap();
        s.insert(key)
    };
    if first {
        let note = match quality {
            Quality::Reference => "no optimized overlay",
            Quality::CompletenessOnly => "completeness-only, KNOWN-LOW-QUALITY",
        };
        eprintln!(
            "hipfire: generic fallback — {op} {precision} {} on {arch} ({note})",
            mode.as_str()
        );
    }
}

/// Number of distinct `(op, precision, mode, arch)` generic fallbacks seen so far.
/// For eval/CI assertions ("hot path took no generic fallback").
pub fn generic_fallback_count() -> usize {
    seen().lock().unwrap().len()
}

/// Clear the dedup set (test/eval harness use — lets a fresh assertion window start).
pub fn reset_generic_warnings() {
    seen().lock().unwrap().clear();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dedups_per_tuple_and_counts() {
        reset_generic_warnings();
        let base = generic_fallback_count();
        // Same tuple three times → counted once.
        for _ in 0..3 {
            warn_generic_once(
                "gate_up",
                "W4A8",
                KernelMode::Prefill,
                "gfx1103",
                Quality::Reference,
            );
        }
        // A different cell (mode) → a second entry.
        warn_generic_once(
            "gate_up",
            "W4A8",
            KernelMode::Decode,
            "gfx1103",
            Quality::Reference,
        );
        // A different precision → a third.
        warn_generic_once(
            "qkv",
            "W4A4",
            KernelMode::Decode,
            "gfx1103",
            Quality::CompletenessOnly,
        );
        assert_eq!(generic_fallback_count() - base, 3);
    }
}
