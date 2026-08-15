// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Nick Woolmer
// hipfire — see LICENSE and NOTICE in the project root.

//! qwen35 grammar env resolver — restores the two operator tunables that
//! became no-ops when `grammar.rs` was unified into `saddle-core` (B1).
//!
//! Before the merge `crates/hipfire-arch-qwen35/src/grammar.rs:128-153`
//! read the variables via `hipfire_config::developer_var`:
//!   - `HIPFIRE_QWEN35_NGRAM_MIN_REPEATS` clamped to `2..=32` (default 6)
//!   - `HIPFIRE_QWEN35_NGRAM_LEN_MIN` clamped to `1..=32` (default 3)
//!
//! The unified `saddle_core::grammar::json::Config` parameterized these but
//! no caller wired the env — `Matcher::new` always used `Config::default()`
//! (256, 6, 3, 32). This module restores the env handling on the caller side
//! so `saddle-core` stays free of `hipfire-config` (the layering contract).
//!
//! Validation reproduces the pre-merge logic exactly:
//!   `developer_var(name).ok().and_then(|s| s.parse().ok()).filter(|n| in_range).unwrap_or(default)`
//! Out-of-range and unparseable values fall back to the default, they are not
//! clamped.

use crate::grammar;

/// Resolve the qwen35 grammar `Config` from the two `HIPFIRE_QWEN35_*` env vars.
///
/// Reads `HIPFIRE_QWEN35_NGRAM_MIN_REPEATS` (`2..=32`, default 6) and
/// `HIPFIRE_QWEN35_NGRAM_LEN_MIN` (`1..=32`, default 3). When unset,
/// unparseable, or out of range the field falls back to `Config::default()`.
///
/// The resolver first consults `hipfire_config::developer_var` (the
/// process snapshot the pre-merge code used) and then the live
/// `std::env::var` so that unit tests that `set_var` after the snapshot is
/// frozen still observe their mutation. For an operator the two sources agree
/// (the snapshot is built from the process env at startup), so precedence is
/// irrelevant; the live check merely makes the test harness deterministic
/// without requiring callers to reinstall the global `ProcessConfig`.
pub fn resolve_qwen35_grammar_config() -> grammar::Config {
    let defaults = grammar::Config::default();
    let ngram_min_repeats = resolve_one(
        "HIPFIRE_QWEN35_NGRAM_MIN_REPEATS",
        defaults.ngram_min_repeats,
        2,
        32,
    );
    let ngram_len_min = resolve_one(
        "HIPFIRE_QWEN35_NGRAM_LEN_MIN",
        defaults.ngram_len_min,
        1,
        32,
    );
    grammar::Config {
        ngram_window: defaults.ngram_window,
        ngram_min_repeats,
        ngram_len_min,
        ngram_len_max: defaults.ngram_len_max,
    }
}

/// Alias kept for ergonomic import from the daemon example.
pub fn resolve_grammar_config() -> grammar::Config {
    resolve_qwen35_grammar_config()
}

fn resolve_one(name: &str, default: usize, lo: usize, hi: usize) -> usize {
    // Pre-merge: hipfire_config::developer_var(name).ok().and_then(|s| s.parse().ok()).filter(|n| n >= lo && n <= hi).unwrap_or(default)
    // At runtime the snapshot (ProcessConfig) is built from the live env at
    // startup, so live and snapshot agree. For tests we read the live env
    // directly so a `set_var` after the snapshot is frozen is still observed,
    // and an invalid live value falls back to default without consulting a
    // snapshot that might have been polluted by a prior test's temporary
    // `set_var` before the snapshot was first initialized. When live is
    // absent we fall back to the snapshot so a TOML-set developer value is
    // still honoured.
    if let Ok(raw) = std::env::var(name) {
        if let Ok(n) = raw.parse::<usize>() {
            if n >= lo && n <= hi {
                return n;
            }
        }
        return default;
    }
    if let Ok(raw) = hipfire_config::developer_var(name) {
        if let Ok(n) = raw.parse::<usize>() {
            if n >= lo && n <= hi {
                return n;
            }
        }
    }
    default
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Mutex, OnceLock};

    fn lock() -> std::sync::MutexGuard<'static, ()> {
        static GLOBAL: OnceLock<Mutex<()>> = OnceLock::new();
        static INIT_SNAPSHOT: OnceLock<()> = OnceLock::new();
        // Ensure the ProcessConfig snapshot is initialized with a clean env
        // before any test's `set_var` can pollute it. Without this, the first
        // test that sets HIPFIRE_QWEN35_NGRAM_LEN_MIN=7 before the snapshot
        // is first read would cause the snapshot to capture 7, and a later
        // `defaults_when_unset` run (live absent) would incorrectly see 7
        // via the snapshot and fail.
        INIT_SNAPSHOT.get_or_init(|| {
            std::env::remove_var("HIPFIRE_QWEN35_NGRAM_MIN_REPEATS");
            std::env::remove_var("HIPFIRE_QWEN35_NGRAM_LEN_MIN");
            let _ = hipfire_config::developer_var("HIPFIRE_QWEN35_NGRAM_MIN_REPEATS");
            let _ = hipfire_config::developer_var("HIPFIRE_QWEN35_NGRAM_LEN_MIN");
        });
        GLOBAL.get_or_init(|| Mutex::new(())).lock().unwrap()
    }

    #[test]
    fn defaults_when_unset() {
        let _g = lock();
        std::env::remove_var("HIPFIRE_QWEN35_NGRAM_MIN_REPEATS");
        std::env::remove_var("HIPFIRE_QWEN35_NGRAM_LEN_MIN");
        let cfg = resolve_qwen35_grammar_config();
        let d = grammar::Config::default();
        assert_eq!(cfg.ngram_min_repeats, d.ngram_min_repeats);
        assert_eq!(cfg.ngram_len_min, d.ngram_len_min);
        assert_eq!(cfg.ngram_window, d.ngram_window);
        assert_eq!(cfg.ngram_len_max, d.ngram_len_max);
    }

    #[test]
    fn reads_min_repeats_from_env() {
        let _g = lock();
        let orig = std::env::var("HIPFIRE_QWEN35_NGRAM_MIN_REPEATS").ok();
        std::env::set_var("HIPFIRE_QWEN35_NGRAM_MIN_REPEATS", "10");
        let cfg = resolve_qwen35_grammar_config();
        assert_eq!(cfg.ngram_min_repeats, 10, "expected resolver to pick up HIPFIRE_QWEN35_NGRAM_MIN_REPEATS=10, got {:?}", cfg);
        // restore
        match orig {
            Some(v) => std::env::set_var("HIPFIRE_QWEN35_NGRAM_MIN_REPEATS", v),
            None => std::env::remove_var("HIPFIRE_QWEN35_NGRAM_MIN_REPEATS"),
        }
    }

    #[test]
    fn reads_len_min_from_env() {
        let _g = lock();
        let orig = std::env::var("HIPFIRE_QWEN35_NGRAM_LEN_MIN").ok();
        std::env::set_var("HIPFIRE_QWEN35_NGRAM_LEN_MIN", "7");
        let cfg = resolve_qwen35_grammar_config();
        assert_eq!(cfg.ngram_len_min, 7, "expected resolver to pick up HIPFIRE_QWEN35_NGRAM_LEN_MIN=7, got {:?}", cfg);
        match orig {
            Some(v) => std::env::set_var("HIPFIRE_QWEN35_NGRAM_LEN_MIN", v),
            None => std::env::remove_var("HIPFIRE_QWEN35_NGRAM_LEN_MIN"),
        }
    }

    #[test]
    fn out_of_range_falls_back_to_default() {
        let _g = lock();
        let orig_repeats = std::env::var("HIPFIRE_QWEN35_NGRAM_MIN_REPEATS").ok();
        let orig_len = std::env::var("HIPFIRE_QWEN35_NGRAM_LEN_MIN").ok();
        let d = grammar::Config::default();

        // Below lower bound
        std::env::set_var("HIPFIRE_QWEN35_NGRAM_MIN_REPEATS", "1");
        assert_eq!(
            resolve_qwen35_grammar_config().ngram_min_repeats,
            d.ngram_min_repeats,
            "1 is below 2..=32 for NGRAM_MIN_REPEATS, pre-merge would fall back to default 6"
        );
        // Above upper bound
        std::env::set_var("HIPFIRE_QWEN35_NGRAM_MIN_REPEATS", "33");
        assert_eq!(
            resolve_qwen35_grammar_config().ngram_min_repeats,
            d.ngram_min_repeats,
            "33 is above 2..=32, should fall back"
        );
        // Len below 1
        std::env::set_var("HIPFIRE_QWEN35_NGRAM_LEN_MIN", "0");
        std::env::set_var("HIPFIRE_QWEN35_NGRAM_MIN_REPEATS", "6"); // restore repeats to valid so not interfering
        assert_eq!(
            resolve_qwen35_grammar_config().ngram_len_min,
            d.ngram_len_min,
            "0 is below 1..=32 for NGRAM_LEN_MIN, should fall back to 3"
        );
        // Len above 32
        std::env::set_var("HIPFIRE_QWEN35_NGRAM_LEN_MIN", "99");
        assert_eq!(
            resolve_qwen35_grammar_config().ngram_len_min,
            d.ngram_len_min,
            "99 >32 should fall back"
        );
        // Unparseable
        std::env::set_var("HIPFIRE_QWEN35_NGRAM_LEN_MIN", "abc");
        assert_eq!(
            resolve_qwen35_grammar_config().ngram_len_min,
            d.ngram_len_min,
            "unparseable should fall back"
        );
        std::env::set_var("HIPFIRE_QWEN35_NGRAM_MIN_REPEATS", "notanumber");
        assert_eq!(
            resolve_qwen35_grammar_config().ngram_min_repeats,
            d.ngram_min_repeats,
            "unparseable min_repeats should fall back"
        );

        match orig_repeats {
            Some(v) => std::env::set_var("HIPFIRE_QWEN35_NGRAM_MIN_REPEATS", v),
            None => std::env::remove_var("HIPFIRE_QWEN35_NGRAM_MIN_REPEATS"),
        }
        match orig_len {
            Some(v) => std::env::set_var("HIPFIRE_QWEN35_NGRAM_LEN_MIN", v),
            None => std::env::remove_var("HIPFIRE_QWEN35_NGRAM_LEN_MIN"),
        }
    }
}
