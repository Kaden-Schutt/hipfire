// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! AR vs DFlash token-parity comparison.
//!
//! Unlike the single-stream `Detector`s, parity compares TWO committed token
//! streams for exact equality: the autoregressive baseline (`--ar-baseline`)
//! and the speculative DFlash run. Any divergence is a hard fail while proving
//! rollback parity.
//!
//! Source: the `PARITY_PY` heredoc in `tests/coherence-gate-dflash.sh`. The
//! JSON shape is preserved so the gate's `.ok` parse and reported window are
//! unchanged.

use serde::Serialize;

/// Result of comparing an AR token stream against a DFlash token stream.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(untagged)]
pub enum ParityReport {
    /// Streams are byte-identical.
    Match { ok: bool, tokens: usize },
    /// Streams diverge. Carries the first mismatch index and a ±4 context
    /// window around it from each stream.
    Mismatch {
        ok: bool,
        reason: &'static str,
        first_mismatch: usize,
        ar_len: usize,
        dflash_len: usize,
        ar_token: Option<u32>,
        dflash_token: Option<u32>,
        window_start: usize,
        ar_window: Vec<u32>,
        dflash_window: Vec<u32>,
    },
}

impl ParityReport {
    pub fn ok(&self) -> bool {
        match self {
            ParityReport::Match { ok, .. } | ParityReport::Mismatch { ok, .. } => *ok,
        }
    }
}

/// Compare an AR baseline token stream against a DFlash stream.
///
/// On divergence, `first_mismatch` is the first differing index, or — when one
/// stream is a strict prefix of the other — the length of the shorter stream.
/// The reported windows span `[first-4, first+5)`, clamped to each stream.
pub fn compare(ar: &[u32], df: &[u32]) -> ParityReport {
    if ar == df {
        return ParityReport::Match {
            ok: true,
            tokens: df.len(),
        };
    }
    let first = ar
        .iter()
        .zip(df.iter())
        .position(|(a, d)| a != d)
        .unwrap_or_else(|| ar.len().min(df.len()));
    let lo = first.saturating_sub(4);
    let hi = ar.len().max(df.len()).min(first + 5);
    ParityReport::Mismatch {
        ok: false,
        reason: "token_mismatch",
        first_mismatch: first,
        ar_len: ar.len(),
        dflash_len: df.len(),
        ar_token: ar.get(first).copied(),
        dflash_token: df.get(first).copied(),
        window_start: lo,
        ar_window: window(ar, lo, hi),
        dflash_window: window(df, lo, hi),
    }
}

/// Slice `[lo, hi)` clamped to `v`, never panicking.
fn window(v: &[u32], lo: usize, hi: usize) -> Vec<u32> {
    let end = hi.min(v.len());
    let start = lo.min(end);
    v[start..end].to_vec()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identical_streams_match() {
        let v: Vec<u32> = (0..50).collect();
        assert_eq!(compare(&v, &v), ParityReport::Match { ok: true, tokens: 50 });
    }

    #[test]
    fn empty_streams_match() {
        assert_eq!(compare(&[], &[]), ParityReport::Match { ok: true, tokens: 0 });
    }

    #[test]
    fn mismatch_reports_first_index_and_window() {
        let ar: Vec<u32> = (0..20).collect();
        let mut df = ar.clone();
        df[10] = 999;
        let r = compare(&ar, &df);
        match r {
            ParityReport::Mismatch {
                ok,
                first_mismatch,
                ar_token,
                dflash_token,
                window_start,
                ar_window,
                dflash_window,
                ..
            } => {
                assert!(!ok);
                assert_eq!(first_mismatch, 10);
                assert_eq!(ar_token, Some(10));
                assert_eq!(dflash_token, Some(999));
                assert_eq!(window_start, 6); // 10 - 4
                assert_eq!(ar_window, vec![6, 7, 8, 9, 10, 11, 12, 13, 14]); // [6,15)
                assert_eq!(dflash_window, vec![6, 7, 8, 9, 999, 11, 12, 13, 14]);
            }
            _ => panic!("expected mismatch"),
        }
    }

    #[test]
    fn prefix_mismatch_uses_shorter_len() {
        // df is a strict prefix of ar -> first mismatch at len(df).
        let ar: Vec<u32> = (0..20).collect();
        let df: Vec<u32> = (0..12).collect();
        let r = compare(&ar, &df);
        match r {
            ParityReport::Mismatch {
                first_mismatch,
                ar_token,
                dflash_token,
                ..
            } => {
                assert_eq!(first_mismatch, 12);
                assert_eq!(ar_token, Some(12));
                assert_eq!(dflash_token, None); // df exhausted
            }
            _ => panic!("expected mismatch"),
        }
    }

    #[test]
    fn match_serializes_to_minimal_shape() {
        let v = serde_json::to_value(compare(&[1, 2, 3], &[1, 2, 3])).unwrap();
        assert_eq!(v, serde_json::json!({ "ok": true, "tokens": 3 }));
    }
}
