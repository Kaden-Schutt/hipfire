//! Path-A token attractor detectors.
//!
//! Two windows from CLAUDE.md's canonical Path-A specification:
//!   - First 128 committed tokens (before the first EOT)
//!   - Last 128 committed tokens (before the first EOT)
//!
//! Both consume `Event::Committed` only. `Event::Token` is ignored — the
//! attractor lives in the IDs, not in the visible chunks (which may
//! aggregate or drop tokens via `EosFilter`).
//!
//! Source: `scripts/coherence-gate-dflash.sh:191-243` + CLAUDE.md:236-244.
//! The Path-A failure shape is single-token loops (e.g. "numbers(numbers(...")
//! that produce `unique_ratio ≈ 0.05, max_freq ≈ 0.60` in the first 128.

use crate::{Detector, Event, Verdict};
use std::collections::HashMap;

/// Qwen3.5 EOT token IDs. Pre-EOT trim stops at the first occurrence of
/// either; post-EOT tokens (model spamming `#` after a clean function
/// close) are not the failure class we guard against.
pub const EOT_IDS: [u32; 2] = [248044, 248046];

/// Minimum window size below which attractor judgement is suppressed.
/// Clean early termination produces a tiny pre-EOT window — that's fine.
const MIN_WINDOW: usize = 16;

/// Path-A attractor — first 128 tokens.
///
/// Hard fail: `max_freq > 0.50` OR `unique_ratio < 0.15`.
/// Soft warn: `max_freq > 0.40` OR `unique_ratio < 0.30` (and not hard).
pub struct AttractorFirst128 {
    pre_eot: Vec<u32>,
    saw_eot: bool,
}

impl AttractorFirst128 {
    pub fn new() -> Self {
        Self {
            pre_eot: Vec::with_capacity(128),
            saw_eot: false,
        }
    }
}

impl Default for AttractorFirst128 {
    fn default() -> Self {
        Self::new()
    }
}

impl Detector for AttractorFirst128 {
    fn name(&self) -> &'static str {
        "attractor_first_128"
    }

    fn observe(&mut self, ev: &Event<'_>) -> Option<Verdict> {
        if let Event::Committed { tok_id, .. } = ev {
            if !self.saw_eot {
                if EOT_IDS.contains(tok_id) {
                    self.saw_eot = true;
                } else if self.pre_eot.len() < 128 {
                    self.pre_eot.push(*tok_id);
                }
            }
        }
        None
    }

    fn finalize(&mut self) -> Verdict {
        let window: &[u32] = &self.pre_eot;
        if window.len() < MIN_WINDOW {
            return Verdict::Ok;
        }
        verdict_for_window(window, /*hard_unique=*/ 0.15, /*soft_unique=*/ 0.30)
    }
}

/// Path-A attractor — last 128 tokens before first EOT.
///
/// Hard fail: `max_freq > 0.50` OR `unique_ratio < 0.30` (looser unique
/// because the tail of a long completion is naturally less diverse).
/// Soft warn: `max_freq > 0.40` (and not hard).
pub struct AttractorLast128 {
    /// Rolling buffer of pre-EOT tokens. We keep the WHOLE pre-EOT
    /// stream — `last_128` is computed from the tail at finalize. This
    /// is simpler than rolling 128-window invariants and the memory cost
    /// is bounded by `max_tokens`.
    pre_eot: Vec<u32>,
    saw_eot: bool,
}

impl AttractorLast128 {
    pub fn new() -> Self {
        Self {
            pre_eot: Vec::new(),
            saw_eot: false,
        }
    }
}

impl Default for AttractorLast128 {
    fn default() -> Self {
        Self::new()
    }
}

impl Detector for AttractorLast128 {
    fn name(&self) -> &'static str {
        "attractor_last_128"
    }

    fn observe(&mut self, ev: &Event<'_>) -> Option<Verdict> {
        if let Event::Committed { tok_id, .. } = ev {
            if !self.saw_eot {
                if EOT_IDS.contains(tok_id) {
                    self.saw_eot = true;
                } else {
                    self.pre_eot.push(*tok_id);
                }
            }
        }
        None
    }

    fn finalize(&mut self) -> Verdict {
        let n = self.pre_eot.len();
        if n < MIN_WINDOW {
            return Verdict::Ok;
        }
        let start = n.saturating_sub(128);
        let window = &self.pre_eot[start..];
        verdict_for_window(window, /*hard_unique=*/ 0.30, /*soft_unique=*/ 0.40)
    }
}

/// Compute max_freq + unique_ratio over `window`. Apply tiered thresholds:
///   hard: `max_freq > 0.50` OR `unique_ratio < hard_unique`
///   soft: `max_freq > 0.40` OR `unique_ratio < soft_unique`
fn verdict_for_window(window: &[u32], hard_unique: f64, soft_unique: f64) -> Verdict {
    let mut counts: HashMap<u32, usize> = HashMap::with_capacity(window.len());
    for t in window {
        *counts.entry(*t).or_insert(0) += 1;
    }
    let total = window.len() as f64;
    let unique_ratio = counts.len() as f64 / total;
    let (max_tok, max_count) = counts
        .iter()
        .max_by_key(|(_, c)| **c)
        .map(|(t, c)| (*t, *c))
        .expect("window non-empty by MIN_WINDOW guard");
    let max_freq = max_count as f64 / total;

    if max_freq > 0.50 || unique_ratio < hard_unique {
        return Verdict::fail(format!(
            "max_freq {:.2} (tok {}), unique_ratio {:.2} over {} tokens (hard: >0.50 OR <{:.2})",
            max_freq, max_tok, unique_ratio, window.len(), hard_unique
        ));
    }
    if max_freq > 0.40 || unique_ratio < soft_unique {
        return Verdict::warn(format!(
            "max_freq {:.2} (tok {}), unique_ratio {:.2} over {} tokens (soft: >0.40 OR <{:.2})",
            max_freq, max_tok, unique_ratio, window.len(), soft_unique
        ));
    }
    Verdict::Ok
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ev(tok: u32, pos: usize) -> Event<'static> {
        Event::Committed {
            tok_id: tok,
            pos,
            t_ms: pos as u64,
        }
    }

    fn run<D: Detector>(mut d: D, toks: &[u32]) -> Verdict {
        for (i, t) in toks.iter().enumerate() {
            d.observe(&ev(*t, i));
        }
        d.finalize()
    }

    #[test]
    fn first_128_clean_passes() {
        let toks: Vec<u32> = (0..200).map(|i| (i % 50) as u32).collect();
        let v = run(AttractorFirst128::new(), &toks);
        assert!(matches!(v, Verdict::Ok), "got {:?}", v);
    }

    #[test]
    fn first_128_path_a_attractor_fails() {
        // Single-token loop in first 128 — Path A shape.
        let mut toks: Vec<u32> = vec![42; 100];
        toks.extend((100..130).map(|i| i as u32));
        let v = run(AttractorFirst128::new(), &toks);
        assert!(v.is_fail(), "got {:?}", v);
    }

    #[test]
    fn first_128_short_window_skipped() {
        let toks: Vec<u32> = vec![5, 5, 5, 5, EOT_IDS[0]];
        let v = run(AttractorFirst128::new(), &toks);
        assert!(matches!(v, Verdict::Ok), "got {:?}", v);
    }

    #[test]
    fn first_128_eot_trims() {
        // 16 distinct tokens then EOT then 100 of the same — trim ignores tail.
        let mut toks: Vec<u32> = (0..16).map(|i| i as u32).collect();
        toks.push(EOT_IDS[1]);
        toks.extend(std::iter::repeat(7).take(100));
        let v = run(AttractorFirst128::new(), &toks);
        assert!(matches!(v, Verdict::Ok), "got {:?}", v);
    }

    #[test]
    fn last_128_attractor_at_tail_fails() {
        let mut toks: Vec<u32> = (0..200).map(|i| (i % 50) as u32).collect();
        toks.extend(std::iter::repeat(99).take(120));
        let v = run(AttractorLast128::new(), &toks);
        assert!(v.is_fail(), "got {:?}", v);
    }

    #[test]
    fn last_128_clean_diverse_tail_ok() {
        let toks: Vec<u32> = (0..400).map(|i| (i % 60) as u32).collect();
        let v = run(AttractorLast128::new(), &toks);
        assert!(matches!(v, Verdict::Ok), "got {:?}", v);
    }

    #[test]
    fn last_128_soft_warn_at_max_freq_above_0_4() {
        // Tail has one token at 0.42 freq (54/128), unique ratio above 0.30.
        // Should soft-warn, not hard-fail.
        let mut toks: Vec<u32> = Vec::new();
        for _ in 0..54 {
            toks.push(7);
        }
        for i in 0..74 {
            toks.push((i + 100) as u32);
        }
        // Now total len=128, all of which will be the "last 128".
        let v = run(AttractorLast128::new(), &toks);
        assert!(v.is_warn(), "got {:?}", v);
    }
}
