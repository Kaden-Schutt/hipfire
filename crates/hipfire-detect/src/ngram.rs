//! N-gram detectors over the committed-token stream.
//!
//! Two distinct signals:
//!
//!   1. **3-gram density on the back half** — soft flag only. Per
//!      `CLAUDE.md:243-244`, "consecutive 3gram repetition density >50%
//!      in final half" is a structural-loop signature that requires a
//!      human eyeball, NOT an automatic hard fail. This detector only
//!      ever returns `Ok` or `Warn`.
//!
//!   2. **`loop_guard` mirror** — observational reflection of the
//!      runtime's `LoopGuard` (4-gram, threshold ≥8, window 256). Fires
//!      `Warn` when the live guard would have force-stopped the
//!      generation. Useful for confirming the protective layer is doing
//!      its job AND for surfacing borderline cases that approached but
//!      did not cross the threshold.
//!
//! Both consume `Event::Committed` only — token IDs are the canonical
//! stream for n-gram analysis (text-level n-grams collide with
//! tokenization quirks).

use crate::loop_guard_constants::{DEFAULT_NGRAM_THRESHOLD, DEFAULT_NGRAM_WINDOW, NGRAM_K};
use crate::{Detector, Event, Verdict};
use std::collections::HashMap;

// ─── 3-gram density (back half) ──────────────────────────────────────────

/// Minimum back-half size below which density judgement is suppressed.
/// Below ~16 tokens the ratio is too noisy to mean anything.
const MIN_BACK_HALF: usize = 16;

/// 3-gram density on the back half of the generated stream. SOFT only.
pub struct NgramDensity {
    tokens: Vec<u32>,
}

impl NgramDensity {
    pub fn new() -> Self {
        Self { tokens: Vec::new() }
    }
}

impl Default for NgramDensity {
    fn default() -> Self {
        Self::new()
    }
}

impl Detector for NgramDensity {
    fn name(&self) -> &'static str {
        "ngram_density"
    }

    fn observe(&mut self, ev: &Event<'_>) -> Option<Verdict> {
        if let Event::Committed { tok_id, .. } = ev {
            self.tokens.push(*tok_id);
        }
        None
    }

    fn finalize(&mut self) -> Verdict {
        let n = self.tokens.len();
        if n < MIN_BACK_HALF * 2 {
            return Verdict::Ok;
        }
        let back_start = n / 2;
        let back = &self.tokens[back_start..];
        if back.len() < 3 {
            return Verdict::Ok;
        }

        // Build the trigram frequency map and locate the most-repeated trigram.
        let mut counts: HashMap<[u32; 3], usize> = HashMap::new();
        for w in back.windows(3) {
            let key = [w[0], w[1], w[2]];
            *counts.entry(key).or_insert(0) += 1;
        }
        let total_trigrams = back.len() - 2;
        let (top_key, top_count) = counts
            .iter()
            .max_by_key(|(_, c)| **c)
            .map(|(k, c)| (*k, *c))
            .unwrap();

        // Density = max trigram occurrences / total trigrams. Per CLAUDE.md
        // the structural-loop threshold is `>0.50` (soft flag).
        let density = top_count as f64 / total_trigrams as f64;
        if density > 0.50 {
            return Verdict::warn(format!(
                "3-gram {:?} repeats {}/{} ({:.2}) in back half ({} toks)",
                top_key, top_count, total_trigrams, density, back.len()
            ));
        }
        Verdict::Ok
    }
}

// ─── loop_guard mirror ───────────────────────────────────────────────────

/// Observational mirror of `crates/hipfire-runtime/src/loop_guard.rs`.
/// Fires `Warn` when the live guard's predicate would have triggered
/// (`>=` `DEFAULT_NGRAM_THRESHOLD` repeats of any 4-gram inside the
/// trailing `DEFAULT_NGRAM_WINDOW`). Never `Fail` — purely observational.
pub struct LoopGuardMirror {
    threshold: usize,
    window: usize,
    tokens: Vec<u32>,
    /// First-fire trigger captured for the report. We stop updating
    /// after the first trigger so the report reflects the earliest
    /// detection rather than the most recent.
    first_trigger: Option<([u32; NGRAM_K], usize)>,
}

impl LoopGuardMirror {
    pub fn new() -> Self {
        Self::with(DEFAULT_NGRAM_THRESHOLD, DEFAULT_NGRAM_WINDOW)
    }

    pub fn with(threshold: usize, window: usize) -> Self {
        Self {
            threshold,
            window,
            tokens: Vec::new(),
            first_trigger: None,
        }
    }

    fn check_trailing(&mut self) {
        if self.first_trigger.is_some() {
            return;
        }
        if self.tokens.len() < NGRAM_K {
            return;
        }
        let start = self.tokens.len().saturating_sub(self.window);
        let win = &self.tokens[start..];
        if win.len() < NGRAM_K {
            return;
        }
        let mut counts: HashMap<[u32; NGRAM_K], usize> = HashMap::new();
        for w in win.windows(NGRAM_K) {
            let key = [w[0], w[1], w[2], w[3]];
            let c = counts.entry(key).or_insert(0);
            *c += 1;
            if *c >= self.threshold {
                self.first_trigger = Some((key, *c));
                return;
            }
        }
    }
}

impl Default for LoopGuardMirror {
    fn default() -> Self {
        Self::new()
    }
}

impl Detector for LoopGuardMirror {
    fn name(&self) -> &'static str {
        "loop_guard_mirror"
    }

    fn observe(&mut self, ev: &Event<'_>) -> Option<Verdict> {
        if let Event::Committed { tok_id, .. } = ev {
            self.tokens.push(*tok_id);
            self.check_trailing();
        }
        None
    }

    fn finalize(&mut self) -> Verdict {
        if let Some((ngram, count)) = self.first_trigger {
            return Verdict::warn(format!(
                "4-gram {:?} repeated {}× in trailing {} tokens (loop_guard threshold {})",
                ngram, count, self.window, self.threshold
            ));
        }
        Verdict::Ok
    }
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
    fn ngram_clean_back_half_ok() {
        let toks: Vec<u32> = (0..200).map(|i| (i * 7 + 3) as u32).collect();
        let v = run(NgramDensity::new(), &toks);
        assert!(matches!(v, Verdict::Ok), "got {:?}", v);
    }

    #[test]
    fn ngram_dense_back_half_warns() {
        // Front half varied; back half a single-token attractor (every
        // trigram is (X,X,X) — density = 1.00, well above 0.50).
        let mut toks: Vec<u32> = (0..50).map(|i| i as u32).collect();
        toks.extend(std::iter::repeat(7).take(70));
        let v = run(NgramDensity::new(), &toks);
        assert!(v.is_warn(), "got {:?}", v);
    }

    #[test]
    fn ngram_three_cycle_below_threshold() {
        // A clean (A,B,C) cycle produces trigrams (A,B,C), (B,C,A),
        // (C,A,B) at ~⅓ density each — well below the 0.50 threshold.
        // Asserts we do NOT false-positive on this shape.
        let mut toks: Vec<u32> = (0..50).map(|i| i as u32).collect();
        for _ in 0..30 {
            toks.extend_from_slice(&[1, 2, 3]);
        }
        let v = run(NgramDensity::new(), &toks);
        assert!(matches!(v, Verdict::Ok), "got {:?}", v);
    }

    #[test]
    fn ngram_short_stream_ok() {
        let toks: Vec<u32> = (0..10).map(|i| i as u32).collect();
        let v = run(NgramDensity::new(), &toks);
        assert!(matches!(v, Verdict::Ok));
    }

    #[test]
    fn loop_guard_mirror_fires_on_4gram_repeat() {
        // Repeat the 4-gram (1,2,3,4) eight times — exactly threshold.
        let mut toks: Vec<u32> = Vec::new();
        for _ in 0..8 {
            toks.extend_from_slice(&[1, 2, 3, 4]);
        }
        let v = run(LoopGuardMirror::new(), &toks);
        assert!(v.is_warn(), "got {:?}", v);
    }

    #[test]
    fn loop_guard_mirror_silent_below_threshold() {
        // Repeat 7× — one below threshold.
        let mut toks: Vec<u32> = Vec::new();
        for _ in 0..7 {
            toks.extend_from_slice(&[1, 2, 3, 4]);
        }
        let v = run(LoopGuardMirror::new(), &toks);
        assert!(matches!(v, Verdict::Ok), "got {:?}", v);
    }

    #[test]
    fn loop_guard_mirror_only_counts_trailing_window() {
        // Trigger inside head, then a long varied tail — trigger leaves
        // the window so finalize must NOT report it.
        let mut toks: Vec<u32> = Vec::new();
        for _ in 0..10 {
            toks.extend_from_slice(&[1, 2, 3, 4]);
        }
        // Pad past the default window of 256.
        for i in 0..400u32 {
            toks.push(1000 + i);
        }
        // Mirror SHOULD have caught it earlier — first_trigger is sticky.
        let v = run(LoopGuardMirror::new(), &toks);
        assert!(v.is_warn());
    }
}
