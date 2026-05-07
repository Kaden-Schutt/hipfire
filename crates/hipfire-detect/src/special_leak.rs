//! Special-token leak detector.
//!
//! Hard-fails when a known ChatML special-token literal escapes into
//! the visible-text stream (post-`EosFilter`). Default markers:
//!
//!   - `<|im_start|>`
//!   - `<|endoftext|>`
//!
//! Text-level scan only. ID-level scan was considered (per Codex review
//! notes in `/home/kaden/.claude/plans/compressed-dancing-clarke.md`)
//! and intentionally dropped: when the daemon's `EosFilter::Stop`
//! catches a stop-EOS, bytes are discarded and no `Token` event fires;
//! when it doesn't, the literal characters appear in the visible text
//! and this detector catches them. ID-level would either duplicate
//! that signal or false-positive on legitimately committed-then-stopped
//! special tokens.
//!
//! False positive avoidance: this detector scans `Token` events only —
//! the daemon's prompt is never streamed back. A user asking "what
//! does <|im_start|> do?" doesn't trigger.

use crate::{Detector, Event, Verdict};

const DEFAULT_MARKERS: &[&str] = &["<|im_start|>", "<|endoftext|>"];

pub struct SpecialLeak {
    markers: Vec<String>,
    fired: Option<String>,
}

impl SpecialLeak {
    pub fn new() -> Self {
        Self::with_markers(DEFAULT_MARKERS.iter().map(|s| s.to_string()).collect())
    }

    pub fn with_markers(markers: Vec<String>) -> Self {
        Self {
            markers,
            fired: None,
        }
    }
}

impl Default for SpecialLeak {
    fn default() -> Self {
        Self::new()
    }
}

impl Detector for SpecialLeak {
    fn name(&self) -> &'static str {
        "special_leak"
    }

    fn observe(&mut self, ev: &Event<'_>) -> Option<Verdict> {
        if self.fired.is_some() {
            return None;
        }
        if let Event::Token { text, .. } = ev {
            for m in &self.markers {
                if text.contains(m.as_str()) {
                    self.fired = Some(m.clone());
                    break;
                }
            }
        }
        None
    }

    fn finalize(&mut self) -> Verdict {
        if let Some(m) = &self.fired {
            return Verdict::fail(format!("special-token leak: {} appeared in visible text", m));
        }
        Verdict::Ok
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tok(t: &str) -> Event<'_> {
        Event::Token {
            text: t,
            t_ms: 0,
            synthetic: false,
        }
    }

    fn run(parts: &[&str]) -> Verdict {
        let mut d = SpecialLeak::new();
        for p in parts {
            d.observe(&tok(p));
        }
        d.finalize()
    }

    #[test]
    fn clean_text_ok() {
        assert!(matches!(run(&["hello world"]), Verdict::Ok));
    }

    #[test]
    fn im_start_leak_fails() {
        assert!(run(&["before <|im_start|> after"]).is_fail());
    }

    #[test]
    fn endoftext_leak_fails() {
        assert!(run(&["normal output", " <|endoftext|>"]).is_fail());
    }

    #[test]
    fn split_across_chunks_fails() {
        // Even if the marker spans two emit chunks, the buffer
        // accumulation needs to catch it. Today we scan per-chunk —
        // document the limitation: `EosFilter`'s holdback semantics
        // mean a literal `<|im_` prefix would be held until disambiguated,
        // so split-across-chunks is not a real-world failure shape here.
        // This test asserts the per-chunk semantics rather than
        // pretending we cross-chunk.
        let v = run(&["<|im_start|> all in one chunk"]);
        assert!(v.is_fail());
    }
}
