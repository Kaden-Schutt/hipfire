//! `<think>` block detectors.
//!
//! Both run on the visible-text stream (`Event::Token`) and require the
//! probe to have asked the daemon to leave think bytes intact (i.e.
//! `strip_think: false` on the generate request — the probe's
//! `--no-strip-think` flag). With think stripped, neither detector can
//! see the open/close tags.
//!
//! - `think_empty` (SOFT warn): a `<think></think>` pair with zero
//!   non-whitespace content between. Per Codex review of the plan,
//!   this is intentionally a soft warning rather than a hard fail —
//!   Qwen3.5 emits `<think>\n\n</think>\n` followed by a clean answer
//!   for trivial prompts when thinking is enabled-by-default but the
//!   model self-skips reasoning. That's normal behaviour, not a bug.
//!   The genuinely-bad case (think collapse with no answer after) is
//!   already covered by `eos_immediate` + `whitespace_only`.
//! - `think_stall` (HARD fail, OPT-IN): an open `<think>` with no
//!   matching `</think>` after the user-supplied stall budget. Off by
//!   default to avoid penalising long legitimate reasoning.

use crate::{Detector, Event, Verdict};

const OPEN: &str = "<think>";
const CLOSE: &str = "</think>";

/// Hard-fails when an empty `<think></think>` pair appears.
pub struct ThinkEmpty {
    buf: String,
    fired: bool,
}

impl ThinkEmpty {
    pub fn new() -> Self {
        Self {
            buf: String::new(),
            fired: false,
        }
    }
}

impl Default for ThinkEmpty {
    fn default() -> Self {
        Self::new()
    }
}

impl Detector for ThinkEmpty {
    fn name(&self) -> &'static str {
        "think_empty"
    }

    fn observe(&mut self, ev: &Event<'_>) -> Option<Verdict> {
        if let Event::Token { text, .. } = ev {
            self.buf.push_str(text);
            // Keep only the relevant tail. We never need more than the
            // length of the longest possible empty-think sequence —
            // bound the buffer at a few KB so streaming generations
            // don't grow it unbounded.
            const MAX_BUF: usize = 16 * 1024;
            if self.buf.len() > MAX_BUF {
                let drop = self.buf.len() - MAX_BUF;
                self.buf.drain(..drop);
            }
        }
        None
    }

    fn finalize(&mut self) -> Verdict {
        // Scan for any `<think>` ... `</think>` pair whose interior
        // (after trimming whitespace) is empty.
        let mut search_from = 0;
        while let Some(open_at) = self.buf[search_from..].find(OPEN) {
            let open_abs = search_from + open_at;
            let after_open = open_abs + OPEN.len();
            if let Some(close_rel) = self.buf[after_open..].find(CLOSE) {
                let interior = &self.buf[after_open..after_open + close_rel];
                if interior.trim().is_empty() {
                    self.fired = true;
                    return Verdict::warn(format!(
                        "empty <think></think> pair at offset {} (often benign — Qwen3.5 self-skip)",
                        open_abs
                    ));
                }
                search_from = after_open + close_rel + CLOSE.len();
            } else {
                break;
            }
        }
        Verdict::Ok
    }
}

/// Hard-fails when an open `<think>` is not closed after the
/// configured stall budget. OPT-IN — constructed only when the probe
/// receives `--stall-tokens N`.
pub struct ThinkStall {
    /// Budget in committed tokens after the open tag.
    budget: usize,
    /// Visible-text buffer (used to find open/close).
    buf: String,
    /// `Some(commit_pos_at_open)` while a think block is open.
    open_at_commit: Option<usize>,
    /// Last seen commit position.
    last_commit: usize,
    triggered: Option<usize>,
}

impl ThinkStall {
    pub fn new(budget: usize) -> Self {
        Self {
            budget,
            buf: String::new(),
            open_at_commit: None,
            last_commit: 0,
            triggered: None,
        }
    }

    fn rescan(&mut self) {
        // Walk the buffer's open/close tags looking for the most recent
        // unmatched open. Use `find()` on `&str` so we never index into
        // the middle of a multi-byte UTF-8 codepoint (some prompts mix
        // Chinese/Japanese reasoning around the ASCII tags).
        let mut depth: i32 = 0;
        let mut last_open: Option<usize> = None;
        let mut search_from = 0;
        loop {
            let next_open = self.buf[search_from..].find(OPEN).map(|i| search_from + i);
            let next_close = self.buf[search_from..].find(CLOSE).map(|i| search_from + i);
            match (next_open, next_close) {
                (Some(o), Some(c)) if o < c => {
                    depth += 1;
                    last_open = Some(o);
                    search_from = o + OPEN.len();
                }
                (Some(_), Some(c)) => {
                    depth -= 1;
                    if depth <= 0 {
                        last_open = None;
                        depth = 0;
                    }
                    search_from = c + CLOSE.len();
                }
                (Some(o), None) => {
                    depth += 1;
                    last_open = Some(o);
                    search_from = o + OPEN.len();
                }
                (None, Some(c)) => {
                    depth -= 1;
                    if depth <= 0 {
                        last_open = None;
                        depth = 0;
                    }
                    search_from = c + CLOSE.len();
                }
                (None, None) => break,
            }
        }
        self.open_at_commit = last_open.map(|_| self.last_commit);
    }
}

impl Detector for ThinkStall {
    fn name(&self) -> &'static str {
        "think_stall"
    }

    fn observe(&mut self, ev: &Event<'_>) -> Option<Verdict> {
        match ev {
            Event::Committed { pos, .. } => {
                self.last_commit = *pos;
                if let Some(opened_at) = self.open_at_commit {
                    if pos.saturating_sub(opened_at) >= self.budget {
                        self.triggered = Some(*pos - opened_at);
                    }
                }
            }
            Event::Token { text, .. } => {
                self.buf.push_str(text);
                self.rescan();
            }
            Event::Done { .. } => {}
        }
        None
    }

    fn finalize(&mut self) -> Verdict {
        if let Some(elapsed) = self.triggered {
            return Verdict::fail(format!(
                "<think> open for {} tokens with no </think> (budget {})",
                elapsed, self.budget
            ));
        }
        if self.open_at_commit.is_some() {
            // Open at end of stream but never reached the budget.
            return Verdict::warn(format!(
                "<think> still open at stream end (under budget {})",
                self.budget
            ));
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
    fn commit(pos: usize) -> Event<'static> {
        Event::Committed {
            tok_id: 1,
            pos,
            t_ms: pos as u64,
        }
    }

    #[test]
    fn think_empty_warns_on_zero_content() {
        let mut d = ThinkEmpty::new();
        d.observe(&tok("hello <think></think> world"));
        assert!(d.finalize().is_warn());
    }

    #[test]
    fn think_empty_silent_on_real_content() {
        let mut d = ThinkEmpty::new();
        d.observe(&tok("hello <think>real reasoning here</think> ok"));
        assert!(matches!(d.finalize(), Verdict::Ok));
    }

    #[test]
    fn think_empty_warns_on_whitespace_interior() {
        // Whitespace-only interior is treated as empty per the trim() rule.
        let mut d = ThinkEmpty::new();
        d.observe(&tok("<think>   \n\n  </think>"));
        assert!(d.finalize().is_warn());
    }

    #[test]
    fn think_stall_fires_when_budget_exceeded() {
        let mut d = ThinkStall::new(10);
        d.observe(&tok("hello <think>"));
        for i in 1..=20 {
            d.observe(&commit(i));
        }
        assert!(d.finalize().is_fail());
    }

    #[test]
    fn think_stall_silent_when_closed_in_time() {
        let mut d = ThinkStall::new(10);
        d.observe(&tok("hello <think>"));
        for i in 1..=5 {
            d.observe(&commit(i));
        }
        d.observe(&tok("done</think> answer."));
        for i in 6..=20 {
            d.observe(&commit(i));
        }
        assert!(matches!(d.finalize(), Verdict::Ok));
    }
}
