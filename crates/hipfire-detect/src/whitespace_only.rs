//! Whitespace-only output detector — hard-fails when visible text
//! is all whitespace (spaces, tabs, newlines, unicode whitespace).

use crate::{Detector, Event, Verdict};

pub struct WhitespaceOnly {
    buf: String,
    saw_done: bool,
}

impl WhitespaceOnly {
    pub fn new() -> Self {
        Self {
            buf: String::new(),
            saw_done: false,
        }
    }
}

impl Default for WhitespaceOnly {
    fn default() -> Self {
        Self::new()
    }
}

impl Detector for WhitespaceOnly {
    fn name(&self) -> &'static str {
        "whitespace_only"
    }

    fn observe(&mut self, ev: &Event<'_>) -> Option<Verdict> {
        match ev {
            Event::Token { text, synthetic, .. } => {
                if !synthetic {
                    self.buf.push_str(text);
                }
            }
            Event::Done { .. } => self.saw_done = true,
            _ => {}
        }
        None
    }

    fn finalize(&mut self) -> Verdict {
        if !self.saw_done {
            return Verdict::skip("no Done event seen");
        }
        if self.buf.is_empty() {
            // EosImmediate covers this case as a separate signal —
            // skip here to avoid double-counting.
            return Verdict::skip("0 visible bytes (covered by eos_immediate)");
        }
        if self.buf.chars().all(char::is_whitespace) {
            return Verdict::fail(format!(
                "{} visible bytes, all whitespace",
                self.buf.len()
            ));
        }
        Verdict::Ok
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn done() -> Event<'static> {
        Event::Done {
            total_tokens: 0,
            total_visible_bytes: 0,
            wall_ms: 100,
            ttft_ms: 50,
        }
    }
    fn tok(t: &'static str) -> Event<'static> {
        Event::Token {
            text: t,
            t_ms: 0,
            synthetic: false,
        }
    }

    #[test]
    fn whitespace_only_fails() {
        let mut d = WhitespaceOnly::new();
        d.observe(&tok("   \n\n  \t  "));
        d.observe(&done());
        assert!(d.finalize().is_fail());
    }

    #[test]
    fn real_content_passes() {
        let mut d = WhitespaceOnly::new();
        d.observe(&tok("ok\n"));
        d.observe(&done());
        assert!(matches!(d.finalize(), Verdict::Ok));
    }

    #[test]
    fn empty_skipped_to_avoid_double_count() {
        let mut d = WhitespaceOnly::new();
        d.observe(&done());
        assert!(matches!(d.finalize(), Verdict::Skip { .. }));
    }
}
