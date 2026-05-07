//! EOS-immediate detector — hard-fails when generation produced zero
//! visible bytes before stopping.
//!
//! Cousin of the coherence-gate's "zero tokens emitted" hard-error
//! (`CLAUDE.md:219-221`). The runtime-side gate fires on no token
//! events at all; this detector additionally fires when there were
//! token events but every one was empty/synthetic.

use crate::{Detector, Event, Verdict};

pub struct EosImmediate {
    visible_bytes: usize,
    saw_done: bool,
}

impl EosImmediate {
    pub fn new() -> Self {
        Self {
            visible_bytes: 0,
            saw_done: false,
        }
    }
}

impl Default for EosImmediate {
    fn default() -> Self {
        Self::new()
    }
}

impl Detector for EosImmediate {
    fn name(&self) -> &'static str {
        "eos_immediate"
    }

    fn observe(&mut self, ev: &Event<'_>) -> Option<Verdict> {
        match ev {
            Event::Token { text, synthetic, .. } => {
                if !synthetic {
                    self.visible_bytes += text.len();
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
        if self.visible_bytes == 0 {
            return Verdict::fail("0 visible bytes emitted before stop");
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
    fn synth(t: &'static str) -> Event<'static> {
        Event::Token {
            text: t,
            t_ms: 0,
            synthetic: true,
        }
    }

    #[test]
    fn no_visible_fails() {
        let mut d = EosImmediate::new();
        d.observe(&done());
        assert!(d.finalize().is_fail());
    }

    #[test]
    fn synthetic_only_still_fails() {
        let mut d = EosImmediate::new();
        d.observe(&synth("</think>\n"));
        d.observe(&done());
        assert!(d.finalize().is_fail());
    }

    #[test]
    fn one_visible_byte_passes() {
        let mut d = EosImmediate::new();
        d.observe(&tok("ok"));
        d.observe(&done());
        assert!(matches!(d.finalize(), Verdict::Ok));
    }

    #[test]
    fn no_done_event_skipped() {
        let mut d = EosImmediate::new();
        assert!(matches!(d.finalize(), Verdict::Skip { .. }));
    }
}
