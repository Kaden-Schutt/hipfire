//! Observational coherence/behavior detectors.
//!
//! Consumes the daemon's JSONL output and surfaces "model weirdness" — token
//! attractors, special-token leaks, empty/stalled `<think>` blocks, n-gram
//! density spikes, tool-call malformations, EOS-immediate, whitespace-only
//! output. Strictly observational: detectors never block, mutate, or
//! interfere with generation. They produce verdicts.
//!
//! Runtime-protective siblings live in `crates/hipfire-runtime/src/`:
//!   - `loop_guard.rs`   — 4-gram block (token-id layer)
//!   - `sampler.rs`      — repeat penalty, unclosed-attractor mask (logits)
//!   - `eos_filter.rs`   — `<think>` strip, stop-marker holdback (text)
//!
//! This crate mirrors what those guards silently catch and adds detectors
//! they don't have. No HIP / ROCm dependency — runs on any dev box.

use serde::{Deserialize, Serialize};

pub mod attractor;
pub mod eos_immediate;
pub mod ngram;
pub mod report;
pub mod self_check;
pub mod special_leak;
pub mod think;
pub mod timing;
pub mod toolcall;
pub mod whitespace_only;

/// Re-exported `loop_guard` constants. Keep the observational
/// `loop_guard_mirror` detector in lock-step with the runtime guard's
/// thresholds without depending on `hipfire-runtime`.
pub mod loop_guard_constants {
    pub const DEFAULT_NGRAM_THRESHOLD: usize = 8;
    pub const DEFAULT_NGRAM_WINDOW: usize = 256;
    pub const NGRAM_K: usize = 4;
}

/// One event from the daemon's parsed JSONL stream.
///
/// `Committed` fires once per token decoded by the model, BEFORE the
/// runtime's `EosFilter` sees the bytes. `Token` fires once per visible
/// chunk emitted to stdout, AFTER `EosFilter` has decided to release
/// bytes (which may aggregate multiple committed tokens, drop bytes
/// for stripped `<think>`, hold back stop-marker prefixes, or fire
/// synthetic emits like the force-close `</think>\n` at
/// `daemon.rs:2223` with no associated commit).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Event<'a> {
    /// Token was decoded by the model. Always fires when the daemon's
    /// `emit_token_ids` flag is set.
    Committed {
        tok_id: u32,
        pos: usize,
        t_ms: u64,
    },
    /// Visible bytes emitted to stdout. Fires once per
    /// `EosFilter::Emit`. Synthetic emits (no committed token) carry
    /// `synthetic = true`; detectors that correlate to commits should
    /// skip those.
    Token {
        text: &'a str,
        t_ms: u64,
        synthetic: bool,
    },
    /// Generation finished. Final stats follow.
    Done {
        total_tokens: usize,
        total_visible_bytes: usize,
        wall_ms: u64,
        ttft_ms: u64,
    },
}

/// Severity of a detector's finding.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Severity {
    /// Hard failure; verdict for the run is FAIL.
    Fail,
    /// Soft warning; reported but does not flip the run verdict.
    Warn,
}

/// A detector's verdict.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "status", rename_all = "lowercase")]
pub enum Verdict {
    /// Detector did not fire on this run.
    Ok,
    /// Detector was not run (e.g. opt-in flag absent).
    Skip { reason: String },
    /// Detector fired.
    Fired { severity: Severity, detail: String },
}

impl Verdict {
    pub fn fail<S: Into<String>>(detail: S) -> Self {
        Verdict::Fired {
            severity: Severity::Fail,
            detail: detail.into(),
        }
    }
    pub fn warn<S: Into<String>>(detail: S) -> Self {
        Verdict::Fired {
            severity: Severity::Warn,
            detail: detail.into(),
        }
    }
    pub fn skip<S: Into<String>>(reason: S) -> Self {
        Verdict::Skip {
            reason: reason.into(),
        }
    }
    pub fn is_fail(&self) -> bool {
        matches!(
            self,
            Verdict::Fired {
                severity: Severity::Fail,
                ..
            }
        )
    }
    pub fn is_warn(&self) -> bool {
        matches!(
            self,
            Verdict::Fired {
                severity: Severity::Warn,
                ..
            }
        )
    }
    pub fn label(&self) -> &'static str {
        match self {
            Verdict::Ok => "OK",
            Verdict::Skip { .. } => "SKIP",
            Verdict::Fired {
                severity: Severity::Fail,
                ..
            } => "FAIL",
            Verdict::Fired {
                severity: Severity::Warn,
                ..
            } => "WARN",
        }
    }
}

/// One detector. Consumes `Event`s, produces a final `Verdict` once the
/// stream finishes (`Event::Done`).
pub trait Detector: Send {
    /// Stable name used in reports and JSON output.
    fn name(&self) -> &'static str;

    /// Consume one event. Detectors may also return a transient verdict
    /// for live-stream UIs; the canonical verdict is the one returned
    /// by `finalize()`.
    fn observe(&mut self, ev: &Event<'_>) -> Option<Verdict> {
        let _ = ev;
        None
    }

    /// Final verdict once the stream completes. Called exactly once
    /// after the last event.
    fn finalize(&mut self) -> Verdict;
}

/// Holds a slice of detectors and dispatches events to all of them.
pub struct DetectorBank {
    detectors: Vec<Box<dyn Detector>>,
}

impl DetectorBank {
    pub fn new() -> Self {
        Self {
            detectors: Vec::new(),
        }
    }

    pub fn add(&mut self, det: Box<dyn Detector>) {
        self.detectors.push(det);
    }

    pub fn len(&self) -> usize {
        self.detectors.len()
    }

    pub fn is_empty(&self) -> bool {
        self.detectors.is_empty()
    }

    /// Dispatch one event to every detector. Returns transient verdicts
    /// from detectors that produced one (for live-stream printing).
    pub fn observe<'a>(&mut self, ev: &Event<'a>) -> Vec<(&'static str, Verdict)> {
        let mut out = Vec::new();
        for det in self.detectors.iter_mut() {
            if let Some(v) = det.observe(ev) {
                out.push((det.name(), v));
            }
        }
        out
    }

    /// Finalize every detector. Returns one `(name, verdict)` per detector
    /// in the order they were added.
    pub fn finalize(&mut self) -> Vec<(&'static str, Verdict)> {
        self.detectors
            .iter_mut()
            .map(|d| (d.name(), d.finalize()))
            .collect()
    }
}

impl Default for DetectorBank {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Probe(bool);
    impl Detector for Probe {
        fn name(&self) -> &'static str {
            "probe"
        }
        fn finalize(&mut self) -> Verdict {
            if self.0 {
                Verdict::fail("test")
            } else {
                Verdict::Ok
            }
        }
    }

    #[test]
    fn bank_collects_finals() {
        let mut bank = DetectorBank::new();
        bank.add(Box::new(Probe(false)));
        bank.add(Box::new(Probe(true)));
        let finals = bank.finalize();
        assert_eq!(finals.len(), 2);
        assert!(matches!(finals[0].1, Verdict::Ok));
        assert!(matches!(
            finals[1].1,
            Verdict::Fired {
                severity: Severity::Fail,
                ..
            }
        ));
    }

    #[test]
    fn verdict_helpers() {
        assert!(Verdict::fail("x").is_fail());
        assert!(Verdict::warn("x").is_warn());
        assert!(matches!(Verdict::Ok, Verdict::Ok));
    }
}
