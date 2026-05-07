//! Self-check — guards against detector rot.
//!
//! Two phases:
//!
//! - **Phase A (synthetic):** craft in-memory events targeting each
//!   detector. Assert each fires on its own payload and stays silent
//!   when fed a clean payload. Direct port of `agentic-gate.sh:72-144`.
//!
//! - **Phase B (JSONL replay):** parse a captured JSONL fixture (real
//!   daemon output) and dispatch through a `DetectorBank`. Asserts the
//!   bank's verdicts match the fixture's expected outcomes. Catches the
//!   failure mode synthetic-payload self-check misses: a detector regex
//!   that matches a synthetic string but no longer matches real daemon
//!   output (because of `EosFilter` reshaping, encoding quirks, …).
//!
//! Both phases run without GPU. Total wall-clock < 1s.

use crate::{
    attractor::{AttractorFirst128, AttractorLast128, EOT_IDS},
    eos_immediate::EosImmediate,
    ngram::{LoopGuardMirror, NgramDensity},
    special_leak::SpecialLeak,
    think::{ThinkEmpty, ThinkStall},
    timing::StepTimeSpike,
    toolcall::ToolcallShape,
    whitespace_only::WhitespaceOnly,
    Detector, DetectorBank, Event, Verdict,
};

/// One Phase-A check: build a fresh detector, push the payload events,
/// finalize, assert the verdict matches `want_fail`/`want_warn`.
struct PhaseA {
    name: &'static str,
    detector: Box<dyn Detector>,
    events: Vec<OwnedEvent>,
    want: Want,
}

#[derive(Debug, Clone, PartialEq, Eq)]
#[allow(dead_code)]
enum Want {
    Fail,
    Warn,
    Ok,
}

#[derive(Debug, Clone)]
enum OwnedEvent {
    Committed { tok_id: u32, pos: usize, t_ms: u64 },
    Token { text: String, t_ms: u64, synthetic: bool },
    Done {
        total_tokens: usize,
        total_visible_bytes: usize,
        wall_ms: u64,
        ttft_ms: u64,
    },
}

impl OwnedEvent {
    fn as_event(&self) -> Event<'_> {
        match self {
            OwnedEvent::Committed { tok_id, pos, t_ms } => Event::Committed {
                tok_id: *tok_id,
                pos: *pos,
                t_ms: *t_ms,
            },
            OwnedEvent::Token {
                text,
                t_ms,
                synthetic,
            } => Event::Token {
                text: text.as_str(),
                t_ms: *t_ms,
                synthetic: *synthetic,
            },
            OwnedEvent::Done {
                total_tokens,
                total_visible_bytes,
                wall_ms,
                ttft_ms,
            } => Event::Done {
                total_tokens: *total_tokens,
                total_visible_bytes: *total_visible_bytes,
                wall_ms: *wall_ms,
                ttft_ms: *ttft_ms,
            },
        }
    }
}

fn commits(toks: &[u32]) -> Vec<OwnedEvent> {
    toks.iter()
        .enumerate()
        .map(|(i, t)| OwnedEvent::Committed {
            tok_id: *t,
            pos: i,
            t_ms: i as u64 * 10,
        })
        .collect()
}

fn token(text: &str) -> OwnedEvent {
    OwnedEvent::Token {
        text: text.to_string(),
        t_ms: 0,
        synthetic: false,
    }
}

fn done() -> OwnedEvent {
    OwnedEvent::Done {
        total_tokens: 100,
        total_visible_bytes: 100,
        wall_ms: 1000,
        ttft_ms: 100,
    }
}

fn build_phase_a() -> Vec<PhaseA> {
    let mut out: Vec<PhaseA> = Vec::new();

    // 1. attractor_first_128 — single-token loop in first 128.
    let mut bad = vec![42u32; 100];
    bad.extend(100..130);
    out.push(PhaseA {
        name: "attractor_first_128",
        detector: Box::new(AttractorFirst128::new()),
        events: commits(&bad),
        want: Want::Fail,
    });

    // 2. attractor_last_128 — diverse start, attractor in tail.
    let mut bad: Vec<u32> = (0..200).map(|i| (i % 50) as u32).collect();
    bad.extend(std::iter::repeat(99).take(120));
    out.push(PhaseA {
        name: "attractor_last_128",
        detector: Box::new(AttractorLast128::new()),
        events: commits(&bad),
        want: Want::Fail,
    });

    // 3. ngram_density — back half is a single-token attractor
    // (every trigram is (X,X,X) → density 1.00, above the 0.50 soft
    // threshold). A 3-token cycle would only hit ~⅓ density.
    let mut bad: Vec<u32> = (0..50).map(|i| i as u32).collect();
    bad.extend(std::iter::repeat(7).take(70));
    out.push(PhaseA {
        name: "ngram_density",
        detector: Box::new(NgramDensity::new()),
        events: commits(&bad),
        want: Want::Warn,
    });

    // 4. loop_guard_mirror — 4-gram repeats 8× exactly.
    let mut bad: Vec<u32> = Vec::new();
    for _ in 0..8 {
        bad.extend_from_slice(&[1, 2, 3, 4]);
    }
    out.push(PhaseA {
        name: "loop_guard_mirror",
        detector: Box::new(LoopGuardMirror::new()),
        events: commits(&bad),
        want: Want::Warn,
    });

    // 5. think_empty — empty <think></think>. Soft warn (see think.rs
    // header for why this is not a hard fail).
    out.push(PhaseA {
        name: "think_empty",
        detector: Box::new(ThinkEmpty::new()),
        events: vec![token("hello <think></think> ok")],
        want: Want::Warn,
    });

    // 6. think_stall — <think> open with budget exceeded.
    let mut events: Vec<OwnedEvent> = vec![token("hello <think>")];
    for i in 1..=20 {
        events.push(OwnedEvent::Committed {
            tok_id: 1,
            pos: i,
            t_ms: i as u64 * 10,
        });
    }
    out.push(PhaseA {
        name: "think_stall",
        detector: Box::new(ThinkStall::new(10)),
        events,
        want: Want::Fail,
    });

    // 7. special_leak — <|im_start|> in visible text.
    out.push(PhaseA {
        name: "special_leak",
        detector: Box::new(SpecialLeak::new()),
        events: vec![token("hello <|im_start|> world"), done()],
        want: Want::Fail,
    });

    // 8. toolcall_shape — stacked openers + special leak combo.
    out.push(PhaseA {
        name: "toolcall_shape",
        detector: Box::new(ToolcallShape::new()),
        events: vec![token(
            "<tool_call>\n<tool_call>\n{\"arguments\": {\"path\": \"/tmp/x\"}}\n</tool_call>",
        )],
        want: Want::Fail,
    });

    // 9. eos_immediate — Done with 0 visible bytes.
    out.push(PhaseA {
        name: "eos_immediate",
        detector: Box::new(EosImmediate::new()),
        events: vec![done()],
        want: Want::Fail,
    });

    // 10. whitespace_only — visible content is whitespace only.
    out.push(PhaseA {
        name: "whitespace_only",
        detector: Box::new(WhitespaceOnly::new()),
        events: vec![token("   \n\t  "), done()],
        want: Want::Fail,
    });

    // 11. step_time_spike — one 100ms gap on 10ms baseline.
    let mut events: Vec<OwnedEvent> = (0..32u64)
        .map(|i| OwnedEvent::Committed {
            tok_id: 1,
            pos: i as usize,
            t_ms: i * 10,
        })
        .collect();
    events.push(OwnedEvent::Committed {
        tok_id: 1,
        pos: 32,
        t_ms: 32 * 10 + 100,
    });
    for i in 33..40u64 {
        events.push(OwnedEvent::Committed {
            tok_id: 1,
            pos: i as usize,
            t_ms: 32 * 10 + 100 + (i - 32) * 10,
        });
    }
    out.push(PhaseA {
        name: "step_time_spike",
        detector: Box::new(StepTimeSpike::new()),
        events,
        want: Want::Warn,
    });

    out
}

/// Result of running self-check.
#[derive(Debug, Clone)]
pub struct SelfCheckReport {
    /// Per-detector pass/fail. `true` = all expected behaviours observed.
    pub phase_a: Vec<(&'static str, bool, String)>,
    /// Per-fixture pass/fail. Empty when no fixtures were provided.
    pub phase_b: Vec<(String, bool, String)>,
}

impl SelfCheckReport {
    pub fn ok(&self) -> bool {
        self.phase_a.iter().all(|(_, ok, _)| *ok)
            && self.phase_b.iter().all(|(_, ok, _)| *ok)
    }
}

/// Run Phase A only.
pub fn run_phase_a() -> SelfCheckReport {
    let mut out = SelfCheckReport {
        phase_a: Vec::new(),
        phase_b: Vec::new(),
    };
    for mut check in build_phase_a() {
        for ev in &check.events {
            check.detector.observe(&ev.as_event());
        }
        let v = check.detector.finalize();
        let pass = match (&check.want, &v) {
            (Want::Fail, x) if x.is_fail() => true,
            (Want::Warn, x) if x.is_warn() => true,
            (Want::Ok, Verdict::Ok) => true,
            _ => false,
        };
        let detail = format!("want {:?}, got {}", check.want, v.label());
        out.phase_a.push((check.name, pass, detail));
    }
    out
}

// ─── Phase B: captured-JSONL replay ──────────────────────────────────────
//
// Fixtures are embedded at compile time so the probe binary doesn't need
// to know where the source tree lives at runtime. Each fixture is paired
// with a list of `(detector_name, expected_outcome)` assertions — the
// same assertions the integration test in `tests/replay.rs` makes.

const CLEAN_JSONL: &str = include_str!("../tests/fixtures/clean.jsonl");
const PATH_A_JSONL: &str = include_str!("../tests/fixtures/path_a_attractor.jsonl");
const AGENTIC_CORRUPT_JSONL: &str = include_str!("../tests/fixtures/agentic_corrupt.jsonl");

/// Expected outcome for one detector on one fixture.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Expect {
    Quiet, // Verdict::Ok or Verdict::Skip
    Warn,
    Fail,
}

struct FixtureCheck {
    label: &'static str,
    jsonl: &'static str,
    expectations: &'static [(&'static str, Expect)],
}

const FIXTURES: &[FixtureCheck] = &[
    FixtureCheck {
        label: "clean.jsonl",
        jsonl: CLEAN_JSONL,
        expectations: &[
            ("attractor_first_128", Expect::Quiet),
            ("attractor_last_128", Expect::Quiet),
            ("ngram_density", Expect::Quiet),
            ("loop_guard_mirror", Expect::Quiet),
            ("think_empty", Expect::Quiet),
            ("special_leak", Expect::Quiet),
            ("toolcall_shape", Expect::Quiet),
            ("eos_immediate", Expect::Quiet),
            ("whitespace_only", Expect::Quiet),
        ],
    },
    FixtureCheck {
        label: "path_a_attractor.jsonl",
        jsonl: PATH_A_JSONL,
        expectations: &[
            ("attractor_first_128", Expect::Fail),
            ("ngram_density", Expect::Warn),
            ("loop_guard_mirror", Expect::Warn),
            ("special_leak", Expect::Quiet),
            ("toolcall_shape", Expect::Quiet),
            ("think_empty", Expect::Quiet),
        ],
    },
    FixtureCheck {
        label: "agentic_corrupt.jsonl",
        jsonl: AGENTIC_CORRUPT_JSONL,
        expectations: &[
            ("toolcall_shape", Expect::Fail),
            ("special_leak", Expect::Fail),
            ("attractor_first_128", Expect::Quiet),
        ],
    },
];

fn build_full_replay_bank() -> DetectorBank {
    use crate::{
        attractor::{AttractorFirst128, AttractorLast128},
        eos_immediate::EosImmediate,
        ngram::{LoopGuardMirror, NgramDensity},
        special_leak::SpecialLeak,
        think::ThinkEmpty,
        toolcall::ToolcallShape,
        whitespace_only::WhitespaceOnly,
    };
    let mut bank = DetectorBank::new();
    bank.add(Box::new(AttractorFirst128::new()));
    bank.add(Box::new(AttractorLast128::new()));
    bank.add(Box::new(NgramDensity::new()));
    bank.add(Box::new(LoopGuardMirror::new()));
    bank.add(Box::new(ThinkEmpty::new()));
    bank.add(Box::new(SpecialLeak::new()));
    bank.add(Box::new(ToolcallShape::new()));
    bank.add(Box::new(EosImmediate::new()));
    bank.add(Box::new(WhitespaceOnly::new()));
    bank
}

fn evaluate_expectation(verdict: &Verdict, want: Expect) -> bool {
    match want {
        Expect::Quiet => !verdict.is_fail() && !verdict.is_warn(),
        Expect::Warn => verdict.is_warn(),
        Expect::Fail => verdict.is_fail(),
    }
}

/// Run Phase B — replay each shipped JSONL fixture through a fresh
/// `DetectorBank` and assert that every detector produces the expected
/// outcome (Quiet / Warn / Fail).
pub fn run_phase_b() -> Vec<(String, bool, String)> {
    let mut out: Vec<(String, bool, String)> = Vec::new();
    for fx in FIXTURES {
        let events = parse_jsonl_events(fx.jsonl);
        let mut bank = build_full_replay_bank();
        let finals = replay(&mut bank, &events);
        let mut misses: Vec<String> = Vec::new();
        for (det_name, want) in fx.expectations {
            let verdict = finals
                .iter()
                .find(|(n, _)| *n == *det_name)
                .map(|(_, v)| v);
            match verdict {
                None => misses.push(format!("{} not in bank", det_name)),
                Some(v) => {
                    if !evaluate_expectation(v, *want) {
                        misses.push(format!(
                            "{}: want {:?}, got {}",
                            det_name,
                            want,
                            v.label()
                        ));
                    }
                }
            }
        }
        let ok = misses.is_empty();
        let detail = if ok {
            format!("{} expectations met", fx.expectations.len())
        } else {
            misses.join("; ")
        };
        out.push((fx.label.to_string(), ok, detail));
    }
    out
}

/// Run both Phase A and Phase B. Used by `coherence_probe --self-check`.
pub fn run_full() -> SelfCheckReport {
    let mut report = run_phase_a();
    report.phase_b = run_phase_b();
    report
}

/// Parse a JSONL stream of daemon-style events. Each line is either
/// `{"type":"committed",...}`, `{"type":"token",...}`, or
/// `{"type":"done",...}`.
pub fn parse_jsonl_events(jsonl: &str) -> Vec<OwnedEventPub> {
    jsonl
        .lines()
        .filter(|l| !l.trim().is_empty())
        .filter_map(|l| serde_json::from_str::<serde_json::Value>(l).ok())
        .filter_map(|v| {
            let t = v.get("type")?.as_str()?;
            match t {
                "committed" => Some(OwnedEventPub::Committed {
                    tok_id: v.get("tok_id")?.as_u64()? as u32,
                    pos: v.get("pos")?.as_u64()? as usize,
                    t_ms: v.get("t_ms")?.as_u64()?,
                }),
                "token" => Some(OwnedEventPub::Token {
                    text: v.get("text")?.as_str()?.to_string(),
                    t_ms: v.get("t_ms").and_then(|x| x.as_u64()).unwrap_or(0),
                    synthetic: v
                        .get("synthetic")
                        .and_then(|x| x.as_bool())
                        .unwrap_or(false),
                }),
                "done" => Some(OwnedEventPub::Done {
                    total_tokens: v
                        .get("tokens")
                        .or_else(|| v.get("total_tokens"))
                        .and_then(|x| x.as_u64())
                        .unwrap_or(0) as usize,
                    total_visible_bytes: v
                        .get("total_visible_bytes")
                        .and_then(|x| x.as_u64())
                        .unwrap_or(0) as usize,
                    wall_ms: v
                        .get("wall_ms")
                        .and_then(|x| x.as_u64())
                        .unwrap_or(0),
                    ttft_ms: v
                        .get("ttft_ms")
                        .and_then(|x| x.as_u64())
                        .unwrap_or(0),
                }),
                _ => None,
            }
        })
        .collect()
}

/// Public mirror of `OwnedEvent` so external callers can build replay
/// streams without going through JSONL.
#[derive(Debug, Clone)]
pub enum OwnedEventPub {
    Committed { tok_id: u32, pos: usize, t_ms: u64 },
    Token {
        text: String,
        t_ms: u64,
        synthetic: bool,
    },
    Done {
        total_tokens: usize,
        total_visible_bytes: usize,
        wall_ms: u64,
        ttft_ms: u64,
    },
}

impl OwnedEventPub {
    pub fn as_event(&self) -> Event<'_> {
        match self {
            OwnedEventPub::Committed { tok_id, pos, t_ms } => Event::Committed {
                tok_id: *tok_id,
                pos: *pos,
                t_ms: *t_ms,
            },
            OwnedEventPub::Token {
                text,
                t_ms,
                synthetic,
            } => Event::Token {
                text: text.as_str(),
                t_ms: *t_ms,
                synthetic: *synthetic,
            },
            OwnedEventPub::Done {
                total_tokens,
                total_visible_bytes,
                wall_ms,
                ttft_ms,
            } => Event::Done {
                total_tokens: *total_tokens,
                total_visible_bytes: *total_visible_bytes,
                wall_ms: *wall_ms,
                ttft_ms: *ttft_ms,
            },
        }
    }
}

/// Run a captured event stream through a bank. Used by Phase B and by
/// the probe binary's `--from-jsonl` mode (post-hoc analysis of a
/// captured run).
pub fn replay(bank: &mut DetectorBank, events: &[OwnedEventPub]) -> Vec<(&'static str, Verdict)> {
    for ev in events {
        bank.observe(&ev.as_event());
    }
    bank.finalize()
}

/// Use `EOT_IDS` as a sanity check that the export wired up cleanly.
#[doc(hidden)]
pub fn _eot_ids_in_scope() -> [u32; 2] {
    EOT_IDS
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn phase_a_all_pass() {
        let r = run_phase_a();
        for (name, ok, detail) in &r.phase_a {
            assert!(*ok, "Phase A miss for {}: {}", name, detail);
        }
        assert!(r.ok());
    }

    #[test]
    fn parse_jsonl_round_trip() {
        let jsonl = r#"{"type":"committed","id":"r1","tok_id":42,"pos":0,"t_ms":10}
{"type":"token","id":"r1","text":"hello","t_ms":11}
{"type":"done","id":"r1","tokens":1,"wall_ms":20,"ttft_ms":10}"#;
        let evs = parse_jsonl_events(jsonl);
        assert_eq!(evs.len(), 3);
    }

    #[test]
    fn phase_b_all_fixtures_pass() {
        let r = run_phase_b();
        for (label, ok, detail) in &r {
            assert!(*ok, "Phase B miss for {}: {}", label, detail);
        }
        assert_eq!(r.len(), 3, "expected three shipped fixtures");
    }

    #[test]
    fn run_full_runs_both_phases() {
        let r = run_full();
        assert!(!r.phase_a.is_empty(), "phase A should run");
        assert!(!r.phase_b.is_empty(), "phase B should run");
        assert!(r.ok());
    }
}
