//! Phase B JSONL replay — runs detector banks against captured daemon
//! output streams. Catches the failure mode synthetic-payload self-check
//! misses: a detector regex matches a synthetic string but no longer
//! matches real daemon JSONL after `EosFilter` reshaping.
//!
//! Three fixtures shipped:
//!   - `clean.jsonl` — every detector must stay silent (Ok or skip).
//!   - `path_a_attractor.jsonl` — token-id attractor; first-128
//!     attractor and ngram-density must fire.
//!   - `agentic_corrupt.jsonl` — stacked `<tool_call>` openers + special
//!     token leak; toolcall_shape and special_leak must fire.

use hipfire_detect::{
    attractor::{AttractorFirst128, AttractorLast128},
    eos_immediate::EosImmediate,
    ngram::{LoopGuardMirror, NgramDensity},
    self_check::{parse_jsonl_events, replay},
    special_leak::SpecialLeak,
    think::ThinkEmpty,
    toolcall::ToolcallShape,
    whitespace_only::WhitespaceOnly,
    DetectorBank, Verdict,
};

fn build_full_bank() -> DetectorBank {
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

fn run_fixture(name: &str) -> Vec<(&'static str, Verdict)> {
    let path = format!(
        "{}/tests/fixtures/{}",
        env!("CARGO_MANIFEST_DIR"),
        name
    );
    let raw = std::fs::read_to_string(&path).expect("read fixture");
    let events = parse_jsonl_events(&raw);
    assert!(!events.is_empty(), "fixture {} parsed to zero events", name);
    let mut bank = build_full_bank();
    replay(&mut bank, &events)
}

fn verdict_for<'a>(verdicts: &'a [(&'static str, Verdict)], name: &str) -> &'a Verdict {
    verdicts
        .iter()
        .find(|(n, _)| *n == name)
        .map(|(_, v)| v)
        .expect("verdict in bank")
}

#[test]
fn clean_fixture_all_quiet() {
    let v = run_fixture("clean.jsonl");
    for (name, verdict) in &v {
        assert!(
            !verdict.is_fail() && !verdict.is_warn(),
            "{} fired on clean fixture: {:?}",
            name,
            verdict
        );
    }
}

#[test]
fn path_a_attractor_fires() {
    let v = run_fixture("path_a_attractor.jsonl");

    // Mandatory firings.
    assert!(
        verdict_for(&v, "attractor_first_128").is_fail(),
        "attractor_first_128 should hard-fail"
    );
    assert!(
        verdict_for(&v, "ngram_density").is_warn(),
        "ngram_density should soft-warn"
    );
    assert!(
        verdict_for(&v, "loop_guard_mirror").is_warn(),
        "loop_guard_mirror should soft-warn (4-gram repeats >=8 times)"
    );

    // Detectors that should stay quiet on this fixture.
    for name in &["special_leak", "toolcall_shape", "think_empty"] {
        let vd = verdict_for(&v, name);
        assert!(
            !vd.is_fail() && !vd.is_warn(),
            "{} should be silent on path_a fixture, got {:?}",
            name,
            vd
        );
    }
}

#[test]
fn agentic_corrupt_fires() {
    let v = run_fixture("agentic_corrupt.jsonl");

    // Mandatory firings.
    assert!(
        verdict_for(&v, "toolcall_shape").is_fail(),
        "toolcall_shape should hard-fail (stacked openers + JSON parse)"
    );
    assert!(
        verdict_for(&v, "special_leak").is_fail(),
        "special_leak should hard-fail (<|im_start|> in body)"
    );

    // Token-id attractor detectors must NOT fire on this fixture
    // (only 5 tokens, none repeating heavily).
    let aw = verdict_for(&v, "attractor_first_128");
    assert!(!aw.is_fail() && !aw.is_warn(), "attractor_first_128 quiet, got {:?}", aw);
}
