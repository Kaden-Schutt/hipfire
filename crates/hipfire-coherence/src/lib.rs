// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Shared coherence detector policy and report serialization helpers.

use hipfire_detect::{
    attractor::{AttractorFirst128, AttractorLast128, LongStateCollapse},
    eos_immediate::EosImmediate,
    ngram::{LoopGuardMirror, NgramDensity},
    report::Report,
    special_leak::SpecialLeak,
    think::{ThinkEmpty, ThinkStall},
    timing::StepTimeSpike,
    toolcall::ToolcallShape,
    whitespace_only::WhitespaceOnly,
    DetectorBank, Severity, Verdict,
};
use serde_json::{json, Value};

#[derive(Debug, Clone)]
pub struct DetectorProfile {
    pub agentic: bool,
    pub stall_tokens: Option<usize>,
    pub detect_timing: bool,
}

impl DetectorProfile {
    pub fn default_for_prompt(prompt: &str, system: Option<&str>) -> Self {
        Self {
            agentic: decide_agentic(prompt, system),
            stall_tokens: None,
            detect_timing: false,
        }
    }

    pub fn long_state() -> Self {
        Self {
            agentic: false,
            stall_tokens: None,
            detect_timing: false,
        }
    }
}

pub fn build_detector_bank(profile: &DetectorProfile) -> DetectorBank {
    let mut bank = DetectorBank::new();
    bank.add(Box::new(AttractorFirst128::new()));
    bank.add(Box::new(AttractorLast128::new()));
    bank.add(Box::new(LongStateCollapse::new()));
    bank.add(Box::new(NgramDensity::new()));
    bank.add(Box::new(LoopGuardMirror::new()));
    bank.add(Box::new(ThinkEmpty::new()));
    if let Some(budget) = profile.stall_tokens {
        bank.add(Box::new(ThinkStall::new(budget)));
    }
    bank.add(Box::new(SpecialLeak::new()));
    if profile.agentic {
        bank.add(Box::new(ToolcallShape::new()));
    }
    bank.add(Box::new(EosImmediate::new()));
    bank.add(Box::new(WhitespaceOnly::new()));
    if profile.detect_timing {
        bank.add(Box::new(StepTimeSpike::new()));
    }
    bank
}

pub fn decide_agentic(prompt: &str, system: Option<&str>) -> bool {
    let combined = format!("{}\n{}", system.unwrap_or(""), prompt);
    let s = combined.to_ascii_lowercase();
    s.contains("<tool_call>")
        || (s.contains("\"name\"") && s.contains("\"arguments\""))
        || (s.contains("function") && s.contains("\"arguments\""))
}

pub fn detector_rows(report: &Report) -> Vec<Value> {
    report
        .rows
        .iter()
        .map(|row| {
            json!({
                "detector": row.name,
                "status": match &row.verdict {
                    Verdict::Ok => "pass",
                    Verdict::Skip { .. } => "skip",
                    Verdict::Fired { severity: Severity::Warn, .. } => "warn",
                    Verdict::Fired { severity: Severity::Fail, .. } => "fail",
                },
                "detail": match &row.verdict {
                    Verdict::Ok => None,
                    Verdict::Skip { reason } => Some(reason.clone()),
                    Verdict::Fired { detail, .. } => Some(detail.clone()),
                },
            })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use hipfire_detect::report::ReportHeader;

    fn header() -> ReportHeader {
        ReportHeader {
            prompt_md5: "deadbeef".to_string(),
            prompt_label: "test".to_string(),
            model: "qwen3.5-9b-mq4.hfq".to_string(),
            arch: "gfx1100".to_string(),
            host: "host".to_string(),
            total_tokens: 16,
            tok_s: 10.0,
            gen_tok_s: 12.0,
            ttft_ms: 3,
            daemon_prefill_ms: 0.0,
            daemon_prefill_tok_s: 0.0,
            daemon_decode_tok_s: 0.0,
            daemon_ttft_ms: 0.0,
            daemon_tok_s: 0.0,
        }
    }

    #[test]
    fn agentic_detection_uses_prompt_or_system() {
        assert!(decide_agentic(
            "call <tool_call>{\"name\":\"read\",\"arguments\":{}}</tool_call>",
            None
        ));
        assert!(decide_agentic(
            "plain",
            Some("Use function calls with \"arguments\" objects")
        ));
        assert!(!decide_agentic("plain prompt", Some("plain system")));
    }

    #[test]
    fn detector_bank_respects_profile_toggles() {
        let plain = build_detector_bank(&DetectorProfile {
            agentic: false,
            stall_tokens: None,
            detect_timing: false,
        });
        let rich = build_detector_bank(&DetectorProfile {
            agentic: true,
            stall_tokens: Some(128),
            detect_timing: true,
        });
        assert!(rich.len() > plain.len());
    }

    #[test]
    fn detector_rows_match_runtime_artifact_shape() {
        let report = Report::new(
            header(),
            vec![
                ("clean", Verdict::Ok),
                ("optional", Verdict::skip("disabled")),
                ("soft", Verdict::warn("low confidence")),
                ("hard", Verdict::fail("loop detected")),
            ],
        );

        let rows = detector_rows(&report);
        assert_eq!(
            rows,
            vec![
                json!({"detector": "clean", "status": "pass", "detail": null}),
                json!({"detector": "optional", "status": "skip", "detail": "disabled"}),
                json!({"detector": "soft", "status": "warn", "detail": "low confidence"}),
                json!({"detector": "hard", "status": "fail", "detail": "loop detected"}),
            ]
        );
    }
}
