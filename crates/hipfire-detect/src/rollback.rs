// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Rollback-replay stat parsers.
//!
//! These read the single-line stat summaries the DFlash AR-parity run prints
//! (`rollback_parity:` and `verify_graph:`) and turn them into pass/fail
//! verdicts. Both gate the coherence battery: fast GDN-tape replay and direct
//! verify are diagnostic-only, so a non-zero count there is a hard fail, as is
//! the absence of any admitted replay / graph capture.
//!
//! Source: the `ROLLBACK_REPLAY_PY` + `VERIFY_GRAPH_PY` heredocs in
//! `tests/coherence-gate-dflash.sh`. JSON value shape is preserved (the gate
//! reads `.ok`; field key order is irrelevant — the trace record re-serializes
//! sorted).

use regex::Regex;
use serde::Serialize;

/// Verdict for the `rollback_parity:` replay-count line.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(untagged)]
pub enum ReplayReport {
    /// No `rollback_parity:` line was found.
    Missing { ok: bool, reason: &'static str },
    /// Counts parsed. `reason` is present only on the hard-fail branches.
    Counts {
        ok: bool,
        #[serde(skip_serializing_if = "Option::is_none")]
        reason: Option<&'static str>,
        replay_gdn_tape: u64,
        replay_batched_prefill: u64,
        replay_full_prefill: u64,
        replay_prefix_verify: u64,
        replay_serial_tape: u64,
        replay_verify_complete: u64,
    },
}

impl ReplayReport {
    pub fn ok(&self) -> bool {
        match self {
            ReplayReport::Missing { ok, .. } | ReplayReport::Counts { ok, .. } => *ok,
        }
    }
}

/// Parse the `rollback_parity:` line and apply the gate's two-tier policy.
pub fn replay_parity(text: &str) -> ReplayReport {
    // `(?m)` for `^` line anchoring; optional groups mirror the Python regex.
    let re = Regex::new(
        r"(?m)^rollback_parity: .*replay_gdn_tape=(\d+)(?: replay_batched_prefill=(\d+))? replay_full_prefill=(\d+)(?: replay_prefix_verify=(\d+))?(?: replay_serial_tape=(\d+))?(?: replay_verify_complete=(\d+))?",
    )
    .expect("static regex");
    let Some(c) = re.captures(text) else {
        return ReplayReport::Missing {
            ok: false,
            reason: "missing_rollback_parity_stats",
        };
    };
    let g = |i: usize| -> u64 { c.get(i).map_or(0, |m| m.as_str().parse().unwrap_or(0)) };
    let gdn = g(1);
    let batched = g(2);
    let full = g(3);
    let prefix_verify = g(4);
    let serial_tape = g(5);
    let verify_complete = g(6);

    let reason = if gdn != 0 {
        Some("fast_gdn_tape_replay_is_diagnostic_only")
    } else if batched + full + prefix_verify + serial_tape + verify_complete == 0 {
        Some("missing_admitted_rollback_replay")
    } else {
        None
    };
    ReplayReport::Counts {
        ok: reason.is_none(),
        reason,
        replay_gdn_tape: gdn,
        replay_batched_prefill: batched,
        replay_full_prefill: full,
        replay_prefix_verify: prefix_verify,
        replay_serial_tape: serial_tape,
        replay_verify_complete: verify_complete,
    }
}

/// Verdict for the `verify_graph:` count line.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(untagged)]
pub enum VerifyGraphReport {
    Missing {
        ok: bool,
        reason: &'static str,
    },
    Counts {
        ok: bool,
        #[serde(skip_serializing_if = "Option::is_none")]
        reason: Option<&'static str>,
        direct: u64,
        warmup: u64,
        capture: u64,
        replay: u64,
        not_applicable: u64,
    },
}

impl VerifyGraphReport {
    pub fn ok(&self) -> bool {
        match self {
            VerifyGraphReport::Missing { ok, .. } | VerifyGraphReport::Counts { ok, .. } => *ok,
        }
    }
}

/// Parse the `verify_graph:` line and apply the gate's two-tier policy.
pub fn verify_graph(text: &str) -> VerifyGraphReport {
    let re = Regex::new(
        r"(?m)^verify_graph: direct=(\d+) warmup=(\d+) capture=(\d+) replay=(\d+) not_applicable=(\d+)",
    )
    .expect("static regex");
    let Some(c) = re.captures(text) else {
        return VerifyGraphReport::Missing {
            ok: false,
            reason: "missing_verify_graph_stats",
        };
    };
    let g = |i: usize| -> u64 { c.get(i).map_or(0, |m| m.as_str().parse().unwrap_or(0)) };
    let direct = g(1);
    let warmup = g(2);
    let capture = g(3);
    let replay = g(4);
    let not_applicable = g(5);

    let reason = if direct != 0 {
        Some("direct_verify_is_diagnostic_only_for_dflash_ar_parity")
    } else if capture + replay == 0 {
        Some("missing_verify_graph_capture_or_replay")
    } else {
        None
    };
    VerifyGraphReport::Counts {
        ok: reason.is_none(),
        reason,
        direct,
        warmup,
        capture,
        replay,
        not_applicable,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn replay_missing_line() {
        assert_eq!(
            serde_json::to_value(replay_parity("nothing here")).unwrap(),
            json!({ "ok": false, "reason": "missing_rollback_parity_stats" })
        );
    }

    #[test]
    fn replay_gdn_nonzero_is_diagnostic_only() {
        let line = "rollback_parity: foo replay_gdn_tape=2 replay_full_prefill=3";
        let v = serde_json::to_value(replay_parity(line)).unwrap();
        assert_eq!(v["ok"], json!(false));
        assert_eq!(
            v["reason"],
            json!("fast_gdn_tape_replay_is_diagnostic_only")
        );
        assert_eq!(v["replay_gdn_tape"], json!(2));
        assert_eq!(v["replay_full_prefill"], json!(3));
        assert_eq!(v["replay_batched_prefill"], json!(0)); // optional, absent
    }

    #[test]
    fn replay_all_zero_is_missing_admitted() {
        let line = "rollback_parity: x replay_gdn_tape=0 replay_full_prefill=0";
        let v = serde_json::to_value(replay_parity(line)).unwrap();
        assert_eq!(v["ok"], json!(false));
        assert_eq!(v["reason"], json!("missing_admitted_rollback_replay"));
    }

    #[test]
    fn replay_admitted_ok_has_no_reason() {
        let line = "rollback_parity: x replay_gdn_tape=0 replay_batched_prefill=1 replay_full_prefill=4 replay_prefix_verify=2 replay_serial_tape=1 replay_verify_complete=1";
        let v = serde_json::to_value(replay_parity(line)).unwrap();
        assert_eq!(v["ok"], json!(true));
        assert!(v.get("reason").is_none());
        assert_eq!(v["replay_full_prefill"], json!(4));
        assert_eq!(v["replay_verify_complete"], json!(1));
    }

    #[test]
    fn verify_graph_missing_line() {
        assert_eq!(
            serde_json::to_value(verify_graph("nope")).unwrap(),
            json!({ "ok": false, "reason": "missing_verify_graph_stats" })
        );
    }

    #[test]
    fn verify_graph_direct_nonzero_fails() {
        let line = "verify_graph: direct=1 warmup=0 capture=2 replay=2 not_applicable=0";
        let v = serde_json::to_value(verify_graph(line)).unwrap();
        assert_eq!(v["ok"], json!(false));
        assert_eq!(
            v["reason"],
            json!("direct_verify_is_diagnostic_only_for_dflash_ar_parity")
        );
    }

    #[test]
    fn verify_graph_no_capture_or_replay_fails() {
        let line = "verify_graph: direct=0 warmup=3 capture=0 replay=0 not_applicable=1";
        let v = serde_json::to_value(verify_graph(line)).unwrap();
        assert_eq!(v["ok"], json!(false));
        assert_eq!(v["reason"], json!("missing_verify_graph_capture_or_replay"));
    }

    #[test]
    fn verify_graph_ok_has_no_reason() {
        let line = "verify_graph: direct=0 warmup=1 capture=2 replay=3 not_applicable=0";
        let v = serde_json::to_value(verify_graph(line)).unwrap();
        assert_eq!(v["ok"], json!(true));
        assert!(v.get("reason").is_none());
        assert_eq!(v["capture"], json!(2));
        assert_eq!(v["replay"], json!(3));
    }
}
