// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Focused route seam tests for dense lifecycle.
//! Covers abort during prefill/decode, handshake abort, and rollback failure ordering.

#![allow(clippy::all)]

use hipfire_engine::terminal::{
    activate_terminal_control, apply_terminal_control, check_abort, claim_terminal,
    clear_terminal_control, BatchAttemptScope,
};
use hipfire_generate::common::{
    emit_fail_closed_error, emit_spec_cancel_after_rollback, RollbackEpilogue,
};
use hipfire_generate::dense::glimmer_commit_terminal;
use std::sync::{Mutex, MutexGuard};

fn test_lock() -> MutexGuard<'static, ()> {
    static LOCK: Mutex<()> = Mutex::new(());
    LOCK.lock().unwrap_or_else(|e| e.into_inner())
}

fn parse_lines(buf: &[u8]) -> Vec<serde_json::Value> {
    buf.split(|&b| b == b'\n')
        .filter(|l| !l.is_empty())
        .map(|l| serde_json::from_slice(l).unwrap())
        .collect()
}

#[test]
fn abort_during_prefill_is_exact_attempt_and_clears_via_rollback() {
    let _lock = test_lock();
    clear_terminal_control();
    let id = "test-prefill-abort";
    let attempt = 42u64;
    let _scope = BatchAttemptScope::enter(attempt);
    activate_terminal_control(id, attempt);
    assert!(!check_abort(id));
    apply_terminal_control("abort", id, attempt + 1);
    assert!(!check_abort(id));
    apply_terminal_control("abort", id, attempt);
    assert!(check_abort(id));
    let ep = RollbackEpilogue {
        rolled_back: true,
        context: None,
    };
    let mut buf = Vec::new();
    emit_spec_cancel_after_rollback(&mut buf, id, 0, &ep);
    let lines = parse_lines(&buf);
    assert_eq!(lines.len(), 2);
    assert_eq!(lines[0]["type"], "aborted");
    assert_eq!(lines[1]["type"], "done");
    assert_eq!(lines[1]["finish_reason"], "aborted");
    assert_eq!(lines[0]["attempt_id"], attempt);
    assert_eq!(lines[1]["attempt_id"], attempt);
    assert!(!claim_terminal(id, attempt));
    clear_terminal_control();
}

#[test]
fn abort_during_decode_emits_cancel_after_rollback() {
    let _lock = test_lock();
    clear_terminal_control();
    let id = "test-decode-abort";
    let attempt = 99u64;
    let _scope = BatchAttemptScope::enter(attempt);
    activate_terminal_control(id, attempt);
    apply_terminal_control("abort", id, attempt);
    assert!(check_abort(id));
    let ep = RollbackEpilogue {
        rolled_back: true,
        context: None,
    };
    let mut buf = Vec::new();
    emit_spec_cancel_after_rollback(&mut buf, id, 5, &ep);
    let lines = parse_lines(&buf);
    assert_eq!(lines.len(), 2);
    assert_eq!(lines[1]["completion_tokens"], 5);
    clear_terminal_control();
}

#[test]
fn handshake_abort_returns_false_and_caller_resets_before_cancel() {
    let _lock = test_lock();
    clear_terminal_control();
    let id = "test-handshake-abort";
    let attempt = 123u64;
    let _scope = BatchAttemptScope::enter(attempt);
    activate_terminal_control(id, attempt);
    apply_terminal_control("abort", id, attempt);
    let pending = serde_json::json!({
        "type": "done",
        "id": id,
        "tokens": 7,
        "attempt_id": attempt,
        "finish_reason": "stop"
    });
    let mut buf = Vec::new();
    let committed = glimmer_commit_terminal(&mut buf, id, &pending, 7);
    assert!(!committed);
    let lines = parse_lines(&buf);
    assert!(lines.iter().any(|v| v["type"] == "commit_ready"));
    assert!(!lines.iter().any(|v| v["type"] == "aborted"));
    let ep = RollbackEpilogue {
        rolled_back: true,
        context: None,
    };
    emit_spec_cancel_after_rollback(&mut buf, id, 7, &ep);
    let lines2 = parse_lines(&buf);
    assert!(lines2.iter().any(|v| v["type"] == "aborted"));
    assert!(lines2
        .iter()
        .any(|v| v["type"] == "done" && v["finish_reason"] == "aborted"));
    clear_terminal_control();
}

#[test]
fn rollback_failure_ordering_emits_fail_closed_not_done() {
    let _lock = test_lock();
    clear_terminal_control();
    let id = "test-rollback-fail";
    let attempt = 77u64;
    let _scope = BatchAttemptScope::enter(attempt);
    activate_terminal_control(id, attempt);
    apply_terminal_control("abort", id, attempt);
    let ep = RollbackEpilogue {
        rolled_back: false,
        context: Some("gpu sync failed: oom".into()),
    };
    let mut buf = Vec::new();
    emit_spec_cancel_after_rollback(&mut buf, id, 3, &ep);
    let lines = parse_lines(&buf);
    assert_eq!(lines.len(), 1);
    assert_eq!(lines[0]["type"], "error");
    assert_eq!(lines[0]["rolled_back"], false);
    assert_eq!(lines[0]["attempt_id"], attempt);
    assert!(!lines.iter().any(|v| v["type"] == "done"));
    clear_terminal_control();
}

#[test]
fn glimmer_error_rollback_failure_surfaces_fail_closed() {
    let _lock = test_lock();
    clear_terminal_control();
    let id = "test-glimmer-error-fail";
    let attempt = 88u64;
    let _scope = BatchAttemptScope::enter(attempt);
    activate_terminal_control(id, attempt);
    let ep = RollbackEpilogue {
        rolled_back: false,
        context: Some("reset failed".into()),
    };
    let mut buf = Vec::new();
    emit_fail_closed_error(
        &mut buf,
        Some(id),
        "muse_glimmer prefill failed: oom",
        "gpu",
        true,
        &ep,
    );
    let lines = parse_lines(&buf);
    assert_eq!(lines.len(), 1);
    assert_eq!(lines[0]["type"], "error");
    assert_eq!(lines[0]["rolled_back"], false);
    assert!(lines[0]["message"]
        .as_str()
        .unwrap()
        .contains("muse_glimmer prefill failed"));
    clear_terminal_control();
}

#[test]
fn stale_attempt_cannot_claim_terminal_or_abort() {
    let _lock = test_lock();
    clear_terminal_control();
    let id = "test-stale";
    let attempt = 10u64;
    let stale = 11u64;
    let _scope = BatchAttemptScope::enter(attempt);
    activate_terminal_control(id, attempt);
    apply_terminal_control("abort", id, stale);
    assert!(!check_abort(id));
    clear_terminal_control();
}
