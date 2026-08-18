// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Extracted from `crates/hipfire-runtime/examples/daemon.rs`
//! `#[cfg(test)] mod terminal_control_tests` (9 of 10 tests).
//! The `glimmer_marker_suffix_is_char_boundary_safe` test remains in
//! `daemon.rs` because it exercises `super::glimmer_longest_marker_suffix`,
//! a Glimmer-specific helper that is not part of `hipfire-engine`.

    use hipfire_engine::terminal::{
        activate_terminal_control, apply_terminal_control, await_client_terminal_commit,
        check_abort, clear_terminal_control, mark_terminal_control_ready, set_active_attempt_id,
        terminal_control, wait_terminal_control_decision, ClientTerminalDecision,
        TerminalControlDecision,
    };
    use std::sync::{Mutex, MutexGuard, OnceLock};
    use std::time::Duration;

    /// Serializes all tests in this module: they share the process-global
    /// terminal-control singleton and would race under `cargo test` parallelism.
    fn test_lock() -> MutexGuard<'static, ()> {
        static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
        LOCK.get_or_init(|| Mutex::new(())).lock().unwrap()
    }

    /// Acquire the module lock and reset shared state. Hold the returned
    /// guard for the full test body (including any helper threads joined
    /// before drop).
    fn begin_test() -> MutexGuard<'static, ()> {
        let guard = test_lock();
        clear_terminal_control();
        set_active_attempt_id(0);
        guard
    }

    fn reset() {
        clear_terminal_control();
        set_active_attempt_id(0);
    }

    fn decision_of(id: &str, attempt_id: u64) -> Option<TerminalControlDecision> {
        let g = terminal_control().mu.lock().unwrap();
        g.active.as_ref().and_then(|a| {
            if a.id == id && a.attempt_id == attempt_id {
                a.decision
            } else {
                None
            }
        })
    }

    fn is_ready(id: &str, attempt_id: u64) -> bool {
        let g = terminal_control().mu.lock().unwrap();
        match g.active.as_ref() {
            Some(a) if a.id == id && a.attempt_id == attempt_id => a.ready,
            _ => false,
        }
    }

    #[test]
    fn early_commit_before_ready_cannot_commit() {
        let _lock = begin_test();
        activate_terminal_control("r1", 7);
        apply_terminal_control("commit", "r1", 7);
        assert_eq!(
            decision_of("r1", 7),
            None,
            "commit before ready must be ignored"
        );
        assert!(!check_abort("r1"));
        // After ready, a fresh matching commit should still be required.
        assert!(mark_terminal_control_ready("r1", 7));
        assert_eq!(decision_of("r1", 7), None);
        apply_terminal_control("commit", "r1", 7);
        assert_eq!(decision_of("r1", 7), Some(TerminalControlDecision::Commit));
        reset();
    }

    #[test]
    fn exact_id_and_attempt_correlation() {
        let _lock = begin_test();
        activate_terminal_control("r1", 3);
        // Wrong id
        apply_terminal_control("abort", "r2", 3);
        assert_eq!(decision_of("r1", 3), None);
        assert!(!check_abort("r1"));
        // Wrong attempt
        apply_terminal_control("abort", "r1", 99);
        assert_eq!(decision_of("r1", 3), None);
        assert!(!check_abort("r1"));
        // Exact match
        apply_terminal_control("abort", "r1", 3);
        assert_eq!(decision_of("r1", 3), Some(TerminalControlDecision::Abort));
        assert!(check_abort("r1"));
        // check_abort only matches active id
        assert!(!check_abort("r2"));
        reset();
    }

    #[test]
    fn stale_and_malformed_controls_ignored() {
        let _lock = begin_test();
        activate_terminal_control("live", 5);
        // No active match: stale id/attempt
        apply_terminal_control("abort", "stale", 5);
        apply_terminal_control("commit", "live", 1);
        apply_terminal_control("abort", "live", 1);
        // Unknown kind
        apply_terminal_control("nope", "live", 5);
        assert_eq!(decision_of("live", 5), None);
        assert!(!is_ready("live", 5));
        assert!(!check_abort("live"));
        // Empty active: control without activation is a no-op
        clear_terminal_control();
        apply_terminal_control("abort", "live", 5);
        assert!(terminal_control().mu.lock().unwrap().active.is_none());
        reset();
    }

    #[test]
    fn abort_before_ready_wins() {
        let _lock = begin_test();
        activate_terminal_control("r1", 2);
        apply_terminal_control("abort", "r1", 2);
        assert!(check_abort("r1"));
        // Ready after abort does not clear abort; commit cannot overwrite.
        assert!(mark_terminal_control_ready("r1", 2));
        apply_terminal_control("commit", "r1", 2);
        assert_eq!(decision_of("r1", 2), Some(TerminalControlDecision::Abort));
        assert_eq!(
            wait_terminal_control_decision("r1", 2, Duration::from_millis(50)),
            ClientTerminalDecision::Abort
        );
        reset();
    }

    #[test]
    fn abort_after_ready_wins() {
        let _lock = begin_test();
        activate_terminal_control("r1", 4);
        assert!(mark_terminal_control_ready("r1", 4));
        apply_terminal_control("abort", "r1", 4);
        assert_eq!(decision_of("r1", 4), Some(TerminalControlDecision::Abort));
        // Subsequent commit ignored once decided
        apply_terminal_control("commit", "r1", 4);
        assert_eq!(decision_of("r1", 4), Some(TerminalControlDecision::Abort));
        assert!(check_abort("r1"));
        reset();
    }

    #[test]
    fn matching_ready_commit_succeeds() {
        let _lock = begin_test();
        activate_terminal_control("r1", 8);
        set_active_attempt_id(8);
        let mut sink = Vec::new();
        let pending = serde_json::json!({
            "type": "done",
            "id": "r1",
            "attempt_id": 8,
            "finish_reason": "stop",
            "tokens": 3,
        });
        let handle = std::thread::spawn(|| {
            // Spin until ready, then commit.
            for _ in 0..200 {
                if is_ready("r1", 8) {
                    apply_terminal_control("commit", "r1", 8);
                    return;
                }
                std::thread::sleep(Duration::from_millis(1));
            }
            panic!("never became ready");
        });
        let decision = await_client_terminal_commit(&mut sink, "r1", &pending);
        handle.join().unwrap();
        assert_eq!(decision, ClientTerminalDecision::Commit);
        let line = std::str::from_utf8(&sink).unwrap().trim();
        let v: serde_json::Value = serde_json::from_str(line).unwrap();
        assert_eq!(v["type"], "commit_ready");
        assert_eq!(v["id"], "r1");
        assert_eq!(v["attempt_id"], 8);
        assert_eq!(v["finish_reason"], "stop");
        assert_eq!(v["tokens"], 3);
        // commit_ready is pending_done with only type changed.
        let mut as_done = v.clone();
        as_done["type"] = serde_json::json!("done");
        assert_eq!(as_done, pending);
        reset();
    }

    #[test]
    fn timeout_classifies_abort() {
        let _lock = begin_test();
        activate_terminal_control("r1", 11);
        assert!(mark_terminal_control_ready("r1", 11));
        let decision = wait_terminal_control_decision("r1", 11, Duration::from_millis(30));
        assert_eq!(decision, ClientTerminalDecision::Abort);
        assert_eq!(decision_of("r1", 11), Some(TerminalControlDecision::Abort));
        assert!(check_abort("r1"));
        reset();
    }

    #[test]
    fn commit_ready_json_carries_full_pending_done() {
        let _lock = begin_test();
        activate_terminal_control("req-x", 42);
        set_active_attempt_id(42);
        // Pre-latch abort so await returns immediately after emit.
        apply_terminal_control("abort", "req-x", 42);
        let pending = serde_json::json!({
            "type": "done",
            "id": "req-x",
            "attempt_id": 42,
            "finish_reason": "length",
            "tokens": 7,
            "tok_s": 1.5,
        });
        let mut sink = Vec::new();
        let decision = await_client_terminal_commit(&mut sink, "req-x", &pending);
        assert_eq!(decision, ClientTerminalDecision::Abort);
        let line = std::str::from_utf8(&sink).unwrap().trim();
        let v: serde_json::Value = serde_json::from_str(line).unwrap();
        assert_eq!(v["type"], "commit_ready");
        assert_eq!(v["id"], "req-x");
        assert_eq!(v["attempt_id"], 42);
        assert_eq!(v["finish_reason"], "length");
        assert_eq!(v["tokens"], 7);
        assert_eq!(v["tok_s"], 1.5);
        // Only one line
        assert_eq!(sink.iter().filter(|&&b| b == b'\n').count(), 1);
        reset();
    }

    #[test]
    fn check_abort_matches_active_attempt_only() {
        let _lock = begin_test();
        activate_terminal_control("same", 1);
        apply_terminal_control("abort", "same", 1);
        assert!(check_abort("same"));
        // New activation clears prior abort latch.
        activate_terminal_control("same", 2);
        assert!(!check_abort("same"));
        apply_terminal_control("abort", "same", 2);
        assert!(check_abort("same"));
        reset();
    }
