// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Extracted from `crates/hipfire-runtime/examples/daemon.rs`
//! `#[cfg(test)] mod attempt_error_writer_contract` (3 tests).
//! Imports rewritten from `super::` to `hipfire_engine::{terminal,emit}`.
//! Added `write_error`/`write_typed_error` to `hipfire_engine::emit` so the
//! original `super::` assertions remain verbatim.

use hipfire_engine::emit::{emit_active_attempt_error, emit_error_with_id, emit_uncorrelated_error, write_error, write_typed_error};
use hipfire_engine::terminal::{active_attempt_id, set_active_attempt_id};

    fn parse_error_line(line: &str) -> serde_json::Value {
        let v: serde_json::Value = serde_json::from_str(line.trim()).expect("error JSON");
        assert_eq!(v["type"], "error");
        v
    }

    /// Mirrors generate-arm invalid tools/messages/pflash + mid-generation
    /// failures: after TLS activation, the writer used by those sites must echo
    /// the nonzero active id and typed fields. Signature has no attempt_id param.
    #[test]
    fn active_attempt_writer_echoes_nonzero_tls_id() {
        let attempt = 42_u64;
        set_active_attempt_id(attempt);
        assert_eq!(active_attempt_id(), attempt);

        let mut buf = Vec::new();
        // Same writer family as invalid-tools / invalid-messages / pflash override.
        emit_active_attempt_error(
            &mut buf,
            Some("req-tools"),
            "invalid tools field: expected a sequence",
            "validation",
            false,
            false,
        );
        emit_active_attempt_error(
            &mut buf,
            Some("req-msgs"),
            "invalid messages field: expected a sequence",
            "validation",
            false,
            false,
        );
        emit_active_attempt_error(
            &mut buf,
            Some("req-pflash"),
            "invalid pflash override: bad alpha",
            "validation",
            false,
            false,
        );
        // Later generation failure path (emit_error_with_id / write_error).
        emit_error_with_id(&mut buf, "req-gen", "mtp prefill: synthetic");
        write_error(&mut buf, "req-write", "forward failed");
        write_typed_error(
            &mut buf,
            "req-typed",
            "transient blip",
            "transient",
            true,
            false,
        );

        let text = String::from_utf8(buf).unwrap();
        let lines: Vec<&str> = text.lines().filter(|l| !l.is_empty()).collect();
        assert_eq!(lines.len(), 6);

        for (line, expect_id, class, retryable) in [
            (lines[0], "req-tools", "validation", false),
            (lines[1], "req-msgs", "validation", false),
            (lines[2], "req-pflash", "validation", false),
            (lines[3], "req-gen", "internal", false),
            (lines[4], "req-write", "internal", false),
            (lines[5], "req-typed", "transient", true),
        ] {
            let v = parse_error_line(line);
            assert_eq!(v["attempt_id"].as_u64(), Some(attempt), "line={line}");
            assert_ne!(v["attempt_id"].as_u64(), Some(0));
            assert_eq!(v["id"].as_str(), Some(expect_id));
            assert_eq!(v["class"].as_str(), Some(class));
            assert_eq!(v["retryable"].as_bool(), Some(retryable));
            assert_eq!(v["rolled_back"].as_bool(), Some(false));
            assert!(v.get("message").and_then(|m| m.as_str()).is_some());
        }

        set_active_attempt_id(0);
    }

    /// Missing/malformed attempt_id rejects before activation use the separate
    /// uncorrelated writer (attempt_id 0 only).
    #[test]
    fn uncorrelated_writer_emits_zero_before_activation() {
        set_active_attempt_id(0);
        let mut buf = Vec::new();
        emit_uncorrelated_error(
            &mut buf,
            Some("req-missing"),
            "generate missing attempt_id",
            "validation",
            false,
            false,
        );
        let v = parse_error_line(std::str::from_utf8(&buf).unwrap());
        assert_eq!(v["attempt_id"].as_u64(), Some(0));
        assert_eq!(v["class"], "validation");
        assert_eq!(v["id"], "req-missing");
    }

    /// Compile-time + runtime guard: active writer API has no attempt_id
    /// parameter, so callsites cannot supply zero independently. Changing TLS
    /// between emits must be reflected (proves ID is not captured/hard-coded).
    #[test]
    fn active_writer_reads_tls_not_callsite_constant() {
        set_active_attempt_id(7);
        let mut buf = Vec::new();
        emit_active_attempt_error(&mut buf, None, "a", "internal", false, false);
        set_active_attempt_id(99);
        emit_active_attempt_error(&mut buf, None, "b", "internal", false, false);
        let lines: Vec<_> = std::str::from_utf8(&buf)
            .unwrap()
            .lines()
            .filter(|l| !l.is_empty())
            .collect();
        assert_eq!(parse_error_line(lines[0])["attempt_id"].as_u64(), Some(7));
        assert_eq!(parse_error_line(lines[1])["attempt_id"].as_u64(), Some(99));
        set_active_attempt_id(0);
    }
