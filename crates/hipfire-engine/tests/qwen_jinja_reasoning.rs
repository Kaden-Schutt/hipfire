// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Extracted from `crates/hipfire-runtime/examples/daemon.rs`
//! `#[cfg(test)] mod qwen_jinja_reasoning_tests`.
//! Original assertions preserved verbatim; import rewritten from
//! `super::qwen_jinja_reasoning` to `hipfire_engine::prompt::qwen_jinja_reasoning`.

    use hipfire_engine::prompt::qwen_jinja_reasoning;

    #[test]
    fn exact_low_medium_xhigh_pass_through() {
        for (effort, cap) in [("low", 0), ("medium", 1024), ("xhigh", 0), ("low", 512)] {
            let (enable, out) = qwen_jinja_reasoning(Some(effort), cap);
            assert!(enable, "enable for {effort} cap {cap}");
            assert_eq!(out.as_deref(), Some(effort));
        }
    }

    #[test]
    fn unset_and_auto_are_undefined() {
        for raw in [None, Some("auto")] {
            let (enable, out) = qwen_jinja_reasoning(raw, 0);
            assert!(enable, "unset/auto with cap 0 should be enabled");
            assert_eq!(out, None, "unset/auto must be undefined for {:?}", raw);
            let (enable2, out2) = qwen_jinja_reasoning(raw, 1);
            assert!(!enable2, "cap 1 disables even unset/auto");
            assert_eq!(out2, None);
        }
    }

    #[test]
    fn disable_values_disable_and_drop_effort() {
        for eff in ["none", "off", "chat"] {
            let (enable, out) = qwen_jinja_reasoning(Some(eff), 0);
            assert!(!enable, "{eff} should disable");
            assert_eq!(out, None);
            let (enable1, out1) = qwen_jinja_reasoning(Some(eff), 1);
            assert!(!enable1);
            assert_eq!(out1, None);
        }
    }

    #[test]
    fn case_mismatch_is_not_normalized() {
        let (enable, out) = qwen_jinja_reasoning(Some("Low"), 0);
        assert!(enable);
        assert_eq!(out.as_deref(), Some("Low"), "case must be preserved");
        let (enable2, out2) = qwen_jinja_reasoning(Some("MEDIUM"), 0);
        assert!(enable2);
        assert_eq!(out2.as_deref(), Some("MEDIUM"));
        let (enable3, out3) = qwen_jinja_reasoning(Some("Xhigh"), 512);
        assert!(enable3);
        assert_eq!(out3.as_deref(), Some("Xhigh"));
    }

    #[test]
    fn empty_string_is_not_dropped() {
        let (enable, out) = qwen_jinja_reasoning(Some(""), 0);
        assert!(enable, "empty should still be enabled when cap !=1");
        assert_eq!(
            out.as_deref(),
            Some(""),
            "empty must be preserved as Some(\"\")"
        );
        let (enable1, out1) = qwen_jinja_reasoning(Some(""), 1);
        assert!(!enable1, "cap 1 disables even empty");
        assert_eq!(out1, None);
    }

    #[test]
    fn unsupported_high_is_preserved_not_folded() {
        let (enable, out) = qwen_jinja_reasoning(Some("high"), 0);
        assert!(enable);
        assert_eq!(out.as_deref(), Some("high"));
    }

    #[test]
    fn explicit_effort_with_cap_one_is_disabled() {
        let (enable, out) = qwen_jinja_reasoning(Some("low"), 1);
        assert!(!enable, "cap 1 disables thinking regardless of effort");
        assert_eq!(out, None);
    }
