// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Extracted from `crates/hipfire-runtime/examples/daemon.rs`
//! `#[cfg(test)] mod qwen_jinja_reasoning_tests`.
//! Original assertions preserved verbatim; import rewritten from
//! `super::qwen_jinja_reasoning` to `hipfire_engine::prompt::qwen_jinja_reasoning`.
//! Updated for typed `thinking_enabled` authority with legacy fallback.

use hipfire_engine::prompt::qwen_jinja_reasoning;

#[test]
fn exact_low_medium_xhigh_pass_through_legacy() {
    for (effort, cap) in [("low", 0), ("medium", 1024), ("xhigh", 0), ("low", 512)] {
        let (enable, out) = qwen_jinja_reasoning(None, Some(effort), cap);
        assert!(enable, "enable for {effort} cap {cap}");
        assert_eq!(out.as_deref(), Some(effort));
    }
}

#[test]
fn unset_and_auto_are_undefined_legacy() {
    for raw in [None, Some("auto")] {
        let (enable, out) = qwen_jinja_reasoning(None, raw, 0);
        assert!(enable, "unset/auto with cap 0 should be enabled");
        assert_eq!(out, None, "unset/auto must be undefined for {:?}", raw);
        let (enable2, out2) = qwen_jinja_reasoning(None, raw, 1);
        assert!(!enable2, "cap 1 disables even unset/auto");
        assert_eq!(out2, None);
    }
}

#[test]
fn disable_values_disable_and_drop_effort_legacy() {
    for eff in ["none", "off", "chat"] {
        let (enable, out) = qwen_jinja_reasoning(None, Some(eff), 0);
        assert!(!enable, "{eff} should disable");
        assert_eq!(out, None);
        let (enable1, out1) = qwen_jinja_reasoning(None, Some(eff), 1);
        assert!(!enable1);
        assert_eq!(out1, None);
    }
}

#[test]
fn case_mismatch_is_not_normalized_legacy() {
    let (enable, out) = qwen_jinja_reasoning(None, Some("Low"), 0);
    assert!(enable);
    assert_eq!(out.as_deref(), Some("Low"), "case must be preserved");
    let (enable2, out2) = qwen_jinja_reasoning(None, Some("MEDIUM"), 0);
    assert!(enable2);
    assert_eq!(out2.as_deref(), Some("MEDIUM"));
    let (enable3, out3) = qwen_jinja_reasoning(None, Some("Xhigh"), 512);
    assert!(enable3);
    assert_eq!(out3.as_deref(), Some("Xhigh"));
}

#[test]
fn empty_string_is_not_dropped_legacy() {
    let (enable, out) = qwen_jinja_reasoning(None, Some(""), 0);
    assert!(enable, "empty should still be enabled when cap !=1");
    assert_eq!(
        out.as_deref(),
        Some(""),
        "empty must be preserved as Some(\"\")"
    );
    let (enable1, out1) = qwen_jinja_reasoning(None, Some(""), 1);
    assert!(!enable1, "cap 1 disables even empty");
    assert_eq!(out1, None);
}

#[test]
fn unsupported_high_is_preserved_not_folded_legacy() {
    let (enable, out) = qwen_jinja_reasoning(None, Some("high"), 0);
    assert!(enable);
    assert_eq!(out.as_deref(), Some("high"));
}

#[test]
fn explicit_effort_with_cap_one_is_disabled_legacy() {
    let (enable, out) = qwen_jinja_reasoning(None, Some("low"), 1);
    assert!(!enable, "cap 1 disables thinking regardless of effort");
    assert_eq!(out, None);
}

#[test]
fn explicit_true_enables_with_low_and_uncapped() {
    // Acceptance: new client sends thinking_enabled=true with low effort and no max_think_tokens (0 = uncapped)
    let (enable, out) = qwen_jinja_reasoning(Some(true), Some("low"), 0);
    assert!(enable, "explicit true with low should enable");
    assert_eq!(out.as_deref(), Some("low"));
}

#[test]
fn explicit_false_disables_independently() {
    for eff in [None, Some("low"), Some("xhigh"), Some("auto"), Some("off")] {
        let (enable, out) = qwen_jinja_reasoning(Some(false), eff, 0);
        assert!(!enable, "explicit false should disable for eff {:?}", eff);
        assert_eq!(out, None);
        // Even with cap 1, still disabled (independent)
        let (enable2, out2) = qwen_jinja_reasoning(Some(false), eff, 1);
        assert!(!enable2);
        assert_eq!(out2, None);
        // Even with cap 512, still disabled
        let (enable3, out3) = qwen_jinja_reasoning(Some(false), eff, 512);
        assert!(!enable3);
        assert_eq!(out3, None);
    }
}

#[test]
fn explicit_true_effort_independence_from_cap() {
    // max_think_tokens is independent: explicit true ignores cap 1
    let (enable, out) = qwen_jinja_reasoning(Some(true), Some("low"), 1);
    assert!(
        enable,
        "explicit true should remain enabled even with cap 1 (independent)"
    );
    assert_eq!(out.as_deref(), Some("low"));
    let (enable2, out2) = qwen_jinja_reasoning(Some(true), Some("medium"), 1);
    assert!(enable2);
    assert_eq!(out2.as_deref(), Some("medium"));
    // explicit true with xhigh and uncapped
    let (enable3, out3) = qwen_jinja_reasoning(Some(true), Some("xhigh"), 0);
    assert!(enable3);
    assert_eq!(out3.as_deref(), Some("xhigh"));
}

#[test]
fn explicit_true_unset_and_auto_are_undefined() {
    let (enable, out) = qwen_jinja_reasoning(Some(true), None, 0);
    assert!(enable);
    assert_eq!(out, None);
    let (enable2, out2) = qwen_jinja_reasoning(Some(true), Some("auto"), 0);
    assert!(enable2);
    assert_eq!(out2, None);
    // caps should not affect explicit true
    let (enable3, out3) = qwen_jinja_reasoning(Some(true), None, 1);
    assert!(
        enable3,
        "explicit true with None should stay enabled even with cap 1"
    );
    assert_eq!(out3, None);
}

#[test]
fn legacy_fallback_preserved_when_thinking_absent() {
    // Old direct JSONL without thinking_enabled retains legacy behavior
    // cap 0 + low => enabled
    let (e, o) = qwen_jinja_reasoning(None, Some("low"), 0);
    assert!(e);
    assert_eq!(o.as_deref(), Some("low"));
    // cap 1 disables
    let (e, o) = qwen_jinja_reasoning(None, Some("low"), 1);
    assert!(!e);
    assert_eq!(o, None);
    // disable values
    let (e, o) = qwen_jinja_reasoning(None, Some("off"), 0);
    assert!(!e);
    assert_eq!(o, None);
}
