// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt

//! Extracted from `crates/hipfire-daemon/src/main.rs`
//! `#[cfg(test)] mod tool_call_parser_tests` (block 11/22).
//! Original assertions preserved verbatim; import rewritten from
//! `super::extract_tool_calls_from_text` to the public path
//! `hipfire_runtime::emit_text::extract_tool_calls_from_text`.

use hipfire_runtime::emit_text::{extract_tool_call_name_fallback, extract_tool_calls_from_text};

#[test]
fn parses_valid_block() {
    let s = r#"prelude<tool_call>
{"name": "read", "arguments": {"path": "/etc/hostname"}}
</tool_call>tail"#;
    let calls = extract_tool_calls_from_text(s);
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].name, "read");
    assert_eq!(calls[0].arguments["path"], "/etc/hostname");
}

#[test]
fn handles_unclosed_tool_call() {
    // Model truncated at max_tokens before emitting </tool_call>.
    // OLD parser broke out of the loop; NEW parser treats rest of
    // string as body and recovers the call. This was the Pi-session
    // call-9 failure mode that flipped the asst-cache fingerprint
    // from tool_calls=1 (CLI) to tool_calls=0 (daemon) → full reset.
    let s = r#"prelude<tool_call>
{"name": "read", "arguments": {"path": "/etc/hostname"}}"#;
    let calls = extract_tool_calls_from_text(s);
    assert_eq!(calls.len(), 1, "unclosed block dropped — should recover");
    assert_eq!(calls[0].name, "read");
}

#[test]
fn truncated_args_not_emitted_as_empty() {
    // A `write` cut off mid-`content` (max_tokens / grammar force-close):
    // the args object never closes, so no balanced object is recoverable.
    // The OLD fallback fabricated empty `{}` args, presenting write({}) to
    // the client as executable (the write-tool empty-args incident). NEW:
    // drop the call entirely so the emission surfaces as content +
    // finish_reason for the client to retry. Distinct from
    // `handles_unclosed_tool_call`, where the args ARE complete and only
    // the `</tool_call>` marker is missing.
    let s = "<tool_call>\n{\"name\": \"write\", \"arguments\": {\"path\": \"/tmp/big.zig\", \"content\": \"const std = @im";
    let calls = extract_tool_calls_from_text(s);
    assert!(
        calls.is_empty(),
        "truncated args must NOT emit a fabricated-empty call"
    );
}

#[test]
fn loose_json_with_complete_args_still_recovered() {
    // Broken outer JSON (leading `{` lost to special-token leakage) but a
    // COMPLETE balanced args object — the fallback still recovers it,
    // distinguishing real recovery from the truncation case above.
    let s =
        "<tool_call>\nname\": \"read\", \"arguments\": {\"path\": \"/tmp/x\"}\n</tool_call>";
    let calls = extract_tool_calls_from_text(s);
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].name, "read");
    assert_eq!(calls[0].arguments["path"], "/tmp/x");
}

#[test]
fn strips_chatml_special_tokens_in_body() {
    let s = "<tool_call>\n<|im_start|>{\"name\": \"read\", \"arguments\": {\"path\": \"/x\"}}<|im_end|>\n</tool_call>";
    let calls = extract_tool_calls_from_text(s);
    assert_eq!(calls.len(), 1, "ChatML token leakage broke JSON parse");
    assert_eq!(calls[0].name, "read");
}

#[test]
fn nested_opener_stripped() {
    let s = r#"<tool_call>
<tool_call>
{"name": "read", "arguments": {"path": "/x"}}
</tool_call>"#;
    let calls = extract_tool_calls_from_text(s);
    assert_eq!(calls.len(), 1, "nested opener dropped");
    assert_eq!(calls[0].name, "read");
}

#[test]
fn no_block_no_calls() {
    let calls = extract_tool_calls_from_text("just text, no tool call");
    assert!(calls.is_empty());
}

#[test]
fn form4_skips_name_substring_in_other_key() {
    // `firstname` contains `name` — the fallback used to bail when
    // it saw an invalid pre-byte for the first match. Should now
    // skip and find the real `name` key on the next occurrence.
    // (Strict JSON parse handles this trivially; this test exercises
    // the fallback path by wrapping in <tool_call> with off-spec
    // shape that triggers fallback.)
    let body = r#"{"firstname":"X","name":"read","arguments":{"path":"/x"}}"#;
    assert_eq!(
        extract_tool_call_name_fallback(body),
        Some("read".to_string())
    );
}

#[test]
fn form4_handles_trailing_comma() {
    // serde_json rejects trailing commas; the fallback should
    // still find name + arguments.
    let s = r#"<tool_call>
{"name": "read", "arguments": {"path": "/x",},}
</tool_call>"#;
    let calls = extract_tool_calls_from_text(s);
    assert_eq!(calls.len(), 1, "trailing-comma JSON dropped");
    assert_eq!(calls[0].name, "read");
}

#[test]
fn form4_handles_unquoted_key() {
    // Off-spec JSON with unquoted key.
    let body = r#"{name: "read"}"#;
    assert_eq!(
        extract_tool_call_name_fallback(body),
        Some("read".to_string())
    );
}

#[test]
fn empty_body_no_call() {
    // Empty `<tool_call></tool_call>` shouldn't produce a call.
    let s = "<tool_call></tool_call>";
    let calls = extract_tool_calls_from_text(s);
    assert!(calls.is_empty());
}

#[test]
fn multiple_blocks_extract_all() {
    // Two valid tool_call blocks in one emission should yield two calls.
    let s = r#"<tool_call>
{"name":"a","arguments":{}}
</tool_call>prose<tool_call>
{"name":"b","arguments":{}}
</tool_call>"#;
    let calls = extract_tool_calls_from_text(s);
    assert_eq!(calls.len(), 2);
    assert_eq!(calls[0].name, "a");
    assert_eq!(calls[1].name, "b");
}
