// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Draft-context exhaustion terminal contract.
//!
//! A mid-loop `position + block_size >= ctx_capacity` break with
//! `generated < max_tokens` must report `length` (no tool release, no cache
//! store), not a natural `stop`. Kept separate from
//! `qwen_dflash_semantic_terminal_tests.rs` so that file's rustfmt debt is not
//! dragged into this change.

use hipfire_generate::common::qwen_dflash_hit_length_cap;
use hipfire_generate::qwen::{
    qwen_dflash_apply_cache_action, qwen_dflash_cache_action, qwen_dflash_wire_terminal,
    QwenDflashWireTerminal,
};
use hipfire_runtime::prompt_frame::ToolCall;
use hipfire_runtime::spec::{ClientEvent, FinishSummary};

fn summary_tool_calls(calls: Vec<ToolCall>) -> FinishSummary {
    let n = calls.len();
    FinishSummary {
        events: vec![ClientEvent::ToolCalls(calls)],
        finish_reason: "tool_calls",
        tool_calls: n,
        visible_text: "Sure.".into(),
        decoded_eot: false,
        open_think: false,
    }
}

#[test]
fn ctx_exhausted_maps_to_length_with_budget_unspent() {
    let calls = vec![ToolCall {
        id: None,
        name: "t".into(),
        arguments: serde_json::json!({}),
        rendered_body: None,
    }];
    let fin = summary_tool_calls(calls);
    // The budget alone would not have stopped this turn.
    assert!(!qwen_dflash_hit_length_cap(10, 16, false, false));
    // Wrapper-level mapping (`generate_dflash` / dense spec epilogue): the
    // `ctx_exhausted` flag is OR-ed into the length decision.
    let ctx_exhausted = true;
    let hit_length_cap = ctx_exhausted || qwen_dflash_hit_length_cap(10, 16, false, false);
    let term = qwen_dflash_wire_terminal(&fin, hit_length_cap, false, "partial", false);
    match &term {
        QwenDflashWireTerminal::Done {
            finish_reason,
            release_tool_calls,
            store_cache,
            wire_tool_calls,
            fingerprint_text,
        } => {
            assert_eq!(*finish_reason, "length");
            assert!(!*release_tool_calls);
            assert!(!*store_cache);
            assert!(wire_tool_calls.is_empty());
            assert!(fingerprint_text.is_empty());
        }
        other => panic!("expected length Done, got {other:?}"),
    }
    let action = qwen_dflash_cache_action(&term);
    assert!(!action.store);
    assert!(
        qwen_dflash_apply_cache_action(|_, _| panic!("must not insert"), &action, vec![1, 2])
            .is_none()
    );
}
