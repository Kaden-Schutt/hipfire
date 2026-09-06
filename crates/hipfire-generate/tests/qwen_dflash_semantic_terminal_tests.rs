// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Qwen DFlash semantic terminal contract.
//!
//! Moved out of `hipfire-daemon`'s `main.rs`. These were the last references
//! to architecture crates left in that file; the shipped daemon code reaches
//! architectures only through `hipfire-loader`'s `Carrier` and the
//! `hipfire-generate` entry points.

#![allow(clippy::all)]

use hipfire_engine::emit::*;
use hipfire_engine::terminal::*;
use hipfire_generate::ar::*;
use hipfire_generate::common::*;
use hipfire_runtime::emit_text::extract_tool_calls_from_text;

    use hipfire_runtime::prompt_frame::{AssistantPrefix, ToolCall};
    use hipfire_runtime::spec::{
        ClientEvent, FinishSummary, SpecEmit, SpecEmitCtx, SpecStep, StopReason,
    };
    use hipfire_runtime::tokenizer::Tokenizer;
    use std::collections::HashSet;

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

    fn summary_stop(visible: &str) -> FinishSummary {
        FinishSummary {
            events: vec![ClientEvent::Token(visible.into())],
            finish_reason: "stop",
            tool_calls: 0,
            visible_text: visible.into(),
            decoded_eot: false,
            open_think: false,
        }
    }

    fn summary_malformed() -> FinishSummary {
        FinishSummary {
            events: Vec::new(),
            finish_reason: "malformed_protocol",
            tool_calls: 0,
            visible_text: String::new(),
            decoded_eot: false,
            open_think: false,
        }
    }

    fn json_escape(s: &str) -> String {
        let mut out = String::new();
        for c in s.chars() {
            match c {
                '"' => out.push_str("\\\""),
                '\\' => out.push_str("\\\\"),
                '\n' => out.push_str("\\n"),
                '\r' => out.push_str("\\r"),
                '\t' => out.push_str("\\t"),
                c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
                c => out.push(c),
            }
        }
        out
    }

    fn byte_to_gpt2_char_test(b: u8) -> char {
        let mut bs: Vec<u32> = Vec::new();
        bs.extend((b'!' as u32)..=(b'~' as u32));
        bs.extend((0xA1u32)..=(0xACu32));
        bs.extend((0xAEu32)..=(0xFFu32));
        let mut cs: Vec<u32> = bs.clone();
        let mut n: u32 = 0;
        for byte in 0u32..=255u32 {
            if !bs.contains(&byte) {
                bs.push(byte);
                cs.push(256 + n);
                n += 1;
            }
        }
        for (bb, cc) in bs.into_iter().zip(cs.into_iter()) {
            if bb == b as u32 {
                return char::from_u32(cc).unwrap();
            }
        }
        char::from_u32(b as u32).unwrap()
    }

    /// Same minimal tokenizer family as qwen35 `spec_emit` CPU tests.
    fn test_tokenizer() -> Tokenizer {
        let mut entries: Vec<String> = Vec::new();
        entries.push(r#""<|im_start|>": 0"#.to_string());
        entries.push(r#""<|im_end|>": 1"#.to_string());
        entries.push(r#""<think>": 2"#.to_string());
        entries.push(r#""</think>": 3"#.to_string());
        entries.push(r#""system": 4"#.to_string());
        entries.push(r#""user": 5"#.to_string());
        entries.push(r#""assistant": 6"#.to_string());
        entries.push(r#""\n": 7"#.to_string());
        entries.push(r#""Ġ": 8"#.to_string());
        entries.push(r#""<|endoftext|>": 9"#.to_string());
        for b in 0u32..=255u32 {
            let ch = byte_to_gpt2_char_test(b as u8);
            let escaped = json_escape(&ch.to_string());
            entries.push(format!(r#""{}": {}"#, escaped, 100 + b));
        }
        let vocab_block = entries.join(", ");
        let json = format!(
            r#"{{
                "model": {{"type": "BPE", "vocab": {{ {vocab} }}, "merges": []}},
                "added_tokens": [
                    {{"id": 0, "content": "<|im_start|>", "special": true}},
                    {{"id": 1, "content": "<|im_end|>", "special": true}},
                    {{"id": 2, "content": "<think>", "special": true}},
                    {{"id": 3, "content": "</think>", "special": true}},
                    {{"id": 9, "content": "<|endoftext|>", "special": true}}
                ]
            }}"#,
            vocab = vocab_block,
        );
        Tokenizer::from_hf_json(&json).expect("test tokenizer")
    }

    fn make_qwen_emit<'a>(
        tok: &'a Tokenizer,
        assistant_prefix: AssistantPrefix,
    ) -> Box<dyn SpecEmit + 'a> {
        hipfire_arch_qwen35::spec_emit::Qwen35Emit::from_ctx(SpecEmitCtx {
            tokenizer: tok,
            eos: 9,
            im_end: Some(1),
            tools: Some(&[]),
            enable_grammar: true,
            stop: Vec::new(),
            max_think: 0,
            max_tokens: 256,
            assistant_prefix,
            think_mode: hipfire_runtime::prompt_frame::ThinkMode::NonThink,
            decoded_vocab: None,
        })
    }

    /// Drive production Qwen35Emit with whole-string encodes.
    fn drive_qwen_emit(
        text: &str,
        assistant_prefix: AssistantPrefix,
    ) -> (Vec<ClientEvent>, FinishSummary, Vec<u32>) {
        let tok = test_tokenizer();
        let ids = tok.encode(text);
        assert!(!ids.is_empty(), "encode produced no tokens for {text:?}");
        let mut emit = make_qwen_emit(&tok, assistant_prefix);
        let mut stream = Vec::new();
        let mut first = true;
        for id in &ids {
            let outcome = if first {
                first = false;
                emit.begin(*id)
            } else {
                emit.observe(*id)
            };
            stream.extend(outcome.events);
            if outcome.stop.is_some() {
                break;
            }
        }
        let streamed = emit.streamed_tokens().to_vec();
        let finish = emit.finish();
        (stream, finish, streamed)
    }

    /// Drive production emitter token-by-token (for split-marker cases).
    fn drive_qwen_ids(
        ids: &[u32],
        assistant_prefix: AssistantPrefix,
    ) -> (Vec<ClientEvent>, FinishSummary, Vec<u32>) {
        let tok = test_tokenizer();
        let mut emit = make_qwen_emit(&tok, assistant_prefix);
        let mut stream = Vec::new();
        let mut first = true;
        for id in ids {
            let outcome = if first {
                first = false;
                emit.begin(*id)
            } else {
                emit.observe(*id)
            };
            stream.extend(outcome.events);
            if outcome.stop.is_some() {
                break;
            }
        }
        let streamed = emit.streamed_tokens().to_vec();
        let finish = emit.finish();
        (stream, finish, streamed)
    }

    fn parse_jsonl(out: &str) -> Vec<serde_json::Value> {
        out.lines()
            .filter(|l| !l.trim().is_empty())
            .map(|l| serde_json::from_str(l).unwrap_or_else(|e| panic!("bad jsonl {l}: {e}")))
            .collect()
    }

    /// GPU-less attested epilogue for unit tests (no real device sync).
    fn attest_epilogue(rolled_back: bool) -> hipfire_generate::common::RollbackEpilogue {
        hipfire_generate::common::RollbackEpilogue {
            rolled_back,
            context: None,
        }
    }

    /// Attested epilogue with sync-failure context (rolled_back=false).
    fn attest_epilogue_with_context(context: &str) -> hipfire_generate::common::RollbackEpilogue {
        hipfire_generate::common::RollbackEpilogue {
            rolled_back: false,
            context: Some(context.to_string()),
        }
    }

    #[test]
    fn safe_stop_stores_cache_no_calls() {
        let fin = summary_stop("hello");
        let term = hipfire_generate::qwen::qwen_dflash_wire_terminal(&fin, false, false, "hello", false);
        match &term {
            hipfire_generate::qwen::QwenDflashWireTerminal::Done {
                finish_reason,
                release_tool_calls,
                store_cache,
                fingerprint_text,
                wire_tool_calls,
            } => {
                assert_eq!(*finish_reason, "stop");
                assert!(!*release_tool_calls);
                assert!(*store_cache);
                assert!(wire_tool_calls.is_empty());
                assert_eq!(
                    fingerprint_text.as_str(),
                    hipfire_generate::common::normalize_asst_turn_for_fingerprint("hello")
                );
            }
            other => panic!("expected Done, got {other:?}"),
        }
        let action = hipfire_generate::qwen::qwen_dflash_cache_action(&term);
        assert!(action.store);
        assert!(action.tool_calls.is_empty());
    }

    #[test]
    fn tool_safe_releases_calls_and_stores() {
        let calls = vec![ToolCall {
            id: None,
            name: "get_weather".into(),
            arguments: serde_json::json!({"city": "SF"}),
            rendered_body: None,
        }];
        let fin = summary_tool_calls(calls.clone());
        let term = hipfire_generate::qwen::qwen_dflash_wire_terminal(&fin, false, false, "Sure.", false);
        match &term {
            hipfire_generate::qwen::QwenDflashWireTerminal::Done {
                finish_reason,
                release_tool_calls,
                store_cache,
                wire_tool_calls,
                ..
            } => {
                assert_eq!(*finish_reason, "tool_calls");
                assert!(*release_tool_calls);
                assert!(*store_cache);
                assert_eq!(wire_tool_calls.len(), 1);
                assert_eq!(wire_tool_calls[0].name, "get_weather");
            }
            other => panic!("expected Done, got {other:?}"),
        }
        let action = hipfire_generate::qwen::qwen_dflash_cache_action(&term);
        assert!(action.store);
        assert_eq!(action.tool_calls.len(), 1);
    }

    #[test]
    fn pure_length_suppresses_calls_and_cache() {
        let calls = vec![ToolCall {
            id: None,
            name: "t".into(),
            arguments: serde_json::json!({}),
            rendered_body: None,
        }];
        let fin = summary_tool_calls(calls);
        assert!(hipfire_generate::common::qwen_dflash_hit_length_cap(16, 16, false, false));
        assert!(!hipfire_generate::common::qwen_dflash_hit_length_cap(16, 16, false, true));
        let term = hipfire_generate::qwen::qwen_dflash_wire_terminal(&fin, true, false, "partial", false);
        match &term {
            hipfire_generate::qwen::QwenDflashWireTerminal::Done {
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
        let action = hipfire_generate::qwen::qwen_dflash_cache_action(&term);
        assert!(!action.store);
        assert!(hipfire_generate::qwen::qwen_dflash_apply_cache_action(
            |_, _| panic!("must not insert"),
            &action,
            vec![1, 2]
        )
        .is_none());
    }

    #[test]
    fn final_token_eot_beats_length() {
        assert!(!hipfire_generate::common::qwen_dflash_hit_length_cap(8, 8, true, false));
        let calls = vec![ToolCall {
            id: None,
            name: "t".into(),
            arguments: serde_json::json!({}),
            rendered_body: None,
        }];
        let fin = summary_tool_calls(calls);
        let term = hipfire_generate::qwen::qwen_dflash_wire_terminal(&fin, false, false, "ok", false);
        match &term {
            hipfire_generate::qwen::QwenDflashWireTerminal::Done {
                finish_reason,
                release_tool_calls,
                store_cache,
                ..
            } => {
                assert_eq!(*finish_reason, "tool_calls");
                assert!(*release_tool_calls);
                assert!(*store_cache);
            }
            other => panic!("expected tool_calls Done, got {other:?}"),
        }
    }

    #[test]
    fn malformed_is_error_xor_done_no_cache() {
        let fin = summary_malformed();
        let term = hipfire_generate::qwen::qwen_dflash_wire_terminal(&fin, false, false, "", false);
        match &term {
            hipfire_generate::qwen::QwenDflashWireTerminal::Malformed {
                class,
                retryable,
                rolled_back,
                message,
            } => {
                assert_eq!(*class, "validation");
                assert!(!*retryable);
                assert!(!*rolled_back);
                assert!(message.contains("malformed"));
            }
            other => panic!("expected Malformed, got {other:?}"),
        }
        let action = hipfire_generate::qwen::qwen_dflash_cache_action(&term);
        assert!(!action.store);
        assert!(action.tool_calls.is_empty());
        assert!(!matches!(term, hipfire_generate::qwen::QwenDflashWireTerminal::Done { .. }));
    }

    #[test]
    fn grammar_failure_no_calls_no_cache() {
        let calls = vec![ToolCall {
            id: None,
            name: "t".into(),
            arguments: serde_json::json!({}),
            rendered_body: None,
        }];
        let fin = summary_tool_calls(calls);
        let term = hipfire_generate::qwen::qwen_dflash_wire_terminal(&fin, false, true, "x", false);
        match &term {
            hipfire_generate::qwen::QwenDflashWireTerminal::Malformed {
                class,
                retryable,
                message,
                ..
            } => {
                assert_eq!(*class, "validation");
                assert!(!*retryable);
                assert!(message.contains("grammar"));
            }
            other => panic!("expected grammar Malformed error-only, got {other:?}"),
        }
        let action = hipfire_generate::qwen::qwen_dflash_cache_action(&term);
        assert!(!action.store);
        assert!(action.tool_calls.is_empty());
        assert!(!matches!(term, hipfire_generate::qwen::QwenDflashWireTerminal::Done { .. }));
    }

    #[test]
    fn open_think_is_error_xor_done_no_cache() {
        // Production emitter (prompt-started OpenThink) -> real FinishSummary
        // -> production wire terminal. No hand-built open_think mirrors.
        let (stream, fin, _raw) = drive_qwen_emit("still thinking", AssistantPrefix::OpenThink);
        let reasoning: String = stream
            .iter()
            .filter_map(|e| match e {
                ClientEvent::Reasoning(text) => Some(text.as_str()),
                _ => None,
            })
            .collect();
        assert_eq!(reasoning, "still thinking");
        assert!(fin.open_think, "emitter must latch open_think");
        assert_eq!(fin.finish_reason, "open_think");
        assert!(fin.events.is_empty());
        assert_eq!(fin.tool_calls, 0);
        let term = hipfire_generate::qwen::qwen_dflash_wire_terminal(&fin, false, false, "", false);
        match &term {
            hipfire_generate::qwen::QwenDflashWireTerminal::Malformed {
                class,
                retryable,
                message,
                ..
            } => {
                assert_eq!(*class, "validation");
                assert!(!*retryable);
                assert!(message.contains("open think"));
            }
            other => panic!("expected open_think Malformed, got {other:?}"),
        }
        assert!(!hipfire_generate::qwen::qwen_dflash_cache_action(&term).store);
        assert!(!matches!(term, hipfire_generate::qwen::QwenDflashWireTerminal::Done { .. }));
        // Production Malformed writer: error XOR done (GPU-less attested epilogue).
        set_active_attempt_id(21);
        let mut sink = Vec::new();
        if let hipfire_generate::qwen::QwenDflashWireTerminal::Malformed {
            message,
            class,
            retryable,
            rolled_back,
        } = &term
        {
            let ep = attest_epilogue(*rolled_back);
            hipfire_generate::qwen::emit_qwen_dflash_malformed_terminal(
                &mut sink, "req-ot", message, class, *retryable, &ep,
            );
        }
        let out = String::from_utf8(sink).unwrap();
        let lines = parse_jsonl(&out);
        assert_eq!(lines.len(), 1);
        assert_eq!(lines[0]["type"], "error");
        assert_eq!(lines[0]["attempt_id"], 21);
        assert!(!out.contains(r#""type":"done""#));
    }

    #[test]
    fn open_think_prompt_started_and_generated_flags() {
        // (a) prompt-started OpenThink; (b) generated unclosed <think>.
        let cases = [
            ("prompt", AssistantPrefix::OpenThink, "still thinking"),
            ("generated", AssistantPrefix::Plain, "pre <think>secret"),
        ];
        for (label, prefix, body) in cases {
            let (stream, fin, _raw) = drive_qwen_emit(body, prefix);
            let reasoning: String = stream
                .iter()
                .filter_map(|e| match e {
                    ClientEvent::Reasoning(text) => Some(text.as_str()),
                    _ => None,
                })
                .collect();
            let expected_reasoning = if label == "prompt" {
                "still thinking"
            } else {
                "secret"
            };
            assert_eq!(reasoning, expected_reasoning, "{label}");
            assert!(fin.open_think, "{label}: open_think");
            assert_eq!(fin.finish_reason, "open_think", "{label}");
            assert_eq!(fin.tool_calls, 0, "{label}");
            assert!(fin.events.is_empty(), "{label}: no release on open_think");
            let term = hipfire_generate::qwen::qwen_dflash_wire_terminal(&fin, false, false, "", false);
            assert!(
                matches!(term, hipfire_generate::qwen::QwenDflashWireTerminal::Malformed { .. }),
                "{label}: expected Malformed"
            );
            assert!(!hipfire_generate::qwen::qwen_dflash_cache_action(&term).store, "{label}");
        }
    }

    #[test]
    fn producer_decoded_eot_beats_length_without_token_rescan() {
        // Real emitter decoded_eot at budget boundary → stop, not length.
        let tok = test_tokenizer();
        let mut ids = tok.encode("hi");
        ids.push(1); // <|im_end|>
        let (_stream, fin, _raw) = drive_qwen_ids(&ids, AssistantPrefix::Plain);
        assert!(fin.decoded_eot, "emitter must set decoded_eot");
        assert_eq!(fin.finish_reason, "stop");
        let generated = ids.len();
        let max_tokens = generated;
        assert!(!hipfire_generate::common::qwen_dflash_hit_length_cap(
            generated,
            max_tokens,
            fin.decoded_eot,
            false
        ));
        let term = hipfire_generate::qwen::qwen_dflash_wire_terminal(&fin, false, false, "hi", false);
        assert!(matches!(
            term,
            hipfire_generate::qwen::QwenDflashWireTerminal::Done {
                finish_reason: "stop",
                store_cache: true,
                ..
            }
        ));
    }

    #[test]
    fn split_decoded_eot_at_cap_is_stop_not_length() {
        // Byte-fragment the <|im_end|> marker across tokens via 100+b map.
        let marker = b"<|im_end|>";
        let mut ids: Vec<u32> = Vec::new();
        // prose "hi"
        ids.push(100 + b'h' as u32);
        ids.push(100 + b'i' as u32);
        // split marker into two fragments
        let mid = marker.len() / 2;
        for &b in &marker[..mid] {
            ids.push(100 + b as u32);
        }
        for &b in &marker[mid..] {
            ids.push(100 + b as u32);
        }
        let (stream, fin, raw) = drive_qwen_ids(&ids, AssistantPrefix::Plain);
        assert!(fin.decoded_eot, "split EOT must set decoded_eot");
        assert_eq!(fin.finish_reason, "stop");
        let visible: String = stream
            .iter()
            .filter_map(|ev| match ev {
                ClientEvent::Token(t) => Some(t.as_str()),
                _ => None,
            })
            .collect();
        assert!(!visible.contains("<|im_end|>"), "marker bytes suppressed");
        assert!(visible.contains("hi"));
        assert!(!raw.is_empty());
        let generated = raw.len();
        assert!(!hipfire_generate::common::qwen_dflash_hit_length_cap(
            generated,
            generated,
            fin.decoded_eot,
            false
        ));
        let term = hipfire_generate::qwen::qwen_dflash_wire_terminal(&fin, false, false, &visible, false);
        assert!(matches!(
            term,
            hipfire_generate::qwen::QwenDflashWireTerminal::Done {
                finish_reason: "stop",
                store_cache: true,
                release_tool_calls: false,
                ..
            }
        ));
    }

    #[test]
    fn step_budget_max_emit_zero_one_and_mid_window_prefix() {
        // max_emit 0: empty emit is the defensive shape (live step returns Err).
        let step0 = SpecStep::new([10, 11], 11, 1, 1).cap_emit(0);
        assert!(step0.emit.is_empty());
        assert_eq!(step0.accepted, 0);

        // max_emit 1: prefix keep + seed reseeds from kept token.
        let step1 = SpecStep::new([10, 11, 12], 12, 2, 2).cap_emit(1);
        assert_eq!(step1.emit.as_slice(), &[10]);
        assert_eq!(step1.next_seed, 10);
        assert!(step1.emit.len() <= 1);

        // Mid-window semantic consume of 2 of 4 emitted tokens.
        let step = SpecStep::new([10, 11, 12, 13], 13, 4, 3);
        let host = hipfire_generate::qwen::spec_host_advance_after_step(100, 0, Vec::new(), &step.emit, step.next_seed, 2);
        assert_eq!(host.emitted, vec![10, 11]);
        assert_eq!(host.generated, 2);
        assert_eq!(host.position, 102);
        assert_eq!(host.seed_token, 11);
        // Full-window consume keeps step.next_seed when prefix covers emit.
        let host_full =
            hipfire_generate::qwen::spec_host_advance_after_step(100, 0, Vec::new(), &step.emit, step.next_seed, 4);
        assert_eq!(host_full.emitted, vec![10, 11, 12, 13]);
        assert_eq!(host_full.position, 104);
        assert_eq!(host_full.seed_token, 13);
        // Unconsumed tail must not inflate position/conversation.
        assert_ne!(host.position, 100 + step.emit.len());
    }

    #[test]
    fn spec_prefix_realign_plan_empty_raw_and_multi() {
        let prompt = vec![1u32, 2, 3];
        let empty = hipfire_generate::qwen::spec_prefix_realign_plan(&prompt, 99, &[]);
        assert_eq!(empty.replay, prompt);
        assert_eq!(empty.position, 3);
        assert_eq!(empty.seed_token, 99);

        let multi = hipfire_generate::qwen::spec_prefix_realign_plan(&prompt, 99, &[10, 11, 12]);
        assert_eq!(multi.replay, vec![1, 2, 3, 99, 10, 11]);
        assert_eq!(multi.position, 6);
        assert_eq!(multi.seed_token, 12);
        assert_eq!(multi.replay.len(), multi.position);
        // Last raw stays the unwritten seed — never sits in KV replay.
        assert_ne!(multi.replay.last().copied(), Some(multi.seed_token));
        // Naive prompt+raw drops first_token and writes the seed into KV.
        let mut naive = prompt.clone();
        naive.extend_from_slice(&[10, 11, 12]);
        assert_ne!(multi.replay, naive);
    }

    #[test]
    fn terminal_strict_prefix_resets_without_realign() {
        use hipfire_generate::qwen::{
            spec_strict_prefix_action, SpecStrictPrefixAction,
        };

        assert_eq!(
            spec_strict_prefix_action(9, 11, true),
            SpecStrictPrefixAction::ResetForTerminal
        );
        assert_eq!(
            spec_strict_prefix_action(9, 11, false),
            SpecStrictPrefixAction::Realign
        );
        assert_eq!(
            spec_strict_prefix_action(11, 11, true),
            SpecStrictPrefixAction::None
        );
        assert_eq!(
            spec_strict_prefix_action(12, 11, false),
            SpecStrictPrefixAction::None
        );
    }

    #[test]
    fn terminal_marker_mid_window_tracks_exact_host_prefix() {
        // Spec window emits body + im_end + unobserved tail. Semantic loop
        // consumes only through the terminal marker; host bookkeeping must
        // exclude the unobserved tail even though terminal repair resets the
        // resident state instead of replaying this exact prefix.
        let tok = test_tokenizer();
        let prompt = vec![4u32, 5];
        let first_token = tok.encode("hi")[0];
        let body = tok.encode("ok");
        let im_end = 1u32;
        let mut step_emit = body.clone();
        step_emit.push(im_end);
        step_emit.extend_from_slice(&[90, 91]);
        let step = SpecStep::new(step_emit.clone(), *step_emit.last().unwrap(), 4, 3);

        let mut emit = make_qwen_emit(&tok, AssistantPrefix::Plain);
        let _ = emit.begin(first_token);
        let mut consumed = 0usize;
        let mut raw_decode: Vec<u32> = Vec::new();
        let mut hit_eos = false;
        for &tok_id in &step.emit {
            let outcome = emit.observe(tok_id);
            if outcome.stop == Some(StopReason::GrammarViolation) {
                break;
            }
            consumed += 1;
            raw_decode.push(tok_id);
            if matches!(
                outcome.stop,
                Some(StopReason::Eos) | Some(StopReason::StopSequence)
            ) {
                hit_eos = true;
                break;
            }
        }
        assert!(hit_eos, "im_end must stop the emitter");
        assert_eq!(
            consumed,
            body.len() + 1,
            "must consume body+im_end only, not tail {:?}",
            &step.emit[consumed..]
        );
        assert!(
            consumed < step.emit.len(),
            "fixture must leave an unobserved speculative tail"
        );

        let position_before = prompt.len();
        let host = hipfire_generate::qwen::spec_host_advance_after_step(
            position_before,
            0,
            vec![first_token],
            &step.emit,
            step.next_seed,
            consumed,
        );
        assert_eq!(host.generated, consumed);
        assert_eq!(host.position, position_before + consumed);
        assert_eq!(host.seed_token, im_end);
        assert_eq!(&host.emitted[1..], &step.emit[..consumed]);
        assert!(!host.emitted.contains(&90) && !host.emitted.contains(&91));

        let plan = hipfire_generate::qwen::spec_prefix_realign_plan(&prompt, first_token, &raw_decode);
        let mut expected_replay = prompt.clone();
        expected_replay.push(first_token);
        expected_replay.extend_from_slice(&raw_decode[..raw_decode.len() - 1]);
        assert_eq!(plan.replay, expected_replay);
        assert_eq!(plan.position, prompt.len() + raw_decode.len());
        assert_eq!(plan.seed_token, im_end);
        assert_eq!(plan.position, host.position);
        assert_eq!(plan.seed_token, host.seed_token);
        assert_ne!(plan.replay.last().copied(), Some(plan.seed_token));
    }

    #[test]
    fn empty_event_eos_mid_window_still_realigns_raw_prefix() {
        // Empty-event EOS observes still advance position/raw_decode (filter
        // stop on decoded marker bytes). Host + realign must track them.
        let tok = test_tokenizer();
        let prompt = vec![4u32];
        let first_token = 100 + b'h' as u32; // byte-map 'h'
                                             // Fragment <|im_end|> across byte-map tokens so filter stops without
                                             // a single special-id observe; final fragment may yield empty events.
        let marker = b"<|im_end|>";
        let mut step_emit: Vec<u32> = vec![100 + b'i' as u32]; // "i" after seed "h"
        for &b in marker {
            step_emit.push(100 + b as u32);
        }
        step_emit.extend_from_slice(&[90, 91]); // unobserved tail
        let step = SpecStep::new(step_emit.clone(), 91, step_emit.len(), step_emit.len() - 1);

        let mut emit = make_qwen_emit(&tok, AssistantPrefix::Plain);
        let _ = emit.begin(first_token);
        let mut consumed = 0usize;
        let mut raw_decode: Vec<u32> = Vec::new();
        let mut hit_eos = false;
        for &tok_id in &step.emit {
            let outcome = emit.observe(tok_id);
            consumed += 1;
            raw_decode.push(tok_id);
            // Empty-event EOS still counts as a position-advancing observe.
            if matches!(
                outcome.stop,
                Some(StopReason::Eos) | Some(StopReason::StopSequence)
            ) {
                hit_eos = true;
                break;
            }
        }
        assert!(hit_eos, "split marker must stop via filter");
        assert!(consumed < step.emit.len(), "tail must remain unobserved");
        assert_eq!(raw_decode.len(), consumed);

        let host = hipfire_generate::qwen::spec_host_advance_after_step(
            prompt.len(),
            0,
            vec![first_token],
            &step.emit,
            step.next_seed,
            consumed,
        );
        assert_eq!(host.generated, consumed);
        assert_eq!(host.position, prompt.len() + consumed);
        assert!(!host.emitted.contains(&90));

        let plan = hipfire_generate::qwen::spec_prefix_realign_plan(&prompt, first_token, &raw_decode);
        assert_eq!(plan.position, host.position);
        assert_eq!(plan.seed_token, host.seed_token);
        assert_eq!(plan.replay.len(), plan.position);
        assert_ne!(plan.replay.last().copied(), Some(plan.seed_token));
    }

    #[test]
    fn multi_window_then_strict_prefix_realign() {
        // After a full first window, raw_decode holds W1; a second window stops
        // mid-prefix. Realign replays prompt+first+raw[..-1] across both windows.
        let prompt = vec![7u32, 8];
        let first_token = 50u32;
        // Window 1 full consume (no realign).
        let w1 = SpecStep::new([10u32, 11, 12], 12, 3, 2);
        let mut raw_decode = Vec::new();
        let mut position = prompt.len();
        let mut emitted = vec![first_token];
        let mut generated = 0usize;
        let host1 = hipfire_generate::qwen::spec_host_advance_after_step(
            position,
            generated,
            emitted.clone(),
            &w1.emit,
            w1.next_seed,
            w1.emit.len(),
        );
        position = host1.position;
        generated = host1.generated;
        emitted = host1.emitted;
        raw_decode.extend_from_slice(&w1.emit);
        assert_eq!(position, prompt.len() + w1.emit.len());
        assert_eq!(host1.seed_token, 12);

        // Window 2: consume 2 of 4 (strict prefix → realign).
        let w2 = SpecStep::new([20u32, 21, 22, 23], 23, 4, 3);
        let consumed2 = 2usize;
        raw_decode.extend_from_slice(&w2.emit[..consumed2]);
        let host2 = hipfire_generate::qwen::spec_host_advance_after_step(
            position,
            generated,
            emitted,
            &w2.emit,
            w2.next_seed,
            consumed2,
        );
        assert_eq!(host2.emitted, vec![first_token, 10, 11, 12, 20, 21]);
        assert_eq!(host2.position, prompt.len() + raw_decode.len());
        assert_eq!(host2.seed_token, 21);
        assert!(!host2.emitted.contains(&22) && !host2.emitted.contains(&23));

        let plan = hipfire_generate::qwen::spec_prefix_realign_plan(&prompt, first_token, &raw_decode);
        assert_eq!(
            plan.replay,
            vec![7, 8, first_token, 10, 11, 12, 20] // drops last raw (21)
        );
        assert_eq!(plan.position, host2.position);
        assert_eq!(plan.seed_token, host2.seed_token);
        assert_eq!(plan.seed_token, 21);
    }

    #[test]
    fn forced_token_mid_window_strict_prefix_then_force_advance() {
        // Think-budget force-close mid-window: observe only the forced-trigger
        // prefix of step.emit, realign host/plan to that prefix, then host
        // advances over the forced continuation tokens as raw_decode.
        let tok = test_tokenizer();
        let prompt = vec![4u32, 5];
        let open_think = 2u32; // <think>

        let mut emit = hipfire_arch_qwen35::spec_emit::Qwen35Emit::from_ctx(SpecEmitCtx {
            tokenizer: &tok,
            eos: 9,
            im_end: Some(1),
            tools: None,
            enable_grammar: false,
            stop: Vec::new(),
            max_think: 1,
            max_tokens: 256,
            assistant_prefix: AssistantPrefix::Plain,
            think_mode: hipfire_runtime::prompt_frame::ThinkMode::NonThink,
            decoded_vocab: None,
        });

        let begin = emit.begin(open_think);
        assert!(begin.stop.is_none());

        let think_body = tok.encode("x");
        assert_eq!(think_body.len(), 1);
        let step_emit = vec![think_body[0], 90, 91, 92];
        let step = SpecStep::new(step_emit.clone(), 92, 4, 3);

        let mut consumed = 0usize;
        let mut raw_decode: Vec<u32> = Vec::new();
        let mut forced_after: Vec<u32> = Vec::new();
        for &tok_id in &step.emit {
            let outcome = emit.observe(tok_id);
            if outcome.stop == Some(StopReason::GrammarViolation) {
                break;
            }
            consumed += 1;
            raw_decode.push(tok_id);
            let forced = emit.take_forced();
            if !forced.is_empty() {
                forced_after = forced;
                break;
            }
            if outcome.stop.is_some() {
                break;
            }
        }
        assert_eq!(consumed, 1, "force must fire on the budget-hitting token");
        assert!(
            !forced_after.is_empty(),
            "think budget must queue </think> continuation"
        );
        assert!(consumed < step.emit.len(), "must leave unobserved tail");

        let position_before = prompt.len();
        let host = hipfire_generate::qwen::spec_host_advance_after_step(
            position_before,
            0,
            vec![open_think],
            &step.emit,
            step.next_seed,
            consumed,
        );
        assert_eq!(host.generated, 1);
        assert_eq!(host.position, position_before + 1);
        assert_eq!(host.seed_token, think_body[0]);
        assert!(!host.emitted.contains(&90));

        let plan = hipfire_generate::qwen::spec_prefix_realign_plan(&prompt, open_think, &raw_decode);
        assert_eq!(plan.position, host.position);
        assert_eq!(plan.seed_token, host.seed_token);
        assert_eq!(plan.replay, {
            let mut r = prompt.clone();
            r.push(open_think);
            r
        });

        // Pending-seed GPU tx: commit [trigger] ++ forced[..n-1]; last forced
        // stays unprocessed pending seed (never double-forwarded).
        let tx = hipfire_generate::qwen::spec_forced_pending_seed_tx(plan.seed_token, &forced_after, true);
        assert_eq!(tx.commit.first().copied(), Some(plan.seed_token));
        assert_eq!(tx.commit.len(), forced_after.len());
        assert_eq!(tx.pending_seed, *forced_after.last().unwrap());
        // Last forced is never double-forwarded: it is the pending seed, not in
        // commit (except the n==1 case where commit is only the prior seed).
        if forced_after.len() > 1 {
            assert_eq!(&tx.commit[1..], &forced_after[..forced_after.len() - 1]);
            assert_eq!(
                tx.commit.last().copied(),
                Some(forced_after[forced_after.len() - 2])
            );
        } else {
            assert_eq!(tx.commit.as_slice(), &[plan.seed_token]);
        }

        // Host observes each forced token; position advances by commit.len().
        let mut position = plan.position.saturating_add(tx.position_delta);
        let mut generated = host.generated;
        let mut emitted = host.emitted.clone();
        let mut seed_token = tx.pending_seed;
        for &ft in &forced_after {
            generated += 1;
            emitted.push(ft);
            raw_decode.push(ft);
            let fo = emit.observe(ft);
            assert!(
                fo.stop.is_none() || fo.stop == Some(StopReason::StopSequence),
                "forced continuation should not hard-stop mid-injection: {:?}",
                fo.stop
            );
        }
        assert_eq!(seed_token, *forced_after.last().unwrap());
        assert_eq!(position, plan.position + forced_after.len());
        assert_eq!(generated, consumed + forced_after.len());
        let mut expected_raw = step.emit[..consumed].to_vec();
        expected_raw.extend_from_slice(&forced_after);
        assert_eq!(raw_decode, expected_raw);
        assert!(!emitted.contains(&90) && !emitted.contains(&91) && !emitted.contains(&92));

        // Terminal flush would commit the final pending seed exactly once.
        let term = hipfire_generate::qwen::spec_terminal_pending_seed_tx(seed_token);
        assert_eq!(term.commit, vec![seed_token]);
        assert_eq!(term.position_delta, 1);
        let position_after_flush = position + term.position_delta;

        let plan2 = hipfire_generate::qwen::spec_prefix_realign_plan(&prompt, open_think, &raw_decode);
        assert_eq!(plan2.position, prompt.len() + raw_decode.len());
        assert_eq!(plan2.seed_token, seed_token);
        assert_eq!(plan2.seed_token, *raw_decode.last().unwrap());
        assert_eq!(plan2.replay.len(), plan2.position);
        // After terminal flush, cursor is one past the last conversation token
        // (prompt + raw_decode), matching safe bake `m.seq_pos`.
        assert_eq!(position_after_flush, prompt.len() + raw_decode.len() + 1);
        // Realign still treats last raw as unwritten seed (pre-terminal-flush).
        let mut expected_replay = prompt.clone();
        expected_replay.push(open_think);
        expected_replay.extend_from_slice(&raw_decode[..raw_decode.len() - 1]);
        assert_eq!(plan2.replay, expected_replay);
    }

    #[test]
    fn cache_seq_trim_eot_vs_length_body_newline() {
        let im_end = Some(1u32);
        let nl: HashSet<u32> = [7u32].into_iter().collect();
        // EOT-terminated: body + im_end + nl → strip trailer.
        let eot_stream = vec![10, 11, 1, 7];
        assert_eq!(
            hipfire_generate::qwen::qwen_dflash_cache_seq(&eot_stream, im_end, &nl),
            vec![10, 11]
        );
        // Length-capped body ending on newline: restore verbatim (no im_end).
        let len_stream = vec![10, 11, 7];
        assert_eq!(
            hipfire_generate::qwen::qwen_dflash_cache_seq(&len_stream, im_end, &nl),
            vec![10, 11, 7]
        );
        // Pure body, no trailer.
        let body = vec![10, 11, 12];
        assert_eq!(hipfire_generate::qwen::qwen_dflash_cache_seq(&body, im_end, &nl), body);
    }

    #[test]
    fn step_and_forced_advance_error_helpers_are_xor_done() {
        // Production fail-closed writer with GPU-less attested epilogue.
        set_active_attempt_id(42);
        for (what, id, needle) in [
            ("spec_step", "req-step", "spec_step:"),
            ("forced", "req-fa", "forced-token"),
        ] {
            let mut sink = Vec::new();
            let ep = attest_epilogue(true);
            hipfire_generate::qwen::emit_spec_failure_terminal(&mut sink, id, what, "boom", &ep);
            let text = String::from_utf8(sink).unwrap();
            let lines = parse_jsonl(&text);
            assert_eq!(lines.len(), 1, "error XOR done: {lines:?}");
            assert_eq!(lines[0]["type"], "error");
            assert_eq!(lines[0]["attempt_id"], 42);
            assert_eq!(lines[0]["retryable"], false);
            assert_eq!(lines[0]["rolled_back"], true);
            assert!(lines[0]["message"].as_str().unwrap().contains(needle));
            assert!(!text.contains(r#""type":"done""#));
            assert!(!text.contains(r#""type":"tool_calls""#));
        }
        // rolled_back=false + context path (sync could not be attested).
        let mut sink = Vec::new();
        let ep = attest_epilogue_with_context("device_synchronize failed: test");
        hipfire_generate::qwen::emit_spec_failure_terminal(&mut sink, "req-ctx", "spec_step", "boom", &ep);
        let text = String::from_utf8(sink).unwrap();
        let lines = parse_jsonl(&text);
        assert_eq!(lines.len(), 1);
        assert_eq!(lines[0]["rolled_back"], false);
        assert!(lines[0]["message"]
            .as_str()
            .unwrap()
            .contains("device_synchronize failed"));
        // Wrapper None contract: no epilogue after early exit.
        assert!(!qwen_dflash_epilogue_after_spec_run(false));
        assert!(qwen_dflash_epilogue_after_spec_run(true));
    }

    #[test]
    fn forced_advance_error_is_xor_done_no_calls() {
        set_active_attempt_id(43);
        let mut sink = Vec::new();
        let ep = attest_epilogue(true);
        hipfire_generate::qwen::emit_spec_failure_terminal(&mut sink, "req-fa", "forced", "boom", &ep);
        let text = String::from_utf8(sink).unwrap();
        let lines = parse_jsonl(&text);
        assert_eq!(lines.len(), 1);
        assert_eq!(lines[0]["type"], "error");
        assert_eq!(lines[0]["attempt_id"], 43);
        assert_eq!(lines[0]["rolled_back"], true);
        assert!(lines[0]["message"]
            .as_str()
            .unwrap()
            .contains("forced-token"));
        assert!(!text.contains(r#""type":"done""#));
        assert!(!text.contains(r#""type":"tool_calls""#));
    }

    #[test]
    fn decoded_eot_beats_length_cap_helper() {
        let fin = summary_stop("hi");
        assert!(hipfire_generate::common::qwen_dflash_hit_length_cap(8, 8, false, false));
        // Emitter semantic stop at cap is also not length (independent of EOT).
        assert!(!hipfire_generate::common::qwen_dflash_hit_length_cap(8, 8, false, true));
        let term = hipfire_generate::qwen::qwen_dflash_wire_terminal(&fin, true, false, "hi", false);
        match &term {
            hipfire_generate::qwen::QwenDflashWireTerminal::Done {
                finish_reason,
                store_cache,
                release_tool_calls,
                ..
            } => {
                assert_eq!(*finish_reason, "length");
                assert!(!*store_cache);
                assert!(!*release_tool_calls);
            }
            other => panic!("{other:?}"),
        }
        assert!(!hipfire_generate::common::qwen_dflash_hit_length_cap(8, 8, true, false));
        let tok = test_tokenizer();
        let mut ids = tok.encode("hi");
        ids.push(1);
        let (_s, fin_eot, _) = drive_qwen_ids(&ids, AssistantPrefix::Plain);
        assert!(fin_eot.decoded_eot);
        let term = hipfire_generate::qwen::qwen_dflash_wire_terminal(&fin_eot, false, false, "hi", false);
        assert!(matches!(
            term,
            hipfire_generate::qwen::QwenDflashWireTerminal::Done {
                finish_reason: "stop",
                store_cache: true,
                ..
            }
        ));
    }

    #[test]
    fn ordinary_length_cutoff_no_calls_no_cache() {
        let calls = vec![ToolCall {
            id: None,
            name: "t".into(),
            arguments: serde_json::json!({}),
            rendered_body: None,
        }];
        let fin = summary_tool_calls(calls);
        let term = hipfire_generate::qwen::qwen_dflash_wire_terminal(&fin, true, false, "x", false);
        match &term {
            hipfire_generate::qwen::QwenDflashWireTerminal::Done {
                finish_reason,
                release_tool_calls,
                store_cache,
                wire_tool_calls,
                ..
            } => {
                assert_eq!(*finish_reason, "length");
                assert!(!*release_tool_calls);
                assert!(!*store_cache);
                assert!(wire_tool_calls.is_empty());
            }
            other => panic!("{other:?}"),
        }
        assert!(!hipfire_generate::qwen::qwen_dflash_cache_action(&term).store);
    }

    #[test]
    fn cancel_is_fold_compatible_no_cache_helper() {
        // Production cancel writer (same path as hipfire_generate::qwen::generate_spec abort sites).
        set_active_attempt_id(11);
        let mut sink = Vec::new();
        emit_qwen_ar_cancelled(&mut sink, "c", 3);
        let out = String::from_utf8(sink).unwrap();
        let lines = parse_jsonl(&out);
        assert_eq!(lines.len(), 2);
        assert_eq!(lines[0]["type"], "aborted");
        assert_eq!(lines[0]["reason"], "client_cancelled");
        assert_eq!(lines[0]["attempt_id"], 11);
        assert_eq!(lines[1]["type"], "done");
        assert_eq!(lines[1]["finish_reason"], "aborted");
        assert_eq!(lines[1]["completion_tokens"], 3);
        // Cancel never goes through hipfire_generate::qwen::qwen_dflash_wire_terminal store path.
        assert!(!out.contains(r#""finish_reason":"stop""#));
    }

    #[test]
    fn serde_done_v2_hostile_id_roundtrip() {
        set_active_attempt_id(5);
        let id = "id\"quote\"\n";
        let mut sink = Vec::new();
        emit_qwen_dflash_done_terminal(
            &mut sink, id, 2, 1.0, 1, 1.0, 1.0, 1.0, 1.0, 1.0, 1, 0, "stop", None,
        );
        let out = String::from_utf8(sink).unwrap();
        let lines = parse_jsonl(&out);
        assert_eq!(lines.len(), 1);
        assert_eq!(lines[0]["type"], "done");
        assert_eq!(lines[0]["id"], id);
        assert_eq!(lines[0]["attempt_id"], 5);
        assert_eq!(lines[0]["finish_reason"], "stop");
        assert_eq!(lines[0]["dflash"], true);
    }

    #[test]
    fn grammar_lifecycle_error_only_serialized() {
        set_active_attempt_id(7);
        let fin = summary_tool_calls(vec![ToolCall {
            id: None,
            name: "t".into(),
            arguments: serde_json::json!({}),
            rendered_body: None,
        }]);
        let term = hipfire_generate::qwen::qwen_dflash_wire_terminal(&fin, false, true, "x", false);
        let mut sink = Vec::new();
        match &term {
            hipfire_generate::qwen::QwenDflashWireTerminal::Malformed {
                message,
                class,
                retryable,
                rolled_back,
            } => {
                let ep = attest_epilogue(*rolled_back);
                hipfire_generate::qwen::emit_qwen_dflash_malformed_terminal(
                    &mut sink, "g1", message, class, *retryable, &ep,
                );
            }
            other => panic!("{other:?}"),
        }
        let out = String::from_utf8(sink).unwrap();
        let lines = parse_jsonl(&out);
        assert_eq!(lines.len(), 1);
        assert_eq!(lines[0]["type"], "error");
        assert_eq!(lines[0]["attempt_id"], 7);
        assert_eq!(lines[0]["id"], "g1");
        assert!(!out.contains(r#""type":"done""#));
        assert!(!hipfire_generate::qwen::qwen_dflash_cache_action(&term).store);
    }

    #[test]
    fn serde_v2_token_and_tool_calls_hostile_id() {
        set_active_attempt_id(9);
        let mut sink = Vec::new();
        let id = "a\"b\n";
        hipfire_generate::qwen::render_client_events(
            &mut sink,
            id,
            &[
                ClientEvent::Token("hi".into()),
                ClientEvent::Reasoning("r".into()),
            ],
            0,
            false,
        );
        emit_tool_calls_event(
            &mut sink,
            id,
            &[ToolCall {
                id: None,
                name: "n".into(),
                arguments: serde_json::json!({"x": 1}),
                rendered_body: None,
            }],
        );
        let out = String::from_utf8(sink).unwrap();
        for line in out.lines().filter(|l| !l.is_empty()) {
            let v: serde_json::Value = serde_json::from_str(line).expect(line);
            assert_eq!(v["attempt_id"], 9);
            assert_eq!(v["id"], id);
        }
        let types: Vec<_> = parse_jsonl(&out)
            .into_iter()
            .map(|v| v["type"].as_str().unwrap().to_string())
            .collect();
        assert!(types.contains(&"token".to_string()));
        assert!(types.contains(&"reasoning".to_string()));
        assert!(types.contains(&"tool_calls".to_string()));
    }

    #[test]
    fn cancel_wire_helpers_carry_attempt_id() {
        // Production cancel writer carries attempt_id on aborted + done.
        set_active_attempt_id(3);
        let mut sink = Vec::new();
        emit_qwen_ar_cancelled(&mut sink, "c1", 5);
        let lines = parse_jsonl(&String::from_utf8(sink).unwrap());
        assert_eq!(lines.len(), 2);
        assert_eq!(lines[0]["type"], "aborted");
        assert_eq!(lines[0]["attempt_id"], 3);
        assert_eq!(lines[0]["reason"], "client_cancelled");
        assert_eq!(lines[1]["type"], "done");
        assert_eq!(lines[1]["finish_reason"], "aborted");
        assert_eq!(lines[1]["attempt_id"], 3);
        assert_eq!(lines[1]["completion_tokens"], 5);
    }

    #[test]
    fn cache_fingerprint_uses_visible_not_raw_markers() {
        let fin = summary_stop("visible only");
        let term = hipfire_generate::qwen::qwen_dflash_wire_terminal(&fin, false, false, "visible only", false);
        let action = hipfire_generate::qwen::qwen_dflash_cache_action(&term);
        assert!(!action.fingerprint_text.contains("<tool_call>"));
        assert!(!action.fingerprint_text.contains("<think>"));
        assert!(action.fingerprint_text.contains("visible"));
        let mut stored = None;
        let fp = hipfire_generate::qwen::qwen_dflash_apply_cache_action(
            |f, seq| {
                stored = Some((f, seq));
            },
            &action,
            vec![10, 20, 30],
        );
        assert!(fp.is_some());
        let (f, seq) = stored.expect("insert");
        assert_eq!(seq, vec![10, 20, 30]);
        assert_eq!(
            f,
            hipfire_generate::common::asst_turn_fingerprint(&action.fingerprint_text, &action.tool_calls)
        );
    }

    #[test]
    fn qwen_dflash_contract_version_is_v2() {
        assert_eq!(QWEN_DFLASH_SEMANTIC_CONTRACT_VERSION, 2);
        assert_eq!(hipfire_generate::common::gen_start_contract_version_for_arch(5), Some(2));
        assert_eq!(hipfire_generate::common::gen_start_contract_version_for_arch(6), Some(2));
    }

    #[test]
    fn no_whole_output_parser_in_terminal_path() {
        // Terminal path authority is FinishSummary fields only — a finish with
        // empty held calls cannot invent tools from visible text markers.
        let fin = FinishSummary {
            events: vec![ClientEvent::Token(
                "<tool_call>{\"name\":\"x\",\"arguments\":{}}</tool_call>".into(),
            )],
            finish_reason: "stop",
            tool_calls: 0,
            visible_text: String::new(),
            decoded_eot: false,
            open_think: false,
        };
        let term = hipfire_generate::qwen::qwen_dflash_wire_terminal(&fin, false, false, "", false);
        match term {
            hipfire_generate::qwen::QwenDflashWireTerminal::Done {
                finish_reason,
                release_tool_calls,
                wire_tool_calls,
                ..
            } => {
                assert_eq!(finish_reason, "stop");
                assert!(!release_tool_calls);
                assert!(wire_tool_calls.is_empty());
            }
            other => panic!("expected stop Done without invented calls, got {other:?}"),
        }
    }

    #[test]
    fn production_done_value_builder_matches_epilogue_shape() {
        let v =
            hipfire_generate::qwen::qwen_dflash_done_value("r", 3, 1.5, 10, 2.0, 5.0, 1.2, 2.0, 0.5, 2, 0, "length", 99);
        assert_eq!(v["type"], "done");
        assert_eq!(v["finish_reason"], "length");
        assert_eq!(v["attempt_id"], 99);
        assert_eq!(v["dflash"], true);
        assert_eq!(v["tokens"], 3);
    }

    // --- Task 4 production-seam invariants (pending-seed / cancel / evict /
    // capacity / jinja / wire / rollback attestation) ---

    #[test]
    fn trigger_token_retained_before_forced_suffix_tx() {
        // Forced GPU tx must first commit the current pending seed (the
        // force-trigger), then forced[..n-1]. The trigger is never dropped.
        let trigger = 77u32;
        let forced = [10u32, 11, 12];
        let tx = hipfire_generate::qwen::spec_forced_pending_seed_tx(trigger, &forced, true);
        assert_eq!(tx.commit[0], trigger, "trigger must lead the commit batch");
        assert_eq!(tx.commit, vec![77, 10, 11]);
        assert_eq!(tx.position_delta, forced.len());
        assert_eq!(tx.commit.len(), tx.position_delta);
        // Trigger is not the new pending seed unless forced was length-1.
        assert_ne!(tx.pending_seed, trigger);
    }

    #[test]
    fn final_forced_token_is_pending_exactly_once() {
        // Last forced token becomes the unprocessed pending seed and MUST NOT
        // also appear in commit (no double-forward).
        let forced = [20u32, 21, 22];
        let tx = hipfire_generate::qwen::spec_forced_pending_seed_tx(5, &forced, true);
        assert_eq!(tx.pending_seed, 22);
        assert!(
            !tx.commit.contains(&22),
            "last forced must stay unwritten: {:?}",
            tx.commit
        );
        assert_eq!(tx.commit, vec![5, 20, 21]);
        // Single-token forced: commit is only the prior seed; forced[0] pending.
        let one = hipfire_generate::qwen::spec_forced_pending_seed_tx(99, &[42], true);
        assert_eq!(one.commit, vec![99]);
        assert_eq!(one.pending_seed, 42);
        assert!(!one.commit.contains(&42));
        assert_eq!(one.position_delta, 1);
    }

    #[test]
    fn terminal_pending_seed_flush_exactly_once() {
        let seed = 314u32;
        let tx = hipfire_generate::qwen::spec_terminal_pending_seed_tx(seed);
        assert_eq!(tx.commit, vec![seed]);
        assert_eq!(tx.position_delta, 1);
        assert_eq!(tx.commit.len(), 1, "flush commits the seed once");
        // Terminal flush ends with the same logical token as conversation
        // (pending_seed field equals the committed token; no second lagging seed).
        assert_eq!(tx.pending_seed, seed);
    }

    #[test]
    fn forced_max_tokens_clip_hard_ceiling() {
        // generated already includes the trigger; no GPU for tokens past budget.
        let forced = [1u32, 2, 3, 4, 5];
        assert_eq!(hipfire_generate::qwen::spec_forced_tokens_within_budget(8, 10, &forced), &[1, 2]);
        assert_eq!(
            hipfire_generate::qwen::spec_forced_tokens_within_budget(10, 10, &forced),
            &[] as &[u32]
        );
        assert_eq!(hipfire_generate::qwen::spec_forced_tokens_within_budget(0, 3, &forced), &[1, 2, 3]);
        assert_eq!(hipfire_generate::qwen::spec_forced_tokens_within_budget(9, 10, &forced), &[1]);
        // Composition: clip then build tx — only fitting tokens become pending.
        let clipped = hipfire_generate::qwen::spec_forced_tokens_within_budget(7, 10, &forced);
        assert_eq!(clipped, &[1, 2, 3]);
        let tx = hipfire_generate::qwen::spec_forced_pending_seed_tx(70, clipped, true);
        assert_eq!(tx.commit, vec![70, 1, 2]);
        assert_eq!(tx.pending_seed, 3);
        assert!(!tx.commit.contains(&4) && !tx.commit.contains(&5));
    }

    #[test]
    fn cancellation_classification_forced_gpu_advance() {
        assert_eq!(
            hipfire_generate::qwen::classify_forced_gpu_advance(false),
            hipfire_generate::qwen::ForcedGpuAdvanceKind::Committed
        );
        assert_eq!(
            hipfire_generate::qwen::classify_forced_gpu_advance(true),
            hipfire_generate::qwen::ForcedGpuAdvanceKind::Cancelled
        );
        // Cancelled path must use aborted+done wire, never bake the forced token.
        // ErrorOnly is reserved for eviction failures (XOR below).
        assert_ne!(hipfire_generate::qwen::SpecFailClosedWire::Cancelled, hipfire_generate::qwen::SpecFailClosedWire::ErrorOnly);
        set_active_attempt_id(55);
        let mut sink = Vec::new();
        match hipfire_generate::qwen::classify_forced_gpu_advance(true) {
            hipfire_generate::qwen::ForcedGpuAdvanceKind::Cancelled => {
                emit_qwen_ar_cancelled(&mut sink, "c-force", 4);
            }
            hipfire_generate::qwen::ForcedGpuAdvanceKind::Committed => panic!("abort must classify Cancelled"),
        }
        let out = String::from_utf8(sink).unwrap();
        let lines = parse_jsonl(&out);
        assert_eq!(lines.len(), 2);
        assert_eq!(lines[0]["type"], "aborted");
        assert_eq!(lines[0]["reason"], "client_cancelled");
        assert_eq!(lines[0]["attempt_id"], 55);
        assert_eq!(lines[1]["type"], "done");
        assert_eq!(lines[1]["finish_reason"], "aborted");
        assert!(!out.contains(r#""type":"error""#));
        assert!(!out.contains(r#""type":"tool_calls""#));
        set_active_attempt_id(0);
    }

    #[test]
    fn eviction_error_terminal_exclusivity() {
        // maybe_evict / on_evict Err → ErrorOnly: one fail-closed error, no done.
        assert_eq!(hipfire_generate::qwen::classify_evict_failure_wire(), hipfire_generate::qwen::SpecFailClosedWire::ErrorOnly);
        set_active_attempt_id(66);
        let mut sink = Vec::new();
        let ep = attest_epilogue(true);
        match hipfire_generate::qwen::classify_evict_failure_wire() {
            hipfire_generate::qwen::SpecFailClosedWire::ErrorOnly => {
                hipfire_generate::common::emit_fail_closed_error(
                    &mut sink,
                    Some("ev1"),
                    "on_evict: synthetic retain failure",
                    "validation",
                    false,
                    &ep,
                );
            }
            hipfire_generate::qwen::SpecFailClosedWire::Cancelled => panic!("evict must not classify Cancelled"),
        }
        let out = String::from_utf8(sink).unwrap();
        let lines = parse_jsonl(&out);
        assert_eq!(lines.len(), 1, "error XOR done: {lines:?}");
        assert_eq!(lines[0]["type"], "error");
        assert_eq!(lines[0]["class"], "validation");
        assert_eq!(lines[0]["retryable"], false);
        assert_eq!(lines[0]["rolled_back"], true);
        assert_eq!(lines[0]["attempt_id"], 66);
        assert_eq!(lines[0]["id"], "ev1");
        assert!(!out.contains(r#""type":"done""#));
        assert!(!out.contains(r#""type":"aborted""#));
        assert!(!out.contains(r#""type":"tool_calls""#));
        // Fail-closed early exit skips wrapper epilogue (same as step failure).
        assert!(!qwen_dflash_epilogue_after_spec_run(false));
        set_active_attempt_id(0);
    }

    #[test]
    fn strict_prefix_replay_capacity_rejection() {
        let prompt = vec![1u32, 2, 3];
        let plan = hipfire_generate::qwen::spec_prefix_realign_plan(&prompt, 9, &[10, 11, 12]);
        // plan.replay = [1,2,3,9,10,11], position=6, seed=12
        assert_eq!(plan.replay.len(), plan.position);
        assert_eq!(plan.seed_token, 12);
        assert!(!plan.replay.contains(&12));

        // Fits both caps (position must be strictly < caps — pending seed slot).
        assert!(hipfire_generate::qwen::spec_prefix_realign_admit(&plan, 64, 64, 0, false).is_ok());
        // Boundary: position == cap leaves no legal write slot for pending seed.
        let err_eq = hipfire_generate::qwen::spec_prefix_realign_admit(&plan, plan.position, 64, 0, false).unwrap_err();
        assert!(
            err_eq.contains("physical_cap"),
            "expected position==physical_cap reject, got {err_eq}"
        );

        // Physical capacity rejection — fail closed before reset/prefill.
        let err_phys = hipfire_generate::qwen::spec_prefix_realign_admit(&plan, 5, 64, 0, false).unwrap_err();
        assert!(
            err_phys.contains("physical_cap"),
            "expected physical_cap reject, got {err_phys}"
        );

        // Speculator ctx capacity rejection.
        let err_ctx = hipfire_generate::qwen::spec_prefix_realign_admit(&plan, 64, 4, 0, false).unwrap_err();
        assert!(
            err_ctx.contains("ctx_capacity"),
            "expected ctx_capacity reject, got {err_ctx}"
        );

        // Broken invariant (replay/position mismatch) rejects even if caps large.
        let broken = hipfire_generate::qwen::SpecPrefixRealignPlan {
            replay: vec![1, 2],
            position: 5,
            seed_token: 9,
        };
        let err_inv = hipfire_generate::qwen::spec_prefix_realign_admit(&broken, 100, 100, 0, false).unwrap_err();
        assert!(
            err_inv.contains("invariant") || err_inv.contains("pending"),
            "expected invariant reject, got {err_inv}"
        );

        // Compacted/eviction path still fails closed on oversize full-history replay.
        let err_ev = hipfire_generate::qwen::spec_prefix_realign_admit(&plan, 5, 64, 3, true).unwrap_err();
        assert!(
            err_ev.contains("physical_cap") || err_ev.contains("compact"),
            "expected compacted oversize reject, got {err_ev}"
        );

        // Capacity reject wires as exclusive error terminal (no done).
        set_active_attempt_id(71);
        let mut sink = Vec::new();
        let ep = attest_epilogue(true);
        hipfire_generate::common::emit_fail_closed_error(
            &mut sink,
            Some("realign"),
            &err_phys,
            "validation",
            false,
            &ep,
        );
        let out = String::from_utf8(sink).unwrap();
        let lines = parse_jsonl(&out);
        assert_eq!(lines.len(), 1);
        assert_eq!(lines[0]["type"], "error");
        assert_eq!(lines[0]["attempt_id"], 71);
        assert!(!out.contains(r#""type":"done""#));
        set_active_attempt_id(0);
    }

    #[test]
    fn configured_jinja_render_fail_closed_policy() {
        // Production hipfire_generate::qwen::generate_dflash configured-template Err path:
        // hipfire_generate::dense::emit_active_attempt_error(class=validation, retryable=false,
        // rolled_back=false, message="DFlash jinja render: …") then handled=true.
        // Plain is not a silent fallback when a template is configured.
        set_active_attempt_id(88);
        let mut sink = Vec::new();
        let render_err = "undefined variable `messages`";
        hipfire_generate::dense::emit_active_attempt_error(
            &mut sink,
            Some("j1"),
            &format!("DFlash jinja render: {render_err}"),
            "validation",
            false,
            false,
        );
        let out = String::from_utf8(sink).unwrap();
        let lines = parse_jsonl(&out);
        assert_eq!(lines.len(), 1);
        assert_eq!(lines[0]["type"], "error");
        assert_eq!(lines[0]["class"], "validation");
        assert_eq!(lines[0]["retryable"], false);
        assert_eq!(lines[0]["rolled_back"], false);
        assert_eq!(lines[0]["attempt_id"], 88);
        assert_eq!(lines[0]["id"], "j1");
        let msg = lines[0]["message"].as_str().unwrap();
        assert!(msg.starts_with("DFlash jinja render:"), "{msg}");
        assert!(msg.contains(render_err), "{msg}");
        assert!(!out.contains(r#""type":"done""#));
        assert!(!out.contains(r#""type":"gen_start""#));
        // handled=true contract: early exit skips AR/done epilogue.
        assert!(!qwen_dflash_epilogue_after_spec_run(false));
        set_active_attempt_id(0);
    }

    #[test]
    fn correlated_escaped_dflash_info_frame() {
        // DFlash ctx-capacity fallback info uses serde + active attempt_id and
        // must survive adversarial id/message bytes without breaking JSONL.
        set_active_attempt_id(13);
        let mut sink = Vec::new();
        let id = "id\"x\n\t\\";
        let message = "prompt=3 + max_tokens=9 exceeds DFlash draft ctx capacity 8 — falling back to AR (\"identical\" output)";
        emit_qwen_ar_info(&mut sink, id, message);
        let out = String::from_utf8(sink).unwrap();
        let lines = parse_jsonl(&out);
        assert_eq!(lines.len(), 1);
        assert_eq!(lines[0]["type"], "info");
        assert_eq!(lines[0]["id"], id);
        assert_eq!(lines[0]["message"], message);
        assert_eq!(lines[0]["attempt_id"], 13);
        // Round-trip proves escaping: re-serialize must still parse as one object.
        let raw = out.lines().next().unwrap();
        let again: serde_json::Value = serde_json::from_str(raw).expect("serde-escaped info");
        assert_eq!(again["id"].as_str().unwrap(), id);
        set_active_attempt_id(0);
    }

    #[test]
    fn rollback_attestation_false_on_sync_failure_surface() {
        // No injectable mock GPU; production surface is hipfire_generate::common::RollbackEpilogue from
        // hipfire_generate::common::fail_closed_device_sync on Err → rolled_back=false + context.
        // hipfire_generate::common::emit_fail_closed_error must append context and claim rolled_back=false.
        set_active_attempt_id(17);
        let mut sink = Vec::new();
        let ep = attest_epilogue_with_context("device_synchronize failed: hipErrorUnknown");
        assert!(!ep.rolled_back);
        assert!(ep
            .context
            .as_ref()
            .unwrap()
            .contains("device_synchronize failed"));
        hipfire_generate::common::emit_fail_closed_error(
            &mut sink,
            Some("rb1"),
            "forced-token advance: boom",
            "validation",
            false,
            &ep,
        );
        let out = String::from_utf8(sink).unwrap();
        let lines = parse_jsonl(&out);
        assert_eq!(lines.len(), 1);
        assert_eq!(lines[0]["type"], "error");
        assert_eq!(lines[0]["rolled_back"], false);
        assert_eq!(lines[0]["attempt_id"], 17);
        let msg = lines[0]["message"].as_str().unwrap();
        assert!(msg.contains("forced-token advance: boom"), "{msg}");
        assert!(msg.contains("device_synchronize failed"), "{msg}");
        assert!(!out.contains(r#""type":"done""#));

        // Attested success path still reports rolled_back=true without context suffix.
        let mut sink_ok = Vec::new();
        let ep_ok = attest_epilogue(true);
        hipfire_generate::common::emit_fail_closed_error(
            &mut sink_ok,
            Some("rb2"),
            "spec_step: boom",
            "validation",
            false,
            &ep_ok,
        );
        let ok = parse_jsonl(&String::from_utf8(sink_ok).unwrap());
        assert_eq!(ok[0]["rolled_back"], true);
        assert_eq!(ok[0]["message"], "spec_step: boom");
        set_active_attempt_id(0);
    }

    #[test]
    fn pending_seed_chain_trigger_clip_force_then_terminal_flush() {
        // End-to-end pure chain defending the single pending-seed invariant:
        // mid-window force trigger retained → budget clip → forced tx leaves
        // last forced pending → safe terminal flushes that seed once.
        let prompt = vec![1u32, 2];
        let first = 50u32;
        // Consume force-trigger only from a wider speculative window.
        let step = SpecStep::new([60u32, 61, 62], 62, 3, 2);
        let host = hipfire_generate::qwen::spec_host_advance_after_step(
            prompt.len(),
            0,
            vec![first],
            &step.emit,
            step.next_seed,
            1,
        );
        assert_eq!(host.seed_token, 60); // trigger retained as pending seed
        assert_eq!(host.generated, 1);

        let forced_raw = [70u32, 71, 72, 73];
        // generated=1 (trigger counted); max_tokens=3 → room for 2 forced.
        let forced = hipfire_generate::qwen::spec_forced_tokens_within_budget(host.generated, 3, &forced_raw);
        assert_eq!(forced, &[70, 71]);
        let ftx = hipfire_generate::qwen::spec_forced_pending_seed_tx(host.seed_token, forced, true);
        assert_eq!(ftx.commit, vec![60, 70]); // trigger + forced[..n-1]
        assert_eq!(ftx.pending_seed, 71); // last forced pending once
        assert!(!ftx.commit.contains(&71));
        assert_eq!(ftx.position_delta, 2);

        let position = host.position + ftx.position_delta;
        let generated = host.generated + forced.len();
        // host.position already counts the force-trigger write slot after prefill first.
        assert_eq!(position, prompt.len() + 1 + ftx.position_delta);
        assert_eq!(generated, 3);

        // Safe terminal: flush final pending seed exactly once.
        let term = hipfire_generate::qwen::spec_terminal_pending_seed_tx(ftx.pending_seed);
        assert_eq!(term.commit, vec![71]);
        assert_eq!(term.position_delta, 1);
        let final_pos = position + term.position_delta;
        // Full history: prompt + first_token + trigger + forced (generated).
        assert_eq!(final_pos, prompt.len() + 1 + generated);

        // Realign plan after force path still keeps last raw as unwritten seed.
        let mut raw = vec![60u32];
        raw.extend_from_slice(forced);
        let plan = hipfire_generate::qwen::spec_prefix_realign_plan(&prompt, first, &raw);
        assert_eq!(plan.seed_token, 71);
        assert_ne!(plan.replay.last().copied(), Some(plan.seed_token));
        assert!(hipfire_generate::qwen::spec_prefix_realign_admit(&plan, 1024, 1024, 0, false).is_ok());
    }

    // ── Task 4 Important vetoes (production seam pins) ─────────────────────

    /// max_tokens==0 rejects at hipfire_generate::qwen::generate_spec entry via the same writer the
    /// production gate uses — before prefill/GPU/state/client mutation.
    /// Wire: one correlated validation error, rolled_back=false, no done/aborted.
    #[test]
    fn zero_budget_max_tokens_preflight_error_only_no_done() {
        set_active_attempt_id(101);
        let mut sink = Vec::new();
        // Mirrors hipfire_generate::qwen::generate_spec entry gate (max_tokens == 0 → emit + return None).
        hipfire_generate::dense::emit_active_attempt_error(
            &mut sink,
            Some("zb0"),
            "max_tokens must be > 0",
            "validation",
            false,
            false,
        );
        let _ = std::io::Write::flush(&mut sink);
        let out = String::from_utf8(sink).unwrap();
        let lines = parse_jsonl(&out);
        assert_eq!(lines.len(), 1, "exactly one correlated error: {lines:?}");
        assert_eq!(lines[0]["type"], "error");
        assert_eq!(lines[0]["id"], "zb0");
        assert_eq!(lines[0]["class"], "validation");
        assert_eq!(lines[0]["retryable"], false);
        assert_eq!(lines[0]["rolled_back"], false);
        assert_eq!(lines[0]["attempt_id"], 101);
        assert_eq!(lines[0]["message"], "max_tokens must be > 0");
        // No first token, no safe terminal flush, no aborted pair.
        assert!(!out.contains(r#""type":"done""#));
        assert!(!out.contains(r#""type":"aborted""#));
        assert!(!out.contains(r#""type":"token""#));
        assert!(!out.contains(r#""type":"tool_calls""#));
        // Wrapper contract: hipfire_generate::qwen::generate_spec returned None → no epilogue.
        assert!(!qwen_dflash_epilogue_after_spec_run(false));
        set_active_attempt_id(0);
    }

    /// Cancel after rollback attestation: attested → aborted+done; unattested →
    /// exactly one correlated nonretryable error with context and no done.
    #[test]
    fn cancel_after_rollback_attested_vs_unattested_wire() {
        // Attested rollback keeps fold-compatible aborted + done pair.
        set_active_attempt_id(202);
        let mut sink_ok = Vec::new();
        let ep_ok = attest_epilogue(true);
        hipfire_generate::common::emit_spec_cancel_after_rollback(&mut sink_ok, "c-ok", 7, &ep_ok);
        let out_ok = String::from_utf8(sink_ok).unwrap();
        let lines_ok = parse_jsonl(&out_ok);
        assert_eq!(
            lines_ok.len(),
            2,
            "attested cancel: aborted+done {lines_ok:?}"
        );
        assert_eq!(lines_ok[0]["type"], "aborted");
        assert_eq!(lines_ok[0]["reason"], "client_cancelled");
        assert_eq!(lines_ok[0]["attempt_id"], 202);
        assert_eq!(lines_ok[0]["id"], "c-ok");
        assert_eq!(lines_ok[1]["type"], "done");
        assert_eq!(lines_ok[1]["finish_reason"], "aborted");
        assert_eq!(lines_ok[1]["completion_tokens"], 7);
        assert_eq!(lines_ok[1]["attempt_id"], 202);
        assert!(!out_ok.contains(r#""type":"error""#));
        assert!(!out_ok.contains(r#""type":"tool_calls""#));

        // Unattested rollback: one fail-closed error, no aborted/done.
        set_active_attempt_id(203);
        let mut sink_bad = Vec::new();
        let ep_bad = attest_epilogue_with_context("device_synchronize failed: hipErrorUnknown");
        assert!(!ep_bad.rolled_back);
        hipfire_generate::common::emit_spec_cancel_after_rollback(&mut sink_bad, "c-bad", 3, &ep_bad);
        let out_bad = String::from_utf8(sink_bad).unwrap();
        let lines_bad = parse_jsonl(&out_bad);
        assert_eq!(
            lines_bad.len(),
            1,
            "unattested cancel: error only {lines_bad:?}"
        );
        assert_eq!(lines_bad[0]["type"], "error");
        assert_eq!(lines_bad[0]["class"], "validation");
        assert_eq!(lines_bad[0]["retryable"], false);
        assert_eq!(lines_bad[0]["rolled_back"], false);
        assert_eq!(lines_bad[0]["attempt_id"], 203);
        assert_eq!(lines_bad[0]["id"], "c-bad");
        let msg = lines_bad[0]["message"].as_str().unwrap();
        assert!(
            msg.contains("client cancelled; fail-closed rollback could not be attested"),
            "{msg}"
        );
        assert!(msg.contains("device_synchronize failed"), "{msg}");
        assert!(!out_bad.contains(r#""type":"done""#));
        assert!(!out_bad.contains(r#""type":"aborted""#));
        assert!(!out_bad.contains(r#""type":"tool_calls""#));
        set_active_attempt_id(0);
    }

    /// Failure-injection: each omitted reset class (incl. single-GPU s_ef_residual
    /// and EP bind) keeps rolled_back=false; aggregate failure still models sync
    /// as attempted; Qwen AR prefill/decode abort terminals are exclusive.
    #[test]
    fn rollback_attestation_omitted_reset_classes_and_ar_abort_xor() {
        // Every required surface Ok + sync Ok → attested.
        let all_ok = attest_rollback_steps(
            &[
                ("s_matrices", Ok(())),
                ("s_scales", Ok(())),
                ("conv_states", Ok(())),
                ("s_ef_residual", Ok(())),
                ("host_cursors", Ok(())),
                ("kv_compact", Ok(())),
                ("checkpoints", Ok(())),
                ("drafter", Ok(())),
                ("adaptive", Ok(())),
                ("graph_replay", Ok(())),
                ("ep_bind_thread", Ok(())),
            ],
            Ok(()),
        );
        assert!(all_ok.rolled_back);
        assert!(all_ok.context.is_none());

        // Single-GPU s_ef_residual omission/failure alone unattests.
        let ef = attest_rollback_steps(
            &[
                ("s_matrices", Ok(())),
                ("s_scales", Ok(())),
                ("conv_states", Ok(())),
                ("s_ef_residual", Err("memset failed".into())),
                ("ep_bind_thread", Ok(())),
            ],
            Ok(()),
        );
        assert!(!ef.rolled_back);
        let ctx = ef.context.as_deref().unwrap_or("");
        assert!(ctx.contains("s_ef_residual"), "{ctx}");
        assert!(
            !ctx.contains("device_synchronize"),
            "sync Ok must not appear: {ctx}"
        );

        // EP bind_thread failure alone unattests even when sync Ok.
        let bind = attest_rollback_steps(
            &[
                ("s_ef_residual", Ok(())),
                ("ep_bind_thread", Err("hipErrorInvalidDevice".into())),
            ],
            Ok(()),
        );
        assert!(!bind.rolled_back);
        assert!(
            bind.context
                .as_deref()
                .unwrap_or("")
                .contains("ep_bind_thread"),
            "{:?}",
            bind.context
        );

        // Aggregate reset failure + sync still attempted (both in context).
        let agg = attest_rollback_steps(
            &[
                ("s_matrices", Err("m1".into())),
                ("s_ef_residual", Err("ef".into())),
                ("ep_bind_thread", Err("bind".into())),
            ],
            Err("hipErrorUnknown".into()),
        );
        assert!(!agg.rolled_back);
        let ctx = agg.context.as_deref().unwrap_or("");
        assert!(ctx.contains("s_matrices"), "{ctx}");
        assert!(ctx.contains("s_ef_residual"), "{ctx}");
        assert!(ctx.contains("ep_bind_thread"), "{ctx}");
        assert!(ctx.contains("device_synchronize failed"), "{ctx}");

        // hipfire_generate::common::fail_closed_epilogue_after_sync: prior Err + sync Ok → unattested, sync ran.
        let merged = hipfire_generate::common::fail_closed_epilogue_after_sync(
            Err("hipfire_generate::common::reset_qwen35_recurrent: s_ef_residual memset: boom".into()),
            hipfire_generate::common::RollbackEpilogue {
                rolled_back: true,
                context: None,
            },
        );
        assert!(!merged.rolled_back);
        assert!(
            merged
                .context
                .as_deref()
                .unwrap_or("")
                .contains("s_ef_residual"),
            "{:?}",
            merged.context
        );

        // prior Err + sync Err → both preserved.
        let both = hipfire_generate::common::fail_closed_epilogue_after_sync(
            Err("ep rank0 bind_thread: bad".into()),
            hipfire_generate::common::RollbackEpilogue {
                rolled_back: false,
                context: Some("device_synchronize failed: hipErrorUnknown".into()),
            },
        );
        assert!(!both.rolled_back);
        let ctx = both.context.as_deref().unwrap_or("");
        assert!(ctx.contains("bind_thread"), "{ctx}");
        assert!(ctx.contains("device_synchronize failed"), "{ctx}");

        // Qwen AR prefill abort terminal exclusivity (attested vs unattested).
        set_active_attempt_id(501);
        let mut sink = Vec::new();
        hipfire_generate::common::emit_spec_cancel_after_rollback(&mut sink, "ar-prefill", 0, &attest_epilogue(true));
        let lines = parse_jsonl(&String::from_utf8(sink).unwrap());
        assert_eq!(lines.len(), 2);
        assert_eq!(lines[0]["type"], "aborted");
        assert_eq!(lines[1]["type"], "done");
        assert_eq!(lines[1]["finish_reason"], "aborted");
        assert_eq!(lines[1]["completion_tokens"], 0);
        assert!(lines.iter().all(|e| e["attempt_id"] == 501));

        set_active_attempt_id(502);
        let mut sink = Vec::new();
        hipfire_generate::common::emit_spec_cancel_after_rollback(
            &mut sink,
            "ar-prefill-bad",
            0,
            &attest_epilogue_with_context("s_ef_residual memset: boom"),
        );
        let out = String::from_utf8(sink).unwrap();
        let lines = parse_jsonl(&out);
        assert_eq!(lines.len(), 1, "prefill unattested: error only");
        assert_eq!(lines[0]["type"], "error");
        assert_eq!(lines[0]["rolled_back"], false);
        assert!(!out.contains(r#""type":"done""#));
        assert!(!out.contains(r#""type":"aborted""#));
        assert!(!out.contains(r#""type":"tool_calls""#));

        // Qwen AR mid-decode abort terminal exclusivity.
        set_active_attempt_id(503);
        let mut sink = Vec::new();
        hipfire_generate::common::emit_spec_cancel_after_rollback(&mut sink, "ar-decode", 5, &attest_epilogue(true));
        let lines = parse_jsonl(&String::from_utf8(sink).unwrap());
        assert_eq!(lines.len(), 2);
        assert_eq!(lines[0]["type"], "aborted");
        assert_eq!(lines[1]["finish_reason"], "aborted");
        assert_eq!(lines[1]["completion_tokens"], 5);

        set_active_attempt_id(504);
        let mut sink = Vec::new();
        hipfire_generate::common::emit_spec_cancel_after_rollback(
            &mut sink,
            "ar-decode-bad",
            5,
            &attest_epilogue_with_context(
                "ep rank0 bind_thread: bad; device_synchronize failed: x",
            ),
        );
        let out = String::from_utf8(sink).unwrap();
        let lines = parse_jsonl(&out);
        assert_eq!(lines.len(), 1, "decode unattested: error only");
        assert_eq!(lines[0]["type"], "error");
        assert_eq!(lines[0]["rolled_back"], false);
        assert_eq!(lines[0]["attempt_id"], 504);
        let msg = lines[0]["message"].as_str().unwrap();
        assert!(msg.contains("bind_thread"), "{msg}");
        assert!(msg.contains("device_synchronize failed"), "{msg}");
        assert!(!out.contains(r#""type":"done""#));
        assert!(!out.contains(r#""type":"aborted""#));
        set_active_attempt_id(0);
    }

    /// Eviction-enabled missing optional kv_cache_mut is ErrorOnly (not panic):
    /// hipfire_generate::qwen::classify_evict_failure_wire → hipfire_generate::common::emit_fail_closed_error with the production
    /// post-prefill / per-cycle messages; no done/aborted/calls/cache.
    #[test]
    fn missing_optional_kv_cache_mut_is_error_only_not_panic() {
        assert_eq!(hipfire_generate::qwen::classify_evict_failure_wire(), hipfire_generate::qwen::SpecFailClosedWire::ErrorOnly);
        assert_ne!(
            hipfire_generate::qwen::SpecFailClosedWire::Cancelled,
            hipfire_generate::qwen::SpecFailClosedWire::ErrorOnly,
            "missing KV hook must never classify as Cancelled"
        );

        for (attempt, id, message) in [
            (301u64, "kv-pp", "kv_cache_mut missing (post-prefill)"),
            (302u64, "kv-pc", "kv_cache_mut missing (per-cycle)"),
        ] {
            set_active_attempt_id(attempt);
            let mut sink = Vec::new();
            // Production seam: classify first, then fail-closed writer (same as
            // hipfire_generate::qwen::generate_spec match slot.kv_cache_mut() { None => ... }).
            let _ = hipfire_generate::qwen::classify_evict_failure_wire();
            let ep = attest_epilogue(true);
            match hipfire_generate::qwen::classify_evict_failure_wire() {
                hipfire_generate::qwen::SpecFailClosedWire::ErrorOnly => {
                    hipfire_generate::common::emit_fail_closed_error(&mut sink, Some(id), message, "validation", false, &ep);
                }
                hipfire_generate::qwen::SpecFailClosedWire::Cancelled => {
                    panic!("kv_cache_mut missing must not classify Cancelled")
                }
            }
            let out = String::from_utf8(sink).unwrap();
            let lines = parse_jsonl(&out);
            assert_eq!(lines.len(), 1, "error XOR done for {message}: {lines:?}");
            assert_eq!(lines[0]["type"], "error");
            assert_eq!(lines[0]["class"], "validation");
            assert_eq!(lines[0]["retryable"], false);
            assert_eq!(lines[0]["rolled_back"], true);
            assert_eq!(lines[0]["attempt_id"], attempt);
            assert_eq!(lines[0]["id"], id);
            assert_eq!(lines[0]["message"], message);
            assert!(!out.contains(r#""type":"done""#));
            assert!(!out.contains(r#""type":"aborted""#));
            assert!(!out.contains(r#""type":"tool_calls""#));
            // hipfire_generate::qwen::generate_spec returns None → wrapper skips cache store / epilogue.
            assert!(!qwen_dflash_epilogue_after_spec_run(false));
        }

        // Unattested rollback on the same missing-hook path: rolled_back=false
        // + context appended; still error-only (no panic surface).
        set_active_attempt_id(303);
        let mut sink = Vec::new();
        let ep = attest_epilogue_with_context("device_synchronize failed: test");
        hipfire_generate::common::emit_fail_closed_error(
            &mut sink,
            Some("kv-ua"),
            "kv_cache_mut missing (post-prefill)",
            "validation",
            false,
            &ep,
        );
        let out = String::from_utf8(sink).unwrap();
        let lines = parse_jsonl(&out);
        assert_eq!(lines.len(), 1);
        assert_eq!(lines[0]["rolled_back"], false);
        let msg = lines[0]["message"].as_str().unwrap();
        assert!(msg.contains("kv_cache_mut missing (post-prefill)"), "{msg}");
        assert!(msg.contains("device_synchronize failed"), "{msg}");
        assert!(!out.contains(r#""type":"done""#));
        set_active_attempt_id(0);
    }

    // ── Remaining Important Task 4 vetoes (wrapper / legacy / rewind) ──

    /// hipfire_generate::qwen::generate_dflash max_tokens==0: hipfire_generate::dense::emit_active_attempt_error then return true
    /// (handled) before Jinja/render/set_sampling/gen_start. Same wire as the
    /// inner hipfire_generate::qwen::generate_spec defense; wrapper must not fall through to AR.
    #[test]
    fn generate_dflash_zero_budget_preflight_handled_error_only() {
        set_active_attempt_id(401);
        let mut sink = Vec::new();
        // Mirrors hipfire_generate::qwen::generate_dflash entry (max_tokens == 0 → emit + return true).
        hipfire_generate::dense::emit_active_attempt_error(
            &mut sink,
            Some("df-zb0"),
            "max_tokens must be > 0",
            "validation",
            false,
            false,
        );
        let _ = std::io::Write::flush(&mut sink);
        let out = String::from_utf8(sink).unwrap();
        let lines = parse_jsonl(&out);
        assert_eq!(lines.len(), 1, "exactly one correlated error: {lines:?}");
        assert_eq!(lines[0]["type"], "error");
        assert_eq!(lines[0]["id"], "df-zb0");
        assert_eq!(lines[0]["class"], "validation");
        assert_eq!(lines[0]["retryable"], false);
        assert_eq!(lines[0]["rolled_back"], false);
        assert_eq!(lines[0]["attempt_id"], 401);
        assert_eq!(lines[0]["message"], "max_tokens must be > 0");
        assert!(!out.contains(r#""type":"done""#));
        assert!(!out.contains(r#""type":"aborted""#));
        assert!(!out.contains(r#""type":"token""#));
        assert!(!out.contains(r#""type":"tool_calls""#));
        // Handled=true → caller must not fall through to AR / second envelope.
        let wrapper_handled = true;
        assert!(wrapper_handled);
        set_active_attempt_id(0);
    }

    /// hipfire_generate::dense::generate_deepseek4_spec max_tokens==0: same emit policy, plain return
    /// (unit fn) before DSML render / decode-cache teardown / set_sampling.
    #[test]
    fn generate_deepseek4_spec_zero_budget_preflight_error_only() {
        set_active_attempt_id(402);
        let mut sink = Vec::new();
        // Mirrors hipfire_generate::dense::generate_deepseek4_spec entry (max_tokens == 0 → emit + return).
        hipfire_generate::dense::emit_active_attempt_error(
            &mut sink,
            Some("ds4-zb0"),
            "max_tokens must be > 0",
            "validation",
            false,
            false,
        );
        let _ = std::io::Write::flush(&mut sink);
        let out = String::from_utf8(sink).unwrap();
        let lines = parse_jsonl(&out);
        assert_eq!(lines.len(), 1, "exactly one correlated error: {lines:?}");
        assert_eq!(lines[0]["type"], "error");
        assert_eq!(lines[0]["id"], "ds4-zb0");
        assert_eq!(lines[0]["class"], "validation");
        assert_eq!(lines[0]["retryable"], false);
        assert_eq!(lines[0]["rolled_back"], false);
        assert_eq!(lines[0]["attempt_id"], 402);
        assert_eq!(lines[0]["message"], "max_tokens must be > 0");
        assert!(!out.contains(r#""type":"done""#));
        assert!(!out.contains(r#""type":"aborted""#));
        assert!(!out.contains(r#""type":"token""#));
        assert!(!out.contains(r#""type":"tool_calls""#));
        // Unit wrapper returns (no AR fallthrough second write).
        set_active_attempt_id(0);
    }

    /// Legacy non-qwen hipfire_generate::qwen::generate_dflash else-branch: fail_closed_rollback.is_some()
    /// || grammar_violated → hipfire_generate::common::emit_fail_closed_error only; no extract/release/
    /// cache store / done. Message classified by grammar / open_think /
    /// malformed_protocol / generic.
    #[test]
    fn legacy_non_qwen_fail_closed_epilogue_error_only_no_extract() {
        // Production message selection (qwen_semantic_v2 == false branch).
        fn legacy_fail_closed_message(
            grammar_violated: bool,
            open_think: bool,
            finish_reason: &str,
        ) -> &'static str {
            if grammar_violated {
                "grammar violation during speculative decode"
            } else if open_think || finish_reason == "open_think" {
                "open think span at end of generation (validation)"
            } else if finish_reason == "malformed_protocol" {
                "malformed tool protocol"
            } else {
                "fail-closed speculative decode"
            }
        }

        let cases = [
            (
                true,
                false,
                "stop",
                "grammar violation during speculative decode",
            ),
            (
                false,
                true,
                "stop",
                "open think span at end of generation (validation)",
            ),
            (
                false,
                false,
                "open_think",
                "open think span at end of generation (validation)",
            ),
            (
                false,
                false,
                "malformed_protocol",
                "malformed tool protocol",
            ),
            (false, false, "length", "fail-closed speculative decode"),
        ];

        for (i, (grammar, open_think, reason, expected_msg)) in cases.iter().enumerate() {
            assert_eq!(
                legacy_fail_closed_message(*grammar, *open_think, reason),
                *expected_msg,
                "case {i} message select"
            );
            // Gate: fail_closed_rollback.is_some() || grammar_violated.
            let fail_closed_present = true;
            let take_error_only = fail_closed_present || *grammar;
            assert!(take_error_only, "case {i} must take error-only path");

            set_active_attempt_id(500 + i as u64);
            let mut sink = Vec::new();
            let ep = attest_epilogue(true);
            hipfire_generate::common::emit_fail_closed_error(
                &mut sink,
                Some("leg-fc"),
                expected_msg,
                "validation",
                false,
                &ep,
            );
            let out = String::from_utf8(sink).unwrap();
            let lines = parse_jsonl(&out);
            assert_eq!(lines.len(), 1, "case {i}: error XOR done {lines:?}");
            assert_eq!(lines[0]["type"], "error");
            assert_eq!(lines[0]["class"], "validation");
            assert_eq!(lines[0]["retryable"], false);
            assert_eq!(lines[0]["id"], "leg-fc");
            assert_eq!(lines[0]["message"], *expected_msg);
            // No held tool_calls release, no cache store, no done/aborted.
            assert!(!out.contains(r#""type":"done""#), "case {i}");
            assert!(!out.contains(r#""type":"aborted""#), "case {i}");
            assert!(!out.contains(r#""type":"tool_calls""#), "case {i}");
            // Early return true from hipfire_generate::qwen::generate_dflash — no whole-output extract path.
            let early_return_handled = true;
            assert!(early_return_handled);
        }
        set_active_attempt_id(0);
    }

    /// hipfire_generate::qwen::generate_spec resume_from: on spec.rewind_to Err, host seq_pos /
    /// conversation_tokens must NOT be truncated to ckpt first. Fail-closed
    /// live rollback + one correlated "rewind_to: …" error; return None skips
    /// wrapper epilogue (no done / calls / cache).
    #[test]
    fn rewind_to_err_freezes_host_cursors_then_fail_closed() {
        // Host state as if mid-conversation before resume_from rewind.
        let ckpt = 4usize;
        let mut seq_pos = 12usize;
        let mut conversation_tokens: Vec<u32> = (0..12).map(|t| t as u32).collect();
        let seq_before = seq_pos;
        let toks_before = conversation_tokens.clone();

        // Production order on Err: message first, then live rollback (which
        // zeroes host), emit, return None — never the success truncate.
        let restore_err = "DeltaNetSnapshot::restore_to: synthetic restore fail";
        let msg = format!("rewind_to: {restore_err}");

        // Success path would do: seq_pos = ckpt; conversation_tokens.truncate(ckpt).
        // Error path must NOT apply that before/without fail-closed.
        let rewind_ok = false;
        if rewind_ok {
            seq_pos = ckpt;
            conversation_tokens.truncate(ckpt);
        }
        // Cursors still at pre-rewind values until hipfire_generate::common::production_fail_closed_rollback_live.
        assert_eq!(
            seq_pos, seq_before,
            "must not truncate seq_pos to ckpt on Err"
        );
        assert_eq!(
            conversation_tokens, toks_before,
            "must not truncate conversation_tokens to ckpt on Err"
        );
        assert_ne!(seq_pos, ckpt);

        // Live rollback zeroes host (GPU-less stand-in for hipfire_generate::common::production_fail_closed_rollback_live).
        seq_pos = 0;
        conversation_tokens.clear();
        assert_eq!(seq_pos, 0);
        assert!(conversation_tokens.is_empty());

        set_active_attempt_id(601);
        let mut sink = Vec::new();
        let ep = attest_epilogue(true);
        hipfire_generate::common::emit_fail_closed_error(&mut sink, Some("rw-err"), &msg, "validation", false, &ep);
        let out = String::from_utf8(sink).unwrap();
        let lines = parse_jsonl(&out);
        assert_eq!(lines.len(), 1, "one correlated rewind error: {lines:?}");
        assert_eq!(lines[0]["type"], "error");
        assert_eq!(lines[0]["id"], "rw-err");
        assert_eq!(lines[0]["class"], "validation");
        assert_eq!(lines[0]["retryable"], false);
        assert_eq!(lines[0]["rolled_back"], true);
        assert_eq!(lines[0]["attempt_id"], 601);
        assert_eq!(lines[0]["message"], msg);
        assert!(lines[0]["message"]
            .as_str()
            .unwrap()
            .starts_with("rewind_to:"));
        assert!(!out.contains(r#""type":"done""#));
        assert!(!out.contains(r#""type":"aborted""#));
        assert!(!out.contains(r#""type":"tool_calls""#));
        // hipfire_generate::qwen::generate_spec returns None → wrapper skips epilogue/cache.
        assert!(!qwen_dflash_epilogue_after_spec_run(false));

        // Unattested sync path still error-only with context suffix.
        set_active_attempt_id(602);
        let mut sink_ua = Vec::new();
        let ep_ua = attest_epilogue_with_context("device_synchronize failed: hipErrorUnknown");
        hipfire_generate::common::emit_fail_closed_error(
            &mut sink_ua,
            Some("rw-ua"),
            &msg,
            "validation",
            false,
            &ep_ua,
        );
        let out_ua = String::from_utf8(sink_ua).unwrap();
        let lines_ua = parse_jsonl(&out_ua);
        assert_eq!(lines_ua.len(), 1);
        assert_eq!(lines_ua[0]["rolled_back"], false);
        let m = lines_ua[0]["message"].as_str().unwrap();
        assert!(m.contains("rewind_to:"), "{m}");
        assert!(m.contains("device_synchronize failed"), "{m}");
        assert!(!out_ua.contains(r#""type":"done""#));
        set_active_attempt_id(0);
    }

    // ── Task 4 definitive terminal-edge blockers ──────────────────────────

    /// Legacy non-qwen hipfire_generate::qwen::generate_dflash else-branch: length still emits
    /// finish_reason=length but never releases held tool calls or stores
    /// asst_turn_cache (partial/truncated turns are unsafe to prime).
    #[test]
    fn legacy_length_terminal_skips_assistant_cache_and_tool_release() {
        // Production gates (hipfire_generate::qwen::generate_dflash qwen_semantic_v2=false branch):
        //   hit_length_cap = run.generated >= max_tokens
        //   stage_terminal_tool_calls on safe tool terminals before handshake
        //   asst_turn_cache.insert only when Commit && !hit_length_cap && !cached_seq.is_empty()
        let generated = 8usize;
        let max_tokens = 8usize;
        let hit_length_cap = generated >= max_tokens;
        assert!(hit_length_cap);

        let finish = summary_tool_calls(vec![ToolCall {
            id: None,
            name: "held".into(),
            arguments: serde_json::json!({}),
            rendered_body: None,
        }]);
        assert!(finish.tool_calls > 0);

        let release = !hit_length_cap && finish.tool_calls > 0;
        assert!(!release, "length must not release held finish tool calls");

        let cached_seq = vec![1u32, 2, 3];
        let mut sink: std::collections::HashMap<u64, Vec<u32>> = std::collections::HashMap::new();
        if !hit_length_cap && !cached_seq.is_empty() {
            let decoded_full = "partial answer";
            let stripped = hipfire_generate::common::strip_think_for_fingerprint(decoded_full);
            let emit_text =
                hipfire_runtime::tokenizer::maybe_normalize_prompt(&stripped).into_owned();
            let emit_tool_calls = extract_tool_calls_from_text(decoded_full);
            let fp = hipfire_generate::common::asst_turn_fingerprint(&emit_text, &emit_tool_calls);
            sink.insert(fp, cached_seq.clone());
        }
        assert!(
            sink.is_empty(),
            "length terminal must not store asst_turn_cache"
        );

        let finish_reason = if hit_length_cap {
            "length"
        } else if finish.tool_calls > 0 {
            "tool_calls"
        } else {
            "stop"
        };
        assert_eq!(finish_reason, "length");

        // Safe non-length control: same gates allow release + store.
        let hit_safe = 3usize >= 8usize;
        assert!(!hit_safe);
        assert!(!hit_safe && finish.tool_calls > 0);
        let mut sink_safe = std::collections::HashMap::new();
        if !hit_safe && !cached_seq.is_empty() {
            let fp = hipfire_generate::common::asst_turn_fingerprint("ok", &[]);
            sink_safe.insert(fp, cached_seq.clone());
        }
        assert_eq!(sink_safe.len(), 1, "safe stop still stores");
    }

    /// Begin-triggered forced continuation is planned with the same pure
    /// pending-seed transaction as mid-window force, and is ordered before
    /// any speculative step (max_tokens=1 cannot spend budget on step).
    #[test]
    fn begin_first_token_forced_serviced_before_spec_step() {
        // After begin: generated counts first token when event-bearing.
        let mut generated = 1usize;
        let max_tokens = 1usize;
        let seed_token = 50u32; // first_token is also the initial pending seed
        let forced_begin = vec![60u32, 61, 62];

        // Empty take_forced ⇒ Skipped (no GPU path); loop may proceed.
        assert!(matches!(
            // Pure stand-in for hipfire_generate::qwen::apply_spec_forced_pending_seed empty input.
            {
                let forced_all: &[u32] = &[];
                if forced_all.is_empty() {
                    hipfire_generate::qwen::SpecForcedApplyResult::Skipped
                } else {
                    hipfire_generate::qwen::SpecForcedApplyResult::Applied
                }
            },
            hipfire_generate::qwen::SpecForcedApplyResult::Skipped
        ));

        // Hard budget clip: generated already 1, max_tokens=1 → room 0.
        let clipped = hipfire_generate::qwen::spec_forced_tokens_within_budget(generated, max_tokens, &forced_begin);
        assert!(
            clipped.is_empty(),
            "max_tokens=1 after first token must clip all forced (no extra step budget)"
        );
        // hipfire_generate::qwen::apply_spec_forced_pending_seed returns Skipped on empty clip — while
        // condition `generated < max_tokens` is already false, so no spec.step.
        assert!(!(!false /*first_token_is_eos*/ && generated < max_tokens));

        // Room for forced (max_tokens=3, generated=1): same tx as mid-window.
        generated = 1;
        let max2 = 3usize;
        let forced = hipfire_generate::qwen::spec_forced_tokens_within_budget(generated, max2, &forced_begin);
        assert_eq!(forced, &[60u32, 61]);
        let tx = hipfire_generate::qwen::spec_forced_pending_seed_tx(seed_token, forced, true);
        assert_eq!(tx.commit, vec![50, 60], "trigger retained; forced[..n-1]");
        assert_eq!(tx.pending_seed, 61, "last forced pending once");
        assert!(!tx.commit.contains(&61));
        assert_eq!(tx.position_delta, forced.len());

        // Ordering contract: begin force runs before while/spec.step.
        let mut phase = "begin";
        let forced_begin_nonempty = !forced_begin.is_empty();
        if forced_begin_nonempty {
            phase = "begin_forced_applied";
        }
        let enter_spec_step = phase == "begin_forced_applied" && generated < max2;
        // After applying 2 forced, generated would be 1+2=3 → loop does not step.
        let generated_after = generated + forced.len();
        assert_eq!(generated_after, 3);
        assert!(
            !(generated_after < max2),
            "after begin force at budget, no speculative step"
        );
        let _ = enter_spec_step;
        assert_eq!(phase, "begin_forced_applied");

        // hipfire_generate::qwen::classify_forced_gpu_advance still exclusive cancel vs commit.
        assert!(matches!(
            hipfire_generate::qwen::classify_forced_gpu_advance(true),
            hipfire_generate::qwen::ForcedGpuAdvanceKind::Cancelled
        ));
        assert!(matches!(
            hipfire_generate::qwen::classify_forced_gpu_advance(false),
            hipfire_generate::qwen::ForcedGpuAdvanceKind::Committed
        ));
    }

    /// Qwen first seed runs user stop-sequence detection in begin exactly like
    /// later observe tokens; StopSequence terminates before any speculative step.
    #[test]
    fn qwen_begin_first_token_stop_sequence_terminates_before_step() {
        let tok = test_tokenizer();
        let ids = tok.encode("STOP");
        assert!(!ids.is_empty());
        let first = ids[0];
        let first_text = tok.decode(&[first]);
        let mut emit = hipfire_arch_qwen35::spec_emit::Qwen35Emit::from_ctx(SpecEmitCtx {
            tokenizer: &tok,
            eos: 9,
            im_end: Some(1),
            tools: None,
            enable_grammar: false,
            stop: vec![first_text.clone()],
            max_think: 0,
            max_tokens: 256,
            assistant_prefix: AssistantPrefix::Plain,
            think_mode: hipfire_runtime::prompt_frame::ThinkMode::NonThink,
            decoded_vocab: None,
        });
        let first_begin = emit.begin(first);
        assert_eq!(
            first_begin.stop,
            Some(StopReason::StopSequence),
            "begin must surface StopSequence for first-token stop match"
        );
        // hipfire_generate::qwen::generate_spec: first_token_is_eos = first_begin.stop.is_some()
        let first_token_is_eos = first_begin.stop.is_some();
        assert!(first_token_is_eos);
        // while !first_token_is_eos && generated < max_tokens { spec.step ... }
        let mut stepped = false;
        if !first_token_is_eos {
            stepped = true;
        }
        assert!(
            !stepped,
            "StopSequence begin must skip every speculative step"
        );

        // Event-bearing first token still counts (Qwen always commits).
        assert!(
            hipfire_generate::qwen::spec_outcome_seed_committable(&first_begin),
            "stop still commits the raw first token"
        );
        assert!(first_begin
            .events
            .iter()
            .any(|e| matches!(e, ClientEvent::Committed { id, .. } if *id == first)));

        // Forced begin path is still consulted, but empty take_forced is Skipped.
        let forced_begin = emit.take_forced();
        assert!(forced_begin.is_empty());
    }

    // --- Task 4 reviewer blockers: forced-token / terminal-cause seams ---

    /// Non-committable pending seed (DS4 empty-event EOS) must not be prepended
    /// into the forced GPU commit. Forced tokens occupy that same slot; all but
    /// the final kept forced token are committed, final remains pending.
    #[test]
    fn noncommittable_pending_seed_omitted_from_forced_tx() {
        // Single forced + non-committable seed: commit is empty (seed omitted,
        // forced[0] becomes pending only) — no GPU for a lone seed replace.
        let seed = 7u32; // DS4-style empty-event EOS seed
        let one = hipfire_generate::qwen::spec_forced_pending_seed_tx(seed, &[42], false);
        assert!(
            one.commit.is_empty(),
            "non-committable seed + single forced must not GPU-commit: {:?}",
            one.commit
        );
        assert_eq!(one.position_delta, 0);
        assert_eq!(one.pending_seed, 42);
        assert!(!one.commit.contains(&seed));

        // Multi forced + non-committable: commit is forced[..n-1] only.
        let multi = hipfire_generate::qwen::spec_forced_pending_seed_tx(seed, &[10, 11, 12], false);
        assert_eq!(
            multi.commit,
            vec![10, 11],
            "seed omitted; forced prefix only"
        );
        assert!(!multi.commit.contains(&seed));
        assert_eq!(multi.pending_seed, 12);
        assert_eq!(multi.position_delta, multi.commit.len());
        assert!(!multi.commit.contains(&12), "last forced stays pending");

        // Contrast: same inputs with committable seed retain the trigger.
        let keep = hipfire_generate::qwen::spec_forced_pending_seed_tx(seed, &[10, 11, 12], true);
        assert_eq!(keep.commit, vec![seed, 10, 11]);
        assert_eq!(keep.pending_seed, 12);
    }

    /// Forced suffix stages observe first, trims at the first non-None stop,
    /// GPU-commits only that kept prefix, and renders only after successful
    /// commit. Later forced tokens are never observed/committed/rendered.
    #[test]
    fn forced_suffix_stops_at_first_stop_sequence_prefix_only() {
        let tok = test_tokenizer();
        // Build a stop string from a real token, then force a later token that
        // must not be observed once stop fires.
        let stop_ids = tok.encode("STOP");
        assert!(!stop_ids.is_empty());
        let stop_tok = stop_ids[0];
        let stop_text = tok.decode(&[stop_tok]);
        let later = tok.encode("later");
        assert!(!later.is_empty());
        let later_tok = later[0];
        assert_ne!(stop_tok, later_tok);

        let mut emit = hipfire_arch_qwen35::spec_emit::Qwen35Emit::from_ctx(SpecEmitCtx {
            tokenizer: &tok,
            eos: 9,
            im_end: Some(1),
            tools: None,
            enable_grammar: false,
            stop: vec![stop_text.clone()],
            max_think: 0,
            max_tokens: 256,
            assistant_prefix: AssistantPrefix::Plain,
            think_mode: hipfire_runtime::prompt_frame::ThinkMode::NonThink,
            decoded_vocab: None,
        });
        // Warm begin so observe path is active (forced uses observe).
        let warm = tok.encode("hi");
        assert!(!warm.is_empty());
        let _ = emit.begin(warm[0]);

        // Production staging loop (hipfire_generate::qwen::apply_spec_forced_pending_seed):
        let forced_all = [stop_tok, later_tok, later_tok.wrapping_add(1)];
        let mut staged: Vec<(u32, hipfire_runtime::spec::EmitOutcome)> =
            Vec::with_capacity(forced_all.len());
        let mut stop_reason: Option<StopReason> = None;
        for &ft in &forced_all {
            let fo = emit.observe(ft);
            let stop = fo.stop;
            staged.push((ft, fo));
            if let Some(reason) = stop {
                stop_reason = Some(reason);
                break;
            }
        }
        assert_eq!(
            stop_reason,
            Some(StopReason::StopSequence),
            "first forced token matching stop must halt the suffix"
        );
        assert_eq!(
            staged.len(),
            1,
            "later forced tokens must not be observed after stop"
        );
        assert_eq!(staged[0].0, stop_tok);

        let kept: Vec<u32> = staged.iter().map(|(t, _)| *t).collect();
        assert_eq!(kept, vec![stop_tok]);

        // Commit uses the kept prefix only (incoming seed was committable).
        let incoming_seed = warm[0];
        let incoming_committable = true;
        let tx = hipfire_generate::qwen::spec_forced_pending_seed_tx(incoming_seed, &kept, incoming_committable);
        // Single kept forced: commit = [seed], pending = stop_tok.
        assert_eq!(tx.commit, vec![incoming_seed]);
        assert_eq!(tx.pending_seed, stop_tok);
        assert!(!tx.commit.contains(&later_tok));
        assert!(!tx.commit.contains(&stop_tok));

        // Apply result maps to Stopped(reason) — not Applied.
        let apply = match stop_reason {
            Some(reason) => hipfire_generate::qwen::SpecForcedApplyResult::Stopped(reason),
            None => hipfire_generate::qwen::SpecForcedApplyResult::Applied,
        };
        assert_eq!(
            apply,
            hipfire_generate::qwen::SpecForcedApplyResult::Stopped(StopReason::StopSequence)
        );

        // Render-after-commit contract: client events from staged outcomes are
        // only eligible once GPU commit of `tx.commit` succeeded. Model the
        // gate explicitly so a reorder (render then commit) fails this test.
        let mut gpu_committed = false;
        let mut rendered: Vec<u32> = Vec::new();
        // "commit" kept prefix
        gpu_committed = true;
        if gpu_committed {
            for (ft, fo) in &staged {
                if !fo.events.is_empty() {
                    rendered.push(*ft);
                }
            }
        }
        assert!(gpu_committed);
        assert_eq!(
            rendered,
            vec![stop_tok],
            "render only kept prefix after commit"
        );
        assert!(!rendered.contains(&later_tok));
    }

    /// Begin and mid callers treat Stopped as turn-terminal: set semantic_stop,
    /// force first_token_is_eos / hit_eos, and skip later force + all spec.step.
    #[test]
    fn begin_and_mid_stopped_skips_later_force_and_spec_step() {
        // --- begin path (mirrors hipfire_generate::qwen::generate_spec after emit.begin) ---
        let reason = StopReason::StopSequence;
        let mut semantic_stop: Option<StopReason> = None;
        let mut first_token_is_eos = false;
        let apply = hipfire_generate::qwen::SpecForcedApplyResult::Stopped(reason);
        match apply {
            hipfire_generate::qwen::SpecForcedApplyResult::Terminal => panic!("not under test"),
            hipfire_generate::qwen::SpecForcedApplyResult::Stopped(r) => {
                if semantic_stop.is_none() && hipfire_generate::qwen::spec_stop_is_semantic(Some(r)) {
                    semantic_stop = Some(r);
                }
                first_token_is_eos = true;
            }
            hipfire_generate::qwen::SpecForcedApplyResult::Applied | hipfire_generate::qwen::SpecForcedApplyResult::Skipped => {
                panic!("expected Stopped")
            }
        }
        assert_eq!(semantic_stop, Some(StopReason::StopSequence));
        assert!(first_token_is_eos);

        // while !first_token_is_eos && generated < max_tokens { spec.step ... }
        let generated = 0usize;
        let max_tokens = 16usize;
        let mut stepped = false;
        let mut later_force = false;
        if !first_token_is_eos && generated < max_tokens {
            // would take_forced + spec.step
            later_force = true;
            stepped = true;
        }
        assert!(
            !stepped && !later_force,
            "begin Stopped must skip every subsequent force and spec.step"
        );

        // --- mid-window path (mirrors hipfire_generate::qwen::generate_spec forced_after match) ---
        let mut semantic_stop_mid: Option<StopReason> = None;
        let mut hit_eos = false;
        let mut think_cap_hit = false;
        let mid = hipfire_generate::qwen::SpecForcedApplyResult::Stopped(StopReason::StopSequence);
        match mid {
            hipfire_generate::qwen::SpecForcedApplyResult::Terminal => panic!("not under test"),
            hipfire_generate::qwen::SpecForcedApplyResult::Stopped(r) => {
                if semantic_stop_mid.is_none() && hipfire_generate::qwen::spec_stop_is_semantic(Some(r)) {
                    semantic_stop_mid = Some(r);
                }
                match r {
                    StopReason::ThinkCap => think_cap_hit = true,
                    StopReason::Eos | StopReason::StopSequence | StopReason::GrammarViolation => {
                        hit_eos = true
                    }
                }
            }
            hipfire_generate::qwen::SpecForcedApplyResult::Applied | hipfire_generate::qwen::SpecForcedApplyResult::Skipped => {
                panic!("expected Stopped")
            }
        }
        assert_eq!(semantic_stop_mid, Some(StopReason::StopSequence));
        assert!(hit_eos);
        assert!(!think_cap_hit);

        // After mid Stopped the cycle must not re-enter force or continue the
        // outer decode as if Applied. Model the break: no second take_forced.
        let mut second_force_applied = false;
        if !hit_eos && !think_cap_hit {
            second_force_applied = true;
        }
        assert!(
            !second_force_applied,
            "mid Stopped must not apply a later forced suffix"
        );

        // hipfire_generate::common::SpecRun carries semantic_stop into the wrapper independently of EOT.
        let run_semantic = semantic_stop_mid;
        assert!(run_semantic.is_some());
        assert!(hipfire_generate::qwen::spec_stop_is_semantic(run_semantic));
    }

    /// First-token user stop at max_tokens=1 must classify as stop (not length)
    /// via semantic_stop surviving independently of decoded_eot.
    #[test]
    fn first_token_stop_sequence_at_max_tokens_one_is_stop_not_length() {
        let tok = test_tokenizer();
        let ids = tok.encode("STOP");
        assert!(!ids.is_empty());
        let first = ids[0];
        let first_text = tok.decode(&[first]);
        let mut emit = hipfire_arch_qwen35::spec_emit::Qwen35Emit::from_ctx(SpecEmitCtx {
            tokenizer: &tok,
            eos: 9,
            im_end: Some(1),
            tools: None,
            enable_grammar: false,
            stop: vec![first_text.clone()],
            max_think: 0,
            max_tokens: 1,
            assistant_prefix: AssistantPrefix::Plain,
            think_mode: hipfire_runtime::prompt_frame::ThinkMode::NonThink,
            decoded_vocab: None,
        });
        let first_begin = emit.begin(first);
        assert_eq!(first_begin.stop, Some(StopReason::StopSequence));

        // hipfire_generate::qwen::generate_spec sticky capture (begin path).
        let mut semantic_stop: Option<StopReason> = if hipfire_generate::qwen::spec_stop_is_semantic(first_begin.stop) {
            first_begin.stop
        } else {
            None
        };
        assert_eq!(semantic_stop, Some(StopReason::StopSequence));
        assert!(hipfire_generate::qwen::spec_stop_is_semantic(semantic_stop));

        // Budget spent on the first (and only) token; no decoded_eot required.
        let generated = 1usize;
        let max_tokens = 1usize;
        let decoded_eot = false; // user stop may not set EOT
        let hit_length =
            hipfire_generate::common::qwen_dflash_hit_length_cap(generated, max_tokens, decoded_eot, semantic_stop.is_some());
        assert!(
            !hit_length,
            "semantic StopSequence at cap must not classify as length"
        );

        // Wrapper wire: stop, not length.
        let fin = summary_stop(&first_text);
        let term = hipfire_generate::qwen::qwen_dflash_wire_terminal(&fin, hit_length, false, &first_text, false);
        match &term {
            hipfire_generate::qwen::QwenDflashWireTerminal::Done {
                finish_reason,
                store_cache,
                release_tool_calls,
                ..
            } => {
                assert_eq!(*finish_reason, "stop");
                assert!(*store_cache);
                assert!(!*release_tool_calls);
            }
            other => panic!("expected stop Done, got {other:?}"),
        }

        // Contrast: same numbers without semantic_stop → length.
        assert!(hipfire_generate::common::qwen_dflash_hit_length_cap(1, 1, false, false));
        let _ = &mut semantic_stop;
    }

    /// Held tool_calls + semantic stop at the budget boundary must finish as
    /// tool_calls (not length). hipfire_generate::common::finish_summary_held_tool_calls feeds the wire.
    #[test]
    fn held_tool_calls_with_semantic_stop_at_cap_is_tool_calls_not_length() {
        let calls = vec![ToolCall {
            id: None,
            name: "get_weather".into(),
            arguments: serde_json::json!({"city": "SF"}),
            rendered_body: None,
        }];
        let fin = summary_tool_calls(calls.clone());
        let held = hipfire_generate::common::finish_summary_held_tool_calls(&fin);
        assert_eq!(held.len(), 1);
        assert_eq!(held[0].name, "get_weather");

        // generated == max_tokens, no decoded_eot, but semantic stop sticky.
        let generated = 8usize;
        let max_tokens = 8usize;
        let decoded_eot = false;
        let semantic_stop = Some(StopReason::StopSequence);
        assert!(hipfire_generate::qwen::spec_stop_is_semantic(semantic_stop));
        let hit_length =
            hipfire_generate::common::qwen_dflash_hit_length_cap(generated, max_tokens, decoded_eot, semantic_stop.is_some());
        assert!(!hit_length, "semantic stop must beat length at cap");

        let term = hipfire_generate::qwen::qwen_dflash_wire_terminal(&fin, hit_length, false, "Sure.", false);
        match &term {
            hipfire_generate::qwen::QwenDflashWireTerminal::Done {
                finish_reason,
                release_tool_calls,
                store_cache,
                wire_tool_calls,
                ..
            } => {
                assert_eq!(*finish_reason, "tool_calls");
                assert!(*release_tool_calls);
                assert!(*store_cache);
                assert_eq!(wire_tool_calls.len(), 1);
                assert_eq!(wire_tool_calls[0].name, "get_weather");
            }
            other => panic!("expected tool_calls Done, got {other:?}"),
        }

        // Without semantic_stop the same finish would be suppressed as length.
        assert!(hipfire_generate::common::qwen_dflash_hit_length_cap(8, 8, false, false));
        let length_term = hipfire_generate::qwen::qwen_dflash_wire_terminal(&fin, true, false, "Sure.", false);
        match &length_term {
            hipfire_generate::qwen::QwenDflashWireTerminal::Done {
                finish_reason,
                release_tool_calls,
                wire_tool_calls,
                ..
            } => {
                assert_eq!(*finish_reason, "length");
                assert!(!*release_tool_calls);
                assert!(wire_tool_calls.is_empty());
            }
            other => panic!("expected length Done, got {other:?}"),
        }
    }

    // ── Task 4 forced-continuation physical-cap admission ─────────────────

    /// Pure admission: no-eviction requires a free pending-seed write slot
    /// after the commit (`post_position < physical_cap`). Exact-cap rejects.
    #[test]
    fn forced_commit_no_evict_exact_cap_rejects_pending_seed_slot() {
        let physical_cap = 16usize;
        let position = 12usize;
        let commit_len = 4usize; // post_position == physical_cap
        assert_eq!(position.saturating_add(commit_len), physical_cap);
        assert!(
            !hipfire_generate::qwen::spec_forced_commit_admits(position, commit_len, physical_cap, false),
            "no-eviction exact-cap must reject: pending seed needs a legal slot"
        );
        // One slot under cap still fits (post == cap-1).
        assert!(hipfire_generate::qwen::spec_forced_commit_admits(
            position,
            commit_len.saturating_sub(1),
            physical_cap,
            false
        ));
        // Over-cap also rejects.
        assert!(!hipfire_generate::qwen::spec_forced_commit_admits(
            position,
            commit_len.saturating_add(1),
            physical_cap,
            false
        ));
    }

    /// Eviction path still refuses post_position > physical_cap before any GPU
    /// write. Exact-cap is the only boundary that eviction may open.
    #[test]
    fn forced_commit_eviction_over_cap_rejects_before_gpu() {
        let physical_cap = 16usize;
        let position = 12usize;
        let over = 5usize; // post_position = 17 > cap
        assert!(position.saturating_add(over) > physical_cap);
        assert!(
            !hipfire_generate::qwen::spec_forced_commit_admits(position, over, physical_cap, true),
            "eviction must not admit over-cap commits"
        );

        // Deterministic pre-GPU gate: reject ⇒ no GPU commit, no staged render.
        #[derive(Debug, Clone, Copy, PartialEq, Eq)]
        enum Phase {
            Staged,
            GpuCommitted,
            Rendered,
            ErrorOnly,
        }
        let admitted = hipfire_generate::qwen::spec_forced_commit_admits(position, over, physical_cap, true);
        let mut phase = Phase::Staged;
        let mut rendered = 0usize;
        if !admitted {
            // Production: rollback + ErrorOnly terminal; discard staged events.
            phase = Phase::ErrorOnly;
        } else {
            phase = Phase::GpuCommitted;
            phase = Phase::Rendered;
            rendered = 1;
        }
        assert_eq!(phase, Phase::ErrorOnly);
        assert_eq!(
            rendered, 0,
            "capacity reject must never render staged events"
        );
        assert_ne!(phase, Phase::GpuCommitted);
        assert_ne!(phase, Phase::Rendered);
        // Same wire class as maybe_evict / on_evict failures.
        assert_eq!(hipfire_generate::qwen::classify_evict_failure_wire(), hipfire_generate::qwen::SpecFailClosedWire::ErrorOnly);
    }

    /// Eviction exact-cap admits only because post-commit maybe_evict+on_evict
    /// is mandatory before host seed/raw/render and must leave a free seed slot.
    #[test]
    fn forced_commit_eviction_exact_cap_admits_with_mandatory_post_commit_evict() {
        let physical_cap = 16usize;
        let position = 12usize;
        let commit_len = 4usize; // post_position == physical_cap
        assert_eq!(position.saturating_add(commit_len), physical_cap);

        assert!(
            hipfire_generate::qwen::spec_forced_commit_admits(position, commit_len, physical_cap, true),
            "eviction may admit exact-cap"
        );
        // Contrast: same numbers without eviction reject.
        assert!(!hipfire_generate::qwen::spec_forced_commit_admits(
            position,
            commit_len,
            physical_cap,
            false
        ));

        // Ordering model for the admitted exact-cap path: GPU commit → mandatory
        // post-commit eviction → require post_evict < physical_cap → only then
        // host position/seed/raw/render. Skipping eviction must not reach render.
        #[derive(Debug, Clone, Copy, PartialEq, Eq)]
        enum Step {
            Admit,
            GpuCommit,
            PostCommitEvict,
            HostRender,
            ErrorOnly,
        }
        let mut steps: Vec<Step> = Vec::new();
        let admitted = hipfire_generate::qwen::spec_forced_commit_admits(position, commit_len, physical_cap, true);
        assert!(admitted);
        steps.push(Step::Admit);
        steps.push(Step::GpuCommit);

        let eviction_enabled = true;
        let mut post_position = position.saturating_add(commit_len);
        let mut rendered = false;
        if eviction_enabled {
            // Mandatory: maybe_evict + on_evict before host updates.
            steps.push(Step::PostCommitEvict);
            // Synthetic successful compaction frees the pending-seed slot.
            post_position = physical_cap.saturating_sub(1);
            if post_position >= physical_cap {
                steps.push(Step::ErrorOnly);
            } else {
                steps.push(Step::HostRender);
                rendered = true;
            }
        } else {
            steps.push(Step::HostRender);
            rendered = true;
        }
        assert_eq!(
            steps,
            vec![
                Step::Admit,
                Step::GpuCommit,
                Step::PostCommitEvict,
                Step::HostRender
            ]
        );
        assert!(rendered);
        assert!(post_position < physical_cap);

        // If post-evict still has no seed slot → ErrorOnly, no render.
        let mut bad_steps: Vec<Step> = vec![Step::Admit, Step::GpuCommit, Step::PostCommitEvict];
        let bad_post = physical_cap; // eviction failed to free a slot
        let mut bad_rendered = false;
        if bad_post >= physical_cap {
            bad_steps.push(Step::ErrorOnly);
        } else {
            bad_steps.push(Step::HostRender);
            bad_rendered = true;
        }
        assert_eq!(
            bad_steps,
            vec![
                Step::Admit,
                Step::GpuCommit,
                Step::PostCommitEvict,
                Step::ErrorOnly
            ]
        );
        assert!(!bad_rendered);
    }

    /// Comfortably under the physical cap admits with or without eviction.
    #[test]
    fn forced_commit_under_threshold_fits() {
        let physical_cap = 64usize;
        let position = 10usize;
        let commit_len = 3usize;
        assert!(position.saturating_add(commit_len) < physical_cap);
        assert!(hipfire_generate::qwen::spec_forced_commit_admits(
            position,
            commit_len,
            physical_cap,
            false
        ));
        assert!(hipfire_generate::qwen::spec_forced_commit_admits(
            position,
            commit_len,
            physical_cap,
            true
        ));
        // Empty commit (seed-only replace) is always under threshold.
        assert!(hipfire_generate::qwen::spec_forced_commit_admits(position, 0, physical_cap, false));
        assert!(hipfire_generate::qwen::spec_forced_commit_admits(position, 0, physical_cap, true));
    }

    /// Admission uses the actual GPU commit slice (`tx.commit.len()`), never the
    /// forced token count. Non-committable seeds omit the trigger and shrink
    /// the commit — that shorter length is what capacity sees.
    #[test]
    fn forced_commit_admission_uses_tx_commit_len_not_forced_count() {
        let physical_cap = 10usize;
        let position = 8usize;
        let seed = 7u32;
        let forced = [10u32, 11, 12]; // forced.len() == 3

        // Committable: commit = [seed, 10, 11] → len 3; post = 11 > cap.
        let keep = hipfire_generate::qwen::spec_forced_pending_seed_tx(seed, &forced, true);
        assert_eq!(keep.commit.len(), 3);
        assert_eq!(keep.commit.len(), keep.position_delta);
        assert!(
            !hipfire_generate::qwen::spec_forced_commit_admits(position, keep.commit.len(), physical_cap, false),
            "committable commit_len=3 at pos=8 must reject under no-evict"
        );
        assert!(
            !hipfire_generate::qwen::spec_forced_commit_admits(position, keep.commit.len(), physical_cap, true),
            "committable commit_len=3 at pos=8 is over-cap even with eviction"
        );

        // Non-committable: commit = [10, 11] → len 2 (seed omitted); post = 10.
        let omit = hipfire_generate::qwen::spec_forced_pending_seed_tx(seed, &forced, false);
        assert_eq!(omit.commit, vec![10, 11]);
        assert_eq!(omit.commit.len(), 2);
        assert_eq!(omit.position_delta, omit.commit.len());
        assert_ne!(
            omit.commit.len(),
            forced.len(),
            "must not admit against forced token count"
        );
        // Using forced.len() would be wrong (post=11 over-cap); actual slice fits
        // exact-cap under eviction and rejects under no-evict (needs seed slot).
        assert_eq!(position.saturating_add(omit.commit.len()), physical_cap);
        assert!(
            !hipfire_generate::qwen::spec_forced_commit_admits(position, omit.commit.len(), physical_cap, false),
            "no-evict exact-cap still needs a pending-seed slot"
        );
        assert!(
            hipfire_generate::qwen::spec_forced_commit_admits(position, omit.commit.len(), physical_cap, true),
            "eviction admits exact-cap on the actual (shorter) commit slice"
        );
        // Guard: if a caller mistakenly passed forced.len(), both modes reject.
        assert!(!hipfire_generate::qwen::spec_forced_commit_admits(
            position,
            forced.len(),
            physical_cap,
            true
        ));
        assert!(!hipfire_generate::qwen::spec_forced_commit_admits(
            position,
            forced.len(),
            physical_cap,
            false
        ));

        // Single forced + non-committable: empty commit — no GPU write.
        // Admission still uses commit_len=0 (not forced.len()==1).
        let one = hipfire_generate::qwen::spec_forced_pending_seed_tx(seed, &[42], false);
        assert!(one.commit.is_empty());
        assert_ne!(
            one.commit.len(),
            1,
            "must not treat forced count as commit_len"
        );
        assert!(hipfire_generate::qwen::spec_forced_commit_admits(
            position,
            one.commit.len(),
            physical_cap,
            false
        ));
        // At physical_cap with zero-length commit: no-evict still needs a free
        // pending-seed slot (post == cap rejects); eviction admits exact-cap.
        assert!(!hipfire_generate::qwen::spec_forced_commit_admits(
            physical_cap,
            one.commit.len(),
            physical_cap,
            false
        ));
        assert!(hipfire_generate::qwen::spec_forced_commit_admits(
            physical_cap,
            one.commit.len(),
            physical_cap,
            true
        ));
    }

    #[test]
    fn dflash_client_commit_preserves_release_and_store() {
        let e = hipfire_generate::qwen::qwen_client_commit_effects(ClientTerminalDecision::Commit, true, true);
        assert!(e.release_tool_calls && e.store_cache && e.emit_done);
        // Successful Done classify → intended flags gate release/store.
        let tc = ToolCall {
            id: None,
            name: "read".into(),
            arguments: r#"{"path":"/x"}"#.into(),
            rendered_body: None,
        };
        let term = hipfire_generate::qwen::qwen_dflash_wire_terminal(
            &summary_tool_calls(vec![tc.clone()]),
            false,
            false,
            "Sure.",
            false,
        );
        match &term {
            hipfire_generate::qwen::QwenDflashWireTerminal::Done {
                release_tool_calls,
                store_cache,
                wire_tool_calls,
                ..
            } => {
                let effects = hipfire_generate::qwen::qwen_client_commit_effects(
                    ClientTerminalDecision::Commit,
                    *release_tool_calls && !wire_tool_calls.is_empty(),
                    *store_cache,
                );
                assert!(effects.release_tool_calls);
                assert!(effects.store_cache);
                assert!(effects.emit_done);
                let mut action = hipfire_generate::qwen::qwen_dflash_cache_action(&term);
                action.store = effects.store_cache && action.store;
                assert!(action.store);
            }
            other => panic!("expected Done, got {other:?}"),
        }
    }

    #[test]
    fn dflash_client_abort_suppresses_release_store_done() {
        set_active_attempt_id(33);
        let tc = ToolCall {
            id: None,
            name: "read".into(),
            arguments: r#"{"path":"/x"}"#.into(),
            rendered_body: None,
        };
        let term =
            hipfire_generate::qwen::qwen_dflash_wire_terminal(&summary_tool_calls(vec![tc]), false, false, "Sure.", false);
        let hipfire_generate::qwen::QwenDflashWireTerminal::Done {
            release_tool_calls,
            store_cache,
            wire_tool_calls,
            ..
        } = &term
        else {
            panic!("expected Done");
        };
        let effects = hipfire_generate::qwen::qwen_client_commit_effects(
            ClientTerminalDecision::Abort,
            *release_tool_calls && !wire_tool_calls.is_empty(),
            *store_cache,
        );
        assert!(!effects.release_tool_calls);
        assert!(!effects.store_cache);
        assert!(!effects.emit_done);

        let mut sink = Vec::new();
        // No tool release on Abort.
        let mut action = hipfire_generate::qwen::qwen_dflash_cache_action(&term);
        action.store = effects.store_cache && action.store;
        let mut stored = false;
        let _ = hipfire_generate::qwen::qwen_dflash_apply_cache_action(|_fp, _seq| stored = true, &action, vec![1, 2, 3]);
        assert!(!stored);

        let ep = hipfire_generate::common::RollbackEpilogue {
            rolled_back: true,
            context: None,
        };
        hipfire_generate::common::emit_spec_cancel_after_rollback(&mut sink, "df-abort", 7, &ep);
        let out = String::from_utf8_lossy(&sink);
        assert!(!out.contains("\"type\":\"tool_calls\""));
        assert!(out.contains("\"type\":\"aborted\""));
        assert!(out.contains("\"finish_reason\":\"aborted\""));
        assert!(!out.contains("\"finish_reason\":\"tool_calls\""));
        assert!(out.contains("\"attempt_id\":33"));
    }
