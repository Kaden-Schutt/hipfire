// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! DeepSeek4 malformed-DSML terminal handling.
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

    use hipfire_generate::common::{ds4_spec_finish_route, emit_ds4_malformed_terminal};
use hipfire_engine::emit::emit_visible_token;
use hipfire_engine::terminal::{set_active_attempt_id, ClientTerminalDecision};
    use hipfire_generate::{common::asst_turn_fingerprint, common::ds4_apply_cache_action, common::ds4_ar_ep_cache_action, common::ds4_ar_ep_finish_route, dense::ds4_cache_action, common::ds4_client_commit_effects, common::ds4_ep_abort_wire_events, common::ds4_gen_start_contract_version, common::ds4_malformed_terminal_action, dense::ds4_spec_wire_terminal, common::ds4_stream_event_wireable, qwen::emit_ds4_ep_gen_start, common::emit_ds4_malformed_action, common::gen_start_contract_version_for_arch, common::normalize_asst_turn_for_fingerprint, qwen::spec_outcome_seed_committable, qwen::spec_should_flush_pending_seed, common::Ds4ArEpRouteTerminal, common::Ds4ClientCommitEffects, dense::Ds4SpecWireTerminal};
    use hipfire_arch_deepseek4::dsml::{
        DsmlDeferredCalls, DsmlDeferredOutcome, StreamEvent, StreamParser, TOOL_CALLS_CLOSE,
        TOOL_CALLS_OPEN,
    };
    use hipfire_runtime::prompt_frame::ToolCall;
    use hipfire_runtime::spec::{ClientEvent, FinishSummary, SpecEmit};

    fn complete_invoke(name: &str, arg_name: &str, arg_val: &str) -> String {
        format!(
            "{open}\n<｜DSML｜invoke name=\"{name}\">\n\
             <｜DSML｜parameter name=\"{arg_name}\" string=\"true\">{arg_val}</｜DSML｜parameter>\n\
             </｜DSML｜invoke>\n{close}",
            open = TOOL_CALLS_OPEN,
            close = TOOL_CALLS_CLOSE,
            name = name,
            arg_name = arg_name,
            arg_val = arg_val,
        )
    }

    /// Feed a full turn through the production deferred absorber (same API as
    /// Deepseek4Emit::feed_and_emit / finish).
    fn deferred_from_text(text: &str) -> DsmlDeferredCalls {
        let mut p = StreamParser::new();
        let mut deferred = DsmlDeferredCalls::new();
        let _visible = deferred.absorb_all(p.feed(text));
        let _tail = deferred.absorb_all(p.finish());
        deferred
    }

    /// AR/EP production path: deferred finalize → shared pure route.
    fn ar_ep_from_deferred(d: DsmlDeferredCalls, hit_length_cap: bool) -> hipfire_generate::common::Ds4ArEpRouteTerminal {
        match d.finalize(hit_length_cap) {
            DsmlDeferredOutcome::Malformed { detail } => {
                hipfire_generate::common::ds4_ar_ep_finish_route(Some(detail), Vec::new(), hit_length_cap)
            }
            DsmlDeferredOutcome::Length => hipfire_generate::common::ds4_ar_ep_finish_route(None, Vec::new(), true),
            DsmlDeferredOutcome::Stop => hipfire_generate::common::ds4_ar_ep_finish_route(None, Vec::new(), false),
            DsmlDeferredOutcome::ToolCalls(calls) => {
                let wire: Vec<ToolCall> = calls
                    .into_iter()
                    .map(|c| ToolCall {
                        id: None,
                        name: c.name,
                        arguments: c.arguments,
                        rendered_body: None,
                    })
                    .collect();
                hipfire_generate::common::ds4_ar_ep_finish_route(None, wire, false)
            }
        }
    }

    /// Spec path: provisional finalize(false) as Deepseek4Emit::finish does,
    /// then wrapper applies length via hipfire_generate::dense::ds4_spec_wire_terminal.
    fn spec_wire_from_deferred(d: DsmlDeferredCalls, hit_length_cap: bool) -> hipfire_generate::dense::Ds4SpecWireTerminal {
        let (finish_reason, tool_calls) = if d.is_malformed() {
            let _ = d.finalize(false);
            ("malformed_protocol", 0usize)
        } else {
            let n = d.buffered_len();
            match d.finalize(false) {
                DsmlDeferredOutcome::ToolCalls(_) => ("tool_calls", n),
                DsmlDeferredOutcome::Stop | DsmlDeferredOutcome::Length => ("stop", 0),
                DsmlDeferredOutcome::Malformed { .. } => ("malformed_protocol", 0),
            }
        };
        hipfire_generate::dense::ds4_spec_wire_terminal(finish_reason, tool_calls, hit_length_cap)
    }

    #[test]
    fn malformed_action_is_typed_validation_non_retryable() {
        let action =
            hipfire_generate::common::ds4_malformed_terminal_action("unclosed DSML tool_calls block at end of output");
        assert_eq!(action.class, "validation");
        assert!(!action.retryable);
        assert!(!action.rolled_back);
        assert!(action.message.contains("malformed"));
        assert!(action.message.contains("unclosed"));
        assert!(action.message.contains("tool_calls"));
    }

    #[test]
    fn malformed_action_suppresses_done_cache_and_calls() {
        let action =
            hipfire_generate::common::ds4_malformed_terminal_action("unclosed DSML tool_calls block at end of output");
        assert!(!action.emit_done, "error XOR done");
        assert!(!action.store_cache, "no assistant-cache write");
        assert!(!action.expose_tool_calls, "no executable calls");
    }

    #[test]
    fn complete_call_then_unclosed_discards_all_on_ar_ep() {
        let mut p = StreamParser::new();
        let mut deferred = DsmlDeferredCalls::new();
        let _ = deferred.absorb_all(p.feed(&complete_invoke("alpha", "x", "1")));
        assert_eq!(deferred.buffered_len(), 1, "first complete call buffers");
        let _ = deferred.absorb_all(p.feed(TOOL_CALLS_OPEN));
        let _ = deferred.absorb_all(p.feed("\n<｜DSML｜invoke name=\"beta\">"));
        let _ = deferred.absorb_all(p.finish());
        assert!(
            deferred.is_malformed(),
            "unclosed second block latches malformed"
        );

        let terminal = ar_ep_from_deferred(deferred, false);
        match terminal {
            hipfire_generate::common::Ds4ArEpRouteTerminal::Malformed(action) => {
                assert_eq!(action.class, "validation");
                assert!(!action.retryable);
                assert!(!action.emit_done);
                assert!(!action.store_cache);
                assert!(!action.expose_tool_calls);
            }
            other => panic!("expected Malformed discard of earlier calls, got {other:?}"),
        }
    }

    #[test]
    fn complete_call_safe_terminal_releases_on_ar_ep() {
        let deferred = deferred_from_text(&complete_invoke("alpha", "x", "1"));
        assert_eq!(deferred.buffered_len(), 1);
        let terminal = ar_ep_from_deferred(deferred, false);
        match terminal {
            hipfire_generate::common::Ds4ArEpRouteTerminal::Safe {
                finish_reason,
                wire_tool_calls,
                store_cache,
            } => {
                assert_eq!(finish_reason, "tool_calls");
                assert_eq!(wire_tool_calls.len(), 1);
                assert_eq!(wire_tool_calls[0].name, "alpha");
                assert!(store_cache);
            }
            other => panic!("expected Safe tool_calls release, got {other:?}"),
        }
    }

    #[test]
    fn length_cap_is_not_tool_safe_even_with_complete_calls() {
        let deferred = deferred_from_text(&complete_invoke("alpha", "x", "1"));
        assert_eq!(deferred.buffered_len(), 1);
        let terminal = ar_ep_from_deferred(deferred, true);
        match terminal {
            hipfire_generate::common::Ds4ArEpRouteTerminal::Safe {
                finish_reason,
                wire_tool_calls,
                store_cache,
            } => {
                assert_eq!(finish_reason, "length");
                assert!(wire_tool_calls.is_empty(), "length never releases calls");
                assert!(!store_cache);
            }
            other => panic!("expected Safe length with empty calls, got {other:?}"),
        }
    }

    #[test]
    fn speculative_complete_then_unclosed_discards_via_production_deferred() {
        let mut p = StreamParser::new();
        let mut deferred = DsmlDeferredCalls::new();
        let visible = deferred.absorb_all(p.feed(&complete_invoke("alpha", "x", "1")));
        assert!(
            visible.iter().all(|e| hipfire_generate::common::ds4_stream_event_wireable(e)),
            "absorb returns only wireable visible events"
        );
        assert_eq!(deferred.buffered_len(), 1);
        let _ = deferred.absorb_all(p.feed(TOOL_CALLS_OPEN));
        let _ = deferred.absorb_all(p.finish());
        assert!(deferred.is_malformed());
        assert_eq!(
            deferred.buffered_len(),
            1,
            "buffer retains until finalize; discard is finalize's job"
        );

        match deferred.finalize(false) {
            DsmlDeferredOutcome::Malformed { .. } => {}
            other => panic!("expected Malformed outcome discarding calls, got {other:?}"),
        }

        let wire = hipfire_generate::dense::ds4_spec_wire_terminal("malformed_protocol", 0, false);
        match wire {
            hipfire_generate::dense::Ds4SpecWireTerminal::Malformed(action) => {
                assert_eq!(action.class, "validation");
                assert!(!action.retryable);
                assert!(!action.emit_done);
                assert!(!action.store_cache);
                assert!(!action.expose_tool_calls);
            }
            other => panic!("expected Malformed wire terminal, got {other:?}"),
        }
        assert!(ds4_spec_finish_route("stop", 0).is_none());
        assert!(ds4_spec_finish_route("tool_calls", 1).is_none());
    }

    #[test]
    fn speculative_safe_stop_releases_held_calls() {
        // Production Deepseek4Emit::finish path: finalize(false) → held ToolCalls
        // on FinishSummary; wrapper releases only when length is false.
        let deferred = deferred_from_text(&complete_invoke("alpha", "x", "1"));
        let wire = spec_wire_from_deferred(deferred, false);
        match wire {
            hipfire_generate::dense::Ds4SpecWireTerminal::Done {
                finish_reason,
                release_tool_calls,
                store_cache,
            } => {
                assert_eq!(finish_reason, "tool_calls");
                assert!(release_tool_calls, "safe stop must release held calls");
                assert!(store_cache);
            }
            other => panic!("expected Done tool_calls release, got {other:?}"),
        }
    }

    #[test]
    fn speculative_length_suppresses_held_calls_and_cache() {
        // Same provisional finish as Deepseek4Emit (finalize false → tool_calls
        // count), but wrapper length wins: no release, finish_reason=length.
        let deferred = deferred_from_text(&complete_invoke("alpha", "x", "1"));
        let wire = spec_wire_from_deferred(deferred, true);
        match wire {
            hipfire_generate::dense::Ds4SpecWireTerminal::Done {
                finish_reason,
                release_tool_calls,
                store_cache,
            } => {
                assert_eq!(finish_reason, "length");
                assert!(!release_tool_calls, "length must not release held calls");
                assert!(!store_cache);
            }
            other => panic!("expected Done length suppress, got {other:?}"),
        }
    }

    #[test]
    fn speculative_complete_then_malformed_never_releases() {
        let mut p = StreamParser::new();
        let mut deferred = DsmlDeferredCalls::new();
        let _ = deferred.absorb_all(p.feed(&complete_invoke("alpha", "x", "1")));
        let _ = deferred.absorb_all(p.feed(TOOL_CALLS_OPEN));
        let _ = deferred.absorb_all(p.finish());
        let wire = spec_wire_from_deferred(deferred, false);
        match wire {
            hipfire_generate::dense::Ds4SpecWireTerminal::Malformed(action) => {
                assert!(!action.expose_tool_calls);
                assert!(!action.emit_done);
                assert!(!action.store_cache);
            }
            other => panic!("expected Malformed, got {other:?}"),
        }
        // Length cannot flip a malformed finish into a done/tool_calls release.
        let mut p = StreamParser::new();
        let mut deferred = DsmlDeferredCalls::new();
        let _ = deferred.absorb_all(p.feed(&complete_invoke("alpha", "x", "1")));
        let _ = deferred.absorb_all(p.feed(TOOL_CALLS_OPEN));
        let _ = deferred.absorb_all(p.finish());
        let wire_len = spec_wire_from_deferred(deferred, true);
        assert!(
            matches!(wire_len, hipfire_generate::dense::Ds4SpecWireTerminal::Malformed(_)),
            "malformed wins over length"
        );
    }

    #[test]
    fn stream_event_tool_calls_not_wireable_mid_turn() {
        let ev = StreamEvent::ToolCalls(vec![hipfire_arch_deepseek4::dsml::ToolCall {
            name: "x".into(),
            arguments: serde_json::json!({}),
        }]);
        assert!(!hipfire_generate::common::ds4_stream_event_wireable(&ev));
        assert!(hipfire_generate::common::ds4_stream_event_wireable(&StreamEvent::Token("hi".into())));
        assert!(hipfire_generate::common::ds4_stream_event_wireable(&StreamEvent::Reasoning(
            "r".into()
        )));
        assert!(!hipfire_generate::common::ds4_stream_event_wireable(&StreamEvent::Malformed {
            detail: "x".into()
        }));
        // Production absorber never returns ToolCalls as visible.
        let mut d = DsmlDeferredCalls::new();
        assert!(d.absorb(ev).is_none());
        assert_eq!(d.buffered_len(), 1);
    }

    #[test]
    fn emit_writes_one_validation_error_no_done_or_calls() {
        set_active_attempt_id(17);
        let mut buf = Vec::new();
        let action =
            hipfire_generate::common::ds4_malformed_terminal_action("unclosed DSML tool_calls block at end of output");
        hipfire_generate::common::emit_ds4_malformed_action(&mut buf, "req-ds4", &action);
        let text = String::from_utf8(buf).unwrap();
        let lines: Vec<&str> = text.lines().filter(|l| !l.is_empty()).collect();
        assert_eq!(lines.len(), 1, "exactly one terminal envelope, got {text}");
        let v: serde_json::Value = serde_json::from_str(lines[0]).unwrap();
        assert_eq!(v["type"], "error");
        assert_eq!(v["id"], "req-ds4");
        assert_eq!(v["class"], "validation");
        assert_eq!(v["retryable"], false);
        assert_eq!(v["rolled_back"], false);
        assert_eq!(v["attempt_id"].as_u64(), Some(17));
        let msg = v["message"].as_str().unwrap_or("");
        assert!(msg.contains("malformed") && msg.contains("unclosed"));
        assert!(!text.contains("\"type\":\"done\""));
        assert!(!text.contains("\"type\":\"tool_calls\""));
        let mut buf2 = Vec::new();
        emit_ds4_malformed_terminal(
            &mut buf2,
            "req-ds4",
            "unclosed DSML tool_calls block at end of output",
        );
        assert_eq!(
            String::from_utf8(buf2)
                .unwrap()
                .lines()
                .filter(|l| !l.is_empty())
                .count(),
            1
        );
        set_active_attempt_id(0);
    }

    #[test]
    fn ds4_gen_start_contract_selection_is_unset() {
        assert_eq!(hipfire_generate::common::gen_start_contract_version_for_arch(9), None);
        assert_eq!(hipfire_generate::common::ds4_gen_start_contract_version(), None);
        assert_eq!(hipfire_generate::common::gen_start_contract_version_for_arch(5), Some(2));
        assert_eq!(hipfire_generate::common::gen_start_contract_version_for_arch(6), Some(2));
        assert_eq!(QWEN_AR_SEMANTIC_CONTRACT_VERSION, 2);
    }

    #[test]
    fn ds4_ep_opens_wire_contract_before_first_token() {
        use hipfire_runtime::prompt_frame::ThinkMode;

        set_active_attempt_id(31);
        let mut sink = Vec::new();
        hipfire_generate::qwen::emit_ds4_ep_gen_start(&mut sink, "req-ep", ThinkMode::NonThink);
        emit_visible_token(&mut sink, "req-ep", "hello");

        let events: Vec<serde_json::Value> = String::from_utf8(sink)
            .unwrap()
            .lines()
            .map(|line| serde_json::from_str(line).unwrap())
            .collect();
        assert_eq!(events.len(), 2);
        assert_eq!(events[0]["type"], "gen_start");
        assert_eq!(events[0]["id"], "req-ep");
        assert_eq!(events[0]["started_in_think"], false);
        assert_eq!(events[0]["attempt_id"], 31);
        assert_eq!(events[1]["type"], "token");
        assert_eq!(events[1]["text"], "hello");
        assert_eq!(events[1]["attempt_id"], 31);
        set_active_attempt_id(0);
    }

    // ── Task 4 definitive terminal-edge blockers (DS4 cache + empty EOS) ──

    /// Safe DS4 speculative terminal stores the verbatim raw streamed_tokens
    /// body through hipfire_generate::dense::ds4_cache_action + hipfire_generate::common::ds4_apply_cache_action (same seam as
    /// hipfire_generate::dense::generate_deepseek4_spec Done branch).
    #[test]
    fn ds4_safe_terminal_stores_verbatim_raw_replay_tokens() {
        let calls = vec![ToolCall {
            id: None,
            name: "lookup".into(),
            arguments: serde_json::json!({"q": "x"}),
            rendered_body: None,
        }];
        let finish = FinishSummary {
            events: vec![
                ClientEvent::Token("Sure.".into()),
                ClientEvent::ToolCalls(calls.clone()),
            ],
            finish_reason: "tool_calls",
            tool_calls: 1,
            visible_text: "Sure.".into(),
            decoded_eot: false,
            open_think: false,
        };
        let wire = hipfire_generate::dense::ds4_spec_wire_terminal("tool_calls", 1, false);
        match &wire {
            hipfire_generate::dense::Ds4SpecWireTerminal::Done {
                finish_reason,
                release_tool_calls,
                store_cache,
            } => {
                assert_eq!(*finish_reason, "tool_calls");
                assert!(*release_tool_calls);
                assert!(*store_cache, "safe stop must authorize cache store");
            }
            other => panic!("expected Done, got {other:?}"),
        }
        let action = hipfire_generate::dense::ds4_cache_action(&wire, &finish, finish.visible_text.as_str());
        assert!(action.store);
        assert_eq!(
            action.fingerprint_text,
            hipfire_generate::common::normalize_asst_turn_for_fingerprint("Sure.")
        );
        assert_eq!(action.tool_calls.len(), 1);
        assert_eq!(action.tool_calls[0].name, "lookup");

        // Verbatim raw body — no surround EOS/Assistant markers (DSML replay).
        let streamed = vec![11u32, 22, 33, 44];
        let mut sink: std::collections::HashMap<u64, Vec<u32>> = std::collections::HashMap::new();
        let fp = hipfire_generate::common::ds4_apply_cache_action(
            |k, v| {
                sink.insert(k, v);
            },
            &action,
            streamed.clone(),
        );
        assert!(fp.is_some(), "safe terminal must mutate cache sink");
        let stored = sink.get(&fp.unwrap()).expect("stored under fingerprint");
        assert_eq!(
            stored, &streamed,
            "cache body must be verbatim run.streamed_tokens"
        );
        // Fingerprint key matches hipfire_generate::common::build_deepseek4_dsml_prompt lookup shape.
        let expected_fp = hipfire_generate::common::asst_turn_fingerprint(&action.fingerprint_text, &action.tool_calls);
        assert_eq!(fp, Some(expected_fp));
    }

    /// Length and fail-closed/malformed never store via hipfire_generate::common::ds4_apply_cache_action
    /// even when a non-empty raw body is offered.
    #[test]
    fn ds4_length_and_fail_closed_skip_cache_store() {
        let finish_tools = FinishSummary {
            events: vec![ClientEvent::ToolCalls(vec![ToolCall {
                id: None,
                name: "alpha".into(),
                arguments: serde_json::json!({}),
                rendered_body: None,
            }])],
            finish_reason: "tool_calls",
            tool_calls: 1,
            visible_text: "partial".into(),
            decoded_eot: false,
            open_think: false,
        };
        let streamed = vec![7u32, 8, 9];

        // Length: store_cache=false, no release, no sink mutation.
        let wire_len = hipfire_generate::dense::ds4_spec_wire_terminal("tool_calls", 1, true);
        match &wire_len {
            hipfire_generate::dense::Ds4SpecWireTerminal::Done {
                finish_reason,
                release_tool_calls,
                store_cache,
            } => {
                assert_eq!(*finish_reason, "length");
                assert!(!*release_tool_calls);
                assert!(!*store_cache);
            }
            other => panic!("expected length Done, got {other:?}"),
        }
        let action_len =
            hipfire_generate::dense::ds4_cache_action(&wire_len, &finish_tools, finish_tools.visible_text.as_str());
        assert!(!action_len.store);
        assert!(
            action_len.tool_calls.is_empty(),
            "length suppresses held calls"
        );
        let mut sink_len = std::collections::HashMap::new();
        assert!(hipfire_generate::common::ds4_apply_cache_action(
            |k, v| {
                sink_len.insert(k, v);
            },
            &action_len,
            streamed.clone()
        )
        .is_none());
        assert!(
            sink_len.is_empty(),
            "length must not populate asst_turn_cache"
        );

        // Malformed fail-closed: no store, no done path.
        let finish_mal = FinishSummary {
            events: Vec::new(),
            finish_reason: "malformed_protocol",
            tool_calls: 0,
            visible_text: String::new(),
            decoded_eot: false,
            open_think: false,
        };
        let wire_mal = hipfire_generate::dense::ds4_spec_wire_terminal("malformed_protocol", 0, false);
        assert!(matches!(wire_mal, hipfire_generate::dense::Ds4SpecWireTerminal::Malformed(_)));
        let action_mal = hipfire_generate::dense::ds4_cache_action(&wire_mal, &finish_mal, finish_mal.visible_text.as_str());
        assert!(!action_mal.store);
        let mut sink_mal = std::collections::HashMap::new();
        assert!(hipfire_generate::common::ds4_apply_cache_action(
            |k, v| {
                sink_mal.insert(k, v);
            },
            &action_mal,
            streamed
        )
        .is_none());
        assert!(sink_mal.is_empty(), "fail-closed must not populate cache");

        // Empty-payload safe stop also refuses store (dead-weight empty turn).
        let finish_empty = FinishSummary {
            events: Vec::new(),
            finish_reason: "stop",
            tool_calls: 0,
            visible_text: String::new(),
            decoded_eot: false,
            open_think: false,
        };
        let wire_empty = hipfire_generate::dense::ds4_spec_wire_terminal("stop", 0, false);
        let action_empty = hipfire_generate::dense::ds4_cache_action(
            &wire_empty,
            &finish_empty,
            finish_empty.visible_text.as_str(),
        );
        assert!(action_empty.store, "wire authorizes stop");
        let mut sink_empty = std::collections::HashMap::new();
        assert!(
            hipfire_generate::common::ds4_apply_cache_action(
                |k, v| {
                    sink_empty.insert(k, v);
                },
                &action_empty,
                vec![1u32],
            )
            .is_none(),
            "empty fingerprint+calls must skip insert"
        );
        assert!(sink_empty.is_empty());
    }

    /// DS4 empty-event EOS is a model terminator only: not committable, not
    /// terminal-flushed, not baked into conversation history. Hidden/raw
    /// Committed events remain committable.
    #[test]
    fn ds4_empty_event_eos_seed_not_terminal_flushed_or_history() {
        use hipfire_runtime::spec::EmitOutcome;

        // Empty-event EOS (Deepseek4Emit::begin/observe on eos_token).
        let eos_out = EmitOutcome {
            events: Vec::new(),
            stop: Some(hipfire_runtime::spec::StopReason::Eos),
        };
        assert!(
            !hipfire_generate::qwen::spec_outcome_seed_committable(&eos_out),
            "empty-event EOS must not be state-committable"
        );
        assert!(
            !hipfire_generate::qwen::spec_should_flush_pending_seed(false, false),
            "non-committable pending seed must skip terminal flush"
        );

        // Event-bearing Committed (including hidden protocol bytes) stays
        // committable so history/GPU flush keep them.
        let committed = EmitOutcome {
            events: vec![ClientEvent::Committed { id: 42, idx: 0 }],
            stop: None,
        };
        assert!(hipfire_generate::qwen::spec_outcome_seed_committable(&committed));
        assert!(hipfire_generate::qwen::spec_should_flush_pending_seed(false, true));

        // Grammar fail-closed always skips flush even if seed was committable.
        assert!(!hipfire_generate::qwen::spec_should_flush_pending_seed(true, true));

        // First-seed init mirrors hipfire_generate::qwen::generate_spec: empty begin → no emitted bake.
        let mut emitted: Vec<u32> = Vec::new();
        let mut generated = 0usize;
        let first_token = 99u32; // eos id in production
        let pending_seed_committable = hipfire_generate::qwen::spec_outcome_seed_committable(&eos_out);
        if pending_seed_committable {
            emitted.push(first_token);
            generated += 1;
        }
        assert!(
            emitted.is_empty() && generated == 0,
            "DS4 first-token EOS must leave history at prompt"
        );

        // Position math for already-processed prior tokens is independent of
        // the non-committable bonus EOS seed (raw_decode may still record it
        // for realign; conversation bake uses emitted only).
        let prompt = vec![1u32, 2, 3];
        let prior_emitted = vec![10u32, 11];
        let conversation = {
            let mut v = prompt.clone();
            v.extend_from_slice(&prior_emitted);
            // Non-committable EOS seed is NOT appended (production bake path).
            v
        };
        assert_eq!(conversation, vec![1, 2, 3, 10, 11]);
        assert!(!conversation.contains(&first_token));

        // Live Deepseek4Emit: EOS begin returns empty events + Eos stop.
        let tok = {
            // Minimal BPE with a single special eos id=7 and printable bytes.
            let mut entries: Vec<String> = Vec::new();
            entries.push(r#""eos": 7"#.to_string());
            for b in 0u32..=255u32 {
                let ch = {
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
                    let idx = bs.iter().position(|&x| x == b).unwrap();
                    char::from_u32(cs[idx]).unwrap()
                };
                let escaped = {
                    let s = ch.to_string();
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
                };
                entries.push(format!(r#""{}": {}"#, escaped, 100 + b));
            }
            let vocab_block = entries.join(", ");
            let json = format!(
                r#"{{
                    "model": {{"type": "BPE", "vocab": {{ {vocab} }}, "merges": []}},
                    "added_tokens": [{{"id": 7, "content": "eos", "special": true}}]
                }}"#,
                vocab = vocab_block,
            );
            hipfire_runtime::tokenizer::Tokenizer::from_hf_json(&json).expect("tok")
        };
        let mut emit = hipfire_arch_deepseek4::spec_emit::Deepseek4Emit::from_ctx(
            hipfire_runtime::spec::SpecEmitCtx {
                tokenizer: &tok,
                eos: 7,
                im_end: None,
                tools: None,
                grammar: true,
                stop: Vec::new(),
                max_think: 0,
                max_tokens: 16,
                assistant_prefix: hipfire_runtime::prompt_frame::AssistantPrefix::Plain,
                think_mode: hipfire_runtime::prompt_frame::ThinkMode::NonThink,
                decoded_vocab: None,
            },
        );
        let begin = emit.begin(7);
        assert!(begin.events.is_empty(), "DS4 EOS begin emits no events");
        assert_eq!(begin.stop, Some(hipfire_runtime::spec::StopReason::Eos));
        assert!(!hipfire_generate::qwen::spec_outcome_seed_committable(&begin));
        assert!(
            emit.streamed_tokens().is_empty(),
            "EOS must not enter streamed_tokens / cache body"
        );
    }

    #[test]
    fn ds4_client_commit_effects_commit_preserves_intended_flags() {
        let e = hipfire_generate::common::ds4_client_commit_effects(ClientTerminalDecision::Commit, true, true);
        assert_eq!(
            e,
            hipfire_generate::common::Ds4ClientCommitEffects {
                release_tool_calls: true,
                store_cache: true,
                emit_done: true,
            }
        );
        let e = hipfire_generate::common::ds4_client_commit_effects(ClientTerminalDecision::Commit, false, true);
        assert!(!e.release_tool_calls);
        assert!(e.store_cache);
        assert!(e.emit_done);
        let e = hipfire_generate::common::ds4_client_commit_effects(ClientTerminalDecision::Commit, false, false);
        assert!(!e.release_tool_calls);
        assert!(!e.store_cache);
        assert!(e.emit_done);
    }

    #[test]
    fn ds4_client_commit_effects_abort_suppresses_all_routes() {
        // Shared gate used by AR / EP / spec Safe terminals.
        for (intended_release, intended_store) in
            [(true, true), (true, false), (false, true), (false, false)]
        {
            let e = hipfire_generate::common::ds4_client_commit_effects(
                ClientTerminalDecision::Abort,
                intended_release,
                intended_store,
            );
            assert_eq!(
                e,
                hipfire_generate::common::Ds4ClientCommitEffects {
                    release_tool_calls: false,
                    store_cache: false,
                    emit_done: false,
                },
                "abort must suppress tools/cache/done regardless of intended flags"
            );
        }
    }

    #[test]
    fn ds4_ar_ep_safe_commit_gate_retains_calls_cache_done() {
        let call = ToolCall {
            id: None,
            name: "search".into(),
            arguments: serde_json::json!({"q": "x"}),
            rendered_body: None,
        };
        let terminal = hipfire_generate::common::ds4_ar_ep_finish_route(None, vec![call.clone()], false);
        let hipfire_generate::common::Ds4ArEpRouteTerminal::Safe {
            finish_reason,
            wire_tool_calls,
            store_cache,
        } = terminal
        else {
            panic!("expected Safe");
        };
        assert_eq!(finish_reason, "tool_calls");
        let effects = hipfire_generate::common::ds4_client_commit_effects(
            ClientTerminalDecision::Commit,
            !wire_tool_calls.is_empty(),
            store_cache,
        );
        assert!(effects.release_tool_calls);
        assert!(effects.store_cache);
        assert!(effects.emit_done);

        let mut action = hipfire_generate::common::ds4_ar_ep_cache_action(
            &hipfire_generate::common::Ds4ArEpRouteTerminal::Safe {
                finish_reason,
                wire_tool_calls: wire_tool_calls.clone(),
                store_cache,
            },
            "hello",
        );
        if !effects.store_cache {
            action.store = false;
        }
        assert!(action.store);
        let mut sink = std::collections::HashMap::new();
        assert!(hipfire_generate::common::ds4_apply_cache_action(
            |k, v| {
                sink.insert(k, v);
            },
            &action,
            vec![1, 2, 3],
        )
        .is_some());
        assert_eq!(sink.len(), 1);
    }

    #[test]
    fn ds4_ar_ep_safe_abort_gate_suppresses_calls_cache_done() {
        let call = ToolCall {
            id: None,
            name: "search".into(),
            arguments: serde_json::json!({"q": "x"}),
            rendered_body: None,
        };
        let terminal = hipfire_generate::common::ds4_ar_ep_finish_route(None, vec![call], false);
        let hipfire_generate::common::Ds4ArEpRouteTerminal::Safe {
            finish_reason,
            wire_tool_calls,
            store_cache,
        } = terminal
        else {
            panic!("expected Safe");
        };
        let effects = hipfire_generate::common::ds4_client_commit_effects(
            ClientTerminalDecision::Abort,
            !wire_tool_calls.is_empty(),
            store_cache,
        );
        assert!(!effects.release_tool_calls);
        assert!(!effects.store_cache);
        assert!(!effects.emit_done);

        let mut action = hipfire_generate::common::ds4_ar_ep_cache_action(
            &hipfire_generate::common::Ds4ArEpRouteTerminal::Safe {
                finish_reason,
                wire_tool_calls,
                store_cache,
            },
            "hello",
        );
        if !effects.store_cache {
            action.store = false;
        }
        assert!(!action.store);
        let mut sink = std::collections::HashMap::new();
        assert!(hipfire_generate::common::ds4_apply_cache_action(
            |k, v| {
                sink.insert(k, v);
            },
            &action,
            vec![1, 2, 3],
        )
        .is_none());
        assert!(sink.is_empty());
    }

    #[test]
    fn ds4_spec_safe_commit_and_abort_gates() {
        let finish = FinishSummary {
            events: Vec::new(),
            finish_reason: "tool_calls",
            tool_calls: 1,
            visible_text: "hi".into(),
            decoded_eot: false,
            open_think: false,
        };
        let wire = hipfire_generate::dense::ds4_spec_wire_terminal("tool_calls", 1, false);
        let hipfire_generate::dense::Ds4SpecWireTerminal::Done {
            release_tool_calls,
            store_cache,
            ..
        } = wire
        else {
            panic!("expected Done");
        };

        let commit = hipfire_generate::common::ds4_client_commit_effects(
            ClientTerminalDecision::Commit,
            release_tool_calls,
            store_cache,
        );
        assert!(commit.release_tool_calls && commit.store_cache && commit.emit_done);
        let terminal_commit = hipfire_generate::dense::Ds4SpecWireTerminal::Done {
            finish_reason: "tool_calls",
            release_tool_calls: commit.release_tool_calls,
            store_cache: commit.store_cache,
        };
        let action_commit =
            hipfire_generate::dense::ds4_cache_action(&terminal_commit, &finish, finish.visible_text.as_str());
        assert!(action_commit.store);

        let abort = hipfire_generate::common::ds4_client_commit_effects(
            ClientTerminalDecision::Abort,
            release_tool_calls,
            store_cache,
        );
        assert!(!abort.release_tool_calls && !abort.store_cache && !abort.emit_done);
        let terminal_abort = hipfire_generate::dense::Ds4SpecWireTerminal::Done {
            finish_reason: "tool_calls",
            release_tool_calls: abort.release_tool_calls,
            store_cache: abort.store_cache,
        };
        let action_abort = hipfire_generate::dense::ds4_cache_action(&terminal_abort, &finish, finish.visible_text.as_str());
        assert!(!action_abort.store);
        assert!(action_abort.tool_calls.is_empty());
    }

    #[test]
    fn ds4_ep_abort_wire_events_carry_attempt_id_on_both() {
        set_active_attempt_id(99);
        let (aborted, done) = hipfire_generate::common::ds4_ep_abort_wire_events("req-ep", 7, 99);
        assert_eq!(aborted["type"], "aborted");
        assert_eq!(aborted["id"], "req-ep");
        assert_eq!(aborted["reason"], "client_cancelled");
        assert_eq!(aborted["attempt_id"], 99);
        assert_eq!(done["type"], "done");
        assert_eq!(done["finish_reason"], "aborted");
        assert_eq!(done["completion_tokens"], 7);
        assert_eq!(done["attempt_id"], 99);
        // Same shape as production semantic helpers.
        assert_eq!(
            aborted,
            hipfire_runtime::semantic::wire_aborted("req-ep", "client_cancelled", 99)
        );
        assert_eq!(
            done,
            hipfire_runtime::semantic::wire_aborted_done("req-ep", 7, 99)
        );
    }
