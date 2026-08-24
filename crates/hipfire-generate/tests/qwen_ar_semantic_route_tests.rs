// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Qwen ar semantic route tests.
//!
//! Moved out of `hipfire-daemon`'s `main.rs`. Compiled into a bin crate these
//! never appeared as their own test target; as integration tests they are
//! reported individually.

#![allow(unused_imports, dead_code, clippy::all)]

use hipfire_engine::emit::*;
use hipfire_engine::scheduler::*;
use hipfire_engine::terminal::*;
use hipfire_generate::ar::*;
use hipfire_generate::batch::*;
use hipfire_generate::common::*;

    
    use hipfire_generate::{common::emit_spec_cancel_after_rollback, qwen::qwen_client_commit_effects, qwen::QwenClientCommitEffects};
    use std::collections::HashMap;

    /// Drive the real shared producer (same object production uses).
    /// Each chunk is raw-committed as a synthetic token before classify.
    fn drive_ar_semantic_path(
        chunks: &[&str],
        started_in_think: bool,
        hit_length_cap: bool,
    ) -> (
        String,
        String,
        Result<QwenArRouteFinish, hipfire_runtime::emit_text::ToolRouteError>,
        bool,
        Vec<u32>,
        Vec<usize>,
    ) {
        set_active_attempt_id(7);
        let mut producer = QwenArSemanticProducer::new("t1", started_in_think);
        let mut sink = Vec::new();
        let mut conversation_tokens = Vec::new();
        let mut streamed_tokens = Vec::new();
        let mut seq_pos = 0usize;
        let mut stopped = false;
        for (i, c) in chunks.iter().enumerate() {
            let token = 1000 + i as u32;
            match producer.commit_and_observe(
                &mut sink,
                &mut conversation_tokens,
                &mut streamed_tokens,
                &mut seq_pos,
                token,
                c.as_bytes(),
            ) {
                Ok(true) => {
                    stopped = true;
                    break;
                }
                Ok(false) => {}
                Err(err) => {
                    let raw = producer.raw_committed.clone();
                    let pos = producer.raw_commit_positions.clone();
                    return (
                        String::from_utf8_lossy(&sink).into_owned(),
                        producer.visible().to_string(),
                        Err(err),
                        stopped,
                        raw,
                        pos,
                    );
                }
            }
        }
        let raw = producer.raw_committed.clone();
        let pos = producer.raw_commit_positions.clone();
        let stopped_flag = producer.stopped_by_filter;
        match producer.finish(&mut sink, hit_length_cap) {
            Ok((fin, visible)) => {
                // Mirror production: caller owns open-think epilogue + terminal.
                // Unit tests have no GPU, so attest rolled_back=false.
                if matches!(fin.cause, QwenArTerminalCause::OpenThink) {
                    let ep = hipfire_generate::common::RollbackEpilogue {
                        rolled_back: false,
                        context: None,
                    };
                    emit_qwen_ar_open_think_terminal(&mut sink, "t1", 0, &ep);
                } else {
                    // Default Commit path: stage calls on done (production
                    // embeds calls in commit_ready/done; no post-commit event).
                    let effects = hipfire_generate::qwen::qwen_client_commit_effects(
                        ClientTerminalDecision::Commit,
                        fin.finish_reason == "tool_calls" && !fin.wire_tool_calls.is_empty(),
                        fin.store_cache,
                    );
                    if effects.emit_done {
                        let mut pending = qwen_ar_done_value(
                            "t1",
                            fin.finish_reason,
                            0,
                            0.0,
                            0,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            0,
                            "",
                        );
                        stage_terminal_tool_calls(
                            &mut pending,
                            fin.finish_reason,
                            &fin.wire_tool_calls,
                        );
                        emit_staged_terminal_done(&mut sink, &pending);
                    }
                }
                (
                    String::from_utf8_lossy(&sink).into_owned(),
                    visible,
                    Ok(fin),
                    stopped || stopped_flag,
                    raw,
                    pos,
                )
            }
            Err(err) => (
                String::from_utf8_lossy(&sink).into_owned(),
                String::new(),
                Err(err),
                stopped || stopped_flag,
                raw,
                pos,
            ),
        }
    }

    fn parse_jsonl(out: &str) -> Vec<serde_json::Value> {
        out.lines()
            .filter(|l| !l.trim().is_empty())
            .map(|l| serde_json::from_str(l).unwrap_or_else(|e| panic!("bad jsonl {l}: {e}")))
            .collect()
    }

    #[test]
    fn tool_free_producer_keeps_tool_like_text_as_content() {
        set_active_attempt_id(17);
        let text = "<tool_call>\n<function=cat\n</function>\n</tool_call>";
        let mut producer =
            QwenArSemanticProducer::new_with_tool_protocol("tool-free", false, false);
        let mut sink = Vec::new();
        let mut conversation_tokens = Vec::new();
        let mut streamed_tokens = Vec::new();
        let mut seq_pos = 0usize;

        let stopped = producer
            .commit_and_observe(
                &mut sink,
                &mut conversation_tokens,
                &mut streamed_tokens,
                &mut seq_pos,
                1000,
                text.as_bytes(),
            )
            .expect("tool-free marker text must not fail");
        assert!(!stopped);
        assert_eq!(producer.visible(), text);

        let (finish, visible) = producer.finish(&mut sink, false).expect("finish");
        assert_eq!(finish.finish_reason, "stop");
        assert!(finish.wire_tool_calls.is_empty());
        assert_eq!(visible, text);
    }

    #[test]
    fn contract_version_constant_is_v2() {
        assert_eq!(QWEN_AR_SEMANTIC_CONTRACT_VERSION, 2);
    }

    #[test]
    fn gen_start_v2_advertises_contract_version() {
        set_active_attempt_id(0);
        let mut sink = Vec::new();
        emit_gen_start(
            &mut sink,
            "req",
            true,
            Some(QWEN_AR_SEMANTIC_CONTRACT_VERSION),
        );
        let v: serde_json::Value = serde_json::from_slice(&sink).unwrap();
        assert_eq!(v["type"], "gen_start");
        assert_eq!(v["contract_version"], 2);
        assert_eq!(v["started_in_think"], true);
        assert_eq!(v["id"], "req");
        assert_eq!(v["attempt_id"], 0);
    }

    #[test]
    fn prose_only_finish_is_stop_and_stores_cache() {
        let (out, visible, fin, stopped, raw, _) =
            drive_ar_semantic_path(&["Hello world"], false, false);
        let fin = fin.expect("finish");
        assert!(!stopped);
        assert_eq!(fin.finish_reason, "stop");
        assert!(fin.wire_tool_calls.is_empty());
        assert!(fin.store_cache);
        assert_eq!(visible, "Hello world");
        assert!(out.contains("Hello world"));
        assert!(!out.contains("<think>"));
        assert_eq!(raw, vec![1000]);
        let events = parse_jsonl(&out);
        assert!(events.iter().any(|e| e["type"] == "token"));
        assert!(events.iter().all(|e| e.get("attempt_id").is_some()));
    }

    #[test]
    fn complete_tool_call_finish_is_tool_calls() {
        let chunks = [
            "Let me check.\n",
            "<tool_call>\n",
            r#"{"name":"read","arguments":{"path":"/x"}}"#,
            "\n</tool_call>",
        ];
        let (out, _visible, fin, _, _, _) = drive_ar_semantic_path(&chunks, false, false);
        let fin = fin.expect("finish");
        assert!(!out.contains("<tool_call>"));
        assert!(out.contains("Let me check."));
        assert_eq!(fin.finish_reason, "tool_calls");
        assert_eq!(fin.wire_tool_calls.len(), 1);
        assert_eq!(fin.wire_tool_calls[0].name, "read");
        assert!(fin.store_cache);
        let events = parse_jsonl(&out);
        // Authoritative calls live on staged done — no separate tool_calls event.
        assert!(events.iter().all(|e| e["type"] != "tool_calls"));
        let done = events
            .iter()
            .find(|e| e["type"] == "done" && e["finish_reason"] == "tool_calls")
            .expect("done with tool_calls");
        assert_eq!(done["calls"].as_array().unwrap().len(), 1);
        assert_eq!(done["calls"][0]["name"], "read");
        assert!(events.iter().all(|e| e["attempt_id"] == 7));
    }

    #[test]
    fn length_cap_suppresses_calls_even_if_complete() {
        let (out, _, fin, _, _, _) = drive_ar_semantic_path(
            &[r#"hi<tool_call>{"name":"read","arguments":{"path":"/x"}}</tool_call>"#],
            false,
            true,
        );
        let fin = fin.expect("finish");
        assert_eq!(fin.finish_reason, "length");
        assert!(fin.wire_tool_calls.is_empty());
        assert!(!fin.store_cache, "every length terminal is cache-unsafe");
        assert!(!out.contains("\"type\":\"tool_calls\""));
    }

    #[test]
    fn length_cap_prose_only_no_cache() {
        let (_, _, fin, _, _, _) = drive_ar_semantic_path(&["just prose"], false, true);
        let fin = fin.expect("finish");
        assert_eq!(fin.finish_reason, "length");
        assert!(fin.wire_tool_calls.is_empty());
        assert!(!fin.store_cache);
    }

    #[test]
    fn length_cap_unclosed_span_no_calls_no_cache() {
        let (_, _, fin, _, _, _) = drive_ar_semantic_path(
            &[r#"hi<tool_call>{"name":"read","arguments":{"path":"/x"}}"#],
            false,
            true,
        );
        let fin = fin.expect("length wins");
        assert_eq!(fin.finish_reason, "length");
        assert!(fin.wire_tool_calls.is_empty());
        assert!(!fin.store_cache, "partial tool turn must not prime cache");
    }

    #[test]
    fn length_cap_partial_opener_no_calls_no_cache() {
        let (_, _, fin, _, _, _) = drive_ar_semantic_path(&["hi<tool_"], false, true);
        let fin = fin.expect("length");
        assert_eq!(fin.finish_reason, "length");
        assert!(fin.wire_tool_calls.is_empty());
        assert!(!fin.store_cache);
    }

    #[test]
    fn unclosed_without_length_is_malformed_error() {
        let (_, _, fin, _, raw, _) = drive_ar_semantic_path(
            &[r#"hi<tool_call>{"name":"read","arguments":{"path":"/x"}}"#],
            false,
            false,
        );
        let err = fin.expect_err("malformed");
        assert!(err.to_string().contains("malformed") || err.to_string().contains("unclosed"));
        assert_eq!(raw, vec![1000]);
    }

    #[test]
    fn split_marker_chunks_still_classify() {
        let chunks = [
            "pre ",
            "<tool_",
            "call>",
            r#"{"name":"bash","arguments":{"cmd":"ls"}}"#,
            "</tool_call>",
            " post",
        ];
        let (out, visible, fin, _, raw, positions) = drive_ar_semantic_path(&chunks, false, false);
        let fin = fin.expect("finish");
        assert!(!out.contains("<tool_call>"));
        assert!(out.contains("pre ") || visible.contains("pre "));
        assert_eq!(fin.finish_reason, "tool_calls");
        assert_eq!(fin.wire_tool_calls[0].name, "bash");
        assert!(
            visible.contains(" post")
                || fin.trailing_visible.iter().any(|s| s.contains("post"))
                || out.contains(" post")
        );
        assert_eq!(raw.len(), chunks.len());
        assert_eq!(positions, (0..chunks.len()).collect::<Vec<_>>());
    }

    #[test]
    fn emit_visible_token_json_shape() {
        set_active_attempt_id(3);
        let mut sink = Vec::new();
        emit_visible_token(&mut sink, "req", "hello");
        let v: serde_json::Value = serde_json::from_slice(&sink).unwrap();
        assert_eq!(v["type"], "token");
        assert_eq!(v["id"], "req");
        assert_eq!(v["text"], "hello");
        assert_eq!(v["attempt_id"], 3);
    }

    #[test]
    fn empty_body_tool_call_latches_malformed_on_push() {
        let (out, _, fin, _, raw, _) =
            drive_ar_semantic_path(&["<tool_call></tool_call>"], false, false);
        assert!(!out.contains("\"type\":\"tool_calls\""));
        assert_eq!(raw, vec![1000], "raw commit precedes classify failure");
        match fin {
            Err(e) => {
                assert!(e.to_string().contains("malformed") || e.detail().contains("empty"));
            }
            Ok(f) => {
                assert!(f.wire_tool_calls.is_empty());
                assert_ne!(f.finish_reason, "tool_calls");
            }
        }
    }

    #[test]
    fn started_in_think_routes_reasoning_until_close() {
        let (out, visible, fin, _, _, _) =
            drive_ar_semantic_path(&["hidden reasoning", "</think>answer"], true, false);
        let fin = fin.expect("finish");
        let events = parse_jsonl(&out);
        assert!(events
            .iter()
            .any(|e| { e["type"] == "reasoning" && e["text"] == "hidden reasoning" }));
        assert!(!out.contains("</think>"));
        assert!(!visible.contains("hidden"));
        assert!(visible.contains("answer") || out.contains("answer"));
        assert_eq!(fin.finish_reason, "stop");
        assert!(fin.store_cache);
    }

    #[test]
    fn paired_think_markers_route_reasoning_separately() {
        let (out, visible, fin, _, _, _) =
            drive_ar_semantic_path(&["pre ", "<think>secret</think>", " post"], false, false);
        let fin = fin.expect("finish");
        let events = parse_jsonl(&out);
        assert!(!out.contains("<think>"));
        assert!(!out.contains("</think>"));
        assert!(events
            .iter()
            .any(|e| e["type"] == "reasoning" && e["text"] == "secret"));
        assert!(!visible.contains("secret"));
        assert!(visible.contains("pre ") || out.contains("pre "));
        assert!(
            visible.contains(" post") || out.contains(" post") || !fin.trailing_visible.is_empty()
        );
        assert_eq!(fin.finish_reason, "stop");
    }

    #[test]
    fn orphan_think_closer_preserves_prose() {
        let (out, visible, fin, _, _, _) =
            drive_ar_semantic_path(&["hidden</think>answer"], false, false);
        let fin = fin.expect("finish");
        assert!(!out.contains("</think>"));
        assert!(!visible.contains("</think>"));
        assert!(visible.contains("hidden"));
        assert!(visible.contains("answer") || out.contains("answer"));
        assert_eq!(fin.finish_reason, "stop");
    }

    #[test]
    fn decoded_im_end_stops_without_emitting_marker() {
        let (out, visible, fin, stopped, _, _) =
            drive_ar_semantic_path(&["hi", "<|im_end|>"], false, false);
        assert!(stopped, "filter must signal Stop on decoded EOT");
        assert!(!out.contains("<|im_end|>"));
        assert!(!visible.contains("<|im_end|>"));
        let fin = fin.expect("finish after EOT");
        assert!(fin.wire_tool_calls.is_empty());
        assert_ne!(fin.finish_reason, "tool_calls");
    }

    #[test]
    fn decoded_endoftext_stops_without_emitting_marker() {
        let (out, visible, fin, stopped, _, _) =
            drive_ar_semantic_path(&["hi", "<|endoftext|>"], false, false);
        assert!(stopped, "aux EOT must stop");
        assert!(!out.contains("<|endoftext|>"));
        assert!(!visible.contains("<|endoftext|>"));
        let fin = fin.expect("finish after aux EOT");
        assert_eq!(fin.finish_reason, "stop");
        assert!(fin.store_cache);
    }

    #[test]
    fn stop_with_prose_same_chunk_emits_prose() {
        let (out, visible, fin, stopped, _, _) =
            drive_ar_semantic_path(&["hello<|im_end|>"], false, false);
        assert!(stopped);
        assert!(visible.contains("hello") || out.contains("hello"));
        assert!(!out.contains("<|im_end|>"));
        let fin = fin.expect("finish");
        assert_eq!(fin.finish_reason, "stop");
    }

    #[test]
    fn terminal_xor_stop_vs_tool_calls_vs_length() {
        let (_, _, fin, _, _, _) = drive_ar_semantic_path(&["ok"], false, false);
        let fin = fin.unwrap();
        assert_eq!(fin.finish_reason, "stop");
        assert!(fin.wire_tool_calls.is_empty());
        let (_, _, fin, _, _, _) = drive_ar_semantic_path(
            &[r#"x<tool_call>{"name":"a","arguments":{}}</tool_call>"#],
            false,
            false,
        );
        let fin = fin.unwrap();
        assert_eq!(fin.finish_reason, "tool_calls");
        assert!(!fin.wire_tool_calls.is_empty());
        let (_, _, fin, _, _, _) = drive_ar_semantic_path(&["ok"], false, true);
        let fin = fin.unwrap();
        assert_eq!(fin.finish_reason, "length");
        assert!(fin.wire_tool_calls.is_empty());
        assert!(!fin.store_cache);
    }

    #[test]
    fn raw_commit_before_classify_is_producer_owned() {
        set_active_attempt_id(9);
        let mut producer = QwenArSemanticProducer::new("t1", false);
        let mut sink = Vec::new();
        let mut conversation_tokens = Vec::new();
        let mut streamed_tokens = Vec::new();
        let mut seq_pos = 10usize;
        let err = producer
            .commit_and_observe(
                &mut sink,
                &mut conversation_tokens,
                &mut streamed_tokens,
                &mut seq_pos,
                42,
                b"<tool_call></tool_call>",
            )
            .expect_err("empty tool body fails closed on classify");
        assert!(err.to_string().contains("malformed") || err.detail().contains("empty"));
        assert_eq!(conversation_tokens, vec![42]);
        assert_eq!(streamed_tokens, vec![42]);
        assert_eq!(seq_pos, 11);
        assert_eq!(producer.raw_committed, vec![42]);
        assert_eq!(producer.raw_commit_positions, vec![0]);
    }

    #[test]
    fn eos_filter_config_delegates_think_and_keeps_both_terminators() {
        let cfg = qwen_ar_eos_filter_config();
        assert!(!cfg.strip_think);
        assert!(!cfg.started_in_think);
        assert!(cfg.stop_at.contains(&b"<|im_end|>".to_vec()));
        assert!(cfg.stop_at.contains(&b"<|endoftext|>".to_vec()));
    }

    #[test]
    fn cancellation_transcript_carries_attempt_id() {
        set_active_attempt_id(42);
        let mut sink = Vec::new();
        emit_qwen_ar_cancelled(&mut sink, "req-1", 3);
        let events = parse_jsonl(&String::from_utf8(sink).unwrap());
        assert_eq!(events.len(), 2);
        assert_eq!(events[0]["type"], "aborted");
        assert_eq!(events[0]["reason"], "client_cancelled");
        assert_eq!(events[0]["attempt_id"], 42);
        assert_eq!(events[1]["type"], "done");
        assert_eq!(events[1]["finish_reason"], "aborted");
        assert_eq!(events[1]["attempt_id"], 42);
        assert_eq!(events[1]["completion_tokens"], 3);
    }

    #[test]
    fn info_event_carries_attempt_id() {
        set_active_attempt_id(11);
        let mut sink = Vec::new();
        emit_qwen_ar_info(
            &mut sink,
            "req",
            "budget_alert skipped: not enough KV headroom",
        );
        let v: serde_json::Value = serde_json::from_slice(&sink).unwrap();
        assert_eq!(v["type"], "info");
        assert_eq!(v["attempt_id"], 11);
        assert_eq!(v["id"], "req");
    }

    #[test]
    fn empty_commit_hold_does_not_panic() {
        let mut producer = QwenArSemanticProducer::new("t1", false);
        let mut sink = Vec::new();
        let stop = producer
            .commit_and_classify(&mut sink, 0, || (0, Vec::<u8>::new()), |_pos, _out| {})
            .unwrap();
        assert!(!stop);
        assert!(sink.is_empty());
    }

    #[test]
    fn runtime_error_path_preserves_prior_raw_commits() {
        set_active_attempt_id(5);
        let mut producer = QwenArSemanticProducer::new("t1", false);
        let mut sink = Vec::new();
        let mut conversation_tokens = Vec::new();
        let mut streamed_tokens = Vec::new();
        let mut seq_pos = 0usize;
        producer
            .commit_and_observe(
                &mut sink,
                &mut conversation_tokens,
                &mut streamed_tokens,
                &mut seq_pos,
                1,
                b"hello ",
            )
            .unwrap();
        let err = producer
            .commit_and_observe(
                &mut sink,
                &mut conversation_tokens,
                &mut streamed_tokens,
                &mut seq_pos,
                2,
                b"<tool_call></tool_call>",
            )
            .expect_err("malformed");
        assert!(!err.to_string().is_empty());
        assert_eq!(producer.raw_committed, vec![1, 2]);
        assert_eq!(conversation_tokens, vec![1, 2]);
        assert!(String::from_utf8_lossy(&sink).contains("hello"));
    }

    #[test]
    fn eos_trailing_marker_prefix_prose_flushed_and_cacheable() {
        // Finding 1: ordinary trailing marker-prefix prose (`answer <`, partial
        // `<|im_`) flushes at true EOS and shares the production finalizer/cache.
        let (out, visible, fin, _, _, _) = drive_ar_semantic_path(&["answer <"], false, false);
        let fin = fin.expect("finish");
        assert!(visible.contains("answer <") || out.contains("answer <"));
        assert_eq!(fin.finish_reason, "stop");
        assert!(fin.store_cache);
        assert_eq!(fin.cause, QwenArTerminalCause::NaturalStop);

        let mut sink = HashMap::new();
        let action = qwen_ar_cache_action(&fin, &visible);
        assert!(action.store);
        let fp = qwen_ar_apply_cache_action(
            |k, v| {
                sink.insert(k, v);
            },
            &action,
            vec![7, 8, 9],
        );
        assert!(fp.is_some());
        assert_eq!(sink.get(&fp.unwrap()).unwrap(), &vec![7, 8, 9]);

        let (out2, visible2, fin2, _, _, _) =
            drive_ar_semantic_path(&["hi", "<|im_"], false, false);
        let fin2 = fin2.expect("finish partial im prefix");
        assert!(
            visible2.contains("hi") && (visible2.contains("<|im_") || out2.contains("<|im_")),
            "partial im prefix must flush as prose: visible={visible2:?} out={out2:?}"
        );
        assert_eq!(fin2.finish_reason, "stop");
        assert!(fin2.store_cache);
    }

    /// Every nonempty proper prefix of every watched think/EOT marker.
    fn qwen_ar_watched_markers() -> &'static [&'static str] {
        &["<think>", "</think>", "<|im_end|>", "<|endoftext|>"]
    }

    fn nonempty_proper_prefixes(marker: &str) -> Vec<&str> {
        (1..marker.len()).map(|n| &marker[..n]).collect()
    }

    #[test]
    fn table_producer_finish_natural_eos_and_length_every_watched_marker_prefix() {
        // Fix round 5: drive watched-prefix finalization through production
        // `QwenArSemanticProducer::finish` for both natural EOS and length —
        // not two identical filter-only calls. Retain completed-marker suppression.
        let prose = "answer ";
        for marker in qwen_ar_watched_markers() {
            for prefix in nonempty_proper_prefixes(marker) {
                let chunk = format!("{prose}{prefix}");
                // Natural EOS (`hit_length_cap = false`).
                let (out, visible, fin, stopped, _, _) =
                    drive_ar_semantic_path(&[&chunk], false, false);
                let fin = fin.expect("natural EOS finish");
                assert!(
                    !stopped,
                    "proper prefix must not complete stop: marker={marker:?} prefix={prefix:?}"
                );
                assert_eq!(fin.cause, QwenArTerminalCause::NaturalStop);
                assert_eq!(fin.finish_reason, "stop");
                assert!(fin.store_cache);
                assert!(
                    visible.contains(prose) && visible.contains(prefix),
                    "natural EOS must flush proper prefix as prose via finish: \
                     marker={marker:?} prefix={prefix:?} visible={visible:?} out={out:?}"
                );

                // Length finalization (`hit_length_cap = true`) — distinct terminal cause.
                let (out_len, visible_len, fin_len, stopped_len, _, _) =
                    drive_ar_semantic_path(&[&chunk], false, true);
                let fin_len = fin_len.expect("length finish");
                assert!(
                    !stopped_len,
                    "proper prefix must not complete stop under length: marker={marker:?}"
                );
                assert_eq!(fin_len.cause, QwenArTerminalCause::LengthCap);
                assert_eq!(fin_len.finish_reason, "length");
                assert!(!fin_len.store_cache);
                assert!(
                    visible_len.contains(prose) && visible_len.contains(prefix),
                    "length finish must also flush proper prefix prose: \
                     marker={marker:?} prefix={prefix:?} visible={visible_len:?} out={out_len:?}"
                );
            }

            // Completed marker suppression through production finish path.
            let full = format!("{prose}{marker}");
            let (out, visible, fin, stopped, _, _) = drive_ar_semantic_path(&[&full], false, false);
            let fin = fin.expect("completed marker finish");
            assert!(
                !visible.contains(marker) && !out.contains(marker),
                "completed marker must be suppressed: marker={marker:?} visible={visible:?}"
            );
            if *marker == "<think>" {
                // Open think after prose → validation terminal (no cache).
                assert_eq!(fin.cause, QwenArTerminalCause::OpenThink);
                assert_eq!(fin.finish_reason, "error");
                assert!(!fin.store_cache);
            } else if *marker == "</think>" {
                // Orphan closer drops closer, keeps prose; not a stop marker.
                assert!(!stopped);
                assert!(visible.contains("answer") || out.contains("answer"));
                assert_eq!(fin.finish_reason, "stop");
            } else {
                // EOT completed markers stop and emit only preceding prose.
                assert!(stopped, "EOT completed marker must stop: {marker}");
                assert!(
                    visible == prose || visible.trim_end() == "answer" || out.contains("answer"),
                    "EOT emits only preceding prose: visible={visible:?}"
                );
                assert_eq!(fin.finish_reason, "stop");
            }
        }
    }

    #[test]
    fn open_think_is_fail_closed_validation_no_cache() {
        // Open think streams reasoning, then fails closed: no calls/done/cache.
        let (out, visible, fin, _, _, _) = drive_ar_semantic_path(&["still thinking"], true, false);
        let fin = fin.expect("open think returns Ok finish with error cause");
        assert_eq!(fin.cause, QwenArTerminalCause::OpenThink);
        assert_eq!(fin.finish_reason, "error");
        assert!(fin.wire_tool_calls.is_empty());
        assert!(!fin.store_cache);
        assert!(visible.is_empty());
        let events = parse_jsonl(&out);
        assert!(events
            .iter()
            .any(|e| { e["type"] == "reasoning" && e["text"] == "still thinking" }));
        assert!(!out.contains("<think>"));
        assert!(
            events
                .iter()
                .any(|e| e["type"] == "error" && e["class"] == "validation"),
            "expected validation error: {out}"
        );
        assert!(
            events.iter().all(|e| e["type"] != "done"),
            "open-think must not emit done (terminal XOR): {out}"
        );
        let errors: Vec<_> = events.iter().filter(|e| e["type"] == "error").collect();
        assert_eq!(errors.len(), 1, "exactly one error terminal: {out}");
        assert_eq!(errors[0]["class"], "validation");
        assert_eq!(errors[0]["retryable"], false);
        assert_eq!(errors[0]["attempt_id"], 7);
        // No unread stale event after the single terminal error.
        let err_idx = events.iter().position(|e| e["type"] == "error").unwrap();
        assert_eq!(
            err_idx,
            events.len() - 1,
            "error must be the last event (no stale unread after terminal): {out}"
        );
        assert!(events.iter().all(|e| e.get("attempt_id").is_some()));

        let action = qwen_ar_cache_action(&fin, &visible);
        assert!(!action.store);
        let mut sink = HashMap::new();
        assert!(qwen_ar_apply_cache_action(
            |k, v| {
                sink.insert(k, v);
            },
            &action,
            vec![1],
        )
        .is_none());
        assert!(sink.is_empty());
    }

    #[test]
    fn open_think_unmatched_generated_think_fail_closed() {
        let (out, visible, fin, _, _, _) =
            drive_ar_semantic_path(&["pre ", "<think>secret"], false, false);
        let fin = fin.expect("open think");
        let events = parse_jsonl(&out);
        assert_eq!(fin.cause, QwenArTerminalCause::OpenThink);
        assert!(!fin.store_cache);
        assert!(visible.is_empty() || !visible.contains("secret"));
        assert!(events
            .iter()
            .any(|e| e["type"] == "reasoning" && e["text"] == "secret"));
        assert!(!out.contains("\"type\":\"tool_calls\""));
    }

    #[test]
    fn decoded_eot_beats_length_on_final_budget_token_primary() {
        // Finding 3: primary EOT on final budget token beats length, cache-safe stop.
        let (out, visible, fin, stopped, _, _) =
            drive_ar_semantic_path(&["hi", "<|im_end|>"], false, true);
        assert!(stopped);
        let fin = fin.expect("finish");
        assert_eq!(fin.cause, QwenArTerminalCause::DecodedEot);
        assert_eq!(fin.finish_reason, "stop");
        assert!(fin.store_cache);
        assert!(fin.wire_tool_calls.is_empty());
        assert!(!out.contains("<|im_end|>"));
        assert!(visible.contains("hi") || out.contains("hi"));

        let mut sink = HashMap::new();
        let action = qwen_ar_cache_action(&fin, &visible);
        assert!(action.store);
        let fp = qwen_ar_apply_cache_action(
            |k, v| {
                sink.insert(k, v);
            },
            &action,
            vec![42],
        )
        .expect("store");
        assert_eq!(sink.get(&fp).unwrap(), &vec![42]);
    }

    #[test]
    fn decoded_eot_beats_length_on_final_budget_token_aux() {
        let (out, _, fin, stopped, _, _) =
            drive_ar_semantic_path(&["hi", "<|endoftext|>"], false, true);
        assert!(stopped);
        let fin = fin.expect("finish");
        assert_eq!(fin.cause, QwenArTerminalCause::DecodedEot);
        assert_eq!(fin.finish_reason, "stop");
        assert!(fin.store_cache);
        assert!(!out.contains("<|endoftext|>"));
    }

    #[test]
    fn decoded_eot_beats_length_with_complete_buffered_call() {
        let chunks = [
            r#"pre<tool_call>{"name":"read","arguments":{"path":"/x"}}</tool_call>"#,
            "<|im_end|>",
        ];
        let (out, _, fin, stopped, _, _) = drive_ar_semantic_path(&chunks, false, true);
        assert!(stopped);
        let fin = fin.expect("finish");
        assert_eq!(fin.cause, QwenArTerminalCause::DecodedEot);
        assert_eq!(fin.finish_reason, "tool_calls");
        assert_eq!(fin.wire_tool_calls.len(), 1);
        assert_eq!(fin.wire_tool_calls[0].name, "read");
        assert!(fin.store_cache);
        // Authoritative calls live on staged done, not a separate tool_calls event.
        assert!(!out.contains("\"type\":\"tool_calls\""));
        assert!(out.contains("\"type\":\"done\""));
        assert!(out.contains("\"finish_reason\":\"tool_calls\""));
        assert!(out.contains("\"name\":\"read\""));
        // Pure length without EOT would suppress the same complete call.
        let (out_len, _, fin_len, _, _, _) = drive_ar_semantic_path(
            &[r#"pre<tool_call>{"name":"read","arguments":{"path":"/x"}}</tool_call>"#],
            false,
            true,
        );
        let fin_len = fin_len.expect("length");
        assert_eq!(fin_len.cause, QwenArTerminalCause::LengthCap);
        assert!(fin_len.wire_tool_calls.is_empty());
        assert!(!fin_len.store_cache);
        assert!(!out_len.contains("\"type\":\"tool_calls\""));
        assert!(!out_len.contains("\"finish_reason\":\"tool_calls\""));
    }

    #[test]
    fn terminal_cause_resolve_priority() {
        assert_eq!(
            QwenArTerminalCause::resolve(true, true, true),
            QwenArTerminalCause::OpenThink
        );
        assert_eq!(
            QwenArTerminalCause::resolve(true, true, false),
            QwenArTerminalCause::DecodedEot
        );
        assert_eq!(
            QwenArTerminalCause::resolve(false, true, false),
            QwenArTerminalCause::LengthCap
        );
        assert_eq!(
            QwenArTerminalCause::resolve(false, false, false),
            QwenArTerminalCause::NaturalStop
        );
    }

    #[test]
    fn real_writers_hostile_request_ids() {
        // Finding 5: shared serde writers + hostile IDs.
        set_active_attempt_id(99);
        let hostile = r#"req"}\n{"type":"pwned"#;
        let mut sink = Vec::new();
        emit_gen_start(&mut sink, hostile, false, Some(2));
        emit_visible_token(&mut sink, hostile, "ok");
        emit_tool_calls_event(
            &mut sink,
            hostile,
            &[hipfire_runtime::prompt_frame::ToolCall {
                id: None,
                name: "n".into(),
                arguments: serde_json::json!({}),
                rendered_body: None,
            }],
        );
        emit_qwen_ar_done(
            &mut sink, hostile, "stop", 1, 1.0, 0, 0.0, 0.0, 1.0, 0.0, 0, "",
        );
        let events = parse_jsonl(&String::from_utf8(sink).unwrap());
        assert_eq!(events.len(), 4);
        for e in &events {
            assert_eq!(e["id"], hostile);
            assert_eq!(e["attempt_id"], 99);
        }
        assert_eq!(events[0]["type"], "gen_start");
        assert_eq!(events[1]["type"], "token");
        assert_eq!(events[2]["type"], "tool_calls");
        assert_eq!(events[3]["type"], "done");
        assert_eq!(events[3]["finish_reason"], "stop");
    }

    #[test]
    fn cancellation_json_through_semantic_fold_contract() {
        // Finding 6: cancel JSON transcript is valid contract-v2 fold input.
        set_active_attempt_id(42);
        let mut sink = Vec::new();
        emit_gen_start(
            &mut sink,
            "c1",
            false,
            Some(QWEN_AR_SEMANTIC_CONTRACT_VERSION),
        );
        emit_visible_token(&mut sink, "c1", "partial ");
        emit_qwen_ar_cancelled(&mut sink, "c1", 1);
        let events = parse_jsonl(&String::from_utf8(sink).unwrap());
        assert_eq!(events[0]["type"], "gen_start");
        assert_eq!(events[0]["contract_version"], 2);
        assert_eq!(events[1]["type"], "token");
        assert_eq!(events[2]["type"], "aborted");
        assert_eq!(events[2]["reason"], "client_cancelled");
        assert_eq!(events[3]["type"], "done");
        assert_eq!(events[3]["finish_reason"], "aborted");
        for e in &events {
            assert_eq!(e["attempt_id"], 42);
            assert_eq!(e["id"], "c1");
        }
    }

    #[test]
    fn marker_byte_splits_enumerate_all_boundaries() {
        // Finding 6: enumerate every byte split for think open/close + tool markers.
        let open = b"<think>";
        let close = b"</think>";
        let tool_open = b"<tool_call>";
        let tool_close = b"</tool_call>";
        // Paired think: split open and close independently, always complete the pair.
        for split in 1..open.len() {
            let left = std::str::from_utf8(&open[..split]).unwrap();
            let right = std::str::from_utf8(&open[split..]).unwrap();
            let chunks = ["pre ", left, right, "secret", "</think>", " post"];
            let (out, visible, fin, _, _, _) = drive_ar_semantic_path(&chunks, false, false);
            let fin = fin.expect("finish");
            assert!(!out.contains("<think>"), "open split={split}");
            let events = parse_jsonl(&out);
            assert!(events
                .iter()
                .any(|e| e["type"] == "reasoning" && e["text"] == "secret"));
            assert_eq!(fin.finish_reason, "stop");
            assert!(visible.contains("pre ") || out.contains("pre "));
            assert!(
                visible.contains(" post")
                    || out.contains(" post")
                    || !fin.trailing_visible.is_empty()
            );
        }
        for split in 1..close.len() {
            let left = std::str::from_utf8(&close[..split]).unwrap();
            let right = std::str::from_utf8(&close[split..]).unwrap();
            let chunks = ["pre ", "<think>secret", left, right, " post"];
            let (out, visible, fin, _, _, _) = drive_ar_semantic_path(&chunks, false, false);
            let fin = fin.expect("finish");
            assert!(!out.contains("</think>"), "close split={split}");
            let events = parse_jsonl(&out);
            assert!(events
                .iter()
                .any(|e| e["type"] == "reasoning" && e["text"] == "secret"));
            assert_eq!(fin.finish_reason, "stop");
            assert!(visible.contains("pre ") || out.contains("pre "));
        }

        for marker in [tool_open.as_slice(), tool_close.as_slice()] {
            for split in 1..marker.len() {
                let left = std::str::from_utf8(&marker[..split]).unwrap();
                let right = std::str::from_utf8(&marker[split..]).unwrap();
                let body = r#"{"name":"bash","arguments":{"cmd":"ls"}}"#;
                let chunks = if marker == tool_open {
                    ["pre ", left, right, body, "</tool_call>"]
                } else {
                    ["pre ", "<tool_call>", body, left, right]
                };
                let (out, _, fin, _, _, _) = drive_ar_semantic_path(&chunks, false, false);
                let fin = fin.expect("finish");
                assert_eq!(fin.finish_reason, "tool_calls", "split={split} out={out}");
                assert_eq!(fin.wire_tool_calls[0].name, "bash");
                assert!(!out.contains("<tool_call>"));
            }
        }

        // Primary + aux EOT byte splits.
        for marker in [b"<|im_end|>".as_slice(), b"<|endoftext|>".as_slice()] {
            for split in 1..marker.len() {
                let left = std::str::from_utf8(&marker[..split]).unwrap();
                let right = std::str::from_utf8(&marker[split..]).unwrap();
                let (out, visible, fin, stopped, _, _) =
                    drive_ar_semantic_path(&["hi", left, right], false, false);
                assert!(
                    stopped,
                    "EOT split={split} marker={marker:?} must stop; out={out}"
                );
                let fin = fin.expect("finish");
                assert_eq!(fin.cause, QwenArTerminalCause::DecodedEot);
                assert!(!visible.contains("<|"));
                assert!(!out.contains("<|im_end|>"));
                assert!(!out.contains("<|endoftext|>"));
            }
        }
    }

    #[test]
    fn cache_sink_mutation_seam_store_and_skip() {
        // Finding 6: real cache sink mutation, not only store_cache bool.
        let (_, visible, fin, _, _, _) = drive_ar_semantic_path(&["Hello world"], false, false);
        let fin = fin.expect("stop");
        let action = qwen_ar_cache_action(&fin, &visible);
        let mut sink = HashMap::new();
        let fp = qwen_ar_apply_cache_action(
            |k, v| {
                sink.insert(k, v);
            },
            &action,
            vec![1, 2, 3],
        )
        .expect("store");
        assert_eq!(sink.len(), 1);
        assert_eq!(sink[&fp], vec![1, 2, 3]);

        let (_, _, fin_len, _, _, _) = drive_ar_semantic_path(&["Hello world"], false, true);
        let fin_len = fin_len.expect("length");
        let action_len = qwen_ar_cache_action(&fin_len, "Hello world");
        assert!(!action_len.store);
        assert!(qwen_ar_apply_cache_action(
            |k, v| {
                sink.insert(k, v);
            },
            &action_len,
            vec![9],
        )
        .is_none());
        assert_eq!(sink.len(), 1, "length must not mutate sink");
    }

    #[test]
    fn commit_and_classify_is_sole_production_entry() {
        // Finding 4: tests exercise the exact commit-then-classify op with
        // on_committed callback ordering (raw stamp before committed emit).
        set_active_attempt_id(3);
        let mut producer = QwenArSemanticProducer::new("t1", false);
        let mut sink = Vec::new();
        let mut conversation_tokens = Vec::new();
        let mut streamed_tokens = Vec::new();
        let mut seq_pos = 0usize;
        let mut committed_positions = Vec::new();
        let stop = producer
            .commit_and_classify(
                &mut sink,
                11,
                || {
                    let pos = qwen_ar_raw_commit_token(
                        &mut conversation_tokens,
                        &mut streamed_tokens,
                        &mut seq_pos,
                        11,
                        QwenArRawCommitDisposition::ClassifiedVisible,
                    );
                    (pos, b"hello".to_vec())
                },
                |pos, out| {
                    committed_positions.push(pos);
                    let _ = writeln!(out, "{}", serde_json::json!({"type":"committed","pos":pos}));
                },
            )
            .unwrap();
        assert!(!stop);
        assert_eq!(producer.raw_committed, vec![11]);
        assert_eq!(producer.raw_commit_positions, vec![0]);
        assert_eq!(committed_positions, vec![0]);
        let events = parse_jsonl(&String::from_utf8_lossy(&sink));
        assert_eq!(events[0]["type"], "committed");
        assert!(events
            .iter()
            .any(|e| e["type"] == "token" && e["text"] == "hello"));
    }

    #[test]
    fn open_think_terminal_xor_error_only_no_done_no_stale() {
        // Fix round 4 #1: open-think → exactly one correlated non-retryable
        // validation error, no done, no unread stale event after terminal.
        // GPU-less: attest epilogue.rolled_back=false (same writer as production).
        set_active_attempt_id(7);
        let mut sink = Vec::new();
        let ep = hipfire_generate::common::RollbackEpilogue {
            rolled_back: false,
            context: None,
        };
        emit_qwen_ar_open_think_terminal(&mut sink, "ot1", 4, &ep);
        let events = parse_jsonl(&String::from_utf8(sink).unwrap());
        assert_eq!(events.len(), 1, "exactly one terminal event: {events:?}");
        assert_eq!(events[0]["type"], "error");
        assert_eq!(events[0]["class"], "validation");
        assert_eq!(events[0]["retryable"], false);
        assert_eq!(events[0]["rolled_back"], false);
        assert_eq!(events[0]["attempt_id"], 7);
        assert_eq!(events[0]["id"], "ot1");
        assert!(
            events[0]["message"]
                .as_str()
                .unwrap_or("")
                .contains("open think"),
            "message={:?}",
            events[0]["message"]
        );
    }

    #[test]
    fn raw_commit_dispositions_exactly_once_visible_and_hidden() {
        // Fix round 4 #2: parameterized disposition; trailer stays client-invisible;
        // exactly-once state mutation across production token path dispositions.
        let mut conversation_tokens = Vec::new();
        let mut streamed_tokens = Vec::new();
        let mut seq_pos = 10usize;

        let pos_v = qwen_ar_raw_commit_token(
            &mut conversation_tokens,
            &mut streamed_tokens,
            &mut seq_pos,
            100,
            QwenArRawCommitDisposition::ClassifiedVisible,
        );
        assert_eq!(pos_v, 0);
        assert_eq!(conversation_tokens, vec![100]);
        assert_eq!(streamed_tokens, vec![100]);
        assert_eq!(seq_pos, 11);

        let pos_h = qwen_ar_raw_commit_token(
            &mut conversation_tokens,
            &mut streamed_tokens,
            &mut seq_pos,
            200,
            QwenArRawCommitDisposition::IntentionallyHidden,
        );
        assert_eq!(pos_h, 1, "hidden returns conversation index");
        assert_eq!(conversation_tokens, vec![100, 200]);
        assert_eq!(
            streamed_tokens,
            vec![100],
            "hidden trailer must not join streamed/client path"
        );
        assert_eq!(seq_pos, 12);

        // Second visible after hidden still only streams the visible tokens.
        let pos_v2 = qwen_ar_raw_commit_token(
            &mut conversation_tokens,
            &mut streamed_tokens,
            &mut seq_pos,
            300,
            QwenArRawCommitDisposition::ClassifiedVisible,
        );
        assert_eq!(pos_v2, 1);
        assert_eq!(conversation_tokens, vec![100, 200, 300]);
        assert_eq!(streamed_tokens, vec![100, 300]);
        assert_eq!(seq_pos, 13);

        // Producer path: visible classify + hidden trailer via sole commit_raw.
        set_active_attempt_id(3);
        let mut producer = QwenArSemanticProducer::new("t1", false);
        let mut sink = Vec::new();
        let mut conv = Vec::new();
        let mut stream = Vec::new();
        let mut sp = 0usize;
        producer
            .commit_and_observe(&mut sink, &mut conv, &mut stream, &mut sp, 11, b"hi")
            .unwrap();
        // Post-EOT hidden trailer through the same producer-owned entry.
        producer
            .commit_raw(
                &mut sink,
                99,
                QwenArRawCommitDisposition::IntentionallyHidden,
                || {
                    let tpos = qwen_ar_raw_commit_token(
                        &mut conv,
                        &mut stream,
                        &mut sp,
                        99,
                        QwenArRawCommitDisposition::IntentionallyHidden,
                    );
                    (tpos, Vec::<u8>::new())
                },
                |_pos, _out| {},
            )
            .unwrap();
        assert_eq!(producer.raw_committed, vec![11, 99]);
        assert_eq!(producer.raw_commit_positions.len(), 2);
        assert_eq!(conv, vec![11, 99]);
        assert_eq!(stream, vec![11], "trailer not streamed");
        let out = String::from_utf8_lossy(&sink);
        assert!(out.contains("hi"));
        assert!(!out.contains("99"));
        // finish must not surface hidden trailer as visible.
        let (fin, visible) = producer.finish(&mut sink, false).expect("finish");
        assert_eq!(fin.finish_reason, "stop");
        assert_eq!(visible, "hi");
    }

    #[test]
    fn wire_helpers_used_by_gen_start_and_cancel_writers() {
        // Fix round 4 #3: production writers use shared semantic wire helpers.
        set_active_attempt_id(42);
        let mut sink = Vec::new();
        emit_gen_start(
            &mut sink,
            "c1",
            false,
            Some(QWEN_AR_SEMANTIC_CONTRACT_VERSION),
        );
        emit_qwen_ar_cancelled(&mut sink, "c1", 1);
        let events = parse_jsonl(&String::from_utf8(sink).unwrap());
        assert_eq!(
            events[0],
            hipfire_runtime::semantic::wire_gen_start("c1", false, 42, Some(2))
        );
        assert_eq!(
            events[1],
            hipfire_runtime::semantic::wire_aborted("c1", "client_cancelled", 42)
        );
        assert_eq!(
            events[2],
            hipfire_runtime::semantic::wire_aborted_done("c1", 1, 42)
        );
    }

    #[test]
    fn client_commit_effects_commit_preserves_intended_flags() {
        let e = hipfire_generate::qwen::qwen_client_commit_effects(ClientTerminalDecision::Commit, true, true);
        assert_eq!(
            e,
            hipfire_generate::qwen::QwenClientCommitEffects {
                release_tool_calls: true,
                store_cache: true,
                emit_done: true,
            }
        );
        let e = hipfire_generate::qwen::qwen_client_commit_effects(ClientTerminalDecision::Commit, false, true);
        assert!(!e.release_tool_calls);
        assert!(e.store_cache);
        assert!(e.emit_done);
        let e = hipfire_generate::qwen::qwen_client_commit_effects(ClientTerminalDecision::Commit, false, false);
        assert!(!e.release_tool_calls);
        assert!(!e.store_cache);
        assert!(e.emit_done);
    }

    #[test]
    fn client_commit_effects_abort_suppresses_all() {
        let e = hipfire_generate::qwen::qwen_client_commit_effects(ClientTerminalDecision::Abort, true, true);
        assert_eq!(
            e,
            hipfire_generate::qwen::QwenClientCommitEffects {
                release_tool_calls: false,
                store_cache: false,
                emit_done: false,
            }
        );
    }

    #[test]
    fn finish_defers_tool_calls_until_commit_effects() {
        set_active_attempt_id(11);
        let mut producer = QwenArSemanticProducer::new("t-commit", false);
        let mut sink = Vec::new();
        let mut conv = Vec::new();
        let mut stream = Vec::new();
        let mut pos = 0usize;
        let chunks = [
            "Let me check.\n",
            "<tool_call>\n",
            r#"{"name":"read","arguments":{"path":"/x"}}"#,
            "\n</tool_call>",
        ];
        for (i, c) in chunks.iter().enumerate() {
            producer
                .commit_and_observe(
                    &mut sink,
                    &mut conv,
                    &mut stream,
                    &mut pos,
                    2000 + i as u32,
                    c.as_bytes(),
                )
                .unwrap();
        }
        let (fin, visible) = producer.finish(&mut sink, false).expect("finish");
        assert_eq!(fin.finish_reason, "tool_calls");
        assert_eq!(fin.wire_tool_calls.len(), 1);
        assert!(visible.contains("Let me check."));
        let pre = String::from_utf8_lossy(&sink);
        assert!(
            !pre.contains("\"type\":\"tool_calls\""),
            "finish must not release tool_calls before Commit"
        );

        // Commit path: release + cache + done.
        let effects = hipfire_generate::qwen::qwen_client_commit_effects(
            ClientTerminalDecision::Commit,
            fin.finish_reason == "tool_calls" && !fin.wire_tool_calls.is_empty(),
            fin.store_cache,
        );
        assert!(effects.release_tool_calls && effects.store_cache && effects.emit_done);
        let mut cache = HashMap::new();
        let action = qwen_ar_cache_action(&fin, &visible);
        assert!(action.store);
        let fp = qwen_ar_apply_cache_action(
            |k, v| {
                cache.insert(k, v);
            },
            &action,
            vec![1, 2, 3],
        );
        assert!(fp.is_some());
        assert_eq!(cache.len(), 1);
        let mut pending = qwen_ar_done_value(
            "t-commit",
            fin.finish_reason,
            4,
            1.0,
            0,
            0.0,
            0.0,
            1.0,
            0.0,
            0,
            "",
        );
        stage_terminal_tool_calls(&mut pending, fin.finish_reason, &fin.wire_tool_calls);
        emit_staged_terminal_done(&mut sink, &pending);
        let events = parse_jsonl(&String::from_utf8_lossy(&sink));
        assert!(events.iter().all(|e| e["type"] != "tool_calls"));
        let done = events
            .iter()
            .find(|e| e["type"] == "done" && e["finish_reason"] == "tool_calls")
            .expect("done tool_calls");
        assert!(done["calls"].is_array());
        assert_eq!(done["calls"].as_array().unwrap().len(), 1);
        assert!(events
            .iter()
            .all(|e| e.get("type") != Some(&serde_json::json!("aborted"))));
    }

    #[test]
    fn abort_effects_suppress_calls_cache_and_normal_done() {
        set_active_attempt_id(12);
        let mut producer = QwenArSemanticProducer::new("t-abort", false);
        let mut sink = Vec::new();
        let mut conv = Vec::new();
        let mut stream = Vec::new();
        let mut pos = 0usize;
        let chunks = [
            "Let me check.\n",
            "<tool_call>\n",
            r#"{"name":"read","arguments":{"path":"/x"}}"#,
            "\n</tool_call>",
        ];
        for (i, c) in chunks.iter().enumerate() {
            producer
                .commit_and_observe(
                    &mut sink,
                    &mut conv,
                    &mut stream,
                    &mut pos,
                    3000 + i as u32,
                    c.as_bytes(),
                )
                .unwrap();
        }
        let (fin, visible) = producer.finish(&mut sink, false).expect("finish");
        assert_eq!(fin.finish_reason, "tool_calls");
        let effects = hipfire_generate::qwen::qwen_client_commit_effects(
            ClientTerminalDecision::Abort,
            fin.finish_reason == "tool_calls" && !fin.wire_tool_calls.is_empty(),
            fin.store_cache,
        );
        assert!(!effects.release_tool_calls && !effects.store_cache && !effects.emit_done);

        // No tool release / cache store / normal done on Abort.
        let mut cache = HashMap::new();
        let mut action = qwen_ar_cache_action(&fin, &visible);
        action.store = effects.store_cache && action.store;
        assert!(qwen_ar_apply_cache_action(
            |k, v| {
                cache.insert(k, v);
            },
            &action,
            vec![9, 9]
        )
        .is_none());
        assert!(cache.is_empty());

        // Attested cancel terminal only (no GPU rollback in unit test).
        let ep = hipfire_generate::common::RollbackEpilogue {
            rolled_back: true,
            context: None,
        };
        hipfire_generate::common::emit_spec_cancel_after_rollback(&mut sink, "t-abort", 4, &ep);
        let out = String::from_utf8_lossy(&sink);
        assert!(!out.contains("\"type\":\"tool_calls\""));
        let events = parse_jsonl(&out);
        assert!(events.iter().any(|e| e["type"] == "aborted"));
        assert!(events
            .iter()
            .any(|e| e["type"] == "done" && e["finish_reason"] == "aborted"));
        assert!(events
            .iter()
            .all(|e| !(e["type"] == "done" && e["finish_reason"] == "tool_calls")));
        assert!(events.iter().all(|e| e["attempt_id"] == 12));
    }
