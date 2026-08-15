// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Glimmer channel recorder tests.
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

    use hipfire_runtime::prompt_frame::{CachedAssistantBody, CachedAssistantToolBody};

    #[test]
    fn splits_self_then_user() {
        let mut rec = hipfire_generate::dense::GlimmerChannelRecorder::new();
        // Production shape: `add_generation_prompt` already emitted `<|start|>assistant`,
        // so the model's FIRST emission is just ` to=self` — no `<|start|>`, no `assistant`.
        rec.push(100, " to=self");
        rec.push(hipfire_generate::dense::GLIMMER_MESSAGE_ID, "<|message|>");
        rec.push(101, "reasoning body");
        rec.push(102, " more");
        rec.push(hipfire_generate::dense::GLIMMER_EOM_ID, "<|eom|>");
        rec.push(hipfire_generate::dense::GLIMMER_START_ID, "<|start|>");
        rec.push(103, "assistant to=user");
        rec.push(hipfire_generate::dense::GLIMMER_MESSAGE_ID, "<|message|>");
        rec.push(104, "answer body");
        rec.push(hipfire_generate::dense::GLIMMER_EOT_ID, "<|eot|>");
        let turn = rec.into_cached_turn(&[]).expect("should succeed");
        assert!(turn.reasoning.is_some());
        assert_eq!(turn.reasoning.unwrap().text, "reasoning body more");
        assert_eq!(turn.tools.len(), 0);
        assert!(turn.content.is_some());
        assert_eq!(turn.content.unwrap().text, "answer body");
    }

    #[test]
    fn terminal_open_user_body_is_accepted() {
        // GAP3: self closed by eom, user body left OPEN (no <|eot|> fed) must be accepted.
        let mut rec = hipfire_generate::dense::GlimmerChannelRecorder::new();
        // Production shape: `add_generation_prompt` already emitted `<|start|>assistant`,
        // so the model's FIRST emission is just ` to=self` — no `<|start|>`, no `assistant`.
        rec.push(100, " to=self");
        rec.push(hipfire_generate::dense::GLIMMER_MESSAGE_ID, "<|message|>");
        rec.push(101, "reasoning body");
        rec.push(hipfire_generate::dense::GLIMMER_EOM_ID, "<|eom|>");
        rec.push(hipfire_generate::dense::GLIMMER_START_ID, "<|start|>");
        rec.push(103, "assistant to=user");
        rec.push(hipfire_generate::dense::GLIMMER_MESSAGE_ID, "<|message|>");
        rec.push(104, "answer body");
        rec.push(105, " more");
        // Intentionally leave user body OPEN — no EOT, decode stopped on <|eot|> without feeding it.
        let turn = rec
            .into_cached_turn(&[])
            .expect("open terminal user body should be accepted");
        assert!(turn.reasoning.is_some());
        assert_eq!(turn.reasoning.unwrap().text, "reasoning body");
        assert!(turn.content.is_some());
        assert_eq!(turn.content.unwrap().text, "answer body more");
        assert!(turn.tools.is_empty());
    }

    #[test]
    fn splits_self_then_tool() {
        let mut rec = hipfire_generate::dense::GlimmerChannelRecorder::new();
        // Production shape: `add_generation_prompt` already emitted `<|start|>assistant`,
        // so the model's FIRST emission is just ` to=self` — no `<|start|>`, no `assistant`.
        rec.push(100, " to=self");
        rec.push(hipfire_generate::dense::GLIMMER_MESSAGE_ID, "<|message|>");
        rec.push(101, "reasoning");
        rec.push(hipfire_generate::dense::GLIMMER_EOM_ID, "<|eom|>");
        rec.push(hipfire_generate::dense::GLIMMER_START_ID, "<|start|>");
        rec.push(102, "assistant to=weather.get_forecast");
        rec.push(hipfire_generate::dense::GLIMMER_MESSAGE_ID, "<|message|>");
        let atem = "<atem:function_calls>\n<atem:invoke name=\"weather.get_forecast\">\n<atem:parameter name=\"location\">Paris</atem:parameter>\n</atem:invoke>\n</atem:function_calls>";
        for (i, c) in atem.chars().enumerate() {
            rec.push(200 + i as u32, &c.to_string());
        }
        rec.push(hipfire_generate::dense::GLIMMER_EOT_ID, "<|eot|>");
        let tool_call = hipfire_runtime::prompt_frame::ToolCall {
            id: Some("call_0".into()),
            name: "weather.get_forecast".into(),
            arguments: serde_json::json!({"location":"Paris"}),
            rendered_body: None,
        };
        let turn = rec.into_cached_turn(&[tool_call]).expect("should succeed");
        assert!(turn.reasoning.is_some());
        assert_eq!(turn.tools.len(), 1);
        assert_eq!(turn.tools[0].recipient, "weather.get_forecast");
        assert!(turn.content.is_none());
    }

    #[test]
    fn refuses_forced_reasoning_close() {
        let mut rec = hipfire_generate::dense::GlimmerChannelRecorder::new();
        // Production shape: `add_generation_prompt` already emitted `<|start|>assistant`,
        // so the model's FIRST emission is just ` to=self` — no `<|start|>`, no `assistant`.
        rec.push(100, " to=self");
        rec.push(hipfire_generate::dense::GLIMMER_MESSAGE_ID, "<|message|>");
        rec.push(101, "reasoning");
        rec.mark_forced_reasoning_close();
        rec.push(hipfire_generate::dense::GLIMMER_EOM_ID, "<|eom|>");
        let res = rec.into_cached_turn(&[]);
        assert_eq!(res.unwrap_err(), hipfire_generate::dense::GlimmerRecordRefusal::ForcedReasoningClose);
    }

    #[test]
    fn refuses_empty_self_body() {
        let mut rec = hipfire_generate::dense::GlimmerChannelRecorder::new();
        // Production shape: `add_generation_prompt` already emitted `<|start|>assistant`,
        // so the model's FIRST emission is just ` to=self` — no `<|start|>`, no `assistant`.
        rec.push(100, " to=self");
        rec.push(hipfire_generate::dense::GLIMMER_MESSAGE_ID, "<|message|>");
        rec.push(hipfire_generate::dense::GLIMMER_EOM_ID, "<|eom|>");
        let res = rec.into_cached_turn(&[]);
        assert_eq!(res.unwrap_err(), hipfire_generate::dense::GlimmerRecordRefusal::EmptySelfBody);
    }

    #[test]
    fn records_self_body_regardless_of_think_budget() {
        // Muse Glimmer has no non-thinking mode: the Onyx system block always carries
        // `Reasoning strength:`, so the model always opens a `to=self` channel. A low think
        // budget caps the span, it does not remove it — the turn must still be recordable, or
        // the prefix cache would go permanently inert whenever thinking was "off".
        let mut rec = hipfire_generate::dense::GlimmerChannelRecorder::new();
        rec.push(100, " to=self");
        rec.push(hipfire_generate::dense::GLIMMER_MESSAGE_ID, "<|message|>");
        rec.push(101, "reasoning");
        rec.push(hipfire_generate::dense::GLIMMER_EOM_ID, "<|eom|>");
        rec.push(hipfire_generate::dense::GLIMMER_START_ID, "<|start|>");
        rec.push(103, "assistant to=user");
        rec.push(hipfire_generate::dense::GLIMMER_MESSAGE_ID, "<|message|>");
        rec.push(104, "answer");
        let turn = rec
            .into_cached_turn(&[])
            .expect("self body must be recorded");
        assert_eq!(turn.reasoning.expect("reasoning slot").text, "reasoning");
        assert_eq!(turn.content.expect("content slot").text, "answer");
    }

    #[test]
    fn store_cached_turn_self_then_user_inserts_both_channels() {
        let mut rec = hipfire_generate::dense::GlimmerChannelRecorder::new();
        // Production shape: `add_generation_prompt` already emitted `<|start|>assistant`,
        // so the model's FIRST emission is just ` to=self` — no `<|start|>`, no `assistant`.
        rec.push(100, " to=self");
        rec.push(hipfire_generate::dense::GLIMMER_MESSAGE_ID, "<|message|>");
        rec.push(101, "reasoning body");
        rec.push(102, " more");
        rec.push(hipfire_generate::dense::GLIMMER_EOM_ID, "<|eom|>");
        rec.push(hipfire_generate::dense::GLIMMER_START_ID, "<|start|>");
        rec.push(103, "assistant to=user");
        rec.push(hipfire_generate::dense::GLIMMER_MESSAGE_ID, "<|message|>");
        rec.push(104, "answer body");
        rec.push(105, "!");
        rec.push(hipfire_generate::dense::GLIMMER_EOT_ID, "<|eot|>");
        let mut cache = hipfire_loader::AsstTurnCache::new_from_env();
        cache.clear();
        let ok = hipfire_generate::dense::glimmer_store_cached_turn(&mut cache, rec, &[], 0);
        assert!(ok, "store should succeed");
        let normalized =
            hipfire_runtime::tokenizer::maybe_normalize_prompt("answer body!").into_owned();
        let fp_raw = hipfire_generate::common::asst_turn_fingerprint(&normalized, &[]);
        let fp = hipfire_generate::dense::glimmer_turn_key(fp_raw, 0);
        let turn = cache
            .get(&fp)
            .expect("cache should contain inserted turn")
            .clone();
        assert!(turn.reasoning.is_some(), "reasoning should be Some");
        assert!(turn.content.is_some(), "content should be Some");
        assert_eq!(turn.reasoning.unwrap().token_ids, vec![101, 102]);
        assert_eq!(turn.content.unwrap().token_ids, vec![104, 105]);
        assert!(turn.tools.is_empty());
    }

    #[test]
    fn tool_channel_does_not_emit_visible_token() {
        // GAP6: to=weather.get_forecast envelope must not produce visible Token events.
        let mut router = hipfire_generate::dense::GlimmerHarmonyRouter::new(0);
        // Feed header + atem body split across fragments to exercise suffix hold logic
        let header = "<|start|>assistant to=weather.get_forecast<|message|>";
        let atem = "<atem:function_calls>\n<atem:invoke name=\"weather.get_forecast\">\n<atem:parameter name=\"location\">Paris</atem:parameter>\n</atem:invoke>\n</atem:function_calls>";
        let (events, _) = router.push(header);
        assert!(events.is_empty(), "header alone should emit nothing");
        let (events, _) = router.push(atem);
        // Tool channel text must be Tool, not Token
        let tool_text: String = events
            .iter()
            .filter_map(|e| match e {
                hipfire_generate::dense::GlimmerEmit::Tool(s) => Some(s.as_str()),
                _ => None,
            })
            .collect();
        let token_text: String = events
            .iter()
            .filter_map(|e| match e {
                hipfire_generate::dense::GlimmerEmit::Token(s) => Some(s.as_str()),
                _ => None,
            })
            .collect();
        assert!(
            token_text.is_empty(),
            "tool envelope must produce zero visible Token events, got {:?}",
            token_text
        );
        assert!(
            !tool_text.is_empty(),
            "tool envelope should produce Tool events"
        );
        // Accumulated tool body should parse to one call
        let calls = hipfire_generate::dense::parse_glimmer_atem(&tool_text).expect("parse should succeed");
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "weather.get_forecast");
        assert_eq!(
            calls[0].arguments["location"],
            serde_json::Value::String("Paris".into())
        );
    }
