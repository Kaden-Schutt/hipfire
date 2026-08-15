// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Glimmer history prep tests.
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


    #[test]
    fn normalize_arguments_object() {
        let v = serde_json::json!({"a":1});
        assert_eq!(hipfire_generate::dense::normalize_glimmer_tool_arguments(&v).unwrap(), v);
    }

    #[test]
    fn normalize_arguments_null() {
        let v = serde_json::Value::Null;
        assert_eq!(
            hipfire_generate::dense::normalize_glimmer_tool_arguments(&v).unwrap(),
            serde_json::json!({})
        );
    }

    #[test]
    fn normalize_arguments_string_object() {
        let v = serde_json::Value::String("{\"a\":1}".into());
        assert_eq!(
            hipfire_generate::dense::normalize_glimmer_tool_arguments(&v).unwrap(),
            serde_json::json!({"a":1})
        );
    }

    #[test]
    fn normalize_arguments_string_invalid() {
        let v = serde_json::Value::String("not json".into());
        assert!(hipfire_generate::dense::normalize_glimmer_tool_arguments(&v).is_err());
    }

    #[test]
    fn normalize_arguments_string_non_object() {
        let v = serde_json::Value::String("[1,2]".into());
        assert!(hipfire_generate::dense::normalize_glimmer_tool_arguments(&v).is_err());
    }

    #[test]
    fn prepare_history_resolves_name() {
        let assistant = hipfire_runtime::prompt_frame::Message {
            role: hipfire_runtime::prompt_frame::Role::Assistant,
            content: String::new(),
            reasoning_content: None,
            name: None,
            rendered_name: None,
            tool_calls: vec![hipfire_runtime::prompt_frame::ToolCall {
                id: Some("call_0".into()),
                name: "weather.get_forecast".into(),
                arguments: serde_json::json!({"location":"Paris"}),
                rendered_body: None,
            }],
            tool_call_id: None,
            tool_plan: String::new(),
        };
        let tool = hipfire_runtime::prompt_frame::Message {
            role: hipfire_runtime::prompt_frame::Role::Tool,
            content: "sunny".into(),
            reasoning_content: None,
            name: None,
            rendered_name: None,
            tool_calls: vec![],
            tool_call_id: Some("call_0".into()),
            tool_plan: String::new(),
        };
        let out = hipfire_generate::dense::prepare_glimmer_onyx_history(&[assistant, tool]).expect("should succeed");
        assert_eq!(out[1].rendered_name, Some("weather.get_forecast".into()));
    }
