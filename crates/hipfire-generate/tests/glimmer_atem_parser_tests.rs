// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Glimmer atem parser tests.
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
    fn parses_representative_block() {
        let body = "<atem:function_calls>\n<atem:invoke name=\"weather.get_forecast\">\n<atem:parameter name=\"location\">Paris</atem:parameter>\n<atem:parameter name=\"options\">{\"units\":\"celsius\",\"days\":[1,2]}</atem:parameter>\n<atem:parameter name=\"include_alerts\">true</atem:parameter>\n<atem:parameter name=\"fallback\">null</atem:parameter>\n</atem:invoke>\n</atem:function_calls>";
        let calls = hipfire_generate::dense::parse_glimmer_atem(body).expect("parse should succeed");
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "weather.get_forecast");
        assert_eq!(
            calls[0].arguments["location"],
            serde_json::Value::String("Paris".into())
        );
        assert_eq!(
            calls[0].arguments["options"]["units"],
            serde_json::Value::String("celsius".into())
        );
        assert_eq!(
            calls[0].arguments["options"]["days"],
            serde_json::json!([1, 2])
        );
        assert_eq!(
            calls[0].arguments["include_alerts"],
            serde_json::Value::Bool(true)
        );
        assert_eq!(calls[0].arguments["fallback"], serde_json::Value::Null);
    }

    #[test]
    fn parses_adversarial_chunk_splits() {
        let body = "<atem:function_calls>\n<atem:invoke name=\"test.func\">\n<atem:parameter name=\"a\">1</atem:parameter>\n<atem:parameter name=\"b\">{\"x\":1}</atem:parameter>\n</atem:invoke>\n</atem:function_calls>";
        for split in 1..body.len() {
            if !body.is_char_boundary(split) {
                continue;
            }
            let (left, right) = body.split_at(split);
            let combined = left.to_string() + right;
            let calls = hipfire_generate::dense::parse_glimmer_atem(&combined).expect("should parse after split");
            assert_eq!(calls.len(), 1);
            assert_eq!(calls[0].name, "test.func");
        }
        let body2 = "<atem:function_calls>\n<atem:invoke name=\"test.func\">\n<atem:parameter name=\"msg\">hello \u{1F30D}</atem:parameter>\n</atem:invoke>\n</atem:function_calls>";
        let calls2 = hipfire_generate::dense::parse_glimmer_atem(body2).expect("should parse multibyte");
        assert_eq!(
            calls2[0].arguments["msg"],
            serde_json::Value::String("hello \u{1F30D}".into())
        );
    }

    #[test]
    fn parses_multiple_invokes() {
        let body = "<atem:function_calls>\n<atem:invoke name=\"func1\">\n<atem:parameter name=\"a\">1</atem:parameter>\n</atem:invoke>\n</atem:function_calls>\n<atem:function_calls>\n<atem:invoke name=\"func2\">\n<atem:parameter name=\"b\">2</atem:parameter>\n</atem:invoke>\n</atem:function_calls>";
        let calls = hipfire_generate::dense::parse_glimmer_atem(body).expect("multiple");
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].name, "func1");
        assert_eq!(calls[1].name, "func2");
    }
