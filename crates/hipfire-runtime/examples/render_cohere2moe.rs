// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
//! Repro for the Pi multi-turn agentic derail: render a conversation that has
//! an assistant turn WITH tool_calls (+ a tool result) through the Cohere
//! tool_use template, exactly like the daemon's generate_cohere2moe does. If
//! render_messages ERRORS, generate_cohere2moe falls back to the hand-rolled
//! ChatML frame → the model sees <|im_start|>/<|im_end|> → derails.
//!   usage: render_cohere2moe <model.hfq>

use hipfire_runtime::hfq::HfqFile;
use hipfire_runtime::prompt_frame::{JinjaChatFrame, Message, Role, ToolCall};
use hipfire_runtime::tokenizer::Tokenizer;
use std::path::Path;

fn main() {
    let model = std::env::args()
        .nth(1)
        .expect("usage: render_cohere2moe <model.hfq>");
    let hfq = HfqFile::open(Path::new(&model)).expect("open model");
    // Mirror the daemon's arch-12 selection + START_RESPONSE→START_TEXT rewrite.
    let template = hfq
        .chat_template_named("tool_use")
        .expect("no tool_use template")
        .replace("<|START_RESPONSE|>", "<|START_TEXT|>")
        .replace("<|END_RESPONSE|>", "<|END_TEXT|>")
        // The daemon's Message/ToolCall are flat {name, arguments} with no
        // tool_plan; the upstream template reads message.tool_plan + the
        // OpenAI-nested tc['function'][...]. Bridge the shape:
        .replace("{{message.tool_plan}}", "{{ message.tool_plan or '' }}")
        .replace("{{ tc['function']['name'] }}", "{{ tc.name }}")
        .replace(
            "{{ tc['function']['arguments']|tojson }}",
            "{{ tc.arguments|tojson }}",
        );
    let tok = Tokenizer::from_hfq_metadata(&hfq.metadata_json).expect("tokenizer");
    let frame = JinjaChatFrame {
        tokenizer: &tok,
        template: &template,
        system: None,
        user: "",
        enable_thinking: true,
        bos_token: None,
        reasoning_strength: None,
    };

    let tools = serde_json::json!([{
        "type": "function",
        "function": { "name": "bash", "description": "Run a bash command.",
            "parameters": {"type":"object","properties":{"command":{"type":"string"}},"required":["command"]} }
    }]);
    let tools_arr = tools.as_array().unwrap();

    // (A) plain multi-turn — known good.
    let plain = vec![
        Message {
            role: Role::User,
            content: "Hi".into(),
            reasoning_content: None,
            name: None,
            rendered_name: None,
            tool_calls: vec![],
            tool_call_id: None,
            tool_plan: String::new(),
        },
        Message {
            role: Role::Assistant,
            content: "Hello!".into(),
            reasoning_content: None,
            name: None,
            rendered_name: None,
            tool_calls: vec![],
            tool_call_id: None,
            tool_plan: String::new(),
        },
        Message {
            role: Role::User,
            content: "List files.".into(),
            reasoning_content: None,
            name: None,
            rendered_name: None,
            tool_calls: vec![],
            tool_call_id: None,
            tool_plan: String::new(),
        },
    ];
    match frame.render_messages(&plain, Some(tools_arr), None) {
        Ok(_) => println!("(A) plain multi-turn + tools: OK"),
        Err(e) => println!("(A) plain multi-turn + tools: ERR -> {e}"),
    }

    // (B) THE PI CASE: assistant turn WITH tool_calls + the reasoning that
    // preceded it (tool_plan), then a tool result. North "interleaved thinking"
    // requires that prior reasoning survive into the rendered prompt's
    // <|START_THINKING|>{{message.tool_plan}}<|END_THINKING|> slot.
    let reasoning = "I need to inspect the repo layout first, so I will run ls.";
    let agentic = vec![
        Message {
            role: Role::User,
            content: "Implement a Blink-hash tree.".into(),
            reasoning_content: None,
            name: None,
            rendered_name: None,
            tool_calls: vec![],
            tool_call_id: None,
            tool_plan: String::new(),
        },
        Message {
            role: Role::Assistant,
            content: "".into(),
            reasoning_content: None,
            name: None,
            rendered_name: None,
            tool_calls: vec![ToolCall {
                id: None,
                name: "bash".into(),
                arguments: serde_json::json!({"command":"ls -la"}),
                rendered_body: None,
            }],
            tool_call_id: None,
            tool_plan: reasoning.into(),
        },
        Message {
            role: Role::Tool,
            content: "total 7896\ndrwx... blink_hash.pdf".into(),
            reasoning_content: None,
            name: None,
            rendered_name: None,
            tool_calls: vec![],
            tool_call_id: Some("0".into()),
            tool_plan: String::new(),
        },
    ];
    match frame.render_messages(&agentic, Some(tools_arr), None) {
        Ok(r) => {
            println!(
                "(B) agentic (tool_calls in history): OK — {} chars",
                r.len()
            );
            println!("--- tail ---\n{}", &r[r.len().saturating_sub(400)..]);
            if r.contains(reasoning) {
                println!("(B) PASS: assistant tool_plan reasoning preserved in rendered prompt");
            } else {
                eprintln!("(B) FAIL: assistant reasoning was DROPPED from the rendered prompt");
                std::process::exit(1);
            }
        }
        Err(e) => {
            eprintln!(
                "(B) agentic (tool_calls in history): ERR -> {e}   <<< render_messages errored"
            );
            std::process::exit(1);
        }
    }
}
