// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Render-vs-HF byte audit (durability scar #1: render skew).
//!
//! Renders a conversation through hipfire's REAL minijinja env (byte-for-byte
//! the same config as `JinjaChatFrame::render_messages`: trim_blocks +
//! lstrip_blocks + strict-undefined + pycompat + raise_exception) and writes:
//!   - <out>.rust.txt  : the rendered prompt string (exact bytes)
//!   - <out>.ctx.json  : the EXACT context dict used (messages, tools,
//!                       bos_token, flags) so the Python/HF jinja2 reference
//!                       renders byte-identical INPUTS — isolating purely the
//!                       engine divergence (minijinja vs HF's jinja2).
//!
//! Faithful to the daemon: `preserve_thinking` is included in the context
//! ONLY when the fixture sets it (the daemon's render_messages does NOT pass
//! it, so the template's `... if preserve_thinking is defined else false`
//! default fires — interleaved behavior). Set it explicitly to test the
//! intended item-4 cache path.
//!
//! Usage: jinja_render_dump <model.hfq> <template.jinja> <fixture.json> <out_prefix>

use hipfire_runtime::hfq::HfqFile;
use minijinja::{Environment, Error, ErrorKind, Value};
use minijinja_contrib::pycompat::unknown_method_callback;
use std::path::Path;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let model_path = args.get(1).expect("usage: <model.hfq> <template.jinja> <fixture.json> <out_prefix>");
    let tmpl_path = args.get(2).expect("template path");
    let fixture_path = args.get(3).expect("fixture path");
    let out_prefix = args.get(4).expect("out prefix");

    let hfq = HfqFile::open(Path::new(model_path)).expect("open model");
    let tok = hipfire_runtime::tokenizer::Tokenizer::from_hfq_metadata(&hfq.metadata_json)
        .expect("tokenizer");
    let bos_bytes = tok.decode_bytes(&[tok.bos_id]);
    let bos = String::from_utf8_lossy(&bos_bytes).to_string();
    let template = std::fs::read_to_string(tmpl_path).expect("read template");

    let fixture: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(fixture_path).expect("read fixture"))
            .expect("parse fixture");

    let messages = fixture.get("messages").cloned().unwrap_or(serde_json::json!([]));
    let tools = fixture.get("tools").cloned().unwrap_or(serde_json::Value::Null);
    let enable_thinking = fixture.get("enable_thinking").and_then(|v| v.as_bool()).unwrap_or(true);
    let add_gen = fixture.get("add_generation_prompt").and_then(|v| v.as_bool()).unwrap_or(true);
    // null / absent => omit (undefined, daemon-faithful). bool => include.
    let preserve = fixture.get("preserve_thinking").and_then(|v| v.as_bool());

    // Build the EXACT context dict (mirror render_messages), as JSON so the
    // Python reference consumes identical inputs.
    let mut ctx_map = serde_json::Map::new();
    ctx_map.insert("messages".into(), messages.clone());
    ctx_map.insert("add_generation_prompt".into(), serde_json::Value::Bool(add_gen));
    ctx_map.insert("enable_thinking".into(), serde_json::Value::Bool(enable_thinking));
    ctx_map.insert("bos_token".into(), serde_json::Value::String(bos.clone()));
    ctx_map.insert(
        "tools".into(),
        if tools.is_null() { serde_json::json!([]) } else { tools.clone() },
    );
    ctx_map.insert("documents".into(), serde_json::json!([]));
    ctx_map.insert("tool_call_kwargs".into(), serde_json::json!({}));
    if let Some(p) = preserve {
        ctx_map.insert("preserve_thinking".into(), serde_json::Value::Bool(p));
    }
    let ctx_json = serde_json::Value::Object(ctx_map);

    let mut env = Environment::new();
    env.set_undefined_behavior(minijinja::UndefinedBehavior::Strict);
    env.set_trim_blocks(true);
    env.set_lstrip_blocks(true);
    env.set_unknown_method_callback(unknown_method_callback);
    env.add_function("raise_exception", |msg: String| -> Result<Value, Error> {
        Err(Error::new(ErrorKind::InvalidOperation, msg))
    });
    // Mirror render_messages: HF-spaced tojson override.
    env.add_filter("tojson", hipfire_runtime::prompt_frame::hf_tojson);
    env.add_template("chat", &template).expect("template parse");
    let tmpl = env.get_template("chat").expect("template lookup");

    let render_result = tmpl.render(Value::from_serialize(&ctx_json));

    let rust_out = format!("{out_prefix}.rust.txt");
    let ctx_out = format!("{out_prefix}.ctx.json");
    match render_result {
        Ok(s) => {
            std::fs::write(&rust_out, s.as_bytes()).expect("write rust out");
            println!("OK   rust_bytes={} -> {rust_out}", s.as_bytes().len());
        }
        Err(e) => {
            // Record the error so the audit shows render failures explicitly.
            std::fs::write(&rust_out, format!("__RENDER_ERROR__\n{e:#}")).expect("write err");
            println!("ERR  render failed: {e:#} -> {rust_out}");
        }
    }
    std::fs::write(&ctx_out, serde_json::to_string_pretty(&ctx_json).unwrap())
        .expect("write ctx");
    println!("CTX  -> {ctx_out}");
}
