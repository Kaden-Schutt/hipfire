// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! GPU-free A/B of chat-template prefix-cache behaviour for thinking models.
//!
//! Question (item #37 / froggeric eval): does a `preserve_thinking` template
//! give a PURE FORWARD EXTENSION across a user turn (so plain LCP caches with
//! no verbatim-splice and no DeltaNet rewind), whereas the official
//! interleaved-thinking template diverges at the prior assistant turn?
//!
//! Method: simulate turn 1 = render([user1]) + the model's emitted assistant
//! body (`{reasoning}\n</think>\n\n{answer}`) = the KV the daemon would hold.
//! Then render turn 2 = [user1, assistant(content=answer, reasoning_content=
//! reasoning), user2] and measure the longest-common-prefix of the two token
//! streams. lcp == turn1_kv.len() ⇒ pure forward extension (100% cache).
//!
//! Renders through the SAME minijinja env config as
//! `JinjaChatFrame::render_messages` (trim_blocks + lstrip_blocks + strict +
//! pycompat + raise_exception), so the result reflects the daemon's real
//! tokenization — not a Jinja2 reference.
//!
//! Usage: template_cache_eval <model.hfq> <template.jinja> <preserve_thinking:true|false>

use hipfire_runtime::hfq::HfqFile;
use minijinja::{context, Environment, Error, ErrorKind, Value};
use minijinja_contrib::pycompat::unknown_method_callback;
use std::path::Path;

fn render(
    template: &str,
    bos_token: &str,
    messages: &serde_json::Value,
    preserve_thinking: bool,
) -> Result<String, String> {
    let mut env = Environment::new();
    env.set_undefined_behavior(minijinja::UndefinedBehavior::Strict);
    env.set_trim_blocks(true);
    env.set_lstrip_blocks(true);
    env.set_unknown_method_callback(unknown_method_callback);
    env.add_function("raise_exception", |msg: String| -> Result<Value, Error> {
        Err(Error::new(ErrorKind::InvalidOperation, msg))
    });
    env.add_template("chat", template).map_err(|e| format!("parse: {e}"))?;
    let tmpl = env.get_template("chat").map_err(|e| format!("lookup: {e}"))?;
    let empty: Vec<serde_json::Value> = Vec::new();
    let ctx = context! {
        messages => Value::from_serialize(messages),
        add_generation_prompt => true,
        enable_thinking => true,
        preserve_thinking => preserve_thinking,
        bos_token => bos_token,
        tools => Value::from_serialize(&empty),
        documents => Value::from_serialize(&empty),
        tool_call_kwargs => Value::from_serialize(&serde_json::Map::new()),
    };
    tmpl.render(ctx).map_err(|e| format!("render: {e}"))
}

fn lcp(a: &[u32], b: &[u32]) -> usize {
    let n = a.len().min(b.len());
    let mut i = 0;
    while i < n && a[i] == b[i] {
        i += 1;
    }
    i
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let model_path = args.get(1).expect("usage: <model.hfq> <template.jinja> <preserve:true|false>");
    let tmpl_path = args.get(2).expect("template path");
    let preserve = args.get(3).map(|s| s == "true").unwrap_or(false);

    let hfq = HfqFile::open(Path::new(model_path)).expect("open model");
    let tok = hipfire_runtime::tokenizer::Tokenizer::from_hfq_metadata(&hfq.metadata_json)
        .expect("tokenizer");
    let bos_bytes = tok.decode_bytes(&[tok.bos_id]);
    let bos = String::from_utf8_lossy(&bos_bytes).to_string();
    let template = std::fs::read_to_string(tmpl_path).expect("read template");

    // Conversation pieces (short, deterministic).
    let q1 = "Name the three primary colors in one short sentence.";
    let reasoning = "The user wants the three primary colors. In the standard RYB model those are red, yellow, and blue.";
    let answer = "The three primary colors are red, yellow, and blue.";
    let q2 = "Now name the three secondary colors in one short sentence.";

    // Turn 1 KV the daemon would hold = render([user1]) + the model's emitted
    // assistant body. The render ends on `<|im_start|>assistant\n<think>\n`
    // (enable_thinking primer); the model then emits `{reasoning}\n</think>\n\n
    // {answer}`. We approximate the generated tokens by encoding that body text.
    // Match the production `Message` shape (tool_calls always present as []) so
    // templates that assume the field exists (the official one) don't trip our
    // strict-undefined env — a fair comparison, not a harness artifact.
    let t1_msgs = serde_json::json!([{ "role": "user", "content": q1, "tool_calls": [] }]);
    let t1_prompt = render(&template, &bos, &t1_msgs, preserve).expect("t1 render");
    let asst_body = format!("{reasoning}\n</think>\n\n{answer}");
    let t1_kv_text = format!("{t1_prompt}{asst_body}");
    let t1_kv = tok.encode(&t1_kv_text);

    // Turn 2: the OpenAI client sends back content=answer; we additionally
    // supply reasoning_content server-side (from asst_turn_cache) so a
    // preserve_thinking template can reconstruct the think block.
    let t2_msgs = serde_json::json!([
        { "role": "user", "content": q1, "tool_calls": [] },
        { "role": "assistant", "content": answer, "reasoning_content": reasoning, "tool_calls": [] },
        { "role": "user", "content": q2, "tool_calls": [] },
    ]);
    let t2_text = render(&template, &bos, &t2_msgs, preserve).expect("t2 render");
    let t2 = tok.encode(&t2_text);

    let l = lcp(&t1_kv, &t2);
    let forward_ext = l == t1_kv.len();
    let pct = if t1_kv.is_empty() { 0.0 } else { 100.0 * l as f64 / t1_kv.len() as f64 };

    println!("template       : {}", Path::new(tmpl_path).file_name().unwrap().to_string_lossy());
    println!("preserve_thinking: {preserve}");
    println!("turn1_kv tokens : {}", t1_kv.len());
    println!("turn2    tokens : {}", t2.len());
    println!("lcp            : {l}  ({pct:.1}% of turn1_kv)");
    println!("forward_extension(100% cache): {forward_ext}");
    // Show the first divergence in decoded text for diagnosis.
    if !forward_ext {
        let lo = l.saturating_sub(6);
        let a_hi = (l + 6).min(t1_kv.len());
        let b_hi = (l + 6).min(t2.len());
        println!("  diverge@ {l}: kv …{:?}", tok.decode(&t1_kv[lo..a_hi]));
        println!("           t2 …{:?}", tok.decode(&t2[lo..b_hi]));
    }
}
