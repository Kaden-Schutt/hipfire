// SPDX-License-Identifier: Apache-2.0
//! Dump the daemon's JinjaChatFrame render of an HFQ's embedded chat_template,
//! for byte-comparison against HF apply_chat_template. CPU-only.
//! Usage: dump_jinja_render --model <hfq> [--user <text>] [--think 0|1]
use hipfire_runtime::hfq::HfqFile;
use hipfire_runtime::tokenizer::Tokenizer;
use hipfire_runtime::prompt_frame::JinjaChatFrame;
fn main() {
    let a: Vec<String> = std::env::args().collect();
    let mut model = String::new();
    let mut user = "What is the capital of France?".to_string();
    let mut think = false;
    let mut i = 1;
    while i < a.len() { match a[i].as_str() {
        "--model" => { model = a[i+1].clone(); i+=2; }
        "--user" => { user = a[i+1].clone(); i+=2; }
        "--think" => { think = a[i+1] == "1"; i+=2; }
        _ => { i+=1; }
    }}
    let hfq = HfqFile::open(std::path::Path::new(&model)).expect("open");
    let ct = hfq.chat_template();
    println!("chat_template present: {}", ct.is_some());
    let tok = Tokenizer::from_hfq_metadata(&hfq.metadata_json).expect("tok");
    println!("bos_id={} eos_id={}", tok.bos_id, tok.eos_id);
    let Some(t) = ct else { println!("NO TEMPLATE"); return; };
    let frame = JinjaChatFrame { tokenizer: &tok, template: &t, system: None, user: &user, enable_thinking: think, bos_token: None };
    match frame.render() {
        Ok(s) => { println!("=== RENDERED (think={think}) ===\n{s}\n=== END ==="); let ids = tok.encode(&s); println!("encoded {} ids: {:?}", ids.len(), ids); }
        Err(e) => println!("RENDER ERR: {e}"),
    }
}
