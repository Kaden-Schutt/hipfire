#![allow(
    clippy::duplicated_attributes,
    clippy::doc_lazy_continuation,
    clippy::doc_overindented_list_items,
    clippy::explicit_counter_loop,
    clippy::field_reassign_with_default,
    clippy::manual_checked_ops,
    clippy::manual_clamp,
    clippy::manual_div_ceil,
    clippy::needless_range_loop,
    clippy::ptr_arg,
    clippy::same_item_push,
    clippy::too_many_arguments,
    clippy::type_complexity,
    clippy::unnecessary_cast,
    clippy::useless_vec,
    clippy::while_let_loop
)]
// hipfire example clippy sweep: examples are GPU probes/benches, not reusable APIs.

// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Render a model's upstream HF chat_template via JinjaChatFrame and
//! print the result to stdout. Diagnostic for issue #171 — verifies
//! minijinja can parse the Qwen3 family template and that the rendered
//! string matches the transformers/jinja2 reference output byte-for-byte.
//!
//! Usage:
//!   cargo run --release -p hipfire-runtime --example render_chat_template -- \
//!     <model-mq4.hfq> <prompt.txt>
//!
//! Prints rendered length + first/last bytes. Compare against the
//! Python jinja2 reference render to confirm parity.

use hipfire_prompt::JinjaChatFrame;
use hipfire_runtime::hfq::HfqFile;
use hipfire_runtime::tokenizer::Tokenizer;
use std::fs;
use std::path::Path;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("usage: render_chat_template <model-mq4.hfq> <prompt.txt>");
        std::process::exit(2);
    }
    let model_path = &args[1];
    let prompt_path = &args[2];

    let hfq = HfqFile::open(Path::new(model_path)).expect("open model");
    let template = hfq.chat_template().expect("model has no chat_template");
    let tokenizer = Tokenizer::from_hfq_metadata(&hfq.metadata_json).expect("tokenizer not found");
    let prompt = fs::read_to_string(prompt_path).expect("read prompt");

    let frame = JinjaChatFrame {
        tokenizer: &tokenizer,
        template: &template,
        system: None,
        user: &prompt,
        enable_thinking: true,
        bos_token: None,
    };

    eprintln!("=== template lines around 45 ===");
    for (i, line) in template.split("\n").enumerate() {
        if (40..=55).contains(&i) {
            eprintln!("{:3}  {}", i + 1, line);
        }
    }
    match frame.render() {
        Ok(rendered) => {
            println!("=== rendered length: {}", rendered.len());
            println!("=== first 500 chars ===");
            println!("{:?}", &rendered[..rendered.len().min(500)]);
            println!("=== last 200 chars ===");
            let n = rendered.len();
            let start = n.saturating_sub(200);
            println!("{:?}", &rendered[start..]);

            // Also tokenize and report token count for a sanity check.
            let tokens = tokenizer.encode(&rendered);
            println!("=== token count: {}", tokens.len());
            println!("=== first 8 tokens: {:?}", &tokens[..tokens.len().min(8)]);
            println!(
                "=== last 8 tokens: {:?}",
                &tokens[tokens.len().saturating_sub(8)..]
            );
        }
        Err(e) => {
            eprintln!("render failed: {e}");
            std::process::exit(1);
        }
    }
}
