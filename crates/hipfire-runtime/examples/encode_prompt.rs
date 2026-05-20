// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Encode a prompt file to token IDs (and optionally count rare-token positions).
//! Usage: encode_prompt <model.hfq> <prompt.txt> [--normalize] [--heat]

use hipfire_runtime::hfq::HfqFile;
use std::path::Path;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let model_path = args.get(1).expect("model path");
    let prompt_path = args.get(2).expect("prompt path");
    let normalize = args.iter().any(|a| a == "--normalize");
    let heat = args.iter().any(|a| a == "--heat");

    let hfq = HfqFile::open(Path::new(model_path)).expect("open model");
    let tokenizer = hipfire_runtime::tokenizer::Tokenizer::from_hfq_metadata(&hfq.metadata_json)
        .expect("tokenizer");

    let raw = std::fs::read_to_string(prompt_path).expect("read prompt");
    let text = if normalize {
        if std::env::var("HIPFIRE_NORMALIZE_PROMPT").ok().as_deref() != Some("1") {
            std::env::set_var("HIPFIRE_NORMALIZE_PROMPT", "1");
        }
        hipfire_runtime::tokenizer::maybe_normalize_prompt(&raw).into_owned()
    } else {
        raw
    };
    if heat {
        tokenizer.dump_prompt_heat(&text);
        return;
    }
    let ids = tokenizer.encode(&text);
    eprintln!("text bytes: {}", text.len());
    eprintln!("token count: {}", ids.len());
    let mut counts = std::collections::HashMap::<u32, usize>::new();
    for id in &ids { *counts.entry(*id).or_insert(0) += 1; }
    let nl_ids = [198u32, 271, 1358];
    eprintln!("newline-id counts:");
    for nl in nl_ids { eprintln!("  id {:>5}: {}", nl, counts.get(&nl).copied().unwrap_or(0)); }
    for id in ids { println!("{id}"); }
}
