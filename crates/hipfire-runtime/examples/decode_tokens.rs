// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Decode a token dump file to text using a model's tokenizer.
//! Usage: decode_tokens <model.hfq> <tokens.txt>

use hipfire_runtime::hfq::HfqFile;
use std::path::Path;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let model_path = args.get(1).expect("model path");
    let tokens_path = args.get(2).expect("tokens path");

    let hfq = HfqFile::open(Path::new(model_path)).expect("open model");
    let tokenizer = hipfire_runtime::tokenizer::Tokenizer::from_hfq_metadata(&hfq.metadata_json)
        .expect("tokenizer");

    let content = std::fs::read_to_string(tokens_path).expect("read tokens");
    let tokens: Vec<u32> = content.lines()
        .filter_map(|l| l.trim().parse().ok())
        .collect();
    let text = tokenizer.decode(&tokens);
    print!("{text}");
}
