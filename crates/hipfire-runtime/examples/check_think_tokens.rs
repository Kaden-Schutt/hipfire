// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Quick check: what do <think> / </think> tokenize to in the Qwen3.5 tokenizer?
//! If they're NOT single special tokens, the infer_qwen35 think-end detection fails
//! (think_end_token = None) and the thinking block is never terminated by the host.

use hipfire_runtime::hfq::HfqFile;
use std::path::Path;

fn main() {
    let model_path = std::env::args().nth(1).expect("usage: check_think_tokens <model.hfq>");
    let hfq = HfqFile::open(Path::new(&model_path)).expect("open model");
    let tokenizer = hipfire_runtime::tokenizer::Tokenizer::from_hfq_metadata(&hfq.metadata_json)
        .expect("need tokenizer");

    let probes = [
        "<think>",
        "</think>",
        "\n<think>\n",
        "\n</think>\n",
        "<|im_start|>",
        "<|im_end|>",
        "assistant",
        "user",
        "<|endoftext|>",
    ];
    for p in &probes {
        let ids = tokenizer.encode(p);
        let back: Vec<String> = ids.iter().map(|&id| tokenizer.decode(&[id])).collect();
        println!("{:<20?} -> {} tokens: {:?}   decoded: {:?}", p, ids.len(), ids, back);
    }
}
