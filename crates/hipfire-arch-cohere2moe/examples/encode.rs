// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
//! Tokenize text with the HFQ's embedded tokenizer → JSON id array on stdout.
//! Used to build the fixed token list for the KLD/PPL sweep (no Python lib
//! available). Usage:
//!   encode --model <hfq> [--text "..."] [--text-file <path>] [--max N]

use hipfire_runtime::hfq::HfqFile;
use hipfire_runtime::tokenizer::Tokenizer;
use std::path::PathBuf;

fn main() {
    let argv: Vec<String> = std::env::args().collect();
    let mut model: Option<PathBuf> = None;
    let mut text: Option<String> = None;
    let mut text_file: Option<PathBuf> = None;
    let mut max: usize = usize::MAX;
    let mut i = 1;
    while i < argv.len() {
        match argv[i].as_str() {
            "--model" => {
                model = Some(PathBuf::from(&argv[i + 1]));
                i += 2;
            }
            "--text" => {
                text = Some(argv[i + 1].clone());
                i += 2;
            }
            "--text-file" => {
                text_file = Some(PathBuf::from(&argv[i + 1]));
                i += 2;
            }
            "--max" => {
                max = argv[i + 1].parse().expect("--max");
                i += 2;
            }
            other => {
                eprintln!("unknown arg {other}");
                std::process::exit(1);
            }
        }
    }
    let model = model.expect("--model required");
    let text = match (text, text_file) {
        (Some(t), _) => t,
        (None, Some(f)) => std::fs::read_to_string(&f).expect("read --text-file"),
        (None, None) => {
            eprintln!("--text or --text-file required");
            std::process::exit(1);
        }
    };

    let hfq = HfqFile::open(&model).expect("open model");
    let tok = Tokenizer::from_hfq_metadata(&hfq.metadata_json).expect("tokenizer");
    let mut ids = tok.encode(&text);
    if ids.len() > max {
        ids.truncate(max);
    }
    eprintln!("encoded {} chars → {} tokens", text.len(), ids.len());
    println!("{}", serde_json::to_string(&ids).expect("serialize ids"));
}
