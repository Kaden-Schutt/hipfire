// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Coherence eval for ZAYA1: greedy completion across diverse prompts
//! (factual / arithmetic / sequence / code / science), tokenized from the HFQ's
//! embedded tokenizer — no daemon needed. Prints prompt → completion for review.
//!
//! Run: cargo run --release -p hipfire-arch-zaya --example eval -- <hfq> [ntok]

use hipfire_arch_zaya::arch::ZayaModel;
use hipfire_arch_zaya::ZayaConfig;
use hipfire_model::tokenizer::Tokenizer;
use hipfire_runtime::arch::SimpleAr;
use hipfire_runtime::hfq::HfqFile;
use hipfire_rdna::Gpu;
use std::path::Path;

fn argmax(v: &[f32]) -> usize {
    v.iter()
        .enumerate()
        .fold((0usize, f32::NEG_INFINITY), |b, (i, &x)| {
            if x > b.1 {
                (i, x)
            } else {
                b
            }
        })
        .0
}

fn main() {
    let mut args = std::env::args().skip(1);
    let hfq_path = args
        .next()
        .unwrap_or_else(|| "/home/sadara/zaya1-8b-native.mq4.hfq".to_string());
    let ntok: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(40);

    let hfq = HfqFile::open(Path::new(&hfq_path)).expect("open hfq");
    let meta: serde_json::Value = serde_json::from_str(&hfq.metadata_json).expect("metadata");
    let cfg = ZayaConfig::from_json(meta.get("config").unwrap_or(&meta)).expect("config");
    let eos = cfg.eos_token_id;
    let tok = Tokenizer::from_hfq_metadata(&hfq.metadata_json).expect("tokenizer");

    let mut gpu = Gpu::init().expect("gpu init");
    eprintln!("loading {hfq_path} ...");
    let mut model = ZayaModel::from_hfq(&mut gpu, &hfq, cfg, 4096).expect("load");
    eprintln!("loaded.\n");

    let prompts = [
        "The capital of Japan is",
        "The first five prime numbers are",
        "2 + 2 =",
        "Water is made of hydrogen and",
        "The opposite of hot is",
        "def square(x):\n    return",
        "Roses are red, violets are",
    ];

    for p in prompts {
        let mut ids = tok.encode(p);
        // drop a trailing eos so generation doesn't stop immediately.
        if ids.last() == Some(&eos) {
            ids.pop();
        }
        model.prefill(&mut gpu, &ids).expect("prefill");
        let mut gen = Vec::new();
        let mut pos = ids.len();
        for _ in 0..ntok {
            let logits = gpu.download_f32(model.logits()).expect("logits");
            let next = argmax(&logits) as u32;
            if next == eos {
                break;
            }
            gen.push(next);
            model.decode_step(&mut gpu, next, pos).expect("decode");
            pos += 1;
        }
        let text = tok.decode(&gen);
        println!("PROMPT: {p:?}\n  → {text:?}\n");
    }
}
