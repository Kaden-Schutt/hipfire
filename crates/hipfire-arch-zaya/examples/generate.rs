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

//! End-to-end greedy generation smoke test for ZAYA1 via the SimpleAr seam.
//! Prefills a prompt (default: the golden ids for "The capital of France is")
//! then greedily decodes, printing the generated token ids (detokenize
//! separately). Proves the prefill + decode_step path works on the GPU.
//!
//! Run: cargo run --release -p hipfire-arch-zaya --example generate -- \
//!   /home/sadara/zaya1-8b-native.bf16.hfq 20

#![allow(clippy::explicit_counter_loop)]

use hipfire_arch_zaya::arch::ZayaModel;
use hipfire_arch_zaya::ZayaConfig;
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
        .unwrap_or_else(|| "/home/sadara/zaya1-8b-native.bf16.hfq".to_string());
    let max_new: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(20);

    let hfq = HfqFile::open(Path::new(&hfq_path)).expect("open hfq");
    let meta: serde_json::Value = serde_json::from_str(&hfq.metadata_json).expect("metadata");
    let cfg = ZayaConfig::from_json(meta.get("config").unwrap_or(&meta)).expect("config");
    let eos = cfg.eos_token_id;

    let mut gpu = Gpu::init().expect("gpu init");
    eprintln!("loading weights...");
    let mut model = ZayaModel::from_hfq(&mut gpu, &hfq, cfg, 4096).expect("load model");
    eprintln!("loaded.");

    // "The capital of France is" (bos=2 + tokens), without the trailing eos.
    let prompt: Vec<u32> = vec![2, 818, 5279, 529, 7001, 563];
    eprintln!("prompt ids = {prompt:?}");

    model.prefill(&mut gpu, &prompt).expect("prefill");
    let mut generated = Vec::new();
    let mut pos = prompt.len();
    for _ in 0..max_new {
        let logits = gpu.download_f32(model.logits()).expect("logits");
        let next = argmax(&logits) as u32;
        if next == eos {
            eprintln!("[eos]");
            break;
        }
        generated.push(next);
        model.decode_step(&mut gpu, next, pos).expect("decode");
        pos += 1;
    }

    println!("generated ids: {generated:?}");
}
