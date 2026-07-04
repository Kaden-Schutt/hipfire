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

//! End-to-end greedy generation for LFM2.5-MoE **via the `SimpleAr` seam**
//! ([`Lfm2Backend`]), proving the serving-seam path produces the same coherent
//! output as the free-function `infer_lfm2moe` example. Loads an HFQ through
//! `Lfm2Backend::from_hfq`, prefills, then greedily decodes through the trait
//! methods (`prefill` / `decode_step` / `logits`) and prints the text.
//!
//! Run (coordinate the GPU lock around it):
//!   cargo run --release -p hipfire-arch-lfm2moe --features deltanet \
//!     --example seam_generate -- --model <hfq> [--prompt <text>] [--max N] [--eos ID]

#![allow(clippy::explicit_counter_loop)]

#[cfg(not(feature = "deltanet"))]
fn main() {
    eprintln!("build with --features deltanet");
}

#[cfg(feature = "deltanet")]
fn main() {
    use hipfire_arch_lfm2moe::Lfm2Backend;
    use hipfire_runtime::arch::SimpleAr;
    use hipfire_runtime::hfq::HfqFile;
    use hipfire_runtime::tokenizer::Tokenizer;
    use std::path::PathBuf;

    let argv: Vec<String> = std::env::args().collect();
    let mut model: Option<PathBuf> = None;
    let mut prompt = "The capital of France is".to_string();
    let mut max: usize = 48;
    // A1B (8b-a1b MoE) eos = 124900; dense LFM2.5 = 7. Default to A1B.
    let mut eos: u32 = 124900;
    let mut i = 1;
    while i < argv.len() {
        match argv[i].as_str() {
            "--model" => {
                model = Some(PathBuf::from(&argv[i + 1]));
                i += 2;
            }
            "--prompt" => {
                prompt = argv[i + 1].clone();
                i += 2;
            }
            "--max" => {
                max = argv[i + 1].parse().expect("--max");
                i += 2;
            }
            "--eos" => {
                eos = argv[i + 1].parse().expect("--eos");
                i += 2;
            }
            other => {
                eprintln!("unknown arg {other}");
                std::process::exit(1);
            }
        }
    }
    let model = model.expect("--model required");

    let mut gpu = hipfire_rdna::Gpu::init().expect("gpu init");
    let mut hfq = HfqFile::open(&model).expect("open model");
    let tok = Tokenizer::from_hfq_metadata(&hfq.metadata_json).expect("tokenizer");
    let prompt_ids = tok.encode(&prompt);
    let max_seq = prompt_ids.len() + max + 16;

    let t_load = std::time::Instant::now();
    let mut backend =
        Lfm2Backend::from_hfq(&mut gpu, &mut hfq, max_seq, max_seq, eos).expect("Lfm2Backend");
    eprintln!(
        "loaded via Lfm2Backend (vocab={}) in {:.1}s; prompt {:?} -> {} ids",
        backend.vocab_size(),
        t_load.elapsed().as_secs_f64(),
        prompt,
        prompt_ids.len(),
    );

    let argmax = |v: &[f32]| -> u32 {
        let mut bi = 0u32;
        let mut bv = f32::NEG_INFINITY;
        for (i, &x) in v.iter().enumerate() {
            if x > bv {
                bv = x;
                bi = i as u32;
            }
        }
        bi
    };

    let t0 = std::time::Instant::now();
    backend
        .prefill(&mut gpu, &prompt_ids)
        .expect("seam prefill");
    eprintln!(
        "prefill {} tok in {:.2}s",
        prompt_ids.len(),
        t0.elapsed().as_secs_f64()
    );

    let mut gen: Vec<u32> = Vec::new();
    let mut pos = prompt_ids.len();
    for _ in 0..max {
        let logits = gpu.download_f32(backend.logits()).expect("download logits");
        let next = argmax(&logits);
        if next == eos {
            break;
        }
        gen.push(next);
        backend
            .decode_step(&mut gpu, next, pos)
            .expect("seam decode_step");
        pos += 1;
    }

    let text = tok.decode(&gen);
    println!("=== SEAM OUTPUT ===\n{prompt}{text}");
    eprintln!("generated {} tokens via the SimpleAr seam", gen.len());
}
