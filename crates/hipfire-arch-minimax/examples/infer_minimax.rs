// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Minimal MiniMax-M2 greedy inference — real-model e2e coherence check.
//! Loads an HFQ, runs prefill + greedy decode via `decode_step`, prints text.
//!
//! Usage: infer_minimax --model <hfq> [--prompt <text>] [--max N]

#[cfg(not(feature = "deltanet"))]
fn main() {
    eprintln!("build with --features deltanet");
}

#[cfg(feature = "deltanet")]
fn main() {
    use hipfire_arch_minimax::forward::decode_step;
    use hipfire_arch_minimax::minimax::{MiniMaxConfig, MiniMaxState, MiniMaxWeights};
    use hipfire_runtime::hfq::HfqFile;
    use hipfire_runtime::tokenizer::Tokenizer;
    use std::path::PathBuf;

    let argv: Vec<String> = std::env::args().collect();
    let mut model: Option<PathBuf> = None;
    let mut prompt = "The capital of France is".to_string();
    let mut max: usize = 64;
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
            other => {
                eprintln!("unknown arg {other}");
                std::process::exit(1);
            }
        }
    }
    let model = model.expect("--model required");

    let mut gpu = rdna_compute::Gpu::init().expect("gpu init");
    let mut hfq = HfqFile::open(&model).expect("open model");
    let cfg = MiniMaxConfig::from_hfq(&hfq).expect("config");
    eprintln!(
        "minimax hidden={} layers={} experts={}/{} vocab={}",
        cfg.hidden_size,
        cfg.num_hidden_layers,
        cfg.num_local_experts,
        cfg.num_experts_per_tok,
        cfg.vocab_size
    );
    let tok = Tokenizer::from_hfq_metadata(&hfq.metadata_json).expect("tokenizer");
    let t_load = std::time::Instant::now();
    let weights = MiniMaxWeights::load(&mut hfq, &cfg, &mut gpu, None).expect("weights");
    eprintln!("loaded weights in {:.1}s", t_load.elapsed().as_secs_f64());

    let prompt_ids = tok.encode(&prompt);
    eprintln!("prompt {:?} → {} tokens", prompt, prompt_ids.len());
    let max_seq = prompt_ids.len() + max + 16;
    let mut state = MiniMaxState::new_with_max_seq(&mut gpu, &cfg, max_seq).expect("state");

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

    // Prefill (per-token; correctness-first).
    let t0 = std::time::Instant::now();
    let mut logits = Vec::new();
    for (pos, &t) in prompt_ids.iter().enumerate() {
        logits = decode_step(&cfg, &weights, &mut state, &mut gpu, t, pos as u32).expect("prefill");
    }
    eprintln!(
        "prefill {} tok in {:.2}s",
        prompt_ids.len(),
        t0.elapsed().as_secs_f64()
    );

    // Greedy decode.
    let mut gen = Vec::new();
    let mut pos = prompt_ids.len();
    let t1 = std::time::Instant::now();
    for _ in 0..max {
        let next = argmax(&logits);
        gen.push(next);
        // common MiniMax/Qwen EOS ids; stop early if hit
        if matches!(next, 200020 | 151643 | 151645 | 2) {
            break;
        }
        logits =
            decode_step(&cfg, &weights, &mut state, &mut gpu, next, pos as u32).expect("decode");
        pos += 1;
    }
    let dt = t1.elapsed().as_secs_f64();
    eprintln!(
        "decoded {} tok in {:.2}s ({:.1} tok/s)",
        gen.len(),
        dt,
        gen.len() as f64 / dt
    );
    println!(
        "=== PROMPT ===\n{prompt}\n=== GENERATION ===\n{}",
        tok.decode(&gen)
    );
    eprintln!("token ids: {:?}", &gen[..gen.len().min(40)]);
}
