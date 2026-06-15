// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
//! Minimal LFM2.5-MoE greedy inference — real-model e2e coherence check.
//! Loads an HFQ, runs prefill + greedy decode via `decode_step`, prints text.
//!
//! Usage: infer_lfm2moe --model <hfq> [--prompt <text>] [--max N]

#[cfg(not(feature = "deltanet"))]
fn main() {
    eprintln!("build with --features deltanet");
}

#[cfg(feature = "deltanet")]
fn main() {
    use hipfire_arch_lfm2moe::config::Lfm2MoeConfig;
    use hipfire_arch_lfm2moe::forward::decode_step;
    use hipfire_arch_lfm2moe::lfm2moe::{Lfm2MoeState, Lfm2MoeWeights};
    use hipfire_runtime::hfq::HfqFile;
    use hipfire_runtime::tokenizer::Tokenizer;
    use std::path::PathBuf;

    let argv: Vec<String> = std::env::args().collect();
    let mut model: Option<PathBuf> = None;
    let mut prompt = "The capital of France is".to_string();
    let mut max: usize = 64;
    // --tokens <json>: pre-tokenized prompt ids (bypass the embedded tokenizer;
    // e.g. HF apply_chat_template output). --eos <id>: extra stop token.
    let mut tokens_path: Option<PathBuf> = None;
    let mut eos_extra: Option<u32> = None;
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
            "--tokens" => {
                tokens_path = Some(PathBuf::from(&argv[i + 1]));
                i += 2;
            }
            "--eos" => {
                eos_extra = Some(argv[i + 1].parse().expect("--eos"));
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
    let cfg = Lfm2MoeConfig::from_hfq(&hfq).expect("config");
    eprintln!(
        "lfm2moe hidden={} layers={} experts={}/{} vocab={}",
        cfg.hidden_size,
        cfg.num_hidden_layers,
        cfg.num_experts,
        cfg.num_experts_per_tok,
        cfg.vocab_size
    );
    let tok = Tokenizer::from_hfq_metadata(&hfq.metadata_json).expect("tokenizer");
    let t_load = std::time::Instant::now();
    let weights = Lfm2MoeWeights::load(&mut hfq, &cfg, &mut gpu).expect("weights");
    eprintln!("loaded weights in {:.1}s", t_load.elapsed().as_secs_f64());

    let prompt_ids: Vec<u32> = if let Some(tp) = &tokens_path {
        let s = std::fs::read_to_string(tp).expect("read --tokens");
        let v: Vec<i64> = serde_json::from_str(&s).expect("parse --tokens json");
        v.into_iter().map(|t| t as u32).collect()
    } else {
        tok.encode(&prompt)
    };
    eprintln!("prompt {:?} → {} tokens (src: {})", prompt, prompt_ids.len(),
        if tokens_path.is_some() { "--tokens" } else { "embedded tokenizer" });
    let max_seq = prompt_ids.len() + max + 16;
    let mut state = Lfm2MoeState::new_with_max_seq(&mut gpu, &cfg, max_seq).expect("state");

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
        // A1B EOS = 124900; dense LFM2.5 eos = 7 (<|im_end|>), pass via --eos.
        if matches!(next, 124900 | 124899 | 2) || Some(next) == eos_extra {
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
        "=== PROMPT ===\n{prompt}\n=== GENERATION (embedded-tokenizer decode) ===\n{}",
        tok.decode(&gen)
    );
    // Full generated id list for an external (e.g. HF) decode — the embedded
    // tokenizer may be wrong for a freshly-brought-up arch.
    println!("GEN_IDS_JSON: {}", serde_json::to_string(&gen).unwrap());
}
