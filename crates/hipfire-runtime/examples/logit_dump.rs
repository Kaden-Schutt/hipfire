// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Logit divergence diagnostic for Qwen3.5 DeltaNet models.
//! Dumps per-token greedy sequence for cross-GPU comparison.
//! Usage: logit_dump <model.hfq> <output_dir>

use hipfire_runtime::hfq::HfqFile;
use hipfire_arch_qwen35::qwen35::{self, DeltaNetState, Qwen35Scratch};
use hipfire_runtime::llama::{self, KvCache};
use std::io::Write;
use std::path::Path;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let model_path = args.get(1).expect("Usage: logit_dump <model.hfq> <output_dir>");
    let out_dir = args.get(2).expect("Usage: logit_dump <model.hfq> <output_dir>");
    std::fs::create_dir_all(out_dir).unwrap();

    let prompt_text = "The quick brown fox jumps over the lazy dog. Explain step by step how a combustion engine works, covering the four stroke cycle, fuel injection, and exhaust.";

    let mut hfq = HfqFile::open(Path::new(model_path)).expect("failed to open model");
    let config = qwen35::config_from_hfq(&hfq).expect("failed to read config");
    let tokenizer = hipfire_runtime::tokenizer::Tokenizer::from_hfq_metadata(&hfq.metadata_json)
        .expect("need tokenizer");

    let prompt_tokens = tokenizer.encode(prompt_text);
    eprintln!("Config: dim={}, layers={}, heads={}, kv_heads={}",
        config.dim, config.n_layers, config.n_heads, config.n_kv_heads);
    eprintln!("Prompt: {} tokens", prompt_tokens.len());

    let mut gpu = rdna_compute::Gpu::init().expect("GPU init failed");
    let arch = gpu.arch.clone();
    eprintln!("GPU: {arch}");

    let weights = qwen35::load_weights(&mut hfq, &config, &mut gpu).expect("failed to load weights");
    let max_seq = 2048usize;
    let mut kv_cache = KvCache::new_gpu_q8(
        &mut gpu, config.n_layers, config.n_kv_heads, config.head_dim, max_seq
    ).unwrap();
    let mut dn_state = DeltaNetState::new(&mut gpu, &config).unwrap();
    let scratch = Qwen35Scratch::new(&mut gpu, &config, 128).unwrap();

    let mut token_file = std::fs::File::create(format!("{out_dir}/token_sequence.txt")).unwrap();
    let mut text_file = std::fs::File::create(format!("{out_dir}/generated_text.txt")).unwrap();

    let mut meta_file = std::fs::File::create(format!("{out_dir}/meta.txt")).unwrap();
    writeln!(meta_file, "arch={arch}").unwrap();
    writeln!(meta_file, "model={model_path}").unwrap();
    writeln!(meta_file, "prompt={prompt_text}").unwrap();
    writeln!(meta_file, "prompt_tokens={}", prompt_tokens.len()).unwrap();
    writeln!(meta_file, "kv=q8, state=default, temp=greedy").unwrap();

    // Prefill
    eprintln!("Prefilling {} tokens...", prompt_tokens.len());
    for (pos, &token) in prompt_tokens.iter().enumerate() {
        qwen35::forward_scratch(
            &mut gpu, &weights, &config, token, pos,
            &mut kv_cache, &mut dn_state, &scratch,
        ).expect("prefill forward failed");
    }

    // Get first token (greedy)
    let logits = gpu.download_f32(&scratch.logits).unwrap();
    let mut next_token = llama::argmax(&logits);

    eprintln!("Generating 600 tokens (greedy)...");
    for step in 0..600 {
        let text = tokenizer.decode(&[next_token]);
        writeln!(token_file, "{next_token}").unwrap();
        write!(text_file, "{text}").unwrap();

        if step < 10 || step % 20 == 0 {
            eprintln!("  step {:3}: token={:6} {:20}", step, next_token,
                text.replace('\n', "\\n").chars().take(20).collect::<String>());
        }

        if next_token == config.eos_token { break; }

        let pos = prompt_tokens.len() + step;
        if pos >= max_seq { eprintln!("  hit max_seq at step {step}"); break; }

        qwen35::forward_scratch(
            &mut gpu, &weights, &config, next_token, pos,
            &mut kv_cache, &mut dn_state, &scratch,
        ).expect("generation forward failed");

        let new_logits = gpu.download_f32(&scratch.logits).unwrap();
        next_token = llama::argmax(&new_logits);
    }

    token_file.flush().unwrap();
    text_file.flush().unwrap();
    eprintln!("\nDumped to {out_dir}/");
    eprintln!("Next: swap GPU, run again with different output_dir, then diff token_sequence.txt");
}
