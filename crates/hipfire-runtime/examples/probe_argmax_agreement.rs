// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Double-D feasibility probe — p_outer measurement.
//!
//! Loads BOTH a small qwen3.5 model (e.g. 0.8B) and a large target
//! (e.g. 27B), prefills both with the same prompt, then runs a
//! **teacher-forced** per-position argmax comparison:
//!
//!   For each step i:
//!     1. Both models receive the same prefix (target's committed history).
//!     2. Run one forward on each.
//!     3. Record each model's argmax at position i.
//!     4. Advance BOTH models' state with target's argmax (teacher forcing).
//!
//! Report: match rate p_outer = (# agreeing positions) / (# positions).
//! This is the exact quantity that drives speculative-decode acceptance
//! when the small model is used as a draft for the large one.
//!
//! Usage:
//!   probe_argmax_agreement <small.mq4> <large.mq4> "<prompt>" [N=128]

use hipfire_runtime::hfq::HfqFile;
use hipfire_runtime::llama;
use hipfire_arch_qwen35::qwen35::{self, DeltaNetState, Qwen35Scratch};
use hipfire_runtime::tokenizer::Tokenizer;
use std::path::Path;
use std::time::Instant;

fn argmax_u32(logits: &[f32]) -> u32 {
    logits.iter().enumerate().fold((0u32, f32::NEG_INFINITY), |(b, bv), (i, &v)| {
        if v > bv { (i as u32, v) } else { (b, bv) }
    }).0
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 4 {
        eprintln!("usage: {} <small.mq4> <large.mq4> \"<prompt>\" [N=128]", args[0]);
        std::process::exit(1);
    }
    let small_path = &args[1];
    let large_path = &args[2];
    let prompt_text = &args[3];
    let n_steps: usize = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(128);
    let use_chatml = std::env::var("HIPFIRE_CHATML").ok().as_deref() == Some("1");

    let mut gpu = rdna_compute::Gpu::init().expect("gpu init");

    // Load both models.
    let t0 = Instant::now();
    let mut small_hfq = HfqFile::open(Path::new(small_path)).expect("small hfq");
    let small_cfg = qwen35::config_from_hfq(&small_hfq).expect("small cfg");
    eprintln!("small: dim={} layers={} heads={} vocab={}",
        small_cfg.dim, small_cfg.n_layers, small_cfg.n_heads, small_cfg.vocab_size);
    let small_weights = qwen35::load_weights(&mut small_hfq, &small_cfg, &mut gpu).expect("small load");
    eprintln!("small loaded in {:.1}s", t0.elapsed().as_secs_f32());

    let t1 = Instant::now();
    let mut large_hfq = HfqFile::open(Path::new(large_path)).expect("large hfq");
    let large_cfg = qwen35::config_from_hfq(&large_hfq).expect("large cfg");
    eprintln!("large: dim={} layers={} heads={} vocab={}",
        large_cfg.dim, large_cfg.n_layers, large_cfg.n_heads, large_cfg.vocab_size);
    assert_eq!(small_cfg.vocab_size, large_cfg.vocab_size,
        "vocab sizes must match for argmax comparison");
    let large_weights = qwen35::load_weights(&mut large_hfq, &large_cfg, &mut gpu).expect("large load");
    eprintln!("large loaded in {:.1}s", t1.elapsed().as_secs_f32());

    let tokenizer = Tokenizer::from_hfq_metadata(&large_hfq.metadata_json)
        .expect("tokenizer from large hfq");

    // Build prompt tokens.
    let mut prompt_tokens = tokenizer.encode(prompt_text);
    if use_chatml {
        let im_start = tokenizer.encode("<|im_start|>");
        let im_end   = tokenizer.encode("<|im_end|>");
        let user     = tokenizer.encode("user");
        let asst     = tokenizer.encode("assistant");
        let nl       = tokenizer.encode("\n");
        let mut chat = Vec::new();
        chat.extend_from_slice(&im_start); chat.extend_from_slice(&user); chat.extend_from_slice(&nl);
        chat.extend_from_slice(&prompt_tokens);
        chat.extend_from_slice(&im_end); chat.extend_from_slice(&nl);
        chat.extend_from_slice(&im_start); chat.extend_from_slice(&asst); chat.extend_from_slice(&nl);
        prompt_tokens = chat;
    }
    eprintln!("prompt: {} tokens", prompt_tokens.len());

    // KV caches sized for prompt + n_steps + headroom.
    let kv_seq = prompt_tokens.len() + n_steps + 16;
    let kv_mode_str = std::env::var("HIPFIRE_KV_MODE").unwrap_or_else(|_| "asym3".into());
    let mk_kv = |gpu: &mut rdna_compute::Gpu, cfg: &qwen35::Qwen35Config| -> llama::KvCache {
        match kv_mode_str.as_str() {
            "asym3" => llama::KvCache::new_gpu_asym3(gpu, cfg.n_layers, cfg.n_kv_heads, cfg.head_dim, kv_seq).unwrap(),
            "asym4" => llama::KvCache::new_gpu_asym4(gpu, cfg.n_layers, cfg.n_kv_heads, cfg.head_dim, kv_seq).unwrap(),
            _       => llama::KvCache::new_gpu_q8 (gpu, cfg.n_layers, cfg.n_kv_heads, cfg.head_dim, kv_seq).unwrap(),
        }
    };
    let mut small_kv = mk_kv(&mut gpu, &small_cfg);
    let mut large_kv = mk_kv(&mut gpu, &large_cfg);
    let mut small_dn = DeltaNetState::new(&mut gpu, &small_cfg).unwrap();
    let mut large_dn = DeltaNetState::new(&mut gpu, &large_cfg).unwrap();
    let small_scratch = Qwen35Scratch::new(&mut gpu, &small_cfg, 1).unwrap();
    let large_scratch = Qwen35Scratch::new(&mut gpu, &large_cfg, 1).unwrap();

    // Prefill both with prompt tokens (single-token forwards).
    let t_pf = Instant::now();
    for (pos, &tok) in prompt_tokens.iter().enumerate() {
        qwen35::forward_scratch(&mut gpu, &small_weights, &small_cfg, tok, pos,
            &mut small_kv, &mut small_dn, &small_scratch).expect("small prefill");
        qwen35::forward_scratch(&mut gpu, &large_weights, &large_cfg, tok, pos,
            &mut large_kv, &mut large_dn, &large_scratch).expect("large prefill");
    }
    eprintln!("prefill: {:.2}s", t_pf.elapsed().as_secs_f32());

    // Teacher-forced argmax comparison.
    let mut matches: u32 = 0;
    let mut first_divergence: Option<usize> = None;
    let mut committed: Vec<u32> = Vec::with_capacity(n_steps);
    let mut small_argmaxes: Vec<u32> = Vec::with_capacity(n_steps);

    // Seed argmax at the last prompt position (logits already in scratches
    // after prefill).
    let mut large_logits = gpu.download_f32(&large_scratch.logits).unwrap();
    let mut small_logits = gpu.download_f32(&small_scratch.logits).unwrap();

    for step in 0..n_steps {
        let large_tok = argmax_u32(&large_logits);
        let small_tok = argmax_u32(&small_logits);
        committed.push(large_tok);
        small_argmaxes.push(small_tok);
        let matched = large_tok == small_tok;
        if matched {
            matches += 1;
        } else if first_divergence.is_none() {
            first_divergence = Some(step);
        }

        // Teacher-force: advance both models with large_tok.
        let pos = prompt_tokens.len() + step;
        qwen35::forward_scratch(&mut gpu, &small_weights, &small_cfg, large_tok, pos,
            &mut small_kv, &mut small_dn, &small_scratch).expect("small fwd");
        qwen35::forward_scratch(&mut gpu, &large_weights, &large_cfg, large_tok, pos,
            &mut large_kv, &mut large_dn, &large_scratch).expect("large fwd");
        large_logits = gpu.download_f32(&large_scratch.logits).unwrap();
        small_logits = gpu.download_f32(&small_scratch.logits).unwrap();

        if step < 8 || (step % 16 == 0) {
            let lt = tokenizer.decode(&[large_tok]);
            let st = tokenizer.decode(&[small_tok]);
            eprintln!("[{:3}] large={:>6} ({:?})  small={:>6} ({:?})  match={}",
                step, large_tok, lt, small_tok, st, matched);
        }
    }

    let p_outer = matches as f32 / n_steps as f32;
    eprintln!();
    eprintln!("=== p_outer probe ===");
    eprintln!("steps:           {n_steps}");
    eprintln!("matches:         {matches}");
    eprintln!("p_outer:         {:.4}", p_outer);
    eprintln!("first divergence: {:?}", first_divergence);
    eprintln!();
    eprintln!("large (committed) text:");
    println!("{}", tokenizer.decode(&committed));
}
