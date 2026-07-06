// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
//! Minimal Cohere2-MoE (North-Mini-Code) greedy inference — real-model e2e
//! coherence gate. Loads an HFQ, prefills + greedy-decodes via `decode_step`,
//! prints text. This is the FIRST numerical-correctness gate for the forward
//! (parallel block + interleaved RoPE + NoPE + sigmoid MoE): coherent output
//! on a simple prompt validates the bring-up.
//!
//! Usage: infer --model <hfq> [--prompt <text>] [--max N] [--tokens <json>] [--eos <id>]

use hipfire_arch_cohere2moe::cohere2moe::{Cohere2MoeState, Cohere2MoeWeights};
use hipfire_arch_cohere2moe::config::Cohere2MoeConfig;
use hipfire_arch_cohere2moe::forward::{decode_step, forward_batch, forward_batch_supported};
use hipfire_runtime::hfq::HfqFile;
use hipfire_runtime::tokenizer::Tokenizer;
use std::path::PathBuf;

fn main() {
    let argv: Vec<String> = std::env::args().collect();
    let mut model: Option<PathBuf> = None;
    let mut prompt = "The capital of France is".to_string();
    let mut max: usize = 64;
    let mut tokens_path: Option<PathBuf> = None;
    let mut eos_extra: Option<u32> = None;
    let mut chunk: usize = 256;
    // Fetch the value for a value-taking flag, or exit cleanly instead of
    // panicking with an index-out-of-bounds when the flag is trailing.
    let need_val = |i: usize| -> String {
        argv.get(i + 1).cloned().unwrap_or_else(|| {
            eprintln!("missing value for {}", argv[i]);
            std::process::exit(1);
        })
    };
    let mut i = 1;
    while i < argv.len() {
        match argv[i].as_str() {
            "--model" => {
                model = Some(PathBuf::from(need_val(i)));
                i += 2;
            }
            "--prompt" => {
                prompt = need_val(i);
                i += 2;
            }
            "--max" => {
                max = need_val(i).parse().expect("--max");
                i += 2;
            }
            "--tokens" => {
                tokens_path = Some(PathBuf::from(need_val(i)));
                i += 2;
            }
            "--eos" => {
                eos_extra = Some(need_val(i).parse().expect("--eos"));
                i += 2;
            }
            "--chunk" => {
                chunk = need_val(i).parse().expect("--chunk");
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
    assert_eq!(
        hfq.arch_id, 12,
        "infer(cohere2moe): expected arch_id 12, got {}",
        hfq.arch_id
    );
    let cfg = Cohere2MoeConfig::from_hfq(&hfq).expect("config");
    eprintln!(
        "cohere2moe hidden={} layers={} experts={}/{} dense_prefix={} vocab={}",
        cfg.hidden_size,
        cfg.num_hidden_layers,
        cfg.num_experts,
        cfg.num_experts_per_tok,
        cfg.first_k_dense_replace,
        cfg.vocab_size,
    );
    let tok = Tokenizer::from_hfq_metadata(&hfq.metadata_json).expect("tokenizer");
    let t_load = std::time::Instant::now();
    let weights = Cohere2MoeWeights::load(&mut hfq, &cfg, &mut gpu).expect("weights");
    eprintln!("loaded weights in {:.1}s", t_load.elapsed().as_secs_f64());

    let prompt_ids: Vec<u32> = if let Some(tp) = &tokens_path {
        let s = std::fs::read_to_string(tp).expect("read --tokens");
        let v: Vec<i64> = serde_json::from_str(&s).expect("parse --tokens json");
        v.into_iter().map(|t| t as u32).collect()
    } else {
        tok.encode(&prompt)
    };
    eprintln!("prompt {:?} → {} tokens", prompt, prompt_ids.len());
    let max_seq = prompt_ids.len() + max + 16;
    let mut state = Cohere2MoeState::new_with_max_seq(&mut gpu, &cfg, max_seq).expect("state");

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

    // DPM warmup BEFORE the timed prefill. `infer` is not one of the canonical
    // bench tools, so without this each fresh process measures at idle DPM
    // clocks — ~5-10% low decode and a worse cold prefill (see
    // docs/methodology/perf-benchmarking.md). Mirrors bench_qwen35_mq4. Memset
    // loop pins sclk/mclk high; does NOT dispatch the model kernels (so JIT is
    // warmed separately by a throwaway run, not here). Default OFF.
    if let Ok(secs_str) = std::env::var("HIPFIRE_DPM_WARMUP_SECS") {
        let secs: f32 = secs_str.parse().unwrap_or(0.0);
        if secs > 0.0 {
            eprintln!("=== DPM warmup ({secs:.1}s, pre-prefill) ===");
            gpu.dpm_warmup(secs).expect("dpm warmup");
        }
    }

    // Prefill: batched (read each weight once for all prompt tokens, chunked
    // ≤64) when the tier supports it (MQ4/MQ6); per-token decode_step otherwise.
    let t0 = std::time::Instant::now();
    let mut logits = Vec::new();
    let batched = forward_batch_supported(&weights);
    if batched {
        let mut i = 0;
        while i < prompt_ids.len() {
            let end = (i + chunk).min(prompt_ids.len());
            logits = forward_batch(&cfg, &weights, &mut state, &mut gpu, &prompt_ids[i..end], i)
                .expect("forward_batch");
            i = end;
        }
    } else {
        for (pos, &t) in prompt_ids.iter().enumerate() {
            logits =
                decode_step(&cfg, &weights, &mut state, &mut gpu, t, pos as u32).expect("prefill");
        }
    }
    eprintln!(
        "prefill {} tok in {:.2}s [{}]",
        prompt_ids.len(),
        t0.elapsed().as_secs_f64(),
        if batched { "batched" } else { "per-token" }
    );

    // Greedy decode. Cohere2 eos = <|END_OF_TURN_TOKEN|> (255001); bos=2, pad=0.
    let mut gen = Vec::new();
    let mut pos = prompt_ids.len();
    let t1 = std::time::Instant::now();
    for _ in 0..max {
        let next = argmax(&logits);
        gen.push(next);
        if matches!(next, 255001 | 0) || Some(next) == eos_extra {
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
    println!("GEN_IDS_JSON: {}", serde_json::to_string(&gen).unwrap());
}
