// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Minimal Gemma 4 dense greedy inference — real-model e2e coherence check.
//! Loads an HFQ, runs prefill + greedy decode via `decode_step`, prints text.
//!
//! - Prepends the BOS token (2) if absent.
//! - Greedy argmax + a repetition penalty (default 1.3) over the recent window.
//! - Stops on EOS {1, 106} (`<end_of_turn>`).
//! - The embed √dim scale + final logit softcap are applied inside the forward.
//!
//! Usage: infer_gemma4 --model <hfq> [--prompt <text>] [--max N] [--rep-pen R]
#[cfg(feature = "deltanet")]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum FullKvMode {
    Asym3,
    Fwht3,
}

#[cfg(feature = "deltanet")]
fn full_kv_mode(value: &str) -> FullKvMode {
    if value == "fwht3" {
        FullKvMode::Fwht3
    } else {
        FullKvMode::Asym3
    }
}

#[cfg(all(test, feature = "deltanet"))]
mod tests {
    use super::{full_kv_mode, FullKvMode};

    #[test]
    fn kv_mode_selects_fwht3_full_cache() {
        assert_eq!(full_kv_mode("fwht3"), FullKvMode::Fwht3);
        assert_eq!(full_kv_mode(""), FullKvMode::Asym3);
    }
}

#[cfg(not(feature = "deltanet"))]
fn main() {
    eprintln!("build with --features deltanet");
}

#[cfg(feature = "deltanet")]
fn main() {
    use hipfire_arch_gemma4::lowered;
    use hipfire_runtime::hfq::HfqFile;
    use hipfire_runtime::llama::KvCache;
    use hipfire_runtime::tokenizer::Tokenizer;
    use std::path::PathBuf;

    let argv: Vec<String> = std::env::args().collect();
    let mut model: Option<PathBuf> = None;
    let mut prompt = "The capital of France is".to_string();
    let mut max: usize = 8192;
    let mut rep_pen: f32 = 1.3;
    let mut kv_mode = String::new();
    let mut token_ids: Option<Vec<u32>> = None;
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
            "--prompt-file" => {
                prompt = std::fs::read_to_string(&argv[i + 1])
                    .expect("--prompt-file: read");
                i += 2;
            }
            // Bypass the tokenizer: feed comma-separated input token ids (e.g.
            // HF-tokenized). Output token ids are printed for external decode.
            "--token-ids" => {
                token_ids = Some(
                    argv[i + 1]
                        .split(',')
                        .filter(|s| !s.is_empty())
                        .map(|s| s.trim().parse::<u32>().expect("--token-ids"))
                        .collect(),
                );
                i += 2;
            }
            "--max" => {
                max = argv[i + 1].parse().expect("--max");
                i += 2;
            }
            "--rep-pen" => {
                rep_pen = argv[i + 1].parse().expect("--rep-pen");
                i += 2;
            }
            "--kv-mode" => {
                kv_mode = argv[i + 1].clone();
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
    let cfg = lowered::config_from_hfq(&hfq).expect("config");
    let n_sliding = cfg
        .layer_types
        .iter()
        .filter(|layer| matches!(layer, lowered::LayerType::Sliding))
        .count();
    let n_full = cfg
        .layer_types
        .iter()
        .filter(|layer| matches!(layer, lowered::LayerType::Full))
        .count();
    eprintln!(
        "gemma4 dim={} layers={} sliding={} full={} vocab={} softcap={}",
        cfg.dim, cfg.n_layers, n_sliding, n_full, cfg.vocab_size, cfg.final_logit_softcapping
    );
    // Tokenizer only needed for text encode/decode; --token-ids bypasses it.
    let tok = if token_ids.is_none() {
        Some(Tokenizer::from_hfq_metadata(&hfq.metadata_json).expect("tokenizer"))
    } else {
        None
    };
    let t_load = std::time::Instant::now();
    let weights = lowered::load_weights(&mut hfq, &cfg, &mut gpu).expect("weights");
    eprintln!("loaded weights in {:.1}s", t_load.elapsed().as_secs_f64());

    // Prepend BOS (2) if absent.
    let mut prompt_ids = match token_ids {
        Some(ids) => ids,
        None => tok.as_ref().unwrap().encode(&prompt),
    };
    if prompt_ids.first() != Some(&cfg.bos_token) {
        prompt_ids.insert(0, cfg.bos_token);
    }
    eprintln!("prompt {:?} → {} tokens", prompt, prompt_ids.len());
    let max_seq = prompt_ids.len() + max + 16;
    let scratch = lowered::Gemma4Scratch::new(&mut gpu, &cfg, 1).expect("scratch");
    lowered::init_scratch_constants(&mut gpu, &scratch, cfg.full_head_dim)
        .expect("scratch constants");
    let mut kv_sliding = KvCache::new_gpu_q8_capped(
        &mut gpu,
        cfg.n_layers,
        cfg.sliding_n_kv_heads,
        cfg.sliding_head_dim,
        max_seq,
        max_seq,
    )
    .expect("sliding KV");
    let mut kv_full = match full_kv_mode(&kv_mode) {
        FullKvMode::Fwht3 => {
            eprintln!("kv_mode=fwht3: full-attention layers use FWHT-512 3-bit K + Q8_0 V");
            gpu.ensure_mq_signs().expect("fwht3 signs");
            let all_true = vec![true; n_full];
            KvCache::new_gpu_fwht3_capped_filtered_gemma4(
                &mut gpu,
                &all_true,
                cfg.full_n_kv_heads,
                cfg.full_head_dim,
                max_seq,
                max_seq,
            )
            .expect("full KV (fwht3)")
        }
        FullKvMode::Asym3 => KvCache::new_gpu_asym3_gemma4(
            &mut gpu,
            cfg.n_layers,
            cfg.full_n_kv_heads,
            cfg.full_head_dim,
            max_seq,
        )
        .expect("full KV"),
    };

    // Greedy argmax with a repetition penalty over already-emitted tokens.
    let argmax_with_penalty = |v: &mut [f32], history: &[u32], pen: f32| -> u32 {
        if pen != 1.0 {
            for &t in history {
                let idx = t as usize;
                if idx < v.len() {
                    let x = v[idx];
                    v[idx] = if x > 0.0 { x / pen } else { x * pen };
                }
            }
        }
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
        lowered::forward_scratch(
            &mut gpu,
            &weights,
            &cfg,
            t,
            pos,
            &mut kv_sliding,
            &mut kv_full,
            &scratch,
        )
        .expect("prefill");
        logits = gpu.download_f32(&scratch.logits).expect("prefill logits");
    }
    eprintln!(
        "prefill {} tok in {:.2}s",
        prompt_ids.len(),
        t0.elapsed().as_secs_f64()
    );

    // Greedy decode.
    let mut gen = Vec::new();
    let mut history = prompt_ids.clone();
    let mut pos = prompt_ids.len();
    let t1 = std::time::Instant::now();
    for _ in 0..max {
        let next = argmax_with_penalty(&mut logits, &history, rep_pen);
        // Stop on EOS (1) or <end_of_turn> (106).
        if matches!(next, 1 | 106) {
            break;
        }
        gen.push(next);
        history.push(next);
        lowered::forward_scratch(
            &mut gpu,
            &weights,
            &cfg,
            next,
            pos,
            &mut kv_sliding,
            &mut kv_full,
            &scratch,
        )
        .expect("decode");
        logits = gpu.download_f32(&scratch.logits).expect("decode logits");
        pos += 1;
    }
    let dt = t1.elapsed().as_secs_f64();
    eprintln!(
        "decoded {} tok in {:.2}s ({:.1} tok/s)",
        gen.len(),
        dt,
        gen.len() as f64 / dt
    );
    match &tok {
        Some(t) => println!(
            "=== PROMPT ===\n{prompt}\n=== GENERATION ===\n{}",
            t.decode(&gen)
        ),
        None => println!("=== GENERATION token ids ===\n{:?}", gen),
    }
    eprintln!("token ids: {:?}", &gen[..gen.len().min(60)]);
    let mut final_logit_hash = 0xcbf29ce484222325u64;
    for value in &logits {
        for byte in value.to_bits().to_le_bytes() {
            final_logit_hash ^= u64::from(byte);
            final_logit_hash = final_logit_hash.wrapping_mul(0x100000001b3);
        }
    }
    eprintln!("final logits fnv1a64: 0x{final_logit_hash:016x}");
}
