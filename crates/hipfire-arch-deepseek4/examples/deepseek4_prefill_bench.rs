// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Nick Woolmer
// hipfire — see LICENSE and NOTICE in the project root.

//! Prefill (prompt-processing) throughput benchmark for DeepSeek V4 Flash.
//!
//! Mirrors the antirez/ds4 "prefill t/s" number: process an N-token prompt
//! at pos 0 in one `forward_prefill_batch_chunked` call and report
//! tokens / wall-second. Does a throwaway warmup prefill first to JIT the
//! kernels (a cold first chunk is 10×+ slower — see CLAUDE.md), then
//! resets state and measures `--reps` times, reporting the median.
//!
//! Usage:
//!   deepseek4_prefill_bench <model.mq2lloyd> [--prompt FILE] [--tokens N]
//!       [--reps R] [--warmup W] [--batch B]
//!
//! Defaults: --tokens 7047 (antirez DGX-Spark prompt size), --reps 3,
//!           --warmup 1, --batch 1024 (HIPFIRE_DEEPSEEK4_PP_BATCH).
//!
//! If the tokenized prompt is shorter than --tokens it is tiled to length;
//! if longer it is truncated. This keeps the FLOP count fixed across runs
//! so prefill throughput is comparable regardless of corpus.

use hipfire_arch_deepseek4::{forward::forward_prefill_batch_chunked, DeepseekV4, DeepseekV4State};
use hipfire_runtime::arch::Architecture;
use hipfire_runtime::hfq::HfqFile;
use hipfire_runtime::tokenizer::Tokenizer;
use rdna_compute::Gpu;
use std::time::Instant;

fn main() -> Result<(), String> {
    let mut args = std::env::args().skip(1);
    let model_path = args.next().unwrap_or_else(|| {
        std::env::var("HIPFIRE_DEEPSEEK4_MODEL").unwrap_or_else(|_| {
            "/home/bjoern/.hipfire/models/deepseek-v4-flash.mq2lloyd".to_string()
        })
    });

    let mut prompt_file: Option<String> = None;
    let mut variants: Vec<String> = vec!["default".to_string()];
    let mut target_tokens: usize = 7047;
    let mut reps: usize = 3;
    let mut warmup: usize = 1;
    let mut batches: Vec<usize> = vec![std::env::var("HIPFIRE_DEEPSEEK4_PP_BATCH")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(1024)];
    while let Some(flag) = args.next() {
        match flag.as_str() {
            "--prompt" => prompt_file = Some(args.next().expect("--prompt FILE")),
            "--tokens" => target_tokens = args.next().expect("--tokens N").parse().unwrap(),
            "--reps" => reps = args.next().expect("--reps R").parse().unwrap(),
            "--warmup" => warmup = args.next().expect("--warmup W").parse().unwrap(),
            "--batch" => {
                batches = args
                    .next()
                    .expect("--batch B[,B2,...]")
                    .split(',')
                    .map(|s| s.parse().unwrap())
                    .collect()
            }
            "--variants" => {
                variants = args
                    .next()
                    .expect("--variants v1[,v2,...]")
                    .split(',')
                    .map(|s| s.to_string())
                    .collect()
            }
            other => panic!("unknown flag: {other}"),
        }
    }

    eprintln!("Loading DeepSeek V4 from {model_path}...");
    let mut hfq =
        HfqFile::open(std::path::Path::new(&model_path)).map_err(|e| format!("open: {e:?}"))?;
    let cfg = DeepseekV4::config_from_hfq(&hfq)?;
    let tokenizer = Tokenizer::from_hfq_metadata(&hfq.metadata_json)
        .map_err(|e| format!("tokenizer: {e:?}"))?;

    // Build a deterministic token sequence of exactly target_tokens length.
    let base: Vec<u32> = if let Some(pf) = &prompt_file {
        let text = std::fs::read_to_string(pf).map_err(|e| format!("read prompt: {e}"))?;
        tokenizer.encode(&text)
    } else {
        // Default filler: tokenize a chunk of pangram-ish prose.
        tokenizer.encode(
            "The quick brown fox jumps over the lazy dog. \
             Pack my box with five dozen liquor jugs. ",
        )
    };
    assert!(!base.is_empty(), "empty prompt token stream");
    let mut tokens: Vec<u32> = Vec::with_capacity(target_tokens);
    while tokens.len() < target_tokens {
        let take = (target_tokens - tokens.len()).min(base.len());
        tokens.extend_from_slice(&base[..take]);
    }
    tokens.truncate(target_tokens);

    let mut gpu = Gpu::init().map_err(|e| format!("gpu: {e:?}"))?;
    eprintln!("GPU: {}", gpu.arch.clone());
    let t_load = Instant::now();
    let weights = DeepseekV4::load_weights(&mut hfq, &cfg, &mut gpu)?;
    eprintln!("Loaded weights in {:.1}s", t_load.elapsed().as_secs_f64());
    eprintln!(
        "Config: layers={} hidden={} vocab={} window={} | prefill tokens={} batches={:?}",
        cfg.num_hidden_layers,
        cfg.hidden_size,
        cfg.vocab_size,
        cfg.sliding_window,
        target_tokens,
        batches
    );

    // Env vars that select MQ2-Lloyd grouped-GEMM variants in the dispatch.
    const MOE_VARS: &[&str] = &[
        "HIPFIRE_DEEPSEEK4_MOE_N32",
        "HIPFIRE_DEEPSEEK4_MOE_CND",
        "HIPFIRE_DEEPSEEK4_MOE_8W",
        "HIPFIRE_DEEPSEEK4_MOE_MMQLOAD",
        "HIPFIRE_DEEPSEEK4_MOE_NOSYNC",
    ];
    let apply_variant = |v: &str| {
        for k in MOE_VARS {
            std::env::remove_var(k);
        }
        let key = match v {
            "default" | "4w" | "lloyd4w" => None,
            "n32" => Some("HIPFIRE_DEEPSEEK4_MOE_N32"),
            "cnd" => Some("HIPFIRE_DEEPSEEK4_MOE_CND"),
            "8w" => Some("HIPFIRE_DEEPSEEK4_MOE_8W"),
            "mmqload" => Some("HIPFIRE_DEEPSEEK4_MOE_MMQLOAD"),
            "nosync" => Some("HIPFIRE_DEEPSEEK4_MOE_NOSYNC"),
            other => panic!("unknown variant {other}"),
        };
        if let Some(k) = key {
            std::env::set_var(k, "1");
        }
    };

    for variant in &variants {
        apply_variant(variant);
        for &batch in &batches {
            let pbs =
                hipfire_arch_deepseek4::forward::PrefillBatchScratch::new(&mut gpu, &cfg, batch)?;

            let run_once = |gpu: &mut Gpu| -> Result<f64, String> {
                let mut state = DeepseekV4State::new(&cfg)?;
                gpu.hip
                    .device_synchronize()
                    .map_err(|e| format!("pre-sync: {e:?}"))?;
                let t = Instant::now();
                let _ = forward_prefill_batch_chunked(
                    &cfg, &weights, &mut state, gpu, &tokens, 0, &pbs,
                )?;
                gpu.hip
                    .device_synchronize()
                    .map_err(|e| format!("post-sync: {e:?}"))?;
                Ok(t.elapsed().as_secs_f64())
            };

            for w in 0..warmup {
                let s = run_once(&mut gpu)?;
                eprintln!(
                    "[batch {} warmup {}] {} tok in {:.3}s = {:.2} tok/s",
                    batch,
                    w,
                    target_tokens,
                    s,
                    target_tokens as f64 / s
                );
            }

            let mut secs: Vec<f64> = Vec::with_capacity(reps);
            for r in 0..reps {
                let s = run_once(&mut gpu)?;
                eprintln!(
                    "[batch {} measure {}] {} tok in {:.3}s = {:.2} tok/s",
                    batch,
                    r,
                    target_tokens,
                    s,
                    target_tokens as f64 / s
                );
                secs.push(s);
            }
            secs.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let median = secs[secs.len() / 2];
            let best = secs[0];
            println!(
            "PREFILL variant={} median {:.2} tok/s (best {:.2}) | {} tokens | batch {} | target=343 t/s",
            variant,
            target_tokens as f64 / median,
            target_tokens as f64 / best,
            target_tokens,
            batch
        );
        }
    }
    Ok(())
}
