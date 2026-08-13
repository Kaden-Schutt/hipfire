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
//!       [--reps R] [--warmup W] [--batch B] [--dump-dense-acts DIR]
//!
//! Defaults: --tokens 7047 (antirez DGX-Spark prompt size), --reps 3,
//!           --warmup 1, --batch 1024 (HIPFIRE_DEEPSEEK4_PP_BATCH).
//!
//! If the tokenized prompt is shorter than --tokens it is tiled to length;
//! if longer it is truncated. This keeps the FLOP count fixed across runs
//! so prefill throughput is comparable regardless of corpus.

use hipfire_arch_deepseek4::{
    forward::{finish_dense_activation_dump, forward_prefill_batch_chunked},
    DeepseekV4, DeepseekV4State,
};
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
    let mut dump_dense_acts: Option<String> = None;
    let mut variants: Vec<String> = vec!["default".to_string()];
    let mut target_tokens: usize = 7047;
    let mut reps: usize = 3;
    let mut warmup: usize = 1;
    // Ceilings for the batched E8 decode GEMV, swept in-process against one
    // loaded trunk. 0 = WMMA token tile (the shipped path).
    let mut e8_batched: Vec<usize> = vec![0];
    // KV depth at which every arm is measured. A pos-0 window sees an empty
    // SWA window and top-k index, which understates attention; DS4 Flash caps
    // attention at window+index_topk, so any prefix past ~640 is equivalent.
    let mut prefix: usize = 0;
    // AR decode steps to time as the break-even reference (0 = skip).
    let mut ar_ref: usize = 0;
    let mut batches: Vec<usize> = vec![std::env::var("HIPFIRE_DEEPSEEK4_PP_BATCH")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(1024)];
    while let Some(flag) = args.next() {
        match flag.as_str() {
            "--prompt" => prompt_file = Some(args.next().expect("--prompt FILE")),
            "--dump-dense-acts" => {
                dump_dense_acts = Some(args.next().expect("--dump-dense-acts DIR"))
            }
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
            "--e8-batched" => {
                e8_batched = args
                    .next()
                    .expect("--e8-batched N[,N2,...]")
                    .split(',')
                    .map(|s| s.parse().unwrap())
                    .collect()
            }
            "--prefix" => prefix = args.next().expect("--prefix P").parse().unwrap(),
            "--ar-ref" => ar_ref = args.next().expect("--ar-ref N").parse().unwrap(),
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

    if let Some(out_dir) = &dump_dense_acts {
        if warmup != 0
            || reps != 1
            || variants.len() != 1
            || batches.len() != 1
            || e8_batched.len() != 1
            || prefix != 0
            || ar_ref != 0
            || target_tokens == 0
        {
            return Err(
                "--dump-dense-acts requires --warmup 0 --reps 1, one variant/batch/e8 arm, \
                 --prefix 0, --ar-ref 0, and --tokens > 0"
                    .to_string(),
            );
        }
        std::env::set_var("HIPFIRE_DS4_DENSE_ACT_DIR", out_dir);
        eprintln!("Hipfire-native P3 activation capture: {out_dir}");
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
    // `--tokens 0` is window mode: each batch size runs exactly ONE chunk of
    // `batch` tokens, which is the speculative-verify window rather than a
    // prefill. Any other value keeps the fixed-FLOP prefill sweep.
    let max_batch = batches.iter().copied().max().unwrap_or(1);
    let pool_len = prefix + target_tokens.max(max_batch);
    let mut tokens: Vec<u32> = Vec::with_capacity(pool_len);
    while tokens.len() < pool_len {
        let take = (pool_len - tokens.len()).min(base.len());
        tokens.extend_from_slice(&base[..take]);
    }
    tokens.truncate(pool_len);

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

    // One state, prefilled once to `prefix` and then shared by every arm. Each
    // measured window rewrites the same KV slots, so re-prefilling per rep
    // (9.7 s at 2048 tokens) buys nothing.
    let mut state = DeepseekV4State::new(&cfg)?;
    let start_pos = prefix as u32;
    if prefix > 0 {
        let chunk = prefix.min(1024);
        let mut pre_pbs =
            hipfire_arch_deepseek4::forward::PrefillBatchScratch::new(&mut gpu, &cfg, chunk)?;
        let t = Instant::now();
        let _ = forward_prefill_batch_chunked(
            &cfg,
            &weights,
            &mut state,
            &mut gpu,
            &tokens[..prefix],
            0,
            &mut pre_pbs,
        )?;
        gpu.hip
            .device_synchronize()
            .map_err(|e| format!("prefix sync: {e:?}"))?;
        eprintln!(
            "Prefilled {}-token prefix in {:.2}s",
            prefix,
            t.elapsed().as_secs_f64()
        );
        pre_pbs.free_gpu(&mut gpu);
    }

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

        // AR decode reference: same process, same thermal state, same KV
        // depth as the windows below, so window(B) can be divided by it
        // directly instead of against a number from another binary.
        if ar_ref > 0 {
            let tok = tokens[prefix.saturating_sub(1)];
            let mut secs: Vec<f64> = Vec::with_capacity(ar_ref);
            for i in 0..warmup + ar_ref {
                gpu.hip
                    .device_synchronize()
                    .map_err(|e| format!("ar pre-sync: {e:?}"))?;
                let t = Instant::now();
                let _ = hipfire_arch_deepseek4::forward::decode_step(
                    &cfg, &weights, &mut state, &mut gpu, tok, start_pos,
                )?;
                gpu.hip
                    .device_synchronize()
                    .map_err(|e| format!("ar post-sync: {e:?}"))?;
                let s = t.elapsed().as_secs_f64();
                eprintln!(
                    "[ar {}] 1 tok in {:.4}s = {:.2} tok/s{}",
                    i,
                    s,
                    1.0 / s,
                    if i < warmup { " (warmup)" } else { "" }
                );
                if i >= warmup {
                    secs.push(s);
                }
            }
            secs.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let med = secs[secs.len() / 2];
            println!(
                "AR-REF variant={} {:.2} ms/token ({:.2} tok/s) | pos {}",
                variant,
                med * 1e3,
                1.0 / med,
                start_pos
            );
        }

        // Batch outer, arm inner: the two arms of each B run back-to-back
        // against one `PrefillBatchScratch`, so neither an allocation nor a
        // sweep-length thermal drift sits between them.
        for &batch in &batches {
            let mut pbs =
                hipfire_arch_deepseek4::forward::PrefillBatchScratch::new(&mut gpu, &cfg, batch)?;
            let n_tok = if target_tokens == 0 {
                batch
            } else {
                target_tokens
            };
            let toks = &tokens[prefix..prefix + n_tok];

            for &e8b in &e8_batched {
                hipfire_arch_deepseek4::forward::set_e8_batched_gemv_max_batch(e8b);

                let mut secs: Vec<f64> = Vec::with_capacity(reps);
                for i in 0..warmup + reps {
                    gpu.hip
                        .device_synchronize()
                        .map_err(|e| format!("pre-sync: {e:?}"))?;
                    let t = Instant::now();
                    // Every rep re-runs the SAME positions [start_pos,
                    // start_pos+n_tok), overwriting those KV slots, so the
                    // measured shape never drifts across reps.
                    let _ = forward_prefill_batch_chunked(
                        &cfg, &weights, &mut state, &mut gpu, toks, start_pos, &mut pbs,
                    )?;
                    gpu.hip
                        .device_synchronize()
                        .map_err(|e| format!("post-sync: {e:?}"))?;
                    let s = t.elapsed().as_secs_f64();
                    eprintln!(
                        "[batch {} e8b {} {}] {} tok in {:.4}s = {:.2} tok/s{}",
                        batch,
                        e8b,
                        i,
                        n_tok,
                        s,
                        n_tok as f64 / s,
                        if i < warmup { " (warmup)" } else { "" }
                    );
                    if i >= warmup {
                        secs.push(s);
                    }
                }
                secs.sort_by(|a, b| a.partial_cmp(b).unwrap());
                let median = secs[secs.len() / 2];
                let best = secs[0];
                let calls = (n_tok as f64 / batch as f64).ceil();
                println!(
                    "WINDOW variant={} e8b={} B={} | {:.2} ms/call (best {:.2}) | {:.2} tok/s | {} tokens | pos {}",
                    variant,
                    e8b,
                    batch,
                    median * 1e3 / calls,
                    best * 1e3 / calls,
                    n_tok as f64 / median,
                    n_tok,
                    start_pos
                );
            }
            pbs.free_gpu(&mut gpu);
        }
    }
    finish_dense_activation_dump()?;
    Ok(())
}
