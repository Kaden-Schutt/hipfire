// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Nick Woolmer
// hipfire — see LICENSE and NOTICE in the project root.

//! Per-kernel profiler for DeepSeek V4 batched PREFILL
//! (forward_prefill_batch_chunked). Companion to the qwen35 profiler at
//! `profile_prefill_qwen35.rs`.
//!
//! Usage:
//!   profile_prefill_deepseek4 <model.mq2lloyd> [--prefill N] [--warmup N]
//!                              [--pp-batch N] [--mtp-fill]
//!                              [--gen N] [--no-profile]

#[cfg(not(feature = "deltanet"))]
fn main() { eprintln!("build with --features deltanet"); }

#[cfg(feature = "deltanet")]
fn main() {
    use hipfire_runtime::hfq::HfqFile;
    use hipfire_arch_deepseek4::{DeepseekV4, DeepseekV4State};
    use hipfire_runtime::arch::Architecture;
    use rdna_compute::profile;
    use std::collections::BTreeMap;
    use std::path::Path;
    use std::time::Instant;

    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: profile_prefill_deepseek4 <model.mq2lloyd> [--prefill N] [--warmup N] [--pp-batch N] [--mtp-fill]");
        std::process::exit(1);
    }
    let model_path = &args[1];

    let mut prefill_len: usize = 2048;
    let mut warmup_iters: usize = 1;
    let mut pp_batch: usize = 1024;
    let mut mtp_fill = false;
    let mut gen_steps: usize = 0;
    let mut no_profile = false;
    let mut i = 2;
    while i < args.len() {
        match args[i].as_str() {
            "--prefill"  => { prefill_len = args[i + 1].parse().unwrap(); i += 2; }
            "--warmup"   => { warmup_iters = args[i + 1].parse().unwrap(); i += 2; }
            "--pp-batch" => { pp_batch = args[i + 1].parse().unwrap(); i += 2; }
            "--mtp-fill" => { mtp_fill = true; i += 1; }
            "--gen"      => { gen_steps = args[i + 1].parse().unwrap(); i += 2; }
            "--no-profile" => { no_profile = true; i += 1; }
            other => { eprintln!("unknown arg: {other}"); std::process::exit(1); }
        }
    }

    eprintln!("=== profile_prefill_deepseek4 ===");
    eprintln!("Model: {model_path}");
    eprintln!(
        "Prefill: {prefill_len}  Warmup: {warmup_iters}  PP-batch: {pp_batch}  MTP-fill: {mtp_fill}  Gen: {gen_steps}  No-profile: {no_profile}"
    );

    let mut hfq = HfqFile::open(Path::new(model_path)).expect("open model");
    let config = <DeepseekV4 as Architecture>::config_from_hfq(&hfq).expect("read config");
    eprintln!("Config: hidden={} layers={} heads={} kv_heads={}",
        config.hidden_size, config.num_hidden_layers,
        config.num_attention_heads, config.num_key_value_heads);

    let mut gpu = rdna_compute::Gpu::init().expect("gpu init");
    eprintln!("GPU: {}", gpu.arch);

    let t_load = Instant::now();
    let weights = <DeepseekV4 as Architecture>::load_weights(&mut hfq, &config, &mut gpu)
        .expect("load weights");
    eprintln!("Weights loaded in {:.2}s", t_load.elapsed().as_secs_f64());

    let mut state = DeepseekV4State::new(&config).expect("state");
    let pbs = hipfire_arch_deepseek4::forward::PrefillBatchScratch::new(&mut gpu, &config, pp_batch)
        .expect("pbs");

    // Deterministic synthetic prompt.
    let prompt_tokens: Vec<u32> = (0..prefill_len as u32).map(|t| (t % 1000) + 100).collect();

    let run_prefill = |state: &mut DeepseekV4State, gpu: &mut rdna_compute::Gpu| {
        state.reset();
        let _ = gpu.hip.device_synchronize();
        let t = Instant::now();
        let logits = if mtp_fill {
            hipfire_arch_deepseek4::forward::prefill_with_mtp_fill(
                &config, &weights, state, gpu, &pbs, &prompt_tokens, 0,
            ).expect("mtp-fill prefill failed")
        } else {
            hipfire_arch_deepseek4::forward::forward_prefill_batch_chunked(
                &config, &weights, state, gpu, &prompt_tokens, 0, &pbs,
            ).expect("prefill failed")
        };
        let _ = gpu.hip.device_synchronize();
        (t.elapsed().as_secs_f64() * 1000.0, logits)
    };

    // Warmup.
    for w in 0..warmup_iters {
        let (ms, _) = run_prefill(&mut state, &mut gpu);
        eprintln!(
            "warmup {}: {:.1}ms ({:.1} tok/s)",
            w + 1,
            ms,
            prefill_len as f64 * 1000.0 / ms.max(1.0),
        );
    }

    if no_profile {
        let (prefill_ms, logits) = run_prefill(&mut state, &mut gpu);
        {
            // Correctness probe for MoE-kernel A/B (e.g. HIPFIRE_DEEPSEEK4_MOE_N32):
            // n32 should be bit-identical to the 4w path. argmax + sum + max over
            // the final-position vocab logits is a cheap full-output checksum.
            let am = hipfire_arch_deepseek4::spec_decode::logits_argmax(&logits);
            let s: f64 = logits.iter().map(|&v| v as f64).sum();
            let mx = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            eprintln!("PREFILL_CHECK argmax={am} logit_sum={s:.4} logit_max={mx:.6} n={}", logits.len());
        }
        let prefill_tok_s = prefill_len as f64 * 1000.0 / prefill_ms.max(1.0);
        eprintln!(
            "real prefill: {prefill_ms:.1}ms ({prefill_tok_s:.1} tok/s)"
        );

        let mut decode_ms = 0.0f64;
        let mut decode_tok_s = 0.0f64;
        if gen_steps > 0 {
            let mut next_tok = hipfire_arch_deepseek4::spec_decode::logits_argmax(&logits) as u32;
            let pos_after_prefill = state.n_tokens as u32;
            let _ = gpu.hip.device_synchronize();
            let t_decode = Instant::now();
            for step in 0..gen_steps {
                let next_logits = hipfire_arch_deepseek4::forward::decode_step_with_graph(
                    &config,
                    &weights,
                    &mut state,
                    &mut gpu,
                    next_tok,
                    pos_after_prefill + step as u32,
                )
                .expect("decode failed");
                next_tok = hipfire_arch_deepseek4::spec_decode::logits_argmax(&next_logits) as u32;
            }
            let _ = gpu.hip.device_synchronize();
            decode_ms = t_decode.elapsed().as_secs_f64() * 1000.0;
            decode_tok_s = gen_steps as f64 * 1000.0 / decode_ms.max(1.0);
            eprintln!(
                "real decode: {gen_steps} steps in {decode_ms:.1}ms ({decode_tok_s:.2} tok/s)"
            );
        }
        println!(
            "REAL_SUMMARY  prefill_tok_s={prefill_tok_s:.1}  prefill_wall_ms={prefill_ms:.2}  gen_steps={gen_steps}  gen_tok_s={decode_tok_s:.2}  gen_wall_ms={decode_ms:.2}"
        );
        return;
    }

    eprintln!("\n=== profiled forward_prefill_batch_chunked (prompt={prefill_len}) ===");
    profile::start();
    let (profile_wall_ms, _) = run_prefill(&mut state, &mut gpu);
    let entries = profile::stop().unwrap_or_default();
    eprintln!("Captured {} profile entries", entries.len());
    eprintln!(
        "Wall under profiling: {profile_wall_ms:.1}ms ({:.1} tok/s; profiler serializes launches)",
        prefill_len as f64 * 1000.0 / profile_wall_ms.max(1.0),
    );

    #[derive(Default)]
    struct Agg { calls: usize, total_us: f64, total_bytes: usize }
    let mut by_kernel: BTreeMap<(&'static str, &'static str), Agg> = BTreeMap::new();
    let mut total_us = 0.0f64;
    let mut total_bytes = 0usize;
    for e in &entries {
        let a = by_kernel.entry((e.category, e.kernel)).or_default();
        a.calls += 1;
        a.total_us += e.time_us;
        a.total_bytes += e.bytes;
        total_us += e.time_us;
        total_bytes += e.bytes;
    }
    let mut sorted: Vec<_> = by_kernel.into_iter().collect();
    sorted.sort_by(|a, b| b.1.total_us.partial_cmp(&a.1.total_us).unwrap());

    println!();
    println!(
        "{:<4} {:<10} {:<48} {:>8} {:>12} {:>10} {:>12} {:>9} {:>5}",
        "rnk", "category", "kernel", "calls", "total_us", "avg_us", "total_MiB", "GiB/s", "%"
    );
    println!("{:-<128}", "");
    for (rank, ((cat, name), a)) in sorted.iter().enumerate().take(40) {
        let avg_us = a.total_us / a.calls as f64;
        let mib = a.total_bytes as f64 / (1024.0 * 1024.0);
        let gbps = if a.total_us > 0.0 {
            (a.total_bytes as f64 / (1024.0_f64.powi(3))) / (a.total_us / 1_000_000.0)
        } else { 0.0 };
        let pct = a.total_us / total_us * 100.0;
        println!(
            "{:<4} {:<10} {:<48} {:>8} {:>12.1} {:>10.2} {:>12.1} {:>9.1} {:>5.1}",
            rank + 1, cat, name, a.calls, a.total_us, avg_us, mib, gbps, pct
        );
    }
    println!("{:-<128}", "");
    let total_mib = total_bytes as f64 / (1024.0 * 1024.0);
    println!(
        "TOTAL    {:<58}              {:>12.1} {:<10} {:>12.1}",
        "", total_us, "", total_mib
    );
}
