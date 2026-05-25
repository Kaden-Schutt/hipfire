// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Per-kernel profiler for Qwen3.5 MQ4 forward pass.
//!
//! Runs the exact same forward_scratch path as bench_qwen35_mq4 /
//! greedy_dump, but wraps N generation steps in rdna_compute::profile::{start,stop}
//! and aggregates the collected ProfileEntries by kernel name. Reports
//! per-kernel total time, call count, average time per call, total bytes,
//! and effective bandwidth — sorted by total time descending so the hot
//! kernels are at the top.
//!
//! Profiling serializes kernel launches (event sync after each), so the
//! total time is NOT the same as a real wall-clock bench — it's longer
//! because async pipelining is disabled. Use bench_qwen35_mq4 for the
//! real tok/s number. Use this to see where the time goes.
//!
//! Usage: profile_qwen35_mq4 <model.hfq> [--prefill N] [--warmup N] [--profile-steps N]
//!
//! Prefill uses forward_prefill_batch (batched, fast) so you can prime to
//! ctx=4096 in a few hundred ms. The profiled phase uses forward_scratch
//! (single-token decode path) so the per-kernel breakdown reflects the
//! actual decode hot path at that context length.

#[cfg(not(feature = "deltanet"))]
fn main() { eprintln!("build with --features deltanet"); }

#[cfg(feature = "deltanet")]
fn main() {
    use hipfire_runtime::hfq::HfqFile;
    use hipfire_arch_qwen35::qwen35::{self, DeltaNetState, Qwen35Scratch};
    use hipfire_runtime::llama::{self, KvCache};
    use rdna_compute::profile;
    use std::collections::BTreeMap;
    use std::path::Path;
    use std::time::Instant;

    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: profile_qwen35_mq4 <model.hfq> [--prefill N] [--warmup N] [--profile-steps N]");
        std::process::exit(1);
    }
    let model_path = &args[1];

    let mut prefill_len: usize = 32;
    let mut warmup_len: usize = 5;
    let mut profile_steps: usize = 10;
    let mut i = 2;
    while i < args.len() {
        match args[i].as_str() {
            "--prefill"       => { prefill_len    = args[i + 1].parse().unwrap(); i += 2; }
            "--warmup"        => { warmup_len     = args[i + 1].parse().unwrap(); i += 2; }
            "--profile-steps" => { profile_steps  = args[i + 1].parse().unwrap(); i += 2; }
            other => { eprintln!("unknown arg: {other}"); std::process::exit(1); }
        }
    }

    eprintln!("=== profile_qwen35_mq4 ===");
    eprintln!("Model: {model_path}");
    eprintln!("Prefill: {prefill_len}  Warmup: {warmup_len}  Profile: {profile_steps}");

    let mut hfq = HfqFile::open(Path::new(model_path)).expect("open model");
    let config = qwen35::config_from_hfq(&hfq).expect("read config");
    eprintln!("Config: dim={} layers={} heads={} kv_heads={}",
        config.dim, config.n_layers, config.n_heads, config.n_kv_heads);

    let mut gpu = rdna_compute::Gpu::init().expect("gpu init");
    eprintln!("GPU: {}", gpu.arch);

    let t_load = Instant::now();
    let weights = qwen35::load_weights(&mut hfq, &config, &mut gpu).expect("load weights");
    eprintln!("Weights loaded in {:.2}s", t_load.elapsed().as_secs_f64());

    let kv_seq = (prefill_len + warmup_len + profile_steps + 16).max(512);
    let mut kv_cache = KvCache::new_gpu_q8(
        &mut gpu, config.n_layers, config.n_kv_heads, config.head_dim, kv_seq
    ).unwrap();
    let mut dn_state = DeltaNetState::new(&mut gpu, &config).unwrap();
    let scratch = Qwen35Scratch::new(&mut gpu, &config, 128).unwrap();

    // Deterministic fake prompt: tokens 0..prefill_len-1 (matches bench_qwen35_mq4).
    let prompt_tokens: Vec<u32> = (0..prefill_len as u32).collect();
    eprintln!("\nPrefill {prefill_len} tokens (batched, untimed)...");
    let t_prefill = Instant::now();
    qwen35::forward_prefill_batch(
        &mut gpu, &weights, &config, &prompt_tokens, 0,
        &mut kv_cache, &mut dn_state, &scratch,
        None, None, None, None,
    ).expect("prefill forward failed");
    eprintln!("  prefill: {:.1}ms", t_prefill.elapsed().as_secs_f64() * 1000.0);
    let logits = gpu.download_f32(&scratch.logits).unwrap();
    let mut next_token = llama::argmax(&logits);

    eprintln!("Warmup {warmup_len} steps (untimed)...");
    for step in 0..warmup_len {
        let pos = prompt_tokens.len() + step;
        qwen35::forward_scratch(
            &mut gpu, &weights, &config, next_token, pos,
            &mut kv_cache, &mut dn_state, &scratch,
        ).expect("warmup forward failed");
        let logits = gpu.download_f32(&scratch.logits).unwrap();
        next_token = llama::argmax(&logits);
    }

    // === PROFILED PHASE ===
    eprintln!("\n=== profiled run: {profile_steps} gen steps at ctx ~{} ===",
        prompt_tokens.len() + warmup_len);
    profile::start();
    let t_profile = Instant::now();
    for step in 0..profile_steps {
        let pos = prompt_tokens.len() + warmup_len + step;
        qwen35::forward_scratch(
            &mut gpu, &weights, &config, next_token, pos,
            &mut kv_cache, &mut dn_state, &scratch,
        ).expect("profile forward failed");
        let logits = gpu.download_f32(&scratch.logits).unwrap();
        next_token = llama::argmax(&logits);
    }
    let profile_wall_ms = t_profile.elapsed().as_secs_f64() * 1000.0;
    let entries = profile::stop().unwrap_or_default();
    eprintln!("Captured {} profile entries over {} steps", entries.len(), profile_steps);
    eprintln!("Wall time under profiling: {profile_wall_ms:.1}ms ({:.2}ms/step)",
        profile_wall_ms / profile_steps as f64);

    // Aggregate by (category, kernel)
    #[derive(Default)]
    struct Agg {
        calls: usize,
        total_us: f64,
        total_bytes: usize,
    }
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

    // Sort by total time descending
    let mut sorted: Vec<_> = by_kernel.into_iter().collect();
    sorted.sort_by(|a, b| b.1.total_us.partial_cmp(&a.1.total_us).unwrap());

    println!();
    println!(
        "{:<4} {:<10} {:<36} {:>8} {:>11} {:>10} {:>12} {:>9}",
        "rnk", "category", "kernel", "calls", "total_us", "avg_us", "total_MiB", "GiB/s"
    );
    println!("{:-<105}", "");
    let per_step_us = total_us / profile_steps as f64;
    for (rank, ((cat, name), a)) in sorted.iter().enumerate() {
        let avg_us = a.total_us / a.calls as f64;
        let mib = a.total_bytes as f64 / (1024.0 * 1024.0);
        let gbps = if a.total_us > 0.0 {
            (a.total_bytes as f64 / (1024.0 * 1024.0 * 1024.0))
                / (a.total_us / 1_000_000.0)
        } else {
            0.0
        };
        let pct = a.total_us * 100.0 / total_us;
        println!(
            "{:<4} {:<10} {:<36} {:>8} {:>10.1}us {:>9.2}us {:>10.1} MiB {:>8.1}  ({:.1}%)",
            rank + 1, cat, name, a.calls, a.total_us, avg_us, mib, gbps, pct
        );
    }
    println!("{:-<105}", "");
    println!(
        "{:<4} {:<10} {:<36} {:>8} {:>10.1}us {:>9} {:>10.1} MiB {:>8.1}",
        "",
        "TOTAL",
        "",
        entries.len(),
        total_us,
        "",
        total_bytes as f64 / (1024.0 * 1024.0),
        (total_bytes as f64 / (1024.0 * 1024.0 * 1024.0)) / (total_us / 1_000_000.0)
    );
    println!();
    println!("Per-step (averaged over {profile_steps} profiled steps):");
    println!("  kernel time: {:.2}ms", per_step_us / 1000.0);
    println!("  wall time:   {:.2}ms (profiling serializes launches)", profile_wall_ms / profile_steps as f64);
    println!("  kernel/wall overhead factor: {:.2}x",
        (profile_wall_ms / profile_steps as f64) / (per_step_us / 1000.0));
}
