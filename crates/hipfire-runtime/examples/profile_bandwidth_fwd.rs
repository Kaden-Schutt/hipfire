// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Phase 1 of the bandwidth-ceiling branch: profile where memory bandwidth
//! goes in one forward pass on gfx1100. Uses hipEvent per-kernel timing and
//! analytical byte counts, aggregates by category, and prints breakdown tables.
//!
//! Usage: profile_bandwidth_fwd <model.hfq> [warmup] [measure]
//!   warmup:  number of warmup forward passes before measuring (default 32)
//!   measure: number of forward passes to aggregate into the tables (default 1)

#[cfg(not(feature = "deltanet"))]
fn main() { eprintln!("Build with --features deltanet"); }

#[cfg(feature = "deltanet")]
fn main() {
    use hipfire_runtime::hfq::HfqFile;
    use hipfire_runtime::llama;
    use hipfire_arch_qwen35::qwen35::{self, DeltaNetState, Qwen35Scratch};
    use rdna_compute::profile;
    use std::collections::BTreeMap;
    use std::path::Path;

    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: profile_bandwidth_fwd <model.hfq> [warmup=32] [measure=1]");
        std::process::exit(1);
    }
    let model_path = &args[1];
    let warmup: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(32);
    let measure: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(1);

    // Hardware ceiling for gfx1100 RX 7900 XTX: 960 GB/s theoretical VRAM bandwidth.
    // Reference: spec'd at 960 GB/s (384-bit × 20 Gbps GDDR6).
    const PEAK_BW_GBS: f64 = 960.0;

    let mut gpu = rdna_compute::Gpu::init().expect("GPU init failed");
    eprintln!("Loading {}...", model_path);
    let mut hfq = HfqFile::open(Path::new(model_path)).expect("open model");
    let config = qwen35::config_from_hfq(&hfq).expect("config");
    let weights = qwen35::load_weights(&mut hfq, &config, &mut gpu).expect("weights");
    eprintln!("Loaded: {} layers, dim={}", config.n_layers, config.dim);

    let max_seq = 2048usize;
    let mut kv_cache = llama::KvCache::new_gpu_q8(
        &mut gpu, config.n_layers, config.n_kv_heads, config.head_dim, max_seq,
    ).unwrap();
    let mut dn_state = DeltaNetState::new_with_quant(
        &mut gpu, &config, qwen35::StateQuant::Q8,
    ).unwrap();
    let scratch = Qwen35Scratch::new(&mut gpu, &config, 128).unwrap();

    // Warmup: no profiling. Fills KV cache and hot kernel cache.
    let probe_token: u32 = 1;
    for pos in 0..warmup {
        qwen35::forward_scratch(&mut gpu, &weights, &config, probe_token, pos,
            &mut kv_cache, &mut dn_state, &scratch).unwrap();
    }
    gpu.hip.device_synchronize().unwrap();

    // Measurement: enable profiling, run `measure` forwards, drain entries.
    profile::start();
    let start_pos = warmup;
    for i in 0..measure {
        qwen35::forward_scratch(&mut gpu, &weights, &config, probe_token, start_pos + i,
            &mut kv_cache, &mut dn_state, &scratch).unwrap();
    }
    gpu.hip.device_synchronize().unwrap();
    let entries = profile::stop().unwrap_or_default();

    if entries.is_empty() {
        eprintln!("No profile entries collected — instrumentation may be missing.");
        std::process::exit(1);
    }

    // ─── Aggregate per-category ───────────────────────────────────────────
    #[derive(Default, Clone)]
    struct CatStats {
        time_us: f64,
        bytes: u128,
        count: usize,
    }
    let mut by_category: BTreeMap<&'static str, CatStats> = BTreeMap::new();
    let mut by_kernel: BTreeMap<&'static str, CatStats> = BTreeMap::new();

    for e in &entries {
        let c = by_category.entry(e.category).or_default();
        c.time_us += e.time_us;
        c.bytes += e.bytes as u128;
        c.count += 1;
        let k = by_kernel.entry(e.kernel).or_default();
        k.time_us += e.time_us;
        k.bytes += e.bytes as u128;
        k.count += 1;
    }

    let total_time_us: f64 = by_category.values().map(|c| c.time_us).sum();
    let total_bytes: u128 = by_category.values().map(|c| c.bytes).sum();
    let total_per_fwd_us = total_time_us / measure as f64;

    // ─── Table 1: category breakdown ──────────────────────────────────────
    println!("\n=== Bandwidth breakdown: {} (measured over {} forward passes) ===",
        Path::new(model_path).file_name().and_then(|s| s.to_str()).unwrap_or(model_path),
        measure);
    println!("Peak DRAM bandwidth (gfx1100 assumed): {:.0} GB/s\n", PEAK_BW_GBS);
    println!("{:<15} | {:>10} | {:>8} | {:>10} | {:>10} | {:>7}",
        "Category", "Time (ms)", "% fwd", "Bytes (MB)", "BW (GB/s)", "Util %");
    println!("{}", "-".repeat(80));

    for (cat, s) in &by_category {
        let time_ms = s.time_us / 1000.0 / measure as f64;
        let pct = s.time_us / total_time_us * 100.0;
        let bytes_mb = s.bytes as f64 / 1e6 / measure as f64;
        let bw_gbs = if s.time_us > 0.0 {
            (s.bytes as f64) / (s.time_us / 1e6) / 1e9
        } else { 0.0 };
        let util = bw_gbs / PEAK_BW_GBS * 100.0;
        println!("{:<15} | {:>10.3} | {:>7.1}% | {:>10.2} | {:>10.1} | {:>6.1}%",
            cat, time_ms, pct, bytes_mb, bw_gbs, util);
    }
    println!("{}", "-".repeat(80));
    let total_time_ms = total_time_us / 1000.0 / measure as f64;
    let total_bytes_mb = total_bytes as f64 / 1e6 / measure as f64;
    let total_bw = if total_time_us > 0.0 {
        (total_bytes as f64) / (total_time_us / 1e6) / 1e9
    } else { 0.0 };
    let total_util = total_bw / PEAK_BW_GBS * 100.0;
    println!("{:<15} | {:>10.3} | {:>7.1}% | {:>10.2} | {:>10.1} | {:>6.1}%",
        "TOTAL", total_time_ms, 100.0, total_bytes_mb, total_bw, total_util);
    println!("Effective tok/s (summed per-kernel wall time): {:.1}",
        1000.0 / total_time_ms);

    // ─── Table 2: kernel breakdown within GEMV category ───────────────────
    println!("\n=== GEMV kernel breakdown (sorted by time) ===");
    let mut gemv_kernels: Vec<(&&'static str, &CatStats)> = by_kernel.iter()
        .filter(|(name, _)| {
            by_kernel.contains_key(*name) &&
            ["gemv_hfq4g256", "gemm_hfq4g256"].contains(name)
        })
        .collect();
    gemv_kernels.sort_by(|a, b| b.1.time_us.partial_cmp(&a.1.time_us).unwrap());

    println!("{:<28} | {:>12} | {:>10} | {:>8} | {:>10} | {:>7} | {:>8}",
        "Kernel", "Avg µs", "Total ms", "Count", "Bytes MB", "BW GB/s", "Util %");
    println!("{}", "-".repeat(96));
    for (name, s) in &gemv_kernels {
        let avg_us = s.time_us / s.count as f64;
        let total_ms = s.time_us / 1000.0 / measure as f64;
        let bytes_mb = s.bytes as f64 / 1e6 / measure as f64;
        let bw = (s.bytes as f64) / (s.time_us / 1e6) / 1e9;
        let util = bw / PEAK_BW_GBS * 100.0;
        println!("{:<28} | {:>12.2} | {:>10.3} | {:>8} | {:>10.2} | {:>7.1} | {:>7.1}%",
            name, avg_us, total_ms, s.count / measure.max(1), bytes_mb, bw, util);
    }

    // ─── Table 3: all kernels by total time (top 20) ──────────────────────
    println!("\n=== Top kernels by total time (all categories) ===");
    let mut all_kernels: Vec<(&&'static str, &CatStats)> = by_kernel.iter().collect();
    all_kernels.sort_by(|a, b| b.1.time_us.partial_cmp(&a.1.time_us).unwrap());
    println!("{:<28} | {:>12} | {:>10} | {:>8} | {:>10} | {:>7} | {:>8}",
        "Kernel", "Avg µs", "Total ms", "Count", "Bytes MB", "BW GB/s", "Util %");
    println!("{}", "-".repeat(96));
    for (name, s) in all_kernels.iter().take(20) {
        let avg_us = s.time_us / s.count as f64;
        let total_ms = s.time_us / 1000.0 / measure as f64;
        let bytes_mb = s.bytes as f64 / 1e6 / measure as f64;
        let bw = (s.bytes as f64) / (s.time_us / 1e6) / 1e9;
        let util = bw / PEAK_BW_GBS * 100.0;
        println!("{:<28} | {:>12.2} | {:>10.3} | {:>8} | {:>10.2} | {:>7.1} | {:>7.1}%",
            name, avg_us, total_ms, s.count / measure.max(1), bytes_mb, bw, util);
    }
}
