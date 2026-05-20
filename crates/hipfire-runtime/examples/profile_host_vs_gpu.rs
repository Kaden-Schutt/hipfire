// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Phase 3a Option C: isolate where forward-pass wall clock goes.
//!
//! Splits a single forward pass into three measurements:
//!   1. Time spent inside `hipModuleLaunchKernel` FFI calls (host-side
//!      launch overhead, summed across every kernel launched in the
//!      forward). This is what hipGraph would replace with one
//!      hipGraphLaunch call.
//!   2. `forward_scratch_layers` total wall time. Includes the FFI calls
//!      from (1) plus any host-side bookkeeping in qwen35.rs / dispatch.rs.
//!   3. `device_synchronize` time AFTER forward returns. This is the GPU
//!      time the host had to wait for once it had stopped issuing work.
//!
//! Interpretation:
//!   - If (1) is most of (2), the host is the bottleneck → hipGraph helps.
//!   - If (1) is small relative to (2), Rust-side dispatch.rs has overhead
//!     beyond the FFI call (kernarg packing, ensure_kernel, profiling
//!     branches) → hipGraph helps less than expected.
//!   - If (3) is most of total, the GPU is the bottleneck → hipGraph
//!     can't help much.
//!
//! Usage: profile_host_vs_gpu <model.hfq> [iters=200]

#[cfg(not(feature = "deltanet"))]
fn main() { eprintln!("build with --features deltanet"); }

#[cfg(feature = "deltanet")]
fn main() {
    use hipfire_runtime::hfq::HfqFile;
    use hipfire_runtime::llama;
    use hipfire_arch_qwen35::qwen35::{self, DeltaNetState, Qwen35Scratch};
    use std::path::Path;
    use std::time::Instant;

    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: profile_host_vs_gpu <model.hfq> [iters=200]");
        std::process::exit(1);
    }
    let model_path = &args[1];
    let iters: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(200);

    let mut gpu = rdna_compute::Gpu::init().expect("Gpu::init");
    eprintln!("Loading {}...", model_path);
    let mut hfq = HfqFile::open(Path::new(model_path)).expect("open model");
    let config = qwen35::config_from_hfq(&hfq).expect("config");
    let weights = qwen35::load_weights(&mut hfq, &config, &mut gpu).expect("weights");
    eprintln!("Loaded: {} layers, dim={}", config.n_layers, config.dim);

    let max_seq = 2048;
    let mut kv_cache = llama::KvCache::new_gpu_q8(
        &mut gpu, config.n_layers, config.n_kv_heads, config.head_dim, max_seq,
    ).unwrap();
    let mut dn_state = DeltaNetState::new_with_quant(
        &mut gpu, &config, qwen35::StateQuant::Q8,
    ).unwrap();
    let scratch = Qwen35Scratch::new(&mut gpu, &config, 128).unwrap();

    // Warmup
    let probe = 1u32;
    for pos in 0..32 {
        qwen35::forward_scratch(&mut gpu, &weights, &config, probe, pos,
            &mut kv_cache, &mut dn_state, &scratch).unwrap();
    }
    gpu.hip.device_synchronize().unwrap();

    // Measure: per-iter, capture (forward time, sync time, per-API breakdown)
    let mut forward_us: Vec<f64> = Vec::with_capacity(iters);
    let mut sync_us: Vec<f64> = Vec::with_capacity(iters);
    let mut launch_us: Vec<f64> = Vec::with_capacity(iters);
    let mut launch_count: Vec<u64> = Vec::with_capacity(iters);
    let mut dtod_us: Vec<f64> = Vec::with_capacity(iters);
    let mut dtod_count: Vec<u64> = Vec::with_capacity(iters);
    let mut htod_us: Vec<f64> = Vec::with_capacity(iters);
    let mut htod_count: Vec<u64> = Vec::with_capacity(iters);
    let mut dtoh_us: Vec<f64> = Vec::with_capacity(iters);
    let mut dtoh_count: Vec<u64> = Vec::with_capacity(iters);
    let mut memset_us: Vec<f64> = Vec::with_capacity(iters);
    let mut memset_count: Vec<u64> = Vec::with_capacity(iters);

    let outer_start = Instant::now();
    for i in 0..iters {
        hip_bridge::launch_counters::reset();

        let t1 = Instant::now();
        qwen35::forward_scratch(&mut gpu, &weights, &config, probe, 32 + i,
            &mut kv_cache, &mut dn_state, &scratch).unwrap();
        let forward = t1.elapsed();

        launch_us.push(hip_bridge::launch_counters::launch_kernel::time_ns() as f64 / 1000.0);
        launch_count.push(hip_bridge::launch_counters::launch_kernel::count());
        dtod_us.push(hip_bridge::launch_counters::memcpy_dtod::time_ns() as f64 / 1000.0);
        dtod_count.push(hip_bridge::launch_counters::memcpy_dtod::count());
        htod_us.push(hip_bridge::launch_counters::memcpy_htod::time_ns() as f64 / 1000.0);
        htod_count.push(hip_bridge::launch_counters::memcpy_htod::count());
        dtoh_us.push(hip_bridge::launch_counters::memcpy_dtoh::time_ns() as f64 / 1000.0);
        dtoh_count.push(hip_bridge::launch_counters::memcpy_dtoh::count());
        memset_us.push(hip_bridge::launch_counters::memset::time_ns() as f64 / 1000.0);
        memset_count.push(hip_bridge::launch_counters::memset::count());

        let t2 = Instant::now();
        gpu.hip.device_synchronize().unwrap();
        let sync = t2.elapsed();

        forward_us.push(forward.as_secs_f64() * 1e6);
        sync_us.push(sync.as_secs_f64() * 1e6);
    }
    let total_outer = outer_start.elapsed();

    // ─── Aggregate and report ─────────────────────────────────────────────
    let median = |v: &[f64]| -> f64 {
        let mut sorted = v.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        sorted[sorted.len() / 2]
    };
    let mean = |v: &[f64]| -> f64 { v.iter().sum::<f64>() / v.len() as f64 };
    let p99 = |v: &[f64]| -> f64 {
        let mut sorted = v.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        sorted[(sorted.len() as f64 * 0.99) as usize]
    };

    let med_count = |v: &[u64]| -> u64 {
        let mut s = v.to_vec();
        s.sort();
        s[s.len() / 2]
    };

    let n_launch = med_count(&launch_count);
    let n_dtod = med_count(&dtod_count);
    let n_htod = med_count(&htod_count);
    let n_dtoh = med_count(&dtoh_count);
    let n_memset = med_count(&memset_count);

    println!("\n=== Phase 3a Option C+: host bookkeeping breakdown ===");
    println!("Model: {}", Path::new(model_path).file_name().and_then(|s| s.to_str()).unwrap_or(model_path));
    println!("Iters: {iters}");
    println!();

    println!("{:<32} | {:>9} | {:>10} | {:>10}", "API", "calls", "median µs", "µs/call");
    println!("{}", "-".repeat(72));

    let total_us: Vec<f64> = forward_us.iter().zip(sync_us.iter()).map(|(f, s)| f + s).collect();

    let print_api = |label: &str, t: &[f64], n: u64| {
        let med = median(t);
        let per = if n > 0 { med / n as f64 } else { 0.0 };
        println!("{:<32} | {:>9} | {:>10.1} | {:>10.3}", label, n, med, per);
    };
    print_api("hipModuleLaunchKernel", &launch_us, n_launch);
    print_api("hipMemcpy (D2D)", &dtod_us, n_dtod);
    print_api("hipMemcpy (H2D)", &htod_us, n_htod);
    print_api("hipMemcpy (D2H)", &dtoh_us, n_dtoh);
    print_api("hipMemset", &memset_us, n_memset);
    println!("{}", "-".repeat(72));

    let med_launch = median(&launch_us);
    let med_dtod = median(&dtod_us);
    let med_htod = median(&htod_us);
    let med_dtoh = median(&dtoh_us);
    let med_memset = median(&memset_us);
    let total_ffi = med_launch + med_dtod + med_htod + med_dtoh + med_memset;
    let med_forward = median(&forward_us);
    let med_sync = median(&sync_us);
    let med_total = median(&total_us);
    let med_other_host = med_forward - total_ffi;

    println!("{:<32} | {:>9} | {:>10.1} |", "  → all HIP FFI", n_launch + n_dtod + n_htod + n_dtoh + n_memset, total_ffi);
    println!();

    println!("{:<32} | {:>10}", "Phase", "median µs");
    println!("{}", "-".repeat(72));
    println!("{:<32} | {:>10.1}", "forward_scratch (host wall)", med_forward);
    println!("{:<32} | {:>10.1}", "  → all HIP FFI", total_ffi);
    println!("{:<32} | {:>10.1}", "  → Rust-only (no FFI)", med_other_host);
    println!("{:<32} | {:>10.1}", "device_synchronize after", med_sync);
    println!("{:<32} | {:>10.1}", "TOTAL", med_total);
    println!();
    println!("Wall-clock attribution:");
    println!("  HIP FFI calls:           {:6.0} µs  ({:5.1}%)", total_ffi, total_ffi / med_total * 100.0);
    println!("    └ launch_kernel:       {:6.0} µs  ({:5.1}%)", med_launch, med_launch / med_total * 100.0);
    println!("    └ memcpy D2D:          {:6.0} µs  ({:5.1}%)", med_dtod, med_dtod / med_total * 100.0);
    println!("    └ memcpy H2D+D2H:      {:6.0} µs  ({:5.1}%)", med_htod + med_dtoh, (med_htod + med_dtoh) / med_total * 100.0);
    println!("    └ memset:              {:6.0} µs  ({:5.1}%)", med_memset, med_memset / med_total * 100.0);
    println!("  Rust dispatch path:      {:6.0} µs  ({:5.1}%)", med_other_host, med_other_host / med_total * 100.0);
    println!("  device_sync wait (GPU):  {:6.0} µs  ({:5.1}%)", med_sync, med_sync / med_total * 100.0);
    println!();

    let tok_per_s = iters as f64 / total_outer.as_secs_f64();
    println!("Effective throughput: {:.1} tok/s ({:.2} ms/tok)",
        tok_per_s, total_outer.as_secs_f64() * 1000.0 / iters as f64);
}
