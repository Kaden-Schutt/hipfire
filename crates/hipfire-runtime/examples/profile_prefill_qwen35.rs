// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kevin Read
// hipfire — see LICENSE and NOTICE in the project root.

//! Per-kernel profiler for Qwen3.5 batched PREFILL (forward_prefill_batch).
//!
//! Companion to `profile_qwen35_mq4` (which profiles the decode hot path).
//! This binary wraps a single `forward_prefill_batch` call in
//! `rdna_compute::profile::{start,stop}` so we get per-kernel wall-time
//! attribution for the batched prefill path — i.e. the QKVZA / QKV / gate_up
//! MMQ paths that the gfx906 fused-projection MMQ port targets.
//!
//! Why batched-only:
//!   forward_prefill_batch uses the batched MMQ + fused-dp4a kernels at
//!   batch_size = n_prompt_tokens. forward_scratch (the decode path) uses
//!   batch_size=1 single-token gemv kernels. The gfx906 fused-projection MMQ
//!   port lives entirely in the batched path; profiling decode would miss it.
//!
//! Use case: docs/plans/gfx906_prefill_kernels.md §6 step 1 probe — confirm
//! that QKVZA's 4 separate gemm_hfq4g256_mmq_set_gfx906 calls dominate enough
//! wall time that fusing into one kernel is worth the engineering.
//!
//! Usage:
//!   profile_prefill_qwen35 <model.hfq> [--prefill N] [--warmup N] [--kv-mode asym3|q8]

#[cfg(not(feature = "deltanet"))]
fn main() { eprintln!("build with --features deltanet"); }

#[cfg(feature = "deltanet")]
fn main() {
    use hipfire_runtime::hfq::HfqFile;
    use hipfire_arch_qwen35::qwen35::{self, DeltaNetState, Qwen35Scratch};
    use hipfire_runtime::llama::KvCache;
    use rdna_compute::profile;
    use std::collections::BTreeMap;
    use std::path::Path;
    use std::time::Instant;

    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: profile_prefill_qwen35 <model.hfq> [--prefill N] [--warmup N] [--kv-mode asym3|q8]");
        std::process::exit(1);
    }
    let model_path = &args[1];

    let mut prefill_len: usize = 256;
    let mut warmup_iters: usize = 1;
    let mut kv_mode: String = "asym3".into();
    let mut i = 2;
    while i < args.len() {
        match args[i].as_str() {
            "--prefill" => { prefill_len = args[i + 1].parse().unwrap(); i += 2; }
            "--warmup"  => { warmup_iters = args[i + 1].parse().unwrap(); i += 2; }
            "--kv-mode" => { kv_mode = args[i + 1].clone(); i += 2; }
            other => { eprintln!("unknown arg: {other}"); std::process::exit(1); }
        }
    }

    eprintln!("=== profile_prefill_qwen35 ===");
    eprintln!("Model: {model_path}");
    eprintln!("Prefill: {prefill_len}  Warmup iters: {warmup_iters}  KV mode: {kv_mode}");

    let mut hfq = HfqFile::open(Path::new(model_path)).expect("open model");
    let config = qwen35::config_from_hfq(&hfq).expect("read config");
    eprintln!("Config: dim={} layers={} heads={} kv_heads={}",
        config.dim, config.n_layers, config.n_heads, config.n_kv_heads);

    let mut gpu = rdna_compute::Gpu::init().expect("gpu init");
    eprintln!("GPU: {}", gpu.arch);

    let t_load = Instant::now();
    let weights = qwen35::load_weights(&mut hfq, &config, &mut gpu).expect("load weights");
    eprintln!("Weights loaded in {:.2}s", t_load.elapsed().as_secs_f64());

    // KV cache sized for prefill + a few extra tokens of headroom.
    let kv_seq = prefill_len + 32;
    let mut kv_cache = match kv_mode.as_str() {
        "asym3" => KvCache::new_gpu_asym3(
            &mut gpu, config.n_layers, config.n_kv_heads, config.head_dim, kv_seq,
        ).unwrap(),
        "q8" => KvCache::new_gpu_q8(
            &mut gpu, config.n_layers, config.n_kv_heads, config.head_dim, kv_seq,
        ).unwrap(),
        other => {
            eprintln!("unknown --kv-mode {other}; use asym3 or q8");
            std::process::exit(1);
        }
    };
    let mut dn_state = DeltaNetState::new(&mut gpu, &config).unwrap();
    let scratch = Qwen35Scratch::new(&mut gpu, &config, 128).unwrap();

    // Deterministic synthetic prompt: tokens 0..prefill_len-1.
    let prompt_tokens: Vec<u32> = (0..prefill_len as u32).collect();

    // Warmup pass(es). Resets DN state and reuses position 0 each iter so the
    // KV cache writes overwrite themselves; the profiler is only enabled for
    // the final measured pass.
    for w in 0..warmup_iters {
        dn_state.reset(&mut gpu);
        let t = Instant::now();
        qwen35::forward_prefill_batch(
            &mut gpu, &weights, &config, &prompt_tokens, 0,
            &mut kv_cache, &mut dn_state, &scratch,
            None, None, None, None,
        ).expect("warmup prefill failed");
        eprintln!("warmup {}: {:.1}ms", w + 1, t.elapsed().as_secs_f64() * 1000.0);
    }

    // === PROFILED PASS ===
    dn_state.reset(&mut gpu);
    eprintln!("\n=== profiled forward_prefill_batch (B={prefill_len}) ===");
    profile::start();
    let t_profile = Instant::now();
    qwen35::forward_prefill_batch(
        &mut gpu, &weights, &config, &prompt_tokens, 0,
        &mut kv_cache, &mut dn_state, &scratch,
        None, None, None, None,
    ).expect("profile prefill failed");
    let profile_wall_ms = t_profile.elapsed().as_secs_f64() * 1000.0;
    let entries = profile::stop().unwrap_or_default();
    eprintln!("Captured {} profile entries", entries.len());
    eprintln!("Wall time under profiling: {profile_wall_ms:.1}ms (profiling serializes launches, so this is slower than a real prefill)");

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

    let mut sorted: Vec<_> = by_kernel.into_iter().collect();
    sorted.sort_by(|a, b| b.1.total_us.partial_cmp(&a.1.total_us).unwrap());

    println!();
    println!(
        "{:<4} {:<10} {:<48} {:>8} {:>12} {:>10} {:>12} {:>9}",
        "rnk", "category", "kernel", "calls", "total_us", "avg_us", "total_MiB", "GiB/s"
    );
    println!("{:-<118}", "");
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
            "{:<4} {:<10} {:<48} {:>8} {:>10.1}us {:>9.2}us {:>10.1} MiB {:>8.1}  ({:.1}%)",
            rank + 1, cat, name, a.calls, a.total_us, avg_us, mib, gbps, pct
        );
    }
    println!("{:-<118}", "");
    println!(
        "{:<4} {:<10} {:<48} {:>8} {:>10.1}us {:>9} {:>10.1} MiB {:>8.1}",
        "", "TOTAL", "", entries.len(), total_us, "",
        total_bytes as f64 / (1024.0 * 1024.0),
        (total_bytes as f64 / (1024.0 * 1024.0 * 1024.0)) / (total_us / 1_000_000.0)
    );

    // Fused-projection MMQ attribution summary — the §6.1 probe answer.
    // Counts gemm_hfq4g256_mmq_set_gfx906 calls and infers per-layer
    // QKV / QKVZA / gate_up wall, assuming the dispatcher pattern:
    //   FullAttention layer:    QKV (3 calls) + gate_up (2 calls) = 5 calls/layer
    //   LinearAttention layer:  QKVZA (2-4 calls) + gate_up (2 calls) = 4-6 calls/layer
    // We can't distinguish QKV vs QKVZA vs gate_up by name alone, but per-call
    // wall time × call count gives us the upper bound on fusion savings.
    println!();
    println!("=== fused-projection MMQ attribution (gfx906 §6.1 probe) ===");
    let mmq_set = sorted.iter()
        .find(|((_, name), _)| *name == "gemm_hfq4g256_mmq_set_gfx906")
        .map(|(_, a)| (a.calls, a.total_us));
    let mmq_residual = sorted.iter()
        .find(|((_, name), _)| *name == "gemm_hfq4g256_residual_mmq" || *name == "gemm_hfq4g256_residual_mmq_rdna2")
        .map(|(_, a)| (a.calls, a.total_us));
    if let Some((calls, t_us)) = mmq_set {
        let avg = t_us / calls as f64;
        let n_layers = config.n_layers as f64;
        let per_layer_calls = calls as f64 / n_layers;
        let per_layer_us = t_us / n_layers;
        println!("mmq_set_gfx906: {calls} total calls = {per_layer_calls:.1}/layer  ({avg:.1}us avg, {t_us:.0}us total = {:.1}% of all profiled work)",
            t_us * 100.0 / total_us);
        println!("  per-layer mmq_set wall:        {per_layer_us:.0}us");
        // Upper bound for fusion: if 4-way QKVZA collapses to 1 kernel at the same per-call cost,
        // we save (3/4) of the 4 calls' wall. But realistically only the X-tile-load is saved,
        // which is some fraction. Report the linear upper bound + the bandwidth at which the
        // mmq_set is operating.
        println!("  upper-bound fusion savings (3 of 4 calls' X-tile-loads on QKVZA): up to ~{:.0}% of QKVZA wall",
            75.0);
        println!("  (actual upside is X-load_fraction × {:.0}% — confirm via L2 hit rate or analytical X-bytes/total-bytes ratio)", 75.0);
    } else {
        println!("mmq_set_gfx906: NOT SEEN (dispatcher took fallback path) — re-run with longer prompt or check should_use_mmq gating");
    }
    if let Some((calls, t_us)) = mmq_residual {
        println!("residual_mmq (wo): {calls} calls, {t_us:.0}us total ({:.1}% of all profiled work)",
            t_us * 100.0 / total_us);
    }
    println!();
    println!("(prefill batch_size = {prefill_len}; for arch gfx906, MMQ enables at batch_size >= 8 by default)");
}
