//! Per-kernel profiler for ONE Qwen3.5 batched prefill call.
//!
//! Sister to `profile_qwen35_mq4` (which profiles the decode hot path);
//! this one wraps a single `forward_prefill_batch(N=240)` invocation
//! with `profile::start/stop` so the GEMM kernels (MMQ family, qkvza
//! split, residual y64) and the non-GEMM kernels (RMSnorm, rotate,
//! attention, KV writes, lm_head) all appear in the same per-kernel
//! breakdown — letting us see where the ~13% of prefill time spent
//! outside MMQ actually goes.
//!
//! Profiling serializes launches via hipEvent sync per kernel, so the
//! reported total is NOT the same as a wall-clock prefill bench.
//! Use it for *relative* attribution, not absolute tok/s.
//!
//! Usage: profile_prefill_qwen35 <model.hfq> [--prefill N] [--warmup-len N]
//!        defaults: prefill=240 (the daemon LRU bench size), warmup-len=8

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
        eprintln!("Usage: profile_prefill_qwen35 <model.hfq> [--prefill N] [--warmup-len N]");
        std::process::exit(1);
    }
    let model_path = &args[1];

    let mut prefill_len: usize = 240;
    let mut warmup_len: usize = 8;
    let mut i = 2;
    while i < args.len() {
        match args[i].as_str() {
            "--prefill"    => { prefill_len = args[i + 1].parse().unwrap(); i += 2; }
            "--warmup-len" => { warmup_len  = args[i + 1].parse().unwrap(); i += 2; }
            other => { eprintln!("unknown arg: {other}"); std::process::exit(1); }
        }
    }

    eprintln!("=== profile_prefill_qwen35 ===");
    eprintln!("Model: {model_path}");
    eprintln!("Prefill: {prefill_len}  Warmup: {warmup_len} (untimed pre-prefill to warm caches)");

    let mut hfq = HfqFile::open(Path::new(model_path)).expect("open model");
    let config = qwen35::config_from_hfq(&hfq).expect("read config");
    eprintln!("Config: dim={} layers={} heads={} kv_heads={}",
        config.dim, config.n_layers, config.n_heads, config.n_kv_heads);

    let mut gpu = rdna_compute::Gpu::init().expect("gpu init");
    eprintln!("GPU: {}", gpu.arch);

    let t_load = Instant::now();
    let weights = qwen35::load_weights(&mut hfq, &config, &mut gpu).expect("load weights");
    eprintln!("Weights loaded in {:.2}s", t_load.elapsed().as_secs_f64());

    let kv_seq = (prefill_len + warmup_len + 16).max(512);
    let mut kv_cache = KvCache::new_gpu_q8(
        &mut gpu, config.n_layers, config.n_kv_heads, config.head_dim, kv_seq
    ).unwrap();
    let mut dn_state = DeltaNetState::new(&mut gpu, &config).unwrap();
    let scratch = Qwen35Scratch::new(&mut gpu, &config, prefill_len.max(128)).unwrap();

    // Warmup: do a small prefill first to amortize one-time init (kernel
    // JIT, allocator warmth). Untimed.
    if warmup_len > 0 {
        let warm_tokens: Vec<u32> = (0..warmup_len as u32).collect();
        eprintln!("\nWarmup prefill of {warmup_len} tokens (untimed)...");
        let t = Instant::now();
        qwen35::forward_prefill_batch(
            &mut gpu, &weights, &config, &warm_tokens, 0,
            &mut kv_cache, &mut dn_state, &scratch,
            None, None, None, None,
        ).expect("warmup prefill failed");
        eprintln!("  warmup: {:.1}ms", t.elapsed().as_secs_f64() * 1000.0);
    }

    // Reset KV / DeltaNet state so the profiled prefill starts from pos=0.
    // We just discard them and rebuild — cheap.
    let mut kv_cache = KvCache::new_gpu_q8(
        &mut gpu, config.n_layers, config.n_kv_heads, config.head_dim, kv_seq
    ).unwrap();
    let mut dn_state = DeltaNetState::new(&mut gpu, &config).unwrap();

    // === PROFILED PREFILL ===
    let prompt_tokens: Vec<u32> = (0..prefill_len as u32).collect();
    eprintln!("\n=== profiled prefill: {prefill_len} tokens ===");
    profile::start();
    let t_profile = Instant::now();
    qwen35::forward_prefill_batch(
        &mut gpu, &weights, &config, &prompt_tokens, 0,
        &mut kv_cache, &mut dn_state, &scratch,
        None, None, None, None,
    ).expect("profiled prefill failed");
    let profile_wall_ms = t_profile.elapsed().as_secs_f64() * 1000.0;
    let entries = profile::stop().unwrap_or_default();
    eprintln!("Captured {} profile entries", entries.len());
    eprintln!("Wall time under profiling: {profile_wall_ms:.1}ms");

    // Aggregate by (category, kernel)
    #[derive(Default)]
    struct Agg {
        calls: usize,
        total_us: f64,
        total_bytes: usize,
    }
    let mut by_kernel: BTreeMap<(&'static str, &'static str), Agg> = BTreeMap::new();
    let mut by_category: BTreeMap<&'static str, Agg> = BTreeMap::new();
    let mut total_us = 0.0f64;
    let mut total_bytes = 0usize;
    for e in &entries {
        let a = by_kernel.entry((e.category, e.kernel)).or_default();
        a.calls += 1;
        a.total_us += e.time_us;
        a.total_bytes += e.bytes;
        let c = by_category.entry(e.category).or_default();
        c.calls += 1;
        c.total_us += e.time_us;
        c.total_bytes += e.bytes;
        total_us += e.time_us;
        total_bytes += e.bytes;
    }

    // ── Per-kernel breakdown (sorted by total time descending) ───────────
    let mut sorted: Vec<_> = by_kernel.into_iter().collect();
    sorted.sort_by(|a, b| b.1.total_us.partial_cmp(&a.1.total_us).unwrap());

    println!();
    println!(
        "{:<4} {:<12} {:<42} {:>6} {:>11} {:>10} {:>12} {:>9}  pct",
        "rnk", "category", "kernel", "calls", "total_us", "avg_us", "total_MiB", "GiB/s"
    );
    println!("{:-<115}", "");
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
            "{:<4} {:<12} {:<42} {:>6} {:>10.1}us {:>9.2}us {:>10.1} MiB {:>8.1}  {:>4.1}%",
            rank + 1, cat, name, a.calls, a.total_us, avg_us, mib, gbps, pct
        );
    }
    println!("{:-<115}", "");

    // ── Per-category roll-up ────────────────────────────────────────────
    println!();
    println!("=== category roll-up ===");
    let mut cat_sorted: Vec<_> = by_category.into_iter().collect();
    cat_sorted.sort_by(|a, b| b.1.total_us.partial_cmp(&a.1.total_us).unwrap());
    println!("{:<14} {:>6} {:>11} {:>12} {:>9}  pct", "category", "calls", "total_us", "total_MiB", "GiB/s");
    println!("{:-<70}", "");
    for (cat, a) in &cat_sorted {
        let mib = a.total_bytes as f64 / (1024.0 * 1024.0);
        let gbps = if a.total_us > 0.0 {
            (a.total_bytes as f64 / (1024.0 * 1024.0 * 1024.0))
                / (a.total_us / 1_000_000.0)
        } else {
            0.0
        };
        let pct = a.total_us * 100.0 / total_us;
        println!("{:<14} {:>6} {:>10.1}us {:>10.1} MiB {:>8.1}  {:>4.1}%",
                 cat, a.calls, a.total_us, mib, gbps, pct);
    }
    println!("{:-<70}", "");
    println!("{:<14} {:>6} {:>10.1}us {:>10.1} MiB {:>8.1}",
             "TOTAL",
             entries.len(),
             total_us,
             total_bytes as f64 / (1024.0 * 1024.0),
             (total_bytes as f64 / (1024.0 * 1024.0 * 1024.0)) / (total_us / 1_000_000.0));
    println!();
    println!("Wall time under profiling: {profile_wall_ms:.1}ms (NOT a real tok/s number; profiling serializes launches)");
    println!("Implied tok/s under profiling: {:.1}",
             prefill_len as f64 / (profile_wall_ms / 1000.0));
}
