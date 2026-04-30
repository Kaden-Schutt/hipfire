//! Focused perf benchmark for Qwen3.5 MQ4 forward pass.
//!
//! Separates prefill from generation, strips first-run kernel JIT overhead
//! via an explicit warmup phase, and reports per-token latency stats plus
//! an effective memory bandwidth estimate (weights_bytes × gen_tok/s).
//!
//! Usage: bench_qwen35_mq4 <model.hfq> [--prefill <N>] [--prefill-runs <N>] [--gen <N>] [--warmup <N>]

#[cfg(not(feature = "deltanet"))]
fn main() { eprintln!("build with --features deltanet"); }

#[cfg(feature = "deltanet")]
fn main() {
    use engine::hfq::HfqFile;
    use engine::qwen35::{self, DeltaNetState, Qwen35Scratch};
    use engine::llama::{self, KvCache};
    use std::path::Path;
    use std::time::Instant;

    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: bench_qwen35_mq4 <model.hfq> [--prefill N] [--prefill-runs N] [--gen N] [--warmup N]");
        std::process::exit(1);
    }
    let model_path = &args[1];

    // Defaults: 32-token prefill, 5-token warmup, 100-token bench.
    let mut prefill_len: usize = 32;
    let mut prefill_runs: usize = 1;
    let mut gen_len: usize = 100;
    let mut warmup_len: usize = 5;
    let mut i = 2;
    while i < args.len() {
        match args[i].as_str() {
            "--prefill" => { prefill_len = args[i + 1].parse().unwrap(); i += 2; }
            "--prefill-runs" => { prefill_runs = args[i + 1].parse::<usize>().unwrap().max(1); i += 2; }
            "--gen"     => { gen_len     = args[i + 1].parse().unwrap(); i += 2; }
            "--warmup"  => { warmup_len  = args[i + 1].parse().unwrap(); i += 2; }
            other => { eprintln!("unknown arg: {other}"); std::process::exit(1); }
        }
    }

    eprintln!("=== bench_qwen35_mq4 ===");
    eprintln!("Model: {model_path}");
    eprintln!("Phases: prefill={prefill_len} prefill_runs={prefill_runs} warmup={warmup_len} gen={gen_len}");

    let hfq = HfqFile::open(Path::new(model_path)).expect("open model");
    let config = qwen35::config_from_hfq(&hfq).expect("read config");
    eprintln!(
        "Config: dim={} layers={} heads={} kv_heads={} vocab={}",
        config.dim, config.n_layers, config.n_heads, config.n_kv_heads, config.vocab_size
    );
    let model_bytes = std::fs::metadata(model_path).map(|m| m.len()).unwrap_or(0);
    eprintln!("Model size: {:.3} GiB ({} bytes)", model_bytes as f64 / (1024.0 * 1024.0 * 1024.0), model_bytes);

    let mut gpu = rdna_compute::Gpu::init().expect("gpu init");
    eprintln!("GPU: {}", gpu.arch);

    let t_load = Instant::now();
    let weights = qwen35::load_weights(&hfq, &config, &mut gpu).expect("load weights");
    eprintln!("Weights loaded in {:.2}s", t_load.elapsed().as_secs_f64());

    let kv_seq = (prefill_len + warmup_len + gen_len + 16).max(512);
    // KV cache mode via HIPFIRE_KV_MODE env var:
    //   q8 (default) | asym4 | asym3 | asym2
    let kv_mode = std::env::var("HIPFIRE_KV_MODE").unwrap_or_else(|_| "q8".to_string());
    eprintln!("KV mode: {kv_mode}");
    let mut kv_cache = match kv_mode.as_str() {
        "q8" => KvCache::new_gpu_q8(
            &mut gpu, config.n_layers, config.n_kv_heads, config.head_dim, kv_seq
        ).unwrap(),
        "asym4" | "turbo4" => KvCache::new_gpu_asym4(
            &mut gpu, config.n_layers, config.n_kv_heads, config.head_dim, kv_seq
        ).unwrap(),
        "asym3" | "turbo3" | "turbo" => KvCache::new_gpu_asym3(
            &mut gpu, config.n_layers, config.n_kv_heads, config.head_dim, kv_seq
        ).unwrap(),
        "asym2" | "turbo2" => KvCache::new_gpu_asym2(
            &mut gpu, config.n_layers, config.n_kv_heads, config.head_dim, kv_seq
        ).unwrap(),
        other => panic!("unknown HIPFIRE_KV_MODE: {other}  (use q8|asym4|asym3|asym2)"),
    };
    let mut dn_state = DeltaNetState::new(&mut gpu, &config).unwrap();
    let scratch = Qwen35Scratch::new_with_kv_max(&mut gpu, &config, 128, kv_seq).unwrap();

    // Deterministic fake-prompt: token 0, 1, 2, ... prefill_len-1. Keeps the
    // benchmark independent of tokenizer / chat template behaviour.
    let prompt_tokens: Vec<u32> = (0..prefill_len as u32).collect();

    // DPM warmup BEFORE prefill (issue #65). The default `HIPFIRE_DPM_WARMUP_SECS`
    // hook fires AFTER the warmup tokens, before the timed gen — useless for
    // prefill measurement when the GPU is in DPM step 0/1 from idle. This
    // mirrors that hook for the prefill phase: stabilizes clocks before the
    // timed forward_prefill_batch.
    if let Ok(secs_str) = std::env::var("HIPFIRE_DPM_WARMUP_SECS") {
        let secs: f32 = secs_str.parse().unwrap_or(0.0);
        if secs > 0.0 {
            eprintln!("\n=== DPM warmup ({secs:.1}s, pre-prefill) ===");
            gpu.dpm_warmup(secs).expect("dpm warmup");
        }
    }

    // === PREFILL ===
    // Route through forward_prefill_batch so the bench measures the production
    // prefill path (daemon + greedy_dump both go through it). Inside, this
    // takes the batched LA kernel path for MQ4 models and the FA gather/scatter
    // fallback for FA layers.
    let do_profile = std::env::var("HIPFIRE_PROFILE").ok().as_deref() == Some("1");
    // When profiling, do an unprofile warm-up prefill first to JIT all kernels,
    // then reset state and profile the second pass.
    if do_profile {
        eprintln!("\n=== warm-up prefill (JIT kernels) ===");
        qwen35::forward_prefill_batch(
            &mut gpu, &weights, &config, &prompt_tokens, 0,
            &mut kv_cache, &mut dn_state, &scratch,
            None, None, None, None,
        ).expect("warmup prefill failed");
        // Reset DeltaNet state for the profiled run
        dn_state = DeltaNetState::new(&mut gpu, &config).unwrap();
        kv_cache = match kv_mode.as_str() {
            "q8" => KvCache::new_gpu_q8(&mut gpu, config.n_layers, config.n_kv_heads, config.head_dim, kv_seq).unwrap(),
            "asym4" | "turbo4" => KvCache::new_gpu_asym4(&mut gpu, config.n_layers, config.n_kv_heads, config.head_dim, kv_seq).unwrap(),
            "asym3" | "turbo3" | "turbo" => KvCache::new_gpu_asym3(&mut gpu, config.n_layers, config.n_kv_heads, config.head_dim, kv_seq).unwrap(),
            "asym2" | "turbo2" => KvCache::new_gpu_asym2(&mut gpu, config.n_layers, config.n_kv_heads, config.head_dim, kv_seq).unwrap(),
            _ => KvCache::new_gpu_q8(&mut gpu, config.n_layers, config.n_kv_heads, config.head_dim, kv_seq).unwrap(),
        };
        eprintln!("  JIT complete, profiling next pass...");
        rdna_compute::profile::start();
    }
    eprintln!("\n=== prefill ({prefill_len} tokens) ===");
    let mut prefill_samples_ms = Vec::with_capacity(prefill_runs);
    for run in 0..prefill_runs {
        if run > 0 {
            dn_state = DeltaNetState::new(&mut gpu, &config).unwrap();
            kv_cache = match kv_mode.as_str() {
                "q8" => KvCache::new_gpu_q8(&mut gpu, config.n_layers, config.n_kv_heads, config.head_dim, kv_seq).unwrap(),
                "asym4" | "turbo4" => KvCache::new_gpu_asym4(&mut gpu, config.n_layers, config.n_kv_heads, config.head_dim, kv_seq).unwrap(),
                "asym3" | "turbo3" | "turbo" => KvCache::new_gpu_asym3(&mut gpu, config.n_layers, config.n_kv_heads, config.head_dim, kv_seq).unwrap(),
                "asym2" | "turbo2" => KvCache::new_gpu_asym2(&mut gpu, config.n_layers, config.n_kv_heads, config.head_dim, kv_seq).unwrap(),
                _ => KvCache::new_gpu_q8(&mut gpu, config.n_layers, config.n_kv_heads, config.head_dim, kv_seq).unwrap(),
            };
        }
        let t_prefill = Instant::now();
        qwen35::forward_prefill_batch(
            &mut gpu, &weights, &config, &prompt_tokens, 0,
            &mut kv_cache, &mut dn_state, &scratch,
            None, None, None, None,
        ).expect("prefill forward failed");
        gpu.hip.device_synchronize().expect("sync after prefill");
        let ms = t_prefill.elapsed().as_secs_f64() * 1000.0;
        prefill_samples_ms.push(ms);
        if prefill_runs > 1 {
            eprintln!("  run {:>2}: {:.1}ms  {:.1} tok/s", run + 1, ms, prefill_len as f64 / (ms / 1000.0));
        }
    }
    let prefill_ms = *prefill_samples_ms.last().unwrap();
    if do_profile {
        if let Some(entries) = rdna_compute::profile::stop() {
            let mut by_kernel: std::collections::HashMap<&str, (f64, usize, usize)> = Default::default();
            for e in &entries {
                let (time, count, bytes) = by_kernel.entry(e.kernel).or_default();
                *time += e.time_us;
                *count += 1;
                *bytes += e.bytes;
            }
            eprintln!("\n=== PROFILE ({} launches, {:.1}ms wall) ===", entries.len(), prefill_ms);
            let mut kerns: Vec<_> = by_kernel.iter().collect();
            kerns.sort_by(|a, b| b.1.0.partial_cmp(&a.1.0).unwrap());
            let total_us: f64 = kerns.iter().map(|(_, (t, _, _))| t).sum();
            for (kern, (us, n, bytes)) in &kerns {
                let gib_s = if *us > 0.0 {
                    (*bytes as f64 / (1024.0 * 1024.0 * 1024.0)) / (*us / 1_000_000.0)
                } else {
                    0.0
                };
                eprintln!("  {kern:45} {n:5}x  {:.1}ms  ({:.0}µs/call)  {:.1}%  {:.1} GiB/s",
                    us / 1000.0, us / *n as f64, us / total_us * 100.0, gib_s);
            }
            eprintln!("  {:45} {:5}   {:.1}ms", "TOTAL (serialized)", "", total_us / 1000.0);
        }
    }
    let prefill_tok_s = prefill_len as f64 / (prefill_ms / 1000.0);
    if prefill_samples_ms.len() > 1 {
        let mut sorted_prefill = prefill_samples_ms.clone();
        sorted_prefill.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let median_ms = sorted_prefill[sorted_prefill.len() / 2];
        eprintln!("  median: {median_ms:.1}ms  {:.1} tok/s", prefill_len as f64 / (median_ms / 1000.0));
    }
    eprintln!("  total: {prefill_ms:.1}ms");
    eprintln!("  tok/s: {prefill_tok_s:.1}");
    eprintln!("  NOTE: first prefill run includes kernel JIT compile cost");

    // (deferred-conversion mode removed with givens — asym modes are natively
    //  batched so there's no prefill/decode cache swap to measure.)

    // Read logits to get a valid next token
    let logits = gpu.download_f32(&scratch.logits).unwrap();
    let mut next_token = llama::argmax(&logits);

    // === WARMUP ===
    eprintln!("\n=== warmup ({warmup_len} tokens — untimed, lets JIT settle) ===");
    let t_warmup = Instant::now();
    for step in 0..warmup_len {
        let pos = prefill_len + step;
        if pos >= kv_seq { break; }
        qwen35::forward_scratch(
            &mut gpu, &weights, &config, next_token, pos,
            &mut kv_cache, &mut dn_state, &scratch,
        ).expect("warmup forward failed");
        let logits = gpu.download_f32(&scratch.logits).unwrap();
        next_token = llama::argmax(&logits);
    }
    let warmup_ms = t_warmup.elapsed().as_secs_f64() * 1000.0;
    eprintln!("  total: {warmup_ms:.1}ms  avg: {:.2}ms/tok", warmup_ms / warmup_len as f64);

    // HIPFIRE_DPM_WARMUP_SECS: optional DPM-stabilization pass before the
    // timed decode. See crates/rdna-compute/src/dispatch.rs `dpm_warmup`.
    if let Ok(secs_str) = std::env::var("HIPFIRE_DPM_WARMUP_SECS") {
        let secs: f32 = secs_str.parse().unwrap_or(0.0);
        if secs > 0.0 {
            gpu.dpm_warmup(secs).expect("dpm warmup");
        }
    }

    // === GEN BENCHMARK ===
    eprintln!("\n=== gen ({gen_len} tokens — timed) ===");
    let mut per_token_ms: Vec<f64> = Vec::with_capacity(gen_len);
    let t_gen_start = Instant::now();
    for step in 0..gen_len {
        let pos = prefill_len + warmup_len + step;
        if pos >= kv_seq { break; }
        let t = Instant::now();
        qwen35::forward_scratch(
            &mut gpu, &weights, &config, next_token, pos,
            &mut kv_cache, &mut dn_state, &scratch,
        ).expect("gen forward failed");
        let logits = gpu.download_f32(&scratch.logits).unwrap();
        let t_ms = t.elapsed().as_secs_f64() * 1000.0;
        per_token_ms.push(t_ms);
        next_token = llama::argmax(&logits);
    }
    let gen_total_ms = t_gen_start.elapsed().as_secs_f64() * 1000.0;

    // Stats
    let mut sorted = per_token_ms.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = sorted.len();
    if n == 0 {
        eprintln!("  total: {gen_total_ms:.1}ms over 0 tokens");
        eprintln!("  tok/s (gen): 0.0");
        eprintln!();
        eprintln!("SUMMARY  gen_tok_s=0.0  bw_gib_s=0.0  prefill_tok_s={prefill_tok_s:.1}  avg_ms=0.00  p50_ms=0.00");
        return;
    }
    let sum: f64 = sorted.iter().sum();
    let avg_ms = sum / n as f64;
    let min_ms = sorted[0];
    let max_ms = sorted[n - 1];
    let p50_ms = sorted[n / 2];
    let p90_ms = sorted[(n * 90) / 100];
    let p99_ms = sorted[(n.saturating_sub(1) * 99) / 100];
    let gen_tok_s = n as f64 / (gen_total_ms / 1000.0);

    // BW estimate: each gen token reads ~all weights (minus KV cache writes,
    // which are separate). Effective BW = model_bytes × tok/s.
    let bw_gbps = (model_bytes as f64 * gen_tok_s) / (1024.0 * 1024.0 * 1024.0);

    eprintln!("  total: {gen_total_ms:.1}ms over {n} tokens");
    eprintln!("  per-token ms:");
    eprintln!("    min={min_ms:.2}  p50={p50_ms:.2}  avg={avg_ms:.2}  p90={p90_ms:.2}  p99={p99_ms:.2}  max={max_ms:.2}");
    eprintln!("  tok/s (gen): {gen_tok_s:.1}");
    eprintln!("  effective BW: {bw_gbps:.1} GiB/s (model {:.2} GiB × {gen_tok_s:.1} tok/s)",
        model_bytes as f64 / (1024.0 * 1024.0 * 1024.0));
    eprintln!();
    eprintln!("SUMMARY  gen_tok_s={gen_tok_s:.1}  bw_gib_s={bw_gbps:.1}  prefill_tok_s={prefill_tok_s:.1}  avg_ms={avg_ms:.2}  p50_ms={p50_ms:.2}");
}
