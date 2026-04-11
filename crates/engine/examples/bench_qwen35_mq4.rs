//! Focused perf benchmark for Qwen3.5 MQ4 forward pass.
//!
//! Separates prefill from generation, strips first-run kernel JIT overhead
//! via an explicit warmup phase, and reports per-token latency stats plus
//! an effective memory bandwidth estimate (weights_bytes × gen_tok/s).
//!
//! Usage: bench_qwen35_mq4 <model.hfq> [--prefill <N>] [--gen <N>] [--warmup <N>]

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
        eprintln!("Usage: bench_qwen35_mq4 <model.hfq> [--prefill N] [--gen N] [--warmup N]");
        std::process::exit(1);
    }
    let model_path = &args[1];

    // Defaults: 32-token prefill, 5-token warmup, 100-token bench.
    let mut prefill_len: usize = 32;
    let mut gen_len: usize = 100;
    let mut warmup_len: usize = 5;
    let mut i = 2;
    while i < args.len() {
        match args[i].as_str() {
            "--prefill" => { prefill_len = args[i + 1].parse().unwrap(); i += 2; }
            "--gen"     => { gen_len     = args[i + 1].parse().unwrap(); i += 2; }
            "--warmup"  => { warmup_len  = args[i + 1].parse().unwrap(); i += 2; }
            other => { eprintln!("unknown arg: {other}"); std::process::exit(1); }
        }
    }

    eprintln!("=== bench_qwen35_mq4 ===");
    eprintln!("Model: {model_path}");
    eprintln!("Phases: prefill={prefill_len} warmup={warmup_len} gen={gen_len}");

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
    //   q8 (default) | turbo4 | turbo4_adaptive | asym (Q8 K + turbo4 V)
    let kv_mode = std::env::var("HIPFIRE_KV_MODE").unwrap_or_else(|_| "q8".to_string());
    eprintln!("KV mode: {kv_mode}");
    let mut kv_cache = match kv_mode.as_str() {
        "q8" => KvCache::new_gpu_q8(
            &mut gpu, config.n_layers, config.n_kv_heads, config.head_dim, kv_seq
        ).unwrap(),
        "turbo4" => KvCache::new_gpu_turbo_adaptive(
            &mut gpu, config.n_layers, config.n_kv_heads, config.head_dim, kv_seq, 4, false
        ).unwrap(),
        "turbo4_adaptive" => KvCache::new_gpu_turbo_adaptive(
            &mut gpu, config.n_layers, config.n_kv_heads, config.head_dim, kv_seq, 4, true
        ).unwrap(),
        "asym" => KvCache::new_gpu_asym_q8k_turbo4v(
            &mut gpu, config.n_layers, config.n_kv_heads, config.head_dim, kv_seq
        ).unwrap(),
        other => panic!("unknown HIPFIRE_KV_MODE: {other}  (use q8|turbo4|turbo4_adaptive|asym)"),
    };
    let mut dn_state = DeltaNetState::new(&mut gpu, &config).unwrap();
    let scratch = Qwen35Scratch::new(&mut gpu, &config, 128).unwrap();

    // Deterministic fake-prompt: token 0, 1, 2, ... prefill_len-1. Keeps the
    // benchmark independent of tokenizer / chat template behaviour.
    let prompt_tokens: Vec<u32> = (0..prefill_len as u32).collect();

    // === PREFILL ===
    // Route through forward_prefill_batch so the bench measures the production
    // prefill path (daemon + greedy_dump both go through it). Inside, this
    // takes the batched LA kernel path for MQ4 models and the FA gather/scatter
    // fallback for FA layers.
    eprintln!("\n=== prefill ({prefill_len} tokens) ===");
    let t_prefill = Instant::now();
    qwen35::forward_prefill_batch(
        &mut gpu, &weights, &config, &prompt_tokens, 0,
        &mut kv_cache, &mut dn_state, &scratch,
    ).expect("prefill forward failed");
    let prefill_ms = t_prefill.elapsed().as_secs_f64() * 1000.0;
    let prefill_tok_s = prefill_len as f64 / (prefill_ms / 1000.0);
    eprintln!("  total: {prefill_ms:.1}ms");
    eprintln!("  tok/s: {prefill_tok_s:.1}");
    eprintln!("  NOTE: first prefill run includes kernel JIT compile cost");

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
