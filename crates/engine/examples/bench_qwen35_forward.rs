//! Deterministic forward-pass benchmark for Qwen3.5. Warms up and measures
//! N forward_scratch calls at a fixed KV position, removing sampling variance.
//!
//! Usage: bench_qwen35_forward <model.hfq> [iters] [--sample]
//!   --sample  also run the GPU top-p sampling kernel + history upload per step

#[cfg(not(feature = "deltanet"))]
fn main() { eprintln!("Build with --features deltanet"); }

#[cfg(feature = "deltanet")]
fn main() {
    use engine::hfq::HfqFile;
    use engine::llama;
    use engine::qwen35::{self, DeltaNetState, Qwen35Scratch};
    use std::path::Path;
    use std::time::Instant;

    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: bench_qwen35_forward <model.hfq> [iters] [--sample]");
        std::process::exit(1);
    }
    let model_path = &args[1];
    let iters: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(100);
    let with_sample = args.iter().any(|a| a == "--sample");

    let mut gpu = rdna_compute::Gpu::init().expect("GPU init failed");
    eprintln!("Loading {}...", model_path);
    let hfq = HfqFile::open(Path::new(model_path)).expect("open model");
    let config = qwen35::config_from_hfq(&hfq).expect("config");
    let weights = qwen35::load_weights(&hfq, &config, &mut gpu).expect("weights");
    eprintln!("Loaded: {} layers, dim={}", config.n_layers, config.dim);

    let max_seq = 2048;
    let mut kv_cache = llama::KvCache::new_gpu_q8(
        &mut gpu, config.n_layers, config.n_kv_heads, config.head_dim, max_seq,
    ).unwrap();
    let mut dn_state = DeltaNetState::new_with_quant(
        &mut gpu, &config, qwen35::StateQuant::Q8,
    ).unwrap();
    let scratch = Qwen35Scratch::new(&mut gpu, &config, 128).unwrap();
    let sc = llama::SamplingConfig::text_thinking();

    // Warmup: 16 forwards at positions 0..16 (fills some KV).
    let warmup_tok: u32 = 1;
    for pos in 0..16 {
        qwen35::forward_scratch(&mut gpu, &weights, &config, warmup_tok, pos,
            &mut kv_cache, &mut dn_state, &scratch).unwrap();
    }
    gpu.hip.device_synchronize().unwrap();

    // Measure: `iters` forwards at consecutive positions, synchronized.
    // If --sample is set, also runs the sampling pipeline per step so we can
    // quantify its overhead precisely.
    let start = Instant::now();
    let mut rng_state: u32 = 0xDEAD_BEEFu32;
    let mut history: Vec<u32> = Vec::with_capacity(iters + 16);
    for _ in 0..16 { history.push(warmup_tok); }
    for i in 0..iters {
        qwen35::forward_scratch(&mut gpu, &weights, &config, warmup_tok, 16 + i,
            &mut kv_cache, &mut dn_state, &scratch).unwrap();
        if with_sample {
            // Baseline CPU sampling path: download full logits and run top-p on CPU
            let mut logits = gpu.download_f32(&scratch.logits).unwrap();
            llama::apply_repeat_penalty(&mut logits, &history, sc.repeat_window, sc.repeat_penalty);
            let _tok = llama::sample_top_p(&logits, sc.answer_temp, sc.top_p);
            let _ = rng_state;
            history.push(warmup_tok);
        }
    }
    gpu.hip.device_synchronize().unwrap();
    let elapsed = start.elapsed();
    let ms_per_tok = elapsed.as_secs_f64() * 1000.0 / iters as f64;
    let tok_per_s = iters as f64 / elapsed.as_secs_f64();

    let tag = if with_sample { "forwards+sample" } else { "forwards" };
    println!("{iters} {tag}: {:.1}ms total, {ms_per_tok:.2}ms/tok, {tok_per_s:.1} tok/s",
        elapsed.as_millis());
}
