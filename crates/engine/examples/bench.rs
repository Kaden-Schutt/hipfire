//! Benchmark: profile where time is spent in the forward pass.

use engine::gguf::GgufFile;
use engine::llama::{self, LlamaConfig, KvCache};
use std::path::Path;
use std::time::Instant;

fn main() {
    let model_path = std::env::args().nth(1).unwrap_or_else(|| {
        "/home/kaden/llama.cpp/models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf".to_string()
    });

    eprintln!("=== hipfire baseline benchmark ===");
    let gguf = GgufFile::open(Path::new(&model_path)).unwrap();
    let config = LlamaConfig::from_gguf(&gguf).unwrap();
    eprintln!("Model: {} (dim={}, layers={}, heads={}, kv_heads={}, vocab={})",
        config.arch as u8, config.dim, config.n_layers, config.n_heads, config.n_kv_heads, config.vocab_size);

    let mut gpu = rdna_compute::Gpu::init().unwrap();
    eprintln!("Loading weights...");
    let weights = llama::load_weights(&gguf, &config, &gpu).unwrap();

    let kv_dim = config.n_kv_heads * config.head_dim;
    let mut kv_cache = KvCache::new_gpu(&gpu, config.n_layers, config.n_kv_heads, config.head_dim, config.max_seq_len).unwrap();

    // Warmup: 2 tokens
    let _ = llama::forward(&mut gpu, &weights, &config, config.bos_token, 0, &mut kv_cache);
    let _ = llama::forward(&mut gpu, &weights, &config, 15043, 1, &mut kv_cache);

    // Benchmark: generate 20 tokens, measure each
    let n_tokens = 20;
    let mut times_ms = Vec::new();
    let mut next_token = 15043u32;

    let t_total = Instant::now();
    for i in 0..n_tokens {
        let pos = 2 + i;
        let t = Instant::now();
        let logits = llama::forward(&mut gpu, &weights, &config, next_token, pos, &mut kv_cache).unwrap();
        let elapsed = t.elapsed().as_secs_f64() * 1000.0;
        times_ms.push(elapsed);
        next_token = llama::argmax(&logits);
    }
    let total_ms = t_total.elapsed().as_secs_f64() * 1000.0;

    let avg_ms = times_ms.iter().sum::<f64>() / times_ms.len() as f64;
    let min_ms = times_ms.iter().cloned().fold(f64::INFINITY, f64::min);
    let max_ms = times_ms.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let tok_s = n_tokens as f64 / (total_ms / 1000.0);

    eprintln!("\n=== Baseline Results ===");
    eprintln!("Tokens: {n_tokens}");
    eprintln!("Total: {total_ms:.1}ms");
    eprintln!("Per token: avg={avg_ms:.1}ms min={min_ms:.1}ms max={max_ms:.1}ms");
    eprintln!("Throughput: {tok_s:.1} tok/s");

    // Print per-token times for variance analysis
    eprintln!("\nPer-token times (ms):");
    for (i, &t) in times_ms.iter().enumerate() {
        eprint!("  pos={}: {t:.1}", 2 + i);
        if (i + 1) % 5 == 0 { eprintln!(); }
    }
    eprintln!();

    // VRAM usage
    let vram_out = std::process::Command::new("/opt/rocm/bin/rocm-smi")
        .args(["--showmemuse"])
        .output();
    if let Ok(out) = vram_out {
        let s = String::from_utf8_lossy(&out.stdout);
        for line in s.lines() {
            if line.contains("Used") || line.contains("VRAM") || line.contains("GTT") {
                eprintln!("  {}", line.trim());
            }
        }
    }

    // Output TSV line for results.tsv
    println!("0\tbaseline\tall\t{tok_s:.1}\t-\t-\tPASS\t-\tbaseline: TinyLlama 1.1B F32 dequant, CPU attention\tYES");
}
