//! Run inference on a GGUF LLaMA model using the RX 5700 XT.
//! Usage: cargo run --release --example infer [model.gguf] [prompt_token_ids...]

use engine::gguf::GgufFile;
use engine::llama::{self, LlamaConfig, KvCache};
use std::path::Path;
use std::time::Instant;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let model_path = args
        .get(1)
        .map(|s| s.as_str())
        .unwrap_or("/home/kaden/llama.cpp/models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf");

    // Default prompt tokens: "<s>Hello" in LLaMA tokenizer
    // 1 = <s> (BOS), 15043 = Hello
    let prompt_tokens: Vec<u32> = if args.len() > 2 {
        args[2..].iter().map(|s| s.parse().unwrap()).collect()
    } else {
        vec![1, 15043]
    };

    eprintln!("=== rx-rustane inference engine ===");
    eprintln!("Model: {model_path}");
    eprintln!("Prompt tokens: {prompt_tokens:?}");
    eprintln!();

    // Parse GGUF
    eprintln!("Parsing GGUF...");
    let t0 = Instant::now();
    let gguf = GgufFile::open(Path::new(model_path)).expect("failed to parse GGUF");
    let config = LlamaConfig::from_gguf(&gguf).expect("failed to read model config");
    eprintln!("  Config: dim={}, layers={}, heads={}, kv_heads={}, vocab={}",
        config.dim, config.n_layers, config.n_heads, config.n_kv_heads, config.vocab_size);
    eprintln!("  GGUF parsed in {:.1}ms", t0.elapsed().as_millis());

    // Init GPU
    eprintln!("\nInitializing GPU...");
    let t1 = Instant::now();
    let mut gpu = rdna_compute::Gpu::init().expect("GPU init failed");
    eprintln!("  GPU ready in {:.1}ms", t1.elapsed().as_millis());

    // Load weights to GPU (dequantize Q4_K/Q6_K → F32, upload)
    eprintln!("\nLoading weights to GPU (dequantize → F32 → VRAM)...");
    let t2 = Instant::now();
    let weights = llama::load_weights(&gguf, &config, &gpu).expect("failed to load weights");
    eprintln!("  Weights loaded in {:.1}s", t2.elapsed().as_secs_f64());

    // Initialize KV cache
    let kv_dim = config.n_kv_heads * config.head_dim;
    let mut kv_cache = KvCache::new(config.n_layers, kv_dim, config.max_seq_len);

    // Process prompt tokens
    eprintln!("\nProcessing prompt...");
    let t3 = Instant::now();
    let mut logits = Vec::new();
    for (pos, &token) in prompt_tokens.iter().enumerate() {
        logits = llama::forward(&mut gpu, &weights, &config, token, pos, &mut kv_cache)
            .expect("forward pass failed");
        let next = llama::argmax(&logits);
        eprintln!("  pos={pos} token={token} → next={next}");
    }
    let prompt_ms = t3.elapsed().as_millis();
    eprintln!("  Prompt processed in {prompt_ms}ms ({} tokens)", prompt_tokens.len());

    // Generate tokens
    let max_gen = 32;
    eprintln!("\nGenerating {max_gen} tokens...");
    let t4 = Instant::now();
    let mut generated = Vec::new();
    let mut next_token = llama::argmax(&logits);

    for i in 0..max_gen {
        generated.push(next_token);
        let pos = prompt_tokens.len() + i;
        logits = llama::forward(&mut gpu, &weights, &config, next_token, pos, &mut kv_cache)
            .expect("forward pass failed");
        next_token = llama::argmax(&logits);

        // Stop on EOS (token 2 in LLaMA)
        if next_token == 2 {
            break;
        }
    }

    let gen_ms = t4.elapsed().as_millis();
    let tokens_per_sec = if gen_ms > 0 {
        generated.len() as f64 / (gen_ms as f64 / 1000.0)
    } else {
        0.0
    };

    eprintln!("\n=== Results ===");
    eprintln!("Generated {} tokens in {}ms ({:.1} tok/s)", generated.len(), gen_ms, tokens_per_sec);
    eprintln!("Token IDs: {:?}", generated);

    // Note: without a tokenizer, we can't decode to text.
    // But correct token generation proves the engine works.
    eprintln!("\nrx-rustane inference: COMPLETE");
}
