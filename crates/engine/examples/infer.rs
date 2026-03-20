//! Run inference on a GGUF LLaMA/Qwen3 model.
//! Usage: cargo run --release --example infer [model.gguf] [prompt text...]

use engine::gguf::GgufFile;
use engine::llama::{self, LlamaConfig, KvCache};
use engine::tokenizer::Tokenizer;
use std::io::Write;
use std::path::Path;
use std::time::Instant;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let model_path = args
        .get(1)
        .map(|s| s.as_str())
        .unwrap_or("/home/kaden/llama.cpp/models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf");

    // Collect remaining args as prompt text (or use default)
    let prompt_text = if args.len() > 2 {
        args[2..].join(" ")
    } else {
        "Hello".to_string()
    };

    eprintln!("=== hipfire inference engine ===");
    eprintln!("Model: {model_path}");

    // Parse GGUF
    let gguf = GgufFile::open(Path::new(model_path)).expect("failed to parse GGUF");
    let config = LlamaConfig::from_gguf(&gguf).expect("failed to read model config");
    eprintln!("Config: dim={}, layers={}, heads={}, kv_heads={}, vocab={}",
        config.dim, config.n_layers, config.n_heads, config.n_kv_heads, config.vocab_size);

    // Load tokenizer
    let tokenizer = Tokenizer::from_gguf(&gguf).expect("failed to load tokenizer");
    eprintln!("Tokenizer: {} tokens", tokenizer.vocab_size());

    // Tokenize prompt
    let mut prompt_tokens = tokenizer.encode(&prompt_text);

    // For Qwen3: wrap in chat template if architecture is qwen3
    if config.arch == llama::ModelArch::Qwen3 {
        // <|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n
        let im_start = tokenizer.encode("<|im_start|>");
        let im_end = tokenizer.encode("<|im_end|>");
        let user_tok = tokenizer.encode("user");
        let asst_tok = tokenizer.encode("assistant");
        let nl_tok = tokenizer.encode("\n");

        let mut chat_tokens = Vec::new();
        chat_tokens.extend_from_slice(&im_start);
        chat_tokens.extend_from_slice(&user_tok);
        chat_tokens.extend_from_slice(&nl_tok);
        chat_tokens.extend_from_slice(&prompt_tokens);
        chat_tokens.extend_from_slice(&im_end);
        chat_tokens.extend_from_slice(&nl_tok);
        chat_tokens.extend_from_slice(&im_start);
        chat_tokens.extend_from_slice(&asst_tok);
        chat_tokens.extend_from_slice(&nl_tok);
        prompt_tokens = chat_tokens;
    }

    eprintln!("Prompt: \"{}\" → {} tokens", prompt_text, prompt_tokens.len());

    // Init GPU
    let mut gpu = rdna_compute::Gpu::init().expect("GPU init failed");

    // Load weights
    eprintln!("Loading weights...");
    let t0 = Instant::now();
    let weights = llama::load_weights(&gguf, &config, &gpu).expect("failed to load weights");
    eprintln!("  Loaded in {:.1}s", t0.elapsed().as_secs_f64());

    // KV cache
    // Cap KV cache at 2048 positions to save VRAM (enough for inference)
    let kv_seq_len = config.max_seq_len.min(2048);
    let mut kv_cache = KvCache::new_gpu(
        &gpu, config.n_layers, config.n_kv_heads, config.head_dim, kv_seq_len,
    ).unwrap();

    // Process prompt
    let t1 = Instant::now();
    let mut logits = Vec::new();
    for (pos, &token) in prompt_tokens.iter().enumerate() {
        logits = llama::forward(&mut gpu, &weights, &config, token, pos, &mut kv_cache)
            .expect("forward pass failed");
    }
    let prompt_ms = t1.elapsed().as_millis();
    eprintln!("Prompt: {}ms ({} tokens, {:.0} tok/s)",
        prompt_ms, prompt_tokens.len(),
        prompt_tokens.len() as f64 / (prompt_ms as f64 / 1000.0));

    // Generate
    let max_gen = 128;
    eprintln!("\nGenerating (max {max_gen} tokens)...\n");
    let t2 = Instant::now();
    let mut next_token = llama::argmax(&logits);
    let mut generated = Vec::new();

    for _ in 0..max_gen {
        generated.push(next_token);

        // Decode and print token immediately (streaming)
        let text = tokenizer.decode(&[next_token]);
        print!("{text}");
        std::io::stdout().flush().ok();

        if next_token == config.eos_token {
            break;
        }

        let pos = prompt_tokens.len() + generated.len() - 1;
        logits = llama::forward(&mut gpu, &weights, &config, next_token, pos, &mut kv_cache)
            .expect("forward pass failed");
        next_token = llama::argmax(&logits);
    }

    let gen_ms = t2.elapsed().as_millis();
    let tok_s = if gen_ms > 0 {
        generated.len() as f64 / (gen_ms as f64 / 1000.0)
    } else {
        0.0
    };

    eprintln!("\n\n=== Done: {} tokens in {}ms ({:.1} tok/s) ===",
        generated.len(), gen_ms, tok_s);
}
