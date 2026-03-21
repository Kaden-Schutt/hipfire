//! Run inference on a .hfq (hipfire-quantized) model.
//! Usage: cargo run --release --example infer_hfq <model.hfq> [prompt text...]

use engine::hfq::{self, HfqFile};
use engine::llama::{self, KvCache};
use std::io::Write;
use std::path::Path;
use std::time::Instant;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let model_path = args.get(1)
        .unwrap_or_else(|| { eprintln!("Usage: infer_hfq <model.hfq> [prompt...]"); std::process::exit(1); });

    let prompt_text = if args.len() > 2 {
        args[2..].join(" ")
    } else {
        "Hello".to_string()
    };

    eprintln!("=== hipfire inference engine (HFQ) ===");
    eprintln!("Model: {model_path}");

    // Parse HFQ
    let hfq = HfqFile::open(Path::new(model_path)).expect("failed to parse HFQ");
    let config = hfq::config_from_hfq(&hfq).expect("failed to read model config");
    eprintln!("Config: dim={}, layers={}, heads={}, kv_heads={}, vocab={}",
        config.dim, config.n_layers, config.n_heads, config.n_kv_heads, config.vocab_size);

    // Load tokenizer from a matching GGUF (fallback until tokenizer is embedded in HFQ)
    let gguf_path = if config.arch == llama::ModelArch::Qwen3 {
        "/home/kaden/llama.cpp/models/Qwen3-0.6B-Q8_0.gguf"
    } else {
        "/home/kaden/llama.cpp/models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
    };
    let gguf = engine::gguf::GgufFile::open(Path::new(gguf_path)).expect("need GGUF for tokenizer");
    let tokenizer = engine::tokenizer::Tokenizer::from_gguf(&gguf).expect("failed to load tokenizer");
    eprintln!("Tokenizer: {} tokens (from GGUF: {})", tokenizer.vocab_size(),
        std::path::Path::new(gguf_path).file_name().unwrap().to_str().unwrap());

    let mut prompt_tokens = tokenizer.encode(&prompt_text);

    // Qwen3 chat template: <|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n
    if config.arch == llama::ModelArch::Qwen3 {
        let im_start = tokenizer.encode("<|im_start|>");
        let im_end = tokenizer.encode("<|im_end|>");
        let user_tok = tokenizer.encode("user");
        let asst_tok = tokenizer.encode("assistant");
        let nl_tok = tokenizer.encode("\n");

        let sys_tok = tokenizer.encode("system");
        let sys_msg = tokenizer.encode("You are a helpful assistant.");

        let mut chat = Vec::new();
        // system message
        chat.extend_from_slice(&im_start);
        chat.extend_from_slice(&sys_tok);
        chat.extend_from_slice(&nl_tok);
        chat.extend_from_slice(&sys_msg);
        chat.extend_from_slice(&im_end);
        chat.extend_from_slice(&nl_tok);
        // user message
        chat.extend_from_slice(&im_start);
        chat.extend_from_slice(&user_tok);
        chat.extend_from_slice(&nl_tok);
        chat.extend_from_slice(&prompt_tokens);
        chat.extend_from_slice(&im_end);
        chat.extend_from_slice(&nl_tok);
        chat.extend_from_slice(&im_start);
        chat.extend_from_slice(&asst_tok);
        chat.extend_from_slice(&nl_tok);
        prompt_tokens = chat;
    }

    eprintln!("Prompt: \"{}\" → {} tokens", prompt_text, prompt_tokens.len());

    // Init GPU
    let mut gpu = rdna_compute::Gpu::init().expect("GPU init failed");

    // Load weights from HFQ
    eprintln!("Loading weights...");
    let t0 = Instant::now();
    let weights = hfq::load_weights_hfq(&hfq, &config, &mut gpu).expect("failed to load weights");
    eprintln!("  Loaded in {:.1}s", t0.elapsed().as_secs_f64());

    // KV cache
    let kv_seq_len = config.max_seq_len.min(2048);
    let mut kv_cache = KvCache::new_gpu(
        &mut gpu, config.n_layers, config.n_kv_heads, config.head_dim, kv_seq_len,
    ).unwrap();

    // Persistent scratch buffers — zero allocations in the hot loop
    let scratch = llama::ForwardScratch::new(&mut gpu, &config).unwrap();
    let mut rng_state = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH).unwrap().subsec_nanos();
    if rng_state == 0 { rng_state = 1; }

    // Process prompt
    let t1 = Instant::now();
    for (pos, &token) in prompt_tokens.iter().enumerate() {
        let (_, rng) = llama::forward_scratch(
            &mut gpu, &weights, &config, token, pos, &mut kv_cache,
            &scratch, 0.7, 0.8, rng_state,
        ).expect("forward_scratch failed");
        rng_state = rng;
    }
    let prompt_ms = t1.elapsed().as_millis();
    eprintln!("Prompt: {}ms ({} tokens, {:.0} tok/s)",
        prompt_ms, prompt_tokens.len(),
        prompt_tokens.len() as f64 / (prompt_ms as f64 / 1000.0));

    // Get first token from last prompt sample
    let mut out_bytes = [0u8; 8];
    gpu.hip.memcpy_dtoh(&mut out_bytes, &scratch.sample_buf.buf).unwrap();
    let mut next_token = u32::from_ne_bytes([out_bytes[0], out_bytes[1], out_bytes[2], out_bytes[3]]);

    // Generate
    let max_gen = 2048;
    eprintln!("\nGenerating (max {max_gen} tokens)...\n");
    let t2 = Instant::now();
    let mut generated = Vec::new();

    for _ in 0..max_gen {
        generated.push(next_token);
        let text = tokenizer.decode(&[next_token]);
        print!("{text}");
        std::io::stdout().flush().ok();

        if next_token == config.eos_token {
            break;
        }

        let pos = prompt_tokens.len() + generated.len() - 1;
        let (tok, rng) = llama::forward_scratch(
            &mut gpu, &weights, &config, next_token, pos, &mut kv_cache,
            &scratch, 0.7, 0.8, rng_state,
        ).expect("forward_scratch failed");
        next_token = tok;
        rng_state = rng;
    }

    let gen_ms = t2.elapsed().as_millis();
    let tok_s = if gen_ms > 0 {
        generated.len() as f64 / (gen_ms as f64 / 1000.0)
    } else { 0.0 };

    eprintln!("\n\n=== Done: {} tokens in {}ms ({:.1} tok/s) ===",
        generated.len(), gen_ms, tok_s);
}
