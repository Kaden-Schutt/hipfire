//! Run inference on a .hfq (hipfire-quantized) model.
//! Usage: cargo run --release --example infer_hfq <model.hfq> [flags] [prompt text...]
//! Flags: --q8kv, --fp32kv, --givens4, --givens2, --hfq4kv, --temp T

use engine::hfq::{self, HfqFile};
use engine::llama::{self, KvCache, ForwardScratch};
use std::io::Write;
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;

static RUNNING: AtomicBool = AtomicBool::new(true);
extern "C" fn handle_sigint(_: libc::c_int) { RUNNING.store(false, Ordering::SeqCst); }

fn main() {
    unsafe { libc::signal(libc::SIGINT, handle_sigint as libc::sighandler_t); }
    let args: Vec<String> = std::env::args().collect();
    let model_path = args.get(1)
        .unwrap_or_else(|| { eprintln!("Usage: infer_hfq <model.hfq> [--q8kv|--fp32kv|--givens4|--givens2|--hfq4kv] [--temp T] [prompt...]"); std::process::exit(1); });

    // Parse flags
    let use_givens4 = args.iter().any(|a| a == "--givens4");
    let use_givens2 = args.iter().any(|a| a == "--givens2");
    let use_q8kv = args.iter().any(|a| a == "--q8kv");
    let use_fp32kv = args.iter().any(|a| a == "--fp32kv");
    let use_hfq4kv = args.iter().any(|a| a == "--hfq4kv");
    let temp: f32 = args.iter().position(|a| a == "--temp")
        .map(|i| args[i + 1].parse().unwrap_or(0.6))
        .unwrap_or(0.0);
    let top_p: f32 = if temp == 0.0 { 1.0 } else { 0.8 };

    let prompt_text = {
        let skip_flags = ["--q8kv", "--fp32kv", "--hfq4kv", "--givens4", "--givens2", "--temp", "--maxgen"];
        let mut skip_next = false;
        let parts: Vec<&str> = args[2..].iter().filter(|a| {
            if skip_next { skip_next = false; return false; }
            if skip_flags.contains(&a.as_str()) { skip_next = a.as_str() == "--temp" || a.as_str() == "--maxgen"; return false; }
            true
        }).map(|s| s.as_str()).collect();
        if parts.is_empty() { "Hello".to_string() } else { parts.join(" ") }
    };

    eprintln!("=== hipfire inference engine (HFQ) ===");
    eprintln!("Model: {model_path}");
    if temp == 0.0 { eprintln!("Sampling: GREEDY"); }
    else { eprintln!("Sampling: temp={temp}, top_p={top_p}"); }

    // Parse HFQ
    let hfq = HfqFile::open(Path::new(model_path)).expect("failed to parse HFQ");
    let config = hfq::config_from_hfq(&hfq).expect("failed to read model config");
    eprintln!("Config: dim={}, layers={}, heads={}, kv_heads={}, vocab={}",
        config.dim, config.n_layers, config.n_heads, config.n_kv_heads, config.vocab_size);

    // Load tokenizer from HFQ metadata, fallback to GGUF
    let tokenizer: engine::tokenizer::Tokenizer = if let Some(t) = engine::tokenizer::Tokenizer::from_hfq_metadata(&hfq.metadata_json) {
        eprintln!("Tokenizer: {} tokens (from HFQ)", t.vocab_size());
        t
    } else {
        let gguf_path = if config.arch == llama::ModelArch::Qwen3 {
            "/home/kaden/llama.cpp/models/Qwen3-0.6B-Q8_0.gguf"
        } else {
            "/home/kaden/llama.cpp/models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
        };
        let gguf = engine::gguf::GgufFile::open(Path::new(gguf_path)).expect("need GGUF for tokenizer");
        let t = engine::tokenizer::Tokenizer::from_gguf(&gguf).expect("failed to load tokenizer");
        eprintln!("Tokenizer: {} tokens (from GGUF)", t.vocab_size());
        t
    };

    let mut prompt_tokens = tokenizer.encode(&prompt_text);

    // ChatML: auto-detect
    let has_chatml = tokenizer.encode("<|im_start|>").len() == 1
        && tokenizer.encode("<|im_end|>").len() == 1;
    if has_chatml {
        let im_start = tokenizer.encode("<|im_start|>");
        let im_end = tokenizer.encode("<|im_end|>");
        let user_tok = tokenizer.encode("user");
        let asst_tok = tokenizer.encode("assistant");
        let nl_tok = tokenizer.encode("\n");
        let mut chat = Vec::new();
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

    // Load weights
    eprintln!("Loading weights...");
    let t0 = Instant::now();
    let weights = hfq::load_weights_hfq(&hfq, &config, &mut gpu).expect("failed to load weights");
    eprintln!("  Loaded in {:.1}s", t0.elapsed().as_secs_f64());

    // KV cache
    let kv_seq_len = config.max_seq_len.min(2048);
    let mut kv_cache = if use_givens4 {
        KvCache::new_gpu_asym3(&mut gpu, config.n_layers, config.n_kv_heads, config.head_dim, kv_seq_len).unwrap()
    } else if use_givens2 {
        KvCache::new_gpu_asym2(&mut gpu, config.n_layers, config.n_kv_heads, config.head_dim, kv_seq_len).unwrap()
    } else if use_hfq4kv {
        KvCache::new_gpu_hfq4kv(&mut gpu, config.n_layers, config.n_kv_heads, config.head_dim, kv_seq_len).unwrap()
    } else if use_q8kv {
        KvCache::new_gpu_q8(&mut gpu, config.n_layers, config.n_kv_heads, config.head_dim, kv_seq_len).unwrap()
    } else if use_fp32kv {
        KvCache::new_gpu(&mut gpu, config.n_layers, config.n_kv_heads, config.head_dim, kv_seq_len).unwrap()
    } else {
        // Default: Q8 KV cache
        KvCache::new_gpu_q8(&mut gpu, config.n_layers, config.n_kv_heads, config.head_dim, kv_seq_len).unwrap()
    };

    // Persistent scratch buffers (zero-alloc forward pass)
    let scratch = ForwardScratch::new(&mut gpu, &config).unwrap();
    let mut rng_state = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH).unwrap().subsec_nanos();
    if rng_state == 0 { rng_state = 1; }

    let mut token_history: Vec<u32> = prompt_tokens.clone();
    let repeat_penalty: f32 = 1.1;
    let repeat_window: usize = 64;

    // Prefill: WMMA-fused batched path via llama::forward_prefill_batch when
    // weights + KV qualify; per-token forward_scratch fallback otherwise. The
    // batched path natively supports Q8 + asym{2,3,4}; HFQ4 KV stays sequential.
    let t1 = Instant::now();
    let batched_ok = !use_hfq4kv;
    let mut next_token = if batched_ok {
        llama::forward_prefill_batch(
            &mut gpu, &weights, &config, &prompt_tokens, 0, &mut kv_cache, &scratch, None,
        ).expect("forward_prefill_batch failed");
        let prompt_ms = t1.elapsed().as_millis();
        eprintln!("Prompt: {}ms ({} tokens, {:.0} tok/s) [batched]",
            prompt_ms, prompt_tokens.len(),
            prompt_tokens.len() as f64 / (prompt_ms as f64 / 1000.0));
        let logits = gpu.download_f32(&scratch.logits).expect("download logits");
        llama::argmax(&logits)
    } else {
        for (pos, &token) in prompt_tokens.iter().enumerate() {
            let (_, rng) = llama::forward_scratch(
                &mut gpu, &weights, &config, token, pos, &mut kv_cache,
                &scratch, temp.max(0.01), top_p, rng_state, 0, 1.0,
            ).expect("forward_scratch failed");
            rng_state = rng;
        }
        let prompt_ms = t1.elapsed().as_millis();
        eprintln!("Prompt: {}ms ({} tokens, {:.0} tok/s) [sequential]",
            prompt_ms, prompt_tokens.len(),
            prompt_tokens.len() as f64 / (prompt_ms as f64 / 1000.0));
        let mut out_bytes = [0u8; 8];
        gpu.hip.memcpy_dtoh(&mut out_bytes, &scratch.sample_buf.buf).unwrap();
        u32::from_ne_bytes([out_bytes[0], out_bytes[1], out_bytes[2], out_bytes[3]])
    };

    // Generate
    let max_gen: usize = args.iter().position(|a| a == "--maxgen")
        .map(|i| args[i + 1].parse().unwrap_or(128)).unwrap_or(128);
    eprintln!("\nGenerating (max {max_gen} tokens)...\n");
    let t2 = Instant::now();
    let mut generated = Vec::new();

    for _ in 0..max_gen {
        generated.push(next_token);
        let text = tokenizer.decode(&[next_token]);
        print!("{text}");
        std::io::stdout().flush().ok();

        if next_token == config.eos_token || !RUNNING.load(Ordering::Relaxed) {
            break;
        }

        token_history.push(next_token);
        let hist_start = token_history.len().saturating_sub(repeat_window);
        let hist_slice = &token_history[hist_start..];
        let hist_bytes: Vec<u8> = hist_slice.iter().flat_map(|t| t.to_ne_bytes()).collect();
        gpu.hip.memcpy_htod(&scratch.repeat_buf.buf, &hist_bytes).unwrap();

        let pos = prompt_tokens.len() + generated.len() - 1;
        let (tok, rng) = llama::forward_scratch(
            &mut gpu, &weights, &config, next_token, pos, &mut kv_cache,
            &scratch, temp.max(0.01), top_p, rng_state,
            hist_slice.len(), repeat_penalty,
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
