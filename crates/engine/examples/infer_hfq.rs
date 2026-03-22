//! Run inference on a .hfq (hipfire-quantized) model.
//! Usage: cargo run --release --example infer_hfq <model.hfq> [--temp T] [--debug] [prompt text...]

use engine::hfq::{self, HfqFile};
use engine::llama::{self, KvCache};
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
        .unwrap_or_else(|| { eprintln!("Usage: infer_hfq <model.hfq> [--temp T] [--debug] [prompt...]"); std::process::exit(1); });

    // Parse flags
    let temp: f32 = args.iter().position(|a| a == "--temp")
        .map(|i| args[i + 1].parse().unwrap_or(0.6))
        .unwrap_or(0.6);
    let debug = args.iter().any(|a| a == "--debug");
    let top_p: f32 = if temp == 0.0 { 1.0 } else { 0.8 };
    let repeat_penalty: f32 = args.iter().position(|a| a == "--repeat-penalty")
        .map(|i| args[i + 1].parse().unwrap_or(1.1)).unwrap_or(1.1);
    let repeat_window: usize = args.iter().position(|a| a == "--repeat-window")
        .map(|i| args[i + 1].parse().unwrap_or(64)).unwrap_or(64);
    let use_q4kv = args.iter().any(|a| a == "--q4kv");
    let use_q8kv = args.iter().any(|a| a == "--q8kv");

    // Collect prompt text (skip flags)
    let mut prompt_parts = Vec::new();
    let mut skip_next = false;
    for (_i, a) in args.iter().enumerate().skip(2) {
        if skip_next { skip_next = false; continue; }
        if a == "--temp" || a == "--debug" || a == "--q4kv" || a == "--q8kv" || a == "--repeat-penalty" || a == "--repeat-window" {
            if a != "--debug" { skip_next = true; } continue;
        }
        prompt_parts.push(a.as_str());
    }
    let prompt_text = if prompt_parts.is_empty() { "Hello".to_string() } else { prompt_parts.join(" ") };

    eprintln!("=== hipfire inference engine (HFQ) ===");
    eprintln!("Model: {model_path}");
    if temp == 0.0 { eprintln!("Sampling: GREEDY (temp=0)"); }
    else { eprintln!("Sampling: temp={temp}, top_p={top_p}"); }

    // Parse HFQ
    let hfq = HfqFile::open(Path::new(model_path)).expect("failed to parse HFQ");
    let config = hfq::config_from_hfq(&hfq).expect("failed to read model config");
    eprintln!("Config: dim={}, layers={}, heads={}, kv_heads={}, vocab={}",
        config.dim, config.n_layers, config.n_heads, config.n_kv_heads, config.vocab_size);

    // Load tokenizer from HFQ metadata (embedded tokenizer.json)
    let tokenizer = engine::tokenizer::Tokenizer::from_hfq_metadata(&hfq.metadata_json)
        .expect("no tokenizer in HFQ file — requantize with tokenizer.json in the model directory");
    eprintln!("Tokenizer: {} tokens (from HFQ)", tokenizer.vocab_size());

    let mut prompt_tokens = tokenizer.encode(&prompt_text);

    // Qwen3 chat template
    if config.arch == llama::ModelArch::Qwen3 {
        let im_start = tokenizer.encode("<|im_start|>");
        let im_end = tokenizer.encode("<|im_end|>");
        let user_tok = tokenizer.encode("user");
        let asst_tok = tokenizer.encode("assistant");
        let nl_tok = tokenizer.encode("\n");
        let sys_tok = tokenizer.encode("system");
        let sys_msg = tokenizer.encode("You are a helpful assistant.");

        let mut chat = Vec::new();
        chat.extend_from_slice(&im_start);
        chat.extend_from_slice(&sys_tok);
        chat.extend_from_slice(&nl_tok);
        chat.extend_from_slice(&sys_msg);
        chat.extend_from_slice(&im_end);
        chat.extend_from_slice(&nl_tok);
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
    let mut kv_cache = if use_q8kv {
        eprintln!("KV cache: Q8 quantized (3.88x compression)");
        KvCache::new_gpu_q8(&mut gpu, config.n_layers, config.n_kv_heads, config.head_dim, kv_seq_len).unwrap()
    } else if use_q4kv {
        eprintln!("KV cache: HFQ4 quantized (3.56x compression)");
        KvCache::new_gpu_q4(&mut gpu, config.n_layers, config.n_kv_heads, config.head_dim, kv_seq_len).unwrap()
    } else {
        KvCache::new_gpu(&mut gpu, config.n_layers, config.n_kv_heads, config.head_dim, kv_seq_len).unwrap()
    };

    // Persistent scratch buffers
    let scratch = llama::ForwardScratch::new(&mut gpu, &config).unwrap();
    let mut rng_state = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH).unwrap().subsec_nanos();
    if rng_state == 0 { rng_state = 1; }

    let mut token_history: Vec<u32> = prompt_tokens.clone();

    // Process prompt: try batched prefill, fall back to sequential
    let t1 = Instant::now();
    let prefill_logits = llama::prefill_forward(
        &mut gpu, &weights, &config, &prompt_tokens, &mut kv_cache,
    );
    let mut next_token = if let Ok(logits) = prefill_logits {
        let prompt_ms = t1.elapsed().as_millis();
        eprintln!("Prompt: {}ms ({} tokens, {:.0} tok/s) [batched]",
            prompt_ms, prompt_tokens.len(),
            prompt_tokens.len() as f64 / (prompt_ms as f64 / 1000.0));
        llama::argmax(&logits)
    } else {
        // Fallback: sequential token-at-a-time prefill
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
    let max_gen = 2048;
    eprintln!("\nGenerating (max {max_gen} tokens)...\n");
    let t2 = Instant::now();
    let mut generated = Vec::new();

    for _ in 0..max_gen {
        generated.push(next_token);
        let text = tokenizer.decode(&[next_token]);
        print!("{text}");
        std::io::stdout().flush().ok();

        if debug && (generated.len() % 50 == 0 || generated.len() <= 5) {
            let pos = prompt_tokens.len() + generated.len() - 1;
            eprintln!("[DEBUG] gen={}, pos={}, token_id={}, text={:?}",
                generated.len(), pos, next_token, text);
        }

        if next_token == config.eos_token || !RUNNING.load(Ordering::Relaxed) {
            break;
        }

        // Upload recent token history for repetition penalty
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
