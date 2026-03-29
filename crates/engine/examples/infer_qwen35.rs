//! Run inference on Qwen3.5 via DeltaNet.
//! Usage: cargo run --release --features deltanet --example infer_qwen35 -- <model.hfq> [prompt]

#[cfg(not(feature = "deltanet"))]
fn main() { eprintln!("Build with --features deltanet"); }

#[cfg(feature = "deltanet")]
fn main() {
    use engine::hfq::HfqFile;
    use engine::qwen35;
    use std::io::Write;
    use std::path::Path;
    use std::time::Instant;

    let args: Vec<String> = std::env::args().collect();
    let model_path = args.get(1).expect("Usage: infer_qwen35 <model.hfq> [prompt]");
    let prompt_text = if args.len() > 2 { args[2..].join(" ") } else { "Hello".to_string() };

    let hfq = HfqFile::open(Path::new(model_path)).expect("failed to open HFQ");
    let config = qwen35::config_from_hfq(&hfq).expect("failed to parse config");
    eprintln!("Model: {model_path}");
    eprintln!("Config: dim={}, layers={} ({} DeltaNet + {} FullAttn), heads={}, vocab={}",
        config.dim, config.n_layers,
        config.layer_types.iter().filter(|t| **t == qwen35::LayerType::LinearAttention).count(),
        config.layer_types.iter().filter(|t| **t == qwen35::LayerType::FullAttention).count(),
        config.n_heads, config.vocab_size);

    let tokenizer = engine::tokenizer::Tokenizer::from_hfq_metadata(&hfq.metadata_json)
        .expect("no tokenizer in HFQ");
    eprintln!("Tokenizer: {} tokens", tokenizer.vocab_size());

    let mut gpu = rdna_compute::Gpu::init().expect("GPU init failed");

    eprintln!("Loading weights...");
    let t0 = Instant::now();
    let weights = qwen35::load_weights(&hfq, &config, &mut gpu).expect("failed to load weights");
    eprintln!("  Loaded in {:.1}s", t0.elapsed().as_secs_f64());

    // KV cache for full attention layers only (6 layers for 0.8B)
    let n_full_attn = config.layer_types.iter().filter(|t| **t == qwen35::LayerType::FullAttention).count();
    // KV cache needs to be indexed by layer_idx, so allocate for ALL layers but only full attn ones use it
    let kv_seq_len = 2048usize.min(262144); // cap at 2048 for now
    let mut kv_cache = engine::llama::KvCache::new_gpu(
        &mut gpu, config.n_layers, config.n_kv_heads, config.head_dim, kv_seq_len,
    ).unwrap();

    // DeltaNet state (check for --q8-state flag)
    let state_quant = if std::env::var("Q4_STATE").is_ok() {
        qwen35::StateQuant::Q4
    } else if std::env::var("FP32_STATE").is_ok() {
        qwen35::StateQuant::FP32
    } else {
        qwen35::StateQuant::Q8  // default: validated, 4x compression, no quality loss
    };
    let mut dn_state = qwen35::DeltaNetState::new_with_quant(&mut gpu, &config, state_quant).unwrap();
    eprintln!("DeltaNet state: {} S matrices ({:?}), {} conv states",
        dn_state.s_matrices.len(), state_quant, dn_state.conv_states.len());

    // Tokenize with ChatML (skip if NO_CHATML env var set)
    let mut prompt_tokens = tokenizer.encode(&prompt_text);
    let has_chatml = tokenizer.encode("<|im_start|>").len() == 1 && std::env::var("NO_CHATML").is_err();
    if has_chatml {
        let im_start = tokenizer.encode("<|im_start|>");
        let im_end = tokenizer.encode("<|im_end|>");
        let nl = tokenizer.encode("\n");
        let sys = tokenizer.encode("system");
        let sys_msg = tokenizer.encode("You are a helpful assistant.");
        let user = tokenizer.encode("user");
        let asst = tokenizer.encode("assistant");
        let mut chat = Vec::new();
        chat.extend_from_slice(&im_start); chat.extend_from_slice(&sys); chat.extend_from_slice(&nl);
        chat.extend_from_slice(&sys_msg); chat.extend_from_slice(&im_end); chat.extend_from_slice(&nl);
        chat.extend_from_slice(&im_start); chat.extend_from_slice(&user); chat.extend_from_slice(&nl);
        chat.extend_from_slice(&prompt_tokens); chat.extend_from_slice(&im_end); chat.extend_from_slice(&nl);
        chat.extend_from_slice(&im_start); chat.extend_from_slice(&asst); chat.extend_from_slice(&nl);
        // Qwen3.5 thinking mode: add <think> tag to trigger reasoning
        let think_tag = tokenizer.encode("<think>");
        chat.extend_from_slice(&think_tag); chat.extend_from_slice(&nl);
        prompt_tokens = chat;
    }
    eprintln!("Prompt: \"{}\" → {} tokens: {:?}", prompt_text, prompt_tokens.len(), &prompt_tokens[..prompt_tokens.len().min(10)]);

    // Process prompt
    let t1 = Instant::now();
    let mut logits = Vec::new();
    for (pos, &token) in prompt_tokens.iter().enumerate() {
        logits = qwen35::forward(&mut gpu, &weights, &config, token, pos, &mut kv_cache, &mut dn_state)
            .expect("forward failed");
        if pos % 5 == 0 { eprint!("."); }
    }
    let prompt_ms = t1.elapsed().as_millis();
    eprintln!("\nPrompt: {}ms ({} tokens, {:.0} tok/s)",
        prompt_ms, prompt_tokens.len(),
        prompt_tokens.len() as f64 / (prompt_ms as f64 / 1000.0));

    // Debug: check logit distribution
    let top5: Vec<(usize, f32)> = {
        let mut indexed: Vec<(usize, f32)> = logits.iter().enumerate().map(|(i, &v)| (i, v)).collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        indexed.into_iter().take(5).collect()
    };
    eprintln!("Top 5 logits after prompt:");
    for (id, val) in &top5 {
        let text = tokenizer.decode(&[*id as u32]);
        eprintln!("  id={id} logit={val:.4} text={text:?}");
    }
    // Check specific tokens
    eprintln!("logit[248068] (<think>): {:.4}", logits.get(248068).unwrap_or(&f32::NAN));
    eprintln!("logit[248069] (</think>): {:.4}", logits.get(248069).unwrap_or(&f32::NAN));
    eprintln!("logit[248045] (<|im_start|>): {:.4}", logits.get(248045).unwrap_or(&f32::NAN));
    let has_nan = logits.iter().any(|v| v.is_nan());
    let has_inf = logits.iter().any(|v| v.is_infinite());
    eprintln!("NaN: {has_nan}, Inf: {has_inf}, min: {:.4}, max: {:.4}",
        logits.iter().cloned().fold(f32::INFINITY, f32::min),
        logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max));

    // Generate
    let max_gen = 128;
    let t2 = Instant::now();
    let mut next_token = engine::llama::argmax(&logits);
    let mut generated = Vec::new();

    for gi in 0..max_gen {
        generated.push(next_token);
        let text = tokenizer.decode(&[next_token]);
        print!("{text}");
        std::io::stdout().flush().ok();
        if gi < 5 { eprintln!("[gen {gi}: id={next_token} text={text:?}]"); }

        if next_token == config.eos_token { break; }

        let pos = prompt_tokens.len() + generated.len() - 1;
        logits = qwen35::forward(&mut gpu, &weights, &config, next_token, pos, &mut kv_cache, &mut dn_state)
            .expect("forward failed");
        next_token = engine::llama::argmax(&logits);
    }

    let gen_ms = t2.elapsed().as_millis();
    let tok_s = if gen_ms > 0 { generated.len() as f64 / (gen_ms as f64 / 1000.0) } else { 0.0 };
    eprintln!("\n\n=== Done: {} tokens in {}ms ({:.1} tok/s) ===", generated.len(), gen_ms, tok_s);
}
