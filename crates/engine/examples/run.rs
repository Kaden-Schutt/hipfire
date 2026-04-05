//! Interactive REPL for hipfire — like `ollama run`.
//! Usage: hipfire-run <model.hfq> [--system "prompt"] [--turbo N]

#[cfg(not(feature = "deltanet"))]
fn main() { eprintln!("Build with --features deltanet"); }

#[cfg(feature = "deltanet")]
fn main() {
    use engine::hfq::HfqFile;
    use engine::qwen35::{self, DeltaNetState, Qwen35Scratch};
    use engine::llama;
    use std::io::Write;
    use std::path::Path;
    use std::time::Instant;

    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: run <model.hfq> [--system \"prompt\"] [--turbo N] [--temp F] [--max-seq N]");
        std::process::exit(1);
    }
    let model_path = &args[1];

    // Parse flags
    let mut system_prompt: Option<String> = None;
    let mut turbo_bits: u8 = 0;
    let mut asym_kv = false;
    let mut boundary: u8 = 0;
    let mut temp: f32 = 0.3;
    let mut max_seq: usize = 4096;
    let mut i = 2;
    while i < args.len() {
        match args[i].as_str() {
            "--system" | "-s" => { i += 1; system_prompt = Some(args[i].clone()); }
            "--turbo" => { i += 1; turbo_bits = args[i].parse().unwrap_or(4); }
            "--asym" => { asym_kv = true; }
            "--boundary" => { i += 1; boundary = args[i].parse().unwrap_or(2); }
            "--temp" => { i += 1; temp = args[i].parse().unwrap_or(0.3); }
            "--max-seq" => { i += 1; max_seq = args[i].parse().unwrap_or(4096); }
            _ => {}
        }
        i += 1;
    }

    // Load model
    let mut gpu = rdna_compute::Gpu::init().expect("GPU init failed");
    eprintln!("Loading {}...", model_path);

    let hfq = HfqFile::open(Path::new(model_path)).expect("failed to open model");
    let config = qwen35::config_from_hfq(&hfq).expect("failed to read config");
    let weights = qwen35::load_weights(&hfq, &config, &mut gpu).expect("failed to load weights");

    let n_kv_layers = config.layer_types.iter().filter(|t| **t == engine::qwen35::LayerType::FullAttention).count();
    let kv_cache = if asym_kv {
        eprintln!("KV cache: asymmetric q8-K + turbo4-V (boundary={})", boundary);
        llama::KvCache::new_gpu_asym_q8k_turbo4v_boundary(&mut gpu, config.n_layers, config.n_kv_heads, config.head_dim, max_seq, boundary, n_kv_layers).unwrap()
    } else if turbo_bits >= 2 && turbo_bits <= 4 {
        eprintln!("KV cache: turbo{}", turbo_bits);
        llama::KvCache::new_gpu_turbo(&mut gpu, config.n_layers, config.n_kv_heads, config.head_dim, max_seq, turbo_bits).unwrap()
    } else {
        llama::KvCache::new_gpu_q8(&mut gpu, config.n_layers, config.n_kv_heads, config.head_dim, max_seq).unwrap()
    };
    let dn_state = DeltaNetState::new(&mut gpu, &config).unwrap();
    let scratch = Qwen35Scratch::new(&mut gpu, &config, 128).unwrap();
    let tokenizer = engine::tokenizer::Tokenizer::from_hfq_metadata(&hfq.metadata_json)
        .expect("failed to load tokenizer");

    eprintln!("Model: {} layers, dim={}, vocab={}", config.n_layers, config.dim, config.vocab_size);
    eprintln!("GPU: {} ({:.1} GB VRAM)", gpu.arch, gpu.hip.get_vram_info().map(|(_, t)| t as f64 / 1e9).unwrap_or(0.0));
    if let Some(ref s) = system_prompt {
        eprintln!("System: {}", if s.len() > 60 { format!("{}...", &s[..60]) } else { s.clone() });
    }
    eprintln!("Type /help for commands. Ctrl+C to quit.\n");

    // ChatML token IDs
    let im_start = tokenizer.encode("<|im_start|>");
    let im_end = tokenizer.encode("<|im_end|>");
    let nl = tokenizer.encode("\n");
    let user_tok = tokenizer.encode("user");
    let asst_tok = tokenizer.encode("assistant");
    let im_end_token = if im_end.len() == 1 { Some(im_end[0]) } else { None };
    let sc = llama::SamplingConfig::text_thinking();

    let mut seq_pos: usize = 0;
    let mut conversation_tokens: Vec<u32> = Vec::new();
    let mut kv_cache = kv_cache;
    let mut dn_state = dn_state;
    let mut total_tokens: usize = 0;

    // REPL
    let stdin = std::io::stdin();
    loop {
        // Prompt
        print!(">>> ");
        std::io::stdout().flush().unwrap();

        let mut input = String::new();
        if stdin.read_line(&mut input).unwrap() == 0 { break; } // EOF
        let input = input.trim();
        if input.is_empty() { continue; }

        // Commands
        match input {
            "/quit" | "/exit" | "/q" => break,
            "/reset" | "/clear" => {
                seq_pos = 0;
                conversation_tokens.clear();
                total_tokens = 0;
                for s in &dn_state.s_matrices { let _ = gpu.hip.memset(&s.buf, 0, s.buf.size()); }
                for s in &dn_state.s_scales { let _ = gpu.hip.memset(&s.buf, 0, s.buf.size()); }
                for s in &dn_state.conv_states { let _ = gpu.hip.memset(&s.buf, 0, s.buf.size()); }
                eprintln!("Conversation reset.\n");
                continue;
            }
            "/help" | "/?" => {
                eprintln!("Commands:");
                eprintln!("  /reset  — clear conversation history");
                eprintln!("  /quit   — exit");
                eprintln!("  /stats  — show token counts and speed");
                eprintln!("  /help   — this message\n");
                continue;
            }
            "/stats" => {
                eprintln!("Position: {}/{} tokens used", seq_pos, max_seq);
                eprintln!("Total generated: {} tokens\n", total_tokens);
                continue;
            }
            _ => {}
        }

        // Capacity guard
        let prompt_est = tokenizer.encode(input).len() + 20;
        if seq_pos + prompt_est + 512 > max_seq {
            eprintln!("[context full — auto-resetting]\n");
            seq_pos = 0;
            conversation_tokens.clear();
            for s in &dn_state.s_matrices { let _ = gpu.hip.memset(&s.buf, 0, s.buf.size()); }
            for s in &dn_state.s_scales { let _ = gpu.hip.memset(&s.buf, 0, s.buf.size()); }
            for s in &dn_state.conv_states { let _ = gpu.hip.memset(&s.buf, 0, s.buf.size()); }
        }

        // Build ChatML tokens for this turn
        let q_tokens = tokenizer.encode(input);
        let mut new_tokens: Vec<u32> = Vec::new();

        // System prompt on first turn
        if seq_pos == 0 {
            if let Some(ref sys) = system_prompt {
                let sys_tok = tokenizer.encode("system");
                let sys_content = tokenizer.encode(sys);
                new_tokens.extend_from_slice(&im_start);
                new_tokens.extend_from_slice(&sys_tok);
                new_tokens.extend_from_slice(&nl);
                new_tokens.extend_from_slice(&sys_content);
                new_tokens.extend_from_slice(&im_end);
                new_tokens.extend_from_slice(&nl);
            }
        }
        new_tokens.extend_from_slice(&im_start);
        new_tokens.extend_from_slice(&user_tok);
        new_tokens.extend_from_slice(&nl);
        new_tokens.extend_from_slice(&q_tokens);
        new_tokens.extend_from_slice(&im_end);
        new_tokens.extend_from_slice(&nl);
        new_tokens.extend_from_slice(&im_start);
        new_tokens.extend_from_slice(&asst_tok);
        new_tokens.extend_from_slice(&nl);

        // Prefill
        let t0 = Instant::now();
        for (i, &tok) in new_tokens.iter().enumerate() {
            qwen35::forward_scratch(&mut gpu, &weights, &config, tok, seq_pos + i, &mut kv_cache, &mut dn_state, &scratch).unwrap();
        }
        seq_pos += new_tokens.len();
        conversation_tokens.extend_from_slice(&new_tokens);

        // Generate
        let mut logits = gpu.download_f32(&scratch.logits).unwrap();
        let mut next_token = llama::sample_top_p(&logits, temp, sc.top_p);
        let mut generated = 0;
        let mut in_thinking = false;
        let mut thinking_shown = false;

        loop {
            generated += 1;
            conversation_tokens.push(next_token);
            let text = tokenizer.decode(&[next_token]);

            // Handle <think>...</think> blocks
            if text.contains("<think>") {
                in_thinking = true;
                if !thinking_shown {
                    eprint!("\x1b[2m"); // dim
                    thinking_shown = true;
                }
            }
            if in_thinking {
                eprint!("{}", text);
                if text.contains("</think>") {
                    in_thinking = false;
                    eprint!("\x1b[0m\n"); // reset
                }
            } else {
                print!("{}", text);
                std::io::stdout().flush().unwrap();
            }

            if next_token == config.eos_token { break; }
            if im_end_token == Some(next_token) { break; }
            if generated >= 2048 { break; } // safety limit

            let pos = seq_pos + generated - 1;
            if pos >= max_seq { break; } // KV capacity
            qwen35::forward_scratch(&mut gpu, &weights, &config, next_token, pos, &mut kv_cache, &mut dn_state, &scratch).unwrap();
            logits = gpu.download_f32(&scratch.logits).unwrap();
            llama::apply_ngram_block(&mut logits, &conversation_tokens);
            llama::apply_repeat_penalty(&mut logits, &conversation_tokens, 128, 1.3);
            next_token = llama::sample_top_p(&logits, temp, sc.top_p);
        }
        seq_pos += generated;
        total_tokens += generated;
        conversation_tokens.extend_from_slice(&im_end);
        conversation_tokens.extend_from_slice(&nl);

        let elapsed = t0.elapsed();
        let tok_s = generated as f64 / elapsed.as_secs_f64();
        eprintln!("\n\x1b[2m({} tokens, {:.1} tok/s)\x1b[0m\n", generated, tok_s);
    }

    eprintln!("Bye!");
}
