//! Interactive REPL for hipfire — like `ollama run`.
//! Usage: hipfire-run <model.hfq> [--system "prompt"] [--kv givens4|givens2]

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
        eprintln!("Usage: run <model.hfq> [--draft-model <path>] [--system \"prompt\"] [--kv givens4|givens2] [--temp F] [--max-seq N]");
        std::process::exit(1);
    }
    let model_path = &args[1];

    // Parse flags
    let mut system_prompt: Option<String> = None;
    let mut kv_mode_str: String = "q8".to_string();
    let mut temp: f32 = 0.3;
    let mut max_seq: usize = 4096;
    let mut q4_state = false;
    let mut draft_model: Option<String> = None;
    let mut speculative = false;
    let mut spec_k: usize = 4;
    let mut no_penalty = false;
    let mut i = 2;
    while i < args.len() {
        match args[i].as_str() {
            "--system" | "-s" => { i += 1; system_prompt = Some(args[i].clone()); }
            "--kv" => { i += 1; kv_mode_str = args[i].clone(); }
            "--q4-state" => { q4_state = true; }
            "--temp" => { i += 1; temp = args[i].parse().unwrap_or(0.3); }
            "--max-seq" => { i += 1; max_seq = args[i].parse().unwrap_or(4096); }
            "--draft-model" => { i += 1; draft_model = Some(args[i].clone()); }
            "--speculative" => { speculative = true; }
            "--spec-k" => { i += 1; spec_k = args[i].parse().unwrap_or(4).max(1); }
            "--no-penalty" => { no_penalty = true; }
            _ => {}
        }
        i += 1;
    }

    // Load model
    let mut gpu = rdna_compute::Gpu::init().expect("GPU init failed");
    eprintln!("Loading {}...", model_path);

    use engine::speculative::{KvMode, ModelSlot, ModelSlotConfig};
    let state_quant = if q4_state { qwen35::StateQuant::Q4 } else { qwen35::StateQuant::Q8 };
    if q4_state { eprintln!("DeltaNet state: Q4 (half VRAM vs Q8)"); }
    eprintln!("KV cache: {kv_mode_str}");
    let target_kv_mode = KvMode::Q8;
    let target_cfg = ModelSlotConfig {
        max_seq, kv_mode: target_kv_mode, repeat_window: 128, state_quant,
    };
    let mut target_slot = ModelSlot::load(&mut gpu, Path::new(model_path), "target", target_cfg)
        .expect("failed to load target model");
    let tokenizer = target_slot.load_tokenizer().expect("failed to load tokenizer");

    // Optional draft model slot (Phase 1 of speculative decode). Validated for
    // tokenizer compatibility, smoke-tested, then parked. The REPL still runs
    // the target model alone until Phase 2 wires in the verify-and-accept loop.
    let mut draft_slot: Option<engine::speculative::ModelSlot> = None;
    if let Some(ref dpath) = draft_model {
        use engine::speculative::{KvMode, ModelSlot, ModelSlotConfig};
        let vram_before = gpu.hip.get_vram_info().map(|(f, _)| f).unwrap_or(0);

        let draft_cfg = ModelSlotConfig {
            max_seq,
            kv_mode: KvMode::Q8,
            repeat_window: 128,
            state_quant,
        };

        eprintln!("Loading draft {}...", dpath);
        let mut slot = ModelSlot::load(&mut gpu, Path::new(dpath), "draft", draft_cfg)
            .expect("failed to load draft model");

        // Tokenizer compatibility check (vocab size + probe round-trip).
        let draft_tok = slot.load_tokenizer().expect("draft has no tokenizer in HFQ metadata");
        assert_eq!(
            tokenizer.vocab_size(), draft_tok.vocab_size(),
            "tokenizer mismatch: target vocab={} draft vocab={} — speculative decode requires identical vocabularies",
            tokenizer.vocab_size(), draft_tok.vocab_size()
        );
        let probe = "<|im_start|>user\nHello world\n<|im_end|>";
        assert_eq!(
            tokenizer.encode(probe), draft_tok.encode(probe),
            "tokenizer merge rules diverge between target and draft"
        );

        // Smoke test: 8 forward passes with a placeholder token. Must produce finite logits.
        for pos in 0..8 {
            slot.forward(&mut gpu, 1u32, pos).expect("draft smoke-test forward failed");
        }
        let draft_logits = gpu.download_f32(&slot.scratch.logits).unwrap();
        let draft_ok = draft_logits.iter().take(1024).all(|x| x.is_finite());
        assert!(draft_ok, "draft smoke test produced non-finite logits");
        slot.reset_state(&mut gpu);

        let vram_after = gpu.hip.get_vram_info().map(|(f, _)| f).unwrap_or(0);
        let draft_mb = (vram_before.saturating_sub(vram_after)) as f64 / 1e6;
        eprintln!(
            "Draft: {} layers, dim={}, vocab={} — VRAM {:.0} MB, smoke test OK",
            slot.config.n_layers, slot.config.dim, slot.config.vocab_size, draft_mb
        );
        draft_slot = Some(slot);
    }

    // Speculative decode mode requires a draft model.
    let mut spec_active = speculative && draft_slot.is_some();
    if speculative && draft_slot.is_none() {
        eprintln!("--speculative ignored: no --draft-model provided");
    }
    // Snapshots for DeltaNet state rollback during verify-and-accept. Allocated
    // once and reused across REPL turns. Only materialized in spec mode.
    let mut target_snap: Option<engine::speculative::DeltaNetSnapshot> = None;
    let mut draft_snap: Option<engine::speculative::DeltaNetSnapshot> = None;
    if spec_active {
        use engine::speculative::DeltaNetSnapshot;
        target_snap = Some(DeltaNetSnapshot::new_for(&mut gpu, &target_slot.dn_state).unwrap());
        if let Some(ref d) = draft_slot {
            draft_snap = Some(DeltaNetSnapshot::new_for(&mut gpu, &d.dn_state).unwrap());
        }
        eprintln!(
            "Speculative decode: greedy, K={}, draft={}",
            spec_k,
            draft_slot.as_ref().map(|d| d.name.as_str()).unwrap_or("?")
        );
    }

    eprintln!("Model: {} layers, dim={}, vocab={}", target_slot.config.n_layers, target_slot.config.dim, target_slot.config.vocab_size);
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
    let mut total_tokens: usize = 0;
    // Aggregate speculative decode stats across REPL turns (only populated when
    // --speculative is active). Shown via /stats.
    let mut spec_stats = engine::speculative::SpecStats::new(spec_k);

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
                target_slot.reset_state(&mut gpu);
                if let Some(ref mut d) = draft_slot { d.reset_state(&mut gpu); }
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
                eprintln!("Total generated: {} tokens", total_tokens);
                if spec_active && spec_stats.cycles > 0 {
                    eprintln!(
                        "Speculative: {} cycles, tau={:.2} (accepted/cycle), committed/cycle={:.2}",
                        spec_stats.cycles, spec_stats.tau(), spec_stats.mean_committed()
                    );
                    eprint!("  acceptance histogram: ");
                    for (i, &c) in spec_stats.acceptance_hist.iter().enumerate() {
                        eprint!("a{}={} ", i, c);
                    }
                    eprintln!();
                }
                eprintln!();
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
            target_slot.reset_state(&mut gpu);
            if let Some(ref mut d) = draft_slot { d.reset_state(&mut gpu); }
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

        // Prefill: run the prompt through BOTH models so their state is
        // aligned at the same position. In non-spec mode the draft model is
        // still fed the prompt so that /toggle-mid-session works cleanly,
        // though the draft's state is unused until speculative is enabled.
        let t0 = Instant::now();
        for (i, &tok) in new_tokens.iter().enumerate() {
            target_slot.forward(&mut gpu, tok, seq_pos + i).unwrap();
            if spec_active {
                if let Some(ref mut d) = draft_slot {
                    d.forward(&mut gpu, tok, seq_pos + i).unwrap();
                }
            }
        }
        seq_pos += new_tokens.len();
        conversation_tokens.extend_from_slice(&new_tokens);

        let mut generated = 0usize;
        let mut in_thinking = false;
        let mut thinking_shown = false;
        // Capture EOS token IDs as plain values so the emit_token closure
        // doesn't borrow from `target_slot` (which would conflict with the
        // later &mut target_slot passed into spec_step_greedy).
        let eos_token = target_slot.config.eos_token;
        let im_end_token_val = im_end_token;

        // Helper closure: prints a token and returns true if generation should stop.
        let mut emit_token = |tok: u32,
                              conversation_tokens: &mut Vec<u32>,
                              in_thinking: &mut bool,
                              thinking_shown: &mut bool,
                              generated: &mut usize| -> bool {
            *generated += 1;
            conversation_tokens.push(tok);
            let text = tokenizer.decode(&[tok]);
            if text.contains("<think>") {
                *in_thinking = true;
                if !*thinking_shown {
                    eprint!("\x1b[2m");
                    *thinking_shown = true;
                }
            }
            if *in_thinking {
                eprint!("{}", text);
                if text.contains("</think>") {
                    *in_thinking = false;
                    eprint!("\x1b[0m\n");
                }
            } else {
                print!("{}", text);
                std::io::stdout().flush().unwrap();
            }
            tok == eos_token || im_end_token_val == Some(tok) || tokenizer.is_terminator(tok)
        };

        if spec_active {
            // Speculative decode loop. Each cycle drafts spec_k tokens, the
            // target verifies them sequentially (Phase 2 naive path), and the
            // accepted prefix + bonus is committed to both models.
            let ts = target_snap.as_mut().unwrap();
            let ds = draft_snap.as_mut().unwrap();
            let draft_ref = draft_slot.as_mut().unwrap();
            'outer: loop {
                let pos = seq_pos + generated;
                if pos + spec_k + 1 >= max_seq { break; }

                let step = engine::speculative::spec_step_greedy(
                    &mut gpu, &mut target_slot, draft_ref, pos, spec_k, ts, ds,
                ).unwrap();
                spec_stats.record(&step);

                for tok in &step.committed {
                    let stop = emit_token(
                        *tok, &mut conversation_tokens,
                        &mut in_thinking, &mut thinking_shown, &mut generated,
                    );
                    if stop { break 'outer; }
                    if generated >= 2048 { break 'outer; }
                }
            }
        } else {
            // Target-only generation path (baseline, unchanged behavior).
            let mut logits = gpu.download_f32(&target_slot.scratch.logits).unwrap();
            let mut next_token = llama::sample_top_p(&logits, temp, sc.top_p);
            loop {
                let stop = emit_token(
                    next_token, &mut conversation_tokens,
                    &mut in_thinking, &mut thinking_shown, &mut generated,
                );
                if stop { break; }
                if generated >= 2048 { break; }

                let pos = seq_pos + generated - 1;
                if pos >= max_seq { break; }
                target_slot.forward(&mut gpu, next_token, pos).unwrap();
                logits = gpu.download_f32(&target_slot.scratch.logits).unwrap();
                if !no_penalty {
                    llama::apply_ngram_block(&mut logits, &conversation_tokens);
                    llama::apply_repeat_penalty(&mut logits, &conversation_tokens, sc.repeat_window, sc.repeat_penalty);
                }
                next_token = llama::sample_top_p(&logits, temp, sc.top_p);
            }
        }

        seq_pos += generated;
        total_tokens += generated;
        conversation_tokens.extend_from_slice(&im_end);
        conversation_tokens.extend_from_slice(&nl);

        let elapsed = t0.elapsed();
        let tok_s = generated as f64 / elapsed.as_secs_f64();
        if spec_active && spec_stats.cycles > 0 {
            eprintln!(
                "\n\x1b[2m({} tokens, {:.1} tok/s | spec: {} cycles, tau={:.2})\x1b[0m\n",
                generated, tok_s, spec_stats.cycles, spec_stats.tau()
            );
        } else {
            eprintln!("\n\x1b[2m({} tokens, {:.1} tok/s)\x1b[0m\n", generated, tok_s);
        }
    }

    eprintln!("Bye!");
}
