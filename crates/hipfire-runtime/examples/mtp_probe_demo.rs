//! mtp_probe_demo: target-only Qualcomm-style MTP probe bench harness.
//!
//! Loads a Qwen3.5 target (.hfq), prefills the prompt, then loops
//! `mtp_probe::mtp_probe_step` (single mask token per cycle, greedy lossless
//! verify, no drafter) until N tokens committed or EOS. Prints τ + tok/s +
//! prompt md5. v1 — engine-surface validation.
//!
//! Usage:
//!   mtp_probe_demo --target <target.hfq> \
//!                  (--prompt "Hello" | --prompt-file path) \
//!                  [--max 64] [--ctx 4096] [--lambda 0.1] [--temp 0.0]
//!                  [--no-chatml]

#[cfg(not(feature = "deltanet"))]
fn main() {
    eprintln!("build with --features deltanet");
}

#[cfg(feature = "deltanet")]
fn main() {
    use hipfire_arch_qwen35::mtp_probe::{self, MtpProbeState, MtpProbeStats};
    use hipfire_arch_qwen35::speculative::{ModelSlot, ModelSlotConfig};
    use hipfire_detect::report::prompt_md5;
    use hipfire_runtime::tokenizer::Tokenizer;
    use std::path::Path;
    use std::time::Instant;

    // ── Parse args ─────────────────────────────────────────────────────
    let args: Vec<String> = std::env::args().collect();
    let mut target_path: Option<String> = None;
    let mut prompt_str: Option<String> = None;
    let mut prompt_file: Option<String> = None;
    let mut max_tokens: usize = 64;
    let mut ctx_capacity: usize = 4096;
    let mut lambda: f32 = 0.1;
    let mut temp: f32 = 0.0;
    let mut chatml: bool = true;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--target" => { target_path = Some(args[i + 1].clone()); i += 2; }
            "--prompt" => { prompt_str = Some(args[i + 1].clone()); i += 2; }
            "--prompt-file" => { prompt_file = Some(args[i + 1].clone()); i += 2; }
            "--max" => { max_tokens = args[i + 1].parse().unwrap(); i += 2; }
            "--ctx" => { ctx_capacity = args[i + 1].parse().unwrap(); i += 2; }
            "--lambda" => { lambda = args[i + 1].parse().unwrap(); i += 2; }
            "--temp" => { temp = args[i + 1].parse().unwrap(); i += 2; }
            "--no-chatml" => { chatml = false; i += 1; }
            "--chatml" => { chatml = true; i += 1; }
            "-h" | "--help" => {
                eprintln!("Usage: mtp_probe_demo --target <target.hfq> \\\n  (--prompt \"Hello\" | --prompt-file <path>) \\\n  [--max 64] [--ctx 4096] [--lambda 0.1] [--temp 0.0] [--no-chatml]");
                std::process::exit(0);
            }
            other => {
                eprintln!("unknown arg: {other}");
                std::process::exit(2);
            }
        }
    }

    let target_path = target_path.expect("--target required");
    if prompt_str.is_some() == prompt_file.is_some() {
        eprintln!("exactly one of --prompt or --prompt-file is required");
        std::process::exit(2);
    }
    let prompt_raw = if let Some(s) = prompt_str {
        s
    } else {
        let p = prompt_file.unwrap();
        std::fs::read_to_string(&p).unwrap_or_else(|e| {
            eprintln!("failed to read --prompt-file {p}: {e}");
            std::process::exit(2);
        })
    };
    if temp != 0.0 {
        eprintln!("error: MTP probe v1 is greedy-only (--temp must be 0.0); got {temp}");
        std::process::exit(2);
    }
    let prompt = hipfire_runtime::tokenizer::maybe_normalize_prompt(&prompt_raw).into_owned();
    // md5 of the normalized bare prompt string (pre-ChatML wrap), consistent
    // with the dflash_spec_demo convention. Used for cross-session bench
    // reproducibility per CLAUDE.md prompt-structure τ rule.
    let prompt_hash = prompt_md5(prompt.as_bytes());

    eprintln!("=== mtp_probe_demo ===");
    eprintln!("target: {target_path}");
    eprintln!("prompt md5: {prompt_hash}");
    eprintln!("max={max_tokens} ctx={ctx_capacity} lambda={lambda} chatml={chatml}");

    // ── Init GPU + load target ─────────────────────────────────────────
    let mut gpu = rdna_compute::Gpu::init().expect("gpu init");
    eprintln!("gpu: {}", gpu.arch);

    let mut slot_cfg = ModelSlotConfig::default();
    // Worst-case KV consumption: prompt_len + max_tokens * MTP_PROBE_MAX_BATCH (3)
    // per cycle (probe advances by 2 or 3 KV slots regardless of how many tokens
    // commit; the candidate slot writes to KV even on rejection). +8 padding.
    slot_cfg.max_seq = ctx_capacity + max_tokens * 3 + 8;
    let t_load = Instant::now();
    let mut target = ModelSlot::load(
        &mut gpu, Path::new(&target_path), "target", slot_cfg,
    ).expect("load target");
    eprintln!("target loaded in {:.2}s", t_load.elapsed().as_secs_f64());

    // ── Tokenize prompt ───────────────────────────────────────────────
    let tokenizer: Tokenizer = target.load_tokenizer().expect("target tokenizer");
    let mut prompt_tokens = tokenizer.encode(&prompt);
    if chatml {
        // Match dflash_spec_demo daemon path: <|im_start|>user\n{p}<|im_end|>\n<|im_start|>assistant\n
        let im_start = tokenizer.encode("<|im_start|>");
        let im_end = tokenizer.encode("<|im_end|>");
        let user = tokenizer.encode("user");
        let asst = tokenizer.encode("assistant");
        let nl = tokenizer.encode("\n");
        assert!(im_start.len() == 1, "tokenizer has no <|im_start|> special");
        let mut chat = Vec::new();
        chat.extend_from_slice(&im_start);
        chat.extend_from_slice(&user);
        chat.extend_from_slice(&nl);
        chat.extend_from_slice(&prompt_tokens);
        chat.extend_from_slice(&im_end);
        chat.extend_from_slice(&nl);
        chat.extend_from_slice(&im_start);
        chat.extend_from_slice(&asst);
        chat.extend_from_slice(&nl);
        prompt_tokens = chat;
        eprintln!("chatml wrapping: prompt {} tokens after wrap", prompt_tokens.len());
    } else {
        eprintln!("prompt: {} tokens (no chatml)", prompt_tokens.len());
    }
    assert!(!prompt_tokens.is_empty(), "empty prompt after tokenization");
    assert!(
        prompt_tokens.len() + max_tokens * 3 + 8 <= ctx_capacity,
        "prompt ({}) + max ({}) * 3 KV slots/cycle won't fit in --ctx {}; raise --ctx",
        prompt_tokens.len(), max_tokens, ctx_capacity,
    );

    // ── Prefill the target's KV cache one token at a time ─────────────
    //
    // Prefill the FULL prompt (positions 0..prompt_len). The MTP probe
    // contract (mtp_probe.rs lines 224-227) is that slot 0 of cycle 0
    // carries the `last_committed` = final prompt token, written at
    // cur_pos = prompt_len. The KV row at prompt_len thus repeats the
    // K/V of the final prompt token, and slot 0's argmax becomes the
    // first emitted token at logical position prompt_len + 1.
    //
    // CAVEAT (2026-05-14): on this branch, both this probe path AND the
    // bare `infer_qwen35 --guards on` AR path produce a `!!!!!!` single-
    // token attractor on Qwen3.5 0.8b/9b mq4 with greedy decode. The
    // probe is consistent with the AR baseline (same engine surface,
    // same degenerate output), so the symptom is environmental rather
    // than introduced by the probe wiring. Task 4's 27B canonical bench
    // will reveal whether the larger model recovers under the same path.
    eprintln!("prefilling {} tokens...", prompt_tokens.len());
    let t_prefill = Instant::now();
    for (pos, &token) in prompt_tokens.iter().enumerate() {
        target.forward(&mut gpu, token, pos).expect("prefill forward");
    }
    let prefill_secs = t_prefill.elapsed().as_secs_f64();
    let prefill_tok_s = prompt_tokens.len() as f64 / prefill_secs.max(1e-9);
    eprintln!("prefill: {:.2}s ({:.1} tok/s)", prefill_secs, prefill_tok_s);

    let eos_token = target.config.eos_token;

    // ── Init probe state ──────────────────────────────────────────────
    let mut probe_state = MtpProbeState::new_for_prompt(
        &mut gpu, &target.weights, &target.config, &prompt_tokens,
    ).expect("init mtp probe state");
    probe_state.lambda = lambda;
    probe_state.last_committed = Some(*prompt_tokens.last().unwrap());

    // KV holds positions [0..prompt_len). Slot 0 of cycle 0 lands at
    // cur_pos = prompt_len, carrying the final prompt token (per the
    // mtp_probe_step doc seed contract).
    let mut cur_pos: usize = prompt_tokens.len();

    // ── Main loop ─────────────────────────────────────────────────────
    let mut stats = MtpProbeStats::default();
    let mut emitted: Vec<u32> = Vec::with_capacity(max_tokens + 4);
    let t_decode = Instant::now();
    while emitted.len() < max_tokens {
        // Per the mtp_probe_step doc: KV advance is `2 + pending.is_some()`,
        // NOT committed.len(). Compute the advance BEFORE the call so we know
        // exactly how many slots this cycle consumes.
        let kv_advance = 2 + probe_state.pending_candidate.is_some() as usize;
        let (committed, eos_hit) = mtp_probe::mtp_probe_step(
            &mut gpu, &mut target, &mut probe_state, cur_pos, eos_token,
        ).expect("mtp_probe_step");

        for &t in &committed {
            emitted.push(t);
        }
        stats.cycles += 1;
        stats.committed_real += 1;
        stats.committed_speculative += committed.len().saturating_sub(1);
        stats.mask_proposed += 1;
        cur_pos += kv_advance;

        if eos_hit {
            stats.eos_hit = true;
            break;
        }
        if emitted.len() >= max_tokens {
            break;
        }
    }
    let decode_secs = t_decode.elapsed().as_secs_f64();

    // Trim if we overshot max (a 2-token cycle on the boundary).
    let total_committed = emitted.len();
    let tok_per_s = total_committed as f64 / decode_secs.max(1e-9);
    let text = tokenizer.decode(&emitted);

    println!("\n=== output ===\n{text}\n=== end ===");
    println!();
    println!("prompt_md5:           {prompt_hash}");
    println!("prompt_tokens:        {}", prompt_tokens.len());
    println!("cycles:               {}", stats.cycles);
    println!("committed_total:      {}", total_committed);
    println!("committed_real:       {}", stats.committed_real);
    println!("committed_spec:       {}", stats.committed_speculative);
    println!("mask_proposed:        {}", stats.mask_proposed);
    println!("tau:                  {:.4}", stats.tau());
    println!("decode_secs:          {:.3}", decode_secs);
    println!("tok_s:                {:.2}", tok_per_s);
    println!("prefill_tok_s:        {:.2}", prefill_tok_s);
    println!("eos_hit:              {}", if stats.eos_hit { "y" } else { "n" });
    println!("lambda:               {lambda}");

    // Free probe GPU buffers (cosmetic — process exits next).
    probe_state.free_gpu(&mut gpu);
}
