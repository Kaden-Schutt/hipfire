//! dflash_mtp_demo: DFlash + MTP composition spec-decode bench harness (Task 11).
//!
//! Loads a Qwen3.5 trunk (.mq4 / .hfq), a matched DFlash drafter (.hfq), and a
//! native MTP head (.mtp). Each cycle:
//!   1. Run dflash drafter (B=16 candidates).
//!   2. MTP fanout (K candidates) seeded from drafter's last hidden.
//!   3. Trunk verify on composite [seed, c1..c15, m1..mK] — single batched
//!      forward over B+K positions.
//!   4. Greedy accept-prefix; the bonus is trunk's argmax at the first miss.
//!
//! Bench output mirrors dflash_spec_demo / mtp_only_demo so the operator can
//! compare directly. Greedy / temp=0 only.
//!
//! Usage:
//!   dflash_mtp_demo --target <trunk.mq4> --drafter <drafter.hfq> \
//!                   --mtp-head <head.mtp> \
//!                   (--prompt "..." | --prompt-file <path>) \
//!                   [--max 120] [--temp 0] [--dflash-b 16] [--mtp-k 2] \
//!                   [--ctx 4096] [--no-chatml] [--kv-mode q8]

#[cfg(not(feature = "deltanet"))]
fn main() {
    eprintln!("build with --features deltanet");
}

#[cfg(feature = "deltanet")]
fn main() {
    use hipfire_arch_qwen35::mtp_compose::{self, MtpComposeState};
    use hipfire_arch_qwen35::mtp_head;
    use hipfire_arch_qwen35::speculative::{
        self, DeltaNetSnapshot, GdnTape, HiddenStateRingBuffer, ModelSlot, ModelSlotConfig,
        VerifyScratch,
    };
    use hipfire_detect::report::prompt_md5;
    use hipfire_runtime::dflash::{DflashConfig, DflashScratch, DflashWeights};
    use hipfire_runtime::hfq::HfqFile;
    use hipfire_runtime::tokenizer::Tokenizer;
    use std::path::Path;
    use std::time::Instant;

    // ── Parse args ─────────────────────────────────────────────────────
    let args: Vec<String> = std::env::args().collect();
    let mut target_path: Option<String> = None;
    let mut drafter_path: Option<String> = None;
    let mut mtp_path: Option<String> = None;
    let mut prompt_str: Option<String> = None;
    let mut prompt_file: Option<String> = None;
    let mut max_tokens: usize = 120;
    let mut ctx_capacity: usize = 4096;
    let mut temp: f32 = 0.0;
    let mut dflash_b: Option<usize> = None;
    let mut mtp_k: usize = 2;
    let mut chatml: bool = true;
    let mut kv_mode_str = String::from("q8");

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--target" => { target_path = Some(args[i + 1].clone()); i += 2; }
            "--drafter" | "--draft" => { drafter_path = Some(args[i + 1].clone()); i += 2; }
            "--mtp-head" => { mtp_path = Some(args[i + 1].clone()); i += 2; }
            "--prompt" => { prompt_str = Some(args[i + 1].clone()); i += 2; }
            "--prompt-file" => { prompt_file = Some(args[i + 1].clone()); i += 2; }
            "--max" => { max_tokens = args[i + 1].parse().unwrap(); i += 2; }
            "--ctx" => { ctx_capacity = args[i + 1].parse().unwrap(); i += 2; }
            "--temp" => { temp = args[i + 1].parse().unwrap(); i += 2; }
            "--dflash-b" => { dflash_b = Some(args[i + 1].parse().unwrap()); i += 2; }
            "--mtp-k" => { mtp_k = args[i + 1].parse().unwrap(); i += 2; }
            "--no-chatml" => { chatml = false; i += 1; }
            "--chatml" => { chatml = true; i += 1; }
            "--kv-mode" => { kv_mode_str = args[i + 1].clone(); i += 2; }
            "-h" | "--help" => {
                eprintln!(
                    "Usage: dflash_mtp_demo --target <trunk.mq4> --drafter <drafter.hfq> \\\n\
                     \t--mtp-head <head.mtp> \\\n\
                     \t(--prompt \"...\" | --prompt-file <path>) \\\n\
                     \t[--max 120] [--temp 0] [--dflash-b 16] [--mtp-k 2] \\\n\
                     \t[--ctx 4096] [--no-chatml] [--kv-mode q8]"
                );
                std::process::exit(0);
            }
            other => {
                eprintln!("unknown arg: {other}");
                std::process::exit(2);
            }
        }
    }

    let target_path = target_path.expect("--target required");
    let drafter_path = drafter_path.expect("--drafter required");
    let mtp_path = mtp_path.expect("--mtp-head required");
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
        eprintln!("error: dflash_mtp_demo v1 is greedy-only (--temp must be 0.0); got {temp}");
        std::process::exit(2);
    }
    assert!(mtp_k >= 1 && mtp_k <= 8, "--mtp-k must be in [1,8]");

    let prompt = hipfire_runtime::tokenizer::maybe_normalize_prompt(&prompt_raw).into_owned();
    let prompt_hash = prompt_md5(prompt.as_bytes());

    eprintln!("=== dflash_mtp_demo ===");
    eprintln!("target:     {target_path}");
    eprintln!("drafter:    {drafter_path}");
    eprintln!("mtp-head:   {mtp_path}");
    eprintln!("prompt md5: {prompt_hash}");
    eprintln!("max={max_tokens} ctx={ctx_capacity} mtp_k={mtp_k} kv_mode={kv_mode_str} chatml={chatml}");

    // ── Init GPU + load drafter cfg ────────────────────────────────────
    let mut gpu = rdna_compute::Gpu::init().expect("gpu init");
    eprintln!("gpu: {}", gpu.arch);

    let draft_hfq = HfqFile::open(Path::new(&drafter_path)).expect("open drafter");
    let mut draft_cfg = DflashConfig::from_hfq(&draft_hfq).expect("parse DflashConfig");
    if let Some(b) = dflash_b {
        let orig = draft_cfg.block_size;
        if b != orig {
            eprintln!(
                "block_size override: {orig} -> {b} (drafter trained at {orig}; \
                 smaller B lowers per-cycle cost but may reduce τ)"
            );
            draft_cfg.block_size = b;
        }
    }
    let b = draft_cfg.block_size;
    let n_verify = b + mtp_k; // composite chain length each verify

    eprintln!(
        "drafter: layers={} hidden={} block={} target_layers={:?}",
        draft_cfg.n_layers, draft_cfg.hidden, b, draft_cfg.target_layer_ids,
    );

    // ── Load trunk ─────────────────────────────────────────────────────
    let mut slot_cfg = ModelSlotConfig::default();
    // Per cycle worst case: position advances by accept_len + 1 ≤ b + mtp_k.
    // Verify writes B+K KV slots before rollback so we need padding.
    slot_cfg.max_seq = ctx_capacity + max_tokens * (b + mtp_k) / b + n_verify + 16;
    slot_cfg.kv_mode = match kv_mode_str.as_str() {
        "q8" => speculative::KvMode::Q8,
        "asym4" => speculative::KvMode::Asym4,
        "asym3" => speculative::KvMode::Asym3,
        "asym2" => speculative::KvMode::Asym2,
        other => {
            eprintln!("unknown --kv-mode: {other}");
            std::process::exit(1);
        }
    };
    let max_seq_total = slot_cfg.max_seq;

    let t_load = Instant::now();
    let mut target = ModelSlot::load(
        &mut gpu, Path::new(&target_path), "target", slot_cfg,
    ).expect("load trunk");
    eprintln!("trunk loaded in {:.2}s", t_load.elapsed().as_secs_f64());

    // ── Load drafter ───────────────────────────────────────────────────
    let t_d = Instant::now();
    let draft_weights = DflashWeights::load(&mut gpu, &draft_hfq, &draft_cfg)
        .expect("load drafter");
    eprintln!("drafter loaded in {:.2}s", t_d.elapsed().as_secs_f64());
    assert_eq!(
        target.config.vocab_size, draft_cfg.vocab_size,
        "trunk vocab ({}) != drafter vocab ({})",
        target.config.vocab_size, draft_cfg.vocab_size,
    );
    assert_eq!(
        target.config.dim, draft_cfg.hidden,
        "trunk dim ({}) != drafter hidden ({}) — task 11 requires matched drafter",
        target.config.dim, draft_cfg.hidden,
    );

    // Drafter scratch sized for B alone (drafter doesn't see MTP slots).
    let mut draft_scratch = DflashScratch::new_with_mq(
        &mut gpu, &draft_cfg, b, ctx_capacity, draft_weights.has_mq,
    ).expect("alloc draft scratch");

    // ── Load MTP head ──────────────────────────────────────────────────
    let t_mtp = Instant::now();
    let head = mtp_head::load_mtp_head(
        Path::new(&mtp_path), &mut gpu, max_seq_total,
    ).expect("load mtp head");
    eprintln!(
        "mtp head loaded in {:.2}s — n_embd={} vocab={} n_rot={} rope_theta={}",
        t_mtp.elapsed().as_secs_f64(),
        head.config.n_embd, head.config.vocab_size,
        head.config.n_rot, head.config.rope_theta,
    );
    assert_eq!(head.config.n_embd, target.config.dim);
    assert_eq!(head.config.vocab_size, target.config.vocab_size);

    // ── Tokenize prompt ────────────────────────────────────────────────
    let tokenizer: Tokenizer = target.load_tokenizer().expect("trunk tokenizer");
    let mut prompt_tokens = tokenizer.encode(&prompt);
    if chatml {
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
        eprintln!("chatml wrap: prompt {} tokens after wrap", prompt_tokens.len());
    } else {
        eprintln!("prompt: {} tokens (no chatml)", prompt_tokens.len());
    }
    assert!(!prompt_tokens.is_empty(), "empty prompt after tokenization");
    assert!(
        prompt_tokens.len() + max_tokens + n_verify + 16 <= max_seq_total,
        "prompt ({}) + max ({}) + n_verify ({}) won't fit in max_seq {}",
        prompt_tokens.len(), max_tokens, n_verify, max_seq_total,
    );

    // ── Hidden ring buffer + snapshot + verify scratch ─────────────────
    //
    // Sized for n_verify = b + mtp_k positions per verify.
    let mut hidden_rb = HiddenStateRingBuffer::new(
        &mut gpu,
        target.config.n_layers,
        draft_cfg.num_extract(),
        draft_cfg.hidden,
        ctx_capacity + n_verify,
        hipfire_arch_qwen35::qwen35::PREFILL_MAX_BATCH.max(n_verify),
    ).expect("alloc hidden_rb");

    let mut target_snap = DeltaNetSnapshot::new_for(&mut gpu, &target.dn_state).expect("snap");
    let mut gdn_tape = GdnTape::new_for_config(
        &mut gpu, &target.config, n_verify,
    ).expect("alloc gdn tape");
    let verify_scratch = VerifyScratch::with_prefill(
        &mut gpu,
        n_verify,
        target.config.dim,
        target.config.vocab_size,
        target.weights.output.k,
        &target.config,
    ).expect("alloc verify scratch");

    let mut compose_state = MtpComposeState::new(&mut gpu, &target, &head, mtp_k)
        .expect("alloc MtpComposeState");

    // ── Prefill: seed target_hidden via per-token forward_with_hidden ──
    let mut target_hidden_host: Vec<f32> =
        Vec::with_capacity(ctx_capacity * draft_cfg.num_extract() * draft_cfg.hidden);
    eprintln!("seeding target_hidden from prompt ({} tokens)...", prompt_tokens.len());
    let t_prefill = Instant::now();
    speculative::seed_target_hidden_from_prompt(
        &mut gpu, &mut target, &mut hidden_rb, &mut target_hidden_host, &prompt_tokens,
    ).expect("seed target hidden");
    speculative::scatter_hidden_block_to_interleaved(
        &gpu,
        &hidden_rb,
        &draft_scratch.target_hidden,
        0,
        prompt_tokens.len(),
        prompt_tokens.len(),
    ).expect("seed scatter");
    draft_scratch.uploaded_target_hidden_rows = prompt_tokens.len();
    draft_scratch.target_hidden_abs_positions = (0..prompt_tokens.len() as i32).collect();
    let prefill_secs = t_prefill.elapsed().as_secs_f64();
    let prefill_tok_s = prompt_tokens.len() as f64 / prefill_secs.max(1e-9);
    eprintln!("prefill in {:.2}s ({:.1} tok/s)", prefill_secs, prefill_tok_s);

    // ── Initial seed_token: trunk's greedy pick after prefill ──────────
    let logits0 = gpu.download_f32(&target.scratch.logits).expect("download logits");
    let mut seed_token = 0u32;
    let mut best = f32::NEG_INFINITY;
    for (i, &v) in logits0.iter().enumerate() {
        if v > best { best = v; seed_token = i as u32; }
    }
    eprintln!(
        "seed token (greedy after prefill): {} ('{}')",
        seed_token,
        tokenizer.decode(&[seed_token]).chars().take(16).collect::<String>(),
    );

    // ── Decode loop ───────────────────────────────────────────────────
    let eos_token = target.config.eos_token;
    let mut emitted: Vec<u32> = vec![seed_token];
    let mut position: usize = prompt_tokens.len();

    let mut cycles = 0usize;
    let mut accept_dflash_total = 0usize;
    let mut accept_mtp_total = 0usize;
    let mut full_dflash_cycles = 0usize; // cycles where dflash accepted all B-1
    let mut hit_eos = tokenizer.is_terminator(seed_token);

    let t_decode = Instant::now();

    while !hit_eos && emitted.len() < max_tokens {
        if position + n_verify >= max_seq_total {
            eprintln!("hit max_seq {}; stopping", max_seq_total);
            break;
        }

        let result = mtp_compose::spec_step_dflash_mtp(
            &mut gpu,
            &mut target,
            &draft_weights,
            &draft_cfg,
            &mut draft_scratch,
            &mut hidden_rb,
            &mut target_snap,
            &verify_scratch,
            Some(&mut gdn_tape),
            &head,
            &mut compose_state,
            position,
            seed_token,
            Some(b),
            mtp_k,
        ).expect("spec_step_dflash_mtp");

        cycles += 1;
        accept_dflash_total += result.accept_dflash;
        accept_mtp_total += result.accept_mtp;
        if result.accept_dflash == b - 1 {
            full_dflash_cycles += 1;
        }

        // step.committed[0] is the seed; emit the rest.
        for &tok in result.committed.iter().skip(1) {
            emitted.push(tok);
            if tok == eos_token {
                hit_eos = true;
            }
        }
        // position advances by accept + 1 (= committed.len() - 1).
        let advance = result.committed.len() - 1;
        position += advance;
        seed_token = *result.committed.last().expect("non-empty commit");

        if hit_eos || emitted.len() >= max_tokens {
            break;
        }
    }
    let decode_secs = t_decode.elapsed().as_secs_f64();

    let total_committed = emitted.len();
    let tok_per_s = total_committed as f64 / decode_secs.max(1e-9);

    // τ_dflash = avg dflash candidates accepted per cycle
    // τ_mtp    = avg MTP candidates accepted per cycle (across ALL cycles, not just full-dflash ones)
    // τ_total  = avg committed per cycle (= 1 + accept_dflash + accept_mtp avg)
    let tau_dflash = if cycles > 0 {
        accept_dflash_total as f64 / cycles as f64
    } else { 0.0 };
    let tau_mtp = if cycles > 0 {
        accept_mtp_total as f64 / cycles as f64
    } else { 0.0 };
    let tau_total = if cycles > 0 {
        ((total_committed - 1) as f64) / cycles as f64
    } else { 0.0 };

    let text = tokenizer.decode(&emitted);
    println!("\n=== output ===\n{text}\n=== end ===");
    println!();
    println!("prompt_md5:           {prompt_hash}");
    println!("prompt_tokens:        {}", prompt_tokens.len());
    println!("dflash_b:             {}", b);
    println!("mtp_k:                {}", mtp_k);
    println!("cycles:               {}", cycles);
    println!("committed_total:      {}", total_committed);
    println!("accept_dflash_total:  {}", accept_dflash_total);
    println!("accept_mtp_total:     {}", accept_mtp_total);
    println!("full_dflash_cycles:   {} ({:.1}%)",
        full_dflash_cycles,
        100.0 * full_dflash_cycles as f64 / cycles.max(1) as f64);
    println!("tau_dflash:           {:.4}", tau_dflash);
    println!("tau_mtp:              {:.4}", tau_mtp);
    println!("tau_total:            {:.4}", tau_total);
    println!("prefill_secs:         {:.3}", prefill_secs);
    println!("prefill_tok_s:        {:.2}", prefill_tok_s);
    println!("decode_secs:          {:.3}", decode_secs);
    println!("tok_s:                {:.2}", tok_per_s);
    println!("eos_hit:              {}", if hit_eos { "y" } else { "n" });

    let preview: String = text.chars().take(200).collect();
    println!("preview_200:          {:?}", preview);

    // Cleanup.
    compose_state.free_gpu(&mut gpu);
    head.free_gpu(&mut gpu);
}
