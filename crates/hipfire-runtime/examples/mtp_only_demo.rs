//! mtp_only_demo: standalone Qwen3.5 MTP-only spec-decode bench harness.
//!
//! Loads a Qwen3.5 trunk (.hfq / .mq4 / etc.) and a native MTP head (.mtp,
//! produced by `mtp_extract`, Task 8). Prefills the prompt, then loops
//! `mtp_spec::spec_step_mtp` until N tokens committed or EOS. Prints τ +
//! tok/s + prompt md5 + decoded output. v1 (greedy, --temp 0).
//!
//! Usage:
//!   mtp_only_demo --target <trunk.hfq> --mtp-head <head.mtp> \
//!                 (--prompt "Hello" | --prompt-file <path>) \
//!                 [--max 64] [--ctx 4096] [--temp 0.0] [--max-n 3]
//!                 [--no-chatml]

#[cfg(not(feature = "deltanet"))]
fn main() {
    eprintln!("build with --features deltanet");
}

#[cfg(feature = "deltanet")]
fn main() {
    use hipfire_arch_qwen35::mtp_head;
    use hipfire_arch_qwen35::mtp_spec::{self, MtpSpecState};
    use hipfire_arch_qwen35::speculative::{ModelSlot, ModelSlotConfig};
    use hipfire_detect::report::prompt_md5;
    use hipfire_runtime::tokenizer::Tokenizer;
    use std::path::Path;
    use std::time::Instant;

    // ── Parse args ─────────────────────────────────────────────────────
    let args: Vec<String> = std::env::args().collect();
    let mut target_path: Option<String> = None;
    let mut mtp_path: Option<String> = None;
    let mut prompt_str: Option<String> = None;
    let mut prompt_file: Option<String> = None;
    let mut max_tokens: usize = 64;
    let mut ctx_capacity: usize = 4096;
    let mut temp: f32 = 0.0;
    let mut max_n: usize = 3;
    let mut chatml: bool = true;
    let mut compressed: bool = false;
    let mut compressed_serial: bool = false;
    let mut kv_mode_str: String = String::from("q8");
    let mut p_min: f32 = 0.0;
    // Sampling parameters (Unsloth-recommended for Qwen3.5/3.6 MTP):
    //   thinking-mode default: temp=1.0, top_p=0.95, top_k=20, min_p=0.0
    //   coding-mode:          temp=0.6, top_p=0.95, top_k=20, min_p=0.0
    // temp=0.0 keeps the legacy greedy / argmax-match accept path.
    let mut top_p: f32 = 1.0;
    let mut top_k: usize = 0;
    let mut min_p: f32 = 0.0;
    let mut seed: u64 = 42;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--target" => { target_path = Some(args[i + 1].clone()); i += 2; }
            "--mtp-head" => { mtp_path = Some(args[i + 1].clone()); i += 2; }
            "--prompt" => { prompt_str = Some(args[i + 1].clone()); i += 2; }
            "--prompt-file" => { prompt_file = Some(args[i + 1].clone()); i += 2; }
            "--max" => { max_tokens = args[i + 1].parse().unwrap(); i += 2; }
            "--ctx" => { ctx_capacity = args[i + 1].parse().unwrap(); i += 2; }
            "--temp" => { temp = args[i + 1].parse().unwrap(); i += 2; }
            "--max-n" => { max_n = args[i + 1].parse().unwrap(); i += 2; }
            "--no-chatml" => { chatml = false; i += 1; }
            "--chatml" => { chatml = true; i += 1; }
            "--compressed" => { compressed = true; i += 1; }
            "--compressed-serial" => { compressed = true; compressed_serial = true; i += 1; }
            "--kv-mode" => { kv_mode_str = args[i + 1].clone(); i += 2; }
            "--mtp-p-min" => { p_min = args[i + 1].parse().unwrap(); i += 2; }
            "--top-p" => { top_p = args[i + 1].parse().unwrap(); i += 2; }
            "--top-k" => { top_k = args[i + 1].parse().unwrap(); i += 2; }
            "--min-p" => { min_p = args[i + 1].parse().unwrap(); i += 2; }
            "--seed" => { seed = args[i + 1].parse().unwrap(); i += 2; }
            "-h" | "--help" => {
                eprintln!(
                    "Usage: mtp_only_demo --target <trunk.hfq> --mtp-head <head.mtp> \\\n\
                     \t(--prompt \"Hello\" | --prompt-file <path>) \\\n\
                     \t[--max 64] [--ctx 4096] [--temp 0.0] [--max-n 3] \\\n\
                     \t[--no-chatml] [--compressed]\n\
                     \n\
                     --compressed: use FastMTP-style compressed lm_head_draft (K=1 path).\n\
                                   Requires .mtp head built with --vocab-sidecar.\n\
                                   Forces max_n=1 since the compressed spec is K=1 only."
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
    // --mtp-head is optional when the target is a bundled .mq4-mtp file
    // (the trailer points at the embedded MTP section). For plain .mq4
    // targets it's required.
    let target_is_bundle = target_path.ends_with(".mq4-mtp");
    let mtp_path = match (mtp_path, target_is_bundle) {
        (Some(p), _) => Some(p),
        (None, true) => None,  // resolved from bundle below
        (None, false) => {
            eprintln!("--mtp-head required for non-bundle .mq4 target (got '{target_path}')");
            std::process::exit(2);
        }
    };
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
    // temp > 0 enables residual-sampling spec decode (Unsloth/llama.cpp
    // canonical MTP config). temp == 0 keeps the legacy greedy /
    // argmax-match accept rule.
    if temp < 0.0 {
        eprintln!("error: --temp must be >= 0.0, got {temp}");
        std::process::exit(2);
    }
    if temp > 0.0 && !compressed_serial {
        eprintln!("error: --temp > 0 is only supported on --compressed-serial path \
                   (other paths are greedy-only); got temp={temp}");
        std::process::exit(2);
    }
    assert!(max_n >= 1 && max_n <= 8, "--max-n must be in [1,8]");
    if compressed && max_n > 1 {
        eprintln!("compressed K={max_n}: chains K block forwards with lossy embedding \
                   override (same OOD risk as plain K-step path). Batched compressed \
                   lm_head over the K outputs amortizes the BW saving across the chain.");
    }

    let prompt = hipfire_runtime::tokenizer::maybe_normalize_prompt(&prompt_raw).into_owned();
    let prompt_hash = prompt_md5(prompt.as_bytes());

    eprintln!("=== mtp_only_demo ===");
    eprintln!("target:     {target_path}");
    eprintln!("mtp-head:   {}",
        mtp_path.as_deref().unwrap_or("<bundled in target .mq4-mtp>"));
    eprintln!("prompt md5: {prompt_hash}");
    eprintln!("max={max_tokens} ctx={ctx_capacity} max_n={max_n} chatml={chatml}");

    // ── Init GPU + load trunk + load MTP head ──────────────────────────
    let mut gpu = rdna_compute::Gpu::init().expect("gpu init");
    eprintln!("gpu: {}", gpu.arch);

    let mut slot_cfg = ModelSlotConfig::default();
    // Worst case per cycle: max_n + 1 KV slots written by trunk verify;
    // we replay back to advance ≤ max_n + 1, but the verify path actually
    // fills positions [cur_pos..cur_pos + max_n + 1) before the rollback
    // truncates back. Size for the FULL verify width plus padding.
    slot_cfg.max_seq = ctx_capacity + max_tokens * (max_n + 1) + 16;
    let max_seq_total = slot_cfg.max_seq;
    let t_load = Instant::now();
    let mut target = ModelSlot::load(
        &mut gpu, Path::new(&target_path), "target", slot_cfg,
    ).expect("load target");
    eprintln!("trunk loaded in {:.2}s", t_load.elapsed().as_secs_f64());

    // MTP head's max_seq mirrors the trunk's. The head's KV cache is one
    // single layer, so even max_seq = 100K is only ~250 MB at dim=5120.
    let t_mtp = Instant::now();
    let head = if let Some(ref mp) = mtp_path {
        // Explicit --mtp-head: standalone .mtp file (legacy path).
        mtp_head::load_mtp_head(Path::new(mp), &mut gpu, max_seq_total)
            .expect("load mtp head")
    } else {
        // Bundled .mq4-mtp: load MTP section embedded in target file.
        mtp_head::load_mtp_head_bundled(Path::new(&target_path), &mut gpu, max_seq_total)
            .expect("load bundled mtp head")
            .expect("target ends in .mq4-mtp but no MTP bundle trailer found; \
                     was the file produced by mq4_merge_mtp?")
    };
    let head_source = if mtp_path.is_some() { "standalone .mtp" } else { "bundled .mq4-mtp" };
    eprintln!("mtp head loaded in {:.2}s ({head_source}) — n_embd={} vocab={} n_rot={} rope_theta={} compressed_lm_head_draft={}",
              t_mtp.elapsed().as_secs_f64(),
              head.config.n_embd, head.config.vocab_size,
              head.config.n_rot, head.config.rope_theta,
              head.weights.lm_head_draft.is_some());

    // Sanity dims
    assert_eq!(head.config.n_embd, target.config.dim, "trunk/head dim mismatch");
    assert_eq!(head.config.vocab_size, target.config.vocab_size,
               "trunk/head vocab mismatch");

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
        prompt_tokens.len() + max_tokens * (max_n + 1) + 16 <= max_seq_total,
        "prompt ({}) + max ({}) × (max_n + 1) ({}) won't fit in max_seq {}",
        prompt_tokens.len(), max_tokens, max_n + 1, max_seq_total,
    );

    // ── Allocate spec state ────────────────────────────────────────────
    let kv_mode = hipfire_arch_qwen35::mtp_head::MtpKvMode::parse(&kv_mode_str)
        .unwrap_or_else(|e| { eprintln!("{e}"); std::process::exit(2); });
    eprintln!("mtp-head kv-mode: {kv_mode:?}");
    let mut state = MtpSpecState::new_for_slot_with_kv_mode(&mut gpu, &target, &head, max_n, kv_mode)
        .expect("alloc MtpSpecState");
    if p_min > 0.0 {
        if !compressed_serial {
            eprintln!("warning: --mtp-p-min only affects --compressed-serial path; ignored for the other two");
        }
        state.set_p_min(p_min);
        eprintln!("mtp-p-min: {p_min} (early-exit chain at log P(argmax) < {:.4})", p_min.ln());
    }
    if temp > 0.0 {
        let cfg = hipfire_arch_qwen35::mtp_spec::MtpSamplingConfig {
            temp,
            top_k,
            top_p,
            min_p,
        };
        state.set_sampling(cfg, seed);
        eprintln!("sampling: temp={temp} top_k={top_k} top_p={top_p} min_p={min_p} seed={seed}");
    }

    // Compressed mode: two sub-cases now:
    //   (a) Head has a compressed lm_head_draft sidecar (`compressed_vocab_size:
    //       Some`): allocate logits_compressed scratch + sub-batched scratch.
    //   (b) Head has NO sidecar (e.g. bundled .mq4-mtp using full-vocab trunk
    //       lm_head): nothing to allocate; spec_step_mtp_compressed_serial
    //       branches internally on lm_head_draft.is_none() and uses the
    //       trunk's lm_head for the K-step draft GEMV. compressed-serial path
    //       is the only one that supports this; plain `--compressed` (batched
    //       lossy K-step path) still requires a sidecar.
    if compressed {
        match head.weights.compressed_vocab_size {
            Some(cvs) => {
                state.mtp_scratch
                    .ensure_compressed_logits(&mut gpu, cvs)
                    .expect("alloc logits_compressed");
                state.ensure_compressed_lm_logits(&mut gpu, cvs)
                    .expect("alloc mtp_lm_logits_compressed");
                eprintln!("compressed: ON (cvs={cvs}, K={max_n}, mode=sidecar)");
            }
            None if compressed_serial => {
                // Full-vocab discrete-token chain via trunk's lm_head.
                // spec_step_mtp_compressed_serial dispatches against
                // trunk_weights.output and writes into state.mtp_lm_logits.
                eprintln!(
                    "compressed: ON (mode=full-vocab, K={max_n}, vocab={}) — \
                     trunk lm_head used per-step",
                    target.config.vocab_size
                );
            }
            None => {
                eprintln!(
                    "error: --compressed (batched lossy path) requires a sidecar; \
                     loaded head has no compressed_lm_head_draft. Use \
                     --compressed-serial instead (supports full-vocab trunk \
                     lm_head fallback) or pass a sidecar-extracted .mtp."
                );
                std::process::exit(2);
            }
        }
    }

    let eos_token = target.config.eos_token;

    // ── Prefill prompt one token at a time ─────────────────────────────
    //
    // Per-token AR prefill. After the LAST token, target.scratch.tmp holds
    // the post-output-norm hidden at position prompt_len - 1, which IS the
    // hidden whose argmax produces the seed_token (= cycle 0's last_committed).
    //
    // Note: we use forward_scratch (not the batched prefill path) because we
    // need the per-token-final post-output-norm hidden snapshot at the END,
    // and forward_scratch leaves it in target.scratch.tmp. The batched path
    // does too (only-last-token path, line 5800-5802), but per-token gives
    // us a coherent baseline first.
    eprintln!("prefilling {} tokens...", prompt_tokens.len());
    let t_prefill = Instant::now();
    for (pos, &token) in prompt_tokens.iter().enumerate() {
        target.forward(&mut gpu, token, pos).expect("prefill forward");
    }
    let prefill_secs = t_prefill.elapsed().as_secs_f64();
    let prefill_tok_s = prompt_tokens.len() as f64 / prefill_secs.max(1e-9);
    eprintln!("prefill: {:.2}s ({:.1} tok/s)", prefill_secs, prefill_tok_s);

    // Snapshot trunk's prev_hidden (post-output-norm at last prefill position).
    state.capture_prev_hidden_from_scratch_tmp(
        &gpu, &target.scratch.tmp, target.config.dim,
    ).expect("capture prev_hidden");

    // Pick the seed_token: argmax of the trunk's logits for the last prefill
    // position. This becomes cycle 0's `last_committed`.
    let logits0 = gpu.download_f32(&target.scratch.logits)
        .expect("download seed logits");
    let mut seed_token = 0u32;
    let mut best = f32::NEG_INFINITY;
    for (i, &v) in logits0.iter().enumerate() {
        if v > best { best = v; seed_token = i as u32; }
    }
    eprintln!("seed token (greedy after prefill): {} ('{}')",
              seed_token,
              tokenizer.decode(&[seed_token]).chars().take(16).collect::<String>());

    // ── Spec-decode loop ───────────────────────────────────────────────
    //
    // Convention: cycle's `cur_pos` = position where last_committed lives.
    // For cycle 0, last_committed = seed_token at position `prompt_tokens.len()`.
    // We emit `seed_token` to the output stream first, then on each cycle
    // append `result.committed` (which excludes the seed but includes any
    // newly accepted MTP candidates and the bonus).
    let mut emitted: Vec<u32> = Vec::with_capacity(max_tokens + max_n + 1);
    emitted.push(seed_token);

    let mut last_committed = seed_token;
    let mut cur_pos = prompt_tokens.len();

    let mut cycles = 0usize;
    let mut accepted_total = 0usize;  // sum of accept_count across cycles
    let mut bonus_total = 0usize;     // sum of "bonus committed" across cycles
    let mut truncated_cycles = 0usize;  // cycles where p_min fired early-exit
    let mut drafts_generated_total = 0usize;  // sum of drafts_generated across cycles
    let mut replay_skipped_cycles = 0usize;  // full-accept cycles that skipped replay

    let t_decode = Instant::now();
    let mut hit_eos = tokenizer.is_terminator(seed_token);
    while !hit_eos && emitted.len() < max_tokens {
        // Bound check: cur_pos + max_n + 1 must fit in max_seq.
        if cur_pos + max_n + 1 >= max_seq_total {
            eprintln!("hit max_seq {}; stopping", max_seq_total);
            break;
        }
        let result = if compressed_serial {
            mtp_spec::spec_step_mtp_compressed_serial(
                &mut gpu, &mut target, &head, &mut state,
                cur_pos, last_committed, eos_token,
            ).expect("spec_step_mtp_compressed_serial")
        } else if compressed {
            mtp_spec::spec_step_mtp_compressed(
                &mut gpu, &mut target, &head, &mut state,
                cur_pos, last_committed, eos_token,
            ).expect("spec_step_mtp_compressed")
        } else {
            mtp_spec::spec_step_mtp(
                &mut gpu, &mut target, &head, &mut state,
                cur_pos, last_committed, eos_token,
            ).expect("spec_step_mtp")
        };

        cycles += 1;
        accepted_total += result.accept_count;
        drafts_generated_total += result.drafts_generated;
        if result.chain_truncated { truncated_cycles += 1; }
        if result.replay_skipped { replay_skipped_cycles += 1; }
        if !result.hit_eos || (result.committed.last().copied() != Some(eos_token)
                               && result.accept_count < max_n) {
            // bonus committed unless we EOS-broke inside the chain. Counts
            // the explicit bonus argmax slot.
            bonus_total += 1;
        }

        for &t in &result.committed {
            emitted.push(t);
        }
        last_committed = *result.committed.last().expect("non-empty commit");
        cur_pos += result.advance;

        if result.hit_eos {
            hit_eos = true;
            break;
        }
        if emitted.len() >= max_tokens {
            break;
        }
    }
    let decode_secs = t_decode.elapsed().as_secs_f64();

    let total_committed = emitted.len();
    let tok_per_s = total_committed as f64 / decode_secs.max(1e-9);

    // τ = average tokens committed per cycle (including the per-cycle bonus).
    // Real-decode MTP τ floor is 1.0 (always at least bonus); 1.0 means MTP
    // never accepted, so it's pure AR with overhead. > 1.0 means MTP is
    // contributing real speedup. The llama.cpp baseline we're measuring
    // against reports τ ≈ 2.5-3.0 on Qwen3 with max_n=3.
    //
    // Note: cycles count is "spec cycles", and per cycle we commit
    // (accept_count + 1) tokens (or fewer on early-EOS). The seed_token
    // contributes one extra to total_committed but it's NOT a cycle
    // commit — exclude it from τ.
    let tau = if cycles > 0 {
        ((total_committed - 1) as f64) / cycles as f64
    } else {
        0.0
    };

    let text = tokenizer.decode(&emitted);
    println!("\n=== output ===\n{text}\n=== end ===");
    println!();
    println!("prompt_md5:           {prompt_hash}");
    println!("prompt_tokens:        {}", prompt_tokens.len());
    println!("max_n:                {}", max_n);
    println!("cycles:               {}", cycles);
    {
        let skip_pct = 100.0 * replay_skipped_cycles as f64 / cycles.max(1) as f64;
        println!("replay_skipped:       {replay_skipped_cycles} cycles ({skip_pct:.1}%)");
    }
    if p_min > 0.0 {
        let truncated_pct = 100.0 * truncated_cycles as f64 / cycles.max(1) as f64;
        let avg_drafts = drafts_generated_total as f64 / cycles.max(1) as f64;
        println!("p_min:                {}", p_min);
        println!("chain_truncated:      {} cycles ({:.1}%)", truncated_cycles, truncated_pct);
        println!("avg_drafts_per_cycle: {:.3} (of max_n={})", avg_drafts, max_n);
    }
    println!("committed_total:      {}", total_committed);
    println!("committed_seed:       1");
    println!("committed_per_cycle_avg: {:.4}", tau);
    println!("accepted_mtp_total:   {}", accepted_total);
    println!("bonus_total:          {}", bonus_total);
    println!("tau:                  {:.4}", tau);
    println!("prefill_secs:         {:.3}", prefill_secs);
    println!("prefill_tok_s:        {:.2}", prefill_tok_s);
    println!("decode_secs:          {:.3}", decode_secs);
    println!("tok_s:                {:.2}", tok_per_s);
    println!("eos_hit:              {}", if hit_eos { "y" } else { "n" });

    // First 200 chars of output for visual coherence check.
    let preview: String = text.chars().take(200).collect();
    println!("preview_200:          {:?}", preview);

    state.free_gpu(&mut gpu);
    head.free_gpu(&mut gpu);
}
