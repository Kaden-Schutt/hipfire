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
            "-h" | "--help" => {
                eprintln!(
                    "Usage: mtp_only_demo --target <trunk.hfq> --mtp-head <head.mtp> \\\n\
                     \t(--prompt \"Hello\" | --prompt-file <path>) \\\n\
                     \t[--max 64] [--ctx 4096] [--temp 0.0] [--max-n 3] [--no-chatml]"
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
        eprintln!("error: mtp_only_demo v1 is greedy-only (--temp must be 0.0); got {temp}");
        std::process::exit(2);
    }
    assert!(max_n >= 1 && max_n <= 8, "--max-n must be in [1,8]");

    let prompt = hipfire_runtime::tokenizer::maybe_normalize_prompt(&prompt_raw).into_owned();
    let prompt_hash = prompt_md5(prompt.as_bytes());

    eprintln!("=== mtp_only_demo ===");
    eprintln!("target:     {target_path}");
    eprintln!("mtp-head:   {mtp_path}");
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
    let head = mtp_head::load_mtp_head(
        Path::new(&mtp_path), &mut gpu, max_seq_total,
    ).expect("load mtp head");
    eprintln!("mtp head loaded in {:.2}s — n_embd={} vocab={} n_rot={} rope_theta={}",
              t_mtp.elapsed().as_secs_f64(),
              head.config.n_embd, head.config.vocab_size,
              head.config.n_rot, head.config.rope_theta);

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
    let mut state = MtpSpecState::new_for_slot(&mut gpu, &target, &head, max_n)
        .expect("alloc MtpSpecState");

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

    let t_decode = Instant::now();
    let mut hit_eos = tokenizer.is_terminator(seed_token);
    while !hit_eos && emitted.len() < max_tokens {
        // Bound check: cur_pos + max_n + 1 must fit in max_seq.
        if cur_pos + max_n + 1 >= max_seq_total {
            eprintln!("hit max_seq {}; stopping", max_seq_total);
            break;
        }
        let result = mtp_spec::spec_step_mtp(
            &mut gpu, &mut target, &head, &mut state,
            cur_pos, last_committed, eos_token,
        ).expect("spec_step_mtp");

        cycles += 1;
        accepted_total += result.accept_count;
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
