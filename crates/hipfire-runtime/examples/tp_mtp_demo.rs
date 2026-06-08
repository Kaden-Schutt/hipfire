//! tp_mtp_demo: end-to-end TENSOR-PARALLEL MTP spec-decode harness.
//!
//! The TP analogue of `mtp_only_demo`: loads a DENSE Qwen3.5/3.6 trunk sharded
//! across `--tp` GPUs (tensor-parallel, via `forward_prefill_chunk_tp`) + a
//! native MTP head (.mtp) resident on rank 0, prefills the prompt, then loops
//! `mtp_spec::spec_step_mtp_tp` until N tokens committed or EOS. Prints τ +
//! tok/s + prompt md5 + decoded output for coherence + the first real TP+MTP
//! end-to-end number (the trunk-cost bench `tp_mtp_cost` only measured the
//! verify+replay forwards with a synthetic draft).
//!
//! Greedy only (temp 0). Dense models only (TP forward is dense-only).
//!
//! Usage:
//!   tp_mtp_demo --target <dense.mq4> --mtp-head <head.mtp> \
//!               (--prompt "Hello" | --prompt-file <path>) \
//!               [--tp 4] [--max 128] [--ctx 4096] [--max-n 4] [--no-chatml]
//!               [--kv-mode q8]

#[cfg(not(feature = "deltanet"))]
fn main() {
    eprintln!("build with --features deltanet,arch-qwen35");
}

#[cfg(feature = "deltanet")]
fn main() {
    use hipfire_arch_qwen35::mtp_head::{self, MtpKvMode};
    use hipfire_arch_qwen35::mtp_spec::{self, MtpSpecState};
    use hipfire_arch_qwen35::qwen35::{self, DeltaNetState, Qwen35Scratch, StateQuant};
    use hipfire_arch_qwen35::speculative::DeltaNetSnapshot;
    use hipfire_detect::report::prompt_md5;
    use hipfire_runtime::hfq::HfqFile;
    use hipfire_runtime::llama::KvCache;
    use hipfire_runtime::multi_gpu::Gpus;
    use hipfire_runtime::tokenizer::Tokenizer;
    use hipfire_runtime::tp_shard::{ExpertAssign, ShardConfig};
    use rdna_compute::DType;
    use std::path::Path;
    use std::time::Instant;

    // ── Parse args ─────────────────────────────────────────────────────
    let args: Vec<String> = std::env::args().collect();
    let mut target_path: Option<String> = None;
    let mut mtp_path: Option<String> = None;
    let mut prompt_str: Option<String> = None;
    let mut prompt_file: Option<String> = None;
    let mut tp: usize = 4;
    let mut max_tokens: usize = 128;
    let mut ctx_capacity: usize = 4096;
    let mut max_n: usize = 4;
    let mut chatml: bool = true;
    let mut kv_mode_str: String = String::from("q8");

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--target" => { target_path = Some(args[i + 1].clone()); i += 2; }
            "--mtp-head" => { mtp_path = Some(args[i + 1].clone()); i += 2; }
            "--prompt" => { prompt_str = Some(args[i + 1].clone()); i += 2; }
            "--prompt-file" => { prompt_file = Some(args[i + 1].clone()); i += 2; }
            "--tp" => { tp = args[i + 1].parse().unwrap(); i += 2; }
            "--max" => { max_tokens = args[i + 1].parse().unwrap(); i += 2; }
            "--ctx" => { ctx_capacity = args[i + 1].parse().unwrap(); i += 2; }
            "--max-n" => { max_n = args[i + 1].parse().unwrap(); i += 2; }
            "--no-chatml" => { chatml = false; i += 1; }
            "--chatml" => { chatml = true; i += 1; }
            "--kv-mode" => { kv_mode_str = args[i + 1].clone(); i += 2; }
            "-h" | "--help" => {
                eprintln!(
                    "Usage: tp_mtp_demo --target <dense.mq4> --mtp-head <head.mtp> \\\n\
                     \t(--prompt \"Hello\" | --prompt-file <path>) \\\n\
                     \t[--tp 4] [--max 128] [--ctx 4096] [--max-n 4] [--no-chatml] [--kv-mode q8]"
                );
                std::process::exit(0);
            }
            other => { eprintln!("unknown arg: {other}"); std::process::exit(2); }
        }
    }

    let target_path = target_path.expect("--target required");
    let mtp_path = mtp_path.expect("--mtp-head required (dense .mtp sidecar)");
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
    assert!(max_n >= 1 && max_n <= 8, "--max-n must be in [1,8]");
    assert!(tp >= 1 && tp <= 8, "--tp must be in [1,8]");

    let prompt = hipfire_runtime::tokenizer::maybe_normalize_prompt(&prompt_raw).into_owned();
    let prompt_hash = prompt_md5(prompt.as_bytes());

    eprintln!("=== tp_mtp_demo ===");
    eprintln!("target:     {target_path}");
    eprintln!("mtp-head:   {mtp_path}");
    eprintln!("prompt md5: {prompt_hash}");
    eprintln!("tp={tp} max={max_tokens} ctx={ctx_capacity} max_n={max_n} chatml={chatml} kv-mode={kv_mode_str}");

    // ── Open trunk + config + tokenizer ────────────────────────────────
    let mut hfq = HfqFile::open(Path::new(&target_path)).expect("open hfq");
    let config = qwen35::config_from_hfq(&hfq).expect("config");
    assert_eq!(
        config.num_experts, 0,
        "tp_mtp_demo: TP forward is dense-only (got num_experts={})",
        config.num_experts
    );
    assert_eq!(
        config.n_kv_heads % tp, 0,
        "n_kv_heads {} not divisible by tp {}", config.n_kv_heads, tp
    );
    let dim = config.dim;
    let eos_token = config.eos_token;
    let tokenizer: Tokenizer =
        Tokenizer::from_hfq_metadata(&hfq.metadata_json).expect("tokenizer");

    // ── Tokenize prompt (+ optional chatml wrap) ───────────────────────
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

    let max_seq_total = ctx_capacity + max_tokens * (max_n + 1) + 16;
    assert!(
        prompt_tokens.len() + max_tokens * (max_n + 1) + 16 <= max_seq_total,
        "prompt won't fit in max_seq"
    );

    // ── TP topology + per-rank trunk resources ─────────────────────────
    let shard = ShardConfig::new(tp, false, config.num_experts, ExpertAssign::Stride).unwrap();
    shard.validate(config.n_heads, config.n_kv_heads).unwrap();
    let mut gpus = Gpus::init_tp(tp, config.n_layers).expect("init_tp");
    for dev in gpus.devices.iter_mut() {
        dev.bind_thread().unwrap();
        let st = dev.hip.stream_create().unwrap();
        dev.active_stream = Some(st);
    }
    eprintln!("tp init: {} devices", gpus.devices.len());

    let configs: Vec<qwen35::Qwen35Config> = (0..tp)
        .map(|_| if tp == 1 { config.clone() } else { qwen35::local_attn_config(&config, &shard) })
        .collect();

    // prefill chunk size: bounds the one-time prompt prefill batch as well as
    // the per-cycle verify (max_n+1) / replay (≤ max_n+1).
    const PREFILL_CHUNK: usize = 128;
    let pb_batch = PREFILL_CHUNK.max(max_n + 1).max(16);

    let t_load = Instant::now();
    let mut weights = Vec::with_capacity(tp);
    let mut scratches = Vec::with_capacity(tp);
    let mut kvs = Vec::with_capacity(tp);
    let mut dns = Vec::with_capacity(tp);
    let mut pbs_vec = Vec::with_capacity(tp);
    let mut partials = Vec::with_capacity(tp);
    let mut snaps = Vec::with_capacity(tp);
    for r in 0..tp {
        gpus.devices[r].bind_thread().unwrap();
        weights.push(if tp == 1 {
            qwen35::load_weights(&mut hfq, &config, &mut gpus.devices[r]).unwrap()
        } else {
            qwen35::load_weights_tp(&mut hfq, &config, &mut gpus.devices[r], &shard, r).unwrap()
        });
        scratches.push(Qwen35Scratch::new(&mut gpus.devices[r], &configs[r], 128).unwrap());
        kvs.push(
            KvCache::new_gpu_q8(
                &mut gpus.devices[r],
                configs[r].n_layers,
                configs[r].n_kv_heads,
                configs[r].head_dim,
                max_seq_total,
            )
            .unwrap(),
        );
        dns.push(DeltaNetState::new_with_quant(&mut gpus.devices[r], &configs[r], StateQuant::Q8).unwrap());
        pbs_vec.push(qwen35::PrefillBatchScratch::new(&mut gpus.devices[r], &configs[r], pb_batch).unwrap());
        partials.push(gpus.devices[r].zeros(&[pb_batch * dim], DType::F32).unwrap());
        snaps.push(DeltaNetSnapshot::new_for(&mut gpus.devices[r], &dns[r]).unwrap());
    }
    eprintln!("trunk loaded ({tp} ranks) in {:.2}s", t_load.elapsed().as_secs_f64());

    // ── MTP head on rank 0 ─────────────────────────────────────────────
    gpus.devices[0].bind_thread().unwrap();
    let t_mtp = Instant::now();
    let head = mtp_head::load_mtp_head(Path::new(&mtp_path), &mut gpus.devices[0], max_seq_total)
        .expect("load mtp head");
    eprintln!(
        "mtp head loaded in {:.2}s — n_embd={} vocab={}",
        t_mtp.elapsed().as_secs_f64(),
        head.config.n_embd,
        head.config.vocab_size
    );
    assert_eq!(head.config.n_embd, dim, "trunk/head dim mismatch");
    assert_eq!(head.config.vocab_size, config.vocab_size, "trunk/head vocab mismatch");

    // ── Spec state (dev0 draft+verify scratch) ─────────────────────────
    let kv_mode = MtpKvMode::parse(&kv_mode_str).unwrap_or_else(|e| {
        eprintln!("{e}");
        std::process::exit(2);
    });
    gpus.devices[0].bind_thread().unwrap();
    let mut state = MtpSpecState::new_for_config_with_kv_mode(
        &mut gpus.devices[0], &configs[0], &dns[0], &head, max_n, kv_mode,
    )
    .expect("alloc MtpSpecState");

    // ── Prefill (chunked TP forward, capturing nothing — None path leaves
    //    scratches[0].tmp = last-pos hidden, scratches[0].logits = seed) ──
    eprintln!("prefilling {} tokens (TP, chunk={PREFILL_CHUNK})...", prompt_tokens.len());
    let t_prefill = Instant::now();
    let mut start = 0usize;
    while start < prompt_tokens.len() {
        let end = (start + pb_batch).min(prompt_tokens.len());
        qwen35::forward_prefill_chunk_tp(
            &mut gpus, &shard, &weights, &configs, &prompt_tokens[start..end], start,
            &mut kvs, &mut dns, &pbs_vec, &partials, &scratches, None,
        )
        .expect("prefill forward_prefill_chunk_tp");
        start = end;
    }
    let prefill_secs = t_prefill.elapsed().as_secs_f64();
    let prefill_tok_s = prompt_tokens.len() as f64 / prefill_secs.max(1e-9);
    eprintln!("prefill: {:.2}s ({:.1} tok/s)", prefill_secs, prefill_tok_s);

    // Seed: capture prev_hidden + greedy seed token from the last prefill pos.
    gpus.devices[0].bind_thread().unwrap();
    state
        .capture_prev_hidden_from_scratch_tmp(&gpus.devices[0], &scratches[0].tmp, dim)
        .expect("capture prev_hidden");
    let logits0 = gpus.devices[0].download_f32(&scratches[0].logits).expect("seed logits");
    let mut seed_token = 0u32;
    let mut best = f32::NEG_INFINITY;
    for (i, &v) in logits0.iter().enumerate() {
        if v > best {
            best = v;
            seed_token = i as u32;
        }
    }
    eprintln!(
        "seed token: {} ('{}')",
        seed_token,
        tokenizer.decode(&[seed_token]).chars().take(16).collect::<String>()
    );

    // ── TP MTP spec-decode loop ────────────────────────────────────────
    let mut emitted: Vec<u32> = Vec::with_capacity(max_tokens + max_n + 1);
    emitted.push(seed_token);
    let mut last_committed = seed_token;
    let mut cur_pos = prompt_tokens.len();
    let mut cycles = 0usize;
    let mut accepted_total = 0usize;

    let t_decode = Instant::now();
    let mut hit_eos = tokenizer.is_terminator(seed_token);
    while !hit_eos && emitted.len() < max_tokens {
        if cur_pos + max_n + 1 >= max_seq_total {
            eprintln!("hit max_seq {}; stopping", max_seq_total);
            break;
        }
        let result = mtp_spec::spec_step_mtp_tp(
            &mut gpus, &shard, &weights, &configs, &mut kvs, &mut dns,
            &pbs_vec, &partials, &scratches, &mut snaps, &head, &mut state,
            cur_pos, last_committed, eos_token,
        )
        .expect("spec_step_mtp_tp");

        cycles += 1;
        accepted_total += result.accept_count;
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
    let tau = if cycles > 0 {
        ((total_committed - 1) as f64) / cycles as f64
    } else {
        0.0
    };
    let accept_rate = if cycles > 0 {
        accepted_total as f64 / cycles as f64
    } else {
        0.0
    };

    let text = tokenizer.decode(&emitted);
    println!("\n=== output ===\n{text}\n=== end ===");
    println!();
    println!("tp:                   {tp}");
    println!("prompt_md5:           {prompt_hash}");
    println!("prompt_tokens:        {}", prompt_tokens.len());
    println!("max_n:                {max_n}");
    println!("cycles:               {cycles}");
    println!("committed_total:      {total_committed}");
    println!("accepted_mtp_total:   {accepted_total}");
    println!("accept_rate_per_cyc:  {accept_rate:.4}");
    println!("tau:                  {tau:.4}");
    println!("prefill_secs:         {prefill_secs:.3}");
    println!("prefill_tok_s:        {prefill_tok_s:.2}");
    println!("decode_secs:          {decode_secs:.3}");
    println!("tok_s:                {tok_per_s:.2}");
    println!("eos_hit:              {}", if hit_eos { "y" } else { "n" });
    let preview: String = text.chars().take(200).collect();
    println!("preview_200:          {preview:?}");

    state.free_gpu(&mut gpus.devices[0]);
}
