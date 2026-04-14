//! dflash_spec_demo: end-to-end speculative decoding demo.
//!
//! Loads a Qwen3.5 target (.hfq) + a matching DFlash draft (.hfq, arch=20),
//! tokenizes a prompt, seeds target_hidden, and runs
//! `spec_step_dflash` in a loop until N tokens committed or an EOS is hit.
//! Prints tokens as they commit, plus final stats (accept rate, tok/s).
//!
//! Usage:
//!   dflash_spec_demo --target <target.hfq> --draft <draft.hfq> \
//!                    --prompt "Hello" [--max 64] [--ctx 512] [--ctx-slice N]
//!
//! --ctx-slice N: for accept-rate bisect only. Restricts the draft's
//! context view to the last N positions (instead of the full accumulated
//! history). Useful if the draft was trained on shorter contexts than the
//! prompt+decode length we're handing it at inference.

#[cfg(not(feature = "deltanet"))]
fn main() {
    eprintln!("build with --features deltanet");
}

#[cfg(feature = "deltanet")]
fn main() {
    use engine::dflash::{DflashConfig, DflashScratch, DflashWeights};
    use engine::hfq::HfqFile;
    use engine::speculative::{
        self, DeltaNetSnapshot, HiddenStateRingBuffer, ModelSlot, ModelSlotConfig, SpecStats,
    };
    use engine::tokenizer::Tokenizer;
    use std::path::Path;
    use std::time::Instant;

    // ── Parse args ─────────────────────────────────────────────────────
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!(
            "Usage: dflash_spec_demo --target <target.hfq> --draft <draft.hfq> \\\n                             --prompt \"Hello\" [--max 64] [--ctx 512]"
        );
        std::process::exit(1);
    }
    let mut target_path: Option<String> = None;
    let mut draft_path: Option<String> = None;
    let mut prompt: Option<String> = None;
    let mut max_tokens: usize = 64;
    let mut ctx_capacity: usize = 512;
    let mut ctx_slice: Option<usize> = None;
    let mut kv_mode_str = String::from("q8");
    let mut block_size_override: Option<usize> = None;
    let mut temp: f32 = 0.0;
    let mut seed: u64 = 42;
    let mut adaptive_b: bool = false;
    let mut ngram: bool = false;
    let mut ngram_min_count: u32 = 3;
    // CACTUS bumped acceptance (Hao & Mou 2026). 0.0 = vanilla SpS;
    // paper's strongest setting is 1.0. Only affects temp > 0 runs.
    let mut cactus_delta: f32 = 0.0;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--target" => {
                target_path = Some(args[i + 1].clone());
                i += 2;
            }
            "--draft" => {
                draft_path = Some(args[i + 1].clone());
                i += 2;
            }
            "--prompt" => {
                prompt = Some(args[i + 1].clone());
                i += 2;
            }
            "--max" => {
                max_tokens = args[i + 1].parse().unwrap();
                i += 2;
            }
            "--ctx" => {
                ctx_capacity = args[i + 1].parse().unwrap();
                i += 2;
            }
            "--ctx-slice" => {
                ctx_slice = Some(args[i + 1].parse().unwrap());
                i += 2;
            }
            "--kv-mode" => {
                kv_mode_str = args[i + 1].clone();
                i += 2;
            }
            "--block-size" => {
                block_size_override = Some(args[i + 1].parse().unwrap());
                i += 2;
            }
            "--temp" => {
                temp = args[i + 1].parse().unwrap();
                i += 2;
            }
            "--seed" => {
                seed = args[i + 1].parse().unwrap();
                i += 2;
            }
            "--adaptive-b" => {
                adaptive_b = true;
                i += 1;
            }
            "--ngram" => {
                ngram = true;
                i += 1;
            }
            "--ngram-min" => {
                ngram_min_count = args[i + 1].parse().unwrap();
                i += 2;
            }
            "--cactus-delta" => {
                cactus_delta = args[i + 1].parse().unwrap();
                i += 2;
            }
            other => {
                eprintln!("unknown arg: {other}");
                std::process::exit(1);
            }
        }
    }
    let target_path = target_path.expect("--target required");
    let draft_path = draft_path.expect("--draft required");
    let prompt = prompt.expect("--prompt required");

    eprintln!("=== dflash_spec_demo ===");
    eprintln!("target: {target_path}");
    eprintln!("draft:  {draft_path}");
    if let Some(n) = ctx_slice {
        eprintln!("ctx_slice: last {n} positions only (bisect mode)");
    }

    // ── Init GPU ──────────────────────────────────────────────────────
    let mut gpu = rdna_compute::Gpu::init().expect("gpu init");
    eprintln!("gpu: {}", gpu.arch);

    // ── Load draft ────────────────────────────────────────────────────
    let draft_hfq = HfqFile::open(Path::new(&draft_path)).expect("open draft");
    let mut draft_cfg = DflashConfig::from_hfq(&draft_hfq).expect("parse DflashConfig");
    if let Some(b) = block_size_override {
        let orig = draft_cfg.block_size;
        draft_cfg.block_size = b;
        eprintln!("block_size override: {orig} -> {b} (draft was trained at {orig}; smaller B lowers per-iter cost but may reduce τ)");
    }
    eprintln!(
        "draft: layers={} hidden={} heads={} kv_heads={} block={} target_layers={:?}",
        draft_cfg.n_layers,
        draft_cfg.hidden,
        draft_cfg.n_heads,
        draft_cfg.n_kv_heads,
        draft_cfg.block_size,
        draft_cfg.target_layer_ids,
    );
    let t0 = Instant::now();
    let draft_weights = DflashWeights::load(&mut gpu, &draft_hfq, &draft_cfg).expect("load draft");
    eprintln!("draft loaded in {:.2}s", t0.elapsed().as_secs_f64());

    let mut draft_scratch = DflashScratch::new_with_mq(
        &mut gpu, &draft_cfg, draft_cfg.block_size, ctx_capacity, draft_weights.has_mq,
    ).expect("alloc draft scratch");
    if draft_weights.has_mq {
        eprintln!("draft: MQ4 weights detected, FWHT rotation scratch enabled");
    }

    // ── Load target ───────────────────────────────────────────────────
    let mut slot_cfg = ModelSlotConfig::default();
    slot_cfg.max_seq = ctx_capacity + draft_cfg.block_size + 16;
    slot_cfg.kv_mode = match kv_mode_str.as_str() {
        "q8" => engine::speculative::KvMode::Q8,
        "asym4" | "turbo4" => engine::speculative::KvMode::Asym4,
        "asym3" | "turbo3" | "turbo" => engine::speculative::KvMode::Asym3,
        "asym2" | "turbo2" => engine::speculative::KvMode::Asym2,
        other => {
            eprintln!("unknown --kv-mode: {other}. Valid: q8, asym4, asym3, asym2");
            std::process::exit(1);
        }
    };
    eprintln!("kv_mode: {:?}", slot_cfg.kv_mode);
    let t1 = Instant::now();
    let mut target =
        ModelSlot::load(&mut gpu, Path::new(&target_path), "target", slot_cfg).expect("load target");
    eprintln!("target loaded in {:.2}s", t1.elapsed().as_secs_f64());

    // ── Check vocab compatibility ─────────────────────────────────────
    assert_eq!(
        target.config.vocab_size, draft_cfg.vocab_size,
        "target vocab ({}) != draft vocab ({})",
        target.config.vocab_size, draft_cfg.vocab_size
    );

    let tokenizer: Tokenizer = target.load_tokenizer().expect("target tokenizer");
    let prompt_tokens = tokenizer.encode(&prompt);
    eprintln!("prompt: {:?}", prompt);
    eprintln!("prompt tokens ({}): {:?}", prompt_tokens.len(), prompt_tokens);

    // ── Hidden ring buffer + snapshot + target_hidden_host ────────────
    let mut hidden_rb = HiddenStateRingBuffer::new(
        &mut gpu,
        target.config.n_layers,
        draft_cfg.num_extract(),
        draft_cfg.hidden,
        ctx_capacity + draft_cfg.block_size,
    )
    .expect("alloc hidden_rb");

    let mut target_snap = DeltaNetSnapshot::new_for(&mut gpu, &target.dn_state).expect("snap");
    // GdnTape: per-LA-layer (q, k, v, α, β) innovation tape — sized for B
    // positions, allocated once and reused every spec step. Enables the
    // rollback path to replay GDN recurrence without re-running the target.
    let mut gdn_tape = engine::speculative::GdnTape::new_for_config(
        &mut gpu, &target.config, draft_cfg.block_size,
    ).expect("alloc gdn tape");
    let mut target_hidden_host: Vec<f32> =
        Vec::with_capacity(ctx_capacity * draft_cfg.num_extract() * draft_cfg.hidden);

    // ── Prefill: seed target_hidden via per-token forward_with_hidden ──
    eprintln!("seeding target_hidden from prompt ({} tokens)...", prompt_tokens.len());
    let t2 = Instant::now();
    speculative::seed_target_hidden_from_prompt(
        &mut gpu,
        &mut target,
        &mut hidden_rb,
        &mut target_hidden_host,
        &prompt_tokens,
    )
    .expect("seed target hidden");
    eprintln!("prefill (per-token) in {:.2}s", t2.elapsed().as_secs_f64());

    // ── Initial seed_token: target's greedy pick after prefill ───────
    // Target state is at position `prompt_len` after seed_target_hidden_from_prompt.
    // Its scratch.logits at this point corresponds to the LAST prompt token's output —
    // i.e., the prediction for position prompt_len. Argmax = first emitted token.
    let first_logits = gpu.download_f32(&target.scratch.logits).expect("download logits");
    let first_token = first_logits
        .iter()
        .enumerate()
        .fold((0u32, f32::NEG_INFINITY), |(best, bv), (i, &v)| {
            if v > bv {
                (i as u32, v)
            } else {
                (best, bv)
            }
        })
        .0;

    // ── Decode loop ───────────────────────────────────────────────────
    let mut emitted: Vec<u32> = vec![first_token];
    let mut position: usize = prompt_tokens.len();
    let mut seed_token: u32 = first_token;
    let mut stats = SpecStats::new(draft_cfg.block_size);
    let eos_id: u32 = tokenizer.eos_id;

    eprintln!("decoding (max {max_tokens} tokens, block_size {})...", draft_cfg.block_size);

    // Rolling τ window for live emit + future adaptive routing decisions.
    // τ_window[i] = accepted draft tokens in cycle i. Running mean over the
    // last N cycles is a good proxy for whether the draft is keeping up.
    const TAU_WINDOW: usize = 8;
    let mut accepts_window: std::collections::VecDeque<usize> =
        std::collections::VecDeque::with_capacity(TAU_WINDOW);
    let live_tau = std::env::var("DFLASH_LIVE_TAU").is_ok();

    let mut rng_state: u64 = seed | 1; // xorshift state must be non-zero
    if temp > 0.0 {
        eprintln!("temp sampling: T={temp}, seed={seed}");
        if cactus_delta > 0.0 {
            eprintln!(
                "cactus: δ={cactus_delta} (bumped acceptance γ* = min(q + √(2·δ·q·(1−q)), 1))"
            );
        }
    } else if cactus_delta > 0.0 {
        eprintln!("cactus_delta={cactus_delta} ignored at temp=0 (greedy path has no distribution)");
    }
    // N-gram cache: built incrementally from committed output each iter.
    // Seeded from the prompt so multi-turn repetitions in the prompt get
    // cached. min_count gates how aggressive overrides are.
    let mut ngram_cache = if ngram {
        let mut c = engine::speculative::NgramCache::new(ngram_min_count);
        c.observe_many(&prompt_tokens);
        eprintln!(
            "ngram cache: bigrams seeded from prompt, min_count={ngram_min_count}"
        );
        Some(c)
    } else {
        None
    };

    let t_decode = Instant::now();
    while emitted.len() < max_tokens {
        if position + draft_cfg.block_size >= ctx_capacity {
            eprintln!("hit ctx_capacity {}; stopping", ctx_capacity);
            break;
        }
        // Adaptive B: when rolling τ falls below threshold, shrink block_size
        // to 8 to cut per-iter cost. Raise back to 16 when τ recovers. Only
        // active when --adaptive-b is set.
        let block_override = if adaptive_b {
            if accepts_window.len() >= 4 {
                let win_tau: f64 =
                    accepts_window.iter().copied().sum::<usize>() as f64 / accepts_window.len() as f64;
                if win_tau < 4.0 { Some(8usize) } else { None }
            } else {
                None
            }
        } else {
            None
        };
        let step = speculative::spec_step_dflash(
            &mut gpu,
            &mut target,
            &draft_weights,
            &draft_cfg,
            &mut draft_scratch,
            &mut hidden_rb,
            &mut target_hidden_host,
            &mut target_snap,
            position,
            seed_token,
            ctx_slice,
            Some(&mut gdn_tape),
            temp,
            &mut rng_state,
            block_override,
            ngram_cache.as_ref(),
            &emitted,
            cactus_delta,
        )
        .expect("spec step");

        // Populate n-gram cache from newly committed tokens. `step.committed`
        // is [seed, accepted draft tokens, bonus]; we record all consecutive
        // triples within the committed span plus the join with prior context.
        if let Some(ref mut ng) = ngram_cache {
            // The 2 tokens right before step.committed[0] are the last 2 of
            // `emitted` (since seed_token == prev iter's bonus = last emitted).
            // Walk windows across (tail-2 of emitted ++ step.committed).
            let tail_len = emitted.len().min(2);
            let mut window: Vec<u32> = Vec::with_capacity(tail_len + step.committed.len());
            window.extend_from_slice(&emitted[emitted.len() - tail_len..]);
            window.extend_from_slice(&step.committed);
            ng.observe_many(&window);
        }
        stats.record(&step);

        // Rolling τ.
        if accepts_window.len() == TAU_WINDOW {
            accepts_window.pop_front();
        }
        accepts_window.push_back(step.accepted);
        if live_tau {
            let win_tau: f64 = accepts_window.iter().copied().sum::<usize>() as f64
                / accepts_window.len() as f64;
            let cum_tau: f64 = stats.accepted_tokens as f64 / stats.cycles as f64;
            eprintln!(
                "[cycle {:3}] accepted={:2} seed={:5} τ_win={:.2} τ_cum={:.2} position={}",
                stats.cycles, step.accepted, seed_token, win_tau, cum_tau, position,
            );
        }

        // `step.committed[0]` is the seed_token (already emitted). Emit [1..].
        for (&tok, _) in step.committed.iter().skip(1).zip(0..) {
            emitted.push(tok);
        }

        // Advance position + pick next seed (= bonus_token).
        position += step.accepted + 1;
        seed_token = step.bonus_token;

        // Stop on EOS.
        if step.committed.iter().skip(1).any(|&t| t == eos_id) {
            eprintln!("eos");
            break;
        }
    }
    let elapsed = t_decode.elapsed().as_secs_f64();
    let tok_s = emitted.len() as f64 / elapsed;

    // ── Report ────────────────────────────────────────────────────────
    let text = tokenizer.decode(&emitted);
    eprintln!("--- OUTPUT ---");
    println!("{text}");
    eprintln!("--------------");
    eprintln!(
        "emitted: {} tokens in {:.2}s  ({:.2} tok/s)",
        emitted.len(),
        elapsed,
        tok_s
    );
    eprintln!(
        "cycles: {}  committed: {}  accepted: {}  τ={:.3}  mean_committed={:.3}",
        stats.cycles,
        stats.committed_tokens,
        stats.accepted_tokens,
        stats.tau(),
        stats.mean_committed(),
    );
    let accept_rate = if stats.cycles > 0 {
        stats.accepted_tokens as f32 / (stats.cycles * (draft_cfg.block_size - 1)) as f32
    } else {
        0.0
    };
    eprintln!("accept_rate (accepted / (cycles × (B-1))): {accept_rate:.3}");
    eprintln!(
        "histogram: {:?}",
        stats.acceptance_hist.iter().enumerate().collect::<Vec<_>>()
    );
}
