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
    // Goose bypass-mode PLD spine (Jin et al. 2026, arXiv:2604.02047).
    // When enabled, each cycle checks the last-N-of-context for an earlier
    // occurrence in context; on match, its continuation is used as the
    // draft spine instead of the DFlash forward pass (cheaper + higher
    // acceptance on repetition-heavy content). No kernel work; hybrid-arch
    // safe (pure linear verify, no tree state forking).
    let mut pld_enabled: bool = false;
    let mut pld_min_extract: usize = 3;  // matcher floor; ≥3 tokens to record a match
    let mut pld_max_extract: usize = 8;  // paper cap
    let mut pld_ngrams: Vec<usize> = vec![5, 4, 3];  // paper defaults
    // Goose §4.3 bypass-mode confidence gate: only USE a PLD match when
    // it's confident enough to beat DFlash. Paper uses consensus ≥ 2
    // (at least two n-gram lengths agree on first token) and chain length
    // ≥ 8. 0 disables the gate (use every matcher hit — useful for
    // diagnostics; usually a net loss on content where DFlash is strong).
    let mut pld_min_consensus: usize = 2;
    let mut pld_min_chain: usize = 5;  // conservative: below paper's 8 but still filters noise
    // DDTree (Ringel & Romano 2026): tree-structured verification built from
    // DFlash per-position draft marginals. Per-path DFS verify (no batched
    // tree attention) — slower per cycle but correct on hybrid arch. Spike
    // measurement: does τ improve with the tree structure?
    let mut ddtree_enabled: bool = false;
    let mut ddtree_budget: usize = 16;  // paper uses 60; cheaper spike default
    let mut ddtree_topk: usize = 8;     // paper uses B-1 * budget_fanout; small k keeps tree shallow
    // --ddtree-batched: use spec_step_ddtree_batched (single tree-attention
    // forward) instead of the per-path DFS. Requires FA batched path (Q8 /
    // asym3 / asym4 KV). Tree-exact on FA side, linear-replay on GDN.
    let mut ddtree_batched: bool = false;

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
            "--pld" => {
                pld_enabled = true;
                i += 1;
            }
            "--pld-min" => {
                pld_min_extract = args[i + 1].parse().unwrap();
                i += 2;
            }
            "--pld-max" => {
                pld_max_extract = args[i + 1].parse().unwrap();
                i += 2;
            }
            "--pld-ngrams" => {
                pld_ngrams = args[i + 1]
                    .split(',')
                    .map(|s| s.trim().parse::<usize>().expect("--pld-ngrams: comma-separated positive ints"))
                    .collect();
                // Sort descending — longest-first is required by the matcher.
                pld_ngrams.sort_by(|a, b| b.cmp(a));
                i += 2;
            }
            "--pld-min-consensus" => {
                pld_min_consensus = args[i + 1].parse().unwrap();
                i += 2;
            }
            "--pld-min-chain" => {
                pld_min_chain = args[i + 1].parse().unwrap();
                i += 2;
            }
            "--ddtree" => {
                ddtree_enabled = true;
                i += 1;
            }
            "--ddtree-budget" => {
                ddtree_budget = args[i + 1].parse().unwrap();
                i += 2;
            }
            "--ddtree-topk" => {
                ddtree_topk = args[i + 1].parse().unwrap();
                i += 2;
            }
            "--ddtree-batched" => {
                ddtree_batched = true;
                ddtree_enabled = true; // implies --ddtree
                i += 1;
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
    // DDTree needs a SECOND snapshot for the post-seed branch point (shared
    // across all DFS paths in a cycle). Allocate unconditionally — a single
    // DeltaNetSnapshot is cheap (~100 MB on 9B) and unused if --ddtree is off.
    let mut post_seed_snap = DeltaNetSnapshot::new_for(&mut gpu, &target.dn_state).expect("post-seed snap");
    // GdnTape: per-LA-layer (q, k, v, α, β) innovation tape — sized for B
    // positions, allocated once and reused every spec step. Enables the
    // rollback path to replay GDN recurrence without re-running the target.
    //
    // Tree verify extends the block size: `1 + tree_budget` rows per forward
    // (seed + tree nodes). Size max_n = max(block_size, 1 + tree_budget) so
    // the tape is large enough whether we run per-path DFS, batched tree,
    // or plain DFlash.
    let tape_max_n = draft_cfg.block_size.max(1 + ddtree_budget);
    let mut gdn_tape = engine::speculative::GdnTape::new_for_config(
        &mut gpu, &target.config, tape_max_n,
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
    // PLD matcher: stateless, scans (prompt ++ emitted) suffix each cycle.
    let pld_matcher = if pld_enabled {
        let m = engine::speculative::PldMatcher {
            ngram_lens: pld_ngrams.clone(),
            max_extract: pld_max_extract,
            min_extract: pld_min_extract,
        };
        eprintln!(
            "pld: enabled (ngrams={:?}, min_extract={}, max_extract={})",
            m.ngram_lens, m.min_extract, m.max_extract
        );
        Some(m)
    } else {
        None
    };
    // PLD stats: hits = cycles where a spine was substituted for DFlash;
    // accepted_from_pld = accepted count on those cycles (for τ_pld).
    let mut pld_hits: usize = 0;
    let mut pld_accepted: usize = 0;

    if ddtree_enabled {
        if temp > 0.0 {
            eprintln!(
                "WARNING: --ddtree with temp>0 falls back to greedy on the verify side for \
                this spike (rejection-sampling integration is deferred)."
            );
        }
        if pld_enabled {
            eprintln!("WARNING: --pld is ignored when --ddtree is enabled.");
        }
        if ddtree_batched {
            eprintln!(
                "ddtree: enabled (budget={}, topk={}; BATCHED tree verify via FA tree-attention mask + GDN linear replay)",
                ddtree_budget, ddtree_topk,
            );
        } else {
            eprintln!(
                "ddtree: enabled (budget={}, topk={}; per-path DFS verify, ~{}× DFlash per-cycle cost)",
                ddtree_budget,
                ddtree_topk,
                ddtree_budget / (draft_cfg.block_size.saturating_sub(1).max(1)),
            );
        }
    }

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
        // PLD lookup: context = prompt ++ emitted (everything committed so
        // far). The matcher finds a suffix self-match and extracts up to
        // pld_max_extract continuation tokens. `pld_spine` is passed as a
        // borrowed slice — when Some, spec_step_dflash bypasses the
        // DFlash forward entirely for this cycle.
        let pld_match = pld_matcher.as_ref().and_then(|m| {
            // Build context = prompt ++ emitted ++ seed_token, making sure
            // the context suffix ENDS at seed_token — the matcher predicts
            // what follows the suffix, and block[1..] lives right after
            // seed_token. At cycle K≥1, emitted[-1] is already seed_token
            // (pushed as the prior cycle's bonus) so we skip the extra push;
            // at cycle 0 (emitted empty) we need to append it explicitly.
            let mut ctx = Vec::with_capacity(prompt_tokens.len() + emitted.len() + 1);
            ctx.extend_from_slice(&prompt_tokens);
            ctx.extend_from_slice(&emitted);
            if ctx.last() != Some(&seed_token) {
                ctx.push(seed_token);
            }
            m.lookup(&ctx)
        });
        // Goose §4.3 bypass-mode gate: only use PLD if both consensus AND
        // chain length clear their thresholds. Weaker matches are a net loss
        // when DFlash is strong (repetition-heavy content where literal
        // 3-gram matches predict the wrong number/variable in a list).
        let pld_spine: Option<&[u32]> = pld_match.as_ref().and_then(|m| {
            if m.consensus >= pld_min_consensus && m.tokens.len() >= pld_min_chain {
                Some(m.tokens.as_slice())
            } else {
                None
            }
        });
        let used_pld = pld_spine.is_some();
        if used_pld {
            pld_hits += 1;
        }
        let step = if ddtree_enabled {
            let step_fn = if ddtree_batched {
                speculative::spec_step_ddtree_batched
            } else {
                speculative::spec_step_ddtree
            };
            step_fn(
                &mut gpu,
                &mut target,
                &draft_weights,
                &draft_cfg,
                &mut draft_scratch,
                &mut hidden_rb,
                &mut target_hidden_host,
                &mut target_snap,
                &mut post_seed_snap,
                &mut gdn_tape,
                position,
                seed_token,
                ctx_slice,
                ddtree_budget,
                ddtree_topk,
            )
            .expect("ddtree spec step")
        } else {
            speculative::spec_step_dflash(
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
                pld_spine,
            )
            .expect("spec step")
        };
        if used_pld {
            pld_accepted += step.accepted;
        }

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
    if pld_matcher.is_some() {
        let hit_rate = if stats.cycles > 0 {
            pld_hits as f32 / stats.cycles as f32
        } else {
            0.0
        };
        let tau_pld = if pld_hits > 0 {
            pld_accepted as f32 / pld_hits as f32
        } else {
            0.0
        };
        let tau_dflash = if stats.cycles > pld_hits {
            (stats.accepted_tokens - pld_accepted) as f32
                / (stats.cycles - pld_hits) as f32
        } else {
            0.0
        };
        eprintln!(
            "pld: hits={}/{} ({:.1}%)  τ_pld={:.3}  τ_dflash={:.3}",
            pld_hits,
            stats.cycles,
            hit_rate * 100.0,
            tau_pld,
            tau_dflash,
        );
    }
}
