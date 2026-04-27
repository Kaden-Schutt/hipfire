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
    use engine::cask::CaskCtx;
    use engine::dflash::{DflashConfig, DflashScratch, DflashWeights};
    use engine::hfq::HfqFile;
    use engine::qwen35::LayerType;
    use engine::speculative::{
        self, DeltaNetSnapshot, HiddenStateRingBuffer, ModelSlot, ModelSlotConfig, SpecStats,
    };
    use engine::tokenizer::Tokenizer;
    use engine::triattn::{EvictionCtx, EvictionResult, TriAttnCenters};
    use std::path::Path;
    use std::time::Instant;

    enum CaskPolicy { Plain(EvictionCtx), Cask(CaskCtx) }
    impl CaskPolicy {
        fn maybe_evict(&self, gpu: &mut rdna_compute::Gpu, kv: &mut engine::llama::KvCache, physical: usize)
            -> hip_bridge::HipResult<Option<EvictionResult>>
        {
            match self {
                CaskPolicy::Plain(c) => c.maybe_evict(gpu, kv, physical),
                CaskPolicy::Cask(c) => c.maybe_evict(gpu, kv, physical),
            }
        }
        fn eviction_count(&self) -> usize {
            match self {
                CaskPolicy::Plain(c) => c.eviction_count.get(),
                CaskPolicy::Cask(c) => c.eviction_count(),
            }
        }
    }

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
    let mut repeat_penalty: f32 = 1.0;
    let mut repeat_window: usize = 128;
    // Adaptive block size: on by default.
    //
    // 2026-04-16: initial two-level version shrank B from 16→8 when rolling
    // τ dropped below 4.
    //
    // 2026-04-24 (Task #93 Phase B fallback): replaced with a continuous
    // scheduler that adjusts B across the range [ADAPTIVE_B_MIN ..
    // ADAPTIVE_B_MAX] using
    // EWMA of accept_len with hysteresis + cooldown. Raises B when the
    // recent cycle is under-budgeted (draft accepting almost everything →
    // amortize verify over more positions). Drops B when draft is losing
    // early (small B cuts verify cost and lets τ recover). Override range
    // with --adaptive-b-range MIN:MAX.
    //
    // ADAPTIVE_B_MAX caps: scratches are pre-allocated to it, so setting a
    // larger max raises VRAM cost. Default 16 preserves the pre-Task-#93
    // behaviour (draft trained at block_size=16; larger B is OOD for the
    // draft's positional encoding and may degrade τ).
    let mut adaptive_b: bool = true;
    let mut adaptive_b_min: usize = 8;
    let mut adaptive_b_max: usize = 16;
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
    // ChatML wrapping: <|im_start|>user\n{p}<|im_end|>\n<|im_start|>assistant\n —
    // matches how the daemon / infer_qwen35 call the instruction-tuned Qwen3.5.
    // Default ON (2026-04-17): bare prompts send the model off-distribution.
    // Empirically the draft's acceptance rate on raw Qwen3.5 creative prompts
    // drops to τ<1.5 (vs τ≈5 with ChatML) — measured on 27B rivers-essay:
    // bare gives 20 tok/s, ChatML gives 40 tok/s with identical target, draft,
    // and kv_mode. The gap is the draft predicting a structured distribution
    // vs a garbled one. Opt out via --no-chatml for the diagnostic
    // "pure continuation" case.
    let mut chatml: bool = true;
    // --ar-baseline: skip DFlash entirely, greedy-decode via target only.
    // Diagnostic for comparing DFlash outputs against pure-AR on the
    // same tokenized prompt.
    let mut ar_baseline: bool = false;
    // --debug-cycle N: dump the seed/block/drafted/argmax_per_pos/accept
    // for the first N cycles to help diagnose divergence.
    let mut debug_cycles: usize = 0;
    // --no-tape: disable GdnTape capture so spec_step_dflash replays via
    // forward_prefill_batch on committed tokens (byte-exact vs AR when
    // combined with HIPFIRE_PREFILL_BATCHED=0).
    let mut no_tape: bool = false;

    // FlashCASK: TriAttention scoring + CASK core-aware m-folding merge
    // applied to target.kv_cache between spec_step cycles. Passes the
    // compact_offset math through target's forward pass automatically
    // (qwen35::forward_scratch already reads kv_cache.compact_offset for
    // RoPE phase). Only opt-in — keep spec demo unchanged by default.
    let mut cask_sidecar: Option<String> = None;
    let mut cask_budget: usize = 512;
    let mut cask_beta: usize = 128;
    let mut use_cask: bool = false;
    let mut cask_core_frac: f32 = 0.5;
    let mut cask_fold_m: usize = 2;

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
            "--repeat-penalty" => {
                repeat_penalty = args[i + 1].parse().unwrap();
                i += 2;
            }
            "--repeat-window" => {
                repeat_window = args[i + 1].parse().unwrap();
                i += 2;
            }
            "--adaptive-b" => {
                adaptive_b = true;
                i += 1;
            }
            "--no-adaptive-b" => {
                adaptive_b = false;
                i += 1;
            }
            "--adaptive-b-range" => {
                // Format: "MIN:MAX" e.g. "8:20". Both inclusive.
                let v = &args[i + 1];
                let (lo, hi) = v.split_once(':').unwrap_or_else(||
                    panic!("--adaptive-b-range expects MIN:MAX, got {v:?}"));
                adaptive_b_min = lo.parse().expect("--adaptive-b-range MIN");
                adaptive_b_max = hi.parse().expect("--adaptive-b-range MAX");
                assert!(adaptive_b_min >= 2 && adaptive_b_max >= adaptive_b_min,
                    "--adaptive-b-range invalid: {adaptive_b_min}..{adaptive_b_max}");
                i += 2;
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
            "--chatml" => {
                chatml = true;
                i += 1;
            }
            "--no-chatml" => {
                chatml = false;
                i += 1;
            }
            "--ar-baseline" => {
                ar_baseline = true;
                i += 1;
            }
            "--debug-cycle" => {
                debug_cycles = args[i + 1].parse().unwrap();
                i += 2;
            }
            "--no-tape" => {
                no_tape = true;
                i += 1;
            }
            "--cask-sidecar" => {
                cask_sidecar = Some(args[i + 1].clone());
                i += 2;
            }
            "--cask" => {
                use_cask = true;
                i += 1;
            }
            "--cask-budget" => {
                cask_budget = args[i + 1].parse().unwrap();
                i += 2;
            }
            "--cask-beta" => {
                cask_beta = args[i + 1].parse().unwrap();
                i += 2;
            }
            "--cask-core-frac" => {
                cask_core_frac = args[i + 1].parse().unwrap();
                i += 2;
            }
            "--cask-fold-m" => {
                cask_fold_m = args[i + 1].parse().unwrap();
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
    let prompt = engine::tokenizer::maybe_normalize_prompt(&prompt).into_owned();

    eprintln!("=== dflash_spec_demo ===");
    eprintln!("target: {target_path}");
    eprintln!("draft:  {draft_path}");
    if let Some(n) = ctx_slice {
        eprintln!("ctx_slice: last {n} positions only (bisect mode)");
    }

    // ── Init GPU ──────────────────────────────────────────────────────
    let mut gpu = rdna_compute::Gpu::init().expect("gpu init");
    eprintln!("gpu: {}", gpu.arch);
    let vram_report = |hip: &hip_bridge::HipRuntime, label: &str| {
        if let Ok((free, total)) = hip.get_vram_info() {
            let used_gb = (total - free) as f64 / 1e9;
            let free_gb = free as f64 / 1e9;
            eprintln!("VRAM @ {label}: used {used_gb:.2} GB, free {free_gb:.2} GB");
        }
    };
    vram_report(&gpu.hip, "init");

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
    // Load target first — its 15 GB of weights need contiguous VRAM.
    // Draft fits afterward because pool::alloc uses EXACT HIP allocation
    // (pool.rs::alloc), so the target's per-layer buckets don't pad up
    // to the next power of 2 and waste the room the draft needs.
    // Compute adaptive-B scratch ceiling early so KV + ring-buffer sizing
    // upstream of the draft-load accounts for the max possible B we'll use.
    let cfg_block_size_for_slot = draft_cfg.block_size.max(if adaptive_b { adaptive_b_max } else { 0 });
    let mut slot_cfg = ModelSlotConfig::default();
    slot_cfg.max_seq = ctx_capacity + cfg_block_size_for_slot + 16;
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
    vram_report(&gpu.hip, "after target load");

    let t0 = Instant::now();
    let draft_weights = DflashWeights::load(&mut gpu, &draft_hfq, &draft_cfg).expect("load draft");
    eprintln!("draft loaded in {:.2}s", t0.elapsed().as_secs_f64());
    vram_report(&gpu.hip, "after draft load");

    // Adaptive-B scratch sizing: the draft was trained at a specific
    // block_size; going past it is out-of-distribution for its positional
    // encoding. Measured on 27B MQ4 (2026-04-24, 3-run median) with range
    // 8:20:
    //   code: 161.5 → 113.7 tok/s (-30 %) as B grew to 17+, τ only
    //         dropped 8 % so the loss is dominated by verify cost × B
    //         at OOD positions, not by τ collapse.
    //   prose/instr: unchanged (B never grew past 10 on low-τ workloads).
    // → clamp adaptive_b_max to draft_cfg.block_size with a warning when
    // the user explicitly widens. Opt out via HIPFIRE_ADAPTIVE_B_UNSAFE=1
    // for experiments on a refit draft.
    let unsafe_adaptive = std::env::var("HIPFIRE_ADAPTIVE_B_UNSAFE").ok().as_deref() == Some("1");
    if adaptive_b && adaptive_b_max > draft_cfg.block_size && !unsafe_adaptive {
        eprintln!(
            "adaptive-b: WARN requested MAX={} > draft trained block_size={}; clamping to {} (past-trained B regresses code by ~30 %; set HIPFIRE_ADAPTIVE_B_UNSAFE=1 to override)",
            adaptive_b_max, draft_cfg.block_size, draft_cfg.block_size,
        );
        adaptive_b_max = draft_cfg.block_size;
    }
    // Scratches allocated for the *effective* max B we'll ever use. When
    // user overrides with UNSAFE, this is larger; normally this equals
    // draft_cfg.block_size.
    let draft_scratch_b = if adaptive_b {
        draft_cfg.block_size.max(adaptive_b_max)
    } else {
        draft_cfg.block_size
    };
    if draft_scratch_b > draft_cfg.block_size {
        eprintln!(
            "adaptive-b: pre-sizing draft scratch for B_MAX={} (trained at {}) [UNSAFE=on]",
            draft_scratch_b, draft_cfg.block_size,
        );
    }
    let mut draft_scratch = DflashScratch::new_with_mq(
        &mut gpu, &draft_cfg, draft_scratch_b, ctx_capacity, draft_weights.has_mq,
    ).expect("alloc draft scratch");
    if draft_weights.has_mq {
        eprintln!("draft: MQ4 weights detected, FWHT rotation scratch enabled");
    }

    // ── Check vocab compatibility ─────────────────────────────────────
    assert_eq!(
        target.config.vocab_size, draft_cfg.vocab_size,
        "target vocab ({}) != draft vocab ({})",
        target.config.vocab_size, draft_cfg.vocab_size
    );

    let tokenizer: Tokenizer = target.load_tokenizer().expect("target tokenizer");
    if std::env::var("HIPFIRE_PROMPT_TOKEN_HEAT").ok().as_deref() == Some("1") {
        tokenizer.dump_prompt_heat(&prompt);
    }
    let mut prompt_tokens = tokenizer.encode(&prompt);
    if chatml {
        // Match daemon.rs production path: <|im_start|>user\n{p}<|im_end|>\n<|im_start|>assistant\n
        // Do NOT pre-append `<think>\n` — Qwen3.5 opens a think block itself when
        // needed, and forcing it pushes open-ended prompts into runaway
        // chain-of-thought that loops (measured on rivers essay: baseline AR
        // decays into ".*Wait, I need to be careful.*" repeats after ~600 tokens).
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
        eprintln!("chatml wrapping enabled: prompt is {} tokens after wrap", prompt_tokens.len());
    }
    eprintln!("prompt: {:?}", prompt);
    eprintln!("prompt tokens ({}): {:?}", prompt_tokens.len(), prompt_tokens);

    // ── Hidden ring buffer + snapshot + target_hidden_host ────────────
    // Size for the max block we may use this session so adaptive-B-up
    // doesn't overflow.
    let hrb_max_block = draft_scratch_b;
    let mut hidden_rb = HiddenStateRingBuffer::new(
        &mut gpu,
        target.config.n_layers,
        draft_cfg.num_extract(),
        draft_cfg.hidden,
        ctx_capacity + hrb_max_block,
        engine::qwen35::PREFILL_MAX_BATCH.max(hrb_max_block),
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
    let tape_max_n = draft_scratch_b.max(1 + ddtree_budget);
    let mut gdn_tape = engine::speculative::GdnTape::new_for_config(
        &mut gpu, &target.config, tape_max_n,
    ).expect("alloc gdn tape");
    // DdtreeScratch: persistent attention-bias buffer for batched tree verify.
    // One allocation at startup (sized for max_budget), reused every cycle —
    // avoids the per-cycle malloc+htod+free churn that dominated early wall-
    // clock numbers. Also allocated for non-ddtree runs (cheap, small) so
    // callers can switch strategies at runtime without reinit.
    // KV-gather + tape-gather scratch are sized here too (slow-path-kill,
    // 2026-04-23). Widths come from the target config: FA K/V row byte
    // counts depend on n_kv_heads × head_dim × quant, and the GdnTape's
    // qkv_dim = 2*k_dim + v_dim on the LA side.
    let ddtree_qkv_dim = {
        let kd = target.config.linear_num_key_heads * target.config.linear_key_head_dim;
        let vd = target.config.linear_num_value_heads * target.config.linear_value_head_dim;
        kd * 2 + vd
    };
    let ddtree_n_fa_layers = target.config.layer_types.iter()
        .filter(|t| **t == engine::qwen35::LayerType::FullAttention)
        .count();
    let ddtree_scratch = engine::speculative::DdtreeScratch::new(
        &mut gpu,
        ddtree_budget,
        target.config.n_kv_heads,
        target.config.head_dim,
        ddtree_qkv_dim,
        ddtree_n_fa_layers,
    ).expect("alloc ddtree scratch");
    // VerifyScratch: persistent per-cycle tensors (final_hidden, logits,
    // rotation scratch, argmax buf). Sized to max_n = max(block_size,
    // 1 + ddtree_budget) to cover plain DFlash and DDTree. Drops ~8
    // hipMalloc/hipFree pairs per cycle (biggest is 16 MB logits buffer),
    // saving 0.5-1.5 ms/cycle.
    let verify_max_n = draft_scratch_b.max(1 + ddtree_budget);
    let verify_scratch = engine::speculative::VerifyScratch::with_prefill(
        &mut gpu,
        verify_max_n,
        target.config.dim,
        target.config.vocab_size,
        target.weights.output.k,
        &target.config,
    ).expect("alloc verify scratch");
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
    // Mirror the prompt rows from the hidden ring buffer straight into
    // draft_scratch.target_hidden on GPU. This primes the GPU-resident
    // path in spec_step_dflash (ctx_slice=None) so it doesn't need to
    // round-trip target_hidden through the CPU shadow each cycle.
    speculative::scatter_hidden_block_to_interleaved(
        &gpu,
        &hidden_rb,
        &draft_scratch.target_hidden,
        0,
        prompt_tokens.len(), // block_size: seed wrote prompt_len contiguous slots
        prompt_tokens.len(), // n_rows:     keep all of them
    )
    .expect("seed scatter");
    draft_scratch.uploaded_target_hidden_rows = prompt_tokens.len();
    // Seed per-row absolute positions for the draft's cross-attention RoPE.
    // Pre-eviction these match [0..prompt_len) exactly, so FlashCASK-free runs
    // stay byte-identical to the old contiguous-range behaviour.
    draft_scratch.target_hidden_abs_positions =
        (0..prompt_tokens.len() as i32).collect();
    let prefill_secs = t2.elapsed().as_secs_f64();
    let prefill_tok_s = prompt_tokens.len() as f64 / prefill_secs.max(1e-9);
    eprintln!("prefill in {:.2}s ({:.1} tok/s)", prefill_secs, prefill_tok_s);
    vram_report(&gpu.hip, "after_prefill");

    // ── Build FlashCASK policy (opt-in via --cask-sidecar) ──────────
    // The policy evicts target.kv_cache between spec_step cycles.
    // compact_offset is maintained on kv_cache itself, so qwen35's
    // forward_scratch sees the right RoPE phase without extra plumbing.
    let cask_policy: Option<CaskPolicy> = if let Some(path) = cask_sidecar.as_ref() {
        let centers = TriAttnCenters::load(Path::new(path)).expect("load cask sidecar");
        let fa_layer_ids: Vec<usize> = target.config.layer_types.iter().enumerate()
            .filter_map(|(i, t)| if *t == LayerType::FullAttention { Some(i) } else { None })
            .collect();
        let n_rot = (target.config.head_dim as f32 * target.config.partial_rotary_factor) as usize;
        // Ensure target KV has enough headroom for budget+beta+B+margin. The
        // existing slot_cfg sized it to ctx_capacity + block_size + 16 — we
        // don't resize here; just assert.
        assert!(
            target.kv_cache.max_seq >= cask_budget + cask_beta + draft_scratch_b + 4,
            "target.kv_cache.max_seq ({}) < cask_budget+beta+B+4 ({}) — raise --ctx or lower --cask-budget/beta",
            target.kv_cache.max_seq,
            cask_budget + cask_beta + draft_scratch_b + 4,
        );
        let base = EvictionCtx::new(
            &mut gpu, &centers, fa_layer_ids,
            cask_budget, cask_beta,
            target.config.n_heads, target.config.n_kv_heads, target.config.head_dim,
            n_rot, target.config.rope_theta, target.kv_cache.max_seq,
        ).expect("build EvictionCtx for FlashCASK");
        Some(if use_cask {
            eprintln!("FlashCASK: CASK α={:.2} m={} budget={} β={}", cask_core_frac, cask_fold_m, cask_budget, cask_beta);
            CaskPolicy::Cask(CaskCtx::new(base, cask_core_frac, cask_fold_m))
        } else {
            eprintln!("FlashCASK: TriAttention (plain) budget={} β={}", cask_budget, cask_beta);
            CaskPolicy::Plain(base)
        })
    } else { None };

    // Post-prefill eviction: if the prompt already filled past the
    // threshold, compact once before decoding so the spec loop starts at
    // budget-sized physical state.
    let mut position: usize = prompt_tokens.len();
    if let Some(ref p) = cask_policy {
        if let Some(ev) = p.maybe_evict(&mut gpu, &mut target.kv_cache, position)
            .expect("post-prefill cask evict") {
            let pre_phys = position;
            eprintln!(
                "FlashCASK: post-prefill compact {} -> {} (compact_offset={})",
                pre_phys, ev.new_physical, target.kv_cache.compact_offset,
            );
            position = ev.new_physical;
            // Mirror the KV eviction into the draft's target_hidden so
            // draft/target see the same windowed context.
            if !ev.retain_mask.is_empty() {
                speculative::apply_eviction_retain_to_draft(
                    &mut gpu,
                    &mut draft_scratch,
                    &ev.retain_mask,
                    draft_cfg.num_extract(),
                    draft_cfg.hidden,
                    pre_phys,
                ).expect("mirror eviction to draft (post-prefill)");
            }
        }
    }

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
    // `position` was already declared above (it may have been advanced by a
    // post-prefill CASK eviction). Keep it as-is.
    let mut seed_token: u32 = first_token;

    // Loop-break cycle detector (HIPFIRE_DFLASH_LOOP_BREAK=stop|escalate|temp).
    // After each commit, hash the trailing 32-token window. Hit = the
    // model has entered a structural repeat (block-level attractor — the
    // kind that slips past the first-128-tok coherence gate).
    //
    // Modes:
    //   stop (=1, default): track consecutive hits; once >= STOP_AFTER
    //     terminate decode as if EOS was emitted. Simple, gracefully
    //     stops degeneration.
    //   escalate: on each STOP_AFTER threshold trigger, bump runtime
    //     repeat_penalty by RP_STEP (cap at RP_MAX). The bumped RP
    //     penalizes the loop's vocabulary; gen continues. After
    //     RECOVERY_CYCLES cycles with no hits, decay RP back toward
    //     baseline. Terminate only if MAX_ESCALATIONS hit (RP cap reached
    //     AND the bumped value still loops).
    //   temp: legacy diagnostic — bump temp on each hit, reset on miss.
    //     Empirically bounces the model between attractors.
    //
    // Detection cost: one DefaultHasher pass over 32 u32s per cycle (~ns).
    // Memory: HashSet<u64>, ≤ max_tokens / 12 entries (~125 for max=1500).
    let loop_break_raw = std::env::var("HIPFIRE_DFLASH_LOOP_BREAK").ok();
    let loop_break_mode: &str = match loop_break_raw.as_deref() {
        Some("temp") => "temp",
        Some("escalate") | Some("recover") => "escalate",
        Some("stop") | Some("1") | Some("on") | Some("true") => "stop",
        _ => "off",
    };
    let loop_break_on = loop_break_mode != "off";
    let loop_break_temp: f32 = std::env::var("HIPFIRE_DFLASH_LOOP_BREAK_TEMP")
        .ok().and_then(|s| s.parse().ok()).unwrap_or(1.0);
    let loop_break_stop_after: usize = std::env::var("HIPFIRE_DFLASH_LOOP_BREAK_STOP_AFTER")
        .ok().and_then(|s| s.parse().ok()).unwrap_or(3);
    let loop_break_rp_step: f32 = std::env::var("HIPFIRE_DFLASH_LOOP_BREAK_RP_STEP")
        .ok().and_then(|s| s.parse().ok()).unwrap_or(0.10);
    let loop_break_rp_max: f32 = std::env::var("HIPFIRE_DFLASH_LOOP_BREAK_RP_MAX")
        .ok().and_then(|s| s.parse().ok()).unwrap_or(1.30);
    let loop_break_recovery: usize = std::env::var("HIPFIRE_DFLASH_LOOP_BREAK_RECOVERY")
        .ok().and_then(|s| s.parse().ok()).unwrap_or(32);
    let loop_break_max_escalations: usize = std::env::var("HIPFIRE_DFLASH_LOOP_BREAK_MAX_ESCALATIONS")
        .ok().and_then(|s| s.parse().ok()).unwrap_or(4);
    const LOOP_BREAK_WINDOW: usize = 32;
    let mut runtime_temp: f32 = temp;
    let mut runtime_repeat_penalty: f32 = repeat_penalty;
    let mut loop_break_hits: usize = 0;
    let mut loop_break_consecutive: usize = 0;
    let mut loop_break_escalations: usize = 0;
    let mut loop_break_clean_streak: usize = 0;
    let mut window_hashes: std::collections::HashSet<u64> = std::collections::HashSet::new();
    if loop_break_on {
        match loop_break_mode {
            "stop" => eprintln!(
                "[loop-break] enabled: mode=stop window={LOOP_BREAK_WINDOW} stop_after={loop_break_stop_after}"
            ),
            "escalate" => eprintln!(
                "[loop-break] enabled: mode=escalate window={LOOP_BREAK_WINDOW} stop_after={loop_break_stop_after} \
                 rp_step={loop_break_rp_step} rp_max={loop_break_rp_max} recovery={loop_break_recovery} \
                 max_escalations={loop_break_max_escalations} (baseline rp={repeat_penalty})"
            ),
            _ => eprintln!(
                "[loop-break] enabled: mode=temp window={LOOP_BREAK_WINDOW} bump_temp={loop_break_temp} (canonical temp={temp})"
            ),
        }
    }

    // SpecStats histogram must fit the max accept_len we'll ever see, so
    // size by draft_scratch_b (which accounts for adaptive_b_max).
    let mut stats = SpecStats::new(draft_scratch_b);

    if adaptive_b && draft_scratch_b != draft_cfg.block_size {
        eprintln!(
            "decoding (max {max_tokens} tokens, adaptive-B range {adaptive_b_min}..={adaptive_b_max}, draft trained at {})...",
            draft_cfg.block_size,
        );
    } else {
        eprintln!("decoding (max {max_tokens} tokens, block_size {})...", draft_cfg.block_size);
    }

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

    // HIPFIRE_PROFILE=1: enable per-kernel profiling for `--profile-cycles N`
    // worth of cycles (default 5) starting at cycle 1 (after a warm-up cycle
    // 0 to settle the JIT). Prints kernel breakdown after the limit.
    let do_profile = std::env::var("HIPFIRE_PROFILE").ok().as_deref() == Some("1");
    let profile_cycles_target: usize = std::env::var("HIPFIRE_PROFILE_CYCLES")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(5);
    let mut profile_cycle_count: usize = 0;
    let mut profile_armed = false;

    // ── AR baseline branch: skip DFlash, pure greedy AR via target ───
    // Used to confirm whether the prompt + target alone are coherent.
    // Same tokenization, same model, same greedy; isolates DFlash vs model.
    if ar_baseline {
        eprintln!("AR-BASELINE MODE: pure greedy target decode (no DFlash)");
        let t_ar = Instant::now();
        // Position already advanced to prompt_tokens.len() during prefill.
        // seed_token = target's argmax at position `prompt_len` (first emit).
        let mut cur_token = seed_token;
        while emitted.len() < max_tokens {
            if position >= ctx_capacity {
                eprintln!("hit ctx_capacity {}; stopping", ctx_capacity);
                break;
            }
            engine::qwen35::forward_scratch(
                &mut gpu,
                &target.weights,
                &target.config,
                cur_token,
                position,
                &mut target.kv_cache,
                &mut target.dn_state,
                &target.scratch,
            ).expect("ar forward");
            let lg = gpu.download_f32(&target.scratch.logits).expect("logits");
            let next = lg.iter().enumerate().fold((0u32, f32::NEG_INFINITY), |(best, bv), (i, &v)| {
                if v > bv { (i as u32, v) } else { (best, bv) }
            }).0;
            emitted.push(next);
            position += 1;
            if let Some(ref p) = cask_policy {
                if let Some(ev) = p.maybe_evict(&mut gpu, &mut target.kv_cache, position)
                    .expect("ar cask evict") {
                    // AR baseline doesn't touch draft state — no mirror needed.
                    position = ev.new_physical;
                }
            }
            if tokenizer.is_terminator(next) {
                eprintln!("eos (id={next})");
                break;
            }
            cur_token = next;
        }
        let ar_elapsed = t_ar.elapsed().as_secs_f64();
        let text = tokenizer.decode(&emitted);
        eprintln!("--- AR-BASELINE OUTPUT ---");
        println!("{text}");
        eprintln!("--------------------------");
        eprintln!("emitted: {} tokens in {:.2}s  ({:.2} tok/s)",
                  emitted.len(), ar_elapsed, emitted.len() as f64 / ar_elapsed);
        eprintln!("AR tokens: {:?}", emitted);
        return;
    }

    // HIPFIRE_HOST_TIMING=1: dump per-cycle host-side wall-clock breakdown
    // (launch overhead vs D2D/D2H/H2D vs other host work) by diffing the
    // hip-bridge launch_counters around each cycle.
    let host_timing = std::env::var("HIPFIRE_HOST_TIMING").ok().as_deref() == Some("1");
    let mut per_cycle_wall_us: Vec<u64> = Vec::new();
    let mut per_cycle_api_us: Vec<(u64, u64, u64, u64, u64, u64, u64, u64, u64)> = Vec::new();
    // columns: launch, h2d, d2h, d2d, memset, stream_sync, event_sync, device_sync, graph_launch

    // HIPFIRE_DPM_WARMUP_SECS: run a memset loop on a 256 MB scratch before
    // the decode timer starts, to pin the GPU at high DPM. See dispatch.rs
    // `dpm_warmup` for rationale — between-process DPM variance has been
    // observed at 7× wall-clock (52 ms vs 358 ms/cycle on the same bench),
    // which is orders of magnitude more than the ±10-15% noise band our
    // methodology doc calls out. Default 0 (disabled) for backward compat.
    if let Ok(secs_str) = std::env::var("HIPFIRE_DPM_WARMUP_SECS") {
        let secs: f32 = secs_str.parse().unwrap_or(0.0);
        if secs > 0.0 {
            gpu.dpm_warmup(secs).expect("dpm warmup");
        }
    }

    // Reset Task #93 Phase B seed-oracle counters so stats reflect this run
    // only (process-cumulative counters would poison multi-run harnesses).
    engine::speculative::reset_seed_oracle_stats();
    engine::speculative::reset_ddtree_meta_stats();

    // Adaptive-B state: tracks current B between cycles, plus a cooldown
    // counter and a histogram for end-of-run reporting.
    let mut current_adaptive_b: usize = draft_cfg.block_size;
    let mut adaptive_b_cycles_since_change: usize = 0;
    let mut adaptive_b_histogram: std::collections::HashMap<usize, u32> =
        std::collections::HashMap::new();
    let mut adaptive_b_changes: u32 = 0;
    const ADAPTIVE_B_STEP: usize = 2;
    const ADAPTIVE_B_COOLDOWN: usize = 3;
    // Hysteresis thresholds on util = EWMA(accept_len) / (current_B - 1):
    //   util > UP   → draft keeps up, stretch B further.
    //   util < DOWN → draft lags, shrink B to cut verify cost.
    // Gap between the two prevents flapping at one util value.
    //
    // Measured util ceilings at fixed-B16 (2026-04-24 3-run median):
    //   code  τ≈7.68 / (B-1=15) = 0.51
    //   prose τ≈1.65 / 15 = 0.11
    //   instr τ≈1.82 / 15 = 0.12
    // UP=0.70 never triggers — B never grows past start. UP=0.45 picks up
    // code's high-confidence stretches without firing on prose/instr, where
    // shrinking is the right move. Env override for tuning:
    //   HIPFIRE_ADAPTIVE_B_UP=0.XX / HIPFIRE_ADAPTIVE_B_DOWN=0.XX
    let adaptive_b_up: f64 = std::env::var("HIPFIRE_ADAPTIVE_B_UP")
        .ok().and_then(|s| s.parse().ok()).unwrap_or(0.45);
    let adaptive_b_down: f64 = std::env::var("HIPFIRE_ADAPTIVE_B_DOWN")
        .ok().and_then(|s| s.parse().ok()).unwrap_or(0.25);

    let t_decode = Instant::now();
    // TTFT capture: production-realistic measure excluding DPM warmup
    // (which inflates wall-clock by HIPFIRE_DPM_WARMUP_SECS during benches).
    // Reported value = prefill_ms + first_cycle_ms, taken from t_decode
    // (set AFTER warmup) plus the previously-recorded prefill_secs.
    let mut ttft_ms: Option<f64> = None;
    while emitted.len() < max_tokens {
        if position + draft_scratch_b >= ctx_capacity {
            eprintln!("hit ctx_capacity {}; stopping", ctx_capacity);
            break;
        }
        // Per-cycle host timing snapshot (before the step).
        let (
            wall_start,
            l_start,
            htod_start,
            dtoh_start,
            dtod_start,
            memset_start,
            ssync_start,
            esync_start,
            dsync_start,
            glaunch_start,
        ) = if host_timing {
            use hip_bridge::launch_counters as lc;
            (
                Instant::now(),
                lc::launch_kernel::time_ns(),
                lc::memcpy_htod::time_ns(),
                lc::memcpy_dtoh::time_ns(),
                lc::memcpy_dtod::time_ns(),
                lc::memset::time_ns(),
                lc::stream_sync::time_ns(),
                lc::event_sync::time_ns(),
                lc::device_sync::time_ns(),
                lc::graph_launch::time_ns(),
            )
        } else {
            (Instant::now(), 0, 0, 0, 0, 0, 0, 0, 0, 0)
        };
        if do_profile && stats.cycles == 1 && !profile_armed {
            // First cycle was the JIT warm-up. Arm profiling now and drain
            // after `profile_cycles_target` more cycles.
            rdna_compute::profile::start();
            profile_armed = true;
        }
        if do_profile && profile_armed
            && stats.cycles >= 1 + profile_cycles_target
            && profile_cycle_count == 0
        {
            profile_cycle_count = stats.cycles - 1;
            if let Some(entries) = rdna_compute::profile::stop() {
                use std::collections::HashMap;
                let mut by_kernel: HashMap<&str, (f64, usize, usize)> = HashMap::new();
                for e in &entries {
                    let entry = by_kernel.entry(e.kernel).or_insert((0.0, 0, 0));
                    entry.0 += e.time_us;
                    entry.1 += 1;
                    entry.2 += e.bytes;
                }
                let mut kerns: Vec<_> = by_kernel.into_iter().collect();
                kerns.sort_by(|a, b| b.1.0.partial_cmp(&a.1.0).unwrap());
                let total_us: f64 = kerns.iter().map(|(_, (t, _, _))| t).sum();
                eprintln!(
                    "\n=== PROFILE ({} kernel calls over {} cycles, {:.1}ms total kernel time) ===",
                    entries.len(), profile_cycle_count, total_us / 1000.0,
                );
                eprintln!(
                    "  {:50} {:>6} {:>10} {:>10} {:>7} {:>10}",
                    "kernel", "calls", "total_ms", "us/call", "%", "MB",
                );
                for (kern, (us, n, bytes)) in &kerns {
                    if *us / total_us < 0.005 { continue; } // skip <0.5%
                    eprintln!(
                        "  {kern:50} {n:>6} {:>10.2} {:>10.0} {:>6.1}% {:>10.1}",
                        us / 1000.0,
                        us / *n as f64,
                        us / total_us * 100.0,
                        *bytes as f64 / 1.0e6,
                    );
                }
            }
        }
        // Adaptive-B scheduler (Task #93 Phase B replacement — 2026-04-24).
        //
        // Online rule, no training. Tracks recent EWMA(accept_len), divides
        // by current_B-1 to get "utilization". When draft is over-performing
        // (util > UP), bump B — amortize verify over more positions. When
        // draft is under-performing (util < DOWN), shrink B — cut verify
        // cost, let τ recover. Hysteresis + cooldown prevent flapping.
        //
        // adaptive_b_min/max default to 8..=16 (pre-2026-04-24 behaviour
        // bounded by the draft's trained block_size). User can widen with
        // --adaptive-b-range MIN:MAX; scratches upstream pre-sized for MAX.
        let block_override = if adaptive_b {
            if accepts_window.len() >= 4 && adaptive_b_cycles_since_change >= ADAPTIVE_B_COOLDOWN {
                let ewma: f64 = accepts_window.iter().copied().sum::<usize>() as f64
                    / accepts_window.len() as f64;
                let util = ewma / (current_adaptive_b.saturating_sub(1).max(1)) as f64;
                if util > adaptive_b_up
                    && current_adaptive_b + ADAPTIVE_B_STEP <= adaptive_b_max
                {
                    current_adaptive_b += ADAPTIVE_B_STEP;
                    adaptive_b_cycles_since_change = 0;
                    adaptive_b_changes += 1;
                } else if util < adaptive_b_down
                    && current_adaptive_b >= adaptive_b_min + ADAPTIVE_B_STEP
                {
                    current_adaptive_b -= ADAPTIVE_B_STEP;
                    adaptive_b_cycles_since_change = 0;
                    adaptive_b_changes += 1;
                }
            }
            adaptive_b_cycles_since_change += 1;
            *adaptive_b_histogram.entry(current_adaptive_b).or_insert(0) += 1;
            Some(current_adaptive_b)
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
            if ddtree_batched {
                speculative::spec_step_ddtree_batched(
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
                    &ddtree_scratch,
                    &verify_scratch,
                    position,
                    seed_token,
                    ctx_slice,
                    ddtree_budget,
                    ddtree_topk,
                )
                .expect("ddtree-batched spec step")
            } else {
                speculative::spec_step_ddtree(
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
                    &verify_scratch,
                    position,
                    seed_token,
                    ctx_slice,
                    ddtree_budget,
                    ddtree_topk,
                )
                .expect("ddtree spec step")
            }
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
                &verify_scratch,
                position,
                seed_token,
                ctx_slice,
                if no_tape { None } else { Some(&mut gdn_tape) },
                runtime_temp,
                &mut rng_state,
                block_override,
                ngram_cache.as_ref(),
                &emitted,
                cactus_delta,
                pld_spine,
                runtime_repeat_penalty,
                repeat_window,
            )
            .expect("spec step")
        };
        if used_pld {
            pld_accepted += step.accepted;
        }

        // Per-cycle debug for the first N cycles.
        if stats.cycles < debug_cycles {
            eprintln!(
                "[cycle {}] pos={} seed={} committed={:?} bonus={} accepted={} τ={:.3}",
                stats.cycles,
                position,
                seed_token,
                step.committed.iter().skip(1).take(4).collect::<Vec<_>>(),
                step.bonus_token,
                step.accepted,
                step.accepted as f64,
            );
            // Decode the first few committed tokens for visibility.
            let preview: Vec<u32> = step.committed.iter().skip(1).copied().collect();
            let tx = tokenizer.decode(&preview);
            eprintln!("  decoded-committed[1..]: {:?}", tx);
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

        // Per-cycle host timing snapshot (after the step).
        if host_timing {
            use hip_bridge::launch_counters as lc;
            let wall_us = wall_start.elapsed().as_micros() as u64;
            let launch_us = (lc::launch_kernel::time_ns() - l_start) / 1000;
            let htod_us = (lc::memcpy_htod::time_ns() - htod_start) / 1000;
            let dtoh_us = (lc::memcpy_dtoh::time_ns() - dtoh_start) / 1000;
            let dtod_us = (lc::memcpy_dtod::time_ns() - dtod_start) / 1000;
            let memset_us = (lc::memset::time_ns() - memset_start) / 1000;
            let ssync_us = (lc::stream_sync::time_ns() - ssync_start) / 1000;
            let esync_us = (lc::event_sync::time_ns() - esync_start) / 1000;
            let dsync_us = (lc::device_sync::time_ns() - dsync_start) / 1000;
            let glaunch_us = (lc::graph_launch::time_ns() - glaunch_start) / 1000;
            per_cycle_wall_us.push(wall_us);
            per_cycle_api_us.push((
                launch_us, htod_us, dtoh_us, dtod_us, memset_us,
                ssync_us, esync_us, dsync_us, glaunch_us,
            ));
        }

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
        if ttft_ms.is_none() && step.committed.len() > 1 {
            // Use t_decode (post-warmup) + prefill_secs so the reported
            // TTFT is what a serving-mode client would actually see.
            let first_cycle_ms = t_decode.elapsed().as_secs_f64() * 1000.0;
            ttft_ms = Some(prefill_secs * 1000.0 + first_cycle_ms);
        }

        // Loop-break cycle detector: hash trailing 32-token window, look
        // up in known-hash set. Hit = repeating block. The shift-by-cycle
        // (each iter advances ~12 tokens, window=32) means consecutive
        // windows overlap heavily but yield distinct hashes; only a true
        // verbatim repeat re-hashes to a previously-seen value.
        let mut loop_break_force_stop = false;
        if loop_break_on && emitted.len() >= LOOP_BREAK_WINDOW {
            use std::hash::{Hash, Hasher};
            let tail = &emitted[emitted.len() - LOOP_BREAK_WINDOW..];
            let mut hasher = std::collections::hash_map::DefaultHasher::new();
            tail.hash(&mut hasher);
            let h = hasher.finish();
            let hit = window_hashes.contains(&h);
            if hit {
                loop_break_hits += 1;
                loop_break_consecutive += 1;
                loop_break_clean_streak = 0;
                let trigger = loop_break_consecutive >= loop_break_stop_after;
                match loop_break_mode {
                    "stop" => {
                        if trigger {
                            eprintln!(
                                "[loop-break] cycle {} pos {}: {} consecutive repeats → terminating decode (synthetic EOS); total_hits={}",
                                stats.cycles, position, loop_break_consecutive, loop_break_hits
                            );
                            loop_break_force_stop = true;
                        } else {
                            eprintln!(
                                "[loop-break] cycle {} pos {}: window repeat ({}/{} consecutive)",
                                stats.cycles, position, loop_break_consecutive, loop_break_stop_after
                            );
                        }
                    }
                    "escalate" => {
                        if trigger {
                            // Either escalate RP or give up if escalation
                            // ladder exhausted (max bumps reached AND the
                            // bumped value still loops).
                            if loop_break_escalations >= loop_break_max_escalations
                                || runtime_repeat_penalty >= loop_break_rp_max - 1e-4
                            {
                                eprintln!(
                                    "[loop-break] cycle {} pos {}: escalation exhausted (rp={:.2} after {} bumps) → terminating; total_hits={}",
                                    stats.cycles, position, runtime_repeat_penalty,
                                    loop_break_escalations, loop_break_hits
                                );
                                loop_break_force_stop = true;
                            } else {
                                let prev_rp = runtime_repeat_penalty;
                                runtime_repeat_penalty = (runtime_repeat_penalty + loop_break_rp_step)
                                    .min(loop_break_rp_max);
                                if runtime_repeat_penalty < 1.0 + loop_break_rp_step {
                                    runtime_repeat_penalty = 1.0 + loop_break_rp_step;
                                }
                                loop_break_escalations += 1;
                                loop_break_consecutive = 0;
                                // Wipe the hash set so the bumped RP gets
                                // a fresh judgment window — otherwise the
                                // same stale tokens trigger immediately.
                                window_hashes.clear();
                                eprintln!(
                                    "[loop-break] cycle {} pos {}: ESCALATE rp {:.2} → {:.2} (bump {}/{}); total_hits={}",
                                    stats.cycles, position, prev_rp, runtime_repeat_penalty,
                                    loop_break_escalations, loop_break_max_escalations, loop_break_hits
                                );
                            }
                        } else {
                            eprintln!(
                                "[loop-break] cycle {} pos {}: window repeat ({}/{} consecutive, rp={:.2})",
                                stats.cycles, position, loop_break_consecutive,
                                loop_break_stop_after, runtime_repeat_penalty
                            );
                        }
                    }
                    _ => {
                        // legacy temp-bump mode
                        runtime_temp = loop_break_temp;
                        eprintln!(
                            "[loop-break] cycle {} pos {}: window repeat → temp={} for next cycle (count={})",
                            stats.cycles, position, loop_break_temp, loop_break_hits
                        );
                    }
                }
            } else {
                loop_break_consecutive = 0;
                loop_break_clean_streak += 1;
                if loop_break_mode == "temp" {
                    runtime_temp = temp;
                }
                // Escalate-mode RP decay: after RECOVERY cycles of clean
                // output, halve the gap toward baseline. Repeat each
                // RECOVERY cycles until back at baseline. Lets the model
                // regain natural entropy once the basin is escaped.
                if loop_break_mode == "escalate"
                    && runtime_repeat_penalty > repeat_penalty + 1e-4
                    && loop_break_clean_streak > 0
                    && loop_break_clean_streak % loop_break_recovery == 0
                {
                    let prev_rp = runtime_repeat_penalty;
                    runtime_repeat_penalty =
                        repeat_penalty + (runtime_repeat_penalty - repeat_penalty) * 0.5;
                    if runtime_repeat_penalty < repeat_penalty + 0.01 {
                        runtime_repeat_penalty = repeat_penalty;
                        loop_break_escalations = loop_break_escalations.saturating_sub(1);
                    }
                    eprintln!(
                        "[loop-break] cycle {} pos {}: clean for {} cycles → DECAY rp {:.2} → {:.2}",
                        stats.cycles, position, loop_break_clean_streak, prev_rp, runtime_repeat_penalty
                    );
                }
            }
            window_hashes.insert(h);
        }

        if loop_break_force_stop {
            eprintln!("eos");
            break;
        }

        // Advance position + pick next seed (= bonus_token).
        position += step.accepted + 1;
        seed_token = step.bonus_token;

        // FlashCASK eviction. Fires when target.kv_cache physical hits
        // budget+β. compact_offset is maintained on the cache so the next
        // cycle's target.forward_scratch uses the right RoPE phase.
        if let Some(ref p) = cask_policy {
            if let Some(ev) = p.maybe_evict(&mut gpu, &mut target.kv_cache, position)
                .expect("spec cask evict") {
                let pre_phys = position;
                position = ev.new_physical;
                // Mirror the KV eviction into the draft's target_hidden view.
                // Empty retain_mask (CASK m-fold path) → skip; draft keeps the
                // pre-eviction buffer, the old τ-collapse behaviour still
                // applies there until m-fold gets a compatible draft mirror.
                if !ev.retain_mask.is_empty() {
                    speculative::apply_eviction_retain_to_draft(
                        &mut gpu,
                        &mut draft_scratch,
                        &ev.retain_mask,
                        draft_cfg.num_extract(),
                        draft_cfg.hidden,
                        pre_phys,
                    ).expect("mirror eviction to draft");
                }
            }
        }

        // Stop on any terminator (eos_id OR eot_id). Checking only eos_id
        // misses `<|endoftext|>` when running --no-chatml on a raw-text draft
        // — see findings/dflash-benchmark-2026-04-24.md §3.5 (post-EOT
        // attractor loop). `is_terminator` covers both.
        if step.committed.iter().skip(1).any(|&t| tokenizer.is_terminator(t)) {
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
    if let Some(ref p) = cask_policy {
        eprintln!(
            "FlashCASK: {} evictions  final compact_offset={}",
            p.eviction_count(),
            target.kv_cache.compact_offset,
        );
    }
    let accept_rate = if stats.cycles > 0 {
        stats.accepted_tokens as f32 / (stats.cycles * (draft_cfg.block_size - 1)) as f32
    } else {
        0.0
    };

    // ── BENCH METRICS (machine-parseable) ─────────────────────────────────
    // Single source of truth for downstream submission scripts (LMX,
    // benchmarks/results/*.py). Keep flat key=value lines so a simple regex
    // pulls each field; do not change format without updating callers.
    // hip.get_vram_info returns (free_bytes, total_bytes) — see line 356.
    let (vram_free_bytes, vram_total_bytes) = gpu.hip.get_vram_info().unwrap_or((0, 0));
    let vram_used_mb = ((vram_total_bytes.saturating_sub(vram_free_bytes)) as f64 / (1024.0 * 1024.0)) as u64;
    let vram_total_mb = (vram_total_bytes as f64 / (1024.0 * 1024.0)) as u64;
    eprintln!("=== BENCH METRICS ===");
    eprintln!("prompt_tokens: {}", prompt_tokens.len());
    eprintln!("prefill_secs: {:.4}", prefill_secs);
    eprintln!("prefill_tok_s: {:.2}", prefill_tok_s);
    eprintln!("ttft_ms: {:.2}", ttft_ms.unwrap_or(0.0));
    eprintln!("decode_tokens_emitted: {}", emitted.len());
    eprintln!("decode_secs: {:.4}", elapsed);
    eprintln!("decode_tok_s: {:.2}", tok_s);
    eprintln!("decode_tau: {:.4}", stats.tau());
    eprintln!("decode_accept_rate: {:.4}", accept_rate);
    eprintln!("vram_used_mb: {}", vram_used_mb);
    eprintln!("vram_total_mb: {}", vram_total_mb);
    eprintln!("=====================");
    eprintln!("accept_rate (accepted / (cycles × (B-1))): {accept_rate:.3}");
    eprintln!(
        "histogram: {:?}",
        stats.acceptance_hist.iter().enumerate().collect::<Vec<_>>()
    );
    // DDTree meta-verifier pruner stats — only meaningful under --ddtree-*
    // with HIPFIRE_DDTREE_LOGW_CUTOFF set. Cycles == 0 on pure-DFlash runs.
    let meta = engine::speculative::read_ddtree_meta_stats();
    if meta.cycles > 0 {
        let mean_nodes = meta.total_nodes as f32 / meta.cycles as f32;
        eprintln!(
            "ddtree-meta: cycles={} mean_nodes={:.2} min={} max={} (cutoff={:?})",
            meta.cycles, mean_nodes, meta.min_nodes, meta.max_nodes,
            std::env::var("HIPFIRE_DDTREE_LOGW_CUTOFF").unwrap_or_else(|_| "off".to_string()),
        );
    }
    // Adaptive-B usage report — only meaningful when --adaptive-b is on.
    if adaptive_b && !adaptive_b_histogram.is_empty() {
        let mut buckets: Vec<(usize, u32)> = adaptive_b_histogram.iter()
            .map(|(&b, &c)| (b, c)).collect();
        buckets.sort_by_key(|(b, _)| *b);
        let total: u32 = buckets.iter().map(|(_, c)| *c).sum();
        let mean_b: f32 = buckets.iter()
            .map(|(b, c)| (*b as f32) * (*c as f32))
            .sum::<f32>() / total.max(1) as f32;
        let dist: String = buckets.iter()
            .map(|(b, c)| format!("B={b}:{:.1}%", *c as f32 * 100.0 / total.max(1) as f32))
            .collect::<Vec<_>>().join(" ");
        eprintln!(
            "adaptive-b: range={}..={} mean_B={:.2} changes={} dist=[{}]",
            adaptive_b_min, adaptive_b_max, mean_b, adaptive_b_changes, dist,
        );
    }
    // Task #93 Phase B seed-prediction oracle. Zero cycles = pure-AR or tree
    // paths that didn't invoke spec_step_dflash; skip in that case.
    let s = engine::speculative::read_seed_oracle_stats();
    if s.total > 0 {
        let denom = s.total as f32;
        eprintln!(
            "seed-oracle: cycles={} full_accept={} mean_accept_len={:.3} | rej_match={:.3} tail_match={:.3} anypos_match={:.3}",
            s.total, s.full_accept, s.accept_len_sum as f32 / denom,
            s.rej_match as f32 / denom,
            s.tail_match as f32 / denom,
            s.anypos_match as f32 / denom,
        );
    }
    if host_timing && !per_cycle_wall_us.is_empty() {
        // Skip first 2 cycles (JIT warm-up), summarize the rest as mean / median.
        let skip = 2.min(per_cycle_wall_us.len().saturating_sub(1));
        let wall: Vec<u64> = per_cycle_wall_us.iter().skip(skip).copied().collect();
        let api: Vec<(u64, u64, u64, u64, u64, u64, u64, u64, u64)> =
            per_cycle_api_us.iter().skip(skip).copied().collect();
        let n = wall.len().max(1);
        let mean_wall = wall.iter().sum::<u64>() / n as u64;
        let mean_launch = api.iter().map(|x| x.0).sum::<u64>() / n as u64;
        let mean_htod = api.iter().map(|x| x.1).sum::<u64>() / n as u64;
        let mean_dtoh = api.iter().map(|x| x.2).sum::<u64>() / n as u64;
        let mean_dtod = api.iter().map(|x| x.3).sum::<u64>() / n as u64;
        let mean_memset = api.iter().map(|x| x.4).sum::<u64>() / n as u64;
        let mean_ssync = api.iter().map(|x| x.5).sum::<u64>() / n as u64;
        let mean_esync = api.iter().map(|x| x.6).sum::<u64>() / n as u64;
        let mean_dsync = api.iter().map(|x| x.7).sum::<u64>() / n as u64;
        let mean_glaunch = api.iter().map(|x| x.8).sum::<u64>() / n as u64;
        let tracked = mean_launch + mean_htod + mean_dtoh + mean_dtod + mean_memset
            + mean_ssync + mean_esync + mean_dsync + mean_glaunch;
        let untracked = mean_wall.saturating_sub(tracked);
        // Cumulative counts — post-run totals divided by elapsed cycles give
        // mean per-cycle API call counts. Helpful for isolating which op is
        // the hot path (few-but-fat vs many-but-thin).
        use hip_bridge::launch_counters as lc;
        let total_cycles = per_cycle_wall_us.len() as u64;
        let n_launch = lc::launch_kernel::count() / total_cycles.max(1);
        let n_htod = lc::memcpy_htod::count() / total_cycles.max(1);
        let n_dtoh = lc::memcpy_dtoh::count() / total_cycles.max(1);
        let n_dtod = lc::memcpy_dtod::count() / total_cycles.max(1);
        let n_memset = lc::memset::count() / total_cycles.max(1);
        let n_ssync = lc::stream_sync::count() / total_cycles.max(1);
        let n_glaunch = lc::graph_launch::count() / total_cycles.max(1);
        let b_dtoh = lc::memcpy_dtoh::bytes() / total_cycles.max(1);
        let b_memset = lc::memset::bytes() / total_cycles.max(1);
        eprintln!(
            "host timing (mean over {} cycles, µs): wall={}\n  launch={} (n={}) h2d={} (n={}) d2h={} (n={}, {}KB) d2d={} (n={}) memset={} (n={}, {}MB) glaunch={} (n={})\n  ssync={} (n={}) esync={} dsync={} → other={}",
            n, mean_wall,
            mean_launch, n_launch, mean_htod, n_htod, mean_dtoh, n_dtoh, b_dtoh / 1024,
            mean_dtod, n_dtod, mean_memset, n_memset, b_memset / (1024*1024), mean_glaunch, n_glaunch,
            mean_ssync, n_ssync, mean_esync, mean_dsync, untracked,
        );
    }
    eprintln!("DFlash tokens: {:?}", emitted);
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
