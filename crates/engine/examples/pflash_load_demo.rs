//! Smoke: load a Qwen3-family drafter into PflashState and verify
//! tokenizer compatibility against a Qwen3.5 target.
//!
//! Usage:
//!   cargo run --release --features deltanet --example pflash_load_demo -- \
//!     <target.hfq> <drafter.hfq>
//!
//! Reports drafter VRAM estimate + actual load wall-time + tokenizer-compat
//! verdict. Exit 0 on PASS (loaded + compat), 1 on tokenizer mismatch, 2 on
//! load failure.

use engine::hfq::HfqFile;
use engine::pflash::{self, PflashConfig, PflashState};
use engine::qwen35;
use engine::tokenizer::Tokenizer;
use std::path::Path;
use std::time::Instant;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("Usage: pflash_load_demo <target.hfq> <drafter.hfq>");
        std::process::exit(2);
    }
    let target_path = &args[1];
    let drafter_path = &args[2];

    eprintln!("=== PFlash drafter load + tokenizer-compat smoke ===");
    eprintln!("target:  {target_path}");
    eprintln!("drafter: {drafter_path}");

    // Load target tokenizer (Qwen3.5 hybrid). We don't load weights — the
    // target is already running in production by the time the daemon calls
    // load_drafter, so this smoke just needs the tokenizer.
    let target_hfq = HfqFile::open(Path::new(target_path)).expect("open target HFQ");
    let target_tokenizer = Tokenizer::from_hfq_metadata(&target_hfq.metadata_json)
        .expect("target tokenizer");
    let target_cfg = qwen35::config_from_hfq(&target_hfq).expect("target qwen35 config");
    eprintln!("target tokenizer: {} tokens", target_tokenizer.vocab_size());
    eprintln!("target arch: dim={} layers={} heads={}", target_cfg.dim, target_cfg.n_layers, target_cfg.n_heads);

    let mut gpu = rdna_compute::Gpu::init().expect("GPU init");

    // VRAM estimate. arch_id distinguishes hybrid (5/6) from plain (1).
    let drafter_hfq_peek = HfqFile::open(Path::new(drafter_path)).expect("open drafter HFQ");
    let max_kv_seq = 4096usize;
    let is_hybrid = drafter_hfq_peek.arch_id == 5 || drafter_hfq_peek.arch_id == 6;
    let est_layers_hidden = if is_hybrid {
        let c = qwen35::config_from_hfq(&drafter_hfq_peek).expect("hybrid config");
        ("hybrid", c.n_layers, c.hidden_dim)
    } else if let Some(c) = engine::hfq::config_from_hfq(&drafter_hfq_peek) {
        ("plain", c.n_layers, c.hidden_dim)
    } else {
        ("unknown", 0, 0)
    };
    eprintln!("drafter family: {} ({} layers × {} hidden, max_kv_seq={max_kv_seq})",
        est_layers_hidden.0, est_layers_hidden.1, est_layers_hidden.2);
    drop(drafter_hfq_peek);

    // Build a minimal config to construct PflashState, then load.
    let cfg = PflashConfig {
        drafter_path: Some(drafter_path.clone()),
        ..Default::default()
    };
    let mut state = PflashState::new(&cfg);

    let t_load = Instant::now();
    let res = pflash::load_drafter(
        &mut state, &mut gpu, Path::new(drafter_path), &target_tokenizer, max_kv_seq,
    );
    let load_ms = t_load.elapsed().as_millis();
    match res {
        Ok(()) => {}
        Err(e) => {
            eprintln!("load failed in {load_ms} ms: {e:?}");
            std::process::exit(2);
        }
    }
    eprintln!("loaded in {load_ms} ms");
    eprintln!("drafter_loaded:    {}", state.drafter_loaded);
    eprintln!("tokenizer_compat:  {}", state.tokenizer_compat);
    if let Some(ref m) = state.drafter_model {
        eprintln!("drafter variant: {} (layers={} kv_heads={} head_dim={})",
            m.variant_name(), m.n_layers(), m.n_kv_heads(), m.head_dim());
        eprintln!("auto score_layer_idx: {:?} (None = no FullAttention layer)",
            m.score_layer_idx());
    }
    if let Some(ref t) = state.drafter_tokenizer {
        eprintln!("drafter tokenizer: {} tokens", t.vocab_size());
    }

    // Demonstrate the gating result that the daemon will see.
    use engine::pflash::{decide_bypass, PflashMode, RequestKind};
    let demo_cfg = PflashConfig { mode: PflashMode::Always, ..cfg.clone() };
    let probe_tokens = vec![1u32; 100];
    let bypass = decide_bypass(&state, &demo_cfg, &probe_tokens, RequestKind::Text);
    eprintln!("decide_bypass (Always, 100 tok, Text): {bypass:?}");

    // Capture the verdict BEFORE unload — unload_drafter resets
    // tokenizer_compat to false (idempotency invariant), so checking it
    // afterward would always FAIL even on a compatible pair.
    let compat = state.tokenizer_compat;

    // If tokenizers match, exercise compute_scores_cpu on a tiny synthetic
    // prompt so the Phase 1.2 scoring path gets one end-to-end smoke per
    // demo run. Skip if the drafter was not the matching pair (compat=false)
    // since the scoring fn assumes drafter handles the target's token ids.
    //
    // Track scoring outcome as a separate exit-code component so the demo
    // FAILs when scoring is broken even if tokenizer-compat is fine — Codex
    // caught that hiding scoring errors behind tokenizer-pass would let
    // regressions ship undetected.
    let scoring_ok = if compat {
        // 32-token toy prompt. Build a PflashConfig with Always mode +
        // small block/sink/recent so the budget actually compresses on
        // such a tiny prompt. Drive maybe_compress_prompt end-to-end.
        let toy_prompt: Vec<u32> = (0..32u32).map(|i| 100 + i).collect();
        let demo_cfg2 = PflashConfig {
            mode: PflashMode::Always,
            keep_ratio: 0.5,
            sink_tokens: 4,
            recent_tokens: 4,
            block_size: 8,
            min_keep_tokens: 0,
            ..cfg.clone()
        };
        match engine::pflash::maybe_compress_prompt(
            &mut gpu, &mut state, &demo_cfg2, &toy_prompt, RequestKind::Text, &[],
        ) {
            Ok(engine::pflash::PflashDecision::Compressed(cp)) => {
                eprintln!("maybe_compress_prompt: source={} kept={} ratio={:.3}",
                    cp.source_tokens, cp.kept_tokens,
                    cp.kept_tokens as f32 / cp.source_tokens.max(1) as f32);
                eprintln!("source_md5    = {}", cp.source_md5);
                eprintln!("compressed_md5= {}", cp.compressed_md5);
                eprintln!("kept_spans    = {:?}", cp.kept_spans);
                eprintln!("timings: score={}ms select={}ms gather={}ms total={}ms",
                    cp.timings.score_ms, cp.timings.select_ms,
                    cp.timings.gather_ms, cp.timings.total_ms);

                let span_total: usize = cp.kept_spans.iter().map(|(s, e)| e - s).sum();
                let length_ok = cp.kept_tokens == span_total
                    && cp.kept_tokens == cp.token_ids.len()
                    && cp.kept_tokens < cp.source_tokens;
                let spans_disjoint = cp.kept_spans.windows(2).all(|w| w[0].1 < w[1].0);
                let monotone_tokens = cp.kept_spans.iter()
                    .flat_map(|&(s, e)| (s..e).map(|i| toy_prompt[i]))
                    .eq(cp.token_ids.iter().copied());
                let md5_present = !cp.source_md5.is_empty() && !cp.compressed_md5.is_empty();
                // Re-run compute_scores_cpu in isolation as a scorer-health
                // probe. maybe_compress_prompt already filters degenerate
                // scores via BlockScores::well_formed, but the demo still
                // surfaces the raw scores so a regression that makes
                // scoring meaningless (all-zero, all-nan) is visible to
                // anyone running the smoke. NOTE: this re-prefills the
                // drafter; safe because the scoring path tolerates an
                // already-advanced cache via state.unload_drafter on exit.
                // Cross-check: CPU-batched and GPU-batched scoring paths
                // should agree to within numerical tolerance. If they don't,
                // the new HIP kernel is wrong.
                let scorer_health_ok = {
                    let mut probe_state = state.drafter_loaded;
                    if probe_state {
                        let cpu = engine::pflash::compute_scores_batched(
                            &mut state, &mut gpu, &toy_prompt, demo_cfg2.block_size,
                        );
                        let gpu_res = engine::pflash::compute_scores_batched_gpu(
                            &mut state, &mut gpu, &toy_prompt, demo_cfg2.block_size,
                        );
                        match (cpu, gpu_res) {
                            (Ok(c), Ok(g)) => {
                                let max_err: f32 = c.scores.iter().zip(g.scores.iter())
                                    .map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
                                eprintln!("scorer xcheck: cpu={:?} gpu={:?} max_abs_err={:.3e}",
                                    c.scores, g.scores, max_err);
                                let any_nonzero = g.scores.iter().any(|s| s.abs() > 1e-6);
                                let all_finite = g.scores.iter().all(|s| s.is_finite());
                                eprintln!("scorer health: any_nonzero={any_nonzero} all_finite={all_finite}");
                                // Parallel-reduce vs sequential f32 sum order.
                                // Plain path (kv_dim=1024) sees 1e-3; hybrid
                                // path (kv_dim varies, 24-32 layers, deeper
                                // accumulation) sees up to ~3e-2. The PFlash
                                // contract is "ranks the same blocks", and
                                // the GPU path is what production uses.
                                any_nonzero && all_finite && max_err < 5e-2
                            }
                            (Err(e), _) | (_, Err(e)) => {
                                eprintln!("scorer probe errored: {e:?}");
                                probe_state = false;
                                false
                            }
                        }
                    } else {
                        eprintln!("scorer health probe skipped: drafter not loaded");
                        false
                    }
                };
                eprintln!("length_ok={length_ok} spans_disjoint={spans_disjoint} \
                           monotone={monotone_tokens} md5_present={md5_present} \
                           scorer_health_ok={scorer_health_ok}");
                length_ok && spans_disjoint && monotone_tokens && md5_present && scorer_health_ok
            }
            Ok(engine::pflash::PflashDecision::Bypass { reason }) => {
                eprintln!("maybe_compress_prompt unexpectedly bypassed: {reason:?}");
                false
            }
            Err(e) => {
                eprintln!("maybe_compress_prompt failed: {e:?}");
                false
            }
        }
    } else {
        // Skipped: not a regression, scoring requires matched tokenizers.
        true
    };

    // Free GPU resources before exit so the next bench/test sees a clean pool.
    state.unload_drafter(&mut gpu);

    if !compat {
        eprintln!("FAIL: tokenizer_compat = false (drafter and target tokenizers diverge)");
        std::process::exit(1);
    }
    if !scoring_ok {
        eprintln!("FAIL: compute_scores_cpu errored or returned degenerate / non-finite scores");
        std::process::exit(2);
    }
    eprintln!("PASS");
}
