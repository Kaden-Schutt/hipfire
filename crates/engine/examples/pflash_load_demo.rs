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

    // Estimate VRAM by peeking at the drafter HFQ config without uploading.
    let drafter_hfq_peek = HfqFile::open(Path::new(drafter_path)).expect("open drafter HFQ");
    let drafter_cfg_peek = engine::hfq::config_from_hfq(&drafter_hfq_peek)
        .expect("drafter llama config");
    let max_kv_seq = 4096usize;
    let est_bytes = pflash::drafter_vram_estimate_bytes(&drafter_cfg_peek, max_kv_seq);
    eprintln!("drafter VRAM estimate: {:.2} MB ({} layers × {} hidden, max_kv_seq={max_kv_seq})",
        est_bytes as f64 / (1024.0 * 1024.0),
        drafter_cfg_peek.n_layers, drafter_cfg_peek.hidden_dim);
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
    if let Some(ref c) = state.drafter_config {
        eprintln!("drafter arch: dim={} layers={} heads={} kv_heads={} head_dim={}",
            c.dim, c.n_layers, c.n_heads, c.n_kv_heads, c.head_dim);
    }
    if let Some(ref t) = state.drafter_tokenizer {
        eprintln!("drafter tokenizer: {} tokens", t.vocab_size());
    }

    // Demonstrate the gating result that the daemon will see.
    use engine::pflash::{decide_bypass, PflashMode, RequestKind};
    let demo_cfg = PflashConfig { mode: PflashMode::Always, ..cfg };
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
        let toy_prompt: Vec<u32> = (0..32u32).map(|i| 100 + i).collect();
        let block_size = 8usize;
        match engine::pflash::compute_scores_cpu(
            &mut state, &mut gpu, &toy_prompt, block_size,
        ) {
            Ok(bs) => {
                eprintln!("compute_scores_cpu: {} blocks of {} tokens, scores {:?}",
                    bs.n_blocks, bs.block_size, bs.scores);
                let any_nonzero = bs.scores.iter().any(|s| s.abs() > 1e-6);
                let any_finite = bs.scores.iter().all(|s| s.is_finite());
                eprintln!("scores any_nonzero={any_nonzero}, all_finite={any_finite}");
                any_nonzero && any_finite
            }
            Err(e) => {
                eprintln!("compute_scores_cpu failed: {e:?}");
                false
            }
        }
    } else {
        // Skipped — not a regression, scoring requires matched tokenizers.
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
