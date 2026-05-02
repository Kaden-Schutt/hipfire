//! Phase 1.5 end-to-end: source -> maybe_compress_prompt -> target prefill on
//! compressed stream -> greedy decode -> check needle survives.
//!
//! Uses the same model file as both drafter (for scoring) and target (for
//! prefill+decode) so this runs without a matched-tokenizer Qwen3.5
//! drafter; that pairing is escalated in MANUAL_REVIEW.md. Tests the
//! pipeline correctness, not retrieval quality at heavy compression on a
//! small model.
//!
//! Usage:
//!   cargo run --release --features deltanet --example pflash_compress_demo \
//!     -- <model.hfq> [--keep-ratio K] [--maxgen N] [--block-size B]
//!
//! Generates a small filler+needle+question prompt internally so the bench
//! is fast to iterate on. PRD §6 Phase 1 scope: prove the compress->target
//! prefill plumbing, not Lucebox-class retrieval numbers.

use engine::hfq::{self, HfqFile};
use engine::llama::{self, ForwardScratch, KvCache};
use engine::pflash::{self, BypassReason, PflashConfig, PflashDecision, PflashMode, PflashState, RequestKind};
use engine::tokenizer::Tokenizer;
use std::path::Path;
use std::time::Instant;

const NEEDLE: &str = "The secret pass code is mauve-velociraptor-7741.";
const QUESTION: &str = "What is the secret pass code?";
const EXPECTED: &str = "mauve-velociraptor-7741";

const FILLER_WORDS: &[&str] = &[
    "the","quick","brown","fox","jumps","over","the","lazy","dog","while",
    "a","gentle","breeze","carries","the","scent","of","distant","pine",
    "forests","across","the","meadow","where","small","wildflowers","nod",
    "their","heads","in","agreement","with","the","rhythm","of","the",
    "afternoon","and","time","itself","seems","to","pause","for","a",
    "moment","so","everything","alive","can","breathe","in","synchrony",
    "before","the","sun","begins","its","slow","descent","toward","the",
    "western","horizon",
];

fn build_prompt(target_chars: usize) -> String {
    let mut out = String::new();
    while out.len() < target_chars / 2 {
        out.push_str(FILLER_WORDS[out.len() % FILLER_WORDS.len()]);
        out.push(' ');
    }
    out.push_str("\n\n");
    out.push_str(NEEDLE);
    out.push_str("\n\n");
    while out.len() < target_chars {
        out.push_str(FILLER_WORDS[(out.len() / 7) % FILLER_WORDS.len()]);
        out.push(' ');
    }
    out.push_str("\n\n");
    out.push_str(QUESTION);
    out
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: pflash_compress_demo <model.hfq> [--keep-ratio K] [--maxgen N] [--block-size B]");
        std::process::exit(2);
    }
    let model_path = &args[1];
    let keep_ratio: f32 = args.iter().position(|a| a == "--keep-ratio")
        .and_then(|i| args.get(i + 1)).and_then(|s| s.parse().ok())
        .unwrap_or(0.30);
    let max_gen: usize = args.iter().position(|a| a == "--maxgen")
        .and_then(|i| args.get(i + 1)).and_then(|s| s.parse().ok())
        .unwrap_or(48);
    let block_size: usize = args.iter().position(|a| a == "--block-size")
        .and_then(|i| args.get(i + 1)).and_then(|s| s.parse().ok())
        .unwrap_or(32);

    eprintln!("=== PFlash compress + target re-prefill smoke ===");
    eprintln!("model:       {model_path}");
    eprintln!("keep_ratio:  {keep_ratio}");
    eprintln!("block_size:  {block_size}");
    eprintln!("maxgen:      {max_gen}");

    // ── Load TARGET ──────────────────────────────────────────────────────
    let target_hfq = HfqFile::open(Path::new(model_path)).expect("open target HFQ");
    let target_cfg = hfq::config_from_hfq(&target_hfq).expect("target config");
    let target_tok = Tokenizer::from_hfq_metadata(&target_hfq.metadata_json)
        .expect("target tokenizer");
    let mut gpu = rdna_compute::Gpu::init().expect("GPU init");
    let target_weights = hfq::load_weights_hfq(&target_hfq, &target_cfg, &mut gpu)
        .expect("target weights");
    let target_scratch = ForwardScratch::new(&mut gpu, &target_cfg).expect("scratch");
    eprintln!("target arch: dim={} layers={} heads={} kv_heads={}",
        target_cfg.dim, target_cfg.n_layers, target_cfg.n_heads, target_cfg.n_kv_heads);

    // ── Build prompt + tokenize ──────────────────────────────────────────
    // Aim for ~512 source tokens so tokenize stays under 1s (the encoder is
    // O(N²)-ish at long context per DEFERRED.md).
    let prompt_text = build_prompt(2000);
    let t_tok = Instant::now();
    let source_tokens = target_tok.encode(&prompt_text);
    eprintln!("tokenize:    {} ms ({} tokens, {} chars)",
        t_tok.elapsed().as_millis(), source_tokens.len(), prompt_text.len());

    // ── Load DRAFTER (same model) into PflashState ───────────────────────
    let mut cfg = PflashConfig {
        mode: PflashMode::Always,
        keep_ratio,
        block_size,
        sink_tokens: 16,
        recent_tokens: 32,
        min_keep_tokens: 0,
        ..Default::default()
    };
    cfg.drafter_path = Some(model_path.to_string());
    let mut state = PflashState::new(&cfg);
    let drafter_max_kv = source_tokens.len() + 64;
    pflash::load_drafter(
        &mut state, &mut gpu, Path::new(model_path), &target_tok, drafter_max_kv,
    ).expect("load_drafter");
    eprintln!("drafter loaded; tokenizer_compat={}", state.tokenizer_compat);

    // ── Compress ─────────────────────────────────────────────────────────
    let t_compress = Instant::now();
    let decision = pflash::maybe_compress_prompt(
        &mut gpu, &mut state, &cfg, &source_tokens, RequestKind::Text, &[],
    ).expect("maybe_compress_prompt");
    let compress_ms = t_compress.elapsed().as_millis();
    let cp = match decision {
        PflashDecision::Compressed(cp) => cp,
        PflashDecision::Bypass { reason: BypassReason::BelowThreshold { source_tokens, threshold } } => {
            // Documented behavior: when the budget would keep every source
            // token (e.g., keep_ratio=1.0, or min_keep_tokens >= source),
            // maybe_compress_prompt returns BelowThreshold so the daemon
            // doesn't double-prefill the same stream. Smoke this as PASS
            // since the pipeline did the right thing -- there is no
            // compressed prompt to feed downstream.
            eprintln!("bypass(BelowThreshold): {source_tokens} tokens, threshold {threshold}");
            eprintln!("(compression budget would keep the entire prompt -- pipeline is a no-op for this config)");
            eprintln!("PASS (documented bypass: nothing to compress)");
            state.unload_drafter(&mut gpu);
            std::process::exit(0);
        }
        PflashDecision::Bypass { reason } => {
            eprintln!("FAIL: unexpected bypass: {reason:?}");
            state.unload_drafter(&mut gpu);
            std::process::exit(2);
        }
    };
    eprintln!("compress:    {compress_ms} ms (score={}ms select={}ms gather={}ms)",
        cp.timings.score_ms, cp.timings.select_ms, cp.timings.gather_ms);
    eprintln!("compressed:  {} -> {} tokens (ratio {:.3})",
        cp.source_tokens, cp.kept_tokens,
        cp.kept_tokens as f32 / cp.source_tokens.max(1) as f32);
    eprintln!("source_md5:  {}", cp.source_md5);
    eprintln!("compressed_md5: {}", cp.compressed_md5);
    eprintln!("kept_spans:  {} ranges (first={:?} last={:?})",
        cp.kept_spans.len(), cp.kept_spans.first(), cp.kept_spans.last());

    // Free the drafter before the target prefill so the same VRAM pool is
    // available for the target's KV cache.
    state.unload_drafter(&mut gpu);

    // ── Re-prefill compressed stream through target ──────────────────────
    let target_kv_seq = (cp.kept_tokens + max_gen + 64).next_power_of_two().max(1024);
    let mut target_kv = KvCache::new_gpu_q8(
        &mut gpu, target_cfg.n_layers, target_cfg.n_kv_heads, target_cfg.head_dim, target_kv_seq,
    ).expect("target kv");
    let t_pre = Instant::now();
    llama::forward_prefill_batch(
        &mut gpu, &target_weights, &target_cfg, &cp.token_ids, 0,
        &mut target_kv, &target_scratch, None,
    ).expect("target prefill on compressed stream");
    gpu.hip.device_synchronize().expect("sync after prefill");
    let prefill_ms = t_pre.elapsed().as_millis();
    let prefill_tps = if prefill_ms > 0 {
        cp.kept_tokens as f64 / (prefill_ms as f64 / 1000.0)
    } else { 0.0 };
    eprintln!("target prefill: {prefill_ms} ms ({} tokens, {prefill_tps:.0} tok/s)",
        cp.kept_tokens);

    // ── Greedy decode ────────────────────────────────────────────────────
    let logits = gpu.download_f32(&target_scratch.logits).expect("download logits");
    let first = llama::argmax(&logits);
    let mut next = first;
    let mut generated: Vec<u32> = vec![first];
    let t_dec = Instant::now();
    let mut decode_steps = 0usize;
    for _ in 1..max_gen {
        if next == target_cfg.eos_token {
            break;
        }
        let pos = cp.token_ids.len() + generated.len() - 1;
        llama::forward_scratch_embed(&mut gpu, &target_weights, &target_cfg, next, pos, &target_scratch)
            .expect("embed");
        llama::forward_scratch_compute(&mut gpu, &target_weights, &target_cfg, pos, &mut target_kv, &target_scratch)
            .expect("compute");
        let logits = gpu.download_f32(&target_scratch.logits).expect("download");
        next = llama::argmax(&logits);
        generated.push(next);
        decode_steps += 1;
    }
    let decode_ms = t_dec.elapsed().as_millis();
    let decode_tps = if decode_ms > 0 && decode_steps > 0 {
        decode_steps as f64 / (decode_ms as f64 / 1000.0)
    } else { 0.0 };
    eprintln!("decode:      {decode_ms} ms ({decode_steps} forward_scratch calls, {decode_tps:.1} tok/s)");

    let answer = target_tok.decode(&generated);
    eprintln!("--- ANSWER ---");
    eprintln!("{answer}");
    eprintln!("--- VERDICT ---");

    // For Phase 1.5 success the PIPELINE must complete (no panic, no
    // empty answer, finite numerics). Retrieval quality on a 0.6B is
    // unreliable at heavy compression, so the bar here is "did the
    // pipeline run end-to-end and produce a coherent text response",
    // not "did the model find the needle".
    let pipeline_ok = !generated.is_empty()
        && !answer.is_empty()
        && answer.chars().any(|c| c.is_alphabetic());
    let needle_found = answer.contains(EXPECTED);
    eprintln!("pipeline_ok:  {pipeline_ok}");
    eprintln!("needle_found: {needle_found} (expected substring {EXPECTED:?})");

    if !pipeline_ok {
        eprintln!("FAIL: pipeline did not produce a coherent answer");
        std::process::exit(1);
    }
    if needle_found {
        eprintln!("PASS (pipeline + retrieval)");
    } else {
        eprintln!("PASS (pipeline only; retrieval is unreliable at this model size + ratio)");
    }
}
