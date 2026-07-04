#![allow(
    clippy::duplicated_attributes,
    clippy::doc_lazy_continuation,
    clippy::doc_overindented_list_items,
    clippy::explicit_counter_loop,
    clippy::field_reassign_with_default,
    clippy::manual_checked_ops,
    clippy::manual_clamp,
    clippy::manual_div_ceil,
    clippy::needless_range_loop,
    clippy::ptr_arg,
    clippy::same_item_push,
    clippy::too_many_arguments,
    clippy::type_complexity,
    clippy::unnecessary_cast,
    clippy::useless_vec,
    clippy::while_let_loop
)]
// hipfire example clippy sweep: examples are GPU probes/benches, not reusable APIs.

// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
//! LFM2.5 batched-prefill parity check.
//!
//! Runs the same prompt through:
//!   A) legacy per-token decode replay
//!   B) `prefill_batch`
//!
//! Then compares last-prompt logits and one forced continuation token. The
//! continuation check validates that batched prefill advanced both KV cache and
//! LIV conv state to the same trajectory as decode replay.
//!
//! Usage:
//!   cargo run -p hipfire-arch-lfm2moe --example prefill_parity_lfm2moe -- \
//!     --model <model.hfq> [--prompt <text>] [--tokens <tokens.json>] \
//!     [--capture-layers 2,5,8,10,13]

#[cfg(not(feature = "deltanet"))]
fn main() {
    eprintln!("build with --features deltanet");
}

#[cfg(feature = "deltanet")]
fn main() {
    use hipfire_arch_lfm2moe::config::Lfm2MoeConfig;
    use hipfire_arch_lfm2moe::forward::{
        decode_step, prefill_batch, prefill_batch_with_hidden, Lfm2HiddenCapture,
    };
    use hipfire_arch_lfm2moe::lfm2moe::{Lfm2MoeState, Lfm2MoeWeights};
    use hipfire_runtime::hfq::HfqFile;
    use hipfire_runtime::tokenizer::Tokenizer;
    use std::path::PathBuf;

    let argv: Vec<String> = std::env::args().collect();
    let mut model: Option<PathBuf> = None;
    let mut prompt = "The capital of France is".to_string();
    let mut tokens_path: Option<PathBuf> = None;
    let mut capture_layers: Option<Vec<usize>> = None;
    let mut i = 1;
    while i < argv.len() {
        match argv[i].as_str() {
            "--model" => {
                model = Some(PathBuf::from(&argv[i + 1]));
                i += 2;
            }
            "--prompt" => {
                prompt = argv[i + 1].clone();
                i += 2;
            }
            "--tokens" => {
                tokens_path = Some(PathBuf::from(&argv[i + 1]));
                i += 2;
            }
            "--capture-layers" => {
                capture_layers = Some(
                    argv[i + 1]
                        .split(',')
                        .filter(|s| !s.is_empty())
                        .map(|s| {
                            s.parse::<usize>()
                                .expect("--capture-layers expects usize CSV")
                        })
                        .collect(),
                );
                i += 2;
            }
            other => {
                eprintln!("unknown arg: {other}");
                std::process::exit(1);
            }
        }
    }
    let model = model.expect("--model required");

    let mut gpu = hipfire_rdna::Gpu::init().expect("gpu init");
    let mut hfq = HfqFile::open(&model).expect("open model");
    let cfg = Lfm2MoeConfig::from_hfq(&hfq).expect("config");
    eprintln!(
        "lfm2moe hidden={} layers={} experts={}/{} vocab={}",
        cfg.hidden_size,
        cfg.num_hidden_layers,
        cfg.num_experts,
        cfg.num_experts_per_tok,
        cfg.vocab_size
    );
    let tok = Tokenizer::from_hfq_metadata(&hfq.metadata_json).expect("tokenizer");
    let weights = Lfm2MoeWeights::load(&mut hfq, &cfg, &mut gpu).expect("weights");

    let prompt_ids: Vec<u32> = if let Some(path) = &tokens_path {
        let s = std::fs::read_to_string(path).expect("read --tokens");
        let v: Vec<i64> = serde_json::from_str(&s).expect("parse --tokens json");
        v.into_iter().map(|t| t as u32).collect()
    } else {
        tok.encode(&prompt)
    };
    if prompt_ids.len() < 2 {
        eprintln!("prompt must tokenize to at least two tokens for batched-prefill parity");
        std::process::exit(1);
    }
    eprintln!(
        "prompt {:?} -> {} tokens (src: {})",
        prompt,
        prompt_ids.len(),
        if tokens_path.is_some() {
            "--tokens"
        } else {
            "embedded tokenizer"
        }
    );

    let max_seq = prompt_ids.len() + 8;
    let mut state_ref = Lfm2MoeState::new_with_max_seq(&mut gpu, &cfg, max_seq).expect("state ref");
    let mut state_bat = Lfm2MoeState::new_with_max_seq(&mut gpu, &cfg, max_seq).expect("state bat");
    let mut state_cap = if capture_layers.is_some() {
        Some(Lfm2MoeState::new_with_max_seq(&mut gpu, &cfg, max_seq).expect("state cap"))
    } else {
        None
    };

    let mut ref_logits = Vec::new();
    for (pos, &token) in prompt_ids.iter().enumerate() {
        ref_logits = decode_step(&cfg, &weights, &mut state_ref, &mut gpu, token, pos as u32)
            .expect("reference decode_step");
    }
    let bat_logits = prefill_batch(&cfg, &weights, &mut state_bat, &mut gpu, &prompt_ids)
        .expect("prefill_batch");
    let cap_stats =
        if let (Some(layers), Some(state)) = (capture_layers.as_ref(), state_cap.as_mut()) {
            let mut cap =
                Lfm2HiddenCapture::new(cfg.num_hidden_layers, cfg.hidden_size, layers.clone())
                    .expect("hidden capture config");
            let cap_logits =
                prefill_batch_with_hidden(&cfg, &weights, state, &mut gpu, &prompt_ids, &mut cap)
                    .expect("prefill_batch_with_hidden");
            let expected = prompt_ids.len() * layers.len() * cfg.hidden_size;
            assert_eq!(cap.rows().len(), expected, "hidden capture row count");
            assert_eq!(cap.position_count(), prompt_ids.len());
            eprintln!(
                "hidden capture: layers={:?} positions={} floats={}",
                cap.target_layers(),
                cap.position_count(),
                cap.rows().len()
            );
            Some(compare("capture-prefill-last", &bat_logits, &cap_logits))
        } else {
            None
        };

    let ref_next = argmax(&ref_logits) as u32;
    let ref_cont = decode_step(
        &cfg,
        &weights,
        &mut state_ref,
        &mut gpu,
        ref_next,
        prompt_ids.len() as u32,
    )
    .expect("reference continuation");
    let bat_cont = decode_step(
        &cfg,
        &weights,
        &mut state_bat,
        &mut gpu,
        ref_next,
        prompt_ids.len() as u32,
    )
    .expect("batched continuation");

    let prompt_stats = compare("prompt-last", &ref_logits, &bat_logits);
    let cont_stats = compare("continuation", &ref_cont, &bat_cont);
    eprintln!(
        "forced continuation token={} decoded={:?}",
        ref_next,
        tok.decode(&[ref_next])
    );

    let pass = prompt_stats.argmax_match
        && cont_stats.argmax_match
        && cap_stats
            .map(|s| s.argmax_match && s.mean_abs <= 0.5 && s.max_abs <= 5.0)
            .unwrap_or(true)
        && prompt_stats.mean_abs <= 0.5
        && cont_stats.mean_abs <= 0.5
        && prompt_stats.max_abs <= 5.0
        && cont_stats.max_abs <= 5.0;
    if pass {
        println!("LFM2 PREFILL PARITY PASS");
    } else {
        println!("LFM2 PREFILL PARITY FAIL");
        std::process::exit(2);
    }
}

#[cfg(feature = "deltanet")]
#[derive(Debug, Clone, Copy)]
struct CompareStats {
    max_abs: f64,
    mean_abs: f64,
    cosine: f64,
    argmax_match: bool,
}

#[cfg(feature = "deltanet")]
fn argmax(v: &[f32]) -> usize {
    let mut bi = 0usize;
    let mut bv = f32::NEG_INFINITY;
    for (i, &x) in v.iter().enumerate() {
        if x > bv {
            bv = x;
            bi = i;
        }
    }
    bi
}

#[cfg(feature = "deltanet")]
fn compare(label: &str, a: &[f32], b: &[f32]) -> CompareStats {
    assert_eq!(a.len(), b.len(), "{label}: logits length mismatch");
    let mut dot = 0.0f64;
    let mut na = 0.0f64;
    let mut nb = 0.0f64;
    let mut sum_abs = 0.0f64;
    let mut max_abs = 0.0f64;
    for (&x, &y) in a.iter().zip(b.iter()) {
        let x = x as f64;
        let y = y as f64;
        dot += x * y;
        na += x * x;
        nb += y * y;
        let d = (x - y).abs();
        sum_abs += d;
        max_abs = max_abs.max(d);
    }
    let am_a = argmax(a);
    let am_b = argmax(b);
    let stats = CompareStats {
        max_abs,
        mean_abs: sum_abs / a.len() as f64,
        cosine: dot / (na.sqrt() * nb.sqrt()),
        argmax_match: am_a == am_b,
    };
    eprintln!(
        "{label}: cos={:.8} max|delta|={:.6e} mean|delta|={:.6e} argmax_ref={} argmax_batched={} {}",
        stats.cosine,
        stats.max_abs,
        stats.mean_abs,
        am_a,
        am_b,
        if stats.argmax_match { "" } else { "<<< ARGMAX MISMATCH" }
    );
    stats
}
