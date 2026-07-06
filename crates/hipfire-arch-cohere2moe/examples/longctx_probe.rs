// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! longctx_probe (cohere2moe) — pin where North's long-context forward collapses
//! into a degenerate `<PAD>` / single-token attractor (observed in a Pi session
//! at ~30K tokens, where the model emitted `<PAD>` until the client aborted).
//!
//! It runs ONE continuous BATCHED prefill (`forward_batch`, chunk 256 — exactly
//! the daemon's path) over a long token stream, and at each checkpoint length L
//! reports the LAST-position next-token distribution:
//!   argmax id + decoded string, max softmax prob, entropy (nats), is_pad.
//! A healthy length yields a real token + moderate entropy; a collapsed length
//! argmaxes to `<PAD>` with ~0 entropy. The onset L is where is_pad flips true.
//!
//! `--per-token` switches to `decode_step` (independent per-position prefill) to
//! test whether the collapse is specific to the batched path or shared.
//!
//! Usage:
//!   longctx_probe --model <model.hfq> --text <file.txt> \
//!       [--lengths 4096,8192,16384,24576,32000] [--chunk 256] [--per-token]

use hipfire_arch_cohere2moe::cohere2moe::{Cohere2MoeState, Cohere2MoeWeights};
use hipfire_arch_cohere2moe::config::Cohere2MoeConfig;
use hipfire_arch_cohere2moe::forward::{decode_step, forward_batch, forward_batch_supported};
use hipfire_runtime::hfq::HfqFile;
use hipfire_runtime::tokenizer::Tokenizer;
use std::path::Path;

fn arg(flag: &str) -> Option<String> {
    let a: Vec<String> = std::env::args().collect();
    a.iter()
        .position(|x| x == flag)
        .and_then(|i| a.get(i + 1).cloned())
}
fn has(flag: &str) -> bool {
    std::env::args().any(|x| x == flag)
}

/// Softmax stats of a logit row: (argmax_id, max_prob, entropy_nats, top5).
fn analyze(logits: &[f32]) -> (u32, f32, f32, Vec<(u32, f32)>) {
    let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f64;
    for &v in logits {
        sum += ((v - max) as f64).exp();
    }
    let mut ent = 0.0f64;
    let mut probs: Vec<(u32, f32)> = Vec::with_capacity(logits.len());
    for (i, &v) in logits.iter().enumerate() {
        let p = (((v - max) as f64).exp() / sum) as f32;
        if p > 0.0 {
            ent -= (p as f64) * (p as f64).ln();
        }
        probs.push((i as u32, p));
    }
    probs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    let argmax = probs[0].0;
    let pmax = probs[0].1;
    (
        argmax,
        pmax,
        ent as f32,
        probs.into_iter().take(5).collect(),
    )
}

fn main() {
    let model = arg("--model").expect("--model required");
    let text_path = arg("--text").expect("--text required");
    let chunk: usize = arg("--chunk").map(|s| s.parse().unwrap()).unwrap_or(256);
    let per_token = has("--per-token");
    let mut lengths: Vec<usize> = arg("--lengths")
        .unwrap_or_else(|| "4096,8192,16384,24576,32000".to_string())
        .split(',')
        .map(|s| s.trim().parse().unwrap())
        .collect();
    lengths.sort_unstable();
    lengths.dedup();

    let mut gpu = rdna_compute::Gpu::init().expect("gpu init");
    let mut hfq = HfqFile::open(Path::new(&model)).expect("open model");
    assert_eq!(hfq.arch_id, 12, "expected arch_id 12, got {}", hfq.arch_id);
    let cfg = Cohere2MoeConfig::from_hfq(&hfq).expect("config");
    let tok = Tokenizer::from_hfq_metadata(&hfq.metadata_json).expect("tokenizer");
    let weights = Cohere2MoeWeights::load(&mut hfq, &cfg, &mut gpu).expect("weights");
    let supported = forward_batch_supported(&weights);
    let pad = tok.special_token_id("<PAD>");
    let bos = tok.special_token_id("<BOS_TOKEN>").unwrap_or(2);

    // Tokenize text; prepend BOS; tile to reach the longest checkpoint.
    let text = std::fs::read_to_string(&text_path).expect("read text");
    let body = tok.encode(&text);
    let want = *lengths.iter().max().unwrap();
    let mut toks: Vec<u32> = Vec::with_capacity(want + chunk);
    toks.push(bos);
    while toks.len() < want {
        toks.extend_from_slice(&body);
    }
    toks.truncate(want);
    let use_batch = supported && !per_token;
    eprintln!(
        "[probe] text={} tokens, tiled to {} | BOS={bos} PAD={:?} | batched_supported={supported} | mode={}",
        body.len(), toks.len(), pad,
        if use_batch { "forward_batch" } else { "decode_step(per-token)" }
    );

    let mut state = Cohere2MoeState::new_with_max_seq(&mut gpu, &cfg, want + 16).expect("state");
    let _ = state.reset(&mut gpu);

    println!(
        "# mode={}  chunk={chunk}  (entropy in nats; healthy≫0, collapsed≈0)",
        if use_batch { "batched" } else { "per-token" }
    );
    println!("#     L    argmax    pmax  entropy  is_pad  argmax_decoded  | top5(id:prob)");

    let mut cp_idx = 0usize;
    let mut i = 0usize;
    let mut last: Vec<f32> = Vec::new();
    while i < want && cp_idx < lengths.len() {
        let cp = lengths[cp_idx];
        let end = if use_batch {
            (i + chunk).min(cp)
        } else {
            i + 1
        };
        let start_pos = state.n_tokens;
        if use_batch {
            last = forward_batch(
                &cfg,
                &weights,
                &mut state,
                &mut gpu,
                &toks[i..end],
                start_pos,
            )
            .expect("forward_batch");
        } else {
            last = decode_step(&cfg, &weights, &mut state, &mut gpu, toks[i], i as u32)
                .expect("decode_step");
        }
        i = end;
        if i >= cp {
            let (argmax, pmax, ent, _top5) = analyze(&last);
            let is_pad = Some(argmax) == pad;
            let dec = tok.decode(&[argmax]);
            // The ACTUAL next token of the (tiled) stream. If argmax == this, the
            // model is just COPYING the repeated doc (correct induction-head
            // behavior on the 2nd pass) — NOT a degenerate collapse.
            let actual = toks.get(cp).copied();
            let actual_dec = actual.map(|t| tok.decode(&[t])).unwrap_or_default();
            let copying = actual == Some(argmax);
            println!(
                "  {:>6}  argmax={:>8} {:>10?}  pmax={:.4} ent={:>6.3} is_pad={:<5} | actual_next={:?} COPYING={}",
                cp, argmax, dec, pmax, ent, is_pad, actual_dec, copying
            );
            cp_idx += 1;
        }
    }
}
