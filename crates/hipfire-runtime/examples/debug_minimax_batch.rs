// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Pinpoint forward_batch correctness: compare forward_batch (B tokens, one
//! pass) last-token logits to the matching sequential decode_step logits.
//! B=1 isolating a per-call/layout bug from a multi-row batching bug.

use hipfire_arch_minimax as minimax;
use hipfire_runtime::arch::Architecture;
use hipfire_runtime::hfq::HfqFile;
use std::path::Path;

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
fn cosine(a: &[f32], b: &[f32]) -> f64 {
    let (mut dot, mut na, mut nb) = (0.0f64, 0.0f64, 0.0f64);
    for i in 0..a.len().min(b.len()) {
        dot += a[i] as f64 * b[i] as f64;
        na += (a[i] as f64) * (a[i] as f64);
        nb += (b[i] as f64) * (b[i] as f64);
    }
    dot / (na.sqrt() * nb.sqrt() + 1e-12)
}

fn main() {
    let path = std::env::args()
        .nth(1)
        .expect("usage: debug_minimax_batch <model.hfq>");
    let mut gpu = rdna_compute::Gpu::init().expect("gpu init");
    let mut hfq = HfqFile::open(Path::new(&path)).expect("open model");
    let config = <minimax::MiniMaxM2 as Architecture>::config_from_hfq(&hfq).expect("config");
    let weights =
        <minimax::MiniMaxM2 as Architecture>::load_weights(&mut hfq, &config, &mut gpu).expect("weights");
    let mut state =
        minimax::MiniMaxState::new_with_max_seq(&mut gpu, &config, 4096).expect("state");

    let toks: Vec<u32> = vec![1u32, 1037, 2055, 3000, 410, 5202, 666, 7777];

    // Sequential reference.
    let mut seq_logits: Vec<Vec<f32>> = Vec::new();
    state.reset();
    for (i, &t) in toks.iter().enumerate() {
        let l = minimax::forward::decode_step(&config, &weights, &mut state, &mut gpu, t, i as u32)
            .expect("decode_step");
        seq_logits.push(l);
    }
    eprintln!("sequential done; vocab={}", seq_logits[0].len());

    for &b in &[1usize, 2, 4, 8] {
        state.reset();
        let bl = minimax::forward::forward_batch(&config, &weights, &mut state, &mut gpu, &toks[..b], 0)
            .expect("forward_batch");
        let am_b = argmax(&bl);
        let am_s = argmax(&seq_logits[b - 1]);
        let cos = cosine(&bl, &seq_logits[b - 1]);
        eprintln!(
            "B={:>2}: last-tok argmax batch={:>6} seq={:>6} match={:>5}  cosine={:.6}",
            b,
            am_b,
            am_s,
            am_b == am_s,
            cos
        );
    }
}
