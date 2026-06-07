// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
//! hipGraph parity check for LFM2.5-MoE (HIPFIRE_LFM2_GRAPH).
//!
//! Runs the SAME token sequence twice on two independent decode states:
//!   A) direct path     (decode_step_inner via decode_step)
//!   B) hipGraph path   (decode_step_with_graph, warmup + capture + replay)
//! and reports per-position cosine + max-abs-delta between the full logits
//! vectors. Graph capture corrupts silently via stale kernarg scalars, so a
//! per-position logits cos >= 0.999999 (and argmax match) is the correctness
//! gate. Validation scaffold — not part of the shipped engine.
//!
//! Usage: graph_parity_lfm2moe --model <hfq> --tokens <tokens.json>

#[cfg(not(feature = "deltanet"))]
fn main() {
    eprintln!("build with --features deltanet");
}

#[cfg(feature = "deltanet")]
fn main() {
    use hipfire_arch_lfm2moe::config::Lfm2MoeConfig;
    use hipfire_arch_lfm2moe::forward::{decode_step, decode_step_with_graph};
    use hipfire_arch_lfm2moe::lfm2moe::{Lfm2MoeState, Lfm2MoeWeights};
    use hipfire_runtime::hfq::HfqFile;
    use std::path::PathBuf;

    let argv: Vec<String> = std::env::args().collect();
    let mut model: Option<PathBuf> = None;
    let mut tokens_path: Option<PathBuf> = None;
    let mut i = 1;
    while i < argv.len() {
        match argv[i].as_str() {
            "--model" => {
                model = Some(PathBuf::from(&argv[i + 1]));
                i += 2;
            }
            "--tokens" => {
                tokens_path = Some(PathBuf::from(&argv[i + 1]));
                i += 2;
            }
            other => {
                eprintln!("unknown arg: {other}");
                std::process::exit(1);
            }
        }
    }
    let model = model.expect("--model required");
    let tokens_path = tokens_path.expect("--tokens required");

    let tokens: Vec<u32> = {
        let s = std::fs::read_to_string(&tokens_path).expect("read tokens");
        let v: Vec<i64> = serde_json::from_str(&s).expect("parse tokens json");
        v.into_iter().map(|t| t as u32).collect()
    };
    let n = tokens.len();

    let mut gpu = rdna_compute::Gpu::init().expect("gpu init");
    let mut hfq = HfqFile::open(&model).expect("open model");
    let cfg = Lfm2MoeConfig::from_hfq(&hfq).expect("config");
    let weights = Lfm2MoeWeights::load(&mut hfq, &cfg, &mut gpu).expect("weights");

    // ---- Pass A: direct ----
    let mut state_a = Lfm2MoeState::new_with_max_seq(&mut gpu, &cfg, n + 16).expect("state A");
    let mut logits_a: Vec<Vec<f32>> = Vec::with_capacity(n);
    for (pos, &t) in tokens.iter().enumerate() {
        let l = decode_step(&cfg, &weights, &mut state_a, &mut gpu, t, pos as u32)
            .expect("direct decode_step");
        logits_a.push(l);
    }

    // ---- Pass B: hipGraph capture/replay (fresh state) ----
    let mut state_b = Lfm2MoeState::new_with_max_seq(&mut gpu, &cfg, n + 16).expect("state B");
    let mut logits_b: Vec<Vec<f32>> = Vec::with_capacity(n);
    for (pos, &t) in tokens.iter().enumerate() {
        let l = decode_step_with_graph(&cfg, &weights, &mut state_b, &mut gpu, t, pos as u32)
            .expect("graph decode_step_with_graph");
        logits_b.push(l);
    }

    let argmax = |v: &[f32]| -> usize {
        let mut bi = 0;
        let mut bv = f32::NEG_INFINITY;
        for (i, &x) in v.iter().enumerate() {
            if x > bv {
                bv = x;
                bi = i;
            }
        }
        bi
    };

    let mut min_cos = f64::INFINITY;
    let mut max_abs = 0f64;
    let mut argmax_mismatch = 0usize;
    println!("=== LFM2.5-MoE hipGraph parity (direct vs graph), {n} positions ===");
    for pos in 0..n {
        let a = &logits_a[pos];
        let b = &logits_b[pos];
        let (mut dot, mut na, mut nb, mut md) = (0f64, 0f64, 0f64, 0f64);
        for (x, y) in a.iter().zip(b.iter()) {
            let (x, y) = (*x as f64, *y as f64);
            dot += x * y;
            na += x * x;
            nb += y * y;
            md = md.max((x - y).abs());
        }
        let cos = dot / (na.sqrt() * nb.sqrt());
        let am_a = argmax(a);
        let am_b = argmax(b);
        if am_a != am_b {
            argmax_mismatch += 1;
        }
        min_cos = min_cos.min(cos);
        max_abs = max_abs.max(md);
        println!(
            "  pos {pos:2}: cos={cos:.8} max|delta|={md:.6e} argmax_direct={am_a} argmax_graph={am_b} {}",
            if am_a == am_b { "" } else { "<<< ARGMAX MISMATCH" }
        );
    }
    println!("--- min cos={min_cos:.8}  max|delta|={max_abs:.6e}  argmax_mismatches={argmax_mismatch}/{n} ---");
    if min_cos >= 0.999999 && argmax_mismatch == 0 {
        println!("GRAPH PARITY PASS");
    } else {
        println!("GRAPH PARITY FAIL");
        std::process::exit(2);
    }
}
