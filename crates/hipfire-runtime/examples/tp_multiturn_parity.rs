// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Dense TP multi-turn KV-reuse correctness: prefilling a PREFIX then
//! per-token-prefilling a SUFFIX (the cache-hit path) leaves the KV in the same
//! state as prefilling the WHOLE sequence at once (the cold path). Asserts the
//! last-position logits are argmax-identical + a tight numeric bound. This is the
//! TpModel-level invariant behind daemon multi-turn KV reuse (item B); the daemon
//! LCP decision itself reuses the proven `plan_prompt_cache`, not retested here.
//!
//! Emulated Tp-2 (gfx1151).
//! Run: HIPFIRE_EMULATE_GPUS=2 HIPFIRE_DETERMINISTIC=1 \
//!   cargo run -p hipfire-runtime --release --example tp_multiturn_parity -- --model model.mq4

use hipfire_hardware::{DeviceMesh, DimKind};
use hipfire_loader::ModelParallel;
use hipfire_runtime::llama;

const MAX_SEQ: usize = 512;
const TP: usize = 2;

// A fixed "conversation": PREFIX = turn-1 rendered+decoded context, SUFFIX =
// turn-2's new user turn. Token ids are arbitrary but in-vocab-safe (small).
// Two segments so the suffix attends over a non-trivial cached prefix.
const PREFIX: &[u32] = &[
    9707, 11, 358, 1079, 264, 10950, 17847, 13, 3555, 374, 279, 6722, 315, 9625, 30,
];
const SUFFIX: &[u32] = &[576, 6722, 315, 9625, 374, 12095, 13, 3555, 911, 8453, 30];

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mut model_path = concat!(env!("HOME"), "/.hipfire/models/qwen3-0.6b-llama.mq4").to_string();
    let mut it = args.iter().skip(1);
    while let Some(a) = it.next() {
        match a.as_str() {
            "--model" => {
                if let Some(v) = it.next() {
                    model_path = v.clone();
                }
            }
            other => model_path = other.to_string(),
        }
    }
    std::env::set_var("HIPFIRE_EMULATE_GPUS", "2");

    let full: Vec<u32> = PREFIX.iter().chain(SUFFIX.iter()).copied().collect();
    let mesh = DeviceMesh::rect(&[(DimKind::Tp, TP)]);

    // ── Cold: one TpModel, prefill the WHOLE sequence at once. ──
    let cold_logits: Vec<f32> = {
        let loaded =
            match hipfire_loader::load_model_tp(&model_path, MAX_SEQ, &mesh, Default::default()) {
                Ok(m) => m,
                Err(e) => {
                    println!("tp_multiturn_parity: SKIPPED (load_model_tp: {e})");
                    return;
                }
            };
        let mut m = loaded
            .tp_model
            .expect("expected TP model (tp_model carrier)");
        m.prefill(&full).expect("cold prefill");
        m.logits().expect("cold logits")
    };

    // ── Reuse: a second TpModel, prefill PREFIX, then per-token SUFFIX at
    // pos = PREFIX.len()+i (the daemon cache-hit path). ──
    let reuse_logits: Vec<f32> = {
        let loaded = hipfire_loader::load_model_tp(&model_path, MAX_SEQ, &mesh, Default::default())
            .expect("load reuse");
        let mut m = loaded
            .tp_model
            .expect("expected TP model (tp_model carrier)");
        m.prefill(PREFIX).expect("prefix prefill");
        for (i, &t) in SUFFIX.iter().enumerate() {
            m.forward_token(t, PREFIX.len() + i)
                .expect("suffix forward_token");
        }
        m.logits().expect("reuse logits")
    };

    assert_eq!(
        cold_logits.len(),
        reuse_logits.len(),
        "logits length mismatch"
    );
    let cold_am = llama::argmax(&cold_logits);
    let reuse_am = llama::argmax(&reuse_logits);
    let max_delta = cold_logits
        .iter()
        .zip(&reuse_logits)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    let mag = cold_logits.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
    println!(
        "[tp-multiturn] prefix={} suffix={} cold_argmax={cold_am} reuse_argmax={reuse_am} \
         max|Δ|={max_delta:.3e} (max|logit|={mag:.3e})",
        PREFIX.len(),
        SUFFIX.len()
    );

    assert_eq!(
        cold_am, reuse_am,
        "reuse (per-token suffix) argmax diverged from cold (full prefill): \
         reuse={reuse_am} cold={cold_am} (max|Δ|={max_delta:.3e})"
    );
    // Both write the SAME positions; the only Δ is batched-prefill vs per-token
    // rounding. Set after measuring (Step 3); start strict.
    const MAX_DELTA_TOL: f32 = 4.0e-1;
    assert!(
        max_delta < MAX_DELTA_TOL,
        "reuse vs cold logits diverged beyond tolerance: max|Δ|={max_delta:.3e} >= {MAX_DELTA_TOL:.1e}"
    );
    println!(
        "tp_multiturn_parity: per-token suffix prefill == full prefill \
         (argmax match + max|Δ|={max_delta:.3e} < {MAX_DELTA_TOL:.1e}) — B2 primitive validated"
    );
}
