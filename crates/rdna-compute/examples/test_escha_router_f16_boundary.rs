// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kevin Read
// hipfire — see LICENSE and NOTICE in the project root.

//! Task 9 review Fix 1 — proof that `gpu.router_logits_round_f16_rne`
//! actually changes MoE top-k SELECTION on a constructed boundary case,
//! on BOTH selection routes (`moe_router_softmax_topk_k8_wave64_exact`,
//! the production gfx1151 kernel, and the reference two-launch
//! `softmax_f32` + `moe_topk_renorm_k8` pair).
//!
//! Construction: 256 router logits.
//!   - experts 0..6: distinct large values (20.0 down to 14.0) — always
//!     in the top-8, on either side of rounding.
//!   - expert A_IDX (50): value `v_a`.
//!   - expert B_IDX (200): value `v_b`, the LARGEST f32 strictly greater
//!     than `v_a` that still rounds to `v_a`'s f16 bit pattern (found by
//!     walking `v_a`'s bit pattern up one ULP at a time and checking
//!     `half::f16::from_f32` — no hand-derived f16 arithmetic).
//!   - every other expert: -50.0 (never competitive).
//!
//! Raw f32 order: v_b > v_a, so unrounded top-8 = {0..6, B_IDX} and A_IDX
//! is the (correctly) excluded 9th-place expert.
//!
//! Every kernel on this path (`moe_router_softmax_topk_k8_wave64_exact`'s
//! `ROUTER_EXACT_CONSIDER`/`router_exact_wave32_chunk_argmax`, and
//! `moe_topk_renorm_k8`'s `r4_topk_pair_reduce`) picks a new candidate only
//! on a STRICT `>` — so on an exact tie the lower index always wins. Once
//! `v_a` and `v_b` are rounded to the identical f16-widened f32 value, that
//! tie-break flips the winner from B_IDX (higher index) to A_IDX (lower
//! index): rounded top-8 = {0..6, A_IDX}, and B_IDX is now excluded.
//!
//! This is exactly the class of divergence the background describes:
//! EschaLabs selects top-k from f16(logits) widened back to f32; hipfire's
//! default path selects top-k from full-precision F32 logits. The escha-only
//! `router_logits_round_f16_rne` step makes hipfire's escha decode path
//! reproduce Escha's selection; every other model's `run_moe_decode` call
//! never invokes it and keeps the current (raw F32) selection untouched.
//!
//! Run: `cargo run --release -p rdna-compute --example test_escha_router_f16_boundary`

use rdna_compute::{DType, Gpu};

const N_EXP: usize = 256;
const TOP_K: usize = 8;
const A_IDX: usize = 50;
const B_IDX: usize = 200;

fn main() {
    // ── Construct the boundary pair on the host, byte-exact, no hand f16 math ──
    let v_a: f32 = 3.0;
    let h_a = half::f16::from_f32(v_a);
    let mut bits = v_a.to_bits();
    loop {
        let candidate = f32::from_bits(bits + 1);
        if half::f16::from_f32(candidate) != h_a {
            break;
        }
        bits += 1;
    }
    let v_b = f32::from_bits(bits);
    assert!(v_b > v_a, "expected v_b > v_a, got v_b={v_b} v_a={v_a}");
    assert_eq!(
        half::f16::from_f32(v_a),
        half::f16::from_f32(v_b),
        "v_a={v_a} and v_b={v_b} must round to the identical f16 value"
    );
    println!(
        "boundary pair: v_a(idx {A_IDX})={v_a:.9} v_b(idx {B_IDX})={v_b:.9} \
         both round to f16 {:#06x} ({:.9})",
        h_a.to_bits(),
        h_a.to_f32()
    );

    let mut logits = vec![-50.0f32; N_EXP];
    for i in 0..7 {
        logits[i] = 20.0 - i as f32;
    }
    logits[A_IDX] = v_a;
    logits[B_IDX] = v_b;

    // Host oracle: "escha selection" = top-8 by f32(f16(logit)), stable on
    // ties in ascending-index order (matches every kernel's strict `>`
    // tie-break: lower index wins). Independent of any GPU kernel.
    let mut rounded: Vec<(usize, f32)> = logits
        .iter()
        .enumerate()
        .map(|(i, &v)| (i, half::f16::from_f32(v).to_f32()))
        .collect();
    rounded.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    let mut oracle_escha_set: Vec<usize> = rounded[..TOP_K].iter().map(|(i, _)| *i).collect();
    oracle_escha_set.sort_unstable();

    let mut raw: Vec<(usize, f32)> = logits.iter().enumerate().map(|(i, &v)| (i, v)).collect();
    raw.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    let mut oracle_raw_set: Vec<usize> = raw[..TOP_K].iter().map(|(i, _)| *i).collect();
    oracle_raw_set.sort_unstable();

    assert!(
        oracle_raw_set.contains(&B_IDX) && !oracle_raw_set.contains(&A_IDX),
        "sanity: raw f32 oracle should include B_IDX={B_IDX} and exclude A_IDX={A_IDX}, got {oracle_raw_set:?}"
    );
    assert!(
        oracle_escha_set.contains(&A_IDX) && !oracle_escha_set.contains(&B_IDX),
        "sanity: f16-rounded oracle should include A_IDX={A_IDX} and exclude B_IDX={B_IDX}, got {oracle_escha_set:?}"
    );
    assert_ne!(
        oracle_raw_set, oracle_escha_set,
        "constructed case failed to produce a differing SET at the host-oracle level"
    );
    println!("host oracle: raw-F32 top-8 set    = {oracle_raw_set:?}");
    println!("host oracle: f16-rounded top-8 set = {oracle_escha_set:?}");

    // ── GPU: exact-wave64 kernel (production gfx1151/gfx1100 route) ──
    let mut gpu = Gpu::init().expect("GPU init");
    let unrounded_exact = run_wave64_exact(&mut gpu, &logits, false);
    let rounded_exact = run_wave64_exact(&mut gpu, &logits, true);
    println!("GPU exact-wave64 kernel: unrounded set = {unrounded_exact:?}");
    println!("GPU exact-wave64 kernel: rounded set   = {rounded_exact:?}");
    assert_eq!(
        unrounded_exact, oracle_raw_set,
        "exact-wave64 kernel's unrounded selection must match the raw-F32 oracle \
         (this is what every non-escha model gets today, unchanged)"
    );
    assert_eq!(
        rounded_exact, oracle_escha_set,
        "exact-wave64 kernel's rounded selection must match the f16-widened oracle \
         (this is what the escha decode path now gets)"
    );
    assert_ne!(
        unrounded_exact, rounded_exact,
        "router_logits_round_f16_rne had no effect on the exact-wave64 route"
    );

    // ── GPU: reference two-launch fallback (softmax_f32 + moe_topk_renorm_k8) ──
    let unrounded_fallback = run_softmax_fallback(&mut gpu, &logits, false);
    let rounded_fallback = run_softmax_fallback(&mut gpu, &logits, true);
    println!("GPU softmax+renorm fallback: unrounded set = {unrounded_fallback:?}");
    println!("GPU softmax+renorm fallback: rounded set   = {rounded_fallback:?}");
    assert_eq!(
        unrounded_fallback, oracle_raw_set,
        "fallback pair's unrounded selection must match the raw-F32 oracle"
    );
    assert_eq!(
        rounded_fallback, oracle_escha_set,
        "fallback pair's rounded selection must match the f16-widened oracle"
    );
    assert_ne!(
        unrounded_fallback, rounded_fallback,
        "router_logits_round_f16_rne had no effect on the softmax+renorm fallback route"
    );

    println!(
        "PASS: router_logits_round_f16_rne flips the top-8 SET at the constructed \
         boundary (A_IDX={A_IDX} in, B_IDX={B_IDX} out) on both selection routes, \
         matching the f16-widened oracle; the unrounded routes are byte-identical \
         to the pre-existing raw-F32 selection."
    );
}

/// Runs `moe_router_softmax_topk_k8_wave64_exact`, optionally rounding the
/// logits through `router_logits_round_f16_rne` first (mirrors exactly what
/// `run_moe_decode` now does when `MoeDtypes::has_escha_experts()` is true).
/// Returns the selected top-8 expert indices, sorted ascending.
fn run_wave64_exact(gpu: &mut Gpu, logits: &[f32], round_f16: bool) -> Vec<usize> {
    let logits_gpu = gpu.upload_f32(logits, &[N_EXP]).expect("upload logits");
    if round_f16 {
        gpu.router_logits_round_f16_rne(&logits_gpu)
            .expect("round logits to f16");
    }
    let idx = gpu.zeros(&[TOP_K], DType::F32).expect("idx tensor");
    let w = gpu.zeros(&[TOP_K], DType::F32).expect("weight tensor");
    gpu.moe_router_softmax_topk_k8_wave64_exact(&logits_gpu, &idx, &w, N_EXP, true)
        .expect("exact wave64 router");
    let set = download_index_set(gpu, &idx);
    gpu.free_tensor(logits_gpu).expect("free logits");
    gpu.free_tensor(idx).expect("free idx");
    gpu.free_tensor(w).expect("free w");
    set
}

/// Runs the reference two-launch fallback (`softmax_f32` + `moe_topk_renorm_k8`),
/// optionally rounding first. Same contract as `run_wave64_exact`.
fn run_softmax_fallback(gpu: &mut Gpu, logits: &[f32], round_f16: bool) -> Vec<usize> {
    let logits_gpu = gpu.upload_f32(logits, &[N_EXP]).expect("upload logits");
    if round_f16 {
        gpu.router_logits_round_f16_rne(&logits_gpu)
            .expect("round logits to f16");
    }
    let idx = gpu.zeros(&[TOP_K], DType::F32).expect("idx tensor");
    let w = gpu.zeros(&[TOP_K], DType::F32).expect("weight tensor");
    gpu.softmax_f32(&logits_gpu).expect("softmax");
    gpu.moe_topk_renorm_k8(&logits_gpu, &idx, &w, N_EXP, true)
        .expect("topk renorm");
    let set = download_index_set(gpu, &idx);
    gpu.free_tensor(logits_gpu).expect("free logits");
    gpu.free_tensor(idx).expect("free idx");
    gpu.free_tensor(w).expect("free w");
    set
}

/// Downloads a `[TOP_K]` "i32-in-F32 alias" index buffer (the same
/// bit-reinterpretation convention `moe_ffn_decode_impl::capture_expert_stats`
/// and `escha_router_topk_for_test` use) and returns it as a sorted `Vec<usize>`.
fn download_index_set(gpu: &Gpu, idx: &rdna_compute::GpuTensor) -> Vec<usize> {
    let idx_f32 = gpu.download_f32(idx).expect("download idx");
    let mut set: Vec<usize> = idx_f32
        .iter()
        .map(|v| (v.to_bits() as i32) as usize)
        .collect();
    set.sort_unstable();
    set
}
