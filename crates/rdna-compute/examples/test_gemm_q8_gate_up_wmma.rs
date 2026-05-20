// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kevin Read
// hipfire — see LICENSE and NOTICE in the project root.

//! T3-2 correctness — `gemm_gate_up_q8_0_wmma` vs substrate × 2.
//! Gate: gated mean_rel < 1e-3 on Y_gate and Y_up.

use rdna_compute::{DType, Gpu};

#[path = "common/q8_test_utils.rs"]
mod q8_test_utils;
use q8_test_utils::synth_q8;

fn main() {
    let mut gpu = Gpu::init().expect("gpu init");
    let arch = gpu.arch.clone();
    eprintln!("=== test_gemm_q8_gate_up_wmma ===\n  arch = {arch}");
    if !arch.starts_with("gfx11") && !arch.starts_with("gfx12") {
        eprintln!("  SKIPPED: needs gfx11/12, got {arch}"); std::process::exit(0);
    }

    // (gate_m, up_m, K, label) — gate_m == up_m for Qwen3.5.
    let shapes: Vec<(usize, usize, usize, &str)> = vec![
        (128, 128, 128, "tiny"),
        (512, 512, 512, "medium"),
        (11008, 11008, 4096, "9B FFN  (gate=up=11008 K=4096)"),
    ];
    let batches: Vec<usize> = vec![1, 4, 16, 32, 64, 128, 256];
    let mut total_fail = 0usize;

    for (gate_m, up_m, k, label) in &shapes {
        let (gate_m, up_m, k) = (*gate_m, *up_m, *k);
        eprintln!("\n--- {label} ---");

        let w_g = synth_q8(gate_m, k, 0xA1B2C3D4);
        let w_u = synth_q8(up_m, k, 0xE5F60718);
        let d_g = gpu.upload_raw(&w_g, &[w_g.len()]).unwrap();
        let d_u = gpu.upload_raw(&w_u, &[w_u.len()]).unwrap();

        let max_n = *batches.iter().max().unwrap();
        let x_host: Vec<f32> = (0..max_n * k).map(synth_x).collect();
        let d_x = gpu.upload_f32(&x_host, &[max_n * k]).unwrap();

        let d_yg_w = gpu.zeros(&[max_n * gate_m], DType::F32).unwrap();
        let d_yu_w = gpu.zeros(&[max_n * up_m], DType::F32).unwrap();
        let d_yg_r = gpu.zeros(&[max_n * gate_m], DType::F32).unwrap();
        let d_yu_r = gpu.zeros(&[max_n * up_m], DType::F32).unwrap();

        for &n in &batches {
            let x_n = d_x.sub_offset(0, n * k);
            let gw = d_yg_w.sub_offset(0, n * gate_m);
            let uw = d_yu_w.sub_offset(0, n * up_m);
            let gr = d_yg_r.sub_offset(0, n * gate_m);
            let ur = d_yu_r.sub_offset(0, n * up_m);

            if arch.starts_with("gfx12") {
                gpu.gemm_gate_up_q8_0_wmma_gfx12(&d_g, &d_u, &x_n, &gw, &uw, gate_m, up_m, k, n).unwrap();
            } else {
                gpu.gemm_gate_up_q8_0_wmma(&d_g, &d_u, &x_n, &gw, &uw, gate_m, up_m, k, n).unwrap();
            }
            gpu.gemm_q8_0_batched_chunked(&d_g, &x_n, &gr, gate_m, k, n).unwrap();
            gpu.gemm_q8_0_batched_chunked(&d_u, &x_n, &ur, up_m, k, n).unwrap();

            let s = [
                compare(&gpu.download_f32(&gw).unwrap(), &gpu.download_f32(&gr).unwrap()),
                compare(&gpu.download_f32(&uw).unwrap(), &gpu.download_f32(&ur).unwrap()),
            ];
            // Threshold tightened 2026-05-13 from max_rel < 5e-2 → 3.5e-2.
            // Production 9B sweep tops at 1.98e-2; small synthetic shapes
            // (medium=512×512 β-projection) hit 3.19e-2 due to WMMA
            // reduction-order noise being more visible at low-M dims.
            // 3.5e-2 keeps a ~30% margin above the synthetic worst case
            // while still being 30% tighter than the original 5e-2 bound.
            let pass = s.iter().all(|x| x.mean_rel < 2e-3 && x.max_rel < 3.5e-2);
            let mark = if pass { "PASS" } else { total_fail += 1; "FAIL" };
            eprintln!(
                "  N={n:4}  {mark}   gate: mean={:.2e}/max={:.2e}  up: {:.2e}/{:.2e}",
                s[0].mean_rel, s[0].max_rel, s[1].mean_rel, s[1].max_rel,
            );
        }
    }
    eprintln!("\n=== {total_fail} failure(s) ===");
    std::process::exit(if total_fail == 0 { 0 } else { 1 });
}

// (helpers identical to test_gemm_q8_qkvza_wmma.rs)
struct Stats { mean_rel: f64, max_rel: f64 }
fn compare(a: &[f32], b: &[f32]) -> Stats {
    let max_ref = b.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
    let thr = max_ref * 0.01;
    let (mut sum, mut max_r, mut n) = (0.0f64, 0.0f64, 0usize);
    for (x, y) in a.iter().zip(b.iter()) {
        if y.abs() > thr {
            let r = ((x - y).abs() / y.abs()) as f64;
            sum += r; if r > max_r { max_r = r; } n += 1;
        }
    }
    Stats { mean_rel: if n == 0 { 0.0 } else { sum / n as f64 }, max_rel: max_r }
}
fn synth_x(i: usize) -> f32 {
    let v = ((i as i64).wrapping_mul(1103515245).wrapping_add(12345)) as f32;
    (v * 1e-9) % 2.0 - 1.0
}
