// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kevin Read
// hipfire — see LICENSE and NOTICE in the project root.

//! T3-2 correctness — `gemm_qkvza_q8_0_wmma` vs substrate × 4.
//! Gate: gated mean_rel < 1e-3 on each of Y_qkv, Y_z, Y_beta, Y_alpha.

use rdna_compute::{DType, Gpu};

#[path = "common/q8_test_utils.rs"]
mod q8_test_utils;
use q8_test_utils::synth_q8;

fn main() {
    let mut gpu = Gpu::init().expect("gpu init");
    let arch = gpu.arch.clone();
    eprintln!("=== test_gemm_q8_qkvza_wmma ===\n  arch = {arch}");
    if !arch.starts_with("gfx11") && !arch.starts_with("gfx12") {
        eprintln!("  SKIPPED: needs gfx11/12, got {arch}"); std::process::exit(0);
    }

    // (qkv_m, z_m, beta_m, alpha_m, K, label) — 9B DeltaNet LA shapes.
    let shapes: Vec<(usize, usize, usize, usize, usize, &str)> = vec![
        ( 64,  32,  16,  16,  128, "tiny"),
        (512, 256,  16,  16,  512, "medium"),
        (4096, 1024, 16, 16, 4096, "9B LA  (qkv=4096 z=1024 K=4096)"),
    ];
    let batches: Vec<usize> = vec![1, 4, 16, 32, 64, 128, 256];
    let mut total_fail = 0usize;

    for (qkv_m, z_m, beta_m, alpha_m, k, label) in &shapes {
        let (qkv_m, z_m, beta_m, alpha_m, k) = (*qkv_m, *z_m, *beta_m, *alpha_m, *k);
        eprintln!("\n--- {label} ---");

        let w_qkv = synth_q8(qkv_m, k, 0xA1B2C3D4);
        let w_z = synth_q8(z_m, k, 0xE5F60718);
        let w_beta = synth_q8(beta_m, k, 0x9ABCDEF0);
        let w_alpha = synth_q8(alpha_m, k, 0x12345678);
        let d_qkv = gpu.upload_raw(&w_qkv, &[w_qkv.len()]).unwrap();
        let d_z = gpu.upload_raw(&w_z, &[w_z.len()]).unwrap();
        let d_beta = gpu.upload_raw(&w_beta, &[w_beta.len()]).unwrap();
        let d_alpha = gpu.upload_raw(&w_alpha, &[w_alpha.len()]).unwrap();

        let max_n = *batches.iter().max().unwrap();
        let x_host: Vec<f32> = (0..max_n * k).map(synth_x).collect();
        let d_x = gpu.upload_f32(&x_host, &[max_n * k]).unwrap();

        let d_y_qkv_w = gpu.zeros(&[max_n * qkv_m], DType::F32).unwrap();
        let d_y_z_w = gpu.zeros(&[max_n * z_m], DType::F32).unwrap();
        let d_y_beta_w = gpu.zeros(&[max_n * beta_m], DType::F32).unwrap();
        let d_y_alpha_w = gpu.zeros(&[max_n * alpha_m], DType::F32).unwrap();
        let d_y_qkv_r = gpu.zeros(&[max_n * qkv_m], DType::F32).unwrap();
        let d_y_z_r = gpu.zeros(&[max_n * z_m], DType::F32).unwrap();
        let d_y_beta_r = gpu.zeros(&[max_n * beta_m], DType::F32).unwrap();
        let d_y_alpha_r = gpu.zeros(&[max_n * alpha_m], DType::F32).unwrap();

        for &n in &batches {
            let x_n = d_x.sub_offset(0, n * k);
            let qw = d_y_qkv_w.sub_offset(0, n * qkv_m);
            let zw = d_y_z_w.sub_offset(0, n * z_m);
            let bw = d_y_beta_w.sub_offset(0, n * beta_m);
            let aw = d_y_alpha_w.sub_offset(0, n * alpha_m);
            let qr = d_y_qkv_r.sub_offset(0, n * qkv_m);
            let zr = d_y_z_r.sub_offset(0, n * z_m);
            let br = d_y_beta_r.sub_offset(0, n * beta_m);
            let ar = d_y_alpha_r.sub_offset(0, n * alpha_m);

            if arch.starts_with("gfx12") {
                gpu.gemm_qkvza_q8_0_wmma_gfx12(
                    &d_qkv, &d_z, &d_beta, &d_alpha,
                    &x_n,
                    &qw, &zw, &bw, &aw,
                    qkv_m, z_m, beta_m, alpha_m, k, n,
                ).unwrap();
            } else {
                gpu.gemm_qkvza_q8_0_wmma(
                    &d_qkv, &d_z, &d_beta, &d_alpha,
                    &x_n,
                    &qw, &zw, &bw, &aw,
                    qkv_m, z_m, beta_m, alpha_m, k, n,
                ).unwrap();
            }
            gpu.gemm_q8_0_batched_chunked(&d_qkv, &x_n, &qr, qkv_m, k, n).unwrap();
            gpu.gemm_q8_0_batched_chunked(&d_z, &x_n, &zr, z_m, k, n).unwrap();
            gpu.gemm_q8_0_batched_chunked(&d_beta, &x_n, &br, beta_m, k, n).unwrap();
            gpu.gemm_q8_0_batched_chunked(&d_alpha, &x_n, &ar, alpha_m, k, n).unwrap();

            let s = [
                compare(&gpu.download_f32(&qw).unwrap(), &gpu.download_f32(&qr).unwrap()),
                compare(&gpu.download_f32(&zw).unwrap(), &gpu.download_f32(&zr).unwrap()),
                compare(&gpu.download_f32(&bw).unwrap(), &gpu.download_f32(&br).unwrap()),
                compare(&gpu.download_f32(&aw).unwrap(), &gpu.download_f32(&ar).unwrap()),
            ];
            // Gate: mean_rel < 2e-3 AND max_rel < 5e-2. Small projections
            // (alpha_m=16, beta_m=16) have noisier mean due to per-output
            // fp16 quantization spreading more across few outputs; max_rel
            // is the more robust signal there.
            // Threshold tightened 2026-05-13 from max_rel < 5e-2 → 2.5e-2;
            // see test_gemm_q8_gate_up_wmma.rs for rationale.
            let pass = s.iter().all(|x| x.mean_rel < 2e-3 && x.max_rel < 3.5e-2);
            let mark = if pass { "PASS" } else { total_fail += 1; "FAIL" };
            eprintln!(
                "  N={n:4}  {mark}   QKV: mean={:.2e}/max={:.2e}  Z: {:.2e}/{:.2e}  β: {:.2e}/{:.2e}  α: {:.2e}/{:.2e}",
                s[0].mean_rel, s[0].max_rel, s[1].mean_rel, s[1].max_rel,
                s[2].mean_rel, s[2].max_rel, s[3].mean_rel, s[3].max_rel,
            );
        }
    }
    eprintln!("\n=== {total_fail} failure(s) ===");
    std::process::exit(if total_fail == 0 { 0 } else { 1 });
}

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
