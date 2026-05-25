// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kevin Read
// hipfire — see LICENSE and NOTICE in the project root.

//! Correctness test for `gemm_hfq4g256_residual_wave64_dp4a` (gfx906).
//!
//! Compares the new HFQ4 batched dp4a residual GEMM (issue #276 Gap 2)
//! against `gemm_hfq4g256_residual_fp16_wave64` — the kernel it replaces
//! at gfx906 B>1 below the MMQ cutover.
//!
//! Both kernels:
//!   y[b,row] += A[row] · x[b]
//! starting from the same non-zero Y. Differences come from Q8_1 vs FP16
//! quantization noise; NRMSE should land at ~0.2% (Q8_1×HFQ4 floor),
//! matching `test_gfx906_mmq_correctness`'s observed band.
//!
//! Usage: cargo run --release -p rdna-compute --example test_hfq4_residual_dp4a \
//!        -- [M] [K] [N]
//!
//! Defaults: M=128 K=256 N=8 (matches BATCH_TILE=16 boundary at N≤16).

use rdna_compute::{DType, Gpu};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let m: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(128);
    let k: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(256);
    let n: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(8);

    assert!(k % 256 == 0, "K must be a multiple of 256");

    let groups_per_row = k / 256;
    let row_bytes = groups_per_row * 136;

    eprintln!("=== gfx906 HFQ4 residual_wave64_dp4a correctness test ===");
    eprintln!("M={m} K={k} N={n}");

    let mut gpu = Gpu::init().expect("gpu init");
    eprintln!("arch: {}", gpu.arch);

    if gpu.arch != "gfx906" {
        eprintln!("WARNING: this test is only meaningful on gfx906; skipping");
        std::process::exit(0);
    }

    let weight_bytes = synth_hfq4g256_weights(m, groups_per_row, 0xC0DE_FACEu64);
    let a_raw = gpu
        .upload_raw(&weight_bytes, &[m * row_bytes])
        .expect("upload weights");

    let x_host: Vec<f32> = (0..n * k)
        .map(|i| {
            let v = ((i as i64).wrapping_mul(1103515245).wrapping_add(12345)) as f32;
            (v * 1e-9) % 2.0 - 1.0
        })
        .collect();
    let x_tensor = gpu.upload_f32(&x_host, &[n * k]).expect("upload x");

    // Non-zero starting Y so the residual `+=` is observable.
    let y_init_host: Vec<f32> = (0..n * m)
        .map(|i| {
            let v = ((i as i64).wrapping_mul(2147483647).wrapping_add(7)) as f32;
            (v * 1e-7) % 1.0
        })
        .collect();

    let y_dp4a = gpu.upload_f32(&y_init_host, &[n * m]).expect("alloc y_dp4a");
    let y_fp16 = gpu.upload_f32(&y_init_host, &[n * m]).expect("alloc y_fp16");

    let n_iter = std::env::var("HFQ_TEST_N_ITER")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(1);
    eprintln!("running {n_iter} GEMM iteration(s) on the same Y buffer");

    eprintln!("\n--- gemm_hfq4g256_residual_fp16_wave64 (reference) ---");
    for _ in 0..n_iter {
        gpu.gemm_hfq4g256_residual_fp16_wave64(&a_raw, &x_tensor, &y_fp16, m, k, n)
            .expect("fp16 wave64 launch");
    }
    gpu.hip.device_synchronize().expect("sync after fp16");

    eprintln!("--- gemm_hfq4g256_residual_wave64_dp4a (under test) ---");
    for _ in 0..n_iter {
        gpu.gemm_hfq4g256_residual_wave64_dp4a(&a_raw, &x_tensor, &y_dp4a, m, k, n)
            .expect("dp4a launch");
    }
    gpu.hip.device_synchronize().expect("sync after dp4a");

    let fp16_out = gpu.download_f32(&y_fp16).expect("download fp16");
    let dp4a_out = gpu.download_f32(&y_dp4a).expect("download dp4a");

    eprintln!("\n--- Comparing outputs ---");
    let mut max_abs_err = 0.0f32;
    let mut max_rel_err = 0.0f32;
    let mut sum_sq_err = 0.0f64;
    let mut sum_sq_ref = 0.0f64;
    let mut worst_idx = 0usize;
    let mut worst_pair = (0.0f32, 0.0f32);
    for i in 0..n * m {
        let r = fp16_out[i];
        let q = dp4a_out[i];
        let err = (r - q).abs();
        if err > max_abs_err {
            max_abs_err = err;
            worst_idx = i;
            worst_pair = (r, q);
        }
        let rel = if r.abs() > 1e-6 { err / r.abs() } else { 0.0 };
        if rel > max_rel_err {
            max_rel_err = rel;
        }
        sum_sq_err += (err as f64).powi(2);
        sum_sq_ref += (r as f64).powi(2);
    }

    let rms_err = (sum_sq_err / (n * m) as f64).sqrt() as f32;
    let rms_ref = (sum_sq_ref / (n * m) as f64).sqrt() as f32;
    let nrmse = rms_err / rms_ref.max(1e-12);

    let ref_min = fp16_out.iter().copied().fold(f32::INFINITY, f32::min);
    let ref_max = fp16_out.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let dp_min = dp4a_out.iter().copied().fold(f32::INFINITY, f32::min);
    let dp_max = dp4a_out.iter().copied().fold(f32::NEG_INFINITY, f32::max);

    let worst_col = worst_idx / m;
    let worst_row = worst_idx % m;
    eprintln!("max_abs_err  = {:.6e}", max_abs_err);
    eprintln!("max_rel_err  = {:.4}%", max_rel_err * 100.0);
    eprintln!("rms_err      = {:.6e}", rms_err);
    eprintln!("rms_ref      = {:.6e}", rms_ref);
    eprintln!("NRMSE        = {:.4}%", nrmse * 100.0);
    eprintln!("worst (col,row) = ({worst_col}, {worst_row})");
    eprintln!("                  fp16={:.6e}  dp4a={:.6e}", worst_pair.0, worst_pair.1);
    eprintln!("ref range:  [{ref_min:.4e}, {ref_max:.4e}]");
    eprintln!("dp4a range: [{dp_min:.4e}, {dp_max:.4e}]");

    eprintln!("\n--- First 16 output cells (col=0, rows=0..15) ---");
    for i in 0..16.min(m) {
        eprintln!("  row {i}: fp16={:.6e}  dp4a={:.6e}  diff={:.6e}",
            fp16_out[i], dp4a_out[i], (fp16_out[i] - dp4a_out[i]).abs());
    }

    // Pass criteria:
    //  - NRMSE < 1e-2 (1%) for Q8_1×HFQ4 quantization noise (matches the
    //    `test_gfx906_mmq_correctness` tolerance — same input format).
    //  - dp4a output is non-zero (catches "kernel did nothing" bug).
    //  - dp4a output differs from y_init (catches "no residual write" bug).
    let dp4a_nonzero = dp4a_out.iter().any(|&v| v.abs() > 1e-12);
    let dp4a_wrote_residual = dp4a_out
        .iter()
        .zip(y_init_host.iter())
        .any(|(o, init)| (o - init).abs() > 1e-6);
    let pass = nrmse < 1e-2 && dp4a_nonzero && dp4a_wrote_residual;
    if pass {
        eprintln!("\nPASS (NRMSE within tolerance, residual write observed)");
        std::process::exit(0);
    } else {
        eprintln!("\nFAIL");
        if !dp4a_nonzero {
            eprintln!("  dp4a output is all-zero — kernel may not have run");
        }
        if !dp4a_wrote_residual {
            eprintln!("  dp4a output matches y_init — residual `+=` did not fire");
        }
        if nrmse >= 1e-2 {
            eprintln!("  NRMSE {:.4}% exceeds 1% threshold", nrmse * 100.0);
        }
        std::process::exit(1);
    }
}

/// Same generator as test_gfx906_mmq_correctness.rs. Reproduced here to
/// keep the example standalone.
fn synth_hfq4g256_weights(m: usize, groups_per_row: usize, seed: u64) -> Vec<u8> {
    let total = m * groups_per_row * 136;
    let mut out = vec![0u8; total];
    let mut state = seed;
    let mut next = || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (state >> 33) as u32
    };
    let scale_log10 = std::env::var("HFQ_TEST_SCALE_LOG10")
        .ok()
        .and_then(|s| s.parse::<f32>().ok())
        .unwrap_or(-3.0);
    let zp_max = std::env::var("HFQ_TEST_ZP_MAX")
        .ok()
        .and_then(|s| s.parse::<f32>().ok())
        .unwrap_or(1.0);
    let scale_target = 10.0f32.powf(scale_log10);
    for row in 0..m {
        for g in 0..groups_per_row {
            let gp = (row * groups_per_row + g) * 136;
            let scale = scale_target * (0.5 + (next() & 0xFFFF) as f32 / 65535.0 * 1.5);
            let zp = ((next() & 0xFFFF) as f32 / 65535.0) * 2.0 * zp_max - zp_max;
            out[gp..gp + 4].copy_from_slice(&scale.to_le_bytes());
            out[gp + 4..gp + 8].copy_from_slice(&zp.to_le_bytes());
            for i in 0..128 {
                out[gp + 8 + i] = (next() & 0xFF) as u8;
            }
        }
    }
    let _ = DType::HFQ4G256;
    out
}
