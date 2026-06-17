// SPDX-License-Identifier: Apache-2.0
// hipfire — correctness test for the Tier-1 calibration reduction kernels.
//
//! Verifies `calib_sumsq_reduce_f32` (per-column Σx²) and
//! `calib_hessian_outer_f32` (Σxxᵀ) against a CPU reference, including the
//! accumulate-in-place contract (two calls must sum). No model/forward/daemon.
//!
//! Run: cargo run --release -p rdna-compute --example test_calib_reduce

use rdna_compute::DType;

fn fract_sin(x: f32) -> f32 {
    (x.sin() * 12345.6789f32).fract() * 2.0f32 - 1.0f32
}

fn main() {
    let mut gpu = rdna_compute::Gpu::init().unwrap();
    eprintln!("GPU: {}", gpu.arch);
    let mut any_fail = false;

    // Non-tile-aligned shapes to exercise the boundary guards (K%16≠0, N%16≠0).
    for &(n, k) in &[(8usize, 16usize), (40, 100), (37, 257)] {
        eprintln!("\n=== N={n} K={k} ===");
        // Two activation blocks — we accumulate both to test the += contract.
        let x1: Vec<f32> = (0..n * k)
            .map(|i| fract_sin(i as f32 * 0.731 + 1.0))
            .collect();
        let x2: Vec<f32> = (0..n * k)
            .map(|i| fract_sin(i as f32 * 0.517 + 9.0))
            .collect();

        // CPU reference (accumulated over both blocks).
        let mut sumsq_ref = vec![0.0f32; k];
        let mut h_ref = vec![0.0f32; k * k];
        for x in [&x1, &x2] {
            for row in 0..n {
                for c in 0..k {
                    sumsq_ref[c] += x[row * k + c] * x[row * k + c];
                }
                for i in 0..k {
                    let xi = x[row * k + i];
                    for j in 0..k {
                        h_ref[i * k + j] += xi * x[row * k + j];
                    }
                }
            }
        }

        // GPU: zero the accumulators once, then two accumulate calls.
        let d_acc = gpu.zeros(&[k], DType::F32).unwrap();
        let d_h = gpu.zeros(&[k, k], DType::F32).unwrap();
        for x in [&x1, &x2] {
            let d_x = gpu.upload_f32(x, &[n, k]).unwrap();
            gpu.calib_sumsq_reduce_f32(&d_x, &d_acc, n, k).unwrap();
            gpu.calib_hessian_outer_f32(&d_x, &d_h, n, k).unwrap();
        }
        let sumsq_gpu = gpu.download_f32(&d_acc).unwrap();
        let h_gpu = gpu.download_f32(&d_h).unwrap();

        let mut max_s = 0.0f32;
        for c in 0..k {
            max_s = max_s.max((sumsq_gpu[c] - sumsq_ref[c]).abs());
        }
        let mut max_h = 0.0f32;
        for i in 0..k * k {
            max_h = max_h.max((h_gpu[i] - h_ref[i]).abs());
        }
        // fp32 accumulation over N rows; tolerance scaled by N.
        let tol = 1e-3 * (n as f32);
        let ok_s = max_s <= tol;
        let ok_h = max_h <= tol;
        any_fail |= !(ok_s && ok_h);
        eprintln!(
            "  sumsq max|Δ|={max_s:.3e} [{}]   hessian max|Δ|={max_h:.3e} [{}]  (tol {tol:.1e})",
            if ok_s { "PASS" } else { "FAIL" },
            if ok_h { "PASS" } else { "FAIL" }
        );
        // Spot-check Hessian symmetry (H = xᵀx is symmetric).
        let mut max_asym = 0.0f32;
        for i in 0..k {
            for j in 0..k {
                max_asym = max_asym.max((h_gpu[i * k + j] - h_gpu[j * k + i]).abs());
            }
        }
        eprintln!("  hessian symmetry max|H-Hᵀ|={max_asym:.3e}");
    }

    if any_fail {
        eprintln!("\n[FAIL] calibration reduction kernels disagree with CPU reference.");
        std::process::exit(1);
    }
    eprintln!("\n[PASS] calib_sumsq_reduce_f32 + calib_hessian_outer_f32 match CPU (accumulate-in-place OK).");
}
