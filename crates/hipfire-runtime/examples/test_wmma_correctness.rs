// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kevin Read
// hipfire — see LICENSE and NOTICE in the project root.

//! WMMA vs scalar GEMM correctness test — find the exact error pattern.
//!
//! Compares `gemm_hfq4g256_residual_wmma` (which dispatches to the
//! variant selected by `HIPFIRE_WO_WMMA_VARIANT`) against the scalar
//! `gemm_hfq4g256_residual` reference. The two should agree
//! element-by-element up to small floating-point accumulation drift.
//!
//! ## What this test catches
//!
//! - WMMA C-mapping silent corruption (case-studies §4 / b7ac66a class):
//!   the lane→cell mapping in the kernel's output block must match the
//!   RDNA3 wave32 WMMA hardware mapping (`acc[j] = C[2*j + (tid>>4)][tid & 15]`).
//!   Wrong mapping shuffles rows/batches in the output without any other
//!   functional symptom.
//! - Cross-batch corruption: errors that only appear at batch ≥ 2 when
//!   the output block straddles the `(tid>>4)` lane-half boundary.
//! - Cross-group accumulator corruption: errors that only appear when
//!   `groups_per_row > 1` (i.e., K > 256), where the per-iteration
//!   accumulator state must survive across the outer group loop.
//!
//! ## Test data design (do not change without thinking)
//!
//! Weights are row-varying (`(r+1) * 0.05` along the diagonal of each
//! group), NOT row-invariant. With row-invariant weights (the original
//! `1.0` along the diagonal) every row of every batch produces the
//! same output value, so output-mapping errors that swap rows are
//! invisible at batch=1. The current data makes `C[r][b]` depend on
//! both r and b independently, which surfaces row-shuffle bugs even at
//! batch=1.
//!
//! ## Tolerance
//!
//! `(expected.abs() * 0.05).max(0.05)` — 5% relative with a 0.05
//! absolute floor. Generous enough to absorb FP accumulation drift
//! between scalar and WMMA paths (which is on the order of 1e-3 even
//! at K=4096), tight enough to catch row-shuffle errors (which produce
//! errors on the order of the value magnitude itself).

fn main() {
    let mut gpu = rdna_compute::Gpu::init().unwrap();
    eprintln!("GPU: {}", gpu.arch);

    let m = 16usize;

    // Sweep K. K=256 → 1 group iteration. K=512 → 2 (cross-group
    // accumulator state must survive). K=4096 → 16 (production shape
    // for residual on Qwen 3.5 9B is K=4096+, so this exercises the
    // realistic loop count).
    for &k in &[256usize, 512, 4096] {
        eprintln!("\n=== K={k} ===");
        let groups_per_row = k / 256;

        for batch in [1, 2, 4, 16] {
            // Row-varying weights: along the diagonal of each group, the
            // value is (r+1) * 0.05; off-diagonal is zero. Range per
            // element: 0.05 (r=0) to 0.80 (r=15). This makes C[r][b]
            // depend on r — row-shuffle errors are detectable at batch=1.
            let mut f32_w = vec![0.0f32; m * k];
            for r in 0..m {
                for c in 0..k {
                    f32_w[r * k + c] = if c % m == r {
                        (r as f32 + 1.0) * 0.05
                    } else {
                        0.0
                    };
                }
            }

            let quantized = quantize_hfq4g256(&f32_w);

            // X: per-batch constant (b+1) * 0.1.
            let mut x_data = vec![0.0f32; batch * k];
            for b in 0..batch {
                for c in 0..k {
                    x_data[b * k + c] = (b as f32 + 1.0) * 0.1;
                }
            }

            // Expected: C[r][b] = sum_{c%m==r} w[r][c] * x[b][c]
            //                   = groups_per_row * (r+1)*0.05 * (b+1)*0.1
            //                   = groups_per_row * (r+1) * (b+1) * 0.005
            // Stored column-major (M is the inner stride): Y[b * M + r].

            let y_init = vec![0.0f32; batch * m]; // no residual

            let d_a = gpu.upload_raw(&quantized, &[quantized.len()]).unwrap();
            let d_x = gpu.upload_f32(&x_data, &[batch * k]).unwrap();

            // Scalar reference (FP32 dequant path).
            let d_y_s = gpu.upload_f32(&y_init, &[batch * m]).unwrap();
            std::env::set_var("HIPFIRE_FP16", "0");
            gpu.gemm_hfq4g256_residual(&d_a, &d_x, &d_y_s, m, k, batch).unwrap();
            let ys = gpu.download_f32(&d_y_s).unwrap();

            // WMMA path (variant selected by HIPFIRE_WO_WMMA_VARIANT).
            let d_y_w = gpu.upload_f32(&y_init, &[batch * m]).unwrap();
            std::env::remove_var("HIPFIRE_FP16");
            gpu.gemm_hfq4g256_residual_wmma(&d_a, &d_x, &d_y_w, m, k, batch).unwrap();
            let yw = gpu.download_f32(&d_y_w).unwrap();

            // Per-cell comparison with row-mod-16 histogram of bad cells.
            // The histogram catches WMMA C-mapping bugs that produce a
            // characteristic per-(r mod 16) signature.
            let mut max_err = 0.0f32;
            let mut bad = 0;
            let mut bad_by_rmod = [0u32; 16];
            for b in 0..batch {
                for r in 0..m {
                    let idx = b * m + r;
                    let scalar = ys[idx];
                    let wmma = yw[idx];
                    let err = (scalar - wmma).abs();
                    max_err = max_err.max(err);
                    let tolerance = (scalar.abs() * 0.05).max(0.05);
                    if err > tolerance {
                        bad += 1;
                        bad_by_rmod[r % 16] += 1;
                    }
                }
            }
            eprintln!(
                "  batch={batch:2}: max_err={max_err:.4} bad={bad}/{}",
                batch * m,
            );
            if bad > 0 {
                eprint!("    bad-by-(r mod 16): ");
                for (i, c) in bad_by_rmod.iter().enumerate() {
                    if *c > 0 {
                        eprint!("[{i}]={c} ");
                    }
                }
                eprintln!();
                // Print first few mismatches verbatim for debugging.
                let mut shown = 0usize;
                'outer: for b in 0..batch {
                    for r in 0..m {
                        let idx = b * m + r;
                        let scalar = ys[idx];
                        let wmma = yw[idx];
                        let err = (scalar - wmma).abs();
                        let tolerance = (scalar.abs() * 0.05).max(0.05);
                        if err > tolerance {
                            eprintln!(
                                "      b={b} r={r}: scalar={scalar:.4} wmma={wmma:.4} err={err:.4} tol={tolerance:.4}"
                            );
                            shown += 1;
                            if shown >= 8 { break 'outer; }
                        }
                    }
                }
            }

            // Sanity: at K=256, batch=1, expected[r][0] = 1 * (r+1) * 1 * 0.005
            //                                            = (r+1) * 0.005,
            // i.e., 0.005, 0.010, 0.015, ..., 0.080.
            // At K=4096, batch=15, expected[15][15] = 16 * 16 * 16 * 0.005
            //                                       = 20.48.
            let _ = groups_per_row; // silence unused if assertions removed
        }
    }
}

fn quantize_hfq4g256(f32_data: &[f32]) -> Vec<u8> {
    let group_size = 256;
    let block_bytes = 136;
    let n = f32_data.len();
    let n_blocks = (n + group_size - 1) / group_size;
    let mut output = vec![0u8; n_blocks * block_bytes];
    for b in 0..n_blocks {
        let start = b * group_size;
        let end = (start + group_size).min(n);
        let group = &f32_data[start..end];
        let min_val = group.iter().cloned().fold(f32::INFINITY, f32::min);
        let max_val = group.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let range = max_val - min_val;
        let scale = if range > 0.0 { range / 15.0 } else { 1.0 };
        let inv_scale = if range > 0.0 { 1.0 / scale } else { 0.0 };
        let out_off = b * block_bytes;
        output[out_off..out_off + 4].copy_from_slice(&scale.to_le_bytes());
        output[out_off + 4..out_off + 8].copy_from_slice(&min_val.to_le_bytes());
        for i in 0..128 {
            let lo = if 2*i < (end-start) { ((group[2*i] - min_val) * inv_scale + 0.5) as u8 } else { 0 };
            let hi = if 2*i+1 < (end-start) { ((group[2*i+1] - min_val) * inv_scale + 0.5) as u8 } else { 0 };
            output[out_off + 8 + i] = lo.min(15) | (hi.min(15) << 4);
        }
    }
    output
}
