// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
//
// MQ6G256 batched GEMM correctness verification.
//
// Test A — GEMV equivalence: gemm_mq6g256_batched_lmhead at batch=1 must
//          agree with gemv_mq6g256_prerotated (the known-good oracle) on
//          identical bytes. Both run the same math; agreement should be at
//          fp32 rounding level.
// Test B — batch consistency: for batch in {1,2,4,8,16}, each output row
//          must match the single-token GEMV for that row's activation.
// Test C — shape coverage: exercise two real Gemma4-12B shapes plus a K
//          that is a multiple of 256 but not of 1024 (group-boundary check).

use rdna_compute::{DType, Gpu, GpuTensor};

/// Synthesize MQ6/HFQ6-G256 packed weights: per row, K/256 groups of
/// 200 bytes = [f32 scale][f32 zero][192 B packed 6-bit data (4 wts / 3 B)].
fn synth_mq6g256(m: usize, k: usize, seed: u64) -> Vec<u8> {
    assert!(k % 256 == 0, "mq6g256 requires K%256==0 (groups = K/256)");
    let groups = k / 256;
    let row_bytes = groups * 200;
    let mut out = vec![0u8; m * row_bytes];
    let mut state = seed;
    let mut rng = || -> u32 {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (state >> 33) as u32
    };
    for row in 0..m {
        for g in 0..groups {
            let off = row * row_bytes + g * 200;
            let sc: f32 = 0.003 + (rng() & 0x3F) as f32 * 1e-4;
            out[off..off + 4].copy_from_slice(&sc.to_bits().to_le_bytes());
            let zp: f32 = -0.02;
            out[off + 4..off + 8].copy_from_slice(&zp.to_bits().to_le_bytes());
            // 192 bytes of packed 6-bit data = 48 u32 words
            for w in 0..48 {
                let pk = rng();
                out[off + 8 + w * 4..off + 8 + w * 4 + 4].copy_from_slice(&pk.to_le_bytes());
            }
        }
    }
    out
}

fn make_x(n: usize, seed: u64) -> Vec<f32> {
    let mut state = seed;
    (0..n)
        .map(|_| {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((state >> 33) as f32) * 2.3e-10 - 0.5
        })
        .collect()
}

/// Compare two f32 slices and ENFORCE the bit-exactness contract.
///
/// This must assert, not merely report. Both kernels walk the same lanes over
/// the same weights in the same group order and reduce with the identical
/// `16,8,4,2,1` shuffle tree, so every output element is accumulated in the
/// same order and the results are required to be bit-identical — not
/// "close". A tolerance here would let a genuinely wrong kernel through.
///
/// NaN is checked explicitly: `NaN != NaN` is false and `NaN > max` is false,
/// so a NaN-producing kernel would otherwise sail past a max-error scan with
/// the maxima still reading 0.0.
fn max_err(a: &[f32], b: &[f32], ctx: &str) -> (f64, f64) {
    assert_eq!(a.len(), b.len(), "{ctx}: length mismatch");
    let mut max_abs = 0.0f64;
    let mut max_rel = 0.0f64;
    for (i, (&x, &y)) in a.iter().zip(b.iter()).enumerate() {
        assert!(
            x.is_finite() && y.is_finite(),
            "{ctx}: non-finite output at index {i}: got {x}, oracle {y}"
        );
        assert!(
            x.to_bits() == y.to_bits(),
            "{ctx}: MQ6 batched GEMM diverged from the GEMV oracle at index {i}: \
             got {x:e} ({:#010x}), oracle {y:e} ({:#010x})",
            x.to_bits(),
            y.to_bits()
        );
        let abs = (x as f64 - y as f64).abs();
        let denom = (y as f64).abs().max(1e-12);
        let rel = abs / denom;
        if abs > max_abs {
            max_abs = abs;
        }
        if rel > max_rel {
            max_rel = rel;
        }
    }
    (max_abs, max_rel)
}

fn run_shape(gpu: &mut Gpu, m: usize, k: usize, label: &str) {
    eprintln!("--- {label} (m={m}, k={k}, groups={}) ---", k / 256);

    // ── Weight upload ────────────────────────────────────────────────────
    let w_data = synth_mq6g256(m, k, 0x6D7136 ^ m as u64 ^ k as u64);
    let w = gpu.upload_raw(&w_data, &[w_data.len()]).expect("upload w");

    // ── Test A: batch=1 GEMV equivalence ─────────────────────────────────
    let x1_host = make_x(k, 0x1234);
    let x1 = gpu.upload_f32(&x1_host, &[k]).expect("upload x1");
    let y_gemv = gpu.alloc_tensor(&[m], DType::F32).expect("alloc y_gemv");
    let y_gemm1 = gpu.alloc_tensor(&[m], DType::F32).expect("alloc y_gemm1");

    gpu.gemv_mq6g256_prerotated(&w, &x1, &y_gemv, m, k)
        .expect("gemv_mq6g256_prerotated");
    gpu.gemm_mq6g256_batched_lmhead(&w, &x1, &y_gemm1, m, k, 1)
        .expect("gemm_mq6g256_batched b=1");

    let y_gemv_h = gpu.download_f32(&y_gemv).expect("download gemv");
    let y_gemm1_h = gpu.download_f32(&y_gemm1).expect("download gemm1");
    let (a_abs, a_rel) =
        max_err(&y_gemm1_h, &y_gemv_h, &format!("{label} TestA batch=1"));
    {
        let g = gpu.download_f32(&y_gemv).expect("dl gemv");
        let nz = g.iter().filter(|v| **v != 0.0).count();
        let mx = g.iter().fold(0.0f32, |a, b| a.max(b.abs()));
        eprintln!("     oracle out: {} / {} nonzero, max|y|={:.4e}", nz, g.len(), mx);
        // Guard the trivially-passing failure mode: if BOTH kernels emitted
        // zeros, every max_abs below would read 0.000e0 and the comparison
        // would prove nothing. Require the oracle to have done real work.
        assert!(nz == g.len(), "oracle produced {} zero outputs of {}", g.len() - nz, g.len());
        assert!(mx > 1.0, "oracle output magnitude {mx:.3e} too small to discriminate");
    }
    eprintln!("  A: batch=1 vs gemv   max_abs={a_abs:.3e}  max_rel={a_rel:.3e}");

    // ── Test B: batch consistency {1,2,4,8,16} ───────────────────────────
    for &b in &[1usize, 2, 3, 4, 5, 7, 8, 9, 12, 15, 16, 17] {
        // Build batched x: each row is an independent random vector
        let mut xb_host = Vec::with_capacity(b * k);
        for i in 0..b {
            xb_host.extend_from_slice(&make_x(k, 0x5000 + i as u64));
        }
        let xb = gpu.upload_f32(&xb_host, &[b, k]).expect("upload xb");
        let y_b = gpu.alloc_tensor(&[b, m], DType::F32).expect("alloc y_b");

        gpu.gemm_mq6g256_batched_lmhead(&w, &xb, &y_b, m, k, b)
            .expect("gemm batched");

        let y_b_h = gpu.download_f32(&y_b).expect("download y_b");

        // Reference: run the single-token GEMV for each row separately
        let mut worst_abs = 0.0f64;
        let mut worst_rel = 0.0f64;
        for i in 0..b {
            let xi_host = make_x(k, 0x5000 + i as u64);
            let xi = gpu.upload_f32(&xi_host, &[k]).expect("upload xi");
            let yi = gpu.alloc_tensor(&[m], DType::F32).expect("alloc yi");
            gpu.gemv_mq6g256_prerotated(&w, &xi, &yi, m, k)
                .expect("gemv row ref");
            let yi_h = gpu.download_f32(&yi).expect("download yi");
            let row_start = i * m;
            let (abs, rel) = max_err(
                &y_b_h[row_start..row_start + m],
                &yi_h,
                &format!("{label} TestB b={b} row={i}"),
            );
            if abs > worst_abs {
                worst_abs = abs;
            }
            if rel > worst_rel {
                worst_rel = rel;
            }
            gpu.free_tensor(xi).ok();
            gpu.free_tensor(yi).ok();
        }
        eprintln!("  B: batch={b:<2}           max_abs={worst_abs:.3e}  max_rel={worst_rel:.3e}");
        gpu.free_tensor(xb).ok();
        gpu.free_tensor(y_b).ok();
    }

    gpu.free_tensor(x1).ok();
    gpu.free_tensor(y_gemv).ok();
    gpu.free_tensor(y_gemm1).ok();
    gpu.free_tensor(w).ok();
    eprintln!();
}

fn main() {
    let mut gpu = Gpu::init().expect("gpu init");
    eprintln!("arch={}", gpu.arch);
    eprintln!();

    // Test C shapes: two real Gemma4-12B projections + one group-boundary K
    // Gemma4-12B: hidden=3840, ffn=15360 (dim=4*hidden with SwiGLU halving →
    //   v_proj:   m=2048 (n_kv_heads*head_dim), k=3840 (hidden)
    //   down_proj: m=3840 (hidden), k=15360/2*2=15360... use the actual FFN
    //   intermediate=15360 → down_proj k=15360? No: SwiGLU gate/up produce
    //   ffn_hidden of size ffn/2=7680 each; down takes 7680. But the real
    //   12B has ffn_dim=15360 meaning gate/up produce 7680 each... use the
    //   shapes the assignment names: m=2048/k=3840 and m=3840/k=15360.
    // Group-boundary K: 768 = 3×256, not a multiple of 1024.
    run_shape(&mut gpu, 2048, 3840, "gemma4-12b v_proj    m=2048  k=3840");
    run_shape(&mut gpu, 3840, 15360, "gemma4-12b down_proj m=3840  k=15360");
    run_shape(&mut gpu, 512, 768, "group-boundary         m=512   k=768  (K%256=0, K%1024≠0)");

    eprintln!("PASS: all shapes ran without error.");
}
