// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Correctness check for the routed RMSNorm kernel.
//!
//! On gfx1151 this exercises `rmsnorm_f32_gfx1151`; on other arches it still
//! compares the generic route against the same CPU reference.

use rdna_compute::{DType, Gpu};

fn rmsnorm_ref(x: &[f32], weight: &[f32], n: usize, eps: f32) -> Vec<f32> {
    let batch = x.len() / n;
    let mut out = vec![0.0f32; x.len()];
    for b in 0..batch {
        let row = &x[b * n..(b + 1) * n];
        let mut sum_sq = 0.0f32;
        for &v in row {
            sum_sq += v * v;
        }
        let inv_rms = (sum_sq / n as f32 + eps).sqrt().recip();
        for i in 0..n {
            out[b * n + i] = row[i] * weight[i] * inv_rms;
        }
    }
    out
}

fn check_close(label: &str, got: &[f32], expected: &[f32]) {
    let mut max_abs = 0.0f32;
    let mut max_rel = 0.0f32;
    let mut n_finite = 0usize;
    for (&g, &e) in got.iter().zip(expected.iter()) {
        if g.is_finite() {
            n_finite += 1;
        }
        let abs = (g - e).abs();
        max_abs = max_abs.max(abs);
        max_rel = max_rel.max(abs / e.abs().max(1.0e-6));
    }
    eprintln!(
        "{label}: n_finite={n_finite}/{} max_abs={max_abs:.6e} max_rel={max_rel:.6e}",
        got.len()
    );
    assert_eq!(n_finite, got.len(), "{label}: non-finite output");
    assert!(
        max_abs <= 4.0e-6,
        "{label}: max_abs {max_abs:.6e} exceeds tolerance"
    );
}

fn main() {
    let mut gpu = Gpu::init().unwrap();
    eprintln!("GPU: {}", gpu.arch);

    for &(batch, n) in &[(1usize, 128usize), (1, 4096), (7, 4096), (3, 12288)] {
        eprintln!("\n=== batch={batch} n={n} ===");
        let x: Vec<f32> = (0..batch * n)
            .map(|i| ((i * 17 + 11) % 251) as f32 / 67.0 - 1.4)
            .collect();
        let weight: Vec<f32> = (0..n)
            .map(|i| 0.75 + ((i * 13 + 5) % 101) as f32 / 311.0)
            .collect();
        let expected = rmsnorm_ref(&x, &weight, n, 1.0e-6);

        let d_x = gpu.upload_f32(&x, &[batch, n]).unwrap();
        let d_w = gpu.upload_f32(&weight, &[n]).unwrap();
        let d_out = gpu.zeros(&[batch, n], DType::F32).unwrap();

        if batch == 1 {
            gpu.rmsnorm_f32(&d_x, &d_w, &d_out, 1.0e-6).unwrap();
        } else {
            gpu.rmsnorm_batched(&d_x, &d_w, &d_out, batch, n, 1.0e-6)
                .unwrap();
        }

        let got = gpu.download_f32(&d_out).unwrap();
        check_close(&format!("batch={batch} n={n}"), &got, &expected);

        gpu.free_tensor(d_x).unwrap();
        gpu.free_tensor(d_w).unwrap();
        gpu.free_tensor(d_out).unwrap();
    }
}
