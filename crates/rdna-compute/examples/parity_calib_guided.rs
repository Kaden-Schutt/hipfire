// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 hipfire contributors
// hipfire — see LICENSE and NOTICE in the project root.
//! Parity for the GuidedQuant calibration primitives vs a CPU oracle:
//!   calib_row_meansq_f32:            w[n]   = (1/Kd) Σ_c d[n,c]²
//!   calib_hessian_outer_weighted_f32 H[i,j] = Σ_n w[n]·x[n,i]·x[n,j]
//! Together these form the per-token Fisher-weighted Hessian H̄ = Xᵀ diag(w) X
//! from a linear's input X[N,Kx] and its output-grad d[N,Kd].
//!
//!   cargo run --release -p rdna-compute --example parity_calib_guided [N Kx Kd]

use rdna_compute::Gpu;

fn lcgf(seed: u32, n: usize) -> Vec<f32> {
    let mut s = seed.max(1);
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(1_103_515_245).wrapping_add(12345) & 0x7fff_ffff;
            -1.0 + (s as f32 / 2_147_483_648.0) * 2.0
        })
        .collect()
}
fn up(gpu: &mut Gpu, v: &[f32], shape: &[usize]) -> rdna_compute::GpuTensor {
    gpu.upload_raw(
        &v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>(),
        shape,
    )
    .unwrap()
}

fn main() {
    let mut a = std::env::args().skip(1);
    let n: usize = a.next().and_then(|s| s.parse().ok()).unwrap_or(48);
    let kx: usize = a.next().and_then(|s| s.parse().ok()).unwrap_or(64);
    let kd: usize = a.next().and_then(|s| s.parse().ok()).unwrap_or(96);

    let x = lcgf(7, n * kx); // activations [N, Kx]
    let d = lcgf(101, n * kd); // output grads [N, Kd]

    let mut gpu = Gpu::init().unwrap();
    let xd = up(&mut gpu, &x, &[n, kx]);
    let dd = up(&mut gpu, &d, &[n, kd]);
    let w = gpu.upload_raw(&vec![0u8; n * 4], &[n]).unwrap();
    let h = gpu.upload_raw(&vec![0u8; kx * kx * 4], &[kx, kx]).unwrap();

    gpu.calib_row_meansq_f32(&dd, &w, n, kd).unwrap();
    gpu.calib_hessian_outer_weighted_f32(&xd, &w, &h, n, kx)
        .unwrap();
    gpu.device_synchronize().unwrap();
    let wv = gpu.download_f32(&w).unwrap();
    let hv = gpu.download_f32(&h).unwrap();

    // CPU oracle.
    let mut w_ref = vec![0.0f32; n];
    for nn in 0..n {
        let mut s = 0.0f32;
        for c in 0..kd {
            let v = d[nn * kd + c];
            s += v * v;
        }
        w_ref[nn] = s / kd as f32;
    }
    let mut w_err = 0.0f32;
    for nn in 0..n {
        w_err = w_err.max((wv[nn] - w_ref[nn]).abs());
    }

    let mut h_err = 0.0f32;
    let mut h_mag = 0.0f32;
    for i in 0..kx {
        for j in 0..kx {
            let mut acc = 0.0f32;
            for nn in 0..n {
                acc += w_ref[nn] * x[nn * kx + i] * x[nn * kx + j];
            }
            let got = hv[i * kx + j];
            h_err = h_err.max((got - acc).abs());
            h_mag = h_mag.max(acc.abs());
        }
    }
    let tol = 1e-3 * h_mag.max(1.0);
    let pass = w_err <= 1e-4 && h_err <= tol;
    println!(
        "parity_calib_guided N={n} Kx={kx} Kd={kd} on {}: w_err={w_err:.6} h_err={h_err:.5} (mag={h_mag:.2}) -> {}",
        gpu.arch,
        if pass { "PASS" } else { "FAIL" }
    );
    if !pass {
        std::process::exit(1);
    }
}
