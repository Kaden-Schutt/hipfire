// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! gpu-vs-cpu correctness for `relu2_f32` (out = max(0,x)^2), nemotron_h MLP act.

use rdna_compute::{DType, Gpu};

fn main() {
    let mut gpu = Gpu::init().expect("GPU init failed");
    eprintln!("GPU: {}", gpu.arch);
    let n = 4096usize;
    let mut seed = 0x9E3779B9u32;
    let x: Vec<f32> = (0..n)
        .map(|_| {
            seed ^= seed << 13;
            seed ^= seed >> 17;
            seed ^= seed << 5;
            (seed as f32 / u32::MAX as f32) * 6.0 - 3.0 // [-3,3]
        })
        .collect();
    let cpu: Vec<f32> = x.iter().map(|&v| v.max(0.0) * v.max(0.0)).collect();

    let d_x = gpu.upload_f32(&x, &[n]).unwrap();
    let d_out = gpu.zeros(&[n], DType::F32).unwrap();
    gpu.relu2_f32(&d_x, &d_out).unwrap();
    gpu.hip.device_synchronize().unwrap();
    let out = gpu.download_f32(&d_out).unwrap();

    let max_diff = out
        .iter()
        .zip(&cpu)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    eprintln!("max|Δ|={max_diff:.3e}");
    if max_diff > 1e-5 {
        eprintln!("FAIL");
        std::process::exit(1);
    }
    println!("PASS: relu2_f32 matches CPU reference");
}
