// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! gpu-vs-cpu correctness for the nemotron_h ReLU² MLP block (`-`).
//! Compares `MlpRelu2Gpu::forward` (gemv → relu2 → gemv) against the CPU oracle
//! `mlp::mlp_relu2`. Uses non-pow2 dims to also exercise the gemv_f32 fix.
//!
//!   hipfire lock acquire test_mlp_gpu --watch-pid $$
//!   cargo run -p hipfire-arch-nemotron --example test_mlp_gpu

use hipfire_arch_nemotron::mlp::{mlp_relu2, MlpRelu2Gpu};
use hipfire_rdna::Gpu;

fn main() {
    let mut gpu = Gpu::init().expect("GPU init failed");
    eprintln!("GPU: {}", gpu.arch);

    let hidden = 24usize; // non-pow2
    let intermediate = 40usize; // non-pow2

    let mut seed = 0x7A1F_9C3Du32;
    let mut rng = || {
        seed ^= seed << 13;
        seed ^= seed >> 17;
        seed ^= seed << 5;
        (seed as f32 / u32::MAX as f32) * 0.4 - 0.2
    };
    let up: Vec<f32> = (0..intermediate * hidden).map(|_| rng()).collect();
    let down: Vec<f32> = (0..hidden * intermediate).map(|_| rng()).collect();

    let mut mlp = MlpRelu2Gpu::new(&mut gpu, hidden, intermediate, &up, &down).unwrap();

    let mut worst = 0.0f32;
    for step in 0..4 {
        let x: Vec<f32> = (0..hidden).map(|_| rng()).collect();
        let cpu = mlp_relu2(&up, &down, &x, hidden, intermediate);

        let d_x = gpu.upload_f32(&x, &[hidden]).unwrap();
        let out_t = mlp.forward(&mut gpu, &d_x).unwrap();
        gpu.hip.device_synchronize().unwrap();
        let g = gpu.download_f32(out_t).unwrap();

        let md = g
            .iter()
            .zip(&cpu)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        eprintln!("step {step}: max|Δ|={md:.3e}");
        worst = worst.max(md);
    }
    if worst > 1e-4 {
        eprintln!("FAIL (worst={worst:.3e})");
        std::process::exit(1);
    }
    println!("PASS: MlpRelu2Gpu matches CPU oracle");
}
