// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! gpu-vs-cpu correctness for the Mamba-2 RMSNormGated **prefill** kernel (N6,
//! `mamba2_gated_norm_seq_f32`). Each position is independent (no recurrence), so
//! the batched kernel must equal the single-token gated group-RMSNorm applied
//! per position. Compares against a CPU reference.
//!
//!   hipfire lock acquire test_gated_norm_seq_gpu --watch-pid $$
//!   cargo run -p hipfire-arch-nemotron --example test_gated_norm_seq_gpu

use hipfire_rdna::{DType, Gpu};

fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

fn main() {
    let mut gpu = Gpu::init().expect("GPU init failed");
    eprintln!("GPU: {}", gpu.arch);

    let (n_groups, group_size, seq) = (4usize, 20usize, 37usize);
    let d_inner = n_groups * group_size;
    let eps = 1e-5f32;

    let mut seed = 0x2468_ACE0u32;
    let mut rng = || {
        seed ^= seed << 13;
        seed ^= seed >> 17;
        seed ^= seed << 5;
        (seed as f32 / u32::MAX as f32) * 0.8 - 0.4
    };
    let y: Vec<f32> = (0..seq * d_inner).map(|_| rng()).collect();
    let z: Vec<f32> = (0..seq * d_inner).map(|_| rng()).collect();
    let weight: Vec<f32> = (0..d_inner).map(|_| rng()).collect();

    // CPU reference: gate-then-group-RMSNorm, per position, per group.
    let mut out_cpu = vec![0.0f32; seq * d_inner];
    for p in 0..seq {
        for g in 0..n_groups {
            let base = p * d_inner + g * group_size;
            let wbase = g * group_size;
            let mut ss = 0.0f32;
            let mut gated = vec![0.0f32; group_size];
            for i in 0..group_size {
                gated[i] = y[base + i] * silu(z[base + i]);
                ss += gated[i] * gated[i];
            }
            let inv_rms = 1.0 / (ss / group_size as f32 + eps).sqrt();
            for i in 0..group_size {
                out_cpu[base + i] = gated[i] * inv_rms * weight[wbase + i];
            }
        }
    }

    // GPU.
    let out_g = gpu.zeros(&[seq * d_inner], DType::F32).unwrap();
    let y_g = gpu.upload_f32(&y, &[seq * d_inner]).unwrap();
    let z_g = gpu.upload_f32(&z, &[seq * d_inner]).unwrap();
    let w_g = gpu.upload_f32(&weight, &[d_inner]).unwrap();
    gpu.mamba2_gated_norm_seq_f32(&out_g, &y_g, &z_g, &w_g, seq, n_groups, group_size, eps)
        .unwrap();
    gpu.hip.device_synchronize().unwrap();
    let out_gpu = gpu.download_f32(&out_g).unwrap();

    let max_d = out_cpu
        .iter()
        .zip(&out_gpu)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    eprintln!("seq={seq} n_groups={n_groups} group_size={group_size}  max|Δ|={max_d:.3e}");

    if max_d < 1e-5 {
        println!("PASS: mamba2_gated_norm_seq_f32 matches per-position reference (Δ={max_d:.2e})");
    } else {
        println!("FAIL: mamba2_gated_norm_seq_f32 diverges (Δ={max_d:.2e})");
        std::process::exit(1);
    }
}
