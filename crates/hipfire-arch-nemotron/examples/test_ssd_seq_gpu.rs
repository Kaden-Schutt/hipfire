// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! gpu-vs-cpu correctness for the Mamba-2 SSD **prefill** scan kernel (N6,
//! `mamba2_ssd_seq_f32`). Runs one single-launch GPU scan over a `seq`-token
//! random sequence and compares both the per-position outputs `y` AND the final
//! recurrent `state` against the CPU sequential reference `ssd::ssd_sequence`
//! (the kernel is bit-faithful to the decode loop, so the bar is tight). A
//! sign/stride/group error in the HIP kernel surfaces here against a readable
//! oracle instead of as a prefill attractor.
//!
//! Run gpu-tcas-coordinated:
//!   hipfire lock acquire test_ssd_seq_gpu --watch-pid $$
//!   cargo run -p hipfire-arch-nemotron --example test_ssd_seq_gpu

use hipfire_arch_nemotron::ssd::{ssd_sequence, SsdParams};
use rdna_compute::{DType, Gpu};

fn main() {
    let mut gpu = Gpu::init().expect("GPU init failed");
    eprintln!("GPU: {}", gpu.arch);

    // Structurally faithful small dims (non-pow2 head_dim) + nemotron-shaped
    // group interleave (num_heads/n_groups = 3 heads per group).
    let (h, dh, n, g, seq) = (6usize, 5usize, 8usize, 2usize, 37usize);
    let xd = h * dh;
    let bd = g * n;

    let mut seed = 0x51ED_270Bu32;
    let mut rng = || {
        seed ^= seed << 13;
        seed ^= seed >> 17;
        seed ^= seed << 5;
        (seed as f32 / u32::MAX as f32) * 0.4 - 0.2
    };

    let p = SsdParams {
        num_heads: h,
        head_dim: dh,
        state_size: n,
        n_groups: g,
        dt_min: 0.0,
        dt_max: f32::INFINITY,
        a_log: (0..h).map(|_| rng() * 0.5).collect(),
        d: (0..h).map(|_| rng()).collect(),
        dt_bias: (0..h).map(|_| rng()).collect(),
    };
    let x: Vec<f32> = (0..seq * xd).map(|_| rng()).collect();
    let b: Vec<f32> = (0..seq * bd).map(|_| rng()).collect();
    let c: Vec<f32> = (0..seq * bd).map(|_| rng()).collect();
    let dt: Vec<f32> = (0..seq * h).map(|_| rng()).collect();

    // CPU reference (sequential scan from zero state).
    let mut state_cpu = vec![0.0f32; h * dh * n];
    let mut y_cpu = vec![0.0f32; seq * xd];
    ssd_sequence(&p, &mut state_cpu, &x, &b, &c, &dt, &mut y_cpu);

    // GPU: single-launch prefill scan from zero state.
    let y_g = gpu.zeros(&[seq * xd], DType::F32).unwrap();
    let state_g = gpu.zeros(&[h * dh * n], DType::F32).unwrap();
    let x_g = gpu.upload_f32(&x, &[seq * xd]).unwrap();
    let b_g = gpu.upload_f32(&b, &[seq * bd]).unwrap();
    let c_g = gpu.upload_f32(&c, &[seq * bd]).unwrap();
    let dt_g = gpu.upload_f32(&dt, &[seq * h]).unwrap();
    let alog_g = gpu.upload_f32(&p.a_log, &[h]).unwrap();
    let d_g = gpu.upload_f32(&p.d, &[h]).unwrap();
    let dtb_g = gpu.upload_f32(&p.dt_bias, &[h]).unwrap();

    gpu.mamba2_ssd_seq_f32(
        &y_g, &state_g, &x_g, &b_g, &c_g, &dt_g, &alog_g, &d_g, &dtb_g, seq, h, dh, n, g, p.dt_min,
        p.dt_max,
    )
    .unwrap();
    gpu.hip.device_synchronize().unwrap();

    let y_gpu = gpu.download_f32(&y_g).unwrap();
    let state_gpu = gpu.download_f32(&state_g).unwrap();

    let max_y = y_cpu
        .iter()
        .zip(&y_gpu)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    let max_s = state_cpu
        .iter()
        .zip(&state_gpu)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);

    eprintln!("seq={seq} heads={h} head_dim={dh} state={n} groups={g}");
    eprintln!("max|Δy|={max_y:.3e}  max|Δstate|={max_s:.3e}");

    // Same op order as the decode loop → only libdevice expf/log1pf differences.
    if max_y < 1e-4 && max_s < 1e-4 {
        println!(
            "PASS: mamba2_ssd_seq_f32 matches ssd_sequence (Δy={max_y:.2e}, Δstate={max_s:.2e})"
        );
    } else {
        println!("FAIL: mamba2_ssd_seq_f32 diverges (Δy={max_y:.2e}, Δstate={max_s:.2e})");
        std::process::exit(1);
    }
}
