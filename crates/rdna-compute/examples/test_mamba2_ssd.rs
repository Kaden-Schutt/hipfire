// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! gpu-vs-cpu correctness for the Mamba-2 SSD decode kernel
//! (`mamba2_ssd_decode_f32`). The CPU reference here mirrors
//! `hipfire_arch_nemotron::ssd::ssd_decode_step` (kept in sync); runs two decode
//! steps (so state decay is exercised) and compares `y` + `state` each step.
//!
//! Run (GPU; coordinate with `hipfire lock`): `cargo run -p rdna-compute
//! --example test_mamba2_ssd`. Exits 0 on pass, 1 on mismatch.

use rdna_compute::{DType, Gpu};

/// CPU reference for one SSD decode step — mirrors
/// `hipfire_arch_nemotron::ssd::ssd_decode_step` (kept in sync).
#[allow(clippy::too_many_arguments)]
fn cpu_ssd_step_full(
    num_heads: usize,
    head_dim: usize,
    state_size: usize,
    n_groups: usize,
    dt_min: f32,
    dt_max: f32,
    a_log: &[f32],
    d: &[f32],
    dt_bias: &[f32],
    dt_raw: &[f32],
    state: &mut [f32],
    x: &[f32],
    b: &[f32],
    c: &[f32],
    y: &mut [f32],
) {
    let softplus = |v: f32| if v > 20.0 { v } else { v.exp().ln_1p() };
    let heads_per_group = num_heads / n_groups;
    for head in 0..num_heads {
        // Interleave (mamba-ssm fast path / HF decode): head h → group h/(H/G).
        let grp = head / heads_per_group;
        let a = -(a_log[head].exp());
        let dt = softplus(dt_raw[head] + dt_bias[head]).clamp(dt_min, dt_max);
        let da = (dt * a).exp();
        let bg = &b[grp * state_size..grp * state_size + state_size];
        let cg = &c[grp * state_size..grp * state_size + state_size];
        for p in 0..head_dim {
            let idx = head * head_dim + p;
            let xp = x[idx];
            let dbx = dt * xp;
            let srow = &mut state[idx * state_size..idx * state_size + state_size];
            let mut acc = 0.0f32;
            for n in 0..state_size {
                srow[n] = da * srow[n] + dbx * bg[n];
                acc += cg[n] * srow[n];
            }
            y[idx] = acc + d[head] * xp;
        }
    }
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

fn main() {
    let mut gpu = Gpu::init().expect("GPU init failed");
    eprintln!("GPU: {}", gpu.arch);

    // Small but representative shapes (n_groups < num_heads to exercise sharing).
    let (num_heads, head_dim, state_size, n_groups) = (4usize, 8usize, 16usize, 2usize);
    let (dt_min, dt_max) = (0.0f32, f32::INFINITY); // forward clamp is (0, inf)

    // Deterministic pseudo-random params.
    let mut seed = 0x2545F491u32;
    let mut rnd = || {
        seed ^= seed << 13;
        seed ^= seed >> 17;
        seed ^= seed << 5;
        (seed as f32 / u32::MAX as f32) * 2.0 - 1.0 // [-1,1]
    };

    let a_log: Vec<f32> = (0..num_heads).map(|_| rnd() * 0.5).collect();
    let d: Vec<f32> = (0..num_heads).map(|_| rnd()).collect();
    let dt_bias: Vec<f32> = (0..num_heads).map(|_| rnd() * 0.1).collect();

    let mut state_cpu = vec![0.0f32; num_heads * head_dim * state_size];
    let d_state = gpu
        .upload_f32(&state_cpu, &[num_heads * head_dim * state_size])
        .unwrap();
    let d_alog = gpu.upload_f32(&a_log, &[num_heads]).unwrap();
    let d_d = gpu.upload_f32(&d, &[num_heads]).unwrap();
    let d_dtbias = gpu.upload_f32(&dt_bias, &[num_heads]).unwrap();

    let mut ok = true;
    for step in 0..2 {
        let x: Vec<f32> = (0..num_heads * head_dim).map(|_| rnd()).collect();
        let b: Vec<f32> = (0..n_groups * state_size).map(|_| rnd()).collect();
        let c: Vec<f32> = (0..n_groups * state_size).map(|_| rnd()).collect();
        let dt_raw: Vec<f32> = (0..num_heads).map(|_| rnd()).collect();

        // CPU reference (mutates state_cpu).
        let mut y_cpu = vec![0.0f32; num_heads * head_dim];
        cpu_ssd_step_full(
            num_heads,
            head_dim,
            state_size,
            n_groups,
            dt_min,
            dt_max,
            &a_log,
            &d,
            &dt_bias,
            &dt_raw,
            &mut state_cpu,
            &x,
            &b,
            &c,
            &mut y_cpu,
        );

        // GPU.
        let d_x = gpu.upload_f32(&x, &[num_heads * head_dim]).unwrap();
        let d_b = gpu.upload_f32(&b, &[n_groups * state_size]).unwrap();
        let d_c = gpu.upload_f32(&c, &[n_groups * state_size]).unwrap();
        let d_dt = gpu.upload_f32(&dt_raw, &[num_heads]).unwrap();
        let d_y = gpu.zeros(&[num_heads * head_dim], DType::F32).unwrap();
        gpu.mamba2_ssd_decode_f32(
            &d_y, &d_state, &d_x, &d_b, &d_c, &d_dt, &d_alog, &d_d, &d_dtbias, num_heads, head_dim,
            state_size, n_groups, dt_min, dt_max,
        )
        .unwrap();
        gpu.hip.device_synchronize().unwrap();

        let y_gpu = gpu.download_f32(&d_y).unwrap();
        let state_gpu = gpu.download_f32(&d_state).unwrap();
        let dy = max_abs_diff(&y_cpu, &y_gpu);
        let ds = max_abs_diff(&state_cpu, &state_gpu);
        eprintln!("step {step}: max|Δy|={dy:.3e}  max|Δstate|={ds:.3e}");
        if dy > 1e-4 || ds > 1e-4 {
            ok = false;
        }
    }

    if ok {
        println!("PASS: mamba2_ssd_decode_f32 matches CPU reference");
    } else {
        eprintln!("FAIL: gpu-vs-cpu mismatch");
        std::process::exit(1);
    }
}
