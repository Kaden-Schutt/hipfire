// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! gpu-vs-cpu correctness for the Mamba-2 xBC short-conv **prefill** scan kernel
//! (N6, `conv1d_bias_silu_seq_f32`). Runs the single-launch GPU scan over a
//! random `seq`-token input and compares both per-position outputs AND the final
//! rolling conv `state` against a CPU reference that applies the decode conv math
//! (4-tap causal depthwise + bias + SiLU) position-by-position — i.e. the kernel
//! must equal the decode kernel repeated.
//!
//!   hipfire lock acquire test_conv1d_seq_gpu --watch-pid $$
//!   cargo run -p hipfire-arch-nemotron --example test_conv1d_seq_gpu

use hipfire_rdna::{DType, Gpu};

fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

fn main() {
    let mut gpu = Gpu::init().expect("GPU init failed");
    eprintln!("GPU: {}", gpu.arch);

    let (channels, seq) = (40usize, 37usize);

    let mut seed = 0x1357_BDF9u32;
    let mut rng = || {
        seed ^= seed << 13;
        seed ^= seed >> 17;
        seed ^= seed << 5;
        (seed as f32 / u32::MAX as f32) * 0.4 - 0.2
    };
    let input: Vec<f32> = (0..seq * channels).map(|_| rng()).collect();
    let weight: Vec<f32> = (0..4 * channels).map(|_| rng()).collect();
    let bias: Vec<f32> = (0..channels).map(|_| rng()).collect();
    // non-zero incoming state to exercise the history hand-off.
    let state0: Vec<f32> = (0..channels * 3).map(|_| rng()).collect();

    // CPU reference: decode conv math repeated. p1=in[t-1], p2=in[t-2], p3=in[t-3].
    let mut y_cpu = vec![0.0f32; seq * channels];
    let mut state_cpu = state0.clone();
    for c in 0..channels {
        let (w3, w2, w1, w0) = (
            weight[c * 4 + 3],
            weight[c * 4 + 2],
            weight[c * 4 + 1],
            weight[c * 4],
        );
        let (mut p1, mut p2, mut p3) =
            (state_cpu[c * 3], state_cpu[c * 3 + 1], state_cpu[c * 3 + 2]);
        for t in 0..seq {
            let x = input[t * channels + c];
            let acc = w3 * x + w2 * p1 + w1 * p2 + w0 * p3 + bias[c];
            y_cpu[t * channels + c] = silu(acc);
            p3 = p2;
            p2 = p1;
            p1 = x;
        }
        state_cpu[c * 3] = p1;
        state_cpu[c * 3 + 1] = p2;
        state_cpu[c * 3 + 2] = p3;
    }

    // GPU.
    let out_g = gpu.zeros(&[seq * channels], DType::F32).unwrap();
    let in_g = gpu.upload_f32(&input, &[seq * channels]).unwrap();
    let w_g = gpu.upload_f32(&weight, &[4 * channels]).unwrap();
    let b_g = gpu.upload_f32(&bias, &[channels]).unwrap();
    let st_g = gpu.upload_f32(&state0, &[channels * 3]).unwrap();
    gpu.conv1d_bias_silu_seq_f32(&out_g, &in_g, &w_g, &b_g, &st_g, seq, channels)
        .unwrap();
    gpu.hip.device_synchronize().unwrap();
    let y_gpu = gpu.download_f32(&out_g).unwrap();
    let state_gpu = gpu.download_f32(&st_g).unwrap();

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
    eprintln!("seq={seq} channels={channels}  max|Δy|={max_y:.3e}  max|Δstate|={max_s:.3e}");

    if max_y < 1e-5 && max_s < 1e-6 {
        println!("PASS: conv1d_bias_silu_seq_f32 matches decode-repeated (Δy={max_y:.2e})");
    } else {
        println!("FAIL: conv1d_bias_silu_seq_f32 diverges (Δy={max_y:.2e}, Δstate={max_s:.2e})");
        std::process::exit(1);
    }
}
