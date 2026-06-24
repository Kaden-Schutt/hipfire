// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! gpu-vs-cpu correctness for the full Mamba-2 mixer block decode step (N3).
//! Runs the GPU forward (`Mamba2BlockGpu::decode_step`) and the CPU oracle
//! (`block::mamba2_block_decode_step`) from identical zero state over several
//! decode steps with identical random weights, comparing the `[hidden_size]`
//! output each step. Uses structurally-faithful small dims so the whole
//! pipeline (in_proj → conv → split → SSD → gated-norm → out_proj) is exercised
//! including the recurrent conv + SSM state advance.
//!
//! Run gpu-tcas-coordinated:
//!   hipfire lock acquire test_block_gpu --watch-pid $$
//!   cargo run -p hipfire-arch-nemotron --example test_block_gpu

use hipfire_arch_nemotron::block::{
    mamba2_block_decode_step, Mamba2BlockState, Mamba2BlockWeights, Mamba2Dims,
};
use hipfire_arch_nemotron::block_gpu::Mamba2BlockGpu;
use rdna_compute::Gpu;

fn main() {
    let mut gpu = Gpu::init().expect("GPU init failed");
    eprintln!("GPU: {}", gpu.arch);

    // Small but structurally faithful, with deliberately NON-power-of-two
    // hidden_size (24) and d_inner (4×10=40) so both gemv_f32 calls take K<256
    // non-pow2 — exercising the gemv_f32 block-reduction pow2-rounding fix
    // (these dims produced a ~7% mismatch before that fix). state 16, 2 groups,
    // K=4, group_size = 40/2 = 20.
    let dims = Mamba2Dims {
        hidden_size: 24,
        num_heads: 4,
        head_dim: 10,
        state_size: 16,
        n_groups: 2,
        conv_kernel: 4,
        rms_norm_eps: 1e-5,
        dt_min: 0.0,
        dt_max: f32::INFINITY,
    };

    let mut seed = 0x51ED_270Bu32;
    let mut rng = || {
        seed ^= seed << 13;
        seed ^= seed >> 17;
        seed ^= seed << 5;
        (seed as f32 / u32::MAX as f32) * 0.4 - 0.2
    };
    let in_proj: Vec<f32> = (0..dims.projection_size() * dims.hidden_size)
        .map(|_| rng())
        .collect();
    let conv_weight: Vec<f32> = (0..dims.conv_dim() * dims.conv_kernel)
        .map(|_| rng())
        .collect();
    let conv_bias: Vec<f32> = (0..dims.conv_dim()).map(|_| rng()).collect();
    let a_log: Vec<f32> = (0..dims.num_heads).map(|_| rng()).collect();
    let dd: Vec<f32> = (0..dims.num_heads).map(|_| rng()).collect();
    let dt_bias: Vec<f32> = (0..dims.num_heads).map(|_| rng()).collect();
    let norm_weight: Vec<f32> = (0..dims.d_inner()).map(|_| 1.0 + rng()).collect();
    let out_proj: Vec<f32> = (0..dims.hidden_size * dims.d_inner())
        .map(|_| rng())
        .collect();

    let w = Mamba2BlockWeights {
        in_proj: &in_proj,
        conv_weight: &conv_weight,
        conv_bias: &conv_bias,
        a_log: &a_log,
        d: &dd,
        dt_bias: &dt_bias,
        norm_weight: &norm_weight,
        out_proj: &out_proj,
    };

    let mut cpu_state = Mamba2BlockState::zeros(&dims);
    let mut gpu_block = Mamba2BlockGpu::new(&mut gpu, dims.clone(), &w).expect("upload");

    let mut worst = 0.0f32;
    for step in 0..6 {
        let hidden: Vec<f32> = (0..dims.hidden_size).map(|_| rng()).collect();

        let cpu_out = mamba2_block_decode_step(&dims, &w, &mut cpu_state, &hidden);

        let d_hidden = gpu.upload_f32(&hidden, &[dims.hidden_size]).unwrap();
        let out_t = gpu_block.decode_step(&mut gpu, &d_hidden).unwrap();
        gpu.hip.device_synchronize().unwrap();
        let gpu_out = gpu.download_f32(out_t).unwrap();

        let max_diff = gpu_out
            .iter()
            .zip(&cpu_out)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        eprintln!("step {step}: max|Δout|={max_diff:.3e}");
        worst = worst.max(max_diff);
    }

    if worst > 1e-4 {
        eprintln!("FAIL (worst max|Δout|={worst:.3e})");
        std::process::exit(1);
    }
    println!("PASS: Mamba2BlockGpu matches CPU oracle over 6 decode steps");
}
