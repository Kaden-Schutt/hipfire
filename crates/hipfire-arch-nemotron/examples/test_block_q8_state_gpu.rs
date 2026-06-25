// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire - see LICENSE and NOTICE in the project root.

//! gpu-vs-gpu correctness for opt-in Q8 Mamba-2 SSM state.
//!
//! Checks two invariants:
//! - q8-state batched prefill matches the q8-state decode loop.
//! - q8-state decode remains close to the fp32-state decode reference.
//!
//!   hipfire lock acquire test_block_q8_state_gpu --watch-pid $$
//!   PATH=/usr/lib/llvm-21/bin:$PATH \
//!     cargo run -p hipfire-arch-nemotron --example test_block_q8_state_gpu

use hipfire_arch_nemotron::block::{Mamba2BlockWeights, Mamba2Dims};
use hipfire_arch_nemotron::block_gpu::{Mamba2BlockGpu, Mamba2StateQuant};
use rdna_compute::Gpu;

fn main() {
    let mut gpu = Gpu::init().expect("GPU init failed");
    eprintln!("GPU: {}", gpu.arch);

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
    let seq = 29usize;

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

    let hidden = dims.hidden_size;
    let hidden_seq: Vec<f32> = (0..seq * hidden).map(|_| rng()).collect();

    let mut fp32_dec =
        Mamba2BlockGpu::new_with_state_quant(&mut gpu, dims.clone(), &w, Mamba2StateQuant::Fp32)
            .expect("upload fp32");
    let mut q8_dec =
        Mamba2BlockGpu::new_with_state_quant(&mut gpu, dims.clone(), &w, Mamba2StateQuant::Q8)
            .expect("upload q8 decode");
    let mut q8_pf =
        Mamba2BlockGpu::new_with_state_quant(&mut gpu, dims.clone(), &w, Mamba2StateQuant::Q8)
            .expect("upload q8 prefill");

    let mut out_fp32_dec = vec![0.0f32; seq * hidden];
    let mut out_q8_dec = vec![0.0f32; seq * hidden];
    for t in 0..seq {
        let row = gpu
            .upload_f32(&hidden_seq[t * hidden..(t + 1) * hidden], &[hidden])
            .unwrap();
        let fp32 = fp32_dec.decode_step(&mut gpu, &row).unwrap();
        gpu.hip.device_synchronize().unwrap();
        let fp32_host = gpu.download_f32(fp32).unwrap();
        out_fp32_dec[t * hidden..(t + 1) * hidden].copy_from_slice(&fp32_host);

        let q8 = q8_dec.decode_step(&mut gpu, &row).unwrap();
        gpu.hip.device_synchronize().unwrap();
        let q8_host = gpu.download_f32(q8).unwrap();
        out_q8_dec[t * hidden..(t + 1) * hidden].copy_from_slice(&q8_host);
        let _ = gpu.free_tensor(row);
    }

    let hs_g = gpu.upload_f32(&hidden_seq, &[seq * hidden]).unwrap();
    let out_q8_pf_t = q8_pf.prefill(&mut gpu, &hs_g, seq).expect("q8 prefill");
    gpu.hip.device_synchronize().unwrap();
    let out_q8_pf = gpu.download_f32(&out_q8_pf_t).unwrap();

    let prefill_vs_decode = out_q8_pf
        .iter()
        .zip(&out_q8_dec)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    let q8_vs_fp32 = out_q8_dec
        .iter()
        .zip(&out_fp32_dec)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);

    eprintln!("q8 prefill vs q8 decode max|delta|={prefill_vs_decode:.3e}");
    eprintln!("q8 decode vs fp32 decode max|delta|={q8_vs_fp32:.3e}");

    if prefill_vs_decode > 1e-4 {
        eprintln!("FAIL: q8 prefill diverges from q8 decode");
        std::process::exit(1);
    }
    if q8_vs_fp32 > 1e-3 {
        eprintln!("FAIL: q8 state drift exceeds fp32 reference tolerance");
        std::process::exit(1);
    }
    println!("PASS: q8 SSM state preserves prefill handoff and stays close to fp32 decode");
}
