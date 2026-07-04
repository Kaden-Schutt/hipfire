// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! gpu equivalence for the batched ReLU² MLP prefill (N6, `MlpRelu2Gpu::prefill`)
//! vs the per-token `forward` loop. The MLP is position-independent, so this just
//! confirms `gemm_seq` + the elementwise `relu2_f32` batch correctly.
//!
//!   hipfire lock acquire test_mlp_prefill_gpu --watch-pid $$
//!   cargo run -p hipfire-arch-nemotron --example test_mlp_prefill_gpu

use hipfire_arch_nemotron::mlp::MlpRelu2Gpu;
use hipfire_rdna::Gpu;

fn main() {
    let mut gpu = Gpu::init().expect("GPU init failed");
    eprintln!("GPU: {}", gpu.arch);

    let (hidden, intermediate, seq) = (24usize, 40usize, 31usize);
    let mut seed = 0x7E57_1234u32;
    let mut rng = || {
        seed ^= seed << 13;
        seed ^= seed >> 17;
        seed ^= seed << 5;
        (seed as f32 / u32::MAX as f32) * 0.4 - 0.2
    };
    let up: Vec<f32> = (0..intermediate * hidden).map(|_| rng()).collect();
    let down: Vec<f32> = (0..hidden * intermediate).map(|_| rng()).collect();
    let x: Vec<f32> = (0..seq * hidden).map(|_| rng()).collect();

    // batched prefill
    let mut mlp = MlpRelu2Gpu::new(&mut gpu, hidden, intermediate, &up, &down).unwrap();
    let xg = gpu.upload_f32(&x, &[seq * hidden]).unwrap();
    let out_pf_t = mlp.prefill(&mut gpu, &xg, seq).unwrap();
    gpu.hip.device_synchronize().unwrap();
    let out_pf = gpu.download_f32(&out_pf_t).unwrap();

    // per-token forward
    let mut out_loop = vec![0.0f32; seq * hidden];
    for t in 0..seq {
        let row = gpu
            .upload_f32(&x[t * hidden..(t + 1) * hidden], &[hidden])
            .unwrap();
        let o = mlp.forward(&mut gpu, &row).unwrap();
        gpu.hip.device_synchronize().unwrap();
        let ov = gpu.download_f32(o).unwrap();
        out_loop[t * hidden..(t + 1) * hidden].copy_from_slice(&ov);
        let _ = gpu.free_tensor(row);
    }

    let max_d = out_pf
        .iter()
        .zip(&out_loop)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    eprintln!("seq={seq} hidden={hidden} intermediate={intermediate}  max|Δ|={max_d:.3e}");

    if max_d < 1e-4 {
        println!("PASS: MlpRelu2Gpu::prefill matches the forward loop (max|Δ|={max_d:.2e})");
    } else {
        println!("FAIL: MLP prefill diverges (max|Δ|={max_d:.2e})");
        std::process::exit(1);
    }
}
