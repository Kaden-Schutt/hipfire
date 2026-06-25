// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Tiny GPU smoke for the Nemotron-H routed MoE block.
//!
//! This validates the FU6 decode path in isolation: router GEMV + sigmoid
//! top-k, shared ReLU2 expert, selected routed ReLU2 experts, and weighted
//! accumulation.

use hipfire_arch_nemotron::moe::{moe_relu2, MoeExpertWeights, MoeRelu2Gpu, MoeWeights};
use hipfire_arch_nemotron::MoeConfig;
use rdna_compute::Gpu;

fn main() {
    let mut gpu = Gpu::init().expect("GPU init");
    let hidden = 4usize;
    let cfg = tiny_cfg();
    let weights = MoeWeights {
        router: vec![
            1.2, -0.2, 0.1, 0.3, // expert 0
            -0.1, 1.0, 0.2, -0.4, // expert 1
            -0.5, -0.2, 0.1, 0.0, // expert 2
            0.1, -0.4, -0.3, 0.2, // expert 3
        ],
        expert_bias: vec![0.0, 0.05, 0.0, -0.1],
        shared_up: vec![
            0.4, -0.1, 0.2, 0.0, 0.1, 0.3, -0.2, 0.2, -0.2, 0.0, 0.5, 0.1,
        ],
        shared_down: vec![
            0.3, -0.1, 0.2, -0.2, 0.4, 0.1, 0.0, 0.2, -0.3, 0.1, -0.1, 0.5,
        ],
        experts: (0..cfg.n_routed_experts)
            .map(|e| {
                let s = 0.1 + e as f32 * 0.03;
                MoeExpertWeights {
                    up: vec![
                        s,
                        -0.2,
                        0.1,
                        0.3,
                        -0.1,
                        0.25 + s,
                        0.2,
                        -0.05,
                        0.05,
                        0.1,
                        0.35 + s,
                        -0.2,
                    ],
                    down: vec![
                        0.2 + s,
                        -0.1,
                        0.05,
                        -0.1,
                        0.3 + s,
                        0.2,
                        0.1,
                        -0.05,
                        0.25 + s,
                        0.0,
                        0.15,
                        -0.2,
                    ],
                }
            })
            .collect(),
    };
    let x = vec![0.5, 1.25, -0.75, 0.25];
    let cpu = moe_relu2(&cfg, &weights, &x, hidden);
    let x_gpu = gpu.upload_f32(&x, &[hidden]).expect("upload x");
    let mut block = MoeRelu2Gpu::new(&mut gpu, hidden, cfg, &weights).expect("build moe");
    let out = block.forward(&mut gpu, &x_gpu).expect("forward");
    gpu.hip.device_synchronize().expect("sync");
    let got = gpu.download_f32(out).expect("download");
    let max_delta = got
        .iter()
        .zip(&cpu)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    eprintln!("max|delta|={max_delta:.3e}");
    assert!(max_delta < 2e-5, "MoE GPU mismatch: max delta {max_delta}");

    let seq_x = vec![
        0.5, 1.25, -0.75, 0.25, //
        -0.25, 0.75, 0.5, -1.0, //
        1.0, -0.5, 0.125, 0.625,
    ];
    let seq = seq_x.len() / hidden;
    let seq_cpu = (0..seq)
        .flat_map(|row| {
            moe_relu2(
                &cfg,
                &weights,
                &seq_x[row * hidden..(row + 1) * hidden],
                hidden,
            )
        })
        .collect::<Vec<_>>();
    let seq_gpu = gpu
        .upload_f32(&seq_x, &[seq, hidden])
        .expect("upload seq x");
    let seq_out = block.prefill(&mut gpu, &seq_gpu, seq).expect("prefill");
    gpu.hip.device_synchronize().expect("sync prefill");
    let seq_got = gpu.download_f32(&seq_out).expect("download seq out");
    let seq_max_delta = seq_got
        .iter()
        .zip(&seq_cpu)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    eprintln!("prefill max|delta|={seq_max_delta:.3e}");
    assert!(
        seq_max_delta < 2e-5,
        "MoE GPU prefill mismatch: max delta {seq_max_delta}"
    );

    let _ = gpu.free_tensor(x_gpu);
    let _ = gpu.free_tensor(seq_gpu);
    let _ = gpu.free_tensor(seq_out);
    block.free(&mut gpu);
    println!("PASS: Nemotron MoE GPU block matches CPU oracle");
}

fn tiny_cfg() -> MoeConfig {
    MoeConfig {
        n_routed_experts: 4,
        num_experts_per_tok: 2,
        intermediate_size: 3,
        n_shared_experts: 1,
        shared_expert_intermediate_size: 3,
        n_group: 1,
        topk_group: 1,
        norm_topk_prob: true,
        routed_scaling_factor: 1.25,
    }
}
