#![allow(
    clippy::duplicated_attributes,
    clippy::doc_lazy_continuation,
    clippy::doc_overindented_list_items,
    clippy::explicit_counter_loop,
    clippy::field_reassign_with_default,
    clippy::manual_checked_ops,
    clippy::manual_clamp,
    clippy::manual_div_ceil,
    clippy::needless_range_loop,
    clippy::ptr_arg,
    clippy::same_item_push,
    clippy::too_many_arguments,
    clippy::type_complexity,
    clippy::unnecessary_cast,
    clippy::useless_vec,
    clippy::while_let_loop
)]
// hipfire example clippy sweep: examples are GPU probes/benches, not reusable APIs.

// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
#![allow(
    clippy::doc_lazy_continuation,
    clippy::doc_overindented_list_items,
    clippy::drop_non_drop,
    clippy::excessive_precision,
    clippy::identity_op,
    clippy::manual_div_ceil,
    clippy::manual_is_multiple_of,
    clippy::needless_range_loop,
    clippy::print_literal,
    clippy::redundant_closure,
    clippy::too_many_arguments,
    clippy::type_complexity,
    clippy::unusual_byte_groupings,
    clippy::useless_vec,
    clippy::unnecessary_cast
)]

//! gpu-vs-cpu correctness for `conv1d_bias_silu_decode_f32` — nemotron_h Mamba-2
//! xBC short-conv decode (depthwise causal K=4 + bias + SiLU). CPU reference
//! mirrors `kernels/src/conv1d_bias_silu_decode.hip`, including the in-place
//! rolling-state advance, validated over several decode steps. Uses Nano-4B
//! conv_dim = 9728 channels.

use hipfire_rdna::{DType, Gpu};

fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

fn main() {
    let mut gpu = Gpu::init().expect("GPU init failed");
    eprintln!("GPU: {}", gpu.arch);

    let nc = 9728usize; // conv_dim for Nano-4B
    let mut seed = 0xCAFEBABEu32;
    let mut rng = || {
        seed ^= seed << 13;
        seed ^= seed >> 17;
        seed ^= seed << 5;
        (seed as f32 / u32::MAX as f32) * 2.0 - 1.0 // [-1,1]
    };
    let weight: Vec<f32> = (0..nc * 4).map(|_| rng() * 0.5).collect();
    let bias: Vec<f32> = (0..nc).map(|_| rng() * 0.2).collect();

    // CPU + GPU both start from zero state; run a few decode steps so the
    // rolling-state advance is exercised, comparing y each step.
    let mut cpu_state = vec![0.0f32; nc * 3]; // [c*3 + t], t=0 newest-1 .. t=2 oldest
    let d_weight = gpu.upload_f32(&weight, &[nc, 4]).unwrap();
    let d_bias = gpu.upload_f32(&bias, &[nc]).unwrap();
    let d_state = gpu.zeros(&[nc, 3], DType::F32).unwrap();
    let d_out = gpu.zeros(&[nc], DType::F32).unwrap();

    let mut worst = 0.0f32;
    for step in 0..5 {
        let input: Vec<f32> = (0..nc).map(|_| rng()).collect();

        // CPU reference (mirror kernel exactly).
        let mut cpu_out = vec![0.0f32; nc];
        for c in 0..nc {
            let x = input[c];
            let s0 = cpu_state[c * 3];
            let s1 = cpu_state[c * 3 + 1];
            let s2 = cpu_state[c * 3 + 2];
            let acc = weight[c * 4 + 3] * x
                + weight[c * 4 + 2] * s0
                + weight[c * 4 + 1] * s1
                + weight[c * 4] * s2
                + bias[c];
            cpu_out[c] = silu(acc);
            cpu_state[c * 3 + 2] = s1;
            cpu_state[c * 3 + 1] = s0;
            cpu_state[c * 3] = x;
        }

        let d_in = gpu.upload_f32(&input, &[nc]).unwrap();
        gpu.conv1d_bias_silu_decode_f32(&d_out, &d_in, &d_weight, &d_bias, &d_state, nc)
            .unwrap();
        gpu.hip.device_synchronize().unwrap();
        let out = gpu.download_f32(&d_out).unwrap();

        let max_diff = out
            .iter()
            .zip(&cpu_out)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        eprintln!("step {step}: max|Δy|={max_diff:.3e}");
        worst = worst.max(max_diff);
    }

    if worst > 1e-5 {
        eprintln!("FAIL (worst max|Δy|={worst:.3e})");
        std::process::exit(1);
    }
    println!("PASS: conv1d_bias_silu_decode_f32 matches CPU reference over 5 steps");
}
