// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
//
// Validates the batched LFM2 gated-conv forward (conv1d_gated_batched_f32 +
// conv1d_gated_state_advance_f32) against Bb sequential single-token decode
// steps (conv1d_gated_decode_f32). Since bx is non-recurrent and the K-window
// sum order is identical in both kernels, the result must be byte-exact.

use rdna_compute::{DType, Gpu};

fn frand(seed: usize) -> f32 {
    // deterministic pseudo-random in [-1, 1)
    ((seed.wrapping_mul(2654435761) % 1000) as f32 / 500.0) - 1.0
}

fn main() {
    let mut gpu = Gpu::init().expect("GPU init failed");

    let bb: usize = 16; // block positions
    let c: usize = 1024; // channels (LFM2.5-350M hidden)
    let k: usize = 3; // conv kernel
    let hist = k - 1;

    let bcx: Vec<f32> = (0..bb * 3 * c).map(|i| frand(i + 1)).collect();
    let state0: Vec<f32> = (0..c * hist).map(|i| frand(i + 10_000)).collect();
    let weight: Vec<f32> = (0..c * k).map(|i| frand(i + 20_000)).collect();

    let w = gpu.upload_f32(&weight, &[c, k]).unwrap();

    // -------- batched --------
    let bcx_g = gpu.upload_f32(&bcx, &[bb, 3 * c]).unwrap();
    let state_b = gpu.upload_f32(&state0, &[c, hist]).unwrap();
    let out_b = gpu.zeros(&[bb, c], DType::F32).unwrap();
    gpu.conv1d_gated_batched_f32(&bcx_g, &state_b, &w, &out_b, bb, c, k)
        .unwrap();
    let state_adv = gpu.zeros(&[c, hist], DType::F32).unwrap();
    gpu.conv1d_gated_state_advance_f32(&bcx_g, &state_b, &state_adv, bb, c, k)
        .unwrap();
    let out_batched = gpu.download_f32(&out_b).unwrap();
    let state_batched = gpu.download_f32(&state_adv).unwrap();

    // -------- sequential decode (Bb steps, state advanced in place) --------
    let state_seq = gpu.upload_f32(&state0, &[c, hist]).unwrap();
    let mut out_seq = vec![0f32; bb * c];
    for j in 0..bb {
        let bcx_j: Vec<f32> = bcx[j * 3 * c..(j + 1) * 3 * c].to_vec();
        let bcx_jg = gpu.upload_f32(&bcx_j, &[1, 3 * c]).unwrap();
        let out_jg = gpu.zeros(&[1, c], DType::F32).unwrap();
        gpu.conv1d_gated_decode_f32(&bcx_jg, &state_seq, &w, &out_jg, 1, c, k)
            .unwrap();
        let out_j = gpu.download_f32(&out_jg).unwrap();
        out_seq[j * c..(j + 1) * c].copy_from_slice(&out_j);
    }
    let state_seq_final = gpu.download_f32(&state_seq).unwrap();

    // -------- compare --------
    let max_out = out_batched
        .iter()
        .zip(&out_seq)
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    let max_state = state_batched
        .iter()
        .zip(&state_seq_final)
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);

    println!(
        "batched vs sequential-decode (Bb={bb}, C={c}, K={k}): max|out|={max_out:.3e}  max|state|={max_state:.3e}"
    );
    if max_out < 1e-5 && max_state < 1e-5 {
        println!("conv1d_gated_batched: PASS");
    } else {
        println!("conv1d_gated_batched: FAIL");
        std::process::exit(1);
    }
}
