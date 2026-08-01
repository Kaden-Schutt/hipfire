// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Element-wise channel test for the HFQ4-G128 batched embedding + GEMM path.

use rdna_compute::{DType, Gpu};

const GROUP: usize = 128;
const GROUP_BYTES: usize = 72;

fn append_group(dst: &mut Vec<u8>, scale: f32, zero: f32, codes: &[u8; GROUP]) {
    dst.extend_from_slice(&scale.to_le_bytes());
    dst.extend_from_slice(&zero.to_le_bytes());
    for pair in codes.chunks_exact(2) {
        dst.push(pair[0] | (pair[1] << 4));
    }
}

fn main() {
    let mut gpu = Gpu::init().expect("GPU init");

    // Embedding: vary token, group and element independently so row/group
    // swaps cannot hide behind invariant synthetic data.
    let vocab = 4usize;
    let dim = 256usize;
    let token_ids = [3i32, 1, 0, 2];
    let mut table = Vec::with_capacity(vocab * (dim / GROUP) * GROUP_BYTES);
    let mut reference = vec![0.0f32; token_ids.len() * dim];
    for token in 0..vocab {
        for group in 0..(dim / GROUP) {
            let scale = 0.03125 * (1 + token + group) as f32;
            let zero = -0.125 * (1 + group) as f32;
            let mut codes = [0u8; GROUP];
            for (within, code) in codes.iter_mut().enumerate() {
                *code = ((token * 7 + group * 5 + within * 3) & 15) as u8;
            }
            append_group(&mut table, scale, zero, &codes);
        }
    }
    for (batch, &token) in token_ids.iter().enumerate() {
        for elem in 0..dim {
            let group = elem / GROUP;
            let within = elem % GROUP;
            let scale = 0.03125 * (1 + token as usize + group) as f32;
            let zero = -0.125 * (1 + group) as f32;
            let code = ((token as usize * 7 + group * 5 + within * 3) & 15) as f32;
            reference[batch * dim + elem] = scale * code + zero;
        }
    }
    let token_bytes: Vec<u8> = token_ids
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect();
    let table_gpu = gpu
        .upload_raw(&table, &[table.len()])
        .expect("table upload");
    let tokens_gpu = gpu
        .upload_raw(&token_bytes, &[token_bytes.len()])
        .expect("token upload");
    let output_gpu = gpu
        .zeros(&[token_ids.len() * dim], DType::F32)
        .expect("output alloc");
    gpu.embedding_lookup_hfq4g128_batched(
        &table_gpu,
        &output_gpu,
        &tokens_gpu,
        token_ids.len(),
        dim,
    )
    .expect("batched embedding");
    let output = gpu.download_f32(&output_gpu).expect("embedding download");
    let embed_max = output
        .iter()
        .zip(&reference)
        .map(|(got, want)| (got - want).abs())
        .fold(0.0f32, f32::max);
    assert!(embed_max <= 1.0e-6, "embedding max abs error {embed_max}");

    // GEMM: row- and batch-varying values expose output-lane or batch-tile
    // permutations. This is the exact kernel admitted by llama batched prefill.
    let m = 32usize;
    let k = 256usize;
    let batch_size = 8usize;
    let mut weights = Vec::with_capacity(m * (k / GROUP) * GROUP_BYTES);
    let mut dequant = vec![0.0f32; m * k];
    for row in 0..m {
        for group in 0..(k / GROUP) {
            let scale = 0.0025 * (1 + row % 7 + group) as f32;
            let zero = -0.01 * (1 + row % 3) as f32;
            let mut codes = [0u8; GROUP];
            for (within, code) in codes.iter_mut().enumerate() {
                *code = ((row * 11 + group * 5 + within * 7) & 15) as u8;
                dequant[row * k + group * GROUP + within] = scale * *code as f32 + zero;
            }
            append_group(&mut weights, scale, zero, &codes);
        }
    }
    let x: Vec<f32> = (0..batch_size * k)
        .map(|index| {
            let batch = index / k;
            let col = index % k;
            ((batch * 13 + col * 3) % 29) as f32 * 0.001 - 0.014
        })
        .collect();
    let mut gemm_reference = vec![0.0f32; batch_size * m];
    for batch in 0..batch_size {
        for row in 0..m {
            gemm_reference[batch * m + row] = (0..k)
                .map(|col| dequant[row * k + col] * x[batch * k + col])
                .sum();
        }
    }
    let weights_gpu = gpu
        .upload_raw(&weights, &[weights.len()])
        .expect("weight upload");
    let x_gpu = gpu.upload_f32(&x, &[batch_size, k]).expect("x upload");
    let y_gpu = gpu.zeros(&[batch_size, m], DType::F32).expect("y alloc");
    gpu.gemm_hfq4g128(&weights_gpu, &x_gpu, &y_gpu, m, k, batch_size)
        .expect("G128 GEMM");
    let y = gpu.download_f32(&y_gpu).expect("GEMM download");
    let gemm_max = y
        .iter()
        .zip(&gemm_reference)
        .map(|(got, want)| (got - want).abs())
        .fold(0.0f32, f32::max);
    assert!(gemm_max <= 2.0e-5, "GEMM max abs error {gemm_max}");

    println!("PASS embedding_max_abs={embed_max:.3e} gemm_max_abs={gemm_max:.3e}");
}
