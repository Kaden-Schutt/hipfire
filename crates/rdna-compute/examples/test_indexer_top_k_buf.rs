// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Exactness check and isolated timing probe for DeepSeek V4's graph-safe
//! indexer top-K kernel.
//!
//! Run on the target GPU:
//!   cargo run --release -p rdna-compute --example test_indexer_top_k_buf -- 20

use rdna_compute::{DType, Gpu, GpuTensor};
use std::time::Instant;

const MAX_N: usize = 2048;
const K: usize = 512;

fn upload_i32(gpu: &mut Gpu, values: &[i32]) -> GpuTensor {
    let tensor = gpu
        .alloc_tensor(&[values.len() * 4], DType::Raw)
        .expect("alloc i32 tensor");
    let bytes = unsafe {
        std::slice::from_raw_parts(values.as_ptr().cast::<u8>(), std::mem::size_of_val(values))
    };
    gpu.hip
        .memcpy_htod(&tensor.buf, bytes)
        .expect("upload i32 tensor");
    tensor
}

fn download_i32(gpu: &Gpu, tensor: &GpuTensor, len: usize) -> Vec<i32> {
    let mut values = vec![0i32; len];
    let bytes =
        unsafe { std::slice::from_raw_parts_mut(values.as_mut_ptr().cast::<u8>(), len * 4) };
    gpu.hip
        .memcpy_dtoh(bytes, &tensor.buf)
        .expect("download i32 tensor");
    values
}

fn expected_top_k(scores: &[f32], n: usize) -> Vec<i32> {
    if n <= K {
        return (0..K)
            .map(|index| if index < n { index as i32 } else { -1 })
            .collect();
    }

    let mut indices: Vec<usize> = (0..n).collect();
    indices.sort_unstable_by(|&a, &b| scores[b].total_cmp(&scores[a]).then_with(|| a.cmp(&b)));
    indices
        .into_iter()
        .take(K)
        .map(|index| index as i32)
        .collect()
}

fn run_case(gpu: &mut Gpu, n: usize, iters: usize) {
    assert!(n <= MAX_N);
    // 109 is coprime with 2048, so the long-context case is a permutation
    // of 0..2047. The n=513 case deliberately repeats scores to exercise
    // the lower-index stable tiebreak used by the serial reference.
    let scores: Vec<f32> = if n == K + 1 {
        (0..MAX_N).map(|index| (index % 64) as f32).collect()
    } else {
        (0..MAX_N)
            .map(|index| ((index * 109 + 17) % MAX_N) as f32)
            .collect()
    };
    let scores_gpu = gpu.upload_f32(&scores, &[MAX_N]).expect("upload scores");
    let indices_gpu = gpu
        .alloc_tensor(&[K * 4], DType::Raw)
        .expect("alloc top-k output");
    let n_gpu = upload_i32(gpu, &[n as i32]);
    let k_gpu = upload_i32(gpu, &[K as i32]);

    gpu.indexer_top_k_buf(
        &scores_gpu,
        &indices_gpu,
        &n_gpu,
        &k_gpu,
        1,
        MAX_N as i32,
        K as i32,
    )
    .expect("warm indexer_top_k_buf");
    gpu.hip.device_synchronize().expect("warm synchronize");

    let actual = download_i32(gpu, &indices_gpu, K);
    let expected = expected_top_k(&scores, n);
    assert_eq!(actual, expected, "top-k mismatch at n={n}");

    let started = Instant::now();
    for _ in 0..iters {
        gpu.indexer_top_k_buf(
            &scores_gpu,
            &indices_gpu,
            &n_gpu,
            &k_gpu,
            1,
            MAX_N as i32,
            K as i32,
        )
        .expect("timed indexer_top_k_buf");
    }
    gpu.hip.device_synchronize().expect("timed synchronize");
    let us = started.elapsed().as_secs_f64() * 1e6 / iters as f64;
    println!("PASS n={n} k={K} exact=true {us:.3} us/call");
}

fn main() {
    let iters = std::env::args()
        .nth(1)
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(20)
        .max(1);
    let mut gpu = Gpu::init().expect("GPU init");
    assert_eq!(gpu.arch, "gfx1151", "parallel route is gfx1151-specific");
    gpu.deepseek4_mq2r_route_v1 = true;
    eprintln!("GPU: {} MQ2R route active, iters={iters}", gpu.arch);

    run_case(&mut gpu, K, iters);
    run_case(&mut gpu, K + 1, iters);
    run_case(&mut gpu, MAX_N, iters);
}
