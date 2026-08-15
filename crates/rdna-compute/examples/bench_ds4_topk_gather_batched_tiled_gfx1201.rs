// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Product-shape screen for the gfx1201 batched top-K KV gather transpose.

use rdna_compute::{DType, Gpu, GpuTensor};

const K: usize = 512;
const D: usize = 512;
const N: usize = 512;

fn bytes_i32(values: &[i32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}

fn bytes_f32(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}

fn wrap(raw: *mut std::ffi::c_void, bytes: usize, shape: Vec<usize>, dtype: DType) -> GpuTensor {
    GpuTensor {
        buf: unsafe { hip_bridge::DeviceBuffer::from_raw(raw, bytes) },
        shape,
        dtype,
    }
}

fn inputs(batch: usize) -> (Vec<f32>, Vec<i32>) {
    let cache: Vec<f32> = (0..N * D)
        .map(|index| ((index.wrapping_mul(37) % 4093) as f32 - 2046.0) / 2048.0)
        .collect();
    let mut indices = Vec::with_capacity(batch * K);
    for b in 0..batch {
        // Match the second B1024 chunk: compressed visibility grows from 256
        // through 512, leaving exact -1 sentinels past each row's active K.
        let n_active = (256 + b / 4).min(N);
        for k in 0..K {
            indices.push(if k < n_active {
                ((k * 73 + b * 17) % n_active) as i32
            } else {
                -1
            });
        }
    }
    (cache, indices)
}

fn allocate_case(gpu: &mut Gpu, batch: usize) -> (GpuTensor, GpuTensor, GpuTensor, GpuTensor) {
    let (cache, indices) = inputs(batch);
    let cache_bytes = bytes_f32(&cache);
    let index_bytes = bytes_i32(&indices);
    let output_bytes = batch * D * K * 4;

    let d_cache = gpu.hip.malloc(cache_bytes.len()).unwrap();
    let d_indices = gpu.hip.malloc(index_bytes.len()).unwrap();
    let d_reference = gpu.hip.malloc(output_bytes).unwrap();
    let d_candidate = gpu.hip.malloc(output_bytes).unwrap();
    gpu.hip.memcpy_htod(&d_cache, &cache_bytes).unwrap();
    gpu.hip.memcpy_htod(&d_indices, &index_bytes).unwrap();

    let tensors = (
        wrap(d_cache.as_ptr(), cache_bytes.len(), vec![N, D], DType::F32),
        wrap(
            d_indices.as_ptr(),
            index_bytes.len(),
            vec![batch, K],
            DType::Raw,
        ),
        wrap(
            d_reference.as_ptr(),
            output_bytes,
            vec![batch, D, K],
            DType::F32,
        ),
        wrap(
            d_candidate.as_ptr(),
            output_bytes,
            vec![batch, D, K],
            DType::F32,
        ),
    );
    // Ownership moves into the GpuTensor wrappers created from these raw
    // allocations; prevent the original handles from freeing them twice.
    std::mem::forget(d_cache);
    std::mem::forget(d_indices);
    std::mem::forget(d_reference);
    std::mem::forget(d_candidate);
    tensors
}

fn reference(gpu: &mut Gpu, cache: &GpuTensor, indices: &GpuTensor, out: &GpuTensor, batch: usize) {
    gpu.deepseek4_topk_kv_gather_batched_f32(
        cache,
        indices,
        out,
        K as i32,
        D as i32,
        N as i32,
        K as i32,
        0,
        1.0,
        batch as i32,
    )
    .unwrap();
}

fn candidate(gpu: &mut Gpu, cache: &GpuTensor, indices: &GpuTensor, out: &GpuTensor, batch: usize) {
    gpu.deepseek4_topk_kv_gather_batched_tiled_gfx1201(
        cache,
        indices,
        out,
        K as i32,
        D as i32,
        N as i32,
        K as i32,
        0,
        1.0,
        batch as i32,
    )
    .unwrap();
}

fn correctness(gpu: &mut Gpu) {
    let batch = 8;
    let (cache, indices, reference_out, candidate_out) = allocate_case(gpu, batch);
    reference(gpu, &cache, &indices, &reference_out, batch);
    candidate(gpu, &cache, &indices, &candidate_out, batch);
    gpu.hip.device_synchronize().unwrap();

    let output_bytes = batch * D * K * 4;
    let mut a = vec![0_u8; output_bytes];
    let mut b = vec![0_u8; output_bytes];
    gpu.hip.memcpy_dtoh(&mut a, &reference_out.buf).unwrap();
    gpu.hip.memcpy_dtoh(&mut b, &candidate_out.buf).unwrap();
    let mismatches = a
        .chunks_exact(4)
        .zip(b.chunks_exact(4))
        .filter(|(left, right)| left != right)
        .count();
    println!("correctness: B={batch} raw_mismatches={mismatches}");
    assert_eq!(mismatches, 0);
}

fn timing(gpu: &mut Gpu) {
    let batch = 1024;
    let (cache, indices, reference_out, candidate_out) = allocate_case(gpu, batch);
    for _ in 0..2 {
        reference(gpu, &cache, &indices, &reference_out, batch);
        candidate(gpu, &cache, &indices, &candidate_out, batch);
    }
    gpu.hip.device_synchronize().unwrap();

    let time = |gpu: &mut Gpu, launch: &dyn Fn(&mut Gpu)| -> f64 {
        let start = gpu.hip.event_create().unwrap();
        let stop = gpu.hip.event_create().unwrap();
        gpu.hip.event_record(&start, None).unwrap();
        for _ in 0..5 {
            launch(gpu);
        }
        gpu.hip.event_record(&stop, None).unwrap();
        gpu.hip.event_synchronize(&stop).unwrap();
        gpu.hip.event_elapsed_ms(&start, &stop).unwrap() as f64 * 200.0
    };
    let reference_us = time(gpu, &|gpu| {
        reference(gpu, &cache, &indices, &reference_out, batch)
    });
    let candidate_us = time(gpu, &|gpu| {
        candidate(gpu, &cache, &indices, &candidate_out, batch)
    });
    println!(
        "timing: B={batch} K={K} D={D} reference={reference_us:.3}us tiled={candidate_us:.3}us speedup={:.3}x",
        reference_us / candidate_us
    );
}

fn main() {
    let mut gpu = Gpu::init().expect("gpu init");
    assert_eq!(gpu.arch, "gfx1201", "requires exact gfx1201");
    println!("arch={}", gpu.arch);
    correctness(&mut gpu);
    timing(&mut gpu);
}
