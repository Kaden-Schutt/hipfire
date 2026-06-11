// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Synthetic gfx1151 IU4/IU8 WMMA probe.
//!
//! This is step 1 for the Q4-activation MMQ path: prove the packed operand
//! width, accumulator layout, and rough issue-rate delta before touching model
//! kernels. It intentionally does not exercise HFQ/MQ reconstruction math.
//!
//! Run:
//!   cargo run --release -p rdna-compute --example probe_gfx1151_iu4_wmma

use rdna_compute::{DType, Gpu, GpuTensor};

fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(default)
}

fn download_i32(gpu: &Gpu, tensor: &GpuTensor, n: usize) -> Vec<i32> {
    let mut data = vec![0i32; n];
    let bytes = unsafe { std::slice::from_raw_parts_mut(data.as_mut_ptr() as *mut u8, n * 4) };
    gpu.hip.memcpy_dtoh(bytes, &tensor.buf).expect("dtoh i32");
    data
}

fn verify(label: &str, data: &[i32], expected: i32) {
    let mut worst = 0i32;
    let mut bad = 0usize;
    let mut first_bad = None;
    for (idx, &v) in data.iter().enumerate() {
        let err = (v - expected).abs();
        if err > worst {
            worst = err;
        }
        if v != expected {
            bad += 1;
            if first_bad.is_none() {
                first_bad = Some((idx, v));
            }
        }
    }
    if let Some((idx, v)) = first_bad {
        panic!("{label}: mismatch at linear accumulator {idx}: got {v}, expected {expected}; bad={bad} worst_abs={worst}");
    }
}

fn time_probe(gpu: &mut Gpu, out: &GpuTensor, blocks: usize, iters: usize, use_iu4: bool) -> f64 {
    let start = gpu.hip.event_create().expect("event start");
    let stop = gpu.hip.event_create().expect("event stop");
    gpu.hip
        .event_record(&start, gpu.active_stream.as_ref())
        .expect("record start");
    gpu.bench_iu_wmma_gfx1151(out, blocks, iters, use_iu4)
        .expect("launch probe");
    gpu.hip
        .event_record(&stop, gpu.active_stream.as_ref())
        .expect("record stop");
    gpu.hip.event_synchronize(&stop).expect("sync stop");
    let ms = gpu.hip.event_elapsed_ms(&start, &stop).expect("elapsed ms") as f64;
    gpu.hip.event_destroy(start).ok();
    gpu.hip.event_destroy(stop).ok();
    ms
}

fn bench_one(
    gpu: &mut Gpu,
    out: &GpuTensor,
    blocks: usize,
    iters: usize,
    trials: usize,
    use_iu4: bool,
) -> (f64, f64) {
    let label = if use_iu4 { "IU4" } else { "IU8" };
    gpu.bench_iu_wmma_gfx1151(out, blocks, iters, use_iu4)
        .expect("warmup launch");
    gpu.hip.device_synchronize().expect("warmup sync");
    let expected = (16 * iters) as i32;
    let data = download_i32(gpu, out, blocks * 32 * 8);
    verify(label, &data, expected);

    let mut samples = Vec::with_capacity(trials);
    for _ in 0..trials {
        samples.push(time_probe(gpu, out, blocks, iters, use_iu4));
    }
    samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median_ms = samples[samples.len() / 2];
    let ops = blocks as f64 * iters as f64 * 8192.0;
    let gops = ops / (median_ms / 1000.0) / 1.0e9;
    (median_ms, gops)
}

fn main() {
    let blocks = env_usize("HIPFIRE_IU4_PROBE_BLOCKS", 4096);
    let iters = env_usize("HIPFIRE_IU4_PROBE_ITERS", 512);
    let trials = env_usize("HIPFIRE_IU4_PROBE_TRIALS", 7).max(1);
    let out_i32 = blocks * 32 * 8;

    let mut gpu = Gpu::init().expect("gpu init");
    if !gpu.arch.starts_with("gfx1151") {
        println!("SKIP: arch {} is not gfx1151", gpu.arch);
        return;
    }

    let out = gpu
        .alloc_tensor(&[out_i32 * 4], DType::Raw)
        .expect("alloc output");
    gpu.hip
        .memset(&out.buf, 0, out_i32 * 4)
        .expect("clear output");

    println!("=== gfx1151 IU4/IU8 WMMA probe ===");
    println!("blocks={blocks} iters={iters} trials={trials}");
    println!("expected accumulator value per cell={}", 16 * iters);

    let (iu4_ms, iu4_gops) = bench_one(&mut gpu, &out, blocks, iters, trials, true);
    let (iu8_ms, iu8_gops) = bench_one(&mut gpu, &out, blocks, iters, trials, false);
    println!("IU4 median: {iu4_ms:.4} ms  {iu4_gops:.1} GOPS");
    println!("IU8 median: {iu8_ms:.4} ms  {iu8_gops:.1} GOPS");
    println!("IU4/IU8 throughput ratio: {:.3}x", iu4_gops / iu8_gops);
}
