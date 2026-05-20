// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Profile Q4_K vs Q4_F16 vs F32 GEMV with different thread configs.
//! Tests the hypothesis that occupancy (threads/CU) explains the 40% vs 70% gap.

fn main() {
    let mut gpu = rdna_compute::Gpu::init().expect("GPU init");
    
    let path = "/home/kaden/llama.cpp/models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf";
    let gguf = hipfire_runtime::gguf::GgufFile::open(std::path::Path::new(path)).unwrap();
    let tensor_info = gguf.find_tensor("blk.0.attn_q.weight").unwrap();
    let raw_q4k = gguf.tensor_data(tensor_info);
    let m = 2048usize;
    let k = 2048usize;
    
    // Prepare data
    let a_f32 = hipfire_runtime::llama::dequantize_q4_k(raw_q4k, m * k);
    let x_data: Vec<f32> = (0..k).map(|i| ((i % 7) as f32 - 3.0) * 0.01).collect();
    let d_x = gpu.upload_f32(&x_data, &[k]).unwrap();
    
    // Upload all formats
    let d_q4k = gpu.upload_raw(raw_q4k, &[raw_q4k.len()]).unwrap();
    let d_f32 = gpu.upload_f32(&a_f32, &[m, k]).unwrap();
    let q4f16 = hipfire_runtime::llama::convert_q4k_to_q4f16_g64(raw_q4k, m*k);
    let d_q4f16 = gpu.upload_raw(&q4f16, &[q4f16.len()]).unwrap();
    
    let n_warmup = 20;
    let n_iter = 500;
    
    // F32 GEMV: 256 threads, shared memory — THE REFERENCE
    {
        let d_y = gpu.zeros(&[m], rdna_compute::DType::F32).unwrap();
        for _ in 0..n_warmup { gpu.gemv_f32(&d_f32, &d_x, &d_y).unwrap(); }
        let start = gpu.hip.event_create().unwrap();
        let stop = gpu.hip.event_create().unwrap();
        gpu.hip.event_record(&start, None).unwrap();
        for _ in 0..n_iter { gpu.gemv_f32(&d_f32, &d_x, &d_y).unwrap(); }
        gpu.hip.event_record(&stop, None).unwrap();
        gpu.hip.event_synchronize(&stop).unwrap();
        let ms = gpu.hip.event_elapsed_ms(&start, &stop).unwrap();
        let bytes = (m * k * 4 + k * 4) as f64;
        let bw = (bytes * n_iter as f64) / (ms as f64 / 1000.0) / 1e9;
        let us = ms * 1000.0 / n_iter as f32;
        eprintln!("F32 GEMV (256 thr, shmem): {us:.1} us, {bw:.1} GB/s ({:.1}% peak)", bw/448.0*100.0);
        gpu.free_tensor(d_y).unwrap();
    }
    
    // Q4_K GEMV: 32 threads, warp shuffle
    {
        let d_y = gpu.zeros(&[m], rdna_compute::DType::F32).unwrap();
        for _ in 0..n_warmup { gpu.gemv_q4k(&d_q4k, &d_x, &d_y, m, k).unwrap(); }
        let start = gpu.hip.event_create().unwrap();
        let stop = gpu.hip.event_create().unwrap();
        gpu.hip.event_record(&start, None).unwrap();
        for _ in 0..n_iter { gpu.gemv_q4k(&d_q4k, &d_x, &d_y, m, k).unwrap(); }
        gpu.hip.event_record(&stop, None).unwrap();
        gpu.hip.event_synchronize(&stop).unwrap();
        let ms = gpu.hip.event_elapsed_ms(&start, &stop).unwrap();
        let bytes = (m * (k/256) * 144 + k * 4) as f64;
        let bw = (bytes * n_iter as f64) / (ms as f64 / 1000.0) / 1e9;
        let us = ms * 1000.0 / n_iter as f32;
        eprintln!("Q4_K GEMV (32 thr, warp):  {us:.1} us, {bw:.1} GB/s ({:.1}% peak)", bw/448.0*100.0);
        gpu.free_tensor(d_y).unwrap();
    }
    
    // Q4_F16_G64 GEMV: 32 threads, warp shuffle (current)
    {
        let d_y = gpu.zeros(&[m], rdna_compute::DType::F32).unwrap();
        for _ in 0..n_warmup { gpu.gemv_q4f16_g64(&d_q4f16, &d_x, &d_y, m, k).unwrap(); }
        let start = gpu.hip.event_create().unwrap();
        let stop = gpu.hip.event_create().unwrap();
        gpu.hip.event_record(&start, None).unwrap();
        for _ in 0..n_iter { gpu.gemv_q4f16_g64(&d_q4f16, &d_x, &d_y, m, k).unwrap(); }
        gpu.hip.event_record(&stop, None).unwrap();
        gpu.hip.event_synchronize(&stop).unwrap();
        let ms = gpu.hip.event_elapsed_ms(&start, &stop).unwrap();
        let bytes = (m * (k/64) * 36 + k * 4) as f64;
        let bw = (bytes * n_iter as f64) / (ms as f64 / 1000.0) / 1e9;
        let us = ms * 1000.0 / n_iter as f32;
        eprintln!("Q4_F16 G64 (32 thr, warp): {us:.1} us, {bw:.1} GB/s ({:.1}% peak)", bw/448.0*100.0);
        gpu.free_tensor(d_y).unwrap();
    }
    
    eprintln!("\nKey config comparison:");
    eprintln!("  F32:  256 thr/block × ~8 blocks/CU = ~2048 thr/CU = ~10 waves/SIMD");
    eprintln!("  Q4_K: 32 thr/block × 20 blocks/CU = 640 thr/CU = 5 waves/SIMD");
    eprintln!("  Occupancy ratio: 2x → explains ~2x bandwidth gap");
}
