// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Head-to-head benchmark: 32-thread warp vs 256-thread wide Q4_F16 GEMV
fn main() {
    let mut gpu = rdna_compute::Gpu::init().expect("GPU init");
    
    let path = "/home/kaden/llama.cpp/models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf";
    let gguf = hipfire_runtime::gguf::GgufFile::open(std::path::Path::new(path)).unwrap();
    let ti = gguf.find_tensor("blk.0.attn_q.weight").unwrap();
    let raw = gguf.tensor_data(ti);
    let m = 2048usize;
    let k = 2048usize;
    
    let x_data: Vec<f32> = (0..k).map(|i| ((i % 7) as f32 - 3.0) * 0.01).collect();
    let d_x = gpu.upload_f32(&x_data, &[k]).unwrap();
    
    // Prepare all formats
    let d_q4k = gpu.upload_raw(raw, &[raw.len()]).unwrap();
    let a_f32 = hipfire_runtime::llama::dequantize_q4_k(raw, m*k);
    let d_f32 = gpu.upload_f32(&a_f32, &[m, k]).unwrap();
    let q4f16 = hipfire_runtime::llama::convert_q4k_to_q4f16_g64(raw, m*k);
    let d_q4f16 = gpu.upload_raw(&q4f16, &[q4f16.len()]).unwrap();
    
    let n_warmup = 50;
    let n_iter = 500;
    
    // Pre-allocate output buffers
    let d_y1 = gpu.zeros(&[m], rdna_compute::DType::F32).unwrap();
    let d_y2 = gpu.zeros(&[m], rdna_compute::DType::F32).unwrap();
    let d_y3 = gpu.zeros(&[m], rdna_compute::DType::F32).unwrap();
    let d_y4 = gpu.zeros(&[m], rdna_compute::DType::F32).unwrap();
    
    // Warmup all
    for _ in 0..n_warmup {
        gpu.gemv_f32(&d_f32, &d_x, &d_y1).unwrap();
        gpu.gemv_q4k(&d_q4k, &d_x, &d_y2, m, k).unwrap();
        gpu.gemv_q4f16_g64(&d_q4f16, &d_x, &d_y3, m, k).unwrap();
        gpu.gemv_q4f16_g64_wide(&d_q4f16, &d_x, &d_y4, m, k).unwrap();
    }
    
    let start = gpu.hip.event_create().unwrap();
    let stop = gpu.hip.event_create().unwrap();
    
    // F32
    gpu.hip.event_record(&start, None).unwrap();
    for _ in 0..n_iter { gpu.gemv_f32(&d_f32, &d_x, &d_y1).unwrap(); }
    gpu.hip.event_record(&stop, None).unwrap();
    gpu.hip.event_synchronize(&stop).unwrap();
    let ms = gpu.hip.event_elapsed_ms(&start, &stop).unwrap();
    let bytes = (m*k*4 + k*4) as f64;
    let bw = bytes * n_iter as f64 / (ms as f64 / 1000.0) / 1e9;
    eprintln!("F32      (256thr shmem): {:6.1} us  {:6.1} GB/s  {:4.1}% peak", 
        ms*1000.0/n_iter as f32, bw, bw/448.0*100.0);
    
    // Q4_K
    gpu.hip.event_record(&start, None).unwrap();
    for _ in 0..n_iter { gpu.gemv_q4k(&d_q4k, &d_x, &d_y2, m, k).unwrap(); }
    gpu.hip.event_record(&stop, None).unwrap();
    gpu.hip.event_synchronize(&stop).unwrap();
    let ms = gpu.hip.event_elapsed_ms(&start, &stop).unwrap();
    let bytes = (m*(k/256)*144 + k*4) as f64;
    let bw = bytes * n_iter as f64 / (ms as f64 / 1000.0) / 1e9;
    eprintln!("Q4_K     (32thr warp):   {:6.1} us  {:6.1} GB/s  {:4.1}% peak", 
        ms*1000.0/n_iter as f32, bw, bw/448.0*100.0);
    
    // Q4_F16 G64 (32 thread)
    gpu.hip.event_record(&start, None).unwrap();
    for _ in 0..n_iter { gpu.gemv_q4f16_g64(&d_q4f16, &d_x, &d_y3, m, k).unwrap(); }
    gpu.hip.event_record(&stop, None).unwrap();
    gpu.hip.event_synchronize(&stop).unwrap();
    let ms = gpu.hip.event_elapsed_ms(&start, &stop).unwrap();
    let bytes = (m*(k/64)*36 + k*4) as f64;
    let bw = bytes * n_iter as f64 / (ms as f64 / 1000.0) / 1e9;
    eprintln!("Q4_F16   (32thr warp):   {:6.1} us  {:6.1} GB/s  {:4.1}% peak",
        ms*1000.0/n_iter as f32, bw, bw/448.0*100.0);

    // Q4_F16 G64 Wide (256 thread)
    gpu.hip.event_record(&start, None).unwrap();
    for _ in 0..n_iter { gpu.gemv_q4f16_g64_wide(&d_q4f16, &d_x, &d_y4, m, k).unwrap(); }
    gpu.hip.event_record(&stop, None).unwrap();
    gpu.hip.event_synchronize(&stop).unwrap();
    let ms = gpu.hip.event_elapsed_ms(&start, &stop).unwrap();
    let bytes = (m*(k/64)*36 + k*4) as f64;
    let bw = bytes * n_iter as f64 / (ms as f64 / 1000.0) / 1e9;
    eprintln!("Q4_F16_W (256thr shmem): {:6.1} us  {:6.1} GB/s  {:4.1}% peak",
        ms*1000.0/n_iter as f32, bw, bw/448.0*100.0);
    
    // Verify correctness of wide variant
    let y3 = gpu.download_f32(&d_y3).unwrap();
    let y4 = gpu.download_f32(&d_y4).unwrap();
    let max_diff: f32 = y3.iter().zip(y4.iter()).map(|(a,b)| (a-b).abs()).fold(0.0f32, f32::max);
    eprintln!("\nWide vs narrow max diff: {max_diff:.8}");
}
