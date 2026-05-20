// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Benchmark Q8_0 vs Q4_K vs F32 — testing the occupancy hypothesis.
//! Q8 = byte loads, no nibble extraction → fewer VGPRs → more waves → higher bandwidth.
fn main() {
    let mut gpu = rdna_compute::Gpu::init().expect("GPU init");
    
    let path = "/home/kaden/llama.cpp/models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf";
    let gguf = hipfire_runtime::gguf::GgufFile::open(std::path::Path::new(path)).unwrap();
    let ti = gguf.find_tensor("blk.0.attn_q.weight").unwrap();
    let raw_q4k = gguf.tensor_data(ti);
    let m = 2048usize;
    let k = 2048usize;
    
    // Dequant Q4_K to F32 for reference
    let a_f32 = hipfire_runtime::llama::dequantize_q4_k(raw_q4k, m * k);
    let x_data: Vec<f32> = (0..k).map(|i| ((i % 7) as f32 - 3.0) * 0.01).collect();
    let d_x = gpu.upload_f32(&x_data, &[k]).unwrap();
    
    // Create synthetic Q8_0 data from the F32 weights
    // Q8_0 block: 2 bytes f16 scale + 32 bytes int8 = 34 bytes per 32 elements
    let mut q8_data = Vec::new();
    for block_start in (0..m*k).step_by(32) {
        let block_end = (block_start + 32).min(m * k);
        let block = &a_f32[block_start..block_end];
        
        let max_abs = block.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
        let scale = max_abs / 127.0;
        let inv_scale = if scale > 0.0 { 1.0 / scale } else { 0.0 };
        
        // Write f16 scale
        let scale_f16 = hipfire_runtime::llama::f32_to_f16(scale);
        q8_data.extend_from_slice(&scale_f16.to_le_bytes());
        
        // Write int8 quantized values
        for i in 0..32 {
            let val = if block_start + i < m * k { block[i] } else { 0.0 };
            let q = (val * inv_scale).round().max(-128.0).min(127.0) as i8;
            q8_data.push(q as u8);
        }
    }
    eprintln!("Q8_0 data: {} bytes ({:.3} bytes/weight)", q8_data.len(), q8_data.len() as f64 / (m*k) as f64);
    
    // Upload everything
    let d_q4k = gpu.upload_raw(raw_q4k, &[raw_q4k.len()]).unwrap();
    let d_q8 = gpu.upload_raw(&q8_data, &[q8_data.len()]).unwrap();
    let d_f32 = gpu.upload_f32(&a_f32, &[m, k]).unwrap();
    
    let d_y1 = gpu.zeros(&[m], rdna_compute::DType::F32).unwrap();
    let d_y2 = gpu.zeros(&[m], rdna_compute::DType::F32).unwrap();
    let d_y3 = gpu.zeros(&[m], rdna_compute::DType::F32).unwrap();
    
    let n_warmup = 50;
    let n_iter = 500;
    
    // Warmup
    for _ in 0..n_warmup {
        gpu.gemv_f32(&d_f32, &d_x, &d_y1).unwrap();
        gpu.gemv_q4k(&d_q4k, &d_x, &d_y2, m, k).unwrap();
        gpu.gemv_q8_0(&d_q8, &d_x, &d_y3, m, k).unwrap();
    }
    
    let start = gpu.hip.event_create().unwrap();
    let stop = gpu.hip.event_create().unwrap();
    
    // F32
    gpu.hip.event_record(&start, None).unwrap();
    for _ in 0..n_iter { gpu.gemv_f32(&d_f32, &d_x, &d_y1).unwrap(); }
    gpu.hip.event_record(&stop, None).unwrap();
    gpu.hip.event_synchronize(&stop).unwrap();
    let ms = gpu.hip.event_elapsed_ms(&start, &stop).unwrap();
    let f32_bytes = (m * k * 4 + k * 4) as f64;
    let f32_bw = f32_bytes * n_iter as f64 / (ms as f64 / 1000.0) / 1e9;
    let f32_us = ms * 1000.0 / n_iter as f32;
    eprintln!("F32   (4.00 B/w, 256thr): {:6.1}us  {:6.1} GB/s  {:4.1}% peak", f32_us, f32_bw, f32_bw/448.0*100.0);
    
    // Q4_K
    gpu.hip.event_record(&start, None).unwrap();
    for _ in 0..n_iter { gpu.gemv_q4k(&d_q4k, &d_x, &d_y2, m, k).unwrap(); }
    gpu.hip.event_record(&stop, None).unwrap();
    gpu.hip.event_synchronize(&stop).unwrap();
    let ms = gpu.hip.event_elapsed_ms(&start, &stop).unwrap();
    let q4k_bytes = (m * (k/256) * 144 + k * 4) as f64;
    let q4k_bw = q4k_bytes * n_iter as f64 / (ms as f64 / 1000.0) / 1e9;
    let q4k_us = ms * 1000.0 / n_iter as f32;
    eprintln!("Q4_K  (0.56 B/w, 32thr):  {:6.1}us  {:6.1} GB/s  {:4.1}% peak", q4k_us, q4k_bw, q4k_bw/448.0*100.0);

    // Q8_0
    gpu.hip.event_record(&start, None).unwrap();
    for _ in 0..n_iter { gpu.gemv_q8_0(&d_q8, &d_x, &d_y3, m, k).unwrap(); }
    gpu.hip.event_record(&stop, None).unwrap();
    gpu.hip.event_synchronize(&stop).unwrap();
    let ms = gpu.hip.event_elapsed_ms(&start, &stop).unwrap();
    let q8_bytes = (m * (k/32) * 34 + k * 4) as f64;
    let q8_bw = q8_bytes * n_iter as f64 / (ms as f64 / 1000.0) / 1e9;
    let q8_us = ms * 1000.0 / n_iter as f32;
    eprintln!("Q8_0  (1.06 B/w, 32thr):  {:6.1}us  {:6.1} GB/s  {:4.1}% peak", q8_us, q8_bw, q8_bw/448.0*100.0);
    
    // Create Q8_HFQ data (split-metadata layout)
    let n_groups = k / 32;
    let scales_bytes = n_groups * 2;
    let raw_row = scales_bytes + k;
    let row_stride = (raw_row + 127) & !127;
    let mut q8hfq_data = vec![0u8; m * row_stride];
    for row in 0..m {
        for g in 0..n_groups {
            let start = row * k + g * 32;
            let block = &a_f32[start..start + 32];
            let max_abs = block.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
            let scale = max_abs / 127.0;
            let inv_scale = if scale > 0.0 { 1.0 / scale } else { 0.0 };
            let row_off = row * row_stride;
            let scale_f16 = hipfire_runtime::llama::f32_to_f16(scale);
            q8hfq_data[row_off + g * 2..row_off + g * 2 + 2].copy_from_slice(&scale_f16.to_le_bytes());
            for i in 0..32 {
                let q = (block[i] * inv_scale).round().max(-128.0).min(127.0) as i8;
                q8hfq_data[row_off + scales_bytes + g * 32 + i] = q as u8;
            }
        }
    }
    eprintln!("Q8HFQ data: {} bytes ({:.3} bytes/weight, stride={})", q8hfq_data.len(), q8hfq_data.len() as f64 / (m*k) as f64, row_stride);

    let d_q8hfq = gpu.upload_raw(&q8hfq_data, &[q8hfq_data.len()]).unwrap();
    let d_y4 = gpu.zeros(&[m], rdna_compute::DType::F32).unwrap();

    // Warmup Q8HFQ
    for _ in 0..n_warmup {
        gpu.gemv_q8hfq(&d_q8hfq, &d_x, &d_y4, m, k, row_stride).unwrap();
    }

    // Q8_HFQ
    gpu.hip.event_record(&start, None).unwrap();
    for _ in 0..n_iter { gpu.gemv_q8hfq(&d_q8hfq, &d_x, &d_y4, m, k, row_stride).unwrap(); }
    gpu.hip.event_record(&stop, None).unwrap();
    gpu.hip.event_synchronize(&stop).unwrap();
    let ms = gpu.hip.event_elapsed_ms(&start, &stop).unwrap();
    let q8hfq_bytes = (m * row_stride + k * 4) as f64;
    let q8hfq_bw = q8hfq_bytes * n_iter as f64 / (ms as f64 / 1000.0) / 1e9;
    let q8hfq_us = ms * 1000.0 / n_iter as f32;
    eprintln!("Q8HFQ (split, 32thr):     {:6.1}us  {:6.1} GB/s  {:4.1}% peak", q8hfq_us, q8hfq_bw, q8hfq_bw/448.0*100.0);

    // Effective throughput comparison (elements/second)
    let f32_elems = m as f64 * k as f64 * n_iter as f64 / (f32_us as f64 * n_iter as f64 / 1e6);
    let q4k_elems = m as f64 * k as f64 * n_iter as f64 / (q4k_us as f64 * n_iter as f64 / 1e6);
    let q8_elems = m as f64 * k as f64 * n_iter as f64 / (q8_us as f64 * n_iter as f64 / 1e6);
    let q8hfq_elems = m as f64 * k as f64 * n_iter as f64 / (q8hfq_us as f64 * n_iter as f64 / 1e6);

    eprintln!("\n=== Effective throughput (what matters for tok/s) ===");
    eprintln!("F32:   {:6.1}us/GEMV → {:.1} Gelem/s", f32_us, f32_elems / 1e9);
    eprintln!("Q4_K:  {:6.1}us/GEMV → {:.1} Gelem/s  ({:.1}x vs F32 time)", q4k_us, q4k_elems / 1e9, f32_us / q4k_us);
    eprintln!("Q8_0:  {:6.1}us/GEMV → {:.1} Gelem/s  ({:.1}x vs F32 time)", q8_us, q8_elems / 1e9, f32_us / q8_us);
    eprintln!("Q8HFQ: {:6.1}us/GEMV → {:.1} Gelem/s  ({:.1}x vs F32 time)", q8hfq_us, q8hfq_elems / 1e9, f32_us / q8hfq_us);

    // Model-level estimate
    eprintln!("\n=== TinyLlama 1.1B tok/s estimate ===");
    let gemvs_per_token = 220.0; // ~10 GEMVs × 22 layers
    let other_overhead_ms = 1.0; // non-GEMV ops

    let q4k_tok = 1000.0 / (q4k_us as f64 / 1000.0 * gemvs_per_token + other_overhead_ms);
    let q8_tok = 1000.0 / (q8_us as f64 / 1000.0 * gemvs_per_token + other_overhead_ms);
    let q8hfq_tok = 1000.0 / (q8hfq_us as f64 / 1000.0 * gemvs_per_token + other_overhead_ms);
    eprintln!("Q4_K:  ~{:.0} tok/s  (GEMV={:.1}ms + overhead={:.1}ms)", q4k_tok, q4k_us as f64 / 1000.0 * gemvs_per_token, other_overhead_ms);
    eprintln!("Q8_0:  ~{:.0} tok/s  (GEMV={:.1}ms + overhead={:.1}ms)", q8_tok, q8_us as f64 / 1000.0 * gemvs_per_token, other_overhead_ms);
    eprintln!("Q8HFQ: ~{:.0} tok/s  (GEMV={:.1}ms + overhead={:.1}ms)", q8hfq_tok, q8hfq_us as f64 / 1000.0 * gemvs_per_token, other_overhead_ms);

    // Verify correctness
    let y_ref = gpu.download_f32(&d_y1).unwrap();
    let y_q8 = gpu.download_f32(&d_y3).unwrap();
    let y_hfq = gpu.download_f32(&d_y4).unwrap();
    let max_err_q8: f32 = y_ref.iter().zip(y_q8.iter()).map(|(a,b)| (a-b).abs()).fold(0.0f32, f32::max);
    let max_err_hfq: f32 = y_ref.iter().zip(y_hfq.iter()).map(|(a,b)| (a-b).abs()).fold(0.0f32, f32::max);
    eprintln!("\nQ8_0  vs F32 max error: {max_err_q8:.6}");
    eprintln!("Q8HFQ vs F32 max error: {max_err_hfq:.6}");
}
