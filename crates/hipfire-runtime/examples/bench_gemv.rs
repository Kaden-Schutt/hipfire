// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Benchmark Q4K GEMV at various sizes matching real inference workloads.

fn main() {
    let mut gpu = rdna_compute::Gpu::init().unwrap();

    let sizes: Vec<(usize, usize, &str)> = vec![
        (2048, 2048, "attn_q (TinyLlama)"),
        (256, 2048, "attn_k (TinyLlama)"),
        (5632, 2048, "ffn_gate (TinyLlama)"),
        (2048, 5632, "ffn_down (TinyLlama)"),
        (32000, 2048, "output (TinyLlama)"),
        (4096, 4096, "attn_q (Qwen3-8B)"),
        (12288, 4096, "ffn_gate (Qwen3-8B)"),
    ];

    eprintln!("{:<30} {:>8} {:>8} {:>10} {:>8} {:>10} {:>8}",
        "Name", "M", "K", "Q4K us", "Q4K GB/s", "F32 us", "F32 GB/s");
    eprintln!("{}", "-".repeat(90));

    for (m, k, name) in &sizes {
        // Skip sizes where K is not multiple of 256
        if k % 256 != 0 { continue; }

        let m = *m;
        let k = *k;

        // Create synthetic Q4K data
        let blocks_per_row = k / 256;
        let row_bytes = blocks_per_row * 144;
        let total_bytes = m * row_bytes;
        let fake_data = vec![0x55u8; total_bytes];
        let d_raw = gpu.upload_raw(&fake_data, &[total_bytes]).unwrap();
        let x_data: Vec<f32> = vec![0.01; k];
        let d_x = gpu.upload_f32(&x_data, &[k]).unwrap();
        let d_y = gpu.zeros(&[m], rdna_compute::DType::F32).unwrap();

        // Warmup
        gpu.gemv_q4k(&d_raw, &d_x, &d_y, m, k).unwrap();

        // Benchmark Q4K
        let start = gpu.hip.event_create().unwrap();
        let stop = gpu.hip.event_create().unwrap();
        let n = 50;
        gpu.hip.event_record(&start, None).unwrap();
        for _ in 0..n {
            gpu.gemv_q4k(&d_raw, &d_x, &d_y, m, k).unwrap();
        }
        gpu.hip.event_record(&stop, None).unwrap();
        gpu.hip.event_synchronize(&stop).unwrap();
        let ms_q4k = gpu.hip.event_elapsed_ms(&start, &stop).unwrap();
        let us_q4k = ms_q4k * 1000.0 / n as f32;
        let bytes_q4k = (m * row_bytes + k * 4) as f64;
        let bw_q4k = bytes_q4k * n as f64 / (ms_q4k as f64 / 1000.0) / 1e9;

        // Benchmark F32
        let a_f32 = vec![0.01f32; m * k];
        let d_a = gpu.upload_f32(&a_f32, &[m, k]).unwrap();
        let d_y2 = gpu.zeros(&[m], rdna_compute::DType::F32).unwrap();
        gpu.gemv_f32(&d_a, &d_x, &d_y2).unwrap(); // warmup
        gpu.hip.event_record(&start, None).unwrap();
        for _ in 0..n {
            gpu.gemv_f32(&d_a, &d_x, &d_y2).unwrap();
        }
        gpu.hip.event_record(&stop, None).unwrap();
        gpu.hip.event_synchronize(&stop).unwrap();
        let ms_f32 = gpu.hip.event_elapsed_ms(&start, &stop).unwrap();
        let us_f32 = ms_f32 * 1000.0 / n as f32;
        let bytes_f32 = (m * k * 4 + k * 4) as f64;
        let bw_f32 = bytes_f32 * n as f64 / (ms_f32 as f64 / 1000.0) / 1e9;

        eprintln!("{:<30} {:>8} {:>8} {:>10.1} {:>8.1} {:>10.1} {:>8.1}",
            name, m, k, us_q4k, bw_q4k, us_f32, bw_f32);

        gpu.free_tensor(d_raw).unwrap();
        gpu.free_tensor(d_x).unwrap();
        gpu.free_tensor(d_y).unwrap();
        gpu.free_tensor(d_a).unwrap();
        gpu.free_tensor(d_y2).unwrap();
        gpu.hip.event_destroy(start).unwrap();
        gpu.hip.event_destroy(stop).unwrap();
    }
}
