// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! 5-stage correctness harness for gemv_q4k kernel.
//! Compares GPU quantized GEMV against CPU F32 dequant + GEMV reference.

fn main() {
    let mut gpu = rdna_compute::Gpu::init().expect("GPU init failed");

    // Load a real Q4_K tensor from the TinyLlama model for realistic data
    let path = "/home/kaden/llama.cpp/models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf";
    let gguf = hipfire_runtime::gguf::GgufFile::open(std::path::Path::new(path)).unwrap();

    // Use blk.0.attn_q.weight: Q4_K, shape [2048, 2048] in GGUF = GEMV [2048, 2048]
    let tensor_info = gguf.find_tensor("blk.0.attn_q.weight").unwrap();
    let raw_data = gguf.tensor_data(tensor_info);
    let m = 2048usize; // output dim (ne[1])
    let k = 2048usize; // input dim (ne[0])
    eprintln!("Tensor: {} {:?} {:?} raw_bytes={}", tensor_info.name, tensor_info.dtype, tensor_info.shape, raw_data.len());

    // ═══ Stage 1: Smoke test ═══
    eprintln!("\n=== Stage 1: Smoke test ===");
    let d_raw = gpu.upload_raw(raw_data, &[raw_data.len()]).unwrap();
    let x_data: Vec<f32> = (0..k).map(|i| ((i % 7) as f32 - 3.0) * 0.01).collect();
    let d_x = gpu.upload_f32(&x_data, &[k]).unwrap();
    let d_y = gpu.zeros(&[m], rdna_compute::DType::F32).unwrap();

    gpu.gemv_q4k(&d_raw, &d_x, &d_y, m, k).unwrap();
    let y_gpu = gpu.download_f32(&d_y).unwrap();

    let has_nan = y_gpu.iter().any(|v| v.is_nan());
    let has_inf = y_gpu.iter().any(|v| v.is_infinite());
    let all_zero = y_gpu.iter().all(|v| *v == 0.0);
    eprintln!("  NaN: {has_nan}, Inf: {has_inf}, all_zero: {all_zero}");
    assert!(!has_nan, "Stage 1 FAIL: NaN in output");
    assert!(!has_inf, "Stage 1 FAIL: Inf in output");
    assert!(!all_zero, "Stage 1 FAIL: all zeros");
    eprintln!("  PASS");

    // ═══ Stage 2: Numerical correctness vs CPU reference ═══
    eprintln!("\n=== Stage 2: Correctness vs CPU F32 reference ===");
    // CPU reference: dequantize + GEMV
    let a_f32 = hipfire_runtime::llama::dequantize_q4_k(raw_data, m * k);
    let mut y_ref = vec![0.0f32; m];
    for i in 0..m {
        let mut sum = 0.0f32;
        for j in 0..k {
            sum += a_f32[i * k + j] * x_data[j];
        }
        y_ref[i] = sum;
    }

    let mut max_abs_err: f32 = 0.0;
    let mut max_rel_err: f32 = 0.0;
    let mut errors = 0;
    for i in 0..m {
        let abs_err = (y_gpu[i] - y_ref[i]).abs();
        max_abs_err = max_abs_err.max(abs_err);
        if y_ref[i].abs() > 1e-6 {
            max_rel_err = max_rel_err.max(abs_err / y_ref[i].abs());
        }
        if abs_err > 0.01 {
            errors += 1;
            if errors <= 3 {
                eprintln!("  row {i}: gpu={:.6} ref={:.6} err={:.6}", y_gpu[i], y_ref[i], abs_err);
            }
        }
    }
    eprintln!("  max_abs_err={max_abs_err:.8} max_rel_err={max_rel_err:.8} errors={errors}/{m}");
    assert!(max_abs_err < 0.05, "Stage 2 FAIL: max_abs_err={max_abs_err}");
    eprintln!("  PASS");

    // ═══ Stage 3: Shape sweep ═══
    eprintln!("\n=== Stage 3: Shape sweep ===");
    // Test with different k values from the model (256, 512, 1024, 2048, 5632)
    for &test_k in &[256, 512, 1024, 2048] {
        let test_m = 64;
        // Create synthetic Q4_K data
        let blocks_per_row = test_k / 256;
        let row_bytes = blocks_per_row * 144;
        let total_bytes = test_m * row_bytes;
        let fake_data = vec![0x55u8; total_bytes]; // all same nibble pattern
        let d_fake = gpu.upload_raw(&fake_data, &[total_bytes]).unwrap();
        let x_ones: Vec<f32> = vec![1.0; test_k];
        let d_x2 = gpu.upload_f32(&x_ones, &[test_k]).unwrap();
        let d_y2 = gpu.zeros(&[test_m], rdna_compute::DType::F32).unwrap();

        gpu.gemv_q4k(&d_fake, &d_x2, &d_y2, test_m, test_k).unwrap();
        let y2 = gpu.download_f32(&d_y2).unwrap();
        let has_nan = y2.iter().any(|v| v.is_nan());
        eprintln!("  M={test_m} K={test_k}: nan={has_nan} y[0]={:.6}", y2[0]);
        assert!(!has_nan, "Stage 3 FAIL: NaN for M={test_m} K={test_k}");

        gpu.free_tensor(d_fake).unwrap();
        gpu.free_tensor(d_x2).unwrap();
        gpu.free_tensor(d_y2).unwrap();
    }
    eprintln!("  PASS");

    // ═══ Stage 4: Determinism ═══
    eprintln!("\n=== Stage 4: Determinism ===");
    let d_y3 = gpu.zeros(&[m], rdna_compute::DType::F32).unwrap();
    gpu.gemv_q4k(&d_raw, &d_x, &d_y3, m, k).unwrap();
    let y3 = gpu.download_f32(&d_y3).unwrap();
    let max_diff: f32 = y_gpu.iter().zip(y3.iter()).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
    eprintln!("  max diff between two runs: {max_diff:.8}");
    assert!(max_diff < 1e-6, "Stage 4 FAIL: non-deterministic");
    gpu.free_tensor(d_y3).unwrap();
    eprintln!("  PASS");

    // ═══ Stage 5: Performance baseline ═══
    eprintln!("\n=== Stage 5: Performance ===");
    let start = gpu.hip.event_create().unwrap();
    let stop = gpu.hip.event_create().unwrap();
    let n_iter = 100;

    gpu.hip.event_record(&start, None).unwrap();
    for _ in 0..n_iter {
        gpu.gemv_q4k(&d_raw, &d_x, &d_y, m, k).unwrap();
    }
    gpu.hip.event_record(&stop, None).unwrap();
    gpu.hip.event_synchronize(&stop).unwrap();
    let ms = gpu.hip.event_elapsed_ms(&start, &stop).unwrap();

    let avg_us = (ms * 1000.0) / n_iter as f32;
    // Data read per GEMV: M * K/256 * 144 bytes (Q4K) + K * 4 bytes (x)
    let bytes_read = (m * (k / 256) * 144 + k * 4) as f64;
    let bandwidth_gbs = (bytes_read * n_iter as f64) / (ms as f64 / 1000.0) / 1e9;
    let pct_peak = bandwidth_gbs / 448.0 * 100.0;

    eprintln!("  {m}x{k} Q4_K GEMV: {avg_us:.1} us/call ({n_iter} iterations)");
    eprintln!("  Effective bandwidth: {bandwidth_gbs:.1} GB/s ({pct_peak:.1}% of 448 GB/s peak)");

    // Compare with F32 GEMV
    let a_gpu = gpu.upload_f32(&a_f32, &[m, k]).unwrap();
    let d_y4 = gpu.zeros(&[m], rdna_compute::DType::F32).unwrap();
    gpu.hip.event_record(&start, None).unwrap();
    for _ in 0..n_iter {
        gpu.gemv_f32(&a_gpu, &d_x, &d_y4).unwrap();
    }
    gpu.hip.event_record(&stop, None).unwrap();
    gpu.hip.event_synchronize(&stop).unwrap();
    let ms_f32 = gpu.hip.event_elapsed_ms(&start, &stop).unwrap();
    let avg_us_f32 = (ms_f32 * 1000.0) / n_iter as f32;
    let bytes_f32 = (m * k * 4 + k * 4) as f64;
    let bw_f32 = (bytes_f32 * n_iter as f64) / (ms_f32 as f64 / 1000.0) / 1e9;

    eprintln!("  {m}x{k} F32 GEMV:  {avg_us_f32:.1} us/call");
    eprintln!("  F32 bandwidth: {bw_f32:.1} GB/s ({:.1}% of peak)", bw_f32 / 448.0 * 100.0);
    eprintln!("  Q4K speedup: {:.2}x (data is {:.1}x smaller)", avg_us_f32 / avg_us, (m * k * 4) as f64 / (m * (k / 256) * 144) as f64);

    // Cleanup
    gpu.free_tensor(d_raw).unwrap();
    gpu.free_tensor(d_x).unwrap();
    gpu.free_tensor(d_y).unwrap();
    gpu.free_tensor(a_gpu).unwrap();
    gpu.free_tensor(d_y4).unwrap();

    eprintln!("\n=== ALL 5 STAGES PASSED ===");

    // TSV output for results.tsv
    println!("2\ttask2\tgemv_q4k\t-\t{bandwidth_gbs:.1}\t{pct_peak:.1}\tPASS\t-\tbaseline Q4K GEMV {m}x{k}: {avg_us:.1}us, {bandwidth_gbs:.1} GB/s\tYES");
}
