// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! 5-stage correctness + performance harness for Q4_F16 GEMV kernels.
//! Tests both G32 and G64 variants against CPU F32 reference.
//! Compares bandwidth vs Q4_K baseline.

fn main() {
    let mut gpu = rdna_compute::Gpu::init().expect("GPU init failed");

    let path = "/home/kaden/llama.cpp/models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf";
    let gguf = hipfire_runtime::gguf::GgufFile::open(std::path::Path::new(path)).unwrap();

    let tensor_info = gguf.find_tensor("blk.0.attn_q.weight").unwrap();
    let raw_q4k = gguf.tensor_data(tensor_info);
    let m = 2048usize;
    let k = 2048usize;
    let n_elements = m * k;
    eprintln!("Tensor: {} {:?} raw_bytes={}", tensor_info.name, tensor_info.dtype, raw_q4k.len());

    // CPU reference: dequant Q4_K to F32, then GEMV
    let a_f32 = hipfire_runtime::llama::dequantize_q4_k(raw_q4k, n_elements);
    let x_data: Vec<f32> = (0..k).map(|i| ((i % 7) as f32 - 3.0) * 0.01).collect();
    let mut y_ref = vec![0.0f32; m];
    for i in 0..m {
        let mut sum = 0.0f32;
        for j in 0..k {
            sum += a_f32[i * k + j] * x_data[j];
        }
        y_ref[i] = sum;
    }
    let d_x = gpu.upload_f32(&x_data, &[k]).unwrap();

    // Convert to both Q4_F16 formats
    let q4f16_g32 = hipfire_runtime::llama::convert_q4k_to_q4f16_g32(raw_q4k, n_elements);
    let q4f16_g64 = hipfire_runtime::llama::convert_q4k_to_q4f16_g64(raw_q4k, n_elements);
    eprintln!("G32 data: {} bytes (Q4_K: {} bytes, ratio: {:.3}x)",
        q4f16_g32.len(), raw_q4k.len(), q4f16_g32.len() as f64 / raw_q4k.len() as f64);
    eprintln!("G64 data: {} bytes (Q4_K: {} bytes, ratio: {:.3}x)",
        q4f16_g64.len(), raw_q4k.len(), q4f16_g64.len() as f64 / raw_q4k.len() as f64);

    // Upload all formats
    let d_q4k = gpu.upload_raw(raw_q4k, &[raw_q4k.len()]).unwrap();
    let d_g32 = gpu.upload_raw(&q4f16_g32, &[q4f16_g32.len()]).unwrap();
    let d_g64 = gpu.upload_raw(&q4f16_g64, &[q4f16_g64.len()]).unwrap();

    // ═══════════════════════════════════════════════
    // Test Q4_F16_G32
    // ═══════════════════════════════════════════════
    eprintln!("\n============================================================");
    eprintln!("  Q4_F16_G32 (20 bytes / 32 elements, {:.4} bytes/weight)", 20.0 / 32.0);
    eprintln!("============================================================");

    // Stage 1: Smoke
    eprintln!("\n--- Stage 1: Smoke test (G32) ---");
    let d_y = gpu.zeros(&[m], rdna_compute::DType::F32).unwrap();
    gpu.gemv_q4f16_g32(&d_g32, &d_x, &d_y, m, k).unwrap();
    let y_g32 = gpu.download_f32(&d_y).unwrap();
    let has_nan = y_g32.iter().any(|v| v.is_nan());
    let has_inf = y_g32.iter().any(|v| v.is_infinite());
    let all_zero = y_g32.iter().all(|v| *v == 0.0);
    eprintln!("  NaN={has_nan} Inf={has_inf} all_zero={all_zero} y[0]={:.6}", y_g32[0]);
    assert!(!has_nan && !has_inf && !all_zero, "G32 smoke FAIL");
    eprintln!("  PASS");

    // Stage 2: Correctness
    eprintln!("\n--- Stage 2: Correctness vs F32 reference (G32) ---");
    let (max_abs, max_rel, errs) = check_correctness(&y_g32, &y_ref, m);
    eprintln!("  max_abs_err={max_abs:.8} max_rel_err={max_rel:.8} errors(>0.01)={errs}/{m}");
    assert!(max_abs < 0.05, "G32 correctness FAIL: max_abs={max_abs}");
    eprintln!("  PASS");

    // Stage 3: Determinism
    eprintln!("\n--- Stage 3: Determinism (G32) ---");
    let d_y2 = gpu.zeros(&[m], rdna_compute::DType::F32).unwrap();
    gpu.gemv_q4f16_g32(&d_g32, &d_x, &d_y2, m, k).unwrap();
    let y_g32_2 = gpu.download_f32(&d_y2).unwrap();
    let max_diff: f32 = y_g32.iter().zip(y_g32_2.iter()).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
    eprintln!("  max diff between two runs: {max_diff:.8}");
    assert!(max_diff < 1e-6, "G32 determinism FAIL");
    gpu.free_tensor(d_y2).unwrap();
    eprintln!("  PASS");

    // Stage 4: Performance
    eprintln!("\n--- Stage 4: Performance (G32) ---");
    let bw_g32 = bench_gemv(&mut gpu, |gpu| gpu.gemv_q4f16_g32(&d_g32, &d_x, &d_y, m, k),
        m, k, q4f16_g32.len() / m, "Q4_F16_G32");

    gpu.free_tensor(d_y).unwrap();

    // ═══════════════════════════════════════════════
    // Test Q4_F16_G64
    // ═══════════════════════════════════════════════
    eprintln!("\n============================================================");
    eprintln!("  Q4_F16_G64 (36 bytes / 64 elements, {:.4} bytes/weight)", 36.0 / 64.0);
    eprintln!("============================================================");

    // Stage 1: Smoke
    eprintln!("\n--- Stage 1: Smoke test (G64) ---");
    let d_y = gpu.zeros(&[m], rdna_compute::DType::F32).unwrap();
    gpu.gemv_q4f16_g64(&d_g64, &d_x, &d_y, m, k).unwrap();
    let y_g64 = gpu.download_f32(&d_y).unwrap();
    let has_nan = y_g64.iter().any(|v| v.is_nan());
    let has_inf = y_g64.iter().any(|v| v.is_infinite());
    let all_zero = y_g64.iter().all(|v| *v == 0.0);
    eprintln!("  NaN={has_nan} Inf={has_inf} all_zero={all_zero} y[0]={:.6}", y_g64[0]);
    assert!(!has_nan && !has_inf && !all_zero, "G64 smoke FAIL");
    eprintln!("  PASS");

    // Stage 2: Correctness
    eprintln!("\n--- Stage 2: Correctness vs F32 reference (G64) ---");
    let (max_abs, max_rel, errs) = check_correctness(&y_g64, &y_ref, m);
    eprintln!("  max_abs_err={max_abs:.8} max_rel_err={max_rel:.8} errors(>0.01)={errs}/{m}");
    // G64 has coarser grouping so allow slightly more error
    assert!(max_abs < 0.1, "G64 correctness FAIL: max_abs={max_abs}");
    eprintln!("  PASS");

    // Stage 3: Determinism
    eprintln!("\n--- Stage 3: Determinism (G64) ---");
    let d_y2 = gpu.zeros(&[m], rdna_compute::DType::F32).unwrap();
    gpu.gemv_q4f16_g64(&d_g64, &d_x, &d_y2, m, k).unwrap();
    let y_g64_2 = gpu.download_f32(&d_y2).unwrap();
    let max_diff: f32 = y_g64.iter().zip(y_g64_2.iter()).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
    eprintln!("  max diff: {max_diff:.8}");
    assert!(max_diff < 1e-6, "G64 determinism FAIL");
    gpu.free_tensor(d_y2).unwrap();
    eprintln!("  PASS");

    // Stage 4: Performance
    eprintln!("\n--- Stage 4: Performance (G64) ---");
    let bw_g64 = bench_gemv(&mut gpu, |gpu| gpu.gemv_q4f16_g64(&d_g64, &d_x, &d_y, m, k),
        m, k, q4f16_g64.len() / m, "Q4_F16_G64");

    gpu.free_tensor(d_y).unwrap();

    // ═══════════════════════════════════════════════
    // Q4_K baseline for comparison
    // ═══════════════════════════════════════════════
    eprintln!("\n============================================================");
    eprintln!("  Q4_K baseline (144 bytes / 256 elements)");
    eprintln!("============================================================");

    let d_y = gpu.zeros(&[m], rdna_compute::DType::F32).unwrap();
    let bw_q4k = bench_gemv(&mut gpu, |gpu| gpu.gemv_q4k(&d_q4k, &d_x, &d_y, m, k),
        m, k, (k / 256) * 144, "Q4_K");

    // ═══════════════════════════════════════════════
    // Summary
    // ═══════════════════════════════════════════════
    eprintln!("\n============================================================");
    eprintln!("  SUMMARY ({m}x{k} GEMV)");
    eprintln!("============================================================");
    eprintln!("  Q4_K:       {bw_q4k:.1} GB/s ({:.1}% peak)", bw_q4k / 448.0 * 100.0);
    eprintln!("  Q4_F16_G32: {bw_g32:.1} GB/s ({:.1}% peak) [{:.2}x vs Q4_K]",
        bw_g32 / 448.0 * 100.0, bw_g32 / bw_q4k);
    eprintln!("  Q4_F16_G64: {bw_g64:.1} GB/s ({:.1}% peak) [{:.2}x vs Q4_K]",
        bw_g64 / 448.0 * 100.0, bw_g64 / bw_q4k);
    eprintln!("\n=== ALL TESTS PASSED ===");

    // Cleanup
    gpu.free_tensor(d_q4k).unwrap();
    gpu.free_tensor(d_g32).unwrap();
    gpu.free_tensor(d_g64).unwrap();
    gpu.free_tensor(d_x).unwrap();
    gpu.free_tensor(d_y).unwrap();
}

fn check_correctness(y_gpu: &[f32], y_ref: &[f32], m: usize) -> (f32, f32, usize) {
    let mut max_abs: f32 = 0.0;
    let mut max_rel: f32 = 0.0;
    let mut errors = 0;
    for i in 0..m {
        let abs_err = (y_gpu[i] - y_ref[i]).abs();
        max_abs = max_abs.max(abs_err);
        if y_ref[i].abs() > 1e-6 {
            max_rel = max_rel.max(abs_err / y_ref[i].abs());
        }
        if abs_err > 0.01 {
            errors += 1;
            if errors <= 3 {
                eprintln!("  row {i}: gpu={:.6} ref={:.6} err={:.6}", y_gpu[i], y_ref[i], abs_err);
            }
        }
    }
    (max_abs, max_rel, errors)
}

fn bench_gemv<F>(gpu: &mut rdna_compute::Gpu, mut kernel_fn: F,
    m: usize, k: usize, weight_bytes_per_row: usize, label: &str) -> f64
where F: FnMut(&mut rdna_compute::Gpu) -> hip_bridge::HipResult<()>
{
    let n_warmup = 10;
    let n_iter = 200;

    // Warmup
    for _ in 0..n_warmup {
        kernel_fn(gpu).unwrap();
    }

    let start = gpu.hip.event_create().unwrap();
    let stop = gpu.hip.event_create().unwrap();
    gpu.hip.event_record(&start, None).unwrap();
    for _ in 0..n_iter {
        kernel_fn(gpu).unwrap();
    }
    gpu.hip.event_record(&stop, None).unwrap();
    gpu.hip.event_synchronize(&stop).unwrap();
    let ms = gpu.hip.event_elapsed_ms(&start, &stop).unwrap();

    let avg_us = (ms * 1000.0) / n_iter as f32;
    // Total bytes read: weight data + activation vector per row
    let bytes_per_call = (m * weight_bytes_per_row + k * 4) as f64;
    let bandwidth_gbs = (bytes_per_call * n_iter as f64) / (ms as f64 / 1000.0) / 1e9;

    eprintln!("  {label} {m}x{k}: {avg_us:.1} us/call, {bandwidth_gbs:.1} GB/s ({:.1}% peak)",
        bandwidth_gbs / 448.0 * 100.0);

    bandwidth_gbs
}
