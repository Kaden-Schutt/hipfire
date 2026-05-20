// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Benchmark batched GEMM vs repeated GEMV for HFQ4-G256.
//! Verifies correctness at batch=1 then measures throughput scaling.

fn main() {
    let mut gpu = rdna_compute::Gpu::init().unwrap();

    // Qwen3-8B dimensions
    let test_cases: &[(usize, usize, &str)] = &[
        (4096, 4096, "8B attn (q_proj)"),
        (12288, 4096, "8B FFN (gate_proj)"),
        (4096, 12288, "8B FFN (down_proj)"),
    ];

    for &(m, k, name) in test_cases {
        eprintln!("\n=== {name} [{m}×{k}] ===");

        // Create HFQ4-G256 weight data (136 bytes per 256 elements)
        let groups_per_row = k / 256;
        let row_bytes = groups_per_row * 136;
        let total_bytes = m * row_bytes;

        // Fill with known pattern: scale=0.01, zero=-0.005, nibbles=alternating
        let mut weight_data = vec![0u8; total_bytes];
        for r in 0..m {
            for g in 0..groups_per_row {
                let off = r * row_bytes + g * 136;
                let scale_bytes = 0.01f32.to_le_bytes();
                let zero_bytes = (-0.005f32).to_le_bytes();
                weight_data[off..off+4].copy_from_slice(&scale_bytes);
                weight_data[off+4..off+8].copy_from_slice(&zero_bytes);
                for i in 0..64 {
                    weight_data[off + 8 + i] = 0x53; // nibbles: 3 and 5
                }
            }
        }
        let d_a = gpu.upload_raw(&weight_data, &[total_bytes]).unwrap();

        // ─── Correctness: batch=1 GEMM vs GEMV ───
        let x_data: Vec<f32> = (0..k).map(|i| 0.001 * (i % 100) as f32).collect();
        let d_x = gpu.upload_f32(&x_data, &[k]).unwrap();
        let d_y_gemv = gpu.zeros(&[m], rdna_compute::DType::F32).unwrap();
        let d_y_gemm = gpu.zeros(&[m], rdna_compute::DType::F32).unwrap();

        gpu.gemv_hfq4g256(&d_a, &d_x, &d_y_gemv, m, k).unwrap();
        gpu.gemm_hfq4g256(&d_a, &d_x, &d_y_gemm, m, k, 1).unwrap();

        let y_gemv = gpu.download_f32(&d_y_gemv).unwrap();
        let y_gemm = gpu.download_f32(&d_y_gemm).unwrap();

        let max_diff: f32 = y_gemv.iter().zip(y_gemm.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        let any_nan = y_gemm.iter().any(|v| v.is_nan());
        eprintln!("  batch=1 correctness: max_diff={max_diff:.6e} nan={any_nan}");
        assert!(max_diff < 1e-5, "GEMM batch=1 does not match GEMV! max_diff={max_diff}");
        assert!(!any_nan, "GEMM produced NaN!");

        gpu.free_tensor(d_x).unwrap();
        gpu.free_tensor(d_y_gemv).unwrap();
        gpu.free_tensor(d_y_gemm).unwrap();

        // ─── Throughput scaling ───
        let batches = [1, 4, 8, 16, 20, 32];
        let n_iters = 100;

        // GEMV baseline (repeated single-vector calls)
        let d_x1 = gpu.upload_f32(&vec![0.01f32; k], &[k]).unwrap();
        let d_y1 = gpu.zeros(&[m], rdna_compute::DType::F32).unwrap();
        // Warmup
        for _ in 0..10 { gpu.gemv_hfq4g256(&d_a, &d_x1, &d_y1, m, k).unwrap(); }

        let start = gpu.hip.event_create().unwrap();
        let stop = gpu.hip.event_create().unwrap();

        // Single GEMV timing
        gpu.hip.event_record(&start, None).unwrap();
        for _ in 0..n_iters { gpu.gemv_hfq4g256(&d_a, &d_x1, &d_y1, m, k).unwrap(); }
        gpu.hip.event_record(&stop, None).unwrap();
        gpu.hip.event_synchronize(&stop).unwrap();
        let ms_gemv = gpu.hip.event_elapsed_ms(&start, &stop).unwrap();
        let us_gemv = ms_gemv * 1000.0 / n_iters as f32;

        eprintln!("  GEMV (1 vector):  {us_gemv:.1}µs");

        for &bs in &batches {
            let d_xb = gpu.upload_f32(&vec![0.01f32; k * bs], &[bs, k]).unwrap();
            let d_yb = gpu.zeros(&[bs * m], rdna_compute::DType::F32).unwrap();
            // Warmup
            for _ in 0..5 { gpu.gemm_hfq4g256(&d_a, &d_xb, &d_yb, m, k, bs).unwrap(); }

            gpu.hip.event_record(&start, None).unwrap();
            for _ in 0..n_iters { gpu.gemm_hfq4g256(&d_a, &d_xb, &d_yb, m, k, bs).unwrap(); }
            gpu.hip.event_record(&stop, None).unwrap();
            gpu.hip.event_synchronize(&stop).unwrap();
            let ms_batch = gpu.hip.event_elapsed_ms(&start, &stop).unwrap();
            let us_batch = ms_batch * 1000.0 / n_iters as f32;
            let us_per_vec = us_batch / bs as f32;
            let speedup = us_gemv / us_per_vec;
            let toks_per_sec = bs as f32 * 1_000_000.0 / us_batch;

            eprintln!("  batch={bs:>2}: {us_batch:>8.1}µs total, {us_per_vec:>6.1}µs/vec, {speedup:>5.1}x vs GEMV, {toks_per_sec:>8.0} effective tok/s");

            gpu.free_tensor(d_xb).unwrap();
            gpu.free_tensor(d_yb).unwrap();
        }

        gpu.free_tensor(d_x1).unwrap();
        gpu.free_tensor(d_y1).unwrap();
        gpu.free_tensor(d_a).unwrap();
        gpu.hip.event_destroy(start).unwrap();
        gpu.hip.event_destroy(stop).unwrap();
    }

    eprintln!("\n=== ALL TESTS PASSED ===");
}
