// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Correctness + perf test for `gemm_qkvza_hfq4g256`, the batched
//! counterpart of `fused_qkvza_hfq4g256` (Qwen3.5 LA preamble, 4-way).
//!
//! Compares batched GEMM × 1 against the fused GEMV × N on the same
//! synthetic weights, byte-exact. Uses realistic 0.8B LA shapes by
//! default: qkv_m=6144, z_m=2048, beta_m=16, alpha_m=16, K=1024.

use rdna_compute::{DType, Gpu};
use std::time::Instant;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let qkv_m: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(6144);
    let z_m:   usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(2048);
    let beta_m: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(16);
    let alpha_m: usize = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(16);
    let k:     usize = args.get(5).and_then(|s| s.parse().ok()).unwrap_or(1024);
    let n_list: Vec<usize> = if args.len() > 6 {
        args[6..].iter().filter_map(|s| s.parse().ok()).collect()
    } else {
        vec![1, 4, 8, 16, 32, 64]
    };

    assert!(k % 256 == 0, "K must be a multiple of 256 for HFQ4-G256");
    let groups_per_row = k / 256;
    let row_bytes = groups_per_row * 136;

    eprintln!("=== gemm_qkvza_hfq4g256 test ===");
    eprintln!("qkv_m={qkv_m}  z_m={z_m}  beta_m={beta_m}  alpha_m={alpha_m}  K={k}");
    eprintln!("groups_per_row={groups_per_row}, row_bytes={row_bytes}");

    let mut gpu = Gpu::init().expect("gpu init");

    let w_qkv   = gpu.upload_raw(&synth(qkv_m,   groups_per_row, 0xA1),   &[qkv_m   * row_bytes]).unwrap();
    let w_z     = gpu.upload_raw(&synth(z_m,     groups_per_row, 0xB2),   &[z_m     * row_bytes]).unwrap();
    let w_beta  = gpu.upload_raw(&synth(beta_m,  groups_per_row, 0xC3),   &[beta_m  * row_bytes]).unwrap();
    let w_alpha = gpu.upload_raw(&synth(alpha_m, groups_per_row, 0xD4),   &[alpha_m * row_bytes]).unwrap();

    let max_n = *n_list.iter().max().unwrap();
    let x_host: Vec<f32> = (0..max_n * k)
        .map(|i| {
            let v = ((i as i64).wrapping_mul(1103515245).wrapping_add(12345)) as f32;
            (v * 1e-9) % 2.0 - 1.0
        })
        .collect();

    // GEMV path scratch buffers (single-token inputs/outputs).
    let x_gemv = gpu.alloc_tensor(&[k], DType::F32).unwrap();
    let y_qkv_1   = gpu.alloc_tensor(&[qkv_m],   DType::F32).unwrap();
    let y_z_1     = gpu.alloc_tensor(&[z_m],     DType::F32).unwrap();
    let y_beta_1  = gpu.alloc_tensor(&[beta_m],  DType::F32).unwrap();
    let y_alpha_1 = gpu.alloc_tensor(&[alpha_m], DType::F32).unwrap();

    // Collected GEMV outputs across all N batch elements.
    let y_qkv_gemv_col   = gpu.alloc_tensor(&[max_n * qkv_m],   DType::F32).unwrap();
    let y_z_gemv_col     = gpu.alloc_tensor(&[max_n * z_m],     DType::F32).unwrap();
    let y_beta_gemv_col  = gpu.alloc_tensor(&[max_n * beta_m],  DType::F32).unwrap();
    let y_alpha_gemv_col = gpu.alloc_tensor(&[max_n * alpha_m], DType::F32).unwrap();

    // Batched GEMM path.
    let x_gemm       = gpu.alloc_tensor(&[max_n * k],     DType::F32).unwrap();
    let y_qkv_gemm   = gpu.alloc_tensor(&[max_n * qkv_m], DType::F32).unwrap();
    let y_z_gemm     = gpu.alloc_tensor(&[max_n * z_m],   DType::F32).unwrap();
    let y_beta_gemm  = gpu.alloc_tensor(&[max_n * beta_m], DType::F32).unwrap();
    let y_alpha_gemm = gpu.alloc_tensor(&[max_n * alpha_m], DType::F32).unwrap();

    gpu.hip.memcpy_htod(&x_gemm.buf, bytes_of(&x_host)).unwrap();

    for &n in &n_list {
        eprintln!("\n--- N = {n} ---");

        // GEMV × N
        let mut gemv_us: f64 = 0.0;
        for i in 0..n {
            gpu.hip.memcpy_htod(&x_gemv.buf, bytes_of(&x_host[i * k..(i + 1) * k])).unwrap();
            gpu.hip.device_synchronize().unwrap();
            let t = Instant::now();
            gpu.fused_qkvza_hfq4g256(
                &w_qkv, &w_z, &w_beta, &w_alpha,
                &x_gemv,
                &y_qkv_1, &y_z_1, &y_beta_1, &y_alpha_1,
                qkv_m, z_m, beta_m, alpha_m,
                k,
            ).unwrap();
            gpu.hip.device_synchronize().unwrap();
            gemv_us += t.elapsed().as_secs_f64() * 1e6;

            gpu.hip.memcpy_dtod_at(&y_qkv_gemv_col.buf,   i * qkv_m   * 4, &y_qkv_1.buf,   0, qkv_m   * 4).unwrap();
            gpu.hip.memcpy_dtod_at(&y_z_gemv_col.buf,     i * z_m     * 4, &y_z_1.buf,     0, z_m     * 4).unwrap();
            gpu.hip.memcpy_dtod_at(&y_beta_gemv_col.buf,  i * beta_m  * 4, &y_beta_1.buf,  0, beta_m  * 4).unwrap();
            gpu.hip.memcpy_dtod_at(&y_alpha_gemv_col.buf, i * alpha_m * 4, &y_alpha_1.buf, 0, alpha_m * 4).unwrap();
        }

        // GEMM × 1
        gpu.hip.device_synchronize().unwrap();
        let t = Instant::now();
        gpu.gemm_qkvza_hfq4g256(
            &w_qkv, &w_z, &w_beta, &w_alpha,
            &x_gemm,
            &y_qkv_gemm, &y_z_gemm, &y_beta_gemm, &y_alpha_gemm,
            qkv_m, z_m, beta_m, alpha_m,
            k, n,
        ).unwrap();
        gpu.hip.device_synchronize().unwrap();
        let gemm_us = t.elapsed().as_secs_f64() * 1e6;

        // Compare each of the 4 outputs byte-exact.
        let compare = |label: &str, col: &rdna_compute::GpuTensor, gemm: &rdna_compute::GpuTensor, m: usize| -> bool {
            let a = gpu.download_f32(col).unwrap()[..n * m].to_vec();
            let b = gpu.download_f32(gemm).unwrap()[..n * m].to_vec();
            for i in 0..n * m {
                if a[i].to_bits() != b[i].to_bits() {
                    let batch = i / m;
                    let row = i % m;
                    eprintln!(
                        "  {label}: DIVERGENT at batch={batch} row={row}  gemv={:.6e} ({:#010x})  gemm={:.6e} ({:#010x})",
                        a[i], a[i].to_bits(), b[i], b[i].to_bits()
                    );
                    let count: usize = a.iter().zip(b.iter()).filter(|(a, b)| a.to_bits() != b.to_bits()).count();
                    eprintln!("  {label}: {count}/{} elements diverged", n * m);
                    return false;
                }
            }
            true
        };

        let ok_qkv   = compare("qkv",   &y_qkv_gemv_col,   &y_qkv_gemm,   qkv_m);
        let ok_z     = compare("z",     &y_z_gemv_col,     &y_z_gemm,     z_m);
        let ok_beta  = compare("beta",  &y_beta_gemv_col,  &y_beta_gemm,  beta_m);
        let ok_alpha = compare("alpha", &y_alpha_gemv_col, &y_alpha_gemm, alpha_m);

        let all_ok = ok_qkv && ok_z && ok_beta && ok_alpha;
        let status = if all_ok { "byte-exact OK" } else { "DIVERGENT" };
        let speedup = gemv_us / gemm_us;
        eprintln!(
            "  gemv × {n}: {:8.1} µs   gemm × 1: {:8.1} µs   speedup: {:5.2}x   [{status}]",
            gemv_us, gemm_us, speedup
        );
        if !all_ok {
            std::process::exit(1);
        }
    }

    eprintln!("\n=== All N passed byte-exact ===");
}

fn synth(m: usize, groups_per_row: usize, seed: u64) -> Vec<u8> {
    let total = m * groups_per_row * 136;
    let mut out = vec![0u8; total];
    let mut state = seed;
    let mut next = || {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (state >> 33) as u32
    };
    for row in 0..m {
        for g in 0..groups_per_row {
            let gp = (row * groups_per_row + g) * 136;
            let scale_exp: u32 = 0x43 + (next() & 0x7);
            let scale_bits = (scale_exp << 23) | (next() & 0x007F_FFFF);
            let zp_bits = ((next() & 0xFF) << 23) | (next() & 0x007F_FFFF);
            let scale = f32::from_bits(scale_bits);
            let zp = f32::from_bits(zp_bits);
            let scale_ok = if scale.is_finite() && scale.abs() < 1e-2 && scale > 0.0 { scale } else { 1e-3 };
            let zp_ok    = if zp.is_finite() && zp.abs() < 1.0 { zp } else { -0.5 };
            out[gp..gp + 4].copy_from_slice(&scale_ok.to_le_bytes());
            out[gp + 4..gp + 8].copy_from_slice(&zp_ok.to_le_bytes());
            for i in 0..128 {
                out[gp + 8 + i] = (next() & 0xFF) as u8;
            }
        }
    }
    out
}

fn bytes_of(v: &[f32]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 4) }
}
