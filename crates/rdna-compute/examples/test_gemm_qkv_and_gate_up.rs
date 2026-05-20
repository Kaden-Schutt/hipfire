// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Compact correctness + perf test for the two smaller batched GEMMs:
//!   * gemm_qkv_hfq4g256     (FA preamble, 3-way)
//!   * gemm_gate_up_hfq4g256 (FFN preamble, 2-way)
//!
//! Compares batched GEMM × 1 against the fused GEMV × N on synthetic
//! random HFQ4-G256 weights. Byte-exact required.

use rdna_compute::{DType, Gpu, GpuTensor};
use std::time::Instant;

fn main() {
    let n_list: Vec<usize> = vec![1, 4, 16, 64];
    let k: usize = 1024;
    let groups_per_row = k / 256;
    let row_bytes = groups_per_row * 136;

    let mut gpu = Gpu::init().expect("gpu init");

    test_qkv(&mut gpu, &n_list, k, row_bytes);
    test_gate_up(&mut gpu, &n_list, k, row_bytes);

    eprintln!("\n=== BOTH KERNELS PASSED ===");
}

fn test_qkv(gpu: &mut Gpu, n_list: &[usize], k: usize, row_bytes: usize) {
    let q_m: usize = 2048;
    let k_m: usize = 512;
    let v_m: usize = 512;

    eprintln!("=== gemm_qkv_hfq4g256 ===");
    eprintln!("q_m={q_m} k_m={k_m} v_m={v_m} K={k}");
    let groups_per_row = k / 256;
    let _ = row_bytes;

    let w_q = gpu.upload_raw(&synth(q_m, groups_per_row, 0xAA), &[q_m * row_bytes]).unwrap();
    let w_k = gpu.upload_raw(&synth(k_m, groups_per_row, 0xBB), &[k_m * row_bytes]).unwrap();
    let w_v = gpu.upload_raw(&synth(v_m, groups_per_row, 0xCC), &[v_m * row_bytes]).unwrap();

    let max_n = *n_list.iter().max().unwrap();
    let x_host: Vec<f32> = (0..max_n * k)
        .map(|i| ((i as i64).wrapping_mul(1103515245).wrapping_add(12345) & 0xFFFFFF) as f32 * 1e-7 - 0.5)
        .collect();

    let x_gemv = gpu.alloc_tensor(&[k], DType::F32).unwrap();
    let y_q_1 = gpu.alloc_tensor(&[q_m], DType::F32).unwrap();
    let y_k_1 = gpu.alloc_tensor(&[k_m], DType::F32).unwrap();
    let y_v_1 = gpu.alloc_tensor(&[v_m], DType::F32).unwrap();
    let y_q_col = gpu.alloc_tensor(&[max_n * q_m], DType::F32).unwrap();
    let y_k_col = gpu.alloc_tensor(&[max_n * k_m], DType::F32).unwrap();
    let y_v_col = gpu.alloc_tensor(&[max_n * v_m], DType::F32).unwrap();

    let x_gemm = gpu.alloc_tensor(&[max_n * k], DType::F32).unwrap();
    let y_q_gemm = gpu.alloc_tensor(&[max_n * q_m], DType::F32).unwrap();
    let y_k_gemm = gpu.alloc_tensor(&[max_n * k_m], DType::F32).unwrap();
    let y_v_gemm = gpu.alloc_tensor(&[max_n * v_m], DType::F32).unwrap();

    gpu.hip.memcpy_htod(&x_gemm.buf, bytes_of(&x_host)).unwrap();

    for &n in n_list {
        let mut gemv_us: f64 = 0.0;
        for i in 0..n {
            gpu.hip.memcpy_htod(&x_gemv.buf, bytes_of(&x_host[i * k..(i + 1) * k])).unwrap();
            gpu.hip.device_synchronize().unwrap();
            let t = Instant::now();
            gpu.fused_qkv_hfq4g256(
                &w_q, &w_k, &w_v,
                &x_gemv,
                &y_q_1, &y_k_1, &y_v_1,
                q_m, k_m, v_m, k,
            ).unwrap();
            gpu.hip.device_synchronize().unwrap();
            gemv_us += t.elapsed().as_secs_f64() * 1e6;

            gpu.hip.memcpy_dtod_at(&y_q_col.buf, i * q_m * 4, &y_q_1.buf, 0, q_m * 4).unwrap();
            gpu.hip.memcpy_dtod_at(&y_k_col.buf, i * k_m * 4, &y_k_1.buf, 0, k_m * 4).unwrap();
            gpu.hip.memcpy_dtod_at(&y_v_col.buf, i * v_m * 4, &y_v_1.buf, 0, v_m * 4).unwrap();
        }

        gpu.hip.device_synchronize().unwrap();
        let t = Instant::now();
        gpu.gemm_qkv_hfq4g256(
            &w_q, &w_k, &w_v,
            &x_gemm,
            &y_q_gemm, &y_k_gemm, &y_v_gemm,
            q_m, k_m, v_m, k, n,
        ).unwrap();
        gpu.hip.device_synchronize().unwrap();
        let gemm_us = t.elapsed().as_secs_f64() * 1e6;

        let ok_q = cmp_bit_exact(gpu, &y_q_col, &y_q_gemm, n * q_m, "q");
        let ok_k = cmp_bit_exact(gpu, &y_k_col, &y_k_gemm, n * k_m, "k");
        let ok_v = cmp_bit_exact(gpu, &y_v_col, &y_v_gemm, n * v_m, "v");
        let all_ok = ok_q && ok_k && ok_v;
        let status = if all_ok { "byte-exact OK" } else { "DIVERGENT" };
        eprintln!(
            "  N={n:3}  gemv×N: {:8.1} µs   gemm×1: {:8.1} µs   speedup: {:5.2}x   [{status}]",
            gemv_us, gemm_us, gemv_us / gemm_us
        );
        if !all_ok { std::process::exit(1); }
    }
}

fn test_gate_up(gpu: &mut Gpu, n_list: &[usize], k: usize, row_bytes: usize) {
    let gate_m: usize = 4096;
    let up_m:   usize = 4096;

    eprintln!("\n=== gemm_gate_up_hfq4g256 ===");
    eprintln!("gate_m={gate_m} up_m={up_m} K={k}");
    let groups_per_row = k / 256;
    let _ = row_bytes;

    let w_g = gpu.upload_raw(&synth(gate_m, groups_per_row, 0xDD), &[gate_m * row_bytes]).unwrap();
    let w_u = gpu.upload_raw(&synth(up_m,   groups_per_row, 0xEE), &[up_m   * row_bytes]).unwrap();

    let max_n = *n_list.iter().max().unwrap();
    let x_host: Vec<f32> = (0..max_n * k)
        .map(|i| ((i as i64).wrapping_mul(2246822507).wrapping_add(42) & 0xFFFFFF) as f32 * 1e-7 - 0.5)
        .collect();

    let x_gemv = gpu.alloc_tensor(&[k], DType::F32).unwrap();
    let y_g_1 = gpu.alloc_tensor(&[gate_m], DType::F32).unwrap();
    let y_u_1 = gpu.alloc_tensor(&[up_m], DType::F32).unwrap();
    let y_g_col = gpu.alloc_tensor(&[max_n * gate_m], DType::F32).unwrap();
    let y_u_col = gpu.alloc_tensor(&[max_n * up_m], DType::F32).unwrap();

    let x_gemm = gpu.alloc_tensor(&[max_n * k], DType::F32).unwrap();
    let y_g_gemm = gpu.alloc_tensor(&[max_n * gate_m], DType::F32).unwrap();
    let y_u_gemm = gpu.alloc_tensor(&[max_n * up_m], DType::F32).unwrap();

    gpu.hip.memcpy_htod(&x_gemm.buf, bytes_of(&x_host)).unwrap();

    for &n in n_list {
        let mut gemv_us: f64 = 0.0;
        for i in 0..n {
            gpu.hip.memcpy_htod(&x_gemv.buf, bytes_of(&x_host[i * k..(i + 1) * k])).unwrap();
            gpu.hip.device_synchronize().unwrap();
            let t = Instant::now();
            gpu.fused_gate_up_hfq4g256(
                &w_g, &w_u,
                &x_gemv,
                &y_g_1, &y_u_1,
                gate_m, up_m, k,
            ).unwrap();
            gpu.hip.device_synchronize().unwrap();
            gemv_us += t.elapsed().as_secs_f64() * 1e6;

            gpu.hip.memcpy_dtod_at(&y_g_col.buf, i * gate_m * 4, &y_g_1.buf, 0, gate_m * 4).unwrap();
            gpu.hip.memcpy_dtod_at(&y_u_col.buf, i * up_m   * 4, &y_u_1.buf, 0, up_m   * 4).unwrap();
        }

        gpu.hip.device_synchronize().unwrap();
        let t = Instant::now();
        gpu.gemm_gate_up_hfq4g256(
            &w_g, &w_u,
            &x_gemm,
            &y_g_gemm, &y_u_gemm,
            gate_m, up_m, k, n,
        ).unwrap();
        gpu.hip.device_synchronize().unwrap();
        let gemm_us = t.elapsed().as_secs_f64() * 1e6;

        let ok_g = cmp_bit_exact(gpu, &y_g_col, &y_g_gemm, n * gate_m, "gate");
        let ok_u = cmp_bit_exact(gpu, &y_u_col, &y_u_gemm, n * up_m,   "up");
        let all_ok = ok_g && ok_u;
        let status = if all_ok { "byte-exact OK" } else { "DIVERGENT" };
        eprintln!(
            "  N={n:3}  gemv×N: {:8.1} µs   gemm×1: {:8.1} µs   speedup: {:5.2}x   [{status}]",
            gemv_us, gemm_us, gemv_us / gemm_us
        );
        if !all_ok { std::process::exit(1); }
    }
}

fn cmp_bit_exact(gpu: &mut Gpu, a: &GpuTensor, b: &GpuTensor, n: usize, label: &str) -> bool {
    let av = gpu.download_f32(a).unwrap()[..n].to_vec();
    let bv = gpu.download_f32(b).unwrap()[..n].to_vec();
    for i in 0..n {
        if av[i].to_bits() != bv[i].to_bits() {
            eprintln!(
                "  {label}: DIVERGENT at i={i}  gemv={:.6e} ({:#010x})  gemm={:.6e} ({:#010x})",
                av[i], av[i].to_bits(), bv[i], bv[i].to_bits()
            );
            let count: usize = av.iter().zip(bv.iter()).filter(|(a, b)| a.to_bits() != b.to_bits()).count();
            eprintln!("  {label}: {count}/{n} elements diverged");
            return false;
        }
    }
    true
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
