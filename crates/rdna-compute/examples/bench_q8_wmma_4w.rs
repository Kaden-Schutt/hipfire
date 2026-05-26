// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Nick Woolmer
// hipfire — see LICENSE and NOTICE in the project root.

//! Correctness + perf A/B for the new 4-warp Q8 WMMA kernel
//! (`gemm_q8_0_wmma_4w.hip`) vs the production single-warp kernel
//! (`gemm_q8_0_wmma.hip`).
//!
//! Shapes target the DeepSeek V4 hot path: M ≈ 4096 (wq_a), 32768 (wq_b),
//! K ≈ 4096, batch sizes 16..1024.

use rdna_compute::{DType, Gpu, GpuTensor};
use std::time::Instant;

fn f32_to_f16_bits(v: f32) -> u16 {
    let bits = v.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = (((bits >> 23) & 0xff) as i32) - 127 + 15;
    let mant = (bits & 0x7fffff) as u32;
    if exp <= 0 {
        return sign;
    }
    if exp >= 31 {
        return sign | 0x7c00;
    }
    sign | ((exp as u16) << 10) | ((mant >> 13) as u16)
}

const WARMUP: usize = 4;
const TRIALS: usize = 30;

fn wrap_buf(raw_ptr: *mut std::ffi::c_void, bytes: usize, shape: Vec<usize>, dtype: DType) -> GpuTensor {
    GpuTensor {
        buf: unsafe { hip_bridge::DeviceBuffer::from_raw(raw_ptr, bytes) },
        shape,
        dtype,
    }
}

fn quantize_q8_block(weights_f32: &[f32]) -> Vec<u8> {
    // Q8_0 format: per 32-element block, [fp16 scale | 32 int8 weights].
    assert_eq!(weights_f32.len() % 32, 0);
    let mut out = Vec::with_capacity(weights_f32.len() / 32 * 34);
    for block in weights_f32.chunks_exact(32) {
        let absmax = block.iter().fold(0f32, |a, &b| a.max(b.abs()));
        let scale = if absmax > 0.0 { absmax / 127.0 } else { 1.0 };
        // fp16 cast
        let scale_bits = f32_to_f16_bits(scale);
        out.extend_from_slice(&scale_bits.to_le_bytes());
        for &w in block {
            let q = (w / scale).round().clamp(-127.0, 127.0) as i8;
            out.push(q as u8);
        }
    }
    out
}

fn f32_to_f16_bytes(f: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(f.len() * 2);
    for &v in f {
        let h = f32_to_f16_bits(v);
        out.extend_from_slice(&h.to_le_bytes());
    }
    out
}

fn cpu_gemm(a: &[f32], x: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    let mut y = vec![0f32; n * m];
    for ni in 0..n {
        for mi in 0..m {
            let mut s = 0.0f32;
            for ki in 0..k {
                s += a[mi * k + ki] * x[ni * k + ki];
            }
            y[ni * m + mi] = s;
        }
    }
    y
}

fn main() {
    let mut gpu = Gpu::init().expect("gpu init");
    println!("Arch: {}", gpu.arch);

    let shapes: &[(usize, usize, usize, &str)] = &[
        (256, 4096, 64, "small (M=256, K=4096, B=64)"),
        (1024, 4096, 64, "med (M=1024, K=4096, B=64)"),
        (4096, 4096, 64, "large (M=4096, K=4096, B=64)"),
        (4096, 4096, 256, "large B=256"),
        (4096, 4096, 1024, "large B=1024"),
        (32768, 1536, 1024, "wq_b shape"),
    ];

    for &(m, k, n, label) in shapes {
        println!("\n=== {label} ===");
        if m % 64 != 0 || n % 64 != 0 {
            println!("  SKIP — 4-warp kernel requires M%64==0 and N%64==0");
            continue;
        }
        // Synthesize weights (small range to avoid huge int8 saturation losses).
        let weights_f32: Vec<f32> = (0..m*k).map(|i| ((i % 17) as f32 - 8.0) * 0.01).collect();
        let x_f32: Vec<f32> = (0..n*k).map(|i| ((i % 13) as f32 - 6.0) * 0.01).collect();
        let weights_q8 = quantize_q8_block(&weights_f32);
        let x_f16 = f32_to_f16_bytes(&x_f32);

        let a_gpu = gpu.hip.malloc(weights_q8.len()).expect("malloc A");
        let x_gpu = gpu.hip.malloc(x_f16.len()).expect("malloc X");
        let y_gpu = gpu.hip.malloc(n * m * 4).expect("malloc Y");
        let y2_gpu = gpu.hip.malloc(n * m * 4).expect("malloc Y2");
        gpu.hip.memcpy_htod(&a_gpu, &weights_q8).expect("htod A");
        gpu.hip.memcpy_htod(&x_gpu, &x_f16).expect("htod X");

        let a_tensor = wrap_buf(a_gpu.as_ptr(), weights_q8.len(), vec![m, k], DType::Q8_0);
        let x_tensor = wrap_buf(x_gpu.as_ptr(), x_f16.len(), vec![n, k], DType::F16);
        let y_tensor = wrap_buf(y_gpu.as_ptr(), n * m * 4, vec![n, m], DType::F32);
        let y2_tensor = wrap_buf(y2_gpu.as_ptr(), n * m * 4, vec![n, m], DType::F32);

        // ── Run reference kernel
        gpu.gemm_q8_0_wmma(&a_tensor, &x_tensor, &y_tensor, m, k, n).expect("ref");
        gpu.hip.device_synchronize().unwrap();
        let mut y_ref_bytes = vec![0u8; n * m * 4];
        gpu.hip.memcpy_dtoh(&mut y_ref_bytes, &y_gpu).unwrap();
        let y_ref: &[f32] = unsafe {
            std::slice::from_raw_parts(y_ref_bytes.as_ptr() as *const f32, n * m)
        };

        // ── Run new 4w kernel
        // Need to call via launch_kernel since no wrapper yet.
        gpu.gemm_q8_0_wmma_4w(&a_tensor, &x_tensor, &y2_tensor, m, k, n).expect("4w");
        gpu.hip.device_synchronize().unwrap();
        let mut y_new_bytes = vec![0u8; n * m * 4];
        gpu.hip.memcpy_dtoh(&mut y_new_bytes, &y2_gpu).unwrap();
        let y_new: &[f32] = unsafe {
            std::slice::from_raw_parts(y_new_bytes.as_ptr() as *const f32, n * m)
        };

        // Compare
        let mut max_diff = 0.0f32;
        let mut diff_idx = 0;
        let mut nan_count = 0;
        for i in 0..n*m {
            if !y_new[i].is_finite() { nan_count += 1; continue; }
            let d = (y_ref[i] - y_new[i]).abs();
            if d > max_diff { max_diff = d; diff_idx = i; }
        }
        let rel_max = max_diff / y_ref[diff_idx].abs().max(1e-6);
        println!("  max_abs_diff: {max_diff:.6e}  rel: {rel_max:.6e}  nan: {nan_count}");
        if nan_count > 0 || max_diff > 0.1 {
            println!("  ✗ CORRECTNESS FAIL — skipping perf");
            continue;
        }

        // ── Perf A/B
        for _ in 0..WARMUP {
            gpu.gemm_q8_0_wmma(&a_tensor, &x_tensor, &y_tensor, m, k, n).unwrap();
        }
        gpu.hip.device_synchronize().unwrap();
        let t0 = Instant::now();
        for _ in 0..TRIALS {
            gpu.gemm_q8_0_wmma(&a_tensor, &x_tensor, &y_tensor, m, k, n).unwrap();
        }
        gpu.hip.device_synchronize().unwrap();
        let ref_us = t0.elapsed().as_secs_f64() / TRIALS as f64 * 1e6;

        for _ in 0..WARMUP {
            gpu.gemm_q8_0_wmma_4w(&a_tensor, &x_tensor, &y2_tensor, m, k, n).unwrap();
        }
        gpu.hip.device_synchronize().unwrap();
        let t0 = Instant::now();
        for _ in 0..TRIALS {
            gpu.gemm_q8_0_wmma_4w(&a_tensor, &x_tensor, &y2_tensor, m, k, n).unwrap();
        }
        gpu.hip.device_synchronize().unwrap();
        let new_us = t0.elapsed().as_secs_f64() / TRIALS as f64 * 1e6;

        let speedup = ref_us / new_us;
        let flops = 2.0 * m as f64 * k as f64 * n as f64;
        let ref_gflops = flops / ref_us / 1e3;
        let new_gflops = flops / new_us / 1e3;
        println!(
            "  ref (1w16x16): {ref_us:7.1} µs ({ref_gflops:6.1} GFLOPS)   \
             4w (64x64):    {new_us:7.1} µs ({new_gflops:6.1} GFLOPS)   speedup: {speedup:.2}×"
        );

        // forget views to avoid double-free
        std::mem::forget(a_tensor);
        std::mem::forget(x_tensor);
        std::mem::forget(y_tensor);
        std::mem::forget(y2_tensor);
    }
}
