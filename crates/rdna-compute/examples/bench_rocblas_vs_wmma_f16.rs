// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Nick Woolmer
// hipfire — see LICENSE and NOTICE in the project root.

//! A/B perf microbench: rocBLAS gemm_ex (F16×F16→F32) vs hand-rolled
//! `gemm_f16_x_f16_wmma` at DeepSeek V4 compressor shapes.
//!
//! Phase B2 of the deepseek4 prefill catch-up plan — quantify whether
//! rocBLAS is faster than the hand-rolled WMMA kernel for the compressor
//! GEMM shapes (M ≈ 256–1024, K ≈ 7168, batch varied).
//!
//! Usage:
//!   cargo run --release -p rdna-compute --example bench_rocblas_vs_wmma_f16

use rdna_compute::{DType, Gpu, GpuTensor};
use std::time::Instant;

const WARMUP: usize = 8;
const TRIALS: usize = 60;

fn wrap_buf(raw_ptr: *mut std::ffi::c_void, bytes: usize, shape: Vec<usize>, dtype: DType) -> GpuTensor {
    GpuTensor {
        buf: unsafe { hip_bridge::DeviceBuffer::from_raw(raw_ptr, bytes) },
        shape,
        dtype,
    }
}

fn main() {
    let mut gpu = Gpu::init().expect("gpu init");
    println!("Arch: {}", gpu.arch);

    // Force-load rocBLAS even if arch isn't CDNA3 (gfx1151).
    if gpu.rocblas.is_none() {
        match hip_bridge::Rocblas::load() {
            Ok(rb) => {
                println!("[bench] forced rocBLAS load on {}", gpu.arch);
                gpu.rocblas = Some(rb);
            }
            Err(e) => {
                println!("[bench] rocBLAS unavailable: {e}");
                std::process::exit(1);
            }
        }
    }

    let shapes: &[(usize, usize, &str)] = &[
        (256, 7168, "comp_idx M=256 K=7168"),
        (1024, 7168, "comp_main_r4 M=1024 K=7168"),
    ];
    let batches: &[usize] = &[16, 64, 128, 256, 512];

    for &(m, k, label) in shapes {
        println!("\n=== {label} ===");
        let w_gpu_buf = gpu.hip.malloc(m * k * 2).expect("malloc W");
        let mut w_bytes = vec![0u8; m * k * 2];
        for (i, b) in w_bytes.iter_mut().enumerate() {
            *b = ((i * 31) & 0xff) as u8;
        }
        gpu.hip.memcpy_htod(&w_gpu_buf, &w_bytes).expect("copy W");
        let w_ptr = w_gpu_buf.as_ptr();

        for &b in batches {
            let x_gpu_buf = gpu.hip.malloc(b * k * 2).expect("malloc X");
            let y_gpu_buf = gpu.hip.malloc(b * m * 4).expect("malloc Y");
            let mut x_bytes = vec![0u8; b * k * 2];
            for (i, bb) in x_bytes.iter_mut().enumerate() {
                *bb = ((i * 17) & 0xff) as u8;
            }
            gpu.hip.memcpy_htod(&x_gpu_buf, &x_bytes).expect("copy X");
            let x_ptr = x_gpu_buf.as_ptr();
            let y_ptr = y_gpu_buf.as_ptr();

            // ── rocBLAS path (uses DeviceBuffer directly via rocblas_gemm_hfq4_prefill)
            for _ in 0..WARMUP {
                gpu.rocblas_gemm_hfq4_prefill(&w_gpu_buf, &x_gpu_buf, &y_gpu_buf, m, b, k).unwrap();
            }
            gpu.hip.device_synchronize().unwrap();
            let t0 = Instant::now();
            for _ in 0..TRIALS {
                gpu.rocblas_gemm_hfq4_prefill(&w_gpu_buf, &x_gpu_buf, &y_gpu_buf, m, b, k).unwrap();
            }
            gpu.hip.device_synchronize().unwrap();
            let rocblas_us = t0.elapsed().as_secs_f64() / TRIALS as f64 * 1e6;

            // ── Hand-rolled WMMA path
            // Wrap raw pointers as GpuTensor views — non-owning, do NOT free.
            let w_tensor = wrap_buf(w_ptr, m * k * 2, vec![m, k], DType::F16);
            let x_tensor = wrap_buf(x_ptr, b * k * 2, vec![b, k], DType::F16);
            let y_tensor = wrap_buf(y_ptr, b * m * 4, vec![b, m], DType::F32);
            for _ in 0..WARMUP {
                gpu.gemm_f16_x_f16_wmma(&w_tensor, &x_tensor, &y_tensor, m, k, b).unwrap();
            }
            gpu.hip.device_synchronize().unwrap();
            let t0 = Instant::now();
            for _ in 0..TRIALS {
                gpu.gemm_f16_x_f16_wmma(&w_tensor, &x_tensor, &y_tensor, m, k, b).unwrap();
            }
            gpu.hip.device_synchronize().unwrap();
            let wmma_us = t0.elapsed().as_secs_f64() / TRIALS as f64 * 1e6;

            // forget the views so Drop doesn't double-free.
            std::mem::forget(w_tensor);
            std::mem::forget(x_tensor);
            std::mem::forget(y_tensor);

            let flops = 2.0 * m as f64 * k as f64 * b as f64;
            let rocblas_gflops = flops / rocblas_us / 1e3;
            let wmma_gflops = flops / wmma_us / 1e3;
            let speedup = wmma_us / rocblas_us;
            let winner = if speedup > 1.0 { "rocBLAS" } else { "WMMA" };

            println!(
                "  B={b:4}  rocBLAS: {rocblas_us:7.1} µs ({rocblas_gflops:6.1} GFLOPS)  \
                 WMMA: {wmma_us:7.1} µs ({wmma_gflops:6.1} GFLOPS)  \
                 winner: {winner} ({:.2}×)",
                if speedup > 1.0 { speedup } else { 1.0 / speedup }
            );
        }
    }
}
