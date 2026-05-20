// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Minimal sanity check for the rocBLAS FFI + HFQ4 dequant path.
//!
//! What it verifies:
//!   1. librocblas.so loads + `rocblas_create_handle` succeeds.
//!   2. A straight FP16 × FP16 → FP32 GEMM via `rocblas_gemm_hfq4_prefill`
//!      returns the expected numerical result on a known-value test case.
//!
//! Does NOT require an HFQ4 weight file — we synthesize a known FP16 weight
//! in device memory, skip the dequantize kernel entirely, and wrap the
//! pointer in a throwaway `DeviceBuffer` to call the helper directly.
//!
//! Usage:
//!   cargo run --release -p rdna-compute --example rocblas_sanity

use rdna_compute::Gpu;

fn main() {
    let mut gpu = Gpu::init().expect("gpu init");
    println!("Arch: {}", gpu.arch);

    // try_init_rocblas runs automatically in Gpu::init(), but only on CDNA3.
    // Force it here so non-CDNA3 boxes also exercise the FFI smoke path
    // (rocBLAS still loads on RDNA3 — the hip-bridge doesn't gate on arch).
    if gpu.rocblas.is_none() {
        match hip_bridge::Rocblas::load() {
            Ok(rb) => {
                println!("[sanity] forcing rocBLAS load for non-CDNA3 (arch={})", gpu.arch);
                gpu.rocblas = Some(rb);
            }
            Err(e) => {
                println!("[sanity] rocBLAS unavailable: {e}");
                println!("[sanity] FAIL — cannot exercise the MFMA path without the library");
                std::process::exit(1);
            }
        }
    }

    // Test case: 4 rows × 8 cols weight (M=4, K=8). Two rows of batch (N=2).
    //   W = [[1, 0, 0, 0, 0, 0, 0, 0],
    //        [0, 1, 0, 0, 0, 0, 0, 0],
    //        [0, 0, 1, 0, 0, 0, 0, 0],
    //        [0, 0, 0, 1, 0, 0, 0, 0]]
    //   X = [[1,2,3,4,5,6,7,8],
    //        [8,7,6,5,4,3,2,1]]
    //   Y = X @ W^T; Y[0] = [1,2,3,4]; Y[1] = [8,7,6,5]
    const M: usize = 4;
    const K: usize = 8;
    const N: usize = 2;

    let w_host: Vec<f32> = vec![
        1., 0., 0., 0., 0., 0., 0., 0.,
        0., 1., 0., 0., 0., 0., 0., 0.,
        0., 0., 1., 0., 0., 0., 0., 0.,
        0., 0., 0., 1., 0., 0., 0., 0.,
    ];
    let x_host: Vec<f32> = vec![
        1., 2., 3., 4., 5., 6., 7., 8.,
        8., 7., 6., 5., 4., 3., 2., 1.,
    ];

    // Convert to f16 on host (using `half` crate would be cleaner, but the
    // engine currently has no direct dep; cast via bits).
    fn f32_to_f16_bits(x: f32) -> u16 {
        let v = x as f32;
        let bits = v.to_bits();
        let sign = ((bits >> 16) & 0x8000) as u16;
        let exp = (((bits >> 23) & 0xff) as i32) - 127 + 15;
        let mant = (bits & 0x7fffff) as u32;
        if exp <= 0 {
            // subnormal / zero — acceptable for our 0s + small ints
            if v == 0.0 { return sign; }
            return sign;
        }
        if exp >= 31 { return sign | 0x7c00; } // inf
        sign | ((exp as u16) << 10) | ((mant >> 13) as u16)
    }

    let w_fp16_bits: Vec<u16> = w_host.iter().map(|&v| f32_to_f16_bits(v)).collect();
    let x_fp16_bits: Vec<u16> = x_host.iter().map(|&v| f32_to_f16_bits(v)).collect();
    let w_bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(w_fp16_bits.as_ptr() as *const u8, w_fp16_bits.len() * 2)
    };
    let x_bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(x_fp16_bits.as_ptr() as *const u8, x_fp16_bits.len() * 2)
    };

    let w_gpu = gpu.hip.malloc(M * K * 2).expect("malloc W");
    let x_gpu = gpu.hip.malloc(N * K * 2).expect("malloc X");
    let y_gpu = gpu.hip.malloc(N * M * 4).expect("malloc Y");
    gpu.hip.memcpy_htod(&w_gpu, w_bytes).expect("copy W");
    gpu.hip.memcpy_htod(&x_gpu, x_bytes).expect("copy X");

    println!("[sanity] running rocBLAS GEMM (M={M}, N={N}, K={K}, FP16×FP16→FP32)...");
    gpu.rocblas_gemm_hfq4_prefill(&w_gpu, &x_gpu, &y_gpu, M, N, K)
        .expect("rocblas gemm");

    // Download Y and check
    let mut y_bytes = vec![0u8; N * M * 4];
    gpu.hip.memcpy_dtoh(&mut y_bytes, &y_gpu).expect("copy Y back");
    let y_host: &[f32] = unsafe {
        std::slice::from_raw_parts(y_bytes.as_ptr() as *const f32, N * M)
    };

    let expected = [1.0f32, 2., 3., 4., 8., 7., 6., 5.];
    let mut max_err = 0.0f32;
    for (i, (&got, &want)) in y_host.iter().zip(expected.iter()).enumerate() {
        let err = (got - want).abs();
        if err > max_err { max_err = err; }
        println!("  Y[{i:2}] = {got:.4}  (expected {want:.4}, err {err:.4e})");
    }
    if max_err > 1e-2 {
        println!("[sanity] FAIL — max error {max_err} exceeds tolerance");
        std::process::exit(1);
    }
    println!("[sanity] PASS — rocBLAS GEMM returns expected values (max err {max_err:.4e})");
}
