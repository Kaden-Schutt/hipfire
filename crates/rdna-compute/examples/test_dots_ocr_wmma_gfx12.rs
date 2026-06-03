// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire - see LICENSE and NOTICE in the project root.

//! Focused channel test for the dots.ocr vision WMMA fast paths.
//!
//! This covers the two production kernels dots.ocr needs before gfx12 can
//! safely leave the scalar vision fallback:
//! - `gemm_f16_wmma_mb8`: fused-transpose F16 vision linear
//! - `attention_dflash_wmma_m64_n32_f16kv_v5_f32`: large vision attention

use rdna_compute::{DType, Gpu};

fn lcg_data(seed: u32, n: usize) -> Vec<f32> {
    let mut s = seed;
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(1_103_515_245).wrapping_add(12_345);
            let u = (s >> 16) & 0x7fff;
            (u as f32 / 32_768.0 - 0.5) * 0.2
        })
        .collect()
}

fn max_abs(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

fn assert_close(name: &str, got: &[f32], expected: &[f32], tol: f32) {
    let diff = max_abs(got, expected);
    println!("{name}: max_abs_diff={diff:.3e} tol={tol:.3e}");
    assert!(
        diff < tol,
        "{name}: max_abs_diff={diff:.3e} exceeds tol={tol:.3e}"
    );
}

fn test_gemm_f16_wmma_mb8(gpu: &mut Gpu) {
    let m = 37usize;
    let k = 64usize;
    let n = 130usize;

    let w_f32 = lcg_data(0x1111_2222, m * k);
    let x_f32 = lcg_data(0x3333_4444, n * k);

    let w_src = gpu.upload_f32(&w_f32, &[m * k]).unwrap();
    let w_f16 = gpu.alloc_tensor(&[m * k], DType::F16).unwrap();
    gpu.cast_f32_to_f16(&w_src, &w_f16).unwrap();
    let x = gpu.upload_f32(&x_f32, &[n * k]).unwrap();

    let y_ref_t = gpu.zeros(&[m * n], DType::F32).unwrap();
    let y_ref = gpu.zeros(&[n, m], DType::F32).unwrap();
    gpu.gemm_f16(&w_f16, &x, &y_ref_t, m, k, n).unwrap();
    gpu.transpose_f32(&y_ref_t, &y_ref, m, n).unwrap();

    let y_wmma = gpu.zeros(&[n, m], DType::F32).unwrap();
    gpu.gemm_f16_wmma_mb8(&w_f16, &x, &y_wmma, m, k, n).unwrap();
    gpu.hip.device_synchronize().unwrap();

    let ref_host = gpu.download_f32(&y_ref).unwrap();
    let wmma_host = gpu.download_f32(&y_wmma).unwrap();
    assert_close("gemm_f16_wmma_mb8", &wmma_host, &ref_host, 5.0e-3);
}

fn test_attention_v5(gpu: &mut Gpu) {
    let b = 65usize;
    let l = 96usize;
    let n_heads = 2usize;
    let n_kv_heads = 2usize;
    let hd = 128usize;

    let q = lcg_data(0xa5a5_a5a5, b * n_heads * hd);
    let k = lcg_data(0xc3c3_c3c3, l * n_kv_heads * hd);
    let v = lcg_data(0x9696_9696, l * n_kv_heads * hd);

    let d_q = gpu.upload_f32(&q, &[b * n_heads * hd]).unwrap();
    let d_k = gpu.upload_f32(&k, &[l * n_kv_heads * hd]).unwrap();
    let d_v = gpu.upload_f32(&v, &[l * n_kv_heads * hd]).unwrap();

    let out_scalar = gpu.zeros(&[b * n_heads * hd], DType::F32).unwrap();
    gpu.attention_dflash_f32(&d_q, &d_k, &d_v, &out_scalar, b, l, n_heads, n_kv_heads, hd)
        .unwrap();

    let d_k_f16 = gpu.alloc_tensor(&[l * n_kv_heads * hd], DType::F16).unwrap();
    let d_v_f16 = gpu.alloc_tensor(&[l * n_kv_heads * hd], DType::F16).unwrap();
    gpu.cast_f32_to_f16(&d_k, &d_k_f16).unwrap();
    gpu.cast_f32_to_f16(&d_v, &d_v_f16).unwrap();

    let out_wmma = gpu.zeros(&[b * n_heads * hd], DType::F32).unwrap();
    gpu.attention_dflash_wmma_m64_n32_f16kv_v5_f32(
        &d_q,
        &d_k_f16,
        &d_v_f16,
        &out_wmma,
        b,
        l,
        n_heads,
        n_kv_heads,
        hd,
    )
    .unwrap();
    gpu.hip.device_synchronize().unwrap();

    let scalar_host = gpu.download_f32(&out_scalar).unwrap();
    let wmma_host = gpu.download_f32(&out_wmma).unwrap();
    assert_close("attention_dflash_wmma_m64_n32_f16kv_v5_f32", &wmma_host, &scalar_host, 5.0e-3);
}

fn main() {
    let mut gpu = Gpu::init().expect("GPU init failed");
    println!("GPU initialized: {}", gpu.arch);

    if !(gpu.arch_caps.has_wmma_w32() || gpu.arch_caps.has_wmma_w32_gfx12()) {
        println!("SKIP dots.ocr WMMA channel test: {} lacks wave32 WMMA", gpu.arch);
        return;
    }

    test_gemm_f16_wmma_mb8(&mut gpu);
    test_attention_v5(&mut gpu);
    println!("PASS dots.ocr WMMA channel test");
}
