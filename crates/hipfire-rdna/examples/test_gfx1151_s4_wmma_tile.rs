// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Channel test for the gfx1151 S4xS4 IU4-WMMA tile probe.
//!
//! Run:
//!   cargo run --release -p hipfire-rdna --example test_gfx1151_s4_wmma_tile

use hipfire_rdna::{DType, Gpu, GpuTensor};

fn lcg(state: &mut u32) -> u32 {
    *state = state.wrapping_mul(1664525).wrapping_add(1013904223);
    *state
}

fn build_s4_matrix(rows: usize, cols: usize, seed: u32) -> (Vec<u8>, Vec<i8>) {
    assert_eq!(cols % 2, 0);
    let mut s = seed;
    let mut vals = Vec::with_capacity(rows * cols);
    let mut packed = vec![0u8; rows * cols / 2];
    for row in 0..rows {
        for col_pair in 0..(cols / 2) {
            let lo = ((lcg(&mut s) & 0x0f) as i8) - 8;
            let hi = ((lcg(&mut s) & 0x0f) as i8) - 8;
            vals.push(lo);
            vals.push(hi);
            packed[row * (cols / 2) + col_pair] = ((lo as u8) & 0x0f) | (((hi as u8) & 0x0f) << 4);
        }
    }
    (packed, vals)
}

fn cpu_s4_dot(a: &[i8], x: &[i8], m: usize, k: usize, n: usize) -> Vec<i32> {
    let mut y = vec![0i32; n * m];
    for col in 0..n {
        for row in 0..m {
            let mut acc = 0i32;
            for kk in 0..k {
                acc += a[row * k + kk] as i32 * x[col * k + kk] as i32;
            }
            y[col * m + row] = acc;
        }
    }
    y
}

fn upload_raw(gpu: &mut Gpu, data: &[u8]) -> GpuTensor {
    let t = gpu
        .alloc_tensor(&[data.len()], DType::Raw)
        .expect("alloc raw");
    gpu.hip.memcpy_htod(&t.buf, data).expect("upload raw");
    t
}

fn alloc_i32_zeros(gpu: &mut Gpu, n: usize) -> GpuTensor {
    let t = gpu
        .alloc_tensor(&[n * 4], DType::Raw)
        .expect("alloc i32 output");
    gpu.hip.memset(&t.buf, 0, n * 4).expect("zero output");
    t
}

fn download_i32(gpu: &Gpu, tensor: &GpuTensor, n: usize) -> Vec<i32> {
    let mut out = vec![0i32; n];
    let bytes = unsafe { std::slice::from_raw_parts_mut(out.as_mut_ptr() as *mut u8, n * 4) };
    gpu.hip
        .memcpy_dtoh(bytes, &tensor.buf)
        .expect("download i32");
    out
}

fn run_case(gpu: &mut Gpu, label: &str, m: usize, k: usize, n: usize) {
    assert_eq!(k % 16, 0);
    assert_eq!(k % 2, 0);

    let (a_packed, a_vals) = build_s4_matrix(m, k, 0x1234_0000 ^ m as u32);
    let (x_packed, x_vals) = build_s4_matrix(n, k, 0x5678_0000 ^ n as u32);
    let expected = cpu_s4_dot(&a_vals, &x_vals, m, k, n);

    let a_gpu = upload_raw(gpu, &a_packed);
    let x_gpu = upload_raw(gpu, &x_packed);
    let y_gpu = alloc_i32_zeros(gpu, m * n);
    gpu.gemm_s4s4_wmma_tile_gfx1151(&a_gpu, &x_gpu, &y_gpu, m, k, n)
        .expect("launch u4 wmma tile");
    gpu.hip.device_synchronize().expect("sync");
    let got = download_i32(gpu, &y_gpu, m * n);

    let mut bad = 0usize;
    let mut first = None;
    let mut worst = 0i32;
    for (idx, (&g, &e)) in got.iter().zip(expected.iter()).enumerate() {
        let err = (g - e).abs();
        worst = worst.max(err);
        if g != e {
            bad += 1;
            if first.is_none() {
                first = Some((idx, g, e));
            }
        }
    }
    if let Some((idx, g, e)) = first {
        panic!("{label}: mismatch at {idx}: got {g}, expected {e}; bad={bad} worst_abs={worst}");
    }
    println!("PASS {label}: M={m} K={k} N={n}");
}

fn main() {
    let mut gpu = Gpu::init().expect("gpu init");
    if gpu.arch != "gfx1151" {
        println!("SKIP: arch {} is not gfx1151", gpu.arch);
        return;
    }

    run_case(&mut gpu, "aligned", 32, 64, 32);
    run_case(&mut gpu, "edge", 24, 48, 20);
}
