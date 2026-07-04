#![allow(
    clippy::duplicated_attributes,
    clippy::doc_lazy_continuation,
    clippy::doc_overindented_list_items,
    clippy::explicit_counter_loop,
    clippy::field_reassign_with_default,
    clippy::manual_checked_ops,
    clippy::manual_clamp,
    clippy::manual_div_ceil,
    clippy::needless_range_loop,
    clippy::ptr_arg,
    clippy::same_item_push,
    clippy::too_many_arguments,
    clippy::type_complexity,
    clippy::unnecessary_cast,
    clippy::useless_vec,
    clippy::while_let_loop
)]
// hipfire example clippy sweep: examples are GPU probes/benches, not reusable APIs.

// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
#![allow(
    clippy::doc_lazy_continuation,
    clippy::doc_overindented_list_items,
    clippy::drop_non_drop,
    clippy::excessive_precision,
    clippy::identity_op,
    clippy::manual_div_ceil,
    clippy::manual_is_multiple_of,
    clippy::needless_range_loop,
    clippy::print_literal,
    clippy::redundant_closure,
    clippy::too_many_arguments,
    clippy::type_complexity,
    clippy::unusual_byte_groupings,
    clippy::useless_vec,
    clippy::unnecessary_cast
)]

//! Channel test for the gfx1151 HFQ4-G256 x S4-activation IU4-WMMA probe.
//!
//! Run:
//!   cargo run --release -p hipfire-rdna --example test_gfx1151_hfq4_s4_mmq

use hipfire_rdna::{DType, Gpu, GpuTensor};

fn lcg(state: &mut u32) -> u32 {
    *state = state.wrapping_mul(1664525).wrapping_add(1013904223);
    *state
}

fn push_f32_le(out: &mut [u8], offset: usize, value: f32) {
    out[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
}

fn read_f32_le(bytes: &[u8], offset: usize) -> f32 {
    f32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap())
}

fn build_hfq4(m: usize, k: usize, seed: u32) -> Vec<u8> {
    assert_eq!(k % 256, 0);
    let groups = k / 256;
    let mut s = seed;
    let mut out = vec![0u8; m * groups * 136];
    for row in 0..m {
        for g in 0..groups {
            let base = (row * groups + g) * 136;
            let scale = 0.015 + ((lcg(&mut s) & 0xff) as f32) * 0.00002;
            let zero = -0.13 + ((lcg(&mut s) & 0xff) as f32) * 0.0003;
            push_f32_le(&mut out, base, scale);
            push_f32_le(&mut out, base + 4, zero);
            for b in 0..128 {
                let lo = (lcg(&mut s) & 0x0f) as u8;
                let hi = (lcg(&mut s) & 0x0f) as u8;
                out[base + 8 + b] = lo | (hi << 4);
            }
        }
    }
    out
}

fn build_s4_scratch(n: usize, k: usize, seed: u32) -> (Vec<u8>, Vec<f32>, Vec<i32>, Vec<i8>) {
    assert_eq!(k % 32, 0);
    let mut s = seed;
    let mut q_vals = vec![0i8; n * k];
    let mut packed = vec![0u8; n * k / 2];
    let mut scales = vec![0f32; n * (k / 32)];
    let mut sums = vec![0i32; n * (k / 32)];
    for col in 0..n {
        for sb in 0..(k / 32) {
            let scale = 0.02 + ((lcg(&mut s) & 0xff) as f32) * 0.00004;
            scales[col * (k / 32) + sb] = scale;
            let mut sum = 0i32;
            for i in 0..32 {
                let kk = sb * 32 + i;
                let q = ((lcg(&mut s) & 0x0f) as i8) - 8;
                q_vals[col * k + kk] = q;
                sum += q as i32;
            }
            sums[col * (k / 32) + sb] = sum;
        }
        for pair in 0..(k / 2) {
            let lo = q_vals[col * k + 2 * pair] as u8 & 0x0f;
            let hi = q_vals[col * k + 2 * pair + 1] as u8 & 0x0f;
            packed[col * (k / 2) + pair] = lo | (hi << 4);
        }
    }
    (packed, scales, sums, q_vals)
}

fn hfq4_nibble(weights: &[u8], k: usize, row: usize, kk: usize) -> u8 {
    let groups = k / 256;
    let g = kk / 256;
    let local = kk & 255;
    let base = (row * groups + g) * 136;
    let byte = weights[base + 8 + local / 2];
    if local & 1 == 0 {
        byte & 0x0f
    } else {
        byte >> 4
    }
}

fn cpu_ref(
    weights: &[u8],
    x_scales: &[f32],
    x_sums: &[i32],
    x_q: &[i8],
    m: usize,
    k: usize,
    n: usize,
) -> Vec<f32> {
    let groups = k / 256;
    let mut y = vec![0f32; n * m];
    for col in 0..n {
        for row in 0..m {
            let mut acc = 0f32;
            for g in 0..groups {
                let base = (row * groups + g) * 136;
                let sc = read_f32_le(weights, base);
                let zp = read_f32_le(weights, base + 4);
                let bias = zp + 8.0 * sc;
                for sb in 0..8 {
                    let meta = g * 8 + sb;
                    let dx = x_scales[col * (k / 32) + meta];
                    let sum_qx = x_sums[col * (k / 32) + meta] as f32;
                    let mut dot = 0i32;
                    for i in 0..32 {
                        let kk = g * 256 + sb * 32 + i;
                        let qw_s = hfq4_nibble(weights, k, row, kk) as i32 - 8;
                        let qx = x_q[col * k + kk] as i32;
                        dot += qw_s * qx;
                    }
                    acc += sc * dx * dot as f32 + bias * dx * sum_qx;
                }
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

fn upload_f32(gpu: &mut Gpu, data: &[f32]) -> GpuTensor {
    let bytes = unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 4) };
    let t = gpu
        .alloc_tensor(&[data.len()], DType::F32)
        .expect("alloc f32");
    gpu.hip.memcpy_htod(&t.buf, bytes).expect("upload f32");
    t
}

fn upload_i32(gpu: &mut Gpu, data: &[i32]) -> GpuTensor {
    let bytes = unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 4) };
    upload_raw(gpu, bytes)
}

fn alloc_f32_zeros(gpu: &mut Gpu, n: usize) -> GpuTensor {
    let t = gpu.alloc_tensor(&[n], DType::F32).expect("alloc y");
    gpu.hip.memset(&t.buf, 0, n * 4).expect("zero y");
    t
}

fn download_f32(gpu: &Gpu, tensor: &GpuTensor, n: usize) -> Vec<f32> {
    let mut out = vec![0f32; n];
    let bytes = unsafe { std::slice::from_raw_parts_mut(out.as_mut_ptr() as *mut u8, n * 4) };
    gpu.hip
        .memcpy_dtoh(bytes, &tensor.buf)
        .expect("download f32");
    out
}

fn run_case(gpu: &mut Gpu, label: &str, m: usize, k: usize, n: usize) {
    assert_eq!(k % 256, 0);
    let weights = build_hfq4(m, k, 0x1020_3000 ^ m as u32);
    let (x_qs, x_d, x_sum, x_q_vals) = build_s4_scratch(n, k, 0x4050_6000 ^ n as u32);
    let expected = cpu_ref(&weights, &x_d, &x_sum, &x_q_vals, m, k, n);

    let a_gpu = upload_raw(gpu, &weights);
    let x_qs_gpu = upload_raw(gpu, &x_qs);
    let x_d_gpu = upload_f32(gpu, &x_d);
    let x_sum_gpu = upload_i32(gpu, &x_sum);
    let y_gpu = alloc_f32_zeros(gpu, m * n);
    gpu.gemm_hfq4g256_s4_mmq_gfx1151(&a_gpu, &x_qs_gpu, &x_d_gpu, &x_sum_gpu, &y_gpu, m, k, n)
        .expect("launch hfq4 s4 mmq");
    gpu.hip.device_synchronize().expect("sync");
    let got = download_f32(gpu, &y_gpu, m * n);

    let mut max_abs = 0.0f32;
    let mut first_bad = None;
    for (idx, (&g, &e)) in got.iter().zip(expected.iter()).enumerate() {
        let err = (g - e).abs();
        max_abs = max_abs.max(err);
        if err > 1.0e-4 && first_bad.is_none() {
            first_bad = Some((idx, g, e, err));
        }
    }
    if let Some((idx, g, e, err)) = first_bad {
        panic!("{label}: mismatch at {idx}: got {g}, expected {e}, abs={err}, max_abs={max_abs}");
    }
    println!("PASS {label}: M={m} K={k} N={n} max_abs={max_abs:.3e}");
}

fn main() {
    let mut gpu = Gpu::init().expect("gpu init");
    if gpu.arch != "gfx1151" {
        println!("SKIP: arch {} is not gfx1151", gpu.arch);
        return;
    }

    run_case(&mut gpu, "aligned", 32, 256, 32);
    run_case(&mut gpu, "edge", 24, 256, 20);
}
