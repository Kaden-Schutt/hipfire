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

//! Synthetic gfx1151 HFQ4-G256 S4/IU4 MMQ benchmark.
//!
//! Compares the existing Q8_1 + IU8 MMQ control path against the signed-Q4
//! activation + IU4 probe path on the same HFQ4 weights and S4-derived
//! activation matrix. This is a routing blocker, not a production route.
//!
//! Run:
//!   cargo run --release -p hipfire-rdna --example bench_gfx1151_hfq4_s4_mmq

use std::time::Instant;

use hipfire_rdna::{DType, Gpu, GpuTensor};

fn lcg(state: &mut u32) -> u32 {
    *state = state.wrapping_mul(1664525).wrapping_add(1013904223);
    *state
}

fn push_f32_le(out: &mut [u8], offset: usize, value: f32) {
    out[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
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

fn build_s4_scratch(n: usize, k: usize, seed: u32) -> (Vec<u8>, Vec<f32>, Vec<i32>, Vec<f32>) {
    assert_eq!(k % 32, 0);
    let mut s = seed;
    let mut packed = vec![0u8; n * k / 2];
    let mut scales = vec![0f32; n * (k / 32)];
    let mut sums = vec![0i32; n * (k / 32)];
    let mut x_f32 = vec![0f32; n * k];
    for col in 0..n {
        for sb in 0..(k / 32) {
            let scale = 0.02 + ((lcg(&mut s) & 0xff) as f32) * 0.00004;
            scales[col * (k / 32) + sb] = scale;
            let mut sum = 0i32;
            for i in 0..32 {
                let kk = sb * 32 + i;
                let q = ((lcg(&mut s) & 0x0f) as i8) - 8;
                sum += q as i32;
                x_f32[col * k + kk] = scale * q as f32;
            }
            sums[col * (k / 32) + sb] = sum;
        }
        for pair in 0..(k / 2) {
            let lo = (x_f32[col * k + 2 * pair] / scales[col * (k / 32) + pair / 16]).round() as i8;
            let hi =
                (x_f32[col * k + 2 * pair + 1] / scales[col * (k / 32) + pair / 16]).round() as i8;
            packed[col * (k / 2) + pair] = ((lo as u8) & 0x0f) | (((hi as u8) & 0x0f) << 4);
        }
    }
    (packed, scales, sums, x_f32)
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

fn median_ms(mut values: Vec<f64>) -> f64 {
    values.sort_by(|a, b| a.partial_cmp(b).unwrap());
    values[values.len() / 2]
}

fn time_ms<F: FnMut(&mut Gpu)>(gpu: &mut Gpu, trials: usize, mut f: F) -> Vec<f64> {
    let mut out = Vec::with_capacity(trials);
    for _ in 0..trials {
        let start = Instant::now();
        f(gpu);
        gpu.hip.device_synchronize().expect("sync");
        out.push(start.elapsed().as_secs_f64() * 1000.0);
    }
    out
}

fn main() {
    let mut gpu = Gpu::init().expect("gpu init");
    if gpu.arch != "gfx1151" {
        println!("SKIP: arch {} is not gfx1151", gpu.arch);
        return;
    }

    let m = std::env::var("M")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(4096);
    let k = std::env::var("K")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(2048);
    let n = std::env::var("N")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(128);
    let trials = std::env::var("TRIALS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(9);
    assert_eq!(m % 16, 0);
    assert_eq!(k % 256, 0);
    assert_eq!(n % 16, 0);

    let weights = build_hfq4(m, k, 0x1020_3000);
    let (x_qs, x_d, x_sum, x_f32) = build_s4_scratch(n, k, 0x4050_6000);

    let a_gpu = upload_raw(&mut gpu, &weights);
    let x_qs_gpu = upload_raw(&mut gpu, &x_qs);
    let x_d_gpu = upload_f32(&mut gpu, &x_d);
    let x_sum_gpu = upload_i32(&mut gpu, &x_sum);
    let x_f32_gpu = upload_f32(&mut gpu, &x_f32);
    let y_s4 = alloc_f32_zeros(&mut gpu, m * n);
    let y_q8 = alloc_f32_zeros(&mut gpu, m * n);

    gpu.gemm_hfq4g256_s4_mmq_gfx1151(&a_gpu, &x_qs_gpu, &x_d_gpu, &x_sum_gpu, &y_s4, m, k, n)
        .expect("warm s4");
    gpu.gemm_hfq4g256_mmq_set(&a_gpu, &x_f32_gpu, &y_q8, m, k, n)
        .expect("warm q8");
    gpu.hip.device_synchronize().expect("warm sync");

    let s4_times = time_ms(&mut gpu, trials, |gpu| {
        gpu.gemm_hfq4g256_s4_mmq_gfx1151(&a_gpu, &x_qs_gpu, &x_d_gpu, &x_sum_gpu, &y_s4, m, k, n)
            .expect("s4 launch");
    });
    let q8_times = time_ms(&mut gpu, trials, |gpu| {
        gpu.gemm_hfq4g256_mmq_set(&a_gpu, &x_f32_gpu, &y_q8, m, k, n)
            .expect("q8 launch");
    });

    let got_s4 = download_f32(&gpu, &y_s4, m * n);
    let got_q8 = download_f32(&gpu, &y_q8, m * n);
    let mut max_abs = 0.0f32;
    let mut rms = 0.0f64;
    let mut ref_rms = 0.0f64;
    for (&s4, &q8) in got_s4.iter().zip(got_q8.iter()) {
        let d = (s4 - q8) as f64;
        max_abs = max_abs.max(d.abs() as f32);
        rms += d * d;
        ref_rms += (q8 as f64) * (q8 as f64);
    }
    rms = (rms / got_s4.len() as f64).sqrt();
    ref_rms = (ref_rms / got_s4.len() as f64).sqrt();

    let s4_ms = median_ms(s4_times);
    let q8_ms = median_ms(q8_times);
    println!("shape M={m} K={k} N={n} trials={trials}");
    println!("q8_1_iu8_control_median_ms={q8_ms:.4}");
    println!("s4_iu4_probe_median_ms={s4_ms:.4}");
    println!("s4_vs_q8_speedup={:.3}x", q8_ms / s4_ms);
    println!("s4_vs_q8_max_abs={max_abs:.6e}");
    println!("s4_vs_q8_rms={rms:.6e}");
    println!("s4_vs_q8_rel_rms={:.6e}", rms / ref_rms.max(1.0e-12));
}
