// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Focused gfx12 channel test for the two DS4 batched-prefill WMMA sisters.
//! This is a kernel-channel test, not product or serving evidence.

use rdna_compute::{DType, Gpu, GpuTensor};
use std::time::Instant;

fn f32_to_f16_bits(v: f32) -> u16 {
    let bits = v.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = (((bits >> 23) & 0xff) as i32) - 127 + 15;
    let mant = bits & 0x7f_ffff;
    if exp <= 0 {
        return sign;
    }
    if exp >= 31 {
        return sign | 0x7c00;
    }
    sign | ((exp as u16) << 10) | ((mant >> 13) as u16)
}

fn as_bytes_f32(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

fn as_bytes_i32(values: &[i32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

fn wrap(raw: *mut std::ffi::c_void, bytes: usize, shape: Vec<usize>, dtype: DType) -> GpuTensor {
    GpuTensor {
        buf: unsafe { hip_bridge::DeviceBuffer::from_raw(raw, bytes) },
        shape,
        dtype,
    }
}

fn build_mq2(rows: usize, k: usize) -> Vec<u8> {
    let groups = k / 256;
    let mut out = Vec::with_capacity(rows * groups * 72);
    for row in 0..rows {
        for g in 0..groups {
            for value in [-3.0f32, -1.0, 1.0, 3.0] {
                out.extend_from_slice(&f32_to_f16_bits(value).to_le_bytes());
            }
            for byte in 0..64 {
                let mut packed = 0u8;
                for i in 0..4 {
                    let q = ((row * 17 + g * 13 + byte * 5 + i) & 3) as u8;
                    packed |= q << (i * 2);
                }
                out.push(packed);
            }
        }
    }
    out
}

fn mq2_value(weights: &[u8], row: usize, k: usize, col: usize) -> f32 {
    let groups = k / 256;
    let g = col / 256;
    let within = col & 255;
    let gp = (row * groups + g) * 72;
    let byte = weights[gp + 8 + within / 4];
    let q = (byte >> ((within & 3) * 2)) & 3;
    [-3.0f32, -1.0, 1.0, 3.0][q as usize]
}

fn grouped_case(gpu: &mut Gpu, m: usize, k: usize, slots: usize, label: &str) {
    assert_eq!(slots % 16, 0);
    let weights = build_mq2(m, k);
    let x: Vec<f32> = (0..slots * k)
        .map(|i| ((i % 9) as f32 - 4.0) * 0.25)
        .collect();
    let x_bytes = as_bytes_f32(&x);
    let w = gpu.hip.malloc(weights.len()).unwrap();
    let x_dev = gpu.hip.malloc(x_bytes.len()).unwrap();
    let y = gpu.hip.malloc(slots * m * 4).unwrap();
    gpu.hip.memcpy_htod(&w, &weights).unwrap();
    gpu.hip.memcpy_htod(&x_dev, &x_bytes).unwrap();

    let ep_bytes = (w.as_ptr() as u64).to_le_bytes();
    let ep = gpu.hip.malloc(8).unwrap();
    gpu.hip.memcpy_htod(&ep, &ep_bytes).unwrap();
    let tile_ids = vec![0i32; slots / 16];
    let tile_bytes = as_bytes_i32(&tile_ids);
    let tile = gpu.hip.malloc(tile_bytes.len()).unwrap();
    gpu.hip.memcpy_htod(&tile, &tile_bytes).unwrap();
    let permutation: Vec<i32> = (0..slots as i32).collect();
    let permutation_bytes = as_bytes_i32(&permutation);
    let perm = gpu.hip.malloc(permutation_bytes.len()).unwrap();
    gpu.hip.memcpy_htod(&perm, &permutation_bytes).unwrap();

    let ep_t = wrap(ep.as_ptr(), 8, vec![1], DType::Raw);
    let tile_t = wrap(tile.as_ptr(), tile_bytes.len(), vec![slots / 16], DType::Raw);
    let perm_t = wrap(perm.as_ptr(), permutation_bytes.len(), vec![slots], DType::Raw);
    let x_t = wrap(x_dev.as_ptr(), x_bytes.len(), vec![slots, k], DType::F32);
    let y_t = wrap(y.as_ptr(), slots * m * 4, vec![slots, m], DType::F32);

    gpu.gemm_mq2g256_lloyd_moe_grouped_wmma_4w_k2(
        &ep_t, &tile_t, &perm_t, &x_t, &y_t, m, k, 1, slots, slots,
    )
    .unwrap();
    gpu.hip.device_synchronize().unwrap();
    let mut y_bytes = vec![0u8; slots * m * 4];
    gpu.hip.memcpy_dtoh(&mut y_bytes, &y).unwrap();
    let y_host: &[f32] = unsafe {
        std::slice::from_raw_parts(y_bytes.as_ptr() as *const f32, slots * m)
    };

    let mut max_abs = 0.0f32;
    let mut max_rel = 0.0f32;
    let mut finite = true;
    let sample_rows = [0usize, m / 3, m / 2, m - 1];
    let sample_slots = [0usize, slots / 3, slots / 2, slots - 1];
    for &slot in &sample_slots {
        for &row in &sample_rows {
            let mut reference = 0.0f32;
            for col in 0..k {
                reference += mq2_value(&weights, row, k, col) * x[slot * k + col];
            }
            let got = y_host[slot * m + row];
            finite &= got.is_finite();
            let abs = (got - reference).abs();
            let rel = abs / reference.abs().max(1e-6);
            max_abs = max_abs.max(abs);
            max_rel = max_rel.max(rel);
        }
    }
    assert!(finite && max_rel < 2e-3, "{label}: max_rel={max_rel}");

    for _ in 0..3 {
        gpu.gemm_mq2g256_lloyd_moe_grouped_wmma_4w_k2(
            &ep_t, &tile_t, &perm_t, &x_t, &y_t, m, k, 1, slots, slots,
        )
        .unwrap();
    }
    gpu.hip.device_synchronize().unwrap();
    let start = Instant::now();
    for _ in 0..10 {
        gpu.gemm_mq2g256_lloyd_moe_grouped_wmma_4w_k2(
            &ep_t, &tile_t, &perm_t, &x_t, &y_t, m, k, 1, slots, slots,
        )
        .unwrap();
    }
    gpu.hip.device_synchronize().unwrap();
    let us = start.elapsed().as_secs_f64() * 1e5;
    println!(
        "grouped {label}: M={m} K={k} slots={slots} us={us:.3} max_abs={max_abs:.6} max_rel={max_rel:.6}"
    );

    std::mem::forget(ep_t);
    std::mem::forget(tile_t);
    std::mem::forget(perm_t);
    std::mem::forget(x_t);
    std::mem::forget(y_t);
}

fn indexer_case(gpu: &mut Gpu) {
    const B: usize = 64;
    const H: usize = 64;
    const D: usize = 128;
    const N: usize = 512;
    let q: Vec<f32> = (0..B * H * D)
        .map(|i| ((i % 29) as f32 - 14.0) / 64.0)
        .collect();
    let k: Vec<f32> = (0..N * D)
        .map(|i| ((i % 31) as f32 - 15.0) / 64.0)
        .collect();
    let weights: Vec<f32> = (0..B * H).map(|i| 0.5 + (i % 7) as f32 / 16.0).collect();
    let valid = vec![N as i32; B];
    let qb = as_bytes_f32(&q);
    let kb = as_bytes_f32(&k);
    let wb = as_bytes_f32(&weights);
    let nb = as_bytes_i32(&valid);
    let qd = gpu.hip.malloc(qb.len()).unwrap();
    let kd = gpu.hip.malloc(kb.len()).unwrap();
    let wd = gpu.hip.malloc(wb.len()).unwrap();
    let nd = gpu.hip.malloc(nb.len()).unwrap();
    let reference = gpu.hip.malloc(B * N * 4).unwrap();
    let candidate = gpu.hip.malloc(B * N * 4).unwrap();
    gpu.hip.memcpy_htod(&qd, &qb).unwrap();
    gpu.hip.memcpy_htod(&kd, &kb).unwrap();
    gpu.hip.memcpy_htod(&wd, &wb).unwrap();
    gpu.hip.memcpy_htod(&nd, &nb).unwrap();
    let qt = wrap(qd.as_ptr(), qb.len(), vec![B, H, D], DType::F32);
    let kt = wrap(kd.as_ptr(), kb.len(), vec![N, D], DType::F32);
    let wt = wrap(wd.as_ptr(), wb.len(), vec![B, H], DType::F32);
    let nt = wrap(nd.as_ptr(), nb.len(), vec![B], DType::Raw);
    let rt = wrap(reference.as_ptr(), B * N * 4, vec![B, N], DType::F32);
    let ct = wrap(candidate.as_ptr(), B * N * 4, vec![B, N], DType::F32);

    gpu.indexer_relu_score_batched_f32(&qt, &kt, &wt, &nt, &rt, H as i32, D as i32, N as i32, B as i32)
        .unwrap();
    gpu.indexer_relu_score_wmma_batched_f32(&qt, &kt, &wt, &nt, &ct, H as i32, D as i32, N as i32, B as i32)
        .unwrap();
    gpu.hip.device_synchronize().unwrap();
    let mut rb = vec![0u8; B * N * 4];
    let mut cb = vec![0u8; B * N * 4];
    gpu.hip.memcpy_dtoh(&mut rb, &reference).unwrap();
    gpu.hip.memcpy_dtoh(&mut cb, &candidate).unwrap();
    let r: &[f32] = unsafe { std::slice::from_raw_parts(rb.as_ptr() as *const f32, B * N) };
    let c: &[f32] = unsafe { std::slice::from_raw_parts(cb.as_ptr() as *const f32, B * N) };
    let mut err2 = 0.0f64;
    let mut ref2 = 0.0f64;
    let mut max_abs = 0.0f32;
    for (&rv, &cv) in r.iter().zip(c) {
        assert!(cv.is_finite());
        let d = cv - rv;
        err2 += (d as f64) * (d as f64);
        ref2 += (rv as f64) * (rv as f64);
        max_abs = max_abs.max(d.abs());
    }
    let rel_rmse = (err2 / ref2.max(1e-30)).sqrt();
    assert!(rel_rmse < 2e-3, "indexer rel_rmse={rel_rmse}");

    let time = |wmma: bool, gpu: &mut Gpu| {
        let start = Instant::now();
        for _ in 0..20 {
            if wmma {
                gpu.indexer_relu_score_wmma_batched_f32(&qt, &kt, &wt, &nt, &ct, H as i32, D as i32, N as i32, B as i32)
                    .unwrap();
            } else {
                gpu.indexer_relu_score_batched_f32(&qt, &kt, &wt, &nt, &rt, H as i32, D as i32, N as i32, B as i32)
                    .unwrap();
            }
        }
        gpu.hip.device_synchronize().unwrap();
        start.elapsed().as_secs_f64() * 5e4
    };
    let scalar_us = time(false, gpu);
    let wmma_us = time(true, gpu);
    println!(
        "indexer B={B} N={N}: scalar_us={scalar_us:.3} gfx12_wmma_us={wmma_us:.3} speedup={:.3}x rel_rmse={rel_rmse:.6} max_abs={max_abs:.6}",
        scalar_us / wmma_us
    );

    std::mem::forget(qt);
    std::mem::forget(kt);
    std::mem::forget(wt);
    std::mem::forget(nt);
    std::mem::forget(rt);
    std::mem::forget(ct);
}

fn main() {
    let mut gpu = Gpu::init().expect("gpu init");
    assert!(gpu.arch_caps.has_wmma_w32_gfx12(), "requires gfx12");
    println!("arch={}", gpu.arch);
    grouped_case(&mut gpu, 64, 256, 16, "sanity");
    grouped_case(&mut gpu, 2048, 4096, 384, "gate-up-B64-k6");
    grouped_case(&mut gpu, 4096, 2048, 384, "down-B64-k6");
    indexer_case(&mut gpu);
}
