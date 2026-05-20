// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

use hipfire_runtime::llama::f32_to_f16;
use rdna_compute::{DType, Gpu};
use std::time::Instant;

fn quantize_hfq6g256(f32_data: &[f32]) -> Vec<u8> {
    let group_size = 256;
    let block_bytes = 200;
    let n = f32_data.len();
    let n_blocks = (n + group_size - 1) / group_size;
    let mut output = vec![0u8; n_blocks * block_bytes];
    for b in 0..n_blocks {
        let start = b * group_size;
        let end = (start + group_size).min(n);
        let group = &f32_data[start..end];
        let min_val = group.iter().cloned().fold(f32::INFINITY, f32::min);
        let max_val = group.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let range = max_val - min_val;
        let scale = if range > 0.0 { range / 63.0 } else { 1.0 };
        let inv_scale = if range > 0.0 { 1.0 / scale } else { 0.0 };
        let out_off = b * block_bytes;
        output[out_off..out_off + 4].copy_from_slice(&scale.to_le_bytes());
        output[out_off + 4..out_off + 8].copy_from_slice(&min_val.to_le_bytes());
        let actual_len = end - start;
        for i in (0..256).step_by(4) {
            let q0 = if i < actual_len { ((group[i] - min_val) * inv_scale + 0.5) as u8 } else { 0 };
            let q1 = if i + 1 < actual_len { ((group[i+1] - min_val) * inv_scale + 0.5) as u8 } else { 0 };
            let q2 = if i + 2 < actual_len { ((group[i+2] - min_val) * inv_scale + 0.5) as u8 } else { 0 };
            let q3 = if i + 3 < actual_len { ((group[i+3] - min_val) * inv_scale + 0.5) as u8 } else { 0 };
            let byte_off = 8 + (i / 4) * 3;
            output[out_off + byte_off]     = q0.min(63) | (q1.min(63) << 6);
            output[out_off + byte_off + 1] = (q1.min(63) >> 2) | (q2.min(63) << 4);
            output[out_off + byte_off + 2] = (q2.min(63) >> 4) | (q3.min(63) << 2);
        }
    }
    output
}

fn f32_to_f16_bytes(f32_data: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(f32_data.len() * 2);
    for &v in f32_data {
        let bits = f32_to_f16(v);
        out.extend_from_slice(&bits.to_le_bytes());
    }
    out
}

fn bench_shape(gpu: &mut Gpu, m: usize, k: usize, batch: usize, label: &str) {
    let weights_f32: Vec<f32> = (0..m*k).map(|i| ((i as f32) * 1e-4) % 1.0 - 0.5).collect();
    let x_f32: Vec<f32> = (0..batch*k).map(|i| ((i as f32) * 1e-4) % 1.0 - 0.5).collect();
    let y_init: Vec<f32> = (0..batch*m).map(|i| ((i as f32) * 7e-5) % 0.5 - 0.25).collect();

    let x_tensor = gpu.upload_f32(&x_f32, &[batch * k]).expect("x");
    let y_mw16 = gpu.alloc_tensor(&[batch * m], DType::F32).expect("y_mw16");
    let y_hfq6 = gpu.alloc_tensor(&[batch * m], DType::F32).expect("y_hfq6");

    let f16_bytes = f32_to_f16_bytes(&weights_f32);
    let w_f16 = gpu.upload_raw(&f16_bytes, &[m * k]).expect("w_f16");

    let hfq6_bytes = quantize_hfq6g256(&weights_f32);
    let w_hfq6 = gpu.upload_raw(&hfq6_bytes, &[m * k / 256 * 200]).expect("w_hfq6");

    let y_bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(y_init.as_ptr() as *const u8, y_init.len() * 4)
    };

    for _ in 0..5 {
        gpu.hip.memcpy_htod(&y_mw16.buf, y_bytes).unwrap();
        gpu.gemm_f16_batched_lmhead(&w_f16, &x_tensor, &y_mw16, m, k, batch).unwrap();
        gpu.hip.memcpy_htod(&y_hfq6.buf, y_bytes).unwrap();
        gpu.gemm_hfq6g256_residual_wmma(&w_hfq6, &x_tensor, &y_hfq6, m, k, batch).unwrap();
    }
    gpu.hip.device_synchronize().unwrap();

    let n_iters = 100;
    gpu.hip.memcpy_htod(&y_mw16.buf, y_bytes).unwrap();
    gpu.hip.device_synchronize().unwrap();
    let t = Instant::now();
    for _ in 0..n_iters {
        gpu.gemm_f16_batched_lmhead(&w_f16, &x_tensor, &y_mw16, m, k, batch).unwrap();
    }
    gpu.hip.device_synchronize().unwrap();
    let mw16_us = t.elapsed().as_secs_f64() * 1e6 / n_iters as f64;

    gpu.hip.memcpy_htod(&y_hfq6.buf, y_bytes).unwrap();
    gpu.hip.device_synchronize().unwrap();
    let t = Instant::now();
    for _ in 0..n_iters {
        gpu.gemm_hfq6g256_residual_wmma(&w_hfq6, &x_tensor, &y_hfq6, m, k, batch).unwrap();
    }
    gpu.hip.device_synchronize().unwrap();
    let hfq6_us = t.elapsed().as_secs_f64() * 1e6 / n_iters as f64;

    let mw16_bw = (m * k * 2) as f64 / mw16_us / 1e3;
    let hfq6_bw = (m * k / 256 * 200) as f64 / hfq6_us / 1e3;
    let mw16_mib = (m * k * 2) as f64 / 1024.0 / 1024.0;
    let hfq6_mib = (m * k / 256 * 200) as f64 / 1024.0 / 1024.0;

    eprintln!("{label} M={m} K={k} N={batch}:");
    eprintln!("  f16_mw16:   {mw16_us:7.1} µs  {mw16_bw:6.1} GiB/s  (weight {mw16_mib:.1} MiB)");
    eprintln!("  hfq6g256:   {hfq6_us:7.1} µs  {hfq6_bw:6.1} GiB/s  (weight {hfq6_mib:.1} MiB)");
    eprintln!("  ratio:      {:.3}×  ({})", mw16_us / hfq6_us, if hfq6_us < mw16_us { "HFQ6 faster" } else { "F16 faster" });
}

fn main() {
    eprintln!("=== draft F16 (mw16 WMMA) vs HFQ6G256 (WMMA K2) at 27B DFlash draft shapes ===");
    let mut gpu = Gpu::init().expect("gpu");

    bench_shape(&mut gpu, 5120, 5120, 16, "wo-dim");
    bench_shape(&mut gpu, 13824, 5120, 16, "ffn-up-dim");
    bench_shape(&mut gpu, 2560, 5120, 16, "qkv-slice");
}
