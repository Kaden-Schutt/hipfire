// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Time HFQ4 vs HFQ6 WMMA GEMM at prefill-realistic sizes.
//! Same matrix shape, same X — just different weight format.

use std::time::Instant;

fn quantize_hfq4g256(f32_data: &[f32]) -> Vec<u8> {
    let group_size = 256;
    let block_bytes = 136;
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
        let scale = if range > 0.0 { range / 15.0 } else { 1.0 };
        let inv_scale = if range > 0.0 { 1.0 / scale } else { 0.0 };
        let out_off = b * block_bytes;
        output[out_off..out_off + 4].copy_from_slice(&scale.to_le_bytes());
        output[out_off + 4..out_off + 8].copy_from_slice(&min_val.to_le_bytes());
        let actual_len = end - start;
        for i in 0..128 {
            let lo = if 2*i < actual_len { ((group[2*i] - min_val) * inv_scale + 0.5) as u8 } else { 0 };
            let hi = if 2*i+1 < actual_len { ((group[2*i+1] - min_val) * inv_scale + 0.5) as u8 } else { 0 };
            output[out_off + 8 + i] = lo.min(15) | (hi.min(15) << 4);
        }
    }
    output
}

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

fn main() {
    let mut gpu = rdna_compute::Gpu::init().unwrap();
    eprintln!("GPU: {}", gpu.arch);

    // 4B model proj shapes: gate/up = 14336 × 2560, qkv/wo = 2560 × 2560
    let shapes = [
        ("wo (2560x2560)", 2560usize, 2560usize),
        ("gate (14336x2560)", 14336usize, 2560usize),
    ];
    let n = 128usize; // prefill batch

    for (name, m, k) in &shapes {
        let (m, k) = (*m, *k);
        let w: Vec<f32> = (0..m*k).map(|i| ((i as f32 * 0.123) % 1.0) - 0.5).collect();
        let x: Vec<f32> = (0..n*k).map(|i| ((i as f32 * 0.456) % 1.0) - 0.5).collect();

        let q4 = quantize_hfq4g256(&w);
        let q6 = quantize_hfq6g256(&w);

        let d_x = gpu.upload_f32(&x, &[n * k]).unwrap();
        let d_a4 = gpu.upload_raw(&q4, &[q4.len()]).unwrap();
        let d_a6 = gpu.upload_raw(&q6, &[q6.len()]).unwrap();
        let y_init = vec![0.0f32; n * m];

        // Warmup both kernels to compile / cache
        for _ in 0..3 {
            let d_y = gpu.upload_f32(&y_init, &[n * m]).unwrap();
            gpu.gemm_hfq4g256_residual(&d_a4, &d_x, &d_y, m, k, n).unwrap();
            let d_y = gpu.upload_f32(&y_init, &[n * m]).unwrap();
            gpu.gemm_hfq6g256_residual(&d_a6, &d_x, &d_y, m, k, n).unwrap();
        }
        gpu.hip.device_synchronize().unwrap();

        // Time HFQ4 (20 iters)
        let iters = 20;
        let t0 = Instant::now();
        for _ in 0..iters {
            let d_y = gpu.upload_f32(&y_init, &[n * m]).unwrap();
            gpu.gemm_hfq4g256_residual(&d_a4, &d_x, &d_y, m, k, n).unwrap();
        }
        gpu.hip.device_synchronize().unwrap();
        let t_hfq4 = t0.elapsed().as_secs_f64() / iters as f64 * 1000.0;

        let t0 = Instant::now();
        for _ in 0..iters {
            let d_y = gpu.upload_f32(&y_init, &[n * m]).unwrap();
            gpu.gemm_hfq6g256_residual(&d_a6, &d_x, &d_y, m, k, n).unwrap();
        }
        gpu.hip.device_synchronize().unwrap();
        let t_hfq6 = t0.elapsed().as_secs_f64() / iters as f64 * 1000.0;

        // BW calc
        let bytes_hfq4 = m * k / 2 + n * k * 4 + n * m * 4;
        let bytes_hfq6 = (m * k / 256) * 200 + n * k * 4 + n * m * 4;
        let bw4 = bytes_hfq4 as f64 / t_hfq4 / 1e6;
        let bw6 = bytes_hfq6 as f64 / t_hfq6 / 1e6;

        eprintln!("{name}: HFQ4={t_hfq4:.2}ms ({bw4:.0} GiB/s)  HFQ6={t_hfq6:.2}ms ({bw6:.0} GiB/s)  ratio={:.2}×",
            t_hfq6 / t_hfq4);
    }
}
