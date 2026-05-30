// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Nick Woolmer
// hipfire — see LICENSE and NOTICE in the project root.

//! Correctness + perf A/B for DeepSeek V4 `wo_per_group_batched_q8_0_wmma_4w`
//! against the existing one-warp scalar `wo_per_group_batched_q8_0` kernel.

use rdna_compute::{DType, Gpu};
use std::time::Instant;

const WARMUP: usize = 4;
const TRIALS: usize = 20;
const ATOL: f32 = 2e-2;
const RTOL: f32 = 2e-2;

fn f32_to_f16_bits(v: f32) -> u16 {
    let bits = v.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = (((bits >> 23) & 0xff) as i32) - 127 + 15;
    let mant = bits & 0x7fffff;
    if exp <= 0 {
        return sign;
    }
    if exp >= 31 {
        return sign | 0x7c00;
    }
    sign | ((exp as u16) << 10) | ((mant >> 13) as u16)
}

fn quantize_q8(g: usize, m: usize, k: usize) -> Vec<u8> {
    assert_eq!(k % 32, 0);
    let blocks_per_row = k / 32;
    let mut out = Vec::with_capacity(g * m * blocks_per_row * 34);
    for gg in 0..g {
        for r in 0..m {
            for bi in 0..blocks_per_row {
                out.extend_from_slice(&f32_to_f16_bits(1.0 / 64.0).to_le_bytes());
                for lane in 0..32 {
                    let v = ((gg * 17 + r * 13 + bi * 7 + lane) % 31) as i8 - 15;
                    out.push(v as u8);
                }
            }
        }
    }
    out
}

fn run_shape(gpu: &mut Gpu, g: usize, m: usize, k: usize, batch: usize, label: &str) {
    println!("\n=== {label} | G={g} M={m} K={k} B={batch} ===");
    let w_bytes = quantize_q8(g, m, k);
    let x: Vec<f32> = (0..batch * g * k)
        .map(|i| ((i % 23) as f32 - 11.0) / 16.0)
        .collect();

    let w = gpu.upload_raw(&w_bytes, &[w_bytes.len()]).expect("upload W");
    let x = gpu.upload_f32(&x, &[batch, g, k]).expect("upload X");
    let y1 = gpu.zeros(&[batch, g, m], DType::F32).expect("alloc Y1");
    let yw = gpu.zeros(&[batch, g, m], DType::F32).expect("alloc YW");

    gpu.wo_per_group_batched_q8_0_1w(&w, &x, &y1, g as i32, m as i32, k as i32, batch as i32)
        .expect("1w correctness");
    gpu.hip.device_synchronize().expect("sync 1w");
    gpu.wo_per_group_batched_q8_0_wmma_4w(&w, &x, &yw, g as i32, m as i32, k as i32, batch as i32)
        .expect("wmma correctness");
    gpu.hip.device_synchronize().expect("sync wmma");

    let y1_host = gpu.download_f32(&y1).expect("download y1");
    let yw_host = gpu.download_f32(&yw).expect("download yw");
    let mut max_abs = 0.0f32;
    let mut max_rel = 0.0f32;
    let mut bad = 0usize;
    let mut nan = 0usize;
    for (&a, &b) in y1_host.iter().zip(yw_host.iter()) {
        if !b.is_finite() {
            nan += 1;
            continue;
        }
        let d = (a - b).abs();
        let r = d / a.abs().max(1e-6);
        max_abs = max_abs.max(d);
        max_rel = max_rel.max(r);
        if d > ATOL && r > RTOL {
            bad += 1;
        }
    }
    println!(
        "  correctness: max_abs={max_abs:.3e} max_rel={max_rel:.3e} bad={bad}/{} nan={nan} {}",
        y1_host.len(),
        if bad == 0 && nan == 0 { "OK" } else { "FAIL" },
    );
    if bad != 0 || nan != 0 {
        return;
    }

    for _ in 0..WARMUP {
        gpu.wo_per_group_batched_q8_0_1w(&w, &x, &y1, g as i32, m as i32, k as i32, batch as i32)
            .unwrap();
    }
    gpu.hip.device_synchronize().unwrap();
    let t0 = Instant::now();
    for _ in 0..TRIALS {
        gpu.wo_per_group_batched_q8_0_1w(&w, &x, &y1, g as i32, m as i32, k as i32, batch as i32)
            .unwrap();
    }
    gpu.hip.device_synchronize().unwrap();
    let one_us = t0.elapsed().as_secs_f64() * 1e6 / TRIALS as f64;

    for _ in 0..WARMUP {
        gpu.wo_per_group_batched_q8_0_wmma_4w(&w, &x, &yw, g as i32, m as i32, k as i32, batch as i32)
            .unwrap();
    }
    gpu.hip.device_synchronize().unwrap();
    let t0 = Instant::now();
    for _ in 0..TRIALS {
        gpu.wo_per_group_batched_q8_0_wmma_4w(&w, &x, &yw, g as i32, m as i32, k as i32, batch as i32)
            .unwrap();
    }
    gpu.hip.device_synchronize().unwrap();
    let wmma_us = t0.elapsed().as_secs_f64() * 1e6 / TRIALS as f64;

    let flops = 2.0 * (g * m * k * batch) as f64;
    println!(
        "  1w: {one_us:>9.1} us ({:>7.0} GFLOPS)   wmma: {wmma_us:>9.1} us ({:>7.0} GFLOPS)   speedup: {:.2}x",
        flops / one_us / 1e3,
        flops / wmma_us / 1e3,
        one_us / wmma_us,
    );
}

fn main() {
    let mut gpu = Gpu::init().expect("gpu init");
    println!("Arch: {}", gpu.arch);

    let shapes = [
        (8, 1024, 4096, 64, "small B=64"),
        (8, 1024, 4096, 256, "mid B=256"),
        (8, 1024, 4096, 1024, "prefill B=1024"),
    ];
    for (g, m, k, batch, label) in shapes {
        run_shape(&mut gpu, g, m, k, batch, label);
    }
}
