// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Decode-shape bandwidth probe for `gemv_bf16_xf32` (the OvisOCR2 / any
//! bf16-source trunk decode path) against `gemv_q8_0`, which measures at the
//! memory roof on the same device.
//!
//! GEMV at batch 1 is pure weight streaming: the only figure of merit is
//! achieved GB/s vs the card's peak. FLOPs are irrelevant here.
//!
//! Usage:
//!   cargo run --release -p rdna-compute --example bench_gemv_bf16

use rdna_compute::{DType, Gpu};
use std::time::Instant;

const WARMUP: usize = 20;
const TRIALS: usize = 200;
/// RX 9070 XT (gfx1201) peak memory bandwidth.
const ROOF_GBS: f64 = 640.0;

fn main() {
    let mut gpu = Gpu::init().expect("gpu init");
    println!("Arch: {}\n", gpu.arch);

    // Qwen3.5-VL 0.8B decoder shapes: hidden 1024, 24 layers.
    let shapes: &[(usize, usize, &str)] = &[
        (1024, 1024, "o_proj    1024x1024"),
        (2048, 1024, "qkv       2048x1024"),
        (3072, 1024, "gate/up   3072x1024"),
        (1024, 3072, "down      1024x3072"),
        (8192, 1024, "lm_head-ish 8192x1024"),
    ];

    println!("{:24} {:>10} {:>10} {:>9}   {:>10} {:>10} {:>9}",
             "shape", "bf16 us", "bf16 GB/s", "% roof", "q8 us", "q8 GB/s", "% roof");

    for &(m, k, label) in shapes {
        let x = gpu.zeros(&[k], DType::F32).expect("x");
        let y = gpu.zeros(&[m], DType::F32).expect("y");

        // bf16 weights: [M, K] raw u16.
        let wb = gpu.hip.malloc(m * k * 2).expect("malloc bf16 W");
        gpu.hip.memcpy_htod(&wb, &vec![0x3Fu8; m * k * 2]).expect("copy bf16");
        let w_bf16 = rdna_compute::GpuTensor {
            buf: wb,
            shape: vec![m, k],
            dtype: DType::BF16,
        };

        for _ in 0..WARMUP {
            gpu.gemv_bf16_xf32(&w_bf16, &x, &y, m, k).expect("bf16 warmup");
        }
        gpu.hip.device_synchronize().expect("sync");
        let t0 = Instant::now();
        for _ in 0..TRIALS {
            gpu.gemv_bf16_xf32(&w_bf16, &x, &y, m, k).expect("bf16 trial");
        }
        gpu.hip.device_synchronize().expect("sync");
        let bf16_us = t0.elapsed().as_secs_f64() / TRIALS as f64 * 1e6;
        let bf16_gbs = (m * k * 2) as f64 / bf16_us / 1e3;

        // q8_0 control: 34 bytes per 32-element block.
        let q8_bytes = m * (k / 32) * 34;
        let wq = gpu.hip.malloc(q8_bytes).expect("malloc q8 W");
        gpu.hip.memcpy_htod(&wq, &vec![0x10u8; q8_bytes]).expect("copy q8");
        let w_q8 = rdna_compute::GpuTensor {
            buf: wq,
            shape: vec![m, k],
            dtype: DType::Q8_0,
        };
        for _ in 0..WARMUP {
            gpu.gemv_q8_0(&w_q8, &x, &y, m, k).expect("q8 warmup");
        }
        gpu.hip.device_synchronize().expect("sync");
        let t1 = Instant::now();
        for _ in 0..TRIALS {
            gpu.gemv_q8_0(&w_q8, &x, &y, m, k).expect("q8 trial");
        }
        gpu.hip.device_synchronize().expect("sync");
        let q8_us = t1.elapsed().as_secs_f64() / TRIALS as f64 * 1e6;
        let q8_gbs = q8_bytes as f64 / q8_us / 1e3;

        println!("{label:24} {bf16_us:10.1} {bf16_gbs:10.1} {:8.1}%   {q8_us:10.1} {q8_gbs:10.1} {:8.1}%",
                 bf16_gbs / ROOF_GBS * 100.0, q8_gbs / ROOF_GBS * 100.0);

        gpu.free_tensor(w_bf16).ok();
        gpu.free_tensor(w_q8).ok();
        gpu.free_tensor(x).ok();
        gpu.free_tensor(y).ok();
    }
    println!("\nroof = {ROOF_GBS} GB/s. Batch-1 GEMV is weight-streaming; % roof is the whole story.");
}
