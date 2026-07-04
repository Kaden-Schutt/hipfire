// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 hipfire contributors
// hipfire — see LICENSE and NOTICE in the project root.
//! Bandwidth scaling of the generic GEMV tier vs bytes/weight on gfx1103.
//!
//! Decode is bandwidth-bound on this UMA APU: the weight matrix is streamed
//! from shared DRAM once per token. This probe times each generic GEMV dtype at
//! a weight size that exceeds all caches and reports achieved GB/s on the weight
//! stream and throughput relative to f16. If the platform is bandwidth-bound,
//! throughput scales ~1/bytes-per-weight: int8 ~2× f16, int4 ~4× f16.
//!
//!   cargo run -p hipfire-rdna --example bench_gemv_dtype_bw [M] [K]

use hipfire_rdna::Gpu;
use std::time::Instant;

fn main() {
    let mut a = std::env::args().skip(1);
    let m: usize = a.next().and_then(|s| s.parse().ok()).unwrap_or(8192);
    let k: usize = a.next().and_then(|s| s.parse().ok()).unwrap_or(8192);

    let mut gpu = Gpu::init().unwrap();
    if !gpu.arch_caps.has_wmma_w32() {
        println!("SKIP: {} lacks wave32 WMMA", gpu.arch);
        return;
    }

    // Zero-filled inputs are fine: this is a bandwidth/throughput probe, not a
    // correctness check (correctness is covered by parity_gemv_generic).
    let w16 = gpu.upload_raw(&vec![0u8; m * k * 2], &[m, k]).unwrap(); // f16/bf16 weight
    let x16 = gpu.upload_raw(&vec![0u8; k * 2], &[k]).unwrap();
    let w8 = gpu.upload_raw(&vec![0u8; m * k], &[m, k]).unwrap(); // int8 weight
    let x8 = gpu.upload_raw(&vec![0u8; k], &[k]).unwrap();
    let w4 = gpu.upload_raw(&vec![0u8; m * k / 2], &[m, k / 2]).unwrap(); // int4 weight
    let x4 = gpu.upload_raw(&vec![0u8; k / 2], &[k / 2]).unwrap();
    let y32 = gpu.upload_raw(&vec![0u8; m * 4], &[m]).unwrap();
    let y16 = gpu.upload_raw(&vec![0u8; m * 2], &[m]).unwrap();

    let warmup = 30;
    let trials = 200;

    println!(
        "gemv dtype bandwidth  M={m} K={k}  weight={} MiB(f16)  on {}\n",
        m * k * 2 / (1024 * 1024),
        gpu.arch
    );
    println!(
        "{:<16} {:>8} {:>11} {:>11} {:>9}",
        "kernel", "B/w", "µs/call", "GB/s(w)", "vs f16"
    );
    println!("{}", "-".repeat(60));

    // (label, bytes/weight, closure)
    let mut runners: Vec<(&str, f64, Box<dyn FnMut(&mut Gpu)>)> = vec![
        (
            "f16->f32",
            2.0,
            Box::new(|g: &mut Gpu| g.gemv_f16_f32(&w16, &x16, &y32, m, k).unwrap()),
        ),
        (
            "bf16->bf16",
            2.0,
            Box::new(|g: &mut Gpu| g.gemv_bf16_bf16(&w16, &x16, &y16, m, k).unwrap()),
        ),
        (
            "iu8->i32",
            1.0,
            Box::new(|g: &mut Gpu| g.gemv_iu8_i32(&w8, &x8, &y32, m, k).unwrap()),
        ),
        (
            "iu4->i32",
            0.5,
            Box::new(|g: &mut Gpu| g.gemv_iu4_i32(&w4, &x4, &y32, m, k).unwrap()),
        ),
    ];

    let mut f16_us = 0.0f64;
    for (label, bpw, run) in runners.iter_mut() {
        for _ in 0..warmup {
            run(&mut gpu);
        }
        gpu.hip.device_synchronize().unwrap();
        let t = Instant::now();
        for _ in 0..trials {
            run(&mut gpu);
        }
        gpu.hip.device_synchronize().unwrap();
        let us = t.elapsed().as_secs_f64() * 1e6 / trials as f64;
        let w_bytes = (m * k) as f64 * *bpw;
        let gbps = w_bytes / (us * 1e-6) / 1e9;
        if *label == "f16->f32" {
            f16_us = us;
        }
        let speedup = if f16_us > 0.0 { f16_us / us } else { 1.0 };
        println!("{label:<16} {bpw:>8.1} {us:>11.2} {gbps:>11.1} {speedup:>8.2}x");
    }

    println!("\nbandwidth-bound expectation: int8 ~2.0x, int4 ~4.0x f16.");
}
