// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Microbench: `v_wmma_f32_16x16x16_bf16` wave32 (`_w32`) vs wave64 (`_w64`).
//!
//! Both kernels chain `ITERS` 16x16x16 WMMA accumulations per wave over a grid
//! of `G` waves (1 wave/block: blockDim 32 for w32, 64 for w64). Each variant
//! therefore issues the SAME `G*ITERS` WMMA instructions, each producing one
//! 8192-FLOP tile — so equal aggregate WGP throughput ⇒ equal time. Any gap is
//! the cost of scheduling a 64-lane wave for a tile on wave32-native RDNA3 (vs
//! the lower acc register footprint of wave64). Reports µs, TFLOP/s, ratio.
//!
//! Run (gpu-lock-coordinated):
//!   hipfire lock acquire
//!   cargo run --release -p hipfire-rdna --example bench_wmma_wave32_vs_wave64

use hip_bridge::KernargBlob;
use hipfire_rdna::{DType, Gpu};
use std::ffi::c_void;

const KERNEL_SRC_W32: &str = include_str!("../../../kernels/src/bench_wmma_bf16_wave.hip");
const KERNEL_SRC_W64: &str = include_str!("../../../kernels/src/bench_wmma_bf16_wave64.hip");

fn f32_to_bf16_bits(x: f32) -> u16 {
    let b = x.to_bits();
    ((b + 0x0000_7fff + ((b >> 16) & 1)) >> 16) as u16
}

#[allow(clippy::too_many_arguments)]
fn time_variant(
    gpu: &Gpu,
    kernel: &str,
    block: u32,
    grid: u32,
    iters: i32,
    a: &hipfire_rdna::GpuTensor,
    b: &hipfire_rdna::GpuTensor,
    y: &hipfire_rdna::GpuTensor,
    timed: usize,
) -> f64 {
    let launch = || {
        let mut kb = KernargBlob::new();
        kb.push_ptr(a.buf.as_ptr() as *const c_void);
        kb.push_ptr(b.buf.as_ptr() as *const c_void);
        kb.push_ptr(y.buf.as_ptr() as *const c_void);
        kb.push_i32(iters);
        kb.pad_to(16);
        gpu.launch_kernel_blob(kernel, [grid, 1, 1], [block, 1, 1], 0, kb.as_mut_slice())
            .unwrap();
    };
    for _ in 0..5 {
        launch();
    }
    gpu.hip.device_synchronize().unwrap();
    let ev0 = gpu.hip.event_create().unwrap();
    let ev1 = gpu.hip.event_create().unwrap();
    gpu.hip.event_record(&ev0, None).unwrap();
    for _ in 0..timed {
        launch();
    }
    gpu.hip.event_record(&ev1, None).unwrap();
    gpu.hip.event_synchronize(&ev1).unwrap();
    gpu.hip.event_elapsed_ms(&ev0, &ev1).unwrap() as f64 / timed as f64
}

fn main() {
    let mut gpu = Gpu::init().expect("gpu init");
    let arch = gpu.arch.clone();
    eprintln!("=== WMMA bf16 wave32 vs wave64 microbench ===");
    eprintln!("  arch = {arch}");
    if !arch.starts_with("gfx11") && !arch.starts_with("gfx12") {
        eprintln!("  ERROR: WMMA requires RDNA3+ (gfx11/gfx12). Got {arch}.");
        std::process::exit(1);
    }

    gpu.ensure_kernel_public("bench_wmma_w32", KERNEL_SRC_W32, "bench_wmma_bf16_w32")
        .expect("ensure w32");
    gpu.ensure_kernel_public("bench_wmma_w64", KERNEL_SRC_W64, "bench_wmma_bf16_w64")
        .expect("ensure w64");

    // Tiny shared A/B fragment (16x16 bf16).
    let a_bits: Vec<u16> = (0..256)
        .map(|i| f32_to_bf16_bits(0.01 + (i as f32) * 1e-4))
        .collect();
    let b_bits: Vec<u16> = (0..256)
        .map(|i| f32_to_bf16_bits(0.02 - (i as f32) * 1e-4))
        .collect();
    let a_bytes: &[u8] =
        unsafe { std::slice::from_raw_parts(a_bits.as_ptr() as *const u8, a_bits.len() * 2) };
    let b_bytes: &[u8] =
        unsafe { std::slice::from_raw_parts(b_bits.as_ptr() as *const u8, b_bits.len() * 2) };
    let d_a = gpu.upload_raw(a_bytes, &[a_bytes.len()]).unwrap();
    let d_b = gpu.upload_raw(b_bytes, &[b_bytes.len()]).unwrap();

    let iters: i32 = 8192;
    const TIMED: usize = 50;

    println!(
        "{:>8} {:>12} {:>12} {:>12} {:>12} {:>10}",
        "waves", "w32 us", "w64 us", "w32 TFLOP/s", "w64 TFLOP/s", "w32/w64"
    );
    for &g in &[1024u32, 2048, 4096, 8192, 16384] {
        // Y must hold blockDim*8 (w32) and blockDim*4 (w64) per block = 256 f32/block either way.
        let y = gpu.zeros(&[(g as usize) * 256], DType::F32).unwrap();

        let w32_us = time_variant(
            &gpu,
            "bench_wmma_bf16_w32",
            32,
            g,
            iters,
            &d_a,
            &d_b,
            &y,
            TIMED,
        ) * 1000.0;
        let w64_us = time_variant(
            &gpu,
            "bench_wmma_bf16_w64",
            64,
            g,
            iters,
            &d_a,
            &d_b,
            &y,
            TIMED,
        ) * 1000.0;

        // Both issue g*iters WMMA tiles, 8192 FLOPs each.
        let flop = (g as f64) * (iters as f64) * 8192.0;
        let w32_tflops = flop / (w32_us * 1e-6) / 1e12;
        let w64_tflops = flop / (w64_us * 1e-6) / 1e12;
        println!(
            "{g:>8} {w32_us:>12.1} {w64_us:>12.1} {w32_tflops:>12.2} {w64_tflops:>12.2} {:>10.3}",
            w32_us / w64_us
        );
    }
}
