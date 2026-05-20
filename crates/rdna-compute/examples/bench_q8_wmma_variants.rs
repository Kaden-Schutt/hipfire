// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kevin Read
// hipfire — see LICENSE and NOTICE in the project root.

//! T3-1a microbench — Q8_0 batched GEMM via FP16 WMMA vs the Tier 2 substrate.
//!
//! **Status (2026-05-13):** T3-1a recipe pick is LOCKED to FP16-WMMA — the
//! bench in this file delivered 11–30× speedup over the Tier-2 substrate
//! and the 4 production fused kernels (`gemm_qkv/qkvza/gate_up/q8_0_residual
//! _wmma`) all derive from the same template. This harness is retained as
//! a regression probe: re-run if anyone touches the dequant prologue or the
//! WMMA inner-loop layout to confirm the speedup hasn't eroded.
//!
//! Picks the recipe for Tier 3 (see docs/plans/q8-fused-prefill-kernels.md
//! §Element format choice). This pass benches the FP16-WMMA variant against
//! the existing `gemm_q8_0_batched_chunked` substrate at production
//! Qwen3.5-9B Q8 prefill shapes. INT8-WMMA sibling is the next-turn add-on
//! (`bench_q8_int8wmma.hip` + extension to this harness).
//!
//! Outputs per (M, K, N) shape:
//!   - kernel time µs/call
//!   - effective weight-bandwidth GB/s
//!   - speedup ratio vs substrate
//!   - max relative output diff (sanity — fp16 tolerance expected)
//!
//! Usage:
//!   cargo run --release --example bench_q8_wmma_variants

use hip_bridge::KernargBlob;
use rdna_compute::{DType, Gpu};
use std::ffi::c_void;

const KERNEL_SRC: &str = include_str!("../../../kernels/src/bench_q8_fp16wmma.hip");
const KERNEL_NAME: &str = "bench_q8_fp16wmma";

fn main() {
    let mut gpu = Gpu::init().expect("gpu init");
    let arch = gpu.arch.clone();
    eprintln!("=== Q8 WMMA variant microbench ===");
    eprintln!("  arch = {arch}");
    if !arch.starts_with("gfx11") && !arch.starts_with("gfx12") {
        eprintln!("  ERROR: WMMA path requires RDNA3+ (gfx11/gfx12). Got {arch}.");
        std::process::exit(1);
    }

    gpu.ensure_kernel_public(KERNEL_NAME, KERNEL_SRC, KERNEL_NAME)
        .expect("ensure_kernel_public");

    // Production Qwen3.5-9B Q8 prefill projection shapes.
    let shapes: Vec<(usize, usize, &str)> = vec![
        (4096, 4096, "QKV-ish    M=4096 K=4096"),
        (11008, 4096, "gate/up    M=11008 K=4096"),
        (4096, 11008, "w_down     M=4096 K=11008"),
    ];

    let batches: Vec<usize> = vec![16, 64, 128, 256];

    for (m, k, label) in &shapes {
        let (m, k) = (*m, *k);
        assert!(k % 32 == 0, "K must be a multiple of 32 (Q8_0 block size)");
        assert!(m % 16 == 0, "M should be a multiple of 16 for WMMA tile (got {m})");
        eprintln!("\n--- {label} ---");

        // Synthetic Q8_0 weights: [M, K/32 * 34]. Random int8 + per-block fp16 scale.
        let blocks_per_row = k / 32;
        let row_bytes = blocks_per_row * 34;
        let total_bytes = m * row_bytes;
        let mut q8 = vec![0u8; total_bytes];
        let mut seed: u32 = 0xA1B2C3D4;
        let mut prng = || {
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            seed
        };
        for row in 0..m {
            for bi in 0..blocks_per_row {
                let off = row * row_bytes + bi * 34;
                // fp16 scale ~ uniform in [0.001, 0.05] (typical Q8 scale range).
                let sf = 0.001 + (prng() as f32 / u32::MAX as f32) * 0.049;
                let sb = f32_to_f16_bits(sf);
                q8[off] = (sb & 0xFF) as u8;
                q8[off + 1] = (sb >> 8) as u8;
                for j in 0..32 {
                    let r = prng();
                    q8[off + 2 + j] = ((r as i32 % 255) - 127) as i8 as u8;
                }
            }
        }
        let d_a = gpu.upload_raw(&q8, &[total_bytes]).unwrap();

        let max_n = *batches.iter().max().unwrap();

        // f32 X for substrate, f16 X for WMMA (both bit-derived from the same f32 source).
        let x_f32: Vec<f32> = (0..max_n * k)
            .map(|i| {
                let v = ((i as i64).wrapping_mul(1103515245).wrapping_add(12345)) as f32;
                (v * 1e-9) % 2.0 - 1.0
            })
            .collect();
        let x_f16_bits: Vec<u16> = x_f32.iter().map(|&v| f32_to_f16_bits(v)).collect();
        let x_f16_bytes: &[u8] = unsafe {
            std::slice::from_raw_parts(x_f16_bits.as_ptr() as *const u8, x_f16_bits.len() * 2)
        };
        let d_x_f32 = gpu.upload_f32(&x_f32, &[max_n * k]).unwrap();
        let d_x_f16 = gpu.upload_raw(x_f16_bytes, &[x_f16_bytes.len()]).unwrap();

        let d_y_sub = gpu.zeros(&[max_n * m], DType::F32).unwrap();
        let d_y_wmma = gpu.zeros(&[max_n * m], DType::F32).unwrap();

        for &n in &batches {
            // Sub-views so the substrate's chunked path sees exactly N rows.
            let x_f32_n = d_x_f32.sub_offset(0, n * k);
            let x_f16_n = d_x_f16.sub_offset(0, n * k); // raw u8 view, layout matches
            let y_sub_n = d_y_sub.sub_offset(0, n * m);
            let y_wmma_n = d_y_wmma.sub_offset(0, n * m);

            // Warmup.
            const WARMUP: usize = 20;
            const ITERS: usize = 200;
            for _ in 0..WARMUP {
                gpu.gemm_q8_0_batched_chunked(&d_a, &x_f32_n, &y_sub_n, m, k, n)
                    .unwrap();
                launch_wmma(&gpu, &d_a, &x_f16_n, &y_wmma_n, m, k, n);
            }
            gpu.hip.device_synchronize().unwrap();

            // --- Substrate timing ---
            let ev0 = gpu.hip.event_create().unwrap();
            let ev1 = gpu.hip.event_create().unwrap();
            gpu.hip.event_record(&ev0, None).unwrap();
            for _ in 0..ITERS {
                gpu.gemm_q8_0_batched_chunked(&d_a, &x_f32_n, &y_sub_n, m, k, n)
                    .unwrap();
            }
            gpu.hip.event_record(&ev1, None).unwrap();
            gpu.hip.event_synchronize(&ev1).unwrap();
            let sub_ms = gpu.hip.event_elapsed_ms(&ev0, &ev1).unwrap() as f64;
            let sub_us = sub_ms * 1000.0 / ITERS as f64;

            // --- WMMA timing ---
            let ev2 = gpu.hip.event_create().unwrap();
            let ev3 = gpu.hip.event_create().unwrap();
            gpu.hip.event_record(&ev2, None).unwrap();
            for _ in 0..ITERS {
                launch_wmma(&gpu, &d_a, &x_f16_n, &y_wmma_n, m, k, n);
            }
            gpu.hip.event_record(&ev3, None).unwrap();
            gpu.hip.event_synchronize(&ev3).unwrap();
            let wmma_ms = gpu.hip.event_elapsed_ms(&ev2, &ev3).unwrap() as f64;
            let wmma_us = wmma_ms * 1000.0 / ITERS as f64;

            // Effective BW (weight bytes per call — the dominant read).
            let w_bytes = total_bytes as f64;
            let sub_gbps = w_bytes / (sub_us * 1e-6) / 1e9;
            let wmma_gbps = w_bytes / (wmma_us * 1e-6) / 1e9;
            let speedup = sub_us / wmma_us;

            // --- Correctness sanity (last iteration's output) ---
            // Note: relative error is unreliable near zero (small |sub| amplifies
            // tiny absolute drift into huge relative drift). Gate rel-error on
            // |sub| > 1% of max — this filters genuine near-zero outliers
            // without hiding real bias.
            let y_sub_host = gpu.download_f32(&y_sub_n).unwrap();
            let y_wmma_host = gpu.download_f32(&y_wmma_n).unwrap();
            let max_sub = y_sub_host.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
            let threshold = max_sub * 0.01;
            let mut max_abs = 0.0f32;
            let mut sum_abs = 0.0f64;
            let mut max_rel_gated = 0.0f32;
            let mut sum_rel_gated = 0.0f64;
            let mut gated_count = 0usize;
            let mut close_count = 0usize;
            for (a, b) in y_sub_host.iter().zip(y_wmma_host.iter()) {
                let d = (a - b).abs();
                if d > max_abs { max_abs = d; }
                sum_abs += d as f64;
                if a.abs() > threshold {
                    let rel = d / a.abs();
                    if rel > max_rel_gated { max_rel_gated = rel; }
                    sum_rel_gated += rel as f64;
                    gated_count += 1;
                    if rel < 0.05 { close_count += 1; }
                }
            }
            let total = y_sub_host.len();
            let mean_abs = sum_abs / total as f64;
            let mean_rel_gated = sum_rel_gated / gated_count.max(1) as f64;
            let close_pct = 100.0 * close_count as f64 / gated_count.max(1) as f64;

            eprintln!(
                "  N={n:4}  substrate {sub_us:7.1}µs ({sub_gbps:5.1} GB/s)  \
                 wmma {wmma_us:7.1}µs ({wmma_gbps:5.1} GB/s)  \
                 speedup ×{speedup:4.2}"
            );
            eprintln!(
                "        |sub|_max={max_sub:.1}  mean|err|={mean_abs:.3e}  max|err|={max_abs:.3e}  \
                 rel(|sub|>{:.2}): mean={mean_rel_gated:.3e}  max={max_rel_gated:.3e}  \
                 <5%={close_pct:.1}% of {gated_count}/{total}",
                threshold
            );
        }
    }
}

fn launch_wmma(
    gpu: &Gpu,
    a_q8: &rdna_compute::GpuTensor,
    x_f16: &rdna_compute::GpuTensor,
    y_f32: &rdna_compute::GpuTensor,
    m: usize,
    k: usize,
    n: usize,
) {
    let mut kb = KernargBlob::new();
    kb.push_ptr(a_q8.buf.as_ptr() as *const c_void);
    kb.push_ptr(x_f16.buf.as_ptr() as *const c_void);
    kb.push_ptr(y_f32.buf.as_ptr() as *const c_void);
    kb.push_i32(m as i32);
    kb.push_i32(k as i32);
    kb.push_i32(n as i32);
    kb.pad_to(16);

    let grid_m = ((m + 15) / 16) as u32;
    let grid_n = ((n + 15) / 16) as u32;
    gpu.launch_kernel_blob(
        KERNEL_NAME,
        [grid_m, grid_n, 1],
        [32, 1, 1],
        0,
        kb.as_mut_slice(),
    )
    .unwrap();
}

// Inline f32→f16 bit conversion (rdna-compute can't depend on hipfire-runtime).
fn f32_to_f16_bits(x: f32) -> u16 {
    let bits = x.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp_f32 = ((bits >> 23) & 0xff) as i32;
    let mant = bits & 0x7fffff;
    if exp_f32 == 0 {
        return sign;
    }
    if exp_f32 == 0xff {
        return sign | 0x7c00 | if mant != 0 { 1 } else { 0 };
    }
    let exp = exp_f32 - 127 + 15;
    if exp <= 0 {
        return sign;
    }
    if exp >= 31 {
        return sign | 0x7c00;
    }
    sign | ((exp as u16) << 10) | ((mant >> 13) as u16)
}
