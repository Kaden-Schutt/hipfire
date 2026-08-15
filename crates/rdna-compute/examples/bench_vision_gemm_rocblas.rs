// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! rocBLAS vs the production `gemm_f16` at Qwen3.5-VL / OvisOCR2 vision-tower
//! projection shapes (n=7600 patches, h=768, 12 blocks).
//!
//! Motivation: on gfx1201 the hand-rolled `gemm_f16_x_f16_wmma` fails to
//! compile (`__builtin_amdgcn_wmma_f32_16x16x16_f16_w32` is a gfx11 intrinsic
//! needing `wmma-256b-insts`), so the vision tower has no WMMA path there.
//! rocBLAS does load on gfx1201 but `try_init_rocblas` is CDNA3-gated.
//!
//! Usage:
//!   cargo run --release -p rdna-compute --example bench_vision_gemm_rocblas

use rdna_compute::{DType, Gpu, GpuTensor};
use std::time::Instant;

const WARMUP: usize = 5;
const TRIALS: usize = 30;
const B: usize = 7600;

fn main() {
    let mut gpu = Gpu::init().expect("gpu init");
    println!("Arch: {}", gpu.arch);
    if gpu.rocblas.is_none() {
        match hip_bridge::Rocblas::load() {
            Ok(rb) => {
                println!("[bench] forced rocBLAS load on {} (try_init_rocblas is CDNA3-gated)", gpu.arch);
                gpu.rocblas = Some(rb);
            }
            Err(e) => {
                println!("[bench] rocBLAS unavailable: {e}");
                std::process::exit(1);
            }
        }
    }

    // (M, K, per-block?) at B = 7600. h=768, mlp 4h=3072, patch_dim=1536.
    let shapes: &[(usize, usize, bool, &str)] = &[
        (768, 1536, false, "patch_embed  1536->768"),
        (2304, 768, true, "qkv           768->2304"),
        (768, 768, true, "attn_out      768->768"),
        (3072, 768, true, "fc1           768->3072"),
        (768, 3072, true, "fc2          3072->768"),
    ];

    println!("\n{:26} {:>9} {:>11} {:>10} {:>11} {:>9}",
             "shape (B=7600)", "roc us", "roc GFLOP/s", "cur us", "cur GFLOP/s", "speedup");
    let (mut roc_block_us, mut cur_block_us) = (0.0f64, 0.0f64);

    for &(m, k, per_block, label) in shapes {
        let w = gpu.hip.malloc(m * k * 2).expect("malloc W");
        let x = gpu.hip.malloc(B * k * 2).expect("malloc X");
        let y = gpu.hip.malloc(B * m * 4).expect("malloc Y");
        gpu.hip.memcpy_htod(&w, &vec![0x3Cu8; m * k * 2]).expect("copy W");
        gpu.hip.memcpy_htod(&x, &vec![0x34u8; B * k * 2]).expect("copy X");

        for _ in 0..WARMUP {
            gpu.rocblas_gemm_hfq4_prefill(&w, &x, &y, m, B, k).expect("rocblas warmup");
        }
        gpu.hip.device_synchronize().expect("sync");
        let t0 = Instant::now();
        for _ in 0..TRIALS {
            gpu.rocblas_gemm_hfq4_prefill(&w, &x, &y, m, B, k).expect("rocblas trial");
        }
        gpu.hip.device_synchronize().expect("sync");
        let roc_us = t0.elapsed().as_secs_f64() / TRIALS as f64 * 1e6;

        // Production path: gemm_f16 (F16 weights x F32 activations -> F32).
        let wt = GpuTensor {
            buf: unsafe { hip_bridge::DeviceBuffer::from_raw(w.as_ptr(), m * k * 2) },
            shape: vec![m, k],
            dtype: DType::F16,
        };
        let xt = gpu.zeros(&[B * k], DType::F32).expect("x f32");
        let yt = gpu.zeros(&[m * B], DType::F32).expect("y f32");
        for _ in 0..WARMUP {
            gpu.gemm_f16(&wt, &xt, &yt, m, k, B).expect("gemm_f16 warmup");
        }
        gpu.hip.device_synchronize().expect("sync");
        let t1 = Instant::now();
        for _ in 0..TRIALS {
            gpu.gemm_f16(&wt, &xt, &yt, m, k, B).expect("gemm_f16 trial");
        }
        gpu.hip.device_synchronize().expect("sync");
        let cur_us = t1.elapsed().as_secs_f64() / TRIALS as f64 * 1e6;

        let flops = 2.0 * m as f64 * k as f64 * B as f64;
        println!("{label:26} {roc_us:9.1} {:11.1} {cur_us:10.1} {:11.1} {:8.1}x",
                 flops / roc_us / 1e3, flops / cur_us / 1e3, cur_us / roc_us);
        if per_block {
            roc_block_us += roc_us;
            cur_block_us += cur_us;
        }

        std::mem::forget(wt);
        gpu.free_tensor(xt).ok();
        gpu.free_tensor(yt).ok();
        gpu.hip.free(w).ok();
        gpu.hip.free(x).ok();
        gpu.hip.free(y).ok();
    }

    println!("\nPer-block projections (qkv+attn_out+fc1+fc2), x12 blocks:");
    println!("  rocBLAS      : {:.3} s", roc_block_us * 12.0 / 1e6);
    println!("  current path : {:.3} s", cur_block_us * 12.0 / 1e6);
    println!("  saving       : {:.3} s", (cur_block_us - roc_block_us) * 12.0 / 1e6);
    println!("\nMeasured full vision tower on the OvisOCR2 fixture: 8.22 s.");
    println!("Attention (2*n^2*h*2 = 177 GFLOP/block, ~62% of tower FLOPs) is a separate kernel.");
}
