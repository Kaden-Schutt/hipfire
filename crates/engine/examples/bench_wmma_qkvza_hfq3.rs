//! Microbench for `gemm_qkvza_hfq3g256_wmma` — measures the new
//! kernel's throughput on the Qwen3.5-9B DeltaNet LA preamble shape
//! and compares to the equivalent per-row `gemv_hfq3g256` (the path
//! the eligibility check currently falls back to for MQ3 prefill).
//!
//! Qwen3.5-9B DeltaNet projection dimensions:
//!   dim          = 4096   (hidden)
//!   linear_num_value_heads * head_dim = 32 * 128 = 4096
//!   wqkv: [num_key_heads * key_head_dim * 2 + num_value_heads * value_head_dim, dim]
//!         = [32*128*2 + 32*128, 4096] = [8192 + 4096, 4096] = [12288, 4096]
//!         (kv_proj_qkv split into Q/K/V — actual qkv_m for our kernel
//!         is the combined 12288 in the published config; we use 8192
//!         as the standard "qkv" dim and 4096 for z below.)
//!   wz: [num_value_heads * value_head_dim, dim] = [4096, 4096]
//!   w_alpha / w_beta: [num_value_heads, dim] = [32, 4096]
//!
//! Practical bench shape (round to multiples of 16 where needed):
//!   qkv_m = 8192, z_m = 4096, beta_m = 32, alpha_m = 32, K = 4096
//!   N = 128 (typical prefill batch)

use rdna_compute::{DType, Gpu};
use std::time::Instant;

fn main() {
    let mut gpu = Gpu::init().expect("GPU init");
    eprintln!("GPU: {}", gpu.arch);

    // Skip on archs without RDNA3 wave32 WMMA.
    if !matches!(gpu.arch.as_str(), "gfx1100" | "gfx1101" | "gfx1102") {
        eprintln!("SKIP: gfx11 wave32 WMMA only — current arch {}", gpu.arch);
        std::process::exit(0);
    }

    // Realistic Qwen3.5-9B DeltaNet LA preamble shape.
    // beta/alpha are 32 in the real model; round to 16-multiples for the
    // kernel (which routes per-thread but requires total_m % 16 == 0).
    let qkv_m = 8192usize;
    let z_m = 4096usize;
    let beta_m = 32usize;
    let alpha_m = 32usize;
    let k = 4096usize;

    for &n in &[1usize, 8, 32, 64, 128, 256] {
        eprintln!();
        eprintln!("=== batch N={n} (qkv={qkv_m} z={z_m} b={beta_m} a={alpha_m} K={k}) ===");
        run_bench(&mut gpu, qkv_m, z_m, beta_m, alpha_m, k, n);
    }
}

fn run_bench(
    gpu: &mut Gpu,
    qkv_m: usize, z_m: usize, beta_m: usize, alpha_m: usize,
    k: usize, n: usize,
) {
    // Allocate inputs. Random fp16 X, deterministic HFQ3 weights.
    let aq = upload_random_hfq3(gpu, qkv_m, k, 0xA1);
    let az = upload_random_hfq3(gpu, z_m, k, 0xB2);
    let ab = upload_random_hfq3(gpu, beta_m, k, 0xC3);
    let aa = upload_random_hfq3(gpu, alpha_m, k, 0xD4);

    let x_f32: Vec<f32> = (0..(n * k))
        .map(|i| ((i * 13 + 7) as f32 % 31.0 - 15.0) * 0.05)
        .collect();
    let x = gpu.upload_f32(&x_f32, &[n, k]).unwrap();

    let yq = gpu.alloc_tensor(&[n, qkv_m], DType::F32).unwrap();
    let yz = gpu.alloc_tensor(&[n, z_m], DType::F32).unwrap();
    let yb = gpu.alloc_tensor(&[n, beta_m], DType::F32).unwrap();
    let ya = gpu.alloc_tensor(&[n, alpha_m], DType::F32).unwrap();

    // Warmup x3
    for _ in 0..3 {
        gpu.gemm_qkvza_hfq3g256_wmma(
            &aq, &az, &ab, &aa, &x,
            &yq, &yz, &yb, &ya,
            qkv_m, z_m, beta_m, alpha_m, k, n,
        ).unwrap();
    }
    gpu.hip.device_synchronize().unwrap();

    // Time the new WMMA kernel.
    let runs = 32;
    let t0 = Instant::now();
    for _ in 0..runs {
        gpu.gemm_qkvza_hfq3g256_wmma(
            &aq, &az, &ab, &aa, &x,
            &yq, &yz, &yb, &ya,
            qkv_m, z_m, beta_m, alpha_m, k, n,
        ).unwrap();
    }
    gpu.hip.device_synchronize().unwrap();
    let wmma_us = t0.elapsed().as_micros() as f64 / runs as f64;

    // Time the per-row GEMV fallback (4 separate calls, one per projection).
    // gemv_hfq3g256 is single-row GEMV; for batch N we need n×4 calls per
    // projection. Use the simplest equivalent: call gemv_hfq3g256 per (batch,
    // projection). For a fair comparison, we measure ALL the work (n × 4
    // separate GEMVs) the dispatch-fallback path would do.
    let yqg = gpu.alloc_tensor(&[n, qkv_m], DType::F32).unwrap();
    let yzg = gpu.alloc_tensor(&[n, z_m], DType::F32).unwrap();
    let ybg = gpu.alloc_tensor(&[n, beta_m], DType::F32).unwrap();
    let yag = gpu.alloc_tensor(&[n, alpha_m], DType::F32).unwrap();

    // Warmup
    for _ in 0..2 {
        do_gemv_fallback(gpu, &aq, &az, &ab, &aa, &x, &yqg, &yzg, &ybg, &yag,
                        qkv_m, z_m, beta_m, alpha_m, k, n);
    }
    gpu.hip.device_synchronize().unwrap();

    let t0 = Instant::now();
    for _ in 0..runs {
        do_gemv_fallback(gpu, &aq, &az, &ab, &aa, &x, &yqg, &yzg, &ybg, &yag,
                        qkv_m, z_m, beta_m, alpha_m, k, n);
    }
    gpu.hip.device_synchronize().unwrap();
    let gemv_us = t0.elapsed().as_micros() as f64 / runs as f64;

    let total_m = qkv_m + z_m + beta_m + alpha_m;
    let groups_per_row = k / 256;
    let weight_bytes = total_m * groups_per_row * 104;  // HFQ3 storage
    let xbytes = n * k * 2;  // fp16 X
    let ybytes = n * total_m * 4;  // fp32 Y
    let bytes_total = weight_bytes + xbytes + ybytes;

    let wmma_gibs = bytes_total as f64 / 1024.0 / 1024.0 / 1024.0 / (wmma_us * 1e-6);
    let gemv_gibs = bytes_total as f64 / 1024.0 / 1024.0 / 1024.0 / (gemv_us * 1e-6);
    let speedup = gemv_us / wmma_us;

    eprintln!("  WMMA new : {:8.1} µs  /call  →  {:6.1} GiB/s effective", wmma_us, wmma_gibs);
    eprintln!("  GEMV old : {:8.1} µs  /call  →  {:6.1} GiB/s effective", gemv_us, gemv_gibs);
    eprintln!("  speedup  : {:6.2}×  (WMMA vs per-row GEMV fallback)", speedup);
}

fn do_gemv_fallback(
    gpu: &mut Gpu,
    aq: &rdna_compute::GpuTensor, az: &rdna_compute::GpuTensor,
    ab: &rdna_compute::GpuTensor, aa: &rdna_compute::GpuTensor,
    x: &rdna_compute::GpuTensor,
    yq: &rdna_compute::GpuTensor, yz: &rdna_compute::GpuTensor,
    yb: &rdna_compute::GpuTensor, ya: &rdna_compute::GpuTensor,
    qkv_m: usize, z_m: usize, beta_m: usize, alpha_m: usize,
    k: usize, n: usize,
) {
    // Per-row GEMV: each batch row is a separate call. This mirrors
    // what `forward_prefill_batch` falls back to when the eligibility
    // check excludes MQ3.
    for b in 0..n {
        let x_row = x.sub_offset(b * k, k);
        let yq_row = yq.sub_offset(b * qkv_m, qkv_m);
        let yz_row = yz.sub_offset(b * z_m, z_m);
        let yb_row = yb.sub_offset(b * beta_m, beta_m);
        let ya_row = ya.sub_offset(b * alpha_m, alpha_m);
        gpu.gemv_hfq3g256(aq, &x_row, &yq_row, qkv_m, k).unwrap();
        gpu.gemv_hfq3g256(az, &x_row, &yz_row, z_m, k).unwrap();
        gpu.gemv_hfq3g256(ab, &x_row, &yb_row, beta_m, k).unwrap();
        gpu.gemv_hfq3g256(aa, &x_row, &ya_row, alpha_m, k).unwrap();
    }
}

fn upload_random_hfq3(gpu: &mut Gpu, m: usize, k: usize, seed: u8) -> rdna_compute::GpuTensor {
    let groups_per_row = k / 256;
    let bytes_per_row = groups_per_row * 104;
    let mut out = vec![0u8; m * bytes_per_row];
    let mix = |x: u64| {
        let h = x.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((h ^ (h >> 33)).wrapping_mul(0xff51afd7ed558ccd)) ^ (h >> 28)
    };
    let s0 = seed as u64;
    for row in 0..m {
        for g in 0..groups_per_row {
            let off = row * bytes_per_row + g * 104;
            let r1 = mix(s0 ^ ((row as u64) << 16) ^ (g as u64));
            let r2 = mix(s0 ^ ((row as u64) * 7 + g as u64));
            let scale = 0.01 + (((r1 as u32) % 4001) as f32) * 1e-5;
            let zero = (((r2 as u32) % 1500) as f32) * 1e-4 - 0.075;
            out[off..off + 4].copy_from_slice(&scale.to_le_bytes());
            out[off + 4..off + 8].copy_from_slice(&zero.to_le_bytes());
            for byte_i in 0..96 {
                let r = mix(s0 ^ ((row as u64) << 24) ^ ((g as u64) << 12) ^ (byte_i as u64));
                out[off + 8 + byte_i] = (r & 0xff) as u8;
            }
        }
    }
    gpu.upload_raw(&out, &[m, k]).unwrap()
}
