// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Throughput A/B for the gfx1201 dp4a Phase-B spike
//! (`gemv_hfq4g256_moe_gate_up_k8_indexed_dp4a_gfx1201`) at the real A3B
//! decode grid (M=1024, K=2048, K_TOP=8), COLD regime (256 experts, 32
//! disjoint top-8 sets cycled so the working set exceeds L3 — same shape
//! discipline as `bench_indexed_moe_keystone.rs`, which gates itself to
//! RDNA3 wave32 only and therefore SKIPS on gfx1201).
//!
//! The dp4a-vs-scalar branch is chosen once at `Gpu::init()` time, so run
//! this binary TWICE:
//!   cargo run --release -p rdna-compute --example bench_gate_up_dp4a_gfx1201
//!   HIPFIRE_GFX1201_DP4A=1 cargo run --release -p rdna-compute --example bench_gate_up_dp4a_gfx1201

use rdna_compute::{DType, Gpu, GpuTensor};
use std::time::Instant;

const MI: usize = 512;
const M: usize = 2 * MI;
const K: usize = 2048;
const N_EXP: usize = 256;
const K_TOP: usize = 8;

fn upload_u8(gpu: &mut Gpu, data: &[u8]) -> GpuTensor {
    let t = gpu
        .alloc_tensor(&[data.len()], DType::Raw)
        .expect("alloc u8");
    gpu.hip.memcpy_htod(&t.buf, data).expect("htod u8");
    t
}
fn upload_f32(gpu: &mut Gpu, data: &[f32]) -> GpuTensor {
    let bytes: &[u8] =
        unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 4) };
    let t = gpu
        .alloc_tensor(&[data.len()], DType::F32)
        .expect("alloc f32");
    gpu.hip.memcpy_htod(&t.buf, bytes).expect("htod f32");
    t
}
fn upload_i32(gpu: &mut Gpu, data: &[i32]) -> GpuTensor {
    let bytes: &[u8] =
        unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 4) };
    let t = gpu
        .alloc_tensor(&[data.len() * 4], DType::Raw)
        .expect("alloc i32");
    gpu.hip.memcpy_htod(&t.buf, bytes).expect("htod i32");
    t
}
fn upload_u64(gpu: &mut Gpu, data: &[u64]) -> GpuTensor {
    let bytes: &[u8] =
        unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 8) };
    let t = gpu
        .alloc_tensor(&[data.len() * 8], DType::Raw)
        .expect("alloc u64");
    gpu.hip.memcpy_htod(&t.buf, bytes).expect("htod u64");
    t
}
fn alloc_f32_zeros(gpu: &mut Gpu, n: usize) -> GpuTensor {
    let t = gpu.alloc_tensor(&[n], DType::F32).expect("alloc zeros");
    gpu.hip.memset(&t.buf, 0, n * 4).expect("memset");
    t
}

fn synth_hfq4g256(m: usize, k: usize, seed: u64) -> Vec<u8> {
    let groups = k / 256;
    let row_bytes = groups * 136;
    let mut out = vec![0u8; m * row_bytes];
    let mut st = seed;
    let mut rng = || -> u32 {
        st = st
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (st >> 33) as u32
    };
    for row in 0..m {
        for g in 0..groups {
            let off = row * row_bytes + g * 136;
            let sc: f32 = 0.003 + (rng() & 0x3F) as f32 * 1e-4;
            out[off..off + 4].copy_from_slice(&sc.to_bits().to_le_bytes());
            out[off + 4..off + 8].copy_from_slice(&(-0.02f32).to_bits().to_le_bytes());
            for w in 0..32 {
                let pk = rng();
                out[off + 8 + w * 4..off + 8 + w * 4 + 4].copy_from_slice(&pk.to_le_bytes());
            }
        }
    }
    out
}

fn make_x(n: usize, seed: u64) -> Vec<f32> {
    let mut st = seed;
    (0..n)
        .map(|_| {
            st = st
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((st >> 40) as f32 / (1u64 << 24) as f32) - 0.5
        })
        .collect()
}

fn main() {
    let mut gpu = Gpu::init().expect("Gpu::init");
    let dp4a_flag = std::env::var("HIPFIRE_GFX1201_DP4A").ok();
    eprintln!(
        "arch={}  M={M} K={K} n_exp={N_EXP} k_top={K_TOP}  HIPFIRE_GFX1201_DP4A={:?}",
        gpu.arch, dp4a_flag
    );

    let mut keep: Vec<GpuTensor> = Vec::new();
    let mut ptrs: Vec<u64> = Vec::with_capacity(N_EXP);
    for e in 0..N_EXP {
        let t = upload_u8(&mut gpu, &synth_hfq4g256(M, K, 0x4D51 ^ e as u64));
        ptrs.push(t.buf.as_ptr() as u64);
        keep.push(t);
    }
    let expert_ptrs = upload_u64(&mut gpu, &ptrs);

    let topk_sets: Vec<GpuTensor> = (0..N_EXP / K_TOP)
        .map(|s| {
            let idx: Vec<i32> = (0..K_TOP).map(|j| (s * K_TOP + j) as i32).collect();
            upload_i32(&mut gpu, &idx)
        })
        .collect();

    let x = upload_f32(&mut gpu, &make_x(K, 0xABCD));
    let y_gate = alloc_f32_zeros(&mut gpu, K_TOP * MI);
    let y_up = alloc_f32_zeros(&mut gpu, K_TOP * MI);

    let n: usize = std::env::var("HIPFIRE_BENCH_N")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(500);

    // Warmup (JIT + DPM ramp) — throwaway, not timed.
    for i in 0..50 {
        let tk = &topk_sets[i % topk_sets.len()];
        gpu.gemv_hfq4g256_moe_gate_up_k8_indexed(&expert_ptrs, tk, &x, &y_gate, &y_up, M, K, K_TOP)
            .expect("warmup");
    }
    gpu.hip.device_synchronize().expect("sync warmup");

    let start = Instant::now();
    for i in 0..n {
        let tk = &topk_sets[i % topk_sets.len()];
        gpu.gemv_hfq4g256_moe_gate_up_k8_indexed(&expert_ptrs, tk, &x, &y_gate, &y_up, M, K, K_TOP)
            .expect("launch");
    }
    gpu.hip.device_synchronize().expect("sync");
    let elapsed = start.elapsed();
    let us_per_launch = elapsed.as_secs_f64() * 1e6 / n as f64;

    let bytes_per_launch = (K_TOP * rdna_compute::profile::gemv_hfq4g256_bytes(M, K)) as f64;
    let gbs = bytes_per_launch / (us_per_launch * 1e-6) / 1e9;

    println!(
        "n={n} us_per_launch={us_per_launch:.3} bytes_per_launch={bytes_per_launch:.0} achieved_GB_s={gbs:.1}"
    );
}
