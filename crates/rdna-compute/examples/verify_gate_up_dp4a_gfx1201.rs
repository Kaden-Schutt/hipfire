// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! One-shot correctness probe for the gfx1201 dp4a Phase-B spike
//! (`gemv_hfq4g256_moe_gate_up_k8_indexed_dp4a_gfx1201`). Builds ONE
//! deterministic synthetic HFQ4G256 expert at the real A3B gate_up decode
//! shape (M=1024, K=2048), calls the SAME public dispatch function
//! (`gemv_hfq4g256_moe_gate_up_k8_indexed`) the scalar path uses, and
//! prints y_gate/y_up to stdout as plain text.
//!
//! The dp4a-vs-scalar branch is chosen once at `Gpu::init()` time (cached
//! `FeatureFlags`), so this binary must be run TWICE — once with
//! `HIPFIRE_GFX1201_DP4A` unset (scalar/baseline) and once with
//! `HIPFIRE_GFX1201_DP4A=1` (dp4a) — and the two stdout captures diffed
//! externally (max abs / relative error over all M values).
//!
//! Run:
//!   cargo run --release -p rdna-compute --example verify_gate_up_dp4a_gfx1201 > baseline.txt
//!   HIPFIRE_GFX1201_DP4A=1 cargo run --release -p rdna-compute --example verify_gate_up_dp4a_gfx1201 > dp4a.txt

use rdna_compute::{DType, Gpu, GpuTensor};

const MI: usize = 512;
const M: usize = 2 * MI; // 1024 — kernel splits gate vs up at M/2
const K: usize = 2048; // hidden, A3B gate_up decode shape
const K_TOP: usize = 1; // single rank is enough for a correctness probe

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

/// Deterministic HFQ4-G256 expert: groups = K/256 groups of 136 B
/// ([f32 scale][f32 zp][32×u32 nibbles]). Scale/zp chosen in a realistic
/// range; nibbles cover the full [0,15] domain via the PRNG.
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
            let zp: f32 = -0.02 + (rng() & 0x3F) as f32 * 1e-4;
            out[off..off + 4].copy_from_slice(&sc.to_bits().to_le_bytes());
            out[off + 4..off + 8].copy_from_slice(&zp.to_bits().to_le_bytes());
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
        "arch={}  M={M} K={K}  HIPFIRE_GFX1201_DP4A={:?}",
        gpu.arch, dp4a_flag
    );

    let expert_bytes = synth_hfq4g256(M, K, 0x4D51);
    let expert = upload_u8(&mut gpu, &expert_bytes);
    let expert_ptrs = upload_u64(&mut gpu, &[expert.buf.as_ptr() as u64]);
    let topk = upload_i32(&mut gpu, &vec![0i32; K_TOP]);
    let x = upload_f32(&mut gpu, &make_x(K, 0xABCD));
    let y_gate = alloc_f32_zeros(&mut gpu, K_TOP * MI);
    let y_up = alloc_f32_zeros(&mut gpu, K_TOP * MI);

    gpu.gemv_hfq4g256_moe_gate_up_k8_indexed(
        &expert_ptrs,
        &topk,
        &x,
        &y_gate,
        &y_up,
        M,
        K,
        K_TOP,
    )
    .expect("gemv_hfq4g256_moe_gate_up_k8_indexed");
    gpu.hip.device_synchronize().expect("sync");

    let mut gate_out = vec![0u8; K_TOP * MI * 4];
    gpu.hip
        .memcpy_dtoh(&mut gate_out, &y_gate.buf)
        .expect("dtoh gate");
    let mut up_out = vec![0u8; K_TOP * MI * 4];
    gpu.hip
        .memcpy_dtoh(&mut up_out, &y_up.buf)
        .expect("dtoh up");

    let gate_f: Vec<f32> = gate_out
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    let up_f: Vec<f32> = up_out
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();

    let mut nan_or_inf = 0usize;
    for v in gate_f.iter().chain(up_f.iter()) {
        if !v.is_finite() {
            nan_or_inf += 1;
        }
    }
    eprintln!(
        "gate: n={} min={:.6} max={:.6}  up: n={} min={:.6} max={:.6}  nan_or_inf={}",
        gate_f.len(),
        gate_f.iter().cloned().fold(f32::INFINITY, f32::min),
        gate_f.iter().cloned().fold(f32::NEG_INFINITY, f32::max),
        up_f.len(),
        up_f.iter().cloned().fold(f32::INFINITY, f32::min),
        up_f.iter().cloned().fold(f32::NEG_INFINITY, f32::max),
        nan_or_inf,
    );

    for v in &gate_f {
        println!("{v:.6}");
    }
    for v in &up_f {
        println!("{v:.6}");
    }
}
