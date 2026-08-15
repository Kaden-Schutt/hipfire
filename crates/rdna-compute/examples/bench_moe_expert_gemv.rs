// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! What could a GEMV-shaped MoE expert kernel achieve?
//!
//! The multi-slot decode path runs routed experts through
//! `gemm_hfq4g256_moe_grouped_mmq_gfx1151`, a 16x16 i8-MMQ tile GEMM built for
//! prefill. At decode each tile carries 1-4 real rows against 16, and the
//! measured throughput is 136 GB/s against the 214 GB/s this box streams on the
//! lm_head (same HFQ4-G256 layout).
//!
//! Before writing a replacement kernel, this measures the ceiling of the
//! alternative shape using an already-validated kernel: `gemv_hfq4g256_xbatch`,
//! one weight pass over B activation vectors. It is run per expert here, which
//! is NOT how a real implementation would launch (one launch per expert per
//! projection would be ~2000 launches/step); the point is the achievable
//! bandwidth of the access pattern, with per-launch overhead reported
//! separately so it can be discounted.
//!
//! Shapes are the real ones for qwen3.6-35b-a3b: gate_up M=1024 K=2048,
//! down M=2048 K=512, 25.7 live experts per layer at 4 slots, 40 layers.

use rdna_compute::{DType, Gpu};

const LIVE_EXPERTS: usize = 26; // measured mean 25.7 at 4 slots
const N_LAYERS: usize = 40;
const GATE_UP_M: usize = 1024; // 2 * moe_intermediate (512)
const GATE_UP_K: usize = 2048; // dim
const DOWN_M: usize = 2048; // dim
const DOWN_K: usize = 512; // moe_intermediate

fn build_experts(gpu: &mut Gpu, m: usize, k: usize, n: usize) -> Vec<rdna_compute::GpuTensor> {
    let groups = k / 256;
    let row_bytes = groups * 136;
    let total = m * row_bytes;
    let mut w = vec![0u8; total];
    let mut seed: u32 = 0x9e37_79b9;
    let mut next = move || {
        seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        seed
    };
    for r in 0..m {
        for g in 0..groups {
            let off = r * row_bytes + g * 136;
            let sc = 0.002 + ((next() >> 9) % 500) as f32 * 1e-5;
            let zp = -0.003 + ((next() >> 9) % 400) as f32 * 1e-5;
            w[off..off + 4].copy_from_slice(&sc.to_le_bytes());
            w[off + 4..off + 8].copy_from_slice(&zp.to_le_bytes());
            for i in 0..128 {
                w[off + 8 + i] = (next() >> 11) as u8;
            }
        }
    }
    (0..n)
        .map(|_| gpu.upload_raw(&w, &[total]).unwrap())
        .collect()
}

fn main() {
    let mut gpu = Gpu::init().unwrap();
    let per_expert_mb =
        (GATE_UP_M * (GATE_UP_K / 256) * 136 + DOWN_M * (DOWN_K / 256) * 136) as f64 / 1e6;
    eprintln!("=== MoE expert GEMV ceiling probe ===");
    eprintln!(
        "  gate_up {GATE_UP_M}x{GATE_UP_K}, down {DOWN_M}x{DOWN_K}, {per_expert_mb:.3} MB/expert"
    );
    eprintln!("  allocating {LIVE_EXPERTS} experts (one layer's live set)...");

    let gate_up = build_experts(&mut gpu, GATE_UP_M, GATE_UP_K, LIVE_EXPERTS);
    let down = build_experts(&mut gpu, DOWN_M, DOWN_K, LIVE_EXPERTS);

    let max_b = 4usize;
    let xg: Vec<f32> = (0..max_b * GATE_UP_K)
        .map(|i| 0.01 * ((i % 331) as f32 * 0.13).sin())
        .collect();
    let xd: Vec<f32> = (0..max_b * DOWN_K)
        .map(|i| 0.01 * ((i % 197) as f32 * 0.17).cos())
        .collect();
    let d_xg = gpu.upload_f32(&xg, &[max_b * GATE_UP_K]).unwrap();
    let d_xd = gpu.upload_f32(&xd, &[max_b * DOWN_K]).unwrap();
    let d_yg = gpu.zeros(&[max_b * GATE_UP_M], DType::F32).unwrap();
    let d_yd = gpu.zeros(&[max_b * DOWN_M], DType::F32).unwrap();

    eprintln!("\n  B = rows sharing one expert (1 = no slot overlap, 4 = all four slots agree)");
    eprintln!(
        "  {:<4}{:>12}{:>12}{:>14}{:>16}",
        "B", "per-layer", "GB/s", "40-layer est", "vs 136 GB/s now"
    );

    for b in 1..=max_b {
        // warmup
        for _ in 0..3 {
            for e in 0..LIVE_EXPERTS {
                gpu.gemv_hfq4g256_xbatch(&gate_up[e], &d_xg, &d_yg, GATE_UP_M, GATE_UP_K, b)
                    .unwrap();
                gpu.gemv_hfq4g256_xbatch(&down[e], &d_xd, &d_yd, DOWN_M, DOWN_K, b)
                    .unwrap();
            }
        }
        gpu.hip.device_synchronize().unwrap();

        let iters = 10;
        let t0 = std::time::Instant::now();
        for _ in 0..iters {
            for e in 0..LIVE_EXPERTS {
                gpu.gemv_hfq4g256_xbatch(&gate_up[e], &d_xg, &d_yg, GATE_UP_M, GATE_UP_K, b)
                    .unwrap();
                gpu.gemv_hfq4g256_xbatch(&down[e], &d_xd, &d_yd, DOWN_M, DOWN_K, b)
                    .unwrap();
            }
        }
        gpu.hip.device_synchronize().unwrap();
        let per_layer_ms = t0.elapsed().as_secs_f64() * 1e3 / iters as f64;

        let bytes = LIVE_EXPERTS as f64 * per_expert_mb * 1e6;
        let gbs = bytes / (per_layer_ms * 1e-3) / 1e9;
        let est_40 = per_layer_ms * N_LAYERS as f64;
        eprintln!(
            "  {b:<4}{per_layer_ms:>10.3} ms{gbs:>11.0}{est_40:>12.2} ms{:>15.2}x",
            12.631 / est_40
        );
    }

    eprintln!(
        "\n  Reference: current grouped MMQ does 40 layers in 12.631 ms at 136 GB/s effective."
    );
    eprintln!(
        "  Per-launch overhead is included above ({} launches per layer);",
        LIVE_EXPERTS * 2
    );
    eprintln!("  a real kernel would fuse these into 2 launches per layer.");
}
