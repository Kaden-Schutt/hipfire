// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 hipfire contributors
// hipfire — see LICENSE and NOTICE in the project root.
//! Decode-shape bandwidth bench for the indexed-MoE OQ4/OQ8 GEMV kernels at
//! MiniMax-M2 dims (gate_up [2*1536, 3072], down [3072, 1536], k_top=8). Builds
//! the k_top routed expert blobs, runs each kernel in a timed loop, and reports
//! GiB/s = analytical bytes / wall time.
//!
//!   cargo run --release -p rdna-compute --example bench_oq_moe [iters]

use rdna_compute::profile::{gemv_oq4g256_moe_bytes, gemv_oq8g256_moe_bytes};
use rdna_compute::Gpu;
use std::time::Instant;

fn rng(seed: u32, n: usize) -> Vec<u8> {
    let mut s = seed.max(1);
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(1_103_515_245).wrapping_add(12345) & 0x7fff_ffff;
            (s >> 13) as u8
        })
        .collect()
}

fn build_experts(
    gpu: &mut Gpu,
    k_top: usize,
    m: usize,
    k: usize,
    blk: usize,
) -> (Vec<rdna_compute::GpuTensor>, rdna_compute::GpuTensor) {
    let ng = k / 256;
    let bytes_per = m * ng * blk;
    let mut tensors = Vec::new();
    let mut ptrs: Vec<u64> = Vec::new();
    for e in 0..k_top {
        let blob = rng(1 + e as u32, bytes_per);
        let t = gpu.upload_raw(&blob, &[blob.len()]).unwrap();
        ptrs.push(t.buf.as_ptr() as u64);
        tensors.push(t);
    }
    let ptr_bytes: Vec<u8> = ptrs.iter().flat_map(|p| p.to_le_bytes()).collect();
    let ptr_tensor = gpu.upload_raw(&ptr_bytes, &[k_top]).unwrap();
    (tensors, ptr_tensor)
}

fn time_loop<F: FnMut(&mut Gpu)>(gpu: &mut Gpu, iters: usize, mut f: F) -> f64 {
    for _ in 0..16 {
        f(gpu);
    }
    gpu.device_synchronize().unwrap();
    let t0 = Instant::now();
    for _ in 0..iters {
        f(gpu);
    }
    gpu.device_synchronize().unwrap();
    t0.elapsed().as_secs_f64() / iters as f64
}

fn main() {
    let iters: usize = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(2000);
    let hidden = 3072usize;
    let inter = 1536usize;
    let k_top = 8usize;
    let gu_m = 2 * inter; // 3072
    let dn_m = hidden; // 3072

    let mut gpu = Gpu::init().unwrap();
    println!(
        "bench_oq_moe on {} — decode shape, k_top={k_top}, iters={iters}",
        gpu.arch
    );
    println!("  gate_up [{gu_m}x{hidden}] (12 groups), down [{dn_m}x{inter}] (6 groups)\n");

    let topk: Vec<i32> = (0..k_top as i32).collect();
    let topk_t = gpu
        .upload_raw(
            &topk
                .iter()
                .flat_map(|i| i.to_le_bytes())
                .collect::<Vec<u8>>(),
            &[k_top],
        )
        .unwrap();
    let x_gu = gpu
        .upload_raw(&vec![0u8; hidden * 4], &[1, hidden])
        .unwrap();
    let rot = gpu
        .upload_raw(&vec![0u8; k_top * inter * 4], &[k_top, inter])
        .unwrap();
    let y_gate = gpu
        .upload_raw(&vec![0u8; k_top * inter * 4], &[k_top, inter])
        .unwrap();
    let y_up = gpu
        .upload_raw(&vec![0u8; k_top * inter * 4], &[k_top, inter])
        .unwrap();
    let dn_out = gpu
        .upload_raw(&vec![0u8; k_top * dn_m * 4], &[k_top, dn_m])
        .unwrap();

    // ── HFQ4 / mq4 reference (136 B/group, the tuned baseline) ──────────────
    {
        let (_gu, gu_ptr) = build_experts(&mut gpu, k_top, gu_m, hidden, 136);
        let (_dn, dn_ptr) = build_experts(&mut gpu, k_top, dn_m, inter, 136);
        let dt = time_loop(&mut gpu, iters, |gpu| {
            gpu.gemv_hfq4g256_moe_gate_up_k8_indexed(
                &gu_ptr, &topk_t, &x_gu, &y_gate, &y_up, gu_m, hidden,
            )
            .unwrap();
        });
        let bytes = k_top * (gu_m * (hidden / 256) * 136 + hidden * 4 + gu_m * 4);
        println!(
            "  HFQ4 gate_up:{:7.2} us  {:6.1} GiB/s",
            dt * 1e6,
            bytes as f64 / dt / 1e9
        );
        let dt = time_loop(&mut gpu, iters, |gpu| {
            gpu.gemv_hfq4g256_moe_down_k8_indexed_batched_expanded(
                &dn_ptr, &topk_t, &rot, &dn_out, dn_m, inter, k_top, 1,
            )
            .unwrap();
        });
        let bytes = k_top * (dn_m * (inter / 256) * 136 + inter * 4 + dn_m * 4);
        println!(
            "  HFQ4 down:   {:7.2} us  {:6.1} GiB/s",
            dt * 1e6,
            bytes as f64 / dt / 1e9
        );
    }

    // ── OQ4 (132 B/group) ──────────────────────────────────────────────────
    {
        let (_gu, gu_ptr) = build_experts(&mut gpu, k_top, gu_m, hidden, 132);
        let (_dn, dn_ptr) = build_experts(&mut gpu, k_top, dn_m, inter, 132);
        let dt = time_loop(&mut gpu, iters, |gpu| {
            gpu.gemv_oq4g256_moe_gate_up_k8_indexed(
                &gu_ptr, &topk_t, &x_gu, &y_gate, &y_up, gu_m, hidden,
            )
            .unwrap();
        });
        let gibs = gemv_oq4g256_moe_bytes(gu_m, hidden, k_top) as f64 / dt / 1e9;
        println!("  OQ4 gate_up: {:7.2} us  {:6.1} GiB/s", dt * 1e6, gibs);
        let dt = time_loop(&mut gpu, iters, |gpu| {
            gpu.gemv_oq4g256_moe_down_k8_indexed_batched_expanded(
                &dn_ptr, &topk_t, &rot, &dn_out, dn_m, inter, k_top, 1,
            )
            .unwrap();
        });
        let gibs = gemv_oq4g256_moe_bytes(dn_m, inter, k_top) as f64 / dt / 1e9;
        println!("  OQ4 down:    {:7.2} us  {:6.1} GiB/s", dt * 1e6, gibs);
    }

    // ── OQ8 (260 B/group) ──────────────────────────────────────────────────
    {
        let (_gu, gu_ptr) = build_experts(&mut gpu, k_top, gu_m, hidden, 260);
        let (_dn, dn_ptr) = build_experts(&mut gpu, k_top, dn_m, inter, 260);
        let dt = time_loop(&mut gpu, iters, |gpu| {
            gpu.gemv_oq8g256_moe_gate_up_k8_indexed(
                &gu_ptr, &topk_t, &x_gu, &y_gate, &y_up, gu_m, hidden,
            )
            .unwrap();
        });
        let gibs = gemv_oq8g256_moe_bytes(gu_m, hidden, k_top) as f64 / dt / 1e9;
        println!("  OQ8 gate_up: {:7.2} us  {:6.1} GiB/s", dt * 1e6, gibs);
        let dt = time_loop(&mut gpu, iters, |gpu| {
            gpu.gemv_oq8g256_moe_down_k8_indexed_batched_expanded(
                &dn_ptr, &topk_t, &rot, &dn_out, dn_m, inter, k_top, 1,
            )
            .unwrap();
        });
        let gibs = gemv_oq8g256_moe_bytes(dn_m, inter, k_top) as f64 / dt / 1e9;
        println!("  OQ8 down:    {:7.2} us  {:6.1} GiB/s", dt * 1e6, gibs);
    }
}
