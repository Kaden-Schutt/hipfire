// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Production-layout microbench for the gfx1201 gathered DSA WMMA kernel.
//!
//! Compares `deepseek4_attn_swa_topk_batched_wmma_gfx12` with the established
//! F32 kernel on the exact `[B,H,D]`, `[B,D,window]`, K=V-tied layouts. H=24
//! exercises TP3's masked final eight-head group; H=16 exercises rank 2.
//!
//! Usage:
//! `cargo run --release -p rdna-compute --example bench_dsa_gathered_wmma_gfx12 -- [B] [H] [ITERS]`

use rdna_compute::{DType, Gpu};

fn u2f(x: u32) -> f32 {
    ((x >> 8) as f32 / 16_777_216.0) * 2.0 - 1.0
}

fn i32_bytes(values: &[i32]) -> Vec<u8> {
    let mut out = vec![0_u8; values.len() * 4];
    for (index, value) in values.iter().enumerate() {
        out[index * 4..index * 4 + 4].copy_from_slice(&value.to_le_bytes());
    }
    out
}

fn main() {
    let mut gpu = Gpu::init().expect("gpu init");
    assert!(
        gpu.arch_caps.is_gfx1201(),
        "requires gfx1201, got {}",
        gpu.arch
    );

    let mut args = std::env::args().skip(1);
    let batch = args
        .next()
        .and_then(|v| v.parse().ok())
        .unwrap_or(128_usize);
    let heads = args.next().and_then(|v| v.parse().ok()).unwrap_or(24_usize);
    let iters = args.next().and_then(|v| v.parse().ok()).unwrap_or(40_usize);
    let head_dim = 512_usize;
    let swa_window = 128_usize;
    let topk_window = 512_usize;

    eprintln!(
        "=== gfx1201 gathered DSA WMMA === arch={} B={batch} H={heads} D={head_dim} SWA={swa_window} topK={topk_window}",
        gpu.arch
    );

    let mut seed = 0xc0ff_ee11_u32;
    let mut next = || {
        seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        seed
    };
    let q: Vec<f32> = (0..batch * heads * head_dim).map(|_| u2f(next())).collect();
    let swa: Vec<f32> = (0..batch * head_dim * swa_window)
        .map(|_| u2f(next()))
        .collect();
    let topk: Vec<f32> = (0..batch * head_dim * topk_window)
        .map(|_| u2f(next()))
        .collect();
    let sink: Vec<f32> = (0..heads).map(|_| u2f(next()) * 0.5).collect();

    let mut n_valid = vec![0_i32; batch];
    let mut n_active = vec![0_i32; batch];
    for row in 0..batch {
        // Cover the full production range while preserving a 640-row maximum.
        n_valid[row] = (32 + ((next() >> 8) as usize % 97)) as i32;
        n_active[row] = (16 + ((next() >> 8) as usize % 497)) as i32;
    }
    let max_n_total = n_valid
        .iter()
        .zip(&n_active)
        .map(|(a, b)| a + b)
        .max()
        .unwrap_or(0);
    eprintln!("max_n_total={max_n_total}");

    let d_q = gpu.upload_f32(&q, &[batch * heads * head_dim]).unwrap();
    let d_swa = gpu
        .upload_f32(&swa, &[batch * head_dim * swa_window])
        .unwrap();
    let d_topk = gpu
        .upload_f32(&topk, &[batch * head_dim * topk_window])
        .unwrap();
    let d_sink = gpu.upload_f32(&sink, &[heads]).unwrap();
    let d_nv = gpu.upload_raw(&i32_bytes(&n_valid), &[batch * 4]).unwrap();
    let d_na = gpu.upload_raw(&i32_bytes(&n_active), &[batch * 4]).unwrap();
    let d_ref = gpu.zeros(&[batch * heads * head_dim], DType::F32).unwrap();
    let d_wmma = gpu.zeros(&[batch * heads * head_dim], DType::F32).unwrap();

    let launch_ref = |gpu: &mut Gpu| {
        gpu.deepseek4_attn_swa_topk_batched_f32(
            &d_q,
            &d_swa,
            &d_swa,
            &d_topk,
            &d_topk,
            &d_sink,
            &d_nv,
            &d_na,
            &d_ref,
            heads as i32,
            head_dim as i32,
            swa_window as i32,
            topk_window as i32,
            batch as i32,
        )
        .unwrap();
    };
    let launch_wmma = |gpu: &mut Gpu| {
        gpu.deepseek4_attn_swa_topk_batched_wmma_gfx12(
            &d_q,
            &d_swa,
            &d_topk,
            &d_sink,
            &d_nv,
            &d_na,
            &d_wmma,
            heads as i32,
            head_dim as i32,
            swa_window as i32,
            topk_window as i32,
            batch as i32,
            max_n_total,
        )
        .unwrap();
    };

    launch_ref(&mut gpu);
    launch_wmma(&mut gpu);
    gpu.hip.device_synchronize().unwrap();

    let reference = gpu.download_f32(&d_ref).unwrap();
    let candidate = gpu.download_f32(&d_wmma).unwrap();
    let ref_max = reference.iter().map(|v| v.abs()).fold(0.0_f32, f32::max);
    let rel_gate = ref_max * 0.01;
    let mut max_abs = 0.0_f32;
    let mut max_rel = 0.0_f32;
    let mut sum_rel = 0.0_f64;
    let mut rel_count = 0_usize;
    let mut nonfinite = 0_usize;
    for (r, c) in reference.iter().zip(&candidate) {
        if !c.is_finite() {
            nonfinite += 1;
            continue;
        }
        let delta = (r - c).abs();
        max_abs = max_abs.max(delta);
        if r.abs() > rel_gate {
            let rel = delta / r.abs();
            max_rel = max_rel.max(rel);
            sum_rel += rel as f64;
            rel_count += 1;
        }
    }
    eprintln!(
        "correctness: max_abs={max_abs:.6e} max_rel={max_rel:.6e} mean_rel={:.6e} nonfinite={nonfinite}",
        sum_rel / rel_count.max(1) as f64
    );

    for _ in 0..5 {
        launch_ref(&mut gpu);
        launch_wmma(&mut gpu);
    }
    gpu.hip.device_synchronize().unwrap();

    let time = |gpu: &mut Gpu, launch: &dyn Fn(&mut Gpu)| -> f64 {
        let start = gpu.hip.event_create().unwrap();
        let stop = gpu.hip.event_create().unwrap();
        gpu.hip.event_record(&start, None).unwrap();
        for _ in 0..iters {
            launch(gpu);
        }
        gpu.hip.event_record(&stop, None).unwrap();
        gpu.hip.event_synchronize(&stop).unwrap();
        gpu.hip.event_elapsed_ms(&start, &stop).unwrap() as f64 * 1_000.0 / iters as f64
    };
    let ref_us = time(&mut gpu, &launch_ref);
    let wmma_us = time(&mut gpu, &launch_wmma);
    eprintln!(
        "timing: f32={ref_us:.3} us/call gfx12_wmma={wmma_us:.3} us/call speedup={:.3}x",
        ref_us / wmma_us
    );
}
