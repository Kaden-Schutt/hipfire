// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

use rdna_compute::{DType, Gpu};
use std::time::Instant;

const SEED: u64 = 0xC0DEFACEu64;
const GROUP: usize = 256;
const ROW_HDR: usize = 8;
const ROW_PAYLOAD: usize = 128;
const ROW_BYTES: usize = ROW_HDR + ROW_PAYLOAD;

fn synth_hfq4_weights(m: usize, groups_per_row: usize) -> Vec<u8> {
    let total = m * groups_per_row * ROW_BYTES;
    let mut out = vec![0u8; total];
    let mut state = SEED;
    let mut next = || {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (state >> 33) as u32
    };
    let scale = 1e-3_f32.to_le_bytes();
    let zp = (-0.5_f32).to_le_bytes();
    for row in 0..m {
        for g in 0..groups_per_row {
            let gp = (row * groups_per_row + g) * ROW_BYTES;
            out[gp..gp + 4].copy_from_slice(&scale);
            out[gp + 4..gp + 8].copy_from_slice(&zp);
            for i in 0..ROW_PAYLOAD {
                out[gp + ROW_HDR + i] = (next() & 0xFF) as u8;
            }
        }
    }
    out
}

fn bytes_of(v: &[f32]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 4) }
}

fn swap_active(gpu: &mut Gpu, next: hip_bridge::Stream) -> Option<hip_bridge::Stream> {
    gpu.active_stream.replace(next)
}

fn main() {
    let m: usize = std::env::var("BENCH_M").ok().and_then(|s| s.parse().ok()).unwrap_or(5120);
    let k: usize = std::env::var("BENCH_K").ok().and_then(|s| s.parse().ok()).unwrap_or(5120);
    let batch: usize = std::env::var("BENCH_BATCH").ok().and_then(|s| s.parse().ok()).unwrap_or(16);
    let warmup_secs: f64 = std::env::var("HIPFIRE_DPM_WARMUP_SECS").ok().and_then(|s| s.parse().ok()).unwrap_or(0.0);
    assert!(k % GROUP == 0, "K must be a multiple of 256");
    let gpr = k / GROUP;

    eprintln!("=== bench_stream_overlap: gemm_hfq4g256_residual M={m} K={k} N={batch} ===");
    eprintln!("weight tensor: {:.2} MiB", (m * gpr * ROW_BYTES) as f64 / (1024.0 * 1024.0));

    let mut gpu = Gpu::init().expect("gpu init");

    let weight_bytes = synth_hfq4_weights(m, gpr);
    let a_raw = gpu.upload_raw(&weight_bytes, &[m * gpr * ROW_BYTES]).expect("upload weights");
    let x_host: Vec<f32> = (0..batch * k).map(|i| ((i as f32) * 1e-4) % 1.0 - 0.5).collect();
    let y_init_host: Vec<f32> = (0..batch * m).map(|i| ((i as f32) * 7e-5) % 0.5 - 0.25).collect();

    let x_a = gpu.upload_f32(&x_host, &[batch * k]).expect("x_a");
    let x_b = gpu.upload_f32(&x_host, &[batch * k]).expect("x_b");
    let y_a = gpu.alloc_tensor(&[batch * m], DType::F32).expect("y_a");
    let y_b = gpu.alloc_tensor(&[batch * m], DType::F32).expect("y_b");
    gpu.hip.memcpy_htod(&y_a.buf, bytes_of(&y_init_host)).unwrap();
    gpu.hip.memcpy_htod(&y_b.buf, bytes_of(&y_init_host)).unwrap();
    gpu.hip.device_synchronize().unwrap();

    if warmup_secs > 0.0 {
        eprintln!("[dpm] pinning DPM for {warmup_secs:.1}s...");
        let pad = gpu.alloc_tensor(&[64 * 1024 * 1024], DType::F32).unwrap();
        let pad_bytes = 64 * 1024 * 1024 * 4;
        let t = Instant::now();
        while t.elapsed().as_secs_f64() < warmup_secs {
            for _ in 0..16 {
                gpu.hip.memset(&pad.buf, 0, pad_bytes).unwrap();
            }
        }
        gpu.hip.device_synchronize().unwrap();
        let _ = gpu.free_tensor(pad);
    }

    let stream_warm = gpu.hip.stream_create().expect("stream_warm");
    swap_active(&mut gpu, stream_warm);
    for _ in 0..20 {
        gpu.gemm_hfq4g256_residual(&a_raw, &x_a, &y_a, m, k, batch).unwrap();
    }
    gpu.hip.device_synchronize().unwrap();

    let n_total_list: [usize; 4] = [8, 16, 32, 64];
    eprintln!("\n  N   T_serial(µs)  T_parallel(µs)  overlap_ratio");
    eprintln!("  --  ------------  --------------  -------------");

    for &n_total in &n_total_list {
        gpu.hip.memcpy_htod(&y_a.buf, bytes_of(&y_init_host)).unwrap();
        gpu.hip.memcpy_htod(&y_b.buf, bytes_of(&y_init_host)).unwrap();
        gpu.hip.device_synchronize().unwrap();

        let serial_stream = gpu.hip.stream_create().unwrap();
        let prev = swap_active(&mut gpu, serial_stream);
        let t = Instant::now();
        for _ in 0..n_total {
            gpu.gemm_hfq4g256_residual(&a_raw, &x_a, &y_a, m, k, batch).unwrap();
        }
        gpu.hip.device_synchronize().unwrap();
        let t_serial = t.elapsed().as_secs_f64() * 1e6;
        drop(gpu.active_stream.take());
        gpu.active_stream = prev;

        gpu.hip.memcpy_htod(&y_a.buf, bytes_of(&y_init_host)).unwrap();
        gpu.hip.memcpy_htod(&y_b.buf, bytes_of(&y_init_host)).unwrap();
        gpu.hip.device_synchronize().unwrap();

        let half = n_total / 2;
        let sa = gpu.hip.stream_create().unwrap();
        let sb = gpu.hip.stream_create().unwrap();
        let prev = swap_active(&mut gpu, sa);
        let t = Instant::now();
        for _ in 0..half {
            gpu.gemm_hfq4g256_residual(&a_raw, &x_a, &y_a, m, k, batch).unwrap();
        }
        let sa_back = swap_active(&mut gpu, sb).unwrap();
        for _ in 0..(n_total - half) {
            gpu.gemm_hfq4g256_residual(&a_raw, &x_b, &y_b, m, k, batch).unwrap();
        }
        gpu.hip.device_synchronize().unwrap();
        let t_parallel = t.elapsed().as_secs_f64() * 1e6;
        drop(gpu.active_stream.take());
        drop(sa_back);
        gpu.active_stream = prev;

        let ratio = t_serial / t_parallel;
        eprintln!("  {n_total:2}   {t_serial:11.1}   {t_parallel:13.1}    {ratio:6.3}x");
    }

    eprintln!("\nInterpretation:");
    eprintln!("  ratio ≈ 1.0  → no overlap (compute pipe saturated by one stream)");
    eprintln!("  ratio ≈ 1.5  → partial ACE concurrency, A-full modest win");
    eprintln!("  ratio ≈ 2.0  → full ACE concurrency, A-full full projected win");

    eprintln!("\n=== asymmetric probe: LARGE verify + SMALL draft concurrent ===");
    eprintln!("(real A-full: 64-layer verify on stream_A + 5-layer draft on stream_B)");
    let draft_m = std::env::var("BENCH_DRAFT_M").ok().and_then(|s| s.parse().ok()).unwrap_or(5120usize);
    let draft_k = std::env::var("BENCH_DRAFT_K").ok().and_then(|s| s.parse().ok()).unwrap_or(5120usize);
    let draft_n = std::env::var("BENCH_DRAFT_N").ok().and_then(|s| s.parse().ok()).unwrap_or(16usize);
    let draft_gpr = draft_k / GROUP;
    let draft_weights = synth_hfq4_weights(draft_m, draft_gpr);
    let a_draft = gpu.upload_raw(&draft_weights, &[draft_m * draft_gpr * ROW_BYTES]).expect("a_draft");
    let x_draft_host: Vec<f32> = (0..draft_n * draft_k).map(|i| ((i as f32) * 1e-4) % 1.0 - 0.5).collect();
    let y_draft_init: Vec<f32> = (0..draft_n * draft_m).map(|i| ((i as f32) * 7e-5) % 0.5 - 0.25).collect();
    let x_draft = gpu.upload_f32(&x_draft_host, &[draft_n * draft_k]).unwrap();
    let y_draft = gpu.alloc_tensor(&[draft_n * draft_m], DType::F32).unwrap();
    gpu.hip.memcpy_htod(&y_draft.buf, bytes_of(&y_draft_init)).unwrap();
    gpu.hip.device_synchronize().unwrap();

    let verify_n = 16;
    let draft_layers: usize = std::env::var("BENCH_DRAFT_LAYERS").ok().and_then(|s| s.parse().ok()).unwrap_or(5usize);
    let verify_layers: usize = std::env::var("BENCH_VERIFY_LAYERS").ok().and_then(|s| s.parse().ok()).unwrap_or(5usize);
    eprintln!("\n  verify: {verify_layers} layers of gemm(M=5120 K=5120 N={verify_n})");
    eprintln!("  draft:  {draft_layers} layers of gemm(M={draft_m} K={draft_k} N={draft_n})");

    gpu.hip.memcpy_htod(&y_a.buf, bytes_of(&y_init_host)).unwrap();
    gpu.hip.memcpy_htod(&y_draft.buf, bytes_of(&y_draft_init)).unwrap();
    gpu.hip.device_synchronize().unwrap();

    let sv = gpu.hip.stream_create().unwrap();
    let prev = swap_active(&mut gpu, sv);
    let t = Instant::now();
    for _ in 0..verify_layers {
        gpu.gemm_hfq4g256_residual(&a_raw, &x_a, &y_a, m, k, verify_n).unwrap();
    }
    gpu.hip.device_synchronize().unwrap();
    let t_verify_alone = t.elapsed().as_secs_f64() * 1e6;
    drop(gpu.active_stream.take());

    let sd = gpu.hip.stream_create().unwrap();
    gpu.active_stream = Some(sd);
    let t = Instant::now();
    for _ in 0..draft_layers {
        gpu.gemm_hfq4g256_residual(&a_draft, &x_draft, &y_draft, draft_m, draft_k, draft_n).unwrap();
    }
    gpu.hip.device_synchronize().unwrap();
    let t_draft_alone = t.elapsed().as_secs_f64() * 1e6;
    drop(gpu.active_stream.take());

    let sv = gpu.hip.stream_create().unwrap();
    let sd = gpu.hip.stream_create().unwrap();
    gpu.active_stream = Some(sv);
    let t = Instant::now();
    for _ in 0..verify_layers {
        gpu.gemm_hfq4g256_residual(&a_raw, &x_a, &y_a, m, k, verify_n).unwrap();
    }
    let sv_back = swap_active(&mut gpu, sd).unwrap();
    for _ in 0..draft_layers {
        gpu.gemm_hfq4g256_residual(&a_draft, &x_draft, &y_draft, draft_m, draft_k, draft_n).unwrap();
    }
    gpu.hip.device_synchronize().unwrap();
    let t_both = t.elapsed().as_secs_f64() * 1e6;
    drop(gpu.active_stream.take());
    drop(sv_back);
    gpu.active_stream = prev;

    let serial_sum = t_verify_alone + t_draft_alone;
    let overlap_gain = serial_sum / t_both;
    eprintln!("  t_verify_alone: {t_verify_alone:.1} µs");
    eprintln!("  t_draft_alone:  {t_draft_alone:.1} µs");
    eprintln!("  t_both (2 streams): {t_both:.1} µs");
    eprintln!("  asymm_overlap_ratio = (verify+draft)/both = {overlap_gain:.3}x");
    eprintln!();
    eprintln!("Gate for A-full (task #93):");
    eprintln!("  asymm_overlap_ratio ≥ 1.5  → proceed with full A-full build");
    eprintln!("  1.3 ≤ ratio < 1.5         → proceed cautiously, expect <half projected gain");
    eprintln!("  ratio < 1.3               → A-full doomed on gfx1100, pivot to #74/#75 kernel grinds");
}
