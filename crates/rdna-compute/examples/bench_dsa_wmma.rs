// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Nick Woolmer
// hipfire — see LICENSE and NOTICE in the project root.

//! DSA-attention WMMA head-batching microbench.
//!
//! De-risks the DSA prefill optimization: does a head-batched f16-WMMA
//! flash-attention (shared K/V across heads, score+output as WMMA GEMMs)
//! match the f32 per-head reference in precision, and is it faster?
//!
//!   reference : CPU f32 (ground truth)
//!   baseline  : dsa_attn_f32_baseline (GPU, one block per (head,batch))
//!   wmma      : dsa_attn_wmma_hb       (GPU, one warp per (16-head,batch))
//!
//! Usage: cargo run --release --example bench_dsa_wmma

use hip_bridge::KernargBlob;
use rdna_compute::{DType, Gpu};
use std::ffi::c_void;

const SRC: &str = include_str!("../../../kernels/src/bench_dsa_wmma.hip");

fn f32_to_f16_bits(x: f32) -> u16 {
    let bits = x.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp_f32 = ((bits >> 23) & 0xff) as i32;
    let mant = bits & 0x7fffff;
    if exp_f32 == 0 { return sign; }
    if exp_f32 == 0xff { return sign | 0x7c00 | if mant != 0 { 1 } else { 0 }; }
    let exp = exp_f32 - 127 + 15;
    if exp <= 0 { return sign; }
    if exp >= 31 { return sign | 0x7c00; }
    sign | ((exp as u16) << 10) | ((mant >> 13) as u16)
}

fn to_f16_bytes(v: &[f32]) -> Vec<u8> {
    let bits: Vec<u16> = v.iter().map(|&x| f32_to_f16_bits(x)).collect();
    let mut out = vec![0u8; bits.len() * 2];
    for (i, &b) in bits.iter().enumerate() {
        out[2 * i] = (b & 0xff) as u8;
        out[2 * i + 1] = (b >> 8) as u8;
    }
    out
}

fn main() {
    let mut gpu = Gpu::init().expect("gpu init");
    eprintln!("=== DSA WMMA head-batching microbench ===");
    eprintln!("  arch = {}", gpu.arch);

    let b_n: usize = 256; // batch
    let h: usize = 64;    // heads
    let d: usize = 512;   // head_dim
    let n: usize = 512;   // n_total keys
    eprintln!("  shape: B={b_n} H={h} D={d} N={n}  (K/V shared across heads per batch)");

    gpu.ensure_kernel_public("dsa_attn_f32_baseline", SRC, "dsa_attn_f32_baseline")
        .expect("ensure baseline");
    gpu.ensure_kernel_public("dsa_attn_wmma_hb", SRC, "dsa_attn_wmma_hb")
        .expect("ensure wmma");

    // ── Synthetic data (uniform [-1,1]); scores ~ N(0, ~0.33) after scaling. ──
    let mut seed: u32 = 0x1234_5678;
    let mut rng = || {
        seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
        ((seed >> 8) as f32 / 16_777_216.0) * 2.0 - 1.0
    };

    let q: Vec<f32> = (0..b_n * h * d).map(|_| rng()).collect();
    let k: Vec<f32> = (0..b_n * n * d).map(|_| rng()).collect(); // [B,N,D]
    let v: Vec<f32> = (0..b_n * n * d).map(|_| rng()).collect(); // [B,N,D]

    // Vt = V transposed per batch: [B, D, N], Vt[b,d,p] = V[b,p,d]
    let mut vt = vec![0f32; b_n * d * n];
    for bb in 0..b_n {
        for p in 0..n {
            for dd in 0..d {
                vt[(bb * d + dd) * n + p] = v[(bb * n + p) * d + dd];
            }
        }
    }

    // Upload f32 (baseline) + f16 (wmma).
    let d_q = gpu.upload_f32(&q, &[b_n * h * d]).unwrap();
    let d_k = gpu.upload_f32(&k, &[b_n * n * d]).unwrap();
    let d_v = gpu.upload_f32(&v, &[b_n * n * d]).unwrap();
    let d_qf16 = gpu.upload_raw(&to_f16_bytes(&q), &[b_n * h * d * 2]).unwrap();
    let d_kf16 = gpu.upload_raw(&to_f16_bytes(&k), &[b_n * n * d * 2]).unwrap();
    let d_vtf16 = gpu.upload_raw(&to_f16_bytes(&vt), &[b_n * d * n * 2]).unwrap();
    let d_o_base = gpu.zeros(&[b_n * h * d], DType::F32).unwrap();
    let d_o_wmma = gpu.zeros(&[b_n * h * d], DType::F32).unwrap();

    let inv_scale = 1.0f32 / (d as f32).sqrt();

    // ── CPU reference (f32) for a representative sample (first 2 batches). ──
    let sample_b = 2usize;
    let mut o_ref = vec![0f32; sample_b * h * d];
    for bb in 0..sample_b {
        for hh in 0..h {
            let qoff = (bb * h + hh) * d;
            let mut scores = vec![0f32; n];
            let mut mx = f32::NEG_INFINITY;
            for p in 0..n {
                let koff = (bb * n + p) * d;
                let mut s = 0f32;
                for i in 0..d { s += q[qoff + i] * k[koff + i]; }
                s *= inv_scale;
                scores[p] = s;
                if s > mx { mx = s; }
            }
            let mut sum = 0f32;
            for p in 0..n { scores[p] = (scores[p] - mx).exp(); sum += scores[p]; }
            let inv = 1.0 / sum;
            for dd in 0..d {
                let mut acc = 0f32;
                for p in 0..n { acc += scores[p] * inv * v[(bb * n + p) * d + dd]; }
                o_ref[(bb * h + hh) * d + dd] = acc;
            }
        }
    }

    // ── Launchers ──
    let launch_base = |gpu: &Gpu| {
        let mut kb = KernargBlob::new();
        kb.push_ptr(d_q.buf.as_ptr() as *const c_void);
        kb.push_ptr(d_k.buf.as_ptr() as *const c_void);
        kb.push_ptr(d_v.buf.as_ptr() as *const c_void);
        kb.push_ptr(d_o_base.buf.as_ptr() as *const c_void);
        kb.push_i32(h as i32);
        kb.push_i32(d as i32);
        kb.push_i32(n as i32);
        kb.push_i32(b_n as i32);
        kb.pad_to(16);
        let lds = (d + n) * 4;
        gpu.launch_kernel_blob("dsa_attn_f32_baseline",
            [h as u32, b_n as u32, 1], [512, 1, 1], lds as u32, kb.as_mut_slice()).unwrap();
    };
    let launch_wmma = |gpu: &Gpu| {
        let mut kb = KernargBlob::new();
        kb.push_ptr(d_qf16.buf.as_ptr() as *const c_void);
        kb.push_ptr(d_kf16.buf.as_ptr() as *const c_void);
        kb.push_ptr(d_vtf16.buf.as_ptr() as *const c_void);
        kb.push_ptr(d_o_wmma.buf.as_ptr() as *const c_void);
        kb.push_i32(h as i32);
        kb.push_i32(d as i32);
        kb.push_i32(n as i32);
        kb.push_i32(b_n as i32);
        kb.pad_to(16);
        let lds = 16 * n * 4; // S f32 (exp-scores); P normalized inline
        gpu.launch_kernel_blob("dsa_attn_wmma_hb",
            [(h / 16) as u32, b_n as u32, 1], [32, 1, 1], lds as u32, kb.as_mut_slice()).unwrap();
    };

    // ── Run once for correctness ──
    launch_base(&gpu);
    launch_wmma(&gpu);
    gpu.hip.device_synchronize().unwrap();
    let o_base = gpu.download_f32(&d_o_base).unwrap();
    let o_wmma = gpu.download_f32(&d_o_wmma).unwrap();

    let cmp = |name: &str, got: &[f32]| {
        let mut max_abs = 0f32;
        let mut max_rel = 0f32;
        let mut sum_rel = 0f64;
        let mut cnt = 0usize;
        let refmax = o_ref.iter().map(|x| x.abs()).fold(0f32, f32::max);
        let thr = refmax * 0.01;
        for i in 0..sample_b * h * d {
            let r = o_ref[i];
            let g = got[i];
            let dd = (r - g).abs();
            if dd > max_abs { max_abs = dd; }
            if r.abs() > thr {
                let rel = dd / r.abs();
                if rel > max_rel { max_rel = rel; }
                sum_rel += rel as f64;
                cnt += 1;
            }
        }
        eprintln!(
            "  {name:18} vs CPU-f32:  max|err|={max_abs:.4e}  max_rel={max_rel:.4e}  \
             mean_rel={:.4e}  (|ref|max={refmax:.3}, gated {cnt})",
            sum_rel / cnt.max(1) as f64
        );
    };
    eprintln!("\n── correctness (sample: first {sample_b} batches × {h} heads) ──");
    cmp("f32 baseline", &o_base);
    cmp("wmma head-batch", &o_wmma);

    // ── Timing ──
    const WARM: usize = 10;
    const IT: usize = 100;
    for _ in 0..WARM { launch_base(&gpu); launch_wmma(&gpu); }
    gpu.hip.device_synchronize().unwrap();

    let time = |gpu: &Gpu, f: &dyn Fn(&Gpu)| -> f64 {
        let e0 = gpu.hip.event_create().unwrap();
        let e1 = gpu.hip.event_create().unwrap();
        gpu.hip.event_record(&e0, None).unwrap();
        for _ in 0..IT { f(gpu); }
        gpu.hip.event_record(&e1, None).unwrap();
        gpu.hip.event_synchronize(&e1).unwrap();
        gpu.hip.event_elapsed_ms(&e0, &e1).unwrap() as f64 * 1000.0 / IT as f64
    };
    let base_us = time(&gpu, &launch_base);
    let wmma_us = time(&gpu, &launch_wmma);
    eprintln!("\n── timing (full grid B={b_n} H={h}) ──");
    eprintln!("  f32 baseline:    {base_us:8.1} µs/call");
    eprintln!("  wmma head-batch: {wmma_us:8.1} µs/call");
    eprintln!("  SPEEDUP: ×{:.2}", base_us / wmma_us);
}
