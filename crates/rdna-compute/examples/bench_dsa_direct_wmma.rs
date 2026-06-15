// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Nick Woolmer
// hipfire — see LICENSE and NOTICE in the project root.

//! Validates the production DSA-attention WMMA kernel
//! (`deepseek4_attn_swa_topk_direct_wmma`) against the f32 reference
//! (`deepseek4_attn_swa_topk_direct_batched_f32`) on the REAL buffer layouts
//! (K=V tied swa_kv [B,D,win] + kv_cache [n_comp,D] via topk_idx + sink +
//! per-batch variable n_valid/n_active). Correctness-first; also times both.
//!
//! cargo run --release --example bench_dsa_direct_wmma

use rdna_compute::{DType, Gpu};

fn u2f(x: u32) -> f32 { ((x >> 8) as f32 / 16_777_216.0) * 2.0 - 1.0 }

fn main() {
    let mut gpu = Gpu::init().expect("gpu init");
    eprintln!("=== DSA direct WMMA vs f32 reference ===  arch={}", gpu.arch);

    let b_n = std::env::args().nth(1).and_then(|s| s.parse().ok()).unwrap_or(256usize); // batch
    let h = 64usize;        // heads
    let d = 512usize;       // head_dim
    let swa_window = 128usize;
    let topk_window = 512usize;
    let n_comp = 1024usize;
    eprintln!("  B={b_n} H={h} D={d} swa_window={swa_window} topk_window={topk_window} n_comp={n_comp}");

    let mut seed: u32 = 0xC0FFEE11;
    let mut nxt = || { seed = seed.wrapping_mul(1664525).wrapping_add(1013904223); seed };

    let q: Vec<f32> = (0..b_n * h * d).map(|_| u2f(nxt())).collect();
    let swa_kv: Vec<f32> = (0..b_n * d * swa_window).map(|_| u2f(nxt())).collect(); // [B,D,win]
    let kv_cache: Vec<f32> = (0..n_comp * d).map(|_| u2f(nxt())).collect();         // [n_comp,D]
    let sink: Vec<f32> = (0..h).map(|_| u2f(nxt()) * 0.5).collect();

    // per-batch n_valid (≤win) and n_active (≤topk_window), varied.
    let mut n_valid = vec![0i32; b_n];
    let mut n_active = vec![0i32; b_n];
    for bb in 0..b_n {
        n_valid[bb] = (32 + ((nxt() >> 8) as usize % (swa_window - 32 + 1))) as i32; // 32..win
        n_active[bb] = (16 + ((nxt() >> 8) as usize % (topk_window - 16 + 1))) as i32; // 16..topk_window
    }
    let max_n_total = (0..b_n).map(|i| n_valid[i] + n_active[i]).max().unwrap();
    eprintln!("  max_n_total = {max_n_total}  (n_valid {:?}, n_active {:?})", n_valid, n_active);

    // topk_idx [B, topk_window]: valid random in [0,n_comp) for the active range;
    // sprinkle a few -1 (invalid) to exercise that path.
    let mut topk_idx = vec![0i32; b_n * topk_window];
    for bb in 0..b_n {
        for t in 0..topk_window {
            let r = (nxt() >> 8) as usize;
            topk_idx[bb * topk_window + t] =
                if r % 37 == 0 { -1 } else { (r % n_comp) as i32 };
        }
    }

    let i32_bytes = |v: &[i32]| -> Vec<u8> {
        let mut o = vec![0u8; v.len() * 4];
        for (i, &x) in v.iter().enumerate() { o[i*4..i*4+4].copy_from_slice(&x.to_le_bytes()); }
        o
    };

    let d_q = gpu.upload_f32(&q, &[b_n * h * d]).unwrap();
    let d_swa = gpu.upload_f32(&swa_kv, &[b_n * d * swa_window]).unwrap();
    let d_kv = gpu.upload_f32(&kv_cache, &[n_comp * d]).unwrap();
    let d_sink = gpu.upload_f32(&sink, &[h]).unwrap();
    let d_tk = gpu.upload_raw(&i32_bytes(&topk_idx), &[b_n * topk_window * 4]).unwrap();
    let d_nv = gpu.upload_raw(&i32_bytes(&n_valid), &[b_n * 4]).unwrap();
    let d_na = gpu.upload_raw(&i32_bytes(&n_active), &[b_n * 4]).unwrap();
    let d_ref = gpu.zeros(&[b_n * h * d], DType::F32).unwrap();
    let d_wmma = gpu.zeros(&[b_n * h * d], DType::F32).unwrap();

    // reference (f32 production kernel; swa_k=swa_v=swa_kv, K=V tied)
    gpu.deepseek4_attn_swa_topk_direct_batched_f32(
        &d_q, &d_swa, &d_swa, &d_kv, &d_tk, &d_sink, &d_nv, &d_na, &d_ref,
        h as i32, d as i32, swa_window as i32, topk_window as i32, n_comp as i32, b_n as i32,
    ).unwrap();
    // wmma
    gpu.deepseek4_attn_swa_topk_direct_wmma(
        &d_q, &d_swa, &d_kv, &d_tk, &d_sink, &d_nv, &d_na, &d_wmma,
        h as i32, d as i32, swa_window as i32, topk_window as i32, n_comp as i32, b_n as i32,
        max_n_total,
    ).unwrap();
    gpu.hip.device_synchronize().unwrap();

    let o_ref = gpu.download_f32(&d_ref).unwrap();
    let o_wmma = gpu.download_f32(&d_wmma).unwrap();

    let refmax = o_ref.iter().map(|x| x.abs()).fold(0f32, f32::max);
    let thr = refmax * 0.01;
    let (mut maxabs, mut maxrel, mut sumrel, mut cnt, mut nbad) = (0f32, 0f32, 0f64, 0usize, 0usize);
    for i in 0..o_ref.len() {
        let r = o_ref[i]; let g = o_wmma[i];
        if !g.is_finite() { nbad += 1; continue; }
        let dd = (r - g).abs();
        if dd > maxabs { maxabs = dd; }
        if r.abs() > thr {
            let rel = dd / r.abs();
            if rel > maxrel { maxrel = rel; }
            sumrel += rel as f64; cnt += 1;
        }
    }
    eprintln!("\n  vs f32 ref:  max|err|={maxabs:.4e}  max_rel={maxrel:.4e}  mean_rel={:.4e}  \
               nonfinite={nbad}  (|ref|max={refmax:.3}, gated {cnt})",
              sumrel / cnt.max(1) as f64);

    // timing
    let it = 100;
    for _ in 0..10 {
        gpu.deepseek4_attn_swa_topk_direct_batched_f32(&d_q,&d_swa,&d_swa,&d_kv,&d_tk,&d_sink,&d_nv,&d_na,&d_ref,h as i32,d as i32,swa_window as i32,topk_window as i32,n_comp as i32,b_n as i32).unwrap();
        gpu.deepseek4_attn_swa_topk_direct_wmma(&d_q,&d_swa,&d_kv,&d_tk,&d_sink,&d_nv,&d_na,&d_wmma,h as i32,d as i32,swa_window as i32,topk_window as i32,n_comp as i32,b_n as i32,max_n_total).unwrap();
    }
    gpu.hip.device_synchronize().unwrap();
    let e0=gpu.hip.event_create().unwrap(); let e1=gpu.hip.event_create().unwrap();
    gpu.hip.event_record(&e0,None).unwrap();
    for _ in 0..it { gpu.deepseek4_attn_swa_topk_direct_batched_f32(&d_q,&d_swa,&d_swa,&d_kv,&d_tk,&d_sink,&d_nv,&d_na,&d_ref,h as i32,d as i32,swa_window as i32,topk_window as i32,n_comp as i32,b_n as i32).unwrap(); }
    gpu.hip.event_record(&e1,None).unwrap(); gpu.hip.event_synchronize(&e1).unwrap();
    let ref_us = gpu.hip.event_elapsed_ms(&e0,&e1).unwrap() as f64 *1000.0/it as f64;
    let e2=gpu.hip.event_create().unwrap(); let e3=gpu.hip.event_create().unwrap();
    gpu.hip.event_record(&e2,None).unwrap();
    for _ in 0..it { gpu.deepseek4_attn_swa_topk_direct_wmma(&d_q,&d_swa,&d_kv,&d_tk,&d_sink,&d_nv,&d_na,&d_wmma,h as i32,d as i32,swa_window as i32,topk_window as i32,n_comp as i32,b_n as i32,max_n_total).unwrap(); }
    gpu.hip.event_record(&e3,None).unwrap(); gpu.hip.event_synchronize(&e3).unwrap();
    let wmma_us = gpu.hip.event_elapsed_ms(&e2,&e3).unwrap() as f64 *1000.0/it as f64;
    eprintln!("  timing: f32 ref {ref_us:.1} µs/call   wmma {wmma_us:.1} µs/call   ×{:.2}", ref_us/wmma_us);
}
