// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 hipfire contributors
// hipfire — see LICENSE and NOTICE in the project root.
//! Head-to-head decode-GEMV microbench: SPLIT layout (gemv_oq4_grouped:
//! [nibbles | distant scales], 2 memory streams) vs INTERLEAVED layout
//! (gemv_oq4_interleaved: [f32 scale][128 nibbles] contiguous per group, 1
//! stream). B=1, decode shapes. Measures whether the interleaved access pattern
//! reads faster — the de-risk for an interleaved per-arch loader repack.
//!
//!   cargo run --release -p rdna-compute --example bench_oq4_layout

use rdna_compute::{DType, Gpu};
use std::time::Instant;

fn lcg(seed: u32, n: usize) -> Vec<u8> {
    let mut s = seed.max(1);
    (0..n).map(|_| { s = s.wrapping_mul(1_103_515_245).wrapping_add(12345) & 0x7fff_ffff; (s >> 13) as u8 }).collect()
}
fn lcgf(seed: u32, n: usize) -> Vec<f32> {
    let mut s = seed.max(1);
    (0..n).map(|_| { s = s.wrapping_mul(1_103_515_245).wrapping_add(12345) & 0x7fff_ffff; -1.0 + (s as f32/2_147_483_648.0)*2.0 }).collect()
}

fn main() {
    let k = 1024usize;
    let group = 256usize;
    let ng = k / group;
    let iters = 300;
    let mut gpu = Gpu::init().unwrap();
    let x: Vec<f32> = lcgf(3, k);
    let xd = gpu.upload_f32(&x, &[1, k]).unwrap();

    println!("decode-GEMV layout microbench (K={k}, B=1, {iters} iters/measure) on {}", gpu.arch);
    println!("{:>8}  {:>12}  {:>12}  {:>8}  {:>8}", "M", "split us", "interleaved us", "speedup", "GB/s il");
    for &m in &[1024usize, 4864, 6144] {
        let nib = lcg(1, m * (k / 2));
        let sc: Vec<f32> = lcgf(0x11, m * ng).iter().map(|v| 0.01 + v.abs() * 0.2).collect();
        // split: [nibbles m*k/2 | scales m*ng*4]
        let mut split = nib.clone();
        for s in &sc { split.extend_from_slice(&s.to_le_bytes()); }
        let wd = gpu.upload_raw(&split, &[split.len()]).unwrap();
        let ws = wd.sub_offset(m * (k / 2), m * ng * 4);
        // interleaved: per row, per group [scale 4B][128 nibbles]; stride ng*132
        let gstride = 4 + group / 2;
        let mut il = vec![0u8; m * ng * gstride];
        for r in 0..m {
            for g in 0..ng {
                let off = (r * ng + g) * gstride;
                il[off..off + 4].copy_from_slice(&sc[r * ng + g].to_le_bytes());
                let nsrc = r * (k / 2) + g * (group / 2);
                il[off + 4..off + 4 + group / 2].copy_from_slice(&nib[nsrc..nsrc + group / 2]);
            }
        }
        let wid = gpu.upload_raw(&il, &[il.len()]).unwrap();
        let ys = gpu.alloc_tensor(&[m], DType::F32).unwrap();
        let yi = gpu.alloc_tensor(&[m], DType::F32).unwrap();

        // correctness: both should match
        gpu.gemv_oq4_grouped(&wd, &ws, &xd, &ys, m, k, group).unwrap();
        gpu.gemv_oq4_interleaved(&wid, &xd, &yi, m, k, group).unwrap();
        gpu.device_synchronize().unwrap();
        let a = gpu.download_f32(&ys).unwrap();
        let b = gpu.download_f32(&yi).unwrap();
        let mut md = 0f32; for i in 0..m { md = md.max((a[i]-b[i]).abs()); }

        // warm
        for _ in 0..30 { gpu.gemv_oq4_grouped(&wd, &ws, &xd, &ys, m, k, group).unwrap(); }
        gpu.device_synchronize().unwrap();
        let t = Instant::now();
        for _ in 0..iters { gpu.gemv_oq4_grouped(&wd, &ws, &xd, &ys, m, k, group).unwrap(); }
        gpu.device_synchronize().unwrap();
        let split_us = t.elapsed().as_secs_f64() * 1e6 / iters as f64;

        for _ in 0..30 { gpu.gemv_oq4_interleaved(&wid, &xd, &yi, m, k, group).unwrap(); }
        gpu.device_synchronize().unwrap();
        let t = Instant::now();
        for _ in 0..iters { gpu.gemv_oq4_interleaved(&wid, &xd, &yi, m, k, group).unwrap(); }
        gpu.device_synchronize().unwrap();
        let il_us = t.elapsed().as_secs_f64() * 1e6 / iters as f64;

        let wbytes = (m * (k / 2) + m * ng * 4) as f64; // weight + scales
        let gbps = wbytes / (il_us * 1e-6) / 1e9;
        println!("{:>8}  {:>12.2}  {:>12.2}  {:>7.2}x  {:>7.1}  (max_diff {:.4})",
                 m, split_us, il_us, split_us / il_us, gbps, md);
    }
}
