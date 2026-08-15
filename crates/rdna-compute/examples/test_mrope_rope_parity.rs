// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! With t == h == w, 3D mrope degenerates to 1D RoPE. This example asserts
//! the new mrope kernels reproduce rope_partial_halfsplit_f32 EXACTLY in
//! that case — which is also why gating mrope on image presence is safe.
//!
//! Run: cargo run --release -p rdna-compute --example test_mrope_rope_parity --features deltanet

use rdna_compute::Gpu;

fn lcg(seed: u32, n: usize) -> Vec<f32> {
    let mut s = seed;
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(1_103_515_245).wrapping_add(12_345);
            ((s >> 16) & 0x7fff) as f32 / 32_768.0 - 0.5
        })
        .collect()
}

fn main() {
    let mut gpu = Gpu::init().expect("GPU init");
    let (nhq, nhk, hd, n_rot) = (16usize, 8usize, 256usize, 64usize);
    let freq_base = 1_000_000.0f32;
    let section = [11usize, 11, 10];
    let pos = 137i32;

    let qd = lcg(0xa1, nhq * hd);
    let kd = lcg(0xb2, nhk * hd);

    // Reference: existing 1D kernel.
    let q1 = gpu.upload_f32(&qd, &[nhq * hd]).unwrap();
    let k1 = gpu.upload_f32(&kd, &[nhk * hd]).unwrap();
    let p1 = gpu.hip.malloc(4).unwrap();
    gpu.hip.memcpy_htod(&p1, &pos.to_le_bytes()).unwrap();
    gpu.rope_partial_interleaved_f32(&q1, &k1, &p1, nhq, nhk, hd, n_rot, freq_base)
        .unwrap();

    // Candidate: mrope with t == h == w.
    let q2 = gpu.upload_f32(&qd, &[nhq * hd]).unwrap();
    let k2 = gpu.upload_f32(&kd, &[nhk * hd]).unwrap();
    let p3: Vec<u8> = [pos, pos, pos].iter().flat_map(|v| v.to_le_bytes()).collect();
    let p2 = gpu.hip.malloc(12).unwrap();
    gpu.hip.memcpy_htod(&p2, &p3).unwrap();
    gpu.rope_mrope_halfsplit_f32(&q2, &k2, &p2, nhq, nhk, hd, n_rot, freq_base, section)
        .unwrap();
    gpu.hip.device_synchronize().unwrap();

    let (a, b) = (gpu.download_f32(&q1).unwrap(), gpu.download_f32(&q2).unwrap());
    let (c, d) = (gpu.download_f32(&k1).unwrap(), gpu.download_f32(&k2).unwrap());
    let dq = a.iter().zip(&b).map(|(x, y)| (x - y).abs()).fold(0.0f32, f32::max);
    let dk = c.iter().zip(&d).map(|(x, y)| (x - y).abs()).fold(0.0f32, f32::max);
    println!("max|dq| = {dq:.3e}   max|dk| = {dk:.3e}");
    assert!(dq == 0.0 && dk == 0.0, "mrope with t==h==w must be BIT-IDENTICAL to 1D rope");
    println!("PASS: mrope degenerates exactly to 1D RoPE");
}
