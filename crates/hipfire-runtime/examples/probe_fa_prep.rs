// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt

//! Numerical probe for the gfx1100 fused FA prep: runs the legacy sequence
//! (deinterleave + rmsnorm_batched Q/K + rope_partial_halfsplit) and the fused
//! `qwen35_fa_prep_gfx1100` on identical synthetic input, then diffs fa_q,
//! fa_gate, and fa_k bitwise. 16Q/4K/head_dim 256/n_rot 64 (Qwen3.5-4B shape).
//!
//! Usage: probe_fa_prep [out_prefix]

use rdna_compute::{DType, Gpu};

fn main() {
    let prefix = std::env::args().nth(1).unwrap_or_else(|| "/tmp/fap".into());
    let mut gpu = Gpu::init().expect("GPU init failed");
    let n_q = 16usize;
    let n_kv = 4usize;
    let hd = 256usize;
    let n_rot = 64usize;

    // Deterministic interleaved q/gate projection output: [16 heads x 512].
    let q_full: Vec<f32> = (0..n_q * hd * 2)
        .map(|i| ((i as f64 * 0.013).sin() * 1.7 + (i as f64 * 0.0071).cos() * 0.4) as f32)
        .collect();
    // Pre-norm K: [4 heads x 256].
    let k_in: Vec<f32> = (0..n_kv * hd)
        .map(|i| ((i as f64 * 0.029).cos() * 1.3) as f32)
        .collect();
    let q_w: Vec<f32> = (0..hd)
        .map(|i| 0.8 + 0.4 * ((i as f64 * 0.01).sin()) as f32)
        .collect();
    let k_w: Vec<f32> = (0..hd)
        .map(|i| 0.9 + 0.2 * ((i as f64 * 0.013).cos()) as f32)
        .collect();

    let q_full_t = gpu.upload_f32(&q_full, &[n_q * hd * 2]).expect("upload");
    let k_t = gpu.upload_f32(&k_in, &[n_kv * hd]).expect("upload");
    let qw_t = gpu.upload_f32(&q_w, &[hd]).expect("upload qw");
    let kw_t = gpu.upload_f32(&k_w, &[hd]).expect("upload kw");
    let pos: i32 = 7;
    let pos_buf = gpu.hip.malloc(4).expect("pos alloc");
    gpu.hip
        .memcpy_htod(&pos_buf, &pos.to_ne_bytes())
        .expect("pos copy");

    // ── Legacy arm ──
    let fa_q = gpu.alloc_tensor(&[n_q * hd], DType::F32).expect("q");
    let fa_gate = gpu.alloc_tensor(&[n_q * hd], DType::F32).expect("g");
    let fa_k = gpu.upload_f32(&k_in, &[n_kv * hd]).expect("k upload");
    gpu.deinterleave_f32(&q_full_t, &fa_q, &fa_gate, n_q, hd)
        .expect("deinterleave");
    gpu.rmsnorm_batched(&fa_q, &qw_t, &fa_q, n_q, hd, 1e-6)
        .expect("qnorm");
    gpu.rmsnorm_batched(&fa_k, &kw_t, &fa_k, n_kv, hd, 1e-6)
        .expect("knorm");
    gpu.rope_partial_interleaved_f32(&fa_q, &fa_k, &pos_buf, n_q, n_kv, hd, n_rot, 1_000_000.0)
        .expect("rope");
    gpu.hip.device_synchronize().expect("sync");
    let l_q = gpu.download_f32(&fa_q).unwrap();
    let l_g = gpu.download_f32(&fa_gate).unwrap();
    let l_k = gpu.download_f32(&fa_k).unwrap();

    // ── Fused arm (fresh buffers) ──
    let fa_q2 = gpu.alloc_tensor(&[n_q * hd], DType::F32).expect("q2");
    let fa_g2 = gpu.alloc_tensor(&[n_q * hd], DType::F32).expect("g2");
    let fa_k2 = gpu.upload_f32(&k_in, &[n_kv * hd]).expect("k2");
    gpu.qwen35_fa_prep_gfx1100(
        &q_full_t,
        &fa_q2,
        &fa_g2,
        &fa_k2,
        &qw_t,
        &kw_t,
        &pos_buf,
        1e-6,
        1_000_000.0,
        n_q,
        n_kv,
    )
    .expect("fused prep");
    gpu.hip.device_synchronize().expect("sync2");
    let f_q = gpu.download_f32(&fa_q2).unwrap();
    let f_g = gpu.download_f32(&fa_g2).unwrap();
    let f_k = gpu.download_f32(&fa_k2).unwrap();

    for (name, a, b) in [
        ("fa_q", &l_q, &f_q),
        ("fa_gate", &l_g, &f_g),
        ("fa_k", &l_k, &f_k),
    ] {
        let diffs: Vec<usize> = a
            .iter()
            .zip(b.iter())
            .enumerate()
            .filter(|(_, (x, y))| x != y)
            .map(|(i, _)| i)
            .collect();
        if diffs.is_empty() {
            eprintln!("{name}: IDENTICAL ({} elems)", a.len());
        } else {
            let mut buckets = [0usize; 8];
            for &i in &diffs {
                buckets[(i % hd) / 32] += 1;
            }
            eprintln!(
                "{name}: {} DIFFS; per-dim-bucket(32): {:?}; first idx {} (head {} dim {}): legacy={} fused={}",
                diffs.len(),
                buckets,
                diffs[0],
                diffs[0] / hd,
                diffs[0] % hd,
                a[diffs[0]],
                b[diffs[0]]
            );
        }
        std::fs::write(format!("{prefix}_{name}.bin"), unsafe {
            std::slice::from_raw_parts(b.as_ptr() as *const u8, b.len() * 4)
        })
        .expect("write");
    }
}
