// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Parity for `attention_swa_gqa_batched` (GQA sliding-window batched attention
//! over a staged per-kv-head window cache). Feeds a hand-built staged cache +
//! Q and compares against an f64 CPU windowed-attention reference — verifying
//! the GQA query→kv-head mapping, per-kv-head `[batch, n_kv, head_dim, window]`
//! indexing, the passed scale, and the `n_valid` window truncation.
//!
//!   cargo run --release -p hipfire-rdna --example parity_attention_swa_gqa

use hipfire_rdna::{DType, Gpu};

fn lcg(seed: u64, n: usize) -> Vec<f32> {
    let mut s = seed;
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((s >> 40) as f32 / (1u32 << 24) as f32) * 2.0 - 1.0
        })
        .collect()
}

fn main() {
    let (nh, nkv, hd, win, b) = (8usize, 2usize, 128usize, 64usize, 5usize);
    let scale = 1.0f32 / (hd as f32).sqrt();
    let group = nh / nkv;
    let mut gpu = Gpu::init().expect("gpu init");

    // Per-row valid window count (≤ win). Columns [0, n_valid) are meaningful.
    let n_valid: Vec<i32> = vec![1, 17, 64, 40, 64];
    let q = lcg(0x11, b * nh * hd);
    // Staged caches: [batch, n_kv, head_dim, window]
    let kst = lcg(0x22, b * nkv * hd * win);
    let vst = lcg(0x33, b * nkv * hd * win);

    let dq = gpu.upload_f32(&q, &[b * nh * hd]).unwrap();
    let dk = gpu.upload_f32(&kst, &[b * nkv * hd * win]).unwrap();
    let dv = gpu.upload_f32(&vst, &[b * nkv * hd * win]).unwrap();
    let dout = gpu.zeros(&[b * nh * hd], DType::F32).unwrap();
    let nvb: Vec<u8> = n_valid.iter().flat_map(|p| p.to_ne_bytes()).collect();
    let dnv = gpu.alloc_tensor(&[b], DType::F32).unwrap();
    gpu.hip.memcpy_htod(&dnv.buf, &nvb).unwrap();

    gpu.attention_swa_gqa_batched(&dq, &dk, &dv, &dnv, &dout, nh, nkv, hd, win, b, scale)
        .unwrap();
    gpu.hip.device_synchronize().unwrap();
    let got = gpu.download_f32(&dout).unwrap();

    // CPU reference. Head-major staged idx (kvh, bb, d, p) = ((kvh*b+bb)*hd + d)*win + p.
    let kget = |bb: usize, kvh: usize, d: usize, p: usize| -> f64 {
        kst[((kvh * b + bb) * hd + d) * win + p] as f64
    };
    let vget = |bb: usize, kvh: usize, d: usize, p: usize| -> f64 {
        vst[((kvh * b + bb) * hd + d) * win + p] as f64
    };
    let mut refv = vec![0f32; b * nh * hd];
    for bb in 0..b {
        let nv = n_valid[bb] as usize;
        for h in 0..nh {
            let kvh = h / group;
            let mut sc = vec![0f64; nv];
            let mut mx = f64::MIN;
            for (p, s) in sc.iter_mut().enumerate() {
                let mut dot = 0f64;
                for d in 0..hd {
                    dot += q[(bb * nh + h) * hd + d] as f64 * kget(bb, kvh, d, p);
                }
                *s = dot * scale as f64;
                mx = mx.max(*s);
            }
            let mut den = 0f64;
            for s in sc.iter_mut() {
                *s = (*s - mx).exp();
                den += *s;
            }
            for d in 0..hd {
                let mut acc = 0f64;
                for (p, &s) in sc.iter().enumerate() {
                    acc += s * vget(bb, kvh, d, p);
                }
                refv[(bb * nh + h) * hd + d] = (acc / den) as f32;
            }
        }
    }

    let err = got
        .iter()
        .zip(&refv)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    let mag = refv.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
    println!("attention_swa_gqa_batched  max_abs_err={err:.3e}  mag={mag:.4}  (n_valid={n_valid:?})");
    if err < 3e-4 {
        println!("OK");
    } else {
        eprintln!("PARITY FAIL err={err:.3e}");
        std::process::exit(1);
    }
}
