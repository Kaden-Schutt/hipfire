// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Parity for the gfx1103 no-LDS batched causal attention kernel.
//! Compares the dispatcher output against an f64 CPU reference (causal
//! softmax, ctx_len = qpos+1). On gfx1103 (default) runs the no-LDS kernel;
//! with HIPFIRE_FORCE_GENERIC=1 runs the generic LDS kernel — both must match.
//!
//!   cargo run --release -p hipfire-rdna --example parity_attention_causal_gfx1103
//!   HIPFIRE_FORCE_GENERIC=1 cargo run ... --example parity_attention_causal_gfx1103

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
    let forced = std::env::var("HIPFIRE_FORCE_GENERIC").is_ok();
    let (nh, nkv, hd, seq) = (8usize, 2usize, 128usize, 96usize);
    let q_stride = nh * hd;
    let kv_stride = nkv * hd;
    let mut gpu = Gpu::init().expect("gpu init");
    println!(
        "force_generic={forced} (path: {})  nh={nh} nkv={nkv} hd={hd} seq={seq}",
        if forced { "generic LDS" } else { "arch-selected" }
    );

    let qh = lcg(0x11, seq * q_stride);
    let kh = lcg(0x22, seq * kv_stride);
    let vh = lcg(0x33, seq * kv_stride);
    let dq = gpu.upload_f32(&qh, &[seq * q_stride]).unwrap();
    let dk = gpu.upload_f32(&kh, &[seq * kv_stride]).unwrap();
    let dv = gpu.upload_f32(&vh, &[seq * kv_stride]).unwrap();
    let dout = gpu.zeros(&[seq * q_stride], DType::F32).unwrap();

    gpu.attention_causal_batched(&dq, &dk, &dv, &dout, seq, nh, nkv, hd).unwrap();
    gpu.hip.device_synchronize().unwrap();
    let got = gpu.download_f32(&dout).unwrap();

    // f64 CPU reference
    let scale = 1.0f64 / (hd as f64).sqrt();
    let mut refv = vec![0f32; seq * q_stride];
    for qpos in 0..seq {
        for h in 0..nh {
            let kv_h = h / (nh / nkv);
            let ctx = qpos + 1;
            let mut sc = vec![0f64; ctx];
            let mut mx = f64::MIN;
            for (t, s) in sc.iter_mut().enumerate() {
                let mut dot = 0f64;
                for d in 0..hd {
                    dot += qh[qpos * q_stride + h * hd + d] as f64
                        * kh[t * kv_stride + kv_h * hd + d] as f64;
                }
                *s = dot * scale;
                mx = mx.max(*s);
            }
            let mut den = 0f64;
            for s in sc.iter_mut() { *s = (*s - mx).exp(); den += *s; }
            for d in 0..hd {
                let mut acc = 0f64;
                for (t, &s) in sc.iter().enumerate() {
                    acc += s * vh[t * kv_stride + kv_h * hd + d] as f64;
                }
                refv[qpos * q_stride + h * hd + d] = (acc / den) as f32;
            }
        }
    }
    let err = got.iter().zip(&refv).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
    let mag = refv.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
    println!("  attention_causal_batched  max_abs_err={err:.3e}  mag={mag:.4}");
    if err < 3e-4 {
        println!("OK");
    } else {
        eprintln!("PARITY FAIL err={err:.3e}");
        std::process::exit(1);
    }
}
