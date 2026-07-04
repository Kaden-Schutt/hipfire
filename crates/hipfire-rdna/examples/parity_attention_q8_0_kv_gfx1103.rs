// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Parity check for the gfx1103 no-LDS flash-decode attention kernel.
//!
//! Builds a real Q8_0 KV cache (via kv_cache_write_q8_0_batched), runs
//! `attention_q8_0_kv` through the live dispatcher, and compares against an f64
//! CPU reference that decodes the SAME quantized bytes. On gfx1103 (default)
//! this exercises the new `attention_q8_0_kv_gfx1103` online-softmax kernel;
//! with `HIPFIRE_FORCE_GENERIC=1` it exercises the generic LDS kernel — so the
//! same harness proves both paths agree with the reference.
//!
//!   cargo run --release -p hipfire-rdna --example parity_attention_q8_0_kv_gfx1103
//!   HIPFIRE_FORCE_GENERIC=1 cargo run --release -p hipfire-rdna \
//!       --example parity_attention_q8_0_kv_gfx1103

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

fn f16_to_f32(bits: u16) -> f32 {
    let sign = (bits >> 15) & 1;
    let exp = (bits >> 10) & 0x1f;
    let frac = bits & 0x3ff;
    let val = if exp == 0 {
        (frac as f32) * 2f32.powi(-24)
    } else if exp == 0x1f {
        if frac == 0 { f32::INFINITY } else { f32::NAN }
    } else {
        (1.0 + frac as f32 / 1024.0) * 2f32.powi(exp as i32 - 15)
    };
    if sign == 1 { -val } else { val }
}

fn main() {
    let forced = std::env::var("HIPFIRE_FORCE_GENERIC").is_ok();
    let n_heads = 8usize;
    let n_kv_heads = 2usize;
    let head_dim = 128usize;
    let seq_len = 333usize;
    let max_seq = 512usize;
    let kv_group = n_heads / n_kv_heads;
    let bph = head_dim / 32; // blocks per head
    let kv_dim = n_kv_heads * head_dim;
    let total_bpp = n_kv_heads * bph; // blocks per position
    let scale_attn = 1.0f32 / (head_dim as f32).sqrt();

    let mut gpu = Gpu::init().expect("gpu init");
    println!(
        "force_generic={forced} (path: {})  n_heads={n_heads} kv={n_kv_heads} hd={head_dim} seq={seq_len}",
        if forced { "generic LDS kernel" } else { "arch-selected kernel" }
    );

    // Random Q and F32 KV, then quantize KV to Q8_0.
    let q = lcg(0xa5a5, n_heads * head_dim);
    let k_f32 = lcg(0xc3c3, max_seq * kv_dim);
    let v_f32 = lcg(0x9696, max_seq * kv_dim);

    let d_q = gpu.upload_f32(&q, &[n_heads * head_dim]).unwrap();
    let d_k = gpu.upload_f32(&k_f32, &[max_seq * kv_dim]).unwrap();
    let d_v = gpu.upload_f32(&v_f32, &[max_seq * kv_dim]).unwrap();
    let d_out = gpu.zeros(&[n_heads * head_dim], DType::F32).unwrap();

    let d_kq8 = gpu.alloc_tensor(&[max_seq * kv_dim], DType::Q8_0).unwrap();
    let d_vq8 = gpu.alloc_tensor(&[max_seq * kv_dim], DType::Q8_0).unwrap();
    let pos_all: Vec<u8> = (0..max_seq as i32).flat_map(|p| p.to_ne_bytes()).collect();
    let pos_all_t = gpu.alloc_tensor(&[max_seq], DType::F32).unwrap();
    gpu.hip.memcpy_htod(&pos_all_t.buf, &pos_all).unwrap();
    gpu.kv_cache_write_q8_0_batched(&d_kq8, &d_k, &pos_all_t, n_kv_heads, head_dim, max_seq)
        .unwrap();
    gpu.kv_cache_write_q8_0_batched(&d_vq8, &d_v, &pos_all_t, n_kv_heads, head_dim, max_seq)
        .unwrap();

    let pos_i32 = (seq_len - 1) as i32;
    let pos_buf = gpu.hip.malloc(4).unwrap();
    gpu.hip.memcpy_htod(&pos_buf, &pos_i32.to_ne_bytes()).unwrap();

    gpu.attention_q8_0_kv(
        &d_q, &d_kq8, &d_vq8, &d_out, &pos_buf, seq_len, n_heads, n_kv_heads, head_dim, max_seq,
    )
    .unwrap();
    gpu.hip.device_synchronize().unwrap();
    let got = gpu.download_f32(&d_out).unwrap();

    // ── f64 CPU reference from the SAME quantized bytes ──────────────────
    // Only positions [0, seq_len) are read by the kernel; download just those.
    let kbytes = gpu.download_raw(&d_kq8, seq_len * total_bpp * 34).unwrap();
    let vbytes = gpu.download_raw(&d_vq8, seq_len * total_bpp * 34).unwrap();
    let deq = |bytes: &[u8], t: usize, kv_h: usize, d: usize| -> f64 {
        let bi = d / 32;
        let bj = d % 32;
        let blk = (t * total_bpp + kv_h * bph + bi) * 34;
        let scale = f16_to_f32(u16::from_le_bytes([bytes[blk], bytes[blk + 1]])) as f64;
        let qv = bytes[blk + 2 + bj] as i8 as f64;
        scale * qv
    };

    let mut reference = vec![0f32; n_heads * head_dim];
    for h in 0..n_heads {
        let kv_h = h / kv_group;
        // scores
        let mut scores = vec![0f64; seq_len];
        let mut mx = f64::MIN;
        for (t, sc) in scores.iter_mut().enumerate() {
            let mut dot = 0f64;
            for d in 0..head_dim {
                dot += q[h * head_dim + d] as f64 * deq(&kbytes, t, kv_h, d);
            }
            *sc = dot * scale_attn as f64;
            mx = mx.max(*sc);
        }
        let mut denom = 0f64;
        for sc in scores.iter_mut() {
            *sc = (*sc - mx).exp();
            denom += *sc;
        }
        for d in 0..head_dim {
            let mut acc = 0f64;
            for (t, &sc) in scores.iter().enumerate() {
                acc += sc * deq(&vbytes, t, kv_h, d);
            }
            reference[h * head_dim + d] = (acc / denom) as f32;
        }
    }

    let err = got
        .iter()
        .zip(&reference)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    let refmag = reference.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
    // Q8 KV quant error dominates; both kernels read identical bytes so the
    // online-softmax and two-pass paths must agree with the ref to ~fp slack.
    let tol = 5e-4f32;
    println!("  attention_q8_0_kv  max_abs_err={err:.3e}  (ref_mag={refmag:.3})");
    if err < tol {
        println!("OK — within tol={tol:.1e}");
    } else {
        eprintln!("PARITY FAIL — err {err:.3e} >= tol {tol:.1e}");
        std::process::exit(1);
    }
}
