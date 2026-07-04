// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 hipfire contributors
// hipfire — see LICENSE and NOTICE in the project root.
//! Validation of `attention_flash_f16k_q8v_batched_masked` (KVarN v1 read-path
//! flash) against a host reference flash computed over the EXACT f16 K shadow +
//! Q8_0 V the kernel reads. Single causal query at the last position over a GQA
//! cache. Confirms the f16 K-load path + reused Q8 V path produce correct
//! attention to f16/Q8 tolerance.
//!
//!   cargo run --release -p hipfire-rdna --example parity_kvarn_flash

use hipfire_rdna::{DType, Gpu};

fn f16_to_f32(bits: u16) -> f32 {
    let s = (bits >> 15) & 1;
    let e = (bits >> 10) & 0x1f;
    let m = bits & 0x3ff;
    let v = if e == 0 {
        (m as f32) * 2f32.powi(-24)
    } else if e == 31 {
        if m == 0 {
            f32::INFINITY
        } else {
            f32::NAN
        }
    } else {
        (1.0 + m as f32 / 1024.0) * 2f32.powi(e as i32 - 15)
    };
    if s == 1 {
        -v
    } else {
        v
    }
}

fn f32_to_f16(x: f32) -> u16 {
    let bits = x.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let mut exp = ((bits >> 23) & 0xff) as i32 - 127 + 15;
    let mant = bits & 0x7f_ffff;
    if exp >= 0x1f {
        return sign | 0x7c00;
    }
    if exp <= 0 {
        if exp < -10 {
            return sign;
        }
        let mant = mant | 0x80_0000;
        let shift = (14 - exp) as u32;
        let mut h = (mant >> shift) as u16;
        if (mant >> (shift - 1)) & 1 == 1 {
            let sticky = mant & ((1 << (shift - 1)) - 1);
            if sticky != 0 || (h & 1) == 1 {
                h += 1;
            }
        }
        return sign | h;
    }
    let mut h_mant = (mant >> 13) as u16;
    if (mant >> 12) & 1 == 1 {
        let sticky = mant & 0xfff;
        if sticky != 0 || (h_mant & 1) == 1 {
            h_mant += 1;
            if h_mant == 0x400 {
                h_mant = 0;
                exp += 1;
            }
        }
    }
    sign | ((exp as u16) << 10) | h_mant
}

fn lcg(seed: u32, n: usize) -> Vec<f32> {
    let mut s = seed.max(1);
    let mut u = || {
        s = s.wrapping_mul(1_103_515_245).wrapping_add(12345) & 0x7fff_ffff;
        (s as f32 + 0.5) / 2_147_483_648.0
    };
    (0..n)
        .map(|_| {
            let u1 = u().max(1e-7);
            let u2 = u();
            (-2.0 * u1.ln()).sqrt() * (std::f32::consts::TAU * u2).cos()
        })
        .collect()
}

fn main() {
    let n_heads = 4usize;
    let n_kv_heads = 2usize;
    // This flash family processes 8 dims/thread × 32 threads = 256 dims, i.e. it
    // is specialized for head_dim=256 (qwen3.5). Match that here.
    let head_dim = 256usize;
    let seq_len = 200usize;
    let max_seq = 256usize;
    let kv_group = n_heads / n_kv_heads;
    let kv_dim = n_kv_heads * head_dim;
    let blocks_per_head = head_dim / 32;
    let v_row_stride = n_kv_heads * blocks_per_head * 34;

    let mut gpu = Gpu::init().unwrap();

    // Query at the last position.
    let q: Vec<f32> = lcg(3, n_heads * head_dim).iter().map(|v| v * 0.2).collect();
    // f16 K shadow [max_seq × kv_dim] (only [0,seq_len) populated).
    let kbase = lcg(5, seq_len * kv_dim);
    let mut k_f16 = vec![0u16; max_seq * kv_dim];
    let mut k_f32 = vec![0.0f32; max_seq * kv_dim];
    for t in 0..seq_len {
        for j in 0..kv_dim {
            let v = kbase[t * kv_dim + j] * 0.3;
            let h16 = f32_to_f16(v);
            k_f16[t * kv_dim + j] = h16;
            k_f32[t * kv_dim + j] = f16_to_f32(h16);
        }
    }
    // Q8_0 V cache [max_seq × v_row_stride].
    let vbase = lcg(7, seq_len * kv_dim);
    let mut v_cache = vec![0u8; max_seq * v_row_stride];
    let mut v_deq = vec![0.0f32; max_seq * kv_dim]; // [t, kv_h*head_dim + d]
    for t in 0..seq_len {
        for kvh in 0..n_kv_heads {
            for b in 0..blocks_per_head {
                let mut amax = 0.0f32;
                for e in 0..32 {
                    let d = b * 32 + e;
                    amax = amax.max(vbase[t * kv_dim + kvh * head_dim + d].abs() * 0.3);
                }
                let scale = (amax / 127.0).max(1e-8);
                let blk = t * v_row_stride + (kvh * blocks_per_head + b) * 34;
                v_cache[blk..blk + 2].copy_from_slice(&f32_to_f16(scale).to_le_bytes());
                for e in 0..32 {
                    let d = b * 32 + e;
                    let x = vbase[t * kv_dim + kvh * head_dim + d] * 0.3;
                    let qd = (x / scale).round().clamp(-127.0, 127.0) as i8;
                    v_cache[blk + 2 + e] = qd as u8;
                    v_deq[t * kv_dim + kvh * head_dim + d] = scale * qd as f32;
                }
            }
        }
    }

    // Host reference flash: causal, single query at pos=seq_len-1.
    let scale_attn = 1.0f32 / (head_dim as f32).sqrt();
    let mut ref_out = vec![0.0f32; n_heads * head_dim];
    for h in 0..n_heads {
        let kvh = h / kv_group;
        let mut scores = vec![0.0f32; seq_len];
        let mut mx = f32::NEG_INFINITY;
        for t in 0..seq_len {
            let mut s = 0.0f32;
            for d in 0..head_dim {
                s += q[h * head_dim + d] * k_f32[t * kv_dim + kvh * head_dim + d];
            }
            s *= scale_attn;
            scores[t] = s;
            mx = mx.max(s);
        }
        let mut sum = 0.0f32;
        for t in 0..seq_len {
            scores[t] = (scores[t] - mx).exp();
            sum += scores[t];
        }
        for d in 0..head_dim {
            let mut acc = 0.0f32;
            for t in 0..seq_len {
                acc += scores[t] * v_deq[t * kv_dim + kvh * head_dim + d];
            }
            ref_out[h * head_dim + d] = acc / sum;
        }
    }

    // GPU flash.
    let qd = gpu
        .upload_raw(
            &q.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>(),
            &[n_heads * head_dim],
        )
        .unwrap();
    // K is a raw f16 byte buffer; the flash kernel casts buf to _Float16*, so the
    // tensor's nominal dtype is irrelevant (the launcher passes buf.as_ptr()).
    let kd = gpu
        .upload_raw(
            &k_f16
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect::<Vec<_>>(),
            &[max_seq * kv_dim],
        )
        .unwrap();
    let vd = gpu.upload_raw(&v_cache, &[max_seq * v_row_stride]).unwrap();
    let outd = gpu
        .upload_raw(&vec![0u8; n_heads * head_dim * 4], &[n_heads * head_dim])
        .unwrap();
    let posd = gpu
        .upload_raw(&((seq_len - 1) as i32).to_le_bytes(), &[1])
        .unwrap();
    let max_tiles = max_seq.div_ceil(128);
    let partials = gpu
        .zeros(&[n_heads * max_tiles * (2 + head_dim)], DType::F32)
        .unwrap();

    gpu.attention_flash_f16k_q8v_batched_masked(
        &qd, &kd, &vd, &outd, &posd, n_heads, n_kv_heads, head_dim, max_seq, seq_len, 1, &partials,
        None, 0, 0,
    )
    .unwrap();
    gpu.device_synchronize().unwrap();
    let got = gpu.download_f32(&outd).unwrap();

    let mut max_abs = 0.0f32;
    let mut max_rel = 0.0f32;
    for i in 0..n_heads * head_dim {
        let e = (got[i] - ref_out[i]).abs();
        max_abs = max_abs.max(e);
        max_rel = max_rel.max(e / ref_out[i].abs().max(1e-3));
    }
    let pass = max_abs < 2e-3;
    println!(
        "parity_kvarn_flash on {}: max-abs-err={max_abs:.2e} max-rel-err={max_rel:.2e} -> {}",
        gpu.arch,
        if pass { "PASS" } else { "FAIL" }
    );
    if !pass {
        std::process::exit(1);
    }
}
