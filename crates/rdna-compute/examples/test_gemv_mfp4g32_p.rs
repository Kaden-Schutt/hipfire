// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! GPU vs CPU correctness gate for mfp4+P (`gemv_mfp4g32_p` +
//! `dequantize_mfp4g32_p_to_f16`).
//!
//! mfp4+P = mfp4 (E2M1 + FP16 row scale + offline FWHT) with the per-32-block
//! UE8M0 scale promoted to E4M3 (FP8, non-power-of-2). Byte layout IDENTICAL to
//! mfp4 (16-B hdr + n_blocks×17 B, NO prefix). Recon:
//!   value = row_scale_a * e4m3_decode(scale_byte) * E2M1_LUT[nibble].
//!
//! Tests per K in {256,512,1024,1280,1536,1792,2048}:
//!   1. Size invariant: packed.len() == m*(16+17*(k/32)).
//!   2. E4M3 codec round-trip: e4m3_decode(encode_roundup(s)) >= s and is the
//!      smallest such code (ceil) — verified over a value sweep.
//!   3. dequantize_mfp4g32_p_to_f16 GPU vs CPU dequant: max diff < 1e-2.
//!   4. gemv_mfp4g32_p GPU y vs CPU dot(cpu_dequant, x_rot): max rel err < 1e-2.
//!   5. Quant-error sanity vs ORIGINAL weights (rotated): NRMSE < ~0.15 (E2M1
//!      4-bit grade), i.e. not garbage.

use rdna_compute::{DType, Gpu};

// ── fp16 helpers ──────────────────────────────────────────────────────────────
fn f32_to_f16_bits(v: f32) -> u16 {
    let bits = v.to_bits();
    let sign = ((bits >> 31) & 0x1) as u16;
    let exp = ((bits >> 23) & 0xff) as i32;
    let mant = bits & 0x7fffff;
    if exp == 0xff {
        (sign << 15) | (0x1f << 10) | if mant != 0 { 0x200 } else { 0 }
    } else if exp - 127 + 15 < 1 {
        sign << 15
    } else if exp - 127 + 15 > 30 {
        (sign << 15) | (0x1f << 10)
    } else {
        let new_exp = (exp - 127 + 15) as u16;
        let m13 = mant & 0x1fff;
        let mut new_mant = (mant >> 13) as u16;
        if m13 > 0x1000 || (m13 == 0x1000 && (new_mant & 1) != 0) {
            new_mant += 1;
        }
        let mut exp_bits = new_exp;
        if new_mant == 0x400 {
            new_mant = 0;
            exp_bits += 1;
        }
        (sign << 15) | (exp_bits << 10) | new_mant
    }
}
fn f16_bits_to_f32(h: u16) -> f32 {
    let sign = ((h >> 15) & 1) as u32;
    let exp = ((h >> 10) & 0x1f) as i32;
    let mant = (h & 0x3ff) as u32;
    let bits = if exp == 0 {
        if mant == 0 {
            sign << 31
        } else {
            let mut m = mant;
            let mut e = -1i32;
            while m & 0x400 == 0 {
                m <<= 1;
                e -= 1;
            }
            (sign << 31) | (((e + 127 - 14) as u32) << 23) | ((m & 0x3ff) << 13)
        }
    } else if exp == 0x1f {
        (sign << 31) | (0xff << 23) | (mant << 13)
    } else {
        (sign << 31) | (((exp - 15 + 127) as u32) << 23) | (mant << 13)
    };
    f32::from_bits(bits)
}

// ── FWHT ──────────────────────────────────────────────────────────────────────
fn cpu_fwht_256(x: &mut [f32], signs1: &[f32], signs2: &[f32]) {
    assert_eq!(x.len(), 256);
    for i in 0..256 {
        x[i] *= signs1[i];
    }
    let mut stride = 1usize;
    while stride < 256 {
        let mut i = 0;
        while i < 256 {
            for j in 0..stride {
                let a = x[i + j];
                let b = x[i + j + stride];
                x[i + j] = a + b;
                x[i + j + stride] = a - b;
            }
            i += stride * 2;
        }
        stride <<= 1;
    }
    let scale = 0.0625f32;
    for i in 0..256 {
        x[i] *= scale * signs2[i];
    }
}
fn gen_fwht_signs(seed: u32, n: usize) -> Vec<f32> {
    let mut state = seed;
    (0..n)
        .map(|_| {
            state = state.wrapping_mul(1103515245).wrapping_add(12345) & 0x7fffffff;
            if (state >> 16) & 1 == 1 {
                1.0f32
            } else {
                -1.0f32
            }
        })
        .collect()
}

// ── E2M1 + E4M3 codecs (bit-identical to quantizer + kernel) ───────────────────
const E2M1_LUT: [f32; 16] = [
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
];
fn e2m1_round(x: f32) -> u8 {
    let mut bi = 0u8;
    let mut be = f32::INFINITY;
    for (i, &c) in E2M1_LUT.iter().enumerate() {
        let e = (c - x).abs();
        if e < be {
            be = e;
            bi = i as u8;
        }
    }
    bi
}
fn e2m1_to_f32(n: u8) -> f32 {
    E2M1_LUT[(n & 0x0f) as usize]
}
fn e4m3_scale_decode(byte: u8) -> f32 {
    let exp = ((byte >> 3) & 0xf) as i32;
    let mant = (byte & 0x7) as u32;
    if exp == 0 {
        return (2.0f32).powi(-6) * (mant as f32) / 8.0;
    }
    if exp == 0xf && mant == 7 {
        return 448.0;
    }
    (2.0f32).powi(exp - 7) * (1.0 + (mant as f32) / 8.0)
}
fn e4m3_scale_encode_roundup(s: f32) -> u8 {
    if !(s > 0.0) {
        return 0x00;
    }
    if s >= 448.0 {
        return 0x7E;
    }
    for code in 0u8..=0x7E {
        if e4m3_scale_decode(code) >= s {
            return code;
        }
    }
    0x7E
}

// ── mfp4+P quantizer (inline, mirrors hipfire-quantize) ────────────────────────
fn quantize_mfp4g32_p_row(row: &[f32]) -> Vec<u8> {
    let k = row.len();
    let n_blocks = k / 32;
    let row_bytes = 16 + n_blocks * 17;
    let mut out = vec![0u8; row_bytes];
    let row_max_abs = row.iter().cloned().fold(0.0f32, |m, v| m.max(v.abs()));
    let row_scale_a = if row_max_abs > 0.0 {
        row_max_abs / 6.0
    } else {
        1.0
    };
    let inv_row_scale = if row_max_abs > 0.0 {
        1.0 / row_scale_a
    } else {
        0.0
    };
    // store row_scale_a as fp16 (round-trip through fp16, kernel reads fp16)
    let rsa16 = f32_to_f16_bits(row_scale_a);
    out[0..2].copy_from_slice(&rsa16.to_le_bytes());
    out[4..6].copy_from_slice(&(n_blocks as u16).to_le_bytes());
    out[6] = 0x05;
    for b in 0..n_blocks {
        let block = &row[b * 32..b * 32 + 32];
        let bmax = block.iter().cloned().fold(0.0f32, |m, v| m.max(v.abs()));
        let bmn = bmax * inv_row_scale;
        let s = if bmn > 0.0 { bmn / 6.0 } else { 0.0 };
        let scale_byte = e4m3_scale_encode_roundup(s);
        let bsf = e4m3_scale_decode(scale_byte);
        let inv_bs = if bsf > 0.0 { 1.0 / bsf } else { 0.0 };
        let po = 16 + b * 17;
        out[po] = scale_byte;
        for i in 0..16 {
            let lo = block[2 * i] * inv_row_scale * inv_bs;
            let hi = block[2 * i + 1] * inv_row_scale * inv_bs;
            out[po + 1 + i] = (e2m1_round(lo) & 0x0F) | ((e2m1_round(hi) & 0x0F) << 4);
        }
    }
    out
}
fn quantize_mfp4g32_p_2d(data: &[f32], m: usize, k: usize, s1: &[f32], s2: &[f32]) -> Vec<u8> {
    let row_bytes = 16 + 17 * (k / 32);
    let mut out = Vec::with_capacity(m * row_bytes);
    let mut rb = vec![0.0f32; k];
    for r in 0..m {
        rb.copy_from_slice(&data[r * k..(r + 1) * k]);
        for seg in 0..(k / 256) {
            cpu_fwht_256(&mut rb[seg * 256..(seg + 1) * 256], s1, s2);
        }
        out.extend_from_slice(&quantize_mfp4g32_p_row(&rb));
    }
    out
}
// CPU dequant ref — reads fp16 row scale back (matches kernel + loader).
fn dequant_mfp4g32_p(packed: &[u8], m: usize, k: usize) -> Vec<f32> {
    let row_bytes = 16 + 17 * (k / 32);
    let mut out = vec![0.0f32; m * k];
    for r in 0..m {
        let base = r * row_bytes;
        let rsa = f16_bits_to_f32(u16::from_le_bytes([packed[base], packed[base + 1]]));
        for b in 0..(k / 32) {
            let po = base + 16 + b * 17;
            let scale = rsa * e4m3_scale_decode(packed[po]);
            for i in 0..16 {
                let byte = packed[po + 1 + i];
                out[r * k + b * 32 + 2 * i] = scale * e2m1_to_f32(byte & 0x0F);
                out[r * k + b * 32 + 2 * i + 1] = scale * e2m1_to_f32((byte >> 4) & 0x0F);
            }
        }
    }
    out
}

fn test_e4m3_ceil() -> bool {
    // Round-up property: decode(encode(s)) >= s, and it's the smallest such code.
    let mut ok = true;
    let mut s = 0.001f32;
    while s < 460.0 {
        let code = e4m3_scale_encode_roundup(s);
        let dec = e4m3_scale_decode(code);
        if dec + 1e-6 < s && s < 448.0 {
            eprintln!("[FAIL] e4m3 ceil: s={s} code={code} dec={dec} < s");
            ok = false;
            break;
        }
        // Smaller code must decode < s (otherwise not the ceil) — skip code 0.
        if code > 0 && s < 448.0 {
            let prev = e4m3_scale_decode(code - 1);
            if prev >= s {
                eprintln!("[FAIL] e4m3 not minimal: s={s} code={code} prev_dec={prev}>=s");
                ok = false;
                break;
            }
        }
        s *= 1.013;
    }
    println!(
        "[{}] e4m3 ceil round-trip",
        if ok { "PASS" } else { "FAIL" }
    );
    ok
}

fn run(gpu: &mut Gpu, m: usize, k: usize, s1: &[f32], s2: &[f32]) -> bool {
    let mut st: u64 = 0x1234_5678_9abc_def0u64.wrapping_add(k as u64 * 7 + m as u64);
    let mut rnd = || {
        st ^= st << 13;
        st ^= st >> 7;
        st ^= st << 17;
        ((st & 0xFFFFFF) as f32 / 0xFFFFFF as f32) * 2.0 - 1.0
    };
    let data: Vec<f32> = (0..m * k).map(|_| 0.5 * rnd()).collect();

    let packed = quantize_mfp4g32_p_2d(&data, m, k, s1, s2);
    let row_bytes = 16 + 17 * (k / 32);
    if packed.len() != m * row_bytes {
        eprintln!("[FAIL] size {} != {}", packed.len(), m * row_bytes);
        return false;
    }

    let cpu_dq = dequant_mfp4g32_p(&packed, m, k);

    // Rotated original (for NRMSE sanity).
    let mut rot = data.clone();
    for r in 0..m {
        for seg in 0..(k / 256) {
            cpu_fwht_256(&mut rot[r * k + seg * 256..r * k + (seg + 1) * 256], s1, s2);
        }
    }
    let mut se = 0.0f64;
    let mut sn = 0.0f64;
    for i in 0..m * k {
        let d = (cpu_dq[i] - rot[i]) as f64;
        se += d * d;
        sn += (rot[i] as f64) * (rot[i] as f64);
    }
    let nrmse = (se / sn.max(1e-30)).sqrt();
    let nrmse_ok = nrmse < 0.15;
    println!(
        "[{}] quant K={} NRMSE(vs rotated orig)={:.4}",
        if nrmse_ok { "PASS" } else { "FAIL" },
        k,
        nrmse
    );
    if !nrmse_ok {
        return false;
    }

    // GPU dequant-to-f16.
    let d_packed = gpu.upload_raw(&packed, &[packed.len()]).unwrap();
    let f16_bytes = m * k * 2;
    let d_f16 = gpu.upload_raw(&vec![0u8; f16_bytes], &[f16_bytes]).unwrap();
    gpu.dequantize_mfp4g32_p_to_f16(&d_packed.buf, &d_f16.buf, m, k)
        .unwrap();
    gpu.hip.device_synchronize().unwrap();
    let mut f16raw = vec![0u8; f16_bytes];
    gpu.hip.memcpy_dtoh(&mut f16raw, &d_f16.buf).unwrap();
    let mut gpu_dq = vec![0.0f32; m * k];
    for i in 0..m * k {
        gpu_dq[i] = f16_bits_to_f32(u16::from_le_bytes([f16raw[2 * i], f16raw[2 * i + 1]]));
    }
    let mut max_dq = 0.0f32;
    for i in 0..m * k {
        let d = (gpu_dq[i] - cpu_dq[i]).abs();
        if d > max_dq {
            max_dq = d;
        }
    }
    let dq_ok = max_dq < 1e-2;
    println!(
        "[{}] dequant K={} max_gpu_vs_cpu={:.3e}",
        if dq_ok { "PASS" } else { "FAIL" },
        k,
        max_dq
    );
    if !dq_ok {
        return false;
    }

    // GPU GEMV.
    let x: Vec<f32> = (0..k)
        .map(|i| ((i as i32 % 13) as f32 - 6.0) * 0.05)
        .collect();
    let mut x_rot = x.clone();
    for seg in 0..(k / 256) {
        cpu_fwht_256(&mut x_rot[seg * 256..(seg + 1) * 256], s1, s2);
    }
    let mut y_ref = vec![0.0f32; m];
    for r in 0..m {
        let mut acc = 0.0f64;
        for c in 0..k {
            acc += cpu_dq[r * k + c] as f64 * x_rot[c] as f64;
        }
        y_ref[r] = acc as f32;
    }
    let d_xrot = gpu.upload_f32(&x_rot, &[k]).unwrap();
    let d_y = gpu.zeros(&[m], DType::F32).unwrap();
    gpu.gemv_mfp4g32_p(&d_packed, &d_xrot, &d_y, m, k).unwrap();
    let y_gpu = gpu.download_f32(&d_y).unwrap();
    let mut max_rel = 0.0f32;
    for r in 0..m {
        let den = y_ref[r].abs().max(1e-4);
        let rel = (y_gpu[r] - y_ref[r]).abs() / den;
        if rel > max_rel {
            max_rel = rel;
        }
    }
    let gemv_ok = max_rel < 1e-2;
    println!(
        "[{}] gemv  K={} max_rel={:.3e}",
        if gemv_ok { "PASS" } else { "FAIL" },
        k,
        max_rel
    );
    gemv_ok
}

fn main() {
    let mut gpu = Gpu::init().expect("Gpu::init failed");
    println!("arch: {}", gpu.arch);
    let s1 = gen_fwht_signs(42, 256);
    let s2 = gen_fwht_signs(1042, 256);
    let m = 64usize;
    let mut all = test_e4m3_ceil();
    for gpr in [1usize, 2, 4, 5, 6, 7, 8] {
        let k = gpr * 256;
        if !run(&mut gpu, m, k, &s1, &s2) {
            all = false;
        }
    }
    println!("{}", if all { "ALL PASS" } else { "SOME FAIL" });
    if !all {
        std::process::exit(1);
    }
}
