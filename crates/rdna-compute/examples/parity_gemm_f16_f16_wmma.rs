// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 hipfire contributors
// hipfire — see LICENSE and NOTICE in the project root.
//! Numeric parity for the generic-library kernel `gemm_f16_f16_wmma`
//! (F16 × F16 → F16, gfx1103 zero-LDS WMMA, F32 accumulate + F16 store).
//!
//! GPU and the CPU reference consume the SAME f16 input bytes, so the test
//! validates the kernel (layout + accumulation) independent of host-conversion
//! nuance; the tolerance only absorbs a couple f16 ULP on the output round.
//! Run on an RDNA3 box (gfx1103/1100/1151).
//!
//!   cargo run -p rdna-compute --example parity_gemm_f16_f16_wmma [M K B]

use rdna_compute::Gpu;

fn lcg(seed: u32, n: usize) -> Vec<f32> {
    let mut s = seed;
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(1_103_515_245).wrapping_add(12345);
            ((s >> 16) & 0x7fff) as f32 / 32768.0 - 0.5
        })
        .collect()
}

/// f32 -> f16 (round half up; deterministic, valid f16 — exactness not required
/// since both sides consume identical input bytes).
fn f32_to_f16(f: f32) -> u16 {
    let bits = f.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xff) as i32;
    let mant = bits & 0x7fffff;
    if exp == 0xff {
        return sign | 0x7c00 | (if mant != 0 { 0x200 } else { 0 });
    }
    let new_exp = exp - 127 + 15;
    if new_exp >= 0x1f {
        return sign | 0x7c00;
    }
    if new_exp <= 0 {
        if new_exp < -10 {
            return sign;
        }
        let m = mant | 0x800000;
        let shift = (14 - new_exp) as u32;
        let rounded = m + (1 << (shift - 1));
        return sign | (rounded >> shift) as u16;
    }
    let half_mant = (mant >> 13) as u16;
    let round_bit = (mant >> 12) & 1;
    let mut h = sign | ((new_exp as u16) << 10) | half_mant;
    if round_bit == 1 {
        h = h.wrapping_add(1);
    }
    h
}

fn f16_to_f32(h: u16) -> f32 {
    let sign = ((h >> 15) & 1) as u32;
    let exp = ((h >> 10) & 0x1f) as u32;
    let mant = (h & 0x3ff) as u32;
    let bits = if exp == 0 {
        if mant == 0 {
            sign << 31
        } else {
            let mut e = -1i32;
            let mut m = mant;
            while (m & 0x400) == 0 {
                m <<= 1;
                e -= 1;
            }
            m &= 0x3ff;
            (sign << 31) | (((127 - 15 + 1 + e) as u32) << 23) | (m << 13)
        }
    } else if exp == 0x1f {
        (sign << 31) | 0x7f80_0000 | (mant << 13)
    } else {
        (sign << 31) | (((exp as i32 - 15 + 127) as u32) << 23) | (mant << 13)
    };
    f32::from_bits(bits)
}

fn to_f16_bytes(v: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(v.len() * 2);
    for &x in v {
        out.extend_from_slice(&f32_to_f16(x).to_le_bytes());
    }
    out
}

fn main() {
    let mut args = std::env::args().skip(1);
    let m: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(48);
    let k: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(64);
    let b: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(32);
    assert_eq!(k % 16, 0, "K must be a multiple of 16");

    let mut gpu = Gpu::init().unwrap();
    if !gpu.arch_caps.has_wmma_w32() {
        println!("SKIP gemm_f16_f16_wmma parity: {} lacks wave32 WMMA", gpu.arch);
        return;
    }

    let a_f32 = lcg(1, m * k); // [M, K]
    let x_f32 = lcg(2, b * k); // [B, K]

    // Reference reads the SAME f16 bytes, accumulates in F32, rounds to F16.
    let a_r: Vec<f32> = a_f32.iter().map(|&v| f16_to_f32(f32_to_f16(v))).collect();
    let x_r: Vec<f32> = x_f32.iter().map(|&v| f16_to_f32(f32_to_f16(v))).collect();
    let mut y_ref = vec![0.0f32; b * m]; // [B, M]
    for bi in 0..b {
        for mi in 0..m {
            let mut acc = 0.0f32;
            for ki in 0..k {
                acc += a_r[mi * k + ki] * x_r[bi * k + ki];
            }
            y_ref[bi * m + mi] = f16_to_f32(f32_to_f16(acc));
        }
    }

    let a_dev = gpu.upload_raw(&to_f16_bytes(&a_f32), &[m, k]).unwrap();
    let x_dev = gpu.upload_raw(&to_f16_bytes(&x_f32), &[b, k]).unwrap();
    let y_dev = gpu.upload_raw(&vec![0u8; b * m * 2], &[b, m]).unwrap();

    gpu.gemm_f16_f16_wmma(&a_dev, &x_dev, &y_dev, m, k, b).unwrap();
    gpu.device_synchronize().unwrap();

    let y_bytes = gpu.download_raw(&y_dev, b * m * 2).unwrap();
    let y_gpu: Vec<f32> = y_bytes
        .chunks_exact(2)
        .map(|c| f16_to_f32(u16::from_le_bytes([c[0], c[1]])))
        .collect();

    let mut max_abs = 0.0f32;
    let mut argmax = 0usize;
    for i in 0..b * m {
        let d = (y_gpu[i] - y_ref[i]).abs();
        if d > max_abs {
            max_abs = d;
            argmax = i;
        }
    }

    // F32 accumulate + single f16 store: residual is ~a couple f16 ULP of the
    // output magnitude (f16 mantissa is 10 bits -> 2^-10).
    let max_mag = y_ref.iter().fold(0.0f32, |a, &v| a.max(v.abs()));
    let tol_abs = 3.0 * max_mag * 2.0f32.powi(-10) + 1e-4;
    let pass = max_abs <= tol_abs;
    println!(
        "gemm_f16_f16_wmma parity M={m} K={k} B={b} on {}: \
         max_abs={max_abs:.5} (ref={:.5}, gpu={:.5} @ b={},m={}) tol_abs={tol_abs:.5} -> {}",
        gpu.arch,
        y_ref[argmax],
        y_gpu[argmax],
        argmax / m,
        argmax % m,
        if pass { "PASS" } else { "FAIL" }
    );
    if !pass {
        std::process::exit(1);
    }
}
