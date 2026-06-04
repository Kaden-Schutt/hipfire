// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire -- see LICENSE and NOTICE in the project root.

//! Channel test for `gemm_f16_x_f32_wmma`.
//!
//! Validates the dispatcher path that stages FP32 activations through FP16
//! scratch before using the F16 WMMA kernel.

use rdna_compute::{DType, Gpu};

fn f32_to_f16_bits(x: f32) -> u16 {
    let bits = x.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xff) as i32;
    let mant = bits & 0x7f_ffff;

    if exp == 0xff {
        return sign | if mant == 0 { 0x7c00 } else { 0x7e00 };
    }

    let half_exp = exp - 127 + 15;
    if half_exp >= 0x1f {
        return sign | 0x7c00;
    }
    if half_exp <= 0 {
        if half_exp < -10 {
            return sign;
        }
        let mantissa = mant | 0x80_0000;
        let shift = (14 - half_exp) as u32;
        let mut half_mant = (mantissa >> shift) as u16;
        let round_bit = 1u32 << (shift - 1);
        let remainder = mantissa & (round_bit - 1);
        if (mantissa & round_bit) != 0 && (remainder != 0 || (half_mant & 1) != 0) {
            half_mant += 1;
        }
        return sign | half_mant;
    }

    let mut half = sign | ((half_exp as u16) << 10) | ((mant >> 13) as u16);
    let round = mant & 0x1fff;
    if round > 0x1000 || (round == 0x1000 && (half & 1) != 0) {
        half += 1;
    }
    half
}

fn f16_bits_to_f32(x: u16) -> f32 {
    let sign = ((x as u32 & 0x8000) << 16) as u32;
    let exp = ((x >> 10) & 0x1f) as i32;
    let mant = (x & 0x03ff) as u32;
    let bits = if exp == 0 {
        if mant == 0 {
            sign
        } else {
            let mut mantissa = mant;
            let mut e = -14i32;
            while (mantissa & 0x0400) == 0 {
                mantissa <<= 1;
                e -= 1;
            }
            mantissa &= 0x03ff;
            sign | (((e + 127) as u32) << 23) | (mantissa << 13)
        }
    } else if exp == 0x1f {
        sign | 0x7f80_0000 | (mant << 13)
    } else {
        sign | (((exp - 15 + 127) as u32) << 23) | (mant << 13)
    };
    f32::from_bits(bits)
}

fn f32_to_f16_value(x: f32) -> f32 {
    f16_bits_to_f32(f32_to_f16_bits(x))
}

fn f16_bytes(values: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(values.len() * 2);
    for &v in values {
        out.extend_from_slice(&f32_to_f16_bits(v).to_le_bytes());
    }
    out
}

fn main() {
    let mut gpu = Gpu::init().expect("gpu init");
    println!("arch={}", gpu.arch);

    let m = 48usize;
    let k = 64usize;
    let b = 17usize;

    let weights: Vec<f32> = (0..m * k)
        .map(|i| {
            let row = i / k;
            let col = i % k;
            ((row as i32 % 11) as f32 - 5.0) * 0.011 + ((col as i32 % 17) as f32 - 8.0) * 0.007
        })
        .collect();
    let x: Vec<f32> = (0..b * k)
        .map(|i| {
            let row = i / k;
            let col = i % k;
            ((row as i32 % 7) as f32 - 3.0) * 0.019 - ((col as i32 % 13) as f32 - 6.0) * 0.006
        })
        .collect();

    let mut w_gpu = gpu
        .upload_raw(&f16_bytes(&weights), &[m, k])
        .expect("upload f16 weights");
    w_gpu.dtype = DType::F16;
    let x_gpu = gpu.upload_f32(&x, &[b, k]).expect("upload x");
    let y_gpu = gpu.alloc_tensor(&[b, m], DType::F32).expect("alloc y");

    gpu.gemm_f16_x_f32_wmma(&w_gpu, &x_gpu, &y_gpu, m, k, b)
        .expect("gemm_f16_x_f32_wmma");
    gpu.hip.device_synchronize().expect("sync");

    let y = gpu.download_f32(&y_gpu).expect("download y");

    let mut max_abs = 0.0f32;
    let mut rms = 0.0f64;
    for bb in 0..b {
        for mm in 0..m {
            let mut acc = 0.0f32;
            for kk in 0..k {
                let wv = f32_to_f16_value(weights[mm * k + kk]);
                let xv = f32_to_f16_value(x[bb * k + kk]);
                acc += wv * xv;
            }
            let got = y[bb * m + mm];
            let diff = (got - acc).abs();
            max_abs = max_abs.max(diff);
            rms += (diff as f64) * (diff as f64);
        }
    }
    rms = (rms / (b * m) as f64).sqrt();
    println!("max_abs={max_abs:.6e} rms={rms:.6e}");

    assert!(
        max_abs < 1.0e-4,
        "F16 WMMA mismatch: max_abs={max_abs:.6e} rms={rms:.6e}"
    );
}
