// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Channel test for `gemm_bf16_x_bf16_wmma`.
//!
//! Runs a small BF16 weight x F32 activation GEMM through the dispatcher
//! path used by the KLD reference builder. The dispatcher stages X through
//! F32->BF16 on GPU; the CPU reference applies the same BF16 rounding to
//! both operands before accumulating in F32.

use rdna_compute::{DType, Gpu};

fn f32_to_bf16_bits(x: f32) -> u16 {
    let bits = x.to_bits();
    let lsb = (bits >> 16) & 1;
    ((bits + 0x7fff + lsb) >> 16) as u16
}

fn bf16_bits_to_f32(x: u16) -> f32 {
    f32::from_bits((x as u32) << 16)
}

fn f32_to_bf16_value(x: f32) -> f32 {
    bf16_bits_to_f32(f32_to_bf16_bits(x))
}

fn bf16_bytes(values: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(values.len() * 2);
    for &v in values {
        out.extend_from_slice(&f32_to_bf16_bits(v).to_le_bytes());
    }
    out
}

fn main() {
    let mut gpu = Gpu::init().expect("gpu init");
    println!("arch={}", gpu.arch);

    let m = 32usize;
    let k = 32usize;
    let b = 32usize;

    let weights: Vec<f32> = (0..m * k)
        .map(|i| {
            let row = i / k;
            let col = i % k;
            ((row as i32 % 9) as f32 - 4.0) * 0.017 + ((col as i32 % 11) as f32 - 5.0) * 0.013
        })
        .collect();
    let x: Vec<f32> = (0..b * k)
        .map(|i| {
            let row = i / k;
            let col = i % k;
            ((row as i32 % 7) as f32 - 3.0) * 0.021 - ((col as i32 % 13) as f32 - 6.0) * 0.009
        })
        .collect();

    let mut w_gpu = gpu
        .upload_raw(&bf16_bytes(&weights), &[m, k])
        .expect("upload bf16 weights");
    w_gpu.dtype = DType::BF16;
    let x_gpu = gpu.upload_f32(&x, &[b, k]).expect("upload x");
    let y_gpu = gpu.alloc_tensor(&[b, m], DType::F32).expect("alloc y");

    gpu.gemm_bf16_x_bf16_wmma(&w_gpu, &x_gpu, &y_gpu, m, k, b)
        .expect("gemm_bf16_x_bf16_wmma");
    gpu.hip.device_synchronize().expect("sync");

    let y = gpu.download_f32(&y_gpu).expect("download y");

    let mut max_abs = 0.0f32;
    let mut rms = 0.0f64;
    for bb in 0..b {
        for mm in 0..m {
            let mut acc = 0.0f32;
            for kk in 0..k {
                let wv = f32_to_bf16_value(weights[mm * k + kk]);
                let xv = f32_to_bf16_value(x[bb * k + kk]);
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
        "BF16 WMMA mismatch: max_abs={max_abs:.6e} rms={rms:.6e}"
    );
}
