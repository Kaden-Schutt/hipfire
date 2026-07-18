// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
//! Cross-architecture channel test for the LFM2.5-8B-A1B routed-expert down kernel.
//!
//! The production shape is intentional: K=1792 spans seven HFQ4-G256 groups,
//! which catches fixed-K=512 kernels that silently consume only two groups.

use hipfire_arch_lfm2moe::kernels::{
    lfm2_a1b_moe_down, LFM2_A1B_DOWN_ARCHES, LFM2_A1B_HIDDEN, LFM2_A1B_MOE_INTERMEDIATE,
    LFM2_A1B_TOP_K,
};
use rdna_compute::{DType, Gpu};
use std::process::ExitCode;

const GROUP: usize = 256;
const BLOCK_BYTES: usize = 136;
const NRMSE_LIMIT: f32 = 1.0e-4;

fn main() -> ExitCode {
    match run() {
        Ok(nrmse) => {
            eprintln!("LFM2 A1B MOE DOWN PASS nrmse={nrmse:.8}");
            ExitCode::SUCCESS
        }
        Err(TestError::Skip(message)) => {
            eprintln!("LFM2 A1B MOE DOWN SKIP: {message}");
            ExitCode::from(10)
        }
        Err(TestError::Fail(message)) => {
            eprintln!("LFM2 A1B MOE DOWN FAIL: {message}");
            ExitCode::FAILURE
        }
    }
}

enum TestError {
    Skip(String),
    Fail(String),
}

fn run() -> Result<f32, TestError> {
    let mut gpu = Gpu::init().map_err(|e| TestError::Skip(format!("GPU init unavailable: {e}")))?;
    if !LFM2_A1B_DOWN_ARCHES.contains(&gpu.arch.as_str()) {
        return Err(TestError::Skip(format!(
            "requires one of {}, got {}",
            LFM2_A1B_DOWN_ARCHES.join(", "),
            gpu.arch
        )));
    }

    let m = LFM2_A1B_HIDDEN;
    let k = LFM2_A1B_MOE_INTERMEDIATE;
    let k_top = LFM2_A1B_TOP_K;
    let selected = [3_i32, 1, 2, 0];

    let quantized: Vec<Vec<u8>> = (0..k_top)
        .map(|expert| make_weights(expert, m, k))
        .collect();
    let device_experts = quantized
        .iter()
        .map(|bytes| {
            gpu.upload_raw(bytes, &[bytes.len()])
                .map_err(|e| TestError::Fail(format!("upload expert: {e}")))
        })
        .collect::<Result<Vec<_>, _>>()?;
    let expert_ptrs: Vec<u64> = device_experts
        .iter()
        .map(|tensor| tensor.buf.as_ptr() as usize as u64)
        .collect();
    let d_expert_ptrs = gpu
        .upload_raw(as_bytes(&expert_ptrs), &[expert_ptrs.len()])
        .map_err(|e| TestError::Fail(format!("upload expert pointers: {e}")))?;
    let d_topk = gpu
        .upload_raw(as_bytes(&selected), &[selected.len()])
        .map_err(|e| TestError::Fail(format!("upload top-k indices: {e}")))?;

    let mut x = vec![0.0_f32; k_top * k];
    for rank in 0..k_top {
        for col in 0..k {
            x[rank * k + col] = (((rank * 29 + col * 17) % 67) as f32 - 33.0) * 0.003;
        }
    }
    let d_x = gpu
        .upload_f32(&x, &[x.len()])
        .map_err(|e| TestError::Fail(format!("upload activations: {e}")))?;
    let d_out = gpu
        .zeros(&[k_top * m], DType::F32)
        .map_err(|e| TestError::Fail(format!("allocate output: {e}")))?;

    lfm2_a1b_moe_down(
        &mut gpu,
        &d_expert_ptrs,
        &d_topk,
        &d_x,
        &d_out,
        DType::MQ4G256,
        m,
        k,
        k_top,
    )
    .map_err(TestError::Fail)?;
    let got = gpu
        .download_f32(&d_out)
        .map_err(|e| TestError::Fail(format!("download output: {e}")))?;

    let mut expected = vec![0.0_f32; k_top * m];
    for rank in 0..k_top {
        let expert = selected[rank] as usize;
        cpu_down(
            &quantized[expert],
            &x[rank * k..(rank + 1) * k],
            &mut expected[rank * m..(rank + 1) * m],
            m,
            k,
        );
    }
    let nrmse = nrmse(&got, &expected);
    if !nrmse.is_finite() || nrmse > NRMSE_LIMIT {
        return Err(TestError::Fail(format!(
            "production K=1792 NRMSE {nrmse:.8} is non-finite or exceeds {NRMSE_LIMIT:.8}"
        )));
    }
    Ok(nrmse)
}

fn make_weights(expert: usize, m: usize, k: usize) -> Vec<u8> {
    let groups = k / GROUP;
    let mut out = vec![0_u8; m * groups * BLOCK_BYTES];
    for row in 0..m {
        for group in 0..groups {
            let offset = (row * groups + group) * BLOCK_BYTES;
            let scale = 0.00625 * (1 + ((expert + row + group) % 5)) as f32;
            let zero = -0.03125 + 0.003 * ((expert + group) % 3) as f32;
            out[offset..offset + 4].copy_from_slice(&scale.to_le_bytes());
            out[offset + 4..offset + 8].copy_from_slice(&zero.to_le_bytes());
            for byte in 0..128 {
                let col = byte * 2;
                let lo = ((expert * 11 + row * 7 + group * 5 + col * 3) & 0xF) as u8;
                let hi = ((expert * 13 + row * 3 + group * 7 + (col + 1) * 5) & 0xF) as u8;
                out[offset + 8 + byte] = lo | (hi << 4);
            }
        }
    }
    out
}

fn cpu_down(weights: &[u8], x: &[f32], out: &mut [f32], m: usize, k: usize) {
    let groups = k / GROUP;
    for (row, dst) in out.iter_mut().enumerate().take(m) {
        let mut acc = 0.0_f32;
        for group in 0..groups {
            let offset = (row * groups + group) * BLOCK_BYTES;
            let scale = f32::from_le_bytes(weights[offset..offset + 4].try_into().unwrap());
            let zero = f32::from_le_bytes(weights[offset + 4..offset + 8].try_into().unwrap());
            for col in 0..GROUP {
                let packed = weights[offset + 8 + col / 2];
                let q = if col & 1 == 0 {
                    packed & 0xF
                } else {
                    packed >> 4
                };
                acc += (scale * q as f32 + zero) * x[group * GROUP + col];
            }
        }
        *dst = acc;
    }
}

fn nrmse(got: &[f32], expected: &[f32]) -> f32 {
    let n = got.len() as f32;
    let mse = got
        .iter()
        .zip(expected)
        .map(|(a, b)| {
            let d = a - b;
            d * d
        })
        .sum::<f32>()
        / n;
    let reference_ms = expected.iter().map(|v| v * v).sum::<f32>() / n;
    mse.sqrt() / (reference_ms.sqrt() + 1.0e-8)
}

fn as_bytes<T>(values: &[T]) -> &[u8] {
    // SAFETY: the slice remains alive for the synchronous upload call and its
    // exact initialized byte representation is copied without reinterpretation.
    unsafe {
        std::slice::from_raw_parts(values.as_ptr().cast::<u8>(), std::mem::size_of_val(values))
    }
}
