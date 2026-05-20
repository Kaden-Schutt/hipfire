// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! QA mirror for Q8 KV cache validation.

use std::process::ExitCode;

const SKIP_EXIT: u8 = 10;

fn main() -> ExitCode {
    match run() {
        Ok(msg) => {
            eprintln!("Q8 KV QA PASS: {msg}");
            ExitCode::SUCCESS
        }
        Err(Outcome::Skip(msg)) => {
            eprintln!("Q8 KV QA SKIP: {msg}");
            ExitCode::from(SKIP_EXIT)
        }
        Err(Outcome::Fail(msg)) => {
            eprintln!("Q8 KV QA FAIL: {msg}");
            ExitCode::from(1)
        }
    }
}

enum Outcome {
    Skip(String),
    Fail(String),
}

fn run() -> Result<String, Outcome> {
    let mut gpu = rdna_compute::Gpu::init()
        .map_err(|e| Outcome::Skip(format!("GPU init unavailable: {e}")))?;

    let pos_buf = gpu.hip.malloc(4).map_err(|e| Outcome::Fail(format!("malloc pos buffer failed: {e}")))?;

    let head_dim = 32usize;
    let max_seq = 4usize;
    let kv_data: Vec<f32> = (0..head_dim).map(|i| 0.1 * (i + 1) as f32).collect();
    let d_src = gpu.upload_f32(&kv_data, &[head_dim]).map_err(|e| Outcome::Fail(format!("upload source failed: {e}")))?;
    let total_blocks = head_dim / 32;
    let cache_bytes = max_seq * total_blocks * 34;
    let cache_elems = (cache_bytes + 3) / 4;
    let d_cache = gpu.zeros(&[cache_elems], rdna_compute::DType::F32).map_err(|e| Outcome::Fail(format!("alloc cache failed: {e}")))?;

    gpu.hip.memcpy_htod(&pos_buf, &0i32.to_ne_bytes()).map_err(|e| Outcome::Fail(format!("write pos failed: {e}")))?;
    gpu.kv_cache_write_q8_0(&d_cache, &d_src, &pos_buf, 1, head_dim).map_err(|e| Outcome::Fail(format!("kv write failed: {e}")))?;
    gpu.hip.device_synchronize().map_err(|e| Outcome::Fail(format!("device sync failed: {e}")))?;

    let mut raw = vec![0u8; 34];
    gpu.hip.memcpy_dtoh(&mut raw, &d_cache.buf).map_err(|e| Outcome::Fail(format!("readback failed: {e}")))?;
    let scale_bits = u16::from_le_bytes([raw[0], raw[1]]);
    let scale = f16_to_f32(scale_bits);

    let mut max_err = 0.0f32;
    for i in 0..head_dim {
        let q = raw[2 + i] as i8;
        let dequant = scale * q as f32;
        max_err = max_err.max((kv_data[i] - dequant).abs());
    }
    if max_err >= 0.05 {
        return Err(Outcome::Fail(format!("single block roundtrip error too large: {max_err:.6}")));
    }

    let d_q = gpu.upload_f32(&vec![1.0f32; head_dim], &[head_dim]).map_err(|e| Outcome::Fail(format!("upload q failed: {e}")))?;
    let d_out = gpu.zeros(&[head_dim], rdna_compute::DType::F32).map_err(|e| Outcome::Fail(format!("alloc out failed: {e}")))?;
    gpu.attention_q8_0_kv(&d_q, &d_cache, &d_cache, &d_out, &pos_buf, 1, 1, 1, head_dim, max_seq)
        .map_err(|e| Outcome::Fail(format!("attention_q8_0_kv failed: {e}")))?;
    let out = gpu.download_f32(&d_out).map_err(|e| Outcome::Fail(format!("download out failed: {e}")))?;
    if !out[0].is_finite() {
        return Err(Outcome::Fail(format!("attention output is non-finite: {}", out[0])));
    }

    gpu.hip.free(pos_buf).map_err(|e| Outcome::Fail(format!("free pos buffer failed: {e}")))?;
    gpu.free_tensor(d_src).map_err(|e| Outcome::Fail(format!("free src failed: {e}")))?;
    gpu.free_tensor(d_cache).map_err(|e| Outcome::Fail(format!("free cache failed: {e}")))?;
    gpu.free_tensor(d_q).map_err(|e| Outcome::Fail(format!("free q failed: {e}")))?;
    gpu.free_tensor(d_out).map_err(|e| Outcome::Fail(format!("free out failed: {e}")))?;

    Ok(format!("max_roundtrip_err={max_err:.6} attention_out0={:.4}", out[0]))
}

fn f16_to_f32(bits: u16) -> f32 {
    let sign = ((bits >> 15) & 1) as u32;
    let exp = ((bits >> 10) & 0x1F) as i32;
    let frac = (bits & 0x3FF) as u32;
    if exp == 0 {
        if frac == 0 {
            return if sign == 1 { -0.0 } else { 0.0 };
        }
        let v = (frac as f32) / 1024.0 * 2.0f32.powi(-14);
        return if sign == 1 { -v } else { v };
    }
    if exp == 31 {
        return if frac == 0 {
            if sign == 1 { f32::NEG_INFINITY } else { f32::INFINITY }
        } else {
            f32::NAN
        };
    }
    let v = 2.0f32.powi(exp - 15) * (1.0 + frac as f32 / 1024.0);
    if sign == 1 { -v } else { v }
}
