// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! QA mirror for the Q4_F16 GEMV harness.

use std::path::Path;
use std::process::ExitCode;

const SKIP_EXIT: u8 = 10;

fn main() -> ExitCode {
    match run() {
        Ok(msg) => {
            eprintln!("Q4F16 QA PASS: {msg}");
            ExitCode::SUCCESS
        }
        Err(Outcome::Skip(msg)) => {
            eprintln!("Q4F16 QA SKIP: {msg}");
            ExitCode::from(SKIP_EXIT)
        }
        Err(Outcome::Fail(msg)) => {
            eprintln!("Q4F16 QA FAIL: {msg}");
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

    let path = std::env::var("TINYLLAMA_GGUF")
        .unwrap_or_else(|_| "/home/kaden/llama.cpp/models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf".to_string());
    if !Path::new(&path).exists() {
        return Err(Outcome::Skip(format!("GGUF not found at {path}")));
    }

    let gguf = hipfire_runtime::gguf::GgufFile::open(Path::new(&path))
        .map_err(|e| Outcome::Fail(format!("failed to open GGUF: {e}")))?;
    let tensor_info = gguf.find_tensor("blk.0.attn_q.weight")
        .ok_or_else(|| Outcome::Fail("tensor blk.0.attn_q.weight not found".to_string()))?;
    let raw_q4k = gguf.tensor_data(tensor_info);
    let m = 2048usize;
    let k = 2048usize;
    let n_elements = m * k;

    let a_f32 = hipfire_runtime::llama::dequantize_q4_k(raw_q4k, n_elements);
    let x_data: Vec<f32> = (0..k).map(|i| ((i % 7) as f32 - 3.0) * 0.01).collect();
    let mut y_ref = vec![0.0f32; m];
    for i in 0..m {
        for j in 0..k {
            y_ref[i] += a_f32[i * k + j] * x_data[j];
        }
    }

    let d_x = gpu.upload_f32(&x_data, &[k]).map_err(|e| Outcome::Fail(format!("upload x failed: {e}")))?;
    let q4f16_g32 = hipfire_runtime::llama::convert_q4k_to_q4f16_g32(raw_q4k, n_elements);
    let q4f16_g64 = hipfire_runtime::llama::convert_q4k_to_q4f16_g64(raw_q4k, n_elements);
    let d_g32 = gpu.upload_raw(&q4f16_g32, &[q4f16_g32.len()]).map_err(|e| Outcome::Fail(format!("upload g32 failed: {e}")))?;
    let d_g64 = gpu.upload_raw(&q4f16_g64, &[q4f16_g64.len()]).map_err(|e| Outcome::Fail(format!("upload g64 failed: {e}")))?;
    let d_y = gpu.zeros(&[m], rdna_compute::DType::F32).map_err(|e| Outcome::Fail(format!("alloc y failed: {e}")))?;

    gpu.gemv_q4f16_g32(&d_g32, &d_x, &d_y, m, k).map_err(|e| Outcome::Fail(format!("gemv_q4f16_g32 failed: {e}")))?;
    let y_g32 = gpu.download_f32(&d_y).map_err(|e| Outcome::Fail(format!("download g32 failed: {e}")))?;
    let max_abs_g32 = y_g32.iter().zip(&y_ref).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
    if max_abs_g32 >= 0.05 || y_g32.iter().any(|v| !v.is_finite()) {
        return Err(Outcome::Fail(format!("G32 validation failed: max_abs={max_abs_g32:.6}")));
    }

    gpu.gemv_q4f16_g64(&d_g64, &d_x, &d_y, m, k).map_err(|e| Outcome::Fail(format!("gemv_q4f16_g64 failed: {e}")))?;
    let y_g64 = gpu.download_f32(&d_y).map_err(|e| Outcome::Fail(format!("download g64 failed: {e}")))?;
    let max_abs_g64 = y_g64.iter().zip(&y_ref).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
    if max_abs_g64 >= 0.1 || y_g64.iter().any(|v| !v.is_finite()) {
        return Err(Outcome::Fail(format!("G64 validation failed: max_abs={max_abs_g64:.6}")));
    }

    gpu.free_tensor(d_x).map_err(|e| Outcome::Fail(format!("free x failed: {e}")))?;
    gpu.free_tensor(d_g32).map_err(|e| Outcome::Fail(format!("free g32 failed: {e}")))?;
    gpu.free_tensor(d_g64).map_err(|e| Outcome::Fail(format!("free g64 failed: {e}")))?;
    gpu.free_tensor(d_y).map_err(|e| Outcome::Fail(format!("free y failed: {e}")))?;

    Ok(format!("G32 max_abs={max_abs_g32:.6} G64 max_abs={max_abs_g64:.6}"))
}
