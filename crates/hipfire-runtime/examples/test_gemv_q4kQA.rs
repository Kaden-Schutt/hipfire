// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! QA mirror for the Q4_K GEMV harness.

use std::path::Path;
use std::process::ExitCode;

const SKIP_EXIT: u8 = 10;

fn main() -> ExitCode {
    match run() {
        Ok(msg) => {
            eprintln!("Q4K GEMV QA PASS: {msg}");
            ExitCode::SUCCESS
        }
        Err(Outcome::Skip(msg)) => {
            eprintln!("Q4K GEMV QA SKIP: {msg}");
            ExitCode::from(SKIP_EXIT)
        }
        Err(Outcome::Fail(msg)) => {
            eprintln!("Q4K GEMV QA FAIL: {msg}");
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
    let tensor = gguf.find_tensor("blk.0.attn_q.weight")
        .ok_or_else(|| Outcome::Fail("tensor blk.0.attn_q.weight not found".to_string()))?;
    let raw = gguf.tensor_data(tensor);
    let m = 2048usize;
    let k = 2048usize;

    let d_raw = gpu.upload_raw(raw, &[raw.len()]).map_err(|e| Outcome::Fail(format!("upload raw failed: {e}")))?;
    let x_data: Vec<f32> = (0..k).map(|i| ((i % 7) as f32 - 3.0) * 0.01).collect();
    let d_x = gpu.upload_f32(&x_data, &[k]).map_err(|e| Outcome::Fail(format!("upload x failed: {e}")))?;
    let d_y = gpu.zeros(&[m], rdna_compute::DType::F32).map_err(|e| Outcome::Fail(format!("alloc y failed: {e}")))?;

    gpu.gemv_q4k(&d_raw, &d_x, &d_y, m, k).map_err(|e| Outcome::Fail(format!("gemv_q4k failed: {e}")))?;
    let y_gpu = gpu.download_f32(&d_y).map_err(|e| Outcome::Fail(format!("download y failed: {e}")))?;
    if y_gpu.iter().any(|v| !v.is_finite()) {
        return Err(Outcome::Fail("GPU output contains non-finite values".to_string()));
    }

    let a_f32 = hipfire_runtime::llama::dequantize_q4_k(raw, m * k);
    let mut y_ref = vec![0.0f32; m];
    for i in 0..m {
        for j in 0..k {
            y_ref[i] += a_f32[i * k + j] * x_data[j];
        }
    }
    let max_abs_err = y_gpu.iter().zip(&y_ref).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
    if max_abs_err >= 0.05 {
        return Err(Outcome::Fail(format!("max_abs_err too large: {max_abs_err:.6}")));
    }

    gpu.free_tensor(d_raw).map_err(|e| Outcome::Fail(format!("free raw failed: {e}")))?;
    gpu.free_tensor(d_x).map_err(|e| Outcome::Fail(format!("free x failed: {e}")))?;
    gpu.free_tensor(d_y).map_err(|e| Outcome::Fail(format!("free y failed: {e}")))?;

    Ok(format!("tensor={} max_abs_err={max_abs_err:.6}", tensor.name))
}
