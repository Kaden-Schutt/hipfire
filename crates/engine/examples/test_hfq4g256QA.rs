//! QA mirror for HFQ4-G256 correctness checks.

use std::process::ExitCode;

const SKIP_EXIT: u8 = 10;

fn main() -> ExitCode {
    match run() {
        Ok(msg) => {
            eprintln!("HFQ4G256 QA PASS: {msg}");
            ExitCode::SUCCESS
        }
        Err(Outcome::Skip(msg)) => {
            eprintln!("HFQ4G256 QA SKIP: {msg}");
            ExitCode::from(SKIP_EXIT)
        }
        Err(Outcome::Fail(msg)) => {
            eprintln!("HFQ4G256 QA FAIL: {msg}");
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

    let m = 4usize;
    let k = 256usize;
    let mut weights = vec![0.0f32; m * k];
    for row in 0..m {
        for col in 0..k {
            weights[row * k + col] = row as f32 * 0.1 + col as f32 / k as f32;
        }
    }

    let x = vec![1.0f32; k];
    let quantized = quantize_hfq4g256(&weights);

    let mut y_ref = vec![0.0f32; m];
    for row in 0..m {
        for col in 0..k {
            y_ref[row] += weights[row * k + col] * x[col];
        }
    }

    let d_a = gpu.upload_raw(&quantized, &[quantized.len()]).map_err(|e| Outcome::Fail(format!("upload quantized failed: {e}")))?;
    let d_x = gpu.upload_f32(&x, &[k]).map_err(|e| Outcome::Fail(format!("upload x failed: {e}")))?;
    let d_y = gpu.zeros(&[m], rdna_compute::DType::F32).map_err(|e| Outcome::Fail(format!("alloc y failed: {e}")))?;

    gpu.gemv_hfq4g256(&d_a, &d_x, &d_y, m, k).map_err(|e| Outcome::Fail(format!("gemv_hfq4g256 failed: {e}")))?;
    let y_gpu = gpu.download_f32(&d_y).map_err(|e| Outcome::Fail(format!("download y failed: {e}")))?;

    let mut y_cpu_dequant = vec![0.0f32; m];
    for row in 0..m {
        let off = row * 136;
        let scale = f32::from_le_bytes([quantized[off], quantized[off + 1], quantized[off + 2], quantized[off + 3]]);
        let zero = f32::from_le_bytes([quantized[off + 4], quantized[off + 5], quantized[off + 6], quantized[off + 7]]);
        for i in 0..k {
            let byte_idx = i / 2;
            let nibble = if i % 2 == 0 { quantized[off + 8 + byte_idx] & 0xF } else { quantized[off + 8 + byte_idx] >> 4 };
            y_cpu_dequant[row] += (scale * nibble as f32 + zero) * x[i];
        }
    }

    let max_gpu_cpu_err = y_gpu.iter().zip(&y_cpu_dequant).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
    if max_gpu_cpu_err > 0.05 {
        return Err(Outcome::Fail(format!("GPU vs CPU dequant mismatch too large: {max_gpu_cpu_err:.6}")));
    }

    let d_out = gpu.zeros(&[k], rdna_compute::DType::F32).map_err(|e| Outcome::Fail(format!("alloc embedding out failed: {e}")))?;
    gpu.embedding_lookup_hfq4g256(&d_a, &d_out, 2, k).map_err(|e| Outcome::Fail(format!("embedding lookup failed: {e}")))?;
    let emb = gpu.download_f32(&d_out).map_err(|e| Outcome::Fail(format!("download embedding failed: {e}")))?;
    if !emb[0].is_finite() {
        return Err(Outcome::Fail(format!("embedding output non-finite: {}", emb[0])));
    }

    gpu.free_tensor(d_a).map_err(|e| Outcome::Fail(format!("free A failed: {e}")))?;
    gpu.free_tensor(d_x).map_err(|e| Outcome::Fail(format!("free x failed: {e}")))?;
    gpu.free_tensor(d_y).map_err(|e| Outcome::Fail(format!("free y failed: {e}")))?;
    gpu.free_tensor(d_out).map_err(|e| Outcome::Fail(format!("free out failed: {e}")))?;

    let mmq_result = if supports_mmq_i8_wmma(&gpu.arch) {
        let mmq_m = 128usize;
        let mmq_n = 128usize;
        let mmq_k = 256usize;
        let mut mmq_w = vec![0.0f32; mmq_m * mmq_k];
        for row in 0..mmq_m {
            for col in 0..mmq_k {
                mmq_w[row * mmq_k + col] = (row as f32 * 0.007) - 0.5 + (col as f32 * 0.003).sin();
            }
        }
        let mut mmq_x = vec![0.0f32; mmq_n * mmq_k];
        for n in 0..mmq_n {
            for col in 0..mmq_k {
                mmq_x[n * mmq_k + col] = ((n * 17 + col * 13) as f32 * 0.01).sin();
            }
        }
        let mmq_q = quantize_hfq4g256(&mmq_w);
        let mut mmq_ref = vec![0.0f32; mmq_n * mmq_m];
        for n in 0..mmq_n {
            for row in 0..mmq_m {
                let off = row * 136;
                let scale = f32::from_le_bytes([mmq_q[off], mmq_q[off + 1], mmq_q[off + 2], mmq_q[off + 3]]);
                let zero = f32::from_le_bytes([mmq_q[off + 4], mmq_q[off + 5], mmq_q[off + 6], mmq_q[off + 7]]);
                let mut acc = 0.0f32;
                for col in 0..mmq_k {
                    let byte_idx = col / 2;
                    let nibble = if col % 2 == 0 { mmq_q[off + 8 + byte_idx] & 0xF } else { mmq_q[off + 8 + byte_idx] >> 4 };
                    acc += (scale * nibble as f32 + zero) * mmq_x[n * mmq_k + col];
                }
                mmq_ref[n * mmq_m + row] = acc;
            }
        }

        let d_mmq_a = gpu.upload_raw(&mmq_q, &[mmq_q.len()]).map_err(|e| Outcome::Fail(format!("upload mmq A failed: {e}")))?;
        let d_mmq_x = gpu.upload_f32(&mmq_x, &[mmq_n * mmq_k]).map_err(|e| Outcome::Fail(format!("upload mmq X failed: {e}")))?;
        let d_mmq_y = gpu.zeros(&[mmq_n * mmq_m], rdna_compute::DType::F32).map_err(|e| Outcome::Fail(format!("alloc mmq Y failed: {e}")))?;
        gpu.gemm_hfq4g256_residual_mmq(&d_mmq_a, &d_mmq_x, &d_mmq_y, mmq_m, mmq_k, mmq_n)
            .map_err(|e| Outcome::Fail(format!("gemm_hfq4g256_residual_mmq failed: {e}")))?;
        let mmq_gpu = gpu.download_f32(&d_mmq_y).map_err(|e| Outcome::Fail(format!("download mmq Y failed: {e}")))?;
        let max_mmq_err = mmq_gpu.iter().zip(&mmq_ref).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        if max_mmq_err > 0.5 {
            return Err(Outcome::Fail(format!("MMQ residual mismatch too large: {max_mmq_err:.6} gpu0={} ref0={}", mmq_gpu[0], mmq_ref[0])));
        }
        gpu.free_tensor(d_mmq_a).map_err(|e| Outcome::Fail(format!("free mmq A failed: {e}")))?;
        gpu.free_tensor(d_mmq_x).map_err(|e| Outcome::Fail(format!("free mmq X failed: {e}")))?;
        gpu.free_tensor(d_mmq_y).map_err(|e| Outcome::Fail(format!("free mmq Y failed: {e}")))?;
        format!("mmq_err={max_mmq_err:.6}")
    } else {
        format!("mmq_skipped_arch={}", gpu.arch)
    };

    let max_ref_err = y_cpu_dequant.iter().zip(&y_ref).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
    Ok(format!("gpu_cpu_err={max_gpu_cpu_err:.6} quant_ref_err={max_ref_err:.6} {mmq_result}"))
}

fn supports_mmq_i8_wmma(arch: &str) -> bool {
    matches!(arch, "gfx1100" | "gfx1101" | "gfx1102" | "gfx1103" | "gfx1150" | "gfx1151")
}

fn quantize_hfq4g256(f32_data: &[f32]) -> Vec<u8> {
    let group_size = 256usize;
    let block_bytes = 136usize;
    let n_blocks = (f32_data.len() + group_size - 1) / group_size;
    let mut out = vec![0u8; n_blocks * block_bytes];

    for b in 0..n_blocks {
        let start = b * group_size;
        let end = (start + group_size).min(f32_data.len());
        let group = &f32_data[start..end];
        let min_val = group.iter().copied().fold(f32::INFINITY, f32::min);
        let max_val = group.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let range = max_val - min_val;
        let scale = if range > 0.0 { range / 15.0 } else { 1.0 };
        let inv_scale = if range > 0.0 { 1.0 / scale } else { 0.0 };
        let off = b * block_bytes;
        out[off..off + 4].copy_from_slice(&scale.to_le_bytes());
        out[off + 4..off + 8].copy_from_slice(&min_val.to_le_bytes());

        let actual_len = end - start;
        for i in 0..128 {
            let lo_idx = 2 * i;
            let hi_idx = 2 * i + 1;
            let lo_val = if lo_idx < actual_len { group[lo_idx] } else { min_val };
            let hi_val = if hi_idx < actual_len { group[hi_idx] } else { min_val };
            let lo_q = ((lo_val - min_val) * inv_scale + 0.5) as u8;
            let hi_q = ((hi_val - min_val) * inv_scale + 0.5) as u8;
            out[off + 8 + i] = lo_q.min(15) | (hi_q.min(15) << 4);
        }
    }

    out
}
