// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Correctness test for HFQ6-G256 quantize → GPU dequant round-trip.
//! Quantize known F32 values, upload, run GEMV, compare against F32 GEMV.

fn main() {
    let mut gpu = rdna_compute::Gpu::init().unwrap();

    let m = 4usize;
    let k = 256usize;

    // Known weights
    let mut f32_weights = vec![0.0f32; m * k];
    for row in 0..m {
        for col in 0..k {
            f32_weights[row * k + col] = row as f32 * 0.1 + col as f32 / k as f32;
        }
    }

    let x = vec![1.0f32; k];

    // CPU reference: y = W * x
    let mut y_ref = vec![0.0f32; m];
    for row in 0..m {
        for col in 0..k {
            y_ref[row] += f32_weights[row * k + col] * x[col];
        }
    }

    // Quantize to HFQ6-G256
    let quantized = quantize_hfq6g256(&f32_weights);
    eprintln!("Quantized {} floats → {} bytes (expect {})", f32_weights.len(), quantized.len(), m * 200);

    // Verify scale/zero
    for row in 0..m {
        let off = row * 200;
        let scale = f32::from_le_bytes([quantized[off], quantized[off+1], quantized[off+2], quantized[off+3]]);
        let zero = f32::from_le_bytes([quantized[off+4], quantized[off+5], quantized[off+6], quantized[off+7]]);
        eprintln!("Row {}: scale={:.6}, zero={:.6}", row, scale, zero);

        // CPU dequant first 8 weights (2 groups of 4 from 6 bytes)
        for chunk in 0..2 {
            let bo = off + 8 + chunk * 3;
            let b0 = quantized[bo] as u32;
            let b1 = quantized[bo + 1] as u32;
            let b2 = quantized[bo + 2] as u32;
            let q0 = (b0 & 0x3F) as f32;
            let q1 = (((b0 >> 6) | (b1 << 2)) & 0x3F) as f32;
            let q2 = (((b1 >> 4) | (b2 << 4)) & 0x3F) as f32;
            let q3 = ((b2 >> 2) & 0x3F) as f32;
            let base = chunk * 4;
            for (i, q) in [q0, q1, q2, q3].iter().enumerate() {
                let dequant = scale * q + zero;
                let orig = f32_weights[row * k + base + i];
                eprintln!("  w[{}][{}]: orig={:.4}, q={:.0}, dequant={:.4}, err={:.4}",
                    row, base + i, orig, q, dequant, (dequant - orig).abs());
            }
        }
    }

    // Upload to GPU
    let d_a = gpu.upload_raw(&quantized, &[quantized.len()]).unwrap();
    let d_x = gpu.upload_f32(&x, &[k]).unwrap();
    let d_y = gpu.zeros(&[m], rdna_compute::DType::F32).unwrap();

    // GPU GEMV
    gpu.gemv_hfq6g256(&d_a, &d_x, &d_y, m, k).unwrap();

    let mut y_gpu = vec![0.0f32; m];
    let y_bytes = unsafe { std::slice::from_raw_parts_mut(y_gpu.as_mut_ptr() as *mut u8, m * 4) };
    gpu.hip.memcpy_dtoh(y_bytes, &d_y.buf).unwrap();

    // CPU dequant GEMV
    let mut y_cpu = vec![0.0f32; m];
    for row in 0..m {
        let off = row * 200;
        let scale = f32::from_le_bytes([quantized[off], quantized[off+1], quantized[off+2], quantized[off+3]]);
        let zero = f32::from_le_bytes([quantized[off+4], quantized[off+5], quantized[off+6], quantized[off+7]]);
        for i in (0..k).step_by(4) {
            let bo = off + 8 + (i / 4) * 3;
            let b0 = quantized[bo] as u32;
            let b1 = quantized[bo + 1] as u32;
            let b2 = quantized[bo + 2] as u32;
            let q0 = (b0 & 0x3F) as f32;
            let q1 = (((b0 >> 6) | (b1 << 2)) & 0x3F) as f32;
            let q2 = (((b1 >> 4) | (b2 << 4)) & 0x3F) as f32;
            let q3 = ((b2 >> 2) & 0x3F) as f32;
            y_cpu[row] += (scale * q0 + zero) * x[i];
            y_cpu[row] += (scale * q1 + zero) * x[i + 1];
            y_cpu[row] += (scale * q2 + zero) * x[i + 2];
            y_cpu[row] += (scale * q3 + zero) * x[i + 3];
        }
    }

    eprintln!("\n{:<6} {:>12} {:>12} {:>12} {:>12}", "Row", "F32 ref", "CPU dequant", "GPU dequant", "GPU-CPU err");
    let mut max_err = 0.0f32;
    for row in 0..m {
        let err = (y_gpu[row] - y_cpu[row]).abs();
        max_err = max_err.max(err);
        eprintln!("{:<6} {:>12.4} {:>12.4} {:>12.4} {:>12.6}",
            row, y_ref[row], y_cpu[row], y_gpu[row], err);
    }
    eprintln!("\nMax GPU-CPU error: {:.6}", max_err);
    if max_err > 0.1 {
        eprintln!("FAIL: GPU-CPU error too large!");
        std::process::exit(1);
    }
    eprintln!("PASS");
}

fn quantize_hfq6g256(f32_data: &[f32]) -> Vec<u8> {
    let group_size = 256;
    let block_bytes = 200;
    let n = f32_data.len();
    let n_blocks = (n + group_size - 1) / group_size;
    let mut output = vec![0u8; n_blocks * block_bytes];

    for b in 0..n_blocks {
        let start = b * group_size;
        let end = (start + group_size).min(n);
        let group = &f32_data[start..end];

        let min_val = group.iter().cloned().fold(f32::INFINITY, f32::min);
        let max_val = group.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let range = max_val - min_val;
        let scale = if range > 0.0 { range / 63.0 } else { 1.0 };
        let inv_scale = if range > 0.0 { 1.0 / scale } else { 0.0 };

        let out_off = b * block_bytes;
        output[out_off..out_off + 4].copy_from_slice(&scale.to_le_bytes());
        output[out_off + 4..out_off + 8].copy_from_slice(&min_val.to_le_bytes());

        let actual_len = end - start;
        for i in (0..256).step_by(4) {
            let q0 = if i < actual_len { ((group[i] - min_val) * inv_scale + 0.5) as u8 } else { 0 };
            let q1 = if i + 1 < actual_len { ((group[i+1] - min_val) * inv_scale + 0.5) as u8 } else { 0 };
            let q2 = if i + 2 < actual_len { ((group[i+2] - min_val) * inv_scale + 0.5) as u8 } else { 0 };
            let q3 = if i + 3 < actual_len { ((group[i+3] - min_val) * inv_scale + 0.5) as u8 } else { 0 };
            let q0 = q0.min(63);
            let q1 = q1.min(63);
            let q2 = q2.min(63);
            let q3 = q3.min(63);

            let byte_off = 8 + (i / 4) * 3;
            output[out_off + byte_off]     = q0 | (q1 << 6);
            output[out_off + byte_off + 1] = (q1 >> 2) | (q2 << 4);
            output[out_off + byte_off + 2] = (q2 >> 4) | (q3 << 2);
        }
    }
    output
}
