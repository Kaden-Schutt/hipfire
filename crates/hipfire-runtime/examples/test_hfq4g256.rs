// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Correctness test for HFQ4-G256 quantize → GPU dequant round-trip.
//! Quantize known F32 values, upload, run GEMV, compare against F32 GEMV.

fn main() {
    let mut gpu = rdna_compute::Gpu::init().unwrap();

    // Small test: 4 rows × 256 cols (one group per row)
    let m = 4usize;
    let k = 256usize;

    // Known weights: row i has linearly spaced values from i*0.1 to i*0.1 + 1.0
    let mut f32_weights = vec![0.0f32; m * k];
    for row in 0..m {
        for col in 0..k {
            f32_weights[row * k + col] = row as f32 * 0.1 + col as f32 / k as f32;
        }
    }

    // Known input: all 1.0
    let x = vec![1.0f32; k];

    // CPU reference: y = W * x
    let mut y_ref = vec![0.0f32; m];
    for row in 0..m {
        for col in 0..k {
            y_ref[row] += f32_weights[row * k + col] * x[col];
        }
    }

    // Quantize to HFQ4-G256
    let quantized = quantize_hfq4g256(&f32_weights);
    eprintln!("Quantized {} floats → {} bytes", f32_weights.len(), quantized.len());

    // Verify quantized layout
    for row in 0..m {
        let off = row * 136;
        let scale = f32::from_le_bytes([quantized[off], quantized[off+1], quantized[off+2], quantized[off+3]]);
        let zero = f32::from_le_bytes([quantized[off+4], quantized[off+5], quantized[off+6], quantized[off+7]]);
        eprintln!("Row {}: scale={:.6}, zero={:.6}", row, scale, zero);

        // CPU dequant first 8 weights
        for i in 0..8 {
            let byte_idx = i / 2;
            let nibble = if i % 2 == 0 { quantized[off + 8 + byte_idx] & 0xF } else { quantized[off + 8 + byte_idx] >> 4 };
            let dequant = scale * nibble as f32 + zero;
            eprintln!("  w[{}][{}]: orig={:.4}, q={}, dequant={:.4}, err={:.4}",
                row, i, f32_weights[row * k + i], nibble, dequant, (dequant - f32_weights[row * k + i]).abs());
        }
    }

    // Upload to GPU
    let d_a = gpu.upload_raw(&quantized, &[quantized.len()]).unwrap();
    let d_x = gpu.upload_f32(&x, &[k]).unwrap();
    let d_y = gpu.zeros(&[m], rdna_compute::DType::F32).unwrap();

    // GPU GEMV
    gpu.gemv_hfq4g256(&d_a, &d_x, &d_y, m, k).unwrap();

    // Read back
    let mut y_gpu = vec![0.0f32; m];
    let y_bytes = unsafe { std::slice::from_raw_parts_mut(y_gpu.as_mut_ptr() as *mut u8, m * 4) };
    gpu.hip.memcpy_dtoh(y_bytes, &d_y.buf).unwrap();

    // Also do CPU dequant GEMV for comparison
    let mut y_cpu_dequant = vec![0.0f32; m];
    for row in 0..m {
        let off = row * 136;
        let scale = f32::from_le_bytes([quantized[off], quantized[off+1], quantized[off+2], quantized[off+3]]);
        let zero = f32::from_le_bytes([quantized[off+4], quantized[off+5], quantized[off+6], quantized[off+7]]);
        for i in 0..k {
            let byte_idx = i / 2;
            let nibble = if i % 2 == 0 { quantized[off + 8 + byte_idx] & 0xF } else { quantized[off + 8 + byte_idx] >> 4 };
            let w = scale * nibble as f32 + zero;
            y_cpu_dequant[row] += w * x[i];
        }
    }

    eprintln!("\n{:<6} {:>12} {:>12} {:>12} {:>12}", "Row", "F32 ref", "CPU dequant", "GPU dequant", "GPU-CPU err");
    for row in 0..m {
        let err = (y_gpu[row] - y_cpu_dequant[row]).abs();
        eprintln!("{:<6} {:>12.4} {:>12.4} {:>12.4} {:>12.6}",
            row, y_ref[row], y_cpu_dequant[row], y_gpu[row], err);
    }

    // Now test embedding lookup
    eprintln!("\n=== Embedding lookup test ===");
    let d_embd = gpu.upload_raw(&quantized, &[quantized.len()]).unwrap();
    let d_out = gpu.zeros(&[k], rdna_compute::DType::F32).unwrap();

    // Lookup row 2
    gpu.embedding_lookup_hfq4g256(&d_embd, &d_out, 2, k).unwrap();
    let mut embd_gpu = vec![0.0f32; k];
    let embd_bytes = unsafe { std::slice::from_raw_parts_mut(embd_gpu.as_mut_ptr() as *mut u8, k * 4) };
    gpu.hip.memcpy_dtoh(embd_bytes, &d_out.buf).unwrap();

    // Compare first 16 values
    let off = 2 * 136;
    let scale = f32::from_le_bytes([quantized[off], quantized[off+1], quantized[off+2], quantized[off+3]]);
    let zero = f32::from_le_bytes([quantized[off+4], quantized[off+5], quantized[off+6], quantized[off+7]]);
    eprintln!("{:<6} {:>10} {:>10} {:>10}", "Idx", "Original", "GPU", "Error");
    let mut max_err = 0.0f32;
    for i in 0..16 {
        let byte_idx = i / 2;
        let nibble = if i % 2 == 0 { quantized[off + 8 + byte_idx] & 0xF } else { quantized[off + 8 + byte_idx] >> 4 };
        let expected = scale * nibble as f32 + zero;
        let err = (embd_gpu[i] - expected).abs();
        max_err = max_err.max(err);
        eprintln!("{:<6} {:>10.4} {:>10.4} {:>10.6}", i, expected, embd_gpu[i], err);
    }
    eprintln!("Max embedding error: {:.6}", max_err);
}

fn quantize_hfq4g256(f32_data: &[f32]) -> Vec<u8> {
    let group_size = 256;
    let block_bytes = 136;
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
        let scale = if range > 0.0 { range / 15.0 } else { 1.0 };
        let inv_scale = if range > 0.0 { 1.0 / scale } else { 0.0 };

        let out_off = b * block_bytes;
        output[out_off..out_off + 4].copy_from_slice(&scale.to_le_bytes());
        output[out_off + 4..out_off + 8].copy_from_slice(&min_val.to_le_bytes());

        let actual_len = end - start;
        for i in 0..128 {
            let idx_lo = 2 * i;
            let idx_hi = 2 * i + 1;
            let lo_val = if idx_lo < actual_len { group[idx_lo] } else { min_val };
            let hi_val = if idx_hi < actual_len { group[idx_hi] } else { min_val };

            let lo_q = ((lo_val - min_val) * inv_scale + 0.5) as u8;
            let hi_q = ((hi_val - min_val) * inv_scale + 0.5) as u8;

            output[out_off + 8 + i] = lo_q.min(15) | (hi_q.min(15) << 4);
        }
    }

    output
}
