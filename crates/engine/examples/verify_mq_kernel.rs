//! Q0 — Bit-exact kernel verification for MQ3/MQ2 GEMV pipelines.
//!
//! Generates synthetic weights, quantizes to MQ3 and MQ2, then compares
//! the GPU `gemv_mq{3,2}g256_with_rotate` output against a CPU reference
//! that reconstructs the rotated weights and rotates x using the same
//! FWHT math as the quantizer.
//!
//! Run:
//!   cargo run --release --example verify_mq_kernel
//!
//! Acceptance (per Q0):
//!   max_abs_err <= 1e-3  => kernel is correct; close Q0.
//!   max_abs_err > 1e-3   => kernel has a bug; investigate.

fn main() {
    let mut gpu = rdna_compute::Gpu::init().unwrap();

    // Test multiple shapes: 1-group, 2-group, 4-group rows.
    let shapes = [(4usize, 256usize), (4, 512), (8, 1024)];

    let mut any_fail = false;

    for &(m, k) in &shapes {
        eprintln!("\n========== shape {} x {} ==========", m, k);

        // Deterministic pseudo-random weights and input (reproducible across runs)
        let f32_weights: Vec<f32> = (0..m * k).map(|i| fract_sin(i as f32 * 0.731f32 + 1.337f32)).collect();
        let x: Vec<f32> = (0..k).map(|i| fract_sin(i as f32 * 0.513f32 + 2.719f32)).collect();

        // ---- MQ3 ----
        let mq3_bytes = quantize_mq3g256(&f32_weights, k);
        let y_mq3_cpu = cpu_reference_mq(&mq3_bytes, &x, m, k, 104, 3, |scale, zero, q| scale * q as f32 + zero);
        let y_mq3_gpu = gpu_mq_gemv(&mut gpu, &mq3_bytes, &x, m, k, rdna_compute::DType::MQ3G256);
        let (ok3, max_err3, mean_err3) = compare("MQ3", &y_mq3_cpu, &y_mq3_gpu);
        any_fail |= !ok3;

        // ---- MQ2 ----
        let mq2_bytes = quantize_mq2g256(&f32_weights, k);
        let y_mq2_cpu = cpu_reference_mq(&mq2_bytes, &x, m, k, 72, 2, |scale, zero, q| scale * q as f32 + zero);
        let y_mq2_gpu = gpu_mq_gemv(&mut gpu, &mq2_bytes, &x, m, k, rdna_compute::DType::MQ2G256);
        let (ok2, max_err2, mean_err2) = compare("MQ2", &y_mq2_cpu, &y_mq2_gpu);
        any_fail |= !ok2;

        // Also verify the *rotation-only* step in isolation.
        let x_rot_cpu = cpu_rotate_x_mq(&x);
        let x_rot_gpu = gpu_rotate_x_mq(&mut gpu, &x, k);
        let (ok_rot, max_err_rot, _) = compare("rot", &x_rot_cpu, &x_rot_gpu);
        any_fail |= !ok_rot;
        eprintln!("  rotate_x max_err={:.6e}", max_err_rot);
    }

    if any_fail {
        eprintln!("\n[FAIL] One or more checks exceeded 1e-3 threshold.");
        std::process::exit(1);
    } else {
        eprintln!("\n[PASS] All MQ3/MQ2 kernels bit-exact within 1e-3.");
    }
}

fn fract_sin(x: f32) -> f32 {
    (x.sin() * 12345.6789f32).fract() * 2.0f32 - 1.0f32
}

fn compare(name: &str, cpu: &[f32], gpu: &[f32]) -> (bool, f32, f32) {
    let mut max_err = 0.0f32;
    let mut sum_err = 0.0f32;
    let mut bit_exact = 0usize;
    for i in 0..cpu.len() {
        let err = (cpu[i] - gpu[i]).abs();
        max_err = max_err.max(err);
        sum_err += err;
        if cpu[i] == gpu[i] {
            bit_exact += 1;
        }
    }
    let mean_err = sum_err / cpu.len().max(1) as f32;
    let ok = max_err <= 1e-3;
    let status = if ok { "PASS" } else { "FAIL" };
    eprintln!(
        "  {:<6} {}  max_err={:.6e}  mean_err={:.6e}  bit_exact={}/{}",
        name, status, max_err, mean_err, bit_exact, cpu.len()
    );
    (ok, max_err, mean_err)
}

// ---------------------------------------------------------------------------
// CPU reference: rotate x, dequantize weights, compute y = W_rot * x_rot
// ---------------------------------------------------------------------------

fn cpu_reference_mq(
    bytes: &[u8],
    x: &[f32],
    m: usize,
    k: usize,
    group_bytes: usize,
    bits: u8,
    recon: impl Fn(f32, f32, u8) -> f32,
) -> Vec<f32> {
    let groups_per_row = k / 256;
    let mut y = vec![0.0f32; m];

    // Rotate x
    let x_rot = cpu_rotate_x_mq(x);

    for row in 0..m {
        let row_off = row * groups_per_row * group_bytes;
        let mut acc = 0.0f32;

        for g in 0..groups_per_row {
            let g_off = row_off + g * group_bytes;
            let scale = f32::from_le_bytes([bytes[g_off], bytes[g_off + 1], bytes[g_off + 2], bytes[g_off + 3]]);
            let zero = f32::from_le_bytes([bytes[g_off + 4], bytes[g_off + 5], bytes[g_off + 6], bytes[g_off + 7]]);
            let data = &bytes[g_off + 8..g_off + group_bytes];

            let base_idx = g * 256;
            let mut q_vals: Vec<u8> = Vec::with_capacity(256);

            if bits == 3 {
                // 256 weights = 32 chunks * 8 weights * 3 bits = 96 bytes
                for chunk in 0..32 {
                    let ci = chunk * 8;
                    let b0 = data[chunk * 3];
                    let b1 = data[chunk * 3 + 1];
                    let b2 = data[chunk * 3 + 2];
                    q_vals.push(b0 & 7);
                    q_vals.push((b0 >> 3) & 7);
                    q_vals.push(((b0 >> 6) | (b1 << 2)) & 7);
                    q_vals.push((b1 >> 1) & 7);
                    q_vals.push((b1 >> 4) & 7);
                    q_vals.push(((b1 >> 7) | (b2 << 1)) & 7);
                    q_vals.push((b2 >> 2) & 7);
                    q_vals.push((b2 >> 5) & 7);
                }
            } else if bits == 2 {
                // 256 weights = 64 bytes, 4 weights per byte
                for i in 0..64 {
                    let b = data[i];
                    q_vals.push(b & 3);
                    q_vals.push((b >> 2) & 3);
                    q_vals.push((b >> 4) & 3);
                    q_vals.push((b >> 6) & 3);
                }
            } else {
                panic!("unsupported bits {}", bits);
            }

            for j in 0..256 {
                let w = recon(scale, zero, q_vals[j]);
                acc += w * x_rot[base_idx + j];
            }
        }
        y[row] = acc;
    }
    y
}

fn cpu_rotate_x_mq(x: &[f32]) -> Vec<f32> {
    let k = x.len();
    assert!(k % 256 == 0, "k must be multiple of 256");
    let signs1 = gen_fwht_signs(42, 256);
    let signs2 = gen_fwht_signs(1042, 256);
    let mut out = vec![0.0f32; k];
    for g in 0..(k / 256) {
        let mut group = [0.0f32; 256];
        group.copy_from_slice(&x[g * 256..(g + 1) * 256]);
        cpu_fwht_256(&mut group, &signs1, &signs2);
        out[g * 256..(g + 1) * 256].copy_from_slice(&group);
    }
    out
}

fn gen_fwht_signs(seed: u32, n: usize) -> Vec<f32> {
    let mut state = seed;
    (0..n)
        .map(|_| {
            state = state.wrapping_mul(1103515245).wrapping_add(12345) & 0x7fffffff;
            if (state >> 16) & 1 == 1 { 1.0f32 } else { -1.0f32 }
        })
        .collect()
}

fn cpu_fwht_256(x: &mut [f32], signs1: &[f32], signs2: &[f32]) {
    assert!(x.len() == 256);
    for i in 0..256 {
        x[i] *= signs1[i];
    }
    let mut stride = 1;
    while stride < 256 {
        let mut i = 0;
        while i < 256 {
            for j in 0..stride {
                let a = x[i + j];
                let b = x[i + j + stride];
                x[i + j] = a + b;
                x[i + j + stride] = a - b;
            }
            i += stride * 2;
        }
        stride <<= 1;
    }
    let scale = 0.0625; // 1/16
    for i in 0..256 {
        x[i] *= scale * signs2[i];
    }
}

// ---------------------------------------------------------------------------
// GPU wrappers
// ---------------------------------------------------------------------------

fn gpu_mq_gemv(
    gpu: &mut rdna_compute::Gpu,
    bytes: &[u8],
    x: &[f32],
    m: usize,
    k: usize,
    dtype: rdna_compute::DType,
) -> Vec<f32> {
    let d_a = gpu.upload_raw(bytes, &[bytes.len()]).unwrap();
    let d_x = gpu.upload_f32(x, &[k]).unwrap();
    let d_y = gpu.zeros(&[m], rdna_compute::DType::F32).unwrap();

    match dtype {
        rdna_compute::DType::MQ3G256 => {
            let d_tmp = gpu.zeros(&[k], rdna_compute::DType::F32).unwrap();
            gpu.gemv_mq3g256_with_rotate(&d_a, &d_x, &d_y, &d_tmp, m, k)
                .unwrap();
        }
        rdna_compute::DType::MQ2G256 => {
            let d_tmp = gpu.zeros(&[k], rdna_compute::DType::F32).unwrap();
            gpu.gemv_mq2g256_with_rotate(&d_a, &d_x, &d_y, &d_tmp, m, k)
                .unwrap();
        }
        _ => panic!("unexpected dtype"),
    }

    let mut y = vec![0.0f32; m];
    let y_bytes = unsafe { std::slice::from_raw_parts_mut(y.as_mut_ptr() as *mut u8, m * 4) };
    gpu.hip.memcpy_dtoh(y_bytes, unsafe { &d_y.buf }).unwrap();
    y
}

fn gpu_rotate_x_mq(gpu: &mut rdna_compute::Gpu, x: &[f32], k: usize) -> Vec<f32> {
    let d_x = gpu.upload_f32(x, &[k]).unwrap();
    let d_xr = gpu.zeros(&[k], rdna_compute::DType::F32).unwrap();
    gpu.rotate_x_mq(&d_x, &d_xr, k).unwrap();
    let mut out = vec![0.0f32; k];
    let out_bytes = unsafe { std::slice::from_raw_parts_mut(out.as_mut_ptr() as *mut u8, k * 4) };
    gpu.hip.memcpy_dtoh(out_bytes, unsafe { &d_xr.buf }).unwrap();
    out
}

// ---------------------------------------------------------------------------
// Quantizers (mirroring hipfire-quantize/src/main.rs)
// ---------------------------------------------------------------------------

fn quantize_mq3g256(f32_data: &[f32], k: usize) -> Vec<u8> {
    let group_size = 256;
    let block_bytes = 104;
    let n = f32_data.len();
    let n_blocks = (n + group_size - 1) / group_size;
    let mut output = vec![0u8; n_blocks * block_bytes];
    let signs1 = gen_fwht_signs(42, 256);
    let signs2 = gen_fwht_signs(1042, 256);

    for b in 0..n_blocks {
        let start = b * group_size;
        let end = (start + group_size).min(n);
        let mut group = [0.0f32; 256];
        let actual_len = end - start;
        group[..actual_len].copy_from_slice(&f32_data[start..end]);
        cpu_fwht_256(&mut group, &signs1, &signs2);

        let min_val = group.iter().cloned().fold(f32::INFINITY, f32::min);
        let max_val = group.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let range = max_val - min_val;
        let scale = if range > 0.0 { range / 7.0 } else { 1.0 };
        let inv_scale = if range > 0.0 { 1.0 / scale } else { 0.0 };

        let out_off = b * block_bytes;
        output[out_off..out_off + 4].copy_from_slice(&scale.to_le_bytes());
        output[out_off + 4..out_off + 8].copy_from_slice(&min_val.to_le_bytes());

        for chunk in 0..32 {
            let ci = chunk * 8;
            let mut q = [0u8; 8];
            for j in 0..8 {
                q[j] = ((group[ci + j] - min_val) * inv_scale + 0.5).clamp(0.0, 7.0) as u8;
            }
            let b0 = (q[0] & 7) | ((q[1] & 7) << 3) | ((q[2] & 3) << 6);
            let b1 = ((q[2] >> 2) & 1) | ((q[3] & 7) << 1) | ((q[4] & 7) << 4) | ((q[5] & 1) << 7);
            let b2 = ((q[5] >> 1) & 3) | ((q[6] & 7) << 2) | ((q[7] & 7) << 5);

            let bo = out_off + 8 + chunk * 3;
            output[bo] = b0;
            output[bo + 1] = b1;
            output[bo + 2] = b2;
        }
    }
    output
}

fn quantize_mq2g256(f32_data: &[f32], k: usize) -> Vec<u8> {
    let group_size = 256;
    let block_bytes = 72;
    let n = f32_data.len();
    let n_blocks = (n + group_size - 1) / group_size;
    let mut output = vec![0u8; n_blocks * block_bytes];
    let signs1 = gen_fwht_signs(42, 256);
    let signs2 = gen_fwht_signs(1042, 256);

    for b in 0..n_blocks {
        let start = b * group_size;
        let end = (start + group_size).min(n);
        let mut group = [0.0f32; 256];
        let actual_len = end - start;
        group[..actual_len].copy_from_slice(&f32_data[start..end]);
        cpu_fwht_256(&mut group, &signs1, &signs2);

        let min_val = group.iter().cloned().fold(f32::INFINITY, f32::min);
        let max_val = group.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let range = max_val - min_val;
        let scale = if range > 0.0 { range / 3.0 } else { 1.0 };
        let inv_scale = if range > 0.0 { 1.0 / scale } else { 0.0 };

        let out_off = b * block_bytes;
        output[out_off..out_off + 4].copy_from_slice(&scale.to_le_bytes());
        output[out_off + 4..out_off + 8].copy_from_slice(&min_val.to_le_bytes());

        for i in 0..64 {
            let mut byte_val = 0u8;
            for j in 0..4 {
                let q = ((group[4 * i + j] - min_val) * inv_scale + 0.5) as u8;
                byte_val |= q.min(3) << (j * 2);
            }
            output[out_off + 8 + i] = byte_val;
        }
    }
    output
}
