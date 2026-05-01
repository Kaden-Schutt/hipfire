//! Channel test for the new HFQ3-G256 GEMV with fused residual add.
//!
//! Compares `gemv_hfq3g256_residual(A, x_rot, y_init)` against
//! `gemv_hfq3g256(A, x_rot, y_tmp); y_init + y_tmp` byte-exactly. Both
//! kernels share the same K4-unrolled 4-accumulator combine ordering on
//! gfx1100, so the residual variant should match the non-residual + sequential
//! add to within FP rounding (1e-4 typical).
//!
//! Run:
//!   cargo run --release --example test_gemv_hfq3g256_residual

fn main() {
    let mut gpu = rdna_compute::Gpu::init().unwrap();

    // Cover: minimal (1 group), 4-group quad, 16-group quad+tail, FFN-shape (43 groups).
    let shapes = [
        (4usize, 256usize),
        (16, 1024),
        (32, 4096),
        (64, 11008), // 43 groups -> tests tail handling
    ];

    let mut any_fail = false;

    for &(m, k) in &shapes {
        eprintln!("\n========== shape m={} k={} ==========", m, k);

        // Synthetic weights + x in pre-rotated coordinate (residual kernel
        // operates on whatever x it's given — quant-time FWHT pre-rotation
        // is implicit at the engine level).
        let f32_weights: Vec<f32> = (0..m * k)
            .map(|i| fract_sin(i as f32 * 0.731f32 + 1.337f32))
            .collect();
        let x_rot: Vec<f32> = (0..k)
            .map(|i| fract_sin(i as f32 * 0.513f32 + 2.719f32))
            .collect();
        let y_init: Vec<f32> = (0..m)
            .map(|i| fract_sin(i as f32 * 0.281f32 + 4.111f32) * 0.5)
            .collect();

        let mq3_bytes = quantize_mq3g256(&f32_weights);

        // ---- Reference: gemv_hfq3g256 + sequential add ----
        let d_a = gpu.upload_raw(&mq3_bytes, &[mq3_bytes.len()]).unwrap();
        let d_x = gpu.upload_f32(&x_rot, &[k]).unwrap();
        let d_y_ref = gpu.upload_f32(&y_init, &[m]).unwrap();
        let d_y_tmp = gpu.zeros(&[m], rdna_compute::DType::F32).unwrap();

        gpu.gemv_hfq3g256(&d_a, &d_x, &d_y_tmp, m, k).unwrap();
        gpu.add_inplace_f32(&d_y_ref, &d_y_tmp).unwrap();

        let mut y_ref = vec![0.0f32; m];
        let y_ref_bytes = unsafe {
            std::slice::from_raw_parts_mut(y_ref.as_mut_ptr() as *mut u8, m * 4)
        };
        gpu.hip
            .memcpy_dtoh(y_ref_bytes, unsafe { &d_y_ref.buf })
            .unwrap();

        // ---- New: gemv_hfq3g256_residual ----
        let d_y_test = gpu.upload_f32(&y_init, &[m]).unwrap();
        gpu.gemv_hfq3g256_residual(&d_a, &d_x, &d_y_test, m, k)
            .unwrap();

        let mut y_test = vec![0.0f32; m];
        let y_test_bytes = unsafe {
            std::slice::from_raw_parts_mut(y_test.as_mut_ptr() as *mut u8, m * 4)
        };
        gpu.hip
            .memcpy_dtoh(y_test_bytes, unsafe { &d_y_test.buf })
            .unwrap();

        let (ok, max_err, mean_err) = compare("residual", &y_ref, &y_test);
        eprintln!(
            "  residual   max_err={:.6e}  mean_err={:.6e}",
            max_err, mean_err
        );
        any_fail |= !ok;
    }

    if any_fail {
        eprintln!("\n[FAIL] residual kernel diverged from non-residual + add reference.");
        std::process::exit(1);
    } else {
        eprintln!("\n[PASS] gemv_hfq3g256_residual matches reference on all shapes.");
    }
}

fn fract_sin(x: f32) -> f32 {
    (x.sin() * 12345.6789f32).fract() * 2.0f32 - 1.0f32
}

fn compare(_name: &str, a: &[f32], b: &[f32]) -> (bool, f32, f32) {
    let mut max_err = 0.0f32;
    let mut sum_err = 0.0f32;
    for i in 0..a.len() {
        let err = (a[i] - b[i]).abs();
        max_err = max_err.max(err);
        sum_err += err;
    }
    let mean_err = sum_err / a.len().max(1) as f32;
    let ok = max_err <= 1e-3;
    (ok, max_err, mean_err)
}

/// Replicates the MQ3 quantizer's per-group affine scheme:
///   one fp32 scale + one fp32 zero + 96 B of packed 8x3-bit weights = 104 B/group.
/// Cross-byte unpack pattern matches kernels/src/gemv_hfq3g256.hip.
fn quantize_mq3g256(f32_data: &[f32]) -> Vec<u8> {
    assert!(f32_data.len() % 256 == 0, "must be multiple of 256");
    let n_groups = f32_data.len() / 256;
    let mut bytes = Vec::with_capacity(n_groups * 104);

    for g in 0..n_groups {
        let group = &f32_data[g * 256..(g + 1) * 256];
        let min_v = group.iter().cloned().fold(f32::INFINITY, f32::min);
        let max_v = group.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let scale = if (max_v - min_v).abs() < 1e-9 {
            1e-6
        } else {
            (max_v - min_v) / 7.0
        };
        let zero = min_v;

        bytes.extend_from_slice(&scale.to_le_bytes());
        bytes.extend_from_slice(&zero.to_le_bytes());

        // Quantize each weight to 0..7
        let mut q = vec![0u8; 256];
        for i in 0..256 {
            let qi = ((group[i] - zero) / scale).round().clamp(0.0, 7.0) as u8;
            q[i] = qi;
        }

        // Pack 8 × 3-bit into 3 bytes per chunk; 32 chunks per group.
        for chunk in 0..32 {
            let ci = chunk * 8;
            let q0 = q[ci] as u32;
            let q1 = q[ci + 1] as u32;
            let q2 = q[ci + 2] as u32;
            let q3 = q[ci + 3] as u32;
            let q4 = q[ci + 4] as u32;
            let q5 = q[ci + 5] as u32;
            let q6 = q[ci + 6] as u32;
            let q7 = q[ci + 7] as u32;
            let pk = q0
                | (q1 << 3)
                | (q2 << 6)
                | (q3 << 9)
                | (q4 << 12)
                | (q5 << 15)
                | (q6 << 18)
                | (q7 << 21);
            bytes.push((pk & 0xFF) as u8);
            bytes.push(((pk >> 8) & 0xFF) as u8);
            bytes.push(((pk >> 16) & 0xFF) as u8);
        }
    }

    bytes
}
