// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Correctness test for all HFQ6-G256 GEMM kernels (residual, gate_up, qkv, qkvza).
//! Compares WMMA + scalar variants against a CPU reference dequant + FMA.

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

fn cpu_dequant(q: &[u8], m: usize, k: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; m * k];
    let groups_per_row = k / 256;
    for r in 0..m {
        for g in 0..groups_per_row {
            let off = (r * groups_per_row + g) * 200;
            let scale = f32::from_le_bytes([q[off], q[off+1], q[off+2], q[off+3]]);
            let zero = f32::from_le_bytes([q[off+4], q[off+5], q[off+6], q[off+7]]);
            for i in (0..256).step_by(4) {
                let bo = off + 8 + (i / 4) * 3;
                let b0 = q[bo] as u32;
                let b1 = q[bo+1] as u32;
                let b2 = q[bo+2] as u32;
                let q0 = (b0 & 0x3F) as f32;
                let q1 = (((b0 >> 6) | (b1 << 2)) & 0x3F) as f32;
                let q2 = (((b1 >> 4) | (b2 << 4)) & 0x3F) as f32;
                let q3 = ((b2 >> 2) & 0x3F) as f32;
                let base = r * k + g * 256 + i;
                out[base    ] = scale * q0 + zero;
                out[base + 1] = scale * q1 + zero;
                out[base + 2] = scale * q2 + zero;
                out[base + 3] = scale * q3 + zero;
            }
        }
    }
    out
}

fn cpu_gemm(w: &[f32], x: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    let mut y = vec![0.0f32; n * m];
    for b in 0..n {
        for r in 0..m {
            let mut acc = 0.0f32;
            for c in 0..k {
                acc += w[r * k + c] * x[b * k + c];
            }
            y[b * m + r] = acc;
        }
    }
    y
}

fn max_err(a: &[f32], b: &[f32]) -> (f32, usize, usize) {
    let mut maxe = 0.0f32;
    let mut pos = 0;
    let mut bad = 0;
    for (i, (&x, &y)) in a.iter().zip(b.iter()).enumerate() {
        let e = (x - y).abs();
        let rel_e = if y.abs() > 0.001 { e / y.abs() } else { e };
        if rel_e > 0.1 { bad += 1; }
        if e > maxe { maxe = e; pos = i; }
    }
    (maxe, pos, bad)
}

fn main() {
    let mut gpu = rdna_compute::Gpu::init().unwrap();
    eprintln!("GPU: {}", gpu.arch);

    // Test at sizes realistic for 4B model: k=2560 (hidden dim), n=128 (prefill batch)
    let k = 2560usize;  // 10 groups of 256
    let n = 128usize;   // realistic prefill batch

    // Reproducible pseudo-random data
    let rand = |seed: u32, i: usize| -> f32 {
        let mut s = seed.wrapping_add(i as u32).wrapping_mul(1103515245).wrapping_add(12345);
        s = s.wrapping_mul(1103515245).wrapping_add(12345);
        (s % 1000) as f32 / 500.0 - 1.0
    };

    // Build X [n × k]
    let x_data: Vec<f32> = (0..n*k).map(|i| rand(42, i) * 0.5).collect();
    let d_x = gpu.upload_f32(&x_data, &[n * k]).unwrap();

    // === Test 1: gemm_hfq6g256_residual (scalar) ===
    {
        let m = 64;
        let w_data: Vec<f32> = (0..m*k).map(|i| rand(100, i) * 0.3).collect();
        let q = quantize_hfq6g256(&w_data);
        let w_dq = cpu_dequant(&q, m, k);
        let y_ref = cpu_gemm(&w_dq, &x_data, m, k, n);

        let d_a = gpu.upload_raw(&q, &[q.len()]).unwrap();
        let y_init = vec![0.0f32; n * m];
        let d_y = gpu.upload_f32(&y_init, &[n * m]).unwrap();
        // Force scalar (non-WMMA) path
        std::env::set_var("HIPFIRE_FP16", "0");
        gpu.gemm_hfq6g256_residual(&d_a, &d_x, &d_y, m, k, n).unwrap();
        std::env::remove_var("HIPFIRE_FP16");
        let y_gpu = gpu.download_f32(&d_y).unwrap();
        let (e, pos, bad) = max_err(&y_ref, &y_gpu);
        eprintln!("residual scalar: max_err={e:.4} at [{}], bad={bad}/{}", pos, n*m);
    }

    // === Test 2: gemm_hfq6g256_residual_wmma ===
    {
        let m = 64;
        let w_data: Vec<f32> = (0..m*k).map(|i| rand(200, i) * 0.3).collect();
        let q = quantize_hfq6g256(&w_data);
        let w_dq = cpu_dequant(&q, m, k);
        let y_ref = cpu_gemm(&w_dq, &x_data, m, k, n);
        let d_a = gpu.upload_raw(&q, &[q.len()]).unwrap();
        let y_init = vec![0.0f32; n * m];
        let d_y = gpu.upload_f32(&y_init, &[n * m]).unwrap();
        gpu.gemm_hfq6g256_residual_wmma(&d_a, &d_x, &d_y, m, k, n).unwrap();
        let y_gpu = gpu.download_f32(&d_y).unwrap();
        let (e, pos, bad) = max_err(&y_ref, &y_gpu);
        eprintln!("residual WMMA  : max_err={e:.4} at [{}], bad={bad}/{}", pos, n*m);
    }

    // === Test 3: gemm_gate_up_hfq6g256 (scalar) ===
    {
        let gate_m = 32;
        let up_m = 32;
        let wg: Vec<f32> = (0..gate_m*k).map(|i| rand(300, i) * 0.3).collect();
        let wu: Vec<f32> = (0..up_m*k).map(|i| rand(400, i) * 0.3).collect();
        let qg = quantize_hfq6g256(&wg);
        let qu = quantize_hfq6g256(&wu);
        let wg_dq = cpu_dequant(&qg, gate_m, k);
        let wu_dq = cpu_dequant(&qu, up_m, k);
        let yg_ref = cpu_gemm(&wg_dq, &x_data, gate_m, k, n);
        let yu_ref = cpu_gemm(&wu_dq, &x_data, up_m, k, n);

        let d_ag = gpu.upload_raw(&qg, &[qg.len()]).unwrap();
        let d_au = gpu.upload_raw(&qu, &[qu.len()]).unwrap();
        let d_yg = gpu.zeros(&[n * gate_m], rdna_compute::DType::F32).unwrap();
        let d_yu = gpu.zeros(&[n * up_m], rdna_compute::DType::F32).unwrap();

        std::env::set_var("HIPFIRE_FP16", "0");
        gpu.gemm_gate_up_hfq6g256(&d_ag, &d_au, &d_x, &d_yg, &d_yu, gate_m, up_m, k, n).unwrap();
        std::env::remove_var("HIPFIRE_FP16");
        let yg = gpu.download_f32(&d_yg).unwrap();
        let yu = gpu.download_f32(&d_yu).unwrap();
        let (eg, _, bg) = max_err(&yg_ref, &yg);
        let (eu, _, bu) = max_err(&yu_ref, &yu);
        eprintln!("gate_up scalar : gate_err={eg:.4} bad={bg}, up_err={eu:.4} bad={bu}");
    }

    // === Test 4: gemm_gate_up_hfq6g256_wmma ===
    {
        let gate_m = 32;
        let up_m = 32;
        let wg: Vec<f32> = (0..gate_m*k).map(|i| rand(500, i) * 0.3).collect();
        let wu: Vec<f32> = (0..up_m*k).map(|i| rand(600, i) * 0.3).collect();
        let qg = quantize_hfq6g256(&wg);
        let qu = quantize_hfq6g256(&wu);
        let wg_dq = cpu_dequant(&qg, gate_m, k);
        let wu_dq = cpu_dequant(&qu, up_m, k);
        let yg_ref = cpu_gemm(&wg_dq, &x_data, gate_m, k, n);
        let yu_ref = cpu_gemm(&wu_dq, &x_data, up_m, k, n);

        let d_ag = gpu.upload_raw(&qg, &[qg.len()]).unwrap();
        let d_au = gpu.upload_raw(&qu, &[qu.len()]).unwrap();
        let d_yg = gpu.zeros(&[n * gate_m], rdna_compute::DType::F32).unwrap();
        let d_yu = gpu.zeros(&[n * up_m], rdna_compute::DType::F32).unwrap();

        gpu.gemm_gate_up_hfq6g256_wmma(&d_ag, &d_au, &d_x, &d_yg, &d_yu, gate_m, up_m, k, n).unwrap();
        let yg = gpu.download_f32(&d_yg).unwrap();
        let yu = gpu.download_f32(&d_yu).unwrap();
        let (eg, _, bg) = max_err(&yg_ref, &yg);
        let (eu, _, bu) = max_err(&yu_ref, &yu);
        eprintln!("gate_up WMMA   : gate_err={eg:.4} bad={bg}, up_err={eu:.4} bad={bu}");
    }

    // === Test 5: gemm_qkv_hfq6g256_wmma ===
    {
        let q_m = 32; let k_m = 16; let v_m = 16;
        let wq: Vec<f32> = (0..q_m*k).map(|i| rand(700, i) * 0.3).collect();
        let wk: Vec<f32> = (0..k_m*k).map(|i| rand(800, i) * 0.3).collect();
        let wv: Vec<f32> = (0..v_m*k).map(|i| rand(900, i) * 0.3).collect();
        let qq = quantize_hfq6g256(&wq);
        let qk = quantize_hfq6g256(&wk);
        let qv = quantize_hfq6g256(&wv);
        let wq_dq = cpu_dequant(&qq, q_m, k);
        let wk_dq = cpu_dequant(&qk, k_m, k);
        let wv_dq = cpu_dequant(&qv, v_m, k);
        let yq_ref = cpu_gemm(&wq_dq, &x_data, q_m, k, n);
        let yk_ref = cpu_gemm(&wk_dq, &x_data, k_m, k, n);
        let yv_ref = cpu_gemm(&wv_dq, &x_data, v_m, k, n);

        let d_aq = gpu.upload_raw(&qq, &[qq.len()]).unwrap();
        let d_ak = gpu.upload_raw(&qk, &[qk.len()]).unwrap();
        let d_av = gpu.upload_raw(&qv, &[qv.len()]).unwrap();
        let d_yq = gpu.zeros(&[n * q_m], rdna_compute::DType::F32).unwrap();
        let d_yk = gpu.zeros(&[n * k_m], rdna_compute::DType::F32).unwrap();
        let d_yv = gpu.zeros(&[n * v_m], rdna_compute::DType::F32).unwrap();

        gpu.gemm_qkv_hfq6g256_wmma(&d_aq, &d_ak, &d_av, &d_x, &d_yq, &d_yk, &d_yv, q_m, k_m, v_m, k, n).unwrap();
        let yq = gpu.download_f32(&d_yq).unwrap();
        let yk = gpu.download_f32(&d_yk).unwrap();
        let yv = gpu.download_f32(&d_yv).unwrap();
        let (eq, _, bq) = max_err(&yq_ref, &yq);
        let (ek, _, bk) = max_err(&yk_ref, &yk);
        let (ev, _, bv) = max_err(&yv_ref, &yv);
        eprintln!("qkv WMMA       : q_err={eq:.4} bad={bq}, k_err={ek:.4} bad={bk}, v_err={ev:.4} bad={bv}");
    }

    // === Test 6: gemm_qkvza_hfq6g256_wmma ===
    {
        let qkv_m = 32; let z_m = 16; let beta_m = 16; let alpha_m = 16;
        let wqkv: Vec<f32> = (0..qkv_m*k).map(|i| rand(1100, i) * 0.3).collect();
        let wz: Vec<f32> = (0..z_m*k).map(|i| rand(1200, i) * 0.3).collect();
        let wbeta: Vec<f32> = (0..beta_m*k).map(|i| rand(1300, i) * 0.3).collect();
        let walpha: Vec<f32> = (0..alpha_m*k).map(|i| rand(1400, i) * 0.3).collect();
        let qqkv = quantize_hfq6g256(&wqkv);
        let qz = quantize_hfq6g256(&wz);
        let qbeta = quantize_hfq6g256(&wbeta);
        let qalpha = quantize_hfq6g256(&walpha);
        let wqkv_dq = cpu_dequant(&qqkv, qkv_m, k);
        let wz_dq = cpu_dequant(&qz, z_m, k);
        let wbeta_dq = cpu_dequant(&qbeta, beta_m, k);
        let walpha_dq = cpu_dequant(&qalpha, alpha_m, k);
        let y1_ref = cpu_gemm(&wqkv_dq, &x_data, qkv_m, k, n);
        let y2_ref = cpu_gemm(&wz_dq, &x_data, z_m, k, n);
        let y3_ref = cpu_gemm(&wbeta_dq, &x_data, beta_m, k, n);
        let y4_ref = cpu_gemm(&walpha_dq, &x_data, alpha_m, k, n);

        let d_a1 = gpu.upload_raw(&qqkv, &[qqkv.len()]).unwrap();
        let d_a2 = gpu.upload_raw(&qz, &[qz.len()]).unwrap();
        let d_a3 = gpu.upload_raw(&qbeta, &[qbeta.len()]).unwrap();
        let d_a4 = gpu.upload_raw(&qalpha, &[qalpha.len()]).unwrap();
        let d_y1 = gpu.zeros(&[n * qkv_m], rdna_compute::DType::F32).unwrap();
        let d_y2 = gpu.zeros(&[n * z_m], rdna_compute::DType::F32).unwrap();
        let d_y3 = gpu.zeros(&[n * beta_m], rdna_compute::DType::F32).unwrap();
        let d_y4 = gpu.zeros(&[n * alpha_m], rdna_compute::DType::F32).unwrap();

        gpu.gemm_qkvza_hfq6g256_wmma(&d_a1, &d_a2, &d_a3, &d_a4, &d_x,
            &d_y1, &d_y2, &d_y3, &d_y4,
            qkv_m, z_m, beta_m, alpha_m, k, n).unwrap();
        let y1 = gpu.download_f32(&d_y1).unwrap();
        let y2 = gpu.download_f32(&d_y2).unwrap();
        let y3 = gpu.download_f32(&d_y3).unwrap();
        let y4 = gpu.download_f32(&d_y4).unwrap();
        let (e1, _, b1) = max_err(&y1_ref, &y1);
        let (e2, _, b2) = max_err(&y2_ref, &y2);
        let (e3, _, b3) = max_err(&y3_ref, &y3);
        let (e4, _, b4) = max_err(&y4_ref, &y4);
        eprintln!("qkvza WMMA     : qkv={e1:.4}({b1}) z={e2:.4}({b2}) beta={e3:.4}({b3}) alpha={e4:.4}({b4})");
    }
}
