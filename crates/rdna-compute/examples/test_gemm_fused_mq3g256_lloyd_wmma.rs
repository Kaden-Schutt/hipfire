//! Phase B1 parity test for the 3 fused MQ3-Lloyd WMMA GEMMs (qkvza / qkv /
//! gate_up). Each test is small — covers a few canonical shapes, exercises
//! tile-row straddling projection boundaries, and uses **distinct codebooks
//! per projection** so a swapped weight pointer produces a detectable
//! per-projection error pattern rather than silently passing.
//!
//! CPU reference accumulates in fp64; X is f16-roundtripped to match the
//! GPU's view. Tolerance set to 3× max-abs observed at K=12288 from the
//! Phase A devlog (1.75e-4).

use rdna_compute::Gpu;

// ---------- f16 helpers (same as Phase A test) ----------

fn f32_to_f16_le(v: f32) -> [u8; 2] {
    let bits = v.to_bits();
    let sign = ((bits >> 31) & 0x1) as u16;
    let exp = ((bits >> 23) & 0xff) as i32;
    let mant = bits & 0x7fffff;
    let h: u16 = if exp == 0xff {
        (sign << 15) | (0x1f << 10) | if mant != 0 { 0x200 } else { 0 }
    } else if exp - 127 + 15 < 1 {
        sign << 15
    } else if exp - 127 + 15 > 30 {
        (sign << 15) | (0x1f << 10)
    } else {
        let new_exp = (exp - 127 + 15) as u16;
        let m13 = mant & 0x1fff;
        let mut new_mant = (mant >> 13) as u16;
        if m13 > 0x1000 || (m13 == 0x1000 && (new_mant & 1) != 0) { new_mant += 1; }
        let mut exp_bits = new_exp;
        if new_mant == 0x400 { new_mant = 0; exp_bits += 1; }
        (sign << 15) | (exp_bits << 10) | new_mant
    };
    h.to_le_bytes()
}

fn f16_le_to_f32(b: [u8; 2]) -> f32 {
    let h = u16::from_le_bytes(b);
    let sign = ((h >> 15) & 1) as u32;
    let exp = ((h >> 10) & 0x1f) as i32;
    let mant = (h & 0x3ff) as u32;
    let bits = if exp == 0 {
        if mant == 0 { sign << 31 } else {
            let mut m = mant; let mut e = -1i32;
            while m & 0x400 == 0 { m <<= 1; e -= 1; }
            (sign << 31) | (((e + 127 - 14) as u32) << 23) | ((m & 0x3ff) << 13)
        }
    } else if exp == 0x1f {
        (sign << 31) | (0xff << 23) | (mant << 13)
    } else {
        (sign << 31) | (((exp - 15 + 127) as u32) << 23) | (mant << 13)
    };
    f32::from_bits(bits)
}

fn pack_3bit_group(qs: &[u8; 256]) -> [u8; 96] {
    let mut out = [0u8; 96];
    for tid in 0..32 {
        let mut pk: u32 = 0;
        for i in 0..8 { pk |= (qs[tid * 8 + i] as u32 & 7) << (3 * i); }
        out[tid * 3]     = (pk        & 0xff) as u8;
        out[tid * 3 + 1] = ((pk >>  8) & 0xff) as u8;
        out[tid * 3 + 2] = ((pk >> 16) & 0xff) as u8;
    }
    out
}

/// Build one MQ3-Lloyd weight matrix [m × k] with **per-projection-distinct**
/// codebooks so a swapped pointer in dispatch produces a non-zero parity
/// error.  proj_id mixes into the codebook seed.
fn build_lloyd_matrix(
    m: usize, k: usize, proj_id: usize,
) -> (Vec<u8>, Vec<Vec<[f32; 8]>>, Vec<Vec<[u8; 256]>>) {
    let groups_per_row = k / 256;
    let mut all_bytes = Vec::with_capacity(m * groups_per_row * 112);
    let mut codebooks_per_row = Vec::with_capacity(m);
    let mut indices_per_row = Vec::with_capacity(m);
    for row in 0..m {
        let mut cbs = Vec::with_capacity(groups_per_row);
        let mut idxs = Vec::with_capacity(groups_per_row);
        for g in 0..groups_per_row {
            // Distinct centroids per (proj, row, g) — proj_id offset visible
            // in centroid magnitudes, so a swapped A_pointer produces a
            // detectable Y delta proportional to proj_id*0.05.
            let proj_off = proj_id as f32 * 0.05;
            let base = ((row * 7 + g * 11 + proj_id * 31) % 19) as f32 * 0.013 - 0.1 + proj_off;
            let cb: [f32; 8] = std::array::from_fn(|i| base + (i as f32 - 3.5) * 0.025);
            let mut cb_rt = [0.0f32; 8];
            for (i, &v) in cb.iter().enumerate() {
                let bytes = f32_to_f16_le(v);
                all_bytes.extend_from_slice(&bytes);
                cb_rt[i] = f16_le_to_f32(bytes);
            }
            cbs.push(cb_rt);

            let mut q = [0u8; 256];
            for i in 0..256 {
                q[i] = ((row.wrapping_mul(31) ^ g.wrapping_mul(53) ^ i.wrapping_mul(7)
                       ^ proj_id.wrapping_mul(101)) & 7) as u8;
            }
            let packed = pack_3bit_group(&q);
            all_bytes.extend_from_slice(&packed);
            idxs.push(q);
        }
        codebooks_per_row.push(cbs);
        indices_per_row.push(idxs);
    }
    assert_eq!(all_bytes.len(), m * groups_per_row * 112);
    (all_bytes, codebooks_per_row, indices_per_row)
}

/// CPU reference: Y[col][row] = sum_k A[row][k] * X[col][k] (no residual —
/// fused kernels use overwrite semantics).
fn cpu_reference(
    m: usize, k: usize, n: usize,
    cbs: &[Vec<[f32; 8]>],
    idxs: &[Vec<[u8; 256]>],
    x_fp32_rt: &[f32],   // already f16-roundtripped
) -> Vec<f32> {
    let groups_per_row = k / 256;
    let mut y = vec![0.0f32; n * m];
    for col in 0..n {
        for row in 0..m {
            let mut acc = 0.0f64;
            for g in 0..groups_per_row {
                let cb = &cbs[row][g];
                let qs = &idxs[row][g];
                for i in 0..256 {
                    let q = qs[i] as usize & 7;
                    acc += cb[q] as f64 * x_fp32_rt[col * k + g * 256 + i] as f64;
                }
            }
            y[col * m + row] = acc as f32;
        }
    }
    y
}

fn make_x(n: usize, k: usize) -> (Vec<f32>, Vec<f32>) {
    let x: Vec<f32> = (0..(n * k))
        .map(|i| ((i as i32 % 13) as f32 - 6.0) * 0.05)
        .collect();
    let x_rt: Vec<f32> = x.iter().map(|&v| f16_le_to_f32(f32_to_f16_le(v))).collect();
    (x, x_rt)
}

fn diff_metrics(actual: &[f32], expected: &[f32]) -> (f32, f32) {
    let mut max_abs = 0f32;
    let mut max_rel = 0f32;
    for i in 0..actual.len() {
        let d = (actual[i] - expected[i]).abs();
        let denom = expected[i].abs().max(1e-3);
        max_abs = max_abs.max(d);
        max_rel = max_rel.max(d / denom);
    }
    (max_abs, max_rel)
}

// ---------- per-kernel runners ----------

const PHASE_A_TOL: f32 = 1.75e-4;  // 3× observed max-abs at K=12288.

fn test_qkvza(gpu: &mut Gpu, qkv_m: usize, z_m: usize, beta_m: usize, alpha_m: usize, k: usize, n: usize) -> bool {
    use rdna_compute::DType;
    println!("--- qkvza M=({}+{}+{}+{}) K={} N={} ---", qkv_m, z_m, beta_m, alpha_m, k, n);

    let (a_qkv_b, cb_qkv, idx_qkv) = build_lloyd_matrix(qkv_m, k, 0);
    let (a_z_b,   cb_z,   idx_z)   = build_lloyd_matrix(z_m,   k, 1);
    let (a_beta_b, cb_beta, idx_beta) = build_lloyd_matrix(beta_m, k, 2);
    let (a_alpha_b, cb_alpha, idx_alpha) = build_lloyd_matrix(alpha_m, k, 3);
    let (x, x_rt) = make_x(n, k);

    let d_a_qkv = gpu.upload_raw(&a_qkv_b, &[a_qkv_b.len()]).unwrap();
    let d_a_z   = gpu.upload_raw(&a_z_b,   &[a_z_b.len()]).unwrap();
    let d_a_beta = gpu.upload_raw(&a_beta_b, &[a_beta_b.len()]).unwrap();
    let d_a_alpha = gpu.upload_raw(&a_alpha_b, &[a_alpha_b.len()]).unwrap();
    let d_x = gpu.upload_f32(&x, &[n, k]).unwrap();
    let d_y_qkv = gpu.zeros(&[n, qkv_m], DType::F32).unwrap();
    let d_y_z   = gpu.zeros(&[n, z_m], DType::F32).unwrap();
    let d_y_beta = gpu.zeros(&[n, beta_m], DType::F32).unwrap();
    let d_y_alpha = gpu.zeros(&[n, alpha_m], DType::F32).unwrap();

    gpu.gemm_qkvza_mq3g256_lloyd_wmma(
        &d_a_qkv, &d_a_z, &d_a_beta, &d_a_alpha,
        &d_x,
        &d_y_qkv, &d_y_z, &d_y_beta, &d_y_alpha,
        qkv_m, z_m, beta_m, alpha_m, k, n,
    ).unwrap();
    let y_qkv_gpu = gpu.download_f32(&d_y_qkv).unwrap();
    let y_z_gpu = gpu.download_f32(&d_y_z).unwrap();
    let y_beta_gpu = gpu.download_f32(&d_y_beta).unwrap();
    let y_alpha_gpu = gpu.download_f32(&d_y_alpha).unwrap();

    let y_qkv_ref = cpu_reference(qkv_m, k, n, &cb_qkv, &idx_qkv, &x_rt);
    let y_z_ref = cpu_reference(z_m, k, n, &cb_z, &idx_z, &x_rt);
    let y_beta_ref = cpu_reference(beta_m, k, n, &cb_beta, &idx_beta, &x_rt);
    let y_alpha_ref = cpu_reference(alpha_m, k, n, &cb_alpha, &idx_alpha, &x_rt);

    let (a_qkv_ma, _) = diff_metrics(&y_qkv_gpu, &y_qkv_ref);
    let (a_z_ma, _) = diff_metrics(&y_z_gpu, &y_z_ref);
    let (a_beta_ma, _) = diff_metrics(&y_beta_gpu, &y_beta_ref);
    let (a_alpha_ma, _) = diff_metrics(&y_alpha_gpu, &y_alpha_ref);

    let max_abs = a_qkv_ma.max(a_z_ma).max(a_beta_ma).max(a_alpha_ma);
    let pass = max_abs < PHASE_A_TOL;
    println!("  qkv max_abs={:.3e}  z={:.3e}  beta={:.3e}  alpha={:.3e}  {}",
             a_qkv_ma, a_z_ma, a_beta_ma, a_alpha_ma,
             if pass { "PASS" } else { "FAIL" });

    for d in [d_a_qkv, d_a_z, d_a_beta, d_a_alpha, d_x, d_y_qkv, d_y_z, d_y_beta, d_y_alpha] {
        gpu.free_tensor(d).unwrap();
    }
    pass
}

fn test_qkv(gpu: &mut Gpu, q_m: usize, k_m: usize, v_m: usize, k: usize, n: usize) -> bool {
    use rdna_compute::DType;
    println!("--- qkv M=({}+{}+{}) K={} N={} ---", q_m, k_m, v_m, k, n);

    let (a_q_b, cb_q, idx_q) = build_lloyd_matrix(q_m, k, 0);
    let (a_k_b, cb_k, idx_k) = build_lloyd_matrix(k_m, k, 1);
    let (a_v_b, cb_v, idx_v) = build_lloyd_matrix(v_m, k, 2);
    let (x, x_rt) = make_x(n, k);

    let d_a_q = gpu.upload_raw(&a_q_b, &[a_q_b.len()]).unwrap();
    let d_a_k = gpu.upload_raw(&a_k_b, &[a_k_b.len()]).unwrap();
    let d_a_v = gpu.upload_raw(&a_v_b, &[a_v_b.len()]).unwrap();
    let d_x = gpu.upload_f32(&x, &[n, k]).unwrap();
    let d_y_q = gpu.zeros(&[n, q_m], DType::F32).unwrap();
    let d_y_k = gpu.zeros(&[n, k_m], DType::F32).unwrap();
    let d_y_v = gpu.zeros(&[n, v_m], DType::F32).unwrap();

    gpu.gemm_qkv_mq3g256_lloyd_wmma(
        &d_a_q, &d_a_k, &d_a_v,
        &d_x,
        &d_y_q, &d_y_k, &d_y_v,
        q_m, k_m, v_m, k, n,
    ).unwrap();
    let y_q_gpu = gpu.download_f32(&d_y_q).unwrap();
    let y_k_gpu = gpu.download_f32(&d_y_k).unwrap();
    let y_v_gpu = gpu.download_f32(&d_y_v).unwrap();

    let y_q_ref = cpu_reference(q_m, k, n, &cb_q, &idx_q, &x_rt);
    let y_k_ref = cpu_reference(k_m, k, n, &cb_k, &idx_k, &x_rt);
    let y_v_ref = cpu_reference(v_m, k, n, &cb_v, &idx_v, &x_rt);

    let (a_q_ma, _) = diff_metrics(&y_q_gpu, &y_q_ref);
    let (a_k_ma, _) = diff_metrics(&y_k_gpu, &y_k_ref);
    let (a_v_ma, _) = diff_metrics(&y_v_gpu, &y_v_ref);

    let max_abs = a_q_ma.max(a_k_ma).max(a_v_ma);
    let pass = max_abs < PHASE_A_TOL;
    println!("  q max_abs={:.3e}  k={:.3e}  v={:.3e}  {}",
             a_q_ma, a_k_ma, a_v_ma, if pass { "PASS" } else { "FAIL" });

    for d in [d_a_q, d_a_k, d_a_v, d_x, d_y_q, d_y_k, d_y_v] {
        gpu.free_tensor(d).unwrap();
    }
    pass
}

fn test_gate_up(gpu: &mut Gpu, gate_m: usize, up_m: usize, k: usize, n: usize) -> bool {
    use rdna_compute::DType;
    println!("--- gate_up M=({}+{}) K={} N={} ---", gate_m, up_m, k, n);

    let (a_gate_b, cb_gate, idx_gate) = build_lloyd_matrix(gate_m, k, 0);
    let (a_up_b,   cb_up,   idx_up)   = build_lloyd_matrix(up_m,   k, 1);
    let (x, x_rt) = make_x(n, k);

    let d_a_gate = gpu.upload_raw(&a_gate_b, &[a_gate_b.len()]).unwrap();
    let d_a_up   = gpu.upload_raw(&a_up_b,   &[a_up_b.len()]).unwrap();
    let d_x = gpu.upload_f32(&x, &[n, k]).unwrap();
    let d_y_gate = gpu.zeros(&[n, gate_m], DType::F32).unwrap();
    let d_y_up   = gpu.zeros(&[n, up_m], DType::F32).unwrap();

    gpu.gemm_gate_up_mq3g256_lloyd_wmma(
        &d_a_gate, &d_a_up,
        &d_x,
        &d_y_gate, &d_y_up,
        gate_m, up_m, k, n,
    ).unwrap();
    let y_gate_gpu = gpu.download_f32(&d_y_gate).unwrap();
    let y_up_gpu = gpu.download_f32(&d_y_up).unwrap();

    let y_gate_ref = cpu_reference(gate_m, k, n, &cb_gate, &idx_gate, &x_rt);
    let y_up_ref = cpu_reference(up_m, k, n, &cb_up, &idx_up, &x_rt);

    let (a_gate_ma, _) = diff_metrics(&y_gate_gpu, &y_gate_ref);
    let (a_up_ma, _) = diff_metrics(&y_up_gpu, &y_up_ref);

    let max_abs = a_gate_ma.max(a_up_ma);
    let pass = max_abs < PHASE_A_TOL;
    println!("  gate max_abs={:.3e}  up={:.3e}  {}",
             a_gate_ma, a_up_ma, if pass { "PASS" } else { "FAIL" });

    for d in [d_a_gate, d_a_up, d_x, d_y_gate, d_y_up] {
        gpu.free_tensor(d).unwrap();
    }
    pass
}

fn main() {
    let mut gpu = Gpu::init().expect("GPU init failed");
    eprintln!("GPU: {}", gpu.arch);

    let mut all_pass = true;

    // qkvza — distinct projection sizes, one shape that straddles boundaries.
    all_pass &= test_qkvza(&mut gpu, 64, 16, 8, 8, 1024, 16);     // total_m=96, 6 tiles
    all_pass &= test_qkvza(&mut gpu, 256, 32, 16, 16, 4096, 64);  // larger
    all_pass &= test_qkvza(&mut gpu, 512, 64, 32, 32, 4096, 32);

    // qkv — Q is typically 8x larger than K=V (GQA); test both balanced and
    // asymmetric.
    all_pass &= test_qkv(&mut gpu, 64, 64, 64, 1024, 16);
    all_pass &= test_qkv(&mut gpu, 256, 32, 32, 4096, 64);
    all_pass &= test_qkv(&mut gpu, 512, 64, 64, 4096, 32);

    // gate_up — gate and up are typically equal-sized.
    all_pass &= test_gate_up(&mut gpu, 256, 256, 1024, 16);
    all_pass &= test_gate_up(&mut gpu, 1024, 1024, 4096, 64);

    println!();
    if !all_pass {
        eprintln!("FAIL: one or more fused-kernel parity tests failed");
        std::process::exit(1);
    }
    println!("ALL PASS");
}
