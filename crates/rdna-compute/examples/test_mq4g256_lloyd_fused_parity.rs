//! Parity test for MQ4-Lloyd residual + fused (gate+up, QKV, QKVZA) kernels.
//!
//! Each test sets up synthetic Lloyd-MQ4 weights, runs the GPU kernel, and
//! compares against a CPU reference that mirrors the kernel's per-row math.
//! Conformance gate: max-abs error < 5e-3 (fp32 reorder noise scales with K).
//!
//! Set `HIPFIRE_LLOYD_FORCE_BASELINE=1` to test the slow generic variants;
//! unset (default) tests the gfx1100 fast variants. Both should pass.

use rdna_compute::{Gpu, DType, GpuTensor};
use hip_bridge::HipResult;

// ─── f16 helpers (verbatim from test_gemv_mq4g256_lloyd_tail.rs) ────────────

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
        if m13 > 0x1000 || (m13 == 0x1000 && (new_mant & 1) != 0) {
            new_mant += 1;
        }
        let mut exp_bits = new_exp;
        if new_mant == 0x400 {
            new_mant = 0;
            exp_bits += 1;
        }
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
        if mant == 0 {
            sign << 31
        } else {
            let mut m = mant;
            let mut e = -1i32;
            while m & 0x400 == 0 {
                m <<= 1;
                e -= 1;
            }
            let mant32 = (m & 0x3ff) << 13;
            let exp32 = (e + 127 - 14) as u32;
            (sign << 31) | (exp32 << 23) | mant32
        }
    } else if exp == 0x1f {
        (sign << 31) | (0xff << 23) | (mant << 13)
    } else {
        let exp32 = (exp - 15 + 127) as u32;
        (sign << 31) | (exp32 << 23) | (mant << 13)
    };
    f32::from_bits(bits)
}

fn pack_4bit_group(qs: &[u8; 256]) -> [u8; 128] {
    let mut out = [0u8; 128];
    for i in 0..128 {
        out[i] = (qs[2 * i] & 0x0F) | ((qs[2 * i + 1] & 0x0F) << 4);
    }
    out
}

/// Build one row of MQ4-Lloyd-formatted bytes (160 B / group), returning both
/// the packed bytes AND the round-tripped (fp16-decoded) codebook so the CPU
/// reference can match the GPU's reconstruction exactly.
fn build_row(
    groups_per_row: usize,
    codebooks: &[[f32; 16]],
    indices: &[[u8; 256]],
) -> (Vec<u8>, Vec<[f32; 16]>) {
    let mut out = Vec::with_capacity(groups_per_row * 160);
    let mut roundtripped = Vec::with_capacity(groups_per_row);
    for g in 0..groups_per_row {
        let cb = &codebooks[g];
        let mut cb_rt = [0.0f32; 16];
        for (i, &v) in cb.iter().enumerate() {
            let bytes = f32_to_f16_le(v);
            out.extend_from_slice(&bytes);
            cb_rt[i] = f16_le_to_f32(bytes);
        }
        roundtripped.push(cb_rt);
        let packed = pack_4bit_group(&indices[g]);
        out.extend_from_slice(&packed);
    }
    (out, roundtripped)
}

/// Build a deterministic synthetic MQ4-Lloyd weight matrix [m × k]. Returns
/// the flat upload-ready bytes plus per-row codebook + indices for the CPU
/// reference. `seed` keeps separate matrices distinct (so e.g. A_gate and
/// A_up don't trivially share weights).
fn build_matrix(
    m: usize,
    groups_per_row: usize,
    seed: u32,
) -> (Vec<u8>, Vec<Vec<[f32; 16]>>, Vec<Vec<[u8; 256]>>) {
    let mut a_flat = Vec::with_capacity(m * groups_per_row * 160);
    let mut codebooks_per_row = Vec::with_capacity(m);
    let mut indices_per_row = Vec::with_capacity(m);
    for row in 0..m {
        let mut cbs = Vec::with_capacity(groups_per_row);
        let mut idxs = Vec::with_capacity(groups_per_row);
        for g in 0..groups_per_row {
            let base = (((row * 7 + g * 11 + seed as usize * 23) % 19) as f32) * 0.013 - 0.1;
            let cb: [f32; 16] = std::array::from_fn(|i| base + (i as f32 - 7.5) * 0.018);
            cbs.push(cb);
            let mut q = [0u8; 256];
            for i in 0..256usize {
                let h: usize = row.wrapping_mul(31)
                    ^ g.wrapping_mul(53)
                    ^ i.wrapping_mul(7)
                    ^ (seed as usize).wrapping_mul(101);
                q[i] = (h & 0xF) as u8;
            }
            idxs.push(q);
        }
        let (row_bytes, cbs_rt) = build_row(groups_per_row, &cbs, &idxs);
        a_flat.extend_from_slice(&row_bytes);
        codebooks_per_row.push(cbs_rt);
        indices_per_row.push(idxs);
    }
    (a_flat, codebooks_per_row, indices_per_row)
}

fn cpu_gemv_one_row(
    cbs: &[[f32; 16]],
    idxs: &[[u8; 256]],
    x: &[f32],
    groups_per_row: usize,
) -> f32 {
    let mut acc = 0.0f32;
    for g in 0..groups_per_row {
        let cb = &cbs[g];
        let qs = &idxs[g];
        for i in 0..256 {
            let q = qs[i] as usize & 0xF;
            acc += cb[q] * x[g * 256 + i];
        }
    }
    acc
}

fn cpu_gemv(
    m: usize,
    groups_per_row: usize,
    codebooks_per_row: &[Vec<[f32; 16]>],
    indices_per_row: &[Vec<[u8; 256]>],
    x: &[f32],
) -> Vec<f32> {
    (0..m)
        .map(|row| cpu_gemv_one_row(&codebooks_per_row[row], &indices_per_row[row], x, groups_per_row))
        .collect()
}

fn diff(label: &str, gpu: &[f32], cpu: &[f32]) -> bool {
    let mut max_abs = 0f32;
    let mut max_rel = 0f32;
    for i in 0..gpu.len() {
        let abs = (gpu[i] - cpu[i]).abs();
        let denom = cpu[i].abs().max(1e-6);
        max_abs = max_abs.max(abs);
        max_rel = max_rel.max(abs / denom);
    }
    let pass = max_abs < 5e-3;
    let verdict = if pass { "PASS" } else { "FAIL" };
    println!(
        "  {label:32}  max_abs={max_abs:.3e}  max_rel={max_rel:.3e}  {verdict}",
    );
    pass
}

// ─── Test bodies ────────────────────────────────────────────────────────────

fn test_residual(gpu: &mut Gpu) -> HipResult<bool> {
    let m = 64;
    let groups_per_row = 16;  // K=4096
    let k = groups_per_row * 256;
    let (a_flat, cbs_per_row, idxs_per_row) = build_matrix(m, groups_per_row, 0);
    let x: Vec<f32> = (0..k).map(|i| ((i as i32 % 13) as f32 - 6.0) * 0.05).collect();
    let y_initial: Vec<f32> = (0..m).map(|row| 0.5 - (row as f32) * 0.013).collect();

    let d_a = gpu.upload_raw(&a_flat, &[a_flat.len()])?;
    let d_x = gpu.upload_f32(&x, &[k])?;
    let d_y = gpu.upload_f32(&y_initial, &[m])?;

    gpu.gemv_mq4g256_lloyd_residual(&d_a, &d_x, &d_y, m, k)?;
    let y_gpu = gpu.download_f32(&d_y)?;

    let gemv_cpu = cpu_gemv(m, groups_per_row, &cbs_per_row, &idxs_per_row, &x);
    let cpu_residual: Vec<f32> = (0..m).map(|i| y_initial[i] + gemv_cpu[i]).collect();

    gpu.free_tensor(d_a)?; gpu.free_tensor(d_x)?; gpu.free_tensor(d_y)?;
    Ok(diff("residual (y += A·x)", &y_gpu, &cpu_residual))
}

fn test_fused_gate_up(gpu: &mut Gpu) -> HipResult<bool> {
    let gate_m = 48;
    let up_m = 48;
    let groups_per_row = 16;
    let k = groups_per_row * 256;
    let (a_gate, cbs_gate, idxs_gate) = build_matrix(gate_m, groups_per_row, 1);
    let (a_up,   cbs_up,   idxs_up)   = build_matrix(up_m,   groups_per_row, 2);
    let x: Vec<f32> = (0..k).map(|i| ((i as i32 % 13) as f32 - 6.0) * 0.05).collect();

    let d_ag = gpu.upload_raw(&a_gate, &[a_gate.len()])?;
    let d_au = gpu.upload_raw(&a_up, &[a_up.len()])?;
    let d_x  = gpu.upload_f32(&x, &[k])?;
    let d_yg = gpu.zeros(&[gate_m], DType::F32)?;
    let d_yu = gpu.zeros(&[up_m], DType::F32)?;

    gpu.fused_gate_up_mq4g256_lloyd(&d_ag, &d_au, &d_x, &d_yg, &d_yu, gate_m, up_m, k)?;
    let y_gate_gpu = gpu.download_f32(&d_yg)?;
    let y_up_gpu = gpu.download_f32(&d_yu)?;

    let y_gate_cpu = cpu_gemv(gate_m, groups_per_row, &cbs_gate, &idxs_gate, &x);
    let y_up_cpu   = cpu_gemv(up_m,   groups_per_row, &cbs_up,   &idxs_up,   &x);

    gpu.free_tensor(d_ag)?; gpu.free_tensor(d_au)?;
    gpu.free_tensor(d_x)?; gpu.free_tensor(d_yg)?; gpu.free_tensor(d_yu)?;
    let p1 = diff("fused_gate_up (y_gate)", &y_gate_gpu, &y_gate_cpu);
    let p2 = diff("fused_gate_up (y_up)",   &y_up_gpu,   &y_up_cpu);
    Ok(p1 && p2)
}

fn test_fused_qkv(gpu: &mut Gpu) -> HipResult<bool> {
    let q_m = 32; let k_m = 16; let v_m = 16;
    let groups_per_row = 16;
    let k = groups_per_row * 256;
    let (a_q, cbs_q, idxs_q) = build_matrix(q_m, groups_per_row, 3);
    let (a_k, cbs_k, idxs_k) = build_matrix(k_m, groups_per_row, 4);
    let (a_v, cbs_v, idxs_v) = build_matrix(v_m, groups_per_row, 5);
    let x: Vec<f32> = (0..k).map(|i| ((i as i32 % 13) as f32 - 6.0) * 0.05).collect();

    let d_aq = gpu.upload_raw(&a_q, &[a_q.len()])?;
    let d_ak = gpu.upload_raw(&a_k, &[a_k.len()])?;
    let d_av = gpu.upload_raw(&a_v, &[a_v.len()])?;
    let d_x  = gpu.upload_f32(&x, &[k])?;
    let d_yq = gpu.zeros(&[q_m], DType::F32)?;
    let d_yk = gpu.zeros(&[k_m], DType::F32)?;
    let d_yv = gpu.zeros(&[v_m], DType::F32)?;

    gpu.fused_qkv_mq4g256_lloyd(&d_aq, &d_ak, &d_av, &d_x, &d_yq, &d_yk, &d_yv, q_m, k_m, v_m, k)?;
    let yq_gpu = gpu.download_f32(&d_yq)?;
    let yk_gpu = gpu.download_f32(&d_yk)?;
    let yv_gpu = gpu.download_f32(&d_yv)?;

    let yq_cpu = cpu_gemv(q_m, groups_per_row, &cbs_q, &idxs_q, &x);
    let yk_cpu = cpu_gemv(k_m, groups_per_row, &cbs_k, &idxs_k, &x);
    let yv_cpu = cpu_gemv(v_m, groups_per_row, &cbs_v, &idxs_v, &x);

    gpu.free_tensor(d_aq)?; gpu.free_tensor(d_ak)?; gpu.free_tensor(d_av)?;
    gpu.free_tensor(d_x)?;
    gpu.free_tensor(d_yq)?; gpu.free_tensor(d_yk)?; gpu.free_tensor(d_yv)?;
    let p1 = diff("fused_qkv (y_q)", &yq_gpu, &yq_cpu);
    let p2 = diff("fused_qkv (y_k)", &yk_gpu, &yk_cpu);
    let p3 = diff("fused_qkv (y_v)", &yv_gpu, &yv_cpu);
    Ok(p1 && p2 && p3)
}

fn test_fused_qkvza(gpu: &mut Gpu) -> HipResult<bool> {
    let qkv_m = 32; let z_m = 16; let beta_m = 8; let alpha_m = 8;
    let groups_per_row = 16;
    let k = groups_per_row * 256;
    let (a_qkv, cbs_qkv, idxs_qkv) = build_matrix(qkv_m, groups_per_row, 6);
    let (a_z,   cbs_z,   idxs_z)   = build_matrix(z_m,   groups_per_row, 7);
    let (a_b,   cbs_b,   idxs_b)   = build_matrix(beta_m, groups_per_row, 8);
    let (a_a,   cbs_a,   idxs_a)   = build_matrix(alpha_m, groups_per_row, 9);
    let x: Vec<f32> = (0..k).map(|i| ((i as i32 % 13) as f32 - 6.0) * 0.05).collect();

    let d_aqkv = gpu.upload_raw(&a_qkv, &[a_qkv.len()])?;
    let d_az = gpu.upload_raw(&a_z, &[a_z.len()])?;
    let d_ab = gpu.upload_raw(&a_b, &[a_b.len()])?;
    let d_aa = gpu.upload_raw(&a_a, &[a_a.len()])?;
    let d_x  = gpu.upload_f32(&x, &[k])?;
    let d_yqkv = gpu.zeros(&[qkv_m], DType::F32)?;
    let d_yz = gpu.zeros(&[z_m], DType::F32)?;
    let d_yb = gpu.zeros(&[beta_m], DType::F32)?;
    let d_ya = gpu.zeros(&[alpha_m], DType::F32)?;

    gpu.fused_qkvza_mq4g256_lloyd(
        &d_aqkv, &d_az, &d_ab, &d_aa, &d_x,
        &d_yqkv, &d_yz, &d_yb, &d_ya,
        qkv_m, z_m, beta_m, alpha_m, k,
    )?;
    let yqkv_gpu = gpu.download_f32(&d_yqkv)?;
    let yz_gpu = gpu.download_f32(&d_yz)?;
    let yb_gpu = gpu.download_f32(&d_yb)?;
    let ya_gpu = gpu.download_f32(&d_ya)?;

    let yqkv_cpu = cpu_gemv(qkv_m, groups_per_row, &cbs_qkv, &idxs_qkv, &x);
    let yz_cpu = cpu_gemv(z_m, groups_per_row, &cbs_z, &idxs_z, &x);
    let yb_cpu = cpu_gemv(beta_m, groups_per_row, &cbs_b, &idxs_b, &x);
    let ya_cpu = cpu_gemv(alpha_m, groups_per_row, &cbs_a, &idxs_a, &x);

    gpu.free_tensor(d_aqkv)?; gpu.free_tensor(d_az)?;
    gpu.free_tensor(d_ab)?; gpu.free_tensor(d_aa)?;
    gpu.free_tensor(d_x)?;
    gpu.free_tensor(d_yqkv)?; gpu.free_tensor(d_yz)?;
    gpu.free_tensor(d_yb)?; gpu.free_tensor(d_ya)?;
    let p1 = diff("fused_qkvza (y_qkv)",   &yqkv_gpu, &yqkv_cpu);
    let p2 = diff("fused_qkvza (y_z)",     &yz_gpu,   &yz_cpu);
    let p3 = diff("fused_qkvza (y_beta)",  &yb_gpu,   &yb_cpu);
    let p4 = diff("fused_qkvza (y_alpha)", &ya_gpu,   &ya_cpu);
    Ok(p1 && p2 && p3 && p4)
}

fn main() {
    let mut gpu = Gpu::init().expect("GPU init failed");
    eprintln!("GPU: {}", gpu.arch);
    eprintln!(
        "HIPFIRE_LLOYD_FORCE_BASELINE: {:?}",
        std::env::var("HIPFIRE_LLOYD_FORCE_BASELINE").unwrap_or_default()
    );
    println!("--- residual ---");
    let p1 = test_residual(&mut gpu).expect("residual test");
    println!("--- fused_gate_up ---");
    let p2 = test_fused_gate_up(&mut gpu).expect("fused_gate_up test");
    println!("--- fused_qkv ---");
    let p3 = test_fused_qkv(&mut gpu).expect("fused_qkv test");
    println!("--- fused_qkvza ---");
    let p4 = test_fused_qkvza(&mut gpu).expect("fused_qkvza test");
    if !(p1 && p2 && p3 && p4) {
        eprintln!("\nFAIL: at least one MQ4-Lloyd kernel failed parity vs CPU reference");
        std::process::exit(1);
    }
    println!("\nALL PASS — residual + 3 fused MQ4-Lloyd kernels match CPU reference");
}
