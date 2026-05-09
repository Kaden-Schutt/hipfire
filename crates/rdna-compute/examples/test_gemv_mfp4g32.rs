//! GPU vs CPU correctness test for `gemv_mfp4g32_with_rotate`.
//!
//! MFP4G32 = HFP4G32 + offline FWHT rotation (drop-in MQ4 replacement).
//! Builds a deterministic MFP4G32 weight matrix by applying the same 256-element
//! FWHT MQ4 ships with (signs1=seed 42, signs2=seed 1042) to each row segment,
//! then quantizing as HFP4G32 with `format_flags=0x05` stamped in the row header.
//!
//! Runs the GPU dispatch path (which rotates x via `mq_rotate_x` then calls
//! `gemv_hfp4g32` on the rotated activations) and compares against a CPU reference
//! that dequantizes the rotated weights and dots them with FWHT(x).
//!
//! Sweeps groups_per_row ∈ {2, 4, 5, 6, 7, 8} (K = groups_per_row × 256) to exercise
//! the same kernel paths as the HFP4 anchor test, including all 3 tail-by-g%4 paths.

use rdna_compute::{Gpu, DType};

const E2M1_LUT: [f32; 16] = [
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
    -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
];

fn e2m1_round(x: f32) -> u8 {
    let mut best_idx = 0u8;
    let mut best_err = f32::INFINITY;
    for (i, &code) in E2M1_LUT.iter().enumerate() {
        let err = (code - x).abs();
        if err < best_err {
            best_err = err;
            best_idx = i as u8;
        }
    }
    best_idx
}

fn f32_to_f16_le_bits(v: f32) -> u16 {
    let bits = v.to_bits();
    let sign = ((bits >> 31) & 0x1) as u16;
    let exp = ((bits >> 23) & 0xff) as i32;
    let mant = bits & 0x7fffff;
    if exp == 0xff {
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
    }
}

fn f16_le_bits_to_f32(h: u16) -> f32 {
    let sign = ((h >> 15) & 1) as u32;
    let exp = ((h >> 10) & 0x1f) as i32;
    let mant = (h & 0x3ff) as u32;
    let bits = if exp == 0 {
        if mant == 0 { sign << 31 }
        else {
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

/// FWHT on a 256-element segment. Mirrors `cpu_fwht_256` in hipfire-quantize and
/// the GPU `mq_rotate_x` kernel: signs1 → butterfly → 1/sqrt(256) scale → signs2.
fn cpu_fwht_256(x: &mut [f32], signs1: &[f32], signs2: &[f32]) {
    assert_eq!(x.len(), 256);
    for i in 0..256 { x[i] *= signs1[i]; }
    let mut stride = 1usize;
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
    let scale = 0.0625; // 1/sqrt(256) = 1/16
    for i in 0..256 { x[i] *= scale * signs2[i]; }
}

/// Same LCG sign generator MQ4 ships with (`gen_fwht_signs(seed, 256)`).
fn gen_fwht_signs(seed: u32, n: usize) -> Vec<f32> {
    let mut state = seed;
    (0..n).map(|_| {
        state = state.wrapping_mul(1103515245).wrapping_add(12345) & 0x7fffffff;
        if (state >> 16) & 1 == 1 { 1.0f32 } else { -1.0f32 }
    }).collect()
}

/// Quantize one row of K f32 weights to HFP4G32 byte format with `format_flags=0x05`
/// (offline FWHT rotation present). The caller is responsible for applying the FWHT
/// to the row before calling this — mirrors `quantize_mfp4g32_2d` in hipfire-quantize.
fn quantize_row_with_rotation_flag(row: &[f32]) -> Vec<u8> {
    let k = row.len();
    assert!(k % 32 == 0);
    let n_blocks = k / 32;
    let row_bytes = 16 + n_blocks * 17;
    let mut out = vec![0u8; row_bytes];

    let row_max_abs = row.iter().cloned().fold(0.0f32, |m, v| m.max(v.abs()));
    let row_scale_a = if row_max_abs > 0.0 { row_max_abs / 6.0 } else { 1.0 };
    let inv_row = if row_max_abs > 0.0 { 1.0 / row_scale_a } else { 0.0 };

    out[0..2].copy_from_slice(&f32_to_f16_le_bits(row_scale_a).to_le_bytes());
    out[2..4].copy_from_slice(&0u16.to_le_bytes());
    out[4..6].copy_from_slice(&(n_blocks as u16).to_le_bytes());
    out[6] = 0x05; // bit 0 + bits 2-3 = 01: rotation present, offline FWHT
    out[7] = 0u8;

    for b in 0..n_blocks {
        let block = &row[b * 32..(b + 1) * 32];
        let block_max_abs = block.iter().cloned().fold(0.0f32, |m, v| m.max(v.abs()));
        let block_max_normalized = block_max_abs * inv_row;
        let block_e: u8 = if block_max_normalized > 0.0 {
            let log_ratio = (block_max_normalized / 6.0).log2();
            let e_signed = log_ratio.ceil() as i32 + 127;
            e_signed.clamp(0, 254) as u8
        } else { 0u8 };

        let block_scale_factor = ((block_e as i32 - 127) as f32).exp2();
        let inv_block = if block_scale_factor > 0.0 { 1.0 / block_scale_factor } else { 0.0 };

        let off = 16 + b * 17;
        out[off] = block_e;
        for i in 0..16 {
            let lo = block[2 * i] * inv_row * inv_block;
            let hi = block[2 * i + 1] * inv_row * inv_block;
            let lo_n = e2m1_round(lo);
            let hi_n = e2m1_round(hi);
            out[off + 1 + i] = (lo_n & 0x0F) | ((hi_n & 0x0F) << 4);
        }
    }
    out
}

fn dequant_row(packed: &[u8], k: usize) -> Vec<f32> {
    let n_blocks = k / 32;
    let row_scale_a = f16_le_bits_to_f32(u16::from_le_bytes([packed[0], packed[1]]));

    let mut out = vec![0.0f32; k];
    for b in 0..n_blocks {
        let off = 16 + b * 17;
        let block_e = packed[off] as i32;
        let block_scale_factor = ((block_e - 127) as f32).exp2();
        let scale = row_scale_a * block_scale_factor;
        for i in 0..16 {
            let byte = packed[off + 1 + i];
            let lo = (byte & 0x0F) as usize;
            let hi = ((byte >> 4) & 0x0F) as usize;
            out[b * 32 + 2 * i]     = scale * E2M1_LUT[lo];
            out[b * 32 + 2 * i + 1] = scale * E2M1_LUT[hi];
        }
    }
    out
}

/// Build the M-row × K-col packed weight matrix. Each row is rotated per-256-element
/// segment with the same FWHT signs the runtime uses, then HFP4G32-quantized with
/// `format_flags=0x05` stamped. Returns (packed bytes, dequantized rotated weights).
/// The dequantized weights are what the GPU kernel "sees" after E2M1 LUT + UE8M0
/// scale + FP16 row scale; multiplying by FWHT(x) gives the reference y.
fn build_test_matrix(
    m: usize,
    k: usize,
    seed: u64,
    signs1: &[f32],
    signs2: &[f32],
) -> (Vec<u8>, Vec<f32>) {
    let mut packed: Vec<u8> = Vec::new();
    let mut seen: Vec<f32> = Vec::with_capacity(m * k);
    let mut state = seed;
    for _r in 0..m {
        let mut row = Vec::with_capacity(k);
        for _ in 0..(k / 2) {
            state ^= state << 13; state ^= state >> 7; state ^= state << 17;
            let u1 = ((state & 0xFFFFFF) as f32 / 0x1000000 as f32).max(1e-7);
            state ^= state << 13; state ^= state >> 7; state ^= state << 17;
            let u2 = ((state & 0xFFFFFF) as f32 / 0x1000000 as f32).max(1e-7);
            let r_mag = (-2.0 * u1.ln()).sqrt();
            let theta = 2.0 * std::f32::consts::PI * u2;
            row.push(r_mag * theta.cos() * 0.4);
            row.push(r_mag * theta.sin() * 0.4);
        }
        // Apply per-segment FWHT — exactly the offline-rotation step MFP4G32 quant does.
        for seg in 0..(k / 256) {
            cpu_fwht_256(&mut row[seg * 256..(seg + 1) * 256], signs1, signs2);
        }
        let row_packed = quantize_row_with_rotation_flag(&row);
        let row_dq = dequant_row(&row_packed, k);
        packed.extend_from_slice(&row_packed);
        seen.extend_from_slice(&row_dq);
    }
    (packed, seen)
}

fn cpu_reference(seen: &[f32], x_rot: &[f32], m: usize, k: usize) -> Vec<f32> {
    let mut y = vec![0.0f32; m];
    for r in 0..m {
        let mut acc = 0.0f32;
        for c in 0..k {
            acc += seen[r * k + c] * x_rot[c];
        }
        y[r] = acc;
    }
    y
}

fn run_one(gpu: &mut Gpu, groups_per_row: usize, signs1: &[f32], signs2: &[f32]) -> (usize, f32, f32) {
    let m = 64;
    let k = groups_per_row * 256;

    let (packed, seen_w_rot) = build_test_matrix(
        m, k,
        0xc0ffee_dead_c0ffeeu64.wrapping_add(k as u64),
        signs1, signs2,
    );

    // Original (UN-rotated) x — this is what callers pass to gemv_mfp4g32_with_rotate.
    // Same shape as the HFP4 anchor's x for direct comparability.
    let x: Vec<f32> = (0..k).map(|i| ((i as i32 % 13) as f32 - 6.0) * 0.05).collect();

    // CPU-side rotation of x (per-256-element FWHT) — gives the activation that the
    // GPU kernel sees after `mq_rotate_x` runs internally.
    let mut x_rot = x.clone();
    for seg in 0..(k / 256) {
        cpu_fwht_256(&mut x_rot[seg * 256..(seg + 1) * 256], signs1, signs2);
    }

    let d_a = gpu.upload_raw(&packed, &[packed.len()]).unwrap();
    let d_x = gpu.upload_f32(&x, &[k]).unwrap();
    let d_y = gpu.zeros(&[m], DType::F32).unwrap();

    // Allocate the FWHT scratch (mq_x_rot) by ensuring signs are loaded; the
    // dispatch wrapper rotates x into the GPU's internal scratch for us.
    gpu.ensure_mq_signs().unwrap();
    let x_rot_alias = rdna_compute::GpuTensor {
        buf: unsafe { gpu.mq_x_rot.as_ref().unwrap().buf.alias() },
        shape: vec![gpu.mq_x_rot.as_ref().unwrap().buf.size() / 4],
        dtype: DType::F32,
    };
    gpu.gemv_mfp4g32_with_rotate(&d_a, &d_x, &d_y, &x_rot_alias, m, k).unwrap();
    let y_gpu = gpu.download_f32(&d_y).unwrap();

    let y_ref = cpu_reference(&seen_w_rot, &x_rot, m, k);

    let mut max_abs = 0.0f32;
    let mut max_rel = 0.0f32;
    for r in 0..m {
        let abs = (y_gpu[r] - y_ref[r]).abs();
        if abs > max_abs { max_abs = abs; }
        let denom = y_ref[r].abs().max(1.0);
        let rel = abs / denom;
        if rel > max_rel { max_rel = rel; }
    }
    (k, max_abs, max_rel)
}

fn main() {
    let mut gpu = Gpu::init().expect("Gpu::init failed");
    println!("arch: {}", gpu.arch);

    let signs1 = gen_fwht_signs(42, 256);
    let signs2 = gen_fwht_signs(1042, 256);

    let mut any_fail = false;
    for groups_per_row in [2usize, 4, 5, 6, 7, 8] {
        let (k, max_abs, max_rel) = run_one(&mut gpu, groups_per_row, &signs1, &signs2);
        // Same tolerance as the HFP4 anchor test: FP32 sum-of-K reorder noise +
        // FP16-row-scale rounding interaction. FWHT scaling by 1/16 keeps |w_rot|
        // in the same magnitude band as un-rotated random data.
        let pass = max_abs < 5e-3 && max_rel < 5e-3;
        let tag = if pass { "PASS" } else { "FAIL" };
        println!("[{}] groups_per_row={} K={} max_abs={:.6e} max_rel={:.6e}",
                 tag, groups_per_row, k, max_abs, max_rel);
        if !pass { any_fail = true; }
    }

    if any_fail {
        std::process::exit(1);
    }
    println!("ALL PASS");
}
