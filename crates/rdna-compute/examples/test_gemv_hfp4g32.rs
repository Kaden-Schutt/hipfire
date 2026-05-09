//! GPU vs CPU correctness test for `gemv_hfp4g32`.
//!
//! Builds a deterministic HFP4G32 weight matrix in-process, runs the GPU kernel,
//! and compares against a CPU reference that bit-mirrors the kernel's dequant
//! arithmetic (E2M1 LUT, UE8M0 ldexp scale, FP16 row scale). Fails if max-abs
//! error exceeds a small absolute bound (FP32 summation reorder noise).
//!
//! Format: 16-B per-row header + (K/32) × 17-B per-block payload.
//!         See docs/quant-formats/hfp4.md.
//!
//! Sweeps groups_per_row ∈ {2, 4, 5, 6, 7, 8} (K = groups_per_row × 256) to
//! exercise quad-clean and all 3 tail-by-g%4 paths in the kernel.

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

/// Quantize one row of K f32 weights to HFP4G32 byte format.
/// Returns 16-B header + (K/32) × 17-B blocks.
/// Mirrors hipfire-quantize::quantize_hfp4g32_row.
fn quantize_row(row: &[f32]) -> Vec<u8> {
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
    out[6] = 0u8; out[7] = 0u8;

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

/// CPU reference dequant of one row to f32. Reads the FP16 row scale through the
/// same f16→f32 path the GPU uses (LUT compiles to FP16 immediates; row scale is
/// loaded as u16 then cast to __half then to float). Round-tripping the row scale
/// through fp16 ensures the CPU reference uses the SAME row_scale_a value the GPU
/// will read, eliminating a bogus 1e-3 / scale fp16 conversion error.
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

/// Build the M-row × K-col packed weight matrix (concatenated per-row blobs).
/// Returns (packed bytes, the FP32 weights *as the GPU will see them after dequant*).
fn build_test_matrix(m: usize, k: usize, seed: u64) -> (Vec<u8>, Vec<f32>) {
    let mut packed: Vec<u8> = Vec::new();
    let mut seen: Vec<f32> = Vec::with_capacity(m * k);
    let mut state = seed;
    for r in 0..m {
        // Gaussian-ish row data via Box-Muller from xorshift.
        let mut row = Vec::with_capacity(k);
        for _ in 0..(k / 2) {
            state ^= state << 13; state ^= state >> 7; state ^= state << 17;
            let u1 = ((state & 0xFFFFFF) as f32 / 0x1000000 as f32).max(1e-7);
            state ^= state << 13; state ^= state >> 7; state ^= state << 17;
            let u2 = ((state & 0xFFFFFF) as f32 / 0x1000000 as f32).max(1e-7);
            let r_mag = (-2.0 * u1.ln()).sqrt();
            let theta = 2.0 * std::f32::consts::PI * u2;
            // N(0, 0.4) — well-utilized E2M1 lattice without saturation.
            row.push(r_mag * theta.cos() * 0.4);
            row.push(r_mag * theta.sin() * 0.4);
        }
        let row_packed = quantize_row(&row);
        let row_dq = dequant_row(&row_packed, k);
        // Note: r isn't used directly; we just want a unique row index.
        let _ = r;
        packed.extend_from_slice(&row_packed);
        seen.extend_from_slice(&row_dq);
    }
    (packed, seen)
}

fn cpu_reference(seen: &[f32], x: &[f32], m: usize, k: usize) -> Vec<f32> {
    let mut y = vec![0.0f32; m];
    for r in 0..m {
        let mut acc = 0.0f32;
        for c in 0..k {
            acc += seen[r * k + c] * x[c];
        }
        y[r] = acc;
    }
    y
}

fn run_one(gpu: &mut Gpu, groups_per_row: usize) -> (usize, f32, f32) {
    let m = 64;
    let k = groups_per_row * 256;

    let (packed, seen_w) = build_test_matrix(m, k, 0xdead_beef_dead_beefu64.wrapping_add(k as u64));

    // x in [-0.5, 0.5)
    let x: Vec<f32> = (0..k).map(|i| ((i as i32 % 13) as f32 - 6.0) * 0.05).collect();

    let d_a = gpu.upload_raw(&packed, &[packed.len()]).unwrap();
    let d_x = gpu.upload_f32(&x, &[k]).unwrap();
    let d_y = gpu.zeros(&[m], DType::F32).unwrap();

    gpu.gemv_hfp4g32(&d_a, &d_x, &d_y, m, k).unwrap();
    let y_gpu = gpu.download_f32(&d_y).unwrap();

    let y_ref = cpu_reference(&seen_w, &x, m, k);

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

    let mut any_fail = false;
    for groups_per_row in [2usize, 4, 5, 6, 7, 8] {
        let (k, max_abs, max_rel) = run_one(&mut gpu, groups_per_row);
        // FP32 sum-of-K terms reorder noise: empirically ~1e-3 for K=2048 with
        // |w| ≤ ~3 and |x| ≤ 0.6 (worst-case sum magnitude ~K). Allow 5e-3 to
        // absorb FP16-row-scale rounding interaction.
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
