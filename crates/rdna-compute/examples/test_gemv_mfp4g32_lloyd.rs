// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! GPU vs CPU correctness test for `gemv_mfp4g32_lloyd`.
//!
//! MFP4G32Lloyd = MFP4G32 with per-tensor 16-entry Lloyd-Max codebook.
//! Layout: [32-B fp16 codebook prefix][M rows identical to MFP4G32].
//! Reconstructs: row_scale_a * 2^(block_e-127) * codebook[nibble].
//!
//! Tests:
//!   1. Size invariant: packed.len() == 32 + m*(16+17*(k/32)).
//!   2. dequantize_mfp4g32_lloyd_to_f16 GPU output vs CPU dequant: max diff < 1e-2.
//!   3. gemv_mfp4g32_lloyd GPU y vs CPU dot(cpu_dequant, x_rot): max rel err < 1e-2.
//!
//! Sweeps K in {256, 512, 1024, 1280, 1536, 1792, 2048} (all K%256==0).

use rdna_compute::{Gpu, DType};

// ── fp16 helpers ──────────────────────────────────────────────────────────────

fn f32_to_f16_bits(v: f32) -> u16 {
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
        if m13 > 0x1000 || (m13 == 0x1000 && (new_mant & 1) != 0) { new_mant += 1; }
        let mut exp_bits = new_exp;
        if new_mant == 0x400 { new_mant = 0; exp_bits += 1; }
        (sign << 15) | (exp_bits << 10) | new_mant
    }
}

fn f16_bits_to_f32(h: u16) -> f32 {
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

// ── FWHT ──────────────────────────────────────────────────────────────────────

fn cpu_fwht_256(x: &mut [f32], signs1: &[f32], signs2: &[f32]) {
    assert_eq!(x.len(), 256);
    for i in 0..256 { x[i] *= signs1[i]; }
    let mut stride = 1usize;
    while stride < 256 {
        let mut i = 0;
        while i < 256 {
            for j in 0..stride {
                let a = x[i + j]; let b = x[i + j + stride];
                x[i + j] = a + b; x[i + j + stride] = a - b;
            }
            i += stride * 2;
        }
        stride <<= 1;
    }
    let scale = 0.0625f32;
    for i in 0..256 { x[i] *= scale * signs2[i]; }
}

fn gen_fwht_signs(seed: u32, n: usize) -> Vec<f32> {
    let mut state = seed;
    (0..n).map(|_| {
        state = state.wrapping_mul(1103515245).wrapping_add(12345) & 0x7fffffff;
        if (state >> 16) & 1 == 1 { 1.0f32 } else { -1.0f32 }
    }).collect()
}

// ── Lloyd codebook fitting ────────────────────────────────────────────────────

fn fit_mfp4_lloyd_codebook(vals: &[f32]) -> [f32; 16] {
    let e2m1: [f32; 16] = [
        0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
        -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
    ];
    let mut sorted: Vec<f32> = vals.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = sorted.len();
    let mut cb = [0.0f32; 16];
    if n == 0 {
        for k in 0..16 { cb[k] = e2m1[k]; }
        return cb;
    }
    for k in 0..16 {
        let frac = (2 * k + 1) as f32 / 32.0;
        let idx = ((frac * (n as f32 - 1.0)).round() as usize).min(n - 1);
        cb[k] = sorted[idx];
    }
    let range = sorted[n - 1] - sorted[0];
    if range > 0.0 {
        for it in 0..8 {
            let mut sums = [0.0f64; 16];
            let mut counts = [0u64; 16];
            for &w in vals {
                let mut best = 0usize;
                let mut best_d = (w - cb[0]).abs();
                for k in 1..16 { let d = (w - cb[k]).abs(); if d < best_d { best_d = d; best = k; } }
                sums[best] += w as f64; counts[best] += 1;
            }
            let mut moved = false;
            for k in 0..16 {
                if counts[k] > 0 {
                    let c = (sums[k] / counts[k] as f64) as f32;
                    if c != cb[k] { moved = true; } cb[k] = c;
                }
            }
            if it > 0 && !moved { break; }
        }
    }
    cb.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    cb
}

fn nearest_cb_idx(x: f32, cb: &[f32; 16]) -> u8 {
    let mut best = 0u8; let mut best_err = f32::INFINITY;
    for (i, &c) in cb.iter().enumerate() {
        let cf = f16_bits_to_f32(f32_to_f16_bits(c));
        let e = (cf - x).abs(); if e < best_err { best_err = e; best = i as u8; }
    }
    best
}

// ── MFP4G32Lloyd quantizer ────────────────────────────────────────────────────

fn quantize_mfp4g32_lloyd_2d(
    data: &[f32], m: usize, k: usize, signs1: &[f32], signs2: &[f32],
) -> Vec<u8> {
    assert!(k % 256 == 0);
    let mut rotated = vec![0.0f32; m * k];
    let mut norm_vals: Vec<f32> = Vec::with_capacity(m * k);
    let mut row_buf = vec![0.0f32; k];
    for r in 0..m {
        row_buf.copy_from_slice(&data[r * k..(r + 1) * k]);
        for seg in 0..(k / 256) { cpu_fwht_256(&mut row_buf[seg * 256..(seg + 1) * 256], signs1, signs2); }
        rotated[r * k..(r + 1) * k].copy_from_slice(&row_buf);
        let row_max = row_buf.iter().cloned().fold(0.0f32, |a, v| a.max(v.abs()));
        let inv_row = if row_max > 0.0 { 6.0 / row_max } else { 0.0 };
        for b in 0..(k / 32) {
            let block = &row_buf[b * 32..b * 32 + 32];
            let bmax = block.iter().cloned().fold(0.0f32, |a, v| a.max(v.abs()));
            let bmn = bmax * inv_row;
            let block_e: i32 = if bmn > 0.0 { (bmn / 6.0).log2().ceil() as i32 + 127 } else { 0 };
            let inv_block = (-(block_e.clamp(0, 254) - 127) as f32).exp2();
            for &v in block { norm_vals.push(v * inv_row * inv_block); }
        }
    }
    let cb = fit_mfp4_lloyd_codebook(&norm_vals);
    let row_bytes = 16 + 17 * (k / 32);
    let mut out = Vec::with_capacity(32 + m * row_bytes);
    for ki in 0..16 { out.extend_from_slice(&f32_to_f16_bits(cb[ki]).to_le_bytes()); }
    for r in 0..m {
        let row = &rotated[r * k..(r + 1) * k];
        let row_max = row.iter().cloned().fold(0.0f32, |a, v| a.max(v.abs()));
        let row_scale_a = if row_max > 0.0 { row_max / 6.0 } else { 1.0 };
        let inv_row = if row_max > 0.0 { 1.0 / row_scale_a } else { 0.0 };
        let mut rb = vec![0u8; row_bytes];
        rb[0..2].copy_from_slice(&f32_to_f16_bits(row_scale_a).to_le_bytes());
        rb[4..6].copy_from_slice(&((k / 32) as u16).to_le_bytes());
        rb[6] = 0x05;
        for b in 0..(k / 32) {
            let block = &row[b * 32..b * 32 + 32];
            let bmax = block.iter().cloned().fold(0.0f32, |a, v| a.max(v.abs()));
            let bmn = bmax * inv_row;
            let block_e: u8 = if bmn > 0.0 { ((bmn / 6.0).log2().ceil() as i32 + 127).clamp(0, 254) as u8 } else { 0 };
            let bsf = ((block_e as i32 - 127) as f32).exp2();
            let inv_block = if bsf > 0.0 { 1.0 / bsf } else { 0.0 };
            let po = 16 + b * 17;
            rb[po] = block_e;
            for i in 0..16 {
                let lo = block[2 * i] * inv_row * inv_block;
                let hi = block[2 * i + 1] * inv_row * inv_block;
                rb[po + 1 + i] = (nearest_cb_idx(lo, &cb) & 0x0F) | ((nearest_cb_idx(hi, &cb) & 0x0F) << 4);
            }
        }
        out.extend_from_slice(&rb);
    }
    out
}

// ── CPU dequant reference ─────────────────────────────────────────────────────

fn dequant_mfp4g32_lloyd(packed: &[u8], m: usize, k: usize) -> Vec<f32> {
    let row_bytes = 16 + 17 * (k / 32);
    let mut cb = [0.0f32; 16];
    for i in 0..16 { cb[i] = f16_bits_to_f32(u16::from_le_bytes([packed[2*i], packed[2*i+1]])); }
    let mut out = vec![0.0f32; m * k];
    for r in 0..m {
        let base = 32 + r * row_bytes;
        let row_scale_a = f16_bits_to_f32(u16::from_le_bytes([packed[base], packed[base+1]]));
        for b in 0..(k / 32) {
            let po = base + 16 + b * 17;
            let block_e = packed[po] as i32;
            let scale = row_scale_a * ((block_e - 127) as f32).exp2();
            for i in 0..16 {
                let byte = packed[po + 1 + i];
                out[r*k + b*32 + 2*i]     = scale * cb[(byte & 0x0F) as usize];
                out[r*k + b*32 + 2*i + 1] = scale * cb[((byte >> 4) & 0x0F) as usize];
            }
        }
    }
    out
}

// ── test one (m, k) configuration ────────────────────────────────────────────

fn run_one(gpu: &mut Gpu, m: usize, k: usize, signs1: &[f32], signs2: &[f32]) -> bool {
    // Build synthetic weights
    let mut s: u64 = 0x1234_5678_9abc_def0u64.wrapping_add(k as u64 * 7 + m as u64);
    let mut rnd = || -> f32 {
        s ^= s << 13; s ^= s >> 7; s ^= s << 17;
        ((s & 0xFFFFFF) as f32 / 0xFFFFFF as f32) * 2.0 - 1.0
    };
    let data: Vec<f32> = (0..m*k).map(|_| 0.5 * rnd()).collect();

    // Quantize
    let packed = quantize_mfp4g32_lloyd_2d(&data, m, k, signs1, signs2);
    let row_bytes = 16 + 17 * (k / 32);
    let expected_len = 32 + m * row_bytes;
    if packed.len() != expected_len {
        eprintln!("[FAIL] size: got {} expected {}", packed.len(), expected_len);
        return false;
    }

    // CPU dequant (rotated domain)
    let cpu_dq = dequant_mfp4g32_lloyd(&packed, m, k);

    // ── GPU dequant-to-f16 ────────────────────────────────────────────────────
    let d_packed = gpu.upload_raw(&packed, &[packed.len()]).unwrap();
    // Allocate f16 output as Raw bytes (m*k * 2 bytes)
    let f16_bytes = m * k * 2;
    let zero_f16 = vec![0u8; f16_bytes];
    let d_f16_out = gpu.upload_raw(&zero_f16, &[f16_bytes]).unwrap();
    gpu.dequantize_mfp4g32_lloyd_to_f16(&d_packed.buf, &d_f16_out.buf, m, k).unwrap();
    gpu.hip.device_synchronize().unwrap();

    // Download f16 bytes
    let mut f16_raw = vec![0u8; f16_bytes];
    gpu.hip.memcpy_dtoh(&mut f16_raw, &d_f16_out.buf).unwrap();
    let mut gpu_dq = vec![0.0f32; m * k];
    for i in 0..m*k {
        gpu_dq[i] = f16_bits_to_f32(u16::from_le_bytes([f16_raw[2*i], f16_raw[2*i+1]]));
    }

    // Check dequant consistency
    let mut max_dq = 0.0f32;
    for i in 0..m*k { let d = (gpu_dq[i] - cpu_dq[i]).abs(); if d > max_dq { max_dq = d; } }
    let dq_ok = max_dq < 1e-2;
    println!("[{}] dequant K={} max_gpu_vs_cpu={:.3e}", if dq_ok {"PASS"} else {"FAIL"}, k, max_dq);
    if !dq_ok {
        eprintln!("FATAL: dequant GPU vs CPU divergence {:.3e} >= 1e-2", max_dq);
        return false;
    }

    // ── GPU GEMV ──────────────────────────────────────────────────────────────
    let x: Vec<f32> = (0..k).map(|i| ((i as i32 % 13) as f32 - 6.0) * 0.05).collect();
    let mut x_rot = x.clone();
    for seg in 0..(k / 256) { cpu_fwht_256(&mut x_rot[seg * 256..(seg + 1) * 256], signs1, signs2); }

    // CPU reference y = dot(cpu_dq_row, x_rot)
    let mut y_ref = vec![0.0f32; m];
    for r in 0..m {
        let mut acc = 0.0f64;
        for c in 0..k { acc += cpu_dq[r*k+c] as f64 * x_rot[c] as f64; }
        y_ref[r] = acc as f32;
    }

    let d_x_rot = gpu.upload_f32(&x_rot, &[k]).unwrap();
    let d_y = gpu.zeros(&[m], DType::F32).unwrap();
    // gemv_mfp4g32_lloyd(a_raw, x, y, m, k) — x is already FWHT-rotated
    gpu.gemv_mfp4g32_lloyd(&d_packed, &d_x_rot, &d_y, m, k).unwrap();
    let y_gpu = gpu.download_f32(&d_y).unwrap();

    let mut max_rel = 0.0f32;
    for r in 0..m {
        let denom = y_ref[r].abs().max(1e-4);
        let rel = (y_gpu[r] - y_ref[r]).abs() / denom;
        if rel > max_rel { max_rel = rel; }
    }
    let gemv_ok = max_rel < 1e-2;
    println!("[{}] gemv  K={} max_rel={:.3e}", if gemv_ok {"PASS"} else {"FAIL"}, k, max_rel);
    if !gemv_ok {
        eprintln!("FATAL: GEMV rel-error {:.3e} >= 1e-2 at K={}", max_rel, k);
        return false;
    }
    true
}

fn main() {
    let mut gpu = Gpu::init().expect("Gpu::init failed");
    println!("arch: {}", gpu.arch);

    let signs1 = gen_fwht_signs(42, 256);
    let signs2 = gen_fwht_signs(1042, 256);

    let m = 64usize;
    let mut all_pass = true;
    for gpr in [1usize, 2, 4, 5, 6, 7, 8] {
        let k = gpr * 256;
        if !run_one(&mut gpu, m, k, &signs1, &signs2) {
            all_pass = false;
        }
    }

    if !all_pass {
        std::process::exit(1);
    }
    println!("ALL PASS");
}
