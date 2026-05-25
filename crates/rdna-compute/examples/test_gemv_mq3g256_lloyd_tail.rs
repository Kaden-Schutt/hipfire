// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kevin Read
// hipfire — see LICENSE and NOTICE in the project root.

//! Tail K-sweep parity test for gemv_mq3g256_lloyd.
//!
//! Sweeps groups_per_row ∈ {4, 5, 6, 7, 8} (K ∈ {1024, 1280, 1536, 1792, 2048})
//! to exercise quad-clean (4, 8) and all three tail cases (5, 6, 7) of the
//! K4-unrolled gfx1100 kernel. CPU reference is the bit-exact per-row formula
//! from the Lloyd-MQ3 block layout: 8 fp16 centroids per group, 256 weights per
//! group, packed 3-bit indices at byte_off = 16 + tid*3.
//!
//! Compares GPU output vs CPU reference. Fails if max-abs error > 1e-3 (fp32
//! summation reorder noise; tighter than the typical decode logits-Δ bar).

use rdna_compute::{Gpu, DType};

/// f32 → IEEE 754 binary16 little-endian, round-to-nearest-even on the trailing
/// 13 mantissa bits we drop. Adequate for synthetic test data; doesn't need to
/// match an exotic rocBLAS path.
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
        // Round-to-nearest-even on the dropped 13 LSBs.
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

/// Inverse for the CPU reference — read the f16 we wrote and produce the same
/// f32 the GPU's `__half2float` will. Critical: must match the GPU's read-side
/// quantization, otherwise the CPU reference diverges from "what the GPU sees".
fn f16_le_to_f32(b: [u8; 2]) -> f32 {
    let h = u16::from_le_bytes(b);
    let sign = ((h >> 15) & 1) as u32;
    let exp = ((h >> 10) & 0x1f) as i32;
    let mant = (h & 0x3ff) as u32;
    let bits = if exp == 0 {
        if mant == 0 {
            sign << 31
        } else {
            // Subnormal — normalize.
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

fn pack_3bit_group(qs: &[u8; 256]) -> [u8; 96] {
    // Layout per existing kernel comment block:
    //   thread tid owns 3 bytes at byte_off = 16 + tid*3
    //   q0..q7 packed cross-byte:  q0|q1|q2 in low/high of bytes 0,1,2
    // Concretely: pk = data[0] | (data[1] << 8) | (data[2] << 16)
    //             q_i = (pk >> (3*i)) & 7
    let mut out = [0u8; 96];
    for tid in 0..32 {
        let mut pk: u32 = 0;
        for i in 0..8 {
            let q = qs[tid * 8 + i] as u32 & 7;
            pk |= q << (3 * i);
        }
        out[tid * 3]     = (pk        & 0xff) as u8;
        out[tid * 3 + 1] = ((pk >>  8) & 0xff) as u8;
        out[tid * 3 + 2] = ((pk >> 16) & 0xff) as u8;
    }
    out
}

/// Builds the row bytes AND returns the round-tripped fp16→fp32 codebooks so
/// the CPU reference uses the SAME values the GPU will dequantize. Without
/// this, fp16 quantization noise (~1e-3 per cell) shows up as a bogus error.
fn build_lloyd_mq3_row(
    groups_per_row: usize,
    codebooks: &[[f32; 8]],
    indices: &[[u8; 256]],
) -> (Vec<u8>, Vec<[f32; 8]>) {
    let mut out = Vec::with_capacity(groups_per_row * 112);
    let mut roundtripped = Vec::with_capacity(groups_per_row);
    for g in 0..groups_per_row {
        let cb = &codebooks[g];
        let mut cb_rt = [0.0f32; 8];
        for (i, &v) in cb.iter().enumerate() {
            let bytes = f32_to_f16_le(v);
            out.extend_from_slice(&bytes);
            cb_rt[i] = f16_le_to_f32(bytes);
        }
        roundtripped.push(cb_rt);
        let packed = pack_3bit_group(&indices[g]);
        out.extend_from_slice(&packed);
    }
    assert_eq!(out.len(), groups_per_row * 112);
    (out, roundtripped)
}

fn cpu_reference(
    groups_per_row: usize,
    m: usize,
    a_rows: &[Vec<u8>],
    x: &[f32],
    codebooks_per_row: &[Vec<[f32; 8]>],
    indices_per_row: &[Vec<[u8; 256]>],
) -> Vec<f32> {
    let mut y = vec![0.0f32; m];
    for row in 0..m {
        let _row_bytes = &a_rows[row];
        let cbs = &codebooks_per_row[row];
        let idxs = &indices_per_row[row];
        let mut acc = 0.0f32;
        for g in 0..groups_per_row {
            let cb = &cbs[g];
            let qs = &idxs[g];
            for i in 0..256 {
                let q = qs[i] as usize & 7;
                acc += cb[q] * x[g * 256 + i];
            }
        }
        y[row] = acc;
    }
    y
}

fn run_one(gpu: &mut Gpu, groups_per_row: usize) -> (f32, f32) {
    let m = 64;
    let k = groups_per_row * 256;

    // Build deterministic per-row Lloyd-MQ3 data.
    let mut a_rows = Vec::with_capacity(m);
    let mut codebooks_per_row = Vec::with_capacity(m);
    let mut indices_per_row = Vec::with_capacity(m);
    for row in 0..m {
        let mut cbs = Vec::with_capacity(groups_per_row);
        let mut idxs = Vec::with_capacity(groups_per_row);
        for g in 0..groups_per_row {
            // Synthetic codebook: 8 ascending centroids around zero.
            // Different per (row, g) to avoid degenerate sums.
            let base = ((row * 7 + g * 11) % 19) as f32 * 0.013 - 0.1;
            let cb: [f32; 8] = std::array::from_fn(|i| base + (i as f32 - 3.5) * 0.025);
            cbs.push(cb);

            // Synthetic indices in [0, 8).
            let mut q = [0u8; 256];
            for i in 0..256 {
                q[i] = ((row.wrapping_mul(31) ^ g.wrapping_mul(53) ^ i.wrapping_mul(7)) & 7) as u8;
            }
            idxs.push(q);
        }
        let (row_bytes, cbs_rt) = build_lloyd_mq3_row(groups_per_row, &cbs, &idxs);
        a_rows.push(row_bytes);
        codebooks_per_row.push(cbs_rt);
        indices_per_row.push(idxs);
    }

    // x in [-0.5, 0.5)
    let x: Vec<f32> = (0..k).map(|i| ((i as i32 % 13) as f32 - 6.0) * 0.05).collect();

    // Concatenate rows into one buffer.
    let mut a_flat: Vec<u8> = Vec::with_capacity(m * groups_per_row * 112);
    for row in &a_rows {
        a_flat.extend_from_slice(row);
    }

    // Upload as raw bytes — GpuTensor with shape [a_flat.len()] (1-D byte buf).
    let d_a = gpu.upload_raw(&a_flat, &[a_flat.len()]).unwrap();
    let d_x = gpu.upload_f32(&x, &[k]).unwrap();
    let d_y = gpu.zeros(&[m], DType::F32).unwrap();

    gpu.gemv_mq3g256_lloyd(&d_a, &d_x, &d_y, m, k).unwrap();
    let y_gpu = gpu.download_f32(&d_y).unwrap();

    let y_ref = cpu_reference(groups_per_row, m, &a_rows, &x, &codebooks_per_row, &indices_per_row);

    let mut max_abs = 0f32;
    let mut max_rel = 0f32;
    for i in 0..m {
        let abs = (y_gpu[i] - y_ref[i]).abs();
        let denom = y_ref[i].abs().max(1e-6);
        max_abs = max_abs.max(abs);
        max_rel = max_rel.max(abs / denom);
    }
    gpu.free_tensor(d_a).unwrap();
    gpu.free_tensor(d_x).unwrap();
    gpu.free_tensor(d_y).unwrap();
    (max_abs, max_rel)
}

fn main() {
    let mut gpu = Gpu::init().expect("GPU init failed");
    eprintln!("GPU: {}", gpu.arch);

    let mut all_pass = true;
    for &gpr in &[4usize, 5, 6, 7, 8] {
        let (max_abs, max_rel) = run_one(&mut gpu, gpr);
        let pass = max_abs < 1e-3;
        let tag = if pass { "PASS" } else { "FAIL" };
        let g_layout = match gpr {
            4 => "(4 quads, 0 tail)",
            5 => "(1 quad,  1 tail)",
            6 => "(1 quad,  2 tail)",
            7 => "(1 quad,  3 tail)",
            8 => "(2 quads, 0 tail)",
            _ => "(?)",
        };
        println!(
            "groups_per_row={gpr} K={:5}  max_abs={max_abs:.3e}  max_rel={max_rel:.3e}  {tag}  {g_layout}",
            gpr * 256
        );
        if !pass { all_pass = false; }
    }
    if !all_pass {
        eprintln!("\nFAIL: one or more tail cases produced max_abs > 1e-3");
        std::process::exit(1);
    }
    println!("\nALL PASS — quad-clean and all 3 tail sizes match CPU reference");
}
