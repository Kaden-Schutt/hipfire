//! Phase A parity test for gemm_mq3g256_lloyd_residual_wmma.
//!
//! Mirrors test_gemv_mq3g256_lloyd_tail's reference structure but for batched
//! WMMA GEMM (Y[col*M + row] += sum_k A[row][k] * X[col][k]). Sweeps a
//! representative subset of Phase A shapes from the plan: (M, K, N) ∈
//! {64, 256, 1024} × {1024, 4096, 12288} × {16, 64, 256}.
//!
//! CPU reference uses fp64 accumulation for a clean ground truth; max-abs
//! tolerance is logged-then-set empirically per Phase A acceptance criterion
//! (plan §"Phase A": "tolerance is measured-and-set, not specified upfront").

use rdna_compute::{Gpu, DType};

/// f32 → IEEE 754 binary16 little-endian, RTNE on dropped 13 mantissa bits.
/// Matches gemv_mq3g256_lloyd_tail's helper exactly so f16-roundtripped values
/// agree with the GPU's __half2float read side.
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

fn pack_3bit_group(qs: &[u8; 256]) -> [u8; 96] {
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

/// Builds row bytes AND returns f16-roundtripped codebooks so the CPU
/// reference uses exactly the values the GPU will dequant.
fn build_lloyd_row(
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

/// CPU reference: y[col*m + row] = y_init[col*m + row] + sum_k(A[row][k] * X[col][k]).
/// Inner accumulation in f64 for a clean ground truth; X is also f16-roundtripped
/// to match what the GPU sees after fp32→fp16 conversion in `ensure_fp16_x`.
fn cpu_reference_gemm(
    m: usize, k: usize, n: usize,
    codebooks_per_row: &[Vec<[f32; 8]>],
    indices_per_row: &[Vec<[u8; 256]>],
    x_fp32: &[f32],
    y_init: &[f32],
) -> Vec<f32> {
    let groups_per_row = k / 256;
    // Roundtrip X through f16 to match the GPU's view.
    let x_rt: Vec<f32> = x_fp32.iter().map(|&v| f16_le_to_f32(f32_to_f16_le(v))).collect();
    let mut y = y_init.to_vec();
    for col in 0..n {
        for row in 0..m {
            let mut acc = 0.0f64;
            let cbs = &codebooks_per_row[row];
            let idxs = &indices_per_row[row];
            for g in 0..groups_per_row {
                let cb = &cbs[g];
                let qs = &idxs[g];
                for i in 0..256 {
                    let q = qs[i] as usize & 7;
                    let k_idx = g * 256 + i;
                    acc += cb[q] as f64 * x_rt[col * k + k_idx] as f64;
                }
            }
            y[col * m + row] += acc as f32;
        }
    }
    y
}

fn run_one(gpu: &mut Gpu, m: usize, k: usize, n: usize) -> (f32, f32, f32) {
    assert_eq!(k % 256, 0, "K must be a multiple of 256");
    let groups_per_row = k / 256;

    let mut a_rows = Vec::with_capacity(m);
    let mut codebooks_per_row = Vec::with_capacity(m);
    let mut indices_per_row = Vec::with_capacity(m);
    for row in 0..m {
        let mut cbs = Vec::with_capacity(groups_per_row);
        let mut idxs = Vec::with_capacity(groups_per_row);
        for g in 0..groups_per_row {
            // Synthetic codebook: 8 ascending centroids around zero, different
            // per (row, g) so different rows produce distinguishable outputs.
            let base = ((row.wrapping_mul(7) + g.wrapping_mul(11)) % 19) as f32 * 0.013 - 0.1;
            let cb: [f32; 8] = std::array::from_fn(|i| base + (i as f32 - 3.5) * 0.025);
            cbs.push(cb);

            // Synthetic indices in [0, 8).
            let mut q = [0u8; 256];
            for i in 0..256 {
                q[i] = ((row.wrapping_mul(31) ^ g.wrapping_mul(53) ^ i.wrapping_mul(7)) & 7) as u8;
            }
            idxs.push(q);
        }
        let (row_bytes, cbs_rt) = build_lloyd_row(groups_per_row, &cbs, &idxs);
        a_rows.push(row_bytes);
        codebooks_per_row.push(cbs_rt);
        indices_per_row.push(idxs);
    }

    // X in [-0.3, 0.3) — fp16-friendly magnitude.
    let x: Vec<f32> = (0..(n * k))
        .map(|i| ((i as i32 % 13) as f32 - 6.0) * 0.05)
        .collect();
    // Y_init small but nonzero so we test the residual semantics.
    let y_init: Vec<f32> = (0..(n * m))
        .map(|i| ((i as i32 % 11) as f32 - 5.0) * 0.001)
        .collect();

    let mut a_flat: Vec<u8> = Vec::with_capacity(m * groups_per_row * 112);
    for row in &a_rows {
        a_flat.extend_from_slice(row);
    }

    let d_a = gpu.upload_raw(&a_flat, &[a_flat.len()]).unwrap();
    let d_x = gpu.upload_f32(&x, &[n, k]).unwrap();
    let d_y = gpu.upload_f32(&y_init, &[n, m]).unwrap();

    gpu.gemm_mq3g256_lloyd_residual_wmma(&d_a, &d_x, &d_y, m, k, n).unwrap();
    let y_gpu = gpu.download_f32(&d_y).unwrap();

    let y_ref = cpu_reference_gemm(m, k, n, &codebooks_per_row, &indices_per_row, &x, &y_init);

    let mut max_abs = 0f32;
    let mut max_rel = 0f32;
    let mut sum_sq_err = 0.0f64;
    let total = n * m;
    for i in 0..total {
        let abs = (y_gpu[i] - y_ref[i]).abs();
        let denom = y_ref[i].abs().max(1e-3);
        max_abs = max_abs.max(abs);
        max_rel = max_rel.max(abs / denom);
        sum_sq_err += (abs as f64) * (abs as f64);
    }
    let rms_err = (sum_sq_err / total as f64).sqrt() as f32;

    gpu.free_tensor(d_a).unwrap();
    gpu.free_tensor(d_x).unwrap();
    gpu.free_tensor(d_y).unwrap();
    (max_abs, max_rel, rms_err)
}

fn main() {
    let mut gpu = Gpu::init().expect("GPU init failed");
    eprintln!("GPU: {}", gpu.arch);

    // Phase A canonical shapes (subset of (M, K, N) ∈
    // {64, 256, 1024} × {1024, 4096, 12288} × {16, 64, 256}). Selected to cover
    // small/medium/large extents without exploding total kernel time.
    let cases: &[(usize, usize, usize)] = &[
        (64,   1024,  16),  // smallest — single tile
        (64,   1024,  64),  // canonical small
        (256,  1024,  64),  // wider M
        (64,   4096,  64),  // longer K
        (256,  4096,  16),
        (256,  4096, 256),  // 16×16 tile sweep
        (1024, 4096,  64),  // wider M
        (1024, 12288, 64),  // qwen3.5-9b mlp.down_proj K dim
    ];

    // Tightened post-Phase-A from the initial 5e-3 budget. Worst observed
    // max-abs across all shapes is 5.83e-5 at K=12288 (Phase A devlog
    // 2026-05-07). 2e-4 = ~3× observed, matching the fused-test tolerance
    // discipline (test_gemm_fused_mq3g256_lloyd_wmma.rs:162 = 1.75e-4).
    // The original 5e-3 was 86× looser than observed and would silently
    // pass 1e-4-class regressions — flagged in the multi-reviewer code
    // review at docs/plans/mq3-lloyd-wmma-code-rev-claude.md (S3).
    let phase_a_tolerance = 2e-4f32;

    let mut all_pass = true;
    let mut global_max_abs = 0f32;
    println!("{:>5} {:>6} {:>4}  {:>11}  {:>11}  {:>11}  {}",
             "M", "K", "N", "max_abs", "max_rel", "rms", "verdict");

    for &(m, k, n) in cases {
        let (max_abs, max_rel, rms) = run_one(&mut gpu, m, k, n);
        let pass = max_abs < phase_a_tolerance;
        let tag = if pass { "PASS" } else { "FAIL" };
        println!(
            "{:>5} {:>6} {:>4}  {:>11.3e}  {:>11.3e}  {:>11.3e}  {tag}",
            m, k, n, max_abs, max_rel, rms
        );
        if !pass { all_pass = false; }
        if max_abs > global_max_abs { global_max_abs = max_abs; }
    }
    println!();
    println!("Max-abs across all shapes  : {:.3e}", global_max_abs);
    println!("Phase A tolerance (initial): {:.3e}", phase_a_tolerance);

    if !all_pass {
        eprintln!("\nFAIL: one or more shapes exceeded {} absolute", phase_a_tolerance);
        std::process::exit(1);
    }
    println!("\nALL PASS");
}
