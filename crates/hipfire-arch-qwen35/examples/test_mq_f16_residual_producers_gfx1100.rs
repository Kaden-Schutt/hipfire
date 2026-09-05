// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt

//! S4-f16-residual-inputs parity gate on exact gfx1100.
//!
//! For N in {1, 8, 16}, head layouts {32x128 (LA), 48x128 (FA)}, AWQ
//! absent/present, and nonzero initial residuals, requires:
//!  1. producer F16 memcmp: each S4 producer's sidecar must equal the old
//!     F32 pipeline (gated_norm+rotate / sigmoid_mul+rotate /
//!     fused_silu_mul_rotate, plain and AWQ) followed by `convert_f32_to_f16`
//!     — compared via an exact host round-to-nearest-even conversion that is
//!     self-tested on boundary values below (HW `v_cvt_f16_f32` semantics).
//!  2. final residual-output memcmp: old
//!     `gemm_mq4g256v2_residual_wmma` (F32 X, internal convert) vs new
//!     `gemm_mq4g256v2_residual_wmma_f16` (sidecar X) agree byte-for-byte on
//!     the same nonzero Y init — pure GPU-vs-GPU, no host conversion.
//!
//! Weight bytes are synthetic random (parity needs identical inputs, not
//! meaningful weights). On any other arch the harness SKIPs cleanly
//! (exit 0, no GPU work).

use rdna_compute::{DType, Gpu};

const NS: [usize; 3] = [1, 8, 16];
const EPS: f32 = 1e-5;
const RES_M: usize = 256;

// ── deterministic PRNG (xorshift64*) ──────────────────────────────────────
struct Rng(u64);
impl Rng {
    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
    fn next_f32(&mut self, lo: f32, hi: f32) -> f32 {
        // Uniform in [lo, hi) from the top 24 bits — always finite.
        let u = ((self.next_u64() >> 11) as f32) / ((1u64 << 53) as f32);
        lo + (hi - lo) * u
    }
}

fn rand_vec(rng: &mut Rng, n: usize, lo: f32, hi: f32) -> Vec<f32> {
    (0..n).map(|_| rng.next_f32(lo, hi)).collect()
}

// ── exact host f32 -> f16 bits (round-to-nearest-even) ────────────────────
//
// Matches hardware `v_cvt_f16_f32` (what `convert_f32_to_f16`'s
// `(_Float16)` cast lowers to): RN-even mantissa rounding, subnormals,
// overflow to Inf, NaN payload preservation (quieted).
fn f32_to_f16_bits(x: f32) -> u16 {
    let b = x.to_bits();
    let sign = ((b >> 16) & 0x8000) as u16;
    let exp = ((b >> 23) & 0xff) as i32;
    let mant = b & 0x007f_ffff;
    if exp == 0xff {
        // Inf / NaN: keep payload (quiet bit forced, like cvt).
        return if mant == 0 {
            sign | 0x7c00
        } else {
            sign | 0x7c00 | (((mant >> 13) as u16) | 0x0200)
        };
    }
    let e = exp - 127; // unbiased exponent
    if e > 15 {
        return sign | 0x7c00; // overflow -> Inf
    }
    if e >= -14 {
        // Normal range: round the 24-bit significand to 11 bits, RN-even.
        let m = mant | 0x0080_0000;
        let rest = m & 0x1fff;
        let mut hm = (m >> 13) as u16; // 11 bits incl. hidden 1
        if rest > 0x1000 || (rest == 0x1000 && (hm & 1) == 1) {
            hm += 1;
            if hm == 0x0800 {
                // Mantissa overflow carries into the exponent.
                return sign | (((e + 16) as u16) << 10);
            }
        }
        return sign | (((e + 15) as u16) << 10) | (hm & 0x03ff);
    }
    if e < -26 {
        return sign; // rounds to zero (max magnitude < quarter-ulp)
    }
    // Subnormal range e in [-26, -15]: value = M24 * 2^(e-23); one
    // subnormal ulp = 2^-24, so round M24 * 2^(e+1) to int, RN-even.
    let m = mant | 0x0080_0000;
    let shift = (-e - 1) as u32; // 14..=25
    let half = 1u32 << (shift - 1);
    let rest = m & (half * 2 - 1);
    let mut m10 = (m >> shift) as u16;
    if rest > half || (rest == half && (m10 & 1) == 1) {
        m10 += 1;
        if m10 == 0x0400 {
            // Rounded up to the smallest normal (2^-14).
            return sign | 0x0400;
        }
    }
    sign | (m10 & 0x03ff)
}

fn self_test_conversion() {
    // (f32 bits, expected f16 bits)
    let cases: &[(u32, u16)] = &[
        (0x0000_0000, 0x0000), // +0
        (0x8000_0000, 0x8000), // -0
        (0x3f80_0000, 0x3c00), // 1
        (0xbf80_0000, 0xbc00), // -1
        (0x3880_0000, 0x0400), // 2^-14 (smallest normal)
        (0x387f_e000, 0x0400), // tie halfway 0x03FF/0x0400 -> even (0x0400)
        (0x387f_c000, 0x03ff), // 0x03FF exact (1023 subnormal ulps)
        (0x3380_0000, 0x0001), // 2^-24 (smallest subnormal)
        (0x3300_0000, 0x0000), // 2^-25: exact tie at half min-subnormal -> even (0)
        (0x7f80_0000, 0x7c00), // +Inf
        (0xff80_0000, 0xfc00), // -Inf
        (0x477f_e000, 0x7bff), // 65504 (max f16)
        (0x4780_0000, 0x7c00), // 65536 -> Inf
        (0x3dcc_cccd, 0x2e66), // 0.1f
        (0x4049_0fdb, 0x4248), // pi
    ];
    for &(fb, expected) in cases {
        let want = expected;
        let got = f32_to_f16_bits(f32::from_bits(fb));
        assert_eq!(
            got, want,
            "host f32->f16 mismatch for {:08x}: got {:04x} want {:04x}",
            fb, got, want
        );
    }
    // Exhaustive-ish sweep over small magnitudes incl. subnormal ties:
    // compare against f64-based RN-even reference.
    let mut rng = Rng(0x1234_5678_9abc_def0);
    for _ in 0..200_000 {
        let fb = rng.next_u64() as u32;
        let x = f32::from_bits(fb);
        if !x.is_finite() {
            continue;
        }
        let got = f32_to_f16_bits(x);
        let want = f64_ref(x);
        assert_eq!(got, want, "sweep mismatch for {:08x} ({:e})", fb, x);
    }
}

/// Independent f64 reference: nearest f16 grid value, ties to even.
fn f64_ref(x: f32) -> u16 {
    let v = x as f64;
    if v == 0.0 {
        return if x.to_bits() & 0x8000_0000 == 0 {
            0
        } else {
            0x8000
        };
    }
    let sign = if v < 0.0 { 0x8000u16 } else { 0 };
    let a = v.abs();
    if a.is_infinite() || a >= 65520.0 {
        // Halfway between 65504 and Inf is (65504+65536)/2 = 65520.
        return sign | 0x7c00;
    }
    // Grid spacing depends on magnitude; emulate by scaling.
    // Candidate: brute-force over neighbor integers is overkill —
    // use frexp-style scaling to an integer grid.
    let exp2 = a.log2().floor() as i32;
    // Normal f16 spacing at this binade: 2^(exp2-10); subnormal: 2^-24.
    let ulp = if exp2 >= -14 {
        2f64.powi(exp2 - 10)
    } else {
        2f64.powi(-24)
    };
    let q = a / ulp;
    // RN-even to integer.
    let lo = q.floor();
    let frac = q - lo;
    let mut qi = if frac > 0.5 || (frac == 0.5 && (lo as u64 % 2 == 1)) {
        lo + 1.0
    } else {
        lo
    };
    // Re-encode; handle carry into next binade by recomputing.
    let rounded = qi * ulp;
    if rounded >= 65520.0 {
        return sign | 0x7c00;
    }
    if rounded == 0.0 {
        return sign;
    }
    // Encode the rounded value exactly (it is on-grid by construction).
    let e2 = rounded.log2().floor() as i32;
    if e2 >= -14 {
        let mant = ((rounded / 2f64.powi(e2) - 1.0) * 1024.0).round() as u16;
        if mant == 1024 {
            return sign | (((e2 + 16) as u16) << 10);
        }
        sign | (((e2 + 15) as u16) << 10) | mant
    } else {
        qi = (rounded / 2f64.powi(-24)).round();
        if qi >= 1024.0 {
            return sign | 0x0400;
        }
        sign | (qi as u16)
    }
}

// ── gpu helpers ───────────────────────────────────────────────────────────
fn dtoh_bytes(gpu: &Gpu, t: &rdna_compute::GpuTensor) -> Vec<u8> {
    let mut b = vec![0u8; t.buf.size()];
    gpu.hip.memcpy_dtoh(&mut b, &t.buf).unwrap();
    b
}

fn check_f16(tag: &str, got_bytes: &[u8], want_f32: &[f32]) -> bool {
    assert_eq!(got_bytes.len(), want_f32.len() * 2);
    let mut bad = 0;
    for (i, &w) in want_f32.iter().enumerate() {
        let got = u16::from_le_bytes([got_bytes[2 * i], got_bytes[2 * i + 1]]);
        let want = f32_to_f16_bits(w);
        if got != want {
            if bad < 8 {
                eprintln!(
                    "  MISMATCH {tag}[{i}]: f32={:e} want_f16={:04x} got_f16={:04x}",
                    w, want, got
                );
            }
            bad += 1;
        }
    }
    if bad > 0 {
        eprintln!("  {tag}: {bad}/{} words differ", want_f32.len());
        return false;
    }
    println!("  {tag}: producer F16 memcmp ok ({} words)", want_f32.len());
    true
}

fn main() {
    self_test_conversion();
    println!("[s4] host f32->f16 conversion self-test ok");

    let mut gpu = match Gpu::init() {
        Ok(g) => g,
        Err(e) => {
            eprintln!("SKIP: no GPU ({e})");
            return;
        }
    };
    let arch = gpu.arch.clone();
    if !(gpu.arch_caps.is_gfx1100() && arch == "gfx1100") {
        eprintln!("SKIP: arch {arch} is not exact gfx1100 — harness requires gfx1100 only");
        return;
    }
    println!("[s4] arch {arch} confirmed exact gfx1100");

    let mut rng = Rng(0x9e37_79b9_7f4a_7c15);
    let mut fails = 0;

    // Family P1: LA post-GDN — (n_heads, hd) in {(32,128), (48,128)} x {plain, awq}.
    for &(nh, hd) in &[(32usize, 128usize), (48usize, 128usize)] {
        for &awq in &[false, true] {
            for &n in &NS {
                let k = nh * hd;
                if !run_p1(&mut gpu, &mut rng, n, nh, hd, k, awq) {
                    fails += 1;
                }
            }
        }
    }
    // Family P2: FA post-attention — K in {4096, 6144} x {plain, awq}.
    for &k in &[4096usize, 6144usize] {
        for &awq in &[false, true] {
            for &n in &NS {
                if !run_p2(&mut gpu, &mut rng, n, k, awq) {
                    fails += 1;
                }
            }
        }
    }
    // Family P3: FFN down — K in {4096, 8192} x {plain, awq}, plus K=768
    // plain (split-K table miss -> base-kernel mirror arm).
    for &k in &[4096usize, 8192usize] {
        for &awq in &[false, true] {
            for &n in &NS {
                if !run_p3(&mut gpu, &mut rng, n, k, awq) {
                    fails += 1;
                }
            }
        }
    }
    for &n in &NS {
        if !run_p3(&mut gpu, &mut rng, n, 768, false) {
            fails += 1;
        }
    }

    if fails > 0 {
        eprintln!("[s4] FAIL: {fails} case(s) mismatched");
        std::process::exit(1);
    }
    println!("[s4] PASS: all producer + residual-output memcmps exact");
}

/// P1 oracle: gated_norm_f32_batched + rotate_x_mq[_awq]_batched.
fn run_p1(
    gpu: &mut Gpu,
    rng: &mut Rng,
    n: usize,
    nh: usize,
    hd: usize,
    k: usize,
    awq: bool,
) -> bool {
    let tag = format!("p1 nh={nh}x{hd} n={n} awq={awq}");
    let x = rand_vec(rng, n * k, -2.0, 2.0);
    let z = rand_vec(rng, n * k, -2.0, 2.0);
    let w = rand_vec(rng, hd, 0.5, 1.5);
    let dx = gpu.upload_f32(&x, &[n * k]).unwrap();
    let dz = gpu.upload_f32(&z, &[n * k]).unwrap();
    let dw = gpu.upload_f32(&w, &[hd]).unwrap();
    // Oracle arm.
    let d_norm = gpu.alloc_tensor(&[n * k], DType::F32).unwrap();
    gpu.gated_norm_f32_batched(&dx, &dz, &dw, &d_norm, nh, hd, EPS, n)
        .unwrap();
    let d_rot = gpu.alloc_tensor(&[n * k], DType::F32).unwrap();
    let dawq = if awq {
        Some(gpu.upload_f32(&rand_vec(rng, k, 0.5, 2.0), &[k]).unwrap())
    } else {
        None
    };
    if let Some(ref a) = dawq {
        gpu.rotate_x_mq_awq_batched(&d_norm, a, &d_rot, k, n)
            .unwrap();
    } else {
        gpu.rotate_x_mq_batched(&d_norm, &d_rot, k, n).unwrap();
    }
    let rot_f32 = gpu.download_f32(&d_rot).unwrap();
    // Candidate arm.
    let d_out = gpu.alloc_tensor(&[n * k], DType::F16).unwrap();
    if let Some(ref a) = dawq {
        gpu.gated_norm_rotate_mq_awq_f16_batched(&dx, &dz, &dw, a, &d_out, nh, hd, EPS, n)
            .unwrap();
    } else {
        gpu.gated_norm_rotate_mq_f16_batched(&dx, &dz, &dw, &d_out, nh, hd, EPS, n)
            .unwrap();
    }
    let got = dtoh_bytes(gpu, &d_out);
    let mut ok = check_f16(&tag, &got, &rot_f32);
    ok &= run_residual(gpu, rng, &tag, &d_rot, &d_out, RES_M, k, n);
    ok
}

/// P2 oracle: sigmoid_mul_f32 (in-place on its own copy) + rotate.
fn run_p2(gpu: &mut Gpu, rng: &mut Rng, n: usize, k: usize, awq: bool) -> bool {
    let tag = format!("p2 K={k} n={n} awq={awq}");
    let attn = rand_vec(rng, n * k, -2.0, 2.0);
    let gate = rand_vec(rng, n * k, -3.0, 3.0);
    let d_attn = gpu.upload_f32(&attn, &[n * k]).unwrap();
    let d_gate = gpu.upload_f32(&gate, &[n * k]).unwrap();
    // Oracle arm on its own attn copy (sigmoid_mul is in-place).
    let d_sig = gpu.upload_f32(&attn, &[n * k]).unwrap();
    gpu.sigmoid_mul_f32(&d_sig, &d_gate).unwrap();
    let d_rot = gpu.alloc_tensor(&[n * k], DType::F32).unwrap();
    let dawq = if awq {
        Some(gpu.upload_f32(&rand_vec(rng, k, 0.5, 2.0), &[k]).unwrap())
    } else {
        None
    };
    if let Some(ref a) = dawq {
        gpu.rotate_x_mq_awq_batched(&d_sig, a, &d_rot, k, n)
            .unwrap();
    } else {
        gpu.rotate_x_mq_batched(&d_sig, &d_rot, k, n).unwrap();
    }
    let rot_f32 = gpu.download_f32(&d_rot).unwrap();
    // Candidate arm (pristine attn — never sigmoided in place).
    let d_out = gpu.alloc_tensor(&[n * k], DType::F16).unwrap();
    if let Some(ref a) = dawq {
        gpu.sigmoid_mul_rotate_mq_awq_f16_batched(&d_attn, &d_gate, a, &d_out, k, n)
            .unwrap();
    } else {
        gpu.sigmoid_mul_rotate_mq_f16_batched(&d_attn, &d_gate, &d_out, k, n)
            .unwrap();
    }
    let got = dtoh_bytes(gpu, &d_out);
    let mut ok = check_f16(&tag, &got, &rot_f32);
    ok &= run_residual(gpu, rng, &tag, &d_rot, &d_out, RES_M, k, n);
    ok
}

/// P3 oracle: fused_silu_mul_rotate_mq[_awq]_batched.
fn run_p3(gpu: &mut Gpu, rng: &mut Rng, n: usize, k: usize, awq: bool) -> bool {
    let tag = format!("p3 K={k} n={n} awq={awq}");
    let gate = rand_vec(rng, n * k, -3.0, 3.0);
    let up = rand_vec(rng, n * k, -2.0, 2.0);
    let d_gate = gpu.upload_f32(&gate, &[n * k]).unwrap();
    let d_up = gpu.upload_f32(&up, &[n * k]).unwrap();
    let d_rot = gpu.alloc_tensor(&[n * k], DType::F32).unwrap();
    let dawq = if awq {
        Some(gpu.upload_f32(&rand_vec(rng, k, 0.5, 2.0), &[k]).unwrap())
    } else {
        None
    };
    if let Some(ref a) = dawq {
        gpu.fused_silu_mul_rotate_mq_awq_batched(&d_gate, &d_up, a, &d_rot, k, n)
            .unwrap();
    } else {
        gpu.fused_silu_mul_rotate_mq_batched(&d_gate, &d_up, &d_rot, k, n)
            .unwrap();
    }
    let rot_f32 = gpu.download_f32(&d_rot).unwrap();
    let d_out = gpu.alloc_tensor(&[n * k], DType::F16).unwrap();
    if let Some(ref a) = dawq {
        gpu.fused_silu_mul_rotate_mq_awq_f16_batched(&d_gate, &d_up, a, &d_out, k, n)
            .unwrap();
    } else {
        gpu.fused_silu_mul_rotate_mq_f16_batched(&d_gate, &d_up, &d_out, k, n)
            .unwrap();
    }
    let got = dtoh_bytes(gpu, &d_out);
    let mut ok = check_f16(&tag, &got, &rot_f32);
    ok &= run_residual(gpu, rng, &tag, &d_rot, &d_out, RES_M, k, n);
    ok
}

/// Residual-output memcmp: old F32-X GEMM vs new sidecar-X GEMM on
/// identical synthetic weights and identical nonzero Y init.
fn run_residual(
    gpu: &mut Gpu,
    rng: &mut Rng,
    tag: &str,
    x_f32: &rdna_compute::GpuTensor,
    x_f16: &rdna_compute::GpuTensor,
    m: usize,
    k: usize,
    n: usize,
) -> bool {
    let groups = k / 256;
    let wbytes = m * groups * 136; // MQ4V2: 136 B/group
    let mut wb = vec![0u8; wbytes];
    for b in wb.iter_mut() {
        *b = (rng.next_u64() & 0xff) as u8;
    }
    let dw = gpu.upload_raw(&wb, &[wbytes]).unwrap();
    let y0 = rand_vec(rng, n * m, -1.0, 1.0);
    let dy_old = gpu.upload_f32(&y0, &[n * m]).unwrap();
    let dy_new = gpu.upload_f32(&y0, &[n * m]).unwrap();
    gpu.gemm_mq4g256v2_residual_wmma(&dw, x_f32, &dy_old, m, k, n)
        .unwrap();
    let x_view = x_f16.sub_offset(0, n * k);
    gpu.gemm_mq4g256v2_residual_wmma_f16(&dw, &x_view, &dy_new, m, k, n)
        .unwrap();
    let yo = gpu.download_f32(&dy_old).unwrap();
    let yn = gpu.download_f32(&dy_new).unwrap();
    if yo.len() != yn.len() {
        eprintln!("  {tag}: residual len mismatch");
        return false;
    }
    let mut bad = 0;
    for (i, (&a, &b)) in yo.iter().zip(yn.iter()).enumerate() {
        if a.to_bits() != b.to_bits() {
            if bad < 8 {
                eprintln!("  RESIDUAL MISMATCH {tag}[{i}]: old={:e} new={:e}", a, b);
            }
            bad += 1;
        }
    }
    if bad > 0 {
        eprintln!("  {tag}: residual {bad}/{} words differ", yo.len());
        return false;
    }
    println!("  {tag}: residual-output memcmp ok ({} words)", yo.len());
    true
}
