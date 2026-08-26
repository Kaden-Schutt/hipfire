// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026
// hipfire — see LICENSE and NOTICE in the project root.

//! Task 8 — GPU vs CPU correctness gate for `gemv_bq1g128`
//! (PrismML Q1_0-layout binary/1-bit decode GEMV, `kernels/src/gemv_bq1g128.hip`).
//!
//! This is the on-device correctness proof for the 1-bit decode GEMV on
//! gfx1151, mirroring `test_gemv_tq2g128.rs` (the ternary Q2_0 sibling) but
//! for the binary Q1_0 layout: exercises the full gemv (per-row group loop,
//! per-lane 4-element decode, warp-shuffle tree reduction) against `x`.
//!
//! Block layout (18 bytes per 128-weight group), matches
//! `reference/hfq1g128/q1_0.rs` and the kernel body:
//!   [FP16 d (2B)][16 packed sign-bit bytes, 8 bits/byte, LSB-first]
//!   element e = 8*j + i (byte j, bit i) -> bit==1 => +d, bit==0 => -d
//!
//! Launch geometry (from the kernel body): grid [M,1,1], block [32,1,1] —
//! ONE warp per output row, each lane handles 4 elements per group
//! (byte_idx = tid>>1, bit shift = (tid&1)*4) and the group loop runs
//! sequentially inside the kernel (unlike the dequant kernel's
//! [M, groups_per_row, 1] grid).
//!
//! Two shapes are tested to exercise the group loop (same shape pair as the
//! ternary sibling):
//!   - M=64, K=128  (groups_per_row=1, single group, no loop)
//!   - M=48, K=512  (groups_per_row=4, exercises the accumulate-across-groups path)
//!
//! Tolerance: the GPU accumulates via `float acc` across `groups_per_row`
//! sequential FMAs plus a warp-shuffle butterfly reduction over 32 lanes —
//! a DIFFERENT summation order than the CPU oracle's straight left-to-right
//! sum. f32 has ~7 decimal digits of precision, so reduction-order error
//! accumulates roughly in proportion to the magnitude of the sum; bit-exact
//! equality is not expected. We use a combined relative+absolute bound:
//!   max_abs_err < 1e-3 * (1.0 + max|expected|)
//! which is generous enough to absorb K=512 reduction-order drift while
//! still tight enough to catch a wrong block-indexing/stride/reduction bug
//! (which would produce errors many orders of magnitude larger, as the
//! negative control below demonstrates).
//!
//! Includes a negative control: an intentionally WRONG CPU oracle (treating
//! every bit as `+d`, i.e. dropping the sign bit entirely) must NOT match
//! the GPU output. This proves the parity assertion actually has
//! discriminating power.
//!
//! Run:
//!   cd ~/.hipfire/src
//!   cargo run -p rdna-compute --example test_gemv_bq1g128 --release

use hip_bridge::KernargBlob;
use rdna_compute::{DType, Gpu};
use std::ffi::c_void;

// `rdna_compute::kernels` is a private module (only `GEMV_SRC` is
// re-exported from lib.rs), so — mirroring the established pattern in
// bench_q8_wmma_variants.rs, test_dequant_tq2g128.rs, and
// test_gemv_tq2g128.rs — this example includes the kernel source directly
// rather than adding new public surface to the crate.
const KERNEL_SRC: &str = include_str!("../../../kernels/src/gemv_bq1g128.hip");

const GROUP: usize = 128;
const BLOCK_BYTES: usize = 18; // 2 (fp16 d) + 16 (128 sign bits @ 1 bit/element)
const KERNEL_NAME: &str = "gemv_bq1g128";

// ── fp16 helpers (rdna-compute examples can't depend on hipfire-runtime;
// copied verbatim from the established pattern in test_gemv_tq2g128.rs /
// test_dequant_tq2g128.rs / test_gemv_mfp4g32_p.rs / bench_q8_wmma_variants.rs) ──
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

fn f16_bits_to_f32(h: u16) -> f32 {
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
            (sign << 31) | (((e + 127 - 14) as u32) << 23) | ((m & 0x3ff) << 13)
        }
    } else if exp == 0x1f {
        (sign << 31) | (0xff << 23) | (mant << 13)
    } else {
        (sign << 31) | (((exp - 15 + 127) as u32) << 23) | (mant << 13)
    };
    f32::from_bits(bits)
}

/// Tiny deterministic LCG PRNG (no external rand dep needed for a
/// self-contained correctness example).
struct Lcg(u32);
impl Lcg {
    fn next_u32(&mut self) -> u32 {
        self.0 = self.0.wrapping_mul(1664525).wrapping_add(1013904223);
        self.0
    }
    /// Uniform f32 in [0, 1).
    fn next_f32(&mut self) -> f32 {
        (self.next_u32() >> 8) as f32 / (1u32 << 24) as f32
    }
    /// Uniform sign bit in {0, 1}.
    fn next_bit(&mut self) -> u8 {
        (self.next_u32() & 0x1) as u8
    }
    /// Uniform f32 in [-1, 1).
    fn next_signed_f32(&mut self) -> f32 {
        self.next_f32() * 2.0 - 1.0
    }
}

/// Run one (M, K) shape end-to-end. Returns true iff PARITY OK and
/// NEG-CONTROL OK both hold for this shape.
fn run(gpu: &mut Gpu, m: usize, k: usize, seed: u32) -> bool {
    let groups_per_row = k / GROUP;
    assert_eq!(groups_per_row * GROUP, k, "K must be a multiple of 128");
    let row_bytes = groups_per_row * BLOCK_BYTES;
    let total_bytes = m * row_bytes;

    let mut rng = Lcg(seed);

    // ── 1. Generate synthetic packed Q1_0 binary weight blocks ─────────────
    let mut packed = vec![0u8; total_bytes];
    let mut d_scales = vec![0f32; m * groups_per_row];
    let mut bits = vec![0u8; m * k]; // row-major, matches kernel's traversal order

    for row in 0..m {
        for g in 0..groups_per_row {
            // Positive scale in (0.02, 1.5) — d is always > 0 for Q1_0
            // (sum-of-abs-values-based scale; never negative).
            let d = 0.02 + rng.next_f32() * (1.5 - 0.02);
            let d_h = f32_to_f16_bits(d);
            // Round-trip through fp16 so the CPU oracle uses the SAME
            // quantized scale the GPU kernel actually reads (the kernel
            // loads a _Float16 and widens to float).
            let d_rt = f16_bits_to_f32(d_h);
            d_scales[row * groups_per_row + g] = d_rt;

            let block_off = row * row_bytes + g * BLOCK_BYTES;
            packed[block_off] = (d_h & 0xFF) as u8;
            packed[block_off + 1] = (d_h >> 8) as u8;

            // element e = 8*j + i (byte j, bit i, LSB-first): bit=1 -> +d, bit=0 -> -d.
            for j in 0..GROUP {
                let bit = rng.next_bit();
                bits[row * k + g * GROUP + j] = bit;
                let byte_idx = j >> 3;
                let bit_offset = j & 7;
                if bit == 1 {
                    packed[block_off + 2 + byte_idx] |= 1u8 << bit_offset;
                }
            }
        }
    }

    // ── 2. Random f32 activation x[K] in [-1, 1) ────────────────────────────
    let x: Vec<f32> = (0..k).map(|_| rng.next_signed_f32()).collect();

    // ── 3. CPU oracle: expected[row] = sum_k (bit ? +d : -d) * x[k] ─────────
    // Negative-control oracle: intentionally WRONG map (drops the sign bit
    // entirely — every element treated as +d).
    let mut expected = vec![0f32; m];
    let mut expected_wrong = vec![0f32; m];
    for row in 0..m {
        let mut acc = 0f32;
        let mut acc_wrong = 0f32;
        for g in 0..groups_per_row {
            let d = d_scales[row * groups_per_row + g];
            for j in 0..GROUP {
                let idx = row * k + g * GROUP + j;
                let bit = bits[idx];
                let xv = x[g * GROUP + j];
                let sign = if bit == 1 { 1.0f32 } else { -1.0f32 };
                acc += sign * d * xv;
                acc_wrong += d * xv;
            }
        }
        expected[row] = acc;
        expected_wrong[row] = acc_wrong;
    }

    // ── 4. GPU: compile + upload + launch + download ───────────────────────
    gpu.ensure_kernel_public(KERNEL_NAME, KERNEL_SRC, KERNEL_NAME)
        .expect("ensure_kernel_public (JIT compile gemv_bq1g128)");

    let d_a = gpu
        .upload_raw(&packed, &[total_bytes])
        .expect("upload_raw A");
    let d_x = gpu.upload_f32(&x, &[k]).expect("upload_f32 x");
    let d_y = gpu.zeros(&[m], DType::F32).expect("zeros y");

    let mut kb = KernargBlob::new();
    kb.push_ptr(d_a.buf.as_ptr() as *const c_void);
    kb.push_ptr(d_x.buf.as_ptr() as *const c_void);
    kb.push_ptr(d_y.buf.as_ptr() as *const c_void);
    kb.push_i32(m as i32);
    kb.push_i32(k as i32);
    kb.pad_to(16);

    gpu.launch_kernel_blob(
        KERNEL_NAME,
        [m as u32, 1, 1],
        [32, 1, 1],
        0,
        kb.as_mut_slice(),
    )
    .expect("launch_kernel_blob gemv_bq1g128");
    gpu.hip.device_synchronize().expect("device_synchronize");

    let gpu_out = gpu.download_f32(&d_y).expect("download_f32 y");

    // ── 5. Parity assertion: combined relative+absolute tolerance ──────────
    // GPU reduction order (sequential group accumulation + warp-shuffle
    // butterfly over 32 lanes) differs from the CPU's straight left-to-right
    // sum, so bit-exactness is not expected — only agreement up to f32
    // reduction-order error, which scales with the magnitude of the sum.
    let max_abs_expected = expected.iter().fold(0f32, |acc, &v| acc.max(v.abs()));
    let tol = 1e-3 * (1.0 + max_abs_expected);

    let mut max_err = 0f32;
    let mut max_err_row = 0usize;
    for row in 0..m {
        let e = (gpu_out[row] - expected[row]).abs();
        if e > max_err {
            max_err = e;
            max_err_row = row;
        }
    }
    let rel = max_err / (1.0 + max_abs_expected);
    println!(
        "shape M={m} K={k} groups_per_row={groups_per_row}: sample row={max_err_row} cpu={:.6} gpu={:.6}",
        expected[max_err_row], gpu_out[max_err_row]
    );

    if max_err >= tol {
        eprintln!(
            "\nFAIL: PARITY MISMATCH M={m} K={k} max_err={:e} >= tol={:e} (max_abs_expected={:e})",
            max_err, tol, max_abs_expected
        );
        return false;
    }
    println!(
        "PARITY OK M={m} K={k} max_err={:e} rel={:e} tol={:e}",
        max_err, rel, tol
    );

    // ── 6. Negative control: GPU output must NOT match the wrong oracle ────
    let mut wrong_max_err = 0f32;
    for row in 0..m {
        let e = (gpu_out[row] - expected_wrong[row]).abs();
        if e > wrong_max_err {
            wrong_max_err = e;
        }
    }
    if wrong_max_err < tol {
        eprintln!(
            "\nFAIL: NEG-CONTROL DID NOT REJECT M={m} K={k} — wrong oracle also matched \
             within tol (wrong_max_err={:e} < tol={:e}). The parity check above has no \
             discriminating power.",
            wrong_max_err, tol
        );
        return false;
    }
    println!(
        "NEG-CONTROL OK M={m} K={k} (wrong map rejected) wrong_max_err={:e}",
        wrong_max_err
    );

    true
}

fn main() {
    let mut gpu = Gpu::init().expect("Gpu::init failed");
    println!("arch: {}", gpu.arch);

    let mut all_ok = true;
    // Shape 1: groups_per_row=1 (single group, no loop iteration).
    all_ok &= run(&mut gpu, 64, 128, 0xC0FFEE42);
    // Shape 2: groups_per_row=4 (exercises the accumulate-across-groups path).
    all_ok &= run(&mut gpu, 48, 512, 0xDEADBEEF);

    // Shape 3: groups_per_row=3 — ODD, so the 2-group unrolled loop runs once
    // and leaves one group to the tail. Even counts alone never exercise that
    // handoff, which is where an off-by-one in the unroll would hide.
    all_ok &= run(&mut gpu, 32, 384, 0x5EED0003);
    // Shape 4: groups_per_row=5 — two unrolled iterations plus a tail group.
    all_ok &= run(&mut gpu, 40, 640, 0x5EED0005);

    if !all_ok {
        eprintln!("\nFAIL: one or more shapes did not pass parity/neg-control.");
        std::process::exit(1);
    }
    println!("ALL PASS");
}
