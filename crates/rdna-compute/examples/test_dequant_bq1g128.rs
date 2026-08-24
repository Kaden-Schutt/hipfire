// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026
// hipfire — see LICENSE and NOTICE in the project root.

//! Task 9 — GPU vs CPU correctness gate for `dequant_bq1g128_to_f16`
//! (PrismML Q1_0-layout binary/1-bit dequant, `kernels/src/dequant_bq1g128_to_f16.hip`).
//!
//! This is the on-device correctness proof for the binary Q1_0 dequant
//! kernel on gfx1151, mirroring `test_dequant_tq2g128.rs` (the ternary Q2_0
//! sibling) but for the 1-bit binary layout, and reusing the Q1_0
//! weight-generation helpers from `test_gemv_bq1g128.rs` (Task 8, already
//! device-verified — this closes the "both bq1g128 kernels verified on
//! hardware" goal for SP-B).
//!
//! Block layout (18 bytes per 128-weight group), matches
//! `reference/hfq1g128/q1_0.rs` and the kernel body:
//!   [FP16 d (2B)][16 packed sign-bit bytes, 8 bits/byte, LSB-first]
//!   element e = 8*j + i (byte j, bit i) -> bit==1 => +d, bit==0 => -d
//!
//! Launch geometry (from the kernel's own header comment):
//!   Grid:  [M, groups_per_row, 1]   (groups_per_row = K / 128)
//!   Block: [32, 1, 1]               (each thread dequants 4 elements)
//!
//! Two shapes are tested to exercise the grid's group dimension (same shape
//! pair as the gemv_bq1g128 sibling):
//!   - M=64, K=128  (groups_per_row=1, single group)
//!   - M=48, K=512  (groups_per_row=4, exercises the group-indexing/stride path)
//!
//! Includes a negative control: an intentionally WRONG CPU oracle (all `+d`,
//! ignoring the sign bit entirely) must NOT match the GPU output. This
//! proves the parity assertion actually has discriminating power and isn't
//! vacuously true (e.g. from a zeroed buffer or a no-op kernel).
//!
//! Run:
//!   cd ~/.hipfire/src
//!   cargo run -p rdna-compute --example test_dequant_bq1g128 --release

use hip_bridge::KernargBlob;
use rdna_compute::{DType, Gpu};
use std::ffi::c_void;

// `rdna_compute::kernels` is a private module (only `GEMV_SRC` is
// re-exported from lib.rs), so — mirroring the established pattern in
// bench_q8_wmma_variants.rs, test_dequant_tq2g128.rs, and
// test_gemv_bq1g128.rs — this example includes the kernel source directly
// rather than adding new public surface to the crate.
const KERNEL_SRC: &str = include_str!("../../../kernels/src/dequant_bq1g128_to_f16.hip");

const GROUP: usize = 128;
const BLOCK_BYTES: usize = 18; // 2 (fp16 d) + 16 (128 sign bits @ 1 bit/element)
const KERNEL_NAME: &str = "dequant_bq1g128_to_f16";

// ── fp16 helpers (rdna-compute examples can't depend on hipfire-runtime;
// copied verbatim from the established pattern in test_dequant_tq2g128.rs /
// test_gemv_bq1g128.rs / test_gemv_mfp4g32_p.rs / bench_q8_wmma_variants.rs) ──
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
    let mut bits = vec![0u8; m * k]; // row-major, matches output layout

    for row in 0..m {
        for g in 0..groups_per_row {
            // Positive scale in (0.02, 1.5) — d is always > 0 for Q1_0
            // (sum-of-abs-values-based scale; never negative).
            let d = 0.02 + rng.next_f32() * (1.5 - 0.02);
            let d_h = f32_to_f16_bits(d);
            // Round-trip through fp16 so the CPU oracle uses the SAME
            // quantized scale the GPU kernel actually reads (the kernel
            // loads a _Float16 and widens to float — any host-side f32
            // scale would silently mismatch by fp16 rounding).
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

    // ── 2. CPU oracle: value[e] = bit_e ? +d : -d ───────────────────────────
    // Negative-control oracle: intentionally WRONG map (drops the sign bit
    // entirely — every element treated as +d).
    let mut cpu_ref = vec![0f32; m * k];
    let mut cpu_wrong = vec![0f32; m * k];
    for row in 0..m {
        for g in 0..groups_per_row {
            let d = d_scales[row * groups_per_row + g];
            for j in 0..GROUP {
                let idx = row * k + g * GROUP + j;
                let bit = bits[idx];
                cpu_ref[idx] = if bit == 1 { d } else { -d };
                cpu_wrong[idx] = d;
            }
        }
    }

    // ── 3. GPU: compile + upload + launch + download ───────────────────────
    gpu.ensure_kernel_public(KERNEL_NAME, KERNEL_SRC, KERNEL_NAME)
        .expect("ensure_kernel_public (JIT compile dequant_bq1g128_to_f16)");

    let d_packed = gpu.upload_raw(&packed, &[total_bytes]).expect("upload_raw");
    let d_out = gpu
        .zeros(&[m * k], DType::F16)
        .expect("zeros (output f16 buffer)");

    let mut kb = KernargBlob::new();
    kb.push_ptr(d_packed.buf.as_ptr() as *const c_void);
    kb.push_ptr(d_out.buf.as_ptr() as *const c_void);
    kb.push_i32(m as i32);
    kb.push_i32(k as i32);
    kb.pad_to(16);

    gpu.launch_kernel_blob(
        KERNEL_NAME,
        [m as u32, groups_per_row as u32, 1],
        [32, 1, 1],
        0,
        kb.as_mut_slice(),
    )
    .expect("launch_kernel_blob dequant_bq1g128_to_f16");
    gpu.hip.device_synchronize().expect("device_synchronize");

    // Download raw f16 bytes and widen to f32.
    let mut out_bytes = vec![0u8; m * k * 2];
    gpu.hip
        .memcpy_dtoh(&mut out_bytes, &d_out.buf)
        .expect("memcpy_dtoh output");
    let gpu_out: Vec<f32> = (0..m * k)
        .map(|i| f16_bits_to_f32(u16::from_le_bytes([out_bytes[2 * i], out_bytes[2 * i + 1]])))
        .collect();

    // ── 4. Parity assertion (fp16 rounding tolerance) ───────────────────────
    // Dequant is exact up to fp16 representation of ±d — no reduction
    // involved — so error should be ~0.
    let mut max_err = 0f32;
    let mut max_err_idx = 0usize;
    for i in 0..m * k {
        let e = (gpu_out[i] - cpu_ref[i]).abs();
        if e > max_err {
            max_err = e;
            max_err_idx = i;
        }
    }
    println!(
        "shape M={m} K={k} groups_per_row={groups_per_row}: sample row={} j={} bit={} d={:.6} cpu={:.6} gpu={:.6}",
        max_err_idx / k,
        max_err_idx % k,
        bits[max_err_idx],
        d_scales[(max_err_idx / k) * groups_per_row + (max_err_idx % k) / GROUP],
        cpu_ref[max_err_idx],
        gpu_out[max_err_idx]
    );

    if max_err >= 1e-3 {
        eprintln!(
            "\nFAIL: PARITY MISMATCH M={m} K={k} max_err={:e} >= 1e-3",
            max_err
        );
        return false;
    }
    println!("PARITY OK M={m} K={k} max_err={:e}", max_err);

    // ── 5. Negative control: GPU output must NOT match the wrong oracle ────
    let mut wrong_max_err = 0f32;
    for i in 0..m * k {
        let e = (gpu_out[i] - cpu_wrong[i]).abs();
        if e > wrong_max_err {
            wrong_max_err = e;
        }
    }
    // The wrong oracle differs from the correct one by exactly `2d` on every
    // element where bit==0 (the sign flip is dropped), so a real,
    // discriminating kernel must show a LARGE mismatch here (statistically
    // ~half the elements, each off by 2d). Use the same 1e-3 threshold: if
    // the "wrong" oracle also matched within 1e-3, the parity check above
    // would be vacuous (e.g. GPU buffer left as all-zero/all-+d, or bits
    // trivially uniform).
    if wrong_max_err < 1e-3 {
        eprintln!(
            "\nFAIL: NEG-CONTROL DID NOT REJECT M={m} K={k} — wrong oracle also matched \
             within 1e-3 (wrong_max_err={:e}). The parity check above has no \
             discriminating power.",
            wrong_max_err
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
    // Shape 1: groups_per_row=1 (single group, exercises grid.y==0 only).
    all_ok &= run(&mut gpu, 64, 128, 0xC0FFEE42);
    // Shape 2: groups_per_row=4 (exercises the grid.y group-indexing/stride path).
    all_ok &= run(&mut gpu, 48, 512, 0xDEADBEEF);

    if !all_ok {
        eprintln!("\nFAIL: one or more shapes did not pass parity/neg-control.");
        std::process::exit(1);
    }
    println!("ALL PASS");
}
