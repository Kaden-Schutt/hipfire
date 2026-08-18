// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026
// hipfire — see LICENSE and NOTICE in the project root.

//! Task 8v — GPU vs CPU correctness gate for `dequant_tq2g128_to_f16`
//! (PrismML Q2_0-layout ternary dequant, `kernels/src/dequant_tq2g128_to_f16.hip`).
//!
//! This is the FIRST on-device run of the ternary path: it isolates the
//! 2-bit unpack + `(code-1)*d` reconstruction in complete isolation from
//! the gemv that will be built on top of it (Task 9). If this kernel's
//! byte layout, lane mapping, or launch geometry is wrong, everything
//! downstream (gemv_tq2g128, MoE ternary experts) will silently produce
//! garbage — so we prove it here, cheaply, before writing a single line
//! of gemv code against it.
//!
//! Block layout (34 bytes per 128-weight group), matches the kernel header:
//!   [FP16 d (2B)][32 packed-code bytes, 4 codes/byte, LSB-first]
//!   qs[j/4] |= code_j << ((j%4)*2)
//!   code -> (code - 1) * d, i.e. {0,1,2,3} -> {-d, 0, +d, +2d}
//!
//! Launch geometry (from the kernel's own header comment):
//!   Grid:  [M, groups_per_row, 1]   (groups_per_row = K / 128)
//!   Block: [32, 1, 1]               (each thread dequants 4 elements)
//!
//! Includes a negative control: an intentionally WRONG CPU oracle (using
//! `code` instead of `code - 1`) must NOT match the GPU output. This
//! proves the parity assertion actually has discriminating power and
//! isn't vacuously true (e.g. from a zeroed buffer or a no-op kernel).
//!
//! Run:
//!   cd ~/.hipfire/src
//!   cargo run -p rdna-compute --example test_dequant_tq2g128

use hip_bridge::KernargBlob;
use rdna_compute::{DType, Gpu};
use std::ffi::c_void;

// `rdna_compute::kernels` is a private module (only `GEMV_SRC` is
// re-exported from lib.rs), so — mirroring the established pattern in
// bench_q8_wmma_variants.rs — this example includes the kernel source
// directly rather than adding new public surface to the crate.
const KERNEL_SRC: &str = include_str!("../../../kernels/src/dequant_tq2g128_to_f16.hip");

const GROUP: usize = 128;
const BLOCK_BYTES: usize = 34; // 2 (fp16 d) + 32 (128 codes @ 2 bits)
const KERNEL_NAME: &str = "dequant_tq2g128_to_f16";

// ── fp16 helpers (rdna-compute examples can't depend on hipfire-runtime;
// copied verbatim from the established pattern in
// test_gemv_mfp4g32_p.rs / bench_q8_wmma_variants.rs) ──────────────────────
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
    /// Uniform 2-bit code in {0,1,2,3}.
    fn next_code(&mut self) -> u8 {
        (self.next_u32() & 0x3) as u8
    }
}

fn main() {
    let mut gpu = Gpu::init().expect("Gpu::init failed");
    println!("arch: {}", gpu.arch);

    const M: usize = 64;
    const K: usize = GROUP; // groups_per_row = 1
    let groups_per_row = K / GROUP;
    let row_bytes = groups_per_row * BLOCK_BYTES;
    let total_bytes = M * row_bytes;

    let mut rng = Lcg(0xC0FFEE42);

    // ── 1. Generate synthetic packed Q2_0 blocks ───────────────────────────
    let mut packed = vec![0u8; total_bytes];
    let mut d_scales = vec![0f32; M * groups_per_row];
    let mut codes = vec![0u8; M * K]; // row-major, matches output layout

    for row in 0..M {
        for g in 0..groups_per_row {
            let d = 0.05 + rng.next_f32() * (2.0 - 0.05); // (0.05, 2.0)
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

            for j in 0..GROUP {
                let code = rng.next_code();
                codes[row * K + g * GROUP + j] = code;
                let byte_idx = 2 + j / 4;
                let shift = (j % 4) * 2;
                packed[block_off + byte_idx] |= code << shift;
            }
        }
    }

    // ── 2. CPU oracle: value[j] = (code_j - 1) * d ─────────────────────────
    let mut cpu_ref = vec![0f32; M * K];
    // Negative-control oracle: intentionally WRONG map (code instead of code-1).
    let mut cpu_wrong = vec![0f32; M * K];
    for row in 0..M {
        for g in 0..groups_per_row {
            let d = d_scales[row * groups_per_row + g];
            for j in 0..GROUP {
                let idx = row * K + g * GROUP + j;
                let code = codes[idx];
                cpu_ref[idx] = (code as i32 - 1) as f32 * d;
                cpu_wrong[idx] = code as f32 * d;
            }
        }
    }

    // ── 3. GPU: compile + upload + launch + download ───────────────────────
    gpu.ensure_kernel_public(KERNEL_NAME, KERNEL_SRC, KERNEL_NAME)
        .expect("ensure_kernel_public (JIT compile dequant_tq2g128_to_f16)");

    let d_packed = gpu.upload_raw(&packed, &[total_bytes]).expect("upload_raw");
    let d_out = gpu
        .zeros(&[M * K], DType::F16)
        .expect("zeros (output f16 buffer)");

    let mut kb = KernargBlob::new();
    kb.push_ptr(d_packed.buf.as_ptr() as *const c_void);
    kb.push_ptr(d_out.buf.as_ptr() as *const c_void);
    kb.push_i32(M as i32);
    kb.push_i32(K as i32);
    kb.pad_to(16);

    gpu.launch_kernel_blob(
        KERNEL_NAME,
        [M as u32, groups_per_row as u32, 1],
        [32, 1, 1],
        0,
        kb.as_mut_slice(),
    )
    .expect("launch_kernel_blob dequant_tq2g128_to_f16");
    gpu.hip.device_synchronize().expect("device_synchronize");

    // Download raw f16 bytes and widen to f32.
    let mut out_bytes = vec![0u8; M * K * 2];
    gpu.hip
        .memcpy_dtoh(&mut out_bytes, &d_out.buf)
        .expect("memcpy_dtoh output");
    let gpu_out: Vec<f32> = (0..M * K)
        .map(|i| f16_bits_to_f32(u16::from_le_bytes([out_bytes[2 * i], out_bytes[2 * i + 1]])))
        .collect();

    // ── 4. Parity assertion (fp16 rounding tolerance) ───────────────────────
    let mut max_err = 0f32;
    let mut max_err_idx = 0usize;
    for i in 0..M * K {
        let e = (gpu_out[i] - cpu_ref[i]).abs();
        if e > max_err {
            max_err = e;
            max_err_idx = i;
        }
    }
    println!(
        "sample: row={} j={} code={} d={:.6} cpu={:.6} gpu={:.6}",
        max_err_idx / K,
        max_err_idx % K,
        codes[max_err_idx],
        d_scales[(max_err_idx / K) * groups_per_row + (max_err_idx % K) / GROUP],
        cpu_ref[max_err_idx],
        gpu_out[max_err_idx]
    );

    if max_err >= 1e-3 {
        eprintln!(
            "\nFAIL: PARITY MISMATCH max_err={:e} >= 1e-3 (M={M} K={K})",
            max_err
        );
        std::process::exit(1);
    }
    println!("PARITY OK max_err={:e}", max_err);

    // ── 5. Negative control: GPU output must NOT match the wrong oracle ────
    let mut wrong_max_err = 0f32;
    for i in 0..M * K {
        let e = (gpu_out[i] - cpu_wrong[i]).abs();
        if e > wrong_max_err {
            wrong_max_err = e;
        }
    }
    // The wrong oracle differs from the correct one by exactly `d` per
    // element in expectation (except where code==... — see below), so a
    // real, discriminating kernel must show a LARGE mismatch here. Use the
    // same 1e-3 threshold: if the "wrong" oracle also matched within
    // 1e-3, the parity check above would be vacuous (e.g. GPU buffer left
    // as all-zero, or codes trivially uniform).
    if wrong_max_err < 1e-3 {
        eprintln!(
            "\nFAIL: NEG-CONTROL DID NOT REJECT — wrong oracle also matched \
             within 1e-3 (wrong_max_err={:e}). The parity check above has no \
             discriminating power.",
            wrong_max_err
        );
        std::process::exit(1);
    }
    println!(
        "NEG-CONTROL OK (wrong map rejected) wrong_max_err={:e}",
        wrong_max_err
    );
}
