// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
//! MQ2-G256-Lloyd **residual** GEMV (`y[row] += A[row] · x`), both variants,
//! on the dense o_proj / out_proj decode shape M=2048, K=4096.
//!
//! Covers:
//!   * `kernels/src/gemv_mq2g256_lloyd_residual.hip`
//!     — single-accumulator baseline, symbol `gemv_mq2g256_lloyd_residual`
//!   * `kernels/src/gemv_mq2g256_lloyd_residual.gfx1100.hip`
//!     — RDNA3 K4-unrolled / LDS-codebook variant.
//!
//! !! SYMBOL COLLISION !! Both source files export the SAME extern "C" symbol
//! `gemv_mq2g256_lloyd_residual` (the task brief guessed the gfx1100 file used
//! a `_rdna3` suffix — it does not; verified by reading both sources). The JIT
//! function cache in `scratch::compile_and_load_kernel` is keyed by SYMBOL NAME
//! ONLY (`if functions.contains_key(func_name) { return Ok(()) }`), so loading
//! both under their on-disk names would silently make the second
//! `ensure_kernel_public` a NO-OP and run the FIRST kernel twice — a benchmark
//! that reports a perfectly plausible "+0.0%" while measuring nothing. This
//! bench therefore rewrites the gfx1100 arm's declaration to
//! `gemv_mq2g256_lloyd_residual_rdna3` in-memory (no kernel file is touched)
//! and JITs it under a distinct module name. The rewrite is asserted, and if it
//! failed the subsequent `hipModuleGetFunction` for the `_rdna3` symbol would
//! error out rather than silently alias.
//!
//! ## REGIME CAVEAT — READ BEFORE QUOTING ANY NUMBER FROM THIS FILE
//!
//! This is a **HIP-dispatch microbenchmark**. It is a TRIAGE FILTER for gross
//! defects — wrong numbers, pathological slowness — and NOT a verdict on a
//! kernel. Host launch latency masks device-level effects, and the same kernel
//! can measure differently once lowered to retained PM4 replay. A prior HIP
//! microbench measured the MQ2GL kernel at -3.1% and that number was nearly
//! used to kill the format. Final acceptance is the golden bundle
//! (`registry/redline-golden-v1.json`) with both HIP and PM4 arms. Treat the
//! timing section here as "is it in the right order of magnitude", nothing more.
//!
//! ## What is verified
//!
//! 1. **Numerics vs an independent CPU reference.** The reference is derived
//!    from the FORMAT SPEC (72 B group = 4×fp16 codebook then 64 B of 2-bit
//!    indices, byte i holding weights 4i..4i+3 at bit offsets 0/2/4/6), decoded
//!    in f64 in plain column order — NOT from the kernel's lane/tid mapping,
//!    its 8-way unroll, or its wave reduction. A shared misunderstanding of the
//!    thread mapping therefore cannot cancel out.
//! 2. **Residual semantics: ADD, not overwrite.** `y` is pre-seeded with a
//!    distinctive nonzero pattern (mixed signs, plus exact zeros and one large
//!    value). Pass requires `y == seed + dot`; the overwrite failure mode
//!    (`y == dot`) is measured and reported separately so a FAIL says *which*
//!    way it broke. A second back-to-back launch must yield `seed + 2·dot`.
//! 3. **Both variants agree with the reference and with each other.** Note the
//!    gfx1100 variant's own header states its K4 reassociation makes it NOT
//!    bit-identical to the baseline, so cross-variant equality is checked at
//!    the same 1e-4 relative tolerance; the bit-identical count is printed as
//!    information only. An "all 2048 rows bit-identical" result is flagged as
//!    SUSPICIOUS (it is the signature of the symbol collision above).
//! 4. **Tail-path coverage.** K=4096 gives groups_per_row=16, i.e. tail==0, so
//!    the gfx1100 variant's `TAIL_LOAD_AND_DOT` blocks never execute at the
//!    production shape. Extra small-M shapes with groups_per_row ∈ {9,10,11}
//!    exercise tail ∈ {1,2,3}. These K values are NOT a3b shapes — they exist
//!    purely to reach that code.
//!
//! ## Conventions
//!
//! * `x` is nominally FWHT-256 pre-rotated by the caller for MQ formats. This
//!   microbench feeds an arbitrary deterministic `x` and uses the SAME bytes on
//!   the GPU and in the CPU reference, so the rotation is irrelevant to parity.
//! * Synthetic codebook entries are round-tripped through the fp16 encoder and
//!   the reference reads back the ENCODED bits, so encoder rounding cannot
//!   contribute error to the parity number.
//!
//! Run: cargo run --release -p rdna-compute --example bench_mq2g256_lloyd_residual

use hip_bridge::KernargBlob;
use rdna_compute::{Gpu, GpuTensor};
use std::time::Instant;

const BASE_SRC: &str = include_str!("../../../kernels/src/gemv_mq2g256_lloyd_residual.hip");
const GFX1100_SRC: &str =
    include_str!("../../../kernels/src/gemv_mq2g256_lloyd_residual.gfx1100.hip");

const BASE_FN: &str = "gemv_mq2g256_lloyd_residual";
const RDNA3_FN: &str = "gemv_mq2g256_lloyd_residual_rdna3";
const BASE_MOD: &str = "bench_mq2res_base";
const RDNA3_MOD: &str = "bench_mq2res_gfx1100";

/// MQ2-Lloyd on-disk group stride, from the format spec: 8 B codebook + 64 B
/// of 2-bit indices = 72 B per 256 weights = 2.25 bpw.
const GROUP_BYTES: usize = 72;
const GROUP_WEIGHTS: usize = 256;

/// Textbook Lloyd–Max levels for a unit Gaussian, 2 bit. Only a plausible
/// starting shape — the builder perturbs these per group so the four entries
/// are four *distinct, asymmetric* values (a kernel that reconstructs the
/// codebook from a formula instead of reading all four fp16 slots fails).
const LLOYD_UNIT: [f32; 4] = [-1.5104, -0.4528, 0.4528, 1.5104];

/// Relative tolerance for f32 accumulation at these K.
const TOL: f64 = 1e-4;

// ───────────────────────── deterministic PRNG ─────────────────────────

fn mix(x: u64) -> u64 {
    let h = x
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    ((h ^ (h >> 33)).wrapping_mul(0xff51afd7ed558ccd)) ^ (h >> 28)
}

fn unit(x: u64) -> f32 {
    ((mix(x) >> 11) as f64 / (1u64 << 53) as f64) as f32
}

// ───────────────────────── IEEE-754 binary16 ─────────────────────────
// Full round-to-nearest-even encoder + exact decoder (subnormals included), so
// the reference sees byte-for-byte what `__half2float` sees on device.

fn f32_to_f16(f: f32) -> u16 {
    let x = f.to_bits();
    let sign = ((x >> 16) & 0x8000) as u16;
    let exp = ((x >> 23) & 0xff) as i32;
    let mant = x & 0x007f_ffff;
    if exp == 0xff {
        // inf / nan
        return sign | 0x7c00 | if mant != 0 { 0x200 } else { 0 };
    }
    let e = exp - 127 + 15;
    if e >= 0x1f {
        return sign | 0x7c00; // overflow -> inf
    }
    if e <= 0 {
        // subnormal (or underflow to zero)
        let shift = (126 - exp) as u32; // >= 14
        if shift > 24 {
            return sign;
        }
        let m = mant | 0x0080_0000;
        let h = (m >> shift) as u16;
        let rem = m & ((1u32 << shift) - 1);
        let halfway = 1u32 << (shift - 1);
        let mut out = sign | h;
        if rem > halfway || (rem == halfway && (h & 1) == 1) {
            out += 1;
        }
        return out;
    }
    let h = (mant >> 13) as u16;
    let rem = mant & 0x1fff;
    let mut out = sign | ((e as u16) << 10) | h;
    // carry out of the mantissa propagates into the exponent field, which is
    // exactly the IEEE behaviour (and saturates to inf at e==31).
    if rem > 0x1000 || (rem == 0x1000 && (h & 1) == 1) {
        out += 1;
    }
    out
}

fn f16_to_f32(h: u16) -> f32 {
    let sign = ((h >> 15) & 1) as u32;
    let exp = ((h >> 10) & 0x1f) as u32;
    let man = (h & 0x3ff) as u32;
    let bits = if exp == 0 {
        if man == 0 {
            sign << 31
        } else {
            // subnormal: value = man * 2^-24; renormalise into f32
            let mut m = man;
            let mut e: i32 = -14;
            while m & 0x400 == 0 {
                m <<= 1;
                e -= 1;
            }
            m &= 0x3ff;
            (sign << 31) | (((e + 127) as u32) << 23) | (m << 13)
        }
    } else if exp == 0x1f {
        (sign << 31) | (0xffu32 << 23) | (man << 13)
    } else {
        (sign << 31) | ((exp + 112) << 23) | (man << 13)
    };
    f32::from_bits(bits)
}

// ───────────────────── synthetic weights (spec layout) ─────────────────────

/// Per-group codebook, spec-shaped: four ascending fp16 entries. Returned as
/// the ENCODED bit patterns so the builder and the reference cannot disagree.
fn group_codebook_bits(seed: u64, row: usize, g: usize) -> [u16; 4] {
    let key = seed ^ ((row as u64) << 24) ^ ((g as u64) << 3);
    // per-group scale in a realistic MQ2 range, plus an asymmetric offset that
    // preserves ascending order but breaks the ±symmetry of the Lloyd levels.
    let s = 0.0030f32 + unit(key) * 0.0025;
    let off = (unit(key ^ 0xa5) - 0.5) * 0.7;
    let mut out = [0u16; 4];
    for (e, slot) in out.iter_mut().enumerate() {
        *slot = f32_to_f16((LLOYD_UNIT[e] + off) * s);
    }
    out
}

/// Build a row-major MQ2-Lloyd weight blob in the exact on-disk byte layout:
/// per row, `groups_per_row` × [8 B fp16 codebook | 64 B packed 2-bit indices].
fn build_mq2_lloyd(m: usize, k: usize, seed: u64) -> Vec<u8> {
    let gpr = k / GROUP_WEIGHTS;
    let mut out = vec![0u8; m * gpr * GROUP_BYTES];
    for row in 0..m {
        for g in 0..gpr {
            let off = (row * gpr + g) * GROUP_BYTES;
            for (e, bits) in group_codebook_bits(seed, row, g).iter().enumerate() {
                out[off + 2 * e] = (*bits & 0xff) as u8;
                out[off + 2 * e + 1] = (*bits >> 8) as u8;
            }
            for b in 0..64 {
                out[off + 8 + b] =
                    (mix(seed ^ ((row as u64) << 40) ^ ((g as u64) << 12) ^ b as u64) & 0xff) as u8;
            }
        }
    }
    out
}

// ───────────────────────── independent CPU reference ─────────────────────────

/// Decode the blob straight from the FORMAT SPEC and accumulate in f64, in
/// plain ascending column order. Deliberately does NOT mirror the kernel's
/// thread mapping, unrolling, or reduction tree.
///
/// Spec used, verbatim:
///   * row r starts at r * (K/256) * 72 bytes
///   * group g starts at row_base + g*72
///   * bytes [0..8) are 4 fp16 codebook entries, little-endian
///   * bytes [8..72) are 64 index bytes; byte i carries the four weights
///     4i, 4i+1, 4i+2, 4i+3 at bit offsets 0, 2, 4, 6
///   * that weight's column within the row is g*256 + 4i + j
fn cpu_reference(a: &[u8], x: &[f32], m: usize, k: usize) -> Vec<f64> {
    let gpr = k / GROUP_WEIGHTS;
    let mut out = vec![0f64; m];
    for (row, slot) in out.iter_mut().enumerate() {
        let row_base = row * gpr * GROUP_BYTES;
        let mut acc = 0f64;
        for g in 0..gpr {
            let gb = row_base + g * GROUP_BYTES;
            let mut cb = [0f64; 4];
            for (e, c) in cb.iter_mut().enumerate() {
                *c = f16_to_f32(u16::from_le_bytes([a[gb + 2 * e], a[gb + 2 * e + 1]])) as f64;
            }
            for i in 0..64 {
                let byte = a[gb + 8 + i];
                for j in 0..4 {
                    let q = ((byte >> (2 * j)) & 3) as usize;
                    let col = g * GROUP_WEIGHTS + 4 * i + j;
                    acc += cb[q] * x[col] as f64;
                }
            }
        }
        *slot = acc;
    }
    out
}

// ───────────────────────────── error reporting ─────────────────────────────

struct ErrStat {
    max_abs: f64,
    max_rel: f64,
    worst: usize,
    got: f64,
    want: f64,
}

/// Relative error uses `max(|want|, floor)` as the denominator, where `floor`
/// is the RMS of the reference vector. Without the floor a row whose dot
/// product happens to land near zero reports a meaningless 1e+3 relative error
/// while being absolutely correct to 1e-8.
fn compare(got: &[f32], want: &[f64], floor: f64) -> ErrStat {
    let mut st = ErrStat {
        max_abs: 0.0,
        max_rel: 0.0,
        worst: 0,
        got: 0.0,
        want: 0.0,
    };
    for (i, (&g, &w)) in got.iter().zip(want.iter()).enumerate() {
        let d = (g as f64 - w).abs();
        let rel = d / w.abs().max(floor);
        if d > st.max_abs {
            st.max_abs = d;
        }
        if rel > st.max_rel {
            st.max_rel = rel;
            st.worst = i;
            st.got = g as f64;
            st.want = w;
        }
    }
    st
}

fn rms(v: &[f64]) -> f64 {
    (v.iter().map(|a| a * a).sum::<f64>() / v.len().max(1) as f64).sqrt()
}

// ───────────────────────────── gpu helpers ─────────────────────────────

fn reseed(gpu: &Gpu, t: &GpuTensor, seed: &[f32]) {
    let bytes =
        unsafe { std::slice::from_raw_parts(seed.as_ptr() as *const u8, std::mem::size_of_val(seed)) };
    gpu.hip.memcpy_htod(&t.buf, bytes).expect("reseed y");
}

fn blob_for(a: &GpuTensor, x: &GpuTensor, y: &GpuTensor, m: usize, k: usize) -> Vec<u8> {
    let mut b = KernargBlob::new();
    b.push_ptr(a.buf.as_ptr() as *const _);
    b.push_ptr(x.buf.as_ptr() as *const _);
    b.push_ptr(y.buf.as_ptr() as *const _);
    b.push_i32(m as i32);
    b.push_i32(k as i32);
    b.into_vec()
}

/// In-memory rename of the gfx1100 arm's exported symbol. Touches no file on
/// disk; asserted so a source edit that changes the declaration shape breaks
/// loudly here instead of silently aliasing onto the baseline kernel.
fn rdna3_source() -> String {
    let decl = "void gemv_mq2g256_lloyd_residual(";
    let n = GFX1100_SRC.matches(decl).count();
    assert_eq!(
        n, 1,
        "expected exactly one '{decl}' declaration in the gfx1100 source, found {n}; \
         the symbol-collision workaround in this bench needs updating"
    );
    let out = GFX1100_SRC.replace(decl, "void gemv_mq2g256_lloyd_residual_rdna3(");
    assert!(out.contains(RDNA3_FN), "symbol rewrite produced no {RDNA3_FN}");
    out
}

// ─────────────────────────────── main ───────────────────────────────

fn main() {
    let mut gpu = Gpu::init().expect("GPU init");
    println!("arch = {}", gpu.arch);
    println!(
        "\n=== MQ2-G256-Lloyd residual GEMV  (y += A·x)  —  HIP-dispatch microbench ===\n\
         REGIME: triage filter only. Host launch latency dominates at these shapes; a\n\
         kernel that looks flat here can separate under retained PM4 replay. Final\n\
         acceptance is the golden bundle (registry/redline-golden-v1.json), HIP + PM4."
    );

    // JIT both arms. Base keeps its true on-disk symbol; gfx1100 is renamed
    // in-memory because both files export the same name (see file header).
    gpu.ensure_kernel_public(BASE_MOD, BASE_SRC, BASE_FN)
        .expect("JIT baseline residual kernel");
    let rdna3_src = rdna3_source();
    gpu.ensure_kernel_public(RDNA3_MOD, &rdna3_src, RDNA3_FN)
        .expect("JIT gfx1100 residual kernel (renamed symbol)");
    println!("\nJIT OK: '{BASE_FN}' (module {BASE_MOD}), '{RDNA3_FN}' (module {RDNA3_MOD})");
    if !gpu.arch.starts_with("gfx1100") {
        println!(
            "NOTE: arch is {} — the .gfx1100 variant is being run OFF-TARGET. Its math is\n\
             arch-neutral (wave32 shuffles + LDS), so the parity check is still meaningful;\n\
             its TIMING on this arch is not a statement about RDNA3.",
            gpu.arch
        );
    }

    let mut all_ok = true;

    // ── main shape: dense o_proj / out_proj residual, M=2048 K=4096 ──
    let m = 2048usize;
    let k = 4096usize;
    all_ok &= run_shape(&mut gpu, m, k, 0xd0_5eed, true);

    // ── tail-path coverage for the gfx1100 K4 variant ──
    // K=4096 → groups_per_row=16 → tail==0, so TAIL_LOAD_AND_DOT is dead code
    // at the production shape. These K are NOT a3b shapes; they exist only to
    // reach tail ∈ {1,2,3}.
    println!(
        "\n--- tail-path coverage (gfx1100 K4 variant): non-a3b K, small M, parity only ---"
    );
    for (kk, tail) in [(2304usize, 1), (2560, 2), (2816, 3)] {
        println!(
            "\n[tail={tail}] M=256 K={kk}  (groups_per_row={}, quads={}, tail={})",
            kk / 256,
            (kk / 256) >> 2,
            (kk / 256) & 3
        );
        all_ok &= run_shape(&mut gpu, 256, kk, tail_seed(kk), false);
    }

    println!("\n================================================================");
    if all_ok {
        println!("RESULT: PASS  — both variants match the spec-derived CPU reference and");
        println!("               each other within {TOL:.0e} relative, and ADD into y.");
    } else {
        println!("RESULT: FAIL  — see the FAIL lines above.");
        std::process::exit(1);
    }
}

fn tail_seed(k: usize) -> u64 {
    0x7a11_0000 ^ k as u64
}

/// Full parity + (optionally) timing for one shape. Returns true on PASS.
fn run_shape(gpu: &mut Gpu, m: usize, k: usize, seed: u64, do_timing: bool) -> bool {
    let gpr = k / GROUP_WEIGHTS;
    let a_bytes = m * gpr * GROUP_BYTES;

    // Weights in the exact on-disk layout.
    let a_host = build_mq2_lloyd(m, k, seed);
    assert_eq!(a_host.len(), a_bytes);

    // Arbitrary deterministic x. In production x is FWHT-256 pre-rotated; here
    // the same bytes go to the GPU and the CPU reference, so it does not matter.
    let x_host: Vec<f32> = (0..k)
        .map(|i| (unit(seed ^ 0x1234 ^ i as u64) - 0.5) * 1.2)
        .collect();

    // Distinctive y seed: mixed signs, exact zeros, one large value. An
    // overwrite bug cannot reproduce this pattern by accident.
    let y_seed: Vec<f32> = (0..m)
        .map(|i| match i % 7 {
            0 => 0.0,
            1 => 1.0e3,
            2 => -0.75,
            3 => 0.125,
            _ => (unit(seed ^ 0xbeef ^ i as u64) - 0.5) * 2.0,
        })
        .collect();

    // Reference dot products, f64, spec-derived.
    let dot = cpu_reference(&a_host, &x_host, m, k);
    let floor = rms(&dot).max(1e-12);
    let want1: Vec<f64> = dot
        .iter()
        .zip(y_seed.iter())
        .map(|(d, s)| d + *s as f64)
        .collect();
    let want2: Vec<f64> = dot
        .iter()
        .zip(y_seed.iter())
        .map(|(d, s)| 2.0 * d + *s as f64)
        .collect();
    let floor_total = rms(&want1).max(floor);

    let a_t = gpu.upload_raw(&a_host, &[a_bytes]).expect("upload A");
    let x_t = gpu.upload_f32(&x_host, &[k]).expect("upload x");
    let y_base = gpu.upload_f32(&y_seed, &[m]).expect("alloc y base");
    let y_rdna3 = gpu.upload_f32(&y_seed, &[m]).expect("alloc y rdna3");

    let mut blob_base = blob_for(&a_t, &x_t, &y_base, m, k);
    let mut blob_rdna3 = blob_for(&a_t, &x_t, &y_rdna3, m, k);

    let grid = [m as u32, 1, 1];
    let block = [32u32, 1, 1];

    println!(
        "\nshape M={m} K={k}  groups_per_row={gpr}  A={} KiB ({:.4} B/elem, {:.4} bpw)",
        a_bytes / 1024,
        GROUP_BYTES as f64 / GROUP_WEIGHTS as f64,
        GROUP_BYTES as f64 * 8.0 / GROUP_WEIGHTS as f64
    );

    let mut ok = true;

    // ---- pass 1: single launch, must equal seed + dot ----
    for (name, fnname, blob, y) in [
        ("baseline", BASE_FN, &mut blob_base, &y_base),
        ("gfx1100 ", RDNA3_FN, &mut blob_rdna3, &y_rdna3),
    ] {
        reseed(gpu, y, &y_seed);
        gpu.launch_kernel_blob(fnname, grid, block, 0, blob)
            .expect("launch");
        gpu.hip.device_synchronize().expect("sync");
        let got = gpu.download_f32(y).expect("download");

        let st = compare(&got, &want1, floor_total);
        // How far is the output from the OVERWRITE result (y == dot)? If this
        // is ~0 the kernel dropped the residual instead of adding to it.
        let ow = compare(&got, &dot, floor);
        println!(
            "  {name} y+=A·x : max_abs={:.3e}  max_rel={:.3e}  worst[{}] got={:.9} want={:.9}",
            st.max_abs, st.max_rel, st.worst, st.got, st.want
        );
        if st.max_rel <= TOL {
            println!("    PASS  (residual ADD confirmed; distance from overwrite-semantics output: max_rel={:.3e})", ow.max_rel);
        } else {
            ok = false;
            if ow.max_rel <= TOL {
                println!("    FAIL  <<< OVERWRITE, not add: output equals A·x with the y seed DISCARDED.");
            } else {
                println!("    FAIL  <<< numerics wrong (and it is not a plain overwrite either).");
            }
        }
    }

    // ---- pass 2: relaunch on the already-accumulated buffer -> seed + 2·dot ----
    for (name, fnname, blob, y) in [
        ("baseline", BASE_FN, &mut blob_base, &y_base),
        ("gfx1100 ", RDNA3_FN, &mut blob_rdna3, &y_rdna3),
    ] {
        gpu.launch_kernel_blob(fnname, grid, block, 0, blob)
            .expect("launch 2");
        gpu.hip.device_synchronize().expect("sync");
        let got = gpu.download_f32(y).expect("download 2");
        let st = compare(&got, &want2, floor_total);
        let verdict = if st.max_rel <= TOL { "PASS" } else { "FAIL" };
        if st.max_rel > TOL {
            ok = false;
        }
        println!(
            "  {name} 2nd launch accumulates : max_rel={:.3e} worst[{}] got={:.9} want={:.9}  {verdict}",
            st.max_rel, st.worst, st.got, st.want
        );
    }

    // ---- cross-variant agreement (on a fresh single launch each) ----
    reseed(gpu, &y_base, &y_seed);
    reseed(gpu, &y_rdna3, &y_seed);
    gpu.launch_kernel_blob(BASE_FN, grid, block, 0, &mut blob_base)
        .expect("xv base");
    gpu.launch_kernel_blob(RDNA3_FN, grid, block, 0, &mut blob_rdna3)
        .expect("xv rdna3");
    gpu.hip.device_synchronize().expect("sync");
    let gb = gpu.download_f32(&y_base).expect("dl base");
    let gr = gpu.download_f32(&y_rdna3).expect("dl rdna3");
    let mut max_abs = 0f64;
    let mut max_rel = 0f64;
    let mut worst = 0usize;
    let mut identical = 0usize;
    for i in 0..m {
        if gb[i].to_bits() == gr[i].to_bits() {
            identical += 1;
        }
        let d = (gb[i] as f64 - gr[i] as f64).abs();
        let r = d / (gb[i] as f64).abs().max(floor_total);
        if d > max_abs {
            max_abs = d;
        }
        if r > max_rel {
            max_rel = r;
            worst = i;
        }
    }
    let verdict = if max_rel <= TOL { "PASS" } else { "FAIL" };
    if max_rel > TOL {
        ok = false;
    }
    println!(
        "  cross-variant baseline vs gfx1100 : max_abs={max_abs:.3e} max_rel={max_rel:.3e} \
         worst[{worst}] base={:.9} rdna3={:.9}  {verdict}",
        gb[worst], gr[worst]
    );
    println!(
        "    bit-identical rows: {identical}/{m}  (exact identity is NOT expected — the\n\
         \x20   gfx1100 K4 reassociation reorders the FMA chain, per its own header)"
    );
    if identical == m {
        println!(
            "    SUSPICIOUS: every row bit-identical. If the in-memory symbol rename failed,\n\
             \x20   both arms would be the SAME loaded function. Check the JIT lines above."
        );
    }

    if !do_timing {
        return ok;
    }

    // ---- timing ----
    // NOTE: the very first launch of each kernel already happened above, so the
    // JIT/module-load cost is paid. Even so, treat any FIRST-PASS number as
    // JIT- and cache-contaminated; the reported MIN/MEDIAN come from batches
    // taken after an untimed warmup batch.
    const BATCH: usize = 100;
    const ITERS: usize = 9;

    println!("\n  timing: warmup batch (untimed, JIT-contaminated) + {ITERS} timed batches of {BATCH} launches");
    println!(
        "  {:<10} {:>11} {:>11} {:>12} {:>12}",
        "variant", "min us/call", "med us/call", "eff GB/s", "bytes/call"
    );

    // Bytes moved per call: the weight blob (streamed once), x (read; in
    // practice L2-resident at this size), and y (read + write, 4 B each).
    let bytes_per_call = a_bytes as f64 + (k * 4) as f64 + (m * 8) as f64;

    let mut results = Vec::new();
    for (name, fnname, blob, y) in [
        ("baseline", BASE_FN, &mut blob_base, &y_base),
        ("gfx1100", RDNA3_FN, &mut blob_rdna3, &y_rdna3),
    ] {
        reseed(gpu, y, &y_seed);
        // untimed warmup
        for _ in 0..BATCH {
            gpu.launch_kernel_blob(fnname, grid, block, 0, blob)
                .expect("warmup");
        }
        gpu.hip.device_synchronize().expect("sync");

        let mut us: Vec<f64> = Vec::with_capacity(ITERS);
        for _ in 0..ITERS {
            let t0 = Instant::now();
            for _ in 0..BATCH {
                gpu.launch_kernel_blob(fnname, grid, block, 0, blob)
                    .expect("timed");
            }
            gpu.hip.device_synchronize().expect("sync");
            us.push(t0.elapsed().as_secs_f64() * 1e6 / BATCH as f64);
        }
        us.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let min = us[0];
        let med = us[ITERS / 2];
        println!(
            "  {name:<10} {min:>11.3} {med:>11.3} {:>12.1} {:>12.0}",
            bytes_per_call / (med * 1e-6) / 1e9,
            bytes_per_call
        );
        results.push((name, min, med));
    }

    let (_, _, med_base) = results[0];
    let (_, _, med_rdna3) = results[1];
    println!(
        "  gfx1100 variant is {:+.2}% on median us/call vs baseline  \
         (dispatch-bound microbench — see REGIME CAVEAT in the file header)",
        100.0 * (med_rdna3 / med_base - 1.0)
    );
    println!(
        "  tok-shaped: this is ONE decode-token o_proj residual GEMV, so median\n\
         \x20 {med_base:.3} us/call (baseline) / {med_rdna3:.3} us/call (gfx1100) is the per-token\n\
         \x20 cost of this single op — not a model-level tok/s number."
    );

    ok
}
