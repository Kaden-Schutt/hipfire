// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! `gemv_mfp4g32_e8_residual` — correctness + throughput triage on the dense
//! o_proj / out_proj decode shape (M=2048, K=4096), with
//! `gemv_mq3g256_lloyd_residual` on the identical shape as a scale reference.
//!
//! Both kernels are JIT'd from source and launched through the kernarg-blob
//! path (`ensure_kernel_public` + `launch_kernel_blob`), so this bench touches
//! no dispatch plumbing, no kernel source, and no shared file.
//!
//! ## REGIME CAVEAT — read before quoting any number from this file
//!
//! This is a **HIP-dispatch microbenchmark**. It is a **TRIAGE FILTER for gross
//! defects** (wrong numbers, pathological slowness), **NOT a verdict on a
//! kernel**. Host launch latency masks device-level effects, and the same
//! kernel can measure differently once lowered to retained PM4 replay. A prior
//! HIP microbench measured the MQ2GL kernel at −3.1% and that was nearly used
//! to kill the format. Final acceptance is the golden bundle
//! (`registry/redline-golden-v1.json`) with HIP **and** PM4 arms. A first-pass
//! timing number is additionally JIT-contaminated and is discarded here.
//!
//! MFP4-E8 and MQ3-Lloyd encode different weights by construction, so the two
//! kernels are NOT expected to agree with each other. Each is checked only
//! against its own CPU reference; the cross-kernel comparison is bytes and
//! time.
//!
//! ## What is checked
//!
//! `gemv_mfp4g32_e8_residual` is a RESIDUAL kernel: `y[row] += A[row]·x`. The
//! single most likely silent bug is accumulate-vs-overwrite, so:
//!   1. `y` is pre-seeded with a distinctive nonzero pattern and the reference
//!      is `seed + dot` — an overwriting kernel fails by the whole seed.
//!   2. The kernel is launched a SECOND time without re-seeding and must land
//!      on `seed + 2·dot` — an overwriting kernel returns the same value twice
//!      and is caught even if the seed happened to be small.
//!   3. Row 0 is built with E4M3 block-scale byte 0x00 (decoded scale = 0), so
//!      its dot product is EXACTLY zero and `y[0]` must come back
//!      **bit-identical** to its seed.
//!   4. A second parity shape with K=1792 (7 groups → quads=1, tail=3) runs the
//!      three `tail >= n` branches, which K=4096 (16 groups, tail=0) never
//!      touches.
//!   5. An over-provisioned grid (M+5 blocks) writing into a padded `y` must
//!      leave the pad bit-identical — the `if (row >= M) return;` guard. The
//!      pad lives inside our own allocation, so a missing guard fails loudly
//!      instead of scribbling on a neighbouring pooled buffer.
//!
//! ## CPU reference independence
//!
//! The reference is NOT a transliteration of the kernel's decode. The host
//! *chooses* an E8 lattice point (biased coords e[0..8], even sum), keeps the
//! coordinates it chose, and derives the on-disk bytes from the FORMAT SPEC's
//! packing rule (`crates/hipfire-quantize/src/e8.rs` header):
//!
//!   bits[4i .. 4i+4) = e[i], i = 0..6 | bits[28..31) = e[7]>>1 | bit[31] = coset
//!
//! with e[7]'s LSB dropped and recoverable only because the D8 constraint
//! forces sum(e) even. The reference dot product uses the chosen coordinates,
//! never a decode — so a shared misunderstanding of the *decode* cannot cancel
//! out. The E4M3 block scale is re-derived from the OCP spec (unsigned, bias 7,
//! 3 mantissa bits, exp=0 subnormal, the exp=15/mant=7 NaN slot repurposed as
//! max-finite 448) rather than copied from the kernel's `__builtin_bit_cast`
//! formulation. A host self-test confirms the packer round-trips through an
//! independently written spec unpacker before any GPU work, so a FAIL points at
//! the kernel and not at the bench's own encoder.
//!
//! Weight value (format spec, qt=34):
//!   `w = row_scale_fp16 · e4m3(block_scale_byte) · QUANT_STEP(0.88) · coord`
//! Container: 16 B row header (fp16 row scale @0, n_blocks:u16 @4, flags 0x05
//! @6) then (K/32) × 17 B blocks = [1 B E4M3 scale][4 × u32 E8 codeword];
//! one 256-group = 8 blocks = 136 B. Rows are concatenated, no global prefix.
//!
//! `x` is FWHT-256 pre-rotated by the caller in production. This bench feeds an
//! ARBITRARY deterministic `x` and uses the identical bytes on both the GPU and
//! the host reference, which is all the parity check requires. Likewise the
//! synthetic E8 coordinates are uniform over the box rather than the real
//! FWHT-Gaussian, so nothing here is a quantization-QUALITY measurement.
//!
//! Run: cargo run --release -p rdna-compute --example bench_mfp4g32_e8_residual

use hip_bridge::KernargBlob;
use rdna_compute::{Gpu, GpuTensor};
use std::time::Instant;

const E8_SRC: &str = include_str!("../../../kernels/src/gemv_mfp4g32_e8_residual.hip");
const MQ3_SRC: &str = include_str!("../../../kernels/src/gemv_mq3g256_lloyd_residual.hip");

// ── format constants (authoritative: e8.rs / quantize_mfp4g32_e8_row) ────────
/// mfp4-E8 recenter bias: biased coord [0,15] → integer coord [-7,8].
const E8_COORD_BIAS: i32 = 7;
/// e8.rs::QUANT_STEP.
const E8_QUANT_STEP: f64 = 0.88;
/// 1 B E4M3 block scale + 4 × u32 E8 codeword.
const E8_BLOCK_BYTES: usize = 17;
/// fp16 row scale @0, n_blocks:u16 @4, flags @6, zero-padded to 16.
const E8_ROW_HEADER: usize = 16;
/// MQ3-Lloyd (qt=20): 8 × fp16 codebook (16 B) + 96 B of 3-bit indices.
const MQ3_GROUP_BYTES: usize = 112;

/// Relative tolerance for f32 accumulation at this size.
const TOL_REL: f64 = 1e-4;
/// Rows whose |reference| falls below `REL_FLOOR × L1(row)` are numerically
/// ill-conditioned by cancellation (a handful are expected out of 2048); the
/// relative error for those is measured against the floor instead of against a
/// near-zero denominator. Reported separately so the slack is visible.
const REL_FLOOR: f64 = 1e-3;

/// E4M3 scale bytes exercised on the first `EDGE_ROWS` rows: zero, three
/// subnormals/min-normal, unity, and the top of the range including the
/// repurposed NaN slot (0x7f → 448).
const EDGE_SCALE_BYTES: [u8; 8] = [0x00, 0x01, 0x07, 0x08, 0x38, 0x77, 0x78, 0x7f];
const EDGE_ROWS: usize = 64;

// ── deterministic hash (not kernel math) ────────────────────────────────────
fn mix(x: u64) -> u64 {
    let h = x
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    ((h ^ (h >> 33)).wrapping_mul(0xff51afd7ed558ccd)) ^ (h >> 28)
}

// ── IEEE-754 binary16, derived from the standard (not from the kernel) ──────
/// Truncating f32 → binary16 bits. Rounding is irrelevant here: whatever bits
/// land on disk are what the CPU reference decodes, so the two sides cannot
/// disagree because of a conversion mode.
fn f16_bits(f: f32) -> u16 {
    let b = f.to_bits();
    let sign = ((b >> 16) & 0x8000) as u16;
    let mut e = ((b >> 23) & 0xff) as i32 - 127 + 15;
    let mant = b & 0x7f_ffff;
    if e <= 0 {
        return sign; // flush to zero — inputs here are far from subnormal
    }
    if e >= 31 {
        e = 30;
    }
    sign | ((e as u16) << 10) | ((mant >> 13) as u16)
}

/// binary16 bits → f32, spec form (sign · significand · 2^(e-15)), subnormals
/// and inf/nan handled explicitly.
fn f16_to_f32(h: u16) -> f32 {
    let sign = if h & 0x8000 != 0 { -1.0f32 } else { 1.0f32 };
    let e = ((h >> 10) & 0x1f) as i32;
    let m = (h & 0x3ff) as f32;
    if e == 0 {
        sign * m * (2.0f32).powi(-24)
    } else if e == 31 {
        if m == 0.0 {
            sign * f32::INFINITY
        } else {
            f32::NAN
        }
    } else {
        sign * (1.0 + m / 1024.0) * (2.0f32).powi(e - 15)
    }
}

/// UNSIGNED OCP E4M3 (4 exp bits, bias 7, 3 mantissa bits) → f32, from the
/// spec. exp=0 is subnormal 2^-6·(mant/8); the single NaN code (exp=15,
/// mant=7) is repurposed by this format as the max finite value 448. Bit 7 is
/// not part of the scale encoding and is ignored.
fn e4m3_decode(byte: u8) -> f32 {
    let exp = ((byte >> 3) & 0xf) as i32;
    let mant = (byte & 0x7) as f32;
    if exp == 0 {
        (2.0f32).powi(-6) * (mant / 8.0)
    } else if exp == 0xf && (byte & 0x7) == 7 {
        448.0
    } else {
        (2.0f32).powi(exp - 7) * (1.0 + mant / 8.0)
    }
}

// ── E8: choose a lattice point, then pack it per the format spec ────────────
struct E8Word {
    packed: u32,
    coords: [f32; 8],
}

/// Pick an E8 lattice point and emit its 32-bit codeword.
///
/// The point is chosen FIRST (biased coords e[0..8] in [0,15]), so the caller
/// holds the ground-truth coordinates without ever running a decoder. The
/// D8 membership constraint — sum of the 8 biased coords must be EVEN — is what
/// makes e[7]'s low bit redundant and therefore droppable by the packing, so it
/// is enforced here by construction rather than repaired afterwards.
fn gen_e8_word(h: u64) -> E8Word {
    let mut e = [0u32; 8];
    let mut sum7 = 0u32;
    for (i, slot) in e.iter_mut().take(7).enumerate() {
        *slot = ((h >> (4 * i)) & 0xF) as u32;
        sum7 += *slot;
    }
    // e[7] is stored as its high 3 bits only; its LSB is whatever makes the
    // total sum even.
    let p7 = ((h >> 28) & 0x7) as u32; // = e[7] >> 1
    let lsb = (sum7 + 2 * p7) % 2;
    e[7] = 2 * p7 + lsb;
    let coset = ((h >> 40) & 1) as u32;

    let mut packed = 0u32;
    for (i, &v) in e.iter().take(7).enumerate() {
        packed |= v << (4 * i as u32);
    }
    packed |= p7 << 28;
    packed |= coset << 31;

    let mut coords = [0.0f32; 8];
    for (i, c) in coords.iter_mut().enumerate() {
        let integer = e[i] as i32 - E8_COORD_BIAS;
        *c = integer as f32 + if coset == 1 { 0.5 } else { 0.0 };
    }
    E8Word { packed, coords }
}

/// Independently written unpacker used ONLY to self-test `gen_e8_word` on the
/// host before any GPU work. Derived from the bit-layout spec, so a FAIL of the
/// GPU parity check cannot be blamed on this bench's own packer.
fn unpack_e8_spec(idx: u32) -> [f32; 8] {
    let coset = (idx >> 31) & 1;
    let mut e = [0u32; 8];
    let mut sum7 = 0u32;
    for (i, slot) in e.iter_mut().take(7).enumerate() {
        *slot = (idx >> (4 * i as u32)) & 0xF;
        sum7 += *slot;
    }
    let p7 = (idx >> 28) & 0x7;
    // Solve the even-sum constraint for the dropped LSB.
    e[7] = 2 * p7 + ((sum7 + 2 * p7) % 2);
    let mut coords = [0.0f32; 8];
    for (i, c) in coords.iter_mut().enumerate() {
        let integer = e[i] as i32 - E8_COORD_BIAS;
        *c = integer as f32 + if coset == 1 { 0.5 } else { 0.0 };
    }
    coords
}

fn selftest_e8_packing() -> bool {
    let mut bad = 0usize;
    for n in 0..200_000u64 {
        let h = mix(0xE8_5E1F ^ n);
        let w = gen_e8_word(h);
        // (a) the packed word must round-trip through the spec unpacker
        let back = unpack_e8_spec(w.packed);
        for i in 0..8 {
            if (back[i] - w.coords[i]).abs() > 1e-6 {
                bad += 1;
            }
        }
        // (b) the point must actually be in E8: integer (or half-integer)
        //     coords whose underlying D8 sum is even
        let shift = if (w.packed >> 31) & 1 == 1 { 0.5 } else { 0.0 };
        let s: i32 = w
            .coords
            .iter()
            .map(|&c| (c - shift).round() as i32)
            .sum::<i32>();
        if s % 2 != 0 {
            bad += 1;
        }
    }
    if bad == 0 {
        println!("  host packer self-test  : OK (200k codewords round-trip, all D8-even)");
    } else {
        println!("  host packer self-test  : FAILED ({bad} mismatches) — bench bug, not kernel");
    }
    bad == 0
}

// ── synthetic weight builders (bytes + CPU reference in one pass) ───────────
struct Built {
    bytes: Vec<u8>,
    /// exact per-row dot product A[row]·x in f64
    dot: Vec<f64>,
    /// per-row sum |w·x| — the conditioning of the f32 accumulation
    l1: Vec<f64>,
}

/// MFP4-E8 (qt=34) in the exact on-disk byte layout, plus the reference.
fn build_e8(m: usize, k: usize, x: &[f32], seed: u64, edge_rows: usize) -> Built {
    assert!(k % 256 == 0, "E8 GEMV assumes K%256==0");
    let n_blocks = k / 32;
    let row_bytes = E8_ROW_HEADER + n_blocks * E8_BLOCK_BYTES;
    let mut bytes = vec![0u8; m * row_bytes];
    let mut dot = vec![0.0f64; m];
    let mut l1 = vec![0.0f64; m];

    for row in 0..m {
        let base = row * row_bytes;

        // 16-B row header: fp16 row scale @0, n_blocks:u16 @4, flags 0x05 @6.
        let rs_f = 0.0625f32 + ((mix(seed ^ 0x51 ^ (row as u64) << 8) % 256) as f32) * (0.1875 / 256.0);
        let rs_bits = f16_bits(rs_f);
        let row_scale = f16_to_f32(rs_bits) as f64;
        bytes[base..base + 2].copy_from_slice(&rs_bits.to_le_bytes());
        bytes[base + 4..base + 6].copy_from_slice(&(n_blocks as u16).to_le_bytes());
        bytes[base + 6] = 0x05;

        for b in 0..n_blocks {
            // Edge rows pin one E4M3 scale byte for the whole row so the row's
            // magnitudes stay homogeneous (row 0 → 0x00 → exact-zero row).
            let sb = if row < edge_rows {
                EDGE_SCALE_BYTES[row % EDGE_SCALE_BYTES.len()]
            } else {
                0x20u8 + (mix(seed ^ 0xB10C ^ ((row as u64) << 24) ^ b as u64) % 0x20) as u8
            };
            let scale = row_scale * e4m3_decode(sb) as f64 * E8_QUANT_STEP;

            let off = base + E8_ROW_HEADER + b * E8_BLOCK_BYTES;
            bytes[off] = sb;

            for cw in 0..4usize {
                let w = gen_e8_word(mix(
                    seed ^ 0xC0DE ^ ((row as u64) << 40) ^ ((b as u64) << 8) ^ cw as u64,
                ));
                bytes[off + 1 + cw * 4..off + 5 + cw * 4].copy_from_slice(&w.packed.to_le_bytes());
                for i in 0..8usize {
                    // Format layout: block b holds weights [32b, 32b+32);
                    // codeword cw holds its 8-element slice of that block.
                    let col = b * 32 + cw * 8 + i;
                    let term = scale * w.coords[i] as f64 * x[col] as f64;
                    dot[row] += term;
                    l1[row] += term.abs();
                }
            }
        }
    }
    Built { bytes, dot, l1 }
}

/// MQ3-Lloyd (qt=20) in the exact on-disk byte layout, plus the reference.
/// 112 B/group = 8 × fp16 codebook then 96 B of 3-bit indices, element `e`
/// occupying bits [3e, 3e+3) of the little-endian index bitstream.
fn build_mq3(m: usize, k: usize, x: &[f32], seed: u64) -> Built {
    assert!(k % 256 == 0);
    let gpr = k / 256;
    let mut bytes = vec![0u8; m * gpr * MQ3_GROUP_BYTES];
    let mut dot = vec![0.0f64; m];
    let mut l1 = vec![0.0f64; m];

    // Lloyd-ish 8-level shape; the absolute values are irrelevant to parity.
    const LEVELS: [f32; 8] = [-1.62, -1.02, -0.62, -0.20, 0.20, 0.62, 1.02, 1.62];

    for row in 0..m {
        for g in 0..gpr {
            let off = (row * gpr + g) * MQ3_GROUP_BYTES;
            let gs =
                0.010f32 + ((mix(seed ^ ((row as u64) << 20) ^ g as u64) % 4000) as f32) * 1e-5;
            let mut cb = [0.0f64; 8];
            for (j, &lvl) in LEVELS.iter().enumerate() {
                let bits = f16_bits(lvl * gs);
                bytes[off + 2 * j..off + 2 * j + 2].copy_from_slice(&bits.to_le_bytes());
                cb[j] = f16_to_f32(bits) as f64;
            }
            let idx_base = off + 16;
            for e in 0..256usize {
                let q = (mix(seed ^ 0x3B17 ^ ((row as u64) << 36) ^ ((g as u64) << 12) ^ e as u64)
                    & 7) as usize;
                let bit = 3 * e;
                for j in 0..3 {
                    if (q >> j) & 1 == 1 {
                        bytes[idx_base + (bit + j) / 8] |= 1 << ((bit + j) % 8);
                    }
                }
                let col = g * 256 + e;
                let term = cb[q] * x[col] as f64;
                dot[row] += term;
                l1[row] += term.abs();
            }
        }
    }
    Built { bytes, dot, l1 }
}

// ── parity metric ───────────────────────────────────────────────────────────
struct Parity {
    max_abs: f64,
    abs_idx: usize,
    max_rel: f64,
    rel_idx: usize,
    ill: usize,
}

/// `rel_i = |err_i| / max(|want_i|, REL_FLOOR · l1_i)`. The floor keeps a row
/// whose reference happens to cancel to ~0 from producing a meaningless
/// division; such rows are counted in `ill`. A row with l1 == 0 (zero scale
/// byte) must match EXACTLY or it reports infinite relative error.
fn parity(got: &[f32], want: &[f64], l1: &[f64]) -> Parity {
    let mut p = Parity {
        max_abs: 0.0,
        abs_idx: 0,
        max_rel: 0.0,
        rel_idx: 0,
        ill: 0,
    };
    for i in 0..want.len() {
        let err = (got[i] as f64 - want[i]).abs();
        if err > p.max_abs {
            p.max_abs = err;
            p.abs_idx = i;
        }
        let floor = REL_FLOOR * l1[i];
        if want[i].abs() < floor {
            p.ill += 1;
        }
        let den = want[i].abs().max(floor);
        let rel = if den > 0.0 {
            err / den
        } else if err == 0.0 {
            0.0
        } else {
            f64::INFINITY
        };
        if rel > p.max_rel {
            p.max_rel = rel;
            p.rel_idx = i;
        }
    }
    p
}

fn report(tag: &str, p: &Parity, got: &[f32], want: &[f64]) -> bool {
    let ok = p.max_rel <= TOL_REL;
    println!(
        "  {tag:<28} max|abs|={:.3e} @row {}   max rel={:.3e} @row {}   (gpu {:.6} vs cpu {:.6})",
        p.max_abs, p.abs_idx, p.max_rel, p.rel_idx, got[p.rel_idx], want[p.rel_idx]
    );
    if p.ill > 0 {
        println!(
            "  {:<28} {} ill-conditioned row(s) measured against the {:.0e}·L1 floor",
            "", p.ill, REL_FLOOR
        );
    }
    println!(
        "  {:<28} {}  (tolerance {:.0e} relative)",
        "",
        if ok { "PASS" } else { "*** FAIL ***" },
        TOL_REL
    );
    ok
}

// ── helpers ─────────────────────────────────────────────────────────────────
fn reseed(gpu: &Gpu, t: &GpuTensor, data: &[f32]) {
    let raw =
        unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, std::mem::size_of_val(data)) };
    gpu.hip.memcpy_htod(&t.buf, raw).expect("reseed y");
}

fn gemv_args(a: &GpuTensor, x: &GpuTensor, y: &GpuTensor, m: usize, k: usize) -> Vec<u8> {
    let mut b = KernargBlob::new();
    b.push_ptr(a.buf.as_ptr() as *const _);
    b.push_ptr(x.buf.as_ptr() as *const _);
    b.push_ptr(y.buf.as_ptr() as *const _);
    b.push_i32(m as i32);
    b.push_i32(k as i32);
    b.into_vec()
}

/// Warmup then `reps` timed reps of `iters` launches each. Returns per-call
/// microseconds for every rep. Rep 0 of the FIRST kernel touched is the one
/// most exposed to JIT/cache effects; the warmup exists to absorb that, and MIN
/// / MEDIAN are reported instead of a mean so a straggler cannot dominate.
fn bench(
    gpu: &Gpu,
    name: &str,
    grid: [u32; 3],
    block: [u32; 3],
    args: &mut [u8],
    iters: usize,
    reps: usize,
) -> Vec<f64> {
    for _ in 0..20 {
        gpu.launch_kernel_blob(name, grid, block, 0, args).unwrap();
    }
    gpu.hip.device_synchronize().unwrap();
    let mut out = Vec::with_capacity(reps);
    for _ in 0..reps {
        let t0 = Instant::now();
        for _ in 0..iters {
            gpu.launch_kernel_blob(name, grid, block, 0, args).unwrap();
        }
        gpu.hip.device_synchronize().unwrap();
        out.push(t0.elapsed().as_secs_f64() * 1e6 / iters as f64);
    }
    out
}

fn min_med(v: &[f64]) -> (f64, f64) {
    let mut s = v.to_vec();
    s.sort_by(|a, b| a.partial_cmp(b).unwrap());
    (s[0], s[s.len() / 2])
}

fn main() {
    let mut gpu = Gpu::init().expect("GPU init");
    println!("arch={}", gpu.arch);

    // dense o_proj / out_proj (the residual shape)
    let m = 2048usize;
    let k = 4096usize;
    // tail-exercising shape: 7 groups → quads=1, tail=3 (K=4096 has tail=0)
    let m_tail = 128usize;
    let k_tail = 1792usize;

    println!("\nformat / shape");
    println!("  dense o_proj              : M={m} K={k}  (a3b hidden=2048)");
    println!("  tail-branch shape         : M={m_tail} K={k_tail} (groups=7 → quads=1 tail=3)");

    println!("\npre-flight (host only)");
    let packer_ok = selftest_e8_packing();

    // Arbitrary deterministic x (production x is FWHT-256 pre-rotated; parity
    // only needs GPU and host to see identical bytes).
    let x: Vec<f32> = (0..k)
        .map(|i| ((mix(0xA11CE ^ i as u64) % 2001) as f32 - 1000.0) * 0.001)
        .collect();

    // Distinctive nonzero seed for y: an overwriting kernel loses all of it.
    let seed: Vec<f32> = (0..m)
        .map(|r| {
            let s = if r % 2 == 0 { 1.0 } else { -1.0 };
            s * (7.0 + (r % 13) as f32 * 0.5)
        })
        .collect();
    let seed_tail: Vec<f32> = (0..m_tail)
        .map(|r| {
            let s = if r % 3 == 0 { -1.0 } else { 1.0 };
            s * (3.0 + (r % 7) as f32 * 0.25)
        })
        .collect();

    println!("\nbuilding synthetic weights + CPU reference (f64)");
    let e8 = build_e8(m, k, &x, 0x1234_5678, EDGE_ROWS);
    let e8_tail = build_e8(m_tail, k_tail, &x, 0x9ABC_DEF0, 8);
    let mq3 = build_mq3(m, k, &x, 0x0F0F_0F0F);

    let e8_row_bytes = E8_ROW_HEADER + (k / 32) * E8_BLOCK_BYTES;
    let mq3_row_bytes = (k / 256) * MQ3_GROUP_BYTES;
    let e8_bytes = m * e8_row_bytes;
    let mq3_bytes = m * mq3_row_bytes;
    println!(
        "  MFP4-E8   {:>9} B  ({:.4} B/elem, {:.4} bpw)",
        e8_bytes,
        e8_bytes as f64 / (m * k) as f64,
        e8_bytes as f64 * 8.0 / (m * k) as f64
    );
    println!(
        "  MQ3-Lloyd {:>9} B  ({:.4} B/elem, {:.4} bpw)",
        mq3_bytes,
        mq3_bytes as f64 / (m * k) as f64,
        mq3_bytes as f64 * 8.0 / (m * k) as f64
    );

    let a_e8 = gpu.upload_raw(&e8.bytes, &[e8.bytes.len()]).unwrap();
    let a_e8t = gpu.upload_raw(&e8_tail.bytes, &[e8_tail.bytes.len()]).unwrap();
    let a_mq3 = gpu.upload_raw(&mq3.bytes, &[mq3.bytes.len()]).unwrap();
    let x_t = gpu.upload_f32(&x, &[k]).unwrap();
    let y_e8 = gpu.upload_f32(&seed, &[m]).unwrap();
    let y_e8t = gpu.upload_f32(&seed_tail, &[m_tail]).unwrap();
    let y_mq3 = gpu.upload_f32(&seed, &[m]).unwrap();
    // padded y for the over-provisioned-grid guard test; the pad is ours, so a
    // missing `row >= M` guard cannot reach another pooled buffer.
    const PAD: usize = 8;
    let mut seed_pad = seed_tail.clone();
    for j in 0..PAD {
        seed_pad.push(-99.0 - j as f32);
    }
    let y_pad = gpu.upload_f32(&seed_pad, &[m_tail + PAD]).unwrap();

    gpu.ensure_kernel_public(
        "gemv_mfp4g32_e8_residual",
        E8_SRC,
        "gemv_mfp4g32_e8_residual",
    )
    .expect("JIT gemv_mfp4g32_e8_residual");
    gpu.ensure_kernel_public(
        "gemv_mq3g256_lloyd_residual",
        MQ3_SRC,
        "gemv_mq3g256_lloyd_residual",
    )
    .expect("JIT gemv_mq3g256_lloyd_residual");

    let block = [32u32, 1, 1];
    let grid = [m as u32, 1, 1];
    let grid_tail = [m_tail as u32, 1, 1];

    let mut args_e8 = gemv_args(&a_e8, &x_t, &y_e8, m, k);
    let mut args_e8t = gemv_args(&a_e8t, &x_t, &y_e8t, m_tail, k_tail);
    let mut args_mq3 = gemv_args(&a_mq3, &x_t, &y_mq3, m, k);
    let mut args_pad = gemv_args(&a_e8t, &x_t, &y_pad, m_tail, k_tail);

    // ── correctness ─────────────────────────────────────────────────────────
    println!("\ncorrectness — each kernel vs its own CPU reference");
    let mut all_ok = packer_ok;

    // pass 1: y must be seed + dot
    gpu.launch_kernel_blob("gemv_mfp4g32_e8_residual", grid, block, 0, &mut args_e8)
        .unwrap();
    gpu.hip.device_synchronize().unwrap();
    let got1 = gpu.download_f32(&y_e8).unwrap();
    let want1: Vec<f64> = (0..m).map(|i| seed[i] as f64 + e8.dot[i]).collect();
    let cond1: Vec<f64> = (0..m).map(|i| seed[i].abs() as f64 + e8.l1[i]).collect();
    all_ok &= report(
        "E8 residual pass1 seed+dot",
        &parity(&got1, &want1, &cond1),
        &got1,
        &want1,
    );

    // pass 2 WITHOUT re-seeding: y must be seed + 2·dot. An overwriting kernel
    // returns the identical value twice and cannot pass this.
    gpu.launch_kernel_blob("gemv_mfp4g32_e8_residual", grid, block, 0, &mut args_e8)
        .unwrap();
    gpu.hip.device_synchronize().unwrap();
    let got2 = gpu.download_f32(&y_e8).unwrap();
    let want2: Vec<f64> = (0..m).map(|i| seed[i] as f64 + 2.0 * e8.dot[i]).collect();
    let cond2: Vec<f64> = (0..m)
        .map(|i| seed[i].abs() as f64 + 2.0 * e8.l1[i])
        .collect();
    all_ok &= report(
        "E8 residual pass2 seed+2dot",
        &parity(&got2, &want2, &cond2),
        &got2,
        &want2,
    );

    // Explicit accumulate-vs-overwrite verdict, independent of the tolerances.
    let mut overwrote = 0usize;
    let mut advanced = 0usize;
    for i in 0..m {
        if e8.dot[i].abs() > 1e-6 {
            if (got2[i] - got1[i]).abs() < 1e-6 {
                overwrote += 1;
            } else {
                advanced += 1;
            }
        }
    }
    println!(
        "  {:<28} {} of {} nonzero rows advanced on the 2nd launch, {} were static → {}",
        "accumulate-vs-overwrite",
        advanced,
        advanced + overwrote,
        overwrote,
        if overwrote == 0 {
            "ACCUMULATES (correct)"
        } else {
            "*** OVERWRITES ***"
        }
    );
    all_ok &= overwrote == 0;

    // Row 0 has E4M3 scale byte 0x00 → dot is exactly 0 → the seed must be
    // returned bit-identical after both launches.
    let zero_ok = got1[0].to_bits() == seed[0].to_bits() && got2[0].to_bits() == seed[0].to_bits();
    println!(
        "  {:<28} row0 scale byte 0x00 → dot==0; y stayed {} (seed {}) → {}",
        "zero-scale row bit-identity", got2[0], seed[0], if zero_ok { "PASS" } else { "*** FAIL ***" }
    );
    all_ok &= zero_ok;

    // tail branches (quads=1, tail=3)
    gpu.launch_kernel_blob(
        "gemv_mfp4g32_e8_residual",
        grid_tail,
        block,
        0,
        &mut args_e8t,
    )
    .unwrap();
    gpu.hip.device_synchronize().unwrap();
    let gott = gpu.download_f32(&y_e8t).unwrap();
    let wantt: Vec<f64> = (0..m_tail)
        .map(|i| seed_tail[i] as f64 + e8_tail.dot[i])
        .collect();
    let condt: Vec<f64> = (0..m_tail)
        .map(|i| seed_tail[i].abs() as f64 + e8_tail.l1[i])
        .collect();
    all_ok &= report(
        "E8 K=1792 tail branches",
        &parity(&gott, &wantt, &condt),
        &gott,
        &wantt,
    );

    // over-provisioned grid: M+5 blocks, padded y. In-range rows must still be
    // seed+dot and every pad slot must come back bit-identical.
    gpu.launch_kernel_blob(
        "gemv_mfp4g32_e8_residual",
        [(m_tail + 5) as u32, 1, 1],
        block,
        0,
        &mut args_pad,
    )
    .unwrap();
    gpu.hip.device_synchronize().unwrap();
    let gotp = gpu.download_f32(&y_pad).unwrap();
    all_ok &= report(
        "E8 grid M+5 in-range rows",
        &parity(&gotp[..m_tail], &wantt, &condt),
        &gotp[..m_tail],
        &wantt,
    );
    let pad_ok = (0..PAD).all(|j| gotp[m_tail + j].to_bits() == seed_pad[m_tail + j].to_bits());
    println!(
        "  {:<28} {} pad slot(s) past M bit-identical → {}",
        "row>=M guard", PAD,
        if pad_ok { "PASS" } else { "*** FAIL (guard missing / OOB write) ***" }
    );
    all_ok &= pad_ok;

    // MQ3 reference kernel on the same shape (different format — parity is
    // against its OWN CPU reference only, never against E8).
    gpu.launch_kernel_blob("gemv_mq3g256_lloyd_residual", grid, block, 0, &mut args_mq3)
        .unwrap();
    gpu.hip.device_synchronize().unwrap();
    let gotq = gpu.download_f32(&y_mq3).unwrap();
    let wantq: Vec<f64> = (0..m).map(|i| seed[i] as f64 + mq3.dot[i]).collect();
    let condq: Vec<f64> = (0..m).map(|i| seed[i].abs() as f64 + mq3.l1[i]).collect();
    all_ok &= report(
        "MQ3 residual pass1",
        &parity(&gotq, &wantq, &condq),
        &gotq,
        &wantq,
    );

    // ── timing ──────────────────────────────────────────────────────────────
    reseed(&gpu, &y_e8, &seed);
    reseed(&gpu, &y_mq3, &seed);
    let iters = 50usize;
    let reps = 7usize;
    println!("\ntiming — {reps} reps × {iters} launches, after a 20-launch warmup");
    println!("  (a first-pass number is JIT-contaminated; the warmup absorbs it and MIN/MEDIAN");
    println!("   are reported instead of a mean. HIP-dispatch regime — see the header caveat.)");

    let t_e8 = bench(
        &gpu,
        "gemv_mfp4g32_e8_residual",
        grid,
        block,
        &mut args_e8,
        iters,
        reps,
    );
    let t_mq3 = bench(
        &gpu,
        "gemv_mq3g256_lloyd_residual",
        grid,
        block,
        &mut args_mq3,
        iters,
        reps,
    );
    let (e8_min, e8_med) = min_med(&t_e8);
    let (q_min, q_med) = min_med(&t_mq3);

    let gbs = |bytes: usize, us: f64| bytes as f64 / (us * 1e-6) / 1e9;
    println!(
        "\n  {:<12} {:>10} {:>10} {:>12} {:>12} {:>11} {:>10}",
        "kernel", "min us", "med us", "GB/s @min", "GB/s @med", "calls/s", "B/elem"
    );
    println!(
        "  {:<12} {:>10.2} {:>10.2} {:>12.1} {:>12.1} {:>11.0} {:>10.4}",
        "MFP4-E8",
        e8_min,
        e8_med,
        gbs(e8_bytes, e8_min),
        gbs(e8_bytes, e8_med),
        1e6 / e8_med,
        e8_bytes as f64 / (m * k) as f64
    );
    println!(
        "  {:<12} {:>10.2} {:>10.2} {:>12.1} {:>12.1} {:>11.0} {:>10.4}",
        "MQ3-Lloyd",
        q_min,
        q_med,
        gbs(mq3_bytes, q_min),
        gbs(mq3_bytes, q_med),
        1e6 / q_med,
        mq3_bytes as f64 / (m * k) as f64
    );
    println!(
        "  per-rep us  E8 {:?}",
        t_e8.iter().map(|v| (v * 100.0).round() / 100.0).collect::<Vec<_>>()
    );
    println!(
        "  per-rep us MQ3 {:?}",
        t_mq3.iter().map(|v| (v * 100.0).round() / 100.0).collect::<Vec<_>>()
    );
    println!(
        "\n  MFP4-E8 is {:+.2}% on median time vs MQ3-Lloyd on the same M×K, moving {:+.1}% bytes.",
        100.0 * (e8_med / q_med - 1.0),
        100.0 * (e8_bytes as f64 / mq3_bytes as f64 - 1.0)
    );
    println!("  (different formats — this is a scale reference, NOT a parity or quality claim)");

    println!(
        "\n=== {} === gemv_mfp4g32_e8_residual  M={m} K={k}  (triage only; acceptance = golden bundle, HIP + PM4 arms)",
        if all_ok { "PASS" } else { "FAIL" }
    );
    if !all_ok {
        std::process::exit(1);
    }
}
