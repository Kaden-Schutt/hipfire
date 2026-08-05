// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
//! MQ3-GL (global 8-entry codebook + per-block fp16 scale, 96 B indices +
//! 2 B scale per group = 3.0625 bpw) vs MQ3-Lloyd (per-block 8-entry fp16
//! codebook, 112 B/group = 3.5 bpw) on the a3b routed-expert **down**
//! projection decode shape, M=2048 K=512 k_top=8.
//!
//! Kernels under test (both JIT'd from source, launched via the kernarg-blob
//! path — this bench touches no dispatch plumbing and no kernel source):
//!   * kernels/src/gemv_mq3g256gl_moe_down_indexed.hip
//!       gemv_mq3g256gl_moe_down_residual_scaled_k8_indexed
//!   * kernels/src/gemv_mq3g256_lloyd_moe_down_indexed.hip
//!       gemv_mq3g256_lloyd_moe_down_residual_scaled_k8_indexed
//!
//! ## What makes the down kernels different from the gate_up ones
//!
//! ATOMIC SELF-COMBINING. There is no expanded per-expert output buffer and no
//! separate down-combine kernel: all `k_top` grid-y blocks `atomicAdd` into the
//! SAME `x_residual[row]` cell, on top of whatever the caller left there. So
//! the CPU reference must (a) start from the pre-seeded residual, (b) sum over
//! all k_top experts, (c) apply `topk_weights[krank]`, and (d) use each krank's
//! OWN `rot_batch[krank*K ..]` activation slice. Every one of those is checked
//! here, and each is checked in a way that a kernel getting it wrong cannot
//! accidentally pass:
//!   - the seed residual is a non-zero, row-varying pattern (a kernel that
//!     stores instead of accumulating fails);
//!   - `topk_weights` are distinct and two are NEGATIVE (a kernel ignoring the
//!     weights, or folding |w|, fails);
//!   - each krank gets a completely different x slice (a kernel reading
//!     rot_batch[0] for every krank fails);
//!   - the topk index list contains DUPLICATE expert ids (ranks 0/2 and 1/7),
//!     which both exercises real self-contention on one expert blob and
//!     detects a kernel that dedupes or that indexes the ptr table by krank
//!     instead of by topk_indices[krank].
//!
//! ## Independence of the CPU reference
//!
//! The reference decodes the weight blob from the FORMAT SPEC (the encoders
//! `quantize_mq3g256gl` / `quantize_mq3g256_lloyd` in hipfire-quantize), not
//! from the kernel's arithmetic:
//!   - the 3-bit cross-byte unpack is done ONE BIT AT A TIME from
//!     "little-endian 24-bit bitstream, code j at bit offset 3j", sharing no
//!     shift/mask expression with either the kernel's `(pk >> 3j) & 7` or the
//!     encoder's explicit `b0/b1/b2` byte formulas;
//!   - the packer in this file uses the ENCODER's byte formulas, so packer and
//!     unpacker are two independent readings of the same spec — a startup
//!     self-test asserts they agree (a shared misunderstanding would have to
//!     survive both forms);
//!   - the column mapping comes from the encoder ("group element i ↔ column
//!     g*256+i, code i packed at chunk i/8, slot i%8"), never from the kernel's
//!     "thread tid owns chunk tid" formulation;
//!   - accumulation is in f64 over the full output vector (all M rows), not a
//!     single spot check.
//!
//! `x` is nominally FWHT-256 pre-rotated by the caller for these formats. This
//! bench feeds arbitrary deterministic pseudo-random `x` and uses the SAME
//! bytes for GPU and CPU reference, so the rotation is irrelevant to what is
//! being validated here (decode + routing + accumulation), and no FWHT is
//! applied anywhere.
//!
//! ## REGIME CAVEAT — read before quoting any number from this bench
//!
//! This is a HIP-dispatch microbenchmark. It is a TRIAGE FILTER for gross
//! defects (wrong numbers, pathological slowness) — NOT a verdict on a kernel.
//! Host launch latency masks device-level effects, and the same kernel can
//! measure differently once lowered to retained PM4 replay. A prior HIP
//! microbench measured the MQ2GL kernel at -3.1% and that was nearly used to
//! kill the format. Final acceptance is the golden bundle
//! (registry/redline-golden-v1.json) with HIP and PM4 arms. The first timed
//! pass of any (kernel × shape) cell is JIT-contaminated; a warmup burst runs
//! before every measurement here and the reported numbers exclude it.
//!
//! Run: cargo run --release -p rdna-compute --example bench_mq3g256gl_moe_down

use hip_bridge::KernargBlob;
use rdna_compute::{Gpu, GpuTensor};
use std::time::Instant;

const MQ3GL_SRC: &str = include_str!("../../../kernels/src/gemv_mq3g256gl_moe_down_indexed.hip");
const MQ3L_SRC: &str =
    include_str!("../../../kernels/src/gemv_mq3g256_lloyd_moe_down_indexed.hip");

const GL_FN: &str = "gemv_mq3g256gl_moe_down_residual_scaled_k8_indexed";
const L_FN: &str = "gemv_mq3g256_lloyd_moe_down_residual_scaled_k8_indexed";

/// The tensor-global 3-bit Lloyd codebook (`GL_CB3` in hipfire-quantize).
const CB3: [f32; 8] = [
    -2.1520, -1.3439, -0.7560, -0.2451, 0.2451, 0.7560, 1.3439, 2.1520,
];

/// Bytes of packed indices per 256-weight group (both MQ3 formats: 8 codes ×
/// 3 bits = 3 B per chunk of 8, 32 chunks).
const IDX_BYTES_PER_GROUP: usize = 96;
/// MQ3-GL: 96 B indices (region 1) + 2 B fp16 scale (region 2).
const GL_BYTES_PER_GROUP: usize = 98;
/// MQ3-Lloyd: 16 B header (8 × fp16 codebook) + 96 B indices, interleaved.
const L_BYTES_PER_GROUP: usize = 112;

// ── deterministic PRNG (same mixer as bench_mq2gl_vs_mq2l) ──────────────────

fn mix(x: u64) -> u64 {
    let h = x
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    ((h ^ (h >> 33)).wrapping_mul(0xff51afd7ed558ccd)) ^ (h >> 28)
}

// ── fp16 helpers ───────────────────────────────────────────────────────────
// Truncating (round-toward-zero) f32→f16, same converter as the MQ2GL/MQ2L
// sibling bench. The production encoder rounds half-to-even instead; that
// difference is irrelevant to every GATED check here, because the reference
// reads back exactly the bytes this file wrote. It shows up in exactly one
// INFORMATIONAL number — the case-A cross-format compare, where truncation
// makes fp16(scale·cb) sit up to a full fp16 ulp (2^-10 ≈ 1e-3 relative, always
// toward zero) below scale·cb. Every value fed through here is well inside the
// fp16 normal range, so no subnormal/overflow path is exercised.

fn half_bits(f: f32) -> u16 {
    let b = f.to_bits();
    let sign = ((b >> 16) & 0x8000) as u16;
    let mut e = ((b >> 23) & 0xff) as i32 - 127 + 15;
    let mant = b & 0x7f_ffff;
    if e <= 0 {
        return sign;
    }
    if e >= 31 {
        e = 30;
    }
    sign | ((e as u16) << 10) | ((mant >> 13) as u16)
}

fn half_to_f32(h: u16) -> f32 {
    let sign = ((h & 0x8000) as u32) << 16;
    let e = ((h >> 10) & 0x1f) as u32;
    let m = (h & 0x3ff) as u32;
    if e == 0 {
        return f32::from_bits(sign);
    }
    f32::from_bits(sign | ((e + 112) << 23) | (m << 13))
}

fn fp16_round(f: f32) -> f32 {
    half_to_f32(half_bits(f))
}

fn put_half(dst: &mut [u8], off: usize, v: f32) {
    let h = half_bits(v);
    dst[off] = (h & 0xff) as u8;
    dst[off + 1] = (h >> 8) as u8;
}

fn get_half(src: &[u8], off: usize) -> f32 {
    half_to_f32(u16::from_le_bytes([src[off], src[off + 1]]))
}

// ── synthetic weight generation ────────────────────────────────────────────

/// 3-bit code for weight `i` of group `g` in row `row`. Shared by both formats
/// so the two blobs encode the same index stream.
fn gen_codes(m: usize, k: usize, seed: u64) -> Vec<u8> {
    let gpr = k / 256;
    let mut codes = vec![0u8; m * gpr * 256];
    for row in 0..m {
        for g in 0..gpr {
            let base = (row * gpr + g) * 256;
            for i in 0..256 {
                let h = mix(seed ^ ((row as u64) << 32) ^ ((g as u64) << 16) ^ (i as u64) ^ 0x5bf0);
                codes[base + i] = (h & 7) as u8;
            }
        }
    }
    codes
}

/// Per-(row, group) block scale.
///
/// `pow2 = true` picks an exact power of two. That matters for the strict
/// cross-format check: `scale * cb[e]` is then EXACTLY representable in fp16
/// (a power-of-two multiply only shifts the exponent), so the MQ3-Lloyd blob
/// built from `scale * CB3` and the MQ3-GL blob built from `scale` + global
/// `CB3` dequantize to bit-identical weights and the two kernels must agree to
/// f32-accumulation-order noise.
///
/// `pow2 = false` picks an arbitrary fp16 scale — the realistic case (the
/// encoder stores `fp16(rms)`), and the one that actually exercises fp16 scale
/// decode with a non-trivial mantissa.
fn gen_scales(m: usize, k: usize, seed: u64, pow2: bool) -> Vec<f32> {
    let gpr = k / 256;
    let mut out = vec![0f32; m * gpr];
    for row in 0..m {
        for g in 0..gpr {
            let h = mix(seed ^ ((row as u64) << 20) ^ ((g as u64).wrapping_mul(0x9e37)) ^ 0xa17);
            out[row * gpr + g] = if pow2 {
                // 2^-9 .. 2^-6
                f32::powi(2.0, -(6 + (h % 4) as i32))
            } else {
                fp16_round(0.004f32 + ((h % 4000) as f32) * 1e-6)
            };
        }
    }
    out
}

/// Pack the encoder's 8-codes-into-3-bytes form, verbatim from
/// `quantize_mq3g256_lloyd` / `quantize_mq3g256gl` (byte formulas, not the
/// LE24 shortcut).
fn pack_chunk(q: &[u8]) -> [u8; 3] {
    let b0 = q[0] | (q[1] << 3) | ((q[2] & 3) << 6);
    let b1 = (q[2] >> 2) | (q[3] << 1) | (q[4] << 4) | ((q[5] & 1) << 7);
    let b2 = (q[5] >> 1) | (q[6] << 2) | (q[7] << 5);
    [b0, b1, b2]
}

/// MQ3-GL (qt=39) SoA blob:
///   `[0 .. M*gpr*96)`            packed 3-bit indices, row-major in (row, g)
///   `[M*gpr*96 .. +M*gpr*2)`     fp16 per-block scales, row-major in (row, g)
fn build_mq3gl(m: usize, k: usize, codes: &[u8], scales: &[f32]) -> Vec<u8> {
    let gpr = k / 256;
    let idx_bytes = m * gpr * IDX_BYTES_PER_GROUP;
    let mut out = vec![0u8; idx_bytes + m * gpr * 2];
    for row in 0..m {
        for g in 0..gpr {
            let cbase = (row * gpr + g) * 256;
            let obase = (row * gpr + g) * IDX_BYTES_PER_GROUP;
            for c in 0..32 {
                let b = pack_chunk(&codes[cbase + 8 * c..cbase + 8 * c + 8]);
                out[obase + 3 * c] = b[0];
                out[obase + 3 * c + 1] = b[1];
                out[obase + 3 * c + 2] = b[2];
            }
            put_half(&mut out, idx_bytes + (row * gpr + g) * 2, scales[row * gpr + g]);
        }
    }
    out
}

/// MQ3-Lloyd (qt=20) blob: 112 B per group, `[8 × fp16 codebook][96 B indices]`,
/// row-major in (row, g). The per-block codebook here is `scale_g * CB3` so the
/// two formats describe the same weights (exactly so when `scale_g` is a power
/// of two — see `gen_scales`).
fn build_mq3l(m: usize, k: usize, codes: &[u8], scales: &[f32], cb: &[f32; 8]) -> Vec<u8> {
    let gpr = k / 256;
    let mut out = vec![0u8; m * gpr * L_BYTES_PER_GROUP];
    for row in 0..m {
        for g in 0..gpr {
            let cbase = (row * gpr + g) * 256;
            let obase = (row * gpr + g) * L_BYTES_PER_GROUP;
            let s = scales[row * gpr + g];
            for (e, &c) in cb.iter().enumerate() {
                put_half(&mut out, obase + 2 * e, s * c);
            }
            for c in 0..32 {
                let b = pack_chunk(&codes[cbase + 8 * c..cbase + 8 * c + 8]);
                out[obase + 16 + 3 * c] = b[0];
                out[obase + 16 + 3 * c + 1] = b[1];
                out[obase + 16 + 3 * c + 2] = b[2];
            }
        }
    }
    out
}

// ── independent decode (bit-level, derived from the spec) ──────────────────

/// Code `j` (0..8) of a 3-byte chunk, read one bit at a time from the spec's
/// "little-endian 24-bit bitstream, code j occupies bits [3j, 3j+3)". Shares
/// no expression with the packer above or with the kernel.
fn code_from_chunk(chunk: &[u8], j: usize) -> u8 {
    let mut v = 0u8;
    for t in 0..3 {
        let bit = 3 * j + t;
        v |= ((chunk[bit >> 3] >> (bit & 7)) & 1) << t;
    }
    v
}

/// Dequantize one row of an MQ3-GL blob into K f32 weights.
fn decode_row_gl(blob: &[u8], m: usize, k: usize, row: usize, cb: &[f32; 8], out: &mut [f32]) {
    let gpr = k / 256;
    let idx_bytes = m * gpr * IDX_BYTES_PER_GROUP;
    for g in 0..gpr {
        let s = get_half(blob, idx_bytes + (row * gpr + g) * 2);
        let base = (row * gpr + g) * IDX_BYTES_PER_GROUP;
        for c in 0..32 {
            let chunk = &blob[base + 3 * c..base + 3 * c + 3];
            for j in 0..8 {
                let q = code_from_chunk(chunk, j) as usize;
                out[g * 256 + 8 * c + j] = s * cb[q];
            }
        }
    }
}

/// Dequantize one row of an MQ3-Lloyd blob into K f32 weights. The codebook is
/// per block and lives in the block header; there is no separate scale.
fn decode_row_l(blob: &[u8], k: usize, row: usize, out: &mut [f32]) {
    let gpr = k / 256;
    for g in 0..gpr {
        let base = (row * gpr + g) * L_BYTES_PER_GROUP;
        let mut cb = [0f32; 8];
        for (e, c) in cb.iter_mut().enumerate() {
            *c = get_half(blob, base + 2 * e);
        }
        for c in 0..32 {
            let chunk = &blob[base + 16 + 3 * c..base + 16 + 3 * c + 3];
            for j in 0..8 {
                let q = code_from_chunk(chunk, j) as usize;
                out[g * 256 + 8 * c + j] = cb[q];
            }
        }
    }
}

/// Packer-vs-unpacker self-test: two independent readings of the same packing
/// spec must round-trip every code pattern.
fn packing_self_test() {
    let mut worst = 0usize;
    for pattern in 0..64u64 {
        let codes: Vec<u8> = (0..8)
            .map(|j| (mix(pattern * 977 + j) & 7) as u8)
            .collect();
        let packed = pack_chunk(&codes);
        for j in 0..8 {
            let back = code_from_chunk(&packed, j);
            assert_eq!(
                back, codes[j],
                "3-bit cross-byte pack/unpack disagree: pattern {pattern} slot {j}"
            );
            worst += 1;
        }
    }
    // Cross-byte slots specifically (q2 spans b0/b1, q5 spans b1/b2).
    for a in 0..8u8 {
        let codes = [0, 0, a, 0, 0, a, 0, 0];
        let packed = pack_chunk(&codes);
        assert_eq!(code_from_chunk(&packed, 2), a, "q2 cross-byte slot");
        assert_eq!(code_from_chunk(&packed, 5), a, "q5 cross-byte slot");
    }
    println!("  packing self-test: {worst} round-trips + cross-byte slots q2/q5  OK");
}

// ── one bench/correctness case ─────────────────────────────────────────────

struct Case {
    m: usize,
    k: usize,
    k_top: usize,
    n_exp: usize,
    label: String,
    topk_idx: Vec<i32>,
    topk_w: Vec<f32>,
    xs: Vec<Vec<f32>>,
    seed_res: Vec<f32>,
    gl_host: Vec<Vec<u8>>,
    l_host: Vec<Vec<u8>>,
    cb: [f32; 8],
    gl_args: Vec<u8>,
    l_args: Vec<u8>,
    y: GpuTensor,
    _keep: Vec<GpuTensor>,
}

impl Case {
    fn grid(&self) -> [u32; 3] {
        [self.m as u32, self.k_top as u32, 1]
    }
    fn gpr(&self) -> usize {
        self.k / 256
    }
    /// Weight bytes touched by ONE launch (k_top experts × the full M×K tile).
    fn gl_bytes(&self) -> f64 {
        (self.k_top * self.m * self.gpr() * GL_BYTES_PER_GROUP) as f64
    }
    fn l_bytes(&self) -> f64 {
        (self.k_top * self.m * self.gpr() * L_BYTES_PER_GROUP) as f64
    }
}

#[allow(clippy::too_many_arguments)]
fn build_case(
    gpu: &mut Gpu,
    label: &str,
    m: usize,
    k: usize,
    k_top: usize,
    n_exp: usize,
    seed: u64,
    pow2: bool,
) -> Case {
    assert_eq!(k % 256, 0, "K must be a multiple of 256");
    let gpr = k / 256;

    // fp16-rounded global codebook: the kernel gets these as scalar args and
    // the MQ3-Lloyd blob stores fp16(scale * cb), so rounding the codebook
    // first is what makes the pow2 case exactly cross-format identical.
    let mut cb = [0f32; 8];
    for (d, &s) in cb.iter_mut().zip(CB3.iter()) {
        *d = fp16_round(s);
    }

    // Routing: DUPLICATE experts at ranks 0/2 and 1/7 (self-contention on one
    // blob), distinct weights, two of them negative.
    let base_ids: [i32; 8] = [3, 7, 3, 11, 19, 0, 25, 7];
    let topk_idx: Vec<i32> = (0..k_top)
        .map(|t| base_ids[t % 8] % n_exp as i32)
        .collect();
    let base_w: [f32; 8] = [0.31, -0.17, 0.44, 0.09, 0.22, 0.63, -0.05, 0.28];
    let topk_w: Vec<f32> = (0..k_top).map(|t| base_w[t % 8]).collect();

    // Each krank gets its OWN activation slice, with clearly different content.
    let xs: Vec<Vec<f32>> = (0..k_top)
        .map(|t| {
            (0..k)
                .map(|i| {
                    let h = mix(seed ^ 0xac71 ^ ((t as u64) << 40) ^ i as u64);
                    ((h % 2001) as f32 - 1000.0) * 1e-3
                })
                .collect()
        })
        .collect();

    // Non-zero, row-varying seed residual: a store-instead-of-accumulate
    // kernel cannot pass the reference check.
    let seed_res: Vec<f32> = (0..m)
        .map(|i| ((i % 37) as f32 - 18.0) * 0.011 + 0.037)
        .collect();

    let mut gl_host = Vec::with_capacity(n_exp);
    let mut l_host = Vec::with_capacity(n_exp);
    for e in 0..n_exp {
        let codes = gen_codes(m, k, seed ^ (0x1000 + e as u64));
        let scales = gen_scales(m, k, seed ^ (0x2000 + e as u64), pow2);
        gl_host.push(build_mq3gl(m, k, &codes, &scales));
        l_host.push(build_mq3l(m, k, &codes, &scales, &cb));
    }

    let mut keep = Vec::new();
    let mut gl_ptrs = Vec::with_capacity(n_exp);
    let mut l_ptrs = Vec::with_capacity(n_exp);
    for e in 0..n_exp {
        let tg = gpu.upload_raw(&gl_host[e], &[gl_host[e].len()]).unwrap();
        let tl = gpu.upload_raw(&l_host[e], &[l_host[e].len()]).unwrap();
        gl_ptrs.push(tg.buf.as_ptr() as u64);
        l_ptrs.push(tl.buf.as_ptr() as u64);
        keep.push(tg);
        keep.push(tl);
    }
    let gl_ptr_t = gpu
        .upload_raw(
            &gl_ptrs.iter().flat_map(|p| p.to_le_bytes()).collect::<Vec<u8>>(),
            &[n_exp],
        )
        .unwrap();
    let l_ptr_t = gpu
        .upload_raw(
            &l_ptrs.iter().flat_map(|p| p.to_le_bytes()).collect::<Vec<u8>>(),
            &[n_exp],
        )
        .unwrap();
    let idx_t = gpu
        .upload_raw(
            &topk_idx.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>(),
            &[k_top],
        )
        .unwrap();
    let w_t = gpu.upload_f32(&topk_w, &[k_top]).unwrap();
    let rot: Vec<f32> = xs.iter().flat_map(|v| v.iter().copied()).collect();
    let rot_t = gpu.upload_f32(&rot, &[k_top * k]).unwrap();
    let y = gpu.upload_f32(&seed_res, &[m]).unwrap();

    let mut bg = KernargBlob::new();
    bg.push_ptr(gl_ptr_t.buf.as_ptr() as *const _);
    bg.push_ptr(idx_t.buf.as_ptr() as *const _);
    bg.push_ptr(w_t.buf.as_ptr() as *const _);
    bg.push_ptr(rot_t.buf.as_ptr() as *const _);
    bg.push_ptr(y.buf.as_ptr() as *const _);
    for c in cb {
        bg.push_f32(c);
    }
    bg.push_i32(m as i32);
    bg.push_i32(k as i32);
    let gl_args = bg.into_vec();

    let mut bl = KernargBlob::new();
    bl.push_ptr(l_ptr_t.buf.as_ptr() as *const _);
    bl.push_ptr(idx_t.buf.as_ptr() as *const _);
    bl.push_ptr(w_t.buf.as_ptr() as *const _);
    bl.push_ptr(rot_t.buf.as_ptr() as *const _);
    bl.push_ptr(y.buf.as_ptr() as *const _);
    bl.push_i32(m as i32);
    bl.push_i32(k as i32);
    let l_args = bl.into_vec();

    keep.push(gl_ptr_t);
    keep.push(l_ptr_t);
    keep.push(idx_t);
    keep.push(w_t);
    keep.push(rot_t);

    println!(
        "  case {label}: M={m} K={k} gpr={gpr} k_top={k_top} n_exp={n_exp} scales={}",
        if pow2 { "pow2 (cross-format exact)" } else { "arbitrary fp16" }
    );
    println!(
        "    quad-loop iters={} tail groups={}   topk_idx={:?}  (dupes exercise atomic self-contention)",
        gpr >> 2,
        gpr & 3,
        topk_idx
    );
    println!(
        "    per-expert bytes: MQ3GL {} KiB   MQ3L {} KiB   ({:.2}% less)",
        m * gpr * GL_BYTES_PER_GROUP / 1024,
        m * gpr * L_BYTES_PER_GROUP / 1024,
        100.0 * (1.0 - GL_BYTES_PER_GROUP as f64 / L_BYTES_PER_GROUP as f64)
    );

    Case {
        m,
        k,
        k_top,
        n_exp,
        label: label.to_string(),
        topk_idx,
        topk_w,
        xs,
        seed_res,
        gl_host,
        l_host,
        cb,
        gl_args,
        l_args,
        y,
        _keep: keep,
    }
}

/// Independent CPU reference for the fused down GEMV + scaled residual:
///
///   y[row] = seed[row] + Σ_t topk_w[t] · Σ_col W_{topk_idx[t]}[row][col] · x_t[col]
///
/// with W dequantized from the blob bytes per the format spec. f64 throughout.
fn cpu_reference(case: &Case, gl: bool) -> Vec<f64> {
    let mut y: Vec<f64> = case.seed_res.iter().map(|v| *v as f64).collect();
    let mut w = vec![0f32; case.k];
    for t in 0..case.k_top {
        let e = case.topk_idx[t] as usize;
        let scale = case.topk_w[t] as f64;
        let x = &case.xs[t];
        for row in 0..case.m {
            if gl {
                decode_row_gl(&case.gl_host[e], case.m, case.k, row, &case.cb, &mut w);
            } else {
                decode_row_l(&case.l_host[e], case.k, row, &mut w);
            }
            let mut acc = 0f64;
            for col in 0..case.k {
                acc += w[col] as f64 * x[col] as f64;
            }
            y[row] += scale * acc;
        }
    }
    y
}

struct Stats {
    max_abs: f64,
    max_abs_idx: usize,
    max_rel: f64,
    max_rel_idx: usize,
    max_norm: f64,
    ref_mag: f64,
    skipped: usize,
}

/// `max_rel` is the classic |err|/|want|, but only over rows whose magnitude is
/// at least 1% of the mean output magnitude — a near-zero row makes a relative
/// error meaningless. `max_norm` = max|err| / mean|want| covers every row,
/// including the skipped ones, so nothing hides in the exclusion.
fn compare(got: &[f32], want: &[f64]) -> Stats {
    let ref_mag = want.iter().map(|v| v.abs()).sum::<f64>() / want.len() as f64;
    let floor = ref_mag * 1e-2;
    let mut s = Stats {
        max_abs: 0.0,
        max_abs_idx: 0,
        max_rel: 0.0,
        max_rel_idx: 0,
        max_norm: 0.0,
        ref_mag,
        skipped: 0,
    };
    for (i, (&g, &w)) in got.iter().zip(want.iter()).enumerate() {
        let e = (g as f64 - w).abs();
        if e > s.max_abs {
            s.max_abs = e;
            s.max_abs_idx = i;
        }
        let n = if ref_mag > 0.0 { e / ref_mag } else { e };
        if n > s.max_norm {
            s.max_norm = n;
        }
        if w.abs() >= floor {
            let r = e / w.abs();
            if r > s.max_rel {
                s.max_rel = r;
                s.max_rel_idx = i;
            }
        } else {
            s.skipped += 1;
        }
    }
    s
}

const TOL: f64 = 1e-4;

fn report(label: &str, s: &Stats, got: &[f32], want: &[f64]) -> bool {
    let pass = s.max_rel <= TOL && s.max_norm <= TOL;
    println!(
        "    {label:<22} max_abs={:.3e} @row {}   max_rel={:.3e} @row {}   max_norm={:.3e}",
        s.max_abs, s.max_abs_idx, s.max_rel, s.max_rel_idx, s.max_norm
    );
    println!(
        "    {:<22} mean|y|={:.4e}  worst row: gpu={:.6e} cpu={:.6e}  ({} near-zero rows excluded from max_rel)",
        "",
        s.ref_mag,
        got[s.max_abs_idx],
        want[s.max_abs_idx],
        s.skipped
    );
    println!(
        "    {:<22} [{}]  tolerance {:.0e} relative",
        "",
        if pass { "PASS" } else { "FAIL" },
        TOL
    );
    pass
}

/// Re-seed the residual, launch once, read back.
fn run_once(gpu: &mut Gpu, case: &Case, name: &str, args: &mut [u8]) -> Vec<f32> {
    let bytes = unsafe {
        std::slice::from_raw_parts(case.seed_res.as_ptr() as *const u8, case.seed_res.len() * 4)
    };
    gpu.hip.memcpy_htod(&case.y.buf, bytes).unwrap();
    gpu.launch_kernel_blob(name, case.grid(), [32, 1, 1], 0, args)
        .unwrap();
    gpu.hip.device_synchronize().unwrap();
    gpu.download_f32(&case.y).unwrap()
}

/// One timed burst: `iters` launches back to back, returns µs/launch.
fn burst(gpu: &Gpu, name: &str, grid: [u32; 3], args: &mut [u8], iters: usize) -> f64 {
    let t0 = Instant::now();
    for _ in 0..iters {
        gpu.launch_kernel_blob(name, grid, [32, 1, 1], 0, args)
            .unwrap();
    }
    gpu.hip.device_synchronize().unwrap();
    t0.elapsed().as_secs_f64() * 1e6 / iters as f64
}

fn min_median(v: &mut [f64]) -> (f64, f64) {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    (v[0], v[v.len() / 2])
}

fn main() {
    let mut gpu = Gpu::init().expect("GPU init");
    println!("bench_mq3g256gl_moe_down — arch={}", gpu.arch);
    println!(
        "REGIME: HIP-dispatch microbenchmark = TRIAGE FILTER for gross defects, NOT a kernel verdict."
    );
    println!(
        "        Host launch latency masks device effects; retained PM4 replay can rank differently."
    );
    println!(
        "        A prior HIP microbench put MQ2GL at -3.1% and nearly killed the format. Final"
    );
    println!(
        "        acceptance = golden bundle (registry/redline-golden-v1.json), HIP + PM4 arms.\n"
    );

    packing_self_test();

    gpu.ensure_kernel_public("bench_mq3g256gl_moe_down_indexed", MQ3GL_SRC, GL_FN)
        .expect("JIT MQ3GL down");
    gpu.ensure_kernel_public("bench_mq3g256_lloyd_moe_down_indexed", MQ3L_SRC, L_FN)
        .expect("JIT MQ3L down");

    let mut all_pass = true;

    // ── case A: the real a3b down decode shape (timed) ─────────────────────
    // K=512 → gpr=2 → the kernels' quad loop does ZERO iterations and both
    // groups go through the TAIL path. That is exactly what production runs,
    // and it is why case B exists.
    println!("\n== correctness ==");
    let a = build_case(&mut gpu, "A/down", 2048, 512, 8, 32, 0xa3b0, false);
    let want_gl = cpu_reference(&a, true);
    let want_l = cpu_reference(&a, false);

    let mut gl_args = a.gl_args.clone();
    let mut l_args = a.l_args.clone();
    let got_gl = run_once(&mut gpu, &a, GL_FN, &mut gl_args);
    let got_l = run_once(&mut gpu, &a, L_FN, &mut l_args);
    all_pass &= report("A MQ3GL vs CPU", &compare(&got_gl, &want_gl), &got_gl, &want_gl);
    all_pass &= report("A MQ3L  vs CPU", &compare(&got_l, &want_l), &got_l, &want_l);

    // Atomic-order non-determinism band: k_top blocks atomicAdd into the same
    // cell, so the FP add order across experts is not fixed. Quantify it — a
    // band far above f32 rounding would mean lost or doubled contributions.
    let mut band = 0f64;
    let ref_run = run_once(&mut gpu, &a, GL_FN, &mut gl_args);
    for _ in 0..4 {
        let r = run_once(&mut gpu, &a, GL_FN, &mut gl_args);
        for (x, y) in r.iter().zip(ref_run.iter()) {
            band = band.max((*x as f64 - *y as f64).abs());
        }
    }
    let mag = want_gl.iter().map(|v| v.abs()).sum::<f64>() / want_gl.len() as f64;
    println!(
        "    {:<22} atomic reorder band over 5 identical launches = {:.3e} ({:.2e} of mean|y|)",
        "A MQ3GL", band, band / mag
    );

    // Seed pass-through: y(seed) - y(0) must reproduce the seed exactly, i.e.
    // the kernel accumulates into x_residual rather than overwriting it.
    gpu.fill_f32(&a.y, 0.0).unwrap();
    gpu.launch_kernel_blob(GL_FN, a.grid(), [32, 1, 1], 0, &mut gl_args)
        .unwrap();
    gpu.hip.device_synchronize().unwrap();
    let zero_run = gpu.download_f32(&a.y).unwrap();
    let mut seed_dev = 0f64;
    for i in 0..a.m {
        let d = (ref_run[i] as f64 - zero_run[i] as f64) - a.seed_res[i] as f64;
        seed_dev = seed_dev.max(d.abs());
    }
    let seed_pass = seed_dev <= band + 1e-4 * mag;
    all_pass &= seed_pass;
    println!(
        "    {:<22} seed pass-through dev = {:.3e}  [{}]  (store-instead-of-accumulate would be ~{:.2e})",
        "A residual",
        seed_dev,
        if seed_pass { "PASS" } else { "FAIL" },
        a.seed_res.iter().fold(0f32, |m, v| m.max(v.abs()))
    );

    // ── case B: quad loop + tail, and a strict cross-format check ──────────
    // K=1280 → gpr=5 → 1 quad-loop iteration + 1 tail group, so the main
    // no-barrier K4 loop that case A never touches is covered here. pow2
    // scales make the two blobs dequantize to bit-identical weights, so the
    // two kernels must agree to f32 accumulation-order noise.
    println!();
    let b = build_case(&mut gpu, "B/quads+tail", 256, 1280, 8, 8, 0xb17e, true);
    let bwant_gl = cpu_reference(&b, true);
    let bwant_l = cpu_reference(&b, false);
    let mut bgl_args = b.gl_args.clone();
    let mut bl_args = b.l_args.clone();
    let bgot_gl = run_once(&mut gpu, &b, GL_FN, &mut bgl_args);
    let bgot_l = run_once(&mut gpu, &b, L_FN, &mut bl_args);
    all_pass &= report(
        "B MQ3GL vs CPU",
        &compare(&bgot_gl, &bwant_gl),
        &bgot_gl,
        &bwant_gl,
    );
    all_pass &= report("B MQ3L  vs CPU", &compare(&bgot_l, &bwant_l), &bgot_l, &bwant_l);
    let cross: Vec<f64> = bgot_l.iter().map(|v| *v as f64).collect();
    all_pass &= report(
        "B MQ3GL vs MQ3L",
        &compare(&bgot_gl, &cross),
        &bgot_gl,
        &cross,
    );

    // Case A cross-format is informational only: with arbitrary fp16 scales
    // the MQ3-Lloyd header stores fp16(scale·cb), which the truncating fp16
    // converter above puts up to one full fp16 ulp (2^-10 ≈ 1e-3 relative,
    // always toward zero) below the GL kernel's f32 scale·cb. Measured on the
    // host at 1.1e-5 absolute on ~8.6e-3 weights. This is a property of the
    // bench's synthetic blob construction, NOT of either kernel — hence not
    // gated. Case B (pow2 scales) is the gated cross-format check.
    let across: Vec<f64> = got_l.iter().map(|v| *v as f64).collect();
    let sa = compare(&got_gl, &across);
    println!(
        "    {:<22} max_rel={:.3e} (INFORMATIONAL — fp16(scale·cb) truncation, expect <~1e-3; not gated)",
        "A MQ3GL vs MQ3L", sa.max_rel
    );

    // ── timing (case A only: the production down shape) ────────────────────
    println!("\n== timing (case A: M={} K={} k_top={}) ==", a.m, a.k, a.k_top);
    println!("   first pass of any (kernel × shape) cell is JIT-contaminated — a warmup burst runs");
    println!("   first and is discarded; reported numbers are from the bursts after it.");
    let warm = 32usize;
    let per = 100usize;
    let reps = 9usize;
    let _ = burst(&gpu, GL_FN, a.grid(), &mut gl_args, warm);
    let _ = burst(&gpu, L_FN, a.grid(), &mut l_args, warm);
    let mut t_gl = Vec::with_capacity(reps);
    let mut t_l = Vec::with_capacity(reps);
    for _ in 0..reps {
        t_gl.push(burst(&gpu, GL_FN, a.grid(), &mut gl_args, per));
        t_l.push(burst(&gpu, L_FN, a.grid(), &mut l_args, per));
    }
    let (gl_min, gl_med) = min_median(&mut t_gl);
    let (l_min, l_med) = min_median(&mut t_l);

    let elems = (a.k_top * a.m * a.k) as f64;
    let gb = |bytes: f64, us: f64| bytes / (us * 1e-6) / 1e9;
    println!(
        "\n   {:<8} {:>10} {:>10} {:>12} {:>12} {:>10} {:>8}",
        "format", "min us", "med us", "GB/s (min)", "GB/s (med)", "B/elem", "bpw"
    );
    println!(
        "   {:<8} {:>10.2} {:>10.2} {:>12.1} {:>12.1} {:>10.4} {:>8.4}",
        "MQ3GL",
        gl_min,
        gl_med,
        gb(a.gl_bytes(), gl_min),
        gb(a.gl_bytes(), gl_med),
        a.gl_bytes() / elems,
        8.0 * a.gl_bytes() / elems
    );
    println!(
        "   {:<8} {:>10.2} {:>10.2} {:>12.1} {:>12.1} {:>10.4} {:>8.4}",
        "MQ3L",
        l_min,
        l_med,
        gb(a.l_bytes(), l_min),
        gb(a.l_bytes(), l_med),
        a.l_bytes() / elems,
        8.0 * a.l_bytes() / elems
    );
    println!(
        "   weight bytes/launch: MQ3GL {:.0}  MQ3L {:.0}   ({:.2}% fewer)",
        a.gl_bytes(),
        a.l_bytes(),
        100.0 * (1.0 - a.gl_bytes() / a.l_bytes())
    );
    println!(
        "   activation traffic (not counted above): {} unique x bytes/launch, re-read by all {} rows",
        a.k_top * a.k * 4,
        a.m
    );
    println!(
        "   MQ3GL vs MQ3L: {:+.2}% on median time, {:+.2}% on min time  ({} reps × {} launches, {} n_exp)",
        100.0 * (gl_med / l_med - 1.0),
        100.0 * (gl_min / l_min - 1.0),
        reps,
        per,
        a.n_exp
    );
    println!("   {} / {}", a.label, b.label);

    println!(
        "\n=== {} ===",
        if all_pass {
            "PASS — all gated correctness checks within 1e-4 relative"
        } else {
            "FAIL — at least one gated correctness check exceeded 1e-4 relative"
        }
    );
    if !all_pass {
        std::process::exit(1);
    }
}
