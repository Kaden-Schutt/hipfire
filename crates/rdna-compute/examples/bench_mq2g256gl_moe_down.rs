// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
//! MQ2-GL fused MoE **down** GEMV (`gemv_mq2g256gl_moe_down_residual_scaled_k8_indexed`)
//! — correctness against an independent CPU reference, plus a speed comparison
//! against the MQ2-Lloyd sibling
//! (`gemv_mq2g256_lloyd_moe_down_residual_scaled_k8_indexed`) on the a3b routed
//! expert down shape M=2048, K=512, k_top=8.
//!
//! Both kernels are JIT'd from source and launched via the kernarg-blob path
//! (`ensure_kernel_public` + `launch_kernel_blob`), so this bench touches no
//! dispatch plumbing, no kernel, and no shared file.
//!
//! ## What makes this kernel dangerous to test naively
//!
//! It is **atomic self-combining**. There is no expanded per-expert output and
//! no separate down-combine kernel: lane 0 of every (row, krank) block does
//!
//! ```text
//!     atomicAdd(&x_residual[row], topk_weights[krank] * acc)
//! ```
//!
//! so the observable output is
//!
//! ```text
//!     x_residual[row] += Σ_{krank=0..k_top-1} tw[krank] · Σ_c W_{e(krank)}[row,c] · x[krank·K + c]
//! ```
//!
//! A kernel that decodes ONE expert perfectly but mis-accumulates across k_top
//! (wrong `blockIdx.y` fan-out, `topk_weights[expert_id]` instead of
//! `topk_weights[krank]`, a shared `x` slice instead of `rot_batch + krank*K`,
//! a store instead of an atomicAdd) would sail through a single-expert test and
//! silently corrupt every MoE layer. So this bench:
//!   * verifies the TOTAL over all 8 routed experts, not one expert;
//!   * uses DISTINCT weights per expert, DISTINCT x per krank, and DISTINCT,
//!     non-uniform topk_weights, so any of the four mixups above changes the
//!     answer;
//!   * walks a cumulative-rank ladder (grid.y = 1,2,…,k_top) so a mis-fan-out
//!     is localized to the rank where the ladder diverges;
//!   * PRE-SEEDS x_residual nonzero and confirms `out = seed + Δ` (a store
//!     would drop the seed), then launches twice and confirms `out = seed + 2Δ`
//!     (proves accumulate, not overwrite);
//!   * sweeps K ∈ {512, 1024, 1280, 1792, 2048} because the mandated a3b down
//!     shape K=512 has groups_per_row = 2 → `quads = 0`: the K4 main loop never
//!     runs at the shipping shape, only the `tail >= 2` epilogue. The sweep
//!     covers quads>0 and all three tail epilogues (0/1/2/3).
//!
//! ## CPU reference independence
//!
//! The reference decodes the raw uploaded bytes from the FORMAT SPEC, not from
//! the kernel's arithmetic structure:
//!   * MQ2GL (qt=38), SoA: `[0 .. M·gpr·64)` packed 2-bit indices (4 codes per
//!     byte, little-endian: byte b of group g carries columns g·256+4b+0..3 in
//!     bit fields [0:2],[2:4],[4:6],[6:8]), then `[.. +M·gpr·2)` fp16 per-block
//!     scales. Both regions row-major in (row, group). Codebook = the 4 scalar
//!     kernel args. Reconstruction w = scale_block · cb[q]. 2.0625 bpw.
//!   * MQ2L (qt=19), interleaved: 72 B per group-of-256 = 4×fp16 codebook (8 B)
//!     then 64 B of 2-bit indices; w = cb_block[q]. 2.25 bpw.
//! The reference walks columns in plain 0..K order and accumulates in f64; it
//! never mirrors the kernel's `tid*8` ownership, K4 partition, or shuffle tree.
//!
//! Synthetic weights are built in the exact on-disk byte layout of each format.
//! The GL and Lloyd blobs carry the SAME codes and the SAME per-block scales
//! (the Lloyd per-block codebook is fp16(cb[e]·scale)), so the two kernels are
//! solving the same problem to within fp16 codebook rounding — the speed
//! comparison is apples-to-apples and a cross-format agreement number is
//! printed as a soft signal. Each kernel is still gated against its OWN
//! reference.
//!
//! `x` (rot_batch) is nominally FWHT-256 pre-rotated by the caller. For a
//! microbenchmark that is irrelevant to the arithmetic, so arbitrary
//! deterministic `x` is fed — the SAME bytes to the GPU and to the CPU
//! reference. Stated explicitly so nobody reads the numbers as an end-to-end
//! quality claim.
//!
//! ## REGIME CAVEAT — read before quoting any number from this file
//!
//! This is a HIP-dispatch microbenchmark. It is a TRIAGE FILTER for gross
//! defects (wrong numbers, pathological slowness), NOT a verdict on a kernel.
//! Host launch latency masks device-level effects, and the same kernel can
//! measure differently once lowered to retained PM4 replay. A prior HIP
//! microbench measured the MQ2GL kernel at -3.1% and that was nearly used to
//! kill the format. Final acceptance is the golden bundle
//! (`registry/redline-golden-v1.json`) with HIP and PM4 arms — not this file.
//! The first timed pass is additionally JIT-contaminated; only warm passes are
//! reported, and even those are host-launch-bound at this shape.
//!
//! Run: cargo run --release -p rdna-compute --example bench_mq2g256gl_moe_down

use hip_bridge::KernargBlob;
use rdna_compute::{Gpu, GpuTensor};
use std::time::Instant;

const GL_SRC: &str = include_str!("../../../kernels/src/gemv_mq2g256gl_moe_down_indexed.hip");
const L_SRC: &str = include_str!("../../../kernels/src/gemv_mq2g256_lloyd_moe_down_indexed.hip");

const GL_MOD: &str = "gemv_mq2g256gl_moe_down_indexed";
const L_MOD: &str = "gemv_mq2g256_lloyd_moe_down_indexed";
const GL_FN: &str = "gemv_mq2g256gl_moe_down_residual_scaled_k8_indexed";
const L_FN: &str = "gemv_mq2g256_lloyd_moe_down_residual_scaled_k8_indexed";

/// Canonical GL_CB2 global codebook (textbook Lloyd–Max levels for a unit
/// Gaussian, 2 bit). Passed to the GL kernel as four scalar float args.
const CB: [f32; 4] = [-1.5104, -0.4528, 0.4528, 1.5104];

/// Relative tolerance for the f32 accumulation at this size.
const TOL: f64 = 1e-4;

/// Transformer layers assumed when projecting the microbench into token-shaped
/// terms. Label only — this kernel runs once per MoE layer per token.
const LAYERS: usize = 48;

// ─────────────────────────── host bit twiddling ───────────────────────────

fn mix(x: u64) -> u64 {
    let h = x
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    ((h ^ (h >> 33)).wrapping_mul(0xff51afd7ed558ccd)) ^ (h >> 28)
}

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

fn put_half(dst: &mut [u8], off: usize, v: f32) {
    let h = half_bits(v);
    dst[off] = (h & 0xff) as u8;
    dst[off + 1] = (h >> 8) as u8;
}

fn get_half(src: &[u8], off: usize) -> f32 {
    half_to_f32(u16::from_le_bytes([src[off], src[off + 1]]))
}

// ───────────────────── synthetic weights (on-disk layout) ─────────────────
//
// Both formats draw the SAME index bytes and the SAME per-block scales from
// these two generators, so the two encodings represent the same weights.

fn idx_byte(seed: u64, row: usize, g: usize, b: usize) -> u8 {
    (mix(seed ^ ((row as u64) << 32) ^ ((g as u64) << 12) ^ b as u64) & 0xff) as u8
}

fn blk_scale(seed: u64, row: usize, g: usize) -> f32 {
    0.004f32 + ((mix(seed ^ ((row as u64) << 20) ^ (g as u64 + 7)) % 4000) as f32) * 1e-6
}

/// MQ2GL (qt=38) SoA: `[M·gpr·64 index bytes][M·gpr fp16 scales]`.
fn build_gl(m: usize, k: usize, seed: u64) -> Vec<u8> {
    let gpr = k / 256;
    let idx_bytes = m * gpr * 64;
    let mut out = vec![0u8; idx_bytes + m * gpr * 2];
    for row in 0..m {
        for g in 0..gpr {
            let off = (row * gpr + g) * 64;
            for b in 0..64 {
                out[off + b] = idx_byte(seed, row, g, b);
            }
            put_half(&mut out, idx_bytes + (row * gpr + g) * 2, blk_scale(seed, row, g));
        }
    }
    out
}

/// MQ2-Lloyd (qt=19) interleaved: `[4×fp16 cb][64 B indices]` per group,
/// 72 B stride, row-major in (row, group).
fn build_l(m: usize, k: usize, seed: u64) -> Vec<u8> {
    let gpr = k / 256;
    let mut out = vec![0u8; m * gpr * 72];
    for row in 0..m {
        for g in 0..gpr {
            let off = (row * gpr + g) * 72;
            let s = blk_scale(seed, row, g);
            for (e, &c) in CB.iter().enumerate() {
                put_half(&mut out, off + 2 * e, c * s);
            }
            for b in 0..64 {
                out[off + 8 + b] = idx_byte(seed, row, g, b);
            }
        }
    }
    out
}

// ───────────────── independent decoders (derived from the spec) ───────────

/// Decode row `row` of an MQ2GL blob to K dequantized f32 weights.
fn decode_row_gl(raw: &[u8], m: usize, k: usize, row: usize, out: &mut [f32]) {
    let gpr = k / 256;
    let idx_bytes = m * gpr * 64;
    for g in 0..gpr {
        let scale = get_half(raw, idx_bytes + (row * gpr + g) * 2);
        let base = (row * gpr + g) * 64;
        for b in 0..64 {
            let byte = raw[base + b];
            for j in 0..4 {
                let q = ((byte >> (2 * j)) & 3) as usize;
                out[g * 256 + b * 4 + j] = scale * CB[q];
            }
        }
    }
}

/// Decode row `row` of an MQ2-Lloyd blob to K dequantized f32 weights.
fn decode_row_l(raw: &[u8], k: usize, row: usize, out: &mut [f32]) {
    let gpr = k / 256;
    for g in 0..gpr {
        let off = (row * gpr + g) * 72;
        let mut cb = [0f32; 4];
        for (e, ci) in cb.iter_mut().enumerate() {
            *ci = get_half(raw, off + 2 * e);
        }
        for b in 0..64 {
            let byte = raw[off + 8 + b];
            for j in 0..4 {
                let q = ((byte >> (2 * j)) & 3) as usize;
                out[g * 256 + b * 4 + j] = cb[q];
            }
        }
    }
}

// ───────────────────────────── the test case ──────────────────────────────

struct Case {
    m: usize,
    k: usize,
    k_top: usize,
    n_exp: usize,
    gpr: usize,
    topk: Vec<i32>,
    tw: Vec<f32>,
    x: Vec<f32>,
    seed: Vec<f32>,
    /// Raw bytes of the k_top ROUTED experts, in krank order.
    raw_gl: Vec<Vec<u8>>,
    raw_l: Vec<Vec<u8>>,
    resid: GpuTensor,
    blob_gl: Vec<u8>,
    blob_l: Vec<u8>,
    #[allow(dead_code)]
    keep: Vec<GpuTensor>,
}

fn build_case(gpu: &mut Gpu, m: usize, k: usize, k_top: usize, n_exp: usize, sb: u64) -> Case {
    let gpr = k / 256;
    assert_eq!(k % 256, 0, "K must be a multiple of 256");

    // Routing: distinct experts, spread across the pool.
    let topk: Vec<i32> = (0..k_top).map(|r| ((r * 37 + 5) % n_exp) as i32).collect();
    for a in 0..k_top {
        for b in (a + 1)..k_top {
            assert_ne!(topk[a], topk[b], "routed experts must be distinct");
        }
    }
    // Non-uniform, distinct, sums to 1 — a topk_weights[expert_id] mixup shows.
    let raw_w: Vec<f32> = (0..k_top).map(|r| 1.0 / (r as f32 + 2.0)).collect();
    let wsum: f32 = raw_w.iter().sum();
    let tw: Vec<f32> = raw_w.iter().map(|v| v / wsum).collect();

    // One distinct activation slice per krank — a shared-x bug shows.
    let x: Vec<f32> = (0..k_top * k)
        .map(|i| {
            let r = i / k;
            let c = i % k;
            ((mix(0xABCD ^ ((r as u64) << 40) ^ c as u64) % 1201) as f32 - 600.0) * 1e-3
        })
        .collect();

    // Nonzero pre-seed — a store-instead-of-atomicAdd bug shows.
    let seed: Vec<f32> = (0..m).map(|row| ((row % 17) as f32 - 8.0) * 0.005).collect();

    // Upload every expert of both formats.
    let mut gl_t = Vec::with_capacity(n_exp);
    let mut l_t = Vec::with_capacity(n_exp);
    for e in 0..n_exp {
        let g = build_gl(m, k, sb + e as u64);
        let l = build_l(m, k, sb + e as u64);
        gl_t.push(gpu.upload_raw(&g, &[g.len()]).expect("upload gl"));
        l_t.push(gpu.upload_raw(&l, &[l.len()]).expect("upload l"));
    }
    let gl_ptrs: Vec<u8> = gl_t
        .iter()
        .flat_map(|t| (t.buf.as_ptr() as u64).to_le_bytes())
        .collect();
    let l_ptrs: Vec<u8> = l_t
        .iter()
        .flat_map(|t| (t.buf.as_ptr() as u64).to_le_bytes())
        .collect();
    let gl_ptr_t = gpu.upload_raw(&gl_ptrs, &[n_exp]).expect("gl ptr table");
    let l_ptr_t = gpu.upload_raw(&l_ptrs, &[n_exp]).expect("l ptr table");

    let topk_bytes: Vec<u8> = topk.iter().flat_map(|v| v.to_le_bytes()).collect();
    let topk_t = gpu.upload_raw(&topk_bytes, &[k_top]).expect("topk");
    let tw_t = gpu.upload_f32(&tw, &[k_top]).expect("tw");
    let x_t = gpu.upload_f32(&x, &[k_top * k]).expect("x");
    let resid = gpu.upload_f32(&seed, &[m]).expect("resid");

    let mut b_gl = KernargBlob::new();
    b_gl.push_ptr(gl_ptr_t.buf.as_ptr() as *const _);
    b_gl.push_ptr(topk_t.buf.as_ptr() as *const _);
    b_gl.push_ptr(tw_t.buf.as_ptr() as *const _);
    b_gl.push_ptr(x_t.buf.as_ptr() as *const _);
    b_gl.push_ptr(resid.buf.as_ptr() as *const _);
    for c in CB {
        b_gl.push_f32(c);
    }
    b_gl.push_i32(m as i32);
    b_gl.push_i32(k as i32);

    let mut b_l = KernargBlob::new();
    b_l.push_ptr(l_ptr_t.buf.as_ptr() as *const _);
    b_l.push_ptr(topk_t.buf.as_ptr() as *const _);
    b_l.push_ptr(tw_t.buf.as_ptr() as *const _);
    b_l.push_ptr(x_t.buf.as_ptr() as *const _);
    b_l.push_ptr(resid.buf.as_ptr() as *const _);
    b_l.push_i32(m as i32);
    b_l.push_i32(k as i32);

    let raw_gl = topk.iter().map(|&e| build_gl(m, k, sb + e as u64)).collect();
    let raw_l = topk.iter().map(|&e| build_l(m, k, sb + e as u64)).collect();

    let mut keep = Vec::new();
    keep.extend(gl_t);
    keep.extend(l_t);
    keep.push(gl_ptr_t);
    keep.push(l_ptr_t);
    keep.push(topk_t);
    keep.push(tw_t);
    keep.push(x_t);

    Case {
        m,
        k,
        k_top,
        n_exp,
        gpr,
        topk,
        tw,
        x,
        seed,
        raw_gl,
        raw_l,
        resid,
        blob_gl: b_gl.into_vec(),
        blob_l: b_l.into_vec(),
        keep,
    }
}

impl Case {
    /// Weight bytes touched by ONE launch (k_top experts × M rows).
    fn weight_bytes(&self, gl: bool) -> f64 {
        let per_group = if gl { 64 + 2 } else { 72 };
        (self.k_top * self.m * self.gpr * per_group) as f64
    }

    /// Independent CPU reference for the ATOMIC SELF-COMBINING semantics:
    ///
    ///   out[row] = (seed[row] if seeded) + Σ_{r<ranks} tw[r]·Σ_c W_{e(r)}[row,c]·x[r·K+c]
    ///
    /// Returns (expected output, Σ|term| per row) — the latter is the
    /// conditioning of the f32 sum the GPU performs, used to normalize error.
    fn reference(&self, gl: bool, ranks: usize, seeded: bool, repeats: f64) -> (Vec<f64>, Vec<f64>) {
        let mut out = vec![0.0f64; self.m];
        if seeded {
            for (o, s) in out.iter_mut().zip(&self.seed) {
                *o = *s as f64;
            }
        }
        let mut absum = vec![0.0f64; self.m];
        let mut wrow = vec![0.0f32; self.k];
        for r in 0..ranks {
            let raw = if gl { &self.raw_gl[r] } else { &self.raw_l[r] };
            let w = self.tw[r] as f64;
            let xs = &self.x[r * self.k..(r + 1) * self.k];
            for row in 0..self.m {
                if gl {
                    decode_row_gl(raw, self.m, self.k, row, &mut wrow);
                } else {
                    decode_row_l(raw, self.k, row, &mut wrow);
                }
                let mut dot = 0.0f64;
                let mut mag = 0.0f64;
                for c in 0..self.k {
                    let t = wrow[c] as f64 * xs[c] as f64;
                    dot += t;
                    mag += t.abs();
                }
                out[row] += repeats * w * dot;
                absum[row] += repeats * w * mag;
            }
        }
        (out, absum)
    }
}

// ─────────────────────────────── error stats ──────────────────────────────

struct Stats {
    max_abs: f64,
    abs_idx: usize,
    /// max_i |got-want| / max(|want_i|, rms(want)) — the guard denominator
    /// keeps a near-zero cancelled entry from manufacturing a fake failure.
    max_rel: f64,
    rel_idx: usize,
    /// max_abs / mean Σ|term| — error relative to the magnitude actually
    /// accumulated in f32 (the honest conditioning-aware number).
    rel_terms: f64,
    rms: f64,
}

fn stats(got: &[f32], want: &[f64], absum: &[f64]) -> Stats {
    let n = want.len();
    let rms = (want.iter().map(|v| v * v).sum::<f64>() / n as f64).sqrt();
    let mean_terms = (absum.iter().sum::<f64>() / n as f64).max(1e-30);
    let mut s = Stats {
        max_abs: 0.0,
        abs_idx: 0,
        max_rel: 0.0,
        rel_idx: 0,
        rel_terms: 0.0,
        rms,
    };
    for i in 0..n {
        let d = (got[i] as f64 - want[i]).abs();
        if d > s.max_abs {
            s.max_abs = d;
            s.abs_idx = i;
        }
        let rel = d / want[i].abs().max(rms).max(1e-30);
        if rel > s.max_rel {
            s.max_rel = rel;
            s.rel_idx = i;
        }
    }
    s.rel_terms = s.max_abs / mean_terms;
    s
}

fn report(label: &str, s: &Stats, got: &[f32], want: &[f64]) -> bool {
    let ok = s.max_rel <= TOL && s.max_abs.is_finite();
    eprintln!(
        "  {:<34} max|abs|={:.3e} @row {:<5} max rel={:.3e} @row {:<5} (rel/Σ|term|={:.2e}, rms={:.4})",
        label, s.max_abs, s.abs_idx, s.max_rel, s.rel_idx, s.rel_terms, s.rms
    );
    eprintln!(
        "  {:<34} worst row: gpu={:+.8e}  cpu={:+.8e}   [{}]",
        "", got[s.rel_idx], want[s.rel_idx],
        if ok { "PASS" } else { "FAIL" }
    );
    ok
}

// ─────────────────────────────── GPU helpers ──────────────────────────────

fn write_f32(gpu: &Gpu, t: &GpuTensor, v: &[f32]) {
    let bytes = unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 4) };
    gpu.memcpy_htod_auto(&t.buf, bytes).expect("seed residual");
}

/// Launch `n` times with grid.y = `ranks`, then sync and read back.
fn run(gpu: &mut Gpu, func: &str, c: &Case, gl: bool, ranks: usize, n: usize) -> Vec<f32> {
    let grid = [c.m as u32, ranks as u32, 1];
    let block = [32u32, 1, 1];
    let mut blob = if gl { c.blob_gl.clone() } else { c.blob_l.clone() };
    for _ in 0..n {
        gpu.launch_kernel_blob(func, grid, block, 0, &mut blob)
            .expect("launch");
    }
    gpu.hip.device_synchronize().expect("sync");
    gpu.download_f32(&c.resid).expect("download")
}

// ──────────────────────────────── the bench ───────────────────────────────

fn main() {
    let mut gpu = Gpu::init().expect("GPU init");
    eprintln!("arch={}", gpu.arch);
    eprintln!(
        "REGIME: HIP-dispatch microbench = triage filter for gross defects only.\n\
         Host launch latency masks device effects; PM4 replay can rank differently.\n\
         Final acceptance = golden bundle (registry/redline-golden-v1.json), HIP + PM4 arms.\n"
    );

    gpu.ensure_kernel_public(GL_MOD, GL_SRC, GL_FN).expect("JIT mq2gl down");
    gpu.ensure_kernel_public(L_MOD, L_SRC, L_FN).expect("JIT mq2l down");

    // ── a3b routed-expert DOWN shape: M = hidden = 2048, K = moe_inter = 512 ──
    let (m, k, k_top, n_exp) = (2048usize, 512usize, 8usize, 256usize);
    let c = build_case(&mut gpu, m, k, k_top, n_exp, 0x51D0);
    eprintln!(
        "a3b down: M={m} K={k} (groups/row={}) k_top={k_top} n_experts={}\n\
         routed experts={:?}\n topk_weights={:?}\n\
         NOTE K={k} → groups_per_row={} → quads=0: the K4 main loop does NOT run at this\n\
         shape, only the tail>=2 epilogue. The K sweep below covers the main loop.\n",
        c.gpr,
        c.n_exp,
        c.topk,
        c.tw.iter().map(|v| (v * 1e4).round() / 1e4).collect::<Vec<_>>(),
        c.gpr
    );

    let mut all_ok = true;

    // ── 1. cumulative-rank ladder (zero-seeded): localizes a k_top fan-out bug ──
    eprintln!("[1] cumulative-rank ladder, zero-seeded residual (grid.y = 1..k_top):");
    for (gl, fname, tag) in [(true, GL_FN, "MQ2GL"), (false, L_FN, "MQ2L ")] {
        for ranks in 1..=k_top {
            gpu.fill_f32(&c.resid, 0.0).expect("zero resid");
            let got = run(&mut gpu, fname, &c, gl, ranks, 1);
            let (want, absum) = c.reference(gl, ranks, false, 1.0);
            let s = stats(&got, &want, &absum);
            let ok = s.max_rel <= TOL;
            all_ok &= ok;
            eprintln!(
                "  {tag} ranks=1..{ranks:<2} max|abs|={:.3e} @row {:<5} max rel={:.3e} @row {:<5}  {}",
                s.max_abs, s.abs_idx, s.max_rel, s.rel_idx,
                if ok { "PASS" } else { "FAIL <-- k_top accumulation diverges here" }
            );
        }
    }

    // ── 2. full k_top over a NONZERO pre-seeded residual (the headline check) ──
    eprintln!("\n[2] full k_top={k_top} over NONZERO pre-seeded x_residual (out = seed + Δ):");
    for (gl, fname, tag) in [(true, GL_FN, "MQ2GL down"), (false, L_FN, "MQ2L  down")] {
        write_f32(&gpu, &c.resid, &c.seed);
        let got = run(&mut gpu, fname, &c, gl, k_top, 1);
        let (want, absum) = c.reference(gl, k_top, true, 1.0);
        let s = stats(&got, &want, &absum);
        all_ok &= report(tag, &s, &got, &want);
        // The seed-survival proof is the check above (want = seed + Δ, so a
        // store instead of an atomicAdd fails it). This line documents that no
        // row passed trivially with Δ≈0: min over rows of the delta actually
        // deposited by the kernel.
        let dmin = got
            .iter()
            .zip(&c.seed)
            .map(|(g, s)| (*g - *s).abs())
            .fold(f32::INFINITY, f32::min);
        eprintln!(
            "  {:<34} min over rows of |out-seed| = {dmin:.3e} (Δ actually deposited; \
             0 would mean a row got nothing)",
            ""
        );
    }

    // ── 3. double launch: out = seed + 2Δ (accumulate, not overwrite) ──
    eprintln!("\n[3] double launch on one seeded residual (out = seed + 2Δ):");
    for (gl, fname, tag) in [(true, GL_FN, "MQ2GL down ×2"), (false, L_FN, "MQ2L  down ×2")] {
        write_f32(&gpu, &c.resid, &c.seed);
        let got = run(&mut gpu, fname, &c, gl, k_top, 2);
        let (want, absum) = c.reference(gl, k_top, true, 2.0);
        let s = stats(&got, &want, &absum);
        all_ok &= report(tag, &s, &got, &want);
    }

    // ── 4. cross-format agreement (soft; fp16 codebook rounding differs) ──
    {
        write_f32(&gpu, &c.resid, &c.seed);
        let g_gl = run(&mut gpu, GL_FN, &c, true, k_top, 1);
        write_f32(&gpu, &c.resid, &c.seed);
        let g_l = run(&mut gpu, L_FN, &c, false, k_top, 1);
        let rms = (g_gl.iter().map(|v| (*v as f64) * (*v as f64)).sum::<f64>() / m as f64).sqrt();
        let md = g_gl
            .iter()
            .zip(&g_l)
            .map(|(a, b)| (*a as f64 - *b as f64).abs())
            .fold(0.0f64, f64::max);
        eprintln!(
            "\n[4] cross-format GL vs Lloyd on identical codes+scales: max|Δ|={md:.3e} \
             (rel {:.2e}) — SOFT, expected ≠0 from fp16 codebook rounding",
            md / rms.max(1e-30)
        );
    }

    // ── 5. K sweep: exercises the K4 main loop and all three tail epilogues ──
    eprintln!("\n[5] K sweep (M=256, n_exp=16, k_top=8) — quads/tail coverage:");
    for (ki, &ks) in [512usize, 1024, 1280, 1792, 2048].iter().enumerate() {
        let cs = build_case(&mut gpu, 256, ks, 8, 16, 0x9000 + (ki as u64) * 0x100);
        let (quads, tail) = (cs.gpr / 4, cs.gpr % 4);
        for (gl, fname, tag) in [(true, GL_FN, "MQ2GL"), (false, L_FN, "MQ2L ")] {
            write_f32(&gpu, &cs.resid, &cs.seed);
            let got = run(&mut gpu, fname, &cs, gl, 8, 1);
            let (want, absum) = cs.reference(gl, 8, true, 1.0);
            let s = stats(&got, &want, &absum);
            let ok = s.max_rel <= TOL;
            all_ok &= ok;
            eprintln!(
                "  {tag} K={ks:<5} gpr={:<2} quads={quads} tail={tail}  max|abs|={:.3e} \
                 max rel={:.3e} @row {:<4} {}",
                cs.gpr, s.max_abs, s.max_rel, s.rel_idx,
                if ok { "PASS" } else { "FAIL" }
            );
        }
    }

    eprintln!(
        "\n=========== CORRECTNESS {} (tolerance {:.0e} relative; denominator \
         max(|cpu_i|, rms) so a cancelled near-zero row cannot fake a failure) ===========\n",
        if all_ok { "PASS" } else { "FAIL" },
        TOL
    );

    // ── timing at the a3b down shape ──
    let grid = [m as u32, k_top as u32, 1];
    let block = [32u32, 1, 1];
    let iters = 7usize;
    let inner = 200usize;

    let time_one = |gpu: &mut Gpu, func: &str, blob: &mut [u8]| -> (f64, f64) {
        // Warmup: absorbs JIT + kernarg/module cache. Pass 1 is JIT-contaminated
        // and is deliberately discarded.
        for _ in 0..32 {
            gpu.launch_kernel_blob(func, grid, block, 0, blob).unwrap();
        }
        gpu.hip.device_synchronize().unwrap();
        let mut us: Vec<f64> = Vec::with_capacity(iters);
        for _ in 0..iters {
            let t0 = Instant::now();
            for _ in 0..inner {
                gpu.launch_kernel_blob(func, grid, block, 0, blob).unwrap();
            }
            gpu.hip.device_synchronize().unwrap();
            us.push(t0.elapsed().as_secs_f64() * 1e6 / inner as f64);
        }
        us.sort_by(|a, b| a.partial_cmp(b).unwrap());
        (us[0], us[iters / 2])
    };

    let mut bl_gl = c.blob_gl.clone();
    let mut bl_l = c.blob_l.clone();
    let (gl_min, gl_med) = time_one(&mut gpu, GL_FN, &mut bl_gl);
    let (l_min, l_med) = time_one(&mut gpu, L_FN, &mut bl_l);

    let gib = 1024.0 * 1024.0 * 1024.0;
    let by_gl = c.weight_bytes(true);
    let by_l = c.weight_bytes(false);
    eprintln!("timing: warmup 32 launches, then {iters} timed iterations × {inner} launches each.");
    eprintln!("        MIN and MEDIAN of the per-iteration means. First pass is JIT-contaminated");
    eprintln!("        and is NOT among these — the warmup absorbs it.\n");
    eprintln!(
        "{:<8} {:>9} {:>9} {:>12} {:>11} {:>9} {:>11}",
        "variant", "min us", "med us", "wt bytes", "B/element", "bpw", "GiB/s(med)"
    );
    for (tag, mn, md, by, per_group) in [
        ("MQ2GL", gl_min, gl_med, by_gl, 66.0f64),
        ("MQ2L", l_min, l_med, by_l, 72.0f64),
    ] {
        eprintln!(
            "{:<8} {:>9.2} {:>9.2} {:>12.0} {:>11.4} {:>9.4} {:>11.1}",
            tag,
            mn,
            md,
            by,
            per_group / 256.0,
            per_group * 8.0 / 256.0,
            by / (md * 1e-6) / gib
        );
    }
    eprintln!(
        "\nMQ2GL vs MQ2L: {:+.2}% on median time, {:+.2}% on min time, {:.1}% fewer weight bytes",
        100.0 * (gl_med / l_med - 1.0),
        100.0 * (gl_min / l_min - 1.0),
        100.0 * (1.0 - by_gl / by_l)
    );
    eprintln!(
        "token-shaped (this kernel only, ×{LAYERS} layers/token): MQ2GL {:.3} ms/tok → {:.0} tok/s ceiling; \
         MQ2L {:.3} ms/tok → {:.0} tok/s ceiling",
        gl_med * LAYERS as f64 * 1e-3,
        1e6 / (gl_med * LAYERS as f64),
        l_med * LAYERS as f64 * 1e-3,
        1e6 / (l_med * LAYERS as f64)
    );
    eprintln!(
        "\nCAVEAT AGAIN: at M={m}, K={k} this is a ~{:.0} us launch — host dispatch latency is a\n\
         first-order term. Treat any delta under ~10% here as unresolved, not as a verdict.\n\
         A prior HIP microbench put MQ2GL at -3.1% and nearly killed the format.",
        gl_med
    );

    if !all_ok {
        std::process::exit(1);
    }
}
