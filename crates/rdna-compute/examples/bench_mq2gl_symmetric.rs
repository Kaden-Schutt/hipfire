// SPDX-License-Identifier: MIT OR Apache-2.0
// hipfire — see LICENSE and NOTICE in the project root.
//! THREE-WAY head-to-head on the a3b routed-expert gate_up decode shape
//! (M=1024, K=2048, k_top=8):
//!
//!   1. `gemv_mq2g256_lloyd_moe_gate_up_k8_indexed`      — MQ2-Lloyd, 72 B/group,
//!      per-block 4-entry fp16 codebook re-staged into LDS every K4 iteration.
//!   2. `gemv_mq2g256gl_moe_gate_up_k8_indexed`          — MQ2-GL, 66 B/group,
//!      global 4-entry codebook parked in LDS, one ds_read per weight.
//!   3. `gemv_mq2g256gl_moe_gate_up_sym_k8_indexed`      — MQ2-GL, same bytes,
//!      SYMMETRIC codebook {-b,-a,+a,+b} passed as TWO scalars; no codebook
//!      load and no per-weight multiply (sign-XOR + magnitude-bucket split).
//!
//! This is the headline bench for the sym rewrite: the plain GL kernel
//! previously measured -3.1% vs MQ2L despite reading 8.3% fewer weight bytes,
//! and the sym kernel exists to reverse that.
//!
//! ---------------------------------------------------------------------------
//! REGIME CAVEAT — READ BEFORE QUOTING ANY NUMBER FROM THIS FILE
//! ---------------------------------------------------------------------------
//! This is a HIP-DISPATCH MICROBENCHMARK. It is a TRIAGE FILTER for gross
//! defects (wrong numbers, pathological slowness), NOT a verdict on a kernel.
//! Host launch latency and per-launch driver work mask device-level effects,
//! and the same kernel can measure differently once lowered to retained PM4
//! replay. A prior HIP microbench measured the MQ2GL kernel at -3.1% and that
//! number was nearly used to KILL the format. Treat a HIP-regime result here as
//! INDICATIVE ONLY. Final acceptance is the golden bundle
//! (`registry/redline-golden-v1.json`) with both HIP and PM4 arms.
//!
//! Two further regime caveats specific to this bench:
//!   * the top-k selection is FIXED across every timed launch, so the ~4 MiB
//!     working set stays L2/MALL-resident. That flatters both formats and
//!     UNDERSTATES the byte-count advantage of GL over MQ2L.
//!   * x is not re-rotated per call, so no FWHT cost is included. Real decode
//!     amortizes one FWHT-256 of x across all k_top experts, so this is the
//!     right isolation for the GEMV, but it is not a per-token cost.
//!
//! ---------------------------------------------------------------------------
//! CORRECTNESS METHOD
//! ---------------------------------------------------------------------------
//! Synthetic Gaussian weights are encoded ONCE per expert (per-256-group fp16
//! RMS scale + nearest-neighbour 2-bit codes against the ascending codebook),
//! then emitted into BOTH on-disk layouts from that single code assignment:
//!
//!   MQ2-GL (qt=38) SoA : [0 .. M*gpr*64) packed 2-bit indices (4 codes/byte,
//!                        little-endian), then [.. +M*gpr*2) fp16 per-block
//!                        scales, both row-major in (row, group). 2.0625 bpw.
//!   MQ2-Lloyd (qt=19)  : 72 B/group = [4 × fp16 codebook][64 B indices],
//!                        interleaved. Here the per-block table is
//!                        fp16(scale × CB[e]) — i.e. the SAME reconstruction
//!                        the GL kernel computes, materialized per block.
//!
//! Because both blobs carry the SAME codes and the SAME reconstruction values
//! (up to one fp16 rounding of the per-block table entries), all three kernels
//! should agree to a few ulps of accumulation order. That is deliberate: it
//! turns the three-way comparison into a real cross-check instead of three
//! unrelated numbers.
//!
//! The CPU reference is derived from the FORMAT SPEC (byte b of a group holds
//! codes 4b..4b+3 in ascending bit order, code q reconstructs to CB[q]), NOT
//! from the kernel's arithmetic. In particular it does a plain table lookup on
//! the ascending codebook and never touches the sym kernel's sign-bit / Gray-bit
//! algebra, so a shared misunderstanding of that algebra cannot cancel out —
//! the classic "|c| = bit0 instead of bit0 ^ bit1" bug (which swaps a and b on
//! exactly half the codes and still looks numerically plausible) IS caught.
//! Every one of the k_top × M = 8192 outputs is checked, not just row 0.
//!
//! x is arbitrary (deterministic pseudo-Gaussian) and is NOT FWHT-rotated: the
//! kernels treat x_rot as opaque, and the same bytes feed the GPU and the CPU
//! reference, so the rotation is irrelevant to parity. Real callers must
//! pre-rotate.
//!
//! Output buffers are pre-filled with NaN so a kernel that fails to write a row
//! fails loudly instead of silently inheriting the previous kernel's result.
//!
//! Run: cargo run --release -p rdna-compute --example bench_mq2gl_symmetric
use hip_bridge::KernargBlob;
use rdna_compute::Gpu;
use std::time::Instant;

const MQ2L_SRC: &str =
    include_str!("../../../kernels/src/gemv_mq2g256_lloyd_moe_gate_up_indexed.hip");
const MQ2GL_SRC: &str = include_str!("../../../kernels/src/gemv_mq2g256gl_moe_gate_up_indexed.hip");
const MQ2GL_SYM_SRC: &str =
    include_str!("../../../kernels/src/gemv_mq2g256gl_moe_gate_up_sym_indexed.hip");

/// Textbook Lloyd–Max levels for a unit Gaussian, 2 bit (== `GL_CB2` in
/// `crates/hipfire-quantize/src/main.rs`). Ascending — the encoder picks the
/// nearest neighbour in THIS order, so code q ↦ CB[q].
const CB: [f32; 4] = [-1.5104, -0.4528, 0.4528, 1.5104];
/// The positive half-levels the sym kernel takes as its two scalar args.
const CB_A: f32 = 0.4528;
const CB_B: f32 = 1.5104;

/// Relative tolerance for f32 accumulation of a 2048-term dot product.
const TOL: f64 = 1e-4;

// ── deterministic host RNG ──────────────────────────────────────────────────

fn mix(x: u64) -> u64 {
    let h = x
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    ((h ^ (h >> 33)).wrapping_mul(0xff51afd7ed558ccd)) ^ (h >> 28)
}

fn u01(h: u64) -> f32 {
    (((h >> 40) & 0xff_ffff) as f32) / 16_777_216.0
}

/// Irwin–Hall(4), normalized to unit variance. Good enough as a stand-in for
/// post-FWHT weights, which are Gaussian by CLT.
fn gauss(key: u64) -> f32 {
    let s = u01(mix(key))
        + u01(mix(key ^ (1u64 << 56)))
        + u01(mix(key ^ (2u64 << 56)))
        + u01(mix(key ^ (3u64 << 56)));
    (s - 2.0) * 1.7320508
}

// ── fp16 <-> f32 (round-to-nearest-even; subnormals flush to zero) ──────────

fn f32_to_f16_bits(f: f32) -> u16 {
    let x = f.to_bits();
    let sign = ((x >> 16) & 0x8000) as u16;
    let exp = ((x >> 23) & 0xff) as i32;
    let mant = x & 0x7f_ffff;
    if exp == 0xff {
        return sign | 0x7c00 | if mant != 0 { 0x200 } else { 0 };
    }
    let mut e = exp - 127 + 15;
    if e >= 31 {
        return sign | 0x7c00;
    }
    if e <= 0 {
        // Out of the range this bench produces; flush (documented, asserted).
        return sign;
    }
    let mut hm = mant >> 13;
    let round_bit = (mant >> 12) & 1;
    let sticky = (mant & 0xfff) != 0;
    if round_bit == 1 && (sticky || (hm & 1) == 1) {
        hm += 1;
        if hm == 0x400 {
            hm = 0;
            e += 1;
            if e >= 31 {
                return sign | 0x7c00;
            }
        }
    }
    sign | ((e as u16) << 10) | (hm as u16)
}

fn f16_to_f32(h: u16) -> f32 {
    let sign = ((h & 0x8000) as u32) << 16;
    let e = ((h >> 10) & 0x1f) as u32;
    let m = (h & 0x3ff) as u32;
    if e == 0 {
        return f32::from_bits(sign);
    }
    f32::from_bits(sign | ((e + 112) << 23) | (m << 13))
}

// ── encoder (spec-derived, mirrors quantize_mq2g256gl / gl_encode_block) ────

/// One 256-group → (fp16 RMS scale bits, 256 nearest-neighbour codes).
fn encode_group(w: &[f32; 256], cb: &[f32; 4]) -> (u16, [u8; 256]) {
    let ss: f64 = w.iter().map(|v| (*v as f64) * (*v as f64)).sum();
    let rms = (ss / 256.0).sqrt() as f32;
    let sbits = f32_to_f16_bits(rms);
    let scale = f16_to_f32(sbits);
    assert!(
        scale > 0.0 && scale.is_finite(),
        "bench bug: fp16 scale underflowed to {scale} (rms={rms})"
    );
    let inv = 1.0 / scale;
    let mut codes = [0u8; 256];
    for (i, v) in w.iter().enumerate() {
        let z = *v * inv;
        let mut best = 0usize;
        let mut bd = (z - cb[0]).abs();
        for (q, &c) in cb.iter().enumerate().skip(1) {
            let d = (z - c).abs();
            if d < bd {
                bd = d;
                best = q;
            }
        }
        codes[i] = best as u8;
    }
    (sbits, codes)
}

/// Build one expert's weights and emit BOTH blobs from the SAME code
/// assignment. Returns `(gl_blob, mq2l_blob)`.
fn build_expert(m: usize, k: usize, seed: u64) -> (Vec<u8>, Vec<u8>) {
    let gpr = k / 256;
    let idx_bytes = m * gpr * 64;
    let mut gl = vec![0u8; idx_bytes + m * gpr * 2];
    let mut l = vec![0u8; m * gpr * 72];
    let mut w = [0.0f32; 256];
    for row in 0..m {
        for g in 0..gpr {
            // Per-group magnitude so the fp16 scales carry real mantissa bits
            // (a power-of-two-only scale would hide a mantissa-decode bug).
            let key = (seed << 44) ^ ((row as u64) << 24) ^ ((g as u64) << 12);
            let mag = 0.002f32 + ((mix(key ^ 0xa5a5) % 4096) as f32) * 1e-6;
            for (i, wi) in w.iter_mut().enumerate() {
                *wi = gauss(key ^ i as u64) * mag;
            }
            let (sbits, codes) = encode_group(&w, &CB);
            let scale = f16_to_f32(sbits);

            // --- MQ2-GL SoA: indices region, then fp16 scale region ---
            let base = (row * gpr + g) * 64;
            for b in 0..64 {
                gl[base + b] = codes[4 * b]
                    | (codes[4 * b + 1] << 2)
                    | (codes[4 * b + 2] << 4)
                    | (codes[4 * b + 3] << 6);
            }
            let so = idx_bytes + (row * gpr + g) * 2;
            gl[so] = (sbits & 0xff) as u8;
            gl[so + 1] = (sbits >> 8) as u8;

            // --- MQ2-Lloyd: [4 × fp16 cb][64 B indices] per group ---
            let off = (row * gpr + g) * 72;
            for (e, &c) in CB.iter().enumerate() {
                let h = f32_to_f16_bits(scale * c);
                l[off + 2 * e] = (h & 0xff) as u8;
                l[off + 2 * e + 1] = (h >> 8) as u8;
            }
            l[off + 8..off + 72].copy_from_slice(&gl[base..base + 64]);
        }
    }
    (gl, l)
}

// ── CPU references (derived from the FORMAT SPEC, not from the kernels) ─────

/// MQ2-GL: y[row] = Σ_g scale[row,g] · Σ_i CB[code_i] · x[g*256 + i].
fn ref_gl_rows(blob: &[u8], m: usize, k: usize, x: &[f32], cb: &[f32; 4]) -> Vec<f64> {
    let gpr = k / 256;
    let idx_bytes = m * gpr * 64;
    let mut out = vec![0.0f64; m];
    for (row, o) in out.iter_mut().enumerate() {
        let mut acc = 0.0f64;
        for g in 0..gpr {
            let so = idx_bytes + (row * gpr + g) * 2;
            let s = f16_to_f32(u16::from_le_bytes([blob[so], blob[so + 1]])) as f64;
            let base = (row * gpr + g) * 64;
            let mut sub = 0.0f64;
            for b in 0..64 {
                let byte = blob[base + b];
                for j in 0..4 {
                    let q = ((byte >> (2 * j)) & 3) as usize;
                    sub += cb[q] as f64 * x[g * 256 + b * 4 + j] as f64;
                }
            }
            acc += s * sub;
        }
        *o = acc;
    }
    out
}

/// MQ2-Lloyd: y[row] = Σ_g Σ_i cb_g[code_i] · x[g*256 + i], cb_g read from the
/// group's own 8-byte fp16 header.
fn ref_mq2l_rows(blob: &[u8], m: usize, k: usize, x: &[f32]) -> Vec<f64> {
    let gpr = k / 256;
    let mut out = vec![0.0f64; m];
    for (row, o) in out.iter_mut().enumerate() {
        let mut acc = 0.0f64;
        for g in 0..gpr {
            let off = (row * gpr + g) * 72;
            let mut cb = [0.0f64; 4];
            for (e, c) in cb.iter_mut().enumerate() {
                *c = f16_to_f32(u16::from_le_bytes([blob[off + 2 * e], blob[off + 2 * e + 1]]))
                    as f64;
            }
            let ib = off + 8;
            for b in 0..64 {
                let byte = blob[ib + b];
                for j in 0..4 {
                    let q = ((byte >> (2 * j)) & 3) as usize;
                    acc += cb[q] * x[g * 256 + b * 4 + j] as f64;
                }
            }
        }
        *o = acc;
    }
    out
}

// ── comparison ──────────────────────────────────────────────────────────────

/// Outputs below this fraction of the peak |ref| are catastrophic-cancellation
/// cases (a 2048-term signed sum that landed near zero). A POINTWISE relative
/// error is not a meaningful metric there for f32 accumulation — those entries
/// are covered by `max_abs` / `rel_scaled` instead, which is the
/// cancellation-proof relative measure. A real decode bug perturbs outputs by
/// O(their own magnitude), so this exclusion costs no detection power.
const SMALL_CUT: f64 = 0.05;

struct Stats {
    max_abs: f64,
    /// Pointwise |Δ|/|ref| over entries with |ref| above `SMALL_CUT` × the
    /// output magnitude scale.
    max_rel: f64,
    /// max_abs normalized by max|ref| — the cancellation-proof relative metric.
    rel_scaled: f64,
    worst: usize,
    worst_got: f64,
    worst_ref: f64,
    small_skipped: usize,
}

fn compare(got: &[f64], want: &[f64]) -> Stats {
    assert_eq!(got.len(), want.len());
    let scale = want.iter().fold(0.0f64, |a, v| a.max(v.abs())).max(1e-30);
    let mut s = Stats {
        max_abs: 0.0,
        max_rel: 0.0,
        rel_scaled: 0.0,
        worst: 0,
        worst_got: 0.0,
        worst_ref: 0.0,
        small_skipped: 0,
    };
    for (i, (&g, &w)) in got.iter().zip(want.iter()).enumerate() {
        let d = (g - w).abs();
        if d > s.max_abs || !d.is_finite() {
            s.max_abs = d;
            s.worst = i;
            s.worst_got = g;
            s.worst_ref = w;
        }
        if w.abs() >= SMALL_CUT * scale {
            let r = d / w.abs();
            if r > s.max_rel || !r.is_finite() {
                s.max_rel = r;
            }
        } else {
            s.small_skipped += 1;
        }
    }
    s.rel_scaled = s.max_abs / scale;
    s
}

/// Flatten the split gate/up outputs back into rank-major row order.
fn flatten(gate: &[f32], up: &[f32], m: usize, k_top: usize) -> Vec<f64> {
    let mi = m / 2;
    let mut out = vec![0.0f64; k_top * m];
    for rank in 0..k_top {
        for row in 0..m {
            out[rank * m + row] = if row < mi {
                gate[rank * mi + row] as f64
            } else {
                up[rank * mi + (row - mi)] as f64
            };
        }
    }
    out
}

fn main() {
    // The whole sym trick rests on the codebook being EXACTLY symmetric in f32.
    assert_eq!(CB[2], CB_A, "codebook +a mismatch");
    assert_eq!(CB[3], CB_B, "codebook +b mismatch");
    assert_eq!(CB[1], -CB_A, "codebook -a is not the exact negation of +a");
    assert_eq!(CB[0], -CB_B, "codebook -b is not the exact negation of +b");
    assert!(CB_A < CB_B && CB_A > 0.0, "sym kernel needs 0 < a < b");

    let mut gpu = Gpu::init().expect("GPU init");
    eprintln!("arch={}", gpu.arch);

    // a3b routed-expert gate_up decode shape: M = 2*moe_intermediate (gate|up
    // split), K = hidden, k_top = 8.
    let m = 1024usize;
    let k = 2048usize;
    let k_top = 8usize;
    let n_exp = 32usize;
    let gpr = k / 256;
    let mi = m / 2;

    let mq2l_bytes = m * gpr * 72;
    let mq2gl_bytes = m * gpr * 64 + m * gpr * 2;
    let elems = (m * k) as f64;
    eprintln!(
        "M={m} K={k} k_top={k_top} n_exp={n_exp}\n\
         per-expert weights: MQ2L {} KiB ({:.4} B/elem, {:.4} bpw) | \
         MQ2GL {} KiB ({:.4} B/elem, {:.4} bpw)  -> GL reads {:.2}% fewer bytes",
        mq2l_bytes / 1024,
        mq2l_bytes as f64 / elems,
        8.0 * mq2l_bytes as f64 / elems,
        mq2gl_bytes / 1024,
        mq2gl_bytes as f64 / elems,
        8.0 * mq2gl_bytes as f64 / elems,
        100.0 * (1.0 - mq2gl_bytes as f64 / mq2l_bytes as f64)
    );

    // ---- host-side encode ----
    let t_enc = Instant::now();
    let mut gl_blobs = Vec::with_capacity(n_exp);
    let mut l_blobs = Vec::with_capacity(n_exp);
    for e in 0..n_exp {
        let (g, l) = build_expert(m, k, 0x1000 + e as u64);
        assert_eq!(g.len(), mq2gl_bytes);
        assert_eq!(l.len(), mq2l_bytes);
        gl_blobs.push(g);
        l_blobs.push(l);
    }
    eprintln!(
        "encoded {n_exp} experts in {:.2}s ({:.1} M weights)",
        t_enc.elapsed().as_secs_f64(),
        (n_exp * m * k) as f64 / 1e6
    );

    // ---- upload ----
    let l_ts: Vec<_> = l_blobs
        .iter()
        .map(|b| gpu.upload_raw(b, &[mq2l_bytes]).unwrap())
        .collect();
    let gl_ts: Vec<_> = gl_blobs
        .iter()
        .map(|b| gpu.upload_raw(b, &[mq2gl_bytes]).unwrap())
        .collect();
    let l_ptrs: Vec<u64> = l_ts.iter().map(|t| t.buf.as_ptr() as u64).collect();
    let gl_ptrs: Vec<u64> = gl_ts.iter().map(|t| t.buf.as_ptr() as u64).collect();
    let l_ptr_t = gpu
        .upload_raw(
            &l_ptrs.iter().flat_map(|p| p.to_le_bytes()).collect::<Vec<u8>>(),
            &[n_exp],
        )
        .unwrap();
    let gl_ptr_t = gpu
        .upload_raw(
            &gl_ptrs.iter().flat_map(|p| p.to_le_bytes()).collect::<Vec<u8>>(),
            &[n_exp],
        )
        .unwrap();

    // 8 distinct experts (0,3,6,...,21) — same set for every variant.
    let topk: Vec<i32> = (0..k_top as i32).map(|i| (i * 3) % n_exp as i32).collect();
    let topk_t = gpu
        .upload_raw(
            &topk.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>(),
            &[k_top],
        )
        .unwrap();

    // x: arbitrary but deterministic; NOT FWHT-rotated (see header).
    let x: Vec<f32> = (0..k).map(|i| gauss(0xbeef_0000 ^ i as u64) * 0.5).collect();
    let x_t = gpu.upload_f32(&x, &[k]).unwrap();

    // NaN-prefilled outputs, one pair per kernel — an unwritten row fails loud.
    let nan = f32::NAN;
    let y_l_g = gpu.full_f32(&[k_top * mi], nan).unwrap();
    let y_l_u = gpu.full_f32(&[k_top * mi], nan).unwrap();
    let y_g_g = gpu.full_f32(&[k_top * mi], nan).unwrap();
    let y_g_u = gpu.full_f32(&[k_top * mi], nan).unwrap();
    let y_s_g = gpu.full_f32(&[k_top * mi], nan).unwrap();
    let y_s_u = gpu.full_f32(&[k_top * mi], nan).unwrap();

    // ---- JIT (distinct module names: the kernel cache is keyed by module) ----
    gpu.ensure_kernel_public(
        "gemv_mq2g256_lloyd_moe_gate_up_indexed",
        MQ2L_SRC,
        "gemv_mq2g256_lloyd_moe_gate_up_k8_indexed",
    )
    .expect("JIT mq2l");
    gpu.ensure_kernel_public(
        "gemv_mq2g256gl_moe_gate_up_indexed",
        MQ2GL_SRC,
        "gemv_mq2g256gl_moe_gate_up_k8_indexed",
    )
    .expect("JIT mq2gl");
    gpu.ensure_kernel_public(
        "gemv_mq2g256gl_moe_gate_up_sym_indexed",
        MQ2GL_SYM_SRC,
        "gemv_mq2g256gl_moe_gate_up_sym_k8_indexed",
    )
    .expect("JIT mq2gl-sym");

    let grid = [m as u32, k_top as u32, 1];
    let block = [32u32, 1, 1];

    // ---- kernarg blobs ----
    let mut blob_l = KernargBlob::new();
    blob_l.push_ptr(l_ptr_t.buf.as_ptr() as *const _);
    blob_l.push_ptr(topk_t.buf.as_ptr() as *const _);
    blob_l.push_ptr(x_t.buf.as_ptr() as *const _);
    blob_l.push_ptr(y_l_g.buf.as_ptr() as *const _);
    blob_l.push_ptr(y_l_u.buf.as_ptr() as *const _);
    blob_l.push_i32(m as i32);
    blob_l.push_i32(k as i32);
    let mut bl = blob_l.into_vec();

    let mut blob_g = KernargBlob::new();
    blob_g.push_ptr(gl_ptr_t.buf.as_ptr() as *const _);
    blob_g.push_ptr(topk_t.buf.as_ptr() as *const _);
    blob_g.push_ptr(x_t.buf.as_ptr() as *const _);
    blob_g.push_ptr(y_g_g.buf.as_ptr() as *const _);
    blob_g.push_ptr(y_g_u.buf.as_ptr() as *const _);
    for c in CB {
        blob_g.push_f32(c);
    }
    blob_g.push_i32(m as i32);
    blob_g.push_i32(k as i32);
    let mut bg = blob_g.into_vec();

    // sym: TWO scalars (cb_a, cb_b) in the same slot as the plain kernel's four.
    let mut blob_s = KernargBlob::new();
    blob_s.push_ptr(gl_ptr_t.buf.as_ptr() as *const _);
    blob_s.push_ptr(topk_t.buf.as_ptr() as *const _);
    blob_s.push_ptr(x_t.buf.as_ptr() as *const _);
    blob_s.push_ptr(y_s_g.buf.as_ptr() as *const _);
    blob_s.push_ptr(y_s_u.buf.as_ptr() as *const _);
    blob_s.push_f32(CB_A);
    blob_s.push_f32(CB_B);
    blob_s.push_i32(m as i32);
    blob_s.push_i32(k as i32);
    let mut bs = blob_s.into_vec();

    const NAME_L: &str = "gemv_mq2g256_lloyd_moe_gate_up_k8_indexed";
    const NAME_G: &str = "gemv_mq2g256gl_moe_gate_up_k8_indexed";
    const NAME_S: &str = "gemv_mq2g256gl_moe_gate_up_sym_k8_indexed";

    // ---- one launch of each (this is the JIT-contaminated pass) ----
    for (name, bytes) in [
        (NAME_L, &mut bl),
        (NAME_G, &mut bg),
        (NAME_S, &mut bs),
    ] {
        gpu.launch_kernel_blob(name, grid, block, 0, bytes).unwrap();
    }
    gpu.hip.device_synchronize().unwrap();

    let got_l = flatten(
        &gpu.download_f32(&y_l_g).unwrap(),
        &gpu.download_f32(&y_l_u).unwrap(),
        m,
        k_top,
    );
    let got_g = flatten(
        &gpu.download_f32(&y_g_g).unwrap(),
        &gpu.download_f32(&y_g_u).unwrap(),
        m,
        k_top,
    );
    let got_s = flatten(
        &gpu.download_f32(&y_s_g).unwrap(),
        &gpu.download_f32(&y_s_u).unwrap(),
        m,
        k_top,
    );

    // ---- CPU references, all k_top × M outputs ----
    let t_ref = Instant::now();
    let mut ref_l = vec![0.0f64; k_top * m];
    let mut ref_g = vec![0.0f64; k_top * m];
    for (rank, &e) in topk.iter().enumerate() {
        let e = e as usize;
        let rl = ref_mq2l_rows(&l_blobs[e], m, k, &x);
        let rg = ref_gl_rows(&gl_blobs[e], m, k, &x, &CB);
        ref_l[rank * m..(rank + 1) * m].copy_from_slice(&rl);
        ref_g[rank * m..(rank + 1) * m].copy_from_slice(&rg);
    }
    eprintln!(
        "CPU reference: {} outputs × {k} terms in {:.2}s",
        k_top * m,
        t_ref.elapsed().as_secs_f64()
    );

    let report = |label: &str, s: &Stats, m: usize| -> bool {
        let worst_rank = s.worst / m;
        let worst_row = s.worst % m;
        let half = if worst_row < m / 2 { "gate" } else { "up" };
        let metric = s.max_rel.max(s.rel_scaled);
        let ok = metric < TOL && metric.is_finite();
        eprintln!(
            "  {label:<22} max_abs={:.3e}  max_rel={:.3e}  rel_vs_scale={:.3e}\n\
             {:<24}worst @ rank={worst_rank} row={worst_row} ({half}): gpu={:.6e} cpu={:.6e}  \
             [{} near-zero outputs (<{:.0}% of peak) excluded from max_rel; \
             they are covered by max_abs/rel_vs_scale]",
            s.max_abs,
            s.max_rel,
            s.rel_scaled,
            "",
            s.worst_got,
            s.worst_ref,
            s.small_skipped,
            100.0 * SMALL_CUT
        );
        ok
    };

    eprintln!("\n=== CORRECTNESS: each kernel vs an independent CPU reference ===");
    eprintln!("(reference decodes the blob per the FORMAT SPEC and does a plain");
    eprintln!(" ascending-codebook table lookup — it shares no algebra with the");
    eprintln!(" sym kernel's sign-XOR / Gray-bit magnitude split)");
    let s_l = compare(&got_l, &ref_l);
    let s_g = compare(&got_g, &ref_g);
    let s_s = compare(&got_s, &ref_g); // sym reads the SAME GL blob
    let ok_l = report("MQ2L vs cpu", &s_l, m);
    let ok_g = report("MQ2GL-plain vs cpu", &s_g, m);
    let ok_s = report("MQ2GL-sym  vs cpu", &s_s, m);
    let all_ok = ok_l && ok_g && ok_s;
    eprintln!(
        "\n  >>> {} <<<   (tolerance {:.0e} relative, all {} outputs checked)",
        if all_ok {
            "PASS: all three kernels match their CPU reference"
        } else {
            "FAIL: at least one kernel disagrees with the CPU reference"
        },
        TOL,
        k_top * m
    );

    eprintln!("\n=== CROSS-KERNEL DIVERGENCE (informational) ===");
    let d_sym = compare(&got_s, &got_g);
    eprintln!(
        "  sym vs plain GL      max_abs={:.3e}  max_rel={:.3e}  rel_vs_scale={:.3e}",
        d_sym.max_abs, d_sym.max_rel, d_sym.rel_scaled
    );
    eprintln!(
        "    same blob, same codebook VALUES, different summation order (two\n\
         \x20   magnitude-bucket accumulators with the codebook applied once at the\n\
         \x20   end, vs eight per-weight FMAs). Bit equality is NOT expected."
    );
    let d_fmt = compare(&got_g, &got_l);
    eprintln!(
        "  plain GL vs MQ2L     max_abs={:.3e}  max_rel={:.3e}  rel_vs_scale={:.3e}",
        d_fmt.max_abs, d_fmt.max_rel, d_fmt.rel_scaled
    );
    eprintln!(
        "    same codes; MQ2L stores fp16(scale·CB[e]) per block while GL computes\n\
         \x20   f32 scale × f32 CB[e], so ~2^-11 per-level rounding is expected here."
    );

    // ---- timing ----
    // Warmup is separate and DISCARDED: the first launch of each kernel above
    // was JIT-contaminated (hipcc compile + module load on first use).
    let warm = 64usize;
    let reps = 300usize;
    let rounds = 7usize;
    let bench = |gpu: &mut Gpu, name: &str, bytes: &mut [u8], n: usize| -> f64 {
        let t0 = Instant::now();
        for _ in 0..n {
            gpu.launch_kernel_blob(name, grid, block, 0, bytes).unwrap();
        }
        gpu.hip.device_synchronize().unwrap();
        t0.elapsed().as_secs_f64() * 1e6 / n as f64
    };
    for (name, bytes) in [
        (NAME_L, &mut bl),
        (NAME_G, &mut bg),
        (NAME_S, &mut bs),
    ] {
        let _ = bench(&mut gpu, name, bytes, warm);
    }

    let mut t_l = Vec::with_capacity(rounds);
    let mut t_g = Vec::with_capacity(rounds);
    let mut t_s = Vec::with_capacity(rounds);
    for _ in 0..rounds {
        // interleaved so DPM/thermal drift hits all three arms equally
        t_l.push(bench(&mut gpu, NAME_L, &mut bl, reps));
        t_g.push(bench(&mut gpu, NAME_G, &mut bg, reps));
        t_s.push(bench(&mut gpu, NAME_S, &mut bs, reps));
    }
    let stat = |v: &mut Vec<f64>| -> (f64, f64) {
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        (v[0], v[v.len() / 2])
    };
    let (min_l, med_l) = stat(&mut t_l);
    let (min_g, med_g) = stat(&mut t_g);
    let (min_s, med_s) = stat(&mut t_s);

    let b_l = (k_top * mq2l_bytes) as f64;
    let b_gl = (k_top * mq2gl_bytes) as f64;
    let gbs = |bytes: f64, us: f64| bytes / (us * 1e-6) / 1e9;

    eprintln!("\n=== TIMING ({rounds} rounds × {reps} launches, warmup {warm} discarded) ===");
    eprintln!("NOTE: the very first launch of each kernel (correctness pass above) is");
    eprintln!("      JIT-contaminated and is NOT part of any number below.");
    eprintln!(
        "\n{:<14} {:>10} {:>10} {:>10} {:>12} {:>12}",
        "variant", "min us", "med us", "B/elem", "GB/s(min)", "wt B/call"
    );
    for (n, mn, md, by) in [
        ("MQ2L", min_l, med_l, b_l),
        ("MQ2GL-plain", min_g, med_g, b_gl),
        ("MQ2GL-sym", min_s, med_s, b_gl),
    ] {
        eprintln!(
            "{n:<14} {mn:>10.2} {md:>10.2} {:>10.4} {:>12.1} {:>12.0}",
            by / (k_top as f64 * elems),
            gbs(by, mn),
            by
        );
    }
    eprintln!(
        "\nvs MQ2L (median):  plain GL {:+.2}%   sym GL {:+.2}%   (negative = faster)",
        100.0 * (med_g / med_l - 1.0),
        100.0 * (med_s / med_l - 1.0)
    );
    eprintln!(
        "vs plain GL (median): sym {:+.2}%",
        100.0 * (med_s / med_g - 1.0)
    );
    eprintln!(
        "token-shaped: one gate_up call = k_top×M×K = {:.1} MMAC; at 48 MoE layers\n\
         \x20             the median sym cost is {:.2} ms/token for gate_up alone.",
        (k_top * m * k) as f64 / 1e6,
        med_s * 48.0 / 1000.0
    );

    eprintln!(
        "\nREGIME: HIP-dispatch microbenchmark — INDICATIVE ONLY. Host launch latency\n\
         masks device effects and the fixed top-k keeps the ~{:.1} MiB working set\n\
         L2/MALL-resident, understating GL's byte advantage. A prior HIP microbench\n\
         put MQ2GL at -3.1% and nearly killed the format. Final acceptance is the\n\
         golden bundle (registry/redline-golden-v1.json) with HIP and PM4 arms.",
        b_gl / (1024.0 * 1024.0)
    );

    if !all_ok {
        std::process::exit(1);
    }
}
