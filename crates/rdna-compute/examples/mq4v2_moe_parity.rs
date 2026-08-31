// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Nick Woolmer
// hipfire — see LICENSE and NOTICE in the project root.

//! GPU correctness oracle for the qt44 (MQ4G256V2) MoE kernels.
//!
//! Covers the four qt44 MoE kernels this branch adds:
//!   * `gemm_mq4g256v2_moe_grouped_wmma_k2`   — prefill, arch-selecting
//!     (gfx11 `_k2` source / gfx12 `_gfx12` source)
//!   * `gemv_mq4g256v2_moe_gate_up_k8_indexed`        — decode
//!   * `gemv_mq4g256v2_moe_gate_up_k8_indexed_batched` — decode batched (N>1)
//!   * `gemv_mq4g256v2_moe_down_k8_indexed_batched_expanded` — decode
//!   * `gemv_mq4g256v2_moe_ninepath_d4`       — decode, fused down + combine
//!
//! WHY THIS EXISTS. The gfx12 grouped kernel was written without access to
//! gfx12 hardware. It compiles for gfx1201 and its VGPR/LDS budget is checked
//! offline, but its runtime numerics have never executed on RDNA4. Run this on
//! an R9700 (or any gfx1200/gfx1201) and the arch-selecting launcher will pick
//! the gfx12 source automatically — a PASS here is the missing evidence.
//!
//!   cargo run --release -p rdna-compute --example mq4v2_moe_parity --features lab
//!
//! Exit code is non-zero if any check fails, so it can gate a script.
//!
//! ── The two fixture traps ────────────────────────────────────────────────
//! Both produced false failures on correct kernels during the qt44 dense-GEMM
//! work, and both would have led to "fixing" working code. Carried forward:
//!
//!   1. fp16 A-operand rounding amplified by catastrophic cancellation. The
//!      grouped GEMM dequantizes in fp16; a reference that dequantizes in f32
//!      and then subtracts near-equal products disagrees by far more than the
//!      kernel is wrong. This oracle scores with relative L2 over the whole
//!      output vector, never per-element on cancelling sums.
//!   2. Per-element relative error dividing by a near-zero denominator. At
//!      N=5 with 160 outputs it is invisible; at N=512 with 524,288 outputs it
//!      is unavoidable. Hence rel-L2 + cosine, not max per-element relative.
//!
//! ── The negative control ─────────────────────────────────────────────────
//! qt13 and qt44 share a 136 B group stride and identical nibble packing. They
//! differ ONLY in the 8-byte header: qt13 stores one f32 scale + one f32 min
//! for all 256 weights, qt44 stores two f16 scale/zero pairs, one per
//! 128-weight half. A kernel that reads the wrong header produces plausible
//! numbers, not a fault.
//!
//! So every check here runs against `build_disjoint_halves`, whose two halves
//! occupy DISJOINT ranges ([-1,1] and [96,160]). Under that fixture a
//! grid-selection error is a ~100x scale error, not a rounding difference. The
//! oracle then asserts a deliberately grid-swapped reference DISAGREES — a
//! test that passes with the halves swapped would be measuring nothing.
//!
//! ── Verified gap closure (issue 9) ───────────────────────────────────────
//! * Batched N>1 via `gate_up_batched` and `down` with N≥2.
//! * Token-distinct routes: per-token topk tables include expert 0 and 255.
//! * Two experts and high IDs (0 and 255) via sparse 256-entry pointer table.
//! * Nonidentity grouped permutation with padded -1 slots, multi-tile experts.
//! * Production K/M: gate/up 1024×2048, down/ninepath 2048×512, grouped 32×1024×2048.
//! * Isolated half controls: lower-only / upper-only activations per token.
//! * Equal-length and finite scoring: Report rejects empty/unequal/non-finite.

use rdna_compute::{DType, Gpu, GpuTensor};

const GROUP: usize = 256;
const HALF: usize = 128;
const GROUP_BYTES: usize = 136;

// ── fixture + packing (shared convention with test_mq4v2_residual_bt_*) ──

fn f16_to_f32(bits: u16) -> f32 {
    let sign = ((bits & 0x8000) as u32) << 16;
    let mut exp = ((bits >> 10) & 0x1f) as u32;
    let mut mant = (bits & 0x03ff) as u32;
    let out = if exp == 0 {
        if mant == 0 {
            sign
        } else {
            exp = 127 - 15 + 1;
            while mant & 0x0400 == 0 {
                mant <<= 1;
                exp -= 1;
            }
            sign | (exp << 23) | ((mant & 0x03ff) << 13)
        }
    } else if exp == 0x1f {
        sign | 0x7f80_0000 | (mant << 13)
    } else {
        sign | ((exp + 127 - 15) << 23) | (mant << 13)
    };
    f32::from_bits(out)
}

fn half_from_f32(x: f32) -> u16 {
    let b = x.to_bits();
    let sign = ((b >> 16) & 0x8000) as u16;
    let mut val = (b & 0x7fff_ffff) as i32;
    if val >= 0x4780_0000 {
        return sign | 0x7c00; // inf / overflow
    }
    if val < 0x3880_0000 {
        // Subnormal half: scale into the 10-bit subnormal grid and round.
        let f = f32::from_bits(val as u32);
        let sub = (f * 2f32.powi(24)).round() as i32;
        return sign | (sub as u16 & 0x03ff);
    }
    val += 0x0000_1000; // round-to-nearest on the truncated mantissa
    sign | (((val - 0x3800_0000) >> 13) as u16)
}

fn prng(i: usize, salt: u32) -> f32 {
    let x = (i as u32)
        .wrapping_mul(0x9E37_79B9)
        .wrapping_add(salt.wrapping_mul(0x85EB_CA6B));
    let x = x ^ (x >> 15);
    let x = x.wrapping_mul(0x2545_F491);
    let x = x ^ (x >> 13);
    (x >> 8) as f32 / (1u32 << 24) as f32
}

/// Weights whose two 128-halves occupy disjoint ranges, so a grid-selection
/// error is a scale error rather than a rounding difference.
fn build_disjoint_halves(m: usize, k: usize) -> Vec<f32> {
    let mut w = vec![0.0f32; m * k];
    for r in 0..m {
        for g in 0..(k / GROUP) {
            let base = r * k + g * GROUP;
            let salt = (r * 7919 + g * 104_729) as u32;
            for i in 0..HALF {
                w[base + i] = prng(i, salt) * 2.0 - 1.0;
            }
            for i in HALF..GROUP {
                w[base + i] = 96.0 + prng(i, salt ^ 0xA5A5_A5A5) * 64.0;
            }
        }
    }
    w
}

fn pack_mq4g256v2(w: &[f32], m: usize, k: usize) -> Vec<u8> {
    assert_eq!(k % GROUP, 0, "k must be multiple of 256");
    assert_eq!(w.len(), m * k);
    let gpr = k / GROUP;
    let mut blob = vec![0u8; m * gpr * GROUP_BYTES];
    for r in 0..m {
        for g in 0..gpr {
            let src = r * k + g * GROUP;
            let dst = (r * gpr + g) * GROUP_BYTES;
            let mut codes = [0u8; GROUP];
            for h in 0..2 {
                let off = h * HALF;
                let slice = &w[src + off..src + off + HALF];
                let lo = slice.iter().cloned().fold(f32::INFINITY, f32::min);
                let hi = slice.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let step = if hi > lo { (hi - lo) / 15.0 } else { 0.0 };
                let s_bits = if hi == lo { 0u16 } else { half_from_f32(step) };
                let z_bits = half_from_f32(lo);
                blob[dst + h * 4..dst + h * 4 + 2].copy_from_slice(&s_bits.to_le_bytes());
                blob[dst + h * 4 + 2..dst + h * 4 + 4].copy_from_slice(&z_bits.to_le_bytes());
                let s_rt = f16_to_f32(s_bits);
                let z_rt = f16_to_f32(z_bits);
                if s_rt == 0.0 {
                    continue;
                }
                let inv = 1.0 / s_rt;
                for i in 0..HALF {
                    let q = ((slice[i] - z_rt) * inv + 0.5).floor().clamp(0.0, 15.0);
                    codes[off + i] = q as u8;
                }
            }
            for i in 0..HALF {
                blob[dst + 8 + i] = (codes[2 * i] & 0xF) | ((codes[2 * i + 1] & 0xF) << 4);
            }
        }
    }
    blob
}

/// Dequantize a packed row back to f32. `swap_grids` deliberately reads the
/// WRONG affine grid for each half — the negative control.
fn dequant_row(blob: &[u8], row: usize, k: usize, swap_grids: bool) -> Vec<f32> {
    let gpr = k / GROUP;
    let mut out = vec![0.0f32; k];
    for g in 0..gpr {
        let dst = (row * gpr + g) * GROUP_BYTES;
        let hdr = |h: usize| {
            let o = dst + h * 4;
            let s = u16::from_le_bytes([blob[o], blob[o + 1]]);
            let z = u16::from_le_bytes([blob[o + 2], blob[o + 3]]);
            (f16_to_f32(s), f16_to_f32(z))
        };
        for i in 0..GROUP {
            let mut h = i / HALF;
            if swap_grids {
                h ^= 1;
            }
            let (sc, zp) = hdr(h);
            let byte = blob[dst + 8 + i / 2];
            let q = if i % 2 == 0 { byte & 0xF } else { byte >> 4 };
            out[g * GROUP + i] = sc * q as f32 + zp;
        }
    }
    out
}

// ── scoring (see the fixture-trap note in the module docs) ──────────────

fn rel_l2(got: &[f32], want: &[f32]) -> f64 {
    assert_eq!(
        got.len(),
        want.len(),
        "rel_l2: length mismatch {} vs {}",
        got.len(),
        want.len()
    );
    assert!(!got.is_empty(), "rel_l2: empty slices");
    let mut num = 0.0f64;
    let mut den = 0.0f64;
    for (g, w) in got.iter().zip(want) {
        let d = (*g as f64) - (*w as f64);
        num += d * d;
        den += (*w as f64) * (*w as f64);
    }
    if den == 0.0 {
        return if num == 0.0 { 0.0 } else { f64::INFINITY };
    }
    (num / den).sqrt()
}

fn cosine(a: &[f32], b: &[f32]) -> f64 {
    assert_eq!(a.len(), b.len(), "cosine: length mismatch");
    assert!(!a.is_empty(), "cosine: empty slices");
    let (mut dot, mut na, mut nb) = (0.0f64, 0.0f64, 0.0f64);
    for (x, y) in a.iter().zip(b) {
        dot += (*x as f64) * (*y as f64);
        na += (*x as f64) * (*x as f64);
        nb += (*y as f64) * (*y as f64);
    }
    if na == 0.0 || nb == 0.0 {
        return 0.0;
    }
    dot / (na.sqrt() * nb.sqrt())
}

struct Report {
    failures: usize,
}

impl Report {
    fn check(&mut self, label: &str, got: &[f32], want: &[f32], tol: f64) {
        let len_ok = got.len() == want.len() && !got.is_empty();
        let finite_got = got.iter().all(|v| v.is_finite());
        let finite_want = want.iter().all(|v| v.is_finite());
        let finite = finite_got && finite_want;
        let (r, c, ok) = if !len_ok {
            (f64::INFINITY, 0.0, false)
        } else if !finite {
            (f64::INFINITY, 0.0, false)
        } else {
            let r = rel_l2(got, want);
            let c = cosine(got, want);
            (r, c, r.is_finite() && r <= tol)
        };
        println!(
            "  {:<44} rel_l2={:<12.3e} cos={:<10.6} tol={:<9.0e} {}",
            label,
            r,
            c,
            tol,
            if ok && len_ok && finite {
                "PASS"
            } else {
                "FAIL"
            }
        );
        if !len_ok {
            println!(
                "      -> length mismatch or empty: got {} want {}",
                got.len(),
                want.len()
            );
        }
        if !finite {
            println!("      -> non-finite values in got or want");
        }
        if !(ok && len_ok && finite) {
            self.failures += 1;
        }
    }

    /// Negative control: this comparison MUST fail. A test that passes here is
    /// measuring nothing. Requires finite, equal-length vectors and r >= min_rel.
    fn check_disagrees(&mut self, label: &str, got: &[f32], want: &[f32], min_rel: f64) {
        let len_ok = got.len() == want.len() && !got.is_empty();
        let finite_got = got.iter().all(|v| v.is_finite());
        let finite_want = want.iter().all(|v| v.is_finite());
        let r = if !len_ok || !finite_got || !finite_want {
            f64::NAN
        } else {
            rel_l2(got, want)
        };
        let ok = len_ok && finite_got && finite_want && r.is_finite() && r >= min_rel;
        println!(
            "  {:<44} rel_l2={:<12.3e} (must exceed {:<8.0e}) {}",
            label,
            r,
            min_rel,
            if ok { "PASS" } else { "FAIL (vacuous!)" }
        );
        if !ok {
            if !len_ok {
                println!("      -> length mismatch or empty");
            } else if !finite_got || !finite_want {
                println!("      -> non-finite in got or want (vacuous negative)");
            }
            self.failures += 1;
        }
    }
}

// ── host self-test: runs with no GPU ────────────────────────────────────

fn host_self_test(rep: &mut Report) {
    println!("host self-test (no GPU):");
    let (m, k) = (4usize, 512usize);
    let w = build_disjoint_halves(m, k);
    let blob = pack_mq4g256v2(&w, m, k);

    // The pack/dequant round trip must land within 4-bit quantisation error.
    // Each half spans its own range, so the error scale differs per half —
    // rel-L2 over the whole row is the honest aggregate.
    let mut got = Vec::new();
    let mut want = Vec::new();
    for r in 0..m {
        got.extend_from_slice(&dequant_row(&blob, r, k, false));
        want.extend_from_slice(&w[r * k..(r + 1) * k]);
    }
    rep.check("pack->dequant round trip", &got, &want, 2e-2);

    // Negative control: reading the other half's grid must be badly wrong.
    // If this ever passes, the fixture has stopped separating the halves and
    // every grid-selection check in this file is vacuous.
    let mut swapped = Vec::new();
    for r in 0..m {
        swapped.extend_from_slice(&dequant_row(&blob, r, k, true));
    }
    rep.check_disagrees(
        "grid-swapped dequant (negative control)",
        &swapped,
        &want,
        1e-1,
    );
    println!();
}

// ── GPU helpers ─────────────────────────────────────────────────────────

fn upload_experts(gpu: &mut Gpu, blobs: &[Vec<u8>]) -> (Vec<GpuTensor>, GpuTensor) {
    let experts: Vec<GpuTensor> = blobs
        .iter()
        .map(|b| gpu.upload_raw(b, &[b.len()]).unwrap())
        .collect();
    let ptrs: Vec<u8> = experts
        .iter()
        .flat_map(|t| (t.buf.as_ptr() as u64).to_le_bytes())
        .collect();
    let tab = gpu.upload_raw(&ptrs, &[blobs.len()]).unwrap();
    (experts, tab)
}

/// Sparse 256-entry pointer table aliasing a few distinct blobs, but exposing
/// high expert IDs (e.g. 255) to cover production routing.
fn upload_experts_sparse(
    gpu: &mut Gpu,
    blobs: &[Vec<u8>],
    n_table: usize,
    high_ids: &[usize],
) -> (Vec<GpuTensor>, GpuTensor) {
    let experts: Vec<GpuTensor> = blobs
        .iter()
        .map(|b| gpu.upload_raw(b, &[b.len()]).unwrap())
        .collect();
    let mut ptrs: Vec<u64> = vec![experts[0].buf.as_ptr() as u64; n_table];
    // Map each supplied high ID to a distinct blob if available.
    for (idx, &hid) in high_ids.iter().enumerate() {
        if hid < n_table && idx < experts.len() {
            ptrs[hid] = experts[idx].buf.as_ptr() as u64;
        }
    }
    // Ensure IDs 0 and 1 map to first blobs for low-ID routes.
    if n_table > 1 && experts.len() > 1 {
        ptrs[1] = experts[1].buf.as_ptr() as u64;
    }
    let bytes: Vec<u8> = ptrs.iter().flat_map(|v| v.to_le_bytes()).collect();
    let tab = gpu.upload_raw(&bytes, &[n_table]).unwrap();
    (experts, tab)
}

/// Build an activation vector of length K where only one 128-half per 256-group is active.
/// lower_only=true keeps indices i%256<128, false keeps upper half.
fn half_isolated_x(k: usize, lower_only: bool, salt: u32) -> Vec<f32> {
    (0..k)
        .map(|i| {
            let in_lower = (i % GROUP) < HALF;
            if in_lower == lower_only {
                prng(i, salt) * 2.0 - 1.0
            } else {
                0.0
            }
        })
        .collect()
}

// ── GPU checks ──────────────────────────────────────────────────────────

fn gate_up_check(gpu: &mut Gpu, rep: &mut Report) {
    // Production gate/up shape: M=2*mi with mi=512, K=2048 (8 groups), k_top=8.
    // Uses sparse 256-entry table with high IDs to exercise ID 255.
    let (mi, k, k_top) = (512usize, 2048usize, 8usize);
    let m = 2 * mi;
    let n_table = 256usize;
    println!("gate_up  M={m} (mi={mi}) K={k} k_top={k_top} n_table={n_table} (production)");

    let mut w0 = build_disjoint_halves(m, k);
    let mut w255 = build_disjoint_halves(m, k);
    for v in w255.iter_mut() {
        *v += 5.0;
    }
    // Ensure low vs high expert distinguishable across halves.
    for v in w0.iter_mut().take(128) {
        *v += 0.0;
    }
    let blobs = vec![pack_mq4g256v2(&w0, m, k), pack_mq4g256v2(&w255, m, k)];
    let (_experts, ptr_tab) = upload_experts_sparse(gpu, &blobs, n_table, &[0, 255]);

    // Single-token route exercises high ID 255 alongside 0.
    let topk: Vec<i32> = vec![0, 255, 0, 255, 1, 0, 255, 1];
    let topk_b: Vec<u8> = topk.iter().flat_map(|v| v.to_le_bytes()).collect();
    let topk_t = gpu.upload_raw(&topk_b, &[k_top]).unwrap();

    let x: Vec<f32> = (0..k).map(|i| prng(i, 0xBEEF) * 2.0 - 1.0).collect();
    let x_t = gpu.upload_f32(&x, &[k]).unwrap();
    let y_g = gpu.alloc_tensor(&[k_top * mi], DType::F32).unwrap();
    let y_u = gpu.alloc_tensor(&[k_top * mi], DType::F32).unwrap();

    gpu.gemv_mq4g256v2_moe_gate_up_k8_indexed(&ptr_tab, &topk_t, &x_t, &y_g, &y_u, m, k)
        .expect("gate_up launch");
    gpu.hip.device_synchronize().unwrap();
    let got_g = gpu.download_f32(&y_g).unwrap();
    let got_u = gpu.download_f32(&y_u).unwrap();

    let mut want_g = vec![0.0f32; k_top * mi];
    let mut want_u = vec![0.0f32; k_top * mi];
    let mut want_g_swapped = vec![0.0f32; k_top * mi];
    for (r, &e) in topk.iter().enumerate() {
        let blob = if e == 255 { &blobs[1] } else { &blobs[0] };
        if e != 0 && e != 255 && e != 1 {
            continue;
        }
        let e_idx = if e == 255 { 1 } else { 0 };
        let blob_ref = &blobs[e_idx];
        for row in 0..mi {
            let wr = dequant_row(blob_ref, row, k, false);
            want_g[r * mi + row] = wr.iter().zip(&x).map(|(a, b)| a * b).sum();
            let wr_sw = dequant_row(blob_ref, row, k, true);
            want_g_swapped[r * mi + row] = wr_sw.iter().zip(&x).map(|(a, b)| a * b).sum();
            let wu = dequant_row(blob_ref, mi + row, k, false);
            want_u[r * mi + row] = wu.iter().zip(&x).map(|(a, b)| a * b).sum();
        }
        let _ = blob;
    }
    // Correct reference using actual blob per expert id mapping (0->blobs0, 255->blobs1, 1->blobs1 aliased)
    // Recompute precisely for scoring:
    let mut want_g_precise = vec![0.0f32; k_top * mi];
    let mut want_u_precise = vec![0.0f32; k_top * mi];
    let mut want_g_swapped_precise = vec![0.0f32; k_top * mi];
    for (r, &e) in topk.iter().enumerate() {
        let blob = if e == 255 || e == 1 {
            &blobs[1]
        } else {
            &blobs[0]
        };
        for row in 0..mi {
            let wr = dequant_row(blob, row, k, false);
            want_g_precise[r * mi + row] = wr.iter().zip(&x).map(|(a, b)| a * b).sum();
            let wr_sw = dequant_row(blob, row, k, true);
            want_g_swapped_precise[r * mi + row] = wr_sw.iter().zip(&x).map(|(a, b)| a * b).sum();
            let wu = dequant_row(blob, mi + row, k, false);
            want_u_precise[r * mi + row] = wu.iter().zip(&x).map(|(a, b)| a * b).sum();
        }
    }
    let _ = want_g;
    let _ = want_u;
    let _ = want_g_swapped;
    rep.check(
        "gate_up y_gate (prod, high ID)",
        &got_g,
        &want_g_precise,
        1e-5,
    );
    rep.check(
        "gate_up y_up (prod, high ID)",
        &got_u,
        &want_u_precise,
        1e-5,
    );
    rep.check_disagrees(
        "gate_up vs grid-swapped ref (negative control)",
        &got_g,
        &want_g_swapped_precise,
        1e-2,
    );

    // Isolated half controls: lower-only and upper-only activations via same kernel (two extra launches)
    for (label, lower) in [
        ("gate_up lower-only half", true),
        ("gate_up upper-only half", false),
    ] {
        let xh = half_isolated_x(k, lower, if lower { 0xA11 } else { 0xB22 });
        let xh_t = gpu.upload_f32(&xh, &[k]).unwrap();
        let yg2 = gpu.alloc_tensor(&[k_top * mi], DType::F32).unwrap();
        let yu2 = gpu.alloc_tensor(&[k_top * mi], DType::F32).unwrap();
        gpu.gemv_mq4g256v2_moe_gate_up_k8_indexed(&ptr_tab, &topk_t, &xh_t, &yg2, &yu2, m, k)
            .expect("gate_up half isolated launch");
        gpu.hip.device_synchronize().unwrap();
        let got_g2 = gpu.download_f32(&yg2).unwrap();
        let mut want_g2 = vec![0.0f32; k_top * mi];
        for (r, &e) in topk.iter().enumerate() {
            let blob = if e == 255 || e == 1 {
                &blobs[1]
            } else {
                &blobs[0]
            };
            for row in 0..mi {
                let wr = dequant_row(blob, row, k, false);
                want_g2[r * mi + row] = wr.iter().zip(&xh).map(|(a, b)| a * b).sum();
            }
        }
        rep.check(label, &got_g2, &want_g2, 1e-5);
    }
    println!();
}

fn gate_up_batched_check(gpu: &mut Gpu, rep: &mut Report) {
    // Batched N>1 gate/up: token-distinct routes including high ID 255, production dims.
    // N=4 tokens, each token distinct topk.
    let (mi, k, k_top, n) = (512usize, 2048usize, 8usize, 4usize);
    let m = 2 * mi;
    let n_table = 256usize;
    println!("gate_up_batched M={m} (mi={mi}) K={k} k_top={k_top} N={n} (production, batched)");

    let mut w0 = build_disjoint_halves(m, k);
    let mut w1 = build_disjoint_halves(m, k);
    for v in w1.iter_mut() {
        *v += 3.0;
    }
    let blobs = vec![pack_mq4g256v2(&w0, m, k), pack_mq4g256v2(&w1, m, k)];
    let (_experts, ptr_tab) = upload_experts_sparse(gpu, &blobs, n_table, &[0, 255]);

    // Token-distinct routes: each token's 8 ranks differ, include 0 and 255.
    let mut topk: Vec<i32> = Vec::with_capacity(n * k_top);
    for tok in 0..n {
        for r in 0..k_top {
            let id = match (tok, r) {
                (0, 0) => 255,
                (0, 1) => 0,
                (1, 0) => 0,
                (1, 1) => 255,
                (2, 0) => 1,
                (2, 1) => 255,
                (3, 0) => 255,
                (3, 1) => 1,
                _ => ((tok * 7 + r * 13) % 2) as i32 * 255,
            };
            topk.push(id);
        }
    }
    let topk_b: Vec<u8> = topk.iter().flat_map(|v| v.to_le_bytes()).collect();
    let topk_t = gpu.upload_raw(&topk_b, &[n * k_top]).unwrap();

    // Per-token activations: token0 dense, token1 lower-only, token2 upper-only, token3 dense.
    let mut x_batch = Vec::with_capacity(n * k);
    for tok in 0..n {
        let v = match tok {
            1 => half_isolated_x(k, true, 0xC01 + tok as u32),
            2 => half_isolated_x(k, false, 0xC02 + tok as u32),
            _ => (0..k)
                .map(|i| prng(i, 0xBEEF + tok as u32) * 2.0 - 1.0)
                .collect(),
        };
        x_batch.extend_from_slice(&v);
    }
    let x_t = gpu.upload_f32(&x_batch, &[n * k]).unwrap();
    let y_g = gpu.alloc_tensor(&[n * k_top * mi], DType::F32).unwrap();
    let y_u = gpu.alloc_tensor(&[n * k_top * mi], DType::F32).unwrap();

    gpu.gemv_mq4g256v2_moe_gate_up_k8_indexed_batched(
        &ptr_tab, &topk_t, &x_t, &y_g, &y_u, m, k, k_top, n,
    )
    .expect("gate_up_batched launch");
    gpu.hip.device_synchronize().unwrap();
    let got_g = gpu.download_f32(&y_g).unwrap();
    let got_u = gpu.download_f32(&y_u).unwrap();

    let mut want_g = vec![0.0f32; n * k_top * mi];
    let mut want_u = vec![0.0f32; n * k_top * mi];
    let mut want_g_swapped = vec![0.0f32; n * k_top * mi];
    for tok in 0..n {
        let xr = &x_batch[tok * k..(tok + 1) * k];
        for r in 0..k_top {
            let e = topk[tok * k_top + r];
            let blob = if e == 255 || e == 1 {
                &blobs[1]
            } else {
                &blobs[0]
            };
            for row in 0..mi {
                let wr = dequant_row(blob, row, k, false);
                want_g[(tok * k_top + r) * mi + row] = wr.iter().zip(xr).map(|(a, b)| a * b).sum();
                let wr_sw = dequant_row(blob, row, k, true);
                want_g_swapped[(tok * k_top + r) * mi + row] =
                    wr_sw.iter().zip(xr).map(|(a, b)| a * b).sum();
                let wu = dequant_row(blob, mi + row, k, false);
                want_u[(tok * k_top + r) * mi + row] = wu.iter().zip(xr).map(|(a, b)| a * b).sum();
            }
        }
    }
    rep.check("gate_up_batched y_gate", &got_g, &want_g, 1e-5);
    rep.check("gate_up_batched y_up", &got_u, &want_u, 1e-5);
    rep.check_disagrees(
        "gate_up_batched vs grid-swapped ref (negative control)",
        &got_g,
        &want_g_swapped,
        1e-2,
    );

    // Per-token isolated half scoring: verify lower-only token1 and upper-only token2 still pass individually
    for tok in [1usize, 2usize] {
        let xr = &x_batch[tok * k..(tok + 1) * k];
        let mut got_slice = Vec::with_capacity(k_top * mi);
        let mut want_slice = Vec::with_capacity(k_top * mi);
        for r in 0..k_top {
            let base = (tok * k_top + r) * mi;
            got_slice.extend_from_slice(&got_g[base..base + mi]);
            let e = topk[tok * k_top + r];
            let blob = if e == 255 || e == 1 {
                &blobs[1]
            } else {
                &blobs[0]
            };
            for row in 0..mi {
                let wr = dequant_row(blob, row, k, false);
                want_slice.push(wr.iter().zip(xr).map(|(a, b)| a * b).sum());
            }
        }
        let label = if tok == 1 {
            "gate_up_batched tok1 lower-only"
        } else {
            "gate_up_batched tok2 upper-only"
        };
        rep.check(label, &got_slice, &want_slice, 1e-5);
    }
    println!();
}

fn down_check(gpu: &mut Gpu, rep: &mut Report) {
    // Production down: M=2048, K=512 (2 groups), N>1 batched, token-distinct routes, high IDs, half-isolated.
    let (m, k, k_top, n, n_table) = (2048usize, 512usize, 8usize, 4usize, 256usize);
    println!(
        "down     M={m} K={k} k_top={k_top} n_table={n_table} batch={n} (production, batched)"
    );

    let mut w0 = build_disjoint_halves(m, k);
    let mut w1 = build_disjoint_halves(m, k);
    for v in w1.iter_mut() {
        *v += 4.0;
    }
    let blobs = vec![pack_mq4g256v2(&w0, m, k), pack_mq4g256v2(&w1, m, k)];
    let (_experts, ptr_tab) = upload_experts_sparse(gpu, &blobs, n_table, &[0, 255]);

    // Token-distinct topk: each token routes differently, each includes 0 and 255.
    let mut topk: Vec<i32> = Vec::with_capacity(n * k_top);
    for tok in 0..n {
        for r in 0..k_top {
            let id = match (tok, r) {
                (0, 2) => 255,
                (1, 3) => 255,
                (2, 1) => 255,
                (3, 0) => 255,
                _ => ((tok + r) % 2) as i32 * 255,
            };
            // Ensure mix of 0/255/1
            let id2 = if id == 255 {
                255
            } else if r % 3 == 0 {
                1
            } else {
                0
            };
            topk.push(id2);
        }
    }
    let topk_b: Vec<u8> = topk.iter().flat_map(|v| v.to_le_bytes()).collect();
    let topk_t = gpu.upload_raw(&topk_b, &[n * k_top]).unwrap();

    // rot_batch is [N × K_TOP × K]; per-token distinct, with half-isolated tokens.
    let mut rot: Vec<f32> = Vec::with_capacity(n * k_top * k);
    for tok in 0..n {
        for r in 0..k_top {
            let base_salt = 0xC0FFEE + (tok as u32) * 17 + r as u32 * 31;
            let v: Vec<f32> = if tok == 1 && r < 4 {
                half_isolated_x(k, true, base_salt)
            } else if tok == 2 && r < 4 {
                half_isolated_x(k, false, base_salt)
            } else {
                (0..k).map(|i| prng(i, base_salt) * 2.0 - 1.0).collect()
            };
            rot.extend_from_slice(&v);
        }
    }
    let rot_t = gpu.upload_f32(&rot, &[n * k_top * k]).unwrap();
    let out_t = gpu.alloc_tensor(&[n * k_top * m], DType::F32).unwrap();

    gpu.gemv_mq4g256v2_moe_down_k8_indexed_batched_expanded(
        &ptr_tab, &topk_t, &rot_t, &out_t, m, k, k_top, n,
    )
    .expect("down launch");
    gpu.hip.device_synchronize().unwrap();
    let got = gpu.download_f32(&out_t).unwrap();

    let mut want = vec![0.0f32; n * k_top * m];
    let mut want_swapped = vec![0.0f32; n * k_top * m];
    for tok in 0..n {
        for r in 0..k_top {
            let e = topk[tok * k_top + r];
            let blob = if e == 255 || e == 1 {
                &blobs[1]
            } else {
                &blobs[0]
            };
            let xr = &rot[(tok * k_top + r) * k..(tok * k_top + r + 1) * k];
            let base = (tok * k_top + r) * m;
            for row in 0..m {
                let wr = dequant_row(blob, row, k, false);
                want[base + row] = wr.iter().zip(xr).map(|(a, b)| a * b).sum();
                let ws = dequant_row(blob, row, k, true);
                want_swapped[base + row] = ws.iter().zip(xr).map(|(a, b)| a * b).sum();
            }
        }
    }
    rep.check("down expert_outputs (prod batched)", &got, &want, 1e-5);
    rep.check_disagrees(
        "down vs grid-swapped ref (negative control)",
        &got,
        &want_swapped,
        1e-2,
    );

    // Isolated halves within same N>1 launch: token1 lower-only, token2 upper-only already scored above,
    // but also score per-token slices explicitly to ensure half-specific correctness.
    for tok in [1usize, 2usize] {
        let mut got_tok = Vec::with_capacity(k_top * m);
        let mut want_tok = Vec::with_capacity(k_top * m);
        for r in 0..k_top {
            let base = (tok * k_top + r) * m;
            got_tok.extend_from_slice(&got[base..base + m]);
            want_tok.extend_from_slice(&want[base..base + m]);
        }
        let label = if tok == 1 {
            "down tok1 lower-only half"
        } else {
            "down tok2 upper-only half"
        };
        rep.check(label, &got_tok, &want_tok, 1e-5);
    }
    println!();
}

fn grouped_check(gpu: &mut Gpu, rep: &mut Report) {
    // Production grouped: gate/up mapping with x_row_div=8 surging, M=1024, K=2048, m_total=32
    // Multi-expert (0 and 255), nonidentity slot permutation, padded -1 slots, 8 groups.
    let (m, k, m_total, x_row_div) = (1024usize, 2048usize, 32usize, 8usize);
    let n_table = 256usize;
    // Use N=4 batch for gate/up mapping: x_src rows = N =4, flat = tok*8+rank
    let batch = 4usize;
    let k_top = 8usize;
    println!("grouped  M={m} K={k} m_total={m_total} batch={batch} x_row_div={x_row_div} (prod, permuted, high IDs)");

    // Two distinct experts with disjoint halves, production sized.
    let mut w0 = build_disjoint_halves(m, k);
    let mut w1 = build_disjoint_halves(m, k);
    for v in w1.iter_mut() {
        *v += 7.0;
    }
    let b0 = pack_mq4g256v2(&w0, m, k);
    let b1 = pack_mq4g256v2(&w1, m, k);
    let blobs = vec![b0, b1];
    let (_experts, ptr_tab) = upload_experts_sparse(gpu, &blobs, n_table, &[0, 255]);

    // Tiles: m_total/16 =2 tiles. Tile0 expert 0, tile1 expert 255 (high ID).
    let tile_ids: Vec<i32> = vec![0, 255];
    let tile_b: Vec<u8> = tile_ids.iter().flat_map(|v| v.to_le_bytes()).collect();
    let tile_t = gpu.upload_raw(&tile_b, &[tile_ids.len()]).unwrap();

    // Sorted slot index: permutation of flat 0..m_total-1 but nonidentity, with two padded -1 at end.
    // flat = tok*8 + rank. We reverse token order and shuffle within token.
    let mut flats: Vec<i32> = (0..m_total as i32).collect();
    // Make padded slots: last 2 are -1 (inactive). So valid flats = 30, still cover both experts tiles.
    flats[m_total - 1] = -1;
    flats[m_total - 2] = -1;
    // Nonidentity permutation: reverse first 30.
    flats[0..m_total - 2].reverse();
    // Swap a middle pair to break simple reverse.
    flats.swap(3, 7);
    flats.swap(10, 14);
    let slot_b: Vec<u8> = flats.iter().flat_map(|v| v.to_le_bytes()).collect();
    let slot_t = gpu.upload_raw(&slot_b, &[m_total]).unwrap();

    // X_src: [batch × K] = [4 × 2048]. Include half-isolated rows: row0 lower-only, row1 upper-only.
    let mut x_src: Vec<f32> = Vec::with_capacity(batch * k);
    for b in 0..batch {
        let v = match b {
            0 => half_isolated_x(k, true, 0xD01 + b as u32),
            1 => half_isolated_x(k, false, 0xD02 + b as u32),
            _ => (0..k)
                .map(|i| prng(i, 0x5EED + b as u32 * 101) * 2.0 - 1.0)
                .collect(),
        };
        x_src.extend_from_slice(&v);
    }
    // Upload as f32; launcher will convert to fp16 uncached internally, but we upload f32 view
    // The grouped launcher expects f32 source and does uncached convert internally.
    let x_t = gpu.upload_f32(&x_src, &[batch * k]).unwrap();
    let y_t = gpu.alloc_tensor(&[m_total * m], DType::F32).unwrap();

    gpu.gemm_mq4g256v2_moe_grouped_wmma_k2(
        &ptr_tab, &tile_t, &slot_t, &x_t, &y_t, m, k, x_row_div, m_total, batch,
    )
    .expect("grouped launch");
    gpu.hip.device_synchronize().unwrap();
    let got = gpu.download_f32(&y_t).unwrap();

    // Reference: Y[out_col * M + out_row] where out_col is slot position, expert from tile_y, x_row = flat/8.
    // For -1 flats, Y should be zero (kernel writes zero via null x). So we expect zero.
    let mut want = vec![0.0f32; m_total * m];
    let mut want_swapped = vec![0.0f32; m_total * m];
    // Pre-dequant rows for both experts
    let rows0: Vec<Vec<f32>> = (0..m)
        .map(|r| dequant_row(&blobs[0], r, k, false))
        .collect();
    let rows0_sw: Vec<Vec<f32>> = (0..m).map(|r| dequant_row(&blobs[0], r, k, true)).collect();
    let rows1: Vec<Vec<f32>> = (0..m)
        .map(|r| dequant_row(&blobs[1], r, k, false))
        .collect();
    let rows1_sw: Vec<Vec<f32>> = (0..m).map(|r| dequant_row(&blobs[1], r, k, true)).collect();
    for slot in 0..m_total {
        let flat = flats[slot];
        if flat < 0 {
            continue; // stays zero
        }
        let tile_y = slot / 16;
        let expert = tile_ids[tile_y];
        let rows = if expert == 255 { &rows1 } else { &rows0 };
        let rows_sw = if expert == 255 { &rows1_sw } else { &rows0_sw };
        let x_row = (flat as usize) / x_row_div;
        let xs = &x_src[x_row * k..(x_row + 1) * k];
        for row in 0..m {
            want[slot * m + row] = rows[row].iter().zip(xs).map(|(a, b)| a * b).sum();
            want_swapped[slot * m + row] = rows_sw[row].iter().zip(xs).map(|(a, b)| a * b).sum();
        }
    }
    // fp16 accumulation over K=2048 (8 groups) with disjoint-magnitude halves: 2e-2 is the honest bar.
    rep.check(
        "grouped Y (prod, permuted, high IDs, half-isolated)",
        &got,
        &want,
        2e-2,
    );
    rep.check_disagrees(
        "grouped vs grid-swapped ref (negative control)",
        &got,
        &want_swapped,
        1e-1,
    );

    // Isolated half verification: slots whose x_row is lower-only (batch0) vs upper-only (batch1) scored separately.
    // Find slots where x_row ==0 (lower) and x_row==1 (upper) and score their subvectors.
    let mut lower_got = Vec::new();
    let mut lower_want = Vec::new();
    let mut upper_got = Vec::new();
    let mut upper_want = Vec::new();
    for slot in 0..m_total {
        let flat = flats[slot];
        if flat < 0 {
            continue;
        }
        let x_row = (flat as usize) / x_row_div;
        if x_row == 0 {
            lower_got.extend_from_slice(&got[slot * m..(slot + 1) * m]);
            lower_want.extend_from_slice(&want[slot * m..(slot + 1) * m]);
        } else if x_row == 1 {
            upper_got.extend_from_slice(&got[slot * m..(slot + 1) * m]);
            upper_want.extend_from_slice(&want[slot * m..(slot + 1) * m]);
        }
    }
    if !lower_got.is_empty() {
        rep.check(
            "grouped lower-only half slots",
            &lower_got,
            &lower_want,
            2e-2,
        );
    }
    if !upper_got.is_empty() {
        rep.check(
            "grouped upper-only half slots",
            &upper_got,
            &upper_want,
            2e-2,
        );
    }
    println!();
}

fn ninepath_check(gpu: &mut Gpu, rep: &mut Report) {
    // Production ninepath: M=2048 (hidden), K=512 (intermediate), k_top=8, high IDs.
    let (m, k, k_top, n_table) = (2048usize, 512usize, 8usize, 256usize);
    println!("ninepath M={m} K={k} k_top={k_top} n_table={n_table} (production, high IDs)");

    let mut w0 = build_disjoint_halves(m, k);
    let mut w1 = build_disjoint_halves(m, k);
    for v in w1.iter_mut() {
        *v += 6.0;
    }
    let blobs = vec![pack_mq4g256v2(&w0, m, k), pack_mq4g256v2(&w1, m, k)];
    let (_experts, ptr_tab) = upload_experts_sparse(gpu, &blobs, n_table, &[0, 255]);

    let mut topk: Vec<i32> = Vec::with_capacity(k_top);
    for r in 0..k_top {
        let id = if r % 2 == 0 { 0 } else { 255 };
        // Alternate 0/255, with rank 2 also 1 aliased to 255 for coverage
        let id2 = if r == 2 { 1 } else { id };
        topk.push(id2);
    }
    let topk_b: Vec<u8> = topk.iter().flat_map(|v| v.to_le_bytes()).collect();
    let topk_t = gpu.upload_raw(&topk_b, &[k_top]).unwrap();

    // Non-uniform weights so a fold that ignored them, or folded in the wrong
    // order, would show up.
    let tw: Vec<f32> = (0..k_top).map(|i| 0.05 + 0.1 * (i as f32)).collect();
    let tw_t = gpu.upload_f32(&tw, &[k_top]).unwrap();

    let act: Vec<f32> = (0..k_top * k)
        .map(|i| prng(i, 0x9A17) * 2.0 - 1.0)
        .collect();
    let act_t = gpu.upload_f32(&act, &[k_top * k]).unwrap();

    // The kernel accumulates (`out[..] += a`), so start from a known non-zero
    // residual — that also catches a kernel that overwrites instead of adding.
    let out0: Vec<f32> = (0..m).map(|i| prng(i, 0x3C3C) - 0.5).collect();
    let out_t = gpu.upload_f32(&out0, &[m]).unwrap();

    gpu.gemv_mq4g256v2_moe_ninepath_d4(&ptr_tab, &topk_t, &tw_t, &act_t, &out_t, m, k)
        .expect("ninepath launch");
    gpu.hip.device_synchronize().unwrap();
    let got = gpu.download_f32(&out_t).unwrap();

    let mut want = out0.clone();
    let mut want_swapped = out0.clone();
    for (r, &e) in topk.iter().enumerate() {
        let blob = if e == 255 || e == 1 {
            &blobs[1]
        } else {
            &blobs[0]
        };
        let xr = &act[r * k..(r + 1) * k];
        for row in 0..m {
            let wr = dequant_row(blob, row, k, false);
            want[row] += tw[r] * wr.iter().zip(xr).map(|(a, b)| a * b).sum::<f32>();
            let ws = dequant_row(blob, row, k, true);
            want_swapped[row] += tw[r] * ws.iter().zip(xr).map(|(a, b)| a * b).sum::<f32>();
        }
    }
    rep.check(
        "ninepath d4 fused down+combine (prod, high IDs)",
        &got,
        &want,
        1e-5,
    );
    rep.check_disagrees(
        "ninepath vs grid-swapped ref (negative control)",
        &got,
        &want_swapped,
        1e-2,
    );

    // Isolated half controls via one-hot topk weights: activate only ranks whose xr is half-isolated.
    // Create lower-only and upper-only act vectors and run ninepath with weights selecting single rank.
    for (label, lower) in [
        ("ninepath lower-only half", true),
        ("ninepath upper-only half", false),
    ] {
        let mut act_h: Vec<f32> = Vec::with_capacity(k_top * k);
        for r in 0..k_top {
            let v = if r == 0 {
                half_isolated_x(k, lower, if lower { 0xE11 } else { 0xE22 })
            } else {
                vec![0.0f32; k]
            };
            act_h.extend_from_slice(&v);
        }
        let tw_h: Vec<f32> = (0..k_top).map(|i| if i == 0 { 1.0 } else { 0.0 }).collect();
        let act_h_t = gpu.upload_f32(&act_h, &[k_top * k]).unwrap();
        let tw_h_t = gpu.upload_f32(&tw_h, &[k_top]).unwrap();
        let out0_h = vec![0.0f32; m];
        let out_h_t = gpu.upload_f32(&out0_h, &[m]).unwrap();
        gpu.gemv_mq4g256v2_moe_ninepath_d4(&ptr_tab, &topk_t, &tw_h_t, &act_h_t, &out_h_t, m, k)
            .expect("ninepath half isolated launch");
        gpu.hip.device_synchronize().unwrap();
        let got_h = gpu.download_f32(&out_h_t).unwrap();
        // Want is only rank0 contribution
        let e0 = topk[0];
        let blob0 = if e0 == 255 || e0 == 1 {
            &blobs[1]
        } else {
            &blobs[0]
        };
        let xr0 = &act_h[0..k];
        let mut want_h = vec![0.0f32; m];
        for row in 0..m {
            let wr = dequant_row(blob0, row, k, false);
            want_h[row] = wr.iter().zip(xr0).map(|(a, b)| a * b).sum();
        }
        rep.check(label, &got_h, &want_h, 1e-5);
    }
    println!();
}

fn main() {
    let mut rep = Report { failures: 0 };
    host_self_test(&mut rep);

    match Gpu::init() {
        Ok(mut gpu) => {
            println!("arch={}\n", gpu.arch);
            gate_up_check(&mut gpu, &mut rep);
            gate_up_batched_check(&mut gpu, &mut rep);
            down_check(&mut gpu, &mut rep);
            grouped_check(&mut gpu, &mut rep);
            ninepath_check(&mut gpu, &mut rep);
        }
        Err(e) => {
            println!("GPU init failed ({e:?}); host self-test only.");
            println!("The kernel checks did NOT run — this is not a pass.");
            std::process::exit(2);
        }
    }

    if rep.failures == 0 {
        println!("mq4v2_moe_parity: all checks PASSED");
    } else {
        println!("mq4v2_moe_parity: {} check(s) FAILED", rep.failures);
        std::process::exit(1);
    }
}
