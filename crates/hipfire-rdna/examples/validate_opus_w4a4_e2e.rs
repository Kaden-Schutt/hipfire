#![allow(
    clippy::duplicated_attributes,
    clippy::doc_lazy_continuation,
    clippy::doc_overindented_list_items,
    clippy::explicit_counter_loop,
    clippy::field_reassign_with_default,
    clippy::manual_checked_ops,
    clippy::manual_clamp,
    clippy::manual_div_ceil,
    clippy::needless_range_loop,
    clippy::ptr_arg,
    clippy::same_item_push,
    clippy::too_many_arguments,
    clippy::type_complexity,
    clippy::unnecessary_cast,
    clippy::useless_vec,
    clippy::while_let_loop
)]
// hipfire example clippy sweep: examples are GPU probes/benches, not reusable APIs.

// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 hipfire contributors
// hipfire — see LICENSE and NOTICE in the project root.
//
//! End-to-end Opus Quant W4A4 capstone through the *dedicated* runtime kernels
//! (`quantize_act_oq4` + `gemm_oq4_grouped_wmma`) on gfx1103 — not the host-tiled
//! fused-iu4 simulation in `validate_w4a4_recipe`.
//!
//! Pipeline mirrors how the engine would run a projection at inference:
//!   offline (CPU): SmoothQuant α0.5 → FWHT-256 → clip-search symmetric int4 of W
//!   runtime (GPU): FWHT-256 of X (host here) → `quantize_act_oq4` quantizes X on
//!                  the device → `gemm_oq4_grouped_wmma` does grouped iu4·iu4 with
//!                  per-group scale_w·scale_x rescale, groups handled in-kernel.
//! SmoothQuant + rotation preserve X·Wᵀ, so the GPU Y is compared directly to the
//! original f32 reference. Confirms the dedicated path realizes the ~21 dB recipe
//! (vs ~9 dB naive W4A4) with the activation quantized entirely on-GPU.
//!
//!   cargo run --release -p hipfire-rdna --example validate_opus_w4a4_e2e [M K B]

#![allow(
    clippy::doc_lazy_continuation,
    clippy::doc_overindented_list_items,
    clippy::drop_non_drop,
    clippy::excessive_precision,
    clippy::identity_op,
    clippy::manual_div_ceil,
    clippy::manual_is_multiple_of,
    clippy::needless_range_loop,
    clippy::print_literal,
    clippy::redundant_closure,
    clippy::too_many_arguments,
    clippy::type_complexity,
    clippy::unusual_byte_groupings,
    clippy::useless_vec,
    clippy::unnecessary_cast
)]

use hipfire_primitives::fwht::{cpu_fwht_256, gen_fwht_signs as signs};
use hipfire_rdna::Gpu;

fn lcg_gauss(seed: u32, n: usize) -> Vec<f32> {
    let mut s = seed.max(1);
    let mut u = || {
        s = s.wrapping_mul(1_103_515_245).wrapping_add(12345) & 0x7fff_ffff;
        (s as f32 + 0.5) / 2_147_483_648.0
    };
    let mut out = Vec::with_capacity(n);
    while out.len() < n {
        let u1 = u().max(1e-7);
        let u2 = u();
        let r = (-2.0 * u1.ln()).sqrt();
        out.push(r * (std::f32::consts::TAU * u2).cos());
        if out.len() < n {
            out.push(r * (std::f32::consts::TAU * u2).sin());
        }
    }
    out
}
fn hot(seed: u32, k: usize, n: usize) -> Vec<bool> {
    let mut s = seed ^ 0x9e37_79b9;
    let mut h = vec![false; k];
    for _ in 0..n {
        s = s.wrapping_mul(1_103_515_245).wrapping_add(12345) & 0x7fff_ffff;
        h[(s as usize) % k] = true;
    }
    h
}
fn gain(mut w: Vec<f32>, rows: usize, k: usize, h: &[bool], g: f32) -> Vec<f32> {
    for r in 0..rows {
        for c in 0..k {
            if h[c] {
                w[r * k + c] *= g;
            }
        }
    }
    w
}
fn rotate(m: &mut [f32], rows: usize, k: usize, s1: &[f32], s2: &[f32]) {
    let mut buf = [0.0f32; 256];
    for r in 0..rows {
        for seg in 0..(k / 256) {
            let base = r * k + seg * 256;
            buf.copy_from_slice(&m[base..base + 256]);
            cpu_fwht_256(&mut buf, s1, s2);
            m[base..base + 256].copy_from_slice(&buf);
        }
    }
}
fn smoothquant(
    x: &[f32],
    w: &[f32],
    b: usize,
    m: usize,
    k: usize,
    alpha: f32,
) -> (Vec<f32>, Vec<f32>) {
    let mut xm = vec![1e-9f32; k];
    let mut wm = vec![1e-9f32; k];
    for r in 0..b {
        for c in 0..k {
            xm[c] = xm[c].max(x[r * k + c].abs());
        }
    }
    for r in 0..m {
        for c in 0..k {
            wm[c] = wm[c].max(w[r * k + c].abs());
        }
    }
    let s: Vec<f32> = (0..k)
        .map(|c| (xm[c].powf(alpha) / wm[c].powf(1.0 - alpha)).max(1e-6))
        .collect();
    let mut xo = x.to_vec();
    let mut wo = w.to_vec();
    for r in 0..b {
        for c in 0..k {
            xo[r * k + c] /= s[c];
        }
    }
    for r in 0..m {
        for c in 0..k {
            wo[r * k + c] *= s[c];
        }
    }
    (xo, wo)
}

/// Clip-search symmetric int4 of weights, per K-group. Returns packed nibbles
/// [rows, k/2] (byte = k_even | k_odd<<4, two's-comp) + scales [rows, k/group].
fn quant_w_i4(src: &[f32], rows: usize, k: usize, group: usize) -> (Vec<u8>, Vec<f32>) {
    let qmax = 7.0f32;
    let grid = [1.0f32, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6];
    let ng = k / group;
    let mut packed = vec![0u8; rows * (k / 2)];
    let mut sc = vec![0f32; rows * ng];
    for r in 0..rows {
        for g in 0..ng {
            let g0 = g * group;
            let mut amax = 1e-12f32;
            for c in g0..g0 + group {
                amax = amax.max(src[r * k + c].abs());
            }
            let (mut bc, mut be) = (1.0f32, f32::INFINITY);
            for &cl in &grid {
                let scale = cl * amax / qmax;
                let mut e = 0.0f32;
                for c in g0..g0 + group {
                    let qq = (src[r * k + c] / scale).round().clamp(-qmax, qmax);
                    let d = src[r * k + c] - qq * scale;
                    e += d * d;
                }
                if e < be {
                    be = e;
                    bc = cl;
                }
            }
            let scale = bc * amax / qmax;
            sc[r * ng + g] = scale;
            for j in (0..group).step_by(2) {
                let q = |c: usize| (src[r * k + c] / scale).round().clamp(-qmax, qmax) as i8;
                let lo = (q(g0 + j) as u8) & 0xf;
                let hi = (q(g0 + j + 1) as u8) & 0xf;
                packed[r * (k / 2) + (g0 + j) / 2] = lo | (hi << 4);
            }
        }
    }
    (packed, sc)
}

fn sqnr(rec: &[f32], yref: &[f32]) -> f64 {
    let mut sig = 0.0f64;
    let mut noise = 0.0f64;
    for (&r, &o) in rec.iter().zip(yref) {
        sig += (o as f64) * (o as f64);
        let d = o as f64 - r as f64;
        noise += d * d;
    }
    10.0 * (sig / noise.max(1e-30)).log10()
}

fn main() {
    let mut a = std::env::args().skip(1);
    let m: usize = a.next().and_then(|s| s.parse().ok()).unwrap_or(128);
    let k: usize = a.next().and_then(|s| s.parse().ok()).unwrap_or(2048);
    let b: usize = a.next().and_then(|s| s.parse().ok()).unwrap_or(64);
    let group = 256usize; // aligns FWHT-256 segment, quantize_act_oq4, grouped GEMM
    assert_eq!(k % group, 0, "K must be a multiple of {group}");

    let mut gpu = Gpu::init().unwrap();
    if !gpu.arch_caps.has_wmma_w32() {
        println!("SKIP: {} lacks wave32 WMMA", gpu.arch);
        return;
    }
    let ng = k / group;

    let s1 = signs(42, 256);
    let s2 = signs(1042, 256);
    let w = gain(lcg_gauss(1, m * k), m, k, &hot(7, k, 8), 6.0);
    let x = gain(lcg_gauss(2, b * k), b, k, &hot(9, k, 16), 20.0);

    // f32 reference Y[b,m] = Σ_k X·W.
    let mut yref = vec![0.0f32; b * m];
    for bi in 0..b {
        for mi in 0..m {
            let mut acc = 0.0f32;
            for ki in 0..k {
                acc += x[bi * k + ki] * w[mi * k + ki];
            }
            yref[bi * m + mi] = acc;
        }
    }

    // Naive W4A4 baseline (no SmoothQuant, no rotation): direct per-group int4 of
    // raw W and X, same grouped scheme on CPU — the ~9 dB floor we beat.
    let (nbw, nsw) = quant_w_i4(&w, m, k, group);
    let (nbx, nsx) = quant_w_i4(&x, b, k, group);
    let mut ynaive = vec![0.0f32; b * m];
    for bi in 0..b {
        for mi in 0..m {
            let mut acc = 0.0f32;
            for g in 0..ng {
                let g0 = g * group;
                let mut isum = 0i32;
                for c in g0..g0 + group {
                    let wq = (nbw[mi * (k / 2) + c / 2] >> ((c & 1) * 4)) & 0xf;
                    let xq = (nbx[bi * (k / 2) + c / 2] >> ((c & 1) * 4)) & 0xf;
                    let wq = ((wq as i8) << 4 >> 4) as i32; // sign-extend nibble
                    let xq = ((xq as i8) << 4 >> 4) as i32;
                    isum += wq * xq;
                }
                acc += isum as f32 * nsw[mi * ng + g] * nsx[bi * ng + g];
            }
            ynaive[bi * m + mi] = acc;
        }
    }
    let naive_db = sqnr(&ynaive, &yref);

    // Front-end: SmoothQuant → rotation (X·Wᵀ preserved).
    let (mut xf, mut wf) = smoothquant(&x, &w, b, m, k, 0.5);
    rotate(&mut wf, m, k, &s1, &s2);
    rotate(&mut xf, b, k, &s1, &s2);

    // Offline: clip-search int4 of W → upload packed + scales.
    let (wp, ws) = quant_w_i4(&wf, m, k, group);
    let wd = gpu.upload_raw(&wp, &[m, k / 2]).unwrap();
    let wsd = gpu
        .upload_raw(
            &ws.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>(),
            &[m, ng],
        )
        .unwrap();

    // Runtime: upload rotated X (f32) → quantize on GPU → grouped GEMM.
    let xd = gpu
        .upload_raw(
            &xf.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>(),
            &[b, k],
        )
        .unwrap();
    let xqd = gpu
        .upload_raw(&vec![0u8; b * (k / 2)], &[b, k / 2])
        .unwrap();
    let xsd = gpu.upload_raw(&vec![0u8; b * ng * 4], &[b, ng]).unwrap();
    gpu.quantize_act_oq4(&xd, &xqd, &xsd, b, k, group).unwrap();

    let yd = gpu.upload_raw(&vec![0u8; b * m * 4], &[b, m]).unwrap();
    gpu.gemm_oq4_grouped_wmma(&wd, &wsd, &xqd, &xsd, &yd, m, k, b, group)
        .unwrap();
    gpu.device_synchronize().unwrap();
    let ygpu = gpu.download_f32(&yd).unwrap();

    let gpu_db = sqnr(&ygpu, &yref);

    println!(
        "validate_opus_w4a4_e2e  M={m} K={k} B={b} group={group} on {}",
        gpu.arch
    );
    println!(
        "dedicated path: SmoothQuant α0.5 → FWHT256 → clip-search W int4 │ \
         GPU quantize_act_oq4(X) → gemm_oq4_grouped_wmma\n"
    );
    println!("  naive W4A4 (raw, grouped)  : {naive_db:6.2} dB   (no SQ, no rotation — the floor)");
    println!(
        "  Opus W4A4 (GPU dedicated)  : {gpu_db:6.2} dB   ({ng} K-groups, X quantized on-GPU)"
    );
    println!(
        "  recipe gain                : {:+6.2} dB",
        gpu_db - naive_db
    );
    // The recipe must clear a clear margin over naive and land in the ~18-23 dB band.
    let pass = gpu_db > naive_db + 6.0 && gpu_db > 16.0;
    println!(
        "\n  {} dedicated Opus W4A4 path realizes the quality recipe on HW",
        if pass { "PASS:" } else { "FAIL:" }
    );
    if !pass {
        std::process::exit(1);
    }
}
