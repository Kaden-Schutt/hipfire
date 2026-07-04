// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 hipfire contributors
// hipfire — see LICENSE and NOTICE in the project root.
//
//! End-to-end on-GPU validation of the W4A4 quality recipe through the fused
//! iu4 WMMA path (`gemm_iu4_i32_wmma`) on gfx1103.
//!
//! Front-end (CPU, offline+runtime): SmoothQuant per-channel migration →
//! FWHT-256 rotation → clip-search symmetric int4, per-K-group scales (g128).
//! The product X·Wᵀ is preserved by SmoothQuant and rotation, so we compare the
//! reconstructed GPU output against the original f32 reference.
//!
//! Grouped scales are realized by K-tiling: one fused iu4 GEMM per K-group, each
//! integer partial rescaled by scale_w[m,g]·scale_x[b,g] and accumulated in f32.
//! Confirms (a) GPU == CPU sim of the identical scheme, and (b) the recipe's
//! SQNR (~21 dB) holds on real hardware — vs ~9 dB for naive W4A4.
//!
//!   cargo run --release -p hipfire-rdna --example validate_w4a4_recipe [M K B]

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

/// Clip-search symmetric int4 per K-group. Returns (q in [-7,7] as i8 [rows,k],
/// scales [rows, k/group]).
fn quant_int4(src: &[f32], rows: usize, k: usize, group: usize) -> (Vec<i8>, Vec<f32>) {
    let qmax = 7.0f32;
    let grid = [1.0f32, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6];
    let ng = k / group;
    let mut q = vec![0i8; rows * k];
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
            for c in g0..g0 + group {
                let qq = (src[r * k + c] / scale).round().clamp(-qmax, qmax);
                q[r * k + c] = qq as i8;
            }
        }
    }
    (q, sc)
}

/// Pack one g-th group (group columns) of a [rows,k] int4 matrix into
/// [rows, group/2] bytes: byte = k_even | (k_odd<<4), signed two's-complement.
fn pack_group(q: &[i8], rows: usize, k: usize, group: usize, g: usize) -> Vec<u8> {
    let g0 = g * group;
    let mut out = vec![0u8; rows * (group / 2)];
    for r in 0..rows {
        for j in (0..group).step_by(2) {
            let lo = (q[r * k + g0 + j] as u8) & 0xf;
            let hi = (q[r * k + g0 + j + 1] as u8) & 0xf;
            out[r * (group / 2) + j / 2] = lo | (hi << 4);
        }
    }
    out
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
    let group = 128usize;
    assert_eq!(k % 256, 0, "K must be %256 (FWHT-256)");

    let mut gpu = Gpu::init().unwrap();
    if !gpu.arch_caps.has_wmma_w32() {
        println!("SKIP: {} lacks wave32 WMMA", gpu.arch);
        return;
    }

    let s1 = signs(42, 256);
    let s2 = signs(1042, 256);
    let w = gain(lcg_gauss(1, m * k), m, k, &hot(7, k, 8), 6.0);
    let x = gain(lcg_gauss(2, b * k), b, k, &hot(9, k, 16), 20.0);

    // f32 reference output Y[b,m] = sum_k X[b,k] W[m,k].
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

    // Front-end: SmoothQuant → rotation (product preserved).
    let (mut xf, mut wf) = smoothquant(&x, &w, b, m, k, 0.5);
    rotate(&mut wf, m, k, &s1, &s2);
    rotate(&mut xf, b, k, &s1, &s2);
    // Clip-search int4, per-group scales.
    let (qw, sw) = quant_int4(&wf, m, k, group); // sw [M, ng]
    let (qx, sx) = quant_int4(&xf, b, k, group); // sx [B, ng]
    let ng = k / group;

    // CPU sim of the identical grouped scheme (dequant both, f32 matmul).
    let mut ycpu = vec![0.0f32; b * m];
    for bi in 0..b {
        for mi in 0..m {
            let mut acc = 0.0f32;
            for g in 0..ng {
                let g0 = g * group;
                let s = sw[mi * ng + g] * sx[bi * ng + g];
                let mut isum = 0i32;
                for c in g0..g0 + group {
                    isum += qw[mi * k + c] as i32 * qx[bi * k + c] as i32;
                }
                acc += isum as f32 * s;
            }
            ycpu[bi * m + mi] = acc;
        }
    }

    // GPU: per-group fused iu4 GEMM, rescale + accumulate in f32.
    let mut ygpu = vec![0.0f32; b * m];
    for g in 0..ng {
        let wpacked = pack_group(&qw, m, k, group, g); // [M, group/2]
        let xpacked = pack_group(&qx, b, k, group, g); // [B, group/2]
        let wd = gpu.upload_raw(&wpacked, &[m, group / 2]).unwrap();
        let xd = gpu.upload_raw(&xpacked, &[b, group / 2]).unwrap();
        let yd = gpu.upload_raw(&vec![0u8; b * m * 4], &[b, m]).unwrap();
        gpu.gemm_iu4_i32_wmma(&wd, &xd, &yd, m, group, b).unwrap();
        gpu.device_synchronize().unwrap();
        let yb = gpu.download_raw(&yd, b * m * 4).unwrap();
        for bi in 0..b {
            let sxg = sx[bi * ng + g];
            for mi in 0..m {
                let isum = i32::from_le_bytes([
                    yb[(bi * m + mi) * 4],
                    yb[(bi * m + mi) * 4 + 1],
                    yb[(bi * m + mi) * 4 + 2],
                    yb[(bi * m + mi) * 4 + 3],
                ]);
                ygpu[bi * m + mi] += isum as f32 * sw[mi * ng + g] * sxg;
            }
        }
    }

    let cpu_db = sqnr(&ycpu, &yref);
    let gpu_db = sqnr(&ygpu, &yref);
    // GPU-vs-CPU agreement (should be ~exact: GPU int math is exact, only f32
    // rescale order differs).
    let mut max_rel = 0.0f64;
    for i in 0..b * m {
        let d = (ygpu[i] - ycpu[i]).abs() as f64;
        max_rel = max_rel.max(d / (ycpu[i].abs() as f64).max(1e-3));
    }

    println!(
        "validate_w4a4_recipe  M={m} K={k} B={b} group={group} on {}",
        gpu.arch
    );
    println!("recipe: SmoothQuant α0.5 → FWHT256 → clip-search int4 (g{group}), fused iu4 WMMA\n");
    println!("  CPU-sim output SQNR : {cpu_db:.2} dB");
    println!("  GPU    output SQNR : {gpu_db:.2} dB   (fused iu4, {ng} K-groups)");
    println!("  GPU vs CPU max-rel  : {max_rel:.2e}");
    let agree = (gpu_db - cpu_db).abs() < 0.5 && max_rel < 1e-3;
    println!(
        "\n  {} GPU realizes the recipe ({})",
        if agree { "PASS:" } else { "FAIL:" },
        if agree {
            "GPU==CPU, recipe SQNR holds on HW"
        } else {
            "GPU/CPU mismatch"
        }
    );
    if !agree {
        std::process::exit(1);
    }
}
