// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 hipfire contributors
// hipfire — see LICENSE and NOTICE in the project root.
//! Parity for `gemm_oq4_grouped_wmma` (Opus Quant W4A4 core): grouped
//! signed-int4 × int4 GEMM with per-group scale rescale, vs a CPU reference
//! that quantizes the same way. The kernel does exact iu4 integer dots + f32
//! rescale, so this is an exact-ish match (only f32 accumulation order differs).
//!
//!   cargo run --release -p hipfire-rdna --example parity_gemm_oq4_grouped_wmma [M K B]

use hipfire_rdna::Gpu;

fn lcg(seed: u32, n: usize) -> Vec<f32> {
    let mut s = seed.max(1);
    (0..n)
        .map(|i| {
            s = s.wrapping_mul(1_103_515_245).wrapping_add(12345) & 0x7fff_ffff;
            let base = (s as f32 / 2_147_483_648.0) - 0.5;
            if i % 97 == 0 {
                base * 9.0
            } else {
                base
            } // sparse outliers
        })
        .collect()
}

/// Symmetric int4 per-group quant. Returns (packed nibbles [rows,K/2], scales
/// [rows,K/group], dequant-able q values [rows,K] as i8 for the reference).
fn quant_i4(src: &[f32], rows: usize, k: usize, group: usize) -> (Vec<u8>, Vec<f32>, Vec<i8>) {
    let ng = k / group;
    let mut packed = vec![0u8; rows * (k / 2)];
    let mut scales = vec![0f32; rows * ng];
    let mut qvals = vec![0i8; rows * k];
    for r in 0..rows {
        for g in 0..ng {
            let g0 = g * group;
            let mut amax = 1e-12f32;
            for c in g0..g0 + group {
                amax = amax.max(src[r * k + c].abs());
            }
            let scale = amax / 7.0;
            scales[r * ng + g] = scale;
            for c in g0..g0 + group {
                let q = (src[r * k + c] / scale).round().clamp(-7.0, 7.0) as i8;
                qvals[r * k + c] = q;
            }
        }
        for j in (0..k).step_by(2) {
            let lo = (qvals[r * k + j] as u8) & 0xf;
            let hi = (qvals[r * k + j + 1] as u8) & 0xf;
            packed[r * (k / 2) + j / 2] = lo | (hi << 4);
        }
    }
    (packed, scales, qvals)
}

fn main() {
    let mut a = std::env::args().skip(1);
    let m: usize = a.next().and_then(|s| s.parse().ok()).unwrap_or(64);
    let k: usize = a.next().and_then(|s| s.parse().ok()).unwrap_or(512);
    let b: usize = a.next().and_then(|s| s.parse().ok()).unwrap_or(32);
    let group = 256usize;
    assert_eq!(k % group, 0, "K must be a multiple of {group}");

    let mut gpu = Gpu::init().unwrap();
    if !gpu.arch_caps.has_wmma_w32() {
        println!(
            "SKIP gemm_oq4_grouped_wmma parity: {} lacks wave32 WMMA",
            gpu.arch
        );
        return;
    }
    let ng = k / group;

    let w = lcg(1, m * k);
    let x = lcg(2, b * k);
    let (wp, ws, wq) = quant_i4(&w, m, k, group);
    let (xp, xs, xq) = quant_i4(&x, b, k, group);

    // CPU reference: Σ_g sw·sx · Σ_{k∈g} q_w·q_x  (exactly what the kernel does).
    let mut yref = vec![0.0f32; b * m];
    for bi in 0..b {
        for mi in 0..m {
            let mut acc = 0.0f32;
            for g in 0..ng {
                let g0 = g * group;
                let mut isum = 0i32;
                for c in g0..g0 + group {
                    isum += wq[mi * k + c] as i32 * xq[bi * k + c] as i32;
                }
                acc += isum as f32 * ws[mi * ng + g] * xs[bi * ng + g];
            }
            yref[bi * m + mi] = acc;
        }
    }

    // GPU
    let wd = gpu.upload_raw(&wp, &[m, k / 2]).unwrap();
    let wsd = gpu
        .upload_raw(
            &ws.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>(),
            &[m, ng],
        )
        .unwrap();
    let xd = gpu.upload_raw(&xp, &[b, k / 2]).unwrap();
    let xsd = gpu
        .upload_raw(
            &xs.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>(),
            &[b, ng],
        )
        .unwrap();
    let yd = gpu.upload_raw(&vec![0u8; b * m * 4], &[b, m]).unwrap();
    gpu.gemm_oq4_grouped_wmma(&wd, &wsd, &xd, &xsd, &yd, m, k, b, group)
        .unwrap();
    gpu.device_synchronize().unwrap();
    let yb = gpu.download_f32(&yd).unwrap();

    let mut max_abs = 0.0f32;
    let mut max_mag = 0.0f32;
    for i in 0..b * m {
        max_abs = max_abs.max((yb[i] - yref[i]).abs());
        max_mag = max_mag.max(yref[i].abs());
    }
    let tol = 1e-3 * max_mag.max(1.0);
    let pass = max_abs <= tol;
    println!(
        "gemm_oq4_grouped_wmma parity M={m} K={k} B={b} g={group} on {}: \
         max_abs={max_abs:.5} (max_mag={max_mag:.3}) tol={tol:.5} -> {}",
        gpu.arch,
        if pass { "PASS" } else { "FAIL" }
    );
    if !pass {
        std::process::exit(1);
    }
}
