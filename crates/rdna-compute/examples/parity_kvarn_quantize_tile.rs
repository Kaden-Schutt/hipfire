// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 hipfire contributors
// hipfire — see LICENSE and NOTICE in the project root.
//! Validation of the GPU `kvarn_quantize_tile` against the KVarN quality bar
//! (the `kvarn.rs` CPU oracle uses the same dequant + a cos-sim ≥ 0.995 gate).
//! Builds a tile with heavy per-row AND per-col variance spread (what KVarN's
//! Sinkhorn variance-normalization targets), GPU-quantizes to the on-device
//! record, host-dequants `(q*scale_abs[r]+zp_abs[r])*s_col[c]`, and checks (a)
//! cos-sim vs the original tile is high, and (b) it beats naive per-row 4-bit.
//!
//!   cargo run --release -p rdna-compute --example parity_kvarn_quantize_tile [r_dim c_dim]

use rdna_compute::Gpu;

fn f16_to_f32(bits: u16) -> f32 {
    let s = (bits >> 15) & 1;
    let e = (bits >> 10) & 0x1f;
    let m = bits & 0x3ff;
    let v = if e == 0 {
        (m as f32) * 2f32.powi(-24)
    } else if e == 31 {
        if m == 0 { f32::INFINITY } else { f32::NAN }
    } else {
        (1.0 + m as f32 / 1024.0) * 2f32.powi(e as i32 - 15)
    };
    if s == 1 { -v } else { v }
}

fn lcg_normal(seed: u32, n: usize) -> Vec<f32> {
    let mut s = seed.max(1);
    let mut u = || {
        s = s.wrapping_mul(1_103_515_245).wrapping_add(12345) & 0x7fff_ffff;
        (s as f32 + 0.5) / 2_147_483_648.0
    };
    (0..n)
        .map(|_| {
            let u1 = u().max(1e-7);
            let u2 = u();
            (-2.0 * u1.ln()).sqrt() * (std::f32::consts::TAU * u2).cos()
        })
        .collect()
}

fn cos_sim(a: &[f32], b: &[f32]) -> f64 {
    let (mut dot, mut na, mut nb) = (0.0f64, 0.0f64, 0.0f64);
    for (&x, &y) in a.iter().zip(b) {
        dot += x as f64 * y as f64;
        na += (x as f64).powi(2);
        nb += (y as f64).powi(2);
    }
    dot / (na.sqrt() * nb.sqrt()).max(1e-30)
}

fn main() {
    let mut a = std::env::args().skip(1);
    let r: usize = a.next().and_then(|s| s.parse().ok()).unwrap_or(128);
    let c: usize = a.next().and_then(|s| s.parse().ok()).unwrap_or(64);
    assert_eq!(c % 2, 0);

    let mut gpu = Gpu::init().unwrap();

    // Tile with heavy per-row and per-col variance spread.
    let base = lcg_normal(7, r * c);
    let mut tile = vec![0.0f32; r * c];
    for ri in 0..r {
        let row_scale = (0.02f32 * 50f32.powf(ri as f32 / r as f32)).max(1e-3); // 0.02..1.0
        for ci in 0..c {
            let col_scale = 0.1f32 * 30f32.powf(ci as f32 / c as f32); // 0.1..3.0
            tile[ri * c + ci] = base[ri * c + ci] * row_scale * col_scale;
        }
    }

    let record_bytes = (r * c).div_ceil(2) + r * 2 * 2 + c * 2;
    let td = gpu
        .upload_raw(&tile.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>(), &[r * c])
        .unwrap();
    let rd = gpu.upload_raw(&vec![0u8; record_bytes], &[record_bytes]).unwrap();
    gpu.kvarn_quantize_tile(&td, &rd, 1, r, c, record_bytes).unwrap();
    gpu.device_synchronize().unwrap();
    let rec = gpu.download_raw(&rd, record_bytes).unwrap();

    // Host dequant from the GPU record.
    let n = r * c;
    let qbytes = n.div_ceil(2);
    let off_scale = qbytes;
    let off_zp = off_scale + r * 2;
    let off_scol = off_zp + r * 2;
    let rd16 = |off: usize| f16_to_f32(u16::from_le_bytes([rec[off], rec[off + 1]]));
    let scale_abs: Vec<f32> = (0..r).map(|i| rd16(off_scale + i * 2)).collect();
    let zp_abs: Vec<f32> = (0..r).map(|i| rd16(off_zp + i * 2)).collect();
    let s_col: Vec<f32> = (0..c).map(|i| rd16(off_scol + i * 2)).collect();
    let mut deq = vec![0.0f32; n];
    for ri in 0..r {
        for ci in 0..c {
            let gi = ri * c + ci;
            let byte = rec[gi >> 1];
            let q = if gi & 1 == 0 { byte & 0xf } else { byte >> 4 } as f32;
            deq[gi] = (q * scale_abs[ri] + zp_abs[ri]) * s_col[ci];
        }
    }
    let cs = cos_sim(&deq, &tile);

    // GPU dequant kernel: must match the host dequant of the same record.
    let outd = gpu.upload_raw(&vec![0u8; n * 2], &[n]).unwrap();
    gpu.kvarn_dequant_tile(&rd, &outd, 1, r, c, record_bytes).unwrap();
    gpu.device_synchronize().unwrap();
    let outb = gpu.download_raw(&outd, n * 2).unwrap();
    let mut gpu_deq = vec![0.0f32; n];
    let mut max_deq_err = 0.0f32;
    for i in 0..n {
        gpu_deq[i] = f16_to_f32(u16::from_le_bytes([outb[i * 2], outb[i * 2 + 1]]));
        max_deq_err = max_deq_err.max((gpu_deq[i] - deq[i]).abs() / deq[i].abs().max(1e-4));
    }
    let cs_gpu_deq = cos_sim(&gpu_deq, &tile);

    // Naive per-row 4-bit (no variance-normalization) baseline.
    let mut naive = vec![0.0f32; n];
    for ri in 0..r {
        let (mut lo, mut hi) = (f32::INFINITY, f32::NEG_INFINITY);
        for ci in 0..c { lo = lo.min(tile[ri * c + ci]); hi = hi.max(tile[ri * c + ci]); }
        let sc = ((hi - lo) / 15.0).max(1e-8);
        for ci in 0..c {
            let q = ((tile[ri * c + ci] - lo) / sc).round().clamp(0.0, 15.0);
            naive[ri * c + ci] = q * sc + lo;
        }
    }
    let cs_naive = cos_sim(&naive, &tile);

    // GPU dequant must agree with host dequant (f16 round-trip tolerance) and
    // reproduce the same reconstruction quality.
    let pass = cs >= 0.99 && cs > cs_naive && max_deq_err < 5e-3 && (cs_gpu_deq - cs).abs() < 1e-3;
    println!(
        "parity_kvarn_quantize_tile r={r} c={c} on {}: GPU cos-sim={cs:.5}  naive-4bit={cs_naive:.5}  \
         deq-kernel cos-sim={cs_gpu_deq:.5} max-rel-err={max_deq_err:.2e}  -> {}",
        gpu.arch,
        if pass { "PASS" } else { "FAIL" }
    );
    if !pass {
        std::process::exit(1);
    }
}
