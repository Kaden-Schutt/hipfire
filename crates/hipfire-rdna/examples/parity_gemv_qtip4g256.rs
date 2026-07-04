// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 hipfire contributors
// hipfire — see LICENSE and NOTICE in the project root.
//! Parity for the dense QTIP-4 decode GEMV (`gemv_qtip4g256`) vs a CPU
//! trellis-decode oracle. Per-group block = 132 B [f32 scale | 128 B of 4-bit
//! nibble symbols]; the codebook is the computed 1MAD hash (state→Gaussian) +
//! the build_codebook renorm affine — replicated here so the oracle matches the
//! on-device decode bit-for-bit. Random symbols exercise the kernel's decode +
//! matvec without needing the beam encoder.
//!
//!   cargo run --release -p hipfire-rdna --example parity_gemv_qtip4g256 [M K]

use hipfire_rdna::Gpu;

const STATE_BITS: u32 = 12;
const NUM_STATES: usize = 1 << STATE_BITS;
const STATE_MASK: u32 = (NUM_STATES as u32) - 1;

fn decode_1mad(state: u32) -> f32 {
    let x = (state as u64) & 0xFFFF_FFFF;
    let x = x.wrapping_mul(34_038_481).wrapping_add(76_625_530) & 0xFFFF_FFFF;
    let bs = (x & 0xFF) + ((x >> 8) & 0xFF) + ((x >> 16) & 0xFF) + ((x >> 24) & 0xFF);
    (bs as f32 - 510.0) / 147.800_537_109_375
}
fn build_codebook() -> Vec<f32> {
    let mut cb: Vec<f64> = (0..NUM_STATES as u32)
        .map(|s| decode_1mad(s) as f64)
        .collect();
    let mean = cb.iter().sum::<f64>() / cb.len() as f64;
    for v in cb.iter_mut() {
        *v -= mean;
    }
    let var = cb.iter().map(|v| v * v).sum::<f64>() / cb.len() as f64;
    let inv = if var > 0.0 { 1.0 / var.sqrt() } else { 1.0 };
    cb.iter().map(|v| (v * inv) as f32).collect()
}

fn lcg_u8(seed: u32, n: usize) -> Vec<u8> {
    let mut s = seed.max(1);
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(1_103_515_245).wrapping_add(12345) & 0x7fff_ffff;
            (s >> 13) as u8
        })
        .collect()
}
fn lcg_f32(seed: u32, n: usize) -> Vec<f32> {
    let mut s = seed.max(1);
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(1_103_515_245).wrapping_add(12345) & 0x7fff_ffff;
            -1.0 + (s as f32 / 2_147_483_648.0) * 2.0
        })
        .collect()
}

const BLK: usize = 132; // [f32 scale | 128 B nibble symbols]

fn main() {
    let mut a = std::env::args().skip(1);
    let m: usize = a.next().and_then(|s| s.parse().ok()).unwrap_or(64);
    let k: usize = a.next().and_then(|s| s.parse().ok()).unwrap_or(1024);
    assert_eq!(k % 256, 0, "K must be a multiple of 256");
    let ng = k / 256;
    let cb = build_codebook();

    let mut gpu = Gpu::init().unwrap();

    // Random 4-bit symbols [M*ng*256] and positive per-group scales [M*ng].
    let sym: Vec<u8> = lcg_u8(1, m * ng * 256).iter().map(|b| b & 0xf).collect();
    let sc: Vec<f32> = lcg_f32(2, m * ng)
        .iter()
        .map(|v| 0.02 + 0.03 * v.abs())
        .collect();

    // Pack the 132 B blocks: [f32 scale][128 nibble bytes], two symbols/byte.
    let mut blob = vec![0u8; m * ng * BLK];
    for r in 0..m {
        for g in 0..ng {
            let blk = (r * ng + g) * BLK;
            let scale = sc[r * ng + g];
            blob[blk..blk + 4].copy_from_slice(&scale.to_le_bytes());
            let sbase = (r * ng + g) * 256;
            for j in 0..128 {
                blob[blk + 4 + j] =
                    (sym[sbase + 2 * j] & 0xf) | ((sym[sbase + 2 * j + 1] & 0xf) << 4);
            }
        }
    }

    let ad = gpu.upload_raw(&blob, &[blob.len()]).unwrap();
    let x = lcg_f32(3, k);
    let xd = gpu
        .upload_raw(
            &x.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>(),
            &[1, k],
        )
        .unwrap();
    let yd = gpu.upload_raw(&vec![0u8; m * 4], &[1, m]).unwrap();

    gpu.gemv_qtip4g256(&ad, &xd, &yd, m, k).unwrap();
    gpu.device_synchronize().unwrap();
    let yg = gpu.download_f32(&yd).unwrap();

    // Oracle: per row, walk the 256-symbol trellis per group. At 4 bits/symbol
    // the 12-bit state is the last 3 symbols (leading zeros at group start):
    // state = (s[i-2]<<8 | s[i-1]<<4 | s[i]) & 0xFFF, value = scale*cb[state].
    let mut max_abs = 0.0f32;
    let mut max_mag = 0.0f32;
    for row in 0..m {
        let mut acc = 0.0f32;
        for g in 0..ng {
            let sbase = (row * ng + g) * 256;
            let scale = sc[row * ng + g];
            let (mut s2, mut s1) = (0u32, 0u32);
            for j in 0..256 {
                let q = sym[sbase + j] as u32;
                let state = ((s2 << 8) | (s1 << 4) | q) & STATE_MASK;
                acc += scale * cb[state as usize] * x[g * 256 + j];
                s2 = s1;
                s1 = q;
            }
        }
        max_abs = max_abs.max((yg[row] - acc).abs());
        max_mag = max_mag.max(acc.abs());
    }

    let tol = 1e-3 * max_mag.max(1.0);
    let pass = max_abs <= tol;
    println!(
        "parity_gemv_qtip4g256 M={m} K={k} on {}: max_abs={max_abs:.6} (mag={max_mag:.2}) -> {}",
        gpu.arch,
        if pass { "PASS" } else { "FAIL" }
    );
    if !pass {
        std::process::exit(1);
    }
}
