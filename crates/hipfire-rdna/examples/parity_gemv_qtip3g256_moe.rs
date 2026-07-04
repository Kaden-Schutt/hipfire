// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 hipfire contributors
// hipfire — see LICENSE and NOTICE in the project root.
//! Parity for the indexed-MoE QTIP3 gate_up kernel
//! (`gemv_qtip3g256_moe_gate_up_k8_indexed`) vs a CPU trellis-decode oracle.
//! Per-group block = 100 B [f32 scale | 96 B of 3-bit symbols]; the codebook is
//! the computed 1MAD hash (state→Gaussian) + the build_codebook renorm affine —
//! replicated here so the oracle matches the on-device decode bit-for-bit.
//!
//!   cargo run --release -p hipfire-rdna --example parity_gemv_qtip3g256_moe [M K]

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

fn lcg(seed: u32, n: usize) -> Vec<u8> {
    let mut s = seed.max(1);
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(1_103_515_245).wrapping_add(12345) & 0x7fff_ffff;
            (s >> 13) as u8
        })
        .collect()
}
fn lcgf(seed: u32, n: usize) -> Vec<f32> {
    let mut s = seed.max(1);
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(1_103_515_245).wrapping_add(12345) & 0x7fff_ffff;
            -1.0 + (s as f32 / 2_147_483_648.0) * 2.0
        })
        .collect()
}

const BLK: usize = 100; // [f32 scale | 96 B symbols]

fn main() {
    let mut a = std::env::args().skip(1);
    let m: usize = a.next().and_then(|s| s.parse().ok()).unwrap_or(64);
    let k: usize = a.next().and_then(|s| s.parse().ok()).unwrap_or(1024);
    assert_eq!(k % 256, 0);
    assert_eq!(m % 2, 0);
    let ng = k / 256;
    let mi = m / 2;
    let n_exp = 8usize;
    let k_top = 8usize;
    let cb = build_codebook();

    let mut gpu = Gpu::init().unwrap();

    // Per-expert: random 3-bit symbols + scales; build the 100 B blocks; keep
    // symbols/scales for the oracle.
    let mut e_sym: Vec<Vec<u8>> = Vec::new(); // [e] -> M*ng*256 symbols (0..7)
    let mut e_sc: Vec<Vec<f32>> = Vec::new(); // [e] -> M*ng scales
    let mut tensors = Vec::new();
    let mut ptrs: Vec<u64> = Vec::new();
    for e in 0..n_exp {
        let raw = lcg(1 + e as u32, m * ng * 256);
        let sym: Vec<u8> = raw.iter().map(|&b| b & 7).collect();
        let sc: Vec<f32> = lcgf(0x11 + e as u32, m * ng)
            .iter()
            .map(|v| 0.01 + v.abs() * 0.25)
            .collect();
        let mut blob = vec![0u8; m * ng * BLK];
        for r in 0..m {
            for g in 0..ng {
                let blk = (r * ng + g) * BLK;
                blob[blk..blk + 4].copy_from_slice(&sc[r * ng + g].to_le_bytes());
                let sbase = (r * ng + g) * 256;
                // 32 chunks of 8 symbols → 3 bytes each (symbol n at bit 3n).
                for c in 0..32 {
                    let mut pk = 0u32;
                    for n in 0..8 {
                        pk |= (sym[sbase + c * 8 + n] as u32) << (3 * n);
                    }
                    let b = pk.to_le_bytes();
                    blob[blk + 4 + c * 3] = b[0];
                    blob[blk + 4 + c * 3 + 1] = b[1];
                    blob[blk + 4 + c * 3 + 2] = b[2];
                }
            }
        }
        let t = gpu.upload_raw(&blob, &[blob.len()]).unwrap();
        ptrs.push(t.buf.as_ptr() as u64);
        tensors.push(t);
        e_sym.push(sym);
        e_sc.push(sc);
    }

    let ptr_t = gpu
        .upload_raw(
            &ptrs
                .iter()
                .flat_map(|p| p.to_le_bytes())
                .collect::<Vec<u8>>(),
            &[n_exp],
        )
        .unwrap();
    let topk: Vec<i32> = (0..k_top as i32).collect();
    let topk_t = gpu
        .upload_raw(
            &topk
                .iter()
                .flat_map(|i| i.to_le_bytes())
                .collect::<Vec<u8>>(),
            &[k_top],
        )
        .unwrap();
    let x = lcgf(3, k);
    let xd = gpu
        .upload_raw(
            &x.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>(),
            &[1, k],
        )
        .unwrap();
    let yg = gpu
        .upload_raw(&vec![0u8; k_top * mi * 4], &[k_top, mi])
        .unwrap();
    let yu = gpu
        .upload_raw(&vec![0u8; k_top * mi * 4], &[k_top, mi])
        .unwrap();

    gpu.gemv_qtip3g256_moe_gate_up_k8_indexed(&ptr_t, &topk_t, &xd, &yg, &yu, m, k)
        .unwrap();
    gpu.device_synchronize().unwrap();
    let ygv = gpu.download_f32(&yg).unwrap();
    let yuv = gpu.download_f32(&yu).unwrap();

    // Oracle: per (expert,row), walk the 256-symbol trellis per group (state =
    // last 4 symbols, leading zeros at group start), value = scale*cb[state].
    let mut max_abs = 0.0f32;
    let mut max_mag = 0.0f32;
    for krank in 0..k_top {
        let e = topk[krank] as usize;
        for row in 0..m {
            let mut acc = 0.0f32;
            for g in 0..ng {
                let sbase = (row * ng + g) * 256;
                let scale = e_sc[e][row * ng + g];
                let mut s3 = 0u32;
                let mut s2 = 0u32;
                let mut s1 = 0u32;
                for j in 0..256 {
                    let q = e_sym[e][sbase + j] as u32;
                    let state = ((s3 << 9) | (s2 << 6) | (s1 << 3) | q) & STATE_MASK;
                    acc += scale * cb[state as usize] * x[g * 256 + j];
                    s3 = s2;
                    s2 = s1;
                    s1 = q;
                }
            }
            let got = if row < mi {
                ygv[krank * mi + row]
            } else {
                yuv[krank * mi + (row - mi)]
            };
            max_abs = max_abs.max((got - acc).abs());
            max_mag = max_mag.max(acc.abs());
        }
    }

    let tol = 1e-3 * max_mag.max(1.0);
    let pass = max_abs <= tol;
    println!(
        "parity_gemv_qtip3g256_moe gate_up M={m} K={k} n_exp={n_exp} k_top={k_top} on {}: \
         max_abs={max_abs:.5} (mag={max_mag:.2}) -> {}",
        gpu.arch,
        if pass { "PASS" } else { "FAIL" }
    );
    if !pass {
        std::process::exit(1);
    }
}
