// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 hipfire contributors
// hipfire — see LICENSE and NOTICE in the project root.
//! Parity for the indexed-MoE OQ8G256 gate_up kernel
//! (`gemv_oq8g256_moe_gate_up_k8_indexed`) vs a CPU oracle.
//! Per-group block = 260 B [f32 scale | 256 int8], dequant w = sc * (int8).
//!
//!   cargo run --release -p hipfire-rdna --example parity_gemv_oq8g256_moe [M K]

use hipfire_rdna::Gpu;

fn lcg_i8(seed: u32, n: usize) -> Vec<i8> {
    let mut s = seed.max(1);
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(1_103_515_245).wrapping_add(12345) & 0x7fff_ffff;
            ((s >> 13) as i32 % 127) as i8
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

const BLK: usize = 260; // [f32 scale | 256 int8]

fn main() {
    let mut a = std::env::args().skip(1);
    let m: usize = a.next().and_then(|s| s.parse().ok()).unwrap_or(64);
    let k: usize = a.next().and_then(|s| s.parse().ok()).unwrap_or(1024);
    let group = 256usize;
    assert_eq!(k % group, 0);
    assert_eq!(m % 2, 0);
    let ng = k / group;
    let mi = m / 2;
    let n_exp = 8usize;
    let k_top = 8usize;

    let mut gpu = Gpu::init().unwrap();

    let mut expert_w8: Vec<Vec<i8>> = Vec::new();
    let mut expert_sc: Vec<Vec<f32>> = Vec::new();
    let mut expert_tensors = Vec::new();
    let mut ptrs: Vec<u64> = Vec::new();
    for e in 0..n_exp {
        let w8 = lcg_i8(1 + e as u32, m * ng * 256);
        let sc: Vec<f32> = lcgf(0x11 + e as u32, m * ng)
            .iter()
            .map(|v| 0.001 + v.abs() * 0.02)
            .collect();
        let mut blob = vec![0u8; m * ng * BLK];
        for r in 0..m {
            for g in 0..ng {
                let blk = (r * ng + g) * BLK;
                blob[blk..blk + 4].copy_from_slice(&sc[r * ng + g].to_le_bytes());
                let wsrc = (r * ng + g) * 256;
                for i in 0..256 {
                    blob[blk + 4 + i] = w8[wsrc + i] as u8;
                }
            }
        }
        let t = gpu.upload_raw(&blob, &[blob.len()]).unwrap();
        ptrs.push(t.buf.as_ptr() as u64);
        expert_tensors.push(t);
        expert_w8.push(w8);
        expert_sc.push(sc);
    }

    let ptr_bytes: Vec<u8> = ptrs.iter().flat_map(|p| p.to_le_bytes()).collect();
    let ptr_tensor = gpu.upload_raw(&ptr_bytes, &[n_exp]).unwrap();
    let topk: Vec<i32> = (0..k_top as i32).collect();
    let topk_bytes: Vec<u8> = topk.iter().flat_map(|i| i.to_le_bytes()).collect();
    let topk_tensor = gpu.upload_raw(&topk_bytes, &[k_top]).unwrap();
    let x = lcgf(3, k);
    let xbytes: Vec<u8> = x.iter().flat_map(|v| v.to_le_bytes()).collect();
    let xd = gpu.upload_raw(&xbytes, &[1, k]).unwrap();
    let y_gate = gpu
        .upload_raw(&vec![0u8; k_top * mi * 4], &[k_top, mi])
        .unwrap();
    let y_up = gpu
        .upload_raw(&vec![0u8; k_top * mi * 4], &[k_top, mi])
        .unwrap();

    gpu.gemv_oq8g256_moe_gate_up_k8_indexed(&ptr_tensor, &topk_tensor, &xd, &y_gate, &y_up, m, k)
        .unwrap();
    gpu.device_synchronize().unwrap();
    let yg = gpu.download_f32(&y_gate).unwrap();
    let yu = gpu.download_f32(&y_up).unwrap();

    let mut max_abs = 0.0f32;
    let mut max_mag = 0.0f32;
    for krank in 0..k_top {
        let e = topk[krank] as usize;
        for row in 0..m {
            let mut acc = 0.0f32;
            for g in 0..ng {
                let mut gsum = 0.0f32;
                let wbase = (row * ng + g) * 256;
                for j in 0..group {
                    gsum += expert_w8[e][wbase + j] as f32 * x[g * group + j];
                }
                acc += gsum * expert_sc[e][row * ng + g];
            }
            let got = if row < mi {
                yg[krank * mi + row]
            } else {
                yu[krank * mi + (row - mi)]
            };
            max_abs = max_abs.max((got - acc).abs());
            max_mag = max_mag.max(acc.abs());
        }
    }

    let tol = 1e-3 * max_mag.max(1.0);
    let pass = max_abs <= tol;
    println!(
        "parity_gemv_oq8g256_moe gate_up M={m} K={k} n_exp={n_exp} k_top={k_top} on {}: \
         max_abs={max_abs:.5} (mag={max_mag:.2}) -> {}",
        gpu.arch,
        if pass { "PASS" } else { "FAIL" }
    );
    if !pass {
        std::process::exit(1);
    }
}
