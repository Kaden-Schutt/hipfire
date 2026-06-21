// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 hipfire contributors
// hipfire — see LICENSE and NOTICE in the project root.
//! Parity for `fused_qkvza_oq4_wmma`: the one-launch 4-way fused projection must
//! equal four separate (validated) `gemm_oq4_grouped_wmma` calls over the shared
//! int4 activation. Bit-exact expected. Uses qwen3.5-0.8b DeltaNet shapes by
//! default (qkv=6144, z=2048, beta=16, alpha=16, K=1024).
//!
//!   cargo run --release -p rdna-compute --example parity_fused_qkvza_oq4 [qkv z beta alpha K B]

use rdna_compute::{Gpu, GpuTensor};

fn lcg(seed: u32, n: usize) -> Vec<u8> {
    let mut s = seed.max(1);
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(1_103_515_245).wrapping_add(12345) & 0x7fff_ffff;
            (s >> 13) as u8
        })
        .collect()
}
fn lcgf(seed: u32, n: usize) -> Vec<u8> {
    let mut s = seed.max(1);
    (0..n)
        .flat_map(|_| {
            s = s.wrapping_mul(1_103_515_245).wrapping_add(12345) & 0x7fff_ffff;
            (0.01 + (s as f32 / 2_147_483_648.0) * 0.5).to_le_bytes()
        })
        .collect()
}
fn weight_buf(seed: u32, m: usize, k: usize, ng: usize) -> Vec<u8> {
    let mut buf = lcg(seed, m * (k / 2));
    buf.extend_from_slice(&lcgf(seed ^ 0x5a5a, m * ng));
    buf
}

fn main() {
    let mut a = std::env::args().skip(1);
    let qkv: usize = a.next().and_then(|s| s.parse().ok()).unwrap_or(6144);
    let z: usize = a.next().and_then(|s| s.parse().ok()).unwrap_or(2048);
    let beta: usize = a.next().and_then(|s| s.parse().ok()).unwrap_or(16);
    let alpha: usize = a.next().and_then(|s| s.parse().ok()).unwrap_or(16);
    let k: usize = a.next().and_then(|s| s.parse().ok()).unwrap_or(1024);
    let b: usize = a.next().and_then(|s| s.parse().ok()).unwrap_or(64);
    let group = 256usize;
    assert_eq!(k % group, 0);
    let ng = k / group;

    let mut gpu = Gpu::init().unwrap();
    if !gpu.arch_caps.has_wmma_w32() {
        println!("SKIP parity_fused_qkvza_oq4: {} lacks wave32 WMMA", gpu.arch);
        return;
    }

    let d_wqkv = gpu.upload_raw(&weight_buf(1, qkv, k, ng), &[1]).unwrap();
    let d_wz = gpu.upload_raw(&weight_buf(2, z, k, ng), &[1]).unwrap();
    let d_wbeta = gpu.upload_raw(&weight_buf(3, beta, k, ng), &[1]).unwrap();
    let d_walpha = gpu.upload_raw(&weight_buf(4, alpha, k, ng), &[1]).unwrap();
    let d_xq = gpu.upload_raw(&lcg(5, b * (k / 2)), &[b, k / 2]).unwrap();
    let d_xs = gpu.upload_raw(&lcgf(6, b * ng), &[b, ng]).unwrap();

    let mk = |gpu: &mut Gpu, m: usize| gpu.upload_raw(&vec![0u8; b * m * 4], &[b, m]).unwrap();
    let (yq, yz, yb, ya) = (mk(&mut gpu, qkv), mk(&mut gpu, z), mk(&mut gpu, beta), mk(&mut gpu, alpha));
    gpu.fused_qkvza_oq4_wmma(
        &d_wqkv, &d_wz, &d_wbeta, &d_walpha, &d_xq, &d_xs, &yq, &yz, &yb, &ya, qkv, z, beta,
        alpha, k, b, group,
    )
    .unwrap();
    gpu.device_synchronize().unwrap();
    let f_q = gpu.download_f32(&yq).unwrap();
    let f_z = gpu.download_f32(&yz).unwrap();
    let f_b = gpu.download_f32(&yb).unwrap();
    let f_a = gpu.download_f32(&ya).unwrap();

    // Reference: 4 separate grouped GEMMs (each already validated bit-exact).
    let refm = |gpu: &mut Gpu, w: &GpuTensor, m: usize| -> Vec<f32> {
        let ws = w.sub_offset(m * (k / 2), m * ng * 4);
        let y = gpu.upload_raw(&vec![0u8; b * m * 4], &[b, m]).unwrap();
        gpu.gemm_oq4_grouped_wmma(w, &ws, &d_xq, &d_xs, &y, m, k, b, group).unwrap();
        gpu.device_synchronize().unwrap();
        gpu.download_f32(&y).unwrap()
    };
    let r_q = refm(&mut gpu, &d_wqkv, qkv);
    let r_z = refm(&mut gpu, &d_wz, z);
    let r_b = refm(&mut gpu, &d_wbeta, beta);
    let r_a = refm(&mut gpu, &d_walpha, alpha);

    let mx = |f: &[f32], r: &[f32]| f.iter().zip(r).fold(0.0f32, |m, (&x, &y)| m.max((x - y).abs()));
    let (dq, dz, db, da) = (mx(&f_q, &r_q), mx(&f_z, &r_z), mx(&f_b, &r_b), mx(&f_a, &r_a));
    let pass = dq == 0.0 && dz == 0.0 && db == 0.0 && da == 0.0;
    println!(
        "parity_fused_qkvza_oq4 qkv={qkv} z={z} beta={beta} alpha={alpha} K={k} B={b} on {}: \
         Δ=({dq},{dz},{db},{da}) -> {}",
        gpu.arch,
        if pass { "PASS (bit-exact)" } else { "FAIL" }
    );
    if !pass {
        std::process::exit(1);
    }
}
