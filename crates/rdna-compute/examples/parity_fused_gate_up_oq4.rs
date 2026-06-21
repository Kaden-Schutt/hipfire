// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 hipfire contributors
// hipfire — see LICENSE and NOTICE in the project root.
//! Parity for `fused_gate_up_oq4_wmma`: the one-launch fused gate+up must equal
//! two separate (already-validated) `gemm_oq4_grouped_wmma` calls over the same
//! shared int4 activation. Bit-exact expected (identical integer math + epilogue).
//!
//!   cargo run --release -p rdna-compute --example parity_fused_gate_up_oq4 [gate_m up_m K B]

use rdna_compute::Gpu;

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

/// Build a combined Oq4 weight buffer [packed nibbles M*(K/2) | f32 scales M*ng].
fn weight_buf(seed: u32, m: usize, k: usize, ng: usize) -> Vec<u8> {
    let mut buf = lcg(seed, m * (k / 2));
    buf.extend_from_slice(&lcgf(seed ^ 0x5a5a, m * ng));
    buf
}

fn main() {
    let mut a = std::env::args().skip(1);
    let gate_m: usize = a.next().and_then(|s| s.parse().ok()).unwrap_or(256);
    let up_m: usize = a.next().and_then(|s| s.parse().ok()).unwrap_or(128);
    let k: usize = a.next().and_then(|s| s.parse().ok()).unwrap_or(1024);
    let b: usize = a.next().and_then(|s| s.parse().ok()).unwrap_or(48);
    let group = 256usize;
    assert_eq!(k % group, 0);
    let ng = k / group;

    let mut gpu = Gpu::init().unwrap();
    if !gpu.arch_caps.has_wmma_w32() {
        println!("SKIP parity_fused_gate_up_oq4: {} lacks wave32 WMMA", gpu.arch);
        return;
    }

    let wg = weight_buf(1, gate_m, k, ng);
    let wu = weight_buf(2, up_m, k, ng);
    let xq = lcg(3, b * (k / 2));
    let xs = lcgf(4, b * ng);

    let wgd = gpu.upload_raw(&wg, &[wg.len()]).unwrap();
    let wud = gpu.upload_raw(&wu, &[wu.len()]).unwrap();
    let xqd = gpu.upload_raw(&xq, &[b, k / 2]).unwrap();
    let xsd = gpu.upload_raw(&xs, &[b, ng]).unwrap();

    // Fused.
    let ygd = gpu.upload_raw(&vec![0u8; b * gate_m * 4], &[b, gate_m]).unwrap();
    let yud = gpu.upload_raw(&vec![0u8; b * up_m * 4], &[b, up_m]).unwrap();
    gpu.fused_gate_up_oq4_wmma(&wgd, &wud, &xqd, &xsd, &ygd, &yud, gate_m, up_m, k, b, group)
        .unwrap();
    gpu.device_synchronize().unwrap();
    let yg_fused = gpu.download_f32(&ygd).unwrap();
    let yu_fused = gpu.download_f32(&yud).unwrap();

    // Reference: two separate grouped GEMMs with split scale views.
    let wgs = wgd.sub_offset(gate_m * (k / 2), gate_m * ng * 4);
    let wus = wud.sub_offset(up_m * (k / 2), up_m * ng * 4);
    let ygr = gpu.upload_raw(&vec![0u8; b * gate_m * 4], &[b, gate_m]).unwrap();
    let yur = gpu.upload_raw(&vec![0u8; b * up_m * 4], &[b, up_m]).unwrap();
    gpu.gemm_oq4_grouped_wmma(&wgd, &wgs, &xqd, &xsd, &ygr, gate_m, k, b, group).unwrap();
    gpu.gemm_oq4_grouped_wmma(&wud, &wus, &xqd, &xsd, &yur, up_m, k, b, group).unwrap();
    gpu.device_synchronize().unwrap();
    let yg_ref = gpu.download_f32(&ygr).unwrap();
    let yu_ref = gpu.download_f32(&yur).unwrap();

    let max = |f: &[f32], r: &[f32]| f.iter().zip(r).fold(0.0f32, |m, (&x, &y)| m.max((x - y).abs()));
    let dg = max(&yg_fused, &yg_ref);
    let du = max(&yu_fused, &yu_ref);
    let pass = dg == 0.0 && du == 0.0;
    println!(
        "parity_fused_gate_up_oq4 gate_m={gate_m} up_m={up_m} K={k} B={b} on {}: \
         max|Δgate|={dg} max|Δup|={du} -> {}",
        gpu.arch,
        if pass { "PASS (bit-exact)" } else { "FAIL" }
    );
    if !pass {
        std::process::exit(1);
    }
}
