// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 hipfire contributors
// hipfire — see LICENSE and NOTICE in the project root.
//! Parity for `quantize_act_oq4` (Opus Quant W4A4 dynamic int4 activation
//! quantizer): GPU output vs a CPU reference using identical symmetric-int4
//! per-group quant. Validates the runtime quantizer that feeds
//! `gemm_oq4_grouped_wmma`.
//!
//!   cargo run --release -p rdna-compute --example parity_quantize_act_oq4 [B K]

use rdna_compute::Gpu;

fn lcg(seed: u32, n: usize) -> Vec<f32> {
    let mut s = seed.max(1);
    (0..n)
        .map(|i| {
            s = s.wrapping_mul(1_103_515_245).wrapping_add(12345) & 0x7fff_ffff;
            let base = (s as f32 / 2_147_483_648.0) - 0.5;
            if i % 53 == 0 {
                base * 12.0
            } else {
                base
            }
        })
        .collect()
}

fn main() {
    let mut a = std::env::args().skip(1);
    let b: usize = a.next().and_then(|s| s.parse().ok()).unwrap_or(64);
    let k: usize = a.next().and_then(|s| s.parse().ok()).unwrap_or(1024);
    let group = 256usize;
    assert_eq!(k % group, 0);
    let ng = k / group;

    let mut gpu = Gpu::init().unwrap();
    if !gpu.arch_caps.has_wmma_w32() {
        println!(
            "SKIP quantize_act_oq4 parity: {} lacks wave32 WMMA",
            gpu.arch
        );
        return;
    }

    let x = lcg(7, b * k);

    // CPU reference: symmetric int4 per group, packed signed nibbles + scales.
    let mut cpu_packed = vec![0u8; b * (k / 2)];
    let mut cpu_scales = vec![0f32; b * ng];
    for r in 0..b {
        for g in 0..ng {
            let g0 = g * group;
            let mut amax = 0.0f32;
            for c in g0..g0 + group {
                amax = amax.max(x[r * k + c].abs());
            }
            let scale = if amax > 0.0 { amax / 7.0 } else { 1.0 };
            cpu_scales[r * ng + g] = scale;
            let inv = if amax > 0.0 { 7.0 / amax } else { 0.0 };
            for j in (0..group).step_by(2) {
                let q = |c: usize| (x[r * k + c] * inv).round().clamp(-7.0, 7.0) as i8;
                let lo = (q(g0 + j) as u8) & 0xf;
                let hi = (q(g0 + j + 1) as u8) & 0xf;
                cpu_packed[r * (k / 2) + (g0 + j) / 2] = lo | (hi << 4);
            }
        }
    }

    // GPU.
    let xd = gpu
        .upload_raw(
            &x.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>(),
            &[b, k],
        )
        .unwrap();
    let xqd = gpu
        .upload_raw(&vec![0u8; b * (k / 2)], &[b, k / 2])
        .unwrap();
    let xsd = gpu.upload_raw(&vec![0u8; b * ng * 4], &[b, ng]).unwrap();
    gpu.quantize_act_oq4(&xd, &xqd, &xsd, b, k, group).unwrap();
    gpu.device_synchronize().unwrap();
    let gpu_packed = gpu.download_raw(&xqd, b * (k / 2)).unwrap();
    let gpu_scales = gpu.download_f32(&xsd).unwrap();

    // Scales: exact-ish.
    let mut max_scale_rel = 0.0f32;
    for i in 0..b * ng {
        let d = (gpu_scales[i] - cpu_scales[i]).abs() / cpu_scales[i].abs().max(1e-9);
        max_scale_rel = max_scale_rel.max(d);
    }
    // Nibbles: count mismatches (rare ±1 from .5-rounding edge cases tolerated).
    let mut nib_mismatch = 0usize;
    for i in 0..b * (k / 2) {
        let (gl, gh) = (gpu_packed[i] & 0xf, gpu_packed[i] >> 4);
        let (cl, ch) = (cpu_packed[i] & 0xf, cpu_packed[i] >> 4);
        if gl != cl {
            nib_mismatch += 1;
        }
        if gh != ch {
            nib_mismatch += 1;
        }
    }
    let total_nib = b * k;
    let frac = nib_mismatch as f64 / total_nib as f64;
    let pass = max_scale_rel < 1e-3 && frac < 1e-3;
    println!(
        "quantize_act_oq4 parity B={b} K={k} g={group} on {}: \
         max_scale_rel={max_scale_rel:.2e} nib_mismatch={nib_mismatch}/{total_nib} ({frac:.2e}) -> {}",
        gpu.arch,
        if pass { "PASS" } else { "FAIL" }
    );
    if !pass {
        std::process::exit(1);
    }
}
