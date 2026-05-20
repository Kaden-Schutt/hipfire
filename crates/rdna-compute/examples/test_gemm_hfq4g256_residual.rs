// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Correctness + perf test for `gemm_hfq4g256_residual` (batched GEMM
//! with fused residual add).
//!
//! Calls `gemv_hfq4g256_residual` N times and `gemm_hfq4g256_residual`
//! once with batch=N on the same random HFQ4-G256 weights and the same
//! input activation batch. The outputs MUST be bitwise identical — any
//! divergence means the batched kernel would fail the MQ4 quality gate.
//!
//! Then measures wall time for each path at several N values so we can
//! see at what batch size the batched kernel starts winning.
//!
//! Usage: cargo run --release --example test_gemm_hfq4g256_residual \
//!        -p rdna-compute -- [M] [K] [N1 N2 ...]
//!
//! Defaults: M=4096, K=1024, N=[1, 4, 8, 16, 32, 64].

use rdna_compute::{DType, Gpu};
use std::time::Instant;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let m: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(4096);
    let k: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(1024);
    let n_list: Vec<usize> = if args.len() > 3 {
        args[3..].iter().filter_map(|s| s.parse().ok()).collect()
    } else {
        vec![1, 4, 8, 16, 32, 64]
    };

    assert!(k % 256 == 0, "K must be a multiple of 256 for HFQ4-G256");
    let groups_per_row = k / 256;
    let row_bytes = groups_per_row * 136;

    eprintln!("=== gemm_hfq4g256_residual test: M={m} K={k} ===");
    eprintln!("groups_per_row={groups_per_row}, row_bytes={row_bytes}");
    eprintln!("weight tensor size: {} bytes ({:.2} MiB)",
        m * row_bytes, (m * row_bytes) as f64 / (1024.0 * 1024.0));

    let mut gpu = Gpu::init().expect("gpu init");

    // ── Random HFQ4-G256 weight buffer. Deterministic PRNG seeded with
    // a constant so runs are reproducible.
    let weight_bytes: Vec<u8> = synth_hfq4g256_weights(m, groups_per_row, 0xC0DE_FACEu64);
    let a_raw = gpu.upload_raw(&weight_bytes, &[m * row_bytes]).expect("upload weights");

    // ── Host-side activation batch & residual, sized to the max N used.
    let max_n = *n_list.iter().max().unwrap();
    let x_host: Vec<f32> = (0..max_n * k)
        .map(|i| {
            let v = ((i as i64).wrapping_mul(1103515245).wrapping_add(12345)) as f32;
            (v * 1e-9) % 2.0 - 1.0
        })
        .collect();
    let y_init_host: Vec<f32> = (0..max_n * m)
        .map(|i| {
            let v = ((i as i64).wrapping_mul(2147483647).wrapping_add(7)) as f32;
            (v * 1e-7) % 1.0
        })
        .collect();

    // ── Scratch buffers for the GEMV path (single-row x and y).
    let x_gemv = gpu.alloc_tensor(&[k], DType::F32).expect("alloc x_gemv");
    let y_gemv_scratch = gpu.alloc_tensor(&[m], DType::F32).expect("alloc y_gemv_scratch");
    let y_gemv_collected = gpu.alloc_tensor(&[max_n * m], DType::F32).expect("alloc y_gemv_collected");

    // ── Batch buffers for the GEMM path.
    let x_gemm = gpu.alloc_tensor(&[max_n * k], DType::F32).expect("alloc x_gemm");
    let y_gemm = gpu.alloc_tensor(&[max_n * m], DType::F32).expect("alloc y_gemm");

    // Upload the whole x batch once (both paths read the same activations).
    gpu.hip.memcpy_htod(&x_gemm.buf, bytes_of(&x_host)).unwrap();

    for &n in &n_list {
        eprintln!("\n--- N = {n} ---");

        // ─────── GEMV × N path ───────
        // Rebuild the GEMV path's state for each batch element:
        //  1. Upload y_init_host[i*m..(i+1)*m] → y_gemv_scratch
        //  2. Upload x_host[i*k..(i+1)*k] → x_gemv
        //  3. Run gemv_hfq4g256_residual → accumulates into y_gemv_scratch
        //  4. dtod-copy y_gemv_scratch into y_gemv_collected[i*m..(i+1)*m]
        //
        // Time just the N kernel launches + their post-launch syncs. The
        // memcpy prelude is untimed because it's test scaffolding that
        // wouldn't exist in the real prefill path.
        let mut gemv_kernel_us: f64 = 0.0;
        for i in 0..n {
            gpu.hip.memcpy_htod(
                &y_gemv_scratch.buf,
                bytes_of(&y_init_host[i * m..(i + 1) * m]),
            ).unwrap();
            gpu.hip.memcpy_htod(
                &x_gemv.buf,
                bytes_of(&x_host[i * k..(i + 1) * k]),
            ).unwrap();
            gpu.hip.device_synchronize().unwrap();

            let t = Instant::now();
            gpu.gemv_hfq4g256_residual(&a_raw, &x_gemv, &y_gemv_scratch, m, k).unwrap();
            gpu.hip.device_synchronize().unwrap();
            gemv_kernel_us += t.elapsed().as_secs_f64() * 1e6;

            gpu.hip.memcpy_dtod_at(
                &y_gemv_collected.buf, i * m * 4,
                &y_gemv_scratch.buf, 0,
                m * 4,
            ).unwrap();
        }

        // ─────── GEMM × 1 path ───────
        // Reset y_gemm to residual init values, then fire once.
        gpu.hip.memcpy_htod(
            &y_gemm.buf,
            bytes_of(&y_init_host[..n * m]),
        ).unwrap();
        gpu.hip.device_synchronize().unwrap();

        let t = Instant::now();
        gpu.gemm_hfq4g256_residual(&a_raw, &x_gemm, &y_gemm, m, k, n).unwrap();
        gpu.hip.device_synchronize().unwrap();
        let gemm_kernel_us = t.elapsed().as_secs_f64() * 1e6;

        // ─────── Byte-exact compare ───────
        let gemv_out = gpu.download_f32(&y_gemv_collected).unwrap()[..n * m].to_vec();
        let gemm_out = gpu.download_f32(&y_gemm).unwrap()[..n * m].to_vec();

        let mut first_divergent: Option<(usize, f32, f32)> = None;
        for i in 0..n * m {
            if gemv_out[i].to_bits() != gemm_out[i].to_bits() {
                first_divergent = Some((i, gemv_out[i], gemm_out[i]));
                break;
            }
        }

        let correct = first_divergent.is_none();
        let status = if correct { "byte-exact OK" } else { "DIVERGENT" };
        let speedup = gemv_kernel_us / gemm_kernel_us;
        eprintln!(
            "  gemv × {n}: {:8.1} µs   gemm × 1: {:8.1} µs   speedup: {:5.2}x   [{status}]",
            gemv_kernel_us, gemm_kernel_us, speedup
        );

        if let Some((i, a, b)) = first_divergent {
            let batch = i / m;
            let row = i % m;
            eprintln!(
                "  first divergent element: batch={batch} row={row}  gemv={a:.6e} ({:#010x})  gemm={b:.6e} ({:#010x})",
                a.to_bits(), b.to_bits()
            );
            let diverge_count: usize = gemv_out.iter().zip(gemm_out.iter())
                .filter(|(a, b)| a.to_bits() != b.to_bits())
                .count();
            eprintln!("  total divergent: {diverge_count}/{} elements", n * m);
            std::process::exit(1);
        }
    }

    eprintln!("\n=== All N passed byte-exact ===");
}

fn synth_hfq4g256_weights(m: usize, groups_per_row: usize, seed: u64) -> Vec<u8> {
    let total = m * groups_per_row * 136;
    let mut out = vec![0u8; total];
    let mut state = seed;
    let mut next = || {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (state >> 33) as u32
    };
    for row in 0..m {
        for g in 0..groups_per_row {
            let gp = (row * groups_per_row + g) * 136;
            // Scale: small finite positive FP32. Clamp exponent to [-20, -4]
            // so scale * 15.0 stays well below FP32 overflow and dequantized
            // weights land in a sane range. Random mantissa.
            let scale_exp: u32 = 0x43 + (next() & 0x7);  // exp 0x43..0x4A → 2^-60..2^-53 hmm too small
            let scale_bits = (scale_exp << 23) | (next() & 0x007F_FFFF);
            // zp: random small magnitude, either sign
            let zp_bits = ((next() & 0xFF) << 23) | (next() & 0x007F_FFFF);
            // Guard both against non-finite
            let scale = f32::from_bits(scale_bits);
            let zp = f32::from_bits(zp_bits);
            let scale_ok = if scale.is_finite() && scale.abs() < 1e-2 && scale > 0.0 { scale } else { 1e-3 };
            let zp_ok    = if zp.is_finite() && zp.abs() < 1.0 { zp } else { -0.5 };
            out[gp..gp + 4].copy_from_slice(&scale_ok.to_le_bytes());
            out[gp + 4..gp + 8].copy_from_slice(&zp_ok.to_le_bytes());
            for i in 0..128 {
                out[gp + 8 + i] = (next() & 0xFF) as u8;
            }
        }
    }
    out
}

fn bytes_of(v: &[f32]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 4) }
}
