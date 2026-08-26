// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Correctness + perf oracle for `gemm_qkvza_hfq4g256` (batched QKVZA), with
//! Muse Glimmer path-specific extension.
//!
//! Original: compares batched GEMM × 1 against fused GEMV × N on Qwen3.5 LA
//! shapes qkv_m=6144, z=2048, beta=16, alpha=16, K=1024.
//!
//! Muse extension: `muse` / `--muse` mode exercises the Muse-exact
//! prefill QKVG shapes q=4096, k=256, v=256, gate=4096, K=6656 at
//! B=128,192,256 (configurable) and compares the batched fused QKVG path
//! (overwrite semantics, gfx1100-gated in production) against four
//! independent established `gemm_hfq4g256_batched_lmhead` calls.
//! Reports bit-difference count + max abs/rel error and timing, exits
//! nonzero on violated declared tolerance, prints reproducible
//! shape/batch details, and reuses allocations across B.

use rdna_compute::{DType, Gpu};
use std::time::Instant;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let is_muse = args.iter().any(|a| a == "muse" || a == "--muse");
    if is_muse {
        run_muse_qkvg_oracle(args);
        return;
    }

    // ── original Qwen LA path (preserved byte-for-byte behavior) ──
    let qkv_m: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(6144);
    let z_m:   usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(2048);
    let beta_m: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(16);
    let alpha_m: usize = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(16);
    let k:     usize = args.get(5).and_then(|s| s.parse().ok()).unwrap_or(1024);
    let n_list: Vec<usize> = if args.len() > 6 {
        args[6..].iter().filter_map(|s| s.parse().ok()).collect()
    } else {
        vec![1, 4, 8, 16, 32, 64]
    };

    assert!(k % 256 == 0, "K must be a multiple of 256 for HFQ4-G256");
    let groups_per_row = k / 256;
    let row_bytes = groups_per_row * 136;

    eprintln!("=== gemm_qkvza_hfq4g256 test ===");
    eprintln!("qkv_m={qkv_m}  z_m={z_m}  beta_m={beta_m}  alpha_m={alpha_m}  K={k}");
    eprintln!("groups_per_row={groups_per_row}, row_bytes={row_bytes}");

    let mut gpu = Gpu::init().expect("gpu init");

    let w_qkv   = gpu.upload_raw(&synth(qkv_m,   groups_per_row, 0xA1),   &[qkv_m   * row_bytes]).unwrap();
    let w_z     = gpu.upload_raw(&synth(z_m,     groups_per_row, 0xB2),   &[z_m     * row_bytes]).unwrap();
    let w_beta  = gpu.upload_raw(&synth(beta_m,  groups_per_row, 0xC3),   &[beta_m  * row_bytes]).unwrap();
    let w_alpha = gpu.upload_raw(&synth(alpha_m, groups_per_row, 0xD4),   &[alpha_m * row_bytes]).unwrap();

    let max_n = *n_list.iter().max().unwrap();
    let x_host: Vec<f32> = (0..max_n * k)
        .map(|i| {
            let v = ((i as i64).wrapping_mul(1103515245).wrapping_add(12345)) as f32;
            (v * 1e-9) % 2.0 - 1.0
        })
        .collect();

    // GEMV path scratch buffers (single-token inputs/outputs).
    let x_gemv = gpu.alloc_tensor(&[k], DType::F32).unwrap();
    let y_qkv_1   = gpu.alloc_tensor(&[qkv_m],   DType::F32).unwrap();
    let y_z_1     = gpu.alloc_tensor(&[z_m],     DType::F32).unwrap();
    let y_beta_1  = gpu.alloc_tensor(&[beta_m],  DType::F32).unwrap();
    let y_alpha_1 = gpu.alloc_tensor(&[alpha_m], DType::F32).unwrap();

    // Collected GEMV outputs across all N batch elements.
    let y_qkv_gemv_col   = gpu.alloc_tensor(&[max_n * qkv_m],   DType::F32).unwrap();
    let y_z_gemv_col     = gpu.alloc_tensor(&[max_n * z_m],     DType::F32).unwrap();
    let y_beta_gemv_col  = gpu.alloc_tensor(&[max_n * beta_m],  DType::F32).unwrap();
    let y_alpha_gemv_col = gpu.alloc_tensor(&[max_n * alpha_m], DType::F32).unwrap();

    // Batched GEMM path.
    let x_gemm       = gpu.alloc_tensor(&[max_n * k],     DType::F32).unwrap();
    let y_qkv_gemm   = gpu.alloc_tensor(&[max_n * qkv_m], DType::F32).unwrap();
    let y_z_gemm     = gpu.alloc_tensor(&[max_n * z_m],   DType::F32).unwrap();
    let y_beta_gemm  = gpu.alloc_tensor(&[max_n * beta_m], DType::F32).unwrap();
    let y_alpha_gemm = gpu.alloc_tensor(&[max_n * alpha_m], DType::F32).unwrap();

    gpu.hip.memcpy_htod(&x_gemm.buf, bytes_of(&x_host)).unwrap();

    for &n in &n_list {
        eprintln!("\n--- N = {n} ---");

        // GEMV × N
        let mut gemv_us: f64 = 0.0;
        for i in 0..n {
            gpu.hip.memcpy_htod(&x_gemv.buf, bytes_of(&x_host[i * k..(i + 1) * k])).unwrap();
            gpu.hip.device_synchronize().unwrap();
            let t = Instant::now();
            gpu.fused_qkvza_hfq4g256(
                &w_qkv, &w_z, &w_beta, &w_alpha,
                &x_gemv,
                &y_qkv_1, &y_z_1, &y_beta_1, &y_alpha_1,
                qkv_m, z_m, beta_m, alpha_m,
                k,
            ).unwrap();
            gpu.hip.device_synchronize().unwrap();
            gemv_us += t.elapsed().as_secs_f64() * 1e6;

            gpu.hip.memcpy_dtod_at(&y_qkv_gemv_col.buf,   i * qkv_m   * 4, &y_qkv_1.buf,   0, qkv_m   * 4).unwrap();
            gpu.hip.memcpy_dtod_at(&y_z_gemv_col.buf,     i * z_m     * 4, &y_z_1.buf,     0, z_m     * 4).unwrap();
            gpu.hip.memcpy_dtod_at(&y_beta_gemv_col.buf,  i * beta_m  * 4, &y_beta_1.buf,  0, beta_m  * 4).unwrap();
            gpu.hip.memcpy_dtod_at(&y_alpha_gemv_col.buf, i * alpha_m * 4, &y_alpha_1.buf, 0, alpha_m * 4).unwrap();
        }

        // GEMM × 1
        gpu.hip.device_synchronize().unwrap();
        let t = Instant::now();
        gpu.gemm_qkvza_hfq4g256(
            &w_qkv, &w_z, &w_beta, &w_alpha,
            &x_gemm,
            &y_qkv_gemm, &y_z_gemm, &y_beta_gemm, &y_alpha_gemm,
            qkv_m, z_m, beta_m, alpha_m,
            k, n,
        ).unwrap();
        gpu.hip.device_synchronize().unwrap();
        let gemm_us = t.elapsed().as_secs_f64() * 1e6;

        // Compare each of the 4 outputs byte-exact.
        let compare = |label: &str, col: &rdna_compute::GpuTensor, gemm: &rdna_compute::GpuTensor, m: usize| -> bool {
            let a = gpu.download_f32(col).unwrap()[..n * m].to_vec();
            let b = gpu.download_f32(gemm).unwrap()[..n * m].to_vec();
            for i in 0..n * m {
                if a[i].to_bits() != b[i].to_bits() {
                    let batch = i / m;
                    let row = i % m;
                    eprintln!(
                        "  {label}: DIVERGENT at batch={batch} row={row}  gemv={:.6e} ({:#010x})  gemm={:.6e} ({:#010x})",
                        a[i], a[i].to_bits(), b[i], b[i].to_bits()
                    );
                    let count: usize = a.iter().zip(b.iter()).filter(|(a, b)| a.to_bits() != b.to_bits()).count();
                    eprintln!("  {label}: {count}/{} elements diverged", n * m);
                    return false;
                }
            }
            true
        };

        let ok_qkv   = compare("qkv",   &y_qkv_gemv_col,   &y_qkv_gemm,   qkv_m);
        let ok_z     = compare("z",     &y_z_gemv_col,     &y_z_gemm,     z_m);
        let ok_beta  = compare("beta",  &y_beta_gemv_col,  &y_beta_gemm,  beta_m);
        let ok_alpha = compare("alpha", &y_alpha_gemv_col, &y_alpha_gemm, alpha_m);

        let all_ok = ok_qkv && ok_z && ok_beta && ok_alpha;
        let status = if all_ok { "byte-exact OK" } else { "DIVERGENT" };
        let speedup = gemv_us / gemm_us;
        eprintln!(
            "  gemv × {n}: {:8.1} µs   gemm × 1: {:8.1} µs   speedup: {:5.2}x   [{status}]",
            gemv_us, gemm_us, speedup
        );
        if !all_ok {
            std::process::exit(1);
        }
    }

    eprintln!("\n=== All N passed byte-exact ===");
}

// ── Muse QKVG oracle ────────────────────────────────────────────────
//
// Muse exact prefill QKVG projection family (MQ4G256/HFQ4G256, FWHT-rotated
// at quantize time):
//
//   q_proj      [4096, 6656]
//   k_proj      [ 256, 6656]
//   v_proj      [ 256, 6656]
//   attn_gate   [4096, 6656]   K=6656 for all four.
//
// Batches: 128, 192, 256 (prefill chunk sizes used in
// crates/hipfire-arch-muse-glimmer/src/forward.rs::glimmer_prefill_chunk_size).
// Command-line shape selection is preferred; the oracle reuses allocations
// across B (max_n sizing) rather than reallocating per batch.
//
// Fused path under test: the established direct
// `Gpu::gemm_qkvza_hfq4g256_wmma` batched overwrite kernel, called with
// Muse's exact shape after the production callsite's gfx1100 gate.
// Numerical ordering must match the established gfx11 WMMA qkvza kernel
// (kernels/src/gemm_qkvza_hfq4g256_wmma.hip); any deviation needs a
// bounded tolerance justified in the PR. This oracle is intentionally
// path-specific: it compares the fused QKVG against the four independent
// `gemm_hfq4g256_batched_lmhead` calls that constitute the pre-fusion
// baseline (separate rotates are shared, same as forward.rs shared Rot).
//
// Declared tolerance: byte-exact (bitdiff==0, max_abs==0, max_rel==0).
// Justification: same dequant (scale/zp + 4-bit unpack), same FWHT
// prerotated activation, same WMMA accumulation order per batch element
// (16×16×16 f16 WMMA, pairwise combine). The fused kernel only fuses the
// four GEMMs' grid.y batch dim; per-element arithmetic is unchanged.
// Override via CLI: --tol-abs=N --tol-rel=N --allow-bitdiff=N
// Env:  HIPFIRE_MUSE_QKVG_TOL_ABS / HIPFIRE_MUSE_QKVG_TOL_REL / HIPFIRE_MUSE_QKVG_ALLOW_BITDIFF
//
// The oracle exits nonzero on violated tolerance and prints reproducible
// shape/batch/arch details for CI.

fn run_muse_qkvg_oracle(args: Vec<String>) {
    // Parse CLI: `muse` token may be at any position; remaining numeric tokens are B list.
    let mut batches: Vec<usize> = Vec::new();
    let mut tol_abs: f32 = std::env::var("HIPFIRE_MUSE_QKVG_TOL_ABS").ok().and_then(|s| s.parse().ok()).unwrap_or(0.0);
    let mut tol_rel: f32 = std::env::var("HIPFIRE_MUSE_QKVG_TOL_REL").ok().and_then(|s| s.parse().ok()).unwrap_or(0.0);
    let mut allow_bitdiff: usize = std::env::var("HIPFIRE_MUSE_QKVG_ALLOW_BITDIFF").ok().and_then(|s| s.parse().ok()).unwrap_or(0);
    let mut help = false;
    // repeats for timing stability (1 JIT warmup discarded)
    let mut repeats: usize = 5;

    for a in args.iter().skip(1) {
        if a == "muse" || a == "--muse" {
            continue;
        } else if a == "--help" || a == "-h" {
            help = true;
        } else if let Some(v) = a.strip_prefix("--tol-abs=") {
            tol_abs = v.parse().unwrap_or(tol_abs);
        } else if let Some(v) = a.strip_prefix("--tol-rel=") {
            tol_rel = v.parse().unwrap_or(tol_rel);
        } else if let Some(v) = a.strip_prefix("--allow-bitdiff=") {
            allow_bitdiff = v.parse().unwrap_or(allow_bitdiff);
        } else if let Some(v) = a.strip_prefix("--repeats=") {
            repeats = v.parse().unwrap_or(repeats).clamp(1, 100);
        } else if let Ok(b) = a.parse::<usize>() {
            // Numeric batch size. Filter implausible values (keep 1..8192).
            if b >= 1 && b <= 8192 {
                batches.push(b);
            }
        } else if a.starts_with('-') {
            eprintln!("unknown flag {a}");
            help = true;
        } else {
            eprintln!("unknown arg {a}");
            help = true;
        }
    }
    if help {
        eprintln!("Usage: test_gemm_qkvza_hfq4g256 muse [B...] [--tol-abs=N --tol-rel=N --allow-bitdiff=N --repeats=N]");
        eprintln!("  Muse exact shapes: q=4096 k=256 v=256 gate=4096 K=6656 (HFQ4G256/MQ4G256, FWHT-rotated)");
        eprintln!("  Default B: 128 192 256   (use e.g. `muse 128` for single batch)");
        eprintln!("  Compares fused batched QKVG (overwrite) vs 4× gemm_hfq4g256_batched_lmhead");
        eprintln!("  Declared tolerance: bitdiff==0, max_abs==0, max_rel==0 (byte-exact WMMA ordering)");
        std::process::exit(0);
    }
    if batches.is_empty() {
        batches = vec![128, 192, 256];
    }
    batches.sort_unstable();
    batches.dedup();

    const Q_M: usize = 4096;
    const K_M: usize = 256;
    const V_M: usize = 256;
    const GATE_M: usize = 4096;
    const K: usize = 6656;
    assert!(K % 256 == 0);
    let groups_per_row = K / 256;
    let row_bytes = groups_per_row * 136;

    eprintln!("=== Muse QKVG batched oracle (gfx1100 path-specific) ===");
    eprintln!("shapes: q={Q_M} k={K_M} v={V_M} gate={GATE_M} K={K}  (HFQ4G256/MQ4G256, FWHT-rotated)");
    eprintln!("batches: {:?}  groups_per_row={} row_bytes={}", batches, groups_per_row, row_bytes);
    eprintln!("fused: single batched QKVG (overwrite) vs 4× gemm_hfq4g256_batched_lmhead");
    eprintln!("tolerance: bitdiff<={}  max_abs<={:.3e}  max_rel<={:.3e}  repeats={}", allow_bitdiff, tol_abs, tol_rel, repeats);
    eprintln!("note: numerical ordering must match gfx11 WMMA qkvza kernel (kernels/src/gemm_qkvza_hfq4g256_wmma.hip); bounded tolerance only if justified");

    let mut gpu = Gpu::init().expect("gpu init");
    let arch = gpu.arch.clone();
    let is_gfx1100 = gpu.arch_caps.is_gfx1100();
    let has_wmma = gpu.arch_caps.has_wmma_w32();
    let has_wmma_gfx12 = gpu.arch_caps.has_wmma_w32_gfx12();
    eprintln!("arch={arch} is_gfx1100={} has_wmma_w32={} has_wmma_w32_gfx12={} device_id={}", is_gfx1100, has_wmma, has_wmma_gfx12, gpu.device_id);
    // Muse production gate is `is_gfx1100 && exact Muse dims && MQ4/HFQ4`. This oracle runs on any
    // arch for CI but flags when the production Muse method would be ineligible.
    if !is_gfx1100 {
        eprintln!("info: production Muse batched QKVG is gfx1100-gated (gpu.arch_caps.is_gfx1100() && exact dims && MQ4/HFQ4, returns false when ineligible); oracle still validates numerics on this arch");
    }

    // Synthetic HFQ4G256 weights (136 B/group: f32 scale, f32 zp, 128 packed nibbles).
    // Same generator as bench_glimmer_wmma_ceiling.rs; deterministic per shape.
    let w_q    = gpu.upload_raw(&synth(Q_M,    groups_per_row, 0x51), &[Q_M    * row_bytes]).unwrap();
    let w_k    = gpu.upload_raw(&synth(K_M,    groups_per_row, 0x52), &[K_M    * row_bytes]).unwrap();
    let w_v    = gpu.upload_raw(&synth(V_M,    groups_per_row, 0x53), &[V_M    * row_bytes]).unwrap();
    let w_gate = gpu.upload_raw(&synth(GATE_M, groups_per_row, 0x54), &[GATE_M * row_bytes]).unwrap();

    let max_n = *batches.iter().max().unwrap();
    // Host activation: deterministic f32 in [-0.5, 0.5), same across B for reproducibility.
    let x_host: Vec<f32> = (0..max_n * K).map(|i| {
        let mut s = (i as u64).wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((s >> 40) as f32 / (1u64 << 24) as f32) - 0.5
    }).collect();

    // Reused allocation: one set sized to max_n, sub-ranges used per B.
    // Avoids enormous redundant allocations across the B sweep.
    let x_raw = gpu.alloc_tensor(&[max_n * K], DType::F32).unwrap();
    let x_rot = gpu.alloc_tensor(&[max_n * K], DType::F32).unwrap();
    gpu.hip.memcpy_htod(&x_raw.buf, bytes_of(&x_host)).unwrap();

    // Fused outputs (overwrite semantics).
    let y_q_fused    = gpu.alloc_tensor(&[max_n * Q_M], DType::F32).unwrap();
    let y_k_fused    = gpu.alloc_tensor(&[max_n * K_M], DType::F32).unwrap();
    let y_v_fused    = gpu.alloc_tensor(&[max_n * V_M], DType::F32).unwrap();
    let y_gate_fused = gpu.alloc_tensor(&[max_n * GATE_M], DType::F32).unwrap();

    // Separate baseline outputs.
    let y_q_sep    = gpu.alloc_tensor(&[max_n * Q_M], DType::F32).unwrap();
    let y_k_sep    = gpu.alloc_tensor(&[max_n * K_M], DType::F32).unwrap();
    let y_v_sep    = gpu.alloc_tensor(&[max_n * V_M], DType::F32).unwrap();
    let y_gate_sep = gpu.alloc_tensor(&[max_n * GATE_M], DType::F32).unwrap();

    // Helper to compute per-projection error metrics.
    let compute_metrics = |a: &[f32], b: &[f32]| -> (usize, f32, f32) {
        let mut bitdiff = 0usize;
        let mut max_abs = 0.0f32;
        let mut max_rel = 0.0f32;
        for (x, y) in a.iter().zip(b.iter()) {
            if x.to_bits() != y.to_bits() {
                bitdiff += 1;
            }
            let abs = (x - y).abs();
            if abs > max_abs { max_abs = abs; }
            // relative: |x-y| / max(|x|,|y|,1e-6) to avoid div-by-zero blowup
            let denom = x.abs().max(y.abs()).max(1e-6);
            let rel = abs / denom;
            if rel > max_rel { max_rel = rel; }
        }
        (bitdiff, max_abs, max_rel)
    };

    let mut any_failure = false;

    for &b in &batches {
        eprintln!("\n--- Muse QKVG B={b}  q={Q_M} k={K_M} v={V_M} gate={GATE_M} K={K} arch={arch} ---");

        // Rotate activation once per B (shared across all four projections, matching
        // forward.rs::fused_rmsnorm_rotate_mq_batched_for / rotate_x_mq_batched_for).
        // Only the first b*K elements are rotated; tail remains zeroed.
        gpu.hip.device_synchronize().unwrap();
        // Ensure x_raw contains fresh host data for this b (prefix already uploaded; full max_n uploaded once).
        // Rotate batched: x_raw[0..b*K] -> x_rot[0..b*K]
        gpu.rotate_x_mq_batched(&x_raw, &x_rot, K, b).expect("rotate_x_mq_batched failed — is MQ sign table available?");

        gpu.hip.device_synchronize().unwrap();

        // Warmup both paths once to JIT and populate caches.
        gpu.gemm_hfq4g256_batched_lmhead(&w_q, &x_rot, &y_q_sep, Q_M, K, b)
            .unwrap();
        gpu.gemm_hfq4g256_batched_lmhead(&w_k, &x_rot, &y_k_sep, K_M, K, b)
            .unwrap();
        gpu.gemm_hfq4g256_batched_lmhead(&w_v, &x_rot, &y_v_sep, V_M, K, b)
            .unwrap();
        gpu.gemm_hfq4g256_batched_lmhead(&w_gate, &x_rot, &y_gate_sep, GATE_M, K, b)
            .unwrap();
        gpu.hip.device_synchronize().unwrap();

        // The Muse production callsite exact-gates gfx1100 and then invokes this
        // direct WMMA method. Do not use generic dispatch on gfx1100 here: it may
        // select MMQ before WMMA and would not validate the shipped path.
        if gpu.arch_caps.is_gfx1100() {
            gpu.gemm_qkvza_hfq4g256_wmma(
                &w_q, &w_k, &w_v, &w_gate, &x_rot, &y_q_fused, &y_k_fused,
                &y_v_fused, &y_gate_fused, Q_M, K_M, V_M, GATE_M, K, b,
            )
            .unwrap();
        } else {
            gpu.gemm_qkvza_hfq4g256(
                &w_q, &w_k, &w_v, &w_gate, &x_rot, &y_q_fused, &y_k_fused,
                &y_v_fused, &y_gate_fused, Q_M, K_M, V_M, GATE_M, K, b,
            )
            .unwrap();
        }
        gpu.hip.device_synchronize().unwrap();

        // Timed separate: 4× batched lmhead
        let mut sep_us: f64 = 0.0;
        let mut fused_us: f64 = 0.0;
        // Do `repeats` timed iterations, discarding the warmup above.
        for _ in 0..repeats {
            gpu.hip.device_synchronize().unwrap();
            let t0 = Instant::now();
            gpu.gemm_hfq4g256_batched_lmhead(&w_q, &x_rot, &y_q_sep, Q_M, K, b).unwrap();
            gpu.gemm_hfq4g256_batched_lmhead(&w_k, &x_rot, &y_k_sep, K_M, K, b).unwrap();
            gpu.gemm_hfq4g256_batched_lmhead(&w_v, &x_rot, &y_v_sep, V_M, K, b).unwrap();
            gpu.gemm_hfq4g256_batched_lmhead(&w_gate, &x_rot, &y_gate_sep, GATE_M, K, b).unwrap();
            gpu.hip.device_synchronize().unwrap();
            sep_us += t0.elapsed().as_secs_f64() * 1e6;
        }
        sep_us /= repeats as f64;

        for _ in 0..repeats {
            gpu.hip.device_synchronize().unwrap();
            let t0 = Instant::now();
            if gpu.arch_caps.is_gfx1100() {
                gpu.gemm_qkvza_hfq4g256_wmma(&w_q, &w_k, &w_v, &w_gate, &x_rot, &y_q_fused, &y_k_fused, &y_v_fused, &y_gate_fused, Q_M, K_M, V_M, GATE_M, K, b).unwrap();
            } else {
                gpu.gemm_qkvza_hfq4g256(&w_q, &w_k, &w_v, &w_gate, &x_rot, &y_q_fused, &y_k_fused, &y_v_fused, &y_gate_fused, Q_M, K_M, V_M, GATE_M, K, b).unwrap();
            }
            gpu.hip.device_synchronize().unwrap();
            fused_us += t0.elapsed().as_secs_f64() * 1e6;
        }
        fused_us /= repeats as f64;

        // Download and compare per projection. Use only first b*m elements (prefix).
        let q_fused_v = gpu.download_f32(&y_q_fused).unwrap();
        let k_fused_v = gpu.download_f32(&y_k_fused).unwrap();
        let v_fused_v = gpu.download_f32(&y_v_fused).unwrap();
        let gate_fused_v = gpu.download_f32(&y_gate_fused).unwrap();
        let q_sep_v = gpu.download_f32(&y_q_sep).unwrap();
        let k_sep_v = gpu.download_f32(&y_k_sep).unwrap();
        let v_sep_v = gpu.download_f32(&y_v_sep).unwrap();
        let gate_sep_v = gpu.download_f32(&y_gate_sep).unwrap();

        let q_fused = &q_fused_v[..b * Q_M];
        let q_sep = &q_sep_v[..b * Q_M];
        let k_fused = &k_fused_v[..b * K_M];
        let k_sep = &k_sep_v[..b * K_M];
        let v_fused = &v_fused_v[..b * V_M];
        let v_sep = &v_sep_v[..b * V_M];
        let gate_fused = &gate_fused_v[..b * GATE_M];
        let gate_sep = &gate_sep_v[..b * GATE_M];

        let (q_bd, q_abs, q_rel) = compute_metrics(q_sep, q_fused);
        let (k_bd, k_abs, k_rel) = compute_metrics(k_sep, k_fused);
        let (v_bd, v_abs, v_rel) = compute_metrics(v_sep, v_fused);
        let (g_bd, g_abs, g_rel) = compute_metrics(gate_sep, gate_fused);

        let total_bd = q_bd + k_bd + v_bd + g_bd;
        let max_abs = q_abs.max(k_abs).max(v_abs).max(g_abs);
        let max_rel = q_rel.max(k_rel).max(v_rel).max(g_rel);

        let flops = 2.0 * ((Q_M + K_M + V_M + GATE_M) as f64) * (K as f64) * (b as f64);
        let tflops_fused = flops / (fused_us * 1e-6) / 1e12;
        let tflops_sep = flops / (sep_us * 1e-6) / 1e12;
        let speedup = sep_us / fused_us;

        eprintln!("  separate 4× : {:8.1} µs  ({:5.2} TFLOP/s)", sep_us, tflops_sep);
        eprintln!("  fused   1×  : {:8.1} µs  ({:5.2} TFLOP/s)  speedup {:5.2}x", fused_us, tflops_fused, speedup);
        eprintln!("  q_proj   bitdiff {}/{}  max_abs {:.3e}  max_rel {:.3e}", q_bd, b*Q_M, q_abs, q_rel);
        eprintln!("  k_proj   bitdiff {}/{}  max_abs {:.3e}  max_rel {:.3e}", k_bd, b*K_M, k_abs, k_rel);
        eprintln!("  v_proj   bitdiff {}/{}  max_abs {:.3e}  max_rel {:.3e}", v_bd, b*V_M, v_abs, v_rel);
        eprintln!("  gate     bitdiff {}/{}  max_abs {:.3e}  max_rel {:.3e}", g_bd, b*GATE_M, g_abs, g_rel);
        eprintln!("  total    bitdiff {}/{}  max_abs {:.3e}  max_rel {:.3e}", total_bd, b*(Q_M+K_M+V_M+GATE_M), max_abs, max_rel);

        let tol_ok = total_bd <= allow_bitdiff && max_abs <= tol_abs && max_rel <= tol_rel;
        // Byte-exact expectation: if any bitdiff, also report first divergent element per projection.
        if !tol_ok || total_bd != 0 {
            for (label, sep, fused, m) in [
                ("q", q_sep, q_fused, Q_M),
                ("k", k_sep, k_fused, K_M),
                ("v", v_sep, v_fused, V_M),
                ("gate", gate_sep, gate_fused, GATE_M),
            ] {
                let mut first = None;
                for i in 0..b*m {
                    if sep[i].to_bits() != fused[i].to_bits() {
                        first = Some((i, sep[i], fused[i]));
                        break;
                    }
                }
                if let Some((idx, s, f)) = first {
                    let batch = idx / m;
                    let row = idx % m;
                    eprintln!("  {label}: first divergent at batch={batch} row={row}  sep={:.6e} ({:#010x})  fused={:.6e} ({:#010x})  abs={:.3e} rel={:.3e}", s, s.to_bits(), f, f.to_bits(), (s-f).abs(), (s-f).abs()/s.abs().max(f.abs()).max(1e-6));
                }
            }
        }
        let status = if tol_ok { "PASS" } else { "FAIL (tolerance violated)" };
        eprintln!("  [{status}] B={b} arch={arch} shapes q={Q_M} k={K_M} v={V_M} gate={GATE_M} K={K}");
        if !tol_ok {
            eprintln!("  declared tolerance violated: bitdiff {total_bd} > {allow_bitdiff} or max_abs {max_abs:.3e} > {tol_abs:.3e} or max_rel {max_rel:.3e} > {tol_rel:.3e}");
            any_failure = true;
        }
    }

    if any_failure {
        eprintln!("\n=== Muse QKVG oracle FAILED (tolerance violated) ===");
        std::process::exit(1);
    } else {
        eprintln!("\n=== Muse QKVG oracle PASS (all B byte-exact within tolerance) ===");
    }
}

fn synth(m: usize, groups_per_row: usize, seed: u64) -> Vec<u8> {
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
            let scale_exp: u32 = 0x43 + (next() & 0x7);
            let scale_bits = (scale_exp << 23) | (next() & 0x007F_FFFF);
            let zp_bits = ((next() & 0xFF) << 23) | (next() & 0x007F_FFFF);
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
