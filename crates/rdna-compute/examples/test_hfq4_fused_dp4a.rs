// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kevin Read
// hipfire — see LICENSE and NOTICE in the project root.

//! Correctness test for the 3 new HFQ4 fused dp4a kernels (gfx906).
//!
//! Issue #276 Gap 2:
//!   - `gemm_qkvza_hfq4g256_wave64_dp4a` (4-way)
//!   - `gemm_qkv_hfq4g256_wave64_dp4a`    (3-way)
//!   - `gemm_gate_up_hfq4g256_wave64_dp4a` (2-way)
//!
//! Each compared against its `gemm_*_hfq4g256_fp16_wave64` reference.
//! NRMSE should land at the Q8_1×HFQ4 quantization-noise floor
//! (~0.2-0.4%), matching the residual sibling's observed band.
//!
//! Usage: cargo run --release -p rdna-compute --example test_hfq4_fused_dp4a \
//!        -- [K] [N]
//!
//! Defaults: K=512 N=8 (exercises BT=16 partial-tile boundary).
//! Tests row-routing for 4-way/3-way/2-way fan-out.

use rdna_compute::{DType, Gpu, GpuTensor};

const QKV_M: usize = 64;
const Z_M: usize = 32;
const BETA_M: usize = 32;
const ALPHA_M: usize = 32;
// QKV variant uses Q/K/V sizes
const Q_M: usize = 128;
const K_M: usize = 64;
const V_M: usize = 64;
// gate_up uses gate/up sizes
const GATE_M: usize = 128;
const UP_M: usize = 128;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let k: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(512);
    let n: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(8);

    assert!(k % 256 == 0, "K must be a multiple of 256");
    let groups_per_row = k / 256;

    let mut gpu = Gpu::init().expect("gpu init");
    eprintln!("arch: {} K={k} N={n}", gpu.arch);
    if gpu.arch != "gfx906" {
        eprintln!("WARNING: this test is only meaningful on gfx906; skipping");
        std::process::exit(0);
    }

    let mut all_pass = true;

    // ── qkvza (4-way) ─────────────────────────────────────────────────
    {
        eprintln!("\n=== gemm_qkvza_hfq4g256_wave64_dp4a (4-way) ===");
        let a_qkv = upload_weights(&mut gpu, QKV_M, groups_per_row, 0xAA01);
        let a_z = upload_weights(&mut gpu, Z_M, groups_per_row, 0xAA02);
        let a_beta = upload_weights(&mut gpu, BETA_M, groups_per_row, 0xAA03);
        let a_alpha = upload_weights(&mut gpu, ALPHA_M, groups_per_row, 0xAA04);
        let x = upload_x(&mut gpu, n, k);

        let y_q_dp = gpu.zeros(&[n * QKV_M], DType::F32).unwrap();
        let y_z_dp = gpu.zeros(&[n * Z_M], DType::F32).unwrap();
        let y_b_dp = gpu.zeros(&[n * BETA_M], DType::F32).unwrap();
        let y_a_dp = gpu.zeros(&[n * ALPHA_M], DType::F32).unwrap();
        let y_q_ref = gpu.zeros(&[n * QKV_M], DType::F32).unwrap();
        let y_z_ref = gpu.zeros(&[n * Z_M], DType::F32).unwrap();
        let y_b_ref = gpu.zeros(&[n * BETA_M], DType::F32).unwrap();
        let y_a_ref = gpu.zeros(&[n * ALPHA_M], DType::F32).unwrap();

        gpu.gemm_qkvza_hfq4g256_wave64_dp4a(
            &a_qkv, &a_z, &a_beta, &a_alpha, &x,
            &y_q_dp, &y_z_dp, &y_b_dp, &y_a_dp,
            QKV_M, Z_M, BETA_M, ALPHA_M, k, n,
        ).expect("qkvza dp4a");
        gpu.gemm_qkvza_hfq4g256_fp16_wave64(
            &a_qkv, &a_z, &a_beta, &a_alpha, &x,
            &y_q_ref, &y_z_ref, &y_b_ref, &y_a_ref,
            QKV_M, Z_M, BETA_M, ALPHA_M, k, n,
        ).expect("qkvza fp16_wave64");
        gpu.hip.device_synchronize().expect("sync");

        let pass = compare(&mut gpu, &y_q_dp, &y_q_ref, "qkv")
            && compare(&mut gpu, &y_z_dp, &y_z_ref, "z")
            && compare(&mut gpu, &y_b_dp, &y_b_ref, "beta")
            && compare(&mut gpu, &y_a_dp, &y_a_ref, "alpha");
        all_pass &= pass;
    }

    // ── qkvza tail case: qkv_m=0, z_m=0 (exercised by the MMQ-split path
    //    where MMQ handles qkv+z and the tail runs beta+alpha only). This
    //    also exercises the `_prequant` entry point used by the dispatcher
    //    to skip re-quantization. ─────────────────────────────────────────
    {
        eprintln!("\n=== qkvza tail (qkv_m=0, z_m=0) — exercises row-routing prologue + _prequant ===");
        let a_qkv = upload_weights(&mut gpu, QKV_M, groups_per_row, 0xAA01);
        let a_z = upload_weights(&mut gpu, Z_M, groups_per_row, 0xAA02);
        let a_beta = upload_weights(&mut gpu, BETA_M, groups_per_row, 0xAA03);
        let a_alpha = upload_weights(&mut gpu, ALPHA_M, groups_per_row, 0xAA04);
        let x = upload_x(&mut gpu, n, k);

        // For the tail path, only beta/alpha get written. qkv/z buffers are
        // passed (since the kernel takes them as kernargs) but the kernel's
        // row-routing prologue should not touch their outputs when M=0.
        let y_q_dp = gpu.zeros(&[n * QKV_M], DType::F32).unwrap();
        let y_z_dp = gpu.zeros(&[n * Z_M], DType::F32).unwrap();
        let y_b_dp = gpu.zeros(&[n * BETA_M], DType::F32).unwrap();
        let y_a_dp = gpu.zeros(&[n * ALPHA_M], DType::F32).unwrap();
        let y_q_ref = gpu.zeros(&[n * QKV_M], DType::F32).unwrap();
        let y_z_ref = gpu.zeros(&[n * Z_M], DType::F32).unwrap();
        let y_b_ref = gpu.zeros(&[n * BETA_M], DType::F32).unwrap();
        let y_a_ref = gpu.zeros(&[n * ALPHA_M], DType::F32).unwrap();

        // qkv/z outputs are zero-initialised via `gpu.zeros` above; if the
        // kernel respects the row-routing prologue (gid >= total_m skip),
        // they stay zero after the call.

        // Call _prequant directly — the path used by the dispatcher's MMQ-tail.
        let xq_ptr = gpu.ensure_q8_1_mmq_x(&x, n, k).expect("quantize x");
        gpu.gemm_qkvza_hfq4g256_wave64_dp4a_prequant(
            &a_qkv, &a_z, &a_beta, &a_alpha,
            xq_ptr,
            &y_q_dp, &y_z_dp, &y_b_dp, &y_a_dp,
            0, 0, BETA_M, ALPHA_M, k, n,
        ).expect("qkvza dp4a tail prequant");
        gpu.gemm_qkvza_hfq4g256_fp16_wave64(
            &a_qkv, &a_z, &a_beta, &a_alpha, &x,
            &y_q_ref, &y_z_ref, &y_b_ref, &y_a_ref,
            0, 0, BETA_M, ALPHA_M, k, n,
        ).expect("qkvza fp16_wave64 tail");
        gpu.hip.device_synchronize().expect("sync");

        // qkv + z outputs should remain zero (kernel skipped them via gid >= total_m).
        let yq_dp_host = gpu.download_f32(&y_q_dp).expect("download yq_dp");
        let yz_dp_host = gpu.download_f32(&y_z_dp).expect("download yz_dp");
        let qkv_untouched = yq_dp_host.iter().all(|&v| v == 0.0);
        let z_untouched = yz_dp_host.iter().all(|&v| v == 0.0);
        if !qkv_untouched {
            eprintln!("  [qkv  ] FAIL — kernel wrote into qkv output despite qkv_m=0");
        } else {
            eprintln!("  [qkv  ] PASS — untouched (qkv_m=0 skip works)");
        }
        if !z_untouched {
            eprintln!("  [z    ] FAIL — kernel wrote into z output despite z_m=0");
        } else {
            eprintln!("  [z    ] PASS — untouched (z_m=0 skip works)");
        }
        let pass_routing = qkv_untouched && z_untouched;

        // beta + alpha should match the fp16_wave64 reference within
        // Q8_1×HFQ4 quantization-noise tolerance.
        let pass_beta = compare(&mut gpu, &y_b_dp, &y_b_ref, "beta");
        let pass_alpha = compare(&mut gpu, &y_a_dp, &y_a_ref, "alpha");

        all_pass &= pass_routing && pass_beta && pass_alpha;
    }

    // ── qkv (3-way) ───────────────────────────────────────────────────
    {
        eprintln!("\n=== gemm_qkv_hfq4g256_wave64_dp4a (3-way) ===");
        let a_q = upload_weights(&mut gpu, Q_M, groups_per_row, 0xBB01);
        let a_k = upload_weights(&mut gpu, K_M, groups_per_row, 0xBB02);
        let a_v = upload_weights(&mut gpu, V_M, groups_per_row, 0xBB03);
        let x = upload_x(&mut gpu, n, k);

        let y_q_dp = gpu.zeros(&[n * Q_M], DType::F32).unwrap();
        let y_k_dp = gpu.zeros(&[n * K_M], DType::F32).unwrap();
        let y_v_dp = gpu.zeros(&[n * V_M], DType::F32).unwrap();
        let y_q_ref = gpu.zeros(&[n * Q_M], DType::F32).unwrap();
        let y_k_ref = gpu.zeros(&[n * K_M], DType::F32).unwrap();
        let y_v_ref = gpu.zeros(&[n * V_M], DType::F32).unwrap();

        gpu.gemm_qkv_hfq4g256_wave64_dp4a(
            &a_q, &a_k, &a_v, &x, &y_q_dp, &y_k_dp, &y_v_dp,
            Q_M, K_M, V_M, k, n,
        ).expect("qkv dp4a");
        gpu.gemm_qkv_hfq4g256_fp16_wave64(
            &a_q, &a_k, &a_v, &x, &y_q_ref, &y_k_ref, &y_v_ref,
            Q_M, K_M, V_M, k, n,
        ).expect("qkv fp16_wave64");
        gpu.hip.device_synchronize().expect("sync");

        let pass = compare(&mut gpu, &y_q_dp, &y_q_ref, "q")
            && compare(&mut gpu, &y_k_dp, &y_k_ref, "k")
            && compare(&mut gpu, &y_v_dp, &y_v_ref, "v");
        all_pass &= pass;
    }

    // ── gate_up (2-way) ───────────────────────────────────────────────
    {
        eprintln!("\n=== gemm_gate_up_hfq4g256_wave64_dp4a (2-way) ===");
        let a_g = upload_weights(&mut gpu, GATE_M, groups_per_row, 0xCC01);
        let a_u = upload_weights(&mut gpu, UP_M, groups_per_row, 0xCC02);
        let x = upload_x(&mut gpu, n, k);

        let y_g_dp = gpu.zeros(&[n * GATE_M], DType::F32).unwrap();
        let y_u_dp = gpu.zeros(&[n * UP_M], DType::F32).unwrap();
        let y_g_ref = gpu.zeros(&[n * GATE_M], DType::F32).unwrap();
        let y_u_ref = gpu.zeros(&[n * UP_M], DType::F32).unwrap();

        gpu.gemm_gate_up_hfq4g256_wave64_dp4a(
            &a_g, &a_u, &x, &y_g_dp, &y_u_dp, GATE_M, UP_M, k, n,
        ).expect("gate_up dp4a");
        gpu.gemm_gate_up_hfq4g256_fp16_wave64(
            &a_g, &a_u, &x, &y_g_ref, &y_u_ref, GATE_M, UP_M, k, n,
        ).expect("gate_up fp16_wave64");
        gpu.hip.device_synchronize().expect("sync");

        let pass = compare(&mut gpu, &y_g_dp, &y_g_ref, "gate")
            && compare(&mut gpu, &y_u_dp, &y_u_ref, "up");
        all_pass &= pass;
    }

    if all_pass {
        eprintln!("\nALL PASS");
        std::process::exit(0);
    } else {
        eprintln!("\nFAIL");
        std::process::exit(1);
    }
}

fn upload_weights(gpu: &mut Gpu, m: usize, groups_per_row: usize, seed: u64) -> GpuTensor {
    let bytes = synth_hfq4g256_weights(m, groups_per_row, seed);
    gpu.upload_raw(&bytes, &[m * groups_per_row * 136]).expect("upload weights")
}

fn upload_x(gpu: &mut Gpu, n: usize, k: usize) -> GpuTensor {
    let x_host: Vec<f32> = (0..n * k)
        .map(|i| {
            let v = ((i as i64).wrapping_mul(1103515245).wrapping_add(12345)) as f32;
            (v * 1e-9) % 2.0 - 1.0
        })
        .collect();
    gpu.upload_f32(&x_host, &[n * k]).expect("upload x")
}

fn compare(gpu: &mut Gpu, dp4a: &GpuTensor, ref_: &GpuTensor, label: &str) -> bool {
    let dp = gpu.download_f32(dp4a).expect("download dp4a");
    let rf = gpu.download_f32(ref_).expect("download ref");
    assert_eq!(dp.len(), rf.len());
    let n = dp.len();

    let mut sum_sq_err = 0.0f64;
    let mut sum_sq_ref = 0.0f64;
    let mut max_abs_err = 0.0f32;
    for i in 0..n {
        let err = (dp[i] - rf[i]).abs();
        if err > max_abs_err { max_abs_err = err; }
        sum_sq_err += (err as f64).powi(2);
        sum_sq_ref += (rf[i] as f64).powi(2);
    }
    let rms_err = (sum_sq_err / n as f64).sqrt() as f32;
    let rms_ref = (sum_sq_ref / n as f64).sqrt() as f32;
    let nrmse = rms_err / rms_ref.max(1e-12);

    let dp_nonzero = dp.iter().any(|&v| v.abs() > 1e-12);
    let pass = nrmse < 1e-2 && dp_nonzero;
    let verdict = if pass { "PASS" } else { "FAIL" };
    eprintln!(
        "  [{label:5}] NRMSE={:.4}%  max_abs_err={:.4e}  rms_ref={:.4e}  {verdict}",
        nrmse * 100.0, max_abs_err, rms_ref
    );
    if !dp_nonzero {
        eprintln!("    dp4a output is all-zero — kernel may not have run");
    }
    pass
}

fn synth_hfq4g256_weights(m: usize, groups_per_row: usize, seed: u64) -> Vec<u8> {
    let total = m * groups_per_row * 136;
    let mut out = vec![0u8; total];
    let mut state = seed;
    let mut next = || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (state >> 33) as u32
    };
    let scale_target = 1e-3f32;
    let zp_max = 1.0f32;
    for row in 0..m {
        for g in 0..groups_per_row {
            let gp = (row * groups_per_row + g) * 136;
            let scale = scale_target * (0.5 + (next() & 0xFFFF) as f32 / 65535.0 * 1.5);
            let zp = ((next() & 0xFFFF) as f32 / 65535.0) * 2.0 * zp_max - zp_max;
            out[gp..gp + 4].copy_from_slice(&scale.to_le_bytes());
            out[gp + 4..gp + 8].copy_from_slice(&zp.to_le_bytes());
            for i in 0..128 {
                out[gp + 8 + i] = (next() & 0xFF) as u8;
            }
        }
    }
    out
}
