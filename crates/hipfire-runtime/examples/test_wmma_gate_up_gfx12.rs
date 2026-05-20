// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Robin Van Cauter
// hipfire — see LICENSE and NOTICE in the project root.

//! Channel-test for the gfx12 (RDNA4) WMMA gate_up scaffold.
//!
//! Sister to test_wmma_qkv_gfx12 — same harness pattern, applied to the
//! 2-output `gemm_gate_up_hfq4g256_wmma_gfx12` kernel. Compares against
//! the validated `gemm_gate_up_hfq4g256_dot2` reference on synthetic
//! HFQ4G256 weights + f16 X.
//!
//! Bails on non-gfx12 (the `_w32_gfx12` builtin doesn't exist on gfx11).
//!
//! Run: cargo run --release --features deltanet -p hipfire-runtime \
//!         --example test_wmma_gate_up_gfx12

use rdna_compute::{DType, Gpu};

fn main() {
    let mut gpu = Gpu::init().expect("GPU init failed");
    let arch = gpu.arch.clone();
    eprintln!("GPU: {} ({:.1} GB VRAM)", arch, {
        let (_, total) = gpu.hip.get_vram_info().unwrap_or((0, 0));
        total as f64 / 1e9
    });

    if !arch.starts_with("gfx12") {
        eprintln!(
            "SKIP: this test requires gfx12 (RDNA4). Current arch: {arch}.\n\
             The `_w32_gfx12` WMMA builtin does not exist on gfx11."
        );
        std::process::exit(0);
    }

    let mut total_pass = 0;
    let mut total_fail = 0;

    // Same shape coverage rationale as the QKV test:
    //   minimal / multi-K-group / multi-batch-tile / multi-row-tile-per-proj
    //   / multi-tile-everywhere / projection-boundary-straddle.
    let shapes: &[(usize, usize, usize, usize)] = &[
        // (gate_m, up_m, K, N)
        (16, 16, 256, 16),
        (16, 16, 512, 16),
        (16, 16, 256, 32),
        (32, 16, 256, 16),
        (32, 32, 512, 32),
        (48, 16, 256, 16),
    ];

    for &(g_m, u_m, k, n) in shapes {
        let label = format!("gate={g_m} up={u_m} K={k} N={n}");
        eprintln!("\n--- {label} ---");
        match run_one(&mut gpu, g_m, u_m, k, n) {
            Ok(()) => {
                total_pass += 1;
                eprintln!("  {label:50} OK");
            }
            Err(e) => {
                total_fail += 1;
                eprintln!("  {label:50} FAIL");
                eprintln!("{e}");
            }
        }
    }

    eprintln!("\n--- Summary ---");
    eprintln!("  Passed: {total_pass}");
    eprintln!("  Failed: {total_fail}");
    if total_fail > 0 {
        std::process::exit(1);
    }
}

fn run_one(
    gpu: &mut Gpu,
    gate_m: usize,
    up_m: usize,
    k: usize,
    n: usize,
) -> Result<(), String> {
    assert_eq!(gate_m % 16, 0);
    assert_eq!(up_m % 16, 0);
    assert_eq!(k % 256, 0);
    assert_eq!(n % 16, 0);

    let a_gate_bytes = build_hfq4g256(gate_m, k, 0xD4);
    let a_up_bytes = build_hfq4g256(up_m, k, 0xE5);

    let a_gate = gpu
        .upload_raw(&a_gate_bytes, &[gate_m, k])
        .map_err(|e| format!("upload a_gate: {e}"))?;
    let a_up = gpu
        .upload_raw(&a_up_bytes, &[up_m, k])
        .map_err(|e| format!("upload a_up: {e}"))?;

    let x_f32: Vec<f32> = (0..(n * k))
        .map(|i| {
            let b = (i / k) as i32;
            let kk = (i % k) as i32;
            ((b * 7 + kk * 11) % 31 - 15) as f32 * 0.05
        })
        .collect();
    let x = gpu
        .upload_f32(&x_f32, &[n, k])
        .map_err(|e| format!("upload x: {e}"))?;

    let y_g_ref = gpu.alloc_tensor(&[n, gate_m], DType::F32).map_err(|e| format!("alloc yg_ref: {e}"))?;
    let y_u_ref = gpu.alloc_tensor(&[n, up_m], DType::F32).map_err(|e| format!("alloc yu_ref: {e}"))?;

    gpu.gemm_gate_up_hfq4g256_dot2(
        &a_gate, &a_up, &x, &y_g_ref, &y_u_ref, gate_m, up_m, k, n,
    )
    .map_err(|e| format!("dot2: {e}"))?;

    let ref_g = gpu.download_f32(&y_g_ref).map_err(|e| format!("download yg_ref: {e}"))?;
    let ref_u = gpu.download_f32(&y_u_ref).map_err(|e| format!("download yu_ref: {e}"))?;

    let y_g = gpu.alloc_tensor(&[n, gate_m], DType::F32).map_err(|e| format!("alloc yg: {e}"))?;
    let y_u = gpu.alloc_tensor(&[n, up_m], DType::F32).map_err(|e| format!("alloc yu: {e}"))?;

    gpu.gemm_gate_up_hfq4g256_wmma_gfx12(
        &a_gate, &a_up, &x, &y_g, &y_u, gate_m, up_m, k, n,
    )
    .map_err(|e| format!("wmma_gfx12: {e}"))?;

    let cand_g = gpu.download_f32(&y_g).map_err(|e| format!("download yg: {e}"))?;
    let cand_u = gpu.download_f32(&y_u).map_err(|e| format!("download yu: {e}"))?;

    gpu.free_tensor(a_gate).ok();
    gpu.free_tensor(a_up).ok();
    gpu.free_tensor(x).ok();
    gpu.free_tensor(y_g_ref).ok();
    gpu.free_tensor(y_u_ref).ok();
    gpu.free_tensor(y_g).ok();
    gpu.free_tensor(y_u).ok();

    let mut report = String::new();
    let ok_g = compare_proj("Y_gate", n, gate_m, &cand_g, &ref_g, &mut report);
    let ok_u = compare_proj("Y_up", n, up_m, &cand_u, &ref_u, &mut report);

    if ok_g && ok_u {
        Ok(())
    } else {
        Err(report)
    }
}

fn compare_proj(
    name: &str,
    n: usize,
    m: usize,
    cand: &[f32],
    refr: &[f32],
    report: &mut String,
) -> bool {
    assert_eq!(cand.len(), refr.len());
    assert_eq!(cand.len(), n * m);

    let mut max_abs = 0f32;
    let mut max_rel = 0f32;
    let mut n_bad = 0usize;
    let mut first_bad: Option<(usize, usize, f32, f32)> = None;
    let abs_tol = 5e-2_f32;
    let rel_tol = 1e-2_f32;
    let mut hist_row_mod16 = [0usize; 16];

    for batch in 0..n {
        for row in 0..m {
            let idx = batch * m + row;
            let a = cand[idx];
            let b = refr[idx];
            let abs = (a - b).abs();
            let rel = abs / b.abs().max(1e-3);
            if abs > max_abs { max_abs = abs; }
            if rel > max_rel { max_rel = rel; }
            if abs > abs_tol && rel > rel_tol {
                n_bad += 1;
                hist_row_mod16[row % 16] += 1;
                if first_bad.is_none() {
                    first_bad = Some((batch, row, a, b));
                }
            }
        }
    }

    use std::fmt::Write;
    let _ = write!(
        report,
        "    {name}: max_abs={max_abs:.4e} max_rel={max_rel:.4e} bad={n_bad}/{}",
        n * m
    );
    if n_bad > 0 {
        let _ = writeln!(report);
        if let Some((b, r, a, ref_v)) = first_bad {
            let _ = writeln!(
                report,
                "      first mismatch at (batch={b}, row={r}): cand={a:.4} ref={ref_v:.4} diff={:.4e}",
                a - ref_v
            );
        }
        let _ = writeln!(report, "      mismatches by (row % 16): {hist_row_mod16:?}");
        false
    } else {
        let _ = writeln!(report);
        true
    }
}

/// Build deterministic HFQ4G256 weight bytes (same helper as
/// test_wmma_qkv_gfx12; duplicated here so each example is self-contained).
fn build_hfq4g256(m: usize, k: usize, seed: u8) -> Vec<u8> {
    assert_eq!(k % 256, 0);
    let groups_per_row = k / 256;
    let bytes_per_row = groups_per_row * 136;
    let mut out = vec![0u8; m * bytes_per_row];

    let mix = |x: u64| {
        let h = x
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((h ^ (h >> 33)).wrapping_mul(0xff51afd7ed558ccd)) ^ (h >> 28)
    };
    let s0 = seed as u64;

    for row in 0..m {
        for g in 0..groups_per_row {
            let off = row * bytes_per_row + g * 136;
            let r1 = mix(s0 ^ ((row as u64) << 16) ^ (g as u64));
            let r2 = mix(s0 ^ ((row as u64) * 7 + g as u64));
            let scale = 0.01 + (((r1 as u32) % 4001) as f32) * 1e-5;
            let zero = (((r2 as u32) % 1500) as f32) * 1e-4 - 0.075;

            out[off..off + 4].copy_from_slice(&scale.to_le_bytes());
            out[off + 4..off + 8].copy_from_slice(&zero.to_le_bytes());

            for byte_i in 0..128 {
                let r = mix(s0 ^ ((row as u64) << 24) ^ ((g as u64) << 12) ^ (byte_i as u64));
                out[off + 8 + byte_i] = (r & 0xff) as u8;
            }
        }
    }
    out
}
