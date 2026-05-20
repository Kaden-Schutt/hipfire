// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Robin Van Cauter
// hipfire — see LICENSE and NOTICE in the project root.

//! Channel-test for the gfx12 (RDNA4) WMMA QKVZA HFQ6 scaffold.
//!
//! Final kernel in the gfx11 → gfx12 WMMA port series. Combines the
//! hfq6 weight builder (from test_wmma_qkv_hfq6_gfx12) with the
//! 4-output qkvza routing coverage (from test_wmma_qkvza_gfx12).
//! Compares against the validated `gemm_qkvza_hfq6g256_dot2` reference.
//!
//! Run: cargo run --release --features deltanet -p hipfire-runtime \
//!         --example test_wmma_qkvza_hfq6_gfx12

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

    let shapes: &[(usize, usize, usize, usize, usize, usize)] = &[
        (16, 16, 16, 16, 256, 16),
        (16, 16, 16, 16, 512, 16),
        (16, 16, 16, 16, 256, 32),
        (32, 16, 16, 16, 256, 16),
        (32, 32, 16, 16, 512, 32),
        (48, 16, 16, 16, 256, 16),
    ];

    for &(qkv_m, z_m, beta_m, alpha_m, k, n) in shapes {
        let label = format!("qkv={qkv_m} z={z_m} b={beta_m} a={alpha_m} K={k} N={n}");
        eprintln!("\n--- {label} ---");
        match run_one(&mut gpu, qkv_m, z_m, beta_m, alpha_m, k, n) {
            Ok(()) => {
                total_pass += 1;
                eprintln!("  {label:60} OK");
            }
            Err(e) => {
                total_fail += 1;
                eprintln!("  {label:60} FAIL");
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
    qkv_m: usize,
    z_m: usize,
    beta_m: usize,
    alpha_m: usize,
    k: usize,
    n: usize,
) -> Result<(), String> {
    assert_eq!(qkv_m % 16, 0);
    assert_eq!(z_m % 16, 0);
    assert_eq!(beta_m % 16, 0);
    assert_eq!(alpha_m % 16, 0);
    assert_eq!(k % 256, 0);
    assert_eq!(n % 16, 0);

    let aq_bytes = build_hfq6g256(qkv_m, k, 0xA1);
    let az_bytes = build_hfq6g256(z_m, k, 0xB2);
    let ab_bytes = build_hfq6g256(beta_m, k, 0xC3);
    let aa_bytes = build_hfq6g256(alpha_m, k, 0xD4);

    let aq = gpu.upload_raw(&aq_bytes, &[qkv_m, k]).map_err(|e| format!("upload aq: {e}"))?;
    let az = gpu.upload_raw(&az_bytes, &[z_m, k]).map_err(|e| format!("upload az: {e}"))?;
    let ab = gpu.upload_raw(&ab_bytes, &[beta_m, k]).map_err(|e| format!("upload ab: {e}"))?;
    let aa = gpu.upload_raw(&aa_bytes, &[alpha_m, k]).map_err(|e| format!("upload aa: {e}"))?;

    let x_f32: Vec<f32> = (0..(n * k))
        .map(|i| {
            let b = (i / k) as i32;
            let kk = (i % k) as i32;
            ((b * 7 + kk * 11) % 31 - 15) as f32 * 0.05
        })
        .collect();
    let x = gpu.upload_f32(&x_f32, &[n, k]).map_err(|e| format!("upload x: {e}"))?;

    let yq_ref = gpu.alloc_tensor(&[n, qkv_m], DType::F32).map_err(|e| format!("alloc yq_ref: {e}"))?;
    let yz_ref = gpu.alloc_tensor(&[n, z_m], DType::F32).map_err(|e| format!("alloc yz_ref: {e}"))?;
    let yb_ref = gpu.alloc_tensor(&[n, beta_m], DType::F32).map_err(|e| format!("alloc yb_ref: {e}"))?;
    let ya_ref = gpu.alloc_tensor(&[n, alpha_m], DType::F32).map_err(|e| format!("alloc ya_ref: {e}"))?;

    gpu.gemm_qkvza_hfq6g256_dot2(
        &aq, &az, &ab, &aa, &x,
        &yq_ref, &yz_ref, &yb_ref, &ya_ref,
        qkv_m, z_m, beta_m, alpha_m, k, n,
    )
    .map_err(|e| format!("dot2: {e}"))?;

    let ref_q = gpu.download_f32(&yq_ref).map_err(|e| format!("download yq_ref: {e}"))?;
    let ref_z = gpu.download_f32(&yz_ref).map_err(|e| format!("download yz_ref: {e}"))?;
    let ref_b = gpu.download_f32(&yb_ref).map_err(|e| format!("download yb_ref: {e}"))?;
    let ref_a = gpu.download_f32(&ya_ref).map_err(|e| format!("download ya_ref: {e}"))?;

    let yq = gpu.alloc_tensor(&[n, qkv_m], DType::F32).map_err(|e| format!("alloc yq: {e}"))?;
    let yz = gpu.alloc_tensor(&[n, z_m], DType::F32).map_err(|e| format!("alloc yz: {e}"))?;
    let yb = gpu.alloc_tensor(&[n, beta_m], DType::F32).map_err(|e| format!("alloc yb: {e}"))?;
    let ya = gpu.alloc_tensor(&[n, alpha_m], DType::F32).map_err(|e| format!("alloc ya: {e}"))?;

    gpu.gemm_qkvza_hfq6g256_wmma_gfx12(
        &aq, &az, &ab, &aa, &x,
        &yq, &yz, &yb, &ya,
        qkv_m, z_m, beta_m, alpha_m, k, n,
    )
    .map_err(|e| format!("wmma_gfx12: {e}"))?;

    let cand_q = gpu.download_f32(&yq).map_err(|e| format!("download yq: {e}"))?;
    let cand_z = gpu.download_f32(&yz).map_err(|e| format!("download yz: {e}"))?;
    let cand_b = gpu.download_f32(&yb).map_err(|e| format!("download yb: {e}"))?;
    let cand_a = gpu.download_f32(&ya).map_err(|e| format!("download ya: {e}"))?;

    gpu.free_tensor(aq).ok();
    gpu.free_tensor(az).ok();
    gpu.free_tensor(ab).ok();
    gpu.free_tensor(aa).ok();
    gpu.free_tensor(x).ok();
    gpu.free_tensor(yq_ref).ok();
    gpu.free_tensor(yz_ref).ok();
    gpu.free_tensor(yb_ref).ok();
    gpu.free_tensor(ya_ref).ok();
    gpu.free_tensor(yq).ok();
    gpu.free_tensor(yz).ok();
    gpu.free_tensor(yb).ok();
    gpu.free_tensor(ya).ok();

    let mut report = String::new();
    let ok_q = compare_proj("Y_qkv", n, qkv_m, &cand_q, &ref_q, &mut report);
    let ok_z = compare_proj("Y_z", n, z_m, &cand_z, &ref_z, &mut report);
    let ok_b = compare_proj("Y_beta", n, beta_m, &cand_b, &ref_b, &mut report);
    let ok_a = compare_proj("Y_alpha", n, alpha_m, &cand_a, &ref_a, &mut report);

    if ok_q && ok_z && ok_b && ok_a {
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

/// Build deterministic HFQ6G256 weight bytes (same helper as the other
/// hfq6 tests; duplicated so each example is self-contained).
fn build_hfq6g256(m: usize, k: usize, seed: u8) -> Vec<u8> {
    assert_eq!(k % 256, 0);
    let groups_per_row = k / 256;
    let bytes_per_row = groups_per_row * 200;
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
            let off = row * bytes_per_row + g * 200;
            let r1 = mix(s0 ^ ((row as u64) << 16) ^ (g as u64));
            let r2 = mix(s0 ^ ((row as u64) * 7 + g as u64));
            let scale = 0.005 + (((r1 as u32) % 1500) as f32) * 1e-5;
            let zero = (((r2 as u32) % 12000) as f32) * 1e-4 - 0.6;

            out[off..off + 4].copy_from_slice(&scale.to_le_bytes());
            out[off + 4..off + 8].copy_from_slice(&zero.to_le_bytes());

            let mut vals = [0u8; 256];
            for (i, slot) in vals.iter_mut().enumerate() {
                let r = mix(s0 ^ ((row as u64) << 24) ^ ((g as u64) << 12) ^ (i as u64));
                *slot = (r & 0x3f) as u8;
            }
            for chunk in 0..64 {
                let v0 = vals[chunk * 4] as u32;
                let v1 = vals[chunk * 4 + 1] as u32;
                let v2 = vals[chunk * 4 + 2] as u32;
                let v3 = vals[chunk * 4 + 3] as u32;
                let bits = v0 | (v1 << 6) | (v2 << 12) | (v3 << 18);
                out[off + 8 + chunk * 3] = (bits & 0xff) as u8;
                out[off + 8 + chunk * 3 + 1] = ((bits >> 8) & 0xff) as u8;
                out[off + 8 + chunk * 3 + 2] = ((bits >> 16) & 0xff) as u8;
            }
        }
    }
    out
}
