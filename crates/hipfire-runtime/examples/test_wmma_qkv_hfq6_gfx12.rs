// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Robin Van Cauter
// hipfire — see LICENSE and NOTICE in the project root.

//! Channel-test for the gfx12 (RDNA4) WMMA QKV HFQ6 scaffold.
//!
//! Sister to test_wmma_qkv_gfx12 (which exercises HFQ4). This one
//! validates the HFQ6 dequant inner loop on top of the same gfx12
//! WMMA recipe. Compares against the validated
//! `gemm_qkv_hfq6g256_dot2` reference on synthetic HFQ6G256 weights
//! + f16 X.
//!
//! HFQ6G256 group layout: 200 bytes / 256 elements
//!   [0..4)   f32 scale
//!   [4..8)   f32 zero
//!   [8..200) 192B packed 6-bit values
//!
//! Run: cargo run --release --features deltanet -p hipfire-runtime \
//!         --example test_wmma_qkv_hfq6_gfx12

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

    let shapes: &[(usize, usize, usize, usize, usize)] = &[
        (16, 16, 16, 256, 16),
        (16, 16, 16, 512, 16),
        (16, 16, 16, 256, 32),
        (32, 16, 16, 256, 16),
        (32, 32, 32, 512, 32),
        (48, 16, 16, 256, 16),
    ];

    for &(q_m, k_m, v_m, k, n) in shapes {
        let label = format!("q={q_m} k={k_m} v={v_m} K={k} N={n}");
        eprintln!("\n--- {label} ---");
        match run_one(&mut gpu, q_m, k_m, v_m, k, n) {
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
    q_m: usize,
    k_m: usize,
    v_m: usize,
    k: usize,
    n: usize,
) -> Result<(), String> {
    assert_eq!(q_m % 16, 0);
    assert_eq!(k_m % 16, 0);
    assert_eq!(v_m % 16, 0);
    assert_eq!(k % 256, 0);
    assert_eq!(n % 16, 0);

    let aq_bytes = build_hfq6g256(q_m, k, 0xA1);
    let ak_bytes = build_hfq6g256(k_m, k, 0xB2);
    let av_bytes = build_hfq6g256(v_m, k, 0xC3);

    let aq = gpu.upload_raw(&aq_bytes, &[q_m, k]).map_err(|e| format!("upload aq: {e}"))?;
    let ak = gpu.upload_raw(&ak_bytes, &[k_m, k]).map_err(|e| format!("upload ak: {e}"))?;
    let av = gpu.upload_raw(&av_bytes, &[v_m, k]).map_err(|e| format!("upload av: {e}"))?;

    let x_f32: Vec<f32> = (0..(n * k))
        .map(|i| {
            let b = (i / k) as i32;
            let kk = (i % k) as i32;
            ((b * 7 + kk * 11) % 31 - 15) as f32 * 0.05
        })
        .collect();
    let x = gpu.upload_f32(&x_f32, &[n, k]).map_err(|e| format!("upload x: {e}"))?;

    let y_q_ref = gpu.alloc_tensor(&[n, q_m], DType::F32).map_err(|e| format!("alloc yq_ref: {e}"))?;
    let y_k_ref = gpu.alloc_tensor(&[n, k_m], DType::F32).map_err(|e| format!("alloc yk_ref: {e}"))?;
    let y_v_ref = gpu.alloc_tensor(&[n, v_m], DType::F32).map_err(|e| format!("alloc yv_ref: {e}"))?;

    gpu.gemm_qkv_hfq6g256_dot2(
        &aq, &ak, &av, &x, &y_q_ref, &y_k_ref, &y_v_ref, q_m, k_m, v_m, k, n,
    )
    .map_err(|e| format!("dot2: {e}"))?;

    let ref_q = gpu.download_f32(&y_q_ref).map_err(|e| format!("download yq_ref: {e}"))?;
    let ref_k = gpu.download_f32(&y_k_ref).map_err(|e| format!("download yk_ref: {e}"))?;
    let ref_v = gpu.download_f32(&y_v_ref).map_err(|e| format!("download yv_ref: {e}"))?;

    let y_q = gpu.alloc_tensor(&[n, q_m], DType::F32).map_err(|e| format!("alloc yq: {e}"))?;
    let y_k = gpu.alloc_tensor(&[n, k_m], DType::F32).map_err(|e| format!("alloc yk: {e}"))?;
    let y_v = gpu.alloc_tensor(&[n, v_m], DType::F32).map_err(|e| format!("alloc yv: {e}"))?;

    gpu.gemm_qkv_hfq6g256_wmma_gfx12(
        &aq, &ak, &av, &x, &y_q, &y_k, &y_v, q_m, k_m, v_m, k, n,
    )
    .map_err(|e| format!("wmma_gfx12: {e}"))?;

    let cand_q = gpu.download_f32(&y_q).map_err(|e| format!("download yq: {e}"))?;
    let cand_k = gpu.download_f32(&y_k).map_err(|e| format!("download yk: {e}"))?;
    let cand_v = gpu.download_f32(&y_v).map_err(|e| format!("download yv: {e}"))?;

    gpu.free_tensor(aq).ok();
    gpu.free_tensor(ak).ok();
    gpu.free_tensor(av).ok();
    gpu.free_tensor(x).ok();
    gpu.free_tensor(y_q_ref).ok();
    gpu.free_tensor(y_k_ref).ok();
    gpu.free_tensor(y_v_ref).ok();
    gpu.free_tensor(y_q).ok();
    gpu.free_tensor(y_k).ok();
    gpu.free_tensor(y_v).ok();

    let mut report = String::new();
    let ok_q = compare_proj("Y_q", n, q_m, &cand_q, &ref_q, &mut report);
    let ok_k = compare_proj("Y_k", n, k_m, &cand_k, &ref_k, &mut report);
    let ok_v = compare_proj("Y_v", n, v_m, &cand_v, &ref_v, &mut report);

    if ok_q && ok_k && ok_v {
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

/// Build deterministic HFQ6G256 weight bytes for an [m × k] matrix.
/// Layout per group (256 elems): 4B f32 scale | 4B f32 zero | 192B packed 6-bit.
/// Each pair of 4 6-bit values takes 3 bytes (24 bits).
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

            // Scale ≈ small random in [0.005, 0.02]; zero ≈ small random in
            // [-0.6, +0.6] (6-bit values up to 63 give a wider product range
            // than 4-bit, so dial scale down for similar |w|).
            let r1 = mix(s0 ^ ((row as u64) << 16) ^ (g as u64));
            let r2 = mix(s0 ^ ((row as u64) * 7 + g as u64));
            let scale = 0.005 + (((r1 as u32) % 1500) as f32) * 1e-5; // [0.005, 0.02]
            let zero = (((r2 as u32) % 12000) as f32) * 1e-4 - 0.6;   // [-0.6, +0.6]

            out[off..off + 4].copy_from_slice(&scale.to_le_bytes());
            out[off + 4..off + 8].copy_from_slice(&zero.to_le_bytes());

            // Pack 256 6-bit values into 192 bytes.
            // Strategy: generate 256 6-bit values, pack 4 values (24 bits) per
            // 3-byte group → 64 such groups for the 192 bytes.
            let mut vals = [0u8; 256];
            for (i, slot) in vals.iter_mut().enumerate() {
                let r = mix(s0 ^ ((row as u64) << 24) ^ ((g as u64) << 12) ^ (i as u64));
                *slot = (r & 0x3f) as u8; // 0..63
            }
            for chunk in 0..64 {
                let v0 = vals[chunk * 4] as u32;
                let v1 = vals[chunk * 4 + 1] as u32;
                let v2 = vals[chunk * 4 + 2] as u32;
                let v3 = vals[chunk * 4 + 3] as u32;
                let bits = v0 | (v1 << 6) | (v2 << 12) | (v3 << 18); // 24 bits
                out[off + 8 + chunk * 3] = (bits & 0xff) as u8;
                out[off + 8 + chunk * 3 + 1] = ((bits >> 8) & 0xff) as u8;
                out[off + 8 + chunk * 3 + 2] = ((bits >> 16) & 0xff) as u8;
            }
        }
    }
    out
}
