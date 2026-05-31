// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Robin Van Cauter
// hipfire — see LICENSE and NOTICE in the project root.

//! Channel-test for the gfx12 (RDNA4) WMMA residual GEMM.
//!
//! Compiles `gemm_hfq4g256_residual_wmma.gfx12.hip` and compares its output
//! against the validated dot2-fp16 reference (`gemm_hfq4g256_residual_fp16`)
//! on identical synthetic inputs. The fp16 path is the current gfx12
//! production fallback (gfx12 dispatch falls through to it before this PR),
//! so any divergence from it would be a real correctness regression.
//!
//! What this validates:
//!   - The kernel compiles for gfx1200 / gfx1201.
//!   - The C-output mapping
//!     (`acc[j] = C[8*(tid>>4) + j][tid & 15]`) is correct on silicon.
//!   - The K-split across lane-groups (k_grp = tid >> 4) reads the
//!     correct half of each K-tile.
//!   - Residual-add semantics (`Y += W·X` not `Y = W·X`) match the dot2
//!     reference.
//!
//! Bails with a clear message on non-gfx12 archs (this kernel uses the
//! `_w32_gfx12` builtin which does not exist on gfx11).
//!
//! Run: cargo run --release --features deltanet -p hipfire-runtime \
//!         --example test_wmma_residual_gfx12

use rdna_compute::Gpu;

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

    // Sweep shapes that exercise the four lever points of the kernel:
    //   - one row-tile, one batch-tile, one K-group (smallest case)
    //   - multiple K-groups (the K accumulation loop)
    //   - multiple batch-tiles (the second grid dim)
    //   - multiple row-tiles (the first grid dim)
    //   - all dims multi-tile (combined coverage)
    //   - shape that mirrors a real 9B residual call site (intermediate=12288 → dim=4096)
    let shapes: &[(usize, usize, usize)] = &[
        // (M, K, N)
        (16, 256, 16),       // minimal: 1 row-tile, 1 K-grp, 1 batch-tile
        (16, 512, 16),       // K=2 groups: exercises K accumulation
        (16, 256, 32),       // batch=2 tiles
        (32, 256, 16),       // row=2 tiles
        (32, 512, 32),       // multi-tile in every dim
        (64, 1024, 32),      // larger but still tractable
        (4096, 1024, 16),    // 9B-shape band: exercises real M/N ratio at small K
    ];

    for &(m, k, n) in shapes {
        let label = format!("M={m} K={k} N={n}");
        eprintln!("\n--- {label} ---");
        match run_one(&mut gpu, m, k, n) {
            Ok(()) => {
                total_pass += 1;
                eprintln!("  residual {label:41} OK");
            }
            Err(e) => {
                total_fail += 1;
                eprintln!("  residual {label:41} FAIL");
                eprintln!("{e}");
            }
        }
        match run_one_lmhead(&mut gpu, m, k, n) {
            Ok(()) => {
                total_pass += 1;
                eprintln!("  lmhead   {label:41} OK");
            }
            Err(e) => {
                total_fail += 1;
                eprintln!("  lmhead   {label:41} FAIL");
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

fn run_one_lmhead(gpu: &mut Gpu, m: usize, k: usize, n: usize) -> Result<(), String> {
    assert_eq!(m % 16, 0);
    assert_eq!(k % 256, 0, "K must be multiple of 256 (HFQ4G256 group size)");
    assert_eq!(n % 16, 0, "N must be multiple of 16 (WMMA batch tile)");

    let a_bytes = build_hfq4g256(m, k, 0xB4);
    let a = gpu
        .upload_raw(&a_bytes, &[m, k])
        .map_err(|e| format!("upload a: {e}"))?;

    let x_f32: Vec<f32> = (0..(n * k))
        .map(|i| {
            let b = (i / k) as i32;
            let kk = (i % k) as i32;
            ((b * 5 + kk * 3) % 37 - 18) as f32 * 0.04
        })
        .collect();
    let x = gpu
        .upload_f32(&x_f32, &[n, k])
        .map_err(|e| format!("upload x: {e}"))?;

    let y_zero = vec![0.0f32; n * m];
    let y_ref = gpu
        .upload_f32(&y_zero, &[n, m])
        .map_err(|e| format!("upload y_ref init: {e}"))?;
    gpu.gemm_hfq4g256_residual_fp16(&a, &x, &y_ref, m, k, n)
        .map_err(|e| format!("dot2-fp16 residual zero-init: {e}"))?;
    let ref_y = gpu
        .download_f32(&y_ref)
        .map_err(|e| format!("download y_ref: {e}"))?;

    let y_dirty: Vec<f32> = (0..(n * m))
        .map(|i| ((i * 19) % 29) as f32 * 0.01 + 0.25)
        .collect();
    let y_cand = gpu
        .upload_f32(&y_dirty, &[n, m])
        .map_err(|e| format!("upload y_cand dirty init: {e}"))?;
    gpu.gemm_hfq4g256_lmhead_wmma_gfx12(&a, &x, &y_cand, m, k, n)
        .map_err(|e| format!("wmma_gfx12 lmhead: {e}"))?;
    let cand_y = gpu
        .download_f32(&y_cand)
        .map_err(|e| format!("download y_cand: {e}"))?;

    gpu.free_tensor(a).ok();
    gpu.free_tensor(x).ok();
    gpu.free_tensor(y_ref).ok();
    gpu.free_tensor(y_cand).ok();

    let mut report = String::new();
    if compare("Y_lmhead", n, m, &cand_y, &ref_y, &mut report) {
        Ok(())
    } else {
        Err(report)
    }
}

fn run_one(gpu: &mut Gpu, m: usize, k: usize, n: usize) -> Result<(), String> {
    assert_eq!(m % 16, 0);
    assert_eq!(k % 256, 0, "K must be multiple of 256 (HFQ4G256 group size)");
    assert_eq!(n % 16, 0, "N must be multiple of 16 (WMMA batch tile)");

    // ── Build synthetic HFQ4G256 weight bytes ──────────────────────────────
    let a_bytes = build_hfq4g256(m, k, 0xD7);
    let a = gpu
        .upload_raw(&a_bytes, &[m, k])
        .map_err(|e| format!("upload a: {e}"))?;

    // ── X as f32 (the dispatch wrapper converts to fp16 internally) ────────
    // Distinct values per (batch, k) to surface any row/col-swap mapping bug
    // similar to the gfx11 6-week silent corruption (commit b7ac66a).
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

    // ── Pre-residual Y init (the "skip connection" value Y starts at) ──────
    // Use a non-zero pattern so the residual `+=` semantics get exercised:
    // a kernel that overwrites instead of adding would silently match a
    // zeros pre-init.
    let y_init: Vec<f32> = (0..(n * m))
        .map(|i| {
            let b = (i / m) as i32;
            let r = (i % m) as i32;
            ((b * 13 + r * 17) % 23 - 11) as f32 * 0.01
        })
        .collect();

    // ── Reference: dot2-fp16 path (current gfx12 production fallback) ──────
    // upload_f32 allocates + initializes in one shot, so each path gets a
    // fresh Y seeded with `y_init` (testing residual `+=` semantics).
    let y_ref = gpu
        .upload_f32(&y_init, &[n, m])
        .map_err(|e| format!("upload y_ref init: {e}"))?;
    gpu.gemm_hfq4g256_residual_fp16(&a, &x, &y_ref, m, k, n)
        .map_err(|e| format!("dot2-fp16 residual: {e}"))?;
    let ref_y = gpu
        .download_f32(&y_ref)
        .map_err(|e| format!("download y_ref: {e}"))?;

    // ── Candidate: gfx12 WMMA residual ─────────────────────────────────────
    let y_cand = gpu
        .upload_f32(&y_init, &[n, m])
        .map_err(|e| format!("upload y_cand init: {e}"))?;
    gpu.gemm_hfq4g256_residual_wmma_gfx12(&a, &x, &y_cand, m, k, n)
        .map_err(|e| format!("wmma_gfx12 residual: {e}"))?;
    let cand_y = gpu
        .download_f32(&y_cand)
        .map_err(|e| format!("download y_cand: {e}"))?;

    gpu.free_tensor(a).ok();
    gpu.free_tensor(x).ok();
    gpu.free_tensor(y_ref).ok();
    gpu.free_tensor(y_cand).ok();

    // ── Compare ────────────────────────────────────────────────────────────
    // Tolerance: WMMA does fp16×fp16→fp32 fma; dot2 does the same algebra
    // but with a different operation order. Differences are accumulated
    // rounding noise. Same band as test_wmma_qkv_gfx12.
    let mut report = String::new();
    if compare("Y", n, m, &cand_y, &ref_y, &mut report) {
        Ok(())
    } else {
        Err(report)
    }
}

fn compare(name: &str, n: usize, m: usize, cand: &[f32], refr: &[f32], report: &mut String) -> bool {
    assert_eq!(cand.len(), refr.len());
    assert_eq!(cand.len(), n * m);

    let mut max_abs = 0f32;
    let mut max_rel = 0f32;
    let mut n_bad = 0usize;
    let mut first_bad: Option<(usize, usize, f32, f32)> = None;
    let abs_tol = 5e-2_f32;
    let rel_tol = 1e-2_f32;

    // Per-row-mod-16 and per-batch-mod-16 mismatch histograms. A clustering
    // in {0..7} or {8..15} on either axis points at a lane-group → output
    // dimension mapping bug (the QKV port hit one of these during R9700
    // bring-up — see PR #56 channel-test scaffold).
    let mut hist_row_mod16 = [0usize; 16];
    let mut hist_batch_mod16 = [0usize; 16];

    for batch in 0..n {
        for row in 0..m {
            // Layout is [N × M] row-major: y[batch, row] = data[batch*M + row]
            let idx = batch * m + row;
            let a = cand[idx];
            let b = refr[idx];
            let abs = (a - b).abs();
            let rel = abs / b.abs().max(1e-3);
            if abs > max_abs {
                max_abs = abs;
            }
            if rel > max_rel {
                max_rel = rel;
            }
            if abs > abs_tol && rel > rel_tol {
                n_bad += 1;
                hist_row_mod16[row % 16] += 1;
                hist_batch_mod16[batch % 16] += 1;
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
        let _ = writeln!(report, "      mismatches by (row % 16):   {hist_row_mod16:?}");
        let _ = writeln!(report, "      mismatches by (batch % 16): {hist_batch_mod16:?}");
        false
    } else {
        let _ = writeln!(report);
        true
    }
}

/// Build deterministic HFQ4G256 weight bytes for an [m × k] matrix.
/// Layout per group (256 elems): 4B f32 scale | 4B f32 zero | 128B nibbles.
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
