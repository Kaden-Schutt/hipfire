// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Robin Van Cauter
// hipfire — see LICENSE and NOTICE in the project root.

//! Numerical parity test for the gfx11 LDS-staged HFQ4-G256 WMMA kernels.
//!
//! Each case runs a validated non-staged reference followed by the staged
//! candidate through the public dispatch entry point with its feature flag
//! forced on. This exercises the real K/batch dispatch guard without relying
//! on ambient process configuration.
//!
//! Run: cargo run --release --features deltanet -p hipfire-runtime \
//!         --example test_wmma_ldsstage_gfx11

use rdna_compute::Gpu;
use std::sync::Arc;

const ABS_TOL: f32 = 5.0e-4;
const REL_TOL: f32 = 1.0e-5;
const REL_NEAR_ZERO: f32 = 1.0e-3;

#[derive(Debug)]
struct Metrics {
    max_abs_err: f32,
    max_rel_err: f32,
    differing: usize,
    total: usize,
    row_mod16: [usize; 16],
    batch_mod16: [usize; 16],
    row_parity: [usize; 2],
    first_differences: Vec<(usize, usize, f32, f32, f32)>,
}

impl Metrics {
    fn passed(&self) -> bool {
        self.differing == 0
    }
}

fn main() {
    let mut gpu = Gpu::init().expect("GPU init failed");
    let arch = gpu.arch.clone();
    let (_, total_vram) = gpu.hip.get_vram_info().unwrap_or((0, 0));

    println!("GPU arch={arch} vram_gb={:.1}", total_vram as f64 / 1e9);
    println!(
        "TOLERANCE abs_tol={ABS_TOL:.1e} rel_tol={REL_TOL:.1e} \
         max_rel_excludes_abs_ref_below={REL_NEAR_ZERO:.1e}"
    );
    println!(
        "RATIONALE staged reduction reorders FP32 accumulation; expected relative drift is around \
         1e-6, so 1e-5 relative plus 5e-4 absolute headroom for long-K cancellation rejects \
         mapping errors without requiring bit-exact output (and is 100x tighter than the \
         established gfx12 harness's 5e-2 absolute tolerance)"
    );
    println!("DISPATCH reference_flag=false candidate_flag=true; candidate uses the public guarded entry point");
    println!("RESIDUAL_SEMANTICS seeded_nonzero_y=true expected=Y_seed+W*X");
    println!("GATE_UP_SEMANTICS seeded_nonzero_y=true expected=overwrite_with_W*X");

    if !arch.starts_with("gfx11") {
        eprintln!(
            "SKIP: this example requires a gfx11-family wave32 WMMA device; current arch={arch}"
        );
        std::process::exit(2);
    }

    let residual_shapes = [
        // (M, K, batch). Real 5120-wide projection, guard ceiling, and row tail.
        (5120usize, 5120usize, 16usize),
        (5120, 5120, 64),
        (1000, 1024, 16),
    ];
    let gate_up_shapes = [
        // (gate_m, up_m, K, batch). Equal projections mirror the gfx12 harness.
        (5120usize, 5120usize, 5120usize, 16usize),
        (5120, 5120, 5120, 64),
        // gate_m=1000 makes a 16-row tile straddle the gate/up boundary.
        (1000, 1000, 1024, 16),
    ];

    let mut cases_passed = 0usize;
    let mut cases_failed = 0usize;

    for &(m, k, batch) in &residual_shapes {
        match run_residual(&mut gpu, m, k, batch) {
            Ok(metrics) => {
                print_case(
                    "residual",
                    &format!("M={m} K={k} batch={batch}"),
                    None,
                    &metrics,
                );
                if metrics.passed() {
                    cases_passed += 1;
                } else {
                    cases_failed += 1;
                }
            }
            Err(error) => {
                cases_failed += 1;
                println!(
                    "RESULT kernel=residual shape=\"M={m} K={k} batch={batch}\" verdict=FAIL error={error:?}"
                );
            }
        }
    }

    for &(gate_m, up_m, k, batch) in &gate_up_shapes {
        let shape = format!("gate_m={gate_m} up_m={up_m} K={k} batch={batch}");
        match run_gate_up(&mut gpu, gate_m, up_m, k, batch) {
            Ok((gate_metrics, up_metrics)) => {
                print_case("gate_up", &shape, Some("gate"), &gate_metrics);
                print_case("gate_up", &shape, Some("up"), &up_metrics);
                let passed = gate_metrics.passed() && up_metrics.passed();
                println!(
                    "RESULT_COMBINED kernel=gate_up shape=\"{shape}\" verdict={} \
                     differing={}/{}",
                    if passed { "PASS" } else { "FAIL" },
                    gate_metrics.differing + up_metrics.differing,
                    gate_metrics.total + up_metrics.total,
                );
                if passed {
                    cases_passed += 1;
                } else {
                    cases_failed += 1;
                }
            }
            Err(error) => {
                cases_failed += 1;
                println!(
                    "RESULT_COMBINED kernel=gate_up shape=\"{shape}\" verdict=FAIL error={error:?}"
                );
            }
        }
    }

    println!(
        "SUMMARY arch={arch} cases_passed={cases_passed} cases_failed={cases_failed} verdict={}",
        if cases_failed == 0 { "PASS" } else { "FAIL" }
    );
    if cases_failed != 0 {
        std::process::exit(1);
    }
}

fn set_ldsstage(gpu: &mut Gpu, enabled: bool) {
    let mut flags = (*gpu.flags).clone();
    flags.hfq4g256_ldsstage_wmma = enabled;
    // The MW16 branch precedes the staged residual guard. Keep it disabled so
    // enabled=true unambiguously reaches the kernel under test.
    flags.mw16 = false;
    gpu.flags = Arc::new(flags);
}

fn run_residual(gpu: &mut Gpu, m: usize, k: usize, batch: usize) -> Result<Metrics, String> {
    assert_eq!(k % 512, 0, "staged guard requires K % 512 == 0");
    assert!(batch <= 64, "staged gfx11 guard requires batch <= 64");

    let a_bytes = build_hfq4g256(m, k, 0xD7);
    let a = gpu
        .upload_raw(&a_bytes, &[m, k])
        .map_err(|e| format!("upload residual weights: {e}"))?;

    let x_f32: Vec<f32> = (0..(batch * k))
        .map(|i| {
            let b = (i / k) as i32;
            let kk = (i % k) as i32;
            ((b * 7 + kk * 11) % 31 - 15) as f32 * 0.05
        })
        .collect();
    let x = gpu
        .upload_f32(&x_f32, &[batch, k])
        .map_err(|e| format!("upload residual X: {e}"))?;

    // Non-zero seed makes an accidental overwrite fail even if W*X is correct.
    let y_seed: Vec<f32> = (0..(batch * m))
        .map(|i| {
            let b = (i / m) as i32;
            let row = (i % m) as i32;
            ((b * 13 + row * 17) % 23 - 11) as f32 * 0.01
        })
        .collect();

    set_ldsstage(gpu, false);
    let y_ref = gpu
        .upload_f32(&y_seed, &[batch, m])
        .map_err(|e| format!("upload residual reference Y: {e}"))?;
    gpu.gemm_hfq4g256_residual_wmma(&a, &x, &y_ref, m, k, batch)
        .map_err(|e| format!("residual non-staged reference launch: {e}"))?;
    let reference = gpu
        .download_f32(&y_ref)
        .map_err(|e| format!("download residual reference Y: {e}"))?;

    set_ldsstage(gpu, true);
    let y_candidate = gpu
        .upload_f32(&y_seed, &[batch, m])
        .map_err(|e| format!("upload residual candidate Y: {e}"))?;
    gpu.gemm_hfq4g256_residual_wmma(&a, &x, &y_candidate, m, k, batch)
        .map_err(|e| format!("residual staged dispatch launch: {e}"))?;
    let candidate = gpu
        .download_f32(&y_candidate)
        .map_err(|e| format!("download residual candidate Y: {e}"))?;

    set_ldsstage(gpu, false);
    gpu.free_tensor(a).ok();
    gpu.free_tensor(x).ok();
    gpu.free_tensor(y_ref).ok();
    gpu.free_tensor(y_candidate).ok();

    Ok(compare(batch, m, &candidate, &reference))
}

fn run_gate_up(
    gpu: &mut Gpu,
    gate_m: usize,
    up_m: usize,
    k: usize,
    batch: usize,
) -> Result<(Metrics, Metrics), String> {
    assert_eq!(k % 512, 0, "staged guard requires K % 512 == 0");
    assert!(batch <= 64, "staged gfx11 guard requires batch <= 64");

    let gate_bytes = build_hfq4g256(gate_m, k, 0xD4);
    let up_bytes = build_hfq4g256(up_m, k, 0xE5);
    let a_gate = gpu
        .upload_raw(&gate_bytes, &[gate_m, k])
        .map_err(|e| format!("upload gate weights: {e}"))?;
    let a_up = gpu
        .upload_raw(&up_bytes, &[up_m, k])
        .map_err(|e| format!("upload up weights: {e}"))?;

    let x_f32: Vec<f32> = (0..(batch * k))
        .map(|i| {
            let b = (i / k) as i32;
            let kk = (i % k) as i32;
            ((b * 5 + kk * 13) % 37 - 18) as f32 * 0.04
        })
        .collect();
    let x = gpu
        .upload_f32(&x_f32, &[batch, k])
        .map_err(|e| format!("upload gate_up X: {e}"))?;

    // Seed outputs even though gate_up is an overwrite operation. A candidate
    // that accidentally accumulates into Y will differ from the reference.
    let gate_seed: Vec<f32> = (0..(batch * gate_m))
        .map(|i| 0.5 + ((i * 17) % 31) as f32 * 0.01)
        .collect();
    let up_seed: Vec<f32> = (0..(batch * up_m))
        .map(|i| -0.75 + ((i * 19) % 29) as f32 * 0.01)
        .collect();

    set_ldsstage(gpu, false);
    let y_gate_ref = gpu
        .upload_f32(&gate_seed, &[batch, gate_m])
        .map_err(|e| format!("upload gate reference Y: {e}"))?;
    let y_up_ref = gpu
        .upload_f32(&up_seed, &[batch, up_m])
        .map_err(|e| format!("upload up reference Y: {e}"))?;
    gpu.gemm_gate_up_hfq4g256_wmma(
        &a_gate,
        &a_up,
        &x,
        &y_gate_ref,
        &y_up_ref,
        gate_m,
        up_m,
        k,
        batch,
    )
    .map_err(|e| format!("gate_up non-staged reference launch: {e}"))?;
    let gate_reference = gpu
        .download_f32(&y_gate_ref)
        .map_err(|e| format!("download gate reference Y: {e}"))?;
    let up_reference = gpu
        .download_f32(&y_up_ref)
        .map_err(|e| format!("download up reference Y: {e}"))?;

    set_ldsstage(gpu, true);
    let y_gate_candidate = gpu
        .upload_f32(&gate_seed, &[batch, gate_m])
        .map_err(|e| format!("upload gate candidate Y: {e}"))?;
    let y_up_candidate = gpu
        .upload_f32(&up_seed, &[batch, up_m])
        .map_err(|e| format!("upload up candidate Y: {e}"))?;
    gpu.gemm_gate_up_hfq4g256_wmma(
        &a_gate,
        &a_up,
        &x,
        &y_gate_candidate,
        &y_up_candidate,
        gate_m,
        up_m,
        k,
        batch,
    )
    .map_err(|e| format!("gate_up staged dispatch launch: {e}"))?;
    let gate_candidate = gpu
        .download_f32(&y_gate_candidate)
        .map_err(|e| format!("download gate candidate Y: {e}"))?;
    let up_candidate = gpu
        .download_f32(&y_up_candidate)
        .map_err(|e| format!("download up candidate Y: {e}"))?;

    set_ldsstage(gpu, false);
    gpu.free_tensor(a_gate).ok();
    gpu.free_tensor(a_up).ok();
    gpu.free_tensor(x).ok();
    gpu.free_tensor(y_gate_ref).ok();
    gpu.free_tensor(y_up_ref).ok();
    gpu.free_tensor(y_gate_candidate).ok();
    gpu.free_tensor(y_up_candidate).ok();

    Ok((
        compare(batch, gate_m, &gate_candidate, &gate_reference),
        compare(batch, up_m, &up_candidate, &up_reference),
    ))
}

fn compare(batch: usize, m: usize, candidate: &[f32], reference: &[f32]) -> Metrics {
    assert_eq!(candidate.len(), reference.len());
    assert_eq!(candidate.len(), batch * m);

    let mut metrics = Metrics {
        max_abs_err: 0.0,
        max_rel_err: 0.0,
        differing: 0,
        total: candidate.len(),
        row_mod16: [0; 16],
        batch_mod16: [0; 16],
        row_parity: [0; 2],
        first_differences: Vec::new(),
    };

    for b in 0..batch {
        for row in 0..m {
            let idx = b * m + row;
            let cand = candidate[idx];
            let refr = reference[idx];
            let finite = cand.is_finite() && refr.is_finite();
            let abs = if finite {
                (cand - refr).abs()
            } else {
                f32::INFINITY
            };
            let reference_is_near_zero = refr.abs() < REL_NEAR_ZERO;
            let rel = if finite && !reference_is_near_zero {
                abs / refr.abs()
            } else if finite {
                0.0
            } else {
                f32::INFINITY
            };

            metrics.max_abs_err = metrics.max_abs_err.max(abs);
            if !reference_is_near_zero {
                metrics.max_rel_err = metrics.max_rel_err.max(rel);
            }

            let differs = !finite || (abs > ABS_TOL && (reference_is_near_zero || rel > REL_TOL));
            if differs {
                metrics.differing += 1;
                metrics.row_mod16[row % 16] += 1;
                metrics.batch_mod16[b % 16] += 1;
                metrics.row_parity[row % 2] += 1;
                if metrics.first_differences.len() < 8 {
                    metrics
                        .first_differences
                        .push((b, row, cand, refr, cand - refr));
                }
            }
        }
    }

    metrics
}

fn print_case(kernel: &str, shape: &str, projection: Option<&str>, metrics: &Metrics) {
    let projection_field = projection
        .map(|name| format!(" projection={name}"))
        .unwrap_or_default();
    println!(
        "RESULT kernel={kernel}{projection_field} shape=\"{shape}\" verdict={} \
         max_abs_err={:.8e} max_rel_err={:.8e} differing={}/{} \
         tolerance=\"abs<={ABS_TOL:.1e} OR rel<={REL_TOL:.1e}; rel excludes |ref|<{REL_NEAR_ZERO:.1e}\"",
        if metrics.passed() { "PASS" } else { "FAIL" },
        metrics.max_abs_err,
        metrics.max_rel_err,
        metrics.differing,
        metrics.total,
    );

    if metrics.passed() {
        println!("PATTERN kernel={kernel}{projection_field} structured_error=none");
    } else {
        println!(
            "PATTERN kernel={kernel}{projection_field} row_mod16={:?} batch_mod16={:?} \
             row_parity_even_odd={:?}",
            metrics.row_mod16, metrics.batch_mod16, metrics.row_parity,
        );
        for &(batch, row, cand, refr, diff) in &metrics.first_differences {
            println!(
                "MISMATCH kernel={kernel}{projection_field} batch={batch} row={row} \
                 candidate={cand:.8e} reference={refr:.8e} diff={diff:.8e}"
            );
        }
    }
}

/// Build deterministic HFQ4G256 weight bytes for an [m × k] matrix.
/// Layout per group (256 elements): 4-byte f32 scale, 4-byte f32 zero,
/// then 128 bytes containing two 4-bit values each.
fn build_hfq4g256(m: usize, k: usize, seed: u8) -> Vec<u8> {
    assert_eq!(k % 256, 0);
    let groups_per_row = k / 256;
    let bytes_per_row = groups_per_row * 136;
    let mut output = vec![0u8; m * bytes_per_row];

    let mix = |x: u64| {
        let h = x
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((h ^ (h >> 33)).wrapping_mul(0xff51afd7ed558ccd)) ^ (h >> 28)
    };
    let seed = seed as u64;

    for row in 0..m {
        for group in 0..groups_per_row {
            let offset = row * bytes_per_row + group * 136;
            let r1 = mix(seed ^ ((row as u64) << 16) ^ group as u64);
            let r2 = mix(seed ^ ((row as u64) * 7 + group as u64));
            let scale = 0.01 + ((r1 as u32 % 4001) as f32) * 1e-5;
            let zero = ((r2 as u32 % 1500) as f32) * 1e-4 - 0.075;

            output[offset..offset + 4].copy_from_slice(&scale.to_le_bytes());
            output[offset + 4..offset + 8].copy_from_slice(&zero.to_le_bytes());
            for byte_index in 0..128 {
                let random =
                    mix(seed ^ ((row as u64) << 24) ^ ((group as u64) << 12) ^ byte_index as u64);
                output[offset + 8 + byte_index] = (random & 0xff) as u8;
            }
        }
    }

    output
}
