// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Accuracy probe for Muse Glimmer gfx1100 RM2/BV6 FP16-accumulator OPSEL pair.
//!
//! Compares the frozen candidate host API
//! `Gpu::gemm_hfq4g256_residual_muse_gfx1100_rm2_f16acc_pair` against the
//! F32-accumulator oracle `Gpu::gemm_hfq4g256_residual_muse_gfx1100_rm(rm=2)`
//! at the exact gate/up prefill shape M=19968 K=6656 B=192.
//!
//! Measurement-only: no timing, no production selector, no threshold tuning.
//!
//! Usage:
//!   cargo run --release -p rdna-compute --example probe_glimmer_f16acc_pair
//!
//! Exit nonzero on unsupported launch or any frozen accuracy threshold failure.

use rdna_compute::{DType, Gpu, GpuTensor};

const M: usize = 19_968;
const K: usize = 6_656;
const B: usize = 192;

/// Frozen absolute significance floor: max(1% of |oracle| peak, 1e-3).
const ABS_FLOOR: f32 = 1e-3;
const ABS_FRAC: f32 = 0.01;
/// Frozen relative ceiling on significant elements only.
const MAX_REL_LIMIT: f32 = 0.05;

fn build_hfq4g256(m: usize, k: usize, seed: u8) -> Vec<u8> {
    assert_eq!(k % 256, 0);
    let gpr = k / 256;
    let bpr = gpr * 136;
    let mut out = vec![0u8; m * bpr];
    let mix = |x: u64| {
        let h = x
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((h ^ (h >> 33)).wrapping_mul(0xff51afd7ed558ccd)) ^ (h >> 28)
    };
    let s0 = seed as u64;
    for row in 0..m {
        for g in 0..gpr {
            let off = row * bpr + g * 136;
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

struct AccStats {
    nonfinite_oracle: usize,
    nonfinite_cand: usize,
    ref_max: f32,
    abs_limit: f32,
    max_abs: f32,
    max_rel: f32,
    bitdiff: usize,
}

fn accuracy_stats(oracle: &[f32], cand: &[f32]) -> AccStats {
    assert_eq!(oracle.len(), cand.len());
    let mut nonfinite_oracle = 0usize;
    let mut nonfinite_cand = 0usize;
    let mut ref_max = 0.0f32;
    for &r in oracle {
        if !r.is_finite() {
            nonfinite_oracle += 1;
        } else {
            ref_max = ref_max.max(r.abs());
        }
    }
    for &c in cand {
        if !c.is_finite() {
            nonfinite_cand += 1;
        }
    }
    let abs_limit = (ABS_FRAC * ref_max).max(ABS_FLOOR);

    let mut max_abs = 0.0f32;
    let mut max_rel = 0.0f32;
    let mut bitdiff = 0usize;
    for (&r, &c) in oracle.iter().zip(cand.iter()) {
        if r.to_bits() != c.to_bits() {
            bitdiff += 1;
        }
        if !r.is_finite() || !c.is_finite() {
            continue;
        }
        let d = (r - c).abs();
        if d > max_abs {
            max_abs = d;
        }
        if r.abs() > abs_limit {
            let rel = d / r.abs();
            if rel > max_rel {
                max_rel = rel;
            }
        }
    }
    AccStats {
        nonfinite_oracle,
        nonfinite_cand,
        ref_max,
        abs_limit,
        max_abs,
        max_rel,
        bitdiff,
    }
}

fn row_pass(s: &AccStats) -> bool {
    s.nonfinite_oracle == 0
        && s.nonfinite_cand == 0
        && s.max_abs <= s.abs_limit
        && s.max_rel <= MAX_REL_LIMIT
}

fn print_row(shape: &str, s: &AccStats, pass: bool) {
    // Machine-readable single line per shape.
    println!(
        "shape={} verdict={} nonfinite_oracle={} nonfinite_cand={} ref_max={:.6e} abs_limit={:.6e} max_abs={:.6e} max_rel={:.6e} bitdiff={}",
        shape,
        if pass { "PASS" } else { "FAIL" },
        s.nonfinite_oracle,
        s.nonfinite_cand,
        s.ref_max,
        s.abs_limit,
        s.max_abs,
        s.max_rel,
        s.bitdiff
    );
}

fn run_shape(
    gpu: &mut Gpu,
    shape: &str,
    w: &GpuTensor,
    x: &GpuTensor,
) -> Result<bool, String> {
    let y_elems = B * M;
    let y_oracle = gpu
        .alloc_tensor(&[B, M], DType::F32)
        .map_err(|e| format!("{shape} alloc y_oracle: {e}"))?;
    let y_cand = gpu
        .alloc_tensor(&[B, M], DType::F32)
        .map_err(|e| format!("{shape} alloc y_cand: {e}"))?;

    // Oracle: F32 RM2/BV6 from zeroed Y.
    gpu.hip
        .memset(&y_oracle.buf, 0, y_elems * 4)
        .map_err(|e| format!("{shape} zero oracle: {e}"))?;
    let oracle_used = gpu
        .gemm_hfq4g256_residual_muse_gfx1100_rm(w, x, &y_oracle, M, K, B, 2)
        .map_err(|e| format!("{shape} oracle launch: {e}"))?;
    if !oracle_used {
        let _ = gpu.free_tensor(y_oracle);
        let _ = gpu.free_tensor(y_cand);
        return Err(format!(
            "{shape}: oracle gemm_hfq4g256_residual_muse_gfx1100_rm(rm=2) returned Ok(false)"
        ));
    }
    gpu.hip
        .device_synchronize()
        .map_err(|e| format!("{shape} oracle sync: {e}"))?;

    // Candidate: FP16 OPSEL-pair from separately zeroed Y.
    gpu.hip
        .memset(&y_cand.buf, 0, y_elems * 4)
        .map_err(|e| format!("{shape} zero cand: {e}"))?;
    let cand_used = gpu
        .gemm_hfq4g256_residual_muse_gfx1100_rm2_f16acc_pair(w, x, &y_cand, M, K, B)
        .map_err(|e| format!("{shape} candidate launch: {e}"))?;
    if !cand_used {
        let _ = gpu.free_tensor(y_oracle);
        let _ = gpu.free_tensor(y_cand);
        return Err(format!(
            "{shape}: candidate gemm_hfq4g256_residual_muse_gfx1100_rm2_f16acc_pair returned Ok(false)"
        ));
    }
    gpu.hip
        .device_synchronize()
        .map_err(|e| format!("{shape} cand sync: {e}"))?;

    let oracle_host = gpu
        .download_f32(&y_oracle)
        .map_err(|e| format!("{shape} dl oracle: {e}"))?;
    let cand_host = gpu
        .download_f32(&y_cand)
        .map_err(|e| format!("{shape} dl cand: {e}"))?;

    let _ = gpu.free_tensor(y_oracle);
    let _ = gpu.free_tensor(y_cand);

    if oracle_host.len() != y_elems || cand_host.len() != y_elems {
        return Err(format!(
            "{shape}: output length mismatch oracle={} cand={} expected={}",
            oracle_host.len(),
            cand_host.len(),
            y_elems
        ));
    }

    let stats = accuracy_stats(&oracle_host, &cand_host);
    let pass = row_pass(&stats);
    print_row(shape, &stats, pass);
    Ok(pass)
}

fn main() {
    let mut gpu = Gpu::init().expect("GPU init");
    let arch = gpu.arch_caps.arch().to_string();
    println!(
        "probe_glimmer_f16acc_pair arch={} is_gfx1100={} M={} K={} B={}",
        arch,
        gpu.arch_caps.is_gfx1100(),
        M,
        K,
        B
    );
    println!(
        "thresholds nonfinite==0 max_abs<=max({ABS_FRAC}*ref_max,{ABS_FLOOR}) max_rel<={MAX_REL_LIMIT} (significant |oracle|>abs_limit)"
    );

    // Deterministic weights: gate 0xB1, up 0xB2 (bench_glimmer_prefill_shapes).
    let gate_w = gpu
        .upload_raw(&build_hfq4g256(M, K, 0xB1), &[M, K])
        .expect("upload gate w");
    let up_w = gpu
        .upload_raw(&build_hfq4g256(M, K, 0xB2), &[M, K])
        .expect("upload up w");
    let x_host: Vec<f32> = (0..B * K)
        .map(|i| ((i % 97) as f32 - 48.0) * 0.01)
        .collect();
    let x = gpu.upload_f32(&x_host, &[B, K]).expect("upload x");

    let mut all_pass = true;
    match run_shape(&mut gpu, "gate_proj", &gate_w, &x) {
        Ok(p) => all_pass &= p,
        Err(e) => {
            eprintln!("ERROR {e}");
            all_pass = false;
        }
    }
    match run_shape(&mut gpu, "up_proj", &up_w, &x) {
        Ok(p) => all_pass &= p,
        Err(e) => {
            eprintln!("ERROR {e}");
            all_pass = false;
        }
    }

    let _ = gpu.free_tensor(x);
    let _ = gpu.free_tensor(gate_w);
    let _ = gpu.free_tensor(up_w);

    if !all_pass {
        println!("verdict=FAIL");
        std::process::exit(1);
    }
    println!("verdict=PASS");
}
