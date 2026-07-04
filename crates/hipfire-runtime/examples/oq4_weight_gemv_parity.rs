#![allow(
    clippy::duplicated_attributes,
    clippy::doc_lazy_continuation,
    clippy::doc_overindented_list_items,
    clippy::explicit_counter_loop,
    clippy::field_reassign_with_default,
    clippy::manual_checked_ops,
    clippy::manual_clamp,
    clippy::manual_div_ceil,
    clippy::needless_range_loop,
    clippy::ptr_arg,
    clippy::same_item_push,
    clippy::too_many_arguments,
    clippy::type_complexity,
    clippy::unnecessary_cast,
    clippy::useless_vec,
    clippy::while_let_loop
)]
// hipfire example clippy sweep: examples are GPU probes/benches, not reusable APIs.

// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 hipfire contributors
// hipfire — see LICENSE and NOTICE in the project root.
//
//! Numerical validation of the Opus Quant W4A4 *forward dispatch* — the
//! `weight_gemv` Oq4G256 arm — end-to-end against an unquantized f32 reference.
//!
//! Builds a synthetic OQ4G256 weight in the SAME on-disk→kernel layout the qt=32
//! loader produces (packed nibbles [M,K/2] then per-group f32 scales [M,K/256],
//! one buffer), constructs a `WeightTensor{gpu_dtype: Oq4G256}`, and runs
//! `weight_gemv`. The arm does: rotate_x_mq_for (FWHT-256, seeds 42/1042, the
//! shared MQ pairing) → quantize_act_oq4 → gemm_oq4_grouped_wmma with the weight
//! scales addressed via `sub_offset`. The weight is FWHT-rotated offline here with
//! the identical `cpu_fwht_256` convention, so <fwht(w),fwht(x)> = <w,x> and the
//! GPU output reconstructs W·x up to int4 quant noise (~18-22 dB).
//!
//!   cargo run --release -p hipfire-runtime --example oq4_weight_gemv_parity [M K]

use hipfire_primitives::fwht::{cpu_fwht_256, gen_fwht_signs};
use hipfire_runtime::weights::{weight_gemv, WeightTensor};
use hipfire_rdna::{DType, Gpu};

fn lcg(seed: u32, n: usize) -> Vec<f32> {
    let mut s = seed.max(1);
    (0..n)
        .map(|i| {
            s = s.wrapping_mul(1_103_515_245).wrapping_add(12345) & 0x7fff_ffff;
            let v = (s as f32 / 2_147_483_648.0) - 0.5;
            if i % 89 == 0 {
                v * 8.0
            } else {
                v
            } // sparse outliers
        })
        .collect()
}

fn main() {
    let mut a = std::env::args().skip(1);
    let m: usize = a.next().and_then(|s| s.parse().ok()).unwrap_or(512);
    let k: usize = a.next().and_then(|s| s.parse().ok()).unwrap_or(2048);
    const GROUP: usize = 256;
    assert_eq!(k % GROUP, 0);
    let ng = k / GROUP;

    let mut gpu = Gpu::init().unwrap();
    if !gpu.arch_caps.has_wmma_w32() {
        println!(
            "SKIP oq4_weight_gemv_parity: {} lacks wave32 WMMA",
            gpu.arch
        );
        return;
    }

    let w = lcg(1, m * k);
    let x = lcg(2, k);

    // f32 reference y[m] = Σ_k W[m,k]·x[k].
    let mut yref = vec![0.0f32; m];
    for mi in 0..m {
        let mut acc = 0.0f32;
        for ki in 0..k {
            acc += w[mi * k + ki] * x[ki];
        }
        yref[mi] = acc;
    }

    // Quantize W to the loader's combined buffer: FWHT-256 each group → symmetric
    // int4 (absmax/7) → [packed nibbles M*(K/2)] ++ [f32 scales M*ng].
    let s1 = gen_fwht_signs(42, 256);
    let s2 = gen_fwht_signs(1042, 256);
    let packed_bytes = m * (k / 2);
    let mut combined = vec![0u8; packed_bytes + m * ng * 4];
    for r in 0..m {
        for g in 0..ng {
            let mut grp = [0.0f32; 256];
            grp.copy_from_slice(&w[r * k + g * GROUP..r * k + g * GROUP + GROUP]);
            cpu_fwht_256(&mut grp, &s1, &s2);
            let amax = grp.iter().fold(1e-12f32, |a, &v| a.max(v.abs()));
            // Clip-search the symmetric scale (MSE-optimal), as the production
            // codec does — confirms the forward path reaches the validated band.
            let mut best = (1.0f32, f32::INFINITY);
            for &cl in &[1.0f32, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6] {
                let sc = cl * amax / 7.0;
                let e: f32 = grp
                    .iter()
                    .map(|&v| {
                        let q = (v / sc).round().clamp(-7.0, 7.0);
                        (v - q * sc).powi(2)
                    })
                    .sum();
                if e < best.1 {
                    best = (cl, e);
                }
            }
            let scale = best.0 * amax / 7.0;
            let inv = 1.0 / scale;
            let dst = r * (k / 2) + g * (GROUP / 2);
            for j in 0..128 {
                let q = |v: f32| (v * inv).round().clamp(-7.0, 7.0) as i8;
                let lo = (q(grp[2 * j]) as u8) & 0xf;
                let hi = (q(grp[2 * j + 1]) as u8) & 0xf;
                combined[dst + j] = lo | (hi << 4);
            }
            let so = packed_bytes + (r * ng + g) * 4;
            combined[so..so + 4].copy_from_slice(&scale.to_le_bytes());
        }
    }

    let buf = gpu.upload_raw(&combined, &[combined.len()]).unwrap();
    let wt = WeightTensor {
        buf,
        gpu_dtype: DType::Oq4G256,
        m,
        k,
        row_stride: 0,
        paro: None,
        awq_scale: None,
    };

    let xd = gpu
        .upload_raw(
            &x.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>(),
            &[k],
        )
        .unwrap();
    let yd = gpu.upload_raw(&vec![0u8; m * 4], &[m]).unwrap();
    weight_gemv(&mut gpu, &wt, &xd, &yd).unwrap();
    gpu.device_synchronize().unwrap();
    let y = gpu.download_f32(&yd).unwrap();

    let mut sig = 0.0f64;
    let mut noise = 0.0f64;
    for mi in 0..m {
        sig += (yref[mi] as f64).powi(2);
        noise += ((yref[mi] - y[mi]) as f64).powi(2);
    }
    let db = 10.0 * (sig / noise.max(1e-30)).log10();
    // ~15-17 dB is the rotation-only W4A4 floor (FWHT + int4 both sides, NO
    // SmoothQuant — awq_scale is None here). In production the awq_scale sidecar
    // migrates activation outliers into the weight offline and rotate_x_mq_for
    // divides x by it, lifting this to the ~20 dB capstone band. The point of this
    // test is that the forward DISPATCH is numerically faithful (a layout/scale
    // bug would read as <3 dB garbage), not to re-measure the recipe.
    let pass = db > 13.0;
    println!(
        "oq4_weight_gemv_parity M={m} K={k} on {}: forward-dispatch SQNR={db:.2} dB -> {}",
        gpu.arch,
        if pass { "PASS" } else { "FAIL" }
    );
    if !pass {
        std::process::exit(1);
    }
}
