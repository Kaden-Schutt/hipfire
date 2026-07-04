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
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
#![allow(
    clippy::doc_lazy_continuation,
    clippy::doc_overindented_list_items,
    clippy::drop_non_drop,
    clippy::excessive_precision,
    clippy::identity_op,
    clippy::manual_div_ceil,
    clippy::manual_is_multiple_of,
    clippy::needless_range_loop,
    clippy::print_literal,
    clippy::redundant_closure,
    clippy::too_many_arguments,
    clippy::type_complexity,
    clippy::unusual_byte_groupings,
    clippy::useless_vec,
    clippy::unnecessary_cast
)]

//! gpu-vs-cpu correctness for `mamba2_gated_norm_f32` — nemotron_h Mamba-2
//! `MambaRMSNormGated` (gate-then-group-RMSNorm, norm_before_gate=False).
//! CPU reference mirrors `kernels/src/mamba2_gated_norm.hip`:
//!   gated = y * silu(z); out[g] = gated[g] * rsqrt(mean(gated[g]^2)+eps) * w[g]
//! per group of `group_size` elements. Uses Nano-4B-ish dims:
//! d_inner = 7680, n_groups = 8, group_size = 960.

use hipfire_rdna::{DType, Gpu};

fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

fn main() {
    let mut gpu = Gpu::init().expect("GPU init failed");
    eprintln!("GPU: {}", gpu.arch);

    let n_groups = 8usize;
    let group_size = 960usize;
    let n = n_groups * group_size; // 7680
    let eps = 1e-5f32;

    let mut seed = 0x1234_5678u32;
    let mut rng = || {
        seed ^= seed << 13;
        seed ^= seed >> 17;
        seed ^= seed << 5;
        (seed as f32 / u32::MAX as f32) * 4.0 - 2.0 // [-2,2]
    };
    let y: Vec<f32> = (0..n).map(|_| rng()).collect();
    let z: Vec<f32> = (0..n).map(|_| rng()).collect();
    let w: Vec<f32> = (0..n).map(|_| 0.5 + rng().abs()).collect();

    // CPU reference: gate-then-group-RMSNorm.
    let mut cpu = vec![0.0f32; n];
    for g in 0..n_groups {
        let base = g * group_size;
        let mut ss = 0.0f32;
        for i in 0..group_size {
            let gated = y[base + i] * silu(z[base + i]);
            ss += gated * gated;
        }
        let inv = 1.0f32 / (ss / group_size as f32 + eps).sqrt();
        for i in 0..group_size {
            let gated = y[base + i] * silu(z[base + i]);
            cpu[base + i] = gated * inv * w[base + i];
        }
    }

    let d_y = gpu.upload_f32(&y, &[n]).unwrap();
    let d_z = gpu.upload_f32(&z, &[n]).unwrap();
    let d_w = gpu.upload_f32(&w, &[n]).unwrap();
    let d_out = gpu.zeros(&[n], DType::F32).unwrap();
    gpu.mamba2_gated_norm_f32(&d_out, &d_y, &d_z, &d_w, n_groups, group_size, eps)
        .unwrap();
    gpu.hip.device_synchronize().unwrap();
    let out = gpu.download_f32(&d_out).unwrap();

    let max_diff = out
        .iter()
        .zip(&cpu)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    eprintln!("max|Δ|={max_diff:.3e}");
    if max_diff > 1e-4 {
        eprintln!("FAIL");
        std::process::exit(1);
    }
    println!("PASS: mamba2_gated_norm_f32 matches CPU reference");
}
