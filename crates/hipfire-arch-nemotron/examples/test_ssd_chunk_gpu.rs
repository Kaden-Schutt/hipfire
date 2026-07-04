// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! gpu-vs-cpu correctness for the Mamba-2 SSD **chunked** prefill kernel
//! (`mamba2_ssd_chunk_f32`), the parallel intra-chunk decomposition and the
//! correctness floor for the bf16-WMMA prefill kernel. The GPU chunk kernel
//! mirrors the CPU oracle `ssd::ssd_chunked` op-for-op, so it is compared both
//! against `ssd_chunked` (tight — only libdevice expf/log1pf differences) and
//! against the canonical sequential scan `ssd_sequence` (the ~1e-4 reassociation
//! bar that the whole chunked decomposition must hold). Several GPU chunk tiles
//! are exercised (single-chunk and multi-chunk) so a stride/group/carry error
//! surfaces against a readable oracle instead of as a prefill attractor.
//!
//! Run gpu-lock-coordinated:
//!   hipfire lock acquire
//!   cargo run -p hipfire-arch-nemotron --example test_ssd_chunk_gpu

use hipfire_arch_nemotron::ssd::{ssd_chunked, ssd_sequence, SsdParams};
use hipfire_rdna::{DType, Gpu};

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

fn main() {
    let mut gpu = Gpu::init().expect("GPU init failed");
    eprintln!("GPU: {}", gpu.arch);

    // Structurally faithful small dims (non-pow2 head_dim) + nemotron-shaped
    // group interleave (num_heads/n_groups = 3 heads per group).
    let (h, dh, n, g, seq) = (6usize, 5usize, 8usize, 2usize, 37usize);
    let xd = h * dh;
    let bd = g * n;

    let mut seed = 0x51ED_270Bu32;
    let mut rng = || {
        seed ^= seed << 13;
        seed ^= seed >> 17;
        seed ^= seed << 5;
        (seed as f32 / u32::MAX as f32) * 0.4 - 0.2
    };

    let p = SsdParams {
        num_heads: h,
        head_dim: dh,
        state_size: n,
        n_groups: g,
        dt_min: 0.0,
        dt_max: f32::INFINITY,
        a_log: (0..h).map(|_| rng() * 0.5).collect(),
        d: (0..h).map(|_| rng()).collect(),
        dt_bias: (0..h).map(|_| rng()).collect(),
    };
    let x: Vec<f32> = (0..seq * xd).map(|_| rng()).collect();
    let b: Vec<f32> = (0..seq * bd).map(|_| rng()).collect();
    let c: Vec<f32> = (0..seq * bd).map(|_| rng()).collect();
    let dt: Vec<f32> = (0..seq * h).map(|_| rng()).collect();

    // Canonical sequential reference (from zero state).
    let mut state_seq = vec![0.0f32; h * dh * n];
    let mut y_seq = vec![0.0f32; seq * xd];
    ssd_sequence(&p, &mut state_seq, &x, &b, &c, &dt, &mut y_seq);

    let alog_g = gpu.upload_f32(&p.a_log, &[h]).unwrap();
    let d_g = gpu.upload_f32(&p.d, &[h]).unwrap();
    let dtb_g = gpu.upload_f32(&p.dt_bias, &[h]).unwrap();
    let x_g = gpu.upload_f32(&x, &[seq * xd]).unwrap();
    let b_g = gpu.upload_f32(&b, &[seq * bd]).unwrap();
    let c_g = gpu.upload_f32(&c, &[seq * bd]).unwrap();
    let dt_g = gpu.upload_f32(&dt, &[seq * h]).unwrap();

    let mut ok = true;
    // single-chunk (chunk >= seq), and several multi-chunk tilings.
    for &chunk in &[64usize, 16, 8, 7] {
        // CPU chunked oracle (same op order as the GPU kernel) from zero state.
        let mut state_chunk = vec![0.0f32; h * dh * n];
        let mut y_chunk = vec![0.0f32; seq * xd];
        ssd_chunked(&p, &mut state_chunk, &x, &b, &c, &dt, &mut y_chunk, chunk);

        // GPU chunked kernel from zero state.
        let y_g = gpu.zeros(&[seq * xd], DType::F32).unwrap();
        let state_g = gpu.zeros(&[h * dh * n], DType::F32).unwrap();
        gpu.mamba2_ssd_chunk_f32(
            &y_g, &state_g, &x_g, &b_g, &c_g, &dt_g, &alog_g, &d_g, &dtb_g, seq, h, dh, n, g,
            p.dt_min, p.dt_max, chunk,
        )
        .unwrap();
        gpu.hip.device_synchronize().unwrap();

        let y_gpu = gpu.download_f32(&y_g).unwrap();
        let state_gpu = gpu.download_f32(&state_g).unwrap();

        // tight vs the CPU chunked oracle (identical reassociation)
        let dy_c = max_abs_diff(&y_chunk, &y_gpu);
        let ds_c = max_abs_diff(&state_chunk, &state_gpu);
        // ~1e-4 vs the canonical sequential scan
        let dy_s = max_abs_diff(&y_seq, &y_gpu);
        let ds_s = max_abs_diff(&state_seq, &state_gpu);

        eprintln!(
            "chunk={chunk:>2}: vs ssd_chunked Δy={dy_c:.2e} Δs={ds_c:.2e} | vs ssd_sequence Δy={dy_s:.2e} Δs={ds_s:.2e}"
        );
        if dy_c > 1e-4 || ds_c > 1e-4 || dy_s > 1e-4 || ds_s > 1e-4 {
            ok = false;
        }
    }

    if ok {
        println!("PASS: mamba2_ssd_chunk_f32 matches ssd_chunked and ssd_sequence");
    } else {
        eprintln!("FAIL: mamba2_ssd_chunk_f32 diverges");
        std::process::exit(1);
    }
}
