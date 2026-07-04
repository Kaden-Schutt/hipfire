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

//! Correctness for the Mamba-2 SSD chunked **bf16-WMMA** prefill kernel
//! (`mamba2_ssd_chunk_wmma`): chunked over the sequence with state carried
//! across chunks. bf16 inputs carry ~1% relative error, so this is graded by
//! **cosine similarity / relative error** against `ssd_sequence` — NOT the 1e-4
//! f32 bar. Configs exercise the single-tile path, the multi-K-tile /
//! multi-J-tile path (N=128, P=80), and multi-chunk sequences (state carry).
//!
//! Run gpu-lock-coordinated:
//!   hipfire lock acquire
//!   cargo run -p hipfire-arch-nemotron --example test_ssd_chunk_wmma_gpu

use hipfire_arch_nemotron::ssd::{ssd_sequence, SsdParams};
use hipfire_rdna::{DType, Gpu};

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let mut dot = 0.0f64;
    let mut na = 0.0f64;
    let mut nb = 0.0f64;
    for (x, y) in a.iter().zip(b) {
        dot += (*x as f64) * (*y as f64);
        na += (*x as f64) * (*x as f64);
        nb += (*y as f64) * (*y as f64);
    }
    if na == 0.0 || nb == 0.0 {
        return 1.0;
    }
    (dot / (na.sqrt() * nb.sqrt())) as f32
}

fn rel_l2(a: &[f32], b: &[f32]) -> f32 {
    let mut num = 0.0f64;
    let mut den = 0.0f64;
    for (x, y) in a.iter().zip(b) {
        let d = (*x - *y) as f64;
        num += d * d;
        den += (*x as f64) * (*x as f64);
    }
    if den == 0.0 {
        return 0.0;
    }
    (num.sqrt() / den.sqrt()) as f32
}

fn run_case(
    gpu: &mut Gpu,
    h: usize,
    dh: usize,
    n: usize,
    g: usize,
    seq: usize,
    chunk: usize,
) -> bool {
    let xd = h * dh;
    let bd = g * n;

    let mut seed = 0x1357_9BDFu32 ^ ((h * 131 + dh * 17 + n + seq * 7 + chunk) as u32);
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

    // f32 reference (sequential = single-chunk floor from zero state).
    let mut state_ref = vec![0.0f32; h * dh * n];
    let mut y_ref = vec![0.0f32; seq * xd];
    ssd_sequence(&p, &mut state_ref, &x, &b, &c, &dt, &mut y_ref);

    let alog_g = gpu.upload_f32(&p.a_log, &[h]).unwrap();
    let d_g = gpu.upload_f32(&p.d, &[h]).unwrap();
    let dtb_g = gpu.upload_f32(&p.dt_bias, &[h]).unwrap();
    let x_g = gpu.upload_f32(&x, &[seq * xd]).unwrap();
    let b_g = gpu.upload_f32(&b, &[seq * bd]).unwrap();
    let c_g = gpu.upload_f32(&c, &[seq * bd]).unwrap();
    let dt_g = gpu.upload_f32(&dt, &[seq * h]).unwrap();
    let y_g = gpu.zeros(&[seq * xd], DType::F32).unwrap();
    let state_g = gpu.zeros(&[h * dh * n], DType::F32).unwrap();

    gpu.mamba2_ssd_chunk_wmma(
        &y_g, &state_g, &x_g, &b_g, &c_g, &dt_g, &alog_g, &d_g, &dtb_g, seq, h, dh, n, g, p.dt_min,
        p.dt_max, chunk,
    )
    .unwrap();
    gpu.hip.device_synchronize().unwrap();

    let y_gpu = gpu.download_f32(&y_g).unwrap();
    let state_gpu = gpu.download_f32(&state_g).unwrap();

    let yc = cosine(&y_ref, &y_gpu);
    let yr = rel_l2(&y_ref, &y_gpu);
    let sc = cosine(&state_ref, &state_gpu);
    let sr = rel_l2(&state_ref, &state_gpu);

    eprintln!(
        "  h={h} dh={dh} n={n} g={g} seq={seq} chunk={chunk}: y cos={yc:.5} relL2={yr:.2e} | state cos={sc:.5} relL2={sr:.2e}"
    );
    yc > 0.999 && sc > 0.999 && yr < 3e-2 && sr < 3e-2
}

fn main() {
    let mut gpu = Gpu::init().expect("GPU init failed");
    eprintln!("GPU: {}", gpu.arch);

    let mut ok = true;
    // single chunk: tiny (all dims < 16) and nano-shaped (N=128, P=80).
    ok &= run_case(&mut gpu, 6, 5, 8, 2, 12, 16);
    ok &= run_case(&mut gpu, 4, 80, 128, 2, 12, 16);
    ok &= run_case(&mut gpu, 4, 80, 128, 2, 16, 16);
    // multi-chunk (state carried across chunks): tiny and nano-shaped.
    ok &= run_case(&mut gpu, 6, 5, 8, 2, 37, 16);
    ok &= run_case(&mut gpu, 4, 80, 128, 2, 37, 16);
    ok &= run_case(&mut gpu, 4, 80, 128, 2, 40, 8);

    if ok {
        println!("PASS: mamba2_ssd_chunk_wmma matches ssd_sequence within bf16 tol (single + multi-chunk)");
    } else {
        eprintln!("FAIL: mamba2_ssd_chunk_wmma diverges beyond bf16 tolerance");
        std::process::exit(1);
    }
}
