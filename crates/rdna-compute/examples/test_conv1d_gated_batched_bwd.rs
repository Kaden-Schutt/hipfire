// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
//
// Finite-difference gradient check for the batched LFM2 gated-conv backward
// (conv1d_gated_batched_bwd). Defines a scalar probe loss L(x) = sum(g_out*out),
// and checks analytic d_bcx / d_weight / d_state against central finite
// differences on a sample of elements. L is summed in f64 on the host so f32
// roundoff in the large reduction does not swamp the small finite-diff signal.

use rdna_compute::{DType, Gpu};

fn frand(seed: usize) -> f32 {
    ((seed.wrapping_mul(2654435761) % 2000) as f32 / 1000.0) - 1.0
}

fn loss(gpu: &mut Gpu, bcx: &[f32], state: &[f32], weight: &[f32], g_out: &[f32], bb: usize, c: usize, k: usize) -> f64 {
    let hist = k - 1;
    let b = gpu.upload_f32(bcx, &[bb, 3 * c]).unwrap();
    let s = gpu.upload_f32(state, &[c, hist]).unwrap();
    let w = gpu.upload_f32(weight, &[c, k]).unwrap();
    let o = gpu.zeros(&[bb, c], DType::F32).unwrap();
    gpu.conv1d_gated_batched_f32(&b, &s, &w, &o, bb, c, k).unwrap();
    let ov = gpu.download_f32(&o).unwrap();
    ov.iter().zip(g_out).map(|(a, b)| (*a as f64) * (*b as f64)).sum()
}

// central finite-diff over a sample of indices of `base`; compares to `analytic`.
fn check(
    gpu: &mut Gpu, name: &str, which: u8, analytic: &[f32], eps: f32,
    bcx: &[f32], state: &[f32], weight: &[f32], g_out: &[f32], bb: usize, c: usize, k: usize,
) -> f32 {
    let mut base = match which { 0 => bcx.to_vec(), 1 => state.to_vec(), _ => weight.to_vec() };
    let n = base.len();
    let step = (n / 24).max(1);
    let mut maxrel = 0f32;
    let mut i = 0;
    while i < n {
        let orig = base[i];
        base[i] = orig + eps;
        let lp = match which {
            0 => loss(gpu, &base, state, weight, g_out, bb, c, k),
            1 => loss(gpu, bcx, &base, weight, g_out, bb, c, k),
            _ => loss(gpu, bcx, state, &base, g_out, bb, c, k),
        };
        base[i] = orig - eps;
        let lm = match which {
            0 => loss(gpu, &base, state, weight, g_out, bb, c, k),
            1 => loss(gpu, bcx, &base, weight, g_out, bb, c, k),
            _ => loss(gpu, bcx, state, &base, g_out, bb, c, k),
        };
        base[i] = orig;
        let fd = ((lp - lm) / (2.0 * eps as f64)) as f32;
        let an = analytic[i];
        let rel = (fd - an).abs() / (an.abs() + 1e-3);
        if rel > maxrel { maxrel = rel; }
        i += step;
    }
    println!("  {name}: max rel err = {maxrel:.3e}");
    maxrel
}

fn main() {
    let mut gpu = Gpu::init().expect("GPU init failed");
    let bb = 16usize;
    let c = 64usize;
    let k = 3usize;
    let hist = k - 1;
    let eps = 1e-3f32;

    let bcx: Vec<f32> = (0..bb * 3 * c).map(|i| frand(i + 1)).collect();
    let state: Vec<f32> = (0..c * hist).map(|i| frand(i + 7_000)).collect();
    let weight: Vec<f32> = (0..c * k).map(|i| frand(i + 19_000)).collect();
    let g_out: Vec<f32> = (0..bb * c).map(|i| frand(i + 31_000)).collect();

    // -------- analytic backward --------
    let bcx_g = gpu.upload_f32(&bcx, &[bb, 3 * c]).unwrap();
    let state_g = gpu.upload_f32(&state, &[c, hist]).unwrap();
    let w_g = gpu.upload_f32(&weight, &[c, k]).unwrap();
    let g_g = gpu.upload_f32(&g_out, &[bb, c]).unwrap();
    let d_bcx = gpu.zeros(&[bb, 3 * c], DType::F32).unwrap();
    let d_weight = gpu.zeros(&[c, k], DType::F32).unwrap();
    let d_state = gpu.zeros(&[c, hist], DType::F32).unwrap();
    let d_conv = gpu.zeros(&[bb, c], DType::F32).unwrap();
    gpu.conv1d_gated_batched_bwd(&bcx_g, &state_g, &w_g, &g_g, &d_bcx, &d_weight, &d_state, &d_conv, bb, c, k)
        .unwrap();
    let a_dbcx = gpu.download_f32(&d_bcx).unwrap();
    let a_dw = gpu.download_f32(&d_weight).unwrap();
    let a_ds = gpu.download_f32(&d_state).unwrap();

    // -------- finite-diff checks --------
    println!("conv1d_gated_batched backward gradient check (Bb={bb}, C={c}, K={k}, eps={eps:.0e}):");
    let r_bcx = check(&mut gpu, "d_bcx  (B|C|xin)", 0, &a_dbcx, eps, &bcx, &state, &weight, &g_out, bb, c, k);
    let r_st = check(&mut gpu, "d_state", 1, &a_ds, eps, &bcx, &state, &weight, &g_out, bb, c, k);
    let r_w = check(&mut gpu, "d_weight", 2, &a_dw, eps, &bcx, &state, &weight, &g_out, bb, c, k);

    let worst = r_bcx.max(r_st).max(r_w);
    if worst < 1e-2 {
        println!("conv1d_gated_batched_bwd: PASS (worst rel err {worst:.3e})");
    } else {
        println!("conv1d_gated_batched_bwd: FAIL (worst rel err {worst:.3e})");
        std::process::exit(1);
    }
}
