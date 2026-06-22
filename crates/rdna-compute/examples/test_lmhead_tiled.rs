// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
//
// Validate using gemm_f32_register_tiled (register-tiled, weight-bandwidth-
// amortized) for the lm_head fwd + bwd vs the naive linear kernels. The lm_head
// GEMM over vocab=248K is the training bottleneck; the tiled kernel reads each
// 5GB weight row once per 8-batch-tile instead of once per batch row.
//   fwd: logits[B,V] = out[B,K] . lmhead[V,K]^T  ==  tiled(lmhead, out)
//   bwd: d_out[B,K]  = dlogits[B,V] . lmhead[V,K] ==  tiled(lmhead^T, dlogits)

use rdna_compute::{DType, Gpu};

fn frand(s: usize) -> f32 { ((s.wrapping_mul(2654435761) % 2000) as f32 / 1000.0) - 1.0 }

fn main() {
    let mut gpu = Gpu::init().unwrap();
    let (b, k, v) = (16usize, 256usize, 4096usize); // B, d_tgt, vocab (small proxy)
    let out: Vec<f32> = (0..b * k).map(|i| frand(i + 1)).collect();
    let lmhead: Vec<f32> = (0..v * k).map(|i| frand(i + 9000) * 0.1).collect();
    let dlogits: Vec<f32> = (0..b * v).map(|i| frand(i + 5)).collect();

    let outg = gpu.upload_f32(&out, &[b, k]).unwrap();
    let lmg = gpu.upload_f32(&lmhead, &[v, k]).unwrap();
    let dlg = gpu.upload_f32(&dlogits, &[b, v]).unwrap();

    // ---- fwd ----
    let log_naive = gpu.zeros(&[b, v], DType::F32).unwrap();
    gpu.linear_fwd_f32(&outg, &lmg, &log_naive, b, k, v).unwrap();
    let log_tiled = gpu.zeros(&[b, v], DType::F32).unwrap();
    gpu.gemm_f32_register_tiled(&lmg, &outg, &log_tiled, v, k, b).unwrap();
    let (a, c) = (gpu.download_f32(&log_naive).unwrap(), gpu.download_f32(&log_tiled).unwrap());
    let fwd_err = a.iter().zip(&c).map(|(x, y)| (x - y).abs()).fold(0f32, f32::max);

    // ---- bwd ----
    let lmt = gpu.zeros(&[k, v], DType::F32).unwrap();
    gpu.transpose_f32(&lmg, &lmt, v, k).unwrap(); // [v,k] -> [k,v]
    let dout_naive = gpu.zeros(&[b, k], DType::F32).unwrap();
    gpu.linear_bwd_dx_f32(&dlg, &lmg, &dout_naive, b, k, v).unwrap();
    let dout_tiled = gpu.zeros(&[b, k], DType::F32).unwrap();
    gpu.gemm_f32_register_tiled(&lmt, &dlg, &dout_tiled, k, v, b).unwrap();
    let (a2, c2) = (gpu.download_f32(&dout_naive).unwrap(), gpu.download_f32(&dout_tiled).unwrap());
    let bwd_err = a2.iter().zip(&c2).map(|(x, y)| (x - y).abs()).fold(0f32, f32::max);

    let mag = a.iter().chain(&a2).map(|x| x.abs()).fold(0f32, f32::max);
    println!("lm_head fwd max|diff| = {fwd_err:.3e}, bwd max|diff| = {bwd_err:.3e} (mag {mag:.2})");
    if fwd_err < 1e-2 && bwd_err < 1e-2 {
        println!("lmhead_tiled: PASS (tiled == naive)");
    } else {
        println!("lmhead_tiled: FAIL");
        std::process::exit(1);
    }
}
