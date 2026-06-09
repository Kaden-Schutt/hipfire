// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
//
// Validate the bf16-MFMA lm_head fwd/bwd GEMMs against the fp32 tiled kernel.
// fwd: logits[b,V] = sum_d lmhead[V,d]*out[b,d]
// bwd: d_out[b,d]  = sum_V lmhead_t[d,V]*dlogits[b,V]
// Layout bug => rel-L2 O(1); correct + bf16 cast => ~1e-2. Also reports the
// argmax-agreement (matters for the per_pos eval diagnostic, not the loss).

use rdna_compute::{DType, Gpu, GpuTensor};

fn frand(s: usize) -> f32 { ((s.wrapping_mul(2654435761) % 4000) as f32 / 2000.0) - 1.0 }
fn up(gpu: &mut Gpu, v: &[f32], sh: &[usize]) -> GpuTensor { gpu.upload_f32(v, sh).unwrap() }

fn rel_l2(a: &[f32], b: &[f32]) -> f32 {
    let mut num = 0f64; let mut den = 0f64;
    for i in 0..a.len() { let d = (a[i] - b[i]) as f64; num += d * d; den += (a[i] as f64).powi(2); }
    (num / den.max(1e-30)).sqrt() as f32
}

fn main() {
    let mut gpu = Gpu::init().unwrap();
    let (b, d_tgt, vocab) = (16usize, 512usize, 4096usize); // lm_head-shaped (scaled vocab)
    let lmhead: Vec<f32> = (0..vocab * d_tgt).map(|i| frand(i + 1) * 0.05).collect();
    let out: Vec<f32> = (0..b * d_tgt).map(|i| frand(i + 7) * 0.5).collect();
    let dlogits: Vec<f32> = (0..b * vocab).map(|i| frand(i + 13) * 0.1).collect();

    let lmhead_g = up(&mut gpu, &lmhead, &[vocab, d_tgt]);
    let out_g = up(&mut gpu, &out, &[b, d_tgt]);
    let dlog_g = up(&mut gpu, &dlogits, &[b, vocab]);
    // transpose for bwd
    let lmhead_t = gpu.zeros(&[d_tgt, vocab], DType::F32).unwrap();
    gpu.transpose_f32(&lmhead_g, &lmhead_t, vocab, d_tgt).unwrap();
    // bf16 copies
    let lmhead_bf16 = gpu.zeros(&[vocab, d_tgt], DType::F16).unwrap();
    gpu.to_bf16_f32(&lmhead_g, &lmhead_bf16, vocab * d_tgt).unwrap();
    let lmhead_t_bf16 = gpu.zeros(&[d_tgt, vocab], DType::F16).unwrap();
    gpu.to_bf16_f32(&lmhead_t, &lmhead_t_bf16, d_tgt * vocab).unwrap();
    let out_bf16 = gpu.zeros(&[b, d_tgt], DType::F16).unwrap();
    gpu.to_bf16_f32(&out_g, &out_bf16, b * d_tgt).unwrap();
    let dlog_bf16 = gpu.zeros(&[b, vocab], DType::F16).unwrap();
    gpu.to_bf16_f32(&dlog_g, &dlog_bf16, b * vocab).unwrap();

    // fwd
    let lg_f32 = gpu.zeros(&[b, vocab], DType::F32).unwrap();
    gpu.gemm_f32_register_tiled(&lmhead_g, &out_g, &lg_f32, vocab, d_tgt, b).unwrap();
    let lg_bf16 = gpu.zeros(&[b, vocab], DType::F32).unwrap();
    gpu.gemm_bf16_mfma(&lmhead_bf16, &out_bf16, &lg_bf16, vocab, d_tgt, b).unwrap();
    let lf = gpu.download_f32(&lg_f32).unwrap();
    let lb = gpu.download_f32(&lg_bf16).unwrap();
    let fwd_l2 = rel_l2(&lf, &lb);
    // argmax agreement per batch row
    let mut am_ok = 0;
    for r in 0..b {
        let am = |v: &[f32]| v[r * vocab..(r + 1) * vocab].iter().enumerate().max_by(|x, y| x.1.partial_cmp(y.1).unwrap()).unwrap().0;
        if am(&lf) == am(&lb) { am_ok += 1; }
    }

    // bwd
    let do_f32 = gpu.zeros(&[b, d_tgt], DType::F32).unwrap();
    gpu.gemm_f32_register_tiled(&lmhead_t, &dlog_g, &do_f32, d_tgt, vocab, b).unwrap();
    let do_bf16 = gpu.zeros(&[b, d_tgt], DType::F32).unwrap();
    gpu.gemm_bf16_mfma(&lmhead_t_bf16, &dlog_bf16, &do_bf16, d_tgt, vocab, b).unwrap();
    let df = gpu.download_f32(&do_f32).unwrap();
    let db = gpu.download_f32(&do_bf16).unwrap();
    let bwd_l2 = rel_l2(&df, &db);

    println!("  fwd logits  rel_L2 = {fwd_l2:.3e}   argmax agree = {am_ok}/{b}");
    println!("  bwd d_out   rel_L2 = {bwd_l2:.3e}");
    let ok = fwd_l2 < 5e-2 && bwd_l2 < 5e-2;
    if ok { println!("lmhead_bf16: PASS (layout correct, bf16-precision agreement)"); }
    else { println!("lmhead_bf16: FAIL"); std::process::exit(1); }
}
