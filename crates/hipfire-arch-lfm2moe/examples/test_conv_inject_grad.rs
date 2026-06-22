// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
//
// Gradient-check the conv-gate-injection (W_c) path through the module's
// body_forward/backward: validates the new cgate_add + colsum_strided kernels +
// the d_w_c / d_ctx wiring against central finite differences (f64 host loss).

use hipfire_arch_lfm2moe::dflash_train as dt;
use rdna_compute::{DType, Gpu, GpuTensor};

fn frand(s: usize) -> f32 { ((s.wrapping_mul(2654435761) % 2000) as f32 / 1000.0) - 1.0 }
fn up(gpu: &mut Gpu, v: &[f32], sh: &[usize]) -> GpuTensor { gpu.upload_f32(v, sh).unwrap() }

struct HW { op: Vec<f32>, fn_: Vec<f32>, inp: Vec<f32>, cw: Vec<f32>, op2: Vec<f32>, wc: Vec<f32>, w1: Vec<f32>, w3: Vec<f32>, w2: Vec<f32> }

fn main() {
    // central-fd needs eps above the bf16 quantum; under MFMA the forward is
    // bf16, so eps=1e-3 perturbations quantize away. Layout validated by
    // test_lin_parity/test_dw_mfma/test_dx_mfma; loss-match covers training.
    if hipfire_arch_lfm2moe::dflash_train::dflash_use_mfma() {
        println!("conv_inject_grad: SKIP (bf16 forward under HIPFIRE_DFLASH_MFMA; fd invalid)");
        return;
    }
    let mut gpu = Gpu::init().unwrap();
    // single conv layer (the only one that carries W_c) + dims
    let (d, hd, inter) = (32usize, 8usize, 48usize);
    let conv_k = 3usize;
    let cfg = dt::Cfg { d, is_attn: vec![false], nh: 4, nkv: 2, hd, conv_k, inter, theta: 1.0e6, eps: 1e-5, d_tgt: 24, vocab: 48, n_tgt_layers: 5 };
    let b = 4usize; let n_ctx = 3usize;
    let ws = 1.0 / (d as f32).sqrt();
    let h = HW {
        op: (0..d).map(|i| 1.0 + 0.1 * frand(i)).collect(),
        fn_: (0..d).map(|i| 1.0 + 0.1 * frand(i + 7)).collect(),
        inp: (0..3 * d * d).map(|i| frand(i + 100) * ws).collect(),
        cw: (0..d * conv_k).map(|i| frand(i + 200) * 0.3).collect(),
        op2: (0..d * d).map(|i| frand(i + 300) * ws).collect(),
        wc: (0..d * d).map(|i| frand(i + 400) * 0.2).collect(),
        w1: (0..inter * d).map(|i| frand(i + 500) * ws).collect(),
        w3: (0..inter * d).map(|i| frand(i + 600) * ws).collect(),
        w2: (0..d * inter).map(|i| frand(i + 700) * (1.0 / (inter as f32).sqrt())).collect(),
    };
    let body_in: Vec<f32> = (0..b * d).map(|i| 0.5 * frand(i + 3)).collect();
    let ctx_v: Vec<f32> = (0..n_ctx * d).map(|i| 0.5 * frand(i + 8000)).collect();
    let aprobe: Vec<f32> = (0..b * d).map(|i| frand(i + 555)).collect();
    let block_pos: Vec<i32> = (0..b).map(|i| (n_ctx + i) as i32).collect();
    let mut full: Vec<i32> = (0..n_ctx).map(|i| i as i32).collect(); full.extend(block_pos.iter().copied());
    let bpos = dt::upos(&mut gpu, &block_pos); let fpos = dt::upos(&mut gpu, &full);
    let cs = gpu.zeros(&[d, conv_k - 1], DType::F32).unwrap();

    let build = |gpu: &mut Gpu, h: &HW| -> Vec<dt::LW> {
        vec![dt::LW {
            op_norm: up(gpu, &h.op, &[d]), ffn_norm: up(gpu, &h.fn_, &[d]),
            in_proj: Some(up(gpu, &h.inp, &[3 * d, d])), conv_w: Some(up(gpu, &h.cw, &[d, conv_k])), out_proj: Some(up(gpu, &h.op2, &[d, d])),
            wq: None, wk: None, wv: None, wo: None, q_norm: None, k_norm: None,
            w_c: Some(up(gpu, &h.wc, &[d, d])),
            w1: up(gpu, &h.w1, &[inter, d]), w3: up(gpu, &h.w3, &[inter, d]), w2: up(gpu, &h.w2, &[d, inter]),
        }]
    };

    let run_loss = |gpu: &mut Gpu, h: &HW, bin: &[f32], cv: &[f32]| -> f64 {
        let layers = build(gpu, h);
        let h0 = up(gpu, bin, &[b, d]); let ctx = up(gpu, cv, &[n_ctx, d]);
        let (out, _t) = dt::body_forward(gpu, &cfg, &layers, &h0, &ctx, &bpos, &fpos, &cs, b, n_ctx);
        let o = gpu.download_f32(&out).unwrap();
        aprobe.iter().zip(&o).map(|(x, y)| *x as f64 * *y as f64).sum()
    };

    // analytic grads
    let layers = build(&mut gpu, &h);
    let h0 = up(&mut gpu, &body_in, &[b, d]); let ctx = up(&mut gpu, &ctx_v, &[n_ctx, d]);
    let (_out, tape) = dt::body_forward(&mut gpu, &cfg, &layers, &h0, &ctx, &bpos, &fpos, &cs, b, n_ctx);
    let dh_out = up(&mut gpu, &aprobe, &[b, d]);
    let (_dbi, d_ctx, glayers) = dt::body_backward(&mut gpu, &cfg, &layers, &tape, &dh_out, &ctx, &bpos, &fpos, &cs, b, n_ctx);
    let d_wc = gpu.download_f32(glayers[0].w_c.as_ref().unwrap()).unwrap();
    let d_ctx_h = gpu.download_f32(&d_ctx).unwrap();

    let eps = 1e-3f64;
    let samp = |n: usize| -> Vec<usize> { (0..12.min(n)).map(|i| (i * 2654435761) % n).collect() };
    let mut worst = 0f64;
    // d_w_c
    for t in samp(d * d) {
        let mut hp = clone_hw(&h); hp.wc[t] += eps as f32;
        let mut hm = clone_hw(&h); hm.wc[t] -= eps as f32;
        let fd = (run_loss(&mut gpu, &hp, &body_in, &ctx_v) - run_loss(&mut gpu, &hm, &body_in, &ctx_v)) / (2.0 * eps);
        let den = fd.abs().max(d_wc[t].abs() as f64).max(1e-4);
        worst = worst.max((fd - d_wc[t] as f64).abs() / den);
    }
    println!("  d_w_c max rel err = {worst:.3e}");
    let mut worst_ctx = 0f64;
    for t in samp(n_ctx * d) {
        let mut p = ctx_v.clone(); p[t] += eps as f32;
        let mut m = ctx_v.clone(); m[t] -= eps as f32;
        let fd = (run_loss(&mut gpu, &h, &body_in, &p) - run_loss(&mut gpu, &h, &body_in, &m)) / (2.0 * eps);
        let den = fd.abs().max(d_ctx_h[t].abs() as f64).max(1e-4);
        worst_ctx = worst_ctx.max((fd - d_ctx_h[t] as f64).abs() / den);
    }
    println!("  d_ctx max rel err = {worst_ctx:.3e}");
    let w = worst.max(worst_ctx);
    if w < 2e-2 { println!("conv_inject_grad: PASS (w_c + d_ctx correct, {w:.3e})"); }
    else { println!("conv_inject_grad: FAIL ({w:.3e})"); std::process::exit(1); }
}

fn clone_hw(h: &HW) -> HW {
    HW { op: h.op.clone(), fn_: h.fn_.clone(), inp: h.inp.clone(), cw: h.cw.clone(), op2: h.op2.clone(), wc: h.wc.clone(), w1: h.w1.clone(), w3: h.w3.clone(), w2: h.w2.clone() }
}
