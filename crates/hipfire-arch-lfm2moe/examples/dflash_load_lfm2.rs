// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
//
// DFlash step 6a: load the REAL LFM2.5-350M-Base body weights (bf16 safetensors)
// into the trainer Net at fp32 (warm-start), and confirm the block-parallel draft
// body forward runs finite/sane on them. Adapters are fresh-init; ctx is random
// (the real fc-projected context arrives with the data pipeline in 6c).

use hipfire_arch_lfm2moe::dflash_train as dt;
use rdna_compute::{DType, Gpu};
use std::path::Path;

fn frand(seed: usize) -> f32 { ((seed.wrapping_mul(2654435761) % 2000) as f32 / 1000.0) - 1.0 }

fn main() {
    let st_path = std::env::args().nth(1).unwrap_or_else(|| {
        "/root/.cache/huggingface/hub/models--LiquidAI--LFM2.5-350M-Base/snapshots/9960764e30892e01f29a6dc23df2533fcd8bd5ae/model.safetensors".to_string()
    });
    let mut gpu = Gpu::init().expect("GPU init");
    // d_tgt/vocab/n_tgt_layers only matter for adapters/head (not loaded here).
    let cfg = dt::Cfg::lfm2_350m(5120, 248320, 5);
    let d = cfg.d;

    println!("loading LFM2.5-350M body from {st_path}");
    let (layers, final_norm) = dt::load_lfm2_warmstart(&mut gpu, &cfg, Path::new(&st_path)).expect("load");
    println!("loaded {} layers + final_norm", layers.len());

    // quick stats on a couple of loaded tensors to confirm sane magnitudes
    for li in [0usize, 2] {
        let lw = &layers[li];
        let w1 = gpu.download_f32(&lw.w1).unwrap();
        let mean = w1.iter().sum::<f32>() / w1.len() as f32;
        let absmax = w1.iter().fold(0f32, |a, &x| a.max(x.abs()));
        let kind = if cfg.is_attn[li] { "attn" } else { "conv" };
        println!("  L{li} [{kind}] w1: n={} mean={mean:.5} absmax={absmax:.4} finite={}", w1.len(), w1.iter().all(|x| x.is_finite()));
    }

    // run the body forward on real weights (random body_in + random ctx)
    let b = 16usize; let n_ctx = 24usize;
    let body_in: Vec<f32> = (0..b * d).map(|i| 0.3 * frand(i + 1)).collect();
    let ctx_v: Vec<f32> = (0..n_ctx * d).map(|i| 0.3 * frand(i + 9999)).collect();
    let h0 = dt::up(&mut gpu, &body_in, &[b, d]);
    let ctx = dt::up(&mut gpu, &ctx_v, &[n_ctx, d]);
    let block_pos: Vec<i32> = (0..b).map(|i| (n_ctx + i) as i32).collect();
    let mut full: Vec<i32> = (0..n_ctx).map(|i| i as i32).collect();
    full.extend(block_pos.iter().copied());
    let block_pos_g = dt::upos(&mut gpu, &block_pos);
    let full_pos_g = dt::upos(&mut gpu, &full);
    let conv_state = gpu.zeros(&[d, cfg.conv_k - 1], DType::F32).unwrap();

    let (hout, _tape) = dt::body_forward(&mut gpu, &cfg, &layers, &h0, &ctx, &block_pos_g, &full_pos_g, &conv_state, b, n_ctx);
    let out = gpu.download_f32(&hout).unwrap();
    let finite = out.iter().all(|x| x.is_finite());
    let mean = out.iter().sum::<f32>() / out.len() as f32;
    let var = out.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / out.len() as f32;
    let fnw = gpu.download_f32(&final_norm).unwrap();
    println!("final_norm: mean={:.4} (warm-started from embedding_norm)", fnw.iter().sum::<f32>() / fnw.len() as f32);
    println!("body forward on REAL weights: shape {b}x{d} finite={finite} mean={mean:.4} std={:.4}", var.sqrt());

    if finite && var.sqrt() < 1e4 {
        println!("dflash_load_lfm2: PASS (real warm-start loads + body forward finite/sane)");
    } else {
        println!("dflash_load_lfm2: FAIL (finite={finite} std={:.3e})", var.sqrt());
        std::process::exit(1);
    }
}
