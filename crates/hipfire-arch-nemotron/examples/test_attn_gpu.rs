// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! gpu-vs-cpu correctness for the nemotron_h GQA attention block (`*`, NoPE).
//! Runs `NemotronAttnGpu::forward` and the CPU oracle (`attn::gqa_attention` +
//! o_proj) from identical state over several decode steps, accumulating the KV
//! cache, comparing the `[hidden]` output each step.
//!
//!   hipfire lock acquire test_attn_gpu --watch-pid $$
//!   cargo run -p hipfire-arch-nemotron --example test_attn_gpu

use hipfire_arch_nemotron::attn::{gqa_attention, NemotronAttnGpu};
use hipfire_arch_nemotron::AttnConfig;
use hipfire_rdna::Gpu;

fn matvec(w: &[f32], x: &[f32], out: usize, n_in: usize) -> Vec<f32> {
    (0..out)
        .map(|i| {
            w[i * n_in..i * n_in + n_in]
                .iter()
                .zip(x)
                .map(|(a, b)| a * b)
                .sum()
        })
        .collect()
}

fn main() {
    let mut gpu = Gpu::init().expect("GPU init failed");
    eprintln!("GPU: {}", gpu.arch);

    // GQA: 4 query heads, 2 kv heads, head_dim 8 (q_dim=32, kv_dim=16). hidden 24.
    let cfg = AttnConfig {
        num_heads: 4,
        num_kv_heads: 2,
        head_dim: 8,
        bias: false,
    };
    let hidden = 24usize;
    let max_seq = 32usize;
    let q_dim = cfg.num_heads * cfg.head_dim;
    let kv_dim = cfg.num_kv_heads * cfg.head_dim;

    let mut seed = 0x3C6E_F35Fu32;
    let mut rng = || {
        seed ^= seed << 13;
        seed ^= seed >> 17;
        seed ^= seed << 5;
        (seed as f32 / u32::MAX as f32) * 0.4 - 0.2
    };
    let q_proj: Vec<f32> = (0..q_dim * hidden).map(|_| rng()).collect();
    let k_proj: Vec<f32> = (0..kv_dim * hidden).map(|_| rng()).collect();
    let v_proj: Vec<f32> = (0..kv_dim * hidden).map(|_| rng()).collect();
    let o_proj: Vec<f32> = (0..hidden * q_dim).map(|_| rng()).collect();

    let mut attn = NemotronAttnGpu::new(
        &mut gpu, cfg, hidden, max_seq, &q_proj, &k_proj, &v_proj, &o_proj,
    )
    .unwrap();

    let mut k_hist: Vec<Vec<f32>> = Vec::new();
    let mut v_hist: Vec<Vec<f32>> = Vec::new();
    let mut worst = 0.0f32;

    for pos in 0..8 {
        let x: Vec<f32> = (0..hidden).map(|_| rng()).collect();

        // CPU oracle.
        let q = matvec(&q_proj, &x, q_dim, hidden);
        let k = matvec(&k_proj, &x, kv_dim, hidden);
        let v = matvec(&v_proj, &x, kv_dim, hidden);
        k_hist.push(k);
        v_hist.push(v);
        let a = gqa_attention(
            &q,
            &k_hist,
            &v_hist,
            cfg.num_heads,
            cfg.num_kv_heads,
            cfg.head_dim,
        );
        let cpu_out = matvec(&o_proj, &a, hidden, q_dim);

        // GPU.
        let d_x = gpu.upload_f32(&x, &[hidden]).unwrap();
        let out_t = attn.forward(&mut gpu, &d_x, pos).unwrap();
        gpu.hip.device_synchronize().unwrap();
        let g = gpu.download_f32(out_t).unwrap();

        let md = g
            .iter()
            .zip(&cpu_out)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        eprintln!("pos {pos}: max|Δout|={md:.3e}");
        worst = worst.max(md);
    }

    if worst > 1e-4 {
        eprintln!("FAIL (worst={worst:.3e})");
        std::process::exit(1);
    }
    println!("PASS: NemotronAttnGpu matches CPU GQA-attention oracle over 8 steps");
}
