// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! gpu equivalence for the batched NoPE GQA attention prefill (N6,
//! `NemotronAttnGpu::prefill`) vs the per-token `forward` decode loop. Two blocks
//! with identical f32 weights and fresh KV caches: one prefills the whole prompt
//! with a single causal-masked flash, the other decodes token-by-token. The
//! per-position outputs must agree — confirming the KV-cache write layout, the
//! `positions`/`block_*` parameterisation of `attention_f32_batched_masked`, and
//! that the causal mask matches the decode causality.
//!
//!   hipfire lock acquire test_attn_prefill_gpu --watch-pid $$
//!   cargo run -p hipfire-arch-nemotron --example test_attn_prefill_gpu

use hipfire_arch_nemotron::attn::NemotronAttnGpu;
use hipfire_arch_nemotron::AttnConfig;
use hipfire_rdna::Gpu;

fn main() {
    let mut gpu = Gpu::init().expect("GPU init failed");
    eprintln!("GPU: {}", gpu.arch);

    let cfg = AttnConfig {
        num_heads: 4,
        num_kv_heads: 2,
        head_dim: 8,
        bias: false,
    };
    let hidden = 24usize;
    let seq = 27usize;
    let max_seq = 64usize;
    let q_dim = cfg.num_heads * cfg.head_dim;
    let kv_dim = cfg.num_kv_heads * cfg.head_dim;

    let mut seed = 0x0A11_CE99u32;
    let mut rng = || {
        seed ^= seed << 13;
        seed ^= seed >> 17;
        seed ^= seed << 5;
        (seed as f32 / u32::MAX as f32) * 0.4 - 0.2
    };
    let qw: Vec<f32> = (0..q_dim * hidden).map(|_| rng()).collect();
    let kw: Vec<f32> = (0..kv_dim * hidden).map(|_| rng()).collect();
    let vw: Vec<f32> = (0..kv_dim * hidden).map(|_| rng()).collect();
    let ow: Vec<f32> = (0..hidden * q_dim).map(|_| rng()).collect();
    let hidden_seq: Vec<f32> = (0..seq * hidden).map(|_| rng()).collect();

    // batched prefill
    let mut a_pf =
        NemotronAttnGpu::new(&mut gpu, cfg, hidden, max_seq, &qw, &kw, &vw, &ow).unwrap();
    let hs_g = gpu.upload_f32(&hidden_seq, &[seq * hidden]).unwrap();
    let out_pf_t = a_pf.prefill(&mut gpu, &hs_g, seq).unwrap();
    gpu.hip.device_synchronize().unwrap();
    let out_pf = gpu.download_f32(&out_pf_t).unwrap();

    // per-token decode loop
    let mut a_dec =
        NemotronAttnGpu::new(&mut gpu, cfg, hidden, max_seq, &qw, &kw, &vw, &ow).unwrap();
    let mut out_dec = vec![0.0f32; seq * hidden];
    for t in 0..seq {
        let row = gpu
            .upload_f32(&hidden_seq[t * hidden..(t + 1) * hidden], &[hidden])
            .unwrap();
        let o = a_dec.forward(&mut gpu, &row, t).unwrap();
        gpu.hip.device_synchronize().unwrap();
        let ov = gpu.download_f32(o).unwrap();
        out_dec[t * hidden..(t + 1) * hidden].copy_from_slice(&ov);
        let _ = gpu.free_tensor(row);
    }

    let max_d = out_pf
        .iter()
        .zip(&out_dec)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    // per-position worst, to spot a single mis-masked row.
    let mut worst_pos = 0usize;
    let mut worst = 0.0f32;
    for t in 0..seq {
        let d = (0..hidden)
            .map(|i| (out_pf[t * hidden + i] - out_dec[t * hidden + i]).abs())
            .fold(0.0f32, f32::max);
        if d > worst {
            worst = d;
            worst_pos = t;
        }
    }
    eprintln!(
        "seq={seq} heads={}/{} head_dim={}  max|Δ|={max_d:.3e} (worst pos {worst_pos})",
        cfg.num_heads, cfg.num_kv_heads, cfg.head_dim
    );

    if max_d < 1e-4 {
        println!("PASS: NemotronAttnGpu::prefill matches the decode loop (max|Δ|={max_d:.2e})");
    } else {
        println!("FAIL: attention prefill diverges (max|Δ|={max_d:.2e} at pos {worst_pos})");
        std::process::exit(1);
    }
}
