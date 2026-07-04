// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! End-to-end equivalence for the batched N6 model prefill
//! (`NemotronModel::prefill_batched`) vs the per-token `forward_gpu` decode loop.
//! Two identical f32 synthetic models (pattern "M-*-": Mamba2, MLP, Attention,
//! MLP) process the same prompt; the LAST-position `[vocab]` logits (the
//! next-token distribution that drives the first generated token) must agree.
//! This is the FU5 gate: prefill == per-token decode over the prompt.
//!
//!   hipfire lock acquire test_model_prefill_gpu --watch-pid $$
//!   cargo run -p hipfire-arch-nemotron --example test_model_prefill_gpu

use hipfire_arch_nemotron::model::{HostBlock, NemotronModel, NemotronWeights};
use hipfire_arch_nemotron::{BlockKind, NemotronHConfig};
use hipfire_rdna::Gpu;

fn main() {
    let mut gpu = Gpu::init().expect("GPU init failed");
    eprintln!("GPU: {}", gpu.arch);

    let json = serde_json::json!({
        "model_type": "nemotron_h",
        "hidden_size": 16,
        "vocab_size": 16,
        "num_hidden_layers": 4,
        "rms_norm_eps": 1e-5,
        "tie_word_embeddings": false,
        "hybrid_override_pattern": "M-*-",
        "mamba_num_heads": 2,
        "mamba_head_dim": 4,
        "ssm_state_size": 4,
        "n_groups": 2,
        "conv_kernel": 4,
        "chunk_size": 256,
        "use_conv_bias": true,
        "mamba_proj_bias": false,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "head_dim": 8,
        "attention_bias": false,
        "intermediate_size": 20,
        "mlp_hidden_act": "relu2",
        "time_step_min": 0.001,
        "time_step_max": 0.1,
    });
    let cfg = NemotronHConfig::from_json(&json).unwrap();
    let hidden = cfg.hidden_size;
    let vocab = cfg.vocab_size;
    let dims = cfg.mamba2_dims();

    let mut seed = 0x1357_BD2Fu32;
    let mut rng = move || {
        seed ^= seed << 13;
        seed ^= seed >> 17;
        seed ^= seed << 5;
        (seed as f32 / u32::MAX as f32) * 0.4 - 0.2
    };
    let v = |n: usize, rng: &mut dyn FnMut() -> f32| (0..n).map(|_| rng()).collect::<Vec<f32>>();

    let mut blocks = Vec::new();
    for k in &cfg.blocks {
        blocks.push(match k {
            BlockKind::Mamba2 => HostBlock::Mamba2 {
                in_proj: v(dims.projection_size() * hidden, &mut rng),
                conv_weight: v(dims.conv_dim() * dims.conv_kernel, &mut rng),
                conv_bias: v(dims.conv_dim(), &mut rng),
                a_log: v(cfg.mamba.num_heads, &mut rng),
                d: v(cfg.mamba.num_heads, &mut rng),
                dt_bias: v(cfg.mamba.num_heads, &mut rng),
                mixer_norm: (0..dims.d_inner()).map(|_| 1.0 + rng()).collect(),
                out_proj: v(hidden * dims.d_inner(), &mut rng),
            },
            BlockKind::Mlp => HostBlock::Mlp {
                up: v(cfg.mlp_intermediate * hidden, &mut rng),
                down: v(hidden * cfg.mlp_intermediate, &mut rng),
            },
            BlockKind::Attention => {
                let a = cfg.attn;
                let qd = a.num_heads * a.head_dim;
                let kvd = a.num_kv_heads * a.head_dim;
                HostBlock::Attn {
                    q: v(qd * hidden, &mut rng),
                    k: v(kvd * hidden, &mut rng),
                    v: v(kvd * hidden, &mut rng),
                    o: v(hidden * qd, &mut rng),
                }
            }
            BlockKind::Moe => unreachable!("synthetic test config has no MoE blocks"),
        });
    }
    let weights = NemotronWeights {
        embeddings: v(vocab * hidden, &mut rng),
        layer_norm: (0..cfg.num_layers)
            .map(|_| (0..hidden).map(|_| 1.0 + rng()).collect())
            .collect(),
        blocks,
        norm_f: (0..hidden).map(|_| 1.0 + rng()).collect(),
        lm_head: v(vocab * hidden, &mut rng),
    };

    let tokens = [3u32, 7, 1, 12, 0, 5];

    // batched prefill → last-position logits
    let mut model_pf = NemotronModel::new(&mut gpu, cfg.clone(), &weights, 32).unwrap();
    assert!(
        model_pf.can_batched_prefill(),
        "f32 model should allow batched prefill"
    );
    model_pf.prefill_batched(&mut gpu, &tokens).unwrap();
    gpu.hip.device_synchronize().unwrap();
    let logits_pf = gpu.download_f32(model_pf.logits_tensor()).unwrap();

    // per-token decode loop → last-position logits
    let mut model_dec = NemotronModel::new(&mut gpu, cfg.clone(), &weights, 32).unwrap();
    for (pos, &tok) in tokens.iter().enumerate() {
        model_dec.forward_gpu(&mut gpu, tok, pos).unwrap();
    }
    gpu.hip.device_synchronize().unwrap();
    let logits_dec = gpu.download_f32(model_dec.logits_tensor()).unwrap();

    let max_d = logits_pf
        .iter()
        .zip(&logits_dec)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    let amax = |x: &[f32]| {
        let mut bi = 0;
        for i in 1..x.len() {
            if x[i] > x[bi] {
                bi = i;
            }
        }
        bi
    };
    eprintln!(
        "tokens={tokens:?}  max|Δlogit|={max_d:.3e}  argmax pf={} dec={}",
        amax(&logits_pf),
        amax(&logits_dec)
    );

    if max_d < 1e-3 && amax(&logits_pf) == amax(&logits_dec) {
        println!("PASS: prefill_batched last-pos logits == decode loop (max|Δ|={max_d:.2e})");
    } else {
        println!("FAIL: prefill_batched diverges from decode (max|Δ|={max_d:.2e})");
        std::process::exit(1);
    }
}
