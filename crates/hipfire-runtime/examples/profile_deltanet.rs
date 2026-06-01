// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Profile one DeltaNet layer: per-op wall-clock timing.
//! Usage: cargo run --release --features deltanet --example profile_deltanet -- <model.hfq>

#[cfg(not(feature = "deltanet"))]
fn main() { eprintln!("Build with --features deltanet"); }

#[cfg(feature = "deltanet")]
fn main() {
    use hipfire_runtime::hfq::HfqFile;
    use hipfire_arch_qwen35::qwen35;
    use hipfire_runtime::llama;
    use rdna_compute::DType;
    use std::path::Path;
    use std::time::Instant;

    let args: Vec<String> = std::env::args().collect();
    let model_path = args.get(1).expect("Usage: profile_deltanet <model.hfq>");

    let mut hfq = HfqFile::open(Path::new(model_path)).expect("failed to open HFQ");
    let config = qwen35::config_from_hfq(&hfq).expect("failed to parse config");
    eprintln!("dim={}, heads={}, linear_heads={}, head_dim={}",
        config.dim, config.n_heads, config.linear_num_key_heads, config.linear_key_head_dim);

    let mut gpu = rdna_compute::Gpu::init().expect("GPU init failed");
    let weights = qwen35::load_weights(&mut hfq, &config, &mut gpu).expect("load weights failed");
    let mut dn_state = qwen35::DeltaNetState::new(&mut gpu, &config).unwrap();
    let mut kv_cache = llama::KvCache::new_gpu(
        &mut gpu, config.n_layers, config.n_kv_heads, config.head_dim, 128,
    ).unwrap();

    // Warm up: run 5 tokens through entire model
    for i in 0..5u32 {
        qwen35::forward(&mut gpu, &weights, &config, 9419 + i, i as usize, &mut kv_cache, &mut dn_state).unwrap();
    }

    // Now profile layer 0 (DeltaNet) in isolation by timing a full forward pass
    // Use hipDeviceSynchronize between ops
    let layer = match &weights.layers[0] {
        qwen35::LayerWeights::DeltaNet(l) => l,
        _ => panic!("layer 0 not DeltaNet"),
    };

    let dim = config.dim;
    let qkv_dim = config.linear_num_key_heads * config.linear_key_head_dim * 2
                 + config.linear_num_value_heads * config.linear_value_head_dim;
    let d_inner = config.linear_num_value_heads * config.linear_value_head_dim;
    let n_v_heads = config.linear_num_value_heads;
    let k_dim = config.linear_num_key_heads * config.linear_key_head_dim;
    let v_dim = config.linear_num_value_heads * config.linear_value_head_dim;

    let _gemv = llama::gemv_family();
    let _ctx = llama::DispatchCtx::new(&mut gpu);

    // Allocate persistent buffers
    let x = gpu.alloc_tensor(&[dim], DType::F32).unwrap();
    let tmp = gpu.alloc_tensor(&[dim], DType::F32).unwrap();

    // Fill x with embedding of token 9419
    gpu.embedding_lookup_hfq4g128(&weights.token_embd, &x, 9419, dim).unwrap();

    let n_iters = 50;
    let mut timings: Vec<(&str, f64)> = Vec::new();

    macro_rules! timed {
        ($name:expr, $body:expr) => {{
            gpu.hip.device_synchronize().unwrap();
            let t0 = Instant::now();
            for _ in 0..n_iters { $body }
            gpu.hip.device_synchronize().unwrap();
            let us = t0.elapsed().as_micros() as f64 / n_iters as f64;
            timings.push(($name, us));
        }};
    }

    // 1. RMSNorm
    timed!("rmsnorm", {
        gpu.rmsnorm_f32(&x, &layer.attn_norm, &tmp, config.norm_eps).unwrap();
    });

    // 2. QKV GEMV
    let qkv = gpu.alloc_tensor(&[qkv_dim], DType::F32).unwrap();
    timed!("qkv_gemv", {
        _gemv.run_auto(&_ctx, &mut gpu, &layer.wqkv.dispatch_ref(), &tmp, &qkv).unwrap();
    });

    // 3. Z GEMV
    let z = gpu.alloc_tensor(&[d_inner], DType::F32).unwrap();
    timed!("z_gemv", {
        _gemv.run_auto(&_ctx, &mut gpu, &layer.wz.dispatch_ref(), &tmp, &z).unwrap();
    });

    // 4. Beta GEMV + sigmoid
    let beta_out = gpu.alloc_tensor(&[n_v_heads], DType::F32).unwrap();
    timed!("beta_gemv", {
        _gemv.run_auto(&_ctx, &mut gpu, &layer.w_beta.dispatch_ref(), &tmp, &beta_out).unwrap();
    });
    timed!("sigmoid", {
        gpu.sigmoid_f32(&beta_out).unwrap();
    });

    // 5. Alpha GEMV + GPU gate compute
    let alpha_out = gpu.alloc_tensor(&[n_v_heads], DType::F32).unwrap();
    timed!("alpha_gemv", {
        _gemv.run_auto(&_ctx, &mut gpu, &layer.w_alpha.dispatch_ref(), &tmp, &alpha_out).unwrap();
    });
    timed!("alpha_gate_gpu", {
        gpu.alpha_gate_f32(&alpha_out, &layer.dt_bias, &layer.a_log, n_v_heads).unwrap();
    });

    // 6. Fused conv1d + SiLU
    let conv_silu = gpu.alloc_tensor(&[qkv_dim], DType::F32).unwrap();
    timed!("conv1d_silu", {
        gpu.conv1d_silu_f32(&conv_silu, &qkv, &layer.conv_weight,
            &dn_state.conv_states[0], qkv_dim).unwrap();
    });

    // 8. D2D copies (split Q/K/V)
    let q_part = gpu.alloc_tensor(&[k_dim], DType::F32).unwrap();
    let k_part = gpu.alloc_tensor(&[k_dim], DType::F32).unwrap();
    let v_part = gpu.alloc_tensor(&[v_dim], DType::F32).unwrap();
    timed!("qkv_split", {
        gpu.hip.memcpy_dtod_at(&q_part.buf, 0, &conv_silu.buf, 0, k_dim * 4).unwrap();
        gpu.hip.memcpy_dtod_at(&k_part.buf, 0, &conv_silu.buf, k_dim * 4, k_dim * 4).unwrap();
        gpu.hip.memcpy_dtod_at(&v_part.buf, 0, &conv_silu.buf, k_dim * 2 * 4, v_dim * 4).unwrap();
    });

    // 9. L2 norm (x2)
    timed!("l2_norm_q", {
        gpu.l2_norm_f32(&q_part, config.linear_num_key_heads, config.linear_key_head_dim, config.norm_eps).unwrap();
    });
    timed!("l2_norm_k", {
        gpu.l2_norm_f32(&k_part, config.linear_num_key_heads, config.linear_key_head_dim, config.norm_eps).unwrap();
    });

    // 10. Q scale (GPU)
    timed!("q_scale_gpu", {
        gpu.scale_f32(&q_part, 1.0 / (config.linear_key_head_dim as f32).sqrt()).unwrap();
    });

    // 11. GDN recurrence
    let attn_out = gpu.alloc_tensor(&[v_dim], DType::F32).unwrap();
    timed!("gdn_recurrence", {
        gpu.gated_delta_net_f32(
            &q_part, &k_part, &v_part, &alpha_out, &beta_out,
            &dn_state.s_matrices[0], &attn_out,
            1, n_v_heads, config.linear_value_head_dim,
        ).unwrap();
    });

    // 12. Gated norm
    let normed_out = gpu.alloc_tensor(&[v_dim], DType::F32).unwrap();
    timed!("gated_norm", {
        gpu.gated_norm_f32(&attn_out, &z, &layer.norm_weight, &normed_out,
            n_v_heads, config.linear_value_head_dim, config.norm_eps).unwrap();
    });

    // 13. Output GEMV
    let o = gpu.alloc_tensor(&[dim], DType::F32).unwrap();
    timed!("out_gemv", {
        _gemv.run_auto(&_ctx, &mut gpu, &layer.wo.dispatch_ref(), &normed_out, &o).unwrap();
    });

    // 14. Residual add
    timed!("residual_add", {
        gpu.add_inplace_f32(&x, &o).unwrap();
    });

    // 15. FFN
    timed!("ffn_norm", {
        gpu.rmsnorm_f32(&x, &layer.ffn_norm, &tmp, config.norm_eps).unwrap();
    });
    let gate_buf = gpu.alloc_tensor(&[config.hidden_dim], DType::F32).unwrap();
    let up_buf = gpu.alloc_tensor(&[config.hidden_dim], DType::F32).unwrap();
    timed!("ffn_gate_gemv", {
        _gemv.run_auto(&_ctx, &mut gpu, &layer.w_gate.dispatch_ref(), &tmp, &gate_buf).unwrap();
    });
    timed!("ffn_up_gemv", {
        _gemv.run_auto(&_ctx, &mut gpu, &layer.w_up.dispatch_ref(), &tmp, &up_buf).unwrap();
    });
    let ffn_h = gpu.alloc_tensor(&[config.hidden_dim], DType::F32).unwrap();
    timed!("silu_mul", {
        gpu.silu_mul_f32(&gate_buf, &up_buf, &ffn_h).unwrap();
    });
    let ffn_out = gpu.alloc_tensor(&[dim], DType::F32).unwrap();
    timed!("ffn_down_gemv", {
        _gemv.run_auto(&_ctx, &mut gpu, &layer.w_down.dispatch_ref(), &ffn_h, &ffn_out).unwrap();
    });

    // Print results
    let total: f64 = timings.iter().map(|(_, us)| us).sum();
    eprintln!("\n=== DeltaNet Layer 0 Profile ({n_iters} iterations) ===");
    eprintln!("{:<20} {:>8} {:>6}", "Op", "µs", "%");
    eprintln!("{:-<36}", "");
    for (name, us) in &timings {
        let pct = us / total * 100.0;
        eprintln!("{:<20} {:>8.1} {:>5.1}%", name, us, pct);
    }
    eprintln!("{:-<36}", "");
    eprintln!("{:<20} {:>8.1}", "TOTAL", total);
    eprintln!("\nAt 18 DeltaNet + 6 FullAttn layers:");
    let dn_us = total * 18.0;
    eprintln!("  DeltaNet-only: {:.0}µs = {:.0} tok/s (DeltaNet layers only)", dn_us, 1e6 / dn_us);
    let full_tok_us = total * 18.0 + total * 0.6 * 6.0; // rough: full_attn ~60% of DeltaNet cost
    eprintln!("  Estimated full: {:.0}µs ≈ {:.0} tok/s", full_tok_us, 1e6 / full_tok_us);
}
