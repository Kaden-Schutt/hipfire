// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Phase 1 smoke test: load Qwen3.5-35B-A3B (or any qwen3_5_moe HFQ) end-to-end
//! and report success/failure. No forward pass — just exercises the loader so we
//! catch tensor-name mismatches, dimension mismatches, and unsupported quant
//! types before wiring up the (much more involved) MoE forward path.
//!
//! Usage: cargo run --release --features deltanet --example a3b_load_check -- \
//!     ~/.hipfire/models/qwen3.5-35b-a3b.mq4

#[cfg(not(feature = "deltanet"))]
fn main() { eprintln!("build with --features deltanet"); }

#[cfg(feature = "deltanet")]
fn main() {
    use hipfire_runtime::hfq::HfqFile;
    use hipfire_arch_qwen35::qwen35::{self, LayerWeights};
    use std::path::Path;

    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: a3b_load_check <model.mq4>");
        std::process::exit(1);
    }
    let model_path = &args[1];
    eprintln!("Opening: {model_path}");

    let mut hfq = HfqFile::open(Path::new(model_path)).expect("open model");
    let config = qwen35::config_from_hfq(&hfq).expect("read config");

    eprintln!("Config:");
    eprintln!("  dim = {}",                  config.dim);
    eprintln!("  n_layers = {}",             config.n_layers);
    eprintln!("  vocab_size = {}",           config.vocab_size);
    eprintln!("  hidden_dim (dense FFN) = {}", config.hidden_dim);
    eprintln!("  n_heads = {}",              config.n_heads);
    eprintln!("  n_kv_heads = {}",           config.n_kv_heads);
    eprintln!("  head_dim = {}",             config.head_dim);
    eprintln!("  partial_rotary_factor = {}", config.partial_rotary_factor);
    eprintln!("  rope_theta = {}",           config.rope_theta);
    eprintln!("  ─── MoE ───");
    eprintln!("  num_experts = {}",          config.num_experts);
    eprintln!("  num_experts_per_tok = {}",  config.num_experts_per_tok);
    eprintln!("  moe_intermediate = {}",     config.moe_intermediate_size);
    eprintln!("  shared_expert_intermediate = {}", config.shared_expert_intermediate_size);
    eprintln!("  has_shared_expert = {}",    config.has_shared_expert);

    let n_la = config.layer_types.iter().filter(|t| matches!(t, qwen35::LayerType::LinearAttention)).count();
    let n_fa = config.n_layers - n_la;
    eprintln!("  layer mix = {n_la} LA + {n_fa} FA");

    eprintln!("\nInitializing GPU + loading weights ...");
    let mut gpu = rdna_compute::Gpu::init().expect("Gpu::init failed");
    let weights = qwen35::load_weights(&mut hfq, &config, &mut gpu).expect("load_weights failed");

    eprintln!("\n=== LOAD SUCCEEDED ===");
    eprintln!("Loaded {} layers:", weights.layers.len());
    let mut counts = (0, 0, 0, 0);
    for layer in &weights.layers {
        match layer {
            LayerWeights::DeltaNet(_)    => counts.0 += 1,
            LayerWeights::FullAttn(_)    => counts.1 += 1,
            LayerWeights::DeltaNetMoe(_) => counts.2 += 1,
            LayerWeights::FullAttnMoe(_) => counts.3 += 1,
        }
    }
    eprintln!("  DeltaNet (dense)    = {}", counts.0);
    eprintln!("  FullAttn (dense)    = {}", counts.1);
    eprintln!("  DeltaNet + MoE      = {}", counts.2);
    eprintln!("  FullAttn + MoE      = {}", counts.3);
    eprintln!("\nAll {} expert tensors loaded successfully across {} MoE layers.",
        config.num_experts * (counts.2 + counts.3) * 2,
        counts.2 + counts.3);
}
