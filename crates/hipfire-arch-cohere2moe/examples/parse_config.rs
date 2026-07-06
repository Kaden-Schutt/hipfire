// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
//! Validate the Cohere2MoeConfig parser against a real config.json (no GPU).
//!   cargo run -p hipfire-arch-cohere2moe --example parse_config -- [config.json]

use hipfire_arch_cohere2moe::config::{AttnKind, Cohere2MoeConfig};

fn main() {
    let path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "/data/hipfire-models/North-Mini-Code-1.0/config.json".to_string());
    let v: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&path).expect("read config.json"))
            .expect("parse json");
    let cfg = Cohere2MoeConfig::from_config_value(&v).expect("parse Cohere2MoeConfig");
    println!("{cfg:#?}");
    let n_full = cfg
        .layer_types
        .iter()
        .filter(|&&k| k == AttnKind::Full)
        .count();
    let n_slide = cfg.num_hidden_layers - n_full;
    println!("---");
    println!("q_dim={} kv_dim={}", cfg.q_dim(), cfg.kv_dim());
    println!(
        "full(NoPE)={n_full}  sliding(RoPE)={n_slide}  dense_prefix={}",
        cfg.first_k_dense_replace
    );
    println!(
        "layer0 dense={}  ffn_inter(L0)={}  ffn_inter(L1)={}",
        cfg.is_dense_ffn(0),
        cfg.ffn_intermediate(0),
        cfg.ffn_intermediate(1)
    );
    println!(
        "uses_rope: L0={} L1={} L4={}",
        cfg.uses_rope(0),
        cfg.uses_rope(1),
        cfg.uses_rope(4)
    );
}
