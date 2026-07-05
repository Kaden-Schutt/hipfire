// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! `Architecture` trait impl for MiniMax-M2 (arch_id = 10).
//!
//! Thin marker + delegation, mirroring `hipfire-arch-deepseek4`'s
//! `arch.rs`. The forward pass is NOT on the trait — it lives as free
//! functions in `crate::forward` (hot-path static dispatch), called
//! directly by the daemon's `arch_id == 10` generate branch.

use crate::minimax::{MiniMaxConfig, MiniMaxState, MiniMaxWeights};
use hipfire_runtime::tp_shard::ExpertAssign;
use hipfire_runtime::weight_manifest::{
    FusedQkvLayout, PinTarget, ShardPolicy, StateEntry, StateKind, WeightEntry,
};
use rdna_compute::DType;
use hipfire_runtime::arch::Architecture;
use hipfire_runtime::hfq::HfqFile;
use rdna_compute::Gpu;

/// Zero-sized type marker for MiniMax-M2. Trait dispatch uses the type.
pub struct MiniMaxM2;

impl Architecture for MiniMaxM2 {
    type Weights = MiniMaxWeights;
    type State = MiniMaxState;
    type Config = MiniMaxConfig;

    /// Canonical family marker. Reserved in docs/architecture-ids.md
    /// (next free after DeepSeek V4 = 9).
    fn arch_id() -> u32 {
        10
    }

    fn name() -> &'static str {
        "minimax"
    }

    fn config_from_hfq(hfq: &HfqFile) -> Result<Self::Config, String> {
        MiniMaxConfig::from_hfq(hfq)
    }

    fn load_weights(
        hfq: &mut HfqFile,
        cfg: &Self::Config,
        gpu: &mut Gpu,
    ) -> Result<Self::Weights, String> {
        MiniMaxWeights::load(hfq, cfg, gpu, None)
    }

    fn new_state(gpu: &mut Gpu, cfg: &Self::Config) -> Result<Self::State, String> {
        MiniMaxState::new(gpu, cfg)
    }

    /// MoE weight manifest (device-mesh Phase 2) — first production use of
    /// `ExpertSharded`. Every layer is MoE: dense attention (Q/K/V column, O
    /// row) + a replicated router + `num_local_experts` experts distributed
    /// across the `Ep` group (the zeroed-dummy convention lives in the engine's
    /// fulfillment). Expert gate/up/down are one `ExpertSharded` entry each.
    fn weight_manifest(cfg: &Self::Config) -> Vec<WeightEntry> {
        use ShardPolicy::*;
        let (d, ff, hd) = (cfg.hidden_size, cfg.intermediate_size, cfg.head_dim);
        let (nh, nkv, ne) = (cfg.num_attention_heads, cfg.num_key_value_heads, cfg.num_local_experts);
        let expert = || ExpertSharded { n_experts: ne, assign: ExpertAssign::Stride };
        let mut m = Vec::new();
        m.push(WeightEntry::model("token_embd", vec![cfg.vocab_size, d], DType::F16, Pin(PinTarget::Embed)));
        for l in 0..cfg.num_hidden_layers {
            // Attention (dense).
            m.push(WeightEntry::layer("wq", l, vec![nh * hd, d], DType::F16, FusedQkv { q_heads: nh, kv_heads: nkv, head_dim: hd, layout: FusedQkvLayout::Qkv }));
            m.push(WeightEntry::layer("wk", l, vec![nkv * hd, d], DType::F16, ColumnShard { axis: 0 }));
            m.push(WeightEntry::layer("wv", l, vec![nkv * hd, d], DType::F16, ColumnShard { axis: 0 }));
            m.push(WeightEntry::layer("wo", l, vec![d, nh * hd], DType::F16, RowShard { axis: 1 }));
            // MoE: router replicated; experts sharded across the Ep group.
            m.push(WeightEntry::layer("router", l, vec![ne, d], DType::F32, Replicate));
            m.push(WeightEntry::layer("experts_gate", l, vec![ne, ff, d], DType::F16, expert()));
            m.push(WeightEntry::layer("experts_up", l, vec![ne, ff, d], DType::F16, expert()));
            m.push(WeightEntry::layer("experts_down", l, vec![ne, d, ff], DType::F16, expert()));
            m.push(WeightEntry::layer("attn_norm", l, vec![d], DType::F32, Replicate));
            m.push(WeightEntry::layer("ffn_norm", l, vec![d], DType::F32, Replicate));
        }
        m.push(WeightEntry::model("output_norm", vec![d], DType::F32, Replicate));
        m.push(WeightEntry::model("lm_head", vec![cfg.vocab_size, d], DType::F16, Pin(PinTarget::Output)));
        m
    }

    /// MiniMax is full-attention (Mixtral-style) → one KV `StateEntry`/layer.
    fn state_manifest(cfg: &Self::Config) -> Vec<StateEntry> {
        (0..cfg.num_hidden_layers)
            .map(|l| StateEntry::new(StateKind::Kv { quant: String::new() }, l))
            .collect()
    }

    // Optional overrides left at the ChatML defaults for the scaffold.
    // MiniMax-M2 ships its own chat template; revisit prompt_frame /
    // eos_filter overrides once tokenizer wiring lands.
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn minimax_arch_id_and_name() {
        assert_eq!(MiniMaxM2::arch_id(), 10);
        assert_eq!(MiniMaxM2::name(), "minimax");
    }
}
