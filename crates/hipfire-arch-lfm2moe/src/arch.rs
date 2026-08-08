// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! `Architecture` trait impl for LFM2.5-MoE (arch_id = 11).
//!
//! Thin marker + delegation, mirroring `hipfire-arch-minimax`'s
//! `arch.rs`. The forward pass is NOT on the trait — it lives as free
//! functions in `crate::forward` (hot-path static dispatch), called
//! directly by the daemon's `arch_id == 11` generate branch.

use crate::config::Lfm2MoeConfig;
use crate::lfm2moe::{Lfm2MoeState, Lfm2MoeWeights};
use hipfire_runtime::arch::Architecture;
use hipfire_runtime::hfq::HfqFile;
use hipfire_runtime::moe_plan::MoEExecutionPolicy;
use hipfire_runtime::tp_shard::ExpertAssign;
use hipfire_runtime::weight_manifest::{
    ExpertGroupSpec, PinTarget, ShardPolicy, StateEntry, StateKind, WeightEntry,
};
use rdna_compute::{DType, Gpu};

/// Zero-sized type marker for LFM2.5-MoE. Trait dispatch uses the type.
pub struct Lfm2Moe;

impl Architecture for Lfm2Moe {
    type Weights = Lfm2MoeWeights;
    type State = Lfm2MoeState;
    type Config = Lfm2MoeConfig;

    /// Canonical architecture ID for LFM2.5-MoE.
    fn arch_id() -> u32 {
        11
    }

    fn name() -> &'static str {
        "lfm2moe"
    }

    fn config_from_hfq(hfq: &HfqFile) -> Result<Self::Config, String> {
        Lfm2MoeConfig::from_hfq(hfq)
    }

    fn load_weights(
        hfq: &mut HfqFile,
        cfg: &Self::Config,
        gpu: &mut Gpu,
    ) -> Result<Self::Weights, String> {
        Lfm2MoeWeights::load(hfq, cfg, gpu)
    }

    fn new_state(gpu: &mut Gpu, cfg: &Self::Config) -> Result<Self::State, String> {
        Lfm2MoeState::new(gpu, cfg)
    }

    /// MoE weight manifest (device-mesh Phase 2, STEP-004 follow-up).
    /// LFM2.5 interleaves conv/attention mixer layers (`layer_types`) and
    /// dense/MoE FFN layers (the first `num_dense_layers` are dense SwiGLU).
    /// The packed fused expert gate_up/down projections are declared as
    /// static `ExpertSharded` surrogates (MiniMax convention): under Single
    /// the policy-aware resolver projects them to Replicate; TP/EP
    /// resolution materializes the shards without manifest changes.
    fn weight_manifest(cfg: &Self::Config) -> Vec<WeightEntry> {
        use ShardPolicy::*;
        let d = cfg.hidden_size;
        let q_dim = cfg.num_attention_heads * cfg.head_dim;
        let kv_dim = cfg.num_key_value_heads * cfg.head_dim;
        let is_moe = cfg.num_experts > 0;
        let expert = || ShardPolicy::ExpertSharded {
            n_experts: cfg.num_experts,
            assign: ExpertAssign::Stride,
        };
        let mut m = Vec::new();
        m.push(WeightEntry::model(
            "token_embd",
            vec![cfg.vocab_size, d],
            DType::F16,
            Pin(PinTarget::Embed),
        ));
        m.push(WeightEntry::model(
            "embedding_norm",
            vec![d],
            DType::F32,
            Replicate,
        ));
        let lm_policy = if cfg.tie_word_embeddings {
            ShardPolicy::Tied {
                source: "token_embd".to_string(),
            }
        } else {
            Pin(PinTarget::Output)
        };
        m.push(WeightEntry::model(
            "lm_head",
            vec![cfg.vocab_size, d],
            DType::F16,
            lm_policy,
        ));

        for (layer, mixer) in cfg.layer_types.iter().enumerate() {
            m.push(WeightEntry::layer(
                "operator_norm",
                layer,
                vec![d],
                DType::F32,
                Replicate,
            ));
            match mixer {
                crate::config::MixerKind::Conv => {
                    m.push(WeightEntry::layer(
                        "in_proj",
                        layer,
                        vec![3 * d, d],
                        DType::F16,
                        Replicate,
                    ));
                    m.push(WeightEntry::layer(
                        "conv_weight",
                        layer,
                        vec![d, cfg.conv_kernel_size],
                        DType::F32,
                        Replicate,
                    ));
                    m.push(WeightEntry::layer(
                        "out_proj",
                        layer,
                        vec![d, d],
                        DType::F16,
                        Replicate,
                    ));
                }
                crate::config::MixerKind::Attention => {
                    m.push(WeightEntry::layer(
                        "wq",
                        layer,
                        vec![q_dim, d],
                        DType::F16,
                        Replicate,
                    ));
                    m.push(WeightEntry::layer(
                        "wk",
                        layer,
                        vec![kv_dim, d],
                        DType::F16,
                        Replicate,
                    ));
                    m.push(WeightEntry::layer(
                        "wv",
                        layer,
                        vec![kv_dim, d],
                        DType::F16,
                        Replicate,
                    ));
                    m.push(WeightEntry::layer(
                        "wo",
                        layer,
                        vec![d, q_dim],
                        DType::F16,
                        Replicate,
                    ));
                    m.push(WeightEntry::layer(
                        "q_norm",
                        layer,
                        vec![cfg.head_dim],
                        DType::F32,
                        Replicate,
                    ));
                    m.push(WeightEntry::layer(
                        "k_norm",
                        layer,
                        vec![cfg.head_dim],
                        DType::F32,
                        Replicate,
                    ));
                }
            }

            m.push(WeightEntry::layer(
                "ffn_norm",
                layer,
                vec![d],
                DType::F32,
                Replicate,
            ));
            if is_moe && layer >= cfg.num_dense_layers {
                m.push(WeightEntry::layer(
                    "router",
                    layer,
                    vec![cfg.num_experts, d],
                    DType::F32,
                    Replicate,
                ));
                // Packed fused expert gate_up (2·moe_inter rows per expert)
                // and down — one entry per logical group (MiniMax-style
                // packed surrogate; the loader's per-expert fused buffers
                // fulfill the group).
                m.push(WeightEntry::layer(
                    "experts_gate_up",
                    layer,
                    vec![cfg.num_experts, 2 * cfg.moe_intermediate_size, d],
                    DType::F16,
                    expert(),
                ));
                m.push(WeightEntry::layer(
                    "experts_down",
                    layer,
                    vec![cfg.num_experts, d, cfg.moe_intermediate_size],
                    DType::F16,
                    expert(),
                ));
            } else {
                m.push(WeightEntry::layer(
                    "w1",
                    layer,
                    vec![cfg.intermediate_size, d],
                    DType::F16,
                    Replicate,
                ));
                m.push(WeightEntry::layer(
                    "w3",
                    layer,
                    vec![cfg.intermediate_size, d],
                    DType::F16,
                    Replicate,
                ));
                m.push(WeightEntry::layer(
                    "w2",
                    layer,
                    vec![d, cfg.intermediate_size],
                    DType::F16,
                    Replicate,
                ));
            }
        }
        m
    }

    /// State manifest (device-mesh Phase 2): LFM2.5 mixer layers own either a
    /// short-conv state (`StateKind::Conv`) or a KV cache
    /// (`StateKind::Kv`), keyed by the GLOBAL layer index from
    /// `config.layer_types`.
    fn state_manifest(cfg: &Self::Config) -> Vec<StateEntry> {
        cfg.layer_types
            .iter()
            .enumerate()
            .map(|(layer, mixer)| match mixer {
                crate::config::MixerKind::Conv => StateEntry::new(StateKind::Conv, layer),
                crate::config::MixerKind::Attention => StateEntry::new(
                    StateKind::Kv {
                        quant: String::new(),
                    },
                    layer,
                ),
            })
            .collect()
    }

    /// Policy-aware MoE expert-group manifest (STEP-004 follow-up): one
    /// layer-local group per MoE layer (`num_dense_layers..num_hidden_layers`).
    /// Parallelism derives EXCLUSIVELY from the caller-supplied policy.
    /// Router identity is `sigmoid_topk` (LFM2 scores are sigmoid-activated
    /// and routed by the bias-aware top-k — the routed program's
    /// `ScoreActivation` + `MoeRoute` steps); `indexed_quantized` matches the
    /// typed `IndexedMoeGemv` execution the forward lowers with. Sources
    /// reference the packed expert manifest entries and the replicated
    /// router.
    fn expert_group_manifest(
        cfg: &Self::Config,
        policy: &MoEExecutionPolicy,
    ) -> Vec<ExpertGroupSpec> {
        use hipfire_runtime::moe_plan::MoEExecutionKind;
        use hipfire_runtime::weight_manifest::{
            ExpertExecutionIdentity, ExpertGroupSpec, ExpertParallelism,
            ExpertResourceRequirements, ExpertSourceLayout,
        };
        if cfg.num_experts == 0 {
            return Vec::new();
        }
        let parallelism = match policy.kind() {
            MoEExecutionKind::Single => ExpertParallelism::Single,
            MoEExecutionKind::Tp => ExpertParallelism::TensorParallel,
            MoEExecutionKind::Ep => ExpertParallelism::ExpertParallel,
        };
        (cfg.num_dense_layers..cfg.num_hidden_layers)
            .map(|layer| ExpertGroupSpec {
                group: "moe".into(),
                layer: Some(layer),
                n_experts: cfg.num_experts,
                parallelism,
                assignment: ExpertAssign::Stride,
                source_layout: ExpertSourceLayout::PackedFused {
                    gate_up: "experts_gate_up".into(),
                    down: "experts_down".into(),
                    sidecars: Vec::new(),
                },
                resources: ExpertResourceRequirements {
                    // Truthful declared on-disk footprint of one expert: the
                    // fused gate_up (2·moe_inter × hidden) + down
                    // (hidden × moe_inter) projections at 4-bit (MQ4). The
                    // lowerer never reads this field — manifest metadata.
                    bytes_per_expert: (3 * cfg.hidden_size * cfg.moe_intermediate_size) / 2,
                    alignment: 256,
                },
                router: "router".into(),
                router_identity: "sigmoid_topk".into(),
                allowed_executions: vec![ExpertExecutionIdentity::IndexedQuantized],
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lfm2moe_arch_id_and_name() {
        assert_eq!(Lfm2Moe::arch_id(), 11);
        assert_eq!(Lfm2Moe::name(), "lfm2moe");
    }

    /// Build a synthetic 6-layer LFM2 config: 2 dense + 4 MoE FFN layers,
    /// conv/attention mixers interleaved, 8 experts.
    fn test_config() -> Lfm2MoeConfig {
        use crate::config::MixerKind;
        Lfm2MoeConfig {
            vocab_size: 32000,
            hidden_size: 512,
            num_hidden_layers: 6,
            num_attention_heads: 8,
            num_key_value_heads: 4,
            head_dim: 64,
            conv_kernel_size: 4,
            intermediate_size: 1024,
            moe_intermediate_size: 1024,
            num_experts: 8,
            num_experts_per_tok: 4,
            num_dense_layers: 2,
            rope_theta: 10000.0,
            rms_norm_eps: 1e-6,
            max_position_embeddings: 4096,
            norm_topk_prob: true,
            use_expert_bias: true,
            routed_scaling_factor: 1.0,
            tie_word_embeddings: true,
            layer_types: vec![
                MixerKind::Conv,
                MixerKind::Attention,
                MixerKind::Conv,
                MixerKind::Attention,
                MixerKind::Conv,
                MixerKind::Attention,
            ],
            reap_keep: None,
        }
    }

    #[test]
    fn expert_group_manifest_resolves_single_policy_plans_for_moe_span() {
        let cfg = test_config();
        let policy = hipfire_runtime::moe_plan::MoEExecutionPolicy::single();
        let specs = Lfm2Moe::expert_group_manifest(&cfg, &policy);
        // Dense prefix (layers 0..2) contributes no groups; 4 MoE layers.
        assert_eq!(specs.len(), 4);
        assert!(specs.iter().all(|s| s.layer.unwrap() >= 2));
        let resolution = hipfire_runtime::weight_manifest::resolve_expert_manifest_for_policy(
            &specs,
            &Lfm2Moe::weight_manifest(&cfg),
            &policy,
        )
        .expect("single-policy manifest resolution");
        assert_eq!(resolution.plans.len(), 4);
        for (i, plan) in resolution.plans.iter().enumerate() {
            assert_eq!(plan.layer, Some(2 + i));
            assert_eq!(plan.router_identity, "sigmoid_topk");
            assert!(plan.allowed_executions.contains(
                &hipfire_runtime::weight_manifest::ExpertExecutionIdentity::IndexedQuantized
            ));
        }
        let plans = crate::lfm2moe::Lfm2MoeGroupPlans {
            plans: resolution.plans,
        };
        assert!(plans.by_layer(2).is_ok());
        assert!(plans.by_layer(5).is_ok());
        assert!(plans.by_layer(0).is_err(), "dense layers have no plan");
        assert!(plans.by_layer(6).is_err(), "out of range");
    }

    #[test]
    fn dense_only_config_resolves_no_groups() {
        let mut cfg = test_config();
        cfg.num_experts = 0;
        cfg.num_dense_layers = cfg.num_hidden_layers;
        let plans = crate::lfm2moe::Lfm2MoeGroupPlans::resolve(&cfg).expect("dense resolve");
        assert_eq!(plans.plans.len(), 0);
    }
}
