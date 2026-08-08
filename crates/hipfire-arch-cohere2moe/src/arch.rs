// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! `Architecture` trait impl for Cohere2-MoE (North-Mini-Code). arch_id = 12
//! (next free after lfm2/lfm2_moe = 11; see docs/architecture-ids.md).
//!
//! Maps onto hipfire's existing kernels with NO new GPU kernels: GQA +
//! interleaved (GPT-J) RoPE (skipped on the global/NoPE layers; sliding layers
//! use the windowed flash path) + Q8 KV attention, `rmsnorm_batched` (RMSNorm at
//! `rms_norm_eps` — NOT base Cohere2's mean-centered LayerNorm), `moe_topk_renorm_k8`
//! (sigmoid scoring, `norm_topk_prob=false`), and the qwen35/lfm2/minimax
//! indexed-MoE GEMV family for the MQ4/MQ6 expert tiers.

use crate::cohere2moe::{Cohere2MoeState, Cohere2MoeWeights};
use crate::config::Cohere2MoeConfig;
use hipfire_runtime::arch::{Architecture, EosFilterOverrides};
use hipfire_runtime::hfq::HfqFile;
use hipfire_runtime::moe_plan::MoEExecutionPolicy;
use hipfire_runtime::tp_shard::ExpertAssign;
use hipfire_runtime::weight_manifest::{
    ExpertExecutionIdentity, ExpertGroupSpec, ExpertParallelism, ExpertResourceRequirements,
    ExpertSourceLayout, PinTarget, ShardPolicy, StateEntry, StateKind, WeightEntry,
};
use rdna_compute::{DType, Gpu};

/// Type marker for the Cohere2-MoE family (`arch_id = 12`).
pub struct Cohere2Moe;

impl Architecture for Cohere2Moe {
    type Weights = Cohere2MoeWeights;
    type State = Cohere2MoeState;
    type Config = Cohere2MoeConfig;

    fn arch_id() -> u32 {
        12
    }

    fn name() -> &'static str {
        "cohere2moe"
    }

    fn config_from_hfq(hfq: &HfqFile) -> Result<Self::Config, String> {
        Cohere2MoeConfig::from_hfq(hfq)
    }

    fn load_weights(
        hfq: &mut HfqFile,
        cfg: &Self::Config,
        gpu: &mut Gpu,
    ) -> Result<Self::Weights, String> {
        Cohere2MoeWeights::load(hfq, cfg, gpu)
    }

    fn new_state(gpu: &mut Gpu, cfg: &Self::Config) -> Result<Self::State, String> {
        Cohere2MoeState::new(gpu, cfg)
    }

    /// Cohere2 ends a turn with `<|END_OF_TURN_TOKEN|>` (id 255001), not the
    /// ChatML `<|im_end|>`. Stop the visible stream on those bytes. (Prompt
    /// framing for the Cohere multi-template chat format is a serving-time
    /// follow-up; the KLD/PPL harness feeds raw token ids and bypasses it.)
    fn eos_filter_overrides(_cfg: &Self::Config) -> EosFilterOverrides {
        EosFilterOverrides {
            stop_at: vec![b"<|END_OF_TURN_TOKEN|>".to_vec()],
            holdback_prefixes: vec![b"<|END_OF_TURN_TOKEN".to_vec(), b"<|".to_vec()],
            strip_think: Some(false),
        }
    }

    /// MoE weight manifest (device-mesh Phase 2, STEP-004 follow-up).
    /// Cohere2-MoE is the parallel block: one `input_norm` feeds both the
    /// attention and the FFN branch of every layer. The first
    /// `first_k_dense_replace` FFN layers are dense SwiGLU; the rest are MoE.
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
            "final_norm",
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

        for layer in 0..cfg.num_hidden_layers {
            m.push(WeightEntry::layer(
                "input_norm",
                layer,
                vec![d],
                DType::F32,
                Replicate,
            ));
            // Attention (dense GQA).
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
            if is_moe && layer >= cfg.first_k_dense_replace {
                m.push(WeightEntry::layer(
                    "router",
                    layer,
                    vec![cfg.num_experts, d],
                    DType::F32,
                    Replicate,
                ));
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
                    "dense_gate",
                    layer,
                    vec![cfg.dense_intermediate_size, d],
                    DType::F16,
                    Replicate,
                ));
                m.push(WeightEntry::layer(
                    "dense_up",
                    layer,
                    vec![cfg.dense_intermediate_size, d],
                    DType::F16,
                    Replicate,
                ));
                m.push(WeightEntry::layer(
                    "dense_down",
                    layer,
                    vec![d, cfg.dense_intermediate_size],
                    DType::F16,
                    Replicate,
                ));
            }
        }
        m
    }

    /// State manifest (device-mesh Phase 2): every Cohere2 layer is
    /// full/sliding-window attention → one KV `StateEntry` per layer.
    fn state_manifest(cfg: &Self::Config) -> Vec<StateEntry> {
        (0..cfg.num_hidden_layers)
            .map(|layer| {
                StateEntry::new(
                    StateKind::Kv {
                        quant: String::new(),
                    },
                    layer,
                )
            })
            .collect()
    }

    /// Policy-aware MoE expert-group manifest (STEP-004 follow-up): one
    /// layer-local group per MoE layer (`first_k_dense_replace..
    /// num_hidden_layers`). Parallelism derives EXCLUSIVELY from the
    /// caller-supplied policy. Router identity is `sigmoid_topk` (Cohere2
    /// scores are sigmoid-activated; the top-k itself stays OUTSIDE the
    /// routed program — its norm_topk_prob=false renorm has no Step variant —
    /// so the forward's router phase is empty and only the expert phases
    /// lower). `indexed_quantized` matches the typed `IndexedMoeGemv`
    /// execution the forward lowers with. Sources reference the packed
    /// expert manifest entries and the replicated router.
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
        (cfg.first_k_dense_replace..cfg.num_hidden_layers)
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
    fn arch_id_and_name() {
        assert_eq!(Cohere2Moe::arch_id(), 12);
        assert_eq!(Cohere2Moe::name(), "cohere2moe");
    }

    /// Build a synthetic 6-layer Cohere2 config: 2 dense + 4 MoE FFN layers,
    /// 8 experts.
    fn test_config() -> Cohere2MoeConfig {
        use crate::config::AttnKind;
        Cohere2MoeConfig {
            vocab_size: 32000,
            hidden_size: 512,
            num_hidden_layers: 6,
            num_attention_heads: 8,
            num_key_value_heads: 4,
            head_dim: 64,
            moe_intermediate_size: 1024,
            dense_intermediate_size: 1024,
            num_experts: 8,
            num_experts_per_tok: 4,
            first_k_dense_replace: 2,
            prefix_dense_sliding_window_pattern: 0,
            num_shared_experts: 0,
            rope_theta: 10000.0,
            layer_norm_eps: 1e-5,
            rms_norm_eps: 1e-6,
            max_position_embeddings: 4096,
            sliding_window: 4096,
            norm_topk_prob: false,
            logit_scale: 1.0,
            tie_word_embeddings: true,
            layer_types: vec![AttnKind::Full; 6],
        }
    }

    #[test]
    fn expert_group_manifest_resolves_single_policy_plans_for_moe_span() {
        let cfg = test_config();
        let policy = hipfire_runtime::moe_plan::MoEExecutionPolicy::single();
        let specs = Cohere2Moe::expert_group_manifest(&cfg, &policy);
        // Dense prefix (layers 0..2) contributes no groups; 4 MoE layers.
        assert_eq!(specs.len(), 4);
        assert!(specs.iter().all(|s| s.layer.unwrap() >= 2));
        let resolution = hipfire_runtime::weight_manifest::resolve_expert_manifest_for_policy(
            &specs,
            &Cohere2Moe::weight_manifest(&cfg),
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
        let plans = crate::cohere2moe::Cohere2MoeGroupPlans {
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
        cfg.first_k_dense_replace = cfg.num_hidden_layers;
        let plans = crate::cohere2moe::Cohere2MoeGroupPlans::resolve(&cfg).expect("dense resolve");
        assert_eq!(plans.plans.len(), 0);
    }
}
