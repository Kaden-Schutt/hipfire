// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 hipfire contributors
// hipfire — see LICENSE and NOTICE in the project root.

//! Lean offline spec for the MiniMax-M2 MoE family (arch_id 10): identity + the
//! `Ingest` quant-policy (shared transformer prior). Deps only `hipfire-arch-api`.

use hipfire_arch_api::{
    default_importance, default_requires, register_arch, transformer_role, Arch, ArchId, CapReq,
    Ingest, Init, TensorRole, TensorSpec, ToyFixture, ToyModel,
};

/// MiniMax-M2 family header id.
pub const MINIMAX_ARCH_ID: ArchId = ArchId(10);

/// Lean identity marker for the MiniMax-M2 offline spec.
pub struct MinimaxSpec;

impl Arch for MinimaxSpec {
    fn id(&self) -> ArchId {
        MINIMAX_ARCH_ID
    }
    fn family(&self) -> &'static str {
        "minimax"
    }
}

impl Ingest for MinimaxSpec {
    fn role(&self, tensor: &str) -> TensorRole {
        transformer_role(tensor)
    }
    fn importance(&self, tensor: &str) -> u8 {
        default_importance(self.role(tensor))
    }
    fn requires(&self, tensor: &str) -> CapReq {
        default_requires(self.role(tensor))
    }
}

/// Tiny MiniMax-M2 (arch 10) Mixtral-style MoE config. Distinct from the
/// Qwen3.5 MoE: per-expert pre-split `w1/w3/w2` tensors (no stacked-3D),
/// per-layer flat QK-norm, partial rotate_half RoPE, sigmoid routing with a
/// per-expert `e_score_correction_bias`, no shared expert, and **untied**
/// lm_head. Exercises the indexed-MoE GEMV kernel family. Expert input dim
/// (hidden, inter) must be a multiple of 256 for the grouped-codec expert path.
struct MiniMaxTiny {
    hidden: usize,
    inter: usize,
    vocab: usize,
    layers: usize,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    rotary_dim: usize,
    experts: usize,
    experts_per_tok: usize,
}

impl MiniMaxTiny {
    fn preset() -> Self {
        Self {
            hidden: 256,
            inter: 256,
            vocab: 4096,
            layers: 2,
            n_heads: 2,
            n_kv_heads: 1,
            head_dim: 128,
            rotary_dim: 32,
            experts: 8,
            experts_per_tok: 2,
        }
    }

    fn config_json(&self) -> serde_json::Value {
        serde_json::json!({
            "architectures": ["MiniMaxM2ForCausalLM"],
            "model_type": "minimax_m2",
            "hidden_size": self.hidden,
            "intermediate_size": self.inter,
            "vocab_size": self.vocab,
            "num_hidden_layers": self.layers,
            "num_attention_heads": self.n_heads,
            "num_key_value_heads": self.n_kv_heads,
            "head_dim": self.head_dim,
            "rotary_dim": self.rotary_dim,
            "num_local_experts": self.experts,
            "num_experts_per_tok": self.experts_per_tok,
            "use_qk_norm": true,
            "use_routing_bias": true,
            "scoring_func": "sigmoid",
            "rope_theta": 5_000_000.0,
            "rms_norm_eps": 1e-6,
            "max_position_embeddings": 4096,
            "tie_word_embeddings": false,
            "dtype": "bfloat16",
            "_comment": "hipfire tiny random-init gating fixture — not a real model",
        })
    }

    fn manifest(&self) -> Vec<TensorSpec> {
        let h = self.hidden;
        let q_dim = self.n_heads * self.head_dim;
        let kv_dim = self.n_kv_heads * self.head_dim;
        let mut t = Vec::new();
        // Untied: embed + separate lm_head.
        t.push(TensorSpec::new(
            "model.embed_tokens.weight",
            vec![self.vocab, h],
            Init::Uniform(0.05),
        ));
        t.push(TensorSpec::f16(
            "model.norm.weight",
            vec![h],
            Init::NormOnes,
        ));
        t.push(TensorSpec::new(
            "lm_head.weight",
            vec![self.vocab, h],
            Init::Uniform(0.05),
        ));
        for i in 0..self.layers {
            let p = format!("model.layers.{i}");
            let sa = format!("{p}.self_attn");
            let moe = format!("{p}.block_sparse_moe");
            t.push(TensorSpec::f16(
                format!("{p}.input_layernorm.weight"),
                vec![h],
                Init::NormOnes,
            ));
            t.push(TensorSpec::f16(
                format!("{p}.post_attention_layernorm.weight"),
                vec![h],
                Init::NormOnes,
            ));
            // Per-layer QK-norm on the flat projection (q_dim / kv_dim wide).
            t.push(TensorSpec::f16(
                format!("{sa}.q_norm.weight"),
                vec![q_dim],
                Init::NormOnes,
            ));
            t.push(TensorSpec::f16(
                format!("{sa}.k_norm.weight"),
                vec![kv_dim],
                Init::NormOnes,
            ));
            t.push(TensorSpec::new(
                format!("{sa}.q_proj.weight"),
                vec![q_dim, h],
                Init::Uniform(0.05),
            ));
            t.push(TensorSpec::new(
                format!("{sa}.k_proj.weight"),
                vec![kv_dim, h],
                Init::Uniform(0.05),
            ));
            t.push(TensorSpec::new(
                format!("{sa}.v_proj.weight"),
                vec![kv_dim, h],
                Init::Uniform(0.05),
            ));
            t.push(TensorSpec::new(
                format!("{sa}.o_proj.weight"),
                vec![h, q_dim],
                Init::Uniform(0.05),
            ));
            // Router + per-expert bias (loaded unconditionally by the minimax loader).
            t.push(TensorSpec::new(
                format!("{moe}.gate.weight"),
                vec![self.experts, h],
                Init::Uniform(0.05),
            ));
            t.push(TensorSpec::f16(
                format!("{moe}.e_score_correction_bias"),
                vec![self.experts],
                Init::Uniform(0.02),
            ));
            for e in 0..self.experts {
                let ep = format!("{moe}.experts.{e}");
                t.push(TensorSpec::new(
                    format!("{ep}.w1.weight"),
                    vec![self.inter, h],
                    Init::Uniform(0.05),
                ));
                t.push(TensorSpec::new(
                    format!("{ep}.w3.weight"),
                    vec![self.inter, h],
                    Init::Uniform(0.05),
                ));
                t.push(TensorSpec::new(
                    format!("{ep}.w2.weight"),
                    vec![h, self.inter],
                    Init::Uniform(0.05),
                ));
            }
        }
        t
    }
}

impl ToyModel for MinimaxSpec {
    // Tiny random-init gating fixture, declared arch-side. Ported verbatim from the
    // quantizer's `MiniMaxTiny` so the emitted bytes stay identical (the tiny-quant
    // golden baselines depend on them). The quantizer owns the seeded RNG +
    // safetensors/tokenizer writing; this only describes shape + config.
    fn fixture(&self, _seed: u64) -> ToyFixture {
        let m = MiniMaxTiny::preset();
        ToyFixture {
            config_json: serde_json::to_string_pretty(&m.config_json())
                .expect("serialize minimax toy config"),
            tensors: m.manifest(),
        }
    }
}

static MINIMAX_SPEC: MinimaxSpec = MinimaxSpec;
register_arch!(MINIMAX_SPEC, Ingest, ToyModel);

#[cfg(test)]
mod tests {
    use super::*;
    use hipfire_arch_api::ArchRegistry;

    #[test]
    fn registers_ingest() {
        let reg = ArchRegistry::build();
        let a = reg.get(MINIMAX_ARCH_ID).expect("minimax spec registered");
        assert_eq!(a.family, "minimax");
        assert!(a.caps.ingest.is_some());
    }

    #[test]
    fn toy_fixture_declared() {
        let f = MinimaxSpec.fixture(0);
        assert!(!f.tensors.is_empty(), "minimax fixture emits tensors");
        assert!(f.config_json.contains("\"model_type\": \"minimax_m2\""));
    }
}
