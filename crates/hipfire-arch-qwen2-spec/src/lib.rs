// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 hipfire contributors
// hipfire — see LICENSE and NOTICE in the project root.

//! Lean offline spec for the Qwen2/Qwen3 dense family (arch_id 1): identity + the
//! `Ingest` quant-policy (shared transformer prior). Deps only `hipfire-arch-api`.

use hipfire_arch_api::{
    default_importance, default_requires, register_arch, transformer_role, Arch, ArchId, CapReq,
    Ingest, Init, TensorRole, TensorSpec, ToyFixture, ToyModel,
};

/// Qwen2/Qwen3 dense family header id.
pub const QWEN2_ARCH_ID: ArchId = ArchId(1);

/// Lean identity marker for the Qwen2/Qwen3 dense offline spec.
pub struct Qwen2Spec;

impl Arch for Qwen2Spec {
    fn id(&self) -> ArchId {
        QWEN2_ARCH_ID
    }
    fn family(&self) -> &'static str {
        "qwen2"
    }
}

impl Ingest for Qwen2Spec {
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

/// Tiny Qwen2 (arch 7) dense text config. The distinguishing feature vs LLaMA is
/// Q/K/V **bias** (attention_bias=true) — routed through the dedicated
/// hipfire-arch-qwen2 crate, which the LLaMA-default arch_id=1 path silently
/// drops. The emit-time config carries `model_type:"qwen2"` (auto-detect →
/// arch_id 1); the quant step must pass `--arch-id 7` to reach the qwen2 loader.
struct Qwen2Tiny {
    hidden: usize,
    inter: usize,
    vocab: usize,
    layers: usize,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
}

impl Qwen2Tiny {
    fn preset() -> Self {
        Self {
            hidden: 256,
            inter: 512,
            vocab: 4096,
            layers: 2,
            n_heads: 2,
            n_kv_heads: 1,
            head_dim: 128,
        }
    }

    fn config_json(&self) -> serde_json::Value {
        serde_json::json!({
            "architectures": ["Qwen2ForCausalLM"],
            "model_type": "qwen2",
            "hidden_size": self.hidden,
            "intermediate_size": self.inter,
            "vocab_size": self.vocab,
            "num_hidden_layers": self.layers,
            "num_attention_heads": self.n_heads,
            "num_key_value_heads": self.n_kv_heads,
            "head_dim": self.head_dim,
            "attention_bias": true,
            "hidden_act": "silu",
            "rms_norm_eps": 1e-6,
            "rope_theta": 1_000_000.0,
            "max_position_embeddings": 4096,
            "tie_word_embeddings": true,
            "dtype": "bfloat16",
            "_comment": "hipfire tiny random-init gating fixture — not a real model",
        })
    }

    fn manifest(&self) -> Vec<TensorSpec> {
        let h = self.hidden;
        let q_dim = self.n_heads * self.head_dim;
        let kv_dim = self.n_kv_heads * self.head_dim;
        let mut t = Vec::new();
        // tie_word_embeddings ⇒ no separate lm_head.
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
        for i in 0..self.layers {
            let p = format!("model.layers.{i}");
            t.push(TensorSpec::f16(
                format!("{p}.input_layernorm.weight"),
                vec![h],
                Init::NormOnes,
            ));
            t.push(TensorSpec::new(
                format!("{p}.self_attn.q_proj.weight"),
                vec![q_dim, h],
                Init::Uniform(0.05),
            ));
            t.push(TensorSpec::f16(
                format!("{p}.self_attn.q_proj.bias"),
                vec![q_dim],
                Init::Uniform(0.02),
            ));
            t.push(TensorSpec::new(
                format!("{p}.self_attn.k_proj.weight"),
                vec![kv_dim, h],
                Init::Uniform(0.05),
            ));
            t.push(TensorSpec::f16(
                format!("{p}.self_attn.k_proj.bias"),
                vec![kv_dim],
                Init::Uniform(0.02),
            ));
            t.push(TensorSpec::new(
                format!("{p}.self_attn.v_proj.weight"),
                vec![kv_dim, h],
                Init::Uniform(0.05),
            ));
            t.push(TensorSpec::f16(
                format!("{p}.self_attn.v_proj.bias"),
                vec![kv_dim],
                Init::Uniform(0.02),
            ));
            t.push(TensorSpec::new(
                format!("{p}.self_attn.o_proj.weight"),
                vec![h, q_dim],
                Init::Uniform(0.05),
            ));
            t.push(TensorSpec::f16(
                format!("{p}.post_attention_layernorm.weight"),
                vec![h],
                Init::NormOnes,
            ));
            t.push(TensorSpec::new(
                format!("{p}.mlp.gate_proj.weight"),
                vec![self.inter, h],
                Init::Uniform(0.05),
            ));
            t.push(TensorSpec::new(
                format!("{p}.mlp.up_proj.weight"),
                vec![self.inter, h],
                Init::Uniform(0.05),
            ));
            t.push(TensorSpec::new(
                format!("{p}.mlp.down_proj.weight"),
                vec![h, self.inter],
                Init::Uniform(0.05),
            ));
        }
        t
    }
}

impl ToyModel for Qwen2Spec {
    fn fixture(&self, _seed: u64) -> ToyFixture {
        let m = Qwen2Tiny::preset();
        ToyFixture {
            config_json: serde_json::to_string_pretty(&m.config_json())
                .expect("serialize qwen2 toy config"),
            tensors: m.manifest(),
        }
    }
}

static QWEN2_SPEC: Qwen2Spec = Qwen2Spec;
register_arch!(QWEN2_SPEC, Ingest, ToyModel);

#[cfg(test)]
mod tests {
    use super::*;
    use hipfire_arch_api::ArchRegistry;

    #[test]
    fn registers_ingest() {
        let reg = ArchRegistry::build();
        let a = reg.get(QWEN2_ARCH_ID).expect("qwen2 spec registered");
        assert_eq!(a.family, "qwen2");
        assert!(a.caps.ingest.is_some());
    }

    #[test]
    fn toy_fixture_populated_and_qwen2() {
        let f = Qwen2Spec.fixture(0);
        assert!(!f.tensors.is_empty(), "fixture must emit tensors");
        assert!(f.config_json.contains("\"model_type\": \"qwen2\""));
    }
}
