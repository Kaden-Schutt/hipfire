// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 hipfire contributors
// hipfire — see LICENSE and NOTICE in the project root.

//! Lean offline spec for the Gemma-3 family (arch_id 12): identity + the `Ingest`
//! quant-policy, no runtime/kernel deps (the quantizer links this without the GPU
//! stack). The serving crate (`hipfire-arch-gemma3`) declares runtime capabilities
//! on the same [`ArchId`]; the registry merges them.
//!
//! Gemma-3 is a dense transformer with llama-shaped tensor names (plus per-head
//! `q_norm`/`k_norm`). Like the llama spec, `importance` is a STRUCTURAL PRIOR, not
//! a tuned bit assignment — exact per-tensor precision is the quantizer's job later,
//! with a safe high-precision (bf16) fallback for coherence.

use hipfire_arch_api::{
    register_arch, Arch, ArchId, CapReq, Ingest, Init, TensorRole, TensorSpec, ToyFixture, ToyModel,
};

/// Gemma-3 family header id.
pub const GEMMA3_ARCH_ID: ArchId = ArchId(12);

/// Lean identity marker for the Gemma-3 family's offline spec.
pub struct Gemma3Spec;

impl Arch for Gemma3Spec {
    fn id(&self) -> ArchId {
        GEMMA3_ARCH_ID
    }
    fn family(&self) -> &'static str {
        "gemma3"
    }
}

impl Ingest for Gemma3Spec {
    fn role(&self, tensor: &str) -> TensorRole {
        if tensor.contains("embed_tokens") {
            TensorRole::Embed
        } else if tensor.contains("lm_head") {
            TensorRole::LmHead
        } else if tensor.contains("q_proj")
            || tensor.contains("k_proj")
            || tensor.contains("v_proj")
            || tensor.contains("o_proj")
        {
            TensorRole::AttnProj
        } else if tensor.contains("gate_proj")
            || tensor.contains("up_proj")
            || tensor.contains("down_proj")
        {
            TensorRole::Mlp
        } else if tensor.contains("norm") {
            // Includes gemma's per-head q_norm/k_norm and the layer RMSNorms.
            TensorRole::Norm
        } else {
            TensorRole::Other
        }
    }

    fn importance(&self, tensor: &str) -> u8 {
        // Structural prior: protect the gather-indexed tables, attention, and the
        // (tiny, sensitive) norms; compress the MLP bulk. Refined by the quantizer.
        match self.role(tensor) {
            TensorRole::Embed | TensorRole::LmHead => 255,
            TensorRole::Norm => 255,
            TensorRole::AttnProj => 255,
            TensorRole::Mlp => 128,
            _ => 160,
        }
    }

    fn requires(&self, tensor: &str) -> CapReq {
        match self.role(tensor) {
            TensorRole::Embed | TensorRole::LmHead => CapReq::RANDOM_ACCESS,
            _ => CapReq::NONE,
        }
    }
}

/// Tiny Gemma3 (arch 12) dense text config. Exercises the Gemma quirks the
/// ingest+forward special-case: per-head QK-norm, 4 norms/layer (the
/// pre/post feed-forward norms), GeGLU, head_dim independent of dim/n_heads,
/// dual-θ sliding-window interleave, and the (1+w) RMSNorm offset the quantizer
/// bakes at ingest (arch_id 12). `sliding_window_pattern:2` over 4 layers gives
/// both local-SWA and global layers.
struct Gemma3Tiny {
    hidden: usize,
    inter: usize,
    vocab: usize,
    layers: usize,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    sliding_window_pattern: usize,
}

impl Gemma3Tiny {
    fn preset() -> Self {
        Self {
            hidden: 256,
            inter: 512,
            vocab: 4096,
            layers: 4,
            n_heads: 2,
            n_kv_heads: 1,
            head_dim: 128, // must be % 32 == 0 for the q8 KV path (forward.rs)
            sliding_window_pattern: 2,
        }
    }

    fn config_json(&self) -> serde_json::Value {
        serde_json::json!({
            "architectures": ["Gemma3ForCausalLM"],
            "model_type": "gemma3_text",
            "hidden_size": self.hidden,
            "intermediate_size": self.inter,
            "vocab_size": self.vocab,
            "num_hidden_layers": self.layers,
            "num_attention_heads": self.n_heads,
            "num_key_value_heads": self.n_kv_heads,
            "head_dim": self.head_dim,
            "query_pre_attn_scalar": self.head_dim,
            "sliding_window": 64,
            "sliding_window_pattern": self.sliding_window_pattern,
            "rope_theta": 1_000_000.0,
            "rope_local_base_freq": 10_000.0,
            "hidden_activation": "gelu_pytorch_tanh",
            "rms_norm_eps": 1e-6,
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
            let sa = format!("{p}.self_attn");
            t.push(TensorSpec::f16(
                format!("{p}.input_layernorm.weight"),
                vec![h],
                Init::NormOnes,
            ));
            t.push(TensorSpec::f16(
                format!("{sa}.q_norm.weight"),
                vec![self.head_dim],
                Init::NormOnes,
            ));
            t.push(TensorSpec::f16(
                format!("{sa}.k_norm.weight"),
                vec![self.head_dim],
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
            t.push(TensorSpec::f16(
                format!("{p}.post_attention_layernorm.weight"),
                vec![h],
                Init::NormOnes,
            ));
            t.push(TensorSpec::f16(
                format!("{p}.pre_feedforward_layernorm.weight"),
                vec![h],
                Init::NormOnes,
            ));
            t.push(TensorSpec::f16(
                format!("{p}.post_feedforward_layernorm.weight"),
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

impl ToyModel for Gemma3Spec {
    // Tiny random-init gating fixture, declared arch-side. Ported verbatim from the
    // quantizer's old `Gemma3Tiny` preset so the emitted bytes stay identical (the
    // tiny-quant golden baselines depend on them). The quantizer owns the seeded RNG
    // + safetensors/tokenizer writing; this only describes shape + config.
    fn fixture(&self, _seed: u64) -> ToyFixture {
        let m = Gemma3Tiny::preset();
        ToyFixture {
            config_json: serde_json::to_string_pretty(&m.config_json())
                .expect("serialize gemma3 toy config"),
            tensors: m.manifest(),
        }
    }
}

static GEMMA3_SPEC: Gemma3Spec = Gemma3Spec;
register_arch!(GEMMA3_SPEC, Ingest, ToyModel);

#[cfg(test)]
mod tests {
    use super::*;
    use hipfire_arch_api::ArchRegistry;

    #[test]
    fn gemma3_spec_registers_ingest() {
        let reg = ArchRegistry::build();
        let a = reg.get(GEMMA3_ARCH_ID).expect("gemma3 spec registered");
        assert_eq!(a.family, "gemma3");
        let ing = a.caps.ingest.expect("Ingest declared");
        assert_eq!(
            ing.requires("model.embed_tokens.weight"),
            CapReq::RANDOM_ACCESS
        );
        assert!(
            ing.importance("model.layers.0.self_attn.q_proj.weight")
                > ing.importance("model.layers.0.mlp.up_proj.weight")
        );
        // The lean spec crate now also declares the offline ToyModel fixture.
        assert!(a.caps.toy_model.is_some());
    }

    #[test]
    fn gemma3_toy_fixture_is_tiny_and_gemma3() {
        let f = Gemma3Spec.fixture(0);
        assert!(!f.tensors.is_empty(), "fixture must emit tensors");
        // config is valid JSON declaring the gemma3_text family.
        assert!(f.config_json.contains("\"model_type\": \"gemma3_text\""));
    }
}
