// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 hipfire contributors
// hipfire — see LICENSE and NOTICE in the project root.

//! Lean offline spec for the Llama family (arch_id 0): identity + the `Ingest`
//! quant-policy, and NOTHING that needs the GPU/runtime — so the offline quantizer
//! links this without pulling in the serving stack. The serving crate
//! (`hipfire-arch-llama`) declares runtime capabilities (`BatchedPrefill`, …) on the
//! same [`ArchId`]; the registry merges the two halves into one arch.
//!
//! The `importance` values below are a reasonable STRUCTURAL PRIOR, not tuned bit
//! assignments — exact per-tensor precision is the quantizer's job (calibrated
//! later against KLD). Anything the deployment cannot otherwise place falls back to
//! a safe high-precision (bf16) codec, keeping the model coherent in the meantime.

use hipfire_arch_api::{
    register_arch, Arch, ArchId, CapReq, Ingest, Init, TensorRole, TensorSpec, ToyFixture, ToyModel,
};

/// Llama family header id.
pub const LLAMA_ARCH_ID: ArchId = ArchId(0);

/// Lean identity marker for the Llama family's offline spec.
pub struct LlamaSpec;

impl Arch for LlamaSpec {
    fn id(&self) -> ArchId {
        LLAMA_ARCH_ID
    }
    fn family(&self) -> &'static str {
        "llama"
    }
}

impl Ingest for LlamaSpec {
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
            TensorRole::Norm
        } else {
            TensorRole::Other
        }
    }

    fn importance(&self, tensor: &str) -> u8 {
        // Structural prior only; the quantizer refines the actual bit assignment.
        // The "protected" tensors (embed/lm_head/attn/norm) sit in the top tier so
        // they keep high precision under the default budget — mirroring the current
        // coherent high-precision-attention policy. Finer tiers (e.g. 4-bit attn
        // with a learned rotation) are deferred to quantizer tuning. MLP bulk is
        // compressible.
        match self.role(tensor) {
            TensorRole::Embed | TensorRole::LmHead => 255, // gather-indexed, critical
            TensorRole::Norm => 255,                       // tiny + numerically sensitive
            TensorRole::AttnProj => 255,                   // error-sensitive → protect
            TensorRole::Mlp => 128,                        // the bulk of the weights
            _ => 160,                                      // safe-ish default
        }
    }

    fn requires(&self, tensor: &str) -> CapReq {
        match self.role(tensor) {
            // Embeddings / lm_head are gathered one row at a time → need random access.
            TensorRole::Embed | TensorRole::LmHead => CapReq::RANDOM_ACCESS,
            _ => CapReq::NONE,
        }
    }
}

impl ToyModel for LlamaSpec {
    // Tiny random-init gating fixture, declared arch-side. Ported verbatim from the
    // quantizer's old `emit_fixture` match arm so the emitted bytes stay identical
    // (the tiny-quant golden baselines depend on them). The quantizer owns the seeded
    // RNG + safetensors/tokenizer writing; this only describes shape + config.
    fn fixture(&self, _seed: u64) -> ToyFixture {
        let (h, inter, vocab, layers, n_heads, n_kv_heads, head_dim) = (
            256usize, 512usize, 4096usize, 2usize, 2usize, 1usize, 128usize,
        );
        let config = serde_json::json!({
            "architectures": ["LlamaForCausalLM"],
            "model_type": "llama",
            "hidden_size": h,
            "intermediate_size": inter,
            "vocab_size": vocab,
            "num_hidden_layers": layers,
            "num_attention_heads": n_heads,
            "num_key_value_heads": n_kv_heads,
            "head_dim": head_dim,
            "hidden_act": "silu",
            "rms_norm_eps": 1e-6,
            "rope_theta": 500_000.0,
            "max_position_embeddings": 4096,
            "tie_word_embeddings": false,
            "dtype": "bfloat16",
            "_comment": "hipfire tiny random-init gating fixture — not a real model",
        });
        let q_dim = n_heads * head_dim;
        let kv_dim = n_kv_heads * head_dim;
        let mut tensors = vec![
            // Untied: embed + separate lm_head.
            TensorSpec::new(
                "model.embed_tokens.weight",
                vec![vocab, h],
                Init::Uniform(0.05),
            ),
            TensorSpec::f16("model.norm.weight", vec![h], Init::NormOnes),
            TensorSpec::new("lm_head.weight", vec![vocab, h], Init::Uniform(0.05)),
        ];
        for i in 0..layers {
            let p = format!("model.layers.{i}");
            let sa = format!("{p}.self_attn");
            tensors.push(TensorSpec::f16(
                format!("{p}.input_layernorm.weight"),
                vec![h],
                Init::NormOnes,
            ));
            // No bias, no q_norm/k_norm — the LLaMA loader rejects bias tensors.
            tensors.push(TensorSpec::new(
                format!("{sa}.q_proj.weight"),
                vec![q_dim, h],
                Init::Uniform(0.05),
            ));
            tensors.push(TensorSpec::new(
                format!("{sa}.k_proj.weight"),
                vec![kv_dim, h],
                Init::Uniform(0.05),
            ));
            tensors.push(TensorSpec::new(
                format!("{sa}.v_proj.weight"),
                vec![kv_dim, h],
                Init::Uniform(0.05),
            ));
            tensors.push(TensorSpec::new(
                format!("{sa}.o_proj.weight"),
                vec![h, q_dim],
                Init::Uniform(0.05),
            ));
            tensors.push(TensorSpec::f16(
                format!("{p}.post_attention_layernorm.weight"),
                vec![h],
                Init::NormOnes,
            ));
            tensors.push(TensorSpec::new(
                format!("{p}.mlp.gate_proj.weight"),
                vec![inter, h],
                Init::Uniform(0.05),
            ));
            tensors.push(TensorSpec::new(
                format!("{p}.mlp.up_proj.weight"),
                vec![inter, h],
                Init::Uniform(0.05),
            ));
            tensors.push(TensorSpec::new(
                format!("{p}.mlp.down_proj.weight"),
                vec![h, inter],
                Init::Uniform(0.05),
            ));
        }
        ToyFixture {
            config_json: serde_json::to_string_pretty(&config).expect("serialize llama toy config"),
            tensors,
        }
    }
}

static LLAMA_SPEC: LlamaSpec = LlamaSpec;
register_arch!(LLAMA_SPEC, Ingest, ToyModel);

#[cfg(test)]
mod tests {
    use super::*;
    use hipfire_arch_api::ArchRegistry;

    #[test]
    fn llama_spec_registers_ingest_only() {
        let reg = ArchRegistry::build();
        let a = reg.get(LLAMA_ARCH_ID).expect("llama spec registered");
        assert_eq!(a.family, "llama");

        let ing = a.caps.ingest.expect("Ingest declared");
        // Gather-indexed tables demand random access and outrank the bulk MLP.
        // (Values are structural priors, not tuned bit assignments.)
        assert_eq!(
            ing.requires("model.embed_tokens.weight"),
            CapReq::RANDOM_ACCESS
        );
        assert!(
            ing.importance("model.embed_tokens.weight")
                > ing.importance("model.layers.0.mlp.up_proj.weight")
        );
        // The lean spec crate on its own carries no serving capability…
        assert!(a.caps.batched_prefill.is_none());
        // …but it does now declare the offline ToyModel fixture.
        assert!(a.caps.toy_model.is_some());
    }

    #[test]
    fn toy_fixture_is_bias_free_and_tiny() {
        // Moved from the quantizer's fixture.rs, now co-located with the manifest.
        let f = LlamaSpec.fixture(0);
        let has = |suf: &str| f.tensors.iter().any(|s| s.name.ends_with(suf));
        assert!(has("lm_head.weight"), "untied lm_head");
        assert!(has("mlp.gate_proj.weight"), "dense SwiGLU");
        // The LLaMA loader rejects bias + qk-norm tensors — must emit none.
        assert!(
            !f.tensors.iter().any(|s| s.name.ends_with(".bias")),
            "no biases"
        );
        assert!(!f
            .tensors
            .iter()
            .any(|s| s.name.contains("q_norm") || s.name.contains("k_norm")));
        let n_params: usize = f
            .tensors
            .iter()
            .map(|s| s.shape.iter().product::<usize>())
            .sum();
        assert!(n_params < 10_000_000, "llama fixture must stay <10M params");
        // config is valid JSON declaring the llama family.
        assert!(f.config_json.contains("\"model_type\": \"llama\""));
    }
}
