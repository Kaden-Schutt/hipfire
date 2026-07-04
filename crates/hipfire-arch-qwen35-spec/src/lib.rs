// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 hipfire contributors
// hipfire — see LICENSE and NOTICE in the project root.

//! Lean offline spec for the Qwen3.5 family: dense (arch_id 5) and MoE (arch_id 6).
//! Identity + the `Ingest` quant-policy (shared transformer prior, which covers the
//! DeltaNet linear-attention + short-conv mixer and the MoE router/experts). Deps
//! only `hipfire-arch-api`.

use hipfire_arch_api::{
    default_importance, default_requires, register_arch, transformer_role, Arch, ArchId, CapReq,
    Ingest, Init, TensorRole, TensorSpec, ToyFixture, ToyModel,
};

/// Qwen3.5 dense header id.
pub const QWEN35_ARCH_ID: ArchId = ArchId(5);
/// Qwen3.5 MoE header id.
pub const QWEN35_MOE_ARCH_ID: ArchId = ArchId(6);

/// The Qwen3.5 quant-policy — identical for the dense and MoE variants (the shared
/// `transformer_role` already distinguishes router/expert tensors).
fn qwen35_importance(tensor: &str) -> u8 {
    default_importance(transformer_role(tensor))
}
fn qwen35_requires(tensor: &str) -> CapReq {
    default_requires(transformer_role(tensor))
}

/// Lean identity marker for the Qwen3.5 dense offline spec.
pub struct Qwen35Spec;
impl Arch for Qwen35Spec {
    fn id(&self) -> ArchId {
        QWEN35_ARCH_ID
    }
    fn family(&self) -> &'static str {
        "qwen3.5"
    }
}
impl Ingest for Qwen35Spec {
    fn role(&self, tensor: &str) -> TensorRole {
        transformer_role(tensor)
    }
    fn importance(&self, tensor: &str) -> u8 {
        qwen35_importance(tensor)
    }
    fn requires(&self, tensor: &str) -> CapReq {
        qwen35_requires(tensor)
    }
}

/// Lean identity marker for the Qwen3.5 MoE offline spec.
pub struct Qwen35MoeSpec;
impl Arch for Qwen35MoeSpec {
    fn id(&self) -> ArchId {
        QWEN35_MOE_ARCH_ID
    }
    fn family(&self) -> &'static str {
        "qwen3.5-moe"
    }
}
impl Ingest for Qwen35MoeSpec {
    fn role(&self, tensor: &str) -> TensorRole {
        transformer_role(tensor)
    }
    fn importance(&self, tensor: &str) -> u8 {
        qwen35_importance(tensor)
    }
    fn requires(&self, tensor: &str) -> CapReq {
        qwen35_requires(tensor)
    }
}

/// Tiny Qwen3.5 (arch 5) dense text config. Mirrors the real text_config
/// fields the ingest/arch-detect path reads, at fixture dims.
///
/// Ported verbatim from the quantizer's old `fixture.rs` so the emitted bytes
/// stay identical (the tiny-quant golden baselines depend on them). The
/// quantizer owns the seeded RNG + safetensors/tokenizer writing; this only
/// describes shape + config.
struct Qwen35Tiny {
    hidden: usize,
    inter: usize,
    vocab: usize,
    layers: usize,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    full_attn_interval: usize,
    // linear-attn (DeltaNet)
    l_key_heads: usize,
    l_key_head_dim: usize,
    l_val_heads: usize,
    l_val_head_dim: usize,
    conv_kernel: usize,
    // MoE (arch 6). `experts == 0` ⇒ dense (arch 5).
    experts: usize,
    experts_per_tok: usize,
    moe_inter: usize,
    shared_inter: usize,
}

impl Qwen35Tiny {
    /// ~3.9M params: 4 layers (3 linear-attn + 1 full-attn), tiny vocab.
    /// head_dim is pinned to 128 — the gated DeltaNet kernels are specialized
    /// for HD=128 (and full-attn supports it), so smaller HDs hard-error.
    fn preset() -> Self {
        Self {
            hidden: 256,
            inter: 512,
            vocab: 4096,
            layers: 4,
            n_heads: 2,
            n_kv_heads: 1,
            head_dim: 128,
            full_attn_interval: 4,
            l_key_heads: 2,
            l_key_head_dim: 128,
            l_val_heads: 2,
            l_val_head_dim: 128,
            conv_kernel: 4,
            experts: 0,
            experts_per_tok: 0,
            moe_inter: 0,
            shared_inter: 0,
        }
    }

    /// ~6M params: arch-6 MoE. Same hybrid attention as the dense preset, but
    /// every layer's FFN is MoE (8 experts top-2 + an always-on shared expert),
    /// matching the A3B layout (all layers MoE; attention type still varies).
    fn moe_preset() -> Self {
        Self {
            experts: 8,
            experts_per_tok: 2,
            moe_inter: 128,
            shared_inter: 128,
            ..Self::preset()
        }
    }

    fn is_moe(&self) -> bool {
        self.experts > 0
    }

    /// `full_attention` every `full_attn_interval`-th layer (positions
    /// interval-1, 2*interval-1, …), else `linear_attention` — matches the
    /// real checkpoint's layer_types pattern.
    fn layer_types(&self) -> Vec<&'static str> {
        (0..self.layers)
            .map(|i| {
                if (i + 1) % self.full_attn_interval == 0 {
                    "full_attention"
                } else {
                    "linear_attention"
                }
            })
            .collect()
    }

    fn config_json(&self) -> serde_json::Value {
        let mut c = serde_json::json!({
            "architectures": ["Qwen3_5ForCausalLM"],
            "model_type": "qwen3_5_text",
            "hidden_size": self.hidden,
            "intermediate_size": self.inter,
            "vocab_size": self.vocab,
            "num_hidden_layers": self.layers,
            "num_attention_heads": self.n_heads,
            "num_key_value_heads": self.n_kv_heads,
            "head_dim": self.head_dim,
            "attn_output_gate": true,
            "full_attention_interval": self.full_attn_interval,
            "layer_types": self.layer_types(),
            "linear_num_key_heads": self.l_key_heads,
            "linear_key_head_dim": self.l_key_head_dim,
            "linear_num_value_heads": self.l_val_heads,
            "linear_value_head_dim": self.l_val_head_dim,
            "linear_conv_kernel_dim": self.conv_kernel,
            "hidden_act": "silu",
            "rms_norm_eps": 1e-6,
            "max_position_embeddings": 4096,
            "tie_word_embeddings": true,
            "dtype": "bfloat16",
            "_comment": "hipfire tiny random-init gating fixture — not a real model",
        });
        if self.is_moe() {
            let o = c.as_object_mut().unwrap();
            o.insert("model_type".into(), "qwen3_5_moe_text".into());
            o.insert("num_experts".into(), self.experts.into());
            o.insert("num_experts_per_tok".into(), self.experts_per_tok.into());
            o.insert("moe_intermediate_size".into(), self.moe_inter.into());
            o.insert(
                "shared_expert_intermediate_size".into(),
                self.shared_inter.into(),
            );
            o.insert("norm_topk_prob".into(), true.into());
            o.insert("decoder_sparse_step".into(), 1.into());
            o.insert("mlp_only_layers".into(), serde_json::json!([]));
        }
        c
    }

    fn manifest(&self) -> Vec<TensorSpec> {
        let h = self.hidden;
        let mut t = Vec::new();
        // Globals (tie_word_embeddings ⇒ no separate lm_head).
        t.push(TensorSpec::new(
            "model.embed_tokens.weight",
            vec![self.vocab, h],
            Init::Uniform(0.05),
        ));
        t.push(TensorSpec::new(
            "model.norm.weight",
            vec![h],
            Init::NormOnes,
        ));

        let qkv =
            self.l_key_heads * self.l_key_head_dim * 2 + self.l_val_heads * self.l_val_head_dim;
        let v_dim = self.l_val_heads * self.l_val_head_dim;
        let attn_q = self.n_heads * self.head_dim * 2; // attn_output_gate ⇒ 2× wide
        let kv_dim = self.n_kv_heads * self.head_dim;
        let o_in = self.n_heads * self.head_dim;

        for (i, kind) in self.layer_types().into_iter().enumerate() {
            let p = format!("model.layers.{i}");
            t.push(TensorSpec::new(
                format!("{p}.input_layernorm.weight"),
                vec![h],
                Init::NormOnes,
            ));
            t.push(TensorSpec::new(
                format!("{p}.post_attention_layernorm.weight"),
                vec![h],
                Init::NormOnes,
            ));
            if self.is_moe() {
                // MoE FFN: router + stacked-3D routed experts + always-on shared expert.
                let mi = self.moe_inter;
                let si = self.shared_inter;
                t.push(TensorSpec::new(
                    format!("{p}.mlp.gate.weight"),
                    vec![self.experts, h],
                    Init::Uniform(0.05),
                ));
                t.push(TensorSpec::new(
                    format!("{p}.mlp.experts.gate_up_proj"),
                    vec![self.experts, 2 * mi, h],
                    Init::Uniform(0.05),
                ));
                t.push(TensorSpec::new(
                    format!("{p}.mlp.experts.down_proj"),
                    vec![self.experts, h, mi],
                    Init::Uniform(0.05),
                ));
                t.push(TensorSpec::new(
                    format!("{p}.mlp.shared_expert.gate_proj.weight"),
                    vec![si, h],
                    Init::Uniform(0.05),
                ));
                t.push(TensorSpec::new(
                    format!("{p}.mlp.shared_expert.up_proj.weight"),
                    vec![si, h],
                    Init::Uniform(0.05),
                ));
                t.push(TensorSpec::new(
                    format!("{p}.mlp.shared_expert.down_proj.weight"),
                    vec![h, si],
                    Init::Uniform(0.05),
                ));
                t.push(TensorSpec::new(
                    format!("{p}.mlp.shared_expert_gate.weight"),
                    vec![1, h],
                    Init::Uniform(0.05),
                ));
            } else {
                // Dense MLP (SwiGLU).
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

            if kind == "linear_attention" {
                let la = format!("{p}.linear_attn");
                t.push(TensorSpec::new(
                    format!("{la}.in_proj_qkv.weight"),
                    vec![qkv, h],
                    Init::Uniform(0.05),
                ));
                t.push(TensorSpec::new(
                    format!("{la}.in_proj_z.weight"),
                    vec![v_dim, h],
                    Init::Uniform(0.05),
                ));
                t.push(TensorSpec::new(
                    format!("{la}.in_proj_a.weight"),
                    vec![self.l_val_heads, h],
                    Init::Uniform(0.05),
                ));
                t.push(TensorSpec::new(
                    format!("{la}.in_proj_b.weight"),
                    vec![self.l_val_heads, h],
                    Init::Uniform(0.05),
                ));
                t.push(TensorSpec::new(
                    format!("{la}.A_log"),
                    vec![self.l_val_heads],
                    Init::ALog,
                ));
                t.push(TensorSpec::new(
                    format!("{la}.dt_bias"),
                    vec![self.l_val_heads],
                    Init::Zeros,
                ));
                t.push(TensorSpec::new(
                    format!("{la}.conv1d.weight"),
                    vec![qkv, 1, self.conv_kernel],
                    Init::Uniform(0.1),
                ));
                t.push(TensorSpec::new(
                    format!("{la}.norm.weight"),
                    vec![self.l_val_head_dim],
                    Init::NormOnes,
                ));
                t.push(TensorSpec::new(
                    format!("{la}.out_proj.weight"),
                    vec![h, v_dim],
                    Init::Uniform(0.05),
                ));
            } else {
                let sa = format!("{p}.self_attn");
                t.push(TensorSpec::new(
                    format!("{sa}.q_proj.weight"),
                    vec![attn_q, h],
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
                    vec![h, o_in],
                    Init::Uniform(0.05),
                ));
                t.push(TensorSpec::new(
                    format!("{sa}.q_norm.weight"),
                    vec![self.head_dim],
                    Init::NormOnes,
                ));
                t.push(TensorSpec::new(
                    format!("{sa}.k_norm.weight"),
                    vec![self.head_dim],
                    Init::NormOnes,
                ));
            }
        }
        t
    }
}

impl ToyModel for Qwen35Spec {
    fn fixture(&self, _seed: u64) -> ToyFixture {
        let m = Qwen35Tiny::preset();
        ToyFixture {
            config_json: serde_json::to_string_pretty(&m.config_json())
                .expect("serialize qwen3.5 dense toy config"),
            tensors: m.manifest(),
        }
    }
}

impl ToyModel for Qwen35MoeSpec {
    fn fixture(&self, _seed: u64) -> ToyFixture {
        let m = Qwen35Tiny::moe_preset();
        ToyFixture {
            config_json: serde_json::to_string_pretty(&m.config_json())
                .expect("serialize qwen3.5 moe toy config"),
            tensors: m.manifest(),
        }
    }
}

static QWEN35_SPEC: Qwen35Spec = Qwen35Spec;
static QWEN35_MOE_SPEC: Qwen35MoeSpec = Qwen35MoeSpec;
register_arch!(QWEN35_SPEC, Ingest, ToyModel);
register_arch!(QWEN35_MOE_SPEC, Ingest, ToyModel);

#[cfg(test)]
mod tests {
    use super::*;
    use hipfire_arch_api::ArchRegistry;

    #[test]
    fn registers_both_dense_and_moe() {
        let reg = ArchRegistry::build();
        assert_eq!(reg.get(QWEN35_ARCH_ID).unwrap().family, "qwen3.5");
        assert!(reg.get(QWEN35_ARCH_ID).unwrap().caps.ingest.is_some());
        assert_eq!(reg.get(QWEN35_MOE_ARCH_ID).unwrap().family, "qwen3.5-moe");
        assert!(reg.get(QWEN35_MOE_ARCH_ID).unwrap().caps.ingest.is_some());
    }

    #[test]
    fn dense_toy_fixture_populated() {
        let f = Qwen35Spec.fixture(0);
        assert!(!f.tensors.is_empty(), "dense fixture must emit tensors");
        assert!(f.config_json.contains("\"model_type\": \"qwen3_5_text\""));
    }

    #[test]
    fn moe_toy_fixture_populated() {
        let f = Qwen35MoeSpec.fixture(0);
        assert!(!f.tensors.is_empty(), "moe fixture must emit tensors");
        assert!(f
            .config_json
            .contains("\"model_type\": \"qwen3_5_moe_text\""));
        // Routed-expert tensors distinguish the MoE manifest from the dense one.
        assert!(f
            .tensors
            .iter()
            .any(|s| s.name.ends_with(".mlp.experts.gate_up_proj")));
    }
}
