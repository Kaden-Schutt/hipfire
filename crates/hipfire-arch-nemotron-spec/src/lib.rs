// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 hipfire contributors
// hipfire — see LICENSE and NOTICE in the project root.

//! Lean offline spec for the Mamba-2 hybrid families: Nemotron-H (arch_id 14) and
//! pure Mamba-2 (arch_id 15). Identity + the `Ingest` quant-policy (shared
//! transformer prior, which classifies the SSM `in_proj`/`out_proj`/`conv1d` mixer
//! tensors). Deps only `hipfire-arch-api`.

use hipfire_arch_api::{
    default_importance, default_precision_class, default_requires, register_arch, transformer_role,
    Arch, ArchId, CapReq, Ingest, Init, PrecisionClass, TensorRole, TensorSpec, ToyFixture,
    ToyModel,
};

/// Mamba-2 block tensors that corrupt SSM state when lossy: the mixer ingress
/// (`in_proj`, generating the gate/x/B/C/dt streams) and the residual writers
/// (`out_proj`/`down_proj`/`o_proj`). Pinned above a tight low-bit budget — this is the
/// model-definition the quantizer reads instead of the old
/// `is_nemotron_h_mq4_q8_protected` name-match (shared by Nemotron-H + pure Mamba-2).
fn is_state_critical(tensor: &str) -> bool {
    tensor.starts_with("backbone.layers.")
        && (tensor.ends_with(".mixer.in_proj.weight")
            || tensor.ends_with(".mixer.out_proj.weight")
            || tensor.ends_with(".mixer.down_proj.weight")
            || tensor.ends_with(".mixer.o_proj.weight"))
}

/// Shared `precision_class`: pin the state-critical mixer tensors, else role default.
fn state_aware_precision_class(role: TensorRole, tensor: &str) -> PrecisionClass {
    if is_state_critical(tensor) {
        PrecisionClass::Pinned
    } else {
        default_precision_class(role)
    }
}

/// Nemotron-H header id.
pub const NEMOTRON_H_ARCH_ID: ArchId = ArchId(14);
/// Pure Mamba-2 header id.
pub const MAMBA2_ARCH_ID: ArchId = ArchId(15);

/// Lean identity marker for the Nemotron-H offline spec.
pub struct NemotronHSpec;
impl Arch for NemotronHSpec {
    fn id(&self) -> ArchId {
        NEMOTRON_H_ARCH_ID
    }
    fn family(&self) -> &'static str {
        "nemotron-h"
    }
}
impl Ingest for NemotronHSpec {
    fn role(&self, tensor: &str) -> TensorRole {
        transformer_role(tensor)
    }
    fn importance(&self, tensor: &str) -> u8 {
        default_importance(self.role(tensor))
    }
    fn requires(&self, tensor: &str) -> CapReq {
        default_requires(self.role(tensor))
    }
    fn precision_class(&self, tensor: &str) -> PrecisionClass {
        state_aware_precision_class(self.role(tensor), tensor)
    }
}

/// Lean identity marker for the pure Mamba-2 offline spec.
pub struct Mamba2Spec;
impl Arch for Mamba2Spec {
    fn id(&self) -> ArchId {
        MAMBA2_ARCH_ID
    }
    fn family(&self) -> &'static str {
        "mamba2"
    }
}
impl Ingest for Mamba2Spec {
    fn role(&self, tensor: &str) -> TensorRole {
        transformer_role(tensor)
    }
    fn importance(&self, tensor: &str) -> u8 {
        default_importance(self.role(tensor))
    }
    fn requires(&self, tensor: &str) -> CapReq {
        default_requires(self.role(tensor))
    }
    fn precision_class(&self, tensor: &str) -> PrecisionClass {
        state_aware_precision_class(self.role(tensor), tensor)
    }
}

/// Tiny pure Mamba-2 (arch 15) config. Mirrors state-spaces tensor names:
/// `backbone.embedding.weight`, `backbone.layers.L.mixer.*`, `backbone.norm_f`.
/// Ported verbatim from the quantizer's old `fixture.rs` so the emitted bytes stay
/// identical (the tiny-quant golden baselines depend on them).
struct Mamba2Tiny {
    hidden: usize,
    vocab: usize,
    layers: usize,
    expand: usize,
    head_dim: usize,
    d_state: usize,
    ngroups: usize,
    conv_kernel: usize,
    chunk_size: usize,
}

impl Mamba2Tiny {
    fn preset() -> Self {
        Self {
            hidden: 256,
            vocab: 4096,
            layers: 2,
            expand: 2,
            head_dim: 64,
            d_state: 128,
            ngroups: 1,
            conv_kernel: 4,
            chunk_size: 64,
        }
    }

    fn d_inner(&self) -> usize {
        self.hidden * self.expand
    }

    fn num_heads(&self) -> usize {
        self.d_inner() / self.head_dim
    }

    fn conv_dim(&self) -> usize {
        self.d_inner() + 2 * self.ngroups * self.d_state
    }

    fn projection_size(&self) -> usize {
        self.d_inner() + self.conv_dim() + self.num_heads()
    }

    fn config_json(&self) -> serde_json::Value {
        serde_json::json!({
            "architectures": ["Mamba2ForCausalLM"],
            "d_model": self.hidden,
            "d_intermediate": 0,
            "n_layer": self.layers,
            "vocab_size": self.vocab,
            "ssm_cfg": {
                "layer": "Mamba2",
                "d_state": self.d_state,
                "d_conv": self.conv_kernel,
                "expand": self.expand,
                "headdim": self.head_dim,
                "ngroups": self.ngroups,
                "chunk_size": self.chunk_size,
            },
            "attn_layer_idx": [],
            "attn_cfg": {},
            "rms_norm": true,
            "residual_in_fp32": true,
            "fused_add_norm": true,
            "pad_vocab_size_multiple": 16,
            "tie_embeddings": true,
            "rms_norm_eps": 1e-5,
            "_comment": "hipfire tiny random-init gating fixture — not a real model",
        })
    }

    fn manifest(&self) -> Vec<TensorSpec> {
        let h = self.hidden;
        let d_inner = self.d_inner();
        let conv_dim = self.conv_dim();
        let projection_size = self.projection_size();
        let heads = self.num_heads();
        let mut t = Vec::new();
        t.push(TensorSpec::new(
            "backbone.embedding.weight",
            vec![self.vocab, h],
            Init::Uniform(0.05),
        ));
        t.push(TensorSpec::f16(
            "backbone.norm_f.weight",
            vec![h],
            Init::NormOnes,
        ));
        t.push(TensorSpec::new(
            "lm_head.weight",
            vec![self.vocab, h],
            Init::Uniform(0.05),
        ));
        for i in 0..self.layers {
            let p = format!("backbone.layers.{i}");
            let m = format!("{p}.mixer");
            t.push(TensorSpec::f16(
                format!("{p}.norm.weight"),
                vec![h],
                Init::NormOnes,
            ));
            t.push(TensorSpec::new(
                format!("{m}.in_proj.weight"),
                vec![projection_size, h],
                Init::Uniform(0.04),
            ));
            t.push(TensorSpec::f16(
                format!("{m}.conv1d.weight"),
                vec![conv_dim, 1, self.conv_kernel],
                Init::Uniform(0.03),
            ));
            t.push(TensorSpec::f16(
                format!("{m}.conv1d.bias"),
                vec![conv_dim],
                Init::Zeros,
            ));
            t.push(TensorSpec::f16(
                format!("{m}.A_log"),
                vec![heads],
                Init::ALog,
            ));
            t.push(TensorSpec::f16(
                format!("{m}.D"),
                vec![heads],
                Init::NormOnes,
            ));
            t.push(TensorSpec::f16(
                format!("{m}.dt_bias"),
                vec![heads],
                Init::Zeros,
            ));
            t.push(TensorSpec::f16(
                format!("{m}.norm.weight"),
                vec![d_inner],
                Init::NormOnes,
            ));
            t.push(TensorSpec::new(
                format!("{m}.out_proj.weight"),
                vec![h, d_inner],
                Init::Uniform(0.04),
            ));
        }
        t
    }
}

impl ToyModel for Mamba2Spec {
    // Tiny random-init gating fixture, declared arch-side. Ported verbatim from the
    // quantizer's old fixture so the emitted bytes stay identical (the tiny-quant
    // golden baselines depend on them). Only the pure Mamba-2 arch emits a fixture;
    // the Nemotron-H spec does not.
    fn fixture(&self, _seed: u64) -> ToyFixture {
        let m = Mamba2Tiny::preset();
        ToyFixture {
            config_json: serde_json::to_string_pretty(&m.config_json())
                .expect("serialize mamba2 toy config"),
            tensors: m.manifest(),
        }
    }
}

static NEMOTRON_H_SPEC: NemotronHSpec = NemotronHSpec;
static MAMBA2_SPEC: Mamba2Spec = Mamba2Spec;
register_arch!(NEMOTRON_H_SPEC, Ingest);
register_arch!(MAMBA2_SPEC, Ingest, ToyModel);

#[cfg(test)]
mod tests {
    use super::*;
    use hipfire_arch_api::ArchRegistry;

    #[test]
    fn registers_nemotron_and_mamba2() {
        let reg = ArchRegistry::build();
        assert_eq!(reg.get(NEMOTRON_H_ARCH_ID).unwrap().family, "nemotron-h");
        assert!(reg.get(NEMOTRON_H_ARCH_ID).unwrap().caps.ingest.is_some());
        assert_eq!(reg.get(MAMBA2_ARCH_ID).unwrap().family, "mamba2");
        assert!(reg.get(MAMBA2_ARCH_ID).unwrap().caps.ingest.is_some());
    }

    #[test]
    fn state_critical_mixer_tensors_are_pinned_faithfully() {
        // Exactly reproduces the old quantizer `is_nemotron_h_mq4_q8_protected`
        // truth table, now declared arch-side and shared by both Mamba-2 archs.
        let reg = ArchRegistry::build();
        for id in [NEMOTRON_H_ARCH_ID, MAMBA2_ARCH_ID] {
            let ing = reg.get(id).unwrap().caps.ingest.unwrap();
            for t in [
                "backbone.layers.0.mixer.in_proj.weight",
                "backbone.layers.0.mixer.out_proj.weight",
                "backbone.layers.1.mixer.down_proj.weight",
                "backbone.layers.12.mixer.o_proj.weight",
            ] {
                assert_eq!(ing.precision_class(t), PrecisionClass::Pinned, "{t}");
            }
            // up_proj is NOT protected (was `!is_nemotron_h_mq4_q8_protected`).
            assert!(
                ing.precision_class("backbone.layers.0.mixer.up_proj.weight")
                    < PrecisionClass::Pinned
            );
            // The router (structurally protected) is High, not Pinned — so the
            // low-bit pinned path won't over-reach it.
            assert!(
                ing.precision_class("backbone.layers.1.mixer.gate.weight") < PrecisionClass::Pinned
            );
        }
    }

    #[test]
    fn mamba2_toy_fixture_is_declared() {
        // Only the pure Mamba-2 arch emits the toy fixture (ported verbatim from the
        // quantizer). Nemotron-H declares no ToyModel.
        let f = Mamba2Spec.fixture(0);
        assert!(!f.tensors.is_empty(), "mamba2 fixture emits tensors");
        assert!(
            f.config_json.contains("Mamba2ForCausalLM"),
            "config declares the Mamba-2 model type: {}",
            f.config_json
        );

        let reg = ArchRegistry::build();
        assert!(reg.get(MAMBA2_ARCH_ID).unwrap().caps.toy_model.is_some());
        assert!(reg
            .get(NEMOTRON_H_ARCH_ID)
            .unwrap()
            .caps
            .toy_model
            .is_none());
    }
}
