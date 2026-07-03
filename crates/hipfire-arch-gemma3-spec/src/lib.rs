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

use hipfire_arch_api::{register_arch, Arch, ArchId, CapReq, Ingest, TensorRole};

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

static GEMMA3_SPEC: Gemma3Spec = Gemma3Spec;
register_arch!(GEMMA3_SPEC, Ingest);

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
    }
}
