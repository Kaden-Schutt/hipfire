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

use hipfire_arch_api::{register_arch, Arch, ArchId, CapReq, Ingest, TensorRole};

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

static LLAMA_SPEC: LlamaSpec = LlamaSpec;
register_arch!(LLAMA_SPEC, Ingest);

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
        // The lean spec crate on its own carries no serving capability.
        assert!(a.caps.batched_prefill.is_none());
    }
}
