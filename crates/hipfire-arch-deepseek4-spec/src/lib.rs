// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 hipfire contributors
// hipfire — see LICENSE and NOTICE in the project root.

//! Lean offline spec for the DeepSeek-V4 family (arch_id 9): identity + the
//! `Ingest` quant-policy. Deps only `hipfire-arch-api`.
//!
//! Uses the shared transformer prior plus a family override: the MLA compressor and
//! indexer projections are numerically critical (they generate the compressed-KV
//! and index streams), so they sit at max importance — the arch-neutral replacement
//! for the old `is_deepseek4_keep_f16` name-match. No format is named here; the
//! deployment maps max importance to its highest-precision codec.

use hipfire_arch_api::{
    default_importance, default_requires, register_arch, transformer_role, Arch, ArchId, CapReq,
    Ingest, TensorRole,
};

/// DeepSeek-V4 family header id.
pub const DEEPSEEK4_ARCH_ID: ArchId = ArchId(9);

/// Lean identity marker for the DeepSeek-V4 offline spec.
pub struct Deepseek4Spec;

impl Deepseek4Spec {
    /// MLA compressor / indexer projections — precision-critical stream generators.
    fn is_critical_stream(name: &str) -> bool {
        name.ends_with(".compressor.wkv.weight")
            || name.ends_with(".compressor.wgate.weight")
            || name.ends_with(".indexer.wq_b.weight")
            || name.ends_with(".indexer.weights_proj.weight")
    }
}

impl Arch for Deepseek4Spec {
    fn id(&self) -> ArchId {
        DEEPSEEK4_ARCH_ID
    }
    fn family(&self) -> &'static str {
        "deepseek4"
    }
}

impl Ingest for Deepseek4Spec {
    fn role(&self, tensor: &str) -> TensorRole {
        transformer_role(tensor)
    }
    fn importance(&self, tensor: &str) -> u8 {
        if Self::is_critical_stream(tensor) {
            255
        } else {
            default_importance(self.role(tensor))
        }
    }
    fn requires(&self, tensor: &str) -> CapReq {
        default_requires(self.role(tensor))
    }
}

static DEEPSEEK4_SPEC: Deepseek4Spec = Deepseek4Spec;
register_arch!(DEEPSEEK4_SPEC, Ingest);

#[cfg(test)]
mod tests {
    use super::*;
    use hipfire_arch_api::ArchRegistry;

    #[test]
    fn registers_ingest_and_protects_mla_streams() {
        let reg = ArchRegistry::build();
        let a = reg
            .get(DEEPSEEK4_ARCH_ID)
            .expect("deepseek4 spec registered");
        assert_eq!(a.family, "deepseek4");
        let ing = a.caps.ingest.expect("Ingest declared");
        // The MLA compressor/indexer are max-importance (was is_deepseek4_keep_f16).
        assert_eq!(
            ing.importance("model.layers.0.self_attn.compressor.wkv.weight"),
            255
        );
    }
}
