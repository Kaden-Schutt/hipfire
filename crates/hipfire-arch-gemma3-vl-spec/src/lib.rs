// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 hipfire contributors
// hipfire — see LICENSE and NOTICE in the project root.

//! Lean offline spec for the Gemma-3 multimodal family (arch_id 13): identity + the
//! `Ingest` quant-policy (shared transformer prior; the SigLIP vision tensors fall
//! through to the generic roles). Deps only `hipfire-arch-api`.

use hipfire_arch_api::{
    default_importance, default_requires, register_arch, transformer_role, Arch, ArchId, CapReq,
    Ingest, TensorRole,
};

/// Gemma-3 multimodal family header id.
pub const GEMMA3_VL_ARCH_ID: ArchId = ArchId(13);

/// Lean identity marker for the Gemma-3 multimodal offline spec.
pub struct Gemma3VlSpec;

impl Arch for Gemma3VlSpec {
    fn id(&self) -> ArchId {
        GEMMA3_VL_ARCH_ID
    }
    fn family(&self) -> &'static str {
        "gemma3-vl"
    }
}

impl Ingest for Gemma3VlSpec {
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

static GEMMA3_VL_SPEC: Gemma3VlSpec = Gemma3VlSpec;
register_arch!(GEMMA3_VL_SPEC, Ingest);

#[cfg(test)]
mod tests {
    use super::*;
    use hipfire_arch_api::ArchRegistry;

    #[test]
    fn registers_ingest() {
        let reg = ArchRegistry::build();
        let a = reg
            .get(GEMMA3_VL_ARCH_ID)
            .expect("gemma3-vl spec registered");
        assert_eq!(a.family, "gemma3-vl");
        assert!(a.caps.ingest.is_some());
    }
}
