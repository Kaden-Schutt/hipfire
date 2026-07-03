// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 hipfire contributors
// hipfire — see LICENSE and NOTICE in the project root.

//! Lean offline spec for the LFM2.5 family (arch_id 11; dense + MoE variants):
//! identity + the `Ingest` quant-policy (shared transformer prior, which covers the
//! short-conv mixer). Deps only `hipfire-arch-api`.

use hipfire_arch_api::{
    default_importance, default_requires, register_arch, transformer_role, Arch, ArchId, CapReq,
    Ingest, TensorRole,
};

/// LFM2.5 family header id.
pub const LFM2_ARCH_ID: ArchId = ArchId(11);

/// Lean identity marker for the LFM2.5 offline spec.
pub struct Lfm2Spec;

impl Arch for Lfm2Spec {
    fn id(&self) -> ArchId {
        LFM2_ARCH_ID
    }
    fn family(&self) -> &'static str {
        "lfm2"
    }
}

impl Ingest for Lfm2Spec {
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

static LFM2_SPEC: Lfm2Spec = Lfm2Spec;
register_arch!(LFM2_SPEC, Ingest);

#[cfg(test)]
mod tests {
    use super::*;
    use hipfire_arch_api::ArchRegistry;

    #[test]
    fn registers_ingest() {
        let reg = ArchRegistry::build();
        let a = reg.get(LFM2_ARCH_ID).expect("lfm2 spec registered");
        assert_eq!(a.family, "lfm2");
        assert!(a.caps.ingest.is_some());
    }
}
