// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 hipfire contributors
// hipfire — see LICENSE and NOTICE in the project root.

//! Lean offline spec for the Qwen2/Qwen3 dense family (arch_id 1): identity + the
//! `Ingest` quant-policy (shared transformer prior). Deps only `hipfire-arch-api`.

use hipfire_arch_api::{
    default_importance, default_requires, register_arch, transformer_role, Arch, ArchId, CapReq,
    Ingest, TensorRole,
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

static QWEN2_SPEC: Qwen2Spec = Qwen2Spec;
register_arch!(QWEN2_SPEC, Ingest);

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
}
