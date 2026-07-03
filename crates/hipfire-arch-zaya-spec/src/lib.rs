// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 hipfire contributors
// hipfire — see LICENSE and NOTICE in the project root.

//! Lean offline spec for the Zyphra ZAYA1 MoE family (arch_id 16): identity + the
//! `Ingest` quant-policy (shared transformer prior). Deps only `hipfire-arch-api`.

use hipfire_arch_api::{
    default_importance, default_requires, register_arch, transformer_role, Arch, ArchId, CapReq,
    Ingest, TensorRole,
};

/// ZAYA1 family header id.
pub const ZAYA_ARCH_ID: ArchId = ArchId(16);

/// Lean identity marker for the ZAYA1 offline spec.
pub struct ZayaSpec;

impl Arch for ZayaSpec {
    fn id(&self) -> ArchId {
        ZAYA_ARCH_ID
    }
    fn family(&self) -> &'static str {
        "zaya"
    }
}

impl Ingest for ZayaSpec {
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

static ZAYA_SPEC: ZayaSpec = ZayaSpec;
register_arch!(ZAYA_SPEC, Ingest);

#[cfg(test)]
mod tests {
    use super::*;
    use hipfire_arch_api::ArchRegistry;

    #[test]
    fn registers_ingest() {
        let reg = ArchRegistry::build();
        let a = reg.get(ZAYA_ARCH_ID).expect("zaya spec registered");
        assert_eq!(a.family, "zaya");
        assert!(a.caps.ingest.is_some());
    }
}
