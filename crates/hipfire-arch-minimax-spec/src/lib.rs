// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 hipfire contributors
// hipfire — see LICENSE and NOTICE in the project root.

//! Lean offline spec for the MiniMax-M2 MoE family (arch_id 10): identity + the
//! `Ingest` quant-policy (shared transformer prior). Deps only `hipfire-arch-api`.

use hipfire_arch_api::{
    default_importance, default_requires, register_arch, transformer_role, Arch, ArchId, CapReq,
    Ingest, TensorRole,
};

/// MiniMax-M2 family header id.
pub const MINIMAX_ARCH_ID: ArchId = ArchId(10);

/// Lean identity marker for the MiniMax-M2 offline spec.
pub struct MinimaxSpec;

impl Arch for MinimaxSpec {
    fn id(&self) -> ArchId {
        MINIMAX_ARCH_ID
    }
    fn family(&self) -> &'static str {
        "minimax"
    }
}

impl Ingest for MinimaxSpec {
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

static MINIMAX_SPEC: MinimaxSpec = MinimaxSpec;
register_arch!(MINIMAX_SPEC, Ingest);

#[cfg(test)]
mod tests {
    use super::*;
    use hipfire_arch_api::ArchRegistry;

    #[test]
    fn registers_ingest() {
        let reg = ArchRegistry::build();
        let a = reg.get(MINIMAX_ARCH_ID).expect("minimax spec registered");
        assert_eq!(a.family, "minimax");
        assert!(a.caps.ingest.is_some());
    }
}
