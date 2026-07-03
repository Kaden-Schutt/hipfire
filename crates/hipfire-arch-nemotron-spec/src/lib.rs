// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 hipfire contributors
// hipfire — see LICENSE and NOTICE in the project root.

//! Lean offline spec for the Mamba-2 hybrid families: Nemotron-H (arch_id 14) and
//! pure Mamba-2 (arch_id 15). Identity + the `Ingest` quant-policy (shared
//! transformer prior, which classifies the SSM `in_proj`/`out_proj`/`conv1d` mixer
//! tensors). Deps only `hipfire-arch-api`.

use hipfire_arch_api::{
    default_importance, default_requires, register_arch, transformer_role, Arch, ArchId, CapReq,
    Ingest, TensorRole,
};

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
}

static NEMOTRON_H_SPEC: NemotronHSpec = NemotronHSpec;
static MAMBA2_SPEC: Mamba2Spec = Mamba2Spec;
register_arch!(NEMOTRON_H_SPEC, Ingest);
register_arch!(MAMBA2_SPEC, Ingest);

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
}
