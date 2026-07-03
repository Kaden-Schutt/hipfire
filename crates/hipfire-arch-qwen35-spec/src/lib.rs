// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 hipfire contributors
// hipfire — see LICENSE and NOTICE in the project root.

//! Lean offline spec for the Qwen3.5 family: dense (arch_id 5) and MoE (arch_id 6).
//! Identity + the `Ingest` quant-policy (shared transformer prior, which covers the
//! DeltaNet linear-attention + short-conv mixer and the MoE router/experts). Deps
//! only `hipfire-arch-api`.

use hipfire_arch_api::{
    default_importance, default_requires, register_arch, transformer_role, Arch, ArchId, CapReq,
    Ingest, TensorRole,
};

/// Qwen3.5 dense header id.
pub const QWEN35_ARCH_ID: ArchId = ArchId(5);
/// Qwen3.5 MoE header id.
pub const QWEN35_MOE_ARCH_ID: ArchId = ArchId(6);

/// The Qwen3.5 quant-policy — identical for the dense and MoE variants (the shared
/// `transformer_role` already distinguishes router/expert tensors).
fn qwen35_importance(tensor: &str) -> u8 {
    default_importance(transformer_role(tensor))
}
fn qwen35_requires(tensor: &str) -> CapReq {
    default_requires(transformer_role(tensor))
}

/// Lean identity marker for the Qwen3.5 dense offline spec.
pub struct Qwen35Spec;
impl Arch for Qwen35Spec {
    fn id(&self) -> ArchId {
        QWEN35_ARCH_ID
    }
    fn family(&self) -> &'static str {
        "qwen3.5"
    }
}
impl Ingest for Qwen35Spec {
    fn role(&self, tensor: &str) -> TensorRole {
        transformer_role(tensor)
    }
    fn importance(&self, tensor: &str) -> u8 {
        qwen35_importance(tensor)
    }
    fn requires(&self, tensor: &str) -> CapReq {
        qwen35_requires(tensor)
    }
}

/// Lean identity marker for the Qwen3.5 MoE offline spec.
pub struct Qwen35MoeSpec;
impl Arch for Qwen35MoeSpec {
    fn id(&self) -> ArchId {
        QWEN35_MOE_ARCH_ID
    }
    fn family(&self) -> &'static str {
        "qwen3.5-moe"
    }
}
impl Ingest for Qwen35MoeSpec {
    fn role(&self, tensor: &str) -> TensorRole {
        transformer_role(tensor)
    }
    fn importance(&self, tensor: &str) -> u8 {
        qwen35_importance(tensor)
    }
    fn requires(&self, tensor: &str) -> CapReq {
        qwen35_requires(tensor)
    }
}

static QWEN35_SPEC: Qwen35Spec = Qwen35Spec;
static QWEN35_MOE_SPEC: Qwen35MoeSpec = Qwen35MoeSpec;
register_arch!(QWEN35_SPEC, Ingest);
register_arch!(QWEN35_MOE_SPEC, Ingest);

#[cfg(test)]
mod tests {
    use super::*;
    use hipfire_arch_api::ArchRegistry;

    #[test]
    fn registers_both_dense_and_moe() {
        let reg = ArchRegistry::build();
        assert_eq!(reg.get(QWEN35_ARCH_ID).unwrap().family, "qwen3.5");
        assert!(reg.get(QWEN35_ARCH_ID).unwrap().caps.ingest.is_some());
        assert_eq!(reg.get(QWEN35_MOE_ARCH_ID).unwrap().family, "qwen3.5-moe");
        assert!(reg.get(QWEN35_MOE_ARCH_ID).unwrap().caps.ingest.is_some());
    }
}
