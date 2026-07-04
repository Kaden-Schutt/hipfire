// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 hipfire contributors
// hipfire — see LICENSE and NOTICE in the project root.

//! Lean offline spec for the Mamba-2 hybrid families: Nemotron-H (arch_id 14) and
//! pure Mamba-2 (arch_id 15). Identity + the `Ingest` quant-policy (shared
//! transformer prior, which classifies the SSM `in_proj`/`out_proj`/`conv1d` mixer
//! tensors). Deps only `hipfire-arch-api`.

use hipfire_arch_api::{
    default_importance, default_precision_class, default_requires, register_arch, transformer_role,
    Arch, ArchId, CapReq, Ingest, PrecisionClass, TensorRole,
};

/// Mamba-2 block tensors that corrupt SSM state when lossy: the mixer ingress
/// (`in_proj`, generating the gate/x/B/C/dt streams) and the residual writers
/// (`out_proj`/`down_proj`/`o_proj`). Pinned above a tight low-bit budget — this is the
/// model-definition the quantizer reads instead of the old
/// `is_nemotron_h_mq4_q8_protected` name-match (shared by Nemotron-H + pure Mamba-2).
fn is_state_critical(tensor: &str) -> bool {
    tensor.starts_with("backbone.layers.")
        && (tensor.ends_with(".mixer.in_proj.weight")
            || tensor.ends_with(".mixer.out_proj.weight")
            || tensor.ends_with(".mixer.down_proj.weight")
            || tensor.ends_with(".mixer.o_proj.weight"))
}

/// Shared `precision_class`: pin the state-critical mixer tensors, else role default.
fn state_aware_precision_class(role: TensorRole, tensor: &str) -> PrecisionClass {
    if is_state_critical(tensor) {
        PrecisionClass::Pinned
    } else {
        default_precision_class(role)
    }
}

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
    fn precision_class(&self, tensor: &str) -> PrecisionClass {
        state_aware_precision_class(self.role(tensor), tensor)
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
    fn precision_class(&self, tensor: &str) -> PrecisionClass {
        state_aware_precision_class(self.role(tensor), tensor)
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

    #[test]
    fn state_critical_mixer_tensors_are_pinned_faithfully() {
        // Exactly reproduces the old quantizer `is_nemotron_h_mq4_q8_protected`
        // truth table, now declared arch-side and shared by both Mamba-2 archs.
        let reg = ArchRegistry::build();
        for id in [NEMOTRON_H_ARCH_ID, MAMBA2_ARCH_ID] {
            let ing = reg.get(id).unwrap().caps.ingest.unwrap();
            for t in [
                "backbone.layers.0.mixer.in_proj.weight",
                "backbone.layers.0.mixer.out_proj.weight",
                "backbone.layers.1.mixer.down_proj.weight",
                "backbone.layers.12.mixer.o_proj.weight",
            ] {
                assert_eq!(ing.precision_class(t), PrecisionClass::Pinned, "{t}");
            }
            // up_proj is NOT protected (was `!is_nemotron_h_mq4_q8_protected`).
            assert!(
                ing.precision_class("backbone.layers.0.mixer.up_proj.weight")
                    < PrecisionClass::Pinned
            );
            // The router (structurally protected) is High, not Pinned — so the
            // low-bit pinned path won't over-reach it.
            assert!(
                ing.precision_class("backbone.layers.1.mixer.gate.weight") < PrecisionClass::Pinned
            );
        }
    }
}
