// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 hipfire contributors
// hipfire — see LICENSE and NOTICE in the project root.

//! Lean offline spec for the Krea2 diffusion denoiser (MMDiT, arch_id 17):
//! identity + the `Diffusion` modality marker (so routing tells it apart from
//! LLMs without a magic id) + the `Ingest` quant-policy (shared MMDiT role prior).
//! Deps only `hipfire-arch-api`.

use hipfire_arch_api::{
    default_importance, mmdit_role, register_arch, Arch, ArchId, CapReq, Diffusion, Ingest,
    TensorRole, ARCH_ID_KREA2,
};

/// Krea2 denoiser header id.
pub const KREA2_ARCH_ID: ArchId = ArchId(ARCH_ID_KREA2 as u16);

/// Lean identity marker for the Krea2 offline spec.
pub struct Krea2Spec;

impl Arch for Krea2Spec {
    fn id(&self) -> ArchId {
        KREA2_ARCH_ID
    }
    fn family(&self) -> &'static str {
        "krea2"
    }
}

impl Diffusion for Krea2Spec {
    fn denoiser_family(&self) -> &'static str {
        "krea2-mmdit"
    }
}

impl Ingest for Krea2Spec {
    fn role(&self, tensor: &str) -> TensorRole {
        mmdit_role(tensor)
    }
    fn importance(&self, tensor: &str) -> u8 {
        default_importance(mmdit_role(tensor))
    }
    /// Diffusion has no gather-indexed tables. Rank-4 conv weights that need a
    /// random-access (ungrouped) codec are steered by shape at encode time, not
    /// by name here — so the name-only requirement is `NONE`.
    fn requires(&self, _tensor: &str) -> CapReq {
        CapReq::NONE
    }
}

static KREA2_SPEC: Krea2Spec = Krea2Spec;
register_arch!(KREA2_SPEC, Diffusion, Ingest);

#[cfg(test)]
mod tests {
    use super::*;
    use hipfire_arch_api::ArchRegistry;

    #[test]
    fn registers_as_diffusion_arch() {
        let reg = ArchRegistry::build();
        assert!(reg.is_diffusion(KREA2_ARCH_ID));
        assert_eq!(reg.diffusion_family(KREA2_ARCH_ID), Some("krea2-mmdit"));
        assert!(reg.get(KREA2_ARCH_ID).unwrap().caps.ingest.is_some());
    }

    #[test]
    fn mmdit_role_prior_protects_and_compresses() {
        // Output projection + embedders + attention + modulation -> protect (255).
        assert_eq!(Krea2Spec.importance("final_layer.linear.weight"), 255);
        assert_eq!(Krea2Spec.importance("img_in.weight"), 255);
        assert_eq!(
            Krea2Spec.importance("transformer_blocks.3.attn.to_q.weight"),
            255
        );
        assert_eq!(
            Krea2Spec.importance("transformer_blocks.3.img_mod.linear.weight"),
            255
        );
        // Block feed-forward -> compress (128).
        assert_eq!(
            Krea2Spec.importance("transformer_blocks.3.img_mlp.net.0.proj.weight"),
            128
        );
        // Diffusion never needs random access.
        assert!(!Krea2Spec.requires("img_in.weight").random_access);
    }
}
