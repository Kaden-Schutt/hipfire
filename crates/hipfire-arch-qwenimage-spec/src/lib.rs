// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 hipfire contributors
// hipfire — see LICENSE and NOTICE in the project root.

//! Lean offline spec for the Qwen-Image diffusion denoiser (MMDiT, arch_id 18):
//! identity + the `Diffusion` modality marker + the `Ingest` quant-policy (shared
//! MMDiT role prior). Deps only `hipfire-arch-api`.

use hipfire_arch_api::{
    default_importance, mmdit_role, register_arch, Arch, ArchId, CapReq, Diffusion, Ingest,
    TensorRole, ARCH_ID_QWEN_IMAGE,
};

/// Qwen-Image denoiser header id.
pub const QWEN_IMAGE_ARCH_ID: ArchId = ArchId(ARCH_ID_QWEN_IMAGE as u16);

/// Lean identity marker for the Qwen-Image offline spec.
pub struct QwenImageSpec;

impl Arch for QwenImageSpec {
    fn id(&self) -> ArchId {
        QWEN_IMAGE_ARCH_ID
    }
    fn family(&self) -> &'static str {
        "qwen-image"
    }
}

impl Diffusion for QwenImageSpec {
    fn denoiser_family(&self) -> &'static str {
        "qwen-image-mmdit"
    }
}

impl Ingest for QwenImageSpec {
    fn role(&self, tensor: &str) -> TensorRole {
        mmdit_role(tensor)
    }
    fn importance(&self, tensor: &str) -> u8 {
        default_importance(mmdit_role(tensor))
    }
    /// Diffusion has no gather-indexed tables; conv rank-4 weights are steered by
    /// shape at encode time, so the name-only requirement is `NONE`.
    fn requires(&self, _tensor: &str) -> CapReq {
        CapReq::NONE
    }
}

static QWEN_IMAGE_SPEC: QwenImageSpec = QwenImageSpec;
register_arch!(QWEN_IMAGE_SPEC, Diffusion, Ingest);

#[cfg(test)]
mod tests {
    use super::*;
    use hipfire_arch_api::ArchRegistry;

    #[test]
    fn registers_as_diffusion_arch() {
        let reg = ArchRegistry::build();
        assert!(reg.is_diffusion(QWEN_IMAGE_ARCH_ID));
        assert_eq!(
            reg.diffusion_family(QWEN_IMAGE_ARCH_ID),
            Some("qwen-image-mmdit")
        );
        assert!(reg.get(QWEN_IMAGE_ARCH_ID).unwrap().caps.ingest.is_some());
    }

    #[test]
    fn text_stream_modules_classify() {
        // Text-stream adds are attention; text modulation protects.
        assert_eq!(
            QwenImageSpec.importance("transformer_blocks.0.attn.add_q_proj.weight"),
            255
        );
        assert_eq!(
            QwenImageSpec.importance("transformer_blocks.0.txt_mod.linear.weight"),
            255
        );
        assert_eq!(
            QwenImageSpec.importance("transformer_blocks.0.txt_mlp.net.2.weight"),
            128
        );
    }
}
