// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 hipfire contributors
// hipfire — see LICENSE and NOTICE in the project root.

//! Lean offline spec for the dots.ocr (Qwen2-VL) family (arch_id 8): identity + the
//! `Ingest` quant-policy (shared transformer prior). Deps only `hipfire-arch-api`.

use hipfire_arch_api::{
    default_importance, default_requires, register_arch, transformer_role, Arch, ArchId, CapReq,
    Ingest, TensorRole,
};

/// dots.ocr family header id.
pub const DOTS_OCR_ARCH_ID: ArchId = ArchId(8);

/// Lean identity marker for the dots.ocr offline spec.
pub struct DotsOcrSpec;

impl Arch for DotsOcrSpec {
    fn id(&self) -> ArchId {
        DOTS_OCR_ARCH_ID
    }
    fn family(&self) -> &'static str {
        "dots-ocr"
    }
}

impl Ingest for DotsOcrSpec {
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

static DOTS_OCR_SPEC: DotsOcrSpec = DotsOcrSpec;
register_arch!(DOTS_OCR_SPEC, Ingest);

#[cfg(test)]
mod tests {
    use super::*;
    use hipfire_arch_api::ArchRegistry;

    #[test]
    fn registers_ingest() {
        let reg = ArchRegistry::build();
        let a = reg.get(DOTS_OCR_ARCH_ID).expect("dots-ocr spec registered");
        assert_eq!(a.family, "dots-ocr");
        assert!(a.caps.ingest.is_some());
    }
}
