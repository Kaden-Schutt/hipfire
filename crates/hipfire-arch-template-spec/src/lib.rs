// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 hipfire contributors
// hipfire — see LICENSE and NOTICE in the project root.

//! Copy-paste template for a new family's OFFLINE spec (the `Ingest` quant-policy).
//!
//! This crate depends on NOTHING but `hipfire-arch-api`, so the quantizer links it
//! without the runtime/GPU stack. It declares the same [`ArchId`] as its serving
//! sibling `hipfire-arch-template`; the registry merges the two (this crate's
//! `Ingest` + the serving crate's `ToyModel`) into one arch.
//!
//! To add a family, copy this crate, rename `Template*` → your family, set the id,
//! and (usually) leave the `Ingest` body delegating to the shared `transformer_role`
//! prior — override `importance`/`requires` only for genuinely special tensors (see
//! `hipfire-arch-deepseek4-spec` for an MLA-compressor override example).

use hipfire_arch_api::{
    default_importance, default_requires, register_arch, transformer_role, Arch, ArchId, CapReq,
    Ingest, TensorRole,
};

/// Reserved template id — never shipped. Same id as the serving sibling.
pub const TEMPLATE_ARCH_ID: ArchId = ArchId(0xFF);

/// Lean identity marker for the template family's offline spec.
pub struct TemplateSpec;

impl Arch for TemplateSpec {
    fn id(&self) -> ArchId {
        TEMPLATE_ARCH_ID
    }
    fn family(&self) -> &'static str {
        "template"
    }
}

impl Ingest for TemplateSpec {
    // The arch states only WHAT each tensor is and how important it is; it never
    // names a format or a codec — the deployment's `allocate()` picks the codec. Most
    // families just delegate to the shared transformer prior:
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

static TEMPLATE_SPEC: TemplateSpec = TemplateSpec;
register_arch!(TEMPLATE_SPEC, Ingest);

#[cfg(test)]
mod tests {
    use super::*;
    use hipfire_arch_api::{allocate, ArchRegistry, CodecCaps};

    #[test]
    fn registers_ingest() {
        let reg = ArchRegistry::build();
        let a = reg.get(TEMPLATE_ARCH_ID).expect("template spec registered");
        assert_eq!(a.family, "template");
        assert!(a.caps.ingest.is_some());
    }

    /// The reference the whole capability layer exists for: the arch states needs,
    /// the DEPLOYMENT's codec menu (format names live only here) is matched by
    /// `allocate`, and the embedding lands on a random-access high-bit codec — with
    /// no format name in the arch. This is what replaces `is_q8_tensor`.
    #[test]
    fn ingest_derives_codec_without_naming_formats_arch_side() {
        let reg = ArchRegistry::build();
        let ing = reg.get(TEMPLATE_ARCH_ID).unwrap().caps.ingest.unwrap();

        // The codec menu is the DEPLOYMENT's — format names appear only here.
        let codecs = [
            CodecCaps {
                name: "compressed-4b",
                bits_per_weight: 4.0,
                group_size: 256,
                random_access: false,
            },
            CodecCaps {
                name: "high-8b-ra",
                bits_per_weight: 8.0,
                group_size: 0,
                random_access: true,
            },
        ];

        // Embedding: important + random-access → the high-precision random-access
        // codec is DERIVED (the old is_q8_tensor outcome, no format named arch-side).
        let embed = "model.embed_tokens.weight";
        let sel = allocate(ing.importance(embed), ing.requires(embed), 4096, &codecs).unwrap();
        assert_eq!(sel.name, "high-8b-ra");

        // A generic MLP projection → the compressed codec.
        let mlp = "model.layers.0.mlp.up_proj.weight";
        let sel = allocate(ing.importance(mlp), ing.requires(mlp), 4096, &codecs).unwrap();
        assert_eq!(sel.bits_per_weight, 4.0);
    }
}
