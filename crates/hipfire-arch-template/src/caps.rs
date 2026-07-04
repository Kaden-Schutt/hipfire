// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Capability-layer wiring for the template arch (SERVING half) — the reference a
//! new arch's serving crate copies.
//!
//! Three steps, all here so a contributor has one place to look:
//!  1. `impl hipfire_arch_api::Arch for Template` — identity (id + family). This is
//!     declared in BOTH halves (here and in `hipfire-arch-template-spec`) on the
//!     same [`TEMPLATE_ARCH_ID`]; the registry merges the two entries into one arch.
//!  2. `impl` each *serving* capability the arch genuinely supports. The template is
//!     a stub, so it supports only [`ToyModel`] (synthesize a tiny fixture). It has
//!     no real batched prefill, so it does NOT impl `BatchedPrefill` — the daemon
//!     consulting `caps.batched_prefill` then sees `None` and takes its safe
//!     one-token fallback. Declare a capability ONLY when the arch really has it.
//!  3. `register_arch!(INSTANCE, ...)` — list exactly the capabilities impl'd above.
//!     Listing one you didn't impl fails to compile; forgetting one you did is
//!     caught by the completeness gate.
//!
//! The OFFLINE `Ingest` quant-policy is NOT here — it lives in the lean
//! `hipfire-arch-template-spec` crate (deps only `hipfire-arch-api`, so the
//! quantizer links it without the runtime/GPU stack). The two register on the same
//! id and the registry unions their capabilities.
//!
//! [`ToyModel`]: hipfire_arch_api::ToyModel

use crate::arch::Template;
use crate::template_model::TemplateConfig;
use hipfire_arch_api::{register_arch, Arch, ArchId, Init, TensorSpec, ToyFixture, ToyModel};

/// Reserved template id — never shipped in a real `.hfq`. Kept as a named constant
/// so the id lives in one place (mirrors `Architecture::arch_id() == 0xFF`). The
/// `-spec` sibling re-declares the SAME id.
pub const TEMPLATE_ARCH_ID: ArchId = ArchId(0xFF);

impl Arch for Template {
    fn id(&self) -> ArchId {
        TEMPLATE_ARCH_ID
    }
    fn family(&self) -> &'static str {
        "template"
    }
}

impl ToyModel for Template {
    fn fixture(&self, _seed: u64) -> ToyFixture {
        // Shape is seed-independent; the quantizer seeds the actual bytes. The
        // template is a stub, so this is a minimal one-tensor manifest.
        let cfg = TemplateConfig {
            vocab_size: 256,
            dim: 8,
            layers: 1,
        };
        ToyFixture {
            config_json: format!(
                "{{\"vocab_size\":{},\"hidden_size\":{},\"num_hidden_layers\":{}}}",
                cfg.vocab_size, cfg.dim, cfg.layers
            ),
            tensors: vec![TensorSpec::new(
                "model.embed_tokens.weight",
                vec![cfg.vocab_size, cfg.dim],
                Init::Uniform(0.02),
            )],
        }
    }
}

/// The serving singleton. Trait dispatch uses the type; the registry needs a
/// `'static` value to hand out as `&dyn Arch` / `&dyn Cap`.
static TEMPLATE_INSTANCE: Template = Template;
register_arch!(TEMPLATE_INSTANCE, ToyModel);

#[cfg(test)]
mod tests {
    use super::*;
    use hipfire_arch_api::ArchRegistry;

    #[test]
    fn template_registers_and_consult_pattern_holds() {
        let reg = ArchRegistry::build();
        let a = reg
            .get(TEMPLATE_ARCH_ID)
            .expect("template arch should be link-time registered");
        assert_eq!(a.family, "template");

        // Supported capability → Some → dispatch (the daemon's "do it" branch).
        let tm = a.caps.toy_model.expect("template declares ToyModel");
        let f = tm.fixture(0xABCD);
        assert!(!f.tensors.is_empty() && !f.config_json.is_empty());

        // Unsupported capability → None → the daemon's safe fallback branch. This is
        // the whole point: the daemon can't call batched prefill on the template.
        assert!(a.caps.batched_prefill.is_none());
    }
}
