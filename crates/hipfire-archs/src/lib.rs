// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 hipfire contributors
// hipfire — see LICENSE and NOTICE in the project root.

//! # hipfire-archs — the arch aggregation point
//!
//! Every architecture registers itself at link time via
//! [`hipfire_arch_api::register_arch!`], but Rust drops rlibs whose items are
//! never referenced — which would drop the registrations with them. This crate
//! force-links each arch crate so all registrations survive, and exposes the
//! process-wide [`ArchRegistry`] built from them.
//!
//! ## Adding an architecture
//!
//! 1. add the arch crate to this crate's `Cargo.toml` `[dependencies]`;
//! 2. add a `use <crate> as _;` line to [`force_link`] below.
//!
//! That's it — no daemon/scheduler/quantizer edits. The completeness gate
//! (`no-gpu-ci`) fails if a shipped catalog id has no registered arch.

use hipfire_arch_api::ArchRegistry;
use std::sync::OnceLock;

/// Force-link every arch crate so its `register_arch!` submissions are pulled into
/// the final binary. Referencing the crate (even as `_`) creates the link edge.
mod force_link {
    // Every migrated family's offline Ingest spec (llama, qwen, gemma3, deepseek4,
    // nemotron, …), via the lean bundle.
    #[allow(unused_imports)]
    use hipfire_arch_specs as _;
    // The template's two halves (serving ToyModel + offline Ingest) on one id.
    #[allow(unused_imports)]
    use hipfire_arch_template as _;
    #[allow(unused_imports)]
    use hipfire_arch_template_spec as _;
}

pub use hipfire_arch_api::{self as api, Arch, ArchId, Caps, RegisteredArch};

static REGISTRY: OnceLock<ArchRegistry> = OnceLock::new();

/// The process-wide arch registry, built once from all linked registrations.
pub fn registry() -> &'static ArchRegistry {
    REGISTRY.get_or_init(ArchRegistry::build)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bundle_merges_template_spec_and_serving() {
        // The template registers from TWO crates on id 0xFF — its serving crate
        // (ToyModel) and its offline `-spec` crate (Ingest). Reaching it through the
        // bundle proves (a) force-linking preserved both inventory submissions
        // across the crate boundary (the real daemon path), and (b) the registry
        // MERGED them into one arch carrying both capabilities.
        let reg = registry();
        let t = reg
            .get(ArchId(0xFF))
            .expect("template arch reachable through the bundle");
        assert_eq!(t.family, "template");
        assert!(
            t.caps.toy_model.is_some(),
            "ToyModel from the serving crate"
        );
        assert!(t.caps.ingest.is_some(), "Ingest from the -spec crate");
        assert!(t.caps.batched_prefill.is_none());
    }

    #[test]
    fn bundle_exposes_llama_spec_ingest() {
        // The lean llama `-spec` crate's Ingest quant-policy is reachable through
        // the bundle — the path the quantizer will use to consult an arch's needs.
        let llama = registry()
            .get(ArchId(0x00))
            .expect("llama spec reachable through the bundle");
        assert_eq!(llama.family, "llama");
        assert!(llama.caps.ingest.is_some(), "llama declares Ingest");
    }

    /// Completeness gate. Two invariants that hold today and guard migration:
    ///  1. no two archs claim the same id, and every arch has a family name;
    ///  2. a migration LEDGER — the exact set of ids on the capability layer.
    ///
    /// Bullet 2 forces every migration to be an intentional one-line edit here
    /// (and catches an accidental dropped registration). Full catalog
    /// completeness — asserting every *shipped* catalog id is registered — turns
    /// on once all families have migrated; until then this ledger tracks progress.
    #[test]
    fn registry_integrity_and_migration_ledger() {
        use std::collections::BTreeSet;
        let reg = registry();

        let ids: Vec<u16> = reg.iter().map(|a| a.id.0).collect();
        let unique: BTreeSet<u16> = ids.iter().copied().collect();
        assert_eq!(
            ids.len(),
            unique.len(),
            "duplicate arch ids registered: {ids:?}"
        );
        for a in reg.iter() {
            assert!(!a.family.is_empty(), "arch {} has an empty family", a.id);
        }

        // Ids CURRENTLY migrated onto the capability layer. Add one per family as
        // it moves over; a mismatch means either a dropped registration or an
        // untracked addition. Ids: 0 llama, 1 qwen2/3, 5 qwen3.5, 6 qwen3.5-moe,
        // 8 dots-ocr, 9 deepseek4, 10 minimax, 11 lfm2, 12 gemma3, 13 gemma3-vl,
        // 14 nemotron-h, 15 mamba2, 16 zaya, 0xFF template.
        let expected: BTreeSet<u16> = [0, 1, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15, 16, 0xFF]
            .into_iter()
            .collect();
        assert_eq!(
            unique, expected,
            "arch migration ledger drift — update the expected set as families \
             move onto the capability layer (added or dropped id detected)"
        );
    }
}
