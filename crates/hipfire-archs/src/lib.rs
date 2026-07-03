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
    #[allow(unused_imports)]
    use hipfire_arch_llama_spec as _;
    #[allow(unused_imports)]
    use hipfire_arch_toy as _;
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
    fn bundle_exposes_toy_registration() {
        // The toy arch (id 0xFF) must be reachable THROUGH the bundle — proving
        // force-linking preserved its inventory submission across the crate
        // boundary (the real daemon path, unlike an in-crate test).
        let reg = registry();
        let toy = reg
            .get(ArchId(0xFF))
            .expect("toy arch reachable through the bundle");
        assert_eq!(toy.family, "toy");
        assert!(toy.caps.toy_model.is_some());
        assert!(toy.caps.batched_prefill.is_none());
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
        // untracked addition. (0x00 = llama offline spec, 0xFF = toy.)
        let expected: BTreeSet<u16> = [0x00, 0xFF].into_iter().collect();
        assert_eq!(
            unique, expected,
            "arch migration ledger drift — update the expected set as families \
             move onto the capability layer (added or dropped id detected)"
        );
    }
}
