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
}
