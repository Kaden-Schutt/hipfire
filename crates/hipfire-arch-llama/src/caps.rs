// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 hipfire contributors
// hipfire — see LICENSE and NOTICE in the project root.

//! Serving-side capabilities for the Llama family — the runtime half of the
//! offline/serving split.
//!
//! This crate registers the capabilities that need the runtime (here
//! [`BatchedPrefill`]) on the SAME [`ArchId`] as the lean offline spec
//! ([`hipfire_arch_llama_spec`], which contributes [`Ingest`]). Because this crate
//! depends on the spec crate, linking the serving crate pulls in BOTH
//! registrations, and [`ArchRegistry`] merges them into one Llama arch. That is the
//! split working end-to-end on a production arch: the quantizer can link only the
//! lean spec (offline `Ingest`), while the daemon links this serving crate and gets
//! `Ingest` + `BatchedPrefill` merged.
//!
//! [`Ingest`]: hipfire_arch_api::Ingest
//! [`ArchRegistry`]: hipfire_arch_api::ArchRegistry

use crate::arch::Llama;
use hipfire_arch_api::{register_arch, Arch, ArchId, BatchedPrefill};
use hipfire_arch_llama_spec::LLAMA_ARCH_ID;

impl Arch for Llama {
    fn id(&self) -> ArchId {
        // Same id the -spec crate declares; the registry merges the two entries.
        LLAMA_ARCH_ID
    }
    fn family(&self) -> &'static str {
        "llama"
    }
}

impl BatchedPrefill for Llama {
    fn max_prefill_batch(&self) -> usize {
        // Llama prefills the whole prompt in one batched forward. The real
        // per-model cap is wired from config during the deeper migration; unbounded
        // is the safe placeholder (the scheduler already clamps to context length).
        usize::MAX
    }
}

/// Serving singleton. Shares [`LLAMA_ARCH_ID`] with the offline spec; the registry
/// unions their capability tables.
static LLAMA: Llama = Llama;
register_arch!(LLAMA, BatchedPrefill);

#[cfg(test)]
mod tests {
    use super::*;
    use hipfire_arch_api::ArchRegistry;

    #[test]
    fn spec_and_serving_merge_on_one_id() {
        // This crate registers BatchedPrefill; its dep hipfire-arch-llama-spec
        // registers Ingest on the same id. Linked together (the daemon's view), the
        // registry must merge both into one llama arch — the offline/serving split
        // working end-to-end on a production arch.
        let reg = ArchRegistry::build();
        let a = reg.get(LLAMA_ARCH_ID).expect("llama registered");
        assert_eq!(a.family, "llama");
        assert!(
            a.caps.ingest.is_some(),
            "Ingest merged in from the -spec crate"
        );
        assert!(
            a.caps.batched_prefill.is_some(),
            "BatchedPrefill from this serving crate"
        );
        assert_eq!(
            a.caps.batched_prefill.unwrap().max_prefill_batch(),
            usize::MAX
        );
        // Exactly one llama arch — the two registrations collapsed, not duplicated.
        assert_eq!(reg.iter().filter(|x| x.id == LLAMA_ARCH_ID).count(), 1);
    }
}
