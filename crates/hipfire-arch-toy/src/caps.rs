// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Capability-layer wiring for the toy arch — the reference every new arch copies.
//!
//! Three steps, all here so a contributor has one place to look:
//!  1. `impl hipfire_arch_api::Arch for Toy` — identity (id + family).
//!  2. `impl` each *capability* trait the arch genuinely supports. The toy is a
//!     stub, so it supports only [`ToyModel`] (synthesize a fixture). It has no
//!     real batched prefill, so it does NOT impl `BatchedPrefill` — the daemon
//!     consulting `caps.batched_prefill` then sees `None` and takes its safe
//!     one-token fallback. Declare a capability ONLY when the arch really has it.
//!  3. `register_arch!(INSTANCE, ...)` — list exactly the capabilities impl'd
//!     above. Listing one you didn't impl fails to compile; forgetting one you
//!     did is caught by the completeness gate.
//!
//! [`ToyModel`]: hipfire_arch_api::ToyModel

use crate::arch::Toy;
use crate::toy_model::{ToyConfig, ToyWeights};
use hipfire_arch_api::{register_arch, Arch, ArchId, ToyModel};
use std::path::Path;

/// Reserved toy id — never shipped in a real `.hfq`. Kept as a named constant so
/// the id lives in one place (mirrors `Architecture::arch_id() == 0xFF`).
pub const TOY_ARCH_ID: ArchId = ArchId(0xFF);

impl Arch for Toy {
    fn id(&self) -> ArchId {
        TOY_ARCH_ID
    }
    fn family(&self) -> &'static str {
        "toy"
    }
}

impl ToyModel for Toy {
    fn emit_fixture(&self, out_dir: &Path, seed: u64) -> Result<String, String> {
        let cfg = ToyConfig {
            vocab_size: 256,
            dim: 8,
            layers: 1,
        };
        let w = ToyWeights::from_seed(&cfg, seed);
        let stem = format!("toy-{seed:016x}");
        let path = out_dir.join(format!("{stem}.bin"));
        let mut bytes = Vec::with_capacity(w.embeddings.len() * 4);
        for v in &w.embeddings {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        std::fs::write(&path, &bytes).map_err(|e| format!("write {}: {e}", path.display()))?;
        Ok(stem)
    }
}

/// The arch singleton. Trait dispatch uses the type; the registry needs a
/// `'static` value to hand out as `&dyn Arch` / `&dyn Cap`.
static TOY_INSTANCE: Toy = Toy;
register_arch!(TOY_INSTANCE, ToyModel);

#[cfg(test)]
mod tests {
    use super::*;
    use hipfire_arch_api::ArchRegistry;

    #[test]
    fn toy_registers_and_consult_pattern_holds() {
        let reg = ArchRegistry::build();
        let a = reg
            .get(TOY_ARCH_ID)
            .expect("toy arch should be link-time registered");
        assert_eq!(a.family, "toy");

        // Supported capability → Some → dispatch (the daemon's "do it" branch).
        let tm = a.caps.toy_model.expect("toy declares ToyModel");
        let dir = std::env::temp_dir();
        let stem = tm.emit_fixture(&dir, 0xABCD).expect("fixture emit");
        assert!(dir.join(format!("{stem}.bin")).exists());

        // Unsupported capability → None → the daemon's safe fallback branch. This
        // is the whole point: the daemon can't call batched prefill on the toy.
        assert!(a.caps.batched_prefill.is_none());
    }
}
