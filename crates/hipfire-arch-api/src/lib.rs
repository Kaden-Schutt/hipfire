// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 hipfire contributors
// hipfire — see LICENSE and NOTICE in the project root.

//! # hipfire-arch-api — the capability layer
//!
//! Every model architecture is a small `static` singleton implementing the base
//! [`Arch`] trait plus whichever *capability* traits it actually supports
//! ([`BatchedPrefill`], [`SpecDecodeChain`], [`ToyModel`], …). The arch crate then
//! calls [`register_arch!`] to publish itself and declare its capabilities.
//!
//! The daemon, scheduler and quantizer never branch on `arch_id == N`. They build
//! an [`ArchRegistry`] once and ask, per capability, "does this arch support it?":
//!
//! ```ignore
//! match reg.get(id).and_then(|a| a.caps.batched_prefill) {
//!     Some(bp) => schedule_batched(bp, ...),   // supported → dispatch
//!     None     => schedule_one_at_a_time(...), // unsupported → safe fallback
//! }
//! ```
//!
//! so the daemon can never call `arch_batch_prefill` on an arch that does not
//! implement it — the type system makes that unrepresentable.
//!
//! ## Why declarative (not autoref specialization)
//!
//! A spike (2026-07-03) showed autoref-based specialization — deriving the `Caps`
//! automatically from which traits are `impl`'d — is toolchain-fragile (two call
//! forms gave two different wrong resolutions). So capabilities are *listed* in
//! `register_arch!`. The `as &dyn Cap` cast means you **cannot over-claim** (listing
//! a capability you don't `impl` fails to compile); the completeness gate catches
//! **under-claiming** (an `impl` you forgot to list). See
//! `docs/plans/2026-07-03-arch-capability-layer.md`.
//!
//! ## Leaf-crate invariant
//!
//! This crate depends on nothing but `std` + `inventory`. Capability trait method
//! signatures use only plain data, so the offline quantizer can link the trait
//! definitions without pulling in the serving/kernel stack. Serving-heavy method
//! bodies live in the arch crates that `impl` them, not here.

/// Stable numeric identity of an architecture family (the on-disk/header id).
///
/// Named constants for the concrete families live alongside the registry as they
/// migrate onto this layer; the registry keys on this value.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ArchId(pub u16);

impl core::fmt::Display for ArchId {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "arch#{}", self.0)
    }
}

/// The base every architecture implements. Identity only — behaviour lives in the
/// capability traits so unsupported behaviour is `None`, not a panic.
pub trait Arch: Sync + 'static {
    /// Stable numeric id (registry key, on-disk header value).
    fn id(&self) -> ArchId;
    /// Human family name for logs/errors, e.g. `"llama"`, `"nemotron-h"`.
    fn family(&self) -> &'static str;
}

// ---------------------------------------------------------------------------
// Capability traits. Add one here + a `Caps` field + a `__set_cap!` arm.
// Names describe BEHAVIOUR + AXIS — never a codename (no `*Flash`). See the
// naming table in the plan.
// ---------------------------------------------------------------------------

/// The arch can prefill a whole prompt in one batched forward pass (rather than
/// one token at a time). Presence of the capability is the "yes"; the method
/// carries the batch-shaping knobs migrated in later.
pub trait BatchedPrefill: Sync + 'static {
    /// Largest prompt length accepted in a single batched prefill call.
    fn max_prefill_batch(&self) -> usize;
}

/// The arch supports a speculative-decode *chain* drafter (linear draft of N
/// tokens verified in one step). Distinct from a tree drafter.
pub trait SpecDecodeChain: Sync + 'static {
    /// Max draft length proposed per verification step.
    fn max_draft_len(&self) -> usize;
}

/// The arch can synthesize a tiny deterministic fixture checkpoint for tests/CI
/// (the `generate_toy_model_from_seed`-style helpers, co-located with the arch
/// instead of scattered in test harnesses).
pub trait ToyModel: Sync + 'static {
    /// Write a minimal self-consistent `.hfq` (+ any sidecars) under `out_dir`,
    /// seeded deterministically. Returns the artifact stem on success.
    fn emit_fixture(&self, out_dir: &std::path::Path, seed: u64) -> Result<String, String>;
}

/// Per-arch capability table: `Some(&dyn Cap)` iff the arch declared it. Built at
/// registry construction from the arch's `register_arch!` list.
pub struct Caps {
    pub batched_prefill: Option<&'static dyn BatchedPrefill>,
    pub spec_decode_chain: Option<&'static dyn SpecDecodeChain>,
    pub toy_model: Option<&'static dyn ToyModel>,
}

impl Caps {
    /// All capabilities absent — the base every `register_arch!` starts from.
    pub const fn none() -> Self {
        Caps {
            batched_prefill: None,
            spec_decode_chain: None,
            toy_model: None,
        }
    }
}

/// One link-time registration record. Constructed by [`register_arch!`] and
/// collected by `inventory`; the registry turns each into a [`RegisteredArch`].
pub struct ArchEntry {
    /// The arch singleton, as its base trait object.
    pub base: &'static dyn Arch,
    /// Builds the capability table (runs the `as &dyn Cap` coercions at startup,
    /// so `ArchEntry` itself stays const-constructible for `inventory::submit!`).
    pub make_caps: fn() -> Caps,
}

inventory::collect!(ArchEntry);

/// Re-exported so `register_arch!` can reference `$crate::inventory` regardless of
/// where the macro is invoked.
pub use inventory;

/// A fully-resolved arch: identity + capability table. Yielded by [`ArchRegistry`].
pub struct RegisteredArch {
    pub id: ArchId,
    pub family: &'static str,
    pub base: &'static dyn Arch,
    pub caps: Caps,
}

/// Runtime index over every `register_arch!`-published architecture linked into the
/// binary. Build once (e.g. into a `OnceLock`), then look up by [`ArchId`].
pub struct ArchRegistry {
    archs: Vec<RegisteredArch>,
}

impl ArchRegistry {
    /// Collect all link-time registrations. Cheap; call once and cache.
    pub fn build() -> Self {
        let mut archs = Vec::new();
        for entry in inventory::iter::<ArchEntry> {
            archs.push(RegisteredArch {
                id: entry.base.id(),
                family: entry.base.family(),
                base: entry.base,
                caps: (entry.make_caps)(),
            });
        }
        ArchRegistry { archs }
    }

    /// The registered arch for `id`, if any is linked in.
    pub fn get(&self, id: ArchId) -> Option<&RegisteredArch> {
        self.archs.iter().find(|a| a.id == id)
    }

    /// Iterate every registered arch (used by the completeness gate).
    pub fn iter(&self) -> impl Iterator<Item = &RegisteredArch> {
        self.archs.iter()
    }

    pub fn len(&self) -> usize {
        self.archs.len()
    }

    pub fn is_empty(&self) -> bool {
        self.archs.is_empty()
    }
}

/// Internal: map a capability ident to its `Caps` field with the `as &dyn Cap`
/// coercion. The coercion is what makes over-claiming a compile error. One arm per
/// capability trait — keep in sync with the `Caps` fields.
#[doc(hidden)]
#[macro_export]
macro_rules! __set_cap {
    ($caps:ident, $inst:expr, BatchedPrefill) => {
        $caps.batched_prefill =
            ::core::option::Option::Some($inst as &'static dyn $crate::BatchedPrefill);
    };
    ($caps:ident, $inst:expr, SpecDecodeChain) => {
        $caps.spec_decode_chain =
            ::core::option::Option::Some($inst as &'static dyn $crate::SpecDecodeChain);
    };
    ($caps:ident, $inst:expr, ToyModel) => {
        $caps.toy_model = ::core::option::Option::Some($inst as &'static dyn $crate::ToyModel);
    };
    ($caps:ident, $inst:expr, $other:ident) => {
        ::core::compile_error!(::core::concat!(
            "unknown capability `",
            ::core::stringify!($other),
            "` in register_arch! — add a `__set_cap!` arm + a `Caps` field for it"
        ));
    };
}

/// Publish an architecture and declare its capabilities.
///
/// `$inst` is a `static` singleton implementing [`Arch`]; each listed capability
/// must be `impl`'d on it (else the `as &dyn Cap` cast fails to compile — you
/// cannot over-claim). Capabilities you `impl` but forget to list are caught by the
/// completeness gate.
///
/// ```ignore
/// static INSTANCE: LlamaArch = LlamaArch;
/// hipfire_arch_api::register_arch!(INSTANCE, BatchedPrefill, ToyModel);
/// ```
#[macro_export]
macro_rules! register_arch {
    ($inst:path $(, $cap:ident)* $(,)?) => {
        $crate::inventory::submit! {
            $crate::ArchEntry {
                base: &$inst,
                make_caps: {
                    // A named fn (const fn item) keeps `ArchEntry` const-constructible
                    // for `inventory::submit!`; the `as &dyn Cap` casts run when the
                    // registry calls it, not in the static initializer.
                    fn __make_caps() -> $crate::Caps {
                        let mut caps = $crate::Caps::none();
                        $( $crate::__set_cap!(caps, &$inst, $cap); )*
                        caps
                    }
                    __make_caps
                },
            }
        }
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    struct TestArch;
    impl Arch for TestArch {
        fn id(&self) -> ArchId {
            ArchId(0x7E57)
        }
        fn family(&self) -> &'static str {
            "test"
        }
    }
    impl ToyModel for TestArch {
        fn emit_fixture(&self, _out_dir: &std::path::Path, _seed: u64) -> Result<String, String> {
            Ok("test-fixture".to_string())
        }
    }
    impl SpecDecodeChain for TestArch {
        fn max_draft_len(&self) -> usize {
            4
        }
    }
    // Deliberately does NOT impl BatchedPrefill.

    static TEST_INSTANCE: TestArch = TestArch;
    register_arch!(TEST_INSTANCE, ToyModel, SpecDecodeChain);

    #[test]
    fn registry_discovers_declared_caps() {
        let reg = ArchRegistry::build();
        let a = reg
            .get(ArchId(0x7E57))
            .expect("TestArch should be link-time registered");
        assert_eq!(a.family, "test");

        // Declared → Some.
        assert!(a.caps.toy_model.is_some(), "ToyModel was listed");
        assert!(
            a.caps.spec_decode_chain.is_some(),
            "SpecDecodeChain was listed"
        );
        // Not declared / not impl'd → None (the daemon's safe-fallback branch).
        assert!(
            a.caps.batched_prefill.is_none(),
            "BatchedPrefill was neither impl'd nor listed"
        );

        // Dispatch actually works through the capability object.
        assert_eq!(a.caps.spec_decode_chain.unwrap().max_draft_len(), 4);
        assert_eq!(
            a.caps
                .toy_model
                .unwrap()
                .emit_fixture(std::path::Path::new("/tmp"), 7)
                .unwrap(),
            "test-fixture"
        );
    }
}
