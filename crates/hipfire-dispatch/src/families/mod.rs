// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Björn Bösel
// hipfire — see LICENSE and NOTICE in the project root.

//! Kernel families — dispatch surfaces with multiple (quant tier × arch ISA)
//! kernel arms behind one typed entry point.
//!
//! # Adding a new family (the contract — #397 Phase 0.5)
//!
//! A new kernel family MUST:
//!
//! 1. **Add `KernelKey` arm(s)** in the flat enum (`types.rs`) — one key per
//!    (quant tier × fusion variant) the resolver must distinguish.
//! 2. **Write `<family>_table::populate`** registering exactly one
//!    `KernelVariant` per key, each with the **narrowest** correct
//!    `ArchPredicate` (prefer `HasWmma`/`HasDp4a`/`Always` over hand-rolled
//!    arch ORs — Phase 0.4), an optional `ShapePredicate` for
//!    runtime-dimension gating, the `PipelineOp` `steps`, and the `has_awq`
//!    flag. **Register in priority order** (most-specific/fastest first) —
//!    the resolver returns the first passing variant; the
//!    gfx12 > gfx11 > dp4a ladder is encoded in registration order.
//! 3. **Add a `<Family>` struct** owning its `KernelRegistry`, built once in
//!    `new()` via `populate()` + `registry.validate()` (hard-errors on empty
//!    entries).
//! 4. **Expose `resolve()` + typed `run*()`** — models call the typed entry
//!    point, never the registry directly, and **never** `match` on `DType`.
//! 5. **Register the module** here and the table in `tables/mod.rs`.
//! 6. **Add per-arch golden tests** in `hipfire-dispatch-tests` asserting the
//!    resolved key for each (arch × dtype) the family supports, **including
//!    the RDNA4 row** — per-arch goldens are exactly what catches the
//!    dead-gate class (Phase 0.4 / 953ea648).
//! 7. **A new `ArchPredicate` variant may only land in the same change as the
//!    kernel it gates** (0.4 lesson: dead predicates cause dead gates).
//!
//! Single-impl ops (rmsnorm, silu, …) do NOT need a family — they dispatch
//! directly through `launch_op` in the interpreter. The family abstraction is
//! for ops with multiple quant-format × arch-ISA dispatch arms.
//! Full rationale: `docs/plans/dispatch-phase0-decisions.md` §0.5.

pub mod gemv;

pub mod moe;

pub mod rotation;

pub mod attention;
pub mod fused_qkv;
pub mod gemm;
pub mod kv_tier;
