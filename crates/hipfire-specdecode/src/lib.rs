// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Arch-agnostic speculative-decode core.
//!
//! This crate holds the draft-verify machinery that is *not* specific to any one
//! architecture: the [`SpecDecodeTarget`] model boundary, the shared seam types
//! (`SpecPair`, `SpecStepResult`, `KvMode`, `ModelSlotConfig`, verify-graph and
//! rollback-replay policies, `HiddenStateRingBuffer`, n-gram / PLD matchers,
//! stats), and the generic `ModelSlot<T>` driver.
//!
//! Strategy families (dflash, ddtree, mtp, dspark) live in their own
//! `hipfire-specdecode-*` crates and are generic over [`SpecDecodeTarget`]; an
//! arch crate (e.g. `hipfire-arch-qwen35`) implements the trait for its own
//! config/weights/scratch so no strategy crate ever names a concrete arch.
//!
//! Extraction is staged (see `docs/specdecode-extraction-plan.md`); this file is
//! the P0 scaffold — the seam types land in P1 and the trait in P2.
