// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Per-arch kernel-dispatch *overlays* — the whole-method `*_gfxNNNN` GPU
//! dispatch overlays, file-separated from the reference floor (Phase 2 of
//! docs/plans/2026-06-23-dispatch-refactor.md). Each is an `impl Gpu` block; the
//! family files keep the reference path + the `if arch_caps.is_gfxNNNN() { return
//! self.<overlay>() }` selection. Isolating overlay code per arch lets a
//! single-arch kernel-dispatch change scope to that arch's gate cell.
mod gfx11;
mod gfx1151;
mod gfx12;
mod gfx906;
mod gfx942;
