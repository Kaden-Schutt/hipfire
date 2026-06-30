// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 hipfire contributors
// hipfire — see LICENSE and NOTICE in the project root.

//! `hipfire-primitives` — dependency-free numeric primitives shared across the
//! workspace.
//!
//! These are small, pure-Rust building blocks that several crates need
//! independently of the GPU runtime, the on-disk format, or any heavy
//! dependency: IEEE half-float conversion and the per-256 signed FWHT used by
//! the rotated quant formats. They previously lived in `hipfire-kvquant`
//! (whose actual job is the KV-cache codec) and were copy-pasted into a dozen
//! other crates; homing them here lets every crate share one implementation.
//!
//! Keep this crate a zero-dependency leaf: only add primitives that are
//! genuinely cross-cutting and free of external deps.

pub mod conv;
pub mod fwht;
