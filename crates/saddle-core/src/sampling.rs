// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Per-architecture sampling policy defaults.
//!
//! STUB — filled by lean-up map item **C4**. See
//! `docs/governance/2026-08-15-hipfire-leanup-map.md` § 5b for the file
//! ownership contract governing this module.
//!
//! Values are taken verbatim from the two duplicated ladders in
//! `crates/hipfire-runtime/examples/daemon.rs:1310` and `:14618` plus the
//! repeat-penalty branches at `:1336`/`:14697`. No guesses — the numbers
//! below are the code.

/// Default temperature / top-p / repeat-penalty for an architecture.
///
/// `temp` and `top_p` are the fallback when neither the `.hfq`
/// `generation_config` (`rec_temperature`/`rec_top_p`) nor an explicit
/// request field provides a value. `repeat_penalty` is the fallback when
/// the request omits `repeat_penalty`/`repetition_penalty`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SamplingDefaults {
    /// Fallback temperature when the request and the `.hfq` both omit it.
    pub temp: f64,
    /// Fallback top-p when the request and the `.hfq` both omit it.
    pub top_p: f64,
    /// Fallback repeat penalty when the request omits it.
    pub repeat_penalty: f64,
}

impl SamplingDefaults {
    /// Create a new set of defaults.
    pub const fn new(temp: f64, top_p: f64, repeat_penalty: f64) -> Self {
        Self {
            temp,
            top_p,
            repeat_penalty,
        }
    }
}

impl Default for SamplingDefaults {
    fn default() -> Self {
        // The `else` arm of both daemon ladders (covers 0,1,7,8,14 …).
        Self {
            temp: 0.3,
            top_p: 0.8,
            repeat_penalty: 1.0,
        }
    }
}
