// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Björn Bösel
// hipfire — see LICENSE and NOTICE in the project root.

//! In-process override toggle for the MoE decode pre-down decomposition.
//!
//! Scaffolding for the D2a A/B harness (Task 1). The toggle lets test code
//! force the decomposed Step path ON or OFF in the same process without
//! relying on a `OnceLock`-backed env read (which cannot flip mid-process).
//!
//! Production default (env unset, override None) = OFF = direct kernels.
//! Will be removed at Task 7 once the decomposed path is the only path.

use std::sync::atomic::{AtomicU8, Ordering};

/// 0 = unset (read env once), 1 = force OFF, 2 = force ON.
static OVERRIDE: AtomicU8 = AtomicU8::new(0);

/// Test-only: force the decode pre-down decomposition on/off IN-PROCESS.
/// A bare env `OnceLock` cannot flip in-process (D1 trap); this can.
pub fn set_moe_step_predown_override(v: Option<bool>) {
    OVERRIDE.store(
        match v {
            None => 0,
            Some(false) => 1,
            Some(true) => 2,
        },
        Ordering::SeqCst,
    );
}

/// True → arch builds the decomposed pre-down Step stream; false → direct kernels.
pub fn moe_step_predown_enabled() -> bool {
    match OVERRIDE.load(Ordering::SeqCst) {
        1 => false,
        2 => true,
        _ => std::env::var("HIPFIRE_MOE_STEP_PREDOWN")
            .map(|v| v == "1")
            .unwrap_or(false),
    }
}
