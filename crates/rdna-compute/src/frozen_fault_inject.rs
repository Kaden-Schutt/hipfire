// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Test-only free-failure injection for `Gpu::free_tensor_checked`
//! (feature `frozen-fault-inject`, propagated from hipfire-arch-qwen35).
//!
//! While `HIPFIRE_FROZEN_FAIL_FREE=1`, the FIRST checked-free attempt fails
//! without consuming the tensor; every later attempt succeeds. The flag is
//! one-shot so a test can assert that the retained-owner path carries the
//! tensor, then retry (with the env cleared) and observe recovery.

use std::sync::atomic::{AtomicBool, Ordering};

static CONSUMED: AtomicBool = AtomicBool::new(false);

/// One-shot free-failure injection. Returns `true` exactly once while the
/// env var is set; returns `false` when unset (and resets the one-shot so a
/// later re-arm starts fresh).
pub fn free_should_fail() -> bool {
    let armed = std::env::var("HIPFIRE_FROZEN_FAIL_FREE")
        .map(|v| v == "1")
        .unwrap_or(false);
    if !armed {
        CONSUMED.store(false, Ordering::SeqCst);
        return false;
    }
    !CONSUMED.swap(true, Ordering::SeqCst)
}

/// Reset the one-shot flag unconditionally (belt-and-braces for tests whose
/// env state is not the canonical set→fail→clear→retry sequence).
pub fn reset() {
    CONSUMED.store(false, Ordering::SeqCst);
}
