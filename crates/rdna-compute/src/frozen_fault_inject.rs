// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Test-only free-failure injection for `Gpu::free_tensor_checked`
//! (feature `frozen-fault-inject`, propagated from hipfire-arch-qwen35).
//!
//! While `HIPFIRE_FROZEN_FAIL_FREE` is set (any non-empty value other than
//! `"0"`), EVERY checked-free attempt fails without consuming the tensor —
//! the caller's `Option` stays `Some`, exactly like a real bind_thread
//! failure. Tests clear the env var before the retry phase, so the
//! retained-owner path is exercised end to end: fail → owners retained in
//! the error → env cleared → retry succeeds → VRAM recovered.

/// Returns `true` while `HIPFIRE_FROZEN_FAIL_FREE` is set (non-empty, not
/// `"0"`). Every call fails in that window, so an initial teardown AND its
/// retry can both fail — required to exercise the retained-owner backlog.
pub fn free_should_fail() -> bool {
    std::env::var("HIPFIRE_FROZEN_FAIL_FREE")
        .map(|v| !v.is_empty() && v != "0")
        .unwrap_or(false)
}

/// Kept for API compatibility with tests that call it before arming;
/// continuous-while-set injection has no internal state to reset.
pub fn reset() {}
