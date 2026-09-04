// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Björn Bösel
// hipfire — see LICENSE and NOTICE in the project root.

//! Shared dispatch macros.
//!
//! Single home for the `hip!` helper. Declared with `#[macro_use]` before
//! every other module in `lib.rs` so textual macro scoping makes it visible
//! to all `families::*` and `pipeline` call sites.

/// Map a fallible HIP call into DispatchError::Hip, preserving the message.
macro_rules! hip {
    ($e:expr) => {
        $e.map_err(|e| DispatchError::Hip(e.to_string()))
    };
}
