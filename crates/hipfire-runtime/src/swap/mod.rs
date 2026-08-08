// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Nick Woolmer
// hipfire — see LICENSE and NOTICE in the project root.
//
// KV swap: an idle session's GPU state survives losing its slot.
//
// The governing invariant, from which everything here follows: **the tokens
// are authoritative and KV is only a cache**. Every failure path in this
// module degrades to "re-prefill from the session's tokens" — slow, never
// wrong. There is deliberately no path that yields silently incorrect output,
// because that failure surfaces as a subtly worse agent rather than an error.

pub mod snapshot;
pub mod store;

/// Why a swap operation could not be completed.
///
/// Every variant means the same thing to a caller: drop the snapshot, mark the
/// session cold, and re-prefill. They are distinguished only so the log says
/// something useful.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SwapError {
    /// The snapshot was taken against a different model, dtype or layout.
    Stamp(String),
    /// Length or checksum mismatch — the bytes are not what was written.
    Corrupt(String),
    /// Filesystem failure.
    Io(String),
    /// A device copy failed.
    Gpu(String),
}

impl std::fmt::Display for SwapError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SwapError::Stamp(m) => write!(f, "swap stamp mismatch: {m}"),
            SwapError::Corrupt(m) => write!(f, "swap payload corrupt: {m}"),
            SwapError::Io(m) => write!(f, "swap io error: {m}"),
            SwapError::Gpu(m) => write!(f, "swap gpu error: {m}"),
        }
    }
}

impl std::error::Error for SwapError {}
