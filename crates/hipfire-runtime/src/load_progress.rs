// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire - see LICENSE and NOTICE in the project root.

//! Structured model-load progress sink.
//!
//! Model loaders (`hfq.rs` here, plus the per-arch loaders in the
//! `hipfire-arch-*` crates) walk their layers synchronously and historically
//! only `eprintln!`'d "loading layer N/M" for humans. That human line was being
//! scraped off the daemon's stderr to drive the chat UI's load bar — fragile
//! (couples to log wording; six different format strings) and, on a piped
//! stderr, deadlock-prone on non-UTF-8 bytes.
//!
//! Instead, loaders call [`report`] at the same points. The daemon installs a
//! sink (via [`set_sink`]) before a load that serializes each report into a
//! `load_progress` frame on its framed stdout channel — the same structured,
//! UTF-8-safe path used for tokens and other events. When no sink is installed
//! (CLI loads, tests, eval batteries), [`report`] is a cheap no-op.
//!
//! This lives in `hipfire-runtime` because it is the lowest crate reachable by
//! every loader: `hfq.rs` is in this crate, and the arch crates all depend on
//! it. The daemon (which also depends on this crate) is the only installer.

use std::sync::Mutex;

/// Sink signature: `(current, total, phase)`. Kept simple (no error return) so
/// loader call sites stay one-liners next to the existing `eprintln!`.
pub type ProgressFn = dyn Fn(u32, u32, &str) + Send + Sync;

static SINK: Mutex<Option<Box<ProgressFn>>> = Mutex::new(None);

/// Install (or clear, with `None`) the process-wide load-progress sink. The
/// daemon installs one for the duration of a `Load` op and clears it after, so
/// stray reports outside a load are dropped.
pub fn set_sink(sink: Option<Box<ProgressFn>>) {
    if let Ok(mut guard) = SINK.lock() {
        *guard = sink;
    }
}

/// Report load progress. `current`/`total` are phase-relative unit counts
/// (e.g. layer `i+1` of `n`); `phase` is a coarse label such as `"weights"`.
/// No-op when no sink is installed. The `SINK` lock is held across the sink
/// call, which serializes concurrent reports into whole, non-interleaved frames.
pub fn report(current: u32, total: u32, phase: &str) {
    if let Ok(guard) = SINK.lock() {
        if let Some(sink) = guard.as_ref() {
            sink(current, total, phase);
        }
    }
}
