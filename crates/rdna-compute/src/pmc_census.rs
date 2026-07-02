// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Per-arch PMC (performance-monitor counter) census + per-cell promotion gate
//! for the RDNA kernel Oracle (spec §4c / §5.5).
//!
//! # Why this module exists
//!
//! On stock RDNA under an `auto`/`high` DPM governor the perfmon clock is
//! gated, so rocprofv3 `--pmc` reads ZERO for the *derived* roofline metrics
//! (`FetchSize`/`WriteSize`/`VALUBusy`/`MemUnitBusy`/`OccupancyPercent`) even
//! though the *raw accumulators* (`SQ_BUSY_CYCLES`/`Wavefronts`) still tick.
//! Forcing `profile_standard` (the `STABLE_STD` perf level) un-gates the
//! perfmon clock and REVIVES the derived counters — but this is **per arch**:
//! a counter that revives on gfx1201 may be entirely absent from gfx1100's
//! metrics DB. This module measures, per arch, *which* derived counters
//! actually came back to life under `profile_standard`, and **never transfers
//! that verdict across arches**.
//!
//! # Provenance discipline (data-not-tags, withheld-never-faked)
//!
//! The GPU only ever produces the raw rocprofv3 `--pmc` CSV. Everything here is
//! pure, no-GPU, and unit-tested offline. The committed per-arch census JSON
//! stores **raw counts only** (`rows`/`nonzero_rows`/`value_sum` per counter) —
//! the `CounterKind`, `nonzero_fraction`, and `derived_revived` verdicts are
//! **re-derived at load**, never persisted (mirrors `chip_profile.rs`). A cell
//! that fails the promotion gate is `WITHHELD{reason}` and its `gate_value()`
//! is `None` — a withheld cell NEVER fabricates a value.
//!
//! # Scope boundary
//!
//! This module is a self-contained decision surface. It does NOT rewire
//! `roofline::Roofline::analyze` or `kernel_perf_instrument` to consume the
//! census; dynamic-domain evidence flows IN at data-population time via
//! [`CellEvidence`] (an input), not as a build dependency.

// ---------------------------------------------------------------------------
// §4c — counter classification
// ---------------------------------------------------------------------------

/// The five roofline-relevant DERIVED metrics. These are computed by
/// rocprofv3 from perfmon hardware events and read ZERO when the perfmon clock
/// is gated (auto/high DPM) — they revive only under `profile_standard`
/// ([`REQUIRED_PERF_LEVEL`]). Whether each actually revives is measured
/// **per arch** (a counter may be absent from a given arch's metrics DB).
pub const REQUIRED_DERIVED_COUNTERS: [&str; 5] = [
    "FetchSize",
    "WriteSize",
    "VALUBusy",
    "MemUnitBusy",
    "OccupancyPercent",
];

/// Raw hardware accumulators that keep ticking regardless of DPM governor —
/// the "working controls" that prove the capture itself ran (vs. a dead
/// profiling session). If even these are dead, the CSV is not a real capture.
pub const RAW_ACCUMULATOR_COUNTERS: [&str; 2] = ["SQ_BUSY_CYCLES", "Wavefronts"];

/// How a rocprofv3 `--pmc` counter name classifies for census purposes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CounterKind {
    /// A perfmon-clock-gated derived roofline metric ([`REQUIRED_DERIVED_COUNTERS`]).
    Derived,
    /// A raw hardware accumulator control ([`RAW_ACCUMULATOR_COUNTERS`]).
    Accumulator,
    /// A counter we do not model — never guessed into either bucket.
    Unknown,
}

/// Classify a counter name (case-insensitive). Unrecognized names are
/// [`CounterKind::Unknown`] — we never coerce an unknown counter into the
/// derived or accumulator bucket.
pub fn classify_counter(name: &str) -> CounterKind {
    if REQUIRED_DERIVED_COUNTERS
        .iter()
        .any(|c| c.eq_ignore_ascii_case(name))
    {
        CounterKind::Derived
    } else if RAW_ACCUMULATOR_COUNTERS
        .iter()
        .any(|c| c.eq_ignore_ascii_case(name))
    {
        CounterKind::Accumulator
    } else {
        CounterKind::Unknown
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn required_derived_set_is_the_five_roofline_counters() {
        // The five roofline-relevant DERIVED metrics that gate on the perfmon
        // clock (revive only under `profile_standard`).
        assert_eq!(REQUIRED_DERIVED_COUNTERS.len(), 5);
        for c in [
            "FetchSize",
            "WriteSize",
            "VALUBusy",
            "MemUnitBusy",
            "OccupancyPercent",
        ] {
            assert!(
                REQUIRED_DERIVED_COUNTERS.contains(&c),
                "REQUIRED_DERIVED_COUNTERS missing {c}"
            );
        }
    }

    #[test]
    fn classify_counter_maps_derived_accumulator_unknown() {
        // Derived (case-insensitive).
        assert_eq!(classify_counter("FetchSize"), CounterKind::Derived);
        assert_eq!(classify_counter("fetchsize"), CounterKind::Derived);
        assert_eq!(classify_counter("OCCUPANCYPERCENT"), CounterKind::Derived);
        // Raw accumulators (case-insensitive).
        assert_eq!(classify_counter("SQ_BUSY_CYCLES"), CounterKind::Accumulator);
        assert_eq!(classify_counter("wavefronts"), CounterKind::Accumulator);
        // Anything else is Unknown — we never guess a kind we don't recognize.
        assert_eq!(classify_counter("GRBM_GUI_ACTIVE"), CounterKind::Unknown);
        assert_eq!(classify_counter(""), CounterKind::Unknown);
    }
}
