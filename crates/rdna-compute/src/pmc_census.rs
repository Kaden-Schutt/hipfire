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

// ---------------------------------------------------------------------------
// §4c — per-arch census from a rocprofv3 --pmc CSV
// ---------------------------------------------------------------------------

use std::collections::BTreeMap;

/// A counter is considered ALIVE (revived) when it is nonzero in at least this
/// fraction of its rows. A perfmon-gated derived counter reads exactly zero on
/// every row, so any sane threshold rejects it; 0.5 tolerates a stray
/// zero-dispatch without declaring a live counter dead.
pub const DEFAULT_MIN_NONZERO_FRACTION: f64 = 0.5;

/// The DPM/perf level a capture MUST have run under for its DERIVED counters to
/// be trustworthy. Under `auto`/`high` the perfmon clock is gated and the
/// derived metrics read a fake zero (see [`crate::profile_rocprof`]).
pub const REQUIRED_PERF_LEVEL: &str = "profile_standard";

/// Raw per-counter tallies as captured — **the only thing persisted to disk**
/// (data-not-tags: kind/fraction/alive are all re-derived at load).
#[derive(Debug, Clone, PartialEq)]
pub struct CounterCounts {
    /// Number of CSV rows (dispatches) that reported this counter.
    pub rows: u64,
    /// How many of those rows carried a strictly-nonzero value.
    pub nonzero_rows: u64,
    /// Sum of the counter values across all rows (diagnostic; not a verdict).
    pub value_sum: f64,
}

/// A per-counter verdict, DERIVED (never persisted) from the raw counts + the
/// census min-nonzero-fraction threshold.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CounterVerdict {
    /// Derived / Accumulator / Unknown (from [`classify_counter`]).
    pub kind: CounterKind,
    /// `nonzero_rows / rows` (0.0 for an empty counter).
    pub nonzero_fraction: f64,
    /// `nonzero_fraction >= min_nonzero_fraction` — the counter revived.
    pub alive: bool,
}

/// Compute a per-counter verdict from raw counts. Free function so callers can
/// classify a counter independently of a full census.
pub fn counter_verdict(
    name: &str,
    counts: &CounterCounts,
    min_nonzero_fraction: f64,
) -> CounterVerdict {
    let nonzero_fraction = if counts.rows == 0 {
        0.0
    } else {
        counts.nonzero_rows as f64 / counts.rows as f64
    };
    CounterVerdict {
        kind: classify_counter(name),
        nonzero_fraction,
        alive: nonzero_fraction >= min_nonzero_fraction,
    }
}

/// A single arch's PMC census: the raw per-counter tallies captured under a
/// stated perf level. Verdicts (kind/revival) are re-derived, never stored.
#[derive(Debug, Clone, PartialEq)]
pub struct ArchPmcCensus {
    /// The arch this census belongs to (e.g. `"gfx1201"`). NEVER transferred
    /// to another arch — a verdict here says nothing about any other arch.
    pub arch: String,
    /// The DPM/perf level the capture ran under (compared to
    /// [`REQUIRED_PERF_LEVEL`] in [`ArchPmcCensus::is_valid_capture`]).
    pub perf_level: String,
    /// The revival threshold applied to every counter.
    pub min_nonzero_fraction: f64,
    /// Raw per-counter tallies, keyed by the counter name as captured.
    pub counters: BTreeMap<String, CounterCounts>,
}

impl ArchPmcCensus {
    /// Case-insensitive lookup of a counter's raw counts.
    fn counts_ci(&self, name: &str) -> Option<&CounterCounts> {
        // Fast path: exact key. Fall back to a case-insensitive scan.
        if let Some(c) = self.counters.get(name) {
            return Some(c);
        }
        self.counters
            .iter()
            .find(|(k, _)| k.eq_ignore_ascii_case(name))
            .map(|(_, v)| v)
    }

    /// The DERIVED-vs-revival verdict for a single counter, or `None` if the
    /// census never captured it.
    pub fn verdict(&self, name: &str) -> Option<CounterVerdict> {
        self.counts_ci(name)
            .map(|c| counter_verdict(name, c, self.min_nonzero_fraction))
    }
}

/// Parse a rocprofv3 `--pmc` counter-collection CSV into a per-arch census.
///
/// The header MUST end in `Counter_Name,Counter_Value` (case-insensitive) —
/// a kernel-stats CSV (`Name,Calls,...`) is rejected rather than mis-parsed.
/// Each data row is right-anchored on its final two columns so a mangled
/// kernel name full of commas cannot corrupt the counter parse. Counts are
/// aggregated per counter name into a `BTreeMap` (deterministic ordering).
pub fn census_from_counter_csv_text(
    arch: &str,
    perf_level: &str,
    min_nonzero_fraction: f64,
    text: &str,
) -> Result<ArchPmcCensus, String> {
    let mut lines = text.lines();
    let header = match lines.next() {
        Some(h) => h.trim().to_ascii_lowercase(),
        None => return Err("PMC counter CSV is empty".to_string()),
    };
    // Right-anchoring contract: the final two columns are the counter name and
    // its value. Reject anything that is not a --pmc counter-collection CSV.
    if !header.ends_with("counter_name,counter_value") {
        return Err(format!(
            "PMC CSV header does not end in Counter_Name,Counter_Value: {header:?}"
        ));
    }

    let mut counters: BTreeMap<String, CounterCounts> = BTreeMap::new();
    for (line_no, raw) in lines.enumerate() {
        let line = raw.trim();
        if line.is_empty() {
            continue;
        }
        let parts: Vec<&str> = line.split(',').collect();
        // Need at least a kernel-name column + the two counter columns.
        if parts.len() < 3 {
            eprintln!(
                "WARN PMC CSV line {}: expected >=3 columns, got {}; skipping",
                line_no + 2,
                parts.len()
            );
            continue;
        }
        let n = parts.len();
        let counter_name = parts[n - 2].trim();
        let value_str = parts[n - 1].trim();
        if counter_name.is_empty() {
            continue;
        }
        let value: f64 = match value_str.parse() {
            Ok(v) => v,
            Err(e) => {
                eprintln!(
                    "WARN PMC CSV line {}: Counter_Value parse error ({value_str:?}): {e}; skipping",
                    line_no + 2
                );
                continue;
            }
        };
        let entry = counters
            .entry(counter_name.to_string())
            .or_insert(CounterCounts {
                rows: 0,
                nonzero_rows: 0,
                value_sum: 0.0,
            });
        entry.rows += 1;
        if value != 0.0 {
            entry.nonzero_rows += 1;
        }
        entry.value_sum += value;
    }

    Ok(ArchPmcCensus {
        arch: arch.to_string(),
        perf_level: perf_level.to_string(),
        min_nonzero_fraction,
        counters,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A `profile_standard` capture where every DERIVED metric revived
    /// (nonzero) alongside the raw-accumulator controls.
    const CSV_ALL_REVIVED: &str = "\
Kernel_Name,Counter_Name,Counter_Value
gemm_q8_0_wmma,FetchSize,1048576
gemm_q8_0_wmma,WriteSize,524288
gemm_q8_0_wmma,VALUBusy,42.5
gemm_q8_0_wmma,MemUnitBusy,30.0
gemm_q8_0_wmma,OccupancyPercent,75.0
gemm_q8_0_wmma,SQ_BUSY_CYCLES,999999
gemm_q8_0_wmma,Wavefronts,320
";

    /// A capture where the DERIVED metrics stayed DEAD (perfmon-gated → zero)
    /// but the raw accumulators still ticked — the "only accumulators" arm.
    const CSV_ONLY_ACCUMULATORS: &str = "\
Kernel_Name,Counter_Name,Counter_Value
gemm_q8_0_wmma,FetchSize,0
gemm_q8_0_wmma,WriteSize,0
gemm_q8_0_wmma,VALUBusy,0
gemm_q8_0_wmma,MemUnitBusy,0
gemm_q8_0_wmma,OccupancyPercent,0
gemm_q8_0_wmma,SQ_BUSY_CYCLES,999999
gemm_q8_0_wmma,Wavefronts,320
";

    #[test]
    fn census_parses_counts_classifies_and_derives_revival() {
        let c = census_from_counter_csv_text(
            "gfx1201",
            REQUIRED_PERF_LEVEL,
            DEFAULT_MIN_NONZERO_FRACTION,
            CSV_ALL_REVIVED,
        )
        .expect("parse ok");
        assert_eq!(c.arch, "gfx1201");
        assert_eq!(c.perf_level, REQUIRED_PERF_LEVEL);
        assert_eq!(c.min_nonzero_fraction, DEFAULT_MIN_NONZERO_FRACTION);
        // All 7 counters present (5 derived + 2 accumulator controls).
        assert_eq!(c.counters.len(), 7);

        // Per-counter raw counts.
        let fetch = c.counters.get("FetchSize").expect("FetchSize present");
        assert_eq!(fetch.rows, 1);
        assert_eq!(fetch.nonzero_rows, 1);
        assert_eq!(fetch.value_sum, 1_048_576.0);

        // Per-counter verdict: classified + revival (alive) derived, not stored.
        let v = c.verdict("FetchSize").expect("verdict");
        assert_eq!(v.kind, CounterKind::Derived);
        assert!((v.nonzero_fraction - 1.0).abs() < 1e-9);
        assert!(v.alive, "FetchSize is nonzero in every row → revived");

        let acc = c.verdict("SQ_BUSY_CYCLES").expect("verdict");
        assert_eq!(acc.kind, CounterKind::Accumulator);
        assert!(acc.alive);
    }

    #[test]
    fn census_rejects_header_not_ending_in_counter_name_value() {
        // A kernel-stats header (ends in the timing columns), not a --pmc
        // counter-collection header → hard reject, never silently mis-parse.
        let bad = "\
Name,Calls,TotalDurationNs,AverageNs,Percentage,MinNs,MaxNs,StdDev
gemm_q8,1,1000,1000,100.0,1000,1000,0
";
        let r = census_from_counter_csv_text("gfx1201", REQUIRED_PERF_LEVEL, 0.5, bad);
        assert!(r.is_err(), "must reject a non-counter-collection header");
    }

    #[test]
    fn census_right_anchors_past_commas_in_kernel_name() {
        // Mangled kernel names carry commas; the parser must right-anchor on
        // the final two columns (Counter_Name,Counter_Value) and NOT let the
        // name's commas corrupt the counter parse.
        let csv = "\
Kernel_Name,Counter_Name,Counter_Value
void gemm<32, 64, false>(float*, int),FetchSize,2048
void gemm<32, 64, false>(float*, int),Wavefronts,64
";
        let c =
            census_from_counter_csv_text("gfx1100", REQUIRED_PERF_LEVEL, 0.5, csv).expect("parse");
        let fetch = c.counters.get("FetchSize").expect("FetchSize parsed");
        assert_eq!(fetch.rows, 1);
        assert_eq!(fetch.value_sum, 2048.0);
        let waves = c.counters.get("Wavefronts").expect("Wavefronts parsed");
        assert_eq!(waves.value_sum, 64.0);
    }

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
