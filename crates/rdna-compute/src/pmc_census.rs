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

    /// A single named counter is alive iff it was captured AND revived (nonzero
    /// in at least `min_nonzero_fraction` of its rows).
    fn counter_alive(&self, name: &str) -> bool {
        self.verdict(name).map(|v| v.alive).unwrap_or(false)
    }

    /// True iff this capture ran under [`REQUIRED_PERF_LEVEL`]
    /// (`profile_standard`). Under any other governor the perfmon clock may be
    /// gated, so the DERIVED counters cannot be trusted even when nonzero.
    pub fn is_valid_capture(&self) -> bool {
        self.perf_level == REQUIRED_PERF_LEVEL
    }

    /// True iff EVERY [`REQUIRED_DERIVED_COUNTERS`] entry was captured and
    /// revived AND the capture perf level is valid. Any of: invalid capture,
    /// a missing required derived counter, or a dead (perfmon-gated) one blocks
    /// revival — WITHHELD, never a fabricated pass.
    pub fn derived_revived(&self) -> bool {
        self.is_valid_capture()
            && REQUIRED_DERIVED_COUNTERS
                .iter()
                .all(|c| self.counter_alive(c))
    }

    /// True iff EVERY raw accumulator control was captured and revived. These
    /// tick regardless of DPM governor, so this does NOT require a valid
    /// capture perf level — it is the "the profiling session actually ran"
    /// control.
    pub fn accumulators_alive(&self) -> bool {
        RAW_ACCUMULATOR_COUNTERS
            .iter()
            .all(|c| self.counter_alive(c))
    }

    /// True iff the working controls survived but the derived metrics did not —
    /// the diagnostic "profile_standard un-gated nothing new on this arch" arm.
    pub fn only_accumulators_revived(&self) -> bool {
        self.accumulators_alive() && !self.derived_revived()
    }

    /// Serialize to the committed on-disk form: **raw counts + capture context
    /// only**. The `CounterKind`, `nonzero_fraction`, and revival verdicts are
    /// deliberately NOT written — they are re-derived on load (data-not-tags),
    /// so a future threshold or classification change re-derives cleanly rather
    /// than reading a stale baked-in tag.
    pub fn to_json(&self) -> serde_json::Value {
        let mut counters = serde_json::Map::new();
        for (name, c) in &self.counters {
            counters.insert(
                name.clone(),
                serde_json::json!({
                    "rows": c.rows,
                    "nonzero_rows": c.nonzero_rows,
                    "value_sum": c.value_sum,
                }),
            );
        }
        serde_json::json!({
            "arch": self.arch,
            "perf_level": self.perf_level,
            "min_nonzero_fraction": self.min_nonzero_fraction,
            "counters": serde_json::Value::Object(counters),
        })
    }

    /// Parse the committed on-disk form. **Fail-loud** on any missing/malformed
    /// field — never silently default a raw count. Verdicts are re-derived from
    /// the loaded raw counts, never read from disk.
    pub fn from_json(v: &serde_json::Value) -> Result<Self, String> {
        let arch = v["arch"]
            .as_str()
            .ok_or_else(|| "PMC census JSON missing string field 'arch'".to_string())?
            .to_string();
        let perf_level = v["perf_level"]
            .as_str()
            .ok_or_else(|| "PMC census JSON missing string field 'perf_level'".to_string())?
            .to_string();
        let min_nonzero_fraction = v["min_nonzero_fraction"].as_f64().ok_or_else(|| {
            "PMC census JSON missing number field 'min_nonzero_fraction'".to_string()
        })?;
        let counters_obj = v["counters"]
            .as_object()
            .ok_or_else(|| "PMC census JSON missing object field 'counters'".to_string())?;

        let mut counters = BTreeMap::new();
        for (name, cv) in counters_obj {
            let rows = cv["rows"].as_u64().ok_or_else(|| {
                format!("PMC census counter '{name}' missing integer field 'rows'")
            })?;
            let nonzero_rows = cv["nonzero_rows"].as_u64().ok_or_else(|| {
                format!("PMC census counter '{name}' missing integer field 'nonzero_rows'")
            })?;
            let value_sum = cv["value_sum"].as_f64().ok_or_else(|| {
                format!("PMC census counter '{name}' missing number field 'value_sum'")
            })?;
            counters.insert(
                name.clone(),
                CounterCounts {
                    rows,
                    nonzero_rows,
                    value_sum,
                },
            );
        }

        Ok(ArchPmcCensus {
            arch,
            perf_level,
            min_nonzero_fraction,
            counters,
        })
    }

    /// Relative path (from the workspace root) of the committed census JSON for
    /// `arch`, mirroring `chip_profile::ChipProfile::committed_path`.
    pub fn committed_path(arch: &str) -> std::path::PathBuf {
        std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../tests/pmc-census")
            .join(format!("{arch}.json"))
    }

    /// Load a census from a JSON file (fail-loud on read/parse/schema error).
    pub fn load(path: impl AsRef<std::path::Path>) -> Result<Self, String> {
        let path = path.as_ref();
        let text = std::fs::read_to_string(path)
            .map_err(|e| format!("ArchPmcCensus::load: failed to read {path:?}: {e}"))?;
        let value: serde_json::Value = serde_json::from_str(&text)
            .map_err(|e| format!("ArchPmcCensus::load: invalid JSON in {path:?}: {e}"))?;
        Self::from_json(&value)
    }

    /// Load the committed reference census for `arch` from
    /// `tests/pmc-census/<arch>.json`.
    pub fn load_committed(arch: &str) -> Result<Self, String> {
        Self::load(Self::committed_path(arch))
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
    fn all_revived_arm() {
        let c = census_from_counter_csv_text("gfx1201", REQUIRED_PERF_LEVEL, 0.5, CSV_ALL_REVIVED)
            .unwrap();
        assert!(c.is_valid_capture());
        assert!(c.derived_revived(), "all 5 derived counters nonzero");
        assert!(c.accumulators_alive());
        assert!(!c.only_accumulators_revived());
    }

    #[test]
    fn only_accumulators_arm() {
        let c = census_from_counter_csv_text(
            "gfx1201",
            REQUIRED_PERF_LEVEL,
            0.5,
            CSV_ONLY_ACCUMULATORS,
        )
        .unwrap();
        assert!(c.is_valid_capture());
        assert!(
            !c.derived_revived(),
            "derived counters all read zero → dead"
        );
        assert!(c.accumulators_alive(), "raw controls still ticked");
        assert!(
            c.only_accumulators_revived(),
            "accumulators alive but derived withheld"
        );
    }

    #[test]
    fn missing_required_derived_counter_blocks_revival() {
        // VALUBusy is absent entirely (e.g. not in this arch's metrics DB).
        // A missing required derived counter blocks revival even though the
        // present ones are nonzero.
        let csv = "\
Kernel_Name,Counter_Name,Counter_Value
k,FetchSize,1024
k,WriteSize,512
k,MemUnitBusy,30.0
k,OccupancyPercent,75.0
k,SQ_BUSY_CYCLES,999
k,Wavefronts,64
";
        let c = census_from_counter_csv_text("gfx1100", REQUIRED_PERF_LEVEL, 0.5, csv).unwrap();
        assert!(c.is_valid_capture());
        assert!(
            !c.derived_revived(),
            "VALUBusy absent → required-derived set incomplete → not revived"
        );
        assert!(c.accumulators_alive());
        assert!(c.only_accumulators_revived());
    }

    #[test]
    fn invalid_perf_level_forces_derived_withheld_even_if_nonzero() {
        // Same all-nonzero CSV, but captured under `auto` (perfmon gated).
        // Even though every derived counter reads nonzero, an invalid capture
        // perf level WITHHOLDS the derived verdict — we do not trust a derived
        // number that could be a perfmon-clock artifact.
        let c = census_from_counter_csv_text("gfx1201", "auto", 0.5, CSV_ALL_REVIVED).unwrap();
        assert!(!c.is_valid_capture(), "auto is not profile_standard");
        assert!(
            !c.derived_revived(),
            "invalid perf level withholds derived revival regardless of values"
        );
        // Raw accumulators are governor-independent — still a working control.
        assert!(c.accumulators_alive());
    }

    #[test]
    fn census_json_is_raw_counts_only_and_round_trips() {
        let c = census_from_counter_csv_text("gfx1201", REQUIRED_PERF_LEVEL, 0.5, CSV_ALL_REVIVED)
            .unwrap();
        let json = c.to_json();
        let text = serde_json::to_string_pretty(&json).unwrap();

        // data-not-tags: the DERIVED verdict fields must NOT be on disk.
        assert!(
            !text.contains("revived"),
            "revival is derived at load, never persisted:\n{text}"
        );
        assert!(
            !text.contains("\"kind\""),
            "counter kind is derived at load, never persisted:\n{text}"
        );
        // The per-counter DERIVED fraction verdict must NOT be persisted. Note
        // `min_nonzero_fraction` (the THRESHOLD / capture context) IS persisted;
        // match the quoted verdict field name so we don't false-positive on it.
        assert!(
            !text.contains("\"nonzero_fraction\""),
            "per-counter fraction is derived at load, never persisted:\n{text}"
        );
        assert!(
            !text.contains("\"alive\""),
            "revival flag is derived at load, never persisted:\n{text}"
        );
        // Raw counts + capture context ARE on disk.
        assert!(text.contains("\"rows\""));
        assert!(text.contains("\"nonzero_rows\""));
        assert!(text.contains("\"value_sum\""));
        assert!(text.contains("\"min_nonzero_fraction\""));
        assert!(text.contains("profile_standard"));

        // Round-trips byte-for-byte back to the same census, and verdicts
        // re-derive identically.
        let back = ArchPmcCensus::from_json(&json).expect("from_json ok");
        assert_eq!(back, c);
        assert!(back.derived_revived());
    }

    #[test]
    fn from_json_fails_loud_on_missing_field() {
        let c = census_from_counter_csv_text("gfx1201", REQUIRED_PERF_LEVEL, 0.5, CSV_ALL_REVIVED)
            .unwrap();
        // Drop a required top-level field.
        let mut v = c.to_json();
        v.as_object_mut().unwrap().remove("perf_level");
        assert!(
            ArchPmcCensus::from_json(&v).is_err(),
            "missing perf_level must fail loud, not default"
        );

        // Drop a required per-counter field.
        let mut v2 = c.to_json();
        v2["counters"]["FetchSize"]
            .as_object_mut()
            .unwrap()
            .remove("nonzero_rows");
        assert!(
            ArchPmcCensus::from_json(&v2).is_err(),
            "missing per-counter nonzero_rows must fail loud"
        );
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
