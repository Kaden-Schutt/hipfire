// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! No-GPU aggregation binary for the RDNA kernel Oracle PMC census (task 0c).
//!
//! Folds one or more captured rocprofv3 `--pmc` counter-collection CSVs (one per
//! counter-limited pass) into a single committed per-arch census JSON. **The GPU
//! only ever produces the input CSVs** (rocprofv3 `--pmc` under
//! `profile_standard`); this binary is pure file I/O + parse + JSON emit — no
//! HIP/GPU. It stores RAW COUNTS ONLY (data-not-tags); the kind/revival verdicts
//! are re-derived at load by [`rdna_compute::pmc_census::ArchPmcCensus`].
//!
//! Usage:
//! ```sh
//! cargo run -p rdna-compute --example pmc_census -- \
//!     --arch gfx1201 --perf-level profile_standard \
//!     --csv /tmp/pmc-gfx1201/p1/c_counter_collection.csv \
//!     --csv /tmp/pmc-gfx1201/p2/c_counter_collection.csv \
//!     --csv /tmp/pmc-gfx1201/p3/c_counter_collection.csv \
//!     --out tests/pmc-census/gfx1201.json
//! ```
//! With no `--out`, the census JSON is printed to stdout. The derived verdicts
//! (valid-capture / derived-revived / accumulators-alive) are reported to stderr
//! for the operator — they are NEVER written to the committed JSON.

use rdna_compute::pmc_census::{
    fold_counter_csv_texts, DEFAULT_MIN_NONZERO_FRACTION, REQUIRED_PERF_LEVEL,
};
use std::path::PathBuf;
use std::process::exit;

fn flag_value(argv: &[String], flag: &str) -> Option<String> {
    argv.iter()
        .position(|a| a == flag)
        .and_then(|i| argv.get(i + 1))
        .cloned()
}

/// Collect every value that follows an occurrence of `flag` (supports repeating
/// `--csv <path>` for multi-pass folds).
fn flag_values(argv: &[String], flag: &str) -> Vec<String> {
    let mut out = Vec::new();
    for (i, a) in argv.iter().enumerate() {
        if a == flag {
            if let Some(v) = argv.get(i + 1) {
                out.push(v.clone());
            }
        }
    }
    out
}

fn main() {
    let argv: Vec<String> = std::env::args().collect();

    let arch = match flag_value(&argv, "--arch") {
        Some(a) => a,
        None => {
            eprintln!("pmc_census: --arch <gfxNNNN> is required");
            exit(2);
        }
    };
    let perf_level =
        flag_value(&argv, "--perf-level").unwrap_or_else(|| REQUIRED_PERF_LEVEL.to_string());
    let min_nonzero_fraction = match flag_value(&argv, "--min-nonzero-fraction") {
        Some(s) => match s.parse::<f64>() {
            Ok(v) => v,
            Err(e) => {
                eprintln!("pmc_census: invalid --min-nonzero-fraction {s:?}: {e}");
                exit(2);
            }
        },
        None => DEFAULT_MIN_NONZERO_FRACTION,
    };

    let csv_paths = flag_values(&argv, "--csv");
    if csv_paths.is_empty() {
        eprintln!("pmc_census: at least one --csv <path> is required");
        exit(2);
    }
    let out = flag_value(&argv, "--out").map(PathBuf::from);

    // Read every pass's CSV up front (fail loud on a missing/unreadable file —
    // never silently fold a partial census).
    let mut texts = Vec::with_capacity(csv_paths.len());
    for p in &csv_paths {
        match std::fs::read_to_string(p) {
            Ok(t) => texts.push(t),
            Err(e) => {
                eprintln!("pmc_census: failed to read --csv {p}: {e}");
                exit(1);
            }
        }
    }
    let text_refs: Vec<&str> = texts.iter().map(String::as_str).collect();

    let census = match fold_counter_csv_texts(&arch, &perf_level, min_nonzero_fraction, &text_refs)
    {
        Ok(c) => c,
        Err(e) => {
            eprintln!("pmc_census: census fold failed: {e}");
            exit(1);
        }
    };

    let pretty =
        serde_json::to_string_pretty(&census.to_json()).expect("census JSON always serializes");

    match &out {
        Some(path) => {
            if let Some(parent) = path.parent() {
                if let Err(e) = std::fs::create_dir_all(parent) {
                    eprintln!("pmc_census: failed to create {parent:?}: {e}");
                    exit(1);
                }
            }
            if let Err(e) = std::fs::write(path, format!("{pretty}\n")) {
                eprintln!("pmc_census: failed to write {path:?}: {e}");
                exit(1);
            }
            eprintln!("pmc_census: wrote {path:?}");
        }
        None => println!("{pretty}"),
    }

    // Report the DERIVED verdicts to stderr only — these are re-derived at load
    // and are deliberately NOT part of the committed raw-counts JSON.
    eprintln!(
        "pmc_census: arch={} counters={} valid_capture={} derived_revived={} accumulators_alive={} only_accumulators_revived={}",
        census.arch,
        census.counters.len(),
        census.is_valid_capture(),
        census.derived_revived(),
        census.accumulators_alive(),
        census.only_accumulators_revived(),
    );
}
