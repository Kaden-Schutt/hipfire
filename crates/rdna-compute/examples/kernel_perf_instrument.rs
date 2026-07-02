// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! `kernel_perf_instrument` — the static (NO-GPU) half of the Phase A
//! chip-tied kernel-perf instrument
//! (`docs/superpowers/plans/2026-07-01-gfx1201-phaseA-perf-instrument.md`).
//!
//! `--self-check` runs the full static pipeline — `IsaHistogram` + static
//! `Roofline` (`achieved_bw: None`) + a ledger-diff against the committed
//! baseline + a JIT module-name collision scan — over a directory of
//! committed `.hsaco` fixtures, with NO GPU/ROCm-runtime access required
//! (only the offline LLVM tools, `clang-offload-bundler`/`llvm-objdump`/
//! `llvm-readelf`).
//!
//! `--diff` re-runs the same static pipeline over a (possibly modified)
//! `--current-dir` and prints ONLY the ledger deltas vs the committed
//! baseline — the fast path for "did anything in this kernel change
//! shape".
//!
//! The dynamic half (rocprof-measured `achieved_bw`, live occupancy,
//! surface-map/Amdahl render) is Task 9 (GPU-required) — deliberately
//! out of scope here.
//!
//! Usage:
//!   cargo run -p rdna-compute --example kernel_perf_instrument -- --self-check
//!   cargo run -p rdna-compute --example kernel_perf_instrument -- \
//!       --self-check --arch gfx1201 --fixtures-dir tests/kernel-fixtures/gfx1201 \
//!       --ledger tests/kernel-ledger/gfx1201.jsonl
//!   cargo run -p rdna-compute --example kernel_perf_instrument -- \
//!       --diff --current-dir /path/to/modified/fixtures

use rdna_compute::chip_profile::ChipProfile;
use rdna_compute::isa_histogram::IsaHistogram;
use rdna_compute::kernel_ledger::{
    module_collision_scan, KernelLedger, LedgerDelta, LedgerKey, LedgerRow, Reproducer,
};
use rdna_compute::roofline::Roofline;
use std::collections::HashMap;
use std::path::{Path, PathBuf};

struct Args {
    mode: Mode,
    arch: String,
    fixtures_dir: PathBuf,
    ledger_path: PathBuf,
}

enum Mode {
    SelfCheck,
    Diff,
}

fn parse_args() -> Args {
    let argv: Vec<String> = std::env::args().collect();
    let mode = if argv.iter().any(|a| a == "--diff") {
        Mode::Diff
    } else {
        Mode::SelfCheck // --self-check is the default when neither flag is given
    };
    let arch = flag_value(&argv, "--arch").unwrap_or_else(|| "gfx1201".to_string());

    // `--current-dir` is the `--diff` mode's input directory; `--fixtures-dir`
    // is the `--self-check` mode's. Both fall back to the same committed
    // no-GPU fixture set for `arch` when unspecified.
    let default_dir = workspace_path(&format!("tests/kernel-fixtures/{arch}"));
    let fixtures_dir = flag_value(&argv, "--current-dir")
        .or_else(|| flag_value(&argv, "--fixtures-dir"))
        .map(PathBuf::from)
        .unwrap_or(default_dir);

    let default_ledger = workspace_path(&format!("tests/kernel-ledger/{arch}.jsonl"));
    let ledger_path = flag_value(&argv, "--ledger")
        .map(PathBuf::from)
        .unwrap_or(default_ledger);

    Args {
        mode,
        arch,
        fixtures_dir,
        ledger_path,
    }
}

fn flag_value(argv: &[String], flag: &str) -> Option<String> {
    argv.iter()
        .position(|a| a == flag)
        .and_then(|i| argv.get(i + 1))
        .cloned()
}

/// Resolve a path relative to the workspace root (two levels up from this
/// crate's `CARGO_MANIFEST_DIR`) — independent of the `cargo run` caller's
/// CWD, matching how `ChipProfile`/`LedgerRow`'s own committed-path
/// resolution works.
fn workspace_path(rel: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .join(rel)
}

/// One fixture's fully-computed static measurement.
struct FixtureMeasurement {
    path: PathBuf,
    kernel: String,
    row: LedgerRow,
    roofline: Roofline,
}

/// Resolve the full shape-keyed [`LedgerKey`] for a fixture. The kernel
/// symbol name (`path.file_stem()`) is the identity anchor — it's
/// filesystem-guaranteed unique within the scanned directory, unlike the
/// ELF-embedded `.kd` symbol `module_collision_scan` checks (which CAN
/// legitimately collide across distinct files — that's the bug it exists
/// to catch). When a committed baseline shares that kernel identity, its
/// `quant`/`workload`/`phase`/`shape_bucket` are the real, curated
/// serving-context values (not recoverable from raw ISA bytes alone) —
/// adopt them wholesale so `KernelLedger::find`/`diff` run a genuine
/// FULL-key match instead of the kernel-name-only shortcut this replaces.
/// A fixture with no committed baseline (e.g. the deliberate
/// `collision_probe_*` demo pair) has no known serving-context identity to
/// recover, so its `quant`/`workload` genuinely are "unknown" — and since
/// no baseline exists, the diff is skipped for it regardless.
fn resolve_ledger_key(kernel: &str, arch: &str, ledger: &KernelLedger) -> LedgerKey {
    if let Some(row) = ledger.rows.iter().find(|r| r.key.kernel == kernel) {
        return row.key.clone();
    }
    LedgerKey {
        arch: arch.to_string(),
        kernel: kernel.to_string(),
        shape_bucket: "decode_gemv".to_string(),
        quant: "unknown".to_string(),
        workload: "unknown".to_string(),
        phase: "decode".to_string(),
    }
}

fn measure_fixture(
    path: &Path,
    arch: &str,
    chip: &ChipProfile,
    ledger: &KernelLedger,
) -> Result<FixtureMeasurement, String> {
    let kernel = path
        .file_stem()
        .and_then(|s| s.to_str())
        .ok_or_else(|| format!("{path:?}: could not derive a module name from the file stem"))?
        .to_string();

    let hist = IsaHistogram::from_hsaco(path, arch)?;
    let mut map = HashMap::new();
    map.insert(kernel.clone(), path.to_path_buf());
    let (_cap, profiles) = rdna_compute::profiler::profile_kernels(arch, 0, &map);
    let kprofile = profiles
        .into_iter()
        .next()
        .ok_or_else(|| format!("{path:?}: profiler could not parse a kernel descriptor"))?;

    let roofline = Roofline::analyze(&hist, &kprofile, chip, None);

    let key = resolve_ledger_key(&kernel, arch, ledger);
    let mut row = LedgerRow::from_fixture(
        key,
        &hist,
        &kprofile,
        Reproducer {
            cmd: "kernel_perf_instrument --self-check".to_string(),
            fixture_path: Some(path.display().to_string()),
            prompt_md5: None,
        },
    );
    row.bound_class = roofline.binding;

    Ok(FixtureMeasurement {
        path: path.to_path_buf(),
        kernel,
        row,
        roofline,
    })
}

fn list_hsaco_files(dir: &Path) -> Result<Vec<PathBuf>, String> {
    let mut files: Vec<PathBuf> = std::fs::read_dir(dir)
        .map_err(|e| format!("failed to read {dir:?}: {e}"))?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().and_then(|s| s.to_str()) == Some("hsaco"))
        .collect();
    files.sort();
    Ok(files)
}

/// Machine-readable JSONL line for one fixture's measurement.
fn emit_jsonl(m: &FixtureMeasurement, ledger_key_found: bool, deltas: &[LedgerDelta]) {
    let json = serde_json::json!({
        "kernel": m.kernel,
        "path": m.path.display().to_string(),
        "vgpr": m.row.vgpr,
        "sgpr": m.row.sgpr,
        "lds": m.row.lds,
        "scratch": m.row.scratch,
        "isa_fingerprint": m.row.isa_fingerprint,
        "bound_class": format!("{:?}", m.roofline.binding),
        "verdict": m.roofline.verdict,
        "trust_score": m.roofline.trust_score,
        "has_baseline": ledger_key_found,
        "deltas": deltas.iter().map(|d| format!("{d:?}")).collect::<Vec<_>>(),
    });
    println!("{json}");
}

fn main() {
    let args = parse_args();

    let chip = match ChipProfile::load_committed(&args.arch) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("kernel_perf_instrument: failed to load committed ChipProfile: {e}");
            std::process::exit(1);
        }
    };

    let ledger = match KernelLedger::load(&args.ledger_path) {
        Ok(l) => l,
        Err(e) => {
            eprintln!(
                "kernel_perf_instrument: WARN: no usable ledger baseline at {:?} ({e}) — \
                 every fixture will report as unbaselined",
                args.ledger_path
            );
            KernelLedger { rows: Vec::new() }
        }
    };

    let files = match list_hsaco_files(&args.fixtures_dir) {
        Ok(f) => f,
        Err(e) => {
            eprintln!("kernel_perf_instrument: {e}");
            std::process::exit(1);
        }
    };
    if files.is_empty() {
        eprintln!(
            "kernel_perf_instrument: no .hsaco files found in {:?}",
            args.fixtures_dir
        );
        std::process::exit(1);
    }

    let mut hard_regressions = 0usize;
    let mut measurements = Vec::new();
    for path in &files {
        match measure_fixture(path, &args.arch, &chip, &ledger) {
            Ok(m) => measurements.push(m),
            Err(e) => {
                eprintln!("kernel_perf_instrument: FAIL measuring {path:?}: {e}");
                std::process::exit(1);
            }
        }
    }

    for m in &measurements {
        // Full shape-key match (arch/kernel/quant/workload/phase/shape_bucket)
        // — `measure_fixture` already adopted the matching committed row's
        // real key via `resolve_ledger_key`, so this is a genuine
        // `KernelLedger::find` lookup, not a kernel-name-only shortcut.
        let baseline = ledger.find(&m.row.key);
        let deltas = match baseline {
            Some(committed) => KernelLedger::diff(committed, &m.row),
            None => Vec::new(),
        };
        hard_regressions += deltas
            .iter()
            .filter(|d| matches!(d, LedgerDelta::RegressionHard { .. }))
            .count();

        if matches!(args.mode, Mode::SelfCheck) {
            emit_jsonl(m, baseline.is_some(), &deltas);
        } else {
            // --diff: only the deltas, one JSONL line per fixture that HAS
            // a baseline row (nothing to diff for an unbaselined fixture).
            if baseline.is_some() {
                let json = serde_json::json!({
                    "kernel": m.kernel,
                    "deltas": deltas.iter().map(|d| format!("{d:?}")).collect::<Vec<_>>(),
                });
                println!("{json}");
            }
        }
    }

    let collisions = match module_collision_scan(&args.fixtures_dir, &args.arch) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("kernel_perf_instrument: module_collision_scan failed: {e}");
            std::process::exit(1);
        }
    };

    // Human TL;DR.
    eprintln!(
        "\n== kernel_perf_instrument {} ({}) ==",
        match args.mode {
            Mode::SelfCheck => "--self-check",
            Mode::Diff => "--diff",
        },
        args.arch
    );
    eprintln!(
        "fixtures scanned: {} ({:?})",
        measurements.len(),
        args.fixtures_dir
    );
    eprintln!(
        "hard regressions (static-field EXACT mismatch vs committed ledger): {hard_regressions}"
    );
    if collisions.is_empty() {
        eprintln!("module-name collisions: 0");
    } else {
        eprintln!(
            "module-name collisions: {} — JIT module-name collision guard would fire on these \
             at runtime (see reference_kernel_module_cache_collision):",
            collisions.len()
        );
        for c in &collisions {
            eprintln!(
                "  FLAGGED: kernel symbol '{}' shared by {} files with DIFFERING bytes: {:?}",
                c.module_name,
                c.files.len(),
                c.files
            );
        }
    }

    // "Fail loud" per the plan's Step 6: a real static-shape regression
    // against the committed ledger baseline is a hard stop. The
    // deliberately-committed collision-demo fixture pair is reported
    // above (FLAGGED) but does not by itself fail the run — the fixture
    // set intentionally ships one so the detector's own logic is
    // exercised every time `--self-check` runs; a REAL collision found
    // against a production kernel cache dir should be treated as fatal
    // by the caller (non-empty `collisions` in the JSONL/exit summary).
    if hard_regressions > 0 {
        eprintln!("\nFAIL: {hard_regressions} hard regression(s) vs committed ledger baseline.");
        std::process::exit(1);
    }
    eprintln!("\nPASS (collisions, if any, are listed above — see FLAGGED lines).");
}
