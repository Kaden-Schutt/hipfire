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
//! `--dynamic --rocprof-csv <path>` is the "dynamic" sub-task of Task 9: a
//! **CSV-analysis mode**, not a decode-orchestrator. It loads an
//! already-generated rocprofv3 `--kernel-trace --stats` CSV (produced by a
//! prior, separate, out-of-process profiling run — see
//! `rdna_compute::profile_rocprof`'s module docs for the exact invocation),
//! computes achieved-BW for the A3B `mq4r` decode hot-kernel family via
//! `rdna_compute::profile_rocprof::compute_achieved_bw`, feeds each into
//! `Roofline::analyze(hist, kprofile, chip, Some(achieved_bw))`, and emits
//! one JSONL line per hot kernel — ranked by decode-time — plus a human
//! TL;DR. This is the productionized form of tonight's
//! `docs/gfx1201-native-surface.md` analysis. It deliberately never invokes
//! rocprofv3 or spawns the daemon itself (avoids the rocprof anti-hang
//! documented in that same doc) — only the (separate, prior) CSV-generation
//! step needs a GPU; this mode itself does not.
//!
//! Live occupancy + surface-map/Amdahl markdown render remain future work
//! (Task 9 steps 3-4).
//!
//! Usage:
//!   cargo run -p rdna-compute --example kernel_perf_instrument -- --self-check
//!   cargo run -p rdna-compute --example kernel_perf_instrument -- \
//!       --self-check --arch gfx1201 --fixtures-dir tests/kernel-fixtures/gfx1201 \
//!       --ledger tests/kernel-ledger/gfx1201.jsonl
//!   cargo run -p rdna-compute --example kernel_perf_instrument -- \
//!       --diff --current-dir /path/to/modified/fixtures
//!   cargo run -p rdna-compute --example kernel_perf_instrument -- \
//!       --dynamic --rocprof-csv /path/to/a3b_decode_kernel_stats.csv

use rdna_compute::chip_profile::ChipProfile;
use rdna_compute::isa_histogram::IsaHistogram;
use rdna_compute::kernel_ledger::{
    module_collision_scan, KernelLedger, LedgerDelta, LedgerKey, LedgerRow, Reproducer,
};
use rdna_compute::profile_rocprof;
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
    /// CSV-analysis dynamic mode — see the module docs above.
    Dynamic {
        rocprof_csv: PathBuf,
    },
}

fn parse_args() -> Args {
    let argv: Vec<String> = std::env::args().collect();
    let mode = if argv.iter().any(|a| a == "--dynamic") {
        let rocprof_csv = match flag_value(&argv, "--rocprof-csv") {
            Some(p) => PathBuf::from(p),
            None => {
                eprintln!(
                    "kernel_perf_instrument: --dynamic requires --rocprof-csv <path> \
                     (a pre-generated rocprofv3 --kernel-trace --stats CSV; this mode \
                     never invokes rocprofv3 itself)"
                );
                std::process::exit(1);
            }
        };
        Mode::Dynamic { rocprof_csv }
    } else if argv.iter().any(|a| a == "--diff") {
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

/// `achieved_bw` is `None` on the static `--self-check`/`--diff` path
/// (matches `Roofline::analyze`'s own `achieved_bw: Option<f64>` — no
/// measurement, no achieved-BW claim) and `Some(gbps)` on the `--dynamic`
/// path once a CSV-derived achieved-BW has been computed for this fixture's
/// kernel family (see `run_dynamic`).
fn measure_fixture(
    path: &Path,
    arch: &str,
    chip: &ChipProfile,
    ledger: &KernelLedger,
    achieved_bw: Option<f64>,
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

    let roofline = Roofline::analyze(&hist, &kprofile, chip, achieved_bw);

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
    row.achieved_bw = achieved_bw;
    row.roofline_pct = achieved_bw.map(|_| roofline.bw * 100.0);

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

// --- `--dynamic` CSV-analysis mode ------------------------------------------
//
// Real A3B `.mq4r` checkpoint shapes (gfx1201, validated
// `docs/gfx1201-native-surface.md` 2026-07-02, commit cb8c9e26): hidden=2048,
// 40 layers (30 linear-attn + 10 full-attn), 256 experts / 8-per-token,
// moe_intermediate=512, vocab=248320, all HFQ4G256.
const A3B_HIDDEN: usize = 2048;
const A3B_MOE_INTERMEDIATE: usize = 512;
const A3B_K_TOP: usize = 8;
const A3B_VOCAB: usize = 248_320;
/// n_heads(16) × head_dim(256) — the Qwen3.5/3.6 full-attention Wo
/// (`gemv_hfq4g256_residual`) input width; see
/// `rdna_compute::profile`'s `residual_wo_matches_gfx1201_a3b_surface_map`
/// test for the derivation.
const A3B_ATTN_CONCAT_DIM: usize = 16 * 256;

/// The gfx1201 A3B `mq4r` decode "hot family" byte model: kernel alias ->
/// WEIGHT-ONLY bytes moved per call (the `hfq4g256_weight_bytes` convention
/// — `(rows*cols/256)*136`: f32 scale + f32 zp + 128B nibbles per 256-elem
/// group — matching `docs/gfx1201-native-surface.md`'s own "bytes/call"
/// column, NOT the activation-inclusive `profile::gemv_hfq4g256_*_bytes`
/// wrapper functions, which additionally count the small per-call x/y
/// activation-vector traffic the surface map's column omits).
///
/// Four of five entries reproduce the surface map's own bytes/call figure
/// EXACTLY from the real shapes above (see `profile_rocprof`'s
/// `achieved_bw_reproduces_gfx1201_a3b_surface_map_all_five_hot_kernels`
/// test for the ±2% achieved-BW cross-check against the documented
/// measurement). `fused_qkvza_hfq4g256`'s blended two-call-site figure is
/// carried as a documented literal constant — see
/// `rdna_compute::profile::a3b_decode_byte_model_tests::fused_qkvza_blended_matches_gfx1201_a3b_surface_map`
/// for why an exact re-derivation needs an unverified DeltaNet head-count
/// guess (n_heads=24) rather than a real checkpoint-header value.
fn a3b_mq4r_hot_kernel_byte_model() -> HashMap<String, usize> {
    let mut m = HashMap::new();
    m.insert(
        "gemv_hfq4g256_moe_gate_up_k8_indexed".to_string(),
        A3B_K_TOP
            * rdna_compute::profile::hfq4g256_weight_bytes(2 * A3B_MOE_INTERMEDIATE, A3B_HIDDEN),
    );
    m.insert(
        "gemv_hfq4g256_moe_down_k8_indexed".to_string(),
        A3B_K_TOP * rdna_compute::profile::hfq4g256_weight_bytes(A3B_HIDDEN, A3B_MOE_INTERMEDIATE),
    );
    m.insert(
        "gemv_hfq4g256_residual".to_string(),
        rdna_compute::profile::hfq4g256_weight_bytes(A3B_HIDDEN, A3B_ATTN_CONCAT_DIM),
    );
    m.insert(
        "gemv_hfq4g256_multirow_r2".to_string(),
        rdna_compute::profile::hfq4g256_weight_bytes(A3B_VOCAB, A3B_HIDDEN),
    );
    // Blended two-call-site figure — see doc comment above.
    m.insert("fused_qkvza_hfq4g256".to_string(), 6_555_977);
    m
}

/// Coarse family tag extracted from a kernel name (byte-model alias OR a
/// committed `.hsaco` fixture's `file_stem`), used for best-effort ISA
/// enrichment in [`run_dynamic`]. The real committed fixture symbol names
/// and the surface map's documented aliases are known to differ in exact
/// suffix (e.g. the committed fixture
/// `gemv_hfq4g256_moe_gate_up_indexed_batched.hsaco` vs. the documented
/// alias `gemv_hfq4g256_moe_gate_up_k8_indexed`), so exact/substring
/// matching on the full name is unreliable — this coarse tag is the robust
/// common ground between the two naming conventions.
fn kernel_family_tag(name: &str) -> Option<&'static str> {
    let n = name.to_ascii_lowercase();
    if n.contains("gate_up") {
        Some("gate_up")
    } else if n.contains("down") {
        Some("down")
    } else if n.contains("multirow") || n.contains("lm_head") {
        Some("lm_head")
    } else if n.contains("qkvza") {
        Some("qkvza")
    } else if n.contains("residual") {
        Some("residual")
    } else {
        None
    }
}

/// A zero-signal `KernelProfile` for the case a hot kernel has no matching
/// committed `.hsaco` fixture (so no real ISA histogram is available). Fed
/// alongside `IsaHistogram::default()` into `Roofline::analyze` — the
/// `latency_score` short-circuits to `0.0` (WITHHELD) whenever
/// `hist.global_load_b128 == 0`, so an all-zero `KernelProfile`'s otherwise-
/// unused `occupancy_waves`/`max_waves` fields never spuriously bind the
/// verdict. This leaves ONLY the achieved-BW-derived `bw` score carrying
/// real signal for these kernels — an honest degradation, not a fabricated
/// VALU/latency verdict.
fn degenerate_kernel_profile() -> rdna_compute::profiler::KernelProfile {
    rdna_compute::profiler::KernelProfile {
        name: "csv_only_no_fixture".to_string(),
        vgprs: 0,
        sgprs: 0,
        lds_bytes: 0,
        scratch_bytes: 0,
        kernarg_bytes: 0,
        occupancy_waves: 0,
        max_waves: 0,
        occupancy_limiter: "unknown (no .hsaco fixture — CSV-analysis-only)",
    }
}

/// Task 9 "dynamic" sub-task: CSV-analysis mode. See the module docs at the
/// top of this file for the full contract. Never invokes rocprofv3 or
/// spawns the daemon — only reads a CSV a prior, separate profiling run
/// already produced.
fn run_dynamic(args: &Args, chip: &ChipProfile, rocprof_csv: &Path) {
    let rocprof = match profile_rocprof::parse_rocprof_stats_csv(rocprof_csv) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("kernel_perf_instrument: --dynamic: {e}");
            std::process::exit(1);
        }
    };
    if rocprof.is_empty() {
        eprintln!("kernel_perf_instrument: --dynamic: {rocprof_csv:?} parsed to zero kernel rows");
        std::process::exit(1);
    }

    let byte_model = a3b_mq4r_hot_kernel_byte_model();
    let mut achieved = profile_rocprof::compute_achieved_bw(&rocprof, &byte_model, chip);
    if achieved.is_empty() {
        eprintln!(
            "kernel_perf_instrument: --dynamic: none of the {} A3B mq4r hot-kernel aliases \
             matched any row in {rocprof_csv:?} (CSV may be from a different model/arch)",
            byte_model.len()
        );
        std::process::exit(1);
    }
    // Ranked by decode-time — the whole point of the surface-map ranking is
    // "what dominates decode", i.e. highest total profiled wall time first.
    achieved.sort_by(|a, b| b.total_us.total_cmp(&a.total_us));

    // Best-effort ISA-fixture enrichment (see `kernel_family_tag` docs).
    let fixtures = list_hsaco_files(&args.fixtures_dir).unwrap_or_default();

    eprintln!("\n== kernel_perf_instrument --dynamic ({}) ==", args.arch);
    eprintln!(
        "rocprof CSV: {rocprof_csv:?} ({} total rows parsed, {} matched a hot-kernel alias)",
        rocprof.len(),
        achieved.len()
    );
    eprintln!("ranked by decode-time (total_us desc):");

    for entry in &achieved {
        let fixture = kernel_family_tag(&entry.kernel).and_then(|family| {
            fixtures
                .iter()
                .find(|path| {
                    path.file_stem()
                        .and_then(|s| s.to_str())
                        .and_then(kernel_family_tag)
                        == Some(family)
                })
                .cloned()
        });

        let (hist, kprofile, has_fixture) = match &fixture {
            Some(path) => match (IsaHistogram::from_hsaco(path, &args.arch), {
                let mut map = HashMap::new();
                map.insert(
                    path.file_stem()
                        .and_then(|s| s.to_str())
                        .unwrap_or("fixture")
                        .to_string(),
                    path.clone(),
                );
                rdna_compute::profiler::profile_kernels(&args.arch, 0, &map)
                    .1
                    .into_iter()
                    .next()
            }) {
                (Ok(h), Some(k)) => (h, k, true),
                _ => {
                    eprintln!(
                        "  WARN: fixture {path:?} matched family for {} but failed to \
                         disassemble/profile — falling back to CSV-only degenerate roofline",
                        entry.kernel
                    );
                    (IsaHistogram::default(), degenerate_kernel_profile(), false)
                }
            },
            None => (IsaHistogram::default(), degenerate_kernel_profile(), false),
        };

        let roofline = Roofline::analyze(&hist, &kprofile, chip, Some(entry.achieved_gbps));

        let json = serde_json::json!({
            "mode": "dynamic",
            "kernel": entry.kernel,
            "rocprof_name": entry.rocprof_name,
            "calls": entry.calls,
            "bytes_per_call": entry.bytes_per_call,
            "us_per_call": entry.us_per_call,
            "total_us": entry.total_us,
            "achieved_gbps": entry.achieved_gbps,
            "pct_of_peak": entry.pct_of_peak,
            "achieved_bw_trust_score": entry.trust_score,
            "bound_class": format!("{:?}", roofline.binding),
            "bw": roofline.bw,
            "valu_issue": roofline.valu_issue,
            "latency": roofline.latency,
            "second_wall_margin": roofline.second_wall_margin,
            "verdict": roofline.verdict,
            "roofline_trust_score": roofline.trust_score,
            "has_fixture": has_fixture,
            "fixture_path": fixture.as_ref().map(|p| p.display().to_string()),
        });
        println!("{json}");

        eprintln!(
            "  {:>10.1} us  {:<40} {:>6}  {:?}{}",
            entry.total_us,
            entry.kernel,
            entry
                .pct_of_peak
                .map(|p| format!("{:.1}%", p * 100.0))
                .unwrap_or_else(|| "n/a".to_string()),
            roofline.binding,
            if has_fixture {
                ""
            } else {
                " (bw-only, no fixture)"
            },
        );
    }
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

    // `--dynamic` is a self-contained CSV-analysis mode — it does not touch
    // the `.hsaco`-fixtures-required self-check/diff flow below (fixtures
    // are used opportunistically, best-effort, inside `run_dynamic` itself).
    if let Mode::Dynamic { rocprof_csv } = &args.mode {
        run_dynamic(&args, &chip, rocprof_csv);
        return;
    }

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
        match measure_fixture(path, &args.arch, &chip, &ledger, None) {
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
            // Unreachable: `main` returns from the `Mode::Dynamic` branch
            // before this self-check/diff-only TL;DR section runs.
            Mode::Dynamic { .. } => "--dynamic",
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
