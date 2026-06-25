// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! The `tiny_quant` battery executor: a self-contained, tokenizer-free, multi-
//! family quant-quality matrix. For each model family it emits a seeded tiny
//! random-init fixture, quantizes it to that family's loader-supported formats
//! (+ a calibrated cell), builds a near-full-precision anchor, generates a tiny
//! Hessian/imatrix (`collect`), and scores each candidate's KL divergence vs the
//! anchor over a fixed synthetic token stream. Exercises the whole pipeline —
//! quantizer → loader → kernels → output — without real checkpoints or a daemon.
//!
//! Drives two binaries: `hipfire-quantize` (emit + quantize) and the
//! `tiny_quant_probe` example (`kld` / `collect`, see
//! `hipfire-serving-core::tiny_harness`).
//!
//! **Adding a model family:** see `docs/howto/add-tiny-quant-family.md`.
//!
//! Verdict (computed in-executor, not via admission — the baseline is a file
//! keyed by `gpu_arch × family × format`, not a same-case reference row):
//!   - crash / panic / nonzero exit                       → Fail
//!   - non-finite KLD or zero positions scored            → Fail
//!   - baseline present and |kld − base| > drift budget   → Fail
//!   - baseline absent                                    → Skip (Pass under --record)

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::process::Command;

use serde_json::{json, Value};

use crate::*;

/// Committed per-GPU baselines (mirrors `tests/fixture-golden-baselines.txt`).
const TINYQUANT_BASELINES: &str = "tests/tiny-quant-baselines.txt";
/// Default relative drift budget when a baseline row omits its own tolerance.
const DEFAULT_REL_TOL: f64 = 0.25;
/// Absolute floor so near-zero KLD cells (e.g. q8f16) still tripwire.
const ABS_FLOOR: f64 = 1.0e-5;
/// Synthetic-stream length / warmup for KLD + collect (small = fast).
const KLD_LEN: usize = 24;
const KLD_WARMUP: usize = 4;

/// One model family's plan. `anchor` is the highest-fidelity *loadable* format
/// for that arch (the KLD reference); `candidates` are the formats whose loaders
/// + dequant kernels we exercise; `calibrated` consumes a generated Hessian.
struct FamilyPlan {
    arch: &'static str,
    anchor: &'static str,
    candidates: &'static [&'static str],
    /// Extra `--format`/value flags every quantize for this family needs
    /// (e.g. qwen2 must route to arch_id 7, not the LLaMA-default 1).
    quant_flags: &'static [&'static str],
    /// Calibrated cells: `(format, true)` quantizes from the HF dir with
    /// `HIPFIRE_QTIP_HESSIAN=<calib>` set, scored vs the anchor.
    calibrated: &'static [&'static str],
}

/// The validated matrix. Anchors/candidates are bounded by each arch loader's
/// supported quant_types (qwen2/gemma3: F16/Q8/HFQ4; minimax MoE kernels need
/// MQ4/MQ6 experts so its anchor is mq6; qwen3.5 is the broad arch).
fn families() -> &'static [FamilyPlan] {
    &[
        // LLaMA/Mistral (arch 0): simplest dense family — no bias, no QK-norm.
        // Its loader supports q8f16/hfq4/mq4/mq3 but NOT F16/BF16 weights, so the
        // anchor is q8f16 (not fp16) and there's no calibrated cell (qtip3-sim
        // emits bf16, which this loader can't load — same as qwen2/gemma3).
        FamilyPlan {
            arch: "llama",
            anchor: "q8f16",
            candidates: &["hfq4", "mq4", "mq3"],
            quant_flags: &[],
            calibrated: &[],
        },
        FamilyPlan {
            arch: "qwen2",
            anchor: "fp16",
            candidates: &["q8f16", "hfq4"],
            quant_flags: &["--arch-id", "7"],
            calibrated: &[],
        },
        FamilyPlan {
            arch: "gemma3",
            anchor: "fp16",
            candidates: &["q8f16", "hfq4"],
            quant_flags: &[],
            calibrated: &[],
        },
        FamilyPlan {
            arch: "minimax",
            anchor: "mq6",
            candidates: &["mq4"],
            quant_flags: &[],
            calibrated: &[],
        },
        FamilyPlan {
            arch: "qwen3_5",
            anchor: "fp16",
            candidates: &["q8f16", "mq6", "mq4", "mq3"],
            quant_flags: &[],
            // qtip3-sim is the runtime format that consumes our HFQM Hessian
            // (LDLQ); emits bf16, which only the qwen3.5 loader accepts.
            calibrated: &["qtip3-sim"],
        },
        // MoE path coverage: 3D-stacked expert quant + per-expert decode loop +
        // 99-tensor collect (dense attn + router captured; routed experts are
        // imatrix-only). mq4 == mq6 here because the quantizer tiers both
        // `--format mq4` and `mq6` routed experts to the identical
        // gate_up=MQ6G256 / down=HFQ4G128 layout — kept as two cells to verify
        // both CLI paths produce finite output. (These cells NaN'd before the
        // qwen35.rs per-expert-loop fix that stopped feeding MQ6 gate_up through
        // the MQ4-only pre-rotated GEMV — see that commit; the committed gfx1151
        // golden's identical mq4==mq6 hash was the latent symptom.)
        FamilyPlan {
            arch: "qwen3_5_moe",
            anchor: "fp16",
            candidates: &["q8f16", "mq6", "mq4", "mq3"],
            quant_flags: &[],
            calibrated: &[],
        },
    ]
}

/// `target/release/hipfire-quantize` (or debug), honoring an env override.
fn resolve_quantize_bin() -> Option<PathBuf> {
    if let Ok(p) = std::env::var("HIPFIRE_QUANTIZE_BIN") {
        let p = PathBuf::from(p);
        if p.exists() {
            return Some(p);
        }
    }
    let exe = std::env::consts::EXE_SUFFIX;
    let repo = repo_root()?;
    newest_existing_path([
        repo.join(format!("target/release/hipfire-quantize{exe}")),
        repo.join(format!("target/debug/hipfire-quantize{exe}")),
    ])
}

/// `target/release/examples/tiny_quant_probe` (or debug).
fn resolve_probe_bin() -> Option<PathBuf> {
    if let Ok(p) = std::env::var("HIPFIRE_TINY_QUANT_PROBE_BIN") {
        let p = PathBuf::from(p);
        if p.exists() {
            return Some(p);
        }
    }
    let exe = std::env::consts::EXE_SUFFIX;
    let repo = repo_root()?;
    newest_existing_path([
        repo.join(format!("target/release/examples/tiny_quant_probe{exe}")),
        repo.join(format!("target/debug/examples/tiny_quant_probe{exe}")),
    ])
}

/// Parse a `key: value` line out of probe stdout.
fn parse_kv<'a>(out: &'a str, key: &str) -> Option<&'a str> {
    out.lines().find_map(|l| {
        let l = l.trim();
        l.strip_prefix(key)
            .and_then(|r| r.strip_prefix(':'))
            .map(|v| v.trim())
    })
}

/// Committed baselines: `(gpu_arch, family, format) -> (mean_kld, rel_tol)`.
fn load_baselines() -> BTreeMap<(String, String, String), (f64, f64)> {
    let mut m = BTreeMap::new();
    let Some(path) = resolve_repo_path(TINYQUANT_BASELINES) else {
        return m;
    };
    let Ok(body) = std::fs::read_to_string(path) else {
        return m;
    };
    for line in body.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let f: Vec<&str> = line.split_whitespace().collect();
        if f.len() < 4 {
            continue;
        }
        let mean: f64 = match f[3].parse() {
            Ok(v) => v,
            Err(_) => continue,
        };
        let tol: f64 = f
            .get(4)
            .and_then(|s| s.parse().ok())
            .unwrap_or(DEFAULT_REL_TOL);
        m.insert(
            (f[0].to_string(), f[1].to_string(), f[2].to_string()),
            (mean, tol),
        );
    }
    m
}

/// A finished KLD measurement (or an error reason).
struct KldCell {
    mean_kld: f64,
    max_kld: f64,
    n_scored: usize,
    finite: bool,
}

fn run_quantize(
    quant: &Path,
    input: &Path,
    output: &Path,
    format: &str,
    extra_flags: &[&str],
    qtip_hessian: Option<&Path>,
) -> Result<(), String> {
    let mut cmd = Command::new(quant);
    cmd.arg("--input")
        .arg(input)
        .arg("--output")
        .arg(output)
        .arg("--format")
        .arg(format);
    for f in extra_flags {
        cmd.arg(f);
    }
    if let Some(h) = qtip_hessian {
        cmd.env("HIPFIRE_QTIP_HESSIAN", h);
    }
    let out = cmd.output().map_err(|e| format!("spawn quantize: {e}"))?;
    if !out.status.success() || !output.exists() {
        let tail: String = String::from_utf8_lossy(&out.stderr)
            .lines()
            .rev()
            .take(2)
            .collect::<Vec<_>>()
            .join(" | ");
        return Err(format!(
            "quantize {format} exit={:?}: {tail}",
            out.status.code()
        ));
    }
    Ok(())
}

fn run_kld(probe: &Path, arch: &str, anchor: &Path, cand: &Path) -> Result<KldCell, String> {
    let out = Command::new(probe)
        .arg("kld")
        .arg("--arch")
        .arg(arch)
        .arg("--ref")
        .arg(anchor)
        .arg("--cand")
        .arg(cand)
        .arg("--len")
        .arg(KLD_LEN.to_string())
        .arg("--warmup")
        .arg(KLD_WARMUP.to_string())
        // Disable the O_DIRECT slab loader (fails on the tiny file on some FS /
        // integrated GPUs; the mmap path handles every arch).
        .env("HIPFIRE_GPU_SLAB_LOAD", "0")
        .output()
        .map_err(|e| format!("spawn probe kld: {e}"))?;
    let stdout = String::from_utf8_lossy(&out.stdout);
    if !out.status.success() {
        let tail: String = String::from_utf8_lossy(&out.stderr)
            .lines()
            .rev()
            .take(2)
            .collect::<Vec<_>>()
            .join(" | ");
        return Err(format!("probe kld exit={:?}: {tail}", out.status.code()));
    }
    let mean = parse_kv(&stdout, "mean_kld")
        .and_then(|s| s.parse().ok())
        .ok_or("probe kld: no mean_kld")?;
    let max = parse_kv(&stdout, "max_kld")
        .and_then(|s| s.parse().ok())
        .unwrap_or(mean);
    let n = parse_kv(&stdout, "n_scored")
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);
    let finite = parse_kv(&stdout, "finite") == Some("true");
    Ok(KldCell {
        mean_kld: mean,
        max_kld: max,
        n_scored: n,
        finite,
    })
}

/// Verdict for a KLD cell. `base` = committed `(mean, rel_tol)` if any.
fn kld_status(
    cell: &KldCell,
    base: Option<(f64, f64)>,
    record: bool,
) -> (EvalStatus, Option<String>) {
    if !cell.finite || !cell.mean_kld.is_finite() {
        return (EvalStatus::Fail, Some("non-finite KLD".into()));
    }
    if cell.n_scored == 0 {
        return (EvalStatus::Fail, Some("zero positions scored".into()));
    }
    match base {
        Some((b, tol)) => {
            let budget = (tol * b).max(ABS_FLOOR);
            if (cell.mean_kld - b).abs() > budget {
                (
                    EvalStatus::Fail,
                    Some(format!(
                        "KLD drift {:.6} vs baseline {:.6} (budget ±{:.6})",
                        cell.mean_kld, b, budget
                    )),
                )
            } else {
                (EvalStatus::Pass, None)
            }
        }
        // No committed baseline → the cell ran but nothing was COMPARED. Report
        // Skip (not Pass), so a fresh gpu_arch / newly-added format isn't a
        // misleading green — it mirrors fixture-golden's "inconclusive" status.
        // The hard-fail checks above still catch crashes/NaN/zero-token cells.
        // During `--record` we ARE establishing the baseline this run, so Pass.
        None if record => (EvalStatus::Pass, Some("recording new baseline".into())),
        None => (
            EvalStatus::Skip,
            Some("no committed baseline (run --record)".into()),
        ),
    }
}

#[allow(clippy::too_many_arguments)]
fn kld_metrics(
    family: &str,
    fmt: &str,
    calibrated: bool,
    gpu_arch: &str,
    cell: &KldCell,
    base: Option<(f64, f64)>,
) -> BTreeMap<String, Value> {
    let mut m = BTreeMap::new();
    m.insert("executor".into(), json!("tinyquant"));
    m.insert("implemented".into(), json!(true));
    m.insert("family".into(), json!(family));
    m.insert("format".into(), json!(fmt));
    m.insert("calibrated".into(), json!(calibrated));
    m.insert("gpu_arch".into(), json!(gpu_arch));
    m.insert("mean_kld".into(), json!(cell.mean_kld));
    m.insert("max_kld".into(), json!(cell.max_kld));
    m.insert("n_scored".into(), json!(cell.n_scored));
    if let Some((b, tol)) = base {
        m.insert("baseline_kld".into(), json!(b));
        m.insert("baseline_tol".into(), json!(tol));
        m.insert("kld_drift".into(), json!(cell.mean_kld - b));
    }
    m
}

pub(crate) fn tiny_quant_rows(config: &EvalConfig, ctx: &EvalContext) -> Vec<EvalResult> {
    let gpu_arch = ctx.arch.clone().unwrap_or_else(|| "unknown".to_string());
    let record = std::env::var("HIPFIRE_TINYQUANT_RECORD").ok().as_deref() == Some("1");
    // Optional comma-separated family allowlist (`HIPFIRE_TINYQUANT_FAMILIES`).
    // Lets a host run a subset — e.g. skip a family whose kernels fault on a
    // specific arch (minimax topk GPU-faults on gfx1151) so the rest still
    // record. Empty/unset ⇒ all families. Excluded families are logged, not
    // silently dropped.
    let only: Option<Vec<String>> = std::env::var("HIPFIRE_TINYQUANT_FAMILIES").ok().map(|s| {
        s.split(',')
            .map(|x| x.trim().to_string())
            .filter(|x| !x.is_empty())
            .collect()
    });
    if let Some(ref allow) = only {
        let skipped: Vec<&str> = families()
            .iter()
            .map(|p| p.arch)
            .filter(|a| !allow.iter().any(|x| x == a))
            .collect();
        if !skipped.is_empty() {
            eprintln!(
                "tiny_quant: HIPFIRE_TINYQUANT_FAMILIES={} → skipping {}",
                allow.join(","),
                skipped.join(",")
            );
        }
    }
    let mut rows = Vec::new();

    let (Some(quant), Some(probe)) = (resolve_quantize_bin(), resolve_probe_bin()) else {
        return vec![skip_row(
            BatteryId::TinyQuant,
            None,
            "binaries",
            None,
            "tiny_quant requires `hipfire-quantize` + the `tiny_quant_probe` example \
             (cargo build --release -p hipfire-quantize -p hipfire-serving-core \
             --example tiny_quant_probe)",
            config,
            ctx,
            None,
        )];
    };

    let work = std::env::temp_dir().join(format!("hipfire-tinyquant-{}", std::process::id()));
    let _ = std::fs::create_dir_all(&work);
    let baselines = load_baselines();
    // (gpu_arch, family, format) -> observed mean_kld, for --record.
    let mut observed: Vec<(String, String, String, f64)> = Vec::new();

    let push = |family: &str,
                cell: &str,
                status: EvalStatus,
                reason: Option<String>,
                metrics: BTreeMap<String, Value>,
                rows: &mut Vec<EvalResult>| {
        let case = format!("{family}/{cell}");
        rows.push(row_for_model(
            BatteryId::TinyQuant,
            None,
            &case,
            None,
            status,
            reason,
            metrics,
            config,
            ctx,
            None,
            0,
            format!("tiny:{family}:{cell}"),
        ));
    };

    for plan in families() {
        let fam = plan.arch;
        if let Some(ref allow) = only {
            if !allow.iter().any(|x| x == fam) {
                continue;
            }
        }
        let dir = work.join(fam);
        // ── emit ──
        let emit = Command::new(&quant)
            .arg("--emit-fixture")
            .arg(fam)
            .arg("--out")
            .arg(&dir)
            .arg("--seed")
            .arg("42")
            .output();
        if emit.as_ref().map(|o| !o.status.success()).unwrap_or(true) {
            let r = emit
                .as_ref()
                .err()
                .map(|e| e.to_string())
                .unwrap_or_else(|| "emit-fixture nonzero exit".into());
            let mut m = BTreeMap::new();
            m.insert("family".into(), json!(fam));
            push(fam, "emit", EvalStatus::Fail, Some(r), m, &mut rows);
            continue;
        }

        // ── anchor (near-full-precision, loadable) ──
        let anchor = work.join(format!("{fam}.{}.hfq", plan.anchor));
        if let Err(e) = run_quantize(&quant, &dir, &anchor, plan.anchor, plan.quant_flags, None) {
            let mut m = BTreeMap::new();
            m.insert("family".into(), json!(fam));
            push(fam, "anchor", EvalStatus::Fail, Some(e), m, &mut rows);
            continue;
        }

        // ── collect: generate a tiny Hessian/imatrix (.calib.hfq) ──
        let calib = work.join(format!("{fam}.calib.hfq"));
        let collect = Command::new(&probe)
            .arg("collect")
            .arg("--arch")
            .arg(fam)
            .arg("--model")
            .arg(&anchor)
            .arg("--out")
            .arg(&calib)
            .arg("--len")
            .arg(KLD_LEN.to_string())
            .env("HIPFIRE_GPU_SLAB_LOAD", "0")
            .output();
        let mut have_calib = false;
        match collect {
            Ok(o) if o.status.success() => {
                let so = String::from_utf8_lossy(&o.stdout);
                let n_tensors: usize = parse_kv(&so, "n_tensors")
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(0);
                let consistency: f64 = parse_kv(&so, "consistency")
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(f64::NAN);
                let mut m = BTreeMap::new();
                m.insert("executor".into(), json!("tinyquant"));
                m.insert("implemented".into(), json!(true));
                m.insert("family".into(), json!(fam));
                m.insert("n_tensors".into(), json!(n_tensors));
                m.insert("consistency".into(), json!(consistency));
                // Hard-fail if nothing captured or the diag(H)≈Σx² check blew up.
                let (st, rs) = if n_tensors == 0 {
                    (EvalStatus::Fail, Some("collect: 0 tensors captured".into()))
                } else if !consistency.is_finite() || consistency > 0.05 {
                    (
                        EvalStatus::Fail,
                        Some(format!("collect: consistency {consistency:.4}")),
                    )
                } else {
                    have_calib = true;
                    (EvalStatus::Pass, None)
                };
                push(fam, "collect", st, rs, m, &mut rows);
            }
            other => {
                let r = match other {
                    Ok(o) => {
                        let tail: String = String::from_utf8_lossy(&o.stderr)
                            .lines()
                            .rev()
                            .take(2)
                            .collect::<Vec<_>>()
                            .join(" | ");
                        format!("collect exit={:?}: {tail}", o.status.code())
                    }
                    Err(e) => format!("spawn collect: {e}"),
                };
                let mut m = BTreeMap::new();
                m.insert("family".into(), json!(fam));
                push(fam, "collect", EvalStatus::Fail, Some(r), m, &mut rows);
            }
        }

        // ── base-format candidate cells: quantize → KLD vs anchor ──
        for &fmt in plan.candidates {
            let cand = work.join(format!("{fam}.{fmt}.hfq"));
            if let Err(e) = run_quantize(&quant, &dir, &cand, fmt, plan.quant_flags, None) {
                let mut m = BTreeMap::new();
                m.insert("family".into(), json!(fam));
                m.insert("format".into(), json!(fmt));
                push(
                    fam,
                    &format!("quantize:{fmt}"),
                    EvalStatus::Fail,
                    Some(e),
                    m,
                    &mut rows,
                );
                continue;
            }
            match run_kld(&probe, fam, &anchor, &cand) {
                Ok(cell) => {
                    let base = baselines
                        .get(&(gpu_arch.clone(), fam.to_string(), fmt.to_string()))
                        .copied();
                    if record && cell.finite {
                        observed.push((
                            gpu_arch.clone(),
                            fam.to_string(),
                            fmt.to_string(),
                            cell.mean_kld,
                        ));
                    }
                    let (st, rs) = kld_status(&cell, base, record);
                    let m = kld_metrics(fam, fmt, false, &gpu_arch, &cell, base);
                    push(fam, &format!("kld:{fmt}"), st, rs, m, &mut rows);
                }
                Err(e) => {
                    let mut m = BTreeMap::new();
                    m.insert("family".into(), json!(fam));
                    m.insert("format".into(), json!(fmt));
                    push(
                        fam,
                        &format!("kld:{fmt}"),
                        EvalStatus::Fail,
                        Some(e),
                        m,
                        &mut rows,
                    );
                }
            }
        }

        // ── calibrated cells: quantize with the generated Hessian → KLD ──
        for &fmt in plan.calibrated {
            if !have_calib {
                let mut m = BTreeMap::new();
                m.insert("family".into(), json!(fam));
                m.insert("format".into(), json!(fmt));
                push(
                    fam,
                    &format!("kld:{fmt}(calib)"),
                    EvalStatus::Skip,
                    Some("no calib artifact (collect failed)".into()),
                    m,
                    &mut rows,
                );
                continue;
            }
            let cand = work.join(format!("{fam}.{}.hfq", fmt.replace('-', "_")));
            if let Err(e) = run_quantize(&quant, &dir, &cand, fmt, plan.quant_flags, Some(&calib)) {
                let mut m = BTreeMap::new();
                m.insert("family".into(), json!(fam));
                m.insert("format".into(), json!(fmt));
                m.insert("calibrated".into(), json!(true));
                push(
                    fam,
                    &format!("quantize:{fmt}(calib)"),
                    EvalStatus::Fail,
                    Some(e),
                    m,
                    &mut rows,
                );
                continue;
            }
            match run_kld(&probe, fam, &anchor, &cand) {
                Ok(cell) => {
                    let key = format!("{fmt}-calib");
                    let base = baselines
                        .get(&(gpu_arch.clone(), fam.to_string(), key.clone()))
                        .copied();
                    if record && cell.finite {
                        observed.push((gpu_arch.clone(), fam.to_string(), key, cell.mean_kld));
                    }
                    let (st, rs) = kld_status(&cell, base, record);
                    let m = kld_metrics(fam, fmt, true, &gpu_arch, &cell, base);
                    push(fam, &format!("kld:{fmt}(calib)"), st, rs, m, &mut rows);
                }
                Err(e) => {
                    let mut m = BTreeMap::new();
                    m.insert("family".into(), json!(fam));
                    m.insert("format".into(), json!(fmt));
                    push(
                        fam,
                        &format!("kld:{fmt}(calib)"),
                        EvalStatus::Fail,
                        Some(e),
                        m,
                        &mut rows,
                    );
                }
            }
        }
    }

    if record && !observed.is_empty() {
        if let Err(e) = write_baselines(&observed) {
            eprintln!("tiny_quant: --record write failed: {e}");
        } else {
            eprintln!("tiny_quant: recorded {} baseline cells", observed.len());
        }
    }
    let _ = std::fs::remove_dir_all(&work);
    rows
}

/// Rewrite `tests/tiny-quant-baselines.txt` from observed cells (`--record`).
/// Upserts per `(gpu_arch, family, format)` CELL: only cells present in this run's
/// observation set are replaced; every other existing row is preserved verbatim —
/// other GPUs AND same-GPU families/formats not recorded this run (e.g. a subset
/// record via `HIPFIRE_TINYQUANT_FAMILIES`, or minimax excluded on a faulting arch).
fn write_baselines(observed: &[(String, String, String, f64)]) -> std::io::Result<()> {
    let path = repo_root()
        .map(|r| r.join(TINYQUANT_BASELINES))
        .ok_or_else(|| std::io::Error::other("repo root not found"))?;
    // Set of cells being (re-)recorded this run, keyed by the full cell.
    let recording: std::collections::HashSet<(&str, &str, &str)> = observed
        .iter()
        .map(|(g, fam, fmt, _)| (g.as_str(), fam.as_str(), fmt.as_str()))
        .collect();
    let mut kept: Vec<String> = Vec::new();
    let mut prior_tol: BTreeMap<(String, String, String), f64> = BTreeMap::new();
    if let Ok(body) = std::fs::read_to_string(&path) {
        for line in body.lines() {
            let t = line.trim();
            if t.is_empty() || t.starts_with('#') {
                continue;
            }
            let f: Vec<&str> = t.split_whitespace().collect();
            if f.len() < 4 {
                continue; // malformed row — drop
            }
            let key = (f[0], f[1], f[2]);
            if recording.contains(&key) {
                // Re-recorded this run: remember a hand-tuned tol so the rewrite
                // preserves it instead of resetting to the default; drop the row
                // (it's re-added from `observed` below).
                if let Some(tol) = f.get(4).and_then(|s| s.parse::<f64>().ok()) {
                    prior_tol.insert((f[0].into(), f[1].into(), f[2].into()), tol);
                }
            } else {
                // Not in this observation set — preserve verbatim (other GPUs,
                // and same-GPU families/formats skipped this run).
                kept.push(t.to_string());
            }
        }
    }
    let mut out = String::new();
    out.push_str("# tiny-quant KLD baselines — gpu_arch family format mean_kld rel_tol\n");
    out.push_str(
        "# regenerate per GPU: HIPFIRE_TINYQUANT_RECORD=1 ./tests/tiny-quant-gate.sh --record\n",
    );
    let mut all: Vec<String> = kept;
    for (g, fam, fmt, mean) in observed {
        let tol = prior_tol
            .get(&(g.clone(), fam.clone(), fmt.clone()))
            .copied()
            .unwrap_or(DEFAULT_REL_TOL);
        all.push(format!("{g} {fam} {fmt} {mean:.8} {tol}"));
    }
    all.sort();
    for l in all {
        out.push_str(&l);
        out.push('\n');
    }
    std::fs::write(&path, out)
}
