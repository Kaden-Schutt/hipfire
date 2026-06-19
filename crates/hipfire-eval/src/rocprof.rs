// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! rocprofv3 speed-anchor + kernel-stats CSV parsing for the eval harness.
//!
//! `run_rocprof_speed_anchor` runs a profiled forward under rocprofv3 and turns
//! its kernel-stats CSV into an EvalResult anchor; the rest resolve the rocprofv3
//! binary and parse its CSV output. Extracted verbatim from the former
//! `hipfire-eval/src/lib.rs` monolith (no behavior change).

use std::path::{Path, PathBuf};
use std::process::Command;

use serde_json::{json, Value};

use crate::*;

pub(crate) fn run_rocprof_speed_anchor(config: &EvalConfig, ctx: &EvalContext) -> EvalResult {
    let prompt_path = "benchmarks/prompts/dflash_resident_smoke.txt";
    let prompt_ref = prompt(prompt_path);
    let base_metrics = BTreeMap::from([
        ("executor".to_string(), json!("examples")),
        ("profiling_requested".to_string(), json!(true)),
        ("profiling_collector".to_string(), json!("rocprofv3")),
    ]);
    if !matches!(
        config.executor,
        EvalExecutorMode::Auto | EvalExecutorMode::Examples | EvalExecutorMode::Direct
    ) {
        return skip_row_with_metrics(
            BatteryId::Profile,
            None,
            "rocprof_speed_anchor",
            None,
            "passive rocprof collection requires --executor auto, examples, or direct",
            config,
            ctx,
            prompt_ref,
            base_metrics,
        );
    }
    if Path::new(&config.model).canonicalize().is_err() {
        return skip_row_with_metrics(
            BatteryId::Profile,
            None,
            "rocprof_speed_anchor",
            None,
            "passive rocprof collection requires --model to be a local filesystem path",
            config,
            ctx,
            prompt_ref,
            base_metrics,
        );
    }
    let Some(rocprof) = resolve_rocprofv3_bin() else {
        return skip_row_with_metrics(
            BatteryId::Profile,
            None,
            "rocprof_speed_anchor",
            None,
            "rocprofv3 not found; passive profiling evidence not collected",
            config,
            ctx,
            prompt_ref,
            base_metrics,
        );
    };
    let Some(bin) = resolve_dflash_spec_demo_bin() else {
        return skip_row_with_metrics(
            BatteryId::Profile,
            None,
            "rocprof_speed_anchor",
            None,
            "dflash_spec_demo example binary not found; build with `cargo build --release --features deltanet -p hipfire-runtime --example dflash_spec_demo`",
            config,
            ctx,
            prompt_ref,
            base_metrics,
        );
    };
    let Some(prompt_abs) = resolve_repo_path(prompt_path) else {
        return skip_row_with_metrics(
            BatteryId::Profile,
            None,
            "rocprof_speed_anchor",
            None,
            "rocprof speed prompt fixture not found",
            config,
            ctx,
            prompt_ref,
            base_metrics,
        );
    };

    let evidence_dir = runtime_evidence_dir(config, "rocprof-speed-anchor", &config.model);
    let raw_dir = config.out_dir.join("artifacts").join("rocprof");
    if let Err(err) = fs::create_dir_all(&evidence_dir) {
        return skip_row_with_metrics(
            BatteryId::Profile,
            None,
            "rocprof_speed_anchor",
            None,
            &format!("create rocprof evidence dir: {err}"),
            config,
            ctx,
            prompt_ref,
            base_metrics,
        );
    }
    if let Err(err) = fs::create_dir_all(&raw_dir) {
        return skip_row_with_metrics(
            BatteryId::Profile,
            None,
            "rocprof_speed_anchor",
            None,
            &format!("create rocprof artifact dir: {err}"),
            config,
            ctx,
            prompt_ref,
            base_metrics,
        );
    }

    let prefix = format!("rocprof-speed-{}", utc_stamp_compact());
    let mut target_args = vec![
        "--target".to_string(),
        config.model.clone(),
        "--prompt-file".to_string(),
        prompt_abs.display().to_string(),
        "--max".to_string(),
        config.max_tokens.to_string(),
        "--ctx".to_string(),
        "2048".to_string(),
        "--kv-mode".to_string(),
        config.kv_mode.clone().unwrap_or_else(|| "q8".to_string()),
        "--no-adaptive-b".to_string(),
        "--no-chatml".to_string(),
        "--ar-baseline".to_string(),
    ];
    add_runtime_evidence_arg(&mut target_args, &evidence_dir);
    let rocprof_args = vec![
        "--kernel-trace".to_string(),
        "--stats".to_string(),
        "-S".to_string(),
        "--output-format".to_string(),
        "csv".to_string(),
        "-d".to_string(),
        raw_dir.display().to_string(),
        "-o".to_string(),
        prefix.clone(),
        "--".to_string(),
        bin.display().to_string(),
    ];
    let command_display = format!(
        "{} {} {}",
        rocprof.display(),
        rocprof_args.join(" "),
        target_args.join(" ")
    );
    let started = SystemTime::now();
    let mut command = Command::new(&rocprof);
    command.args(&rocprof_args);
    command.args(&target_args);
    command.env("HIPFIRE_PROFILE", "1");
    command.env("HIPFIRE_PROFILE_CYCLES", "1");
    let output = match command.output() {
        Ok(output) => output,
        Err(err) => {
            let mut metrics = base_metrics.clone();
            metrics.insert("command".to_string(), json!(command_display));
            return skip_row_with_metrics(
                BatteryId::Profile,
                None,
                "rocprof_speed_anchor",
                None,
                &format!("spawn rocprofv3: {err}"),
                config,
                ctx,
                prompt_ref,
                metrics,
            );
        }
    };
    let elapsed_ms = elapsed_since_ms(started);
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    let mut metrics = base_metrics.clone();
    metrics.extend(parse_bench_metrics(&stderr));
    metrics.insert("command".to_string(), json!(command_display));
    metrics.insert(
        "rocprof_bin".to_string(),
        json!(rocprof.display().to_string()),
    );
    metrics.insert(
        "rocprof_output_dir".to_string(),
        json!(raw_dir.display().to_string()),
    );
    metrics.insert("rocprof_prefix".to_string(), json!(prefix));
    metrics.insert(
        "runtime_evidence_dir".to_string(),
        json!(evidence_dir.display().to_string()),
    );
    metrics.insert(
        "stdout_hash".to_string(),
        json!(stable_hash_bytes(stdout.as_bytes())),
    );
    metrics.insert(
        "stderr_hash".to_string(),
        json!(stable_hash_bytes(stderr.as_bytes())),
    );
    if !output.status.success() {
        return row(
            BatteryId::Profile,
            None,
            "rocprof_speed_anchor",
            None,
            EvalStatus::Skip,
            Some(format!("rocprofv3 exited with {}", output.status)),
            metrics,
            config,
            ctx,
            prompt_ref,
            elapsed_ms,
        );
    }

    match write_rocprof_profile_evidence(&raw_dir, &evidence_dir, config, ctx, &metrics) {
        Ok(count) if count > 0 => {
            metrics.insert("rocprof_kernel_rows".to_string(), json!(count));
            row(
                BatteryId::Profile,
                None,
                "rocprof_speed_anchor",
                None,
                EvalStatus::Pass,
                None,
                metrics,
                config,
                ctx,
                prompt_ref,
                elapsed_ms,
            )
        }
        Ok(_) => row(
            BatteryId::Profile,
            None,
            "rocprof_speed_anchor",
            None,
            EvalStatus::Skip,
            Some("rocprofv3 completed but no kernel stats CSV rows were found".to_string()),
            metrics,
            config,
            ctx,
            prompt_ref,
            elapsed_ms,
        ),
        Err(err) => row(
            BatteryId::Profile,
            None,
            "rocprof_speed_anchor",
            None,
            EvalStatus::Skip,
            Some(err),
            metrics,
            config,
            ctx,
            prompt_ref,
            elapsed_ms,
        ),
    }
}

pub(crate) fn resolve_rocprofv3_bin() -> Option<PathBuf> {
    if let Ok(path) = std::env::var("HIPFIRE_ROCPROF_BIN") {
        let path = PathBuf::from(path);
        if path.exists() {
            return Some(path);
        }
    }
    resolve_path_tool("rocprofv3")
}

pub(crate) fn resolve_path_tool(name: &str) -> Option<PathBuf> {
    std::env::var_os("PATH")
        .into_iter()
        .flat_map(|paths| std::env::split_paths(&paths).collect::<Vec<_>>())
        .map(|dir| dir.join(name))
        .find(|path| path.exists())
}

pub(crate) fn write_rocprof_profile_evidence(
    raw_dir: &Path,
    evidence_dir: &Path,
    config: &EvalConfig,
    ctx: &EvalContext,
    command_metrics: &BTreeMap<String, Value>,
) -> Result<usize, String> {
    let csvs = rocprof_kernel_stats_csvs(raw_dir);
    let mut records = Vec::new();
    for csv in &csvs {
        let kernels = parse_rocprof_kernel_stats_csv(csv)?;
        for kernel in kernels {
            records.push(json!({
                "kind": "profiling",
                "collector": "rocprofv3",
                "source_path": csv.display().to_string(),
                "metrics": {
                    "kernel_name": kernel.name,
                    "duration_us": kernel.duration_us,
                    "calls": kernel.calls,
                    "percentage": kernel.percentage,
                    "average_us": kernel.average_us,
                    "min_us": kernel.min_us,
                    "max_us": kernel.max_us,
                }
            }));
        }
    }
    let row_count = records.len();
    let value = json!({
        "schema": 1,
        "kind": "profiling",
        "status": if row_count > 0 { "collected" } else { "not_collected" },
        "collector": "rocprofv3",
        "provenance": run_provenance_value(ctx),
        "collection": {
            "source": "hipfire-eval",
            "profiling_mode": config.profile.as_str(),
            "raw_dir": raw_dir.display().to_string(),
            "csv_files": csvs.iter().map(|path| json!({
                "path": path.display().to_string(),
                "hash": file_hash(path),
            })).collect::<Vec<_>>(),
        },
        "command_metrics": command_metrics,
        "records": records,
    });
    write_json_pretty(&evidence_dir.join("profiling.json"), &value)?;
    Ok(row_count)
}

#[derive(Debug, Clone)]
pub(crate) struct RocprofKernelStats {
    pub(crate) name: String,
    pub(crate) calls: u64,
    pub(crate) duration_us: f64,
    pub(crate) percentage: f64,
    pub(crate) average_us: f64,
    pub(crate) min_us: f64,
    pub(crate) max_us: f64,
}

pub(crate) fn rocprof_kernel_stats_csvs(dir: &Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    collect_rocprof_kernel_stats_csvs(dir, 0, &mut out);
    out.sort();
    out
}

pub(crate) fn collect_rocprof_kernel_stats_csvs(dir: &Path, depth: usize, out: &mut Vec<PathBuf>) {
    if depth > 3 {
        return;
    }
    let Ok(entries) = fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            collect_rocprof_kernel_stats_csvs(&path, depth + 1, out);
            continue;
        }
        let Some(name) = path.file_name().and_then(OsStr::to_str) else {
            continue;
        };
        if name.ends_with("_kernel_stats.csv") {
            out.push(path);
        }
    }
}

pub(crate) fn parse_rocprof_kernel_stats_csv(
    path: &Path,
) -> Result<Vec<RocprofKernelStats>, String> {
    let text = fs::read_to_string(path)
        .map_err(|err| format!("read rocprof CSV {}: {err}", path.display()))?;
    parse_rocprof_kernel_stats_csv_text(&text)
}

pub(crate) fn parse_rocprof_kernel_stats_csv_text(
    text: &str,
) -> Result<Vec<RocprofKernelStats>, String> {
    let mut lines = text.lines();
    let header = lines
        .next()
        .ok_or_else(|| "rocprofv3 CSV is empty".to_string())?
        .trim()
        .to_ascii_lowercase();
    if !header.contains("name") || !header.contains("calls") {
        return Err(format!(
            "rocprofv3 CSV header does not look like kernel stats: {header:?}"
        ));
    }
    let mut kernels = Vec::new();
    for raw in lines {
        let line = raw.trim();
        if line.is_empty() {
            continue;
        }
        let parts = split_rocprof_csv_line(line);
        if parts.len() < 8 {
            continue;
        }
        let n = parts.len();
        let name = parts[..n - 7]
            .join(",")
            .trim()
            .trim_matches('"')
            .to_string();
        let calls = parse_rocprof_u64(&parts[n - 7]);
        let total_ns = parse_rocprof_f64(&parts[n - 6]);
        let average_ns = parse_rocprof_f64(&parts[n - 5]);
        let percentage = parse_rocprof_f64(&parts[n - 4]);
        let min_ns = parse_rocprof_f64(&parts[n - 3]);
        let max_ns = parse_rocprof_f64(&parts[n - 2]);
        if let (Some(calls), Some(total_ns), Some(average_ns), Some(percentage)) =
            (calls, total_ns, average_ns, percentage)
        {
            kernels.push(RocprofKernelStats {
                name,
                calls,
                duration_us: total_ns / 1_000.0,
                percentage,
                average_us: average_ns / 1_000.0,
                min_us: min_ns.unwrap_or(0.0) / 1_000.0,
                max_us: max_ns.unwrap_or(0.0) / 1_000.0,
            });
        }
    }
    kernels.sort_by(|a, b| {
        b.duration_us
            .partial_cmp(&a.duration_us)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    Ok(kernels)
}

pub(crate) fn split_rocprof_csv_line(line: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut cur = String::new();
    let mut in_quotes = false;
    for ch in line.chars() {
        match ch {
            '"' => {
                in_quotes = !in_quotes;
                cur.push(ch);
            }
            ',' if !in_quotes => {
                out.push(cur.trim().to_string());
                cur.clear();
            }
            _ => cur.push(ch),
        }
    }
    out.push(cur.trim().to_string());
    out
}

pub(crate) fn parse_rocprof_f64(raw: &str) -> Option<f64> {
    raw.trim().trim_matches('"').parse().ok()
}

pub(crate) fn parse_rocprof_u64(raw: &str) -> Option<u64> {
    raw.trim().trim_matches('"').parse().ok()
}
