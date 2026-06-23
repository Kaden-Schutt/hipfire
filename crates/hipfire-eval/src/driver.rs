// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! The battery-execution driver: result caching, benchmark aggregation, and the
//! per-battery dispatch into the executors.
//!
//! `run_eval_batteries` drives each requested battery through `run_battery`
//! (which picks the mock/examples/daemon/direct executor), with result-cache
//! read/write and multi-run benchmark aggregation (percentile summaries) layered
//! on top. Extracted verbatim from the former `hipfire-eval/src/lib.rs` monolith
//! (no behavior change).

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use crate::*;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct ResultCacheEntry {
    schema: u32,
    key: String,
    created_utc: String,
    cache_mode: EvalCacheMode,
    rows: Vec<EvalResult>,
}

pub(crate) fn run_eval_batteries(
    config: &EvalConfig,
    ctx: &EvalContext,
    datasets: &[DatasetManifestEntry],
) -> Result<Vec<EvalResult>, String> {
    if !daemon_shared_model_load_enabled(config) {
        let mut results = Vec::new();
        for battery in &config.batteries {
            results.extend(run_battery_cached(*battery, config, ctx, datasets)?);
        }
        return Ok(results);
    }

    let mut rows_by_battery: BTreeMap<BatteryId, Vec<EvalResult>> = BTreeMap::new();
    let mut shared_misses = Vec::new();
    for battery in &config.batteries {
        let key = result_cache_key(*battery, config, ctx, datasets)?;
        let path = result_cache_path(config, &key);
        if config.cache_mode == EvalCacheMode::Regenerate {
            let _ = fs::remove_file(&path);
        }
        if config.cache_mode.reads() {
            if let Some(rows) = read_result_cache_entry(&path, &key) {
                rows_by_battery.insert(
                    *battery,
                    rows.into_iter()
                        .map(|row| mark_cache_hit(row, &key, &path))
                        .collect(),
                );
                continue;
            }
        }

        if daemon_shared_model_load_battery(*battery) {
            shared_misses.push((*battery, key, path));
        } else {
            rows_by_battery.insert(
                *battery,
                run_battery_cached(*battery, config, ctx, datasets)?,
            );
        }
    }

    if !shared_misses.is_empty() {
        let shared_batteries = shared_misses
            .iter()
            .map(|(battery, _, _)| *battery)
            .collect::<Vec<_>>();
        let mut shared_rows = run_daemon_shared_model_load_rows(config, ctx, &shared_batteries);
        for (battery, key, path) in shared_misses {
            let rows = shared_rows.remove(&battery).unwrap_or_default();
            if config.cache_mode.writes() {
                if let Err(err) = write_result_cache_entry(&path, &key, config.cache_mode, &rows) {
                    eprintln!(
                        "warning: failed to write eval result cache {}: {err}",
                        path.display()
                    );
                }
            }
            rows_by_battery.insert(battery, rows);
        }
    }

    let mut results = Vec::new();
    for battery in &config.batteries {
        if let Some(rows) = rows_by_battery.remove(battery) {
            results.extend(rows);
        }
    }
    Ok(results)
}

pub(crate) fn daemon_shared_model_load_enabled(config: &EvalConfig) -> bool {
    matches!(
        config.executor,
        EvalExecutorMode::Auto | EvalExecutorMode::Daemon
    ) && config.runs == 1
        && config.warmup_runs == 0
        && !config.benchmark
        && config
            .batteries
            .iter()
            .filter(|battery| daemon_shared_model_load_battery(**battery))
            .count()
            > 1
        && config
            .batteries
            .iter()
            .filter(|battery| daemon_shared_model_load_battery(**battery))
            .all(|battery| daemon_executor_available_for(config, *battery))
}

pub(crate) fn daemon_shared_model_load_battery(battery: BatteryId) -> bool {
    matches!(
        battery,
        BatteryId::Smoke | BatteryId::Speed | BatteryId::Profile
    )
}

pub(crate) fn daemon_executor_available_for(config: &EvalConfig, battery: BatteryId) -> bool {
    if !matches!(
        battery,
        BatteryId::Smoke | BatteryId::Speed | BatteryId::Profile | BatteryId::Vision
    ) {
        return false;
    }
    if !Path::new(&config.model).exists() {
        return true;
    }
    hipfire_daemon_adapter::find_daemon_bin().is_some()
}

pub(crate) fn run_battery_cached(
    battery: BatteryId,
    config: &EvalConfig,
    ctx: &EvalContext,
    datasets: &[DatasetManifestEntry],
) -> Result<Vec<EvalResult>, String> {
    let key = result_cache_key(battery, config, ctx, datasets)?;
    let path = result_cache_path(config, &key);
    if config.cache_mode == EvalCacheMode::Regenerate {
        let _ = fs::remove_file(&path);
    }
    if config.cache_mode.reads() {
        if let Some(rows) = read_result_cache_entry(&path, &key) {
            return Ok(rows
                .into_iter()
                .map(|row| mark_cache_hit(row, &key, &path))
                .collect());
        }
    }

    for _ in 0..config.warmup_runs {
        let _ = run_battery(battery, config, ctx, datasets);
    }
    let collect_samples = config.runs > 1 || config.benchmark;
    let mut rows = Vec::new();
    for run_idx in 0..config.runs {
        let mut run_rows = run_battery(battery, config, ctx, datasets);
        if collect_samples {
            for row in &mut run_rows {
                mark_benchmark_sample(row, run_idx + 1, config);
            }
        }
        rows.extend(run_rows);
    }
    if collect_samples {
        rows.extend(benchmark_aggregate_rows(&rows, config));
    }
    if config.cache_mode.writes() {
        if let Err(err) = write_result_cache_entry(&path, &key, config.cache_mode, &rows) {
            eprintln!(
                "warning: failed to write eval result cache {}: {err}",
                path.display()
            );
        }
    }
    Ok(rows)
}

pub(crate) fn mark_benchmark_sample(row: &mut EvalResult, run_index: usize, config: &EvalConfig) {
    row.metrics
        .insert("benchmark_sample".to_string(), json!(true));
    row.metrics
        .insert("run_index".to_string(), json!(run_index));
    row.metrics
        .insert("run_count".to_string(), json!(config.runs));
    row.metrics
        .insert("warmup_runs".to_string(), json!(config.warmup_runs));
}

pub(crate) fn benchmark_aggregate_rows(
    rows: &[EvalResult],
    config: &EvalConfig,
) -> Vec<EvalResult> {
    let mut groups: BTreeMap<String, Vec<&EvalResult>> = BTreeMap::new();
    for row in rows {
        if row.metrics.get("benchmark_sample").and_then(Value::as_bool) != Some(true) {
            continue;
        }
        if row.case_id.ends_with("::aggregate") {
            continue;
        }
        groups
            .entry(benchmark_group_key(row))
            .or_default()
            .push(row);
    }

    let mut aggregates = Vec::new();
    for group in groups.values() {
        if group.len() < 2 {
            continue;
        }
        let pass_rows = group
            .iter()
            .copied()
            .filter(|row| row.status == EvalStatus::Pass)
            .collect::<Vec<_>>();
        if pass_rows.len() < 2 {
            continue;
        }
        let first = group[0];
        let fail_count = group
            .iter()
            .filter(|row| row.status == EvalStatus::Fail)
            .count();
        let skip_count = group
            .iter()
            .filter(|row| row.status == EvalStatus::Skip)
            .count();
        let mut aggregate = first.clone();
        aggregate.case_id = format!("{}::aggregate", first.case_id);
        aggregate.status = if fail_count == 0 {
            EvalStatus::Pass
        } else {
            EvalStatus::Fail
        };
        aggregate.reason = if fail_count == 0 {
            None
        } else {
            Some("one or more benchmark samples failed".to_string())
        };
        aggregate.started_utc = utc_now();
        aggregate.elapsed_ms = group.iter().map(|row| row.elapsed_ms).sum::<u128>();
        aggregate.metrics = benchmark_aggregate_metrics(
            &pass_rows,
            config,
            &first.case_id,
            group.len(),
            fail_count,
            skip_count,
        );
        aggregates.push(aggregate);
    }
    aggregates
}

pub(crate) fn benchmark_group_key(row: &EvalResult) -> String {
    format!(
        "{}|{}|{}|{}|{}|{}|{}",
        row.battery.as_str(),
        row.suite.map(|s| s.as_str()).unwrap_or(""),
        row.case_id,
        row.dataset_item_id.as_deref().unwrap_or(""),
        row.model,
        row.prompt_hash.as_deref().unwrap_or(""),
        row.kv_mode.as_deref().unwrap_or("")
    )
}

pub(crate) fn benchmark_aggregate_metrics(
    rows: &[&EvalResult],
    config: &EvalConfig,
    source_case_id: &str,
    total_sample_count: usize,
    failed_sample_count: usize,
    skipped_sample_count: usize,
) -> BTreeMap<String, Value> {
    let mut metrics = BTreeMap::from([
        ("benchmark_aggregate".to_string(), json!(true)),
        (
            "aggregate_source_case_id".to_string(),
            json!(source_case_id),
        ),
        ("sample_count".to_string(), json!(rows.len())),
        ("total_sample_count".to_string(), json!(total_sample_count)),
        (
            "failed_sample_count".to_string(),
            json!(failed_sample_count),
        ),
        (
            "skipped_sample_count".to_string(),
            json!(skipped_sample_count),
        ),
        ("run_count".to_string(), json!(config.runs)),
        ("warmup_runs".to_string(), json!(config.warmup_runs)),
    ]);
    let mut numeric: BTreeMap<String, Vec<f64>> = BTreeMap::new();
    for row in rows {
        for (key, value) in &row.metrics {
            if benchmark_metric_excluded(key) {
                continue;
            }
            if let Some(value) = value.as_f64().filter(|value| value.is_finite()) {
                numeric.entry(key.clone()).or_default().push(value);
            }
        }
    }
    for (key, mut values) in numeric {
        if values.len() < 2 {
            continue;
        }
        values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let stats = summarize_f64_samples(&values);
        metrics.insert(format!("{key}_median"), json!(stats.median));
        metrics.insert(format!("{key}_mean"), json!(stats.mean));
        metrics.insert(format!("{key}_stddev"), json!(stats.stddev));
        metrics.insert(format!("{key}_min"), json!(stats.min));
        metrics.insert(format!("{key}_max"), json!(stats.max));
        metrics.insert(format!("{key}_p10"), json!(stats.p10));
        metrics.insert(format!("{key}_p90"), json!(stats.p90));
        if let Some(cv_pct) = stats.cv_pct {
            metrics.insert(format!("{key}_cv_pct"), json!(cv_pct));
        }
    }
    metrics
}

pub(crate) fn benchmark_metric_excluded(key: &str) -> bool {
    matches!(
        key,
        "benchmark_sample"
            | "benchmark_aggregate"
            | "run_index"
            | "run_count"
            | "warmup_runs"
            | "cache_hit"
    ) || key.ends_with("_hash")
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct F64Summary {
    mean: f64,
    median: f64,
    stddev: f64,
    min: f64,
    max: f64,
    p10: f64,
    p90: f64,
    cv_pct: Option<f64>,
}

pub(crate) fn summarize_f64_samples(sorted: &[f64]) -> F64Summary {
    let n = sorted.len();
    let mean = sorted.iter().sum::<f64>() / n as f64;
    let variance = if n > 1 {
        sorted
            .iter()
            .map(|value| {
                let delta = value - mean;
                delta * delta
            })
            .sum::<f64>()
            / (n as f64 - 1.0)
    } else {
        0.0
    };
    let stddev = variance.sqrt();
    let cv_pct = if mean.abs() > f64::EPSILON {
        Some(stddev / mean.abs() * 100.0)
    } else {
        None
    };
    F64Summary {
        mean,
        median: percentile(sorted, 0.5),
        stddev,
        min: *sorted.first().unwrap_or(&0.0),
        max: *sorted.last().unwrap_or(&0.0),
        p10: percentile(sorted, 0.10),
        p90: percentile(sorted, 0.90),
        cv_pct,
    }
}

pub(crate) fn percentile(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    if sorted.len() == 1 {
        return sorted[0];
    }
    let rank = p.clamp(0.0, 1.0) * (sorted.len() - 1) as f64;
    let lo = rank.floor() as usize;
    let hi = rank.ceil() as usize;
    if lo == hi {
        sorted[lo]
    } else {
        let weight = rank - lo as f64;
        sorted[lo] * (1.0 - weight) + sorted[hi] * weight
    }
}

pub(crate) fn result_cache_key(
    battery: BatteryId,
    config: &EvalConfig,
    ctx: &EvalContext,
    datasets: &[DatasetManifestEntry],
) -> Result<String, String> {
    let relevant_suites = if battery == BatteryId::Barrage {
        config.suites.clone()
    } else {
        Vec::new()
    };
    let relevant_datasets = if battery == BatteryId::Barrage {
        datasets.to_vec()
    } else {
        Vec::new()
    };
    let prompt_fingerprints = result_cache_prompt_fingerprints(battery);
    let evidence_json = config
        .evidence_json
        .iter()
        .map(|path| {
            json!({
                "path": path.display().to_string(),
                "hash": file_hash(path),
            })
        })
        .collect::<Vec<_>>();
    let evidence_dirs = config
        .evidence_dirs
        .iter()
        .map(|path| {
            json!({
                "path": path.display().to_string(),
                "hash": directory_hash(path),
            })
        })
        .collect::<Vec<_>>();
    let key_doc = json!({
        "schema": 1,
        "scope": "hipfire-eval-result-cache-battery-v1",
        "battery": battery,
        "suites": relevant_suites,
        "model": {
            "identifier": config.model,
            "hash": model_hash(&config.model),
        },
        "draft": config.draft.as_ref().map(|draft| json!({
            "identifier": draft,
            "hash": model_hash(draft),
        })),
        "baseline": config.baseline.as_ref().map(|baseline| json!({
            "identifier": baseline,
            "hash": model_hash(baseline),
        })),
        "reference": config.reference.as_ref().map(|reference| json!({
            "identifier": reference,
            "hash": model_hash(reference),
        })),
        "runtime": {
            "hipfire_version": env!("CARGO_PKG_VERSION"),
            "git_commit": ctx.commit_sha,
            "git_branch": ctx.git_branch,
            "git_dirty": ctx.git_dirty,
            "binary_hash": ctx.binary_hash,
            "rocm": ctx.rocm,
            "arch": ctx.arch,
            "host_profile_hash": ctx.host_profile.host_profile_hash,
            "hardware_bucket": ctx.host_profile.hardware_bucket,
            "executor": config.executor.as_str(),
            "kv_mode": config.kv_mode,
            "max_tokens": config.max_tokens,
            "dflash": config.dflash.as_str(),
            "profile": config.profile.as_str(),
            "runs": config.runs,
            "warmup_runs": config.warmup_runs,
            "benchmark": config.benchmark,
            "host_memory_class": config.host_memory_class,
            "host_memory_width_bits": config.host_memory_width_bits,
            "host_memory_bandwidth_gbps": config.host_memory_bandwidth_gbps,
        },
        "inputs": {
            "quality_json": config.quality_json.as_ref().map(|path| json!({
                "path": path.display().to_string(),
                "hash": file_hash(path),
            })),
            "kldref": config.kldref.as_ref().map(|path| json!({
                "path": path.display().to_string(),
                "hash": file_hash(path),
            })),
            "performance_json": config.performance_json.as_ref().map(|path| json!({
                "path": path.display().to_string(),
                "hash": file_hash(path),
            })),
            "evidence_json": evidence_json,
            "evidence_dirs": evidence_dirs,
            "datasets": relevant_datasets,
            "prompt_fingerprints": prompt_fingerprints,
        },
    });
    serde_json::to_string(&key_doc)
        .map(|s| stable_hash_bytes(s.as_bytes()))
        .map_err(|e| format!("serialize result cache key: {e}"))
}

pub(crate) fn result_cache_prompt_fingerprints(battery: BatteryId) -> Vec<Value> {
    result_cache_prompt_paths(battery)
        .into_iter()
        .map(|path| {
            json!({
                "path": path,
                "hash": prompt(path).map(|p| p.hash),
            })
        })
        .collect()
}

pub(crate) fn result_cache_prompt_paths(battery: BatteryId) -> Vec<&'static str> {
    match battery {
        BatteryId::Smoke => vec![
            "benchmarks/prompts/qwen2_smoke.txt",
            "benchmarks/prompts/trains-meet.txt",
        ],
        BatteryId::Coherence => vec![
            "benchmarks/prompts/coherence_capital_france.txt",
            "benchmarks/prompts/coherence_square_function.txt",
            "benchmarks/prompts/coherence_sheep_reason.txt",
            "benchmarks/prompts/tool_call_read_file.txt",
            "benchmarks/prompts/tool_call_system.txt",
            "benchmarks/prompts/coherence_lloyd_long.txt",
        ],
        BatteryId::Quality => vec!["benchmarks/quality-baselines/harness/canary.md"],
        BatteryId::Retrieval => vec!["benchmarks/prompts/trains-meet.txt"],
        BatteryId::Speed => vec!["benchmarks/prompts/lru_cache_single_blank.txt"],
        BatteryId::Dflash => vec!["benchmarks/prompts/dflash_resident_smoke.txt"],
        BatteryId::Pflash => pflash_niah_cases()
            .iter()
            .map(|case| case.fixture)
            .collect(),
        BatteryId::Agentic => vec![
            "benchmarks/prompts/agentic_pi_system.txt",
            "benchmarks/prompts/agentic_hermes_system.txt",
            "benchmarks/prompts/agentic_user_read.txt",
            "benchmarks/prompts/agentic_jinja_tools_system.txt",
            "benchmarks/prompts/agentic_jinja_tools_user.txt",
        ],
        BatteryId::Runtime => Vec::new(),
        BatteryId::PromptShape => vec!["benchmarks/prompts/lru_cache_pep8_strict.txt"],
        BatteryId::Structured => vec!["benchmarks/prompts/tool_call_read_file.txt"],
        BatteryId::Longctx => vec!["benchmarks/prompts/longprose_multidoc.jsonl"],
        BatteryId::Profile => vec!["benchmarks/prompts/dflash_resident_smoke.txt"],
        BatteryId::Calibrate => vec!["benchmarks/prompts/lru_cache_single_blank.txt"],
        BatteryId::Perplexity => {
            vec!["benchmarks/quality-baselines/slice/wikitext2-1024s-2048ctx.txt"]
        }
        BatteryId::Vision => vec!["benchmarks/prompts/vision_describe_image.txt"],
        BatteryId::Barrage | BatteryId::Cask => Vec::new(),
    }
}

pub(crate) fn result_cache_path(config: &EvalConfig, key: &str) -> PathBuf {
    config
        .result_cache
        .join(&key[..2])
        .join(format!("{key}.json"))
}

pub(crate) fn read_result_cache_entry(path: &Path, key: &str) -> Option<Vec<EvalResult>> {
    let body = fs::read_to_string(path).ok()?;
    let entry: ResultCacheEntry = serde_json::from_str(&body).ok()?;
    if entry.schema == 1 && entry.key == key {
        Some(entry.rows)
    } else {
        None
    }
}

pub(crate) fn write_result_cache_entry(
    path: &Path,
    key: &str,
    cache_mode: EvalCacheMode,
    rows: &[EvalResult],
) -> std::io::Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let entry = ResultCacheEntry {
        schema: 1,
        key: key.to_string(),
        created_utc: utc_now(),
        cache_mode,
        rows: rows.to_vec(),
    };
    write_json_pretty(path, &entry).map_err(std::io::Error::other)
}

pub(crate) fn mark_cache_hit(mut row: EvalResult, key: &str, path: &Path) -> EvalResult {
    row.metrics.insert("cache_hit".to_string(), json!(true));
    row.metrics
        .insert("cache_key".to_string(), json!(key.to_string()));
    row.metrics
        .insert("cache_path".to_string(), json!(path.display().to_string()));
    row
}

pub(crate) fn run_battery(
    battery: BatteryId,
    config: &EvalConfig,
    ctx: &EvalContext,
    datasets: &[DatasetManifestEntry],
) -> Vec<EvalResult> {
    if battery == BatteryId::Quality {
        if let Some(rows) = quality_json_rows(config, ctx) {
            return rows;
        }
        // Daemon-resident KLD scoring is the ONLY model-backed Quality path: one
        // resident forward + the shared hipfire-kld core. The standalone eval_hipfire
        // example (and the cross-engine HFKLDR `.kldref.bin` it read) has been
        // removed. Mock falls through to mock rows.
        if !matches!(config.executor, EvalExecutorMode::Mock) {
            return run_daemon_quality_rows(config, ctx);
        }
    }
    if battery == BatteryId::Speed {
        if let Some(rows) = performance_json_rows(config, ctx) {
            return rows;
        }
    }
    if config.executor == EvalExecutorMode::Mock {
        if let Some(rows) = mock_battery_rows(battery, config, ctx, datasets) {
            return rows;
        }
    }
    if config.executor == EvalExecutorMode::Direct {
        if let Some(rows) = direct_battery_rows(battery, config, ctx, datasets) {
            return rows;
        }
    }
    if config.executor == EvalExecutorMode::Daemon {
        if let Some(rows) = daemon_battery_rows(battery, config, ctx, datasets) {
            return rows;
        }
    }
    if config.executor == EvalExecutorMode::Auto && daemon_executor_available_for(config, battery) {
        if let Some(rows) = daemon_battery_rows(battery, config, ctx, datasets) {
            return rows;
        }
    }
    if config.executor == EvalExecutorMode::Examples
        || (config.executor == EvalExecutorMode::Auto && examples_executor_available_for(battery))
    {
        if let Some(rows) = examples_battery_rows(battery, config, ctx, datasets) {
            return rows;
        }
    }
    match battery {
        BatteryId::Smoke => vec![
            skip_row(
                battery,
                None,
                "load_metadata",
                None,
                "daemon-backed model load is not implemented yet",
                config,
                ctx,
                None,
            ),
            skip_row(
                battery,
                None,
                "finite_greedy_decode",
                None,
                "daemon-backed greedy decode is not implemented yet",
                config,
                ctx,
                prompt("benchmarks/prompts/qwen2_smoke.txt"),
            ),
            skip_row(
                battery,
                None,
                "multi_turn_reset_recall",
                None,
                "daemon-backed multi-turn session is not implemented yet",
                config,
                ctx,
                prompt("benchmarks/prompts/trains-meet.txt"),
            ),
        ],
        BatteryId::Coherence => vec![skip_row(
            battery,
            None,
            "runtime_detector_canary",
            None,
            "daemon-backed coherence probe is not available in this environment",
            config,
            ctx,
            prompt("benchmarks/prompts/qwen2_smoke.txt"),
        )],
        BatteryId::Quality => vec![skip_row(
            battery,
            None,
            "kld_reference_slice",
            None,
            "quality-baseline subprocess integration is not implemented yet",
            config,
            ctx,
            prompt("benchmarks/quality-baselines/harness/canary.md"),
        )],
        BatteryId::Retrieval => vec![pass_row(
            battery,
            None,
            "synthetic_seed_fixture",
            None,
            config,
            ctx,
            prompt("benchmarks/prompts/trains-meet.txt"),
            BTreeMap::from([
                (
                    "fixture_kind".to_string(),
                    json!("hipfire_native_synthetic"),
                ),
                ("seed".to_string(), json!(1)),
            ]),
        )],
        BatteryId::Speed => vec![skip_row(
            battery,
            None,
            "pp32_pp128_ttft_decode",
            None,
            "daemon-backed speed anchors are not implemented yet",
            config,
            ctx,
            prompt("benchmarks/prompts/lru_cache_single_blank.txt"),
        )],
        BatteryId::Dflash => {
            let mut rows = vec![skip_row(
                battery,
                None,
                "ar_coherence_anchor",
                None,
                "daemon-backed AR anchor is not implemented yet",
                config,
                ctx,
                prompt("benchmarks/prompts/dflash_resident_smoke.txt"),
            )];
            let reason = if matches!(config.dflash, DflashMode::Off) {
                "DFlash disabled by --dflash off"
            } else if config.draft.is_none() {
                "DFlash draft not provided or discoverable by this runner"
            } else {
                "daemon-backed DFlash anchor is not implemented yet"
            };
            rows.push(skip_row(
                battery,
                None,
                "dflash_anchor",
                None,
                reason,
                config,
                ctx,
                prompt("benchmarks/prompts/dflash_resident_smoke.txt"),
            ));
            rows
        }
        BatteryId::Pflash => pflash_niah_cases()
            .iter()
            .map(|case| {
                skip_row(
                    battery,
                    None,
                    case.label,
                    None,
                    "pflash examples executor is not available in this environment",
                    config,
                    ctx,
                    prompt(case.fixture),
                )
            })
            .collect(),
        BatteryId::Agentic => agentic_cases()
            .iter()
            .map(|case| {
                skip_row(
                    battery,
                    None,
                    case.label,
                    None,
                    "daemon-backed agentic coherence executor is not available in this environment",
                    config,
                    ctx,
                    combined_prompt_ref(
                        case.system_path,
                        "benchmarks/prompts/agentic_user_read.txt",
                    ),
                )
            })
            .collect(),
        BatteryId::Runtime => runtime_cases()
            .iter()
            .map(|case| {
                skip_row(
                    battery,
                    None,
                    case.label,
                    None,
                    "server/runtime evidence script executor is not available in this environment",
                    config,
                    ctx,
                    None,
                )
            })
            .collect(),
        BatteryId::PromptShape => vec![pass_row(
            battery,
            None,
            "whitespace_fixture_hash",
            None,
            config,
            ctx,
            prompt("benchmarks/prompts/lru_cache_pep8_strict.txt"),
            BTreeMap::from([("normalization_probe".to_string(), json!("newline_runs"))]),
        )],
        BatteryId::Structured => vec![pass_row(
            battery,
            None,
            "tool_call_fixture_hash",
            None,
            config,
            ctx,
            prompt("benchmarks/prompts/tool_call_read_file.txt"),
            BTreeMap::from([("structured_probe".to_string(), json!("tool_call_jsonish"))]),
        )],
        BatteryId::Barrage => barrage_rows(config, ctx, datasets),
        BatteryId::Calibrate => vec![skip_row(
            battery,
            None,
            "single_load_hessian_consistency",
            None,
            "calibration requires the examples executor (collect_artifacts); none was available",
            config,
            ctx,
            prompt("benchmarks/prompts/lru_cache_single_blank.txt"),
        )],
        BatteryId::Perplexity => vec![skip_row(
            battery,
            None,
            "corpus_perplexity",
            None,
            "perplexity requires the examples executor (perplexity bin); none was available",
            config,
            ctx,
            prompt("benchmarks/quality-baselines/slice/wikitext2-1024s-2048ctx.txt"),
        )],
        BatteryId::Longctx | BatteryId::Vision | BatteryId::Cask | BatteryId::Profile => {
            vec![skip_row(
                battery,
                None,
                "not_implemented",
                None,
                "not implemented in this scaffold",
                config,
                ctx,
                None,
            )]
        }
    }
}
