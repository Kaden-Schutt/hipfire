// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Evidence, provenance, comparison, and admission artifact assembly.
//!
//! Builds the run's evidence/provenance/metadata artifacts, the comparison
//! artifact (vs a baseline), and the admission artifact (required-evidence
//! findings + verdict) from the collected EvalResult rows. Extracted verbatim
//! from the former `hipfire-eval/src/lib.rs` monolith (no behavior change).

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use serde_json::{json, Value};

use crate::*;

pub(crate) fn write_evidence_artifacts(
    dir: &Path,
    config: &EvalConfig,
    datasets: &[DatasetManifestEntry],
    results: &[EvalResult],
    comparison: &ComparisonArtifact,
    admission: &AdmissionArtifact,
    ctx: &EvalContext,
) -> Result<BTreeMap<String, Value>, String> {
    let mut out = BTreeMap::new();
    for spec in STANDARD_EVIDENCE_ARTIFACT_SPECS {
        let value = evidence_artifact_value(
            spec.kind,
            spec.expected_metrics,
            config,
            datasets,
            results,
            ctx,
        );
        let status = value
            .get("status")
            .and_then(Value::as_str)
            .unwrap_or("not_collected")
            .to_string();
        write_json_pretty(&dir.join(spec.file), &value)?;
        out.insert(
            spec.kind.to_string(),
            artifact_index_entry_from_value(
                format!("artifacts/{}", spec.file),
                status,
                &value,
                ctx,
            ),
        );
    }
    let comparison_json = comparison_artifact_value(comparison)?;
    write_json_pretty(&dir.join("comparisons.json"), &comparison_json)?;
    let comparisons_entry = comparison_artifact_index_entry_json(
        "artifacts/comparisons.json",
        format!("{:?}", comparison.status).to_lowercase(),
        comparison.cases.len(),
        &artifact_index_context(ctx),
    );
    out.insert("comparisons".to_string(), comparisons_entry);
    let admission_json = admission_artifact_value(admission)?;
    write_json_pretty(&dir.join("admission.json"), &admission_json)?;
    let admission_entry = admission_artifact_index_entry_json(
        "artifacts/admission.json",
        format!("{:?}", admission.status).to_lowercase(),
        admission.verdict.clone(),
        admission.findings.len(),
        &artifact_index_context(ctx),
    );
    out.insert("admission".to_string(), admission_entry);
    if let Some((path, row_count)) = write_gpqa_prompt_artifact(dir, config, datasets)? {
        let entry = prompt_artifact_index_entry_json(
            path,
            "materialized",
            "gpqa_prompts",
            row_count,
            &artifact_index_context(ctx),
        );
        out.insert("gpqa_prompts".to_string(), entry);
    }
    if let Some((path, row_count)) = write_barrage_prompt_artifact(dir, datasets)? {
        let entry = prompt_artifact_index_entry_json(
            path,
            "materialized",
            "barrage_prompts",
            row_count,
            &artifact_index_context(ctx),
        );
        out.insert("barrage_prompts".to_string(), entry);
    }
    let run_metadata = run_metadata_artifact_value(config, ctx);
    write_json_pretty(&dir.join("run_metadata.json"), &run_metadata)?;
    out.insert(
        "run_metadata".to_string(),
        artifact_index_entry("artifacts/run_metadata.json", "collected", ctx),
    );
    if dir.join("host_profile.json").exists() {
        let artifact_status = fs::read_to_string(dir.join("host_profile.json"))
            .ok()
            .and_then(|body| serde_json::from_str::<Value>(&body).ok())
            .and_then(|value| {
                value
                    .get("status")
                    .and_then(Value::as_str)
                    .map(str::to_string)
            })
            .unwrap_or_else(|| "collected".to_string());
        let entry = host_profile_artifact_index_entry_json(
            "artifacts/host_profile.json",
            artifact_status,
            &artifact_index_context(ctx),
        );
        out.insert("host_profile".to_string(), entry);
    }
    Ok(out)
}

pub(crate) fn comparison_artifact_value(comparison: &ComparisonArtifact) -> Result<Value, String> {
    let cases = serde_json::to_value(&comparison.cases)
        .map_err(|err| format!("serialize comparison cases: {err}"))?
        .as_array()
        .cloned()
        .unwrap_or_default();
    Ok(comparison_artifact_json(EvidenceComparisonArtifact {
        provenance: comparison.provenance.clone(),
        status: eval_status_str(comparison.status).to_string(),
        reason: comparison.reason.clone(),
        baseline: comparison.baseline.clone(),
        reference: comparison.reference.clone(),
        cases,
    }))
}

pub(crate) fn admission_artifact_value(admission: &AdmissionArtifact) -> Result<Value, String> {
    let findings = serde_json::to_value(&admission.findings)
        .map_err(|err| format!("serialize admission findings: {err}"))?
        .as_array()
        .cloned()
        .unwrap_or_default();
    Ok(admission_artifact_json(EvidenceAdmissionArtifact {
        provenance: admission.provenance.clone(),
        status: eval_status_str(admission.status).to_string(),
        verdict: admission.verdict.clone(),
        reason: admission.reason.clone(),
        required_evidence: admission
            .required_evidence
            .iter()
            .map(evidence_admission_evidence)
            .collect(),
        observed_evidence: admission
            .observed_evidence
            .iter()
            .map(evidence_admission_evidence)
            .collect(),
        findings,
    }))
}

pub(crate) fn evidence_admission_evidence(
    evidence: &AdmissionEvidence,
) -> EvidenceAdmissionEvidence {
    EvidenceAdmissionEvidence {
        kind: evidence.kind.clone(),
        status: eval_status_str(evidence.status).to_string(),
        rows: evidence.rows,
        reason: evidence.reason.clone(),
    }
}

pub(crate) fn artifact_index_entry(
    path: impl Into<String>,
    status: impl Into<String>,
    ctx: &EvalContext,
) -> Value {
    evidence_artifact_index_entry_json(path, status, &artifact_index_context(ctx))
}

pub(crate) fn artifact_index_entry_from_value(
    path: impl Into<String>,
    status: impl Into<String>,
    value: &Value,
    ctx: &EvalContext,
) -> Value {
    evidence_artifact_index_entry_from_value_json(path, status, value, &artifact_index_context(ctx))
}

pub(crate) fn artifact_index_context(ctx: &EvalContext) -> EvidenceArtifactIndexContext {
    EvidenceArtifactIndexContext {
        provenance: run_provenance(ctx),
        host_profile_hash: ctx.host_profile.host_profile_hash.clone(),
        hardware_bucket: ctx.host_profile.hardware_bucket.clone(),
    }
}

pub(crate) fn run_provenance(ctx: &EvalContext) -> RunProvenance {
    RunProvenance {
        runner: "hipfire-eval".to_string(),
        runner_version: env!("CARGO_PKG_VERSION").to_string(),
        hipfire_version: env!("CARGO_PKG_VERSION").to_string(),
        git_commit: ctx.commit_sha.clone(),
        git_branch: ctx.git_branch.clone(),
        git_describe: ctx.git_describe.clone(),
        git_dirty: ctx.git_dirty,
        binary_hash: ctx.binary_hash.clone(),
        arch: ctx.arch.clone(),
        rocm: ctx.rocm.clone(),
    }
}

pub(crate) fn run_provenance_value(ctx: &EvalContext) -> Value {
    run_provenance_json(run_provenance(ctx))
}

pub(crate) fn run_metadata_artifact_value(config: &EvalConfig, ctx: &EvalContext) -> Value {
    run_metadata_artifact_json(RunMetadataArtifact {
        created_utc: utc_now(),
        provenance: run_provenance(ctx),
        host_profile: serde_json::to_value(&ctx.host_profile).unwrap_or_else(|_| json!({})),
        host_profile_hash: ctx.host_profile.host_profile_hash.clone(),
        hardware_bucket: ctx.host_profile.hardware_bucket.clone(),
        config: RunMetadataConfig {
            tier: config.tier.as_str().to_string(),
            tier_budget: serde_json::to_value(config.tier.budget()).unwrap_or_else(|_| json!({})),
            batteries: config
                .batteries
                .iter()
                .map(|b| b.as_str().to_string())
                .collect(),
            suites: config
                .suites
                .iter()
                .map(|s| s.as_str().to_string())
                .collect(),
            executor: config.executor.as_str().to_string(),
            kv_mode: config.kv_mode.clone(),
            max_tokens: config.max_tokens,
            profile: config.profile.as_str().to_string(),
            dflash: config.dflash.as_str().to_string(),
            runs: config.runs,
            warmup_runs: config.warmup_runs,
            benchmark: config.benchmark,
            host_memory_class: config.host_memory_class.clone(),
            host_memory_width_bits: config.host_memory_width_bits,
            host_memory_bandwidth_gbps: config.host_memory_bandwidth_gbps,
            result_cache: config.result_cache.display().to_string(),
            cache_mode: config.cache_mode.as_str().to_string(),
        },
        models: RunMetadataModels {
            candidate: config.model.clone(),
            draft: config.draft.clone(),
            baseline: config.baseline.clone(),
            reference: config.reference.clone(),
        },
    })
}

pub(crate) fn evidence_artifact_value(
    kind: &str,
    expected_metrics: &[&str],
    config: &EvalConfig,
    datasets: &[DatasetManifestEntry],
    results: &[EvalResult],
    ctx: &EvalContext,
) -> Value {
    let mut records = evidence_records(kind, results);
    let (external_records, external_errors) = external_evidence_records(kind, config, results, ctx);
    records.extend(external_records);
    let collection_policy = evidence_collection_policy(
        kind,
        records.len(),
        &external_errors,
        config.profile.as_str(),
    );
    evidence_artifact_json(EvidenceArtifact {
        kind: kind.to_string(),
        provenance: run_provenance_value(ctx),
        collection_policy,
        collection: EvidenceArtifactCollection {
            source: "hipfire-eval".to_string(),
            executor: config.executor.as_str().to_string(),
            evidence_json: config
                .evidence_json
                .iter()
                .map(|p| p.display().to_string())
                .collect(),
            evidence_dirs: config
                .evidence_dirs
                .iter()
                .map(|p| p.display().to_string())
                .collect(),
            requires_model_execution: true,
            profiling_mode: config.profile.as_str().to_string(),
            dflash_mode: config.dflash.as_str().to_string(),
        },
        config: EvidenceArtifactConfig {
            tier: config.tier.as_str().to_string(),
            batteries: config
                .batteries
                .iter()
                .map(|b| b.as_str().to_string())
                .collect(),
            suites: config
                .suites
                .iter()
                .map(|s| s.as_str().to_string())
                .collect(),
            kv_mode: config.kv_mode.clone(),
            max_tokens: config.max_tokens,
        },
        models: EvidenceArtifactModels {
            candidate: config.model.clone(),
            draft: config.draft.clone(),
            baseline: config.baseline.clone(),
            reference: config.reference.clone(),
        },
        datasets: EvidenceArtifactDatasetStatus {
            total: datasets.len(),
            pass: datasets
                .iter()
                .filter(|d| d.status == EvalStatus::Pass)
                .count(),
            skip: datasets
                .iter()
                .filter(|d| d.status == EvalStatus::Skip)
                .count(),
            fail: datasets
                .iter()
                .filter(|d| d.status == EvalStatus::Fail)
                .count(),
        },
        expected_metrics: expected_metrics
            .iter()
            .map(|metric| (*metric).to_string())
            .collect(),
        records,
    })
}

pub(crate) fn external_evidence_records(
    kind: &str,
    config: &EvalConfig,
    results: &[EvalResult],
    ctx: &EvalContext,
) -> (Vec<Value>, Vec<String>) {
    let mut records = Vec::new();
    let mut errors = Vec::new();
    let mut sources = config.evidence_json.clone();
    for dir in &config.evidence_dirs {
        match runtime_evidence_paths_in_dir(dir) {
            Ok(mut paths) => sources.append(&mut paths),
            Err(err) => errors.push(err),
        }
    }
    for dir in runtime_evidence_dirs_from_results(results) {
        match runtime_evidence_paths_in_dir(&dir) {
            Ok(mut paths) => sources.append(&mut paths),
            Err(err) => errors.push(err),
        }
    }
    sources.sort();
    sources.dedup();
    for path in sources {
        match external_evidence_records_from_path(kind, &path, config, ctx) {
            Ok(mut extracted) => records.append(&mut extracted),
            Err(err) => errors.push(err),
        }
    }
    (records, errors)
}

pub(crate) fn runtime_evidence_dirs_from_results(results: &[EvalResult]) -> Vec<PathBuf> {
    let mut dirs = Vec::new();
    for row in results {
        if row.status != EvalStatus::Pass {
            continue;
        }
        if let Some(dir) = row
            .metrics
            .get("runtime_evidence_dir")
            .and_then(Value::as_str)
            .filter(|dir| !dir.is_empty())
        {
            dirs.push(PathBuf::from(dir));
        }
    }
    dirs.sort();
    dirs.dedup();
    dirs
}

pub(crate) fn runtime_evidence_paths_in_dir(dir: &Path) -> Result<Vec<PathBuf>, String> {
    standard_evidence_paths_in_dir(dir)
}

pub(crate) fn external_evidence_records_from_path(
    kind: &str,
    path: &Path,
    config: &EvalConfig,
    ctx: &EvalContext,
) -> Result<Vec<Value>, String> {
    let body = fs::read_to_string(path).map_err(|err| format!("read {}: {err}", path.display()))?;
    let value: Value =
        serde_json::from_str(&body).map_err(|err| format!("parse {}: {err}", path.display()))?;
    Ok(extract_external_evidence_records_json(
        kind,
        path,
        &value,
        external_evidence_context(config, ctx),
    ))
}

pub(crate) fn external_evidence_context(config: &EvalConfig, ctx: &EvalContext) -> Value {
    json!({
        "schema": 1,
        "runner": "hipfire-eval",
        "runner_version": env!("CARGO_PKG_VERSION"),
        "hipfire_version": env!("CARGO_PKG_VERSION"),
        "model": config.model.clone(),
        "draft": config.draft.clone(),
        "baseline": config.baseline.clone(),
        "reference": config.reference.clone(),
        "model_hash": model_hash(&config.model),
        "draft_hash": config.draft.as_deref().and_then(model_hash),
        "baseline_hash": config.baseline.as_deref().and_then(model_hash),
        "reference_hash": config.reference.as_deref().and_then(model_hash),
        "git_commit": ctx.commit_sha.clone(),
        "git_branch": ctx.git_branch.clone(),
        "git_describe": ctx.git_describe.clone(),
        "git_dirty": ctx.git_dirty,
        "binary_hash": ctx.binary_hash.clone(),
        "arch": ctx.arch.clone(),
        "rocm": ctx.rocm.clone(),
    })
}

pub(crate) fn evidence_records(kind: &str, results: &[EvalResult]) -> Vec<Value> {
    let batteries: &[BatteryId] = match kind {
        "quality" => &[],
        "performance" => &[],
        "phase_timings" => &[],
        "launch_counts" => &[],
        "moe_router_histogram" => &[],
        "memory" => &[],
        "dflash_trace" => &[BatteryId::Dflash],
        "path_c_trace" => &[BatteryId::Dflash],
        "module_evidence" => &[],
        "profiling" => &[],
        "coherence" => &[
            BatteryId::Coherence,
            BatteryId::Longctx,
            BatteryId::Dflash,
            BatteryId::Agentic,
        ],
        _ => return Vec::new(),
    };
    results
        .iter()
        .filter(|row| {
            (row.status == EvalStatus::Pass
                || (kind == "coherence" && row.status == EvalStatus::Fail))
                && if kind == "performance" {
                    has_performance_metric(&row.metrics)
                } else if kind == "quality" {
                    has_quality_metric(row)
                } else if kind == "phase_timings" {
                    has_phase_timing_metric(&row.metrics, row.elapsed_ms)
                } else if kind == "memory" {
                    has_memory_metric(&row.metrics)
                } else if kind == "launch_counts" {
                    has_launch_count_metric(&row.metrics)
                } else if kind == "moe_router_histogram" {
                    has_moe_router_metric(&row.metrics)
                } else if kind == "profiling" {
                    has_profiling_metric(&row.metrics)
                } else if kind == "path_c_trace" {
                    has_path_c_trace_metric(&row.metrics)
                } else if kind == "module_evidence" {
                    has_module_evidence_metric(&row.metrics)
                } else {
                    batteries.contains(&row.battery)
                }
        })
        .map(|row| {
            let metrics = match kind {
                "phase_timings" => phase_timing_metrics(&row.metrics, row.elapsed_ms),
                "memory" => memory_metrics(&row.metrics),
                "launch_counts" => launch_count_metrics(&row.metrics),
                "moe_router_histogram" => moe_router_metrics(&row.metrics),
                "profiling" => profiling_metrics(&row.metrics),
                "dflash_trace" => dflash_trace_metrics(&row.metrics),
                "path_c_trace" => path_c_trace_metrics(&row.metrics),
                "module_evidence" => module_evidence_metrics(&row.metrics),
                _ => row.metrics.clone(),
            };
            evidence_record_json(EvidenceRecord {
                battery: row.battery.as_str().to_string(),
                suite: row.suite.map(|s| s.as_str().to_string()),
                case_id: row.case_id.clone(),
                dataset_item_id: row.dataset_item_id.clone(),
                dataset_source: row.dataset_source.clone(),
                dataset_repo_id: row.dataset_repo_id.clone(),
                dataset_revision: row.dataset_revision.clone(),
                dataset_digest: row.dataset_digest.clone(),
                dataset_license: row.dataset_license.clone(),
                dataset_cache_path: row.dataset_cache_path.clone(),
                model: row.model.clone(),
                model_hash: row.model_hash.clone(),
                draft: row.draft.clone(),
                draft_hash: row.draft_hash.clone(),
                baseline: row.baseline.clone(),
                baseline_hash: row.baseline_hash.clone(),
                reference: row.reference.clone(),
                reference_hash: row.reference_hash.clone(),
                prompt_hash: row.prompt_hash.clone(),
                prompt_path: row.prompt_path.clone(),
                metrics,
                elapsed_ms: row.elapsed_ms,
            })
        })
        .collect()
}

pub(crate) fn has_quality_metric(row: &EvalResult) -> bool {
    row.battery == BatteryId::Quality || has_quality_signal_metric(&row.metrics)
}

pub(crate) fn metric_string(metrics: &BTreeMap<String, Value>, key: &str) -> Option<String> {
    metrics.get(key).and_then(Value::as_str).map(str::to_string)
}

pub(crate) fn add_dataset_provenance_metrics(
    metrics: &mut BTreeMap<String, Value>,
    dataset: &DatasetManifestEntry,
) {
    metrics.insert("dataset_source".to_string(), json!(dataset.source));
    metrics.insert("dataset_cache_path".to_string(), json!(dataset.cache_path));
    if let Some(repo_id) = &dataset.repo_id {
        metrics.insert("dataset_repo_id".to_string(), json!(repo_id));
    }
    if let Some(revision) = &dataset.revision {
        metrics.insert("dataset_revision".to_string(), json!(revision));
    }
    if let Some(digest) = &dataset.digest {
        metrics.insert("dataset_digest".to_string(), json!(digest));
    }
    if let Some(license) = &dataset.license {
        metrics.insert("dataset_license".to_string(), json!(license));
    }
}

pub(crate) fn build_comparison_artifact(
    config: &EvalConfig,
    rows: &[EvalResult],
    ctx: &EvalContext,
) -> ComparisonArtifact {
    let mut cases = Vec::new();
    let baseline_model = config.baseline.as_deref();
    let reference_model = config.reference.as_deref();
    if baseline_model.is_none() && reference_model.is_none() {
        return ComparisonArtifact {
            schema: 1,
            provenance: run_provenance(ctx),
            status: EvalStatus::Skip,
            reason: Some("no --baseline or --reference provided".to_string()),
            baseline: None,
            reference: None,
            cases,
        };
    }

    let mut by_key: BTreeMap<String, Vec<&EvalResult>> = BTreeMap::new();
    for row in rows {
        by_key.entry(comparison_key(row)).or_default().push(row);
    }

    for (key, grouped) in by_key {
        let Some(candidate) = grouped.iter().copied().find(|r| r.model == config.model) else {
            continue;
        };
        let baseline = baseline_model.and_then(|baseline_model| {
            grouped
                .iter()
                .copied()
                .find(|r| r.model == baseline_model)
                .and_then(|base| compare_metric_maps(candidate, base))
        });
        let reference = reference_model.and_then(|reference_model| {
            grouped
                .iter()
                .copied()
                .find(|r| r.model == reference_model)
                .and_then(|reference| compare_metric_maps(candidate, reference))
        });
        if baseline.is_none() && reference.is_none() {
            continue;
        }
        cases.push(ComparisonCase {
            key,
            battery: candidate.battery,
            suite: candidate.suite,
            case_id: candidate.case_id.clone(),
            dataset_item_id: candidate.dataset_item_id.clone(),
            baseline,
            reference,
        });
    }

    let status = if cases.is_empty() {
        EvalStatus::Skip
    } else {
        EvalStatus::Pass
    };
    let reason = if cases.is_empty() {
        Some(
            "candidate rows exist, but matching baseline/reference metric rows are absent"
                .to_string(),
        )
    } else {
        None
    };
    ComparisonArtifact {
        schema: 1,
        provenance: run_provenance(ctx),
        status,
        reason,
        baseline: config.baseline.clone(),
        reference: config.reference.clone(),
        cases,
    }
}

pub(crate) fn comparison_key(row: &EvalResult) -> String {
    format!(
        "{}|{}|{}|{}",
        row.battery.as_str(),
        row.suite.map(|s| s.as_str()).unwrap_or(""),
        row.case_id,
        row.dataset_item_id.as_deref().unwrap_or("")
    )
}

pub(crate) fn compare_metric_maps(
    candidate: &EvalResult,
    comparator: &EvalResult,
) -> Option<MetricComparison> {
    let mut metrics = BTreeMap::new();
    for (name, candidate_value) in &candidate.metrics {
        let Some(candidate_num) = candidate_value.as_f64() else {
            continue;
        };
        let Some(comparator_num) = comparator.metrics.get(name).and_then(Value::as_f64) else {
            continue;
        };
        let delta = candidate_num - comparator_num;
        let relative_delta = if comparator_num.abs() > f64::EPSILON {
            Some(delta / comparator_num)
        } else {
            None
        };
        metrics.insert(
            name.clone(),
            MetricDelta {
                candidate: candidate_num,
                comparator: comparator_num,
                delta,
                relative_delta,
                direction: evidence_metric_direction(name, delta),
            },
        );
    }
    if metrics.is_empty() {
        None
    } else {
        Some(MetricComparison {
            model: comparator.model.clone(),
            metrics,
        })
    }
}

pub(crate) fn build_admission_artifact(
    config: &EvalConfig,
    rows: &[EvalResult],
    comparison: &ComparisonArtifact,
    ctx: &EvalContext,
) -> AdmissionArtifact {
    let required_kinds = required_admission_evidence(config);
    let required_evidence: Vec<AdmissionEvidence> = required_kinds
        .iter()
        .map(|(kind, batteries)| {
            let pass_rows = rows
                .iter()
                .filter(|row| {
                    row.status == EvalStatus::Pass
                        && required_evidence_row_matches(kind, batteries, row)
                })
                .count();
            AdmissionEvidence {
                kind: (*kind).to_string(),
                status: if pass_rows > 0 {
                    EvalStatus::Pass
                } else {
                    EvalStatus::Skip
                },
                rows: pass_rows,
                reason: if pass_rows > 0 {
                    None
                } else {
                    Some(format!("no passing {kind} evidence rows"))
                },
            }
        })
        .collect();
    let observed_evidence = observed_admission_evidence(config, rows, ctx);

    let missing = required_evidence
        .iter()
        .filter(|e| e.status != EvalStatus::Pass)
        .count();
    if config.baseline.is_none() {
        return AdmissionArtifact {
            schema: 1,
            provenance: run_provenance(ctx),
            status: EvalStatus::Skip,
            verdict: "incomplete".to_string(),
            reason: Some("admission requires --baseline for candidate comparison".to_string()),
            required_evidence,
            observed_evidence,
            findings: Vec::new(),
        };
    }
    if missing > 0 {
        return AdmissionArtifact {
            schema: 1,
            provenance: run_provenance(ctx),
            status: EvalStatus::Skip,
            verdict: "incomplete".to_string(),
            reason: Some(format!("{missing} required evidence kind(s) missing")),
            required_evidence,
            observed_evidence,
            findings: Vec::new(),
        };
    }
    if comparison.status != EvalStatus::Pass {
        return AdmissionArtifact {
            schema: 1,
            provenance: run_provenance(ctx),
            status: EvalStatus::Skip,
            verdict: "incomplete".to_string(),
            reason: comparison
                .reason
                .clone()
                .or_else(|| Some("no comparable baseline/reference metric rows".to_string())),
            required_evidence,
            observed_evidence,
            findings: Vec::new(),
        };
    }

    let mut findings = Vec::new();
    for case in &comparison.cases {
        collect_admission_findings(case, case.baseline.as_ref(), "baseline", &mut findings);
        collect_admission_findings(case, case.reference.as_ref(), "reference", &mut findings);
    }
    let has_reject = findings.iter().any(|f| f.severity == "reject");
    let has_review = findings.iter().any(|f| f.severity == "review");
    let verdict_policy = admission_verdict_policy(has_reject, has_review);

    AdmissionArtifact {
        schema: 1,
        provenance: run_provenance(ctx),
        status: eval_status_from_artifact_status(verdict_policy.status),
        verdict: verdict_policy.verdict.to_string(),
        reason: verdict_policy.reason.map(str::to_string),
        required_evidence,
        observed_evidence,
        findings,
    }
}

pub(crate) fn eval_status_from_artifact_status(status: &str) -> EvalStatus {
    match status {
        "pass" => EvalStatus::Pass,
        "fail" => EvalStatus::Fail,
        "skip" => EvalStatus::Skip,
        _ => EvalStatus::Skip,
    }
}

pub(crate) fn required_evidence_row_matches(
    kind: &str,
    batteries: &[BatteryId],
    row: &EvalResult,
) -> bool {
    if kind == "quality" {
        return has_quality_metric(row);
    }
    batteries.contains(&row.battery)
}

pub(crate) fn required_admission_evidence(
    config: &EvalConfig,
) -> Vec<(&'static str, Vec<BatteryId>)> {
    required_admission_evidence_requirements(
        config.batteries.iter().map(|battery| battery.as_str()),
    )
    .into_iter()
    .map(|requirement| {
        let batteries = requirement
            .batteries
            .into_iter()
            .map(|battery| BatteryId::parse(battery).expect("evidence admission battery is known"))
            .collect();
        (requirement.kind, batteries)
    })
    .collect()
}

pub(crate) fn observed_admission_evidence(
    config: &EvalConfig,
    rows: &[EvalResult],
    ctx: &EvalContext,
) -> Vec<AdmissionEvidence> {
    OBSERVED_ADMISSION_EVIDENCE_KINDS
        .iter()
        .copied()
        .map(|kind| observed_evidence_for_kind(kind, config, rows, ctx))
        .collect()
}

pub(crate) fn observed_evidence_for_kind(
    kind: &str,
    config: &EvalConfig,
    rows: &[EvalResult],
    ctx: &EvalContext,
) -> AdmissionEvidence {
    let mut records = evidence_records(kind, rows);
    let (mut external_records, external_errors) =
        external_evidence_records(kind, config, rows, ctx);
    records.append(&mut external_records);
    if !external_errors.is_empty() {
        return AdmissionEvidence {
            kind: kind.to_string(),
            status: EvalStatus::Fail,
            rows: records.len(),
            reason: Some(external_errors.join("; ")),
        };
    }
    if !records.is_empty() {
        return AdmissionEvidence {
            kind: kind.to_string(),
            status: EvalStatus::Pass,
            rows: records.len(),
            reason: None,
        };
    }
    AdmissionEvidence {
        kind: kind.to_string(),
        status: EvalStatus::Skip,
        rows: 0,
        reason: Some(observed_evidence_missing_reason(kind, config)),
    }
}

pub(crate) fn observed_evidence_missing_reason(kind: &str, config: &EvalConfig) -> String {
    if kind == "profiling" && config.profile == ProfileMode::Off {
        "profiling disabled by --profile off".to_string()
    } else if kind == "profiling" && config.profile == ProfileMode::Passive {
        "passive profiling requested; no profiling evidence rows collected".to_string()
    } else {
        format!("no observed {kind} evidence rows")
    }
}

pub(crate) fn collect_admission_findings(
    case: &ComparisonCase,
    comparison: Option<&MetricComparison>,
    comparator: &str,
    out: &mut Vec<AdmissionFinding>,
) {
    let Some(comparison) = comparison else {
        return;
    };
    for (metric, delta) in &comparison.metrics {
        if delta.direction != "regressed" {
            continue;
        }
        let severity = if admission_metric_is_quality(case.battery.as_str(), metric) {
            "reject"
        } else {
            "review"
        };
        out.push(AdmissionFinding {
            severity: severity.to_string(),
            battery: case.battery,
            suite: case.suite,
            case_id: case.case_id.clone(),
            dataset_item_id: case.dataset_item_id.clone(),
            comparator: comparator.to_string(),
            metric: metric.clone(),
            direction: delta.direction.clone(),
            delta: delta.delta,
            relative_delta: delta.relative_delta,
        });
    }
}
