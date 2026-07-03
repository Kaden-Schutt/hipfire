// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Shared framework for the `hipfire-eval` runner.
//!
//! This module establishes the stable CLI, manifest, JSONL, dataset provenance,
//! comparison, and evidence-artifact contract. Model-backed scoring uses
//! daemon-backed rows where that path is implemented, falls back to Hipfire
//! example binaries for specialized gates, and otherwise emits explicit skip
//! rows rather than silently dropping batteries.

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::{BTreeMap, HashMap};
use std::ffi::OsStr;
use std::fs::{self, File, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

use hipfire_evidence::{
    admission_artifact_index_entry_json, admission_artifact_json, admission_metric_is_quality,
    admission_verdict_policy, comparison_artifact_index_entry_json,
    comparison_artifact_json, dflash_trace_metrics, directory_hash,
    evidence_artifact_index_entry_from_value_json, evidence_artifact_index_entry_json,
    evidence_artifact_json, evidence_collection_policy, evidence_metric_direction,
    evidence_record_json, extract_external_evidence_records_json,
    has_launch_count_metric, has_memory_metric, has_module_evidence_metric, has_moe_router_metric,
    has_path_c_trace_metric, has_performance_metric, has_phase_timing_metric, has_profiling_metric,
    has_quality_metric as has_quality_signal_metric, host_profile_artifact_index_entry_json,
    launch_count_metrics, list_files, memory_metrics, module_evidence_metrics,
    moe_router_metrics, path_c_trace_metrics, phase_timing_metrics, profiling_metrics,
    prompt_artifact_index_entry_json, required_admission_evidence_requirements,
    run_metadata_artifact_json, run_provenance_json, standard_evidence_paths_in_dir,
    AdmissionArtifact as EvidenceAdmissionArtifact, AdmissionEvidence as EvidenceAdmissionEvidence,
    ComparisonArtifact as EvidenceComparisonArtifact, EvalStatus, EvidenceArtifact,
    EvidenceArtifactCollection, EvidenceArtifactConfig, EvidenceArtifactDatasetStatus,
    EvidenceArtifactIndexContext, EvidenceArtifactModels, EvidenceRecord, HostProfile,
    RunMetadataArtifact, RunMetadataConfig, RunMetadataModels, RunProvenance,
    OBSERVED_ADMISSION_EVIDENCE_KINDS, STANDARD_EVIDENCE_ARTIFACT_SPECS,
};
use hipfire_hash::{file_hash, stable_hash_bytes, stable_hash_file_fallback};
use hipfire_model::{
    discover_dflash_draft_for_model, model_artifact_stem, model_hash, model_manifest_entry,
    ModelLoadParams, ModelLoadedResponse, ModelManifestEntry,
};

mod config;
pub use config::*;
mod datasets;
use datasets::*;
mod driver;
use driver::*;
mod executor_daemon;
use executor_daemon::*;
mod executor_examples;
use executor_examples::*;
mod executor_mock;
use executor_mock::*;
mod executor_tinyquant;
use executor_tinyquant::*;
// Host-profile collection lives in the HIP-independent leaf crate
// hipfire-sysinfo so hipfire-runtime can collect it without depending on this
// eval harness (which pulls tokio-process via the daemon adapter).
use hipfire_sysinfo::{collect_host_profile, detect_arch, HostProfileOverrides};
mod evidence;
use evidence::*;
mod performance;
use performance::*;
mod quality;
use quality::*;
mod rocprof;
use rocprof::*;
mod result;
use result::*;
mod run;
pub use run::*;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum EvalTier {
    Fast,
    Medium,
    Long,
    Extensive,
}

impl EvalTier {
    pub fn parse(raw: &str) -> Result<Self, String> {
        match raw {
            "fast" => Ok(Self::Fast),
            "medium" | "targeted" => Ok(Self::Medium),
            "long" => Ok(Self::Long),
            "extensive" => Ok(Self::Extensive),
            other => Err(format!("unknown tier: {other}")),
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Fast => "fast",
            Self::Medium => "medium",
            Self::Long => "long",
            Self::Extensive => "extensive",
        }
    }

    pub fn budget(self) -> TierBudget {
        match self {
            Self::Fast => TierBudget {
                tier: self,
                target_max_seconds: 60,
                ci_suitable: true,
                description: "small-model smoke and admission canaries".to_string(),
            },
            Self::Medium => TierBudget {
                tier: self,
                target_max_seconds: 300,
                ci_suitable: false,
                description: "bounded model-eval subset under five minutes".to_string(),
            },
            Self::Long => TierBudget {
                tier: self,
                target_max_seconds: 1200,
                ci_suitable: false,
                description: "broader quality and long-context subset under twenty minutes"
                    .to_string(),
            },
            Self::Extensive => TierBudget {
                tier: self,
                target_max_seconds: 0,
                ci_suitable: false,
                description: "full native/evidence suite; no fixed wall-clock target".to_string(),
            },
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TierBudget {
    pub tier: EvalTier,
    pub target_max_seconds: u64,
    pub ci_suitable: bool,
    pub description: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BatteryId {
    Smoke,
    Coherence,
    Quality,
    Retrieval,
    Speed,
    Dflash,
    Pflash,
    Agentic,
    Runtime,
    PromptShape,
    Structured,
    Barrage,
    Longctx,
    Vision,
    Cask,
    Profile,
    Calibrate,
    Perplexity,
    TinyQuant,
}

impl BatteryId {
    pub fn parse(raw: &str) -> Result<Self, String> {
        match raw {
            "smoke" => Ok(Self::Smoke),
            "coherence" | "runtime_coherence" | "runtime-coherence" => Ok(Self::Coherence),
            "quality" => Ok(Self::Quality),
            "retrieval" => Ok(Self::Retrieval),
            "speed" => Ok(Self::Speed),
            "dflash" => Ok(Self::Dflash),
            "pflash" => Ok(Self::Pflash),
            "agentic" | "tool_call" | "tool-call" => Ok(Self::Agentic),
            "runtime" | "server-runtime" | "server_runtime" => Ok(Self::Runtime),
            "prompt_shape" | "prompt-shape" => Ok(Self::PromptShape),
            "structured" => Ok(Self::Structured),
            "barrage" => Ok(Self::Barrage),
            "longctx" | "long_ctx" | "long-context" => Ok(Self::Longctx),
            "vision" => Ok(Self::Vision),
            "cask" => Ok(Self::Cask),
            "profile" => Ok(Self::Profile),
            "calibrate" | "calibration" => Ok(Self::Calibrate),
            "perplexity" | "ppl" => Ok(Self::Perplexity),
            "tinyquant" | "tiny_quant" | "tiny-quant" => Ok(Self::TinyQuant),
            other => Err(format!("unknown battery: {other}")),
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Smoke => "smoke",
            Self::Coherence => "coherence",
            Self::Quality => "quality",
            Self::Retrieval => "retrieval",
            Self::Speed => "speed",
            Self::Dflash => "dflash",
            Self::Pflash => "pflash",
            Self::Agentic => "agentic",
            Self::Runtime => "runtime",
            Self::PromptShape => "prompt_shape",
            Self::Structured => "structured",
            Self::Barrage => "barrage",
            Self::Longctx => "longctx",
            Self::Vision => "vision",
            Self::Cask => "cask",
            Self::Profile => "profile",
            Self::Calibrate => "calibrate",
            Self::Perplexity => "perplexity",
            Self::TinyQuant => "tiny_quant",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SuiteId {
    Gpqa,
    LmEvalMicro,
    #[serde(rename = "humaneval", alias = "human_eval", alias = "human-eval")]
    HumanEval,
    DeepSwe,
    SweBench,
    Ruler,
    #[serde(rename = "nolima", alias = "no_lima", alias = "no-lima")]
    NoLiMa,
    NeedleChain,
    Niah,
    SequentialNiah,
}

impl SuiteId {
    pub fn parse(raw: &str) -> Result<Self, String> {
        match raw {
            "gpqa" => Ok(Self::Gpqa),
            "lm_eval_micro" | "lm-eval-micro" => Ok(Self::LmEvalMicro),
            "humaneval" | "human_eval" | "human-eval" => Ok(Self::HumanEval),
            "deep_swe" | "deep-swe" => Ok(Self::DeepSwe),
            "swe_bench" | "swe-bench" => Ok(Self::SweBench),
            "ruler" => Ok(Self::Ruler),
            "nolima" | "no_lima" | "no-lima" => Ok(Self::NoLiMa),
            "needle_chain" | "needle-chain" => Ok(Self::NeedleChain),
            "niah" => Ok(Self::Niah),
            "sequential_niah" | "sequential-niah" => Ok(Self::SequentialNiah),
            other => Err(format!("unknown suite: {other}")),
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Gpqa => "gpqa",
            Self::LmEvalMicro => "lm_eval_micro",
            Self::HumanEval => "humaneval",
            Self::DeepSwe => "deep_swe",
            Self::SweBench => "swe_bench",
            Self::Ruler => "ruler",
            Self::NoLiMa => "nolima",
            Self::NeedleChain => "needle_chain",
            Self::Niah => "niah",
            Self::SequentialNiah => "sequential_niah",
        }
    }

    fn hf_repo_id(self) -> Option<&'static str> {
        match self {
            Self::Gpqa => Some("idavidrein/gpqa"),
            Self::NoLiMa => Some("amodaresi/NoLiMa"),
            Self::NeedleChain => Some("hyeonsss/needlechain"),
            _ => None,
        }
    }

    fn hf_revision(self) -> Option<&'static str> {
        match self {
            Self::Gpqa | Self::NoLiMa | Self::NeedleChain => Some("main"),
            _ => None,
        }
    }

    fn license(self) -> Option<&'static str> {
        match self {
            Self::NoLiMa => Some("Adobe Research License, non-commercial"),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum DflashMode {
    Off,
    Auto,
    On,
}

impl DflashMode {
    pub fn parse(raw: &str) -> Result<Self, String> {
        match raw {
            "off" => Ok(Self::Off),
            "auto" => Ok(Self::Auto),
            "on" => Ok(Self::On),
            other => Err(format!("unknown dflash mode: {other}")),
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Off => "off",
            Self::Auto => "auto",
            Self::On => "on",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ProfileMode {
    Off,
    Passive,
}

impl ProfileMode {
    pub fn parse(raw: &str) -> Result<Self, String> {
        match raw {
            "off" => Ok(Self::Off),
            "passive" => Ok(Self::Passive),
            other => Err(format!("unknown profile mode: {other}")),
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Off => "off",
            Self::Passive => "passive",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum EvalExecutorMode {
    Auto,
    None,
    Examples,
    Daemon,
    Direct,
    Mock,
}

impl EvalExecutorMode {
    pub fn parse(raw: &str) -> Result<Self, String> {
        match raw {
            "auto" => Ok(Self::Auto),
            "none" => Ok(Self::None),
            "examples" | "example" | "subprocess" => Ok(Self::Examples),
            "daemon" | "jsonl" | "adapter" => Ok(Self::Daemon),
            "direct" | "runtime" | "session" => Ok(Self::Direct),
            "mock" => Ok(Self::Mock),
            other => Err(format!("unknown executor mode: {other}")),
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::None => "none",
            Self::Examples => "examples",
            Self::Daemon => "daemon",
            Self::Direct => "direct",
            Self::Mock => "mock",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EvalCacheMode {
    Use,
    Force,
    Regenerate,
    Off,
}

impl EvalCacheMode {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Use => "use",
            Self::Force => "force",
            Self::Regenerate => "regenerate",
            Self::Off => "off",
        }
    }

    fn reads(self) -> bool {
        matches!(self, Self::Use)
    }

    fn writes(self) -> bool {
        !matches!(self, Self::Off)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvalConfig {
    pub model: String,
    pub draft: Option<String>,
    pub baseline: Option<String>,
    pub reference: Option<String>,
    pub tier: EvalTier,
    pub batteries: Vec<BatteryId>,
    pub suites: Vec<SuiteId>,
    pub out_dir: PathBuf,
    pub kv_mode: Option<String>,
    pub max_tokens: usize,
    pub dflash: DflashMode,
    pub profile: ProfileMode,
    pub quality_max_chunks: Option<usize>,
    pub kldref: Option<PathBuf>,
    pub quality_json: Option<PathBuf>,
    pub performance_json: Option<PathBuf>,
    pub evidence_json: Vec<PathBuf>,
    pub evidence_dirs: Vec<PathBuf>,
    pub candidate_variant: Option<String>,
    pub baseline_variant: Option<String>,
    pub reference_variant: Option<String>,
    pub performance_candidate_variant: Option<String>,
    pub performance_baseline_variant: Option<String>,
    pub performance_reference_variant: Option<String>,
    pub executor: EvalExecutorMode,
    pub fetch_datasets: bool,
    pub offline: bool,
    pub dataset_cache: PathBuf,
    pub result_cache: PathBuf,
    pub cache_mode: EvalCacheMode,
    pub runs: usize,
    pub warmup_runs: usize,
    pub benchmark: bool,
    pub host_memory_class: Option<String>,
    pub host_memory_width_bits: Option<u32>,
    pub host_memory_bandwidth_gbps: Option<f64>,
    pub fail_on_admission: bool,
    /// `--models <glob|csv>`: sweep many SKUs (run_from_env expands + loops).
    #[serde(default)]
    pub models_spec: Option<String>,
    /// `--dry-run`: plan only — resolve models/batteries/cache/artifacts and
    /// report, without running tests or fetching/generating anything.
    #[serde(default)]
    pub dry_run: bool,
    /// `--status`: print cache/dataset/hardware status and exit.
    #[serde(default)]
    pub status: bool,
    /// `--fetch`: ensure datasets/corpora are present, then exit.
    #[serde(default)]
    pub fetch: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatasetManifestEntry {
    pub suite: SuiteId,
    pub source: String,
    pub repo_id: Option<String>,
    pub revision: Option<String>,
    pub files: Vec<String>,
    pub digest: Option<String>,
    pub license: Option<String>,
    pub cache_path: String,
    pub selected_item_ids: Vec<String>,
    pub status: EvalStatus,
    pub reason: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvalManifest {
    pub schema: u32,
    pub runner: String,
    pub runner_version: String,
    pub hipfire_version: String,
    pub created_utc: String,
    pub tier_budget: TierBudget,
    pub repo_root: Option<String>,
    pub git_commit: Option<String>,
    pub commit_sha: Option<String>,
    pub git_branch: Option<String>,
    pub git_describe: Option<String>,
    pub git_dirty: Option<bool>,
    pub binary_hash: Option<String>,
    pub arch: Option<String>,
    pub rocm: Option<String>,
    pub host_profile: HostProfile,
    pub model_hash: Option<String>,
    pub draft_hash: Option<String>,
    pub baseline_hash: Option<String>,
    pub reference_hash: Option<String>,
    pub config: EvalConfig,
    pub models: Vec<ModelManifestEntry>,
    pub datasets: Vec<DatasetManifestEntry>,
    pub artifacts: BTreeMap<String, Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvalResult {
    pub schema: u32,
    pub battery: BatteryId,
    pub suite: Option<SuiteId>,
    pub case_id: String,
    pub dataset_item_id: Option<String>,
    pub dataset_source: Option<String>,
    pub dataset_repo_id: Option<String>,
    pub dataset_revision: Option<String>,
    pub dataset_digest: Option<String>,
    pub dataset_license: Option<String>,
    pub dataset_cache_path: Option<String>,
    pub status: EvalStatus,
    pub reason: Option<String>,
    pub metrics: BTreeMap<String, Value>,
    pub prompt_hash: Option<String>,
    pub prompt_path: Option<String>,
    pub model: String,
    pub draft: Option<String>,
    pub baseline: Option<String>,
    pub reference: Option<String>,
    pub model_hash: Option<String>,
    pub draft_hash: Option<String>,
    pub baseline_hash: Option<String>,
    pub reference_hash: Option<String>,
    pub hipfire_version: String,
    pub git_commit: Option<String>,
    pub commit_sha: Option<String>,
    pub git_branch: Option<String>,
    pub git_describe: Option<String>,
    pub git_dirty: Option<bool>,
    pub binary_hash: Option<String>,
    pub arch: Option<String>,
    pub rocm: Option<String>,
    pub host_profile_hash: String,
    pub hardware_bucket: String,
    pub kv_mode: Option<String>,
    pub started_utc: String,
    pub elapsed_ms: u128,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComparisonArtifact {
    pub schema: u32,
    pub provenance: RunProvenance,
    pub status: EvalStatus,
    pub reason: Option<String>,
    pub baseline: Option<String>,
    pub reference: Option<String>,
    pub cases: Vec<ComparisonCase>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComparisonCase {
    pub key: String,
    pub battery: BatteryId,
    pub suite: Option<SuiteId>,
    pub case_id: String,
    pub dataset_item_id: Option<String>,
    pub baseline: Option<MetricComparison>,
    pub reference: Option<MetricComparison>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricComparison {
    pub model: String,
    pub metrics: BTreeMap<String, MetricDelta>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricDelta {
    pub candidate: f64,
    pub comparator: f64,
    pub delta: f64,
    pub relative_delta: Option<f64>,
    pub direction: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdmissionArtifact {
    pub schema: u32,
    pub provenance: RunProvenance,
    pub status: EvalStatus,
    pub verdict: String,
    pub reason: Option<String>,
    pub required_evidence: Vec<AdmissionEvidence>,
    pub observed_evidence: Vec<AdmissionEvidence>,
    pub findings: Vec<AdmissionFinding>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdmissionEvidence {
    pub kind: String,
    pub status: EvalStatus,
    pub rows: usize,
    pub reason: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdmissionFinding {
    pub severity: String,
    pub battery: BatteryId,
    pub suite: Option<SuiteId>,
    pub case_id: String,
    pub dataset_item_id: Option<String>,
    pub comparator: String,
    pub metric: String,
    pub direction: String,
    pub delta: f64,
    pub relative_delta: Option<f64>,
}

fn eval_status_str(status: EvalStatus) -> &'static str {
    status.as_str()
}

pub fn run_eval(config: EvalConfig) -> Result<(), String> {
    fs::create_dir_all(&config.out_dir)
        .map_err(|e| format!("create {}: {e}", config.out_dir.display()))?;
    let artifacts_dir = config.out_dir.join("artifacts");
    fs::create_dir_all(&artifacts_dir).map_err(|e| format!("create artifacts dir: {e}"))?;
    let context = EvalContext::new_with_overrides(HostProfileOverrides {
        memory_class: config.host_memory_class.clone(),
        memory_width_bits: config.host_memory_width_bits,
        memory_bandwidth_gbps: config.host_memory_bandwidth_gbps,
    });
    let models = model_manifest_entries(&config);
    let datasets = resolve_datasets(&config)?;

    let mut results = run_eval_batteries(&config, &context, &datasets)?;
    results.extend(run_passive_profile_collectors(&config, &context));
    let results_path = config.out_dir.join("results.jsonl");
    let mut f = OpenOptions::new()
        .create(true)
        .truncate(true)
        .write(true)
        .open(&results_path)
        .map_err(|e| format!("open {}: {e}", results_path.display()))?;
    for row in &results {
        serde_json::to_writer(&mut f, row).map_err(|e| format!("serialize result row: {e}"))?;
        f.write_all(b"\n")
            .map_err(|e| format!("write {}: {e}", results_path.display()))?;
    }

    let comparison = build_comparison_artifact(&config, &results, &context);
    let admission = build_admission_artifact(&config, &results, &comparison, &context);
    let artifacts = write_evidence_artifacts(
        &artifacts_dir,
        &config,
        &datasets,
        &results,
        &comparison,
        &admission,
        &context,
    )?;
    let manifest = EvalManifest {
        schema: 2,
        runner: "hipfire-eval".to_string(),
        runner_version: env!("CARGO_PKG_VERSION").to_string(),
        hipfire_version: env!("CARGO_PKG_VERSION").to_string(),
        created_utc: utc_now(),
        tier_budget: config.tier.budget(),
        repo_root: repo_root().map(|p| p.display().to_string()),
        git_commit: context.commit_sha.clone(),
        commit_sha: context.commit_sha.clone(),
        git_branch: context.git_branch.clone(),
        git_describe: context.git_describe.clone(),
        git_dirty: context.git_dirty,
        binary_hash: context.binary_hash.clone(),
        arch: context.arch.clone(),
        rocm: context.rocm.clone(),
        host_profile: context.host_profile.clone(),
        model_hash: model_hash(&config.model),
        draft_hash: config.draft.as_deref().and_then(model_hash),
        baseline_hash: config.baseline.as_deref().and_then(model_hash),
        reference_hash: config.reference.as_deref().and_then(model_hash),
        config: config.clone(),
        models,
        datasets: datasets.clone(),
        artifacts: artifacts.clone(),
    };
    write_json_pretty(&config.out_dir.join("manifest.json"), &manifest)?;
    write_summary(
        &config.out_dir.join("summary.md"),
        &config,
        &datasets,
        &comparison,
        &admission,
        &artifacts,
        &results,
        &context,
    )?;
    println!("{}", config.out_dir.display());
    print!("{}", render_eval_stdout_findings(&admission, &results));
    if config.fail_on_admission && admission.verdict != "promote" {
        return Err(format!(
            "admission verdict {}: {}; artifacts written to {}",
            admission.verdict,
            admission
                .reason
                .as_deref()
                .unwrap_or("non-promote admission verdict"),
            config.out_dir.display()
        ));
    }
    Ok(())
}

const STDOUT_RESULT_ROW_LIMIT: usize = 12;
const STDOUT_RESULT_METRIC_LIMIT: usize = 8;
const STDOUT_ROW_FINDING_LIMIT: usize = 12;
const STDOUT_METRIC_PRIORITY: &[&str] = &[
    "tok_s_median",
    "tok_s",
    "decode_tok_s_median",
    "decode_tok_s",
    "prefill_tok_s_median",
    "prefill_tok_s",
    "ttft_ms_median",
    "ttft_ms",
    "decode_ms_median",
    "decode_ms",
    "prefill_ms_median",
    "prefill_ms",
    "accuracy",
    "mean_kld",
    "ppl",
    "perplexity",
    "sample_count",
    "total_sample_count",
    "failed_sample_count",
    "skipped_sample_count",
    "cache_hit",
];

fn render_eval_stdout_findings(admission: &AdmissionArtifact, rows: &[EvalResult]) -> String {
    let result_rows = stdout_result_rows(rows);
    let row_findings = rows
        .iter()
        .filter(|row| row.status != EvalStatus::Pass)
        .count();
    let total_findings = admission.findings.len() + row_findings;
    let mut body = String::new();
    body.push_str(&format!("admission: {}", admission.verdict));
    if let Some(reason) = admission.reason.as_deref() {
        body.push_str(&format!(" ({reason})"));
    }
    body.push('\n');
    body.push_str(&format!("results: {}\n", result_rows.len()));
    for row in result_rows.iter().take(STDOUT_RESULT_ROW_LIMIT) {
        body.push_str(&render_stdout_result_row(row));
    }
    if result_rows.len() > STDOUT_RESULT_ROW_LIMIT {
        body.push_str(&format!(
            "  ... {} more result row(s); see summary.md and results.jsonl\n",
            result_rows.len() - STDOUT_RESULT_ROW_LIMIT
        ));
    }
    body.push_str(&format!("findings: {total_findings}\n"));
    body.push_str(&format!(
        "admission findings: {}\n",
        admission.findings.len()
    ));
    for finding in &admission.findings {
        let suite = finding.suite.map(|suite| suite.as_str()).unwrap_or("-");
        let item = finding.dataset_item_id.as_deref().unwrap_or("-");
        let relative = finding
            .relative_delta
            .map(|delta| format!("{:+.2}%", delta * 100.0))
            .unwrap_or_else(|| "-".to_string());
        body.push_str(&format!(
            "  [{}] {}/{} suite={} item={} metric={} comparator={} direction={} delta={:+.6} relative={}\n",
            finding.severity,
            finding.battery.as_str(),
            finding.case_id,
            suite,
            item,
            finding.metric,
            finding.comparator,
            finding.direction,
            finding.delta,
            relative
        ));
    }
    body.push_str(&format!("row findings: {row_findings}\n"));
    let mut rendered_row_findings = 0usize;
    for row in rows.iter().filter(|row| row.status != EvalStatus::Pass) {
        if rendered_row_findings >= STDOUT_ROW_FINDING_LIMIT {
            continue;
        }
        let suite = row.suite.map(|suite| suite.as_str()).unwrap_or("-");
        let item = row.dataset_item_id.as_deref().unwrap_or("-");
        let reason = row
            .reason
            .as_deref()
            .map(stdout_field)
            .unwrap_or_else(|| "-".to_string());
        body.push_str(&format!(
            "  [{}] {}/{} suite={} item={} reason={}\n",
            eval_status_str(row.status),
            row.battery.as_str(),
            row.case_id,
            suite,
            item,
            reason
        ));
        rendered_row_findings += 1;
    }
    if row_findings > STDOUT_ROW_FINDING_LIMIT {
        body.push_str(&format!(
            "  ... {} more row finding(s); see summary.md and results.jsonl\n",
            row_findings - STDOUT_ROW_FINDING_LIMIT
        ));
    }
    body
}

fn stdout_result_rows(rows: &[EvalResult]) -> Vec<&EvalResult> {
    rows.iter()
        .filter(|row| row.status == EvalStatus::Pass)
        .filter(|row| !stdout_has_preferred_aggregate(row, rows))
        .filter(|row| !stdout_metric_summary(row).is_empty())
        .collect()
}

fn stdout_has_preferred_aggregate(row: &EvalResult, rows: &[EvalResult]) -> bool {
    if row.metrics.get("benchmark_sample").and_then(Value::as_bool) != Some(true) {
        return false;
    }
    rows.iter().any(|aggregate| {
        aggregate.status == EvalStatus::Pass
            && aggregate
                .metrics
                .get("benchmark_aggregate")
                .and_then(Value::as_bool)
                == Some(true)
            && aggregate
                .metrics
                .get("aggregate_source_case_id")
                .and_then(Value::as_str)
                == Some(row.case_id.as_str())
            && aggregate.battery == row.battery
            && aggregate.suite == row.suite
            && aggregate.dataset_item_id == row.dataset_item_id
            && aggregate.model == row.model
            && aggregate.prompt_hash == row.prompt_hash
            && aggregate.kv_mode == row.kv_mode
    })
}

fn render_stdout_result_row(row: &EvalResult) -> String {
    let suite = row.suite.map(|suite| suite.as_str()).unwrap_or("-");
    let item = row.dataset_item_id.as_deref().unwrap_or("-");
    let model = stdout_model_label(&row.model);
    let metrics = stdout_metric_summary(row).join(" ");
    format!(
        "  [pass] {}/{} suite={} item={} model={} {}\n",
        row.battery.as_str(),
        row.case_id,
        suite,
        item,
        model,
        metrics
    )
}

fn stdout_metric_summary(row: &EvalResult) -> Vec<String> {
    let mut rendered = Vec::new();
    if row.elapsed_ms > 0 {
        rendered.push(format!("elapsed_ms={}", row.elapsed_ms));
    }
    for key in STDOUT_METRIC_PRIORITY {
        if rendered.len() >= STDOUT_RESULT_METRIC_LIMIT {
            return rendered;
        }
        if let Some(value) = row.metrics.get(*key).and_then(stdout_metric_value) {
            rendered.push(format!("{key}={value}"));
        }
    }
    for (key, value) in &row.metrics {
        if rendered.len() >= STDOUT_RESULT_METRIC_LIMIT {
            break;
        }
        if STDOUT_METRIC_PRIORITY.contains(&key.as_str()) || stdout_metric_excluded(key) {
            continue;
        }
        if let Some(value) = stdout_metric_value(value) {
            rendered.push(format!("{key}={value}"));
        }
    }
    rendered
}

fn stdout_metric_excluded(key: &str) -> bool {
    matches!(
        key,
        "aggregate_source_case_id"
            | "benchmark_aggregate"
            | "benchmark_sample"
            | "run_index"
            | "run_count"
            | "warmup_runs"
            | "cache_key"
            | "cache_path"
            | "dataset_source"
            | "dataset_repo_id"
            | "dataset_revision"
            | "dataset_digest"
            | "dataset_license"
            | "dataset_cache_path"
    ) || key.ends_with("_hash")
        || key.ends_with("_path")
}

fn stdout_metric_value(value: &Value) -> Option<String> {
    if let Some(value) = value.as_bool() {
        return Some(value.to_string());
    }
    if let Some(value) = value.as_i64() {
        return Some(value.to_string());
    }
    if let Some(value) = value.as_u64() {
        return Some(value.to_string());
    }
    let value = value.as_f64()?;
    if !value.is_finite() {
        return None;
    }
    Some(format_stdout_float(value))
}

fn format_stdout_float(value: f64) -> String {
    let formatted = if value.abs() >= 100.0 {
        format!("{value:.2}")
    } else if value.abs() >= 10.0 {
        format!("{value:.3}")
    } else {
        format!("{value:.4}")
    };
    formatted
        .trim_end_matches('0')
        .trim_end_matches('.')
        .to_string()
}

fn stdout_model_label(model: &str) -> String {
    Path::new(model)
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or(model)
        .to_string()
}

fn stdout_field(value: &str) -> String {
    value.replace('\n', " ").replace('\r', " ")
}

fn run_passive_profile_collectors(config: &EvalConfig, ctx: &EvalContext) -> Vec<EvalResult> {
    if config.profile != ProfileMode::Passive {
        return Vec::new();
    }
    vec![
        run_rocprof_speed_anchor(config, ctx),
        run_host_capability_profile_anchor(config, ctx),
    ]
}

fn run_host_capability_profile_anchor(config: &EvalConfig, ctx: &EvalContext) -> EvalResult {
    let mut metrics = BTreeMap::from([
        ("profiling_requested".to_string(), json!(true)),
        (
            "profiling_collector".to_string(),
            json!("hipfire-host-profile"),
        ),
    ]);
    let Some(bin) = resolve_host_profile_bin() else {
        return skip_row_with_metrics(
            BatteryId::Profile,
            None,
            "host_capability_profile",
            None,
            "hipfire-host-profile binary not found; build with `cargo build --release -p hipfire-runtime --bin hipfire-host-profile`",
            config,
            ctx,
            None,
            metrics,
        );
    };
    let out = config.out_dir.join("artifacts").join("host_profile.json");
    let models_dir = home_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join(".hipfire")
        .join("models");
    let started = SystemTime::now();
    let output = match Command::new(&bin)
        .args([
            "--out",
            &out.display().to_string(),
            "--models-dir",
            &models_dir.display().to_string(),
            "--size-mib",
            "32",
            "--storage-size-mib",
            "32",
            "--runs",
            "1",
            "--gpu-max-size-mib",
            "32",
        ])
        .output()
    {
        Ok(output) => output,
        Err(err) => {
            metrics.insert("binary".to_string(), json!(bin.display().to_string()));
            return skip_row_with_metrics(
                BatteryId::Profile,
                None,
                "host_capability_profile",
                None,
                &format!("run hipfire-host-profile: {err}"),
                config,
                ctx,
                None,
                metrics,
            );
        }
    };
    metrics.insert("binary".to_string(), json!(bin.display().to_string()));
    metrics.insert("report_path".to_string(), json!(out.display().to_string()));
    metrics.insert(
        "stdout".to_string(),
        json!(String::from_utf8_lossy(&output.stdout).trim().to_string()),
    );
    if !output.status.success() {
        metrics.insert(
            "stderr".to_string(),
            json!(String::from_utf8_lossy(&output.stderr).trim().to_string()),
        );
        return row(
            BatteryId::Profile,
            None,
            "host_capability_profile",
            None,
            EvalStatus::Skip,
            Some("hipfire-host-profile returned non-zero".to_string()),
            metrics,
            config,
            ctx,
            None,
            elapsed_since_ms(started),
        );
    }
    let mut report_status = None;
    if let Ok(body) = fs::read_to_string(&out) {
        if let Ok(value) = serde_json::from_str::<Value>(&body) {
            if let Some(records) = value.get("records").and_then(Value::as_array) {
                metrics.insert("host_capability_records".to_string(), json!(records.len()));
            }
            if let Some(status) = value.get("status").cloned() {
                report_status = status.as_str().map(str::to_string);
                metrics.insert("host_capability_status".to_string(), status);
            }
            if let Some(build_profile) = value.get("build_profile").cloned() {
                metrics.insert("host_capability_build_profile".to_string(), build_profile);
            }
        }
    }
    if report_status.as_deref() == Some("invalid") {
        return row(
            BatteryId::Profile,
            None,
            "host_capability_profile",
            None,
            EvalStatus::Skip,
            Some("hipfire-host-profile report is invalid evidence".to_string()),
            metrics,
            config,
            ctx,
            None,
            elapsed_since_ms(started),
        );
    }
    row(
        BatteryId::Profile,
        None,
        "host_capability_profile",
        None,
        EvalStatus::Pass,
        None,
        metrics,
        config,
        ctx,
        None,
        elapsed_since_ms(started),
    )
}

#[derive(Clone)]
struct EvalContext {
    commit_sha: Option<String>,
    git_branch: Option<String>,
    git_describe: Option<String>,
    git_dirty: Option<bool>,
    binary_hash: Option<String>,
    arch: Option<String>,
    rocm: Option<String>,
    host_profile: HostProfile,
}

impl EvalContext {
    fn new() -> Self {
        Self::new_with_overrides(HostProfileOverrides::default())
    }

    fn new_with_overrides(overrides: HostProfileOverrides) -> Self {
        let arch = detect_arch();
        Self {
            commit_sha: command_stdout("git", &["rev-parse", "HEAD"]),
            git_branch: command_stdout("git", &["rev-parse", "--abbrev-ref", "HEAD"]),
            git_describe: command_stdout("git", &["describe", "--always", "--dirty", "--tags"]),
            git_dirty: git_dirty(),
            binary_hash: std::env::current_exe().ok().and_then(|p| file_hash(&p)),
            host_profile: collect_host_profile(arch.clone(), overrides),
            arch,
            rocm: rocm_version(),
        }
    }
}

fn resolve_dflash_spec_demo_bin() -> Option<PathBuf> {
    if let Ok(path) = std::env::var("HIPFIRE_DFLASH_SPEC_DEMO_BIN") {
        let p = PathBuf::from(path);
        if p.exists() {
            return Some(p);
        }
    }
    let exe = std::env::consts::EXE_SUFFIX;
    let repo = repo_root()?;
    newest_existing_path([
        repo.join(format!("target/release/examples/dflash_spec_demo{exe}")),
        repo.join(format!("target/debug/examples/dflash_spec_demo{exe}")),
    ])
}

fn resolve_bench_qwen35_speed_bin() -> Option<PathBuf> {
    if let Ok(path) = std::env::var("HIPFIRE_BENCH_QWEN35_SPEED_BIN") {
        let p = PathBuf::from(path);
        if p.exists() {
            return Some(p);
        }
    }
    let exe = std::env::consts::EXE_SUFFIX;
    let repo = repo_root()?;
    [
        repo.join(format!("target/release/examples/bench_qwen35_speed{exe}")),
        repo.join(format!("target/debug/examples/bench_qwen35_speed{exe}")),
    ]
    .into_iter()
    .find(|p| p.exists())
}

fn resolve_pflash_niah_bench_bin() -> Option<PathBuf> {
    if let Ok(path) = std::env::var("HIPFIRE_PFLASH_NIAH_BENCH_BIN") {
        let p = PathBuf::from(path);
        if p.exists() {
            return Some(p);
        }
    }
    let exe = std::env::consts::EXE_SUFFIX;
    let repo = repo_root()?;
    newest_existing_path([
        repo.join(format!("target/release/examples/pflash_niah_bench{exe}")),
        repo.join(format!("target/debug/examples/pflash_niah_bench{exe}")),
    ])
}

fn resolve_run_example_bin() -> Option<PathBuf> {
    if let Ok(path) = std::env::var("HIPFIRE_RUN_EXAMPLE_BIN") {
        let p = PathBuf::from(path);
        if p.exists() {
            return Some(p);
        }
    }
    let exe = std::env::consts::EXE_SUFFIX;
    let repo = repo_root()?;
    newest_existing_path([
        repo.join(format!("target/release/examples/run{exe}")),
        repo.join(format!("target/debug/examples/run{exe}")),
    ])
}

fn resolve_collect_artifacts_bin() -> Option<PathBuf> {
    if let Ok(path) = std::env::var("HIPFIRE_COLLECT_ARTIFACTS_BIN") {
        let p = PathBuf::from(path);
        if p.exists() {
            return Some(p);
        }
    }
    let exe = std::env::consts::EXE_SUFFIX;
    let repo = repo_root()?;
    newest_existing_path([
        repo.join(format!("target/release/examples/collect_artifacts{exe}")),
        repo.join(format!("target/debug/examples/collect_artifacts{exe}")),
    ])
}

fn resolve_perplexity_bin() -> Option<PathBuf> {
    if let Ok(path) = std::env::var("HIPFIRE_PERPLEXITY_BIN") {
        let p = PathBuf::from(path);
        if p.exists() {
            return Some(p);
        }
    }
    let exe = std::env::consts::EXE_SUFFIX;
    let repo = repo_root()?;
    newest_existing_path([
        repo.join(format!("target/release/examples/perplexity{exe}")),
        repo.join(format!("target/debug/examples/perplexity{exe}")),
    ])
}

fn resolve_host_profile_bin() -> Option<PathBuf> {
    if let Ok(path) = std::env::var("HIPFIRE_HOST_PROFILE_BIN") {
        let p = PathBuf::from(path);
        if p.exists() {
            return Some(p);
        }
    }
    let exe = std::env::consts::EXE_SUFFIX;
    let repo = repo_root()?;
    newest_existing_path([
        repo.join(format!("target/release/hipfire-host-profile{exe}")),
        repo.join(format!("target/debug/hipfire-host-profile{exe}")),
        home_dir()
            .unwrap_or_else(|| PathBuf::from("."))
            .join(".hipfire")
            .join("bin")
            .join(format!("hipfire-host-profile{exe}")),
    ])
}

fn newest_existing_path<const N: usize>(paths: [PathBuf; N]) -> Option<PathBuf> {
    paths.into_iter().filter(|p| p.exists()).max_by_key(|p| {
        p.metadata()
            .and_then(|m| m.modified())
            .ok()
            .and_then(|t| t.duration_since(UNIX_EPOCH).ok())
            .map(|d| d.as_secs())
            .unwrap_or(0)
    })
}

fn parse_bench_metrics(stderr: &str) -> BTreeMap<String, Value> {
    let mut metrics = BTreeMap::new();
    let mut in_block = false;
    for raw_line in stderr.lines() {
        let line = raw_line.trim();
        if line == "=== BENCH METRICS ===" {
            in_block = true;
            continue;
        }
        if in_block && line == "=====================" {
            break;
        }
        if !in_block {
            continue;
        }
        let Some((key, raw_value)) = line.split_once(':') else {
            continue;
        };
        let key = key.trim();
        let raw_value = raw_value.trim();
        if let Ok(value) = raw_value.parse::<i64>() {
            metrics.insert(key.to_string(), json!(value));
        } else if let Ok(value) = raw_value.parse::<f64>() {
            metrics.insert(key.to_string(), json!(value));
        } else {
            metrics.insert(key.to_string(), json!(raw_value));
        }
    }
    metrics
}

fn parse_summary_kv_metrics(output: &str) -> BTreeMap<String, Value> {
    let mut metrics = BTreeMap::new();
    for raw_line in output.lines() {
        let line = raw_line.trim();
        if !line.starts_with("SUMMARY") && !line.starts_with("PREFILL_SUMMARY") {
            continue;
        }
        for part in line.split_whitespace().skip(1) {
            let Some((key, value)) = part.split_once('=') else {
                continue;
            };
            let value = value.trim_end_matches(',');
            if let Ok(parsed) = value.parse::<f64>() {
                metrics.insert(key.to_string(), json!(parsed));
            }
        }
    }
    metrics
}

fn elapsed_since_ms(started: SystemTime) -> u128 {
    started.elapsed().map(|d| d.as_millis()).unwrap_or(0)
}

fn barrage_rows(
    config: &EvalConfig,
    ctx: &EvalContext,
    datasets: &[DatasetManifestEntry],
) -> Vec<EvalResult> {
    if datasets.is_empty() {
        return vec![skip_row(
            BatteryId::Barrage,
            None,
            "no_suite_selected",
            None,
            "no barrage suite selected",
            config,
            ctx,
            None,
        )];
    }
    datasets
        .iter()
        .flat_map(|d| {
            if d.suite == SuiteId::Gpqa && d.status == EvalStatus::Pass {
                return match gpqa_materialized_items(Path::new(&d.cache_path), &d.selected_item_ids)
                {
                    Ok(items) => items
                        .into_iter()
                        .map(|item| {
                            let prompt = PromptRef::from_content(
                                format!("dataset:gpqa:{}", item.item_id),
                                item.prompt.as_bytes(),
                            );
                            let mut metrics = BTreeMap::new();
                            metrics
                                .insert("prompt_format".to_string(), json!("gpqa_zero_shot_v1"));
                            metrics.insert("answer_label".to_string(), json!(item.answer_label));
                            metrics.insert(
                                "answer_hash".to_string(),
                                json!(stable_hash_bytes(item.correct_answer.as_bytes())),
                            );
                            metrics.insert("choices_count".to_string(), json!(item.choices.len()));
                            metrics.insert("dataset_file".to_string(), json!(item.dataset_file));
                            add_dataset_provenance_metrics(&mut metrics, d);
                            skip_row_with_metrics(
                                BatteryId::Barrage,
                                Some(SuiteId::Gpqa),
                                "gpqa_zero_shot_native",
                                Some(item.item_id),
                                "native GPQA prompt materialized; model execution is not implemented yet",
                                config,
                                ctx,
                                Some(prompt),
                                metrics,
                            )
                        })
                        .collect::<Vec<_>>(),
                    Err(reason) => d
                        .selected_item_ids
                        .iter()
                        .cloned()
                        .map(|id| {
                            let mut metrics = BTreeMap::new();
                            add_dataset_provenance_metrics(&mut metrics, d);
                            skip_row_with_metrics(
                                BatteryId::Barrage,
                                Some(SuiteId::Gpqa),
                                "gpqa_materialize_failed",
                                Some(id),
                                &reason,
                                config,
                                ctx,
                                None,
                                metrics,
                            )
                        })
                        .collect::<Vec<_>>(),
                };
            }
            if d.suite == SuiteId::LmEvalMicro && d.status == EvalStatus::Pass {
                return match lm_eval_micro_materialized_items(&d.selected_item_ids) {
                    Ok(items) => items
                        .into_iter()
                        .map(|item| {
                            let prompt = PromptRef::from_content(
                                format!("dataset:lm_eval_micro:{}", item.item_id),
                                item.prompt.as_bytes(),
                            );
                            let mut metrics = BTreeMap::from([
                                (
                                    "prompt_format".to_string(),
                                    json!("lm_eval_micro_zero_shot_v1"),
                                ),
                                ("task".to_string(), json!(item.task)),
                                ("answer_label".to_string(), json!(item.answer_label)),
                                ("answer_hash".to_string(), json!(item.answer_hash)),
                                ("choices_count".to_string(), json!(item.choices_count)),
                                (
                                    "dataset_file".to_string(),
                                    json!("builtin:lm_eval_micro:v1"),
                                ),
                                ("scoring_mode".to_string(), json!("exact_letter")),
                            ]);
                            add_dataset_provenance_metrics(&mut metrics, d);
                            skip_row_with_metrics(
                                BatteryId::Barrage,
                                Some(SuiteId::LmEvalMicro),
                                "lm_eval_micro_zero_shot_native",
                                Some(item.item_id),
                                "native lm_eval_micro prompt materialized; model execution is not enabled",
                                config,
                                ctx,
                                Some(prompt),
                                metrics,
                            )
                        })
                        .collect::<Vec<_>>(),
                    Err(reason) => d
                        .selected_item_ids
                        .iter()
                        .cloned()
                        .map(|id| {
                            let mut metrics = BTreeMap::new();
                            add_dataset_provenance_metrics(&mut metrics, d);
                            skip_row_with_metrics(
                                BatteryId::Barrage,
                                Some(SuiteId::LmEvalMicro),
                                "lm_eval_micro_materialize_failed",
                                Some(id),
                                &reason,
                                config,
                                ctx,
                                None,
                                metrics,
                            )
                        })
                        .collect::<Vec<_>>(),
                };
            }
            if d.suite == SuiteId::HumanEval && d.status == EvalStatus::Pass {
                return match humaneval_materialized_items(
                    Path::new(&d.cache_path),
                    &d.selected_item_ids,
                ) {
                    Ok(items) => items
                        .into_iter()
                        .map(|item| {
                            let prompt = PromptRef::from_content(
                                format!("dataset:humaneval:{}", item.item_id),
                                item.prompt.as_bytes(),
                            );
                            let mut metrics = BTreeMap::new();
                            metrics.insert(
                                "prompt_format".to_string(),
                                json!("humaneval_completion_v1"),
                            );
                            metrics.insert("task_id".to_string(), json!(item.task_id));
                            metrics.insert("dataset_file".to_string(), json!(item.dataset_file));
                            metrics.insert("scoring_mode".to_string(), json!("execution_only"));
                            if let Some(hash) = item.canonical_solution_hash {
                                metrics.insert("canonical_solution_hash".to_string(), json!(hash));
                            }
                            if let Some(hash) = item.test_hash {
                                metrics.insert("test_hash".to_string(), json!(hash));
                            }
                            add_dataset_provenance_metrics(&mut metrics, d);
                            skip_row_with_metrics(
                                BatteryId::Barrage,
                                Some(SuiteId::HumanEval),
                                "humaneval_completion_native",
                                Some(item.item_id),
                                "native HumanEval prompt materialized; model execution is not enabled",
                                config,
                                ctx,
                                Some(prompt),
                                metrics,
                            )
                        })
                        .collect::<Vec<_>>(),
                    Err(reason) => d
                        .selected_item_ids
                        .iter()
                        .cloned()
                        .map(|id| {
                            let mut metrics = BTreeMap::new();
                            add_dataset_provenance_metrics(&mut metrics, d);
                            skip_row_with_metrics(
                                BatteryId::Barrage,
                                Some(SuiteId::HumanEval),
                                "humaneval_materialize_failed",
                                Some(id),
                                &reason,
                                config,
                                ctx,
                                None,
                                metrics,
                            )
                        })
                        .collect::<Vec<_>>(),
                };
            }
            if matches!(d.suite, SuiteId::DeepSwe | SuiteId::SweBench)
                && d.status == EvalStatus::Pass
            {
                return match builtin_barrage_materialized_items(d.suite, &d.selected_item_ids) {
                    Ok(items) => items
                        .into_iter()
                        .map(|item| {
                            let prompt = PromptRef::from_content(
                                format!("dataset:{}:{}", item.suite.as_str(), item.item_id),
                                item.prompt.as_bytes(),
                            );
                            let mut metrics = BTreeMap::from([
                                ("prompt_format".to_string(), json!(item.prompt_format)),
                                ("task".to_string(), json!(item.task)),
                                ("answer_label".to_string(), json!(item.answer_label)),
                                ("answer_hash".to_string(), json!(item.answer_hash)),
                                ("choices_count".to_string(), json!(item.choices_count)),
                                ("dataset_file".to_string(), json!(item.dataset_file)),
                                ("scoring_mode".to_string(), json!(item.scoring_mode)),
                            ]);
                            add_dataset_provenance_metrics(&mut metrics, d);
                            skip_row_with_metrics(
                                BatteryId::Barrage,
                                Some(item.suite),
                                "builtin_software_eval_native",
                                Some(item.item_id),
                                "native built-in software-eval prompt materialized; model execution is not enabled",
                                config,
                                ctx,
                                Some(prompt),
                                metrics,
                            )
                        })
                        .collect::<Vec<_>>(),
                    Err(reason) => d
                        .selected_item_ids
                        .iter()
                        .cloned()
                        .map(|id| {
                            let mut metrics = BTreeMap::new();
                            add_dataset_provenance_metrics(&mut metrics, d);
                            skip_row_with_metrics(
                                BatteryId::Barrage,
                                Some(d.suite),
                                "builtin_software_eval_materialize_failed",
                                Some(id),
                                &reason,
                                config,
                                ctx,
                                None,
                                metrics,
                            )
                        })
                        .collect::<Vec<_>>(),
                };
            }

            let reason = d
                .reason
                .clone()
                .unwrap_or_else(|| "native barrage runner is not implemented yet".to_string());
            selected_item_ids(d.suite)
                .into_iter()
                .map(move |id| {
                    let mut metrics = BTreeMap::new();
                    add_dataset_provenance_metrics(&mut metrics, d);
                    skip_row_with_metrics(
                        BatteryId::Barrage,
                        Some(d.suite),
                        "native_barrage_subset",
                        Some(id),
                        &reason,
                        config,
                        ctx,
                        None,
                        metrics,
                    )
                })
                .collect::<Vec<_>>()
        })
        .collect()
}

fn resolve_repo_path(path: &str) -> Option<PathBuf> {
    let direct = PathBuf::from(path);
    if direct.exists() {
        return Some(direct);
    }
    let resolved = repo_root()?.join(path);
    resolved.exists().then_some(resolved)
}

fn write_json_pretty<T: Serialize>(path: &Path, value: &T) -> Result<(), String> {
    let f = File::create(path).map_err(|e| format!("create {}: {e}", path.display()))?;
    serde_json::to_writer_pretty(f, value).map_err(|e| format!("write {}: {e}", path.display()))
}

fn write_summary(
    path: &Path,
    config: &EvalConfig,
    datasets: &[DatasetManifestEntry],
    comparison: &ComparisonArtifact,
    admission: &AdmissionArtifact,
    artifacts: &BTreeMap<String, Value>,
    rows: &[EvalResult],
    ctx: &EvalContext,
) -> Result<(), String> {
    let pass = rows.iter().filter(|r| r.status == EvalStatus::Pass).count();
    let fail = rows.iter().filter(|r| r.status == EvalStatus::Fail).count();
    let skip = rows.iter().filter(|r| r.status == EvalStatus::Skip).count();
    let mut body = String::new();
    body.push_str("# hipfire eval summary\n\n");
    body.push_str(&format!("- model: `{}`\n", config.model));
    if let Some(hash) = model_hash(&config.model) {
        body.push_str(&format!("- model hash: `{hash}`\n"));
    }
    if let Some(draft) = &config.draft {
        body.push_str(&format!("- draft: `{draft}`\n"));
        if let Some(hash) = model_hash(draft) {
            body.push_str(&format!("- draft hash: `{hash}`\n"));
        }
    }
    if let Some(baseline) = &config.baseline {
        body.push_str(&format!("- baseline: `{baseline}`\n"));
        if let Some(hash) = model_hash(baseline) {
            body.push_str(&format!("- baseline hash: `{hash}`\n"));
        }
    }
    if let Some(reference) = &config.reference {
        body.push_str(&format!("- reference: `{reference}`\n"));
        if let Some(hash) = model_hash(reference) {
            body.push_str(&format!("- reference hash: `{hash}`\n"));
        }
    }
    body.push_str(&format!("- tier: `{}`\n", config.tier.as_str()));
    let tier_budget = config.tier.budget();
    body.push_str(&format!(
        "- tier target: `{}` seconds ({})\n",
        tier_budget.target_max_seconds, tier_budget.description
    ));
    body.push_str(&format!("- CI suitable: `{}`\n", tier_budget.ci_suitable));
    body.push_str(&format!(
        "- hipfire version: `{}`\n",
        env!("CARGO_PKG_VERSION")
    ));
    body.push_str(&format!(
        "- runner: `hipfire-eval {}`\n",
        env!("CARGO_PKG_VERSION")
    ));
    if let Some(commit_sha) = &ctx.commit_sha {
        body.push_str(&format!("- git commit: `{commit_sha}`\n"));
    }
    if let Some(branch) = &ctx.git_branch {
        body.push_str(&format!("- git branch: `{branch}`\n"));
    }
    if let Some(describe) = &ctx.git_describe {
        body.push_str(&format!("- git describe: `{describe}`\n"));
    }
    if let Some(dirty) = ctx.git_dirty {
        body.push_str(&format!("- git dirty: `{dirty}`\n"));
    }
    if let Some(binary_hash) = &ctx.binary_hash {
        body.push_str(&format!("- binary hash: `{binary_hash}`\n"));
    }
    if let Some(arch) = &ctx.arch {
        body.push_str(&format!("- arch: `{arch}`\n"));
    }
    if let Some(rocm) = &ctx.rocm {
        body.push_str(&format!("- ROCm: `{rocm}`\n"));
    }
    body.push_str(&format!(
        "- hardware bucket: `{}`\n",
        ctx.host_profile.hardware_bucket
    ));
    body.push_str(&format!(
        "- host profile hash: `{}`\n",
        ctx.host_profile.host_profile_hash
    ));
    body.push_str(&format!(
        "- rows: {pass} pass / {fail} fail / {skip} skip\n\n"
    ));

    body.push_str("## Models\n\n");
    body.push_str(
        "| role | identifier | exists | file hash | tag hash | metadata | quantization hash |\n",
    );
    body.push_str("|---|---|---|---|---|---|---|\n");
    for model in model_manifest_entries(config) {
        body.push_str(&format!(
            "| {} | {} | {} | {} | {} | {} | {} |\n",
            model.role,
            model.identifier,
            model.path_exists,
            model.file_hash.as_deref().unwrap_or(""),
            model.tag_hash.as_deref().unwrap_or(""),
            model.metadata_status,
            model
                .quantization_hash
                .as_ref()
                .map(compact_json)
                .unwrap_or_default()
        ));
    }
    body.push('\n');

    body.push_str("## Datasets\n\n");
    body.push_str(
        "| suite | status | source | repo | revision | digest | license | selected | selected items | cache | reason |\n",
    );
    body.push_str("|---|---|---|---|---|---|---|---:|---|---|---|\n");
    if datasets.is_empty() {
        body.push_str(
            "| none | Skip | none |  |  |  |  | 0 |  |  | no dataset-backed suites selected |\n",
        );
    } else {
        for d in datasets {
            body.push_str(&format!(
                "| {} | {:?} | {} | {} | {} | {} | {} | {} | {} | {} | {} |\n",
                d.suite.as_str(),
                d.status,
                d.source,
                d.repo_id.as_deref().unwrap_or(""),
                d.revision.as_deref().unwrap_or(""),
                d.digest.as_deref().unwrap_or(""),
                d.license.as_deref().unwrap_or(""),
                d.selected_item_ids.len(),
                d.selected_item_ids.join(","),
                d.cache_path,
                d.reason.as_deref().unwrap_or("")
            ));
        }
    }
    body.push('\n');
    body.push_str("## Comparisons\n\n");
    body.push_str(&format!("- status: `{:?}`\n", comparison.status));
    if let Some(reason) = &comparison.reason {
        body.push_str(&format!("- reason: `{reason}`\n"));
    }
    body.push_str(&format!("- cases: `{}`\n\n", comparison.cases.len()));

    body.push_str("## Admission\n\n");
    body.push_str(&format!("- status: `{:?}`\n", admission.status));
    body.push_str(&format!("- verdict: `{}`\n", admission.verdict));
    if let Some(reason) = &admission.reason {
        body.push_str(&format!("- reason: `{reason}`\n"));
    }
    body.push_str(&format!(
        "- required evidence: `{}`\n",
        admission.required_evidence.len()
    ));
    body.push_str(&format!("- findings: `{}`\n\n", admission.findings.len()));
    if !admission.required_evidence.is_empty() {
        body.push_str("| evidence | status | rows | reason |\n");
        body.push_str("|---|---|---|---|\n");
        for evidence in &admission.required_evidence {
            body.push_str(&format!(
                "| {} | {:?} | {} | {} |\n",
                evidence.kind,
                evidence.status,
                evidence.rows,
                evidence.reason.as_deref().unwrap_or("")
            ));
        }
        body.push('\n');
    }
    if !admission.observed_evidence.is_empty() {
        body.push_str("### Observed Evidence\n\n");
        body.push_str("| evidence | status | rows | reason |\n");
        body.push_str("|---|---|---|---|\n");
        for evidence in &admission.observed_evidence {
            body.push_str(&format!(
                "| {} | {:?} | {} | {} |\n",
                evidence.kind,
                evidence.status,
                evidence.rows,
                evidence.reason.as_deref().unwrap_or("")
            ));
        }
        body.push('\n');
    }
    if !admission.findings.is_empty() {
        body.push_str("| severity | battery | case | metric | comparator | direction | delta |\n");
        body.push_str("|---|---|---|---|---|---|---|\n");
        for finding in &admission.findings {
            body.push_str(&format!(
                "| {} | {} | {} | {} | {} | {} | {:.6} |\n",
                finding.severity,
                finding.battery.as_str(),
                finding.case_id,
                finding.metric,
                finding.comparator,
                finding.direction,
                finding.delta
            ));
        }
        body.push('\n');
    }

    body.push_str("## Evidence Artifacts\n\n");
    body.push_str("| artifact | status | path | detail |\n");
    body.push_str("|---|---|---|---|\n");
    if artifacts.is_empty() {
        body.push_str("| none | skip |  | no evidence artifacts collected |\n");
    } else {
        for (name, artifact) in artifacts {
            let status = artifact.get("status").and_then(Value::as_str).unwrap_or("");
            let path = artifact.get("path").and_then(Value::as_str).unwrap_or("");
            let detail = artifact
                .get("verdict")
                .or_else(|| artifact.get("row_count"))
                .or_else(|| artifact.get("case_count"))
                .or_else(|| artifact.get("finding_count"))
                .or_else(|| artifact.get("reason"))
                .and_then(|v| {
                    v.as_str()
                        .map(str::to_string)
                        .or_else(|| v.as_u64().map(|n| n.to_string()))
                })
                .unwrap_or_default();
            body.push_str(&format!("| {name} | {status} | {path} | {detail} |\n"));
        }
    }
    body.push('\n');

    body.push_str("## Rows\n\n");
    body.push_str(
        "| battery | suite | case | item | model | model hash | prompt hash | status | reason |\n",
    );
    body.push_str("|---|---|---|---|---|---|---|---|---|\n");
    for r in rows {
        body.push_str(&format!(
            "| {} | {} | {} | {} | {} | {} | {} | {:?} | {} |\n",
            r.battery.as_str(),
            r.suite.map(|s| s.as_str()).unwrap_or(""),
            r.case_id,
            r.dataset_item_id.as_deref().unwrap_or(""),
            r.model,
            r.model_hash.as_deref().unwrap_or(""),
            r.prompt_hash.as_deref().unwrap_or(""),
            r.status,
            r.reason.as_deref().unwrap_or("")
        ));
    }
    fs::write(path, body).map_err(|e| format!("write {}: {e}", path.display()))
}

fn compact_json(value: &Value) -> String {
    serde_json::to_string(value).unwrap_or_default()
}

fn sanitize_path_component(raw: &str) -> String {
    let sanitized: String = raw
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '-' || c == '_' || c == '.' {
                c
            } else {
                '-'
            }
        })
        .collect();
    sanitized.trim_matches('-').to_string()
}

fn runtime_evidence_dir(config: &EvalConfig, label: &str, model: &str) -> PathBuf {
    config
        .out_dir
        .join("artifacts")
        .join("runtime_evidence")
        .join(format!(
            "{}-{}",
            sanitize_path_component(label),
            sanitize_path_component(&model_artifact_stem(model))
        ))
}

fn add_runtime_evidence_arg(args: &mut Vec<String>, dir: &Path) {
    args.push("--evidence-dir".to_string());
    args.push(dir.display().to_string());
}

fn repo_root() -> Option<PathBuf> {
    let out = Command::new("git")
        .args(["rev-parse", "--show-toplevel"])
        .output()
        .ok();
    if let Some(out) = out {
        if out.status.success() {
            let s = String::from_utf8_lossy(&out.stdout).trim().to_string();
            if !s.is_empty() {
                return Some(PathBuf::from(s));
            }
        }
    }

    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let fallback = manifest_dir.join("../..");
    if fallback.join("Cargo.toml").exists() {
        fallback.canonicalize().ok().or(Some(fallback))
    } else {
        None
    }
}

fn command_stdout(cmd: &str, args: &[&str]) -> Option<String> {
    let out = Command::new(cmd).args(args).output().ok()?;
    if !out.status.success() {
        return None;
    }
    let s = String::from_utf8_lossy(&out.stdout).trim().to_string();
    if s.is_empty() {
        None
    } else {
        Some(s)
    }
}

fn git_dirty() -> Option<bool> {
    let out = Command::new("git")
        .args(["status", "--porcelain"])
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    Some(!out.stdout.is_empty())
}

fn model_manifest_entries(config: &EvalConfig) -> Vec<ModelManifestEntry> {
    let mut out = Vec::new();
    out.push(model_manifest_entry("candidate", &config.model));
    if let Some(draft) = &config.draft {
        out.push(model_manifest_entry("draft", draft));
    }
    if let Some(baseline) = &config.baseline {
        out.push(model_manifest_entry("baseline", baseline));
    }
    if let Some(reference) = &config.reference {
        out.push(model_manifest_entry("reference", reference));
    }
    out
}

fn rocm_version() -> Option<String> {
    command_stdout("hipconfig", &["--version"])
        .or_else(|| command_stdout("/opt/rocm/bin/hipconfig", &["--version"]))
}

fn utc_now() -> String {
    command_stdout("date", &["-u", "+%Y-%m-%dT%H:%M:%SZ"]).unwrap_or_else(|| {
        let secs = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        format!("unix-{secs}")
    })
}

fn utc_stamp_compact() -> String {
    command_stdout("date", &["-u", "+%Y%m%dT%H%M%SZ"]).unwrap_or_else(|| {
        let secs = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        format!("unix-{secs}")
    })
}

fn home_dir() -> Option<PathBuf> {
    std::env::var_os("HOME").map(PathBuf::from)
}

#[cfg(test)]
mod tests {
    use super::*;
    // Used only by the host-profile comparison tests below; the collection
    // helpers themselves now live in hipfire-sysinfo.
    use hipfire_evidence::host_profile_hash;

    fn temp_path(name: &str) -> PathBuf {
        std::env::temp_dir().join(format!("hipfire-eval-test-{}-{name}", std::process::id()))
    }

    fn test_host_profile() -> HostProfile {
        collect_host_profile(
            Some("gfx1151".to_string()),
            HostProfileOverrides {
                memory_class: Some("lpddr5x".to_string()),
                memory_width_bits: Some(256),
                memory_bandwidth_gbps: Some(256.0),
            },
        )
    }

    fn write_minimal_hfq(path: &Path, metadata: &Value) {
        let metadata_json = serde_json::to_string(metadata).unwrap();
        let metadata_bytes = metadata_json.as_bytes();
        let mut index = Vec::new();
        index.extend_from_slice(&0u32.to_le_bytes());
        let metadata_offset = 32u64;
        let data_offset = 32u64 + metadata_bytes.len() as u64 + index.len() as u64;
        let mut f = File::create(path).unwrap();
        f.write_all(b"HFQM").unwrap();
        f.write_all(&1u32.to_le_bytes()).unwrap();
        f.write_all(&1u32.to_le_bytes()).unwrap();
        f.write_all(&0u32.to_le_bytes()).unwrap();
        f.write_all(&metadata_offset.to_le_bytes()).unwrap();
        f.write_all(&data_offset.to_le_bytes()).unwrap();
        f.write_all(metadata_bytes).unwrap();
        f.write_all(&index).unwrap();
    }

    fn write_gpqa_csv(root: &Path) {
        fs::create_dir_all(root.join("dataset")).unwrap();
        let mut f = File::create(root.join("dataset/gpqa_diamond.csv")).unwrap();
        writeln!(
            f,
            "Question,Correct Answer,Incorrect Answer 1,Incorrect Answer 2,Incorrect Answer 3,Record ID"
        )
        .unwrap();
        writeln!(
            f,
            "\"Which particle has charge -1?\",electron,proton,neutron,photon,rec-0"
        )
        .unwrap();
    }

    fn write_malformed_gpqa_csv(root: &Path) {
        fs::create_dir_all(root.join("dataset")).unwrap();
        let mut f = File::create(root.join("dataset/gpqa_diamond.csv")).unwrap();
        writeln!(
            f,
            "Question,Incorrect Answer 1,Incorrect Answer 2,Incorrect Answer 3"
        )
        .unwrap();
        writeln!(f, "\"Which particle has charge -1?\",proton,neutron,photon").unwrap();
    }

    fn write_humaneval_jsonl(root: &Path) {
        fs::create_dir_all(root.join("data")).unwrap();
        let rows = [
            json!({
                "task_id": "HumanEval/0",
                "prompt": "def add(a, b):\n    ",
                "canonical_solution": "return a + b\n",
                "test": "assert add(1, 2) == 3\n",
            }),
            json!({
                "task_id": "HumanEval/53",
                "prompt": "def below_zero(operations):\n    ",
                "canonical_solution": "return False\n",
                "test": "assert below_zero([]) == False\n",
            }),
        ];
        let mut f = File::create(root.join("data/HumanEval.jsonl")).unwrap();
        for row in rows {
            writeln!(f, "{}", serde_json::to_string(&row).unwrap()).unwrap();
        }
    }

    fn env_lock() -> std::sync::MutexGuard<'static, ()> {
        static LOCK: std::sync::OnceLock<std::sync::Mutex<()>> = std::sync::OnceLock::new();
        LOCK.get_or_init(|| std::sync::Mutex::new(()))
            .lock()
            .unwrap()
    }

    struct ScopedEnv {
        key: &'static str,
        old: Option<std::ffi::OsString>,
    }

    impl ScopedEnv {
        fn set(key: &'static str, value: &Path) -> Self {
            let old = std::env::var_os(key);
            unsafe {
                std::env::set_var(key, value);
            }
            Self { key, old }
        }
    }

    impl Drop for ScopedEnv {
        fn drop(&mut self) {
            unsafe {
                if let Some(old) = &self.old {
                    std::env::set_var(self.key, old);
                } else {
                    std::env::remove_var(self.key);
                }
            }
        }
    }

    #[test]
    fn parses_broader_cli_surface() {
        let cfg = parse_args_from([
            "hipfire-eval",
            "candidate.hfq",
            "--compare",
            "baseline.hfq",
            "--reference",
            "bf16",
            "--tier",
            "targeted",
            "--suite",
            "gpqa,human-eval",
            "--fetch-datasets",
            "--draft",
            "draft.hfq",
            "--dflash",
            "auto",
            "--profile",
            "passive",
            "--kldref",
            "ref.kldref.hfq",
            "--executor",
            "mock",
            "--fail-on-admission",
            "--evidence-json",
            "atlas.json",
            "--evidence-json",
            "profiler.json",
            "--evidence-dir",
            "runtime-artifacts",
            "--compare-variant",
            "baseline-json",
            "--performance-compare-variant",
            "baseline-perf",
        ])
        .unwrap();
        assert_eq!(cfg.tier, EvalTier::Medium);
        assert_eq!(cfg.baseline.as_deref(), Some("baseline.hfq"));
        assert_eq!(cfg.reference.as_deref(), Some("bf16"));
        assert_eq!(cfg.suites, vec![SuiteId::Gpqa, SuiteId::HumanEval]);
        assert!(cfg.fetch_datasets);
        assert_eq!(cfg.dflash, DflashMode::Auto);
        assert_eq!(cfg.profile, ProfileMode::Passive);
        assert_eq!(cfg.kldref.as_deref(), Some(Path::new("ref.kldref.hfq")));
        assert_eq!(cfg.executor, EvalExecutorMode::Mock);
        assert!(cfg.fail_on_admission);
        assert_eq!(
            cfg.evidence_json,
            vec![PathBuf::from("atlas.json"), PathBuf::from("profiler.json")]
        );
        assert_eq!(cfg.evidence_dirs, vec![PathBuf::from("runtime-artifacts")]);
        assert_eq!(cfg.baseline_variant.as_deref(), Some("baseline-json"));
        assert_eq!(
            cfg.performance_baseline_variant.as_deref(),
            Some("baseline-perf")
        );
    }

    #[test]
    fn positional_model_is_primary_eval_form() {
        let cfg = parse_args_from([
            "hipfire-eval",
            "candidate.hfq",
            "--battery",
            "speed",
            "--compare",
            "baseline.hfq",
        ])
        .unwrap();
        assert_eq!(cfg.model, "candidate.hfq");
        assert_eq!(cfg.baseline.as_deref(), Some("baseline.hfq"));
        assert_eq!(cfg.batteries, vec![BatteryId::Speed]);
    }

    #[test]
    fn parses_host_profile_overrides() {
        let cfg = parse_args_from([
            "hipfire-eval",
            "--model",
            "candidate.hfq",
            "--host-memory-class",
            "lpddr5x",
            "--host-memory-width-bits",
            "256",
            "--host-memory-bandwidth-gbps",
            "273.5",
        ])
        .unwrap();
        assert_eq!(cfg.host_memory_class.as_deref(), Some("lpddr5x"));
        assert_eq!(cfg.host_memory_width_bits, Some(256));
        assert_eq!(cfg.host_memory_bandwidth_gbps, Some(273.5));
    }

    #[test]
    fn dflash_auto_discovers_matching_qwen_draft() {
        let dir = temp_path("dflash-autodiscover");
        fs::create_dir_all(&dir).unwrap();
        let target = dir.join("qwen3.5-27b-mq4.hfq");
        let draft = dir.join("qwen3.5-27b-mq4.dflash.hfq");
        fs::write(&target, b"target").unwrap();
        fs::write(&draft, b"draft").unwrap();

        let cfg = parse_args_from([
            "hipfire-eval",
            "--model",
            target.to_str().unwrap(),
            "--battery",
            "dflash",
            "--dflash",
            "auto",
        ])
        .unwrap();
        assert_eq!(cfg.draft.as_deref(), Some(draft.to_str().unwrap()));

        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn explicit_dflash_draft_overrides_auto_discovery() {
        let dir = temp_path("dflash-explicit-draft");
        fs::create_dir_all(&dir).unwrap();
        let target = dir.join("qwen3.5-27b-mq4.hfq");
        let discovered = dir.join("qwen3.5-27b-mq4.dflash.hfq");
        let explicit = dir.join("custom-draft.hfq");
        fs::write(&target, b"target").unwrap();
        fs::write(&discovered, b"draft").unwrap();
        fs::write(&explicit, b"explicit").unwrap();

        let cfg = parse_args_from([
            "hipfire-eval",
            "--model",
            target.to_str().unwrap(),
            "--draft",
            explicit.to_str().unwrap(),
            "--battery",
            "dflash",
            "--dflash",
            "auto",
        ])
        .unwrap();
        assert_eq!(cfg.draft.as_deref(), Some(explicit.to_str().unwrap()));

        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn rejects_fetch_and_offline_together() {
        let err = parse_args_from([
            "hipfire-eval",
            "--model",
            "m.hfq",
            "--fetch-datasets",
            "--offline",
        ])
        .unwrap_err();
        assert!(err.contains("mutually exclusive"));
    }

    #[test]
    fn executor_defaults_to_auto_and_accepts_none() {
        let auto = parse_args_from(["hipfire-eval", "--model", "m.hfq"]).unwrap();
        assert_eq!(auto.executor, EvalExecutorMode::Auto);
        let none =
            parse_args_from(["hipfire-eval", "--model", "m.hfq", "--executor", "none"]).unwrap();
        assert_eq!(none.executor, EvalExecutorMode::None);
        let direct =
            parse_args_from(["hipfire-eval", "--model", "m.hfq", "--executor", "direct"]).unwrap();
        assert_eq!(direct.executor, EvalExecutorMode::Direct);
        let daemon =
            parse_args_from(["hipfire-eval", "--model", "m.hfq", "--executor", "daemon"]).unwrap();
        assert_eq!(daemon.executor, EvalExecutorMode::Daemon);
        assert_eq!(daemon.executor.as_str(), "daemon");
    }

    #[test]
    fn daemon_executor_builds_shared_load_and_generate_contracts() {
        let cfg = parse_args_from([
            "hipfire-eval",
            "--model",
            "m.hfq",
            "--executor",
            "daemon",
            "--kv-mode",
            "fp32",
            "--dflash",
            "auto",
            "--draft",
            "m.dflash.hfq",
            "--max-tokens",
            "17",
        ])
        .unwrap();
        let params = daemon_model_load_params(&cfg, 8192);
        assert_eq!(params.max_seq, 8192);
        assert_eq!(params.kv_cache.as_deref(), Some("fp32"));
        assert_eq!(params.dflash_mode.as_deref(), Some("auto"));
        assert_eq!(params.draft.as_deref(), Some("m.dflash.hfq"));

        let req = daemon_generate_request(
            "eval-smoke".to_string(),
            "What is 2+2?".to_string(),
            cfg.max_tokens,
            Some("worker-key".to_string()),
            None,
        );
        assert_eq!(req.id, "eval-smoke");
        assert_eq!(req.prompt, "What is 2+2?");
        assert_eq!(req.worker_key_id.as_deref(), Some("worker-key"));
        assert_eq!(req.sampling.max_tokens, 17);
        assert_eq!(req.evidence_dir, None);
        assert!(req
            .messages
            .as_ref()
            .is_some_and(|messages| messages.len() == 1));

        let evidence_dir = PathBuf::from("/tmp/hipfire-daemon-evidence");
        let req = daemon_generate_request(
            "eval-speed".to_string(),
            "Count quickly.".to_string(),
            cfg.max_tokens,
            Some("worker-key".to_string()),
            Some(&evidence_dir),
        );
        assert_eq!(
            req.evidence_dir.as_deref(),
            Some("/tmp/hipfire-daemon-evidence")
        );
    }

    #[test]
    fn daemon_executor_smoke_missing_model_is_explicit_skip() {
        let cfg = parse_args_from([
            "hipfire-eval",
            "--model",
            "missing-local-model.hfq",
            "--executor",
            "daemon",
            "--battery",
            "smoke",
        ])
        .unwrap();
        let ctx = EvalContext {
            commit_sha: None,
            git_branch: None,
            git_describe: None,
            git_dirty: None,
            binary_hash: None,
            arch: None,
            rocm: None,
            host_profile: test_host_profile(),
        };

        let rows = daemon_battery_rows(BatteryId::Smoke, &cfg, &ctx, &[]).unwrap();
        assert_eq!(rows.len(), 3);
        assert_eq!(rows[0].case_id, "load_metadata");
        assert_eq!(rows[1].case_id, "finite_greedy_decode");
        assert_eq!(rows[2].case_id, "multi_turn_reset_recall");
        assert!(rows.iter().all(|row| row.status == EvalStatus::Skip));
        assert!(rows
            .iter()
            .all(|row| row.metrics.get("executor").and_then(Value::as_str) == Some("daemon")));
        assert!(rows[0]
            .reason
            .as_deref()
            .unwrap_or("")
            .contains("local filesystem path"));
    }

    #[test]
    fn daemon_executor_speed_missing_model_is_explicit_skip() {
        let cfg = parse_args_from([
            "hipfire-eval",
            "--model",
            "missing-local-model.hfq",
            "--executor",
            "daemon",
            "--battery",
            "speed",
        ])
        .unwrap();
        let ctx = EvalContext {
            commit_sha: None,
            git_branch: None,
            git_describe: None,
            git_dirty: None,
            binary_hash: None,
            arch: None,
            rocm: None,
            host_profile: test_host_profile(),
        };

        let rows = daemon_battery_rows(BatteryId::Speed, &cfg, &ctx, &[]).unwrap();
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0].case_id, "daemon_prefill_decode_first");
        assert_eq!(rows[1].case_id, "daemon_prefill_decode_reset");
        assert!(rows.iter().all(|row| row.status == EvalStatus::Skip));
        assert!(rows.iter().all(|row| row
            .reason
            .as_deref()
            .unwrap_or("")
            .contains("local filesystem path")));
        assert!(rows.iter().all(|row| {
            row.metrics.get("executor").and_then(Value::as_str) == Some("daemon")
                && row.metrics.get("suite").and_then(Value::as_str) == Some("daemon_speed_anchor")
                && row.metrics.get("implemented").and_then(Value::as_bool) == Some(true)
        }));
    }

    #[test]
    fn daemon_executor_groups_smoke_speed_and_profile_under_one_shared_load_plan() {
        let cfg = parse_args_from([
            "hipfire-eval",
            "--model",
            "missing-local-model.hfq",
            "--battery",
            "smoke,speed,profile",
            "--no-cache",
        ])
        .unwrap();
        let ctx = EvalContext {
            commit_sha: None,
            git_branch: None,
            git_describe: None,
            git_dirty: None,
            binary_hash: None,
            arch: None,
            rocm: None,
            host_profile: test_host_profile(),
        };

        assert!(daemon_shared_model_load_enabled(&cfg));
        let rows = run_eval_batteries(&cfg, &ctx, &[]).unwrap();
        assert_eq!(rows.len(), 6);
        assert_eq!(
            rows.iter()
                .filter(|row| row.battery == BatteryId::Smoke)
                .count(),
            3
        );
        assert_eq!(
            rows.iter()
                .filter(|row| row.battery == BatteryId::Speed)
                .count(),
            2
        );
        let profile = rows
            .iter()
            .find(|row| row.battery == BatteryId::Profile)
            .expect("profile row");
        assert_eq!(profile.case_id, "model_profile_anchor");
        assert_eq!(
            profile
                .metrics
                .get("collection_scope")
                .and_then(Value::as_str),
            Some("model_backed_daemon_anchor")
        );
        assert!(rows
            .iter()
            .all(|row| row.metrics.get("executor").and_then(Value::as_str) == Some("daemon")));
        assert!(rows.iter().all(|row| row.status == EvalStatus::Skip));
    }

    #[test]
    fn coherence_and_agentic_group_under_shared_daemon_plan() {
        let _lock = env_lock();
        let _daemon = ScopedEnv::set("HIPFIRE_DAEMON_BIN", Path::new("/bin/true"));
        let cfg = parse_args_from([
            "hipfire-eval",
            "--model",
            "missing-local-model.hfq",
            "--battery",
            "coherence,agentic",
            "--no-cache",
        ])
        .unwrap();
        let ctx = EvalContext {
            commit_sha: None,
            git_branch: None,
            git_describe: None,
            git_dirty: None,
            binary_hash: None,
            arch: None,
            rocm: None,
            host_profile: test_host_profile(),
        };

        assert!(coherence_shared_model_load_enabled(&cfg));
        assert!(!daemon_shared_model_load_enabled(&cfg));
        let direct_cfg = parse_args_from([
            "hipfire-eval",
            "--model",
            "missing-local-model.hfq",
            "--executor",
            "direct",
            "--battery",
            "coherence,agentic",
            "--no-cache",
        ])
        .unwrap();
        assert!(!coherence_shared_model_load_enabled(&direct_cfg));
        let rows = run_eval_batteries(&cfg, &ctx, &[]).unwrap();
        assert_eq!(rows.len(), 9);
        assert_eq!(
            rows.iter()
                .filter(|row| row.battery == BatteryId::Coherence)
                .count(),
            5
        );
        assert_eq!(
            rows.iter()
                .filter(|row| row.battery == BatteryId::Agentic)
                .count(),
            4
        );
        assert!(rows.iter().all(|row| row.status == EvalStatus::Skip));
    }

    #[test]
    fn daemon_quality_without_reference_skips_before_spawning_daemon() {
        let _lock = env_lock();
        let model = temp_path("daemon-quality-no-reference-model.hfq");
        let out = temp_path("daemon-quality-no-reference-out");
        let daemon = temp_path("daemon-quality-no-reference-daemon.sh");
        let sentinel = temp_path("daemon-quality-no-reference-spawned");
        let _ = fs::remove_file(&model);
        let _ = fs::remove_dir_all(&out);
        let _ = fs::remove_file(&daemon);
        let _ = fs::remove_file(&sentinel);
        fs::write(&model, b"not a real hfq; no load should happen").unwrap();
        fs::write(
            &daemon,
            format!(
                "#!/bin/sh\nprintf spawned > '{}'\nexit 1\n",
                sentinel.display()
            ),
        )
        .unwrap();
        {
            use std::os::unix::fs::PermissionsExt;
            let mut perms = fs::metadata(&daemon).unwrap().permissions();
            perms.set_mode(0o755);
            fs::set_permissions(&daemon, perms).unwrap();
        }
        let _daemon_env = ScopedEnv::set("HIPFIRE_DAEMON_BIN", &daemon);
        let cfg = parse_args_from([
            "hipfire-eval",
            "--model",
            model.to_str().unwrap(),
            "--battery",
            "quality",
            "--no-cache",
            "--out",
            out.to_str().unwrap(),
        ])
        .unwrap();
        let ctx = EvalContext {
            commit_sha: None,
            git_branch: None,
            git_describe: None,
            git_dirty: None,
            binary_hash: None,
            arch: None,
            rocm: None,
            host_profile: test_host_profile(),
        };

        let rows = run_battery(BatteryId::Quality, &cfg, &ctx, &[]);
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].status, EvalStatus::Skip);
        assert!(rows[0]
            .reason
            .as_deref()
            .unwrap_or("")
            .contains("no KLD reference"));
        assert!(!sentinel.exists());

        let _ = fs::remove_file(model);
        let _ = fs::remove_dir_all(out);
        let _ = fs::remove_file(daemon);
        let _ = fs::remove_file(sentinel);
    }

    #[test]
    fn expands_tiers() {
        assert!(default_batteries(EvalTier::Fast).contains(&BatteryId::Smoke));
        assert!(default_batteries(EvalTier::Medium).contains(&BatteryId::Barrage));
        assert!(default_batteries(EvalTier::Long).contains(&BatteryId::Profile));
        assert!(default_batteries(EvalTier::Extensive).contains(&BatteryId::Vision));
        assert_eq!(EvalTier::Fast.budget().target_max_seconds, 60);
        assert!(EvalTier::Fast.budget().ci_suitable);
        assert_eq!(EvalTier::Medium.budget().target_max_seconds, 300);
        assert_eq!(EvalTier::Long.budget().target_max_seconds, 1200);
        assert_eq!(EvalTier::Extensive.budget().target_max_seconds, 0);
        assert_eq!(default_suites(EvalTier::Fast), vec![SuiteId::Gpqa]);
        assert_eq!(
            default_suites(EvalTier::Medium),
            vec![
                SuiteId::Gpqa,
                SuiteId::LmEvalMicro,
                SuiteId::HumanEval,
                SuiteId::DeepSwe,
                SuiteId::SweBench,
            ]
        );
        assert!(default_suites(EvalTier::Long).contains(&SuiteId::Ruler));
        assert!(default_suites(EvalTier::Long).contains(&SuiteId::NoLiMa));
        assert!(default_suites(EvalTier::Long).contains(&SuiteId::DeepSwe));
        assert!(default_suites(EvalTier::Long).contains(&SuiteId::SweBench));
    }

    #[test]
    fn parses_runtime_battery_aliases() {
        assert_eq!(BatteryId::parse("runtime").unwrap(), BatteryId::Runtime);
        assert_eq!(
            BatteryId::parse("server-runtime").unwrap(),
            BatteryId::Runtime
        );
        assert_eq!(BatteryId::Runtime.as_str(), "runtime");
    }

    #[test]
    fn runtime_battery_has_admission_evidence_rows() {
        let cfg = parse_args_from([
            "hipfire-eval",
            "--model",
            "candidate.hfq",
            "--battery",
            "runtime",
            "--executor",
            "mock",
        ])
        .unwrap();
        let ctx = EvalContext {
            commit_sha: None,
            git_branch: None,
            git_describe: None,
            git_dirty: None,
            binary_hash: None,
            arch: None,
            rocm: None,
            host_profile: test_host_profile(),
        };
        let rows = run_battery(BatteryId::Runtime, &cfg, &ctx, &[]);
        assert!(rows.iter().any(|row| row.case_id == "server_prefill_batch"));
        assert!(rows.iter().any(|row| row.case_id == "shared_prefix_fanout"));
        assert!(rows.iter().any(|row| row.case_id == "kv_reject_http"));
        assert!(rows
            .iter()
            .any(|row| row.case_id == "pipeline_parallel_gate"));
        assert!(rows.iter().all(|row| row.battery == BatteryId::Runtime));
    }

    #[test]
    fn barrage_defaults_to_gpqa_suite() {
        let cfg = parse_args_from([
            "hipfire-eval",
            "--model",
            "m.hfq",
            "--battery",
            "barrage",
            "--offline",
        ])
        .unwrap();
        assert_eq!(cfg.suites, vec![SuiteId::Gpqa]);
    }

    #[test]
    fn medium_barrage_defaults_to_small_native_suite_set() {
        let cfg = parse_args_from([
            "hipfire-eval",
            "--model",
            "m.hfq",
            "--tier",
            "medium",
            "--battery",
            "barrage",
            "--offline",
        ])
        .unwrap();
        assert_eq!(
            cfg.suites,
            vec![
                SuiteId::Gpqa,
                SuiteId::LmEvalMicro,
                SuiteId::HumanEval,
                SuiteId::DeepSwe,
                SuiteId::SweBench,
            ]
        );
    }

    #[test]
    fn lm_eval_micro_resolves_as_builtin_dataset() {
        let cfg = parse_args_from([
            "hipfire-eval",
            "--model",
            "m.hfq",
            "--battery",
            "barrage",
            "--suite",
            "lm_eval_micro",
            "--offline",
        ])
        .unwrap();
        let datasets = resolve_datasets(&cfg).unwrap();
        assert_eq!(datasets.len(), 1);
        assert_eq!(datasets[0].suite, SuiteId::LmEvalMicro);
        assert_eq!(datasets[0].source, "builtin");
        assert_eq!(datasets[0].status, EvalStatus::Pass);
        assert_eq!(
            datasets[0].selected_item_ids,
            selected_item_ids(SuiteId::LmEvalMicro)
        );
        assert!(datasets[0]
            .digest
            .as_deref()
            .unwrap_or("")
            .starts_with("fnv64:"));
    }

    #[test]
    fn lm_eval_micro_barrage_rows_are_native_prompt_canaries() {
        let cfg = parse_args_from([
            "hipfire-eval",
            "--model",
            "m.hfq",
            "--battery",
            "barrage",
            "--suite",
            "lm_eval_micro",
            "--offline",
        ])
        .unwrap();
        let ctx = EvalContext {
            commit_sha: None,
            git_branch: None,
            git_describe: None,
            git_dirty: None,
            binary_hash: None,
            arch: None,
            rocm: None,
            host_profile: test_host_profile(),
        };
        let datasets = resolve_datasets(&cfg).unwrap();
        let rows = barrage_rows(&cfg, &ctx, &datasets);
        assert_eq!(rows.len(), 3);
        assert!(rows
            .iter()
            .all(|row| row.suite == Some(SuiteId::LmEvalMicro)));
        assert!(rows
            .iter()
            .all(|row| row.case_id == "lm_eval_micro_zero_shot_native"));
        assert!(rows.iter().all(|row| row.status == EvalStatus::Skip));
        assert!(rows.iter().all(|row| row.prompt_hash.is_some()));
        assert!(rows
            .iter()
            .all(|row| row.dataset_source.as_deref() == Some("builtin")));
        assert!(rows
            .iter()
            .all(|row| row.dataset_revision.as_deref() == Some("hipfire-native-v1")));
        assert!(rows.iter().all(|row| row
            .dataset_digest
            .as_deref()
            .unwrap_or("")
            .starts_with("fnv64:")));
        assert!(rows
            .iter()
            .all(|row| row.dataset_license.as_deref() == Some("hipfire-native")));
        assert!(rows.iter().all(|row| {
            row.metrics.get("scoring_mode").and_then(Value::as_str) == Some("exact_letter")
        }));
    }

    #[test]
    fn examples_executor_materializes_lm_eval_micro_before_model_skip() {
        let cfg = parse_args_from([
            "hipfire-eval",
            "--model",
            "/tmp/definitely-missing.hfq",
            "--battery",
            "barrage",
            "--suite",
            "lm_eval_micro",
            "--executor",
            "examples",
            "--offline",
        ])
        .unwrap();
        let ctx = EvalContext {
            commit_sha: None,
            git_branch: None,
            git_describe: None,
            git_dirty: None,
            binary_hash: None,
            arch: None,
            rocm: None,
            host_profile: test_host_profile(),
        };
        let datasets = resolve_datasets(&cfg).unwrap();
        let rows = examples_barrage_rows(&cfg, &ctx, &datasets);
        assert_eq!(rows.len(), 3);
        assert!(rows
            .iter()
            .all(|row| row.suite == Some(SuiteId::LmEvalMicro)));
        assert!(rows.iter().all(|row| row.status == EvalStatus::Skip));
        assert!(rows.iter().all(|row| row.prompt_hash.is_some()));
        assert!(rows.iter().all(|row| {
            row.metrics.get("executor").and_then(Value::as_str) == Some("examples")
        }));
        assert!(rows.iter().all(|row| {
            row.reason
                .as_deref()
                .unwrap_or("")
                .contains("local filesystem path")
        }));
    }

    #[test]
    fn examples_barrage_rows_include_configured_comparators() {
        let cfg = parse_args_from([
            "hipfire-eval",
            "--model",
            "/tmp/missing-candidate.hfq",
            "--baseline",
            "/tmp/missing-baseline.hfq",
            "--reference",
            "/tmp/missing-reference.hfq",
            "--battery",
            "barrage",
            "--suite",
            "lm_eval_micro",
            "--executor",
            "examples",
            "--offline",
        ])
        .unwrap();
        let ctx = EvalContext {
            commit_sha: None,
            git_branch: None,
            git_describe: None,
            git_dirty: None,
            binary_hash: None,
            arch: None,
            rocm: None,
            host_profile: test_host_profile(),
        };
        let datasets = resolve_datasets(&cfg).unwrap();
        let rows = examples_barrage_rows(&cfg, &ctx, &datasets);
        let selected = selected_item_ids(SuiteId::LmEvalMicro).len();
        assert_eq!(rows.len(), selected * 3);
        for model in [
            "/tmp/missing-candidate.hfq",
            "/tmp/missing-baseline.hfq",
            "/tmp/missing-reference.hfq",
        ] {
            assert_eq!(
                rows.iter().filter(|row| row.model == model).count(),
                selected
            );
        }
        assert!(rows
            .iter()
            .all(|row| row.suite == Some(SuiteId::LmEvalMicro)));
        assert!(rows.iter().all(|row| row.status == EvalStatus::Skip));
        assert!(rows.iter().all(|row| row.prompt_hash.is_some()));
        assert!(rows.iter().all(|row| {
            row.metrics.get("executor").and_then(Value::as_str) == Some("examples")
        }));
    }

    #[test]
    fn sanitizes_default_output_model_stem() {
        let out = default_output_dir("/tmp/qwen3.5-9b-awq-mq4.hfq", EvalTier::Fast);
        let rendered = out.display().to_string();
        assert!(rendered.contains(".hipfire/eval-results/runs/"));
        assert!(rendered.contains("qwen3.5-9b-awq-mq4-fast"));
    }

    #[test]
    fn default_result_cache_is_outside_source_tree() {
        let rendered = default_result_cache().display().to_string();
        assert!(rendered.contains(".hipfire/eval-results/cache"));
    }

    #[test]
    fn parses_result_cache_flags() {
        let cache = temp_path("custom-result-cache");
        let cfg = parse_args_from([
            "hipfire-eval",
            "--model",
            "candidate.hfq",
            "--result-cache",
            cache.to_str().unwrap(),
            "--force",
        ])
        .unwrap();
        assert_eq!(cfg.result_cache, cache);
        assert_eq!(cfg.cache_mode, EvalCacheMode::Force);

        let cfg = parse_args_from([
            "hipfire-eval",
            "--model",
            "candidate.hfq",
            "--cache-dir",
            "alias-cache",
            "--regenerate",
        ])
        .unwrap();
        assert_eq!(cfg.result_cache, PathBuf::from("alias-cache"));
        assert_eq!(cfg.cache_mode, EvalCacheMode::Regenerate);

        let cfg =
            parse_args_from(["hipfire-eval", "--model", "candidate.hfq", "--no-cache"]).unwrap();
        assert_eq!(cfg.cache_mode, EvalCacheMode::Off);
    }

    #[test]
    fn parse_benchmark_repeat_flags() {
        let cfg =
            parse_args_from(["hipfire-eval", "--model", "candidate.hfq", "--benchmark"]).unwrap();
        assert!(cfg.benchmark);
        assert_eq!(cfg.runs, 5);
        assert_eq!(cfg.warmup_runs, 0);

        let cfg = parse_args_from([
            "hipfire-eval",
            "--model",
            "candidate.hfq",
            "--benchmark",
            "--runs",
            "3",
            "--warmup-runs",
            "1",
        ])
        .unwrap();
        assert!(cfg.benchmark);
        assert_eq!(cfg.runs, 3);
        assert_eq!(cfg.warmup_runs, 1);
    }

    #[test]
    fn result_cache_key_ignores_suites_for_non_barrage_batteries() {
        let ctx = EvalContext {
            commit_sha: Some("commit".to_string()),
            git_branch: Some("branch".to_string()),
            git_describe: Some("describe".to_string()),
            git_dirty: Some(false),
            binary_hash: Some("bin".to_string()),
            arch: Some("arch".to_string()),
            rocm: Some("rocm".to_string()),
            host_profile: test_host_profile(),
        };
        let mut cfg = parse_args_from([
            "hipfire-eval",
            "--model",
            "candidate.hfq",
            "--suite",
            "gpqa",
        ])
        .unwrap();
        let key_a = result_cache_key(BatteryId::Smoke, &cfg, &ctx, &[]).unwrap();
        cfg.suites = vec![SuiteId::Gpqa, SuiteId::LmEvalMicro, SuiteId::DeepSwe];
        let key_b = result_cache_key(BatteryId::Smoke, &cfg, &ctx, &[]).unwrap();
        assert_eq!(key_a, key_b);

        let barrage_key_a = result_cache_key(BatteryId::Barrage, &cfg, &ctx, &[]).unwrap();
        cfg.suites = vec![SuiteId::Gpqa];
        let barrage_key_b = result_cache_key(BatteryId::Barrage, &cfg, &ctx, &[]).unwrap();
        assert_ne!(barrage_key_a, barrage_key_b);
    }

    #[test]
    fn result_cache_key_includes_benchmark_repeat_settings() {
        let ctx = EvalContext {
            commit_sha: Some("commit".to_string()),
            git_branch: Some("branch".to_string()),
            git_describe: Some("describe".to_string()),
            git_dirty: Some(false),
            binary_hash: Some("bin".to_string()),
            arch: Some("arch".to_string()),
            rocm: Some("rocm".to_string()),
            host_profile: test_host_profile(),
        };
        let cfg_a = parse_args_from(["hipfire-eval", "--model", "candidate.hfq"]).unwrap();
        let cfg_b =
            parse_args_from(["hipfire-eval", "--model", "candidate.hfq", "--runs", "3"]).unwrap();
        let key_a = result_cache_key(BatteryId::Speed, &cfg_a, &ctx, &[]).unwrap();
        let key_b = result_cache_key(BatteryId::Speed, &cfg_b, &ctx, &[]).unwrap();
        assert_ne!(key_a, key_b);
    }

    #[test]
    fn result_cache_key_includes_hardware_bucket_identity() {
        let mut host_a = test_host_profile();
        host_a.hardware_bucket = "dgpu:gfx1201:0x7550:64cu:16gib:gddr6:256bit:640gbps".to_string();
        host_a.host_profile_hash = host_profile_hash(&host_a);
        let mut host_b = host_a.clone();
        host_b.hardware_bucket =
            "apu_uma:gfx1151:0x1586:40cu:1gib:lpddr5x:256bit:256gbps".to_string();
        host_b.host_profile_hash = host_profile_hash(&host_b);
        let ctx_a = EvalContext {
            commit_sha: Some("commit".to_string()),
            git_branch: Some("branch".to_string()),
            git_describe: Some("describe".to_string()),
            git_dirty: Some(false),
            binary_hash: Some("bin".to_string()),
            arch: Some("arch".to_string()),
            rocm: Some("rocm".to_string()),
            host_profile: host_a,
        };
        let ctx_b = EvalContext {
            host_profile: host_b,
            ..ctx_a.clone()
        };
        let cfg = parse_args_from(["hipfire-eval", "--model", "candidate.hfq"]).unwrap();
        let key_a = result_cache_key(BatteryId::Speed, &cfg, &ctx_a, &[]).unwrap();
        let key_b = result_cache_key(BatteryId::Speed, &cfg, &ctx_b, &[]).unwrap();
        assert_ne!(key_a, key_b);
    }

    #[test]
    fn run_eval_reuses_cached_battery_rows() {
        let cache = temp_path("result-cache-reuse");
        let out1 = temp_path("result-cache-run-1");
        let out2 = temp_path("result-cache-run-2");
        let _ = fs::remove_dir_all(&cache);
        let _ = fs::remove_dir_all(&out1);
        let _ = fs::remove_dir_all(&out2);

        let cfg1 = parse_args_from([
            "hipfire-eval",
            "--model",
            "candidate.hfq",
            "--battery",
            "smoke",
            "--executor",
            "mock",
            "--out",
            out1.to_str().unwrap(),
            "--result-cache",
            cache.to_str().unwrap(),
        ])
        .unwrap();
        run_eval(cfg1).unwrap();
        let first_rows = read_jsonl_rows(&out1.join("results.jsonl"));
        assert!(first_rows
            .iter()
            .all(|row| row.metrics.get("cache_hit") != Some(&json!(true))));

        let cfg2 = parse_args_from([
            "hipfire-eval",
            "--model",
            "candidate.hfq",
            "--battery",
            "smoke",
            "--executor",
            "mock",
            "--out",
            out2.to_str().unwrap(),
            "--result-cache",
            cache.to_str().unwrap(),
        ])
        .unwrap();
        run_eval(cfg2).unwrap();
        let second_rows = read_jsonl_rows(&out2.join("results.jsonl"));
        assert_eq!(first_rows.len(), second_rows.len());
        assert!(second_rows
            .iter()
            .all(|row| row.metrics.get("cache_hit") == Some(&json!(true))));
        assert!(second_rows.iter().all(|row| row
            .metrics
            .get("cache_key")
            .and_then(Value::as_str)
            .is_some()));

        let _ = fs::remove_dir_all(cache);
        let _ = fs::remove_dir_all(out1);
        let _ = fs::remove_dir_all(out2);
    }

    #[test]
    fn run_eval_repeats_and_emits_benchmark_aggregates() {
        let out = temp_path("benchmark-repeat-run");
        let cache = temp_path("benchmark-repeat-cache");
        let _ = fs::remove_dir_all(&out);
        let _ = fs::remove_dir_all(&cache);
        let cfg = parse_args_from([
            "hipfire-eval",
            "--model",
            "candidate.hfq",
            "--battery",
            "speed",
            "--executor",
            "mock",
            "--runs",
            "3",
            "--out",
            out.to_str().unwrap(),
            "--result-cache",
            cache.to_str().unwrap(),
            "--no-cache",
        ])
        .unwrap();
        run_eval(cfg).unwrap();
        let rows = read_jsonl_rows(&out.join("results.jsonl"));
        let raw = rows
            .iter()
            .filter(|row| {
                row.case_id == "pp32_pp128_ttft_decode"
                    && row.metrics.get("benchmark_sample") == Some(&json!(true))
            })
            .collect::<Vec<_>>();
        assert_eq!(raw.len(), 3);
        assert_eq!(raw[0].metrics["run_index"], json!(1));
        assert_eq!(raw[2].metrics["run_index"], json!(3));
        let aggregate = rows
            .iter()
            .find(|row| row.case_id == "pp32_pp128_ttft_decode::aggregate")
            .expect("aggregate row");
        assert_eq!(aggregate.status, EvalStatus::Pass);
        assert_eq!(aggregate.metrics["benchmark_aggregate"], json!(true));
        assert_eq!(aggregate.metrics["sample_count"], json!(3));
        assert!(aggregate.metrics.get("tok_s_median").is_some());
        assert!(aggregate.metrics.get("tok_s_stddev").is_some());

        let _ = fs::remove_dir_all(out);
        let _ = fs::remove_dir_all(cache);
    }

    #[test]
    fn passive_profile_records_skip_without_examples_executor() {
        let out = temp_path("passive-profile-mock-skip-run");
        let _ = fs::remove_dir_all(&out);
        let cfg = parse_args_from([
            "hipfire-eval",
            "--model",
            "candidate.hfq",
            "--battery",
            "smoke",
            "--executor",
            "mock",
            "--profile",
            "passive",
            "--out",
            out.to_str().unwrap(),
            "--no-cache",
        ])
        .unwrap();
        run_eval(cfg).unwrap();
        let rows = read_jsonl_rows(&out.join("results.jsonl"));
        let profile = rows
            .iter()
            .find(|row| row.battery == BatteryId::Profile && row.case_id == "rocprof_speed_anchor")
            .expect("profile skip row");
        assert_eq!(profile.status, EvalStatus::Skip);
        assert!(profile
            .reason
            .as_deref()
            .unwrap_or("")
            .contains("requires --executor auto, examples, or direct"));
        assert_eq!(profile.metrics["profiling_requested"], json!(true));

        let _ = fs::remove_dir_all(out);
    }

    #[test]
    fn parses_rocprof_kernel_stats_csv() {
        let csv = "\
Name,Calls,TotalDurationNs,AverageNs,Percentage,MinNs,MaxNs,StdDev
attention_dflash,4,2000000,500000,66.7,400000,600000,10000
gemm<foo,bar>,2,1000000,500000,33.3,450000,550000,10000
";
        let rows = parse_rocprof_kernel_stats_csv_text(csv).unwrap();
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0].name, "attention_dflash");
        assert_eq!(rows[0].calls, 4);
        assert_eq!(rows[0].duration_us, 2000.0);
        assert_eq!(rows[0].average_us, 500.0);
        assert_eq!(rows[1].name, "gemm<foo,bar>");
    }

    fn read_jsonl_rows(path: &Path) -> Vec<EvalResult> {
        fs::read_to_string(path)
            .unwrap()
            .lines()
            .filter(|line| !line.trim().is_empty())
            .map(|line| serde_json::from_str(line).unwrap())
            .collect()
    }

    #[test]
    fn serializes_result_row_as_json() {
        let cfg = EvalConfig {
            model: "m.hfq".to_string(),
            draft: None,
            baseline: Some("baseline.hfq".to_string()),
            reference: Some("bf16".to_string()),
            tier: EvalTier::Fast,
            batteries: vec![BatteryId::Retrieval],
            suites: vec![],
            out_dir: PathBuf::from("out"),
            kv_mode: Some("q8".to_string()),
            max_tokens: 8,
            dflash: DflashMode::Off,
            profile: ProfileMode::Off,
            quality_max_chunks: Some(1),
            kldref: None,
            quality_json: None,
            performance_json: None,
            evidence_json: Vec::new(),
            evidence_dirs: Vec::new(),
            candidate_variant: None,
            baseline_variant: None,
            reference_variant: None,
            performance_candidate_variant: None,
            performance_baseline_variant: None,
            performance_reference_variant: None,
            executor: EvalExecutorMode::None,
            fetch_datasets: false,
            offline: true,
            dataset_cache: PathBuf::from("datasets"),
            result_cache: PathBuf::from("cache"),
            cache_mode: EvalCacheMode::Use,
            runs: 1,
            warmup_runs: 0,
            benchmark: false,
            host_memory_class: None,
            host_memory_width_bits: None,
            host_memory_bandwidth_gbps: None,
            fail_on_admission: false,
            models_spec: None,
            dry_run: false,
            status: false,
            fetch: false,
        };
        let ctx = EvalContext {
            commit_sha: Some("abc".to_string()),
            git_branch: Some("evaluation-harness".to_string()),
            git_describe: Some("v0.2.0-1-gabc".to_string()),
            git_dirty: Some(false),
            binary_hash: Some("def".to_string()),
            arch: Some("gfx1151".to_string()),
            rocm: None,
            host_profile: test_host_profile(),
        };
        let row = pass_row(
            BatteryId::Retrieval,
            None,
            "fixture",
            None,
            &cfg,
            &ctx,
            None,
            BTreeMap::new(),
        );
        let s = serde_json::to_string(&row).unwrap();
        assert!(s.contains("\"hipfire_version\""));
        assert!(s.contains("\"battery\":\"retrieval\""));
        assert!(s.contains("\"status\":\"pass\""));
        assert!(s.contains("\"baseline\":\"baseline.hfq\""));
        assert!(s.contains("\"model_hash\":\"tag:"));
        assert!(s.contains("\"baseline_hash\":\"tag:"));
        assert!(s.contains("\"reference_hash\":\"tag:"));
        assert!(s.contains("\"git_commit\":\"abc\""));
        assert!(s.contains("\"commit_sha\":\"abc\""));
        assert!(s.contains("\"git_branch\":\"evaluation-harness\""));
        assert!(s.contains("\"git_describe\":\"v0.2.0-1-gabc\""));
        assert!(s.contains("\"git_dirty\":false"));
        assert_eq!(row.host_profile_hash, ctx.host_profile.host_profile_hash);
        assert_eq!(row.hardware_bucket, ctx.host_profile.hardware_bucket);
        assert!(s.contains("\"host_profile_hash\""));
        assert!(s.contains("\"hardware_bucket\""));
        assert_eq!(
            serde_json::to_string(&SuiteId::HumanEval).unwrap(),
            "\"humaneval\""
        );
        assert_eq!(
            serde_json::to_string(&SuiteId::NoLiMa).unwrap(),
            "\"nolima\""
        );
        assert_eq!(
            serde_json::from_str::<SuiteId>("\"human_eval\"").unwrap(),
            SuiteId::HumanEval
        );
        assert_eq!(
            serde_json::from_str::<SuiteId>("\"no_lima\"").unwrap(),
            SuiteId::NoLiMa
        );
    }

    #[test]
    fn result_rows_record_file_backed_model_hashes() {
        let model = temp_path("row-candidate.hfq");
        let baseline = temp_path("row-baseline.hfq");
        fs::write(&model, b"candidate bytes").unwrap();
        fs::write(&baseline, b"baseline bytes").unwrap();
        let cfg = parse_args_from([
            "hipfire-eval",
            "--model",
            model.to_str().unwrap(),
            "--baseline",
            baseline.to_str().unwrap(),
            "--battery",
            "speed",
        ])
        .unwrap();
        let ctx = EvalContext {
            commit_sha: None,
            git_branch: None,
            git_describe: None,
            git_dirty: None,
            binary_hash: None,
            arch: None,
            rocm: None,
            host_profile: test_host_profile(),
        };

        let row = pass_row(
            BatteryId::Speed,
            None,
            "decode",
            None,
            &cfg,
            &ctx,
            None,
            BTreeMap::from([("tok_s".to_string(), json!(10.0))]),
        );
        assert_eq!(
            row.model_hash.as_deref(),
            model_hash(model.to_str().unwrap()).as_deref()
        );
        assert_eq!(
            row.baseline_hash.as_deref(),
            model_hash(baseline.to_str().unwrap()).as_deref()
        );
        assert!(row.model_hash.as_deref().unwrap_or("").len() >= 16);
        assert!(row.draft_hash.is_none());

        let _ = fs::remove_file(model);
        let _ = fs::remove_file(baseline);
    }

    #[test]
    fn run_metadata_artifact_records_version_and_git() {
        let cfg = parse_args_from(["hipfire-eval", "--model", "candidate.hfq"]).unwrap();
        let ctx = EvalContext {
            commit_sha: Some("abc".to_string()),
            git_branch: Some("evaluation-harness".to_string()),
            git_describe: Some("v0.2.0-1-gabc".to_string()),
            git_dirty: Some(true),
            binary_hash: Some("binhash".to_string()),
            arch: Some("gfx1151".to_string()),
            rocm: Some("7.13.26176".to_string()),
            host_profile: test_host_profile(),
        };

        let artifact = run_metadata_artifact_value(&cfg, &ctx);
        assert_eq!(artifact["kind"], "run_metadata");
        assert_eq!(artifact["status"], "collected");
        assert!(artifact["runner_version"].as_str().is_some());
        assert!(artifact["hipfire_version"].as_str().is_some());
        assert_eq!(artifact["git"]["commit"], "abc");
        assert_eq!(artifact["git"]["branch"], "evaluation-harness");
        assert_eq!(artifact["git"]["describe"], "v0.2.0-1-gabc");
        assert_eq!(artifact["git"]["dirty"], true);
        assert_eq!(artifact["binary"]["hash"], "binhash");
        assert_eq!(artifact["host"]["arch"], "gfx1151");
    }

    #[test]
    fn version_report_includes_runner_and_git_fields() {
        let report = version_report();
        assert!(report.contains("hipfire-eval "));
        assert!(report.contains("hipfire_version "));
        assert!(report.contains("git_commit "));
        assert!(report.contains("git_branch "));
        assert!(report.contains("git_describe "));
        assert!(report.contains("git_dirty "));
        assert!(report.contains("binary_hash "));
    }

    #[test]
    fn extracts_embedded_hfq_quantization_hash_for_manifest() {
        let path = temp_path("candidate.hfq");
        let metadata = json!({
            "architecture": "qwen3",
            "quantization_hash": {
                "algorithm": "xxh64",
                "seed": 0,
                "scope": "hfq_tensor_index_and_payload_v1",
                "value": "0123456789abcdef",
                "producer": {
                    "package": "hipfire-quantize",
                    "hipfire_version": "0.2.0",
                    "git_commit": "abc",
                    "git_dirty": false
                }
            }
        });
        write_minimal_hfq(&path, &metadata);

        let entry = model_manifest_entry("candidate", path.to_str().unwrap());
        assert_eq!(entry.role, "candidate");
        assert!(entry.path_exists);
        assert_eq!(entry.hfq_arch_id, Some(1));
        assert_eq!(entry.metadata_status, "pass");
        assert_eq!(
            entry
                .quantization_hash
                .as_ref()
                .and_then(|v| v.get("value"))
                .and_then(Value::as_str),
            Some("0123456789abcdef")
        );

        let _ = fs::remove_file(path);
    }

    #[test]
    fn records_tag_models_without_hfq_metadata() {
        let entry = model_manifest_entry("candidate", "qwen3.5:9b");
        assert!(!entry.path_exists);
        assert!(entry.file_hash.is_none());
        assert!(entry.tag_hash.as_deref().unwrap_or("").starts_with("tag:"));
        assert_eq!(entry.metadata_status, "skip");
        assert!(entry.quantization_hash.is_none());
    }

    #[test]
    fn examples_executor_routes_fast_shape_batteries_to_committed_prompts() {
        let out = temp_path("examples-shape-longctx");
        let cfg = parse_args_from([
            "hipfire-eval",
            "--model",
            "missing-local-model.hfq",
            "--executor",
            "examples",
            "--battery",
            "smoke,prompt_shape,structured,longctx,profile",
            "--out",
            out.to_str().unwrap(),
        ])
        .unwrap();
        let ctx = EvalContext {
            commit_sha: None,
            git_branch: None,
            git_describe: None,
            git_dirty: None,
            binary_hash: None,
            arch: None,
            rocm: None,
            host_profile: test_host_profile(),
        };

        let smoke = examples_battery_rows(BatteryId::Smoke, &cfg, &ctx, &[]).unwrap();
        assert_eq!(smoke[0].case_id, "finite_greedy_decode");
        assert_eq!(
            smoke[0].prompt_path.as_deref(),
            Some("benchmarks/prompts/qwen2_smoke.txt")
        );
        assert_eq!(smoke[0].status, EvalStatus::Skip);
        assert_eq!(smoke[1].case_id, "multi_turn_reset_recall");
        assert_eq!(smoke[1].status, EvalStatus::Skip);
        assert_eq!(
            smoke[1].metrics.get("executor").and_then(Value::as_str),
            Some("direct")
        );
        assert!(smoke[1]
            .reason
            .as_deref()
            .unwrap_or("")
            .contains("local filesystem path"));

        let prompt_shape = examples_battery_rows(BatteryId::PromptShape, &cfg, &ctx, &[]).unwrap();
        assert_eq!(prompt_shape[0].case_id, "whitespace_template_canary");
        assert_eq!(
            prompt_shape[0].prompt_path.as_deref(),
            Some("benchmarks/prompts/lru_cache_pep8_strict.txt")
        );

        let structured = examples_battery_rows(BatteryId::Structured, &cfg, &ctx, &[]).unwrap();
        assert_eq!(structured[0].case_id, "tool_call_jsonish_canary");
        assert_eq!(
            structured[0].prompt_path.as_deref(),
            Some("benchmarks/prompts/tool_call_read_file.txt")
        );

        let longctx = examples_battery_rows(BatteryId::Longctx, &cfg, &ctx, &[]).unwrap();
        assert_eq!(longctx[0].case_id, "multidoc_needle_long_state");
        assert_eq!(longctx[0].status, EvalStatus::Skip);
        assert!(longctx[0].prompt_hash.is_some());
        assert!(longctx[0]
            .prompt_path
            .as_deref()
            .unwrap_or("")
            .ends_with("artifacts/runtime_prompts/longctx_multidoc_0.txt"));
        assert_eq!(
            longctx[0].metrics.get("fixture").and_then(Value::as_str),
            Some("longprose_multidoc")
        );
        assert!(
            longctx[0]
                .metrics
                .get("longctx_max_seq")
                .and_then(Value::as_u64)
                .unwrap_or(0)
                > 4096
        );

        let profile = examples_battery_rows(BatteryId::Profile, &cfg, &ctx, &[]).unwrap();
        assert_eq!(profile[0].case_id, "model_profile_anchor");
        assert_eq!(
            profile[0].prompt_path.as_deref(),
            Some("benchmarks/prompts/dflash_resident_smoke.txt")
        );
        assert_eq!(profile[0].status, EvalStatus::Skip);
        assert_eq!(
            profile[0]
                .metrics
                .get("collection_scope")
                .and_then(Value::as_str),
            Some("model_backed_run_anchor")
        );
        assert_eq!(
            profile[0]
                .metrics
                .get("moe_router_histogram_expected_when_moe")
                .and_then(Value::as_bool),
            Some(true)
        );
        assert_eq!(
            profile[0]
                .metrics
                .get("expected_runtime_evidence_kinds")
                .and_then(Value::as_array)
                .map(|values| values.iter().filter_map(Value::as_str).collect::<Vec<_>>()),
            Some(vec![
                "performance",
                "memory",
                "launch_counts",
                "moe_router_histogram"
            ])
        );
    }

    #[test]
    fn direct_executor_routes_smoke_reset_to_session_executor() {
        let out = temp_path("direct-session-smoke");
        let cfg = parse_args_from([
            "hipfire-eval",
            "--model",
            "missing-local-model.hfq",
            "--executor",
            "direct",
            "--battery",
            "smoke",
            "--out",
            out.to_str().unwrap(),
        ])
        .unwrap();
        let ctx = EvalContext {
            commit_sha: None,
            git_branch: None,
            git_describe: None,
            git_dirty: None,
            binary_hash: None,
            arch: None,
            rocm: None,
            host_profile: test_host_profile(),
        };

        let rows = direct_battery_rows(BatteryId::Smoke, &cfg, &ctx, &[]).unwrap();
        assert_eq!(rows[1].case_id, "multi_turn_reset_recall");
        assert_eq!(rows[1].status, EvalStatus::Skip);
        assert_eq!(
            rows[1].prompt_path.as_deref(),
            Some("benchmarks/prompts/trains-meet.txt")
        );
        assert_eq!(
            rows[1].metrics.get("executor").and_then(Value::as_str),
            Some("direct")
        );
        assert!(rows[1]
            .reason
            .as_deref()
            .unwrap_or("")
            .contains("local filesystem path"));
    }

    #[test]
    fn auto_executor_uses_daemon_for_implemented_smoke_rows() {
        let cfg = parse_args_from([
            "hipfire-eval",
            "--model",
            "missing-local-model.hfq",
            "--battery",
            "smoke",
        ])
        .unwrap();
        assert_eq!(cfg.executor, EvalExecutorMode::Auto);
        let ctx = EvalContext {
            commit_sha: None,
            git_branch: None,
            git_describe: None,
            git_dirty: None,
            binary_hash: None,
            arch: None,
            rocm: None,
            host_profile: test_host_profile(),
        };

        let rows = run_battery(BatteryId::Smoke, &cfg, &ctx, &[]);
        assert_eq!(rows[0].case_id, "load_metadata");
        assert_eq!(
            rows[0].metrics.get("executor").and_then(Value::as_str),
            Some("daemon")
        );
        assert_eq!(rows[1].case_id, "finite_greedy_decode");
        assert_eq!(
            rows[1].prompt_path.as_deref(),
            Some("benchmarks/prompts/qwen2_smoke.txt")
        );
        assert!(rows[1]
            .reason
            .as_deref()
            .unwrap_or("")
            .contains("local filesystem path"));
    }

    #[test]
    fn explicit_examples_executor_still_uses_examples_when_available() {
        if resolve_run_example_bin().is_none() {
            return;
        }
        let cfg = parse_args_from([
            "hipfire-eval",
            "--model",
            "missing-local-model.hfq",
            "--executor",
            "examples",
            "--battery",
            "smoke",
        ])
        .unwrap();
        let ctx = EvalContext {
            commit_sha: None,
            git_branch: None,
            git_describe: None,
            git_dirty: None,
            binary_hash: None,
            arch: None,
            rocm: None,
            host_profile: test_host_profile(),
        };

        let rows = run_battery(BatteryId::Smoke, &cfg, &ctx, &[]);
        assert_eq!(rows[0].case_id, "finite_greedy_decode");
        assert_eq!(
            rows[0].metrics.get("executor").and_then(Value::as_str),
            Some("examples")
        );
    }

    #[test]
    fn materializes_gpqa_prompt_from_extracted_csv() {
        let root = temp_path("gpqa-cache");
        write_gpqa_csv(&root);
        let items = gpqa_materialized_items(&root, &["gpqa_diamond:0".to_string()]).unwrap();
        assert_eq!(items.len(), 1);
        let item = &items[0];
        assert!(item.item_id.starts_with("gpqa_diamond:0"));
        assert!(item.prompt.contains("Return only the letter"));
        assert!(item.prompt.contains("Which particle has charge -1?"));
        assert_eq!(item.choices.len(), 4);
        assert!(["A", "B", "C", "D"].contains(&item.answer_label.as_str()));

        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn fetch_datasets_can_use_local_mirror_for_gpqa_without_network() {
        let _lock = env_lock();
        let mirror = temp_path("gpqa-mirror");
        let cache = temp_path("gpqa-fetch-cache");
        let out = temp_path("gpqa-fetch-artifact-out");
        let _env = ScopedEnv::set("HIPFIRE_EVAL_DATASET_MIRROR", &mirror);
        write_gpqa_csv(&mirror.join("gpqa"));
        fs::copy(
            mirror.join("gpqa/dataset/gpqa_diamond.csv"),
            mirror.join("gpqa/dataset/gpqa_main.csv"),
        )
        .unwrap();
        fs::create_dir_all(&out).unwrap();

        let cfg = parse_args_from([
            "hipfire-eval",
            "--model",
            "m.hfq",
            "--battery",
            "barrage",
            "--suite",
            "gpqa",
            "--fetch-datasets",
            "--dataset-cache",
            cache.to_str().unwrap(),
        ])
        .unwrap();

        let datasets = resolve_datasets(&cfg).unwrap();
        assert_eq!(datasets.len(), 1);
        let dataset = &datasets[0];
        assert_eq!(dataset.suite, SuiteId::Gpqa);
        assert_eq!(dataset.source, "local_mirror");
        assert_eq!(dataset.repo_id.as_deref(), Some("idavidrein/gpqa"));
        assert_eq!(dataset.revision.as_deref(), Some("main"));
        assert_eq!(dataset.status, EvalStatus::Pass);
        assert!(dataset
            .digest
            .as_deref()
            .unwrap_or("")
            .starts_with("fnv64:"));
        assert!(dataset
            .files
            .iter()
            .any(|p| p.ends_with("gpqa_diamond.csv")));

        let (rel, row_count) = write_gpqa_prompt_artifact(&out, &cfg, &datasets)
            .unwrap()
            .unwrap();
        assert_eq!(rel, "artifacts/gpqa_prompts.jsonl");
        assert_eq!(row_count, dataset.selected_item_ids.len());
        let body = fs::read_to_string(out.join("gpqa_prompts.jsonl")).unwrap();
        assert!(body.contains("\"status\":\"pass\""));
        assert!(body.contains("\"prompt_format\":\"gpqa_zero_shot_v1\""));
        let first: Value = serde_json::from_str(body.lines().next().unwrap()).unwrap();
        assert_eq!(first["schema"], 1);
        assert_eq!(first["dataset_source"], "local_mirror");
        assert_eq!(first["dataset_repo_id"], "idavidrein/gpqa");
        assert_eq!(first["dataset_revision"], "main");
        assert!(first["dataset_digest"]
            .as_str()
            .unwrap_or("")
            .starts_with("fnv64:"));
        assert!(first["dataset_license"].is_null());

        let _ = fs::remove_dir_all(mirror);
        let _ = fs::remove_dir_all(cache);
        let _ = fs::remove_dir_all(out);
    }

    #[test]
    fn examples_executor_materializes_gpqa_rows_before_model_skip() {
        let cache = temp_path("gpqa-examples-skip-cache");
        write_gpqa_csv(&cache);
        let cfg = parse_args_from([
            "hipfire-eval",
            "--model",
            "missing-local-model.hfq",
            "--battery",
            "barrage",
            "--suite",
            "gpqa",
            "--executor",
            "examples",
        ])
        .unwrap();
        let ctx = EvalContext {
            commit_sha: None,
            git_branch: None,
            git_describe: None,
            git_dirty: None,
            binary_hash: None,
            arch: None,
            rocm: None,
            host_profile: test_host_profile(),
        };
        let dataset = DatasetManifestEntry {
            suite: SuiteId::Gpqa,
            source: "local_cache".to_string(),
            repo_id: Some("idavidrein/gpqa".to_string()),
            revision: Some("main".to_string()),
            files: list_files(&cache),
            digest: directory_hash(&cache),
            license: None,
            cache_path: cache.display().to_string(),
            selected_item_ids: vec!["gpqa_diamond:0".to_string()],
            status: EvalStatus::Pass,
            reason: None,
        };

        let rows = examples_battery_rows(BatteryId::Barrage, &cfg, &ctx, &[dataset]).unwrap();
        assert_eq!(rows.len(), 1);
        let row = &rows[0];
        assert_eq!(row.battery, BatteryId::Barrage);
        assert_eq!(row.suite, Some(SuiteId::Gpqa));
        assert_eq!(row.case_id, "gpqa_zero_shot_native");
        assert_eq!(row.status, EvalStatus::Skip);
        assert_eq!(row.dataset_source.as_deref(), Some("local_cache"));
        assert_eq!(row.dataset_repo_id.as_deref(), Some("idavidrein/gpqa"));
        assert_eq!(row.dataset_revision.as_deref(), Some("main"));
        assert_eq!(
            row.dataset_cache_path.as_deref(),
            Some(cache.to_str().unwrap())
        );
        assert!(row
            .dataset_digest
            .as_deref()
            .unwrap_or("")
            .starts_with("fnv64:"));
        assert!(row
            .prompt_hash
            .as_deref()
            .unwrap_or("")
            .starts_with("fnv64:"));
        assert_eq!(row.metrics["executor"], json!("examples"));
        assert_eq!(row.metrics["prompt_format"], json!("gpqa_zero_shot_v1"));
        assert!(row.metrics.get("answer_hash").is_some());

        let _ = fs::remove_dir_all(cache);
    }

    #[test]
    fn examples_humaneval_rows_include_configured_comparators() {
        let cache = temp_path("humaneval-examples-comparators-cache");
        write_humaneval_jsonl(&cache);
        let cfg = parse_args_from([
            "hipfire-eval",
            "--model",
            "candidate.hfq",
            "--baseline",
            "baseline.hfq",
            "--reference",
            "reference.hfq",
            "--battery",
            "barrage",
            "--suite",
            "humaneval",
            "--executor",
            "examples",
        ])
        .unwrap();
        let ctx = EvalContext {
            commit_sha: None,
            git_branch: None,
            git_describe: None,
            git_dirty: None,
            binary_hash: None,
            arch: None,
            rocm: None,
            host_profile: test_host_profile(),
        };
        let dataset = DatasetManifestEntry {
            suite: SuiteId::HumanEval,
            source: "local_cache".to_string(),
            repo_id: Some("openai_humaneval".to_string()),
            revision: None,
            files: list_files(&cache),
            digest: directory_hash(&cache),
            license: None,
            cache_path: cache.display().to_string(),
            selected_item_ids: vec!["HumanEval/0".to_string()],
            status: EvalStatus::Pass,
            reason: None,
        };

        let rows = examples_battery_rows(BatteryId::Barrage, &cfg, &ctx, &[dataset]).unwrap();
        assert_eq!(rows.len(), 3);
        let models: Vec<_> = rows.iter().map(|row| row.model.as_str()).collect();
        assert_eq!(
            models,
            vec!["candidate.hfq", "baseline.hfq", "reference.hfq"]
        );
        assert!(rows.iter().all(|row| row.suite == Some(SuiteId::HumanEval)));
        assert!(rows
            .iter()
            .all(|row| row.case_id == "humaneval_completion_native"));
        assert!(rows.iter().all(|row| row.status == EvalStatus::Skip));
        assert!(rows.iter().all(|row| row
            .prompt_hash
            .as_deref()
            .unwrap_or("")
            .starts_with("fnv64:")));
        assert!(rows
            .iter()
            .all(|row| row.metrics["prompt_format"] == "humaneval_completion_v1"));

        let _ = fs::remove_dir_all(cache);
    }

    #[test]
    fn examples_builtin_software_eval_rows_include_configured_comparators() {
        let cfg = parse_args_from([
            "hipfire-eval",
            "--model",
            "candidate.hfq",
            "--baseline",
            "baseline.hfq",
            "--reference",
            "reference.hfq",
            "--battery",
            "barrage",
            "--suite",
            "deep_swe,swe_bench",
            "--executor",
            "examples",
            "--offline",
        ])
        .unwrap();
        let ctx = EvalContext {
            commit_sha: None,
            git_branch: None,
            git_describe: None,
            git_dirty: None,
            binary_hash: None,
            arch: None,
            rocm: None,
            host_profile: test_host_profile(),
        };
        let datasets = resolve_datasets(&cfg).unwrap();

        let rows = examples_battery_rows(BatteryId::Barrage, &cfg, &ctx, &datasets).unwrap();
        assert_eq!(rows.len(), 6);
        assert!(rows
            .iter()
            .all(|row| row.case_id == "builtin_software_eval_native"));
        assert!(rows.iter().all(|row| row.status == EvalStatus::Skip));
        assert!(rows.iter().all(|row| row
            .prompt_hash
            .as_deref()
            .unwrap_or("")
            .starts_with("fnv64:")));
        assert!(rows.iter().all(|row| row.metrics["executor"] == "examples"));
        assert!(rows.iter().any(|row| row.suite == Some(SuiteId::DeepSwe)));
        assert!(rows.iter().any(|row| row.suite == Some(SuiteId::SweBench)));
        let models: Vec<_> = rows.iter().map(|row| row.model.as_str()).collect();
        assert_eq!(
            models,
            vec![
                "candidate.hfq",
                "baseline.hfq",
                "reference.hfq",
                "candidate.hfq",
                "baseline.hfq",
                "reference.hfq",
            ]
        );
        assert!(rows
            .iter()
            .all(|row| row.metrics.get("answer_hash").is_some()));
        assert!(rows
            .iter()
            .all(|row| row.dataset_source.as_deref() == Some("builtin")));
    }

    #[test]
    fn extracts_first_gpqa_answer_letter_from_generation() {
        assert_eq!(extract_answer_letter(" C\n"), Some("C".to_string()));
        assert_eq!(
            extract_answer_letter("The answer is b."),
            Some("B".to_string())
        );
        assert_eq!(extract_answer_letter("123"), None);
    }

    #[test]
    fn barrage_records_skip_when_gpqa_materialization_fails() {
        let cache = temp_path("gpqa-malformed-cache");
        write_malformed_gpqa_csv(&cache);
        let cfg = parse_args_from([
            "hipfire-eval",
            "--model",
            "candidate.hfq",
            "--battery",
            "barrage",
            "--suite",
            "gpqa",
        ])
        .unwrap();
        let ctx = EvalContext {
            commit_sha: None,
            git_branch: None,
            git_describe: None,
            git_dirty: None,
            binary_hash: None,
            arch: None,
            rocm: None,
            host_profile: test_host_profile(),
        };
        let dataset = DatasetManifestEntry {
            suite: SuiteId::Gpqa,
            source: "local_cache".to_string(),
            repo_id: Some("idavidrein/gpqa".to_string()),
            revision: None,
            files: list_files(&cache),
            digest: directory_hash(&cache),
            license: None,
            cache_path: cache.display().to_string(),
            selected_item_ids: vec!["gpqa_diamond:0".to_string()],
            status: EvalStatus::Pass,
            reason: None,
        };

        let rows = barrage_rows(&cfg, &ctx, &[dataset]);
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].case_id, "gpqa_materialize_failed");
        assert_eq!(rows[0].status, EvalStatus::Skip);
        assert_eq!(rows[0].dataset_item_id.as_deref(), Some("gpqa_diamond:0"));
        assert!(rows[0]
            .reason
            .as_deref()
            .unwrap_or("")
            .contains("GPQA CSV missing header"));

        let _ = fs::remove_dir_all(cache);
    }

    #[test]
    fn deep_swe_and_swe_bench_are_builtin_native_barrage_canaries() {
        let cfg = parse_args_from([
            "hipfire-eval",
            "--model",
            "candidate.hfq",
            "--battery",
            "barrage",
            "--suite",
            "deep_swe,swe_bench",
            "--offline",
        ])
        .unwrap();
        let ctx = EvalContext {
            commit_sha: None,
            git_branch: None,
            git_describe: None,
            git_dirty: None,
            binary_hash: None,
            arch: None,
            rocm: None,
            host_profile: test_host_profile(),
        };
        let datasets = resolve_datasets(&cfg).unwrap();
        assert_eq!(datasets.len(), 2);
        assert!(datasets.iter().all(|d| d.source == "builtin"));
        assert!(datasets.iter().all(|d| d.status == EvalStatus::Pass));
        assert!(datasets
            .iter()
            .all(|d| d.digest.as_deref().unwrap_or("").starts_with("fnv64:")));

        let rows = barrage_rows(&cfg, &ctx, &datasets);
        assert_eq!(rows.len(), 2);
        assert!(rows.iter().all(|row| row.battery == BatteryId::Barrage));
        assert!(rows
            .iter()
            .all(|row| row.case_id == "builtin_software_eval_native"));
        assert!(rows.iter().all(|row| row.status == EvalStatus::Skip));
        assert!(rows.iter().all(|row| row
            .prompt_hash
            .as_deref()
            .unwrap_or("")
            .starts_with("fnv64:")));
        assert!(rows
            .iter()
            .all(|row| row.dataset_source.as_deref() == Some("builtin")));
        assert!(rows
            .iter()
            .all(|row| row.dataset_revision.as_deref() == Some("hipfire-native-v1")));
        assert!(rows
            .iter()
            .all(|row| row.metrics.get("answer_hash").is_some()));
        assert!(rows.iter().any(|row| {
            row.suite == Some(SuiteId::DeepSwe)
                && row.dataset_item_id.as_deref() == Some("deep_swe_verified:0")
                && row.metrics["prompt_format"] == "deep_swe_micro_zero_shot_v1"
        }));
        assert!(rows.iter().any(|row| {
            row.suite == Some(SuiteId::SweBench)
                && row.dataset_item_id.as_deref() == Some("swe_bench_lite:0")
                && row.metrics["prompt_format"] == "swe_bench_micro_zero_shot_v1"
        }));
    }

    #[test]
    fn humaneval_cache_materializes_native_barrage_prompts() {
        let cache = temp_path("humaneval-cache");
        write_humaneval_jsonl(&cache);
        let cfg = parse_args_from([
            "hipfire-eval",
            "--model",
            "candidate.hfq",
            "--battery",
            "barrage",
            "--suite",
            "humaneval",
        ])
        .unwrap();
        let ctx = EvalContext {
            commit_sha: None,
            git_branch: None,
            git_describe: None,
            git_dirty: None,
            binary_hash: None,
            arch: None,
            rocm: None,
            host_profile: test_host_profile(),
        };
        let dataset = DatasetManifestEntry {
            suite: SuiteId::HumanEval,
            source: "local_cache".to_string(),
            repo_id: None,
            revision: None,
            files: list_files(&cache),
            digest: directory_hash(&cache),
            license: None,
            cache_path: cache.display().to_string(),
            selected_item_ids: selected_item_ids(SuiteId::HumanEval),
            status: EvalStatus::Pass,
            reason: None,
        };

        let rows = barrage_rows(&cfg, &ctx, &[dataset]);
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0].suite, Some(SuiteId::HumanEval));
        assert_eq!(rows[0].case_id, "humaneval_completion_native");
        assert_eq!(rows[0].dataset_item_id.as_deref(), Some("HumanEval/0"));
        assert_eq!(rows[0].status, EvalStatus::Skip);
        assert!(rows[0]
            .prompt_hash
            .as_deref()
            .unwrap_or("")
            .starts_with("fnv64:"));
        assert_eq!(rows[0].metrics["prompt_format"], "humaneval_completion_v1");
        assert_eq!(rows[0].metrics["scoring_mode"], "execution_only");
        assert!(rows[0].metrics.get("canonical_solution_hash").is_some());
        assert!(rows[0].metrics.get("test_hash").is_some());

        let _ = fs::remove_dir_all(cache);
    }

    #[test]
    fn evidence_artifacts_include_barrage_prompt_ledger() {
        let cache = temp_path("humaneval-prompt-ledger-cache");
        let out = temp_path("humaneval-prompt-ledger-out");
        write_humaneval_jsonl(&cache);
        fs::create_dir_all(&out).unwrap();
        let cfg = parse_args_from([
            "hipfire-eval",
            "--model",
            "candidate.hfq",
            "--battery",
            "barrage",
            "--suite",
            "humaneval",
        ])
        .unwrap();
        let ctx = EvalContext {
            commit_sha: None,
            git_branch: None,
            git_describe: None,
            git_dirty: None,
            binary_hash: None,
            arch: None,
            rocm: None,
            host_profile: test_host_profile(),
        };
        let dataset = DatasetManifestEntry {
            suite: SuiteId::HumanEval,
            source: "local_cache".to_string(),
            repo_id: None,
            revision: None,
            files: list_files(&cache),
            digest: directory_hash(&cache),
            license: None,
            cache_path: cache.display().to_string(),
            selected_item_ids: selected_item_ids(SuiteId::HumanEval),
            status: EvalStatus::Pass,
            reason: None,
        };
        let rows = barrage_rows(&cfg, &ctx, std::slice::from_ref(&dataset));
        let comparison = build_comparison_artifact(&cfg, &rows, &ctx);
        let admission = build_admission_artifact(&cfg, &rows, &comparison, &ctx);

        let artifacts =
            write_evidence_artifacts(&out, &cfg, &[dataset], &rows, &comparison, &admission, &ctx)
                .unwrap();
        assert_eq!(
            artifacts
                .get("barrage_prompts")
                .and_then(|v| v.get("path"))
                .and_then(Value::as_str),
            Some("artifacts/barrage_prompts.jsonl")
        );
        assert_eq!(
            artifacts
                .get("barrage_prompts")
                .and_then(|v| v.get("row_count"))
                .and_then(Value::as_u64),
            Some(2)
        );
        let body = fs::read_to_string(out.join("barrage_prompts.jsonl")).unwrap();
        assert!(body.contains("\"suite\":\"humaneval\""));
        assert!(body.contains("\"prompt_hash\":\"fnv64:"));
        assert!(body.contains("\"prompt_format\":\"humaneval_completion_v1\""));
        assert!(body.contains("\"canonical_solution_hash\":\"fnv64:"));
        assert!(body.contains("\"test_hash\":\"fnv64:"));
        let first: Value = serde_json::from_str(body.lines().next().unwrap()).unwrap();
        assert_eq!(first["dataset_source"], "local_cache");
        assert!(first["dataset_digest"]
            .as_str()
            .unwrap_or("")
            .starts_with("fnv64:"));
        assert_eq!(first["dataset_cache_path"], cache.display().to_string());

        let _ = fs::remove_dir_all(cache);
        let _ = fs::remove_dir_all(out);
    }

    #[test]
    fn barrage_prompt_ledger_includes_lm_eval_micro_builtin_rows() {
        let out = temp_path("lm-eval-micro-prompt-ledger-out");
        fs::create_dir_all(&out).unwrap();
        let cfg = parse_args_from([
            "hipfire-eval",
            "--model",
            "candidate.hfq",
            "--battery",
            "barrage",
            "--suite",
            "lm_eval_micro",
            "--offline",
        ])
        .unwrap();
        let ctx = EvalContext {
            commit_sha: None,
            git_branch: None,
            git_describe: None,
            git_dirty: None,
            binary_hash: None,
            arch: None,
            rocm: None,
            host_profile: test_host_profile(),
        };
        let datasets = resolve_datasets(&cfg).unwrap();
        let rows = barrage_rows(&cfg, &ctx, &datasets);
        let comparison = build_comparison_artifact(&cfg, &rows, &ctx);
        let admission = build_admission_artifact(&cfg, &rows, &comparison, &ctx);

        let artifacts =
            write_evidence_artifacts(&out, &cfg, &datasets, &rows, &comparison, &admission, &ctx)
                .unwrap();
        assert_eq!(
            artifacts
                .get("barrage_prompts")
                .and_then(|v| v.get("row_count"))
                .and_then(Value::as_u64),
            Some(3)
        );
        let body = fs::read_to_string(out.join("barrage_prompts.jsonl")).unwrap();
        assert!(body.contains("\"suite\":\"lm_eval_micro\""));
        assert!(body.contains("\"prompt_format\":\"lm_eval_micro_zero_shot_v1\""));
        assert!(body.contains("\"answer_hash\":\"fnv64:"));
        assert!(body.contains("\"task\":\"arc_easy\""));
        let first: Value = serde_json::from_str(body.lines().next().unwrap()).unwrap();
        assert_eq!(first["dataset_source"], "builtin");
        assert_eq!(first["dataset_revision"], "hipfire-native-v1");
        assert_eq!(first["dataset_license"], "hipfire-native");
        assert!(first["dataset_digest"]
            .as_str()
            .unwrap_or("")
            .starts_with("fnv64:"));

        let _ = fs::remove_dir_all(out);
    }

    #[test]
    fn barrage_prompt_ledger_includes_builtin_software_eval_rows() {
        let out = temp_path("software-eval-prompt-ledger-out");
        fs::create_dir_all(&out).unwrap();
        let cfg = parse_args_from([
            "hipfire-eval",
            "--model",
            "candidate.hfq",
            "--battery",
            "barrage",
            "--suite",
            "deep_swe,swe_bench",
            "--offline",
        ])
        .unwrap();
        let ctx = EvalContext {
            commit_sha: None,
            git_branch: None,
            git_describe: None,
            git_dirty: None,
            binary_hash: None,
            arch: None,
            rocm: None,
            host_profile: test_host_profile(),
        };
        let datasets = resolve_datasets(&cfg).unwrap();
        let rows = barrage_rows(&cfg, &ctx, &datasets);
        let comparison = build_comparison_artifact(&cfg, &rows, &ctx);
        let admission = build_admission_artifact(&cfg, &rows, &comparison, &ctx);

        let artifacts =
            write_evidence_artifacts(&out, &cfg, &datasets, &rows, &comparison, &admission, &ctx)
                .unwrap();
        assert_eq!(
            artifacts
                .get("barrage_prompts")
                .and_then(|v| v.get("row_count"))
                .and_then(Value::as_u64),
            Some(2)
        );
        let body = fs::read_to_string(out.join("barrage_prompts.jsonl")).unwrap();
        assert!(body.contains("\"suite\":\"deep_swe\""));
        assert!(body.contains("\"suite\":\"swe_bench\""));
        assert!(body.contains("\"prompt_format\":\"deep_swe_micro_zero_shot_v1\""));
        assert!(body.contains("\"prompt_format\":\"swe_bench_micro_zero_shot_v1\""));
        assert!(body.contains("\"answer_hash\":\"fnv64:"));
        assert!(body.contains("\"scoring_mode\":\"exact_letter\""));
        let first: Value = serde_json::from_str(body.lines().next().unwrap()).unwrap();
        assert_eq!(first["dataset_source"], "builtin");
        assert_eq!(first["dataset_revision"], "hipfire-native-v1");
        assert_eq!(first["dataset_license"], "hipfire-native");
        assert!(first["dataset_digest"]
            .as_str()
            .unwrap_or("")
            .starts_with("fnv64:"));

        let _ = fs::remove_dir_all(out);
    }

    #[test]
    fn writes_gpqa_prompt_artifact_for_available_dataset() {
        let cache = temp_path("gpqa-artifact-cache");
        let out = temp_path("gpqa-artifact-out");
        write_gpqa_csv(&cache);
        fs::create_dir_all(&out).unwrap();

        let cfg = parse_args_from([
            "hipfire-eval",
            "--model",
            "m.hfq",
            "--battery",
            "barrage",
            "--suite",
            "gpqa",
            "--dataset-cache",
            cache.parent().unwrap().to_str().unwrap(),
        ])
        .unwrap();
        let dataset = DatasetManifestEntry {
            suite: SuiteId::Gpqa,
            source: "local_cache".to_string(),
            repo_id: Some("idavidrein/gpqa".to_string()),
            revision: None,
            files: list_files(&cache),
            digest: directory_hash(&cache),
            license: None,
            cache_path: cache.display().to_string(),
            selected_item_ids: vec!["gpqa_diamond:0".to_string()],
            status: EvalStatus::Pass,
            reason: None,
        };
        let (rel, row_count) = write_gpqa_prompt_artifact(&out, &cfg, &[dataset])
            .unwrap()
            .unwrap();
        assert_eq!(rel, "artifacts/gpqa_prompts.jsonl");
        assert_eq!(row_count, 1);
        let body = fs::read_to_string(out.join("gpqa_prompts.jsonl")).unwrap();
        assert!(body.contains("\"suite\":\"gpqa\""));
        assert!(body.contains("\"prompt_hash\":\"fnv64:"));
        assert!(body.contains("\"answer_label\":"));
        let first: Value = serde_json::from_str(body.lines().next().unwrap()).unwrap();
        assert_eq!(first["schema"], 1);
        assert_eq!(first["dataset_source"], "local_cache");
        assert_eq!(first["dataset_repo_id"], "idavidrein/gpqa");
        assert!(first["dataset_digest"]
            .as_str()
            .unwrap_or("")
            .starts_with("fnv64:"));

        let _ = fs::remove_dir_all(cache);
        let _ = fs::remove_dir_all(out);
    }

    #[test]
    fn gpqa_prompt_artifact_records_skip_for_malformed_cache() {
        let cache = temp_path("gpqa-artifact-malformed-cache");
        let out = temp_path("gpqa-artifact-malformed-out");
        write_malformed_gpqa_csv(&cache);
        fs::create_dir_all(&out).unwrap();
        let cfg = parse_args_from([
            "hipfire-eval",
            "--model",
            "m.hfq",
            "--battery",
            "barrage",
            "--suite",
            "gpqa",
        ])
        .unwrap();
        let dataset = DatasetManifestEntry {
            suite: SuiteId::Gpqa,
            source: "local_cache".to_string(),
            repo_id: Some("idavidrein/gpqa".to_string()),
            revision: None,
            files: list_files(&cache),
            digest: directory_hash(&cache),
            license: None,
            cache_path: cache.display().to_string(),
            selected_item_ids: vec!["gpqa_diamond:0".to_string()],
            status: EvalStatus::Pass,
            reason: None,
        };

        let (rel, row_count) = write_gpqa_prompt_artifact(&out, &cfg, &[dataset])
            .unwrap()
            .unwrap();
        assert_eq!(rel, "artifacts/gpqa_prompts.jsonl");
        assert_eq!(row_count, 1);
        let body = fs::read_to_string(out.join("gpqa_prompts.jsonl")).unwrap();
        assert!(body.contains("\"status\":\"skip\""));
        assert!(body.contains("GPQA CSV missing header"));
        let first: Value = serde_json::from_str(body.lines().next().unwrap()).unwrap();
        assert_eq!(first["dataset_source"], "local_cache");
        assert_eq!(first["dataset_repo_id"], "idavidrein/gpqa");
        assert!(first["dataset_digest"]
            .as_str()
            .unwrap_or("")
            .starts_with("fnv64:"));

        let _ = fs::remove_dir_all(cache);
        let _ = fs::remove_dir_all(out);
    }

    #[test]
    fn writes_schemaed_evidence_artifacts() {
        let out = temp_path("evidence-artifacts");
        fs::create_dir_all(&out).unwrap();
        let cfg = parse_args_from([
            "hipfire-eval",
            "--model",
            "candidate.hfq",
            "--battery",
            "quality,speed",
            "--profile",
            "off",
        ])
        .unwrap();
        let ctx = EvalContext {
            commit_sha: Some("abc123".to_string()),
            git_branch: Some("evaluation-harness".to_string()),
            git_describe: Some("v0.2.0-1-gabc123".to_string()),
            git_dirty: Some(true),
            binary_hash: Some("binhash".to_string()),
            arch: Some("gfx1151".to_string()),
            rocm: Some("7.13.26176".to_string()),
            host_profile: test_host_profile(),
        };
        let comparison = ComparisonArtifact {
            schema: 1,
            provenance: run_provenance(&ctx),
            status: EvalStatus::Skip,
            reason: Some("no --compare or --reference provided".to_string()),
            baseline: None,
            reference: None,
            cases: Vec::new(),
        };

        let rows = Vec::new();
        let admission = build_admission_artifact(&cfg, &rows, &comparison, &ctx);
        let artifacts =
            write_evidence_artifacts(&out, &cfg, &[], &rows, &comparison, &admission, &ctx)
                .unwrap();
        assert_eq!(
            artifacts
                .get("quality")
                .and_then(|v| v.get("path"))
                .and_then(Value::as_str),
            Some("artifacts/quality.json")
        );
        assert_eq!(
            artifacts
                .get("quality")
                .and_then(|v| v.get("row_count"))
                .and_then(Value::as_u64),
            Some(0)
        );
        assert_eq!(
            artifacts
                .get("quality")
                .and_then(|v| v.get("kind"))
                .and_then(Value::as_str),
            Some("quality")
        );
        assert!(artifacts
            .get("quality")
            .and_then(|v| v.get("runner_version"))
            .and_then(Value::as_str)
            .is_some());
        assert_eq!(
            artifacts
                .get("quality")
                .and_then(|v| v.get("git_commit"))
                .and_then(Value::as_str),
            Some("abc123")
        );
        assert!(artifacts
            .get("quality")
            .and_then(|v| v.get("reason"))
            .and_then(Value::as_str)
            .unwrap()
            .contains("model-backed collection"));
        assert!(artifacts
            .get("quality")
            .and_then(|v| v.get("expected_metrics"))
            .and_then(Value::as_array)
            .unwrap()
            .iter()
            .any(|v| v == "mean_kld"));
        assert_eq!(
            artifacts
                .get("performance")
                .and_then(|v| v.get("path"))
                .and_then(Value::as_str),
            Some("artifacts/performance.json")
        );
        assert_eq!(
            artifacts
                .get("admission")
                .and_then(|v| v.get("path"))
                .and_then(Value::as_str),
            Some("artifacts/admission.json")
        );
        let quality: Value =
            serde_json::from_str(&fs::read_to_string(out.join("quality.json")).unwrap()).unwrap();
        assert_eq!(quality["status"], "not_collected");
        assert_eq!(quality["provenance"]["runner"], "hipfire-eval");
        assert!(quality["provenance"]["runner_version"].as_str().is_some());
        assert_eq!(quality["provenance"]["git_commit"], "abc123");
        assert_eq!(quality["provenance"]["git_branch"], "evaluation-harness");
        assert_eq!(quality["provenance"]["binary_hash"], "binhash");
        assert_eq!(quality["collection"]["source"], "hipfire-eval");
        assert!(quality["expected_metrics"]
            .as_array()
            .unwrap()
            .iter()
            .any(|v| v == "mean_kld"));
        let comparison: Value =
            serde_json::from_str(&fs::read_to_string(out.join("comparisons.json")).unwrap())
                .unwrap();
        assert!(comparison["provenance"]["runner_version"]
            .as_str()
            .is_some());
        assert_eq!(comparison["provenance"]["git_commit"], "abc123");
        let profiling: Value =
            serde_json::from_str(&fs::read_to_string(out.join("profiling.json")).unwrap()).unwrap();
        assert_eq!(profiling["status"], "disabled");
        let admission: Value =
            serde_json::from_str(&fs::read_to_string(out.join("admission.json")).unwrap()).unwrap();
        assert_eq!(admission["verdict"], "incomplete");
        assert!(admission["provenance"]["runner_version"].as_str().is_some());
        assert_eq!(admission["provenance"]["git_commit"], "abc123");

        let _ = fs::remove_dir_all(out);
    }

    #[test]
    fn passive_profiling_artifact_records_requested_status() {
        let out = temp_path("passive-profiling-artifacts");
        fs::create_dir_all(&out).unwrap();
        let cfg = parse_args_from([
            "hipfire-eval",
            "--model",
            "candidate.hfq",
            "--battery",
            "speed",
            "--profile",
            "passive",
        ])
        .unwrap();
        let ctx = EvalContext {
            commit_sha: None,
            git_branch: None,
            git_describe: None,
            git_dirty: None,
            binary_hash: None,
            arch: None,
            rocm: None,
            host_profile: test_host_profile(),
        };
        let comparison = ComparisonArtifact {
            schema: 1,
            provenance: run_provenance(&ctx),
            status: EvalStatus::Skip,
            reason: Some("no --compare or --reference provided".to_string()),
            baseline: None,
            reference: None,
            cases: Vec::new(),
        };
        let rows = Vec::new();
        let admission = build_admission_artifact(&cfg, &rows, &comparison, &ctx);
        let artifacts =
            write_evidence_artifacts(&out, &cfg, &[], &rows, &comparison, &admission, &ctx)
                .unwrap();

        assert_eq!(
            artifacts
                .get("profiling")
                .and_then(|v| v.get("status"))
                .and_then(Value::as_str),
            Some("requested")
        );
        let profiling: Value =
            serde_json::from_str(&fs::read_to_string(out.join("profiling.json")).unwrap()).unwrap();
        assert_eq!(profiling["status"], "requested");
        assert_eq!(profiling["collection"]["profiling_mode"], "passive");
        assert!(profiling["reason"]
            .as_str()
            .unwrap_or("")
            .contains("passive profiling requested"));

        let _ = fs::remove_dir_all(out);
    }

    #[test]
    fn summary_includes_admission_verdict_and_findings() {
        let out = temp_path("summary-admission.md");
        let cfg = parse_args_from([
            "hipfire-eval",
            "--model",
            "candidate.hfq",
            "--baseline",
            "baseline.hfq",
            "--battery",
            "quality,speed",
        ])
        .unwrap();
        let ctx = EvalContext {
            commit_sha: Some("abc".to_string()),
            git_branch: Some("evaluation-harness".to_string()),
            git_describe: Some("v0.2.0-1-gabc".to_string()),
            git_dirty: Some(true),
            binary_hash: Some("binhash".to_string()),
            arch: Some("gfx1151".to_string()),
            rocm: Some("7.13.26176".to_string()),
            host_profile: test_host_profile(),
        };
        let rows = vec![
            row_for_model(
                BatteryId::Quality,
                None,
                "canary",
                None,
                EvalStatus::Pass,
                None,
                BTreeMap::from([("mean_kld".to_string(), json!(0.08))]),
                &cfg,
                &ctx,
                prompt("benchmarks/quality-baselines/harness/canary.md"),
                0,
                "candidate.hfq".to_string(),
            ),
            row_for_model(
                BatteryId::Quality,
                None,
                "canary",
                None,
                EvalStatus::Pass,
                None,
                BTreeMap::from([("mean_kld".to_string(), json!(0.05))]),
                &cfg,
                &ctx,
                None,
                0,
                "baseline.hfq".to_string(),
            ),
            row_for_model(
                BatteryId::Speed,
                None,
                "decode",
                None,
                EvalStatus::Pass,
                None,
                BTreeMap::from([("tok_s".to_string(), json!(120.0))]),
                &cfg,
                &ctx,
                None,
                0,
                "candidate.hfq".to_string(),
            ),
            row_for_model(
                BatteryId::Speed,
                None,
                "decode",
                None,
                EvalStatus::Pass,
                None,
                BTreeMap::from([("tok_s".to_string(), json!(100.0))]),
                &cfg,
                &ctx,
                None,
                0,
                "baseline.hfq".to_string(),
            ),
        ];
        let comparison = build_comparison_artifact(&cfg, &rows, &ctx);
        let admission = build_admission_artifact(&cfg, &rows, &comparison, &ctx);
        let stdout = render_eval_stdout_findings(&admission, &rows);
        assert!(stdout.contains("admission: reject"));
        assert!(stdout.contains("results: 4"));
        assert!(
            stdout.contains("  [pass] speed/decode suite=- item=- model=candidate.hfq tok_s=120")
        );
        assert!(
            stdout.contains("  [pass] speed/decode suite=- item=- model=baseline.hfq tok_s=100")
        );
        assert!(stdout.contains("findings: 1"));
        assert!(stdout.contains("admission findings: 1"));
        assert!(stdout.contains("row findings: 0"));
        assert!(stdout.contains(
            "[reject] quality/canary suite=- item=- metric=mean_kld comparator=baseline direction=regressed delta=+0.030000 relative=+60.00%"
        ));
        let datasets = vec![DatasetManifestEntry {
            suite: SuiteId::LmEvalMicro,
            source: "builtin".to_string(),
            repo_id: None,
            revision: Some("hipfire-native-v1".to_string()),
            files: vec!["builtin:lm_eval_micro:v1".to_string()],
            digest: Some("fnv64:datasetdigest".to_string()),
            license: Some("hipfire-native".to_string()),
            cache_path: "builtin:lm_eval_micro".to_string(),
            selected_item_ids: vec!["arc_easy:0".to_string()],
            status: EvalStatus::Pass,
            reason: None,
        }];
        let artifacts = BTreeMap::from([
            (
                "admission".to_string(),
                json!({
                    "path": "artifacts/admission.json",
                    "status": "fail",
                    "verdict": "reject",
                    "finding_count": 1,
                }),
            ),
            (
                "launch_counts".to_string(),
                json!({"path": "artifacts/launch_counts.json", "status": "collected"}),
            ),
            (
                "moe_router_histogram".to_string(),
                json!({"path": "artifacts/moe_router_histogram.json", "status": "not_collected"}),
            ),
            (
                "profiling".to_string(),
                json!({"path": "artifacts/profiling.json", "status": "disabled"}),
            ),
            (
                "run_metadata".to_string(),
                json!({
                    "path": "artifacts/run_metadata.json",
                    "status": "collected",
                    "hipfire_version": "0.2.0",
                    "git_commit": "abc",
                    "git_describe": "v0.2.0-1-gabc",
                }),
            ),
        ]);

        write_summary(
            &out,
            &cfg,
            &datasets,
            &comparison,
            &admission,
            &artifacts,
            &rows,
            &ctx,
        )
        .unwrap();
        let body = fs::read_to_string(&out).unwrap();
        assert!(body.contains("- model hash: `tag:"));
        assert!(body.contains("- baseline hash: `tag:"));
        assert!(body.contains("- tier target: `60` seconds"));
        assert!(body.contains("- CI suitable: `true`"));
        assert!(body.contains("- hipfire version: `"));
        assert!(body.contains("- runner: `hipfire-eval "));
        assert!(body.contains("- git commit: `abc`"));
        assert!(body.contains("- git branch: `evaluation-harness`"));
        assert!(body.contains("- git describe: `v0.2.0-1-gabc`"));
        assert!(body.contains("- git dirty: `true`"));
        assert!(body.contains("- binary hash: `binhash`"));
        assert!(body.contains("- arch: `gfx1151`"));
        assert!(body.contains("- ROCm: `7.13.26176`"));
        assert!(body.contains("## Models"));
        assert!(body.contains(
            "| role | identifier | exists | file hash | tag hash | metadata | quantization hash |"
        ));
        assert!(body.contains("| candidate | candidate.hfq | false |  | tag:"));
        assert!(body.contains("| baseline | baseline.hfq | false |  | tag:"));
        assert!(body.contains("## Datasets"));
        assert!(body.contains(
            "| lm_eval_micro | Pass | builtin |  | hipfire-native-v1 | fnv64:datasetdigest | hipfire-native | 1 | arc_easy:0 | builtin:lm_eval_micro |  |"
        ));
        assert!(body.contains("## Admission"));
        assert!(body.contains("- verdict: `reject`"));
        assert!(body.contains("| reject | quality | canary | mean_kld | baseline | regressed |"));
        assert!(body.contains("## Evidence Artifacts"));
        assert!(body.contains("| launch_counts | collected | artifacts/launch_counts.json |"));
        assert!(body.contains("| profiling | disabled | artifacts/profiling.json |"));
        assert!(body.contains(
            "| moe_router_histogram | not_collected | artifacts/moe_router_histogram.json |"
        ));
        assert!(body.contains("| run_metadata | collected | artifacts/run_metadata.json |"));
        assert!(body.contains("### Observed Evidence"));
        assert!(body.contains("| profiling | Skip | 0 | profiling disabled by --profile off |"));
        assert!(body.contains("## Rows"));
        assert!(body.contains(
            "| battery | suite | case | item | model | model hash | prompt hash | status | reason |"
        ));
        assert!(body.contains("| quality |  | canary |  | candidate.hfq | tag:"));
        assert!(body.contains("fnv64:"));

        let _ = fs::remove_file(out);
    }

    #[test]
    fn stdout_findings_include_row_skips() {
        let cfg = parse_args_from([
            "hipfire-eval",
            "--model",
            "candidate.hfq",
            "--battery",
            "speed",
        ])
        .unwrap();
        let ctx = EvalContext {
            commit_sha: Some("abc".to_string()),
            git_branch: Some("evaluation-harness".to_string()),
            git_describe: Some("v0.2.0-1-gabc".to_string()),
            git_dirty: Some(false),
            binary_hash: Some("binhash".to_string()),
            arch: Some("gfx1151".to_string()),
            rocm: Some("7.13.26176".to_string()),
            host_profile: test_host_profile(),
        };
        let rows = vec![
            row_for_model(
                BatteryId::Speed,
                None,
                "daemon_prefill_decode_first",
                None,
                EvalStatus::Skip,
                Some(
                    "daemon executor requires the model to resolve to a local filesystem path"
                        .to_string(),
                ),
                BTreeMap::new(),
                &cfg,
                &ctx,
                None,
                0,
                "candidate.hfq".to_string(),
            ),
            row_for_model(
                BatteryId::Speed,
                None,
                "daemon_prefill_decode_reset",
                None,
                EvalStatus::Skip,
                Some(
                    "daemon executor requires the model to resolve to a local filesystem path"
                        .to_string(),
                ),
                BTreeMap::new(),
                &cfg,
                &ctx,
                None,
                0,
                "candidate.hfq".to_string(),
            ),
        ];
        let comparison = build_comparison_artifact(&cfg, &rows, &ctx);
        let admission = build_admission_artifact(&cfg, &rows, &comparison, &ctx);
        let stdout = render_eval_stdout_findings(&admission, &rows);
        assert!(stdout.contains("admission: incomplete"));
        assert!(stdout.contains("results: 0"));
        assert!(stdout.contains("findings: 2"));
        assert!(stdout.contains("admission findings: 0"));
        assert!(stdout.contains("row findings: 2"));
        assert!(stdout.contains(
            "[skip] speed/daemon_prefill_decode_first suite=- item=- reason=daemon executor requires the model to resolve to a local filesystem path"
        ));
        assert!(stdout.contains(
            "[skip] speed/daemon_prefill_decode_reset suite=- item=- reason=daemon executor requires the model to resolve to a local filesystem path"
        ));
    }

    #[test]
    fn stdout_results_prefer_benchmark_aggregates() {
        let cfg = parse_args_from([
            "hipfire-eval",
            "--model",
            "candidate.hfq",
            "--battery",
            "speed",
            "--runs",
            "3",
        ])
        .unwrap();
        let ctx = EvalContext {
            commit_sha: Some("abc".to_string()),
            git_branch: Some("evaluation-harness".to_string()),
            git_describe: Some("v0.2.0-1-gabc".to_string()),
            git_dirty: Some(false),
            binary_hash: Some("binhash".to_string()),
            arch: Some("gfx1151".to_string()),
            rocm: Some("7.13.26176".to_string()),
            host_profile: test_host_profile(),
        };
        let mut rows = Vec::new();
        for (run_index, tok_s) in [(1, 100.0), (2, 110.0), (3, 120.0)] {
            rows.push(row_for_model(
                BatteryId::Speed,
                None,
                "decode",
                None,
                EvalStatus::Pass,
                None,
                BTreeMap::from([
                    ("benchmark_sample".to_string(), json!(true)),
                    ("run_index".to_string(), json!(run_index)),
                    ("run_count".to_string(), json!(3)),
                    ("tok_s".to_string(), json!(tok_s)),
                ]),
                &cfg,
                &ctx,
                None,
                10,
                "candidate.hfq".to_string(),
            ));
        }
        rows.push(row_for_model(
            BatteryId::Speed,
            None,
            "decode::aggregate",
            None,
            EvalStatus::Pass,
            None,
            BTreeMap::from([
                ("benchmark_aggregate".to_string(), json!(true)),
                ("aggregate_source_case_id".to_string(), json!("decode")),
                ("sample_count".to_string(), json!(3)),
                ("total_sample_count".to_string(), json!(3)),
                ("tok_s_median".to_string(), json!(110.0)),
                ("tok_s_stddev".to_string(), json!(10.0)),
            ]),
            &cfg,
            &ctx,
            None,
            30,
            "candidate.hfq".to_string(),
        ));
        let comparison = build_comparison_artifact(&cfg, &rows, &ctx);
        let admission = build_admission_artifact(&cfg, &rows, &comparison, &ctx);
        let stdout = render_eval_stdout_findings(&admission, &rows);
        assert!(stdout.contains("results: 1"));
        assert!(stdout.contains(
            "  [pass] speed/decode::aggregate suite=- item=- model=candidate.hfq elapsed_ms=30 tok_s_median=110 sample_count=3 total_sample_count=3 tok_s_stddev=10"
        ));
        assert!(!stdout.contains("  [pass] speed/decode suite=-"));
    }

    #[test]
    fn summary_includes_empty_dataset_and_artifact_sections() {
        let out = temp_path("summary-empty-sections.md");
        let cfg = parse_args_from([
            "hipfire-eval",
            "--model",
            "candidate.hfq",
            "--battery",
            "smoke,prompt_shape",
        ])
        .unwrap();
        let ctx = EvalContext {
            commit_sha: Some("abc".to_string()),
            git_branch: Some("evaluation-harness".to_string()),
            git_describe: Some("v0.2.0-1-gabc".to_string()),
            git_dirty: Some(false),
            binary_hash: Some("binhash".to_string()),
            arch: Some("gfx1151".to_string()),
            rocm: Some("7.13.26176".to_string()),
            host_profile: test_host_profile(),
        };
        let rows = vec![pass_row(
            BatteryId::Smoke,
            None,
            "boot",
            None,
            &cfg,
            &ctx,
            None,
            BTreeMap::new(),
        )];
        let comparison = build_comparison_artifact(&cfg, &rows, &ctx);
        let admission = build_admission_artifact(&cfg, &rows, &comparison, &ctx);
        let artifacts = BTreeMap::new();

        write_summary(
            &out,
            &cfg,
            &[],
            &comparison,
            &admission,
            &artifacts,
            &rows,
            &ctx,
        )
        .unwrap();
        let body = fs::read_to_string(&out).unwrap();
        assert!(body.contains("## Datasets"));
        assert!(body.contains("no dataset-backed suites selected"));
        assert!(body.contains("## Evidence Artifacts"));
        assert!(body.contains("no evidence artifacts collected"));
        assert!(body.contains("## Rows"));

        let _ = fs::remove_file(out);
    }

    #[test]
    fn evidence_artifacts_include_collected_result_records() {
        let out = temp_path("evidence-collected-artifacts");
        fs::create_dir_all(&out).unwrap();
        let cfg = parse_args_from([
            "hipfire-eval",
            "--model",
            "candidate.hfq",
            "--battery",
            "quality",
        ])
        .unwrap();
        let ctx = EvalContext {
            commit_sha: None,
            git_branch: None,
            git_describe: None,
            git_dirty: None,
            binary_hash: None,
            arch: None,
            rocm: None,
            host_profile: test_host_profile(),
        };
        let row = pass_row(
            BatteryId::Quality,
            None,
            "kld_reference_slice",
            None,
            &cfg,
            &ctx,
            None,
            BTreeMap::from([("mean_kld".to_string(), json!(0.12))]),
        );
        let comparison = ComparisonArtifact {
            schema: 1,
            provenance: run_provenance(&ctx),
            status: EvalStatus::Skip,
            reason: Some("no --compare or --reference provided".to_string()),
            baseline: None,
            reference: None,
            cases: Vec::new(),
        };

        let rows = vec![row];
        let admission = build_admission_artifact(&cfg, &rows, &comparison, &ctx);
        let artifacts =
            write_evidence_artifacts(&out, &cfg, &[], &rows, &comparison, &admission, &ctx)
                .unwrap();
        assert_eq!(
            artifacts
                .get("quality")
                .and_then(|v| v.get("status"))
                .and_then(Value::as_str),
            Some("collected")
        );
        let quality: Value =
            serde_json::from_str(&fs::read_to_string(out.join("quality.json")).unwrap()).unwrap();
        assert_eq!(quality["status"], "collected");
        assert_eq!(quality["reason"], Value::Null);
        assert_eq!(quality["records"][0]["metrics"]["mean_kld"], json!(0.12));

        let _ = fs::remove_dir_all(out);
    }

    #[test]
    fn quality_artifact_collects_barrage_accuracy_rows() {
        let out = temp_path("quality-artifact-barrage-accuracy");
        fs::create_dir_all(&out).unwrap();
        let cfg = parse_args_from([
            "hipfire-eval",
            "--model",
            "candidate.hfq",
            "--battery",
            "barrage",
            "--suite",
            "lm_eval_micro",
            "--offline",
        ])
        .unwrap();
        let ctx = EvalContext {
            commit_sha: None,
            git_branch: None,
            git_describe: None,
            git_dirty: None,
            binary_hash: None,
            arch: None,
            rocm: None,
            host_profile: test_host_profile(),
        };
        let row = row_for_model(
            BatteryId::Barrage,
            Some(SuiteId::LmEvalMicro),
            "lm_eval_micro_zero_shot_native",
            Some("arc_easy:0".to_string()),
            EvalStatus::Pass,
            None,
            BTreeMap::from([
                ("accuracy".to_string(), json!(1.0)),
                ("exact_match".to_string(), json!(1.0)),
                ("scoring_mode".to_string(), json!("exact_letter")),
            ]),
            &cfg,
            &ctx,
            None,
            0,
            "candidate.hfq".to_string(),
        );
        let comparison = build_comparison_artifact(&cfg, std::slice::from_ref(&row), &ctx);
        let admission =
            build_admission_artifact(&cfg, std::slice::from_ref(&row), &comparison, &ctx);
        let artifacts = write_evidence_artifacts(
            &out,
            &cfg,
            &[],
            std::slice::from_ref(&row),
            &comparison,
            &admission,
            &ctx,
        )
        .unwrap();

        assert_eq!(
            artifacts
                .get("quality")
                .and_then(|v| v.get("status"))
                .and_then(Value::as_str),
            Some("collected")
        );
        let quality: Value =
            serde_json::from_str(&fs::read_to_string(out.join("quality.json")).unwrap()).unwrap();
        assert_eq!(quality["expected_metrics"][4], "accuracy");
        assert_eq!(quality["records"][0]["battery"], "barrage");
        assert_eq!(quality["records"][0]["suite"], "lm_eval_micro");
        assert_eq!(quality["records"][0]["metrics"]["accuracy"], json!(1.0));
        assert_eq!(quality["records"][0]["metrics"]["exact_match"], json!(1.0));

        let _ = fs::remove_dir_all(out);
    }

    #[test]
    fn performance_artifact_collects_any_row_with_perf_metrics() {
        let out = temp_path("performance-metric-artifacts");
        fs::create_dir_all(&out).unwrap();
        let cfg = parse_args_from([
            "hipfire-eval",
            "--model",
            "candidate.hfq",
            "--battery",
            "smoke",
        ])
        .unwrap();
        let ctx = EvalContext {
            commit_sha: None,
            git_branch: None,
            git_describe: None,
            git_dirty: None,
            binary_hash: None,
            arch: None,
            rocm: None,
            host_profile: test_host_profile(),
        };
        let row = pass_row(
            BatteryId::Smoke,
            None,
            "finite_greedy_decode",
            None,
            &cfg,
            &ctx,
            prompt("benchmarks/prompts/qwen2_smoke.txt"),
            BTreeMap::from([("tok_s".to_string(), json!(123.4))]),
        );
        let comparison = ComparisonArtifact {
            schema: 1,
            provenance: run_provenance(&ctx),
            status: EvalStatus::Skip,
            reason: Some("no --compare or --reference provided".to_string()),
            baseline: None,
            reference: None,
            cases: Vec::new(),
        };

        let rows = vec![row];
        let admission = build_admission_artifact(&cfg, &rows, &comparison, &ctx);
        let artifacts =
            write_evidence_artifacts(&out, &cfg, &[], &rows, &comparison, &admission, &ctx)
                .unwrap();
        assert_eq!(
            artifacts
                .get("performance")
                .and_then(|v| v.get("status"))
                .and_then(Value::as_str),
            Some("collected")
        );
        let performance: Value =
            serde_json::from_str(&fs::read_to_string(out.join("performance.json")).unwrap())
                .unwrap();
        assert_eq!(performance["status"], "collected");
        assert_eq!(performance["records"][0]["battery"], "smoke");
        assert!(performance["records"][0]["model_hash"]
            .as_str()
            .unwrap_or("")
            .starts_with("tag:"));
        assert_eq!(performance["records"][0]["metrics"]["tok_s"], json!(123.4));

        let _ = fs::remove_dir_all(out);
    }

    #[test]
    fn dflash_rows_are_performance_evidence() {
        let out = temp_path("dflash-performance-artifacts");
        fs::create_dir_all(&out).unwrap();
        let cfg = parse_args_from([
            "hipfire-eval",
            "--model",
            "candidate.hfq",
            "--battery",
            "dflash",
            "--dflash",
            "auto",
        ])
        .unwrap();
        let ctx = EvalContext {
            commit_sha: None,
            git_branch: None,
            git_describe: None,
            git_dirty: None,
            binary_hash: None,
            arch: None,
            rocm: None,
            host_profile: test_host_profile(),
        };
        let row = pass_row(
            BatteryId::Dflash,
            None,
            "dflash_anchor",
            None,
            &cfg,
            &ctx,
            None,
            BTreeMap::from([
                ("tok_s".to_string(), json!(123.4)),
                ("tau".to_string(), json!(2.0)),
                ("accept_rate".to_string(), json!(0.5)),
            ]),
        );
        let comparison = ComparisonArtifact {
            schema: 1,
            provenance: run_provenance(&ctx),
            status: EvalStatus::Skip,
            reason: Some("no --compare or --reference provided".to_string()),
            baseline: None,
            reference: None,
            cases: Vec::new(),
        };

        let rows = vec![row];
        let admission = build_admission_artifact(&cfg, &rows, &comparison, &ctx);
        let artifacts =
            write_evidence_artifacts(&out, &cfg, &[], &rows, &comparison, &admission, &ctx)
                .unwrap();
        assert_eq!(
            artifacts
                .get("performance")
                .and_then(|v| v.get("status"))
                .and_then(Value::as_str),
            Some("collected")
        );
        let performance: Value =
            serde_json::from_str(&fs::read_to_string(out.join("performance.json")).unwrap())
                .unwrap();
        assert_eq!(performance["status"], "collected");
        assert_eq!(performance["records"][0]["battery"], "dflash");
        assert_eq!(performance["records"][0]["metrics"]["tok_s"], json!(123.4));

        let _ = fs::remove_dir_all(out);
    }

    #[test]
    fn dflash_trace_artifact_normalizes_ar_and_dflash_rows() {
        let out = temp_path("dflash-trace-artifacts");
        fs::create_dir_all(&out).unwrap();
        let cfg = parse_args_from([
            "hipfire-eval",
            "--model",
            "candidate.hfq",
            "--battery",
            "dflash",
            "--dflash",
            "auto",
        ])
        .unwrap();
        let ctx = EvalContext {
            commit_sha: None,
            git_branch: None,
            git_describe: None,
            git_dirty: None,
            binary_hash: None,
            arch: None,
            rocm: None,
            host_profile: test_host_profile(),
        };
        let rows = vec![
            pass_row(
                BatteryId::Dflash,
                None,
                "ar_coherence_anchor",
                None,
                &cfg,
                &ctx,
                None,
                BTreeMap::from([
                    ("tok_s".to_string(), json!(90.0)),
                    ("ar_baseline".to_string(), json!(true)),
                ]),
            ),
            pass_row(
                BatteryId::Dflash,
                None,
                "dflash_anchor",
                None,
                &cfg,
                &ctx,
                None,
                BTreeMap::from([
                    ("tok_s".to_string(), json!(130.0)),
                    ("ar_baseline".to_string(), json!(false)),
                    ("tau".to_string(), json!(2.5)),
                    ("accept_rate".to_string(), json!(0.6)),
                    (
                        "rollback_logit_compare".to_string(),
                        json!({
                            "ok": true,
                            "checked": 8,
                            "argmax_mismatches": 0,
                        }),
                    ),
                    (
                        "rollback_state_compare".to_string(),
                        json!({
                            "ok": false,
                            "checked": 1,
                            "reason": "fast_replay_recurrent_state_mismatch",
                            "first_mismatch": {
                                "family": "s_matrix",
                                "index": 0,
                                "stats": {
                                    "f32_bit_diff_words": 702,
                                    "max_abs": 2.53553992e38_f64,
                                },
                            },
                        }),
                    ),
                    (
                        "rollback_fast_replay_admission".to_string(),
                        json!({
                            "verdict": "rejected",
                            "blockers": ["fast_replay_recurrent_state_mismatch"],
                            "logit_checked": 8,
                            "state_checked": 1,
                        }),
                    ),
                ]),
            ),
        ];
        let comparison = ComparisonArtifact {
            schema: 1,
            provenance: run_provenance(&ctx),
            status: EvalStatus::Skip,
            reason: Some("no --compare or --reference provided".to_string()),
            baseline: None,
            reference: None,
            cases: Vec::new(),
        };

        let admission = build_admission_artifact(&cfg, &rows, &comparison, &ctx);
        let artifacts =
            write_evidence_artifacts(&out, &cfg, &[], &rows, &comparison, &admission, &ctx)
                .unwrap();
        assert_eq!(
            artifacts
                .get("dflash_trace")
                .and_then(|v| v.get("status"))
                .and_then(Value::as_str),
            Some("collected")
        );
        let trace: Value =
            serde_json::from_str(&fs::read_to_string(out.join("dflash_trace.json")).unwrap())
                .unwrap();
        assert_eq!(trace["status"], "collected");
        assert_eq!(trace["records"][0]["metrics"]["mode"], json!("ar"));
        assert_eq!(trace["records"][0]["metrics"]["ar_tok_s"], json!(90.0));
        assert_eq!(trace["records"][1]["metrics"]["mode"], json!("dflash"));
        assert_eq!(trace["records"][1]["metrics"]["dflash_tok_s"], json!(130.0));
        assert_eq!(trace["records"][1]["metrics"]["tau"], json!(2.5));
        assert_eq!(trace["records"][1]["metrics"]["accept_rate"], json!(0.6));
        assert_eq!(
            trace["records"][1]["metrics"]["rollback_logit_compare"]["argmax_mismatches"],
            json!(0)
        );
        assert_eq!(
            trace["records"][1]["metrics"]["rollback_state_compare"]["reason"],
            json!("fast_replay_recurrent_state_mismatch")
        );
        assert_eq!(
            trace["records"][1]["metrics"]["rollback_state_compare"]["first_mismatch"]["stats"]
                ["f32_bit_diff_words"],
            json!(702)
        );
        assert_eq!(
            trace["records"][1]["metrics"]["rollback_fast_replay_admission"]["verdict"],
            json!("rejected")
        );

        let _ = fs::remove_dir_all(out);
    }

    #[test]
    fn path_c_trace_artifact_collects_only_path_c_rows() {
        let out = temp_path("path-c-trace-artifacts");
        fs::create_dir_all(&out).unwrap();
        let cfg = parse_args_from([
            "hipfire-eval",
            "--model",
            "candidate.hfq",
            "--battery",
            "dflash",
            "--dflash",
            "auto",
        ])
        .unwrap();
        let ctx = EvalContext {
            commit_sha: None,
            git_branch: None,
            git_describe: None,
            git_dirty: None,
            binary_hash: None,
            arch: None,
            rocm: None,
            host_profile: test_host_profile(),
        };
        let rows = vec![
            pass_row(
                BatteryId::Dflash,
                None,
                "dflash_anchor",
                None,
                &cfg,
                &ctx,
                None,
                BTreeMap::from([
                    ("mode".to_string(), json!("dflash")),
                    ("tok_s".to_string(), json!(130.0)),
                ]),
            ),
            pass_row(
                BatteryId::Dflash,
                None,
                "path_c_phase1_prose",
                None,
                &cfg,
                &ctx,
                None,
                BTreeMap::from([
                    ("mode".to_string(), json!("path-c-phase1")),
                    ("tok_s".to_string(), json!(5.86)),
                    ("tau".to_string(), json!(1.23)),
                    ("max_tokens".to_string(), json!(192)),
                    (
                        "verify_graph".to_string(),
                        json!({
                            "direct": 0,
                            "warmup": 3,
                            "capture": 3,
                            "replay": 81,
                            "not_applicable": 0,
                        }),
                    ),
                ]),
            ),
            pass_row(
                BatteryId::Dflash,
                None,
                "verify_graph_promotion",
                None,
                &cfg,
                &ctx,
                None,
                BTreeMap::from([
                    ("promotion_verdict".to_string(), json!("NOT_PROMOTED")),
                    ("paired_cases".to_string(), json!(4)),
                    (
                        "blockers".to_string(),
                        json!(["path-c-phase2-code: tok/s delta -3.732% < 5.000%"]),
                    ),
                    (
                        "pairs".to_string(),
                        json!([{
                            "case_id": "path-c-phase2-code",
                            "tok_s_delta_pct": -3.732,
                        }]),
                    ),
                ]),
            ),
        ];
        let comparison = ComparisonArtifact {
            schema: 1,
            provenance: run_provenance(&ctx),
            status: EvalStatus::Skip,
            reason: Some("no --compare or --reference provided".to_string()),
            baseline: None,
            reference: None,
            cases: Vec::new(),
        };

        let admission = build_admission_artifact(&cfg, &rows, &comparison, &ctx);
        let artifacts =
            write_evidence_artifacts(&out, &cfg, &[], &rows, &comparison, &admission, &ctx)
                .unwrap();
        assert_eq!(
            artifacts
                .get("path_c_trace")
                .and_then(|v| v.get("status"))
                .and_then(Value::as_str),
            Some("collected")
        );
        let trace: Value =
            serde_json::from_str(&fs::read_to_string(out.join("path_c_trace.json")).unwrap())
                .unwrap();
        assert_eq!(trace["status"], "collected");
        assert_eq!(trace["records"].as_array().unwrap().len(), 2);
        assert_eq!(trace["records"][0]["case_id"], json!("path_c_phase1_prose"));
        assert_eq!(
            trace["records"][0]["metrics"]["mode"],
            json!("path-c-phase1")
        );
        assert_eq!(
            trace["records"][0]["metrics"]["verify_graph"]["direct"],
            json!(0)
        );
        assert_eq!(
            trace["records"][1]["case_id"],
            json!("verify_graph_promotion")
        );
        assert_eq!(
            trace["records"][1]["metrics"]["promotion_verdict"],
            json!("NOT_PROMOTED")
        );
        assert_eq!(
            trace["records"][1]["metrics"]["blockers"][0],
            json!("path-c-phase2-code: tok/s delta -3.732% < 5.000%")
        );

        let _ = fs::remove_dir_all(out);
    }

    #[test]
    fn phase_timing_artifact_collects_example_runtime_metrics() {
        let out = temp_path("phase-timing-artifacts");
        fs::create_dir_all(&out).unwrap();
        let cfg = parse_args_from([
            "hipfire-eval",
            "--model",
            "candidate.hfq",
            "--battery",
            "speed",
        ])
        .unwrap();
        let ctx = EvalContext {
            commit_sha: None,
            git_branch: None,
            git_describe: None,
            git_dirty: None,
            binary_hash: None,
            arch: None,
            rocm: None,
            host_profile: test_host_profile(),
        };
        let row = pass_row(
            BatteryId::Speed,
            None,
            "speed_short_decode",
            None,
            &cfg,
            &ctx,
            prompt("benchmarks/prompts/qwen2_smoke.txt"),
            BTreeMap::from([
                ("prefill_secs".to_string(), json!(0.1234)),
                ("decode_secs".to_string(), json!(0.1000)),
                ("ttft_ms".to_string(), json!(17.25)),
            ]),
        );
        let comparison = ComparisonArtifact {
            schema: 1,
            provenance: run_provenance(&ctx),
            status: EvalStatus::Skip,
            reason: Some("no --compare or --reference provided".to_string()),
            baseline: None,
            reference: None,
            cases: Vec::new(),
        };

        let rows = vec![row];
        let admission = build_admission_artifact(&cfg, &rows, &comparison, &ctx);
        let artifacts =
            write_evidence_artifacts(&out, &cfg, &[], &rows, &comparison, &admission, &ctx)
                .unwrap();
        assert_eq!(
            artifacts
                .get("phase_timings")
                .and_then(|v| v.get("status"))
                .and_then(Value::as_str),
            Some("collected")
        );
        let phase_timings: Value =
            serde_json::from_str(&fs::read_to_string(out.join("phase_timings.json")).unwrap())
                .unwrap();
        assert_eq!(phase_timings["status"], "collected");
        assert_eq!(phase_timings["records"][0]["battery"], "speed");
        assert_eq!(
            phase_timings["records"][0]["metrics"]["prefill_ms"],
            json!(123.4)
        );
        assert_eq!(
            phase_timings["records"][0]["metrics"]["decode_ms"],
            json!(100.0)
        );
        assert_eq!(
            phase_timings["records"][0]["metrics"]["ttft_ms"],
            json!(17.25)
        );

        let _ = fs::remove_dir_all(out);
    }

    #[test]
    fn memory_artifact_collects_example_vram_metrics() {
        let out = temp_path("memory-artifacts");
        fs::create_dir_all(&out).unwrap();
        let cfg = parse_args_from([
            "hipfire-eval",
            "--model",
            "candidate.hfq",
            "--battery",
            "speed",
        ])
        .unwrap();
        let ctx = EvalContext {
            commit_sha: None,
            git_branch: None,
            git_describe: None,
            git_dirty: None,
            binary_hash: None,
            arch: None,
            rocm: None,
            host_profile: test_host_profile(),
        };
        let row = pass_row(
            BatteryId::Speed,
            None,
            "speed_short_decode",
            None,
            &cfg,
            &ctx,
            prompt("benchmarks/prompts/qwen2_smoke.txt"),
            BTreeMap::from([("vram_used_mb".to_string(), json!(1234.0))]),
        );
        let comparison = ComparisonArtifact {
            schema: 1,
            provenance: run_provenance(&ctx),
            status: EvalStatus::Skip,
            reason: Some("no --compare or --reference provided".to_string()),
            baseline: None,
            reference: None,
            cases: Vec::new(),
        };

        let rows = vec![row];
        let admission = build_admission_artifact(&cfg, &rows, &comparison, &ctx);
        let artifacts =
            write_evidence_artifacts(&out, &cfg, &[], &rows, &comparison, &admission, &ctx)
                .unwrap();
        assert_eq!(
            artifacts
                .get("memory")
                .and_then(|v| v.get("status"))
                .and_then(Value::as_str),
            Some("collected")
        );
        let memory: Value =
            serde_json::from_str(&fs::read_to_string(out.join("memory.json")).unwrap()).unwrap();
        assert_eq!(memory["status"], "collected");
        assert_eq!(memory["records"][0]["battery"], "speed");
        assert_eq!(
            memory["records"][0]["metrics"]["vram_peak_bytes"],
            json!(1234.0 * 1024.0 * 1024.0)
        );

        let _ = fs::remove_dir_all(out);
    }

    #[test]
    fn native_result_rows_collect_launch_counts_moe_and_profiling_artifacts() {
        let out = temp_path("native-runtime-evidence-artifacts");
        fs::create_dir_all(&out).unwrap();
        let cfg = parse_args_from([
            "hipfire-eval",
            "--model",
            "candidate.hfq",
            "--battery",
            "speed",
            "--profile",
            "passive",
        ])
        .unwrap();
        let ctx = EvalContext {
            commit_sha: None,
            git_branch: None,
            git_describe: None,
            git_dirty: None,
            binary_hash: None,
            arch: None,
            rocm: None,
            host_profile: test_host_profile(),
        };
        let row = pass_row(
            BatteryId::Speed,
            None,
            "speed_short_decode",
            None,
            &cfg,
            &ctx,
            prompt("benchmarks/prompts/qwen2_smoke.txt"),
            BTreeMap::from([
                ("kernel_launches".to_string(), json!(42)),
                ("graph_launches".to_string(), json!(3)),
                ("memcpy_ops".to_string(), json!(7)),
                ("expert_hits".to_string(), json!({"0": 11, "1": 5})),
                ("router_entropy".to_string(), json!(0.73)),
                ("kernel_name".to_string(), json!("attention_flash")),
                ("duration_us".to_string(), json!(912.5)),
                ("occupancy".to_string(), json!(0.82)),
                ("waves".to_string(), json!(128)),
                ("tok_s".to_string(), json!(123.4)),
            ]),
        );
        let comparison = ComparisonArtifact {
            schema: 1,
            provenance: run_provenance(&ctx),
            status: EvalStatus::Skip,
            reason: Some("no --compare or --reference provided".to_string()),
            baseline: None,
            reference: None,
            cases: Vec::new(),
        };

        let rows = vec![row];
        let admission = build_admission_artifact(&cfg, &rows, &comparison, &ctx);
        let artifacts =
            write_evidence_artifacts(&out, &cfg, &[], &rows, &comparison, &admission, &ctx)
                .unwrap();
        for kind in ["launch_counts", "moe_router_histogram", "profiling"] {
            assert_eq!(
                artifacts
                    .get(kind)
                    .and_then(|v| v.get("status"))
                    .and_then(Value::as_str),
                Some("collected"),
                "{kind}"
            );
            assert_eq!(
                artifacts
                    .get(kind)
                    .and_then(|v| v.get("row_count"))
                    .and_then(Value::as_u64),
                Some(1),
                "{kind}"
            );
        }
        let launch_counts: Value =
            serde_json::from_str(&fs::read_to_string(out.join("launch_counts.json")).unwrap())
                .unwrap();
        assert_eq!(launch_counts["status"], "collected");
        assert_eq!(
            launch_counts["records"][0]["metrics"]["kernel_launches"],
            json!(42)
        );
        assert_eq!(
            launch_counts["records"][0]["metrics"].get("tok_s"),
            None,
            "unrelated performance metrics must not leak into launch evidence"
        );
        let router: Value = serde_json::from_str(
            &fs::read_to_string(out.join("moe_router_histogram.json")).unwrap(),
        )
        .unwrap();
        assert_eq!(router["status"], "collected");
        assert_eq!(
            router["records"][0]["metrics"]["expert_hits"],
            json!({"0": 11, "1": 5})
        );
        let profiling: Value =
            serde_json::from_str(&fs::read_to_string(out.join("profiling.json")).unwrap()).unwrap();
        assert_eq!(profiling["status"], "collected");
        assert_eq!(
            profiling["records"][0]["metrics"]["kernel_name"],
            json!("attention_flash")
        );
        assert_eq!(
            profiling["records"][0]["metrics"]["duration_us"],
            json!(912.5)
        );

        let _ = fs::remove_dir_all(out);
    }

    #[test]
    fn evidence_json_ingest_collects_runtime_artifacts() {
        let out = temp_path("evidence-json-artifacts");
        let evidence_json = temp_path("runtime-evidence.json");
        fs::create_dir_all(&out).unwrap();
        fs::write(
            &evidence_json,
            serde_json::to_string(&json!({
                "launch_counts": {
                    "records": [
                        {
                            "case_id": "pp128",
                            "metrics": {
                                "kernel_launches": 42,
                                "graph_launches": 3,
                                "memcpy_ops": 7
                            }
                        }
                    ]
                },
                "moe_router_histogram": {
                    "records": [
                        {
                            "case_id": "gpqa-0",
                            "metrics": {
                                "expert_hits": {"0": 11, "1": 5},
                                "router_entropy": 0.73
                            }
                        }
                    ]
                },
                "module_evidence": {
                    "records": [
                        {
                            "case_id": "qwen35_dense_ffn_swiglu_down.layer4",
                            "metrics": {
                                "module_kind": "dense_ffn_swiglu_down",
                                "module_id": "qwen35_dense_ffn_swiglu_down.layer4",
                                "preferred_backend": "npu_opt_in",
                                "selected_backend": "gpu_production",
                                "oracle_backend": "cpu_oracle",
                                "fallback_reason": "npu_backend_unavailable",
                                "drift": {
                                    "max_abs": 0.00125,
                                    "mean_abs": 0.00012
                                }
                            }
                        }
                    ]
                }
            }))
            .unwrap(),
        )
        .unwrap();
        let cfg = parse_args_from([
            "hipfire-eval",
            "--model",
            "candidate.hfq",
            "--battery",
            "speed",
            "--evidence-json",
            evidence_json.to_str().unwrap(),
        ])
        .unwrap();
        let ctx = EvalContext {
            commit_sha: Some("abc123".to_string()),
            git_branch: Some("evaluation-harness".to_string()),
            git_describe: Some("v0.2.0-1-gabc123".to_string()),
            git_dirty: Some(true),
            binary_hash: Some("binhash".to_string()),
            arch: Some("gfx1151".to_string()),
            rocm: Some("7.13.26176".to_string()),
            host_profile: test_host_profile(),
        };
        let comparison = ComparisonArtifact {
            schema: 1,
            provenance: run_provenance(&ctx),
            status: EvalStatus::Skip,
            reason: Some("no --compare or --reference provided".to_string()),
            baseline: None,
            reference: None,
            cases: Vec::new(),
        };

        let rows = Vec::new();
        let admission = build_admission_artifact(&cfg, &rows, &comparison, &ctx);
        let artifacts =
            write_evidence_artifacts(&out, &cfg, &[], &rows, &comparison, &admission, &ctx)
                .unwrap();
        assert_eq!(
            artifacts
                .get("launch_counts")
                .and_then(|v| v.get("status"))
                .and_then(Value::as_str),
            Some("collected")
        );
        assert_eq!(
            artifacts
                .get("launch_counts")
                .and_then(|v| v.get("row_count"))
                .and_then(Value::as_u64),
            Some(1)
        );
        assert_eq!(
            artifacts
                .get("launch_counts")
                .and_then(|v| v.get("kind"))
                .and_then(Value::as_str),
            Some("launch_counts")
        );
        assert!(artifacts
            .get("launch_counts")
            .and_then(|v| v.get("expected_metrics"))
            .and_then(Value::as_array)
            .unwrap()
            .iter()
            .any(|v| v == "kernel_launches"));
        assert_eq!(
            artifacts
                .get("moe_router_histogram")
                .and_then(|v| v.get("status"))
                .and_then(Value::as_str),
            Some("collected")
        );
        assert_eq!(
            artifacts
                .get("moe_router_histogram")
                .and_then(|v| v.get("row_count"))
                .and_then(Value::as_u64),
            Some(1)
        );
        assert_eq!(
            artifacts
                .get("moe_router_histogram")
                .and_then(|v| v.get("kind"))
                .and_then(Value::as_str),
            Some("moe_router_histogram")
        );
        assert_eq!(
            artifacts
                .get("module_evidence")
                .and_then(|v| v.get("status"))
                .and_then(Value::as_str),
            Some("collected")
        );
        assert_eq!(
            artifacts
                .get("module_evidence")
                .and_then(|v| v.get("row_count"))
                .and_then(Value::as_u64),
            Some(1)
        );
        assert!(artifacts
            .get("module_evidence")
            .and_then(|v| v.get("expected_metrics"))
            .and_then(Value::as_array)
            .unwrap()
            .iter()
            .any(|v| v == "selected_backend"));
        assert!(artifacts
            .get("moe_router_histogram")
            .and_then(|v| v.get("expected_metrics"))
            .and_then(Value::as_array)
            .unwrap()
            .iter()
            .any(|v| v == "expert_hits"));
        let launch_counts: Value =
            serde_json::from_str(&fs::read_to_string(out.join("launch_counts.json")).unwrap())
                .unwrap();
        assert_eq!(launch_counts["status"], "collected");
        assert_eq!(
            launch_counts["records"][0]["metrics"]["kernel_launches"],
            json!(42)
        );
        assert_eq!(
            launch_counts["records"][0]["source_path"],
            json!(evidence_json.display().to_string())
        );
        assert_eq!(
            launch_counts["records"][0]["hipfire_eval_context"]["model"],
            json!("candidate.hfq")
        );
        assert_eq!(
            launch_counts["records"][0]["hipfire_eval_context"]["runner"],
            json!("hipfire-eval")
        );
        assert!(
            launch_counts["records"][0]["hipfire_eval_context"]["runner_version"]
                .as_str()
                .is_some()
        );
        assert_eq!(
            launch_counts["records"][0]["hipfire_eval_context"]["git_commit"],
            json!("abc123")
        );
        assert_eq!(
            launch_counts["records"][0]["hipfire_eval_context"]["binary_hash"],
            json!("binhash")
        );
        assert_eq!(
            launch_counts["records"][0]["hipfire_eval_context"]["arch"],
            json!("gfx1151")
        );
        assert_eq!(
            launch_counts["records"][0]["hipfire_eval_context"]["rocm"],
            json!("7.13.26176")
        );
        let router: Value = serde_json::from_str(
            &fs::read_to_string(out.join("moe_router_histogram.json")).unwrap(),
        )
        .unwrap();
        assert_eq!(router["status"], "collected");
        assert_eq!(
            router["records"][0]["metrics"]["router_entropy"],
            json!(0.73)
        );
        assert_eq!(
            router["records"][0]["hipfire_eval_context"]["git_branch"],
            json!("evaluation-harness")
        );
        let module_evidence: Value =
            serde_json::from_str(&fs::read_to_string(out.join("module_evidence.json")).unwrap())
                .unwrap();
        assert_eq!(module_evidence["status"], "collected");
        assert_eq!(
            module_evidence["records"][0]["metrics"]["selected_backend"],
            json!("gpu_production")
        );
        assert_eq!(
            module_evidence["records"][0]["metrics"]["fallback_reason"],
            json!("npu_backend_unavailable")
        );
        assert_eq!(
            module_evidence["records"][0]["metrics"]["drift"]["max_abs"],
            json!(0.00125)
        );
        assert_eq!(
            module_evidence["records"][0]["source_path"],
            json!(evidence_json.display().to_string())
        );

        let _ = fs::remove_dir_all(out);
        let _ = fs::remove_file(evidence_json);
    }

    #[test]
    fn evidence_dir_ingest_collects_standard_runtime_artifacts() {
        let out = temp_path("evidence-dir-artifacts");
        let evidence_dir = temp_path("runtime-evidence-dir");
        fs::create_dir_all(&out).unwrap();
        fs::create_dir_all(&evidence_dir).unwrap();
        fs::write(
            evidence_dir.join("launch_counts.json"),
            serde_json::to_string(&json!({
                "schema": 1,
                "kind": "launch_counts",
                "records": [
                    {
                        "case_id": "decode",
                        "metrics": {
                            "kernel_launches": 17,
                            "graph_launches": 2,
                            "memcpy_ops": 1
                        }
                    }
                ]
            }))
            .unwrap(),
        )
        .unwrap();
        fs::write(
            evidence_dir.join("profiling.json"),
            serde_json::to_string(&json!({
                "schema": 1,
                "kind": "profiling",
                "records": [
                    {
                        "case_id": "decode",
                        "metrics": {
                            "kernel_name": "attention_dflash",
                            "duration_us": 42.5,
                            "occupancy": 0.75,
                            "waves": 16
                        }
                    }
                ]
            }))
            .unwrap(),
        )
        .unwrap();
        fs::write(
            evidence_dir.join("dflash_trace.json"),
            serde_json::to_string(&json!({
                "kind": "dflash_trace",
                "records": [
                    {
                        "battery": "dflash",
                        "case_id": "27b-dflash-prose",
                        "status": "OK",
                        "metrics": {
                            "mode": "dflash",
                            "rollback_fast_replay_admission": {
                                "verdict": "rejected",
                                "blockers": ["fast_replay_recurrent_state_mismatch"],
                                "logit_checked": 8,
                                "state_checked": 1
                            },
                            "rollback_state_compare": {
                                "ok": false,
                                "reason": "fast_replay_recurrent_state_mismatch",
                                "first_mismatch": {
                                    "family": "s_matrix",
                                    "stats": {
                                        "f32_bit_diff_words": 702,
                                        "max_abs": 2.53553992e38_f64
                                    }
                                }
                            }
                        }
                    },
                    {
                        "battery": "dflash",
                        "case_id": "rollback_fast_replay_admission_summary",
                        "status": "rejected",
                        "metrics": {
                            "verdict": "rejected",
                            "case_count": 2,
                            "admission_count": 2,
                            "verdict_counts": {
                                "admitted": 0,
                                "rejected": 1,
                                "not_evaluated": 1,
                                "other": 0
                            },
                            "blocker_counts": {
                                "fast_replay_recurrent_state_mismatch": 1
                            },
                            "logit_checked": 8,
                            "state_checked": 1
                        }
                    }
                ]
            }))
            .unwrap(),
        )
        .unwrap();
        fs::write(
            evidence_dir.join("path_c_trace.json"),
            serde_json::to_string(&json!({
                "kind": "path_c_trace",
                "records": [
                    {
                        "battery": "path_c",
                        "case_id": "verify_graph_promotion",
                        "status": "NOT_PROMOTED",
                        "metrics": {
                            "promotion_verdict": "NOT_PROMOTED",
                            "paired_cases": 4,
                            "tok_s_min_delta_pct": 5.0,
                            "tau_min_delta_pct": -1.0,
                            "blockers": [
                                "path-c-phase2-code: tok/s delta -11.553% < 5.000%"
                            ],
                            "pairs": [
                                {
                                    "case_id": "path-c-phase2-code",
                                    "graph_tok_s": 42.0,
                                    "nograph_tok_s": 47.5,
                                    "tok_s_delta_pct": -11.553,
                                    "graph_tau": 2.0,
                                    "nograph_tau": 2.0,
                                    "tau_delta_pct": 0.0
                                }
                            ]
                        }
                    }
                ]
            }))
            .unwrap(),
        )
        .unwrap();
        fs::write(
            evidence_dir.join("module_evidence.json"),
            serde_json::to_string(&json!({
                "kind": "module_evidence",
                "records": [
                    {
                        "case_id": "qwen35_dense_ffn_swiglu_down.layer4",
                        "metrics": {
                            "module_kind": "dense_ffn_swiglu_down",
                            "module_id": "qwen35_dense_ffn_swiglu_down.layer4",
                            "preferred_backend": "npu_opt_in",
                            "selected_backend": "gpu_production",
                            "oracle_backend": "cpu_oracle",
                            "fallback_reason": "npu_backend_unavailable",
                            "drift": {
                                "max_abs": 0.00125,
                                "mean_abs": 0.00012
                            }
                        }
                    }
                ]
            }))
            .unwrap(),
        )
        .unwrap();
        fs::write(
            evidence_dir.join("notes.json"),
            serde_json::to_string(&json!({"kind": "launch_counts", "records": []})).unwrap(),
        )
        .unwrap();
        let cfg = parse_args_from([
            "hipfire-eval",
            "--model",
            "candidate.hfq",
            "--battery",
            "speed",
            "--profile",
            "passive",
            "--evidence-dir",
            evidence_dir.to_str().unwrap(),
        ])
        .unwrap();
        let ctx = EvalContext {
            commit_sha: Some("abc123".to_string()),
            git_branch: Some("evaluation-harness".to_string()),
            git_describe: Some("v0.2.0-1-gabc123".to_string()),
            git_dirty: Some(true),
            binary_hash: Some("binhash".to_string()),
            arch: Some("gfx1151".to_string()),
            rocm: Some("7.13.26176".to_string()),
            host_profile: test_host_profile(),
        };
        let comparison = ComparisonArtifact {
            schema: 1,
            provenance: run_provenance(&ctx),
            status: EvalStatus::Skip,
            reason: Some("no --compare or --reference provided".to_string()),
            baseline: None,
            reference: None,
            cases: Vec::new(),
        };

        let rows = Vec::new();
        let admission = build_admission_artifact(&cfg, &rows, &comparison, &ctx);
        let artifacts =
            write_evidence_artifacts(&out, &cfg, &[], &rows, &comparison, &admission, &ctx)
                .unwrap();
        assert_eq!(
            artifacts
                .get("launch_counts")
                .and_then(|v| v.get("status"))
                .and_then(Value::as_str),
            Some("collected")
        );
        assert_eq!(
            artifacts
                .get("profiling")
                .and_then(|v| v.get("status"))
                .and_then(Value::as_str),
            Some("collected")
        );
        assert_eq!(
            artifacts
                .get("dflash_trace")
                .and_then(|v| v.get("status"))
                .and_then(Value::as_str),
            Some("collected")
        );
        assert_eq!(
            artifacts
                .get("path_c_trace")
                .and_then(|v| v.get("status"))
                .and_then(Value::as_str),
            Some("collected")
        );
        assert_eq!(
            artifacts
                .get("module_evidence")
                .and_then(|v| v.get("status"))
                .and_then(Value::as_str),
            Some("collected")
        );
        let launch_counts: Value =
            serde_json::from_str(&fs::read_to_string(out.join("launch_counts.json")).unwrap())
                .unwrap();
        assert_eq!(
            launch_counts["collection"]["evidence_dirs"][0],
            json!(evidence_dir.display().to_string())
        );
        assert_eq!(
            launch_counts["records"][0]["metrics"]["kernel_launches"],
            json!(17)
        );
        assert_eq!(
            launch_counts["records"][0]["source_path"],
            json!(evidence_dir
                .join("launch_counts.json")
                .display()
                .to_string())
        );
        assert_eq!(
            launch_counts["records"][0]["hipfire_eval_context"]["git_commit"],
            json!("abc123")
        );
        assert!(
            launch_counts["records"][0]["hipfire_eval_context"]["runner_version"]
                .as_str()
                .is_some()
        );
        let profiling: Value =
            serde_json::from_str(&fs::read_to_string(out.join("profiling.json")).unwrap()).unwrap();
        assert_eq!(profiling["status"], "collected");
        assert_eq!(
            profiling["records"][0]["metrics"]["kernel_name"],
            json!("attention_dflash")
        );
        assert_eq!(
            profiling["records"][0]["source_path"],
            json!(evidence_dir.join("profiling.json").display().to_string())
        );
        assert_eq!(
            profiling["records"][0]["hipfire_eval_context"]["arch"],
            json!("gfx1151")
        );
        assert!(
            profiling["records"][0]["hipfire_eval_context"]["runner_version"]
                .as_str()
                .is_some()
        );
        let trace: Value =
            serde_json::from_str(&fs::read_to_string(out.join("dflash_trace.json")).unwrap())
                .unwrap();
        assert_eq!(trace["status"], "collected");
        assert_eq!(
            trace["records"][0]["metrics"]["rollback_fast_replay_admission"]["verdict"],
            json!("rejected")
        );
        assert_eq!(
            trace["records"][0]["metrics"]["rollback_state_compare"]["first_mismatch"]["stats"]
                ["f32_bit_diff_words"],
            json!(702)
        );
        assert_eq!(
            trace["records"][0]["source_path"],
            json!(evidence_dir.join("dflash_trace.json").display().to_string())
        );
        assert_eq!(
            trace["records"][1]["case_id"],
            json!("rollback_fast_replay_admission_summary")
        );
        assert_eq!(trace["records"][1]["metrics"]["verdict"], json!("rejected"));
        assert_eq!(
            trace["records"][1]["metrics"]["blocker_counts"]
                ["fast_replay_recurrent_state_mismatch"],
            json!(1)
        );
        let path_c_trace: Value =
            serde_json::from_str(&fs::read_to_string(out.join("path_c_trace.json")).unwrap())
                .unwrap();
        assert_eq!(path_c_trace["status"], "collected");
        assert_eq!(
            path_c_trace["records"][0]["metrics"]["promotion_verdict"],
            json!("NOT_PROMOTED")
        );
        assert_eq!(
            path_c_trace["records"][0]["metrics"]["blockers"][0],
            json!("path-c-phase2-code: tok/s delta -11.553% < 5.000%")
        );
        assert_eq!(
            path_c_trace["records"][0]["source_path"],
            json!(evidence_dir.join("path_c_trace.json").display().to_string())
        );
        let module_evidence: Value =
            serde_json::from_str(&fs::read_to_string(out.join("module_evidence.json")).unwrap())
                .unwrap();
        assert_eq!(module_evidence["status"], "collected");
        assert_eq!(
            module_evidence["records"][0]["metrics"]["selected_backend"],
            json!("gpu_production")
        );
        assert_eq!(
            module_evidence["records"][0]["metrics"]["fallback_reason"],
            json!("npu_backend_unavailable")
        );
        assert_eq!(
            module_evidence["records"][0]["source_path"],
            json!(evidence_dir
                .join("module_evidence.json")
                .display()
                .to_string())
        );

        let _ = fs::remove_dir_all(out);
        let _ = fs::remove_dir_all(evidence_dir);
    }

    #[test]
    fn result_runtime_evidence_dir_feeds_artifact_collection() {
        let out = temp_path("result-runtime-evidence-artifacts");
        let evidence_dir = temp_path("result-runtime-evidence-dir");
        fs::create_dir_all(&out).unwrap();
        fs::create_dir_all(&evidence_dir).unwrap();
        fs::write(
            evidence_dir.join("performance.json"),
            serde_json::to_string(&json!({
                "schema": 1,
                "kind": "performance",
                "records": [
                    {
                        "case_id": "run_oneshot",
                        "metrics": {
                            "tok_s": 12.5,
                            "prefill_tok_s": 100.0,
                            "ttft_ms": 25.0
                        }
                    }
                ]
            }))
            .unwrap(),
        )
        .unwrap();
        fs::write(
            evidence_dir.join("memory.json"),
            serde_json::to_string(&json!({
                "schema": 1,
                "kind": "memory",
                "records": [
                    {
                        "case_id": "run_oneshot",
                        "metrics": {
                            "vram_peak_bytes": 1234
                        }
                    }
                ]
            }))
            .unwrap(),
        )
        .unwrap();
        fs::write(
            evidence_dir.join("launch_counts.json"),
            serde_json::to_string(&json!({
                "schema": 1,
                "kind": "launch_counts",
                "records": [
                    {
                        "case_id": "run_oneshot",
                        "metrics": {
                            "kernel_launches": 7,
                            "graph_launches": 0,
                            "memcpy_ops": 2,
                            "counting_scope": "model_forward_call_proxy"
                        }
                    }
                ]
            }))
            .unwrap(),
        )
        .unwrap();
        fs::write(
            evidence_dir.join("moe_router_histogram.json"),
            serde_json::to_string(&json!({
                "schema": 1,
                "kind": "moe_router_histogram",
                "records": [
                    {
                        "case_id": "run_oneshot",
                        "metrics": {
                            "expert_hits": {"17": 4, "42": 2},
                            "shared_expert_hits": 6,
                            "router_entropy": 0.69,
                            "routed_tokens": 1,
                            "routed_slots": 8,
                            "num_experts": 256
                        }
                    }
                ]
            }))
            .unwrap(),
        )
        .unwrap();
        let cfg = parse_args_from([
            "hipfire-eval",
            "--model",
            "candidate.hfq",
            "--battery",
            "speed",
        ])
        .unwrap();
        let ctx = EvalContext {
            commit_sha: Some("abc123".to_string()),
            git_branch: Some("evaluation-harness".to_string()),
            git_describe: Some("v0.2.0-1-gabc123".to_string()),
            git_dirty: Some(true),
            binary_hash: Some("binhash".to_string()),
            arch: Some("gfx1151".to_string()),
            rocm: Some("7.13.26176".to_string()),
            host_profile: test_host_profile(),
        };
        let row = row_for_model(
            BatteryId::Speed,
            None,
            "ar_short_decode",
            None,
            EvalStatus::Pass,
            None,
            BTreeMap::from([
                ("tok_s".to_string(), json!(10.0)),
                (
                    "runtime_evidence_dir".to_string(),
                    json!(evidence_dir.display().to_string()),
                ),
            ]),
            &cfg,
            &ctx,
            None,
            1,
            "candidate.hfq".to_string(),
        );
        let comparison = build_comparison_artifact(&cfg, std::slice::from_ref(&row), &ctx);
        let admission =
            build_admission_artifact(&cfg, std::slice::from_ref(&row), &comparison, &ctx);
        let artifacts = write_evidence_artifacts(
            &out,
            &cfg,
            &[],
            std::slice::from_ref(&row),
            &comparison,
            &admission,
            &ctx,
        )
        .unwrap();
        assert_eq!(
            artifacts
                .get("performance")
                .and_then(|v| v.get("status"))
                .and_then(Value::as_str),
            Some("collected")
        );
        assert_eq!(
            artifacts
                .get("memory")
                .and_then(|v| v.get("status"))
                .and_then(Value::as_str),
            Some("collected")
        );
        assert_eq!(
            artifacts
                .get("launch_counts")
                .and_then(|v| v.get("status"))
                .and_then(Value::as_str),
            Some("collected")
        );
        assert_eq!(
            artifacts
                .get("moe_router_histogram")
                .and_then(|v| v.get("status"))
                .and_then(Value::as_str),
            Some("collected")
        );
        let performance: Value =
            serde_json::from_str(&fs::read_to_string(out.join("performance.json")).unwrap())
                .unwrap();
        assert!(performance["records"].as_array().unwrap().len() >= 2);
        assert!(performance["records"]
            .as_array()
            .unwrap()
            .iter()
            .any(|record| {
                record["source_path"]
                    == json!(evidence_dir.join("performance.json").display().to_string())
                    && record["hipfire_eval_context"]["git_commit"] == json!("abc123")
            }));
        let memory: Value =
            serde_json::from_str(&fs::read_to_string(out.join("memory.json")).unwrap()).unwrap();
        assert_eq!(
            memory["records"][0]["source_path"],
            json!(evidence_dir.join("memory.json").display().to_string())
        );
        let launch_counts: Value =
            serde_json::from_str(&fs::read_to_string(out.join("launch_counts.json")).unwrap())
                .unwrap();
        assert_eq!(
            launch_counts["records"][0]["metrics"]["kernel_launches"],
            json!(7)
        );
        assert_eq!(
            launch_counts["records"][0]["metrics"]["counting_scope"],
            json!("model_forward_call_proxy")
        );
        let router: Value = serde_json::from_str(
            &fs::read_to_string(out.join("moe_router_histogram.json")).unwrap(),
        )
        .unwrap();
        assert_eq!(
            router["records"][0]["metrics"]["expert_hits"]["17"],
            json!(4)
        );
        assert_eq!(
            router["records"][0]["source_path"],
            json!(evidence_dir
                .join("moe_router_histogram.json")
                .display()
                .to_string())
        );

        let _ = fs::remove_dir_all(out);
        let _ = fs::remove_dir_all(evidence_dir);
    }

    #[test]
    fn parses_dflash_spec_demo_bench_metrics() {
        let metrics = parse_bench_metrics(
            r#"
noise
=== BENCH METRICS ===
prompt_tokens: 32
prefill_secs: 0.1234
prefill_tok_s: 260.50
ttft_ms: 17.25
decode_tokens_emitted: 8
decode_secs: 0.1000
decode_tok_s: 80.00
decode_tau: 2.5000
decode_accept_rate: 0.6000
vram_used_mb: 1234
=====================
more noise
"#,
        );
        assert_eq!(metrics["prompt_tokens"], json!(32));
        assert_eq!(metrics["decode_tokens_emitted"], json!(8));
        assert_eq!(metrics["decode_tok_s"], json!(80.0));
        assert_eq!(metrics["decode_tau"], json!(2.5));
    }

    #[test]
    fn examples_executor_skips_dflash_anchor_without_draft() {
        let model = temp_path("candidate-local.hfq");
        fs::write(&model, b"placeholder").unwrap();
        let cfg = parse_args_from([
            "hipfire-eval",
            "--model",
            model.to_str().unwrap(),
            "--battery",
            "dflash",
            "--executor",
            "examples",
            "--dflash",
            "auto",
        ])
        .unwrap();
        let ctx = EvalContext {
            commit_sha: None,
            git_branch: None,
            git_describe: None,
            git_dirty: None,
            binary_hash: None,
            arch: None,
            rocm: None,
            host_profile: test_host_profile(),
        };
        let row = run_dflash_spec_demo_anchor(
            BatteryId::Dflash,
            "dflash_anchor",
            false,
            "benchmarks/prompts/dflash_resident_smoke.txt",
            &[],
            BTreeMap::new(),
            &cfg,
            &ctx,
            cfg.model.clone(),
        );
        assert_eq!(row.status, EvalStatus::Skip);
        assert!(row
            .reason
            .as_deref()
            .unwrap_or("")
            .contains("requires --draft"));
        let _ = fs::remove_file(model);
    }

    #[test]
    fn examples_dflash_ar_rows_include_configured_comparators() {
        let cfg = parse_args_from([
            "hipfire-eval",
            "--model",
            "candidate.hfq",
            "--baseline",
            "baseline.hfq",
            "--reference",
            "reference.hfq",
            "--battery",
            "dflash",
            "--executor",
            "examples",
            "--dflash",
            "auto",
        ])
        .unwrap();
        let ctx = EvalContext {
            commit_sha: None,
            git_branch: None,
            git_describe: None,
            git_dirty: None,
            binary_hash: None,
            arch: None,
            rocm: None,
            host_profile: test_host_profile(),
        };

        let rows = examples_battery_rows(BatteryId::Dflash, &cfg, &ctx, &[]).unwrap();
        let ar_rows: Vec<_> = rows
            .iter()
            .filter(|row| row.case_id == "ar_anchor")
            .collect();
        assert_eq!(ar_rows.len(), 3);
        let models: Vec<_> = ar_rows.iter().map(|row| row.model.as_str()).collect();
        assert_eq!(
            models,
            vec!["candidate.hfq", "baseline.hfq", "reference.hfq"]
        );
        assert!(ar_rows.iter().all(|row| row
            .prompt_hash
            .as_deref()
            .is_some_and(|hash| !hash.is_empty())));
        let dflash_rows: Vec<_> = rows
            .iter()
            .filter(|row| row.case_id == "dflash_anchor")
            .collect();
        assert_eq!(dflash_rows.len(), 1);
        assert_eq!(dflash_rows[0].model, "candidate.hfq");
        assert_eq!(dflash_rows[0].status, EvalStatus::Skip);
    }

    #[test]
    fn mock_executor_produces_comparable_quality_rows() {
        let cfg = parse_args_from([
            "hipfire-eval",
            "--model",
            "candidate.hfq",
            "--baseline",
            "baseline.hfq",
            "--battery",
            "quality",
            "--executor",
            "mock",
        ])
        .unwrap();
        let ctx = EvalContext {
            commit_sha: None,
            git_branch: None,
            git_describe: None,
            git_dirty: None,
            binary_hash: None,
            arch: None,
            rocm: None,
            host_profile: test_host_profile(),
        };

        let rows = run_battery(BatteryId::Quality, &cfg, &ctx, &[]);
        assert_eq!(rows.len(), 2);
        assert!(rows.iter().any(|r| r.model == "candidate.hfq"));
        assert!(rows.iter().any(|r| r.model == "baseline.hfq"));
        assert!(rows.iter().all(|r| r.status == EvalStatus::Pass));
        assert!(rows
            .iter()
            .all(|r| r.metrics.get("executor").and_then(Value::as_str) == Some("mock")));
        let comparison = build_comparison_artifact(&cfg, &rows, &ctx);
        assert_eq!(comparison.status, EvalStatus::Pass);
        assert_eq!(comparison.cases.len(), 1);
        assert!(comparison.cases[0]
            .baseline
            .as_ref()
            .unwrap()
            .metrics
            .contains_key("mean_kld"));
    }

    #[test]
    fn quality_json_ingest_produces_comparable_rows() {
        let quality_json = temp_path("quality-result-data.json");
        fs::write(
            &quality_json,
            serde_json::to_string(&json!([
                {
                    "variant": "candidate-v",
                    "arch": "gfx1151",
                    "scoring_mode": "prefill-q8-c1",
                    "n_chunks": 1,
                    "mean_kld": 0.12,
                    "mean_kld_ci_lo": 0.10,
                    "mean_kld_ci_hi": 0.14,
                    "p99_kld": 0.9,
                    "ppl": 8.5,
                    "notes": ""
                },
                {
                    "variant": "baseline-v",
                    "arch": "gfx1151",
                    "scoring_mode": "prefill-q8-c1",
                    "n_chunks": 1,
                    "mean_kld": 0.20,
                    "mean_kld_ci_lo": 0.18,
                    "mean_kld_ci_hi": 0.22,
                    "p99_kld": 1.2,
                    "ppl": 9.1,
                    "notes": "control"
                }
            ]))
            .unwrap(),
        )
        .unwrap();
        let cfg = parse_args_from([
            "hipfire-eval",
            "--model",
            "candidate.hfq",
            "--baseline",
            "baseline.hfq",
            "--battery",
            "quality",
            "--quality-json",
            quality_json.to_str().unwrap(),
            "--candidate-variant",
            "candidate-v",
            "--baseline-variant",
            "baseline-v",
        ])
        .unwrap();
        let ctx = EvalContext {
            commit_sha: None,
            git_branch: None,
            git_describe: None,
            git_dirty: None,
            binary_hash: None,
            arch: None,
            rocm: None,
            host_profile: test_host_profile(),
        };

        let rows = run_battery(BatteryId::Quality, &cfg, &ctx, &[]);
        assert_eq!(rows.len(), 2);
        assert!(rows.iter().all(|r| r.status == EvalStatus::Pass));
        assert!(rows.iter().all(|r| {
            r.metrics.get("executor").and_then(Value::as_str) == Some("quality_json")
        }));
        let comparison = build_comparison_artifact(&cfg, &rows, &ctx);
        assert_eq!(comparison.status, EvalStatus::Pass);
        assert_eq!(
            comparison.cases[0]
                .baseline
                .as_ref()
                .unwrap()
                .metrics
                .get("mean_kld")
                .unwrap()
                .direction,
            "improved"
        );

        let _ = fs::remove_file(quality_json);
    }

    #[test]
    fn quality_json_ingest_rejects_invalid_metrics() {
        let quality_json = temp_path("quality-invalid-result-data.json");
        fs::write(
            &quality_json,
            serde_json::to_string(&json!([
                {
                    "variant": "candidate-v",
                    "arch": "gfx1151",
                    "scoring_mode": "prefill-q8-c1",
                    "n_chunks": 1,
                    "mean_kld": -0.01,
                    "mean_kld_ci_lo": 0.0,
                    "mean_kld_ci_hi": 0.1,
                    "p99_kld": 0.2,
                    "ppl": 8.5
                }
            ]))
            .unwrap(),
        )
        .unwrap();
        let cfg = parse_args_from([
            "hipfire-eval",
            "--model",
            "candidate.hfq",
            "--battery",
            "quality",
            "--quality-json",
            quality_json.to_str().unwrap(),
            "--candidate-variant",
            "candidate-v",
        ])
        .unwrap();
        let ctx = EvalContext {
            commit_sha: None,
            git_branch: None,
            git_describe: None,
            git_dirty: None,
            binary_hash: None,
            arch: None,
            rocm: None,
            host_profile: test_host_profile(),
        };

        let rows = run_battery(BatteryId::Quality, &cfg, &ctx, &[]);
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].status, EvalStatus::Skip);
        assert_eq!(rows[0].case_id, "quality_json_ingest");
        assert!(rows[0]
            .reason
            .as_deref()
            .unwrap_or("")
            .contains("negative mean_kld"));
        let comparison = build_comparison_artifact(&cfg, &rows, &ctx);
        assert_eq!(comparison.status, EvalStatus::Skip);

        let _ = fs::remove_file(quality_json);
    }

    #[test]
    fn performance_json_ingest_produces_comparable_speed_rows() {
        let perf_json = temp_path("performance-result-data.json");
        fs::write(
            &perf_json,
            serde_json::to_string(&json!([
                {
                    "tag": "candidate-v",
                    "mode": "off",
                    "parsed": {
                        "decode_tokS": 140.0,
                        "ttft_ms": 40.0,
                        "prefill_user_tokS": 500.0,
                        "pp128_tokS": 610.0
                    }
                },
                {
                    "tag": "baseline-v",
                    "mode": "off",
                    "parsed": {
                        "decode_tokS": 110.0,
                        "ttft_ms": 50.0,
                        "prefill_user_tokS": 480.0,
                        "pp128_tokS": 590.0
                    }
                }
            ]))
            .unwrap(),
        )
        .unwrap();
        let cfg = parse_args_from([
            "hipfire-eval",
            "--model",
            "candidate.hfq",
            "--baseline",
            "baseline.hfq",
            "--battery",
            "speed",
            "--performance-json",
            perf_json.to_str().unwrap(),
            "--candidate-variant",
            "candidate-v:off",
            "--baseline-variant",
            "baseline-v:off",
        ])
        .unwrap();
        let ctx = EvalContext {
            commit_sha: None,
            git_branch: None,
            git_describe: None,
            git_dirty: None,
            binary_hash: None,
            arch: None,
            rocm: None,
            host_profile: test_host_profile(),
        };

        let rows = run_battery(BatteryId::Speed, &cfg, &ctx, &[]);
        assert_eq!(rows.len(), 2);
        assert!(rows.iter().all(|r| r.status == EvalStatus::Pass));
        assert_eq!(rows[0].metrics.get("tok_s"), Some(&json!(140.0)));
        assert_eq!(rows[0].metrics.get("prefill_tok_s"), Some(&json!(500.0)));
        let comparison = build_comparison_artifact(&cfg, &rows, &ctx);
        assert_eq!(comparison.status, EvalStatus::Pass);
        assert_eq!(
            comparison.cases[0]
                .baseline
                .as_ref()
                .unwrap()
                .metrics
                .get("tok_s")
                .unwrap()
                .direction,
            "improved"
        );
        assert_eq!(
            comparison.cases[0]
                .baseline
                .as_ref()
                .unwrap()
                .metrics
                .get("ttft_ms")
                .unwrap()
                .direction,
            "improved"
        );

        let _ = fs::remove_file(perf_json);
    }

    #[test]
    fn performance_json_ingest_feeds_timing_and_memory_artifacts() {
        let out = temp_path("performance-runtime-artifacts");
        let perf_json = temp_path("performance-runtime-result-data.json");
        fs::create_dir_all(&out).unwrap();
        fs::write(
            &perf_json,
            serde_json::to_string(&json!({
                "performance_rows": [
                    {
                        "variant": "candidate",
                        "tok_s": 140.0,
                        "prefill_secs": 0.1234,
                        "decode_secs": 0.1000,
                        "ttft_ms": 17.25,
                        "vram_used_mb": 1234.0
                    }
                ]
            }))
            .unwrap(),
        )
        .unwrap();
        let cfg = parse_args_from([
            "hipfire-eval",
            "--model",
            "candidate.hfq",
            "--battery",
            "speed",
            "--performance-json",
            perf_json.to_str().unwrap(),
            "--candidate-variant",
            "candidate",
        ])
        .unwrap();
        let ctx = EvalContext {
            commit_sha: None,
            git_branch: None,
            git_describe: None,
            git_dirty: None,
            binary_hash: None,
            arch: None,
            rocm: None,
            host_profile: test_host_profile(),
        };

        let rows = run_battery(BatteryId::Speed, &cfg, &ctx, &[]);
        assert_eq!(rows.len(), 1);
        let comparison = build_comparison_artifact(&cfg, &rows, &ctx);
        let admission = build_admission_artifact(&cfg, &rows, &comparison, &ctx);
        let artifacts =
            write_evidence_artifacts(&out, &cfg, &[], &rows, &comparison, &admission, &ctx)
                .unwrap();
        assert_eq!(
            artifacts
                .get("phase_timings")
                .and_then(|v| v.get("status"))
                .and_then(Value::as_str),
            Some("collected")
        );
        assert_eq!(
            artifacts
                .get("memory")
                .and_then(|v| v.get("status"))
                .and_then(Value::as_str),
            Some("collected")
        );
        let phase_timings: Value =
            serde_json::from_str(&fs::read_to_string(out.join("phase_timings.json")).unwrap())
                .unwrap();
        assert_eq!(
            phase_timings["records"][0]["metrics"]["prefill_ms"],
            json!(123.4)
        );
        assert_eq!(
            phase_timings["records"][0]["metrics"]["decode_ms"],
            json!(100.0)
        );
        let memory: Value =
            serde_json::from_str(&fs::read_to_string(out.join("memory.json")).unwrap()).unwrap();
        assert_eq!(
            memory["records"][0]["metrics"]["vram_peak_bytes"],
            json!(1234.0 * 1024.0 * 1024.0)
        );

        let _ = fs::remove_dir_all(out);
        let _ = fs::remove_file(perf_json);
    }

    #[test]
    fn performance_json_ingest_rejects_invalid_metrics() {
        let perf_json = temp_path("performance-invalid-result-data.json");
        fs::write(
            &perf_json,
            serde_json::to_string(&json!([
                {
                    "tag": "candidate-v",
                    "mode": "off",
                    "parsed": {
                        "decode_tokS": -1.0,
                        "ttft_ms": 40.0
                    }
                }
            ]))
            .unwrap(),
        )
        .unwrap();
        let cfg = parse_args_from([
            "hipfire-eval",
            "--model",
            "candidate.hfq",
            "--battery",
            "speed",
            "--performance-json",
            perf_json.to_str().unwrap(),
            "--candidate-variant",
            "candidate-v:off",
        ])
        .unwrap();
        let ctx = EvalContext {
            commit_sha: None,
            git_branch: None,
            git_describe: None,
            git_dirty: None,
            binary_hash: None,
            arch: None,
            rocm: None,
            host_profile: test_host_profile(),
        };

        let rows = run_battery(BatteryId::Speed, &cfg, &ctx, &[]);
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].status, EvalStatus::Skip);
        assert_eq!(rows[0].case_id, "performance_json_ingest");
        assert!(rows[0]
            .reason
            .as_deref()
            .unwrap_or("")
            .contains("negative tok_s"));
        let comparison = build_comparison_artifact(&cfg, &rows, &ctx);
        assert_eq!(comparison.status, EvalStatus::Skip);

        let _ = fs::remove_file(perf_json);
    }

    #[test]
    fn speed_baseline_parser_matches_model_size_and_prefill() {
        let path = temp_path("speed-baseline.json");
        fs::write(
            &path,
            serde_json::to_vec_pretty(&json!({
                "schema": "hipfire.perf_baseline.v1",
                "arch": "gfx1151",
                "tolerance_pct": 5.0,
                "baselines": {
                    "speed": [
                        {
                            "label": "4b_pp32_prefill_decode",
                            "model_id": "qwen3.5-4b-mq4",
                            "model_size": "4b",
                            "format": "mq4",
                            "prefill_tokens": 32,
                            "prefill_tok_s": 590.7,
                            "gen_tok_s": 65.5
                        }
                    ]
                }
            }))
            .unwrap(),
        )
        .unwrap();

        let baseline = load_speed_baseline(&path, "qwen3.5-4b-mq4", "4b", 32)
            .unwrap()
            .unwrap();
        assert_eq!(baseline.label, "4b_pp32_prefill_decode");
        assert_eq!(baseline.model_id, "qwen3.5-4b-mq4");
        assert_eq!(baseline.format, "mq4");
        assert_eq!(baseline.prefill_tok_s, Some(590.7));
        assert_eq!(baseline.gen_tok_s, Some(65.5));
        assert_eq!(baseline.tolerance_pct, 5.0);
        assert!(load_speed_baseline(&path, "qwen3.5-4b-bf16", "4b", 32)
            .unwrap()
            .is_none());
        assert!(load_speed_baseline(&path, "qwen3.5-9b-mq4", "9b", 32)
            .unwrap()
            .is_none());

        let _ = fs::remove_file(path);
    }

    #[test]
    fn speed_model_size_is_format_neutral() {
        assert_eq!(
            speed_model_size("qwen3.5-0.8b-bf16.hfq"),
            Some("0.8b".to_string())
        );
        assert_eq!(
            speed_model_size("/models/qwen3.5-35b-a3b-awq-mq4.hfq"),
            Some("35b-a3b".to_string())
        );
        assert_eq!(
            speed_model_id("/models/qwen3.5-0.8b-bf16.hfq"),
            "qwen3.5-0.8b-bf16"
        );
    }

    #[test]
    fn speed_baseline_check_fails_below_floor() {
        let _guard = env_lock();
        let path = temp_path("speed-baseline-check.json");
        fs::write(
            &path,
            serde_json::to_vec_pretty(&json!({
                "schema": "hipfire.perf_baseline.v1",
                "arch": "gfx1151",
                "tolerance_pct": 5.0,
                "baselines": {
                    "speed": [
                        {
                            "label": "4b_pp32_prefill_decode",
                            "model_id": "qwen3.5-4b-mq4",
                            "model_size": "4b",
                            "format": "mq4",
                            "prefill_tokens": 32,
                            "prefill_tok_s": 100.0,
                            "gen_tok_s": 50.0
                        }
                    ]
                }
            }))
            .unwrap(),
        )
        .unwrap();
        let _baseline = ScopedEnv::set("HIPFIRE_PERF_BASELINE", &path);
        let ctx = EvalContext {
            commit_sha: None,
            git_branch: None,
            git_describe: None,
            git_dirty: None,
            binary_hash: None,
            arch: Some("gfx1151".to_string()),
            rocm: None,
            host_profile: test_host_profile(),
        };
        let mut metrics = BTreeMap::from([
            ("prefill_tok_s".to_string(), json!(94.0)),
            ("gen_tok_s".to_string(), json!(51.0)),
        ]);

        let check = apply_speed_baseline(&mut metrics, "qwen3.5-4b-mq4.hfq", 32, &ctx).unwrap();
        assert!(check.failed);
        assert_eq!(check.error, None);
        assert_eq!(metrics["perf_baseline_status"], json!("compared"));
        assert_eq!(metrics["perf_baseline_failed"], json!(true));
        assert_eq!(metrics["baseline_prefill_floor_tok_s"], json!(95.0));

        let _ = fs::remove_file(path);
    }

    #[test]
    fn compares_candidate_against_baseline_metrics() {
        let cfg = EvalConfig {
            model: "candidate.hfq".to_string(),
            draft: None,
            baseline: Some("baseline.hfq".to_string()),
            reference: None,
            tier: EvalTier::Fast,
            batteries: vec![BatteryId::Quality],
            suites: vec![],
            out_dir: PathBuf::from("out"),
            kv_mode: None,
            max_tokens: 8,
            dflash: DflashMode::Off,
            profile: ProfileMode::Off,
            quality_max_chunks: Some(1),
            kldref: None,
            quality_json: None,
            performance_json: None,
            evidence_json: Vec::new(),
            evidence_dirs: Vec::new(),
            candidate_variant: None,
            baseline_variant: None,
            reference_variant: None,
            performance_candidate_variant: None,
            performance_baseline_variant: None,
            performance_reference_variant: None,
            executor: EvalExecutorMode::None,
            fetch_datasets: false,
            offline: true,
            dataset_cache: PathBuf::from("datasets"),
            result_cache: PathBuf::from("cache"),
            cache_mode: EvalCacheMode::Use,
            runs: 1,
            warmup_runs: 0,
            benchmark: false,
            host_memory_class: None,
            host_memory_width_bits: None,
            host_memory_bandwidth_gbps: None,
            fail_on_admission: false,
            models_spec: None,
            dry_run: false,
            status: false,
            fetch: false,
        };
        let ctx = EvalContext {
            commit_sha: None,
            git_branch: None,
            git_describe: None,
            git_dirty: None,
            binary_hash: None,
            arch: None,
            rocm: None,
            host_profile: test_host_profile(),
        };
        let mut candidate_metrics = BTreeMap::new();
        candidate_metrics.insert("mean_kld".to_string(), json!(0.04));
        candidate_metrics.insert("tok_s".to_string(), json!(120.0));
        let candidate = row(
            BatteryId::Quality,
            None,
            "canary",
            None,
            EvalStatus::Pass,
            None,
            candidate_metrics,
            &cfg,
            &ctx,
            None,
            0,
        );

        let mut baseline_cfg = cfg.clone();
        baseline_cfg.model = "baseline.hfq".to_string();
        let mut baseline_metrics = BTreeMap::new();
        baseline_metrics.insert("mean_kld".to_string(), json!(0.05));
        baseline_metrics.insert("tok_s".to_string(), json!(100.0));
        let baseline = row(
            BatteryId::Quality,
            None,
            "canary",
            None,
            EvalStatus::Pass,
            None,
            baseline_metrics,
            &baseline_cfg,
            &ctx,
            None,
            0,
        );

        let artifact = build_comparison_artifact(&cfg, &[candidate, baseline], &ctx);
        assert_eq!(artifact.status, EvalStatus::Pass);
        assert_eq!(artifact.cases.len(), 1);
        let metrics = &artifact.cases[0].baseline.as_ref().unwrap().metrics;
        assert_eq!(metrics["mean_kld"].direction, "improved");
        assert_eq!(metrics["tok_s"].direction, "improved");
        assert!((metrics["mean_kld"].delta + 0.01).abs() < 1e-9);
        assert!((metrics["tok_s"].delta - 20.0).abs() < 1e-9);
    }

    #[test]
    fn compares_candidate_against_reference_without_baseline() {
        let cfg = parse_args_from([
            "hipfire-eval",
            "--model",
            "candidate.hfq",
            "--reference",
            "reference-bf16",
            "--battery",
            "quality",
        ])
        .unwrap();
        let ctx = EvalContext {
            commit_sha: None,
            git_branch: None,
            git_describe: None,
            git_dirty: None,
            binary_hash: None,
            arch: None,
            rocm: None,
            host_profile: test_host_profile(),
        };
        let candidate = row_for_model(
            BatteryId::Quality,
            None,
            "canary",
            None,
            EvalStatus::Pass,
            None,
            BTreeMap::from([("mean_kld".to_string(), json!(0.04))]),
            &cfg,
            &ctx,
            None,
            0,
            "candidate.hfq".to_string(),
        );
        let reference = row_for_model(
            BatteryId::Quality,
            None,
            "canary",
            None,
            EvalStatus::Pass,
            None,
            BTreeMap::from([("mean_kld".to_string(), json!(0.02))]),
            &cfg,
            &ctx,
            None,
            0,
            "reference-bf16".to_string(),
        );

        let artifact = build_comparison_artifact(&cfg, &[candidate, reference], &ctx);
        assert_eq!(artifact.status, EvalStatus::Pass);
        assert_eq!(artifact.baseline, None);
        assert_eq!(artifact.reference.as_deref(), Some("reference-bf16"));
        assert_eq!(artifact.cases.len(), 1);
        assert!(artifact.cases[0].baseline.is_none());
        let metrics = &artifact.cases[0].reference.as_ref().unwrap().metrics;
        assert_eq!(metrics["mean_kld"].direction, "regressed");
        assert!((metrics["mean_kld"].delta - 0.02).abs() < 1e-9);
    }

    #[test]
    fn admission_promotes_when_quality_and_speed_are_clean() {
        let cfg = parse_args_from([
            "hipfire-eval",
            "--model",
            "candidate.hfq",
            "--baseline",
            "baseline.hfq",
            "--battery",
            "quality,speed",
        ])
        .unwrap();
        let ctx = EvalContext {
            commit_sha: None,
            git_branch: None,
            git_describe: None,
            git_dirty: None,
            binary_hash: None,
            arch: None,
            rocm: None,
            host_profile: test_host_profile(),
        };
        let rows = vec![
            row_for_model(
                BatteryId::Quality,
                None,
                "canary",
                None,
                EvalStatus::Pass,
                None,
                BTreeMap::from([("mean_kld".to_string(), json!(0.04))]),
                &cfg,
                &ctx,
                None,
                0,
                "candidate.hfq".to_string(),
            ),
            row_for_model(
                BatteryId::Quality,
                None,
                "canary",
                None,
                EvalStatus::Pass,
                None,
                BTreeMap::from([("mean_kld".to_string(), json!(0.05))]),
                &cfg,
                &ctx,
                None,
                0,
                "baseline.hfq".to_string(),
            ),
            row_for_model(
                BatteryId::Speed,
                None,
                "decode",
                None,
                EvalStatus::Pass,
                None,
                BTreeMap::from([("tok_s".to_string(), json!(120.0))]),
                &cfg,
                &ctx,
                None,
                0,
                "candidate.hfq".to_string(),
            ),
            row_for_model(
                BatteryId::Speed,
                None,
                "decode",
                None,
                EvalStatus::Pass,
                None,
                BTreeMap::from([("tok_s".to_string(), json!(100.0))]),
                &cfg,
                &ctx,
                None,
                0,
                "baseline.hfq".to_string(),
            ),
        ];
        let comparison = build_comparison_artifact(&cfg, &rows, &ctx);
        let admission = build_admission_artifact(&cfg, &rows, &comparison, &ctx);
        assert_eq!(admission.status, EvalStatus::Pass);
        assert_eq!(admission.verdict, "promote");
        assert!(admission.findings.is_empty());
    }

    #[test]
    fn admission_measures_speed_without_compare() {
        let cfg = parse_args_from(["hipfire-eval", "candidate.hfq", "--battery", "speed"]).unwrap();
        let ctx = EvalContext {
            commit_sha: None,
            git_branch: None,
            git_describe: None,
            git_dirty: None,
            binary_hash: None,
            arch: None,
            rocm: None,
            host_profile: test_host_profile(),
        };
        let rows = vec![row_for_model(
            BatteryId::Speed,
            None,
            "decode",
            None,
            EvalStatus::Pass,
            None,
            BTreeMap::from([("tok_s".to_string(), json!(120.0))]),
            &cfg,
            &ctx,
            None,
            0,
            "candidate.hfq".to_string(),
        )];
        let comparison = build_comparison_artifact(&cfg, &rows, &ctx);
        let admission = build_admission_artifact(&cfg, &rows, &comparison, &ctx);
        assert_eq!(admission.status, EvalStatus::Pass);
        assert_eq!(admission.verdict, "measured");
        assert!(admission
            .reason
            .as_deref()
            .unwrap_or("")
            .contains("no --compare or --reference"));
    }

    #[test]
    fn admission_accepts_barrage_accuracy_as_quality_evidence() {
        let cfg = parse_args_from([
            "hipfire-eval",
            "--model",
            "candidate.hfq",
            "--baseline",
            "baseline.hfq",
            "--battery",
            "barrage,speed",
            "--suite",
            "lm_eval_micro",
        ])
        .unwrap();
        let ctx = EvalContext {
            commit_sha: None,
            git_branch: None,
            git_describe: None,
            git_dirty: None,
            binary_hash: None,
            arch: None,
            rocm: None,
            host_profile: test_host_profile(),
        };
        let rows = vec![
            row_for_model(
                BatteryId::Barrage,
                Some(SuiteId::LmEvalMicro),
                "lm_eval_micro_zero_shot_native",
                Some("arc_easy:0".to_string()),
                EvalStatus::Pass,
                None,
                BTreeMap::from([
                    ("accuracy".to_string(), json!(1.0)),
                    ("exact_match".to_string(), json!(1.0)),
                ]),
                &cfg,
                &ctx,
                None,
                0,
                "candidate.hfq".to_string(),
            ),
            row_for_model(
                BatteryId::Barrage,
                Some(SuiteId::LmEvalMicro),
                "lm_eval_micro_zero_shot_native",
                Some("arc_easy:0".to_string()),
                EvalStatus::Pass,
                None,
                BTreeMap::from([
                    ("accuracy".to_string(), json!(1.0)),
                    ("exact_match".to_string(), json!(1.0)),
                ]),
                &cfg,
                &ctx,
                None,
                0,
                "baseline.hfq".to_string(),
            ),
            row_for_model(
                BatteryId::Speed,
                None,
                "decode",
                None,
                EvalStatus::Pass,
                None,
                BTreeMap::from([("tok_s".to_string(), json!(120.0))]),
                &cfg,
                &ctx,
                None,
                0,
                "candidate.hfq".to_string(),
            ),
            row_for_model(
                BatteryId::Speed,
                None,
                "decode",
                None,
                EvalStatus::Pass,
                None,
                BTreeMap::from([("tok_s".to_string(), json!(100.0))]),
                &cfg,
                &ctx,
                None,
                0,
                "baseline.hfq".to_string(),
            ),
        ];

        let comparison = build_comparison_artifact(&cfg, &rows, &ctx);
        let admission = build_admission_artifact(&cfg, &rows, &comparison, &ctx);
        assert_eq!(admission.status, EvalStatus::Pass);
        assert_eq!(admission.verdict, "promote");
        assert!(admission
            .required_evidence
            .iter()
            .any(|e| e.kind == "quality" && e.status == EvalStatus::Pass && e.rows == 2));
        assert!(admission
            .required_evidence
            .iter()
            .any(|e| e.kind == "barrage" && e.status == EvalStatus::Pass && e.rows == 2));
    }

    #[test]
    fn admission_records_observed_runtime_evidence_without_requiring_it() {
        let cfg = parse_args_from([
            "hipfire-eval",
            "--model",
            "candidate.hfq",
            "--baseline",
            "baseline.hfq",
            "--battery",
            "quality,speed",
            "--profile",
            "passive",
        ])
        .unwrap();
        let ctx = EvalContext {
            commit_sha: None,
            git_branch: None,
            git_describe: None,
            git_dirty: None,
            binary_hash: None,
            arch: None,
            rocm: None,
            host_profile: test_host_profile(),
        };
        let rows = vec![
            row_for_model(
                BatteryId::Quality,
                None,
                "canary",
                None,
                EvalStatus::Pass,
                None,
                BTreeMap::from([("mean_kld".to_string(), json!(0.04))]),
                &cfg,
                &ctx,
                None,
                0,
                "candidate.hfq".to_string(),
            ),
            row_for_model(
                BatteryId::Quality,
                None,
                "canary",
                None,
                EvalStatus::Pass,
                None,
                BTreeMap::from([("mean_kld".to_string(), json!(0.05))]),
                &cfg,
                &ctx,
                None,
                0,
                "baseline.hfq".to_string(),
            ),
            row_for_model(
                BatteryId::Speed,
                None,
                "decode",
                None,
                EvalStatus::Pass,
                None,
                BTreeMap::from([
                    ("tok_s".to_string(), json!(120.0)),
                    ("kernel_launches".to_string(), json!(42)),
                    ("router_entropy".to_string(), json!(0.73)),
                    ("duration_us".to_string(), json!(912.5)),
                ]),
                &cfg,
                &ctx,
                None,
                12,
                "candidate.hfq".to_string(),
            ),
            row_for_model(
                BatteryId::Speed,
                None,
                "decode",
                None,
                EvalStatus::Pass,
                None,
                BTreeMap::from([("tok_s".to_string(), json!(100.0))]),
                &cfg,
                &ctx,
                None,
                0,
                "baseline.hfq".to_string(),
            ),
        ];
        let comparison = build_comparison_artifact(&cfg, &rows, &ctx);
        let admission = build_admission_artifact(&cfg, &rows, &comparison, &ctx);
        assert_eq!(admission.status, EvalStatus::Pass);
        assert_eq!(admission.verdict, "promote");
        assert_eq!(
            admission
                .observed_evidence
                .iter()
                .find(|e| e.kind == "phase_timings")
                .unwrap()
                .status,
            EvalStatus::Pass
        );
        assert_eq!(
            admission
                .observed_evidence
                .iter()
                .find(|e| e.kind == "launch_counts")
                .unwrap()
                .rows,
            1
        );
        assert_eq!(
            admission
                .observed_evidence
                .iter()
                .find(|e| e.kind == "moe_router_histogram")
                .unwrap()
                .status,
            EvalStatus::Pass
        );
        assert_eq!(
            admission
                .observed_evidence
                .iter()
                .find(|e| e.kind == "profiling")
                .unwrap()
                .status,
            EvalStatus::Pass
        );
    }

    #[test]
    fn admission_rejects_quality_regression() {
        let cfg = parse_args_from([
            "hipfire-eval",
            "--model",
            "candidate.hfq",
            "--baseline",
            "baseline.hfq",
            "--battery",
            "quality,speed",
        ])
        .unwrap();
        let ctx = EvalContext {
            commit_sha: None,
            git_branch: None,
            git_describe: None,
            git_dirty: None,
            binary_hash: None,
            arch: None,
            rocm: None,
            host_profile: test_host_profile(),
        };
        let rows = vec![
            row_for_model(
                BatteryId::Quality,
                None,
                "canary",
                None,
                EvalStatus::Pass,
                None,
                BTreeMap::from([("mean_kld".to_string(), json!(0.08))]),
                &cfg,
                &ctx,
                None,
                0,
                "candidate.hfq".to_string(),
            ),
            row_for_model(
                BatteryId::Quality,
                None,
                "canary",
                None,
                EvalStatus::Pass,
                None,
                BTreeMap::from([("mean_kld".to_string(), json!(0.05))]),
                &cfg,
                &ctx,
                None,
                0,
                "baseline.hfq".to_string(),
            ),
            row_for_model(
                BatteryId::Speed,
                None,
                "decode",
                None,
                EvalStatus::Pass,
                None,
                BTreeMap::from([("tok_s".to_string(), json!(120.0))]),
                &cfg,
                &ctx,
                None,
                0,
                "candidate.hfq".to_string(),
            ),
            row_for_model(
                BatteryId::Speed,
                None,
                "decode",
                None,
                EvalStatus::Pass,
                None,
                BTreeMap::from([("tok_s".to_string(), json!(100.0))]),
                &cfg,
                &ctx,
                None,
                0,
                "baseline.hfq".to_string(),
            ),
        ];
        let comparison = build_comparison_artifact(&cfg, &rows, &ctx);
        let admission = build_admission_artifact(&cfg, &rows, &comparison, &ctx);
        assert_eq!(admission.status, EvalStatus::Fail);
        assert_eq!(admission.verdict, "reject");
        assert_eq!(admission.findings[0].severity, "reject");
        assert_eq!(admission.findings[0].metric, "mean_kld");
    }

    #[test]
    fn fail_on_admission_returns_error_after_writing_artifacts() {
        let out = temp_path("fail-on-admission");
        let quality_json = temp_path("fail-on-admission-quality.json");
        let performance_json = temp_path("fail-on-admission-performance.json");
        let _ = fs::remove_dir_all(&out);
        fs::write(
            &quality_json,
            serde_json::to_string(&json!({
                "quality_rows": [
                    {
                        "variant": "candidate",
                        "arch": "gfx1151",
                        "scoring_mode": "kld_reference_slice",
                        "n_chunks": 1,
                        "mean_kld": 0.08,
                        "mean_kld_ci_lo": 0.07,
                        "mean_kld_ci_hi": 0.09,
                        "p99_kld": 0.12,
                        "ppl": 6.0
                    },
                    {
                        "variant": "baseline",
                        "arch": "gfx1151",
                        "scoring_mode": "kld_reference_slice",
                        "n_chunks": 1,
                        "mean_kld": 0.05,
                        "mean_kld_ci_lo": 0.04,
                        "mean_kld_ci_hi": 0.06,
                        "p99_kld": 0.08,
                        "ppl": 5.0
                    }
                ]
            }))
            .unwrap(),
        )
        .unwrap();
        fs::write(
            &performance_json,
            serde_json::to_string(&json!({
                "performance_rows": [
                    {"variant": "candidate", "tok_s": 120.0, "ttft_ms": 20.0},
                    {"variant": "baseline", "tok_s": 100.0, "ttft_ms": 22.0}
                ]
            }))
            .unwrap(),
        )
        .unwrap();
        let cfg = parse_args_from([
            "hipfire-eval",
            "--model",
            "candidate.hfq",
            "--baseline",
            "baseline.hfq",
            "--battery",
            "quality,speed",
            "--quality-json",
            quality_json.to_str().unwrap(),
            "--performance-json",
            performance_json.to_str().unwrap(),
            "--out",
            out.to_str().unwrap(),
            "--fail-on-admission",
        ])
        .unwrap();

        let err = run_eval(cfg).unwrap_err();
        assert!(err.contains("admission verdict reject"));
        assert!(err.contains("artifacts written to"));
        assert!(out.join("manifest.json").exists());
        assert!(out.join("results.jsonl").exists());
        assert!(out.join("summary.md").exists());
        let manifest: Value =
            serde_json::from_str(&fs::read_to_string(out.join("manifest.json")).unwrap()).unwrap();
        assert_eq!(manifest["runner"], "hipfire-eval");
        assert!(manifest["runner_version"].as_str().is_some());
        assert!(manifest["hipfire_version"].as_str().is_some());
        let admission: Value = serde_json::from_str(
            &fs::read_to_string(out.join("artifacts/admission.json")).unwrap(),
        )
        .unwrap();
        assert_eq!(admission["verdict"], "reject");
        assert_eq!(admission["status"], "fail");
        assert_eq!(admission["findings"][0]["severity"], "reject");

        let _ = fs::remove_dir_all(out);
        let _ = fs::remove_file(quality_json);
        let _ = fs::remove_file(performance_json);
    }

    #[test]
    fn comparison_skips_when_matching_baseline_rows_are_absent() {
        let cfg = parse_args_from([
            "hipfire-eval",
            "--model",
            "candidate.hfq",
            "--baseline",
            "baseline.hfq",
        ])
        .unwrap();
        let ctx = EvalContext {
            commit_sha: None,
            git_branch: None,
            git_describe: None,
            git_dirty: None,
            binary_hash: None,
            arch: None,
            rocm: None,
            host_profile: test_host_profile(),
        };
        let artifact = build_comparison_artifact(&cfg, &[], &ctx);
        assert_eq!(artifact.status, EvalStatus::Skip);
        assert!(artifact.reason.unwrap().contains("matching baseline"));
    }

    #[test]
    fn comparison_skips_when_no_comparator_is_requested() {
        let cfg = parse_args_from(["hipfire-eval", "--model", "candidate.hfq"]).unwrap();
        let ctx = EvalContext {
            commit_sha: None,
            git_branch: None,
            git_describe: None,
            git_dirty: None,
            binary_hash: None,
            arch: None,
            rocm: None,
            host_profile: test_host_profile(),
        };
        let artifact = build_comparison_artifact(&cfg, &[], &ctx);
        assert_eq!(artifact.status, EvalStatus::Skip);
        assert!(artifact
            .reason
            .as_deref()
            .unwrap_or("")
            .contains("no --compare or --reference"));
    }
}
