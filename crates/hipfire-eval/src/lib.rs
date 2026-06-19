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
use std::ffi::{c_void, CString, OsStr};
use std::fs::{self, File, OpenOptions};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

use hipfire_evidence::{
    admission_artifact_index_entry_json, admission_artifact_json, admission_metric_is_quality,
    admission_verdict_policy, classify_hardware_kind, comparison_artifact_index_entry_json,
    comparison_artifact_json, compute_peak_bandwidth_gbps, dflash_trace_metrics, directory_hash,
    evidence_artifact_index_entry_from_value_json, evidence_artifact_index_entry_json,
    evidence_artifact_json, evidence_collection_policy, evidence_metric_direction,
    evidence_record_json, extract_external_evidence_records_json, hardware_bucket,
    has_launch_count_metric, has_memory_metric, has_module_evidence_metric, has_moe_router_metric,
    has_path_c_trace_metric, has_performance_metric, has_phase_timing_metric, has_profiling_metric,
    has_quality_metric as has_quality_signal_metric, host_profile_artifact_index_entry_json,
    host_profile_hash, launch_count_metrics, list_files, memory_metrics, module_evidence_metrics,
    moe_router_metrics, path_c_trace_metrics, phase_timing_metrics, profiling_metrics,
    prompt_artifact_index_entry_json, required_admission_evidence_requirements,
    run_metadata_artifact_json, run_provenance_json, standard_evidence_paths_in_dir,
    AdmissionArtifact as EvidenceAdmissionArtifact, AdmissionEvidence as EvidenceAdmissionEvidence,
    ComparisonArtifact as EvidenceComparisonArtifact, EvalStatus, EvidenceArtifact,
    EvidenceArtifactCollection, EvidenceArtifactConfig, EvidenceArtifactDatasetStatus,
    EvidenceArtifactIndexContext, EvidenceArtifactModels, EvidenceRecord, HostProfile,
    RunMetadataArtifact, RunMetadataConfig, RunMetadataModels, RunProvenance, SourcedField,
    OBSERVED_ADMISSION_EVIDENCE_KINDS, STANDARD_EVIDENCE_ARTIFACT_SPECS,
};
use hipfire_generate::{GenerateTextRequest, GenerationSamplingPolicy};
use hipfire_hash::{file_hash, stable_hash_bytes, stable_hash_file_fallback};
use hipfire_model::{
    discover_dflash_draft_for_model, model_artifact_stem, model_hash, model_manifest_entry,
    ModelLoadParams, ModelLoadedResponse, ModelManifestEntry,
};

mod datasets;
use datasets::*;
mod evidence;
use evidence::*;
mod quality;
use quality::*;
mod rocprof;
use rocprof::*;

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

#[derive(Debug, Clone, Default)]
struct HostProfileOverrides {
    memory_class: Option<String>,
    memory_width_bits: Option<u32>,
    memory_bandwidth_gbps: Option<f64>,
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

pub fn parse_args_from<I, S>(args: I) -> Result<EvalConfig, String>
where
    I: IntoIterator<Item = S>,
    S: Into<String>,
{
    let argv: Vec<String> = args.into_iter().map(Into::into).collect();
    let mut model: Option<String> = None;
    let mut draft: Option<String> = None;
    let mut baseline: Option<String> = None;
    let mut reference: Option<String> = None;
    let mut tier = EvalTier::Fast;
    let mut batteries: Option<Vec<BatteryId>> = None;
    let mut suites: Vec<SuiteId> = Vec::new();
    let mut out_dir: Option<PathBuf> = None;
    let mut kv_mode: Option<String> = None;
    let mut max_tokens = 64usize;
    let mut dflash = DflashMode::Off;
    let mut profile = ProfileMode::Off;
    let mut quality_max_chunks: Option<usize> = None;
    let mut kldref: Option<PathBuf> = None;
    let mut quality_json: Option<PathBuf> = None;
    let mut performance_json: Option<PathBuf> = None;
    let mut evidence_json: Vec<PathBuf> = Vec::new();
    let mut evidence_dirs: Vec<PathBuf> = Vec::new();
    let mut candidate_variant: Option<String> = None;
    let mut baseline_variant: Option<String> = None;
    let mut reference_variant: Option<String> = None;
    let mut performance_candidate_variant: Option<String> = None;
    let mut performance_baseline_variant: Option<String> = None;
    let mut performance_reference_variant: Option<String> = None;
    let mut executor = EvalExecutorMode::Auto;
    let mut fetch_datasets = false;
    let mut offline = false;
    let mut dataset_cache: Option<PathBuf> = None;
    let mut result_cache: Option<PathBuf> = None;
    let mut cache_mode = EvalCacheMode::Use;
    let mut runs = 1usize;
    let mut runs_explicit = false;
    let mut warmup_runs = 0usize;
    let mut benchmark = false;
    let mut host_memory_class: Option<String> = None;
    let mut host_memory_width_bits: Option<u32> = None;
    let mut host_memory_bandwidth_gbps: Option<f64> = None;
    let mut fail_on_admission = false;
    let mut models_spec: Option<String> = None;
    let mut dry_run = false;
    let mut status = false;
    let mut fetch = false;

    let mut i = 1;
    while i < argv.len() {
        match argv[i].as_str() {
            "-h" | "--help" => return Err(usage()),
            "--model" => {
                model = Some(take_value(&argv, i, "--model")?);
                i += 2;
            }
            "--models" => {
                models_spec = Some(take_value(&argv, i, "--models")?);
                i += 2;
            }
            "--dry-run" => {
                dry_run = true;
                i += 1;
            }
            "--status" => {
                status = true;
                i += 1;
            }
            "--fetch" => {
                fetch = true;
                i += 1;
            }
            "--draft" => {
                draft = Some(take_value(&argv, i, "--draft")?);
                i += 2;
            }
            "--baseline" => {
                baseline = Some(take_value(&argv, i, "--baseline")?);
                i += 2;
            }
            "--reference" => {
                reference = Some(take_value(&argv, i, "--reference")?);
                i += 2;
            }
            "--tier" => {
                tier = EvalTier::parse(&take_value(&argv, i, "--tier")?)?;
                i += 2;
            }
            "--battery" | "--batteries" => {
                batteries = Some(parse_csv(
                    &take_value(&argv, i, "--battery")?,
                    BatteryId::parse,
                )?);
                i += 2;
            }
            "--suite" | "--suites" => {
                suites.extend(parse_csv(
                    &take_value(&argv, i, "--suite")?,
                    SuiteId::parse,
                )?);
                i += 2;
            }
            "--out" => {
                out_dir = Some(PathBuf::from(take_value(&argv, i, "--out")?));
                i += 2;
            }
            "--kv-mode" => {
                kv_mode = Some(take_value(&argv, i, "--kv-mode")?);
                i += 2;
            }
            "--max-tokens" => {
                max_tokens = parse_usize(&take_value(&argv, i, "--max-tokens")?, "--max-tokens")?;
                i += 2;
            }
            "--dflash" => {
                dflash = DflashMode::parse(&take_value(&argv, i, "--dflash")?)?;
                i += 2;
            }
            "--profile" => {
                profile = ProfileMode::parse(&take_value(&argv, i, "--profile")?)?;
                i += 2;
            }
            "--quality-max-chunks" => {
                quality_max_chunks = Some(parse_usize(
                    &take_value(&argv, i, "--quality-max-chunks")?,
                    "--quality-max-chunks",
                )?);
                i += 2;
            }
            "--kldref" | "--kld-ref" => {
                kldref = Some(PathBuf::from(take_value(&argv, i, "--kldref")?));
                i += 2;
            }
            "--quality-json" => {
                quality_json = Some(PathBuf::from(take_value(&argv, i, "--quality-json")?));
                i += 2;
            }
            "--performance-json" => {
                performance_json = Some(PathBuf::from(take_value(&argv, i, "--performance-json")?));
                i += 2;
            }
            "--evidence-json" => {
                evidence_json.push(PathBuf::from(take_value(&argv, i, "--evidence-json")?));
                i += 2;
            }
            "--evidence-dir" => {
                evidence_dirs.push(PathBuf::from(take_value(&argv, i, "--evidence-dir")?));
                i += 2;
            }
            "--candidate-variant" => {
                candidate_variant = Some(take_value(&argv, i, "--candidate-variant")?);
                i += 2;
            }
            "--baseline-variant" => {
                baseline_variant = Some(take_value(&argv, i, "--baseline-variant")?);
                i += 2;
            }
            "--reference-variant" => {
                reference_variant = Some(take_value(&argv, i, "--reference-variant")?);
                i += 2;
            }
            "--performance-candidate-variant" => {
                performance_candidate_variant =
                    Some(take_value(&argv, i, "--performance-candidate-variant")?);
                i += 2;
            }
            "--performance-baseline-variant" => {
                performance_baseline_variant =
                    Some(take_value(&argv, i, "--performance-baseline-variant")?);
                i += 2;
            }
            "--performance-reference-variant" => {
                performance_reference_variant =
                    Some(take_value(&argv, i, "--performance-reference-variant")?);
                i += 2;
            }
            "--executor" => {
                executor = EvalExecutorMode::parse(&take_value(&argv, i, "--executor")?)?;
                i += 2;
            }
            "--fetch-datasets" => {
                fetch_datasets = true;
                i += 1;
            }
            "--offline" => {
                offline = true;
                i += 1;
            }
            "--dataset-cache" => {
                dataset_cache = Some(PathBuf::from(take_value(&argv, i, "--dataset-cache")?));
                i += 2;
            }
            "--result-cache" | "--cache-dir" => {
                result_cache = Some(PathBuf::from(take_value(&argv, i, argv[i].as_str())?));
                i += 2;
            }
            "--force" => {
                cache_mode = EvalCacheMode::Force;
                i += 1;
            }
            "--regenerate" => {
                cache_mode = EvalCacheMode::Regenerate;
                i += 1;
            }
            "--no-cache" => {
                cache_mode = EvalCacheMode::Off;
                i += 1;
            }
            "--runs" => {
                runs = parse_usize(&take_value(&argv, i, "--runs")?, "--runs")?;
                runs_explicit = true;
                i += 2;
            }
            "--warmup-runs" => {
                warmup_runs =
                    parse_usize(&take_value(&argv, i, "--warmup-runs")?, "--warmup-runs")?;
                i += 2;
            }
            "--benchmark" => {
                benchmark = true;
                i += 1;
            }
            "--host-memory-class" => {
                host_memory_class = Some(take_value(&argv, i, "--host-memory-class")?);
                i += 2;
            }
            "--host-memory-width-bits" => {
                host_memory_width_bits = Some(parse_u32(
                    &take_value(&argv, i, "--host-memory-width-bits")?,
                    "--host-memory-width-bits",
                )?);
                i += 2;
            }
            "--host-memory-bandwidth-gbps" => {
                host_memory_bandwidth_gbps = Some(parse_f64(
                    &take_value(&argv, i, "--host-memory-bandwidth-gbps")?,
                    "--host-memory-bandwidth-gbps",
                )?);
                i += 2;
            }
            "--fail-on-admission" => {
                fail_on_admission = true;
                i += 1;
            }
            other => return Err(format!("unknown arg: {other}\n\n{}", usage())),
        }
    }

    if fetch_datasets && offline {
        return Err("--fetch-datasets and --offline are mutually exclusive".to_string());
    }
    if benchmark && !runs_explicit {
        runs = 5;
    }
    if runs == 0 {
        return Err("--runs must be at least 1".to_string());
    }
    // --model is required for a single run, but --models (sweep), --status, and
    // --fetch supply or don't need it; use a placeholder that run_from_env
    // replaces per sweep iteration.
    let model = match model {
        Some(m) => m,
        None if models_spec.is_some() || status || fetch => String::new(),
        None => return Err(format!("error: --model is required\n\n{}", usage())),
    };
    let batteries = batteries.unwrap_or_else(|| default_batteries(tier));
    if suites.is_empty() && batteries.contains(&BatteryId::Barrage) {
        suites = default_suites(tier);
    }
    suites.sort();
    suites.dedup();
    if draft.is_none() && matches!(dflash, DflashMode::Auto | DflashMode::On) {
        draft = discover_dflash_draft_for_model(Path::new(&model))
            .map(|path| path.display().to_string());
    }
    let out_dir = out_dir.unwrap_or_else(|| default_output_dir(&model, tier));
    let dataset_cache = dataset_cache.unwrap_or_else(default_dataset_cache);
    let result_cache = result_cache.unwrap_or_else(default_result_cache);

    Ok(EvalConfig {
        model,
        draft,
        baseline,
        reference,
        tier,
        batteries,
        suites,
        out_dir,
        kv_mode,
        max_tokens,
        dflash,
        profile,
        quality_max_chunks,
        kldref,
        quality_json,
        performance_json,
        evidence_json,
        evidence_dirs,
        candidate_variant,
        baseline_variant,
        reference_variant,
        performance_candidate_variant,
        performance_baseline_variant,
        performance_reference_variant,
        executor,
        fetch_datasets,
        offline,
        dataset_cache,
        result_cache,
        cache_mode,
        runs,
        warmup_runs,
        benchmark,
        host_memory_class,
        host_memory_width_bits,
        host_memory_bandwidth_gbps,
        fail_on_admission,
        models_spec,
        dry_run,
        status,
        fetch,
    })
}

pub fn usage() -> String {
    "Usage:\n  hipfire-eval --model <model> [--tier fast|medium|long|extensive]\n\n\
     Options:\n\
       --version                print Hipfire eval runner version/git metadata\n\
       --models <glob|csv>      sweep many SKUs from the model dir (e.g. 'qwen3.5,qwen3.6' or 'qwen3.5-9b-*'); per-model out dirs + a cross-model rollup\n\
       --dry-run                plan only: resolve models/batteries/cache/artifacts and report (no tests run, nothing fetched/generated)\n\
       --status                 print cache/dataset/hardware status and exit\n\
       --fetch                  ensure datasets are present (HF fetch), then exit\n\
       --battery <a,b>          smoke,coherence,quality,retrieval,speed,dflash,pflash,agentic,runtime,prompt_shape,structured,barrage,longctx,vision,cask,profile,perplexity,calibrate\n\
       --suite <a,b>            gpqa,lm_eval_micro,humaneval,deep_swe,swe_bench,ruler,nolima,needle_chain,niah,sequential_niah\n\
       --baseline <model>       baseline quantized model for candidate comparison\n\
       --reference <model>      higher precision reference model or fixture\n\
       --out <dir>              output directory\n\
       --draft <path>           DFlash draft artifact\n\
       --dflash <off|auto|on>   DFlash mode (default: off)\n\
       --kv-mode <mode>         KV mode metadata to record\n\
       --max-tokens <N>         short decode cap for execution batteries (default: 64)\n\
       --profile <off|passive>  profiling mode (default: off)\n\
       --quality-max-chunks <N> quality canary chunk cap\n\
       --kldref <path>          HFQM .kldref.hfq override for quality battery\n\
       --quality-json <path>    ingest kld_reduce.py result-data.json for quality battery\n\
       --performance-json <path> ingest benchmark/perf JSON for speed battery\n\
       --evidence-json <path>   ingest profiler/runtime evidence JSON; repeatable\n\
       --evidence-dir <dir>     ingest standard runtime evidence JSON files from a directory; repeatable\n\
       --candidate-variant <v>  quality-json variant for --model (default: model stem)\n\
       --baseline-variant <v>   quality-json variant for --baseline (default: baseline stem)\n\
       --reference-variant <v>  quality-json variant for --reference (default: reference stem)\n\
       --performance-candidate-variant <v> performance-json variant for --model\n\
       --performance-baseline-variant <v>  performance-json variant for --baseline\n\
       --performance-reference-variant <v> performance-json variant for --reference\n\
       --executor <auto|none|examples|daemon|direct|mock> execution backend (default: auto; daemon uses the JSONL adapter; examples/direct run Hipfire example binaries; mock is no-GPU test-only)\n\
       --fetch-datasets         opt in to Hugging Face dataset fetches\n\
       --offline                forbid network fetches\n\
       --dataset-cache <dir>    dataset cache root (default: ~/.hipfire/datasets)\n\
       --result-cache <dir>     result cache root (default: ~/.hipfire/eval-results/cache)\n\
       --force                  ignore cache hits for this run, but write new cache entries\n\
       --regenerate             delete and replace matching cache entries before running\n\
       --no-cache               disable result cache reads and writes\n\
       --runs <N>               repeat each scored battery N times (default: 1)\n\
       --warmup-runs <N>        run and discard N warmup battery passes before scored repeats\n\
       --benchmark              shorthand for --runs 5 unless --runs is provided; emits aggregate rows\n\
       --host-memory-class <s>  override uncertain host memory class (e.g. gddr6, lpddr5x)\n\
       --host-memory-width-bits <N> override uncertain memory bus width/channel width\n\
       --host-memory-bandwidth-gbps <N> override computed peak memory bandwidth\n\
       --fail-on-admission      exit non-zero after writing artifacts unless admission verdict is promote\n"
        .to_string()
}

pub fn version_report() -> String {
    let context = EvalContext::new();
    let mut lines = vec![
        format!("hipfire-eval {}", env!("CARGO_PKG_VERSION")),
        format!("hipfire_version {}", env!("CARGO_PKG_VERSION")),
        format!(
            "git_commit {}",
            context.commit_sha.unwrap_or_else(|| "unknown".to_string())
        ),
        format!(
            "git_branch {}",
            context.git_branch.unwrap_or_else(|| "unknown".to_string())
        ),
        format!(
            "git_describe {}",
            context
                .git_describe
                .unwrap_or_else(|| "unknown".to_string())
        ),
        format!(
            "git_dirty {}",
            context
                .git_dirty
                .map(|dirty| dirty.to_string())
                .unwrap_or_else(|| "unknown".to_string())
        ),
        format!(
            "binary_hash {}",
            context.binary_hash.unwrap_or_else(|| "unknown".to_string())
        ),
    ];
    if let Some(arch) = context.arch {
        lines.push(format!("arch {arch}"));
    }
    if let Some(rocm) = context.rocm {
        lines.push(format!("rocm {rocm}"));
    }
    lines.push(String::new());
    lines.join("\n")
}

fn take_value(argv: &[String], i: usize, flag: &str) -> Result<String, String> {
    argv.get(i + 1)
        .filter(|v| !v.starts_with("--"))
        .cloned()
        .ok_or_else(|| format!("{flag} requires a value"))
}

fn parse_csv<T>(raw: &str, parse: fn(&str) -> Result<T, String>) -> Result<Vec<T>, String> {
    raw.split(',')
        .filter(|s| !s.trim().is_empty())
        .map(|s| parse(s.trim()))
        .collect()
}

fn parse_usize(raw: &str, flag: &str) -> Result<usize, String> {
    raw.parse()
        .map_err(|_| format!("{flag} must be a positive integer"))
}

fn parse_u32(raw: &str, flag: &str) -> Result<u32, String> {
    raw.parse()
        .map_err(|_| format!("{flag} must be a positive integer"))
}

fn parse_f64(raw: &str, flag: &str) -> Result<f64, String> {
    let value: f64 = raw
        .parse()
        .map_err(|_| format!("{flag} must be a positive number"))?;
    if value.is_finite() && value > 0.0 {
        Ok(value)
    } else {
        Err(format!("{flag} must be a positive finite number"))
    }
}

pub fn default_batteries(tier: EvalTier) -> Vec<BatteryId> {
    let mut out = vec![
        BatteryId::Smoke,
        BatteryId::Coherence,
        BatteryId::Quality,
        BatteryId::Retrieval,
        BatteryId::Speed,
        BatteryId::Dflash,
        BatteryId::Agentic,
        BatteryId::PromptShape,
        BatteryId::Structured,
    ];
    if matches!(
        tier,
        EvalTier::Medium | EvalTier::Long | EvalTier::Extensive
    ) {
        // Perplexity (PPL, + KLD when a reference resolves) joins medium: the
        // run itself is fast when the corpus/kldref are present, and it skips
        // with an actionable reason when they are not.
        out.extend([
            BatteryId::Barrage,
            BatteryId::Longctx,
            BatteryId::Perplexity,
        ]);
    }
    if matches!(tier, EvalTier::Long | EvalTier::Extensive) {
        out.push(BatteryId::Profile);
    }
    if matches!(tier, EvalTier::Extensive) {
        out.extend([BatteryId::Vision, BatteryId::Cask]);
    }
    out
}

pub fn default_suites(tier: EvalTier) -> Vec<SuiteId> {
    match tier {
        EvalTier::Fast => vec![SuiteId::Gpqa],
        EvalTier::Medium => vec![
            SuiteId::Gpqa,
            SuiteId::LmEvalMicro,
            SuiteId::HumanEval,
            SuiteId::DeepSwe,
            SuiteId::SweBench,
        ],
        EvalTier::Long => vec![
            SuiteId::Gpqa,
            SuiteId::LmEvalMicro,
            SuiteId::HumanEval,
            SuiteId::DeepSwe,
            SuiteId::SweBench,
            SuiteId::Ruler,
            SuiteId::NoLiMa,
            SuiteId::Niah,
        ],
        EvalTier::Extensive => vec![
            SuiteId::Gpqa,
            SuiteId::LmEvalMicro,
            SuiteId::HumanEval,
            SuiteId::Ruler,
            SuiteId::NoLiMa,
            SuiteId::Niah,
            SuiteId::NeedleChain,
            SuiteId::SequentialNiah,
            SuiteId::DeepSwe,
            SuiteId::SweBench,
        ],
    }
}

pub fn default_output_dir(model: &str, tier: EvalTier) -> PathBuf {
    let stem = model_artifact_stem(model);
    let leaf = format!("{}-{}-{}", utc_stamp_compact(), stem, tier.as_str());
    home_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join(".hipfire")
        .join("eval-results")
        .join("runs")
        .join(leaf)
}

fn default_dataset_cache() -> PathBuf {
    // Reusable top-level location (shared across eval runs / tools), overridable
    // with --dataset-cache. (Was ~/.hipfire/eval/datasets.)
    home_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join(".hipfire")
        .join("datasets")
}

fn default_result_cache() -> PathBuf {
    home_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join(".hipfire")
        .join("eval-results")
        .join("cache")
}

pub fn collect_default_host_profile() -> HostProfile {
    collect_host_profile(detect_arch(), HostProfileOverrides::default())
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

fn collect_host_profile(arch: Option<String>, overrides: HostProfileOverrides) -> HostProfile {
    let topology = read_primary_kfd_properties();
    let drm = topology
        .as_ref()
        .and_then(|props| props.get("drm_render_minor"))
        .and_then(|minor| minor.parse::<u32>().ok())
        .map(|minor| PathBuf::from(format!("/sys/class/drm/renderD{minor}/device")))
        .filter(|path| path.exists())
        .or_else(primary_amd_drm_device);

    let vendor_id = drm
        .as_ref()
        .and_then(|path| read_sysfs_trimmed(&path.join("vendor")));
    let device_id = drm
        .as_ref()
        .and_then(|path| read_sysfs_trimmed(&path.join("device")));
    let render_node = topology
        .as_ref()
        .and_then(|props| props.get("drm_render_minor"))
        .map(|minor| format!("/dev/dri/renderD{minor}"));
    let libdrm_probe = render_node
        .as_deref()
        .and_then(probe_amdgpu_dev_info_libdrm)
        .or_else(|| probe_amdgpu_dev_info_libdrm("/dev/dri/renderD128"));
    let gfx = arch.or_else(|| {
        topology
            .as_ref()
            .and_then(|props| props.get("gfx_target_version"))
            .and_then(|raw| raw.parse::<u32>().ok())
            .map(gfx_target_version_to_arch)
    });
    let simd_count = topology
        .as_ref()
        .and_then(|props| props.get("simd_count"))
        .and_then(|raw| raw.parse::<u32>().ok());
    let simd_per_cu = topology
        .as_ref()
        .and_then(|props| props.get("simd_per_cu"))
        .and_then(|raw| raw.parse::<u32>().ok())
        .filter(|value| *value > 0);
    let cu_count = libdrm_probe
        .as_ref()
        .and_then(|probe| probe.cu_count)
        .or_else(|| match (simd_count, simd_per_cu) {
            (Some(simd_count), Some(simd_per_cu)) => Some(simd_count / simd_per_cu),
            _ => None,
        });
    let vram_bytes = drm
        .as_ref()
        .and_then(|path| read_sysfs_trimmed(&path.join("mem_info_vram_total")))
        .and_then(|raw| raw.parse::<u64>().ok())
        .or_else(|| {
            topology
                .as_ref()
                .and_then(|props| props.get("local_mem_size"))
                .and_then(|raw| raw.parse::<u64>().ok())
        })
        .or_else(|| libdrm_probe.as_ref().and_then(|probe| probe.vram_bytes));
    let gtt_bytes = drm
        .as_ref()
        .and_then(|path| read_sysfs_trimmed(&path.join("mem_info_gtt_total")))
        .and_then(|raw| raw.parse::<u64>().ok());
    let memory_clock_mhz = libdrm_probe
        .as_ref()
        .and_then(|probe| probe.max_memory_clock_mhz)
        .or_else(|| {
            drm.as_ref()
                .and_then(|path| fs::read_to_string(path.join("pp_dpm_mclk")).ok())
                .and_then(|raw| parse_pp_dpm_mclk_max_mhz(&raw))
        });
    let system_memory_bytes = linux_mem_total_bytes();
    let hardware_kind = classify_hardware_kind(vram_bytes, gtt_bytes);

    let mut memory_class = SourcedField::unknown();
    if let Some(value) = overrides.memory_class.clone() {
        memory_class = SourcedField::override_value(value);
    } else if let Some(value) = libdrm_probe
        .as_ref()
        .and_then(|probe| probe.memory_class.clone())
    {
        memory_class = SourcedField::libdrm_value(value);
    }
    let mut memory_width_bits = SourcedField::unknown();
    if let Some(value) = overrides.memory_width_bits {
        memory_width_bits = SourcedField::override_value(value);
    } else if let Some(value) = libdrm_probe
        .as_ref()
        .and_then(|probe| probe.memory_width_bits)
    {
        memory_width_bits = SourcedField::libdrm_value(value);
    }
    let memory_clock_mhz = memory_clock_mhz
        .map(|value| {
            if libdrm_probe
                .as_ref()
                .and_then(|probe| probe.max_memory_clock_mhz)
                == Some(value)
            {
                SourcedField::libdrm_value(value)
            } else {
                SourcedField::sysfs_value(value)
            }
        })
        .unwrap_or_else(SourcedField::unknown);
    let peak_bandwidth_gbps = if let Some(value) = overrides.memory_bandwidth_gbps {
        SourcedField::override_value(value)
    } else if let (Some(clock), Some(width), Some(class)) = (
        memory_clock_mhz.value,
        memory_width_bits.value,
        memory_class.value.as_deref(),
    ) {
        compute_peak_bandwidth_gbps(clock, width, class)
            .map(SourcedField::computed_value)
            .unwrap_or_else(SourcedField::unknown)
    } else {
        SourcedField::unknown()
    };

    let probe_status = if topology.is_some() || drm.is_some() {
        EvalStatus::Pass
    } else {
        EvalStatus::Skip
    };
    let reason = if probe_status == EvalStatus::Skip {
        Some("no AMD KFD/DRM device metadata found".to_string())
    } else {
        None
    };
    let hardware_bucket = hardware_bucket(
        &hardware_kind,
        gfx.as_deref(),
        device_id.as_deref(),
        cu_count,
        vram_bytes,
        memory_class.value.as_deref(),
        memory_width_bits.value,
        peak_bandwidth_gbps.value,
    );
    let mut profile = HostProfile {
        schema: 1,
        source: if libdrm_probe.is_some() {
            "libdrm_amdgpu+kfd-sysfs".to_string()
        } else {
            "linux-kfd-drm-sysfs".to_string()
        },
        probe_status,
        reason,
        hardware_kind,
        hardware_bucket,
        host_profile_hash: String::new(),
        gpu_model: drm
            .as_ref()
            .and_then(|path| read_sysfs_trimmed(&path.join("product_name"))),
        gfx,
        vendor_id,
        device_id: libdrm_probe
            .as_ref()
            .map(|probe| format!("0x{:04x}", probe.asic_id))
            .or(device_id),
        render_node,
        cu_count,
        vram_bytes,
        gtt_bytes,
        system_memory_bytes,
        memory_class,
        memory_width_bits,
        memory_clock_mhz,
        peak_bandwidth_gbps,
    };
    profile.host_profile_hash = host_profile_hash(&profile);
    profile
}

#[derive(Debug, Clone)]
struct LibDrmAmdgpuProbe {
    asic_id: u32,
    cu_count: Option<u32>,
    vram_bytes: Option<u64>,
    memory_class: Option<String>,
    memory_width_bits: Option<u32>,
    max_memory_clock_mhz: Option<f64>,
}

#[repr(C)]
#[derive(Default)]
struct AmdgpuGpuInfo {
    asic_id: u32,
    chip_rev: u32,
    chip_external_rev: u32,
    family_id: u32,
    ids_flags: u64,
    max_engine_clk: u64,
    max_memory_clk: u64,
    num_shader_engines: u32,
    num_shader_arrays_per_engine: u32,
    avail_quad_shader_pipes: u32,
    max_quad_shader_pipes: u32,
    cache_entries_per_quad_pipe: u32,
    num_hw_gfx_contexts: u32,
    rb_pipes: u32,
    enabled_rb_pipes_mask: u32,
    gpu_counter_freq: u32,
    backend_disable: [u32; 4],
    mc_arb_ramcfg: u32,
    gb_addr_cfg: u32,
    gb_tile_mode: [u32; 32],
    gb_macro_tile_mode: [u32; 16],
    pa_sc_raster_cfg: [u32; 4],
    pa_sc_raster_cfg1: [u32; 4],
    cu_active_number: u32,
    cu_ao_mask: u32,
    cu_bitmap: [[u32; 4]; 4],
    vram_type: u32,
    vram_bit_width: u32,
    ce_ram_size: u32,
    vce_harvest_config: u32,
    pci_rev_id: u32,
}

#[repr(C)]
#[derive(Default)]
struct AmdgpuHeapInfo {
    total_heap_size: u64,
    usable_heap_size: u64,
    heap_usage: u64,
    max_allocation: u64,
}

type AmdgpuDeviceHandle = *mut c_void;
type AmdgpuDeviceInitialize =
    unsafe extern "C" fn(i32, *mut u32, *mut u32, *mut AmdgpuDeviceHandle) -> i32;
type AmdgpuDeviceDeinitialize = unsafe extern "C" fn(AmdgpuDeviceHandle) -> i32;
type AmdgpuQueryGpuInfo = unsafe extern "C" fn(AmdgpuDeviceHandle, *mut AmdgpuGpuInfo) -> i32;
type AmdgpuQueryHeapInfo =
    unsafe extern "C" fn(AmdgpuDeviceHandle, u32, u32, *mut AmdgpuHeapInfo) -> i32;

const AMDGPU_GEM_DOMAIN_VRAM: u32 = 0x4;

fn probe_amdgpu_dev_info_libdrm(render_node: &str) -> Option<LibDrmAmdgpuProbe> {
    unsafe {
        let lib = dlopen_first(&["libdrm_amdgpu.so.1", "libdrm_amdgpu.so"])?;
        let device_initialize: AmdgpuDeviceInitialize =
            std::mem::transmute(dlsym_required(lib, "amdgpu_device_initialize")?);
        let device_deinitialize: AmdgpuDeviceDeinitialize =
            std::mem::transmute(dlsym_required(lib, "amdgpu_device_deinitialize")?);
        let query_gpu_info: AmdgpuQueryGpuInfo =
            std::mem::transmute(dlsym_required(lib, "amdgpu_query_gpu_info")?);
        let query_heap_info: AmdgpuQueryHeapInfo =
            std::mem::transmute(dlsym_required(lib, "amdgpu_query_heap_info")?);

        let path = CString::new(render_node).ok()?;
        let fd = libc::open(path.as_ptr(), libc::O_RDWR | libc::O_CLOEXEC);
        if fd < 0 {
            libc::dlclose(lib);
            return None;
        }
        let mut major = 0u32;
        let mut minor = 0u32;
        let mut handle: AmdgpuDeviceHandle = std::ptr::null_mut();
        if device_initialize(fd, &mut major, &mut minor, &mut handle) != 0 || handle.is_null() {
            libc::close(fd);
            libc::dlclose(lib);
            return None;
        }

        let mut gpu_info = AmdgpuGpuInfo::default();
        let gpu_ok = query_gpu_info(handle, &mut gpu_info) == 0;
        let mut heap_info = AmdgpuHeapInfo::default();
        let heap_ok = query_heap_info(handle, AMDGPU_GEM_DOMAIN_VRAM, 0, &mut heap_info) == 0;
        let _ = device_deinitialize(handle);
        libc::close(fd);
        libc::dlclose(lib);
        if !gpu_ok {
            return None;
        }
        Some(LibDrmAmdgpuProbe {
            asic_id: gpu_info.asic_id,
            cu_count: nonzero_u32(gpu_info.cu_active_number),
            vram_bytes: heap_ok.then_some(heap_info.total_heap_size),
            memory_class: amdgpu_vram_type_name(gpu_info.vram_type).map(str::to_string),
            memory_width_bits: nonzero_u32(gpu_info.vram_bit_width),
            max_memory_clock_mhz: nonzero_u64(gpu_info.max_memory_clk)
                .map(|khz| khz as f64 / 1000.0),
        })
    }
}

unsafe fn dlopen_first(names: &[&str]) -> Option<*mut c_void> {
    for name in names {
        let c_name = CString::new(*name).ok()?;
        let lib = libc::dlopen(c_name.as_ptr(), libc::RTLD_NOW | libc::RTLD_LOCAL);
        if !lib.is_null() {
            return Some(lib);
        }
    }
    None
}

unsafe fn dlsym_required(lib: *mut c_void, name: &str) -> Option<*mut c_void> {
    let c_name = CString::new(name).ok()?;
    let symbol = libc::dlsym(lib, c_name.as_ptr());
    if symbol.is_null() {
        None
    } else {
        Some(symbol)
    }
}

fn nonzero_u32(value: u32) -> Option<u32> {
    (value != 0).then_some(value)
}

fn nonzero_u64(value: u64) -> Option<u64> {
    (value != 0).then_some(value)
}

fn amdgpu_vram_type_name(raw: u32) -> Option<&'static str> {
    match raw {
        2 => Some("ddr2"),
        5 => Some("gddr5"),
        6 => Some("hbm"),
        7 => Some("ddr3"),
        8 => Some("ddr4"),
        9 => Some("gddr6"),
        10 => Some("ddr5"),
        11 => Some("lpddr4"),
        12 => Some("lpddr5"),
        13 => Some("hbm3e"),
        14 => Some("hbm4"),
        _ => None,
    }
}

fn read_primary_kfd_properties() -> Option<BTreeMap<String, String>> {
    for node in ["1", "0"] {
        let path = format!("/sys/class/kfd/kfd/topology/nodes/{node}/properties");
        if let Ok(raw) = fs::read_to_string(path) {
            let props = parse_kfd_properties(&raw);
            if props.get("gfx_target_version").is_some() {
                return Some(props);
            }
        }
    }
    None
}

fn parse_kfd_properties(raw: &str) -> BTreeMap<String, String> {
    raw.lines()
        .filter_map(|line| {
            let (key, value) = line.split_once(' ')?;
            Some((key.trim().to_string(), value.trim().to_string()))
        })
        .collect()
}

fn primary_amd_drm_device() -> Option<PathBuf> {
    let entries = fs::read_dir("/sys/class/drm").ok()?;
    for entry in entries.flatten() {
        let name = entry.file_name();
        let name = name.to_string_lossy();
        if !name.starts_with("renderD") {
            continue;
        }
        let device = entry.path().join("device");
        if read_sysfs_trimmed(&device.join("vendor")).as_deref() == Some("0x1002") {
            return Some(device);
        }
    }
    None
}

fn read_sysfs_trimmed(path: &Path) -> Option<String> {
    fs::read_to_string(path)
        .ok()
        .map(|raw| raw.trim().to_string())
        .filter(|raw| !raw.is_empty())
}

fn parse_pp_dpm_mclk_max_mhz(raw: &str) -> Option<f64> {
    raw.lines()
        .filter_map(|line| {
            let after_colon = line.split_once(':').map(|(_, rest)| rest).unwrap_or(line);
            let token = after_colon
                .split_whitespace()
                .find(|part| part.to_ascii_lowercase().contains("mhz"))?;
            let digits = token
                .trim_end_matches('*')
                .trim_end_matches("Mhz")
                .trim_end_matches("MHz")
                .trim_end_matches("mhz");
            digits.parse::<f64>().ok()
        })
        .filter(|value| value.is_finite())
        .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
}

fn linux_mem_total_bytes() -> Option<u64> {
    let raw = fs::read_to_string("/proc/meminfo").ok()?;
    for line in raw.lines() {
        if let Some(rest) = line.strip_prefix("MemTotal:") {
            let kib = rest.split_whitespace().next()?.parse::<u64>().ok()?;
            return Some(kib * 1024);
        }
    }
    None
}

fn mock_battery_rows(
    battery: BatteryId,
    config: &EvalConfig,
    ctx: &EvalContext,
    datasets: &[DatasetManifestEntry],
) -> Option<Vec<EvalResult>> {
    let rows = match battery {
        BatteryId::Smoke => vec![
            mock_pass_row(
                battery,
                None,
                "load_metadata",
                None,
                config,
                ctx,
                None,
                BTreeMap::from([
                    ("load_metadata_ok".to_string(), json!(1.0)),
                    (
                        "mock_latency_ms".to_string(),
                        json!(mock_metric(&config.model, "load", 10.0, 3.0)),
                    ),
                ]),
                config.model.clone(),
            ),
            mock_pass_row(
                battery,
                None,
                "finite_greedy_decode",
                None,
                config,
                ctx,
                prompt("benchmarks/prompts/qwen2_smoke.txt"),
                BTreeMap::from([
                    ("finite_tokens".to_string(), json!(1.0)),
                    (
                        "generated_tokens".to_string(),
                        json!(config.max_tokens.min(16) as f64),
                    ),
                ]),
                config.model.clone(),
            ),
        ],
        BatteryId::Coherence => mock_metric_family_rows(
            battery,
            "runtime_detector_canary",
            prompt("benchmarks/prompts/qwen2_smoke.txt"),
            config,
            ctx,
            |_model| {
                BTreeMap::from([
                    ("hard_fails".to_string(), json!(0.0)),
                    ("soft_warns".to_string(), json!(0.0)),
                    ("detector_count".to_string(), json!(8.0)),
                    (
                        "detector_profile".to_string(),
                        json!("default_runtime_coherence"),
                    ),
                    ("runtime_path".to_string(), json!("daemon_jsonl_mock")),
                ])
            },
        ),
        BatteryId::Quality => mock_metric_family_rows(
            battery,
            "kld_reference_slice",
            prompt("benchmarks/quality-baselines/harness/canary.md"),
            config,
            ctx,
            |model| {
                BTreeMap::from([
                    (
                        "mean_kld".to_string(),
                        json!(mock_metric(model, "mean_kld", 0.015, 0.02)),
                    ),
                    (
                        "p99_kld".to_string(),
                        json!(mock_metric(model, "p99_kld", 0.04, 0.05)),
                    ),
                    (
                        "ppl".to_string(),
                        json!(mock_metric(model, "ppl", 5.0, 0.5)),
                    ),
                    (
                        "argmax_match_rate".to_string(),
                        json!(mock_metric(model, "argmax", 0.93, 0.05)),
                    ),
                ])
            },
        ),
        BatteryId::Speed => mock_metric_family_rows(
            battery,
            "pp32_pp128_ttft_decode",
            prompt("benchmarks/prompts/lru_cache_single_blank.txt"),
            config,
            ctx,
            |model| {
                BTreeMap::from([
                    (
                        "pp32_ms".to_string(),
                        json!(mock_metric(model, "pp32", 7.0, 2.0)),
                    ),
                    (
                        "pp128_ms".to_string(),
                        json!(mock_metric(model, "pp128", 22.0, 6.0)),
                    ),
                    (
                        "ttft_ms".to_string(),
                        json!(mock_metric(model, "ttft", 30.0, 8.0)),
                    ),
                    (
                        "tok_s".to_string(),
                        json!(mock_metric(model, "tok_s", 110.0, 30.0)),
                    ),
                ])
            },
        ),
        BatteryId::Dflash => mock_metric_family_rows(
            battery,
            "dflash_anchor",
            prompt("benchmarks/prompts/dflash_resident_smoke.txt"),
            config,
            ctx,
            |model| {
                BTreeMap::from([
                    (
                        "ar_tok_s".to_string(),
                        json!(mock_metric(model, "ar_tok_s", 90.0, 20.0)),
                    ),
                    (
                        "dflash_tok_s".to_string(),
                        json!(if config.dflash == DflashMode::Off {
                            0.0
                        } else {
                            mock_metric(model, "dflash_tok_s", 130.0, 35.0)
                        }),
                    ),
                    (
                        "accept_rate".to_string(),
                        json!(if config.dflash == DflashMode::Off {
                            0.0
                        } else {
                            mock_metric(model, "accept_rate", 0.45, 0.2)
                        }),
                    ),
                    (
                        "tau".to_string(),
                        json!(if config.dflash == DflashMode::Off {
                            1.0
                        } else {
                            mock_metric(model, "tau", 2.0, 1.5)
                        }),
                    ),
                ])
            },
        ),
        BatteryId::Agentic => mock_metric_family_rows(
            battery,
            "agentic_tool_call_shape",
            prompt("benchmarks/prompts/agentic_user_read.txt"),
            config,
            ctx,
            |_model| {
                BTreeMap::from([
                    ("hard_fails".to_string(), json!(0.0)),
                    ("soft_warns".to_string(), json!(0.0)),
                    ("detector_count".to_string(), json!(9.0)),
                    ("generated_tokens".to_string(), json!(64.0)),
                    ("structured_probe".to_string(), json!("agentic_tool_call")),
                    ("runtime_path".to_string(), json!("daemon_jsonl_mock")),
                ])
            },
        ),
        BatteryId::Runtime => runtime_cases()
            .iter()
            .map(|case| {
                pass_row(
                    BatteryId::Runtime,
                    None,
                    case.label,
                    None,
                    config,
                    ctx,
                    None,
                    BTreeMap::from([
                        ("runtime_evidence_case".to_string(), json!(case.label)),
                        ("script".to_string(), json!(case.script)),
                        ("executor".to_string(), json!("mock")),
                        ("runtime_path".to_string(), json!("server_runtime_mock")),
                    ]),
                )
            })
            .collect(),
        BatteryId::Barrage => mock_barrage_rows(config, ctx, datasets),
        _ => return None,
    };
    Some(rows)
}

fn performance_json_rows(config: &EvalConfig, ctx: &EvalContext) -> Option<Vec<EvalResult>> {
    let path = config.performance_json.as_ref()?;
    let rows = match load_performance_json_rows(path) {
        Ok(rows) => rows,
        Err(reason) => {
            return Some(vec![skip_row(
                BatteryId::Speed,
                None,
                "performance_json_ingest",
                None,
                &reason,
                config,
                ctx,
                None,
            )]);
        }
    };
    let mut out = Vec::new();
    let candidate_variant = config
        .performance_candidate_variant
        .clone()
        .or_else(|| config.candidate_variant.clone())
        .unwrap_or_else(|| model_artifact_stem(&config.model));
    out.push(performance_json_row_for_variant(
        path,
        &rows,
        "candidate",
        &candidate_variant,
        &config.model,
        config,
        ctx,
    ));
    if let Some(model) = &config.baseline {
        let variant = config
            .performance_baseline_variant
            .clone()
            .or_else(|| config.baseline_variant.clone())
            .unwrap_or_else(|| model_artifact_stem(model));
        out.push(performance_json_row_for_variant(
            path, &rows, "baseline", &variant, model, config, ctx,
        ));
    }
    if let Some(model) = &config.reference {
        let variant = config
            .performance_reference_variant
            .clone()
            .or_else(|| config.reference_variant.clone())
            .unwrap_or_else(|| model_artifact_stem(model));
        out.push(performance_json_row_for_variant(
            path,
            &rows,
            "reference",
            &variant,
            model,
            config,
            ctx,
        ));
    }
    Some(out)
}

fn performance_json_row_for_variant(
    path: &Path,
    rows: &[PerformanceJsonRow],
    role: &str,
    variant: &str,
    model: &str,
    config: &EvalConfig,
    ctx: &EvalContext,
) -> EvalResult {
    let Some(row) = rows.iter().find(|r| r.variant == variant) else {
        return row_for_model(
            BatteryId::Speed,
            None,
            "performance_json_anchor",
            None,
            EvalStatus::Skip,
            Some(format!(
                "performance-json variant {variant:?} not found for {role}"
            )),
            BTreeMap::from([
                (
                    "performance_source".to_string(),
                    json!(path.display().to_string()),
                ),
                ("variant".to_string(), json!(variant)),
            ]),
            config,
            ctx,
            None,
            0,
            model.to_string(),
        );
    };

    let mut metrics = row.metrics.clone();
    metrics.insert("implemented".to_string(), json!(true));
    metrics.insert("executor".to_string(), json!("performance_json"));
    metrics.insert(
        "performance_source".to_string(),
        json!(path.display().to_string()),
    );
    metrics.insert("variant".to_string(), json!(row.variant.clone()));
    row_for_model(
        BatteryId::Speed,
        None,
        "performance_json_anchor",
        None,
        EvalStatus::Pass,
        None,
        metrics,
        config,
        ctx,
        prompt("benchmarks/prompts/lru_cache_single_blank.txt"),
        0,
        model.to_string(),
    )
}

#[derive(Debug, Clone)]
struct PerformanceJsonRow {
    variant: String,
    metrics: BTreeMap<String, Value>,
}

fn load_performance_json_rows(path: &Path) -> Result<Vec<PerformanceJsonRow>, String> {
    let body = fs::read_to_string(path).map_err(|e| format!("read performance json: {e}"))?;
    let value: Value =
        serde_json::from_str(&body).map_err(|e| format!("parse performance json: {e}"))?;
    let raw_rows = if let Some(rows) = value.as_array() {
        rows.clone()
    } else if let Some(rows) = value.get("performance_rows").and_then(Value::as_array) {
        rows.clone()
    } else if let Some(rows) = value.get("runs").and_then(Value::as_array) {
        rows.clone()
    } else {
        return Err("unsupported performance JSON shape".to_string());
    };
    raw_rows
        .iter()
        .map(parse_performance_json_row)
        .collect::<Result<Vec<_>, _>>()
}

fn parse_performance_json_row(value: &Value) -> Result<PerformanceJsonRow, String> {
    let variant = if let Some(variant) = value.get("variant").and_then(Value::as_str) {
        variant.to_string()
    } else {
        let base = ["model", "tag", "name"]
            .iter()
            .find_map(|k| value.get(*k).and_then(Value::as_str))
            .ok_or_else(|| "performance row missing variant/model/tag/name".to_string())?;
        if let Some(mode) = value.get("mode").and_then(Value::as_str) {
            format!("{base}:{mode}")
        } else {
            base.to_string()
        }
    };
    let mut metrics = BTreeMap::new();
    collect_performance_metrics(value, &mut metrics);
    if let Some(parsed) = value.get("parsed") {
        collect_performance_metrics(parsed, &mut metrics);
    }
    if metrics.is_empty() {
        return Err(format!(
            "performance row {variant:?} has no recognized numeric metrics"
        ));
    }
    validate_performance_metrics(&variant, &metrics)?;
    Ok(PerformanceJsonRow { variant, metrics })
}

fn collect_performance_metrics(value: &Value, out: &mut BTreeMap<String, Value>) {
    let Some(obj) = value.as_object() else {
        return;
    };
    for (key, value) in obj {
        let Some(num) = value.as_f64() else {
            continue;
        };
        if let Some(normalized) = normalize_performance_metric(key) {
            out.insert(normalized.to_string(), json!(num));
        }
    }
}

fn validate_performance_metrics(
    variant: &str,
    metrics: &BTreeMap<String, Value>,
) -> Result<(), String> {
    for (name, value) in metrics {
        let Some(num) = value.as_f64() else {
            continue;
        };
        if !num.is_finite() {
            return Err(format!("performance row {variant:?} has non-finite {name}"));
        }
        let non_negative = matches!(
            name.as_str(),
            "tok_s"
                | "wall_tok_s"
                | "ttft_ms"
                | "load_ms"
                | "prefill_ms"
                | "prefill_secs"
                | "decode_ms"
                | "decode_secs"
                | "teardown_ms"
                | "prefill_tok_s"
                | "pp32_tok_s"
                | "pp128_tok_s"
                | "pp512_tok_s"
                | "pp1024_tok_s"
                | "pp2048_tok_s"
                | "tau"
                | "accept_rate"
                | "emitted_tokens"
                | "cycles"
                | "vram_peak_bytes"
                | "vram_used_bytes"
                | "vram_used_mb"
                | "vram_loaded_mb"
                | "vram_free_mb"
                | "kv_bytes"
                | "workspace_bytes"
        );
        if non_negative && num < 0.0 {
            return Err(format!("performance row {variant:?} has negative {name}"));
        }
        if name == "accept_rate" && num > 1.0 {
            return Err(format!(
                "performance row {variant:?} has out-of-range accept_rate"
            ));
        }
        if name == "tau" && num == 0.0 {
            return Err(format!("performance row {variant:?} has zero tau"));
        }
    }
    Ok(())
}

fn normalize_performance_metric(key: &str) -> Option<&'static str> {
    match key {
        "tok_s" | "tokSOut" | "decode_tokS" | "decode_tok_s" | "gen_tok_s"
        | "tokens_per_second" => Some("tok_s"),
        "wall_tokS" | "wall_tok_s" => Some("wall_tok_s"),
        "ttft_ms" | "ttft" => Some("ttft_ms"),
        "load_ms" => Some("load_ms"),
        "prefill_ms" => Some("prefill_ms"),
        "prefill_secs" => Some("prefill_secs"),
        "decode_ms" => Some("decode_ms"),
        "decode_secs" => Some("decode_secs"),
        "teardown_ms" => Some("teardown_ms"),
        "prefill_tok_s" | "prefill_user_tokS" | "prefill_user_tok_s" => Some("prefill_tok_s"),
        "pp32_tok_s" | "pp32_tokS" => Some("pp32_tok_s"),
        "pp128_tok_s" | "pp128_tokS" => Some("pp128_tok_s"),
        "pp512_tok_s" | "pp512_tokS" => Some("pp512_tok_s"),
        "pp1024_tok_s" | "pp1024_tokS" => Some("pp1024_tok_s"),
        "pp2048_tok_s" | "pp2048_tokS" => Some("pp2048_tok_s"),
        "tau" | "decode_tau" => Some("tau"),
        "accept_rate" | "decode_accept_rate" => Some("accept_rate"),
        "emitted" | "emitted_tokens" => Some("emitted_tokens"),
        "cycles" => Some("cycles"),
        "vram_peak_bytes" => Some("vram_peak_bytes"),
        "vram_used_bytes" => Some("vram_used_bytes"),
        "vram_used_mb" => Some("vram_used_mb"),
        "vram_loaded_mb" => Some("vram_loaded_mb"),
        "vram_free_mb" => Some("vram_free_mb"),
        "kv_bytes" => Some("kv_bytes"),
        "workspace_bytes" => Some("workspace_bytes"),
        _ => None,
    }
}

fn mock_metric_family_rows<F>(
    battery: BatteryId,
    case_id: &str,
    prompt: Option<PromptRef>,
    config: &EvalConfig,
    ctx: &EvalContext,
    build_metrics: F,
) -> Vec<EvalResult>
where
    F: Fn(&str) -> BTreeMap<String, Value>,
{
    let mut rows = Vec::new();
    rows.push(mock_pass_row(
        battery,
        None,
        case_id,
        None,
        config,
        ctx,
        prompt.clone(),
        build_metrics(&config.model),
        config.model.clone(),
    ));
    for model in [config.baseline.as_ref(), config.reference.as_ref()]
        .into_iter()
        .flatten()
    {
        rows.push(mock_pass_row(
            battery,
            None,
            case_id,
            None,
            config,
            ctx,
            prompt.clone(),
            build_metrics(model),
            model.clone(),
        ));
    }
    rows
}

fn mock_barrage_rows(
    config: &EvalConfig,
    ctx: &EvalContext,
    datasets: &[DatasetManifestEntry],
) -> Vec<EvalResult> {
    let mut rows = Vec::new();
    for d in datasets {
        if d.status != EvalStatus::Pass {
            continue;
        }
        match d.suite {
            SuiteId::Gpqa => {
                let Ok(items) =
                    gpqa_materialized_items(Path::new(&d.cache_path), &d.selected_item_ids)
                else {
                    continue;
                };
                for item in items {
                    let prompt_ref = PromptRef::from_content(
                        format!("dataset:gpqa:{}", item.item_id),
                        item.prompt.as_bytes(),
                    );
                    let models = std::iter::once(&config.model)
                        .chain(config.baseline.iter())
                        .chain(config.reference.iter());
                    for model in models {
                        let mut metrics = BTreeMap::from([
                            (
                                "accuracy".to_string(),
                                json!(mock_bool_metric(model, &item.item_id)),
                            ),
                            (
                                "exact_match".to_string(),
                                json!(mock_bool_metric(model, &item.correct_answer)),
                            ),
                            ("answer_label".to_string(), json!(item.answer_label.clone())),
                            (
                                "answer_hash".to_string(),
                                json!(stable_hash_bytes(item.correct_answer.as_bytes())),
                            ),
                        ]);
                        add_dataset_provenance_metrics(&mut metrics, d);
                        rows.push(mock_pass_row(
                            BatteryId::Barrage,
                            Some(SuiteId::Gpqa),
                            "gpqa_zero_shot_native",
                            Some(item.item_id.clone()),
                            config,
                            ctx,
                            Some(prompt_ref.clone()),
                            metrics,
                            model.clone(),
                        ));
                    }
                }
            }
            SuiteId::LmEvalMicro => {
                let Ok(items) = lm_eval_micro_materialized_items(&d.selected_item_ids) else {
                    continue;
                };
                for item in items {
                    let prompt_ref = PromptRef::from_content(
                        format!("dataset:lm_eval_micro:{}", item.item_id),
                        item.prompt.as_bytes(),
                    );
                    let models = std::iter::once(&config.model)
                        .chain(config.baseline.iter())
                        .chain(config.reference.iter());
                    for model in models {
                        let mut metrics = BTreeMap::from([
                            (
                                "accuracy".to_string(),
                                json!(mock_bool_metric(model, &item.item_id)),
                            ),
                            (
                                "exact_match".to_string(),
                                json!(mock_bool_metric(model, &item.answer_hash)),
                            ),
                            (
                                "prompt_format".to_string(),
                                json!("lm_eval_micro_zero_shot_v1"),
                            ),
                            ("task".to_string(), json!(item.task.clone())),
                            ("answer_label".to_string(), json!(item.answer_label.clone())),
                            ("answer_hash".to_string(), json!(item.answer_hash.clone())),
                            ("choices_count".to_string(), json!(item.choices_count)),
                            ("scoring_mode".to_string(), json!("mock_exact_letter")),
                        ]);
                        add_dataset_provenance_metrics(&mut metrics, d);
                        rows.push(mock_pass_row(
                            BatteryId::Barrage,
                            Some(SuiteId::LmEvalMicro),
                            "lm_eval_micro_zero_shot_native",
                            Some(item.item_id.clone()),
                            config,
                            ctx,
                            Some(prompt_ref.clone()),
                            metrics,
                            model.clone(),
                        ));
                    }
                }
            }
            SuiteId::DeepSwe | SuiteId::SweBench => {
                let Ok(items) = builtin_barrage_materialized_items(d.suite, &d.selected_item_ids)
                else {
                    continue;
                };
                for item in items {
                    let prompt_ref = PromptRef::from_content(
                        format!("dataset:{}:{}", item.suite.as_str(), item.item_id),
                        item.prompt.as_bytes(),
                    );
                    let models = std::iter::once(&config.model)
                        .chain(config.baseline.iter())
                        .chain(config.reference.iter());
                    for model in models {
                        let mut metrics = BTreeMap::from([
                            (
                                "accuracy".to_string(),
                                json!(mock_bool_metric(model, &item.item_id)),
                            ),
                            (
                                "exact_match".to_string(),
                                json!(mock_bool_metric(model, &item.answer_hash)),
                            ),
                            (
                                "prompt_format".to_string(),
                                json!(item.prompt_format.clone()),
                            ),
                            ("task".to_string(), json!(item.task.clone())),
                            ("answer_label".to_string(), json!(item.answer_label.clone())),
                            ("answer_hash".to_string(), json!(item.answer_hash.clone())),
                            ("choices_count".to_string(), json!(item.choices_count)),
                            ("scoring_mode".to_string(), json!(item.scoring_mode.clone())),
                        ]);
                        add_dataset_provenance_metrics(&mut metrics, d);
                        rows.push(mock_pass_row(
                            BatteryId::Barrage,
                            Some(item.suite),
                            "builtin_software_eval_native",
                            Some(item.item_id.clone()),
                            config,
                            ctx,
                            Some(prompt_ref.clone()),
                            metrics,
                            model.clone(),
                        ));
                    }
                }
            }
            _ => {}
        }
    }
    if rows.is_empty() {
        barrage_rows(config, ctx, datasets)
    } else {
        rows
    }
}

#[allow(clippy::too_many_arguments)]
fn mock_pass_row(
    battery: BatteryId,
    suite: Option<SuiteId>,
    case_id: &str,
    dataset_item_id: Option<String>,
    config: &EvalConfig,
    ctx: &EvalContext,
    prompt: Option<PromptRef>,
    mut metrics: BTreeMap<String, Value>,
    model: String,
) -> EvalResult {
    metrics.insert("implemented".to_string(), json!(true));
    metrics.insert("executor".to_string(), json!("mock"));
    row_for_model(
        battery,
        suite,
        case_id,
        dataset_item_id,
        EvalStatus::Pass,
        Some("deterministic no-GPU mock executor".to_string()),
        metrics,
        config,
        ctx,
        prompt,
        0,
        model,
    )
}

fn mock_metric(model: &str, salt: &str, base: f64, spread: f64) -> f64 {
    base + (stable_score(&format!("{model}:{salt}")) * spread)
}

fn mock_bool_metric(model: &str, salt: &str) -> f64 {
    if stable_score(&format!("{model}:{salt}")) >= 0.5 {
        1.0
    } else {
        0.0
    }
}

fn stable_score(input: &str) -> f64 {
    hipfire_hash::stable_score(input)
}

fn examples_battery_rows(
    battery: BatteryId,
    config: &EvalConfig,
    ctx: &EvalContext,
    datasets: &[DatasetManifestEntry],
) -> Option<Vec<EvalResult>> {
    match battery {
        BatteryId::Smoke => Some(vec![
            run_examples_run_anchor_with_prompt(
                BatteryId::Smoke,
                "finite_greedy_decode",
                "benchmarks/prompts/qwen2_smoke.txt",
                config,
                ctx,
            ),
            run_direct_session_reset_recall(config, ctx),
        ]),
        BatteryId::Coherence => Some(run_examples_coherence_rows(config, ctx)),
        BatteryId::Speed => Some(run_examples_qwen35_speed_rows(config, ctx)),
        BatteryId::PromptShape => Some(vec![run_examples_run_anchor_with_prompt(
            BatteryId::PromptShape,
            "whitespace_template_canary",
            "benchmarks/prompts/lru_cache_pep8_strict.txt",
            config,
            ctx,
        )]),
        BatteryId::Structured => Some(vec![run_examples_run_anchor_with_prompt(
            BatteryId::Structured,
            "tool_call_jsonish_canary",
            "benchmarks/prompts/tool_call_read_file.txt",
            config,
            ctx,
        )]),
        BatteryId::Dflash => {
            let mut rows: Vec<_> = evaluation_models(config)
                .into_iter()
                .map(|model| {
                    run_dflash_spec_demo_anchor(
                        BatteryId::Dflash,
                        "ar_anchor",
                        true,
                        "benchmarks/prompts/dflash_resident_smoke.txt",
                        &[],
                        BTreeMap::new(),
                        config,
                        ctx,
                        model,
                    )
                })
                .collect();
            if matches!(config.dflash, DflashMode::Off) {
                rows.push(skip_row(
                    BatteryId::Dflash,
                    None,
                    "dflash_anchor",
                    None,
                    "DFlash disabled by --dflash off",
                    config,
                    ctx,
                    prompt("benchmarks/prompts/dflash_resident_smoke.txt"),
                ));
            } else {
                rows.push(run_dflash_spec_demo_anchor(
                    BatteryId::Dflash,
                    "dflash_anchor",
                    false,
                    "benchmarks/prompts/dflash_resident_smoke.txt",
                    &[],
                    BTreeMap::new(),
                    config,
                    ctx,
                    config.model.clone(),
                ));
            }
            rows.extend(run_examples_dflash_matrix_rows(config, ctx));
            Some(rows)
        }
        BatteryId::Pflash => Some(run_examples_pflash_rows(config, ctx)),
        BatteryId::Agentic => Some(run_examples_agentic_rows(config, ctx)),
        BatteryId::Runtime => Some(run_examples_runtime_rows(config, ctx)),
        BatteryId::Barrage => Some(examples_barrage_rows(config, ctx, datasets)),
        BatteryId::Longctx => Some(vec![run_examples_longctx_anchor(config, ctx)]),
        BatteryId::Profile => Some(
            evaluation_models(config)
                .into_iter()
                .map(|model| run_examples_profile_anchor(config, ctx, model))
                .collect(),
        ),
        BatteryId::Calibrate => Some(run_examples_calibrate_rows(config, ctx)),
        BatteryId::Perplexity => Some(run_examples_perplexity_rows(config, ctx)),
        _ => None,
    }
}

fn direct_battery_rows(
    battery: BatteryId,
    config: &EvalConfig,
    ctx: &EvalContext,
    _datasets: &[DatasetManifestEntry],
) -> Option<Vec<EvalResult>> {
    match battery {
        BatteryId::Smoke => Some(vec![
            run_examples_run_anchor_with_prompt(
                BatteryId::Smoke,
                "finite_greedy_decode",
                "benchmarks/prompts/qwen2_smoke.txt",
                config,
                ctx,
            ),
            run_direct_session_reset_recall(config, ctx),
        ]),
        _ => examples_battery_rows(battery, config, ctx, _datasets),
    }
}

fn daemon_battery_rows(
    battery: BatteryId,
    config: &EvalConfig,
    ctx: &EvalContext,
    datasets: &[DatasetManifestEntry],
) -> Option<Vec<EvalResult>> {
    match battery {
        BatteryId::Smoke => Some(run_daemon_smoke_rows(config, ctx)),
        BatteryId::Speed => Some(run_daemon_speed_rows(config, ctx)),
        BatteryId::Profile => Some(run_daemon_profile_rows(config, ctx)),
        BatteryId::Coherence | BatteryId::Longctx | BatteryId::Agentic => {
            examples_battery_rows(battery, config, ctx, datasets)
        }
        _ => None,
    }
}

struct DaemonEvalSession {
    engine: hipfire_daemon_adapter::DaemonEngine,
    loaded: ModelLoadedResponse,
    worker_key_id: String,
    max_seq: usize,
}

async fn load_daemon_eval_session(
    config: &EvalConfig,
    bin: &Path,
    max_seq: usize,
) -> anyhow::Result<DaemonEvalSession> {
    let mut engine = hipfire_daemon_adapter::DaemonEngine::spawn(bin).await?;
    let loaded = engine
        .load(&config.model, daemon_model_load_params(config, max_seq))
        .await?;
    let worker_key_id = loaded.worker_key_id.clone();
    Ok(DaemonEvalSession {
        engine,
        loaded,
        worker_key_id,
        max_seq,
    })
}

fn run_daemon_shared_model_load_rows(
    config: &EvalConfig,
    ctx: &EvalContext,
    batteries: &[BatteryId],
) -> BTreeMap<BatteryId, Vec<EvalResult>> {
    let mut out = BTreeMap::new();
    if !Path::new(&config.model).exists() {
        for battery in batteries {
            let rows = match battery {
                BatteryId::Smoke => daemon_smoke_skip_rows(
                    config,
                    ctx,
                    "daemon executor requires --model to be a local filesystem path",
                    "daemon executor requires --model to be a local filesystem path",
                ),
                BatteryId::Speed => daemon_speed_skip_rows(
                    config,
                    ctx,
                    "daemon executor requires --model to be a local filesystem path",
                ),
                BatteryId::Profile => daemon_profile_skip_rows(
                    config,
                    ctx,
                    "daemon executor requires --model to be a local filesystem path",
                ),
                _ => Vec::new(),
            };
            out.insert(*battery, rows);
        }
        return out;
    }

    let Some(bin) = hipfire_daemon_adapter::find_daemon_bin() else {
        for battery in batteries {
            let rows = match battery {
                BatteryId::Smoke => daemon_smoke_skip_rows(
                    config,
                    ctx,
                    "daemon binary not found; build with `cargo build -p hipfire-daemon --bin hipfire-daemon`",
                    "daemon binary not found; build with `cargo build -p hipfire-daemon --bin hipfire-daemon`",
                ),
                BatteryId::Speed => daemon_speed_skip_rows(
                    config,
                    ctx,
                    "daemon binary not found; build with `cargo build -p hipfire-daemon --bin hipfire-daemon`",
                ),
                BatteryId::Profile => daemon_profile_skip_rows(
                    config,
                    ctx,
                    "daemon binary not found; build with `cargo build -p hipfire-daemon --bin hipfire-daemon`",
                ),
                _ => Vec::new(),
            };
            out.insert(*battery, rows);
        }
        return out;
    };

    let started = SystemTime::now();
    let runtime = match tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
    {
        Ok(runtime) => runtime,
        Err(err) => {
            for battery in batteries {
                let rows = match battery {
                    BatteryId::Smoke => daemon_smoke_skip_rows(
                        config,
                        ctx,
                        &format!("create daemon executor runtime: {err}"),
                        "daemon executor runtime creation failed before decode",
                    ),
                    BatteryId::Speed => daemon_speed_skip_rows(
                        config,
                        ctx,
                        &format!("create daemon executor runtime: {err}"),
                    ),
                    BatteryId::Profile => daemon_profile_skip_rows(
                        config,
                        ctx,
                        &format!("create daemon executor runtime: {err}"),
                    ),
                    _ => Vec::new(),
                };
                out.insert(*battery, rows);
            }
            return out;
        }
    };

    let max_seq = (config.max_tokens.max(50) + 2048).max(4096);
    match runtime.block_on(run_daemon_shared_model_load_rows_async(
        config, ctx, &bin, batteries, max_seq,
    )) {
        Ok(mut rows_by_battery) => {
            let elapsed_ms = elapsed_since_ms(started);
            for rows in rows_by_battery.values_mut() {
                for row in rows {
                    row.elapsed_ms = elapsed_ms;
                    row.metrics
                        .insert("shared_daemon_session".to_string(), json!(true));
                }
            }
            rows_by_battery
        }
        Err(err) => {
            for battery in batteries {
                let rows = match battery {
                    BatteryId::Smoke => daemon_shared_smoke_failure_rows(
                        config,
                        ctx,
                        &bin,
                        &format!("daemon-backed shared executor failed: {err}"),
                        elapsed_since_ms(started),
                    ),
                    BatteryId::Speed => daemon_shared_speed_failure_rows(
                        config,
                        ctx,
                        &bin,
                        &format!("daemon-backed shared executor failed: {err}"),
                        elapsed_since_ms(started),
                    ),
                    BatteryId::Profile => daemon_shared_profile_failure_rows(
                        config,
                        ctx,
                        &bin,
                        &format!("daemon-backed shared executor failed: {err}"),
                        elapsed_since_ms(started),
                    ),
                    _ => Vec::new(),
                };
                out.insert(*battery, rows);
            }
            out
        }
    }
}

async fn run_daemon_shared_model_load_rows_async(
    config: &EvalConfig,
    ctx: &EvalContext,
    bin: &Path,
    batteries: &[BatteryId],
    max_seq: usize,
) -> anyhow::Result<BTreeMap<BatteryId, Vec<EvalResult>>> {
    let mut session = load_daemon_eval_session(config, bin, max_seq).await?;
    let mut out = BTreeMap::new();
    for battery in batteries {
        let rows = match battery {
            BatteryId::Smoke => {
                daemon_smoke_rows_with_session(config, ctx, bin, &mut session).await?
            }
            BatteryId::Speed => {
                daemon_speed_rows_with_session(config, ctx, bin, &mut session).await?
            }
            BatteryId::Profile => {
                daemon_profile_rows_with_session(config, ctx, bin, &mut session).await?
            }
            _ => Vec::new(),
        };
        out.insert(*battery, rows);
    }
    Ok(out)
}

fn daemon_model_load_params(config: &EvalConfig, max_seq: usize) -> ModelLoadParams {
    ModelLoadParams {
        max_seq: max_seq.min(u32::MAX as usize) as u32,
        kv_cache: config.kv_mode.clone(),
        dflash_mode: Some(config.dflash.as_str().to_string()),
        draft: config.draft.clone(),
        ..Default::default()
    }
}

fn daemon_generate_request(
    id: String,
    prompt_text: String,
    max_tokens: usize,
    worker_key_id: Option<String>,
    evidence_dir: Option<&Path>,
) -> GenerateTextRequest {
    let mut request = GenerateTextRequest::from_prompt(
        id,
        prompt_text,
        GenerationSamplingPolicy::greedy(max_tokens.min(u32::MAX as usize) as u32),
    )
    .with_worker_key_id(worker_key_id);
    request.evidence_dir = evidence_dir.map(|dir| dir.display().to_string());
    request
}

fn read_repo_prompt_text(prompt_path: &str) -> anyhow::Result<String> {
    let prompt_file = resolve_repo_path(prompt_path)
        .ok_or_else(|| anyhow::anyhow!("resolve {prompt_path} from repo root"))?;
    fs::read_to_string(&prompt_file).map_err(|e| anyhow::anyhow!("read {prompt_path}: {e}"))
}

fn daemon_smoke_skip_rows(
    config: &EvalConfig,
    ctx: &EvalContext,
    load_reason: &str,
    decode_reason: &str,
) -> Vec<EvalResult> {
    vec![
        skip_row_with_metrics(
            BatteryId::Smoke,
            None,
            "load_metadata",
            None,
            load_reason,
            config,
            ctx,
            None,
            BTreeMap::from([("executor".to_string(), json!("daemon"))]),
        ),
        skip_row_with_metrics(
            BatteryId::Smoke,
            None,
            "finite_greedy_decode",
            None,
            decode_reason,
            config,
            ctx,
            prompt("benchmarks/prompts/qwen2_smoke.txt"),
            BTreeMap::from([("executor".to_string(), json!("daemon"))]),
        ),
        skip_row_with_metrics(
            BatteryId::Smoke,
            None,
            "multi_turn_reset_recall",
            None,
            load_reason,
            config,
            ctx,
            prompt("benchmarks/prompts/trains-meet.txt"),
            BTreeMap::from([("executor".to_string(), json!("daemon"))]),
        ),
    ]
}

fn daemon_speed_skip_rows(config: &EvalConfig, ctx: &EvalContext, reason: &str) -> Vec<EvalResult> {
    let prompt_ref = prompt("benchmarks/prompts/lru_cache_single_blank.txt");
    daemon_speed_cases()
        .iter()
        .map(|case| {
            skip_row_with_metrics(
                BatteryId::Speed,
                None,
                case.label,
                None,
                reason,
                config,
                ctx,
                prompt_ref.clone(),
                BTreeMap::from([
                    ("implemented".to_string(), json!(true)),
                    ("executor".to_string(), json!("daemon")),
                    ("suite".to_string(), json!("daemon_speed_anchor")),
                ]),
            )
        })
        .collect()
}

fn daemon_profile_expected_runtime_evidence_kinds() -> Value {
    json!([
        "performance",
        "memory",
        "launch_counts",
        "moe_router_histogram"
    ])
}

fn daemon_profile_base_metrics() -> BTreeMap<String, Value> {
    BTreeMap::from([
        ("implemented".to_string(), json!(true)),
        ("executor".to_string(), json!("daemon")),
        ("suite".to_string(), json!("daemon_profile_anchor")),
        ("profile_requested".to_string(), json!(true)),
        (
            "collection_scope".to_string(),
            json!("model_backed_daemon_anchor"),
        ),
        (
            "moe_router_histogram_expected_when_moe".to_string(),
            json!(true),
        ),
        (
            "expected_runtime_evidence_kinds".to_string(),
            daemon_profile_expected_runtime_evidence_kinds(),
        ),
    ])
}

fn daemon_profile_skip_rows(
    config: &EvalConfig,
    ctx: &EvalContext,
    reason: &str,
) -> Vec<EvalResult> {
    vec![skip_row_with_metrics(
        BatteryId::Profile,
        None,
        "model_profile_anchor",
        None,
        reason,
        config,
        ctx,
        prompt("benchmarks/prompts/dflash_resident_smoke.txt"),
        daemon_profile_base_metrics(),
    )]
}

fn run_daemon_smoke_rows(config: &EvalConfig, ctx: &EvalContext) -> Vec<EvalResult> {
    if !Path::new(&config.model).exists() {
        return daemon_smoke_skip_rows(
            config,
            ctx,
            "daemon executor requires --model to be a local filesystem path",
            "daemon executor requires --model to be a local filesystem path",
        );
    }

    let Some(bin) = hipfire_daemon_adapter::find_daemon_bin() else {
        return daemon_smoke_skip_rows(
            config,
            ctx,
            "daemon binary not found; build with `cargo build -p hipfire-daemon --bin hipfire-daemon`",
            "daemon binary not found; build with `cargo build -p hipfire-daemon --bin hipfire-daemon`",
        );
    };

    let started = SystemTime::now();
    let runtime = match tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
    {
        Ok(runtime) => runtime,
        Err(err) => {
            return daemon_smoke_skip_rows(
                config,
                ctx,
                &format!("create daemon executor runtime: {err}"),
                "daemon executor runtime creation failed before decode",
            );
        }
    };

    match runtime.block_on(run_daemon_smoke_rows_async(config, ctx, &bin)) {
        Ok(mut rows) => {
            let elapsed_ms = elapsed_since_ms(started);
            for row in &mut rows {
                row.elapsed_ms = elapsed_ms;
            }
            rows
        }
        Err(err) => vec![
            row(
                BatteryId::Smoke,
                None,
                "load_metadata",
                None,
                EvalStatus::Fail,
                Some(format!("daemon-backed smoke executor failed: {err}")),
                BTreeMap::from([
                    ("executor".to_string(), json!("daemon")),
                    ("daemon_bin".to_string(), json!(bin.display().to_string())),
                ]),
                config,
                ctx,
                None,
                elapsed_since_ms(started),
            ),
            row(
                BatteryId::Smoke,
                None,
                "finite_greedy_decode",
                None,
                EvalStatus::Skip,
                Some("daemon-backed load failed before decode".to_string()),
                BTreeMap::from([("executor".to_string(), json!("daemon"))]),
                config,
                ctx,
                prompt("benchmarks/prompts/qwen2_smoke.txt"),
                elapsed_since_ms(started),
            ),
            skip_row_with_metrics(
                BatteryId::Smoke,
                None,
                "multi_turn_reset_recall",
                None,
                "daemon-backed load failed before session reset/recall",
                config,
                ctx,
                prompt("benchmarks/prompts/trains-meet.txt"),
                BTreeMap::from([("executor".to_string(), json!("daemon"))]),
            ),
        ],
    }
}

fn run_daemon_speed_rows(config: &EvalConfig, ctx: &EvalContext) -> Vec<EvalResult> {
    if !Path::new(&config.model).exists() {
        return daemon_speed_skip_rows(
            config,
            ctx,
            "daemon executor requires --model to be a local filesystem path",
        );
    }

    let Some(bin) = hipfire_daemon_adapter::find_daemon_bin() else {
        return daemon_speed_skip_rows(
            config,
            ctx,
            "daemon binary not found; build with `cargo build -p hipfire-daemon --bin hipfire-daemon`",
        );
    };

    let started = SystemTime::now();
    let runtime = match tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
    {
        Ok(runtime) => runtime,
        Err(err) => {
            return daemon_speed_skip_rows(
                config,
                ctx,
                &format!("create daemon executor runtime: {err}"),
            );
        }
    };

    match runtime.block_on(run_daemon_speed_rows_async(config, ctx, &bin)) {
        Ok(mut rows) => {
            let elapsed_ms = elapsed_since_ms(started);
            for row in &mut rows {
                row.elapsed_ms = elapsed_ms;
            }
            rows
        }
        Err(err) => daemon_speed_cases()
            .iter()
            .map(|case| {
                row(
                    BatteryId::Speed,
                    None,
                    case.label,
                    None,
                    EvalStatus::Fail,
                    Some(format!("daemon-backed speed executor failed: {err}")),
                    BTreeMap::from([
                        ("implemented".to_string(), json!(true)),
                        ("executor".to_string(), json!("daemon")),
                        ("suite".to_string(), json!("daemon_speed_anchor")),
                        ("daemon_bin".to_string(), json!(bin.display().to_string())),
                    ]),
                    config,
                    ctx,
                    prompt("benchmarks/prompts/lru_cache_single_blank.txt"),
                    elapsed_since_ms(started),
                )
            })
            .collect(),
    }
}

fn run_daemon_profile_rows(config: &EvalConfig, ctx: &EvalContext) -> Vec<EvalResult> {
    if !Path::new(&config.model).exists() {
        return daemon_profile_skip_rows(
            config,
            ctx,
            "daemon executor requires --model to be a local filesystem path",
        );
    }

    let Some(bin) = hipfire_daemon_adapter::find_daemon_bin() else {
        return daemon_profile_skip_rows(
            config,
            ctx,
            "daemon binary not found; build with `cargo build -p hipfire-daemon --bin hipfire-daemon`",
        );
    };

    let started = SystemTime::now();
    let runtime = match tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
    {
        Ok(runtime) => runtime,
        Err(err) => {
            return daemon_profile_skip_rows(
                config,
                ctx,
                &format!("create daemon executor runtime: {err}"),
            );
        }
    };

    match runtime.block_on(run_daemon_profile_rows_async(config, ctx, &bin)) {
        Ok(mut rows) => {
            let elapsed_ms = elapsed_since_ms(started);
            for row in &mut rows {
                row.elapsed_ms = elapsed_ms;
            }
            rows
        }
        Err(err) => daemon_shared_profile_failure_rows(
            config,
            ctx,
            &bin,
            &format!("daemon-backed profile executor failed: {err}"),
            elapsed_since_ms(started),
        ),
    }
}

fn daemon_shared_smoke_failure_rows(
    config: &EvalConfig,
    ctx: &EvalContext,
    bin: &Path,
    reason: &str,
    elapsed_ms: u128,
) -> Vec<EvalResult> {
    vec![
        row(
            BatteryId::Smoke,
            None,
            "load_metadata",
            None,
            EvalStatus::Fail,
            Some(reason.to_string()),
            BTreeMap::from([
                ("executor".to_string(), json!("daemon")),
                ("shared_daemon_session".to_string(), json!(true)),
                ("daemon_bin".to_string(), json!(bin.display().to_string())),
            ]),
            config,
            ctx,
            None,
            elapsed_ms,
        ),
        row(
            BatteryId::Smoke,
            None,
            "finite_greedy_decode",
            None,
            EvalStatus::Skip,
            Some("daemon-backed shared load failed before decode".to_string()),
            BTreeMap::from([
                ("executor".to_string(), json!("daemon")),
                ("shared_daemon_session".to_string(), json!(true)),
            ]),
            config,
            ctx,
            prompt("benchmarks/prompts/qwen2_smoke.txt"),
            elapsed_ms,
        ),
        skip_row_with_metrics(
            BatteryId::Smoke,
            None,
            "multi_turn_reset_recall",
            None,
            "daemon-backed shared load failed before session reset/recall",
            config,
            ctx,
            prompt("benchmarks/prompts/trains-meet.txt"),
            BTreeMap::from([
                ("executor".to_string(), json!("daemon")),
                ("shared_daemon_session".to_string(), json!(true)),
            ]),
        ),
    ]
}

fn daemon_shared_speed_failure_rows(
    config: &EvalConfig,
    ctx: &EvalContext,
    bin: &Path,
    reason: &str,
    elapsed_ms: u128,
) -> Vec<EvalResult> {
    daemon_speed_cases()
        .iter()
        .map(|case| {
            row(
                BatteryId::Speed,
                None,
                case.label,
                None,
                EvalStatus::Fail,
                Some(reason.to_string()),
                BTreeMap::from([
                    ("implemented".to_string(), json!(true)),
                    ("executor".to_string(), json!("daemon")),
                    ("suite".to_string(), json!("daemon_speed_anchor")),
                    ("shared_daemon_session".to_string(), json!(true)),
                    ("daemon_bin".to_string(), json!(bin.display().to_string())),
                ]),
                config,
                ctx,
                prompt("benchmarks/prompts/lru_cache_single_blank.txt"),
                elapsed_ms,
            )
        })
        .collect()
}

fn daemon_shared_profile_failure_rows(
    config: &EvalConfig,
    ctx: &EvalContext,
    bin: &Path,
    reason: &str,
    elapsed_ms: u128,
) -> Vec<EvalResult> {
    let mut metrics = daemon_profile_base_metrics();
    metrics.insert("shared_daemon_session".to_string(), json!(true));
    metrics.insert("daemon_bin".to_string(), json!(bin.display().to_string()));
    vec![row(
        BatteryId::Profile,
        None,
        "model_profile_anchor",
        None,
        EvalStatus::Fail,
        Some(reason.to_string()),
        metrics,
        config,
        ctx,
        prompt("benchmarks/prompts/dflash_resident_smoke.txt"),
        elapsed_ms,
    )]
}

async fn run_daemon_smoke_rows_async(
    config: &EvalConfig,
    ctx: &EvalContext,
    bin: &Path,
) -> anyhow::Result<Vec<EvalResult>> {
    let max_seq = (config.max_tokens + 2048).max(4096);
    let mut session = load_daemon_eval_session(config, bin, max_seq).await?;
    daemon_smoke_rows_with_session(config, ctx, bin, &mut session).await
}

async fn daemon_smoke_rows_with_session(
    config: &EvalConfig,
    ctx: &EvalContext,
    bin: &Path,
    session: &mut DaemonEvalSession,
) -> anyhow::Result<Vec<EvalResult>> {
    let worker_key_id = session.worker_key_id.clone();
    let prompt_path = "benchmarks/prompts/qwen2_smoke.txt";
    let prompt_text = read_repo_prompt_text(prompt_path)?;
    let request = daemon_generate_request(
        "eval-smoke-greedy".to_string(),
        prompt_text,
        config.max_tokens,
        Some(worker_key_id.clone()),
        None,
    );
    let (text, done) = session.engine.generate(request).await?;
    let finite = !text.is_empty() && !text.contains('\u{fffd}');
    let decode_status = if finite {
        EvalStatus::Pass
    } else {
        EvalStatus::Fail
    };
    let decode_reason =
        (!finite).then(|| "daemon returned empty or replacement-character output".to_string());

    let session_prompt_path = "benchmarks/prompts/trains-meet.txt";
    let session_prompt_text = read_repo_prompt_text(session_prompt_path)?;
    session.engine.reset().await?;
    let first_session_request = daemon_generate_request(
        "eval-smoke-session-fresh".to_string(),
        session_prompt_text.clone(),
        config.max_tokens,
        Some(worker_key_id.clone()),
        None,
    );
    let (first_session_text, first_session_done) =
        session.engine.generate(first_session_request).await?;
    let distractor_request = daemon_generate_request(
        "eval-smoke-session-distractor".to_string(),
        "Remember this unrelated code word for the next turn: orchid. Reply with only OK."
            .to_string(),
        config.max_tokens,
        Some(worker_key_id.clone()),
        None,
    );
    let (distractor_text, distractor_done) = session.engine.generate(distractor_request).await?;
    session.engine.reset().await?;
    let second_session_request = daemon_generate_request(
        "eval-smoke-session-reset".to_string(),
        session_prompt_text,
        config.max_tokens,
        Some(worker_key_id.clone()),
        None,
    );
    let (second_session_text, second_session_done) =
        session.engine.generate(second_session_request).await?;
    let session_finite = !first_session_text.is_empty()
        && !second_session_text.is_empty()
        && !first_session_text.contains('\u{fffd}')
        && !second_session_text.contains('\u{fffd}');
    let session_match = first_session_text == second_session_text;
    let session_status = if session_finite && session_match {
        EvalStatus::Pass
    } else {
        EvalStatus::Fail
    };
    let session_reason = if !session_finite {
        Some(
            "daemon session reset smoke returned empty or replacement-character output".to_string(),
        )
    } else if !session_match {
        Some("daemon repeated greedy session request produced different output".to_string())
    } else {
        None
    };

    Ok(vec![
        row(
            BatteryId::Smoke,
            None,
            "load_metadata",
            None,
            EvalStatus::Pass,
            None,
            BTreeMap::from([
                ("executor".to_string(), json!("daemon")),
                ("daemon_bin".to_string(), json!(bin.display().to_string())),
                ("shared_model_loads".to_string(), json!(1)),
                ("worker_key_id".to_string(), json!(worker_key_id.clone())),
                ("arch".to_string(), json!(session.loaded.arch)),
                ("dim".to_string(), json!(session.loaded.dim)),
                ("layers".to_string(), json!(session.loaded.layers)),
                ("vocab".to_string(), json!(session.loaded.vocab)),
                ("max_seq".to_string(), json!(session.max_seq)),
            ]),
            config,
            ctx,
            None,
            0,
        ),
        row(
            BatteryId::Smoke,
            None,
            "finite_greedy_decode",
            None,
            decode_status,
            decode_reason,
            BTreeMap::from([
                ("executor".to_string(), json!("daemon")),
                ("shared_model_loads".to_string(), json!(1)),
                ("worker_key_id".to_string(), json!(worker_key_id.clone())),
                ("tokens".to_string(), json!(done.tokens)),
                ("text_bytes".to_string(), json!(text.len())),
                ("tok_s".to_string(), json!(done.tok_s)),
                ("ttft_ms".to_string(), json!(done.ttft_ms)),
                ("max_tokens".to_string(), json!(config.max_tokens)),
            ]),
            config,
            ctx,
            prompt(prompt_path),
            0,
        ),
        row(
            BatteryId::Smoke,
            None,
            "multi_turn_reset_recall",
            None,
            session_status,
            session_reason,
            BTreeMap::from([
                ("executor".to_string(), json!("daemon")),
                ("implemented".to_string(), json!(true)),
                ("shared_model_loads".to_string(), json!(1)),
                ("worker_key_id".to_string(), json!(worker_key_id)),
                ("reset_count".to_string(), json!(2)),
                ("kv_reset".to_string(), json!(true)),
                ("dn_state_reset".to_string(), json!(true)),
                ("session_turns".to_string(), json!(3)),
                ("first_tokens".to_string(), json!(first_session_done.tokens)),
                (
                    "distractor_tokens".to_string(),
                    json!(distractor_done.tokens),
                ),
                (
                    "second_tokens".to_string(),
                    json!(second_session_done.tokens),
                ),
                (
                    "first_text_hash".to_string(),
                    json!(stable_hash_bytes(first_session_text.as_bytes())),
                ),
                (
                    "second_text_hash".to_string(),
                    json!(stable_hash_bytes(second_session_text.as_bytes())),
                ),
                (
                    "distractor_text_hash".to_string(),
                    json!(stable_hash_bytes(distractor_text.as_bytes())),
                ),
                ("outputs_match".to_string(), json!(session_match)),
                ("max_tokens".to_string(), json!(config.max_tokens)),
            ]),
            config,
            ctx,
            prompt(session_prompt_path),
            0,
        ),
    ])
}

async fn run_daemon_speed_rows_async(
    config: &EvalConfig,
    ctx: &EvalContext,
    bin: &Path,
) -> anyhow::Result<Vec<EvalResult>> {
    let max_seq = (config.max_tokens + 2048).max(4096);
    let mut session = load_daemon_eval_session(config, bin, max_seq).await?;
    daemon_speed_rows_with_session(config, ctx, bin, &mut session).await
}

async fn run_daemon_profile_rows_async(
    config: &EvalConfig,
    ctx: &EvalContext,
    bin: &Path,
) -> anyhow::Result<Vec<EvalResult>> {
    let max_seq = (config.max_tokens.max(50) + 2048).max(4096);
    let mut session = load_daemon_eval_session(config, bin, max_seq).await?;
    daemon_profile_rows_with_session(config, ctx, bin, &mut session).await
}

async fn daemon_speed_rows_with_session(
    config: &EvalConfig,
    ctx: &EvalContext,
    bin: &Path,
    session: &mut DaemonEvalSession,
) -> anyhow::Result<Vec<EvalResult>> {
    let worker_key_id = session.worker_key_id.clone();
    let prompt_path = "benchmarks/prompts/lru_cache_single_blank.txt";
    let prompt_text = read_repo_prompt_text(prompt_path)?;
    let max_tokens = config.max_tokens.max(50);

    let mut rows = Vec::new();
    for case in daemon_speed_cases() {
        session.engine.reset().await?;
        let evidence_dir = runtime_evidence_dir(
            config,
            &format!("daemon-speed-{}", case.label),
            &config.model,
        );
        let request = daemon_generate_request(
            format!("eval-speed-{}", case.label),
            prompt_text.clone(),
            max_tokens,
            Some(worker_key_id.clone()),
            Some(&evidence_dir),
        );
        let (text, done) = session.engine.generate(request).await?;
        let has_timing = done.prefill_tok_s.is_some() && done.decode_tok_s.is_some();
        let finite = !text.is_empty() && !text.contains('\u{fffd}') && done.tokens > 0;
        let status = if finite && has_timing {
            EvalStatus::Pass
        } else {
            EvalStatus::Fail
        };
        let reason = if !finite {
            Some(
                "daemon speed anchor returned empty, zero-token, or replacement-character output"
                    .to_string(),
            )
        } else if !has_timing {
            Some("daemon speed anchor did not emit prefill/decode timing metrics".to_string())
        } else {
            None
        };
        let mut metrics = BTreeMap::from([
            ("implemented".to_string(), json!(true)),
            ("executor".to_string(), json!("daemon")),
            ("suite".to_string(), json!("daemon_speed_anchor")),
            ("shared_model_loads".to_string(), json!(1)),
            ("worker_key_id".to_string(), json!(worker_key_id.clone())),
            ("daemon_bin".to_string(), json!(bin.display().to_string())),
            ("max_tokens".to_string(), json!(max_tokens)),
            ("tokens".to_string(), json!(done.tokens)),
            ("text_bytes".to_string(), json!(text.len())),
            (
                "runtime_evidence_dir".to_string(),
                json!(evidence_dir.display().to_string()),
            ),
        ]);
        if let Some(value) = done.tok_s {
            metrics.insert("tok_s".to_string(), json!(value));
        }
        if let Some(value) = done.prefill_tokens {
            metrics.insert("prefill_tokens".to_string(), json!(value));
        }
        if let Some(value) = done.prefill_ms {
            metrics.insert("prefill_ms".to_string(), json!(value));
        }
        if let Some(value) = done.prefill_tok_s {
            metrics.insert("prefill_tok_s".to_string(), json!(value));
        }
        if let Some(value) = done.decode_tok_s {
            metrics.insert("decode_tok_s".to_string(), json!(value));
            metrics
                .entry("gen_tok_s".to_string())
                .or_insert(json!(value));
        }
        if let Some(value) = done.ttft_ms {
            metrics.insert("ttft_ms".to_string(), json!(value));
        }

        rows.push(row_for_model(
            BatteryId::Speed,
            None,
            case.label,
            None,
            status,
            reason,
            metrics,
            config,
            ctx,
            prompt(prompt_path),
            0,
            config.model.clone(),
        ));
    }

    Ok(rows)
}

async fn daemon_profile_rows_with_session(
    config: &EvalConfig,
    ctx: &EvalContext,
    bin: &Path,
    session: &mut DaemonEvalSession,
) -> anyhow::Result<Vec<EvalResult>> {
    let worker_key_id = session.worker_key_id.clone();
    let prompt_path = "benchmarks/prompts/dflash_resident_smoke.txt";
    let prompt_text = read_repo_prompt_text(prompt_path)?;
    let max_tokens = config.max_tokens.max(50);
    let evidence_dir =
        runtime_evidence_dir(config, "daemon-profile-model_profile_anchor", &config.model);
    session.engine.reset().await?;
    let request = daemon_generate_request(
        "eval-profile-model_profile_anchor".to_string(),
        prompt_text,
        max_tokens,
        Some(worker_key_id.clone()),
        Some(&evidence_dir),
    );
    let (text, done) = session.engine.generate(request).await?;
    let finite = !text.is_empty() && !text.contains('\u{fffd}') && done.tokens > 0;
    let has_timing = done.prefill_tok_s.is_some() && done.decode_tok_s.is_some();
    let status = if finite && has_timing {
        EvalStatus::Pass
    } else {
        EvalStatus::Fail
    };
    let reason = if !finite {
        Some(
            "daemon profile anchor returned empty, zero-token, or replacement-character output"
                .to_string(),
        )
    } else if !has_timing {
        Some("daemon profile anchor did not emit prefill/decode timing metrics".to_string())
    } else {
        None
    };
    let mut metrics = daemon_profile_base_metrics();
    metrics.extend([
        ("shared_model_loads".to_string(), json!(1)),
        ("worker_key_id".to_string(), json!(worker_key_id)),
        ("daemon_bin".to_string(), json!(bin.display().to_string())),
        ("max_tokens".to_string(), json!(max_tokens)),
        ("tokens".to_string(), json!(done.tokens)),
        ("text_bytes".to_string(), json!(text.len())),
        (
            "runtime_evidence_dir".to_string(),
            json!(evidence_dir.display().to_string()),
        ),
    ]);
    if let Some(value) = done.tok_s {
        metrics.insert("tok_s".to_string(), json!(value));
    }
    if let Some(value) = done.prefill_tokens {
        metrics.insert("prefill_tokens".to_string(), json!(value));
    }
    if let Some(value) = done.prefill_ms {
        metrics.insert("prefill_ms".to_string(), json!(value));
    }
    if let Some(value) = done.prefill_tok_s {
        metrics.insert("prefill_tok_s".to_string(), json!(value));
    }
    if let Some(value) = done.decode_tok_s {
        metrics.insert("decode_tok_s".to_string(), json!(value));
        metrics
            .entry("gen_tok_s".to_string())
            .or_insert(json!(value));
    }
    if let Some(value) = done.ttft_ms {
        metrics.insert("ttft_ms".to_string(), json!(value));
    }

    Ok(vec![row(
        BatteryId::Profile,
        None,
        "model_profile_anchor",
        None,
        status,
        reason,
        metrics,
        config,
        ctx,
        prompt(prompt_path),
        0,
    )])
}

fn examples_executor_available_for(battery: BatteryId) -> bool {
    match battery {
        BatteryId::Coherence => hipfire_coherence::daemon_binary_available(),
        BatteryId::Smoke
        | BatteryId::PromptShape
        | BatteryId::Structured
        | BatteryId::Barrage
        | BatteryId::Profile => resolve_run_example_bin().is_some(),
        BatteryId::Longctx => hipfire_coherence::daemon_binary_available(),
        BatteryId::Speed => resolve_bench_qwen35_speed_bin().is_some(),
        BatteryId::Dflash => resolve_dflash_spec_demo_bin().is_some(),
        BatteryId::Pflash => resolve_pflash_niah_bench_bin().is_some(),
        BatteryId::Agentic => hipfire_coherence::daemon_binary_available(),
        BatteryId::Runtime => true,
        BatteryId::Calibrate => resolve_collect_artifacts_bin().is_some(),
        BatteryId::Perplexity => resolve_perplexity_bin().is_some(),
        _ => false,
    }
}

#[derive(Clone, Copy)]
struct CoherenceCase {
    label: &'static str,
    prompt_path: &'static str,
    system_path: Option<&'static str>,
    max_tokens: usize,
}

fn coherence_cases() -> &'static [CoherenceCase] {
    &[
        CoherenceCase {
            label: "capital_short_answer",
            prompt_path: "benchmarks/prompts/coherence_capital_france.txt",
            system_path: None,
            max_tokens: 80,
        },
        CoherenceCase {
            label: "code_square_function",
            prompt_path: "benchmarks/prompts/coherence_square_function.txt",
            system_path: None,
            max_tokens: 180,
        },
        CoherenceCase {
            label: "reason_sheep",
            prompt_path: "benchmarks/prompts/coherence_sheep_reason.txt",
            system_path: None,
            max_tokens: 300,
        },
        CoherenceCase {
            label: "tool_call_read_file",
            prompt_path: "benchmarks/prompts/tool_call_read_file.txt",
            system_path: Some("benchmarks/prompts/tool_call_system.txt"),
            max_tokens: 180,
        },
        CoherenceCase {
            label: "long_prefill_lloyd",
            prompt_path: "benchmarks/prompts/coherence_lloyd_long.txt",
            system_path: None,
            max_tokens: 220,
        },
    ]
}

fn run_examples_coherence_rows(config: &EvalConfig, ctx: &EvalContext) -> Vec<EvalResult> {
    coherence_cases()
        .iter()
        .map(|case| run_examples_coherence_case(config, ctx, *case))
        .collect()
}

fn run_examples_coherence_case(
    config: &EvalConfig,
    ctx: &EvalContext,
    case: CoherenceCase,
) -> EvalResult {
    let prompt_ref = if let Some(system_path) = case.system_path {
        combined_prompt_ref(system_path, case.prompt_path)
    } else {
        prompt(case.prompt_path)
    };
    let metrics = BTreeMap::from([
        ("suite".to_string(), json!("coherence_gate")),
        ("prompt_kind".to_string(), json!(case.label)),
        (
            "detector_profile".to_string(),
            json!(if case.system_path.is_some() {
                "agentic_toolcall_shape"
            } else {
                "default_runtime_coherence"
            }),
        ),
    ]);
    run_daemon_coherence_anchor(
        BatteryId::Coherence,
        case.label,
        case.prompt_path,
        prompt_ref,
        config,
        ctx,
        config.model.clone(),
        Some((case.max_tokens + 2048).max(4096)),
        Some(case.max_tokens),
        metrics,
        case.system_path,
        None,
        None,
        false,
        None,
    )
}

fn run_examples_profile_anchor(
    config: &EvalConfig,
    ctx: &EvalContext,
    model: String,
) -> EvalResult {
    run_examples_run_anchor_with_prompt_ref_for_model(
        BatteryId::Profile,
        "model_profile_anchor",
        "benchmarks/prompts/dflash_resident_smoke.txt",
        prompt("benchmarks/prompts/dflash_resident_smoke.txt"),
        config,
        ctx,
        model,
        None,
        BTreeMap::from([
            ("profile_requested".to_string(), json!(true)),
            (
                "collection_scope".to_string(),
                json!("model_backed_run_anchor"),
            ),
            (
                "moe_router_histogram_expected_when_moe".to_string(),
                json!(true),
            ),
            (
                "expected_runtime_evidence_kinds".to_string(),
                json!([
                    "performance",
                    "memory",
                    "launch_counts",
                    "moe_router_histogram"
                ]),
            ),
        ]),
    )
}

fn run_examples_longctx_anchor(config: &EvalConfig, ctx: &EvalContext) -> EvalResult {
    match materialize_longctx_prompt(config) {
        Ok(longctx) => run_daemon_coherence_anchor(
            BatteryId::Longctx,
            "multidoc_needle_long_state",
            &longctx.prompt_path,
            Some(longctx.prompt_ref),
            config,
            ctx,
            config.model.clone(),
            Some(longctx.max_seq),
            None,
            longctx.metrics,
            None,
            None,
            None,
            false,
            Some(hipfire_coherence::DetectorProfile::long_state()),
        ),
        Err(err) => row(
            BatteryId::Longctx,
            None,
            "multidoc_needle_native",
            None,
            EvalStatus::Fail,
            Some(err),
            BTreeMap::from([("fixture".to_string(), json!("longprose_multidoc"))]),
            config,
            ctx,
            prompt("benchmarks/prompts/longprose_multidoc.jsonl"),
            0,
        ),
    }
}

/// Tier-1 calibration battery: run the single-load `collect_artifacts` example
/// over each evaluation model and assert the on-GPU Hessian/imatrix internal
/// consistency (`diag(Σxxᵀ)==Σx²`) plus a non-zero captured-tensor count. This
/// is a mechanism/correctness check (small token budget), not a full-quality
/// Hessian. Capture fires only at the bf16/f16 chokepoints, so non-bf16 models
/// are skipped. Not in any default tier — opt in with `--battery calibrate`.
fn run_examples_calibrate_rows(config: &EvalConfig, ctx: &EvalContext) -> Vec<EvalResult> {
    evaluation_models(config)
        .into_iter()
        .map(|model| run_examples_calibrate_model(config, ctx, model))
        .collect()
}

fn run_examples_calibrate_model(
    config: &EvalConfig,
    ctx: &EvalContext,
    model: String,
) -> EvalResult {
    let label = "single_load_hessian_consistency";
    let prompt_ref = prompt("benchmarks/prompts/lru_cache_single_blank.txt");
    let base_metrics = BTreeMap::from([
        ("implemented".to_string(), json!(true)),
        ("executor".to_string(), json!("examples")),
        ("suite".to_string(), json!("calibration")),
    ]);

    let skip = |reason: &str| -> EvalResult {
        row_for_model(
            BatteryId::Calibrate,
            None,
            label,
            None,
            EvalStatus::Skip,
            Some(reason.to_string()),
            base_metrics.clone(),
            config,
            ctx,
            prompt_ref.clone(),
            0,
            model.clone(),
        )
    };

    // Capture fires at the bf16/f16 gemm chokepoints, so a faithful calibration
    // pass needs a bf16 source model.
    if !model_artifact_stem(&model).contains("bf16") {
        return skip(
            "calibration requires a bf16 source model (capture fires at the bf16 chokepoint)",
        );
    }
    if !Path::new(&model).exists() {
        return skip("collect_artifacts requires --model to be a local filesystem path");
    }
    let Some(bin) = resolve_collect_artifacts_bin() else {
        return skip("collect_artifacts example binary not found; build with `cargo build --release -p hipfire-runtime --example collect_artifacts`");
    };

    let corpus = repo_root()
        .map(|r| r.join("benchmarks/prompts/lru_cache_single_blank.txt"))
        .unwrap_or_else(|| PathBuf::from("benchmarks/prompts/lru_cache_single_blank.txt"));
    let out = std::env::temp_dir().join(format!(
        "hipfire-eval-calib-{}.calib.hfq",
        std::process::id()
    ));
    let max_tokens = config.max_tokens.clamp(16, 64).to_string();
    let args = vec![
        "--model".to_string(),
        model.clone(),
        "--corpus".to_string(),
        corpus.display().to_string(),
        "--output".to_string(),
        out.display().to_string(),
        "--max-tokens".to_string(),
        max_tokens,
    ];
    let command_display = format!("{} {}", bin.display(), args.join(" "));
    let started = SystemTime::now();
    let output = match Command::new(&bin).args(&args).output() {
        Ok(o) => o,
        Err(err) => {
            let mut m = base_metrics.clone();
            m.insert("command".to_string(), json!(command_display));
            return row_for_model(
                BatteryId::Calibrate,
                None,
                label,
                None,
                EvalStatus::Fail,
                Some(format!("spawn collect_artifacts: {err}")),
                m,
                config,
                ctx,
                prompt_ref,
                elapsed_since_ms(started),
                model,
            );
        }
    };
    let elapsed_ms = elapsed_since_ms(started);
    let _ = std::fs::remove_file(&out);
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    // collect_artifacts prints to stderr:
    //   "collected N hessian tensors in ...s; max diag(H)-vs-Σx² rel-err = ... [CONSISTENT]"
    let combined = format!("{stdout}{stderr}");
    let consistent = combined.contains("[CONSISTENT]");
    let n_hessian = parse_collected_hessian_count(&combined);

    let mut metrics = base_metrics.clone();
    metrics.insert("command".to_string(), json!(command_display));
    metrics.insert("consistent".to_string(), json!(consistent));
    if let Some(n) = n_hessian {
        metrics.insert("n_hessian".to_string(), json!(n));
    }
    metrics.insert(
        "stderr_hash".to_string(),
        json!(stable_hash_bytes(stderr.as_bytes())),
    );

    let ok = output.status.success() && consistent && n_hessian.map_or(false, |n| n > 0);
    let status = if ok {
        EvalStatus::Pass
    } else {
        EvalStatus::Fail
    };
    let reason = if ok {
        None
    } else if !output.status.success() {
        Some(format!("collect_artifacts exited with {}", output.status))
    } else if !consistent {
        Some("Hessian/imatrix consistency check did not report [CONSISTENT]".to_string())
    } else {
        Some("collect_artifacts captured 0 hessian tensors".to_string())
    };

    row_for_model(
        BatteryId::Calibrate,
        None,
        label,
        None,
        status,
        reason,
        metrics,
        config,
        ctx,
        prompt_ref,
        elapsed_ms,
        model,
    )
}

/// Parse the captured-Hessian count from collect_artifacts' summary line
/// ("collected N hessian tensors in ...").
fn parse_collected_hessian_count(s: &str) -> Option<u64> {
    let idx = s.find("collected ")? + "collected ".len();
    let rest = &s[idx..];
    let end = rest.find(' ')?;
    rest[..end].parse::<u64>().ok()
}

/// Perplexity / KLD battery: run the `perplexity` example over a raw corpus and
/// report PPL + NLL/tok (+ KLD/tok when `--kldref` is given). The canonical
/// quant-quality primitive — the place the older bench_quant_quality / megabench
/// / sim_mq3_eval scripts should funnel into instead of invoking `perplexity`
/// raw. Opt-in (`--battery perplexity`); not in any default tier (needs a
/// corpus). Corpus/ctx via HIPFIRE_EVAL_PERPLEXITY_CORPUS / _CTX.
fn run_examples_perplexity_rows(config: &EvalConfig, ctx: &EvalContext) -> Vec<EvalResult> {
    evaluation_models(config)
        .into_iter()
        .map(|model| run_examples_perplexity_model(config, ctx, model))
        .collect()
}

/// Resolve a KLD reference for `model`, in priority order: explicit `--kldref`,
/// then `HIPFIRE_EVAL_KLDREF`, then a sibling `<stem>.kldref.hfq` / `.pkld`,
/// then `~/.hipfire/datasets/kldref/<stem>.{kldref.hfq,pkld}`. `None` ⇒ PPL-only
/// (not a failure). Generate one with `collect_artifacts --kldref` (HFQM) or
/// `perplexity --dump-ref` (.pkld).
fn resolve_kldref(model: &str, config: &EvalConfig) -> Option<PathBuf> {
    if let Some(p) = &config.kldref {
        if p.exists() {
            return Some(p.clone());
        }
    }
    if let Some(p) = std::env::var_os("HIPFIRE_EVAL_KLDREF").map(PathBuf::from) {
        if p.exists() {
            return Some(p);
        }
    }
    let stem = model_artifact_stem(model);
    let sibling_dir = Path::new(model)
        .parent()
        .map(Path::to_path_buf)
        .unwrap_or_else(|| PathBuf::from("."));
    let mut candidates = vec![
        sibling_dir.join(format!("{stem}.kldref.hfq")),
        sibling_dir.join(format!("{stem}.pkld")),
    ];
    if let Some(home) = home_dir() {
        let d = home.join(".hipfire").join("datasets").join("kldref");
        candidates.push(d.join(format!("{stem}.kldref.hfq")));
        candidates.push(d.join(format!("{stem}.pkld")));
    }
    candidates.into_iter().find(|p| p.exists())
}

/// Parse a `"<label>:<ws><number>..."` line from perplexity stdout (e.g.
/// `PPL:      14.6700`, `KLD/tok:  0.012345 (top-128, ...)`).
fn parse_labeled_f64(s: &str, label: &str) -> Option<f64> {
    let line = s.lines().find(|l| l.trim_start().starts_with(label))?;
    line[line.find(label)? + label.len()..]
        .split_whitespace()
        .next()?
        .parse::<f64>()
        .ok()
}

fn run_examples_perplexity_model(
    config: &EvalConfig,
    ctx: &EvalContext,
    model: String,
) -> EvalResult {
    let label = "corpus_perplexity";
    let corpus_rel = std::env::var("HIPFIRE_EVAL_PERPLEXITY_CORPUS").unwrap_or_else(|_| {
        "benchmarks/quality-baselines/slice/wikitext2-1024s-2048ctx.txt".into()
    });
    let prompt_ref = prompt(&corpus_rel);
    // KLD needs a reference; absent ⇒ PPL-only (not a failure). See resolve_kldref.
    let kldref = resolve_kldref(&model, config);
    let kld_on = kldref.is_some();
    let base_metrics = BTreeMap::from([
        ("implemented".to_string(), json!(true)),
        ("executor".to_string(), json!("examples")),
        ("suite".to_string(), json!("perplexity")),
        ("kld_ref".to_string(), json!(kld_on)),
    ]);

    let skip = |reason: &str| -> EvalResult {
        row_for_model(
            BatteryId::Perplexity,
            None,
            label,
            None,
            EvalStatus::Skip,
            Some(reason.to_string()),
            base_metrics.clone(),
            config,
            ctx,
            prompt_ref.clone(),
            0,
            model.clone(),
        )
    };

    if !Path::new(&model).exists() {
        return skip("perplexity requires --model to be a local filesystem path");
    }
    let Some(bin) = resolve_perplexity_bin() else {
        return skip("perplexity example binary not found; build with `cargo build --release -p hipfire-runtime --example perplexity`");
    };
    let corpus = repo_root()
        .map(|r| r.join(&corpus_rel))
        .unwrap_or_else(|| PathBuf::from(&corpus_rel));
    if !corpus.exists() {
        return skip(&format!(
            "corpus not found: {} — set HIPFIRE_EVAL_PERPLEXITY_CORPUS or place a corpus under ~/.hipfire/datasets/",
            corpus.display()
        ));
    }

    let ctx_len = std::env::var("HIPFIRE_EVAL_PERPLEXITY_CTX")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(512);
    let kv_mode = config.kv_mode.as_deref().unwrap_or("q8").to_string();
    let mut args = vec![
        model.clone(),
        corpus.display().to_string(),
        "--ctx".to_string(),
        ctx_len.to_string(),
        "--warmup".to_string(),
        "8".to_string(),
        "--offset".to_string(),
        "0".to_string(),
        "--kv-mode".to_string(),
        kv_mode.clone(),
    ];
    if let Some(kref) = &kldref {
        args.push("--kld-ref".to_string());
        args.push(kref.display().to_string());
    }
    let command_display = format!("{} {}", bin.display(), args.join(" "));
    let started = SystemTime::now();
    let output = match Command::new(&bin).args(&args).output() {
        Ok(o) => o,
        Err(err) => {
            let mut m = base_metrics.clone();
            m.insert("command".to_string(), json!(command_display));
            return row_for_model(
                BatteryId::Perplexity,
                None,
                label,
                None,
                EvalStatus::Fail,
                Some(format!("spawn perplexity: {err}")),
                m,
                config,
                ctx,
                prompt_ref,
                elapsed_since_ms(started),
                model,
            );
        }
    };
    let elapsed_ms = elapsed_since_ms(started);
    let stdout = String::from_utf8_lossy(&output.stdout);
    let ppl = parse_labeled_f64(&stdout, "PPL:");
    let nll = parse_labeled_f64(&stdout, "NLL/tok:");
    let kld = parse_labeled_f64(&stdout, "KLD/tok:");

    let mut metrics = base_metrics.clone();
    metrics.insert("command".to_string(), json!(command_display));
    metrics.insert("ctx".to_string(), json!(ctx_len));
    metrics.insert("kv_mode".to_string(), json!(kv_mode));
    if let Some(p) = ppl {
        metrics.insert("ppl".to_string(), json!(p));
    }
    if let Some(n) = nll {
        metrics.insert("nll_per_tok".to_string(), json!(n));
    }
    if let Some(k) = kld {
        metrics.insert("kld_per_tok".to_string(), json!(k));
    }
    metrics.insert(
        "stdout_hash".to_string(),
        json!(stable_hash_bytes(stdout.as_bytes())),
    );

    // Pass when a finite, positive PPL was produced (and, if a KLD ref was
    // supplied, a KLD value too). Baseline-regression gating is the sweep
    // caller's job; this battery is the measurement primitive.
    let ppl_ok = ppl.is_some_and(|p| p.is_finite() && p > 0.0);
    let kld_ok = !kld_on || kld.is_some_and(|k| k.is_finite());
    let ok = output.status.success() && ppl_ok && kld_ok;
    let status = if ok {
        EvalStatus::Pass
    } else {
        EvalStatus::Fail
    };
    let reason = if ok {
        None
    } else if !output.status.success() {
        Some(format!("perplexity exited with {}", output.status))
    } else if !ppl_ok {
        Some("perplexity produced no finite positive PPL".to_string())
    } else {
        Some("KLD ref supplied but no KLD/tok produced".to_string())
    };

    row_for_model(
        BatteryId::Perplexity,
        None,
        label,
        None,
        status,
        reason,
        metrics,
        config,
        ctx,
        prompt_ref,
        elapsed_ms,
        model,
    )
}

#[derive(Clone, Copy)]
struct Qwen35SpeedCase {
    label: &'static str,
    prefill: usize,
}

#[derive(Clone, Copy)]
struct DaemonSpeedCase {
    label: &'static str,
}

fn qwen35_speed_cases() -> &'static [Qwen35SpeedCase] {
    &[
        Qwen35SpeedCase {
            label: "pp32_prefill_decode",
            prefill: 32,
        },
        Qwen35SpeedCase {
            label: "pp128_prefill_decode",
            prefill: 128,
        },
    ]
}

fn daemon_speed_cases() -> &'static [DaemonSpeedCase] {
    &[
        DaemonSpeedCase {
            label: "daemon_prefill_decode_first",
        },
        DaemonSpeedCase {
            label: "daemon_prefill_decode_reset",
        },
    ]
}

fn run_examples_qwen35_speed_rows(config: &EvalConfig, ctx: &EvalContext) -> Vec<EvalResult> {
    evaluation_models(config)
        .into_iter()
        .flat_map(|model| run_examples_qwen35_speed_model(config, ctx, model))
        .collect()
}

fn run_examples_qwen35_speed_model(
    config: &EvalConfig,
    ctx: &EvalContext,
    model: String,
) -> Vec<EvalResult> {
    let cases = qwen35_speed_cases();
    let prompt_ref = prompt("benchmarks/prompts/lru_cache_single_blank.txt");
    let kv_mode = config.kv_mode.as_deref().unwrap_or("asym3").to_string();
    let mut rows = Vec::new();
    let base_metrics = BTreeMap::from([
        ("implemented".to_string(), json!(true)),
        ("executor".to_string(), json!("examples")),
        ("suite".to_string(), json!("qwen35_speed_gate")),
        ("kv_mode".to_string(), json!(kv_mode.as_str())),
    ]);

    if !Path::new(&model).exists() {
        for case in cases {
            rows.push(row_for_model(
                BatteryId::Speed,
                None,
                case.label,
                None,
                EvalStatus::Skip,
                Some(
                    "bench_qwen35_speed requires --model to be a local filesystem path".to_string(),
                ),
                {
                    let mut m = base_metrics.clone();
                    m.insert("prefill_tokens".to_string(), json!(case.prefill));
                    m
                },
                config,
                ctx,
                prompt_ref.clone(),
                0,
                model.clone(),
            ));
        }
        return rows;
    }

    let Some(bin) = resolve_bench_qwen35_speed_bin() else {
        for case in cases {
            rows.push(row_for_model(
                BatteryId::Speed,
                None,
                case.label,
                None,
                EvalStatus::Skip,
                Some("bench_qwen35_speed example binary not found; build with `cargo build --release --features deltanet -p hipfire-runtime --example bench_qwen35_speed`".to_string()),
                {
                    let mut m = base_metrics.clone();
                    m.insert("prefill_tokens".to_string(), json!(case.prefill));
                    m
                },
                config,
                ctx,
                prompt_ref.clone(),
                0,
                model.clone(),
            ));
        }
        return rows;
    };

    let prefill_list: Vec<String> = cases.iter().map(|case| case.prefill.to_string()).collect();
    let args = vec![
        model.clone(),
        "--prefill-list".to_string(),
        prefill_list.join(","),
        "--prefill-runs".to_string(),
        "2".to_string(),
        "--warmup".to_string(),
        "5".to_string(),
        "--gen".to_string(),
        config.max_tokens.max(50).to_string(),
    ];
    let command_display = format!("{} {}", bin.display(), args.join(" "));
    let started = SystemTime::now();
    let mut command = Command::new(&bin);
    command.args(&args);
    command.env("HIPFIRE_KV_MODE", kv_mode);
    command.env("HIPFIRE_DPM_WARMUP_SECS", "3");
    if !model_artifact_stem(&model).contains("0.8b") {
        command.env("HIPFIRE_GRAPH", "1");
    }
    let output = match command.output() {
        Ok(output) => output,
        Err(err) => {
            for case in cases {
                rows.push(row_for_model(
                    BatteryId::Speed,
                    None,
                    case.label,
                    None,
                    EvalStatus::Fail,
                    Some(format!("spawn bench_qwen35_speed: {err}")),
                    {
                        let mut m = base_metrics.clone();
                        m.insert("prefill_tokens".to_string(), json!(case.prefill));
                        m.insert("command".to_string(), json!(command_display.clone()));
                        m.insert("graph_enabled".to_string(), json!(false));
                        m
                    },
                    config,
                    ctx,
                    prompt_ref.clone(),
                    elapsed_since_ms(started),
                    model.clone(),
                ));
            }
            return rows;
        }
    };
    let elapsed_ms = elapsed_since_ms(started);
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    let case_metrics = parse_case_summary_metrics(&stderr);
    let shared_metrics = parse_summary_kv_metrics(&stderr);
    let stdout_hash = stable_hash_bytes(stdout.as_bytes());
    let stderr_hash = stable_hash_bytes(stderr.as_bytes());

    for case in cases {
        let mut metrics = base_metrics.clone();
        metrics.insert("prefill_tokens".to_string(), json!(case.prefill));
        metrics.insert("command".to_string(), json!(command_display.clone()));
        metrics.insert(
            "graph_enabled".to_string(),
            json!(!model_artifact_stem(&model).contains("0.8b")),
        );
        metrics.insert("stdout_hash".to_string(), json!(stdout_hash));
        metrics.insert("stderr_hash".to_string(), json!(stderr_hash));
        if let Some(case_row_metrics) = case_metrics.get(case.label) {
            metrics.extend(case_row_metrics.clone());
        } else {
            metrics.extend(shared_metrics.clone());
        }
        if let Some(v) = metrics.get("gen_tok_s").cloned() {
            metrics.entry("tok_s".to_string()).or_insert(v);
        }
        let baseline_check = apply_speed_baseline(&mut metrics, &model, case.prefill, ctx);
        let baseline_failed = baseline_check
            .as_ref()
            .map(|check| check.failed)
            .unwrap_or(false);
        let baseline_error = baseline_check
            .and_then(|check| check.error)
            .or_else(|| None);
        let status = if output.status.success()
            && metrics.contains_key("prefill_tok_s")
            && !baseline_failed
            && baseline_error.is_none()
        {
            EvalStatus::Pass
        } else {
            EvalStatus::Fail
        };
        let reason = if let Some(error) = &baseline_error {
            Some(error.to_string())
        } else if baseline_failed {
            Some("bench_qwen35_speed fell below perf baseline floor".to_string())
        } else if !case_metrics.contains_key(case.label) || !metrics.contains_key("prefill_tok_s") {
            Some(format!(
                "bench_qwen35_speed did not emit case metrics for {}",
                case.label
            ))
        } else if !output.status.success() {
            Some(format!("bench_qwen35_speed exited with {}", output.status))
        } else {
            None
        };
        rows.push(row_for_model(
            BatteryId::Speed,
            None,
            case.label,
            None,
            status,
            reason,
            metrics,
            config,
            ctx,
            prompt_ref.clone(),
            elapsed_ms,
            model.clone(),
        ));
    }
    rows
}

fn parse_case_summary_metrics(output: &str) -> HashMap<&'static str, BTreeMap<String, Value>> {
    let mut per_case: HashMap<&'static str, BTreeMap<String, Value>> = HashMap::new();
    for raw_line in output.lines() {
        let line = raw_line.trim();
        if !line.starts_with("CASE_SUMMARY") {
            continue;
        }
        let mut label = None;
        let mut metrics = BTreeMap::new();
        for part in line.split_whitespace().skip(1) {
            let Some((key, value)) = part.split_once('=') else {
                continue;
            };
            if key == "label" {
                label = Some(match value {
                    "pp32_prefill_decode" => "pp32_prefill_decode",
                    "pp128_prefill_decode" => "pp128_prefill_decode",
                    _ => "",
                });
                continue;
            }
            let value = value.trim_end_matches(',');
            if let Ok(v) = value.parse::<u64>() {
                metrics.insert(key.to_string(), json!(v));
            } else if let Ok(v) = value.parse::<f64>() {
                metrics.insert(key.to_string(), json!(v));
            } else {
                metrics.insert(key.to_string(), json!(value));
            }
        }
        if let Some("pp32_prefill_decode" | "pp128_prefill_decode") = label {
            let key = label.unwrap();
            per_case.insert(key, metrics);
        }
    }
    per_case
}

struct SpeedBaselineCheck {
    failed: bool,
    error: Option<String>,
}

fn apply_speed_baseline(
    metrics: &mut BTreeMap<String, Value>,
    model: &str,
    prefill: usize,
    ctx: &EvalContext,
) -> Option<SpeedBaselineCheck> {
    let require = env_truthy("HIPFIRE_REQUIRE_PERF_BASELINE");
    let baseline_path = match resolve_perf_baseline_path(ctx) {
        Ok(Some(path)) => path,
        Ok(None) => {
            metrics.insert("perf_baseline_status".to_string(), json!("missing"));
            if require {
                return Some(SpeedBaselineCheck {
                    failed: false,
                    error: Some("no matching perf baseline found".to_string()),
                });
            }
            return None;
        }
        Err(err) => {
            metrics.insert("perf_baseline_status".to_string(), json!("error"));
            return Some(SpeedBaselineCheck {
                failed: false,
                error: Some(err),
            });
        }
    };
    let model_size = match speed_model_size(model) {
        Some(size) => size,
        None => {
            metrics.insert("perf_baseline_status".to_string(), json!("unmatched_model"));
            if require {
                return Some(SpeedBaselineCheck {
                    failed: false,
                    error: Some(format!("cannot infer model size from {model}")),
                });
            }
            return None;
        }
    };
    let model_id = speed_model_id(model);
    let baseline = match load_speed_baseline(&baseline_path, &model_id, &model_size, prefill) {
        Ok(Some(baseline)) => baseline,
        Ok(None) => {
            metrics.insert("perf_baseline_status".to_string(), json!("missing_row"));
            metrics.insert(
                "perf_baseline_path".to_string(),
                json!(baseline_path.display().to_string()),
            );
            if require {
                return Some(SpeedBaselineCheck {
                    failed: false,
                    error: Some(format!(
                        "perf baseline {} has no speed row for {model_id} pp{prefill}",
                        baseline_path.display()
                    )),
                });
            }
            return None;
        }
        Err(err) => {
            metrics.insert("perf_baseline_status".to_string(), json!("error"));
            return Some(SpeedBaselineCheck {
                failed: false,
                error: Some(err),
            });
        }
    };
    let tolerance = baseline.tolerance_pct / 100.0;
    let prefill_observed = metrics.get("prefill_tok_s").and_then(Value::as_f64);
    let gen_observed = metrics.get("gen_tok_s").and_then(Value::as_f64);
    let mut failed = false;
    metrics.insert("perf_baseline_status".to_string(), json!("compared"));
    metrics.insert(
        "perf_baseline_path".to_string(),
        json!(baseline_path.display().to_string()),
    );
    metrics.insert("baseline_label".to_string(), json!(baseline.label));
    metrics.insert("baseline_model_id".to_string(), json!(baseline.model_id));
    metrics.insert("baseline_format".to_string(), json!(baseline.format));
    metrics.insert(
        "baseline_tolerance_pct".to_string(),
        json!(baseline.tolerance_pct),
    );
    if let Some(floor) = baseline.prefill_tok_s {
        metrics.insert("baseline_prefill_tok_s".to_string(), json!(floor));
        metrics.insert(
            "baseline_prefill_floor_tok_s".to_string(),
            json!(floor * (1.0 - tolerance)),
        );
        if prefill_observed.is_some_and(|observed| observed < floor * (1.0 - tolerance)) {
            failed = true;
        }
    }
    if let Some(floor) = baseline.gen_tok_s {
        metrics.insert("baseline_gen_tok_s".to_string(), json!(floor));
        metrics.insert(
            "baseline_gen_floor_tok_s".to_string(),
            json!(floor * (1.0 - tolerance)),
        );
        if gen_observed.is_some_and(|observed| observed < floor * (1.0 - tolerance)) {
            failed = true;
        }
    }
    metrics.insert("perf_baseline_failed".to_string(), json!(failed));
    Some(SpeedBaselineCheck {
        failed,
        error: None,
    })
}

#[derive(Debug, Clone)]
struct SpeedBaseline {
    label: String,
    model_id: String,
    format: String,
    prefill_tok_s: Option<f64>,
    gen_tok_s: Option<f64>,
    tolerance_pct: f64,
}

fn load_speed_baseline(
    path: &Path,
    model_id: &str,
    model_size: &str,
    prefill: usize,
) -> Result<Option<SpeedBaseline>, String> {
    let body = fs::read_to_string(path)
        .map_err(|err| format!("read perf baseline {}: {err}", path.display()))?;
    let value: Value = serde_json::from_str(&body)
        .map_err(|err| format!("parse perf baseline {}: {err}", path.display()))?;
    let tolerance_pct = value
        .get("tolerance_pct")
        .and_then(Value::as_f64)
        .unwrap_or(5.0);
    let Some(rows) = value
        .get("baselines")
        .and_then(|v| v.get("speed"))
        .and_then(Value::as_array)
    else {
        return Ok(None);
    };
    for row in rows {
        let row_model_id = row.get("model_id").and_then(Value::as_str);
        let row_size = row.get("model_size").and_then(Value::as_str);
        let row_prefill = row.get("prefill_tokens").and_then(Value::as_u64);
        if row_model_id == Some(model_id)
            && row_size == Some(model_size)
            && row_prefill == Some(prefill as u64)
        {
            return Ok(Some(SpeedBaseline {
                label: row
                    .get("label")
                    .and_then(Value::as_str)
                    .unwrap_or("speed")
                    .to_string(),
                model_id: model_id.to_string(),
                format: row
                    .get("format")
                    .and_then(Value::as_str)
                    .unwrap_or("unknown")
                    .to_string(),
                prefill_tok_s: row.get("prefill_tok_s").and_then(Value::as_f64),
                gen_tok_s: row.get("gen_tok_s").and_then(Value::as_f64),
                tolerance_pct: row
                    .get("tolerance_pct")
                    .and_then(Value::as_f64)
                    .unwrap_or(tolerance_pct),
            }));
        }
    }
    Ok(None)
}

fn resolve_perf_baseline_path(ctx: &EvalContext) -> Result<Option<PathBuf>, String> {
    if let Ok(path) = std::env::var("HIPFIRE_PERF_BASELINE") {
        let path = PathBuf::from(path);
        if path.exists() {
            return Ok(Some(path));
        }
        return Err(format!(
            "HIPFIRE_PERF_BASELINE points to missing file: {}",
            path.display()
        ));
    }
    let Some(arch) = ctx.arch.as_deref().or(ctx.host_profile.gfx.as_deref()) else {
        return Ok(None);
    };
    let root = std::env::var("HIPFIRE_PERF_BASELINE_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("benchmarks/perf-baselines"));
    let root = if root.is_absolute() {
        root
    } else {
        repo_root().unwrap_or_else(|| PathBuf::from(".")).join(root)
    };
    let pattern_prefix = format!("{arch}-");
    let entries = match fs::read_dir(&root) {
        Ok(entries) => entries,
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(err) => return Err(format!("read perf baseline dir {}: {err}", root.display())),
    };
    let mut matches = entries
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .filter(|path| {
            path.file_name()
                .and_then(OsStr::to_str)
                .is_some_and(|name| name.starts_with(&pattern_prefix) && name.ends_with(".json"))
        })
        .collect::<Vec<_>>();
    matches.sort();
    match matches.len() {
        0 => Ok(None),
        1 => Ok(matches.pop()),
        _ => Err(format!(
            "multiple perf baselines for {arch}; set HIPFIRE_PERF_BASELINE"
        )),
    }
}

fn speed_model_id(model: &str) -> String {
    model_artifact_stem(model).to_ascii_lowercase()
}

fn speed_model_size(model: &str) -> Option<String> {
    let stem = model_artifact_stem(model).to_ascii_lowercase();
    for size in ["0.8b", "4b", "9b", "27b", "35b-a3b"] {
        if stem.contains(size) {
            return Some(size.to_string());
        }
    }
    None
}

fn env_truthy(name: &str) -> bool {
    matches!(
        std::env::var(name).ok().as_deref(),
        Some("1" | "true" | "TRUE" | "yes" | "YES" | "on" | "ON")
    )
}

#[derive(Clone, Copy)]
struct AgenticCase {
    label: &'static str,
    system_path: &'static str,
    thinking_clamped: bool,
    max_tokens: usize,
}

fn agentic_cases() -> &'static [AgenticCase] {
    &[
        AgenticCase {
            label: "agentic_pi_unclamped",
            system_path: "benchmarks/prompts/agentic_pi_system.txt",
            thinking_clamped: false,
            max_tokens: 256,
        },
        AgenticCase {
            label: "agentic_pi_clamped",
            system_path: "benchmarks/prompts/agentic_pi_system.txt",
            thinking_clamped: true,
            max_tokens: 256,
        },
        AgenticCase {
            label: "agentic_hermes_unclamped",
            system_path: "benchmarks/prompts/agentic_hermes_system.txt",
            thinking_clamped: false,
            max_tokens: 256,
        },
    ]
}

fn run_examples_agentic_rows(config: &EvalConfig, ctx: &EvalContext) -> Vec<EvalResult> {
    let mut rows = evaluation_models(config)
        .into_iter()
        .flat_map(|model| {
            agentic_cases()
                .iter()
                .map(move |case| run_examples_agentic_case(config, ctx, model.clone(), *case))
        })
        .collect::<Vec<_>>();
    rows.push(run_examples_agentic_jinja_tools_case(
        config,
        ctx,
        config.model.clone(),
    ));
    rows
}

fn run_examples_agentic_case(
    config: &EvalConfig,
    ctx: &EvalContext,
    model: String,
    case: AgenticCase,
) -> EvalResult {
    let prompt_path = "benchmarks/prompts/agentic_user_read.txt";
    let prompt_ref = combined_prompt_ref(case.system_path, prompt_path);
    let mut metrics = BTreeMap::from([
        ("suite".to_string(), json!("agentic_tool_call")),
        ("system_prompt".to_string(), json!(case.system_path)),
        ("user_prompt".to_string(), json!(prompt_path)),
        ("thinking_clamped".to_string(), json!(case.thinking_clamped)),
        (
            "detector_profile".to_string(),
            json!("agentic_toolcall_shape"),
        ),
    ]);
    if case.thinking_clamped {
        metrics.insert("max_think_tokens".to_string(), json!(1));
    }
    let mut profile = hipfire_coherence::DetectorProfile::default_for_prompt("", None);
    profile.agentic = true;
    run_daemon_coherence_anchor(
        BatteryId::Agentic,
        case.label,
        prompt_path,
        prompt_ref,
        config,
        ctx,
        model,
        Some(4096),
        Some(case.max_tokens),
        metrics,
        Some(case.system_path),
        None,
        None,
        false,
        Some(profile),
    )
}

fn run_examples_agentic_jinja_tools_case(
    config: &EvalConfig,
    ctx: &EvalContext,
    model: String,
) -> EvalResult {
    let prompt_path = "benchmarks/prompts/agentic_jinja_tools_user.txt";
    let system_path = "benchmarks/prompts/agentic_jinja_tools_system.txt";
    let tools = json!([
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get the current weather for a city.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "city": {
                            "type": "string",
                            "description": "City name."
                        },
                        "unit": {
                            "type": "string",
                            "enum": ["c", "f"],
                            "description": "Temperature unit."
                        }
                    },
                    "required": ["city"]
                }
            }
        }
    ]);
    let prompt_ref = structured_tools_prompt_ref(system_path, prompt_path, &tools);
    let metrics = BTreeMap::from([
        ("suite".to_string(), json!("agentic_jinja_tools")),
        ("system_prompt".to_string(), json!(system_path)),
        ("user_prompt".to_string(), json!(prompt_path)),
        ("tools_count".to_string(), json!(1.0)),
        ("force_jinja_chat".to_string(), json!(true)),
        ("assistant_prefix".to_string(), json!("closed_think")),
        (
            "detector_profile".to_string(),
            json!("agentic_structured_tools"),
        ),
    ]);
    let mut profile = hipfire_coherence::DetectorProfile::default_for_prompt("", None);
    profile.agentic = true;
    run_daemon_coherence_anchor(
        BatteryId::Agentic,
        "agentic_jinja_structured_tools",
        prompt_path,
        prompt_ref,
        config,
        ctx,
        model,
        Some(1024),
        Some(192),
        metrics,
        Some(system_path),
        Some(tools),
        Some("closed_think"),
        true,
        Some(profile),
    )
}

#[derive(Clone, Copy)]
struct RuntimeCase {
    label: &'static str,
    script: &'static str,
    category: &'static str,
}

fn runtime_cases() -> &'static [RuntimeCase] {
    &[
        RuntimeCase {
            label: "server_prefill_batch",
            script: "tests/smoke-server-prefill-batch.sh",
            category: "prefill_batching",
        },
        RuntimeCase {
            label: "server_decode_batch",
            script: "tests/smoke-server-decode-batch.sh",
            category: "decode_batching",
        },
        RuntimeCase {
            label: "daemon_generate_batch_prefill",
            script: "tests/smoke-generate-batch-prefill.sh",
            category: "prefill_batching",
        },
        RuntimeCase {
            label: "prefix_checkpoint_reuse",
            script: "tests/smoke-server-prefix-checkpoint-reuse.sh",
            category: "prefix_reuse",
        },
        RuntimeCase {
            label: "prefix_boundary_reuse",
            script: "tests/smoke-server-prefix-boundary-reuse.sh",
            category: "prefix_reuse",
        },
        RuntimeCase {
            label: "responses_prefix_reuse",
            script: "tests/smoke-server-responses-prefix-reuse.sh",
            category: "prefix_reuse",
        },
        RuntimeCase {
            label: "shared_prefix_fanout",
            script: "tests/smoke-server-shared-prefix-fanout.sh",
            category: "shared_prefix_fanout",
        },
        RuntimeCase {
            label: "multi_model_workers",
            script: "tests/smoke-server-multi-model-workers.sh",
            category: "multi_model_workers",
        },
        RuntimeCase {
            label: "server_concurrency",
            script: "tests/stress-server-concurrency.sh",
            category: "concurrency",
        },
        RuntimeCase {
            label: "kv_budget_reload",
            script: "tests/e2e_kv_budget.sh",
            category: "kv_admission",
        },
        RuntimeCase {
            label: "kv_reject_http",
            script: "tests/e2e_kv_reject.sh",
            category: "kv_admission",
        },
        RuntimeCase {
            label: "kv_reject_run",
            script: "tests/e2e_run_reject.sh",
            category: "kv_admission",
        },
        RuntimeCase {
            label: "pipeline_parallel_gate",
            script: "tests/pp-gate.sh",
            category: "pipeline_parallel",
        },
        RuntimeCase {
            label: "pipeline_parallel_coherence",
            script: "tests/coherence-gate-pp.sh",
            category: "pipeline_parallel",
        },
    ]
}

fn run_examples_runtime_rows(config: &EvalConfig, ctx: &EvalContext) -> Vec<EvalResult> {
    runtime_cases()
        .iter()
        .map(|case| run_examples_runtime_case(config, ctx, *case))
        .collect()
}

fn run_examples_runtime_case(
    config: &EvalConfig,
    ctx: &EvalContext,
    case: RuntimeCase,
) -> EvalResult {
    let script = Path::new(case.script);
    let mut metrics = BTreeMap::from([
        ("runtime_evidence_case".to_string(), json!(case.label)),
        ("runtime_category".to_string(), json!(case.category)),
        ("script".to_string(), json!(case.script)),
        ("runtime_path".to_string(), json!("shell_runtime_gate")),
    ]);

    if !script.is_file() {
        return skip_row_with_metrics(
            BatteryId::Runtime,
            None,
            case.label,
            None,
            "runtime evidence script is not present",
            config,
            ctx,
            None,
            metrics,
        );
    }

    if is_local_filesystem_model(&config.model) && !Path::new(&config.model).is_file() {
        metrics.insert("model_missing".to_string(), json!(true));
        return skip_row_with_metrics(
            BatteryId::Runtime,
            None,
            case.label,
            None,
            "model path is missing; runtime evidence script not run",
            config,
            ctx,
            None,
            metrics,
        );
    }

    let started = SystemTime::now();
    let output = match Command::new("bash")
        .arg(script)
        .env("MODEL", &config.model)
        .env("HIPFIRE_MODEL", &config.model)
        .env("HIPFIRE_EVAL_RUNTIME_ROW", case.label)
        .output()
    {
        Ok(output) => output,
        Err(err) => {
            return row(
                BatteryId::Runtime,
                None,
                case.label,
                None,
                EvalStatus::Fail,
                Some(format!("spawn runtime evidence script: {err}")),
                metrics,
                config,
                ctx,
                None,
                elapsed_since_ms(started),
            );
        }
    };
    let elapsed_ms = elapsed_since_ms(started);
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    let combined = format!("{stdout}\n{stderr}");
    metrics.insert(
        "exit_code".to_string(),
        json!(output.status.code().unwrap_or(-1)),
    );
    metrics.insert(
        "stdout_hash".to_string(),
        json!(stable_hash_bytes(stdout.as_bytes())),
    );
    metrics.insert(
        "stderr_hash".to_string(),
        json!(stable_hash_bytes(stderr.as_bytes())),
    );
    metrics.insert(
        "stdout_excerpt".to_string(),
        json!(truncate_for_metric(stdout.trim(), 512)),
    );
    metrics.insert(
        "stderr_excerpt".to_string(),
        json!(truncate_for_metric(stderr.trim(), 512)),
    );

    if output.status.success() {
        if combined.to_ascii_lowercase().contains("skipping")
            || combined.to_ascii_lowercase().contains("skipped")
        {
            skip_row_with_metrics(
                BatteryId::Runtime,
                None,
                case.label,
                None,
                "runtime evidence script skipped on this host",
                config,
                ctx,
                None,
                metrics,
            )
        } else {
            row(
                BatteryId::Runtime,
                None,
                case.label,
                None,
                EvalStatus::Pass,
                None,
                metrics,
                config,
                ctx,
                None,
                elapsed_ms,
            )
        }
    } else if output.status.code() == Some(2)
        && runtime_script_output_is_environment_skip(&combined)
    {
        skip_row_with_metrics(
            BatteryId::Runtime,
            None,
            case.label,
            None,
            "runtime evidence script prerequisites are unavailable on this host",
            config,
            ctx,
            None,
            metrics,
        )
    } else {
        row(
            BatteryId::Runtime,
            None,
            case.label,
            None,
            EvalStatus::Fail,
            Some("runtime evidence script failed".to_string()),
            metrics,
            config,
            ctx,
            None,
            elapsed_ms,
        )
    }
}

fn is_local_filesystem_model(model: &str) -> bool {
    model.starts_with('/') || model.starts_with("./") || model.starts_with("../")
}

fn runtime_script_output_is_environment_skip(output: &str) -> bool {
    let lower = output.to_ascii_lowercase();
    [
        "missing model",
        "model not found",
        "missing daemon binary",
        "daemon binary",
        "build it with:",
        "fewer than 2",
        "less than 2",
        "skipping",
        "not found",
    ]
    .iter()
    .any(|needle| lower.contains(needle))
}

fn truncate_for_metric(value: &str, max_chars: usize) -> String {
    if value.chars().count() <= max_chars {
        return value.to_string();
    }
    let mut out = value.chars().take(max_chars).collect::<String>();
    out.push_str("...");
    out
}

#[derive(Clone, Copy)]
struct DflashMatrixCase {
    label: &'static str,
    prompt_path: &'static str,
    mode: &'static str,
    max_tokens: usize,
    extra_args: &'static [&'static str],
}

fn dflash_matrix_cases() -> &'static [DflashMatrixCase] {
    &[
        DflashMatrixCase {
            label: "dflash_prose_attractor",
            prompt_path: "benchmarks/prompts/dflash_resident_smoke.txt",
            mode: "dflash",
            max_tokens: 192,
            extra_args: &[],
        },
        DflashMatrixCase {
            label: "dflash_code_attractor",
            prompt_path: "benchmarks/prompts/humaneval_0_has_close_elements.txt",
            mode: "dflash",
            max_tokens: 128,
            extra_args: &[],
        },
        DflashMatrixCase {
            label: "ddtree_b12_k2_prose",
            prompt_path: "benchmarks/prompts/dflash_resident_smoke.txt",
            mode: "ddtree-batched",
            max_tokens: 192,
            extra_args: &[
                "--ddtree-batched",
                "--ddtree-budget",
                "12",
                "--ddtree-topk",
                "2",
            ],
        },
        DflashMatrixCase {
            label: "path_c_phase1_prose",
            prompt_path: "benchmarks/prompts/dflash_resident_smoke.txt",
            mode: "path-c-phase1",
            max_tokens: 192,
            extra_args: &[
                "--ddtree-path-c",
                "phase1",
                "--ddtree-budget",
                "12",
                "--ddtree-topk",
                "2",
            ],
        },
        DflashMatrixCase {
            label: "path_c_phase2_prose",
            prompt_path: "benchmarks/prompts/dflash_resident_smoke.txt",
            mode: "path-c-phase2",
            max_tokens: 192,
            extra_args: &[
                "--ddtree-path-c",
                "phase2",
                "--ddtree-budget",
                "12",
                "--ddtree-topk",
                "2",
            ],
        },
    ]
}

fn run_examples_dflash_matrix_rows(config: &EvalConfig, ctx: &EvalContext) -> Vec<EvalResult> {
    if matches!(config.dflash, DflashMode::Off) {
        return dflash_matrix_cases()
            .iter()
            .map(|case| {
                skip_row(
                    BatteryId::Dflash,
                    None,
                    case.label,
                    None,
                    "DFlash/DDTree matrix disabled by --dflash off",
                    config,
                    ctx,
                    prompt(case.prompt_path),
                )
            })
            .collect();
    }
    dflash_matrix_cases()
        .iter()
        .map(|case| run_examples_dflash_matrix_case(config, ctx, *case))
        .collect()
}

fn run_examples_dflash_matrix_case(
    config: &EvalConfig,
    ctx: &EvalContext,
    case: DflashMatrixCase,
) -> EvalResult {
    let mut metrics = BTreeMap::from([
        ("suite".to_string(), json!("dflash_ddtree_path_c")),
        ("mode".to_string(), json!(case.mode)),
        ("max_tokens".to_string(), json!(case.max_tokens)),
        (
            "token_attractor_detector".to_string(),
            json!("dflash_spec_demo_tokens_v1"),
        ),
    ]);
    let mut extra_args = case.extra_args.to_vec();
    let max_tokens = case.max_tokens.to_string();
    extra_args.extend(["--max", max_tokens.as_str()]);
    metrics.insert("extra_args".to_string(), json!(case.extra_args));
    run_dflash_spec_demo_anchor(
        BatteryId::Dflash,
        case.label,
        false,
        case.prompt_path,
        &extra_args,
        metrics,
        config,
        ctx,
        config.model.clone(),
    )
}

#[derive(Clone, Copy)]
struct PflashNiahCase {
    label: &'static str,
    fixture: &'static str,
    mode: &'static str,
}

fn pflash_niah_cases() -> &'static [PflashNiahCase] {
    &[
        PflashNiahCase {
            label: "niah_8k_baseline",
            fixture: "benchmarks/longctx/niah/niah_8k.jsonl",
            mode: "baseline",
        },
        PflashNiahCase {
            label: "niah_8k_pflash30",
            fixture: "benchmarks/longctx/niah/niah_8k.jsonl",
            mode: "pflash",
        },
        PflashNiahCase {
            label: "niah_16k_baseline",
            fixture: "benchmarks/longctx/niah/niah_16k.jsonl",
            mode: "baseline",
        },
        PflashNiahCase {
            label: "niah_16k_pflash30",
            fixture: "benchmarks/longctx/niah/niah_16k.jsonl",
            mode: "pflash",
        },
        PflashNiahCase {
            label: "niah_multi_16k_baseline",
            fixture: "benchmarks/longctx/niah/niah_multi_16k.jsonl",
            mode: "baseline",
        },
        PflashNiahCase {
            label: "niah_multi_16k_pflash30",
            fixture: "benchmarks/longctx/niah/niah_multi_16k.jsonl",
            mode: "pflash",
        },
        PflashNiahCase {
            label: "longcode_baseline",
            fixture: "benchmarks/prompts/longcode_pflash.jsonl",
            mode: "baseline",
        },
        PflashNiahCase {
            label: "longcode_pflash30",
            fixture: "benchmarks/prompts/longcode_pflash.jsonl",
            mode: "pflash",
        },
        PflashNiahCase {
            label: "longprose_baseline",
            fixture: "benchmarks/prompts/longprose_multidoc.jsonl",
            mode: "baseline",
        },
        PflashNiahCase {
            label: "longprose_pflash30",
            fixture: "benchmarks/prompts/longprose_multidoc.jsonl",
            mode: "pflash",
        },
        PflashNiahCase {
            label: "niah_32k_baseline",
            fixture: "benchmarks/longctx/niah/niah_32k.jsonl",
            mode: "baseline",
        },
        PflashNiahCase {
            label: "niah_32k_pflash30",
            fixture: "benchmarks/longctx/niah/niah_32k.jsonl",
            mode: "pflash",
        },
    ]
}

fn pflash_maxgen_for_fixture(fixture: &str) -> usize {
    if fixture.contains("multi") {
        80
    } else if fixture.contains("longcode") || fixture.contains("longprose") {
        64
    } else {
        32
    }
}

fn run_examples_pflash_rows(config: &EvalConfig, ctx: &EvalContext) -> Vec<EvalResult> {
    pflash_niah_cases()
        .iter()
        .map(|case| run_examples_pflash_case(config, ctx, *case))
        .collect()
}

fn run_examples_pflash_case(
    config: &EvalConfig,
    ctx: &EvalContext,
    case: PflashNiahCase,
) -> EvalResult {
    let model = config.model.clone();
    let prompt_ref = prompt(case.fixture);
    let mut base_metrics = BTreeMap::from([
        ("implemented".to_string(), json!(true)),
        ("executor".to_string(), json!("examples")),
        ("suite".to_string(), json!("pflash_niah")),
        ("fixture".to_string(), json!(case.fixture)),
        ("mode".to_string(), json!(case.mode)),
        ("kv_mode".to_string(), json!("asym3")),
        ("pretok".to_string(), json!(true)),
        (
            "maxgen".to_string(),
            json!(pflash_maxgen_for_fixture(case.fixture)),
        ),
    ]);

    if !Path::new(&model).exists() {
        return row_for_model(
            BatteryId::Pflash,
            None,
            case.label,
            None,
            EvalStatus::Skip,
            Some(
                "pflash examples executor requires --model to be a local filesystem path"
                    .to_string(),
            ),
            base_metrics,
            config,
            ctx,
            prompt_ref,
            0,
            model,
        );
    }
    let Some(bin) = resolve_pflash_niah_bench_bin() else {
        return row_for_model(
            BatteryId::Pflash,
            None,
            case.label,
            None,
            EvalStatus::Skip,
            Some("pflash_niah_bench example binary not found; build with `cargo build --release --features deltanet -p hipfire-runtime --example pflash_niah_bench`".to_string()),
            base_metrics,
            config,
            ctx,
            prompt_ref,
            0,
            model,
        );
    };
    let Some(fixture_path) = resolve_repo_path(case.fixture) else {
        return row_for_model(
            BatteryId::Pflash,
            None,
            case.label,
            None,
            EvalStatus::Fail,
            Some(format!("pflash fixture not found: {}", case.fixture)),
            base_metrics,
            config,
            ctx,
            prompt_ref,
            0,
            model,
        );
    };

    let mut args = vec![
        model.clone(),
        fixture_path.display().to_string(),
        "--maxgen".to_string(),
        pflash_maxgen_for_fixture(case.fixture).to_string(),
        "--asym3".to_string(),
        "--pretok".to_string(),
    ];
    if case.mode == "pflash" {
        let Some(draft) = config.draft.as_deref() else {
            return row_for_model(
                BatteryId::Pflash,
                None,
                case.label,
                None,
                EvalStatus::Skip,
                Some("pflash examples executor requires --draft for pflash mode".to_string()),
                base_metrics,
                config,
                ctx,
                prompt_ref,
                0,
                model,
            );
        };
        if !Path::new(draft).exists() {
            return row_for_model(
                BatteryId::Pflash,
                None,
                case.label,
                None,
                EvalStatus::Skip,
                Some(
                    "pflash examples executor requires --draft to be a local filesystem path"
                        .to_string(),
                ),
                base_metrics,
                config,
                ctx,
                prompt_ref,
                0,
                model,
            );
        }
        base_metrics.insert("keep_ratio".to_string(), json!(0.30));
        base_metrics.insert("block_size".to_string(), json!(64));
        args.extend([
            "--pflash".to_string(),
            draft.to_string(),
            "--keep-ratio".to_string(),
            "0.30".to_string(),
            "--block-size".to_string(),
            "64".to_string(),
        ]);
    }

    let command_display = format!("{} {}", bin.display(), args.join(" "));
    let started = SystemTime::now();
    let output = match Command::new(&bin).args(&args).output() {
        Ok(output) => output,
        Err(err) => {
            base_metrics.insert("command".to_string(), json!(command_display));
            return row_for_model(
                BatteryId::Pflash,
                None,
                case.label,
                None,
                EvalStatus::Fail,
                Some(format!("spawn pflash_niah_bench: {err}")),
                base_metrics,
                config,
                ctx,
                prompt_ref,
                elapsed_since_ms(started),
                model,
            );
        }
    };
    let elapsed_ms = elapsed_since_ms(started);
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    let combined = format!("{stdout}\n{stderr}");
    let mut metrics = parse_pflash_niah_metrics(&combined);
    metrics.extend(base_metrics);
    metrics.insert("command".to_string(), json!(command_display));
    metrics.insert(
        "stdout_hash".to_string(),
        json!(stable_hash_bytes(stdout.as_bytes())),
    );
    metrics.insert(
        "stderr_hash".to_string(),
        json!(stable_hash_bytes(stderr.as_bytes())),
    );

    let verdict = metrics
        .get("niah_verdict")
        .and_then(Value::as_str)
        .map(str::to_string);
    let ok = output.status.success() && metrics.contains_key("total_ms") && verdict.is_some();
    row_for_model(
        BatteryId::Pflash,
        None,
        case.label,
        None,
        if ok {
            EvalStatus::Pass
        } else {
            EvalStatus::Fail
        },
        if ok {
            None
        } else if output.status.success() {
            Some("pflash_niah_bench did not emit total/verdict metrics".to_string())
        } else {
            Some(format!("pflash_niah_bench exited with {}", output.status))
        },
        metrics,
        config,
        ctx,
        prompt_ref,
        elapsed_ms,
        model,
    )
}

fn parse_pflash_niah_metrics(output: &str) -> BTreeMap<String, Value> {
    let mut metrics = BTreeMap::new();
    for raw in output.lines() {
        let line = raw.trim();
        if let Some(value) = prefixed_ms(line, "total:") {
            metrics.insert("total_ms".to_string(), json!(value));
        } else if let Some(value) = prefixed_ms(line, "prefill:") {
            metrics.insert("prefill_ms".to_string(), json!(value));
        } else if let Some(value) = prefixed_ms(line, "decode:") {
            metrics.insert("decode_ms".to_string(), json!(value));
        } else if let Some(value) = prefixed_ms(line, "compress:") {
            metrics.insert("compress_ms".to_string(), json!(value));
        } else if line.starts_with("PASS:") {
            metrics.insert("niah_verdict".to_string(), json!("PASS"));
            metrics.insert("niah_pass".to_string(), json!(1.0));
        } else if line.starts_with("FAIL:") {
            metrics.insert("niah_verdict".to_string(), json!("FAIL"));
            metrics.insert("niah_pass".to_string(), json!(0.0));
        }
    }
    metrics
}

fn prefixed_ms(line: &str, prefix: &str) -> Option<u64> {
    line.strip_prefix(prefix)?
        .split_whitespace()
        .next()?
        .parse()
        .ok()
}

#[allow(clippy::too_many_arguments)]
fn run_daemon_coherence_anchor(
    battery: BatteryId,
    case_id: &str,
    prompt_path: &str,
    prompt_ref: Option<PromptRef>,
    config: &EvalConfig,
    ctx: &EvalContext,
    model: String,
    max_seq: Option<usize>,
    max_tokens: Option<usize>,
    mut metrics: BTreeMap<String, Value>,
    system_path: Option<&str>,
    tools: Option<Value>,
    assistant_prefix: Option<&str>,
    force_jinja_chat: bool,
    profile: Option<hipfire_coherence::DetectorProfile>,
) -> EvalResult {
    metrics.insert("executor".to_string(), json!("daemon"));
    metrics.insert("runtime_path".to_string(), json!("daemon_jsonl"));
    metrics.insert("implemented".to_string(), json!(true));
    if !Path::new(&model).exists() {
        return row_for_model(
            battery,
            None,
            case_id,
            None,
            EvalStatus::Skip,
            Some(
                "daemon coherence executor requires --model to be a local filesystem path"
                    .to_string(),
            ),
            metrics,
            config,
            ctx,
            prompt_ref,
            0,
            model,
        );
    }
    if !hipfire_coherence::daemon_binary_available() {
        return row_for_model(
            battery,
            None,
            case_id,
            None,
            EvalStatus::Skip,
            Some("daemon binary not found; build with `cargo build --release --features deltanet -p hipfire-daemon --bin hipfire-daemon`".to_string()),
            metrics,
            config,
            ctx,
            prompt_ref,
            0,
            model,
        );
    }
    let Some(resolved_prompt) = resolve_repo_path(prompt_path) else {
        return row_for_model(
            battery,
            None,
            case_id,
            None,
            EvalStatus::Fail,
            Some(format!("prompt not found: {prompt_path}")),
            metrics,
            config,
            ctx,
            prompt_ref,
            0,
            model,
        );
    };
    let prompt_text = match fs::read_to_string(&resolved_prompt) {
        Ok(text) => text,
        Err(err) => {
            return row_for_model(
                battery,
                None,
                case_id,
                None,
                EvalStatus::Fail,
                Some(format!("read prompt {}: {err}", resolved_prompt.display())),
                metrics,
                config,
                ctx,
                prompt_ref,
                0,
                model,
            );
        }
    };
    let system_text = if let Some(path) = system_path {
        let Some(resolved_system) = resolve_repo_path(path) else {
            return row_for_model(
                battery,
                None,
                case_id,
                None,
                EvalStatus::Fail,
                Some(format!("system prompt not found: {path}")),
                metrics,
                config,
                ctx,
                prompt_ref,
                0,
                model,
            );
        };
        match fs::read_to_string(&resolved_system) {
            Ok(text) => Some(text),
            Err(err) => {
                return row_for_model(
                    battery,
                    None,
                    case_id,
                    None,
                    EvalStatus::Fail,
                    Some(format!(
                        "read system prompt {}: {err}",
                        resolved_system.display()
                    )),
                    metrics,
                    config,
                    ctx,
                    prompt_ref,
                    0,
                    model,
                );
            }
        }
    } else {
        None
    };
    let profile = profile.unwrap_or_else(|| {
        hipfire_coherence::DetectorProfile::default_for_prompt(&prompt_text, system_text.as_deref())
    });
    let run_config = hipfire_coherence::CoherenceRunConfig {
        model: model.clone(),
        prompt: prompt_text,
        prompt_label: prompt_path.to_string(),
        system: system_text,
        tools,
        assistant_prefix: assistant_prefix.map(str::to_string),
        force_jinja_chat,
        max_tokens: max_tokens.unwrap_or(config.max_tokens),
        temperature: 0.0,
        repeat_penalty: None,
        repeat_window: None,
        max_seq: max_seq.unwrap_or_else(|| (config.max_tokens + 2048).max(4096)),
        state: None,
        profile,
    };
    let started = SystemTime::now();
    let output = match hipfire_coherence::run_coherence(&run_config) {
        Ok(output) => output,
        Err(err) => {
            return row_for_model(
                battery,
                None,
                case_id,
                None,
                EvalStatus::Fail,
                Some(format!("daemon coherence probe failed: {err}")),
                metrics,
                config,
                ctx,
                prompt_ref,
                elapsed_since_ms(started),
                model,
            );
        }
    };
    let elapsed_ms = elapsed_since_ms(started);
    let artifact_dir = config.out_dir.join("artifacts").join("coherence");
    if let Err(err) = fs::create_dir_all(&artifact_dir) {
        return row_for_model(
            battery,
            None,
            case_id,
            None,
            EvalStatus::Fail,
            Some(format!("create coherence artifact dir: {err}")),
            metrics,
            config,
            ctx,
            prompt_ref,
            elapsed_ms,
            model,
        );
    }
    let artifact_name = format!(
        "{}-{}.json",
        sanitize_path_component(case_id),
        stable_hash_bytes(model.as_bytes())
    );
    let artifact_path = artifact_dir.join(artifact_name);
    let artifact_value = output.artifact_value();
    if let Err(err) = write_json_pretty(&artifact_path, &artifact_value) {
        return row_for_model(
            battery,
            None,
            case_id,
            None,
            EvalStatus::Fail,
            Some(format!("write coherence artifact: {err}")),
            metrics,
            config,
            ctx,
            prompt_ref,
            elapsed_ms,
            model,
        );
    }
    metrics.insert("hard_fails".to_string(), json!(output.hard_fails() as f64));
    metrics.insert("soft_warns".to_string(), json!(output.soft_warns() as f64));
    metrics.insert(
        "detector_count".to_string(),
        json!(output.report.rows.len() as f64),
    );
    metrics.insert(
        "detectors".to_string(),
        json!(hipfire_coherence::detector_rows(&output.report)),
    );
    metrics.insert(
        "generated_text_hash".to_string(),
        json!(stable_hash_bytes(output.generated_text.as_bytes())),
    );
    metrics.insert(
        "generated_visible_bytes".to_string(),
        json!(output.generated_text.len()),
    );
    metrics.insert(
        "generated_tokens".to_string(),
        json!(output.token_ids.len()),
    );
    metrics.insert("tok_s".to_string(), json!(output.report.header.tok_s));
    metrics.insert(
        "gen_tok_s".to_string(),
        json!(output.report.header.gen_tok_s),
    );
    metrics.insert("ttft_ms".to_string(), json!(output.report.header.ttft_ms));
    metrics.insert(
        "daemon_prefill_ms".to_string(),
        json!(output.report.header.daemon_prefill_ms),
    );
    metrics.insert(
        "daemon_decode_tok_s".to_string(),
        json!(output.report.header.daemon_decode_tok_s),
    );
    metrics.insert(
        "coherence_artifact_path".to_string(),
        json!(artifact_path.display().to_string()),
    );
    metrics.insert(
        "coherence_status".to_string(),
        json!(if output.hard_fails() > 0 {
            "fail"
        } else {
            "pass"
        }),
    );
    let status = if output.hard_fails() > 0 {
        EvalStatus::Fail
    } else {
        EvalStatus::Pass
    };
    let reason = if output.hard_fails() > 0 {
        Some(format!(
            "{} detector hard fail(s); see {}",
            output.hard_fails(),
            artifact_path.display()
        ))
    } else {
        None
    };
    row_for_model(
        battery, None, case_id, None, status, reason, metrics, config, ctx, prompt_ref, elapsed_ms,
        model,
    )
}

struct LongctxPrompt {
    prompt_path: String,
    prompt_ref: PromptRef,
    max_seq: usize,
    metrics: BTreeMap<String, Value>,
}

fn materialize_longctx_prompt(config: &EvalConfig) -> Result<LongctxPrompt, String> {
    let fixture_rel = "benchmarks/prompts/longprose_multidoc.jsonl";
    let fixture_path = repo_root()
        .unwrap_or_else(|| PathBuf::from("."))
        .join(fixture_rel);
    let fixture = fs::read_to_string(&fixture_path)
        .map_err(|e| format!("read longctx fixture {}: {e}", fixture_path.display()))?;
    let first_line = fixture
        .lines()
        .find(|line| !line.trim().is_empty())
        .ok_or_else(|| format!("longctx fixture {fixture_rel} is empty"))?;
    let item: Value = serde_json::from_str(first_line)
        .map_err(|e| format!("parse longctx fixture {fixture_rel}: {e}"))?;
    let filler = item
        .get("filler_text")
        .and_then(Value::as_str)
        .ok_or_else(|| format!("longctx fixture {fixture_rel} missing filler_text"))?;
    let question = item
        .get("question")
        .and_then(Value::as_str)
        .ok_or_else(|| format!("longctx fixture {fixture_rel} missing question"))?;
    let expected = item
        .get("expected_answer_substring")
        .and_then(Value::as_str)
        .unwrap_or("");
    let context_tokens = item
        .get("context_tokens")
        .and_then(Value::as_u64)
        .unwrap_or(4096) as usize;
    let prompt_text = format!(
        "{filler}\n\nQuestion: {question}\nAnswer in one short sentence and include the exact requested value.\n"
    );
    let prompt_dir = config.out_dir.join("artifacts").join("runtime_prompts");
    fs::create_dir_all(&prompt_dir)
        .map_err(|e| format!("create longctx prompt dir {}: {e}", prompt_dir.display()))?;
    let prompt_path = prompt_dir.join("longctx_multidoc_0.txt");
    fs::write(&prompt_path, &prompt_text)
        .map_err(|e| format!("write longctx prompt {}: {e}", prompt_path.display()))?;
    let prompt_path_string = prompt_path.display().to_string();
    let prompt_ref = PromptRef::from_content(prompt_path_string.clone(), prompt_text.as_bytes());
    let max_seq = (context_tokens + config.max_tokens + 512).max(4096);
    let mut metrics = BTreeMap::new();
    metrics.insert("fixture".to_string(), json!("longprose_multidoc"));
    metrics.insert("fixture_source".to_string(), json!(fixture_rel));
    metrics.insert(
        "fixture_hash".to_string(),
        json!(file_hash(&fixture_path).unwrap_or_else(|| stable_hash_file_fallback(&fixture_path))),
    );
    metrics.insert("context_tokens".to_string(), json!(context_tokens));
    metrics.insert("longctx_max_seq".to_string(), json!(max_seq));
    metrics.insert("prompt_bytes".to_string(), json!(prompt_text.len()));
    metrics.insert(
        "expected_answer_hash".to_string(),
        json!(stable_hash_bytes(expected.as_bytes())),
    );
    if let Some(needle) = item.get("needle").and_then(Value::as_str) {
        metrics.insert(
            "needle_hash".to_string(),
            json!(stable_hash_bytes(needle.as_bytes())),
        );
    }
    if let Some(documents) = item.get("documents").and_then(Value::as_array) {
        metrics.insert("document_count".to_string(), json!(documents.len()));
    }
    Ok(LongctxPrompt {
        prompt_path: prompt_path_string,
        prompt_ref,
        max_seq,
        metrics,
    })
}

fn examples_barrage_rows(
    config: &EvalConfig,
    ctx: &EvalContext,
    datasets: &[DatasetManifestEntry],
) -> Vec<EvalResult> {
    let mut rows = Vec::new();
    for d in datasets {
        match (d.suite, d.status) {
            (SuiteId::Gpqa, EvalStatus::Pass) => {
                match gpqa_materialized_items(Path::new(&d.cache_path), &d.selected_item_ids) {
                    Ok(items) => {
                        rows.extend(items.into_iter().flat_map(|item| {
                            evaluation_models(config).into_iter().map(move |model| {
                                run_examples_gpqa_item(config, ctx, d, item.clone(), model)
                            })
                        }));
                    }
                    Err(reason) => {
                        rows.extend(d.selected_item_ids.iter().cloned().map(|id| {
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
                        }));
                    }
                }
            }
            (SuiteId::LmEvalMicro, EvalStatus::Pass) => {
                match lm_eval_micro_materialized_items(&d.selected_item_ids) {
                    Ok(items) => {
                        rows.extend(items.into_iter().flat_map(|item| {
                            evaluation_models(config).into_iter().map(move |model| {
                                run_examples_lm_eval_micro_item(config, ctx, d, item.clone(), model)
                            })
                        }));
                    }
                    Err(reason) => {
                        rows.extend(d.selected_item_ids.iter().cloned().map(|id| {
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
                        }));
                    }
                }
            }
            (SuiteId::HumanEval, EvalStatus::Pass) => {
                match humaneval_materialized_items(Path::new(&d.cache_path), &d.selected_item_ids) {
                    Ok(items) => {
                        rows.extend(items.into_iter().flat_map(|item| {
                            evaluation_models(config).into_iter().map(move |model| {
                                run_examples_humaneval_item(config, ctx, d, item.clone(), model)
                            })
                        }));
                    }
                    Err(reason) => {
                        rows.extend(d.selected_item_ids.iter().cloned().map(|id| {
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
                        }));
                    }
                }
            }
            (SuiteId::DeepSwe | SuiteId::SweBench, EvalStatus::Pass) => {
                match builtin_barrage_materialized_items(d.suite, &d.selected_item_ids) {
                    Ok(items) => {
                        rows.extend(items.into_iter().flat_map(|item| {
                            evaluation_models(config).into_iter().map(move |model| {
                                run_examples_builtin_barrage_item(
                                    config,
                                    ctx,
                                    d,
                                    item.clone(),
                                    model,
                                )
                            })
                        }));
                    }
                    Err(reason) => {
                        rows.extend(d.selected_item_ids.iter().cloned().map(|id| {
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
                        }));
                    }
                }
            }
            _ => {}
        }
    }
    if rows.is_empty() {
        barrage_rows(config, ctx, datasets)
    } else {
        rows
    }
}

fn evaluation_models(config: &EvalConfig) -> Vec<String> {
    std::iter::once(&config.model)
        .chain(config.baseline.iter())
        .chain(config.reference.iter())
        .cloned()
        .collect()
}

fn run_examples_lm_eval_micro_item(
    config: &EvalConfig,
    ctx: &EvalContext,
    dataset: &DatasetManifestEntry,
    item: LmEvalMicroItem,
    model: String,
) -> EvalResult {
    let prompt_ref = PromptRef::from_content(
        format!("dataset:lm_eval_micro:{}", item.item_id),
        item.prompt.as_bytes(),
    );
    let mut base_metrics = BTreeMap::from([
        (
            "prompt_format".to_string(),
            json!("lm_eval_micro_zero_shot_v1"),
        ),
        ("task".to_string(), json!(item.task.clone())),
        ("answer_label".to_string(), json!(item.answer_label.clone())),
        ("answer_hash".to_string(), json!(item.answer_hash.clone())),
        ("choices_count".to_string(), json!(item.choices_count)),
        (
            "dataset_file".to_string(),
            json!("builtin:lm_eval_micro:v1"),
        ),
        ("scoring_mode".to_string(), json!("exact_letter")),
        ("executor".to_string(), json!("examples")),
    ]);
    add_dataset_provenance_metrics(&mut base_metrics, dataset);

    if !Path::new(&model).exists() {
        return row_for_model(
            BatteryId::Barrage,
            Some(SuiteId::LmEvalMicro),
            "lm_eval_micro_zero_shot_native",
            Some(item.item_id),
            EvalStatus::Skip,
            Some(
                "examples executor requires each evaluated model to be a local filesystem path"
                    .to_string(),
            ),
            base_metrics,
            config,
            ctx,
            Some(prompt_ref),
            0,
            model,
        );
    }
    let Some(bin) = resolve_run_example_bin() else {
        return row_for_model(
            BatteryId::Barrage,
            Some(SuiteId::LmEvalMicro),
            "lm_eval_micro_zero_shot_native",
            Some(item.item_id),
            EvalStatus::Skip,
            Some("run example binary not found; build with `cargo build --release --features deltanet -p hipfire-runtime --example run`".to_string()),
            base_metrics,
            config,
            ctx,
            Some(prompt_ref),
            0,
            model,
        );
    };

    let prompt_dir = config.out_dir.join("artifacts").join("runtime_prompts");
    if let Err(err) = fs::create_dir_all(&prompt_dir) {
        return row_for_model(
            BatteryId::Barrage,
            Some(SuiteId::LmEvalMicro),
            "lm_eval_micro_zero_shot_native",
            Some(item.item_id),
            EvalStatus::Fail,
            Some(format!("create runtime prompt dir: {err}")),
            base_metrics,
            config,
            ctx,
            Some(prompt_ref),
            0,
            model,
        );
    }
    let prompt_file = prompt_dir.join(format!(
        "lm-eval-micro-{}.txt",
        sanitize_path_component(&item.item_id)
    ));
    if let Err(err) = fs::write(&prompt_file, &item.prompt) {
        return row_for_model(
            BatteryId::Barrage,
            Some(SuiteId::LmEvalMicro),
            "lm_eval_micro_zero_shot_native",
            Some(item.item_id),
            EvalStatus::Fail,
            Some(format!("write lm_eval_micro runtime prompt: {err}")),
            base_metrics,
            config,
            ctx,
            Some(prompt_ref),
            0,
            model,
        );
    }

    let evidence_dir =
        runtime_evidence_dir(config, &format!("lm-eval-micro-{}", item.item_id), &model);
    let mut args = vec![
        model.clone(),
        "--prompt-file".to_string(),
        prompt_file.display().to_string(),
        "--max-tokens".to_string(),
        config.max_tokens.to_string(),
        "--kv".to_string(),
        config.kv_mode.clone().unwrap_or_else(|| "q8".to_string()),
        "--temp".to_string(),
        "0.0".to_string(),
    ];
    add_runtime_evidence_arg(&mut args, &evidence_dir);
    if config.max_tokens + 2048 > 4096 {
        args.push("--max-seq".to_string());
        args.push((config.max_tokens + 2048).to_string());
    }
    let command_display = format!("{} {}", bin.display(), args.join(" "));
    let started = SystemTime::now();
    let output = match Command::new(&bin).args(&args).output() {
        Ok(output) => output,
        Err(err) => {
            base_metrics.insert("command".to_string(), json!(command_display));
            return row_for_model(
                BatteryId::Barrage,
                Some(SuiteId::LmEvalMicro),
                "lm_eval_micro_zero_shot_native",
                Some(item.item_id),
                EvalStatus::Fail,
                Some(format!("spawn run example: {err}")),
                base_metrics,
                config,
                ctx,
                Some(prompt_ref),
                elapsed_since_ms(started),
                model,
            );
        }
    };
    let elapsed_ms = elapsed_since_ms(started);
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    let mut metrics = parse_bench_metrics(&stderr);
    metrics.extend(base_metrics);
    metrics.insert("implemented".to_string(), json!(true));
    metrics.insert("command".to_string(), json!(command_display));
    metrics.insert(
        "runtime_prompt_path".to_string(),
        json!(prompt_file.display().to_string()),
    );
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
    if let Some(v) = metrics.get("decode_tok_s").cloned() {
        metrics.entry("tok_s".to_string()).or_insert(v);
    }
    let predicted = extract_answer_letter(&stdout);
    metrics.insert("predicted_label".to_string(), json!(predicted));
    let exact_match = predicted.as_deref() == Some(item.answer_label.as_str());
    metrics.insert(
        "exact_match".to_string(),
        json!(if exact_match { 1.0 } else { 0.0 }),
    );
    metrics.insert(
        "accuracy".to_string(),
        json!(if exact_match { 1.0 } else { 0.0 }),
    );

    if output.status.success() && metrics.contains_key("decode_tok_s") {
        row_for_model(
            BatteryId::Barrage,
            Some(SuiteId::LmEvalMicro),
            "lm_eval_micro_zero_shot_native",
            Some(item.item_id),
            EvalStatus::Pass,
            None,
            metrics,
            config,
            ctx,
            Some(prompt_ref),
            elapsed_ms,
            model,
        )
    } else {
        row_for_model(
            BatteryId::Barrage,
            Some(SuiteId::LmEvalMicro),
            "lm_eval_micro_zero_shot_native",
            Some(item.item_id),
            EvalStatus::Fail,
            Some(if output.status.success() {
                "run example did not emit BENCH METRICS".to_string()
            } else {
                format!("run example exited with {}", output.status)
            }),
            metrics,
            config,
            ctx,
            Some(prompt_ref),
            elapsed_ms,
            model,
        )
    }
}

fn run_examples_builtin_barrage_item(
    config: &EvalConfig,
    ctx: &EvalContext,
    dataset: &DatasetManifestEntry,
    item: BuiltinBarrageItem,
    model: String,
) -> EvalResult {
    let case_id = "builtin_software_eval_native";
    let prompt_ref = PromptRef::from_content(
        format!("dataset:{}:{}", item.suite.as_str(), item.item_id),
        item.prompt.as_bytes(),
    );
    let mut base_metrics = BTreeMap::from([
        (
            "prompt_format".to_string(),
            json!(item.prompt_format.clone()),
        ),
        ("task".to_string(), json!(item.task.clone())),
        ("answer_label".to_string(), json!(item.answer_label.clone())),
        ("answer_hash".to_string(), json!(item.answer_hash.clone())),
        ("choices_count".to_string(), json!(item.choices_count)),
        ("dataset_file".to_string(), json!(item.dataset_file.clone())),
        ("scoring_mode".to_string(), json!(item.scoring_mode.clone())),
        ("executor".to_string(), json!("examples")),
    ]);
    add_dataset_provenance_metrics(&mut base_metrics, dataset);

    if !Path::new(&model).exists() {
        return row_for_model(
            BatteryId::Barrage,
            Some(item.suite),
            case_id,
            Some(item.item_id),
            EvalStatus::Skip,
            Some(
                "examples executor requires each evaluated model to be a local filesystem path"
                    .to_string(),
            ),
            base_metrics,
            config,
            ctx,
            Some(prompt_ref),
            0,
            model,
        );
    }
    let Some(bin) = resolve_run_example_bin() else {
        return row_for_model(
            BatteryId::Barrage,
            Some(item.suite),
            case_id,
            Some(item.item_id),
            EvalStatus::Skip,
            Some("run example binary not found; build with `cargo build --release --features deltanet -p hipfire-runtime --example run`".to_string()),
            base_metrics,
            config,
            ctx,
            Some(prompt_ref),
            0,
            model,
        );
    };

    let prompt_dir = config.out_dir.join("artifacts").join("runtime_prompts");
    if let Err(err) = fs::create_dir_all(&prompt_dir) {
        return row_for_model(
            BatteryId::Barrage,
            Some(item.suite),
            case_id,
            Some(item.item_id),
            EvalStatus::Fail,
            Some(format!("create runtime prompt dir: {err}")),
            base_metrics,
            config,
            ctx,
            Some(prompt_ref),
            0,
            model,
        );
    }
    let prompt_file = prompt_dir.join(format!(
        "{}-{}.txt",
        item.suite.as_str().replace('_', "-"),
        sanitize_path_component(&item.item_id)
    ));
    if let Err(err) = fs::write(&prompt_file, &item.prompt) {
        return row_for_model(
            BatteryId::Barrage,
            Some(item.suite),
            case_id,
            Some(item.item_id),
            EvalStatus::Fail,
            Some(format!("write builtin software-eval runtime prompt: {err}")),
            base_metrics,
            config,
            ctx,
            Some(prompt_ref),
            0,
            model,
        );
    }

    let evidence_dir = runtime_evidence_dir(
        config,
        &format!("{}-{}", item.suite.as_str(), item.item_id),
        &model,
    );
    let mut args = vec![
        model.clone(),
        "--prompt-file".to_string(),
        prompt_file.display().to_string(),
        "--max-tokens".to_string(),
        config.max_tokens.to_string(),
        "--kv".to_string(),
        config.kv_mode.clone().unwrap_or_else(|| "q8".to_string()),
        "--temp".to_string(),
        "0.0".to_string(),
    ];
    add_runtime_evidence_arg(&mut args, &evidence_dir);
    if config.max_tokens + 2048 > 4096 {
        args.push("--max-seq".to_string());
        args.push((config.max_tokens + 2048).to_string());
    }
    let command_display = format!("{} {}", bin.display(), args.join(" "));
    let started = SystemTime::now();
    let output = match Command::new(&bin).args(&args).output() {
        Ok(output) => output,
        Err(err) => {
            base_metrics.insert("command".to_string(), json!(command_display));
            return row_for_model(
                BatteryId::Barrage,
                Some(item.suite),
                case_id,
                Some(item.item_id),
                EvalStatus::Fail,
                Some(format!("spawn run example: {err}")),
                base_metrics,
                config,
                ctx,
                Some(prompt_ref),
                elapsed_since_ms(started),
                model,
            );
        }
    };
    let elapsed_ms = elapsed_since_ms(started);
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    let mut metrics = parse_bench_metrics(&stderr);
    metrics.extend(base_metrics);
    metrics.insert("implemented".to_string(), json!(true));
    metrics.insert("command".to_string(), json!(command_display));
    metrics.insert(
        "runtime_prompt_path".to_string(),
        json!(prompt_file.display().to_string()),
    );
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
    if let Some(v) = metrics.get("decode_tok_s").cloned() {
        metrics.entry("tok_s".to_string()).or_insert(v);
    }
    let predicted = extract_answer_letter(&stdout);
    metrics.insert("predicted_label".to_string(), json!(predicted));
    let exact_match = predicted.as_deref() == Some(item.answer_label.as_str());
    metrics.insert(
        "exact_match".to_string(),
        json!(if exact_match { 1.0 } else { 0.0 }),
    );
    metrics.insert(
        "accuracy".to_string(),
        json!(if exact_match { 1.0 } else { 0.0 }),
    );

    if output.status.success() && metrics.contains_key("decode_tok_s") {
        row_for_model(
            BatteryId::Barrage,
            Some(item.suite),
            case_id,
            Some(item.item_id),
            EvalStatus::Pass,
            None,
            metrics,
            config,
            ctx,
            Some(prompt_ref),
            elapsed_ms,
            model,
        )
    } else {
        row_for_model(
            BatteryId::Barrage,
            Some(item.suite),
            case_id,
            Some(item.item_id),
            EvalStatus::Fail,
            Some(if output.status.success() {
                "run example did not emit BENCH METRICS".to_string()
            } else {
                format!("run example exited with {}", output.status)
            }),
            metrics,
            config,
            ctx,
            Some(prompt_ref),
            elapsed_ms,
            model,
        )
    }
}

fn run_examples_humaneval_item(
    config: &EvalConfig,
    ctx: &EvalContext,
    dataset: &DatasetManifestEntry,
    item: HumanEvalItem,
    model: String,
) -> EvalResult {
    let prompt_ref = PromptRef::from_content(
        format!("dataset:humaneval:{}", item.item_id),
        item.prompt.as_bytes(),
    );
    let mut metrics = BTreeMap::from([
        (
            "prompt_format".to_string(),
            json!("humaneval_completion_v1"),
        ),
        ("task_id".to_string(), json!(item.task_id.clone())),
        ("dataset_file".to_string(), json!(item.dataset_file.clone())),
        ("executor".to_string(), json!("examples")),
        ("scoring_mode".to_string(), json!("execution_only")),
    ]);
    add_dataset_provenance_metrics(&mut metrics, dataset);
    if let Some(hash) = &item.canonical_solution_hash {
        metrics.insert("canonical_solution_hash".to_string(), json!(hash));
    }
    if let Some(hash) = &item.test_hash {
        metrics.insert("test_hash".to_string(), json!(hash));
    }

    if !Path::new(&model).exists() {
        return row_for_model(
            BatteryId::Barrage,
            Some(SuiteId::HumanEval),
            "humaneval_completion_native",
            Some(item.item_id),
            EvalStatus::Skip,
            Some("examples executor requires --model to be a local filesystem path".to_string()),
            metrics,
            config,
            ctx,
            Some(prompt_ref),
            0,
            model,
        );
    }
    let Some(bin) = resolve_run_example_bin() else {
        return row_for_model(
            BatteryId::Barrage,
            Some(SuiteId::HumanEval),
            "humaneval_completion_native",
            Some(item.item_id),
            EvalStatus::Skip,
            Some("run example binary not found; build with `cargo build --release --features deltanet -p hipfire-runtime --example run`".to_string()),
            metrics,
            config,
            ctx,
            Some(prompt_ref),
            0,
            model,
        );
    };

    let prompt_dir = config.out_dir.join("artifacts").join("runtime_prompts");
    if let Err(err) = fs::create_dir_all(&prompt_dir) {
        return row_for_model(
            BatteryId::Barrage,
            Some(SuiteId::HumanEval),
            "humaneval_completion_native",
            Some(item.item_id),
            EvalStatus::Fail,
            Some(format!("create runtime prompt dir: {err}")),
            metrics,
            config,
            ctx,
            Some(prompt_ref),
            0,
            model,
        );
    }
    let prompt_file = prompt_dir.join(format!(
        "humaneval-{}.txt",
        sanitize_path_component(&item.item_id)
    ));
    if let Err(err) = fs::write(&prompt_file, &item.prompt) {
        return row_for_model(
            BatteryId::Barrage,
            Some(SuiteId::HumanEval),
            "humaneval_completion_native",
            Some(item.item_id),
            EvalStatus::Fail,
            Some(format!("write HumanEval runtime prompt: {err}")),
            metrics,
            config,
            ctx,
            Some(prompt_ref),
            0,
            model,
        );
    }

    let evidence_dir = runtime_evidence_dir(config, &format!("humaneval-{}", item.item_id), &model);
    let mut args = vec![
        model.clone(),
        "--prompt-file".to_string(),
        prompt_file.display().to_string(),
        "--max-tokens".to_string(),
        config.max_tokens.to_string(),
        "--kv".to_string(),
        config.kv_mode.clone().unwrap_or_else(|| "q8".to_string()),
        "--temp".to_string(),
        "0.0".to_string(),
    ];
    add_runtime_evidence_arg(&mut args, &evidence_dir);
    let command_display = format!("{} {}", bin.display(), args.join(" "));
    let started = SystemTime::now();
    let output = match Command::new(&bin).args(&args).output() {
        Ok(output) => output,
        Err(err) => {
            metrics.insert("command".to_string(), json!(command_display));
            return row_for_model(
                BatteryId::Barrage,
                Some(SuiteId::HumanEval),
                "humaneval_completion_native",
                Some(item.item_id),
                EvalStatus::Fail,
                Some(format!("spawn run example: {err}")),
                metrics,
                config,
                ctx,
                Some(prompt_ref),
                elapsed_since_ms(started),
                model,
            );
        }
    };
    let elapsed_ms = elapsed_since_ms(started);
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    metrics.extend(parse_bench_metrics(&stderr));
    metrics.insert("implemented".to_string(), json!(true));
    metrics.insert("command".to_string(), json!(command_display));
    metrics.insert(
        "runtime_prompt_path".to_string(),
        json!(prompt_file.display().to_string()),
    );
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
    if let Some(v) = metrics.get("decode_tok_s").cloned() {
        metrics.entry("tok_s".to_string()).or_insert(v);
    }

    if output.status.success() && metrics.contains_key("decode_tok_s") {
        row_for_model(
            BatteryId::Barrage,
            Some(SuiteId::HumanEval),
            "humaneval_completion_native",
            Some(item.item_id),
            EvalStatus::Pass,
            None,
            metrics,
            config,
            ctx,
            Some(prompt_ref),
            elapsed_ms,
            model,
        )
    } else {
        row_for_model(
            BatteryId::Barrage,
            Some(SuiteId::HumanEval),
            "humaneval_completion_native",
            Some(item.item_id),
            EvalStatus::Fail,
            Some(if output.status.success() {
                "run example did not emit BENCH METRICS".to_string()
            } else {
                format!("run example exited with {}", output.status)
            }),
            metrics,
            config,
            ctx,
            Some(prompt_ref),
            elapsed_ms,
            model,
        )
    }
}

fn run_examples_gpqa_item(
    config: &EvalConfig,
    ctx: &EvalContext,
    dataset: &DatasetManifestEntry,
    item: GpqaItem,
    model: String,
) -> EvalResult {
    let prompt_ref = PromptRef::from_content(
        format!("dataset:gpqa:{}", item.item_id),
        item.prompt.as_bytes(),
    );
    let mut base_metrics = BTreeMap::from([
        ("prompt_format".to_string(), json!("gpqa_zero_shot_v1")),
        ("answer_label".to_string(), json!(item.answer_label.clone())),
        (
            "answer_hash".to_string(),
            json!(stable_hash_bytes(item.correct_answer.as_bytes())),
        ),
        ("choices_count".to_string(), json!(item.choices.len())),
        ("dataset_file".to_string(), json!(item.dataset_file.clone())),
        ("executor".to_string(), json!("examples")),
    ]);
    add_dataset_provenance_metrics(&mut base_metrics, dataset);

    if !Path::new(&model).exists() {
        return row_for_model(
            BatteryId::Barrage,
            Some(SuiteId::Gpqa),
            "gpqa_zero_shot_native",
            Some(item.item_id),
            EvalStatus::Skip,
            Some(
                "examples executor requires each evaluated model to be a local filesystem path"
                    .to_string(),
            ),
            base_metrics,
            config,
            ctx,
            Some(prompt_ref),
            0,
            model,
        );
    }
    let Some(bin) = resolve_run_example_bin() else {
        return row_for_model(
            BatteryId::Barrage,
            Some(SuiteId::Gpqa),
            "gpqa_zero_shot_native",
            Some(item.item_id),
            EvalStatus::Skip,
            Some("run example binary not found; build with `cargo build --release --features deltanet -p hipfire-runtime --example run`".to_string()),
            base_metrics,
            config,
            ctx,
            Some(prompt_ref),
            0,
            model,
        );
    };

    let prompt_dir = config.out_dir.join("artifacts").join("runtime_prompts");
    if let Err(err) = fs::create_dir_all(&prompt_dir) {
        return row_for_model(
            BatteryId::Barrage,
            Some(SuiteId::Gpqa),
            "gpqa_zero_shot_native",
            Some(item.item_id),
            EvalStatus::Fail,
            Some(format!("create runtime prompt dir: {err}")),
            base_metrics,
            config,
            ctx,
            Some(prompt_ref),
            0,
            model,
        );
    }
    let prompt_file = prompt_dir.join(format!(
        "gpqa-{}.txt",
        sanitize_path_component(&item.item_id)
    ));
    if let Err(err) = fs::write(&prompt_file, &item.prompt) {
        return row_for_model(
            BatteryId::Barrage,
            Some(SuiteId::Gpqa),
            "gpqa_zero_shot_native",
            Some(item.item_id),
            EvalStatus::Fail,
            Some(format!("write GPQA runtime prompt: {err}")),
            base_metrics,
            config,
            ctx,
            Some(prompt_ref),
            0,
            model,
        );
    }

    let evidence_dir = runtime_evidence_dir(config, &format!("gpqa-{}", item.item_id), &model);
    let mut args = vec![
        model.clone(),
        "--prompt-file".to_string(),
        prompt_file.display().to_string(),
        "--max-tokens".to_string(),
        config.max_tokens.to_string(),
        "--kv".to_string(),
        config.kv_mode.clone().unwrap_or_else(|| "q8".to_string()),
        "--temp".to_string(),
        "0.0".to_string(),
    ];
    add_runtime_evidence_arg(&mut args, &evidence_dir);
    if config.max_tokens + 2048 > 4096 {
        args.push("--max-seq".to_string());
        args.push((config.max_tokens + 2048).to_string());
    }
    let command_display = format!("{} {}", bin.display(), args.join(" "));
    let started = SystemTime::now();
    let output = match Command::new(&bin).args(&args).output() {
        Ok(output) => output,
        Err(err) => {
            base_metrics.insert("command".to_string(), json!(command_display));
            return row_for_model(
                BatteryId::Barrage,
                Some(SuiteId::Gpqa),
                "gpqa_zero_shot_native",
                Some(item.item_id),
                EvalStatus::Fail,
                Some(format!("spawn run example: {err}")),
                base_metrics,
                config,
                ctx,
                Some(prompt_ref),
                elapsed_since_ms(started),
                model,
            );
        }
    };
    let elapsed_ms = elapsed_since_ms(started);
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    let mut metrics = parse_bench_metrics(&stderr);
    metrics.extend(base_metrics);
    metrics.insert("implemented".to_string(), json!(true));
    metrics.insert("command".to_string(), json!(command_display));
    metrics.insert(
        "runtime_prompt_path".to_string(),
        json!(prompt_file.display().to_string()),
    );
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
    if let Some(v) = metrics.get("decode_tok_s").cloned() {
        metrics.entry("tok_s".to_string()).or_insert(v);
    }
    let predicted = extract_answer_letter(&stdout);
    metrics.insert("predicted_label".to_string(), json!(predicted));
    let exact_match = predicted.as_deref() == Some(item.answer_label.as_str());
    metrics.insert(
        "exact_match".to_string(),
        json!(if exact_match { 1.0 } else { 0.0 }),
    );
    metrics.insert(
        "accuracy".to_string(),
        json!(if exact_match { 1.0 } else { 0.0 }),
    );

    if output.status.success() && metrics.contains_key("decode_tok_s") {
        row_for_model(
            BatteryId::Barrage,
            Some(SuiteId::Gpqa),
            "gpqa_zero_shot_native",
            Some(item.item_id),
            EvalStatus::Pass,
            None,
            metrics,
            config,
            ctx,
            Some(prompt_ref),
            elapsed_ms,
            model,
        )
    } else {
        row_for_model(
            BatteryId::Barrage,
            Some(SuiteId::Gpqa),
            "gpqa_zero_shot_native",
            Some(item.item_id),
            EvalStatus::Fail,
            Some(if output.status.success() {
                "run example did not emit BENCH METRICS".to_string()
            } else {
                format!("run example exited with {}", output.status)
            }),
            metrics,
            config,
            ctx,
            Some(prompt_ref),
            elapsed_ms,
            model,
        )
    }
}

fn extract_answer_letter(stdout: &str) -> Option<String> {
    let trimmed = stdout.trim_start();
    if let Some(first) = trimmed.chars().next() {
        if matches!(first.to_ascii_uppercase(), 'A' | 'B' | 'C' | 'D') {
            return Some(first.to_ascii_uppercase().to_string());
        }
    }
    trimmed
        .split(|c: char| !c.is_ascii_alphanumeric())
        .find_map(|token| {
            let mut chars = token.chars();
            let c = chars.next()?;
            if chars.next().is_none() && matches!(c.to_ascii_uppercase(), 'A' | 'B' | 'C' | 'D') {
                Some(c.to_ascii_uppercase().to_string())
            } else {
                None
            }
        })
}

fn run_dflash_spec_demo_anchor(
    battery: BatteryId,
    case_id: &str,
    ar_baseline: bool,
    prompt_path: &str,
    extra_args: &[&str],
    mut extra_metrics: BTreeMap<String, Value>,
    config: &EvalConfig,
    ctx: &EvalContext,
    model: String,
) -> EvalResult {
    let prompt_ref = prompt(prompt_path);
    if !Path::new(&model).exists() {
        return row_for_model(
            battery,
            None,
            case_id,
            None,
            EvalStatus::Skip,
            Some(
                "examples executor requires each evaluated model to be a local filesystem path"
                    .to_string(),
            ),
            BTreeMap::from([("executor".to_string(), json!("examples"))]),
            config,
            ctx,
            prompt_ref,
            0,
            model,
        );
    }
    let Some(draft) = config.draft.as_deref() else {
        if ar_baseline {
            return run_examples_run_anchor_with_prompt_for_model(
                battery,
                case_id,
                prompt_path,
                config,
                ctx,
                model,
            );
        }
        return row_for_model(
            battery,
            None,
            case_id,
            None,
            EvalStatus::Skip,
            Some("examples executor requires --draft for DFlash anchor".to_string()),
            BTreeMap::from([("executor".to_string(), json!("examples"))]),
            config,
            ctx,
            prompt_ref,
            0,
            model,
        );
    };
    if !Path::new(draft).exists() {
        return row_for_model(
            battery,
            None,
            case_id,
            None,
            EvalStatus::Skip,
            Some("examples executor requires --draft to be a local filesystem path".to_string()),
            BTreeMap::from([("executor".to_string(), json!("examples"))]),
            config,
            ctx,
            prompt_ref,
            0,
            model,
        );
    }
    let Some(bin) = resolve_dflash_spec_demo_bin() else {
        return row_for_model(
            battery,
            None,
            case_id,
            None,
            EvalStatus::Skip,
            Some("dflash_spec_demo example binary not found; build with `cargo build --release --features deltanet -p hipfire-runtime --example dflash_spec_demo`".to_string()),
            BTreeMap::from([("executor".to_string(), json!("examples"))]),
            config,
            ctx,
            prompt_ref,
            0,
            model,
        );
    };

    let evidence_dir = runtime_evidence_dir(config, case_id, &model);
    let mut args = vec![
        "--target".to_string(),
        model.clone(),
        "--draft".to_string(),
        draft.to_string(),
        "--prompt-file".to_string(),
        prompt_path.to_string(),
        "--max".to_string(),
        config.max_tokens.to_string(),
        "--ctx".to_string(),
        "2048".to_string(),
        "--kv-mode".to_string(),
        config.kv_mode.clone().unwrap_or_else(|| "q8".to_string()),
        "--no-adaptive-b".to_string(),
        "--no-chatml".to_string(),
    ];
    add_runtime_evidence_arg(&mut args, &evidence_dir);
    if ar_baseline {
        args.push("--ar-baseline".to_string());
    }
    args.extend(extra_args.iter().map(|arg| (*arg).to_string()));
    let command_display = format!("{} {}", bin.display(), args.join(" "));
    let started = SystemTime::now();
    let mut command = Command::new(&bin);
    command.args(&args);
    if config.profile == ProfileMode::Passive && !ar_baseline {
        command.env("HIPFIRE_PROFILE", "1");
        command.env("HIPFIRE_PROFILE_CYCLES", "1");
    }
    let output = match command.output() {
        Ok(output) => output,
        Err(err) => {
            return row_for_model(
                battery,
                None,
                case_id,
                None,
                EvalStatus::Fail,
                Some(format!("spawn dflash_spec_demo: {err}")),
                BTreeMap::from([
                    ("executor".to_string(), json!("examples")),
                    ("command".to_string(), json!(command_display)),
                ]),
                config,
                ctx,
                prompt_ref,
                elapsed_since_ms(started),
                model,
            );
        }
    };
    let elapsed_ms = elapsed_since_ms(started);
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    let mut metrics = parse_bench_metrics(&stderr);
    metrics.append(&mut extra_metrics);
    metrics.insert("implemented".to_string(), json!(true));
    metrics.insert("executor".to_string(), json!("examples"));
    metrics.insert("command".to_string(), json!(command_display));
    metrics.insert("ar_baseline".to_string(), json!(ar_baseline));
    metrics.insert(
        "profiling_requested".to_string(),
        json!(config.profile == ProfileMode::Passive && !ar_baseline),
    );
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
    if let Some(v) = metrics.get("decode_tok_s").cloned() {
        metrics.entry("tok_s".to_string()).or_insert(v);
    }
    if let Some(v) = metrics.get("decode_tau").cloned() {
        metrics.entry("tau".to_string()).or_insert(v);
    }
    if let Some(v) = metrics.get("decode_accept_rate").cloned() {
        metrics.entry("accept_rate".to_string()).or_insert(v);
    }
    if let Some(attractor) = parse_spec_decode_token_attractor_metrics(&stderr) {
        metrics.extend(attractor.metrics);
    }
    let attractor_fail = metrics
        .get("token_attractor_fail")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    if output.status.success() && metrics.contains_key("decode_tok_s") && !attractor_fail {
        row_for_model(
            battery,
            None,
            case_id,
            None,
            EvalStatus::Pass,
            None,
            metrics,
            config,
            ctx,
            prompt_ref,
            elapsed_ms,
            model,
        )
    } else {
        row_for_model(
            battery,
            None,
            case_id,
            None,
            EvalStatus::Fail,
            Some(if output.status.success() {
                if attractor_fail {
                    "dflash_spec_demo token-attractor detector failed".to_string()
                } else {
                    "dflash_spec_demo did not emit BENCH METRICS".to_string()
                }
            } else {
                format!("dflash_spec_demo exited with {}", output.status)
            }),
            metrics,
            config,
            ctx,
            prompt_ref,
            elapsed_ms,
            model,
        )
    }
}

struct TokenAttractorMetrics {
    metrics: BTreeMap<String, Value>,
}

fn parse_spec_decode_token_attractor_metrics(output: &str) -> Option<TokenAttractorMetrics> {
    let tokens = parse_spec_decode_tokens(output)?;
    if tokens.is_empty() {
        let metrics = BTreeMap::from([
            ("token_count".to_string(), json!(0.0)),
            ("token_attractor_fail".to_string(), json!(true)),
            ("token_attractor_reason".to_string(), json!("zero_tokens")),
        ]);
        return Some(TokenAttractorMetrics { metrics });
    }
    let eot = [248044u32, 248046u32];
    let trimmed: Vec<u32> = tokens
        .iter()
        .copied()
        .take_while(|token| !eot.contains(token))
        .take(128)
        .collect();
    if trimmed.len() < 16 {
        let metrics = BTreeMap::from([
            ("token_count".to_string(), json!(trimmed.len() as f64)),
            ("token_attractor_fail".to_string(), json!(false)),
            ("token_attractor_reason".to_string(), json!("short_clean")),
        ]);
        return Some(TokenAttractorMetrics { metrics });
    }
    let mut counts: BTreeMap<u32, usize> = BTreeMap::new();
    for token in &trimmed {
        *counts.entry(*token).or_insert(0) += 1;
    }
    let max_freq = counts.values().copied().max().unwrap_or(0) as f64 / trimmed.len() as f64;
    let unique_ratio = counts.len() as f64 / trimmed.len() as f64;
    let fail = max_freq > 0.40 || unique_ratio < 0.30;
    let metrics = BTreeMap::from([
        ("token_count".to_string(), json!(trimmed.len() as f64)),
        ("unique_token_count".to_string(), json!(counts.len() as f64)),
        ("max_token_frequency".to_string(), json!(max_freq)),
        ("unique_token_ratio".to_string(), json!(unique_ratio)),
        ("token_attractor_fail".to_string(), json!(fail)),
        (
            "token_attractor_reason".to_string(),
            json!(if fail { "attractor" } else { "ok" }),
        ),
    ]);
    Some(TokenAttractorMetrics { metrics })
}

fn parse_spec_decode_tokens(output: &str) -> Option<Vec<u32>> {
    for prefix in ["DFlash tokens:", "AR tokens:"] {
        let Some(start) = output.find(prefix) else {
            continue;
        };
        let rest = &output[start + prefix.len()..];
        let open = rest.find('[')?;
        let close = rest[open + 1..].find(']')? + open + 1;
        let body = &rest[open + 1..close];
        let tokens = body
            .split(',')
            .filter_map(|part| part.trim().parse::<u32>().ok())
            .collect::<Vec<_>>();
        return Some(tokens);
    }
    None
}

fn run_examples_run_anchor_with_prompt(
    battery: BatteryId,
    case_id: &str,
    prompt_path: &str,
    config: &EvalConfig,
    ctx: &EvalContext,
) -> EvalResult {
    run_examples_run_anchor_with_prompt_for_model(
        battery,
        case_id,
        prompt_path,
        config,
        ctx,
        config.model.clone(),
    )
}

fn run_examples_run_anchor_with_prompt_for_model(
    battery: BatteryId,
    case_id: &str,
    prompt_path: &str,
    config: &EvalConfig,
    ctx: &EvalContext,
    model: String,
) -> EvalResult {
    let prompt_ref = prompt(prompt_path);
    run_examples_run_anchor_with_prompt_ref_for_model(
        battery,
        case_id,
        prompt_path,
        prompt_ref,
        config,
        ctx,
        model,
        None,
        BTreeMap::new(),
    )
}

#[allow(clippy::too_many_arguments)]
fn run_examples_run_anchor_with_prompt_ref_for_model(
    battery: BatteryId,
    case_id: &str,
    prompt_path: &str,
    prompt_ref: Option<PromptRef>,
    config: &EvalConfig,
    ctx: &EvalContext,
    model: String,
    max_seq: Option<usize>,
    extra_metrics: BTreeMap<String, Value>,
) -> EvalResult {
    if !Path::new(&model).exists() {
        let mut metrics = BTreeMap::from([("executor".to_string(), json!("examples"))]);
        metrics.extend(extra_metrics);
        return row_for_model(
            battery,
            None,
            case_id,
            None,
            EvalStatus::Skip,
            Some(
                "examples executor requires each evaluated model to be a local filesystem path"
                    .to_string(),
            ),
            metrics,
            config,
            ctx,
            prompt_ref,
            0,
            model,
        );
    }
    let Some(bin) = resolve_run_example_bin() else {
        let mut metrics = BTreeMap::from([("executor".to_string(), json!("examples"))]);
        metrics.extend(extra_metrics);
        return row_for_model(
            battery,
            None,
            case_id,
            None,
            EvalStatus::Skip,
            Some("run example binary not found; build with `cargo build --release --features deltanet -p hipfire-runtime --example run`".to_string()),
            metrics,
            config,
            ctx,
            prompt_ref,
            0,
            model,
        );
    };
    let evidence_dir = runtime_evidence_dir(config, case_id, &model);
    let mut args = vec![
        model.clone(),
        "--prompt-file".to_string(),
        prompt_path.to_string(),
        "--max-tokens".to_string(),
        config.max_tokens.to_string(),
        "--kv".to_string(),
        config.kv_mode.clone().unwrap_or_else(|| "q8".to_string()),
        "--temp".to_string(),
        "0.0".to_string(),
    ];
    add_runtime_evidence_arg(&mut args, &evidence_dir);
    let computed_max_seq = max_seq.unwrap_or_else(|| {
        if config.max_tokens + 2048 > 4096 {
            config.max_tokens + 2048
        } else {
            4096
        }
    });
    if computed_max_seq > 4096 {
        args.push("--max-seq".to_string());
        args.push(computed_max_seq.to_string());
    }
    let command_display = format!("{} {}", bin.display(), args.join(" "));
    let started = SystemTime::now();
    let output = match Command::new(&bin).args(&args).output() {
        Ok(output) => output,
        Err(err) => {
            let mut metrics = BTreeMap::from([
                ("executor".to_string(), json!("examples")),
                ("command".to_string(), json!(command_display)),
            ]);
            metrics.extend(extra_metrics);
            return row_for_model(
                battery,
                None,
                case_id,
                None,
                EvalStatus::Fail,
                Some(format!("spawn run example: {err}")),
                metrics,
                config,
                ctx,
                prompt_ref,
                elapsed_since_ms(started),
                model,
            );
        }
    };
    let elapsed_ms = elapsed_since_ms(started);
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    let mut metrics = parse_bench_metrics(&stderr);
    metrics.insert("implemented".to_string(), json!(true));
    metrics.insert("executor".to_string(), json!("examples"));
    metrics.insert("command".to_string(), json!(command_display));
    metrics.insert("ar_baseline".to_string(), json!(true));
    metrics.insert("max_seq".to_string(), json!(computed_max_seq));
    metrics.extend(extra_metrics);
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
    if let Some(v) = metrics.get("decode_tok_s").cloned() {
        metrics.entry("tok_s".to_string()).or_insert(v);
    }
    if let Some(v) = metrics.get("decode_tau").cloned() {
        metrics.entry("tau".to_string()).or_insert(v);
    }
    if let Some(v) = metrics.get("decode_accept_rate").cloned() {
        metrics.entry("accept_rate".to_string()).or_insert(v);
    }
    if output.status.success() && metrics.contains_key("decode_tok_s") {
        row_for_model(
            battery,
            None,
            case_id,
            None,
            EvalStatus::Pass,
            None,
            metrics,
            config,
            ctx,
            prompt_ref,
            elapsed_ms,
            model,
        )
    } else {
        row_for_model(
            battery,
            None,
            case_id,
            None,
            EvalStatus::Fail,
            Some(if output.status.success() {
                "run example did not emit BENCH METRICS".to_string()
            } else {
                format!("run example exited with {}", output.status)
            }),
            metrics,
            config,
            ctx,
            prompt_ref,
            elapsed_ms,
            model,
        )
    }
}

fn run_direct_session_reset_recall(config: &EvalConfig, ctx: &EvalContext) -> EvalResult {
    let case_id = "multi_turn_reset_recall";
    let prompt_path = "benchmarks/prompts/trains-meet.txt";
    let prompt_ref = prompt(prompt_path);
    let model = config.model.clone();
    let base_metrics = || BTreeMap::from([("executor".to_string(), json!("direct"))]);

    if !Path::new(&model).exists() {
        return row_for_model(
            BatteryId::Smoke,
            None,
            case_id,
            None,
            EvalStatus::Skip,
            Some(
                "direct session executor requires --model to be a local filesystem path"
                    .to_string(),
            ),
            base_metrics(),
            config,
            ctx,
            prompt_ref,
            0,
            model,
        );
    }
    let Some(bin) = resolve_run_example_bin() else {
        return row_for_model(
            BatteryId::Smoke,
            None,
            case_id,
            None,
            EvalStatus::Skip,
            Some("run example binary not found; build with `cargo build --release --features deltanet -p hipfire-runtime --example run`".to_string()),
            base_metrics(),
            config,
            ctx,
            prompt_ref,
            0,
            model,
        );
    };

    let evidence_dir = runtime_evidence_dir(config, case_id, &model);
    let mut args = vec![
        model.clone(),
        "--session-reset-smoke".to_string(),
        "--prompt-file".to_string(),
        prompt_path.to_string(),
        "--kv".to_string(),
        config.kv_mode.clone().unwrap_or_else(|| "q8".to_string()),
        "--temp".to_string(),
        "0.0".to_string(),
    ];
    add_runtime_evidence_arg(&mut args, &evidence_dir);
    let command_display = format!("{} {}", bin.display(), args.join(" "));
    let started = SystemTime::now();
    let output = match Command::new(&bin).args(&args).output() {
        Ok(output) => output,
        Err(err) => {
            let mut metrics = base_metrics();
            metrics.insert("command".to_string(), json!(command_display));
            return row_for_model(
                BatteryId::Smoke,
                None,
                case_id,
                None,
                EvalStatus::Fail,
                Some(format!("spawn direct session executor: {err}")),
                metrics,
                config,
                ctx,
                prompt_ref,
                elapsed_since_ms(started),
                model,
            );
        }
    };
    let elapsed_ms = elapsed_since_ms(started);
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    let report_path = evidence_dir.join("session_reset.json");
    let report = fs::read_to_string(&report_path)
        .ok()
        .and_then(|body| serde_json::from_str::<Value>(&body).ok())
        .or_else(|| serde_json::from_str::<Value>(&stdout).ok());

    let mut metrics = base_metrics();
    metrics.insert("implemented".to_string(), json!(true));
    metrics.insert("command".to_string(), json!(command_display));
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
    if let Some(value) = report.as_ref().and_then(|v| v.get("metrics")) {
        if let Some(obj) = value.as_object() {
            for (key, value) in obj {
                metrics.insert(key.clone(), value.clone());
            }
        }
    }

    let report_status = report
        .as_ref()
        .and_then(|v| v.get("status"))
        .and_then(Value::as_str);
    let pass = output.status.success() && report_status == Some("pass");
    row_for_model(
        BatteryId::Smoke,
        None,
        case_id,
        None,
        if pass {
            EvalStatus::Pass
        } else {
            EvalStatus::Fail
        },
        if pass {
            None
        } else if output.status.success() {
            Some("direct session executor did not emit a passing session reset report".to_string())
        } else {
            Some(format!(
                "direct session executor exited with {}",
                output.status
            ))
        },
        metrics,
        config,
        ctx,
        prompt_ref,
        elapsed_ms,
        model,
    )
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

fn resolve_eval_hipfire_bin() -> Option<PathBuf> {
    if let Ok(path) = std::env::var("HIPFIRE_EVAL_HIPFIRE_BIN") {
        let p = PathBuf::from(path);
        if p.exists() {
            return Some(p);
        }
    }
    let exe = std::env::consts::EXE_SUFFIX;
    let repo = repo_root()?;
    newest_existing_path([
        repo.join(format!("target/release/examples/eval_hipfire{exe}")),
        repo.join(format!("target/debug/examples/eval_hipfire{exe}")),
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

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ResultCacheEntry {
    schema: u32,
    key: String,
    created_utc: String,
    cache_mode: EvalCacheMode,
    rows: Vec<EvalResult>,
}

fn run_eval_batteries(
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

fn daemon_shared_model_load_enabled(config: &EvalConfig) -> bool {
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

fn daemon_shared_model_load_battery(battery: BatteryId) -> bool {
    matches!(
        battery,
        BatteryId::Smoke | BatteryId::Speed | BatteryId::Profile
    )
}

fn daemon_executor_available_for(config: &EvalConfig, battery: BatteryId) -> bool {
    if !matches!(
        battery,
        BatteryId::Smoke | BatteryId::Speed | BatteryId::Profile
    ) {
        return false;
    }
    if !Path::new(&config.model).exists() {
        return true;
    }
    hipfire_daemon_adapter::find_daemon_bin().is_some()
}

fn run_battery_cached(
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

fn mark_benchmark_sample(row: &mut EvalResult, run_index: usize, config: &EvalConfig) {
    row.metrics
        .insert("benchmark_sample".to_string(), json!(true));
    row.metrics
        .insert("run_index".to_string(), json!(run_index));
    row.metrics
        .insert("run_count".to_string(), json!(config.runs));
    row.metrics
        .insert("warmup_runs".to_string(), json!(config.warmup_runs));
}

fn benchmark_aggregate_rows(rows: &[EvalResult], config: &EvalConfig) -> Vec<EvalResult> {
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

fn benchmark_group_key(row: &EvalResult) -> String {
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

fn benchmark_aggregate_metrics(
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

fn benchmark_metric_excluded(key: &str) -> bool {
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
struct F64Summary {
    mean: f64,
    median: f64,
    stddev: f64,
    min: f64,
    max: f64,
    p10: f64,
    p90: f64,
    cv_pct: Option<f64>,
}

fn summarize_f64_samples(sorted: &[f64]) -> F64Summary {
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

fn percentile(sorted: &[f64], p: f64) -> f64 {
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

fn result_cache_key(
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

fn result_cache_prompt_fingerprints(battery: BatteryId) -> Vec<Value> {
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

fn result_cache_prompt_paths(battery: BatteryId) -> Vec<&'static str> {
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
        BatteryId::Barrage | BatteryId::Vision | BatteryId::Cask => Vec::new(),
    }
}

fn result_cache_path(config: &EvalConfig, key: &str) -> PathBuf {
    config
        .result_cache
        .join(&key[..2])
        .join(format!("{key}.json"))
}

fn read_result_cache_entry(path: &Path, key: &str) -> Option<Vec<EvalResult>> {
    let body = fs::read_to_string(path).ok()?;
    let entry: ResultCacheEntry = serde_json::from_str(&body).ok()?;
    if entry.schema == 1 && entry.key == key {
        Some(entry.rows)
    } else {
        None
    }
}

fn write_result_cache_entry(
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

fn mark_cache_hit(mut row: EvalResult, key: &str, path: &Path) -> EvalResult {
    row.metrics.insert("cache_hit".to_string(), json!(true));
    row.metrics
        .insert("cache_key".to_string(), json!(key.to_string()));
    row.metrics
        .insert("cache_path".to_string(), json!(path.display().to_string()));
    row
}

fn run_battery(
    battery: BatteryId,
    config: &EvalConfig,
    ctx: &EvalContext,
    datasets: &[DatasetManifestEntry],
) -> Vec<EvalResult> {
    if battery == BatteryId::Quality {
        if let Some(rows) = quality_json_rows(config, ctx) {
            return rows;
        }
        if config.executor != EvalExecutorMode::Mock {
            if let Some(rows) = kld_reference_rows(config, ctx) {
                return rows;
            }
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

fn pass_row(
    battery: BatteryId,
    suite: Option<SuiteId>,
    case_id: &str,
    dataset_item_id: Option<String>,
    config: &EvalConfig,
    ctx: &EvalContext,
    prompt: Option<PromptRef>,
    mut metrics: BTreeMap<String, Value>,
) -> EvalResult {
    metrics.insert("implemented".to_string(), json!(true));
    row(
        battery,
        suite,
        case_id,
        dataset_item_id,
        EvalStatus::Pass,
        None,
        metrics,
        config,
        ctx,
        prompt,
        0,
    )
}

fn skip_row(
    battery: BatteryId,
    suite: Option<SuiteId>,
    case_id: &str,
    dataset_item_id: Option<String>,
    reason: &str,
    config: &EvalConfig,
    ctx: &EvalContext,
    prompt: Option<PromptRef>,
) -> EvalResult {
    row(
        battery,
        suite,
        case_id,
        dataset_item_id,
        EvalStatus::Skip,
        Some(reason.to_string()),
        BTreeMap::new(),
        config,
        ctx,
        prompt,
        0,
    )
}

fn skip_row_with_metrics(
    battery: BatteryId,
    suite: Option<SuiteId>,
    case_id: &str,
    dataset_item_id: Option<String>,
    reason: &str,
    config: &EvalConfig,
    ctx: &EvalContext,
    prompt: Option<PromptRef>,
    metrics: BTreeMap<String, Value>,
) -> EvalResult {
    row(
        battery,
        suite,
        case_id,
        dataset_item_id,
        EvalStatus::Skip,
        Some(reason.to_string()),
        metrics,
        config,
        ctx,
        prompt,
        0,
    )
}

#[allow(clippy::too_many_arguments)]
fn row(
    battery: BatteryId,
    suite: Option<SuiteId>,
    case_id: &str,
    dataset_item_id: Option<String>,
    status: EvalStatus,
    reason: Option<String>,
    metrics: BTreeMap<String, Value>,
    config: &EvalConfig,
    ctx: &EvalContext,
    prompt: Option<PromptRef>,
    elapsed_ms: u128,
) -> EvalResult {
    row_for_model(
        battery,
        suite,
        case_id,
        dataset_item_id,
        status,
        reason,
        metrics,
        config,
        ctx,
        prompt,
        elapsed_ms,
        config.model.clone(),
    )
}

#[allow(clippy::too_many_arguments)]
fn row_for_model(
    battery: BatteryId,
    suite: Option<SuiteId>,
    case_id: &str,
    dataset_item_id: Option<String>,
    status: EvalStatus,
    reason: Option<String>,
    metrics: BTreeMap<String, Value>,
    config: &EvalConfig,
    ctx: &EvalContext,
    prompt: Option<PromptRef>,
    elapsed_ms: u128,
    model: String,
) -> EvalResult {
    EvalResult {
        schema: 2,
        battery,
        suite,
        case_id: case_id.to_string(),
        dataset_item_id,
        dataset_source: metric_string(&metrics, "dataset_source"),
        dataset_repo_id: metric_string(&metrics, "dataset_repo_id"),
        dataset_revision: metric_string(&metrics, "dataset_revision"),
        dataset_digest: metric_string(&metrics, "dataset_digest"),
        dataset_license: metric_string(&metrics, "dataset_license"),
        dataset_cache_path: metric_string(&metrics, "dataset_cache_path"),
        status,
        reason,
        metrics,
        prompt_hash: prompt.as_ref().map(|p| p.hash.clone()),
        prompt_path: prompt.map(|p| p.path),
        model_hash: model_hash(&model),
        model,
        draft: config.draft.clone(),
        baseline: config.baseline.clone(),
        reference: config.reference.clone(),
        draft_hash: config.draft.as_deref().and_then(model_hash),
        baseline_hash: config.baseline.as_deref().and_then(model_hash),
        reference_hash: config.reference.as_deref().and_then(model_hash),
        hipfire_version: env!("CARGO_PKG_VERSION").to_string(),
        git_commit: ctx.commit_sha.clone(),
        commit_sha: ctx.commit_sha.clone(),
        git_branch: ctx.git_branch.clone(),
        git_describe: ctx.git_describe.clone(),
        git_dirty: ctx.git_dirty,
        binary_hash: ctx.binary_hash.clone(),
        arch: ctx.arch.clone(),
        rocm: ctx.rocm.clone(),
        host_profile_hash: ctx.host_profile.host_profile_hash.clone(),
        hardware_bucket: ctx.host_profile.hardware_bucket.clone(),
        kv_mode: config.kv_mode.clone(),
        started_utc: utc_now(),
        elapsed_ms,
    }
}

#[derive(Clone)]
struct PromptRef {
    path: String,
    hash: String,
}

impl PromptRef {
    fn from_content(path: String, content: &[u8]) -> Self {
        Self {
            path,
            hash: stable_hash_bytes(content),
        }
    }
}

fn prompt(path: &str) -> Option<PromptRef> {
    let p = Path::new(path);
    let owned;
    let p = if p.exists() {
        p
    } else {
        owned = repo_root()?.join(path);
        if !owned.exists() {
            return None;
        }
        &owned
    };
    Some(PromptRef {
        path: path.to_string(),
        hash: file_hash(p).unwrap_or_else(|| stable_hash_file_fallback(p)),
    })
}

fn combined_prompt_ref(system_path: &str, prompt_path: &str) -> Option<PromptRef> {
    let system = fs::read(resolve_repo_path(system_path)?).ok()?;
    let prompt = fs::read(resolve_repo_path(prompt_path)?).ok()?;
    let mut content = Vec::with_capacity(system.len() + prompt.len() + 5);
    content.extend_from_slice(&system);
    content.extend_from_slice(b"\n---\n");
    content.extend_from_slice(&prompt);
    Some(PromptRef::from_content(
        format!("{system_path}+{prompt_path}"),
        &content,
    ))
}

fn structured_tools_prompt_ref(
    system_path: &str,
    prompt_path: &str,
    tools: &Value,
) -> Option<PromptRef> {
    let system = fs::read(resolve_repo_path(system_path)?).ok()?;
    let prompt = fs::read(resolve_repo_path(prompt_path)?).ok()?;
    let tools = serde_json::to_vec(tools).ok()?;
    let mut content = Vec::with_capacity(system.len() + prompt.len() + tools.len() + 16);
    content.extend_from_slice(&system);
    content.extend_from_slice(b"\n---prompt---\n");
    content.extend_from_slice(&prompt);
    content.extend_from_slice(b"\n---tools---\n");
    content.extend_from_slice(&tools);
    Some(PromptRef::from_content(
        format!("{system_path}+{prompt_path}+structured_tools"),
        &content,
    ))
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

fn detect_arch() -> Option<String> {
    for node in ["1", "0"] {
        let path = format!("/sys/class/kfd/kfd/topology/nodes/{node}/properties");
        let raw = match fs::read_to_string(path) {
            Ok(raw) => raw,
            Err(_) => continue,
        };
        for line in raw.lines() {
            if let Some(v) = line.strip_prefix("gfx_target_version") {
                if let Ok(ver) = v.trim().parse() {
                    return Some(gfx_target_version_to_arch(ver));
                }
            }
        }
    }
    None
}

fn gfx_target_version_to_arch(ver: u32) -> String {
    match ver {
        100100 => "gfx1010".to_string(),
        100300 | 100302 => "gfx1030".to_string(),
        110000 | 110001 => "gfx1100".to_string(),
        110501 => "gfx1151".to_string(),
        120000 => "gfx1200".to_string(),
        120001 => "gfx1201".to_string(),
        _ => {
            let major = ver / 10000;
            let minor = (ver % 10000) / 100;
            let step = ver % 100;
            format!("gfx{major}{minor}{step}")
        }
    }
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

pub fn run_from_env() -> Result<(), String> {
    let args: Vec<String> = std::env::args().collect();
    if args.iter().any(|a| a == "-h" || a == "--help") {
        eprint!("{}", usage());
        return Ok(());
    }
    if args.iter().any(|a| a == "--version") {
        print!("{}", version_report());
        return Ok(());
    }
    let config = parse_args_from(args)?;

    if config.status {
        print_eval_status(&config);
        return Ok(());
    }
    if config.fetch {
        return run_fetch(&config);
    }
    if let Some(spec) = config.models_spec.clone() {
        return run_sweep(&config, &spec);
    }
    if config.dry_run {
        let (ctx, datasets) = dry_run_inputs(&config)?;
        let plan = plan_model(&config, &ctx, &datasets)?;
        print_plans(std::slice::from_ref(&plan));
        return Ok(());
    }
    run_eval(config)
}

/// Expand a `--models` spec (comma-separated globs / prefixes / paths) against
/// the model directory (HIPFIRE_MODELS_DIR or ~/.hipfire/models). A token with
/// `/` or an existing `.hfq` path is taken literally; otherwise it matches model
/// filenames by simple `*` glob, falling back to substring match when it has no
/// `*` (so `qwen3.5` matches every `*qwen3.5*.hfq`).
fn expand_models(spec: &str) -> Result<Vec<String>, String> {
    let models_dir = std::env::var_os("HIPFIRE_MODELS_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            home_dir()
                .unwrap_or_else(|| PathBuf::from("."))
                .join(".hipfire")
                .join("models")
        });
    let entries: Vec<(String, PathBuf)> = std::fs::read_dir(&models_dir)
        .map_err(|e| format!("read models dir {}: {e}", models_dir.display()))?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|x| x == "hfq"))
        .filter_map(|p| {
            p.file_name()
                .and_then(|n| n.to_str())
                .map(|n| (n.to_string(), p.clone()))
        })
        .collect();

    let mut out: Vec<String> = Vec::new();
    for tok in spec.split(',').map(str::trim).filter(|s| !s.is_empty()) {
        if tok.contains('/') || (tok.ends_with(".hfq") && Path::new(tok).exists()) {
            out.push(tok.to_string());
            continue;
        }
        let mut hit = false;
        for (name, path) in &entries {
            let stem = name.trim_end_matches(".hfq");
            let matched = if tok.contains('*') {
                glob_match(tok, name) || glob_match(tok, stem)
            } else {
                name.contains(tok)
            };
            if matched {
                out.push(path.display().to_string());
                hit = true;
            }
        }
        if !hit {
            eprintln!(
                "[eval] --models: no match for '{tok}' in {}",
                models_dir.display()
            );
        }
    }
    out.sort();
    out.dedup();
    if out.is_empty() {
        return Err(format!(
            "--models '{spec}' matched no .hfq files in {}",
            models_dir.display()
        ));
    }
    Ok(out)
}

/// Minimal `*`-glob (any-run wildcard; no `?`/classes), enough for model specs.
fn glob_match(pattern: &str, text: &str) -> bool {
    let parts: Vec<&str> = pattern.split('*').collect();
    if parts.len() == 1 {
        return pattern == text;
    }
    let mut pos = 0usize;
    // First segment must be a prefix (unless pattern starts with '*').
    if let Some(first) = parts.first() {
        if !text[pos..].starts_with(first) {
            return false;
        }
        pos += first.len();
    }
    for (idx, seg) in parts.iter().enumerate().skip(1) {
        if seg.is_empty() {
            continue;
        }
        if idx == parts.len() - 1 && !pattern.ends_with('*') {
            // Last segment must match the suffix.
            return text[pos..].ends_with(seg);
        }
        match text[pos..].find(seg) {
            Some(off) => pos += off + seg.len(),
            None => return false,
        }
    }
    true
}

// ── Sweep / dry-run / status / fetch machinery ───────────────────────────────

struct CellPlan {
    battery: BatteryId,
    state: &'static str, // CACHED | READY | BLOCKED
    reason: Option<String>,
}

struct ModelPlan {
    model: String,
    cells: Vec<CellPlan>,
}

fn eval_context(config: &EvalConfig) -> EvalContext {
    EvalContext::new_with_overrides(HostProfileOverrides {
        memory_class: config.host_memory_class.clone(),
        memory_width_bits: config.host_memory_width_bits,
        memory_bandwidth_gbps: config.host_memory_bandwidth_gbps,
    })
}

/// Artifact that would BLOCK a battery from running (the cases the user cares
/// about: missing model / corpus / dataset). `None` ⇒ would run. Deliberately
/// does not try to predict executor-binary availability (too coupled); those
/// surface as normal skips at run time.
fn cell_block_reason(
    battery: BatteryId,
    config: &EvalConfig,
    datasets: &[DatasetManifestEntry],
) -> Option<String> {
    if !config.model.is_empty() && !Path::new(&config.model).exists() {
        return Some(format!("model not found: {}", config.model));
    }
    match battery {
        BatteryId::Perplexity => {
            let corpus_rel = std::env::var("HIPFIRE_EVAL_PERPLEXITY_CORPUS").unwrap_or_else(|_| {
                "benchmarks/quality-baselines/slice/wikitext2-1024s-2048ctx.txt".into()
            });
            let corpus = repo_root()
                .map(|r| r.join(&corpus_rel))
                .unwrap_or_else(|| PathBuf::from(&corpus_rel));
            if !corpus.exists() {
                return Some(format!(
                    "corpus missing ({}) — set HIPFIRE_EVAL_PERPLEXITY_CORPUS or stage under ~/.hipfire/datasets/",
                    corpus.display()
                ));
            }
            None
        }
        BatteryId::Barrage => datasets
            .iter()
            .find(|d| d.status == EvalStatus::Skip)
            .map(|d| {
                format!(
                    "dataset {} unavailable ({}) — run `hipfire eval --fetch`",
                    d.suite.as_str(),
                    d.reason.as_deref().unwrap_or("not cached")
                )
            }),
        _ => None,
    }
}

/// `ctx` (hardware/build) and `datasets` (shared HF datasets) are
/// model-independent — the caller builds them ONCE and passes them in, so a
/// many-model dry-run does one host-profiling pass, not one per SKU.
fn plan_model(
    config: &EvalConfig,
    ctx: &EvalContext,
    datasets: &[DatasetManifestEntry],
) -> Result<ModelPlan, String> {
    let mut cells = Vec::new();
    for &battery in &config.batteries {
        let key = result_cache_key(battery, config, ctx, datasets)?;
        let path = result_cache_path(config, &key);
        let cached = config.cache_mode == EvalCacheMode::Use && path.exists();
        let (state, reason) = if cached {
            ("CACHED", None)
        } else if let Some(r) = cell_block_reason(battery, config, datasets) {
            ("BLOCKED", Some(r))
        } else {
            ("READY", None)
        };
        cells.push(CellPlan {
            battery,
            state,
            reason,
        });
    }
    Ok(ModelPlan {
        model: config.model.clone(),
        cells,
    })
}

/// Build the shared dry-run context + dataset manifest once (no network:
/// datasets are resolved with fetch disabled).
fn dry_run_inputs(config: &EvalConfig) -> Result<(EvalContext, Vec<DatasetManifestEntry>), String> {
    let ctx = eval_context(config);
    let mut probe = config.clone();
    probe.fetch_datasets = false;
    let datasets = resolve_datasets(&probe)?;
    Ok((ctx, datasets))
}

fn print_plans(plans: &[ModelPlan]) {
    let (mut cached, mut ready, mut blocked) = (0u32, 0u32, 0u32);
    let mut missing: std::collections::BTreeMap<String, u32> = std::collections::BTreeMap::new();
    println!("=== dry-run plan (CACHED = skip, READY = would run, BLOCKED = missing artifact) ===");
    for p in plans {
        println!("\n{}", model_artifact_stem(&p.model));
        for c in &p.cells {
            match c.state {
                "CACHED" => cached += 1,
                "READY" => ready += 1,
                _ => {
                    blocked += 1;
                    if let Some(r) = &c.reason {
                        *missing.entry(r.clone()).or_insert(0) += 1;
                    }
                }
            }
            match &c.reason {
                Some(r) => println!("  {:<8} {:<14} {}", c.state, c.battery.as_str(), r),
                None => println!("  {:<8} {}", c.state, c.battery.as_str()),
            }
        }
    }
    println!("\n=== totals: {cached} cached (skip), {ready} would run, {blocked} blocked ===");
    if !missing.is_empty() {
        println!("--- missing artifacts (resolve, then re-run) ---");
        for (reason, n) in &missing {
            println!("  [{n}x] {reason}");
        }
    }
}

fn sweep_base_dir(config: &EvalConfig) -> PathBuf {
    let ts = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    home_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join(".hipfire")
        .join("eval-results")
        .join("sweeps")
        .join(format!("{ts}-{}", config.tier.as_str()))
}

fn run_sweep(config: &EvalConfig, spec: &str) -> Result<(), String> {
    let models = expand_models(spec)?;
    let base = sweep_base_dir(config);
    fs::create_dir_all(&base).map_err(|e| format!("create {}: {e}", base.display()))?;
    eprintln!(
        "[eval] sweep: {} model(s){} -> {}",
        models.len(),
        if config.dry_run { " (dry-run)" } else { "" },
        base.display()
    );

    // Hardware/build ctx + shared dataset manifest are model-independent: build
    // ONCE for the whole dry-run (one host-profiling pass, not one per SKU).
    let dry_inputs = if config.dry_run {
        Some(dry_run_inputs(config)?)
    } else {
        None
    };

    let mut plans = Vec::new();
    let mut outcomes: Vec<(String, PathBuf, String)> = Vec::new();
    for m in &models {
        let mut cfg = config.clone();
        cfg.model = m.clone();
        cfg.models_spec = None;
        cfg.out_dir = base.join(model_artifact_stem(m));
        if cfg.draft.is_none() && matches!(cfg.dflash, DflashMode::Auto | DflashMode::On) {
            cfg.draft =
                discover_dflash_draft_for_model(Path::new(m)).map(|p| p.display().to_string());
        }
        if let Some((ctx, datasets)) = &dry_inputs {
            match plan_model(&cfg, ctx, datasets) {
                Ok(p) => plans.push(p),
                Err(e) => eprintln!("[eval] plan {m}: {e}"),
            }
            continue;
        }
        eprintln!("[eval] === {} ===", model_artifact_stem(m));
        // Fail-isolation: a model that errors or panics (e.g. OOM) is recorded
        // and the sweep continues to the next SKU.
        let run_cfg = cfg.clone();
        let outcome =
            match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| run_eval(run_cfg))) {
                Ok(Ok(())) => "ok".to_string(),
                Ok(Err(e)) => {
                    eprintln!("[eval] {m}: FAILED: {e}");
                    format!("failed: {e}")
                }
                Err(_) => {
                    eprintln!("[eval] {m}: PANIC — skipped, continuing sweep");
                    "panic".to_string()
                }
            };
        outcomes.push((m.clone(), cfg.out_dir.clone(), outcome));
    }

    if config.dry_run {
        print_plans(&plans);
        return Ok(());
    }
    write_sweep_rollup(&base, &outcomes)?;
    Ok(())
}

/// Aggregate each model's results.jsonl into a cross-model rollup (CSV + md).
fn write_sweep_rollup(base: &Path, outcomes: &[(String, PathBuf, String)]) -> Result<(), String> {
    let mut csv = String::from("model,outcome,pass,skip,fail,ppl,gen_tok_s\n");
    let mut md = String::from("| model | outcome | pass | skip | fail | ppl | gen tok/s |\n|---|---|---|---|---|---|---|\n");
    for (model, out_dir, outcome) in outcomes {
        let (mut pass, mut skip, mut fail) = (0u32, 0u32, 0u32);
        let (mut ppl, mut tok_s) = (String::new(), String::new());
        if let Ok(text) = std::fs::read_to_string(out_dir.join("results.jsonl")) {
            for line in text.lines() {
                let Ok(v) = serde_json::from_str::<Value>(line) else {
                    continue;
                };
                match v.get("status").and_then(|s| s.as_str()) {
                    Some("pass") => pass += 1,
                    Some("skip") => skip += 1,
                    Some("fail") => fail += 1,
                    _ => {}
                }
                let metrics = v.get("metrics");
                if ppl.is_empty() {
                    if let Some(p) = metrics.and_then(|m| m.get("ppl")).and_then(|p| p.as_f64()) {
                        ppl = format!("{p:.4}");
                    }
                }
                if tok_s.is_empty() {
                    if let Some(t) = metrics
                        .and_then(|m| m.get("gen_tok_s").or_else(|| m.get("tok_s")))
                        .and_then(|t| t.as_f64())
                    {
                        tok_s = format!("{t:.1}");
                    }
                }
            }
        }
        let stem = model_artifact_stem(model);
        csv.push_str(&format!(
            "{stem},{outcome},{pass},{skip},{fail},{ppl},{tok_s}\n"
        ));
        md.push_str(&format!(
            "| {stem} | {outcome} | {pass} | {skip} | {fail} | {ppl} | {tok_s} |\n"
        ));
    }
    let csv_path = base.join("rollup.csv");
    let md_path = base.join("rollup.md");
    fs::write(&csv_path, &csv).map_err(|e| format!("write {}: {e}", csv_path.display()))?;
    fs::write(&md_path, &md).map_err(|e| format!("write {}: {e}", md_path.display()))?;
    println!("\n{md}");
    println!("rollup: {}", csv_path.display());
    Ok(())
}

fn print_eval_status(config: &EvalConfig) {
    let ctx = eval_context(config);
    let cache_entries = std::fs::read_dir(&config.result_cache)
        .map(|rd| {
            rd.filter_map(|e| e.ok())
                .flat_map(|e| std::fs::read_dir(e.path()).into_iter().flatten())
                .filter_map(|e| e.ok())
                .filter(|e| e.path().extension().is_some_and(|x| x == "json"))
                .count()
        })
        .unwrap_or(0);
    let models_dir = std::env::var_os("HIPFIRE_MODELS_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            home_dir()
                .unwrap_or_else(|| PathBuf::from("."))
                .join(".hipfire")
                .join("models")
        });
    let model_count = std::fs::read_dir(&models_dir)
        .map(|rd| {
            rd.filter_map(|e| e.ok())
                .filter(|e| e.path().extension().is_some_and(|x| x == "hfq"))
                .count()
        })
        .unwrap_or(0);
    println!("hipfire-eval status");
    println!("  hardware_bucket: {}", ctx.host_profile.hardware_bucket);
    println!(
        "  arch / rocm:     {} / {}",
        ctx.arch.as_deref().unwrap_or("?"),
        ctx.rocm.as_deref().unwrap_or("?")
    );
    println!(
        "  binary / commit: {} / {}{}",
        ctx.binary_hash.as_deref().unwrap_or("?"),
        ctx.commit_sha.as_deref().unwrap_or("?"),
        if ctx.git_dirty.unwrap_or(false) {
            " (dirty)"
        } else {
            ""
        }
    );
    println!(
        "  result cache:    {} ({cache_entries} entries)",
        config.result_cache.display()
    );
    println!("  dataset cache:   {}", config.dataset_cache.display());
    println!(
        "  models dir:      {} ({model_count} .hfq)",
        models_dir.display()
    );
}

fn run_fetch(config: &EvalConfig) -> Result<(), String> {
    let mut cfg = config.clone();
    cfg.fetch_datasets = true;
    if cfg.suites.is_empty() {
        cfg.suites = default_suites(EvalTier::Extensive);
    }
    eprintln!(
        "[eval] fetching datasets for suites: {} -> {}",
        cfg.suites
            .iter()
            .map(|s| s.as_str())
            .collect::<Vec<_>>()
            .join(","),
        cfg.dataset_cache.display()
    );
    let datasets = resolve_datasets(&cfg)?;
    for d in &datasets {
        println!(
            "  {:<14} {:<12} {}",
            d.suite.as_str(),
            format!("{:?}", d.status),
            d.reason.as_deref().unwrap_or(&d.source)
        );
    }
    println!(
        "Note: KLD references are GPU-generated, not fetched — make them with `collect_artifacts --kldref` or `perplexity --dump-ref` into ~/.hipfire/datasets/kldref/."
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

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
            "--model",
            "candidate.hfq",
            "--baseline",
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
    fn derives_kldref_names_from_model_artifacts() {
        assert_eq!(
            kldref_name_for_model("/models/qwen3.5-0.8b-bf16.hfq").as_deref(),
            Some("qwen3.5-0.8b-bf16.kldref.hfq")
        );
        assert_eq!(
            kldref_name_for_model("/models/qwen3.5-0.8b-mq4.hfq").as_deref(),
            Some("qwen3.5-0.8b-bf16.kldref.hfq")
        );
        assert_eq!(
            kldref_name_for_model("/models/qwen3.5-35b-a3b-mq4.hfq").as_deref(),
            Some("qwen3.5-35b-a3b-bf16.kldref.hfq")
        );
    }

    #[test]
    fn parses_hfkseq_v2_metrics() {
        let path = temp_path("quality-row.kldseq");
        let mut body = Vec::new();
        body.extend_from_slice(b"HFKSEQ\0\0");
        body.extend_from_slice(&2u32.to_le_bytes());
        body.extend_from_slice(&2u32.to_le_bytes());
        body.extend_from_slice(&0u32.to_le_bytes());
        for (mean, p99, nll) in [(0.1f64, 0.3f64, 2.0f64), (0.2, 0.4, 4.0)] {
            body.extend_from_slice(&mean.to_le_bytes());
            body.extend_from_slice(&p99.to_le_bytes());
            body.extend_from_slice(&nll.to_le_bytes());
        }
        fs::write(&path, body).unwrap();

        let metrics = parse_hfkseq_metrics(&path).unwrap();
        assert_eq!(metrics["n_chunks"], json!(2));
        assert_eq!(metrics["mean_kld"], json!(0.15000000000000002));
        assert_eq!(metrics["p99_kld"], json!(0.4));
        assert_eq!(metrics["mean_nll"], json!(3.0));
        assert_eq!(metrics["ppl"], json!(3.0f64.exp()));

        let _ = fs::remove_file(path);
    }

    #[test]
    fn parses_pp_dpm_mclk_max_clock() {
        let raw = "0: 400Mhz \n1: 800Mhz *\n2: 937Mhz \n";
        assert_eq!(parse_pp_dpm_mclk_max_mhz(raw), Some(937.0));
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
            reason: Some("no --baseline provided".to_string()),
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
            reason: Some("no --baseline provided".to_string()),
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
            reason: Some("no --baseline provided".to_string()),
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
            reason: Some("no --baseline provided".to_string()),
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
            reason: Some("no --baseline provided".to_string()),
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
            reason: Some("no --baseline provided".to_string()),
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
            reason: Some("no --baseline provided".to_string()),
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
            reason: Some("no --baseline provided".to_string()),
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
            reason: Some("no --baseline provided".to_string()),
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
            reason: Some("no --baseline provided".to_string()),
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
            reason: Some("no --baseline provided".to_string()),
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
            reason: Some("no --baseline provided".to_string()),
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
            .contains("no --baseline or --reference"));
    }
}
