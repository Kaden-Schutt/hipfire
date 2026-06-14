// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Evidence provenance helpers shared by Hipfire eval and gate tooling.

use std::collections::BTreeMap;
use std::fs::{self, File};
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::{Mutex, OnceLock};
use std::time::UNIX_EPOCH;

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HfqMetadata {
    pub arch_id: u32,
    pub metadata_json: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EvidenceArtifactSpec {
    pub file: &'static str,
    pub kind: &'static str,
    pub expected_metrics: &'static [&'static str],
}

pub const STANDARD_EVIDENCE_ARTIFACT_SPECS: &[EvidenceArtifactSpec] = &[
    EvidenceArtifactSpec {
        file: "quality.json",
        kind: "quality",
        expected_metrics: &[
            "mean_kld",
            "p99_kld",
            "ppl",
            "argmax_match_rate",
            "accuracy",
            "exact_match",
        ],
    },
    EvidenceArtifactSpec {
        file: "performance.json",
        kind: "performance",
        expected_metrics: &["pp32_ms", "pp128_ms", "ttft_ms", "tok_s"],
    },
    EvidenceArtifactSpec {
        file: "phase_timings.json",
        kind: "phase_timings",
        expected_metrics: &["load_ms", "prefill_ms", "decode_ms", "teardown_ms"],
    },
    EvidenceArtifactSpec {
        file: "launch_counts.json",
        kind: "launch_counts",
        expected_metrics: &["kernel_launches", "graph_launches", "memcpy_ops"],
    },
    EvidenceArtifactSpec {
        file: "moe_router_histogram.json",
        kind: "moe_router_histogram",
        expected_metrics: &["expert_hits", "shared_expert_hits", "router_entropy"],
    },
    EvidenceArtifactSpec {
        file: "memory.json",
        kind: "memory",
        expected_metrics: &["vram_peak_bytes", "kv_bytes", "workspace_bytes"],
    },
    EvidenceArtifactSpec {
        file: "dflash_trace.json",
        kind: "dflash_trace",
        expected_metrics: &["ar_tok_s", "dflash_tok_s", "accept_rate", "tau"],
    },
    EvidenceArtifactSpec {
        file: "path_c_trace.json",
        kind: "path_c_trace",
        expected_metrics: &[
            "tok_s",
            "tau",
            "promotion_verdict",
            "tok_s_delta_pct",
            "tau_delta_pct",
        ],
    },
    EvidenceArtifactSpec {
        file: "module_evidence.json",
        kind: "module_evidence",
        expected_metrics: &[
            "module_kind",
            "module_id",
            "preferred_backend",
            "selected_backend",
            "oracle_backend",
            "fallback_reason",
        ],
    },
    EvidenceArtifactSpec {
        file: "profiling.json",
        kind: "profiling",
        expected_metrics: &["kernel_name", "duration_us", "occupancy", "waves"],
    },
    EvidenceArtifactSpec {
        file: "coherence.json",
        kind: "coherence",
        expected_metrics: &[
            "hard_fails",
            "soft_warns",
            "detector_count",
            "coherence_status",
        ],
    },
];

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AdmissionEvidenceRequirement {
    pub kind: &'static str,
    pub batteries: Vec<&'static str>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AdmissionVerdictPolicy {
    pub status: &'static str,
    pub verdict: &'static str,
    pub reason: Option<&'static str>,
}

pub const OBSERVED_ADMISSION_EVIDENCE_KINDS: &[&str] = &[
    "phase_timings",
    "launch_counts",
    "moe_router_histogram",
    "memory",
    "dflash_trace",
    "path_c_trace",
    "module_evidence",
    "coherence",
    "profiling",
];

#[derive(Debug, Clone, PartialEq)]
pub struct EvidenceRecord {
    pub battery: String,
    pub suite: Option<String>,
    pub case_id: String,
    pub dataset_item_id: Option<String>,
    pub dataset_source: Option<String>,
    pub dataset_repo_id: Option<String>,
    pub dataset_revision: Option<String>,
    pub dataset_digest: Option<String>,
    pub dataset_license: Option<String>,
    pub dataset_cache_path: Option<String>,
    pub model: String,
    pub model_hash: Option<String>,
    pub draft: Option<String>,
    pub draft_hash: Option<String>,
    pub baseline: Option<String>,
    pub baseline_hash: Option<String>,
    pub reference: Option<String>,
    pub reference_hash: Option<String>,
    pub prompt_hash: Option<String>,
    pub prompt_path: Option<String>,
    pub metrics: BTreeMap<String, Value>,
    pub elapsed_ms: u128,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RunProvenance {
    pub runner: String,
    pub runner_version: String,
    pub hipfire_version: String,
    pub git_commit: Option<String>,
    pub git_branch: Option<String>,
    pub git_describe: Option<String>,
    pub git_dirty: Option<bool>,
    pub binary_hash: Option<String>,
    pub arch: Option<String>,
    pub rocm: Option<String>,
}

#[derive(Debug, Clone)]
pub struct EvidenceArtifactIndexContext {
    pub provenance: RunProvenance,
    pub host_profile_hash: String,
    pub hardware_bucket: String,
}

#[derive(Debug, Clone, PartialEq)]
pub struct RunMetadataConfig {
    pub tier: String,
    pub tier_budget: Value,
    pub batteries: Vec<String>,
    pub suites: Vec<String>,
    pub executor: String,
    pub kv_mode: Option<String>,
    pub max_tokens: usize,
    pub profile: String,
    pub dflash: String,
    pub runs: usize,
    pub warmup_runs: usize,
    pub benchmark: bool,
    pub host_memory_class: Option<String>,
    pub host_memory_width_bits: Option<u32>,
    pub host_memory_bandwidth_gbps: Option<f64>,
    pub result_cache: String,
    pub cache_mode: String,
}

#[derive(Debug, Clone, PartialEq)]
pub struct RunMetadataModels {
    pub candidate: String,
    pub draft: Option<String>,
    pub baseline: Option<String>,
    pub reference: Option<String>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct RunMetadataArtifact {
    pub created_utc: String,
    pub provenance: RunProvenance,
    pub host_profile: Value,
    pub host_profile_hash: String,
    pub hardware_bucket: String,
    pub config: RunMetadataConfig,
    pub models: RunMetadataModels,
}

pub fn run_provenance_json(provenance: RunProvenance) -> Value {
    serde_json::to_value(provenance).unwrap_or_else(|_| json!({}))
}

pub fn run_metadata_artifact_json(metadata: RunMetadataArtifact) -> Value {
    json!({
        "schema": 1,
        "kind": "run_metadata",
        "status": "collected",
        "runner": metadata.provenance.runner,
        "runner_version": metadata.provenance.runner_version,
        "hipfire_version": metadata.provenance.hipfire_version,
        "created_utc": metadata.created_utc,
        "git": {
            "commit": metadata.provenance.git_commit,
            "branch": metadata.provenance.git_branch,
            "describe": metadata.provenance.git_describe,
            "dirty": metadata.provenance.git_dirty,
        },
        "binary": {
            "hash": metadata.provenance.binary_hash,
        },
        "host": {
            "arch": metadata.provenance.arch,
            "rocm": metadata.provenance.rocm,
            "profile": metadata.host_profile,
            "host_profile_hash": metadata.host_profile_hash,
            "hardware_bucket": metadata.hardware_bucket,
        },
        "config": {
            "tier": metadata.config.tier,
            "tier_budget": metadata.config.tier_budget,
            "batteries": metadata.config.batteries,
            "suites": metadata.config.suites,
            "executor": metadata.config.executor,
            "kv_mode": metadata.config.kv_mode,
            "max_tokens": metadata.config.max_tokens,
            "profile": metadata.config.profile,
            "dflash": metadata.config.dflash,
            "runs": metadata.config.runs,
            "warmup_runs": metadata.config.warmup_runs,
            "benchmark": metadata.config.benchmark,
            "host_memory_class": metadata.config.host_memory_class,
            "host_memory_width_bits": metadata.config.host_memory_width_bits,
            "host_memory_bandwidth_gbps": metadata.config.host_memory_bandwidth_gbps,
            "result_cache": metadata.config.result_cache,
            "cache_mode": metadata.config.cache_mode,
        },
        "models": {
            "candidate": metadata.models.candidate,
            "draft": metadata.models.draft,
            "baseline": metadata.models.baseline,
            "reference": metadata.models.reference,
        },
    })
}

pub fn evidence_artifact_index_entry_json(
    path: impl Into<String>,
    status: impl Into<String>,
    context: &EvidenceArtifactIndexContext,
) -> Value {
    json!({
        "path": path.into(),
        "status": status.into(),
        "runner_version": context.provenance.runner_version,
        "hipfire_version": context.provenance.hipfire_version,
        "git_commit": context.provenance.git_commit,
        "git_branch": context.provenance.git_branch,
        "git_describe": context.provenance.git_describe,
        "git_dirty": context.provenance.git_dirty,
        "binary_hash": context.provenance.binary_hash,
        "arch": context.provenance.arch,
        "rocm": context.provenance.rocm,
        "host_profile_hash": context.host_profile_hash,
        "hardware_bucket": context.hardware_bucket,
    })
}

pub fn evidence_artifact_index_entry_from_value_json(
    path: impl Into<String>,
    status: impl Into<String>,
    value: &Value,
    context: &EvidenceArtifactIndexContext,
) -> Value {
    let mut entry = evidence_artifact_index_entry_json(path, status, context);
    if let Some(object) = entry.as_object_mut() {
        if let Some(records) = value.get("records").and_then(Value::as_array) {
            object.insert("row_count".to_string(), json!(records.len()));
        }
        if let Some(reason) = value.get("reason").cloned() {
            object.insert("reason".to_string(), reason);
        }
        if let Some(metrics) = value.get("expected_metrics").cloned() {
            object.insert("expected_metrics".to_string(), metrics);
        }
        if let Some(kind) = value.get("kind").cloned() {
            object.insert("kind".to_string(), kind);
        }
    }
    entry
}

pub fn evidence_record_json(record: EvidenceRecord) -> Value {
    json!({
        "battery": record.battery,
        "suite": record.suite,
        "case_id": record.case_id,
        "dataset_item_id": record.dataset_item_id,
        "dataset_source": record.dataset_source,
        "dataset_repo_id": record.dataset_repo_id,
        "dataset_revision": record.dataset_revision,
        "dataset_digest": record.dataset_digest,
        "dataset_license": record.dataset_license,
        "dataset_cache_path": record.dataset_cache_path,
        "model": record.model,
        "model_hash": record.model_hash,
        "draft": record.draft,
        "draft_hash": record.draft_hash,
        "baseline": record.baseline,
        "baseline_hash": record.baseline_hash,
        "reference": record.reference,
        "reference_hash": record.reference_hash,
        "prompt_hash": record.prompt_hash,
        "prompt_path": record.prompt_path,
        "metrics": record.metrics,
        "elapsed_ms": record.elapsed_ms,
    })
}

pub fn evidence_metric_direction(metric: &str, delta: f64) -> String {
    let lower_is_better = [
        "mean_kld",
        "p99_kld",
        "ppl",
        "nll",
        "ttft_ms",
        "decode_ms",
        "prefill_ms",
        "elapsed_ms",
    ]
    .iter()
    .any(|prefix| metric == *prefix || metric.ends_with(prefix));
    let higher_is_better = [
        "tok_s",
        "tokens_per_second",
        "accept_rate",
        "tau",
        "accuracy",
        "exact_match",
    ]
    .iter()
    .any(|prefix| metric == *prefix || metric.ends_with(prefix));

    if lower_is_better {
        if delta < 0.0 {
            "improved".to_string()
        } else if delta > 0.0 {
            "regressed".to_string()
        } else {
            "unchanged".to_string()
        }
    } else if higher_is_better {
        if delta > 0.0 {
            "improved".to_string()
        } else if delta < 0.0 {
            "regressed".to_string()
        } else {
            "unchanged".to_string()
        }
    } else if delta == 0.0 {
        "unchanged".to_string()
    } else {
        "changed".to_string()
    }
}

pub fn admission_metric_is_quality(battery: &str, metric: &str) -> bool {
    matches!(battery, "quality" | "barrage")
        || matches!(
            metric,
            "mean_kld" | "p99_kld" | "ppl" | "nll" | "accuracy" | "exact_match"
        )
}

pub fn required_admission_evidence_requirements(
    selected_batteries: impl IntoIterator<Item = impl AsRef<str>>,
) -> Vec<AdmissionEvidenceRequirement> {
    let mut required = vec![
        AdmissionEvidenceRequirement {
            kind: "quality",
            batteries: vec!["quality"],
        },
        AdmissionEvidenceRequirement {
            kind: "performance",
            batteries: vec!["speed", "dflash", "pflash"],
        },
    ];
    if selected_batteries
        .into_iter()
        .any(|battery| battery.as_ref() == "barrage")
    {
        required.push(AdmissionEvidenceRequirement {
            kind: "barrage",
            batteries: vec!["barrage"],
        });
    }
    required
}

pub fn admission_verdict_policy(has_reject: bool, has_review: bool) -> AdmissionVerdictPolicy {
    if has_reject {
        AdmissionVerdictPolicy {
            status: "fail",
            verdict: "reject",
            reason: Some("quality or correctness regression detected"),
        }
    } else if has_review {
        AdmissionVerdictPolicy {
            status: "pass",
            verdict: "review",
            reason: Some("performance regression detected; quality evidence did not reject"),
        }
    } else {
        AdmissionVerdictPolicy {
            status: "pass",
            verdict: "promote",
            reason: None,
        }
    }
}

pub fn extract_external_evidence_records_json(
    kind: &str,
    path: &Path,
    value: &Value,
    context: Value,
) -> Vec<Value> {
    let Some(selected) = select_external_evidence_value(kind, path, value) else {
        return Vec::new();
    };
    let records = if let Some(records) = selected.get("records").and_then(Value::as_array) {
        records.clone()
    } else if let Some(records) = selected.as_array() {
        records.clone()
    } else {
        vec![selected.clone()]
    };
    records
        .into_iter()
        .map(|record| annotate_external_evidence_record_json(kind, path, record, &context))
        .collect()
}

pub fn standard_evidence_artifact_kind_for_path(path: &Path) -> Option<&'static str> {
    let file_name = path.file_name()?.to_str()?;
    STANDARD_EVIDENCE_ARTIFACT_SPECS
        .iter()
        .find(|spec| spec.file == file_name)
        .map(|spec| spec.kind)
}

pub fn standard_evidence_paths_in_dir(dir: &Path) -> Result<Vec<PathBuf>, String> {
    let mut out = Vec::new();
    let entries =
        fs::read_dir(dir).map_err(|err| format!("read evidence dir {}: {err}", dir.display()))?;
    for entry in entries {
        let entry =
            entry.map_err(|err| format!("read evidence dir entry {}: {err}", dir.display()))?;
        let path = entry.path();
        if path.is_file() && standard_evidence_artifact_kind_for_path(&path).is_some() {
            out.push(path);
        }
    }
    out.sort();
    Ok(out)
}

pub fn file_hash(path: &Path) -> Option<String> {
    command_digest("sha256sum", path).or_else(|| Some(stable_hash_file_fallback(path)))
}

pub fn model_hash(model: &str) -> Option<String> {
    static CACHE: OnceLock<Mutex<BTreeMap<String, Option<String>>>> = OnceLock::new();
    let key = model_hash_cache_key(model);
    let cache = CACHE.get_or_init(|| Mutex::new(BTreeMap::new()));
    if let Ok(cache) = cache.lock() {
        if let Some(hash) = cache.get(&key) {
            return hash.clone();
        }
    }
    let hash = model_hash_uncached(model);
    if let Ok(mut cache) = cache.lock() {
        cache.insert(key, hash.clone());
    }
    hash
}

pub fn stable_hash_file_fallback(path: &Path) -> String {
    let mut f = match File::open(path) {
        Ok(f) => f,
        Err(_) => return "unavailable".to_string(),
    };
    let mut state = Fnv64::new();
    let mut buf = [0u8; 64 * 1024];
    loop {
        match f.read(&mut buf) {
            Ok(0) => break,
            Ok(n) => state.update(&buf[..n]),
            Err(_) => return "unavailable".to_string(),
        }
    }
    format!("fnv64:{:016x}", state.finish())
}

pub fn stable_hash_bytes(bytes: &[u8]) -> String {
    let mut state = Fnv64::new();
    state.update(bytes);
    format!("fnv64:{:016x}", state.finish())
}

pub fn stable_score(input: &str) -> f64 {
    let mut state = Fnv64::new();
    state.update(input.as_bytes());
    (state.finish() as f64) / (u64::MAX as f64)
}

pub fn directory_hash(path: &Path) -> Option<String> {
    let files = list_files(path);
    if files.is_empty() {
        return None;
    }
    Some(stable_hash_bytes(files.join("\n").as_bytes()))
}

pub fn read_hfq_metadata(path: &Path) -> Result<HfqMetadata, String> {
    let mut f = File::open(path).map_err(|e| format!("open model: {e}"))?;
    let mut header = [0u8; 32];
    f.read_exact(&mut header)
        .map_err(|e| format!("read HFQ header: {e}"))?;
    if &header[0..4] != b"HFQM" {
        return Err("not an HFQ container".to_string());
    }
    let arch_id = u32::from_le_bytes(header[8..12].try_into().unwrap());
    let metadata_offset = u64::from_le_bytes(header[16..24].try_into().unwrap()) as usize;
    let data_offset = u64::from_le_bytes(header[24..32].try_into().unwrap()) as usize;
    let span_len = data_offset.saturating_sub(metadata_offset);
    if span_len == 0 || span_len > 256 * 1024 * 1024 {
        return Err(format!(
            "invalid or too-large metadata span: {metadata_offset}..{data_offset}"
        ));
    }
    f.seek(SeekFrom::Start(metadata_offset as u64))
        .map_err(|e| format!("seek HFQ metadata span: {e}"))?;
    let mut span = vec![0u8; span_len];
    f.read_exact(&mut span)
        .map_err(|e| format!("read HFQ metadata span: {e}"))?;
    let json_end = find_json_object_end(&span)
        .ok_or_else(|| "HFQ metadata JSON object was not terminated".to_string())?;
    let metadata_json = String::from_utf8(span[..json_end].to_vec())
        .map_err(|e| format!("HFQ metadata is not UTF-8: {e}"))?;
    Ok(HfqMetadata {
        arch_id,
        metadata_json,
    })
}

fn model_hash_uncached(model: &str) -> Option<String> {
    let p = Path::new(model);
    if p.exists() {
        file_hash(p)
    } else {
        Some(format!("tag:{}", stable_hash_bytes(model.as_bytes())))
    }
}

fn model_hash_cache_key(model: &str) -> String {
    let p = Path::new(model);
    if let Ok(meta) = fs::metadata(p) {
        let modified = meta
            .modified()
            .ok()
            .and_then(|time| time.duration_since(UNIX_EPOCH).ok())
            .map(|duration| duration.as_nanos())
            .unwrap_or(0);
        let canonical = p
            .canonicalize()
            .unwrap_or_else(|_| p.to_path_buf())
            .display()
            .to_string();
        format!("file:{canonical}:{}:{modified}", meta.len())
    } else {
        format!("tag:{model}")
    }
}

fn select_external_evidence_value<'a>(
    kind: &str,
    path: &Path,
    value: &'a Value,
) -> Option<&'a Value> {
    if value
        .get("kind")
        .and_then(Value::as_str)
        .is_some_and(|k| k == kind)
    {
        return Some(value);
    }
    if let Some(mapped) = value.get(kind) {
        return Some(mapped);
    }
    if path
        .file_stem()
        .and_then(|stem| stem.to_str())
        .is_some_and(|stem| stem == kind)
    {
        return Some(value);
    }
    None
}

fn annotate_external_evidence_record_json(
    kind: &str,
    path: &Path,
    record: Value,
    context: &Value,
) -> Value {
    let source_path = path.display().to_string();
    match record {
        Value::Object(mut object) => {
            object
                .entry("kind".to_string())
                .or_insert_with(|| json!(kind));
            object
                .entry("source_path".to_string())
                .or_insert_with(|| json!(source_path));
            object
                .entry("hipfire_eval_context".to_string())
                .or_insert_with(|| context.clone());
            Value::Object(object)
        }
        other => json!({
            "kind": kind,
            "source_path": source_path,
            "hipfire_eval_context": context,
            "value": other,
        }),
    }
}

fn command_digest(tool: &str, path: &Path) -> Option<String> {
    let out = Command::new(tool).arg(path).output().ok()?;
    if !out.status.success() {
        return None;
    }
    String::from_utf8_lossy(&out.stdout)
        .split_whitespace()
        .next()
        .map(str::to_string)
}

fn find_json_object_end(bytes: &[u8]) -> Option<usize> {
    let mut depth = 0i32;
    let mut in_string = false;
    let mut escape = false;
    for (i, &b) in bytes.iter().enumerate() {
        if escape {
            escape = false;
            continue;
        }
        if b == b'\\' && in_string {
            escape = true;
            continue;
        }
        if b == b'"' {
            in_string = !in_string;
            continue;
        }
        if in_string {
            continue;
        }
        if b == b'{' {
            depth += 1;
        } else if b == b'}' {
            depth -= 1;
            if depth == 0 {
                return Some(i + 1);
            }
        }
    }
    None
}

struct Fnv64(u64);

impl Fnv64 {
    fn new() -> Self {
        Self(0xcbf29ce484222325)
    }

    fn update(&mut self, bytes: &[u8]) {
        for &b in bytes {
            self.0 ^= b as u64;
            self.0 = self.0.wrapping_mul(0x100000001b3);
        }
    }

    fn finish(self) -> u64 {
        self.0
    }
}

pub fn list_files(path: &Path) -> Vec<String> {
    let mut out = Vec::new();
    collect_files_relative(path, path, &mut out);
    out.sort();
    out
}

fn collect_files_relative(root: &Path, path: &Path, out: &mut Vec<String>) {
    let Ok(entries) = fs::read_dir(path) else {
        return;
    };
    for entry in entries.flatten() {
        let p = entry.path();
        if p.is_dir() {
            collect_files_relative(root, &p, out);
        } else if p.is_file() {
            if let Ok(rel) = p.strip_prefix(root) {
                out.push(rel.to_string_lossy().replace('\\', "/"));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn stable_hash_bytes_is_deterministic() {
        assert_eq!(stable_hash_bytes(b"abc"), "fnv64:e71fa2190541574b");
        assert_eq!(stable_hash_bytes(b"abc"), stable_hash_bytes(b"abc"));
        assert_ne!(stable_hash_bytes(b"abc"), stable_hash_bytes(b"abcd"));
    }

    #[test]
    fn model_hash_tags_non_files() {
        assert_eq!(
            model_hash("qwen3.5:9b").unwrap(),
            format!("tag:{}", stable_hash_bytes(b"qwen3.5:9b"))
        );
    }

    #[test]
    fn directory_hash_uses_relative_file_list() {
        let root = temp_dir("hipfire-evidence-dir-hash");
        fs::create_dir_all(root.join("nested")).unwrap();
        File::create(root.join("a.txt")).unwrap();
        File::create(root.join("nested/b.txt")).unwrap();
        assert_eq!(
            directory_hash(&root).unwrap(),
            stable_hash_bytes(b"a.txt\nnested/b.txt")
        );
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn standard_artifact_catalog_preserves_schema_names() {
        let specs: Vec<_> = STANDARD_EVIDENCE_ARTIFACT_SPECS
            .iter()
            .map(|spec| (spec.file, spec.kind))
            .collect();
        assert_eq!(
            specs,
            vec![
                ("quality.json", "quality"),
                ("performance.json", "performance"),
                ("phase_timings.json", "phase_timings"),
                ("launch_counts.json", "launch_counts"),
                ("moe_router_histogram.json", "moe_router_histogram"),
                ("memory.json", "memory"),
                ("dflash_trace.json", "dflash_trace"),
                ("path_c_trace.json", "path_c_trace"),
                ("module_evidence.json", "module_evidence"),
                ("profiling.json", "profiling"),
                ("coherence.json", "coherence"),
            ]
        );
        assert!(STANDARD_EVIDENCE_ARTIFACT_SPECS
            .iter()
            .all(|spec| !spec.expected_metrics.is_empty()));
    }

    #[test]
    fn evidence_record_json_preserves_artifact_row_shape() {
        let mut metrics = BTreeMap::new();
        metrics.insert("tok_s".to_string(), json!(123.4));
        let record = EvidenceRecord {
            battery: "speed".to_string(),
            suite: Some("smoke".to_string()),
            case_id: "case-a".to_string(),
            dataset_item_id: Some("item-a".to_string()),
            dataset_source: None,
            dataset_repo_id: None,
            dataset_revision: None,
            dataset_digest: None,
            dataset_license: None,
            dataset_cache_path: None,
            model: "model.hfq".to_string(),
            model_hash: Some("sha256:model".to_string()),
            draft: None,
            draft_hash: None,
            baseline: Some("baseline.hfq".to_string()),
            baseline_hash: Some("sha256:baseline".to_string()),
            reference: None,
            reference_hash: None,
            prompt_hash: Some("fnv64:prompt".to_string()),
            prompt_path: Some("benchmarks/prompts/smoke.txt".to_string()),
            metrics,
            elapsed_ms: 42,
        };

        let json = evidence_record_json(record);
        assert_eq!(json["battery"], "speed");
        assert_eq!(json["suite"], "smoke");
        assert_eq!(json["case_id"], "case-a");
        assert_eq!(json["dataset_item_id"], "item-a");
        assert_eq!(json["model_hash"], "sha256:model");
        assert_eq!(json["baseline_hash"], "sha256:baseline");
        assert_eq!(json["prompt_path"], "benchmarks/prompts/smoke.txt");
        assert_eq!(json["metrics"]["tok_s"], 123.4);
        assert_eq!(json["elapsed_ms"], 42);
    }

    #[test]
    fn evidence_metric_direction_classifies_known_metric_semantics() {
        assert_eq!(evidence_metric_direction("mean_kld", -0.01), "improved");
        assert_eq!(evidence_metric_direction("mean_kld", 0.01), "regressed");
        assert_eq!(evidence_metric_direction("decode_ms", -1.0), "improved");
        assert_eq!(evidence_metric_direction("tok_s", 10.0), "improved");
        assert_eq!(evidence_metric_direction("tok_s", -10.0), "regressed");
        assert_eq!(evidence_metric_direction("accuracy", 0.1), "improved");
        assert_eq!(evidence_metric_direction("exact_match", -0.1), "regressed");
        assert_eq!(
            evidence_metric_direction("phase_prefill_ms", 0.1),
            "regressed"
        );
        assert_eq!(
            evidence_metric_direction("decode_tokens_per_second", 0.1),
            "improved"
        );
        assert_eq!(evidence_metric_direction("custom_metric", 1.0), "changed");
        assert_eq!(evidence_metric_direction("custom_metric", 0.0), "unchanged");
    }

    #[test]
    fn admission_metric_quality_policy_classifies_reject_metrics() {
        assert!(admission_metric_is_quality("quality", "tok_s"));
        assert!(admission_metric_is_quality("barrage", "latency_ms"));
        assert!(admission_metric_is_quality("speed", "mean_kld"));
        assert!(admission_metric_is_quality("dflash", "accuracy"));
        assert!(admission_metric_is_quality("coherence", "exact_match"));
        assert!(!admission_metric_is_quality("speed", "tok_s"));
        assert!(!admission_metric_is_quality("dflash", "accept_rate"));
        assert!(!admission_metric_is_quality("coherence", "hard_fails"));
    }

    #[test]
    fn required_admission_evidence_catalog_preserves_eval_policy() {
        let default = required_admission_evidence_requirements(["quality", "speed"]);
        assert_eq!(
            default,
            vec![
                AdmissionEvidenceRequirement {
                    kind: "quality",
                    batteries: vec!["quality"],
                },
                AdmissionEvidenceRequirement {
                    kind: "performance",
                    batteries: vec!["speed", "dflash", "pflash"],
                },
            ]
        );

        let with_barrage = required_admission_evidence_requirements(["quality", "barrage"]);
        assert_eq!(with_barrage.len(), 3);
        assert_eq!(with_barrage[2].kind, "barrage");
        assert_eq!(with_barrage[2].batteries, vec!["barrage"]);
    }

    #[test]
    fn observed_admission_evidence_catalog_preserves_runtime_telemetry_kinds() {
        assert_eq!(
            OBSERVED_ADMISSION_EVIDENCE_KINDS,
            &[
                "phase_timings",
                "launch_counts",
                "moe_router_histogram",
                "memory",
                "dflash_trace",
                "path_c_trace",
                "module_evidence",
                "coherence",
                "profiling",
            ]
        );
    }

    #[test]
    fn admission_verdict_policy_preserves_eval_outcomes() {
        assert_eq!(
            admission_verdict_policy(true, false),
            AdmissionVerdictPolicy {
                status: "fail",
                verdict: "reject",
                reason: Some("quality or correctness regression detected"),
            }
        );
        assert_eq!(
            admission_verdict_policy(false, true),
            AdmissionVerdictPolicy {
                status: "pass",
                verdict: "review",
                reason: Some("performance regression detected; quality evidence did not reject"),
            }
        );
        assert_eq!(
            admission_verdict_policy(false, false),
            AdmissionVerdictPolicy {
                status: "pass",
                verdict: "promote",
                reason: None,
            }
        );
        assert_eq!(admission_verdict_policy(true, true).verdict, "reject");
    }

    #[test]
    fn run_provenance_json_preserves_artifact_shape() {
        let provenance = RunProvenance {
            runner: "hipfire-eval".to_string(),
            runner_version: "0.2.0".to_string(),
            hipfire_version: "0.2.0".to_string(),
            git_commit: Some("abc123".to_string()),
            git_branch: Some("main".to_string()),
            git_describe: Some("v0.2.0-1-gabc123".to_string()),
            git_dirty: Some(false),
            binary_hash: Some("sha256:binary".to_string()),
            arch: Some("gfx1151".to_string()),
            rocm: Some("6.4".to_string()),
        };

        let json = run_provenance_json(provenance);
        assert_eq!(json["runner"], "hipfire-eval");
        assert_eq!(json["runner_version"], "0.2.0");
        assert_eq!(json["hipfire_version"], "0.2.0");
        assert_eq!(json["git_commit"], "abc123");
        assert_eq!(json["git_dirty"], false);
        assert_eq!(json["binary_hash"], "sha256:binary");
        assert_eq!(json["arch"], "gfx1151");
        assert_eq!(json["rocm"], "6.4");
    }

    #[test]
    fn run_metadata_artifact_json_preserves_runtime_schema() {
        let artifact = run_metadata_artifact_json(RunMetadataArtifact {
            created_utc: "2026-06-14T21:00:00Z".to_string(),
            provenance: RunProvenance {
                runner: "hipfire-eval".to_string(),
                runner_version: "0.2.0".to_string(),
                hipfire_version: "0.2.0".to_string(),
                git_commit: Some("abc123".to_string()),
                git_branch: Some("main".to_string()),
                git_describe: Some("v0.2.0-1-gabc123".to_string()),
                git_dirty: Some(true),
                binary_hash: Some("sha256:binary".to_string()),
                arch: Some("gfx1151".to_string()),
                rocm: Some("6.4".to_string()),
            },
            host_profile: json!({
                "schema": 1,
                "hardware_kind": "gpu",
            }),
            host_profile_hash: "host:abc".to_string(),
            hardware_bucket: "gfx1151:64g".to_string(),
            config: RunMetadataConfig {
                tier: "fast".to_string(),
                tier_budget: json!({
                    "target_max_seconds": 60,
                    "ci_suitable": true,
                }),
                batteries: vec!["smoke".to_string(), "quality".to_string()],
                suites: vec!["canary".to_string()],
                executor: "mock".to_string(),
                kv_mode: Some("q8".to_string()),
                max_tokens: 32,
                profile: "off".to_string(),
                dflash: "off".to_string(),
                runs: 2,
                warmup_runs: 1,
                benchmark: true,
                host_memory_class: Some("gddr6".to_string()),
                host_memory_width_bits: Some(256),
                host_memory_bandwidth_gbps: Some(512.5),
                result_cache: "/tmp/cache".to_string(),
                cache_mode: "use".to_string(),
            },
            models: RunMetadataModels {
                candidate: "candidate.hfq".to_string(),
                draft: Some("draft.hfq".to_string()),
                baseline: Some("baseline.hfq".to_string()),
                reference: None,
            },
        });

        assert_eq!(artifact["schema"], 1);
        assert_eq!(artifact["kind"], "run_metadata");
        assert_eq!(artifact["status"], "collected");
        assert_eq!(artifact["runner"], "hipfire-eval");
        assert_eq!(artifact["runner_version"], "0.2.0");
        assert_eq!(artifact["hipfire_version"], "0.2.0");
        assert_eq!(artifact["created_utc"], "2026-06-14T21:00:00Z");
        assert_eq!(artifact["git"]["commit"], "abc123");
        assert_eq!(artifact["git"]["dirty"], true);
        assert_eq!(artifact["binary"]["hash"], "sha256:binary");
        assert_eq!(artifact["host"]["arch"], "gfx1151");
        assert_eq!(artifact["host"]["rocm"], "6.4");
        assert_eq!(artifact["host"]["profile"]["hardware_kind"], "gpu");
        assert_eq!(artifact["host"]["host_profile_hash"], "host:abc");
        assert_eq!(artifact["host"]["hardware_bucket"], "gfx1151:64g");
        assert_eq!(artifact["config"]["tier"], "fast");
        assert_eq!(artifact["config"]["tier_budget"]["target_max_seconds"], 60);
        assert_eq!(artifact["config"]["batteries"][0], "smoke");
        assert_eq!(artifact["config"]["suites"][0], "canary");
        assert_eq!(artifact["config"]["kv_mode"], "q8");
        assert_eq!(artifact["config"]["max_tokens"], 32);
        assert_eq!(artifact["config"]["benchmark"], true);
        assert_eq!(artifact["config"]["host_memory_width_bits"], 256);
        assert_eq!(artifact["config"]["host_memory_bandwidth_gbps"], 512.5);
        assert_eq!(artifact["config"]["result_cache"], "/tmp/cache");
        assert_eq!(artifact["config"]["cache_mode"], "use");
        assert_eq!(artifact["models"]["candidate"], "candidate.hfq");
        assert_eq!(artifact["models"]["draft"], "draft.hfq");
        assert_eq!(artifact["models"]["baseline"], "baseline.hfq");
        assert_eq!(artifact["models"]["reference"], Value::Null);
    }

    #[test]
    fn artifact_index_entry_json_preserves_shared_metadata() {
        let context = EvidenceArtifactIndexContext {
            provenance: RunProvenance {
                runner: "hipfire-eval".to_string(),
                runner_version: "0.2.0".to_string(),
                hipfire_version: "0.2.0".to_string(),
                git_commit: Some("abc123".to_string()),
                git_branch: Some("main".to_string()),
                git_describe: None,
                git_dirty: Some(true),
                binary_hash: Some("sha256:binary".to_string()),
                arch: Some("gfx1151".to_string()),
                rocm: Some("6.4".to_string()),
            },
            host_profile_hash: "host:abc".to_string(),
            hardware_bucket: "gfx1151:64g".to_string(),
        };
        let artifact = json!({
            "kind": "performance",
            "reason": "ok",
            "expected_metrics": ["tok_s"],
            "records": [
                {"case_id": "a"},
                {"case_id": "b"}
            ]
        });

        let json = evidence_artifact_index_entry_from_value_json(
            "artifacts/performance.json",
            "collected",
            &artifact,
            &context,
        );
        assert_eq!(json["path"], "artifacts/performance.json");
        assert_eq!(json["status"], "collected");
        assert_eq!(json["runner_version"], "0.2.0");
        assert_eq!(json["git_commit"], "abc123");
        assert_eq!(json["git_dirty"], true);
        assert_eq!(json["binary_hash"], "sha256:binary");
        assert_eq!(json["host_profile_hash"], "host:abc");
        assert_eq!(json["hardware_bucket"], "gfx1151:64g");
        assert_eq!(json["row_count"], 2);
        assert_eq!(json["reason"], "ok");
        assert_eq!(json["expected_metrics"][0], "tok_s");
        assert_eq!(json["kind"], "performance");
    }

    #[test]
    fn external_evidence_records_selects_kind_mapping_and_annotates_context() {
        let path = Path::new("/tmp/evidence.json");
        let value = json!({
            "launch_counts": {
                "records": [
                    {"metrics": {"kernel_launches": 3}}
                ]
            }
        });
        let records = extract_external_evidence_records_json(
            "launch_counts",
            path,
            &value,
            json!({"runner": "hipfire-eval"}),
        );

        assert_eq!(records.len(), 1);
        assert_eq!(records[0]["kind"], "launch_counts");
        assert_eq!(records[0]["source_path"], "/tmp/evidence.json");
        assert_eq!(records[0]["hipfire_eval_context"]["runner"], "hipfire-eval");
        assert_eq!(records[0]["metrics"]["kernel_launches"], 3);
    }

    #[test]
    fn external_evidence_records_preserves_existing_annotation_fields() {
        let path = Path::new("/tmp/launch_counts.json");
        let value = json!({
            "kind": "launch_counts",
            "records": [
                {
                    "kind": "custom",
                    "source_path": "/already/set.json",
                    "hipfire_eval_context": {"runner": "custom-runner"},
                    "metrics": {"kernel_launches": 7}
                }
            ]
        });
        let records = extract_external_evidence_records_json(
            "launch_counts",
            path,
            &value,
            json!({"runner": "hipfire-eval"}),
        );

        assert_eq!(records.len(), 1);
        assert_eq!(records[0]["kind"], "custom");
        assert_eq!(records[0]["source_path"], "/already/set.json");
        assert_eq!(
            records[0]["hipfire_eval_context"]["runner"],
            "custom-runner"
        );
        assert_eq!(records[0]["metrics"]["kernel_launches"], 7);
    }

    #[test]
    fn external_evidence_records_uses_path_stem_and_wraps_scalars() {
        let path = Path::new("/tmp/performance.json");
        let records = extract_external_evidence_records_json(
            "performance",
            path,
            &json!(123.4),
            json!({"runner": "hipfire-eval"}),
        );

        assert_eq!(records.len(), 1);
        assert_eq!(records[0]["kind"], "performance");
        assert_eq!(records[0]["source_path"], "/tmp/performance.json");
        assert_eq!(records[0]["hipfire_eval_context"]["runner"], "hipfire-eval");
        assert_eq!(records[0]["value"], 123.4);
    }

    #[test]
    fn standard_evidence_path_discovery_uses_shared_catalog() {
        let root = temp_dir("hipfire-evidence-standard-paths");
        fs::create_dir_all(&root).unwrap();
        fs::write(root.join("quality.json"), "{}").unwrap();
        fs::write(root.join("module_evidence.json"), "{}").unwrap();
        fs::write(root.join("not_evidence.json"), "{}").unwrap();
        fs::create_dir_all(root.join("performance.json")).unwrap();

        assert_eq!(
            standard_evidence_artifact_kind_for_path(&root.join("quality.json")),
            Some("quality")
        );
        assert_eq!(
            standard_evidence_artifact_kind_for_path(&root.join("not_evidence.json")),
            None
        );

        let paths = standard_evidence_paths_in_dir(&root).unwrap();
        let names: Vec<_> = paths
            .iter()
            .map(|path| path.file_name().unwrap().to_string_lossy().to_string())
            .collect();
        assert_eq!(names, vec!["module_evidence.json", "quality.json"]);

        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn read_hfq_metadata_extracts_json_span() {
        let root = temp_dir("hipfire-evidence-hfq");
        fs::create_dir_all(&root).unwrap();
        let path = root.join("test.hfq");
        let metadata = br#"{"quantization_hash":{"kind":"test"}}"#;
        let metadata_offset = 32u64;
        let data_offset = metadata_offset + metadata.len() as u64 + 4;
        let mut header = [0u8; 32];
        header[0..4].copy_from_slice(b"HFQM");
        header[8..12].copy_from_slice(&42u32.to_le_bytes());
        header[16..24].copy_from_slice(&metadata_offset.to_le_bytes());
        header[24..32].copy_from_slice(&data_offset.to_le_bytes());
        let mut f = File::create(&path).unwrap();
        f.write_all(&header).unwrap();
        f.write_all(metadata).unwrap();
        f.write_all(b"xxxx").unwrap();
        drop(f);

        let got = read_hfq_metadata(&path).unwrap();
        assert_eq!(got.arch_id, 42);
        assert_eq!(got.metadata_json, String::from_utf8_lossy(metadata));
        let _ = fs::remove_dir_all(root);
    }

    fn temp_dir(name: &str) -> std::path::PathBuf {
        let unique = format!(
            "{}-{}",
            name,
            std::time::SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        );
        std::env::temp_dir().join(unique)
    }
}
